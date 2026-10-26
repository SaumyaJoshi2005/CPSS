from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from .core import CorpusError, canonical, sha256, utcnow


def assign_splits(problems, duplicate_groups=None, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise CorpusError("invalid_split_fractions")
    parent = {p["_id"]: p["_id"] for p in problems}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        parent[find(b)] = find(a)
    groups = {}
    for problem in problems:
        title = re.sub(r"\s*\((?:easy|hard|simple|medium) version\)\s*", "", problem["title"], flags=re.I).casefold().strip()
        keys = [("contest", problem["contest_id"]), ("text", problem["statement"]["clean_sha256"]),
                ("explicit", problem.get("duplicate_group", problem["_id"]))]
        # A common title such as "Game" is not evidence of duplication across years.
        # Restrict title heuristics to explicit easy/hard variants on the same date.
        if re.search(r"\((?:easy|hard|simple|medium) version\)", problem["title"], re.I):
            keys.append(("title_variant", title, problem["contest"]["started_at"].date()))
        for key in keys:
            if key in groups:
                union(problem["_id"], groups[key])
            else:
                groups[key] = problem["_id"]
    for group in duplicate_groups or []:
        selected = [pid for pid in group if pid in parent]
        for pid in selected[1:]:
            union(selected[0], pid)
    for problem in problems:
        for alias in problem.get("corpus", {}).get("aliases", []):
            match = re.fullmatch(r"([1-9][0-9]*)/([A-Z][0-9]*)", alias)
            alias_id = f"codeforces:{match[1]}:{match[2]}" if match else None
            if alias_id in parent:
                union(problem["_id"], alias_id)
    components = {}
    for problem in problems:
        components.setdefault(find(problem["_id"]), []).append(problem)
    intervals = []
    for component in components.values():
        dates = [p["contest"]["started_at"] for p in component]
        intervals.append([min(dates), max(dates), component])
    intervals.sort(key=lambda x: (x[0], x[1]))
    # Duplicate groups can span dates. Merge overlapping date intervals, so chronology
    # remains strict even when a known variant connects distant contests.
    blocks = []
    for start, end, members in intervals:
        if blocks and start <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], end)
            blocks[-1][2].extend(members)
        else:
            blocks.append([start, end, list(members)])
    result, cursor = {}, 0
    for start, end, members in blocks:
        midpoint = (cursor + len(members) / 2) / max(1, len(problems))
        split = "train" if midpoint < fractions[0] else "validation" if midpoint < sum(fractions[:2]) else "test"
        for p in members:
            result[p["_id"]] = split
        cursor += len(members)
    upstream_tests = [p["contest"]["started_at"] for p in problems if p.get("corpus", {}).get("source_split") == "test"]
    if upstream_tests:
        boundary = min(upstream_tests)
        # Preserve the publisher's held-out problems, including duplicate groups
        # whose intervals cross the first upstream test date.
        for start, end, members in blocks:
            if end >= boundary:
                for p in members:
                    result[p["_id"]] = "test"
    return result


def export_corpus(repo, output: Path, run_id=None, require_editorial=True, duplicate_groups=None,
                  fractions=(0.8, 0.1, 0.1)):
    if output.exists():
        raise CorpusError("export_destination_exists")
    query = {"quality.statement": "ready"}
    if run_id:
        ids = repo.db.ingestion_jobs.distinct("entity_id", {"run_id": run_id, "stage": "statement"})
        query["_id"] = {"$in": ids}
    if require_editorial:
        query["quality.editorial"] = "ready"
    problems = list(repo.db.problems.find(query).sort([("contest.started_at", 1), ("_id", 1)]))
    if not problems:
        raise CorpusError("no_exportable_problems")
    splits = assign_splits(problems, duplicate_groups, fractions)
    statement_records, enriched_records, pins = [], [], []
    version_ids, blob_hashes = set(), set()

    def pin_version(vid):
        version = repo.db.document_versions.find_one({"_id": vid})
        if not version or sha256(version["clean_text"].encode("utf-8")) != version["clean_sha256"]:
            raise CorpusError("export_version_integrity_failure")
        repo.get_blob(version["raw_sha256"])
        version_ids.add(vid)
        blob_hashes.add(version["raw_sha256"])
        return version

    for problem in problems:
        statement = pin_version(problem["statement"]["version_id"])
        if statement["clean_sha256"] != problem["statement"]["clean_sha256"]:
            raise CorpusError("export_stale_statement_projection")
        metadata = {k: problem[k] for k in ("source", "contest_id", "problem_index", "title", "canonical_url", "contest", "metadata_fingerprint")}
        if "rating" in problem:
            metadata["rating"] = problem["rating"]
        if "corpus" in problem:
            metadata["corpus"] = problem["corpus"]
        annotations = []
        for aid in problem["approved_labels"]["annotation_ids"]:
            annotation = repo.db.annotations.find_one({"_id": aid})
            if not annotation or annotation["status"] != "approved":
                raise CorpusError("stale_label_projection_rebuild_required")
            pin_version(annotation["evidence"]["version_id"])
            source_version = repo.db.document_versions.find_one({"_id": annotation["evidence"]["version_id"]})
            source = repo.db.source_documents.find_one({"_id": source_version["source_document_id"]})
            annotations.append({**annotation, "evidence_is_current": source.get("current_version_id") == source_version["_id"]})
        target = {"original_tags": problem["original_tags"], "approved_labels": problem["approved_labels"],
                  "annotation_decisions": annotations}
        base = {"problem_id": problem["_id"], "statement_version_id": statement["_id"],
                "clean_sha256": statement["clean_sha256"], "split": splits[problem["_id"]],
                "metadata": metadata, "target": target}
        statement_records.append({**base, "input": {"statement": statement["clean_text"]}})
        link = None
        if problem.get("primary_editorial_link_id"):
            link = repo.db.editorial_links.find_one({"_id": problem["primary_editorial_link_id"], "review_status": "approved"})
            if not link or link["problem_id"] != problem["_id"]:
                raise CorpusError("invalid_export_editorial_link")
            editorial = pin_version(link["editorial_version_id"])
            section = next((s for s in editorial["sections"] if s["key"] == link["section_key"]), None)
            if not section or section["text"] != link["section_text"] or link["section_key"] != problem["_id"]:
                raise CorpusError("invalid_export_editorial_section")
            solutions = []
            for source in repo.db.source_documents.find({"kind": "solution", "problem_id": problem["_id"],
                                                        "parent_version_id": editorial["_id"]}):
                if not source.get("current_version_id"):
                    continue
                code_version = pin_version(source["current_version_id"])
                solutions.append({"version_id": code_version["_id"], "source_document_id": source["_id"],
                                  "raw_sha256": code_version["raw_sha256"], "clean_sha256": code_version["clean_sha256"],
                                  "code": code_version["clean_text"], "attribution": source.get("attribution", ""),
                                  "programming_language": source.get("programming_language"),
                                  "collection_method": source.get("collection_method"),
                                  "submission_url": source.get("submission_url"), "submission_id": source.get("submission_id")})
            enriched_records.append({**base, "editorial_link_id": link["_id"], "editorial_version_id": editorial["_id"],
                                     "solutions": solutions,
                                     "input": {"statement": statement["clean_text"], "editorial": section["text"]}})
        else:
            solutions = []
        for source in repo.db.source_documents.find({"kind": "solution", "problem_id": problem["_id"],
                "collection_method": "permitted_local_import", "parent_version_id": {"$exists": False}}):
            if source.get("current_version_id"):
                version = pin_version(source["current_version_id"])
                solutions.append({"version_id": version["_id"], "source_document_id": source["_id"],
                                  "raw_sha256": version["raw_sha256"], "clean_sha256": version["clean_sha256"],
                                  "code": version["clean_text"], "attribution": source.get("attribution", ""),
                                  "programming_language": source.get("programming_language"),
                                  "collection_method": source.get("collection_method"),
                                  "submission_url": source.get("submission_url"), "submission_id": source.get("submission_id")})
        pins.append({**base, "editorial_link": link,
                     "solutions": solutions,
                     "statement_source_document_id": statement["source_document_id"],
                     "statement_raw_sha256": statement["raw_sha256"]})
    # No tag/title/editorial keys are permitted in the classifier input object.
    if any(set(r["input"]) != {"statement"} for r in statement_records):
        raise CorpusError("statement_input_leakage")
    building = output.with_name(output.name + ".building-" + uuid.uuid4().hex)
    building.mkdir(parents=True)
    file_metadata = {}
    for filename, records in (("statement-only.jsonl", statement_records), ("editorial-enriched.jsonl", enriched_records)):
        content = b"".join(canonical(row) + b"\n" for row in records)
        (building / filename).write_bytes(content)
        file_metadata[filename] = {"sha256": sha256(content), "count": len(records)}
    # Portable restoration data, including exact bytes, lives alongside the export.
    blobs_dir = building / "blobs"
    blobs_dir.mkdir()
    for digest in sorted(blob_hashes):
        (blobs_dir / digest).write_bytes(repo.get_blob(digest))
    versions = [repo.db.document_versions.find_one({"_id": vid}) for vid in sorted(version_ids)]
    (building / "versions.jsonl").write_bytes(b"".join(canonical(v) + b"\n" for v in versions))
    file_metadata["versions.jsonl"] = {"sha256": sha256((building / "versions.jsonl").read_bytes()), "count": len(versions)}
    config = {"run_id": run_id, "require_editorial": require_editorial, "fractions": list(fractions),
              "duplicate_groups": duplicate_groups or [], "split_policy": "chronological_contests_aliases_and_duplicate_intervals_v2",
              "preserve_upstream_test": True,
              "classifier_input_path": "input.statement", "timestamp_encoding": "ISO-8601 UTC"}
    manifest = {"schema_version": 1, "created_at": utcnow(), "configuration": config,
                "files": file_metadata, "counts": {"problems": len(problems), "editorial_enriched": len(enriched_records),
                "splits": {s: list(splits.values()).count(s) for s in ("train", "validation", "test")}},
                "versions": [{"version_id": v["_id"], "source_document_id": v["source_document_id"],
                              "raw_sha256": v["raw_sha256"], "clean_sha256": v["clean_sha256"],
                              "processor_fingerprint": v["processor_fingerprint"]} for v in versions],
                "blobs": sorted(blob_hashes), "records": pins}
    manifest_bytes = canonical(manifest)
    digest = sha256(manifest_bytes)
    (building / "manifest.json").write_bytes(manifest_bytes)
    (building / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    building.rename(output)
    if run_id:
        repo.db.ingestion_runs.update_one({"_id": run_id}, {"$addToSet": {"export_refs": digest}})
    return {"output": str(output.resolve()), "manifest_sha256": digest, "counts": manifest["counts"]}


def verify_export(path: Path):
    raw = (path / "manifest.json").read_bytes()
    if sha256(raw) != (path / "manifest.sha256").read_text("ascii").strip():
        raise CorpusError("manifest_hash_mismatch")
    manifest = json.loads(raw)
    for filename, expected in manifest["files"].items():
        if Path(filename).name != filename:
            raise CorpusError("invalid_export_path")
        body = (path / filename).read_bytes()
        if sha256(body) != expected["sha256"] or len(body.splitlines()) != expected["count"]:
            raise CorpusError("export_file_mismatch")
    for digest in manifest["blobs"]:
        if not re.fullmatch("[a-f0-9]{64}", digest) or sha256((path / "blobs" / digest).read_bytes()) != digest:
            raise CorpusError("export_blob_mismatch")
    return {"verified": True, "counts": manifest["counts"]}
