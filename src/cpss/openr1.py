"""Pinned Open R1 import. No Codeforces page requests or generated solutions."""
from __future__ import annotations

import json
import math
import re
import time
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .core import CorpusError, canonical, identity, problem_id, sha256, utcnow
from .parsers import Parsed
from .schemas import validate

PROBLEM_COLUMNS = ["id", "aliases", "contest_id", "contest_name", "contest_type", "contest_start",
                   "contest_start_year", "index", "time_limit", "memory_limit", "title", "description",
                   "input_format", "output_format", "interaction_format", "note", "examples", "editorial",
                   "rating", "tags", "input_mode", "official_tests_complete", "executable", "testset_size", "generated_tests"]
SUBMISSION_COLUMNS = ["submission_id", "source", "og_source", "contestId", "problem_index", "programmingLanguage",
                      "verdict", "testset", "passedTestCount", "timeConsumedMillis", "memoryConsumedBytes",
                      "creationTimeSeconds", "problem_id"]
POLICY = "open-r1-complete-v1"


def json_url(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def prepare(output: Path, progress=None):
    """Read only requested Parquet columns over HTTP ranges; skip private test shards."""
    import fsspec
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "snapshot.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
    else:
        manifest = {"created_at": utcnow(), "policy": POLICY, "repositories": {}, "files": []}
        for name, folder in (("codeforces", "data"), ("codeforces-submissions", "selected_accepted")):
            repo = "open-r1/" + name
            metadata = json_url("https://huggingface.co/api/datasets/" + repo)
            revision = metadata["sha"]
            listing = json_url(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}?expand=true")
            readme_url = f"https://huggingface.co/datasets/{repo}/raw/{revision}/README.md"
            with urllib.request.urlopen(readme_url, timeout=30) as response:
                readme = response.read()
            readme_file = f"{name}-README.md"
            (output / readme_file).write_bytes(readme)
            manifest["repositories"][repo] = {"revision": revision, "card_license": metadata.get("cardData", {}).get("license"),
                "readme_file": readme_file, "readme_sha256": sha256(readme),
                "license_note": "README and card license declarations retained verbatim; upstream content attribution retained"}
            for item in listing:
                if not item["path"].endswith(".parquet"):
                    continue
                manifest["files"].append({"repository": repo, "revision": revision, "file": item["path"],
                    "size": item["size"], "upstream_lfs_sha256": item.get("lfs", {}).get("oid"),
                    "split": "test" if "/test-" in item["path"] else "train",
                    "projection_file": name + "-" + Path(item["path"]).stem + ".jsonl"})
        write_json(manifest_path, manifest)
    for entry in manifest["files"]:
        target = output / entry["projection_file"]
        if target.exists() and entry.get("projection_sha256") == file_hash(target):
            continue
        columns = SUBMISSION_COLUMNS if entry["repository"].endswith("submissions") else PROBLEM_COLUMNS
        url = f"https://huggingface.co/datasets/{entry['repository']}/resolve/{entry['revision']}/{entry['file']}"
        error = None
        for attempt in range(3):
            try:
                with fsspec.open(url, "rb", block_size=262144, cache_type="readahead") as stream:
                    parquet = pq.ParquetFile(stream)
                    available = set(parquet.schema_arrow.names)
                    missing = set(columns) - available
                    if missing:
                        raise CorpusError("upstream_schema_missing_columns")
                    temp = target.with_suffix(".jsonl.tmp")
                    count = 0
                    with temp.open("wb") as destination:
                        for batch in parquet.iter_batches(batch_size=256, columns=columns):
                            for row in batch.to_pylist():
                                destination.write(canonical(row) + b"\n")
                                count += 1
                temp.replace(target)
                entry.update(projection_sha256=file_hash(target), rows=count,
                             retained_columns=columns, retention="lossless selected-column rows; full Parquet LFS digest is upstream-declared")
                write_json(manifest_path, manifest)
                if progress:
                    progress({"prepared": entry["projection_file"], "rows": count})
                error = None
                break
            except (OSError, ValueError) as exc:
                error = type(exc).__name__
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if error:
            raise CorpusError("openr1_download_" + error)
    return {"snapshot": str(manifest_path.resolve()), "rows": sum(e["rows"] for e in manifest["files"]),
            "files": len(manifest["files"])}


def rows_from_snapshot(directory, submissions=False):
    directory = Path(directory)
    manifest = json.loads((directory / "snapshot.json").read_bytes())
    for entry in manifest["files"]:
        if entry["repository"].endswith("submissions") != submissions:
            continue
        path = directory / entry["projection_file"]
        if file_hash(path) != entry["projection_sha256"]:
            raise CorpusError("snapshot_integrity_failure")
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream):
                yield json.loads(line), {**entry, "row_number": number}


def cf_id(value):
    match = re.fullmatch(r"([1-9][0-9]*)/([A-Z][0-9]*)", str(value or ""))
    if not match:
        raise CorpusError("invalid_openr1_problem_id")
    return problem_id(int(match[1]), match[2])


def code_text(row):
    # og_source preserves the original contestant code when upstream transformed source.
    return row.get("og_source") or row.get("source") or ""


def accepted_solutions(directory, maximum=3):
    result = defaultdict(list)
    fingerprints = defaultdict(set)
    for row, entry in rows_from_snapshot(directory, submissions=True):
        text = code_text(row)
        if (row.get("verdict") != "OK" or not isinstance(text, str) or not text.strip()
                or not str(row.get("submission_id", "")).isdigit() or not row.get("programmingLanguage")):
            continue
        try:
            pid = cf_id(row["problem_id"])
            if pid != problem_id(int(row["contestId"]), row["problem_index"]):
                continue
        except (CorpusError, ValueError, KeyError):
            continue
        digest = sha256(text.encode("utf-8"))
        if digest not in fingerprints[pid] and len(result[pid]) < maximum:
            result[pid].append((row, entry))
            fingerprints[pid].add(digest)
    return result


def completeness(row):
    reasons = []
    for field in ("title", "contest_name", "description", "input_format", "output_format", "editorial"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            reasons.append("missing_" + field)
    try:
        pid = cf_id(row.get("id"))
        if pid != problem_id(int(row["contest_id"]), row["index"]):
            reasons.append("identity_mismatch")
    except (CorpusError, ValueError, KeyError, TypeError):
        reasons.append("invalid_identity")
    # CF/ICPC/IOI are scoring systems, not ordinary-vs-Gym categories.
    # Educational rounds legitimately use ICPC scoring.
    if row.get("contest_type") not in {"CF", "ICPC", "IOI"} or not re.search(r"(?:Codeforces|Educational).*\bRound\b", row.get("contest_name", ""), re.I):
        reasons.append("outside_regular_contest_scope")
    if re.search(r"April Fools|Kotlin|Marathon|Teams?|Experimental", row.get("contest_name", ""), re.I):
        reasons.append("special_contest")
    if row.get("interaction_format") or "interactive" in (row.get("tags") or []):
        reasons.append("interactive_problem")
    for field in ("time_limit", "memory_limit"):
        value = row.get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            reasons.append("missing_" + field)
    stamp = row.get("contest_start")
    if type(stamp) is not int or not 0 < stamp <= utcnow().timestamp():
        reasons.append("invalid_contest_date")
    if type(row.get("rating")) is not int or row["rating"] <= 0:
        reasons.append("missing_rating")
    if not isinstance(row.get("tags"), list) or not row["tags"] or any(not isinstance(t, str) or not t.strip() for t in row["tags"]):
        reasons.append("missing_tags")
    examples = row.get("examples")
    if (not isinstance(examples, list) or not examples or any(not isinstance(e, dict)
        or not isinstance(e.get("input"), str) or not e["input"].strip()
        or not isinstance(e.get("output"), str) or not e["output"].strip() for e in examples)):
        reasons.append("missing_or_incomplete_examples")
    if row.get("input_mode") not in {"stdio", "file"}:
        reasons.append("missing_input_mode")
    combined = "\n".join(row.get(k) or "" for k in ("description", "editorial"))
    if re.search(r"tutorial is loading|editorial (?:is )?coming soon|cf-chl-|challenge-platform", combined, re.I):
        reasons.append("placeholder_or_challenge")
    if re.search(r"<img\b[^>]*>|!\[[^\]]*\]\([^)]*\)", combined, re.I):
        reasons.append("unresolved_image")
    return reasons


def statement_parsed(row):
    sections = [{"key": "title", "text": row["title"]},
                {"key": "limits", "text": f"Time limit: {row['time_limit']} seconds\nMemory limit: {row['memory_limit']} MB"},
                {"key": "description", "text": row["description"]},
                {"key": "input", "text": row["input_format"]}, {"key": "output", "text": row["output_format"]}]
    for index, example in enumerate(row["examples"]):
        sections.extend([{"key": f"example:{index}:input", "text": example["input"]},
                         {"key": f"example:{index}:output", "text": example["output"]}])
    if row.get("note"):
        sections.append({"key": "note", "text": row["note"]})
    return Parsed(sections)


def solutions_for(row, index, maximum=3):
    found, hashes = [], set()
    for alias in [row["id"], *(row.get("aliases") or [])]:
        try:
            pid = cf_id(alias)
        except CorpusError:
            continue
        for solution, entry in index.get(pid, []):
            digest = sha256(code_text(solution).encode("utf-8"))
            if digest not in hashes and len(found) < maximum:
                found.append((solution, entry))
                hashes.add(digest)
    return found


def dataset_ref(entry, row_id):
    return {"repository": entry["repository"], "revision": entry["revision"], "split": entry["split"],
            "file": entry["file"], "row_id": row_id}


def row_provenance(entry, row_id):
    return {"url": f"https://huggingface.co/datasets/{entry['repository']}/blob/{entry['revision']}/{entry['file']}#row={entry['row_number']}",
            "retrieved_at": utcnow(), "method": "openr1_pinned_parquet_projection",
            "status_code": 200, "attribution": entry["repository"] + "; original Codeforces authors/contestants"}


def import_problem(repo, row, entry, solutions):
    reasons = completeness(row)
    if reasons or not solutions:
        raise CorpusError("incomplete_openr1_problem")
    pid = cf_id(row["id"])
    metadata = {"contestId": int(row["contest_id"]), "index": row["index"], "name": row["title"],
                "tags": row["tags"], "rating": row["rating"]}
    contest = {"name": row["contest_name"], "started_at": datetime.fromtimestamp(row["contest_start"], timezone.utc)}
    repo.metadata(metadata, contest)
    raw = canonical(row)
    provenance = row_provenance(entry, row["id"])
    reference = dataset_ref(entry, row["id"])
    config = {"adapter": POLICY, "dataset": reference, "snapshot_projection_sha256": entry["projection_sha256"]}
    sid = f"open-r1:statement:{pid}:en"
    repo.ensure_source(sid, "statement", provenance["url"], pid, dataset=reference, collection_method="openr1_dataset")
    repo.ingest(sid, repo.observe(sid), raw, statement_parsed(row), provenance, config=config)
    eid = f"open-r1:editorial:{pid}:en"
    repo.ensure_source(eid, "editorial", provenance["url"], pid, dataset=reference, collection_method="openr1_dataset")
    editorial = Parsed([{"key": pid, "text": row["editorial"]}])
    evid, _ = repo.ingest(eid, repo.observe(eid), raw, editorial, provenance, config=config)
    repo.link_editorial(pid, evid, pid, ("dataset_id", 1.0, "approved"))
    submission_ids = []
    for solution, code_entry in solutions:
        submission_id = str(solution["submission_id"])
        code_id = f"open-r1:solution:{pid}:{submission_id}:en"
        code_reference = dataset_ref(code_entry, submission_id)
        code_provenance = row_provenance(code_entry, submission_id)
        repo.ensure_source(code_id, "solution", code_provenance["url"], pid,
            parent_version_id=evid, programming_language=solution["programmingLanguage"],
            submission_id=int(submission_id), submission_url=f"https://codeforces.com/contest/{solution['contestId']}/submission/{submission_id}",
            collection_method="openr1_selected_accepted", dataset=code_reference,
            attribution="Codeforces contestant; author handle unavailable in upstream row")
        from .parsers import solution as parse_solution
        repo.ingest(code_id, repo.observe(code_id), canonical(solution), parse_solution(code_text(solution).encode("utf-8")),
                    code_provenance, config={"adapter": POLICY, "dataset": code_reference,
                    "code_field": "og_source" if solution.get("og_source") else "source",
                    "snapshot_projection_sha256": code_entry["projection_sha256"]})
        submission_ids.append(submission_id)
    corpus = {"provider": "open-r1", "repository": entry["repository"], "revision": entry["revision"],
              "source_split": entry["split"], "row_id": row["id"], "aliases": row.get("aliases") or [],
              "contest_type": row["contest_type"],
              "time_limit_seconds": row["time_limit"], "memory_limit_mb": row["memory_limit"],
              "input_mode": row["input_mode"], "examples": row["examples"], "complete": True,
              "accepted_solution_ids": submission_ids}
    for input_key, output_key in (("official_tests_complete", "official_tests_complete"),
                                   ("executable", "upstream_executable"), ("testset_size", "testset_size")):
        if row.get(input_key) is not None:
            corpus[output_key] = row[input_key]
    problem = repo.db.problems.find_one({"_id": pid})
    group = identity(sorted({row["id"], *(row.get("aliases") or [])}))
    validate("problems", {**problem, "corpus": corpus, "duplicate_group": "open-r1:" + group})
    repo.db.problems.update_one({"_id": pid}, {"$set": {"corpus": corpus, "duplicate_group": "open-r1:" + group}})
    return pid


def import_snapshot(repo, directory, limit=None, progress=None):
    """Admit complete bundles only, in a dedicated staging database."""
    directory = Path(directory)
    manifest = json.loads((directory / "snapshot.json").read_bytes())
    config = {"provider": "open-r1", "policy": POLICY, "repositories": manifest["repositories"],
              "maximum_solutions_per_problem": 3, "scope": "finished_regular_rounds_noninteractive", "limit": limit}
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity(config)))
    now = utcnow()
    run = {"_id": run_id, "schema_version": 1, "started_at": now, "status": "running",
           "config_fingerprint": identity(config), "config": config, "counts": {}, "export_refs": []}
    validate("ingestion_runs", run)
    repo.db.ingestion_runs.update_one({"_id": run_id}, {"$setOnInsert": run}, upsert=True)
    solutions_index = accepted_solutions(directory)
    admitted, rejected, seen = [], [], set()
    for row, entry in rows_from_snapshot(directory):
        reasons = completeness(row)
        if not reasons and row["id"] in seen:
            reasons.append("duplicate_dataset_identity")
        solutions = solutions_for(row, solutions_index) if not reasons else []
        if not reasons and not solutions:
            reasons.append("no_accepted_solution_source")
        if reasons:
            rejected.append({"row_id": row.get("id"), "split": entry["split"], "reasons": reasons})
            continue
        seen.add(row["id"])
        if limit is not None and len(admitted) >= limit:
            continue
        pid = cf_id(row["id"])
        jid = identity({"run_id": run_id, "stage": "statement", "entity_id": pid})
        job = {"_id": jid, "schema_version": 1, "run_id": run_id, "stage": "statement", "entity_id": pid,
               "status": "pending", "attempts": 0, "next_attempt_at": utcnow(), "checkpoint": {}, "updated_at": utcnow()}
        validate("ingestion_jobs", job)
        repo.db.ingestion_jobs.update_one({"_id": jid}, {"$setOnInsert": job}, upsert=True)
        existing = repo.db.problems.find_one({"_id": pid, "corpus.complete": True})
        if not existing:
            repo.db.ingestion_jobs.update_one({"_id": jid}, {"$set": {"status": "running", "updated_at": utcnow()}, "$inc": {"attempts": 1}})
            import_problem(repo, row, entry, solutions)
        repo.db.ingestion_jobs.update_one({"_id": jid}, {"$set": {"status": "succeeded", "checkpoint.completed": True, "updated_at": utcnow()}})
        admitted.append(pid)
        if progress and len(admitted) % 100 == 0:
            progress({"complete_problems": len(admitted), "excluded_rows_so_far": len(rejected)})
    counts = {"complete_problems": len(admitted), "excluded_rows": len(rejected),
              "accepted_solution_sources": repo.db.source_documents.count_documents({"kind": "solution"}),
              "exclusions": dict(Counter(reason for r in rejected for reason in r["reasons"]))}
    repo.db.ingestion_runs.update_one({"_id": run_id}, {"$set": {"status": "completed", "completed_at": utcnow(), "counts": counts}})
    report = {"run_id": run_id, "database": repo.db.name, "counts": counts, "policy": config,
              "admitted_problem_ids": admitted, "excluded_rows": rejected,
              "verification": "Structural completeness and upstream accepted verdict; code not executed locally; no guarantee of full private judge tests"}
    report_path = directory / ("import-report-" + repo.db.name + ".json")
    write_json(report_path, report)
    return {"run_id": run_id, "database": repo.db.name, "counts": counts, "report": str(report_path.resolve())}


def audit_complete(repo):
    issues = []
    count = 0
    for problem in repo.db.problems.find():
        count += 1
        pid = problem["_id"]
        try:
            validate("problems", problem)
            if not problem.get("corpus", {}).get("complete"):
                raise CorpusError("not_a_complete_openr1_bundle")
            if any(problem["quality"][kind] != "ready" for kind in ("statement", "editorial", "solutions")):
                raise CorpusError("component_not_ready")
            statement = repo.db.document_versions.find_one({"_id": problem["statement"]["version_id"]})
            row = json.loads(repo.get_blob(statement["raw_sha256"]))
            if completeness(row) or cf_id(row["id"]) != pid:
                raise CorpusError("raw_problem_incomplete")
            if statement_parsed(row).clean_text != statement["clean_text"]:
                raise CorpusError("statement_projection_changed")
            if problem["corpus"]["examples"] != row["examples"]:
                raise CorpusError("sample_projection_mismatch")
            if any(problem[k] != row[field] for k, field in (("title", "title"), ("rating", "rating"), ("original_tags", "tags"))):
                raise CorpusError("metadata_projection_mismatch")
            link = repo.db.editorial_links.find_one({"_id": problem["primary_editorial_link_id"], "problem_id": pid, "review_status": "approved"})
            if not link or link["section_text"] != row["editorial"]:
                raise CorpusError("editorial_projection_mismatch")
            sources = list(repo.db.source_documents.find({"problem_id": pid, "kind": "solution", "parent_version_id": link["editorial_version_id"]}))
            if {str(s["submission_id"]) for s in sources} != set(problem["corpus"]["accepted_solution_ids"]):
                raise CorpusError("missing_accepted_solution")
            for source in sources:
                version = repo.db.document_versions.find_one({"_id": source["current_version_id"]})
                submission = json.loads(repo.get_blob(version["raw_sha256"]))
                if submission["verdict"] != "OK" or not code_text(submission).strip():
                    raise CorpusError("solution_not_accepted")
                if code_text(submission).replace("\r\n", "\n") != version["clean_text"]:
                    raise CorpusError("solution_projection_mismatch")
                if submission["problem_id"] not in {row["id"], *(row.get("aliases") or [])}:
                    raise CorpusError("solution_identity_mismatch")
        except Exception as exc:
            from .core import safe_error
            issues.append({"problem_id": pid, "code": safe_error(exc)})
    return {"database": repo.db.name, "complete_problems_checked": count, "ok": count > 0 and not issues, "issues": issues}


def activate(repo, staging_database, backup_database=None):
    """Archive old collections, then promote a fully audited staging database.

    Per-collection renames are recoverable, not a multi-collection transaction.
    The maintenance lock excludes all CLI readers/writers during the cutover.
    """
    from dataclasses import replace
    from .repository import Repository
    target = repo.db.name
    if not staging_database.startswith("cpss_openr1_stage_") or staging_database == target:
        raise CorpusError("invalid_staging_database")
    backup_database = backup_database or "cpss_backup_" + utcnow().strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:8]
    if not backup_database.startswith("cpss_backup_") or backup_database in {target, staging_database}:
        raise CorpusError("invalid_backup_database")
    plan_path = Path("data") / f"activation-{staging_database}.json"
    plan_path.parent.mkdir(exist_ok=True)
    if plan_path.exists():
        plan = json.loads(plan_path.read_bytes())
        if plan["target"] != target:
            raise CorpusError("activation_target_mismatch")
        backup_database = plan["backup"]
    else:
        if repo.db.annotations.count_documents({}):
            raise CorpusError("existing_review_decisions_require_merge_before_cutover")
        if backup_database in repo.client.list_database_names():
            raise CorpusError("backup_database_already_exists")
        stage = Repository(replace(repo.settings, database=staging_database))
        try:
            audit = audit_complete(stage)
            if not audit["ok"]:
                raise CorpusError("staging_completeness_audit_failed")
            reconciliation = stage.reconcile()
            if reconciliation["issues"]:
                raise CorpusError("staging_reconciliation_failed")
            names = stage.db.list_collection_names()
            if set(names) - set(repo.db.list_collection_names()) - {"fs.files", "fs.chunks"}:
                raise CorpusError("unexpected_staging_collections")
            plan = {"created_at": utcnow(), "target": target, "staging": staging_database, "backup": backup_database,
                    "old_counts": {n: repo.db[n].count_documents({}) for n in repo.db.list_collection_names()},
                    "new_counts": {n: stage.db[n].count_documents({}) for n in names},
                    "archived": [], "promoted": [], "status": "prepared", "audit": audit}
            write_json(plan_path, plan)
        finally:
            stage.close()
    if plan["status"] == "completed":
        return {"database": target, "backup_database": backup_database, "status": "already_completed", "plan": str(plan_path.resolve())}
    # No dropTarget: a conflicting collection stops the operation without deleting data.
    for name, count in plan["old_counts"].items():
        if name in plan["archived"]:
            continue
        if name in repo.db.list_collection_names():
            if repo.db[name].count_documents({}) != count:
                raise CorpusError("active_database_changed_since_plan")
            repo.client.admin.command("renameCollection", f"{target}.{name}", to=f"{backup_database}.{name}", dropTarget=False)
        elif repo.client[backup_database][name].count_documents({}) != count:
            raise CorpusError("backup_recovery_count_mismatch")
        plan["archived"].append(name)
        plan["status"] = "archiving"
        write_json(plan_path, plan)
    for name, count in plan["new_counts"].items():
        if name in plan["promoted"]:
            continue
        if name in repo.client[staging_database].list_collection_names():
            repo.client.admin.command("renameCollection", f"{staging_database}.{name}", to=f"{target}.{name}", dropTarget=False)
        if repo.db[name].count_documents({}) != count:
            raise CorpusError("promotion_count_mismatch")
        plan["promoted"].append(name)
        plan["status"] = "promoting"
        write_json(plan_path, plan)
    plan["status"] = "completed"
    plan["completed_at"] = utcnow()
    write_json(plan_path, plan)
    return {"database": target, "backup_database": backup_database, "status": "completed",
            "complete_problems": plan["new_counts"]["problems"], "previous_metadata_records_archived": plan["old_counts"].get("problems", 0),
            "plan": str(plan_path.resolve())}
