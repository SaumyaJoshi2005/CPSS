from __future__ import annotations

import csv
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common import ModelingError, atomic_json, hash_file, identity, read_json
from .search import SearchEngine

_WRITE_LOCK = threading.Lock()


def load_queries(path: Path) -> list[dict[str, str]]:
    value = read_json(path)
    queries = value.get("queries", [])
    if value.get("version") != "cpss-development-queries-v1" or not 20 <= len(queries) <= 30:
        raise ModelingError("The fixed development benchmark needs 20–30 versioned queries")
    if (len({q["query_id"] for q in queries}) != len(queries)
            or any(q.get("category") not in {"named", "unnamed", "combined"} or not q.get("text", "").strip() for q in queries)):
        raise ModelingError("Malformed development query identities/categories/text")
    return queries


def pool_candidates(engine: SearchEngine, progress=None) -> dict:
    settings = engine.settings
    queries = load_queries(settings.path("benchmark_path"))
    depth = max(10, settings.values["evaluation"]["pool_depth"])
    specification = {"queries_sha256": hash_file(settings.path("benchmark_path")), "query_version": "cpss-development-queries-v1",
        "configuration_sha256": settings.fingerprint, "source_manifest_sha256": engine.corpus.manifest_sha256,
        "indexes": {v: i.key for v, i in engine.indexes.items()}, "depth": depth, "methods": ["bm25", "dense", "hybrid"],
        "tag_boost": False, "classifier_boost": False, "split": "train"}
    directory = settings.path("evaluation_path")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pool.json"
    if path.exists():
        stored = read_json(path)
        if stored["specification"] != specification:
            raise ModelingError("Frozen review pool uses different queries/indexes; select a new evaluation_path")
        return {"pool": str(path), "pairs": len(stored["pairs"]), "queries": len(queries), "reused": True}
    if set(engine.indexes) != {"statement", "editorial"} or any(i.dense is None for i in engine.indexes.values()):
        raise ModelingError("A three-method, two-view review pool requires both dense indexes; BM25 search remains available")
    pairs, rankings = {}, []
    started_at = datetime.now(timezone.utc).isoformat()
    for query in queries:
        for view in ["statement", "editorial"]:
            for method in specification["methods"]:
                result = engine.search(query["text"], method=method, view=view, limit=depth)
                rankings.append({"query_id": query["query_id"], "view": view, "method": method,
                    "problem_ids": [r["problem_id"] for r in result["results"]],
                    "scores": [r["score"] for r in result["results"]], "query_seconds": result["query_seconds"]})
                for hit in result["results"]:
                    key = (query["query_id"], hit["problem_id"])
                    pairs[key] = {"pair_id": identity([specification, *key]), "query_id": key[0], "problem_id": key[1]}
        if progress:
            progress({"pooled_query": query["query_id"], "pairs_so_far": len(pairs)})
    ordered = sorted(pairs.values(), key=lambda p: identity([settings.values["evaluation"]["seed"], p["pair_id"]]))
    report = {"version": "retrieval-review-pool-v1", "specification": specification, "started_at": started_at,
        "queries": queries, "pairs": ordered, "rankings": rankings, "judgments_generated": False,
        "blinding": "Reviewer API and CSV omit originating methods, ranks, scores and context view; grading occurs once per query/problem"}
    atomic_json(path, report)
    with (directory / "review-sheet.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "query_id", "query", "problem_id", "title", "source_url", "grade", "rationale"])
        writer.writeheader()
        by_query = {q["query_id"]: q for q in queries}
        for pair in ordered:
            document = engine.by_id[pair["problem_id"]]
            writer.writerow({**pair, "query": by_query[pair["query_id"]]["text"], "title": document.title,
                             "source_url": document.url, "grade": "", "rationale": ""})
    return {"pool": str(path.resolve()), "pairs": len(ordered), "queries": len(queries), "human_judgments": 0, "reused": False}


def load_pool(engine: SearchEngine) -> tuple[dict, str]:
    path = engine.settings.path("evaluation_path") / "pool.json"
    if not path.exists():
        raise ModelingError("Prepare the blinded candidate pool first: python -m modeling retrieval-pool")
    pool = read_json(path)
    if (pool["specification"]["indexes"] != {v: i.key for v, i in engine.indexes.items()}
            or pool["specification"]["source_manifest_sha256"] != engine.corpus.manifest_sha256
            or pool["specification"]["configuration_sha256"] != engine.settings.fingerprint
            or pool["specification"]["queries_sha256"] != hash_file(engine.settings.path("benchmark_path"))):
        raise ModelingError("Review pool provenance differs from current search indexes/configuration/queries")
    return pool, hash_file(path)


def latest_judgments(engine: SearchEngine, pool_hash: str) -> dict[str, dict]:
    latest = {}
    for path in sorted((engine.settings.path("evaluation_path") / "judgments").glob("*.json")):
        row = read_json(path)
        if row["pool_sha256"] != pool_hash:
            continue
        if row["reviewer_kind"] != "human" or row["human_attested"] is not True or type(row["grade"]) is not int or row["grade"] not in {0, 1, 2}:
            raise ModelingError("Judgment has invalid human-attestation/grade provenance")
        key = row["pair_id"]
        if key not in latest or row["created_at"] > latest[key]["created_at"]:
            latest[key] = row
    return latest


def review_queue(engine: SearchEngine) -> dict:
    pool, pool_hash = load_pool(engine)
    latest = latest_judgments(engine, pool_hash)
    pending = next((p for p in pool["pairs"] if p["pair_id"] not in latest), None)
    result: dict[str, Any] = {"total_pairs": len(pool["pairs"]), "reviewed_pairs": len(latest), "pending": None,
                              "grade_definitions": {"2": "Directly relevant: central query requirements satisfied", "1": "Partially relevant: some requirements satisfied", "0": "Irrelevant"}}
    if pending:
        document = engine.by_id[pending["problem_id"]]
        query = next(q for q in pool["queries"] if q["query_id"] == pending["query_id"])
        result["pending"] = {**pending, "query": query["text"], "title": document.title, "url": document.url,
            "rating": document.rating, "statement": document.statement, "editorial": document.editorial, "source_provenance": document.provenance}
    # No originating method/view/rank/score is returned to the reviewer.
    return result


def record_judgment(engine: SearchEngine, payload: dict) -> dict:
    required = {"pair_id", "grade", "rationale", "reviewer", "human_attested"}
    if set(payload) != required or payload["human_attested"] is not True:
        raise ModelingError("Judgment requires explicit human attestation and the five supported fields")
    if type(payload["grade"]) is not int or payload["grade"] not in {0, 1, 2}:
        raise ModelingError("Judgment grade must be 0, 1 or 2")
    if not isinstance(payload["reviewer"], str) or not 2 <= len(payload["reviewer"].strip()) <= 80:
        raise ModelingError("Provide a reviewer name/identifier (2–80 characters)")
    if not isinstance(payload["rationale"], str) or not 5 <= len(payload["rationale"].strip()) <= 3000:
        raise ModelingError("A 5–3000 character relevance rationale is required")
    with _WRITE_LOCK:
        pool, pool_hash = load_pool(engine)
        pair = next((p for p in pool["pairs"] if p["pair_id"] == payload["pair_id"]), None)
        if pair is None:
            raise ModelingError("Judgment pair is not in the frozen training-only pool")
        row = {**payload, **pair, "reviewer": payload["reviewer"].strip(), "reviewer_kind": "human", "review_method": "local_ui_or_explicit_csv_import_self_attested",
            "created_at": datetime.now(timezone.utc).isoformat(), "pool_sha256": pool_hash, "source_manifest_sha256": engine.corpus.manifest_sha256,
            "original_labels_modified": False, "version": "retrieval-judgment-v1"}
        file = engine.settings.path("evaluation_path") / "judgments" / (uuid.uuid4().hex + ".json")
        atomic_json(file, row)
    return {"saved": True, "pair_id": pair["pair_id"], "grade": payload["grade"], "provenance": "Human reviewer self-attestation; not identity verified"}


def import_sheet(engine: SearchEngine, path: Path, reviewer: str, human_attested: bool) -> dict:
    if not human_attested:
        raise ModelingError("CSV import requires --human-attested after a person has reviewed the evidence")
    if path.name.casefold() == "token.txt":
        raise ModelingError("Credential files cannot be judgment inputs")
    entries = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("grade", "").strip():
                entries.append({"pair_id": row["pair_id"], "grade": int(row["grade"]), "rationale": row["rationale"], "reviewer": reviewer, "human_attested": True})
    for row in entries:
        record_judgment(engine, row)
    return {"imported": len(entries), "reviewer": reviewer, "human_self_attested": True}
