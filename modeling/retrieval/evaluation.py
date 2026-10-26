from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from ..common import ModelingError, atomic_json, identity
from ..resources import memory_observation
from .judgments import latest_judgments, load_pool, load_queries
from .search import SearchEngine


def ranking_metrics(ranked: list[str], grades: dict[str, int], k=10, relevance_threshold=2) -> dict[str, float]:
    if k < 1 or relevance_threshold not in {1, 2} or any(type(g) is not int or g not in {0, 1, 2} for g in grades.values()):
        raise ModelingError("Invalid graded relevance/cutoff specification")
    order = list(dict.fromkeys(ranked))[:k]
    if any(pid not in grades for pid in order):
        raise ModelingError("Unjudged retrieved documents cannot be silently scored as irrelevant")
    discount = lambda index: np.log2(index + 2)
    dcg = sum((2 ** grades[pid] - 1) / discount(i) for i, pid in enumerate(order))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / discount(i) for i, grade in enumerate(ideal))
    reciprocal = next((1.0 / (i + 1) for i, pid in enumerate(order) if grades[pid] >= relevance_threshold), 0.0)
    return {"ndcg_at_10": float(dcg / idcg) if idcg else 0.0, "mrr_at_10": float(reciprocal)}


def evaluate_retrieval(engine: SearchEngine) -> dict:
    pool, pool_hash = load_pool(engine)
    latest = latest_judgments(engine, pool_hash)
    pairs = {p["pair_id"]: p for p in pool["pairs"]}
    if not set(latest) <= set(pairs):
        raise ModelingError("Judgments contain identities outside the frozen review pool")
    per_query = {}
    complete = []
    for query in pool["queries"]:
        relevant_pairs = [p for p in pool["pairs"] if p["query_id"] == query["query_id"]]
        if relevant_pairs and all(p["pair_id"] in latest for p in relevant_pairs):
            complete.append(query["query_id"])
            per_query[query["query_id"]] = {p["problem_id"]: latest[p["pair_id"]]["grade"] for p in relevant_pairs}
    methods = {}
    for view in ["statement", "editorial"]:
        for method in ["bm25", "dense", "hybrid"]:
            rows = []
            for ranking in pool["rankings"]:
                if ranking["query_id"] in complete and ranking["view"] == view and ranking["method"] == method:
                    rows.append({"query_id": ranking["query_id"], **ranking_metrics(ranking["problem_ids"], per_query[ranking["query_id"]],
                        engine.settings.values["evaluation"]["cutoff"], engine.settings.values["evaluation"]["relevance_threshold"])})
            methods[view + ":" + method] = {"fully_judged_queries": len(rows), "metrics":
                {key: float(np.mean([r[key] for r in rows])) for key in ["ndcg_at_10", "mrr_at_10"]} if rows else None,
                "per_query": rows}
    report = {"version": "retrieval-evaluation-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_sha256": pool_hash, "judgments_sha256": identity(latest), "source_manifest_sha256": engine.corpus.manifest_sha256,
        "status": "human_self_attested_judgments_available" if complete else "awaiting_complete_human_judgments",
        "total_queries": len(pool["queries"]), "fully_judged_query_ids": complete, "judged_pairs": len(latest), "pool_pairs": len(pairs),
        "methods": methods, "relevance_conventions": {"ndcg_gain": "2^grade - 1", "grades": {"2": "directly relevant", "1": "partially relevant", "0": "irrelevant"},
            "mrr_relevance_threshold": engine.settings.values["evaluation"]["relevance_threshold"], "cutoff": 10,
            "ideal_ranking_scope": "Fully judged candidate pool for each query; not exhaustive corpus judgments",
            "incomplete_queries": "Excluded equally from every comparison; unjudged is not irrelevant",
            "recall": "Not reported: no exhaustive corpus-relevant set is known"},
        "human_review_provenance": "Reviewer self-attestation retained in append-only individual records; identity is not independently verified",
        "classifier_metrics_are_different_task": True, "test_used": False}
    directory = engine.settings.path("evaluation_path")
    file = directory / f"evaluation-{pool_hash[:12]}-{report['judgments_sha256'][:12]}.json"
    atomic_json(file, report)
    atomic_json(directory / "retrieval-metrics.json", report)
    return report


def measure_latency(engine: SearchEngine, progress=None) -> dict:
    queries = load_queries(engine.settings.path("benchmark_path"))
    specification = engine.settings.values["latency"]
    runs = {}
    before = memory_observation()
    for view in ["statement", "editorial"]:
        for method in ["bm25", "dense", "hybrid"]:
            cold_encoder = bool(method != "bm25" and engine.encoder is not None and not hasattr(engine.encoder, "_session"))
            tick = time.perf_counter()
            first = engine.search(queries[0]["text"], method=method, view=view)
            first_seconds = time.perf_counter() - tick
            for _ in range(specification["warmups"]):
                for query in queries:
                    engine.search(query["text"], method=method, view=view)
            samples = []
            for repeat in range(specification["repeats"]):
                for query in queries:
                    tick = time.perf_counter()
                    result = engine.search(query["text"], method=method, view=view)
                    samples.append({"query_id": query["query_id"], "repeat": repeat, "seconds": time.perf_counter() - tick, "internal_query_seconds": result["query_seconds"]})
            values = [s["seconds"] for s in samples]
            key = view + ":" + method
            runs[key] = {"sample_count": len(samples), "batch_mean_seconds": float(np.mean(values)), "median_seconds": float(np.median(values)),
                "p95_seconds": float(np.percentile(values, 95)), "first_query_seconds": first_seconds, "first_query_encoder_was_cold": cold_encoder,
                "result_count_first_query": len(first["results"]), "samples": samples}
            if progress:
                progress({"latency_configuration": key, "mean_seconds": runs[key]["batch_mean_seconds"]})
    report = {"version": "retrieval-latency-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": engine.corpus.manifest_sha256, "configuration_sha256": engine.settings.fingerprint,
        "indexes": {v: i.key for v, i in engine.indexes.items()}, "index_loading_seconds": engine.load_seconds,
        "query_count": len(queries), "protocol": {**specification,
            "order": "statement then editorial; BM25 then dense then hybrid; same 24 queries in file order",
            "includes": "Parsing, query encoding when used, exact ranking, optional-filter mask and original-evidence snippets",
            "excludes": "Index loading, process startup, HTTP/browser overhead and human review",
            "query_embedding_cache": False, "boosts_enabled": False, "candidate_limit": engine.settings.values["candidate_limit"],
            "first_call": "Reported separately; the first dense configuration initializes the shared CPU encoder; later configurations reuse it"},
        "runs": runs, "memory": {"before": before, "after": memory_observation()}, "test_used": False}
    atomic_json(engine.settings.path("evaluation_path") / "latency.json", report)
    return {"index_loading_seconds": report["index_loading_seconds"], "runs": {k: {n: v[n] for n in ["sample_count", "batch_mean_seconds", "median_seconds", "p95_seconds", "first_query_seconds"]} for k, v in runs.items()},
            "report": str(engine.settings.path("evaluation_path") / "latency.json")}
