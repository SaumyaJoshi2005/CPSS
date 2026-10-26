from __future__ import annotations

import json
import platform
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(url: str, payload=None, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def smoke(url: str, origin: str, output: Path | None = None) -> dict:
    from .bundle import PROBE_QUERIES
    url = url.rstrip("/")
    status, _ = call(url + "/healthz")
    assert status == 200, "Liveness did not respond"
    assert call(url + "/readyz")[0] == 200, "Runtime is not ready"
    status, info = call(url + "/api/info")
    assert status == 200 and info["features"] == {"review": False, "classifier_boost": False, "tag_boost": True}
    assert info["searchable_count"] == 3716
    timings = {}
    observed = {}
    for view in ["statement", "editorial"]:
        for method in ["bm25", "dense", "hybrid"]:
            values = []
            for _ in range(3):
                for query in PROBE_QUERIES:
                    tick = time.perf_counter()
                    code, result = call(url + "/api/search", {"query": query, "view": view, "method": method}, origin)
                    values.append(time.perf_counter() - tick)
                    assert code == 200 and len(result["results"]) == 10
                    assert len({r["problem_id"] for r in result["results"]}) == 10
                    assert all(r["url"].startswith("https://codeforces.com/problemset/problem/") for r in result["results"])
                    observed[view + ":" + method + ":" + query] = [r["problem_id"] for r in result["results"]]
            timings[view + ":" + method] = {"samples": len(values), "mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values)}
    for view in ["statement", "editorial"]:
        for method in ["bm25", "dense", "hybrid"]:
            code, result = call(url + "/api/search", {"query": PROBE_QUERIES[0], "view": view, "method": method,
                "rating_min": 1200, "rating_max": 1800, "tag_boost": True}, origin)
            assert code == 200 and all(1200 <= r["rating"] <= 1800 for r in result["results"])
    assert call(url + "/api/search", {"query": ""}, origin)[0] == 400
    code, empty = call(url + "/api/search", {"query": "DSU", "rating_min": 5000}, origin)
    assert code == 200 and not empty["results"]
    assert call(url + "/api/search", {"query": "DSU", "rating_min": 2000, "rating_max": 1000}, origin)[0] == 400
    assert call(url + "/api/search", {"query": "DSU", "classifier_boost": True}, origin)[0] == 400
    assert call(url + "/api/search", {"query": "DSU"}, "https://attacker.invalid")[0] == 403
    assert call(url + "/api/judgments", {"grade": 2}, origin)[0] == 403
    assert call(url + "/api/review")[0] == 403
    for path in ["/", "/app.js", "/style.css"]:
        with urllib.request.urlopen(url + path, timeout=10) as response:
            assert response.status == 200 and response.read()
    report = {"passed": True, "platform": platform.system(), "python": platform.python_version(), "url": url,
        "searchable_count": info["searchable_count"], "index_initialization_seconds": info["index_loading_seconds"],
        "warm_http_timings": timings, "observed_rankings": observed,
        "protocol": "Three fixed development queries, three measured repeats per method/view, includes HTTP overhead, encoder already initialized by readiness probes; no relevance claim",
        "rating_filters_and_tag_boost": True, "invalid_inputs": True, "public_origin_validation": True,
        "read_only_public_mode": True, "static_assets": True}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return {k: v for k, v in report.items() if k != "observed_rankings"}
