from __future__ import annotations

import copy
import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modeling.application.server import make_server
from modeling.common import ModelingError, atomic_json, read_json
from modeling.retrieval.configuration import DEFAULT_SEARCH_CONFIG, SearchConfig
from modeling.retrieval.contexts import Document, load_documents, parse_query
from modeling.retrieval.evaluation import evaluate_retrieval, ranking_metrics
from modeling.retrieval.indexing import SearchIndex, build_view, index_key, load_index
from modeling.retrieval.judgments import latest_judgments, load_pool, pool_candidates, record_judgment, review_queue
from modeling.retrieval.lexical import BM25Index
from modeling.retrieval.ranking import boosted_score, ranked_ids, reciprocal_rank_fusion
from modeling.retrieval.search import SearchEngine, evidence


@pytest.fixture
def search_config(tmp_path):
    value = copy.deepcopy(read_json(DEFAULT_SEARCH_CONFIG))
    value["index_path"] = "modeling/index"
    value["evaluation_path"] = "modeling/evaluation"
    value["benchmark_path"] = str(DEFAULT_SEARCH_CONFIG.parent / "retrieval-queries-v1.json")
    path = tmp_path / "search.json"
    atomic_json(path, value)
    return SearchConfig.load(path, tmp_path)


@pytest.fixture
def documents():
    return [Document("codeforces:1:A", "Join components", 1200, ("dsu", "graphs"),
                     "Insert edges and maintain connected groups of vertices.", "A disjoint set union DSU merges components for connectivity queries.",
                     {"split": "train", "statement_version_id": "s1", "editorial_version_id": "e1"}),
            Document("codeforces:2:A", "Feasible limit", 1800, ("binary search", "greedy"),
                     "Find the largest feasible budget threshold for tasks.", "Binary search the monotone answer with a greedy feasibility check.",
                     {"split": "train", "statement_version_id": "s2", "editorial_version_id": "e2"}),
            Document("codeforces:3:A", "Count ways", 900, ("dp",),
                     "Count arrangements using overlapping subproblems.", "Dynamic programming stores each previously computed answer.",
                     {"split": "train", "statement_version_id": "s3", "editorial_version_id": "e3"})]


@pytest.fixture
def engine(documents, search_config, small_config):
    class Encoder:
        def transform(self, texts):
            out = np.zeros((len(texts), 384), dtype=np.float32)
            out[:, 0] = 1
            return out
    matrix = np.zeros((3, 384), dtype=np.float32)
    matrix[:, 0] = [.9, .8, .7]
    matrix[:, 1] = np.sqrt(1 - matrix[:, 0] ** 2)
    indexes = {view: SearchIndex(view, "key-" + view, BM25Index.build([d.text(view) for d in documents], search_config.values["bm25"]),
                               matrix, {"problem_ids": [d.problem_id for d in documents]}) for view in ["statement", "editorial"]}
    corpus = SimpleNamespace(manifest_sha256="a" * 64, split_counts={"train": 3, "validation": 1, "test": 1})
    return SearchEngine(search_config, small_config, corpus, documents, indexes, Encoder())


def test_bm25_matches_hand_computation(search_config):
    config = search_config.values["bm25"]
    index = BM25Index.build(["alpha beta beta", "alpha gamma"], config)
    score = index.scores("beta")
    idf = np.log1p((2 - 1 + .5) / (1 + .5))
    expected = idf * 2 * (1.5 + 1) / (2 + 1.5 * (1 - .75 + .75 * 3 / 2.5))
    np.testing.assert_allclose(score, [expected, 0], rtol=1e-6)
    np.testing.assert_array_equal(index.scores("quarantinezebra"), [0, 0])
    assert index.weights.format == "csr"


def test_rrf_deduplication_ties_and_filter_before_limit():
    scores = reciprocal_rank_fusion([["a", "a", "b"], ["b", "c"]], 60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    ids = ranked_ids(np.array([.9, .8, .7]), ["a", "b", "c"], np.array([False, True, True]), 1)
    assert ids == ["b"]
    assert ranked_ids(np.ones(2), ["b", "a"], np.ones(2, dtype=bool), 2) == ["a", "b"]
    assert boosted_score(-.5, 1, 0, .2, 0) > -.5


def test_query_aliases_explicit_filters_and_unnamed_separation(search_config, taxonomy):
    original = "Disjoint set union with rating >= 1000 and rating <= 2000"
    context = parse_query(original, search_config, taxonomy, rating_min=1200)
    assert context.original_text == original
    assert context.rating_min == 1200 and context.rating_max == 2000
    assert context.explicit_algorithms[0]["algorithm_id"] == "technique:dsu"
    assert "dsu" in context.search_text and "rating" not in context.search_text
    context = parse_query("Maintain connected groups as connections are added", search_config, taxonomy)
    assert context.explicit_algorithms == context.inferred_approaches == ()
    assert context.unknowns and "connected groups" in context.search_text
    fine = parse_query("Dijkstra shortest paths", search_config, taxonomy)
    assert any(a["algorithm_id"] == "algorithm:dijkstra" for a in fine.explicit_algorithms)
    excluded = parse_query("Find a greedy method without dynamic programming", search_config, taxonomy)
    assert excluded.excluded_tags == ("dp",)
    assert "dp" not in [a["original_tag_proxy"] for a in excluded.explicit_algorithms]
    assert "dynamic programming" not in excluded.search_text


@pytest.mark.parametrize("query", ["", "rating >= 2000 rating <= 1200", "DSU rating >= -2", "DSU without Dijkstra"])
def test_invalid_query_constraints(query, search_config, taxonomy):
    with pytest.raises(ModelingError):
        parse_query(query, search_config, taxonomy)


def test_views_preserve_source_and_evidence_offsets(documents, search_config, taxonomy):
    document = documents[0]
    assert document.text("statement") == document.statement
    assert document.editorial not in document.text("statement")
    assert document.editorial in document.text("editorial") and "Original upstream tags: dsu; graphs" in document.text("editorial")
    context = parse_query("disjoint set union connectivity", search_config, taxonomy)
    result = evidence(document, "editorial", context, search_config.values["bm25"])
    source = document.editorial if result["source_kind"] == "editorial" else document.statement
    assert source[result["start"]:result["end"]] == result["quote"]
    assert result["original_tag_overlap"] == ["dsu"]
    assert evidence(document, "statement", context, search_config.values["bm25"])["source_kind"] == "statement"
    with pytest.raises(ModelingError):
        replace(document, provenance={"split": "test"})


def test_search_filters_soft_boost_and_scores_remain_separate(engine, monkeypatch):
    result = engine.search("union find", view="editorial", method="hybrid", rating_max=1500, limit=2, candidates=3)
    assert result["searchable_count"] == 3 and result["complete_corpus_count"] == 5
    assert all(r["rating"] <= 1500 for r in result["results"])
    assert len({r["problem_id"] for r in result["results"]}) == len(result["results"])
    assert result["classifier_prediction"] is None
    assert "not a probability" in result["score_semantics"]
    assert engine.search("connectivity", view="statement", method="dense", rating_min=1700, limit=1)["results"][0]["problem_id"] == "codeforces:2:A"
    assert not engine.search("connectivity", rating_min=4000)["results"]
    assert not engine.search("quarantinezebra", method="bm25")["results"]
    monkeypatch.setattr("modeling.retrieval.search.load_bundle", lambda _: {})
    monkeypatch.setattr("modeling.retrieval.search.predict", lambda *_: {"label_scores": {"dp": .99}, "selected_labels": ["dp"]})
    baseline = engine.search("connectivity", method="dense", candidates=3, limit=3)
    boosted = engine.search("connectivity", method="dense", candidates=3, limit=3, classifier_boost=True)
    assert {r["problem_id"] for r in boosted["results"]} == {r["problem_id"] for r in baseline["results"]}
    assert boosted["classifier_prediction"]["selected_labels"] == ["dp"]
    assert all("classifier_signal" in r and r["original_tags"] == list(engine.by_id[r["problem_id"]].original_tags) for r in boosted["results"])
    with pytest.raises(ModelingError):
        engine.search("query", limit=0)


def test_index_cache_invalidation_integrity_and_lexical_fallback(engine, documents, search_config, small_config):
    corpus = engine.corpus
    first = build_view(documents, corpus, search_config, small_config, "statement", include_dense=False)
    second = build_view(documents, corpus, search_config, small_config, "statement", include_dense=False)
    assert not first["reused"] and second["reused"]
    assert load_index(Path(first["directory"]), first["key"]).dense is None
    changed = [replace(documents[0], statement="Changed original statement"), *documents[1:]]
    assert index_key(changed, "statement", search_config, corpus.manifest_sha256, None) != first["key"]
    assert index_key(documents, "editorial", search_config, corpus.manifest_sha256, None) != first["key"]
    different = replace(search_config, values={**search_config.values, "rrf_k": 61})
    assert index_key(documents, "statement", different, corpus.manifest_sha256, None) != first["key"]
    (Path(first["directory"]) / "vocabulary.json").write_text("{}", encoding="utf8")
    with pytest.raises(ModelingError, match="integrity"):
        load_index(Path(first["directory"]), first["key"])
    engine.indexes["statement"].dense = None
    with pytest.raises(ModelingError, match="unavailable"):
        engine.search("connectivity", method="dense", view="statement")
    assert engine.search("connected", method="bm25", view="statement")["results"]


def test_loader_joins_only_training_sources(tmp_path, records):
    r = records[0][0]
    requests = []
    def selected(splits):
        requests.extend(splits)
        return [r]
    corpus = SimpleNamespace(directory=tmp_path, manifest_sha256="a" * 64,
        structural={r.problem_id: {"editorial_version_id": "editorial-1"}}, records=selected)
    rows = [{"problem_id": "heldout-identity", "input": None, "target": "unread-heldout-content"},
        {"problem_id": r.problem_id, "split": "train", "statement_version_id": r.version_id,
         "editorial_version_id": "editorial-1", "editorial_link_id": "link-1", "input": {"statement": r.statement, "editorial": "Use DSU."},
         "target": {"original_tags": list(r.original_tags)}, "metadata": {"rating": 1200, "corpus": {"repository": "fixture", "revision": "rev", "row_id": "row"}}}]
    (tmp_path / "editorial-enriched.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf8")
    loaded = load_documents(corpus)
    assert requests == ["train"] and [d.problem_id for d in loaded] == [r.problem_id]


def test_hand_judged_metrics_and_no_unjudged_as_zero():
    grades = {"a": 2, "b": 1, "c": 0}
    perfect = ranking_metrics(["a", "b", "c"], grades)
    assert perfect == {"ndcg_at_10": 1., "mrr_at_10": 1.}
    value = ranking_metrics(["c", "b", "a"], grades)
    expected = (1 / np.log2(3) + 3 / np.log2(4)) / (3 + 1 / np.log2(3))
    assert value["ndcg_at_10"] == pytest.approx(expected)
    assert value["mrr_at_10"] == pytest.approx(1 / 3)
    assert ranking_metrics(["a", "a", "b"], grades) == ranking_metrics(["a", "b"], grades)
    with pytest.raises(ModelingError, match="Unjudged"):
        ranking_metrics(["unknown"], grades)


def test_blinded_review_and_evaluation_waits_for_actual_judgments(engine):
    result = pool_candidates(engine)
    assert result["queries"] == 24 and result["human_judgments"] == 0
    assert pool_candidates(engine)["reused"]
    queue = review_queue(engine)
    pending = queue["pending"]
    assert not {"method", "view", "rank", "score"} & pending.keys()
    unjudged = evaluate_retrieval(engine)
    assert unjudged["status"] == "awaiting_complete_human_judgments"
    assert all(r["metrics"] is None for r in unjudged["methods"].values())
    payload = {"pair_id": pending["pair_id"], "grade": 2, "rationale": "Synthetic fixture judgment only", "reviewer": "synthetic-test-reviewer", "human_attested": False}
    with pytest.raises(ModelingError, match="attestation"):
        record_judgment(engine, payload)
    pool, pool_hash = load_pool(engine)
    qid = pending["query_id"]
    for p in pool["pairs"]:
        if p["query_id"] == qid:
            record_judgment(engine, {**payload, "pair_id": p["pair_id"], "human_attested": True})
    report = evaluate_retrieval(engine)
    assert report["fully_judged_query_ids"] == [qid]
    assert all(v["fully_judged_queries"] == 1 for v in report["methods"].values())
    before = len(list((engine.settings.path("evaluation_path") / "judgments").glob("*.json")))
    record_judgment(engine, {**payload, "human_attested": True, "grade": 1})
    assert len(list((engine.settings.path("evaluation_path") / "judgments").glob("*.json"))) == before + 1
    assert latest_judgments(engine, pool_hash)[pending["pair_id"]]["grade"] == 1


def test_http_application_smoke_and_invalid_inputs(engine):
    server = make_server(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/", timeout=5) as response:
            html = response.read().decode()
            assert "Start with an approach" in html
            assert "script-src 'self'" in response.headers["Content-Security-Policy"]
            from bs4 import BeautifulSoup
            page = BeautifulSoup(html, "html.parser")
            assert "search-page" in page.body["class"]
            for control in ["query", "method", "view", "rating-min", "rating-max", "tag-boost", "classifier-boost"]:
                assert page.find(id=control) is not None and page.find("label", attrs={"for": control}) is not None
            assert page.find("textarea", id="query").get_text() == ""
            assert len(page.select("button.example[data-query]")) == 3
            assert page.find(id="reset-search")["type"] == "button"
            assert page.find(id="welcome") is not None
        with urllib.request.urlopen(url + "/api/info", timeout=5) as response:
            assert json.load(response)["searchable_count"] == 3
        request = urllib.request.Request(url + "/api/search", data=json.dumps({"query": "union find", "view": "editorial"}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.load(response)["results"]
        request = urllib.request.Request(url + "/api/search", data=b'{"query":""}', headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        assert caught.value.code == 400
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
