from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from cpss.core import canonical, sha256
from modeling.common import Config, ModelingError, atomic_json, hash_file, read_json
from modeling.contexts import pending_problem, prepare_pilot, validate_context
from modeling.data import Corpus
from modeling.features import feature_key, fit_features, texts
from modeling.inference import load_bundle, predict
from modeling.labels import Taxonomy, encode_targets, select_labels
from modeling.metrics import select_thresholds
from modeling.training import fit_baseline, train_run
from modeling.evaluation import evaluate_run


class FixtureCorpus:
    manifest_sha256 = "a" * 64
    def __init__(self, records):
        self.train, self.validation = records
        self.structural = {r.problem_id: {"split": r.split, "version_id": r.version_id,
            "editorial_version_id": "e" * 64, "editorial_link_id": "f" * 64} for r in [*self.train, *self.validation]}
        self.records_requested = []
    def records(self, splits, final_test=False):
        self.records_requested.extend(splits)
        if "test" in splits and not final_test:
            raise ModelingError("test forbidden")
        return [r for r in [*self.train, *self.validation] if r.split in splits]
    def context_sources(self, problem_id):
        r = next(r for r in self.train if r.problem_id == problem_id)
        return {"statement": {"version_id": r.version_id, "text": r.statement},
                "editorial": {"version_id": "e" * 64, "text": "Use the supported approach."}}


def test_taxonomy_aliases_multi_parent_and_unreviewed_fine_labels(taxonomy):
    assert taxonomy.normalize(["DSU", "union-find", "binary-search"]) == ("binary search", "dsu")
    assert taxonomy.broad_families(["dsu"]) == ("family:data_structures", "family:graphs")
    assert "algorithm:dijkstra" not in taxonomy.normalize(["shortest paths"])
    assert all(d["status"] == "unreviewed" for d in taxonomy.spec["future_fine_labels"].values())


def test_invalid_labels_and_taxonomy_rejected(taxonomy, tmp_path):
    for tags in ([""], [None], ["   "]):
        with pytest.raises(ModelingError):
            taxonomy.normalize(tags)
    spec = copy.deepcopy(taxonomy.spec)
    spec["tags"]["dsu"]["parents"] = ["nonexistent"]
    path = tmp_path / "invalid-taxonomy.json"
    atomic_json(path, spec)
    with pytest.raises(ModelingError, match="parent"):
        Taxonomy.load(path)


def test_label_selection_training_only_missing_and_unsupported(records, taxonomy):
    train, validation = records
    with pytest.raises(ModelingError, match="training"):
        select_labels(validation, taxonomy, 2, 2)
    train = [*train, replace(train[0], problem_id="codeforces:999:A", original_tags=("unsupported future tag",))]
    result = select_labels(train, taxonomy, 2, 2)
    assert result["label_order"] == ["binary search", "dsu", "math"]
    assert result["excluded"]["fft"]["reason"] == "absent_from_training"
    assert result["excluded"]["unsupported future tag"]["reason"] == "unsupported_original_tag"
    y, unknown = encode_targets(train, taxonomy, result["label_order"])
    assert y.shape == (19, 3) and unknown == {"unsupported future tag": 1}
    with pytest.raises(ModelingError, match="No labels"):
        select_labels(train, taxonomy, 100, 2)
    with pytest.raises(ModelingError, match="ordering"):
        encode_targets(train, taxonomy, ["dsu", "dsu"])


def test_features_are_statement_only_sparse_and_cacheable(records, small_config):
    train, validation = records
    changed = [replace(r, problem_id="metadata:" + r.problem_id, original_tags=("other label",), title="LEAK_TOKEN") for r in train]
    assert texts(train) == texts(changed)
    with pytest.raises(ModelingError, match="training"):
        fit_features(validation, small_config.values["features"], small_config.path("cache_path"))
    vectorizer, matrix, first = fit_features(train, small_config.values["features"], small_config.path("cache_path"))
    assert sparse.isspmatrix_csr(matrix)
    assert "validationonlytoken" not in vectorizer.vocabulary_ and "leak_token" not in vectorizer.vocabulary_
    _, repeated, second = fit_features(train, small_config.values["features"], small_config.path("cache_path"))
    assert second["cache_hit"] and (matrix != repeated).nnz == 0
    vectorizer.transform(texts(validation))
    assert "validationonlytoken" not in vectorizer.vocabulary_


def test_cache_keys_invalidate_for_text_and_feature_configuration(records, small_config):
    train, _ = records
    base = feature_key(train, small_config.values["features"])
    changed = [replace(train[0], statement="new training words"), *train[1:]]
    assert feature_key(changed, small_config.values["features"]) != base
    assert feature_key(train, {**small_config.values["features"], "min_df": 2}) != base
    _, _, _ = fit_features(train, small_config.values["features"], small_config.path("cache_path"))
    cache = small_config.path("cache_path") / "features" / base[:24] / "train.npz"
    cache.write_bytes(b"corrupt")
    with pytest.raises(ModelingError, match="integrity"):
        fit_features(train, small_config.values["features"], small_config.path("cache_path"))


def test_validation_thresholds_handle_absent_labels(small_config):
    y = np.array([[1, 0], [0, 0]])
    scores = np.array([[0.7, 0.2], [0.1, 0.1]])
    thresholds, methods = select_thresholds(y, scores, ["dsu", "math"], small_config.values["thresholds"], 0.5)
    assert methods["math"]["method"] == "global_fallback_insufficient_validation_support"
    assert thresholds[1] == 0.5


def test_train_save_load_prediction_and_label_order(records, small_config, tmp_path):
    corpus = FixtureCorpus(records)
    result = train_run(small_config, corpus, "unit-run")
    run = Path(result["run"])
    loaded = load_bundle(run)
    assert loaded["label_order"] == ["binary search", "dsu", "math"]
    assert corpus.records_requested == ["train", "validation"]
    text = "Merge components and answer connectivity queries."
    before = predict(loaded, text)
    after = predict(load_bundle(run), text)
    assert before == after
    assert "confidence" not in before and "Uncalibrated" in before["score_semantics"]
    assert "validationonlytoken" not in loaded["pipeline"].named_steps["tfidf"].vocabulary_
    with pytest.raises(ModelingError):
        predict(loaded, " ")
    with pytest.raises(ModelingError):
        predict(loaded, {"statement": text, "tags": ["dsu"]})
    with pytest.raises(ModelingError, match="already exists"):
        train_run(small_config, corpus, "unit-run")
    with pytest.raises(ModelingError, match="requires"):
        evaluate_run(small_config, corpus, run, "test", final=False)
    # Tampering with label order must be detected before pickle loading.
    (run / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelingError, match="hash"):
        load_bundle(run)


def test_fitting_refuses_overlapping_or_test_records(records, taxonomy, small_config):
    train, validation = records
    with pytest.raises(ModelingError, match="isolated"):
        fit_baseline(train, [replace(validation[0], split="test")], taxonomy, small_config, small_config.path("cache_path"))
    with pytest.raises(ModelingError, match="overlap"):
        fit_baseline(train, [replace(validation[0], problem_id=train[0].problem_id)], taxonomy, small_config, small_config.path("cache_path"))


def test_context_pilot_deterministic_training_only(records, small_config):
    corpus = FixtureCorpus(records)
    result = prepare_pilot(small_config, corpus)
    path = Path(result["sample"])
    before = path.read_bytes()
    prepare_pilot(small_config, corpus)
    assert before == path.read_bytes()
    report = read_json(path)
    assert report["size"] == 6 and not report["bulk_generation_performed"]
    assert all(r["split"] == "train" for r in report["records"])
    assert corpus.records_requested == ["train", "train"]
    assert all(r["fine_annotations"] == "unknown_unreviewed" for r in report["records"])


def test_context_validation_evidence_review_and_holdout_guards(records, taxonomy):
    corpus = FixtureCorpus(records)
    row = records[0][0]
    doc = pending_problem(row.problem_id, taxonomy, corpus.manifest_sha256)
    validate_context(doc, taxonomy, corpus)
    invalid = {**doc, "problem_id": records[1][0].problem_id}
    with pytest.raises(ModelingError, match="training-only"):
        validate_context(invalid, taxonomy, corpus)
    doc.update(status="draft", abstract_objective="Maintain connectivity.")
    doc["supported_approaches"] = [{"algorithm_id": "technique:dsu", "status": "supported",
        "applicability_conditions": ["Insertions only"], "evidence": [{"source_kind": "statement", "version_id": row.version_id,
            "locator": "whole_text", "quote": "Merge components"}]}]
    validate_context(doc, taxonomy, corpus)
    invalid = copy.deepcopy(doc)
    invalid["supported_approaches"][0]["evidence"][0]["quote"] = "a fabricated quote"
    with pytest.raises(ModelingError, match="quote"):
        validate_context(invalid, taxonomy, corpus)
    invalid = copy.deepcopy(doc)
    invalid["supported_approaches"][0]["algorithm_id"] = "invented:label"
    with pytest.raises(ModelingError, match="vocabulary"):
        validate_context(invalid, taxonomy, corpus)
    invalid = {**doc, "status": "human_reviewed"}
    with pytest.raises(ModelingError, match="approval"):
        validate_context(invalid, taxonomy, corpus)


def test_corpus_loader_verification_and_no_test_target_access(tmp_path, small_config, monkeypatch):
    import modeling.data as module
    statements, pins = [], []
    for i, split in enumerate(["train", "validation", "test"]):
        text = f"Complete fixture statement {i}."
        row = {"problem_id": f"codeforces:{i + 1}:A", "split": split, "statement_version_id": sha256(str(i).encode()),
               "clean_sha256": sha256(text.encode()), "metadata": {"contest_id": i + 1, "problem_index": "A", "title": "Fixture",
                   "corpus": {"aliases": [], "source_split": "test" if split == "test" else "train"}},
               "target": {"original_tags": ["dsu"] if split != "test" else {"malformed_test_target": True}},
               "input": {"statement": text}}
        statements.append(row)
        pins.append({k: v for k, v in row.items() if k != "input"})
    body = b"".join(canonical(r) + b"\n" for r in statements)
    export = tmp_path / "export"
    export.mkdir()
    (export / "statement-only.jsonl").write_bytes(body)
    manifest = {"files": {"statement-only.jsonl": {"sha256": sha256(body), "count": 3}}, "blobs": [],
                "records": pins, "configuration": {"duplicate_groups": []}, "counts": {"problems": 3}}
    raw = canonical(manifest)
    (export / "manifest.json").write_bytes(raw)
    (export / "manifest.sha256").write_text(sha256(raw), encoding="ascii")
    values = copy.deepcopy(small_config.values)
    values.update(export_path=str(export), expected_manifest_sha256=sha256(raw), expected_split_counts={"train": 1, "validation": 1, "test": 1})
    config = Config(values)
    calls = []
    original_verifier = module.verify_export
    def verifier(path):
        calls.append(1)
        return original_verifier(path)
    monkeypatch.setattr(module, "verify_export", verifier)
    corpus = Corpus.verified(config)
    assert len(corpus.records(["train"])) == 1
    assert len(corpus.records(["validation"])) == 1
    with pytest.raises(ModelingError, match="explicit"):
        corpus.records(["test"])
    assert len(calls) == 1
    Corpus.verified(config)
    assert len(calls) == 1
    (export / "statement-only.jsonl").write_bytes(body + b"\n")
    with pytest.raises(Exception):
        Corpus.verified(config)
    assert len(calls) == 2


def test_in_memory_and_persisted_inference_scores_match(records, taxonomy, small_config, tmp_path):
    import joblib
    bundle = fit_baseline(*records, taxonomy, small_config, small_config.path("cache_path"))
    bundle["run_id"] = "synthetic-unit"
    statement = "Find a monotone threshold in a sorted sequence."
    before = predict(bundle, statement)
    path = tmp_path / "round-trip.joblib"
    joblib.dump(bundle, path)
    after = predict(joblib.load(path), statement)
    assert before == after


def test_single_supported_label_has_consistent_score_shape(records, taxonomy, small_config):
    train, validation = records
    # Keep DSU as the sole supported target; unknown assignments form negatives.
    train = [replace(r, original_tags=("dsu",) if i % 2 else ("unsupported",)) for i, r in enumerate(train)]
    validation = [replace(r, original_tags=("dsu",) if i % 2 else ("unsupported",)) for i, r in enumerate(validation)]
    bundle = fit_baseline(train, validation, taxonomy, small_config, small_config.path("cache_path"))
    bundle["run_id"] = "single-label-unit"
    assert bundle["label_order"] == ["dsu"]
    assert set(predict(bundle, "Merge connected components.")["label_scores"]) == {"dsu"}


def test_final_evaluation_is_explicit_and_single_use_on_synthetic_data(records, small_config):
    corpus = FixtureCorpus(records)
    result = train_run(small_config, corpus, "heldout-unit")
    run = Path(result["run"])
    original_records = corpus.records
    def records_with_synthetic_test(splits, final_test=False):
        if splits == ["test"]:
            assert final_test
            return [replace(r, split="test", problem_id="heldout:" + r.problem_id) for r in records[1]]
        return original_records(splits, final_test=final_test)
    corpus.records = records_with_synthetic_test
    report = evaluate_run(small_config, corpus, run, "test", final=True)
    assert report["split"] == "test" and not report["thresholds_refitted"]
    with pytest.raises(ModelingError, match="already invoked"):
        evaluate_run(small_config, corpus, run, "test", final=True)


def test_illustrative_contexts_and_query_evidence(taxonomy):
    from modeling.examples.make_illustrations import build
    build()
    root = Path(__file__).parents[1] / "examples"
    sources = read_json(root / "synthetic-sources.json")
    for name in ("problem-dsu.json", "problem-binary-search.json", "query-connectivity.json"):
        document = read_json(root / name)
        validate_context(document, taxonomy, illustrative_sources=sources)
        assert document["status"] == "draft" and document["review"]["decision"] == "pending"
    bad = read_json(root / "query-connectivity.json")
    bad["inferred_approaches"][0]["evidence_spans"] = ["invented user requirement"]
    with pytest.raises(ModelingError, match="span"):
        validate_context(bad, taxonomy)
    bad = read_json(root / "problem-dsu.json")
    bad["status"] = "human_reviewed"
    bad["review"] = {"decision": "approved", "reviewer": "unit-test", "reviewed_at": "2026-10-26T00:00:00Z"}
    with pytest.raises(ModelingError, match="Illustrations"):
        validate_context(bad, taxonomy, illustrative_sources=sources)


def test_invalid_configuration_and_empty_statement(records, small_config, tmp_path):
    values = copy.deepcopy(small_config.values)
    values["thresholds"]["grid"] = [0, 1]
    path = tmp_path / "invalid-config.json"
    atomic_json(path, values)
    with pytest.raises(ModelingError, match="Threshold"):
        Config.load(path)
    with pytest.raises(ModelingError, match="nonempty"):
        texts([replace(records[0][0], statement="")])
