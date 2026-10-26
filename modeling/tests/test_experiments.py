from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.pipeline import Pipeline

from modeling.common import ModelingError, atomic_json, read_json
from modeling.encoder import FrozenEncoder, attention_mean, prepare_embeddings, token_chunks
from modeling.error_review import code_signal, evidence_signal, propose_outcome, select_cases, validate_review, validate_source_signals
from modeling.experiment import choose_candidate, fit_head, oof_thresholds, run_representation
from modeling.grouping import grouped_folds, training_groups
from modeling.implementation_audit import freeze_plan, validate_plan
from modeling.metrics import evaluate_predictions, score_matrix
from modeling.representations import cached_lexical, fold_features


@pytest.fixture
def plan(small_config):
    return {"word": small_config.values["features"],
            "character": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 1, "max_features": 250},
            "character_weight": 0.5}


def test_transitive_groups_and_deterministic_folds(records, tmp_path):
    training = [replace(r, clean_sha256=f"hash-{i}") for i, r in enumerate(records[0])]
    training[4] = replace(training[4], title="A task (easy version)")
    training[5] = replace(training[5], title="A task (hard version)")
    structural = {r.problem_id: {"contest_id": i, "aliases": []} for i, r in enumerate(training)}
    # Contest -> alias -> explicit duplicate is a transitive four-row group.
    structural[training[1].problem_id]["contest_id"] = 0
    structural[training[2].problem_id]["aliases"] = [training[1].problem_id.removeprefix("codeforces:").replace(":", "/")]
    manifest = {"records": [{"problem_id": r.problem_id, "metadata": {"contest": {"started_at": "2020-01-01T00:00:00"}}} for r in training],
                "configuration": {"duplicate_groups": [[training[2].problem_id, training[3].problem_id]]}}
    atomic_json(tmp_path / "manifest.json", manifest)
    corpus = SimpleNamespace(directory=tmp_path, structural=structural)
    groups = training_groups(training, corpus)
    assert len(set(groups[:4])) == 1 and groups[4] == groups[5]
    assert groups[6] != groups[7]  # Generic equal titles are not used.
    folds = grouped_folds(training, groups, 3, 42)
    again = grouped_folds(training, groups, 3, 42)
    assert all(np.array_equal(a, b) for f, g in zip(folds, again) for a, b in zip(f, g))
    assert sorted(np.concatenate([held for _, held in folds])) == list(range(len(training)))
    for fit, held in folds:
        assert not set(np.array(groups)[fit]) & set(np.array(groups)[held])
    with pytest.raises(ModelingError, match="training-only"):
        training_groups(records[1], corpus)


@pytest.mark.parametrize("kind", ["word", "word_char"])
def test_fold_local_vocabulary_sparse_cache_and_invalidation(records, plan, tmp_path, kind):
    training = records[0][:12]
    held = [replace(r, statement=r.statement + " quarantinezebra", clean_sha256="new-" + r.clean_sha256) for r in records[0][12:]]
    transformer, fit, other, key = fold_features(training, held, kind, plan, tmp_path)
    word = transformer if kind == "word" else dict(transformer.transformer_list)["word"]
    assert "quarantinezebra" not in word.vocabulary_
    assert sparse.isspmatrix_csr(fit) and sparse.isspmatrix_csr(other)
    loaded, fit2, other2, same = fold_features(training, held, kind, plan, tmp_path)
    assert same == key and (fit != fit2).nnz == (other != other2).nnz == 0
    changed = copy.deepcopy(plan)
    changed["word"]["max_features"] += 1
    assert fold_features(training, held, kind, changed, tmp_path)[3] != key
    altered = [replace(training[0], statement="Entirely new token", clean_sha256="changed"), *training[1:]]
    assert fold_features(altered, held, kind, plan, tmp_path)[3] != key
    cache = tmp_path / "cv-features" / key[:20] / "other.npz"
    cache.write_bytes(b"corrupt")
    with pytest.raises(ModelingError, match="integrity"):
        fold_features(training, held, kind, plan, tmp_path)
    with pytest.raises(ModelingError, match="overlap"):
        fold_features(training, training[:1], kind, plan, tmp_path)
    with pytest.raises(ModelingError, match="training records only"):
        fold_features(training, records[1], kind, plan, tmp_path)


def test_oof_selection_and_threshold_reliability():
    y = np.array([[1, 1], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0]])
    scores = np.array([[.7, .4], [.2, .3], [.6, .2], [.1, .1], [.65, .1], [.3, .2]])
    folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])),
             (np.array([0, 1, 4, 5]), np.array([2, 3])),
             (np.array([0, 1, 2, 3]), np.array([4, 5]))]
    thresholds, methods, global_choice = oof_thresholds(y, scores, ["supported", "rare"], folds,
        {"grid": [.25, .5, .75], "minimum_positive": 2, "minimum_negative": 2, "minimum_positive_in_every_fold": 1})
    assert methods["supported"]["method"] == "training_oof_per_label_f1"
    assert methods["rare"]["method"] == "training_oof_global_fallback"
    assert thresholds[1] == global_choice["threshold"]
    candidates = [{"C": 1, "class_weight": None, "oof_at_fixed_0_5": {"macro_f1": .5, "micro_f1": .7}},
                  {"C": 3, "class_weight": "balanced", "oof_at_fixed_0_5": {"macro_f1": .4, "micro_f1": .9}}]
    assert choose_candidate(candidates) is candidates[0]
    # Validation performance cannot enter this selection boundary.
    candidates[1]["validation_micro_f1"] = 1.0
    assert choose_candidate(candidates) is candidates[0]


def test_word_character_roundtrip_preserves_order_and_scores(records, small_config, plan, tmp_path):
    training, validation = records
    transformer, fit, other, _ = cached_lexical(training, validation, "word_char", plan, tmp_path)
    labels = ["math", "dsu", "binary search"]  # Deliberately nonalphabetical.
    y = np.array([[int(label in r.original_tags) for label in labels] for r in training])
    head = fit_head(fit, y, 1, "balanced", 42, small_config.values["classifier"])
    pipeline = Pipeline([("representation", transformer), ("classifier", head)])
    before = score_matrix(head, other, len(labels))
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    after = score_matrix(joblib.load(path), [r.statement for r in validation], len(labels))
    np.testing.assert_allclose(before, after, atol=1e-12)
    assert head.classes_.tolist() == [0, 1, 2]
    result = evaluate_predictions(y, score_matrix(head, fit, 3), [.5] * 3, labels)
    assert list(result["per_label"]) == labels


def test_encoder_chunk_coverage_pooling_bounds_and_holdout_guard(records, tmp_path):
    ids = list(range(45))
    chunks = token_chunks(ids, maximum=12, overlap=3, maximum_chunks=10)
    assert set(n for chunk, _ in chunks for n in chunk) == set(ids)
    assert sum(weight for _, weight in chunks) == len(ids)
    assert all(len(chunk) <= 10 for chunk, _ in chunks)
    with pytest.raises(ModelingError, match="no silent truncation"):
        token_chunks(ids, 12, 3, 2)
    with pytest.raises(ModelingError, match="Invalid"):
        token_chunks(ids, 12, 10, 10)
    with pytest.raises(ModelingError, match="Empty"):
        token_chunks([], 12, 3, 10)
    hidden = np.array([[[2., 4.], [4., 8.], [100., 100.]]])
    np.testing.assert_allclose(attention_mean(hidden, np.array([[1, 1, 0]])), [[3., 6.]])
    with pytest.raises(ModelingError, match="attention"):
        attention_mean(hidden, np.zeros((1, 3)))
    with pytest.raises(ModelingError, match="test statements"):
        prepare_embeddings([replace(records[0][0], split="test")], {}, tmp_path, tmp_path)


def test_encoder_document_batch_invariance(monkeypatch):
    class Tokenizer:
        def token_to_id(self, token):
            return {"[CLS]": 1, "[SEP]": 2, "[PAD]": 0}[token]
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[int(n) for n in text.split()])
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=n) for n in ["input_ids", "attention_mask", "token_type_ids"]]
        def run(self, _, inputs):
            # Simulate a batch-sensitive quantized encoder.
            ids = inputs["input_ids"]
            hidden = np.broadcast_to(ids[:, :, None], (*ids.shape, 384)).astype(np.float32).copy()
            hidden[:, :, 0] += ids.sum()
            return [hidden]
    def load(self):
        self._tokenizer, self._session = Tokenizer(), Session()
    monkeypatch.setattr(FrozenEncoder, "_load", load)
    encoder = FrozenEncoder("unused", {"maximum_tokens": 8, "overlap_tokens": 2, "maximum_chunks": 10, "batch_size": 3})
    documents = ["3 4 5 6 7 8 9 10", "11 12 13"]
    batch = encoder.transform(documents)
    singles = np.vstack([encoder.transform([s]) for s in documents])
    np.testing.assert_array_equal(batch, singles)
    assert batch.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-6)
    assert not {"_session", "_tokenizer"} & encoder.__getstate__().keys()
    with pytest.raises(ModelingError, match="nonempty"):
        encoder.transform([""])


def test_missing_encoder_runtime_is_an_explicit_optional_boundary(monkeypatch):
    import modeling.encoder as module
    from importlib.metadata import PackageNotFoundError
    def missing(_):
        raise PackageNotFoundError("synthetic-unavailable-runtime")
    monkeypatch.setattr(module, "version", missing)
    with pytest.raises(ModelingError, match="requirements-encoder.lock"):
        module.runtime_versions()


def test_review_strata_deterministic_and_evidence_qualifiers(records):
    data = records[1]
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    scores = np.array([[.8, .7], [.7, .8], [.2, .2], [.2, .2], [.8, .1], [.1, .8]])
    specification = {"labels": ["dsu", "binary search"], "buckets": ["true_positive", "false_negative", "false_positive", "true_negative"], "cases_per_label_and_bucket": 1}
    selected = select_cases(data, y, scores, specification["labels"], [.5, .5], specification, 42)
    assert selected == select_cases(data, y, scores, specification["labels"], [.5, .5], specification, 42)
    assert len(selected) == 8
    for label, bucket, index in selected:
        j = specification["labels"].index(label)
        assert bool(y[index, j]) == (bucket in {"true_positive", "false_negative"})
    text = "Use binary search to find a monotone threshold."
    signal = evidence_signal(text, r"binary search")
    assert text[signal["start"]:signal["end"]] == signal["quote"]
    assert propose_outcome("false_positive", signal, [signal])[0] == "potentially_incomplete_tagging"
    assert propose_outcome("false_positive", signal, [signal], ambiguity=True)[0] == "unresolved"
    with pytest.raises(ModelingError, match="Invalid automated"):
        validate_review({"schema_version": "error-review-v1", "human_reviewed_cases": 1})
    assert code_signal("#define lb(x) lower_bound(x)\n// lower_bound(template)\ns.lower_bound(x);", "binary search") is None
    assert code_signal("auto p = lower_bound(a.begin(), a.end(), x);", "binary search") is not None


def test_review_quote_and_source_version_validation():
    text = "Use binary search on the answer."
    signal = {"source_kind": "editorial", "version_id": "editorial-1", **evidence_signal(text, "binary search")}
    case = {"problem_id": "codeforces:1:A", "supporting_signals": [signal]}
    row = {"problem_id": "codeforces:1:A", "editorial_version_id": "editorial-1", "input": {"editorial": text}, "solutions": []}
    validate_source_signals(case, row)
    corrupt = copy.deepcopy(case)
    corrupt["supporting_signals"][0]["quote"] = "fabricated evidence"
    with pytest.raises(ModelingError, match="quote/offset/version"):
        validate_source_signals(corrupt, row)
    corrupt = copy.deepcopy(case)
    corrupt["supporting_signals"][0]["version_id"] = "another-version"
    with pytest.raises(ModelingError, match="quote/offset/version"):
        validate_source_signals(corrupt, row)


def test_frozen_plan_guards_baseline_bytes_and_plan(records, small_config, tmp_path, monkeypatch):
    import modeling.implementation_audit as audit
    monkeypatch.setattr(audit, "hardware_observation", lambda: {})
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "frozen.bin").write_bytes(b"keep")
    plan = read_json(small_config.project_root / "modeling/config/experiment-v2.json")
    plan.update(historical_run=str(baseline), output=str(tmp_path / "output"))
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    corpus = SimpleNamespace(manifest_sha256="a" * 64)
    first = freeze_plan(small_config, corpus, path)
    assert freeze_plan(small_config, corpus, path)[2] == first[2]
    (baseline / "frozen.bin").write_bytes(b"changed")
    with pytest.raises(ModelingError, match="baseline changed"):
        freeze_plan(small_config, corpus, path)


def test_invalid_plan_cannot_change_test_or_encoder_contract(small_config):
    plan = read_json(small_config.project_root / "modeling/config/experiment-v2.json")
    validate_plan(plan)
    wrong = copy.deepcopy(plan)
    wrong["test_evaluation"] = "enabled"
    with pytest.raises(ModelingError, match="test reserved"):
        validate_plan(wrong)
    wrong = copy.deepcopy(plan)
    wrong["encoder"]["weights_updated"] = True
    with pytest.raises(ModelingError, match="fixed weights"):
        validate_plan(wrong)
    wrong = copy.deepcopy(plan)
    wrong["thresholds"]["grid"] = [float("nan")]
    with pytest.raises(ModelingError, match="strictly"):
        validate_plan(wrong)


def test_complete_experiment_persistence_and_no_test_access(records, small_config, tmp_path, plan, monkeypatch):
    from cpss.core import sha256
    from modeling.inference import load_bundle, predict
    from modeling.training import train_run
    from modeling.tests.test_modeling import FixtureCorpus
    import modeling.implementation_audit as audit
    monkeypatch.setattr(audit, "hardware_observation", lambda: {})
    training = []
    for i, r in enumerate(records[0]):
        statement = r.statement + f" Unique document number {i}."
        training.append(replace(r, statement=statement, clean_sha256=sha256(statement.encode())))
    corpus = FixtureCorpus((training, records[1]))
    corpus.directory = tmp_path
    for i, r in enumerate(training):
        corpus.structural[r.problem_id].update(contest_id=i, aliases=[])
    atomic_json(tmp_path / "manifest.json", {"configuration": {"duplicate_groups": []}, "records": [
        {"problem_id": r.problem_id, "metadata": {"contest": {"started_at": "2020-01-01"}}} for r in training]})
    historical = train_run(small_config, corpus, "historical-fixture")["run"]
    specification = read_json(small_config.project_root / "modeling/config/experiment-v2.json")
    specification.update(plan, historical_run=historical, output=str(tmp_path / "experiments"), C_candidates=[1.0], class_weight_candidates=[None])
    specification["thresholds"].update(minimum_positive=2, minimum_negative=2, minimum_positive_in_every_fold=1)
    path = tmp_path / "plan.json"
    atomic_json(path, specification)
    corpus.records_requested.clear()
    result = run_representation(small_config, corpus, path, "word_char")
    from pathlib import Path
    bundle = load_bundle(Path(result["run"]))
    frozen = read_json(Path(result["run"]) / "run.json")
    assert bundle["label_order"] == load_bundle(Path(historical))["label_order"]
    assert corpus.records_requested == ["train", "validation"]
    assert frozen["provenance"]["selection_split"] == "training_grouped_oof"
    assert not frozen["provenance"]["test_evaluated"]
    statement = "Find connected components after inserting an edge."
    np.testing.assert_allclose(list(predict(bundle, statement)["label_scores"].values()),
                               score_matrix(bundle["pipeline"], [statement], len(bundle["label_order"]))[0])
    # A repeated invocation loads the frozen run without refitting/re-exposing data.
    assert run_representation(small_config, corpus, path, "word_char")["cached_completed_run"]
    assert corpus.records_requested == ["train", "validation"]
    # Hash-valid artifacts copied from a different plan must not be reused.
    import shutil
    moved_output = tmp_path / "other-experiment"
    moved_output.mkdir()
    shutil.copytree(result["run"], moved_output / "word_char")
    specification["output"] = str(moved_output)
    atomic_json(path, specification)
    with pytest.raises(ModelingError, match="different frozen plan"):
        run_representation(small_config, corpus, path, "word_char")
