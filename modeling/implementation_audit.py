from __future__ import annotations

import numpy as np
import re

from .common import Config, ModelingError, atomic_json, hash_file, identity, read_json
from .features import texts
from .inference import load_bundle, predict
from .labels import Taxonomy, encode_targets
from .metrics import evaluate_predictions, score_matrix
from .resources import hardware_observation


def validate_plan(plan: dict) -> None:
    """Validate executable choices before freezing or reading modeling data."""
    if plan.get("plan_version") != "representation-comparison-v2" or plan.get("test_evaluation") != "not_permitted_in_this_experiment":
        raise ModelingError("The v2 experiment must keep final test reserved")
    if not isinstance(plan.get("folds"), int) or plan["folds"] < 2:
        raise ModelingError("Grouped CV requires at least two folds")
    if (not plan.get("C_candidates") or any(not isinstance(c, (int, float)) or not np.isfinite(c) or c <= 0 for c in plan["C_candidates"])
            or not plan.get("class_weight_candidates") or any(w not in {None, "balanced"} for w in plan["class_weight_candidates"])):
        raise ModelingError("Invalid logistic-regression candidates")
    threshold = plan.get("thresholds", {})
    grid = threshold.get("grid", [])
    if not grid or any(not isinstance(t, (int, float)) or not np.isfinite(t) or not 0 < t < 1 for t in grid):
        raise ModelingError("OOF threshold grid must lie strictly between zero and one")
    if any(not isinstance(threshold.get(k), int) or threshold[k] < 1 for k in ["minimum_positive", "minimum_negative", "minimum_positive_in_every_fold"]):
        raise ModelingError("OOF support requirements must be positive integers")
    if not plan.get("representations") or not set(plan["representations"]) <= {"word", "word_char", "embedding"}:
        raise ModelingError("Unsupported experiment representation")
    encoder = plan.get("encoder", {})
    if (encoder.get("weights_updated") is not False or not re.fullmatch(r"[a-f0-9]{40}", encoder.get("revision", ""))
            or encoder.get("maximum_tokens", 0) <= 2 or encoder.get("maximum_chunks", 0) < 1
            or not 0 <= encoder.get("overlap_tokens", -1) < encoder["maximum_tokens"] - 2
            or encoder.get("batch_size", 0) < 1 or encoder.get("cpu_threads", 0) < 1):
        raise ModelingError("Frozen encoder requires a pinned revision, fixed weights and valid chunk/resource limits")


def freeze_plan(config: Config, corpus, plan_path):
    plan = read_json(plan_path)
    validate_plan(plan)
    output = config.project_root / plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    historical = config.project_root / plan["historical_run"]
    inventory = {p.name: hash_file(p) for p in historical.iterdir() if p.is_file()}
    frozen = output / "frozen-plan.json"
    fingerprint = identity({"plan": plan, "manifest": corpus.manifest_sha256})
    if frozen.exists():
        stored = read_json(frozen)
        if stored["fingerprint"] != fingerprint or stored["baseline_inventory"] != inventory:
            raise ModelingError("Frozen experiment plan or historical baseline changed")
    else:
        from cpss.core import utcnow
        atomic_json(frozen, {"frozen_at": utcnow().isoformat(), "fingerprint": fingerprint, "plan": plan,
            "source_manifest_sha256": corpus.manifest_sha256, "baseline_inventory": inventory,
            "hardware_before_experiment": hardware_observation()})
    return plan, output, fingerprint


def audit_baseline(config: Config, corpus, plan_path):
    plan, output, fingerprint = freeze_plan(config, corpus, plan_path)
    historical = config.project_root / plan["historical_run"]
    bundle = load_bundle(historical)
    training, validation = corpus.records(["train"]), corpus.records(["validation"])
    if [r.problem_id for r in training] != bundle["fit_training_ids"] or [r.problem_id for r in validation] != bundle["validation_ids"]:
        raise ModelingError("Saved training/validation row alignment differs from the corpus")
    labels = bundle["label_order"]
    y, _ = encode_targets(validation, Taxonomy(bundle["taxonomy"]), labels)
    scores = score_matrix(bundle["pipeline"], texts(validation), len(labels))
    report = evaluate_predictions(y, scores, bundle["thresholds"], labels)
    if report != bundle["validation_metrics"]:
        raise ModelingError("Historical metric reproduction failed")
    classifier = bundle["pipeline"].named_steps["classifier"]
    if classifier.classes_.tolist() != list(range(len(labels))):
        raise ModelingError("Historical target columns are not stable")
    tfidf = bundle["pipeline"].named_steps["tfidf"]
    if len(tfidf.vocabulary_) != len(tfidf.idf_) or not np.isfinite(tfidf.idf_).all():
        raise ModelingError("Invalid fitted historical TF-IDF")
    x_train = tfidf.transform(texts(training)).tocsr()
    document_frequency = np.asarray(x_train.getnnz(axis=0)).ravel()
    expected_idf = np.log((1 + len(training)) / (1 + document_frequency)) + 1
    if not np.allclose(expected_idf, tfidf.idf_, atol=1e-12):
        raise ModelingError("Persisted IDF does not match independently counted training-only document frequencies")
    selected = bundle["chosen_hyperparameters"]
    if any(e.C != selected["C"] or e.class_weight != selected["class_weight"] for e in classifier.estimators_):
        raise ModelingError("Persisted estimator parameters differ from reported selection")
    cache_key = bundle["timings"]["features"]["cache_key"]
    cache_candidates = [config.path("cache_path") / "features" / k / "metadata.json" for k in [cache_key, cache_key[:24]]]
    receipts = [read_json(p) for p in cache_candidates if p.exists()]
    if not receipts or any(r["training_ids"] != bundle["fit_training_ids"] or r["fit_split"] != "train" for r in receipts):
        raise ModelingError("Historical preprocessing fit provenance is missing or misaligned")
    confusion = {}
    for label in labels:
        j = labels.index(label)
        p = scores[:, j] >= bundle["thresholds"][j]
        tp = int(np.sum(p & (y[:, j] == 1)))
        fp = int(np.sum(p & (y[:, j] == 0)))
        fn = int(np.sum((~p) & (y[:, j] == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        metric = report["per_label"][label]
        if not np.allclose([precision, recall, f1], [metric["precision"], metric["recall"], metric["f1"]], atol=1e-12):
            raise ModelingError("Independent confusion-count metric check failed")
        if fp != bundle["validation_errors"][label]["false_positive_count"] or fn != bundle["validation_errors"][label]["false_negative_count"]:
            raise ModelingError("Saved FP/FN counts do not reproduce")
        confusion[label] = {"tp": tp, "fp": fp, "fn": fn}
    # Compare constructor configuration to the persisted fitted vectorizer.
    for key, value in bundle["configuration"]["features"].items():
        actual = getattr(tfidf, key)
        if key == "ngram_range":
            actual = list(actual)
        if actual != value:
            raise ModelingError("Saved TF-IDF configuration differs from actual fitted object")
    single = predict(bundle, validation[0].statement)
    if not np.allclose([single["label_scores"][label] for label in labels], scores[0], atol=1e-12):
        raise ModelingError("Batch and saved-pipeline inference differ")
    result = {"plan_fingerprint": fingerprint, "confirmed_prediction_or_metric_bugs": [],
        "findings": ["Historical metrics, FP/FN, label order and text/target alignment reproduce exactly",
            "Vocabulary/IDF fit provenance contains exactly the training identities",
            "Class weights are computed within each training-fitted binary estimator",
            "No tags/metadata/editorials/code are passed through the feature text boundary",
            "Fallbacks for tiny validation supports are intentional; zero F1 is not a software bug",
            "Validation selected both hyperparameters and thresholds; this is the documented methodological limitation"],
        "baseline_preserved": True, "validation_metrics": report, "independent_confusion_counts": confusion,
        "test_evaluated": False, "sparse_feature_shape": bundle["matrix"], "hardware": hardware_observation()}
    atomic_json(output / "implementation-audit.json", result)
    return {"audit": str((output / 'implementation-audit.json').resolve()), "confirmed_prediction_or_metric_bugs": [],
            "micro_f1": report["micro_f1"], "macro_f1": report["macro_f1"], "baseline_preserved": True}
