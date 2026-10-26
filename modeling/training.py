from __future__ import annotations

import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .common import Config, ModelingError, atomic_json, dependency_versions, hash_file, identity, read_json
from .data import Corpus, Record
from .features import fit_features, texts
from .labels import Taxonomy, encode_targets, select_labels
from .metrics import best_global_threshold, error_examples, evaluate_predictions, score_matrix, select_thresholds


def fit_baseline(train: list[Record], validation: list[Record], taxonomy: Taxonomy, config: Config, cache: Path) -> dict[str, Any]:
    if not train or not validation or any(r.split != "train" for r in train) or any(r.split != "validation" for r in validation):
        raise ModelingError("Baseline fitting needs isolated train and validation records")
    if set(r.problem_id for r in train) & set(r.problem_id for r in validation):
        raise ModelingError("Training and validation identities overlap")
    started = time.perf_counter()
    selection = select_labels(train, taxonomy, config.values["labels"]["min_train_positive"], config.values["labels"]["min_train_negative"])
    labels = selection["label_order"]
    y_train, unsupported_train = encode_targets(train, taxonomy, labels)
    y_validation, unsupported_validation = encode_targets(validation, taxonomy, labels)
    vectorizer, x_train, feature_timing = fit_features(train, config.values["features"], cache)
    transform_start = time.perf_counter()
    x_validation = vectorizer.transform(texts(validation)).tocsr()
    transform_seconds = time.perf_counter() - transform_start
    candidates, best = [], None
    for c in config.values["classifier"]["C_candidates"]:
        for weight in config.values["classifier"]["class_weight_candidates"]:
            start = time.perf_counter()
            estimator = OneVsRestClassifier(LogisticRegression(C=c, class_weight=weight,
                solver=config.values["classifier"]["solver"], max_iter=config.values["classifier"]["max_iter"],
                tol=config.values["classifier"]["tol"], random_state=config.values["seed"]), n_jobs=1)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise ModelingError("Logistic regression did not converge; adjust training configuration")
            fit_seconds = time.perf_counter() - start
            scores = score_matrix(estimator, x_validation, len(labels))
            global_threshold, grid_scores = best_global_threshold(y_validation, scores, config.values["thresholds"]["grid"])
            metric = next(r for r in grid_scores if r["threshold"] == global_threshold)
            candidate = {"C": c, "class_weight": weight, "fit_seconds": fit_seconds,
                         "global_threshold": global_threshold, "micro_f1": metric["micro_f1"], "macro_f1": metric["macro_f1"],
                         "threshold_grid": grid_scores}
            candidates.append(candidate)
            ranking = (candidate["micro_f1"], candidate["macro_f1"], -c, weight is None)
            if best is None or ranking > best[0]:
                best = (ranking, estimator, scores, candidate)
    assert best is not None
    _, classifier, scores, chosen = best
    thresholds, threshold_methods = select_thresholds(y_validation, scores, labels, config.values["thresholds"], chosen["global_threshold"])
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    inference_start = time.perf_counter()
    with threadpool_limits(limits=1):
        score_matrix(pipeline, texts(validation), len(labels))
    inference_seconds = time.perf_counter() - inference_start
    metrics = evaluate_predictions(y_validation, scores, thresholds, labels)
    return {"pipeline": pipeline, "label_order": labels, "thresholds": thresholds,
            "taxonomy": taxonomy.spec, "selection": selection, "configuration": config.values,
            "chosen_hyperparameters": {"C": chosen["C"], "class_weight": chosen["class_weight"]},
            "threshold_methods": threshold_methods, "candidate_results": candidates,
            "validation_metrics": metrics, "validation_errors": error_examples(validation, y_validation, scores, thresholds, labels),
            "unmodeled_observed_tags": {"train": unsupported_train, "validation": unsupported_validation},
            "fit_training_ids": [r.problem_id for r in train], "validation_ids": [r.problem_id for r in validation],
            "timings": {"total_development_seconds": time.perf_counter() - started, "features": feature_timing,
                        "validation_transform_seconds": transform_seconds, "selected_estimator_fit_seconds": chosen["fit_seconds"],
                        "batch_inference_seconds": inference_seconds, "batch_inference_records": len(validation),
                        "mean_inference_milliseconds": 1000 * inference_seconds / len(validation)},
            "matrix": {"shape": list(x_train.shape), "nonzero": x_train.nnz, "format": "csr"},
            "score_semantics": "Uncalibrated independent logistic-regression label scores, not calibrated confidence"}


def train_run(config: Config, corpus: Corpus, run_name: str | None = None) -> dict[str, Any]:
    train = corpus.records(["train"])
    validation = corpus.records(["validation"])
    taxonomy = Taxonomy.load(config.path("taxonomy_path"))
    provenance = {"source_manifest_sha256": corpus.manifest_sha256, "configuration_sha256": config.fingerprint,
                  "taxonomy_sha256": taxonomy.fingerprint, "dependency_versions": dependency_versions(), "seed": config.values["seed"],
                  "feature_source": "statement-only.jsonl:input.statement", "fit_split": "train", "selection_split": "validation",
                  "test_evaluated": False, "modeling_version": "0.1.0"}
    run_id = "tfidf-ovr-" + identity(provenance)[:16]
    name = run_name or run_id
    if Path(name).name != name or not name or name in {".", ".."}:
        raise ModelingError("Run name must be a simple directory name")
    output = config.path("artifacts_path") / name
    if output.exists():
        raise ModelingError("Run directory already exists; use its frozen model or choose a new run name")
    bundle = fit_baseline(train, validation, taxonomy, config, config.path("cache_path"))
    bundle.update(provenance=provenance, run_id=run_id)
    building = output.with_name(output.name + ".building-" + uuid.uuid4().hex)
    building.mkdir(parents=True)
    joblib.dump(bundle, building / "pipeline.joblib", compress=3)
    run_metadata = {k: v for k, v in bundle.items() if k not in {"pipeline", "thresholds", "validation_errors"}}
    run_metadata["thresholds"] = bundle["thresholds"].tolist()
    atomic_json(building / "run.json", run_metadata)
    atomic_json(building / "validation-errors.json", bundle["validation_errors"])
    lines = ["# Actual validation results", "", f"Run: `{run_id}`", "",
             f"Micro F1: **{bundle['validation_metrics']['micro_f1']:.4f}**; macro F1: **{bundle['validation_metrics']['macro_f1']:.4f}**.",
             "", "Inputs: statement text only. Training: train only. Hyperparameters and thresholds: validation only. Test: not evaluated.",
             "", "| Label | Precision | Recall | F1 | Support | Threshold |", "|---|---:|---:|---:|---:|---:|"]
    for label, metric in bundle["validation_metrics"]["per_label"].items():
        lines.append(f"| {label} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['f1']:.3f} | {metric['support']} | {bundle['threshold_methods'][label]['threshold']:.2f} |")
    lines.extend(["", "Scores are uncalibrated. Reported errors are against observed platform tags; missing tags do not establish algorithm unsuitability.",
                  "", "The same validation split selected hyperparameters and thresholds, so this is development performance, not an unbiased final estimate."])
    (building / "validation-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(building / "artifact-manifest.json", {"run_id": run_id, "source_manifest_sha256": corpus.manifest_sha256,
        "files": {p.name: hash_file(p) for p in building.iterdir() if p.is_file()}})
    building.rename(output)
    atomic_json(config.path("artifacts_path") / "latest-run.json", {"path": str(output.resolve()), "run_id": run_id})
    return {"run": str(output.resolve()), "validation": bundle["validation_metrics"], "timings": bundle["timings"],
            "chosen_hyperparameters": bundle["chosen_hyperparameters"], "test_evaluated": False}
