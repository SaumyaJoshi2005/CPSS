from __future__ import annotations

import gc
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .common import Config, ModelingError, atomic_json, dependency_versions, hash_file, identity, read_json
from .data import Corpus
from .features import texts
from .grouping import grouped_folds, training_groups
from .implementation_audit import freeze_plan
from .inference import load_bundle
from .labels import Taxonomy, encode_targets
from .metrics import best_global_threshold, error_examples, evaluate_predictions, score_matrix
from .representations import cached_lexical, fold_features
from .resources import memory_observation


def fit_head(x, y, c, weight, seed, configuration):
    classifier = OneVsRestClassifier(LogisticRegression(C=c, class_weight=weight, solver="liblinear",
        random_state=seed, max_iter=configuration["max_iter"], tol=configuration["tol"]), n_jobs=1)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ModelingError("An experiment head failed to converge; this candidate is not silently accepted")
    return classifier


def oof_thresholds(y, scores, labels, folds, specification):
    global_value, global_grid = best_global_threshold(y, scores, specification["grid"])
    thresholds, methods = [], {}
    for j, label in enumerate(labels):
        positives = int(y[:, j].sum())
        negatives = len(y) - positives
        by_fold = [int(y[held, j].sum()) for _, held in folds]
        reliable = positives >= specification["minimum_positive"] and negatives >= specification["minimum_negative"] and min(by_fold) >= specification["minimum_positive_in_every_fold"]
        if reliable:
            values = [(float(f1_score(y[:, j], scores[:, j] >= t, zero_division=0)), t) for t in specification["grid"]]
            _, threshold = max(values, key=lambda v: (v[0], -abs(v[1] - global_value), v[1]))
            method = "training_oof_per_label_f1"
        else:
            threshold, method = global_value, "training_oof_global_fallback"
        thresholds.append(threshold)
        methods[label] = {"threshold": threshold, "method": method, "training_positive": positives,
                          "training_negative": negatives, "heldout_fold_positive_support": by_fold}
    return np.asarray(thresholds), methods, {"threshold": global_value, "grid": global_grid}


def candidates_from_plan(plan):
    return [{"C": c, "class_weight": weight} for c in plan["C_candidates"] for weight in plan["class_weight_candidates"]]


def choose_candidate(reports):
    return max(reports, key=lambda r: (r["oof_at_fixed_0_5"]["macro_f1"], r["oof_at_fixed_0_5"]["micro_f1"],
                                       -r["C"], r["class_weight"] is None))


def verified_experiment_run(path: Path, plan_hash: str, manifest_hash: str, kind: str) -> dict:
    bundle = load_bundle(path)
    if (bundle["provenance"].get("plan_sha256") != plan_hash
            or bundle["provenance"]["source_manifest_sha256"] != manifest_hash
            or bundle["configuration"].get("representation") != kind):
        raise ModelingError("Completed run belongs to a different frozen plan, corpus or representation")
    return read_json(path / "run.json")


def run_representation(config: Config, corpus: Corpus, plan_path: Path, kind: str,
                       progress: Callable[[dict], None] | None = None) -> dict:
    plan, output, plan_hash = freeze_plan(config, corpus, plan_path)
    if kind not in plan["representations"]:
        raise ModelingError("Representation is absent from the frozen plan")
    destination = output / kind
    if destination.exists():
        saved = verified_experiment_run(destination, plan_hash, corpus.manifest_sha256, kind)
        return {"representation": kind, "cached_completed_run": True, "validation": saved["validation_metrics"], "run": str(destination.resolve())}
    historical = load_bundle(config.project_root / plan["historical_run"])
    labels = historical["label_order"]
    taxonomy = Taxonomy(historical["taxonomy"])
    training = corpus.records(["train"])
    validation = corpus.records(["validation"])
    groups = training_groups(training, corpus)
    folds = grouped_folds(training, groups, plan["folds"], plan["seed"])
    y, unsupported_train = encode_targets(training, taxonomy, labels)
    y_validation, unsupported_validation = encode_targets(validation, taxonomy, labels)
    fold_receipt = {"seed": plan["seed"], "source_manifest_sha256": corpus.manifest_sha256,
        "groups": dict(zip([r.problem_id for r in training], groups)),
        "folds": [{"fit_ids": [training[i].problem_id for i in fit], "heldout_ids": [training[i].problem_id for i in held],
                   "heldout_label_support": dict(zip(labels, y[held].sum(axis=0).tolist()))} for fit, held in folds]}
    receipt = output / "training-folds.json"
    if receipt.exists() and read_json(receipt) != fold_receipt:
        raise ModelingError("Frozen training folds changed")
    atomic_json(receipt, fold_receipt)
    started = time.perf_counter()
    before = memory_observation()
    embedding_data, encoder, encoder_metadata = None, None, None
    if kind == "embedding":
        from .encoder import prepare_embeddings
        encoder, embedding_data, encoder_metadata = prepare_embeddings([*training, *validation], plan["encoder"], config.path("cache_path"), output, progress)
    candidates = candidates_from_plan(plan)
    oof = {identity(c): np.zeros(y.shape, dtype=np.float32) for c in candidates}
    fit_times = {identity(c): [] for c in candidates}
    fold_cache_keys, absent_training = [], []
    for fold_number, (fit, held) in enumerate(folds):
        if kind == "embedding":
            x_fit, x_held = np.asarray(embedding_data[fit]), np.asarray(embedding_data[held])
            fold_cache_keys.append(encoder_metadata["key"])
        else:
            _, x_fit, x_held, key = fold_features([training[i] for i in fit], [training[i] for i in held], kind, plan, config.path("cache_path"))
            fold_cache_keys.append(key)
        absent_training.append([label for j, label in enumerate(labels) if y[fit, j].sum() in {0, len(fit)}])
        for candidate in candidates:
            tick = time.perf_counter()
            classifier = fit_head(x_fit, y[fit], candidate["C"], candidate["class_weight"], plan["seed"], config.values["classifier"])
            with threadpool_limits(limits=1):
                scores = score_matrix(classifier, x_held, len(labels))
            oof[identity(candidate)][held] = scores
            fit_times[identity(candidate)].append(time.perf_counter() - tick)
        if progress:
            progress({"representation": kind, "completed_cv_fold": fold_number + 1, "folds": len(folds)})
        del x_fit, x_held, classifier
        gc.collect()
    reports = []
    for candidate in candidates:
        key = identity(candidate)
        reports.append({**candidate, "oof_at_fixed_0_5": evaluate_predictions(y, oof[key], np.full(len(labels), 0.5), labels),
                        "fit_and_score_seconds_by_fold": fit_times[key]})
    chosen = choose_candidate(reports)
    selected_scores = oof[identity({"C": chosen["C"], "class_weight": chosen["class_weight"]})]
    thresholds, methods, global_selection = oof_thresholds(y, selected_scores, labels, folds, plan["thresholds"])
    selection_finished = time.perf_counter()
    if kind == "embedding":
        transformer = encoder
        x_train, x_validation = np.asarray(embedding_data[:len(training)]), np.asarray(embedding_data[len(training):])
    else:
        transformer, x_train, x_validation, key = cached_lexical(training, validation, kind, plan, config.path("cache_path"))
    fit_start = time.perf_counter()
    classifier = fit_head(x_train, y, chosen["C"], chosen["class_weight"], plan["seed"], config.values["classifier"])
    full_fit_seconds = time.perf_counter() - fit_start
    inference_start = time.perf_counter()
    with threadpool_limits(limits=1):
        validation_scores = score_matrix(classifier, x_validation, len(labels))
    head_inference_seconds = time.perf_counter() - inference_start
    pipeline = Pipeline([("representation", transformer), ("classifier", classifier)])
    # A bounded real end-to-end probe includes representation cost (including
    # chunked encoder work); its data are already allowed validation inputs.
    tick = time.perf_counter()
    with threadpool_limits(limits=1):
        probe = score_matrix(pipeline, texts(validation[:8]), len(labels))
    probe_seconds = time.perf_counter() - tick
    if not np.allclose(probe, validation_scores[:8], atol=2e-5):
        raise ModelingError("Cached representation and full-pipeline inference disagree")
    provenance = {"source_manifest_sha256": corpus.manifest_sha256, "configuration_sha256": config.fingerprint,
        "taxonomy_sha256": taxonomy.fingerprint, "dependency_versions": dependency_versions(), "seed": plan["seed"],
        "feature_source": "statement-only.jsonl:input.statement", "fit_split": "train", "selection_split": "training_grouped_oof",
        "test_evaluated": False, "modeling_version": "representation-comparison-v2", "plan_sha256": plan_hash}
    bundle = {"pipeline": pipeline, "run_id": kind + "-" + identity(provenance)[:16], "provenance": provenance,
        "label_order": labels, "thresholds": thresholds, "threshold_methods": methods,
        "global_threshold_selection": global_selection, "taxonomy": taxonomy.spec,
        "configuration": {"baseline": config.values, "representation": kind, "experiment_plan": plan},
        "chosen_hyperparameters": {"C": chosen["C"], "class_weight": chosen["class_weight"]},
        "candidate_results": reports, "validation_metrics": evaluate_predictions(y_validation, validation_scores, thresholds, labels),
        "validation_errors": error_examples(validation, y_validation, validation_scores, thresholds, labels),
        "fit_training_ids": [r.problem_id for r in training], "validation_ids": [r.problem_id for r in validation],
        "unmodeled_observed_tags": {"train": unsupported_train, "validation": unsupported_validation},
        "cv": {"fold_receipt_sha256": hash_file(receipt), "feature_cache_keys": fold_cache_keys,
               "absent_or_constant_training_labels_by_fold": absent_training,
               "selected_oof_metrics": evaluate_predictions(y, selected_scores, thresholds, labels)},
        "timings": {"selection_and_representation_seconds": selection_finished - started, "full_training_fit_seconds": full_fit_seconds,
                    "validation_head_inference_seconds": head_inference_seconds, "end_to_end_probe_records": 8,
                    "end_to_end_probe_seconds": probe_seconds, "total_seconds": time.perf_counter() - started},
        "memory": {"before": before, "after": memory_observation(), "peak_semantics": "process-lifetime peak; sampled before/after RSS is also recorded"},
        "matrix": {"shape": list(x_train.shape), "format": "dense_frozen_embedding" if kind == "embedding" else "csr"},
        "score_semantics": "Uncalibrated independent logistic-regression label scores, not calibrated confidence",
        "encoder": encoder_metadata}
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex[:8])
    temporary.mkdir()
    np.save(temporary / "selected-oof-scores.npy", selected_scores)
    np.save(temporary / "validation-scores.npy", validation_scores)
    joblib.dump(bundle, temporary / "pipeline.joblib", compress=3)
    metadata = {k: v for k, v in bundle.items() if k not in {"pipeline", "thresholds", "validation_errors"}}
    metadata["thresholds"] = thresholds.tolist()
    atomic_json(temporary / "run.json", metadata)
    atomic_json(temporary / "validation-errors.json", bundle["validation_errors"])
    atomic_json(temporary / "artifact-manifest.json", {"run_id": bundle["run_id"], "source_manifest_sha256": corpus.manifest_sha256,
        "files": {p.name: hash_file(p) for p in temporary.iterdir() if p.is_file()}})
    temporary.rename(destination)
    return {"representation": kind, "run": str(destination.resolve()), "validation": bundle["validation_metrics"],
            "chosen_hyperparameters": bundle["chosen_hyperparameters"], "timings": bundle["timings"]}


def summarize_comparison(config: Config, corpus: Corpus, plan_path: Path) -> dict:
    plan, output, plan_hash = freeze_plan(config, corpus, plan_path)
    baseline = load_bundle(config.project_root / plan["historical_run"])
    reports = {"historical_baseline_v1": read_json(config.project_root / plan["historical_run"] / "run.json")}
    for kind in plan["representations"]:
        if (output / kind / "run.json").exists():
            reports[kind] = verified_experiment_run(output / kind, plan_hash, corpus.manifest_sha256, kind)
    if "word" not in reports:
        raise ModelingError("The protocol-matched word control has not completed")
    control = reports["word"]["validation_metrics"]
    adoption = {}
    for name, report in reports.items():
        if name in {"word", "historical_baseline_v1"}:
            continue
        metric = report["validation_metrics"]
        delta = {k: metric[k] - control[k] for k in ["micro_f1", "macro_f1"]}
        case_delta = {k: metric["per_label"][k]["f1"] - control["per_label"][k]["f1"] for k in ["dsu", "binary search"]}
        eligible = min(delta.values()) >= 0 and max(delta.values()) >= plan["adoption_rule"]["minimum_improvement_in_one_aggregate"] and min(case_delta.values()) >= -plan["adoption_rule"]["maximum_case_label_f1_regression"]
        adoption[name] = {"meets_predeclared_rule": eligible, "aggregate_deltas_from_word": delta, "case_label_deltas_from_word": case_delta}
    differences = {}
    for name, report in reports.items():
        metric = report["validation_metrics"]
        differences[name] = {label: {"f1_delta_from_word": metric["per_label"][label]["f1"] - control["per_label"][label]["f1"],
            **metric["per_label"][label]} for label in baseline["label_order"]}
    result = {"plan_sha256": plan_hash, "source_manifest_sha256": corpus.manifest_sha256,
        "runs": {n: {"metrics": r["validation_metrics"], "configuration": r["chosen_hyperparameters"],
                      "thresholds": r["thresholds"], "timings": r["timings"], "case_errors": {k: read_json((config.project_root / plan["historical_run"] if n == "historical_baseline_v1" else output / n) / "validation-errors.json")[k] for k in ['dsu','binary search']}}
                 for n, r in reports.items()}, "per_label_changes": differences, "adoption": adoption,
        "test_evaluated": False, "baseline_inventory_unchanged": True}
    atomic_json(output / "comparison.json", result)
    lines = ["# Controlled representation comparison", "", "All new selection/thresholds use grouped training OOF predictions. Validation is comparison only; test is not evaluated.",
        "", "| Run | Micro F1 | Macro F1 | DSU F1 | Binary-search F1 |", "|---|---:|---:|---:|---:|"]
    for name, r in reports.items():
        m = r["validation_metrics"]
        lines.append(f"| {name} | {m['micro_f1']:.6f} | {m['macro_f1']:.6f} | {m['per_label']['dsu']['f1']:.6f} | {m['per_label']['binary search']['f1']:.6f} |")
    lines += ["", "Historical baseline-v1 used validation-selected settings and is not a protocol-matched control.",
              "", "Adoption decisions use the predeclared rule; no default is replaced automatically."]
    for name, decision in adoption.items():
        lines.append(f"- {name}: {'meets' if decision['meets_predeclared_rule'] else 'does not meet'} the development adoption rule.")
    lines += ["", "## Per-label validation results", "", "P/R/F1 are precision, recall and F1 against original tags. Support is the number of observed positives.", ""]
    for name, report in reports.items():
        lines += [f"### {name}", "", "| Label | P | R | F1 | Support | FP | FN | F1 change from new word |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        run_path = config.project_root / plan["historical_run"] if name == "historical_baseline_v1" else output / name
        errors = read_json(run_path / "validation-errors.json")
        for label in baseline["label_order"]:
            m = differences[name][label]
            lines.append(f"| {label} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['support']} | {errors[label]['false_positive_count']} | {errors[label]['false_negative_count']} | {m['f1_delta_from_word']:+.6f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str((output / 'comparison.md').resolve()), "adoption": adoption, "test_evaluated": False}
