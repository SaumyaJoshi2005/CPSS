from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .common import ModelingError
from .data import Record


def score_matrix(classifier, matrix, label_count: int) -> np.ndarray:
    scores = np.asarray(classifier.predict_proba(matrix), dtype=float)
    rows = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
    if label_count == 1 and scores.shape == (rows, 2):
        scores = scores[:, 1:2]
    if scores.shape != (rows, label_count) or not np.isfinite(scores).all():
        raise ModelingError("Classifier scores do not match the frozen label order")
    return scores


def best_global_threshold(y: np.ndarray, scores: np.ndarray, grid: list[float]) -> tuple[float, list[dict[str, float]]]:
    candidates = [{"threshold": threshold, "micro_f1": float(f1_score(y, scores >= threshold, average="micro", zero_division=0)),
                   "macro_f1": float(f1_score(y, scores >= threshold, average="macro", zero_division=0))} for threshold in grid]
    best = max(candidates, key=lambda r: (r["micro_f1"], r["macro_f1"], -abs(r["threshold"] - 0.5), r["threshold"]))
    return best["threshold"], candidates


def select_thresholds(y, scores, labels, config, global_threshold):
    thresholds, methods = [], {}
    for index, label in enumerate(labels):
        positives, negatives = int(y[:, index].sum()), int(len(y) - y[:, index].sum())
        if (config["mode"] == "per_label" and positives >= config["min_validation_positive"]
                and negatives >= config["min_validation_negative"]):
            candidates = [(float(f1_score(y[:, index], scores[:, index] >= t, zero_division=0)), t) for t in config["grid"]]
            _, threshold = max(candidates, key=lambda item: (item[0], -abs(item[1] - global_threshold), item[1]))
            method = "validation_per_label_f1"
        else:
            threshold = global_threshold
            method = "global_fallback_insufficient_validation_support" if config["mode"] == "per_label" else "validation_global_micro_f1"
        thresholds.append(threshold)
        methods[label] = {"threshold": threshold, "method": method, "validation_positive": positives, "validation_negative": negatives}
    return np.asarray(thresholds), methods


def evaluate_predictions(y, scores, thresholds, labels) -> dict[str, Any]:
    if list(scores.shape) != list(y.shape) or len(thresholds) != len(labels):
        raise ModelingError("Evaluation label/score dimensions disagree")
    predicted = scores >= np.asarray(thresholds)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, average=None, zero_division=0)
    return {"micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "per_label": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                               "support": int(support[i]), "predicted_positive": int(predicted[:, i].sum())}
                          for i, name in enumerate(labels)},
            "records": len(y), "label_count": len(labels),
            "target_semantics": "Observed upstream tag assignments; missing tags are not exhaustive negatives for algorithm suitability"}


def error_examples(records: list[Record], y, scores, thresholds, labels, per_kind=4) -> dict[str, Any]:
    predictions = scores >= thresholds
    report = {}
    for index, label in enumerate(labels):
        errors = {}
        for kind, mask in (("false_positive", (predictions[:, index] == 1) & (y[:, index] == 0)),
                           ("false_negative", (predictions[:, index] == 0) & (y[:, index] == 1))):
            positions = np.flatnonzero(mask).tolist()
            positions.sort(key=lambda i: (-abs(float(scores[i, index]) - thresholds[index]), records[i].problem_id))
            errors[kind] = [{"problem_id": records[i].problem_id, "title": records[i].title,
                            "score": float(scores[i, index]), "threshold": float(thresholds[index]),
                            "observed_tags": list(records[i].original_tags), "statement_excerpt": records[i].statement[:400],
                            "review_status": "unreviewed_observed_tag_error"} for i in positions[:per_kind]]
            errors[kind + "_count"] = len(positions)
        report[label] = errors
    return report
