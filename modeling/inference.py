from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .common import ModelingError, dependency_versions, hash_file, read_json
from .labels import Taxonomy
from .metrics import score_matrix


def load_bundle(run: Path) -> dict[str, Any]:
    manifest = read_json(run / "artifact-manifest.json")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or hash_file(run / name) != digest:
            raise ModelingError("Frozen model artifacts failed hash verification")
    metadata = read_json(run / "run.json")
    if metadata["provenance"]["dependency_versions"] != dependency_versions():
        raise ModelingError("Inference dependencies differ from the frozen run; use modeling/requirements.lock")
    bundle = joblib.load(run / "pipeline.joblib")  # Load only locally produced trusted artifacts.
    if (bundle["label_order"] != metadata["label_order"]
            or len(bundle["thresholds"]) != len(bundle["label_order"])
            or bundle["run_id"] != manifest["run_id"]):
        raise ModelingError("Frozen inference pipeline/label metadata disagree")
    return bundle


def predict(bundle: dict[str, Any], statement: str) -> dict[str, Any]:
    if not isinstance(statement, str) or not statement.strip():
        raise ModelingError("Inference requires a nonempty unseen statement string")
    labels = bundle["label_order"]
    with threadpool_limits(limits=1):
        scores = score_matrix(bundle["pipeline"], [statement], len(labels))[0]
    selected = [label for label, score, threshold in zip(labels, scores, bundle["thresholds"]) if score >= threshold]
    taxonomy = Taxonomy(bundle["taxonomy"])
    return {"run_id": bundle["run_id"], "selected_labels": selected,
            "mapped_broad_families": list(taxonomy.broad_families(selected)),
            "label_scores": {label: float(score) for label, score in zip(labels, scores)},
            "thresholds": {label: float(t) for label, t in zip(labels, bundle["thresholds"])},
            "score_semantics": bundle["score_semantics"], "fine_annotations": "unknown_unreviewed"}
