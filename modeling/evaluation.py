from __future__ import annotations

import time
from pathlib import Path

from .common import Config, ModelingError, atomic_json, hash_file
from .data import Corpus
from .features import texts
from .inference import load_bundle
from .labels import Taxonomy, encode_targets
from .metrics import error_examples, evaluate_predictions, score_matrix


def evaluate_run(config: Config, corpus: Corpus, run: Path, split="validation", final=False):
    if split not in {"validation", "test"} or (split == "test" and not final):
        raise ModelingError("Test evaluation requires --split test --final on a frozen model")
    bundle = load_bundle(run)
    if bundle["provenance"]["source_manifest_sha256"] != corpus.manifest_sha256:
        raise ModelingError("Model and evaluation corpus differ")
    claim = run / "test-evaluation-claim.json"
    if split == "test":
        if claim.exists() or (run / "test-evaluation.json").exists():
            raise ModelingError("Test evaluation was already invoked for this frozen run; read the saved report")
        # Exclusive creation guards accidental concurrent/repeated final evaluations.
        with claim.open("x", encoding="utf-8") as stream:
            import json
            json.dump({"run_id": bundle["run_id"], "model_sha256": hash_file(run / "pipeline.joblib")}, stream)
    records = corpus.records([split], final_test=split == "test" and final)
    taxonomy = Taxonomy(bundle["taxonomy"])
    targets, unmodeled = encode_targets(records, taxonomy, bundle["label_order"])
    start = time.perf_counter()
    scores = score_matrix(bundle["pipeline"], texts(records), len(bundle["label_order"]))
    seconds = time.perf_counter() - start
    report = {"run_id": bundle["run_id"], "split": split, "source_manifest_sha256": corpus.manifest_sha256,
              "frozen_model_sha256": hash_file(run / "pipeline.joblib"),
              "metrics": evaluate_predictions(targets, scores, bundle["thresholds"], bundle["label_order"]),
              "errors": error_examples(records, targets, scores, bundle["thresholds"], bundle["label_order"]),
              "unmodeled_observed_tags": unmodeled, "inference_seconds": seconds,
              "configuration_changed": False, "thresholds_refitted": False}
    atomic_json(run / f"{split}-evaluation.json", report)
    return {"run": str(run.resolve()), "split": split, "metrics": report["metrics"], "inference_seconds": seconds,
            "thresholds_refitted": False}
