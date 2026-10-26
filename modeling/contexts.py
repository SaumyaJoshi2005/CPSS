from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from .common import Config, ModelingError, atomic_json, identity, read_json
from .data import Corpus
from .labels import Taxonomy, select_labels

SCHEMA_PATH = Path(__file__).parent / "schemas" / "context-v1.schema.json"


def pending_problem(problem_id: str, taxonomy: Taxonomy, manifest_sha256: str) -> dict[str, Any]:
    return {"schema_version": "problem-context-v1", "problem_id": problem_id, "taxonomy_version": taxonomy.spec["version"],
            "status": "pending_generation", "abstract_objective": None, "structures": [], "operations": [],
            "supported_approaches": [], "key_observations": [], "applicability_conditions": [], "unknowns": [],
            "provenance": {"kind": "corpus", "source_manifest_sha256": manifest_sha256, "generation_method": "pending",
                           "model_revision": None, "prompt_version": None},
            "review": {"decision": "pending", "reviewer": None, "reviewed_at": None}}


def validate_context(document: dict[str, Any], taxonomy: Taxonomy, corpus: Corpus | None = None,
                     illustrative_sources: dict[str, Any] | None = None) -> None:
    schema = read_json(SCHEMA_PATH)
    try:
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    except ValidationError as exc:
        raise ModelingError("Invalid context schema: " + ".".join(map(str, exc.absolute_path))) from exc
    if document["taxonomy_version"] != taxonomy.spec["version"]:
        raise ModelingError("Context taxonomy version mismatch")
    reviewed = document["review"]["decision"] == "approved"
    if reviewed != (document["status"] == "human_reviewed"):
        raise ModelingError("Human-reviewed status requires an explicit approval decision")
    if (document["status"] == "rejected") != (document["review"]["decision"] == "rejected"):
        raise ModelingError("Rejected status and decision must agree")
    if document["provenance"]["kind"] == "illustrative" and document["status"] == "human_reviewed":
        raise ModelingError("Illustrations cannot be promoted to reviewed corpus annotations")
    approaches = document.get("supported_approaches", document.get("inferred_approaches", []))
    if any(a["algorithm_id"] not in taxonomy.algorithm_ids for a in approaches):
        raise ModelingError("Unknown controlled algorithm vocabulary ID")
    if document["schema_version"] == "query-context-v1":
        for approach in approaches:
            if any(span not in document["original_text"] for span in approach["evidence_spans"]):
                raise ModelingError("Query evidence span is absent from the original query")
        return
    sources = None
    if document["provenance"]["kind"] == "corpus":
        if corpus is None or document["provenance"]["source_manifest_sha256"] != corpus.manifest_sha256:
            raise ModelingError("Context requires the matching frozen corpus")
        info = corpus.structural.get(document["problem_id"])
        if not info or info["split"] != "train":
            raise ModelingError("Problem context pilot is training-only")
        if document["status"] != "pending_generation":
            sources = corpus.context_sources(document["problem_id"])
    elif document["provenance"]["kind"] == "illustrative":
        sources = (illustrative_sources or {}).get(document["problem_id"])
        if not sources:
            raise ModelingError("Illustrative evidence requires its explicitly synthetic sources")
    else:
        raise ModelingError("Invalid problem-context provenance kind")
    evidence = [e for a in approaches for e in a["evidence"]]
    evidence += [e for observation in document["key_observations"] for e in observation["evidence"]]
    for item in evidence:
        if not sources or item["source_kind"] not in sources:
            raise ModelingError("Missing context evidence source")
        source = sources[item["source_kind"]]
        if item["version_id"] != source["version_id"] or item["quote"] not in source["text"]:
            raise ModelingError("Context evidence quote/version mismatch")


def prepare_pilot(config: Config, corpus: Corpus) -> dict[str, Any]:
    taxonomy = Taxonomy.load(config.path("taxonomy_path"))
    records = corpus.records(["train"])
    selection = select_labels(records, taxonomy, config.values["labels"]["min_train_positive"], config.values["labels"]["min_train_negative"])
    normalized = {r.problem_id: taxonomy.normalize(r.original_tags) for r in records}
    ordered = sorted(records, key=lambda r: identity([config.values["seed"], "context-pilot-v1", r.problem_id]))
    chosen: dict[str, Any] = {}
    pilot_config = config.values["context_pilot"]
    for label in pilot_config["case_studies"]:
        for record in [r for r in ordered if label in normalized[r.problem_id]][:pilot_config["case_study_minimum"]]:
            chosen[record.problem_id] = record
    # Round-robin coverage, then a deterministic fill. Selection uses observed
    # training tags only, never validation/test content or model predictions.
    pools = {label: [r for r in ordered if label in normalized[r.problem_id]] for label in selection["label_order"]}
    target = min(pilot_config["size"], len(records))
    if len(chosen) > target:
        raise ModelingError("Pilot size is too small for the requested case-study coverage")
    while len(chosen) < target:
        before = len(chosen)
        for label, pool in pools.items():
            candidate = next((r for r in pool if r.problem_id not in chosen), None)
            if candidate and len(chosen) < target:
                chosen[candidate.problem_id] = candidate
        if len(chosen) == before:
            break
    for record in ordered:
        if len(chosen) >= target:
            break
        chosen.setdefault(record.problem_id, record)
    output = config.path("artifacts_path") / "context-pilot"
    output.mkdir(parents=True, exist_ok=True)
    references, drafts = [], []
    for record in chosen.values():
        info = corpus.structural[record.problem_id]
        references.append({"problem_id": record.problem_id, "split": "train", "statement_version_id": record.version_id,
            "clean_sha256": record.clean_sha256, "editorial_version_id": info["editorial_version_id"],
            "editorial_link_id": info["editorial_link_id"], "observed_original_tags": list(record.original_tags),
            "normalized_tags": list(normalized[record.problem_id]), "fine_annotations": "unknown_unreviewed"})
        draft = pending_problem(record.problem_id, taxonomy, corpus.manifest_sha256)
        validate_context(draft, taxonomy, corpus)
        drafts.append(draft)
    atomic_json(output / "sample.json", {"sample_version": "context-pilot-v1", "source_manifest_sha256": corpus.manifest_sha256,
        "taxonomy_sha256": taxonomy.fingerprint, "taxonomy_version": taxonomy.spec["version"], "seed": config.values["seed"],
        "selection": "training-only hash ordering with case-study quota and round-robin selected-label coverage",
        "size": len(references), "label_coverage": dict(Counter(t for r in references for t in r["normalized_tags"])),
        "records": references, "bulk_generation_performed": False})
    temporary = output / "pending-contexts.jsonl.tmp"
    from cpss.core import canonical
    with temporary.open("wb") as stream:
        for draft in drafts:
            stream.write(canonical(draft) + b"\n")
    temporary.replace(output / "pending-contexts.jsonl")
    return {"sample": str((output / "sample.json").resolve()), "size": len(references),
            "label_coverage": dict(Counter(t for r in references for t in r["normalized_tags"])),
            "bulk_generation_performed": False, "human_reviewed_annotations": 0}
