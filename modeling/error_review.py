from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from .common import ModelingError, atomic_json, hash_file, identity, read_json
from .features import texts
from .implementation_audit import freeze_plan
from .inference import load_bundle
from .labels import Taxonomy, encode_targets
from .metrics import score_matrix

REVIEW_VERSION = "automated-source-evidence-v1"
PATTERNS = {
    "dsu": {"editorial": r"\bdsu\b|disjoint[- ]set|union[- ]find",
            "code": r"\b(?:dsu|uf|ufds)\s*\.\s*(?:merge|unite|join|same|find)|\b(?:unite|union_sets|merge_sets|unionSet)\s*\("},
    "binary search": {"editorial": r"binary[- ]search|\bbisect",
                      "code": r"\b(?:lower_bound|upper_bound|bisect_left|bisect_right)\s*\(|while\s*\(\s*(?:l|lo|left|low)\s*(?:\+\s*1\s*)?<\s*=?\s*(?:r|hi|right|high)\s*\)"}
}


def validate_review(report: dict) -> None:
    schema = read_json(Path(__file__).parent / "schemas" / "error-review-v1.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda e: str(e.path))
    if errors:
        raise ModelingError("Invalid automated error review: " + errors[0].message)
    cases = report["cases"]
    if report["case_count"] != len(cases) or len({c["case_id"] for c in cases}) != len(cases):
        raise ModelingError("Review case count/identities disagree")
    if report["unique_problems"] != len({c["problem_id"] for c in cases}):
        raise ModelingError("Review problem count disagrees")


def validate_source_signals(case: dict, row: dict) -> None:
    sources = {("editorial", row["editorial_version_id"]): row["input"]["editorial"]}
    sources.update({("accepted_code", s["version_id"]): s["code"] for s in row["solutions"]})
    if case["problem_id"] != row["problem_id"]:
        raise ModelingError("Review source belongs to another problem")
    for signal in case["supporting_signals"]:
        source = sources.get((signal["source_kind"], signal["version_id"]))
        if (source is None or not 0 <= signal["start"] < signal["end"] <= len(source)
                or source[signal["start"]:signal["end"]] != signal["quote"]
                or signal["match"] not in signal["quote"]):
            raise ModelingError("Review evidence quote/offset/version does not match its pinned source")


def select_cases(records, y, scores, labels, thresholds, specification, seed):
    """Deterministic observed-tag strata; no semantic label changes."""
    selected = []
    for label in specification["labels"]:
        j = labels.index(label)
        for bucket in specification["buckets"]:
            positive = bucket in {"true_positive", "false_negative"}
            predicted = bucket in {"true_positive", "false_positive"}
            candidates = [i for i, record in enumerate(records) if bool(y[i, j]) == positive and bool(scores[i, j] >= thresholds[j]) == predicted]
            candidates.sort(key=lambda i: identity([seed, REVIEW_VERSION, label, bucket, records[i].problem_id]))
            selected.extend((label, bucket, i) for i in candidates[:specification["cases_per_label_and_bucket"]])
    return selected


def evidence_signal(text: str, pattern: str):
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    start, end = max(0, match.start() - 100), min(len(text), match.end() + 180)
    return {"quote": text[start:end], "start": start, "end": end, "match": match.group(0)}


def code_signal(text: str, label: str):
    # Preserve offsets while excluding comments and preprocessor templates.
    # A template macro or ordered-set member lookup is not corroboration of
    # binary search as a problem-solving approach.
    searchable = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*|^[ \t]*\#[^\n]*", lambda m: " " * len(m.group()), text, flags=re.M)
    pattern = PATTERNS[label]["code"]
    if label == "binary search":
        pattern = pattern.replace(r"\b(?:lower_bound", r"(?<![.\w])(?:lower_bound")
    match = re.search(pattern, searchable, re.I)
    if not match:
        return None
    start, end = max(0, match.start() - 100), min(len(text), match.end() + 180)
    return {"quote": text[start:end], "start": start, "end": end, "match": text[match.start():match.end()],
            "interpretation": "candidate implementation signal; execution and algorithm suitability are not established"}


def propose_outcome(bucket: str, editorial_signal, code_signals, ambiguity=False):
    if ambiguity:
        return "unresolved", "The wording may refer to a different DSU variant or a negated approach; human interpretation is needed."
    if bucket == "false_negative" and (editorial_signal or code_signals):
        return "model_error", "Observed-tag miss is corroborated by source signals; semantic review remains automated."
    if bucket in {"false_positive", "true_negative"} and editorial_signal and code_signals:
        return "potentially_incomplete_tagging", "An unobserved tag has editorial and accepted-code signals; this is a tagging-review candidate, not a new label."
    if bucket == "false_positive" and code_signals:
        return "supported_alternative_approach", "An accepted source contains a candidate implementation signal; intended/alternative status still needs human confirmation."
    return "unresolved", "Observed-tag correctness/error is known, but semantic suitability or exclusivity is not established by these rules."


def review_errors(config, corpus, plan_path):
    plan, output, fingerprint = freeze_plan(config, corpus, plan_path)
    baseline_path = config.project_root / plan["historical_run"]
    bundle = load_bundle(baseline_path)
    records = corpus.records(["validation"])
    labels = bundle["label_order"]
    y, _ = encode_targets(records, Taxonomy(bundle["taxonomy"]), labels)
    scores = score_matrix(bundle["pipeline"], texts(records), len(labels))
    selected = select_cases(records, y, scores, labels, bundle["thresholds"], plan["review"], plan["seed"])
    wanted = {records[i].problem_id for _, _, i in selected}
    evidence_rows = {}
    with (corpus.directory / "editorial-enriched.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["problem_id"] not in wanted:
                continue
            pid = row["problem_id"]
            info = corpus.structural[pid]
            if (row["split"] != "validation" or info["split"] != "validation"
                    or row["editorial_version_id"] != info["editorial_version_id"]
                    or row["statement_version_id"] != info["version_id"] or row["clean_sha256"] != info["clean_sha256"]):
                raise ModelingError("Error-review sources are not pinned validation evidence")
            evidence_rows[pid] = row
    if set(evidence_rows) != wanted:
        raise ModelingError("A selected review case is missing its source bundle")
    cases = []
    for label, bucket, i in selected:
        record, row = records[i], evidence_rows[records[i].problem_id]
        editorial_signal = evidence_signal(row["input"]["editorial"], PATTERNS[label]["editorial"])
        code_signals = []
        for source in row["solutions"]:
            signal = code_signal(source["code"], label)
            if signal:
                code_signals.append({"source_kind": "accepted_code", "version_id": source["version_id"],
                    "source_document_id": source["source_document_id"], "raw_sha256": source["raw_sha256"],
                    "clean_sha256": source["clean_sha256"], "programming_language": source["programming_language"], **signal})
        ambiguity = bool(editorial_signal and (
            re.search(r"dsu on (?:a )?tree|small.to.large", editorial_signal["quote"], re.I) if label == "dsu" else
            re.search(r"(?:cannot|can't|not|without).{0,25}binary[- ]search", editorial_signal["quote"], re.I)))
        outcome, rationale = propose_outcome(bucket, editorial_signal, code_signals, ambiguity)
        source_signals = list(code_signals)
        if editorial_signal:
            source_signals.insert(0, {"source_kind": "editorial", "version_id": row["editorial_version_id"],
                                      "editorial_link_id": row["editorial_link_id"], **editorial_signal})
        decisions = {name: {"score": float(scores[i, labels.index(name)]), "threshold": float(bundle["thresholds"][labels.index(name)]),
                           "selected": bool(scores[i, labels.index(name)] >= bundle["thresholds"][labels.index(name)])}
                     for name in plan["review"]["labels"]}
        cases.append({"case_id": identity([REVIEW_VERSION, label, bucket, record.problem_id]), "problem_id": record.problem_id,
            "title": record.title, "split": "validation", "focus_label": label, "observed_tag_bucket": bucket,
            "original_tags": list(record.original_tags), "predictions": decisions,
            "source_references": {"statement_version_id": row["statement_version_id"], "statement_clean_sha256": row["clean_sha256"],
                "editorial_version_id": row["editorial_version_id"], "editorial_link_id": row["editorial_link_id"],
                "accepted_solution_versions": [s["version_id"] for s in row["solutions"]]},
            "inspected": {"statement_characters": len(row["input"]["statement"]), "editorial_characters": len(row["input"]["editorial"]),
                          "accepted_sources": len(row["solutions"]), "accepted_code_characters": sum(len(s["code"]) for s in row["solutions"])},
            "statement_excerpt": row["input"]["statement"][:400], "editorial_excerpt": row["input"]["editorial"][:400],
            "supporting_signals": source_signals, "review_outcome": outcome, "outcome_rationale": rationale,
            "review_method": REVIEW_VERSION, "review_completed_by": "automated_rules", "human_reviewed": False,
            "outcome_status": "automated_proposal_requires_human_confirmation", "original_labels_modified": False})
    report = {"schema_version": "error-review-v1", "plan_fingerprint": fingerprint, "source_manifest_sha256": corpus.manifest_sha256,
        "baseline_model_sha256": hash_file(baseline_path / "pipeline.joblib"), "selection": "seeded hash, six per label and observed-tag bucket",
        "cases": cases, "case_count": len(cases), "unique_problems": len(wanted),
        "outcomes": dict(Counter(c["review_outcome"] for c in cases)), "human_reviewed_cases": 0,
        "caveat": "Rules inspect complete selected source strings but are not proof of algorithm suitability; original-tag metrics remain unchanged."}
    validate_review(report)
    for case in cases:
        validate_source_signals(case, evidence_rows[case["problem_id"]])
    atomic_json(output / "error-review-v1.json", report)
    lines = ["# Evidence-based automated error review v1", "", f"{len(cases)} label/case pairs from {len(wanted)} validation problems. No test problems. No human-reviewed annotations.",
        "", "Outcomes are automated proposals; absence of an upstream tag is not proof of unsuitability. Editorial/code strings were inspected for every selected case; exact source version references and matching quotes are saved in JSON.",
        "", "| Label | Bucket | Problem | Outcome proposal | Source signals |", "|---|---|---|---|---:|"]
    for case in cases:
        lines.append(f"| {case['focus_label']} | {case['observed_tag_bucket']} | {case['problem_id']} | {case['review_outcome']} | {len(case['supporting_signals'])} |")
    lines += ["", "Counts: " + str(report["outcomes"]), "", "Correct observed-tag predictions may remain semantically unresolved. Code template matches, negated editorial mentions and DSU-on-tree variants require human interpretation. Do not turn these proposals into gold labels automatically."]
    (output / "error-review-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str((output / 'error-review-v1.md').resolve()), "case_count": len(cases), "unique_problems": len(wanted),
            "outcomes": report["outcomes"], "human_reviewed_cases": 0}
