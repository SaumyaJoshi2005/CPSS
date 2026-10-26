from __future__ import annotations

import itertools
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import Config, atomic_json, identity
from .data import Corpus
from .labels import Taxonomy, select_labels


def distribution(values: list[int]) -> dict[str, float]:
    array = np.asarray(values)
    return {"minimum": int(array.min()), "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)), "maximum": int(array.max()), "mean": float(array.mean())}


def audit_labels(config: Config, corpus: Corpus) -> dict[str, Any]:
    records = corpus.records(["train"])
    taxonomy = Taxonomy.load(config.path("taxonomy_path"))
    selection = select_labels(records, taxonomy, config.values["labels"]["min_train_positive"], config.values["labels"]["min_train_negative"])
    normalized = {r.problem_id: taxonomy.normalize(r.original_tags) for r in records}
    original = Counter(tag for r in records for tag in set(r.original_tags))
    normalized_counts = Counter(t for tags in normalized.values() for t in tags)
    pairs = Counter(pair for tags in normalized.values() for pair in itertools.combinations(tags, 2))
    word_lengths = [len(re.findall(r"\S+", r.statement)) for r in records]
    chars = [len(r.statement) for r in records]
    analyzer_lengths = [len(re.findall(r"(?u)\b\w+\b", r.statement)) for r in records]
    budgets = config.values["audit"]["future_token_budgets"]
    examples = {}
    for label in selection["label_order"]:
        matches = [r for r in records if label in normalized[r.problem_id]]
        matches.sort(key=lambda r: identity([config.values["seed"], label, r.problem_id]))
        examples[label] = [{"problem_id": r.problem_id, "title": r.title, "original_tags": list(r.original_tags),
                            "statement_excerpt": r.statement[:config.values["audit"]["excerpt_characters"]],
                            "version_id": r.version_id} for r in matches[:config.values["audit"]["representative_examples_per_label"]]]
    contrasts = {}
    for label, related in (("dsu", {"graphs", "trees", "dfs and similar", "data structures"}),
                           ("binary search", {"sortings", "two pointers", "greedy", "ternary search"})):
        candidates = [r for r in records if label not in normalized[r.problem_id] and set(normalized[r.problem_id]) & related]
        candidates.sort(key=lambda r: identity([config.values["seed"], "contrast", label, r.problem_id]))
        contrasts[label] = [{"problem_id": r.problem_id, "title": r.title, "observed_tags": list(r.original_tags),
                            "statement_excerpt": r.statement[:200], "status": "candidate_contrast_unreviewed_not_a_proven_negative"}
                           for r in candidates[:5]]
    report = {"source_manifest_sha256": corpus.manifest_sha256, "configuration_sha256": config.fingerprint,
              "split_counts_structural_only": corpus.split_counts,
              "identity_checks": {"canonical_ids": True, "contest_groups": True, "exact_duplicates": True,
                                  "aliases": True, "upstream_holdout_preserved": True},
              "decision_statistics_split": "train", "test_content_statistics_computed": False,
              "original_tag_frequencies_train": dict(original.most_common()),
              "normalized_tag_frequencies_train": dict(normalized_counts.most_common()),
              "label_selection": selection,
              "imbalance": {tag: {"positive": normalized_counts[tag], "negative": len(records) - normalized_counts[tag],
                                   "negative_to_positive": (len(records) - normalized_counts[tag]) / normalized_counts[tag]}
                            for tag in selection["label_order"]},
              "co_occurrence_train": [{"labels": list(pair), "count": count} for pair, count in pairs.most_common()],
              "empty_target_training_ids": [r.problem_id for r in records if not r.original_tags],
              "duplicate_tag_assignment_training_ids": [r.problem_id for r in records if len(r.original_tags) != len(set(r.original_tags))],
              "malformed_targets": [], "malformed_target_policy": "Fail explicitly at loading; no malformed records silently discarded",
              "unsupported_training_tags": {t: c for t, c in normalized_counts.items() if t not in taxonomy.spec["tags"]},
              "statement_lengths_train": {"characters": distribution(chars), "whitespace_tokens": distribution(word_lengths),
                                          "word_analyzer_tokens": distribution(analyzer_lengths)},
              "future_encoder_length_proxies_train": {str(b): {"whitespace_token_exceedances": sum(x > b for x in word_lengths),
                    "fraction": sum(x > b for x in word_lengths) / len(records)} for b in budgets},
              "length_caveat": "Whitespace/word counts are proxies, not any encoder's subword tokenizer. Exact truncation requires that encoder's tokenizer; choose budgets on training data only.",
              "representative_training_examples": examples, "case_study_candidate_contrasts_train": contrasts,
              "fine_label_state": "unreviewed; no editorial-derived fine labels assigned",
              "target_caveat": taxonomy.spec["semantics"]}
    output = config.path("artifacts_path") / "audit"
    atomic_json(output / "label-audit.json", report)
    lines = ["# Modeling readiness and label audit", "", f"Frozen manifest: `{corpus.manifest_sha256}`.",
             "", f"Structural counts: {corpus.split_counts}. All content/target statistics below use **training only**.",
             "", "Canonical identities, contests, exact duplicates, known aliases and upstream holdout assignments passed structural checks.",
             "", f"Selected **{len(selection['label_order'])}** observed-tag targets with at least {selection['minimum_positive']} training positives and {selection['minimum_negative']} negatives.",
             "", "| Original tag | Training support | Selected |", "|---|---:|---|"]
    for tag, support in original.most_common():
        lines.append(f"| {tag} | {support} | {'yes' if tag in selection['label_order'] else 'no'} |")
    lines += ["", "## Training length distributions", "", "| Measure | Median | P95 | Maximum |", "|---|---:|---:|---:|"]
    for measure, values in report["statement_lengths_train"].items():
        lines.append(f"| {measure} | {values['median']:.0f} | {values['p95']:.0f} | {values['maximum']} |")
    lines += ["", report["length_caveat"], "", "## Most frequent tag pairs (training)", ""]
    for row in report["co_occurrence_train"][:15]:
        lines.append(f"- {' + '.join(row['labels'])}: {row['count']}")
    lines += ["", "## Representative observed-tag positives (training)", ""]
    for label, rows in examples.items():
        lines.append(f"- **{label}:** " + "; ".join(f"`{r['problem_id']}` ({r['title']})" for r in rows))
    lines += ["", "Candidate case-study contrasts and excerpts are in label-audit.json. They are unreviewed and must not be treated as proven algorithm negatives.",
              "", f"Empty training targets: {len(report['empty_target_training_ids'])}. Unsupported training tags: {report['unsupported_training_tags']}.",
              "Malformed targets cause an explicit loader error. Rare/absent labels and their reasons are in label_selection.excluded.",
              "", taxonomy.spec["semantics"], "No test text, length distribution or tag frequencies were used."]
    (output / "label-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str((output / "label-audit.md").resolve()), "selected_labels": selection["label_order"],
            "training_records": len(records), "statement_lengths_train": report["statement_lengths_train"],
            "unsupported_training_tags": report["unsupported_training_tags"]}
