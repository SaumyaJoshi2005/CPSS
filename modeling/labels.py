from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import ModelingError, identity, read_json
from .data import Record


@dataclass(frozen=True)
class Taxonomy:
    spec: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        spec = read_json(path)
        if not isinstance(spec.get("version"), str) or not spec.get("tags"):
            raise ModelingError("Malformed versioned taxonomy")
        families, tags, aliases = spec["families"], spec["tags"], spec["aliases"]
        node_ids = set(families)
        for name, definition in tags.items():
            if name != name.strip().casefold() or not definition["parents"]:
                raise ModelingError("Taxonomy tags must be normalized with explicit parents")
            if not set(definition["parents"]) <= set(families):
                raise ModelingError("Unknown taxonomy parent")
            if definition["algorithm_id"] in node_ids:
                raise ModelingError("Duplicate controlled vocabulary ID")
            node_ids.add(definition["algorithm_id"])
        if any(target not in tags or alias in tags for alias, target in aliases.items()):
            raise ModelingError("Invalid or shadowing taxonomy alias")
        for key, definition in spec.get("future_fine_labels", {}).items():
            if key in node_ids or definition["status"] != "unreviewed" or not set(definition["parents"]) <= set(families):
                raise ModelingError("Fine labels must be distinct and explicitly unreviewed")
            node_ids.add(key)
        return cls(spec)

    @property
    def fingerprint(self) -> str:
        return identity(self.spec)

    @property
    def algorithm_ids(self) -> list[str]:
        return sorted([d["algorithm_id"] for d in self.spec["tags"].values()] + list(self.spec.get("future_fine_labels", {})))

    def normalize(self, original: Iterable[str]) -> tuple[str, ...]:
        labels = set()
        for value in original:
            if not isinstance(value, str) or not value.strip():
                raise ModelingError("Target labels must be nonempty strings")
            tag = " ".join(value.casefold().split())
            labels.add(self.spec["aliases"].get(tag, tag))
        return tuple(sorted(labels))

    def broad_families(self, original: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted({p for tag in self.normalize(original) if tag in self.spec["tags"]
                             for p in self.spec["tags"][tag]["parents"]}))


def select_labels(records: list[Record], taxonomy: Taxonomy, minimum_positive: int, minimum_negative: int) -> dict[str, Any]:
    if not records or any(r.split != "train" for r in records):
        raise ModelingError("Label selection requires training records only")
    counts = Counter(tag for r in records for tag in taxonomy.normalize(r.original_tags))
    selected, excluded = [], {}
    for tag in sorted(set(taxonomy.spec["tags"]) | set(counts)):
        positive = counts[tag]
        if tag not in taxonomy.spec["tags"]:
            reason = "unsupported_original_tag"
        elif positive == 0:
            reason = "absent_from_training"
        elif positive < minimum_positive:
            reason = "insufficient_training_positives"
        elif len(records) - positive < minimum_negative:
            reason = "insufficient_training_negatives"
        else:
            selected.append(tag)
            continue
        excluded[tag] = {"reason": reason, "training_support": positive}
    if not selected:
        raise ModelingError("No labels meet training support requirements")
    return {"label_order": selected, "excluded": excluded, "training_support": dict(sorted(counts.items())),
            "selection_split": "train", "training_records": len(records), "taxonomy_version": taxonomy.spec["version"],
            "taxonomy_sha256": taxonomy.fingerprint, "minimum_positive": minimum_positive, "minimum_negative": minimum_negative}


def encode_targets(records: list[Record], taxonomy: Taxonomy, label_order: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    if len(label_order) != len(set(label_order)) or not label_order or any(t not in taxonomy.spec["tags"] for t in label_order):
        raise ModelingError("Invalid label ordering")
    positions = {name: i for i, name in enumerate(label_order)}
    targets = np.zeros((len(records), len(label_order)), dtype=np.int8)
    unmodeled: Counter[str] = Counter()
    for i, record in enumerate(records):
        for label in taxonomy.normalize(record.original_tags):
            if label in positions:
                targets[i, positions[label]] = 1
            else:
                unmodeled[label] += 1
    return targets, dict(sorted(unmodeled.items()))
