from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

from .common import ModelingError, identity, read_json
from .data import Corpus, Record


def training_groups(records: list[Record], corpus: Corpus) -> list[str]:
    if not records or any(r.split != "train" for r in records):
        raise ModelingError("Cross-validation grouping is training-only")
    parent = {r.problem_id: r.problem_id for r in records}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def union(a, b):
        parent[find(b)] = find(a)
    shared = {}
    # Read only structural fields/IDs from the pinned manifest. No test targets
    # or text statistics are used to form the training folds.
    manifest = read_json(corpus.directory / "manifest.json")
    metadata = {r["problem_id"]: r["metadata"] for r in manifest["records"] if r["problem_id"] in parent}
    for r in records:
        info = corpus.structural[r.problem_id]
        keys = [("contest", info["contest_id"]), ("clean", r.clean_sha256)]
        if re.search(r"\((?:easy|hard|simple|medium) version\)", r.title, re.I):
            title = re.sub(r"\s*\((?:easy|hard|simple|medium) version\)\s*", "", r.title, flags=re.I).casefold().strip()
            keys.append(("variant", title, metadata[r.problem_id]["contest"]["started_at"][:10]))
        for key in keys:
            if key in shared:
                union(r.problem_id, shared[key])
            else:
                shared[key] = r.problem_id
        for alias in info["aliases"]:
            alias_id = "codeforces:" + alias.replace("/", ":")
            if alias_id in parent:
                union(r.problem_id, alias_id)
    for group in manifest["configuration"].get("duplicate_groups", []):
        chosen = [p for p in group if p in parent]
        for p in chosen[1:]:
            union(chosen[0], p)
    components = defaultdict(list)
    for p in parent:
        components[find(p)].append(p)
    names = {member: identity(sorted(members)) for members in components.values() for member in members}
    return [names[r.problem_id] for r in records]


def grouped_folds(records: list[Record], groups: list[str], count: int, seed: int):
    if len(records) != len(groups) or len(set(groups)) < count or any(r.split != "train" for r in records):
        raise ModelingError("Invalid grouped training-fold specification")
    folds = list(GroupKFold(n_splits=count, shuffle=True, random_state=seed).split(np.zeros(len(records)), groups=groups))
    seen = np.zeros(len(records), dtype=int)
    for training, held in folds:
        if set(groups[i] for i in training) & set(groups[i] for i in held):
            raise ModelingError("A contest/duplicate group crosses a fold boundary")
        seen[held] += 1
    if not np.all(seen == 1):
        raise ModelingError("Every training record must have exactly one held-out prediction")
    return folds
