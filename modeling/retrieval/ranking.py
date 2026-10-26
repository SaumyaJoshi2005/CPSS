from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..common import ModelingError


def ranked_ids(scores: np.ndarray, ids: list[str], eligible: np.ndarray, limit: int, positive_only=False) -> list[str]:
    if scores.shape != (len(ids),) or eligible.shape != scores.shape or not np.isfinite(scores).all():
        raise ModelingError("Ranking score/filter dimensions are invalid")
    positions = [i for i in range(len(ids)) if eligible[i] and (not positive_only or scores[i] > 0)]
    positions.sort(key=lambda i: (-float(scores[i]), ids[i]))
    return list(dict.fromkeys(ids[i] for i in positions))[:limit]


def reciprocal_rank_fusion(rankings: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    if k < 1:
        raise ModelingError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(dict.fromkeys(ranking), 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def boosted_score(base: float, explicit_fraction: float, classifier_signal: float,
                  explicit_weight: float, classifier_weight: float) -> float:
    # Additive relative to magnitude also gives correctly directed soft boosts
    # when a cosine similarity is negative. No candidate is removed by a model.
    return base + abs(base) * (explicit_weight * explicit_fraction + classifier_weight * classifier_signal)
