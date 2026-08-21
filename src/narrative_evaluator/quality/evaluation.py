"""Metrics for pairwise preference, calibration, grades, and generalization."""
from __future__ import annotations

import math
from typing import Iterable


def evaluate_pair_probabilities(
    gold_probability_a: Iterable[float],
    predicted_probability_a: Iterable[float],
) -> dict[str, float | int]:
    gold = [float(v) for v in gold_probability_a]
    pred = [float(v) for v in predicted_probability_a]
    if not gold or len(gold) != len(pred):
        raise ValueError("gold and predicted probabilities must have equal non-zero length")
    eps = 1e-12
    brier = sum((p - y) ** 2 for y, p in zip(gold, pred)) / len(gold)
    log_loss = -sum(
        y * math.log(max(eps, min(1 - eps, p)))
        + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
        for y, p in zip(gold, pred)
    ) / len(gold)
    decisive = [(y, p) for y, p in zip(gold, pred) if abs(y - 0.5) >= 0.125]
    accuracy = (
        sum((p >= 0.5) == (y > 0.5) for y, p in decisive) / len(decisive)
        if decisive else float("nan")
    )
    return {
        "n": len(gold),
        "n_decisive": len(decisive),
        "pairwise_accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
    }
