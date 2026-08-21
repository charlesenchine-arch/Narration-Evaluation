"""Aggregate ordinal pairwise judgments without erasing disagreement."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .schemas import PairwiseJudgment


@dataclass(frozen=True)
class AggregatedPreference:
    pair_id: str
    probability_a_better: float
    mean_preference: float
    majority_preference: int
    agreement: float
    n_raters: int
    effective_weight: float
    dimension_counts: dict[str, int]
    rationales: tuple[str, ...]


def _confidence_weight(confidence: int) -> float:
    # Low-confidence judgments remain visible but contribute less to consensus.
    return 0.4 + 0.15 * int(confidence)


def aggregate_judgments(
    judgments: Iterable[PairwiseJudgment],
    minimum_stay_ms: int = 0,
) -> dict[str, AggregatedPreference]:
    grouped: dict[str, list[PairwiseJudgment]] = defaultdict(list)
    for judgment in judgments:
        if minimum_stay_ms and judgment.stay_ms is not None and judgment.stay_ms < minimum_stay_ms:
            continue
        grouped[judgment.pair_id].append(judgment)

    result: dict[str, AggregatedPreference] = {}
    for pair_id, rows in grouped.items():
        weights = [_confidence_weight(row.confidence) for row in rows]
        total_weight = sum(weights)
        mean = sum(row.preference * w for row, w in zip(rows, weights)) / total_weight
        # Map -2..2 to a soft probability that A is preferred.
        probability = (mean + 2.0) / 4.0
        counts = Counter(row.preference for row in rows)
        majority = sorted(counts, key=lambda value: (counts[value], abs(value), value), reverse=True)[0]
        agreement = counts[majority] / len(rows)
        dimension_counts = Counter(dim for row in rows for dim in row.dimensions)
        rationales = tuple(row.rationale.strip() for row in rows if row.rationale.strip())
        result[pair_id] = AggregatedPreference(
            pair_id=pair_id,
            probability_a_better=max(0.0, min(1.0, probability)),
            mean_preference=mean,
            majority_preference=majority,
            agreement=agreement,
            n_raters=len(rows),
            effective_weight=total_weight,
            dimension_counts=dict(dimension_counts),
            rationales=rationales,
        )
    return result
