"""Human-aligned narrative quality evaluation.

This package is intentionally separate from the legacy human-vs-machine
evaluator.  Quality is represented as an internal continuous utility and an
external A/B/C/D/F grade; supervision comes primarily from pairwise human
preferences and evidence-grounded critiques.
"""

from .grades import DEFAULT_GRADE_SCALE, GradeBand, grade_for_score
from .schemas import NarrativeItem, NarrativePair, PairwiseJudgment
from .aggregation import AggregatedPreference, aggregate_judgments
from .splitting import PairSplit, split_pairs

__all__ = [
    "DEFAULT_GRADE_SCALE",
    "GradeBand",
    "grade_for_score",
    "NarrativeItem",
    "NarrativePair",
    "PairwiseJudgment",
    "AggregatedPreference",
    "aggregate_judgments",
    "PairSplit",
    "split_pairs",
]
