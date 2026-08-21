from narrative_evaluator.quality.aggregation import aggregate_judgments
from narrative_evaluator.quality.schemas import PairwiseJudgment


def test_aggregation_preserves_ties_and_disagreement():
    rows = [
        PairwiseJudgment("p", "r1", 2, 5, ("character",), "A 人物更可信", stay_ms=9000),
        PairwiseJudgment("p", "r2", -2, 5, ("pacing",), "B 节奏更好", stay_ms=9000),
    ]
    result = aggregate_judgments(rows)["p"]
    assert result.probability_a_better == 0.5
    assert result.agreement == 0.5
    assert result.n_raters == 2
    assert result.dimension_counts == {"character": 1, "pacing": 1}


def test_fast_rows_can_be_excluded():
    rows = [
        PairwiseJudgment("p", "fast", 2, 5, stay_ms=100),
        PairwiseJudgment("p", "careful", -1, 3, stay_ms=9000),
    ]
    result = aggregate_judgments(rows, minimum_stay_ms=8000)["p"]
    assert result.n_raters == 1
    assert result.mean_preference == -1
