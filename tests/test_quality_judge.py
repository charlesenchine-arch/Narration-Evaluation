import json

import pytest

from narrative_evaluator.quality.judge import parse_pairwise_judge_response


def _response(**changes):
    obj = {
        "preference": 1,
        "score_a": 82,
        "score_b": 70,
        "grade_a": "B",
        "grade_b": "C",
        "primary_dimensions": ["plot_causality"],
        "evidence_a": "A 有完整因果链",
        "evidence_b": "B 的转折缺少铺垫",
        "critique": "A 的情节推进更成立。",
    }
    obj.update(changes)
    return json.dumps(obj, ensure_ascii=False)


def test_parse_structured_judge_with_d_grade():
    parsed = parse_pairwise_judge_response(
        _response(score_b=55, grade_b="D")
    )
    assert parsed["grade_b"] == "D"


def test_reject_grade_score_mismatch():
    with pytest.raises(ValueError, match="grade"):
        parse_pairwise_judge_response(_response(grade_b="D"))


def test_reject_preference_score_conflict():
    with pytest.raises(ValueError, match="conflicts"):
        parse_pairwise_judge_response(_response(score_a=66, grade_a="C", score_b=82, grade_b="B"))
