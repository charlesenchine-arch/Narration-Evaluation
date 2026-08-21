import pytest

from narrative_evaluator.quality.grades import DEFAULT_GRADE_SCALE, grade_for_score


@pytest.mark.parametrize(
    ("score", "grade"),
    [(100, "A"), (92, "A"), (91.99, "B"), (80, "B"),
     (79.99, "C"), (65, "C"), (64.99, "D"), (50, "D"), (49.99, "F"), (0, "F")],
)
def test_abcdf_boundaries(score, grade):
    assert grade_for_score(score, DEFAULT_GRADE_SCALE).code == grade


def test_grade_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        grade_for_score(100.1)
