"""CLI 数据拆分测试。"""

import pytest

from narrative_evaluator.cli import _train_eval_split


def test_train_eval_split_is_disjoint_and_deterministic():
    items = list(range(20))
    train1, eval1 = _train_eval_split(items, 0.2, 42)
    train2, eval2 = _train_eval_split(items, 0.2, 42)
    assert (train1, eval1) == (train2, eval2)
    assert len(train1) == 16
    assert len(eval1) == 4
    assert set(train1).isdisjoint(eval1)
    assert sorted(train1 + eval1) == items


def test_train_eval_split_rejects_invalid_input():
    with pytest.raises(ValueError):
        _train_eval_split([1], 0.2, 42)
    with pytest.raises(ValueError):
        _train_eval_split([1, 2], 1.0, 42)
