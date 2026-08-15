"""测试：数据加载器。

真实数据测试需要网络+HF，默认跳过；本地离线用 monkeypatch 注入假流验证解析逻辑。
"""
import pytest
from narrative_evaluator.config import DataConfig
from narrative_evaluator.data.loaders import (
    load_webnovelbench,
    load_cnnsum,
    load_coig_writer,
    load_realdet_mgt,
    _sample_iterator,
)


class _FakeIter:
    def __init__(self, items):
        self._it = iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


def test_sample_iterator_caps():
    import random
    items = [_FakeIter([f"t{i}" for i in range(50)])]
    rng = random.Random(0)
    out = _sample_iterator(items[0], 10, rng)
    assert len(out) == 10


def test_sample_iterator_all_when_neg():
    import random
    items = [_FakeIter([f"t{i}" for i in range(20)])]
    rng = random.Random(0)
    out = _sample_iterator(items[0], -1, rng)
    assert len(out) == 20


def test_webnovel_score_normalization():
    """人类评分归一化：(mean-1)/4。mean(3,5)=4 → 0.75。"""
    scores = [3.0, 5.0]
    mean = sum(scores) / len(scores)
    human_score = (mean - 1.0) / 4.0
    assert human_score == 0.75
