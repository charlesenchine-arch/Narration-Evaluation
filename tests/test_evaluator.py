"""测试：Evaluator 端到端（用假模型替换真实判别器/编码器，快速且无网络依赖）。"""
import numpy as np
import pytest

from narrative_evaluator.config import default_config
from narrative_evaluator.evaluator import Evaluator
from narrative_evaluator.models.discriminator import NgramDiscriminator
from narrative_evaluator.models.encoder import DocumentEncoder


class _FakeEncoder(DocumentEncoder):
    def __init__(self, **kw):
        self._d = 4

    def encode_documents(self, texts):
        return np.array([[len(t) % 7 / 7.0, 0.3, 0.2, 0.1] for t in texts], dtype=float)

    def encode_document(self, text):
        return self.encode_documents([text])[0]


def _make_evaluator():
    cfg = default_config()
    ev = Evaluator(cfg)
    # 用假编码器替换，避免加载真实模型
    ev.doc_encoder = _FakeEncoder()
    return ev


@pytest.fixture
def fitted():
    h = ["春眠不觉晓。处处闻啼鸟。夜来风雨声。花落知多少。", "床前明月光。疑是地上霜。"] * 5
    g = ["这是一段由机器生成的较为冗长的文本内容，用于填充机器样本池，字数较多。"] * 10
    ev = _make_evaluator()
    ev.fit(h, g)
    return ev


def test_score_text_returns_float(fitted):
    s = fitted.score_text("测试文本")
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_score_components_keys(fitted):
    c = fitted.score_components("测试文本")
    assert set(c.keys()) == {"S_disc", "S_repr", "S_attr"}


def test_evaluate_set_structure(fitted):
    h = ["春眠不觉晓。处处闻啼鸟。"] * 4
    g = ["机器生成的长文本，用于测试目的。"] * 4
    rep = fitted.evaluate_set(h, g)
    assert "D_total" in rep
    assert "D_disc" in rep and "auc" in rep["D_disc"]
    assert "D_repr" in rep and "mmd" in rep["D_repr"]
    assert "D_attr" in rep and "per_attribute" in rep["D_attr"]


def test_not_fitted_raises():
    ev = _make_evaluator()
    with pytest.raises(RuntimeError):
        ev.score_text("测试")


def test_evaluate_set_human_alignment():
    cfg = default_config()
    ev = Evaluator(cfg)
    ev.doc_encoder = _FakeEncoder()
    h = ["短文甲。", "短文乙。", "短文丙。", "短文丁。"]
    g = ["机器生成的长文本内容用于测试目的。", "机器生成的一段文本。"] * 2
    ev.fit(h, g)
    h_scores = [0.9, 0.8, 0.7, 0.6]
    rep = ev.evaluate_set(h, g, h_scores=h_scores)
    assert "alignment" in rep
    assert "spearman_human_H" in rep["alignment"]


def test_alignment_uses_fused_like_score():
    ev = _make_evaluator()
    ev._fitted = True
    values = {"甲": 0.1, "乙": 0.5, "丙": 0.9}
    ev.scorer.components_batch = lambda texts: [
        {"S_disc": 0.0, "S_repr": values[t], "S_attr": values[t]}
        for t in texts
    ]
    # 如果仍只用判别器，这个反向排序会得到 -1；融合 like 应得到 +1。
    ev.discriminator.predict_human_prob = lambda texts: np.array([0.9, 0.5, 0.1])
    result = ev.alignment_spearman(["甲", "乙", "丙"], [1, 2, 3])
    assert result["spearman_human_H"] == pytest.approx(1.0)
