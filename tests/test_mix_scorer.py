"""测试：三视图融合打分 + 长度严苛度——归一化范围、分量接口、严苛度衰减行为。

用假判别器和假编码器（不加载真实模型），只测 MixScorer 逻辑。
"""
import numpy as np
import pytest

from narrative_evaluator.metrics.mix_scorer import MixScorer, _strictness_kappa


class FakeDiscriminator:
    """给 H 文本高概率、给 G 文本低概率的假判别器。"""

    def fit(self, h_texts, g_texts):
        pass

    def predict_human_prob(self, texts):
        out = []
        for t in texts:
            out.append(0.9 if len(t) < 10 else 0.2)
        return np.array(out)


class FakeEncoder:
    """把文本长度映射到向量空间（短=人类，长=机器）。"""

    def encode_documents(self, texts):
        return np.array([[1.0 - min(len(t), 10) / 10.0, 0.5] for t in texts])

    def encode_document(self, text):
        return self.encode_documents([text])[0]


@pytest.fixture
def scorer():
    h = ["人类文本", "短句", "短文", "叙事", "中文", "好的"]
    g = ["这是一段比较长的机器生成的文本内容用于测试目的", "更长的一段机器生成的文本内容用于测试目的呀"]
    m = MixScorer(1 / 3, 1 / 3, 1 / 3)
    m.fit(h, g, FakeDiscriminator(), FakeEncoder())
    return m


def test_score_in_range(scorer):
    s = scorer.score("一段测试文本内容用于打分")
    assert 0.0 <= s <= 1.0


def test_components_in_range(scorer):
    c = scorer.components("任意文本")
    for k in ("S_disc", "S_repr", "S_attr"):
        assert 0.0 <= c[k] <= 1.0, f"{k} out of range: {c[k]}"


def test_components_keys(scorer):
    c = scorer.components("任意文本")
    assert set(c.keys()) == {"S_disc", "S_repr", "S_attr"}


def test_lambdas_normalized():
    m = MixScorer(0.5, 0.5, 0.5)
    assert abs(m.lambdas.sum() - 1.0) < 1e-9


def test_strictness_kappa_monotonic():
    # κ 随长度递增：长度≤基准=1，超出基准>1
    assert _strictness_kappa(100, 0.5, 1000) == 1.0          # 短，宽容
    assert _strictness_kappa(1100, 0.5, 1000) > 1.0          # 超出基准，严苛
    assert _strictness_kappa(2100, 0.5, 1000) > _strictness_kappa(1100, 0.5, 1000)
    assert _strictness_kappa(1100, 0.0, 1000) == 1.0         # α=0 不严苛


def test_adaptive_base_from_h():
    # fit 后 strict_len_base 取自 H 中位长度
    h = ["短。", "中篇。" * 5, "长文本。" * 20, "更长文本。" * 40]
    g = ["机器。"] * 4
    m = MixScorer(1, 0, 0)  # strict_len_base=None → 自适应
    m.fit(h, g, FakeDiscriminator(), FakeEncoder())
    lens = sorted(len(t) for t in h)
    assert m.strict_len_base == lens[len(lens) // 2]


def test_score_pressed_for_longer():
    # 同样像人分，长文被压得更低（严苛），短文宽容
    # 构造一个假判别器返回固定 0.9 的，看长度衰减效果
    class FixedDisc:
        def fit(self, h, g):
            pass

        def predict_human_prob(self, texts):
            return np.full(len(texts), 0.9)

    class FixedEnc:
        def encode_documents(self, texts):
            return np.zeros((len(texts), 4))

        def encode_document(self, text):
            return np.zeros(4)

    h = ["短。"] * 6
    g = ["长文本。"] * 6
    m = MixScorer(1, 0, 0, strict_alpha=0.5, strict_len_base=100)
    m.fit(h, g, FixedDisc(), FixedEnc())
    short_t = "短文"                       # len 2 < base 100
    long_t = "长" * 200                     # len 200 > base 100
    assert m.score(short_t) == pytest.approx(0.9)   # 短文不衰减
    assert m.score(long_t) < 0.9                   # 长文被压低
    # 长文下，"像人"(0.9) 与 "不像人"(0.6) 的分数差距被拉大（严苛度只压低分）
    class FixedLike:
        def fit(self, h, g):
            pass

        def predict_human_prob(self, texts):
            return np.full(len(texts), 0.9)

    class FixedMid:
        def fit(self, h, g):
            pass

        def predict_human_prob(self, texts):
            return np.full(len(texts), 0.6)

    long_t = "好" * 200  # len 200 > base 100，触发严苛
    m_like = MixScorer(1, 0, 0, strict_alpha=0.5, strict_len_base=100)
    m_like.fit(h, g, FixedLike(), FixedEnc())
    m_mid = MixScorer(1, 0, 0, strict_alpha=0.5, strict_len_base=100)
    m_mid.fit(h, g, FixedMid(), FixedEnc())
    # 0.9^2 = 0.81, 0.6^2 = 0.36，长文下 0.6 被压得更狠
    assert m_like.score(long_t) > m_mid.score(long_t)
    # 严苛度放大了差距：短文下 0.9 vs 0.6 差 0.3，长文下 0.81 vs 0.36 差 0.45
    gap_short = 0.9 - 0.6
    gap_long = m_like.score(long_t) - m_mid.score(long_t)
    assert gap_long > gap_short
