"""三视图融合打分 + 长度严苛度衰减。

像人分：score_like(x) = λ₁S_disc + λ₂S_repr + λ₃S_attr，三视图加权，∈[0,1]，
衡量"这篇文本有多像人类写作"（无监督，不依赖人类评分数据）：
- S_disc: 判别器 P(human)，天然 [0,1]。
- S_repr: 文本向量到 H 表示集合的最近余弦相似度的分位数（相对 H 内部相似度分布）。
- S_attr: 文本属性向量相对 H 属性分布的马氏/z-score 距离，取负后映射到 [0,1]
  （用 H 内部属性距离的经验分布做 CDF）。衡量"离典型人类文本多远"。

严苛度衰减：最终分 = 像人分^κ(len)。
篇幅越长，机器越容易"露馅"（重复、空洞、逻辑断裂更明显），所以对长文更严苛：
- 基准长度 L0 = H 参照分布的中位长度（fit 时自适应，人类典型长度的文本不额外受罚）；
- κ(len) = 1 + α · max(0, len - L0) / L0，随长度单调递增；
- 长度 ≤ L0（人类典型）：κ=1，宽容，像人分直接作为分数；
- 长度 > L0：κ>1，幂指数压低分数，且"越不像人的文本被压得越狠"（x^κ, κ>1
  对较小的 x 惩罚更大）——足够像人的长文几乎不受罚，不够像人的长文明显受罚。

fit() 阶段基于 H 参照集预计算归一化所需的参照分布。
"""
from __future__ import annotations

import numpy as np

from ..features.attributes import ATTRIBUTE_KEYS, extract_attributes


def _to_cdf_scores(raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """用参照分布的秩/经验 CDF 把原始分数映射到 [0,1]。

    raw 越大 → 分数越接近 1（经验 CDF 值）。参照分布提供"正常值"的分布形状。
    """
    ref = np.asarray(ref, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if ref.size < 5 or raw.size == 0:
        return np.clip(raw, 0.0, 1.0)
    ref_sorted = np.sort(ref)
    cdf = np.searchsorted(ref_sorted, raw, side="right") / ref_sorted.size
    return np.clip(cdf, 0.0, 1.0)


def _strictness_kappa(len_chars: int, alpha: float, len_base: int) -> float:
    """严苛度指数 κ(len)：长度≤基准=1（宽容），超出基准>1（严苛）。"""
    over = max(0, len_chars - len_base)
    return 1.0 + alpha * over / max(1, len_base)


class MixScorer:
    """维护 H 参照分布，提供单篇三视图打分 + 长度严苛度调节。"""

    def __init__(self, lambda_disc=1 / 3, lambda_repr=1 / 3, lambda_attr=1 / 3,
                 strict_alpha=0.2, strict_len_base=None, window_chars=0):
        self.lambdas = np.array([lambda_disc, lambda_repr, lambda_attr], dtype=np.float64)
        self.lambdas = self.lambdas / self.lambdas.sum()
        self.strict_alpha = strict_alpha     # 严苛强度（0=不严苛，温和约0.2）
        self.strict_len_base = strict_len_base  # None=fit 时用 H 中位长度
        # 部署加固：S_repr/S_attr 在"前 window_chars 字符"上计算，0=全文本。
        # 防 B1 末尾注水：末尾追加通用填充会拉高整篇表示相似度，窗口化使其失效。
        self.window_chars = int(window_chars or 0)
        # fit 后填充
        self.discriminator = None
        self.doc_encoder = None
        self.h_repr: np.ndarray | None = None          # [n_h, d] 归一化
        self.h_repr_sims: np.ndarray | None = None      # H 内部两两最近余弦
        self.h_attr: np.ndarray | None = None           # [n_h, n_attr]
        self.h_attr_dist: np.ndarray | None = None      # H 内部属性距离分布（用于 CDF）

    def fit(self, h_texts, g_texts, discriminator, doc_encoder) -> "MixScorer":
        """训练判别器、编码 H 参照分布、计算属性参照分布。"""
        self.discriminator = discriminator
        self.doc_encoder = doc_encoder
        h_texts = list(h_texts)
        # 严苛度基准：H 中位长度（人类典型长度不额外受罚）
        if self.strict_len_base is None:
            if h_texts:
                lens = sorted(len(t) for t in h_texts)
                self.strict_len_base = lens[len(lens) // 2]
            else:
                self.strict_len_base = 1000
        # 判别器
        discriminator.fit(h_texts, g_texts)
        # 表示参照
        self.h_repr = doc_encoder.encode_documents(h_texts)
        self._fit_repr_ref()
        # 属性参照
        self.h_attr = np.array([_attr_vec(t) for t in h_texts])
        self._fit_attr_ref()
        return self

    def _fit_repr_ref(self):
        # H 内部最近余弦相似度分布：用于把新样本的最近相似度映射到分位数
        if self.h_repr is None or self.h_repr.shape[0] < 5:
            self.h_repr_sims = None
            return
        sims = _self_nearest_sims(self.h_repr)
        self.h_repr_sims = sims

    def _probe(self, text: str) -> str:
        """窗口化探针文本：window_chars>0 时取前 window_chars 字符用于 S_repr/S_attr。"""
        if self.window_chars <= 0 or len(text) <= self.window_chars:
            return text
        return text[:self.window_chars]

    def _fit_attr_ref(self):
        # H 内部每个属性的马氏距离分布（对角协方差近似）
        if self.h_attr is None or self.h_attr.shape[0] < 5:
            self.h_attr_dist = None
            return
        mu = self.h_attr.mean(axis=0)
        sd = self.h_attr.std(axis=0) + 1e-9
        z = (self.h_attr - mu) / sd
        # 每篇 H 的"离群程度"：各属性 |z| 的均值（实际分布用于 CDF）
        self.h_attr_dist = np.abs(z).mean(axis=1)
        self._attr_mu = mu
        self._attr_sd = sd

    def components(self, text: str) -> dict:
        """返回三视图分量分数（均 [0,1]，高=像人）。

        S_disc 用全文本（判别器稳健）；S_repr/S_attr 用窗口化探针
        （window_chars>0 时取前缀窗口，防末尾注水虚涨）。
        """
        # S_disc
        s_disc = float(self.discriminator.predict_human_prob([text])[0])
        probe = self._probe(text)

        # S_repr：探针到 H 最近余弦相似度 → CDF
        s_repr = 0.5
        if self.h_repr is not None and self.h_repr.shape[0] > 0:
            v = self.doc_encoder.encode_document(probe)
            sim = _nearest_cosine(v, self.h_repr)
            ref = self.h_repr_sims
            if ref is not None and ref.size >= 5:
                s_repr = float(_to_cdf_scores(np.array([sim]), ref)[0])
            else:
                s_repr = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

        # S_attr：探针属性离群度取负 → CDF（离群度低=更像 H=高分）
        s_attr = 0.5
        if self.h_attr is not None and self.h_attr.shape[0] >= 5:
            a = np.array(_attr_vec(probe))
            z = np.abs((a - self._attr_mu) / self._attr_sd).mean()
            ref = self.h_attr_dist
            if ref is not None and ref.size >= 5:
                # 离群度越小越像人：用 1 - CDF(离群度)
                cdf = float(np.searchsorted(np.sort(ref), z, side="right") / ref.size)
                s_attr = 1.0 - cdf
        return {"S_disc": s_disc, "S_repr": s_repr, "S_attr": s_attr}

    def components_batch(self, texts) -> list:
        """批量三视图分量（MacBERT 单样本 forward 慢，批处理必要）。

        返回 [{S_disc, S_repr, S_attr}] 列表。等价于逐个 components()，
        但判别器推理走 predict_human_prob 批量路径。
        S_disc 用全文本；S_repr/S_attr 用窗口化探针（见 window_chars）。
        """
        import numpy as _np
        texts = list(texts)
        if not texts:
            return []
        p = self.discriminator.predict_human_prob(texts)
        probes = [self._probe(t) for t in texts]
        # 表示视图：探针到 H 集最近余弦 → CDF（与 components 内单样本逻辑一致）
        s_repr = []
        if self.h_repr is not None and self.h_repr.shape[0] > 0:
            vecs = self.doc_encoder.encode_documents(probes)
            sims = [float((self.h_repr @ v).max()) for v in vecs]
            ref = self.h_repr_sims
            for sim in sims:
                if ref is not None and ref.size >= 5:
                    s_repr.append(float(_to_cdf_scores(_np.array([sim]), ref)[0]))
                else:
                    s_repr.append(float(_np.clip((sim + 1.0) / 2.0, 0.0, 1.0)))
        else:
            s_repr = [0.5] * len(texts)
        # 属性视图
        s_attr = []
        if self.h_attr is not None and self.h_attr.shape[0] >= 5:
            for t in probes:
                a = _np.array(_attr_vec(t))
                z = float(_np.abs((a - self._attr_mu) / self._attr_sd).mean())
                cdf = float(_np.searchsorted(_np.sort(self.h_attr_dist), z, side="right") / self.h_attr_dist.size)
                s_attr.append(1.0 - cdf)
        else:
            s_attr = [0.5] * len(texts)
        return [{"S_disc": float(d), "S_repr": float(r), "S_attr": float(a)}
                for d, r, a in zip(p, s_repr, s_attr)]

    def like_human_score(self, text: str) -> float:
        """三视图融合"像人分" ∈ [0,1]，未施加长度严苛度。"""
        c = self.components(text)
        return float(np.clip(
            self.lambdas[0] * c["S_disc"]
            + self.lambdas[1] * c["S_repr"]
            + self.lambdas[2] * c["S_attr"],
            0.0, 1.0,
        ))

    def strictness(self, len_chars: int) -> float:
        """当前长度对应的严苛度指数 κ。"""
        return _strictness_kappa(len_chars, self.strict_alpha, self.strict_len_base)

    def score(self, text: str) -> float:
        """综合分 ∈ [0,1]：像人分经长度严苛度衰减。长文需更接近 1 的像人分才维持高分。"""
        like = self.like_human_score(text)
        kappa = self.strictness(len(text))
        return float(np.clip(like ** kappa, 0.0, 1.0))


def _attr_vec(text: str) -> list:
    attrs = extract_attributes(text)
    return [attrs[k] for k in ATTRIBUTE_KEYS]


def _self_nearest_sims(V: np.ndarray) -> np.ndarray:
    """每点到集合内其他点的最大余弦相似度（自身 1.0 排除）。"""
    sims = V @ V.T
    n = V.shape[0]
    np.fill_diagonal(sims, -np.inf)
    return sims.max(axis=1)


def _nearest_cosine(v: np.ndarray, V: np.ndarray) -> float:
    """单点到集合的最大余弦相似度。"""
    s = (V @ v).max()
    return float(s)
