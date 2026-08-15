"""人/机判别器：字符 n-gram + 逻辑回归。

输出 P(human)，用作：
- 判别距离 D_disc 的 AUC（H/G 可区分度）；
- 单篇打分 S_disc（判别视角的"像人程度"）。
Phase 1 扩展 MacBERT 判别器（models/macbert_discriminator.py），同一接口，
通过 build_discriminator 工厂按配置选择。
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from ..features.ngram import NgramFeaturizer


def build_discriminator(cfg, cache_dir=None):
    """按 DiscriminatorConfig 构造判别器。env NARRATIVE_DISC_TYPE 可覆盖 type。"""
    typ = os.environ.get("NARRATIVE_DISC_TYPE", getattr(cfg, "type", "ngram"))
    if typ == "macbert":
        from .macbert_discriminator import MacBertDiscriminator  # 惰性，避免顶层拉 transformers
        return MacBertDiscriminator(
            model_name=cfg.model_name,
            model_path=cfg.model_path or "data/eval_dataset/models/macbert_discriminator",
            max_len=cfg.max_len,
            batch_size=cfg.batch_size,
            device=cfg.device,
            cache_dir=cache_dir,
            epochs=cfg.epochs,
            lr=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            seed=cfg.seed,
            threads=getattr(cfg, "threads", 0),
        )
    return NgramDiscriminator(ngram_range=cfg.ngram_range, max_features=cfg.max_features)


class NgramDiscriminator:
    """字符 n-gram + LR 判别器。

    length_match=True 时训练样本按长度分层对齐 H/G，避免判别器把"长度"当判别线索
    （H 长/G 短的数据分布会让 n-gram 学到长度偏置）。
    """

    def __init__(self, ngram_range=(1, 3), max_features=50000):
        self.featurizer = NgramFeaturizer(ngram_range=ngram_range, max_features=max_features)
        self.clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
        self._fitted = False
        self.length_match_used = False

    def fit(self, h_texts, g_texts, length_match=False, n_buckets=8, seed=42):
        """h_texts/g_texts 为字符串列表。

        length_match=True：按长度分桶，每桶取 H/G 等量样本，消除长度偏置。
        匹配后的样本数记录在 self.used_n_h / self.used_n_g。
        """
        h_texts = list(h_texts)
        g_texts = list(g_texts)
        self.used_n_h = len(h_texts)
        self.used_n_g = len(g_texts)
        if length_match and h_texts and g_texts:
            h_texts, g_texts = _length_match(h_texts, g_texts, n_buckets, seed)
            self.length_match_used = True
            self.used_n_h = len(h_texts)
            self.used_n_g = len(g_texts)
        texts = h_texts + g_texts
        y = np.array([1] * len(h_texts) + [0] * len(g_texts))
        X = self.featurizer.fit_transform(texts)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_human_prob(self, texts) -> np.ndarray:
        """P(human) ∈ [0,1]。未 fit 时抛错。"""
        if not self._fitted:
            raise RuntimeError("discriminator not fitted")
        X = self.featurizer.transform(list(texts))
        return self.clf.predict_proba(X)[:, 1]

    def auc(self, h_texts, g_texts) -> float:
        """在给定 H/G 上的判别 AUC（2*ACC-1 形式的两样本检验统计量）。"""
        texts = list(h_texts) + list(g_texts)
        y = np.array([1] * len(h_texts) + [0] * len(g_texts))
        p = self.predict_human_prob(texts)
        return float(roc_auc_score(y, p))

    def balanced_acc(self, h_texts, g_texts) -> float:
        texts = list(h_texts) + list(g_texts)
        y = np.array([1] * len(h_texts) + [0] * len(g_texts))
        p = self.predict_human_prob(texts)
        return float(balanced_accuracy_score(y, p > 0.5))


def _length_match(h_texts, g_texts, n_buckets=8, seed=42):
    """按长度分桶使 H/G 长度分布对齐。

    用 H 的长度分位数定义桶边界，G 也归入这些桶；每桶取 H/G 的等量样本。
    返回匹配后的 (h_texts, g_texts)。
    """
    rng = np.random.RandomState(seed)
    h_texts = list(h_texts)
    g_texts = list(g_texts)
    h_lens = np.array([len(t) for t in h_texts])
    g_lens = np.array([len(t) for t in g_texts])
    # 桶边界用 H+G 合并长度分位数（保证两桶都有覆盖）
    all_lens = np.concatenate([h_lens, g_lens])
    edges = np.unique(np.percentile(all_lens, np.linspace(0, 100, n_buckets + 1)))
    edges = edges[:-1].tolist() + [np.inf]
    h_idx = np.digitize(h_lens, edges, right=False)
    g_idx = np.digitize(g_lens, edges, right=False)
    kept_h, kept_g = [], []
    for b in range(n_buckets):
        hb = np.where(h_idx == b)[0]
        gb = np.where(g_idx == b)[0]
        n = min(len(hb), len(gb))
        if n == 0:
            continue
        # 等量采样（若桶内样本数相等则全取）
        if n < len(hb):
            hb = rng.choice(hb, size=n, replace=False)
        if n < len(gb):
            gb = rng.choice(gb, size=n, replace=False)
        kept_h.extend([h_texts[i] for i in hb])
        kept_g.extend([g_texts[i] for i in gb])
    return kept_h, kept_g
