"""Evaluator 主类：fit → score_text / evaluate_set。

- fit(h_texts, g_texts)：训练判别器，缓存 H 参照分布（表示+属性）。
- score_text(x)：单篇三合一融合打分（供生成算法当 reward）。
- evaluate_set(h_texts, g_texts)：批报告——D_total、三视图分项、属性明细、
  以及（可选）与人类评分的人类对齐 Spearman。
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from scipy.stats import spearmanr

from .config import Config
from .features.attributes import ATTRIBUTE_KEYS, extract_attributes
from .metrics.distance import (
    mmd_rbf,
    wasserstein_mean,
    frechet_distance,
    human_coverage,
    machine_only_mass,
    wasserstein_1d,
)
from .metrics.mix_scorer import MixScorer
from .models.discriminator import build_discriminator
from .models.encoder import DocumentEncoder


class Evaluator:
    def __init__(self, config: Config):
        self.cfg = config
        self.discriminator = build_discriminator(config.discriminator)
        self.doc_encoder = DocumentEncoder(
            model_name=config.encoder.name,
            device=config.encoder.device,
            max_len=config.encoder.max_len,
            batch_size=config.encoder.batch_size,
        )
        self.scorer = MixScorer(
            lambda_disc=config.mix.lambda_disc,
            lambda_repr=config.mix.lambda_repr,
            lambda_attr=config.mix.lambda_attr,
            strict_alpha=config.mix.strict_alpha,
            strict_len_base=config.mix.strict_len_base or None,  # 0 → 自适应 H 中位
            window_chars=config.mix.window_chars,
        )
        self._fitted = False

    def fit(self, h_texts, g_texts) -> "Evaluator":
        """训练判别器并缓存 H 参照分布。"""
        self.scorer.fit(
            list(h_texts), list(g_texts),
            discriminator=self.discriminator,
            doc_encoder=self.doc_encoder,
        )
        self._fitted = True
        return self

    # ---- 单篇打分 API ----
    def score_text(self, text: str) -> float:
        if not self._fitted:
            raise RuntimeError("Evaluator not fitted: call fit() first")
        return self.scorer.score(text)

    def score_components(self, text: str) -> dict:
        if not self._fitted:
            raise RuntimeError("Evaluator not fitted: call fit() first")
        return self.scorer.components(text)

    # ---- 人类对齐（轻量，仅判别器） ----
    def alignment_spearman(self, h_texts, h_scores, g_texts=None, g_scores=None):
        """评估器分数与人类评分的 Spearman。只需判别器概率，不做距离计算。"""
        if not self._fitted:
            raise RuntimeError("Evaluator not fitted: call fit() first")
        return self._compute_alignment(
            list(h_texts), list(g_texts) if g_texts is not None else [],
            list(h_scores) if h_scores is not None else None,
            list(g_scores) if g_scores is not None else None,
        )

    # ---- 批报告 ----
    def evaluate_set(
        self,
        h_texts: Iterable[str],
        g_texts: Iterable[str],
        h_scores: Optional[Iterable[float]] = None,
        g_scores: Optional[Iterable[float]] = None,
        extra_g_texts_by_name: Optional[dict] = None,
    ) -> dict:
        """对 H/G 两组文本计算三视图距离报告。

        h_scores / g_scores 为人类评分（可选），用于计算人类对齐 Spearman。
        extra_g_texts_by_name: {名称: 文本列表}，用于多组机器文本对比 D_total 排序。
        """
        if not self._fitted:
            raise RuntimeError("Evaluator not fitted: call fit() first")

        h_texts = list(h_texts)
        g_texts = list(g_texts)

        # 判别距离
        auc = self.discriminator.auc(h_texts, g_texts)
        bacc = self.discriminator.balanced_acc(h_texts, g_texts)
        d_disc = 2.0 * auc - 1.0  # trial.txt 的两样本检验形式

        # 表示分布距离
        h_vec = self.doc_encoder.encode_documents(h_texts)
        g_vec = self.doc_encoder.encode_documents(g_texts)
        d_repr = {
            "mmd": mmd_rbf(h_vec, g_vec),
            "wasserstein": wasserstein_mean(h_vec, g_vec),
            "frechet": frechet_distance(h_vec, g_vec),
            "human_coverage": human_coverage(h_vec, g_vec),
            "machine_only_mass": machine_only_mass(h_vec, g_vec),
        }

        # 属性距离：逐属性 z-score 标准化后再算 1D Wasserstein（消除量纲差异）
        h_attr = np.array([_attr_vec(t) for t in h_texts])
        g_attr = np.array([_attr_vec(t) for t in g_texts])
        per_attr = _per_attribute_distance(h_attr, g_attr)
        d_attr_weighted = float(np.mean(list(per_attr.values()))) if per_attr else np.nan

        # 人类对齐：用判别器分数（或融合分数）与人类评分算 Spearman
        alignment = self._compute_alignment(h_texts, g_texts, h_scores, g_scores)

        # 融合总分
        d_total = (
            self.cfg.mix.lambda_disc * d_disc
            + self.cfg.mix.lambda_repr * (d_repr["mmd"] + d_repr["wasserstein"]) / 2.0
            + self.cfg.mix.lambda_attr * d_attr_weighted
        )

        report = {
            "D_total": d_total,
            "D_disc": {"auc": auc, "balanced_acc": bacc, "two_sample_disc": d_disc},
            "D_repr": d_repr,
            "D_attr": {"per_attribute": per_attr, "weighted": d_attr_weighted},
            "n_h": len(h_texts),
            "n_g": len(g_texts),
        }
        if alignment is not None:
            report["alignment"] = alignment

        # 可选：多组机器文本的 D_total 对比
        if extra_g_texts_by_name:
            h_disc_auc = auc
            rankings = {}
            for name, gx in extra_g_texts_by_name.items():
                if not gx:
                    continue
                gx = list(gx)
                dx = self.discriminator.auc(h_texts, gx)
                hx = self.doc_encoder.encode_documents(gx)
                mmx = mmd_rbf(h_vec, hx)
                wx = wasserstein_mean(h_vec, hx)
                ax = np.array([_attr_vec(t) for t in gx])
                per_attr_x = _per_attribute_distance(h_attr, ax)
                axw = float(np.mean(list(per_attr_x.values()))) if per_attr_x else np.nan
                total = (
                    self.cfg.mix.lambda_disc * (2.0 * dx - 1.0)
                    + self.cfg.mix.lambda_repr * (mmx + wx) / 2.0
                    + self.cfg.mix.lambda_attr * axw
                )
                rankings[name] = total
            report["rankings"] = dict(sorted(rankings.items(), key=lambda kv: kv[1]))
        return report

    def _compute_alignment(self, h_texts, g_texts, h_scores, g_scores):
        """评估器分数与人类评分的 Spearman。任一侧缺评分则只算有的一侧。"""
        h_texts = list(h_texts)
        g_texts = list(g_texts)
        h_scores = list(h_scores) if h_scores is not None else []
        g_scores = list(g_scores) if g_scores is not None else []
        if not h_scores and not g_scores:
            return None
        # 分别算 Spearman，避免人为地把 H/G 混在一起制造假相关
        res = {}
        if len(h_scores) >= 3 and len(h_texts) == len(h_scores):
            ev_h = self.discriminator.predict_human_prob(h_texts)
            r_h, p_h = spearmanr(ev_h, h_scores)
            res["spearman_human_H"] = float(r_h)
            res["spearman_p_H"] = float(p_h)
        if len(g_scores) >= 3 and len(g_texts) == len(g_scores):
            ev_g = self.discriminator.predict_human_prob(g_texts)
            r_g, p_g = spearmanr(ev_g, g_scores)
            res["spearman_human_G"] = float(r_g)
            res["spearman_p_G"] = float(p_g)
        # 合并两组的 Spearman（若两组都有分）
        if len(h_scores) >= 3 and len(g_scores) >= 3 and len(h_texts) == len(h_scores) and len(g_texts) == len(g_scores):
            ev_h = self.discriminator.predict_human_prob(h_texts)
            ev_g = self.discriminator.predict_human_prob(g_texts)
            all_ev = np.concatenate([ev_h, ev_g])
            all_hm = np.concatenate([h_scores, g_scores])
            r_all, p_all = spearmanr(all_ev, all_hm)
            res["spearman_human_all"] = float(r_all)
            res["spearman_p_all"] = float(p_all)
        return res or None


def _attr_vec(text: str) -> list:
    attrs = extract_attributes(text)
    return [attrs[k] for k in ATTRIBUTE_KEYS]


def _per_attribute_distance(h_attr: np.ndarray, g_attr: np.ndarray) -> dict:
    """逐属性 1D Wasserstein，属性值先做 z-score（用 H/G 合并分布的均值/方差）。

    消除不同属性量纲差异（如句长方差 vs 标点比例），使各属性距离可比。
    """
    all_attr = np.concatenate([h_attr, g_attr], axis=0)
    per = {}
    for i, k in enumerate(ATTRIBUTE_KEYS):
        col = all_attr[:, i]
        sd = col.std()
        if sd < 1e-12:  # 常量属性无判别信息，跳过
            continue
        hz = (h_attr[:, i] - col.mean()) / sd
        gz = (g_attr[:, i] - col.mean()) / sd
        d = wasserstein_1d(hz, gz)
        if not np.isnan(d):
            per[k] = d
    return per
