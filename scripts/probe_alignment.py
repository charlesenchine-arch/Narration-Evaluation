"""对齐方向验证实验：什么样的特征最贴合 WebNovelBench 人类评分。

背景：判别器 P(human) 与人类评分 Spearman ≈ -0.08（正交）。
诊断发现章节长度与人类评分相关 0.38。
本脚本对比多类特征的对齐度，判断"可监督特征是否有价值"。

用 200 章 WebNovelBench（自带人类评分）做内部对齐：
特征 A：判别器 P(human)
特征 B：章节长度（对数）
特征 C：属性合成分数（相对 H 分布的 z-score 距离取负）
特征 D：判别器 + 长度（简单线性组合，用一半样本拟合权重）
"""
from __future__ import annotations

import tempfile

import numpy as np
from scipy.stats import spearmanr, pearsonr

from narrative_evaluator.config import Config
from narrative_evaluator.data import load_webnovelbench, load_cnnsum, load_coig_writer
from narrative_evaluator.features.attributes import ATTRIBUTE_KEYS, extract_attributes
from narrative_evaluator.models.discriminator import NgramDiscriminator


def main():
    cache = tempfile.mkdtemp(prefix="hf_probe_")
    cfg = Config()
    print("加载数据...", flush=True)
    h_web = load_webnovelbench(cfg.data, cache)
    h_cnn = load_cnnsum(cfg.data, cache)
    g_coig = load_coig_writer(cfg.data, cache)

    # 训练判别器（用全部 H 作正样本 + COIG 作负样本）
    disc = NgramDiscriminator(
        ngram_range=cfg.discriminator.ngram_range,
        max_features=cfg.discriminator.max_features,
    )
    disc.fit(h_web.texts + h_cnn.texts, g_coig.texts)

    # 对齐样本：200 章 WebNovelBench
    texts = [r.text for r in h_web.records]
    scores = np.array([r.human_score for r in h_web.records])

    # 特征 A：判别器 P(human)
    p_human = disc.predict_human_prob(texts)

    # 特征 B：长度（对数）
    lens = np.array([len(t) for t in texts])
    log_len = np.log1p(lens)

    # 特征 C：属性合成（相对 H 分布 z-score 距离取负）
    attr_vals = np.array([_attr_vec(t) for t in texts])
    mu = attr_vals.mean(0)
    sd = attr_vals.std(0) + 1e-9
    attr_dist = np.abs((attr_vals - mu) / sd).mean(1)  # 离群度
    attr_score = -attr_dist  # 离群度小=更像 H=分高

    features = {
        "A 判别器P(human)": p_human,
        "B 长度(对数)": log_len,
        "C 属性合成": attr_score,
    }

    print("\n=== 单特征与人类评分的对齐度（n=200）===")
    for name, f in features.items():
        rho, p = spearmanr(f, scores)
        r, _ = pearsonr(f, scores)
        print(f"  {name:<20} Spearman={rho:+.3f} (p={p:.3f})  Pearson={r:+.3f}")

    # 特征 D：判别器 + 长度线性组合（留一/一半拟合权重）
    # 用前半拟合、后半验证，避免自评估
    n = len(texts)
    half = n // 2
    X = np.stack([p_human, log_len], axis=1)
    # 简单权重：在训练一半上用最小二乘
    A = np.column_stack([X[:half], np.ones(half)])
    w, *_ = np.linalg.lstsq(A, scores[:half], rcond=None)
    combo_train = A @ w
    combo_test = np.column_stack([X[half:], np.ones(n - half)]) @ w
    rho_t, p_t = spearmanr(combo_train, scores[:half])
    rho_v, p_v = spearmanr(combo_test, scores[half:])
    print(f"\n=== D 判别器+长度线性组合 ===")
    print(f"  训练半 Spearman={rho_t:+.3f} (p={p_t:.3f})")
    print(f"  验证半 Spearman={rho_v:+.3f} (p={p_v:.3f})   [关键：验证半>0 才说明可泛化]")

    # 归一化组合权重（可解释）
    print(f"  组合权重(判别器, 长度, 截距) = {np.round(w, 3)}")
    # 长度单独预测在验证半的表现
    rho_len_v, _ = spearmanr(log_len[half:], scores[half:])
    rho_disc_v, _ = spearmanr(p_human[half:], scores[half:])
    print(f"  对照——验证半上 长度单独={rho_len_v:+.3f}, 判别器单独={rho_disc_v:+.3f}")


def _attr_vec(text: str) -> list:
    attrs = extract_attributes(text)
    return [attrs[k] for k in ATTRIBUTE_KEYS]


if __name__ == "__main__":
    main()
