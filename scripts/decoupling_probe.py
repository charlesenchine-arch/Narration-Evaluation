"""解耦性实验：像人程度 vs 写作质量是两个可解耦的维度。

论文主线。核心命题（在 WebNovelBench 现有数据上验证）：

命题 1（正交性）：同一批文本上，判别器"像人分"与人类"质量分"相关性≈0。
命题 2（区分度分离）：像人分能区分 H/G（AUC 高），质量分区分不了 H/G（AUC≈0.5）。
命题 3（双向可构造，留出）：存在"像人但低质"与"不像人但高质"文本（需构造，暂缓）。

方法：WebNovelBench 自带 8 评委质量分（H 侧）；COIG 作 G 侧。
- 判别器像人分：n-gram+LR 的 P(human)。
- 质量分：WebNovelBench 归一化评分（0-1）。
注意：G 侧暂无人评分，命题 1 主要在 H 侧测（H 内部像人分 vs 质量分）。
"""
from __future__ import annotations

import tempfile

import numpy as np
from scipy.stats import spearmanr

from narrative_evaluator.config import Config
from narrative_evaluator.data import load_webnovelbench, load_cnnsum, load_coig_writer
from narrative_evaluator.models.discriminator import NgramDiscriminator


def main():
    cache = tempfile.mkdtemp(prefix="hf_decouple_")
    cfg = Config()
    print("加载数据...", flush=True)
    h_web = load_webnovelbench(cfg.data, cache)
    h_cnn = load_cnnsum(cfg.data, cache)
    g_coig = load_coig_writer(cfg.data, cache)
    H_all = h_web.texts + h_cnn.texts

    print("训练判别器（长度匹配，消除长度偏置）...", flush=True)
    disc = NgramDiscriminator(
        ngram_range=cfg.discriminator.ngram_range,
        max_features=cfg.discriminator.max_features,
    )
    # 长度匹配训练：只用长度重叠的短文本（H 2-6K + G 1-6K），消除长度作为判别线索
    H_mid = [t for t in H_all if 1500 <= len(t) <= 6000]
    G_mid = [t for t in g_coig.texts if 1000 <= len(t) <= 6000]
    print(f"  长度重叠区间: H={len(H_mid)} G={len(G_mid)}", flush=True)
    disc.fit(H_mid, G_mid, length_match=True, n_buckets=8, seed=42)
    print(f"  长度匹配后训练样本: H={disc.used_n_h} G={disc.used_n_g}", flush=True)

    # 命题 1：H 侧（WebNovelBench）像人分 vs 质量分
    texts = [r.text for r in h_web.records]
    scores = np.array([r.human_score for r in h_web.records])
    p_human = disc.predict_human_prob(texts)

    rho, p = spearmanr(p_human, scores)
    print("\n=== 命题 1：正交性（WebNovelBench H 侧）===")
    print(f"  像人分 vs 质量分 Spearman = {rho:+.3f}  (p={p:.3f})")
    print(f"  n={len(texts)}")

    # 命题 2：区分度分离
    print("\n=== 命题 2：区分度分离 ===")
    # 像人分区分 H/G（用长度重叠区间评估，与训练分布一致）
    from sklearn.metrics import roc_auc_score
    all_texts = H_mid + G_mid
    labels = np.array([1] * len(H_mid) + [0] * len(G_mid))
    p_all = disc.predict_human_prob(all_texts)
    auc_like = roc_auc_score(labels, p_all)
    print(f"  像人分区分 H/G: AUC = {auc_like:.3f}  (高 = 能区分)")

    # 质量分区分 H/G：H 有分，G 无分——无法直接算。
    # 用代理：质量分区分"H 高分 vs H 低分"子集的能力，看它是否与"像人分"正交。
    hi = scores > np.median(scores)
    lo = scores <= np.median(scores)
    p_hi = p_human[hi]
    p_lo = p_human[lo]
    # 如果像人分能区分高低质 H 子集 → 解耦不成立；如果不能 → 解耦成立
    auc_qual = roc_auc_score(
        np.concatenate([np.ones(len(p_hi)), np.zeros(len(p_lo))]),
        np.concatenate([p_hi, p_lo]),
    )
    print(f"  像人分区分 H 内部高低质子集: AUC = {auc_qual:.3f}  (≈0.5 = 像人分不携带质量信息 → 解耦)")

    # 质量分本身能区分高低质（自证有效）
    print(f"  [对照] 质量分高低子集本身: 高质 {scores[hi].mean():.3f} vs 低质 {scores[lo].mean():.3f}")

    # 命题 2 完整版需要 G 侧质量分（future work 收集）。
    print("\n[注] G 侧人类质量分未收集，完整区分度分离（质量分 AUC≈0.5）需 G 评分后验证。")


if __name__ == "__main__":
    main()
