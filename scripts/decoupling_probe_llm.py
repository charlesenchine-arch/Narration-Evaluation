"""解耦性实验：自建评分集上"像人分 vs LLM 质量分"正交性验证（P0 步骤 4）。

论文主线（解耦性实证）在自建数据集上的验证。用 LLM-as-Judge 批量评出的
质量分（scripts/llm_quality_rating.py 产出）作为"质量分"一侧，
判别器 P(human) 作为"像人分"一侧。

命题（与 decoupling_probe.py 一致，但数据源换成自建评分集 + LLM 分）：
- 命题 1 正交性：同一批文本上，像人分 vs 质量分 Spearman ≈ 0。
  * H 内（人写文本内部）：像人分不携带质量信息 → 应 ≈0
  * G 内（机器文本内部）：同样 → 应 ≈0
- 命题 2 区分度分离：像人分能区分 H/G（AUC 高），但区分不了质量高低
  （区分 H 内部高低质子集 AUC ≈ 0.5）。
- 命题 3 长度偏置：像人分与长度相关（长度匹配训练后应消除）。

附加：
- 锚定校验：LLM 质量分应对极好锚定(AH)打高分、极差锚定(AL)打低分，
  验证 LLM judge 尺度有效（这是把它当质量分的前提）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from narrative_evaluator.config import Config, load_config
from narrative_evaluator.models.discriminator import build_discriminator


def _load_rating_set(path):
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    return items


def _load_llm_scores(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="配置文件（可用 configs/default_macbert.yaml 切换 MacBERT）")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else Config()
    items = _load_rating_set("data/eval_dataset/rating_set.jsonl")
    llm = _load_llm_scores("data/eval_dataset/raw/llm_quality_ratings.jsonl")
    print(f"评分集 {len(items)} 条，LLM 分 {len(llm)} 条", flush=True)

    # 按来源分组
    by_id = {}
    for it in items:
        prefix = it["id"].split("_")[0]
        by_id[it["id"]] = {"text": it["text"], "src": prefix}

    # 非锚定 H/G 作为判别器训练 + 解耦分析主体；锚定单独做校验
    h_ids = [i for i, v in by_id.items() if v["src"] == "H"]
    g_ids = [i for i, v in by_id.items() if v["src"] in ("G", "GZ")]
    anchor_ids = [i for i, v in by_id.items() if v["src"] in ("AH", "AL")]
    print(f"  H={len(h_ids)}  G={len(g_ids)}  锚定={len(anchor_ids)}", flush=True)

    # 必须有 LLM 分
    h_ids = [i for i in h_ids if i in llm]
    g_ids = [i for i in g_ids if i in llm]
    print(f"  有 LLM 分: H={len(h_ids)} G={len(g_ids)}", flush=True)

    # 判别器（type=ngram 时在评分集上重训 length_match；type=macbert 时加载部署判别器）
    h_texts = [by_id[i]["text"] for i in h_ids]
    g_texts = [by_id[i]["text"] for i in g_ids]
    print(f"构造判别器 type={cfg.discriminator.type}...", flush=True)
    disc = build_discriminator(cfg.discriminator)
    disc.fit(h_texts, g_texts, length_match=True, n_buckets=8, seed=42)
    print(f"  used: H={disc.used_n_h} G={disc.used_n_g} length_match={disc.length_match_used}", flush=True)

    # 对非锚定文本算像人分 + 长度
    p_h = disc.predict_human_prob([by_id[i]["text"] for i in h_ids])
    p_g = disc.predict_human_prob([by_id[i]["text"] for i in g_ids])
    len_h = np.array([len(by_id[i]["text"]) for i in h_ids])
    len_g = np.array([len(by_id[i]["text"]) for i in g_ids])
    s_h = np.array([llm[i] for i in h_ids])
    s_g = np.array([llm[i] for i in g_ids])

    print("\n=== 命题 1：正交性（像人分 vs LLM 质量分）===")
    rho_h, p_h_rho = spearmanr(p_h, s_h)
    rho_g, p_g_rho = spearmanr(p_g, s_g)
    print(f"  H 内: Spearman = {rho_h:+.3f}  (p={p_h_rho:.3f}, n={len(h_ids)})")
    print(f"  G 内: Spearman = {rho_g:+.3f}  (p={p_g_rho:.3f}, n={len(g_ids)})")
    print("  （≈0 = 像人分不携带质量信息 → 解耦成立）")

    print("\n=== 命题 2：区分度分离 ===")
    all_p = np.concatenate([p_h, p_g])
    all_y = np.array([1] * len(h_ids) + [0] * len(g_ids))
    auc_hm = roc_auc_score(all_y, all_p)
    print(f"  像人分区分 H/G: AUC = {auc_hm:.3f}  (高 = 能区分人机)")

    # 像人分区分 H 内部高低质（LLM 分中位切）
    med = np.median(s_h)
    hi = s_h > med
    lo = s_h <= med
    if hi.sum() > 0 and lo.sum() > 0:
        auc_qual = roc_auc_score(
            np.concatenate([np.ones(hi.sum()), np.zeros(lo.sum())]),
            np.concatenate([p_h[hi], p_h[lo]]),
        )
        print(f"  像人分区分 H 内部高低质(LLM分): AUC = {auc_qual:.3f}  (≈0.5 = 不携带质量信息 → 解耦)")
    else:
        print("  H 内 LLM 分无区分，跳过")

    # 对照：LLM 质量分能否区分 H/G
    auc_llm_hm = roc_auc_score(all_y, np.concatenate([s_h, s_g]))
    print(f"  [对照] LLM 质量分区分 H/G: AUC = {auc_llm_hm:.3f}  (质量分本身对人机有何信息)")

    print("\n=== 命题 3：长度偏置 ===")
    for name, p, ln in [("H", p_h, len_h), ("G", p_g, len_g)]:
        rho_l, _ = spearmanr(p, ln)
        print(f"  {name}: 像人分 vs 长度 Spearman = {rho_l:+.3f}")

    print("\n=== 锚定校验（LLM 分尺度有效性）===")
    ah_scores = [llm[i] for i in anchor_ids if i in llm and by_id[i]["src"] == "AH"]
    al_scores = [llm[i] for i in anchor_ids if i in llm and by_id[i]["src"] == "AL"]
    print(f"  极好锚定(AH) LLM 平均分: {np.mean(ah_scores):.2f}  (n={len(ah_scores)}) 应高")
    print(f"  极差锚定(AL) LLM 平均分: {np.mean(al_scores):.2f}  (n={len(al_scores)}) 应低")
    print(f"  [LLM 分对锚定的区分度] AUC = "
          f"{roc_auc_score([1]*len(ah_scores)+[0]*len(al_scores), ah_scores+al_scores):.3f}")

    # LLM 分分布概览
    print("\n=== LLM 质量分分布（非锚定）===")
    for name, s in [("H", s_h), ("G", s_g)]:
        print(f"  {name}: mean={s.mean():.2f} std={s.std():.2f} min={s.min()} max={s.max()} "
              f"hist={np.histogram(s, bins=5, range=(1,6))[0].tolist()}")


if __name__ == "__main__":
    main()
