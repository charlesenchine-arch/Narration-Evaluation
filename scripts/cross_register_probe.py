"""跨寄存器解耦探针：把机器文本按风格寄存器（非章回 G / 章回体 GZ）分开验证解耦。

论文主线（像人分 vs 质量分正交 + 区分度分离）不能只在一个机器文本风格上成立，
要跨"生成器/风格"稳健。用现有评分集（deepseek-chat 非章回 + 章回体两个寄存器）
分别验证：

- 命题 1 正交性：像人分 vs LLM 质量分 Spearman，在 H 内 / G 内 / GZ 内分别 ≈0
- 命题 2 区分度：像人分区分 H vs G、H vs GZ 的 AUC 都应高（人机可分）
- 命题 3 跨寄存器一致性：G 与 GZ 两个寄存器的正交性/区分度应一致

用法：PYTHONPATH=src PYTHONUTF8=1 python scripts/cross_register_probe.py \
      --config configs/default_macbert.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from narrative_evaluator.config import load_config
from narrative_evaluator.models.discriminator import build_discriminator


def _load_rating_set(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _load_llm_scores(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_macbert.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    items = _load_rating_set("data/eval_dataset/rating_set.jsonl")
    llm = _load_llm_scores("data/eval_dataset/raw/llm_quality_ratings.jsonl")
    by_id = {it["id"]: it for it in items}

    def ids_with(prefixes):
        return [i for i in by_id if i.split("_")[0] in prefixes and i in llm]

    h_ids = ids_with(("H",))
    g_ids = ids_with(("G",))
    gz_ids = ids_with(("GZ",))
    print(f"H={len(h_ids)}  G(非章回)={len(g_ids)}  GZ(章回体)={len(gz_ids)}", flush=True)

    disc = build_discriminator(cfg.discriminator)
    disc.fit([by_id[i]["text"] for i in h_ids + g_ids + gz_ids],
             [by_id[i]["text"] for i in g_ids + gz_ids],
             length_match=True, n_buckets=8, seed=42)

    def like(ids):
        return disc.predict_human_prob([by_id[i]["text"] for i in ids])

    p_h = like(h_ids); p_g = like(g_ids); p_gz = like(gz_ids)
    s_h = np.array([llm[i] for i in h_ids])
    s_g = np.array([llm[i] for i in g_ids])
    s_gz = np.array([llm[i] for i in gz_ids])

    print("\n=== 正交性：像人分 vs LLM 质量分 Spearman（应 ≈0）===")
    for name, p, s in [("H 人类", p_h, s_h), ("G 非章回", p_g, s_g), ("GZ 章回体", p_gz, s_gz)]:
        rho, pv = spearmanr(p, s)
        print(f"  {name:<10} Spearman={rho:+.3f}  (p={pv:.3f}, n={len(p)})")

    print("\n=== 区分度：像人分区分 H vs 各机器寄存器（应高）===")
    for name, p_gx in [("G 非章回", p_g), ("GZ 章回体", p_gz)]:
        auc = roc_auc_score([1] * len(p_h) + [0] * len(p_gx),
                            np.concatenate([p_h, p_gx]))
        print(f"  像人分区分 H vs {name:<9} AUC={auc:.3f}")
    auc_gg = roc_auc_score([1] * len(p_g) + [0] * len(p_gz), np.concatenate([p_g, p_gz]))
    print(f"  像人分区分 G vs GZ（风格差异）AUC={auc_gg:.3f}  (低=风格不可分/高=可区分)")

    print("\n=== 机器两寄存器间的质量分与像人分分布 ===")
    for name, p, s in [("G 非章回", p_g, s_g), ("GZ 章回体", p_gz, s_gz)]:
        print(f"  {name:<10} 像人分 mean={p.mean():.3f}  LLM质量分 mean={s.mean():.2f}")

    print("\n=== 跨寄存器结论 ===")
    print("  若 G 与 GZ 的正交性都 ≈0 且 H-vs-两寄存器区分度都高 → 解耦跨风格稳健。")


if __name__ == "__main__":
    main()
