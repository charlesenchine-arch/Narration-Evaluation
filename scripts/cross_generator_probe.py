"""跨生成器解耦验证：DeepSeek(deepseek-chat) vs 通义千问(qwen-flash) 两个真·不同生成器。

论文主线（像人分 vs 质量分正交 + 区分度分离）应跨"不同生成器"稳健：
- 命题 1 区分度：像人分区分 H vs G_ds、H vs G_qwen 的 AUC 都应高
- 命题 2 正交性：像人分 vs LLM 质量分 Spearman，H 内 / G_ds 内 / G_qwen 内都应 ≈0
- 命题 3 生成器一致性：两个生成器的正交性方向/强度一致（如都接近 0，或同为弱负相关）

数据：
- H、G_ds：rating_set.jsonl（G_ 前缀 = deepseek-chat 非章回）
- G_qwen：data/eval_dataset/raw/g_qwen.jsonl + 对应 LLM 质量分（llm_quality_rating 产出）
用法：PYTHONPATH=src PYTHONUTF8=1 python scripts/cross_generator_probe.py \
      --config configs/default_macbert.yaml --qwen data/eval_dataset/raw/g_qwen.jsonl \
      --qwen-scores data/eval_dataset/raw/llm_quality_g_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from narrative_evaluator.config import load_config
from narrative_evaluator.models.discriminator import build_discriminator


def _load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _load_scores(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_macbert.yaml")
    ap.add_argument("--qwen", default="data/eval_dataset/raw/g_qwen.jsonl")
    ap.add_argument("--qwen-scores", default="data/eval_dataset/raw/llm_quality_g_qwen.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    items = _load_jsonl("data/eval_dataset/rating_set.jsonl")
    llm = _load_scores("data/eval_dataset/raw/llm_quality_ratings.jsonl")
    by_id = {it["id"]: it for it in items}

    h_ids = [i for i in by_id if i.split("_")[0] == "H" and i in llm]
    gds_ids = [i for i in by_id if i.split("_")[0] == "G" and i in llm]
    qwen_items = _load_jsonl(args.qwen)
    qwen_ids = [it["id"] for it in qwen_items]
    qwen_by_id = {it["id"]: it for it in qwen_items}
    qwen_scores = _load_scores(args.qwen_scores)
    qwen_ids = [i for i in qwen_ids if i in qwen_scores]
    print(f"H={len(h_ids)}  G_ds(deepseek)={len(gds_ids)}  G_qwen={len(qwen_ids)}", flush=True)

    disc = build_discriminator(cfg.discriminator)
    disc.fit([by_id[i]["text"] for i in h_ids + gds_ids] + [qwen_by_id[i]["text"] for i in qwen_ids],
             [by_id[i]["text"] for i in gds_ids] + [qwen_by_id[i]["text"] for i in qwen_ids],
             length_match=True, n_buckets=8, seed=42)

    def like(ids, text_fn):
        return disc.predict_human_prob([text_fn(i) for i in ids])

    p_h = like(h_ids, lambda i: by_id[i]["text"])
    p_ds = like(gds_ids, lambda i: by_id[i]["text"])
    p_qw = like(qwen_ids, lambda i: qwen_by_id[i]["text"])
    s_h = np.array([llm[i] for i in h_ids])
    s_ds = np.array([llm[i] for i in gds_ids])
    s_qw = np.array([qwen_scores[i] for i in qwen_ids])

    print("\n=== 区分度：像人分区分 H vs 各生成器（应高）===")
    for name, p in [("G_ds DeepSeek", p_ds), ("G_qwen 千问", p_qw)]:
        auc = roc_auc_score([1] * len(p_h) + [0] * len(p), np.concatenate([p_h, p]))
        print(f"  像人分区分 H vs {name:<14} AUC={auc:.3f}")
    auc_gg = roc_auc_score([1] * len(p_ds) + [0] * len(p_qw), np.concatenate([p_ds, p_qw]))
    print(f"  像人分区分 G_ds vs G_qwen（生成器差异）AUC={auc_gg:.3f}")

    print("\n=== 正交性：像人分 vs LLM 质量分 Spearman（应 ≈0）===")
    for name, p, s in [("H 人类", p_h, s_h), ("G_ds DeepSeek", p_ds, s_ds), ("G_qwen 千问", p_qw, s_qw)]:
        rho, pv = spearmanr(p, s)
        print(f"  {name:<14} Spearman={rho:+.3f}  (p={pv:.3f}, n={len(p)})")

    print("\n=== 各生成器像人分分布 ===")
    for name, p in [("H", p_h), ("G_ds", p_ds), ("G_qwen", p_qw)]:
        print(f"  {name:<8} mean={p.mean():.3f}  p25={np.percentile(p,25):.3f}  p75={np.percentile(p,75):.3f}")

    print("\n=== 跨生成器结论 ===")
    print("  若 H-vs-两生成器区分度都高、且两生成器内正交性都≈0 → 解耦跨生成器稳健。")


if __name__ == "__main__":
    main()
