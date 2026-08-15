"""从长叙事中切出 300-800 字的短文块（临时数据集方案）。

人类短文叙事（H）的补充来源：把 WebNovelBench 章节/CNNSum 摘录按段落边界
切分成 300-800 字的连续块，保留自然段落边界。切块可作正式样本（用户确认）。

用法：python scripts/chunk_short_stories.py --out data/eval_dataset/raw/h_chunks.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile

sys.path.insert(0, "src")
from narrative_evaluator.config import DataConfig
from narrative_evaluator.data import load_webnovelbench, load_cnnsum
from narrative_evaluator.data.chunking import chunk_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval_dataset/raw/h_chunks.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lo", type=int, default=300)
    ap.add_argument("--hi", type=int, default=800)
    args = ap.parse_args()

    cache = tempfile.mkdtemp(prefix="hf_chunk_")
    cfg = DataConfig(seed=args.seed)
    print("加载 WebNovelBench + CNNSum...", flush=True)
    h_web = load_webnovelbench(cfg, cache)
    h_cnn = load_cnnsum(cfg, cache)
    all_texts = h_web.texts + h_cnn.texts
    print(f"  文本数: {len(all_texts)}", flush=True)

    # 从所有文本中收集达标块
    all_chunks = []
    for src, t in [("webnovelbench", t) for t in h_web.texts] + [("cnnsum", t) for t in h_cnn.texts]:
        for c in chunk_text(t, args.lo, args.hi):
            all_chunks.append({"text": c, "label": "human", "source": src})

    print(f"  达标块总数: {len(all_chunks)}", flush=True)
    rng = random.Random(args.seed)
    sampled = rng.sample(all_chunks, min(args.n, len(all_chunks)))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, item in enumerate(sampled):
            item["id"] = f"H_{i:04d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  已写入 {len(sampled)} 条到 {args.out}", flush=True)


if __name__ == "__main__":
    main()
