"""生成锚定文本（极好 + 极差），掺入评分集防止审美疲劳。

极好：WebNovelBench 高分章节切块（人类评委认可的好叙事）。
极差：构造的碎片化/重复废话文本（真差，让评分者有对照）。
标记 source='anchor_high'/'anchor_low'，评分时混入但分析时可按 source 区分。

用法：python scripts/generate_anchors.py --n-high 50 --n-low 50 --out data/eval_dataset/raw/anchors.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile

sys.path.insert(0, "src")
from narrative_evaluator.config import DataConfig
from narrative_evaluator.data import load_webnovelbench


def chunk_text(text: str, lo: int = 300, hi: int = 800) -> list:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) > hi and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if lo <= len(c) <= hi]


def make_low_quality(rng, lo=300, hi=800):
    """构造一个真差的文本：碎片化、无逻辑、重复废话、句子破碎。"""
    templates = [
        "今天天气很好。今天天气真的很好。天气好天气好天气好。我不知道要说什么。反正天气就是很好。",
        "他说。她说。他说。她说。他说她说他说她说。没有人知道发生了什么。反正就是这样。",
        "这个很重要。那个也很重要。所有事情都很重要。但是什么是最重要的呢。谁也不知道。",
        "我起床了。然后我吃了饭。然后我出门了。然后我又回来了。然后我不知道该做什么。",
        "第一章。第二章。第三章。第四章。第五章。第六章。故事没有结局。因为作者不想写了。",
    ]
    parts = []
    while sum(len(p) for p in parts) < lo:
        parts.append(rng.choice(templates))
    text = "\n".join(parts)
    return text[:hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-high", type=int, default=50)
    ap.add_argument("--n-low", type=int, default=50)
    ap.add_argument("--out", default="data/eval_dataset/raw/anchors.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cache = tempfile.mkdtemp(prefix="hf_anchor_")
    cfg = DataConfig(seed=args.seed)
    print("加载 WebNovelBench 找高分章节...", flush=True)
    h_web = load_webnovelbench(cfg, cache)
    records = sorted(h_web.records, key=lambda r: r.human_score or 0, reverse=True)

    # 极好：top 章节切块
    top = records[:20]
    high_chunks = []
    for r in top:
        high_chunks.extend(chunk_text(r.text))
    print(f"  高分章节切块: {len(high_chunks)} 块", flush=True)

    # 极差：构造
    low_texts = [make_low_quality(rng) for _ in range(args.n_low)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, c in enumerate(rng.sample(high_chunks, min(args.n_high, len(high_chunks)))):
            f.write(json.dumps({"id": f"AH_{i:03d}", "text": c, "label": "human",
                                "source": "anchor_high"}, ensure_ascii=False) + "\n")
        for i, t in enumerate(low_texts):
            f.write(json.dumps({"id": f"AL_{i:03d}", "text": t, "label": "machine",
                                "source": "anchor_low"}, ensure_ascii=False) + "\n")
    print(f"已写入: 极好 {min(args.n_high, len(high_chunks))} + 极差 {args.n_low} 到 {args.out}", flush=True)


if __name__ == "__main__":
    main()
