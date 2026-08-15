"""探针：句级自相似度能否干净区分"注水填充" vs "自然长文"。

对 G 注水变体 / H 原文 / H 自然长文分别计算句级自相似度（bge 句向量
两两最大余弦的均值），看是否：
- 注水填充（append_filler / self_repeat）→ 高自相似
- H 自然长文（拼接不同块）→ 低自相似
若可分，则可用作 ρ 惩罚（final *= ρ）堵住轻注水，且不伤 H。

用法：PYTHONPATH=src PYTHONUTF8=1 python scripts/probe_self_similarity.py
"""
from __future__ import annotations

import json
import re

import numpy as np

from narrative_evaluator.config import load_config
from narrative_evaluator.evaluator import Evaluator


def _pad_append(text, ratio):
    _FILLER = [
        "他慢慢地走着，脑海里不断浮现出过去的画面，那些画面渐渐模糊，又渐渐清晰。",
        "风从远处吹来，带着草木的气息，他深吸一口气，心里说不出是什么滋味。",
        "时间一分一秒地过去，一切都显得那么平静，仿佛什么都不会发生，又仿佛什么都即将发生。",
        "他抬起头望着天空，云层很厚，阳光透不出来，像极了此刻他复杂的心情。",
        "许多事情就是这样，说不上为什么，也说不清结果，只是那样发生了，就再也回不去了。",
        "远处的灯火明明灭灭，他看着看着，忽然觉得有些恍惚，像是隔着一层雾在看另一个世界。",
        "夜越来越深了，四周安静下来，只剩下自己的呼吸声，还有那一直悬在心头的念头。",
        "他坐了下来，随手拿起一样东西又放下，心里乱糟糟的，一时竟不知道该做什么才好。",
    ]
    target = int(len(text) * ratio)
    out = text
    i = 0
    while len(out) < target:
        out += _FILLER[i % len(_FILLER)]
        i += 1
    return out


def _pad_selfrepeat(text, ratio):
    target = int(len(text) * ratio)
    parts = [p.strip() for p in text.replace("\n", "。").split("。") if p.strip()] or [text]
    out = text
    i = 0
    while len(out) < target and parts:
        out += parts[i % len(parts)] + "。"
        i += 1
    return out


def _sentences(text):
    parts = [p.strip() for p in re.split(r"[。！？!?；\n]", text) if len(p.strip()) >= 6]
    return parts or [text]


def _self_sim(sents, encoder, k=3):
    """句子两两余弦，取每句 top-k 最大相似度的均值（不含自身）。"""
    if len(sents) < 2:
        return 1.0
    vecs = np.asarray(encoder.encode_documents(sents))  # [n, d] L2 归一化
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(k, len(sents) - 1)
    topk = np.sort(sims, axis=1)[:, -kk:]
    return float(topk.mean())


def main():
    cfg = load_config("configs/default_macbert.yaml")
    ev = Evaluator(cfg)
    enc = ev.doc_encoder

    items = [json.loads(l) for l in open("data/eval_dataset/rating_set.jsonl", encoding="utf-8")]
    h = [it["text"] for it in items if it["id"].startswith("H_")]
    g = [it["text"] for it in items if it["id"].startswith(("G_", "GZ_"))]
    rng = np.random.RandomState(42)
    gs = list(rng.choice(g, size=min(60, len(g)), replace=False))
    hs = list(rng.choice(h, size=min(40, len(h)), replace=False))
    rng_h = np.random.RandomState(7)
    shuf = list(rng_h.permutation(len(hs)))
    h_nat = [hs[shuf[i]] + hs[shuf[i + len(hs) // 2]] for i in range(len(hs) // 2)]

    groups = {
        "G 原文": gs,
        "G 1.5x filler": [_pad_append(t, 1.5) for t in gs],
        "G 3.0x filler": [_pad_append(t, 3.0) for t in gs],
        "G 3.0x selfrepeat": [_pad_selfrepeat(t, 3.0) for t in gs],
        "H 原文": hs,
        "H 自然长文": h_nat,
        "H 注水 3.0x": [_pad_append(t, 3.0) for t in hs],
    }
    print(f"{'组':<20}{'句数均值':>10}{'自相似 top3':>14}{'自相似 top1':>14}")
    for name, texts in groups.items():
        ss3, ss1, ns = [], [], []
        for t in texts:
            s = _sentences(t)
            if len(s) < 2:
                continue
            v = np.asarray(enc.encode_documents(s))
            sim = v @ v.T
            np.fill_diagonal(sim, -np.inf)
            ss3.append(float(np.sort(sim, axis=1)[:, -min(3, len(s) - 1):].mean()))
            ss1.append(float(np.sort(sim, axis=1)[:, -1].mean()))
            ns.append(len(s))
        print(f"{name:<20}{np.mean(ns):>10.0f}{np.mean(ss3):>14.3f}{np.mean(ss1):>14.3f}")


if __name__ == "__main__":
    main()
