"""探针：窗口化 S_repr 能否消除"末尾注水"对表示视图的虚涨。

假设：append_filler 在末尾追加通用填充句，把整篇 bge 向量拉向 H 质心，
导致 S_repr 从 0.084 虚涨到 0.288。若 S_repr 只对"前 N 字符窗口"计算，
则末尾填充不影响窗口内容 → S_repr 应基本持平。

对比三种计算方式（对 G 原文 / 1.5x / 3.0x append_filler）：
  - full    : 全文本 S_repr
  - prefix400: 前 400 字符
  - mid400   : 从中间取 400 字符
用法：HF_HUB_OFFLINE=1 PYTHONPATH=src PYTHONUTF8=1 python scripts/probe_window_repr.py
"""
from __future__ import annotations

import json

import numpy as np

from narrative_evaluator.config import load_config
from narrative_evaluator.evaluator import Evaluator

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


def _pad_append(text, ratio):
    target = int(len(text) * ratio)
    out = text
    i = 0
    while len(out) < target:
        out += _FILLER[i % len(_FILLER)]
        i += 1
    return out


def main():
    cfg = load_config("configs/default_macbert.yaml")
    ev = Evaluator(cfg)
    # 用评分集拟合（只做表示参照，不需要判别器重训）
    items = [json.loads(l) for l in open("data/eval_dataset/rating_set.jsonl", encoding="utf-8")]
    h = [it["text"] for it in items if it["id"].startswith("H_")]
    g = [it["text"] for it in items if it["id"].startswith(("G_", "GZ_"))]
    rng = np.random.RandomState(42)
    gs = list(rng.choice(g, size=min(40, len(g)), replace=False))
    ev.scorer.h_repr = ev.doc_encoder.encode_documents(h[:150])
    ev.scorer._fit_repr_ref()

    def s_repr(texts):
        out = []
        for t in texts:
            vecs = ev.doc_encoder.encode_documents([t])
            sim = float((ev.scorer.h_repr @ vecs[0]).max())
            out.append(float(np.searchsorted(np.sort(ev.scorer.h_repr_sims), sim, side="right") / ev.scorer.h_repr_sims.size))
        return out

    def window(t, start, ln):
        return t[start:start + ln] if len(t) > start + 50 else t

    def mid(t, ln):
        s = max(0, (len(t) - ln) // 2)
        return t[s:s + ln]

    groups = {"G 原文": gs, "G 1.5x filler": [_pad_append(t, 1.5) for t in gs],
              "G 3.0x filler": [_pad_append(t, 3.0) for t in gs]}
    print(f"{'组':<16}{'full':>8}{'prefix400':>12}{'mid400':>10}{'len均值':>10}")
    for name, texts in groups.items():
        f = np.mean(s_repr(texts))
        p = np.mean(s_repr([window(t, 0, 400) for t in texts]))
        m = np.mean(s_repr([mid(t, 400) for t in texts]))
        print(f"{name:<16}{f:>8.3f}{p:>12.3f}{m:>10.3f}{np.mean([len(t) for t in texts]):>10.0f}")


if __name__ == "__main__":
    main()
