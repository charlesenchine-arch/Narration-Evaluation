"""B1 加固探针：扫 α（严苛度强度）看攻防平衡，不重复计算三视图分量。

对同一批评分集拟合评估器（MacBERT 部署配置），一次性计算各注水变体的
三视图分量（S_disc/S_repr/S_attr），然后对多个 α 评估：

  攻击侧（G）：
    - append_filler 1.5x：最终分相对原文本的变化（应 ≤0，否则攻击成立）
    - append_filler 3.0x：同上（应明显为负）
    - self_repeat 3.0x：同上（应明显为负）
  防御侧（H）：
    - 自然长文（两段不同 H 拼接，~2x）：最终分变化（应温和，> -15% 可接受）
    - 注水 H 3.0x：内容污染应被压低（负变化属预期）

用法：PYTHONPATH=src PYTHONUTF8=1 python scripts/strictness_harden_probe.py \
      --config configs/default_macbert.yaml
"""
from __future__ import annotations

import argparse
import json
from statistics import mean

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


def _load_rating_set(path):
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    h = [it["text"] for it in items if it["id"].startswith("H_")]
    g = [it["text"] for it in items if it["id"].startswith(("G_", "GZ_"))]
    return h, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_macbert.yaml")
    ap.add_argument("--alphas", default="0.2,0.35,0.5,0.65,0.8")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]

    cfg = load_config(args.config)
    ev = Evaluator(cfg)

    data_path = "data/eval_dataset/rating_set.jsonl"
    print("加载评分集...", flush=True)
    h_all, g_all = _load_rating_set(data_path)
    print(f"  H={len(h_all)}  G={len(g_all)}", flush=True)

    rng = np.random.RandomState(42)
    g_sample = list(rng.choice(g_all, size=min(60, len(g_all)), replace=False))
    h_sample = list(rng.choice(h_all, size=min(40, len(h_all)), replace=False))

    print("拟合评估器（部署配置）...", flush=True)
    ev.fit(h_all, g_all)
    L0 = ev.scorer.strict_len_base
    print(f"  L0(严苛度基准)=H中位长度 {L0}", flush=True)
    lam = ev.scorer.lambdas

    def components(texts):
        cs = ev.scorer.components_batch(texts)
        return [(c["S_disc"], c["S_repr"], c["S_attr"]) for c in cs]

    # 一次性计算各变体的 (S_disc, S_repr, S_attr)
    print("计算三视图分量...", flush=True)
    c_g_orig = components(g_sample)
    c_g_f15 = components([_pad_append(t, 1.5) for t in g_sample])
    c_g_f30 = components([_pad_append(t, 3.0) for t in g_sample])
    c_g_sr30 = components([_pad_selfrepeat(t, 3.0) for t in g_sample])
    c_h_orig = components(h_sample)
    # H 自然长文：两两拼接不同 H 块（~2x，内容多样）
    rng_h = np.random.RandomState(7)
    h_shuf = list(rng_h.permutation(len(h_sample)))
    h_natural = [h_sample[h_shuf[i]] + h_sample[h_shuf[i + len(h_sample) // 2]]
                 for i in range(len(h_sample) // 2)]
    c_h_nat = components(h_natural)
    c_h_w30 = components([_pad_append(t, 3.0) for t in h_sample])

    def like(c):
        return float(np.clip(lam[0] * c[0] + lam[1] * c[1] + lam[2] * c[2], 0.0, 1.0))

    def final(c, alpha, llen):
        kappa = 1.0 + alpha * max(0, llen - L0) / max(1, L0)
        return like(c) ** kappa

    lens_g = [len(t) for t in g_sample]
    lens_g15 = [len(_pad_append(t, 1.5)) for t in g_sample]
    lens_g30 = [len(_pad_append(t, 3.0)) for t in g_sample]
    lens_h = [len(t) for t in h_sample]
    lens_h_nat = [len(t) for t in h_natural]
    lens_h_w30 = [len(_pad_append(t, 3.0)) for t in h_sample]

    print("\n=== 三视图分量（均值）===")
    print(f"  {'变体':<20}{'S_disc':>8}{'S_repr':>8}{'S_attr':>8}")
    for name, cs in [("G 原文", c_g_orig), ("G 1.5x filler", c_g_f15),
                     ("G 3.0x filler", c_g_f30), ("G 3.0x selfrepeat", c_g_sr30),
                     ("H 原文", c_h_orig), ("H 自然长文", c_h_nat), ("H 注水 3.0x", c_h_w30)]:
        d, r, a = [mean(x[i] for x in cs) for i in range(3)]
        print(f"  {name:<20}{d:>8.3f}{r:>8.3f}{a:>8.3f}")

    print("\n=== α 扫描：最终分变化（相对各自基准）===")
    print(f"  {'α':>5} | {'G 1.5x filler':>16}{'G 3.0x filler':>16}{'G 3.0x selfrepeat':>18} | {'H 自然长文(~2x)':>16}{'H 注水3.0x':>14}")
    for alpha in alphas:
        f_g0 = [final(c, alpha, l) for c, l in zip(c_g_orig, lens_g)]
        f_g15 = [final(c, alpha, l) for c, l in zip(c_g_f15, lens_g15)]
        f_g30 = [final(c, alpha, l) for c, l in zip(c_g_f30, lens_g30)]
        f_sr30 = [final(c, alpha, l) for c, l in zip(c_g_sr30, lens_g30)]
        f_h0 = [final(c, alpha, l) for c, l in zip(c_h_orig, lens_h)]
        f_h_nat = [final(c, alpha, l) for c, l in zip(c_h_nat, lens_h_nat)]
        f_h_w30 = [final(c, alpha, l) for c, l in zip(c_h_w30, lens_h_w30)]
        d15 = (mean(f_g15) - mean(f_g0)) / mean(f_g0) * 100
        d30 = (mean(f_g30) - mean(f_g0)) / mean(f_g0) * 100
        dsr = (mean(f_sr30) - mean(f_g0)) / mean(f_g0) * 100
        dhn = (mean(f_h_nat) - mean(f_h0)) / mean(f_h0) * 100
        dhw = (mean(f_h_w30) - mean(f_h0)) / mean(f_h0) * 100
        flag = "[OK]" if d15 <= 0 else "[FAIL]"
        print(f"  {alpha:>5.2f} | {d15:>+13.1f}%{flag}{d30:>+13.1f}%{dsr:>+15.1f}% | {dhn:>+13.1f}%{dhw:>+11.1f}%")

    print("\n  解读：")
    print("    - G 侧三列应 ≤0%（堵住注水攻击）；越负越安全")
    print("    - H 自然长文应尽量温和（> -15%）——否则严苛度过度惩罚人类长文")
    print("    - H 注水 3.0x 负得多是预期的（内容被污染）")
    print("\n  最终选择原则：取最小 α 满足『G 1.5x filler ≤ 0 且 H 自然长文 > -15%』。")


if __name__ == "__main__":
    main()
