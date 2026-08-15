"""标签偏差实验分析：人类/LLM 的评判是否被来源标签影响（Label Bias Probe）。

用户提出的实验（2026-08-13）：对同一批评分文本，设置三组对照评分，
研究 judge（LLM 或人类）的评分是否被"来源标签"的先验影响：

- 组 A none    ：无标签盲评（现有 llm_quality_ratings.jsonl，label_mode=none）
- 组 B correct ：给真实来源标签（H→人类作家，G→AI模型）
- 组 C false   ：给颠倒来源标签（H→AI模型，G→人类作家）—— 最有力的偏差检测

核心测量（C vs A）：
- 若 C 组评分"跟着假标签走"（H 被标成 AI 后分数下降、G 被标成人类后分数上升），
  则说明 judge 被来源先验主导，而非文本本身 → 系统性 bias。

用法：
    PYTHONPATH=src python scripts/llm_quality_rating.py --label-mode none --out data/eval_dataset/raw/llm_none.jsonl
    PYTHONPATH=src python scripts/llm_quality_rating.py --label-mode correct --out data/eval_dataset/raw/llm_correct.jsonl
    PYTHONPATH=src python scripts/llm_quality_rating.py --label-mode false --out data/eval_dataset/raw/llm_false.jsonl
    PYTHONUTF8=1 python scripts/label_bias_probe.py \
        --none data/eval_dataset/raw/llm_none.jsonl \
        --correct data/eval_dataset/raw/llm_correct.jsonl \
        --false data/eval_dataset/raw/llm_false.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr, wilcoxon


def _load(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r["score"]
    return out


def _src_of(iid):
    if iid.startswith("H_"):
        return "H"
    if iid.startswith(("G_", "GZ_")):
        return "G"
    return None


def _paired_report(name, a, b, ids):
    """成对对比：mean 差、Spearman、Wilcoxon 检验（是否有系统性偏移）。"""
    x = np.array([a[i] for i in ids])
    y = np.array([b[i] for i in ids])
    diff = y - x
    if len(ids) < 3:
        return f"{name}: n<3，跳过"
    rho, p_rho = spearmanr(x, y)
    try:
        stat, p_w = wilcoxon(x, y)
    except ValueError:
        p_w = float("nan")
    return (f"{name:<18} A均值={x.mean():.2f}  B均值={y.mean():.2f}  "
            f"Δ={diff.mean():+.2f}  Spearman={rho:+.3f}(p={p_rho:.3f})  "
            f"Wilcoxon p={p_w:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--none", required=True, help="无标签评分 jsonl")
    ap.add_argument("--correct", required=True, help="正确标签评分 jsonl")
    ap.add_argument("--false", required=True, help="错误标签评分 jsonl")
    args = ap.parse_args()

    none, correct, false = _load(args.none), _load(args.correct), _load(args.false)
    ids = sorted(set(none) & set(correct) & set(false))
    h_ids = [i for i in ids if _src_of(i) == "H"]
    g_ids = [i for i in ids if _src_of(i) == "G"]
    print(f"三组共有 {len(ids)} 条（H={len(h_ids)} G={len(g_ids)}）\n")

    print("=== 逐组均值（按真实来源）===")
    for name, d in [("none(无标签)", none), ("correct(正确标签)", correct), ("false(错误标签)", false)]:
        sh = np.mean([d[i] for i in h_ids])
        sg = np.mean([d[i] for i in g_ids])
        print(f"  {name:<20} H={sh:.2f}  G={sg:.2f}  Δ(H-G)={sh-sg:+.2f}")

    print("\n=== 成对对比（对同一批文本）===")
    print(f"  {'对比':<18}  A(无标签) vs B(带标签)  →  标签是否改变评分")
    print("  " + _paired_report("correct vs none", none, correct, ids))
    print("  " + _paired_report("false vs none", none, false, ids))

    print("\n=== 偏差方向分解（错误标签组的来源效应）===")
    # 若 H 被标成 AI 后分数下降 → 标签 bias
    dh = np.array([false[i] for i in h_ids]) - np.array([none[i] for i in h_ids])
    dg = np.array([false[i] for i in g_ids]) - np.array([none[i] for i in g_ids])
    print(f"  H 被标成 AI 后评分变化: {dh.mean():+.3f}  (应≈0=无偏；<0=AI标签拉低分)")
    print(f"  G 被标成 人类 后评分变化: {dg.mean():+.3f}  (应≈0=无偏；>0=人类标签抬高)")
    print(f"  [结论] 若两者显著偏离 0 且方向相反 → 系统性标签偏差；若≈0 → 无偏差")

    print("\n=== 标签效应大小（错误标签 vs 无标签的均差绝对值）===")
    effect = abs(dh.mean()) + abs(dg.mean())
    print(f"  |ΔH|+|ΔG| = {effect:.3f}  (越大=标签越影响评分)")


if __name__ == "__main__":
    main()
