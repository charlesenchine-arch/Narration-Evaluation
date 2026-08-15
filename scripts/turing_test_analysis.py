"""评估者图灵测试分析骨架（协议：docs/evaluator_turing_test_protocol.md v0.1）。

验证评估器的"像人分"判断是否**像人类评委**：
- S1 一致性（主检验）：评估器像人分排序 vs 人类像人判断均值排序的 Spearman，
  以人类-人类一致性 CI 为基准上限。逐寄存器报告（规避 H/G 可混分坑）。
- S2 判别（ABX）：裁判对 (文本, 判断A, 判断B) 判"哪个是真人给的"，检测率≈50% 通过。
- S3 方向性 + LLM-as-Judge 基线：逐寄存器均值对比；DeepSeek 质量分 vs 人类像人判断
  跑同样的 S1，作为"机器判断像人"的既有基线。

数据源：
- 评分集     data/eval_dataset/rating_set.jsonl     {id, text, label, source}
- 人类评分    data/eval_dataset/rating_state.json   state["likes"]：像人滑块 0-100（每篇每评分者）
- LLM 质量分  data/eval_dataset/raw/llm_quality_ratings.jsonl（S3 基线，可缺省）
- ABX 裁判数据 --abx（S2，收集阶段才存在；现在缺省跳过）

评估器像人分：用部署配置（MacBERT + 窗口化 + α=0.5）在评分集 H/G 上 fit 参照后，
对全部条目计算三视图 like 分（S_disc/S_repr/S_attr 融合）。GPU 可用时自动使用（--device auto）。

用法：
    PYTHONPATH=src python scripts/turing_test_analysis.py \
        [--config configs/default_deploy.yaml] [--device auto] \
        [--state data/eval_dataset/rating_state.json] \
        [--abx data/eval_dataset/raw/abx_judgments.jsonl] [--out turing_report.json]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, spearmanr

from narrative_evaluator.config import load_config
from narrative_evaluator.evaluator import Evaluator

REGISTER_LABELS = {
    "H":  "H 人类网文切块",
    "G":  "G 机器(DeepSeek 非章回)",
    "GZ": "GZ 机器(DeepSeek 章回体)",
    "AH": "锚定·极好(对照)",
    "AL": "锚定·极差(对照)",
}
NON_ANCHOR = ("H", "G", "GZ")


def _register(item_id: str) -> str:
    return str(item_id).split("_")[0]


def _load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _human_like_stats(state: dict, item_ids: list) -> dict:
    """从 state["likes"] 汇总每篇像人判断：{id: {mean, n, raters}}。"""
    likes = state.get("likes") or {}
    out = {}
    for iid in item_ids:
        vals = {}
        for rater, d in likes.items():
            v = (d or {}).get(iid)
            if v is not None:
                vals[rater] = float(v)
        if vals:
            arr = list(vals.values())
            out[iid] = {"mean": float(np.mean(arr)), "n": len(arr), "raters": vals}
    return out


def _resolve_encoder_device(device: str) -> str:
    """sentence-transformers 不认 "auto"，需解析成 cuda/cpu。"""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _compute_eval_scores(items: list, cfg) -> dict:
    """fit 部署评估器（MacBERT 幂等加载不重训）→ 全部条目三视图 like/final 分。

    返回 {id: {S_disc, S_repr, S_attr, like, final, len}}。
    like = 三视图融合像人分（S1 用，无长度衰减）；final = 严苛度衰减后的部署分。
    """
    by_id = {it["id"]: it for it in items}
    h_ids = [i for i in by_id if _register(i) == "H"]
    g_ids = [i for i in by_id if _register(i) in ("G", "GZ")]
    if not h_ids or not g_ids:
        raise SystemExit("评分集缺少 H 或 G 条目，无法 fit 参照分布")
    print(f"fit 评估器参照: H={len(h_ids)} G={len(g_ids)} ...", flush=True)
    ev = Evaluator(cfg)
    ev.fit([by_id[i]["text"] for i in h_ids], [by_id[i]["text"] for i in g_ids])
    lam = ev.scorer.lambdas
    comps = ev.scorer.components_batch([by_id[i]["text"] for i in by_id])
    out = {}
    for it, c in zip(items, comps):
        like = float(np.clip(
            lam[0] * c["S_disc"] + lam[1] * c["S_repr"] + lam[2] * c["S_attr"], 0, 1))
        kappa = ev.scorer.strictness(len(it["text"]))
        out[it["id"]] = {**c, "like": like,
                         "final": float(np.clip(like ** kappa, 0, 1)),
                         "len": len(it["text"])}
    return out


def _s1_block(eval_scores: dict, human: dict, ids: list, title: str) -> dict | None:
    """S1 一致性：评估器 like vs 人类像人判断均值 Spearman。"""
    ids = [i for i in ids if i in eval_scores and i in human]
    if len(ids) < 3:
        print(f"  {title}: 像人判断样本不足(n={len(ids)}，需≥3)，跳过", flush=True)
        return None
    ev = np.array([eval_scores[i]["like"] for i in ids])
    hm = np.array([human[i]["mean"] for i in ids])
    rho, p = spearmanr(ev, hm)
    print(f"  {title}: Spearman={rho:+.3f} (p={p:.3f}, n={len(ids)})", flush=True)
    return {"n": len(ids), "rho": float(rho), "p": float(p)}


def _human_consistency(state: dict, min_shared: int = 3) -> list:
    """人类-人类一致性上限：两两评分者在共同条目上像人判断的 Spearman。"""
    likes = state.get("likes") or {}
    raters = sorted(k for k, d in likes.items() if d)
    rhos = []
    for a in range(len(raters)):
        for b in range(a + 1, len(raters)):
            ra, rb = likes[raters[a]] or {}, likes[raters[b]] or {}
            shared = [i for i in ra if i in rb]
            if len(shared) < min_shared:
                continue
            rho, _ = spearmanr([ra[i] for i in shared], [rb[i] for i in shared])
            rhos.append(float(rho))
    return rhos


def _s3_llm_baseline(llm_path: str, human: dict, eval_scores: dict, ids: list) -> dict | None:
    """LLM-as-Judge 基线：DeepSeek 质量分 vs 人类像人判断均值（同 S1 流程）。"""
    if not llm_path or not os.path.exists(llm_path):
        print("  (无 llm_quality_ratings.jsonl，跳过 LLM 基线)", flush=True)
        return None
    llm = {r["id"]: r["score"] for r in _load_jsonl(llm_path)}
    ids = [i for i in ids if i in llm and i in human and i in eval_scores]
    if len(ids) < 3:
        print(f"  LLM 基线: 样本不足(n={len(ids)})，跳过", flush=True)
        return None
    rho, p = spearmanr([llm[i] for i in ids], [human[i]["mean"] for i in ids])
    print(f"  LLM 质量分 vs 人类像人判断: Spearman={rho:+.3f} (p={p:.3f}, n={len(ids)})",
          flush=True)
    return {"n": len(ids), "rho": float(rho), "p": float(p)}


def _s2_abx(path: str) -> None:
    """S2 判别（ABX）：逐寄存器检测率 + 二项检验 H0: p=0.5。

    数据格式（jsonl）：{id, text_id, register, judge, answer, truth}，
    answer/truth ∈ {"A","B"}，answer 为裁判所选，truth 为真人判断真实所在侧。
    """
    rows = _load_jsonl(path)
    if not rows:
        print("  (无 ABX 数据)", flush=True)
        return
    by_reg = defaultdict(list)
    for r in rows:
        by_reg[r.get("register") or _register(r["text_id"])].append(r)
    print("\n=== S2 判别（ABX）===", flush=True)
    tot_k = tot_n = 0
    for reg in sorted(by_reg):
        rs = by_reg[reg]
        n = len(rs)
        corr = sum(1 for r in rs if r["answer"] == r["truth"])
        bt = binomtest(corr, n, p=0.5)
        rate = corr / n
        verdict = "≈50% 通过" if abs(rate - 0.5) < 0.1 else ""
        print(f"  {REGISTER_LABELS.get(reg, reg)}: 检测率={rate:.3f} ({corr}/{n}) "
              f"binom p={bt.pvalue:.3f} {verdict}", flush=True)
        tot_k += corr
        tot_n += n
    if tot_n:
        bt = binomtest(tot_k, tot_n, p=0.5)
        print(f"  合计: 检测率={tot_k / tot_n:.3f} ({tot_k}/{tot_n}) "
              f"binom p={bt.pvalue:.3f}", flush=True)


def _build_abx(out_path: str, human: dict, eval_scores: dict, items_by_id: dict,
               seed: int = 42) -> None:
    """生成 S2 收集用三元组刺激文件：逐篇配对 人类像人均值 vs 校准后的评估器 like。

    校准：把评估器 like 分位映射到人类 0-100 标尺，避免"量纲不同被裁判一眼看穿"。
    输出 jsonl：{id, text_id, register, text, a, b, truth}，judge/answer 留待裁判填。
    """
    rng = np.random.RandomState(seed)
    ids = sorted(i for i in human if i in eval_scores)
    like_sorted = np.sort([eval_scores[i]["like"] for i in ids])
    def calibrate(v):
        return float(np.searchsorted(like_sorted, v, side="right") / like_sorted.size * 100)
    rows = []
    for iid in ids:
        a_val = human[iid]["mean"]
        b_val = calibrate(eval_scores[iid]["like"])
        truth = "A"
        if rng.rand() < 0.5:
            a_val, b_val = b_val, a_val
            truth = "B"
        rows.append({
            "id": f"ABX_{len(rows):03d}", "text_id": iid, "register": _register(iid),
            "text": items_by_id[iid]["text"][:200],  # 展示窗口，避免裁判读全文耗时
            "a": round(a_val, 1), "b": round(b_val, 1), "truth": truth,
            "judge": "", "answer": "",
        })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"ABX 刺激已写入 {out_path}（{len(rows)} 组；a/b 已打乱，truth 字段收集后需隐藏）",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="评估者图灵测试分析")
    ap.add_argument("--config", default="configs/default_deploy.yaml",
                    help="部署配置（MacBERT + 窗口化 + α=0.5）")
    ap.add_argument("--device", default="auto",
                    help="编码器 device（auto=有 GPU 用 GPU，否则 cpu）")
    ap.add_argument("--rating-set", default="data/eval_dataset/rating_set.jsonl")
    ap.add_argument("--state", default="data/eval_dataset/rating_state.json")
    ap.add_argument("--llm", default="data/eval_dataset/raw/llm_quality_ratings.jsonl",
                    help="S3 LLM 基线；给空串跳过")
    ap.add_argument("--abx", default=None, help="S2 裁判数据 jsonl（可选，缺省跳过）")
    ap.add_argument("--build-abx-out", default=None,
                    help="生成 S2 收集用刺激文件路径（不跑分析，只生成）")
    ap.add_argument("--out", default=None, help="JSON 结果落盘路径（可选）")
    args = ap.parse_args()

    items = _load_jsonl(args.rating_set)
    state = json.load(open(args.state, encoding="utf-8"))
    items_by_id = {it["id"]: it for it in items}
    print(f"评分集 {len(items)} 条", flush=True)

    human = _human_like_stats(state, list(items_by_id))
    print(f"人类像人判断: {len(human)} 篇有分"
          f"（{sum(1 for h in human.values() if h['n'] >= 2)} 篇≥2人）", flush=True)
    if not human:
        print("\n还没有人类像人判断数据（state['likes'] 为空）。", flush=True)
        print("等评分者用像人滑块评分后重跑。届时可先 --build-abx-out 生成 S2 刺激。",
              flush=True)
        return

    cfg = load_config(args.config) if os.path.exists(args.config) else None
    if cfg is None:
        from narrative_evaluator.config import default_config
        cfg = default_config()
    cfg.encoder.device = _resolve_encoder_device(args.device)
    print(f"编码器 device = {cfg.encoder.device}", flush=True)

    if args.build_abx_out:
        eval_scores = _compute_eval_scores(items, cfg)
        _build_abx(args.build_abx_out, human, eval_scores, items_by_id)
        return

    eval_scores = _compute_eval_scores(items, cfg)
    print(f"评估器 like 分: {len(eval_scores)} 篇", flush=True)

    report = {"n_human_like": len(human), "device": cfg.encoder.device}

    # ---- S1 一致性（主检验）----
    print("\n=== S1 一致性（评估器 like vs 人类像人判断均值）===", flush=True)
    s1 = {}
    for reg in NON_ANCHOR + ("AH", "AL"):
        ids = [i for i in items_by_id if _register(i) == reg]
        r = _s1_block(eval_scores, human, ids, f"{reg} {REGISTER_LABELS[reg]}")
        if r:
            s1[reg] = r
    report["s1"] = s1

    # 人类-人类一致性基准（上限）
    print("\n=== 基准：人类-人类一致性（共同条目上两两 Spearman）===", flush=True)
    rhos = _human_consistency(state)
    if rhos:
        print(f"  共 {len(rhos)} 对：mean={np.mean(rhos):+.3f} "
              f"range=[{min(rhos):+.3f}, {max(rhos):+.3f}]", flush=True)
        report["human_human_rho"] = {"n_pairs": len(rhos), "mean": float(np.mean(rhos))}
    else:
        print("  暂无 ≥2 名评分者共享 ≥3 篇像人判断，无法计算；随评分增加自动补上",
              flush=True)

    # ---- S3 方向性 + LLM 基线 ----
    print("\n=== S3 方向性（逐寄存器均值：人类像人分 vs 评估器 like）===", flush=True)
    for reg in NON_ANCHOR + ("AH", "AL"):
        ids = [i for i in items_by_id if _register(i) == reg and i in human]
        if len(ids) < 1:
            continue
        hm = np.mean([human[i]["mean"] for i in ids])
        ev = np.mean([eval_scores[i]["like"] for i in ids])
        print(f"  {reg} {REGISTER_LABELS[reg]}: 人类={hm:5.1f}/100  评估器={ev:.3f}  (n={len(ids)})",
              flush=True)
    print("  [LLM 基线] DeepSeek 质量分 vs 人类像人判断（S1 同流程）", flush=True)
    s3_llm = _s3_llm_baseline(args.llm, human, eval_scores, list(items_by_id))
    if s3_llm:
        report["s3_llm"] = s3_llm

    # ---- S2 判别（ABX）----
    print("\n=== S2 判别（ABX）===", flush=True)
    if args.abx and os.path.exists(args.abx):
        _s2_abx(args.abx)
    else:
        print("  暂无裁判数据；收集时：--build-abx-out 生成刺激 → 裁判填 answer → --abx 分析",
              flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
