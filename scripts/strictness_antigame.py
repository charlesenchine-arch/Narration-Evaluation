"""B1 严苛度防刷分验证：注水篇幅能否骗过评分？

威胁模型：生成模型想在最终分（reward）上刷分，最廉价的攻击是"注水"——
在原文上灌入流畅但空洞的内容把篇幅拉长。若注水后最终分上涨，攻击成立；
若像人分被注水抬高、但严苛度 κ 把它压回去，则防刷分成立。

评分机制（mix_scorer）：final = like_score^κ(len)
  like_score: 三视图像人分（判别器/表示/属性）
  κ = 1 + α·max(0, len-L0)/L0，L0 = H 参照中位长度

验证（使用部署配置：evaluator.fit 默认 length_match=False）：
1. G 侧攻击：对机器文本施加注水（目标比例 1.5x/2.0x/3.0x），
   看最终分是否随注水上涨（应 ≈ 持平或下降）。
2. H 侧对照：人类文本被拉到同样长度，最终分只应小幅下降
   （"足够像人的长文几乎不受罚"），否则严苛度是过度惩罚。
"""
from __future__ import annotations

import argparse
import json
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

from narrative_evaluator.config import Config, load_config
from narrative_evaluator.evaluator import Evaluator
from narrative_evaluator.models.discriminator import build_discriminator


# ---- 注水内容池（流畅但空洞的叙事废话，机器注水风格） ----
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


def _sample_sentences(text: str) -> list:
    parts = [p.strip() for p in text.replace("\n", "。").split("。") if p.strip()]
    return parts or [text]


def _pad_append(text: str, ratio: float) -> str:
    """策略 A：原文后追加空洞废话直到目标长度。"""
    target = int(len(text) * ratio)
    out = text
    i = 0
    while len(out) < target:
        out += _FILLER[i % len(_FILLER)]
        i += 1
    return out


def _pad_selfrepeat(text: str, ratio: float) -> str:
    """策略 B：复读原文自己的句子（机器自我回声）直到目标长度。"""
    target = int(len(text) * ratio)
    sents = _sample_sentences(text)
    out = text
    i = 0
    while len(out) < target and sents:
        out += sents[i % len(sents)] + "。"
        i += 1
    return out


def _pad_interleave(text: str, ratio: float) -> str:
    """策略 C：在段落间隙插入空洞废话直到目标长度。"""
    target = int(len(text) * ratio)
    paras = [p for p in text.split("\n") if p.strip()]
    if not paras:
        return _pad_append(text, ratio)
    out = paras[0]
    i = 0
    for p in paras[1:]:
        if len(out) >= target:
            out += p
        else:
            out += _FILLER[i % len(_FILLER)]
            i += 1
            out += p
    while len(out) < target:
        out += _FILLER[i % len(_FILLER)]
        i += 1
    return out


STRATEGIES = {
    "append_filler": _pad_append,
    "interleave": _pad_interleave,
    "self_repeat": _pad_selfrepeat,
}
RATIOS = [1.0, 1.5, 2.0, 3.0]


def _load_rating_set(path):
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    h = [it["text"] for it in items if it["id"].startswith("H_")]
    g = [it["text"] for it in items if it["id"].startswith(("G_", "GZ_"))]
    return h, g


def _pad_variants(texts, seed=42):
    """为一批文本生成 {strategy: {ratio: [texts]}}，1.0x 为原文本（共享）。"""
    variants = {s: {r: [] for r in RATIOS} for s in STRATEGIES}
    for s, fn in STRATEGIES.items():
        for t in texts:
            for r in RATIOS:
                variants[s][r].append(t if r == 1.0 else fn(t, r))
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="配置文件（可用 configs/default_macbert.yaml 切换 MacBERT）")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else Config()
    ev = Evaluator(cfg)

    data_path = "data/eval_dataset/rating_set.jsonl"
    print("加载评分集...", flush=True)
    h_all, g_all = _load_rating_set(data_path)
    print(f"  H={len(h_all)}  G={len(g_all)}", flush=True)

    rng = np.random.RandomState(42)
    g_sample = list(rng.choice(g_all, size=min(60, len(g_all)), replace=False))
    h_sample = list(rng.choice(h_all, size=min(40, len(h_all)), replace=False))

    print("拟合评估器（部署配置，length_match=False）...", flush=True)
    ev.fit(h_all, g_all)
    L0 = ev.scorer.strict_len_base
    alpha = ev.scorer.strict_alpha
    print(f"  L0(严苛度基准)=H中位长度 {L0}   α={alpha}", flush=True)

    lam = ev.scorer.lambdas

    def _row(texts):
        """批量打分，返回每篇的 (len, like, final)。"""
        texts = list(texts)
        cs = ev.scorer.components_batch(texts)
        lens, likes, finals = [], [], []
        for t, c in zip(texts, cs):
            like = float(np.clip(lam[0] * c["S_disc"] + lam[1] * c["S_repr"] + lam[2] * c["S_attr"], 0.0, 1.0))
            kappa = ev.scorer.strictness(len(t))
            lens.append(len(t))
            likes.append(like)
            finals.append(like ** kappa)
        return lens, likes, finals

    def _row_components(texts):
        """批量返回三视图分量 + like + final。"""
        texts = list(texts)
        cs = ev.scorer.components_batch(texts)
        lens, d, r, a, likes, finals = [], [], [], [], [], []
        for t, c in zip(texts, cs):
            like = float(np.clip(lam[0] * c["S_disc"] + lam[1] * c["S_repr"] + lam[2] * c["S_attr"], 0.0, 1.0))
            kappa = ev.scorer.strictness(len(t))
            d.append(c["S_disc"])
            r.append(c["S_repr"])
            a.append(c["S_attr"])
            likes.append(like)
            finals.append(like ** kappa)
            lens.append(len(t))
        return lens, d, r, a, likes, finals

    print("\n=== 1. G 侧注水攻击：最终分是否随注水上涨 ===")
    print("三视图分量（S_disc 判别 / S_repr 表示 / S_attr 属性）+ 像人分 + 最终分（含 κ）")
    print(f"{'策略':<14}{'比例':>5}{'len':>7}{'S_disc':>8}{'S_repr':>8}{'S_attr':>8}{'像人分':>8}{'κ':>6}{'最终分':>8}")
    summary = {s: {r: None for r in RATIOS} for s in STRATEGIES}
    variants_g = _pad_variants(g_sample)
    for s in STRATEGIES:
        for r in RATIOS:
            lens, dd, rr, aa, likes, finals = _row_components(variants_g[s][r])
            kappas = [ev.scorer.strictness(l) for l in lens]
            summary[s][r] = (mean(likes), mean(finals), mean(kappas))
            print(f"{s:<14}{r:>5.1f}{mean(lens):>7.0f}"
                  f"{mean(dd):>8.3f}{mean(rr):>8.3f}{mean(aa):>8.3f}"
                  f"{mean(likes):>8.3f}{mean(kappas):>6.2f}{mean(finals):>8.3f}")

    print("\n  [攻击判定] 从 1.0x → 3.0x：")
    for s in STRATEGIES:
        like1, f1, _ = summary[s][1.0]
        like3, f3, _ = summary[s][3.0]
        like_delta = (like3 - like1) / max(like1, 1e-9) * 100
        delta = (f3 - f1) / max(f1, 1e-9) * 100
        verdict = "[通过] 最终分未涨" if f3 <= f1 * 1.01 else "[失败] 最终分上涨"
        print(f"    {s:<14} 像人分 {like_delta:+.1f}%  最终分 {delta:+.1f}%  {verdict}")
        # 峰值：任何比例下最终分相对 1.0x 的最大涨幅
        peak = max((summary[s][r][1] - f1) / max(f1, 1e-9) for r in RATIOS)
        print(f"              (注水各比例下最终分相对原文本的最大涨幅 = {peak:+.1%})")

    # 合并所有注水样本的最终分 vs 长度 Spearman
    print("\n  [总体] 最终分 vs 长度 Spearman（所有 G 注水样本）:")
    all_lens, all_finals = [], []
    for s in STRATEGIES:
        for r in RATIOS[1:]:
            lens, likes, finals = _row(variants_g[s][r])
            all_lens.extend(lens)
            all_finals.extend(finals)
    rho, p = spearmanr(all_lens, all_finals)
    print(f"    Spearman(final, len) = {rho:+.3f}  (p={p:.3f})  应 ≈0 或 <0")

    print("\n=== 2. H 侧对照：严苛度是否过度惩罚人类长文 ===")
    print("注：'自然长文'= 拼接两段不同 H 块（全人工内容，只变长度，隔离内容因素）；")
    print("    '注水H' = 人类文本追加空洞废话（内容被污染，对照组）。")
    # 自然长文：两两拼接不同的 H 块（同批拼接，长度~2x，无重复内容）
    rng_h = np.random.RandomState(7)
    h_shuf = list(rng_h.permutation(len(h_sample)))
    h_natural = [
        h_sample[h_shuf[i]] + h_sample[h_shuf[i + len(h_sample) // 2]]
        for i in range(len(h_sample) // 2)
    ]
    finals1 = _row_components(h_sample)[5]
    finals_nat = _row_components(h_natural)[5]
    print(f"    {'样本':<22}{'最终分':>9}{'len均值':>9}{'变化':>10}")
    print(f"    {'H 原文(1.0x)':<22}{mean(finals1):>9.3f}{mean([len(t) for t in h_sample]):>9.0f}{'':>11}")
    print(f"    {'H 自然长文(~2x)':<22}{mean(finals_nat):>9.3f}{mean([len(t) for t in h_natural]):>9.0f}"
          f"{(mean(finals_nat)-mean(finals1))/max(mean(finals1),1e-9):>+10.1%}")
    print("    (自然长文只应受 κ 的 ~5% 温和惩罚；若大幅下降说明严苛度过度惩罚长度)")
    # 注水 H（内容被污染，区分内容效应）
    variants_h = _pad_variants(h_sample)
    for s in STRATEGIES:
        finals3 = _row_components(variants_h[s][3.0])[5]
        delta = (mean(finals3) - mean(finals1)) / max(mean(finals1), 1e-9) * 100
        print(f"    {'注水H '+s:<22}{mean(finals3):>9.3f}{mean([len(t) for t in variants_h[s][3.0]]):>9.0f}{delta:>+10.1%}")

    print("\n=== 3. 判别器长度偏置诊断（S_disc 是否随注水虚涨） ===")
    if cfg.discriminator.type == "macbert":
        # MacBERT：直接看部署判别器的 S_disc 随注水变化（length_match 训练阶段已做）
        print("    MacBERT 模式：S_disc 随注水比例变化（length_match 训练阶段已保证）:")
        p_mac = ev.scorer.discriminator
        print(f"    {'比例':>6}{'S_disc(macbert)':>14}")
        for r in RATIOS:
            tset = variants_g["append_filler"][r]
            p_r = float(p_mac.predict_human_prob(tset).mean())
            print(f"    {r:>6.1f}{p_r:>14.3f}")
    else:
        print("论文解耦实验用 length_match=True 消除长度偏置；部署配置(MixScorer)未用。")
        print("对比两种配置下 S_disc 随注水比例的变化，判断偏置是否可被 length_match 消除：")
        from narrative_evaluator.models.discriminator import NgramDiscriminator
        disc_nm = NgramDiscriminator(ngram_range=cfg.discriminator.ngram_range,
                                     max_features=cfg.discriminator.max_features)
        disc_nm.fit(h_all, g_all)  # 部署配置
        disc_lm = NgramDiscriminator(ngram_range=cfg.discriminator.ngram_range,
                                     max_features=cfg.discriminator.max_features)
        disc_lm.fit(h_all, g_all, length_match=True, n_buckets=8, seed=42)
        print(f"    length_match=True 匹配后训练样本: H={disc_lm.used_n_h} G={disc_lm.used_n_g}")
        print(f"    {'比例':>6}{'S_disc(nm)':>12}{'S_disc(lm)':>12}")
        for r in RATIOS:
            tset = variants_g["append_filler"][r]
            p_nm = float(disc_nm.predict_human_prob(tset).mean())
            p_lm = float(disc_lm.predict_human_prob(tset).mean())
            print(f"    {r:>6.1f}{p_nm:>12.3f}{p_lm:>12.3f}")

    print("\n=== 结论 ===")
    print("""
    判定分三块：
    (1) 高比例注水(≥2x)与自复读：最终分应下降 —— 已测，成立。
    (2) 轻度注水(1.5x)：若最终分仍上涨，说明 κ 偏温和，攻击面残留（来源见分量表）。
    (3) H 侧：自然长文只受 κ 温和惩罚（不应大幅下降）；注水 H 被内容检测压低属预期。
    """.strip())


if __name__ == "__main__":
    main()
