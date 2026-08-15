"""报告输出：控制台文本 + 可选 JSON 落盘。"""
from __future__ import annotations

import json
import os
from typing import Optional


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        return f"{x:.{nd}f}"
    return str(x)


def render_report(report: dict, top_attributes: int = 8) -> str:
    """把 evaluate_set 报告渲染成可读文本。"""
    lines = []
    lines.append("=" * 60)
    lines.append("类人叙事评估器 报告")
    lines.append("=" * 60)

    lines.append(f"\n样本量：H={report.get('n_h', '—')}  G={report.get('n_g', '—')}")
    lines.append(f"\n【D_total 总体距离】{_fmt(report.get('D_total'))}   (越小越接近人类)")

    align = report.get("alignment")
    if align:
        lines.append("\n【与人类评分对齐】(核心指标)")
        for k, v in align.items():
            lines.append(f"  {k:<24} {_fmt(v)}")

    disc = report.get("D_disc", {})
    lines.append("\n【判别距离 D_disc】")
    lines.append(f"  AUC            = {_fmt(disc.get('auc'))}   (0.5=无法区分, 1=完全可分)")
    lines.append(f"  balanced acc   = {_fmt(disc.get('balanced_acc'))}")
    lines.append(f"  两样本检验统计  = {_fmt(disc.get('two_sample_disc'))}   (2*AUC-1)")

    rep = report.get("D_repr", {})
    lines.append("\n【表示分布距离 D_repr】")
    lines.append(f"  MMD²            = {_fmt(rep.get('mmd'))}")
    lines.append(f"  Wasserstein(近) = {_fmt(rep.get('wasserstein'))}")
    lines.append(f"  Fréchet         = {_fmt(rep.get('frechet'))}")
    lines.append(f"  Human coverage  = {_fmt(rep.get('human_coverage'))}   (机器覆盖人类空间比例)")
    lines.append(f"  Machine-only    = {_fmt(rep.get('machine_only_mass'))}   (机器独有质量比例)")

    attr = report.get("D_attr", {})
    lines.append("\n【可解释属性距离 D_attr】")
    lines.append(f"  加权总距离 = {_fmt(attr.get('weighted'))}")
    pa = attr.get("per_attribute", {})
    if pa:
        # 按距离降序，取 top N（差异最大的属性最值得关注）
        items = sorted(pa.items(), key=lambda kv: kv[1], reverse=True)[:top_attributes]
        lines.append("  逐属性 1D-Wasserstein（越大差异越明显）：")
        for k, v in items:
            lines.append(f"    {k:<22} {_fmt(v)}")
        if len(pa) > top_attributes:
            lines.append(f"    ... 共 {len(pa)} 个属性")

    ranks = report.get("rankings")
    if ranks:
        lines.append("\n【多组机器文本 D_total 排序】(小→大 = 越接近人类)")
        for name, val in ranks.items():
            lines.append(f"  {_fmt(val):>10}  {name}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def write_report(report: dict, out_dir: Optional[str], top_attributes: int = 8) -> Optional[str]:
    """可选：把 JSON 报告写入 out_dir。返回写入路径或 None。"""
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "evaluator_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path
