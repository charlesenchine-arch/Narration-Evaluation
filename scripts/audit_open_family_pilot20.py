#!/usr/bin/env python3
"""Audit raw pairwise judgments and compute prompt-clustered uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_open_family_pilot20 import JUDGE_SYSTEM, JUDGE_TEMPLATE, MODELS, render_chat


FAMILIES = ("qwen", "hunyuan", "internlm")
FAMILY_LABELS = {"qwen": "Qwen", "hunyuan": "Hunyuan", "internlm": "InternLM"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "NA" if value is None or math.isnan(value) else f"{100 * value:.1f}%"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-tokenization", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(work_dir / "private" / "pairs.jsonl")
    pair_by_id = {row["judgment_id"]: row for row in pairs}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    valid_by_judge: dict[str, dict[str, dict[str, Any]]] = {}
    invalid_by_judge: dict[str, list[dict[str, Any]]] = {}
    for judge in MODELS:
        rows = read_jsonl(work_dir / "judgments" / f"{judge}.jsonl")
        all_rows[judge] = rows
        valid: dict[str, dict[str, Any]] = {}
        invalid = []
        for row in rows:
            if row.get("status") == "ok":
                valid[row["judgment_id"]] = row
            else:
                invalid.append(row)
        valid_by_judge[judge] = valid
        invalid_by_judge[judge] = invalid

    generation_texts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for model in MODELS:
        for row in read_jsonl(work_dir / "generations" / f"{model}.jsonl"):
            if row.get("status") == "ok":
                generation_texts[(row["prompt_id"], model)].add(row["text"])

    pair_audit = Counter()
    han_counts: list[int] = []
    for pair in pairs:
        pair_audit["records"] += 1
        for side in ("a", "b"):
            text = pair[f"text_{side}"]
            model = pair[f"model_{side}"]
            count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
            han_counts.append(count)
            pair_audit["nonempty_texts"] += bool(text.strip())
            pair_audit["texts_in_generation_log"] += text in generation_texts[(pair["prompt_id"], model)]
            pair_audit["texts_in_qc_range"] += 450 <= count <= 550
        pair_audit["identical_a_b"] += pair["text_a"] == pair["text_b"]

    judge_metrics: dict[str, dict[str, Any]] = {}
    merged: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for judge, valid in valid_by_judge.items():
        winners = Counter(row["winner"] for row in valid.values())
        for row in valid.values():
            merged[(judge, row["canonical_pair_id"])].append(row)
        groups = [group for (j, _), group in merged.items() if j == judge and len(group) == 2]
        consistent = sum(group[0]["selected_model"] == group[1]["selected_model"] for group in groups)
        both_a = sum(all(row["winner"] == "A" for row in group) for group in groups)
        raw_success = sum("raw" in row for row in valid.values())
        invalid = invalid_by_judge[judge]
        invalid_starts_think = sum(str(row.get("raw", "")).lstrip().startswith("<think>") for row in invalid)
        invalid_has_closed_think = sum("</think>" in str(row.get("raw", "")) for row in invalid)
        invalid_placeholder = sum("A或B或TIE" in str(row.get("raw", "")) for row in invalid)
        judge_metrics[judge] = {
            "valid_unique": len(valid),
            "a_count": winners["A"],
            "b_count": winners["B"],
            "tie_count": winners["TIE"],
            "a_rate": winners["A"] / len(valid) if valid else None,
            "tie_rate": winners["TIE"] / len(valid) if valid else None,
            "swap_pairs": len(groups),
            "swap_consistency": consistent / len(groups) if groups else None,
            "both_positions_a_rate": both_a / len(groups) if groups else None,
            "historical_terminal_invalid_records": len(invalid),
            "historical_terminal_invalid_record_rate": len(invalid) / len(all_rows[judge]) if all_rows[judge] else None,
            "invalid_starts_think": invalid_starts_think,
            "invalid_has_closed_think": invalid_has_closed_think,
            "invalid_placeholder_answer": invalid_placeholder,
            "final_unresolved": len(pair_by_id.keys() - valid.keys()),
            "successful_raw_logged": raw_success,
        }

    # Merge AB/BA before estimating source-family preferences.
    logical_values: list[dict[str, Any]] = []
    for (judge, canonical_id), group in merged.items():
        if len(group) != 2:
            continue
        base = group[0]
        left_model, right_model = canonical_id.split("__")[1:3]
        left_family, right_family = MODELS[left_model]["family"], MODELS[right_model]["family"]
        if left_family == right_family:
            continue
        for source_model, source_family in ((left_model, left_family), (right_model, right_family)):
            scores = [0.5 if row["selected_model"] == "TIE" else float(row["selected_model"] == source_model) for row in group]
            logical_values.append(
                {
                    "prompt_id": base["prompt_id"],
                    "judge_family": MODELS[judge]["family"],
                    "judge_model": judge,
                    "source_model": source_model,
                    "source_family": source_family,
                    "opponent_family": right_family if source_family == left_family else left_family,
                    "score": mean(scores),
                }
            )

    prompt_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    targeted_cell: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in logical_values:
        prompt_cell[(row["prompt_id"], row["judge_family"], row["source_family"])].append(row["score"])
        targeted_cell[(row["prompt_id"], row["judge_family"], row["source_family"], row["opponent_family"])].append(row["score"])
    prompt_means = {key: mean(values) for key, values in prompt_cell.items()}
    targeted_means = {key: mean(values) for key, values in targeted_cell.items()}
    prompt_ids = sorted({row["prompt_id"] for row in pairs})

    family_matrix: dict[str, dict[str, float]] = {}
    for judge_family in FAMILIES:
        family_matrix[judge_family] = {}
        for source_family in FAMILIES:
            family_matrix[judge_family][source_family] = mean(
                [prompt_means[(prompt_id, judge_family, source_family)] for prompt_id in prompt_ids]
            )

    def family_excess(family: str, sampled_prompts: list[str]) -> float:
        own = [prompt_means[(prompt_id, family, family)] for prompt_id in sampled_prompts]
        other = [
            prompt_means[(prompt_id, judge_family, family)]
            for prompt_id in sampled_prompts
            for judge_family in FAMILIES
            if judge_family != family
        ]
        return mean(own) - mean(other)

    rng = random.Random(20260914)
    family_excess_results: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        boot = sorted(
            family_excess(family, [rng.choice(prompt_ids) for _ in prompt_ids])
            for _ in range(args.bootstrap)
        )
        family_excess_results[family] = {
            "point": family_excess(family, prompt_ids),
            "ci_low": percentile(boot, 0.025),
            "ci_high": percentile(boot, 0.975),
        }

    targeted: dict[str, dict[str, float]] = {}
    for first, second in (("qwen", "hunyuan"), ("qwen", "internlm"), ("hunyuan", "internlm")):
        key = f"{first}_vs_{second}__p_{first}"
        targeted[key] = {
            judge_family: mean(
                [targeted_means[(prompt_id, judge_family, first, second)] for prompt_id in prompt_ids]
            )
            for judge_family in FAMILIES
        }

    token_audit: dict[str, Any] = {}
    if args.check_tokenization:
        from transformers import AutoTokenizer

        for judge, spec in MODELS.items():
            tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, trust_remote_code=True)
            lengths = []
            for pair in pairs:
                user_prompt = JUDGE_TEMPLATE.format(
                    setting=pair["setting"], text_a=pair["text_a"], text_b=pair["text_b"]
                )
                if spec["family"] == "hunyuan":
                    user_prompt = "/no_think\n" + user_prompt
                rendered = render_chat(tokenizer, JUDGE_SYSTEM, user_prompt)
                lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
            ordered = sorted(lengths)
            token_audit[judge] = {
                "minimum": min(ordered),
                "median": percentile(ordered, 0.5),
                "p95": percentile(ordered, 0.95),
                "maximum": max(ordered),
                "over_4000": sum(length > 4000 for length in ordered),
                "prompt_hash": hashlib.sha256(str(ordered).encode()).hexdigest(),
            }

    result = {
        "pair_audit": dict(pair_audit),
        "han_count_min": min(han_counts),
        "han_count_max": max(han_counts),
        "judge_metrics": judge_metrics,
        "family_matrix": family_matrix,
        "family_excess_prompt_cluster_bootstrap": family_excess_results,
        "targeted_family_contrasts": targeted,
        "token_audit": token_audit,
        "limitations": {
            "successful_raw_outputs_available": False,
            "exact_invalid_call_rate_recoverable": False,
            "reason": "Successful raw model outputs and first-attempt retry failures were not logged in the original run.",
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 开放权重家族偏好预实验：原始判断审计",
        "",
        "## 原始记录可审计性",
        "",
        f"- 匿名方向记录：{pair_audit['records']}；六位评委最终有效判断：{sum(m['valid_unique'] for m in judge_metrics.values())}。",
        f"- 两侧文本非空：{pair_audit['nonempty_texts']}/{2 * pair_audit['records']}；可在成功生成日志中逐字匹配：{pair_audit['texts_in_generation_log']}/{2 * pair_audit['records']}。",
        f"- 文本汉字范围：{min(han_counts)}—{max(han_counts)}；450—550 汉字合格：{pair_audit['texts_in_qc_range']}/{2 * pair_audit['records']}。",
        f"- A/B 文本完全相同的记录：{pair_audit['identical_a_b']}。",
        "- 解析器没有默认 A/B 选项；无法解析时写入 `parse_failed`。但成功记录没有保存 raw 输出，第一轮无效但第二轮成功的 raw 也未保存，因此原始逐调用无效率不能精确恢复。",
        "",
        "## 每个评委模型",
        "",
        "| 评委 | A选择率 | 平局率 | 换序一致率 | 两次都选位置A | 历史终止无效记录率 | 最终未解决 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for judge, metric in judge_metrics.items():
        lines.append(
            f"| {judge} | {pct(metric['a_rate'])} | {pct(metric['tie_rate'])} | "
            f"{pct(metric['swap_consistency'])} | {pct(metric['both_positions_a_rate'])} | "
            f"{pct(metric['historical_terminal_invalid_record_rate'])} ({metric['historical_terminal_invalid_records']}条) | "
            f"{metric['final_unresolved']} |"
        )
    lines.extend([
        "",
        "历史终止无效记录率以 JSONL 中保存的终止状态记录为分母；它不是逐调用无效率。每条 `parse_failed` 代表该轮两次解析均失败，而成功前是否发生过一次失败没有日志，故需在重复实验中重新完整记录 raw。",
        "",
        "## 已保存失败 raw 的原因",
        "",
        f"- qwen3_8b 保存了 {judge_metrics['qwen3_8b']['historical_terminal_invalid_records']} 条终止无效记录，均把格式占位文字 `A或B或TIE` 原样作为 winner；修正重试提示后该唯一缺失逻辑方向已补齐。",
        f"- hunyuan_7b 保存了 {judge_metrics['hunyuan_7b']['historical_terminal_invalid_records']} 条终止无效记录，其中 {judge_metrics['hunyuan_7b']['invalid_starts_think']} 条均以 `<think>` 开始、{judge_metrics['hunyuan_7b']['invalid_has_closed_think']} 条包含闭合 `</think>`。它们在 96 token 上限内尚未给出 winner，属于未关闭思考导致的截断，而不是解析器默认成 A。后来修复运行取得了 600 条有效记录，但成功 raw 当时未保存，仍需由本次完整 raw 重复实验复核。",
        "",
        "## AB/BA 合并后的家族矩阵",
        "",
        "| 评委家族 | Qwen文本 | Hunyuan文本 | InternLM文本 |",
        "|---|---:|---:|---:|",
    ])
    for judge_family in FAMILIES:
        lines.append("| " + FAMILY_LABELS[judge_family] + " | " + " | ".join(pct(family_matrix[judge_family][source]) for source in FAMILIES) + " |")
    lines.extend([
        "",
        "## 本家族额外偏好（按 Prompt 成组 bootstrap）",
        "",
        "| 家族 | 点估计 | 95%区间 |",
        "|---|---:|---:|",
    ])
    for family in FAMILIES:
        item = family_excess_results[family]
        lines.append(f"| {FAMILY_LABELS[family]} | {pct(item['point'])} | {pct(item['ci_low'])} 至 {pct(item['ci_high'])} |")
    lines.extend([
        "",
        "## 针对性家族对照",
        "",
        "单元格表示该列评委家族在指定两家族文本直接交锋时选择表中第一个家族的概率。",
        "",
        "| 文本对照 | Qwen评委 | Hunyuan评委 | InternLM评委 |",
        "|---|---:|---:|---:|",
    ])
    for first, second in (("qwen", "hunyuan"), ("qwen", "internlm"), ("hunyuan", "internlm")):
        values = targeted[f"{first}_vs_{second}__p_{first}"]
        lines.append(
            f"| P({FAMILY_LABELS[first]} 胜 {FAMILY_LABELS[second]}) | "
            + " | ".join(pct(values[judge_family]) for judge_family in FAMILIES)
            + " |"
        )
    if token_audit:
        lines.extend([
            "",
            "## 实际输入长度审计",
            "",
            "| 评委 | 最短 | 中位数 | P95 | 最长 | 超过4000 token |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for judge, item in token_audit.items():
            lines.append(
                f"| {judge} | {item['minimum']} | {item['median']} | {item['p95']} | {item['maximum']} | {item['over_4000']} |"
            )
    lines.extend([
        "",
        "## 结论边界",
        "",
        "- AB/BA 已先合并为逻辑对，区间按 20 个 Prompt 成组重采样，而非把 3,600 次调用视为独立样本。",
        "- 该区间只反映 Prompt 抽样变异，尚未覆盖解码重复性、人类质量不确定性和模型家族总体质量混杂。",
        "- 同顺序重复判断需要作为新增审计实验运行；届时必须保存每次 raw、输入哈希、token 数、解析路径和无效输出。",
    ])
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
