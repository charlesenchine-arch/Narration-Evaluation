#!/usr/bin/env python3
"""Analyze the 20-prompt, 80-item, three-judge independent-rating pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = ROOT / "data/human_pilot/private/api_family_pilot20"
DEFAULT_OUTPUT = ROOT / "outputs/api_family_pilot20/independent_rating_analysis.json"
DEFAULT_REPORT = ROOT / "deliverables/API家族偏好预实验_独立评分结果.md"
JUDGES = ("deepseek", "gemini", "grok")
SOURCES = ("human", "deepseek", "gemini", "grok")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def bootstrap_mean_ci(values: list[float], seed: int, n_boot: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def unique_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        for attempt in row.get("attempts", []):
            request_id = str(attempt.get("request_id") or "")
            if not request_id:
                payload = json.dumps(
                    [attempt.get("text"), attempt.get("usage"), attempt.get("returned_model")],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                request_id = f"fallback:{hash(payload)}"
            if request_id in seen:
                continue
            seen.add(request_id)
            result.append(attempt)
    return result


def token_cost(work_dir: Path) -> dict[str, Any]:
    def usage_totals(provider: str, attempts: list[dict[str, Any]]) -> dict[str, int]:
        totals = {"input": 0, "output": 0, "reasoning": 0, "cached": 0, "ticks": 0}
        for attempt in attempts:
            usage = attempt.get("usage") or {}
            if provider == "gemini":
                totals["input"] += int(usage.get("promptTokenCount", 0) or 0)
                totals["output"] += int(usage.get("candidatesTokenCount", 0) or 0)
                totals["reasoning"] += int(usage.get("thoughtsTokenCount", 0) or 0)
            else:
                totals["input"] += int(usage.get("prompt_tokens", 0) or 0)
                totals["output"] += int(usage.get("completion_tokens", 0) or 0)
                totals["cached"] += int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                )
                totals["reasoning"] += int(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                )
                totals["ticks"] += int(usage.get("cost_in_usd_ticks", 0) or 0)
        return totals

    def dollars(provider: str, usage: dict[str, int]) -> tuple[float, float | None]:
        if provider == "deepseek":
            estimate = (usage["input"] * 0.435 + usage["output"] * 0.87) / 1_000_000
            return estimate, None
        if provider == "gemini":
            estimate = (
                usage["input"] * 0.30 + (usage["output"] + usage["reasoning"]) * 2.50
            ) / 1_000_000
            return estimate, 0.0
        uncached = max(0, usage["input"] - usage["cached"])
        estimate = (
            uncached * 1.25 + usage["cached"] * 0.20 + usage["output"] * 2.50
        ) / 1_000_000
        actual = usage["ticks"] / 10_000_000_000 if usage["ticks"] else None
        return estimate, actual

    result: dict[str, Any] = {}
    for provider in JUDGES:
        generation_rows = read_jsonl(work_dir / f"generated_{provider}.jsonl")
        rating_rows = read_jsonl(work_dir / f"ratings_{provider}.jsonl")
        attempts = unique_attempts(generation_rows + rating_rows)
        rating_attempts = unique_attempts(rating_rows)
        usage = usage_totals(provider, attempts)
        rating_usage = usage_totals(provider, rating_attempts)
        estimated, actual = dollars(provider, usage)
        rating_estimated, rating_actual = dollars(provider, rating_usage)
        result[provider] = {
            "unique_api_calls": len(attempts),
            "input_tokens": usage["input"],
            "cached_input_tokens": usage["cached"],
            "output_tokens": usage["output"],
            "reasoning_tokens_reported": usage["reasoning"],
            "paid_equivalent_usd": round(estimated, 6),
            "observed_or_provider_reported_usd": None if actual is None else round(actual, 6),
            "formal_rating_api_calls": len(rating_attempts),
            "formal_rating_paid_equivalent_usd": round(rating_estimated, 6),
            "formal_rating_observed_or_provider_reported_usd": (
                None if rating_actual is None else round(rating_actual, 6)
            ),
        }
    return result


def build_rows(work_dir: Path) -> list[dict[str, Any]]:
    key = {row["blind_id"]: row for row in read_jsonl(work_dir / "blind_key.jsonl")}
    item = {row["blind_id"]: row for row in read_jsonl(work_dir / "blind_items.jsonl")}
    rows: list[dict[str, Any]] = []
    for judge in JUDGES:
        ratings = read_jsonl(work_dir / f"ratings_{judge}.jsonl")
        if len(ratings) != 80 or sum(row.get("status") == "ok" for row in ratings) != 80:
            raise RuntimeError(f"{judge} does not have 80 valid formal ratings")
        for rating in ratings:
            blind_id = rating["blind_id"]
            source = key[blind_id]
            work = item[blind_id]
            rows.append(
                {
                    "judge": judge,
                    "blind_id": blind_id,
                    "human_id": source["human_id"],
                    "prompt_id": work["prompt_id"],
                    "source": source["source_family"],
                    "score": int(rating["rating"]["overall_score"]),
                    "grade": rating["rating"]["grade"],
                    "confidence": int(rating["rating"]["confidence"]),
                    "char_count": int(work["char_count"]),
                }
            )
    return rows


def analyze(rows: list[dict[str, Any]], work_dir: Path, seed: int) -> dict[str, Any]:
    by_judge_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_judge_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_judge_item: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_judge_source[(row["judge"], row["source"])].append(row)
        by_judge_prompt[(row["judge"], row["prompt_id"])].append(row)
        by_judge_item[(row["judge"], row["blind_id"])] = row

    distributions: dict[str, Any] = {}
    source_score_means: dict[str, dict[str, float]] = {}
    source_grade_counts: dict[str, dict[str, dict[str, int]]] = {}
    for judge in JUDGES:
        all_scores = [row["score"] for row in rows if row["judge"] == judge]
        distributions[judge] = {
            "mean": round(mean(all_scores), 3),
            "sd": round(sample_sd(all_scores), 3),
            "min": min(all_scores),
            "max": max(all_scores),
            "n_unique_scores": len(set(all_scores)),
            "most_common_scores": Counter(all_scores).most_common(8),
        }
        source_score_means[judge] = {}
        source_grade_counts[judge] = {}
        for source in SOURCES:
            one = by_judge_source[(judge, source)]
            scores = [row["score"] for row in one]
            source_score_means[judge][source] = round(mean(scores), 3)
            source_grade_counts[judge][source] = dict(sorted(Counter(row["grade"] for row in one).items()))

    rank_points: dict[tuple[str, str], float] = {}
    winner_credit: dict[str, dict[str, float]] = {judge: {source: 0.0 for source in SOURCES} for judge in JUDGES}
    ties_by_judge = {judge: 0 for judge in JUDGES}
    for (judge, prompt_id), prompt_rows in by_judge_prompt.items():
        if len(prompt_rows) != 4:
            raise RuntimeError(f"{judge}/{prompt_id} has {len(prompt_rows)} rows")
        scores = np.asarray([row["score"] for row in prompt_rows], dtype=float)
        ranks = rankdata(-scores, method="average")
        for row, rank in zip(prompt_rows, ranks):
            rank_points[(judge, row["blind_id"])] = float((4.0 - rank) / 3.0)
        maximum = float(scores.max())
        winners = [row for row in prompt_rows if row["score"] == maximum]
        if len(winners) > 1:
            ties_by_judge[judge] += 1
        for winner in winners:
            winner_credit[judge][winner["source"]] += 1.0 / len(winners)

    source_rank_points: dict[str, dict[str, float]] = {}
    for judge in JUDGES:
        source_rank_points[judge] = {}
        for source in SOURCES:
            values = [rank_points[(judge, row["blind_id"])] for row in by_judge_source[(judge, source)]]
            source_rank_points[judge][source] = round(mean(values), 4)

    affinities: dict[str, Any] = {}
    for family_index, family in enumerate(JUDGES):
        family_rows = sorted(by_judge_source[(family, family)], key=lambda row: row["prompt_id"])
        deltas: list[float] = []
        for own_row in family_rows:
            blind_id = own_row["blind_id"]
            own = rank_points[(family, blind_id)]
            other = mean([rank_points[(judge, blind_id)] for judge in JUDGES if judge != family])
            deltas.append(own - other)
        try:
            test = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            p_value = float(test.pvalue)
        except ValueError:
            p_value = 1.0
        affinities[family] = {
            "mean_rank_point_uplift": round(mean(deltas), 4),
            "bootstrap_95_ci": [round(value, 4) for value in bootstrap_mean_ci(deltas, seed + family_index)],
            "wilcoxon_p_unadjusted": round(p_value, 6),
            "n_prompts": len(deltas),
        }

    score_vectors = {
        judge: [by_judge_item[(judge, blind_id)]["score"] for blind_id in sorted({row["blind_id"] for row in rows})]
        for judge in JUDGES
    }
    agreement: dict[str, float] = {}
    for left_index, left in enumerate(JUDGES):
        for right in JUDGES[left_index + 1 :]:
            rho = float(spearmanr(score_vectors[left], score_vectors[right]).statistic)
            agreement[f"{left}__{right}"] = round(rho, 4)

    human_machine_gap: dict[str, Any] = {}
    for judge in JUDGES:
        prompt_deltas = []
        for prompt_id in sorted({row["prompt_id"] for row in rows}):
            one = by_judge_prompt[(judge, prompt_id)]
            human_score = next(row["score"] for row in one if row["source"] == "human")
            machine_scores = [row["score"] for row in one if row["source"] != "human"]
            prompt_deltas.append(human_score - mean(machine_scores))
        human_machine_gap[judge] = {
            "mean_human_minus_machine_score": round(mean(prompt_deltas), 3),
            "bootstrap_95_ci": [round(value, 3) for value in bootstrap_mean_ci(prompt_deltas, seed + 20)],
        }

    return {
        "protocol": "api-family-pilot20-independent-rating-v1",
        "n_prompts": 20,
        "n_items": 80,
        "n_ratings": len(rows),
        "score_distributions": distributions,
        "source_score_means": source_score_means,
        "source_grade_counts": source_grade_counts,
        "source_mean_within_prompt_rank_points": source_rank_points,
        "prompt_winner_credit": winner_credit,
        "prompts_with_top_score_ties": ties_by_judge,
        "own_family_affinity": affinities,
        "interjudge_spearman": agreement,
        "human_minus_machine": human_machine_gap,
        "token_and_cost": token_cost(work_dir),
        "interpretation_limits": [
            "The human references predate the reverse prompts, so this is not the final same-prompt Pool B benchmark.",
            "No human quality labels are present; score gaps cannot by themselves establish evaluator bias.",
            "The 500..1200-character generation target used a logged 3% upper QC tolerance after calibration.",
            "Independent judge scores require within-judge normalization; raw scores are not comparable across judges.",
        ],
    }


def report_markdown(result: dict[str, Any]) -> str:
    means = result["source_score_means"]
    ranks = result["source_mean_within_prompt_rank_points"]
    lines = [
        "# API 家族偏好预实验：240 次独立评分结果",
        "",
        "版本：2026-08-24  ",
        "状态：20题、80篇、3位模型裁判、240次正式评分全部完成。",
        "",
        "## 1. 数据与调用",
        "",
        "- 文本：20篇人类参考＋DeepSeek/Gemini/Grok各20篇，共80篇。",
        "- 评测：每位模型在全新单轮上下文中独立评80篇；请求中不含来源信息。",
        "- 设置：DeepSeek关闭思考、Gemini minimal、Grok non-reasoning；均未提供搜索或工具。",
        "- 长度：Prompt要求500—1200个非空白字符；本地QC记录3%上限计数容差。",
        "",
        "## 2. 来源平均分（只可在同一裁判内横向比较）",
        "",
        "| 裁判 | 人类 | DeepSeek文本 | Gemini文本 | Grok文本 |",
        "|---|---:|---:|---:|---:|",
    ]
    for judge in JUDGES:
        lines.append(
            f"| {judge} | {means[judge]['human']:.2f} | {means[judge]['deepseek']:.2f} | "
            f"{means[judge]['gemini']:.2f} | {means[judge]['grok']:.2f} |"
        )
    lines += [
        "",
        "## 3. 同题相对排名分",
        "",
        "排名分将同一Prompt下四篇文本转换到0—1：单独第一为1，单独最后为0，并列取平均。",
        "",
        "| 裁判 | 人类 | DeepSeek文本 | Gemini文本 | Grok文本 |",
        "|---|---:|---:|---:|---:|",
    ]
    for judge in JUDGES:
        lines.append(
            f"| {judge} | {ranks[judge]['human']:.3f} | {ranks[judge]['deepseek']:.3f} | "
            f"{ranks[judge]['gemini']:.3f} | {ranks[judge]['grok']:.3f} |"
        )
    lines += [
        "",
        "## 4. 本家族亲和增量",
        "",
        "定义：本家族裁判给本家族文本的同题排名分，减去另外两位裁判给完全相同文本的平均排名分。正值表示本家族相对抬高。",
        "",
        "| 家族 | 平均增量 | 95% bootstrap CI | 未校正p值 |",
        "|---|---:|---:|---:|",
    ]
    for family in JUDGES:
        value = result["own_family_affinity"][family]
        ci = value["bootstrap_95_ci"]
        lines.append(
            f"| {family} | {value['mean_rank_point_uplift']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {value['wilcoxon_p_unadjusted']:.4f} |"
        )
    lines += [
        "",
        "## 5. 裁判特性与一致性",
        "",
        "| 裁判 | 平均分 | 标准差 | 不同分值数 | 同题最高分并列题数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for judge in JUDGES:
        dist = result["score_distributions"][judge]
        lines.append(
            f"| {judge} | {dist['mean']:.2f} | {dist['sd']:.2f} | {dist['n_unique_scores']} | "
            f"{result['prompts_with_top_score_ties'][judge]} |"
        )
    lines += [
        "",
        "裁判两两Spearman：",
        "",
    ]
    for pair, rho in result["interjudge_spearman"].items():
        lines.append(f"- {pair}: {rho:+.3f}")
    lines += [
        "",
        "## 6. 人类文本相对三类机器文本",
        "",
        "该结果只是当前样本的模型评分差，不能在没有人工质量标签时解释为来源偏见。",
        "",
        "| 裁判 | 人类分－同题机器平均分 | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for judge in JUDGES:
        gap = result["human_minus_machine"][judge]
        ci = gap["bootstrap_95_ci"]
        lines.append(
            f"| {judge} | {gap['mean_human_minus_machine_score']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |"
        )
    lines += [
        "",
        "## 7. Token与费用",
        "",
        "| 服务商 | 唯一API调用 | 输入token | 输出token | 观察/估算费用（美元） |",
        "|---|---:|---:|---:|---:|",
    ]
    for provider in JUDGES:
        cost = result["token_and_cost"][provider]
        observed = cost["observed_or_provider_reported_usd"]
        display = cost["paid_equivalent_usd"] if observed is None else observed
        suffix = "估算" if observed is None else "观察/服务商报告"
        lines.append(
            f"| {provider} | {cost['unique_api_calls']} | {cost['input_tokens']} | "
            f"{cost['output_tokens']} | ${display:.4f}（{suffix}） |"
        )
    formal_costs = []
    for provider in JUDGES:
        cost = result["token_and_cost"][provider]
        observed = cost["formal_rating_observed_or_provider_reported_usd"]
        formal_costs.append(
            cost["formal_rating_paid_equivalent_usd"] if observed is None else observed
        )
    lines += [
        "",
        f"其中240次正式评分本身合计约 ${sum(formal_costs):.4f}；其余主要来自60篇生成与长度纠偏。",
        "",
        "## 8. 解释边界",
        "",
        "1. 当前人类文本先于反向Prompt存在，不是最终预注册同Prompt人机写作池。",
        "2. 当前没有来源盲化的人工质量金标准；即使出现家族亲和信号，也只能作为下一阶段假设。",
        "3. Grok只使用了 8 个不同分值，分数刻度较粗且集中；应以同题排名与后续配对复核为主。",
        "4. 本轮已查看结果，20题只能继续作为pilot；最终论文测试必须换用全新冻结Prompt。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.work_dir.resolve())
    result = analyze(rows, args.work_dir.resolve(), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
