#!/usr/bin/env python3
"""Calibrate pairwise-judge protocol effects with a balanced factorial design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_open_family_pilot20 import (
    MODELS,
    SEED,
    append_jsonl,
    load_engine,
    read_jsonl,
    release_engine,
    render_chat,
    utc_now,
)


SYSTEM = (
    "你是匿名中文叙事质量评委。只判断文本质量，不推测作者身份、模型来源或写作方式。"
    "严格遵守输出格式。"
)

WORDINGS = {
    "holistic": (
        "以下两篇匿名中文叙事文本回应同一写作任务。请比较它们的整体叙事质量，"
        "综合考虑任务完成、连贯性、人物与情感、语言表达和感染力，但不要机械逐项计分。"
    ),
    "editor": (
        "你是一名不知道作者身份的中文小说编辑。以下两篇文本回应同一写作任务。"
        "如果只能保留整体完成得更好的一篇，请作出选择；只有确实难分高下时才判平局。"
    ),
    "reader": (
        "站在成熟普通读者的角度，比较以下两篇对同一写作任务的匿名回应。"
        "选择整体上更清楚、完整、自然且更值得读完的一篇；若两篇整体质量相当则判平局。"
    ),
}

LABELS = {
    "ABT": {"primary": "A", "secondary": "B", "tie": "T"},
    "120": {"primary": "1", "secondary": "2", "tie": "0"},
}


def family_pair(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((MODELS[row["model_a"]]["family"], MODELS[row["model_b"]]["family"])))


def prepare_plan(work_dir: Path, out_dir: Path, per_family_pair: int) -> list[dict[str, Any]]:
    path = out_dir / "calibration_plan.jsonl"
    if path.exists():
        return read_jsonl(path)
    rows = [row for row in read_jsonl(work_dir / "private" / "pairs.jsonl") if row["orientation"] == "AB"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = family_pair(row)
        if key[0] != key[1]:
            groups[key].append(row)
    chosen: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda row: hashlib.sha256(f"protocol:{SEED}:{row['canonical_pair_id']}".encode()).hexdigest(),
        )
        group_choice: list[dict[str, Any]] = []
        for row in candidates:
            if row["prompt_id"] not in used_prompts:
                group_choice.append(row)
                used_prompts.add(row["prompt_id"])
            if len(group_choice) == per_family_pair:
                break
        if len(group_choice) < per_family_pair:
            for row in candidates:
                if row not in group_choice:
                    group_choice.append(row)
                if len(group_choice) == per_family_pair:
                    break
        chosen.extend(group_choice)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(chosen, 1):
        append_jsonl(path, {"calibration_pair_id": f"C{index:03d}", **row})
    return read_jsonl(path)


def cases_from_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in plan:
        for wording in WORDINGS:
            for scheme in LABELS:
                for mapping in ("normal", "reversed"):
                    for orientation in ("AB", "BA"):
                        if orientation == "AB":
                            first_model, second_model = row["model_a"], row["model_b"]
                            first_text, second_text = row["text_a"], row["text_b"]
                        else:
                            first_model, second_model = row["model_b"], row["model_a"]
                            first_text, second_text = row["text_b"], row["text_a"]
                        label = LABELS[scheme]
                        if mapping == "normal":
                            first_label, second_label = label["primary"], label["secondary"]
                        else:
                            first_label, second_label = label["secondary"], label["primary"]
                        case_id = f"{row['calibration_pair_id']}__{wording}__{scheme}__{mapping}__{orientation}"
                        cases.append(
                            {
                                "case_id": case_id,
                                "calibration_pair_id": row["calibration_pair_id"],
                                "canonical_pair_id": row["canonical_pair_id"],
                                "prompt_id": row["prompt_id"],
                                "setting": row["setting"],
                                "canonical_left_model": row["model_a"],
                                "canonical_right_model": row["model_b"],
                                "wording": wording,
                                "label_scheme": scheme,
                                "label_mapping": mapping,
                                "orientation": orientation,
                                "first_model": first_model,
                                "second_model": second_model,
                                "first_text": first_text,
                                "second_text": second_text,
                                "first_label": first_label,
                                "second_label": second_label,
                                "tie_label": label["tie"],
                                "primary_label": label["primary"],
                            }
                        )
    return cases


def make_prompt(case: dict[str, Any]) -> str:
    return f"""{WORDINGS[case['wording']]}

写作任务：{case['setting']}

文本{case['first_label']}：
{case['first_text']}

文本{case['second_label']}：
{case['second_text']}

只输出一个标签：{case['first_label']}、{case['second_label']}或{case['tie_label']}。不要解释，不要标点。难分高下时输出{case['tie_label']}。
答案："""


def parse_label(raw: str, allowed: set[str]) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    answer = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", cleaned, flags=re.S | re.I)
    if answer:
        cleaned = answer.group(1).strip()
    cleaned = cleaned.strip("`'\"<>《》【】[]()（） \t\r\n。！!，,：:")
    if cleaned not in allowed:
        raise ValueError(f"not one exact allowed label: {raw!r}")
    return cleaned


def top_logprobs(output: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    values = getattr(output.outputs[0], "logprobs", None)
    if not values:
        return result
    for token_id, item in values[0].items():
        result.append(
            {
                "token_id": int(token_id),
                "token": getattr(item, "decoded_token", None),
                "logprob": float(item.logprob),
            }
        )
    return sorted(result, key=lambda item: item["logprob"], reverse=True)


def cmd_run(args: argparse.Namespace) -> int:
    from vllm import SamplingParams

    plan = prepare_plan(args.work_dir, args.out_dir, args.per_family_pair)
    cases = cases_from_plan(plan)
    path = args.out_dir / "protocol_outputs" / f"{args.model}.jsonl"
    completed = {row["case_id"] for row in read_jsonl(path)}
    pending = [case for case in cases if case["case_id"] not in completed]
    print(f"protocol calibration model={args.model} complete={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        return 0
    llm, tokenizer = load_engine(args.model)
    rendered: list[str] = []
    for case in pending:
        user = make_prompt(case)
        if MODELS[args.model]["family"] == "hunyuan":
            user = "/no_think\n" + user
        rendered.append(render_chat(tokenizer, SYSTEM, user))
    is_hunyuan = MODELS[args.model]["family"] == "hunyuan"
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=32 if is_hunyuan else 4,
        stop=["</answer>"] if is_hunyuan else None,
        logprobs=20,
        seed=SEED,
    )
    outputs = llm.generate(rendered, sampling, use_tqdm=True)
    for case, prompt, output in zip(pending, rendered, outputs):
        raw = output.outputs[0].text
        allowed = {case["first_label"], case["second_label"], case["tie_label"]}
        record = {
            **{key: value for key, value in case.items() if key not in {"first_text", "second_text"}},
            "judge_model": args.model,
            "judge_family": MODELS[args.model]["family"],
            "raw": raw,
            "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_token_count": len(tokenizer.encode(prompt, add_special_tokens=False)),
            "output_token_count": len(output.outputs[0].token_ids),
            "first_token_top_logprobs": top_logprobs(output),
            "temperature": 0.0,
            "recorded_at": utc_now(),
        }
        try:
            label = parse_label(raw, allowed)
            if label == case["tie_label"]:
                selected_model, selected_position = "TIE", "TIE"
            elif label == case["first_label"]:
                selected_model, selected_position = case["first_model"], "first"
            else:
                selected_model, selected_position = case["second_model"], "second"
            record.update(
                {
                    "status": "ok",
                    "selected_label": label,
                    "selected_model": selected_model,
                    "selected_position": selected_position,
                    "selected_primary_label": label == case["primary_label"],
                }
            )
        except Exception as exc:
            record.update({"status": "invalid", "error": f"{type(exc).__name__}: {exc}"})
        append_jsonl(path, record)
    release_engine(llm)
    return 0


def score_left(row: dict[str, Any]) -> float:
    if row.get("selected_model") == "TIE":
        return 0.5
    return float(row.get("selected_model") == row["canonical_left_model"])


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{100 * value:+.1f}%" if signed else f"{100 * value:.1f}%"


def judge_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    position_pairs: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    label_pairs: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    wording_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    scheme_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        position_pairs[(row["calibration_pair_id"], row["wording"], row["label_scheme"], row["label_mapping"])][row["orientation"]] = row
        label_pairs[(row["calibration_pair_id"], row["wording"], row["label_scheme"], row["orientation"])][row["label_mapping"]] = row
        wording_groups[(row["calibration_pair_id"], row["label_scheme"], row["label_mapping"], row["orientation"])].append(row)
        scheme_groups[(row["calibration_pair_id"], row["wording"], row["label_mapping"], row["orientation"])].append(row)

    position_effects, swap_agreements = [], []
    for item in position_pairs.values():
        if set(item) == {"AB", "BA"}:
            ab, ba = item["AB"], item["BA"]
            position_effects.append(score_left(ab) - score_left(ba))
            swap_agreements.append(ab["selected_model"] == ba["selected_model"])

    label_effects, mapping_agreements = [], []
    for item in label_pairs.values():
        if set(item) == {"normal", "reversed"}:
            normal, reversed_row = item["normal"], item["reversed"]
            left_primary = normal if normal["orientation"] == "AB" else reversed_row
            left_secondary = reversed_row if normal["orientation"] == "AB" else normal
            label_effects.append(score_left(left_primary) - score_left(left_secondary))
            mapping_agreements.append(normal["selected_model"] == reversed_row["selected_model"])

    wording_agreements = [
        len(group) == len(WORDINGS) and len({row["selected_model"] for row in group}) == 1
        for group in wording_groups.values()
    ]
    scheme_agreements = [
        len(group) == len(LABELS) and len({row["selected_model"] for row in group}) == 1
        for group in scheme_groups.values()
    ]
    per_wording = {}
    for wording in WORDINGS:
        subset = [row for row in valid if row["wording"] == wording]
        pos = [value for key, value in position_pairs.items() if key[1] == wording]
        agreements, effects = [], []
        for item in pos:
            if set(item) == {"AB", "BA"}:
                agreements.append(item["AB"]["selected_model"] == item["BA"]["selected_model"])
                effects.append(score_left(item["AB"]) - score_left(item["BA"]))
        per_wording[wording] = {
            "calls": len([row for row in rows if row["wording"] == wording]),
            "invalid_rate": 1 - len(subset) / max(1, len([row for row in rows if row["wording"] == wording])),
            "first_position_rate": mean([0.5 if row["selected_position"] == "TIE" else float(row["selected_position"] == "first") for row in subset]),
            "position_effect": mean(effects),
            "swap_consistency": mean([float(x) for x in agreements]),
        }
    return {
        "calls": len(rows),
        "invalid_rate": 1 - len(valid) / max(1, len(rows)),
        "tie_rate": mean([float(row["selected_model"] == "TIE") for row in valid]),
        "first_position_rate": mean([0.5 if row["selected_position"] == "TIE" else float(row["selected_position"] == "first") for row in valid]),
        "position_effect": mean(position_effects),
        "swap_consistency": mean([float(x) for x in swap_agreements]),
        "primary_label_rate": mean([0.5 if row["selected_model"] == "TIE" else float(row["selected_primary_label"]) for row in valid]),
        "label_effect": mean(label_effects),
        "label_mapping_consistency": mean([float(x) for x in mapping_agreements]),
        "wording_all_agreement": mean([float(x) for x in wording_agreements]),
        "label_scheme_consistency": mean([float(x) for x in scheme_agreements]),
        "per_wording": per_wording,
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    metrics = {}
    for model in MODELS:
        rows = read_jsonl(args.out_dir / "protocol_outputs" / f"{model}.jsonl")
        metrics[model] = judge_metrics(rows)
    (args.out_dir / "protocol_calibration_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 评委协议小规模校准",
        "",
        "固定 12 个跨家族逻辑文本对，完全交叉 3 种等价措辞、2 套标签、标签正反映射与 AB/BA 方向；每位评委 288 次。位置效应和标签效应均通过同一文本对内的平衡对照显式估计。",
        "",
        "| 评委 | 无效 | 平局 | 首位置率 | 位置效应 | 换序一致 | 主标签率 | 标签效应 | 标签映射一致 | 三措辞全一致 | 标签体系一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, item in metrics.items():
        lines.append(
            f"| {model} | {fmt(item['invalid_rate'])} | {fmt(item['tie_rate'])} | {fmt(item['first_position_rate'])} | "
            f"{fmt(item['position_effect'], True)} | {fmt(item['swap_consistency'])} | {fmt(item['primary_label_rate'])} | "
            f"{fmt(item['label_effect'], True)} | {fmt(item['label_mapping_consistency'])} | "
            f"{fmt(item['wording_all_agreement'])} | {fmt(item['label_scheme_consistency'])} |"
        )
    lines.extend(["", "## 分措辞的换序表现", ""])
    for wording in WORDINGS:
        lines.extend([
            f"### {wording}", "",
            "| 评委 | 无效 | 首位置率 | 位置效应 | 换序一致 |",
            "|---|---:|---:|---:|---:|",
        ])
        for model, item in metrics.items():
            value = item["per_wording"][wording]
            lines.append(
                f"| {model} | {fmt(value['invalid_rate'])} | {fmt(value['first_position_rate'])} | "
                f"{fmt(value['position_effect'], True)} | {fmt(value['swap_consistency'])} |"
            )
        lines.append("")
    lines.extend([
        "## 指标解释", "",
        "- 位置效应：同一逻辑文本在第一位而非第二位时，被选择概率的配对差；0 最理想。",
        "- 标签效应：同一逻辑文本获得主标签 A/1 而非 B/2 时，被选择概率的配对差；0 最理想。",
        "- 换序一致：AB/BA 后仍选择同一底层文本的比例。",
        "- 标签映射一致：显示顺序不变、只交换标签时仍选择同一底层文本的比例。",
        "- 三措辞全一致：三个等价 judge prompt 均选择同一底层文本的比例。",
        "- 标签体系一致：A/B/T 与 1/2/0 两种表达选择同一底层文本的比例。",
        "",
        "校准用于决定协议，不用于报告家族效应。只有当协议在多数评委上同时具有低位置/标签效应、高换序/映射一致率和低无效率时，才扩大正式实验。",
    ])
    report = "\n".join(lines) + "\n"
    (args.out_dir / "protocol_calibration_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-family-pair", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", required=True, choices=MODELS)
    run.set_defaults(func=cmd_run)
    analyze = sub.add_parser("analyze")
    analyze.set_defaults(func=cmd_analyze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.work_dir = args.work_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
