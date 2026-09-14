#!/usr/bin/env python3
"""Run and analyze a small same-order repeat audit with complete raw logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_open_family_pilot20 import (
    JUDGE_SYSTEM,
    JUDGE_TEMPLATE,
    MODELS,
    SEED,
    append_jsonl,
    load_engine,
    parse_judgment,
    read_jsonl,
    release_engine,
    render_chat,
    utc_now,
)


def family_pair(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((MODELS[row["model_a"]]["family"], MODELS[row["model_b"]]["family"])))


def prepare_plan(work_dir: Path, audit_dir: Path, per_family_pair: int) -> list[dict[str, Any]]:
    plan_path = audit_dir / "repeat_plan.jsonl"
    if plan_path.exists():
        return read_jsonl(plan_path)
    rows = [row for row in read_jsonl(work_dir / "private" / "pairs.jsonl") if row["orientation"] == "AB"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = family_pair(row)
        if key[0] != key[1]:
            groups[key].append(row)
    selected = []
    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda row: hashlib.sha256(f"{SEED}:{row['judgment_id']}".encode()).hexdigest(),
        )
        selected.extend(candidates[:per_family_pair])
    random.Random(SEED).shuffle(selected)
    audit_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected, 1):
        append_jsonl(plan_path, {"audit_item_id": f"R{index:03d}", **row})
    return read_jsonl(plan_path)


def cmd_run(args: argparse.Namespace) -> int:
    from vllm import SamplingParams

    plan = prepare_plan(args.work_dir, args.audit_dir, args.per_family_pair)
    out = args.audit_dir / "same_order_repeats" / f"{args.model}.jsonl"
    completed = {
        (row["audit_item_id"], row["repeat_index"])
        for row in read_jsonl(out)
        if row.get("status") in {"ok", "invalid"}
    }
    pending_repeats = [index for index in range(1, args.repeats + 1) if any((row["audit_item_id"], index) not in completed for row in plan)]
    if not pending_repeats:
        print(f"same-order audit already complete for {args.model}")
        return 0
    llm, tokenizer = load_engine(args.model)
    for repeat_index in pending_repeats:
        current = [row for row in plan if (row["audit_item_id"], repeat_index) not in completed]
        rendered = []
        for row in current:
            user_prompt = JUDGE_TEMPLATE.format(setting=row["setting"], text_a=row["text_a"], text_b=row["text_b"])
            if MODELS[args.model]["family"] == "hunyuan":
                user_prompt = "/no_think\n" + user_prompt
            rendered.append(render_chat(tokenizer, JUDGE_SYSTEM, user_prompt))
        sampling = SamplingParams(temperature=0.0, max_tokens=96, seed=SEED)
        outputs = llm.generate(rendered, sampling, use_tqdm=True)
        for row, prompt, output in zip(current, rendered, outputs):
            raw = output.outputs[0].text.strip()
            record = {
                "audit_item_id": row["audit_item_id"],
                "judgment_id": row["judgment_id"],
                "canonical_pair_id": row["canonical_pair_id"],
                "prompt_id": row["prompt_id"],
                "orientation": row["orientation"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "judge_model": args.model,
                "judge_family": MODELS[args.model]["family"],
                "repeat_index": repeat_index,
                "raw": raw,
                "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_token_count": len(tokenizer.encode(prompt, add_special_tokens=False)),
                "output_token_count": len(output.outputs[0].token_ids),
                "temperature": 0.0,
                "seed": SEED,
                "runtime_note": "SQZ vLLM repeat audit; compare repeat-vs-repeat as primary stability estimate",
                "recorded_at": utc_now(),
            }
            try:
                winner, confidence, parse_method = parse_judgment(raw)
                record.update(
                    {
                        "status": "ok",
                        "winner": winner,
                        "confidence": confidence,
                        "parse_method": parse_method,
                        "selected_model": row["model_a"] if winner == "A" else row["model_b"] if winner == "B" else "TIE",
                    }
                )
            except Exception as exc:
                record.update({"status": "invalid", "error": f"{type(exc).__name__}: {exc}"})
            append_jsonl(out, record)
    release_engine(llm)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    plan = prepare_plan(args.work_dir, args.audit_dir, args.per_family_pair)
    original: dict[str, dict[str, dict[str, Any]]] = {}
    for judge in MODELS:
        original[judge] = {
            row["judgment_id"]: row
            for row in read_jsonl(args.work_dir / "judgments" / f"{judge}.jsonl")
            if row.get("status") == "ok"
        }
    metrics = {}
    for judge in MODELS:
        rows = read_jsonl(args.audit_dir / "same_order_repeats" / f"{judge}.jsonl")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["audit_item_id"]].append(row)
        comparable = []
        raw_comparable = []
        original_comparisons = []
        for item in plan:
            reps = sorted(grouped[item["audit_item_id"]], key=lambda row: row["repeat_index"])
            valid = [row for row in reps if row.get("status") == "ok"]
            if len(valid) == args.repeats:
                comparable.append(len({row["selected_model"] for row in valid}) == 1)
                raw_comparable.append(len({row["raw"] for row in valid}) == 1)
                old = original[judge].get(item["judgment_id"])
                if old:
                    original_comparisons.extend(row["selected_model"] == old["selected_model"] for row in valid)
        valid_rows = [row for row in rows if row.get("status") == "ok"]
        metrics[judge] = {
            "calls": len(rows),
            "invalid_calls": sum(row.get("status") != "ok" for row in rows),
            "invalid_rate": sum(row.get("status") != "ok" for row in rows) / len(rows) if rows else None,
            "a_rate": sum(row.get("winner") == "A" for row in valid_rows) / len(valid_rows) if valid_rows else None,
            "repeat_items_comparable": len(comparable),
            "same_order_choice_agreement": sum(comparable) / len(comparable) if comparable else None,
            "same_order_raw_exact_agreement": sum(raw_comparable) / len(raw_comparable) if raw_comparable else None,
            "agreement_with_original": sum(original_comparisons) / len(original_comparisons) if original_comparisons else None,
        }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "same_order_repeat_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 相同顺序重复判断审计",
        "",
        f"固定抽取 {len(plan)} 个匿名方向记录；每位评委在完全相同的顺序、提示和 temperature=0 设置下独立重复 {args.repeats} 次。每次 raw、输入哈希和 token 数均保留。",
        "",
        "| 评委 | 调用数 | 无效率 | 重复选择一致率 | raw完全一致率 | 与原始判断一致率 | 重复中的A率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for judge, item in metrics.items():
        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{100 * value:.1f}%"
        lines.append(
            f"| {judge} | {item['calls']} | {fmt(item['invalid_rate'])} | {fmt(item['same_order_choice_agreement'])} | "
            f"{fmt(item['same_order_raw_exact_agreement'])} | {fmt(item['agreement_with_original'])} | {fmt(item['a_rate'])} |"
        )
    lines.extend([
        "",
        "说明：重复选择一致率直接衡量相同顺序下的稳定性；它与原实验的 AB/BA 换序一致率是两个不同量。与原始判断一致率还受到运行环境版本差异影响，仅作辅助参考。",
    ])
    (args.audit_dir / "same_order_repeat_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--per-family-pair", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
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
    args.audit_dir = args.audit_dir.resolve()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
