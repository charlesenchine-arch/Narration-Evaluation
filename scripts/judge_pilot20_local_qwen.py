#!/usr/bin/env python3
"""Blindly judge pilot-20 pairs with a local Qwen3-VL instruction model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GRADE_MINIMUMS = {"A": 92.0, "B": 80.0, "C": 65.0, "D": 50.0, "F": 0.0}
DIMENSIONS = {
    "plot_causality": "情节与因果：事件推进、铺垫、转折和结果是否成立",
    "character": "人物塑造：动机、行为、关系和变化是否可信且一致",
    "language_style": "语言表现：表达、声音、清晰度和文体控制",
    "atmosphere_impact": "氛围与感染力：情绪、场景和阅读投入",
    "pacing": "节奏：信息密度、场景展开和收束是否合适",
    "narrative_structure_time": "叙事结构与时间：顺序、视角、时距和时间组织",
    "originality": "新颖性：构思、意象和处理方式是否避免套板",
}


def read_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(text_a: str, text_b: str) -> str:
    dimensions = "\n".join(f"- {key}: {value}" for key, value in DIMENSIONS.items())
    return f"""你是一名中文叙事编辑。请盲评下面两篇文本的整体叙事质量，不要猜测作者来源。

先形成内部连续分数（0-100）与 A/B/C/D/F 等级，再判断哪篇整体更好。维度仅作为判断依据，
不能机械等权平均；总体判断优先。必须引用文本中的具体内容，禁止只写通用套话。

ABCDF 标准：A=92—100顶级；B=80—91.99优秀；C=65—79.99合格；
D=50—64.99较弱；F=0—49.99未达标。
偏好强度：2=A明显更好，1=A稍好，0=相当，-1=B稍好，-2=B明显更好。

内部判断维度：
{dimensions}

文本 A：
{text_a}

文本 B：
{text_b}

只输出 JSON：
{{
  "preference": -2|-1|0|1|2,
  "score_a": 0-100,
  "score_b": 0-100,
  "grade_a": "A"|"B"|"C"|"D"|"F",
  "grade_b": "A"|"B"|"C"|"D"|"F",
  "primary_dimensions": ["维度键"],
  "evidence_a": "A中的具体证据",
  "evidence_b": "B中的具体证据",
  "critique": "不超过120字的比较评价"
}}"""


def grade_for_score(score: float) -> str:
    for grade, minimum in GRADE_MINIMUMS.items():
        if score >= minimum:
            return grade
    raise ValueError("score out of range")


def parse_response(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response has no JSON object")
    obj = json.loads(text[start : end + 1])
    original_preference = int(obj["preference"])
    if original_preference not in {-2, -1, 0, 1, 2}:
        raise ValueError("invalid preference")
    score_a, score_b = float(obj["score_a"]), float(obj["score_b"])
    if not 0 <= score_a <= 100 or not 0 <= score_b <= 100:
        raise ValueError("score out of range")
    grade_a, grade_b = grade_for_score(score_a), grade_for_score(score_b)
    dimensions = list(obj.get("primary_dimensions") or [])
    if set(dimensions) - set(DIMENSIONS):
        raise ValueError("unknown dimensions")
    difference = score_a - score_b
    if abs(difference) <= 2:
        preference = 0
    elif difference > 0:
        preference = 2 if difference >= 8 else 1
    else:
        preference = -2 if difference <= -8 else -1
    normalizations: list[str] = []
    if preference != original_preference:
        normalizations.append("preference_recomputed_from_scores")
    return {
        "preference": preference,
        "original_preference": original_preference,
        "score_a": score_a,
        "score_b": score_b,
        "grade_a": grade_a,
        "grade_b": grade_b,
        "primary_dimensions": dimensions,
        "evidence_a": str(obj.get("evidence_a", "")).strip(),
        "evidence_b": str(obj.get("evidence_b", "")).strip(),
        "critique": str(obj.get("critique", "")).strip(),
        "normalizations": normalizations,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_successes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            str(row["pair_id"])
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
            if row.get("status") == "ok"
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/human_pilot/pilot20/blind/human_machine_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/local_qwen_judgments.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/nvme/jqhua/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=7168)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--audit-one-per-prompt",
        action="store_true",
        help="Select one rotating generator pair per prompt for a position audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    pairs = read_pairs(args.pairs)
    if args.audit_one_per_prompt:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            grouped.setdefault(str(pair.get("split_group", "")), []).append(pair)
        pairs = [
            group[index % len(group)]
            for index, (_, group) in enumerate(grouped.items())
        ]
    if args.limit:
        pairs = pairs[: args.limit]
    completed = existing_successes(args.output)
    pending = [pair for pair in pairs if str(pair["id"]) not in completed]
    print(f"pairs={len(pairs)} completed={len(completed)} pending={len(pending)}")
    if not pending:
        return 0

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()

    for index, pair in enumerate(pending, 1):
        pair_id = str(pair["id"])
        item_a, item_b = pair["item_a"], pair["item_b"]
        if args.reverse:
            item_a, item_b = item_b, item_a
        prompt = build_prompt(item_a["text"], item_b["text"])
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        raw = ""
        try:
            try:
                rendered = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            inputs = processor(
                text=[rendered],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prefix = inputs["input_ids"].shape[1]
            raw = processor.batch_decode(
                generated[:, prefix:], skip_special_tokens=True
            )[0].strip()
            parsed = parse_response(raw)
            record = {
                "pair_id": pair_id,
                "status": "ok",
                "result": parsed,
                "raw_response": raw,
                "model": str(args.model),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "judged_at": datetime.now(timezone.utc).isoformat(),
                "source_labels_sent": False,
                "orientation": "reversed" if args.reverse else "original",
            }
        except Exception as exc:
            record = {
                "pair_id": pair_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "raw_response": raw,
                "model": str(args.model),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "judged_at": datetime.now(timezone.utc).isoformat(),
                "source_labels_sent": False,
                "orientation": "reversed" if args.reverse else "original",
            }
        append_jsonl(args.output, record)
        print(f"{index}/{len(pending)} {pair_id}: {record['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
