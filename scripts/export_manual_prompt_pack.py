#!/usr/bin/env python3
"""Export the human-pilot prompts as an API queue and readable Markdown pack.

The exported files contain prompt text and identifiers only. Human reference
texts, source prompts, and source metadata are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def clean(value: str | None) -> str:
    return (value or "").strip()


def read_prompts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    prompts: list[dict[str, str]] = []
    for row in rows:
        prompt = clean(row.get("approved_prompt")) or clean(row.get("prompt_candidate"))
        if not prompt:
            continue
        approval = clean(row.get("approve_text")).lower()
        prompts.append(
            {
                "human_id": clean(row.get("human_id")),
                "approved_prompt": prompt,
                "human_text_review_status": "approved" if approval == "yes" else "pending",
            }
        )
    return prompts


def write_jsonl(path: Path, prompts: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, prompts: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 人机叙事 pilot：手工生成 Prompt 包",
        "",
        f"共 {len(prompts)} 条。每条 Prompt 必须在一个全新聊天中单独发送。",
        "不要附带人类参考文本，不要追问或要求模型二次润色；保存首轮完整回答。",
        "当前 `human_text_review_status=pending` 表示配对的人类文本尚未在审核表中正式勾选通过，不影响 Prompt 本身的复制使用。",
        "",
    ]
    for index, row in enumerate(prompts, start=1):
        lines.extend(
            [
                f"## {index:02d}. {row['human_id']}",
                "",
                f"审核状态：`{row['human_text_review_status']}`",
                "",
                "```text",
                row["approved_prompt"],
                "```",
                "",
                "模型：____________________",
                "",
                "生成日期：________________",
                "",
                "输出保存位置/编号：________",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/human_pilot/review/human_candidates_review.csv"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("data/human_pilot/private/all_prompt_queue.jsonl"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("outputs/human_pilot_20260821/all_prompts_manual.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = read_prompts(args.input)
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.input}")
    write_jsonl(args.jsonl_output, prompts)
    write_markdown(args.markdown_output, prompts)
    approved = sum(row["human_text_review_status"] == "approved" for row in prompts)
    print(f"prompts: {len(prompts)}; human texts formally approved: {approved}")
    print(f"API queue: {args.jsonl_output.resolve()}")
    print(f"manual pack: {args.markdown_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
