#!/usr/bin/env python3
"""Import manually pasted DeepSeek outputs from the readable prompt pack.

The source Markdown is never modified. Filled responses are converted into the
same JSONL-style machine sample records used by API-generated outputs. The user
reported that Deep Thinking and web search were both disabled; the exact model
version was not recorded in the document and is therefore kept as unknown.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(r"(?m)^##\s+\d+\.\s+(?P<human_id>[A-Za-z0-9_]+)\s*$")
CONTENT_RE = re.compile(
    r"```text\r?\n(?P<prompt>.*?)\r?\n```\r?\n(?P<text>.*?)(?=^模型：)",
    re.DOTALL | re.MULTILINE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_filled_sections(path: Path) -> list[dict[str, str]]:
    source = path.read_text(encoding="utf-8")
    matches = list(SECTION_RE.finditer(source))
    parsed: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        section = source[match.end() : end]
        content = CONTENT_RE.search(section)
        if not content:
            raise ValueError(f"Cannot parse prompt block for {match.group('human_id')}")
        prompt = content.group("prompt").strip()
        text = content.group("text").strip()
        if text:
            parsed.append(
                {
                    "human_id": match.group("human_id"),
                    "prompt": prompt,
                    "text": text,
                }
            )
    return parsed


def qc_fields(text: str, min_chars: int, max_chars: int) -> dict[str, Any]:
    char_count = len(text)
    flags = ["model_version_unrecorded", "generation_time_unrecorded"]
    if char_count < min_chars:
        flags.append("below_min_chars")
    if char_count > max_chars:
        flags.append("above_max_chars")
    return {
        "text_char_count": char_count,
        "length_in_range": min_chars <= char_count <= max_chars,
        "protocol_flags": flags,
    }


def build_records(
    filled: list[dict[str, str]],
    queue: dict[str, str],
    source_path: Path,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    captured_at = datetime.fromtimestamp(
        source_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in filled:
        human_id = row["human_id"]
        if human_id in seen:
            raise ValueError(f"Duplicate filled section: {human_id}")
        seen.add(human_id)
        if human_id not in queue:
            raise ValueError(f"Unknown human_id in Markdown: {human_id}")
        if row["prompt"] != queue[human_id]:
            raise ValueError(f"Prompt differs from the approved queue for {human_id}")
        records.append(
            {
                "machine_id": f"DEEPSEEK_MANUAL_{human_id}",
                "human_id": human_id,
                "source_type": "machine",
                "provider": "deepseek",
                "access_surface": "deepseek_consumer_chat_manual",
                "requested_model": "deepseek_ui_unspecified",
                "returned_model": None,
                "model_version_status": "not_displayed_or_not_recorded",
                "reasoning_effort": "deep_thinking_disabled",
                "web_search_enabled": False,
                "settings_source": "user_reported",
                "prompt": row["prompt"],
                "text": row["text"],
                "status": "ok",
                "generated_at": None,
                "captured_at": captured_at,
                "finish_reason": "not_available_manual_capture",
                "usage_metadata": {},
                **qc_fields(row["text"], min_chars, max_chars),
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/human_pilot_20260821/all_prompts_manual.md"),
    )
    parser.add_argument(
        "--prompt-queue",
        type=Path,
        default=Path("data/human_pilot/private/all_prompt_queue.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/human_pilot/generated/manual_deepseek_no_thinking_no_search.jsonl"
        ),
    )
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.min_chars <= args.max_chars:
        raise ValueError("character limits must satisfy 1 <= min <= max")
    queue_rows = read_jsonl(args.prompt_queue)
    queue = {str(row["human_id"]): str(row["approved_prompt"]) for row in queue_rows}
    filled = parse_filled_sections(args.input)
    records = build_records(filled, queue, args.input, args.min_chars, args.max_chars)
    if not records:
        raise RuntimeError(f"No filled responses found in {args.input}")
    write_jsonl(args.output, records)
    compliant = sum(bool(row["length_in_range"]) for row in records)
    print(f"imported: {len(records)}; length compliant: {compliant}")
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
