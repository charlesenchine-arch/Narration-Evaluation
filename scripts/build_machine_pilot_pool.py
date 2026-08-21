#!/usr/bin/env python3
"""Combine successful API and manually captured outputs into one machine pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    Path("data/human_pilot/generated/aihubmix_gemini-3.5-flash-lite-free.jsonl"),
    Path("data/human_pilot/generated/aihubmix_gemini-3.6-flash-free.jsonl"),
    Path("data/human_pilot/generated/manual_deepseek_no_thinking_no_search.jsonl"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_name(row: dict[str, Any]) -> str:
    if row.get("provider") == "deepseek":
        return "deepseek_ui_unspecified__thinking_off__search_off"
    model = str(row.get("requested_model") or "unknown_model")
    effort = str(row.get("reasoning_effort") or "unrecorded")
    return f"aihubmix__{model}__reasoning_{effort}"


def combine(inputs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen_machine_ids: set[str] = set()
    input_manifest: list[dict[str, Any]] = []
    for path in inputs:
        rows = read_jsonl(path)
        successful = [row for row in rows if row.get("status") == "ok"]
        input_manifest.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "successful_rows": len(successful),
            }
        )
        for row in successful:
            machine_id = str(row["machine_id"])
            if machine_id in seen_machine_ids:
                raise ValueError(f"Duplicate machine_id: {machine_id}")
            seen_machine_ids.add(machine_id)
            combined.append(
                {
                    **row,
                    "generator_condition": condition_name(row),
                    "source_file": str(path),
                }
            )
    combined.sort(
        key=lambda row: (
            str(row.get("human_id")),
            str(row.get("generator_condition")),
        )
    )
    per_prompt = Counter(str(row["human_id"]) for row in combined)
    per_condition = Counter(str(row["generator_condition"]) for row in combined)
    manifest = {
        "schema_version": "machine-pilot-pool-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_successful_samples": len(combined),
        "unique_prompts": len(per_prompt),
        "length_compliant_samples": sum(
            bool(row.get("length_in_range")) for row in combined
        ),
        "samples_per_prompt_distribution": dict(
            sorted(Counter(per_prompt.values()).items())
        ),
        "samples_per_condition": dict(sorted(per_condition.items())),
        "inputs": input_manifest,
    }
    return combined, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/human_pilot/generated/machine_pilot_pool.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/human_pilot_20260821/machine_pilot_pool_manifest.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, manifest = combine(args.inputs)
    write_jsonl(args.output, rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"combined: {manifest['total_successful_samples']} samples across "
        f"{manifest['unique_prompts']} prompts"
    )
    print(f"output: {args.output.resolve()}")
    print(f"manifest: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
