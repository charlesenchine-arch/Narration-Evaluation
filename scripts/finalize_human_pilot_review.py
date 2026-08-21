#!/usr/bin/env python3
"""Turn an audited human-pilot CSV into approved data and generation queues."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


YES = {"yes", "y", "1", "true", "approve", "approved", "通过", "是"}
TARGETS = ("chatgpt_free", "gemini_free", "claude_free")
QUEUE_FIELDS = [
    "job_id",
    "human_id",
    "target_system",
    "prompt",
    "status",
    "displayed_model",
    "generated_at",
    "conversation_url",
    "generated_text",
    "notes",
]


def clean(value: str | None) -> str:
    return (value or "").strip()


def prompt_fingerprint(prompt: str) -> str:
    compact = re.sub(r"\s+", "", prompt)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "review_csv",
        nargs="?",
        type=Path,
        default=Path("data/human_pilot/review/human_candidates_review.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/human_pilot/private"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    approved = []
    for row in rows:
        if clean(row.get("approve_text")).lower() not in YES:
            continue
        prompt = clean(row.get("approved_prompt"))
        if not prompt:
            raise ValueError(f"{row.get('human_id')}: approved text has an empty approved_prompt")
        record = dict(row)
        record["approved_prompt"] = prompt
        record["prompt_sha256"] = prompt_fingerprint(prompt)
        record["review_finalized_at"] = datetime.now(timezone.utc).isoformat()
        record["include_status"] = "approved_for_pilot"
        approved.append(record)

    if not approved:
        raise RuntimeError(
            "no approved rows: enter yes/通过 in approve_text and keep or edit approved_prompt"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    approved_path = args.out_dir / "approved_human.jsonl"
    with approved_path.open("w", encoding="utf-8") as handle:
        for record in approved:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    queue_path = args.out_dir / "manual_chat_queue.csv"
    with queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        for record in approved:
            for target in TARGETS:
                writer.writerow(
                    {
                        "job_id": f"{record['human_id']}__{target}",
                        "human_id": record["human_id"],
                        "target_system": target,
                        "prompt": record["approved_prompt"],
                        "status": "pending",
                        "displayed_model": "",
                        "generated_at": "",
                        "conversation_url": "",
                        "generated_text": "",
                        "notes": "first response; fresh chat; no web/search/tools; record the displayed model",
                    }
                )
    print(f"approved human texts: {len(approved)}")
    print(f"manual jobs: {len(approved) * len(TARGETS)}")
    print(f"approved: {approved_path.resolve()}")
    print(f"queue: {queue_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
