#!/usr/bin/env python3
"""Score the pilot-20 narratives with a local sequence-classification reward model.

The model receives only the reverse prompt and narrative text. Source and generator
labels remain in the private key file but are deliberately excluded from inference.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/items_keyed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/local_reward_scores.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/nvme/jqhua/models/Llama-3-8b-rm-700k"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 128:
        raise ValueError("invalid batch size or max length")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = read_jsonl(args.items)
    if args.limit:
        rows = rows[: args.limit]
    completed: set[str] = set()
    if args.output.exists():
        completed = {
            str(row["blind_id"])
            for row in read_jsonl(args.output)
            if row.get("status") == "ok"
        }
    pending = [row for row in rows if str(row["blind_id"]) not in completed]
    print(f"items={len(rows)} completed={len(completed)} pending={len(pending)}")
    if not pending:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=True,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        rendered: list[str] = []
        for row in batch:
            messages = [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["text"])},
            ]
            rendered.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)
        with torch.inference_mode():
            scores = model(**encoded).logits.float().view(-1).cpu().tolist()
        token_counts = encoded["attention_mask"].sum(dim=1).cpu().tolist()
        for row, score, token_count in zip(batch, scores, token_counts):
            append_jsonl(
                args.output,
                {
                    "blind_id": str(row["blind_id"]),
                    "blind_prompt_id": str(row["blind_prompt_id"]),
                    "status": "ok",
                    "score": float(score),
                    "input_tokens": int(token_count),
                    "truncated": int(token_count) >= args.max_length,
                    "model": str(args.model),
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                    "source_labels_sent": False,
                },
            )
        done = min(start + len(batch), len(pending))
        print(f"{done}/{len(pending)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
