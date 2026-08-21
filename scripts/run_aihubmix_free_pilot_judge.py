#!/usr/bin/env python3
"""Run a blinded, free-only AIHubMix judge on the 20-prompt pilot pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_aihubmix import (
    DEFAULT_BASE_URL,
    is_free_quota_exhausted,
    load_api_key,
    make_client,
    normalize_base_url,
    usage_dict,
)
from narrative_evaluator.quality.judge import (
    build_pairwise_judge_prompt,
    parse_pairwise_judge_response,
)
from narrative_evaluator.quality.schemas import NarrativePair, read_jsonl


FREE_JUDGE_ALLOWLIST = {
    "gpt-5.5-free",
    "gpt-4.1-free",
    "qwen3.6-plus-preview-free",
}


def require_free_judge(model: str) -> None:
    if model not in FREE_JUDGE_ALLOWLIST:
        raise ValueError(
            f"judge {model!r} is not in the free-only allowlist: "
            f"{sorted(FREE_JUDGE_ALLOWLIST)}"
        )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
        default=Path("data/human_pilot/pilot20/private/gpt55_free_judgments.jsonl"),
    )
    parser.add_argument("--model", default="gpt-5.5-free")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=1500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.strip()
    require_free_judge(model)
    base_url = normalize_base_url(args.base_url)
    if not 1 <= args.max_pairs <= 1000:
        raise ValueError("--max-pairs must be between 1 and 1000")
    pairs = read_jsonl(args.pairs, NarrativePair)[: args.max_pairs]
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "model": model,
                    "base_url": base_url,
                    "reasoning_effort": args.reasoning_effort,
                    "pairs": len(pairs),
                    "source_labels_sent": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api_key, key_source = load_api_key()
    client = make_client(api_key, base_url, args.timeout)
    existing = read_existing(args.output)
    completed = {
        str(row["pair_id"])
        for row in existing
        if row.get("status") == "ok" and row.get("model") == model
    }
    print(
        f"judge={model}; pairs={len(pairs)}; completed={len(completed)}; "
        f"credential={key_source}"
    )
    for index, pair in enumerate(pairs):
        if pair.id in completed:
            continue
        prompt = build_pairwise_judge_prompt(pair.item_a.text, pair.item_b.text)
        base_record = {
            "pair_id": pair.id,
            "model": model,
            "base_url": base_url,
            "reasoning_effort": args.reasoning_effort,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "source_labels_sent": False,
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            parsed = parse_pairwise_judge_response(content)
            append_jsonl(
                args.output,
                {
                    **base_record,
                    "status": "ok",
                    "result": parsed,
                    "returned_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "finish_reason": response.choices[0].finish_reason,
                    "usage_metadata": usage_dict(response),
                },
            )
            print(f"{index + 1}/{len(pairs)} {pair.id}: ok", flush=True)
        except Exception as exc:
            safe_message = str(exc).replace(api_key, "<redacted>")[:2000]
            append_jsonl(
                args.output,
                {**base_record, "status": "error", "error": safe_message},
            )
            print(f"{index + 1}/{len(pairs)} {pair.id}: {safe_message}", flush=True)
            if is_free_quota_exhausted(safe_message):
                print("free quota exhausted; stopping without paid fallback", flush=True)
                break
        if index + 1 < len(pairs) and args.sleep:
            time.sleep(args.sleep)
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
