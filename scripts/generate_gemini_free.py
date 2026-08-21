#!/usr/bin/env python3
"""Generate machine narratives through the closed Gemini API free-tier model.

Safety defaults:
- dry-run unless --execute is supplied;
- reads GEMINI_API_KEY only from the environment;
- caps each run with --max-items (default: 5);
- sends approved prompts only, never the paired human reference text.

The API cannot determine whether billing is enabled on the user's Google project.
Use a project configured for the Gemini Developer API free tier and verify its
current quota/pricing before executing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
FREE_MODEL_ALLOWLIST = {"gemini-3.6-flash"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_generation(api_key: str, model: str, prompt: str, timeout: int) -> dict[str, Any]:
    url = f"{API_ROOT}/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 4096,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_text(response: dict[str, Any]) -> str:
    chunks = []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                chunks.append(str(part["text"]))
        if chunks:
            break
    return "".join(chunks).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/human_pilot/private/approved_human.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/human_pilot/generated/gemini_2_5_flash.jsonl"),
    )
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model not in FREE_MODEL_ALLOWLIST:
        raise ValueError(
            f"model {args.model!r} is not in this script's reviewed free-model allowlist: "
            f"{sorted(FREE_MODEL_ALLOWLIST)}"
        )
    if not 1 <= args.max_items <= 100:
        raise ValueError("--max-items must be between 1 and 100")
    records = read_jsonl(args.input)
    work = records[: args.max_items]
    preview = [
        {"human_id": row["human_id"], "model": args.model, "prompt": row["approved_prompt"]}
        for row in work
    ]
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(f"dry run only: {len(work)} prompts; add --execute after verifying the free-tier project")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; export it locally and never commit it")
    completed = set()
    if args.output.exists():
        completed = {
            str(row.get("human_id"))
            for row in read_jsonl(args.output)
            if row.get("status") == "ok" and row.get("model") == args.model
        }
    for index, row in enumerate(work):
        if row["human_id"] in completed:
            continue
        base = {
            "machine_id": f"GEMINI_{row['human_id']}",
            "human_id": row["human_id"],
            "source_type": "machine",
            "provider": "google",
            "access_surface": "gemini_developer_api_free_tier",
            "model": args.model,
            "prompt": row["approved_prompt"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            response = request_generation(api_key, args.model, row["approved_prompt"], args.timeout)
            text = extract_text(response)
            if not text:
                raise RuntimeError("API returned no text")
            append_jsonl(
                args.output,
                {
                    **base,
                    "status": "ok",
                    "text": text,
                    "model_version": response.get("modelVersion"),
                    "response_id": response.get("responseId"),
                    "usage_metadata": response.get("usageMetadata", {}),
                    "finish_reason": response.get("candidates", [{}])[0].get("finishReason"),
                },
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            append_jsonl(
                args.output,
                {**base, "status": "http_error", "http_status": exc.code, "error": detail},
            )
            if exc.code in {401, 403, 429}:
                raise RuntimeError(f"Gemini request stopped on HTTP {exc.code}: {detail}") from exc
        if index + 1 < len(work):
            time.sleep(args.delay)
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
