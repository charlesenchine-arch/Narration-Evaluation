#!/usr/bin/env python3
"""Generate pilot narratives through AIHubMix's OpenAI-compatible API.

Safety defaults:
- uses the preferred endpoint https://api.inferera.com/v1;
- accepts credentials only from AIHUBMIX_API_KEY or macOS Keychain;
- dry-runs unless --execute is supplied;
- requires an explicit model id and billing acknowledgement for generation;
- sends approved prompts only, never paired human reference texts.

AIHubMix is a third-party aggregator. A successful request may consume account
credit or incur a charge; this script cannot infer billing status or current
model pricing.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://api.inferera.com/v1"
ALLOWED_BASE_URLS = {
    "https://api.inferera.com/v1",
    "https://aihubmix.com/v1",
}
KEYCHAIN_SERVICE = "narration-aihubmix-api"
REVIEWED_FREE_MODEL_ALLOWLIST = {
    "gemini-3.6-flash-free",
    "gemini-3.5-flash-lite-free",
    "coding-kimi-k3-free",
}


def normalize_base_url(value: str) -> str:
    """Normalize the two documented AIHubMix endpoints and reject others."""
    base_url = value.strip().rstrip("/")
    if base_url in {"https://api.inferera.com", "https://aihubmix.com"}:
        base_url += "/v1"
    if base_url not in ALLOWED_BASE_URLS:
        raise ValueError(
            "AIHUBMIX_BASE_URL must be one of: " + ", ".join(sorted(ALLOWED_BASE_URLS))
        )
    return base_url


def load_api_key() -> tuple[str, str]:
    """Load a key without putting it in source code or command-line history."""
    env_key = os.environ.get("AIHUBMIX_API_KEY", "").strip()
    if env_key:
        return env_key, "environment"

    if sys.platform == "darwin":
        account = os.environ.get("USER", "").strip() or getpass.getuser()
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                account,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        keychain_key = result.stdout.strip()
        if result.returncode == 0 and keychain_key:
            return keychain_key, "macos_keychain"

    raise RuntimeError(
        "No AIHubMix key found. Set AIHUBMIX_API_KEY or store it in macOS "
        f"Keychain under service {KEYCHAIN_SERVICE!r}."
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_model_name(model: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    return value or "unknown_model"


def default_output_path(model: str) -> Path:
    return Path("data/human_pilot/generated") / f"aihubmix_{safe_model_name(model)}.jsonl"


def require_reviewed_free_model(model: str) -> None:
    if model not in REVIEWED_FREE_MODEL_ALLOWLIST:
        raise ValueError(
            f"model {model!r} is not in the reviewed free-only allowlist: "
            f"{sorted(REVIEWED_FREE_MODEL_ALLOWLIST)}"
        )


def is_free_quota_exhausted(message: str) -> bool:
    normalized = message.lower()
    return "limit of the free model quota" in normalized or (
        "error code: 429" in normalized and "free" in normalized
    )


def preview_rows(
    records: list[dict[str, Any]],
    model: str,
    base_url: str,
    max_items: int,
    reasoning_effort: str = "minimal",
) -> list[dict[str, str]]:
    """Build the exact non-sensitive fields that will be sent to the API."""
    return [
        {
            "human_id": str(row["human_id"]),
            "model": model,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "prompt": str(row["approved_prompt"]),
        }
        for row in records[:max_items]
    ]


def make_client(api_key: str, base_url: str, timeout: int) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def probe_models(client: Any) -> int:
    response = client.models.list()
    model_ids = sorted(str(item.id) for item in response.data)
    print(f"connection ok; visible models: {len(model_ids)}")
    for model_id in model_ids[:30]:
        print(model_id)
    if len(model_ids) > 30:
        print(f"... and {len(model_ids) - 30} more")
    return 0


def usage_dict(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def generation_qc(
    text: str, finish_reason: str | None, min_chars: int, max_chars: int
) -> dict[str, Any]:
    char_count = len(text)
    flags: list[str] = []
    if char_count < min_chars:
        flags.append("below_min_chars")
    if char_count > max_chars:
        flags.append("above_max_chars")
    if finish_reason != "stop":
        flags.append("non_stop_finish")
    return {
        "text_char_count": char_count,
        "length_in_range": min_chars <= char_count <= max_chars,
        "protocol_flags": flags,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/human_pilot/private/approved_human.jsonl"),
    )
    parser.add_argument("--model", help="Exact AIHubMix model id; required for generation")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AIHUBMIX_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default="minimal",
        help=(
            "AIHubMix unified reasoning setting. Gemini 3.x cannot guarantee fully "
            "disabled thinking; minimal is the reproducible low-reasoning default."
        ),
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Check credentials and list visible model ids without generating text",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed item and continue the batch instead of stopping",
    )
    parser.add_argument(
        "--acknowledge-billing",
        action="store_true",
        help="Confirm that requests may consume AIHubMix balance or incur charges",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = normalize_base_url(args.base_url)
    if not 1 <= args.max_items <= 100:
        raise ValueError("--max-items must be between 1 and 100")
    if not 1 <= args.max_output_tokens <= 32768:
        raise ValueError("--max-output-tokens must be between 1 and 32768")
    if not 1 <= args.min_chars <= args.max_chars:
        raise ValueError("character limits must satisfy 1 <= --min-chars <= --max-chars")

    if args.probe:
        api_key, key_source = load_api_key()
        print(f"base URL: {base_url}; credential source: {key_source}")
        return probe_models(make_client(api_key, base_url, args.timeout))

    if not args.model or not args.model.strip():
        raise ValueError("--model is required; copy the exact id from the AIHubMix model page")
    model = args.model.strip()
    require_reviewed_free_model(model)
    records = read_jsonl(args.input)
    preview = preview_rows(
        records, model, base_url, args.max_items, args.reasoning_effort
    )
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(
            f"dry run only: {len(preview)} prompts; no API request was sent. "
            "Verify the model price, then add --execute --acknowledge-billing."
        )
        return 0
    if not args.acknowledge_billing:
        raise RuntimeError(
            "Generation may consume AIHubMix balance. Re-run with "
            "--acknowledge-billing after checking the selected model's current price."
        )

    api_key, key_source = load_api_key()
    client = make_client(api_key, base_url, args.timeout)
    output = args.output or default_output_path(model)
    completed: set[str] = set()
    if output.exists():
        completed = {
            str(row.get("human_id"))
            for row in read_jsonl(output)
            if row.get("status") == "ok"
            and row.get("requested_model") == model
            and row.get("base_url") == base_url
        }

    print(
        f"executing {len(preview)} request(s); model={model}; base URL={base_url}; "
        f"credential source={key_source}"
    )
    for index, row in enumerate(preview):
        human_id = row["human_id"]
        if human_id in completed:
            print(f"skip completed: {human_id}")
            continue
        base_record = {
            "machine_id": f"AIHUBMIX_{safe_model_name(model)}_{human_id}",
            "human_id": human_id,
            "source_type": "machine",
            "provider": "aihubmix",
            "access_surface": "aihubmix_promotional_free_model",
            "base_url": base_url,
            "requested_model": model,
            "reasoning_effort": args.reasoning_effort,
            "prompt": row["prompt"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": row["prompt"]}],
                max_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            choice = completion.choices[0]
            text = (choice.message.content or "").strip()
            if not text:
                raise RuntimeError("API returned no text")
            finish_reason = getattr(choice, "finish_reason", None)
            qc = generation_qc(text, finish_reason, args.min_chars, args.max_chars)
            append_jsonl(
                output,
                {
                    **base_record,
                    "status": "ok",
                    "text": text,
                    "returned_model": getattr(completion, "model", None),
                    "response_id": getattr(completion, "id", None),
                    "usage_metadata": usage_dict(completion),
                    "finish_reason": finish_reason,
                    **qc,
                },
            )
            print(
                f"completed: {human_id}; chars={qc['text_char_count']}; "
                f"length_in_range={qc['length_in_range']}"
            )
        except Exception as exc:
            safe_message = str(exc).replace(api_key, "<redacted>")[:2000]
            append_jsonl(
                output,
                {**base_record, "status": "api_error", "error": safe_message},
            )
            print(f"failed: {human_id}; {safe_message}")
            if is_free_quota_exhausted(safe_message):
                print("free quota exhausted; stopping without paid fallback")
                break
            if not args.continue_on_error:
                raise RuntimeError(f"AIHubMix request failed: {safe_message}") from exc
        if index + 1 < len(preview):
            time.sleep(args.delay)

    print(f"output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
