"""Run an OpenAI-compatible LLM-as-a-Judge pairwise baseline with resume."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from narrative_evaluator.quality.judge import build_pairwise_judge_prompt, parse_pairwise_judge_response
from narrative_evaluator.quality.schemas import NarrativePair, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="运行成对叙事 LLM Judge 基线")
    parser.add_argument("--pairs", default="data/eval_dataset/pairwise/pairs.jsonl")
    parser.add_argument("--out", default="data/eval_dataset/pairwise/llm_judgments.jsonl")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("需要 JUDGE_API_KEY 或 DEEPSEEK_API_KEY 环境变量")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url)

    pairs = read_jsonl(args.pairs, NarrativePair)
    if args.limit:
        pairs = pairs[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["pair_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}

    with open(out, "a", encoding="utf-8") as handle:
        for pair in pairs:
            if pair.id in done:
                continue
            prompt = build_pairwise_judge_prompt(pair.item_a.text, pair.item_b.text)
            parsed = None
            error = ""
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=700,
                    )
                    parsed = parse_pairwise_judge_response(response.choices[0].message.content)
                    break
                except Exception as exc:
                    error = str(exc)
                    time.sleep(1.5 * (attempt + 1))
            record = {
                "pair_id": pair.id,
                "model": args.model,
                "base_url": args.base_url,
                "result": parsed,
                "error": error if parsed is None else "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{pair.id}: {'ok' if parsed else 'failed'}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
