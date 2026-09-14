#!/usr/bin/env python3
"""Run the three-family, source-blind pilot-20 generation and rating experiment.

The experiment is deliberately stateless: every narrative generation and every
single-text rating is one independent API request.  Runtime artifacts live under
data/human_pilot/private/ and are ignored by git because they contain source
texts and the blind key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data/human_pilot/review/human_candidates.jsonl"
DEFAULT_WORK_DIR = ROOT / "data/human_pilot/private/api_family_pilot20"
PROVIDER_ORDER = ("deepseek", "gemini", "grok")
GRADE_BANDS = (("A", 92), ("B", 80), ("C", 65), ("D", 50), ("F", 0))


class BudgetError(RuntimeError):
    """Raised when a provider reports balance, quota, or billing exhaustion."""


class RateLimitError(RuntimeError):
    """Raised for a transient provider rate limit with a suggested delay."""

    def __init__(self, message: str, retry_after: float = 20.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    key_env: str
    endpoint: str
    input_usd_per_m: float
    output_usd_per_m: float
    reasoning_mode: str


PROVIDERS = {
    "deepseek": Provider(
        "deepseek",
        "deepseek-v4-pro",
        "DEEPSEEK_API_KEY",
        "https://api.deepseek.com/chat/completions",
        0.435,
        0.87,
        "thinking_disabled",
    ),
    "gemini": Provider(
        "gemini",
        "gemini-3.5-flash-lite",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/models",
        0.30,
        2.50,
        "thinking_minimal",
    ),
    "grok": Provider(
        "grok",
        "grok-4.20-0309-non-reasoning",
        "XAI_API_KEY",
        "https://api.x.ai/v1/chat/completions",
        1.25,
        2.50,
        "non_reasoning",
    ),
}


GENERATION_SYSTEM = (
    "你是中文短篇叙事写作者。严格执行用户给出的题材、体裁、完整性和字符数要求。"
    "不要使用搜索、工具、外部资料或既有对话；只返回最终故事正文。"
)

RATING_SYSTEM = (
    "你是中文叙事质量评审。你不知道作者或生成系统的身份，也不得猜测来源。"
    "只依据给定写作任务与正文评估叙事质量。每次判断相互独立，不引用任何其他文本或历史评分。"
)

RATING_TEMPLATE = """请独立评价下面这一篇匿名中文叙事文本。

原始写作任务：
{prompt}

评分原则：
- overall_score 为0—100整数，表示总体叙事质量，不是各维度机械平均。
- 重点考虑叙事完整性、情节与因果连贯、人物与视角、语言与节奏、情感/意象效果、原创性及任务遵循。
- 不因题材严肃、华丽辞藻、篇幅接近上限而自动加分。
- 不猜测作者身份，不把“像人”或“像AI”当作质量维度。
- 分档锚点：A=92—100，B=80—91，C=65—79，D=50—64，F=0—49。

只输出一个JSON对象，不要Markdown代码框，不要额外文字：
{{
  "overall_score": 0,
  "confidence": 1,
  "dimensions": {{
    "narrative_completeness": 1,
    "coherence": 1,
    "character_and_viewpoint": 1,
    "language_and_rhythm": 1,
    "emotional_and_imagery_effect": 1,
    "originality_and_task_compliance": 1
  }},
  "brief_reason": "不超过100个中文字符的依据"
}}

其中confidence和六个维度均为1—5整数。

匿名正文：
{text}"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def non_whitespace_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def length_valid(char_count: int, lower: int, upper: int) -> bool:
    """QC accepts a preregistered 3% upper counting tolerance.

    The prompt still states the hard 500..1200 target.  The small local tolerance
    handles ambiguity over whether Chinese punctuation is included in 字数 and is
    recorded per item; it is not used to choose among multiple valid drafts.
    """

    return lower <= char_count <= (upper * 103) // 100


def length_band(char_count: int) -> tuple[int, int, int]:
    if not 500 <= char_count <= 1200:
        raise ValueError(f"human reference outside the preregistered 500..1200 range: {char_count}")
    return 500, 1200, 850


def strict_prompt(candidate: dict[str, Any]) -> tuple[str, int, int]:
    original = str(candidate["prompt_candidate"]).strip()
    base = re.sub(r"请创作一篇500[—-]1200字的", "请创作一篇", original, count=1)
    base = base.replace("不要写创作说明，", "")
    observed = non_whitespace_chars(str(candidate["text"]))
    lower, upper, midpoint = length_band(observed)
    rules = (
        f"\n\n长度是硬性约束：最终正文须为{lower}—{upper}个非空白字符，"
        f"建议初稿瞄准{midpoint}个左右。计数时删除空格、制表符和换行，"
        "汉字、字母、数字与标点均各计1个字符，标题如有也计入。"
        "输出前请自行核对；若不足或超出，先在内部删改到范围内，再一次性输出最终稿。"
        "只输出故事正文，不要报告字数，不要输出提纲、注释、创作说明或核对过程。"
    )
    return base + rules, lower, upper


def blind_token(prefix: str, seed: int, value: str) -> str:
    digest = hashlib.sha256(f"{seed}:{prefix}:{value}".encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def prepare(candidates_path: Path, work_dir: Path, seed: int) -> dict[str, int]:
    selected = [
        row
        for row in read_jsonl(candidates_path)
        if re.fullmatch(r"HP_COIG_\d{4}", str(row.get("human_id", "")))
    ]
    selected.sort(key=lambda row: row["human_id"])
    selected = selected[:20]
    if len(selected) != 20:
        raise RuntimeError(f"expected 20 COIG candidates, found {len(selected)}")

    prompts: list[dict[str, Any]] = []
    humans: list[dict[str, Any]] = []
    for row in selected:
        prompt, lower, upper = strict_prompt(row)
        human_count = non_whitespace_chars(row["text"])
        if not length_valid(human_count, lower, upper):
            raise AssertionError(f"human reference outside band: {row['human_id']}")
        prompt_id = blind_token("Q", seed, row["human_id"])
        prompts.append(
            {
                "human_id": row["human_id"],
                "prompt_id": prompt_id,
                "genre": row["genre"],
                "prompt": prompt,
                "min_chars": lower,
                "max_chars": upper,
                "counting_rule": "unicode_non_whitespace_codepoints",
            }
        )
        humans.append(
            {
                "item_id": f"HUMAN_{row['human_id']}",
                "human_id": row["human_id"],
                "prompt_id": prompt_id,
                "source_type": "human",
                "source_family": "human",
                "generator_model": "human_reference",
                "genre": row["genre"],
                "prompt": prompt,
                "text": row["text"],
                "char_count": human_count,
                "min_chars": lower,
                "max_chars": upper,
            }
        )
    write_jsonl(work_dir / "prompts.jsonl", prompts)
    write_jsonl(work_dir / "human_items.jsonl", humans)
    return {"prompts": len(prompts), "human_items": len(humans)}


def grade_for_score(score: int) -> str:
    for grade, minimum in GRADE_BANDS:
        if score >= minimum:
            return grade
    raise AssertionError("grade scale must cover zero")


def json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def parse_rating(text: str) -> dict[str, Any]:
    obj = json_object(text)
    score = int(obj["overall_score"])
    confidence = int(obj["confidence"])
    if not 0 <= score <= 100:
        raise ValueError("overall_score outside 0..100")
    if not 1 <= confidence <= 5:
        raise ValueError("confidence outside 1..5")
    expected = {
        "narrative_completeness",
        "coherence",
        "character_and_viewpoint",
        "language_and_rhythm",
        "emotional_and_imagery_effect",
        "originality_and_task_compliance",
    }
    dimensions = obj.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != expected:
        raise ValueError("dimensions have wrong keys")
    parsed_dimensions = {key: int(value) for key, value in dimensions.items()}
    if any(not 1 <= value <= 5 for value in parsed_dimensions.values()):
        raise ValueError("dimension outside 1..5")
    reason = str(obj.get("brief_reason", "")).strip()
    if not reason:
        raise ValueError("brief_reason is empty")
    return {
        "overall_score": score,
        "grade": grade_for_score(score),
        "confidence": confidence,
        "dimensions": parsed_dimensions,
        "brief_reason": reason,
    }


def _budget_like(status: int, detail: str) -> bool:
    lowered = detail.lower()
    terms = (
        "insufficient balance",
        "insufficient_quota",
        "billing",
        "payment required",
        "余额不足",
        "欠费",
    )
    return status == 402 or any(term in lowered for term in terms)


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:4000]
        if exc.code == 429:
            delay_match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', detail)
            if delay_match is None:
                delay_match = re.search(r"retry in ([0-9.]+)s", detail, flags=re.IGNORECASE)
            retry_after = float(delay_match.group(1)) if delay_match else 20.0
            raise RateLimitError(f"HTTP 429: {detail}", retry_after) from exc
        if _budget_like(exc.code, detail):
            raise BudgetError(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def call_openai_compatible(
    provider: Provider,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
) -> dict[str, Any]:
    key = os.environ.get(provider.key_env, "").strip()
    if not key:
        raise RuntimeError(f"{provider.key_env} is not set")
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "max_tokens": max_tokens,
    }
    if provider.name == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = post_json(
        provider.endpoint,
        payload,
        {"Authorization": f"Bearer {key}"},
        timeout,
    )
    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    if not str(content).strip():
        raise RuntimeError(f"{provider.name} returned no content")
    return {
        "text": str(content).strip(),
        "request_id": response.get("id"),
        "returned_model": response.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
    }


def call_gemini(
    provider: Provider,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
) -> dict[str, Any]:
    key = os.environ.get(provider.key_env, "").strip()
    if not key:
        raise RuntimeError(f"{provider.key_env} is not set")
    generation_config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
        "thinkingConfig": {"thinkingLevel": "minimal"},
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": generation_config,
    }
    response = post_json(
        f"{provider.endpoint}/{provider.model}:generateContent",
        payload,
        {"x-goog-api-key": key},
        timeout,
    )
    candidate = response.get("candidates", [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts).strip()
    if not content:
        raise RuntimeError(f"gemini returned no content: {json.dumps(response, ensure_ascii=False)[:1000]}")
    return {
        "text": content,
        "request_id": response.get("responseId"),
        "returned_model": response.get("modelVersion"),
        "finish_reason": candidate.get("finishReason"),
        "usage": response.get("usageMetadata", {}),
    }


def call_provider(
    provider: Provider,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
) -> dict[str, Any]:
    if provider.name == "gemini":
        return call_gemini(provider, system, user, max_tokens, json_mode, timeout)
    return call_openai_compatible(provider, system, user, max_tokens, json_mode, timeout)


def retry_call(
    provider: Provider,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
    attempts: int = 3,
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_provider(provider, system, user, max_tokens, json_mode, timeout)
        except BudgetError:
            raise
        except RateLimitError as exc:
            last = exc
            if attempt == attempts:
                break
            wait = max(exc.retry_after + 1.0, 2.0 * attempt)
            print(
                f"  {provider.name} rate limited; retry in {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
        except Exception as exc:  # network, rate limit, provider transient errors
            last = exc
            if attempt == attempts:
                break
            wait = 2.0 * attempt
            print(f"  {provider.name} attempt {attempt} failed: {str(exc)[:220]}; retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{provider.name} failed after {attempts} attempts: {last}")


def correction_prompt(prompt: str, draft: str, count: int, lower: int, upper: int) -> str:
    target = (lower + upper) // 2
    ratio = max(20, min(300, round(100 * target / max(count, 1))))
    if count > upper:
        excess = count - upper
        if excess <= max(50, round(upper * 0.05)):
            edit_instruction = (
                f"原稿只超出{excess}个字符。不要重写，不要增加任何内容；"
                f"只从原稿删除大约{excess + 15}—{excess + 35}个非空白字符，"
                "其余措辞、段落和结局尽量原样保留。"
            )
        else:
            edit_instruction = (
                f"原稿超出{excess}个字符。请保留核心人物、因果、转折和结局，"
                f"把新稿压缩到当前篇幅的大约{ratio}%（目标约{target}个非空白字符）。"
            )
    else:
        shortage = lower - count
        if shortage <= max(50, round(lower * 0.05)):
            edit_instruction = (
                f"原稿只缺{shortage}个字符。不要重写或删减原稿；"
                f"只补充大约{shortage + 15}—{shortage + 35}个非空白字符的必要动作或收束。"
            )
        else:
            edit_instruction = (
                f"原稿还缺至少{shortage}个字符。请在原稿基础上补充必要的动作、因果或结尾，"
                f"把新稿扩充到当前篇幅的大约{ratio}%（目标约{target}个非空白字符）；"
                "不要删改成更短的版本。"
            )
    return f"""以下故事根据给定任务生成，但按“删除所有空白后逐个Unicode字符计数”的规则，实测为{count}个字符，未满足{lower}—{upper}个字符的硬约束。

原始任务：
{prompt}

待纠偏初稿：
{draft}

{edit_instruction}
只输出纠偏后的故事正文，不报告字数、过程或说明。"""


def generate_provider(
    provider: Provider,
    work_dir: Path,
    timeout: int,
    max_prompts: int,
    retry_invalid: bool,
    max_corrections: int,
) -> tuple[int, int]:
    prompts = read_jsonl(work_dir / "prompts.jsonl")
    if max_prompts:
        prompts = prompts[:max_prompts]
    output = work_dir / f"generated_{provider.name}.jsonl"
    existing_rows = read_jsonl(output)
    latest = {row["human_id"]: row for row in existing_rows}
    terminal_status = {human_id: row.get("status") for human_id, row in latest.items()}
    terminal = {
        human_id
        for human_id, status in terminal_status.items()
        if status == "ok" or (status == "invalid" and not retry_invalid)
    }
    ok = invalid = 0
    for index, row in enumerate(prompts, start=1):
        if row["human_id"] in terminal:
            continue
        base = {
            "item_id": f"{provider.name.upper()}_{row['human_id']}",
            "human_id": row["human_id"],
            "prompt_id": row["prompt_id"],
            "source_type": "machine",
            "source_family": provider.name,
            "generator_model": provider.model,
            "reasoning_mode": provider.reasoning_mode,
            "search_enabled": False,
            "genre": row["genre"],
            "prompt": row["prompt"],
            "min_chars": row["min_chars"],
            "max_chars": row["max_chars"],
            "generated_at": utc_now(),
        }
        try:
            previous = latest.get(row["human_id"])
            if previous and previous.get("status") == "invalid" and retry_invalid:
                attempts = list(previous.get("attempts", []))
                chosen = {
                    "text": previous["text"],
                    "request_id": None,
                    "returned_model": previous.get("generator_model"),
                    "finish_reason": "resume_invalid",
                    "usage": {},
                }
                chosen_count = int(previous["char_count"])
                resumed = True
            else:
                first = retry_call(provider, GENERATION_SYSTEM, row["prompt"], 4096, False, timeout)
                first_count = non_whitespace_chars(first["text"])
                attempts = [{**first, "char_count": first_count, "attempt": 1}]
                chosen = first
                chosen_count = first_count
                resumed = False
            while (
                not length_valid(chosen_count, row["min_chars"], row["max_chars"])
                and len(attempts) < 1 + max_corrections
            ):
                corrected = retry_call(
                    provider,
                    GENERATION_SYSTEM,
                    correction_prompt(
                        row["prompt"],
                        chosen["text"],
                        chosen_count,
                        row["min_chars"],
                        row["max_chars"],
                    ),
                    4096,
                    False,
                    timeout,
                )
                chosen = corrected
                chosen_count = non_whitespace_chars(corrected["text"])
                attempts.append(
                    {**corrected, "char_count": chosen_count, "attempt": len(attempts) + 1}
                )
            valid = length_valid(chosen_count, row["min_chars"], row["max_chars"])
            append_jsonl(
                output,
                {
                    **base,
                    "status": "ok" if valid else "invalid",
                    "text": chosen["text"],
                    "char_count": chosen_count,
                    "within_requested_range": row["min_chars"] <= chosen_count <= row["max_chars"],
                    "qc_upper_tolerance": (row["max_chars"] * 103) // 100,
                    "attempt_count": len(attempts),
                    "resumed_invalid_record": resumed,
                    "attempts": attempts,
                },
            )
            if valid:
                ok += 1
            else:
                invalid += 1
            print(
                f"  generate {provider.name} {index}/{len(prompts)} {row['human_id']} "
                f"chars={chosen_count} attempts={len(attempts)} status={'ok' if valid else 'invalid'}",
                flush=True,
            )
        except BudgetError as exc:
            print(f"BUDGET_STOP {provider.name}: {str(exc)[:1000]}", file=sys.stderr, flush=True)
            raise
    return ok, invalid


def build_blind_items(work_dir: Path, seed: int, allow_partial: bool = False) -> int:
    prompt_by_human_id = {row["human_id"]: row for row in read_jsonl(work_dir / "prompts.jsonl")}
    items = read_jsonl(work_dir / "human_items.jsonl")
    for provider_name in PROVIDER_ORDER:
        rows = read_jsonl(work_dir / f"generated_{provider_name}.jsonl")
        valid = [row for row in rows if row.get("status") == "ok"]
        items.extend(valid)
    if len(items) != 80 and not allow_partial:
        counts = {name: len([row for row in items if row["source_family"] == name]) for name in ("human", *PROVIDER_ORDER)}
        raise RuntimeError(f"need 80 valid items before rating; found {len(items)}: {counts}")
    private_key: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for row in items:
        canonical_prompt = prompt_by_human_id[row["human_id"]]
        blind_id = blind_token("T", seed, row["item_id"])
        private_key.append(
            {
                "blind_id": blind_id,
                "item_id": row["item_id"],
                "human_id": row["human_id"],
                "source_type": row["source_type"],
                "source_family": row["source_family"],
                "generator_model": row["generator_model"],
            }
        )
        blind.append(
            {
                "blind_id": blind_id,
                "prompt_id": canonical_prompt["prompt_id"],
                "genre": canonical_prompt["genre"],
                "prompt": canonical_prompt["prompt"],
                "text": row["text"],
                "char_count": row["char_count"],
            }
        )
    write_jsonl(work_dir / "blind_key.jsonl", private_key)
    write_jsonl(work_dir / "blind_items.jsonl", blind)
    return len(blind)


def rate_provider(
    provider: Provider,
    work_dir: Path,
    timeout: int,
    max_items: int,
    seed: int,
) -> tuple[int, int]:
    items = read_jsonl(work_dir / "blind_items.jsonl")
    random.Random(seed + sum(map(ord, provider.name))).shuffle(items)
    if max_items:
        items = items[:max_items]
    output = work_dir / f"ratings_{provider.name}.jsonl"
    terminal = {row["blind_id"] for row in read_jsonl(output) if row.get("status") in {"ok", "invalid"}}
    ok = invalid = 0
    for index, item in enumerate(items, start=1):
        if item["blind_id"] in terminal:
            continue
        request_text = RATING_TEMPLATE.format(prompt=item["prompt"], text=item["text"])
        attempts: list[dict[str, Any]] = []
        parsed: dict[str, Any] | None = None
        try:
            for parse_attempt in range(1, 3):
                response = retry_call(provider, RATING_SYSTEM, request_text, 700, True, timeout)
                try:
                    parsed = parse_rating(response["text"])
                    attempts.append({**response, "parse_attempt": parse_attempt, "parse_status": "ok"})
                    break
                except Exception as exc:
                    attempts.append(
                        {
                            **response,
                            "parse_attempt": parse_attempt,
                            "parse_status": "invalid",
                            "parse_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    request_text += "\n\n上一次输出格式不合格。请重新输出且只输出满足上述字段、类型和取值范围的JSON对象。"
            status = "ok" if parsed is not None else "invalid"
            append_jsonl(
                output,
                {
                    "blind_id": item["blind_id"],
                    "prompt_id": item["prompt_id"],
                    "judge_family": provider.name,
                    "judge_model": provider.model,
                    "reasoning_mode": provider.reasoning_mode,
                    "search_enabled": False,
                    "status": status,
                    "rating": parsed,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "judged_at": utc_now(),
                },
            )
            if parsed is None:
                invalid += 1
            else:
                ok += 1
            score_text = "-" if parsed is None else str(parsed["overall_score"])
            print(
                f"  rate {provider.name} {index}/{len(items)} {item['blind_id']} "
                f"score={score_text} attempts={len(attempts)} status={status}",
                flush=True,
            )
            if provider.name == "gemini" and index < len(items):
                time.sleep(4.2)
        except BudgetError as exc:
            print(f"BUDGET_STOP {provider.name}: {str(exc)[:1000]}", file=sys.stderr, flush=True)
            raise
    return ok, invalid


def usage_tokens(provider_name: str, usage: dict[str, Any]) -> tuple[int, int, int]:
    if provider_name == "gemini":
        input_tokens = int(usage.get("promptTokenCount", 0) or 0)
        output_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
        reasoning_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)
        return input_tokens, output_tokens, reasoning_tokens
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
    return input_tokens, output_tokens, reasoning_tokens


def summarize(work_dir: Path) -> dict[str, Any]:
    key = {row["blind_id"]: row for row in read_jsonl(work_dir / "blind_key.jsonl")}
    summary: dict[str, Any] = {
        "protocol": "api-family-pilot20-independent-rating-v1",
        "generated_at": utc_now(),
        "providers": {},
    }
    for name in PROVIDER_ORDER:
        provider = PROVIDERS[name]
        generated = read_jsonl(work_dir / f"generated_{name}.jsonl")
        ratings = read_jsonl(work_dir / f"ratings_{name}.jsonl")
        input_tokens = output_tokens = reasoning_tokens = 0
        for row in generated:
            for attempt in row.get("attempts", []):
                one_in, one_out, one_reason = usage_tokens(name, attempt.get("usage", {}))
                input_tokens += one_in
                output_tokens += one_out
                reasoning_tokens += one_reason
        for row in ratings:
            for attempt in row.get("attempts", []):
                one_in, one_out, one_reason = usage_tokens(name, attempt.get("usage", {}))
                input_tokens += one_in
                output_tokens += one_out
                reasoning_tokens += one_reason
        paid_output = output_tokens + (reasoning_tokens if name == "gemini" else 0)
        estimated_usd = (
            input_tokens * provider.input_usd_per_m + paid_output * provider.output_usd_per_m
        ) / 1_000_000
        by_source: dict[str, list[int]] = {}
        for row in ratings:
            if row.get("status") != "ok" or row["blind_id"] not in key:
                continue
            source = key[row["blind_id"]]["source_family"]
            by_source.setdefault(source, []).append(int(row["rating"]["overall_score"]))
        summary["providers"][name] = {
            "model": provider.model,
            "generation_ok": sum(row.get("status") == "ok" for row in generated),
            "generation_invalid": sum(row.get("status") == "invalid" for row in generated),
            "rating_ok": sum(row.get("status") == "ok" for row in ratings),
            "rating_invalid": sum(row.get("status") == "invalid" for row in ratings),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens_reported": reasoning_tokens,
            "estimated_standard_usd": round(estimated_usd, 6),
            "source_score_means": {
                source: round(sum(values) / len(values), 3) for source, values in sorted(by_source.items())
            },
        }
    write_jsonl(work_dir / "run_summary.jsonl", [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "build-blind", "rate", "summary", "all"),
        default="all",
    )
    parser.add_argument("--providers", nargs="+", choices=PROVIDER_ORDER, default=list(PROVIDER_ORDER))
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=3,
        help="Maximum deterministic length-only rewrites after the initial draft.",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="Retry terminal invalid generation records after a calibrated smoke test.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Build a temporary partial blind set for API smoke testing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    if args.stage in {"prepare", "all"}:
        print(json.dumps(prepare(args.candidates.resolve(), work_dir, args.seed), ensure_ascii=False), flush=True)
    if args.stage in {"generate", "all"}:
        for name in args.providers:
            result = generate_provider(
                PROVIDERS[name],
                work_dir,
                args.timeout,
                args.max_prompts,
                args.retry_invalid,
                args.max_corrections,
            )
            print(f"generate {name}: ok={result[0]} invalid={result[1]}", flush=True)
    if args.stage in {"build-blind", "all"}:
        print(
            f"blind_items={build_blind_items(work_dir, args.seed, args.allow_partial)}",
            flush=True,
        )
    if args.stage in {"rate", "all"}:
        if not (work_dir / "blind_items.jsonl").exists():
            raise RuntimeError("blind_items.jsonl missing; run --stage build-blind first")
        for name in args.providers:
            result = rate_provider(PROVIDERS[name], work_dir, args.timeout, args.max_items, args.seed)
            print(f"rate {name}: ok={result[0]} invalid={result[1]}", flush=True)
    if args.stage in {"summary", "all"}:
        print(json.dumps(summarize(work_dir), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
