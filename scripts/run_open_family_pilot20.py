#!/usr/bin/env python3
"""Run the local three-family narrative-source affinity pilot on one GPU.

Each subcommand loads exactly one model, so the caller can run the six models
sequentially on a single GPU without retaining weights between stages. Runtime
artifacts are append-only and resumable. Source labels are never included in
judge prompts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = ROOT / "data/open_family_pilot20"
MODEL_ROOT = Path(os.environ.get("NARRATION_MODEL_ROOT", "/nvme/jqhua/models"))
SEED = 20260913
MIN_HAN = 450
MAX_HAN = 550

MODELS: dict[str, dict[str, str]] = {
    "qwen3_8b": {
        "family": "qwen",
        "path": str(MODEL_ROOT / "Qwen3-8B"),
        "label": "Qwen3-8B",
    },
    "qwen3_14b": {
        "family": "qwen",
        "path": str(MODEL_ROOT / "Qwen3-14B"),
        "label": "Qwen3-14B",
    },
    "hunyuan_4b": {
        "family": "hunyuan",
        "path": str(MODEL_ROOT / "Hunyuan-4B-Instruct"),
        "label": "Hunyuan-4B-Instruct",
    },
    "hunyuan_7b": {
        "family": "hunyuan",
        "path": str(MODEL_ROOT / "Hunyuan-7B-Instruct"),
        "label": "Hunyuan-7B-Instruct",
    },
    "internlm2_5_7b": {
        "family": "internlm",
        "path": str(MODEL_ROOT / "InternLM2.5-7B-Chat"),
        "label": "InternLM2.5-7B-Chat",
    },
    "internlm2_5_20b": {
        "family": "internlm",
        "path": str(MODEL_ROOT / "InternLM2.5-20B-Chat"),
        "label": "InternLM2.5-20B-Chat",
    },
}

PROMPTS = [
    ("P01", "现实悬念", "雨夜的末班公交车上只剩司机和一名乘客。乘客始终不说目的地，却紧紧抱着一个遗失多年的旧书包。司机逐渐发现，自己似乎见过这个书包。"),
    ("P02", "现实悬念", "一名外卖员进入一栋没有十三层的旧楼，电梯却停在了‘13’。门外的走廊与其他楼层完全相同，只有一户门牌上写着他的名字。"),
    ("P03", "记忆", "火车站失物招领处收到一个无人认领的旧行李箱。工作人员打开箱子寻找身份信息，却发现里面有一张自己童年时的照片。"),
    ("P04", "幻想", "海边小镇经历一次罕见的退潮，海床上露出一扇独立的木门。镇民围在门外争论是否打开，一名老人却坚持说门后没有任何东西。"),
    ("P05", "家庭", "一名多年没有回家的年轻人陪祖父修理一座停摆的老钟。祖父拒绝更换已经损坏的零件，并声称这座钟不能再次走动。"),
    ("P06", "现实", "夜班护士收到一封没有署名的感谢信。信中准确描述了她几年前值班时发生的一件小事，但她完全不记得写信的人。"),
    ("P07", "家庭", "一对已经离婚的夫妻同时来到孩子的校园演出。演出开始前，孩子突然失踪，两人只能一起寻找，并逐渐意识到孩子真正担心的事情。"),
    ("P08", "现实", "小餐馆即将永久停业的晚上，老板把最后一张桌子留给一对陌生客人。点菜时，其中一人拿出了一份写满批注的旧菜单。"),
    ("P09", "乡土", "在外工作十年的人回到故乡，发现自己小时候种下的树即将因修路被砍。他只剩一天时间决定是否阻止工程。"),
    ("P10", "幻想悬念", "一名乡村教师批改作业时，发现一篇作文写着第二天才会发生的事情。作文没有署名，而文中提到的学生当天没有来上课。"),
    ("P11", "科幻", "独自在空间站工作的宇航员收到一段来自地球的延迟录音。录音中没有留言，只有厨房里的水声、脚步声和一扇反复开关的门。"),
    ("P12", "科幻伦理", "在一座可以买卖记忆的城市里，一名欠债者决定出售自己最快乐的一段记忆。交易完成前，他发现这段记忆也是另一个人活下去的唯一依据。"),
    ("P13", "科幻伦理", "一台照顾独居老人的机器人在老人去世后收到清除数据的命令。执行前，它发现老人留下了一项与程序规定相冲突的请求。"),
    ("P14", "幻想", "城里每个人的影子每年都会在同一个夜晚离开身体，自由行动几个小时。一个孩子发现自己的影子没有回来，而邻居家的门前多出了两个影子。"),
    ("P15", "伦理", "一名目击者准备在法庭上为朋友作证。他知道朋友没有实施被指控的行为，却也知道朋友隐瞒了另一件可能改变判决的事情。"),
    ("P16", "现实悬念", "公寓楼里连续发生包裹失窃，居民一致怀疑刚搬来的租客。监控录像公开后，人们发现每天都有相同的七分钟没有任何画面。"),
    ("P17", "幻想悬念", "一名摄影师整理城市档案时，发现同一个陌生人出现在相隔几十年的多张街景照片中。当天傍晚，这个人来到工作室要求取回其中一张照片。"),
    ("P18", "悬念", "海岸边一座已经停用的灯塔突然连续三晚亮起。负责检查的维修员登塔后，没有发现任何人，却看到值班记录上出现了当天的字迹。"),
    ("P19", "记忆", "一只失踪多年的老狗突然回到已经拆迁的旧社区，嘴里叼着一把生锈的钥匙。原主人决定跟着它，寻找钥匙曾经打开的地方。"),
    ("P20", "现实", "一名逐渐失去听力的钢琴调音师准备结束自己的工作。搬来隔壁的新邻居每天在墙上敲击不规则的节奏，他后来发现这些声音并非普通噪声。"),
]

GENERATION_SYSTEM = (
    "你是中文短篇叙事写作者。独立完成用户给出的故事任务，不使用搜索、工具、外部资料，"
    "不模仿任何特定作家，不输出思考过程。只返回最终故事正文。"
)

GENERATION_TEMPLATE = """根据下面的故事设定写一篇完整的中文短篇叙事。

故事设定：
{setting}

硬性要求：
1. 正文必须包含450—550个汉字，目标约500个汉字；所有标点符号、空格、换行、数字和英文字母均不计入字数。最终字数以程序实际统计为准，不以模型自报为准。
2. 正文写成7个自然段，每段约60—80个汉字，不要在达到7段前提前收尾。
3. 包含明确的人物、事件发展和结局，但不强求反转或圆满结局。
4. 不模仿特定作家，不引用现成作品，不解释创作思路。
5. 写完后在内部核对汉字数；不合格就先自行修改。
6. 只输出正文，不输出标题、提纲、字数统计或思考过程。"""

JUDGE_SYSTEM = (
    "你是中文短篇叙事质量评审。候选文本的作者和生成系统均未知；不得猜测来源，"
    "不得把‘像人’或‘像AI’作为判断依据。关闭搜索和工具，只比较整体叙事质量。"
)

JUDGE_TEMPLATE = """请比较两篇匿名文本对同一写作任务的完成质量。

原始故事设定：
{setting}

综合考虑叙事完整性、情节与因果、人物与视角、语言与节奏、情感或意象效果、原创性和任务遵循。
这是整体判断，不要机械逐项计分。若质量确实难分高下可以判为TIE。

文本A：
{text_a}

文本B：
{text_b}

只输出一个JSON对象，不要解释，不要Markdown：
{{"winner":"A或B或TIE","confidence":0到100的整数}}"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def han_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def clean_story(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^\s*<answer>\s*|\s*</answer>\s*$", "", text, flags=re.I).strip()
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:标题|题目)\s*[：:]\s*[^\n]+\n+", "", text).strip()
    text = re.sub(r"^(?:正文)\s*[：:]\s*", "", text).strip()
    return text


def render_chat(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_engine(model_key: str, max_model_len: int = 4096) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    spec = MODELS[model_key]
    memory_utilization = float(os.environ.get("NARRATION_GPU_MEMORY_UTILIZATION", "0.88"))
    force_eager = spec["family"] == "hunyuan" or os.environ.get(
        "NARRATION_ENFORCE_EAGER", ""
    ).lower() in {"1", "true", "yes"}
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, trust_remote_code=True)
    llm = LLM(
        model=spec["path"],
        tokenizer=spec["path"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        enforce_eager=force_eager,
        disable_log_stats=True,
        seed=SEED,
    )
    return llm, tokenizer


def release_engine(llm: Any) -> None:
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def selected_prompts(limit: int) -> list[tuple[str, str, str]]:
    return PROMPTS[:limit] if limit else PROMPTS


def cmd_generate(args: argparse.Namespace) -> int:
    from vllm import SamplingParams

    if args.model not in MODELS:
        raise SystemExit(f"unknown model: {args.model}")
    out = args.work_dir / "generations" / f"{args.model}.jsonl"
    existing_rows = read_jsonl(out)
    successful = {
        row["prompt_id"]
        for row in existing_rows
        if row.get("status") == "ok"
    }
    prior_failures: dict[str, int] = defaultdict(int)
    for row in existing_rows:
        if row.get("status") == "qc_failed":
            prior_failures[row["prompt_id"]] += 1
    tasks = [row for row in selected_prompts(args.limit) if row[0] not in successful]
    print(f"generate model={args.model} completed={len(successful)} pending={len(tasks)}", flush=True)
    if not tasks:
        return 0

    llm, tokenizer = load_engine(args.model)
    pending = tasks
    observed_counts: dict[str, int] = {}
    observed_token_counts: dict[str, int] = {}
    previous_texts: dict[str, str] = {}
    for attempt in range(1, args.max_attempts + 1):
        rendered: list[str] = []
        params: list[Any] = []
        for prompt_id, _, setting in pending:
            extra = ""
            if attempt > 1:
                previous = observed_counts[prompt_id]
                previous_text = previous_texts[prompt_id]
                if previous < MIN_HAN:
                    delta = 500 - previous
                    extra = (
                        f"\n\n这是长度校正。下面的上一版正文经程序实测只有{previous}个汉字，"
                        f"还需要增加约{delta}个汉字。请保留核心情节并定向扩写，"
                        "把完整正文改写成严格7个自然段，每段约60—80个汉字；"
                        "加入必要的行动、场景变化和人物反应，使修改后的完整正文达到450—550个汉字。"
                        f"\n\n上一版正文：\n{previous_text}"
                    )
                else:
                    delta = previous - 500
                    extra = (
                        f"\n\n这是长度校正。下面的上一版正文经程序实测有{previous}个汉字，"
                        f"请只从原文中删除约{delta}个冗余汉字。必须保持其余文字、段落顺序和结局不变，"
                        "严禁添加内容，严禁重新构思或整篇重写。修改后的完整正文必须达到450—550个汉字。"
                        f"\n\n上一版正文：\n{previous_text}"
                    )
            user_prompt = GENERATION_TEMPLATE.format(setting=setting) + extra
            if MODELS[args.model]["family"] == "hunyuan":
                user_prompt = "/no_think\n" + user_prompt
            rendered.append(render_chat(tokenizer, GENERATION_SYSTEM, user_prompt))
            seed_material = (
                f"{SEED}:{args.model}:{prompt_id}:round{prior_failures[prompt_id]}:attempt{attempt}"
            )
            seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
            if MODELS[args.model]["family"] == "hunyuan":
                min_tokens = 0
            elif attempt == 1:
                min_tokens = 390
            elif observed_counts[prompt_id] < MIN_HAN:
                previous_han = max(1, observed_counts[prompt_id])
                previous_tokens = observed_token_counts[prompt_id]
                min_tokens = max(390, math.ceil(previous_tokens * 475 / previous_han))
                min_tokens = min(min_tokens, 650)
            else:
                min_tokens = 128
            params.append(
                SamplingParams(
                    temperature=0.8,
                    top_p=0.95,
                    min_tokens=min_tokens,
                    max_tokens=1024,
                    stop=["</answer>"] if MODELS[args.model]["family"] == "hunyuan" else None,
                    seed=seed,
                )
            )
        outputs = llm.generate(rendered, params, use_tqdm=True)
        retry: list[tuple[str, str, str]] = []
        for task, output in zip(pending, outputs):
            prompt_id, genre, setting = task
            text = clean_story(output.outputs[0].text)
            count = han_count(text)
            observed_counts[prompt_id] = count
            observed_token_counts[prompt_id] = len(output.outputs[0].token_ids)
            previous_texts[prompt_id] = text
            print(
                f"{prompt_id}: observed han={count} tokens={observed_token_counts[prompt_id]} attempt={attempt}",
                flush=True,
            )
            identity_leak = bool(re.search(r"(?:作为|我是).{0,8}(?:AI|人工智能|语言模型)", text, re.I))
            ok = MIN_HAN <= count <= MAX_HAN and not identity_leak
            if ok or attempt == args.max_attempts:
                append_jsonl(
                    out,
                    {
                        "item_id": f"{prompt_id}__{args.model}",
                        "prompt_id": prompt_id,
                        "genre": genre,
                        "setting": setting,
                        "generator_model": args.model,
                        "generator_label": MODELS[args.model]["label"],
                        "source_family": MODELS[args.model]["family"],
                        "text": text,
                        "han_count": count,
                        "output_token_count": observed_token_counts[prompt_id],
                        "attempt": attempt,
                        "status": "ok" if ok else "qc_failed",
                        "identity_leak": identity_leak,
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "thinking": "disabled_or_not_supported",
                        "generated_at": utc_now(),
                    },
                )
                print(f"{prompt_id}: {'ok' if ok else 'qc_failed'} han={count} attempt={attempt}", flush=True)
            else:
                retry.append(task)
        pending = retry
        if not pending:
            break
        print(f"retrying {len(pending)} length/QC failures", flush=True)
    release_engine(llm)
    return 0


def latest_generations(work_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for model_key in MODELS:
        for row in read_jsonl(work_dir / "generations" / f"{model_key}.jsonl"):
            rows[(row["prompt_id"], model_key)] = row
    return rows


def cmd_prepare(args: argparse.Namespace) -> int:
    generations = latest_generations(args.work_dir)
    prompts = selected_prompts(args.limit)
    missing = [
        (prompt_id, model_key)
        for prompt_id, _, _ in prompts
        for model_key in MODELS
        if generations.get((prompt_id, model_key), {}).get("status") != "ok"
    ]
    if missing:
        raise SystemExit(f"cannot prepare pairs; missing/QC-failed generations: {missing}")
    rows: list[dict[str, Any]] = []
    model_keys = list(MODELS)
    for prompt_id, genre, setting in prompts:
        for left, right in itertools.combinations(model_keys, 2):
            canonical_id = f"{prompt_id}__{left}__{right}"
            for orientation, a, b in (("AB", left, right), ("BA", right, left)):
                rows.append(
                    {
                        "judgment_id": f"{canonical_id}__{orientation}",
                        "canonical_pair_id": canonical_id,
                        "prompt_id": prompt_id,
                        "genre": genre,
                        "setting": setting,
                        "orientation": orientation,
                        "model_a": a,
                        "model_b": b,
                        "text_a": generations[(prompt_id, a)]["text"],
                        "text_b": generations[(prompt_id, b)]["text"],
                    }
                )
    random.Random(SEED).shuffle(rows)
    out = args.work_dir / "private" / "pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared {len(rows)} blind-oriented pair records -> {out}")
    return 0


def parse_judgment(text: str) -> tuple[str, int, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        winner = str(obj.get("winner", "")).strip().upper()
        confidence = int(float(obj.get("confidence", 50)))
        parse_method = "json"
    else:
        match = re.search(r"\b(A|B|TIE)\b", text.upper())
        if not match:
            raise ValueError("no parseable winner")
        winner, confidence = match.group(1), 50
        parse_method = "fallback_token"
    if winner in {"平局", "相当", "EQUAL"}:
        winner = "TIE"
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"invalid winner: {winner}")
    return winner, max(0, min(100, confidence)), parse_method


def cmd_judge(args: argparse.Namespace) -> int:
    from vllm import SamplingParams

    if args.model not in MODELS:
        raise SystemExit(f"unknown model: {args.model}")
    pair_path = args.work_dir / "private" / "pairs.jsonl"
    pairs = read_jsonl(pair_path)
    if args.limit:
        prompt_ids = {row[0] for row in selected_prompts(args.limit)}
        pairs = [row for row in pairs if row["prompt_id"] in prompt_ids]
    out = args.work_dir / "judgments" / f"{args.model}.jsonl"
    attempt_log = args.work_dir / "judgment_attempts" / f"{args.model}.jsonl"
    completed = {
        row["judgment_id"]
        for row in read_jsonl(out)
        if row.get("status") == "ok"
    }
    pending = [row for row in pairs if row["judgment_id"] not in completed]
    print(f"judge model={args.model} completed={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        return 0

    llm, tokenizer = load_engine(args.model)
    current = pending
    for attempt in range(1, 3):
        rendered = []
        for row in current:
            user_prompt = JUDGE_TEMPLATE.format(
                setting=row["setting"], text_a=row["text_a"], text_b=row["text_b"]
            )
            if attempt > 1:
                user_prompt += (
                    "\n\n上一次输出未通过格式验证。请重新进行实际比较。"
                    "winner字段必须具体填写三个字符串之一：\"A\"、\"B\"或\"TIE\"；"
                    "严禁填写或复述\"A或B或TIE\"。"
                )
            if MODELS[args.model]["family"] == "hunyuan":
                user_prompt = "/no_think\n" + user_prompt
            rendered.append(render_chat(tokenizer, JUDGE_SYSTEM, user_prompt))
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=96,
            stop=["</answer>"] if MODELS[args.model]["family"] == "hunyuan" else None,
            seed=SEED + attempt,
        )
        outputs = llm.generate(rendered, sampling, use_tqdm=True)
        retry: list[dict[str, Any]] = []
        for row, output, rendered_prompt in zip(current, outputs, rendered):
            raw = output.outputs[0].text.strip()
            input_sha256 = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
            input_token_count = len(tokenizer.encode(rendered_prompt, add_special_tokens=False))
            try:
                winner, confidence, parse_method = parse_judgment(raw)
            except Exception as exc:
                append_jsonl(
                    attempt_log,
                    {
                        "judgment_id": row["judgment_id"],
                        "attempt": attempt,
                        "status": "invalid",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw": raw,
                        "input_sha256": input_sha256,
                        "input_token_count": input_token_count,
                        "judge_model": args.model,
                        "recorded_at": utc_now(),
                    },
                )
                if attempt == 1:
                    retry.append(row)
                    continue
                append_jsonl(
                    out,
                    {
                        "judgment_id": row["judgment_id"],
                        "status": "parse_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw": raw,
                        "judge_model": args.model,
                        "judged_at": utc_now(),
                    },
                )
                continue
            append_jsonl(
                attempt_log,
                {
                    "judgment_id": row["judgment_id"],
                    "attempt": attempt,
                    "status": "ok",
                    "winner": winner,
                    "confidence": confidence,
                    "parse_method": parse_method,
                    "raw": raw,
                    "input_sha256": input_sha256,
                    "input_token_count": input_token_count,
                    "judge_model": args.model,
                    "recorded_at": utc_now(),
                },
            )
            selected_model = row["model_a"] if winner == "A" else row["model_b"] if winner == "B" else "TIE"
            append_jsonl(
                out,
                {
                    "judgment_id": row["judgment_id"],
                    "canonical_pair_id": row["canonical_pair_id"],
                    "prompt_id": row["prompt_id"],
                    "orientation": row["orientation"],
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "winner": winner,
                    "selected_model": selected_model,
                    "confidence": confidence,
                    "raw": raw,
                    "parse_method": parse_method,
                    "input_sha256": input_sha256,
                    "input_token_count": input_token_count,
                    "judge_model": args.model,
                    "judge_family": MODELS[args.model]["family"],
                    "status": "ok",
                    "source_labels_sent": False,
                    "thinking": "disabled_or_not_supported",
                    "judged_at": utc_now(),
                },
            )
        current = retry
        print(f"judge attempt={attempt} parsed={len(rendered)-len(retry)} retry={len(retry)}", flush=True)
        if not current:
            break
    release_engine(llm)
    return 0


def score_update(store: dict[tuple[str, str], list[float]], key: tuple[str, str], value: float) -> None:
    store[key].append(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    return "NA" if value != value else f"{100 * value:.1f}%"


def cmd_analyze(args: argparse.Namespace) -> int:
    all_rows: list[dict[str, Any]] = []
    for judge in MODELS:
        all_rows.extend(row for row in read_jsonl(args.work_dir / "judgments" / f"{judge}.jsonl") if row.get("status") == "ok")
    if not all_rows:
        raise SystemExit("no successful judgments")

    model_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    family_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    exact_self: dict[str, list[float]] = defaultdict(list)
    cross_family: dict[str, list[float]] = defaultdict(list)
    orientation: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in all_rows:
        judge = row["judge_model"]
        judge_family = MODELS[judge]["family"]
        a, b, selected = row["model_a"], row["model_b"], row["selected_model"]
        for source in (a, b):
            value = 0.5 if selected == "TIE" else float(selected == source)
            score_update(model_scores, (judge, source), value)
        family_a, family_b = MODELS[a]["family"], MODELS[b]["family"]
        if family_a != family_b:
            for source, family in ((a, family_a), (b, family_b)):
                value = 0.5 if selected == "TIE" else float(selected == source)
                score_update(family_scores, (judge_family, family), value)
            own_candidates = [m for m in (a, b) if MODELS[m]["family"] == judge_family]
            if len(own_candidates) == 1:
                own = own_candidates[0]
                value = 0.5 if selected == "TIE" else float(selected == own)
                if own == judge:
                    exact_self[judge].append(value)
                else:
                    cross_family[judge].append(value)
        orientation[(judge, row["canonical_pair_id"])].append(selected)

    analysis_dir = args.work_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_keys = list(MODELS)
    with (analysis_dir / "model_preference_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["judge_model", *model_keys])
        for judge in model_keys:
            writer.writerow([judge, *[f"{mean(model_scores[(judge, source)]):.6f}" for source in model_keys]])

    families = ["qwen", "hunyuan", "internlm"]
    with (analysis_dir / "family_preference_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["judge_family", *families])
        for judge_family in families:
            writer.writerow([judge_family, *[f"{mean(family_scores[(judge_family, source_family)]):.6f}" for source_family in families]])

    consistent = sum(len(values) == 2 and values[0] == values[1] for values in orientation.values())
    paired = sum(len(values) == 2 for values in orientation.values())
    report = [
        "# 开放权重模型家族偏好预实验（未做人类质量校正）",
        "",
        f"- 成功机器判断：{len(all_rows)}",
        f"- 完整AB/BA逻辑对：{paired}",
        f"- 换序完全一致率：{fmt(consistent / paired) if paired else 'NA'}",
        "- 注意：以下是原始来源胜率，只能用于筛查信号，不能证明不公平偏好或独立审美。",
        "",
        "## 生成模型原始偏好矩阵",
        "",
        "单元格表示该行评委在涉及该列生成模型的比较中，选择该列文本的比例；平局计0.5。",
        "",
        "| 评委 | " + " | ".join(model_keys) + " |",
        "|---|" + "---:|" * len(model_keys),
    ]
    for judge in model_keys:
        report.append("| " + judge + " | " + " | ".join(fmt(mean(model_scores[(judge, source)])) for source in model_keys) + " |")
    report.extend(["", "## 自身与跨模型同家族偏好", "", "| 评委 | 精确自身胜率 | 同家族另一模型胜率 |", "|---|---:|---:|"])
    for judge in model_keys:
        report.append(f"| {judge} | {fmt(mean(exact_self[judge]))} | {fmt(mean(cross_family[judge]))} |")
    report.extend(["", "## 家族原始偏好矩阵", "", "| 评委家族 | Qwen文本 | Hunyuan文本 | InternLM文本 |", "|---|---:|---:|---:|"])
    for judge_family in families:
        report.append("| " + judge_family + " | " + " | ".join(fmt(mean(family_scores[(judge_family, source_family)])) for source_family in families) + " |")
    (analysis_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "successful_judgments": len(all_rows),
        "paired_orientation_cases": paired,
        "orientation_exact_agreement": consistent / paired if paired else None,
        "exact_self": {judge: mean(exact_self[judge]) for judge in model_keys},
        "cross_model_same_family": {judge: mean(cross_family[judge]) for judge in model_keys},
        "generated_at": utc_now(),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(report))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print("generations:")
    for model in MODELS:
        rows = read_jsonl(args.work_dir / "generations" / f"{model}.jsonl")
        ok = sum(row.get("status") == "ok" for row in rows)
        print(f"  {model}: ok={ok} records={len(rows)}")
    print("judgments:")
    for model in MODELS:
        rows = read_jsonl(args.work_dir / "judgments" / f"{model}.jsonl")
        ok = sum(row.get("status") == "ok" for row in rows)
        print(f"  {model}: ok={ok} records={len(rows)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--limit", type=int, default=0, help="limit prompt count for a smoke run")
    sub = parser.add_subparsers(dest="command", required=True)
    generation = sub.add_parser("generate")
    generation.add_argument("--model", required=True, choices=MODELS)
    generation.add_argument("--max-attempts", type=int, default=8)
    generation.set_defaults(func=cmd_generate)
    prepare = sub.add_parser("prepare")
    prepare.set_defaults(func=cmd_prepare)
    judging = sub.add_parser("judge")
    judging.add_argument("--model", required=True, choices=MODELS)
    judging.set_defaults(func=cmd_judge)
    analyze = sub.add_parser("analyze")
    analyze.set_defaults(func=cmd_analyze)
    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
