"""用 LLM 批量盲评评分集，产出"质量分"（P0-B）。

定位：
- LLM-as-Judge 给评分集（600 条）逐篇打 1-5 质量分 + 一句话评语。
- 评分量表/盲评方式与人类评委完全一致（趣味量表：拉完了/NPC/人上人/顶级/夯），
  使 LLM 分与真人分可比——既是解锁解耦实验的"初步质量分"，也是论文
  "无监督评估器 vs LLM-as-Judge vs 人类" 对比基线的 LLM 侧。
- 金标准仍是真人分；LLM 分只作初步质量分 + 论文对比基线。

设计：
- 逐篇独立调用（与人类"一篇一篇评"模式一致），temperature 0.3。
- 输出 JSON {score, comment}，解析失败重试 1 次；API 异常重试 3 次（指数退避）。
- 断点续传：每评完一条追加写入 --out，重新运行跳过已有 id。

用法：
    PYTHONPATH=src python scripts/llm_quality_rating.py \
        --data data/eval_dataset/rating_set.jsonl \
        --out data/eval_dataset/raw/llm_quality_ratings.jsonl
    # 小样本验证：
    ... --limit 5
需要 DEEPSEEK_API_KEY 环境变量。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCORE_GUIDE = """1 = 拉完了（很糟糕，毫无阅读价值）
2 = NPC（平庸，读完无感）
3 = 人上人（中等偏上，有可读性）
4 = 顶级（优秀，引人入胜）
5 = 夯（惊艳，令人印象深刻）"""

SYSTEM_PROMPT = (
    "你是中文叙事文本的资深编辑，负责评估网络小说的叙事质量。"
    "请独立评分，只依据文本本身的叙事质量（情节、人物、语言、氛围、感染力），"
    "不要猜测或假设作者是谁。"
)

# 标签偏差实验（Label Bias Probe）：
# 组 B（correct）：给真实来源标签；组 C（false）：给颠倒来源标签。
# 对照"无标签盲评"，测量 judge 的评分是否被来源先验影响。
LABEL_NONE = "none"
LABEL_CORRECT = "correct"
LABEL_FALSE = "false"

USER_PROMPT_TEMPLATE = """请给下面的叙事文本打一个 1-5 分的质量分，并给出一句中文评语。

评分标准：
{guide}

只输出 JSON，格式：{{"score": <1-5 的整数>, "comment": "<一句话中文评语>"}}

文本：
{text}"""

USER_PROMPT_WITH_LABEL_TEMPLATE = """请给下面的叙事文本打一个 1-5 分的质量分，并给出一句中文评语。

评分标准：
{guide}

文本来源标注：本文由{source}创作。
请把它当作普通文本评价，只依据叙事质量本身，不受来源标签影响。

只输出 JSON，格式：{{"score": <1-5 的整数>, "comment": "<一句话中文评语>"}}

文本：
{text}"""

SOURCE_HUMAN = "人类作家"
SOURCE_MACHINE = "AI 模型"


def _label_source(item, mode):
    """按模式返回 (是否带标签, 来源文案)。false 模式颠倒 H/G。

    锚定文本（AH/AL，id 前缀）不参与标签实验，返回 None（不显示标签）。
    """
    if item.get("id", "").startswith(("AH_", "AL_")):
        return None
    is_h = (item.get("label") == "human")
    if mode == LABEL_CORRECT:
        return SOURCE_HUMAN if is_h else SOURCE_MACHINE
    if mode == LABEL_FALSE:
        return SOURCE_MACHINE if is_h else SOURCE_HUMAN
    return None


def rate_one(client, model, text, temperature=0.3, max_tokens=200, label_source=None):
    """调用一次并解析 JSON，返回 (score, comment)。解析失败返回 None。"""
    if label_source is None:
        prompt = USER_PROMPT_TEMPLATE.format(guide=SCORE_GUIDE, text=text)
    else:
        prompt = USER_PROMPT_WITH_LABEL_TEMPLATE.format(
            guide=SCORE_GUIDE, source=label_source, text=text)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content.strip()
    # 提取第一个 {...} 块（容忍模型输出多余文本）
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None
    obj = json.loads(content[start:end + 1])
    score = int(obj.get("score"))
    comment = str(obj.get("comment", "")).strip()
    if not (1 <= score <= 5):
        return None
    return score, comment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval_dataset/rating_set.jsonl")
    ap.add_argument("--out", default="data/eval_dataset/raw/llm_quality_ratings.jsonl")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（0=全部）")
    ap.add_argument("--sleep", type=float, default=0.0, help="每篇间隔秒数（限流用）")
    ap.add_argument("--label-mode", default=LABEL_NONE, choices=[LABEL_NONE, LABEL_CORRECT, LABEL_FALSE],
                    help="标签偏差实验：none=无标签盲评；correct=给真实来源；false=给颠倒来源")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("需要 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

    items = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit > 0:
        items = items[:args.limit]

    # 断点续传：已评过的 id 跳过
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
    todo = [it for it in items if it["id"] not in done_ids]
    print(f"评分集 {len(items)} 条，已评 {len(done_ids)}，待评 {len(todo)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ok, fail = 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, it in enumerate(todo):
            iid, text = it["id"], it["text"]
            label_source = _label_source(it, args.label_mode)
            result = None
            # JSON 解析失败重试 1 次；API 异常重试 3 次（指数退避）
            for attempt in range(3):
                try:
                    result = rate_one(client, args.model, text, args.temperature,
                                      label_source=label_source)
                    if result is None:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
                except Exception as e:
                    print(f"  {iid} 第{attempt+1}次调用异常: {str(e)[:120]}", flush=True)
                    time.sleep(2.0 * (attempt + 1))
            if result is None:
                print(f"  {iid} 评分失败，跳过", flush=True)
                fail += 1
                continue
            score, comment = result
            rec = {
                "id": iid,
                "rater": f"llm-{args.model}",
                "score": score,
                "comment": comment,
                "model": args.model,
                "temperature": args.temperature,
                "label_mode": args.label_mode,
                "label_source": label_source or "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            ok += 1
            if ok % 25 == 0:
                print(f"  已评 {ok} 条", flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(f"完成：成功 {ok}，失败 {fail}。输出追加至 {args.out}", flush=True)


if __name__ == "__main__":
    main()
