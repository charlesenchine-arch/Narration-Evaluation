"""用 LLM 生成机器叙事短文（G 样本）。

从 H 短文块启发主题，用 DeepSeek 生成 300-800 字的机器叙事。
覆盖多个 prompt（不同叙事类型），temperature 多样性。

用法：python scripts/generate_g_texts.py --n 300 --out data/eval_dataset/raw/g_gen.jsonl
需要 DEEPSEEK_API_KEY 环境变量。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

# 叙事 prompt 池（覆盖不同叙事类型）
PROMPTS = [
    "写一个300到800字的中文微小说，关于一次意外的重逢。要有情节转折和人物细节。",
    "写一个300到800字的中文叙事片段，描写一个雨夜中发生的小故事。注重氛围和感官描写。",
    "写一个300到800字的中文短篇，讲一个人在城市里寻找丢失的猫的过程。带一点悬念。",
    "写一个300到800字的中文叙事，描述一次家庭聚餐时发生的尴尬时刻。要有对话。",
    "写一个300到800字的中文短故事，关于一个老人整理旧物时发现的一封信。情感真挚。",
    "写一个300到800字的中文叙事片段，讲述一次深夜火车旅行中的邂逅。",
    "写一个300到800字的中文微小说，主角是个刚搬进新小区的年轻人。",
    "写一个300到800字的中文叙事，关于一场停电让陌生人相互认识的夜晚。",
]

# 带章回标记的叙事 prompt 池（机器生成的"章回体"文本，覆盖章回特征）
ZHANGHUI_PROMPTS = [
    "写一个300到800字的中文章回体叙事片段，用“第X章”作为段落标题（至少两个章节）。故事关于一次意外的重逢。要有情节转折和人物细节。",
    "写一个300到800字的中文章回体叙事，用“第X章”分章（至少两章），描写一个雨夜中发生的小故事。注重氛围和感官描写。",
    "写一个300到800字的中文章回体短篇，用“第X章”分章（至少两章），讲一个人在城市里寻找丢失的猫的过程。带一点悬念。",
    "写一个300到800字的中文章回体叙事，用“第X章”分章（至少两章），描述一次家庭聚餐时发生的尴尬时刻。要有对话。",
    "写一个300到800字的中文章回体短故事，用“第X章”分章（至少两章），关于一个老人整理旧物时发现的一封信。情感真挚。",
    "写一个300到800字的中文章回体叙事片段，用“第X章”分章（至少两章），讲述一次深夜火车旅行中的邂逅。",
    "写一个300到800字的中文章回体微小说，用“第X章”分章（至少两章），主角是个刚搬进新小区的年轻人。",
    "写一个300到800字的中文章回体叙事，用“第X章”分章（至少两章），关于一场停电让陌生人相互认识的夜晚。",
]


def generate_one(client, model, prompt, temperature):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/eval_dataset/raw/g_gen.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--temps", default="0.7,1.0,1.3")
    ap.add_argument("--zhanghui", action="store_true",
                    help="生成带章回标记（第X章）的文本，覆盖章回特征避免泄漏")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("需要 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

    prompts = ZHANGHUI_PROMPTS if args.zhanghui else PROMPTS
    temps = [float(x) for x in args.temps.split(",")]
    rng = random.Random(args.seed)

    import os as _os
    _os.makedirs(_os.path.dirname(args.out), exist_ok=True)

    print(f"生成 {args.n} 条{'章回体' if args.zhanghui else ''}叙事短文...", flush=True)
    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            prompt = rng.choice(prompts)
            temp = rng.choice(temps)
            try:
                text = generate_one(client, args.model, prompt, temp)
            except Exception as e:
                print(f"  第{i}条生成失败: {str(e)[:100]}", flush=True)
                continue
            # 清洗：去引号包裹
            text = text.strip('"')
            if len(text) < 200:  # 太短弃用
                print(f"  第{i}条过短({len(text)}字)，跳过", flush=True)
                continue
            item = {
                "id": f"G_{count:04d}",
                "text": text,
                "label": "machine",
                "gen_model": args.model,
                "prompt": prompt,
                "temperature": temp,
                "source": "deepseek_zhanghui" if args.zhanghui else "deepseek",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count % 50 == 0:
                print(f"  已生成 {count} 条", flush=True)
    print(f"完成：{count} 条写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
