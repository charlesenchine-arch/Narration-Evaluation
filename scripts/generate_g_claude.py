"""跨生成器验证：用 DeepSeek v4-pro（第二生成器）生成机器叙事短文 G 样本。

与 DeepSeek 版（generate_g_texts.py，deepseek-chat）不同的模型，同一批叙事 prompt，
温度多样性。输出对齐 g_gen.jsonl 的 schema：{id, text, label, gen_model, prompt, temperature}。

用法：python scripts/generate_g_claude.py --n 100 --out data/eval_dataset/raw/g_pro.jsonl
需要 DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

# 与 generate_g_texts.py 相同的叙事 prompt 池（覆盖不同叙事类型）
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


def generate_one(client, model, prompt, temperature):
    r = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="要生成的条数")
    ap.add_argument("--out", default="data/eval_dataset/raw/g_pro.jsonl")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--per-prompt", type=int, default=13, help="每个 prompt 生成的条数")
    args = ap.parse_args()

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    # 断点续传：读已有条目
    done_ids = set()
    out_path = args.out
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for l in f:
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass
    # 生成循环：prompt 轮转 + 温度轮转
    temps = [0.6, 0.7, 0.85, 1.0]
    count = len(done_ids)
    idx = 0
    n_per_prompt = max(1, args.per_prompt)
    with open(out_path, "a", encoding="utf-8") as f:
        while count < args.n:
            pi = (idx // n_per_prompt) % len(PROMPTS)
            prompt = PROMPTS[pi]
            temperature = temps[idx % len(temps)]
            iid = f"G_P{count:04d}"
            if iid in done_ids:
                idx += 1
                continue
            try:
                text = generate_one(client, args.model, prompt, temperature)
            except Exception as e:
                print(f"  {iid} 失败: {e}", flush=True)
                import time
                time.sleep(5)
                continue
            if not text:
                continue
            rec = {
                "id": iid, "text": text, "label": "machine",
                "gen_model": args.model, "prompt": prompt, "temperature": temperature,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            done_ids.add(iid)
            if count % 10 == 0:
                print(f"  已生成 {count}/{args.n}", flush=True)
            idx += 1
    print(f"完成：{count} 条 -> {out_path}")


if __name__ == "__main__":
    main()
