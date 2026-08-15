"""跨生成器验证：用通义千问（Qwen，第二生成器）生成机器叙事短文 G 样本。

与 DeepSeek 版（generate_g_texts.py）同一批叙事 prompt，温度多样性。
输出对齐 g_gen.jsonl 的 schema：{id, text, label, gen_model, prompt, temperature}。

key 从 secrets.env 或环境变量读取（QWEN_API_KEY / QWEN_BASE_URL）。
用法：python scripts/generate_g_qwen.py --n 100 --out data/eval_dataset/raw/g_qwen.jsonl \
      --model qwen3.5-flash
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


def _load_key():
    """从 secrets.env 或环境变量读 key/base_url。"""
    key = os.environ.get("QWEN_API_KEY", "")
    base = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), "..", "secrets.env")
        if os.path.exists(env_path):
            for l in open(env_path, encoding="utf-8"):
                l = l.strip()
                if l.startswith("QWEN_API_KEY="):
                    key = l.split("=", 1)[1].strip()
                elif l.startswith("QWEN_BASE_URL="):
                    base = l.split("=", 1)[1].strip()
    return key, base


def generate_one(client, model, prompt, temperature, no_think=False):
    kwargs = {}
    if no_think:
        # qwen3.x 默认开 thinking，单条烧数千 reasoning token；关闭后 ~10 倍省量
        kwargs["extra_body"] = {"enable_thinking": False}
    r = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return (r.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/eval_dataset/raw/g_qwen.jsonl")
    ap.add_argument("--model", default="qwen-flash")
    ap.add_argument("--per-prompt", type=int, default=13)
    ap.add_argument("--no-think", action="store_true", help="qwen3.x 关 thinking，省 ~10 倍输出 token")
    ap.add_argument("--tag", default="G_Q", help="id 前缀，区分批次（如 G_Q7）")
    args = ap.parse_args()

    key, base = _load_key()
    if not key:
        print("未找到 QWEN_API_KEY（secrets.env 或环境变量）", file=os.sys.stderr)
        return
    client = OpenAI(api_key=key, base_url=base, timeout=120)

    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for l in f:
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass
    temps = [0.7, 0.85, 1.0, 1.2]
    count = len(done_ids)
    idx = 0
    n_per_prompt = max(1, args.per_prompt)
    with open(args.out, "a", encoding="utf-8") as f:
        while count < args.n:
            # 每条轮换 prompt（避免同 prompt 连排 → 低多样性近重复）
            pi = idx % len(PROMPTS)
            prompt = PROMPTS[pi]
            temperature = temps[idx % len(temps)]
            iid = f"{args.tag}{count:04d}"
            if iid in done_ids:
                idx += 1
                continue
            try:
                text = generate_one(client, args.model, prompt, temperature, no_think=args.no_think)
            except Exception as e:
                print(f"  {iid} 失败: {str(e)[:80]}", flush=True)
                import time
                time.sleep(5)
                continue
            if not text:
                print(f"  {iid} 空响应", flush=True)
                idx += 1
                continue
            rec = {"id": iid, "text": text, "label": "machine",
                   "gen_model": args.model, "prompt": prompt, "temperature": temperature}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            done_ids.add(iid)
            if count % 10 == 0:
                print(f"  已生成 {count}/{args.n}", flush=True)
            idx += 1
    print(f"完成：{count} 条 -> {args.out}")


if __name__ == "__main__":
    main()
