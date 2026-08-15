"""爬取公版现代白话短篇（维基文库）。

合规：维基文库内容为公版/自由版权（CC BY-SA），供研究使用。遵守 robots.txt，用官方 API，
低频抓取，标注来源。目标：鲁迅、朱自清等公版白话短篇，切分成 300-800 字块作 H 样本。

用法：python scripts/scrape_public_domain.py --titles 狂人日记,孔乙己,故乡 --out data/eval_dataset/raw/h_publicdomain.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://zh.wikisource.org/w/api.php"

# 预置公版白话短篇（鲁迅、朱自清等已公版）
DEFAULT_TITLES = [
    "狂人日记", "孔乙己", "故乡", "社戏", "药",
    "风波", "阿Q正傳", "祝福", "傷逝", "在酒樓上",
    "背影", "匆匆", "春", "荷塘月色", "桨声灯影里的秦淮河",
    "遲桂花", "沉淪",
]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "research-evaluator/0.1 (research use)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def get_wikitext(title: str) -> str | None:
    """从维基文库获取页面 wikitext。"""
    params = {
        "action": "parse", "page": title, "prop": "wikitext",
        "format": "json", "formatversion": "2",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    data = json.loads(fetch(url))
    if "error" in data:
        return None
    return data.get("parse", {}).get("wikitext", "")


def clean_wikitext(ws: str) -> str:
    """清洗 wikitext：去掉模板、分类、链接语法，保留正文。"""
    # 去模板 {{...}}
    ws = re.sub(r"\{\{.*?\}\}", "", ws, flags=re.DOTALL)
    # 去分类 [[Category:...]]
    ws = re.sub(r"\[\[(Category|分类):.*?\]\]", "", ws, flags=re.DOTALL)
    # 去注释 <!-- ... -->
    ws = re.sub(r"<!--.*?-->", "", ws, flags=re.DOTALL)
    # 去 [[链接]] 保留文字
    ws = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", ws)
    # 去 ref
    ws = re.sub(r"<ref.*?</ref>", "", ws, flags=re.DOTALL)
    ws = re.sub(r"<[^>]+>", "", ws)
    # 去音译/拼音行（鲁迅原作常附注）
    lines = [l for l in ws.split("\n") if not re.match(r"^\s*[A-Za-zāáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜü]+\s*$", l)]
    return "\n".join(lines).strip()


def chunk_short(text: str, lo: int = 300, hi: int = 800) -> list:
    """按段落切分成 [lo, hi] 字块（复用 chunk 逻辑）。"""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) > hi and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if lo <= len(c) <= hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", default=",".join(DEFAULT_TITLES))
    ap.add_argument("--out", default="data/eval_dataset/raw/h_publicdomain.jsonl")
    ap.add_argument("--lo", type=int, default=300)
    ap.add_argument("--hi", type=int, default=800)
    args = ap.parse_args()

    titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_items = []
    for title in titles:
        try:
            ws = get_wikitext(title)
            time.sleep(0.5)  # 低频
        except Exception as e:
            print(f"  {title}: 抓取失败 {str(e)[:60]}", flush=True)
            continue
        if not ws:
            print(f"  {title}: 未找到", flush=True)
            continue
        clean = clean_wikitext(ws)
        chunks = chunk_short(clean, args.lo, args.hi)
        if not chunks:
            print(f"  {title}: 无达标块", flush=True)
            continue
        for c in chunks:
            all_items.append({
                "text": c, "label": "human", "source": "public_domain",
                "work": title,
            })
        print(f"  {title}: {len(chunks)} 块", flush=True)

    print(f"\n共收集 {len(all_items)} 块", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, item in enumerate(all_items):
            item["id"] = f"PD_{i:04d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
