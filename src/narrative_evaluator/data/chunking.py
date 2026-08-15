"""文本切块与去重工具（训练数据准备复用）。

chunk_text 从 scripts/chunk_short_stories.py 迁入，供 MacBERT 训练脚本把
长章节切成 300-800 字的短文块；dedupe_vs 用于防止训练集泄漏验证集。
"""
from __future__ import annotations

from typing import Iterable, List


def chunk_text(text: str, lo: int = 300, hi: int = 800) -> list:
    """按段落边界切分文本成 [lo, hi] 字的连续块。返回块列表。"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    cur = []
    cur_len = 0
    for p in paras:
        if cur_len + len(p) > hi and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    # 只保留长度达标的块
    return [c for c in chunks if lo <= len(c) <= hi]


def chunk_texts(texts: Iterable[str], lo: int = 300, hi: int = 800) -> List[str]:
    """批量切块，返回所有达标块的列表。"""
    out = []
    for t in texts:
        out.extend(chunk_text(t, lo, hi))
    return out


def dedupe_vs(candidate: Iterable[str], exclude: Iterable[str]) -> List[str]:
    """删除 candidate 中与 exclude 精确文本重复的条目（防训练/验证泄漏）。"""
    block = set(exclude)
    return [c for c in candidate if c not in block]
