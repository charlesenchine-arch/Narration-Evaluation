"""可解释属性提取：把文本映射到一组可解释的统计属性（句子结构、词汇、标点等）。

每个属性是单篇文本上的一个标量统计量，用于：
- 单篇打分时与 H 参照分布的 z-score 距离；
- evaluate_set 时逐属性比较 H/G 分布差异（Wasserstein）。
"""
from __future__ import annotations

import re
from typing import Dict, List

import jieba

# 常用中文连接词/语气词（用于"连接词频率"属性）
CONNECTIVES = set(
    "但 但是 然而 可是 不过 因此 所以 于是 因为 既然 虽然 尽管 即使 如果 只要 "
    "而且 并且 然后 接着 此外 同时 不但 不仅 一边 一方面 结果 从而 以致 随着 只要".split()
)

# 句子切分：中文句末标点 + 常见分隔
_SENT_SPLIT = re.compile(r"[。！？!?；;…]+")

# 抽象词性（用于抽象/具体比例，粗粒度：从 jieba 词性标注提取）
ABSTRACT_POS = {"n", "nz", "nt", "nr", "ns", "vn", "r", "v"}


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def extract_attributes(text: str) -> Dict[str, float]:
    """提取一组可解释属性。空文本返回全零。"""
    if not text or not text.strip():
        return {k: 0.0 for k in ATTRIBUTE_KEYS}

    sents = _split_sentences(text)
    n_sents = len(sents)
    sent_lens = [len(s) for s in sents]
    total_chars = len(text)

    # jieba 分词（全文本，去空白标点后统计）
    words = [w for w in jieba.lcut(text) if w.strip()]
    n_words = len(words)
    unique_words = set(words)

    # 句子长度分布
    mean_sent_len = sum(sent_lens) / n_sents if n_sents else 0.0
    if n_sents >= 2:
        var_sent_len = sum((x - mean_sent_len) ** 2 for x in sent_lens) / n_sents
    else:
        var_sent_len = 0.0

    # 词汇丰富度：TTR（类符/形符比）
    ttr = len(unique_words) / n_words if n_words else 0.0

    # 重复率：出现超过一次的词语比例
    from collections import Counter
    wc = Counter(words)
    repeated = sum(1 for w, c in wc.items() if c > 1)
    repeat_ratio = repeated / len(wc) if wc else 0.0

    # 标点密度与频率
    punct = re.findall(r"[，。！？；：、""''（）《》——…]", text)
    punct_density = len(punct) / total_chars if total_chars else 0.0
    comma_ratio = punct.count("，") / len(punct) if punct else 0.0
    exclam_ratio = punct.count("！") / len(punct) if punct else 0.0

    # 连接词频率
    connective_freq = sum(1 for w in words if w in CONNECTIVES) / n_words if n_words else 0.0

    # 信息密度：每句平均词数（近似）
    info_density = n_words / n_sents if n_sents else 0.0

    # 段落结构：空行分隔的段数（粗略）
    n_paras = len([p for p in text.split("\n") if p.strip()]) if "\n" in text else 1

    return {
        "sentence_len_mean": mean_sent_len,
        "sentence_len_var": var_sent_len,
        "lexical_richness_ttr": ttr,
        "word_repeat_ratio": repeat_ratio,
        "punct_density": punct_density,
        "comma_ratio": comma_ratio,
        "exclam_ratio": exclam_ratio,
        "connective_freq": connective_freq,
        "info_density": info_density,
        "n_paras": n_paras,
        "n_sents": n_sents,
    }


ATTRIBUTE_KEYS = [
    "sentence_len_mean",
    "sentence_len_var",
    "lexical_richness_ttr",
    "word_repeat_ratio",
    "punct_density",
    "comma_ratio",
    "exclam_ratio",
    "connective_freq",
    "info_density",
    "n_paras",
    "n_sents",
]
