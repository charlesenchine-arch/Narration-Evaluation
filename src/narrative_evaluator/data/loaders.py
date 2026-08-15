"""HF 数据源流式加载器。

统一通过 datasets.load_dataset(..., streaming=True) 从远程 json/jsonl 流式读取，
不落盘；HF 缓存由 config.cache_dir 指向临时目录，运行结束后由 cli 清理。

每个 loader 返回一个 Dataset 对象，包含统一的 TextRecord 列表。
sample 参数 <=0 表示取全部。
"""
from __future__ import annotations

import random
from typing import Callable, Iterator, Optional

from datasets import load_dataset

from ..config import DataConfig
from .schemas import TextRecord, Dataset

# 仓库内文件路径
WEB_NOVEL_FILE = "novel_data/novel_data_subset_d_100.json"
CNNSUM_FILES = [
    "cnnsum_L_16k.jsonl",
    "cnnsum_XL_32k.jsonl",
]
COIG_WRITER_FILE = "all_human_data.json"
REALDET_MGT_FILE = "MGT_cn_all.jsonl"


def _make_gen(seed: int) -> random.Random:
    return random.Random(seed)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def _stream_json_lines(
    repo_id: str,
    file_path: str,
    cache_dir: str,
    parse: Callable[[dict], Optional[object]],
    multi: bool = False,
) -> Iterator[TextRecord]:
    """流式读取 HF 仓库中的 json/jsonl 文件并解析。

    multi=False：parse 返回单个 TextRecord 或 None；
    multi=True：parse 返回 TextRecord 列表或 None。
    """
    # load_dataset 对 json 与 jsonl 均按行流式解析；两种扩展名都走 'json' builder
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/{repo_id}/{file_path}",
        streaming=True,
        cache_dir=cache_dir,
        split="train",
    )
    for item in ds:
        rec = parse(item)
        if rec is None:
            continue
        if multi:
            for r in rec:
                yield r
        else:
            yield rec


def _multi_file_stream(
    repo_id: str,
    files: list,
    cache_dir: str,
    parse: Callable[[dict], Optional[TextRecord]],
) -> Iterator[TextRecord]:
    """依次流式读取多个文件。"""
    for fp in files:
        yield from _stream_json_lines(repo_id, fp, cache_dir, parse)


def _sample_iterator(
    it: Iterator[TextRecord],
    sample: int,
    rng: random.Random,
) -> list:
    """蓄水池采样：流式场景下从无限迭代器取至多 sample 条（sample<=0 全取）。"""
    if sample <= 0:
        return list(it)
    reservoir: list = []
    for i, rec in enumerate(it):
        if i < sample:
            reservoir.append(rec)
        else:
            j = rng.randint(0, i)
            if j < sample:
                reservoir[j] = rec
    return reservoir


def load_webnovelbench(
    data_cfg: DataConfig,
    cache_dir: str,
    sample: Optional[int] = None,
) -> Dataset:
    """WebNovelBench novel_data_subset_d_100.json：中文小说 + 人类质量分。

    每条记录含 chapters（章节列表）与 original score（多评委分）。
    将每个章节作为一条 TextRecord，human_score = 归一化后的评委均分。
    章节带书名上下文以保证文本完整性。
    """
    repo = "Oedon42/webnovelbench"

    def parse(item: dict) -> Optional[TextRecord]:
        chapters = item.get("chapters") or []
        if not chapters:
            return None
        scores = item.get("original score") or item.get("original_score") or []
        mean = float(sum(scores)) / len(scores) if scores else None
        # 归一化到 [0,1]（评委分约 1-5）
        human_score = (mean - 1.0) / 4.0 if mean is not None else None
        novel = item.get("novel", "")
        out = []
        for ch in chapters:
            ch_text = _truncate(str(ch), data_cfg.max_chars)
            if not ch_text.strip():
                continue
            out.append(TextRecord(
                text=ch_text,
                label="human",
                human_score=human_score,
                source="webnovelbench",
                meta={"novel": novel},
            ))
        return out if out else None

    rng = _make_gen(data_cfg.seed)
    n = data_cfg.n_h_webnovel if sample is None else sample
    it = _stream_json_lines(repo, WEB_NOVEL_FILE, cache_dir, parse, multi=True)
    return Dataset(name="webnovelbench-human", records=_sample_iterator(it, n, rng))


def load_cnnsum(
    data_cfg: DataConfig,
    cache_dir: str,
    sample: Optional[int] = None,
) -> Dataset:
    """CNNSum：长篇人类中文小说摘录，字段 context/summary。用 context（正文）作文本。"""
    repo = "CxsGhost/CNNSum"

    def parse(item: dict) -> Optional[TextRecord]:
        ctx = item.get("context")
        if not ctx or not str(ctx).strip():
            return None
        return TextRecord(
            text=_truncate(str(ctx), data_cfg.max_chars_cnnsum),
            label="human",
            source="cnnsum",
            meta={"summary_len": len(str(item.get("summary", "")))},
        )

    rng = _make_gen(data_cfg.seed)
    n = data_cfg.n_h_cnnsum if sample is None else sample
    it = _multi_file_stream(repo, CNNSUM_FILES, cache_dir, parse)
    return Dataset(name="cnnsum-human", records=_sample_iterator(it, n, rng))


def load_coig_writer(
    data_cfg: DataConfig,
    cache_dir: str,
    sample: Optional[int] = None,
) -> Dataset:
    """COIG-Writer：AI 中文创意写作（散文/诗歌/作文等），字段 id/answer。"""
    repo = "m-a-p/COIG-Writer"

    def parse(item: dict) -> Optional[TextRecord]:
        ans = item.get("answer")
        if not ans or not str(ans).strip():
            return None
        return TextRecord(
            text=_truncate(str(ans), data_cfg.max_chars),
            label="machine",
            source="coig-writer",
            meta={"id": item.get("id", "")},
        )

    rng = _make_gen(data_cfg.seed)
    n = data_cfg.n_g_coig if sample is None else sample
    it = _stream_json_lines(repo, COIG_WRITER_FILE, cache_dir, parse)
    return Dataset(name="coig-writer-machine", records=_sample_iterator(it, n, rng))


def load_realdet_mgt(
    data_cfg: DataConfig,
    cache_dir: str,
    sample: Optional[int] = None,
) -> Dataset:
    """RealDet MGT_cn_all.jsonl：多模型机器中文文本，字段 text/label（生成器名）。

    内容偏问答/论坛，默认不启用（n_g_realdet=0）。
    """
    repo = "koakuma/RealDet"

    def parse(item: dict) -> Optional[TextRecord]:
        txt = item.get("text")
        if not txt or not str(txt).strip():
            return None
        return TextRecord(
            text=_truncate(str(txt), data_cfg.max_chars),
            label="machine",
            source="realdet",
            meta={"generator": item.get("label", "")},
        )

    rng = _make_gen(data_cfg.seed)
    n = data_cfg.n_g_realdet if sample is None else sample
    it = _stream_json_lines(repo, REALDET_MGT_FILE, cache_dir, parse)
    return Dataset(name="realdet-mgt-machine", records=_sample_iterator(it, n, rng))


REALDET_HWT_FILE = "HWT_cn_all.jsonl"


def load_realdet_hwt(
    data_cfg: DataConfig,
    cache_dir: str,
    sample: Optional[int] = None,
) -> Dataset:
    """RealDet HWT_cn_all.jsonl：人类中文短文（问答/论坛），字段 text，label=Human。

    内容偏问答/论坛，但提供"短文人类样本"，用于需要短文人类参照的实验
    （如构造解耦证据）。
    """
    repo = "koakuma/RealDet"

    def parse(item: dict) -> Optional[TextRecord]:
        txt = item.get("text")
        if not txt or not str(txt).strip():
            return None
        return TextRecord(
            text=_truncate(str(txt), data_cfg.max_chars),
            label="human",
            source="realdet-hwt",
            meta={},
        )

    rng = _make_gen(data_cfg.seed)
    n = data_cfg.n_h_realdet if sample is None else sample
    it = _stream_json_lines(repo, REALDET_HWT_FILE, cache_dir, parse)
    return Dataset(name="realdet-hwt-human", records=_sample_iterator(it, n, rng))
