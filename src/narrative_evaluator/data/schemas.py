"""统一的数据记录结构与数据集容器。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextRecord:
    """一条评估样本。

    label: "human" | "machine" | None（未知）
    human_score: 人类质量评分（0-1 归一化），仅 WebNovelBench 等带评分源有值
    meta: 附加元数据（数据源、生成器名、主题等）
    """
    text: str
    label: Optional[str] = None
    human_score: Optional[float] = None
    source: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    """一组文本记录，附带统计信息。"""
    name: str = ""
    records: list = field(default_factory=list)

    @property
    def texts(self) -> list:
        return [r.text for r in self.records]

    @property
    def labels(self) -> list:
        return [r.label for r in self.records]

    def __len__(self) -> int:
        return len(self.records)
