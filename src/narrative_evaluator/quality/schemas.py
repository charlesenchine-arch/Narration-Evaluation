"""Portable JSONL schemas for pairwise narrative-quality research."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar


PREFERENCE_VALUES = (-2, -1, 0, 1, 2)
QUALITY_DIMENSIONS = (
    "plot_causality",
    "character",
    "language_style",
    "atmosphere_impact",
    "pacing",
    "narrative_structure_time",
    "originality",
)


@dataclass(frozen=True)
class NarrativeItem:
    id: str
    text: str
    source: str = ""
    work_id: str = ""
    prompt_id: str = ""
    generator: str = ""
    genre: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("NarrativeItem.id must not be empty")
        if not self.text or not self.text.strip():
            raise ValueError(f"NarrativeItem {self.id!r} has empty text")

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "NarrativeItem":
        known = {k: obj.get(k, "") for k in (
            "id", "text", "source", "work_id", "prompt_id", "generator", "genre"
        )}
        meta = dict(obj.get("meta") or {})
        for key, value in obj.items():
            if key not in known and key != "meta":
                meta.setdefault(key, value)
        return cls(**known, meta=meta)


@dataclass(frozen=True)
class NarrativePair:
    id: str
    item_a: NarrativeItem
    item_b: NarrativeItem
    match_type: str = "matched"
    split_group: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("NarrativePair.id must not be empty")
        if self.item_a.id == self.item_b.id:
            raise ValueError("a pair must contain two different items")

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "NarrativePair":
        return cls(
            id=str(obj["id"]),
            item_a=NarrativeItem.from_dict(obj["item_a"]),
            item_b=NarrativeItem.from_dict(obj["item_b"]),
            match_type=str(obj.get("match_type", "matched")),
            split_group=str(obj.get("split_group", "")),
            meta=dict(obj.get("meta") or {}),
        )


@dataclass(frozen=True)
class PairwiseJudgment:
    pair_id: str
    rater_id: str
    preference: int
    confidence: int
    dimensions: tuple[str, ...] = ()
    rationale: str = ""
    evidence_a: str = ""
    evidence_b: str = ""
    stay_ms: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pair_id:
            raise ValueError("PairwiseJudgment.pair_id must not be empty")
        if not self.rater_id:
            raise ValueError("PairwiseJudgment.rater_id must not be empty")
        if self.preference not in PREFERENCE_VALUES:
            raise ValueError("preference must be one of -2,-1,0,1,2")
        if not 1 <= int(self.confidence) <= 5:
            raise ValueError("confidence must be in [1, 5]")
        unknown = set(self.dimensions) - set(QUALITY_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown quality dimensions: {sorted(unknown)}")
        if self.stay_ms is not None and self.stay_ms < 0:
            raise ValueError("stay_ms must be non-negative")

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "PairwiseJudgment":
        return cls(
            pair_id=str(obj["pair_id"]),
            rater_id=str(obj["rater_id"]),
            preference=int(obj["preference"]),
            confidence=int(obj["confidence"]),
            dimensions=tuple(obj.get("dimensions") or ()),
            rationale=str(obj.get("rationale", "")),
            evidence_a=str(obj.get("evidence_a", "")),
            evidence_b=str(obj.get("evidence_b", "")),
            stay_ms=(int(obj["stay_ms"]) if obj.get("stay_ms") is not None else None),
            meta=dict(obj.get("meta") or {}),
        )


SchemaT = TypeVar("SchemaT", NarrativeItem, NarrativePair, PairwiseJudgment)


def read_jsonl(path: str | Path, cls: type[SchemaT]) -> list[SchemaT]:
    rows: list[SchemaT] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(cls.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid {cls.__name__} at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
