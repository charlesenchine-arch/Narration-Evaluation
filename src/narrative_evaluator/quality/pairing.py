"""Construct balanced comparison pairs while limiting obvious shortcuts."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable

from .schemas import NarrativeItem, NarrativePair


def _length_ratio(a: NarrativeItem, b: NarrativeItem) -> float:
    short, long = sorted((max(1, len(a.text)), max(1, len(b.text))))
    return long / short


def _pair_cost(a: NarrativeItem, b: NarrativeItem, counts: Counter, rng: random.Random) -> float:
    ratio = _length_ratio(a, b)
    cost = abs(math.log(ratio)) * 4.0
    if a.prompt_id and a.prompt_id == b.prompt_id:
        cost -= 4.0
    if a.genre and a.genre == b.genre:
        cost -= 1.2
    if a.source == b.source:
        cost -= 0.15  # keep same-source comparisons represented
    if a.work_id and a.work_id == b.work_id:
        cost += 2.0   # discourage near-duplicate chapters from the same work
    cost += 0.8 * (counts[a.id] + counts[b.id])
    return cost + rng.random() * 1e-6


def _match_type(a: NarrativeItem, b: NarrativeItem) -> str:
    if a.prompt_id and a.prompt_id == b.prompt_id:
        return "same_prompt"
    if a.source == b.source:
        return "same_source"
    return "cross_source"


def build_balanced_pairs(
    items: Iterable[NarrativeItem],
    comparisons_per_item: int = 2,
    max_length_ratio: float = 1.6,
    seed: int = 42,
) -> list[NarrativePair]:
    """Greedily build a balanced incomplete comparison design.

    Every item is targeted to appear ``comparisons_per_item`` times.  Candidate
    pairs prefer shared prompts/genres and similar lengths, while retaining both
    same-source and cross-source comparisons.  Work/prompt ids are copied to the
    pair's split group so downstream splits can keep related samples together.
    """
    items = list(items)
    if len(items) < 2:
        return []
    if comparisons_per_item < 1:
        raise ValueError("comparisons_per_item must be positive")
    if max_length_ratio < 1.0:
        raise ValueError("max_length_ratio must be >= 1")

    rng = random.Random(seed)
    counts: Counter = Counter()
    used: set[tuple[str, str]] = set()
    pairs: list[NarrativePair] = []
    target_edges = math.ceil(len(items) * comparisons_per_item / 2)

    while len(pairs) < target_edges:
        eligible = [item for item in items if counts[item.id] < comparisons_per_item]
        if len(eligible) < 2:
            break
        # Start from an underrepresented item and scan possible partners.  This
        # is O(edges * items), rather than rebuilding every O(items²) candidate
        # pair for each edge.
        rng.shuffle(eligible)
        eligible.sort(key=lambda item: counts[item.id])
        chosen = None
        for a in eligible:
            strict = [
                b for b in eligible
                if b.id != a.id
                and tuple(sorted((a.id, b.id))) not in used
                and _length_ratio(a, b) <= max_length_ratio
            ]
            relaxed = [
                b for b in eligible
                if b.id != a.id and tuple(sorted((a.id, b.id))) not in used
            ]
            pool = strict or relaxed
            if pool:
                b = min(
                    pool,
                    key=lambda other: _pair_cost(a, other, counts, rng)
                    + (10.0 if other not in strict else 0.0),
                )
                chosen = (a, b)
                break
        if chosen is None:
            break
        a, b = chosen
        if rng.random() < 0.5:
            a, b = b, a
        used.add(tuple(sorted((a.id, b.id))))
        counts[a.id] += 1
        counts[b.id] += 1
        split_group = a.prompt_id or b.prompt_id or a.work_id or b.work_id or ""
        pairs.append(NarrativePair(
            id=f"P_{len(pairs):05d}",
            item_a=a,
            item_b=b,
            match_type=_match_type(a, b),
            split_group=split_group,
            meta={
                "length_ratio": round(_length_ratio(a, b), 4),
                "source_pair": sorted((a.source, b.source)),
            },
        ))
    return pairs
