"""Leakage-aware train/test splits for pairwise quality experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .schemas import NarrativeItem, NarrativePair


@dataclass(frozen=True)
class PairSplit:
    train: tuple[NarrativePair, ...]
    test: tuple[NarrativePair, ...]
    discarded_cross_split: tuple[NarrativePair, ...] = ()
    mode: str = "group_disjoint"


def _group_key(item: NarrativeItem) -> str:
    """Return the strongest available independence unit for an item."""
    if item.work_id:
        return f"work:{item.work_id}"
    if item.prompt_id:
        return f"prompt:{item.prompt_id}"
    return f"item:{item.id}"


def split_pairs(
    pairs: Iterable[NarrativePair],
    eval_fraction: float = 0.2,
    seed: int = 42,
    mode: str = "group_disjoint",
    holdout_generator: str = "",
) -> PairSplit:
    """Split labeled pairs without leaking held-out text into training.

    ``group_disjoint`` holds out complete work/prompt groups.  When metadata is
    absent, the item itself is the group, so an item never appears on both
    sides.  Pairs crossing the boundary are intentionally discarded.
    ``generator_holdout`` reserves every pair containing the named generator.
    ``pair_random`` exists only for smoke tests and must not be used as a paper
    result because the same text can occur in train and test.
    """
    rows = list(pairs)
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    if holdout_generator:
        mode = "generator_holdout"
    if mode not in {"group_disjoint", "generator_holdout", "pair_random"}:
        raise ValueError(f"unsupported split mode: {mode}")

    if mode == "generator_holdout":
        name = holdout_generator.strip()
        if not name:
            raise ValueError("generator_holdout requires holdout_generator")
        test = tuple(
            pair for pair in rows
            if name in {pair.item_a.generator, pair.item_b.generator}
        )
        train = tuple(pair for pair in rows if pair not in test)
        return PairSplit(train=train, test=test, mode=mode)

    rng = random.Random(seed)
    if mode == "pair_random":
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * eval_fraction)) if len(rows) > 1 else 0
        return PairSplit(train=tuple(rows[n_test:]), test=tuple(rows[:n_test]), mode=mode)

    group_keys = sorted({
        _group_key(item)
        for pair in rows
        for item in (pair.item_a, pair.item_b)
    })
    rng.shuffle(group_keys)
    n_test_groups = max(1, round(len(group_keys) * eval_fraction)) if len(group_keys) > 1 else 0
    test_groups = set(group_keys[:n_test_groups])
    train, test, crossing = [], [], []
    for pair in rows:
        a_test = _group_key(pair.item_a) in test_groups
        b_test = _group_key(pair.item_b) in test_groups
        if a_test and b_test:
            test.append(pair)
        elif not a_test and not b_test:
            train.append(pair)
        else:
            crossing.append(pair)
    return PairSplit(
        train=tuple(train),
        test=tuple(test),
        discarded_cross_split=tuple(crossing),
        mode=mode,
    )
