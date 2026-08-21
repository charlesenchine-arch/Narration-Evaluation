from collections import Counter

from narrative_evaluator.quality.pairing import build_balanced_pairs
from narrative_evaluator.quality.schemas import NarrativeItem


def _item(index, length=100, prompt=""):
    return NarrativeItem(
        id=f"i{index}", text="文" * (length + index), source="s" if index % 2 else "t",
        prompt_id=prompt, genre="现实",
    )


def test_balanced_pairs_are_unique_and_position_balanced():
    pairs = build_balanced_pairs([_item(i) for i in range(8)], comparisons_per_item=2, seed=3)
    unordered = {tuple(sorted((p.item_a.id, p.item_b.id))) for p in pairs}
    assert len(unordered) == len(pairs) == 8
    counts = Counter(item.id for pair in pairs for item in (pair.item_a, pair.item_b))
    assert set(counts.values()) == {2}


def test_same_prompt_is_preferred():
    items = [_item(0, prompt="p"), _item(1, prompt="p"), _item(2, prompt="q"), _item(3, prompt="q")]
    pairs = build_balanced_pairs(items, comparisons_per_item=1, seed=1)
    assert all(pair.match_type == "same_prompt" for pair in pairs)
