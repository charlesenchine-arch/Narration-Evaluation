from narrative_evaluator.quality.schemas import NarrativeItem, NarrativePair
from narrative_evaluator.quality.splitting import split_pairs


def _pair(index, a_group, b_group, a_generator="g1", b_generator="g2"):
    a = NarrativeItem(f"a{index}", f"甲{index}", work_id=a_group, generator=a_generator)
    b = NarrativeItem(f"b{index}", f"乙{index}", work_id=b_group, generator=b_generator)
    return NarrativePair(f"p{index}", a, b)


def test_group_split_has_no_item_or_group_leakage():
    pairs = [_pair(i, f"ga{i}", f"gb{i}") for i in range(30)]
    split = split_pairs(pairs, eval_fraction=0.3, seed=7)
    train_groups = {x.work_id for p in split.train for x in (p.item_a, p.item_b)}
    test_groups = {x.work_id for p in split.test for x in (p.item_a, p.item_b)}
    assert train_groups.isdisjoint(test_groups)
    assert len(split.train) + len(split.test) + len(split.discarded_cross_split) == len(pairs)


def test_generator_holdout_removes_generator_from_training():
    pairs = [
        _pair(0, "w0", "w1", "held", "g2"),
        _pair(1, "w2", "w3", "g1", "g2"),
    ]
    split = split_pairs(pairs, holdout_generator="held")
    assert [p.id for p in split.test] == ["p0"]
    assert [p.id for p in split.train] == ["p1"]
