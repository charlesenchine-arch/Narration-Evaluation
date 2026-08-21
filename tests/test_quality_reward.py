from narrative_evaluator.quality.aggregation import AggregatedPreference
from narrative_evaluator.quality.reward import PairwiseTfidfRewardModel
from narrative_evaluator.quality.schemas import NarrativeItem, NarrativePair


def _preference(pair_id, probability):
    return AggregatedPreference(
        pair_id=pair_id,
        probability_a_better=probability,
        mean_preference=(probability * 4) - 2,
        majority_preference=1 if probability > 0.5 else -1,
        agreement=1.0,
        n_raters=3,
        effective_weight=3.0,
        dimension_counts={},
        rationales=(),
    )


def test_reward_model_learns_shared_utility_and_grades(tmp_path):
    strong = NarrativeItem("strong", "人物动机清楚，冲突有铺垫，结尾完成前文伏笔。")
    middle = NarrativeItem("middle", "人物走进房间，发生争执，最后离开。")
    weak = NarrativeItem("weak", "然后然后然后。没有原因。没有结尾。")
    pairs = [
        NarrativePair("p1", strong, middle),
        NarrativePair("p2", middle, weak),
        NarrativePair("p3", strong, weak),
    ]
    preferences = {pair.id: _preference(pair.id, 0.95) for pair in pairs}
    model = PairwiseTfidfRewardModel(max_features=2000).fit(pairs, preferences)

    assert model.predict_pair(strong.text, weak.text) > 0.5
    scores = model.score([strong.text, middle.text, weak.text])
    assert scores[0] > scores[2]
    assert set(model.grade([strong.text, middle.text, weak.text])) <= {"A", "B", "C", "D", "F"}

    path = tmp_path / "reward.pkl"
    model.save(path)
    loaded = PairwiseTfidfRewardModel.load(path)
    assert loaded.predict_pair(strong.text, weak.text) == model.predict_pair(strong.text, weak.text)
