"""Train/evaluate the pairwise TF-IDF Bradley--Terry baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from narrative_evaluator.quality.aggregation import aggregate_judgments
from narrative_evaluator.quality.evaluation import evaluate_pair_probabilities
from narrative_evaluator.quality.reward import PairwiseTfidfRewardModel
from narrative_evaluator.quality.schemas import NarrativePair, PairwiseJudgment, read_jsonl
from narrative_evaluator.quality.splitting import split_pairs


def _load_state(path: str) -> list[PairwiseJudgment]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for rater, judged in (state.get("judgments") or {}).items():
        for pair_id, obj in judged.items():
            rows.append(PairwiseJudgment.from_dict({
                **obj, "pair_id": pair_id, "rater_id": rater,
            }))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="训练成对偏好 reward 基线")
    parser.add_argument("--pairs", default="data/eval_dataset/pairwise/pairs.jsonl")
    parser.add_argument("--state", default="data/eval_dataset/pairwise/pairwise_state.json")
    parser.add_argument("--model-out", default="data/eval_dataset/models/quality_tfidf_bt.pkl")
    parser.add_argument("--report-out", default="data/eval_dataset/pairwise/baseline_report.json")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-stay-ms", type=int, default=8000)
    parser.add_argument(
        "--split-mode",
        choices=("group_disjoint", "generator_holdout", "pair_random"),
        default="group_disjoint",
    )
    parser.add_argument("--holdout-generator", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs, NarrativePair)
    judgments = _load_state(args.state)
    preferences = aggregate_judgments(judgments, minimum_stay_ms=args.minimum_stay_ms)
    labeled = [pair for pair in pairs if pair.id in preferences]
    split = split_pairs(
        labeled,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        mode=args.split_mode,
        holdout_generator=args.holdout_generator,
    )
    train, test = list(split.train), list(split.test)
    if not train:
        raise SystemExit("训练集为空；请先收集成对评价或调整留出设置")
    model = PairwiseTfidfRewardModel().fit(train, preferences)
    model.save(args.model_out)

    report = {
        "protocol": "pairwise-quality-v1",
        "model": "character-tfidf-bradley-terry",
        "n_judgments": len(judgments),
        "n_aggregated_pairs": len(preferences),
        "n_train_pairs": len(train),
        "n_test_pairs": len(test),
        "n_discarded_cross_split_pairs": len(split.discarded_cross_split),
        "split_mode": split.mode,
        "holdout_generator": args.holdout_generator,
    }
    if test:
        gold = [preferences[pair.id].probability_a_better for pair in test]
        predicted = [model.predict_pair(pair.item_a.text, pair.item_b.text) for pair in test]
        report["test"] = evaluate_pair_probabilities(gold, predicted)

    items = {}
    for pair in pairs:
        items[pair.item_a.id] = pair.item_a
        items[pair.item_b.id] = pair.item_b
    ordered = list(items.values())
    scores = model.score([item.text for item in ordered])
    grades = model.grade([item.text for item in ordered])
    report["item_predictions"] = [
        {"id": item.id, "score": round(float(score), 3), "grade": grade}
        for item, score, grade in zip(ordered, scores, grades)
    ]
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "item_predictions"},
                     ensure_ascii=False, indent=2))
    print(f"模型：{args.model_out}")
    print(f"报告：{args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
