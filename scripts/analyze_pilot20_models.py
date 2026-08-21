#!/usr/bin/env python3
"""Combine source probes, local reward scores, and local Qwen blind judgments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile_ci(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["human_id"])].append(row)
    prompts = sorted(by_prompt)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        picked = rng.choice(prompts, size=len(prompts), replace=True)
        sample_rows = [row for prompt in picked for row in by_prompt[str(prompt)]]
        samples.append(float(np.mean([value(row) for row in sample_rows])))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        result[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return result


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
    }


def preference_summary(
    rows: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    win_values = [float(row["human_outcome"]) for row in rows]
    deltas = [float(row["human_delta"]) for row in rows]
    return {
        "n": len(rows),
        "human_wins": sum(value == 1.0 for value in win_values),
        "ties": sum(value == 0.5 for value in win_values),
        "machine_wins": sum(value == 0.0 for value in win_values),
        "human_win_equivalent": float(np.mean(win_values)),
        "human_win_equivalent_95ci": percentile_ci(
            rows,
            lambda row: float(row["human_outcome"]),
            iterations=iterations,
            seed=seed,
        ),
        "mean_human_minus_machine": float(np.mean(deltas)),
        "mean_human_minus_machine_95ci": percentile_ci(
            rows,
            lambda row: float(row["human_delta"]),
            iterations=iterations,
            seed=seed + 1,
        ),
    }


def length_adjusted_delta(
    rows: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    """Exploratory OLS of score delta on human-minus-machine character delta."""

    def fit(sample: list[dict[str, Any]]) -> tuple[float, float]:
        x = np.asarray([float(row["length_delta"]) for row in sample])
        y = np.asarray([float(row["human_delta"]) for row in sample])
        design = np.column_stack([np.ones(len(x)), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(intercept), float(slope * 100.0)

    intercept, slope_100 = fit(rows)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["human_id"])].append(row)
    prompts = sorted(by_prompt)
    rng = np.random.default_rng(seed)
    bootstrap_intercepts: list[float] = []
    bootstrap_slopes: list[float] = []
    for _ in range(iterations):
        picked = rng.choice(prompts, size=len(prompts), replace=True)
        sample = [row for prompt in picked for row in by_prompt[str(prompt)]]
        fitted_intercept, fitted_slope = fit(sample)
        bootstrap_intercepts.append(fitted_intercept)
        bootstrap_slopes.append(fitted_slope)
    return {
        "exploratory_method": "OLS: human-minus-machine score ~ human-minus-machine characters",
        "predicted_delta_at_equal_length": intercept,
        "predicted_delta_at_equal_length_95ci": [
            float(np.quantile(bootstrap_intercepts, 0.025)),
            float(np.quantile(bootstrap_intercepts, 0.975)),
        ],
        "score_delta_per_100_extra_human_chars": slope_100,
        "slope_95ci": [
            float(np.quantile(bootstrap_slopes, 0.025)),
            float(np.quantile(bootstrap_slopes, 0.975)),
        ],
        "guardrail": "Exploratory covariate adjustment is not a substitute for length-matched generation.",
    }


def pair_rows(
    pair_key: list[dict[str, Any]],
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pair_key:
        if pair.get("pilot_pair_type") != "human_machine":
            continue
        a, b = str(pair["blind_id_a"]), str(pair["blind_id_b"])
        if a not in scores or b not in scores:
            continue
        human_a = pair["source_type_a"] == "human"
        human_score = scores[a] if human_a else scores[b]
        machine_score = scores[b] if human_a else scores[a]
        delta = float(human_score - machine_score)
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "human_id": str(pair["human_id"]),
                "generator": str(
                    pair["generator_condition_b"]
                    if human_a
                    else pair["generator_condition_a"]
                ),
                "human_delta": delta,
                "human_outcome": 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5,
                "length_delta": int(
                    (pair["char_count_a"] if human_a else pair["char_count_b"])
                    - (pair["char_count_b"] if human_a else pair["char_count_a"])
                ),
            }
        )
    return rows


def qwen_pair_rows(
    pair_key: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    reversed_orientation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    key = {str(row["pair_id"]): row for row in pair_key}
    rows: list[dict[str, Any]] = []
    item_scores: dict[str, list[float]] = defaultdict(list)
    for judgment in judgments:
        if judgment.get("status") != "ok" or str(judgment["pair_id"]) not in key:
            continue
        pair = key[str(judgment["pair_id"])]
        result = judgment["result"]
        display_pref = int(result["preference"])
        pref = display_pref
        score_a, score_b = float(result["score_a"]), float(result["score_b"])
        if reversed_orientation:
            pref = -pref
            score_a, score_b = score_b, score_a
        human_a = pair["source_type_a"] == "human"
        human_score = score_a if human_a else score_b
        machine_score = score_b if human_a else score_a
        human_pref = pref if human_a else -pref
        item_scores[str(pair["blind_id_a"])].append(score_a)
        item_scores[str(pair["blind_id_b"])].append(score_b)
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "human_id": str(pair["human_id"]),
                "generator": str(
                    pair["generator_condition_b"]
                    if human_a
                    else pair["generator_condition_a"]
                ),
                "human_delta": human_score - machine_score,
                "human_outcome": 1.0 if human_pref > 0 else 0.0 if human_pref < 0 else 0.5,
                "length_delta": int(
                    (pair["char_count_a"] if human_a else pair["char_count_b"])
                    - (pair["char_count_b"] if human_a else pair["char_count_a"])
                ),
                "a_selected": pref > 0,
                "b_selected": pref < 0,
                "display_a_selected": display_pref > 0,
                "display_b_selected": display_pref < 0,
                "normalized": bool(result.get("normalizations")),
            }
        )
    return rows, item_scores


def grouped_summaries(
    rows: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["generator"])].append(row)
    return {
        "overall": preference_summary(rows, iterations=iterations, seed=seed),
        "by_generator": {
            generator: preference_summary(
                subset, iterations=iterations, seed=seed + index * 10
            )
            for index, (generator, subset) in enumerate(sorted(grouped.items()), 1)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/items_keyed.jsonl"),
    )
    parser.add_argument(
        "--pair-key",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/pair_key.jsonl"),
    )
    parser.add_argument(
        "--reward",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/local_reward_scores.jsonl"),
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/local_qwen_judgments.jsonl"),
    )
    parser.add_argument(
        "--qwen-reversed",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/local_qwen_reversed_audit.jsonl"),
    )
    parser.add_argument(
        "--source-probe",
        type=Path,
        default=Path("outputs/human_pilot_20260821/pilot20_source_probe.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/human_pilot_20260821/pilot20_combined_analysis.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = read_jsonl(args.items)
    item_by_id = {str(row["blind_id"]): row for row in items}
    pair_key = read_jsonl(args.pair_key)
    reward_rows = [row for row in read_jsonl(args.reward) if row.get("status") == "ok"]
    reward_scores = {str(row["blind_id"]): float(row["score"]) for row in reward_rows}
    reward_pairs = pair_rows(pair_key, reward_scores)
    qwen_raw = read_jsonl(args.qwen)
    qwen_pairs, qwen_item_scores = qwen_pair_rows(pair_key, qwen_raw)
    qwen_reversed_raw = read_jsonl(args.qwen_reversed) if args.qwen_reversed.exists() else []
    qwen_reversed_pairs, _ = qwen_pair_rows(
        pair_key, qwen_reversed_raw, reversed_orientation=True
    )

    reward_lengths = [float(item_by_id[item]["char_count"]) for item in reward_scores]
    reward_values = [reward_scores[item] for item in reward_scores]
    qwen_ids = sorted(qwen_item_scores)
    qwen_lengths = [float(item_by_id[item]["char_count"]) for item in qwen_ids]
    qwen_values = [float(np.mean(qwen_item_scores[item])) for item in qwen_ids]
    reward_direction = {row["pair_id"]: np.sign(row["human_delta"]) for row in reward_pairs}
    qwen_direction = {row["pair_id"]: np.sign(row["human_delta"]) for row in qwen_pairs}
    common = sorted(set(reward_direction) & set(qwen_direction))
    comparable = [pair for pair in common if reward_direction[pair] and qwen_direction[pair]]

    close_pairs = [row for row in qwen_pairs if abs(row["length_delta"]) <= 250]
    original_qwen_direction = {
        row["pair_id"]: np.sign(row["human_delta"]) for row in qwen_pairs
    }
    reversed_qwen_direction = {
        row["pair_id"]: np.sign(row["human_delta"]) for row in qwen_reversed_pairs
    }
    audit_common = sorted(set(original_qwen_direction) & set(reversed_qwen_direction))
    original_audit_rows = {row["pair_id"]: row for row in qwen_pairs}
    reversed_audit_rows = {row["pair_id"]: row for row in qwen_reversed_pairs}
    averaged_audit_rows: list[dict[str, Any]] = []
    for pair_id in audit_common:
        original_row = original_audit_rows[pair_id]
        reversed_row = reversed_audit_rows[pair_id]
        delta = (original_row["human_delta"] + reversed_row["human_delta"]) / 2.0
        averaged_audit_rows.append(
            {
                **original_row,
                "human_delta": delta,
                "human_outcome": 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5,
            }
        )
    result = {
        "protocol": "pilot20-local-model-analysis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "prompts": len({str(row["human_id"]) for row in items}),
            "items": len(items),
            "human_machine_pairs": sum(
                row.get("pilot_pair_type") == "human_machine" for row in pair_key
            ),
            "human_labels_collected": 0,
        },
        "source_probe": json.loads(args.source_probe.read_text(encoding="utf-8")),
        "reward_model": {
            "model": reward_rows[0]["model"] if reward_rows else "",
            "n_scored": len(reward_rows),
            "pairwise": grouped_summaries(
                reward_pairs, iterations=args.bootstrap, seed=args.seed
            ),
            "score_length_correlation": correlation(reward_lengths, reward_values),
            "pairwise_length_adjustment": length_adjusted_delta(
                reward_pairs, iterations=args.bootstrap, seed=args.seed + 50
            ),
            "truncated_items": sum(bool(row.get("truncated")) for row in reward_rows),
        },
        "qwen_judge": {
            "model": next(
                (row.get("model", "") for row in qwen_raw if row.get("status") == "ok"),
                "",
            ),
            "successful_pairs": len(qwen_pairs),
            "failed_rows": sum(row.get("status") != "ok" for row in qwen_raw),
            "normalized_preference_rows": sum(row["normalized"] for row in qwen_pairs),
            "pairwise": grouped_summaries(
                qwen_pairs, iterations=args.bootstrap, seed=args.seed + 100
            ),
            "score_length_correlation": correlation(qwen_lengths, qwen_values),
            "pairwise_length_adjustment": length_adjusted_delta(
                qwen_pairs, iterations=args.bootstrap, seed=args.seed + 150
            ),
            "position": {
                "a_selected": sum(row["a_selected"] for row in qwen_pairs),
                "b_selected": sum(row["b_selected"] for row in qwen_pairs),
                "ties": sum(not row["a_selected"] and not row["b_selected"] for row in qwen_pairs),
            },
            "length_close_subset_abs_delta_le_250": (
                preference_summary(
                    close_pairs, iterations=args.bootstrap, seed=args.seed + 200
                )
                if close_pairs
                else {"n": 0}
            ),
            "reversed_position_audit": {
                "n": len(qwen_reversed_pairs),
                "directional_consistency": (
                    float(
                        np.mean(
                            [
                                original_qwen_direction[pair]
                                == reversed_qwen_direction[pair]
                                for pair in audit_common
                            ]
                        )
                    )
                    if audit_common
                    else None
                ),
                "display_a_selected": sum(
                    row["display_a_selected"] for row in qwen_reversed_pairs
                ),
                "display_b_selected": sum(
                    row["display_b_selected"] for row in qwen_reversed_pairs
                ),
                "ties": sum(
                    not row["display_a_selected"] and not row["display_b_selected"]
                    for row in qwen_reversed_pairs
                ),
                "position_averaged_score_preference": (
                    preference_summary(
                        averaged_audit_rows,
                        iterations=args.bootstrap,
                        seed=args.seed + 300,
                    )
                    if averaged_audit_rows
                    else {"n": 0}
                ),
            },
        },
        "cross_model": {
            "comparable_non_tie_pairs": len(comparable),
            "directional_agreement": (
                float(
                    np.mean(
                        [reward_direction[pair] == qwen_direction[pair] for pair in comparable]
                    )
                )
                if comparable
                else None
            ),
        },
        "interpretation_guardrails": [
            "The source probe measures source identifiability, not narrative quality.",
            "The reward model and Qwen judge are proxy baselines, not human ground truth.",
            "The human references were pooled from an existing dataset and remain provisionally audited.",
            "Causal source-bias claims require blinded human judgments on quality-overlapping pairs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
