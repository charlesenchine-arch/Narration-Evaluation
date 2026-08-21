#!/usr/bin/env python3
"""Run group-disjoint source-leakage probes on the 20-prompt pilot block."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def surface_features(text: str) -> list[float]:
    compact = re.sub(r"\s+", "", text)
    length = max(1, len(compact))
    sentences = [chunk for chunk in re.split(r"[。！？!?]+", compact) if chunk]
    paragraphs = [chunk for chunk in re.split(r"\n+", text) if chunk.strip()]
    punctuation = sum(char in "，。！？；：、—…,.!?;:" for char in compact)
    quote_marks = sum(char in "“”‘’\"" for char in compact)
    return [
        math.log1p(length),
        float(len(paragraphs)),
        float(len(sentences)),
        length / max(1, len(sentences)),
        punctuation / length,
        quote_marks / length,
        len(set(compact)) / length,
    ]


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def group_bootstrap_ci(
    groups: np.ndarray,
    metric: Callable[[np.ndarray], float],
    iterations: int,
    seed: int,
) -> list[float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        try:
            values.append(float(metric(indices)))
        except ValueError:
            continue
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def add_ci(
    metrics: dict[str, float],
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(metrics)
    result["balanced_accuracy_95ci"] = group_bootstrap_ci(
        groups,
        lambda idx: balanced_accuracy_score(y[idx], probability[idx] >= 0.5),
        iterations,
        seed,
    )
    result["roc_auc_95ci"] = group_bootstrap_ci(
        groups,
        lambda idx: roc_auc_score(y[idx], probability[idx]),
        iterations,
        seed + 1,
    )
    return result


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row["generator_condition"])].append(int(row["char_count"]))
    return {
        condition: {
            "n": len(lengths),
            "char_mean": float(np.mean(lengths)),
            "char_median": float(np.median(lengths)),
            "char_min": int(min(lengths)),
            "char_max": int(max(lengths)),
            "length_500_1200": sum(500 <= value <= 1200 for value in lengths),
        }
        for condition, lengths in sorted(grouped.items())
    }


def analyze(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    texts = [str(row["text"]) for row in rows]
    groups = np.asarray([str(row["human_id"]) for row in rows])
    y_binary = np.asarray([0 if row["source_type"] == "human" else 1 for row in rows])
    conditions = sorted({str(row["generator_condition"]) for row in rows})
    condition_to_index = {condition: index for index, condition in enumerate(conditions)}
    y_multi = np.asarray([condition_to_index[str(row["generator_condition"])] for row in rows])
    cv = GroupKFold(n_splits=5)

    char_model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=30000),
        LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear"),
    )
    char_probability = cross_val_predict(
        char_model, texts, y_binary, groups=groups, cv=cv, method="predict_proba"
    )[:, 1]

    lengths = np.asarray([[math.log1p(len(re.sub(r"\s+", "", text)))] for text in texts])
    length_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    length_probability = cross_val_predict(
        length_model, lengths, y_binary, groups=groups, cv=cv, method="predict_proba"
    )[:, 1]

    surface = np.asarray([surface_features(text) for text in texts])
    surface_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear"),
    )
    surface_probability = cross_val_predict(
        surface_model, surface, y_binary, groups=groups, cv=cv, method="predict_proba"
    )[:, 1]

    multi_model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=30000),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
    )
    multi_prediction = cross_val_predict(
        multi_model, texts, y_multi, groups=groups, cv=cv, method="predict"
    )

    result = {
        "protocol": "pilot20-source-probe-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(rows),
        "n_prompt_groups": len(set(groups)),
        "cv": "5-fold GroupKFold by prompt; no prompt crosses train/test",
        "binary_target": "human=0 versus any machine=1",
        "binary_class_counts": dict(sorted(Counter(y_binary.tolist()).items())),
        "binary_random_balanced_accuracy": 0.5,
        "char_tfidf_probe": add_ci(
            binary_metrics(y_binary, char_probability),
            y_binary,
            char_probability,
            groups,
            bootstrap,
            seed,
        ),
        "length_only_probe": add_ci(
            binary_metrics(y_binary, length_probability),
            y_binary,
            length_probability,
            groups,
            bootstrap,
            seed + 10,
        ),
        "surface_feature_probe": add_ci(
            binary_metrics(y_binary, surface_probability),
            y_binary,
            surface_probability,
            groups,
            bootstrap,
            seed + 20,
        ),
        "four_way_generator_probe": {
            "labels": conditions,
            "accuracy": float(accuracy_score(y_multi, multi_prediction)),
            "macro_f1": float(f1_score(y_multi, multi_prediction, average="macro")),
            "random_accuracy": 0.25,
        },
        "condition_descriptives": condition_summary(rows),
        "interpretation_guardrail": (
            "High source-probe performance shows identifiable source cues, not quality bias. "
            "Quality bias requires independent blind preference labels or a preregistered judge proxy."
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/human_pilot/pilot20/private/items_keyed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/human_pilot_20260821/pilot20_source_probe.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(read_jsonl(args.items), args.bootstrap, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
