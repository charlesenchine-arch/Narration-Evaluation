"""A lightweight Bradley--Terry reward-model baseline over character TF-IDF.

The model learns a shared utility function s(text).  Pair probabilities are
P(A>B)=sigmoid(s(A)-s(B)); public 0--100 scores are empirical percentiles of
the training-item utilities and are mapped to A/B/C/D/F grades.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .aggregation import AggregatedPreference
from .grades import DEFAULT_GRADE_SCALE, GradeBand, grade_for_score
from .schemas import NarrativeItem, NarrativePair


class PairwiseTfidfRewardModel:
    def __init__(self, ngram_range=(1, 3), max_features=50000, c=1.0):
        self.ngram_range = tuple(ngram_range)
        self.max_features = int(max_features)
        self.c = float(c)
        self.vectorizer = None
        self.model = None
        self.reference_utilities: np.ndarray | None = None

    def fit(
        self,
        pairs: Iterable[NarrativePair],
        preferences: Mapping[str, AggregatedPreference],
    ) -> "PairwiseTfidfRewardModel":
        from scipy.sparse import vstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        pairs = [pair for pair in pairs if pair.id in preferences]
        if not pairs:
            raise ValueError("no pairs have aggregated preference labels")
        items: dict[str, NarrativeItem] = {}
        for pair in pairs:
            items[pair.item_a.id] = pair.item_a
            items[pair.item_b.id] = pair.item_b
        ordered = list(items.values())
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=self.ngram_range, max_features=self.max_features
        )
        X = self.vectorizer.fit_transform([item.text for item in ordered])
        index = {item.id: i for i, item in enumerate(ordered)}

        rows, labels, weights = [], [], []
        for pair in pairs:
            p = float(preferences[pair.id].probability_a_better)
            confidence = max(1e-6, float(preferences[pair.id].effective_weight))
            diff = X[index[pair.item_a.id]] - X[index[pair.item_b.id]]
            # Symmetric soft-label expansion implements cross entropy for p.
            for row, label, weight in (
                (diff, 1, p * confidence),
                (diff, 0, (1.0 - p) * confidence),
                (-diff, 1, (1.0 - p) * confidence),
                (-diff, 0, p * confidence),
            ):
                if weight > 0:
                    rows.append(row)
                    labels.append(label)
                    weights.append(weight)
        self.model = LogisticRegression(
            C=self.c, fit_intercept=False, max_iter=2000, solver="liblinear"
        )
        self.model.fit(vstack(rows), np.asarray(labels), sample_weight=np.asarray(weights))
        self.reference_utilities = np.sort(self.raw_utility([item.text for item in ordered]))
        return self

    def _check_fitted(self) -> None:
        if self.vectorizer is None or self.model is None or self.reference_utilities is None:
            raise RuntimeError("reward model is not fitted")

    def raw_utility(self, texts: Iterable[str]) -> np.ndarray:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("reward model is not fitted")
        X = self.vectorizer.transform(list(texts))
        return np.asarray(X @ self.model.coef_.reshape(-1), dtype=np.float64).ravel()

    def score(self, texts: Iterable[str]) -> np.ndarray:
        """Return empirical-percentile utility scores in [0, 100]."""
        self._check_fitted()
        raw = self.raw_utility(texts)
        ref = self.reference_utilities
        return np.searchsorted(ref, raw, side="right") / len(ref) * 100.0

    def grade(
        self,
        texts: Iterable[str],
        scale: tuple[GradeBand, ...] = DEFAULT_GRADE_SCALE,
    ) -> list[str]:
        return [grade_for_score(float(value), scale).code for value in self.score(texts)]

    def predict_pair(self, text_a: str, text_b: str) -> float:
        self._check_fitted()
        X = self.vectorizer.transform([text_a, text_b])
        return float(self.model.predict_proba(X[0] - X[1])[0, 1])

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseTfidfRewardModel":
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain {cls.__name__}")
        return model
