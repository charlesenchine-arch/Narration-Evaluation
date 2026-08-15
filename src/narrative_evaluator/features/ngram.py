"""字符 n-gram 判别特征：TfidfVectorizer 封装。"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NgramFeaturizer:
    """字符级 n-gram TF-IDF 向量化器（轻量、CPU 可跑，中文无需分词）。"""

    def __init__(self, ngram_range=(1, 3), max_features=50000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(ngram_range),
            max_features=max_features,
        )
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts) -> np.ndarray:
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return X


def build_ngram_featurizer(ngram_range=(1, 3), max_features=50000) -> NgramFeaturizer:
    return NgramFeaturizer(ngram_range=ngram_range, max_features=max_features)
