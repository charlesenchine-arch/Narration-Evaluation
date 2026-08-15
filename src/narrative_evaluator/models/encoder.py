"""冻结中文编码器：把文本映射到表示空间。

支持两种粒度：
- SentenceEncoder：单句/短文本编码（直接过 sentence-transformers）。
- DocumentEncoder：任意长度文本，先切句再逐句编码，均值池化成文档向量。

均使用冻结模型（不训练），device 由配置指定（默认 cpu）。
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

# 惰性加载：避免 import 即下载模型
_sent_split = re.compile(r"[。！？!?；;…]+")


class SentenceEncoder:
    """句级编码器封装（sentence-transformers）。"""

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", device="cpu", max_len=512):
        self.model_name = model_name
        self.device = device
        self.max_len = max_len
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """返回 [n, d] 归一化句向量（L2 归一化）。"""
        if not texts:
            return np.zeros((0, 0))
        self._load()
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float64)


class DocumentEncoder:
    """长文本编码器：切句 → 句向量 → 均值池化（按句长加权可选）。"""

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", device="cpu",
                 max_len=512, weight_by_len=True, batch_size=32):
        self.sent_enc = SentenceEncoder(model_name, device, max_len)
        self.weight_by_len = weight_by_len
        self.batch_size = batch_size

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        parts = [p.strip() for p in _sent_split.split(text) if p.strip()]
        return parts if parts else ([text.strip()] if text.strip() else [])

    def encode_document(self, text: str) -> np.ndarray:
        """单篇文档 → 归一化文档向量（走批量路径，模型懒加载一次）。"""
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """批量文档 → [n, d] 归一化文档向量。

        把所有文档的句子扁平化后一次编码（模型前向仅一次），再按文档切回聚合，
        避免 N 次小 batch 前向。
        """
        if not texts:
            return np.zeros((0, 0))
        all_sents = [self.split_sentences(t) for t in texts]
        flat = [s for sents in all_sents for s in sents]
        dim = 0
        if flat:
            vecs = self.sent_enc.encode(flat, batch_size=self.batch_size)
            dim = vecs.shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float64)
        idx = 0
        for i, sents in enumerate(all_sents):
            if not sents:
                continue
            v = vecs[idx:idx + len(sents)]
            idx += len(sents)
            if self.weight_by_len:
                lens = np.array([len(s) for s in sents], dtype=np.float64)
                w = lens / (lens.sum() or 1.0)
                doc = (v * w[:, None]).sum(axis=0)
            else:
                doc = v.mean(axis=0)
            norm = np.linalg.norm(doc)
            out[i] = doc / norm if norm > 0 else doc
        return out
