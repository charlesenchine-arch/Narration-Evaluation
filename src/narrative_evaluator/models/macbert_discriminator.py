"""MacBERT 全量微调判别器（人/机二分类）。

与 NgramDiscriminator 接口完全一致，MixScorer / Evaluator 与实验脚本可无缝切换：
- fit(h_texts, g_texts, length_match=False, n_buckets=8, seed=42,
      val_h_texts=None, val_g_texts=None, force_retrain=False)
- predict_human_prob(texts) -> np.ndarray [n]
- auc(h, g) / balanced_acc(h, g)
- used_n_h / used_n_g / length_match_used

与 ngram 的关键语义差异：
- 训练代价高（CPU 全量微调 80-90min），因此模型持久化到 model_path，
  fit 幂等——model_path 已有产物时 fit 直接返回（loaded=True），不会重训。
  这样实验脚本（A3/B1/解耦）加载部署判别器即可，不对几百条评分集重训。
- 惰性加载（同 encoder.py）：构造不下载不加载，首次 fit/predict 才 from_pretrained。
- length_match=True 复用了 discriminator.py 的 _length_match，长度不变性在训练阶段保证。
- 超过 max_len 的文本按 token 重叠滑窗，文档概率取全部窗口 P(human) 的均值，
  不再只读取开头 512 token。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import torch

from .discriminator import _length_match


class _SimpleDataset:
    """内存中的 token 化数据集。token 化在数据集构造时做一次（训练前）。"""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


def _collate(batch, max_len):
    """动态 padding 到本 batch 最大长度（省 CPU 算力）。"""
    input_ids = [b[0] for b in batch]
    attention_mask = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    L = max(len(x) for x in input_ids)
    ids = np.zeros((len(batch), L), dtype=np.int64)
    mask = np.zeros((len(batch), L), dtype=np.int64)
    for i, (x, m) in enumerate(zip(input_ids, attention_mask)):
        ids[i, :len(x)] = x
        mask[i, :len(m)] = m
    return (
        torch.from_numpy(ids),
        torch.from_numpy(mask),
        torch.tensor(labels, dtype=torch.long),
    )


class MacBertDiscriminator:
    def __init__(self, model_name="hfl/chinese-macbert-base", model_path="",
                 max_len=512, window_overlap=128, batch_size=16, device="cpu", cache_dir=None,
                 epochs=3, lr=2e-5, warmup_ratio=0.05, weight_decay=0.01,
                 seed=42, threads=0, require_trained_model=False,
                 _model=None, _tokenizer=None):
        self.model_name = model_name
        self.model_path = model_path or "data/eval_dataset/models/macbert_discriminator"
        self.max_len = max_len
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.epochs = epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.seed = seed
        self.threads = threads
        self.require_trained_model = require_trained_model
        # 测试注入极小模型/分词器用；否则 None，惰性加载
        self._model = _model
        self._tokenizer = _tokenizer
        self._fitted = False
        self.loaded = False
        self.length_match_used = False
        self.used_n_h = 0
        self.used_n_g = 0
        self._is_training = False

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ---- 加载 / 持久化 ----
    def _load(self):
        """从本地产物目录加载模型+分词器；无则加载预训练模型。"""
        if self._model is not None and self._tokenizer is not None:
            return
        local_model_exists = os.path.isdir(self.model_path) and os.path.exists(
            os.path.join(self.model_path, "config.json")
        )
        if self.require_trained_model and not local_model_exists:
            raise FileNotFoundError(
                f"未找到已训练的 MacBERT 判别器：{self.model_path}\n"
                "部署配置不会自动使用未微调的基础模型。请先运行：\n"
                "PYTHONPATH=src python scripts/train_macbert_discriminator.py "
                "--config configs/default_macbert.yaml"
            )
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._model is None:
            if local_model_exists:
                # 本地产物：默认加载（优先 safetensors，由训练 save_pretrained 生成）
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, cache_dir=self.cache_dir)
                self.loaded = True
                self._fitted = True
            else:
                # 远程：chinese-macbert-base 只有 pytorch_model.bin 且禁用 discussions，
                # transformers 5.x 会自动找转换 PR → 403（后台线程，非阻塞噪音）。
                # use_safetensors=False 避免加载侧要求 safetensors。
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=2, cache_dir=self.cache_dir,
                    use_safetensors=False)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path if self.loaded else self.model_name,
                cache_dir=self.cache_dir)
        self.device = self._resolve_device(self.device)
        self._model.to(self.device)

    def _save(self):
        os.makedirs(self.model_path, exist_ok=True)
        self._model.save_pretrained(self.model_path)
        self._tokenizer.save_pretrained(self.model_path)
        meta = {
            "model_name": self.model_name,
            "used_n_h": self.used_n_h,
            "used_n_g": self.used_n_g,
            "length_match_used": self.length_match_used,
        }
        with open(os.path.join(self.model_path, "train_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 训练 ----
    def fit(self, h_texts, g_texts, length_match=False, n_buckets=8, seed=42,
            val_h_texts=None, val_g_texts=None, force_retrain=False):
        """训练（或加载已有模型）。

        与 NgramDiscriminator 相同签名。若 model_path 已有训练产物且未 force_retrain，
        直接标记已 fit（loaded=True）返回，不重训。
        """
        # 线程数：Windows 上 PyTorch 默认可能不充分利用多核，显式设置
        if self.threads and self.threads > 0:
            torch.set_num_threads(self.threads)

        h_texts = list(h_texts)
        g_texts = list(g_texts)
        self.used_n_h = len(h_texts)
        self.used_n_g = len(g_texts)
        if length_match and h_texts and g_texts:
            h_texts, g_texts = _length_match(h_texts, g_texts, n_buckets, seed)
            self.length_match_used = True
            self.used_n_h = len(h_texts)
            self.used_n_g = len(g_texts)

        self._load()
        if self.loaded and not force_retrain:
            return self  # 已加载部署判别器，跳过训练

        # 验证集：未显式给则内部分层 85/15 切分（保证 H/G 都覆盖）
        if val_h_texts is None or val_g_texts is None:
            h_train, h_val = self._split(h_texts, seed)
            g_train, g_val = self._split(g_texts, seed + 1)
        else:
            h_train, h_val = h_texts, list(val_h_texts)
            g_train, g_val = g_texts, list(val_g_texts)

        self._is_training = True
        try:
            self._train_loop(h_train, g_train, h_val, g_val)
        finally:
            self._is_training = False
        self._fitted = True
        return self

    @staticmethod
    def _split(texts, seed):
        rng = np.random.RandomState(seed)
        n = len(texts)
        perm = rng.permutation(n)
        cut = int(n * 0.85)
        return ([texts[i] for i in perm[:cut]], [texts[i] for i in perm[cut:]])

    def _tokenize(self, texts: List[str]):
        return self._tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

    def _tokenize_windows(self, texts: List[str]):
        """把长文切成重叠 token 窗口，返回 token 结果及窗口到原文的映射。"""
        texts = list(texts)
        if not texts:
            return {"input_ids": [], "attention_mask": []}, []
        # transformers 的 stride 表示相邻窗口重复量；限制到半窗以内，避免无效配置。
        overlap = min(max(0, int(self.window_overlap)), max(0, self.max_len // 2))
        tok = self._tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
            return_overflowing_tokens=True,
            stride=overlap,
        )
        mapping = tok.get("overflow_to_sample_mapping")
        if mapping is None:
            # Fast tokenizer会提供映射；无溢出且未返回时是一篇对应一个窗口。
            if len(tok["input_ids"]) != len(texts):
                raise RuntimeError("tokenizer 未返回 overflow_to_sample_mapping，无法聚合长文窗口")
            mapping = list(range(len(texts)))
        return tok, [int(i) for i in mapping]

    @staticmethod
    def _aggregate_window_probs(window_probs, mapping, n_texts: int) -> np.ndarray:
        """按原文索引平均多个窗口概率。"""
        window_probs = np.asarray(window_probs, dtype=np.float64)
        mapping = np.asarray(mapping, dtype=np.int64)
        if window_probs.size != mapping.size:
            raise ValueError("窗口概率数量与映射数量不一致")
        sums = np.zeros(n_texts, dtype=np.float64)
        counts = np.zeros(n_texts, dtype=np.int64)
        np.add.at(sums, mapping, window_probs)
        np.add.at(counts, mapping, 1)
        if np.any(counts == 0):
            raise RuntimeError("部分文本未生成 MacBERT token 窗口")
        return sums / counts

    def _train_loop(self, h_train, g_train, h_val, g_val):
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        rng = np.random.RandomState(self.seed)
        texts = h_train + g_train
        labels = np.array([1] * len(h_train) + [0] * len(g_train))
        # 混洗
        perm = rng.permutation(len(texts))
        texts = [texts[i] for i in perm]
        labels = labels[perm]

        tok, train_mapping = self._tokenize_windows(texts)
        train_ds = _SimpleDataset(
            [list(x) for x in tok["input_ids"]],
            [list(x) for x in tok["attention_mask"]],
            [int(labels[i]) for i in train_mapping],
        )
        val_texts = h_val + g_val
        val_labels = np.array([1] * len(h_val) + [0] * len(g_val))
        val_tok, val_mapping = self._tokenize_windows(val_texts)
        val_ds = _SimpleDataset(
            [list(x) for x in val_tok["input_ids"]],
            [list(x) for x in val_tok["attention_mask"]],
            [int(val_labels[i]) for i in val_mapping],
        )

        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              collate_fn=lambda b: _collate(b, self.max_len))
        val_dl = DataLoader(val_ds, batch_size=32,
                            collate_fn=lambda b: _collate(b, self.max_len))

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        total_steps = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        self._model.train()
        best_auc = -1.0
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in train_dl:
                ids, mask, lab = [b.to(self.device) for b in batch]
                out = self._model(input_ids=ids, attention_mask=mask, labels=lab)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            val_auc, val_bacc = self._evaluate(val_dl)
            print(f"  [epoch {epoch+1}/{self.epochs}] loss={total_loss/len(train_dl):.4f} "
                  f"val_auc={val_auc:.4f} val_bacc={val_bacc:.4f}", flush=True)
            if val_auc > best_auc:
                best_auc = val_auc
                self._save()
                print(f"    → 保存最优模型 (AUC {val_auc:.4f})", flush=True)
        print(f"  训练完成，最优验证 AUC = {best_auc:.4f}", flush=True)

    def _evaluate(self, dl):
        self._model.eval()
        with torch.no_grad():
            ps, ys = [], []
            for batch in dl:
                ids, mask, lab = [b.to(self.device) for b in batch]
                logits = self._model(input_ids=ids, attention_mask=mask).logits
                prob = torch.softmax(logits, dim=-1)[:, 1]
                ps.extend(prob.cpu().numpy().tolist())
                ys.extend(lab.cpu().numpy().tolist())
        self._model.train()
        return self._auc_from(np.array(ps), np.array(ys)), self._bacc_from(np.array(ps), np.array(ys))

    @staticmethod
    def _auc_from(p, y):
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(y, p))
        except ValueError:
            return 0.5

    @staticmethod
    def _bacc_from(p, y):
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y, p > 0.5))

    # ---- 推理 ----
    def predict_human_prob(self, texts) -> np.ndarray:
        self._load()  # 可能从 model_path 加载并置 _fitted
        if not self._fitted:
            raise RuntimeError("discriminator not fitted")
        from torch.utils.data import DataLoader
        texts = list(texts)
        if not texts:
            return np.zeros((0,), dtype=np.float64)
        tok, mapping = self._tokenize_windows(texts)
        ds = _SimpleDataset(
            [list(x) for x in tok["input_ids"]],
            [list(x) for x in tok["attention_mask"]],
            [0] * len(mapping),
        )
        dl = DataLoader(ds, batch_size=32, collate_fn=lambda b: _collate(b, self.max_len))
        self._model.eval()
        ps = []
        with torch.no_grad():
            for batch in dl:
                ids, mask, _ = [b.to(self.device) for b in batch]
                logits = self._model(input_ids=ids, attention_mask=mask).logits
                prob = torch.softmax(logits, dim=-1)[:, 1]
                ps.extend(prob.cpu().numpy().tolist())
        self._model.train()
        # 一个文档可能对应多个窗口；取窗口人类概率均值作为整篇分数。
        return self._aggregate_window_probs(ps, mapping, len(texts))

    def auc(self, h_texts, g_texts) -> float:
        texts = list(h_texts) + list(g_texts)
        y = np.array([1] * len(h_texts) + [0] * len(g_texts))
        p = self.predict_human_prob(texts)
        return self._auc_from(p, y)

    def balanced_acc(self, h_texts, g_texts) -> float:
        texts = list(h_texts) + list(g_texts)
        y = np.array([1] * len(h_texts) + [0] * len(g_texts))
        p = self.predict_human_prob(texts)
        return self._bacc_from(p, y)
