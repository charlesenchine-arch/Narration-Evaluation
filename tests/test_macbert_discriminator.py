"""测试：MacBertDiscriminator（用极小真实模型注入，不下载、不训练大模型）。"""
import os
import tempfile

import numpy as np
import pytest

from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from narrative_evaluator.models.macbert_discriminator import MacBertDiscriminator


def _tiny_model_path(tmp):
    """构造极小真实模型 + tokenizer 到临时目录，返回 (dir, model, tokenizer)。"""
    cfg = BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=2)
    model = BertForSequenceClassification(cfg)
    vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n人\n工\n机\n器\n本\n文\n测\n试\n写\n作\n的\n样\n品\n"
    vocab_path = os.path.join(tmp, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(vocab)
    tokenizer = BertTokenizerFast(vocab_file=vocab_path)
    return model, tokenizer


def _make_disc(tmp, model, tokenizer):
    disc = MacBertDiscriminator(
        model_path=os.path.join(tmp, "model_out"),
        max_len=128, batch_size=2, epochs=1, lr=3e-4,
        _model=model, _tokenizer=tokenizer,
    )
    return disc


def test_predict_shape_and_range():
    with tempfile.TemporaryDirectory() as tmp:
        model, tok = _tiny_model_path(tmp)
        disc = _make_disc(tmp, model, tok)
        # 直接注入 _fitted，测 predict
        disc._fitted = True
        h = ["这是人类写的文本样本。", "这是另一段人类文本内容。"]
        p = disc.predict_human_prob(h)
        assert p.shape == (2,)
        assert all(0.0 <= x <= 1.0 for x in p)


def test_deploy_mode_fails_fast_without_trained_weights():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "missing_model")
        disc = MacBertDiscriminator(
            model_path=missing,
            require_trained_model=True,
        )
        with pytest.raises(FileNotFoundError, match="未找到已训练的 MacBERT"):
            disc._load()


def test_long_text_tokenization_creates_multiple_windows():
    with tempfile.TemporaryDirectory() as tmp:
        model, tok = _tiny_model_path(tmp)
        disc = MacBertDiscriminator(
            model_path=os.path.join(tmp, "model_out"),
            max_len=8,
            window_overlap=2,
            _model=model,
            _tokenizer=tok,
        )
        encoded, mapping = disc._tokenize_windows(["人工机器文本测试写作样品" * 4])
        assert len(encoded["input_ids"]) > 1
        assert mapping == [0] * len(mapping)


def test_window_probabilities_are_aggregated_per_document():
    result = MacBertDiscriminator._aggregate_window_probs(
        [0.2, 0.4, 0.9], [0, 0, 1], 2
    )
    assert result == pytest.approx([0.3, 0.9])


def test_fit_roundtrip_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        model, tok = _tiny_model_path(tmp)
        disc = _make_disc(tmp, model, tok)
        h = ["这是人类写的文本样本。", "这是另一段人类文本内容。"] * 4
        g = ["这是机器生成的文本内容，用于测试目的。"] * 8
        disc.fit(h, g)
        assert disc._fitted
        # 产物已保存
        assert os.path.isdir(disc.model_path)
        assert any(f.endswith(".safetensors") or f.endswith(".bin")
                   for f in os.listdir(disc.model_path))
        # 新实例从 model_path 加载可 predict
        disc2 = MacBertDiscriminator(model_path=disc.model_path)
        p2 = disc2.predict_human_prob(h[:3])
        assert p2.shape == (3,)


def test_fit_idempotent_when_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        model, tok = _tiny_model_path(tmp)
        disc = _make_disc(tmp, model, tok)
        h = ["这是人类写的文本样本。", "这是另一段人类文本内容。"] * 4
        g = ["这是机器生成的文本内容，用于测试目的。"] * 8
        disc.fit(h, g)
        used_h, used_g = disc.used_n_h, disc.used_n_g
        # 已有产物 + 不 force_retrain → fit 直接返回，不重训
        disc2 = MacBertDiscriminator(model_path=disc.model_path)
        disc2.fit(["新人类文本。"], ["新机器文本。"])
        assert disc2.loaded
        assert not disc2.length_match_used  # 不重采样
        assert disc2._fitted


def test_auc_balanced_acc_hand():
    with tempfile.TemporaryDirectory() as tmp:
        model, tok = _tiny_model_path(tmp)
        disc = _make_disc(tmp, model, tok)
        h = ["这是人类写的文本样本。", "这是另一段人类文本内容。"] * 4
        g = ["这是机器生成的文本内容，用于测试目的。"] * 8
        disc.fit(h, g)
        # 手算 auc/balanced_acc 与接口一致
        p_h = disc.predict_human_prob(h)
        p_g = disc.predict_human_prob(g)
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        y = np.array([1] * len(h) + [0] * len(g))
        p = np.concatenate([p_h, p_g])
        assert disc.auc(h, g) == pytest.approx(float(roc_auc_score(y, p)), abs=1e-6)
        assert disc.balanced_acc(h, g) == pytest.approx(
            float(balanced_accuracy_score(y, p > 0.5)), abs=1e-6)
