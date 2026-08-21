"""测试：build_discriminator 工厂按 type / env 切换。"""
import os

import pytest

from narrative_evaluator.config import DiscriminatorConfig
from narrative_evaluator.models.discriminator import build_discriminator, NgramDiscriminator


def test_default_is_ngram():
    cfg = DiscriminatorConfig()
    disc = build_discriminator(cfg)
    assert isinstance(disc, NgramDiscriminator)


def test_type_macbert_returns_macbert():
    cfg = DiscriminatorConfig(type="macbert")
    disc = build_discriminator(cfg)
    # 不 import MacBertDiscriminator 顶层，用类名判断
    assert type(disc).__name__ == "MacBertDiscriminator"


def test_env_overrides_type():
    cfg = DiscriminatorConfig()  # type=ngram
    old = os.environ.get("NARRATIVE_DISC_TYPE")
    os.environ["NARRATIVE_DISC_TYPE"] = "macbert"
    try:
        disc = build_discriminator(cfg)
        assert type(disc).__name__ == "MacBertDiscriminator"
    finally:
        if old is None:
            os.environ.pop("NARRATIVE_DISC_TYPE", None)
        else:
            os.environ["NARRATIVE_DISC_TYPE"] = old


def test_macbert_path_default():
    cfg = DiscriminatorConfig(type="macbert")
    disc = build_discriminator(cfg)
    assert disc.model_path  # 非空默认路径


def test_macbert_require_trained_model_forwarded():
    cfg = DiscriminatorConfig(type="macbert", require_trained_model=True)
    disc = build_discriminator(cfg)
    assert disc.require_trained_model is True
