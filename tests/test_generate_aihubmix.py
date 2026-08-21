from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_aihubmix", ROOT / "scripts" / "generate_aihubmix.py"
)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_normalize_base_url_adds_v1_to_documented_host() -> None:
    assert (
        generator.normalize_base_url("https://api.inferera.com/")
        == "https://api.inferera.com/v1"
    )


def test_normalize_base_url_rejects_unknown_endpoint() -> None:
    with pytest.raises(ValueError):
        generator.normalize_base_url("https://example.com/v1")


def test_preview_sends_prompt_but_never_human_text() -> None:
    rows = [
        {
            "human_id": "H001",
            "approved_prompt": "请创作一篇中文短篇故事。",
            "text": "这段人类原文绝对不能发给服务商。",
        }
    ]
    preview = generator.preview_rows(
        rows, "vendor/model", "https://api.inferera.com/v1", 1
    )
    assert preview[0]["prompt"] == rows[0]["approved_prompt"]
    assert rows[0]["text"] not in str(preview)
    assert "text" not in preview[0]


def test_default_output_path_sanitizes_model_id() -> None:
    assert generator.default_output_path("vendor/model name") == Path(
        "data/human_pilot/generated/aihubmix_vendor_model_name.jsonl"
    )


def test_environment_key_has_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIHUBMIX_API_KEY", "test-key")
    assert generator.load_api_key() == ("test-key", "environment")


def test_generation_qc_flags_length_and_truncation() -> None:
    assert generator.generation_qc("甲" * 1300, "length", 500, 1200) == {
        "text_char_count": 1300,
        "length_in_range": False,
        "protocol_flags": ["above_max_chars", "non_stop_finish"],
    }


def test_reasoning_effort_defaults_to_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["generate_aihubmix.py", "--model", "test-model"])
    assert generator.parse_args().reasoning_effort == "minimal"


def test_paid_or_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        generator.require_reviewed_free_model("gemini-3.6-flash")
    generator.require_reviewed_free_model("gemini-3.6-flash-free")


def test_free_quota_error_is_detected() -> None:
    assert generator.is_free_quota_exhausted(
        "Error code: 429 - reached the limit of the free model quota"
    )
    assert not generator.is_free_quota_exhausted("Error code: 429 - temporary rate limit")
