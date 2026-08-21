from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "import_manual_deepseek_markdown",
    ROOT / "scripts" / "import_manual_deepseek_markdown.py",
)
assert SPEC and SPEC.loader
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


def test_parser_reads_only_filled_sections_and_preserves_inner_heading(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pack.md"
    source.write_text(
        "## 01. H001\n\n```text\nPrompt 1\n```\n\n## 故事标题\n正文一\n\n"
        "模型：___\n\n## 02. H002\n\n```text\nPrompt 2\n```\n\n"
        "模型：___\n",
        encoding="utf-8",
    )
    assert importer.parse_filled_sections(source) == [
        {"human_id": "H001", "prompt": "Prompt 1", "text": "## 故事标题\n正文一"}
    ]


def test_build_records_rejects_changed_prompt(tmp_path: Path) -> None:
    source = tmp_path / "pack.md"
    source.write_text("fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="Prompt differs"):
        importer.build_records(
            [{"human_id": "H001", "prompt": "changed", "text": "甲" * 600}],
            {"H001": "original"},
            source,
            500,
            1200,
        )


def test_manual_metadata_is_honest_about_unknown_model(tmp_path: Path) -> None:
    source = tmp_path / "pack.md"
    source.write_text("fixture", encoding="utf-8")
    row = importer.build_records(
        [{"human_id": "H001", "prompt": "Prompt", "text": "甲" * 600}],
        {"H001": "Prompt"},
        source,
        500,
        1200,
    )[0]
    assert row["requested_model"] == "deepseek_ui_unspecified"
    assert row["returned_model"] is None
    assert row["reasoning_effort"] == "deep_thinking_disabled"
    assert row["web_search_enabled"] is False
