from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_human_pilot", ROOT / "scripts" / "build_human_pilot.py"
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_normalized_hash_ignores_whitespace_variation() -> None:
    assert builder.normalized_hash("甲 乙\n丙") == builder.normalized_hash("甲乙丙")


def test_coig_filter_rejects_outline_and_assistant_preamble(tmp_path: Path) -> None:
    rows = [
        {
            "id": "valid",
            "query": "请创作一个完整的科幻故事，围绕陌生星球上的求生展开。",
            "answer": "甲" * 500,
        },
        {
            "id": "outline",
            "query": "请帮我写一个游戏小说的世界设定和故事大纲。",
            "answer": "乙" * 500,
        },
        {
            "id": "preamble",
            "query": "请创作一个完整的科幻故事，围绕陌生星球上的求生展开。",
            "answer": "下面将详细描写故事。" + "丙" * 500,
        },
    ]
    path = tmp_path / "coig.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    result = builder.build_coig(path, 500, 1200)
    assert [row["source_record_id"] for row in result] == ["valid"]


def test_storal_full_text_and_prompt_do_not_use_outline(tmp_path: Path) -> None:
    row = {
        "id": "story-1",
        "moral": "遇事要冷静。",
        "outline": "秘密角色在结尾背叛主人公",
        "beginning": "开" * 250,
        "story": "终" * 260,
    }
    path = tmp_path / "storal.json"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    result = builder.build_storal([("test", path)], 500, 1200)
    assert len(result) == 1
    assert result[0]["text"] == "开" * 250 + "终" * 260
    assert "遇事要冷静" in result[0]["prompt_candidate"]
    assert "背叛主人公" not in result[0]["prompt_candidate"]


def test_deduplicate_keeps_first_record() -> None:
    first = {"text_sha256": "same", "source_record_id": "first"}
    second = {"text_sha256": "same", "source_record_id": "second"}
    kept, removed = builder.deduplicate([first, second])
    assert kept == [first]
    assert removed == 1


def test_neutralized_prompt_avoids_instruction_fragment() -> None:
    query = "写一段赛博朋克叙事，关注环境描写以及对生命与求生意志的刻画。"
    prompt = builder.coig_prompt(query, "科幻故事")
    assert "及对生命与求生意志的刻画" not in prompt
    assert "人物、冲突、转折与结局由你独立设计" in prompt
