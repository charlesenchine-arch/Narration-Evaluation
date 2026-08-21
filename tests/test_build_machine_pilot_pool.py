from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_machine_pilot_pool", ROOT / "scripts" / "build_machine_pilot_pool.py"
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_combine_keeps_successes_and_reports_prompt_coverage(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_rows(
        first,
        [
            {
                "machine_id": "M1",
                "human_id": "H1",
                "status": "ok",
                "provider": "aihubmix",
                "requested_model": "model-free",
                "reasoning_effort": "minimal",
                "length_in_range": True,
            },
            {"machine_id": "ERR", "human_id": "H2", "status": "api_error"},
        ],
    )
    write_rows(
        second,
        [
            {
                "machine_id": "M2",
                "human_id": "H1",
                "status": "ok",
                "provider": "deepseek",
                "length_in_range": False,
            }
        ],
    )
    rows, manifest = builder.combine([first, second])
    assert [row["machine_id"] for row in rows] == ["M1", "M2"]
    assert manifest["total_successful_samples"] == 2
    assert manifest["unique_prompts"] == 1
    assert manifest["samples_per_prompt_distribution"] == {2: 1}
    assert manifest["length_compliant_samples"] == 1
