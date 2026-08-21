from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_manual_prompt_pack", ROOT / "scripts" / "export_manual_prompt_pack.py"
)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_export_excludes_human_text_and_marks_pending(tmp_path: Path) -> None:
    input_path = tmp_path / "review.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "human_id",
                "text",
                "approved_prompt",
                "prompt_candidate",
                "approve_text",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "human_id": "H001",
                "text": "不得导出的人类原文",
                "approved_prompt": "请创作一篇故事。",
                "prompt_candidate": "候选 Prompt",
                "approve_text": "",
            }
        )
    prompts = exporter.read_prompts(input_path)
    assert prompts == [
        {
            "human_id": "H001",
            "approved_prompt": "请创作一篇故事。",
            "human_text_review_status": "pending",
        }
    ]
    assert "不得导出的人类原文" not in str(prompts)


def test_export_marks_explicit_yes_as_approved(tmp_path: Path) -> None:
    input_path = tmp_path / "review.csv"
    input_path.write_text(
        "human_id,approved_prompt,prompt_candidate,approve_text\n"
        "H002,最终 Prompt,候选 Prompt,yes\n",
        encoding="utf-8",
    )
    assert exporter.read_prompts(input_path)[0]["human_text_review_status"] == "approved"
