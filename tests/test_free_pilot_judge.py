from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_aihubmix_free_pilot_judge",
    ROOT / "scripts" / "run_aihubmix_free_pilot_judge.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_only_explicit_free_judges_are_allowed() -> None:
    runner.require_free_judge("gpt-5.5-free")
    runner.require_free_judge("qwen3.6-plus-preview-free")
    with pytest.raises(ValueError):
        runner.require_free_judge("gpt-5.5")
    with pytest.raises(ValueError):
        runner.require_free_judge("auto")
