from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_pilot20_source_probe",
    ROOT / "scripts" / "analyze_pilot20_source_probe.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_surface_features_are_finite_and_length_sensitive() -> None:
    short = probe.surface_features("甲。乙。")
    long = probe.surface_features("甲" * 100 + "。乙" * 20)
    assert len(short) == len(long) == 7
    assert long[0] > short[0]


def test_condition_summary_counts_length_compliance() -> None:
    report = probe.condition_summary(
        [
            {"generator_condition": "a", "char_count": 600},
            {"generator_condition": "a", "char_count": 1300},
        ]
    )
    assert report["a"]["n"] == 2
    assert report["a"]["length_500_1200"] == 1
