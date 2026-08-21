from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_pilot20_block", ROOT / "scripts" / "build_pilot20_block.py"
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_complete_block_is_blinded_and_position_balanced() -> None:
    humans = [
        {"human_id": f"H{i}", "text": "人" * 600, "genre": "现实"}
        for i in range(2)
    ]
    machines = []
    for i in range(2):
        for j, condition in enumerate(builder.REQUIRED_MACHINE_CONDITIONS):
            machines.append(
                {
                    "machine_id": f"M{i}_{j}",
                    "human_id": f"H{i}",
                    "status": "ok",
                    "generator_condition": condition,
                    "text": "机" * (700 + j),
                    "prompt": "Prompt",
                }
            )
    keyed, primary, all_pairs, bundle = builder.build_block(humans, machines, 2, 7)
    assert len(keyed) == 8
    assert len(primary) == 6
    assert len(all_pairs) == 12
    assert all(pair["item_a"]["source"] == "" for pair in all_pairs)
    assert all(pair["item_b"]["generator"] == "" for pair in all_pairs)
    positions = Counter()
    for row in bundle["pairs"]:
        if row["pilot_pair_type"] != "human_machine":
            continue
        condition = (
            row["generator_condition_b"]
            if row["source_type_a"] == "human"
            else row["generator_condition_a"]
        )
        positions[(condition, row["source_type_a"] == "human")] += 1
    assert set(positions.values()) == {1}
