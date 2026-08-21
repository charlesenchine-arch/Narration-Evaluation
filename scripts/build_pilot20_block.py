#!/usr/bin/env python3
"""Build a blinded 20-prompt complete block for the first falsification pilot.

Each selected prompt must have one human narrative and successful outputs from
Gemini 3.5, Gemini 3.6, and the manually captured DeepSeek condition. Raw source
labels are written only to the private key; the rating pairs contain anonymous
ids and text only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MACHINE_CONDITIONS = (
    "aihubmix__gemini-3.5-flash-lite-free__reasoning_minimal",
    "aihubmix__gemini-3.6-flash-free__reasoning_minimal",
    "deepseek_ui_unspecified__thinking_off__search_off",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token(prefix: str, seed: int, value: str, length: int = 12) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def blind_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["blind_id"],
        "text": item["text"],
        "source": "",
        "work_id": item["blind_prompt_id"],
        "prompt_id": item["blind_prompt_id"],
        "generator": "",
        "genre": item.get("genre", ""),
        "meta": {"char_count": len(item["text"])},
    }


def build_block(
    human_rows: list[dict[str, Any]],
    machine_rows: list[dict[str, Any]],
    n_prompts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    humans = {str(row["human_id"]): row for row in human_rows}
    machines: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in machine_rows:
        if row.get("status") != "ok":
            continue
        condition = str(row.get("generator_condition") or "")
        if condition in REQUIRED_MACHINE_CONDITIONS:
            machines[str(row["human_id"])][condition] = row

    complete_ids = sorted(
        human_id
        for human_id, by_condition in machines.items()
        if human_id in humans
        and all(condition in by_condition for condition in REQUIRED_MACHINE_CONDITIONS)
    )
    if len(complete_ids) < n_prompts:
        raise ValueError(
            f"need {n_prompts} complete prompts, found {len(complete_ids)}"
        )
    selected = complete_ids[:n_prompts]
    keyed_items: list[dict[str, Any]] = []
    prompt_items: dict[str, list[dict[str, Any]]] = {}
    for human_id in selected:
        human = humans[human_id]
        blind_prompt_id = token("Q", seed, human_id, 10)
        rows = [
            {
                "original_id": f"HUMAN_{human_id}",
                "human_id": human_id,
                "source_type": "human",
                "generator_condition": "human_reference",
                "text": str(human["text"]),
                "prompt": str(human.get("approved_prompt") or human.get("prompt_candidate") or ""),
                "genre": str(human.get("genre") or ""),
                "human_source_dataset": str(human.get("source_dataset") or ""),
            }
        ]
        for condition in REQUIRED_MACHINE_CONDITIONS:
            machine = machines[human_id][condition]
            rows.append(
                {
                    "original_id": str(machine["machine_id"]),
                    "human_id": human_id,
                    "source_type": "machine",
                    "generator_condition": condition,
                    "text": str(machine["text"]),
                    "prompt": str(machine.get("prompt") or ""),
                    "genre": str(human.get("genre") or ""),
                    "human_source_dataset": "",
                }
            )
        for row in rows:
            row["blind_id"] = token("T", seed, row["original_id"])
            row["blind_prompt_id"] = blind_prompt_id
            row["char_count"] = len(row["text"])
            keyed_items.append(row)
        prompt_items[human_id] = rows

    primary_pairs: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    pair_key: list[dict[str, Any]] = []

    def add_pair(
        pair_id: str,
        left: dict[str, Any],
        right: dict[str, Any],
        match_type: str,
    ) -> None:
        pair = {
            "id": pair_id,
            "item_a": blind_item(left),
            "item_b": blind_item(right),
            "match_type": "same_prompt",
            "split_group": left["blind_prompt_id"],
            "meta": {"pilot_pair_type": match_type},
        }
        all_pairs.append(pair)
        if match_type == "human_machine":
            primary_pairs.append(pair)
        pair_key.append(
            {
                "pair_id": pair_id,
                "human_id": left["human_id"],
                "blind_id_a": left["blind_id"],
                "blind_id_b": right["blind_id"],
                "source_type_a": left["source_type"],
                "source_type_b": right["source_type"],
                "generator_condition_a": left["generator_condition"],
                "generator_condition_b": right["generator_condition"],
                "char_count_a": left["char_count"],
                "char_count_b": right["char_count"],
                "pilot_pair_type": match_type,
            }
        )

    for prompt_index, human_id in enumerate(selected):
        items = prompt_items[human_id]
        human, machine_items = items[0], items[1:]
        for generator_index, machine in enumerate(machine_items):
            # Every generator condition has human on each side exactly 10 times.
            human_first = (prompt_index + generator_index) % 2 == 0
            left, right = (human, machine) if human_first else (machine, human)
            add_pair(
                f"P20_HM_{prompt_index:02d}_{generator_index}",
                left,
                right,
                "human_machine",
            )
        for first in range(len(machine_items)):
            for second in range(first + 1, len(machine_items)):
                left, right = machine_items[first], machine_items[second]
                if (prompt_index + first + second) % 2:
                    left, right = right, left
                add_pair(
                    f"P20_MM_{prompt_index:02d}_{first}_{second}",
                    left,
                    right,
                    "machine_machine",
                )

    per_condition = Counter(row["generator_condition"] for row in keyed_items)
    manifest = {
        "protocol": "source-quality-falsification-pilot20-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_complete_prompts_available": len(complete_ids),
        "n_prompts_selected": len(selected),
        "n_items": len(keyed_items),
        "n_primary_human_machine_pairs": len(primary_pairs),
        "n_all_pairs": len(all_pairs),
        "selected_human_ids": selected,
        "items_per_condition": dict(sorted(per_condition.items())),
        "human_text_approval_status": "provisional_pending_formal_review",
        "blinding": "rating files omit source and generator; private key required for analysis",
    }
    return keyed_items, primary_pairs, all_pairs, {"pairs": pair_key, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--humans",
        type=Path,
        default=Path("data/human_pilot/review/human_candidates.jsonl"),
    )
    parser.add_argument(
        "--machines",
        type=Path,
        default=Path("data/human_pilot/generated/machine_pilot_pool.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/human_pilot/pilot20"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/human_pilot_20260821/pilot20_manifest.json"),
    )
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyed, primary_pairs, all_pairs, bundle = build_block(
        read_jsonl(args.humans),
        read_jsonl(args.machines),
        args.n_prompts,
        args.seed,
    )
    blind_items = [blind_item(row) for row in keyed]
    write_jsonl(args.output_dir / "private" / "items_keyed.jsonl", keyed)
    write_jsonl(args.output_dir / "private" / "pair_key.jsonl", bundle["pairs"])
    write_jsonl(args.output_dir / "blind" / "items.jsonl", blind_items)
    write_jsonl(args.output_dir / "blind" / "human_machine_pairs.jsonl", primary_pairs)
    write_jsonl(args.output_dir / "blind" / "all_within_prompt_pairs.jsonl", all_pairs)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(bundle["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
