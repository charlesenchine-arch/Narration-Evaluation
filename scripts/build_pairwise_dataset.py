"""Build matched narrative pairs for the quality-evaluation protocol.

Example:
    PYTHONPATH=src python scripts/build_pairwise_dataset.py \
      --input data/eval_dataset/rating_set.jsonl \
      --output data/eval_dataset/pairwise/pairs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from narrative_evaluator.quality.pairing import build_balanced_pairs
from narrative_evaluator.quality.schemas import NarrativeItem, write_jsonl


def _load_items(path: str, include_anchors: bool) -> list[NarrativeItem]:
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            iid = str(obj.get("id", ""))
            if not include_anchors and iid.startswith(("AH_", "AL_")):
                continue
            meta = dict(obj.get("meta") or {})
            raw_prompt = str(obj.get("prompt") or meta.get("prompt") or "").strip()
            prompt_id = str(obj.get("prompt_id") or meta.get("prompt_id") or "")
            if not prompt_id and raw_prompt:
                prompt_id = "prompt_" + hashlib.sha1(raw_prompt.encode("utf-8")).hexdigest()[:12]
            items.append(NarrativeItem(
                id=iid,
                text=str(obj["text"]),
                source=str(obj.get("source") or obj.get("label") or ""),
                work_id=str(obj.get("work_id") or meta.get("work_id") or meta.get("novel") or ""),
                prompt_id=prompt_id,
                generator=str(obj.get("generator") or obj.get("gen_model") or meta.get("generator") or ""),
                genre=str(obj.get("genre") or meta.get("genre") or ""),
                meta={"original_label": obj.get("label", ""), **meta},
            ))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="构造中文叙事成对偏好数据")
    parser.add_argument("--input", default="data/eval_dataset/rating_set.jsonl")
    parser.add_argument("--output", default="data/eval_dataset/pairwise/pairs.jsonl")
    parser.add_argument("--comparisons-per-item", type=int, default=2)
    parser.add_argument("--max-length-ratio", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-anchors", action="store_true")
    args = parser.parse_args()

    items = _load_items(args.input, args.include_anchors)
    pairs = build_balanced_pairs(
        items,
        comparisons_per_item=args.comparisons_per_item,
        max_length_ratio=args.max_length_ratio,
        seed=args.seed,
    )
    write_jsonl(args.output, pairs)
    counts = Counter(pair.match_type for pair in pairs)
    manifest = {
        "protocol": "pairwise-quality-v1",
        "source": str(Path(args.input)),
        "n_items": len(items),
        "n_pairs": len(pairs),
        "comparisons_per_item": args.comparisons_per_item,
        "max_length_ratio": args.max_length_ratio,
        "seed": args.seed,
        "match_types": dict(counts),
        "grade_scale": {"A": 85, "B": 70, "C": 60, "D": 50, "F": 0},
    }
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {len(pairs)} 对：{args.output}")
    print(f"清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
