#!/usr/bin/env python3
"""Build a provenance-aware human narrative pilot and prompt review sheet.

The builder deliberately keeps downloaded texts and review artifacts in ignored
directories.  COIG-Writer records have an explicit ODC-BY license; STORAL has no
dataset-level license statement in its public repository as of 2026-08-21 and is
therefore marked internal-review-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCES = {
    "coig": {
        "filename": "coig_writer_all_human_data.json",
        "url": "https://cdn.jsdelivr.net/gh/COIG-Writer/COIG-Writer@main/all_human_data.json",
        "sha256": "67ebc804312e92fa2e02d05dce6dd4465101bc1385641df602ce66a62de536a1",
        "homepage": "https://github.com/COIG-Writer/COIG-Writer",
    },
    "storal_valid": {
        "filename": "storal_mo2st_zh_valid.json",
        "url": "https://hf-mirror.com/datasets/Jiann/STORAL/resolve/main/mo2st_data_zh/valid.json",
        "sha256": "a8e5e93f113e61c4d646ce9904520d8510cf10f6a9931088c5312d15c9008855",
        "homepage": "https://github.com/thu-coai/MoralStory",
    },
    "storal_test": {
        "filename": "storal_mo2st_zh_test.json",
        "url": "https://hf-mirror.com/datasets/Jiann/STORAL/resolve/main/mo2st_data_zh/test.json",
        "sha256": "a697a482934e54022e9845dfdeaf537bcbf4ee6f68f2bb85480055abab76e276",
        "homepage": "https://github.com/thu-coai/MoralStory",
    },
}

COIG_REQUIRE = re.compile(
    r"(?:写|创作|构思|讲述|编写|设计).{0,18}(?:故事|小说|叙事|童话|寓言)"
    r"|(?:故事|小说|叙事|童话|寓言).{0,18}(?:写|创作|构思|讲述|编写|设计)"
)
COIG_EXCLUDE = re.compile(
    r"诗歌|诗词|歌词|广告|文案|辩论|论文|新闻|报告|评论|介绍|剧本|脚本|小品|相声|"
    r"公众号|微博|散文|游记|读后感|影评|书评|采访|社交媒体|职场文|RPG|游戏剧情|"
    r"大纲|续写|接着写|片段|章节|长篇小说|修改|改写|润色|仿写|演讲|演讲稿|"
    r"游戏小说|世界设定|小说设定|玄幻小说的开篇|楔子|百年孤独|参考草船借箭|"
    r"模仿.{0,8}(?:风格|作者)"
)
COIG_ANSWER_EXCLUDE = re.compile(
    r"^(?:下面将|以下是|为了满足|这个思路|以蟑螂的视角|一[、.]\s*世界设定)"
)

CSV_FIELDS = [
    "human_id",
    "source_dataset",
    "source_record_id",
    "source_split",
    "license_status",
    "redistribution_status",
    "genre",
    "char_count",
    "text",
    "source_prompt_raw",
    "prompt_origin",
    "prompt_candidate",
    "prompt_leakage_risk",
    "approve_text",
    "approved_prompt",
    "exclude_reason",
    "review_notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\u3000]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalized_hash(value: str) -> str:
    compact = re.sub(r"\s+", "", normalize_text(value))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def download_checked(source: dict[str, str], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / source["filename"]
    if destination.exists() and sha256_file(destination) == source["sha256"]:
        return destination
    if destination.exists():
        destination.unlink()

    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "Narration-Evaluation-human-pilot/1.0"}
    )
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, tmp)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    actual = sha256_file(temporary)
    if actual != source["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {source['filename']}: expected {source['sha256']}, got {actual}"
        )
    os.replace(temporary, destination)
    return destination


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array: {path}")
        return parsed
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def detect_genre(query: str, text: str = "") -> str:
    sample = f"{query}\n{text[:240]}"
    rules = [
        (r"寓言|道理|哲理", "寓言故事"),
        (r"童话|儿童", "童话故事"),
        (r"校园|学生|教室|老师|同学", "校园故事"),
        (r"科幻|太空|宇宙|星球|未来|赛博|机器人", "科幻故事"),
        (r"悬疑|推理|侦探|案件|凶手|惊悚|灵异|恐怖", "悬疑故事"),
        (r"历史|古代|古风|朝代|皇帝|王朝", "历史故事"),
        (r"武侠|江湖|剑客|侠客", "武侠故事"),
        (r"奇幻|玄幻|魔法|妖怪|精灵|龙族|鲛人", "奇幻故事"),
        (r"爱情|恋爱|婚姻|情侣", "情感故事"),
    ]
    for pattern, genre in rules:
        if re.search(pattern, sample):
            return genre
    return "现实主义短篇故事"


def extract_subject(query: str) -> str:
    patterns = [
        r"《([^》\n]{2,30})》",
        r"[“\"]([^”\"\n]{2,30})[”\"]",
        r"(?:以|围绕|关于)(?:一个|一位|一名)?[“\"]?([^，。；：\n\"”]{2,24})",
    ]
    blocked_generic = re.compile(
        r"^(?:及|以及|对)|要求|刻画|描述|风格|视角|情节|元素|作者|作品|结构|语言"
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, query)
        if match:
            value = re.sub(r"^(?:主题|题目|标题)[:：]?", "", match.group(1)).strip()
            if value and (index < 2 or not blocked_generic.search(value)):
                return value
    return "一次改变人物处境的事件"


def coig_prompt(query: str, genre: str) -> str:
    subject = extract_subject(query)
    return (
        f"请创作一篇500—1200字的中文{genre}。题材围绕“{subject}”展开，"
        "人物、冲突、转折与结局由你独立设计。作品必须是一篇完整叙事；"
        "不要写创作说明，不要模仿具体作者，也不要复述任何已知作品。"
    )


def storal_prompt(moral: str) -> str:
    moral = normalize_text(moral).rstrip("。！？； ")
    return (
        f"请创作一篇500—1200字的中文寓言故事，围绕以下主题展开：“{moral}”。"
        "人物、冲突、转折与结局由你独立设计。作品必须是一篇完整叙事；"
        "不要写创作说明，也不要复述任何已知寓言。"
    )


def stable_key(seed: int, *parts: object) -> str:
    value = ":".join([str(seed), *map(str, parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def balanced_sample(
    records: Iterable[dict[str, Any]], n: int, group_field: str, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[group_field])].append(record)
    for name, items in groups.items():
        items.sort(key=lambda item: stable_key(seed, name, item["source_record_id"]))
    selected: list[dict[str, Any]] = []
    names = sorted(groups)
    while len(selected) < n and names:
        remaining = []
        for name in names:
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop(0))
            if groups[name]:
                remaining.append(name)
        names = remaining
    return selected


def build_coig(path: Path, min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row_number, row in enumerate(read_json_or_jsonl(path), start=1):
        text = normalize_text(str(row.get("answer", "")))
        query = normalize_text(str(row.get("query", "")))
        if not (min_chars <= len(text) <= max_chars):
            continue
        if (
            not COIG_REQUIRE.search(query)
            or COIG_EXCLUDE.search(query)
            or COIG_ANSWER_EXCLUDE.search(text)
        ):
            continue
        genre = detect_genre(query, text)
        record_id = str(row.get("id", row_number))
        candidates.append(
            {
                "source_dataset": "COIG-Writer",
                "source_record_id": record_id,
                "source_split": "all_human_data",
                "source_url": SOURCES["coig"]["homepage"],
                "license_name": "ODC-BY-1.0",
                "license_status": "repository_declares_odc_by_1_0",
                "redistribution_status": "review_content_rights_and_attribution_before_publish",
                "provenance_notes": "answer field is the original human work; query is a reverse-engineered prompt",
                "text": text,
                "text_sha256": normalized_hash(text),
                "char_count": len(text),
                "genre": genre,
                "sample_group": genre,
                "source_prompt_raw": query,
                "prompt_origin": "neutralized_from_dataset_reverse_prompt",
                "prompt_candidate": coig_prompt(query, genre),
                "prompt_leakage_risk": "high" if len(query) > 240 else "medium",
                "prompt_status": "pending_human_review",
            }
        )
    return candidates


def build_storal(
    paths: Iterable[tuple[str, Path]], min_chars: int, max_chars: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for split, path in paths:
        for row_number, row in enumerate(read_json_or_jsonl(path), start=1):
            beginning = normalize_text(str(row.get("beginning", "")))
            continuation = normalize_text(str(row.get("story", "")))
            text = normalize_text(f"{beginning}{continuation}")
            moral = normalize_text(str(row.get("moral", "")))
            if not moral or not (min_chars <= len(text) <= max_chars):
                continue
            record_id = str(row.get("id", row_number))
            if len(text) < 700:
                bucket = "500-699"
            elif len(text) < 900:
                bucket = "700-899"
            else:
                bucket = "900-1200"
            candidates.append(
                {
                    "source_dataset": "STORAL",
                    "source_record_id": record_id,
                    "source_split": split,
                    "source_url": SOURCES["storal_valid"]["homepage"],
                    "license_name": "not_stated",
                    "license_status": "dataset_license_not_stated",
                    "redistribution_status": "internal_review_only",
                    "provenance_notes": "human story; full text reconstructed as beginning + story; prompt is observational, not the original writing instruction",
                    "text": text,
                    "text_sha256": normalized_hash(text),
                    "char_count": len(text),
                    "genre": "寓言故事",
                    "sample_group": bucket,
                    "source_prompt_raw": moral,
                    "prompt_origin": "reconstructed_from_moral_label",
                    "prompt_candidate": storal_prompt(moral),
                    "prompt_leakage_risk": "medium",
                    "prompt_status": "pending_human_review",
                }
            )
    return candidates


def deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        fingerprint = record["text_sha256"]
        if fingerprint in seen:
            removed += 1
            continue
        seen.add(fingerprint)
        kept.append(record)
    return kept, removed


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row.update(
                {
                    "approve_text": "",
                    "approved_prompt": record["prompt_candidate"],
                    "exclude_reason": "",
                    "review_notes": "",
                }
            )
            writer.writerow(row)


def write_sqlite(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write a queryable local database without weakening provenance fields."""
    rows = list(records)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS human_candidates")
        connection.execute(
            """
            CREATE TABLE human_candidates (
                human_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_dataset TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                source_split TEXT NOT NULL,
                source_url TEXT NOT NULL,
                license_name TEXT NOT NULL,
                license_status TEXT NOT NULL,
                redistribution_status TEXT NOT NULL,
                provenance_notes TEXT NOT NULL,
                genre TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL UNIQUE,
                prompt_origin TEXT NOT NULL,
                source_prompt_raw TEXT NOT NULL,
                prompt_candidate TEXT NOT NULL,
                prompt_leakage_risk TEXT NOT NULL,
                prompt_status TEXT NOT NULL,
                include_status TEXT NOT NULL
            )
            """
        )
        columns = [row[1] for row in connection.execute("PRAGMA table_info(human_candidates)")]
        placeholders = ",".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO human_candidates ({','.join(columns)}) VALUES ({placeholders})",
            [[record[column] for column in columns] for record in rows],
        )
        connection.execute(
            "CREATE INDEX idx_human_candidates_source ON human_candidates(source_dataset)"
        )
        connection.execute(
            "CREATE INDEX idx_human_candidates_genre ON human_candidates(genre)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/human_pilot/cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/human_pilot/review"))
    parser.add_argument("--coig", type=Path)
    parser.add_argument("--storal-valid", type=Path)
    parser.add_argument("--storal-test", type=Path)
    parser.add_argument("--n-coig", type=int, default=20)
    parser.add_argument("--n-storal", type=int, default=40)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_chars <= 0 or args.min_chars > args.max_chars:
        raise ValueError("invalid character-length range")
    source_paths = {
        "coig": args.coig or download_checked(SOURCES["coig"], args.cache_dir),
        "storal_valid": args.storal_valid
        or download_checked(SOURCES["storal_valid"], args.cache_dir),
        "storal_test": args.storal_test
        or download_checked(SOURCES["storal_test"], args.cache_dir),
    }

    coig_pool = build_coig(source_paths["coig"], args.min_chars, args.max_chars)
    storal_pool = build_storal(
        [("valid", source_paths["storal_valid"]), ("test", source_paths["storal_test"])],
        args.min_chars,
        args.max_chars,
    )
    coig_pool, coig_duplicates = deduplicate(coig_pool)
    storal_pool, storal_duplicates = deduplicate(storal_pool)
    if len(coig_pool) < args.n_coig or len(storal_pool) < args.n_storal:
        raise RuntimeError(
            f"not enough eligible records: COIG {len(coig_pool)}/{args.n_coig}, "
            f"STORAL {len(storal_pool)}/{args.n_storal}"
        )

    coig_selected = balanced_sample(coig_pool, args.n_coig, "sample_group", args.seed)
    storal_selected = balanced_sample(storal_pool, args.n_storal, "sample_group", args.seed)
    selected, cross_duplicates = deduplicate([*coig_selected, *storal_selected])
    if len(selected) != args.n_coig + args.n_storal:
        raise RuntimeError("cross-source duplicates reduced the requested pilot size")

    counters: dict[str, int] = defaultdict(int)
    for record in selected:
        prefix = "COIG" if record["source_dataset"] == "COIG-Writer" else "STORAL"
        counters[prefix] += 1
        record["human_id"] = f"HP_{prefix}_{counters[prefix]:04d}"
        record["source_type"] = "human"
        record["include_status"] = "pending_human_review"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "human_candidates.jsonl", selected)
    write_review_csv(args.out_dir / "human_candidates_review.csv", selected)
    write_sqlite(args.out_dir / "human_candidates.sqlite3", selected)
    manifest = {
        "schema_version": "human-pilot-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "length_range_chars": [args.min_chars, args.max_chars],
        "counts": {
            "coig_pool": len(coig_pool),
            "storal_pool": len(storal_pool),
            "coig_selected": len(coig_selected),
            "storal_selected": len(storal_selected),
            "total_selected": len(selected),
            "duplicates_removed": coig_duplicates + storal_duplicates + cross_duplicates,
        },
        "sources": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "download_url": SOURCES[key]["url"],
                "homepage": SOURCES[key]["homepage"],
            }
            for key, path in source_paths.items()
        },
        "license_warning": "STORAL dataset license is not stated; its texts are internal-review-only until rights are cleared.",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"review: {(args.out_dir / 'human_candidates_review.csv').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
