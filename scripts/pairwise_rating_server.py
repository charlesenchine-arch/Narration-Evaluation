"""Minimal blinded pairwise-rating server for the quality protocol."""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from narrative_evaluator.quality.schemas import QUALITY_DIMENSIONS


PAIRS = []
PAIR_BY_ID = {}
STATE_PATH = Path("pairwise_state.json")
FRONTEND = Path(__file__).with_name("pairwise_rating_frontend.html")
PER_RATER = 20
PASSWORD = None
LOCK = threading.Lock()


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _assignment(state, rater):
    assignments = state.setdefault("assignments", {})
    if rater in assignments:
        return assignments[rater]
    counts = {pair["id"]: 0 for pair in PAIRS}
    for ids in assignments.values():
        for pair_id in ids:
            counts[pair_id] = counts.get(pair_id, 0) + 1
    pool = list(counts)
    random.Random(f"pairwise-v1:{rater}").shuffle(pool)
    pool.sort(key=lambda pair_id: counts[pair_id])
    assignments[rater] = pool[:PER_RATER]
    return assignments[rater]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, status=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self):
        return PASSWORD is None or self.headers.get("X-Access-Password") == PASSWORD

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            raw = FRONTEND.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if url.path == "/api/pairs":
            if not self._authorized():
                return self._json({"error": "访问口令错误"}, 401)
            rater = parse_qs(url.query).get("rater", [""])[0].strip()
            if not rater:
                return self._json({"error": "缺少评分者名字"}, 400)
            with LOCK:
                state = _read_json(STATE_PATH, {"assignments": {}, "judgments": {}})
                ids = _assignment(state, rater)
                _write_json(STATE_PATH, state)
            completed = set((state.get("judgments", {}).get(rater) or {}))
            return self._json({"ids": ids, "completed": sorted(completed)})
        if url.path == "/api/pair":
            if not self._authorized():
                return self._json({"error": "访问口令错误"}, 401)
            pair_id = parse_qs(url.query).get("id", [""])[0]
            pair = PAIR_BY_ID.get(pair_id)
            if not pair:
                return self._json({"error": "pair not found"}, 404)
            # Deliberately omit source/generator labels in blinded collection.
            return self._json({
                "id": pair["id"],
                "text_a": pair["item_a"]["text"],
                "text_b": pair["item_b"]["text"],
            })
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if url.path == "/api/login":
            ok = PASSWORD is None or body.get("password") == PASSWORD
            return self._json({"ok": ok, "require": PASSWORD is not None}, 200 if ok else 401)
        if url.path == "/api/judge":
            if not self._authorized():
                return self._json({"error": "访问口令错误"}, 401)
            required = ("rater_id", "pair_id", "preference", "confidence")
            if any(body.get(key) is None or body.get(key) == "" for key in required):
                return self._json({"error": "请完成总体判断和信心评分"}, 400)
            if body["pair_id"] not in PAIR_BY_ID:
                return self._json({"error": "pair not found"}, 404)
            if int(body["preference"]) not in (-2, -1, 0, 1, 2):
                return self._json({"error": "invalid preference"}, 400)
            if not 1 <= int(body["confidence"]) <= 5:
                return self._json({"error": "invalid confidence"}, 400)
            dimensions = list(body.get("dimensions") or [])
            unknown = set(dimensions) - set(QUALITY_DIMENSIONS)
            if unknown:
                return self._json({"error": f"unknown dimensions: {sorted(unknown)}"}, 400)
            record = {
                "preference": int(body["preference"]),
                "confidence": int(body["confidence"]),
                "dimensions": dimensions,
                "rationale": str(body.get("rationale", "")).strip(),
                "evidence_a": str(body.get("evidence_a", "")).strip(),
                "evidence_b": str(body.get("evidence_b", "")).strip(),
                "stay_ms": int(body.get("stay_ms") or 0),
            }
            with LOCK:
                state = _read_json(STATE_PATH, {"assignments": {}, "judgments": {}})
                state.setdefault("judgments", {}).setdefault(body["rater_id"], {})[
                    body["pair_id"]
                ] = record
                _write_json(STATE_PATH, state)
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


def main() -> int:
    global PAIRS, PAIR_BY_ID, STATE_PATH, PER_RATER, PASSWORD
    parser = argparse.ArgumentParser(description="中文叙事成对盲评服务器")
    parser.add_argument("--pairs", default="data/eval_dataset/pairwise/pairs.jsonl")
    parser.add_argument("--state", default="data/eval_dataset/pairwise/pairwise_state.json")
    parser.add_argument("--per-rater", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()
    PAIRS = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line]
    PAIR_BY_ID = {pair["id"]: pair for pair in PAIRS}
    STATE_PATH = Path(args.state)
    PER_RATER = args.per_rater
    PASSWORD = args.password
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"成对盲评服务器：http://{args.host}:{args.port}（{len(PAIRS)} 对）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
