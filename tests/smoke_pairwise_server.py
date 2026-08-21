"""HTTP integration smoke test for the blinded pairwise rating server."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


BASE = os.environ.get("PAIRWISE_TEST_BASE", "http://127.0.0.1:18780")
STATE = Path(os.environ["PAIRWISE_TEST_STATE"])
PASSWORD = os.environ.get("PAIRWISE_TEST_PASSWORD", "test-pass")


def request(path, payload=None):
    raw = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        BASE + path,
        data=raw,
        headers={
            **({"Content-Type": "application/json"} if raw is not None else {}),
            "X-Access-Password": PASSWORD,
        },
    )
    with urlopen(req, timeout=5) as response:
        body = response.read()
        return response.headers.get_content_type(), body


content_type, html = request("/")
assert content_type == "text/html"
assert "叙事比较评阅" in html.decode("utf-8")

_, raw = request("/api/login", {"password": "test-pass"})
assert json.loads(raw)["ok"] is True

rater = "integration_reader"
_, raw = request("/api/pairs?rater=" + quote(rater))
allocation = json.loads(raw)
assert len(allocation["ids"]) == 2
pair_id = allocation["ids"][0]

_, raw = request("/api/pair?id=" + quote(pair_id))
pair = json.loads(raw)
assert set(pair) == {"id", "text_a", "text_b"}  # blinded: no source/generator

_, raw = request("/api/judge", {
    "rater_id": rater,
    "pair_id": pair_id,
    "preference": 1,
    "confidence": 4,
    "dimensions": ["plot_causality", "character"],
    "rationale": "A 的人物动机与事件结果形成了更完整的因果链。",
    "evidence_a": "A 中的选择在结尾得到回应。",
    "evidence_b": "B 的转折缺少前置铺垫。",
    "stay_ms": 12000,
})
assert json.loads(raw)["ok"] is True

_, raw = request("/api/pairs?rater=" + quote(rater))
assert pair_id in json.loads(raw)["completed"]
state = json.loads(STATE.read_text(encoding="utf-8"))
assert state["judgments"][rater][pair_id]["confidence"] == 4
print("pairwise server smoke test passed")
