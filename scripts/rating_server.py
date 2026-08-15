"""人工短文评分服务器（本地 + 网页）。

用法：
    python scripts/rating_server.py --data data/eval_dataset/raw/g_gen.jsonl
    # 浏览器打开 http://localhost:8770

功能：
- 多评分者独立评分，每人随机抽取部分文本（重叠+均分混合分配）
- 重叠子集：所有评分者共评（用于计算评分者一致性）
- 均分子集：其余文本按评分者轮转分配
- 最终分 = 每篇所有评分者评分的平均值
- 评分记录存 JSON，可导出汇总 CSV

依赖：标准库 http.server（无需额外安装）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HOST = "0.0.0.0"
PORT = 8770
PASSWORD = None  # 运行时由 main 设置；None=不启用口令
STATE_FILE = None  # 运行时由 main 设置
RATINGS_FILE = None
ALLOC_FILE = None
N_PER_RATER = 60  # 每个评分者分配的条数（默认）
# 标签偏差实验（Label Bias Probe）：none=盲评(默认)；correct=给真实来源标签；false=给颠倒来源标签
LABEL_MODE = "none"
# 纠偏参数
MIN_STAY_MS = 2000      # 每篇评分最短停留时间（毫秒），低于此视为敷衍/恶意，删除该条
MIN_RATINGS = 3         # 评分者最少有效评分数，低于此删除该评分者
CONVERGE_STD = 0.3      # 评分者标准差低于此值视为"打分趋同"（区分度过低）
W_MIN = 0.8             # 最低权重（保守/趋同评分者）
W_MAX = 1.2             # 最高权重（区分度好的评分者）
_alloc_lock = threading.Lock()


def load_json(path):
    path = str(path)
    if not os.path.exists(path):
        return None
    if path.endswith(".jsonl"):
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _label_for_item(item):
    """按 LABEL_MODE 返回来源标签文案；锚定文本（AH/AL）不参与标签实验，返回 None。"""
    if LABEL_MODE == "none":
        return None
    src = item.get("source", "") or ""
    if src.startswith("anchor") or item.get("id", "").startswith(("AH_", "AL_")):
        return None  # 锚定不显示标签，避免污染尺度校验
    is_h = (item.get("label") == "human")
    human_txt, machine_txt = "人类作家", "AI 模型"
    if LABEL_MODE == "correct":
        return human_txt if is_h else machine_txt
    if LABEL_MODE == "false":
        return machine_txt if is_h else human_txt
    return None


def _assign_for_new_rater(items, alloc, state, per_rater):
    """给新评分者分配文本。

    优先分配"尚未被评/评分次数少"的文本，使总体覆盖尽量均匀。
    返回该评分者的文本 id 列表。
    """
    all_ids = [it["id"] for it in items]
    # 已分配给其他人的
    assigned_elsewhere = set()
    for others in alloc.values():
        assigned_elsewhere.update(others)
    # 已被评过的文本（评分次数少的优先）
    rating_counts = {}
    for rater, rated in (state.get("raters") or {}).items():
        for iid in rated:
            rating_counts[iid] = rating_counts.get(iid, 0) + 1
    # 优先级：未分配 + 评分次数少
    unassigned = [i for i in all_ids if i not in assigned_elsewhere]
    unassigned.sort(key=lambda i: rating_counts.get(i, 0))
    # 已分配但评分少的（补充覆盖）
    under_rated = [i for i in all_ids if i in assigned_elsewhere and i not in assigned_elsewhere - set()]
    pick = unassigned[:per_rater]
    if len(pick) < per_rater:
        # 不够则从评分最少的已分配文本补
        rest = sorted([i for i in all_ids if i not in set(pick)], key=lambda i: rating_counts.get(i, 0))
        pick.extend(rest[: per_rater - len(pick)])
    return pick


def _pick_more(items, alloc, state, per_rater, exclude):
    """追加分配：评完当前批后，从"未分配给该评分者"的文本里按已评次数从少到多再抽 per_rater 条。

    想要继续评的评委可以一直评下去（分批续期）；exclude = 该评分者已见过的 id，避免重复评。
    """
    all_ids = [it["id"] for it in items]
    exclude = set(exclude)
    rating_counts = {}
    for rater, rated in (state.get("raters") or {}).items():
        for iid in rated:
            rating_counts[iid] = rating_counts.get(iid, 0) + 1
    pool = [i for i in all_ids if i not in exclude]
    pool.sort(key=lambda i: rating_counts.get(i, 0))
    return pool[:per_rater]


def build_allocation(item_ids, n_raters, per_rater, overlap):
    """重叠+均分混合分配。

    item_ids: 所有样本 id
    n_raters: 评分者数
    per_rater: 每人分配的条数
    overlap: 重叠子集条数（所有人共评）
    """
    rng = random.Random(42)
    ids = list(item_ids)
    # 随机选重叠子集
    rng.shuffle(ids)
    overlap_ids = ids[:overlap]
    remain = ids[overlap:]
    # 均分子集：按评分者轮转分配
    rng.shuffle(remain)
    # 每人固定条数 = per_rater；重叠部分算进去
    alloc = {f"rater_{i}": [] for i in range(n_raters)}
    remain_idx = 0
    for i in range(n_raters):
        # 每人 = 重叠(overlap) + 均分部分(per_rater - overlap)
        share = per_rater - overlap
        mine = list(overlap_ids)  # 重叠部分人人有
        for _ in range(share):
            if remain_idx < len(remain):
                mine.append(remain[remain_idx])
                remain_idx += 1
        alloc[f"rater_{i}"] = mine
    return alloc


class RatingHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send_file(os.path.join(os.path.dirname(__file__), "rating_frontend.html"), "text/html; charset=utf-8")
        elif url.path == "/api/items":
            q = parse_qs(url.query)
            rater = q.get("rater", [""])[0]
            if rater:
                # 动态分配：首次访问某名字时分配文本，分配持久化
                alloc = load_json(ALLOC_FILE) or {}
                state = load_json(STATE_FILE) or {"raters": {}}
                with _alloc_lock:
                    alloc = load_json(ALLOC_FILE) or {}
                    items = load_json(DATA_FILE) or []
                    if rater not in alloc:
                        alloc[rater] = _assign_for_new_rater(items, alloc, state, N_PER_RATER)
                    else:
                        # 评完当前批 → 追加下一批（想继续评的评委可一直评下去）
                        rated = set((state.get("raters") or {}).get(rater, {}).keys())
                        allocated = alloc[rater]
                        if allocated and all(iid in rated for iid in allocated):
                            more = _pick_more(items, alloc, state, N_PER_RATER, allocated)
                            if more:
                                alloc[rater] = allocated + more
                    save_json(ALLOC_FILE, alloc)
                self._send_json({"name": rater, "items": alloc.get(rater, [])})
            else:
                self._send_json(load_json(ALLOC_FILE) or {})
        elif url.path == "/api/text":
            q = parse_qs(url.query)
            item_id = q.get("id", [""])[0]
            items = load_json(DATA_FILE) or []
            for it in items:
                if it.get("id") == item_id:
                    # 盲评默认只返回 id/text；label_mode=correct/false 时附来源标签
                    resp = {"id": it.get("id"), "text": it.get("text", "")}
                    label_text = _label_for_item(it)
                    if label_text:
                        resp["label_text"] = label_text
                    self._send_json(resp)
                    return
            self._send_json({"error": "not found"}, 404)
        elif url.path == "/api/state":
            self._send_json(load_json(STATE_FILE) or {"raters": {}})
        elif url.path == "/api/summary":
            state = load_json(STATE_FILE) or {"raters": {}}
            items = load_json(DATA_FILE) or []
            self._send_json(_summarize(state, items))
        elif url.path == "/api/analysis":
            state = load_json(STATE_FILE) or {"raters": {}, "stays": {}}
            items = load_json(DATA_FILE) or []
            self._send_json(corrected_analysis(state, items))
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if url.path == "/api/login":
            password = body.get("password", "")
            if PASSWORD is None:
                self._send_json({"ok": True, "require": False})
            elif password == PASSWORD:
                self._send_json({"ok": True, "require": True})
            else:
                self._send_json({"ok": False, "require": True, "error": "口令错误"}, 401)
        elif url.path == "/api/rate":
            state = load_json(STATE_FILE) or {"raters": {}, "comments": {}, "stays": {}, "likes": {}}
            rater = body.get("rater")
            item_id = body.get("id")
            score = body.get("score")
            comment = body.get("comment", "")
            stay_ms = body.get("stay_ms")
            like = body.get("like")  # 0-100 像人程度（图灵测试需人类像人感知）
            if rater and item_id and score is not None:
                state.setdefault("raters", {}).setdefault(rater, {})[item_id] = float(score)
                if like is not None:
                    state.setdefault("likes", {}).setdefault(rater, {})[item_id] = int(like)
                if stay_ms is not None:
                    state.setdefault("stays", {}).setdefault(rater, {})[item_id] = int(stay_ms)
                if comment:
                    state.setdefault("comments", {}).setdefault(rater, {})[item_id] = comment
                else:
                    state.setdefault("comments", {}).setdefault(rater, {}).pop(item_id, None)
                save_json(STATE_FILE, state)
                self._send_json({"ok": True})
                return
            self._send_json({"error": "bad request"}, 400)
        else:
            self.send_error(404)


def _summarize(state, items):
    """汇总：每篇文本的评分者数、平均分、各评分者、评语、像人分。"""
    by_id = {}
    for it in items:
        by_id[it["id"]] = it
    comments = state.get("comments") or {}
    likes = state.get("likes") or {}
    rows = []
    for it in items:
        iid = it["id"]
        scores = []
        for rater, rated in (state.get("raters") or {}).items():
            if iid in rated:
                score = rated[iid]
                cmt = ""
                for crater, ccom in comments.items():
                    if crater == rater and iid in ccom:
                        cmt = ccom[iid]
                like = ((likes.get(rater) or {}).get(iid))
                scores.append({"rater": rater, "score": score, "comment": cmt, "like": like})
        mean = sum(s["score"] for s in scores) / len(scores) if scores else None
        like_mean = (sum(s["like"] for s in scores if s["like"] is not None) /
                     sum(1 for s in scores if s["like"] is not None)) if any(s["like"] is not None for s in scores) else None
        rows.append({
            "id": iid, "label": it.get("label"), "source": it.get("source", it.get("gen_model", "")),
            "n_ratings": len(scores), "mean": mean, "like_mean": like_mean,
            "ratings": scores,
        })
    return {"rows": rows}


def _compute_rater_stats(state):
    """计算每个评分者的原始统计（剔除停留时间过短的评分）。

    返回 dict: rater -> {items: {id: score}, stay_dropped: n, valid_n, std, mean}
    """
    raters = state.get("raters") or {}
    stays = state.get("stays") or {}
    out = {}
    for rater, rated in raters.items():
        valid = {}
        dropped = 0
        rater_stays = stays.get(rater) or {}
        for iid, score in rated.items():
            stay = rater_stays.get(iid)
            # 无停留时间记录（旧数据）默认保留；有记录且过短则删除
            if stay is not None and stay < MIN_STAY_MS:
                dropped += 1
                continue
            valid[iid] = score
        vals = list(valid.values())
        n = len(vals)
        if n == 0:
            # 全被过滤也记录（带原因），便于追溯
            out[rater] = {
                "items": {}, "stay_dropped": dropped, "valid_n": 0,
                "std": 0.0, "mean": None, "reason": "所有评分因停留过短被删除" if dropped else "无评分",
            }
            continue
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        out[rater] = {
            "items": valid, "stay_dropped": dropped, "valid_n": n,
            "std": std, "mean": mean,
        }
    return out


def _rater_weights(stats):
    """基于方差的评分者权重（贝叶斯收缩 + 温和映射）。

    步骤：
    1. 过滤：有效评分 < MIN_RATINGS 的评分者剔除；std < CONVERGE_STD 视为趋同，给最低权重。
    2. 贝叶斯收缩：s²_adj = (n·s² + k·s²₀)/(n + k)，k=评分者中位样本数，s²₀=全局均值。
       小样本方差噪声大，向先验收缩；样本越多越信任自身。
    3. 权重映射：min-max 归一化 σ_adj 到 [W_MIN, W_MAX]。
    返回 {rater: weight} 及统计信息。
    """
    keep = {r: s for r, s in stats.items() if s["valid_n"] >= MIN_RATINGS}
    # 全局方差均值（先验）
    s2_all = [s["std"] ** 2 for s in keep.values() if s["valid_n"] > 1]
    if not s2_all:
        return {}, {"dropped": list(stats.keys())}
    s2_0 = sum(s2_all) / len(s2_all)
    n_list = sorted(s["valid_n"] for s in keep.values())
    k = n_list[len(n_list) // 2] if n_list else MIN_RATINGS  # 收缩强度 = 中位样本数

    # 贝叶斯收缩后的方差
    sigma_adj = {}
    for r, s in keep.items():
        n = s["valid_n"]
        s2 = s["std"] ** 2 if n > 1 else s2_0
        sigma_adj[r] = ((n * s2 + k * s2_0) / (n + k)) ** 0.5

    # min-max 归一化 σ_adj → [W_MIN, W_MAX]
    sigmas = list(sigma_adj.values())
    lo, hi = min(sigmas), max(sigmas)
    weights = {}
    for r, sg in sigma_adj.items():
        if hi - lo < 1e-9:
            w = W_MIN
        else:
            t = (sg - lo) / (hi - lo)  # [0,1]
            w = W_MIN + (W_MAX - W_MIN) * t
        # 趋同评分者给最低权重（区分度差）
        if keep[r]["valid_n"] > 1 and keep[r]["std"] < CONVERGE_STD:
            w = min(w, W_MIN)
        weights[r] = w
    return weights, {
        "s2_prior": s2_0, "shrink_k": k,
        "dropped": [r for r in stats if r not in keep],
        "sigma_adj": sigma_adj, "std_raw": {r: s["std"] for r, s in keep.items()},
    }


def corrected_analysis(state, items):
    """纠偏汇总：过滤恶意评分 + 方差加权平均。

    返回：
    - rater_stats: 每个评分者的统计与权重
    - items: 每篇文本的原始平均分 + 加权平均分
    - summary: 纠偏说明（删除了哪些评分/评分者）
    """
    stats = _compute_rater_stats(state)
    weights, meta = _rater_weights(stats)

    # 每篇文本：加权平均 vs 简单平均
    rows = []
    for it in items:
        iid = it["id"]
        entries = []
        for r, s in stats.items():
            if iid in s["items"] and r in weights:
                entries.append((r, s["items"][iid], weights[r]))
        if not entries:
            rows.append({"id": iid, "weighted_mean": None, "simple_mean": None, "n": 0, "ratings": []})
            continue
        simple = sum(e[1] for e in entries) / len(entries)
        wsum = sum(e[1] * e[2] for e in entries)
        wtotal = sum(e[2] for e in entries)
        weighted = wsum / wtotal if wtotal > 0 else simple
        rows.append({
            "id": iid, "label": it.get("label"), "source": it.get("source", it.get("gen_model", "")),
            "weighted_mean": weighted, "simple_mean": simple, "n": len(entries),
            "ratings": [{"rater": r, "score": sc, "weight": w} for r, sc, w in entries],
        })

    # 每个评分者：统计 + 权重 + 被删条数 + 锚定一致性
    anchor_by_id = {}
    for it in items:
        src = it.get("source", "")
        if src == "anchor_high":
            anchor_by_id[it["id"]] = "high"
        elif src == "anchor_low":
            anchor_by_id[it["id"]] = "low"

    rater_summary = {}
    for r, s in stats.items():
        # 锚定一致性：该评分者对极好锚定的平均分（应高）、极差锚定的平均分（应低）
        ah = [sc for iid, sc in s["items"].items() if anchor_by_id.get(iid) == "high"]
        al = [sc for iid, sc in s["items"].items() if anchor_by_id.get(iid) == "low"]
        ah_mean = sum(ah) / len(ah) if ah else None
        al_mean = sum(al) / len(al) if al else None
        # 尺度异常：极好锚定给低分（<3）或极差锚定给高分（>3）——评分者尺度可能偏移
        scale_ok = True
        scale_note = ""
        if ah_mean is not None and ah_mean < 3.0:
            scale_ok = False
            scale_note = f"极好锚定平均分偏低({ah_mean:.1f})，尺度可能偏严"
        if al_mean is not None and al_mean > 3.0:
            scale_ok = False
            scale_note += (f" 极差锚定平均分偏高({al_mean:.1f})，尺度可能偏松" if scale_note else
                          f"极差锚定平均分偏高({al_mean:.1f})，尺度可能偏松")
        rater_summary[r] = {
            "valid_n": s["valid_n"], "mean": s["mean"], "std": s["std"],
            "stay_dropped": s["stay_dropped"],
            "weight": weights.get(r, 0.0) if r in weights else 0.0,
            "kept": r in weights,
            "reason": s.get("reason", ""),
            "anchor_mean_high": ah_mean, "anchor_mean_low": al_mean,
            "scale_ok": scale_ok, "scale_note": scale_note,
        }
    for r in meta["dropped"]:
        if r not in rater_summary:
            s = stats.get(r, {})
            reason = s.get("reason", "") or "评分数量不足"
            rater_summary[r] = {"valid_n": s.get("valid_n", 0),
                                "stay_dropped": s.get("stay_dropped", 0),
                                "kept": False, "weight": 0.0, "reason": reason}

    # 锚定整体校验：极好/极差锚定的平均分（检验锚定是否选得好）
    anchor_high_scores = []
    anchor_low_scores = []
    for row in rows:
        if row["weighted_mean"] is None:
            continue
        src = row.get("source", "")
        if src == "anchor_high":
            anchor_high_scores.append(row["weighted_mean"])
        elif src == "anchor_low":
            anchor_low_scores.append(row["weighted_mean"])
    anchor_check = {
        "anchor_high_count": len(anchor_high_scores),
        "anchor_high_mean": sum(anchor_high_scores) / len(anchor_high_scores) if anchor_high_scores else None,
        "anchor_low_count": len(anchor_low_scores),
        "anchor_low_mean": sum(anchor_low_scores) / len(anchor_low_scores) if anchor_low_scores else None,
    }

    return {
        "rows": rows,
        "rater_stats": rater_summary,
        "anchor_check": anchor_check,
        "correction": {
            "drop_reasons": meta["dropped"],
            "shrink_k": meta["shrink_k"], "s2_prior": meta["s2_prior"],
            "std_raw": meta.get("std_raw", {}), "sigma_adj": meta.get("sigma_adj", {}),
        },
    }


def main():
    global DATA_FILE, STATE_FILE, ALLOC_FILE, RATINGS_FILE, N_PER_RATER, PASSWORD, LABEL_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="评分样本 JSONL/JSON 路径")
    ap.add_argument("--per-rater", type=int, default=60, help="每个评分者分配的条数")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default=HOST, help="监听地址，0.0.0.0=局域网/公网可访问")
    ap.add_argument("--password", default=None, help="访问口令（可选，None=不启用）")
    ap.add_argument("--label-mode", default="none", choices=["none", "correct", "false"],
                    help="标签偏差实验：none=盲评；correct=给真实来源；false=给颠倒来源")
    args = ap.parse_args()

    # 加载样本
    if args.data.endswith(".jsonl"):
        items = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        items = load_json(args.data)
    if not items:
        print("样本为空", file=os.sys.stderr)
        return
    print(f"样本数: {len(items)}")
    DATA_FILE = args.data
    N_PER_RATER = args.per_rater
    PASSWORD = args.password
    LABEL_MODE = args.label_mode
    base = Path(args.data).parent
    STATE_FILE = base / "rating_state.json"
    ALLOC_FILE = base / "rating_allocation.json"
    RATINGS_FILE = base / "rating_summary.json"

    srv = HTTPServer((args.host, args.port), RatingHandler)
    print(f"评分服务器运行: http://{args.host}:{args.port}")
    if PASSWORD:
        print(f"口令保护已启用（口令: {PASSWORD}）")
    print("任意评分者名字均可登录，首次访问时自动分配文本（每人 N 条）。Ctrl+C 停止。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")


if __name__ == "__main__":
    main()
