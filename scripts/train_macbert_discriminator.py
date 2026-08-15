"""训练/加载 MacBERT 判别器并输出验证报告。

数据管线：
- H：全量 WebNovelBench 章节 → chunk_text 切 300-800 字块 → dedupe_vs 去掉与
  验证集（评分集 H）精确重叠的文本 → length_match 分层采样
- G：全量 COIG 创意写作 → 过滤 300-800 字 → length_match 分层采样
- 验证集：rating_set 非锚定 500 条（H300+G100+GZ100），绝不混入训练

用法：
    PYTHONPATH=src python scripts/train_macbert_discriminator.py --config configs/default_macbert.yaml
    # 冒烟（≤1min）
    PYTHONPATH=src python scripts/train_macbert_discriminator.py --config configs/default_macbert.yaml \
        --n-h-train 40 --n-g-train 40 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from narrative_evaluator.config import Config, load_config
from narrative_evaluator.data import load_webnovelbench, load_coig_writer
from narrative_evaluator.data.chunking import chunk_texts, dedupe_vs
from narrative_evaluator.models.discriminator import build_discriminator, _length_match


def _load_rating_set(path):
    """返回 (H 文本列表, G 文本列表)，只取非锚定。"""
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    h = [it["text"] for it in items if it["id"].startswith("H_")]
    g = [it["text"] for it in items if it["id"].startswith(("G_", "GZ_"))]
    return h, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="配置文件（configs/default_macbert.yaml）")
    ap.add_argument("--n-h-train", type=int, default=None, help="覆盖训练 H 条数（冒烟用）")
    ap.add_argument("--n-g-train", type=int, default=None, help="覆盖训练 G 条数（冒烟用）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练 epoch（冒烟用）")
    ap.add_argument("--cache-dir", default="data/eval_dataset/hf_cache",
                    help="持久 HF 缓存目录（数据/模型只下载一次）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_h_train:
        cfg.discriminator.n_h_train = args.n_h_train
    if args.n_g_train:
        cfg.discriminator.n_g_train = args.n_g_train
    if args.epochs:
        cfg.discriminator.epochs = args.epochs

    os.makedirs(args.cache_dir, exist_ok=True)
    # 读取验证集（评分集非锚定）—— 必须先于训练去重
    val_h, val_g = _load_rating_set(cfg.discriminator.val_path)
    print(f"验证集: H={len(val_h)} G={len(val_g)}", flush=True)

    # Windows CPU 上显式限制 torch 线程数到物理核，避免超线程争抢
    if cfg.discriminator.threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(cfg.discriminator.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cfg.discriminator.threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cfg.discriminator.threads))

    print("加载 WebNovelBench（全量）+ 切块 300-800 字...", flush=True)
    h_web = load_webnovelbench(cfg.data, args.cache_dir)
    print(f"  WebNovelBench 章节数: {len(h_web)}", flush=True)
    h_chunks = chunk_texts(h_web.texts, 300, 800)
    print(f"  切块总数: {len(h_chunks)}", flush=True)
    # 去重：去掉与验证集 H 精确重叠的文本（防泄漏）
    before = len(h_chunks)
    h_chunks = dedupe_vs(h_chunks, val_h)
    print(f"  去重(与验证集H重叠): {before} → {len(h_chunks)}", flush=True)

    print("加载 COIG-Writer（全量）+ 过滤 300-800 字...", flush=True)
    g_coig = load_coig_writer(cfg.data, args.cache_dir)
    print(f"  COIG 条数: {len(g_coig)}", flush=True)
    g_pool = [t for t in g_coig.texts if 300 <= len(t) <= 800]
    print(f"  长度达标: {len(g_pool)}", flush=True)

    # 先随机采样到目标规模，再长度匹配分层（length_match 只会减少样本，不会增加）
    n_h = min(cfg.discriminator.n_h_train, len(h_chunks))
    n_g = min(cfg.discriminator.n_g_train, len(g_pool))
    rng = np.random.RandomState(cfg.discriminator.seed)
    h_pool = list(rng.choice(h_chunks, size=n_h, replace=False)) if n_h < len(h_chunks) else list(h_chunks)
    g_pool_s = list(rng.choice(g_pool, size=n_g, replace=False)) if n_g < len(g_pool) else list(g_pool)
    print(f"  随机采样: H={len(h_pool)} G={len(g_pool_s)}", flush=True)
    if n_h > 0 and n_g > 0:
        h_train, g_train = _length_match(h_pool, g_pool_s, n_buckets=8, seed=cfg.discriminator.seed)
        print(f"  长度匹配采样: H={len(h_train)} G={len(g_train)}", flush=True)
    else:
        h_train, g_train = h_pool, g_pool_s
        print(f"  （长度匹配跳过）采样: H={len(h_train)} G={len(g_train)}", flush=True)

    # 构造判别器并 fit（length_match=False：长度匹配已在采样阶段做）
    print(f"构造判别器 type={cfg.discriminator.type}，训练 {cfg.discriminator.epochs} epoch...", flush=True)
    disc = build_discriminator(cfg.discriminator, cache_dir=args.cache_dir)
    disc.fit(h_train, g_train, length_match=False,
             val_h_texts=val_h, val_g_texts=val_g,
             force_retrain=True)

    # 验证报告：训练集内 + 验证集上
    print("\n=== 验证报告 ===")
    model_path = disc.model_path if hasattr(disc, "model_path") else "?"
    print(f"模型路径: {model_path}")
    print(f"训练样本: H={disc.used_n_h} G={disc.used_n_g}  length_match={disc.length_match_used}")

    # 验证集 H/G 判别 AUC（用加载的模型）
    p_h = disc.predict_human_prob(val_h)
    p_g = disc.predict_human_prob(val_g)
    y = np.array([1] * len(val_h) + [0] * len(val_g))
    p = np.concatenate([p_h, p_g])
    auc_val = roc_auc_score(y, p)
    print(f"验证集(评分集 H300+G200) 判别 AUC = {auc_val:.4f}")

    # 与 n-gram 对比（同验证集）
    from narrative_evaluator.models.discriminator import NgramDiscriminator
    ng = NgramDiscriminator()
    ng.fit(val_h, val_g, length_match=True, n_buckets=8, seed=42)
    p_ng = ng.predict_human_prob(val_h + val_g)
    auc_ng = roc_auc_score(y, p_ng)
    print(f"[对照] n-gram(长度匹配) 同验证集 AUC = {auc_ng:.4f}")
    print(f"[结论] MacBERT 比 n-gram {'更高' if auc_val > auc_ng else '未超过'} "
          f"({auc_val:.3f} vs {auc_ng:.3f})")


if __name__ == "__main__":
    main()
