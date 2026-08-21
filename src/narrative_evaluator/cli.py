"""命令行入口：流式拉数据 → fit → evaluate_set → 报告 → 清理缓存。

用法：
    python -m narrative_evaluator.cli --config configs/default.yaml [--samples N]
"""
from __future__ import annotations

import argparse
import os
import sys


def _train_eval_split(items, eval_fraction: float, seed: int):
    """确定性拆分训练/测试集；两边非空且互不重叠。"""
    items = list(items)
    if len(items) < 2:
        raise ValueError("训练/测试拆分至少需要 2 条样本")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("data.eval_fraction 必须位于 0 和 1 之间")
    import random
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    n_eval = min(len(items) - 1, max(1, round(len(items) * eval_fraction)))
    eval_idx = set(indices[:n_eval])
    train = [item for i, item in enumerate(items) if i not in eval_idx]
    evaluate = [item for i, item in enumerate(items) if i in eval_idx]
    return train, evaluate


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("需要 pyyaml：pip install pyyaml", file=sys.stderr)
        raise
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv=None):
    parser = argparse.ArgumentParser(description="类人叙事评估器")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("--samples", type=int, default=None,
                        help="覆盖采样数（H/G 各多少条，方便小规模冒烟）")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="保留临时 HF 缓存（默认结束清理）")
    args = parser.parse_args(argv)

    from .config import Config, default_config, load_config_from_dict, cleanup_cache

    cfg: Config
    if args.config and os.path.exists(args.config):
        cfg = load_config_from_dict(_load_yaml_config(args.config))
    else:
        cfg = default_config()
        if args.config:
            print(f"配置 {args.config} 不存在，使用默认配置", file=sys.stderr)
    if args.samples is not None:
        cfg.data.n_h_webnovel = args.samples
        cfg.data.n_h_cnnsum = args.samples
        cfg.data.n_g_coig = args.samples

    # 确保缓存目录存在
    os.makedirs(cfg.cache_dir, exist_ok=True)

    from .data import (
        load_webnovelbench,
        load_cnnsum,
        load_coig_writer,
    )

    print("加载人类叙事数据（WebNovelBench + CNNSum）...", file=sys.stderr)
    h_web = load_webnovelbench(cfg.data, cfg.cache_dir)
    h_cnn = load_cnnsum(cfg.data, cfg.cache_dir)
    h_records = h_web.records + h_cnn.records
    print(f"  H 人类样本 {len(h_records)} 条（WebNovelBench {len(h_web)} + CNNSum {len(h_cnn)}）", file=sys.stderr)

    print("加载机器叙事数据（COIG-Writer）...", file=sys.stderr)
    g_coig = load_coig_writer(cfg.data, cfg.cache_dir)
    g_records = g_coig.records
    print(f"  G 机器样本 {len(g_records)} 条", file=sys.stderr)

    h_train, h_eval = _train_eval_split(h_records, cfg.data.eval_fraction, cfg.data.seed)
    g_train, g_eval = _train_eval_split(g_records, cfg.data.eval_fraction, cfg.data.seed + 1)
    h_train_texts = [r.text for r in h_train]
    g_train_texts = [r.text for r in g_train]
    h_eval_texts = [r.text for r in h_eval]
    g_eval_texts = [r.text for r in g_eval]
    print(
        f"独立拆分：训练 H={len(h_train_texts)} G={len(g_train_texts)}；"
        f"测试 H={len(h_eval_texts)} G={len(g_eval_texts)}",
        file=sys.stderr,
    )

    from .evaluator import Evaluator

    ev = Evaluator(cfg)
    print("fit：训练判别器 + 缓存 H 参照分布...", file=sys.stderr)
    ev.fit(h_train_texts, g_train_texts)

    # 主评估：只在未参与 fit 的测试集上计算分布距离报告
    report = ev.evaluate_set(h_eval_texts, g_eval_texts)

    # 人类对齐验证：WebNovelBench 章节自带人类评分，评估器分数与之算 Spearman
    aligned_h = [r for r in h_eval if r.human_score is not None]
    if len(aligned_h) >= 5:
        print("人类对齐验证：评估器分数 vs WebNovelBench 人类评分...", file=sys.stderr)
        align = ev.alignment_spearman(
            [r.text for r in aligned_h],
            [r.human_score for r in aligned_h],
        )
        if align:
            report["alignment"] = align

    from .report import render_report, write_report

    print(render_report(report, cfg.report.top_attributes))
    path = write_report(report, cfg.report.out_dir, cfg.report.top_attributes)
    if path:
        print(f"\nJSON 报告已写入：{path}")

    if not args.no_cleanup:
        cleanup_cache(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
