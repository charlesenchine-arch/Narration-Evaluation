"""全局配置：λ 权重、采样数、编码器名、缓存目录等。

配置优先级：默认值 < configs/default.yaml < 环境变量覆盖（由 cli 注入）。
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    # 各数据源采样数量（None 表示取该源全部可用条数）
    n_h_webnovel: int = 200      # WebNovelBench subset_d_100 章节采样（同时提供 human_score）
    n_h_cnnsum: int = 300        # CNNSum 长篇人类小说
    n_g_coig: int = 300          # COIG-Writer AI 创意写作
    n_g_realdet: int = 0         # RealDet MGT 补充（默认关闭，偏问答/论坛）
    n_h_realdet: int = 0         # RealDet HWT 人类短文（默认关闭，用于短文人类参照实验）
    # 单条文本最大长度（字符）。超长文本先截断，避免编码器内存压力
    max_chars: int = 8000
    # CNNSum 是长篇小说摘录，放宽截断以保真长度分布（严苛度基准依赖真实长度）
    max_chars_cnnsum: int = 30000
    seed: int = 42
    # CLI 将每个 H/G 数据池独立留出该比例作为测试集，避免训练集上报告 AUC。
    eval_fraction: float = 0.2


@dataclass
class EncoderConfig:
    name: str = "BAAI/bge-small-zh-v1.5"   # CPU 友好的轻量中文句编码器
    max_len: int = 512                     # 编码器 token 上限
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class DiscriminatorConfig:
    # type: "ngram"（字符 n-gram + LR）| "macbert"（MacBERT 全量微调）
    type: str = "ngram"
    # ngram 配置
    ngram_range: tuple = (1, 3)
    max_features: int = 50000
    cv_folds: int = 5
    # macbert 配置
    model_name: str = "hfl/chinese-macbert-base"
    model_path: str = ""          # 空 → 默认 data/eval_dataset/models/macbert_discriminator
    # 部署模式下要求 model_path 已包含微调产物，避免误用未训练的基础 MacBERT。
    require_trained_model: bool = False
    max_len: int = 512
    # 长文按 token 重叠滑窗；这里是相邻窗口重复的 token 数。
    window_overlap: int = 128
    batch_size: int = 16
    device: str = "auto"          # auto=cuda 可用则用 GPU，否则 CPU
    epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42
    n_h_train: int = 3000         # 训练脚本读取
    n_g_train: int = 3000
    val_path: str = "data/eval_dataset/rating_set.jsonl"
    threads: int = 0              # PyTorch 线程数（0=默认；Windows CPU 建议 24）


@dataclass
class MixConfig:
    # 三视图融合权重（判别/表示/属性），初始等权
    lambda_disc: float = 1.0 / 3.0
    lambda_repr: float = 1.0 / 3.0
    lambda_attr: float = 1.0 / 3.0
    # 长度严苛度：score = 像人分^κ(len)，κ = 1 + α·max(0,len-L0)/L0
    # 长文机器更易"露馅"，故对长文更严苛；基准 L0 默认取 H 中位长度（自适应）
    strict_alpha: float = 0.2
    strict_len_base: int = 0    # 0 = fit 时用 H 中位长度
    # 部署加固：S_repr/S_attr 只在"前 window_chars 字符"上计算（0=全文本）。
    # 防 B1 注水攻击：末尾追加通用填充会把整篇向量拉向 H 质心使 S_repr 虚涨；
    # 窗口化后填充不影响窗口内容，表示/属性视图不再被"通用化填充"骗高。
    window_chars: int = 0
    # H 语义/属性参照缓存；空字符串表示不持久化。
    # 缓存内含 H 文本和编码器配置指纹，输入变化时会自动重建。
    reference_cache_path: str = ""


@dataclass
class ReportConfig:
    out_dir: Optional[str] = None     # None 表示不落盘（仅控制台）；否则写 JSON/文本到该目录
    top_attributes: int = 8           # 属性明细表最多显示条数


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    # HF 缓存目录：指向临时目录，运行结束由 cli 清理，满足"不落盘"
    cache_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="hf_cache_"))

    def asdict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    return Config()


def load_config_from_dict(d: dict) -> Config:
    """从嵌套 dict（如 yaml 解析结果）覆盖默认配置。"""
    cfg = default_config()
    if "data" in d:
        for k, v in d["data"].items():
            setattr(cfg.data, k, v)
    if "encoder" in d:
        for k, v in d["encoder"].items():
            setattr(cfg.encoder, k, v)
    if "discriminator" in d:
        for k, v in d["discriminator"].items():
            setattr(cfg.discriminator, k, v)
    if "mix" in d:
        for k, v in d["mix"].items():
            setattr(cfg.mix, k, v)
    if "report" in d:
        for k, v in d["report"].items():
            setattr(cfg.report, k, v)
    if "cache_dir" in d and d["cache_dir"]:
        cfg.cache_dir = os.path.abspath(d["cache_dir"])
    return cfg


def load_config(path: str) -> Config:
    """从 YAML 文件加载配置（cli 与脚本共用）。"""
    import yaml
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return load_config_from_dict(d)


def cleanup_cache(cfg: Config) -> None:
    """删除临时缓存目录（运行结束后调用）。"""
    if cfg.cache_dir and os.path.isdir(cfg.cache_dir):
        try:
            import shutil
            shutil.rmtree(cfg.cache_dir, ignore_errors=True)
        except Exception:
            pass
