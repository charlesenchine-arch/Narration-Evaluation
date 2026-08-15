# 类人叙事评估器 (Human-Like Narrative Evaluator)

构建一个**模仿人类审美的中文叙事文本评估器**。核心定位不是"机器文本检测最准"，而是评估器对文本的判断逻辑和输出**像人类评委**。最终作为叙事时间 Transformer 生成文本的 reward 信号：`score_text(生成文本) → 分数`。

## 核心思想

- **三视图像人分**：`score_text = λ₁·S_disc + λ₂·S_repr + λ₃·S_attr`
  - `S_disc`：判别器 P(human)（字符 n-gram + Logistic Regression，或 MacBERT 升级版）
  - `S_repr`：冻结编码器（bge-small-zh-v1.5）文本向量到 H 参照集合的最近余弦相似度分位
  - `S_attr`：11 个可解释属性（句长/词汇丰富度/标点/重复率等）相对 H 分布的马氏距离
- **长度严苛度**：`final = 像人分^κ(len)`，长文机器更易露馅，故评估器对长文更严苛
- **解耦性**（论文主线）：像人分 vs 质量分是两个**正交维度**——像人分能判断"像不像人"，对"好不好"一无所知

## 目录结构

```
├── src/narrative_evaluator/   # 评估器核心（evaluator/features/models/metrics）
├── scripts/                   # 实验与工具脚本（探针/生成/评分/训练）
├── configs/                   # YAML 配置（默认 / 部署加固 / MacBERT）
├── tests/                     # 单元测试
├── docs/                      # 论文草稿、综述、引用库、协议
├── literature/                # 带注释的文献 PDF/TXT
├── data/eval_dataset/         # 自建数据集与实验数据（raw/ 原始 jsonl）
└── PROJECT_STATUS.md          # 完整项目状态（做了什么/没做什么/接下来）
```

## 安装

```bash
pip install -r requirements.txt
```

GPU 训练 MacBERT 判别器需 CUDA 版 torch（本项目用 `torch 2.13.0+cu126`，RTX 4060 Laptop）。

## 快速使用

```bash
# 评估器 CLI：流式拉数据 → fit → 分布距离报告
PYTHONPATH=src python -m narrative_evaluator.cli --config configs/default.yaml

# MacBERT 判别器训练（GPU ~1min）
PYTHONPATH=src python scripts/train_macbert_discriminator.py --config configs/default_macbert.yaml

# 解耦性探针（像人分 vs 质量分 正交性）
PYTHONPATH=src python scripts/decoupling_probe_llm.py --config configs/default_macbert.yaml

# LLM 质量分盲评（需 DEEPSEEK_API_KEY，断点续传）
PYTHONPATH=src python scripts/llm_quality_rating.py --data data/eval_dataset/rating_set.jsonl \
    --out data/eval_dataset/raw/llm_quality_ratings.jsonl

# B1 严苛度防刷分验证
PYTHONPATH=src python scripts/strictness_antigame.py --config configs/default_deploy.yaml

# 单元测试
PYTHONPATH=src python -m pytest tests/
```

### 人工评分服务器

盲评（前端不显示来源）+ 趣味五档评分 + 评语 + 动态分配 + 方差贝叶斯收缩纠偏 + 锚定校验：

```bash
python scripts/rating_server.py --data data/eval_dataset/rating_set.jsonl \
    --per-rater 15 --host 0.0.0.0 --password narrate2026
```

## 数据集

自建 `data/eval_dataset/rating_set.jsonl`（600 条，评分集合计）：
H 网络小说切块 300 + G 非章回 DeepSeek 100 + G 章回体 DeepSeek 100 + 极好/极差锚定各 50。
`raw/` 下为各批次原始生成/收集/评分数据。模型权重与 HF 缓存不随仓库保存（见 .gitignore）。

## 主要结论（详见 PROJECT_STATUS.md）

| 实验 | 结果 |
|---|---|
| 解耦性：像人分区分 H/G | AUC 0.999（n-gram）/ 0.9995（MacBERT） |
| 解耦性：像人分区分 H 内高低质 | AUC ≈ 0.24（≈随机，正交成立） |
| 长度偏置控制 | G 内 像人分 vs 长度 -0.562 → +0.070（MacBERT） |
| B1 严苛度防刷分 | 三种注水全堵，Spearman(final, len) = -0.859 |
| 跨生成器稳健性 | H vs 千问 qwen3.7 AUC 0.993 |

## 文档

- `PROJECT_STATUS.md` — 最完整的状态与历史
- `docs/paper_draft_en.md` — 英文论文草稿（ACL 风格）
- `docs/literature_review.md` + `docs/references.bib` — 文献综述与引用库
- `docs/evaluator_turing_test_protocol.md` — 评估者图灵测试协议

## 仓库说明

- `literature/` 单独存放带注释的文献 PDF/TXT，与代码分离
- `.gitignore` 排除了 `secrets.env`（API key）、模型权重、HF 缓存、`logs/`、`.claude/`
- 本仓库为私有，仅作代码与研究数据备份
