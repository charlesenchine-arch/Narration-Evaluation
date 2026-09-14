# 类人叙事评估器 (Human-Like Narrative Evaluator)

构建一个**模仿人类审美的中文叙事文本评估器**。核心定位不是"机器文本检测最准"，而是评估器对文本的判断逻辑和输出**像人类评委**。最终作为叙事时间 Transformer 生成文本的 reward 信号：`score_text(生成文本) → 分数`。

> **质量评估 v2**：论文主任务已升级为“人类成对偏好 → 潜在连续效用 → A/B/C/D/F 等级 + 证据化评语”。旧的“像人分”保留为来源/风格分析与负对照，不再充当叙事质量金标准。完整协议见 `docs/QUALITY_EVALUATION_PROTOCOL.md`。

> **当前主实验 v4**：先检验大模型评审是否在控制人类共识质量后偏好自身或同一模型家族的生成文本，以及机器判断是否超出人类评审之间的正常分歧。文本目标约 500 字，人类端采用单人单题的匿名两两比较，模型端采用同一 pair 的位置互换复测；名著节选另设人类文学锚点实验，不混入核心家族偏好系数。完整协议见 `docs/FAMILY_AESTHETIC_BIAS_PROTOCOL.md`。

## 核心思想

- **三视图像人分**：`score_text = λ₁·S_disc + λ₂·S_repr + λ₃·S_attr`
  - `S_disc`：判别器 P(human)（字符 n-gram + Logistic Regression，或 MacBERT 升级版）
  - `S_repr`：冻结编码器（bge-small-zh-v1.5）文本向量到 H 参照集合的最近余弦相似度分位
  - `S_attr`：11 个可解释属性（句长/词汇丰富度/标点/重复率等）相对 H 分布的马氏距离
- **长度严苛度**：`final = 像人分^κ(len)`，长文机器更易露馅，故评估器对长文更严苛
- **长文判别**：MacBERT 以重叠 token 滑窗覆盖全文，再汇总窗口概率，不只读取开头 512 token
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

仓库不提交 MacBERT 权重。使用 `configs/default_deploy.yaml` 前，需先运行下方训练命令，
确保 `data/eval_dataset/models/macbert_discriminator/` 中已有微调产物；部署配置在权重
缺失时会直接报错，避免误用未微调的基础模型。H 语义/属性参照会按配置缓存为 `.npz`，
H 数据或编码器配置变化时自动重建。

## 快速使用

```bash
# 评估器 CLI：流式拉数据 → 独立训练/测试拆分 → fit → 测试集分布距离报告
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

### 成对质量盲评（推荐论文流程）

```bash
# 1. 构造长度/题材尽量匹配的比较对（默认排除旧锚点）
PYTHONPATH=src python scripts/build_pairwise_dataset.py

# 2. 启动“总体判断优先”的盲评界面
PYTHONPATH=src python scripts/pairwise_rating_server.py \
    --pairs data/eval_dataset/pairwise/pairs.jsonl --per-rater 20 --port 8780

# 3. 训练轻量 Bradley–Terry 基线；默认按作品/提示/文本严格留出
PYTHONPATH=src python scripts/train_quality_reward.py --split-mode group_disjoint

# 4. 强 LLM-as-a-Judge 对照（兼容 OpenAI SDK 的服务）
JUDGE_API_KEY=... PYTHONPATH=src python scripts/llm_pairwise_judge.py
```

默认等级为 A≥92、B≥80、C≥65、D≥50、F<50；阈值必须在开发集校准后冻结。配置见 `configs/quality_pairwise.yaml`，数据与基线调研见 `docs/dataset_and_baseline_survey.md`。

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
- `docs/QUALITY_EVALUATION_PROTOCOL.md` — ABCDF + 成对偏好 + 证据化评语的论文协议
- `docs/FAMILY_AESTHETIC_BIAS_PROTOCOL.md` — 当前主协议：模型自身与家族偏好、机器与人类判断偏差
- `docs/HUMAN_LITERARY_ANCHOR_SOURCES.md` — 约 500 字人类文学锚点的首批名著来源与版权审计规则
- `docs/DATASET_CONSTRUCTION_REQUIREMENTS.md` — Pool A 外部训练混池与 Pool B 自建冻结实验的统一规范
- `docs/DATASET_BUILD_PLAN.md` — 外部数据接入、自建同 Prompt 组文与 API 执行计划
- `docs/dataset_and_baseline_survey.md` — 数据集、许可证风险与强基线调研

## 仓库说明

- `literature/` 单独存放带注释的文献 PDF/TXT，与代码分离
- `.gitignore` 排除了 `secrets.env`（API key）、模型权重、HF 缓存、`logs/`、`.claude/`
- 本仓库为私有，仅作代码与研究数据备份
