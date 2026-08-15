# 类人叙事评估器：研究说明与行动大纲

> Human-like Narrative Evaluator — Project Brief & Action Plan

## 1. 项目目标 / Project Goal

构建一个能够**以接近人类审美标准评价叙事文本质量**的自动化评估模型。该模型的核心验证方式是**"评估者图灵测试"（Evaluator Turing Test）**：将模型对文本的评分/二分类判断与人类评委的判断进行对比，如果人类无法区分哪些评价来自机器、哪些来自人类，则评估器通过测试。

通过图灵测试的评估器，将作为叙事时间 Transformer（见 `NARRATIVE_TIME_PROJECT_PLAN.md`）生成文本的自动化评价指标，替代或补充昂贵的人工评价。

The goal is to build an automated evaluation model that judges narrative text quality in a way that aligns with human aesthetic standards. Its core validation is the **Evaluator Turing Test**: if human judges cannot distinguish the model's ratings/classifications from those of human evaluators, the evaluator passes. A validated evaluator will serve as an automated metric for the Narrative Time Transformer's generated text.

## 2. 核心思路：评估者图灵测试 / The Evaluator Turing Test

### 2.1 传统图灵测试 vs 评估者图灵测试

| | 传统图灵测试 | 评估者图灵测试 |
|---|---|---|
| **测试对象** | 对话 AI | 文本评估模型 |
| **判断任务** | 分辨对话者是人还是机器 | 分辨"这条评价是人给的还是模型给的" |
| **通过标准** | 人类无法可靠区分 | 人类无法区分评价来源 |
| **通过后的用途** | 通用对话 | 替代人工评价 |

### 2.2 具体流程

```
阶段 1：训练评估模型
┌─────────────────────────────────────────────────┐
│  训练数据：叙事文本 + 人工质量评分/人机二分类标签    │
│  模型学习：预测人类对文本的评价                      │
│  输出：一个能对任意叙事文本打分的评估器 E            │
└─────────────────────────────────────────────────┘
                          │
                          ▼
阶段 2：图灵测试验证
┌─────────────────────────────────────────────────┐
│  测试集：N 篇叙事文本（含人类写作和机器生成）        │
│  - 评估器 E 给出每篇文本的评分/分类                │
│  - 人类评委 H 给出每篇文本的评分/分类               │
│  - 混合 E 和 H 的评价，去除来源标记                │
│  - 另一组人类裁判 J 判断每条评价来自人还是机器       │
│                                                   │
│  如果 J 的区分准确率 ≈ 随机猜测（50%）：            │
│    → 评估器 E 通过图灵测试                        │
│    → E 可作为人类审美的自动化代理                  │
└─────────────────────────────────────────────────┘
                          │
                          ▼
阶段 3：部署使用
┌─────────────────────────────────────────────────┐
│  用评估器 E 评价叙事时间 Transformer 的生成文本     │
│  - 自动打分：替代人工评价，支持大规模实验            │
│  - 消融对比：不同的叙事时间注入方式 → 文本质量差异   │
│  - 训练信号：评估器可作为生成模型的 reward model    │
└─────────────────────────────────────────────────┘
```

## 3. 核心研究问题 / Research Questions

1. **可行性**：在中文叙事文本领域，当前最优的 AI 文本检测/评价模型能否通过评估者图灵测试？
2. **特征层级**：评估器学到的是深层叙事特征（时间结构、情节逻辑、叙事声音、人物一致性），还是浅层表面特征（词频分布、句长、perplexity、ngram 重叠度）？如何通过可控实验区分？
3. **叙事时间敏感度**：评估器是否对叙事时间操作（倒叙、预叙、时长变化、省略）敏感？人为扰动文本的叙事时间结构后，评分变化是否符合人类直觉？
4. **泛化能力**：在特定类型叙事文本（如短篇小说）上训练的评估器，能否泛化到其他叙事类型（网络小说、传记、新闻叙事）？
5. **与叙事时间场的关系**：加入了叙事时间场的 Transformer 生成的文本，在评估器的评分上是否显著优于无时间场的 baseline？评估器的评分排序是否与人工评价的排序一致？
6. **对抗鲁棒性**：生成模型是否可以通过对抗训练"欺骗"评估器（提高评估器分数但不提高人类评价分数）？如何检测和防止这种解耦？

## 4. 模型设计 / Model Design

### 4.1 两种可选范式

#### 方案 A：二分类判别器（Classifier）

```text
输入：叙事文本片段
输出：P(human-written) ∈ [0, 1]
基础架构：MacBERT / RoBERTa + 分类头
训练目标：二元交叉熵损失
```

**优点**：简单、可解释、数据标注成本低（只需人机标签）。
**缺点**：二分类过于粗糙，无法捕捉连续的质量光谱；容易学到表面的"AI 味儿"特征而非叙事质量。

#### 方案 B：评分回归模型（Scorer）

```text
输入：叙事文本片段
输出：多维评分向量 [叙事时间结构, 情节连贯性, 语言美感, 人物一致性, 整体质量]
基础架构：MacBERT / RoBERTa + 多任务回归头
训练目标：MSE / ordinal regression
```

**优点**：细粒度、与人类审美标准对齐、可用于诊断性评价。
**缺点**：标注成本高，需要多维度人工评分数据。

#### 建议路径

先以**方案 A（二分类）**快速验证图灵测试范式的可行性；若通过，升级为**方案 B（多维评分）**，并在评分维度中显式纳入叙事时间相关维度，直接对接叙事时间场项目。

### 4.2 增强策略（防止学到表面特征）

- **对抗训练**：训练中引入故意改写/扰乱叙事时间结构但仍保持流畅的负样本，迫使模型关注深层的叙事逻辑而非措辞。
- **多粒度输入**：同时输入全文、段落和句子级表征，防止模型仅靠局部词汇统计做判断。
- **解耦探针（Probe）**：在评估器中间层训练探针，检测是否编码了叙事时间、事件因果链、人物状态等结构化信息。

## 5. 数据策略 / Data Strategy

### 5.1 现成可用资源汇总

以下是通过调研确认的、可直接用于评估器项目的公开数据集：

#### 5.1.1 人机文本检测数据集（评估器训练核心）

| 数据集 | 规模 | 模型覆盖 | 领域 | 语言 | 获取 |
|---|---|---|---|---|---|
| **C-ReD** ⭐⭐⭐ | 128,610 条（13K 人类 + 115K AI） | 9 LLM：ChatGPT、Qwen、DeepSeek、Claude 等 | 新闻、QA、**影评**、学术写作、**作文** | 纯中文 | [GitHub](https://github.com/HeraldofLight/C-ReD) / [HF](https://huggingface.co/papers/2604.11796) |
| **RealDet** ⭐⭐⭐ | 836K+ 条（106K+ 人类） | **22 LLM**：GPT-4/4o、Claude 3.7、DeepSeek-V3、LLaMA2、ChatGLM2、Baichuan、Vicuna 等 | 15 域含 **Story Generation**、QA、新闻、评论、学术 | 中英双语 | [HF](https://huggingface.co/datasets/koakuma/RealDet) |
| **HC3-Chinese** ⭐⭐ | 25,706 QA 对 | ChatGPT (GPT-3.5) | QA、百科、金融、医疗、法律、心理 | 纯中文 | [HF](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) |
| **LLM-Detector** ⭐⭐ | ~100K | ChatGPT、GPT-4、QWen-14B、ChatGLM2、Baichuan2、ERNIE-Bot 等 9 模型 | QA、新闻、百科 | 纯中文 | [HF](https://huggingface.co/datasets/QiYuan-tech/LLM-Detector) |
| **GenAI Detect Task 1** | COLING 2025 共享任务 | 多模型 | 多域（含中文子集） | 多语言 | [ACL Anthology](https://aclanthology.org/2025.genaidetect-1.27/) |

> **推荐策略**：C-ReD 和 RealDet 互补使用。C-ReD 的"作文"和"影评"子集最接近叙事文本；RealDet 的 Story Generation 子集直接命中叙事场景，且覆盖模型最广（22 个）。HC3-Chinese 作为最早的 baseline 对比参考。

#### 5.1.2 纯人类叙事文本（正样本补充 + Track A 语料）

| 数据集 | 规模 | 类型 | 标注信息 | 获取难度 |
|---|---|---|---|---|
| **CNNSum** ⭐⭐⭐ | 695 篇小说摘录（最长 190K 字符） | 历史、武侠、科幻等，含人工摘要 | 摘要 | ⭐ 开放 CC-BY-4.0 |
| **WebNovelBench** ⭐⭐⭐ | 4,000 部网络小说 | 网络文学多类型，章节级 | LLM 评分 + 人类评分 | ⭐ 开放 |
| **MultiGenre-ChineseNovel** ⭐⭐ | 260 部小说，13 类型，105K 句 | 仙侠、悬疑、都市、科幻等 | NER 实体标注 | ⭐ 开放 |
| **LFED** ⭐⭐ | 95 部文学作品 + 1,304 QA 题 | 跨世纪中外文学（中文） | 阅读理解题 | ⭐ 开放 [GitHub](https://github.com/tjunlp-lab/LFED.git) |
| **MNBVC** ⭐ | 超大规模中文语料 | 含大量小说子集，需自行筛选 | 无 | ⭐ 开放 |
| **GuoFeng-Webnovel** | 179 部小说，中英对齐 | 14 类型网络小说 | 句级对齐 | ⚠️ 需注册 |
| **Qidian-Webnovel Corpus** | 110 部小说 + 读者评论 | 起点+Webnovel 中英对照 | 读者互动 | ⚠️ 需机构授权 |

> **推荐策略**：CNNSum 是最方便的人类叙事正样本来源（开放、长篇、多类型）。WebNovelBench 额外提供了人类与 LLM 评分的对比，对评估器训练有直接价值。

#### 5.1.3 叙事时间标注资源（Track A 标注规范参考）

| 资源 | 内容 | 语言 | 获取 |
|---|---|---|---|
| **NarrativeTime / TimeBankNT** | 密集时间线标注，全 TLINK 覆盖，双人标注 | 英文 | [ACL Anthology](https://aclanthology.org/2024.lrec-main.1054/) |
| **MATRES** | 事件 BEFORE/AFTER/EQUAL/VAGUE 关系 | 英文 | [GitHub 开放](https://github.com/CogComp/MATRES) |
| **TimeBankDense** | 经典时间关系语料（NarrativeTime 的基座） | 英文 | LDC 可获取 |
| **SANTA 2** | 叙事层级、时间特征、主观叙述风格标注指南 | 英文 | [开放获取](https://culturalanalytics.org/article/id/1277/) |

> 这些资源主要是英文，用于机制预验证和标注规范参考，不能替代中文叙事数据。

#### 5.1.4 预训练检测模型（可直接测试 baseline）

| 模型 | 说明 | 获取 |
|---|---|---|
| **QiYuan-tech/LLM-Detector** | 中文 AI 文本检测器，已开源，指令微调版 | [HF](https://huggingface.co/QiYuan-tech/LLM-Detector) |

### 5.2 数据组合方案

评估器训练数据按以下优先级组合：

```
Layer 1（核心训练集）：C-ReD 作文+影评子集 + RealDet Story Generation 子集
Layer 2（泛化补充）：C-ReD 全量 + RealDet 全量（中英）+ HC3-Chinese
Layer 3（纯叙事正样本）：CNNSum 人类小说摘录
Layer 4（自建数据）：用当前最强模型（DeepSeek-V3、Qwen3、Claude 等）
                     对同一批中文叙事 prompt 生成文本作为负样本
```

### 5.3 数据需求对照

| 数据类型 | 验证阶段目标 | 现成资源满足度 |
|---|---|---|
| 人类写作叙事文本 | 500–1000 篇 | ✅ C-ReD 作文 + CNNSum 即可满足 |
| 多模型生成叙事文本 | 3–5 模型 × 200 篇 | ✅ RealDet 22 模型 + C-ReD 9 模型远超需求 |
| 人工评分数据 | 200–500 篇 × 3 人 | ⚠️ 现成评分数据有限，需自行收集或利用 WebNovelBench 评分 |
| 图灵测试数据 | 100 篇 × 多人评价对 | ❌ 需自行构建，无现成资源 |

### 5.4 生成模型覆盖

为了保证评估器的泛化能力，负样本应覆盖多种来源（RealDet + C-ReD 已覆盖大部分）：

- **小模型**：GPT-2 级别、ChatGLM2-6B、Baichuan-7B ✅ RealDet/C-ReD
- **中等模型**：Qwen2.5-7B、LLaMA3-8B、Vicuna-13B ✅ RealDet
- **大模型**：GPT-4/4o、Claude 3.7、DeepSeek-V3、QWen-14B ✅ RealDet/C-ReD
- **不同解码策略**：temperature、top-p、top-k 的不同组合 ⚠️ 需自行生成
- **不同提示策略**：zero-shot、few-shot、带/不带叙事约束 ⚠️ 需自行构建 prompt

### 5.5 防止数据泄漏

- 按**作品/来源**划分 train/dev/test，而非随机切分。
- 同一 prompt 的不同模型生成结果保留在同一 split 中。
- 训练集中出现的生成模型，在测试集中不出现（**模型级留出**），这是对泛化能力的关键测试。RealDet 和 C-ReD 模型覆盖广，可选取部分模型做留出测试。
- 图灵测试数据与评估器训练数据严格隔离。

## 6. 评测设计：如何评测评测器 / Meta-Evaluation

### 6.1 评估者图灵测试（主要指标）

```text
设置：
  - 100 篇测试文本
  - 每条文本同时获得：人类评委评分 + 评估器评分
  - 混合后由 N 位人类裁判判断每条评价的来源

指标：
  - 裁判区分准确率（期望 ≈ 50% = 无法区分）
  - 裁判置信度分布（人类评价和机器评价的置信度分布应重叠）
  - 裁判间一致性（裁判之间对"这是人评的还是机评的"是否一致）
```

### 6.2 与人类评价的对齐度

| 指标 | 含义 |
|---|---|
| Pearson / Spearman r | 评估器分数与人类平均分的相关性 |
| Kendall's tau | 评估器排序与人类排序的一致性 |
| Exact match (二分类) | 人机标签判断一致率 |
| Mean Absolute Error | 评分场景下的平均绝对误差 |
| 人类间一致性上界 | 多人评分的平均 pairwise 相关性，作为评估器的理论上界 |

### 6.3 表面特征依赖检测

- **叙事时间扰动测试**：对同一文本只改变叙事时间结构（调换段落顺序、插入倒叙标记等），评测评估器分数是否敏感变化。
- **措辞混淆测试**：用同义改写保持叙事结构不变，评测评估器分数是否稳定。
- **跨解码策略泛化**：同一模型用不同 temperature 生成的文本，评估器应给出相似的排序而非仅凭"流畅度"打分。

### 6.4 对抗鲁棒性

- 用评估器作为 reward 进行 RL 训练，检查生成模型是否能"刷高"评估器分但降低人类评分。
- 若出现明显解耦，说明评估器学到的是可被利用的表面特征，需要重新训练。

## 7. 与叙事时间场项目的关系 / Relationship with Narrative Time Project

本评估器项目是叙事时间场项目（Track A）的**并行先行项目**（Track B）：

```text
                    Track B（先行）
              类人叙事评估器
         ┌──────────────────────┐
         │  Phase B1–B3         │
         │  训练 → 图灵测试 → 部署  │
         └──────────┬───────────┘
                    │
                    │  评估器通过图灵测试后
                    │  作为自动化评价指标
                    ▼
         ┌──────────────────────┐
         │  Track A（叙事时间场）   │
         │  Phase 3–7            │
         │  baseline → 时间场 → 生成 │
         └──────────┬───────────┘
                    │
                    │  用评估器自动评价
                    │  叙事时间模型的生成质量
                    ▼
         ┌──────────────────────┐
         │  交叉验证              │
         │  - 评估器对时间操作的敏感度 │
         │  - 时间场模型生成的文本    │
         │    是否在评估器下显著更好  │
         └──────────────────────┘
```

**关键时间节点**：
- 评估器应在 Track A 进入生成阶段（Phase 6）之前通过图灵测试验证。
- 这样 Track A 从第一个生成 baseline 开始，就可以用评估器作为自动化评价指标，无需等待人工评价。

## 8. 行动大纲 / Action Plan

### Phase B0 — 文献与基线调研

**工作**

- 系统调研 AI-generated text detection 文献（包括项目目录中两篇 PDF）。
- 调研现有中文 AI 文本检测工具和基准（GLTR、DetectGPT、GPTZero 等的中文对应方法）。
- **下载并测试现有检测器**：使用 QiYuan-tech/LLM-Detector 在 C-ReD 和 RealDet 的叙事相关子集上测试 baseline 性能。
- 明确评估器图灵测试的具体实验方案。
- **数据获取**：下载 C-ReD（作文+影评子集）、RealDet（Story Generation 子集）、CNNSum 三份核心数据。

**交付物**

- `docs/evaluator_literature_review.md`
- 现有检测器在中文叙事文本上的 baseline 指标（F1、AUC）。
- 已清洗并分好 train/dev/test 的数据集。
- 图灵测试实验方案。

**完成门槛**：确定了"当前最好方法离图灵测试通过还有多远"，数据准备就绪。

### Phase B1 — 建立二分类基线评估器

**工作**

- **组合训练数据**：
  - Layer 1：C-ReD 作文+影评子集 + RealDet Story Generation 子集
  - Layer 2（如需要）：C-ReD 全量 + HC3-Chinese
  - 正样本补充：CNNSum 人类小说摘录
- **自建叙事生成数据**：用 DeepSeek-V3、Qwen3、Claude 等当前最强模型，对中文叙事 prompt（从 CNNSum 摘要反推）生成文本，作为高质量负样本。
- 实现 MacBERT 二分类判别器（支持方案 B 评分回归器的接口预留）。
- 建立可复现训练和评测流程。
- 在模型级留出测试上评估（确保测试集中的生成模型未在训练中出现）。

**交付物**

- `src/evaluator/` 代码。
- 首个二分类评估器模型。
- 在留出测试集上的 baseline 指标（F1、AUC、Precision/Recall）。

**完成门槛**：二分类器在模型级留出测试上的 F1 显著优于随机基线，AUC > 0.8。

### Phase B2 — 图灵测试验证

**工作**

- 收集多位人类评委对测试文本的评价/分类。
- 执行评估者图灵测试实验。
- 分析评估器在哪些类型的文本/评价上最容易被识破。
- 根据失败模式迭代改进评估器。

**交付物**

- 图灵测试实验报告。
- 评估器失败案例分析。
- 改进后的评估器 v2（如需要多轮迭代）。

**完成门槛**：评估器通过图灵测试，或明确了当前方法的天花板和下一步方向。

### Phase B3 — 部署为叙事时间场评价指标

**工作**

- 将评估器集成到叙事时间场项目的评测 pipeline 中。
- 对叙事时间场模型的生成结果（baseline、input-only、deep-injection）进行自动评价。
- 与人工评价对比，验证评估器在叙事时间维度的敏感度。

**交付物**

- 自动化评价 pipeline。
- 叙事时间场模型的评估器评分报告。
- 评估器评分 vs 人工评分的对比分析。

**完成门槛**：评估器能可靠地区分不同叙事时间注入方式带来的生成质量差异。

### Phase B4 — 升级为多维叙事审美评估器（扩展阶段）

**工作**

- 从二分类升级为多维评分（叙事时间结构、情节连贯性、语言美感等）。
- 在评分维度中显式加入叙事时间子维度。
- 探索评估器作为生成模型 reward model 的可行性。

**交付物**

- 多维叙事审美评估器。
- RL reward model 实验报告。

## 9. 最近迭代 / Next Iterations

### Iteration B1：最小闭环（对应 Phase B0）

1. 下载 C-ReD、RealDet、CNNSum 三份核心数据。
2. 筛选叙事相关子集（C-ReD 作文+影评、RealDet Story Generation）。
3. 用 QiYuan-tech/LLM-Detector 在筛选后的数据上测试 baseline。
4. 记录 baseline 指标并与论文报告值对比。
5. 设计图灵测试具体实验方案（裁判人数、评价界面、统计检验方法）。

### Iteration B2：训练首个评估器（对应 Phase B1）

1. 组合训练集（C-ReD + RealDet + CNNSum），按模型级留出划分 train/dev/test。
2. 用 DeepSeek-V3、Qwen3 等对 CNNSum 摘要反推的 prompt 生成叙事文本作为补充负样本。
3. 实现并训练 MacBERT 二分类判别器。
4. 在模型级留出测试上评估 F1 + AUC。
5. 与 Iteration B1 的 baseline 对比，分析提升来源。

### Iteration B3：图灵测试尝试（对应 Phase B2）

1. 收集 5–10 位人类评委的评价数据。
2. 执行第一轮图灵测试。
3. 分析失败模式，决定优先改进方向。
4. 如果通过 → 进入 B4；如果未通过 → 诊断原因，迭代评估器。

### Iteration B4：部署对接 Track A（对应 Phase B3）

## 10. 当前风险 / Key Risks

- **评估器学到表面特征**：通过叙事时间扰动测试和解耦探针检测，若确认只学到表面特征，需要重新设计训练策略（对抗训练、多粒度输入等）。
- **人类评委质量参差**：需制定评委筛选标准、提供评价指南，并计算评委间一致性作为理论上界。
- **中文叙事数据**：C-ReD、RealDet、CNNSum 等已提供可用的叙事相关数据，但专门的"中文叙事文本 + 多模型生成 + 人工质量评分"三位一体的数据集目前不存在，需自行组合构建。图灵测试数据必须从零构建。
- **图灵测试通过门槛模糊**：需预先确定统计检验方法（如 equivalence test），避免"不显著=无差异"的解读陷阱。
- **生成模型快速迭代**：新模型生成的文本质量不断提升，评估器需要持续更新训练数据以保持判别力。
- **评估器被"刷分"**：若后期使用评估器作为 reward model，需持续监控评估器分数与人类评分的相关性，防止解耦。

## 11. 参考文献与数据资源 / References & Data Resources

### 项目相关 PDF

- *Everyone prefers human writers, including AI*（项目目录中 PDF）：人类对 AI 文本的偏好与检测研究。
- *Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI*（项目目录中 PDF）：多语言环境下人类对 AI 文本的检测和偏好。

### 人机文本检测核心数据论文

- **C-ReD**: Ye et al. *C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts*. ACL 2026 Findings. https://arxiv.org/abs/2604.11796 — 128K 条，9 模型，5 领域，纯中文。
- **RealDet**: *RealDet: A Large-Scale Benchmark for AI-Generated Text Detection*. https://huggingface.co/datasets/koakuma/RealDet — 836K+ 条，22 模型，15 域，中英双语。
- **HC3**: Guo et al. *HC3: Human ChatGPT Comparison Corpus*. https://arxiv.org/abs/2301.07597 — 最早的中文人机对比语料，25K QA 对。
- **LLM-Detector**: *Improving AI-Generated Chinese Text Detection with Open-Source LLM Instruction Tuning*. https://arxiv.org/abs/2402.01158 — 中文检测模型与数据集。
- **GenAI Detect Task 1**: Wang et al. *GenAI Content Detection Task 1: English and Multilingual Machine-Generated Text Detection: AI vs. Human*. COLING 2025. https://aclanthology.org/2025.genaidetect-1.27/

### AI 文本检测方法

- Mitchell et al. *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*. https://arxiv.org/abs/2301.11305
- Gehrmann et al. *GLTR: Statistical Detection and Visualization of Generated Text*. https://arxiv.org/abs/1906.04043
- Solaiman et al. *Release Strategies and the Social Impacts of Language Models*. https://arxiv.org/abs/1908.09203
- **MultiSocial**: Macko et al. *MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts*. ACL 2025. https://aclanthology.org/2025.acl-long.36/

### 中文叙事文本数据

- **CNNSum**: *CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels*. ACL 2025 Findings. https://arxiv.org/abs/2412.02819 — 695 篇小说摘录 + 人工摘要，CC-BY-4.0。
- **WebNovelBench**: *WebNovelBench: Placing LLM Novelists on the Web Novel Distribution*. https://arxiv.org/abs/2505.14818 — 4,000 部网络小说 + 人机评分对比。
- **MultiGenre-ChineseNovel**: Zhao et al. *MultiGenre-ChineseNovel: A Genre-oriented NER Corpus of Chinese Novels*. PACLIC 2023. https://github.com/hjzhao73/MultiGenre-ChineseNovel — 260 部小说，13 类型。
- **LFED**: *LFED: A Literary Fiction Evaluation Dataset for Large Language Models*. LREC 2024. https://arxiv.org/abs/2405.10166 — 95 部文学作品 + 1,304 QA 题。
- **GuoFeng-Webnovel**: *GuoFeng-Webnovel: Multilingual Corpus of Web Fiction*. https://github.com/longyuewangdcu/GuoFeng-Webnovel — 179 部小说中英对齐。
- **Qidian-Webnovel Corpus**: Yu et al. *Qidian-Webnovel Corpus: A Dataset of Chinese Web Novels with Multilingual Reader Response*. Journal of Open Humanities Data. https://doi.org/10.5334/johd.368
- **COIG-Writer**: *COIG-Writer: A High-Quality Dataset for Chinese Creative Writing with Thought Processes*. https://arxiv.org/abs/2510.14763 — 1,665 篇 AI 生成中文创意写作，51 类型。
- **MNBVC**: 超大规模中文语料集。https://github.com/esbatmop/MNBVC — 含大量小说子集。

### 中文 AI 文本检测 benchmark

- *Benchmarking the Detection of LLMs-Generated Modern Chinese Poetry*. EMNLP 2025 Findings. https://aclanthology.org/2025.findings-emnlp.507/
- *MLSDET: Multi-LLM Statistical Deep Ensemble for Chinese AI-Generated Text Detection*.

### 文本质量评估与图灵测试

- Hashimoto et al. *Unifying Human and Statistical Evaluation for Natural Language Generation*. https://arxiv.org/abs/1904.02792
- Clark et al. *All That's 'Human' Is Not Gold: Evaluating Human Evaluation of Generated Text*. https://aclanthology.org/2021.acl-long.565/
- Chiang & Lee. *Can Large Language Models Be an Alternative to Human Evaluations?* https://aclanthology.org/2023.acl-long.870/
- Liu et al. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. https://arxiv.org/abs/2303.16634

### 叙事评价

- Siu et al. *Evaluating Human-Language Model Interaction*. https://arxiv.org/abs/2212.09746
- Chakrabarty et al. *Art or Artifice? Large Language Models and the False Promise of Creativity*. https://arxiv.org/abs/2309.14556

### Track A 叙事时间标注资源（交叉参考）

- Rogers et al. *NarrativeTime: Dense Temporal Annotation on a Timeline*. LREC 2024. https://aclanthology.org/2024.lrec-main.1054/
- MATRES: https://github.com/CogComp/MATRES
- Kearns. *SANTA 2 Annotation Guidelines*. https://culturalanalytics.org/article/id/1277/

---

**当前项目状态 / Current status:** 概念草案已建立。这是叙事时间 Transformer（Track A）的并行先行项目（Track B）。评估器应在 Track A 进入生成阶段之前通过图灵测试验证，以便作为自动化评价指标使用。下一项具体工作：测试现有 AI 文本检测器在中文叙事文本上的表现，建立二分类基线。
