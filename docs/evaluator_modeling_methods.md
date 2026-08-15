# 评估模型建模方法：文献综述

> Evaluator Modeling Methods — Literature Review for Track B

## 1. 综述范围

本文综述了可用于构建"类人叙事评估器"的建模方法，覆盖以下领域：

- **方法 1**：微调 Encoder-Only 分类器（BERT/RoBERTa/MacBERT + 分类头）
- **方法 2**：指令微调 Decoder-Only 检测器（LLM Instruction Tuning）
- **方法 3**：对抗鲁棒检测（Adversarial Training + Paraphraser）
- **方法 4**：LLM-as-a-Judge 生成式评估（Prompt-based / G-Eval 类）
- **方法 5**：故事/叙事专用自动评价指标（UNION、DELTASCORE、COHESENTIA 等）

每个方法介绍其核心思路、代表论文、优缺点及对本项目的适用性。

---

## 2. 方法分类与代表论文

### 2.1 微调 Encoder-Only 分类器

#### 核心思路

使用预训练的 Encoder-only 模型（BERT、RoBERTa、XLM-RoBERTa、MacBERT 等），在 `[CLS]` token 之上接一个二分类或多分类头，用标注的人机文本数据进行微调。这是目前**最成熟、最稳定、训练成本最低**的方法。

#### 代表论文

| 论文 | 会议 | 要点 |
|---|---|---|
| **SemEval-2024 Task 8 Overview** (Wang et al.) | SemEval 2024 | 定义了三项子任务：二分类（Subtask A）、模型溯源（Subtask B）、人机切换点检测（Subtask C）。126 队参加英文子任务，59 队多语言。**最佳系统在 Subtask A 英文上达 Accuracy > 99%**（域内），但跨域/跨模型泛化显著下降。 |
| **NCL-UoR at SemEval-2024 Task 8** | SemEval 2024 | 微调 LLM（如 LLaMA）进行分类。发现 encoder-only 模型在域内表现好但 OOD 泛化差，LLM 微调后泛化更好。 |
| **L3i++ at SemEval-2024 Task 8** | SemEval 2024 | 系统性比较了 fine-tuned LLM 在多生成器、多域、多语言场景下的表现。 |
| **FI Group at SemEval-2024 Task 8** | SemEval 2024 | 提出句法驱动的架构，在多语言检测中使用句法特征增强。 |
| **MasonTigers at SemEval-2024 Task 8** | SemEval 2024 | 对多种 Transformer 模型（BERT、RoBERTa、XLNet、DeBERTa 等）在检测任务上的性能进行了系统比较分析。 |
| **LLM-Detector** (QiYuan-tech) | arXiv 2024 | 使用指令微调的开源 LLM 做中文文本检测，指出 BERT/RoBERTa "容易域内过拟合，OOD 检测表现差"。提出了句子级和文档级双粒度检测。已开源模型和数据集。 |
| **Koike et al. OUTFOX** | AAAI 2024 | 使用 In-Context Learning + 对抗生成样本做论文级检测。关键创新：用检测器的反馈指导对抗样本生成，提升检测器鲁棒性。 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| 训练快、显存低，适合 RTX 4060 | 容易学到表面统计特征（域内过拟合） |
| 生态成熟（transformers 库直接可用） | 跨模型/跨域泛化能力有限 |
| 可解释性好（可训练探针分析中间层） | 对改写/对抗攻击鲁棒性不足 |
| 有大量已开源模型可作 baseline | 输出仅为二分类概率，难以做细粒度评分 |

#### 对本项目的适用性

⭐⭐⭐⭐⭐ **Phase B1 首选方案**。建议路径：

1. MacBERT-base + 分类头作为 baseline 架构
2. 训练数据使用 C-ReD（作文+影评子集）+ RealDet（Story Generation 子集）
3. 在模型级留出测试上评估泛化能力
4. 后续升级为多任务回归头（方案 B）

---

### 2.2 对抗鲁棒检测

#### 核心思路

标准分类器容易被改写（paraphrase）攻破。对抗鲁棒方法通过对抗训练（adversarial training）或在训练中引入改写样本，使检测器对文本表面变化不敏感，从而更关注深层语义和叙事结构。

#### 代表论文

| 论文 | 会议 | 要点 |
|---|---|---|
| **RADAR** (Hu et al.) | NeurIPS 2023 | 使用 paraphraser + detector 联合对抗训练。paraphraser 学习生成能欺骗 detector 的改写，detector 学习不被欺骗。在 8 个 LLM × 4 个数据集上显著优于非对抗方法。展示了对未见 LLM 的强迁移能力。 |
| **OUTFOX** (Koike et al.) | AAAI 2024 | 针对论文级检测。通过 In-Context Learning + 对抗生成样本，让检测器学习区分"被改写过的 AI 文本"和"真正的人类文本"。 |
| **Humanizing MGC** (Zhou et al.) | LREC 2024 | 系统研究如何通过对抗攻击使 AI 文本逃避检测，从反面揭示了检测器的脆弱性和改进方向。 |
| **Robust AI-Generated Text Detection by Restricted Embeddings** | arXiv 2024 | 提出在受限嵌入空间中训练检测器，提高对改写攻击的鲁棒性。 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| 对改写/对抗攻击鲁棒性强 | 训练复杂度高（需要同时训练 paraphraser 和 detector） |
| 迫使模型关注深层特征而非表面统计 | 对抗训练可能不稳定 |
| 与"防止学到表面特征"目标高度一致 | 在 RTX 4060 上训练完整 RADAR 可能有显存压力 |

#### 对本项目的适用性

⭐⭐⭐⭐ **Phase B1 后期 / B2 改进阶段推荐**。具体方案：

1. 不直接复现 RADAR（训练成本高），而是采用**对抗数据增强**策略
2. 使用轻量改写模型（如小型 T5）对训练集中的 AI 文本做改写，作为额外负样本
3. 叙事时间扰动（调换段落顺序、插入/删除时间标记）也可视为一种领域特化的对抗增强
4. 这与 §4.2 的"增强策略"直接对应

---

### 2.3 指令微调 Decoder-Only 检测器

#### 核心思路

不满足于 `[CLS]` token 分类，而是用 decoder-only LLM（如 Qwen、LLaMA），通过指令微调（instruction tuning）让模型直接生成"这是人类写的"或"这是 AI 生成的，因为……"的判断。可利用 LLM 预训练阶段积累的知识提升泛化能力。

#### 代表论文

| 论文 | 会议 | 要点 |
|---|---|---|
| **LLM-Detector** (QiYuan-tech) | arXiv 2024 | 针对中文，9 种 LLM 生成的数据，指令微调开源 LLM。文档级 + 句子级双粒度。显著优于 BERT/RoBERTa baseline（尤其在 OOD 场景）。已开源。 |
| **NCL-UoR at SemEval-2024 Task 8** | SemEval 2024 | LLM 微调后进行检测分类，在跨域场景优于 encoder-only。 |
| **DetectGPT** (Mitchell et al.) | arXiv 2023 | 零样本方法。利用"模型生成文本倾向于落在自身概率函数的负曲率区域"这一假设，不需要训练分类器。 |
| **GLTR** (Gehrmann et al.) | ACL 2019 | 最早的统计检测方法之一，可视化 token 级别概率排名。简单但有效。 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| 泛化能力通常优于 encoder-only | 训练和推理显存高（7B 模型在 RTX 4060 上紧张） |
| 可输出可解释的判断理由 | 指令微调数据构造复杂 |
| 天然支持细粒度分析（句子级、维度级） | 不如分类器成熟，调试困难 |

#### 对本项目的适用性

⭐⭐⭐ **方案 B（多维评分回归器）的候选架构**。建议：

- Phase B1 先用 MacBERT 分类器快速验证
- Phase B4 考虑升级为指令微调 Qwen2.5-7B 等多维评分器
- 可与 Encoder-Only 分类器做集成

---

### 2.4 LLM-as-a-Judge 生成式评估

#### 核心思路

不训练专门的检测模型，而是用最强大的 LLM（GPT-4、Claude 等）作为"评委"，通过精心设计的 prompt 对文本进行多维度评价。这是当前 NLG 评估领域最活跃的方向，核心问题是**prompt 设计**和**与人类评价的对齐**。

#### 代表论文

| 论文 | 会议 | 要点 |
|---|---|---|
| **G-Eval** (Liu et al.) | EMNLP 2023 | 使用 GPT-4 + Chain-of-Thought prompt 进行 NLG 评估。在摘要和对话任务上达到与人类评价 Spearman r > 0.5 的相关性。通过 Auto-CoT 生成评估步骤，不需要人工设计评分标准。 |
| **Chiang & Lee** (Can LLMs Be an Alternative to Human Evaluations?) | ACL 2023 | 系统比较了多种 LLM 作为评估者的表现，发现 GPT-4 在某些维度（流畅度、连贯性）接近人类，但在创造性、趣味性等维度差距仍大。 |
| **RevisEval** (Zhang et al.) | ICLR 2025 | 提出通过"响应自适应参考"改进 LLM-as-a-Judge，减少评分偏差。 |
| **LLM-Rubric** (Microsoft) | GitHub 开源 | 使用少量人工标注样例作为评分参照标准（rubric），让 LLM 参考这些标准打分，提升与人类的一致性。 |
| **Clark et al.** (All That's 'Human' Is Not Gold) | ACL 2021 | 警示论文：人类评价本身质量参差不齐，需要仔细设计评价协议。是设计图灵测试实验的重要参考。 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| 不需要训练数据，开箱即用 | 依赖外部 API（成本、延迟、隐私） |
| 可输出多维度细粒度评价 | prompt 敏感，评分可能不稳定 |
| 当前最接近人类审美标准的方法 | 对大模型本身"AI 味"是否敏感未知 |
| 可直接用于图灵测试中生成"机器评价" | 评分逻辑不透明（黑盒） |

#### 对本项目的适用性

⭐⭐⭐⭐ **图灵测试中的"机器评价"生成器 + 评估器训练数据的标注辅助**。

1. 在图灵测试中，LLM-as-a-Judge 本身就是"评估者图灵测试"的参试者（"机器评价"一方）
2. 可用于自动标注训练数据，减少人工标注成本
3. 可作为评估器的评分上界参考（"GPT-4 能做到什么程度？"）
4. 但最终目标是一个**独立的小型评估器**，不依赖外部 API

---

### 2.5 故事/叙事专用自动评价指标

#### 核心思路

传统 NLG 指标（BLEU、ROUGE 等）完全不适合叙事文本评价。近年出现了一批专为故事生成设计的自动评价指标，它们评估的不再是 n-gram 重叠，而是连贯性、一致性、趣味性等叙事特有维度。

#### 代表论文

| 论文 | 会议 | 要点 |
|---|---|---|
| **What Makes a Good Story? A Comprehensive Survey** | arXiv 2024 | 故事评价的系统综述，覆盖传统指标、LLM 指标和人工评价三大类。是本项目评估维度设计的核心参考。 |
| **UNION** (Guan & Huang) | EMNLP 2020 | 无参考故事评价指标。通过对比学习区分"好故事"和"坏故事"，不需要人类写参考文本。 |
| **DELTASCORE** | EMNLP 2023 | 通过细粒度扰动（删除、替换、调序句子）测量故事质量变化。不关注表面措辞，关注结构完整性。 |
| **COHESENTIA** | EMNLP 2023 | 专为叙事连贯性设计的自动评价指标。 |
| **PERSE** | 2024 | 学习个性化故事评价——不同读者有不同的审美标准。与本项目的"评估者图灵测试"思路高度相关。 |
| **CML-Bench** | arXiv 2025 | 用于评估 LLM 生成电影剧本的框架，覆盖叙事一致性、角色发展等维度。 |
| **OpenMEVA** | EMNLP 2022 | 故事评价 benchmark，提供了多个指标在 ROC 故事上的 Pearson 相关性基准。 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| 专为叙事设计，评估维度与项目目标对齐 | 大多基于英文故事，中文适配需额外工作 |
| 无参考评价（UNION、DELTASCORE）特别适合生成场景 | 单指标覆盖维度有限，需要组合使用 |
| 可作为评估器训练的信号或辅助特征 | 一些方法依赖 GPT-4 等大模型，轻量化困难 |

#### 对本项目的适用性

⭐⭐⭐⭐ **评估器"叙事敏感度"验证 + 多维评分维度设计参考**。

1. DELTASCORE 的"扰动→质量变化"思路可直接用于 §6.3 的"叙事时间扰动测试"
2. UNION 的对比学习框架可参考用于训练评估器的排序损失
3. OpenMEVA 的 benchmark 设计可作为我们自建中文叙事评价 benchmark 的模板
4. PERSE 的个性化评价理念与"评估者图灵测试"的核心思想一致

---

## 3. 推荐技术路线

### Phase B1（最小闭环）

```
模型：MacBERT-base + 二分类头
数据：C-ReD（作文+影评） + RealDet（Story Generation）
训练目标：Binary Cross-Entropy
增强：对抗数据增强（轻量改写 + 叙事时间扰动）
评测：模型级留出 F1 + AUC
```

- 参考论文：SemEval-2024 Task 8 teams, LLM-Detector

### Phase B2（图灵测试验证 + 鲁棒性改进）

```
改进 1：引入对比学习辅助损失（参考 UNION）
改进 2：对抗训练 / 对抗数据增强（参考 RADAR 简化版）
改进 3：叙事时间扰动敏感性测试（参考 DELTASCORE 方法论）
```

- 参考论文：RADAR (NeurIPS 2023), OUTFOX (AAAI 2024), DELTASCORE (EMNLP 2023)

### Phase B4（多维评分升级）

```
模型：Qwen2.5-7B + 指令微调（或 MacBERT + 多任务回归头）
评分维度：叙事时间结构、情节连贯性、语言美感、人物一致性、整体质量
训练目标：多任务 MSE / Ordinal Regression
评测：与人类评分的 Spearman r，每个维度的 MAE
```

- 参考论文：LLM-Detector (2024), G-Eval (EMNLP 2023), PERSE (2024)

---

## 4. 关键参考论文速查表

| 论文 | 年份 | 会议/期刊 | 核心方法 | 对本项目的直接用途 |
|---|---|---|---|---|
| SemEval-2024 Task 8 Overview | 2024 | SemEval | 多域多模型多语言检测评测 | Baseline 方法设计 |
| LLM-Detector | 2024 | arXiv | 中文指令微调检测器 | Phase B1 直接 baseline |
| RADAR | 2023 | NeurIPS | 对抗训练鲁棒检测 | Phase B2 鲁棒性改进 |
| OUTFOX | 2024 | AAAI | 对抗 ICL 检测 | Phase B2 对抗增强 |
| G-Eval | 2023 | EMNLP | GPT-4 CoT 评估 | 图灵测试机器评价 + 标注辅助 |
| Can LLMs Be an Alternative to Human Evaluations? | 2023 | ACL | LLM 评价 vs 人类评价对比 | 图灵测试实验设计 |
| What Makes a Good Story? | 2024 | arXiv | 故事评价综述 | 多维评分维度设计 |
| UNION | 2020 | EMNLP | 对比学习故事评价 | 评估器训练 loss 设计 |
| DELTASCORE | 2023 | EMNLP | 扰动法故事评价 | 叙事时间扰动测试 |
| PERSE | 2024 | - | 个性化故事评价 | "评估者图灵测试"方法论 |
| OpenMEVA | 2022 | EMNLP | 故事评价 benchmark | 中文 benchmark 设计模板 |
| HC3 | 2023 | arXiv | 中英人机对比语料 | 训练数据 |
| C-ReD | 2026 | ACL Findings | 中文检测 benchmark | 训练+评测数据 |
| RealDet | - | - | 22 模型大规模检测数据 | 训练数据（Story Generation 子集） |
| DetectGPT | 2023 | arXiv | 零样本概率曲率检测 | 零样本 baseline 对比 |
| AI-Generated Text Detection Survey | 2025 | CL | 全面方法综述 | 整体技术背景 |

---

## 5. 参考文献（BibTeX 格式）

```bibtex
@inproceedings{wang2024semeval,
  title={SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection},
  author={Wang, Yuxia and others},
  booktitle={SemEval},
  year={2024},
  url={https://aclanthology.org/2024.semeval-1.279/}
}

@article{llmdetector2024,
  title={LLM-Detector: Improving AI-Generated Chinese Text Detection with Open-Source LLM Instruction Tuning},
  author={QiYuan-tech},
  year={2024},
  url={https://arxiv.org/abs/2402.01158}
}

@inproceedings{hu2023radar,
  title={RADAR: Robust AI-Text Detection via Adversarial Learning},
  author={Hu, Xiaomeng and Chen, Pin-Yu and Ho, Tsung-Yi},
  booktitle={NeurIPS},
  year={2023},
  url={https://arxiv.org/abs/2307.03838}
}

@inproceedings{koike2024outfox,
  title={OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples},
  author={Koike, Ryuto and Kaneko, Masahiro and Okazaki, Naoaki},
  booktitle={AAAI},
  year={2024},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/30120}
}

@inproceedings{liu2023geval,
  title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
  author={Liu, Yang and others},
  booktitle={EMNLP},
  year={2023},
  url={https://arxiv.org/abs/2303.16634}
}

@inproceedings{chiang2023can,
  title={Can Large Language Models Be an Alternative to Human Evaluations?},
  author={Chiang, Cheng-Han and Lee, Hung-yi},
  booktitle={ACL},
  year={2023},
  url={https://aclanthology.org/2023.acl-long.870/}
}

@inproceedings{guan2020union,
  title={UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation},
  author={Guan, Jian and Huang, Minlie},
  booktitle={EMNLP},
  year={2020}
}

@inproceedings{deltascore2023,
  title={DELTASCORE: Fine-Grained Story Evaluation with Perturbations},
  booktitle={EMNLP},
  year={2023}
}

@article{storysurvey2024,
  title={What Makes a Good Story and How Can We Measure It? A Comprehensive Survey of Story Evaluation},
  year={2024},
  url={https://arxiv.org/abs/2408.14622}
}

@inproceedings{openmeva2022,
  title={OpenMEVA: A Benchmark for Evaluating Open-ended Story Generation Metrics},
  booktitle={EMNLP},
  year={2022}
}

@inproceedings{guo2023hc3,
  title={HC3: Human ChatGPT Comparison Corpus},
  author={Guo, Biyang and others},
  booktitle={arXiv},
  year={2023},
  url={https://arxiv.org/abs/2301.07597}
}

@inproceedings{ye2026cred,
  title={C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts},
  author={Ye, Fengying and others},
  booktitle={ACL Findings},
  year={2026},
  url={https://arxiv.org/abs/2604.11796}
}

@article{survey2025cl,
  title={A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions},
  journal={Computational Linguistics},
  year={2025},
  url={https://aclanthology.org/2025.cl-1.8/}
}

@article{fraser2025jair,
  title={Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods},
  author={Fraser, Kathleen C. and others},
  journal={JAIR},
  year={2025},
  url={https://mlanthology.org/jair/2025/fraser2025jair-detecting/}
}

@inproceedings{clark2021all,
  title={All That's 'Human' Is Not Gold: Evaluating Human Evaluation of Generated Text},
  author={Clark, Elizabeth and others},
  booktitle={ACL},
  year={2021},
  url={https://aclanthology.org/2021.acl-long.565/}
}

@inproceedings{mitchell2023detectgpt,
  title={DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature},
  author={Mitchell, Eric and others},
  booktitle={arXiv},
  year={2023},
  url={https://arxiv.org/abs/2301.11305}
}
```

---

**文档版本**: v0.1  
**最后更新**: 2026-08-06  
**关联文件**: `NARRATIVE_EVALUATOR_PLAN.md` §4（模型设计）和 §5（数据策略）
