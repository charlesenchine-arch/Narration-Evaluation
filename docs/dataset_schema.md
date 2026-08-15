# 自建叙事评估数据集 schema

> 解决解耦实验与图灵测试的数据缺口（短文人类参照、同一文本像人分+质量分）。

## 1. 数据形态

JSONL，每行一条短文叙事样本：

```json
{
  "id": "H_0001",
  "text": "……短文叙事（300-800字）……",
  "label": "human | machine",
  "gen_model": "deepseek-v3 | qwen3 | claude | ...",   // machine 时记录生成模型
  "prompt": "……",                                       // machine 时记录生成 prompt
  "length_bucket": "short | medium | long",
  "quality_scores": [4, 3, 5],                          // 多人评分（1-5）
  "quality_mean": 4.0,
  "rater_ids": ["r1", "r2", "r3"]
}
```

## 2. 规模（用户确认：小规模 600 条）

| 子集 | 数量 | label | 来源 |
|---|---|---|---|
| H 短文叙事 | 300 | human | 短篇小说/散文（需收集） |
| G 短文叙事 | 300 | machine | LLM 生成（DeepSeek/Claude 等） |

## 3. 质量评分

- 维度：整体叙事质量（1-5 整数，含 .5 可选）
- 评分者：3-5 人，每人独立评分，不标注来源
- 覆盖：H 300 + G 300 全部评分（600 条 × 3 人 = 1800 次评分）
- 输出：`quality_scores` 数组 + `quality_mean`

## 4. 用途对照

| 用途 | 需要字段 | 对应实验 |
|---|---|---|
| 解耦正交性 | label + quality_mean | 像人分 vs 质量分 Spearman |
| 区分度分离 | label + quality_mean | 像人分 AUC vs 质量分 AUC |
| 短文参照 | 300 字级 | 构造证据 in-distribution |
| 图灵测试 | quality_scores + 评价理由 | 人评 vs 机评判别 |

## 5. 目录

```
data/eval_dataset/
  raw/                 # 原始文本（H 收集、G 生成）
  scored/              # 带评分后的 JSONL
  prompts/             # G 生成的叙事 prompt
```

## 6. 评分指南（草稿，供评分者）

> 给一篇 300-800 字的叙事文本打分（1-5）：
> 5 = 有吸引力的叙事，结构完整，语言生动
> 4 = 良好的叙事，偶有小瑕疵
> 3 = 完整但平淡，无明显错误
> 2 = 结构松散或语言生硬，可读性差
> 1 = 严重缺陷，几乎不可读
