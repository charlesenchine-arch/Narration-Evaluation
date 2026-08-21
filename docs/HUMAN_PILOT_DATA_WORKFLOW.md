# 人类叙事候选库与机器文本生成流程（pilot v1）

> 本文是具体操作流程。数据集总体目标、硬性纳入标准、标注规范、切分规则和扩容门槛以 [`DATASET_CONSTRUCTION_REQUIREMENTS.md`](DATASET_CONSTRUCTION_REQUIREMENTS.md) 为上位规范；分阶段混池、自建文本与 API 生成安排见 [`DATASET_BUILD_PLAN.md`](DATASET_BUILD_PLAN.md)。若文档冲突，以上位规范为准。

> 按 v1.1 双池协议，本文现有外部成文与反向 Prompt 流程属于 Pool A 工程/训练候选。最终 Pool B 实验必须先冻结全新 Prompt，再让人类作者和 API 模型根据相同 Prompt 独立写作；Pool B 不用于训练。

## 1. 这套数据解决什么问题

本 pilot 的目的不是训练“人机检测器”，而是先建立一组可追溯的人类叙事文本，并为每篇文本构造一个不泄漏具体情节的中性写作 Prompt。审核后，再让多个免费的闭源聊天模型根据同一类 Prompt 生成机器文本。后续由盲审员独立判断叙事质量，再在相近质量带内比较人类文本与机器文本，检验质量评估器是否依赖来源捷径。

反向 Prompt 只是一种实验配对工具，并不等于人类作者当时真实收到的写作指令。因此，此类配对必须标为 `observational pairing`；它不能单独支持“同 Prompt 因果对照”的表述。

## 2. 当前候选库

当前默认生成 60 篇、每篇 500—1200 个中文字符：

- COIG-Writer：20 篇。保留多种叙事体裁；`answer` 被项目说明为原始人类作品，`query` 是反向生成的 Prompt。仓库声明 ODC-BY-1.0，但正式发布前仍需核对该数据库许可对单篇内容权利的覆盖范围并保留署名。来源：<https://github.com/COIG-Writer/COIG-Writer>
- STORAL：40 篇。完整文本按 `beginning + story` 重建，主题标签来自 `moral`。论文称其为从网页收集并经人工处理的人类故事。来源：<https://github.com/thu-coai/MoralStory>；论文：<https://aclanthology.org/2022.naacl-main.374/>

重要授权限制：截至 2026-08-21，STORAL 仓库和数据卡未给出明确的数据集级许可证。论文页面的 CC BY 4.0 只覆盖论文本身，不能自动覆盖底层故事。因此，STORAL 原文当前标为 `internal_review_only`，仅用于本地审核和内部实验；权利核验完成前，不应随 Git 仓库或论文附件再分发。

旧目录 `data/eval_dataset/raw/h_chunks.jsonl` 中的网文切块不进入本 pilot：它们既可能是不完整章节，也不适合作为可再分发的人类质量基准。

## 3. 自动筛选与可追溯性

`scripts/build_human_pilot.py` 会执行以下步骤：

1. 下载三个固定源文件，并校验 SHA-256；也可显式传入本地文件。
2. 规范空白，筛选 500—1200 字的候选。
3. 排除演讲、广告、评论、脚本、设定稿、续写、模仿已有作品、明显助手套话等非完整原创叙事。
4. 按体裁或长度桶做固定种子的平衡抽样。
5. 按规范化全文 SHA-256 精确去重。
6. 保存原始来源字段、授权状态、文本哈希、原始 Prompt、反向 Prompt 草案与泄漏风险。

默认构建命令：

```bash
cd /Users/johonnycake/Documents/Codex/Narration-Evaluation
python3 scripts/build_human_pilot.py
```

本地输出位于 `data/human_pilot/review/`：

- `human_candidates.sqlite3`：可查询的人类候选数据库。
- `human_candidates.jsonl`：训练/分析脚本使用的逐行 JSON。
- `human_candidates_review.csv`：人工审核的机器可读主表，UTF-8 with BOM，可直接用 Excel 打开。
- `manifest.json`：数据源文件哈希、抽样参数和样本数。

这些目录默认被 `.gitignore` 排除，防止把授权待核验的原文误推到 GitHub。

## 4. 反向 Prompt 的审核标准

审核表中每一行需要做两项判断：

1. `approve_text`：文本是否确为完整叙事、长度合适、无明显助手套话、不是提纲或续写片段。通过填 `yes`，排除填 `no`。
2. `approved_prompt`：保留或改写 Prompt 草案。合格 Prompt 可以说明体裁、宽泛主题和长度，但不得暴露原文独有的人名关系、事件顺序、关键转折、结局、标志性句子，也不得要求模仿具体作者或复述已有作品。

建议把 Prompt 控制在以下模板附近：

> 请创作一篇500—1200字的中文短篇故事，围绕“宽泛主题”展开。人物、冲突、转折与结局由你独立设计。作品必须是一篇完整叙事；不要写创作说明，不要模仿具体作者，也不要复述任何已知作品。

如果某篇文本只能依靠泄漏大量情节才能构造出“匹配”的 Prompt，应排除该文本，而不是让 Prompt 变成答案提纲。

## 5. 审核完成后生成任务队列

在 CSV 的 `approve_text` 列填好 `yes/no`，并完成 `approved_prompt` 后运行：

```bash
python3 scripts/finalize_human_pilot_review.py
```

输出位于 `data/human_pilot/private/`：

- `approved_human.jsonl`：只含审核通过的人类文本与最终 Prompt。
- `manual_chat_queue.csv`：每个 Prompt 分配给 ChatGPT Free、Gemini Free、Claude Free，默认每篇产生三条机器文本任务。

## 6. 用免费闭源聊天软件生成

手工生成时统一采用以下协议：

- 每条任务使用全新对话；关闭或不使用联网、工具、项目知识和长期记忆。
- 只粘贴 `prompt`，不粘贴配对的人类原文、来源 Prompt 或道德故事原文。
- 只收集首轮完整回答，不追问、不要求润色。
- 在队列表中记录界面显示的模型名称、生成日期、会话链接（若可得）与完整输出。
- 如果平台拒答、截断或输出明显少于长度下限，保留原始结果并标记状态；不要在同一对话中诱导补写。是否重试应由统一规则决定。
- 免费产品的默认模型与配额会变化，不能只写“ChatGPT/Gemini/Claude”；必须记录当天界面显示的具体模型或“未显示”。

这条手工路径满足“免费 + 闭源”，但消费级聊天产品的版本控制弱，因此适合作为 pilot 探针，不应作为唯一的正式基线。

## 7. 可选：Gemini Developer API 免费层

仓库提供 `scripts/generate_gemini_free.py`。它只允许当前审查过且对新用户开放的闭源模型 `gemini-3.6-flash`，默认 dry-run、每次最多预览 5 条，只有显式添加 `--execute` 才会发送请求。Google 已停止向新用户提供旧的 `gemini-2.5-flash`，因此不要把模型名改回旧版本。

先在自己的终端设置 key，不要把 key 发到聊天中，也不要写入仓库：

```bash
export GEMINI_API_KEY='你的本地API_KEY'
python3 scripts/generate_gemini_free.py --max-items 5
python3 scripts/generate_gemini_free.py --max-items 5 --execute
```

正式执行前必须自行确认 Google 项目使用 Gemini Developer API 免费层且未启用会产生费用的计费设置。API 无法替脚本判断项目是否绑定计费。官方定价页：<https://ai.google.dev/gemini-api/docs/pricing>；REST API：<https://ai.google.dev/api/generate-content>。

免费层提交的内容可能被服务商用于改进产品，所以脚本只发送审核后的 Prompt，不发送人类原文。不要把未公开、含隐私或无权提交的材料放入 Prompt。

## 8. 可选：通过 AIHubMix 聚合接口生成

仓库提供 `scripts/generate_aihubmix.py`，使用 AIHubMix 的 OpenAI 兼容接口。默认采用优选地址 `https://api.inferera.com/v1`；若需切换主地址，可临时设置 `AIHUBMIX_BASE_URL=https://aihubmix.com/v1`。只允许这两个已核对的 HTTPS 地址，防止密钥被误发给其他主机。

AIHubMix 是第三方聚合服务，不是 Gemini、OpenAI 等厂商的官方免费层。赠送余额可以让早期请求暂时不产生现金支出，但不同模型仍可能按 token 扣除平台余额，价格和可用模型也会变化。必须在[模型与价格页](https://api.inferera.com/models)核对当天显示的准确模型 ID 和价格；实验记录中保留请求模型、返回模型、token 用量、日期和接口地址。

为防止共享 Key 误扣余额，当前脚本硬性只允许 `gemini-3.6-flash-free`、`gemini-3.5-flash-lite-free` 和 `coding-kimi-k3-free` 三个经过核对的促销免费模型 ID；去掉 `-free` 后缀的同名模型会被拒绝。免费资源可能随时用尽或限流，因此后缀保护不能替代平台侧的余额与用量检查。

在 macOS 终端中把 key 存入系统钥匙串。输入时字符不会显示；不要把真实 key 写进命令、聊天、`.env` 或 Git：

```zsh
read -s "AIHUBMIX_KEY?粘贴 AIHubMix Key（输入不会显示）: "; echo
security add-generic-password -a "$USER" -s "narration-aihubmix-api" -w "$AIHUBMIX_KEY" -U
unset AIHUBMIX_KEY
```

先做不生成文本的连接检查：

```zsh
cd /Users/johonnycake/Documents/Codex/Narration-Evaluation
python3 scripts/generate_aihubmix.py --probe
```

然后从输出或模型页复制准确的模型 ID。第一条命令只预览将发送的 Prompt，不访问 API；第二条才真实生成一篇：

```zsh
python3 scripts/generate_aihubmix.py --model '准确的模型ID' --max-items 1
python3 scripts/generate_aihubmix.py --model '准确的模型ID' --max-items 1 --execute --acknowledge-billing
```

脚本默认传入 `reasoning_effort=minimal`。对 Gemini 3.x，这会把思考降到最低，但不保证完全关闭；不要使用旧的 `thinking_budget=0` 假定 Gemini 3.6 一定没有推理。正式实验必须固定同一个 reasoning 设置，并记录实际返回的 reasoning token。需要做消融时可显式传入 `--reasoning-effort low`、`medium` 或 `high`，但不同设置产生的文本应当视为不同生成条件。

脚本只向服务商发送 `approved_prompt`，不会发送配对的人类原文。根据 AIHubMix 公布的[数据政策](https://docs.aihubmix.com/en/api/Data-Pravicy)，上游模型提供商的留存规则仍然适用，而且合规事件可能触发额外内容留存，因此不要提交隐私材料、未公开原稿或无权交给第三方处理的文本。接口参数见其[快速入门](https://docs.aihubmix.com/en/quick-start)。

## 9. 进入正式实验前的停止条件

先用审核通过的 20—30 个 Prompt 做极小规模生成和盲审。只有同时满足以下条件才扩到 200—300 篇：

- 人类与机器文本在长度、体裁、主题上的重叠可接受；
- 盲审员能形成基本一致的总体质量排序；
- 高质量机器文本与低质量人类文本确实存在足够重叠，能组成反捷径对照；
- 初步模型在控制质量后仍显示可重复的来源偏差，或者能形成有价值的“无偏差”边界结论；
- 数据授权与发布方案明确。

若这些条件不成立，应调整文本池、Prompt 或标注协议，而不是直接扩大到 2,000 篇。
