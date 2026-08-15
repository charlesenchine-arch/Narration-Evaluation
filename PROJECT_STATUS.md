# 项目状态：类人叙事评估器

> 最后更新：2026-08-14
> 阅读目的：下次打开对话时，先读本文件即可知道"做了什么、没做什么、接下来做什么"。

---

## 1. 项目目标（用户原话要点）

构建一个**模仿人类审美的中文叙事文本评估器**。核心不是"机器文本检测最准"，而是**评估器对文本的判断逻辑和输出像人类评委**。最终作为叙事时间 Transformer（Track A）生成文本的 reward 信号：`score_text(生成文本) → 分数`。

**用户反复强调的定位**：
- 目标是"像人"，不是"最准"。"像人分"与"质量分"是**两个正交维度**（探针实验证明）。
- 综合评分是必要的（人类数据也用得分量化），但长度不是"高分信号"，而是**严苛度参数**（长文机器更易露馅，评估器对长文更严苛）。
- 论文主线：**解耦性实证发现**（像人 vs 质量是两个独立维度），用评估者图灵测试验证。

---

## 2. 做了什么（已完成）

### 2.1 评估器核心（src/narrative_evaluator/）
- **三视图像人分**：`score_text = λ₁S_disc + λ₂S_repr + λ₃S_attr`，衡量"文本有多像人类写作"
  - `S_disc`：判别器 P(human)（字符 n-gram + Logistic Regression）
  - `S_repr`：冻结编码器（bge-small-zh-v1.5）文本向量到 H 参照集合的最近余弦相似度分位
  - `S_attr`：11 个可解释属性（句长/词汇丰富度/标点/重复率等）相对 H 分布的马氏距离
- **长度严苛度**：`final = 像人分^κ(len)`，`κ = 1 + α·max(0,len-L0)/L0`，基准 L0=H 中位长度（自适应），α=0.2 温和。长文机器更易露馅故更严苛。
- **长度匹配判别器**：`NgramDiscriminator.fit(length_match=True)` 按长度分桶等量采样 H/G，消除长度偏置（G 上像人分 vs 长度 Spearman 从 +0.178 → +0.001，AUC 保持 0.999）。
- **CLI**：`python -m narrative_evaluator.cli` 流式加载数据 → fit → 报告。
- 22 个单元测试全过（tests/）。

### 2.2 论文核心证据（scripts/）
- **解耦性实验**（decoupling_probe.py）：像人分 vs 质量分 Spearman **-0.063**（正交）；像人分区分人机 **AUC 0.999**，区分 H 高低质 **AUC 0.47**（≈随机）——**像人分能判断"像不像人"，对"好不好"一无所知**。在长度匹配（无偏置）下依然成立。
- **构造证据**（constructed_evidence.py）：像人低质/不像人高质文本解耦验证。

### 2.3 自建数据集（data/eval_dataset/）
| 来源 | 数量 | 说明 |
|---|---|---|
| H 网络小说切块 | 300 | WebNovelBench 章节切成 300-800 字块 |
| G 非章回 DeepSeek | 100 | 叙事 prompt 生成 |
| G 章回体 DeepSeek | 100 | **新增**，96% 含"第X章"，覆盖章回特征防泄漏 |
| 极好锚定 | 50 | WebNovelBench 高分章节切块 |
| 极差锚定 | 50 | 构造的碎片化废话文本 |
| **评分集合计** | **600** | `rating_set.jsonl`，锚定占 ~17% |

### 2.4 人工评分系统（scripts/rating_server.py + rating_frontend.html）
- **盲评**：前端不显示标签/来源，后端 `/api/text` 只返回 `{id, text}`（从源头防泄漏）
- **自动推送**：登录后直接进入第一篇评分，评分后自动下一篇，进度条显示
- **趣味评分**：拉完了(1)/NPC(2)/人上人(3)/顶级(4)/夯(5)
- **评论区**：每篇可写评语，随评分保存
- **动态分配**：任意名字登录自动分配 80 条（优先未分配，减少重叠）
- **纠偏系统**（`/api/analysis`）：
  - 停留时间 < 2000ms 的评分删除（防恶意/敷衍）
  - 方差贝叶斯收缩：`s²_adj = (n·s² + k·s²₀)/(n+k)`
  - 权重映射 [0.8, 1.2]（区分度好→1.2，趋同→0.8）
  - 打分趋同（std<0.3）评分者权重压到最低
  - **锚定校验**：极好/极差锚定的评分者一致性，检测尺度偏移
- **口令保护**：`--password narrate2026`，局域网分享
- **当前运行**：`http://172.16.14.82:8770`（口令 narrate2026）

### 2.5 记忆（C:\Users\LiuSh\.claude\projects\C--Users-LiuSh-Desktop-Development-Time-for-narration-narrative-evaluator\memory\）
- evaluator_data_sources.md / evaluator_alignment_probe.md / evaluator_paper_direction.md / evaluator_dataset_building.md（MEMORY.md 索引）

### 2.6 B1 严苛度防刷分验证（scripts/strictness_antigame.py，2026-08-13）
在评分集（H300+G200 短文）拟合部署配置评估器，对 G 施三种注水策略 × 比例（1.5x/2x/3x）：
- **高比例注水（≥2x）与自复读 → 最终分下降 -24%~-74%**，全样本 Spearman(final,len)=**-0.528**（防刷分成立）
- **轻度注水（1.5x）追加流畅废话 → 最终分上涨 +33~45%**（残留攻击面）。来源：S_disc 长度/内容偏置 0.20→0.41（length_match 只能轻微缓解）+ S_repr 中心化 0.08→0.29；κ=1.12 太温和
- **H 侧**：自然 2x 长文 -15.7%（κ 理论仅 -4.3%，余为拼接内容位移）；注水 H -70%（内容检测生效）
- 结论：对重度注水+自复读有效；残留"轻度假话注水"漏洞，论文需如实报告或加固

### 2.8 LLM 质量分 + 自建集解耦验证（2026-08-13，P0-B）
- **LLM 盲评**：`scripts/llm_quality_rating.py` 用 DeepSeek 按趣味量表(1-5)+评语盲评 600 条，**600/600 成功，0 失败**。结果 `data/eval_dataset/raw/llm_quality_ratings.jsonl`（断点续传）。
- **锚定校验**：极好锚定(AH) LLM 均分 3.18、极差锚定(AL) 均分 1.00、对锚定区分 AUC=1.000 → LLM judge 尺度有效。
- **自建集解耦**（`scripts/decoupling_probe_llm.py`，length_match=True）：
  - 区分度分离：像人分区分 H/G **AUC=1.000**；区分 H 内高低质 **AUC=0.241**≈随机 → 解耦主线成立
  - 正交性：H 内 Spearman **-0.315(p<0.001)**（稳定，非 length_match 也 -0.295）；G 内 -0.106(ns)
  - **H 内显著负相关**：判别器对人类文本内部质量更低的文本打更高像人分
    （LLM 分=2/3/4 → 像人分 0.766/0.752/0.713 单调下降）。与 A3 空洞废话>真实叙事一致 → 系统性模式
  - 对照组异常：LLM 对 H 切块均分 2.82、对 G 机器叙事均分 3.91（AUC 区分 H/G=0.098）
  - 长度偏置：G 内像人分 vs 长度 -0.562
- 解读：解耦主线成立，但"像人分"在人类内部偏向低质文本，论文需如实呈现；金标准仍是真人分。

### 2.9 MacBERT 判别器升级（2026-08-14）
- **实现**：`src/narrative_evaluator/models/macbert_discriminator.py`（MacBertDiscriminator，与 Ngram 同接口，
  手写 PyTorch 循环，fit 幂等加载部署模型不重训）+ 工厂 `build_discriminator`（env NARRATIVE_DISC_TYPE 覆盖）
  + 训练脚本 `scripts/train_macbert_discriminator.py` + `configs/default_macbert.yaml`
  + MixScorer `components_batch`（批量三视图）。
- **GPU 转折**：本机有 RTX 4060 Laptop，初始 torch 是 cpu 版 → 装 `torch 2.13.0+cu126`，
  训练从 CPU 30min+ 降到 **~1min**。device 用 "auto"。
- **训练**：H=WebNovelBench 切块去重 3922 + G=COIG 270 → 长度匹配 268+268，验证集=评分集非锚定 500。
  验证集判别 **AUC=0.9995**（n-gram 1.0000 持平）。
- **三实验对比（n-gram → MacBERT）**：
  - G 内像人分 vs 长度 **-0.562 → +0.070**（长度偏置大幅消除）
  - G 内像人分 vs LLM 质量分 **-0.106 → -0.029**（≈0）
  - H 内像人分 vs LLM 质量分 -0.315 → **-0.246**（缓解）
  - B1 S_disc 注水虚涨 **0.20→0.41（n-gram） vs 0.007 恒定（MacBERT）**——判别器长度偏置彻底消除
  - **A3 构造证据仍失败**（域漂移：构造样本是抒情/诗意/技术文，训练分布是网文，MacBERT 全判低分）
  - **B1 轻注水漏洞根因转移**：S_disc 不再虚涨，但 S_repr 仍 0.08→0.29（表示视图中心化）→ 需 S_repr 惩罚或 α 提高

### 2.10 标签偏差实验（Label Bias Probe，2026-08-14）
用户提出：研究人类和 LLM 的评判是否被来源标签（H/G）影响。三组对照 none/correct/false。
- **LLM 侧**（DeepSeek 600×3 组，`scripts/llm_quality_rating.py --label-mode`）：
  - H 被标成"AI模型"后评分 **+0.143**（反向偏差）；G 被标成"人类"后 **-0.030**（≈无偏）
  - false vs none Wilcoxon p=0.000 显著但效应小（|ΔH|+|ΔG|=0.173）；correct 组几乎无偏（p=0.140）
  - **结论**：LLM 对标签相对稳健；之前 none 组 LLM 质量分未被来源偏见污染
- **人类侧**：评分服务器支持 `--label-mode`（/api/text 附 label_text），前端显示"来源标注"。
  需启动三套服务器实例（不同端口）收集。与 LLM 侧对比是论文好素材。

### 2.11 B1 轻注水漏洞加固（2026-08-14 上午，已完成）
- **根因确认**：1.5x 注水时 S_repr 0.084→0.288、S_attr 0.255→0.340 同时虚涨（通用化填充把
  整篇向量拉向 H 质心）；S_disc 恒定 0.007 已无长度偏置。
- **加固方案**（`configs/default_deploy.yaml`）：**S_repr/S_attr 窗口化**（只在前 400 字符上
  计算，0=全文本向后兼容）+ **α 0.2→0.5**。窗口化后 G 各注水比例分量完全持平
  （S_repr/S_attr 恒定 0.048/0.106），末尾填充不再抬高 like 分。
- **结果**：append_filler 1.5x **+65.1% → +0.0%**；interleave **+34.5% → +0.0%**；
  self_repeat 保持 +0.0%；全 G 注水样本 Spearman(final,len) = **-0.859**。
- **H 侧代价**：自然长文(~2x) **-15.7% → -11.1%**（纯 κ 惩罚，比基线还好——窗口化消除了
  拼接内容位移的混淆）。
- **试过不行的路**：① 纯 α 提高需 0.8 才堵 1.5x，但 H 自然长文 -29% 过度惩罚；② 句级
  自相似度惩罚无法区分轻注水（G 1.5x 自相似 0.608 < H 原文 0.633）。
- 30 个单元测试全过（window_chars 默认 0=全文本，实验脚本不受影响）。

### 2.12 跨寄存器解耦验证（2026-08-14 上午）
- **真正跨生成器（第二生成器）被堵**：ANTHROPIC_API_KEY 无效(401 authentication_error)；
  DeepSeek 的 v4-flash/v4-pro 均返回空内容（仅 deepseek-chat 可用，与现有 G 同款）。
  需用户提供有效 key 才能做真·跨生成器。
- **改为跨风格寄存器**：非章回 G(100) + 章回体 GZ(100)。
- 区分度：像人分区分 H vs G **AUC=0.999**、H vs GZ **AUC=1.000** —— 两个寄存器都清晰可分。
- 正交性：G 内 **+0.097**(p=0.338)；GZ 内 **-0.185**(p=0.066，弱负相关)——章回寄存器里
  "像人分越高质量越低"的弱趋势，与 H 内 -0.246 同向，论文需如实报告（跨寄存器并不完全干净）。

### 2.13 真·跨生成器解耦验证（2026-08-14 下午，scripts/cross_generator_probe.py）
用通义千问做**真·第二生成器**（与 DeepSeek 不同厂商），生成两批 G 并各评 LLM 质量分：
- 批次 1：`qwen-flash` 100 条（`raw/g_qwen.jsonl` + `llm_quality_g_qwen.jsonl`，旧 key）
- 批次 2：`qwen3.7-flash` 100 条（`raw/g_qwen37.jsonl` + `llm_quality_g_qwen37.jsonl`，新 key，**关 thinking**）
  - 实测 **qwen3.x 默认单条烧 ~4600 reasoning token + 50s**；`extra_body={"enable_thinking": False}`（`generate_g_qwen.py --no-think`）→ ~520 token、6.5s，省 ~10 倍配额
- 结果（MacBERT 判别器，length_match）：

| 指标 | G_ds(DeepSeek) | G_qwen(flash) | G_qwen(3.7) |
|---|---|---|---|
| 像人分区分 H vs 该生成器 AUC | 0.999 | **1.000** | **0.993** |
| 该生成器内 像人分 vs 质量分 Spearman | +0.097 (ns) | **+0.028 (ns)** | **-0.154 (ns)** |
| 平均像人分 | 0.020 | 0.008 | 0.098 |

- 生成器间：G_ds vs G_qwen(flash) AUC=0.340、vs G_qwen(3.7) AUC=0.127（两个机器生成器互相几乎不可分——符合预期）
- **结论**：解耦的**区分度侧跨真·不同生成器稳健**（H vs 千问 0.993/1.000）。**正交性侧机器生成器内部整体≈0 但批次间有噪声**：flash 批 +0.028（干净），qwen3.7 批 -0.154（弱负、不显著，方向与 H 内 -0.246、章回体 GZ 内 -0.185 同向）。论文需如实呈现：解耦方向稳定、强度批次间波动。

### 2.7 A3 构造证据短文分布重测（scripts/constructed_evidence.py，2026-08-13）
三次配置均未通过（单样本级），诊断：
- RealDet HWT（问答/论坛）作短文人类参照 → 全组像人分高，判别器分不开（域不匹配）
- 评分集参照 + ~100 字样本 → 全组偏低（长度低于参照地板 345）
- 评分集参照 + 长度匹配样本(≥348) + 三视图 → 仍失败。根因：**S_repr 对新颖文本系统性≈0**（表示视图不奖励 H 集外的新文本，含真实叙事）+ **S_disc 在少量样本上不可靠**（空洞废话/机器技术文得分高于真实叙事）
- 含义：分布级解耦成立（AUC 0.999），但 n-gram 评分器无法在少量构造单样本上复现；需 MacBERT / 增样本 / 重校准 S_repr

---

## 3. 没做什么 / 未解决

### 3.1 评分数据收集（当前最大阻塞）
- **人工评分还没开始**。评分服务器在跑，但需要人类评委实际去评。600 条需要多人评分才能得到质量分。
- 评完后：质量分 = 加权平均（纠偏后），用于解耦实验的"质量分"一侧。

### 3.2 论文剩余工作
- **跨生成器验证**：解耦性需要多数据集/多生成器，当前只有 WebNovelBench 评分。需要更多"同一文本有像人分+质量分"的数据。
- **评估者图灵测试**：尚未设计/实施。需要收集人类评委的评价对（人评 vs 机评），裁判区分来源。训练无监督，但验证需要小规模人类数据（用户已接受）。
- **~~B1 残留攻击面~~（2026-08-14 上午已解决）**：窗口化 S_repr/S_attr + α=0.5，三种注水全堵（见 2.11）。
- **A3 构造证据仍失败**（MacBERT 也分不开）：根因是**域漂移**——构造样本(抒情/诗意/技术文) vs 训练分布(网文)。2026-08-14 评估：域内"极端分组"方案与来源身份混淆、与既有解耦证据冗余，单样本级构造需重新设计（进训练分布或用真人评分数据），暂时搁置。
- **~~跨生成器（第二生成器）验证被堵~~（2026-08-14 下午已解决）**：QWEN_API_KEY 可用，生成 qwen-flash + qwen3.7 两批 G 并完成解耦验证（见 2.13）。
- **H 内负相关 -0.246 仍在**：判别器对人类内部低质文本打更高像人分（MacBERT 缓解但未消除）。可能反映 n-gram 残余或 LLM 分本身偏差，需进一步研究。

### 3.3 工程未解决
- **长文编码慢**：CNNSum 30K 长文 CPU 编码 2.4h+，用户决定**交给 GPU**（本地不跑长文全量）。
- **爬虫**：维基文库公版短篇爬取受阻（PDF 索引），搁置。H 用切分块即可。
- **评分服务器已配登录自启**（2026-08-14）：`scripts/start_rating_autostart.bat`（幂等）+ 启动文件夹 `.vbs`，重启登录后自动拉起服务器+隧道；新 URL 自动写入 `logs/CURRENT_URL.txt`（trycloudflare URL 每次重启会变，需重发新链接）。
- **固定域名：搁置**（2026-08-14 用户决定先不做）：方案=Cloudflare 命名隧道（不上云），待用户买域名+注册 Cloudflare 账号后实施；当前用随机 trycloudflare 链接。

---

## 4. 接下来要做什么（按优先级）

### P0：收集人工评分（推动实验的关键）
1. 邀请评分者访问公网链接 `https://subscriber-queue-ask-fancy.trycloudflare.com`（口令 narrate2026）。每人首批随机抽 15 条（全量 600 池），**评完自动续批**（想继续评的评委可一直评下去，从评分最少的文本补，2026-08-14 已实现并端到端测试通过）。**每篇额外填"像人滑块"（0-100，人类像人感知，图灵测试用，已上线）**，存于 state["likes"]。链接为免费 trycloudflare 隧道，重启/关机后失效，需重新生成（命令见 §5）。【2026-08-14 下午：已有评分者开始评分】
2. 评分时写评语（图灵测试需要人类评价的形态）
3. 收集到足够评分后（每篇 ≥2-3 人），用 `/api/analysis` 取加权平均分作质量分
4. 【已完成，LLM 分版】解耦实验：自建集上像人分 vs 质量分（LLM 分版已跑，见 2.8；真人分到位后重跑）

### P1：解耦性论文主线
- 用评分数据 + WebNovelBench 完成跨数据集/跨生成器解耦验证
- 【已完成】B1 严苛度防刷分验证（高比例/自复读防住）→ 【2026-08-14 加固完成】窗口化 S_repr/S_attr + α=0.5，轻注水全堵（见 2.11）
- 【已完成 2026-08-13，未通过】A3 构造证据短文分布重测（根因= S_repr 新颖文本≈0 + n-gram 不可靠；路径=MacBERT/增样本/重校准后再测）

### P2：评估者图灵测试（验证"像人"）
- 设计图灵测试协议（评价对、裁判、统计检验）
- 收集人类评委数据

### P3：论文写作
- 按"解耦性实证"主线组织论文：正交性 → 区分度分离 → 长度偏置控制 → 锚定校验 → 图灵测试
- 【2026-08-14 已建】英文草稿 v0.1：`docs/paper_draft_en.md`（§1-§4 含全部实测数字 + §2 Related Work 已扩写带引用，§5 图灵测试待数据；目标英文会议/ACL 风格）+ `docs/paper_outline.md`（大纲）。数字已与 PROJECT_STATUS/探针结果核对
- 【2026-08-14 已建】中文文献综述：`docs/literature_review.md`（精读 3 篇 PDF：Haverals&Martin 归因偏差、Wang et al. HumanEval-MGT、SemEval-2024 Task 8）。
- 【2026-08-14 已建】引用库 `docs/references.bib`（BibTeX，11 条；标注 [PDF]=从 PDF 参考文献表核实 / [known]=投稿前复核）。英文草稿 §2 已扩写并带引用 + 文末 References 段；综述附完整参考文献列表

---

## 5. 环境与命令备忘

- Python 3.13，**RTX 4060 Laptop GPU（torch 2.13.0+cu126）**，device 用 "auto" 自动用 GPU
- 有 ANTHROPIC_API_KEY + DEEPSEEK_API_KEY（LLM 生成可用）
- 依赖：scikit-learn, scipy, jieba, sentence-transformers, transformers, datasets, safetensors（已装）
- 评分服务器启动：`python scripts/rating_server.py --data data/eval_dataset/rating_set.jsonl --per-rater 15 --host 0.0.0.0 --password narrate2026`（标签实验加 `--label-mode correct/false --port 8771/8772`）
- 公网隧道（免费随机 URL，重启后需重新生成）：`cloudflared tunnel --url http://localhost:8770`（URL 出现在 logs/cf8770.err.log）
- Qwen 生成（第二生成器）：`python scripts/generate_g_qwen.py --n 100 --out data/eval_dataset/raw/g_qwen37.jsonl --model qwen3.7-flash --no-think --tag G_Q7`（`--no-think` 关 reasoning，省 ~10 倍 token）
- 跨生成器解耦：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src PYTHONUTF8=1 python scripts/cross_generator_probe.py --config configs/default_macbert.yaml --qwen data/eval_dataset/raw/g_qwen37.jsonl --qwen-scores data/eval_dataset/raw/llm_quality_g_qwen37.jsonl`
- 评估器 CLI：`PYTHONPATH=src python -m narrative_evaluator.cli`
- 部署配置（B1 加固版）：`configs/default_deploy.yaml`（window_chars=400 + strict_alpha=0.5）；B1 验证：`PYTHONPATH=src python scripts/strictness_antigame.py --config configs/default_deploy.yaml`
- LLM 质量分：`PYTHONPATH=src python scripts/llm_quality_rating.py --data data/eval_dataset/rating_set.jsonl --out data/eval_dataset/raw/llm_quality_ratings.jsonl`（需要 DEEPSEEK_API_KEY；断点续传；`--label-mode correct/false` 跑标签实验）
- MacBERT 训练：`PYTHONPATH=src python scripts/train_macbert_discriminator.py --config configs/default_macbert.yaml`（GPU ~1min）
- 自建集解耦实验（MacBERT 版）：`PYTHONPATH=src python scripts/decoupling_probe_llm.py --config configs/default_macbert.yaml`
- 标签偏差分析：`PYTHONUTF8=1 python scripts/label_bias_probe.py --none ... --correct ... --false ...`
- 测试：`PYTHONPATH=src python -m pytest tests/`
