# 文献综述（2026-08-14，草稿）

> 围绕论文主线"像人分与质量分正交 + 评估者像人"组织。已精读三篇核心文献（桌面 PDF），其余为领域常识性引用，待补全条目/页码。

---

## 1. 机器生成文本检测（MGT Detection）

**动机**：LLM 普及使机器文本泛滥，检测用于信息完整性、学术诚信、内容安全（Uchendu et al., 2023; Crothers et al., 2023）。

**基准化**：SemEval-2024 Task 8（Wang et al., 2024）定义了多语言、多模型、多子任务的检测基准：
- **Subtask A**：人类 vs 机器二分类（单语/多语言两轨）
- **Subtask B**：精确生成器归属（GPT/Cohere/BLOOMz 等）
- **Subtask C**：人→机过渡点检测（change point）
- 结果：所有子任务最好系统都用 LLM。

**方法与技术路线**：早期为手工特征（n-gram、burstiness、perplexity）与浅层分类器；现代以 Transformer 分类器为主。判别器核心困难是**域迁移与长度偏置**——这正是我们做"长度匹配判别器 + MacBERT 升级"的原因（本工作 §3.2）。

**人类检测能力**（对"像人"意义重大）：
- 早期研究（GPT-3.5 时代）：人类检测接近随机（Guo et al., 2023; Dugan et al., 2023; Chein et al., 2024）——曾被解读为"LLM 已通过图灵测试"
- **Wang et al. (2025)**（HumanEval-MGT）：16 数据集 × 9 语言 × 9 领域 × 19 标注者，专家平均检测准确率 **87.6%**，显著反驳"随机水平"结论；最大人机差距在 **concreteness、文化细节、多样性**；在 prompt 中显式说明这些差异可弥合 >50% 的案例
- 启示：人类检测能力比想象强，但依赖"具体性/文化/多样性"这类语义线索——评估器若只靠表面统计特征，很难复现人类判断（呼应我们 S_attr 的可解释属性设计）。

## 2. 人类与机器对 AI 文本的感知与偏好

**检测 ≠ 偏好**：Wang et al. (2025) 的另一个核心发现——**人类并不总是偏好人类写的文本，尤其当无法识别来源时**。这直接支持"像人 ≠ 质量/偏好"两个正交维度的假设。

**归因偏差（attribution bias）**：Haverals & Martin (2025) 用 Queneau《风格练习》做受控实验：
- 人类：对标注"AI 生成"的内容有 **+13.7pp** 的贬低偏差（Cohen's h=0.28）
- AI 评估器：**+34.3pp**（h=0.70），是人类的 **2.5 倍强**
- 跨 14×14 生成器矩阵验证：AI 系统性贬低被标为"AI 创作"的内容，**与具体哪个 AI 生成无关**
- **最极端发现**：来源标签会让评估者**反转评判标准**——同一特征因被归因作者不同而得到相反评价
- 与我们工作的关系：我们的**标签偏差实验**（§4.7，LLM judge 对来源标签相对稳健，ΔH+|ΔG|≈0.17）与 Haverals & Martin 的强偏差结论方向一致但幅度更小——差异可能来自任务（质量分 vs 审美风格）、文本域（网文 vs 文学）与标签措辞。这是论文里值得讨论的对比点。

**美学判断的文学理论背景**：读者反应理论、接受美学认为文本通过"期待视野"被解读（Iser; Jauss）；罗兰·巴特"作者之死"主张作者无关，但认知实验证明读者在理解中会构建作者意图推断并受其约束——为归因偏差提供理论解释。

## 3. 生成文本的质量评估与 Reward 建模

**人工评估体系**：为规避归因偏差，采用**盲评**（不显示来源）+ **锚定校验**（极好/极差参照校准尺度）。这正是我们评分平台的设计（§4.1/§4.7）。

**LLM-as-judge**：用 LLM 当质量裁判可以规模化（Zheng et al., 2023），但已知偏差：位置效应、冗长偏好（verbosity）、自模型偏好。我们的实测发现一个此前少见的偏差形态：**DeepSeek judge 把机器叙事质量分打得比人类切块更高（3.91 vs 2.82）**——一个"机器偏好机器"的样本，也是"质量轴独立于像人轴"的证据。

**Reward model / RLHF**：偏好对齐依赖从人类反馈学到的 reward。本工作的评估器定位为叙事生成（Track A）的 reward 信号；我们的**贡献不是又一个 reward 模型，而是实证其"像人"维度与"质量"维度正交**，避免"像人"reward 悄悄奖励"像人但平庸"的文本。

## 4. "像人"评估：图灵测试视角

**面向生成器的图灵测试**：经典图灵测试及现代变体（GPT-4 诗歌能否骗过人类读者并被偏爱，Hitsuwari et al., 2023）。测的是"文本像不像人写的"。

**空白：评估者的图灵测试**：极少有工作测试"**评估器/裁判的判断**像不像人类评委"。我们把图灵测试从生成器侧转到评估器侧：
- 判别式 ABX：裁判看"文本 + 两个判断（一真人一评估器）"，能否分辨哪个是真人给的
- 一致性基准：评估器-人类一致性 vs 人类-人类一致性
- 关键设计约束：①必须收集人类"像人感知"判断（不是质量分）②必须在寄存器内部测（避免 H/G 内容线索混淆）——详见 `docs/evaluator_turing_test_protocol.md`

## 5. 我们工作的定位（一句话）

既有的检测、质量评估、图灵测试分别优化"准/好/像"，我们证明 **"像"和"好"正交**，并设计了验证"评估器判断像人"的协议——填补三者的交叉空白。

---

## 参考文献

（全部条目见 `docs/references.bib`，BibTeX；标注 [PDF]=从 PDF 参考文献表核实，[known]=标准引用待投稿前复核）

1. **[PDF]** Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Osama Mohammed Afzal, Tarek Mahmoud, Giovanni Puccetti, Thomas Arnold, Chenxi Whitehouse, Alham Fikri Aji, Nizar Habash, Iryna Gurevych, and Preslav Nakov. 2024. SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection. *SemEval-2024*, pp. 2059–2078.
2. **[PDF]** Yuxia Wang, Rui Xing, Jonibek Mansurov, et al. 2025. Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI. *arXiv:2502.11614*.
3. **[PDF]** Wouter Haverals and Meredith Martin. 2025. Everyone Prefers Human Writers, Including AI. *arXiv:2510.08831*.
4. **[PDF]** Biyang Guo, Xin Zhang, Ziyuan Wang, et al. 2023. How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection. *arXiv:2301.07597*.
5. **[PDF]** Liam Dugan, Daphne Ippolito, Arun Kirubarajan, Sherry Shi, and Chris Callison-Burch. 2023. Real or Fake Text? Investigating Human Ability to Detect Boundaries Between Human-Written and Machine-Generated Text. *AAAI '23*, pp. 12763–12771.
6. **[PDF]** Jason M. Chein, Steven A. Martinez, and Alexander R. Barone. 2024. Human intelligence can safeguard against artificial intelligence: Individual differences in the discernment of human from AI texts. *Scientific Reports* 14(1):25989.
7. **[PDF]** Jimpei Hitsuwari, Yoshiyuki Ueda, Woojin Yun, and Michio Nomura. 2023. Does human-AI collaboration lead to more creative art? Aesthetic evaluation of human-made and AI-generated haiku poetry. *Computers in Human Behavior* 139:107502.
8. **[PDF]** Evan Crothers, Nathalie Japkowicz, and Herna L. Viktor. 2023. Machine-generated text: A comprehensive survey of threat models and detection methods. *IEEE Access*.
9. **[known]** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 36*.

## 待办
- [x] 补齐所有引用条目（2026-08-14，BibTeX 见 `docs/references.bib`；标注了 [PDF]/[known] 核实状态）
- [ ] 增加 RLHF/LLM-as-judge 的核心文献精读（Zheng et al. 2023 等）
- [ ] 对照 Haverals & Martin 与我们的标签偏差实验，写一节专门讨论
- [ ] 把综述中与论文 §2 重叠的部分去重，避免重复
- [ ] 投稿前复核 [known] 条目的页码/卷期
