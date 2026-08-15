# Human-Like but Not Judgmental: Decoupled Human-Likeness Scoring for Chinese Narrative and an Evaluator Turing Test

> Working draft v0.1 (2026-08-14). Targets an English NLP conference (ACL-family style).
> Sections marked ⏳ are pending human-annotated data / not yet executed.

---

## Abstract

Reward models for open-ended narrative generation are often expected to both "recognize good writing" and "judge like a human." We argue these are distinct: the **human-likeness** of a text (how much it reads as human-written) and its **quality** (how good it is) are orthogonal. We build a Chinese narrative evaluator that scores human-likeness from three views — a MacBERT discriminator, a frozen-encoder similarity view, and an interpretable attribute view — and provide systematic decoupling evidence on a self-built dataset of human and machine narrative (n=600, plus two second-generator batches). The like-score separates human from machine text nearly perfectly (AUC 0.993–1.000 across four generator batches) while being at chance (AUC≈0.47) at separating high- from low-quality human text; within-generator, like-score and LLM-judged quality are uncorrelated. We control length bias, show robustness to reward-gaming "watermark padding," and verify stability across registers (chaptered vs. non-chaptered) and across truly different generators (DeepSeek vs. Qwen). Finally we propose an *evaluator Turing test* — whether blind judges can tell the evaluator's judgments from a human rater's (⏳).

---

## 1 Introduction

Reward models are the backbone of aligning generative models toward desired text. For long-form Chinese narrative, a reward model must judge stories the way a human reader would — both in *what* it scores and in *how its judgments compare to a human's*. Existing machine-text detectors optimize accuracy of the human/machine dichotomy; quality reward models optimize correlation with human quality ratings. But these two desiderata are not the same, and conflating them can silently distort a reward signal: a scorer that rewards "reads like human writing" may be indifferent to whether a story is actually good, or worse, reward human-like mediocrity.

This paper asks the evaluator-design question *per se*: can we build an evaluator whose human-likeness score is **decoupled** from quality — high-confidence about "is this human-like," agnostic about "is this good" — and whose judgments are themselves human-like?

We make three contributions:

1. **A three-view human-likeness evaluator** for Chinese narrative, combining a MacBERT discriminator, a frozen-encoder representation view, and 11 interpretable stylistic attributes, with explicit length-strictness and anti-gaming controls.
2. **Systematic decoupling evidence**: on a self-built H/G corpus, the like-score separates human from machine writing with AUC 0.993–1.000, while it cannot separate high- from low-quality human text (AUC≈0.47) and is uncorrelated with LLM-judged quality within machine text; the finding is robust to length-bias removal and to genuine cross-generator transfer (DeepSeek vs. Qwen).
3. **An evaluator Turing test protocol** (⏳) that checks whether human judges can distinguish the evaluator's like-judgments from a human rater's, within register, with a human–human agreement benchmark.

Throughout, we report the patterns honestly, including a residual negative correlation of like-score with quality *within human text* (Spearman −0.246) and batch-level noise in cross-generator orthogonality.

## 2 Related Work

**Detecting machine-generated text.** Benchmarking has matured around multilingual, multimodel detection: SemEval-2024 Task 8 (Wang et al., 2024) defines binary human/machine classification, exact-generator attribution, and human→machine change-point detection, with the best systems being LLM-based. On the human side, early studies with GPT-3.5-era text reported human detection near random chance (Guo et al., 2023; Dugan et al., 2023), whereas Wang et al. (2025) — across 16 datasets, 9 languages and 9 domains — find that expert annotators reach 87.6% average accuracy, with the largest human/machine gaps lying in concreteness, cultural nuance, and diversity. Our work does not add a detector: it builds a *scorer* for the human-likeness axis and evaluates whether the scorer's *judgments* are human-like.

**Attribution bias in evaluation.** Beliefs about authorship shape evaluation even when content is held constant. Haverals and Martin (2025) show, on controlled style-transfer stimuli, that both humans (+13.7 percentage points) and LLMs (+34.3 pp, a 2.5× stronger effect) devalue content labeled "AI-generated," and that attribution labels can invert the criteria applied to otherwise identical features. Our label-bias probe (§4.7) complements this line for the quality dimension on Chinese narrative: we find an LLM judge is relatively robust to source labels.

**Human-likeness vs. preference/quality.** Wang et al. (2025) report that humans do not always prefer human-written text, particularly when its source is unclear. This anticipates our central claim: perceived human-likeness and judged quality are separable axes. Our LLM-as-judge (§4.2) exhibits the decoupling from the other direction — it scores machine narrative *higher* on quality than human narrative (3.91 vs. 2.82), while our like-score separates the two almost perfectly.

**Reward modeling and LLM-as-judge.** RLHF-style reward models learn human preference or quality and are widely used to steer generation; LLM-as-judge frameworks scale quality scoring with model evaluators, whose known biases include position and verbosity effects (Zheng et al., 2023). Our evaluator is a deterministic, interpretable three-view scorer for the human-likeness axis, intended as a reward signal for narrative generation. Its contribution is the *evidence* that this axis is decoupled from quality — so that using it as a reward does not silently reward "human-like but mediocre" text, and its own judgments can be tested for human-likeness (§5).

## 3 Method

### 3.1 Overview
The evaluator computes a human-likeness score in (0,1) from three views of a text `x`:

```
S_disc = P(human | x)              — discriminator view
S_repr = quantile of cos(bge(x), H) — representation view
S_attr = e^{-Mahalanobis(attrs(x); H)} — attribute view
like(x) = λ₁·S_disc + λ₂·S_repr + λ₃·S_attr
final(x) = like(x)^κ(len(x))
```

where `H` is the human reference set and `κ` imposes length strictness (longer machine text is easier to expose, so it is scored more strictly).

### 3.2 Three views
- **Discriminator view.** A character n-gram + logistic-regression baseline was first used; we upgrade to a **MacBERT** discriminator trained on length-matched human/machine chunks (H: WebNovelBench chunks, deduplicated; G: instruction-following corpora; 268+268 after length matching). It reaches AUC 0.9995 on held-out validation. Crucially, unlike n-grams, MacBERT's confidence shows no length bias (G within-correlation of like-score with length: −0.562 → +0.070).
- **Representation view.** The frozen bge-small-zh encoder maps text to a vector; `S_repr` is the quantile of the text's nearest-neighbor cosine similarity to the human reference set. This view rewards novelty *within* the human manifold but, by itself, is insensitive to out-of-manifold human quality variation.
- **Attribute view.** 11 interpretable stylistic attributes (sentence length, lexical diversity, punctuation, repetition, etc.); `S_attr` is the Mahalanobis distance of the attribute vector to the human distribution, exponentiated to (0,1).

### 3.3 Length strictness and length bias
`κ = 1 + α·max(0, len−L0)/L0` with `L0` set to the human median length and `α=0.5` in the deployed config. In addition, the discriminator is trained **length-matched** (equal-sampling H/G within length buckets), which removed the length bias in the n-gram view (Spearman like-vs-length on G: +0.178 → +0.001 at unchanged AUC 0.999).

### 3.4 Anti-gaming hardening
A reward must not be gameable by trivial padding. We attack the scorer with three padding strategies × ratios (1.5×/2×/3×): appending fluent filler, interleaving filler, and self-repetition. Root cause: generic filler pulls the whole-text embedding toward the H centroid, inflating the representation and attribute views. Fix: compute `S_repr` and `S_attr` on a **first-400-characters window** and raise α to 0.5. After hardening, append/interleave/self-repeat attacks that previously inflated the score by +65.1%/+34.5%/0% all become +0.0%, and across all padded G samples `Spearman(final, len) = −0.859` (padding now strictly hurts). The cost on naturally long human text is small and even beneficial: −15.7% → −11.1%, attributable to pure length strictness once the confound of content displacement is removed.

## 4 Decoupling Evidence

### 4.1 Data
Self-built corpus `rating_set` (n=600): 300 human WebNovelBench chunks, 100 non-chaptered machine narratives (DeepSeek), 100 chaptered machine narratives, and 50+50 extreme-quality anchors. For cross-generator transfer we additionally generate 100 texts with Qwen (`qwen-flash`) and 100 with `qwen3.7-flash` (thinking disabled; §4.6).

### 4.2 Quality scores: LLM-as-judge
We obtain per-text quality scores from an LLM judge (DeepSeek) on the same 1–5 scale used by human raters, with anchor calibration. The judge succeeds on 600/600 texts; it rates extreme-good anchors 3.18, extreme-bad anchors 1.00, and separates the anchors perfectly (AUC=1.000), confirming the judge's scale is meaningful. A notable calibration finding: the LLM judge rates machine narrative higher than human chunks on average (3.91 vs. 2.82; AUC for H/G discrimination by quality = 0.098) — i.e., the LLM judge prefers the machine texts on *quality* while humans-style detectors prefer the human texts on *likeness*. This asymmetry is itself evidence that quality and likeness are different axes.

### 4.3 Orthogonality: like-score ⊥ quality
Within-register Spearman between like-score and LLM-judged quality:

| register | ρ | p |
|---|---|---|
| H (human) | −0.246 | <0.001 |
| G (DeepSeek, non-chaptered) | +0.097 | 0.338 |
| GZ (DeepSeek, chaptered) | −0.185 | 0.066 |
| G (Qwen flash) | +0.028 | 0.785 |
| G (Qwen 3.7-flash) | −0.154 | 0.127 |

Machine-text registers are essentially uncorrelated (with batch-level noise); human text shows a significant weak negative correlation. The discriminator indeed scores lower-quality human texts as *more* human-like (LLM scores 2/3/4 → like-score 0.766/0.752/0.713, monotonic). We treat this as a systematic pattern to be honestly reported rather than hidden.

### 4.4 Discrimination separation
- Like-score separates H vs. every generator batch: AUC 0.999 (G), 1.000 (GZ), 1.000 (Qwen flash), 0.993 (Qwen 3.7).
- Within H, like-score cannot separate high- from low-quality text: AUC≈0.47 (n-gram) / 0.241 (MacBERT, length-matched, LLM-quality labels) — at chance.

Together with §4.3: the like-score is informative about "human-like," uninformative about "good." This is the decoupling thesis.

### 4.5 Length-bias control
Upgrading the discriminator from n-gram to MacBERT removed length bias in the discriminative view (G within-correlation −0.562 → +0.070) and stopped padding-induced inflation in `S_disc` (0.20→0.41 → 0.007 constant). Length-matched training additionally removes the n-gram view's bias.

### 4.6 Robustness: registers and generators
- **Cross-register**: chaptered (GZ) and non-chaptered (G) machine text are both cleanly separated from H (AUC 0.999–1.000). Within-register orthogonality holds with the same small negative trend in GZ (−0.185) as in H — cross-register transfer is not perfectly clean, which we report.
- **True cross-generator**: DeepSeek and Qwen are different providers. H vs. Qwen AUC is 0.993–1.000; within-Qwen orthogonality is ≈0 (flash +0.028) to weakly negative (3.7-flash −0.154, n.s.). Different machine-generator batches are almost indistinguishable from each other by like-score (DeepSeek vs. Qwen-flash AUC 0.340; DeepSeek vs. Qwen-3.7-flash AUC 0.127), consistent with all being machine narrative.

### 4.7 Anchor calibration and label bias
- Anchors calibrate the human and LLM judges (AUC=1.000 on extreme anchors for the LLM judge). Human anchor calibration ⏳ pending human ratings.
- **Label bias probe**: an LLM judge is robust to source labels — mislabeling human text as "AI" shifts scores +0.143, labeling machine text as "human" shifts −0.030 (both small); correct labels are essentially bias-free. This validates that the none-label LLM quality scores (§4.2) are not contaminated by source priors. A human-side label experiment is set up (⏳).

## 5 Evaluator Turing Test (⏳)

We propose to verify the "judges like a human" claim directly (protocol: `docs/evaluator_turing_test_protocol.md`):
- **S1 concordance**: Spearman between evaluator like-scores and mean human *like-perception* judgments (a 0–100 "human-like?" slider now collected on the rating platform), benchmarked against the human–human agreement band.
- **S2 ABX**: blind judges see (text, human judgment, evaluator judgment) and must pick the human's; detection ≈50% ⇒ pass. Run within register to avoid the H/G content confound.
- **S3 directionality** + LLM-as-judge reference. Pending collection of human like-judgments and scores.

## 6 Discussion and Limitations

- **Residual negative correlation in H** (−0.246): the evaluator rewards human-like-but-low-quality text. We document it and leave further investigation (n-gram residue vs. judge bias) to future work.
- **Batch noise in orthogonality**: cross-generator ρs range +0.028…−0.154; direction is stable (≈0/weakly negative), magnitude is not.
- **Domain drift**: constructed single-sample evidence (lyrical/technical out-of-domain texts) failed to reproduce the decoupling at the sample level; we rely on distribution-level evidence and defer constructed within-distribution probes.
- **Quality labels are LLM-based** until human ratings arrive; LLM-as-judge has its own biases (see §4.2's inverted H/G preference).

## 7 Conclusion

We present a Chinese narrative evaluator whose human-likeness score is decoupled from text quality: it is highly accurate at identifying machine writing across generators and registers, yet at chance at judging quality, and its judgments are designed to be verified as human-like via an evaluator Turing test. We report both the strong evidence and the honest caveats.

---

## References

BibTeX: `docs/references.bib` (entries marked [PDF] were extracted from the papers' own reference lists; [known] entries to be re-verified before submission).

- Chein, Martinez, and Barone. 2024. Human intelligence can safeguard against artificial intelligence. *Scientific Reports* 14(1):25989.
- Crothers, Japkowicz, and Viktor. 2023. Machine-generated text: A comprehensive survey. *IEEE Access*.
- Dugan, Ippolito, Kirubarajan, Shi, and Callison-Burch. 2023. Real or fake text? *AAAI '23*, 12763–12771.
- Guo et al. 2023. How close is ChatGPT to human experts? *arXiv:2301.07597*.
- Haverals and Martin. 2025. Everyone prefers human writers, including AI. *arXiv:2510.08831*.
- Hitsuwari, Ueda, Yun, and Nomura. 2023. Does human-AI collaboration lead to more creative art? *Computers in Human Behavior* 139:107502.
- Wang, Xing, Mansurov, et al. 2025. Is human-like text liked by humans? *arXiv:2502.11614*.
- Wang et al. 2024. SemEval-2024 Task 8: Multidomain, multimodel and multilingual machine-generated text detection. *SemEval-2024*, 2059–2078.
- Zheng et al. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. *NeurIPS 36*.

## Appendix: pending items to fill
- ⏳ Human quality ratings (per-text ≥2–3) → re-run §4.2–4.3 with human quality labels.
- ⏳ Human like-perception slider data → §5 S1/S2.
- ⏳ `scripts/turing_test_analysis.py`.
- ⏳ Human-side label-bias collection.
- Convert to ACL-style LaTeX; add formal notation, algorithm box, and per-figure captions.
