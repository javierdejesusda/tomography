# Related work

This document places the experiments of this repository next to the published work they replicate, contrast with or
control for. It is a reference for readers of the paper, not a literature review: each entry says what the cited work
does and how this repository relates to it. It makes no claim about priority and reports no results of this
repository. Experiment ids (`r1`, `p6`, `p9a` and so on) are those of [methodology.md](methodology.md), which also
lists the deviations from the Representation Engineering paper.

Citations use the keys of [references.bib](references.bib). Every entry in that file was checked against a primary
source (arXiv listing, ACL Anthology, PMLR, NeurIPS or OpenReview record, or the publisher) on 2026-10-16. Where a
paper has a venue version, the venue version is cited and its arXiv identifier is given as `eprint`. Three kinds of
source carry a caveat, stated in the text:

- Blog posts (LessWrong) are not peer reviewed.
- Preprints are marked "preprint" and have not been peer reviewed.
- Workshop papers are marked as such.

Contents:

1. [Representation Engineering and its critiques](#representation-engineering-and-its-critiques)
2. [Steering methods](#steering-methods)
3. [Reliability and prompt sensitivity](#reliability-and-prompt-sensitivity)
4. [Linear representations and truth probing](#linear-representations-and-truth-probing)
5. [GPT-2 specific work](#gpt-2-specific-work)
6. [Concept erasure](#concept-erasure)
7. [Lenses](#lenses)
8. [Known versus unknown questions](#known-versus-unknown-questions)
9. [Datasets and evaluators](#datasets-and-evaluators)

## Representation Engineering and its critiques

- **Representation Engineering** [zou2025representationengineeringtopdownapproach]. The paper this repository
  reimplements. It reads concepts with Linear Artificial Tomography (a stimulus template, last-token activations and
  PCA on pair differences) and controls models with reading vectors, contrast vectors and LoRRA. Its experiments use
  chat-tuned LLaMA-2 and Vicuna models (and DeBERTa in side tests). Every `r`, `m`, `c`, `lr` and `p1` to `p4`
  experiment here follows one of its sections or extends it, as the paper mapping in the methodology lists.
- **Taxonomy of Representation Engineering** [wehner2025taxonomy]. A survey of the RepE pipeline, its variants and
  its open challenges, including reliability and the preservation of capabilities. It frames the questions that the
  stimulus-design experiments (`p1` to `p5`) and the reading-versus-steering experiment (`p7`) put to the method.
- **Open Challenges in Representation Engineering** [wehner2025open]. A blog post (not peer reviewed) by two of the
  survey authors. It raises the concern that a concept dataset may not isolate the concept, for example that
  honesty instructions may elicit a representation of following the instruction. `p5` varies the persona wording
  and the stimuli of the honesty reader for this reason.
- **Concept guidance** [vonrutte2024language]. Extends reading and steering beyond truthfulness to many concepts and
  reports that the most accurate probes are not always the best guides. This is the question `p7` asks of the
  reading directions of this repository.
- **RepE for reasoning** [hojer2025improving]. Applies a RepE-style reading-then-control recipe to reasoning on
  Pythia and Mistral models. It is an example of the recipe on smaller open models, which is also the setting of the
  GPT-2 and Qwen3 experiments here.

## Steering methods

- **Activation addition** [turner2023steering]. Adds a vector computed from a pair of prompts to the residual
  stream. The early versions demonstrate it on GPT-2 models, and the authors state that a coefficient, a layer and
  a prompt pair have to be chosen. The sweeps of `c1` and `c2` over coefficient and layer, and the prompt-pair
  variants of `p1`, address the same choices.
- **Steering GPT-2-XL** [turner2023gpt2xl]. A blog post (not peer reviewed) with many GPT-2-XL examples. It reports
  layer dependence and sensitivity to token spacing in the prompt pair. `c5` produces the qualitative galleries
  that correspond to such demonstrations, and `p1` includes a variant with the trailing space stripped.
- **Contrastive activation addition** [rimsky2024steering]. Averages the activation differences between positive and
  negative behavior examples and adds the mean to a chat model. Its averaging over many pairs is the setting of the
  stimulus-set size experiment `p3` and the pairing experiment `p4`. The ACL Anthology names the first author Nina
  Rimsky; the arXiv listing names the same author Nina Panickssery.
- **Inference-time intervention** [li2023inference]. Trains probes on attention heads and shifts the activations of
  the selected heads along the probe directions to raise TruthfulQA scores. It is the probe-then-shift counterpart
  of the reading-then-steering design of `c4`, which scores honesty steering on TruthfulQA MC1.
- **Mean-centring** [jorgensen2023improving]. Preprint. Subtracts the mean activation of a training set from the
  steering vector and studies GPT-2 models, including the effect of the choice of dataset on the vector. The
  `center` preprocessing of `p4` (the default of the reference code of the paper) also subtracts a mean, over the
  pair differences.
- **ReFT** [wu2024reft]. Learns interventions on hidden representations. Like LoRRA, which `lr1` trains and `c4`
  scores, it learns the intervention instead of reading it off activation differences.
- **Persona vectors** [chen2025persona]. Preprint. Extracts trait directions to monitor and steer character traits,
  including shifts caused by finetuning. It is a trait-level counterpart of the persona-based honesty function of `r5`.
- **Refusal direction** [arditi2024refusal]. Finds one direction that reads and controls refusal in chat models, a
  case where the reading direction and the control direction coincide. `p7` tests that coincidence for the concepts
  of this repository, and `p9d` steers abstention.
- **AxBench** [wu2025axbench]. Compares steering methods, including prompting and sparse autoencoder features, and
  reports that simple baselines compete with the other methods. The prompting baseline arms of `c1` and `c4` follow
  this practice, and `a5` compares reading directions with the decoders of public sparse autoencoders.
- **Latent steering vectors** [subramani2022extracting]. Recovers vectors that steer GPT-2 (117M) toward target
  sentences and shows sentiment transfer. It is an early GPT-2 steering study and a precedent for injecting a vector
  in the middle of the stack.
- **Contrastive activation engineering** [hao2025patterns]. Preprint (workshop paper). Studies patterns and
  mechanisms of contrastive activation methods, including how the effect depends on the number of samples. It is
  relevant to `p3`.

## Reliability and prompt sensitivity

- **Reliability of steering vectors** [tan2024analysing]. Finds that steerability varies across inputs and that
  generalization to other prompts can be brittle, and attributes much of the variation to the dataset rather than
  the model. This is the motivation for reporting distributions over stimulus resamples and for the out-of-distribution
  truth sets of `r7`. The NeurIPS title uses the spelling "Analysing"; the arXiv title uses "Analyzing".
- **Unreliability of steering vectors** [braun2025understanding]. Workshop paper. Reports high variance across prompt
  types and ties better steering to directionally consistent activation differences. It motivates the resampling
  statistics of the reading sweeps and the pair-draw intervals described in the methodology.
- **Prompt formatting** [sclar2024quantifying]. Shows that meaning-preserving formatting changes alter the accuracy
  of language models on benchmarks. It is a behavioral result; the template and position experiments `p1` and `p2`
  ask the question about the reading directions.
- **Identifiability** [venkatesh2026on]. Preprint. Argues that steering vectors are not identifiable from their
  behavioral effect. It is a reason to compare steering against random directions of the same norm (`m1`, `p9d`).
- **Inverted detection and control** [torop2026inverted]. Preprint. Reports vectors that detect a concept well but
  steer in the opposite direction. `p7` records how often a direction that reads well has a dose slope of the wrong sign.
- **Measurement confounds across scale** [wu2026when]. Preprint. Reports that an apparent scale trend in steerable
  concept representations came from an uncalibrated pipeline. The scale experiment `p6` compares four model sizes; the
  `relative` coefficient mode scales a steering coefficient by the residual norm of each layer for comparability across
  sizes.
- **Task format versus reasoning mode** [sahoo2026linear]. Workshop paper. Reports that probe accuracy falls when
  task-format confounds are controlled. It is an instance of the stimulus confound that `p5` and the same-dataset
  control of `p9b` target.
- **Limits of truth directions** [poulis2026testing]. Preprint (arXiv listing states BlackboxNLP 2026). Reports that
  truth directions depend on layer, task type, difficulty and prompt instructions. It bears on `r7` and `p2`.
- **Where steering succeeds** [billa2026predicting]. Preprint. Relates a logit-lens profile to the layers at which
  steering works. `p7` also correlates a logit-lens separation with the steering slope.

## Linear representations and truth probing

- **Contrast-consistent search** [burns2023discovering]. Finds a truth direction without labels from the consistency
  of paired statements. The Representation Engineering paper compares its own method with this one on DeBERTa; this
  repository does not run it.
- **Internal state of an LLM** [azaria2023internal]. Trains a classifier on hidden states to separate true from false
  statements. It is the supervised probe baseline for the truth readers of `r7` and for `p9b`.
- **Geometry of truth** [marks2024geometry]. Shows linear structure for true and false statements that grows with
  model scale and checks it with transfer and interventions. `r7` and `p6` ask the same questions on GPT-2 models.
- **Universal truth direction** [burger2024truth]. Finds a low-dimensional subspace for truth and polarity and builds
  a lie detector on it. It relates to the out-of-distribution truth sets of `r7`.
- **Linear representation hypothesis** [park2024linear]. Formalizes linear representations through counterfactual
  pairs and links probing with steering. This is the formal setting of the pair-difference readers.
- **Multi-dimensional features** [engels2025not]. Shows features that occupy more than one dimension, which a single
  PCA direction does not capture. `a1` examines the geometry of the directions across layers, seeds and concepts.
- **Lie detectors** [levinstein2025still]. A paper in Philosophical Studies arguing that probes of the CCS and
  Azaria-Mitchell type fail basic generalization tests. It is cited for the conceptual caution about reading a
  truth direction as honesty. The journal issue is 2025; the paper was published online in 2024.
- **Unsupervised knowledge discovery** [farquhar2023challenges]. Preprint. Shows that unsupervised directions can
  track the most prominent feature of the data instead of knowledge. `p5` and `p9b` control for such features.
- **Deception probes** [goldowskydill2025detecting]. Trains linear probes on honest and deceptive data of the kind
  the honesty function of `r5` uses. It reports that such probes are not yet a robust defense.

## GPT-2 specific work

- **Linear sentiment** [tigges2024language]. Shows that sentiment is linearly represented in models including
  GPT-2 small, with templated prompts, several direction finders that agree closely, and activation addition. This is
  the nearest published setting to `r1`, `c1` and the direction-finder comparison `m1`. The venue title is "Language
  Models Linearly Represent Sentiment"; version 1 on arXiv had a different title.
- **Emotion vectors in small models** [jeong2026extracting]. Preprint. Compares extraction and steering of emotion
  vectors in small language models, including GPT-2 124M, using mean subtraction. It relates to `r4` and `c2`.
- **Sentiment circuits in GPT-2** [hatua2025mechanistic]. Preprint. Patches GPT-2 and separates lexical from
  contextual sentiment layers. It is a reference for where sentiment is computed, next to the layer sweeps of `r1`.
- **Activation addition and latent vectors on GPT-2** [turner2023steering; turner2023gpt2xl; subramani2022extracting;
  jorgensen2023improving]. These four works are listed in the steering section. They are the GPT-2 steering studies
  most relevant to the `c` experiments.

## Concept erasure

- **Iterative nullspace projection** [ravfogel2020null]. Removes a linearly encoded attribute by projecting out
  successive probe directions. The projection operator of the paper (Sec. 3.2.1) removes a single direction.
- **LEACE** [belrose2023leace]. Closed-form linear erasure that prevents any linear classifier from recovering a
  concept. It is the principled counterpart of the single-direction projection above. This repository does not
  implement it.

## Lenses

- **Logit lens** [nostalgebraist2020interpreting]. A blog post (not peer reviewed) that decodes intermediate
  residual states of GPT-2 through the unembedding. `a2` applies the same decoding to the reading directions.
- **Tuned lens** [belrose2023eliciting]. Preprint. Replaces the plain decoding with a learned affine translator per
  layer and evaluates it on several model families including GPT-2. This repository uses the plain logit lens only.

## Known versus unknown questions

The `p9a` to `p9d` experiments ask whether a direction separates questions that a model answers from questions it
cannot answer, how reliable that direction is, and whether adding it changes abstention. The work below is grouped by
what it studies. Papers on other model families are listed for context; the comparison is a matter of setting, not
a claim about results.

### Self-knowledge, answerability and abstention

- **Models know what they know** [kadavath2022language]. Preprint. Evaluates a model's probability that its own
  answer is true and a trained probability that it knows the answer. These are the P(True) and P(IK) baselines in
  the literature on self-assessment.
- **Hidden-state answerability** [slobodkin2023curious]. Finds that the hidden states of instruction-tuned models
  encode whether a question is answerable, even when the output hallucinates.
- **Known unknowns** [yin2023large]. Introduces the SelfAware benchmark of answerable and unanswerable questions
  and measures how far models are from humans at recognizing the second kind.
- **KUQ** [amayuelas2024knowledge]. Introduces a benchmark of known and unknown questions with several sources of
  uncertainty, and studies fine-tuning for the distinction. `p9a` draws unknown-type questions from KUQ categories.
- **Surveys** [li2025knowledge; wen2025know]. A survey of knowledge boundaries and a survey of abstention. They give
  the vocabulary (query, model and human-value perspectives on abstention) used when describing the `p9d` outcome.

### Probing the state before generation

- **Entity recognition** [ferrando2025do]. Finds sparse-autoencoder latents that separate known from unknown
  entities in 2B to 9B models and steer refusal against hallucination. It is the closest design to `p9d` among the
  instruction-tuned models; this repository uses reading directions on GPT-2 and Qwen3 instead of sparse latents.
- **Knowledge from the subject state** [gottesman2024estimating]. Trains a probe on the subject representation to
  predict question accuracy, with GPT2-XL among the models, and compares it with a popularity predictor. The
  popularity baseline of `p9b` follows this comparison.
- **Hallucination risk of a query** [ji2024llm]. Shows that internal states before generation indicate whether a
  query was seen in training and the risk of hallucination.
- **Truthfulness in answer tokens** [orgad2025llms]. Locates truthfulness information in the answer tokens and
  reports that probes do not generalize across datasets. Cross-dataset confounds are the reason `p9b` reports a
  same-dataset contrast and surface baselines.
- **Probing for hallucination** [chwang2024do]. Trains probes on states of grounded generation tasks and finds that
  they work in domain and generalize poorly across tasks.
- **Knowledge checking with RepE-style reading** [zeng2025towards]. Applies PCA of paired differences with a
  classifier to internal knowledge checking in a retrieval setting, on one 7B instruction-tuned model.
- **Linear directions for answerability** [lavi2026detecting]. Selects a direction by its effect on abstention and
  transfers it to other benchmarks. Adding or ablating the direction controls abstention.
- **Post-training and knowledge** [du2025how]. Preprint (arXiv listing states COLM 2025). Compares base and
  post-trained models and reports that the truthfulness direction is similar while the refusal direction differs. It
  bears on the base against post-trained contrast of `p8`.
- **Entity familiarity** [brzezinka2026does; brzezinka2026graded]. Two preprints on entity familiarity readouts in
  instruction-tuned models; the second covers several model families, including Qwen3, and a refusal-steering
  direction. Qwen3 is also the second family of this repository (`p9b` to `p9d`).
- **Detection without control** [galeone2026perfect]. Preprint. Reports that a direction that detects fabricated
  entities is nearly orthogonal to the refusal direction. It is the kind of dissociation that `p9d` tests with random
  directions of the same norm.
- **Unanswerable prompts** [du2026recognition]. Preprint (arXiv listing states EMNLP 2026). Reports a direction for
  structurally unanswerable prompts that is separate from the safety refusal direction and steerable.
- **Confidence and behavior** [kumaran2026causal]. Preprint. Steers a confidence direction and measures the trade
  between abstention and coverage. It motivates reporting risk-coverage outputs next to the steering slope.
- **Correctness probing on GPT-2** [roy2026detection; cho2026confidence]. Two preprints that probe correctness
  signals in models that include GPT-2 sizes; the first also tests whether moving along the probe direction
  corrects errors. They study GPT-2 sizes, as `p9b` and `p9d` do.
- **Neurons and confidence** [stolfo2024confidence]. Identifies entropy and token-frequency neurons in language
  models including GPT-2 small. It is about the regulation of confidence, not about known and unknown questions.
  The base-model outcome of `p9d` is the entropy of the next-token distribution.

### Uncertainty estimation baselines

- **Semantic entropy** [farquhar2024detecting; kossen2024semantic]. Estimates uncertainty from the meaning of
  sampled answers, and trains probes that approximate it from hidden states. They are the standard comparison for
  a hidden-state signal of correctness, next to token probability and entropy.
- **EigenScore** [chen2024inside]. Detects hallucination from the covariance of sampled-response embeddings.

## Datasets and evaluators

Dataset revisions, licenses and the experiments that use them are recorded in the `PinnedSource` rows under
`datasets/`; see the data sources section of the [methodology](methodology.md#data-sources). The entries below
cite the dataset papers only.

- **TruthfulQA** [lin2022truthfulqa]. A benchmark of questions on which models imitate common falsehoods. `r6`
  and `c4` use its MC1 task.
- **ETHICS** [hendrycks2021aligning]. Introduces the ETHICS dataset. `r2`, `r3` and the template ablation of `p1`
  use its utility and commonsense morality parts, following the paper.
- **SST-2** [socher2013recursive]. The Stanford Sentiment Treebank, from which the SST-2 phrases of `r1` come.
- **PopQA** [mallen2023when]. Questions about entities with a popularity measure, introduced to study when
  retrieval helps. `p9a` to `p9c` use PopQA questions in the pool and its popularity as a baseline.
- **TriviaQA** [joshi2017triviaqa]. A reading comprehension dataset of trivia questions. `p9a` uses its closed-book
  questions and the `triviaqa_known` contrast, and the lie gallery of `c5` draws questions from it.
- **SelfAware** [yin2023large]. Listed above with the benchmarks of answerability; it is not in the `p9a` pool.
