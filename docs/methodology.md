# Methodology and deviations

This document lists what this repository does differently from the Representation Engineering paper (Zou et al.,
[arXiv:2310.01405](https://arxiv.org/abs/2310.01405)), why, and how its numbers are produced. It is a reference for
readers of the paper, not a tutorial: the [README](../README.md) covers installation and the experiment ids,
and each config is in `experiments/<id>_*.py`. It reports no results. Module paths are relative to `src/tomography/`.

Contents:

1. [Scope and models](#scope-and-models)
2. [Paper mapping](#paper-mapping)
3. [Pipeline choices and deviations](#pipeline-choices-and-deviations)
4. [Statistics](#statistics)
5. [Known versus unknown questions (P9)](#known-versus-unknown-questions-p9)
6. [Run settings of the GPU runs](#run-settings-of-the-gpu-runs)
7. [Data sources](#data-sources)

## Scope and models

| Family             | Keys                                           | Format                           | Default dtype                |
| ------------------ | ---------------------------------------------- | -------------------------------- | ---------------------------- |
| GPT-2              | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` | plain tags                       | float32 (`gpt2-xl`: float16) |
| Qwen3 base         | `qwen3-0.6b-base`, `qwen3-1.7b-base`           | plain tags                       | bfloat16                     |
| Qwen3 post-trained | `qwen3-0.6b`, `qwen3-1.7b`                     | chat template, thinking disabled | bfloat16                     |

The registry is `MODELS` in `model.py`; every entry pins a full commit SHA.

- **Why a second family.** The paper steers LLaMA-2-Chat. Its 7B and 13B checkpoints do not fit an 8 GB GPU in
  bfloat16, so the second family is Qwen3 at 1.7B and 0.6B, each as a base and a post-trained checkpoint. The
  post-trained Qwen3 checkpoints are small distilled models with a thinking mode; they are used with thinking
  disabled (`enable_thinking=False`). A contrast between a base model and its post-trained counterpart is therefore
  a contrast with this specific recipe and not with instruction tuning in general.
- **Why GPT-2.** GPT-2 is a base model with no instruction tuning. It is the cheap development subject and the
  no-instruction-tuning control: it shows what the recipe does when the model was never trained to follow a
  persona prompt.
- **Architecture interface.** `ARCHS` in `model.py` declares, per `config.model_type`, the decoder blocks, the final
  norm, the embeddings, the LoRA target modules and the BOS policy. Hooks, activation collection, controllers and
  LoRRA use only this interface. Any other architecture raises `UnsupportedArchitectureError`.
- **Post-trained format.** `ChatFormat.from_tokenizer` renders the tokenizer's chat template with the generation
  prompt and tracks character offsets, so the response span is exact.
- **Not replicated.** The paper's LLaMA-2 results are not reproduced; no experiment here claims numerical agreement
  with them.

## Paper mapping

The paper section, appendix or figure each experiment follows, or "extension" when the experiment has no paper
counterpart. The ids are the arguments of `tomography run`; each config and its docstring are in
`experiments/<id>_*.py`.

| Id             | What it does                                                           | Paper counterpart                                                                    |
| -------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `r1`           | Sentiment on SST-2 phrases                                             | Extension (a concept sanity check that anchors the steering experiments)             |
| `r2`           | Utility (pleasantness) on ETHICS pairs, PCA explained variance         | Sec. 5.1, App. D.1.8, Fig. 11                                                        |
| `r3`           | Morality on ETHICS commonsense scenarios                               | Sec. 5.2, App. D.1.9                                                                 |
| `r4`           | Six emotions, one-vs-rest, t-SNE                                       | Sec. 6.1, App. D.1.10, Fig. 18                                                       |
| `r5`           | Honesty function with persona prompts                                  | Sec. 4.3, App. D.1.2, Figs. 8 and 9                                                  |
| `r6`           | TruthfulQA MC1: standard, heuristic, text-only and LAT                 | Sec. 4.1, Table 1, Table 8                                                           |
| `r7`           | Truthfulness on true and false facts, plus out-of-distribution sets    | Sec. 4.1 and 4.2; the out-of-distribution truth sets are an extension                |
| `m1`           | Six direction finders: correlation, manipulation, termination          | Sec. 5.1.1, Fig. 12                                                                  |
| `c1`           | Sentiment steering: sweep, operators, side effects                     | Extension (uses the Sec. 3.2.1 operators and adds a prompting baseline arm)          |
| `c2`           | Emotion steering, steered emotion against measured emotion             | Sec. 6.1.2                                                                           |
| `c4`           | Honesty control on TruthfulQA MC1                                      | Sec. 4.3.3, Table 2; adds a prompting baseline arm                                   |
| `c5`           | Qualitative steering galleries                                         | Figs. 10 and 17                                                                      |
| `lr1`          | One LoRRA adapter (loss curve and held-out loss)                       | Algorithm 1; `c4` scores the adapter through `lorra_run`                             |
| `p1`           | Template ablations                                                     | Sec. 5.1.1, Fig. 13 (ETHICS utility with and without the LAT template); the other variants are an extension |
| `p2`           | Token position and a misleading sentence in the prompt                 | App. B.9, Table 14 and Fig. 27 (LAT at token indices -1, -6, -7 and -8, original and biased prompts), Table 13 (zero-shot heuristic). The code reads `last` (the paper's -1) and `concept` (the paper's concept-token indices, -8 to -6 for "truthfulness"); `mean` goes beyond the paper |
| `p3`           | Stimulus-set size                                                      | Extension                                                                            |
| `p4`           | Pairing and preprocessing                                              | Extension (the options include the paper's `unit` and the reference code's `center`) |
| `p5`           | Persona wording and instruction confounds for honesty                  | Extension                                                                            |
| `p6`           | Scale across the four GPT-2 sizes                                      | Extension                                                                            |
| `p7`           | Whether a reader that reads well also steers well                      | Extension                                                                            |
| `p8`           | Base against post-trained Qwen3                                        | App. B.8, Fig. 26 (base against chat LLaMA-2); the Qwen3 base and post-trained pairs are an extension |
| `p9a` to `p9d` | Known against unknown questions: labels, reading, reliability, steering | Extension                                                                           |
| `a1`           | Direction geometry across layers, seeds and concepts                   | Extension                                                                            |
| `a2`           | Logit lens of the reading directions                                   | Extension                                                                            |
| `a5`           | Reading directions against Qwen-Scope SAE decoders                     | Extension                                                                            |

The PCA explained variance is part of `r2` and the t-SNE maps are part of `r4`; they are not separate experiments.

## Pipeline choices and deviations

Each row states what this repository does, how that differs from the paper, and why. Where the reference
implementation (`andyzoujm/representation-engineering`, pinned at commit `5455d8a`) differs from the paper
text, the default follows the reference code and the other variant is selectable.

### Model and prompt

| Choice                              | Deviation from the paper                                                                                                                                                                                              | Reason                                                                                                                                                                                  | Where                                                    |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Base models without a chat template | The paper prompts chat models. GPT-2 and the Qwen3 base models get plain tags (`USER: ...`, `ASSISTANT: ...`, the paper's placeholders).                                                                                | They have no template. The format is itself varied in `p1`, `p5` and `p8`.                                                                                                              | `templates.py`                                           |
| Trailing space of `ASSISTANT: `     | None. The paper's LLaMA prompts end in the same lone-space token, so it is kept.                                                                                                                                      | `p1` includes a variant with the space stripped.                                                                                                                                        | `templates.py`, `experiments/variants.py`                |
| BOS handling                        | The paper does not say. GPT-2 and the Qwen3 base models prepend the document separator. The post-trained Qwen3 models prepend nothing, because the chat template carries its own special tokens.                      | Without a prepended token the first content token becomes the attention sink and its residual norm dwarfs the rest. A probe on the Qwen3 base models showed a prepended separator takes over that role. | `model.py` (`default_bos`, `LoadedModel.encode`), `tests/test_models_slow.py` |
| Position 0 in token scans           | Not discussed. Position 0 is the BOS, or the sink token when there is no BOS.                                                                                                                                         | The token-scan plots can blank the first column (`mask_first`). The honesty scans of `r5` cover the response span only, so they never contain position 0 and do not use the mask.      | `viz/tokens.py`                                          |
| Context guard                       | None.                                                                                                                                                                                                                 | `LoadedModel.encode` raises `ValueError` when a text exceeds `max_position_embeddings`; an over-long scenario had crashed a GPU run. `r3` reads short scenarios only by default (`short_only`). | `model.py`, `experiments/r3_morality.py`                 |
| Padding                             | None.                                                                                                                                                                                                                 | Right padding with gathered indices for reading; left padding with explicit `position_ids` for batched generation, because the GPT-2 forward does not derive positions from left padding. | `activations.py`, `control/generate.py`                  |

### Activations and directions

| Choice                                      | Deviation from the paper                                                                                                                                                      | Reason                                                                                                                                                                                | Where                                    |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Raw final-layer residual                    | Layer `L` is the output of the last block before the final norm. The Hugging Face `hidden_states[L]` is taken after the final norm (`ln_f` for GPT-2, `norm` for Qwen3).        | One definition of "residual stream at layer l" for every layer, so reads and edits act on the same quantity. Layer 0 is the embedding output.                                          | `hooks.py`, `activations.py`             |
| Single-pass response positions              | The paper's Eq. 2 truncates the response token by token and runs the model on each prefix. The `response` position reads every token of the marked span from one forward pass. | Under causal attention the activation at a token does not depend on later tokens, so the two agree up to numerics, at a fraction of the cost.                                          | `activations.py`                         |
| Pooled positions                            | The paper reads single token positions. The pooled positions reduce a span to one row: `mean` averages the response tokens, `max` takes their feature-wise maximum, and `concept_mean` averages the tokens of the concept span, so it equals `concept` for a one-token concept. | Recent probing work reports that probes pooled over tokens (mean, max or attention) often read better than the last token, so readers can be compared across these readouts. The reduction runs on the model's device, so a pooled row costs the memory of one token. | `activations.py`                         |
| `logistic` as the default finder            | The paper reads with PCA on pair differences (`pca_diff`). `r1` to `r7`, `p1`, `p2`, `p3` and `p5` default to `logistic`; `p4` keeps `pca_diff` as its reference cell.         | On GPT-2 the paper's `pca_diff` stays near chance on unmatched SST-2 pairs and weak on matched ETHICS pairs. `pca_diff` remains selectable (`--set method=pca_diff`) and is the paper-faithful series. | `experiments/sweep.py` (`ReadingConfig`) |
| `pca_diff` preprocessing                    | The paper says the differences are normalized (`unit`). The reference code subtracts the mean (`center`) and does not normalize.                                              | `center` is the default, as in the reference code; `unit` and `none` are options and `p4` compares them.                                                                              | `reading/directions.py`                  |
| Deterministic PCA                           | The reference code uses scikit-learn's randomized solver without a seed.                                                                                                      | A full SVD is used, so a direction is a function of its inputs.                                                                                                                       | `reading/directions.py`                  |
| Held-out splits                             | The reference code evaluates honesty and emotions on strings that also occur in training.                                                                                      | Splits are assigned by hashing the source statement or scenario id before any truncation or pairing, so no source appears in two splits.                                              | `datasets/records.py` (`assign_splits`)  |
| Finders and stimuli the reference omits     | The reference ships only the PCA, cluster-mean and random finders, and no utility or morality code.                                                                           | The logistic and K-means finders follow the paper text, and `pca_raw` is an added variant. Utility and morality stimuli are built from the paper's templates and ETHICS. The prompt difference is a control contrast of `c4` (`control/contrast.py`), not a reading finder. | `reading/directions.py`, `datasets/ethics.py` |
| Whitened mass-mean finder                   | Marks and Tegmark (2024) invert the pooled within-class covariance with a pseudo-inverse at an absolute tolerance of 1e-3. `whitened_mean_diff` shrinks that covariance toward a scaled identity with the Ledoit-Wolf weight instead and solves `S^-1 (m1 - m0)`, the LDA reader's direction. | With GPT-2 and Qwen3 widths the covariance is singular whenever the training rows are fewer than the width, and an absolute tolerance tuned to another model's activation scale has no meaning here; shrinkage keeps every direction and needs no tuned constant. | `reading/directions.py`                  |
| CCS                                         | Burns et al. (2023) centre each contrast set on the mean of the set being evaluated and pick the sign with the test labels (the larger of the accuracy and one minus it); their Appendix G.1 also divides each set "by the average norm of {phi(x_i+)} times sqrt(d)", a phrase that leaves the target norm ambiguous and that they call "less essential than mean normalization", and their code does not scale by default. `fit_ccs` centres each set on its training mean and reuses it for every later pair, divides both by one shared scalar so the centred training rows have an average norm of sqrt(d) (about unit scale per feature), keeps the restart of lowest loss out of 10 (1000 full-batch AdamW epochs, learning rate 0.01 as in the paper, 0.001 in both reference repositories) and orients the probe on the training labels with the rule of `RepReader.fit` applied to the set-centred rows, as Marks and Tegmark (2024) do. | Training means score a single pair without its test set. Features at about unit scale keep the fixed learning rate and the unit-norm initial weights on the same footing at every width, where rows of norm 1 would shrink each feature by sqrt(d); on synthetic pairs this lowered the final loss with the same accuracy. A shared scalar keeps one linear probe for both sets, and the test labels never choose the sign. The probe's bias and its two set means do not fit a `RepReader`, so `CCSProbe` keeps them for its own pair prediction and exports the direction as a reader centred on the mean of the two set means; that reader ignores the per-set centring. | `reading/ccs.py`                         |
| Readers beyond one direction per layer      | The paper reads with one direction per layer. Three readers behind a common interface (`Reader`) go beyond it: LDA, the mass-mean direction under a within-class covariance shrunk toward a scaled identity (Ledoit-Wolf weight by default); MLPs on standardized activations, one per layer or one on concatenated layers, fitted with L-BFGS; and an attention-pooled linear reader on the token rows of the `response` position: a softmax over the tokens of a stimulus, or over the (token, layer) cells of several layers, weights a linear value head; its query starts at zero, which is mean pooling, and L-BFGS minimizes the L2 logistic loss of the logistic finder in float64. | Readers are compared on the same features. LDA stays linear, so it is a `RepReader` that can steer, oriented by the rule of `RepReader.fit`; the attention reader, as in EleutherAI's attention probes, tests whether choosing tokens beats a fixed position or the mean, and exports its value heads as such a reader; the MLP has no direction and tests whether a nonlinear reader adds anything. | `reading/probes.py`                      |

### Stimuli and data

| Choice                      | Deviation from the paper                                                                                                                                                                                                                                                       | Reason                                                                                               | Where                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| TruthfulQA stimulus sources | The paper's TruthfulQA readers include a source of stimuli generated by LLaMA-2-Chat-13B. That source is not reproduced. `r6` trains readers on three other sources: TruthfulQA train questions, ARC-Challenge train items and the true statements as persona pairs.        | GPT-2 cannot generate such stimuli reliably, and LLaMA-2-Chat is not in the model set.               | `experiments/r6_truthfulqa.py`           |
| TruthfulQA snapshot         | The Hub snapshot of January 2024 at a pinned revision, not the January 2025 revision of the GitHub repository.                                                                                                                                                                  | The snapshot is pinned and loads as a plain parquet file. Its validation rows are split by hash into train, validation and test questions. | `datasets/truthfulqa.py`                 |
| MC1 choice order            | In the source, the correct answer of every row is choice 0.                                                                                                                                                                                                                     | Choices are shuffled with a seed and ties are broken at random, so a constant scorer scores at chance. | `datasets/truthfulqa.py`                 |
| Emotion scenarios           | Reused from the reference repository, not regenerated.                                                                                                                                                                                                                          | The scenario sets are the reference code's data; the splits are our own (see held-out splits).        | `datasets/emotions.py`                   |
| Honesty persona words       | The paper's appendix says a "dishonest" persona. The reference code, and the default here, pairs "an honest" with "an untruthful".                                                                                                                                              | The default follows the reference code; `p5` varies the wording.                                     | `templates.py`                           |
| Coefficient tuning          | The paper fixed steering coefficients on a few self-generated examples.                                                                                                                                                                                                         | `c4` chooses each vector arm's coefficient on TruthfulQA train and validation questions and reports on held-out test questions. | `experiments/c4_honesty_control.py`      |

### Steering and LoRRA

| Choice                | Deviation from the paper                                                                                                                                               | Reason                                                                                                                                                | Where                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Relative coefficients | The reference code uses absolute coefficients in raw residual units.                                                                                                   | The `relative` mode scales the coefficient by the mean residual norm at each layer, so one value is comparable across layers and sizes. Absolute mode is available. | `control/controllers.py`       |
| LoRRA target modules  | The reference applies LoRA to `q_proj` and `v_proj`, and Qwen3 does the same. GPT-2 has one fused `c_attn` projection, so its adapter covers query, key and value together. | GPT-2 has no separate projections to target.                                                                                                          | `model.py`, `lorra.py`         |
| LoRRA loss            | None by default: the loss follows the reference code (mean over tokens of the unsquared per-token L2 norm). The paper's summed squared form is the option `sum_squared`. | Follows the reference code.                                                                                                                           | `lorra.py`                     |
| LoRRA model           | The paper adapts 7B and 13B models. `lr1` defaults to the post-trained `qwen3-1.7b`.                                                                                   | The model must fit an 8 GB GPU.                                                                                                                       | `experiments/lr1_lorra.py`     |

## Statistics

### Seeds and run identity

- `derive_seed(base, *keys)` turns a base seed and identifying keys into a 32-bit seed from a SHA-256, the same in
  every process (`experiments/base.py`). Pairing, `RepReader.fit`, generation and every resampling take explicit
  seeds.
- A run is identified by `config_hash()`, the first 12 hex digits of the SHA-256 of its canonical config JSON.
  Every config field is part of the hash, so a change of `batch_size` is a different run (see
  [Run settings](#run-settings-of-the-gpu-runs)). `meta.json` records the code state, device and library versions.
- The reading experiments train `n_seeds` readers on independent training sets: 10 by default (5 for `p9b` and `p9c`), 15
  for `r6`, and 20 resamples per size for `p3`.

### Series and controls

Every reading sweep (`experiments/sweep.py`) writes three series per layer:

- `reader`: the fitted direction.
- `random`: a random unit direction with the same orientation rule. This is the random-direction control.
- `shuffled`: the reader scored against evaluation labels shuffled within each group. This is the permutation null.

Steering experiments compare against random directions of the same norm where a claim needs it (`m1`, `p9d`).

### Intervals and what is resampled

Intervals are 95 percent intervals. `bootstrap_ci` in `reading/metrics.py` is a percentile bootstrap of the mean
that resamples whole runs along axis 0, and `wilson_ci` gives intervals for rates.

| Experiments                                                    | Interval                                                       | Resampled unit                                                            |
| -------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `r1` to `r7`, `p1`, `p2`, `p4`, `p5` curves                    | percentile bootstrap                                           | the seeds (independent training sets)                                     |
| `p3`                                                           | percentile bootstrap                                           | the resamples of each training size                                       |
| `c1`, `c2` dose-response curves                                | percentile bootstrap                                           | the units of each series (seeds or readers)                               |
| `m1`                                                           | percentile bootstrap; Wilson interval for win rates            | the seeds                                                                 |
| `c4`                                                           | paired percentile bootstrap; Wilson interval for win rates     | the held-out test questions                                               |
| `p6`, `p8`                                                     | percentile bootstrap                                           | the seeds; `p8` pairs a base and a post-trained model by seed             |
| `p7` inversion rate                                            | Wilson interval                                                | the units (template, layer, steering reader) that are readable            |
| `a1` agreement band                                            | t-interval on jackknife pseudo-values, clipped to [-1, 1]      | the seeds (at least 3)                                                    |
| `p9b` reader AUROC (`lo`, `hi`, `lo_within`, `hi_within`)      | percentile bootstrap                                           | the `n_seeds` pair draws of the reader, not the test items                |
| `p9c` AUROC (`lo`, `hi`)                                       | percentile bootstrap                                           | the test items                                                            |
| `p9c` within-relation AUROC                                    | percentile bootstrap                                           | the items, resampled inside each relation at its size                     |
| `p9d` slopes                                                   | percentile bootstrap                                           | the items, paired across arms                                             |

`a1` uses a t-interval because a percentile bootstrap of so few runs covers less than its level. The `a2` and `a5`
intervals and nulls are described in the docstrings of `experiments/a2_logit_lens.py` and
`experiments/a5_sae_features.py`. A `p9b` interval describes how much the fitted reader
varies with its training draw; it does not cover the noise of the test items.

## Known versus unknown questions (P9)

P9 asks whether a direction separates questions a model answers from questions it cannot answer. The stages are
`p9a` (labels), `p9b` (reading), `p9c` (reliability) and `p9d` (steering). The code is in `p9/`, `knowledge.py` and
`experiments/p9*_*.py`.

### Labels

`p9a` labels each closed-book question from the model's behavior, using a few-shot prompt (`n_shots` 15), one greedy
answer and `n_samples` (5) samples at temperature 1.0 (`LabelRecipe` in `knowledge.py`):

- `known`: the greedy answer is correct and at least 3 of the 5 samples are correct.
- `unknown`: the greedy answer is wrong and none of the samples is correct.
- `uncertain`: everything else. These items stay out of the contrasts.

Questions without a gold answer (fictional subjects) are unknown by construction.

### Guess control

Some relations have answers that a constant guess often hits. `p9/guess.py` computes, per model and per stratum,
the accuracy of the modal greedy answer. A stratum is `popqa:<relation>` or `award:<relation>`, or the whole dataset
for `triviaqa` and `nq_open`. A stratum whose constant-guess accuracy is above 0.10
(`max_guess_accuracy`) supplies no known items, and `popqa:sport` is excluded by default. An item whose greedy
answer is the modal answer of its stratum is never known. The summary records the demotions (`guessable_stratum`, `modal_answer`, `excluded_stratum`). A model
enters the known versus unknown contrasts only with at least `min_known` (100) guess-adjusted known items. Otherwise
`p9b` reads only the unknown-type contrasts and `p9d` does not run.

### The relation pool

The PopQA pool first held the four relations shared with SyntHal (father, mother, occupation, sport). The guess
control removed `sport` and, on the Qwen3 models, `occupation`, which left too few known items for the
PopQA-matched contrasts to be read at all. The pool now has 12 relations (`POPQA_RELATIONS` in
`datasets/popqa.py`): those four plus director, screenwriter, author, producer, composer, capital, place of birth
and capital of. The guess control is unchanged and one template per relation is used. The SyntHal-based contrasts
stay on the four SyntHal relations.

### Same-dataset control and surface baselines

Contrasts built from different datasets differ in question format, so a reader can separate them by wording alone.
`triviaqa_known` sets TriviaQA questions labeled known against TriviaQA questions labeled unknown, so one question
format stands on both sides. Every `p9b` reader AUROC is reported next to three surface baselines
(`SURFACE_BASELINES` in `p9/reading.py`):

- `text_word` and `text_char`: classifiers on the word and the character n-grams of the question, fitted on the
  train items of the contrast and scored on the exact test items of the reader.
- `popularity`: the AUROC of the subject's popularity on those items. It is null unless every test item on both
  sides has an integer popularity.

### Within-relation headline

Relations differ in how many of their questions are known, so the wording of a question alone can rank the two
sides through the relation. A pooled AUROC cannot tell that apart from reading the question. For the contrasts that
have PopQA questions on a side, the headline is therefore the within-relation AUROC (`grouped_auroc` in
`p9/risk.py`): the AUROC inside each relation, averaged with weights `n_pos * n_neg` over the relations that hold
both classes. It is reported next to the within-relation popularity and text baselines (`WITHIN_BASELINES`). The
pooled AUROC, and the layer choice made on the pooled validation AUROC, are unchanged and stay as a secondary
column.

A contrast is relational when every validation and test item on both sides carries a relation. `triviaqa_known`,
`unknowable_detail`, `false_premise` and `kuq_<category>` are not relational, and their within-relation columns are
null. In `p9c` the same within-relation AUROC is the PopQA headline, and the AUROC of a single relation is reported
only with at least 5 items of each outcome (`min_relation_class`).

### Intervals

- `p9b`: a bootstrap over the `n_seeds` pair draws of the reader. It measures training-draw variability and not the
  sampling of test items, which dominates when few pairs share a relation. Read it next to `n_pairs_within`.
- `p9c` within-relation: `grouped_auroc_ci` resamples items with replacement inside each relation at that
  relation's size. Single-class relations are dropped first, and a draw in which no relation has both classes is
  skipped. With one relation it equals the plain item bootstrap. This scheme was chosen over resampling items
  globally, which is conservative, after a coverage simulation at a nominal 95 percent.

### Steering headline (`p9d`)

The outcome is the abstention indicator (post-trained models) or the entropy of the next-token distribution (base
models, shift mode). For every item it is regressed on the whole coefficient grid. The slope of an arm is the mean
of the item slopes, with a percentile bootstrap over items. The known versus unknown direction (`direction_type`,
default `obscure_real`) is compared with `n_random` (3) random directions of the same norm.

Random directions can move the outcome as far as the steering direction, and in the opposite direction, so the
comparison is of magnitudes. For each random arm separately, the statistic is the absolute mean steering slope minus
the absolute mean slope of that arm, with a paired item bootstrap (`abs_gap`, `abs_lo`, `abs_hi`). Steering is
called distinguishable from a random edit of the same size only if the lower bound of that gap is above zero for
every random arm. A random arm without finite slopes makes the claim false. The signed difference of the slopes is
also reported.

Selectivity is a separate claim. For the steering arm, the statistic is the absolute mean slope over the unknown items
minus the absolute mean slope over the known items (`gap` in `table_selectivity`). The two sides hold different items,
so each bootstrap resample draws the items of each side independently. The unknown side is oriented by the sign of its
mean slope, so a resample gives the signed unknown mean minus the absolute known mean; this avoids the upward bias of
the absolute value of a mean near zero, and the reported `gap` is still the difference of absolute means. The effect is
called selective only if the lower bound of the percentile interval is above zero. Distinguishable and selective are
reported as separate flags, with no correction for multiple comparisons, as for the other P9d intervals. Types that
share a known pool (`obscure_real` with `fictional_generated`, `unknowable_detail` with `false_premise`) share
known-side items, so their gaps are correlated.

The headline cell is the unknown side of the `direction_type` contrast; every other cell is descriptive, with no
correction for multiple comparisons. The abstention patterns are phrase matches, so a title that contains a pattern
word still counts as an abstention; the `hand_check.csv` sample measures this.

## Run settings of the GPU runs

- **Batch size is part of the run.** Sampling in `p9a` is seeded per chunk of prompts, so `batch_size` changes every
  sampled completion and the config hash. The label runs used a batch size of 8 for every model except `gpt2-large`
  (3) and `gpt2-xl` (4).
- **CUDA memory cap.** The runs on the 8 GB laptop GPU called `torch.cuda.set_per_process_memory_fraction(0.8)`
  before the run, so the caching allocator does not spill into shared system memory. This is memory safety only and
  does not change the outputs. The caller of the run sets it; it is not in the repository code.
- **Offline reruns.** Dataset loaders take `offline=True` and read only the cache; for model weights set
  `HF_HUB_OFFLINE=1`. See [Reproduce at scale](../README.md#reproduce-at-scale).
- **Precision.** float32 for GPT-2 small, medium and large, float16 for `gpt2-xl`, bfloat16 for Qwen3. Activations
  are stored as float32 arrays.

## Data sources

Every dataset, its pinned revision or digest, its license and the experiments that use it are recorded in the
`PinnedSource` rows under `datasets/`, and the README describes pinning and offline mode in
[Reproduce at scale](../README.md#reproduce-at-scale). No third-party data is redistributed.
