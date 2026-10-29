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
5. [Reader comparison (M2)](#reader-comparison-m2)
6. [LAT against few-shot prompting (R8)](#lat-against-few-shot-prompting-r8)
7. [Known versus unknown questions (P9)](#known-versus-unknown-questions-p9)
8. [Run settings of the GPU runs](#run-settings-of-the-gpu-runs)
9. [Data sources](#data-sources)

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
| `r8`           | LAT against few-shot prompting on five QA tasks and TruthfulQA MC1     | Fig. 7 and Table 9 (QA tasks), Table 1 (TruthfulQA MC1); see [R8](#lat-against-few-shot-prompting-r8) |
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
| Whitened mass-mean finder                   | Marks and Tegmark (2024) invert the pooled within-class covariance with a pseudo-inverse at an absolute tolerance of 1e-3. `whitened_mean_diff` shrinks that covariance toward a scaled identity with the Ledoit-Wolf weight instead and solves `S^-1 (m1 - m0)`, the LDA reader's direction. The reference also centres each dataset on its mean over all rows, validation rows included; here the means and the covariance come from the training rows only, which changes the reader's mean, not its direction. | With GPT-2 and Qwen3 widths the covariance is singular whenever the training rows are fewer than the width, and an absolute tolerance does not scale with the activations, whose scale in GPT-2 and Qwen3 need not match that of the reference model; shrinkage keeps every direction and needs no tuned constant. | `reading/directions.py`                  |
| CCS                                         | Burns et al. (2023) centre each contrast set on the mean of the set being evaluated and pick the sign with the test labels (the larger of the accuracy and one minus it); their Appendix G.1 also divides each set "by the average norm of {phi(x_i+)} times sqrt(d)", a phrase that leaves the target norm ambiguous and that they call "less essential than mean normalization", and their code does not scale by default. `fit_ccs` centres each set on its training mean and reuses it for every later pair, divides both by one shared scalar so the centred training rows have an average norm of sqrt(d) (about unit scale per feature), keeps the restart of lowest loss out of 10 (1000 full-batch AdamW epochs, learning rate 0.01 as in the paper, 0.001 in both reference repositories) and orients the probe on the training labels with the rule of `RepReader.fit` applied to the set-centred rows, as Marks and Tegmark (2024) do. | Training means score a single pair without its test set. Features at about unit scale keep the fixed learning rate and the unit-norm initial weights on the same footing at every width, where rows of norm 1 would shrink each feature by sqrt(d); on synthetic pairs this lowered the final loss with the same accuracy. A shared scalar keeps one linear probe for both sets, and the test labels never choose the sign. The probe's bias and its two set means do not fit a `RepReader`, so `CCSProbe` keeps them for its own pair prediction and exports the direction as a reader centred on the mean of the two set means; that reader ignores the per-set centring. | `reading/ccs.py`                         |
| Readers beyond one direction per layer      | The paper reads with one direction per layer. Three readers behind a common interface (`Reader`) go beyond it: LDA, the mass-mean direction under a within-class covariance shrunk toward a scaled identity (Ledoit-Wolf weight by default); MLPs on standardized activations, one per layer or one on concatenated layers, fitted with L-BFGS; and an attention-pooled linear reader on the token rows of the `response` position: a softmax over the tokens of a stimulus, or over the (token, layer) cells of several layers, weights a linear value head; its query starts at zero, which is mean pooling, and L-BFGS minimizes the L2 logistic loss of the logistic finder in float64. | Readers are compared on the same features. LDA stays linear, so it is a `RepReader` that can steer, oriented by the rule of `RepReader.fit`; the attention reader, as in EleutherAI's attention probes, tests whether choosing tokens beats a fixed position or the mean, and exports its value heads as such a reader; the MLP has no direction and tests whether a nonlinear reader adds anything. | `reading/probes.py`                      |
| TTPD                                        | Not in the paper. TTPD (truth and polarity directions) fits a general truth direction by least squares on the centered rows, a polarity direction by an unpenalized logistic regression, and an unpenalized logistic regression on the two projections, as in its reference code. One topic, so the rows are centered on their mean. | It is the published structured linear reader that beats logistic regression out of distribution on lies. Its score is linear, so it is a `RepReader`; without polarity labels it is the least-squares truth direction, so `m2` reads it on its polarity task only. | `reading/probes.py` (`fit_ttpd`)         |

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
| `b1` AUROC and within-relation AUROC                           | percentile bootstrap, with the `p9c` settings                  | as for `p9c`                                                              |
| `b1` paired differences                                        | paired percentile bootstrap                                    | the test items                                                            |
| `p9d` slopes                                                   | percentile bootstrap                                           | the items, paired across arms                                             |
| `m2` cell metrics (`pair_lo`, `pair_hi`, `auroc_lo`, `auroc_hi`) | percentile bootstrap                                         | the seeds (independent training sets)                                     |
| `m2` comparisons                                               | paired percentile bootstrap; DeLong; exact McNemar; Holm       | the report pairs (clusters), on the seed ensemble                         |
| `r8` task accuracies                                           | percentile bootstrap; Wilson interval for a single run         | the seeds; the test questions of a zero-shot or single-seed run           |

`a1` uses a t-interval because a percentile bootstrap of so few runs covers less than its level. The `a2` and `a5`
intervals and nulls are described in the docstrings of `experiments/a2_logit_lens.py` and
`experiments/a5_sae_features.py`. A `p9b` interval describes how much the fitted reader
varies with its training draw; it does not cover the noise of the test items.

## Reader comparison (M2)

`m2` (`experiments/m2_readers.py`) compares readers on the same features and items, as a factorial of scorer by
readout by layer rule, so a gain of pooling is never credited to nonlinearity.

- Scorers: LAT (`pca_diff`), difference of means (`mean_diff`), logistic regression with its `C` tuned, LDA and an
  MLP whose `alpha` is tuned on a grid of the same size (matched tuning), both grids listed from the strongest
  penalty so a tie keeps it, and TTPD (`ttpd`). TTPD reads the polarity tasks only and is skipped elsewhere with its
  reason in `summary.json`: without polarity labels it scores as difference of means does.
- Tasks: binary reading tasks (`sentiment`, `truth` and `utility` by default) and the polarity task `truth_polarity`,
  the Geometry of Truth statements of `cities`, `neg_cities`, `sp_en_trans` and `neg_sp_en_trans`. It reads truth as
  `truth_cities` reads one subset, with the loader's splits, and is evaluated on its own `val` and `test` statements.
  A statement's polarity is -1 in a `neg_` subset and +1 otherwise.
- Readouts: `last`, `concept`, `concept_mean`, `mean` (the stimulus mean), `max` (the feature-wise maximum of the
  stimulus tokens), `token_score_max` (the maximum over the stimulus tokens of a scorer's per-token score; the scorer
  is fitted on every token row with its stimulus's label, the max aggregation of recent probe work) and
  `attention` (the attention-pooled logistic reader, read by the logistic scorer only). The concept readouts are
  skipped for a task without a concept slot or word.
- Extraction: one pass over the stimulus tokens gives `mean`, `max` and the token rows of the per-token readouts, one
  pass over the concept tokens gives `concept` and `concept_mean`, and one pass reads `last`. Each pass is reduced
  batch by batch, so the token rows of every layer are held for one batch at a time, and the pooled rows equal those
  of a pass per position (the means within float32 rounding of their sums). `token_layers` keeps the token rows at
  those layers only, so the per-token readouts choose among them while the one-row readouts keep every layer read. A
  task's activations are released before the next task is read.
- Layer rules: `best` (the best select layer), `top_k` (the mean of the select-standardized scores of the `top_k`
  best select layers, among those whose select scores vary) and `joint` (one MLP or one attention reader fitted on
  the layers `top_k` reads, together). The rules of a scorer and readout share the same per-layer fits and seeds.
- Every choice uses the select items of the original prompts only: the penalty per layer (first best on the grid),
  the layer rule (by pair accuracy) and the threshold of accuracy, precision, recall, F1 and the true positive rate at
  `max_fpr`. The report items are the same for every cell and condition.
- Conditions: the original prompts, P2's misleading sentence (`random`, `against`) and the held-out template wordings
  of `shifts` (default `paraphrase`).
- Controls: the text-only classifier on the raw texts (`p9.risk.text_scores`, word n-grams), a random direction at
  every readout, and every scorer refitted on training labels shuffled within their pairs (`shuffled` series), which
  runs the whole procedure on shuffled labels: the select labels are shuffled within their pairs too, so the penalty,
  the layer rule and the thresholds are chosen on them, and only the report metrics use the real labels. The random
  direction is oriented and its layer chosen on the real labels, so it measures an untrained direction, not chance,
  and can read well above 0.5. The text-only classifier reads the raw stimulus texts, so its scores are the same
  under every condition by design: it measures how much the stimulus content alone gives away, as `tb1`
  does, and the decoy and template conditions are measured on the readers.
- Shuffled series alone: `--set reader_series=false` (which needs `shuffled` in `controls`) fits only the `shuffled`
  series, at about half the cost of a full run: no reader fits, random direction or text-only classifier, so no
  comparisons and no AUROC matrix, and the summary says the run holds the shuffled series only. Its seeds do not
  depend on the other series, so its rows and per-seed scores equal the shuffled ones of the full run of the same
  other settings, and pair with that run's reader rows.
- Statistics: each cell's metrics are means over the 10 seeds with seed intervals. Comparisons use the seed
  ensemble: each seed's scores are standardized with their select mean and standard deviation and averaged over the
  seeds. Against each baseline (`logistic/last` and `mean_diff/last` by default, at the same layer rule, or `top_k`
  for a `joint` cell) the report AUROC difference gets a paired bootstrap that resamples whole pairs, the pair
  accuracy difference one that resamples pairs, DeLong's test (rows treated as independent) and the exact McNemar
  test on pair outcomes. A Holm family is one task, condition, layer rule and baseline, separately for the DeLong
  and McNemar p values. All statistics come from `reading/stats.py`.
- Runs load the model in float32.

## Published finder baselines (B2)

`b2` (`experiments/b2_finder_baselines.py`) puts published direction finders beside LAT (`pca_diff`, the method
under study) on truth and sentiment: the same training items, layers, layer rule and test pairs for every series. It
has not been run yet.

- Baselines: the mass-mean probe (`mean_diff`) and the whitened (IID) mass-mean probe (`whitened_mean_diff`) of
  Marks and Tegmark (2024, `marks2024geometry`), the logistic-regression probe (`logistic`, scikit-learn `C` 1.0;
  Alain and Bengio 2016, `alain2016understanding`) and CCS (`reading/ccs.py`; Burns et al. 2023,
  `burns2023discovering`). Their departures from the published recipes are in the CCS and whitened mass-mean rows
  of the deviations table above.
- Controls: `random` (a random unit direction with the same orientation rule) and `shuffled` (the LAT reader scored
  against labels shuffled within each pair), the series of the shared sweep, and `text_word` and `text_char`, the
  word and character n-gram classifiers of `tb1` fitted on the raw training texts of each seed. The `kind` column
  of `table.json` says `reader`, `baseline` or `control` for every row.
- Truth: three Geometry of Truth sets, each statement paired with the row of the same index in its negated set
  (`cities` with `neg_cities`, `larger_than` with `smaller_than`, `sp_en_trans` with `neg_sp_en_trans`), the
  contrast pairs of Marks and Tegmark. The labels must flip in every row, or the run stops. A pair is a group of two
  with exactly one true member and takes the loader's split of its statement, so a negation follows its statement
  into train, validation or test. The stimuli use R7's template and concept word.
- Sentiment: SST-2 with R1's template, training examples and seeds, so the training sets are R1's. SST-2 has no
  labelled test split, so the validation sentences are halved by the seeded draw `task_splits` makes for the reading
  task `sentiment`: one half chooses the layer, the other is the test set. Each half keeps the sentences of at least
  3 words, as R1's validation set does.
- Sizes: 10 training sets (`n_seeds`) of 64 pairs (`n_pairs`), as in R1 and R7, and every validation and test pair
  a split allows. TruthfulQA MC1 is not read.
- CCS contrast pairs: for truth, the statement and its negation, both through R7's template. Sentiment has no
  negation rows, so each text goes into one prompt in the form of Burns et al., `Consider the following example:
  ''' <text> '''` then a line `Between negative and positive, the sentiment of this example is`, completed after
  the assistant tag once with `positive` and once with `negative`; Burns et al. average 8 to 13 prompts, here there
  is one. So on sentiment CCS reads its own prompts while the other finders read R1's, on the same test pairs. The
  layer is chosen on validation like every series, not the last or a fixed layer of the references.
- CCS scoring: accuracy comes from `CCSProbe.pair_probabilities`, never from the probe's exported reader, which
  drops the per-set centring and so shifts every comparison between a statement set and a negation set with
  different means. A stimulus scores the probability that it is true: the pair probability for a statement or a
  sentiment text, its complement for a negation, and the group accuracy reads those scores.
- Layer rule and metric: every layer series takes the layer with the best mean validation group accuracy over the
  seeds on its own validation curve (the first one on ties), and reports the mean test group accuracy of the seeds
  at that layer; the text-only controls read no layer. Each row carries a Wilson 95 percent interval over the test
  pairs (`wilson_lo`, `wilson_hi`, as `tb1`) and a percentile bootstrap over the seeds (`boot_lo`, `boot_hi`): the
  first covers the sampling of the test pairs, the second the variation between training sets.
- Outputs: `table.json`, `curves_<dataset>_val` and `curves_<dataset>_test` (the accuracy of every layer series per
  seed and layer), and `summary.json` (the pair counts and the mean CCS training loss at its validation layer).

## LAT against few-shot prompting (R8)

`r8` (`experiments/r8_qa_correctness.py`) replicates the QA protocol of Fig. 7 and Table 9 on one checkpoint:
LAT against few-shot prompting on OpenBookQA, CommonsenseQA, ARC-Easy, ARC-Challenge and RACE, plus TruthfulQA MC1
(Table 1). The templates, splits, shot counts and prompts are those of `qa_tasks.py`.

- LAT: the reference code's templates as plain text with no chat wrapper, one stimulus per choice, read at the last
  token. Each seed's training questions (`qa_tasks.SHOTS`: 5 for OpenBookQA, 7 for CommonsenseQA, 25 for ARC, 3 for
  RACE, the 6 TruthfulQA primers) are paired inside each question without labels (`within_group`), the direction
  is `pca_diff` on centered differences (`preprocess="unit"` is the App. C.1 variant) and its sign follows the
  reference rule (`RepReader.fit`); another `method` changes the finder, and the finders that fit on labels
  (`mean_diff`, `logistic`) then use the k shots' gold choices. Accuracy is the share of questions whose correct
  choice scores highest, ties sharing credit. Seed `i` draws with `seed + i`, so the default seeds are the
  reference's 0 to 9.
- Layer rules, each giving the layer a seed is tested at: `reference` (the best accuracy on the labeled
  layer-selection questions, the reference's validation split), `k-shot` (the best accuracy on the seed's own
  training questions, so no labeled example beyond the k shots is used) and `test-oracle` (one layer for every
  seed, the best mean test accuracy: an upper bound for one layer shared by every seed, not for a layer chosen per
  seed, and not a protocol). Ties go to the deeper layer, as the
  reference's scan from the last layer does. Training accuracy often reaches 1.0 at many layers, so the `k-shot`
  rule then takes the deepest of them. CommonsenseQA's layer-selection questions depend on the seed (the training
  questions it did not draw); they are read once for the union of the seeds and each seed counts its own. On
  TruthfulQA the layer-selection questions are the 6 training primers, as in the reference, so the `reference` and
  `k-shot` rules score the same questions and pick the same layers there (unless `n_eval_pairs` keeps fewer than
  6); both are still reported, with their own rows.
- Few-shot: the lm-evaluation-harness prompt format (`fewshot_prompt`), with each seed's own k training questions as
  the examples, so the shots are those its LAT reader was fitted on (the Fig. 7 caption: "Using the same few-shot
  examples"). The first `n_fewshot_seeds` seeds are scored, all by default. RACE and TruthfulQA are zero-shot
  (`race_prompt`, `tqa_prompt`) and run once; TruthfulQA MC1 is its `acc`. A run reports `acc` (the summed
  log-likelihood) and `acc_norm` (per character), ties sharing credit. The harness reports `acc` only for RACE and
  TruthfulQA MC1, so the `acc_norm` average of `figure7` takes RACE's `acc` and the `acc_norm` of the other four
  tasks; the `acc` average takes every task's `acc`. Table 9 imports its few-shot numbers from other evaluations, so
  this baseline is measured here, not taken from the paper.
- Call grouping: one `judges.continuation_logprob` call scores every (question, choice) pair of a seed and task
  (one call per task for a zero-shot task). The call runs the token prefix shared by all its contexts once (the
  k-shot block), reuses its key-value cache for every question, and computes logits only at the continuation
  positions. Its scores depend on the grouping at float32 rounding level, because the shared prefix moves with the
  other contexts of the call (up to 2.1e-05 on `gpt2`), so the grouping is fixed at one call per seed and task. No
  BOS is prepended, as in the harness.
- BOS: `bos="none"`, where the Qwen3 base models default to a prepended separator. The templates are plain text for
  every checkpoint and the Qwen3 tokenizer adds no BOS, so the base and post-trained checkpoints read the same
  tokens the reference code would. A config with another `bos` is refused, because the few-shot scorer never
  prepends one; `layers`, when set, must be strictly increasing.
- Memory: runs load the model in float32. The LAT reads run the decoder without the LM head, and the few-shot scorer
  never computes full-vocabulary logits over a whole prompt. `race_batch_size` (1 by default) sets the batch of the
  RACE stimuli, of up to about 1,110 tokens with Qwen3's tokenizer; the 25-shot ARC contexts reach about 1,220
  tokens and the scorer runs each sequence alone. A task's activations are released before the next task is read.
  Within a task, the training sets and the test split are read first and released before the layer-selection split
  is read, so the host holds one evaluation split at a time: for Qwen3-1.7B (float32, 28 layers, about 229 KB per
  stimulus) the largest, ARC-Easy's test split of about 9.5k stimuli, takes about 2.2 GB, and about twice that for a
  moment while its batches are joined.
- Smoke runs: `n_eval_pairs` keeps that many questions of the layer-selection and of the test split; None keeps them
  all. `n_pairs` is not used.
- Outputs: each task writes `curves_<task>` (test accuracy per seed and layer of the reader and the random and
  shuffled controls), `lat_<task>` (the reader's accuracy on its training and layer-selection questions),
  `fewshot_<task>` (the log-likelihood of every test choice per few-shot seed) and last `task_<task>` (its table rows
  and summary). The run then writes `table.json` (one row per task, method, layer rule, metric and seed) and
  `summary.json`, whose `figure7` entry is Table 9's average row over the Fig. 7 tasks of the run, the data of the
  Fig. 7 bars. There is no resume: an interrupted run keeps the files of the tasks it finished, and the others can
  run with `--set tasks=[...]`, which is a new config and run directory.

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

### Published detection baselines

These baselines predict, per question, whether the greedy answer is correct. The `b1` experiment
(`experiments/b1_detection_baselines.py`) puts them beside the `p9c` signals; it has not been run yet. Each states
where it departs from its published recipe.

- Exact-answer-token probe (`p9/answer_probe.py`; Orgad et al. 2025, `orgad2025llms`): a logistic-regression probe
  on the residual stream at the last token of the greedy answer, read from the prompt, a space and the answer. It is
  fitted like the `p9c` `probe` (the `logistic` finder on the train items) at every layer, and the layer with the
  highest validation AUROC is kept. Deviations: the whole stored answer is the exact-answer span, so no extraction
  model runs and no item is dropped (the reference extracts a span from a long generation and drops items without
  one); the residual stream is read instead of the MLP output; the layer is chosen automatically on the validation
  split instead of by hand from a heatmap; the answer is re-tokenized from the stored text (cut at the first newline
  and stripped), so the read token can differ from the emitted one (for example `.` for an emitted `.\n`). An empty
  answer has no token to read and gets no score. Train items are scored in sample, so an AUROC uses only the items
  outside the train and validation splits.
- EigenScore (`p9/eigenscore.py`; Chen et al. 2024, `chen2024inside`): the mean natural log eigenvalue of the
  K x K covariance of the sampled answers' embeddings plus `alpha` I (`alpha` 1e-3). An embedding is the residual
  stream at the last token of the sample, read from the prompt, a space and the sample, at layer `(L + 1) // 2` (the
  released code takes half of the L + 1 hidden states; L / 2 for every model here). Higher means more divergent
  samples. Paper and code both centre each embedding over its own features (`Z^T J_d Z`); the released code also
  divides by d - 1, which the paper's Eq. 4 does not, so `alpha` sits on a different eigenvalue scale. The code is
  followed. Because of the centring, identical samples keep one eigenvalue `K var(z)` and do not score
  `ln(alpha)`. Deviations: K = 5 stored P9a samples at temperature 1.0 (the paper uses 10 at temperature 0.5 with top-p
  0.99); no feature clipping (it is not in the released code); the last token of the stored sample text instead of
  the last token before a stop token; correctness by the P9 rule. Empty samples are dropped and a question with fewer
  than 2 non-empty samples gets no score.
- Discrete semantic entropy (`p9/semantic_entropy.py`; Farquhar et al. 2024, `farquhar2024detecting`): the entropy
  (natural log) of the share of sampled answers per cluster of equal meaning. Two answers are equal when each
  entails the other, by the most likely of the three classes of an entailment model on `"{question} {answer}"`
  with no threshold, and the clusters are built greedily in sampling order as in the released code. Higher means
  more uncertain. The entailment model is DeBERTa-large MNLI (`microsoft/deberta-large-mnli`, the model of Kuhn et
  al. 2023 and the one the paper's text names); the released code runs DeBERTa-v2-xlarge MNLI. Deviations: M = 5
  stored P9a samples at temperature 1.0 and top-p 1.0 (the paper uses 10, and its text names top-p 0.9 and top-k
  50); identical answers are equal and an empty answer equals only another empty answer, with no model call (the
  reference sends every pair through the model); only the discrete variant, because P9a stores no token
  log-probabilities; correctness by the P9 rule.

The `b1` protocol:

- Inputs: a finished `p9c` run of the same model (`reliability_run`) and the `p9a` run it names. The datasets, items
  and splits are the ones `p9c` makes from the same labels (behavior items, the greedy answer's correctness as the
  outcome). Before any model runs, the test outcomes, and the test relations of PopQA, stored in the `p9c` signals
  must equal the ones made here, or the run stops; the `p9c` predictors are read from those signals, not recomputed.
- Scores: higher means more likely correct, as in `p9c`. The probe is fitted on the train items of each dataset at
  every layer the `p9c` run read, and its layer is chosen per dataset on the validation items (the paper also
  chooses per dataset). Semantic entropy and EigenScore keep their papers' sign and are negated; they fit nothing,
  so only the test items are scored and no sign is set on train. A missing score (an empty greedy answer for the
  probe, a question without samples for semantic entropy, fewer than 2 non-empty samples for EigenScore) takes the
  lowest finite score of the train items (probe) or of the test items (the other two), the least confident case,
  as in `p9c`. `n_filled` counts the missing scores among the items a baseline scored: every item of the dataset
  for the probe (as `p9c` counts its signals), the test items for the other two.
- Metrics: the `p9c` test metric with the `n_boot` and `seed` of the `p9c` run: AUROC with its interval, AURC and,
  for PopQA, the within-relation AUROC. `table` holds the `p9c` rows of each dataset, copied, then one row per
  baseline; `table_paired` holds the AUROC of every baseline minus that of each reference (`lat` and `p_true` by
  default) with a paired bootstrap interval over the test items. A reference the `p9c` run lacks (`lat` on GPT-2
  small, whose gate admits no contrast) is named in the summary and skipped.
- Settings: the model loads in float32, because the activations reuse the cached few-shot prefix, which matches the
  uncached path only in float32. `p9a` and `p9c` ran Qwen3 in bfloat16 and `gpt2-xl` in float16, so the baselines
  read the same answers at a higher precision than the `p9c` signals. The entailment model runs on the CPU, so it
  never shares the GPU with the model: about 0.07 s per pair on the laptop CPU (640 pairs timed), and 30k to 45k
  pairs per checkpoint (counted from the `p9a` samples). The summary records the seconds each baseline took.

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
