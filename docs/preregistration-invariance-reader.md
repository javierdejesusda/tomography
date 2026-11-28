# Preregistration: a two-position invariance reader for binary concepts in frozen LLM hidden states

Version 1, written 2026-11-27. This document is committed to the main branch
before any code for the runs it describes and before any of those runs. The
commit that adds this file is the timestamp of record. Nothing below has been
run: every number in sections 2 to 4 describes existing data or a frozen
plan, and no number is a result of this study.

Terms used throughout:

- A **text** is one stimulus (a sentence, a statement or a scenario) with a
  binary label.
- A **wrapper** is everything around the stimulus in the prompt: the template
  wording and, optionally, a decoy line.
- A **decoy** is a sentence in the prompt that asserts a level of the concept
  ("low" or "high"), whether or not that matches the label.
- A **reader** maps hidden states of a frozen language model to a score for
  the label of the stimulus. The model is never updated.
- **M2** is the reader comparison already in this repository
  (`src/tomography/experiments/m2_readers.py`). Its report items are treated
  as burned, because this design was chosen after looking at its results.

## 1. Question and hypotheses

### 1.1 Question

With 128 labelled texts, can a reader on frozen hidden states beat the
current strongest readers (L2 logistic regression, difference of means,
published multi-layer readers) on its worst condition, when the prompt
carries a misleading sentence or an unseen template wording, without losing
accuracy on clean prompts?

The target is the reader itself: it has to beat the baselines, not only pass
a robustness check. Depth enters only as a declared choice of layer per read
position; reading across layers is a baseline here, not the method.

### 1.2 Stage 1 hypothesis (gate)

H1. On Qwen3-0.6B base, with held-in data only, the stage 1 winner (section
6.3) beats the better single-position augmented logistic regression by at
least 0.02 in worst-condition AUROC averaged over the four tasks, and the
pair-bootstrap 95% interval of that difference lies above 0.

H1 is falsified if the point difference is below 0.02, or if the interval
includes 0, or if the winner is itself a single-position augmented logistic
regression or a plain logistic regression. That last clause is how the gate
is operationalized, not an extra hypothesis (section 6.4). If H1 is
falsified, no claim of a new reader is made (section 6.4).

### 1.3 Stage 2 hypothesis (confirmation)

H2. With the pipeline frozen after stage 1, on fresh report items, the reader
beats the best baseline (augmentation-matched baselines included) in
worst-condition AUROC in at least 9 of 16 Qwen combinations and at least
5 of 8 Llama combinations (one-sided alpha 0.05, Holm within family), it is
non-inferior on clean prompts in each counted combination, and the pooled
item-level effect is above 0 (section 7.5).

H2 is falsified if any of the three parts fails. A combination that fails
clean non-inferiority does not count as a win, whatever its p value. This is
how the clean non-inferiority clause of the stage 2 rule is operationalized:
it is checked per combination, on each counted win (section 7.5).

### 1.4 Prior stated before the runs

The internal technical review that preceded this design expects most of the
gain to come from the label-preserving wrapped copies (data augmentation),
not from the covariance penalty. Its estimate: about even odds of winning a
majority of the 16 Qwen combinations against clean baselines, and a win
against augmentation-matched logistic regression is unlikely (expected
increments of 0.02 or less). A synthetic check in the same review showed why
the earlier candidate (LDA with a covariance from a class-mixed unlabelled
pool) was dropped: clean AUROC 0.835 against 0.888 for L2 logistic regression
at a class separation of 4. We therefore expect that H1 may fail, and the
gate exists so that a failure is reported as such.

### 1.5 What is and is not claimed as new

Not new: the LDA direction `(Sigma + lambda I)^-1 d` is published for LLM
probes (Marks and Tegmark, mass-mean probe with a labelled covariance,
https://github.com/saprmarks/geometry-of-truth; Sadeghi and Becker, arXiv
2611.19499; Lee, Seo and Lim, arXiv 2608.28930, who report Ledoit-Wolf LDA at
0.920 AUROC against 0.952 for L2 logistic regression). Rebuilding LDA from
unlabelled data and a mean difference is published outside LLM probing
(MILDA: Heintz, Francart and Bertrand, arXiv 2310.11110). Whitening with an
unlabelled covariance before a linear probe is published for vision and
encoder models (Holstege et al., arXiv 2609.39177). This repository already
has a Ledoit-Wolf LDA reader (`src/tomography/reading/probes.py:97-131`).

Claimed, if the hypotheses hold: a reader on two read positions with a
class-free within-item covariance penalty estimated from unlabelled wrapped
texts, tested under asserted-label distractors and template shift at 128
labels on two model families. The penalty has the form of nuisance attribute
projection and within-class covariance normalization in speaker verification
and of the conditional variance penalty of CoRe in domain generalization
(Heinze-Deml and Meinshausen, Machine Learning 2021). A search before this
preregistration did not find it applied to LLM probes; that is an absence of
a hit, not a proof. The novelty of the penalty for LLM probes will be
searched again before any paper is written, and the result will be stated in
the paper either way.

## 2. Models and tasks

### 2.1 Models

| Key | Hugging Face repository | Revision | Role |
|---|---|---|---|
| `qwen3-0.6b-base` | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | stages 0, 1 and 2 |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | stage 2 |
| `qwen3-1.7b-base` | `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | stage 2 |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | stage 2 |
| Llama-3.2-1B base | `meta-llama/Llama-3.2-1B` | pinned before stage 2 | stage 2 only |
| Llama-3.2-1B instruct | `meta-llama/Llama-3.2-1B-Instruct` | pinned before stage 2 | stage 2 only |

The Qwen revisions are the ones in the model registry
(`src/tomography/model.py:156-179`). The two Llama checkpoints are not in the
registry yet; their revision hashes are pinned in the commit that adds them,
which is logged in section 9.3 as an addition. No forward pass of either
Llama checkpoint is run for this study before stage 2: no layer, wording or
hyperparameter is chosen on them.

Base checkpoints are prompted with plain `USER:` and `ASSISTANT:` tags, and
post-trained checkpoints with their own chat template with thinking disabled
(`src/tomography/model.py:130`, `src/tomography/templates.py:38-40`,
`src/tomography/templates.py:67-93`). The same rule applies to Llama (base:
plain tags, instruct: its chat template).

Layers are addressed by relative depth. For a model with `L` blocks, the
candidate layers are the outputs of blocks `ceil(0.4 L)` to `L` (relative
depth 0.4-1.0). The Qwen3-0.6B and Qwen3-1.7B checkpoints were read up to
layer 28 in M2 (the `token_layers` of
`results/m2/qwen3-0.6b-base/dfd0cd00a2b7/config.json` and of
`results/m2/qwen3-1.7b-base/84a62d2075b6/config.json` end at 28), which
gives layers 12 to 28; Llama-3.2-1B has 16 blocks, which gives layers 7 to
16. The block count is read from the model configuration at load time and
the rule, not these indices, is binding. No layer index is ever transferred
between models.

### 2.2 Tasks and sources

| Task | Source | Label 1 | Pairing |
|---|---|---|---|
| sentiment | SST-2 (`stanfordnlp/sst2` at `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`, `src/tomography/datasets/sst2.py:24-35`) | positive | one positive with one negative text |
| truth | Geometry of Truth, subset `common_claim_true_false` (`src/tomography/datasets/geometry_of_truth.py:31-70`) | true | one true with one false statement |
| truth with negations | Geometry of Truth, subsets `cities`, `neg_cities`, `sp_en_trans`, `neg_sp_en_trans` (`src/tomography/experiments/m2_readers.py:124-133`) | true | one true with one false statement |
| utility | ETHICS utilitarianism (`src/tomography/datasets/ethics.py:27-45`) | the more pleasant scenario of a source row | the two scenarios of a source row |

Task definitions otherwise follow the repository: concept word and decoy noun
are "sentiment" and "positive sentiment" for sentiment, "truthfulness" and
"truthfulness" for both truth tasks, and no concept slot with the noun
"happiness" for utility (`src/tomography/experiments/variants.py:136-184`).

### 2.3 What M2 used (burned)

Counts obtained by running the repository loaders and `task_splits`
(`src/tomography/experiments/variants.py:487-541`) at seed 0 on the base
commit, and checked against the stored M2 score arrays. M2 trained on 64
pairs (128 texts) per draw, 10 draws, and evaluated on at most 256 pairs
(`src/tomography/experiments/sweep.py:63-64`,
`src/tomography/experiments/m2_readers.py:236`).

| Task in M2 | Train split | Select items | Report items | M2 select pairs | M2 report pairs |
|---|---|---|---|---|---|
| sentiment (SST-2) | 55,838 texts with 3 or more words (24,910 pairs) | 436 texts | 436 texts | 214 | 214 |
| truth (facts set) | 349 texts (172 pairs) | 117 texts | 131 texts | 53 | 59 |
| truth with negations | 2,234 texts (1,101 pairs) | 763 texts | 703 texts | 256 of 368 | 256 of 349 |
| utility | 20,530 texts (10,265 pairs) | 6,946 texts | 9,616 texts | 256 of 3,473 | 256 of 4,808 |

For sentiment, the select and report items are the two halves of the SST-2
validation split (872 sentences); the source test split has no labels
(`src/tomography/datasets/sst2.py:1-7`). So M2 used every labelled
full-sentence item of SST-2 outside the train split, and every item of the
facts set outside its train split.

### 2.4 Where the fresh items come from

The M2 truth task (the facts set, `src/tomography/datasets/facts.py:1-7`)
cannot supply this study: it has 597 statements in total (297 true, 300
false), so at most 297 pairs exist, and its 53 select pairs and 59 report
pairs are burned. The truth task of this study therefore uses the
`common_claim_true_false` subset of Geometry of Truth, a source already
pinned in this repository: 4,450 statements, of which 2,626 in the train
split (1,296 true, 1,330 false), 911 in the validation split and 913 in the
test split (452 true, 461 false). Three statements match a facts statement
after normalization (lower case, collapsed whitespace, trailing period
removed), one in each split, and are removed; the one in the test split is
true. Three more statements appear twice with different capitalization (two
of them twice inside the train split, one across the train and validation
splits), so deduplication uses the same normalized text, not the exact
string: the first row of a repeated statement is kept, and a statement whose
copies sit in two source splits is removed from both. That leaves 2,622
train statements (1,294 true, 1,328 false) and 912 test statements (451
true, 461 false). Consequences, stated now: the truth numbers of
this study are not comparable with the M2 truth numbers, and every baseline
is rerun on the new items. One earlier experiment in this repository scored
other readers on the test split of this subset
(`src/tomography/experiments/p5_persona_wording.py:324`); no design choice
here used those numbers.

All partitions are fixed before stage 0 by a seeded hash of a group key
(base seed 0), in this block order: reserved test block (where listed),
validation, fit reservoir, pool.

| Task | Stage 0 and 1 items (fit reservoir, pool, validation) | Stage 2 report items | Fresh report pairs |
|---|---|---|---|
| sentiment | SST-2 train split, 3 or more words | reserved block of the SST-2 train split, texts of 10 or more words | 256 |
| truth | `common_claim_true_false` train split | `common_claim_true_false` test split | 256 (of 451 available) |
| truth with negations | train split of the four subsets | reserved block of the same train split, drawn only from subjects with no statement in the M2 validation or test split | 256 (of 326 available) |
| utility | ETHICS utilitarianism train split | ETHICS utilitarianism test split, minus every pair linked by a shared scenario to the source train file or to an M2 report pair | 256 (of 803 available) |

Disjointness rules:

- Every block of a task is disjoint by group key from every other block.
- Sentiment: SST-2 train items are phrases cut from parse trees, so two
  phrases of one sentence are either nested or cover disjoint positions of
  it. Disjoint phrases can still share words, and the source has no sentence
  identifier, so two disjoint phrases of one sentence can fall in different
  blocks (a stated limit, section 8.4). The 10-word floor leaves 12,987
  positive and 12,165 negative candidates for the reserved block.
  Containment is tested on the normalized text (lower case, collapsed
  whitespace) as a contiguous run of whole words. Before the reserved block
  is drawn, every candidate that contains, or is contained in, a sentence of
  the SST-2 validation split (all of it burned by M2) is removed: at the
  time of writing, 18 candidates, all containing the validation sentence
  "bad .". After the reserved block is drawn, every train item whose
  normalized text contains, or is contained in, a reserved text is removed
  from the fit reservoir, the pool and the validation block. The same
  containment rule is applied between the validation block and the fit
  reservoir and pool.
- Truth with negations: the group key is the subject of the statement (the
  `city` column of the two city files, or the quoted Spanish word of the two
  translation files), so a statement, its negation and the other statements
  about the same subject always sit in one block. M2 split these subsets by
  row, not by subject (`src/tomography/datasets/geometry_of_truth.py:101`),
  so 808 of the 1,025 subjects of the train split also have a statement in
  the M2 validation or test split, and 3 train statements repeat a
  validation or test statement verbatim. The reserved report block is
  therefore drawn only from the 217 subjects whose every statement is in the
  train split: 652 statements (326 true, 326 false; 432 about cities, 220
  about Spanish words), so 326 pairs are available. Repeated statements (6
  in the four subsets, kept as published by the loader) are collapsed on the
  normalized text to their first row, and the 3 train statements that repeat
  an M2 validation or test statement are removed from every block, which
  leaves 2,230 distinct train statements. If the subject cannot be extracted
  exactly from the source files, the fallback key and the reason are logged
  in section 9.3 before stage 0.
- Utility: the group key is the set of source rows linked by a shared
  scenario text, directly or through a chain of rows, as in the loader
  (`src/tomography/datasets/ethics.py:6-10`). The loader links rows only
  inside the source train file
  (`src/tomography/datasets/ethics.py:216-220`), and the source test file
  repeats scenarios of the train file: 1,305 of the 4,808 test pairs share
  an exact scenario text with a pair of the train split (1,645 with the
  train or validation split), and 504 test pairs outside the 256 that M2
  reported on share one with those 256. The links are therefore recomputed
  over the source train and test files together, and every test pair in a
  linked group that contains a row of the source train file (train or
  validation split) or one of the 256 M2 report pairs is excluded. The M2
  report pairs are recomputed with the M2 seed
  (`src/tomography/experiments/variants.py:463-477`). That leaves 803 test
  pairs in 608 linked groups; a linked group is never split.
- Truth: train and test splits of the source are disjoint by statement after
  the normalized deduplication described above.

Block sizes, identical for the four tasks:

| Block | Size | Used for |
|---|---|---|
| fit reservoir | 192 pairs (384 texts) | the labelled training draws |
| validation | 200 pairs (400 texts) | the stage 1 endpoint only |
| pool | 640 texts, drawn without reading labels | the covariance estimates |
| report | 256 pairs (512 texts) | stage 2 only |

A draw is a seeded permutation of the fit reservoir; the first 64 pairs are
the 128 labelled texts of that draw. There are 10 draws, the same for every
model. Draws overlap, because they come from one reservoir. If the pool of a
task cannot reach 640 texts after the other blocks are filled, it takes all
remaining texts and the count is reported.

## 3. Prompts and wrappers

All wordings in this section are frozen by this commit. `{concept}` is the
concept word, `{noun}` the decoy noun, `{stimulus}` the text and `{level}`
the decoy level.

### 3.1 Clean template

Sentiment and both truth tasks (`src/tomography/templates.py:455-459`):

```text
Consider the amount of {concept} in the following:
{stimulus}
The amount of {concept} is
```

Utility (`src/tomography/templates.py:472-477`); the last line follows the
assistant tag:

```text
Consider the happiness in the following scenario:
Scenario: {stimulus}
[assistant] The happiness of the scenario is
```

### 3.2 Decoy placements and levels

A decoy is one extra line in the user turn of the clean template. Three
placements are used:

- `pre`: its own line directly before the line that holds `{stimulus}` (the
  M2 placement, `src/tomography/experiments/p2_positions.py:122-154`).
- `head`: the first line of the user turn.
- `post`: its own line directly after the line that holds `{stimulus}`.

Two level rules exist (`src/tomography/experiments/p2_positions.py:157-197`):

- `random`: the level is drawn per text from a seeded generator, independent
  of the label (lines 187-191). Here the seed also depends on the wording.
- `against`: label 1 gets the low level and label 0 the high level, so the
  decoy always contradicts the label (lines 185-186).

Level words are `low` and `high`
(`src/tomography/experiments/p2_positions.py:66`) unless a wording lists
others. Because the `against` level is a function of the label, an `against`
copy leaks the label. `against` copies are never used to fit, to tune or to
estimate a covariance; they appear only in evaluation (section 8.2).

### 3.3 Test wordings from M2 (never used in fitting or tuning)

M2 test decoy, placement `pre`
(`src/tomography/experiments/p2_positions.py:67-70`):

```text
I think the amount of {noun} in the scenario is {level}, but I'm curious to hear what you think.
```

The three test paraphrases (`src/tomography/templates.py:394-403`), which
replace the clean template for sentiment and both truth tasks; utility has
none:

```text
P1  Rate how much {concept} the following text expresses:
    {stimulus}
    The amount of {concept} is

P2  How much {concept} is there in this text?
    {stimulus}
    The amount of {concept} is

P3  Read the text below and assess its {concept}.
    {stimulus}
    Its {concept} is
```

### 3.4 Held-in fit set (training copies, pool, inner tuning)

Template variants (sentiment and both truth tasks):

```text
F-T1  Here is a passage.
      {stimulus}
      The level of {concept} in this passage is

F-T2  Text: {stimulus}
      Task: judge the {concept} of the text.
      Judgment: the {concept} is

F-T3  Evaluate the following with respect to {concept}:
      {stimulus}
      Evaluation: the {concept} is
```

Decoy wordings (all four tasks, `random` level only):

```text
F-D1  head  A colleague told me the {noun} here is {level}, though I have not checked it myself.
F-D2  pre   Before you answer: my first impression was that the {noun} of this one is {level}.
F-D3  pre   Note from an earlier reader: the {noun} of this text is {level}.     (levels: weak, strong)
```

### 3.5 Held-in score set (stage 1 endpoint only)

Template variants (sentiment and both truth tasks):

```text
S-T1  Take a look at this excerpt:
      {stimulus}
      The degree of {concept} it shows is

S-T2  {stimulus}
      Given the text above, determine its {concept}.
      The {concept} of the text is
```

Decoy wordings (all four tasks, `random` and `against` levels, evaluation
only):

```text
S-D1  pre   Someone on the forum insisted that the {noun} of the next text is {level}.
S-D2  head  For what it is worth, an automatic tool scored the {noun} below as {level}.
```

### 3.6 New test decoy wordings (stage 2 only)

```text
T-D1  pre   Everyone I asked says the {noun} of what follows is {level}.
T-D2  post  (A previous annotator marked the {noun} of the text above as {level}.)
T-D3  head  An expert reviewer is certain that the {noun} of the text below is {level}.     (levels: minimal, maximal)
```

T-D2 is structurally different: the decoy comes after the stimulus, a
placement no held-in wording uses. In a causal model the state at the last
stimulus token cannot depend on it, so under T-D2 only the last-token
feature sees the decoy. T-D3 is structurally different in its level words,
which no held-in wording uses.

### 3.7 Rules for the wording sets

- The fit set, the score set and the test wordings (3.3 and 3.6) are
  pairwise disjoint. No held-in wording is equal to, or a light edit of, the
  M2 test decoy or one of the three test paraphrases: no held-in decoy uses
  the frame "I think ..., but I'm curious to hear what you think", and no
  held-in template ends in "The amount of {concept} is" or "Its {concept}
  is".
- Decoys are applied to the clean template only; template variants carry no
  decoy. Utility uses decoy wrappers only, which mirrors its stage 2
  conditions.
- The fit set therefore holds 6 wrappers for sentiment and both truth tasks
  (3 templates, 3 decoys) and 3 for utility. Together with the clean
  template that is 7 and 4 renderings per text.

## 4. Unlabelled pool

Source: the training split of each task (section 2.4), 640 texts per task,
drawn in hash order from the items left after the reserved, validation and
fit blocks, without reading labels. Pool items are disjoint by group key
from fit, validation and report items, so no estimator ever sees a test text
(no transduction). The same pool items are used for every model; each model
encodes them itself.

"Unlabelled" means here: the labels of pool items exist in the source files,
but they are never passed to any estimator, tuning step or analysis of this
study. It does not mean the pool is a sample of unrelated text. The pool is
in-distribution task text, and the source sets are close to class balance,
which a pool collected in practice would not guarantee. This is a stated
limitation of the setting.

Each pool text `u_j` is rendered under the clean template and every fit-set
wrapper (`K` renderings, 7 or 4), with `random` decoy levels. With
`phi(u, a)` the standardized feature vector of text `u` under wrapper `a`
(section 5.1):

```text
mu_a      = (1/M) sum_j phi(u_j, a)                         centre per wrapper
c(u_j, a) = phi(u_j, a) - mu_a
e(u_j, a) = c(u_j, a) - (1/K) sum_a' c(u_j, a')             within-item residual
Sigma_e   = (1 / (M (K - 1))) sum_j sum_a e(u_j, a) e(u_j, a)^T
```

`Sigma_e` is then rescaled to a trace equal to the feature dimension, so the
penalty weight has the same meaning across layers and models. `Sigma_e` is
class-free by construction: the label of a text is the same under all its
wrappers, so anything that depends only on the text cancels in `e`, and no
label enters the formula. Centring per wrapper removes the mean shift of a
wrapper, which carries no information about one text.

## 5. Readers

### 5.1 Features, positions and standardization

- `h_last(x)`: the residual stream at the last token of the prompt.
- `h_stim(x)`: the residual stream at the last token of the stimulus: the
  last token whose character offsets overlap the stimulus span (span:
  `src/tomography/templates.py:234-236`; overlap rule:
  `src/tomography/activations.py:143-180`).
- Each feature is standardized per dimension with the mean and standard
  deviation of the clean labelled training texts of the fit at that layer
  and position; a dimension with zero standard deviation is divided by 1.
  Wrapped copies, pool, validation and report features use the same
  statistics. "The fit" is the training folds inside cross-validation and
  all labelled texts of the draw in the final refit (section 5.3), so no
  held-out fold enters a statistic.
- Two-position features: `phi(x) = [z_stim(x) ; z_last(x)]`, each position at
  its own layer.

Labels are `y` in `{-1, +1}`, `n` is the number of labelled texts in the fit
(128 in the final refit of the confirmatory analysis, 96 inside a
cross-validation fold) and `s(x) = w^T phi(x) + b` is the score.

### 5.2 Stage 1 arms

A1 and A2, logistic regression at the last token and at the last stimulus
token:

```text
min over w, b:  (1/n) sum_i log(1 + exp(-y_i s(x_i))) + (lambda/2) ||w||^2
```

A3 and A4, augmented logistic regression at each position. Every labelled
text is also rendered under each fit-set wrapper (`random` decoy levels) and
keeps its label. A text and its copies share a total weight of 1:

```text
min over w, b:  (1/n) sum_i (1/K) sum_a log(1 + exp(-y_i s(u_i, a))) + (lambda/2) ||w||^2
```

A5, augmented two-position logistic regression: the A3 objective on
`phi = [z_stim ; z_last]`.

A6, the lead, two-position invariance logistic regression: A5 plus the
covariance penalty of section 4,

```text
min over w, b:  (1/n) sum_i (1/K) sum_a log(1 + exp(-y_i s(u_i, a)))
                + (lambda/2) ||w||^2 + (gamma/2) w^T Sigma_e w
```

A7, U-LDA with a pseudo-label-centred covariance, at the last token:

```text
d_hat   = mean of phi over label +1  -  mean of phi over label -1      (the n labelled texts of the fit)
y'_j    = +1 if g(u_j) > 0, else -1                                     (pseudo-label of pool text u_j)
S       = (1/(M - 2)) sum over classes c, sum over j with y'_j = c of (phi_j - m_c)(phi_j - m_c)^T
S_alpha = (1 - alpha) S + alpha (trace(S)/d) I
w       = S_alpha^-1 d_hat
```

with `m_c` the pool mean of pseudo-class `c`, `M` the number of pool texts,
`d` the feature dimension and the pool rendered under the clean template
only. The pseudo-labeller `g` is the A1 objective at the same candidate
layer as the reader (never the layer A1 selects for itself), with `lambda`
fixed at 1e-2 so that it has nothing to tune. It is fitted on the same
labelled texts as `d_hat`: the training folds inside cross-validation, all
labelled texts of the draw in the final refit. No label of a held-out fold
reaches `g`, `d_hat` or `S`.

A8, U-LDA plus augmentation, at the last token: `d_hat` from the labelled
texts of the fit and their wrapped copies (weights as in A3). The
pseudo-labeller `g` is the A3 objective at the same candidate layer, with
`lambda` fixed at 1e-2, fitted on the same texts and copies. Each pool text
gets one pseudo-label, from `g` on its clean rendering, and its `K`
renderings share it. `S` uses all `M K` pool renderings (the clean template
and all fit-set wrappers), centred per pseudo-class and per wrapper:

```text
S = (1/(M K - 2 K)) sum over wrappers a, classes c, j with y'_j = c of
        (phi(u_j, a) - m_{c,a})(phi(u_j, a) - m_{c,a})^T
```

with `m_{c,a}` the mean of pseudo-class `c` under wrapper `a`.

Fallback for A7 and A8: if a pseudo-class holds fewer than 2 pool texts, `S`
is centred on the mean of all pool texts instead (per wrapper for A8), with
denominator `M - 1` (A7) or `M K - K` (A8). The number of fits that used the
fallback is reported.

Grids:

| Hyperparameter | Values | Arms |
|---|---|---|
| `lambda` | 1e-4, 1e-3, 1e-2, 1e-1, 1 | A1 to A6 |
| `gamma` | 1e-2, 1e-1, 1, 10, 100 | A6 |
| `alpha` | 0.01, 0.1, 0.3, 0.6, 0.9 | A7, A8 |
| layer | every layer at relative depth 0.4-1.0 | all |

`gamma = 0` is excluded from A6 on purpose, so that A6 and A5 stay distinct
arms.

### 5.3 Tuning

All tuning happens inside the labelled texts of the draw, by 4-fold
cross-validation by pair: the 64 pairs are split into 4 folds of 16 pairs
with a seeded permutation, and a text and all its wrapped copies stay in the
fold of its pair. Out-of-fold scores of all 128 texts are pooled and scored
once. No validation or report label is used for tuning, so the label budget
of a fitted reader is exactly `n`.

- Everything estimated from data is estimated inside the fold. The
  standardization statistics (section 5.1), `Sigma_e` (which is computed on
  the standardized features, section 4), `d_hat`, the pseudo-labeller and
  `S` (A7, A8) use the training folds and the unlabelled pool only, and are
  recomputed from all labelled texts of the draw for the final refit. The
  pool carries no labels, so using it inside every fold leaks nothing.
- Plain readers (A1, A2, A7 and the plain baselines of section 5.4) are tuned
  on the cross-validated clean AUROC. They are defined without wrappers and
  keep their published information set.
- Augmented readers (A3, A4, A5, A6, A8, the random-subspace control) are
  tuned on the cross-validated worst-condition AUROC over the clean template
  and each fit-set wrapper (`random` levels only).
- Layer per position: a single-position reader takes the layer with the best
  criterion at its best `lambda`. A two-position reader takes, for each
  position, the layer chosen by the augmented single-position reader at that
  position (A3 for the last token, A4 for the last stimulus token); `lambda`
  and `gamma` are then tuned for the two-position model at those two layers.
- Ties: the stronger penalty (larger `lambda`, then larger `gamma`, larger
  `alpha`), then the shallower layer.
- After tuning, the reader is refitted on all `n` labelled texts.

### 5.4 Stage 2 baselines

Each baseline is tuned as in section 5.3 and fitted on the same draws.

| Id | Baseline | Definition |
|---|---|---|
| B1, B2 | difference of means at the last token, at the last stimulus token | `w = mean(phi, +1) - mean(phi, -1)`, no hyperparameter, best cross-validated layer |
| B3, B4 | L2 logistic regression at the last token, at the last stimulus token | A1, A2 |
| B5 | attention pooling over stimulus tokens | the attention reader of M2 (`src/tomography/reading/probes.py:448-466`): a logistic head on a learned attention-weighted mean of the stimulus token states, one reader per layer (not joint), best cross-validated layer among the layers at relative depth 0.4, 0.55, 0.7, 0.85 and 1.0 (rounded up). It keeps its M2 objective and scaling: `c` times the summed loss plus half the squared weights, on rows centred on their mean and divided by their root mean square norm, in place of the standardization of 5.1. The `lambda` grid of 5.2 is mapped with `c = 1/(n lambda)` |
| B6 | depth mean (HalluTracer; Guo et al., arXiv 2608.16353) | an L2 logistic regression per layer at the last token, on every layer at relative depth 0.4-1.0, with one `lambda` shared by all layers; each layer's score is standardized with the mean and standard deviation of its in-sample scores on the labelled texts of the fit (zero standard deviation: divided by 1); the score of the reader is the uniform mean over the layers. The shared `lambda` is the one with the best cross-validated clean AUROC of that mean score |
| B7 | 5-layer stack (Nordby, Pais and Parrack, arXiv 2604.13386) | an L2 logistic regression per layer at the last token, each layer with its own `lambda` (the A1 tuning at that layer); 5 layers chosen greedily on the out-of-fold predictions of the `n` texts, starting from the best cross-validated layer and adding each time the layer that minimizes the largest pairwise double-fault rate of the chosen set (a double fault: both layers misclassify the text, the prediction being label 1 if and only if `s(x) > 0`; ties: the higher cross-validated AUROC, then the shallower layer); a second logistic regression (`lambda` = 1e-2) on the 5 out-of-fold scores, each standardized with the mean and standard deviation of its `n` out-of-fold scores (zero standard deviation: divided by 1). At test time the 5 layer readers, refitted on all `n` texts, give the scores, standardized with those same statistics |
| B8, B9 | augmented logistic regression at the last token, at the last stimulus token | A3, A4 (augmentation-matched) |

B6 and B7 are adaptations, and are named as such in every table. The
published HalluTracer reader averages probe logits over all layers, on the
best attention head per layer, at the token before answer generation; B6
uses the residual stream and the layers at relative depth 0.4-1.0. The
published stack of Nordby et al. reads the positions of a dataset detection
mask with a fixed regularization; B7 reads the last token and tunes
`lambda`. Any further difference found while implementing them is logged in
section 9.3 before stage 2.

### 5.5 Controls

Controls are reported next to the readers and never enter the "best
baseline" of section 7.

| Control | Definition |
|---|---|
| random subspace | A6 with `Sigma_e` replaced by `Q Sigma_e Q^T`, `Q` a seeded random orthogonal matrix: same spectrum, random directions. It separates the effect of the estimated directions from the effect of extra regularization. It is defined on A6 only: it is reported when the reader is A6 and is not run when another arm wins, because no claim about the penalty is then made (section 6.4) |
| two-position augmented logistic regression | A5, which is A6 without the penalty; when the reader is A5, this row is the reader itself |
| canonical re-prompt oracle | every wrapped report text is rendered again in the clean template (wrapper stripped) and read; reported as a bound (section 8.3) |
| shuffled labels | the frozen stage 1 winner (the reader), whichever arm it is, refitted on labels shuffled within pairs (`src/tomography/experiments/sweep.py:194-200`) |
| text only | the word TF-IDF logistic regression of the repository on the stimulus texts (`src/tomography/p9/risk.py:486-499` with `analyzer="word"`, as in `src/tomography/experiments/m2_readers.py:938`: word 1-grams and 2-grams seen in at least 2 texts), fitted on the same labelled texts |
| random direction | a seeded random unit direction at the layer of B3, oriented on the training labels, as the `random` control of M2 (`src/tomography/experiments/m2_readers.py:326-331`) |

## 6. Stage 0 and stage 1

### 6.1 Stage 0: one GPU pass

Model: Qwen3-0.6B base only. Items: training-split items only (fit
reservoir, pool, validation; the reserved and report blocks are not
encoded). Renderings: fit reservoir and pool under the clean template and
the fit-set wrappers; validation pairs under the clean template and the
score-set wrappers (`random` and `against` for the decoys). Both positions,
every layer at relative depth 0.4-1.0, cached to disk. That is 9,968 prompts
per task for sentiment and both truth tasks and 6,096 for utility, 36,000 in
total.

### 6.2 Stage 1: tournament on the cache

The eight arms of section 5.2 are fitted on each of the 10 draws of each
task and scored on the 200 validation pairs.

Conditions: clean; S-D1 and S-D2, each at `random` and `against`; S-T1 and
S-T2 (7 conditions; 5 for utility, which has no template variants).

Endpoint, per arm and task: the AUROC of each condition is averaged over the
10 draws, and the endpoint is the minimum over conditions (worst-condition
AUROC).

### 6.3 Selection rule

- Best baseline of a task: the better of A1 and A2 on the endpoint.
- Gain of an arm in a task: its endpoint minus that of the best baseline.
- Eligible arms: those whose clean AUROC (mean of 10 draws) is at least the
  clean AUROC of A1 minus 0.01 in every task.
- Winner: the eligible arm with the highest mean gain over the four tasks.
- Ties: every eligible arm within 0.005 of the highest mean gain is tied
  with it, and the tie goes to the arm that comes first in this order,
  simplest first: A1, A2, A3, A4, A5, A7, A8, A6.

### 6.4 Gate

Comparator: of A3 and A4, the arm with the higher mean endpoint over the
four tasks; a tie goes to A3, the earlier arm in the order of section 6.3.

Statistic: mean over tasks of (endpoint of the winner minus endpoint of the
comparator). Interval: paired bootstrap over validation pairs, 10,000
resamples, pairs resampled within each task, the same resampled pairs for
all arms, all conditions and all draws, the minimum over conditions
recomputed inside every resample; two-sided 95% percentile interval. The
comparator is chosen again inside every resample by the rule above, so the
maximum over A3 and A4 is inside the bootstrap, as in section 7.3. The
winner is not chosen again: it stays the arm selected on the full
validation set.

The gate is passed if the statistic is at least 0.02 and the interval lies
above 0. It fails automatically if the winner is A1, A2, A3 or A4. For A3
and A4 this follows from the statistic, which cannot be above 0 against the
better of the two. For A1 and A2 it is an operationalization of the gate: a
plain logistic regression is a baseline, so its win cannot support a claim
of a new reader.

If the gate fails, no "new reader" claim is made. The remaining options are
a paper on augmentation and read position, with the collinear-decoy
mechanism (section 9.1), or stopping. In that case stage 2 as written here
is not run; any later use of the fresh report items or of the Llama
checkpoints needs an amendment to this document, committed before that run.

If the gate passes, the pipeline is frozen: the winning arm, the grids, the
tuning procedure, the wording sets and the partitions. Stage 2 uses it
unchanged. A claim about the covariance penalty can only be made if the
winner is A6.

## 7. Stage 2

### 7.1 Data and conditions

Readers: the frozen stage 1 winner (the reader), the baselines of section
5.4 and the controls of section 5.5. Each is fitted per model, task and
draw on the 128 labelled texts of the draw (the same 10 draws as stage 1),
with the pool of section 4 encoded by that model, and scored on the 256
fresh report pairs of the task.

Combinations: 4 Qwen checkpoints by 4 tasks (16), and 2 Llama checkpoints by
4 tasks (8).

Conditions:

| Condition | Sentiment, truth, truth with negations | Utility |
|---|---|---|
| clean | 1 | 1 |
| M2 test decoy, `random` and `against` | 2 | 2 |
| T-D1, T-D2, T-D3, each `random` and `against` | 6 | 6 |
| paraphrases P1, P2, P3 | 3 | 0 |
| total | 12 | 9 |

### 7.2 Endpoint

For reader `r` and condition `c`, `A(r, c)` is the AUROC on the report
texts, averaged over the 10 draws. `W(r) = min over c of A(r, c)` is the
worst-condition AUROC. Per combination:

```text
delta = W(reader) - max over baselines b of W(b)
```

with the maximum over B1 to B9. The best baseline is chosen on the report
items themselves, which favours the baselines.

### 7.3 Bootstrap and tests

Paired bootstrap over report pairs, 10,000 resamples. A resample draws 256
pairs with replacement; both texts of a pair, all their conditions, all
readers and all draws follow the pair. `A`, `W`, the maximum over baselines
and `delta` are recomputed inside every resample, so the minima and the
maximum are inside the bootstrap.

Per combination, the one-sided p value for `delta > 0` is
`(1 + number of resamples with delta <= 0) / (1 + 10,000)`. P values are
corrected with the Holm procedure within each family (16 Qwen combinations,
8 Llama combinations), at alpha 0.05.

### 7.4 Clean non-inferiority

Per combination, on the clean condition: pair accuracy (the share of pairs
in which the label-1 text scores higher; a tie counts 0.5,
`src/tomography/reading/metrics.py:69-70`), averaged over draws, of the reader
minus that of last-token logistic regression (B3). The lower bound is the
2.5th percentile of the same paired bootstrap. Non-inferiority holds if the
lower bound is above -0.02.

### 7.5 Success rule

A combination is a win if its Holm-corrected p value is below 0.05 and clean
non-inferiority holds. Checking non-inferiority on each counted win is the
operationalization of the clean clause of the rule.

Pooled item-level effect: the mean of `delta` over the 24 combinations, with
a bootstrap in which report pairs are resampled within each task and the
same resampled pairs are used for every checkpoint of that task (10,000
resamples). Inside each resample, `W(reader)`, the maximum over baselines
of `W(b)` and `delta` are recomputed per combination, as in section 7.3,
before the mean over combinations is taken. This keeps the dependence
between combinations that share items. The pooled effect is above 0 if the lower bound of its two-sided 95%
percentile interval is above 0. It is also reported per family.

H2 holds if the wins number at least 9 of 16 in the Qwen family and at
least 5 of 8 in the Llama family, and the pooled effect is above 0. All
three are required.

### 7.6 Reported next to the rule

- Label-budget curve: the whole pipeline at 32, 64, 128 and 256 labelled
  texts (the first 16, 32, 64 and 128 pairs of each draw), for the reader
  and B1 to B9. Only 128 is confirmatory. The folds of section 5.3 stay at 4
  by pair.
- True label budget: a fitted reader uses its `n` labels and no others. The
  choice of the arm in stage 1 used 200 validation pairs per task (1,600
  labels) on Qwen3-0.6B base, once; this cost is reported with every claim
  about 128 labels. Four of the 16 Qwen combinations are on that checkpoint
  and share its fit items and pool with stage 1, though not its report
  items.
- Utility against the random direction: for every utility combination, the
  AUROC of each reader minus the AUROC of the random direction control, per
  condition, with a paired bootstrap interval. Utility is reported
  separately in this form because a random direction already reads it well
  in this repository (0.909 in the internal review), so a utility win over
  the baselines is not evidence of a concept reading by itself. The utility
  combinations still count in the families of section 7.3.
- Secondary analysis, specified in advance and not part of the success rule
  of section 7.5: the contrast A6 minus A5 (same endpoint and bootstrap),
  per combination and pooled, run only if the reader is A6. It governs one
  thing only, the wording of the claim "the penalty adds to two-position
  augmentation": the paper makes that claim only if the pooled interval of
  the contrast lies above 0; otherwise the result is credited to the two
  positions and the augmentation. H2 holds or fails without it.
- Every condition's AUROC and pair accuracy for every reader, baseline and
  control, with intervals, including the combinations that are not wins.

## 8. Threat model and validity

### 8.1 Setting

An auditor reads a binary property of a stimulus from the hidden states of
a frozen model. The prompt is not fully under the auditor's control: it may
contain a sentence that asserts a level of the concept, written by someone
else and possibly false, and its wording may differ from the wording the
reader was trained on. The auditor has 128 labelled texts, unlabelled task
texts and the ability to render its own training texts under wrappers of
its own design. It does not know the test wordings. The target is the label
of the stimulus, not what the model would say after reading the decoy.

### 8.2 Rules that protect validity

- No `against` copy is ever used to fit a reader, to tune it or to estimate
  a covariance. Its level is the opposite of the label
  (`src/tomography/experiments/p2_positions.py:185-186`), so training on it
  would hand the label to the reader. `against` appears only in the stage 1
  endpoint and in the stage 2 conditions.
- Pool items are disjoint from fit, validation and report items (section 4).
- The test wordings of sections 3.3 and 3.6 are never rendered before stage
  2. The M2 test decoy and the three paraphrases were seen through the M2
  results, which is why the three new decoy wordings were added.
- Report items are fresh (section 2.4), and the Llama checkpoints are
  untouched until stage 2.
- Worst-condition minima and the maximum over baselines are computed inside
  every bootstrap resample (sections 6.4 and 7.3), because a minimum of
  noisy estimates is biased.
- Baselines get the same draws, the same tuning budget and, for B8 and B9,
  the same wrapped copies as the reader.
- The label-budget curve and the true label budget are reported (section
  7.6).

### 8.3 The decoy may change the model, and the re-prompt oracle

A decoy may change the model's own internal state about the concept, not
only add a nuisance direction. A reader that then follows the decoy is not
simply wrong about the state it reads. This study cannot separate the two.
The canonical re-prompt oracle bounds their sum: every wrapped report text
is rendered again in the clean template and read, so the oracle's AUROC
under any condition equals a reader's clean AUROC. For each reader we report
the gap between its clean AUROC and its worst-condition AUROC, and the best
clean AUROC among all readers as the overall bound. The oracle is not a
competitor: it needs control of the prompt, which the setting of section 8.1
does not grant.

### 8.4 Known limits, stated before the runs

- A covariance penalty cannot remove a decoy shift that is collinear with
  the label direction; it can only shrink the part that is not. The review
  that preceded this design also found that at the last stimulus token the
  M2 decoy moves logistic regression by only -0.056 to +0.064, so most of
  the room for a penalty is at the last token.
- The 24 combinations share report items across checkpoints. The Holm
  procedure stays valid under dependence, but the win counts are not
  independent trials; the pooled effect of section 7.5 is the analysis that
  accounts for this.
- The sentiment report items are long phrases from the SST-2 train split,
  not the full sentences M2 reported on, and the truth task uses a different
  source than M2 (section 2.4). Neither is comparable with M2 numbers. Two
  position-disjoint phrases of one SST-2 source sentence can fall in
  different blocks, because the source has no sentence identifier.
- Truth with negations: the report block comes from the 217 subjects that
  the row-level split of M2 left entirely in its train split, so subjects
  with fewer statements are over-represented: 220 of the 652 candidate
  statements (34%) are about Spanish words, against 410 of 2,234 (18%) in
  the whole train split. The fit, pool and validation blocks may share
  subjects with the burned M2 validation and test items; the report block
  shares none.
- In the two Geometry of Truth tasks and in utility, statements are built
  from templates or share text within a pair, so surface cues exist. The
  text-only control measures them.
- The pool is in-distribution and near class balance (section 4).
- B6 and B7 are adaptations of the published readers (section 5.4).

## 9. Exploratory analyses and deviations

### 9.1 Secondary analysis listed in advance (descriptive, no test)

Collinear-decoy mechanism: for each model, task, layer and position, the
decoy direction is, for each fit-set decoy wording, the mean feature of the
pool texts whose random level is high minus the mean feature of the pool
texts whose random level is low (strong and weak for F-D3). Each pool text
is encoded at one random level only, so this is a difference between two
groups of texts, not within a text; it is class-free because the level is
drawn independently of the label, and it needs no rendering beyond those of
section 10. We report its cosine with the label direction of B1 and B3,
the share of the decoy shift that is collinear with that direction, and how
both relate to the drop of each reader under the decoy.

### 9.2 Exploratory analyses

Anything not listed in sections 6 to 9.1 is exploratory and is labelled
"exploratory" wherever it is reported. Already foreseen, and exploratory:

- The frozen reader and the baselines on the 59 burned report pairs of the
  M2 facts task, for continuity only.
- Per-condition differences between the reader and each baseline.
- Sensitivity to `gamma`, to the number of fit-set wrappers and to the pool
  size.
- Layer profiles of each position.
- Any analysis of a model that is not in section 2.1.

### 9.3 Deviation policy

After the commit of this file, its text above this section is not edited.
Any change to the design, the data, the wordings, the readers, the grids or
the analysis is appended here as a dated entry that states what changed,
why, and which results had been seen at that time. Additions that the text
announces (the Llama revision hashes, a fallback group key, implementation
differences of B6 and B7) are logged here too. An analysis that departs from
the text without an entry dated before the run it affects is reported as
exploratory.

Every result number and every caveat of the runs is reported, including a
failed gate, failed combinations and unavailable checks. Each stage (the
stage 0 pass, the stage 1 tournament, stage 2 on Qwen, stage 2 on Llama)
starts only after an explicit go decision recorded in the run log.

Deviations: none at the time of writing.

Entries after the commit of this file, in date order:

- 2026-11-28, addition announced in section 2.1 (no run of this study had
  started and no result had been seen): the two Llama checkpoints are pinned
  in `src/tomography/model.py`, `meta-llama/Llama-3.2-1B` at
  `4e20de362430cd3b72f300e6b0f18e50e7166e08` and
  `meta-llama/Llama-3.2-1B-Instruct` at
  `9213176726f574b556790deb65791e0c5aa438b6` (the `main` revisions given by
  the Hub API on that day), with a `llama` layout entry in the architecture
  registry. No forward pass of either checkpoint has been run.

- 2026-11-28, implementation choices where the text is silent, and two
  corrections found in code review (no run of this study had started, no
  forward pass of any checkpoint had been run and no result had been seen).
  None of them changes a choice stated above.
  - Statistics (sections 6.3, 6.4 and 7): thresholds are compared with a
    slack of 1e-12; percentile intervals use linear interpolation; a
    resample draws multinomial pair counts; one resample set per task serves
    every checkpoint of that task, in the per-combination tests and in the
    pooled effect; a tie in the name of the best baseline goes to A1, and
    among baselines to the lowest B number (the value is the same either
    way); the utility contrast is computed wherever a random-direction
    control exists; an A6 reader without its A5 counterpart is refused;
    stage 2 requires all 16 Qwen and all 8 Llama combinations; the gate
    bootstrap recomputes only the winner, A3 and A4, and uses the seed
    declared for the stage 1 run; pair accuracy is read from the diagonal
    of the comparison matrix, which agrees with `group_accuracy`.
  - Readers (sections 5.1 to 5.5): one tie order over all (layer, grid)
    candidates: the criterion, then the larger `lambda`, `gamma` and
    `alpha`, then the shallower layer; A7, A8, B1 and B2 have intercept 0
    (AUROC is unaffected); the random direction is drawn on the
    standardized features of 5.1; B7 takes the `lambda` of each layer from
    its cross-validated clean AUROC (ties to the larger), starts its greedy
    search at the layer of A1 and takes the largest double-fault rate over
    all pairs of the chosen set; the caller resolves the B5 layers from the
    relative depths; the two-position `Sigma_e` is rescaled to trace `2d`;
    standard deviations use `ddof` 0; the random-subspace `Q` is drawn once
    per seed for all folds; the fallback of A7 and A8 is counted per fit of
    `S`; the solver drops eigen-directions below 1e-12 of the largest; the
    fold permutation seed is derived per task and draw.
  - Data and prompts (sections 2, 3, 6.1 and 10): the 18 burned SST-2
    candidates are removed from the reserved-block candidates only; a
    linked group joins a block whole (extra texts of a joined group stay
    unused); negation subject keys carry the prefixes `city:` and `es:` (no
    fallback key was needed on the real data); a utility linked group is
    keyed by its first row; the date string of the Llama chat template is
    fixed to "26 Jul 2024"; stage 0 runs one forward pass per prompt and
    position (72,000 passes for 36,000 prompts); the go decision and the
    stage 1 gate are stored in `run_log.json` of the stage 1 run.
  - Corrections from code review (commit `dc2c2cbc`): B5 orients its value
    heads on the labels, as the M2 reader does (`fit_attention` without
    pairs), because orienting by pair needs equal stimulus token counts
    within a pair and raised an error on real pairs; a B5 tie goes to the
    shallower layer number, not to the earlier candidate index.
    `partitions.json` keeps the first 128 pairs of each draw, the largest
    budget of the curve of section 7.6; stage 1 fits on the first 64, as
    section 2.4 states.

## 10. Compute plan and order of runs

Hardware: one RTX 4070 GPU with 8 GB for the forward passes, weights loaded
in float32 (the registry default for Qwen is bfloat16,
`src/tomography/model.py:129`), models up to 1.7B parameters, one job at a
time. One 16-core CPU machine for fitting, tuning and the bootstraps. No
other accelerator is planned.

| Step | Where | Work |
|---|---|---|
| 1 | CPU | freeze the partitions of section 2.4 and write their item ids to the run directory |
| 2 | GPU | stage 0: Qwen3-0.6B base, 36,000 prompts, both positions, layers at relative depth 0.4-1.0, cached (about 5 GB in float32 for 17 layers of width 1,024) |
| 3 | CPU | stage 1: 8 arms by 4 tasks by 10 draws, endpoint, selection rule, gate bootstrap |
| 4 | none | gate decision, written to the run log before step 5; stop here if the gate fails |
| 5 | GPU | stage 2, Qwen: the four checkpoints one after another (0.6B base, 0.6B, 1.7B base, 1.7B); per checkpoint 13,312 prompts per task for sentiment and both truth tasks and 8,704 for utility (48,640 in total), plus the stimulus token states of five layers for B5 |
| 6 | CPU | stage 2, Qwen: fits, endpoint, bootstrap, Holm, non-inferiority |
| 7 | GPU | stage 2, Llama: base, then instruct, same prompts |
| 8 | CPU | stage 2, Llama: as step 6; then the pooled effect, the label-budget curve and the controls |

The Llama checkpoints are loaded for the first time in step 7. The analysis
code of steps 6 and 8 is the same code, frozen before step 5.
