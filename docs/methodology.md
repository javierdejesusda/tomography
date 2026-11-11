# Methodology and deviations

This document lists what this repository does differently from the Representation Engineering paper (Zou et al.,
[arXiv:2310.01405](https://arxiv.org/abs/2310.01405)), why, and how its numbers are produced. It is a reference for
readers of the paper, not a tutorial: the [README](../README.md) covers installation and the experiment ids,
and each config is in `experiments/<id>_*.py`. It also states the design of each study: the checkpoints, the stimuli
and where their labels come from, the splits, the controls and the published baselines. The results are in the
[findings](findings.md). Module paths are relative to `src/tomography/`.

Contents:

1. [Scope and models](#scope-and-models)
2. [Paper mapping](#paper-mapping)
3. [Controls and baselines](#controls-and-baselines)
4. [Labels and data](#labels-and-data)
5. [Pipeline choices and deviations](#pipeline-choices-and-deviations)
6. [Statistics](#statistics)
7. [Reader comparison (M2)](#reader-comparison-m2)
8. [Full-matrix readers (M3)](#full-matrix-readers-m3)
9. [Published finder baselines (B2)](#published-finder-baselines-b2)
10. [LAT against few-shot prompting (R8)](#lat-against-few-shot-prompting-r8)
11. [Known versus unknown questions (P9)](#known-versus-unknown-questions-p9)
12. [Approximate answers (P10)](#approximate-answers-p10)
13. [Run settings of the GPU runs](#run-settings-of-the-gpu-runs)
14. [Data sources](#data-sources)

## Scope and models

Ten checkpoints in three roles. The roles are reported apart and never pooled in a count.

| Key               | Role               | Parameters | Blocks (L) | Hidden size | Prompt format                    | Default dtype |
| ----------------- | ------------------ | ---------- | ---------- | ----------- | -------------------------------- | ------------- |
| `gpt2`            | GPT-2              | 124M       | 12         | 768         | plain tags                       | float32       |
| `gpt2-medium`     | GPT-2              | 355M       | 24         | 1024        | plain tags                       | float32       |
| `gpt2-large`      | GPT-2              | 774M       | 36         | 1280        | plain tags                       | float32       |
| `gpt2-xl`         | GPT-2              | 1.56B      | 48         | 1600        | plain tags                       | float16       |
| `qwen3-0.6b-base` | Qwen3 base         | 596M       | 28         | 1024        | plain tags                       | bfloat16      |
| `qwen3-0.6b`      | Qwen3 post-trained | 596M       | 28         | 1024        | chat template, thinking disabled | bfloat16      |
| `qwen3-1.7b-base` | Qwen3 base         | 1.72B      | 28         | 2048        | plain tags                       | bfloat16      |
| `qwen3-1.7b`      | Qwen3 post-trained | 1.72B      | 28         | 2048        | chat template, thinking disabled | bfloat16      |
| `qwen3-4b-base`   | Qwen3 base         | 4.02B      | 36         | 2560        | plain tags                       | bfloat16      |
| `qwen3-4b`        | Qwen3 post-trained | 4.02B      | 36         | 2560        | chat template, thinking disabled | bfloat16      |

The registry is `MODELS` in `model.py`; every entry pins a full commit SHA. The parameter counts, blocks and hidden
sizes are those of the checkpoint table of the project site (`site/params.py`, from the stored model configurations,
embeddings included).

Not every experiment has a committed run on every checkpoint. The `results/` tree holds:

| Checkpoints                                           | Experiments                                                                                                          |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| all 10                                                | `r1` to `r7`, `p1` to `p5`, `p7`, `c1`, `c2`, `c4`, `c5`, `m1`, `lr1`, `a1`, `a2`, `p9a` to `p9c`, `p10a` to `p10c` |
| 9 (all but `gpt2`)                                    | `p9d`, which needs the known versus unknown contrast that `gpt2` lacks (see [Guess control](#guess-control))         |
| 8 (the GPT-2 sizes and the Qwen3 0.6B and 1.7B pairs) | `b1`, `b2`, `v1`                                                                                                     |
| 4 (the Qwen3 0.6B and 1.7B pairs)                     | `r8`, `m2`, `m3`                                                                                                     |
| one run that aggregates finished runs                 | `p6` (the four GPT-2 sizes), `p8` (the Qwen3 0.6B and 1.7B pairs)                                                    |
| no model                                              | `tb1` (one run)                                                                                                      |
| no committed run                                      | `a5`                                                                                                                 |

- **Why a second family.** The paper steers LLaMA-2-Chat. Its 7B and 13B checkpoints do not fit an 8 GB GPU in
  bfloat16, so the second family is Qwen3 at 0.6B, 1.7B and 4B, each as a base and a post-trained checkpoint. The
  0.6B and 1.7B pairs fit the 8 GB laptop GPU; every committed run of the 4B pair was made on a rented GPU with
  more memory (the `device` field of its `meta.json`). The post-trained Qwen3 checkpoints are distilled models
  with a thinking mode; they are used with thinking
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
| `tb1`          | Word and character n-gram classifiers on the readers' own pairs, no model | Extension (the text-only control of `r1`, `r2`, `r3`, `r5`, `r7` and `p3`)        |
| `m1`           | Six direction finders: correlation, manipulation, termination          | Sec. 5.1.1, Fig. 12                                                                  |
| `m2`           | Readers compared as scorer by readout by layer rule on the same items  | Extension; see [M2](#reader-comparison-m2)                                           |
| `m3`           | Readers trained through every cell (token and layer) of the prompt     | Extension; see [M3](#full-matrix-readers-m3)                                         |
| `b2`           | Published direction finders beside LAT on truth and sentiment          | Extension; see [B2](#published-finder-baselines-b2)                                  |
| `c1`           | Sentiment steering: sweep, operators, side effects                     | Extension (uses the Sec. 3.2.1 operators and adds a prompting arm)                   |
| `c2`           | Emotion steering, steered emotion against measured emotion             | Sec. 6.1.2                                                                           |
| `c4`           | Honesty control on TruthfulQA MC1                                      | Sec. 4.3.3, Table 2; adds a prompting arm                                            |
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
| `p9a` to `p9d` | Known against unknown questions: labels, reading, reliability, steering | Extension; see [P9](#known-versus-unknown-questions-p9)                             |
| `b1`           | Published error detectors beside the `p9c` signals on the same test items | Extension; see [Published detection baselines](#published-detection-baselines)    |
| `v1`           | The hidden states behind the `p9b` readers: 2D views and their controls | Extension; see [Activation views](#activation-views-of-the-readers-v1)              |
| `p10a` to `p10c` | Range against exact quantities: answers, reading, links to P9 and to hedging | Extension; see [P10](#approximate-answers-p10)                              |
| `a1`           | Direction geometry across layers, seeds and concepts                   | Extension                                                                            |
| `a2`           | Logit lens of the reading directions                                   | Extension                                                                            |
| `a5`           | Reading directions against Qwen-Scope SAE decoders                     | Extension                                                                            |

The PCA explained variance is part of `r2` and the t-SNE maps are part of `r4`; they are not separate experiments.

## Controls and baselines

The two words mean different things here, as on the project site:

- A **control** uses no direction fitted for the concept: a random direction, labels that were shuffled, a classifier
  that sees only the text, or the popularity of the subject. It shows what the same procedure gives without a
  learned signal.
- A **baseline** is a published method run on the same items as the method under study.

A reading is compared with the random direction of its own run, not with 0.5: the random direction is oriented on
the training labels like the reader, so it can score well above 0.5.

Some names in the code and in the result files are older than this vocabulary and are kept, so that committed runs
stay readable: `SURFACE_BASELINES` and `WITHIN_BASELINES` in `p9/reading.py`, `BASELINES` and `baselines.csv` in
`p10b`, the `text_baselines` entry of the `r6` summary and the module name `tb1_text_baselines.py` all name
controls. The code also calls the prompting arm of `c1` and `c4` (an instruction in the prompt) a baseline.

### Controls

| Control                         | What it is                                                                                                                                                                                                                                    | Where                                                                                             |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Random direction, reading       | A Gaussian direction drawn from a seed, oriented on the training labels and scored like the reader (`random` series or finder)                                                                                                                | every reading sweep, `r6`, `r8`, `m1`, `m2`, `m3`, `b2`, `p9b`, `p10b`, `v1`                      |
| Shuffled labels                 | The reader scored against evaluation labels shuffled within each group (`shuffled` series). `m2` and `m3` refit every scorer on training labels shuffled within their pairs; `v1` refits on labels permuted inside each stratum                | every reading sweep, `r8`, `m2`, `m3`, `b2`, `v1`                                                 |
| Text-only classifier            | Logistic regression (`C` 10, class-balanced) on TF-IDF features of the raw text, fitted on the training texts only: word 1-grams and 2-grams seen in at least 2 texts, or character 2-grams to 5-grams inside words seen in at least 3 texts | `tb1` (for `r1`, `r2`, `r3`, `r5`, `r7`, `p3`), `r6`, `m2`, `m3`, `b2`, `p9b`, `p10b`, `v1`       |
| Popularity                      | The popularity of the subject: the popularity score of PopQA (`s_pop`) in P9, `log10(sitelinks + 1)` in `p10b`                                                                                                                                | `p9b`, `p9c`, `p10b`, `v1`                                                                        |
| Other model-free scores         | The accuracy of the relation, the length of the question and the likelihood of the subject's name (`p9c`); the length of the question in characters (`p10b`)                                                                                  | `p9c`, `p10b`                                                                                     |
| Random direction, steering      | `n_random` (3) random unit directions steered through the same controller, so with the same relative coefficients and norm; `m1` steers its `random` finder                                                                                  | `c2`, `c4`, `c5`, `p7`, `p9d`, `m1`                                                               |
| Random band                     | The 2.5th to 97.5th percentiles of the cosine with, or of the AUROC of, `n_random` (200) random unit directions                                                                                                                               | `p10c`                                                                                            |
| Comparison items                | A contrast inside one dataset, so that one question format stands on both sides (`triviaqa_known`); people with one exact height, who are living things with an exact answer                                                                  | `p9b`; `p10a`, `p10b`                                                                             |

`c1` has no random-direction arm. Three committed runs each of `c2`, `c4`, `c5` and `p7` were made before the random
arms existed (their `config.json` has no `n_random`). `p10a` reads no direction, so the people are its only control,
and `lr1` has none of the controls above.

### Published baselines

| Experiments                 | Method under study                              | Published baselines on the same items                                                                                                                                       |
| --------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `b2`                        | LAT (`pca_diff`)                                | Mass-mean and whitened mass-mean probes (Marks and Tegmark 2024), logistic-regression probe (Alain and Bengio 2016), CCS (Burns et al. 2023)                              |
| `p4`, `m1`, `p9b`, `p10b`   | LAT (`pca_diff`)                                | The `logistic` and `mean_diff` finders (the logistic-regression and mass-mean probes) in the same sweep                                                                    |
| `r6`                        | LAT readers                                     | The `standard` and `heuristic` scorers of the paper                                                                                                                         |
| `r8`                        | LAT                                             | Few-shot prompting in the lm-evaluation-harness format                                                                                                                      |
| `m2`, `m3`                  | Every other scorer, readout and reader          | `logistic/last` and `mean_diff/last`; `m3` adds `logistic/concept` and `logistic/stimulus_token`                                                                           |
| `p9c`                       | The reading direction (`lat`) and the `probe`   | The confidence of the model: first-token probability, negative entropy, mean token log-probability, the share of samples equal to the greedy answer, P(True) (Kadavath et al. 2022) |
| `b1`                        | The `p9c` signals                               | Exact-answer-token probe (Orgad et al. 2025), semantic entropy (Farquhar et al. 2024), EigenScore (Chen et al. 2024)                                                       |
| `c4`                        | The reading vector                              | The prompt difference of the paper and the contrast vector; the LoRRA adapter of an `lr1` run when `lorra_run` is set                                                      |

The readers of `r1` to `r7`, `p1`, `p2`, `p3` and `p5` are themselves logistic-regression probes (see the default
finder in [Activations and directions](#activations-and-directions)). The other steering experiments (`c1`, `c2`,
`c5`, `p7`, `p9d`) and P10a and P10c are compared with controls only.

## Labels and data

### Splits

- `assign_splits` (`datasets/records.py`) hashes the dataset name, a key and a seed into train, validation and test
  at 0.6, 0.2 and 0.2. The key is an example id or a grouping key (the subject of a PopQA question, a source
  statement), so everything built from one source stays in one split, whatever the row order.
- ETHICS keeps the test file of the source as its test split and hashes the source train file into train and
  validation at 0.75 and 0.25, with linked scenarios kept together (`datasets/ethics.py`).
- SST-2 has no labeled test split, so no hash is used: the readers train on the phrases of the source train split
  and are scored on the full sentences of its validation split. `r1` scores 256 validation pairs and reports the best
  layer of that same curve. `p1` to `p4`, `m2`, `m3` and `b2` halve the validation sentences with the seeded draw
  of `task_splits` (`experiments/variants.py`): one half selects, the other reports.
- A reading run fits `n_seeds` readers, each on its own draw of `n_pairs` (64) training pairs. The split that picks
  the layer of each study is given in the [reading conventions](findings.md#reading-conventions) of the findings.

### Where the labels come from

| Study                      | Stimuli                                                                                                           | Label and its source                                                                                                      | Evaluation items                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `r1` sentiment             | SST-2 phrases (train) and sentences (validation) of at least 3 words                                              | The binary sentiment label of SST-2                                                                                       | 256 validation pairs                                                       |
| `r2` utility               | ETHICS utilitarianism rows, two scenarios per row                                                                 | The order of the source row: its first scenario is the more pleasant one (label 1)                                        | 3,473 validation and 4,808 test pairs                                      |
| `r3` morality              | ETHICS commonsense scenarios, the short ones only                                                                 | The source label, flipped so that 1 means acceptable                                                                      | 644 validation and 983 test pairs                                          |
| `r4` emotions              | The emotion scenarios of the reference repository, one emotion against the other five                             | The emotion each scenario is filed under                                                                                  | 39 to 48 validation and 36 to 53 test pairs per emotion                    |
| `r5` honesty               | True statements of the reference facts file under an honest and an untruthful persona; plain statements to test   | Training: the persona, which the template sets. Transfer test: the true or false label of the facts file                  | Transfer: 53 validation and 59 test pairs                                  |
| `r7` truth                 | The facts file (597 statements once repeats are collapsed); nine Geometry of Truth sets scored with no refit      | The true or false label of each source file                                                                               | 53 validation and 59 test pairs; 177 to 2,225 pairs per out-of-distribution set |
| `r6` TruthfulQA MC1        | 817 questions, choices shuffled with a seed                                                                       | The correct choice of `mc1_targets`                                                                                       | 511 train, 133 validation and 173 test questions                           |
| `r8` QA tasks              | The tasks, splits and shot counts of `qa_tasks.py`                                                                | The answer key of each dataset                                                                                            | see [R8](#lat-against-few-shot-prompting-r8)                               |
| `b2`                       | Geometry of Truth statements paired with their negations; SST-2 as in `r1`                                        | The source labels; a pair must hold exactly one true member                                                               | 77 to 406 validation and 78 to 382 test pairs per dataset                  |
| P9 (`p9a` to `p9d`, `b1`, `v1`) | A pool of nine question groups, 22,028 questions with the GPT-2 tokenizer and 21,988 with Qwen3's            | From the behavior of each model: its greedy and sampled answers against the gold aliases (see [Labels](#labels))          | per model, see the findings                                                |
| P10 (`p10a` to `p10c`)     | 720 Wikidata quantity questions in three families, and 120 people                                                 | The kind of answer, range or exact, from the bounds of the Wikidata statement (see [P10](#approximate-answers-p10))       | 72 range and 75 exact test questions, and the people                       |
| `c1`, `m1` steering        | Text generated from short prompts                                                                                 | A pinned sentiment classifier, with a second one as a check                                                               | the completions of each arm and coefficient                                |
| `c2` steering              | Text generated from short prompts                                                                                 | A pinned emotion classifier, with a second one as a check                                                                 | the completions of each arm and coefficient                                |
| `c4`                       | TruthfulQA MC1 questions; each arm scores every choice                                                            | The correct choice; each vector arm picks its coefficient on the train and validation questions                           | the 173 test questions                                                     |
| `lr1`                      | Alpaca instructions and responses                                                                                 | None: the training targets are activations of the frozen model                                                            | 64 held-out pairs                                                          |

The two tokenizers give different P9 pools because a fictional subject is generated only for a PopQA subject of at
least two tokens, counted with the tokenizer of the model (7,493 fictional questions for GPT-2, 7,453 for Qwen3;
the other eight groups are the same).

### Automatic labels and what checks them

- **Answers.** In P9 and in the lie gallery of `c5`, an answer is correct when it contains a gold alias as a
  whole-word sequence after normalization (`is_correct` in `knowledge.py`). In P10a a completion is parsed into a
  point, a range, an abstention or nothing, and graded against the sourced interval widened by 5 percent
  (`p10/answers.py`). In `p9d` an abstention is a phrase match.
- **Judges.** Generated text is scored by classifiers pinned by commit SHA in `judges.py`: for sentiment
  `siebert/sentiment-roberta-large-english`, with `distilbert/distilbert-base-uncased-finetuned-sst-2-english` as a
  check; for emotion `j-hartmann/emotion-english-distilroberta-base`, with
  `cardiffnlp/twitter-roberta-base-emotion-latest` as a check. The entailment model of `b1` is
  `microsoft/deberta-large-mnli`.
- **Fluency rule.** A steered text is counted as fluent when the perplexity of neutral text under the steered
  model is at most 1.5 times that of the unsteered model and the distinct-1 of the completions is at least 0.15.
  The rule is applied by the site (`FLUENT_MAX_RATIO` and `FLUENT_MIN_DISTINCT` in `site/stats.py`), not by the
  experiment code.
- **No human labels.** `p9a`, `p9d` and `p10a` write hand-check files with empty columns for a human rater
  (`human_label`; `human_form`, `human_correct` and `human_unit_mismatch`). No committed file has a filled one, so
  the error rates of the alias match, of the abstention patterns and of the answer parser are not measured.
  Generated text is scored by the judges only.

### What is committed

`tomography export` drops the readers, the sweeps, the per-item files, the trials, the transfer arrays, the signals
and the per-item scores (`_EXCLUDED_NAMES` in `export.py`), so the committed `results/` tree holds configs, metadata,
summaries, tables and curves. The experiments that read a dropped file of an earlier run (`p10c`, `v1`) run only
where the full run directories exist.

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
- The reading experiments train `n_seeds` readers on independent training sets: 10 by default (5 for `p9b`, `p9c`
  and `p10b`), 15 for `r6`, and 20 resamples per size for `p3`.

### Series and controls

Every reading sweep (`experiments/sweep.py`) writes three series per layer:

- `reader`: the fitted direction.
- `random`: a random unit direction with the same orientation rule. This is the random-direction control.
- `shuffled`: the reader scored against evaluation labels shuffled within each group. This is the permutation null.

The controls of the other experiments, the steering ones included, are listed in
[Controls and baselines](#controls-and-baselines).

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
| `m3`                                                           | as `m2`                                                        | as `m2`                                                                   |
| `r8` task accuracies                                           | percentile bootstrap; Wilson interval for a single run         | the seeds; the test questions of a zero-shot or single-seed run           |
| `tb1` accuracies                                               | Wilson interval                                                | the test pairs; `seed_min` and `seed_max` give the spread over training sets |
| `b2` accuracies                                                | Wilson interval (`wilson_lo`, `wilson_hi`); percentile bootstrap (`boot_lo`, `boot_hi`) | the test pairs; the seeds                        |
| `p10a` rates of the greedy answers                             | Wilson interval                                                | the questions                                                             |
| `p10a` rates of the sampled answers, and the gaps              | percentile bootstrap, 2,000 resamples                          | the questions, each with all its samples; within each kind for a gap      |
| `p10b` transfer, control and people AUROC                      | percentile bootstrap, 1,000 resamples                          | the test questions                                                        |
| `p10b` curves                                                  | percentile bootstrap                                           | the 5 draws of training pairs                                             |
| `p10c` AUROC and rank correlations                             | percentile bootstrap, 1,000 resamples                          | the questions                                                             |

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

## Full-matrix readers (M3)

`m3` (`experiments/m3_matrix.py`) asks whether a reader trained through an aggregation over the cells of the prompt,
a cell being one token at one layer, reads better than M2's one-row readouts. It runs M2's protocol on M2's items
(its config shares M2's defaults). Stage 1 ran on the four checkpoints of the Qwen3 0.6B and 1.7B pairs. Stage 2 ran
on the `stimulus`, `final` and `upper` cells, one task per run on the 1.7B pair and on a CPU for the 0.6B pair.
A check on `qwen3-1.7b-base` fits `cellattn_prompt` on all 28 layers for truth, once with M2's `C` grid and once
with the grid widened to 0.001 to 1000, and stage 1 was repeated on the four checkpoints with `cellattn_prompt`
alone and that wider grid. With the same cells and grid, the flat reader ran on the four checkpoints and the MLP
head on the 0.6B pair. The results are in the [findings](findings.md#full-matrix-readers-m3).

- Cells and segments: `prompt_cells` reads every token but BOS (whose state is the same for every text) at
  `cell_layers` (default 4, 8, ..., 28, the `token_layers` of the M2 runs) and gives each token a segment: `prefix`
  (every token before the stimulus outside the decoy line: the chat tag, the instruction and a tag such as
  utility's `Scenario:`), `decoy` (P2's misleading line, matched as a whole line against `p2_positions.DECOY_LINE`,
  in the misleading conditions only), `stimulus` (the stimulus slot, M2's `response` rows) and `final` (from the
  stimulus end to the text end, where `last` and `concept` read). A token joins the segment its characters overlap,
  the stimulus winning ties.
- Cell sets: `prompt` (the `decoy`, `stimulus` and `final` segments at `cell_layers`), `stimulus`, `final`, and
  `upper` (the same segments at `upper_layers`, default 16, 20, 24, 28). No set reads the `prefix` (see the
  caveats). The default `cell_sets` is `prompt` (stage 1). Stage 2 (`stimulus`,
  `final`, `upper`, selected with `--set cell_sets=...`) runs only if a `cellattn_prompt`, `cellmax_prompt` or
  `prompt_max` row beats `logistic/last` (Holm-adjusted paired interval above zero) in some checkpoint, task and
  condition of stage 1, a gate declared before stage 1 results.
- Cell readers: `logistic/cellattn_<set>/joint` is M2's joint attention reader (`fit_attention(joint=True)`) on the
  set's cells: a value head and a query per layer and one softmax over the cells of all layers.
  `logistic/cellmax_<set>/joint` (`fit_cell_max`) has a linear head and bias per layer, and a text scores its
  highest cell over all layers. Both are trained through their pooling with the logistic loss, `C` from M2's
  `logistic_c` grid and cells standardized on the draw's training cells only. L-BFGS runs at gradient tolerance
  `cell_tol` (default 1e-10; the probes' default stays 1e-4): at 1e-4 the gradient at the zero start can already
  pass the test, which stopped a cellattn fit at its first step with zero heads. A cellmax fit that stalls at its zero
  start runs again, once, from the mean pooling solution (see the caveats). They keep the scorer name `logistic` so
  M2's comparisons take them.
- MLP head (`--set cell_head=mlp`): each layer's linear head becomes `g(z) = w2 . relu(W1 z + b1) + b` with
  `cell_hidden` units (default 8), read as `logistic/cellattnmlp_<set>/joint` (`fit_attention_mlp`) and
  `logistic/cellmaxmlp_<set>/joint` (`fit_cell_max_mlp`). cellattn's query stays linear and its bias is one for all
  layers; cellmax keeps a bias per layer. The penalty is scikit-learn's, `1/(2Cn)` times the squared norms of `W1`,
  `w2` and the queries, biases unpenalized, with `C` chosen on the same grid and the same loop (training-only
  standardization, float64 L-BFGS-B at `cell_tol`, the stall test against log 2, and the cellmax restart from the mean
  pooling solution, embedded as two units of opposite sign whose hidden and output weights share one size, the least
  penalty for that function). `W1` starts at a standard normal draw over the square
  root of the width, seeded from the run seed, task, readout and draw, so the shuffled series and every `C` of a draw
  start alike; every other parameter starts at zero, and no global random state is read. The loss is not convex: a
  small `C` makes zero a local minimum, where a fit counts as stalled, and the ReLU stops most fits on L-BFGS's
  progress test, not the gradient one. An MLP head has no direction to export (`rep_reader` refuses):
  `cellheads_<task>` keeps `hidden`, `hidden_biases`, `outputs` and `biases` (and cellattn's queries and offsets). A
  linear run leaves `cell_head` and `cell_hidden` out of `config.json`, so its hash is unchanged.
- Flat reader (pooling `flat`, not a default, rows `logistic/flat_<set>/joint`): it standardizes each layer's cells
  on the draw's training cells as the cell readers do, then concatenates them at fixed slots: each segment (decoy
  line, stimulus, final line) gets as many slots as its longest training run, with tokens aligned from the segment's
  end, so a shorter segment leaves its first slots at zero (the training mean) and a longer evaluation segment loses
  its first tokens; `truncation.json` reports the dropped cells per condition. Training uses the paper template, so
  the decoy line has no slots and gets no weight. The reader is an L2 logistic regression with the cell readers' loss
  (mean BCE plus `||w||^2/(2Cn)`, bias unpenalized, so `C` is scikit-learn's), fit in float64 by L-BFGS-B at
  `cell_tol` in dual form through the per-layer Gram matrix; it matches scikit-learn's primal fit to about 1e-9
  relative. `C` comes from the same grid by the first-best select rule, and the reader shares M3's draws, items,
  conditions, shuffled series and baseline comparisons under `best` and `top_k`. It has no MLP head (`cell_head=mlp`
  with pooling `flat` is refused), and `cellheads_<task>` keeps only its `c`, `bias` and `widths`.
- One-row readers: M2's `logistic` and `mean_diff` on `last`, `concept`, `mean` and `max` (rules `best` and
  `top_k`), fitted by M2's helpers on M2's items, and two relabelled readouts, each on its own pass:
  `stimulus_token` (the last stimulus token, M2's `concept` position over the stimulus span) and `prompt_max` (M2's
  `max` position over the whole text). A relabelled readout takes the seeds of the M2 readout whose position it
  uses.
- Choices: every choice uses the select items of the original prompts. A cell reader keeps, per draw, the first
  best grid value on the select pair accuracy, at one fixed set of layers (no layer is chosen), and its threshold
  there. Conditions and report items are M2's.
- Extraction: each draw's 128 training texts are read once and their cells freed after the draw's fits; the select
  items, then each condition's report items, are streamed batch by batch through every fitted reader, so only
  scores, segment shares and cell counts are held. The one-row activations are released before the cell phase.
- Controls: the shuffled series on every cell reader (training and select labels shuffled within their pairs with
  M2's seeds, so the grid value and threshold are chosen on shuffled labels), the random direction at every one-row
  readout, the text-only classifier, and `random/cellmax_<set>/joint`: a standard Gaussian head per layer on the
  training-standardized cells, not fitted, with one sign for all heads set by the training AUROC on the real labels.
- Statistics: M2's, unchanged (seed means with intervals, the seed ensemble, paired bootstraps, DeLong, McNemar,
  Holm). The baselines are `logistic/last`, `mean_diff/last`, `logistic/concept` and `logistic/stimulus_token`. A
  joint cell is compared with each baseline's `top_k` rule, as in M2, and with its `best` rule in a Holm family of
  its own; those rows keep the rule `joint`. M2's comparison code is called, not changed.
- Diagnostics (`diagnostics.json`): per task, condition and cell reader, the share of a report text's reader weight
  on each segment (cellmax: its highest cell, ties split, so the share of texts whose highest cell lies there;
  cellattn: its attention mass), averaged over texts and draws; `cells`, a text's mean number of cells in the set;
  and `cells_rho`, the Spearman correlation between a text's cell count and its score within each label, averaged
  over the labels and draws where it is defined.
- Outputs: M2's files (`table.json`, `comparisons.json`, `scores_<task>`, `auroc_<task>`, `summary.json`),
  `diagnostics.json`, and `cellheads_<task>` (each draw's kept heads, biases and signs, the queries and offsets of
  cellattn, and the chosen `C`), named apart from the `readers*` files the aggregation code parses. `summary.json`
  also counts, per task and fitted cell reader, its fits, their seconds, their `ConvergenceWarning`s, its
  `stalled_fits` and its `restarted_fits`. A run stalls when its loss never drops below its value at the zero start
  (log 2) by more than the relative 64 machine epsilons that L-BFGS-B's `ftol` treats as no progress.
  `restarted_fits` counts the cellmax fits whose zero start stalled and whose second run did not (each warns as a
  `RestartedFitWarning` and keeps the second run); `stalled_fits` counts the fits still stalled at the end (cellattn
  has no second run). A stalled fit warns as a `StalledFitWarning` and is kept, so selection still sees it: its
  heads stay near zero, so its scores spread on a tiny scale, seldom tie and carry no fitted signal.
- Runs load the model in float32.

Caveats:

- Length bias of the max: a text's highest cell rises with its number of cells. The misleading conditions add a
  decoy line, so more cells, and a cellmax gain under `against` is read only beside its `cells` and `cells_rho`.
- Prefix cells (2026-11-03): no cell set reads them, so the diagnostics' `prefix` share is 0. They never see the
  stimulus, and under the causal mask the prefix is the same in every text of a template, so its cells are equal
  across texts. In the first stage 1 run (qwen3-0.6b-base) one of them, a massive activation at layer 28, topped
  every stimulus and held utility cellmax fits at log 2 with head norms of 1e-17 to 3e-13. On misleading utility the
  `Scenario:` tag follows the decoy line, so its cells differ between decoys (they see the decoy, never the
  stimulus); it is dropped with the rest of the prefix.
- Zero-start stall of cellmax (2026-11-05): at the zero start every cell ties, and the gradient, split evenly among
  tied cells, is the mean pooling one. In the stage 1 run on qwen3-0.6b (2026-11-04), along that direction the
  highest cell of 118 to 128 of the 128 training texts was a newline of the chat template after the stimulus, at
  layer 28, about equal in every text, and the max pooled loss rose (slope +2e-7 to +1.3e-6, where the split
  gradient promised -1e-5 to -4e-5). L-BFGS's line search shrank the step to rounding size and stopped at log 2,
  for every `C`, on whole draws: 12 and 16 of the 40 sentiment fits (reader and shuffled), 8 and 8 of utility's.
  Those readers kept heads of norm 1e-21 to 1e-14 and scored at chance (report pair accuracy 0.46 to 0.52), although
  zero was no minimum: other directions lower the loss. Such a fit now runs again from the mean pooling solution,
  the penalized logistic fit of each text's mean cell value over all layers, split into one term per layer (heads
  and biases over the number of layers) so that a cell starts on the scale of the mean. On the CPU, on the five
  stalled reader draws, every refit left log 2 for every `C`, and the kept readers reached select pair accuracies of
  0.75 to 0.85 on sentiment (0.79 to 0.86 on its draws that never stalled) and 0.59 and 0.74 on utility (0.55 to
  0.82), with report pair accuracies on the original condition of 0.62 to 0.85; the undivided solution and a tiny
  step along the split gradient left some draws at chance. A fit that does not stall runs once from zero,
  bit-identical to before. cellattn's loss is smooth at the zero start, so its first step always descends.
- Reproducibility: cellmax fits are bit-identical on one machine, but L-BFGS on the piecewise max objective turns
  input differences near 1e-6 into score differences near 3e-3, so runs on different machines agree only to about
  1e-3.
- A layer whose cells never win keeps a near-zero cellmax head, so its exported direction is mostly noise; a later
  use of the heads weights or drops them by norm.

## Published finder baselines (B2)

`b2` (`experiments/b2_finder_baselines.py`) puts published direction finders beside LAT (`pca_diff`, the method
under study) on truth and sentiment: the same training items, layers, layer rule and test pairs for every series. Its
results on the eight laptop checkpoints are in the findings.

- Baselines: the mass-mean probe (`mean_diff`) and the whitened (IID) mass-mean probe (`whitened_mean_diff`) of
  Marks and Tegmark (2024, `marks2024geometry`), the logistic-regression probe (`logistic`, scikit-learn `C` 1.0;
  Alain and Bengio 2016, `alain2016understanding`) and CCS (`reading/ccs.py`; Burns et al. 2023,
  `burns2023discovering`). Their departures from the published recipes are in the CCS and whitened mass-mean rows
  of the deviations table above. When the logistic solver stops at its all-zero start because the gradient there is
  under its default tolerance, the fit is repeated with a tolerance of 1e-10 (`refit_at_zero`, as in P4) instead of
  stopping the run; it changes no other finder and no fit that ends away from zero.
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

### Same-dataset control and surface controls

Contrasts built from different datasets differ in question format, so a reader can separate them by wording alone.
`triviaqa_known` sets TriviaQA questions labeled known against TriviaQA questions labeled unknown, so one question
format stands on both sides. Every `p9b` reader AUROC is reported next to three surface controls, which use no
model (the code names them `SURFACE_BASELINES`, in `p9/reading.py`):

- `text_word` and `text_char`: classifiers on the word and the character n-grams of the question, fitted on the
  train items of the contrast and scored on the exact test items of the reader.
- `popularity`: the AUROC of the subject's popularity on those items. It is null unless every test item on both
  sides has an integer popularity.

### Within-relation headline

Relations differ in how many of their questions are known, so the wording of a question alone can rank the two
sides through the relation. A pooled AUROC cannot tell that apart from reading the question. For the contrasts that
have PopQA questions on a side, the headline is therefore the within-relation AUROC (`grouped_auroc` in
`p9/risk.py`): the AUROC inside each relation, averaged with weights `n_pos * n_neg` over the relations that hold
both classes. It is reported next to the within-relation popularity and text controls (`WITHIN_BASELINES`). The
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
word still counts as an abstention. The runs on post-trained checkpoints write `hand_check.csv`, a sample of
completions (`n_hand_check`, 100) with an empty `human_label` column for a rater; no committed run has it filled,
so this error is not measured.

### Activation views of the readers (V1)

`v1` (`experiments/v1_knowledge_views.py`) stores what is needed to look at the `p9b` readers and not only score
them. It reads two finished runs of the same checkpoint, the `p9a` labels and the `p9b` reading, and takes the
questions, the draws and their seeds, the template and the layer from the `p9b` run. Its own seed enters only the
shuffled-label control and the text-only view. It needs the local copy of the `p9b` run, because the export drops
`readers.npz` and `transfer.npz`.

- Extraction: the model loads in float32, because the few-shot prefix shared by every prompt is run once and its
  cache reused, which matches the uncached activations only in float32. The questions are the test questions of the
  `p9b` contrasts and the train questions of its committed draws, read at the last token at every layer.
- Reproduction check: the four `p9b` methods (`logistic`, `pca_diff`, `mean_diff` and the random direction) are
  refitted on each committed draw. For every contrast and method, the mean test AUROC over the draws at the `p9b`
  layer is set against the same mean of the `p9b` run with a tolerance of 0.02 (`TOLERANCE`); a miss is recorded in
  the summary, not raised.
- Published reader: the `logistic` reader that `p9b` fitted on all the train pairs of a contrast. `v1` scores the
  test questions on it without refitting it and keeps the scores of the questions it shows.
- Controls: the random direction of `p9b` with the seeds of the committed draws; a `logistic` reader refitted on
  the train questions of each draw with their labels permuted inside each stratum; the surface controls
  (`text_word`, `text_char`, `popularity`), recomputed and compared with the `p9b` table; and a text-only view of
  the same questions.
- Views: test questions only, at most 300 on each side of a contrast (`MAX_DISPLAYED`). The principal-component
  view places them on the two leading components of their hidden states. The supervised view has the score on the
  published reader as x and, as y, the leading principal component of the hidden states once the direction of that
  reader is removed. The text-only view reduces the TF-IDF word features of `text_word` to two components by a
  truncated SVD, which is latent semantic analysis on uncentered features and not a principal-component analysis,
  so its variance ratios are not comparable with those of the hidden states.
- Outputs: `projections.npz`, `projections.json`, `pca2d.npz`, `text2d.npz` and `summary.json`. No question text
  and no hidden state is stored. A run stores AUROC values per draw and no interval.

### Published detection baselines

These baselines predict, per question, whether the greedy answer is correct. The `b1` experiment
(`experiments/b1_detection_baselines.py`) puts them beside the `p9c` signals; its results on the eight laptop
checkpoints are in the findings. Each states where it departs from its published recipe.

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
  probe; a question with no non-empty sample for semantic entropy, which would otherwise read all-blank samples as
  one confident cluster; fewer than 2 non-empty samples for EigenScore) takes the
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
  never shares the GPU with the model: about 0.07 s per pair on the laptop CPU (640 pairs timed), and 34,088 to
  45,078 pairs per checkpoint in the committed runs (`nli_pairs` of each summary, summed over the three datasets).
  The summary records the seconds each baseline took.

## Approximate answers (P10)

P10 asks how a model handles a quantity question whose sourced answer is a range, against one whose answer is one
exact value. The stages are `p10a` (the answers), `p10b` (reading the hidden states of the same questions) and
`p10c` (links of the range direction to the P9 directions and to the answers). The code is in `p10/`,
`datasets/quantities.py` and `experiments/p10*_*.py`. P10 has no counterpart in the paper and no published
baseline.

### Questions and labels

The questions come from Wikidata, one cached query per family and side, with the unit and the bounds of each
statement. The summary of every `p10a` and `p10b` run records the retrieval date and the SHA-256 digest of each
query result.

| Family   | Range side                                                                         | Exact side                                                           |
| -------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| height   | a dog breed: "How tall is a X, in centimetres?"                                    | a skyscraper, tower or tower block: "How tall is the X, in metres?"  |
| wingspan | a taxon: "What is the wingspan of the X, in centimetres?"                          | an aircraft: "What is the wingspan of the X aircraft, in metres?"    |
| duration | the gestation period of a taxon: "How long is the gestation period of the X, in days?" | a film: "How long is the film X, in minutes?"                    |

- **Label.** The label is the kind of the sourced answer and comes from the Wikidata statement, not from a model. A
  range statement has bounds whose relative width is at least 0.10 (`MIN_RANGE_RELATIVE_WIDTH`); narrower bounds
  are read as measurement precision. An exact statement has no bounds or a relative half-width of at most 0.01. An
  entity with several best-rank statements is dropped as ambiguous, except on the dog-breed side, where the
  statements are united into one range.
- **Selection.** An entity needs at least 5 sitelinks (20 in the film query, 100 for the people) and a Latin-script
  label; a taxon whose label is its scientific name is dropped. Per family, the most linked `n_per_side` (120) items
  of each side are kept, so the sides have the same size: 720 questions in all.
- **Splits.** Each family is hashed into train, validation and test at 0.6, 0.2 and 0.2 by `assign_splits`. `p10a`
  asks every question whatever its split; `p10b` and `p10c` use the splits.
- **People.** In every family the range side is a living thing and the exact side an artifact, and the wording
  differs by side. So 120 well-known people with one exact height between 140 and 230 cm are added as test-only
  items ("How tall is X, in centimetres?"): living things with an exact answer. They never train a reader.

### Behavior (`p10a`)

`p10a` generates and reads no activations. Each question is asked in a free condition and in a forced condition,
which tells the model to give one exact number. The few-shot format, shared by every model, has six worked examples
in two orders (`range_last`, and the same list reversed, `exact_last`); the chat format, with no examples, is used
for post-trained checkpoints only. Each prompt gets one greedy completion and `n_samples` (5) sampled ones at
temperature 1.0, of up to 32 new tokens.

- Parsing: a few-shot completion is cut at its first newline and a chat completion is read from its first
  non-empty line. The line is read as a point, a range, an abstention or nothing (`parse_answer` in
  `p10/answers.py`); the abstention patterns are those of P9 plus a few more.
- Grading: an answer is correct when its value lies in, or its range overlaps, the sourced interval widened by 5
  percent on each side (`rel_tol`).
- Measures: the share of answers of each form, the hedge gap (the range share on the range questions minus that on
  the exact questions), how often a forced point lands inside the sourced range, and the people set beside the dog
  breeds and the buildings.
- Intervals: a rate of greedy answers has a Wilson interval. A rate of sampled answers has a bootstrap interval
  over the questions, each resampled with all its samples (2,000 resamples), replaced by the Wilson interval on the
  number of questions when the rate is exactly 0 or 1. A gap resamples the questions within each kind.
- Sampling is seeded per generation call, so `batch_size` is part of the run, as in `p9a`.

### Reading (`p10b`)

Each question is rendered as the bare question (`plain`) and in the concept template of the paper with the word
"variability" (`concept`), and read at the last token at every layer from 1 to L.

- Train sets: each family alone, each pair of families, and all three. A reader is fitted on the train questions of
  its set, with a range question paired with an exact one of the same family, never across families.
- Finders and draws: `logistic`, `pca_diff`, `mean_diff` and `random`, each on `n_seeds` (5) draws of the pairs.
  The random direction is fitted, oriented and given its layer like the others.
- Layer: the layer with the highest mean validation AUROC over the draws, on the validation questions of the
  families of the train set, the first one on ties. No test question chooses anything.
- Metric: the AUROC of the test questions, range as the positive class, per family and pooled. A transfer row is
  the AUROC of the draw-averaged score at the chosen layer.
- Controls: the random direction; and four scores that use no model, each fitted or oriented on the same train
  questions: `text_word`, `text_char`, `length` (characters of the question) and `popularity`
  (`log10(sitelinks + 1)`). Because the layer and the sign are chosen on data, a random direction does not sit at
  0.5, and a reader is compared with it.
- People diagnostic: the AUROC of the dog breeds against the people, and of the people against the buildings, on
  the reader's scores. A reader of range against exact scores the people like the buildings; a reader of living
  against artifact scores them like the breeds.
- Intervals: the transfer, control and people rows resample the test questions (1,000 resamples). The curves
  resample the 5 draws. The two are not comparable.

### Links (`p10c`)

`p10c` runs no new training. It reads three finished runs of the same checkpoint (`p10b`, `p9b` and `p10a`), which
must be the full local run directories, because the export drops the readers and the per-item files.

- Knowledge link: the cosine, at each layer, between the stored `p10b` direction (the `plain/logistic` reader of
  draw 0, fitted on all three families) and the direction of each P9 contrast; and the AUROC of each stored P9
  reader on the P10 questions, rendered with the template of the `p9b` run.
- Behavior link: within one kind of question, the Spearman correlation between the `p10b` score of a test question
  and how often the model answers it with a range in the `p10a` run. Across kinds the link would be trivial,
  because the reader separates the kinds and the behavior differs by kind.
- Controls: a cosine or an AUROC is compared with its random band, the 2.5th to 97.5th percentiles over `n_random`
  (200) random unit directions at the same layer. An AUROC is never compared with 0.5.
- Intervals: percentile bootstrap over the questions (1,000 resamples), with no correction for multiple comparisons.

## Run settings of the GPU runs

- **Batch size is part of the run.** Sampling in `p9a` is seeded per chunk of prompts, so `batch_size` changes every
  sampled completion and the config hash. The label runs used a batch size of 8 for every model except `gpt2-large`
  (3) and `gpt2-xl` (4). `p10a` samples the same way; its runs used 8, except `qwen3-4b-base` (4).
- **Hardware.** The `device` field of each `meta.json` names the device of the run. The committed runs come from
  the 8 GB laptop GPU, from rented GPUs and, for a few runs, from a CPU; the 4B pair ran on rented GPUs only. The
  config hash covers the config and not the machine, so compare `meta.json` before pooling runs.
- **CUDA memory cap.** The runs on the 8 GB laptop GPU called `torch.cuda.set_per_process_memory_fraction(0.8)`
  before the run, so the caching allocator does not spill into shared system memory. This is memory safety only and
  does not change the outputs. The caller of the run sets it; it is not in the repository code.
- **Offline reruns.** Dataset loaders take `offline=True` and read only the cache; for model weights set
  `HF_HUB_OFFLINE=1`. See [Reproduce at scale](../README.md#reproduce-at-scale).
- **Precision.** The registry defaults: float32 for GPT-2 small, medium and large, float16 for `gpt2-xl`, bfloat16
  for Qwen3, the 4B pair included. `r8`, `m2`, `m3`, `b1` and `v1` load every model in float32 instead. Activations
  are stored as float32 arrays.

## Data sources

Every dataset, its pinned revision or digest, its license and the experiments that use it are recorded in the
`PinnedSource` rows under `datasets/`, and the README describes pinning and offline mode in
[Reproduce at scale](../README.md#reproduce-at-scale). No third-party data is redistributed.
