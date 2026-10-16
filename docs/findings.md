# Findings

This document reports the results of this repository so far. It is a reference for the paper, in the style of
[methodology.md](methodology.md) and [related-work.md](related-work.md): it states what each run measured, the number it
produced, and what that number does and does not show. Experiment ids (`r1`, `p6`, `c1` and so on) are those of the
paper mapping in the methodology, which also describes the pipeline choices, the statistics and the run settings that
are not repeated here. Most numbers are read from the files of a finished run (`summary.json`, `table.json` or the
curves files). The derived ones are the R6 text baselines on `gpt2` (see that section), the P6 tables (recomputed from
the runs), the random-direction cosine of A1, the P9d selectivity columns (recomputed from the stored per-item
slopes), the span of the sizes in decades, and differences or ranges quoted in
the text. Runs are identified by config hash in [Run identity](#run-identity). The document is not complete:
[Pending](#pending) lists what it does not yet report.

Contents:

1. [Reading conventions](#reading-conventions)
2. [LAT reading on the GPT-2 ladder (R1 to R7)](#lat-reading-on-the-gpt-2-ladder-r1-to-r7)
3. [LAT reading on Qwen3](#lat-reading-on-qwen3)
4. [TruthfulQA MC1 against text baselines (R6)](#truthfulqa-mc1-against-text-baselines-r6)
5. [Scaling across the GPT-2 sizes (P6)](#scaling-across-the-gpt-2-sizes-p6)
6. [Direction agreement and logit lens on gpt2 (A1, A2)](#direction-agreement-and-logit-lens-on-gpt2-a1-a2)
7. [Sentiment steering on the GPT-2 ladder (C1)](#sentiment-steering-on-the-gpt-2-ladder-c1)
8. [Known versus unknown questions (P9)](#known-versus-unknown-questions-p9)
9. [Run identity](#run-identity)
10. [Pending](#pending)

## Reading conventions

- **What a number is.** A reading run fits 10 readers (seeds), each on its own draw of 64 training pairs, with the
  logistic direction finder and the last-token position (R5 reads the response tokens). Accuracy is the share of
  held-out pairs in which the label-1 member scores higher; ties share the credit, so chance is 0.5. A cell is the mean
  over the 10 readers at one layer. `L<k>` is the residual stream after block `k`; `gpt2`, `gpt2-medium`, `gpt2-large`
  and `gpt2-xl` have 12, 24, 36 and 48 blocks.
- **Which layer.** The layer is the one with the highest mean accuracy of the 10 readers on a validation split, and the
  table reports the test accuracy there (R2, R3, R4, R5 and R7). R5 uses the transfer validation pairs. R1 has no
  separate test split: its curve is scored on 256 SST-2 validation pairs, and the layer is the best layer of that same
  curve, so an R1 number is optimistic by the selection. Validation sizes are 3,473 pairs (R2), 644 (R3), 53 (R5 and R7)
  and 39 to 48 per emotion (R4). Test sizes are 4,808 pairs (R2), 983 (R3), 59 (R5 transfer and R7) and 36 to 53 per
  emotion (R4).
- **Intervals.** Intervals are 95 percent percentile bootstrap intervals over the 10 readers (15 for R6), as in the
  [methodology](methodology.md#intervals-and-what-is-resampled). They describe how much a result moves with the training
  draw. They do not include the sampling of the evaluation pairs, which matters where the test set is small (59 pairs
  for R5 and R7, 36 to 53 per emotion for R4), and they do not include variation between models or runs. Each model has
  one run per experiment.
- **R5 is reported as the transfer test only.** The in-template accuracy is 1.0 on every model by construction, because
  it scores statements inside the persona template the reader was fitted on. It is not a held-out measure and is not
  reported. The transfer test is exploratory, as in the README and the config docstring of `r5`: it scores plain true
  and false statements at the mean token position (`_TRANSFER_POSITION = "mean"`) with no refit, and the sign of each
  layer is fixed on the persona pairs, so a below-chance transfer value is not read as an anti-honesty direction.
- **R7 sets.** Besides the fact test split, R7 scores the out-of-distribution subsets of Geometry of Truth with no
  refit, at the layer chosen on the fact validation split. `neg_` sets negate the statements, `larger_than` and
  `smaller_than` are numeric comparisons, `cities_conj` and `cities_disj` join two city statements.

## LAT reading on the GPT-2 ladder (R1 to R7)

| Experiment               | gpt2                 | gpt2-medium          | gpt2-large           | gpt2-xl              |
| ------------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| R1 sentiment             | 0.772 [0.758, 0.784] | 0.789 [0.771, 0.804] | 0.838 [0.823, 0.853] | 0.817 [0.808, 0.825] |
|   selected layer         | L8                   | L20                  | L35                  | L24                  |
| R2 utility test          | 0.655 [0.644, 0.667] | 0.676 [0.660, 0.691] | 0.680 [0.664, 0.694] | 0.694 [0.684, 0.705] |
|   selected layer         | L12                  | L15                  | L36                  | L33                  |
|   validation accuracy    | 0.680                | 0.708                | 0.722                | 0.736                |
| R3 morality test         | 0.623 [0.601, 0.645] | 0.682 [0.664, 0.697] | 0.727 [0.715, 0.738] | 0.768 [0.759, 0.777] |
|   selected layer         | L11                  | L20                  | L27                  | L34                  |
|   validation accuracy    | 0.623                | 0.682                | 0.724                | 0.765                |
| R5 transfer, exploratory | 0.386 [0.369, 0.403] | 0.598 [0.583, 0.612] | 0.566 [0.556, 0.576] | 0.492 [0.475, 0.510] |
|   selected layer         | L7                   | L9                   | L16                  | L1                   |
|   validation accuracy    | 0.519                | 0.474                | 0.591                | 0.508                |
| R7 truth test            | 0.656 [0.629, 0.681] | 0.573 [0.541, 0.608] | 0.620 [0.592, 0.649] | 0.614 [0.578, 0.646] |
|   selected layer         | L9                   | L16                  | L18                  | L35                  |
|   validation accuracy    | 0.640                | 0.634                | 0.643                | 0.674                |

Test accuracy with its interval; the layer and the validation accuracy at that layer are in the rows below it. R4 and R6
were run on `gpt2` only and have their own tables.

Findings:

- **R1 sentiment.** 0.772 on `gpt2` to 0.838 on `gpt2-large`. `gpt2-xl` (0.817) is not above `gpt2-large`.
- **R2 utility.** Accuracy rises with size from 0.655 to 0.694, a range of about 0.04. At the selected layer the
  validation accuracy is above the test accuracy by 0.025 to 0.042 on every size, the layer having been chosen on it.
- **R3 morality.** Accuracy rises at every step, 0.623, 0.682, 0.727, 0.768, and the intervals of neighbouring sizes do
  not overlap. Validation and test accuracy agree within 0.004.
- **R5 honesty transfer (exploratory).** The four values (0.386, 0.598, 0.566, 0.492) do not follow size. The validation
  accuracy at the selected layer is 0.519, 0.474, 0.591 and 0.508: near or below chance on three of four sizes, so the
  layer pick is noise there and these test values are not a measure of how much honesty transfers. `gpt2` is below
  chance on the test pairs; `gpt2-medium` is above chance although its layer was chosen at a validation accuracy below
  chance; `gpt2-xl` is at chance.
- **R7 truth.** Test accuracy at the validation layer is 0.656, 0.573, 0.620 and 0.614 on 59 pairs, against validation
  accuracy of 0.634 to 0.674.

R7 on the out-of-distribution sets, accuracy at the same layer with its interval:

| Set (pairs)           | gpt2                 | gpt2-medium          | gpt2-large           | gpt2-xl              |
| --------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| cities (748)          | 0.581 [0.562, 0.599] | 0.599 [0.580, 0.622] | 0.560 [0.542, 0.585] | 0.622 [0.595, 0.649] |
| neg_cities (748)      | 0.433 [0.420, 0.448] | 0.428 [0.409, 0.442] | 0.475 [0.453, 0.494] | 0.448 [0.433, 0.465] |
| sp_en_trans (177)     | 0.482 [0.467, 0.498] | 0.560 [0.544, 0.578] | 0.534 [0.516, 0.552] | 0.511 [0.494, 0.530] |
| neg_sp_en_trans (177) | 0.543 [0.531, 0.555] | 0.434 [0.417, 0.450] | 0.481 [0.453, 0.511] | 0.505 [0.488, 0.520] |
| larger_than (990)     | 0.477 [0.447, 0.513] | 0.300 [0.264, 0.339] | 0.336 [0.312, 0.363] | 0.454 [0.410, 0.499] |
| smaller_than (990)    | 0.503 [0.463, 0.538] | 0.730 [0.692, 0.764] | 0.680 [0.648, 0.710] | 0.486 [0.445, 0.531] |
| cities_conj (748)     | 0.542 [0.528, 0.555] | 0.545 [0.536, 0.556] | 0.538 [0.529, 0.549] | 0.532 [0.527, 0.538] |
| cities_disj (748)     | 0.542 [0.528, 0.556] | 0.540 [0.526, 0.558] | 0.521 [0.507, 0.534] | 0.530 [0.515, 0.544] |
| common_claim (2,225)  | 0.656 [0.651, 0.662] | 0.642 [0.634, 0.649] | 0.638 [0.630, 0.647] | 0.639 [0.632, 0.647] |

- Plain city statements are above chance on every size (0.560 to 0.622), and the negated city statements are below
  chance on every size (0.428 to 0.475, upper interval bounds below 0.5). Below chance means the reader ranks the true
  negated statement under the false one, which is consistent with a reader that follows the polarity of the sentence
  rather than its truth; this is not tested here and does not extend to every set (see the Qwen3 section).
- `larger_than` and `smaller_than` mirror each other on `gpt2-medium` (0.300 and 0.730) and `gpt2-large` (0.336 and
  0.680): one is far below chance and the other far above. This mirroring is consistent with a reader that tracks
  magnitude or direction rather than truth; no run here tests that. On `gpt2` and `gpt2-xl` both are within 0.05 of
  chance.
- The conjunction and disjunction sets sit between 0.521 and 0.545. `common_claim` (2,225 pairs) is 0.638 to 0.656.

R4 on `gpt2` (one reader per emotion against the others, same protocol):

| Emotion   | Test accuracy        | Layer | Validation accuracy | Test pairs |
| --------- | -------------------- | ----- | ------------------- | ---------- |
| happiness | 0.876 [0.847, 0.908] | L12   | 0.883               | 38         |
| sadness   | 0.930 [0.916, 0.943] | L9    | 0.979               | 44         |
| anger     | 0.976 [0.971, 0.978] | L11   | 0.929               | 45         |
| fear      | 0.968 [0.964, 0.974] | L11   | 0.971               | 53         |
| disgust   | 0.997 [0.992, 1.000] | L5    | 1.000               | 39         |
| surprise  | 0.922 [0.906, 0.936] | L8    | 0.984               | 36         |

All six emotions are read at 0.876 or above on `gpt2`, lowest for happiness and highest for disgust. Each test set has
36 to 53 pairs.

## LAT reading on Qwen3

Same protocol on the base and post-trained checkpoints of Qwen3 0.6B and 1.7B. The post-trained checkpoints use the chat
template with thinking disabled; the base checkpoints use plain tags (see
[methodology](methodology.md#scope-and-models)). The post-trained checkpoints are small distilled models used with
thinking disabled, so a base against post-trained contrast here is a contrast with that recipe and not with instruction
tuning in general. It is descriptive; the paired comparison (P8) is pending.

| Experiment               | qwen3-0.6b-base      | qwen3-0.6b           | qwen3-1.7b-base      | qwen3-1.7b           |
| ------------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| R1 sentiment             | 0.947 [0.939, 0.955] | 0.857 [0.837, 0.874] | 0.905 [0.898, 0.912] | 0.866 [0.858, 0.873] |
|   selected layer         | L19                  | L18                  | L18                  | L18                  |
| R2 utility test          | 0.744 [0.734, 0.754] | 0.732 [0.717, 0.744] | 0.768 [0.763, 0.774] | 0.756 [0.747, 0.766] |
|   selected layer         | L23                  | L28                  | L18                  | L28                  |
|   validation accuracy    | 0.791                | 0.794                | 0.830                | 0.831                |
| R3 morality test         | 0.793 [0.787, 0.800] | 0.778 [0.762, 0.789] | 0.896 [0.892, 0.900] | 0.892 [0.887, 0.897] |
|   selected layer         | L24                  | L18                  | L20                  | L18                  |
|   validation accuracy    | 0.791                | 0.791                | 0.887                | 0.887                |
| R5 transfer, exploratory | 0.456 [0.444, 0.468] | 0.529 [0.525, 0.534] | 0.778 [0.766, 0.790] | 0.781 [0.746, 0.819] |
|   selected layer         | L27                  | L21                  | L20                  | L20                  |
|   validation accuracy    | 0.651                | 0.734                | 0.830                | 0.857                |
| R7 truth test            | 0.763 [0.749, 0.776] | 0.736 [0.695, 0.776] | 0.934 [0.925, 0.942] | 0.937 [0.924, 0.949] |
|   selected layer         | L23                  | L18                  | L18                  | L18                  |
|   validation accuracy    | 0.864                | 0.738                | 0.908                | 0.898                |

Findings:

- **R1 sentiment.** 0.857 to 0.947. On both sizes the base checkpoint is above its post-trained counterpart (0.947 and
  0.857 at 0.6B, 0.905 and 0.866 at 1.7B), with non-overlapping intervals. The 0.6B base checkpoint is above the 1.7B
  base checkpoint, so size does not order R1 here.
- **R2 utility and R3 morality.** R2 is 0.732 to 0.768 and R3 is 0.778 to 0.896. At 1.7B R3 is 0.896 and 0.892, against
  0.793 and 0.778 at 0.6B. Base and post-trained checkpoints are within 0.02 of each other in both experiments.
- **R5 honesty transfer.** At 1.7B the transfer test is 0.778 (base) and 0.781 (post-trained), with validation accuracy
  0.830 and 0.857. At 0.6B it is 0.456 (base, below chance) and 0.529 (post-trained), although the validation accuracy
  at the selected layer is 0.651 and 0.734. Validation and test disagree at 0.6B, and the test set has 59 pairs.
- **R7 truth.** The test accuracy is 0.934 and 0.937 at 1.7B and 0.763 and 0.736 at 0.6B, with validation accuracy of
  0.908, 0.898, 0.864 and 0.738.

R7 on the out-of-distribution sets:

| Set (pairs)           | qwen3-0.6b-base      | qwen3-0.6b           | qwen3-1.7b-base      | qwen3-1.7b           |
| --------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| cities (748)          | 0.942 [0.932, 0.953] | 0.838 [0.811, 0.864] | 1.000 [1.000, 1.000] | 0.993 [0.992, 0.994] |
| neg_cities (748)      | 0.568 [0.558, 0.577] | 0.553 [0.523, 0.584] | 0.943 [0.939, 0.948] | 0.856 [0.828, 0.883] |
| sp_en_trans (177)     | 0.822 [0.794, 0.851] | 0.937 [0.927, 0.946] | 1.000 [1.000, 1.000] | 0.997 [0.995, 0.999] |
| neg_sp_en_trans (177) | 0.734 [0.708, 0.760] | 0.337 [0.314, 0.358] | 0.508 [0.482, 0.537] | 0.238 [0.216, 0.259] |
| larger_than (990)     | 0.783 [0.739, 0.833] | 0.787 [0.764, 0.811] | 0.997 [0.997, 0.997] | 0.982 [0.980, 0.984] |
| smaller_than (990)    | 0.902 [0.887, 0.917] | 0.868 [0.851, 0.883] | 0.978 [0.976, 0.980] | 0.868 [0.854, 0.881] |
| cities_conj (748)     | 0.779 [0.763, 0.798] | 0.666 [0.640, 0.691] | 0.960 [0.958, 0.962] | 0.943 [0.939, 0.946] |
| cities_disj (748)     | 0.640 [0.620, 0.659] | 0.605 [0.596, 0.616] | 0.852 [0.841, 0.862] | 0.780 [0.773, 0.787] |
| common_claim (2,225)  | 0.695 [0.690, 0.700] | 0.676 [0.668, 0.686] | 0.812 [0.808, 0.815] | 0.791 [0.789, 0.793] |

- At 1.7B the plain sets are at or near ceiling (`cities` 1.000 and 0.993, `sp_en_trans` 1.000 and 0.997, `larger_than`
  0.997 and 0.982), and the negated city statements stay above chance (0.943 base, 0.856 post-trained).
- `neg_sp_en_trans` is the set where values fall below chance, which is consistent with a reader that follows polarity
  there: 0.238 on `qwen3-1.7b` and 0.337 on `qwen3-0.6b` are below chance while the un-negated `sp_en_trans` is 0.997
  and 0.937. The 1.7B base checkpoint is at chance (0.508) and the 0.6B base checkpoint is above it (0.734).
- At 0.6B the negated city statements are above chance (0.568 [0.558, 0.577] and 0.553 [0.523, 0.584]) but far below the
  1.7B models (0.943 and 0.856), so negation is not handled alike across sets or sizes. Every other set is above chance
  except `neg_sp_en_trans` on the post-trained checkpoint (0.337).

## TruthfulQA MC1 against text baselines (R6)

R6 was run on `gpt2` only. The question set is split by hash into 511 train, 133 validation and 173 test questions of
817. Every row below is scored on the same 173 test questions; the chance line (the mean of one over the number of
choices) is 0.228. The LAT rows come from the run; the standard and heuristic rows come from the same run's table. The
two text-only baselines are classifiers on the answer texts alone (TF-IDF with a logistic regression fitted on the train
answers, no question and no model). The `gpt2` run was made before the text baselines were added to R6, so their rows
were computed with the code that now writes them (`text_scores`, the same estimator, and the same split and test
questions as the LAT rows), not read from the run; the rerun of R6 on the GPU is pending. They do not depend on the
model.

| Row            | Layer rule       | Trained on                       | MC1 [95% interval]   |
| -------------- | ---------------- | -------------------------------- | -------------------- |
| standard       | none             | no training                      | 0.168 [0.119, 0.230] |
| heuristic      | none             | no training                      | 0.277 [0.216, 0.348] |
| text_word      | none             | word TF-IDF on answer texts      | 0.676 [0.603, 0.742] |
| text_char      | none             | character TF-IDF on answer texts | 0.671 [0.597, 0.736] |
| LAT truthfulqa | validation-layer | TruthfulQA train questions       | 0.649 [0.640, 0.657] |
| LAT truthfulqa | best-layer, L9   | TruthfulQA train questions       | 0.652 [0.647, 0.659] |
| LAT arc        | validation-layer | ARC-Challenge train items        | 0.383 [0.353, 0.414] |
| LAT arc        | best-layer, L9   | ARC-Challenge train items        | 0.385 [0.344, 0.429] |
| LAT honesty    | validation-layer | true statements as persona pairs | 0.270 [0.264, 0.277] |
| LAT honesty    | best-layer, L12  | true statements as persona pairs | 0.277 [0.271, 0.284] |

Intervals of the LAT rows are over the 15 readers; intervals of the other four rows are Wilson intervals over the 173
test questions, so the two kinds are not comparable in width. The `validation-layer` rule lets each reader take the
layer where its own validation accuracy is highest. The `best-layer` rule uses one layer for all readers, the one with
the best mean test accuracy, so it is selected on the test questions and is optimistic.

- **The TruthfulQA-source LAT number does not exceed the text baselines.** The reader trained on TruthfulQA train
  questions reaches 0.649 at the validation layer, and a classifier that sees only the answer texts reaches 0.676 (word)
  and 0.671 (character). The LAT point estimate is inside both Wilson intervals of the text baselines, so this number
  does not show that the reader recovers more than the wording of the answers carries.
- **The other two sources.** The ARC-trained reader reaches 0.383 and the reader trained on true statements as persona
  pairs reaches 0.270, both below the text baselines. Neither source contains TruthfulQA answers.
- **Standard and heuristic.** The standard log-probability MC1 is 0.168, below the chance line, and the interval reaches
  it (upper bound 0.230). The heuristic is 0.277. On all 817 questions the run records 0.245 [0.217, 0.275] for the
  standard and 0.286 [0.256, 0.318] for the heuristic.

## Scaling across the GPT-2 sizes (P6)

P6 reads finished runs of the four GPT-2 sizes: the R1, R2, R3, R5 and R7 readings above (R5 as its transfer test at the
layer chosen on transfer validation) and the four C1 runs of the C1 section. Its accuracy table gives the same
accuracies and layers as the GPT-2 table; it recomputes the intervals with its own bootstrap, and a bound differs from
the ones above by at most 0.003, so the tables above are the reference for intervals. It adds the least-squares slope of
the accuracy on the logarithm of the parameter count, in accuracy per decade of parameters. The four sizes span
124,439,808 to 1,557,611,200 parameters, about 1.1 decades.

|                                   | R1    | R2    | R3    | R5    | R7     |
| --------------------------------- | ----- | ----- | ----- | ----- | ------ |
| Accuracy per decade of parameters | 0.052 | 0.034 | 0.132 | 0.095 | -0.027 |

- The slopes are fitted to four points, one per size, and have no interval. They summarize the tables above and are not
  a test of a scaling law. R3 has the clearest slope (0.132), consistent with its step-by-step rise. R7 is slightly
  negative.
- **The R5 slope is not a scaling trend.** R5 is 0.386 on `gpt2`, below chance, then 0.598 and 0.566, and 0.492 on
  `gpt2-xl`, at chance; a positive slope (0.095) fitted to a below-chance first point and an at-chance last point does
  not describe a trend. The layers of those runs were chosen at validation accuracies near or below chance. Read the R5
  values per model with the caveat of the reading section.
- The R1 slope (0.052) uses best-layer values and is optimistic as the R1 values are.

The steering half of P6 fits the dose slope of each C1 reader: the least-squares slope of the sentiment score on the
relative coefficient over the grid of the C1 run, for the linear operator.

| Model       | Parameters    | Reading layer | Steer layer | Mean dose slope [range over readers] | Readers |
| ----------- | ------------- | ------------- | ----------- | ------------------------------------ | ------- |
| gpt2        | 124,439,808   | L8            | L12         | 1.784 [1.748, 1.807]                 | 3       |
| gpt2-medium | 354,823,168   | L20           | L24         | 1.763 [1.567, 1.936]                 | 3       |
| gpt2-large  | 774,030,080   | L35           | L36         | 1.688 [1.546, 1.815]                 | 3       |
| gpt2-xl     | 1,557,611,200 | L24           | L48         | 1.480 [1.235, 1.966]                 | 3       |

The mean dose slope goes from 1.784 to 1.480 as size grows, but the ranges over the three readers overlap on every pair
of sizes, so the table does not show a difference in slope between sizes. The slope averages over coefficients where the
score saturates or, on `gpt2-xl`, falls back, and is a summary of the curve, not a rate.

## Direction agreement and logit lens on gpt2 (A1, A2)

Both analyses read the readers saved by the `gpt2` runs of R1, R2, R3, R5 and R7 (hashes under Run identity); neither
runs a forward pass over data.

**A1, seed agreement.** For each reading, the mean cosine between the directions of the 10 seeds, at layer 6 (relative
depth 0.5), with the 95 percent t-interval over seeds from the jackknife of the methodology. Two random unit directions
in the 768-dimensional residual stream have a mean absolute cosine of 0.029.

| Reading | Seed agreement at L6 [95% interval] | Range over layers 1 to 12 |
| ------- | ----------------------------------- | ------------------------- |
| r1      | 0.447 [0.395, 0.498]                | 0.363 to 0.520            |
| r2      | 0.288 [0.205, 0.372]                | 0.171 to 0.356            |
| r3      | 0.252 [0.163, 0.341]                | 0.163 to 0.290            |
| r5      | 0.986 [0.985, 0.987]                | 0.939 to 0.986            |
| r7      | 0.525 [0.487, 0.563]                | 0.450 to 0.592            |

- All five readings agree far above the random level. R1, R2, R3 and R7 agree between 0.252 and 0.525 at layer 6, so two
  seeds recover directions that overlap only partly; the direction depends on the training draw.
- R5 agrees at 0.986 at layer 6 and between 0.939 and 0.986 at every layer, while its transfer test accuracy is 0.386. A
  direction that is stable across seeds is not for that reason a direction that transfers.

**A2, logit lens.** Each seed's direction is read through the final norm and the unembedding. The separation is the mean
z-score (in standard deviations of the logits over the vocabulary) of six positive pole tokens minus that of six
negative pole tokens, listed in the run; no pole word was dropped. The interval is a 95 percent percentile bootstrap
over the 10 seeds (2000 resamples), as for the reading curves. The last column is the mean and standard deviation of the
same quantity for 100 random directions. The best layer is chosen on the same curve, so the values are optimistic.

| Reading | Best layer | Separation z [95% interval] | Random directions: mean, sd |
| ------- | ---------- | --------------------------- | --------------------------- |
| r1      | L12        | 3.33 [3.17, 3.49]           | -0.02, 0.52                 |
| r2      | L12        | 2.87 [2.36, 3.36]           | -0.02, 0.54                 |
| r3      | L7         | 0.91 [0.53, 1.27]           | 0.03, 0.51                  |
| r5      | L7         | 1.68 [1.64, 1.74]           | -0.08, 0.56                 |
| r7      | L8         | 0.89 [0.69, 1.06]           | -0.01, 0.54                 |

- R1 and R2 are highest at the last layer (12) and far above the random spread: 3.33 for R1 against a random standard
  deviation of 0.52. R1 rises from 0.49 at layer 1 to 3.33 at layer 12.
- The mean separation of R3 and R7 never exceeds 0.91 and 0.89 at any layer, below two random-direction standard
  deviations (1.02 for R3 and 1.08 for R7), so their directions do not clearly separate their pole tokens through the
  lens.
- R5 peaks at 1.68 at layer 7.

## Sentiment steering on the GPT-2 ladder (C1)

C1 steers generation with the SST-2 sentiment reading vector and scores the completions of 100 neutral prompts with two
sentiment classifiers; the tables use the primary judge (probability of the positive class) and the check judge is
discussed below. The coefficient is relative to the mean residual norm at the steered layer, so one value is comparable
across sizes. Perplexity is that of held-out neutral text under the steered model, a measure of side effects and not of
the generated text. The layer was chosen by a sweep on a separate prompt list. All four sizes steer at their last block
and over a window of the last three blocks. `judge agreement` is the share of swept layers whose dose slope has the same
sign under both judges.

| Model       | Reading layer | Steer layer | Window     | Judge agreement |
| ----------- | ------------- | ----------- | ---------- | --------------- |
| gpt2        | L8            | L12         | 10, 11, 12 | 1.000           |
| gpt2-medium | L20           | L24         | 22, 23, 24 | 0.833           |
| gpt2-large  | L35           | L36         | 34, 35, 36 | 1.000           |
| gpt2-xl     | L24           | L48         | 46, 47, 48 | 0.833           |

Primary-judge score, mean over the three steering readers with the range in parentheses (single-run arms have no range):

| Arm          | gpt2                | gpt2-medium         | gpt2-large          | gpt2-xl             |
| ------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| none         | 0.58                | 0.55                | 0.63                | 0.63                |
| linear +0.1  | 0.92 (0.90 to 0.94) | 0.89 (0.86 to 0.92) | 0.81 (0.78 to 0.83) | 0.84 (0.83 to 0.86) |
| linear +0.2  | 0.98 (0.95 to 1.00) | 0.95 (0.93 to 0.97) | 0.94 (0.87 to 0.98) | 0.85 (0.77 to 0.98) |
| linear +0.3  | 0.98 (0.95 to 1.00) | 0.98 (0.98 to 0.99) | 0.95 (0.88 to 0.99) | 0.74 (0.60 to 0.99) |
| linear -0.1  | 0.27 (0.25 to 0.30) | 0.20 (0.17 to 0.24) | 0.38 (0.37 to 0.39) | 0.26 (0.16 to 0.32) |
| linear -0.2  | 0.11 (0.09 to 0.14) | 0.10 (0.05 to 0.19) | 0.13 (0.10 to 0.15) | 0.09 (0.03 to 0.14) |
| linear -0.3  | 0.11 (0.07 to 0.14) | 0.13 (0.04 to 0.22) | 0.06 (0.01 to 0.11) | 0.06 (0.01 to 0.09) |
| prompting +1 | 0.78                | 0.76                | 0.76                | 0.83                |
| prompting +2 | 0.77                | 0.80                | 0.81                | 0.80                |
| prompting -1 | 0.42                | 0.38                | 0.47                | 0.55                |
| prompting -2 | 0.34                | 0.40                | 0.38                | 0.51                |

Perplexity of neutral text:

| Arm         | gpt2                   | gpt2-medium               | gpt2-large             | gpt2-xl                |
| ----------- | ---------------------- | ------------------------- | ---------------------- | ---------------------- |
| none        | 53.6                   | 39.2                      | 34.0                   | 30.9                   |
| linear +0.1 | 66.5 (65.1 to 68.6)    | 64.2 (62.7 to 65.4)       | 39.6 (39.3 to 40.1)    | 41.9 (38.9 to 44.1)    |
| linear +0.2 | 139.3 (132.3 to 144.5) | 246.9 (235.5 to 253.5)    | 68.3 (66.7 to 69.1)    | 121.4 (95.3 to 137.2)  |
| linear +0.3 | 438.0 (400.2 to 499.8) | 1254.1 (1084.3 to 1383.4) | 159.0 (144.9 to 168.8) | 468.3 (309.6 to 570.4) |
| linear -0.1 | 72.3 (67.6 to 75.0)    | 60.5 (58.9 to 62.1)       | 40.6 (40.0 to 41.2)    | 44.1 (42.8 to 45.2)    |
| linear -0.2 | 160.5 (139.4 to 177.2) | 191.2 (169.3 to 208.6)    | 67.3 (66.1 to 68.7)    | 118.2 (104.8 to 127.6) |
| linear -0.3 | 503.7 (412.9 to 590.1) | 813.1 (736.6 to 867.5)    | 139.4 (135.4 to 142.0) | 382.6 (311.5 to 434.6) |

- **Direction.** Linear steering at a coefficient of 0.1 raises the score at positive and lowers it at negative
  coefficients on every size: from 0.58, 0.55, 0.63 and 0.63 without steering to 0.92, 0.89, 0.81 and 0.84 at +0.1, and
  to 0.27, 0.20, 0.38 and 0.26 at -0.1. The check judge agrees in direction (at +0.1 it gives 0.876, 0.849, 0.756 and
  0.815 against 0.495, 0.505, 0.583 and 0.524 without steering, and at -0.1 it gives 0.150, 0.093, 0.252 and 0.176).
- **Against prompting.** At a coefficient of 0.1 the steered score is above both prompting arms (+1 and +2) on `gpt2`
  and `gpt2-medium`, and level with the stronger one on `gpt2-large` (0.81 against 0.81) and `gpt2-xl` (0.84 against
  0.83, inside the range over readers). Downward, -0.1 is below both negative prompting arms on `gpt2`, `gpt2-medium`
  and `gpt2-xl`, and level with the stronger one on `gpt2-large` (0.38 against 0.38). The prompting arms are single runs
  and have no range, so these comparisons carry no interval for the prompting side.
- **Cost.** The perplexity of neutral text rises with the coefficient. At +0.1 and -0.1 it goes from 53.6 to 66.5 and
  72.3 on `gpt2`, from 39.2 to 64.2 and 60.5 on `gpt2-medium`, from 34.0 to 39.6 and 40.6 on `gpt2-large`, and from 30.9
  to 41.9 and 44.1 on `gpt2-xl`. At a coefficient of 0.3 in either direction the mean is at least 139.4 (`gpt2-large` at
  -0.3) and at most 1254.1 (`gpt2-medium` at +0.3).
- **Saturation and `gpt2-xl`.** On the three smaller sizes the score at +0.2 and +0.3 stays at or above 0.87 for every
  reader. On `gpt2-xl` it does not: at +0.3 the readers score between 0.60 and 0.99, and the mean falls from 0.85 at
  +0.2 to 0.74 at +0.3. High scores at +0.3 come with degenerate text: distinct-1 (distinct unigrams over all unigrams)
  is 0.043, 0.047, 0.099 and 0.074 at +0.3 (mean over the three readers) against 0.269, 0.285, 0.265 and 0.262 without
  steering. C1 has no control with a random direction of the same norm, so the table shows the effect of the sentiment
  vector at each size but not how it compares with an arbitrary edit of that size.

## Known versus unknown questions (P9)

P9 asks whether a direction in the residual stream separates the questions a model answers from the questions it
cannot answer, and what adding that direction does. P9a labels each question from the behavior of the model, P9b reads
the known against unknown direction, P9c predicts the errors of the model and P9d adds the direction to the model
while it answers. The definitions, the guess control and the statistics are in the
[methodology](methodology.md#known-versus-unknown-questions-p9).

The runs reported here use the current label schema (12 PopQA relations, the guess control) and are those of `gpt2-xl`
and of the Qwen3 0.6B and 1.7B checkpoints. P9a covers all five models. P9b and P9c ran on `gpt2-xl` and the two base
checkpoints, and P9d on the four Qwen3 checkpoints (base and post-trained); P9b and P9c have no post-trained run. The
GPT-2 ladder below `gpt2-xl` and P9d on `gpt2-xl` are in [Pending](#pending). Questions are labeled from the behavior
of each model, so each model has its own question sets and a count or an AUROC is not a measure of the same questions
across models. Each cell is one run, so no interval includes variation between runs. The post-trained checkpoints use
the chat template with thinking disabled (see the Qwen3 section).

### Labels (P9a)

P9a labels each closed-book question from the behavior of the model on a 15-shot prompt: known if the greedy answer is
correct and at least 3 of 5 samples at temperature 1.0 are correct, unknown if the greedy answer is wrong and no
sample is correct, uncertain otherwise. A run labels 21,988 questions (22,028 for `gpt2-xl`). The sources whose
questions can be known are PopQA (9,467 questions over 12 relations), TriviaQA and NQ-open (1,000 each) and the real
award questions; the others (SyntHal, generated names, unknowable details, FalseQA, KUQ) are unknown by construction.
The counts below are after the guess control, which turns a known question that it rejects into an uncertain one. Each
model is labeled on its own behavior, so the question sets differ between models.

Known / unknown / uncertain questions per source:

| Model           | PopQA (9,467)       | TriviaQA (1,000) | NQ-open (1,000) |
| --------------- | ------------------- | ---------------- | --------------- |
| qwen3-0.6b-base | 329 / 8,195 / 943   | 153 / 668 / 179  | 43 / 850 / 107  |
| qwen3-0.6b      | 277 / 8,446 / 744   | 89 / 776 / 135   | 27 / 911 / 62   |
| qwen3-1.7b-base | 501 / 7,928 / 1,038 | 291 / 502 / 207  | 70 / 775 / 155  |
| qwen3-1.7b      | 517 / 8,137 / 813   | 271 / 592 / 137  | 71 / 848 / 81   |
| gpt2-xl         | 324 / 8,397 / 746   | 94 / 710 / 196   | 14 / 909 / 77   |

Guess control, in questions:

| Model           | Known before | Known after | Guessable stratum | Modal answer | Excluded stratum |
| --------------- | ------------ | ----------- | ----------------- | ------------ | ---------------- |
| qwen3-0.6b-base | 724          | 525         | 14                | 7            | 178              |
| qwen3-0.6b      | 669          | 393         | 0                 | 7            | 269              |
| qwen3-1.7b-base | 1,090        | 863         | 42                | 12           | 173              |
| qwen3-1.7b      | 1,118        | 860         | 24                | 11           | 223              |
| gpt2-xl         | 540          | 432         | 0                 | 5            | 103              |

- **Known questions are few.** After the guess control a model has 393 to 863 known questions: 393 (`qwen3-0.6b`), 432
  (`gpt2-xl`), 525 (`qwen3-0.6b-base`), 860 (`qwen3-1.7b`) and 863 (`qwen3-1.7b-base`), against 21,988 to 22,028
  questions. They come from PopQA (277 to 517 per model), TriviaQA (89 to 291) and NQ-open (14 to 71); the real award
  questions give at most 1.
- **The guess control matters.** It demotes 108 (`gpt2-xl`) to 276 (`qwen3-0.6b`) of the known questions of a model,
  103 to 269 of them because their stratum is excluded (`popqa:sport` is excluded by default, and so are the strata
  whose constant-guess accuracy is above 0.10).
- **Base against post-trained.** At 1.7B the two checkpoints have 863 and 860 known questions; at 0.6B the base
  checkpoint has 525 and the post-trained one 393. Labels come from each model's own behavior, so this is a count of
  what each model answers under the labeling recipe, not a comparison on common questions.
- **Gate and cutoff.** All five models pass the gate (at least 100 guess-adjusted known questions), so the known
  against unknown contrasts are built. The cutoff year is a setting (2017 for `gpt2-xl`, 2024 for Qwen3). The cutoff
  estimated from the labels is stored for two runs, 2014 for `qwen3-1.7b-base` and 2007 for `qwen3-1.7b`, and does not
  match the setting. The only result below that uses the cutoff is the `pre_vs_post_cutoff` row of `gpt2-xl`; the
  post-cutoff contrasts have too few items for the Qwen3 models.

### Reading the direction (P9b)

For each contrast P9b fits a logistic reader on 64 pairs (one question of each side), five times on different draws of
pairs, at a single layer for all the contrasts of a model: L27 for `qwen3-0.6b-base`, L21 for `qwen3-1.7b-base` and
L32 for `gpt2-xl`, chosen on the pooled validation AUROC. A value is the mean over the five readers of the AUROC on
the test questions: the chance that a question of the positive side (the known questions in a known against unknown
contrast) scores above one of the negative side, 0.5 being chance. Three baselines are computed on the same test
questions: `text_word` and `text_char` are classifiers on the words and the characters of the question, fitted on the
train questions of the contrast (no model), and `popularity` is the AUROC of the popularity of the subject, defined
only where every question has an integer popularity (PopQA). Every reader value below is next to them.

For the contrasts with PopQA questions on a side the headline is the within-relation AUROC (the AUROC inside each
relation, averaged with weights `n_pos * n_neg` over the relations that hold both sides), shown next to the
within-relation baselines, because relations differ in how many of their questions are known and a pooled AUROC can
rank the two sides through the relation alone. `obscure_real` is PopQA known against PopQA unknown with training pairs
matched inside a relation; `obscure_real_pop` also matches the popularity bin. The interval stored by P9b is a
bootstrap over the five pair draws of the reader. It measures how much the reader moves with its training draw, not
the sampling of the test questions, so it is not an interval on the AUROC and is not shown; the test sides are small
(the PopQA known side has 87, 118 and 78 questions).

PopQA known against unknown, within relation (the headline), mean AUROC over the five readers:

| Model           | Contrast         | Test known / unknown | Pairs  | Reader | popularity | text_word | text_char |
| --------------- | ---------------- | -------------------- | ------ | ------ | ---------- | --------- | --------- |
| qwen3-0.6b-base | obscure_real     | 87 / 1,661           | 6,941  | 0.919  | 0.799      | 0.678     | 0.673     |
| qwen3-0.6b-base | obscure_real_pop | 87 / 1,661           | 6,941  | 0.891  | 0.799      | 0.678     | 0.673     |
| qwen3-1.7b-base | obscure_real     | 118 / 1,611          | 10,113 | 0.959  | 0.879      | 0.625     | 0.636     |
| qwen3-1.7b-base | obscure_real_pop | 118 / 1,611          | 10,113 | 0.960  | 0.879      | 0.625     | 0.636     |
| gpt2-xl         | obscure_real     | 78 / 1,711           | 6,691  | 0.902  | 0.838      | 0.511     | 0.543     |
| gpt2-xl         | obscure_real_pop | 78 / 1,711           | 6,691  | 0.879  | 0.838      | 0.511     | 0.543     |

The same contrasts pooled over relations, with the AUROC of a random direction:

| Model           | Contrast         | Reader | Random direction | popularity | text_word | text_char |
| --------------- | ---------------- | ------ | ---------------- | ---------- | --------- | --------- |
| qwen3-0.6b-base | obscure_real     | 0.948  | 0.682            | 0.840      | 0.944     | 0.941     |
| qwen3-0.6b-base | obscure_real_pop | 0.948  | 0.682            | 0.840      | 0.944     | 0.941     |
| qwen3-1.7b-base | obscure_real     | 0.961  | 0.648            | 0.865      | 0.873     | 0.873     |
| qwen3-1.7b-base | obscure_real_pop | 0.970  | 0.522            | 0.865      | 0.873     | 0.873     |
| gpt2-xl         | obscure_real     | 0.929  | 0.490            | 0.872      | 0.880     | 0.884     |
| gpt2-xl         | obscure_real_pop | 0.935  | 0.477            | 0.872      | 0.880     | 0.884     |

- **Within relation the reader is above both baselines on every model.** On `obscure_real` it reads 0.919, 0.959 and
  0.902, against a popularity baseline of 0.799, 0.879 and 0.838: a margin between 0.06 and 0.12. The word and
  character classifiers are at 0.678 and 0.673, 0.625 and 0.636, and 0.511 and 0.543 within relation.
- **Pooled, the picture depends on the model.** On `qwen3-0.6b-base` the reader (0.948) is level with `text_word`
  (0.944) and `text_char` (0.941), so the pooled value carries nothing that the wording of the question does not; the
  within-relation margin does not depend on this. On `qwen3-1.7b-base` (0.961 against 0.873) and `gpt2-xl` (0.929
  against 0.880 and 0.884) the pooled reader is above the text baselines. A random direction reads 0.682, 0.648 and
  0.490 pooled on `obscure_real`, below the logistic reader on every model.
- **Matching the popularity bin.** On `obscure_real_pop` the within-relation reader reads 0.891, 0.960 and 0.879,
  within 0.03 of `obscure_real`, with the same baselines.
- **What this does not show.** Popularity and the wording of the question are the only surface controls that P9b
  computes. A reader above both of them is not shown to read what the model knows rather than some other property of
  the question that neither control captures, and each value is one set of five readers on one labeling run.

Other contrasts, as reader / `text_word` / `text_char` (`popularity` is not defined for them). The relational
contrasts, which have PopQA questions on a side, are within relation:

| Contrast            | qwen3-0.6b-base       | qwen3-1.7b-base       | gpt2-xl               |
| ------------------- | --------------------- | --------------------- | --------------------- |
| fictional_generated | 0.989 / 0.819 / 0.987 | 1.000 / 0.851 / 0.992 | 0.998 / 0.856 / 0.983 |
| real_vs_synthal     | 0.978 / 0.736 / 1.000 | 0.986 / 0.728 / 1.000 | 0.962 / 0.732 / 1.000 |
| real_vs_generated   | 0.980 / 0.953 / 0.999 | 0.986 / 0.954 / 0.999 | 0.981 / 0.951 / 0.998 |
| pre_vs_post_cutoff  | not read              | not read              | 1.000 / 1.000 / 1.000 |

The others, pooled:

| Contrast             | qwen3-0.6b-base       | qwen3-1.7b-base       | gpt2-xl               |
| -------------------- | --------------------- | --------------------- | --------------------- |
| triviaqa_known       | 0.905 / 0.774 / 0.727 | 0.883 / 0.693 / 0.671 | 0.905 / 0.779 / 0.787 |
| unknowable_detail    | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| false_premise        | 0.951 / 0.978 / 0.973 | 0.974 / 0.977 / 0.975 | 0.964 / 0.980 / 0.973 |
| kuq_ambiguous        | 0.987 / 0.852 / 0.882 | 0.996 / 0.878 / 0.918 | 0.979 / 0.814 / 0.909 |
| kuq_controversial    | 0.985 / 0.969 / 0.980 | 0.993 / 0.975 / 0.987 | 0.982 / 0.942 / 0.961 |
| kuq_counterfactual   | 0.995 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.999 / 0.999 / 0.999 |
| kuq_false_assumption | 0.991 / 0.976 / 0.985 | 0.993 / 0.978 / 0.991 | 0.985 / 0.958 / 0.967 |
| kuq_future_unknown   | 0.999 / 0.997 / 0.993 | 1.000 / 0.992 / 0.986 | 1.000 / 0.992 / 0.990 |
| kuq_unsolved_problem | 0.985 / 0.949 / 0.943 | 0.992 / 0.953 / 0.967 | 0.994 / 0.950 / 0.956 |

- **The reader has a margin over every text baseline on two contrasts.** `triviaqa_known` sets TriviaQA questions
  labeled known against TriviaQA questions labeled unknown, so one question format stands on both sides: the reader
  reads 0.905, 0.883 and 0.905, against a better text baseline of 0.774, 0.693 and 0.787. `kuq_ambiguous` is the other
  one, with margins of 0.10, 0.08 and 0.07. The `triviaqa_known` test sets are small (32, 55 and 19 known questions).
- **On several contrasts the reader does not exceed the text baselines.** `unknowable_detail` is 1.000 for the reader
  and for both baselines on all three models, so the wording alone separates it. On `false_premise` the reader (0.951,
  0.974, 0.964) is below the better baseline (0.978, 0.977, 0.980). The two contrasts between kinds of unknown
  questions (`real_vs_synthal`, `real_vs_generated`; they do not contain known questions) have `text_char` at 1.000 on
  `real_vs_synthal` and 0.999, 0.999 and 0.998 on `real_vs_generated`, above the reader (0.978, 0.986 and 0.962 on
  `real_vs_synthal`, 0.980, 0.986 and 0.981 on `real_vs_generated`).
- **Elsewhere the margin is small.** `fictional_generated` (PopQA known against generated names, within relation) has
  the reader above `text_char` by 0.002 to 0.015; four KUQ categories (`kuq_controversial`, `kuq_counterfactual`,
  `kuq_false_assumption`, `kuq_future_unknown`) are within 0.02 of the better baseline in either direction, and
  `kuq_unsolved_problem` is above it by 0.025 to 0.038. `pre_vs_post_cutoff` (`gpt2-xl` only, 46 against 20 test
  questions) is 1.000 for the reader and both baselines.
- **Read these cross-dataset rows as a format contrast too.** The known side of most of them is PopQA, TriviaQA or
  NQ-open and the unknown side is another source, so the two sides differ in question format. The text classifiers are
  at 0.94 or above (the better of the two) on every row except `triviaqa_known` and `kuq_ambiguous`, which says that
  the wording separates the sides there; a reader at 0.98 to 1.00 on those rows does not show that it reads knowledge.

### Error prediction (P9c)

P9c asks which score predicts whether the greedy few-shot answer of the model is correct, on held-out PopQA (1,937),
TriviaQA (191) and NQ-open (210) questions. Every predictor is a score where higher means more likely correct and the
value is its AUROC for correct against wrong answers (0.5 is chance). The output-confidence baselines read the model's
own output: `answer_logprob` (log-probability of the greedy answer), `first_token_prob` (probability of the most
likely token at the start of the answer), `neg_entropy` (minus the entropy of that distribution),
`mean_token_logprob`, `consistency` (share of the five sampled answers that equal the greedy one) and `p_true` (the
model grades its own answer). `probe` is a logistic probe fitted on the correctness of the train questions of the same
dataset, so it sees the label of the target. `lat` is the label-free reader of the `obscure_real` contrast of P9b,
applied here without refitting. `popularity` and `template` (the accuracy rate of the relation on the train questions)
use no hidden state and no confidence. The probe and `lat` read layer L27 (`qwen3-0.6b-base`), L27 (`qwen3-1.7b-base`)
and L48 (`gpt2-xl`), chosen on validation.

For PopQA the headline is the within-relation AUROC, with the interval of the methodology: the items are resampled
with replacement inside each relation (1,000 resamples), at that relation's size. The pooled AUROC is secondary,
because the accuracy rate of the relation alone (`template`) already ranks the questions: its pooled AUROC is in the
table below and its within-relation AUROC is 0.500 by construction. The pooled intervals resample the items globally.

PopQA, within relation, AUROC with its interval (208, 264 and 170 correct answers of 1,937):

| Predictor          | qwen3-0.6b-base      | qwen3-1.7b-base      | gpt2-xl              |
| ------------------ | -------------------- | -------------------- | -------------------- |
| answer_logprob     | 0.841 [0.796, 0.876] | 0.883 [0.857, 0.907] | 0.860 [0.824, 0.895] |
| probe              | 0.826 [0.781, 0.864] | 0.881 [0.856, 0.905] | 0.814 [0.770, 0.856] |
| first_token_prob   | 0.826 [0.779, 0.865] | 0.879 [0.853, 0.902] | 0.850 [0.809, 0.888] |
| neg_entropy        | 0.824 [0.781, 0.862] | 0.879 [0.853, 0.903] | 0.830 [0.785, 0.870] |
| mean_token_logprob | 0.836 [0.794, 0.870] | 0.862 [0.831, 0.889] | 0.837 [0.797, 0.874] |
| consistency        | 0.773 [0.727, 0.820] | 0.803 [0.763, 0.841] | 0.767 [0.716, 0.818] |
| lat                | 0.787 [0.734, 0.829] | 0.848 [0.815, 0.877] | 0.765 [0.715, 0.814] |
| p_true             | 0.719 [0.667, 0.769] | 0.776 [0.726, 0.821] | 0.431 [0.373, 0.487] |
| popularity         | 0.696 [0.641, 0.749] | 0.776 [0.736, 0.815] | 0.679 [0.619, 0.740] |

PopQA pooled over relations, as AUROC / AURC (area under the risk-coverage curve, lower is better):

| Predictor                            | qwen3-0.6b-base      | qwen3-1.7b-base      | gpt2-xl              |
| ------------------------------------ | -------------------- | -------------------- | -------------------- |
| probe                                | 0.946 / 0.694        | 0.940 / 0.635        | 0.931 / 0.746        |
| answer_logprob                       | 0.933 / 0.703        | 0.910 / 0.655        | 0.924 / 0.749        |
| neg_entropy                          | 0.907 / 0.710        | 0.899 / 0.661        | 0.881 / 0.759        |
| lat                                  | 0.875 / 0.727        | 0.879 / 0.679        | 0.856 / 0.776        |
| template                             | 0.892 / 0.740        | 0.843 / 0.688        | 0.880 / 0.775        |
| popularity                           | 0.588 / 0.811        | 0.648 / 0.748        | 0.600 / 0.838        |
| error rate (risk at full coverage)   | 0.893                | 0.864                | 0.912                |
| risk at 20% coverage, probe          | 0.546 [0.485, 0.598] | 0.456 [0.399, 0.523] | 0.642 [0.585, 0.688] |
| risk at 20% coverage, answer_logprob | 0.554 [0.503, 0.613] | 0.492 [0.436, 0.546] | 0.637 [0.580, 0.688] |

TriviaQA and NQ-open, pooled AUROC with its interval (the items have no relation):

| Predictor                | qwen3-0.6b-base      | qwen3-1.7b-base      | gpt2-xl              |
| ------------------------ | -------------------- | -------------------- | -------------------- |
| TriviaQA answer_logprob  | 0.886 [0.836, 0.931] | 0.819 [0.758, 0.878] | 0.774 [0.682, 0.864] |
| TriviaQA neg_entropy     | 0.862 [0.803, 0.917] | 0.843 [0.784, 0.894] | 0.764 [0.671, 0.850] |
| TriviaQA probe           | 0.868 [0.809, 0.926] | 0.811 [0.753, 0.868] | 0.740 [0.645, 0.831] |
| TriviaQA lat             | 0.716 [0.639, 0.795] | 0.674 [0.601, 0.750] | 0.700 [0.600, 0.792] |
| TriviaQA p_true          | 0.778 [0.699, 0.853] | 0.772 [0.701, 0.840] | 0.481 [0.367, 0.586] |
| TriviaQA correct answers | 48 of 191            | 78 of 191            | 40 of 191            |
| NQ-open answer_logprob   | 0.776 [0.648, 0.880] | 0.790 [0.697, 0.867] | 0.636 [0.448, 0.818] |
| NQ-open neg_entropy      | 0.740 [0.641, 0.826] | 0.745 [0.659, 0.831] | 0.758 [0.617, 0.871] |
| NQ-open probe            | 0.795 [0.688, 0.895] | 0.812 [0.742, 0.878] | 0.725 [0.580, 0.858] |
| NQ-open lat              | 0.783 [0.653, 0.897] | 0.848 [0.777, 0.913] | 0.629 [0.506, 0.750] |
| NQ-open p_true           | 0.788 [0.683, 0.879] | 0.694 [0.590, 0.787] | 0.347 [0.192, 0.511] |
| NQ-open correct answers  | 20 of 210            | 38 of 210            | 12 of 210            |

What `probe` and `lat` add to the output confidence: a logistic stacker of the five confidence columns (fitted on the
validation items, scored on the test items) against the same stacker plus the predictor, as a paired AUROC difference
with its interval. These are pooled AUROCs; no within-relation increment is stored.

| Predictor                 | qwen3-0.6b-base        | qwen3-1.7b-base        | gpt2-xl                |
| ------------------------- | ---------------------- | ---------------------- | ---------------------- |
| PopQA confidence stack    | 0.908                  | 0.900                  | 0.893                  |
| PopQA + lat               | 0.000 [0.000, 0.000]   | 0.002 [-0.002, 0.006]  | 0.011 [-0.006, 0.032]  |
| PopQA + probe             | 0.044 [0.026, 0.063]   | 0.044 [0.028, 0.061]   | 0.048 [0.025, 0.076]   |
| TriviaQA confidence stack | 0.869                  | 0.858                  | 0.774                  |
| TriviaQA + lat            | -0.008 [-0.024, 0.005] | -0.006 [-0.018, 0.006] | 0.003 [-0.004, 0.010]  |
| TriviaQA + probe          | 0.008 [0.000, 0.018]   | 0.013 [-0.004, 0.029]  | 0.003 [-0.013, 0.019]  |
| NQ-open confidence stack  | 0.819                  | 0.800                  | 0.753                  |
| NQ-open + lat             | -0.013 [-0.048, 0.023] | 0.047 [0.015, 0.085]   | -0.005 [-0.027, 0.012] |
| NQ-open + probe           | -0.003 [-0.027, 0.022] | 0.036 [0.007, 0.066]   | 0.023 [-0.009, 0.067]  |

- **The probe does not beat `answer_logprob` within relation.** On PopQA the probe reads 0.826, 0.881 and 0.814,
  against 0.841, 0.883 and 0.860 for `answer_logprob`: lower on every model, with overlapping intervals. The
  label-free `lat` is lower still (0.787, 0.848 and 0.765). The probe, `first_token_prob`, `neg_entropy` and
  `mean_token_logprob` are above the within-relation popularity baseline (0.696, 0.776 and 0.679) with separated
  intervals on every model; `lat` is above it as a point estimate and its interval overlaps the baseline's on every
  model.
- **Pooled, the probe is higher but the pooled value mixes relations.** Pooled over PopQA the probe reads 0.946, 0.940
  and 0.931 against 0.933, 0.910 and 0.924 for `answer_logprob`; the AURC is 0.694, 0.635 and 0.746 for the probe
  against 0.703, 0.655 and 0.749 (lower is better), and the risk at 20 percent coverage is within the other's interval
  on every model. The accuracy rate of the relation alone reads 0.892, 0.843 and 0.880 pooled. Added to the
  output-confidence stack, the probe raises the pooled AUROC by +0.044, +0.044 and +0.048, with intervals above zero;
  `lat` adds +0.000, +0.002 and +0.011, with intervals that include zero. How much of the probe's pooled gain is
  within-relation is not stored.
- **TriviaQA and NQ-open.** On TriviaQA the probe is below `answer_logprob` on every model (0.868 against 0.886, 0.811
  against 0.819, 0.740 against 0.774). On NQ-open its point estimate is above it (0.795 against 0.776, 0.812 against
  0.790, 0.725 against 0.636), but the sets hold 20, 38 and 12 correct answers and the intervals are 0.14 to 0.37 wide
  and overlap. The increments over the confidence stack have intervals that include zero on both sets (the probe on
  TriviaQA at `qwen3-0.6b-base` has a lower bound of exactly 0.000), except `lat` (+0.047) and the probe (+0.036) on
  NQ-open at `qwen3-1.7b-base`.
- **`p_true` on `gpt2-xl` is below chance on PopQA.** It reads 0.293 [0.253, 0.334] pooled and 0.431 [0.373, 0.487]
  within relation, with intervals below 0.5. On TriviaQA (0.481) and NQ-open (0.347) the intervals include 0.5. On the
  Qwen3 models it is 0.719 and 0.776 within relation, below `answer_logprob`.

### Steering (P9d)

P9d adds the `obscure_real` direction (PopQA known against PopQA unknown, logistic, 64 pairs, one layer chosen on the
validation AUROC) to the residual stream while the model answers. The coefficient is relative to the mean residual
norm at the steered layer, on the grid -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4; a positive coefficient moves along the
direction toward the known side. Each condition uses up to 100 prompts per side and type (the selectivity tables give
the numbers), and three random directions of the same norm (`random0` to `random2`) run on the same prompts. A
post-trained checkpoint answers a chat prompt with the instruction to say that it does not know when it does not, and
the behavior is abstention, an indicator from phrase matching on the completion; a base checkpoint cannot abstain, so
the behavior is the shift of the entropy of the next-token distribution (and of the log-probability of the gold
answer, which exists for the PopQA questions). These are different quantities, and base and post-trained results are
not compared as one. `fewshot` reads the direction on the questions in the 15-shot prompt; `concept` reads it with the
paper's concept template (the word "knowledge"), in the chat format; the base checkpoints use `fewshot` only. The
abstention indicator was not checked by hand: the stored sample of 100 completions per run has no human labels.

The slope of an arm is the mean over the items of the least-squares slope of the outcome on the whole coefficient grid
(per unit of relative coefficient), with a percentile bootstrap over the items. Two claims are kept apart.
**Distinguishable** means that the direction moves the outcome more than a random direction of the same norm: on the
unknown side of `obscure_real` (the headline cell), the absolute slope of the direction minus the absolute slope of an
arm has a lower bound above zero for each of the three random arms (`Min gap bound` is the smallest of the three lower
bounds). **Selective** means that the effect is larger on unknown than on known items: the absolute mean slope on the
unknown items minus that on the known items has a lower bound above zero. The two sides hold different items, so each
resample draws them independently, and the interval is oriented by the sign of the unknown mean. The selectivity
numbers were recomputed from the stored per-item slopes with the current code (`selectivity_gap`): the stored
selectivity tables of these runs, made before the interval was oriented, differ in some bounds, and one flag changes
(the `unknowable_detail` row of `qwen3-1.7b`, fewshot: selective in the stored table, not selective now). There is no
correction for multiple comparisons. `obscure_real` and `fictional_generated` share one known pool,
`unknowable_detail` and `false_premise` share another, so their rows are not independent. In the tables `post` is the
post-trained checkpoint, scored on abstention, and `base` the base checkpoint, scored on entropy.

Headline cell (unknown side of `obscure_real`), slope of the direction with its 95 percent interval. The validation
AUROC of the direction at the steered layer is 0.943, 0.922, 0.869, 0.795 for the four post-trained runs and 0.966 and
0.934 for the base ones, in the order of the rows.

| Run (steered layer)      | Slope of the direction  | Slopes of random0 / 1 / 2 | Min gap bound | Distinguishable |
| ------------------------ | ----------------------- | ------------------------- | ------------- | --------------- |
| 1.7b post, fewshot (L24) | 0.040 [-0.031, 0.114]   | 0.388 / -0.017 / 0.095    | -0.481        | no              |
| 0.6b post, fewshot (L28) | 0.114 [0.033, 0.198]    | -0.386 / -0.129 / -0.693  | -0.738        | no              |
| 1.7b post, concept (L12) | -1.112 [-1.224, -1.002] | -0.348 / 0.483 / 0.174    | 0.514         | yes             |
| 0.6b post, concept (L26) | 0.005 [-0.091, 0.102]   | -0.564 / -0.157 / 0.229   | -0.674        | no              |
| 1.7b base (L20)          | -5.882 [-6.095, -5.655] | 0.155 / 0.941 / 0.023     | 4.722         | yes             |
| 0.6b base (L27)          | -4.099 [-4.360, -3.781] | 0.111 / -0.662 / -0.221   | 3.096         | yes             |

Post-trained checkpoints: abstention on the `obscure_real` items at the coefficient chosen on the calibration items,
as the change in the share of abstentions from the unsteered run (95 percent interval). The last row is the false
refusals on the known items.

|                    | 1.7b post, fewshot   | 0.6b post, fewshot   | 1.7b post, concept   | 0.6b post, concept   |
| ------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| Coefficient        | +0.4                 | +0.1                 | -0.4                 | +0.4                 |
| Direction, unknown | -0.01 [-0.06, +0.04] | +0.03 [-0.02, +0.08] | +0.50 [+0.41, +0.60] | -0.01 [-0.09, +0.07] |
| random0, unknown   | +0.09 [+0.03, +0.16] | -0.04 [-0.09, +0.00] | +0.29 [+0.19, +0.40] | -0.03 [-0.11, +0.06] |
| random1, unknown   | +0.01 [-0.04, +0.06] | -0.04 [-0.09, +0.01] | -0.08 [-0.16, +0.00] | -0.05 [-0.13, +0.03] |
| random2, unknown   | +0.01 [-0.06, +0.08] | -0.03 [-0.08, +0.01] | +0.10 [+0.00, +0.20] | +0.28 [+0.19, +0.37] |
| Direction, known   | -0.01 [-0.04, +0.02] | +0.00 [+0.00, +0.00] | +0.05 [+0.02, +0.10] | -0.01 [-0.06, +0.03] |

Selectivity of the post-trained checkpoints, per evaluation type (n = known / unknown items; Known and Unknown are the
abstention slopes of the direction on each side; gap = absolute unknown slope minus absolute known slope):

| Run                | Type                | n (k / u) | Known  | Unknown | Gap [95% interval]     | Selective |
| ------------------ | ------------------- | --------- | ------ | ------- | ---------------------- | --------- |
| 1.7b post, fewshot | obscure_real        | 100 / 100 | -0.014 | 0.040   | 0.026 [-0.064, 0.095]  | no        |
| 1.7b post, fewshot | fictional_generated | 100 / 100 | -0.014 | 0.064   | 0.050 [-0.067, 0.162]  | no        |
| 1.7b post, fewshot | unknowable_detail   | 69 / 60   | 0.000  | -0.028  | 0.028 [-0.024, 0.095]  | no        |
| 1.7b post, fewshot | false_premise       | 69 / 83   | 0.000  | 0.000   | 0.000 [0.000, 0.000]   | no        |
| 0.6b post, fewshot | obscure_real        | 72 / 100  | -0.003 | 0.114   | 0.111 [0.002, 0.171]   | yes       |
| 0.6b post, fewshot | fictional_generated | 72 / 100  | -0.003 | 0.524   | 0.521 [0.358, 0.629]   | yes       |
| 0.6b post, fewshot | unknowable_detail   | 24 / 60   | 0.079  | -0.119  | 0.040 [-0.187, 0.260]  | no        |
| 0.6b post, fewshot | false_premise       | 24 / 83   | 0.079  | -0.026  | -0.054 [-0.213, 0.121] | no        |
| 1.7b post, concept | obscure_real        | 100 / 100 | -0.148 | -1.112  | 0.964 [0.821, 1.102]   | yes       |
| 1.7b post, concept | fictional_generated | 100 / 100 | -0.148 | -1.064  | 0.917 [0.767, 1.048]   | yes       |
| 1.7b post, concept | unknowable_detail   | 69 / 60   | -0.014 | -0.369  | 0.355 [0.200, 0.528]   | yes       |
| 1.7b post, concept | false_premise       | 69 / 83   | -0.014 | -0.442  | 0.428 [0.290, 0.562]   | yes       |
| 0.6b post, concept | obscure_real        | 72 / 100  | -0.026 | 0.005   | -0.022 [-0.143, 0.065] | no        |
| 0.6b post, concept | fictional_generated | 72 / 100  | -0.026 | 0.081   | 0.054 [-0.112, 0.195]  | no        |
| 0.6b post, concept | unknowable_detail   | 24 / 60   | 0.050  | 0.159   | 0.109 [-0.064, 0.274]  | no        |
| 0.6b post, concept | false_premise       | 24 / 83   | 0.050  | 0.181   | 0.131 [0.026, 0.244]   | yes       |

Selectivity of the base checkpoints, same columns (entropy and, on `obscure_real`, the log-probability of the gold
answer; the other types have no gold answer on the unknown side). The post-trained checkpoints are scored on
abstention and the base ones on entropy, so the two tables are not comparable:

| Run       | Type                | Outcome      | n (k / u) | Known  | Unknown | Gap [95% interval]     | Selective |
| --------- | ------------------- | ------------ | --------- | ------ | ------- | ---------------------- | --------- |
| 1.7b base | obscure_real        | entropy      | 100 / 100 | -4.848 | -5.882  | 1.034 [0.545, 1.492]   | yes       |
| 1.7b base | obscure_real        | gold logprob | 100 / 100 | 0.536  | -4.826  | 4.290 [3.056, 5.509]   | yes       |
| 1.7b base | fictional_generated | entropy      | 100 / 100 | -4.848 | -4.766  | -0.082 [-0.559, 0.425] | no        |
| 1.7b base | unknowable_detail   | entropy      | 76 / 60   | -3.838 | -5.087  | 1.249 [0.837, 1.702]   | yes       |
| 1.7b base | false_premise       | entropy      | 76 / 83   | -3.838 | -3.610  | -0.228 [-0.690, 0.249] | no        |
| 0.6b base | obscure_real        | entropy      | 87 / 100  | -0.487 | -4.099  | 3.612 [2.788, 4.154]   | yes       |
| 0.6b base | obscure_real        | gold logprob | 87 / 100  | 0.024  | -2.885  | 2.861 [1.849, 3.463]   | yes       |
| 0.6b base | fictional_generated | entropy      | 87 / 100  | -0.487 | -3.707  | 3.220 [2.397, 3.765]   | yes       |
| 0.6b base | unknowable_detail   | entropy      | 39 / 60   | -2.323 | -6.107  | 3.784 [2.912, 4.673]   | yes       |
| 0.6b base | false_premise       | entropy      | 39 / 83   | -2.323 | -5.331  | 3.008 [2.146, 3.929]   | yes       |

Base checkpoints, the curve behind the slopes: shift of the entropy on `obscure_real` at the two ends of the grid for
the direction (95 percent interval) and the range of the three random directions.

| Run       | Side    | Direction at -0.4       | Direction at +0.4       | Random at -0.4 | Random at +0.4 |
| --------- | ------- | ----------------------- | ----------------------- | -------------- | -------------- |
| 1.7b base | known   | +3.861 [+3.630, +4.097] | -0.140 [-0.338, +0.051] | -0.04 to +0.20 | +0.17 to +0.49 |
| 1.7b base | unknown | +1.618 [+1.447, +1.813] | -2.947 [-3.155, -2.715] | -0.35 to +0.07 | +0.09 to +0.39 |
| 0.6b base | known   | +3.084 [+2.869, +3.298] | +3.059 [+2.471, +3.672] | +1.05 to +1.31 | +0.90 to +1.02 |
| 0.6b base | unknown | +1.137 [+1.011, +1.275] | -1.979 [-2.241, -1.660] | -0.13 to +0.06 | -0.44 to -0.06 |

- **Distinguishable, post-trained.** Only `qwen3-1.7b` with the concept template is distinguishable: its slope is
  -1.112 [-1.224, -1.002], the largest absolute slope of a random arm is 0.483 and the smallest gap bound is 0.514. In
  the other three runs one random arm has a larger absolute slope than the direction (smallest gap bounds -0.481,
  -0.738 and -0.674); for the fewshot template the largest random slopes are 0.388 (1.7b) and 0.693 (0.6b), against
  direction slopes of 0.040 and 0.114.
- **Distinguishable, base.** Both base checkpoints are distinguishable on entropy: slopes -5.882 and -4.099, against
  random slopes of at most 0.941 and 0.662 in absolute value.
- **Selective is a separate result.** `qwen3-0.6b` with the fewshot template is selective on `obscure_real` (gap 0.111
  [0.002, 0.171]) and `fictional_generated` (0.521 [0.358, 0.629]) and is not distinguishable. `qwen3-1.7b` with the
  concept template is selective on all four types (gaps 0.355 to 0.964) and distinguishable on the headline.
  `qwen3-0.6b` with the concept template is selective on `false_premise` only (0.131 [0.026, 0.244]) and is not
  distinguishable. `qwen3-1.7b` with the fewshot template is neither, on any type; its `false_premise` row has a slope
  of exactly 0.000 on both sides.
- **Size of the effect, `qwen3-1.7b`, concept.** At the coefficient -0.4 the share of abstentions on the unknown
  `obscure_real` questions goes from 0.35 to 0.85 (+0.50 [+0.41, +0.60]), and the false refusals on the known
  questions from 0.09 to 0.14 (+0.05 [+0.02, +0.10]). A random direction of the same norm is not neutral: `random0`
  raises abstention on the unknown questions by +0.29 [+0.19, +0.40] and `random2` by +0.10 [+0.00, +0.20]. At 0.6b
  with the concept template `random2` raises it by +0.28 [+0.19, +0.37] while the direction does not (-0.01 [-0.09,
  +0.07]).
- **Base checkpoints: selective by the slope test, but the known items move too.** Both are selective on the headline
  `obscure_real` entropy (gap 1.034 [0.545, 1.492] at 1.7b, 3.612 [2.788, 4.154] at 0.6b), and at 0.6b on all four
  types. At 1.7b the entropy slope on the known items is -4.848, against -5.882 on the unknown ones. The curve shows
  why the slope can mislead: at 1.7b the entropy of the known items rises by +3.861 at -0.4 (the unknown items:
  +1.618) and changes by -0.140 at +0.4 (the unknown items: -2.947); at 0.6b the known items rise by +3.084 at -0.4
  and +3.059 at +0.4, so their slope (-0.487) is a cancellation of two rises, while the unknown items fall
  monotonically (-4.099). A slope-based gap does not see a rise at both ends. The random directions move the known
  items by +0.90 to +1.31 at 0.6b.
- **Gold-answer log-probability (base).** On the known items the slope is flat (0.536 [-0.312, 1.290] at 1.7b, 0.024
  [-0.520, 0.586] at 0.6b); on the unknown items it is negative (-4.826 [-5.840, -3.737] and -2.885 [-3.627, -2.188]).
  The gaps are 4.290 [3.056, 5.509] and 2.861 [1.849, 3.463]. Unlike the entropy, it is flat on the PopQA known items,
  and it is defined only on the questions that have a gold answer. At 1.7b the fictional and false-premise rows are
  not selective on entropy (gaps -0.082 and -0.228, intervals that include zero).
- **What this does and does not show.** On the headline outcome, the direction moves abstention (post-trained, concept
  template at 1.7b) or entropy (both base checkpoints) more than a random direction of the same norm does. It is not
  shown to be a control of what the model knows: a random direction of the same norm also moves the outcome in several
  runs, the effect on the known items is not absent, and the abstention indicator is a phrase match whose errors were
  not measured. The runs are one per cell, and base and post-trained results answer different questions.

## Run identity

All reading runs (R1 to R7) and the A1 and A2 runs were made at commit `a39ae61` with a clean tree, on one laptop GPU
with 8 GB of memory, with the default config of each experiment (batch size 8). The A1 and A2 runs read the `gpt2` runs
of R1, R2, R3, R5 and R7 below. The C1 runs of `gpt2` and `gpt2-xl` were made at `d59a2f7` and those of `gpt2-medium`
and `gpt2-large` at `a39ae61`; the C1 and reading source code is unchanged between them (`git diff --stat` over `src`
lists only `p9d_steering.py` and `p9/behavior.py`). The C1 batch size is 16 for `gpt2` and `gpt2-medium` and 4 for
`gpt2-large` and `gpt2-xl`. The P6 tables were recomputed on the CPU from the finished runs at commit `ca2546c` (no
model is loaded). Library versions and the device are in each run's `meta.json`.

The P9 runs were made on the same laptop GPU with a clean tree: P9a at `ff1177a` (batch size 8, 4 for `gpt2-xl`), P9b
and P9c at `95a8243` (batch size 8, 4 for `gpt2-xl`) and P9d at `a39ae61` (batch size 8). Each P9b, P9c and P9d run
reads the P9a run of the same model, which is the one in the P9a rows below. The selectivity columns of the P9d tables
were recomputed offline from the `item_slopes` of the runs with the current code, not read from the stored
`table_selectivity`. Earlier P9 runs exist, made with the four-relation label pool or before the within-relation
headline, the item resampling of the intervals and the dose-slope headline; they are superseded and are not reported.

Config hashes (12 hex digits), GPT-2:

| Experiment | gpt2         | gpt2-medium  | gpt2-large   | gpt2-xl      |
| ---------- | ------------ | ------------ | ------------ | ------------ |
| R1         | bd4438c46cef | 01fd70a05fa8 | b64b756fad46 | 81e8ecd71341 |
| R2         | e1821ff197ce | 098a48a58a83 | 1a2679b49855 | 83124998c2b8 |
| R3         | a9f7b2a71c22 | d10ebba10935 | 5d0f4f92fc9b | 3507b0628650 |
| R4         | 4702fcb81ed3 | not run      | not run      | not run      |
| R5         | 779f5b093803 | e447c8958b80 | 322bd80f6d19 | 7f0afd7dfe12 |
| R6         | 47e8ec0cf46c | not run      | not run      | not run      |
| R7         | 5908a78acf64 | 3030e4186690 | 796f4c4d19f2 | 5eed122bfd4e |
| C1         | add59c39f574 | 2ed2e53ba7fe | 353f91e1b8dc | de76af1a8a07 |
| P9a        | pending      | pending      | pending      | cc283b36fcae |
| P9b        | pending      | pending      | pending      | 92664dccb7d9 |
| P9c        | pending      | pending      | pending      | 1839575c0eaa |
| P9d        | pending      | pending      | pending      | pending      |

Qwen3:

| Experiment  | qwen3-0.6b-base | qwen3-0.6b   | qwen3-1.7b-base | qwen3-1.7b   |
| ----------- | --------------- | ------------ | --------------- | ------------ |
| R1          | a104feb4f38d    | bd2a906798a0 | 41a75ca84cb0    | f5f06e078f69 |
| R2          | a2adb2792d96    | 2106339f9ff7 | b59c0e034df2    | beca01593887 |
| R3          | c917a73220ea    | cca73836c567 | 84e49fb14d61    | 87868da1fb5b |
| R5          | e2d8b07cd376    | cffc23e593d0 | a5694ff54daa    | 487bd86230f5 |
| R7          | ce3e5ce2ec50    | 9cb5abfedb1b | 66d496ff00fa    | b7d78995c5fc |
| P9a         | 1d9f881153e4    | 10d95352866e | d5d522bfecd4    | f413efcae268 |
| P9b         | 34c0ebff50b0    | not run      | 5ca7821318ef    | not run      |
| P9c         | 264bf7ca3a01    | not run      | 0f684b3c4b66    | not run      |
| P9d fewshot | f9a8f827b6b4    | 9c94025d9ad4 | 9543312bcc45    | 42b7a0daf9b9 |
| P9d concept | not run         | f0b3f4eb3988 | not run         | d43b9e297e42 |

Other runs: R6 text baselines computed at `ca2546c` (not a run); A1 `ad6abbba8fd0`; A2 `be4e34f21b35`; P6
`25ee6de331d4`.

## Pending

Not in this document yet. No dates are promised.

- R4 on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the Qwen3 checkpoints.
- R6 on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the Qwen3 checkpoints, and a rerun of R6 on `gpt2` that writes the
  text baselines itself.
- P1, template ablations.
- P2, token position and a misleading sentence in the prompt.
- P3, stimulus-set size.
- P4, pairing and preprocessing.
- P5, persona wording and instruction confounds for honesty.
- P7, whether a reader that reads well also steers well.
- P8, base against post-trained Qwen3.
- M1, the six direction finders.
- C2, emotion steering; C4, honesty control on TruthfulQA; C5, qualitative galleries (there is no C3).
- LR1, one LoRRA adapter.
- P9a to P9c on `gpt2`, `gpt2-medium` and `gpt2-large`, and P9d on `gpt2-xl` and on any of the three smaller sizes
  that the label gate admits.
- P9b and P9c on the post-trained Qwen3 checkpoints.
- P9: a hand check of the abstention indicator of P9d (the stored samples carry no human labels), an item-level
  interval for the P9b AUROC (only the interval over the pair draws is stored), and a within-relation increment of
  the probe over the output confidence in P9c (only the pooled increment is stored).
- A5, reading directions against Qwen-Scope SAE decoders.