# Findings

This document reports the results of this repository so far. It is a reference for the paper, in the style of
[methodology.md](methodology.md) and [related-work.md](related-work.md): it states what each run measured, the number it
produced, and what that number does and does not show. Experiment ids (`r1`, `p6`, `c1` and so on) are those of the
paper mapping in the methodology, which also describes the pipeline choices, the statistics and the run settings that
are not repeated here. Most numbers are read from the files of a finished run (`summary.json`, `table.json` or the
curves files). The derived ones are the R6 text baselines on `gpt2` (see that section), the P6 tables (recomputed from
the runs), the random-direction cosine of A1, the span of the sizes in decades, and differences or ranges quoted in the
text. Runs are identified by config hash in [Run identity](#run-identity). The document is not complete:
[Pending](#pending) lists what it does not yet report.

Contents:

1. [Reading conventions](#reading-conventions)
2. [LAT reading on the GPT-2 ladder (R1 to R7)](#lat-reading-on-the-gpt-2-ladder-r1-to-r7)
3. [LAT reading on Qwen3](#lat-reading-on-qwen3)
4. [TruthfulQA MC1 against text baselines (R6)](#truthfulqa-mc1-against-text-baselines-r6)
5. [Scaling across the GPT-2 sizes (P6)](#scaling-across-the-gpt-2-sizes-p6)
6. [Direction agreement and logit lens on gpt2 (A1, A2)](#direction-agreement-and-logit-lens-on-gpt2-a1-a2)
7. [Sentiment steering on the GPT-2 ladder (C1)](#sentiment-steering-on-the-gpt-2-ladder-c1)
8. [Run identity](#run-identity)
9. [Pending](#pending)

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

## Run identity

All reading runs (R1 to R7) and the A1 and A2 runs were made at commit `a39ae61` with a clean tree, on one laptop GPU
with 8 GB of memory, with the default config of each experiment (batch size 8). The A1 and A2 runs read the `gpt2` runs
of R1, R2, R3, R5 and R7 below. The C1 runs of `gpt2` and `gpt2-xl` were made at `d59a2f7` and those of `gpt2-medium`
and `gpt2-large` at `a39ae61`; the C1 and reading source code is unchanged between them (`git diff --stat` over `src`
lists only `p9d_steering.py` and `p9/behavior.py`). The C1 batch size is 16 for `gpt2` and `gpt2-medium` and 4 for
`gpt2-large` and `gpt2-xl`. The P6 tables were recomputed on the CPU from the finished runs at commit `ca2546c` (no
model is loaded). Library versions and the device are in each run's `meta.json`.

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

Qwen3:

| Experiment | qwen3-0.6b-base | qwen3-0.6b   | qwen3-1.7b-base | qwen3-1.7b   |
| ---------- | --------------- | ------------ | --------------- | ------------ |
| R1         | a104feb4f38d    | bd2a906798a0 | 41a75ca84cb0    | f5f06e078f69 |
| R2         | a2adb2792d96    | 2106339f9ff7 | b59c0e034df2    | beca01593887 |
| R3         | c917a73220ea    | cca73836c567 | 84e49fb14d61    | 87868da1fb5b |
| R5         | e2d8b07cd376    | cffc23e593d0 | a5694ff54daa    | 487bd86230f5 |
| R7         | ce3e5ce2ec50    | 9cb5abfedb1b | 66d496ff00fa    | b7d78995c5fc |

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
- P9, known against unknown questions (a later part of this document).
- A5, reading directions against Qwen-Scope SAE decoders.