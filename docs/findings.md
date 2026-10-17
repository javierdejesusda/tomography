# Findings

This document reports the results of this repository so far. It is a reference for the paper, in the style of
[methodology.md](methodology.md) and [related-work.md](related-work.md): it states what each run measured, the number it
produced, and what that number does and does not show. Experiment ids (`r1`, `p6`, `c1` and so on) are those of the
paper mapping in the methodology, which also describes the pipeline choices, the statistics and the run settings that
are not repeated here. Most numbers are read from the files of a finished run (`summary.json`, `table.json` or the
curves files). The derived ones are the R6 text baselines on `gpt2` (see that section), the P6 tables (recomputed from
the runs), the random-direction cosine of A1, the P9d selectivity columns (recomputed from the stored per-item
slopes), the span of the sizes in decades, the P1 spans, family ranges and transfer ranges (computed from the table
and matrix files of the P1 runs), the means and ranges over readers of the Qwen3 C1 tables (computed from the
per-reader rows), the maxima over layers of the random-direction series of R6, the P1 spans and cosine ranges of the
`gpt2` truth and emotion tasks (computed from the table file), the C2 and P7 curve values (read from the curves
files), and differences or ranges quoted in the text. The P8 tables are read from the P8 run, which aggregates
finished runs. Runs are identified by config hash in [Run identity](#run-identity). The document is not
complete: [Pending](#pending) lists what it does not yet report.

Contents:

1. [Reading conventions](#reading-conventions)
2. [LAT reading on the GPT-2 ladder (R1 to R7)](#lat-reading-on-the-gpt-2-ladder-r1-to-r7)
3. [LAT reading on Qwen3](#lat-reading-on-qwen3)
4. [TruthfulQA MC1 against text baselines (R6)](#truthfulqa-mc1-against-text-baselines-r6)
5. [Scaling across the GPT-2 sizes (P6)](#scaling-across-the-gpt-2-sizes-p6)
6. [Direction agreement and logit lens on gpt2 (A1, A2)](#direction-agreement-and-logit-lens-on-gpt2-a1-a2)
7. [Template sensitivity of the sentiment reader (P1)](#template-sensitivity-of-the-sentiment-reader-p1)
8. [Position, sample size and direction finder on gpt2 (P2, P3,
P4)](#position-sample-size-and-direction-finder-on-gpt2-p2-p3-p4) 9. [Sentiment steering on the GPT-2 ladder
(C1)](#sentiment-steering-on-the-gpt-2-ladder-c1) 10. [Sentiment steering on Qwen3
(C1)](#sentiment-steering-on-qwen3-c1) 11. [Other steering experiments on gpt2 (M1, C2, P7, LR1, C4,
C5)](#other-steering-experiments-on-gpt2-m1-c2-p7-lr1-c4-c5) 12. [Base against post-trained Qwen3
(P8)](#base-against-post-trained-qwen3-p8) 13. [Known versus unknown questions (P9)](#known-versus-unknown-questions-p9)
14. [Run identity](#run-identity) 15. [Pending](#pending)

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
have their own tables below and in the R6 section.

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

R4 on the four GPT-2 sizes (one reader per emotion against the others, same protocol; test accuracy with its interval
and the selected layer; runs `r4/gpt2/4702fcb81ed3`, `r4/gpt2-medium/981d9dd9916e`, `r4/gpt2-large/788316a35990` and
`r4/gpt2-xl/bee62b37444a`):

| Emotion   | gpt2                     | gpt2-medium              | gpt2-large               | gpt2-xl                  |
| --------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| happiness | 0.876 [0.847, 0.908] L12 | 0.921 [0.900, 0.942] L22 | 0.968 [0.955, 0.984] L36 | 0.937 [0.926, 0.947] L47 |
| sadness   | 0.930 [0.916, 0.943] L9  | 0.930 [0.918, 0.941] L12 | 0.950 [0.936, 0.964] L18 | 0.984 [0.970, 0.995] L21 |
| anger     | 0.976 [0.971, 0.978] L11 | 0.942 [0.931, 0.953] L9  | 0.942 [0.933, 0.951] L13 | 0.949 [0.940, 0.958] L21 |
| fear      | 0.968 [0.964, 0.974] L11 | 0.970 [0.962, 0.977] L18 | 0.960 [0.955, 0.966] L27 | 0.985 [0.977, 0.992] L33 |
| disgust   | 0.997 [0.992, 1.000] L5  | 1.000 [1.000, 1.000] L10 | 0.997 [0.992, 1.000] L18 | 0.995 [0.987, 1.000] L3  |
| surprise  | 0.922 [0.906, 0.936] L8  | 0.967 [0.958, 0.972] L16 | 0.953 [0.944, 0.961] L22 | 0.894 [0.889, 0.903] L1  |

The test sets are the same on every size (38, 44, 45, 53, 39 and 36 pairs for the six emotions in the order of the
table). Every one of the 24 cells is at 0.876 or above, the lowest being `gpt2` for happiness and the highest 1.000 for
disgust on `gpt2-medium`.

- **No trend with size.** Accuracy rises with size for happiness (0.876, 0.921, 0.968, 0.937) and sadness (0.930, 0.930,
  0.950, 0.984), falls from `gpt2` for anger (0.976, then 0.942 to 0.949), and is non-monotonic for surprise (0.922,
  0.967, 0.953, 0.894). With 36 to 53 test pairs and one run per size, these differences are small against the
  evaluation sampling that the intervals do not cover.
- **The selected layer is often early or flat.** At layer 1 the readers already reach 0.70 to 0.99 (for example
  happiness 0.700 on `gpt2-medium`, disgust 0.959 to 0.995 on all four sizes). `gpt2-xl` surprise is selected at L1
  (0.894) and `gpt2-xl` disgust at L3, where the curve is flat. A high accuracy at these layers is consistent with
  emotion words in the statements that a shallow reader can use; this is not tested here.
- **Baselines at the selected layer.** The random-direction series is at 0.558 to 0.703 on `gpt2`, 0.534 to 0.613 on
  `gpt2-medium`, 0.587 to 0.639 on `gpt2-large` and 0.547 to 0.620 on `gpt2-xl`, and the shuffled-label series at 0.477
  to 0.566, 0.486 to 0.537, 0.489 to 0.531 and 0.477 to 0.533 (computed from the curves files). Validation accuracy is
  above test accuracy by at most 0.062 on `gpt2`, 0.035 on `gpt2-medium`, 0.040 on `gpt2-large` and 0.057 on
  `gpt2-xl` (it is below it for some cells).

## LAT reading on Qwen3

Same protocol on the base and post-trained checkpoints of Qwen3 0.6B and 1.7B. The post-trained checkpoints use the chat
template with thinking disabled; the base checkpoints use plain tags (see
[methodology](methodology.md#scope-and-models)). The post-trained checkpoints are small distilled models used with
thinking disabled, so a base against post-trained contrast here is a contrast with that recipe and not with instruction
tuning in general. It is descriptive; the paired comparison is in
[Base against post-trained Qwen3 (P8)](#base-against-post-trained-qwen3-p8).

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

R6 was run on all four GPT-2 sizes; this table is the `gpt2` run and the other sizes follow below it. The question set
is split by hash into 511 train, 133 validation and 173 test questions of 817. Every row below is scored on the same 173
test questions; the chance line (the mean of one over the number of choices) is 0.228. The LAT rows come from the run;
the standard and heuristic rows come from the same run's table. The two text-only baselines are classifiers on the
answer texts alone (TF-IDF with a logistic regression fitted on the train answers, no question and no model). The `gpt2`
run was made before the text baselines were added to R6, so their rows were computed with the code that now writes them
(`text_scores`, the same estimator, and the same split and test questions as the LAT rows), not read from the run. The
runs of the three larger sizes store them, and they are identical there, as they do not depend on the model.

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

R6 on the four GPT-2 sizes (runs `r6/gpt2/47e8ec0cf46c`, `r6/gpt2-medium/04fc7ead5f13`,
`r6/gpt2-large/f058c0f19396` and `r6/gpt2-xl/60cc952b4ec5`). The same 173 test questions, the same text baselines and
the same layer rules in every column; the interval of a LAT row is over the 15 readers and the interval of the other
rows is a Wilson interval over the 173 questions, so the two kinds are not comparable in width. The `gpt2` run does not
store the two text rows (computed values in the table above, equal to those shown for the other sizes). Best-layer rows
give the layer.

| Row                              | gpt2                     | gpt2-medium              | gpt2-large               | gpt2-xl                  |
| -------------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| standard                         | 0.168 [0.119, 0.230]     | 0.173 [0.124, 0.237]     | 0.179 [0.129, 0.243]     | 0.173 [0.124, 0.237]     |
| heuristic                        | 0.277 [0.216, 0.348]     | 0.266 [0.206, 0.336]     | 0.358 [0.291, 0.432]     | 0.289 [0.227, 0.361]     |
| text_word                        | not stored               | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     |
| text_char                        | not stored               | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     |
| LAT truthfulqa, validation-layer | 0.649 [0.640, 0.657]     | 0.667 [0.659, 0.676]     | 0.676 [0.666, 0.687]     | 0.699 [0.687, 0.710]     |
| LAT truthfulqa, best-layer       | 0.652 [0.647, 0.659] L9  | 0.688 [0.680, 0.697] L15 | 0.693 [0.685, 0.701] L22 | 0.716 [0.705, 0.726] L28 |
| LAT arc, validation-layer        | 0.383 [0.353, 0.414]     | 0.489 [0.453, 0.521]     | 0.408 [0.377, 0.440]     | 0.432 [0.384, 0.480]     |
| LAT arc, best-layer              | 0.385 [0.344, 0.429] L9  | 0.464 [0.423, 0.502] L18 | 0.411 [0.379, 0.438] L16 | 0.375 [0.319, 0.433] L25 |
| LAT honesty, validation-layer    | 0.270 [0.264, 0.277]     | 0.476 [0.465, 0.487]     | 0.304 [0.296, 0.311]     | 0.371 [0.358, 0.382]     |
| LAT honesty, best-layer          | 0.277 [0.271, 0.284] L12 | 0.492 [0.489, 0.496] L18 | 0.318 [0.313, 0.324] L29 | 0.380 [0.374, 0.387] L25 |

- **The TruthfulQA-source reader does not separate from the text baselines on any size.** At the validation layer the
  reader reaches 0.649, 0.667, 0.676 and 0.699, against `text_word` 0.676 [0.603, 0.742] and `text_char` 0.671 [0.597,
  0.736]. Every one of the four values lies inside both Wilson intervals. The point estimate rises with size, but only
  `gpt2-xl` (0.699) is above the two text point estimates, by 0.023 and 0.028, and the text intervals are 0.14 wide. The
  best-layer rows (0.652, 0.688, 0.693, 0.716) pick the layer on the test questions, are optimistic, and are also inside
  both text intervals.
- **The other two sources stay below the text baselines.** The ARC-trained reader is at 0.383, 0.489, 0.408 and 0.432
  and the honesty-persona reader at 0.270, 0.476, 0.304 and 0.371 at the validation layer, none above 0.489. Neither is
  trained on TruthfulQA answers. `gpt2-medium` is the highest of the four on both sources, with no pattern in size.
- **Random directions.** The highest mean over layers of the random-direction series of the TruthfulQA source is 0.370,
  0.429, 0.432 and 0.425 (`gpt2` to `gpt2-xl`), below the reader on every size. For the honesty source the best-layer
  reader is below the highest random value on `gpt2` (0.277 against 0.316), `gpt2-large` (0.318 against 0.352) and
  `gpt2-xl` (0.380 against 0.408), and for ARC on `gpt2-xl` (0.375 against 0.400); the ARC and honesty readers are not
  read as separating from a random direction there. The shuffled-label series stays at or below 0.249.
- **Standard and heuristic.** The standard log-probability MC1 is 0.168 to 0.179 on the four sizes (chance 0.228) and
  the heuristic is 0.277, 0.266, 0.358 and 0.289 in the order of the columns, without a trend in size.

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

## Template sensitivity of the sentiment reader (P1)

P1 reads the SST-2 sentiment concept through variants of the prompt template, on `gpt2` and the four Qwen3
checkpoints. The tables of this section are for the sentiment task; `gpt2` also has a second run with the truth and
emotion tasks, reported at the end of the section. The paper template is `Consider the amount of sentiment in the
following:`, the sentence, and `The amount of sentiment is`. Every variant is fitted on the same 10 draws of 64 SST-2
training pairs as the paper variant (logistic finder, last token), so a difference between variants belongs to the
wording and not to the draw. All are scored on the same 214 pairs of the SST-2 validation split (the report half; a
disjoint half of 214 pairs is the select half). One layer is used for every variant: the best layer of the paper
reader on the select half, L8 on `gpt2` and L18 on the Qwen3 checkpoints. A variant is not given a layer of its own in
the tables below, unlike the template spread of [P8](#base-against-post-trained-qwen3-p8). The variants:

- **Frame** (what surrounds the sentence): `no_template` (the sentence alone inside the chat or plain tags), `bare`
  (the sentence alone, with neither template nor tags), `no_chat_tags` (the wording without the tags), `neutral_frame`
  (`Consider the following text:` and the sentence, naming no concept), `no_trailing_question` (the text cut after the
  sentence) and `no_trailing_space` (no space after the assistant tag).
- **Wording**: `paraphrase_0` to `paraphrase_2` (three other wordings of the template) and `scaffold_0` to
  `scaffold_2` (the sentence followed by `Conclusion: This movie is`, `Conclusion: This text is` or `Conclusion:
  Overall, it was`).
- **Concept**: `concept_swap_0` to `concept_swap_3` (the word `sentiment` replaced by `furniture`, `geometry`,
  `weather` or `cooking`).

Two kinds were skipped. `alternate` and `framing_swap` do not apply to the sentiment task (it has no alternate wording
and no function framing), on all five models. `no_trailing_space` needs the plain tag format, so it was skipped on the
two post-trained checkpoints, which use the chat template. The comparisons across models below use the 16 variants
every model has; `no_trailing_space` appears only in the per-variant tables, for the three models that have it.

Accuracy and its interval are as in the [reading conventions](#reading-conventions): the mean over the 10 readers,
with a bootstrap interval over the readers that does not include the sampling of the 214 evaluation pairs. `Random` is
the accuracy of a random unit direction oriented by the same sign rule as the reader and `Shuffled` is the reader
scored against labels shuffled within each pair; both are means over the 10 seeds on the items of the variant, at the
same layer. Two cosines appear. `Seed cosine` is the mean cosine between the directions of two different seeds of the
same variant. `Cosine with paper` is the mean over seeds of the cosine between the reader of a variant and the paper
reader fitted on the same seed (the same training pairs), so the two are not on the same footing.

| Model           | Layer | Paper accuracy [95% interval] | Random | Shuffled | Seed cosine |
| --------------- | ----- | ----------------------------- | ------ | -------- | ----------- |
| gpt2            | L8    | 0.765 [0.744, 0.780]          | 0.497  | 0.505    | 0.520       |
| qwen3-0.6b-base | L18   | 0.952 [0.946, 0.958]          | 0.559  | 0.515    | 0.769       |
| qwen3-0.6b      | L18   | 0.860 [0.843, 0.875]          | 0.555  | 0.507    | 0.629       |
| qwen3-1.7b-base | L18   | 0.943 [0.936, 0.950]          | 0.608  | 0.512    | 0.797       |
| qwen3-1.7b      | L18   | 0.880 [0.866, 0.894]          | 0.553  | 0.500    | 0.654       |

The R1 values of the same models are 0.772, 0.947, 0.857, 0.905 and 0.866 (in the order of the table above). The two
runs score different pairs and select the layer differently (R1 on the pairs it reports), and this document does not
attribute the differences, which are largest for `qwen3-1.7b-base` (0.943 here against 0.905).

Lowest and highest variant among the 16, with the random series of that variant in parentheses, and their difference:

| Model           | Lowest variant: accuracy (random)  | Highest variant: accuracy (random) | Span  |
| --------------- | ---------------------------------- | ---------------------------------- | ----- |
| gpt2            | no_trailing_question 0.752 (0.507) | no_chat_tags 0.843 (0.524)         | 0.090 |
| qwen3-0.6b-base | scaffold_1 0.763 (0.556)           | paraphrase_2 0.966 (0.583)         | 0.204 |
| qwen3-0.6b      | no_chat_tags 0.765 (0.477)         | neutral_frame 0.964 (0.595)        | 0.199 |
| qwen3-1.7b-base | concept_swap_1 0.839 (0.538)       | paraphrase_0 0.975 (0.669)         | 0.136 |
| qwen3-1.7b      | concept_swap_2 0.781 (0.566)       | paraphrase_2 0.964 (0.586)         | 0.183 |

- **Spread over variants.** The span is 0.090 on `gpt2` and 0.204, 0.199, 0.136 and 0.183 on `qwen3-0.6b-base`,
  `qwen3-0.6b`, `qwen3-1.7b-base` and `qwen3-1.7b`. The paper template is the lowest or the highest variant on none of
  the five models. A span takes the two extremes of 16 cells, so it is larger than the difference between two variants
  picked in advance; the interval of a single variant is between 0.004 and 0.075 wide (upper minus lower bound) over
  the 80 cells. On `gpt2`, all variants except `bare` (0.802) and `no_chat_tags` (0.843) lie between 0.752 and 0.782,
  around the paper value of 0.765.
- **Variants against the paper variant.** Counting the 15 other variants whose interval lies entirely above (below)
  the interval of the paper variant: 2 (0) on `gpt2`, 2 (10) on `qwen3-0.6b-base`, 5 (5) on `qwen3-0.6b`, 3 (6) on
  `qwen3-1.7b-base` and 8 (4) on `qwen3-1.7b`.
- **Random and shuffled series.** The shuffled series is between 0.471 and 0.521 on all 80 cells. The random series is
  not at 0.5 on the Qwen3 checkpoints: it is between 0.477 and 0.674 there (0.479 to 0.555 on `gpt2`), highest on
  `qwen3-1.7b-base` with `paraphrase_1` (0.674). The reader is above the random series of the same cell by at least
  0.204 on every one of the 80 cells (the smallest margin is `gpt2` with `no_template`, 0.753 against 0.549), so no
  variant here reads at the level of a random direction.

The `bare` variant, the sentence with nothing around it:

| Model           | Bare accuracy | Random | Shuffled | Cosine with paper | Bare seed cosine |
| --------------- | ------------- | ------ | -------- | ----------------- | ---------------- |
| gpt2            | 0.802         | 0.555  | 0.497    | 0.179             | 0.298            |
| qwen3-0.6b-base | 0.852         | 0.523  | 0.497    | 0.109             | 0.347            |
| qwen3-0.6b      | 0.874         | 0.502  | 0.496    | 0.077             | 0.332            |
| qwen3-1.7b-base | 0.908         | 0.518  | 0.492    | 0.137             | 0.414            |
| qwen3-1.7b      | 0.920         | 0.556  | 0.493    | 0.121             | 0.409            |

The paper reader and the bare reader scored on each other's items:

| Model           | Paper reader on bare items | Bare reader on paper items |
| --------------- | -------------------------- | -------------------------- |
| gpt2            | 0.807                      | 0.686                      |
| qwen3-0.6b-base | 0.852                      | 0.764                      |
| qwen3-0.6b      | 0.761                      | 0.638                      |
| qwen3-1.7b-base | 0.906                      | 0.857                      |
| qwen3-1.7b      | 0.886                      | 0.708                      |

- **Bare accuracy.** The bare reader is at 0.802 on `gpt2` and 0.852, 0.874, 0.908 and 0.920 on the Qwen3 checkpoints,
  with random series between 0.502 and 0.556. Its interval is above the paper interval on `gpt2` and `qwen3-1.7b`,
  below it on the two base checkpoints, and overlaps it on `qwen3-0.6b` (0.874 against 0.860).
- **Bare direction.** The bare reader's direction has a cosine with the paper reader's of 0.077 to 0.179 on the five
  models, against seed cosines of 0.520 to 0.797 for the paper reader and 0.298 to 0.414 for the bare reader. The two
  readers are close to orthogonal in the residual stream, although both read above the random series. The paper
  reader, scored on bare items, reaches 0.761 to 0.906, and the bare reader, scored on paper items, 0.638 to 0.857;
  the first is above the second on all five models. The random series of the cross-scored cells is not among the
  committed files.

The concept-swap and wording variants, as ranges over the variants of the kind, with the random series and the cosine
with the paper reader over the same variants:

| Model           | Variants         | Accuracy       | Random         | Cosine with paper |
| --------------- | ---------------- | -------------- | -------------- | ----------------- |
| gpt2            | concept_swap (4) | 0.762 to 0.778 | 0.479 to 0.500 | 0.963 to 0.976    |
| gpt2            | paraphrase (3)   | 0.762 to 0.778 | 0.503 to 0.524 | 0.870 to 0.960    |
| gpt2            | scaffold (3)     | 0.765 to 0.782 | 0.506 to 0.522 | 0.762 to 0.776    |
| qwen3-0.6b-base | concept_swap (4) | 0.765 to 0.793 | 0.502 to 0.537 | 0.369 to 0.507    |
| qwen3-0.6b-base | paraphrase (3)   | 0.944 to 0.966 | 0.566 to 0.583 | 0.783 to 0.866    |
| qwen3-0.6b-base | scaffold (3)     | 0.763 to 0.929 | 0.478 to 0.589 | 0.450 to 0.615    |
| qwen3-0.6b      | concept_swap (4) | 0.806 to 0.841 | 0.497 to 0.543 | 0.700 to 0.741    |
| qwen3-0.6b      | paraphrase (3)   | 0.783 to 0.947 | 0.488 to 0.585 | 0.476 to 0.616    |
| qwen3-0.6b      | scaffold (3)     | 0.823 to 0.897 | 0.540 to 0.569 | 0.515 to 0.578    |
| qwen3-1.7b-base | concept_swap (4) | 0.839 to 0.898 | 0.529 to 0.556 | 0.548 to 0.635    |
| qwen3-1.7b-base | paraphrase (3)   | 0.951 to 0.975 | 0.643 to 0.674 | 0.743 to 0.893    |
| qwen3-1.7b-base | scaffold (3)     | 0.908 to 0.943 | 0.556 to 0.586 | 0.563 to 0.606    |
| qwen3-1.7b      | concept_swap (4) | 0.781 to 0.848 | 0.546 to 0.577 | 0.593 to 0.630    |
| qwen3-1.7b      | paraphrase (3)   | 0.922 to 0.964 | 0.585 to 0.587 | 0.629 to 0.775    |
| qwen3-1.7b      | scaffold (3)     | 0.894 to 0.913 | 0.557 to 0.567 | 0.496 to 0.570    |

- **Concept swaps.** On `gpt2` the four concept swaps read at 0.762 to 0.778 (paper 0.765) and their cosine with the
  paper reader is 0.963 to 0.976. On the Qwen3 checkpoints the point estimate of every concept swap is below the paper
  accuracy (0.765 to 0.793 against 0.952 on `qwen3-0.6b-base`, 0.806 to 0.841 against 0.860 on `qwen3-0.6b`, 0.839 to
  0.898 against 0.943 on `qwen3-1.7b-base`, 0.781 to 0.848 against 0.880 on `qwen3-1.7b`); the interval is entirely
  below the paper interval for 4, 2, 4 and 3 of the 4 swaps in that order. Their cosines with the paper reader are
  0.369 to 0.741.
- **Wording.** Paraphrases read at 0.762 to 0.778 on `gpt2` and between 0.783 and 0.975 on the Qwen3 checkpoints;
  scaffolds read at 0.765 to 0.782 on `gpt2` and between 0.763 and 0.943. The widest wording spreads are the three
  paraphrases of `qwen3-0.6b` (0.783 to 0.947) and the three scaffolds of `qwen3-0.6b-base` (0.763 to 0.929).

Transfer between variants. The paper reader scored on the items of each of the 15 other variants, the reader of each
variant scored on the paper items, and the random series of the 16 variants on their own items:

| Model           | Paper reader on variant items | Variant reader on paper items | Random, 16 variants |
| --------------- | ----------------------------- | ----------------------------- | ------------------- |
| gpt2            | 0.750 to 0.811                | 0.686 to 0.770                | 0.479 to 0.555      |
| qwen3-0.6b-base | 0.725 to 0.969                | 0.764 to 0.962                | 0.478 to 0.589      |
| qwen3-0.6b      | 0.752 to 0.941                | 0.638 to 0.871                | 0.477 to 0.644      |
| qwen3-1.7b-base | 0.889 to 0.975                | 0.857 to 0.956                | 0.518 to 0.674      |
| qwen3-1.7b      | 0.762 to 0.943                | 0.708 to 0.890                | 0.509 to 0.620      |

The lowest and the highest cell of the full 16 by 16 transfer matrix off the diagonal (reader of the first variant
scored on the items of the second):

| Model           | Lowest cell: trained on, scored on | Highest cell: trained on, scored on       |
| --------------- | ---------------------------------- | ----------------------------------------- |
| gpt2            | 0.654: bare, scaffold_2            | 0.869: no_chat_tags, bare                 |
| qwen3-0.6b-base | 0.556: bare, concept_swap_3        | 0.971: no_chat_tags, paraphrase_2         |
| qwen3-0.6b      | 0.564: bare, no_chat_tags          | 0.951: paraphrase_2, neutral_frame        |
| qwen3-1.7b-base | 0.700: bare, concept_swap_2        | 0.975: paraphrase_1, paraphrase_0         |
| qwen3-1.7b      | 0.591: bare, concept_swap_3        | 0.959: no_trailing_question, paraphrase_2 |

- **Range.** The paper reader scored on other variants spans 0.750 to 0.811 on `gpt2`, 0.725 to 0.969 on
  `qwen3-0.6b-base`, 0.752 to 0.941 on `qwen3-0.6b`, 0.889 to 0.975 on `qwen3-1.7b-base` and 0.762 to 0.943 on
  `qwen3-1.7b`. The reader of each variant scored on the paper items spans 0.686 to 0.770, 0.764 to 0.962, 0.638 to
  0.871, 0.857 to 0.956 and 0.708 to 0.890.
- **Lowest cell.** On all five models the lowest off-diagonal cell is a reader fitted on `bare`. The cell is no higher
  than the highest random value of the run on `qwen3-0.6b-base` (0.556 against 0.589), `qwen3-0.6b` (0.564 against
  0.644) and `qwen3-1.7b` (0.591 against 0.620), and it is higher on `gpt2` (0.654 against 0.555) and
  `qwen3-1.7b-base` (0.700 against 0.674). The random series of the cross-scored cells is not among the committed
  files, so this compares the cell with the random series on the diagonal items.

Every variant of every model, as report accuracy with the random and the shuffled series in parentheses (`not run`
marks a kind that was skipped):

| Variant              | gpt2                 | qwen3-0.6b-base      | qwen3-0.6b           |
| -------------------- | -------------------- | -------------------- | -------------------- |
| paper                | 0.765 (0.497, 0.505) | 0.952 (0.559, 0.515) | 0.860 (0.555, 0.507) |
| no_template          | 0.753 (0.549, 0.497) | 0.864 (0.586, 0.493) | 0.935 (0.644, 0.508) |
| bare                 | 0.802 (0.555, 0.497) | 0.852 (0.523, 0.497) | 0.874 (0.502, 0.496) |
| no_chat_tags         | 0.843 (0.524, 0.480) | 0.953 (0.571, 0.490) | 0.765 (0.477, 0.486) |
| neutral_frame        | 0.766 (0.534, 0.484) | 0.863 (0.545, 0.495) | 0.964 (0.595, 0.488) |
| no_trailing_question | 0.752 (0.507, 0.487) | 0.943 (0.531, 0.488) | 0.903 (0.581, 0.499) |
| no_trailing_space    | 0.771 (0.512, 0.486) | 0.721 (0.499, 0.486) | not run              |
| paraphrase_0         | 0.762 (0.503, 0.493) | 0.964 (0.566, 0.500) | 0.783 (0.547, 0.512) |
| paraphrase_1         | 0.765 (0.521, 0.504) | 0.944 (0.569, 0.517) | 0.829 (0.488, 0.513) |
| paraphrase_2         | 0.778 (0.524, 0.504) | 0.966 (0.583, 0.510) | 0.947 (0.585, 0.515) |
| scaffold_0           | 0.782 (0.506, 0.510) | 0.929 (0.589, 0.521) | 0.897 (0.540, 0.504) |
| scaffold_1           | 0.765 (0.509, 0.488) | 0.763 (0.556, 0.471) | 0.865 (0.569, 0.492) |
| scaffold_2           | 0.780 (0.522, 0.508) | 0.856 (0.478, 0.512) | 0.823 (0.550, 0.500) |
| concept_swap_0       | 0.778 (0.494, 0.501) | 0.765 (0.518, 0.504) | 0.838 (0.497, 0.508) |
| concept_swap_1       | 0.762 (0.500, 0.508) | 0.793 (0.507, 0.492) | 0.806 (0.513, 0.512) |
| concept_swap_2       | 0.771 (0.486, 0.511) | 0.775 (0.537, 0.514) | 0.815 (0.543, 0.508) |
| concept_swap_3       | 0.778 (0.479, 0.502) | 0.765 (0.502, 0.492) | 0.841 (0.519, 0.491) |

| Variant              | qwen3-1.7b-base      | qwen3-1.7b           |
| -------------------- | -------------------- | -------------------- |
| paper                | 0.943 (0.608, 0.512) | 0.880 (0.553, 0.500) |
| no_template          | 0.928 (0.536, 0.497) | 0.941 (0.620, 0.503) |
| bare                 | 0.908 (0.518, 0.492) | 0.920 (0.556, 0.493) |
| no_chat_tags         | 0.935 (0.661, 0.492) | 0.831 (0.509, 0.493) |
| neutral_frame        | 0.941 (0.547, 0.495) | 0.922 (0.576, 0.497) |
| no_trailing_question | 0.968 (0.601, 0.487) | 0.935 (0.610, 0.497) |
| no_trailing_space    | 0.906 (0.583, 0.483) | not run              |
| paraphrase_0         | 0.975 (0.669, 0.496) | 0.922 (0.585, 0.497) |
| paraphrase_1         | 0.951 (0.674, 0.515) | 0.939 (0.587, 0.519) |
| paraphrase_2         | 0.972 (0.643, 0.509) | 0.964 (0.586, 0.512) |
| scaffold_0           | 0.943 (0.586, 0.509) | 0.913 (0.562, 0.507) |
| scaffold_1           | 0.932 (0.556, 0.484) | 0.903 (0.557, 0.482) |
| scaffold_2           | 0.908 (0.563, 0.489) | 0.894 (0.567, 0.489) |
| concept_swap_0       | 0.860 (0.529, 0.499) | 0.848 (0.577, 0.497) |
| concept_swap_1       | 0.839 (0.538, 0.490) | 0.793 (0.546, 0.491) |
| concept_swap_2       | 0.878 (0.529, 0.513) | 0.781 (0.566, 0.512) |
| concept_swap_3       | 0.898 (0.556, 0.505) | 0.814 (0.550, 0.506) |

What this does and does not show. On `gpt2` the wording of the template moves the reader little (0.752 to 0.782
outside `bare` and `no_chat_tags`), and the variants without tags score higher (0.843 for `no_chat_tags`, 0.802 for
`bare`). On the Qwen3 checkpoints the accuracy moves by 0.136 to 0.204 across the variants and the paper template is
neither the best nor the worst variant of any model. Each variant changes one aspect of the prompt, but the table does
not say which aspect drives a difference: every cell is one run, the intervals cover the training draw only, and the
214 evaluation pairs are shared by all variants. The near-orthogonality of the bare and paper directions shows that
readers fitted under different prompts differ in direction; it does not show which of them tracks the concept better.
Base and post-trained checkpoints are different models, so the pairs are compared in P8 and the two are not pooled
here.

P1 with the truth and emotion tasks on `gpt2` (run `p1/gpt2/078b9d6482a2`, made after the sentiment run
`p1/gpt2/673af340b77d` with the same settings and the task list as the only difference in the config; its sentiment
rows are identical to the ones above). The tasks are the truth statements of the Geometry of Truth split and the six
emotions of R4, each read against the others. Every variant is read at the one layer of the paper reader on the select
half, and the report half has the number of pairs in parentheses (59 for truth and 36 to 53 for the emotions). Lowest
and highest variant over all the variants of the task, including `framing_swap` for the emotions:

| Task (report pairs) | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant     | Highest variant         | Span  |
| ------------------- | ----- | ----------------------------- | ------ | -------- | ------------------ | ----------------------- | ----- |
| truth (59)          | L9    | 0.698 [0.681, 0.717]          | 0.517  | 0.495    | scaffold_1 0.608   | paraphrase_0 0.708      | 0.100 |
| happiness (38)      | L12   | 0.850 [0.834, 0.868]          | 0.566  | 0.500    | bare 0.811         | no_template 0.884       | 0.074 |
| sadness (44)        | L12   | 0.930 [0.916, 0.941]          | 0.586  | 0.511    | framing_swap 0.414 | neutral_frame 0.957     | 0.543 |
| anger (45)          | L9    | 0.947 [0.938, 0.956]          | 0.622  | 0.467    | framing_swap 0.478 | no_chat_tags 0.967      | 0.489 |
| fear (53)           | L9    | 0.989 [0.975, 1.000]          | 0.562  | 0.509    | framing_swap 0.802 | no_trailing_space 0.998 | 0.196 |
| disgust (39)        | L12   | 1.000 [1.000, 1.000]          | 0.608  | 0.474    | framing_swap 0.818 | paper 1.000             | 0.182 |
| surprise (36)       | L10   | 0.933 [0.922, 0.942]          | 0.586  | 0.472    | framing_swap 0.686 | scaffold_1 0.947        | 0.261 |

- **Truth.** The paper template reads at 0.698 [0.681, 0.717] on 59 pairs (random 0.517, shuffled 0.495). The 16 other
  variants range from 0.608 to 0.708; the interval of 8 of them lies entirely below the paper interval and none lies
  above it. `bare` (0.675) and `no_chat_tags` (0.676) are below the paper accuracy, unlike the sentiment task, and the
  cosine of the 4 concept swaps with the paper reader is 0.943 to 0.950. The highest random value among the truth
  variants is 0.568 (`no_trailing_question`), so the smallest margin of a variant over its random series is 0.063. The
  test set is 59 pairs and the intervals cover the training draw only.
- **Emotions.** Outside `framing_swap` the variants of each emotion range from 0.811 to 0.884 for happiness, 0.905 to
0.957 for sadness, 0.904 to 0.967 for anger, 0.923 to 0.998 for fear, 0.982 to 1.000 for disgust and 0.908 to 0.947 for
surprise, around the paper values of 0.850, 0.930, 0.947, 0.989, 1.000 and 0.933. The cosine of the concept swaps with
the paper reader is 0.965 to 0.989, and `bare` reads at 0.811, 0.905, 0.964, 0.936, 0.982 and 0.928 with a cosine with
the paper reader of 0.620, 0.600, 0.318, 0.324, 0.618 and 0.413. The random series of the emotion variants is at most
0.658.
- **The function framing is the exception.** `framing_swap` is the function template of an emotion, trained at the
  response position and scored on its own items:

| Emotion   | Function framing accuracy [95% interval] | Random | Shuffled | Cosine with paper | Paper accuracy |
| --------- | ---------------------------------------- | ------ | -------- | ----------------- | -------------- |
| happiness | 0.829 [0.805, 0.853]                     | 0.508  | 0.495    | 0.372             | 0.850          |
| sadness   | 0.414 [0.384, 0.445]                     | 0.527  | 0.486    | 0.049             | 0.930          |
| anger     | 0.478 [0.453, 0.502]                     | 0.449  | 0.509    | 0.075             | 0.947          |
| fear      | 0.802 [0.781, 0.825]                     | 0.496  | 0.530    | 0.174             | 0.989          |
| disgust   | 0.818 [0.792, 0.838]                     | 0.585  | 0.462    | 0.202             | 1.000          |
| surprise  | 0.686 [0.667, 0.708]                     | 0.497  | 0.519    | 0.025             | 0.933          |

  Its direction is nearly orthogonal to the paper direction (cosine 0.025 to 0.372). Sadness (0.414 [0.384, 0.445]) is
  below chance with an interval entirely below 0.5, and anger (0.478 [0.453, 0.502]) is at chance with an interval that
  reaches 0.502; happiness (0.829), fear (0.802), disgust (0.818) and surprise (0.686) are above their random series.
  Below chance on a held-out set means the reader ranks the pairs the wrong way round, which for sadness and anger is
  consistent with a direction that does not carry the emotion under that framing; this is not tested further here.
  This is one run on 36 to 53 pairs per emotion, and the intervals cover the training draw only.

## Position, sample size and direction finder on gpt2 (P2, P3, P4)

These three runs vary one setting of the reading protocol on `gpt2` (logistic finder unless stated, 64 training pairs,
10 seeds, the intervals of the [reading conventions](#reading-conventions)). Each is one run: `p2/gpt2/ee099a704022`,
`p3/gpt2/676c31488397` and `p4/gpt2/8c4a73d1e9db`, all made at `ca2546c`.

### Token position and a misleading sentence (P2)

P2 reads sentiment (214 report pairs), truth (59) and utility (256) at three token positions: `last` (the final token,
as everywhere else), `concept` (the token of the concept word in the prompt) and `mean` (the mean over the tokens). The
decoy is a sentence added to the prompt that states a low or a high amount of the concept ("I think the amount of the
concept in the scenario is low, but I'm curious to hear what you think"). A `random` decoy draws low or high per item
independently of the label; an `against` decoy gives label 1 the low value and label 0 the high value, so that it always
contradicts the label (the `against` decoy is an addition of this study; the random one is the paper's). The reader is
trained on the paper template at each position and scored on the original prompts and on the prompts with the decoy.
The zero-shot rows score the model's own verbalized level for the same scenarios, with no reader. The layer is the one
chosen on the original prompts.

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L8    | 0.765 [0.744, 0.780] | 0.763 [0.750, 0.774] | 0.761 [0.746, 0.773] |
| sentiment | concept   | L10   | 0.829 [0.817, 0.840] | 0.826 [0.815, 0.838] | 0.783 [0.768, 0.797] |
| sentiment | mean      | L9    | 0.823 [0.810, 0.837] | 0.834 [0.820, 0.848] | 0.834 [0.820, 0.849] |
| sentiment | zero-shot | -     | 0.565 [0.495, 0.636] | 0.551 [0.486, 0.617] | 0.150 [0.103, 0.201] |
| truth     | last      | L9    | 0.698 [0.681, 0.717] | 0.663 [0.649, 0.680] | 0.673 [0.654, 0.693] |
| truth     | concept   | L10   | 0.676 [0.656, 0.698] | 0.651 [0.617, 0.678] | 0.578 [0.542, 0.615] |
| truth     | mean      | L8    | 0.671 [0.649, 0.692] | 0.685 [0.654, 0.714] | 0.685 [0.654, 0.714] |
| truth     | zero-shot | -     | 0.542 [0.424, 0.661] | 0.475 [0.339, 0.594] | 0.085 [0.017, 0.169] |
| utility   | last      | L10   | 0.644 [0.628, 0.663] | 0.620 [0.604, 0.637] | 0.133 [0.087, 0.181] |
| utility   | concept   | L8    | 0.650 [0.636, 0.667] | 0.654 [0.636, 0.671] | 0.634 [0.612, 0.657] |
| utility   | mean      | L8    | 0.650 [0.629, 0.668] | 0.647 [0.628, 0.664] | 0.653 [0.635, 0.671] |
| utility   | zero-shot | -     | 0.586 [0.527, 0.648] | 0.570 [0.508, 0.625] | 0.023 [0.008, 0.043] |

- **Position.** On the original prompts the three positions read within 0.07 of each other: sentiment 0.765, 0.829 and
  0.823, truth 0.698, 0.676 and 0.671, utility 0.644, 0.650 and 0.650. Sentiment reads higher at `concept` and `mean`
  than at `last`; truth and utility do not.
- **Random decoy.** The reader moves by at most 0.035 at `last` (truth: 0.698 to 0.663) and at most 0.025 at the other
  positions; the intervals of the original and random columns overlap everywhere except truth at `last`.
- **Against decoy: the effect depends on the task and the position.** At `last` sentiment (0.761) and truth (0.673) are
  within 0.03 of the original, but utility drops to 0.133 [0.087, 0.181], far below chance: the reader ranks the pairs
  the wrong way round when the decoy contradicts the label. At `concept` the drops are 0.045 (sentiment, 0.783) and
  0.098 (truth, 0.578) and utility 0.634 is within 0.016. At `mean` no task moves by more than 0.014 (all three upward)
  and the interval of the decoy column overlaps the original in all three. Utility at `mean` is 0.650 and 0.653, so the
  collapse at `last` belongs to that position and that task; the run does not say why.
- **Zero-shot.** The model's own answer drops from 0.565, 0.542 and 0.586 to 0.150, 0.085 and 0.023 under the `against`
decoy, below chance, and is at 0.551, 0.475 and 0.570 under the random decoy. The reader at `mean` moves by at most
0.014 under the same decoys, so the decoy that turns the model's verbalized answer around does not turn that reader.

### Stimulus-set size (P3)

P3 fits the reader on 2 to 512 pairs, 20 draws per size, and reports the mean accuracy. In parentheses is the failure
rate: the share of the 20 draws whose accuracy is below 0.6. The truth set has 172 pairs, so its sizes stop at 128 (the
256 and 512 cells are skipped). `truth_cities` and `truth_common` fit the reader on those subsets of the truth
statements and score it on the truth target; the intervals are over the draws.

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.526 (1.00) | 0.566 (0.80) | 0.583 (0.65) | 0.646 (0.20) | 0.704 (0.00) | 0.758 (0.00) | 0.790 (0.00) | 0.820 (0.00) | 0.844 (0.00) |
| truth -> truth         | 0.510 (0.90) | 0.537 (0.85) | 0.507 (0.90) | 0.525 (0.95) | 0.551 (0.80) | 0.564 (0.70) | 0.642 (0.25) | -            | -            |
| truth_cities -> truth  | 0.536 (0.80) | 0.518 (0.85) | 0.530 (0.85) | 0.579 (0.70) | 0.584 (0.70) | 0.606 (0.60) | 0.633 (0.05) | 0.638 (0.00) | -            |
| truth_common -> truth  | 0.512 (0.90) | 0.492 (1.00) | 0.535 (0.85) | 0.539 (0.95) | 0.557 (0.80) | 0.565 (0.80) | 0.592 (0.60) | 0.619 (0.30) | 0.636 (0.20) |
| utility -> utility     | 0.518 (0.80) | 0.500 (0.90) | 0.564 (0.65) | 0.581 (0.50) | 0.609 (0.30) | 0.654 (0.00) | 0.657 (0.05) | 0.660 (0.00) | 0.669 (0.00) |

The accuracy of a reader fitted on all the data of the source (the reference) is 0.883 for sentiment (2,048 pairs),
0.644 for truth (172), 0.661 for `truth_cities` (452), 0.678 for `truth_common` (1,296) and 0.711 for utility (2,048).
The shuffled-label series is between 0.46 and 0.53 at every size.

- **Sentiment** reaches 0.704 at 32 pairs with no failing draw and still rises at 512 (0.844 [0.837, 0.851]), below the
  full-data reference of 0.883. **Utility** has no failing draw from 64 pairs (0.654) and gains 0.015 between 64 and 512
  (0.669 [0.661, 0.676]), against 0.711 for the full data.
- **Truth** is the slow case. The mean accuracy is 0.564 at 64 pairs (failure 0.70) and 0.642 [0.619, 0.662] at 128
  (failure 0.25), at its reference of 0.644 but on a source that is exhausted at 172 pairs; it is still rising at 128.
  The cities and common subsets reach 0.638 (256 pairs) and 0.636 (512) without leveling off, and with 0.00 and 0.20
  failing draws. The data do not locate a saturation point for truth.
- The stored saturation size is the smallest size whose mean accuracy is within 0.01 of the mean at the largest size:
  512 for sentiment, 128 for truth, 256 for utility (bootstrap range 64 to 512), 128 for `truth_cities` (range 64 to
  256) and 512 for `truth_common` (range 256 to 512). For truth and `truth_cities` the largest size is the end of a
  small set (128 and 256 pairs), so their saturation sizes are limits of the set and not measured saturation.

### Direction finder, pairing and preprocessing (P4)

P4 compares six finders on the same pairs: `pca_diff` (the paper's finder and the P4 reference), `pca_raw`, `mean_diff`,
`logistic`, `kmeans` and a `random` direction. The pairings are `matched` (a label-1 member of a group with another
member of the same group, available for utility only), `label_contrastive` (a label-1 with a label-0 stimulus anywhere
in the examples) and `random` (any two stimuli, whatever their labels). The preprocessing is `center`, `unit` or `none`.
Report-split accuracy at the layer chosen on the select split, with the interval over the 10 seeds; shuffled-label
accuracy is between 0.45 and 0.52. Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.512 [0.497, 0.527] | 0.496 [0.479, 0.512] | 0.667 [0.634, 0.695] | 0.765 [0.744, 0.781] | 0.495 [0.483, 0.507] | 0.543 [0.514, 0.572] |
| truth (label_contrastive)     | 0.481 [0.463, 0.502] | 0.475 [0.461, 0.493] | 0.537 [0.524, 0.553] | 0.698 [0.681, 0.717] | 0.478 [0.464, 0.493] | 0.510 [0.486, 0.532] |
| utility (matched)             | 0.551 [0.502, 0.585] | 0.555 [0.522, 0.573] | 0.627 [0.609, 0.643] | 0.644 [0.628, 0.663] | 0.557 [0.524, 0.577] | 0.526 [0.495, 0.555] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.512    | 0.496   | 0.667     | 0.765    | 0.495  | 0.543  |
| sentiment | random            | 0.482    | 0.497   | 0.689     | 0.742    | 0.478  | 0.547  |
| truth     | label_contrastive | 0.481    | 0.475   | 0.537     | 0.698    | 0.478  | 0.510  |
| truth     | random            | 0.480    | 0.498   | 0.642     | 0.669    | 0.490  | 0.510  |
| utility   | matched           | 0.551    | 0.555   | 0.627     | 0.644    | 0.557  | 0.526  |
| utility   | label_contrastive | 0.518    | 0.543   | 0.590     | 0.624    | 0.547  | 0.505  |
| utility   | random            | 0.513    | 0.528   | 0.598     | 0.617    | 0.559  | 0.506  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.512 [0.497, 0.527] | 0.503 [0.484, 0.519] | 0.496 [0.479, 0.512] |
| sentiment | pca_raw   | 0.496 [0.479, 0.512] | 0.496 [0.484, 0.507] | 0.523 [0.511, 0.529] |
| sentiment | mean_diff | 0.667 [0.634, 0.695] | 0.678 [0.649, 0.706] | 0.667 [0.638, 0.694] |
| sentiment | logistic  | 0.765 [0.744, 0.781] | 0.746 [0.729, 0.760] | 0.669 [0.636, 0.696] |
| truth     | pca_diff  | 0.481 [0.463, 0.502] | 0.475 [0.458, 0.495] | 0.481 [0.464, 0.502] |
| truth     | pca_raw   | 0.475 [0.461, 0.493] | 0.473 [0.461, 0.488] | 0.627 [0.627, 0.627] |
| truth     | mean_diff | 0.537 [0.524, 0.553] | 0.536 [0.522, 0.551] | 0.537 [0.522, 0.553] |
| truth     | logistic  | 0.698 [0.681, 0.717] | 0.658 [0.634, 0.678] | 0.537 [0.524, 0.553] |
| utility   | pca_diff  | 0.551 [0.502, 0.585] | 0.550 [0.507, 0.582] | 0.550 [0.500, 0.584] |
| utility   | pca_raw   | 0.555 [0.522, 0.573] | 0.555 [0.521, 0.574] | 0.533 [0.529, 0.537] |
| utility   | mean_diff | 0.627 [0.609, 0.643] | 0.623 [0.602, 0.642] | 0.627 [0.609, 0.643] |
| utility   | logistic  | 0.644 [0.628, 0.663] | 0.644 [0.630, 0.660] | 0.627 [0.609, 0.644] |

- **The paper finder does not read here.** `pca_diff` is at 0.512 for sentiment and 0.481 for truth, no better than the
  random direction (0.543 and 0.510), and 0.551 for utility against 0.526. `pca_raw` and `kmeans` are the same. The
  first principal component explains 20 percent (sentiment), 26 percent (truth) and 74 percent (utility, matched) of
  the variance of the pair differences, and the component chosen by cross-validation among the first eight
  reads 0.552 [0.517, 0.595] (sentiment), 0.485 (truth) and 0.506 (utility, matched).
- **The supervised finders read.** `logistic` is the highest on all three tasks (0.765, 0.698, 0.644) and `mean_diff` is
  second (0.667, 0.537, 0.627). On utility the two are within 0.017 of each other and their intervals overlap; on
  truth `mean_diff` (0.537) is only just above the random direction.
- **Pairing.** The best pairing for `logistic` is `label_contrastive` for sentiment (0.765 against 0.742 for `random`)
  and truth (0.698 against 0.669), and `matched` for utility (0.644 against 0.624 for `label_contrastive` and 0.617).
  The differences between pairings are at most 0.03 for `logistic`; for `mean_diff` on truth the `random` pairing is
  higher (0.642 against 0.537).
- **Preprocessing.** `logistic` loses with less preprocessing on sentiment (0.765, 0.746, 0.669) and truth (0.698,
  0.658, 0.537) and not on utility (0.644, 0.644, 0.627). The `pca_raw` truth cell without preprocessing, 0.627, has an
  interval of zero width: the ten readers give the same value, so it is not read as a reader result.
- These are one run on `gpt2`; the finder ranking is not shown to hold on the other sizes.

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

## Sentiment steering on Qwen3 (C1)

The same C1 procedure as above (100 neutral prompts, the primary and the check judge, the coefficient relative to the
mean residual norm at the steered layer, perplexity of held-out neutral text under the steered model) on the base and
post-trained checkpoints of Qwen3 0.6B and 1.7B. The post-trained checkpoints use the chat template with thinking
disabled and the base checkpoints plain tags. Each steering arm has three readers, so the tables give the mean over
the three readers with the range in parentheses, as in the GPT-2 section (single-run arms have no range). Base and
post-trained checkpoints are different models: the columns are read within a pair and never pooled. All four steer at
their last block (L28) and over a window of the last three blocks, and read at L18.

| Model           | Reading layer | Steer layer | Window     | Judge agreement |
| --------------- | ------------- | ----------- | ---------- | --------------- |
| qwen3-0.6b-base | L18           | L28         | 26, 27, 28 | 1.000           |
| qwen3-0.6b      | L18           | L28         | 26, 27, 28 | 1.000           |
| qwen3-1.7b-base | L18           | L28         | 26, 27, 28 | 0.833           |
| qwen3-1.7b      | L18           | L28         | 26, 27, 28 | 1.000           |

Primary-judge score:

| Arm          | qwen3-0.6b-base     | qwen3-0.6b          | qwen3-1.7b-base     | qwen3-1.7b          |
| ------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| none         | 0.62                | 0.59                | 0.66                | 0.62                |
| linear +0.1  | 0.89 (0.84 to 0.93) | 0.76 (0.71 to 0.86) | 0.79 (0.79 to 0.80) | 0.74 (0.71 to 0.78) |
| linear +0.2  | 0.96 (0.90 to 0.99) | 0.84 (0.71 to 0.92) | 0.87 (0.80 to 0.93) | 0.76 (0.63 to 0.92) |
| linear +0.3  | 0.98 (0.94 to 1.00) | 0.90 (0.76 to 0.99) | 0.92 (0.82 to 0.98) | 0.77 (0.67 to 0.95) |
| linear -0.1  | 0.22 (0.21 to 0.24) | 0.30 (0.23 to 0.37) | 0.26 (0.16 to 0.32) | 0.37 (0.35 to 0.38) |
| linear -0.2  | 0.11 (0.05 to 0.21) | 0.12 (0.04 to 0.19) | 0.24 (0.12 to 0.31) | 0.08 (0.01 to 0.17) |
| linear -0.3  | 0.09 (0.01 to 0.17) | 0.08 (0.02 to 0.17) | 0.42 (0.24 to 0.54) | 0.05 (0.01 to 0.11) |
| prompting +1 | 0.91                | 0.83                | 0.87                | 0.89                |
| prompting +2 | 0.90                | 0.85                | 0.90                | 0.90                |
| prompting -1 | 0.41                | 0.54                | 0.40                | 0.48                |
| prompting -2 | 0.39                | 0.53                | 0.37                | 0.48                |

Check-judge score (`score_check`; the unsteered scores of the two judges are not on one scale, so arms are compared
within a judge):

| Arm          | qwen3-0.6b-base     | qwen3-0.6b          | qwen3-1.7b-base     | qwen3-1.7b          |
| ------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| none         | 0.42                | 0.44                | 0.46                | 0.50                |
| linear +0.1  | 0.81 (0.72 to 0.87) | 0.67 (0.61 to 0.75) | 0.69 (0.65 to 0.73) | 0.67 (0.63 to 0.72) |
| linear +0.2  | 0.89 (0.72 to 0.98) | 0.74 (0.62 to 0.81) | 0.80 (0.70 to 0.86) | 0.72 (0.60 to 0.90) |
| linear +0.3  | 0.87 (0.64 to 0.99) | 0.75 (0.68 to 0.87) | 0.85 (0.70 to 0.92) | 0.65 (0.46 to 0.96) |
| linear -0.1  | 0.11 (0.09 to 0.13) | 0.18 (0.13 to 0.23) | 0.14 (0.09 to 0.18) | 0.25 (0.23 to 0.26) |
| linear -0.2  | 0.03 (0.02 to 0.03) | 0.15 (0.08 to 0.23) | 0.04 (0.03 to 0.05) | 0.09 (0.02 to 0.15) |
| linear -0.3  | 0.04 (0.01 to 0.06) | 0.18 (0.07 to 0.24) | 0.03 (0.02 to 0.04) | 0.04 (0.01 to 0.06) |
| prompting +1 | 0.86                | 0.81                | 0.84                | 0.80                |
| prompting +2 | 0.90                | 0.81                | 0.87                | 0.82                |
| prompting -1 | 0.34                | 0.36                | 0.37                | 0.34                |
| prompting -2 | 0.36                | 0.33                | 0.36                | 0.33                |

Perplexity of neutral text (the prompting arms have no steered model, so they have no perplexity):

| Arm         | qwen3-0.6b-base        | qwen3-0.6b              | qwen3-1.7b-base        | qwen3-1.7b             |
| ----------- | ---------------------- | ----------------------- | ---------------------- | ---------------------- |
| none        | 26.9                   | 47.6                    | 18.6                   | 33.7                   |
| linear +0.1 | 34.9 (34.1 to 35.9)    | 62.8 (61.7 to 64.2)     | 22.9 (22.6 to 23.1)    | 37.6 (35.7 to 39.2)    |
| linear +0.2 | 76.3 (74.2 to 80.1)    | 128.2 (122.7 to 131.1)  | 39.6 (39.1 to 40.2)    | 57.4 (51.6 to 63.2)    |
| linear +0.3 | 283.1 (205.1 to 353.6) | 350.4 (335.4 to 367.5)  | 93.3 (83.9 to 99.0)    | 130.6 (112.2 to 142.7) |
| linear -0.1 | 36.7 (35.5 to 37.4)    | 61.5 (58.5 to 63.8)     | 25.3 (24.6 to 25.8)    | 40.6 (39.5 to 42.2)    |
| linear -0.2 | 106.1 (97.0 to 113.3)  | 169.1 (126.5 to 217.1)  | 86.9 (74.4 to 103.3)   | 94.7 (93.4 to 95.8)    |
| linear -0.3 | 461.2 (437.8 to 503.3) | 836.2 (402.0 to 1457.4) | 646.2 (467.5 to 888.9) | 546.1 (533.4 to 571.2) |

Distinct-1 of the generated text (distinct unigrams over all unigrams; lower means more repetition):

| Arm          | qwen3-0.6b-base        | qwen3-0.6b             | qwen3-1.7b-base        | qwen3-1.7b             |
| ------------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| none         | 0.274                  | 0.213                  | 0.271                  | 0.222                  |
| linear +0.1  | 0.274 (0.252 to 0.299) | 0.203 (0.190 to 0.226) | 0.266 (0.252 to 0.281) | 0.215 (0.207 to 0.229) |
| linear +0.2  | 0.126 (0.061 to 0.238) | 0.123 (0.111 to 0.129) | 0.157 (0.091 to 0.216) | 0.152 (0.137 to 0.173) |
| linear +0.3  | 0.070 (0.018 to 0.170) | 0.060 (0.048 to 0.066) | 0.068 (0.028 to 0.095) | 0.061 (0.056 to 0.067) |
| linear -0.1  | 0.225 (0.196 to 0.259) | 0.183 (0.147 to 0.206) | 0.182 (0.147 to 0.219) | 0.181 (0.181 to 0.183) |
| linear -0.2  | 0.077 (0.065 to 0.091) | 0.095 (0.027 to 0.143) | 0.022 (0.012 to 0.032) | 0.052 (0.031 to 0.078) |
| linear -0.3  | 0.044 (0.027 to 0.054) | 0.057 (0.009 to 0.101) | 0.009 (0.006 to 0.013) | 0.015 (0.006 to 0.028) |
| prompting +1 | 0.223                  | 0.188                  | 0.251                  | 0.193                  |
| prompting +2 | 0.214                  | 0.185                  | 0.235                  | 0.184                  |
| prompting -1 | 0.216                  | 0.172                  | 0.232                  | 0.184                  |
| prompting -2 | 0.210                  | 0.174                  | 0.208                  | 0.163                  |

Reading the scores next to the cost. A steered score is not read as steering success where the text it scores has
degenerated, so this section marks where that happens. As a descriptive rule chosen here, and not a threshold of the
code, a coefficient is called fluent for a checkpoint when the mean perplexity is at most 1.5 times the unsteered one
and the mean distinct-1 is at least 0.15. The coefficients +0.1 and -0.1 are fluent on all four checkpoints
(perplexity at most 1.37 times the unsteered one, distinct-1 at least 0.181); +0.2, +0.3, -0.2 and -0.3 are not fluent
on any of them. The rule is applied to the mean over the three steering readers; at -0.1 single readers fall just
below the distinct-1 cutoff (0.147 for a reader of `qwen3-0.6b` and of `qwen3-1.7b-base`).

- **Direction at the fluent coefficients.** At these coefficients (perplexity at most 1.37 times the unsteered one,
  distinct-1 at least 0.181) the score on the primary judge goes from 0.62, 0.59, 0.66 and 0.62 without
  steering (in the column order of the tables) to 0.89, 0.76, 0.79 and 0.74 at +0.1 and to 0.22, 0.30, 0.26 and 0.37
  at -0.1. The check judge agrees in direction: 0.42, 0.44, 0.46 and 0.50 without steering, 0.81, 0.67, 0.69 and 0.67
  at +0.1, and 0.11, 0.18, 0.14 and 0.25 at -0.1.
- **Against prompting at the fluent coefficients.** At +0.1 (same cost bounds as above) the steered score is below
  both positive prompting arms on all four checkpoints, on both judges (primary: 0.89 against 0.91 and 0.90, 0.76
  against 0.83 and 0.85, 0.79 against 0.87 and 0.90, 0.74 against 0.89 and 0.90). The whole range over readers is
  below both arms on the two 1.7B checkpoints, and on the two 0.6B checkpoints the range (0.84 to 0.93 and 0.71 to
  0.86) reaches the prompting values. At -0.1 the steered score is below both negative prompting arms on all four
  checkpoints and the whole range over readers is below them (0.22 against 0.41 and 0.39, 0.30 against 0.54 and 0.53,
  0.26 against 0.40 and 0.37, 0.37 against 0.48 and 0.48). The prompting arms are single runs and have no range; their
  distinct-1 is between 0.163 and 0.251, in the same band as the steered text at the fluent coefficients (0.181 to
  0.274). On GPT-2 the steered score at +0.1 was above or level with the prompting arms on every size, so the Qwen3
  checkpoints do not repeat that.
- **Within a pair.** At +0.1 the base checkpoint scores above its post-trained pair (0.89 against 0.76 at 0.6B, 0.79
  against 0.74 at 1.7B) and at -0.1 below it (0.22 against 0.30, 0.26 against 0.37). The ranges over readers overlap
  in the 0.6B pair at both coefficients and do not in the 1.7B pair.
- **Beyond the fluent coefficients the text degrades.** At +0.2 the perplexity is 2.84, 2.69, 2.13 and 1.70 times the
  unsteered one, so no checkpoint passes the perplexity cutoff. The distinct-1 is 0.126, 0.123, 0.157 and 0.152
  against 0.274, 0.213, 0.271 and 0.222 without steering: it fails the cutoff on the two 0.6B checkpoints and stays
  above it on the two 1.7B ones. At +0.3 the perplexity is 283.1, 350.4, 93.3 and 130.6 (10.5, 7.4, 5.0 and 3.9 times
  the unsteered one) and the distinct-1 is between 0.060 and 0.070. At -0.2 the perplexity is 106.1, 169.1, 86.9 and
  94.7 with a distinct-1 of 0.077, 0.095, 0.022 and 0.052. At -0.3 the perplexity is 461.2, 836.2, 646.2 and 546.1 and
  the distinct-1 falls to 0.044, 0.057, 0.009 and 0.015. The scores at these coefficients, such as 0.98 at +0.3 on
  `qwen3-0.6b-base` (perplexity 283.1, distinct-1 0.070) and 0.05 at -0.3 on `qwen3-1.7b` (perplexity 546.1,
  distinct-1 0.015), are the scores of repetitive text, and are not read as steering that works.
- **The judges disagree where the text is least readable.** On `qwen3-1.7b-base` the primary-judge mean at -0.3 rises
  to 0.42 (range 0.24 to 0.54) from 0.24 at -0.2, while the check judge gives 0.03, with a perplexity of 646.2 against
  18.6 unsteered and a distinct-1 of 0.009. That checkpoint also has the only judge agreement below 1 (0.833, 5 of the
  6 swept layers with the same slope sign under both judges).
- **Saturation on `qwen3-1.7b`.** The mean score is 0.74, 0.76 and 0.77 at +0.1, +0.2 and +0.3 (perplexity 37.6, 57.4
  and 130.6, distinct-1 0.215, 0.152 and 0.061) while the readers range from 0.63 to 0.92 at +0.2 and from 0.67 to
  0.95 at +0.3, and the check judge gives 0.65 (0.46 to 0.96) at +0.3. The mean is almost flat over the coefficients
  where the text degenerates, and the readers differ.
- **No random control.** As for GPT-2, C1 has no control with a random direction of the same norm, so these tables
  show the effect of the sentiment vector at each coefficient but not how it compares with an arbitrary edit of that
  size.

## Other steering experiments on gpt2 (M1, C2, P7, LR1, C4, C5)

Six further runs on `gpt2` that steer or edit the model along a reading direction, all made at `ca2546c` with a clean
tree (C5 on `gpt2-xl`). **None of these runs stores the perplexity or the distinct-1 of the generated text.** The C1
sections show that a high steered score at a large coefficient can come with degenerate text: on `gpt2` the C1
perplexity is 139.3 at +0.2 and 438.0 at +0.3 against 53.6 without steering, and the distinct-1 at +0.3 is between 0.043
and 0.099 across the four sizes. These runs use coefficients of the same relative range (up to 0.3), and nothing in
their files shows whether the steered text is fluent. No score in this section is called fluent or read as steering that
works. Only M1 has a random-direction arm; C2, P7, LR1, C4 and C5 do not.

### Six direction finders on utility (M1)

M1 reads utility (ETHICS) with six finders and compares them on held-out reading, on steering and on erasure (run
`m1/gpt2/4e394ed1a72c`; 10 readers per finder, the first 3 used for steering, 250 prompts, one sample per prompt). The
steering score is a win rate: the share of (prompt, reader) pairs, 750 per cell, in which the completion steered with +s
is scored higher by the primary sentiment judge than the completion steered with -s on the same sampling seed, so 0.5
means no effect. All finders steer at the same control layers (5, 7 and 8), so depth is not a confound. The prompts are
shared by all finders and readers, so the Wilson intervals below are narrower than the variation between prompt draws.
Test accuracy is on 4,808 pairs, at the layer of each finder (the `correlation` task of the run).

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.544 [0.521, 0.557] | L9    | 0.595 [0.559, 0.629] | 0.637 [0.602, 0.671] | 0.713 [0.680, 0.745] | 0.655 [0.620, 0.688] |
| pca_raw   | 0.538 [0.515, 0.551] | L10   | 0.533 [0.498, 0.569] | 0.607 [0.571, 0.641] | 0.672 [0.638, 0.705] | 0.629 [0.594, 0.663] |
| mean_diff | 0.627 [0.614, 0.640] | L12   | 0.721 [0.688, 0.752] | 0.872 [0.846, 0.894] | 0.903 [0.879, 0.922] | 0.869 [0.843, 0.891] |
| logistic  | 0.655 [0.644, 0.667] | L12   | 0.771 [0.739, 0.799] | 0.880 [0.855, 0.901] | 0.925 [0.904, 0.942] | 0.916 [0.894, 0.934] |
| kmeans    | 0.535 [0.513, 0.548] | L10   | 0.497 [0.462, 0.533] | 0.517 [0.482, 0.553] | 0.592 [0.556, 0.627] | 0.543 [0.507, 0.578] |
| random    | 0.533 [0.509, 0.558] | L10   | 0.491 [0.455, 0.526] | 0.456 [0.421, 0.492] | 0.432 [0.397, 0.468] | 0.413 [0.379, 0.449] |

- **Reading and steering rank the finders alike, with one exception.** `logistic` (0.655 test accuracy, 0.925 win rate
  at 0.3) and `mean_diff` (0.627, 0.903) steer most. `pca_raw` (0.672) and `kmeans` (0.592) steer less, and `random` is
  at or below 0.5 at every strength and falls with it (0.491, 0.456, 0.432). The check judge gives the same ranking
  (`logistic` 0.916, `mean_diff` 0.869, `random` 0.413 at 0.3); the share of swept settings with the same dose sign
  under both judges is 0.889.
- **`pca_diff` steers more than it reads.** It reads at 0.544, level with the random direction (0.533), and steers at
  0.713 at 0.3, above the random direction by 0.281. Reading accuracy and steering do not track each other one for one
  for the paper's finder.
- **The random direction is below 0.5.** At 0.3 its interval [0.397, 0.468] excludes 0.5, so a random edit of the same
  relative size moves the completions the other way on average, and a finder's win rate is not read against 0.5 alone.
  The run does not say why.
- **No fluency measure is stored**, so no win rate here is a measure of steering that keeps the text readable. The
  share of empty completions over all generations is 0.00007.

Erasure. Projecting a finder's direction out of the activations and fitting a fresh logistic probe on the stored
activations of the control layers gives a probe accuracy of 0.628 [0.571, 0.696] before. After the projection it is
0.500 for `mean_diff`, 0.563 [0.512, 0.608] for `logistic`, and 0.635, 0.630, 0.633 and 0.627 for `pca_diff`, `pca_raw`,
`kmeans` and `random` (a fit that stops at zero counts as chance in this measure). The behavioral gap (the score after
the more pleasant scenario of a pair minus after the less pleasant one) is 0.042 [-0.082, 0.162] without a projection
and between 0.039 and 0.118 under the projection of each finder (`logistic` [-0.000, 0.082], `random` [0.109, 0.124]).
No finder lowers it by more than 0.003 and all the intervals overlap the unprojected one. The two operationalizations of
erasure disagree: the probe is erased for two finders and the behavioral gap for none.

### Emotion steering (C2)

C2 steers `gpt2` along the logistic reader of each of the six R4 emotions and measures, with the primary emotion
judge, the probability of every emotion in the completions (run `c2/gpt2/85c9a670bf17`; reading layer L10, steered over
layers 9, 10 and 11, relative strength +0.3, 100 prompts, 2 samples per prompt, 3 steering seeds). The range is the
minimum and maximum over the three steering seeds; the change is the own-emotion probability at +0.3 minus the
unsteered one, read from the change matrix of the run.

| Emotion   | Primary judge: no steering | Primary judge: at +0.3 [range over seeds] | Primary judge: change | Check judge: no steering | Check judge: at +0.3 | Check judge: change | Largest off-diagonal change (primary) |
| --------- | -------------------------- | ----------------------------------------- | --------------------- | ------------------------ | -------------------- | ------------------- | ------------------------------------- |
| happiness | 0.126                      | 0.686 [0.593, 0.750]                      | +0.560                | 0.347                    | 0.889                | +0.542              | surprise +0.010                       |
| sadness   | 0.084                      | 0.403 [0.382, 0.441]                      | +0.319                | 0.166                    | 0.424                | +0.258              | disgust -0.005                        |
| anger     | 0.056                      | 0.090 [0.077, 0.102]                      | +0.034                | 0.106                    | 0.278                | +0.172              | sadness +0.036                        |
| fear      | 0.102                      | 0.559 [0.489, 0.683]                      | +0.458                | 0.136                    | 0.166                | +0.030              | anger -0.023                          |
| disgust   | 0.073                      | 0.313 [0.222, 0.412]                      | +0.240                | 0.141                    | 0.269                | +0.128              | anger +0.027                          |
| surprise  | 0.094                      | 0.151 [0.141, 0.165]                      | +0.057                | 0.084                    | 0.073                | -0.011              | anger -0.035                          |

- **On the primary judge four of six emotions rise toward their own score.** Happiness (0.126 to 0.686), fear (0.102 to
  0.559), sadness (0.084 to 0.403) and disgust (0.073 to 0.313) rise, with the range over seeds above the unsteered
value. Anger (0.056 to 0.090) and surprise (0.094 to 0.151) rise by 0.034 and 0.057 only. The curve of disgust peaks at
+0.2 (0.411) and falls at +0.3.
- **Hit rate and specificity** (the share of rows of the change matrix whose largest entry is on the diagonal, and the
  mean diagonal change minus the mean off-diagonal change) are 0.833 and 0.323 on the primary judge and 1.000 and 0.264
on the check judge. The primary judge misses once, for anger, whose largest change is on sadness (+0.036 against +0.034
on the diagonal). The check judge's hit for anger is a margin of 0.001 over disgust (+0.172 against +0.171).
- **The judges disagree on fear and anger.** The diagonal change of fear is +0.458 on the primary judge and +0.030 on
  the check judge, and the check judge puts anger at +0.172 where the primary judge has +0.034. The correlation between
  the two judges over the matrix is 0.792. Fear is steered on one judge and not on the other.
- **Not stored and not controlled.** No perplexity or distinct-1 and no random-direction arm, so the table shows an
  effect of the emotion vector at +0.3 on a judge, not whether the text is fluent and not how an arbitrary edit of the
  same size compares. The probabilities are those of a classifier on short completions; the run does not measure whether
  a human would read them as the emotion.

### Template steering (P7)

P7 extracts the sentiment reader under six templates of P1 and steers with the C1 protocol (run `p7/gpt2/6c19128c5192`;
coefficients -0.3, -0.15, 0, 0.15 and 0.3, 50 prompts, 2 samples per prompt, 2 steering readers per template), over the
sweep layers 1, 3, 5, 8, 10 and 12. A unit is a template, a sweep layer and a steering reader (72 units). It is readable
if the validation accuracy of its reader at that layer is at least 0.6, and inverted if it is readable and its primary
dose slope is negative.

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at L12 | Dose slope, sweep layers 1 to 10 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ----------------- | -------------------------------- | -------------- | -------- |
| paper          | L8         | 0.772         | L12         | 1.535             | 0.155 to 0.885                   | 12             | 0        |
| no_template    | L10        | 0.817         | L12         | 1.664             | 0.029 to 0.853                   | 12             | 0        |
| no_chat_tags   | L11        | 0.870         | L12         | 1.715             | 0.307 to 1.144                   | 12             | 0        |
| neutral_frame  | L10        | 0.798         | L12         | 1.639             | 0.020 to 0.972                   | 12             | 1        |
| scaffold_0     | L11        | 0.862         | L12         | 1.681             | 0.102 to 1.013                   | 12             | 0        |
| concept_swap_0 | L10        | 0.791         | L12         | 1.657             | 0.106 to 0.791                   | 12             | 0        |

- **Every template steers most at the last sweep layer.** The steering layer is L12 for all six templates, the last
  layer of the sweep and the last block of the model, so it is the edge of what was swept and not an interior optimum.
  The slope rises with the layer overall, with a dip at layer 5 for the paper template (0.155, 0.426, 0.271, 0.650,
  0.885 and 1.535 at layers 1, 3, 5, 8, 10 and 12). The slope at the steering layer is measured on the prompts that
  chose that layer and is optimistic.
- **Reading and steering.** Every unit is readable (72 of 72) and one is inverted (`neutral_frame`), a rate of 0.014
  [0.002, 0.075]. The Spearman correlation over the 72 units is 0.875 between reading accuracy and dose slope and 0.937
  between the logit-lens score and dose slope, and the sign of the slope agrees between the two judges for 0.958 of the
  units. The mean slope in the prompt-format transfer (raw, chat and story formats) is 1.507 to 1.679 for the six
  templates.
- **The correlation is not separated from depth.** Both the reading accuracy and the slope rise with depth over the
  sweep, so a rank correlation over units that mixes layers can be driven by depth alone. The run does not give the
  within-layer correlation.
- **Not stored and not controlled.** No perplexity or distinct-1, and no random-direction arm. The grid reaches 0.3,
  where C1 on `gpt2` has a perplexity of 438.0, so a dose slope fitted over it is not a rate of steering within fluent
  text.

### LoRRA adapter (LR1) and the TruthfulQA control (C4)

LR1 trains one LoRRA adapter on `gpt2` (run `lr1/gpt2/1e9cd6f56388`: layers 4, 6 and 8, rank 8, learning rate 3e-4, 70
steps, 196,608 trainable parameters, 2,000 training examples and 64 held-out, with the control prompts "Give a truthful
answer." and "Give an untruthful answer."). The training loss goes from 27.4 to 15.7, and the held-out loss from 7.07 to
3.80 at layer 4, from 8.05 to 4.65 at layer 6 and from 11.11 to 6.72 at layer 8 (reductions of 0.463, 0.423 and 0.396).
LR1 has no honesty evaluation; it trains the adapter that C4 uses.

C4 asks whether steering or the adapter changes TruthfulQA MC1 (run `c4/gpt2/1b571ac45420`, which reads the LR1
adapter). MC1 is scored on the log-probabilities of the answer choices under the edited model, not on generated text.
The 173 test questions are those of R6, the coefficients are chosen on the 644 train and validation questions, and the
chance line is 0.228. The unsteered row (0.168) is the standard row of R6. C4 scores the edited model itself and trains
no reader on the test answers, so the R6 text baselines (0.676 and 0.671) are not its comparison.

| Arm                        | Test MC1 [95% interval] | Change from unsteered [95% interval]       |
| -------------------------- | ----------------------- | ------------------------------------------ |
| none                       | 0.168 [0.119, 0.230]    | 0                                          |
| prompting +1 and -1        | 0.168 [0.119, 0.230]    | 0.000 [-0.023, +0.023]                     |
| difference (0.1)           | 0.214 [0.159, 0.281]    | +0.046 [+0.006, +0.087]                    |
| contrast (0.25)            | 0.173 [0.124, 0.237]    | +0.006 [0.000, +0.017]                     |
| LoRRA                      | 0.179 [0.129, 0.243]    | +0.012 [0.000, +0.029]                     |
| reading (10 readers, mean) | 0.180 (0.168 to 0.197)  | +0.012 (every reader's interval has zero)  |

- **Every arm is below the chance line of 0.228.** The only change whose interval excludes zero is `difference`
  (+0.046, a gain of 8 questions of 173), among all the arms of the table and with no correction for multiple
  comparisons. The `contrast` and LoRRA intervals touch zero at 0.000.
- **The ten readers move MC1 little.** Their test MC1 ranges from 0.168 to 0.197. The coefficient chosen on the train
  and validation questions was negative for five readers (-0.05, -0.1, -0.1, -0.2 and -0.2), zero for two and +0.05 for
  three, so the selection did not find one sign, and the largest changes (+0.023 to +0.029) are those of readers with a
  negative coefficient.
- **No perplexity or distinct-1** is stored for C4. The scoring is not generation, so the cost to the text is not
  measured.

### Lie gallery (C5)

C5 on `gpt2-xl` (run `c5/gpt2-xl/155f0f72bc3f`; 1,000 TriviaQA questions, a 15-shot prompt, one sample per question at
temperature 0.8, relative strength 0.1, steered over layers 13, 17, 22, 27, 31, 35, 40 and 45) records the share of
correct answers without steering and under an honesty-steered and a lie-steered arm: 0.147, 0.107 and 0.080. Of the 147
questions answered correctly without steering, 77 are answered wrongly under the lie arm, a flip rate of 0.524 [0.443,
0.603]. The honesty-steered arm is also below the unsteered one, and there is no random-direction arm, so the flips do
not separate an effect of the lie direction from a generic perturbation of the model. No perplexity or distinct-1 is
stored. This run has no emotion gallery.

## Base against post-trained Qwen3 (P8)

P8 pairs each base checkpoint of Qwen3 with its post-trained one, at 1.7B and 0.6B, and reads the finished runs of the
checkpoints: R1, R2, R3 and R7 (concept templates), R5 (persona templates, as its transfer test only), the P1 runs for
the template spread and the C1 runs for the dose slope. It loads no model. `gpt2` is read the same way as a reference
with no post-trained pair. Seed `i` draws the same stimuli on a base checkpoint and on its post-trained pair, so the
difference is taken per seed (post-trained minus base) and its 95 percent interval is a percentile bootstrap over the
10 seeds. As in the Qwen3 reading section, the post-trained checkpoints are small distilled models used with thinking
disabled, so a difference here is a difference between those two recipes and not a statement about instruction tuning
in general.

Accuracy of each source at its selected layer, with the delta (post-trained minus base). The layers are those of the
reading tables and may differ between base and post-trained; `Selected on` is the split the layer was chosen on (see
the reading conventions). R1 has no separate test split, so its numbers are optimistic by the selection for both
checkpoints. R5 is the transfer test selected on the transfer validation split.

| Pair | Source | Base  | Post-trained | Delta [95% interval]    | Layers (base, post) | Selected on  |
| ---- | ------ | ----- | ------------ | ----------------------- | ------------------- | ------------ |
| 1.7B | R1     | 0.905 | 0.866        | -0.040 [-0.053, -0.027] | L18, L18            | curves       |
| 1.7B | R2     | 0.768 | 0.756        | -0.012 [-0.023, -0.001] | L18, L28            | curves_val   |
| 1.7B | R3     | 0.896 | 0.892        | -0.004 [-0.010, +0.002] | L20, L18            | curves_val   |
| 1.7B | R7     | 0.934 | 0.937        | +0.003 [-0.015, +0.020] | L18, L18            | curves_val   |
| 1.7B | R5     | 0.778 | 0.781        | +0.003 [-0.042, +0.049] | L20, L20            | transfer_val |
| 0.6B | R1     | 0.947 | 0.857        | -0.090 [-0.106, -0.074] | L19, L18            | curves       |
| 0.6B | R2     | 0.744 | 0.732        | -0.012 [-0.024, +0.003] | L23, L28            | curves_val   |
| 0.6B | R3     | 0.793 | 0.778        | -0.016 [-0.031, -0.004] | L24, L18            | curves_val   |
| 0.6B | R7     | 0.763 | 0.736        | -0.027 [-0.066, +0.008] | L23, L18            | curves_val   |
| 0.6B | R5     | 0.456 | 0.529        | +0.073 [+0.059, +0.086] | L27, L21            | transfer_val |

The `gpt2` reference, with an interval recomputed by P8 (a bound differs from the reading tables by at most 0.002):

| Source | gpt2 accuracy [95% interval] | Layer |
| ------ | ---------------------------- | ----- |
| R1     | 0.772 [0.759, 0.784]         | L8    |
| R2     | 0.655 [0.644, 0.668]         | L12   |
| R3     | 0.623 [0.600, 0.646]         | L11   |
| R7     | 0.656 [0.631, 0.683]         | L9    |
| R5     | 0.386 [0.369, 0.403]         | L7    |

- **Concept sources.** Seven of the eight deltas are negative. The largest is R1 (-0.040 at 1.7B and -0.090 at 0.6B)
  and the intervals of R1 exclude zero in both pairs. Beyond R1, the intervals exclude zero for R2 at 1.7B (-0.012)
  and R3 at 0.6B (-0.016); the other four concept deltas have an interval that includes zero.
- **R5, persona templates.** At 1.7B the delta is +0.003 [-0.042, +0.049]. At 0.6B it is +0.073 [+0.059, +0.086], but
  the R5 transfer test of `qwen3-0.6b-base` is 0.456, below chance, and the post-trained value is 0.529: the positive
  delta starts from a below-chance base and ends near chance, and the test set has 59 pairs that the interval does not
  cover. The validation accuracy at the selected layer was 0.651 and 0.734 (see the Qwen3 reading section), so the
  delta is not a measure of how much honesty transfers.

Function gap, the mean delta of the function source (R5) minus the mean delta of the four concept sources, and the
template spread: the population standard deviation (divisor N) over the 16 variants shared by every model (the
variants of P1 without `no_trailing_space`), computed per seed with each variant read at the layer its own validation
curve picks, and averaged over the seeds. The `gpt2` row is the reference and has no pair.

| Pair | Function gap [95% interval] | SD base              | SD post-trained      | SD delta [95% interval] |
| ---- | --------------------------- | -------------------- | -------------------- | ----------------------- |
| 1.7B | +0.017 [-0.033, +0.061]     | 0.039 [0.033, 0.047] | 0.057 [0.050, 0.065] | +0.018 [+0.007, +0.027] |
| 0.6B | +0.109 [+0.092, +0.127]     | 0.059 [0.053, 0.065] | 0.054 [0.049, 0.060] | -0.005 [-0.014, +0.003] |
| gpt2 |                             |                      | 0.029 [0.026, 0.034] |                         |

- **Function gap.** The mean concept delta is -0.013 at 1.7B and -0.036 at 0.6B, so the gap is +0.017 [-0.033, +0.061]
  at 1.7B and +0.109 [+0.092, +0.127] at 0.6B. The 0.6B gap is the R5 delta of the row above (+0.073) minus the
  negative concept mean, and inherits the below-chance base of R5.
- **Template spread.** The spread of the 1.7B pair goes from 0.039 to 0.057 (delta +0.018, interval above zero) and
  that of the 0.6B pair from 0.059 to 0.054 (delta -0.005, interval including zero). All four Qwen3 values are above
  the `gpt2` value (0.029), whose interval overlaps that of `qwen3-1.7b-base` (0.033 to 0.047). The spread is over the
  accuracy of the variants, each at its own layer, so it does not equal the span of the P1 tables, which uses one
  layer.

Dose slopes of the C1 runs. A slope is the least-squares slope of the sentiment score on the relative coefficient over
the grid of the linear operator, one per steering reader; the delta is the mean over readers of the post-trained slope
minus the base slope of the same reader index:

| Pair | Base, readers 1 to 3 | Mean  | Post-trained, readers 1 to 3 | Mean  | Delta  |
| ---- | -------------------- | ----- | ---------------------------- | ----- | ------ |
| 1.7B | 1.103, 1.325, 1.092  | 1.173 | 1.160, 1.730, 1.267          | 1.385 | +0.212 |
| 0.6B | 1.679, 1.899, 1.815  | 1.798 | 1.679, 1.890, 1.127          | 1.565 | -0.233 |

- **Slopes.** At 1.7B the mean slope goes from 1.173 to 1.385 (delta +0.212) and at 0.6B from 1.798 to 1.565 (delta
  -0.233). With three readers per cell there is no interval, and the ranges over readers overlap in both pairs (1.7B:
  1.092 to 1.325 against 1.160 to 1.730; 0.6B: 1.679 to 1.899 against 1.127 to 1.890). At 0.6B two of the three
  readers have almost the same slope on both checkpoints (1.679 and 1.679, 1.899 and 1.890), and the negative delta
  comes from the third (1.815 against 1.127).
- **Fluency caveat.** The slope is fitted over all seven coefficients, including the coefficients of 0.2 and 0.3 in
  either direction, where the steered text has degenerated (perplexity from 1.70 to 34.70 times the unsteered one and
  distinct-1 as low as 0.009, see the Qwen3 C1 section). It summarizes the score curve over those coefficients, not a
  rate of steering within fluent text, and the scores saturate or move back at the larger coefficients.

What this does and does not show. At 1.7B every delta except R1 is within 0.013 of zero, and at 0.6B R1 is -0.090 and
the other concept deltas are between -0.012 and -0.027. The template spread rises for the 1.7B post-trained checkpoint
and not for the 0.6B one. There are two pairs, one run per cell, and the intervals cover the training seeds and not
the evaluation pairs. The data do not say that the post-trained recipe causes any of the differences, and the R5 and
slope rows of the 0.6B pair carry the caveats above.

## Known versus unknown questions (P9)

P9 asks whether a direction in the residual stream separates the questions a model answers from the questions it
cannot answer, and what adding that direction does. P9a labels each question from the behavior of the model, P9b reads
the known against unknown direction, P9c predicts the errors of the model and P9d adds the direction to the model
while it answers. The definitions, the guess control and the statistics are in the
[methodology](methodology.md#known-versus-unknown-questions-p9).

The runs reported here use the current label schema (12 PopQA relations, the guess control) and are those of the four
GPT-2 sizes and of the Qwen3 0.6B and 1.7B checkpoints. P9a covers all eight models. P9b and P9c ran on the four GPT-2
sizes and the two Qwen3 base checkpoints, and P9d on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the four Qwen3
checkpoints (base and post-trained); P9b and P9c have no post-trained run. `gpt2` is the one model whose label gate
closes (70 known questions after the guess control, below the 100 the gate asks for), so it has no known against
unknown contrast: its P9b run reads only the contrasts between kinds of unknown questions, its P9c run has no label-free
`lat` predictor, and P9d did not run on it. Questions are labeled from the behavior
of each model, so each model has its own question sets and a count or an AUROC is not a measure of the same questions
across models. Each cell is one run, so no interval includes variation between runs. The post-trained checkpoints use
the chat template with thinking disabled (see the Qwen3 section).

### Labels (P9a)

P9a labels each closed-book question from the behavior of the model on a 15-shot prompt: known if the greedy answer is
correct and at least 3 of 5 samples at temperature 1.0 are correct, unknown if the greedy answer is wrong and no
sample is correct, uncertain otherwise. A run labels 21,988 questions (22,028 for the four GPT-2 sizes). The sources
whose questions can be known are PopQA (9,467 questions over 12 relations), TriviaQA and NQ-open (1,000 each) and the
real award questions; the others (SyntHal, generated names, unknowable details, FalseQA, KUQ) are unknown by
construction. The counts below are after the guess control, which turns a known question that it rejects into an
uncertain one. Each model is labeled on its own behavior, so the question sets differ between models.

Known / unknown / uncertain questions per source:

| Model           | PopQA (9,467)       | TriviaQA (1,000) | NQ-open (1,000) |
| --------------- | ------------------- | ---------------- | --------------- |
| qwen3-0.6b-base | 329 / 8,195 / 943   | 153 / 668 / 179  | 43 / 850 / 107  |
| qwen3-0.6b      | 277 / 8,446 / 744   | 89 / 776 / 135   | 27 / 911 / 62   |
| qwen3-1.7b-base | 501 / 7,928 / 1,038 | 291 / 502 / 207  | 70 / 775 / 155  |
| qwen3-1.7b      | 517 / 8,137 / 813   | 271 / 592 / 137  | 71 / 848 / 81   |
| gpt2            | 69 / 8,836 / 562    | 1 / 919 / 80     | 0 / 973 / 27    |
| gpt2-medium     | 176 / 8,569 / 722   | 20 / 851 / 129   | 5 / 936 / 59    |
| gpt2-large      | 243 / 8,495 / 729   | 49 / 767 / 184   | 12 / 937 / 51   |
| gpt2-xl         | 324 / 8,397 / 746   | 94 / 710 / 196   | 14 / 909 / 77   |

Guess control, in questions:

| Model           | Known before | Known after | Guessable stratum | Modal answer | Excluded stratum |
| --------------- | ------------ | ----------- | ----------------- | ------------ | ---------------- |
| qwen3-0.6b-base | 724          | 525         | 14                | 7            | 178              |
| qwen3-0.6b      | 669          | 393         | 0                 | 7            | 269              |
| qwen3-1.7b-base | 1,090        | 863         | 42                | 12           | 173              |
| qwen3-1.7b      | 1,118        | 860         | 24                | 11           | 223              |
| gpt2            | 87           | 70          | 0                 | 2            | 15               |
| gpt2-medium     | 301          | 201         | 0                 | 4            | 96               |
| gpt2-large      | 428          | 304         | 0                 | 4            | 120              |
| gpt2-xl         | 540          | 432         | 0                 | 5            | 103              |

- **Known questions are few.** After the guess control a model has 70 to 863 known questions: 70 (`gpt2`), 201
  (`gpt2-medium`), 304 (`gpt2-large`), 393 (`qwen3-0.6b`), 432 (`gpt2-xl`), 525 (`qwen3-0.6b-base`), 860
  (`qwen3-1.7b`) and 863 (`qwen3-1.7b-base`), against 21,988 to 22,028 questions. They come from PopQA (69 to 517 per
  model), TriviaQA (1 to 291) and NQ-open (0 to 71); the real award questions give at most 1, and no source other than
  these three has a known question on `gpt2`, `gpt2-medium` or `gpt2-large`. The count rises with size inside the GPT-2
  family (70, 201, 304, 432).
- **The guess control matters.** It demotes 17 (`gpt2`) to 276 (`qwen3-0.6b`) of the known questions of a model, 15 to
  269 of them because their stratum is excluded (`popqa:sport` is excluded by default, and so are the strata
  whose constant-guess accuracy is above 0.10).
- **Base against post-trained.** At 1.7B the two checkpoints have 863 and 860 known questions; at 0.6B the base
  checkpoint has 525 and the post-trained one 393. Labels come from each model's own behavior, so this is a count of
  what each model answers under the labeling recipe, not a comparison on common questions.
- **Gate and cutoff.** Seven of the eight models pass the gate (at least 100 guess-adjusted known questions), so the
  known against unknown contrasts are built for them. `gpt2` does not (70), and the run records
  `known_contrasts = false`. The cutoff year is a setting (2017 for `gpt2-xl`, 2024 for Qwen3). The cutoff
  estimated from the labels is stored for two runs, 2014 for `qwen3-1.7b-base` and 2007 for `qwen3-1.7b`, and does not
match the setting. The only results below that use the cutoff are the `pre_vs_post_cutoff` rows of the four GPT-2 sizes;
the post-cutoff contrasts have too few items for the Qwen3 models.

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
(the PopQA known side has 87, 118 and 78 questions). In the table below `Test pairs` is `n_pairs_within` of the run:
the number of (known, unknown) pairs of test questions that share a relation, over which the within-relation AUROC
averages. It counts test pairs and is not the 64 training pairs of the reader.

PopQA known against unknown, within relation (the headline), mean AUROC over the five readers:

| Model           | Contrast         | Test known / unknown | Test pairs | Reader | popularity | text_word | text_char |
| --------------- | ---------------- | -------------------- | ---------- | ------ | ---------- | --------- | --------- |
| qwen3-0.6b-base | obscure_real     | 87 / 1,661           | 6,941      | 0.919  | 0.799      | 0.678     | 0.673     |
| qwen3-0.6b-base | obscure_real_pop | 87 / 1,661           | 6,941      | 0.891  | 0.799      | 0.678     | 0.673     |
| qwen3-1.7b-base | obscure_real     | 118 / 1,611          | 10,113     | 0.959  | 0.879      | 0.625     | 0.636     |
| qwen3-1.7b-base | obscure_real_pop | 118 / 1,611          | 10,113     | 0.960  | 0.879      | 0.625     | 0.636     |
| gpt2-xl         | obscure_real     | 78 / 1,711           | 6,691      | 0.902  | 0.838      | 0.511     | 0.543     |
| gpt2-xl         | obscure_real_pop | 78 / 1,711           | 6,691      | 0.879  | 0.838      | 0.511     | 0.543     |

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
  within 0.03 of `obscure_real`. The two contrasts have the same test counts and the same baseline values: popularity
  0.799, 0.879 and 0.838, `text_word` 0.678, 0.625 and 0.511, `text_char` 0.673, 0.636 and 0.543. Pooled,
  `qwen3-0.6b-base` reads 0.948 on both contrasts (0.9475 and 0.9482 before rounding), and its random direction has
  one stored value, 0.682, for both; on the other two models the random direction differs between the two contrasts.
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

- **The reader has a margin of 0.07 or more over the better text baseline on two contrasts.** `triviaqa_known` sets
  TriviaQA questions labeled known against TriviaQA questions labeled unknown, so one question format stands on both
  sides: the reader reads 0.905, 0.883 and 0.905, against a better text baseline of 0.774, 0.693 and 0.787.
  `kuq_ambiguous` is the other one, with margins of 0.105, 0.079 and 0.070. On every other contrast the largest margin
  over the better baseline is 0.038. The `triviaqa_known` test sets are small (32, 55 and 19 known questions).
- **On several contrasts the reader does not exceed the text baselines.** `unknowable_detail` is 1.000 for the reader
  and for both baselines on all three models, so the wording alone separates it. On `false_premise` the reader (0.951,
  0.974, 0.964) is below the better baseline (0.978, 0.977, 0.980). The two contrasts between kinds of unknown
  questions (`real_vs_synthal`, `real_vs_generated`; they do not contain known questions) have `text_char` at 1.000 on
  `real_vs_synthal` and 0.999, 0.999 and 0.998 on `real_vs_generated`, above the reader (0.978, 0.986 and 0.962 on
  `real_vs_synthal`, 0.980, 0.986 and 0.981 on `real_vs_generated`).
- **Elsewhere the margin is small.** `fictional_generated` (PopQA known against generated names, within relation) has
  the reader above `text_char` by 0.002 to 0.015; four KUQ categories (`kuq_controversial`, `kuq_counterfactual`,
  `kuq_false_assumption`, `kuq_future_unknown`) are within 0.025 of the better baseline in either direction, and
  `kuq_unsolved_problem` is above it by 0.025 to 0.038. `pre_vs_post_cutoff` (`gpt2-xl` only, 46 against 20 test
  questions) is 1.000 for the reader and both baselines.
- **Read these cross-dataset rows as a format contrast too.** The known side of most of them is PopQA, TriviaQA or
  NQ-open and the unknown side is another source, so the two sides differ in question format. The text classifiers are
  at 0.94 or above (the better of the two) on every row except `triviaqa_known` and `kuq_ambiguous`, which says that
  the wording separates the sides there; a reader at 0.98 to 1.00 on those rows does not show that it reads knowledge.

#### The other GPT-2 sizes

P9b on `gpt2`, `gpt2-medium` and `gpt2-large` (runs `p9b/gpt2/fd3bdc5a4b0d`, `p9b/gpt2-medium/a5a9c7d7b176` and
`p9b/gpt2-large/7b3fd5d1bfb5`, made at `ca2546c` on the labels of the P9a runs of the same size) uses the same recipe
as above: five logistic readers on 64 pairs each, one layer per model chosen on the pooled validation AUROC, which is
L4 on `gpt2`, L24 on `gpt2-medium` and L36 on `gpt2-large` (the last block of both larger models). Every reader value
is given next to `text_word`, `text_char` and `popularity`. Only the contrasts with at least 16 training questions on
each side and 10 validation questions on each side are read: `gpt2` has no known against unknown contrast (its gate
closed), so it reads only the three contrasts between kinds of unknown questions; `gpt2-medium` reads six contrasts
(`fictional_synthal`, `triviaqa_known`, `unknowable_detail`, `false_premise` and the six KUQ categories are skipped for
lack of items) and `gpt2-large` reads 15 (only `fictional_synthal` is skipped).

PopQA known against unknown, within relation (the headline), mean AUROC over the five readers. `obscure_real_pop` also
matches the popularity bin, so it is the within-popularity version of the contrast:

| Model       | Contrast         | Test known / unknown | Test pairs | Reader | popularity | text_word | text_char |
| ----------- | ---------------- | -------------------- | ---------- | ------ | ---------- | --------- | --------- |
| gpt2-medium | obscure_real     | 46 / 1,746           | 3,500      | 0.778  | 0.728      | 0.556     | 0.557     |
| gpt2-medium | obscure_real_pop | 46 / 1,746           | 3,500      | 0.808  | 0.728      | 0.556     | 0.557     |
| gpt2-large  | obscure_real     | 55 / 1,734           | 3,999      | 0.815  | 0.787      | 0.550     | 0.536     |
| gpt2-large  | obscure_real_pop | 55 / 1,734           | 3,999      | 0.800  | 0.787      | 0.550     | 0.536     |

The same contrasts pooled over relations, with the AUROC of a random direction:

| Model       | Contrast         | Reader | Random direction | popularity | text_word | text_char |
| ----------- | ---------------- | ------ | ---------------- | ---------- | --------- | --------- |
| gpt2-medium | obscure_real     | 0.858  | 0.758            | 0.813      | 0.945     | 0.929     |
| gpt2-medium | obscure_real_pop | 0.850  | 0.758            | 0.813      | 0.945     | 0.929     |
| gpt2-large  | obscure_real     | 0.928  | 0.702            | 0.886      | 0.924     | 0.930     |
| gpt2-large  | obscure_real_pop | 0.937  | 0.702            | 0.886      | 0.924     | 0.930     |

Other relational contrasts, within relation, as reader / `text_word` / `text_char` (`popularity` is not defined for
them):

| Contrast                              | gpt2                  | gpt2-medium           | gpt2-large            |
| ------------------------------------- | --------------------- | --------------------- | --------------------- |
| fictional_generated (within relation) | not read              | 0.924 / 0.836 / 0.924 | 0.985 / 0.816 / 0.976 |
| real_vs_synthal (within relation)     | 0.940 / 0.738 / 1.000 | 0.946 / 0.738 / 1.000 | 0.968 / 0.737 / 1.000 |
| real_vs_generated (within relation)   | 0.977 / 0.951 / 0.998 | 0.956 / 0.951 / 0.998 | 0.971 / 0.951 / 0.998 |
| pre_vs_post_cutoff (within relation)  | 1.000 / 1.000 / 1.000 | 0.947 / 1.000 / 1.000 | 0.984 / 1.000 / 1.000 |

Contrasts without PopQA questions on a side, pooled, as reader / `text_word` / `text_char` (`gpt2-large` only; the
`triviaqa_known` test set has 10 known questions and each KUQ category 11):

| Contrast             | gpt2     | gpt2-medium | gpt2-large            |
| -------------------- | -------- | ----------- | --------------------- |
| triviaqa_known       | not read | not read    | 0.869 / 0.761 / 0.743 |
| unknowable_detail    | not read | not read    | 1.000 / 1.000 / 1.000 |
| false_premise        | not read | not read    | 0.961 / 0.975 / 0.955 |
| kuq_ambiguous        | not read | not read    | 0.960 / 0.842 / 0.925 |
| kuq_controversial    | not read | not read    | 0.977 / 0.901 / 0.927 |
| kuq_counterfactual   | not read | not read    | 0.997 / 0.998 / 1.000 |
| kuq_false_assumption | not read | not read    | 0.963 / 0.980 / 0.960 |
| kuq_future_unknown   | not read | not read    | 0.978 / 0.997 / 1.000 |
| kuq_unsolved_problem | not read | not read    | 0.985 / 0.951 / 0.967 |

- **Within relation the reader is above popularity and the text baselines, by less than on the other models.** On
  `obscure_real` the reader reads 0.778 on `gpt2-medium` and 0.815 on `gpt2-large`, against a popularity baseline of
  0.728 and 0.787: a margin of 0.050 and 0.028, against 0.064 on `gpt2-xl` and 0.080 to 0.120 on the Qwen3 base
  checkpoints. The word and character classifiers are at 0.556 and 0.557 on `gpt2-medium` and 0.550 and 0.536 on
  `gpt2-large`, close to chance. With the popularity bin matched (`obscure_real_pop`) the reader reads 0.808 and 0.800,
  a margin over popularity of 0.080 on `gpt2-medium` and about 0.01 on `gpt2-large`. The margin on `gpt2-large` is small
against a test known side of 55 questions, and the stored interval covers the pair draws of the reader and not the test
sampling, so it does not show that the reader recovers more than popularity there. The margin does not rise with size
along the ladder (0.050, 0.028 and 0.064 for medium, large and xl).
- **Pooled, the reader carries nothing that the wording does not.** On `gpt2-medium` the pooled reader (0.858) is below
  both text baselines (0.945 and 0.929) and above popularity (0.813) and a random direction (0.758); on `gpt2-large` it
  (0.928) is level with them (0.924 and 0.930) and above popularity (0.886) and a random direction (0.702). On
  `fictional_generated` the reader is below `text_char` pooled on both (0.903 against 0.981, 0.970 against 0.989) and
  within relation level with it on `gpt2-medium` (0.924 against 0.924) and 0.009 above it on `gpt2-large` (0.985 against
  0.976).
- **Between kinds of unknown questions the reader is below the character classifier.** `real_vs_synthal`,
  `real_vs_generated` and `pre_vs_post_cutoff` are read on all three sizes. Pooled, in the order `gpt2`, `gpt2-medium`,
`gpt2-large`, the reader reads 0.939, 0.937 and 0.959 on `real_vs_synthal` (`text_char` 1.000 on all three), 0.959,
0.937 and 0.952 on `real_vs_generated` (`text_char` 0.998) and 0.995, 0.953 and 0.974 on `pre_vs_post_cutoff`
(`text_char` 1.000). The reader is below `text_char` on all three contrasts and on all three sizes, so the only
contrasts that `gpt2` reads do not show more than the wording of the question carries. These contrasts do not contain
known questions.
- **`gpt2-large`, the 15 contrasts.** The reader is above both text baselines, pooled, on five: `triviaqa_known` (0.869
  against 0.761 and 0.743), `kuq_controversial` (0.977 against 0.901 and 0.927), `kuq_ambiguous` (0.960 against 0.842
  and 0.925), `kuq_unsolved_problem` (0.985 against 0.951 and 0.967) and `obscure_real_pop` (0.937 against 0.924 and
  0.930, a margin of 0.006). `text_char` is at or above the reader on 8 of the 15 (7 strictly above; `unknowable_detail`
  is 1.000 for the reader and both baselines) and `text_word` on 7 of the 15 (6 strictly). On `false_premise`
  (0.961 against 0.975 and 0.955), `kuq_counterfactual`, `kuq_false_assumption` and `kuq_future_unknown` the reader is
  within 0.022 of the better baseline in either direction. The `triviaqa_known` margin of 0.108 rests on 10 known test
  questions.
- **What this does not show.** As above: popularity and the wording of the question are the only surface controls, each
  value is one set of five readers on one labeling run, and the question sets differ between models.

### Error prediction (P9c)

P9c asks which score predicts whether the greedy few-shot answer of the model is correct, on held-out PopQA (1,937),
TriviaQA (191) and NQ-open (210) questions. Every predictor is a score where higher means more likely correct and the
value is its AUROC for correct against wrong answers (0.5 is chance). The output-confidence baselines read the model's
own output: `answer_logprob` (log-probability of the greedy answer), `first_token_prob` (probability of the most
likely token at the start of the answer), `neg_entropy` (minus the entropy of that distribution),
`mean_token_logprob`, `consistency` (share of the five sampled answers that equal the greedy one) and `p_true` (the
model grades its own answer). `probe` is a logistic probe fitted on the correctness of the train questions of the same
dataset, so it sees the label of the target. `lat` is a reader of the `obscure_real` known against unknown contrast,
with the contrast and the recipe of P9b (logistic, 64 pairs, five draws), refitted in this run and with no correctness
label. `popularity` and `template` (the accuracy rate of the relation on the train questions) use no hidden state and
no confidence. The probe and `lat` read layer L27 (`qwen3-0.6b-base`), L27 (`qwen3-1.7b-base`) and L48 (`gpt2-xl`),
chosen on the validation split of P9c, so not the layers of the P9b readers.

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
  `lat` adds +0.000, +0.002 and +0.011, with intervals that include or touch zero (the interval of `lat` at
  `qwen3-0.6b-base` is [-0.0001, +0.0004]). How much of the probe's pooled gain is within-relation is not stored.
- **TriviaQA and NQ-open.** On TriviaQA the probe is below `answer_logprob` on every model (0.868 against 0.886, 0.811
  against 0.819, 0.740 against 0.774). On NQ-open its point estimate is above it (0.795 against 0.776, 0.812 against
  0.790, 0.725 against 0.636), but the sets hold 20, 38 and 12 correct answers and the intervals are 0.14 to 0.37 wide
  and overlap. The increments over the confidence stack have intervals that include or touch zero on both sets (the
  probe on TriviaQA at `qwen3-0.6b-base` has a lower bound of exactly 0.000), except `lat` (+0.047) and the probe
  (+0.036) on NQ-open at `qwen3-1.7b-base`.
- **`p_true` on `gpt2-xl` is below chance on PopQA.** It reads 0.293 [0.253, 0.334] pooled and 0.431 [0.373, 0.487]
  within relation, with intervals below 0.5. On TriviaQA (0.481) and NQ-open (0.347) the intervals include 0.5. On the
  Qwen3 models it is 0.719 and 0.776 within relation, below `answer_logprob`.

#### The other GPT-2 sizes

P9c on `gpt2`, `gpt2-medium` and `gpt2-large` (runs `p9c/gpt2/7ed884a2e8df`, `p9c/gpt2-medium/b3d936ab3f3b` and
`p9c/gpt2-large/fb5b8a90c085`, made at `ca2546c`) has the same predictors, the same held-out PopQA (1,937), TriviaQA
(191) and NQ-open (210) questions and the same intervals as above. The probe and `lat` read L11 (`gpt2`), L23
(`gpt2-medium`) and L36 (`gpt2-large`), chosen on the validation split of P9c. `gpt2` has no label-free `lat` (its gate
closed, so there is no known against unknown contrast to fit it on). The number of correct answers, which sets how much
a within-relation AUROC can show, is small: 98, 158 and 161 of the 1,937 PopQA questions.

PopQA, within relation, AUROC with its interval (the headline; `template` is 0.500 by construction and is not shown).
The `popularity` row is the within-relation popularity baseline. No AUROC within popularity bins is stored for this
experiment:

| Predictor          | gpt2                 | gpt2-medium          | gpt2-large           |
| ------------------ | -------------------- | -------------------- | -------------------- |
| answer_logprob     | 0.701 [0.629, 0.766] | 0.790 [0.740, 0.840] | 0.832 [0.782, 0.875] |
| first_token_prob   | 0.715 [0.645, 0.778] | 0.773 [0.718, 0.822] | 0.831 [0.782, 0.873] |
| neg_entropy        | 0.670 [0.599, 0.741] | 0.777 [0.721, 0.827] | 0.803 [0.748, 0.850] |
| mean_token_logprob | 0.720 [0.655, 0.781] | 0.773 [0.723, 0.823] | 0.863 [0.819, 0.899] |
| consistency        | 0.595 [0.526, 0.659] | 0.706 [0.650, 0.759] | 0.752 [0.700, 0.805] |
| lat                | not run              | 0.670 [0.609, 0.735] | 0.690 [0.621, 0.754] |
| probe              | 0.789 [0.733, 0.842] | 0.726 [0.664, 0.784] | 0.800 [0.745, 0.847] |
| p_true             | 0.445 [0.368, 0.523] | 0.521 [0.464, 0.579] | 0.529 [0.462, 0.590] |
| popularity         | 0.541 [0.470, 0.618] | 0.577 [0.511, 0.646] | 0.628 [0.558, 0.700] |

PopQA pooled over relations, as AUROC / AURC (lower is better for the AURC):

| Predictor                  | gpt2          | gpt2-medium   | gpt2-large    |
| -------------------------- | ------------- | ------------- | ------------- |
| probe                      | 0.961 / 0.829 | 0.948 / 0.753 | 0.944 / 0.751 |
| answer_logprob             | 0.899 / 0.857 | 0.926 / 0.768 | 0.930 / 0.764 |
| neg_entropy                | 0.913 / 0.855 | 0.897 / 0.778 | 0.917 / 0.770 |
| lat                        | not run       | 0.835 / 0.797 | 0.852 / 0.799 |
| template                   | 0.939 / 0.848 | 0.929 / 0.764 | 0.907 / 0.772 |
| popularity                 | 0.469 / 0.927 | 0.480 / 0.879 | 0.542 / 0.863 |
| correct answers (of 1,937) | 98            | 158           | 161           |

TriviaQA and NQ-open, pooled AUROC with its interval (the items have no relation):

| Predictor                | gpt2                 | gpt2-medium          | gpt2-large           |
| ------------------------ | -------------------- | -------------------- | -------------------- |
| TriviaQA answer_logprob  | 0.730 [0.602, 0.850] | 0.823 [0.742, 0.896] | 0.741 [0.652, 0.827] |
| TriviaQA neg_entropy     | 0.703 [0.511, 0.867] | 0.712 [0.588, 0.820] | 0.704 [0.603, 0.789] |
| TriviaQA probe           | 0.692 [0.530, 0.829] | 0.773 [0.658, 0.888] | 0.749 [0.653, 0.835] |
| TriviaQA lat             | not run              | 0.605 [0.462, 0.739] | 0.703 [0.601, 0.791] |
| TriviaQA p_true          | 0.644 [0.487, 0.784] | 0.389 [0.268, 0.508] | 0.383 [0.276, 0.500] |
| TriviaQA correct answers | 12 of 191            | 23 of 191            | 37 of 191            |
| NQ-open answer_logprob   | 0.834 [0.641, 0.995] | 0.690 [0.425, 0.933] | 0.813 [0.627, 0.949] |
| NQ-open neg_entropy      | 0.760 [0.490, 1.000] | 0.714 [0.554, 0.865] | 0.727 [0.560, 0.891] |
| NQ-open probe            | 0.666 [0.392, 0.928] | 0.700 [0.530, 0.858] | 0.806 [0.678, 0.920] |
| NQ-open lat              | not run              | 0.626 [0.477, 0.766] | 0.781 [0.619, 0.933] |
| NQ-open p_true           | 0.231 [0.057, 0.427] | 0.589 [0.436, 0.757] | 0.547 [0.323, 0.761] |
| NQ-open correct answers  | 2 of 210             | 9 of 210             | 9 of 210             |

What `probe` and `lat` add to the output confidence (the logistic stacker of the five confidence columns of the
sections above, as a paired AUROC difference with its interval; pooled, no within-relation increment is stored):

| Increment over the confidence stack | gpt2                    | gpt2-medium             | gpt2-large              |
| ----------------------------------- | ----------------------- | ----------------------- | ----------------------- |
| PopQA confidence stack              | 0.926                   | 0.912                   | 0.920                   |
| PopQA + lat                         | not run                 | -0.004 [-0.023, +0.020] | -0.009 [-0.020, +0.002] |
| PopQA + probe                       | +0.037 [+0.025, +0.054] | +0.042 [+0.020, +0.067] | +0.033 [+0.017, +0.050] |
| TriviaQA confidence stack           | 0.591                   | 0.755                   | 0.711                   |
| TriviaQA + lat                      | not run                 | +0.002 [-0.004, +0.008] | +0.005 [-0.003, +0.012] |
| TriviaQA + probe                    | +0.032 [-0.051, +0.129] | +0.026 [-0.006, +0.061] | +0.019 [-0.016, +0.052] |
| NQ-open confidence stack            | 0.548                   | 0.714                   | 0.808                   |
| NQ-open + lat                       | not run                 | -0.020 [-0.069, +0.032] | +0.004 [-0.001, +0.014] |
| NQ-open + probe                     | -0.005 [-0.086, +0.075] | +0.013 [-0.025, +0.055] | -0.002 [-0.006, +0.000] |

- **The probe is the best within relation only on `gpt2`.** On `gpt2` the probe reads 0.789 [0.733, 0.842] within
  relation against 0.701 for `answer_logprob`, with the intervals overlapping at 0.733 to 0.766. On `gpt2-medium`
  `answer_logprob` (0.790) is above the probe (0.726) and on `gpt2-large` `mean_token_logprob` (0.863), `answer_logprob`
  (0.832) and `first_token_prob` (0.831) are above the probe (0.800). The probe sees the correctness label of the
target dataset; the output-confidence predictors do not. On `gpt2-medium` and `gpt2-large` the probe, `answer_logprob`,
`first_token_prob`, `neg_entropy` and `mean_token_logprob` are above the within-relation popularity baseline (0.577 and
0.628) with separated intervals; `lat` is above it as a point estimate on both with overlapping intervals (0.690 [0.621,
0.754] against 0.628 [0.558, 0.700] on `gpt2-large`).
- **`lat` is below the output confidence.** The label-free `lat` reads 0.670 and 0.690 within relation on `gpt2-medium`
  and `gpt2-large`, below `answer_logprob` (0.790, 0.832) and the probe. Added to the confidence stack its increment is
  -0.004 [-0.023, +0.020] and -0.009 [-0.020, +0.002] on PopQA, with intervals that include zero.
- **Pooled, the relation does most of the work.** The accuracy rate of the relation alone (`template`) reads 0.939,
  0.929 and 0.907 pooled, against 0.961, 0.948 and 0.944 for the probe and 0.899, 0.926 and 0.930 for `answer_logprob`;
  on `gpt2` `template` is above `answer_logprob` pooled and above the output-confidence stack (0.926). The probe adds
  +0.037, +0.042 and +0.033 to the stack, with intervals above zero on all three sizes. Popularity pooled is at 0.469,
  0.480 and 0.542, and the intervals include 0.5 on all three sizes.
- **TriviaQA and NQ-open are not informative on these sizes.** The sets hold 12, 23 and 37 correct TriviaQA answers
  and 2, 9 and 9 correct NQ-open answers, and the intervals are 0.15 to 0.51 wide, so the ordering of predictors there
  (for example `answer_logprob` above the probe on `gpt2-medium` TriviaQA, 0.823 against 0.773) is within the
  intervals. With 2 correct answers the `gpt2` NQ-open values (such as 0.231 for `p_true`) are not read. No
  increment over the stack on these two sets has an interval that excludes zero.
- **`p_true`** reads 0.445, 0.521 and 0.529 within relation on PopQA, at chance; the model's grade of its own answer
  does not predict correctness on these sizes.

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

Headline cell (unknown side of `obscure_real`), post-trained checkpoints, abstention mode: slope of the share of
abstentions per unit of relative coefficient, for the direction (95 percent interval) and for the three random arms.
The validation AUROC of the direction at the steered layer is 0.943, 0.922, 0.869, 0.795 in the order of the rows.

| Run (steered layer)      | Direction, abstention   | random0 / 1 / 2, abstention | Min gap bound | Distinguishable |
| ------------------------ | ----------------------- | --------------------------- | ------------- | --------------- |
| 1.7b post, fewshot (L24) | 0.040 [-0.031, 0.114]   | 0.388 / -0.017 / 0.095      | -0.481        | no              |
| 0.6b post, fewshot (L28) | 0.114 [0.033, 0.198]    | -0.386 / -0.129 / -0.693    | -0.738        | no              |
| 1.7b post, concept (L12) | -1.112 [-1.224, -1.002] | -0.348 / 0.483 / 0.174      | 0.514         | yes             |
| 0.6b post, concept (L26) | 0.005 [-0.091, 0.102]   | -0.564 / -0.157 / 0.229     | -0.674        | no              |

Headline cell (unknown side of `obscure_real`), base checkpoints, shift mode: slope of the entropy of the next-token
distribution per unit of relative coefficient, same columns. The validation AUROC of the direction is 0.966 and 0.934
in the order of the rows. These slopes are of the entropy, not of a share of abstentions, and are not comparable with
the table above.

| Run (steered layer) | Direction, entropy      | random0 / 1 / 2, entropy | Min gap bound | Distinguishable |
| ------------------- | ----------------------- | ------------------------ | ------------- | --------------- |
| 1.7b base (L20)     | -5.882 [-6.095, -5.655] | 0.155 / 0.941 / 0.023    | 4.722         | yes             |
| 0.6b base (L27)     | -4.099 [-4.360, -3.781] | 0.111 / -0.662 / -0.221  | 3.096         | yes             |

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

#### The GPT-2 sizes (base checkpoints)

P9d on `gpt2-medium`, `gpt2-large` and `gpt2-xl` (runs `p9d/gpt2-medium/e8d925ebaacf` and `p9d/gpt2-large/958cf229514f`,
made at `ca2546c`, and `p9d/gpt2-xl/096a207db665`, made at `a39ae61`) uses the same recipe as the base Qwen3
checkpoints: the `obscure_real` direction added to the residual stream at one layer (the last block of each model),
three random directions of the same norm, the `fewshot` template, and the shift of the entropy of the next-token
distribution (and, on `obscure_real`, of the log-probability of the gold answer) as the behavior, because a GPT-2
checkpoint cannot abstain. There is no P9d on `gpt2` (its gate closed). `gpt2-medium` has no `unknowable_detail` and
`false_premise` rows (fewer than 10 known test items). The units are the entropy of each model's own next-token
distribution, so these slopes are not compared across models here, and they are not comparable with the abstention
slopes of the post-trained Qwen3 checkpoints, which answer a different question.

Headline cell (unknown side of `obscure_real`): slope of the entropy per unit of relative coefficient for the direction
(95 percent interval over the items) and for the three random arms, with the validation AUROC of the direction at the
steered layer. **Distinguishable** means that the direction moves the outcome more than a random direction of the same
norm: the absolute slope of the direction minus the absolute slope of each random arm has a lower bound above zero for
all three (`Min gap bound` is the smallest of the three lower bounds).

| Run (steered layer) | Validation AUROC | Direction, entropy      | random0 / 1 / 2, entropy | Min gap bound | Distinguishable |
| ------------------- | ---------------- | ----------------------- | ------------------------ | ------------- | --------------- |
| medium base (L24)   | 0.954            | -3.728 [-4.020, -3.402] | 0.669 / 0.665 / -0.226   | 2.724         | yes             |
| large base (L36)    | 0.895            | -2.713 [-2.845, -2.572] | -1.024 / 0.100 / -0.014  | 1.524         | yes             |
| xl base (L48)       | 0.949            | -2.815 [-3.087, -2.576] | 0.136 / -0.186 / -0.834  | 1.635         | yes             |

Known items of `obscure_real`, same columns of slopes, to show how much the known items move:

| Run         | Direction, known items  | random0 / 1 / 2, known items |
| ----------- | ----------------------- | ---------------------------- |
| medium base | -0.437 [-1.450, 0.540]  | 1.490 / 2.658 / -0.222       |
| large base  | -1.448 [-2.135, -0.728] | -0.871 / -1.056 / -0.166     |
| xl base     | -1.417 [-2.231, -0.630] | 0.217 / 1.502 / -1.031       |

Selectivity: the absolute mean slope on the unknown items minus that on the known items, with the lower bound above zero
as the criterion. This is a separate claim from the one above. Entropy on every type, and on `obscure_real` also the
log-probability of the gold answer (the other types have no gold answer on the unknown side):

| Run         | Type                | Outcome      | n (k / u) | Known  | Unknown | Gap [95% interval]     | Selective |
| ----------- | ------------------- | ------------ | --------- | ------ | ------- | ---------------------- | --------- |
| medium base | obscure_real        | entropy      | 46 / 100  | -0.437 | -3.728  | 3.292 [2.329, 3.792]   | yes       |
| medium base | obscure_real        | gold logprob | 46 / 100  | -0.902 | -0.635  | -0.267 [-2.022, 1.151] | no        |
| medium base | fictional_generated | entropy      | 46 / 100  | -0.437 | -2.484  | 2.047 [1.069, 2.586]   | yes       |
| large base  | obscure_real        | entropy      | 55 / 100  | -1.448 | -2.713  | 1.265 [0.514, 2.003]   | yes       |
| large base  | obscure_real        | gold logprob | 55 / 100  | -0.046 | -0.683  | 0.638 [-0.247, 0.970]  | no        |
| large base  | fictional_generated | entropy      | 55 / 100  | -1.448 | -2.332  | 0.884 [0.148, 1.673]   | yes       |
| large base  | unknowable_detail   | entropy      | 11 / 60   | -1.725 | -4.327  | 2.602 [1.057, 4.140]   | yes       |
| large base  | false_premise       | entropy      | 11 / 83   | -1.725 | -3.748  | 2.023 [0.463, 3.514]   | yes       |
| xl base     | obscure_real        | entropy      | 78 / 100  | -1.417 | -2.815  | 1.399 [0.582, 2.200]   | yes       |
| xl base     | obscure_real        | gold logprob | 78 / 100  | 0.375  | -1.117  | 0.742 [-0.629, 1.609]  | no        |
| xl base     | fictional_generated | entropy      | 78 / 100  | -1.417 | -2.884  | 1.467 [0.595, 2.305]   | yes       |
| xl base     | unknowable_detail   | entropy      | 21 / 60   | -1.536 | -5.365  | 3.829 [2.606, 5.161]   | yes       |
| xl base     | false_premise       | entropy      | 21 / 83   | -1.536 | -4.344  | 2.808 [1.533, 4.097]   | yes       |

- **Distinguishable on entropy, on all three sizes.** The direction moves the entropy of the unknown questions by
  -3.728, -2.713 and -2.815 per unit coefficient, and the largest absolute slope of a random arm is 0.669, 1.024 and
  0.834. The smallest gap bounds are 2.724, 1.524 and 1.635. The distinction rests on the unknown side only; see the
  known items below.
- **Selective on entropy, as a separate result.** The headline gap is 3.292 [2.329, 3.792] (medium), 1.265 [0.514,
  2.003] (large) and 1.399 [0.582, 2.200] (xl), and `fictional_generated` is selective on all three. On `gpt2-large` and
  `gpt2-xl` `unknowable_detail` and `false_premise` are selective too (gaps 2.602 and 2.023, 3.829 and 2.808). The
  evidence is a slope gap, and it is smaller on `gpt2-large` and `gpt2-xl` than on `gpt2-medium`: the lower bound of the
  headline gap is 2.329, 0.514 and 0.582.
- **The known items move too.** On `gpt2-large` and `gpt2-xl` the direction changes the entropy of the known questions
  at -1.448 [-2.135, -0.728] and -1.417 [-2.231, -0.630], about half of the unknown-side slope, so the effect is larger
  on the unknown side and not absent on the known side. On `gpt2-medium` the known slope is -0.437 [-1.450, 0.540],
  while two random arms move the known items by +1.490 and +2.658 and the third by -0.222: a random direction of the
  same norm moves the known questions of that model more than the direction does, and on `gpt2-xl` one random arm does
  (+1.502). A slope-based gap does not describe this, as for the Qwen3 base checkpoints above.
- **Not distinguishable and not selective on the gold-answer log-probability.** The direction moves the log-probability
  of the gold answer of the unknown `obscure_real` questions by -0.635 [-1.620, 0.375], -0.683 [-1.153, -0.210] and
  -1.117 [-1.985, -0.279], and the gaps against the known items are -0.267 [-2.022, 1.151], 0.638 [-0.247, 0.970] and
  0.742 [-0.629, 1.609]: all three include zero. The absolute gaps against the three random arms also include zero (for
  example 0.396 [-0.523, 1.350] against `random0` on `gpt2-medium`; on `gpt2-xl` the lower bounds against the three arms
  are -0.006, 0.008 and -0.272). The entropy results do not carry over to the probability of the correct answer.
- **What this does and does not show.** The direction changes the entropy of the next-token distribution of a base model
  more than a random direction does and more on unknown questions than on known ones. It is not shown to be a control of
  what the model knows: the gold-answer probability does not move distinguishably, random directions also move the
  entropy on the known items on `gpt2-medium` and `gpt2-xl`, and the stored selectivity of the `gpt2-xl` run, made
  before the interval was oriented, agrees with a recomputation with the current code (every bound within 0.005). The
  runs are one per cell.

## Run identity

All reading runs (R1 to R7) and the A1 and A2 runs were made at commit `a39ae61` with a clean tree, on one laptop GPU
with 8 GB of memory, with the default config of each experiment (batch size 8). The A1 and A2 runs read the `gpt2`
runs of R1, R2, R3, R5 and R7 below. The C1 runs of `gpt2` and `gpt2-xl` were made at `d59a2f7` and those of
`gpt2-medium` and `gpt2-large` at `a39ae61`; the C1 and reading source code is unchanged between them (`git diff
--stat` over `src` lists only `p9d_steering.py` and `p9/behavior.py`). The C1 batch size is 16 for `gpt2` and
`gpt2-medium` and 4 for `gpt2-large` and `gpt2-xl`. The Qwen3 C1 runs were made at `a39ae61` with a clean tree and
batch size 16 on all four checkpoints. The P1 runs of the five models were made at `a39ae61` with a clean tree, with
the sentiment task only, 10 seeds of 64 training pairs, up to 256 evaluation pairs per split (214 available) and batch
size 8. The P6 tables were recomputed on the CPU from the finished runs at commit `ca2546c` (no model is loaded). The
P8 run was recomputed on the CPU at commit `beb5801` (clean tree, no model is loaded), after the runs it reads. The
clean tree refers to the code: P8 read the run folders from the local output root, and the runs it reads are committed
under `results/`. Its config names the five P1 runs as extra runs. Library versions and the device are in each run's
`meta.json`.

The R4 and R6 runs of `gpt2-medium`, `gpt2-large` and `gpt2-xl`, the second P1 run of `gpt2` (sentiment, truth and
emotion tasks), and the P2, P3, P4, M1, C2, LR1, C4, P7 and C5 runs were made at `ca2546c` with a clean tree on the
same laptop GPU, with the default config of each experiment (the batch size of each run is in its `config.json`). The
P1 run `078b9d6482a2` differs from the sentiment run `673af340b77d` in the task list only. The C4 run reads the LR1
adapter named in its config. None of the M1, C2, P7, C4 and C5 runs stores the perplexity or the distinct-1 of the
generated text. The P5 runs on `gpt2` did not finish (the fit of one condition produced a direction of zero norm, and
both attempts stopped), so they are not exported and not reported.

The P9 runs were made on the same laptop GPU with a clean tree: P9a at `ff1177a` (batch size 8, 4 for `gpt2-xl`), P9b
and P9c at `95a8243` (batch size 8, 4 for `gpt2-xl`) and P9d at `a39ae61` (batch size 8). The P9 runs of `gpt2`,
`gpt2-medium` and `gpt2-large` (P9a to P9c, and P9d for the two larger ones) were made at `ca2546c` on the same laptop
GPU. Each P9b, P9c and P9d run reads the P9a run of the same model, which is the one in the P9a rows below. The
selectivity columns of the P9d tables of the Qwen3 checkpoints were recomputed offline from the `item_slopes` of the
runs with the current code, not read from the stored `table_selectivity`; the GPT-2 tables are the stored ones, which
agree with a recomputation (every bound within 0.005). Earlier P9 runs exist, made with the four-relation label pool or
before the within-relation headline, the item resampling of the intervals and the dose-slope headline; they are
superseded and are not reported.

Config hashes (12 hex digits), GPT-2:

| Experiment     | gpt2         | gpt2-medium  | gpt2-large   | gpt2-xl      |
| -------------- | ------------ | ------------ | ------------ | ------------ |
| R1             | bd4438c46cef | 01fd70a05fa8 | b64b756fad46 | 81e8ecd71341 |
| R2             | e1821ff197ce | 098a48a58a83 | 1a2679b49855 | 83124998c2b8 |
| R3             | a9f7b2a71c22 | d10ebba10935 | 5d0f4f92fc9b | 3507b0628650 |
| R4             | 4702fcb81ed3 | 981d9dd9916e | 788316a35990 | bee62b37444a |
| R5             | 779f5b093803 | e447c8958b80 | 322bd80f6d19 | 7f0afd7dfe12 |
| R6             | 47e8ec0cf46c | 04fc7ead5f13 | f058c0f19396 | 60cc952b4ec5 |
| R7             | 5908a78acf64 | 3030e4186690 | 796f4c4d19f2 | 5eed122bfd4e |
| P1 (sentiment) | 673af340b77d | not run      | not run      | not run      |
| P1 (all tasks) | 078b9d6482a2 | not run      | not run      | not run      |
| P2             | ee099a704022 | not run      | not run      | not run      |
| P3             | 676c31488397 | not run      | not run      | not run      |
| P4             | 8c4a73d1e9db | not run      | not run      | not run      |
| P5             | not finished | not run      | not run      | not run      |
| P7             | 6c19128c5192 | not run      | not run      | not run      |
| C1             | add59c39f574 | 2ed2e53ba7fe | 353f91e1b8dc | de76af1a8a07 |
| M1             | 4e394ed1a72c | not run      | not run      | not run      |
| C2             | 85c9a670bf17 | not run      | not run      | not run      |
| C4             | 1b571ac45420 | not run      | not run      | not run      |
| C5             | not run      | not run      | not run      | 155f0f72bc3f |
| LR1            | 1e9cd6f56388 | not run      | not run      | not run      |
| P9a            | 5461200f995c | 4272635aa065 | 59c9bfd59dfd | cc283b36fcae |
| P9b            | fd3bdc5a4b0d | a5a9c7d7b176 | 7b3fd5d1bfb5 | 92664dccb7d9 |
| P9c            | 7ed884a2e8df | b3d936ab3f3b | fb5b8a90c085 | 1839575c0eaa |
| P9d            | gate closed  | e8d925ebaacf | 958cf229514f | 096a207db665 |

Qwen3:

| Experiment  | qwen3-0.6b-base | qwen3-0.6b   | qwen3-1.7b-base | qwen3-1.7b   |
| ----------- | --------------- | ------------ | --------------- | ------------ |
| R1          | a104feb4f38d    | bd2a906798a0 | 41a75ca84cb0    | f5f06e078f69 |
| R2          | a2adb2792d96    | 2106339f9ff7 | b59c0e034df2    | beca01593887 |
| R3          | c917a73220ea    | cca73836c567 | 84e49fb14d61    | 87868da1fb5b |
| R5          | e2d8b07cd376    | cffc23e593d0 | a5694ff54daa    | 487bd86230f5 |
| R7          | ce3e5ce2ec50    | 9cb5abfedb1b | 66d496ff00fa    | b7d78995c5fc |
| P1          | a540a2497ac4    | 31a6ee65d6ab | ef3f8c0d3389    | f4aa9b57824a |
| C1          | 7ae3e2aa34e2    | 832462b7b04e | 0cd5a36d3408    | a91f950ad9f5 |
| P9a         | 1d9f881153e4    | 10d95352866e | d5d522bfecd4    | f413efcae268 |
| P9b         | 34c0ebff50b0    | not run      | 5ca7821318ef    | not run      |
| P9c         | 264bf7ca3a01    | not run      | 0f684b3c4b66    | not run      |
| P9d fewshot | f9a8f827b6b4    | 9c94025d9ad4 | 9543312bcc45    | 42b7a0daf9b9 |
| P9d concept | not run         | f0b3f4eb3988 | not run         | d43b9e297e42 |

Other runs: R6 text baselines of `gpt2` computed at `ca2546c` (not a run); A1 `ad6abbba8fd0`; A2 `be4e34f21b35`; P6
`25ee6de331d4`; P8 `e9f9bc04ba3c`.

The files of every run named here are committed under `results/<experiment>/<model>/<hash>/`, copied with
`tomography export`, which keeps the JSON, NPZ and CSV files the plots read and rewrites local paths. The figures are
not committed; `tomography plot all --out results` redraws them from these files.

## Pending

Not in this document yet. No dates are promised.

- R4 and R6 on the Qwen3 checkpoints, and a rerun of R6 on `gpt2` that writes the text baselines itself (the three
  larger GPT-2 runs store them).
- P1 on the truth and emotion tasks on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the Qwen3 checkpoints, and with the
  `pca_diff` finder (the truth and emotion tasks ran on `gpt2` only; the sentiment task ran with the logistic finder on
  `gpt2` and the four Qwen3 checkpoints).
- P2, P3, P4, P7, M1, C2, C4 and LR1 on the other GPT-2 sizes and on the Qwen3 checkpoints (all ran on `gpt2` only), and
  C5 on any model other than `gpt2-xl`.
- P5, persona wording and instruction confounds for honesty: the `gpt2` runs did not finish (a fit produced a direction
  of zero norm), so the experiment has to be rerun before any result is reported.
- A random-direction arm for C2, P7 and C5, and a fluency measure (perplexity and distinct-1) for M1, C2, P7, C4 and
  C5.
- P9b and P9c on the post-trained Qwen3 checkpoints.
- P9: a hand check of the abstention indicator of P9d (the stored samples carry no human labels), an item-level
  interval for the P9b AUROC (only the interval over the pair draws is stored), and a within-relation increment of
  the probe over the output confidence in P9c (only the pooled increment is stored).
- A5, reading directions against Qwen-Scope SAE decoders.

