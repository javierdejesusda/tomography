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
files), the P7 counts of inverted units by layer on the Qwen3 1.7B pair (recomputed from the per-seed curves), the random and shuffled-label values at the selected layer of R4 on Qwen3, the maxima over layers of the R6
random series on Qwen3, the layer ranges of A1 and the A2 separations on Qwen3 and on the three larger GPT-2 sizes (all computed from the curves files), the
C2 values on Qwen3 (read from the curves and table files), the C4 reader means and ranges on Qwen3 and on the GPT-2 sizes (computed from the
table rows), the fluency means, ratios and counts of cells past the rule and the random-arm ranges
of the runs made at `f4d414d` (computed from the per-cell rows and the curves files), the P7 counts of inverted
units on the GPT-2 sizes (recomputed from the per-seed curves), and differences or ranges quoted in the text. The P8 tables are read from the P8 run, which aggregates
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
(C1)](#sentiment-steering-on-qwen3-c1) 11. [Other steering experiments on gpt2 and the larger GPT-2 sizes (M1, C2, P7, LR1, C4,
C5)](#other-steering-experiments-on-gpt2-and-the-larger-gpt-2-sizes-m1-c2-p7-lr1-c4-c5) 12. [Base against post-trained Qwen3
(P8)](#base-against-post-trained-qwen3-p8) 13. [Known versus unknown questions (P9)](#known-versus-unknown-questions-p9)
14. [Qwen3: emotions, TruthfulQA, seed agreement, logit lens, steering and control (R4, R6, A1, A2, M1, C2, C4, C5,
LR1)](#qwen3-emotions-truthfulqa-seed-agreement-logit-lens-steering-and-control-r4-r6-a1-a2-m1-c2-c4-c5-lr1) 15. [Position
and stimulus-set size on qwen3-1.7b-base and qwen3-1.7b (P2,
P3)](#position-and-stimulus-set-size-on-qwen3-17b-base-and-qwen3-17b-p2-p3) 16. [Persona wording and instruction confounds
(P5)](#persona-wording-and-instruction-confounds-p5) 17. [Template steering on the Qwen3 1.7B pair
(P7)](#template-steering-on-the-qwen3-17b-pair-p7) 18. [Run identity](#run-identity) 19. [Pending](#pending)

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
  happiness 0.700 on `gpt2-medium`, disgust 0.956 to 0.995 on all four sizes). `gpt2-xl` surprise is selected at L1
  (0.894) and `gpt2-xl` disgust at L3, where the curve is flat. A high accuracy at these layers is consistent with
  emotion words in the statements that a shallow reader can use; this is not tested here.
- **Baselines at the selected layer.** The random-direction series is at 0.558 to 0.703 on `gpt2`, 0.534 to 0.613 on
  `gpt2-medium`, 0.587 to 0.639 on `gpt2-large` and 0.547 to 0.620 on `gpt2-xl`, and the shuffled-label series at 0.477
  to 0.566, 0.486 to 0.537, 0.489 to 0.531 and 0.477 to 0.533 (computed from the curves files). Validation accuracy is
  above test accuracy by at most 0.062 on `gpt2`, 0.035 on `gpt2-medium`, 0.040 on `gpt2-large` and 0.057 on
  `gpt2-xl` (it is below it for some cells).

## LAT reading on Qwen3

Same protocol on the base and post-trained checkpoints of Qwen3 0.6B and 1.7B (the 4B pair is in its own subsection
below the 1.7B bullets). The post-trained checkpoints use the chat
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

### The 4B pair

The same protocol on `qwen3-4b-base` and `qwen3-4b` (36 blocks; runs `r1/qwen3-4b-base/705ebc7745d4`,
`r1/qwen3-4b/15a1ed3eab2b`, `r2/qwen3-4b-base/bb176283725a`, `r2/qwen3-4b/a6ab527e0544`,
`r3/qwen3-4b-base/a6e29d000606`, `r3/qwen3-4b/039d5c50bb27`, `r5/qwen3-4b-base/d692205ca9bf`,
`r5/qwen3-4b/1630de90cae8`, `r7/qwen3-4b-base/15d7a0c1017f` and `r7/qwen3-4b/9f28dde7631f`), made at `89ea7cb` on a
rented GPU pod (see [Run identity](#run-identity)). The post-trained checkpoint uses the chat template with thinking
disabled and the base checkpoint plain tags, as above. Test accuracy with its interval over the 10 readers; the layer and
the validation accuracy at that layer are in the rows below it.

| Experiment               | qwen3-4b-base        | qwen3-4b             |
| ------------------------ | -------------------- | -------------------- |
| R1 sentiment             | 0.905 [0.887, 0.920] | 0.760 [0.727, 0.792] |
|   selected layer         | L22                  | L23                  |
| R2 utility test          | 0.772 [0.756, 0.784] | 0.794 [0.786, 0.801] |
|   selected layer         | L22                  | L21                  |
|   validation accuracy    | 0.827                | 0.846                |
| R3 morality test         | 0.917 [0.912, 0.921] | 0.940 [0.936, 0.943] |
|   selected layer         | L24                  | L23                  |
|   validation accuracy    | 0.914                | 0.928                |
| R5 transfer, exploratory | 0.658 [0.639, 0.676] | 0.758 [0.727, 0.795] |
|   selected layer         | L36                  | L30                  |
|   validation accuracy    | 0.742                | 0.804                |
| R7 truth test            | 0.942 [0.937, 0.947] | 0.934 [0.922, 0.946] |
|   selected layer         | L26                  | L23                  |
|   validation accuracy    | 0.960                | 0.923                |

- **R1 sentiment.** The base checkpoint reaches 0.905 [0.887, 0.920] and the post-trained one 0.760 [0.727, 0.792], a
  gap of 0.145 with non-overlapping intervals. The base value equals that of `qwen3-1.7b-base` (0.905) and the
  post-trained value is the lowest R1 of the six Qwen3 checkpoints (the others are 0.857 to 0.947), so size does not
  order R1 on the post-trained checkpoints (0.857, 0.866, 0.760). R1 has no test split, so both values are optimistic by
  the selection.
- **R2 utility and R3 morality.** R2 is 0.772 and 0.794 and R3 is 0.917 and 0.940, above the 1.7B values of both
  experiments (R2 0.768 and 0.756, R3 0.896 and 0.892). At 4B the post-trained checkpoint is above the base one in both
  (by 0.022 and 0.023, intervals not overlapping), the opposite sign from 0.6B and 1.7B, where the base checkpoint is at
  or above the post-trained one in both experiments.
- **R5 honesty transfer.** The transfer test is 0.658 [0.639, 0.676] (base, selected at the last block L36) and 0.758
  [0.727, 0.795] (post-trained), with validation accuracy 0.742 and 0.804 at the selected layer. The base value is below
  the 1.7B base value (0.778) and the post-trained value is near the 1.7B one (0.781). The test set has 59 pairs that
  the intervals do not cover, and the test value is below the validation value by 0.084 and 0.046.
- **R7 truth.** The test accuracy is 0.942 [0.937, 0.947] and 0.934 [0.922, 0.946] at L26 and L23, with validation
  accuracy 0.960 and 0.923. The two values are within 0.01 of each other with overlapping intervals, and equal to the
  1.7B values (0.934 and 0.937) within 0.008.

R7 on the out-of-distribution sets:

| Set (pairs)           | qwen3-4b-base        | qwen3-4b             |
| --------------------- | -------------------- | -------------------- |
| cities (748)          | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| neg_cities (748)      | 0.998 [0.997, 0.998] | 0.991 [0.989, 0.992] |
| sp_en_trans (177)     | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| neg_sp_en_trans (177) | 0.988 [0.985, 0.991] | 0.892 [0.885, 0.899] |
| larger_than (990)     | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| smaller_than (990)    | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| cities_conj (748)     | 0.992 [0.992, 0.993] | 0.990 [0.989, 0.991] |
| cities_disj (748)     | 0.960 [0.958, 0.962] | 0.929 [0.923, 0.935] |
| common_claim (2,225)  | 0.871 [0.870, 0.872] | 0.839 [0.835, 0.844] |

- The four plain sets `cities`, `sp_en_trans`, `larger_than` and `smaller_than` are at 1.000 [1.000, 1.000] on both
  checkpoints, so all 10 readers scored every pair correctly there. The negated and compound sets are high as well
  (`neg_cities` 0.998 and 0.991, `cities_conj` 0.992 and 0.990, `cities_disj` 0.960 and 0.929).
- `neg_sp_en_trans` is the set that fell to or below chance at 1.7B (0.508 base and 0.238 post-trained) and below chance
  on the 0.6B post-trained checkpoint (0.337). At 4B it is 0.988 [0.985, 0.991] on the base checkpoint and 0.892 [0.885,
  0.899] on the post-trained one, both far above chance. No run here tests why.
- On the five sets that are below ceiling the base checkpoint is above the post-trained one, with non-overlapping
  intervals (for example `neg_sp_en_trans` 0.988 against 0.892 and `common_claim` 0.871 against 0.839). `common_claim`
  is above the 1.7B values (0.812 and 0.791).

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

### The other GPT-2 sizes (A1, A2)

A1 and A2 on `gpt2-medium`, `gpt2-large` and `gpt2-xl` (runs `a1/gpt2-medium/d1eeb0585ca6`, `a1/gpt2-large/3ed23ece4e63`, `a1/gpt2-xl/078a72de1cfb` and `a2/gpt2-medium/6c8366625d7a`, `a2/gpt2-large/b6cf80cfb26f`, `a2/gpt2-xl/46d59dbefe95`, made at `fb3b5dd` with a clean tree on a laptop GPU) read the R1, R2, R3, R5 and R7 runs of the same size (hashes under Run identity) with the same recipe as above, and run no forward pass over data. The `gpt2` column repeats the values above so that the four sizes sit side by side. The A1 layer is relative depth 0.5 (L6, L12, L18 and L24 of 12, 24, 36 and 48 blocks); two random unit directions have a mean absolute cosine of 0.029, 0.025, 0.022 and 0.020 in the 768, 1,024, 1,280 and 1,600 dimensions.

A1, mean cosine between the directions of the 10 seeds at the middle layer, with the 95 percent t-interval over seeds and in parentheses the range of the mean over all layers:

| Reading | gpt2                                  | gpt2-medium                           | gpt2-large                            | gpt2-xl                               |
| ------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| r1      | 0.447 [0.395, 0.498] (0.363 to 0.520) | 0.431 [0.369, 0.493] (0.326 to 0.457) | 0.469 [0.423, 0.514] (0.279 to 0.531) | 0.504 [0.467, 0.542] (0.339 to 0.504) |
| r2      | 0.288 [0.205, 0.372] (0.171 to 0.356) | 0.302 [0.215, 0.389] (0.210 to 0.409) | 0.349 [0.263, 0.436] (0.107 to 0.388) | 0.369 [0.285, 0.452] (0.112 to 0.522) |
| r3      | 0.252 [0.163, 0.341] (0.163 to 0.290) | 0.257 [0.162, 0.352] (0.164 to 0.392) | 0.395 [0.319, 0.471] (0.133 to 0.445) | 0.479 [0.398, 0.561] (0.141 to 0.520) |
| r5      | 0.986 [0.985, 0.987] (0.939 to 0.986) | 0.983 [0.982, 0.985] (0.904 to 0.985) | 0.984 [0.983, 0.986] (0.975 to 0.990) | 0.980 [0.978, 0.981] (0.964 to 0.986) |
| r7      | 0.525 [0.487, 0.563] (0.450 to 0.592) | 0.583 [0.537, 0.629] (0.449 to 0.619) | 0.579 [0.538, 0.619] (0.438 to 0.587) | 0.607 [0.576, 0.638] (0.445 to 0.609) |

A2, best layer, separation z with its 95 percent bootstrap interval over the 10 seeds, and in parentheses the mean and standard deviation of the same quantity for 100 random directions (the best layer is chosen on the same curve, so the values are optimistic):

| Reading | gpt2                                | gpt2-medium                         | gpt2-large                          | gpt2-xl                             |
| ------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| r1      | L12 3.33 [3.17, 3.49] (-0.02, 0.52) | L20 3.50 [3.25, 3.73] (0.05, 0.52)  | L32 4.08 [3.87, 4.27] (-0.07, 0.53) | L47 3.92 [3.65, 4.17] (-0.07, 0.51) |
| r2      | L12 2.87 [2.36, 3.36] (-0.02, 0.54) | L24 2.34 [1.97, 2.71] (-0.01, 0.48) | L35 2.65 [2.17, 3.08] (-0.00, 0.59) | L37 3.18 [2.71, 3.69] (-0.10, 0.50) |
| r3      | L7 0.91 [0.53, 1.27] (0.03, 0.51)   | L21 1.16 [1.01, 1.32] (0.07, 0.49)  | L11 0.85 [0.58, 1.13] (0.10, 0.53)  | L5 1.03 [0.62, 1.43] (-0.15, 0.52)  |
| r5      | L7 1.68 [1.64, 1.74] (-0.08, 0.56)  | L10 1.11 [1.07, 1.13] (0.04, 0.74)  | L32 2.68 [2.60, 2.74] (-0.06, 0.70) | L44 2.48 [2.41, 2.54] (0.00, 0.62)  |
| r7      | L8 0.89 [0.69, 1.06] (-0.01, 0.54)  | L8 0.60 [0.35, 0.80] (0.04, 0.54)   | L4 1.02 [0.78, 1.25] (-0.04, 0.53)  | L43 0.67 [0.44, 0.90] (-0.06, 0.63) |

- **All 20 A1 values are far above the random level, and the directions depend on the training draw.** The lowest value is 0.252 (R3 on `gpt2`) against a random level of 0.020 to 0.029. R1, R2, R3 and R7 read between 0.252 and 0.607 at the middle layer across the four sizes, as the Qwen3 checkpoints do at layer 14 (R1 0.423 to 0.578, R2 0.306 to 0.389, R3 0.335 to 0.497, R7 0.562 to 0.645; the GPT-2 ranges are 0.431 to 0.504, 0.288 to 0.369, 0.252 to 0.479 and 0.525 to 0.607).
- **Agreement rises with size for two readings.** The point estimates of R2 (0.288, 0.302, 0.349, 0.369) and R3 (0.252, 0.257, 0.395, 0.479) rise at every step, and the intervals of R3 on `gpt2` [0.163, 0.341] and on `gpt2-xl` [0.398, 0.561] do not overlap. R1 (0.447, 0.431, 0.469, 0.504) and R7 (0.525, 0.583, 0.579, 0.607) do not rise at every step and R5 stays near 0.98, ending below where it starts (0.986, 0.983, 0.984, 0.980). Each size is one run, so no step between neighbouring sizes is shown to be larger than the variation between runs.
- **R5 agrees at 0.980 to 0.986 at the middle layer on every size, and at 0.904 or above at every layer** (lowest layer value 0.939, 0.904, 0.975 and 0.964 on the four sizes), while its transfer test accuracy is 0.386, 0.598, 0.566 and 0.492 (see the reading tables). A direction that is stable across seeds is not for that reason a direction that transfers, as on `gpt2` and on the Qwen3 checkpoints.
- **R1 and R2 separate their pole tokens far above the random spread on every size.** R1 reads 3.33, 3.50, 4.08 and 3.92 against a random standard deviation of 0.51 to 0.53, and R2 2.87, 2.34, 2.65 and 3.18 against 0.48 to 0.59. R5 reads 1.68, 1.11, 2.68 and 2.48; on `gpt2-medium` its 1.11 [1.07, 1.13] is 1.5 random standard deviations (0.74), below two, while on `gpt2-large` and `gpt2-xl` it is above two.
- **R3 and R7 stay small, as on `gpt2`.** R3 reads 0.91, 1.16, 0.85 and 1.03 and R7 0.89, 0.60, 1.02 and 0.67. In 7 of the 8 values the separation is below two random standard deviations (1.02, 0.99, 1.06 and 1.04 for R3; 1.08, 1.07, 1.06 and 1.26 for R7); the exception is R3 on `gpt2-medium` (1.16 [1.01, 1.32] against 0.99). The Qwen3 values are 3.13 to 4.74 for R3 and 2.65 to 5.86 for R7, so none of the four GPT-2 sizes reaches the lowest of them. The best layer of R3 and R7 jumps between sizes (R3: L7, L21, L11, L5; R7: L8, L8, L4, L43), which is what a maximum over a flat curve of small separation looks like, and is not read as a layer location.
- **What this does not show.** A high pole separation is not read as a reading of the concept in the vocabulary (see R5 on `qwen3-1.7b` below), the interval covers the 10 seeds of one run per size, and the best layer is chosen on the curve it is read from.

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

## Other steering experiments on gpt2 and the larger GPT-2 sizes (M1, C2, P7, LR1, C4, C5)

Six further runs on `gpt2` that steer or edit the model along a reading direction, all made at `ca2546c` with a clean
tree (C5 on `gpt2-xl`), are reported first. **None of these runs stores the perplexity or the distinct-1 of the generated
text.** The C1
sections show that a high steered score at a large coefficient can come with degenerate text: on `gpt2` the C1
perplexity is 139.3 at +0.2 and 438.0 at +0.3 against 53.6 without steering, and the distinct-1 at +0.3 is between 0.043
and 0.099 across the four sizes. These runs use coefficients of the same relative range (up to 0.3), and nothing in
their files shows whether the steered text is fluent. No score in this section is called fluent or read as steering that
works. Only M1 has a random-direction arm; C2, P7, LR1, C4 and C5 do not. M1, C2, P7 and C5 were later run on
`gpt2-large` and `gpt2-xl` at `f4d414d`, with random-direction arms and the fluency measure; they are reported in
[The larger GPT-2 sizes (M1, C2, P7, C5)](#the-larger-gpt-2-sizes-m1-c2-p7-c5), after the `gpt2` runs. LR1 and C4 were
later run on `gpt2-medium`, `gpt2-large` and `gpt2-xl` (chain 17, `89ea7cb`), C4 with random-direction arms and the
perplexity of neutral text; they are reported at the end of that section, under
[LR1 and C4 on gpt2-medium, gpt2-large and gpt2-xl](#lr1-and-c4-on-gpt2-medium-gpt2-large-and-gpt2-xl).

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
- **The other sizes.** LR1 and C4 on `gpt2-medium`, `gpt2-large` and `gpt2-xl`, with random-direction arms and the
  perplexity of neutral text, are in [LR1 and C4 on gpt2-medium, gpt2-large and
  gpt2-xl](#lr1-and-c4-on-gpt2-medium-gpt2-large-and-gpt2-xl).

### Lie gallery (C5)

This is the earlier C5 run on `gpt2-xl`, made at `ca2546c` on a laptop GPU. The run reported for `gpt2-xl` in [The larger GPT-2 sizes (M1, C2, P7, C5)](#the-larger-gpt-2-sizes-m1-c2-p7-c5) is `c5/gpt2-xl/0ad95833261d`, which has a random-direction arm and the fluency measure and reads the C2 run of that section; this run stays tracked, and the numbers and limits below are those of this earlier run.

C5 on `gpt2-xl` (run `c5/gpt2-xl/155f0f72bc3f`; 1,000 TriviaQA questions, a 15-shot prompt, one sample per question at
temperature 0.8, relative strength 0.1, steered over layers 13, 17, 22, 27, 31, 35, 40 and 45) records the share of
correct answers without steering and under an honesty-steered and a lie-steered arm: 0.147, 0.107 and 0.080. Of the 147
questions answered correctly without steering, 77 are answered wrongly under the lie arm, a flip rate of 0.524 [0.443,
0.603]. The honesty-steered arm is also below the unsteered one, and there is no random-direction arm, so the flips do
not separate an effect of the lie direction from a generic perturbation of the model. No perplexity or distinct-1 is
stored. This run has no emotion gallery.

### The larger GPT-2 sizes (M1, C2, P7, C5)

M1, C2, P7 and C5 also ran on `gpt2-large` and `gpt2-xl` (runs named under each experiment below), all made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment (batch size 16; C5 with 1,000 questions in the pool, as in the earlier C5 runs). Unlike the `gpt2` runs above, they carry the controls that this section says the earlier runs lack: a random-direction arm (unit directions norm-matched to the steering arm; M1 already had one) and a fluency measure, the perplexity of held-out neutral text under the steered model and the distinct-1 of the generated text, both from `table_effects`. The two sizes are two models: the tables are separate and no metric is pooled over them. To read the cost next to the score, the descriptive rule of [Sentiment steering on Qwen3](#sentiment-steering-on-qwen3-c1) is used: a setting is called fluent when its mean perplexity is at most 1.5 times the unsteered one and its mean distinct-1 is at least 0.15. It is a rule chosen for this document and not a threshold of the code. LR1 and C4 also ran on `gpt2-medium`,
`gpt2-large` and `gpt2-xl`, in chain 17 at `89ea7cb` on an RTX A6000 pod; they are described under their own heading at
the end of this section.

#### M1 on gpt2-large and gpt2-xl

M1 reads utility (ETHICS) with six finders and compares them on held-out reading, on steering and on erasure, as on `gpt2` (runs `m1/gpt2-large/b4995274d1f8` and `m1/gpt2-xl/3309d5fe55b7`; 10 readers per finder, the first 3 used for steering, 250 prompts, one sample per prompt, relative strengths 0.1, 0.2 and 0.3). The steered layers are 14, 20 and 25 on `gpt2-large` and 19, 26 and 34 on `gpt2-xl`. The win rate is the share of (prompt, reader) pairs, 750 per cell (748 or 749 where a pair was lost), in which the completion steered with +s is scored higher by the primary sentiment judge than the completion steered with -s on the same seed, so 0.5 means no effect. The prompts are shared by all finders and readers, so the Wilson intervals are narrower than the variation between prompt draws. Test accuracy is on 4,808 pairs, at the layer of each finder.

`gpt2-large`:

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.558 [0.529, 0.574] | L20   | 0.479 [0.443, 0.514] | 0.517 [0.481, 0.552] | 0.563 [0.527, 0.598] | 0.572 [0.536, 0.607] |
| pca_raw   | 0.571 [0.569, 0.572] | L18   | 0.461 [0.425, 0.496] | 0.539 [0.503, 0.574] | 0.537 [0.502, 0.573] | 0.533 [0.498, 0.569] |
| mean_diff | 0.659 [0.641, 0.676] | L36   | 0.529 [0.493, 0.564] | 0.564 [0.528, 0.599] | 0.600 [0.565, 0.634] | 0.605 [0.570, 0.640] |
| logistic  | 0.680 [0.664, 0.694] | L36   | 0.537 [0.502, 0.573] | 0.497 [0.462, 0.533] | 0.540 [0.504, 0.575] | 0.503 [0.467, 0.538] |
| kmeans    | 0.569 [0.566, 0.572] | L20   | 0.459 [0.424, 0.495] | 0.515 [0.479, 0.550] | 0.533 [0.498, 0.569] | 0.536 [0.500, 0.571] |
| random    | 0.549 [0.522, 0.575] | L32   | 0.493 [0.457, 0.528] | 0.477 [0.442, 0.513] | 0.441 [0.406, 0.477] | 0.489 [0.454, 0.525] |

`gpt2-xl`:

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.565 [0.532, 0.584] | L26   | 0.565 [0.529, 0.600] | 0.683 [0.648, 0.715] | 0.755 [0.723, 0.784] | 0.713 [0.680, 0.745] |
| pca_raw   | 0.562 [0.560, 0.563] | L18   | 0.568 [0.532, 0.603] | 0.601 [0.566, 0.636] | 0.716 [0.683, 0.747] | 0.664 [0.629, 0.697] |
| mean_diff | 0.656 [0.641, 0.671] | L29   | 0.572 [0.536, 0.607] | 0.643 [0.608, 0.676] | 0.727 [0.694, 0.757] | 0.739 [0.706, 0.769] |
| logistic  | 0.694 [0.684, 0.705] | L33   | 0.555 [0.519, 0.590] | 0.604 [0.569, 0.638] | 0.649 [0.615, 0.683] | 0.703 [0.669, 0.734] |
| kmeans    | 0.553 [0.526, 0.567] | L22   | 0.520 [0.484, 0.556] | 0.593 [0.558, 0.628] | 0.611 [0.575, 0.645] | 0.583 [0.547, 0.617] |
| random    | 0.550 [0.532, 0.567] | L27   | 0.477 [0.442, 0.513] | 0.484 [0.448, 0.520] | 0.492 [0.456, 0.528] | 0.488 [0.452, 0.524] |

- **Steering is weak on `gpt2-large` and moderate on `gpt2-xl`, and reading does not order the finders.** On `gpt2-large` the best win rate at 0.3 is 0.600 [0.565, 0.634] (`mean_diff`), and the finder that reads best, `logistic` (0.680 test accuracy), wins 0.540 [0.504, 0.575] at 0.3 and 0.497 [0.462, 0.533] at 0.2; the check judge puts it at 0.503 [0.467, 0.538]. The three finders that read level with the random direction (`pca_diff` 0.558, `kmeans` 0.569, `pca_raw` 0.571, against 0.549) win 0.533 to 0.563 at 0.3, and the interval of `kmeans` ([0.498, 0.569]) includes 0.5. On `gpt2-xl` every finder is above 0.6 at 0.3, in the order `pca_diff` (0.755), `mean_diff` (0.727), `pca_raw` (0.716), `logistic` (0.649) and `kmeans` (0.611); `logistic` reads best (0.694) and steers fourth of the five. As on `gpt2`, `pca_diff` steers more than it reads: its test accuracy on `gpt2-xl` is 0.565, level with the random direction (0.550), and its win rate is the highest of the six.
- **Steering falls from `gpt2` to `gpt2-large` and partly recovers at `gpt2-xl`, while reading rises.** The test accuracy of `logistic` is 0.655, 0.680 and 0.694 on `gpt2`, `gpt2-large` and `gpt2-xl` and its win rate at 0.3 is 0.925, 0.540 and 0.649. The best win rate of any finder at 0.3 is 0.925, 0.600 and 0.755. The layers differ in index between the sizes and each size is one run, so the run does not say why the win rate falls and partly recovers.
- **The random direction.** On `gpt2-large` its win rate is below 0.5 at every strength and falls with it (0.493, 0.477, 0.441 [0.406, 0.477]), as on `gpt2` (0.491, 0.456, 0.432), so a random edit of the same relative size moves the completions the other way on average and a finder's win rate is read against the random one of its own run. On `gpt2-xl` it is 0.477, 0.484 and 0.492 and every interval includes 0.5. At 0.3 the finders are above the random direction by 0.092 (`kmeans`) to 0.159 (`mean_diff`) on `gpt2-large` and by 0.119 (`kmeans`) to 0.263 (`pca_diff`) on `gpt2-xl`.
- **Judges.** The share of swept settings with the same dose sign under both judges is 0.722 on `gpt2-large` and 1.000 on `gpt2-xl`. At 0.3 the check judge puts `mean_diff` first on both sizes (0.605 and 0.739). On `gpt2-large` it puts `logistic` lowest among the finders (0.503, above the random direction at 0.489); on `gpt2-xl` it puts `logistic` third (0.703, above `pca_raw` at 0.664) and `kmeans` last (0.583).

Cost to the text. Unsteered, the neutral text has a perplexity of 34.0 on `gpt2-large` and 30.9 on `gpt2-xl` and the generated text a distinct-1 of 0.201 and 0.204. The table gives the mean over the three steering readers, with the ratio to the unsteered perplexity in parentheses at the largest strength.

`gpt2-large`:

| Finder    | Perplexity +0.1 | Perplexity +0.2 | Perplexity +0.3 | Distinct-1 +0.3 | Perplexity -0.3 | Distinct-1 -0.3 |
| --------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| pca_diff  | 34.7            | 37.8            | 46.0 (1.35x)    | 0.190           | 46.6 (1.37x)    | 0.167           |
| pca_raw   | 34.8            | 38.3            | 47.9 (1.41x)    | 0.191           | 47.7 (1.40x)    | 0.160           |
| mean_diff | 35.6            | 41.1            | 55.8 (1.64x)    | 0.177           | 51.8 (1.53x)    | 0.164           |
| logistic  | 35.9            | 41.9            | 57.1 (1.68x)    | 0.177           | 51.9 (1.53x)    | 0.177           |
| kmeans    | 34.8            | 38.3            | 47.4 (1.39x)    | 0.190           | 47.1 (1.38x)    | 0.162           |
| random    | 34.6            | 37.2            | 43.8 (1.29x)    | 0.176           | 44.6 (1.31x)    | 0.194           |

`gpt2-xl`:

| Finder    | Perplexity +0.1 | Perplexity +0.2 | Perplexity +0.3 | Distinct-1 +0.3 | Perplexity -0.3 | Distinct-1 -0.3 |
| --------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| pca_diff  | 31.4            | 34.1            | 41.4 (1.34x)    | 0.172           | 43.8 (1.42x)    | 0.188           |
| pca_raw   | 31.6            | 34.7            | 43.3 (1.40x)    | 0.178           | 43.9 (1.42x)    | 0.190           |
| mean_diff | 31.8            | 35.8            | 46.8 (1.51x)    | 0.174           | 51.6 (1.67x)    | 0.165           |
| logistic  | 32.0            | 36.5            | 48.2 (1.56x)    | 0.175           | 51.2 (1.66x)    | 0.175           |
| kmeans    | 31.6            | 34.4            | 41.9 (1.36x)    | 0.178           | 42.5 (1.38x)    | 0.188           |
| random    | 31.7            | 34.6            | 41.2 (1.33x)    | 0.192           | 40.2 (1.30x)    | 0.195           |

- **The two finders that read best are the ones that cost the text most.** At +0.3 the perplexity of `mean_diff` and `logistic` is 1.64 and 1.68 times the unsteered one on `gpt2-large` and 1.51 and 1.56 times on `gpt2-xl`, against 1.35 to 1.41 (`gpt2-large`) and 1.34 to 1.40 (`gpt2-xl`) for `pca_diff`, `pca_raw` and `kmeans` and 1.29 and 1.33 for the random direction; at -0.3 they are 1.53 and 1.53 (`gpt2-large`) and 1.67 and 1.66 (`gpt2-xl`). By the rule above, strengths 0.1 and 0.2 are fluent in both directions for every finder and for the random direction on both sizes (ratios at most 1.06 and 1.23 on `gpt2-large` and 1.05 and 1.21 on `gpt2-xl` at 0.1 and 0.2, distinct-1 of at least 0.16), and strength 0.3 is not fluent for `mean_diff` and `logistic` on either size (in both directions). The highest win rate on `gpt2-large` (`mean_diff`, 0.600 at 0.3) is therefore at a setting that fails the rule, and the best win rate among the settings that pass it is 0.564 (`mean_diff` at 0.2); on `gpt2-xl` the best (`pca_diff`, 0.755 at 0.3) passes it.
- The C1 runs, which use a different direction, layer window and coefficient grid, have a neutral-text perplexity of at least 139.4 at 0.3 in either direction on every GPT-2 size (see [Sentiment steering on the GPT-2 ladder](#sentiment-steering-on-the-gpt-2-ladder-c1)), against at most 1.68 times the unsteered one for the M1 finders at 0.3. The two sets of values do not carry over to each other.

Erasure. Projecting a finder's direction out of the activations and fitting a fresh logistic probe on the stored activations of the control layers gives the probe accuracy before and after in the next tables, and the behavioral gap (the score after the more pleasant scenario of a pair minus after the less pleasant one, 100 pairs) is given for the primary and the check judge. The `none` row is the gap without a projection.

`gpt2-large`:

| Finder               | Probe before         | Probe after          | Gap primary            | Gap check            |
| -------------------- | -------------------- | -------------------- | ---------------------- | -------------------- |
| pca_diff             | 0.643 [0.567, 0.715] | 0.659 [0.584, 0.706] | 0.042 [0.024, 0.070]   | 0.097 [0.068, 0.131] |
| pca_raw              | 0.643 [0.567, 0.715] | 0.636 [0.554, 0.702] | 0.057 [0.031, 0.073]   | 0.104 [0.052, 0.188] |
| mean_diff            | 0.643 [0.567, 0.715] | 0.500 [0.500, 0.500] | 0.057 [0.027, 0.083]   | 0.107 [0.028, 0.190] |
| logistic             | 0.643 [0.567, 0.715] | 0.586 [0.541, 0.634] | -0.004 [-0.107, 0.095] | 0.090 [0.029, 0.170] |
| kmeans               | 0.643 [0.567, 0.715] | 0.642 [0.573, 0.701] | 0.081 [0.071, 0.098]   | 0.122 [0.092, 0.176] |
| random               | 0.643 [0.567, 0.715] | 0.643 [0.567, 0.715] | 0.126 [0.109, 0.154]   | 0.185 [0.120, 0.241] |
| none (no projection) |                      |                      | 0.087 [-0.035, 0.203]  | 0.185 [0.058, 0.308] |

`gpt2-xl`:

| Finder               | Probe before         | Probe after          | Gap primary           | Gap check            |
| -------------------- | -------------------- | -------------------- | --------------------- | -------------------- |
| pca_diff             | 0.678 [0.597, 0.736] | 0.676 [0.598, 0.723] | 0.135 [0.095, 0.156]  | 0.101 [0.020, 0.158] |
| pca_raw              | 0.678 [0.597, 0.736] | 0.670 [0.585, 0.721] | 0.132 [0.070, 0.178]  | 0.183 [0.137, 0.254] |
| mean_diff            | 0.678 [0.597, 0.736] | 0.500 [0.500, 0.500] | 0.185 [0.144, 0.233]  | 0.136 [0.095, 0.169] |
| logistic             | 0.678 [0.597, 0.736] | 0.612 [0.561, 0.643] | 0.164 [0.108, 0.197]  | 0.108 [0.047, 0.143] |
| kmeans               | 0.678 [0.597, 0.736] | 0.672 [0.585, 0.718] | 0.119 [0.035, 0.228]  | 0.129 [0.043, 0.206] |
| random               | 0.678 [0.597, 0.736] | 0.678 [0.598, 0.736] | 0.110 [0.044, 0.165]  | 0.083 [0.070, 0.091] |
| none (no projection) |                      |                      | 0.104 [-0.030, 0.231] | 0.120 [0.003, 0.231] |

- **The probe is erased for one finder.** The probe accuracy is 0.643 [0.567, 0.715] before the projection on `gpt2-large` and 0.678 [0.597, 0.736] on `gpt2-xl`. After `mean_diff` it is 0.500, after `logistic` 0.586 and 0.612, after `pca_diff`, `pca_raw` and `kmeans` between 0.636 and 0.659 (`gpt2-large`) and between 0.670 and 0.676 (`gpt2-xl`), and after the random direction it is unchanged.
- **The behavioral gap does not move.** It is 0.087 [-0.035, 0.203] and 0.104 [-0.030, 0.231] without a projection and between -0.004 and 0.126 (`gpt2-large`) and between 0.110 and 0.185 (`gpt2-xl`) under the projection of each finder or of the random direction, with every interval overlapping the unprojected one. The lowest point value is `logistic` on `gpt2-large` (-0.004, a fall of 0.091 with the interval [-0.107, 0.095]); the random projection gives 0.126. On the check judge the unprojected gaps are 0.185 and 0.120 and the projected ones 0.090 to 0.185 and 0.083 to 0.183. As on `gpt2`, the two operationalizations of erasure disagree: the probe is erased for one finder and the behavioral gap for none.

#### C2 on gpt2-large and gpt2-xl

C2 steers along the logistic reader of each of the six R4 emotions and measures, with the primary emotion judge, the probability of every emotion in the completions (runs `c2/gpt2-large/4741abe01da5` and `c2/gpt2-xl/aefba1c6274e`; 100 prompts, 2 samples per prompt, 3 steering seeds, relative strength +0.3). The reading layer is L27 with steering over layers 26, 27 and 28 on `gpt2-large`, and L26 with layers 25, 26 and 27 on `gpt2-xl`. The range is the minimum and maximum over the three steering seeds; the change is the own-emotion probability at +0.3 minus the unsteered one. The last column is the range of the own-emotion change under the three random arms (unit directions norm-matched to the steering arm, with the same window, coefficients, operator, mode and mask).

`gpt2-large`:

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.116         | 0.682 [0.636, 0.756]  | +0.565          | 0.303       | 0.935       | +0.632        | fear -0.122                    | fear -0.159                  | -0.030 to +0.009        |
| sadness   | 0.096         | 0.470 [0.381, 0.525]  | +0.374          | 0.141       | 0.548       | +0.407        | fear -0.059                    | anger -0.117                 | -0.050 to +0.015        |
| anger     | 0.066         | 0.186 [0.172, 0.192]  | +0.119          | 0.137       | 0.416       | +0.278        | happiness -0.083               | disgust +0.264               | -0.004 to +0.017        |
| fear      | 0.137         | 0.348 [0.280, 0.435]  | +0.210          | 0.168       | 0.385       | +0.216        | happiness -0.096               | happiness -0.167             | -0.052 to -0.044        |
| disgust   | 0.086         | 0.456 [0.420, 0.495]  | +0.369          | 0.152       | 0.480       | +0.328        | happiness -0.101               | anger +0.269                 | -0.008 to +0.022        |
| surprise  | 0.089         | 0.173 [0.122, 0.224]  | +0.084          | 0.081       | 0.103       | +0.022        | fear -0.095                    | fear -0.126                  | -0.020 to +0.009        |

`gpt2-xl`:

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.109         | 0.275 [0.231, 0.307]  | +0.166          | 0.307       | 0.663       | +0.356        | surprise +0.110                | fear -0.108                  | -0.024 to +0.015        |
| sadness   | 0.102         | 0.128 [0.078, 0.165]  | +0.026          | 0.158       | 0.210       | +0.053        | fear -0.057                    | anger -0.064                 | -0.051 to -0.009        |
| anger     | 0.081         | 0.099 [0.091, 0.103]  | +0.018          | 0.134       | 0.290       | +0.156        | fear -0.090                    | fear -0.110                  | -0.027 to -0.021        |
| fear      | 0.120         | 0.222 [0.195, 0.249]  | +0.102          | 0.135       | 0.244       | +0.109        | disgust -0.076                 | happiness -0.125             | -0.045 to -0.015        |
| disgust   | 0.095         | 0.166 [0.159, 0.172]  | +0.071          | 0.164       | 0.232       | +0.068        | fear -0.066                    | happiness -0.151             | -0.030 to +0.014        |
| surprise  | 0.079         | 0.127 [0.116, 0.143]  | +0.048          | 0.081       | 0.109       | +0.027        | fear -0.084                    | sadness -0.115               | +0.005 to +0.042        |

- **On `gpt2-large` all six emotions rise toward their own score**, by +0.084 (surprise) to +0.565 (happiness) on the primary judge, with the range over seeds above the unsteered value in every row. The hit rate (the share of rows of the change matrix whose largest entry is on the diagonal) is 1.000 and the specificity (the mean diagonal change minus the mean off-diagonal change) is 0.338; on the check judge they are 1.000 and 0.367, and the correlation between the judges over the matrix is 0.889. The check judge puts anger and disgust on each other (anger row: disgust +0.264 off the diagonal, diagonal +0.278; disgust row: anger +0.269, diagonal +0.328).
- **On `gpt2-xl` steering is weaker and only some emotions separate.** Five emotions have a range over seeds above the unsteered value (happiness +0.166, fear +0.102, disgust +0.071, surprise +0.048 and anger +0.018), and the range of sadness ([0.078, 0.165], change +0.026) contains the unsteered value (0.102). The primary hit rate is 0.833 (the miss is anger, whose largest change is on sadness, +0.028 against +0.018) and the specificity 0.110; on the check judge the hit rate is 0.833 (the miss is surprise, whose largest change is on happiness, +0.045 against +0.027) and the specificity 0.178. The correlation between the judges is 0.788.
- **Random arms.** The own-emotion change under a random arm ranges from -0.052 to +0.022 on `gpt2-large` and from -0.051 to +0.042 on `gpt2-xl`. The largest absolute change of any emotion under any random arm and coefficient is 0.108 and 0.125. On `gpt2-large` that is larger than the change of surprise (+0.084) and close to that of anger (+0.119), so those two are not clearly separated from an arbitrary edit of the same size; the other four (+0.210 to +0.565) are. On `gpt2-xl` only happiness (+0.166) is above the random-arm maximum.
- **Off-diagonal.** In 3 of the 5 non-happiness rows of `gpt2-large` (anger, fear and disgust) the largest off-diagonal change on the primary judge is a fall of happiness (-0.083 to -0.101); in the sadness and surprise rows it is a fall of fear (-0.059 and -0.095), and in the happiness row too (-0.122). On `gpt2-xl` it is a fall of fear in four rows (sadness, anger, disgust and surprise: -0.057, -0.090, -0.066 and -0.084), a fall of disgust in the fear row (-0.076) and a rise of surprise in the happiness row (+0.110). Happiness is not the main loser on `gpt2-xl`, as it is on the Qwen3 pair.

Cost to the text. Averaged over the six emotions and the three readers, the perplexity of held-out neutral text under the emotion arms is 1.12, 1.66 and 3.23 times the unsteered one at +0.1, +0.2 and +0.3 on `gpt2-large`, and 1.07, 1.42 and 2.75 times on `gpt2-xl`; the distinct-1 of the generated text is 0.193 and 0.234 at +0.3 against 0.322 and 0.327 without steering. The three random arms are at 1.34 and 1.30 times at +0.3. The table gives the values at +0.3 by emotion (mean over the three readers, with the range).

`gpt2-large`:

| Arm                   | Perplexity [range over arms or readers] | Ratio to none | Distinct-1 [range]   |
| --------------------- | --------------------------------------- | ------------- | -------------------- |
| none                  | 34.0                                    | 1.00          | 0.322                |
| random (3 arms, +0.3) | 45.6 [43.2, 47.5]                       | 1.34          | 0.295 [0.293, 0.297] |
| happiness (+0.3)      | 128.1 [120.5, 135.3]                    | 3.77          | 0.213 [0.208, 0.222] |
| sadness (+0.3)        | 86.2 [85.2, 86.9]                       | 2.54          | 0.180 [0.171, 0.186] |
| anger (+0.3)          | 109.2 [103.4, 119.4]                    | 3.21          | 0.190 [0.188, 0.192] |
| fear (+0.3)           | 108.1 [104.6, 111.1]                    | 3.18          | 0.193 [0.189, 0.196] |
| disgust (+0.3)        | 141.7 [134.9, 146.1]                    | 4.17          | 0.175 [0.167, 0.183] |
| surprise (+0.3)       | 85.6 [84.6, 86.8]                       | 2.52          | 0.207 [0.205, 0.209] |

`gpt2-xl`:

| Arm                   | Perplexity [range over arms or readers] | Ratio to none | Distinct-1 [range]   |
| --------------------- | --------------------------------------- | ------------- | -------------------- |
| none                  | 30.9                                    | 1.00          | 0.327                |
| random (3 arms, +0.3) | 40.3 [40.3, 40.4]                       | 1.30          | 0.299 [0.291, 0.310] |
| happiness (+0.3)      | 71.8 [66.5, 77.2]                       | 2.32          | 0.243 [0.237, 0.250] |
| sadness (+0.3)        | 70.4 [69.3, 72.3]                       | 2.28          | 0.225 [0.221, 0.230] |
| anger (+0.3)          | 126.9 [84.9, 181.9]                     | 4.11          | 0.187 [0.178, 0.200] |
| fear (+0.3)           | 92.1 [86.5, 96.8]                       | 2.98          | 0.232 [0.229, 0.233] |
| disgust (+0.3)        | 77.3 [73.6, 84.4]                       | 2.50          | 0.273 [0.257, 0.288] |
| surprise (+0.3)       | 71.8 [64.4, 77.5]                       | 2.32          | 0.244 [0.230, 0.261] |

- **The +0.3 column is not fluent text.** At +0.3 every emotion has a perplexity of 2.52 to 4.17 times the unsteered one on `gpt2-large` and 2.28 to 4.11 times on `gpt2-xl` (disgust is the highest on `gpt2-large` and anger on `gpt2-xl`), so by the rule above no change in the first table is the change of fluent text, and none is read as steering that works. On `gpt2-large` +0.1 is the only coefficient that passes on average (+0.2 is 1.66 times); on `gpt2-xl` +0.2 passes on average (1.42 times) with anger (1.53) and fear (1.54) above the cutoff. The own-emotion probability at +0.1 and +0.2 is in the curves file of each run and is not tabulated here.

#### P7 on gpt2-large and gpt2-xl

P7 extracts the sentiment reader under six templates of P1 and steers with the C1 protocol, as on `gpt2` (runs `p7/gpt2-large/d03910535564` and `p7/gpt2-xl/eb67813e769d`; coefficients -0.3, -0.15, 0, 0.15 and 0.3, 50 prompts, 2 samples per prompt, 2 steering readers per template), over the sweep layers 1, 8, 15, 22, 29 and 36 on `gpt2-large` and 1, 10, 20, 29, 39 and 48 on `gpt2-xl` (the last is the last block of each model). A unit is a template, a sweep layer and a steering reader (72 units); it is readable if the validation accuracy of its reader at that layer is at least 0.6, and inverted if it is readable and its primary dose slope is negative. The runs add three random directions (one reader each, steered at every sweep layer with the same grid) and the fluency measure.

`gpt2-large`:

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at steer layer | Dose slope, sweep layers 1 to 29 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ------------------------- | -------------------------------- | -------------- | -------- |
| paper          | L35        | 0.838         | L36         | 0.991                     | 0.102 to 0.698                   | 12             | 0        |
| no_template    | L28        | 0.903         | L29         | 0.923                     | 0.036 to 0.923                   | 12             | 0        |
| no_chat_tags   | L26        | 0.930         | L36         | 1.034                     | -0.052 to 0.838                  | 12             | 2        |
| neutral_frame  | L25        | 0.853         | L36         | 0.777                     | 0.060 to 0.703                   | 12             | 0        |
| scaffold_0     | L27        | 0.910         | L36         | 1.010                     | 0.033 to 0.961                   | 12             | 1        |
| concept_swap_0 | L26        | 0.830         | L36         | 0.880                     | 0.094 to 0.660                   | 12             | 0        |

`gpt2-xl`:

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at steer layer | Dose slope, sweep layers 1 to 39 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ------------------------- | -------------------------------- | -------------- | -------- |
| paper          | L24        | 0.816         | L48         | 1.050                     | -0.260 to 0.623                  | 12             | 3        |
| no_template    | L30        | 0.899         | L48         | 1.072                     | -0.278 to 0.930                  | 12             | 3        |
| no_chat_tags   | L30        | 0.911         | L48         | 1.543                     | -0.041 to 0.885                  | 12             | 2        |
| neutral_frame  | L27        | 0.892         | L48         | 1.034                     | -0.347 to 0.601                  | 12             | 4        |
| scaffold_0     | L30        | 0.886         | L48         | 0.956                     | -0.375 to 0.647                  | 12             | 4        |
| concept_swap_0 | L36        | 0.826         | L48         | 1.154                     | -0.393 to 0.457                  | 12             | 4        |

The read accuracy is the best point of the mean reading curve over the 10 readers, so it is optimistic by the selection. The sweep-layer column is the minimum and maximum of the mean dose slope over the sweep layers other than the last, read from the curves files.

- **Steering layer.** On `gpt2-large` five templates steer most at the last block (L36) and `no_template` at L29 (0.923 against 0.917 at L36); on `gpt2-xl` all six steer most at L48, the edge of what was swept. The slope at the steering layer is 0.777 to 1.034 on `gpt2-large` and 0.956 to 1.543 on `gpt2-xl` (1.543 for `no_chat_tags`). The slope is not monotone in depth: on `gpt2-xl` the mean slope of every template is negative at layer 20 (-0.393 to -0.032) and positive at layers 29, 39 and 48 (for example `paper`: 0.029, 0.007, -0.260, 0.215, 0.623 and 1.050 at layers 1, 10, 20, 29, 39 and 48). The slope at the steering layer is measured on the prompts that chose that layer and is optimistic.
- **Reading layers.** The reading layer is L25 to L35 on `gpt2-large` and L24 to L36 on `gpt2-xl`. The mean absolute gap between the reading depth and the steering depth (layer divided by the number of blocks) is 0.194 on `gpt2-large` and 0.385 on `gpt2-xl`.
- **Inversion.** Every unit is readable (72 of 72 on both). On `gpt2-large` 3 are inverted, a rate of 0.042 [0.014, 0.115], all at layer 1 (`no_chat_tags` 2, `scaffold_0` 1). On `gpt2-xl` 20 are inverted, a rate of 0.278 [0.188, 0.390] (Wilson intervals over the 72 units), at layers 1 (5 units), 10 (4) and 20 (11) and none at layer 29, 39 or 48; the counts by template are 3, 3, 2, 4, 4 and 4. On `gpt2` the rate was 1 of 72 (0.014).
- **Reading and steering.** The Spearman correlation over the 72 units between reading accuracy and dose slope is 0.807 on `gpt2-large` and 0.531 on `gpt2-xl`, and between the logit-lens score and dose slope 0.871 and 0.804. The sign of the slope agrees between the two judges for 0.875 (63 of 72) and 0.903 (65 of 72) of the units. The sweep layer with the largest logit-lens separation is the steering layer for 4 of 6 templates on both (on `gpt2-large` `no_template` has L36 and `no_chat_tags` L29; on `gpt2-xl` `no_template` and `scaffold_0` have L39).
- **The correlation is not separated from depth.** Both the reading accuracy and the slope rise with depth over the sweep, so a rank correlation over units that mixes layers can be driven by depth alone. The summary does not give the within-layer correlation.
- **Prompt format.** The mean primary dose slope by steering prompt format (raw, chat and story) is 0.596 to 1.000, 0.612 to 0.962 and 0.725 to 1.056 over the six templates on `gpt2-large`, and 0.861 to 1.407, 0.875 to 1.346 and 0.873 to 1.462 on `gpt2-xl`.
- **Random directions.** The largest absolute dose slope of a random direction at any sweep layer is 0.210 on `gpt2-large` (layer 22, where the six template means are 0.393 to 0.961) and 0.238 on `gpt2-xl` (layer 39, where the six template means are 0.457 to 0.885), against 0.777 to 1.034 and 0.956 to 1.543 for the templates at the steering layer. The inversion on `gpt2-xl` is not an artifact that random directions share: at layer 20 the three random slopes are -0.192 to 0.172, and five of the six template means (-0.393 to -0.260) are below the lowest of them.
- **Fluency.** On `gpt2-large` the neutral perplexity under steering is at most 1.25 times the unsteered one (34.0) in all 360 cells (288 template cells and 72 random-arm rows), and the lowest distinct-1 is 0.273 (unsteered 0.309), so every cell is fluent. On `gpt2-xl` 9 of the 360 cells (288 template cells and 72 random-arm rows) exceed 1.5 times (1.51 to 1.67; all 9 are template cells), all at layer 48 and coefficient -0.3, and the lowest distinct-1 is 0.155 (unsteered 0.324). The dose slope at the last block of `gpt2-xl` is fitted over a grid that includes those cells.

#### C5 on gpt2-large and gpt2-xl

C5 records the share of correct answers on 1,000 TriviaQA questions of the pool, one sample per question at temperature 0.8 with a 15-shot prompt, without steering and under an honesty-steered arm (+0.1) and a lie-steered arm (-0.1) of the same relative strength (runs `c5/gpt2-large/36f392712827` and `c5/gpt2-xl/0ad95833261d`; steered over layers 10, 13, 17, 20, 23, 27, 30 and 34 on `gpt2-large` and 13, 17, 22, 27, 31, 35, 40 and 45 on `gpt2-xl`). The runs add three random arms (unit directions norm-matched to the steering arm, steered at +0.1 and -0.1) and the fluency measure. A flip is a question the plain model answers correctly and the lie arm answers wrongly. `gpt2-xl` has an earlier C5 run, `c5/gpt2-xl/155f0f72bc3f`, with no random arm and no fluency measure and no emotion gallery; it stays tracked, and the `gpt2-xl` column below is the later run, which is the reported one. The two differ in the C2 run they read (none earlier), in `n_random`, in the commit (`ca2546c` earlier) and in the GPU (a laptop GPU earlier).

| Quantity                                     | gpt2-large           | gpt2-xl              |
| -------------------------------------------- | -------------------- | -------------------- |
| Correct, no steering                         | 0.112                | 0.150                |
| Correct, honesty-steered                     | 0.097                | 0.112                |
| Correct, lie-steered                         | 0.078                | 0.091                |
| Correct, random +0.1 (mean of 3)             | 0.098                | 0.140                |
| Correct, random -0.1 (mean of 3)             | 0.105                | 0.141                |
| Answered correctly without steering          | 112                  | 150                  |
| Of these, answered wrongly under the lie arm | 45                   | 70                   |
| Flip rate [95% interval]                     | 0.402 [0.316, 0.494] | 0.467 [0.389, 0.546] |
| Perplexity: none                             | 11.0                 | 10.2                 |
| Perplexity: honesty-steered                  | 14.1                 | 13.1                 |
| Perplexity: lie-steered                      | 14.7                 | 14.8                 |
| Perplexity: random (range of 6)              | 11.6 to 12.0         | 10.9 to 11.0         |
| Distinct-1: none                             | 0.144                | 0.145                |
| Distinct-1: honesty-steered                  | 0.142                | 0.121                |
| Distinct-1: lie-steered                      | 0.150                | 0.155                |
| Distinct-1: random (range of 6)              | 0.135 to 0.149       | 0.135 to 0.145       |

The random rows are the mean of the three random directions. The perplexity is that of held-out neutral text under the steered model, and the distinct-1 that of the generated answers (about 24 tokens).

- **Both steered arms are below the unsteered one on both sizes**, the lie arm further (by 0.034 and 0.059, the honesty arm by 0.015 and 0.038). The random arms are also below the unsteered share (0.098 and 0.105 against 0.112 on `gpt2-large`; 0.140 and 0.141 against 0.150 on `gpt2-xl`). On `gpt2-large` the honesty arm (0.097) is level with the random arm at +0.1 (0.098), so its drop is not separated from a generic perturbation of the model, and the lie arm (0.078) is 0.020 to 0.027 below the random arms. On `gpt2-xl` the honesty arm is 0.028 to 0.029 below the random arms and the lie arm 0.049 to 0.050 below. The shares carry no interval; for scale, a share of 1,000 questions near 0.1 has a binomial standard error of about 0.01, and the earlier `gpt2-xl` run gave 0.147, 0.107 and 0.080 for the three arms.
- **Flip rate.** 45 of the 112 questions answered correctly are answered wrongly under the lie arm on `gpt2-large` (0.402 [0.316, 0.494]) and 70 of 150 on `gpt2-xl` (0.467 [0.389, 0.546]); the earlier `gpt2-xl` run gave 77 of 147 (0.524 [0.443, 0.603]). The intervals overlap. No flip rate is stored for the random arms, so the flip rate is not separated from the effect of a generic perturbation of the model, and the random arms lower the correct share too.
- **Cost to the text.** The perplexity of neutral text is 1.28 and 1.33 times the unsteered one under the honesty and lie arms on `gpt2-large` and 1.29 and 1.45 times on `gpt2-xl`, against 1.06 to 1.09 and 1.07 to 1.08 for the random arms, so the random arms disturb the model less than the steering arms do at the same relative strength. The distinct-1 cutoff of the rule above (0.15) is not used here: the unsteered answers are already at 0.144 and 0.145, and the steered arms are between 0.121 and 0.155.
- The run also stores a small gallery of completions (4 unfiltered and up to 4 curated per arm, the curated ones being the first flipped questions) and an emotion gallery read from the C2 run of the same size, which are selections and not evidence for a rate and are not read here.

#### LR1 and C4 on gpt2-medium, gpt2-large and gpt2-xl

LR1 and C4 ran on the three larger sizes in chain 17, at `89ea7cb` with a clean tree on a rented GPU pod (NVIDIA RTX
A6000, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment. The batch size of every LR1
and C4 run here is 16 (`batch_size` in each `config.json`), the same as the `gpt2` runs above. None of these runs is a
smaller-batch retry: the chain log has one retry only, for a P9b run of the Qwen3 4B pair, and the LR1 and C4 runs here
each ran once, with exit status 0. The three sizes are three models: the tables are separate and no metric is pooled
over them.

LR1 (runs `lr1/gpt2-medium/c6bd21c82ddc`, `lr1/gpt2-large/5f6fdb0ccda5` and `lr1/gpt2-xl/f54173153b29`): rank 8,
learning rate 3e-4, 70 steps, 2,000 training examples and 64 held-out (none dropped), with the control prompts "Give a
truthful answer." and "Give an untruthful answer.". The layers are 8, 10, 12 and 14 (458,752 trainable parameters) on
`gpt2-medium`, 11, 13, 15, 17, 19 and 21 (860,160) on `gpt2-large` and 15 to 29 in steps of 2 (1,484,800) on `gpt2-xl`.
The training loss goes from 55.2 to 35.0, from 58.5 to 39.1 and from 168.5 to 120.1. The held-out loss of each layer,
before and after, and its reduction:

`gpt2-medium`:

| Layer | Held-out loss: before to after | Reduction |
| ----- | ------------------------------ | --------- |
| L8    | 11.58 to 6.19                  | 0.466     |
| L10   | 12.04 to 7.34                  | 0.391     |
| L12   | 14.52 to 9.60                  | 0.339     |
| L14   | 18.51 to 12.47                 | 0.327     |

`gpt2-large`:

| Layer | Held-out loss: before to after | Reduction |
| ----- | ------------------------------ | --------- |
| L11   | 7.28 to 4.61                   | 0.366     |
| L13   | 7.88 to 5.01                   | 0.365     |
| L15   | 8.57 to 5.44                   | 0.366     |
| L17   | 10.16 to 6.53                  | 0.357     |
| L19   | 12.53 to 8.16                  | 0.348     |
| L21   | 14.83 to 9.98                  | 0.327     |

`gpt2-xl`:

| Layer | Held-out loss: before to after | Reduction |
| ----- | ------------------------------ | --------- |
| L15   | 11.47 to 7.66                  | 0.332     |
| L17   | 12.74 to 8.41                  | 0.340     |
| L19   | 14.88 to 9.91                  | 0.334     |
| L21   | 17.56 to 11.89                 | 0.323     |
| L23   | 22.36 to 15.28                 | 0.316     |
| L25   | 27.03 to 18.63                 | 0.311     |
| L27   | 32.33 to 22.61                 | 0.301     |
| L29   | 38.91 to 27.39                 | 0.296     |

The loss grows with the layer, so the layers are compared by the reduction. It falls with depth on `gpt2-medium` (0.466
to 0.327) and `gpt2-xl` (0.340 at L17 to 0.296 at L29), as on `gpt2` (0.463, 0.423, 0.396), and is nearly flat on
`gpt2-large` from L11 to L17 (0.366, 0.365, 0.366, 0.357) before falling to 0.327 at L21. LR1 has no honesty evaluation:
it measures the loss of the adapter's own objective on held-out examples and trains the adapter that C4 uses.

C4 (runs `c4/gpt2-medium/88cc5d7f6dba`, `c4/gpt2-large/43bf8077183e` and `c4/gpt2-xl/321f3da56ea5`, each reading the LR1
adapter of its size through `lorra_run`) follows the protocol of the `gpt2` C4 run above and of the Qwen3 1.7B C4 runs:
MC1 on the 173 test questions of R6, the coefficients chosen on the 644 train and validation questions, chance line
0.228. Like the Qwen3 1.7B runs, and unlike the `gpt2` run, they carry three random arms (unit directions norm-matched
to the steering arm, with a coefficient chosen on the train and validation questions like the readers) and the
perplexity of neutral held-out text under each edited model; the prompting and `contrast` arms have no edited model.
The unedited row of each size is the standard row of its R6 run (0.173, 0.179 and 0.173). The coefficient grids are
-0.2 to 0.2 for `difference`, the readers and the random arms and -0.5 to 0.5 for `contrast`.

`gpt2-medium`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.173 [0.124, 0.237]       | 0                       | 12.79              |
| prompting +1         | 0.168 [0.119, 0.230]       | -0.006 [-0.029, +0.012] | -                  |
| prompting -1         | 0.168 [0.119, 0.230]       | -0.006 [-0.029, +0.012] | -                  |
| difference (-0.2)    | 0.208 [0.154, 0.275]       | +0.035 [-0.017, +0.087] | 145.84             |
| contrast (0.25)      | 0.168 [0.119, 0.230]       | -0.006 [-0.029, +0.017] | -                  |
| LoRRA                | 0.156 [0.110, 0.218]       | -0.017 [-0.040, +0.006] | 12.78              |
| reading (10 readers) | mean 0.207, 0.191 to 0.220 | +0.017 to +0.046        | 24.89 to 27.77     |
| random (3 arms)      | mean 0.197 (all 3 equal)   | +0.023 (all 3 equal)    | 33.68 to 37.57     |

`gpt2-large`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.179 [0.129, 0.243]       | 0                       | 11.01              |
| prompting +1         | 0.168 [0.119, 0.230]       | -0.012 [-0.035, +0.012] | -                  |
| prompting -1         | 0.185 [0.134, 0.249]       | +0.006 [-0.023, +0.035] | -                  |
| difference (-0.2)    | 0.208 [0.154, 0.275]       | +0.029 [-0.023, +0.087] | 51.05              |
| contrast (-0.5)      | 0.191 [0.139, 0.256]       | +0.012 [-0.023, +0.046] | -                  |
| LoRRA                | 0.162 [0.114, 0.224]       | -0.017 [-0.046, +0.012] | 11.00              |
| reading (10 readers) | mean 0.228, 0.214 to 0.243 | +0.035 to +0.064        | 34.79 to 41.01     |
| random (3 arms)      | mean 0.189, 0.179 to 0.202 | +0.000 to +0.023        | 14.81 to 14.90     |

`gpt2-xl`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.173 [0.124, 0.237]       | 0                       | 10.21              |
| prompting +1         | 0.156 [0.110, 0.218]       | -0.017 [-0.052, +0.017] | -                  |
| prompting -1         | 0.185 [0.134, 0.249]       | +0.012 [-0.017, +0.040] | -                  |
| difference (0.2)     | 0.220 [0.164, 0.287]       | +0.046 [+0.000, +0.092] | 33.24              |
| contrast (0.5)       | 0.150 [0.105, 0.211]       | -0.023 [-0.052, +0.012] | -                  |
| LoRRA                | 0.173 [0.124, 0.237]       | +0.000 [-0.035, +0.035] | 10.17              |
| reading (10 readers) | mean 0.210, 0.191 to 0.220 | +0.017 to +0.046        | 52.60 to 66.76     |
| random (3 arms)      | mean 0.179, 0.168 to 0.191 | -0.006 to +0.017        | 11.00 to 14.21     |

- **No arm clears the chance line by more than a few questions, and the adapter does not raise MC1 on any size.** The
  test MC1 of the unedited model is 0.173, 0.179 and 0.173 and every row of the three tables is at 0.243 or below; the
  ten readers of `gpt2-large` average 0.228 (0.214 to 0.243), at the chance line. LoRRA is -0.017 [-0.040, +0.006],
  -0.017 [-0.046, +0.012] and +0.000 [-0.035, +0.035] from the unedited row, with every interval containing zero, and
  leaves the perplexity of neutral text where it was (12.78 against 12.79, 11.00 against 11.01 and 10.17 against
  10.21). On `gpt2` the adapter gave +0.012 [0.000, +0.029], and on the Qwen3 1.7B base checkpoint +0.081 [+0.035,
  +0.127].
- **Prompting does not move MC1 by more than a question or two.** The truthful and untruthful prompts give -0.006 and
  -0.006 on `gpt2-medium`, -0.012 and +0.006 on `gpt2-large` and -0.017 and +0.012 on `gpt2-xl`; every interval contains
  zero.
- **`difference` and `contrast`.** `difference` gives +0.035 [-0.017, +0.087], +0.029 [-0.023, +0.087] and +0.046
  [+0.000, +0.092]; the intervals contain zero or touch it, and its coefficient is at the edge of the grid on all three
  sizes (-0.2, -0.2 and +0.2), at a perplexity of neutral text of 145.84, 51.05 and 33.24 (11.4, 4.6 and 3.3 times the
  unedited one). `contrast` gives -0.006, +0.012 and -0.023, all with intervals containing zero, and its chosen alpha is
  at the edge of its grid on `gpt2-large` (-0.5) and `gpt2-xl` (0.5).
- **The ten readers against the random arms.** The readers' changes are +0.017 to +0.046 (`gpt2-medium`), +0.035 to
  +0.064 (`gpt2-large`) and +0.017 to +0.046 (`gpt2-xl`), and the interval excludes zero for 3 of 10 readers on the
  first two sizes (readers 2, 6 and 8, and readers 3, 6 and 7) and for none on `gpt2-xl`. The coefficient chosen on the
  train and validation questions was -0.1 for all ten readers on `gpt2-medium` and -0.2, the edge of the grid, for all
  ten on `gpt2-large` and `gpt2-xl`. The random arms move MC1 as well: +0.023 for all three arms on `gpt2-medium`
  (coefficients -0.2, -0.2 and +0.2), 0.000 to +0.023 on `gpt2-large` and -0.006 to +0.017 on `gpt2-xl`. On
  `gpt2-medium` the readers overlap the random arms and on `gpt2-xl` the best random arm (+0.017) equals the weakest
  reader; on `gpt2-large` every reader is above every random arm (the smallest reader +0.035, the largest random arm
  +0.023, a difference of two questions of 173). The cost to the text differs by size: the readers cost more than the
  random arms on `gpt2-large` (3.16 to 3.72 times the unedited perplexity against 1.35) and `gpt2-xl` (5.15 to 6.54
  against 1.08 to 1.39), and less on `gpt2-medium` (1.95 to 2.17 against 2.63 to 2.94).
- **What is and is not supported.** The only intervals that exclude zero are those of 6 of the 30 readers, among 19
  arms per size and with no correction for multiple comparisons; no row separates the adapter, `difference`,
  `contrast` or the readers from a random direction of the same norm at a selected coefficient except, in point
  estimate only, the readers of `gpt2-large`. The scoring is not generation, so the distinct-1 and the fluency of
  generated text are not measured. The perplexity column is that of held-out neutral text under the edited model.

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
GPT-2 sizes and of the Qwen3 0.6B and 1.7B checkpoints. P9a, P9b and P9c cover all eight models, and P9d ran on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the four Qwen3
checkpoints (base and post-trained). `gpt2` is the one model whose label gate
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
- **Pooled on `obscure_real`, the reader carries nothing that the wording does not.** On `gpt2-medium` the pooled reader (0.858) is below
  both text baselines (0.945 and 0.929) and above popularity (0.813) and a random direction (0.758); on `gpt2-large` it
  (0.928) is level with them (0.924 and 0.930) and above popularity (0.886) and a random direction (0.702). On
  `fictional_generated` the reader is below both text baselines pooled on both (0.903 against `text_word` 0.972 and
  `text_char` 0.981, 0.970 against 0.974 and 0.989) and within relation level with `text_char` on `gpt2-medium` (0.924
  against 0.924) and 0.009 above it on `gpt2-large` (0.985 against 0.976), above `text_word` within relation on both
  (0.836 and 0.816).
- **Between kinds of unknown questions the reader is below the character classifier.** `real_vs_synthal`,
  `real_vs_generated` and `pre_vs_post_cutoff` are read on all three sizes. Pooled, in the order `gpt2`, `gpt2-medium`,
`gpt2-large`, the reader reads 0.939, 0.937 and 0.959 on `real_vs_synthal` (`text_word` 0.799, 0.799 and 0.795,
`text_char` 1.000 on all three), 0.959, 0.937 and 0.952 on `real_vs_generated` (`text_word` 0.947, 0.949 and 0.950,
`text_char` 0.998) and 0.995, 0.953 and 0.974 on `pre_vs_post_cutoff` (`text_word` and `text_char` 1.000). Popularity
is not defined for these contrasts. The reader is above `text_word` on `real_vs_synthal` but below `text_char` on all
three contrasts and on all three sizes, so the only
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

#### The post-trained Qwen3 checkpoints

P9b on `qwen3-0.6b` and `qwen3-1.7b` (runs `p9b/qwen3-0.6b/975dbcf95759` and `p9b/qwen3-1.7b/9641ab8235a9`, made at `fb3b5dd` with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU), on the labels of the P9a run of the same checkpoint) uses the recipe above: five logistic readers on 64 pairs each, one layer per model chosen on the pooled validation AUROC, which is L27 on `qwen3-0.6b` and L25 on `qwen3-1.7b` (L27 and L21 on the base checkpoints). The base rows below are those of the base runs above, repeated so that each post-trained row has its base checkpoint next to it. A checkpoint is labeled on its own behavior, so the test counts differ between the base and the post-trained row (72 against 87 known PopQA test questions at 0.6B, 122 against 118 at 1.7B); the rows are never pooled, and a difference between two rows mixes the checkpoint with the question set. Every reader value is next to `text_word`, `text_char` and `popularity`. The P9b and P9c code is unchanged since the commit of the base runs (`95a8243`) except that the two text-baseline names moved into one constant.

PopQA known against unknown, within relation (the headline), mean AUROC over the five readers:

| Model           | Contrast         | Test known / unknown | Test pairs | Reader | popularity | text_word | text_char |
| --------------- | ---------------- | -------------------- | ---------- | ------ | ---------- | --------- | --------- |
| qwen3-0.6b-base | obscure_real     | 87 / 1,661           | 6,941      | 0.919  | 0.799      | 0.678     | 0.673     |
| qwen3-0.6b-base | obscure_real_pop | 87 / 1,661           | 6,941      | 0.891  | 0.799      | 0.678     | 0.673     |
| qwen3-0.6b      | obscure_real     | 72 / 1,713           | 6,765      | 0.909  | 0.752      | 0.619     | 0.657     |
| qwen3-0.6b      | obscure_real_pop | 72 / 1,713           | 6,765      | 0.873  | 0.752      | 0.619     | 0.657     |
| qwen3-1.7b-base | obscure_real     | 118 / 1,611          | 10,113     | 0.959  | 0.879      | 0.625     | 0.636     |
| qwen3-1.7b-base | obscure_real_pop | 118 / 1,611          | 10,113     | 0.960  | 0.879      | 0.625     | 0.636     |
| qwen3-1.7b      | obscure_real     | 122 / 1,657          | 10,680     | 0.893  | 0.850      | 0.561     | 0.602     |
| qwen3-1.7b      | obscure_real_pop | 122 / 1,657          | 10,680     | 0.845  | 0.850      | 0.561     | 0.602     |

The same contrasts pooled over relations, with the AUROC of a random direction:

| Model           | Contrast         | Reader | Random direction | popularity | text_word | text_char |
| --------------- | ---------------- | ------ | ---------------- | ---------- | --------- | --------- |
| qwen3-0.6b-base | obscure_real     | 0.948  | 0.682            | 0.840      | 0.944     | 0.941     |
| qwen3-0.6b-base | obscure_real_pop | 0.948  | 0.682            | 0.840      | 0.944     | 0.941     |
| qwen3-0.6b      | obscure_real     | 0.927  | 0.573            | 0.771      | 0.905     | 0.903     |
| qwen3-0.6b      | obscure_real_pop | 0.885  | 0.578            | 0.771      | 0.905     | 0.903     |
| qwen3-1.7b-base | obscure_real     | 0.961  | 0.648            | 0.865      | 0.873     | 0.873     |
| qwen3-1.7b-base | obscure_real_pop | 0.970  | 0.522            | 0.865      | 0.873     | 0.873     |
| qwen3-1.7b      | obscure_real     | 0.927  | 0.529            | 0.834      | 0.867     | 0.885     |
| qwen3-1.7b      | obscure_real_pop | 0.918  | 0.617            | 0.834      | 0.867     | 0.885     |

- **Within relation the reader is above popularity and both text baselines on three of the four post-trained cells.** On `obscure_real` it reads 0.909 and 0.893 against a popularity baseline of 0.752 and 0.850, margins of 0.157 and 0.043 (0.120 and 0.080 on the base checkpoints); `text_word` and `text_char` are at 0.619 and 0.657 at 0.6B and at 0.561 and 0.602 at 1.7B. The exception is `obscure_real_pop` on `qwen3-1.7b`: the reader reads 0.845 and popularity 0.850, so with the popularity bin matched the reader is 0.006 below the popularity baseline (and far above both text baselines). On `qwen3-0.6b` the matched contrast reads 0.873, 0.121 above popularity.
- **The margin over popularity does not move the same way at the two sizes.** At 0.6B it is larger on the post-trained checkpoint (0.157 against 0.120); at 1.7B it is smaller (0.043 against 0.080), where the reader falls from 0.959 to 0.893 and the popularity baseline from 0.879 to 0.850. Each cell is one run on a test known side of 72 to 122 questions, with an interval over the pair draws only, so the size of these margins is not a measured difference between the checkpoints.
- **Pooled, the post-trained reader is above both text baselines on `obscure_real` and below them on `obscure_real_pop` at 0.6B.** On `obscure_real` it reads 0.927 against 0.905 and 0.903 at 0.6B (a margin of 0.021) and 0.927 against 0.867 and 0.885 at 1.7B (0.042). On `obscure_real_pop` it reads 0.885 at 0.6B, 0.020 below the better text baseline, and 0.918 at 1.7B, 0.033 above it. On `qwen3-0.6b-base` the pooled reader was level with the text baselines (0.948 against 0.944 and 0.941). A random direction reads 0.573 and 0.578 at 0.6B and 0.529 and 0.617 at 1.7B, below the logistic reader in all four cells.

Other relational contrasts, within relation, as reader / `text_word` / `text_char` (`popularity` is not defined for them):

| Contrast                              | qwen3-0.6b-base       | qwen3-0.6b            | qwen3-1.7b-base       | qwen3-1.7b            |
| ------------------------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| fictional_generated (within relation) | 0.989 / 0.819 / 0.987 | 0.988 / 0.845 / 0.981 | 1.000 / 0.851 / 0.992 | 0.993 / 0.845 / 0.989 |
| real_vs_synthal (within relation)     | 0.978 / 0.736 / 1.000 | 0.979 / 0.739 / 1.000 | 0.986 / 0.728 / 1.000 | 0.980 / 0.732 / 1.000 |
| real_vs_generated (within relation)   | 0.980 / 0.953 / 0.999 | 0.974 / 0.952 / 0.999 | 0.986 / 0.954 / 0.999 | 0.968 / 0.952 / 0.999 |
| pre_vs_post_cutoff                    | not read              | not read              | not read              | not read              |

The other contrasts, pooled:

| Contrast             | qwen3-0.6b-base       | qwen3-0.6b            | qwen3-1.7b-base       | qwen3-1.7b            |
| -------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| triviaqa_known       | 0.905 / 0.774 / 0.727 | 0.877 / 0.750 / 0.711 | 0.883 / 0.693 / 0.671 | 0.840 / 0.664 / 0.588 |
| unknowable_detail    | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| false_premise        | 0.951 / 0.978 / 0.973 | 0.947 / 0.978 / 0.988 | 0.974 / 0.977 / 0.975 | 0.951 / 0.974 / 0.967 |
| kuq_ambiguous        | 0.987 / 0.852 / 0.882 | 0.988 / 0.862 / 0.923 | 0.996 / 0.878 / 0.918 | 0.996 / 0.876 / 0.903 |
| kuq_controversial    | 0.985 / 0.969 / 0.980 | 0.987 / 0.947 / 0.978 | 0.993 / 0.975 / 0.987 | 0.983 / 0.971 / 0.985 |
| kuq_counterfactual   | 0.995 / 1.000 / 1.000 | 0.996 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.993 / 1.000 / 1.000 |
| kuq_false_assumption | 0.991 / 0.976 / 0.985 | 0.984 / 0.974 / 0.964 | 0.993 / 0.978 / 0.991 | 0.981 / 0.964 / 0.980 |
| kuq_future_unknown   | 0.999 / 0.997 / 0.993 | 1.000 / 0.994 / 0.992 | 1.000 / 0.992 / 0.986 | 0.999 / 0.990 / 0.981 |
| kuq_unsolved_problem | 0.985 / 0.949 / 0.943 | 0.992 / 0.953 / 0.955 | 0.992 / 0.953 / 0.967 | 0.981 / 0.953 / 0.961 |

- **The reader has a margin of 0.066 or more over the better text baseline on two contrasts, as on the base checkpoints.** `triviaqa_known`: the reader reads 0.877 and 0.840 against a better text baseline of 0.750 and 0.664 (margins of 0.127 and 0.176, against 0.131 and 0.191 on the base checkpoints); its test known sides hold 21 and 54 questions (32 and 55 on the base checkpoints). `kuq_ambiguous`: margins of 0.066 and 0.092 (0.105 and 0.079 on the base checkpoints). On every other cross-dataset contrast the largest margin over the better baseline is 0.036 (`kuq_unsolved_problem` on `qwen3-0.6b`).
- **On several contrasts the reader does not exceed the text baselines.** `unknowable_detail` is 1.000 for the reader and both baselines on all four checkpoints. On `false_premise` the reader (0.947 and 0.951) is below the better baseline (0.988 and 0.974). The two contrasts between kinds of unknown questions (`real_vs_synthal`, `real_vs_generated`) have `text_char` at 1.000 and 0.999, above the reader within relation (0.979 and 0.980 on `real_vs_synthal`, 0.974 and 0.968 on `real_vs_generated`) and pooled.
- **Elsewhere the margin is small.** `fictional_generated` has the reader above `text_char` within relation by 0.006 and 0.004 on the post-trained checkpoints; four KUQ categories (`kuq_controversial`, `kuq_counterfactual`, `kuq_false_assumption`, `kuq_future_unknown`) are within 0.010 of the better baseline in either direction. `pre_vs_post_cutoff` is not read on these checkpoints (the post-cutoff contrasts have too few items).
- **Read these cross-dataset rows as a format contrast too.** The better text baseline is at 0.955 or above on every cross-dataset row of the post-trained checkpoints except `triviaqa_known` (0.750 and 0.664) and `kuq_ambiguous` (0.923 and 0.903), so a reader between 0.947 and 1.000 on the others does not show that it reads knowledge; on `false_premise` the reader is below both text baselines.
- **What this does not show.** As above: popularity and the wording of the question are the only surface controls, each value is one set of five readers on one labeling run per checkpoint, and P9b compares the reader with the text baselines only. Whether the reader predicts the errors of the checkpoint better than its output confidence is the question of P9c.

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

#### The post-trained Qwen3 checkpoints

P9c on `qwen3-0.6b` and `qwen3-1.7b` (runs `p9c/qwen3-0.6b/0f8b118c9692` and `p9c/qwen3-1.7b/7c142334e5fe`, made at `fb3b5dd` with a clean tree on a laptop GPU, on the labels of the P9a run of the same checkpoint) has the same predictors and the same sizes of held-out sets (1,937 PopQA, 191 TriviaQA and 210 NQ-open questions) as above. The probe and `lat` read L27 on all four checkpoints (chosen on the validation split of P9c). The question is whether the label-free reader `lat` predicts the errors of a post-trained checkpoint better than the output confidence of the checkpoint. The base columns are those of the base runs above, repeated; a checkpoint answers different questions correctly (correct answers on PopQA: 208, 176, 264, 222 of 1,937 in the column order below), so columns are separate rows of evidence and are never pooled.

PopQA, within relation, AUROC with its interval:

| Predictor          | qwen3-0.6b-base      | qwen3-0.6b           | qwen3-1.7b-base      | qwen3-1.7b           |
| ------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| answer_logprob     | 0.841 [0.796, 0.876] | 0.843 [0.797, 0.881] | 0.883 [0.857, 0.907] | 0.855 [0.822, 0.887] |
| probe              | 0.826 [0.781, 0.864] | 0.793 [0.737, 0.842] | 0.881 [0.856, 0.905] | 0.853 [0.820, 0.885] |
| first_token_prob   | 0.826 [0.779, 0.865] | 0.832 [0.782, 0.875] | 0.879 [0.853, 0.902] | 0.842 [0.807, 0.875] |
| neg_entropy        | 0.824 [0.781, 0.862] | 0.813 [0.759, 0.862] | 0.879 [0.853, 0.903] | 0.847 [0.812, 0.880] |
| mean_token_logprob | 0.836 [0.794, 0.870] | 0.845 [0.797, 0.884] | 0.862 [0.831, 0.889] | 0.841 [0.807, 0.874] |
| consistency        | 0.773 [0.727, 0.820] | 0.803 [0.755, 0.848] | 0.803 [0.763, 0.841] | 0.807 [0.760, 0.846] |
| lat                | 0.787 [0.734, 0.829] | 0.757 [0.693, 0.814] | 0.848 [0.815, 0.877] | 0.814 [0.773, 0.850] |
| p_true             | 0.719 [0.667, 0.769] | 0.739 [0.689, 0.784] | 0.776 [0.726, 0.821] | 0.772 [0.714, 0.825] |
| popularity         | 0.696 [0.641, 0.749] | 0.637 [0.575, 0.701] | 0.776 [0.736, 0.815] | 0.749 [0.700, 0.795] |

PopQA pooled over relations, as AUROC / AURC (lower AURC is better):

| Predictor                          | qwen3-0.6b-base | qwen3-0.6b    | qwen3-1.7b-base | qwen3-1.7b    |
| ---------------------------------- | --------------- | ------------- | --------------- | ------------- |
| probe                              | 0.946 / 0.694   | 0.946 / 0.730 | 0.940 / 0.635   | 0.939 / 0.678 |
| answer_logprob                     | 0.933 / 0.703   | 0.935 / 0.740 | 0.910 / 0.655   | 0.919 / 0.710 |
| neg_entropy                        | 0.907 / 0.710   | 0.932 / 0.743 | 0.899 / 0.661   | 0.906 / 0.715 |
| lat                                | 0.875 / 0.727   | 0.889 / 0.763 | 0.879 / 0.679   | 0.857 / 0.731 |
| template                           | 0.892 / 0.740   | 0.913 / 0.753 | 0.843 / 0.688   | 0.873 / 0.723 |
| popularity                         | 0.588 / 0.811   | 0.515 / 0.851 | 0.648 / 0.748   | 0.640 / 0.784 |
| error rate (risk at full coverage) | 0.893           | 0.909         | 0.864           | 0.885         |

TriviaQA and NQ-open, pooled AUROC with its interval:

| Predictor                | qwen3-0.6b-base      | qwen3-0.6b           | qwen3-1.7b-base      | qwen3-1.7b           |
| ------------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| TriviaQA answer_logprob  | 0.886 [0.836, 0.931] | 0.796 [0.722, 0.869] | 0.819 [0.758, 0.878] | 0.832 [0.771, 0.888] |
| TriviaQA neg_entropy     | 0.862 [0.803, 0.917] | 0.758 [0.664, 0.842] | 0.843 [0.784, 0.894] | 0.798 [0.730, 0.864] |
| TriviaQA probe           | 0.868 [0.809, 0.926] | 0.764 [0.674, 0.843] | 0.811 [0.753, 0.868] | 0.808 [0.743, 0.865] |
| TriviaQA lat             | 0.716 [0.639, 0.795] | 0.710 [0.623, 0.800] | 0.674 [0.601, 0.750] | 0.686 [0.608, 0.756] |
| TriviaQA p_true          | 0.778 [0.699, 0.853] | 0.724 [0.628, 0.820] | 0.772 [0.701, 0.840] | 0.734 [0.657, 0.805] |
| TriviaQA correct answers | 48 of 191            | 38 of 191            | 78 of 191            | 63 of 191            |
| NQ-open answer_logprob   | 0.776 [0.648, 0.880] | 0.867 [0.801, 0.929] | 0.790 [0.697, 0.867] | 0.834 [0.724, 0.923] |
| NQ-open neg_entropy      | 0.740 [0.641, 0.826] | 0.700 [0.598, 0.785] | 0.745 [0.659, 0.831] | 0.750 [0.668, 0.828] |
| NQ-open probe            | 0.795 [0.688, 0.895] | 0.817 [0.705, 0.917] | 0.812 [0.742, 0.878] | 0.879 [0.811, 0.941] |
| NQ-open lat              | 0.783 [0.653, 0.897] | 0.858 [0.730, 0.959] | 0.848 [0.777, 0.913] | 0.861 [0.771, 0.939] |
| NQ-open p_true           | 0.788 [0.683, 0.879] | 0.591 [0.463, 0.721] | 0.694 [0.590, 0.787] | 0.599 [0.458, 0.722] |
| NQ-open correct answers  | 20 of 210            | 12 of 210            | 38 of 210            | 20 of 210            |

What `probe` and `lat` add to the output confidence (the logistic stacker of the five confidence columns against the same stacker plus the predictor, as a paired AUROC difference with its interval; pooled AUROCs):

| Predictor                 | qwen3-0.6b-base         | qwen3-0.6b              | qwen3-1.7b-base         | qwen3-1.7b              |
| ------------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| PopQA confidence stack    | 0.908                   | 0.935                   | 0.900                   | 0.932                   |
| PopQA + lat               | 0.000 [0.000, 0.000]   | 0.000 [-0.002, +0.001] | +0.002 [-0.002, +0.006] | -0.001 [-0.003, +0.002] |
| PopQA + probe             | +0.044 [+0.026, +0.063] | +0.021 [+0.012, +0.031] | +0.044 [+0.028, +0.061] | +0.016 [+0.006, +0.026] |
| TriviaQA confidence stack | 0.869                   | 0.735                   | 0.858                   | 0.837                   |
| TriviaQA + lat            | -0.008 [-0.024, +0.005] | +0.003 [-0.017, +0.024] | -0.006 [-0.018, +0.006] | -0.005 [-0.037, +0.028] |
| TriviaQA + probe          | +0.008 [+0.000, +0.018] | +0.022 [-0.006, +0.050] | +0.013 [-0.004, +0.029] | +0.020 [-0.017, +0.058] |
| NQ-open confidence stack  | 0.819                   | 0.847                   | 0.800                   | 0.866                   |
| NQ-open + lat             | -0.013 [-0.048, +0.023] | +0.006 [-0.011, +0.023] | +0.047 [+0.015, +0.085] | +0.028 [-0.004, +0.060] |
| NQ-open + probe           | -0.003 [-0.027, +0.022] | +0.004 [-0.005, +0.013] | +0.036 [+0.007, +0.066] | +0.040 [+0.002, +0.077] |

- **`lat` does not beat the output confidence within relation on either post-trained checkpoint.** On PopQA it reads 0.757 [0.693, 0.814] on `qwen3-0.6b` and 0.814 [0.773, 0.850] on `qwen3-1.7b`, against 0.843 [0.797, 0.881] and 0.855 [0.822, 0.887] for `answer_logprob` (the best of the six confidence scores is `mean_token_logprob` at 0.845 on `qwen3-0.6b` and `answer_logprob` on `qwen3-1.7b`). The gap is -0.086 and -0.041 (-0.055 and -0.035 on the base checkpoints); the intervals overlap at both, and no paired interval is stored. `lat` is above the popularity baseline (0.637 and 0.749) as a point estimate, with overlapping intervals at both.
- **Pooled, `lat` is below the confidence and below the accuracy rate of the relation.** It reads 0.889 and 0.857 against 0.935 and 0.919 for `answer_logprob`, and below `template` alone (0.913 and 0.873), as on `qwen3-0.6b-base` (0.875 against 0.892); only on `qwen3-1.7b-base` is it above `template` (0.879 against 0.843). Added to the confidence stack it changes the pooled AUROC by -0.0004 [-0.0016, 0.0008] and -0.0006 [-0.0035, 0.0020] on the post-trained checkpoints (+0.0001 and +0.0022 on the base checkpoints), so it adds nothing that the confidence does not carry on PopQA.
- **The probe, which sees the label, is below `answer_logprob` within relation and above it pooled.** Within relation the probe reads 0.793 [0.737, 0.842] and 0.853 [0.820, 0.885] against 0.843 and 0.855; pooled it reads 0.946 and 0.939 against 0.935 and 0.919. Its gain over the confidence stack is +0.021 [0.012, 0.031] and +0.016 [0.006, 0.026], with intervals above zero and a smaller point value than on the base checkpoints (+0.044 on both), where the stack itself is lower (0.908 and 0.900 against 0.935 and 0.932).
- **TriviaQA and NQ-open.** On TriviaQA `lat` reads 0.710 [0.623, 0.800] and 0.686 [0.608, 0.756], below `answer_logprob` on both (0.796 and 0.832). On NQ-open `lat` reads 0.858 [0.730, 0.959] against 0.867 [0.801, 0.929] at 0.6B and 0.861 [0.771, 0.939] against 0.834 [0.724, 0.923] at 1.7B, on 12 and 20 correct answers, with intervals 0.13 to 0.23 wide that overlap. Its increments over the confidence stack have intervals that include zero on both sets (TriviaQA +0.0033 [-0.0173, 0.0239] and -0.0053 [-0.0367, 0.0276], NQ-open +0.0063 [-0.0105, 0.0226] and +0.0276 [-0.0038, 0.0597]). The probe's increment excludes zero only for NQ-open at 1.7B (+0.0400 [0.0023, 0.0768]). The confidence stack of `qwen3-0.6b` is below `answer_logprob` alone on TriviaQA (0.735 against 0.796) and NQ-open (0.847 against 0.867), which is consistent with a stacker fitted on few correct answers (the test sets hold 38 and 12).
- **What this does not show.** Each cell is one run of one checkpoint, the sets of correct answers are small outside PopQA, and the increments are pooled (no within-relation increment is stored). The result is that `lat` does not predict errors better than the output confidence on these two post-trained checkpoints; it does not say that the reader is useless, since it is label-free and above popularity as a point estimate.

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
  0.742 [-0.629, 1.609]: all three include zero. The absolute gaps against the three random arms include zero on
  `gpt2-medium` and `gpt2-large` (for example 0.396 [-0.523, 1.350] against `random0` on `gpt2-medium`); on `gpt2-xl`
  the lower bounds against the three arms are -0.006, 0.008 and -0.272, so only the gap against `random1` excludes
  zero, and only just. The entropy results do not carry over to the probability of the correct answer.
- **What this does and does not show.** The direction changes the entropy of the next-token distribution of a base model
  more than a random direction does and more on unknown questions than on known ones. It is not shown to be a control of
  what the model knows: the gold-answer probability does not move distinguishably, random directions also move the
  entropy on the known items on `gpt2-medium` and `gpt2-xl`, and the stored selectivity of the `gpt2-xl` run, made
  before the interval was oriented, agrees with a recomputation with the current code (every bound within 0.005). The
  runs are one per cell.

## Qwen3: emotions, TruthfulQA, seed agreement, logit lens, steering and control (R4, R6, A1, A2, M1, C2, C4, C5, LR1)

Runs on the base and post-trained checkpoints of Qwen3. R4, R6, A1, A2 and LR1 and the first M1, C2, C4 and C5 runs
were made at `a6e2333` with a clean tree on the laptop GPU (hashes under [Run identity](#run-identity)). The M1, C2, C4
and C5 runs reported below are later runs at `f4d414d` on a rented GPU pod, which replace the earlier ones as the
reported runs; the earlier runs stay tracked and each section names both. R4, A1 and A2 ran on all six
checkpoints (the 4B pair in chain 17, at `89ea7cb` on a rented GPU pod) and R6 on all but `qwen3-4b-base` (its run was cut
when the chain was stopped and is not exported); LR1 ran on the 1.7B and 4B pairs, C2 on the 1.7B pair and `qwen3-4b`,
and M1, C4 and C5 on the two 1.7B checkpoints only. The 4B results follow the 1.7B text in a `#### The 4B pair` block
of each section. The post-trained checkpoints use the chat
template with thinking disabled and the base checkpoints plain tags, as in [LAT reading on
Qwen3](#lat-reading-on-qwen3). Base and post-trained checkpoints are different models: every table has one column per
checkpoint, they are compared within a size, and they are never pooled into one metric. Each checkpoint has one
reported run per experiment, so no difference between checkpoints is a difference between independent runs; the
earlier and later runs of the same checkpoint also differ in places by more than the stated intervals, which the
sections say. The later M1, C2, C4 and C5 runs carry random-direction arms and store the perplexity of neutral text
under the steered model (and, for M1, C2 and C5, the distinct-1 of the generated text); this is said again under each
of them.

### Emotions (R4)

One reader per emotion against the others, with the protocol of R4 on the GPT-2 sizes (10 readers, last token, test
accuracy with its interval and the selected layer; runs `r4/qwen3-0.6b-base/7a3cbcb17a19`,
`r4/qwen3-0.6b/48d56adb146e`, `r4/qwen3-1.7b-base/7f4457fe5392` and `r4/qwen3-1.7b/1ce1c8182cf0`). The test sets are the
same as on GPT-2 (38, 44, 45, 53, 39 and 36 pairs for the six emotions in the order of the table).

| Emotion   | qwen3-0.6b-base          | qwen3-0.6b               | qwen3-1.7b-base          | qwen3-1.7b               |
| --------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| happiness | 0.995 [0.987, 1.000] L18 | 1.000 [1.000, 1.000] L28 | 1.000 [1.000, 1.000] L26 | 1.000 [1.000, 1.000] L26 |
| sadness   | 0.982 [0.977, 0.989] L18 | 0.968 [0.961, 0.975] L23 | 1.000 [1.000, 1.000] L21 | 0.998 [0.993, 1.000] L18 |
| anger     | 1.000 [1.000, 1.000] L19 | 0.980 [0.971, 0.989] L24 | 0.984 [0.978, 0.991] L20 | 0.971 [0.962, 0.982] L20 |
| fear      | 0.992 [0.987, 0.998] L22 | 0.998 [0.994, 1.000] L23 | 1.000 [1.000, 1.000] L26 | 0.996 [0.991, 1.000] L18 |
| disgust   | 0.936 [0.921, 0.951] L6  | 0.987 [0.979, 0.995] L28 | 1.000 [1.000, 1.000] L18 | 0.974 [0.964, 0.985] L1  |
| surprise  | 0.953 [0.939, 0.969] L18 | 0.975 [0.964, 0.986] L16 | 0.981 [0.969, 0.992] L24 | 0.967 [0.953, 0.981] L23 |

- **Near ceiling on all four checkpoints.** Every one of the 24 cells is at 0.936 or above; the lowest is
  `qwen3-0.6b-base` for disgust (0.936 [0.921, 0.951] at L6). With 36 to 53 test pairs per emotion and one run per
  checkpoint, differences of a few hundredths are small against the evaluation sampling that the intervals do not
  cover. An interval of [1.000, 1.000] means that all 10 readers scored every test pair correctly.
- **Baselines at the selected layer.** The random-direction series is at 0.547 to 0.700 on `qwen3-0.6b-base`, 0.558 to
  0.666 on `qwen3-0.6b`, 0.539 to 0.718 on `qwen3-1.7b-base` and 0.562 to 0.668 on `qwen3-1.7b`, and the shuffled-label
  series at 0.475 to 0.531, 0.475 to 0.538, 0.480 to 0.533 and 0.482 to 0.533 (computed from the curves files, over the
  six emotions). The reader is above its random-direction value in every cell; the smallest margin is `qwen3-0.6b-base`
  for disgust (0.936 against 0.700).
- **Base against post-trained, within a size.** At 0.6B the post-trained checkpoint is at or above the base one in 4 of
  6 emotions (for example disgust 0.987 against 0.936, surprise 0.975 against 0.953) and below it for sadness and
  anger. At 1.7B the base checkpoint is above the post-trained one in 5 of 6 emotions and level for happiness (1.000
  and 1.000), for example anger 0.984 against 0.971 and disgust 1.000 against 0.974. There is no common sign.
- **Early selected layers.** The post-trained 1.7B checkpoint selects L1 for disgust (0.974 [0.964, 0.985], with a
  random-direction value of 0.618 there). A high accuracy at layer 1 is consistent with emotion words in the statements
  that a shallow reader can use, as on GPT-2; this is not tested here.
- **Not reported here.** The cosine matrix of the six emotion directions and the embedding are stored in each run and
  are not read in this document.

#### The 4B pair (R4)

The same protocol on the 4B pair (runs `r4/qwen3-4b-base/dd2c7d3f2fb2` and `r4/qwen3-4b/3e8c537bfac8`; 36 blocks). The
test sets are those above (38, 44, 45, 53, 39 and 36 pairs).

| Emotion   | qwen3-4b-base            | qwen3-4b                 |
| --------- | ------------------------ | ------------------------ |
| happiness | 0.992 [0.984, 1.000] L35 | 1.000 [1.000, 1.000] L30 |
| sadness   | 0.995 [0.989, 1.000] L29 | 0.961 [0.952, 0.970] L26 |
| anger     | 1.000 [1.000, 1.000] L26 | 1.000 [1.000, 1.000] L25 |
| fear      | 1.000 [1.000, 1.000] L24 | 1.000 [1.000, 1.000] L27 |
| disgust   | 1.000 [1.000, 1.000] L24 | 0.995 [0.987, 1.000] L8  |
| surprise  | 0.983 [0.972, 0.994] L24 | 0.961 [0.953, 0.969] L20 |

- **Near ceiling on both checkpoints.** Every one of the 12 cells is at 0.961 or above, the lowest being `qwen3-4b` for
  sadness (0.961 [0.952, 0.970] at L26) and surprise (0.961 [0.953, 0.969] at L20); six cells are at 1.000 [1.000,
  1.000]. With 36 to 53 test pairs per emotion and one run per checkpoint, differences of a few hundredths are small
  against the evaluation sampling that the intervals do not cover.
- **Baselines at the selected layer.** The random-direction series is at 0.536 to 0.672 on `qwen3-4b-base` and 0.605 to
  0.708 on `qwen3-4b`, and the shuffled-label series at 0.480 to 0.531 and 0.468 to 0.531 (computed from the curves
  files, over the six emotions). The reader is above its random-direction value in every cell; the smallest margins are
  `qwen3-4b-base` for disgust (1.000 against 0.672) and `qwen3-4b` for fear (1.000 against 0.708).
- **Base against post-trained.** The base checkpoint is above the post-trained one for sadness (0.995 against 0.961),
  disgust (1.000 against 0.995) and surprise (0.983 against 0.961), the post-trained one is above for happiness (1.000
  against 0.992), and anger and fear are level at 1.000. There is no common sign, as at the smaller sizes.
- **Early selected layer.** `qwen3-4b` selects L8 for disgust (0.995 [0.987, 1.000], with a random-direction value of
  0.628 there); the other cells select L20 to L35. A high accuracy at an early layer is consistent with emotion words in
  the statements that a shallow reader can use, as on GPT-2 and at 1.7B; this is not tested here.

### TruthfulQA MC1 against text baselines (R6)

The same questions, split, readers and layer rules as the GPT-2 table above: 511 train, 133 validation and 173 test
questions of 817, every row scored on the same 173 test questions, chance line 0.228, 15 readers for the LAT rows
(runs `r6/qwen3-0.6b-base/74d5b69b5f57`, `r6/qwen3-0.6b/66616910b862`, `r6/qwen3-1.7b-base/713eccd9a4f3` and
`r6/qwen3-1.7b/98b2517a3cb1`). The two text rows are classifiers on the answer texts alone and are identical in every
column because they do not depend on the model. The interval of a LAT row is over the 15 readers and the interval of
the other rows is a Wilson interval over the 173 questions, so the two kinds are not comparable in width.

| Row                              | qwen3-0.6b-base          | qwen3-0.6b               | qwen3-1.7b-base          | qwen3-1.7b               |
| -------------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| standard                         | 0.231 [0.175, 0.299]     | 0.324 [0.258, 0.397]     | 0.197 [0.144, 0.262]     | 0.295 [0.232, 0.367]     |
| heuristic                        | 0.220 [0.164, 0.287]     | 0.145 [0.100, 0.205]     | 0.185 [0.134, 0.249]     | 0.243 [0.185, 0.312]     |
| text_word                        | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     |
| text_char                        | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     |
| LAT truthfulqa, validation-layer | 0.679 [0.671, 0.686]     | 0.644 [0.627, 0.659]     | 0.703 [0.695, 0.710]     | 0.672 [0.664, 0.681]     |
| LAT truthfulqa, best-layer       | 0.683 [0.677, 0.689] L18 | 0.643 [0.626, 0.658] L18 | 0.704 [0.697, 0.710] L16 | 0.672 [0.663, 0.683] L18 |
| LAT arc, validation-layer        | 0.406 [0.370, 0.439]     | 0.367 [0.348, 0.383]     | 0.400 [0.376, 0.424]     | 0.407 [0.388, 0.428]     |
| LAT arc, best-layer              | 0.368 [0.326, 0.408] L13 | 0.333 [0.313, 0.355] L3  | 0.388 [0.353, 0.420] L9  | 0.397 [0.387, 0.407] L21 |
| LAT honesty, validation-layer    | 0.334 [0.322, 0.348]     | 0.346 [0.338, 0.353]     | 0.443 [0.438, 0.448]     | 0.558 [0.541, 0.572]     |
| LAT honesty, best-layer          | 0.401 [0.394, 0.408] L4  | 0.346 [0.339, 0.353] L2  | 0.459 [0.454, 0.464] L19 | 0.573 [0.566, 0.580] L19 |

- **The TruthfulQA-source reader is not separated from the text baselines on any checkpoint.** At the validation layer
  it reaches 0.679, 0.644, 0.703 and 0.672 (in the order of the columns), against `text_word` 0.676 [0.603, 0.742] and
  `text_char` 0.671 [0.597, 0.736]. All four values lie inside both Wilson intervals, which are 0.14 wide. Only the
  base checkpoints are above both text point estimates (`qwen3-1.7b-base` by 0.026 and 0.032, `qwen3-0.6b-base` by
  0.002 and 0.008); `qwen3-0.6b` is below both (by 0.032 and 0.026) and `qwen3-1.7b` is level with them (0.004 below
  `text_word`, 0.002 above `text_char`). The best-layer rows (0.683, 0.643, 0.704, 0.672) pick the layer on the test
  questions, are optimistic, and are inside both text intervals as well.
- **Base against post-trained.** At both sizes the base checkpoint is above its post-trained pair (0.679 against 0.644
  at 0.6B, 0.703 against 0.672 at 1.7B), and the intervals over readers do not overlap within a pair ([0.671, 0.686]
  against [0.627, 0.659] and [0.695, 0.710] against [0.664, 0.681]). Those intervals do not include the sampling of the
  173 questions, which is the width of the text intervals, so this is a difference between two runs and not a result
  about base against post-trained models in general.
- **The other two sources stay below the text baselines.** The ARC-trained reader is at 0.406, 0.367, 0.400 and 0.407
  and the honesty-persona reader at 0.334, 0.346, 0.443 and 0.558 at the validation layer. Neither source contains
  TruthfulQA answers, and both are below `text_char` 0.671 on every checkpoint. The honesty reader rises with size and
  is highest on the post-trained 1.7B checkpoint (0.558 [0.541, 0.572]); on the 0.6B pair it is near the ARC reader.
- **Random and shuffled series.** The highest mean over layers of the random-direction series is 0.396, 0.348, 0.385
  and 0.352 for the TruthfulQA source, 0.329, 0.309, 0.344 and 0.347 for ARC and 0.349, 0.311, 0.376 and 0.336 for
  honesty, and the shuffled-label series stays at or below 0.242 (chance 0.228). The best-layer reader is above the
  highest random value in every row, by 0.024 to 0.320. The TruthfulQA rows are far above it (0.287 to 0.320); the ARC
  rows (0.024 to 0.050) and the honesty rows of the 0.6B pair and of `qwen3-1.7b-base` (0.035 to 0.083) are close to it,
  and only the honesty row of `qwen3-1.7b` is far above (0.237).
- **Standard and heuristic.** The standard log-probability MC1 on the test questions is 0.231 [0.175, 0.299], 0.324
  [0.258, 0.397], 0.197 [0.144, 0.262] and 0.295 [0.232, 0.367], and the heuristic is 0.220, 0.145, 0.185 and 0.243.
  The post-trained checkpoints are above the base ones on the standard row at both sizes. Chance is 0.228.

#### The 4B pair (R6)

The same questions, split, readers and layer rules (run `r6/qwen3-4b/4e4a60094e47`; 15 readers for the LAT rows, 173 test
questions, chance line 0.228). Only the post-trained checkpoint was run: `r6-qwen3-4b-base` was cut when the chain was
stopped, so there is no 4B base column and no pair. The two text rows are those of every column above.

| Row                              | qwen3-4b                 |
| -------------------------------- | ------------------------ |
| standard                         | 0.318 [0.253, 0.391]     |
| heuristic                        | 0.225 [0.170, 0.293]     |
| text_word                        | 0.676 [0.603, 0.742]     |
| text_char                        | 0.671 [0.597, 0.736]     |
| LAT truthfulqa, validation-layer | 0.741 [0.731, 0.750]     |
| LAT truthfulqa, best-layer       | 0.745 [0.736, 0.754] L22 |
| LAT arc, validation-layer        | 0.545 [0.537, 0.553]     |
| LAT arc, best-layer              | 0.545 [0.536, 0.553] L23 |
| LAT honesty, validation-layer    | 0.575 [0.565, 0.584]     |
| LAT honesty, best-layer          | 0.589 [0.582, 0.595] L36 |

- **The TruthfulQA-source reader is the closest to the text baselines of any checkpoint, and is still not separated from
  them.** At the validation layer it reaches 0.741 [0.731, 0.750], above the point estimates of `text_word` 0.676
  [0.603, 0.742] and `text_char` 0.671 [0.597, 0.736] by 0.065 and 0.070. The value is inside the Wilson interval of
  `text_word` (upper bound 0.742) and 0.005 above the upper bound of `text_char` (0.736); the best-layer row (0.745 at
  L22) picks the layer on the test questions, is optimistic, and is above both upper bounds. The two kinds of interval
  are not comparable: the reader's interval is over 15 readers and does not carry the sampling of the 173 questions
  (a width of about 0.14), so this is not a separation. The margin over the text point estimates is the largest in the
  R6 tables (the next is `qwen3-1.7b-base`, 0.026 and 0.032). On the post-trained checkpoints the TruthfulQA reader rises
  with size, 0.644, 0.672 and 0.741.
- **The other two sources stay below the text baselines.** The ARC-trained reader is at 0.545 [0.537, 0.553] and the
  honesty-persona reader at 0.575 [0.565, 0.584] at the validation layer, 0.126 and 0.096 below `text_char`. Neither
  source contains TruthfulQA answers. Both are the highest of their source among the post-trained checkpoints (ARC 0.367,
  0.407 and 0.545; honesty 0.346, 0.558 and 0.575).
- **Random and shuffled series.** The highest mean over layers of the random-direction series is 0.480 for the
  TruthfulQA source, 0.409 for ARC and 0.403 for honesty, and the shuffled-label series stays at or below 0.243 (chance
  0.228). The best-layer reader is above the highest random value in every row, by 0.265, 0.136 and 0.185. Unlike the
  ARC rows of the smaller checkpoints (0.024 to 0.050 above the random maximum), the ARC row is clearly above it here.
- **Standard and heuristic.** The standard log-probability MC1 on the test questions is 0.318 [0.253, 0.391] and the
  heuristic is 0.225 [0.170, 0.293]. On all 817 questions the run records 0.332 [0.300, 0.365] and 0.237 [0.210, 0.268].
  The standard row is above the chance line as a point estimate, with a lower bound of 0.253.

### Seed agreement and logit lens (A1, A2)

Both analyses read the readers saved by the R1, R2, R3, R5 and R7 runs of the same checkpoint (hashes under Run
identity); neither runs a forward pass over data.

**A1, seed agreement.** The mean cosine between the directions of the 10 seeds at layer 14 (relative depth 0.5 of 28
blocks), with the 95 percent t-interval over seeds, and in parentheses the range of the mean over layers 1 to 28. Two
random unit directions have a mean absolute cosine of 0.025 at 0.6B (1,024 dimensions) and 0.018 at 1.7B (2,048
dimensions). Runs `a1/qwen3-0.6b-base/4e8f8cfa0021`, `a1/qwen3-0.6b/0cb04910086e`, `a1/qwen3-1.7b-base/769590eef456`
and `a1/qwen3-1.7b/2326d9ec9b7d`.

| Reading | qwen3-0.6b-base                       | qwen3-0.6b                            | qwen3-1.7b-base                       | qwen3-1.7b                            |
| ------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| r1      | 0.472 [0.399, 0.545] (0.273 to 0.791) | 0.423 [0.355, 0.491] (0.220 to 0.629) | 0.578 [0.539, 0.616] (0.301 to 0.807) | 0.507 [0.461, 0.552] (0.301 to 0.661) |
| r2      | 0.306 [0.220, 0.392] (0.139 to 0.400) | 0.340 [0.238, 0.442] (0.087 to 0.343) | 0.389 [0.321, 0.457] (0.158 to 0.409) | 0.386 [0.301, 0.471] (0.128 to 0.388) |
| r3      | 0.335 [0.269, 0.402] (0.140 to 0.601) | 0.405 [0.317, 0.492] (0.139 to 0.561) | 0.335 [0.245, 0.425] (0.137 to 0.734) | 0.497 [0.412, 0.582] (0.135 to 0.667) |
| r5      | 0.975 [0.973, 0.977] (0.959 to 0.992) | 0.980 [0.979, 0.981] (0.961 to 0.990) | 0.979 [0.977, 0.981] (0.955 to 0.991) | 0.979 [0.978, 0.980] (0.960 to 0.990) |
| r7      | 0.562 [0.499, 0.624] (0.418 to 0.768) | 0.571 [0.510, 0.633] (0.403 to 0.752) | 0.645 [0.602, 0.688] (0.418 to 0.895) | 0.639 [0.599, 0.678] (0.422 to 0.872) |

- **All 20 values are far above the random level.** R1, R2, R3 and R7 are between 0.306 and 0.645 at layer 14, so two
  seeds recover directions that overlap only partly, as on `gpt2` (0.252 to 0.525). Layer 14 is not where they agree
  most: for R1, R3 and R7 the maximum over layers is at layer 18 to 23 on every checkpoint (R1 0.629 to 0.807, R3 0.561
  to 0.734, R7 0.752 to 0.895).
- **R5 agrees at 0.975 to 0.980 at layer 14 and at 0.955 or above at every layer, on all four checkpoints**, while its
  transfer test accuracy is 0.456 on `qwen3-0.6b-base` (below chance), 0.529, 0.778 and 0.781 (see the Qwen3 reading
  table). A direction that is stable across seeds is not for that reason a direction that transfers, as on `gpt2`.
- **Base against post-trained.** The intervals over seeds overlap in every pair of checkpoints except R5 at 0.6B (0.975
  against 0.980, a difference of 0.005). The point estimates differ in sign by reading: R1 is higher on the base
  checkpoint (0.472 against 0.423, 0.578 against 0.507) and R3 on the post-trained one (0.405 against 0.335, 0.497
  against 0.335).

**A2, logit lens.** Each seed's direction is read through the final norm and the unembedding, scaled to z-scores of the
logits over the vocabulary, with the same six positive and six negative pole tokens per reading as on `gpt2` (no pole
word was dropped). The separation is the mean z-score of the positive poles minus that of the negative ones, at the
layer with the highest mean separation, with a 95 percent percentile bootstrap interval over the 10 seeds; the best
layer is chosen on the same curve, so the values are optimistic. In parentheses are the mean and standard deviation of
the same quantity for 100 random directions, as stored by the run. Runs `a2/qwen3-0.6b-base/109c4aa851bf`,
`a2/qwen3-0.6b/91b62e58953f`, `a2/qwen3-1.7b-base/09b991efaf7f` and `a2/qwen3-1.7b/1e528ddfbffa`.

| Reading | qwen3-0.6b-base                     | qwen3-0.6b                          | qwen3-1.7b-base                     | qwen3-1.7b                          |
| ------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| r1      | L28 5.72 [5.51, 5.93] (0.09, 0.66)  | L19 4.21 [3.90, 4.51] (0.10, 0.67)  | L22 6.20 [6.09, 6.35] (0.06, 0.67)  | L28 5.08 [4.87, 5.29] (0.06, 0.67)  |
| r2      | L22 3.72 [3.31, 4.16] (-0.03, 0.52) | L22 3.62 [3.16, 4.06] (-0.02, 0.51) | L22 3.80 [3.33, 4.31] (0.01, 0.69)  | L22 4.34 [3.64, 4.93] (0.00, 0.69)  |
| r3      | L24 3.67 [3.32, 4.02] (0.00, 0.69)  | L24 3.13 [2.68, 3.56] (0.01, 0.70)  | L26 4.74 [4.56, 4.91] (-0.02, 0.57) | L28 4.03 [3.76, 4.33] (-0.03, 0.58) |
| r5      | L28 3.44 [3.36, 3.52] (-0.03, 0.60) | L27 4.21 [4.12, 4.30] (-0.01, 0.60) | L26 4.61 [4.49, 4.73] (-0.01, 0.68) | L26 4.09 [3.96, 4.20] (0.00, 0.69)  |
| r7      | L22 3.81 [3.52, 4.09] (-0.02, 0.61) | L22 2.65 [2.24, 3.01] (-0.01, 0.64) | L22 5.86 [5.72, 6.01] (-0.05, 0.62) | L25 5.45 [5.11, 5.79] (-0.05, 0.63) |

- **Every reading separates its poles far above the random spread.** The lowest separation is 2.65 (R7, `qwen3-0.6b`),
  and the random-direction standard deviation is 0.51 to 0.70, so every value is more than 4 random standard deviations
  above the random mean. On `gpt2` the R3 and R7 separations (0.91 and 0.89) stayed below two random standard
  deviations; on all four Qwen3 checkpoints they are 3.13 to 4.74 (R3) and 2.65 to 5.86 (R7).
- **The best layer is late.** It lies between layer 19 and layer 28 of 28 for every reading, and is the last block for
  R1 on `qwen3-0.6b-base` and `qwen3-1.7b`, R5 on `qwen3-0.6b-base` and R3 on `qwen3-1.7b`.
- **Base against post-trained.** The base checkpoint has the higher separation in 8 of the 10 pairs of readings. R1 is
  5.72 [5.51, 5.93] against 4.21 [3.90, 4.51] at 0.6B and 6.20 [6.09, 6.35] against 5.08 [4.87, 5.29] at 1.7B, with
  non-overlapping intervals. The exceptions are R5 at 0.6B (3.44 against 4.21) and R2 at 1.7B (3.80 against 4.34, with
  overlapping intervals).
- **The top tokens only partly match the concept.** The stored rows list the 10 highest and lowest tokens per layer. At
  the best layer the lowest R1 tokens are variants of "negative" on all four checkpoints, and on both 1.7B checkpoints
  the highest R7 tokens are variants of "Yes" and the lowest are variants of "False". On the other readings the top
  tokens mix words of the concept (R3 on `qwen3-1.7b-base`, lowest tokens: " incorrect", " wrong", " violating") with
  unrelated or non-English tokens. R5 on `qwen3-1.7b` has a separation of 4.09 while its highest five tokens are
  punctuation marks and a code token and its lowest five are ellipses, not honesty words, so a high pole separation is
  not read as a reading of honesty in the vocabulary.

#### The 4B pair (A1, A2)

Both analyses read the readers saved by the R1, R2, R3, R5 and R7 runs of the same 4B checkpoint (their hashes are stored
in each summary and are those of the 4B runs listed under Run identity). **A1**: runs `a1/qwen3-4b-base/80e2777e6b67`
and `a1/qwen3-4b/819f34624d6a`; the layer is 18 (relative depth 0.5 of 36 blocks) and two random unit directions have a
mean absolute cosine of 0.016 in the 2,560 dimensions. The cell is the mean cosine with its 95 percent t-interval over
seeds and, in parentheses, the range of the mean over layers 1 to 36.

| Reading | qwen3-4b-base                         | qwen3-4b                              |
| ------- | ------------------------------------- | ------------------------------------- |
| r1      | 0.497 [0.450, 0.543] (0.237 to 0.755) | 0.459 [0.404, 0.514] (0.306 to 0.488) |
| r2      | 0.406 [0.335, 0.478] (0.124 to 0.422) | 0.396 [0.316, 0.476] (0.118 to 0.406) |
| r3      | 0.624 [0.585, 0.664] (0.084 to 0.771) | 0.535 [0.469, 0.602] (0.113 to 0.794) |
| r5      | 0.981 [0.980, 0.982] (0.955 to 0.993) | 0.979 [0.978, 0.980] (0.961 to 0.994) |
| r7      | 0.685 [0.652, 0.718] (0.491 to 0.947) | 0.738 [0.708, 0.768] (0.457 to 0.941) |

- **All 10 values are far above the random level.** R1, R2, R3 and R7 are between 0.396 and 0.738 at layer 18, so two
  seeds recover directions that overlap only partly, as at the smaller sizes. Layer 18 is not where they agree most: the
  maximum over layers is at L23 to L24 for R1 (0.755 on the base checkpoint, 0.488 on the post-trained one), L28 and L29
  for R3 (0.771 and 0.794), L32 and L29 for R7 (0.947 and 0.941), and at the last block for R2 on both (0.422 and 0.406).
- **R5 agrees at 0.981 and 0.979 at layer 18 and at 0.955 or above at every layer** (maximum 0.993 and 0.994 at L4),
  while its transfer test accuracy is 0.658 and 0.758 (see the Qwen3 reading table). A direction that is stable across
  seeds is not for that reason a direction that transfers.
- **Base against post-trained.** The intervals over seeds overlap in every pair except R5 (0.981 against 0.979, a
  difference of 0.003). The base checkpoint is higher for R1, R2, R3 and R5, and the post-trained one for R7 (0.738
  against 0.685). R3 is higher on the base checkpoint (0.624 against 0.535), the opposite sign from 0.6B and 1.7B. The
  R1 curve of `qwen3-4b` is low at every layer (0.306 to 0.488), which goes with its R1 accuracy of 0.760; no run here
  tests whether the two are related.

**A2**: runs `a2/qwen3-4b-base/8edf53991111` and `a2/qwen3-4b/2a18226fa458`, with the same pole tokens as above (no pole
word was dropped), the separation at the layer with the highest mean separation, a 95 percent percentile bootstrap
interval over the 10 seeds and, in parentheses, the mean and standard deviation of 100 random directions. The best layer
is chosen on the same curve, so the values are optimistic.

| Reading | qwen3-4b-base                       | qwen3-4b                            |
| ------- | ----------------------------------- | ----------------------------------- |
| r1      | L36 6.22 [5.94, 6.46] (0.00, 0.71)  | L31 2.04 [1.47, 2.60] (0.01, 0.72)  |
| r2      | L31 4.49 [3.95, 5.08] (-0.06, 0.70) | L36 2.85 [2.30, 3.38] (-0.05, 0.67) |
| r3      | L33 3.87 [3.62, 4.09] (-0.04, 0.57) | L27 3.77 [3.52, 3.99] (-0.03, 0.58) |
| r5      | L20 2.71 [2.64, 2.77] (0.07, 0.72)  | L28 3.72 [3.61, 3.83] (0.09, 0.73)  |
| r7      | L23 5.48 [5.34, 5.64] (0.03, 0.62)  | L27 6.36 [6.13, 6.62] (0.04, 0.64)  |

- **Every reading separates its poles above the random spread, but one is close.** The lowest separation is R1 on
  `qwen3-4b` (2.04 [1.47, 2.60]), 2.8 random standard deviations above the random mean; the other nine values are 3.6 to
  9.9 random standard deviations above it, against more than 4 for every value at the smaller sizes.
- **The best layer is late.** It lies between L20 and L36 of 36, and is the last block for R1 on `qwen3-4b-base` and for
  R2 on `qwen3-4b`.
- **Base against post-trained.** The base checkpoint has the higher separation for R1 (6.22 against 2.04), R2 (4.49
  against 2.85) and R3 (3.87 against 3.77, intervals overlapping), the post-trained one for R5 (3.72 against 2.71) and R7
  (6.36 against 5.48); the intervals of R1, R2, R5 and R7 do not overlap. The 4B pair splits 3 to 2, where the base
  checkpoint was higher in 8 of 10 pairs at 0.6B and 1.7B.
- **The top tokens only partly match the concept.** At the best layer the lowest five R1 tokens of the base checkpoint
  are variants of "negative" (English and Chinese) and its highest five include Chinese words of praise and " marvel";
  the highest five R1 tokens of `qwen3-4b` are place words and code fragments, not sentiment words. The lowest five R7 tokens of `qwen3-4b` are variants of
  "incorrect" and "false" (" incorrect", " false", "false", " False", "Incorrect") while its highest five (for example
  " successes" and " abundant") are not truth words. The lowest R3 tokens of `qwen3-4b` are " unacceptable", " illegal",
  " unethical" and two Chinese words, and the highest R3 tokens of the base checkpoint are variants of "yes". Neither
  pole of R5 lists clear honesty or dishonesty words on either checkpoint, so a high pole separation (3.72 for R5 on `qwen3-4b`) is not read
  as a reading of honesty in the vocabulary.

### Six direction finders on utility (M1)

M1 reads utility (ETHICS) with six finders and compares them on held-out reading, on steering and on erasure, as on `gpt2` (runs `m1/qwen3-1.7b-base/e9279a87db07` and `m1/qwen3-1.7b/77752112ddbe`, made at `f4d414d` on a rented GPU pod; 10 readers per finder, the first 3 used for steering, 250 prompts, one sample per prompt, steered at layers 11, 15 and 20). These runs replace `m1/qwen3-1.7b-base/3fe3a8644289` and `m1/qwen3-1.7b/de2cdd3146ac` (made at `a6e2333` on the laptop GPU) as the reported runs; the earlier runs stay tracked. The configs differ in the `fluency` switch only (off in the earlier runs, on here), so these runs add the perplexity and distinct-1 of the steered text. The numbers moved between the two runs of each checkpoint: the test accuracy of a finder by at most 0.002 with the same layers, the win rate of a finder by at most 0.069 (base) and 0.029 (post-trained), and the random direction on the post-trained checkpoint by -0.032, -0.091 and -0.085 at 0.1, 0.2 and 0.3, which is more than the width of the Wilson intervals; those intervals do not include variation between runs. The win rate is the share of (prompt, reader) pairs, 750 per cell (747 to 750 where a pair was lost), in which the completion steered with +s is scored higher by the primary sentiment judge than the completion steered with -s on the same seed, so 0.5 means no effect. The prompts are shared by all finders and readers, so the Wilson intervals are narrower than the variation between prompt draws. Test accuracy is on 4,808 pairs, at the layer of each finder.

`qwen3-1.7b-base`:

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.767 [0.765, 0.768] | L26   | 0.685 [0.651, 0.718] | 0.729 [0.696, 0.760] | 0.805 [0.775, 0.832] | 0.788 [0.757, 0.816] |
| pca_raw   | 0.767 [0.765, 0.770] | L26   | 0.630 [0.594, 0.664] | 0.703 [0.669, 0.734] | 0.757 [0.725, 0.787] | 0.748 [0.716, 0.778] |
| mean_diff | 0.776 [0.774, 0.778] | L26   | 0.720 [0.686, 0.751] | 0.841 [0.813, 0.866] | 0.896 [0.872, 0.916] | 0.801 [0.771, 0.828] |
| logistic  | 0.767 [0.762, 0.772] | L18   | 0.645 [0.610, 0.678] | 0.755 [0.723, 0.784] | 0.807 [0.777, 0.833] | 0.749 [0.717, 0.779] |
| kmeans    | 0.770 [0.769, 0.771] | L26   | 0.591 [0.555, 0.625] | 0.667 [0.632, 0.699] | 0.752 [0.720, 0.782] | 0.785 [0.755, 0.813] |
| random    | 0.645 [0.600, 0.690] | L24   | 0.499 [0.464, 0.535] | 0.475 [0.439, 0.510] | 0.480 [0.444, 0.516] | 0.509 [0.474, 0.545] |

`qwen3-1.7b`:

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.781 [0.777, 0.784] | L20   | 0.738 [0.706, 0.769] | 0.824 [0.795, 0.850] | 0.920 [0.898, 0.937] | 0.856 [0.829, 0.879] |
| pca_raw   | 0.786 [0.785, 0.787] | L18   | 0.716 [0.683, 0.747] | 0.839 [0.811, 0.863] | 0.919 [0.897, 0.936] | 0.881 [0.856, 0.903] |
| mean_diff | 0.786 [0.784, 0.787] | L20   | 0.728 [0.695, 0.759] | 0.868 [0.842, 0.890] | 0.920 [0.898, 0.937] | 0.783 [0.752, 0.811] |
| logistic  | 0.765 [0.758, 0.772] | L18   | 0.593 [0.558, 0.628] | 0.692 [0.658, 0.724] | 0.743 [0.710, 0.773] | 0.580 [0.544, 0.615] |
| kmeans    | 0.783 [0.782, 0.785] | L20   | 0.705 [0.671, 0.736] | 0.840 [0.812, 0.864] | 0.921 [0.900, 0.939] | 0.856 [0.829, 0.879] |
| random    | 0.640 [0.598, 0.680] | L18   | 0.496 [0.460, 0.532] | 0.475 [0.439, 0.510] | 0.477 [0.442, 0.513] | 0.513 [0.478, 0.549] |

- **Reading and steering rank the finders differently on the two checkpoints.** On the base checkpoint `mean_diff` steers most (0.896 [0.872, 0.916] at 0.3) and `kmeans` and `pca_raw` least (0.752 and 0.757). On the post-trained one `kmeans` (0.921), `pca_diff` (0.920) and `mean_diff` (0.920) steer most and `logistic` least (0.743). Test accuracy of the five finders is within 0.009 of each other on the base checkpoint (0.767 to 0.776) and within 0.021 on the post-trained one (0.765 to 0.786), so it does not order them. The `logistic` finder has the lowest test accuracy (0.765) and the lowest win rate on the post-trained checkpoint, and the check judge puts it at 0.580 [0.544, 0.615], near the random direction (0.513 [0.478, 0.549]).
- **The random direction.** Its test accuracy is 0.645 [0.600, 0.690] and 0.640 [0.598, 0.680], 0.12 to 0.15 below the five finders and above the random direction of M1 on `gpt2` (0.533). Its win rate does not leave 0.5 on either checkpoint in this run: every interval includes 0.5 at 0.1, 0.2 and 0.3 on the primary judge (base 0.499, 0.475, 0.480; post-trained 0.496, 0.475, 0.477) and on the check judge at 0.3 (0.509 [0.474, 0.545] and 0.513 [0.478, 0.549]). The earlier run of the post-trained checkpoint had it at 0.565 and 0.563 at 0.2 and 0.3, above 0.5, and of the base checkpoint at 0.440 at 0.3, below it; neither reappears. At 0.3 the finders are above the random win rate by 0.272 (`kmeans`) to 0.416 (`mean_diff`) on the base checkpoint and by 0.265 (`logistic`) to 0.444 (`kmeans`) on the post-trained one.
- **Judges and empty completions.** The share of swept settings with the same dose sign under both judges is 0.889 on the base and 0.833 on the post-trained checkpoint. The share of empty completions over all generations is 0.00004 on the base checkpoint and 0 on the post-trained one.

Cost to the text. Unsteered, the neutral text has a perplexity of 18.6 on the base checkpoint and 33.7 on the post-trained one, and the generated text a distinct-1 of 0.215 and 0.174. The tables give the mean over the three steering readers, with the ratio to the unsteered perplexity in parentheses at the largest strength.

`qwen3-1.7b-base`:

| Finder    | Perplexity +0.1 | Perplexity +0.2 | Perplexity +0.3 | Distinct-1 +0.3 | Perplexity -0.3 | Distinct-1 -0.3 |
| --------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| pca_diff  | 19.4            | 22.6            | 32.7 (1.76x)    | 0.212           | 24.7 (1.33x)    | 0.173           |
| pca_raw   | 19.4            | 23.1            | 36.6 (1.96x)    | 0.179           | 23.7 (1.27x)    | 0.188           |
| mean_diff | 19.5            | 22.4            | 29.5 (1.58x)    | 0.206           | 25.9 (1.39x)    | 0.171           |
| logistic  | 19.4            | 21.8            | 27.2 (1.46x)    | 0.195           | 26.9 (1.44x)    | 0.175           |
| kmeans    | 19.4            | 23.2            | 36.6 (1.97x)    | 0.184           | 24.0 (1.29x)    | 0.195           |
| random    | 19.0            | 20.6            | 24.1 (1.29x)    | 0.210           | 24.1 (1.29x)    | 0.197           |

`qwen3-1.7b`:

| Finder    | Perplexity +0.1 | Perplexity +0.2 | Perplexity +0.3 | Distinct-1 +0.3 | Perplexity -0.3 | Distinct-1 -0.3 |
| --------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| pca_diff  | 36.3            | 44.1            | 65.5 (1.95x)    | 0.169           | 50.4 (1.50x)    | 0.124           |
| pca_raw   | 37.0            | 45.2            | 64.9 (1.93x)    | 0.176           | 48.1 (1.43x)    | 0.110           |
| mean_diff | 37.4            | 47.2            | 74.1 (2.20x)    | 0.169           | 48.9 (1.45x)    | 0.111           |
| logistic  | 37.7            | 47.0            | 66.2 (1.97x)    | 0.155           | 46.9 (1.39x)    | 0.129           |
| kmeans    | 37.3            | 46.2            | 67.6 (2.01x)    | 0.171           | 46.9 (1.39x)    | 0.118           |
| random    | 34.6            | 38.3            | 46.7 (1.39x)    | 0.160           | 48.0 (1.42x)    | 0.151           |

- **The largest strength costs the text on both checkpoints.** At +0.3 the perplexity of the finders is 1.46 to 1.97 times the unsteered one on the base checkpoint and 1.93 to 2.20 times on the post-trained one (the random direction: 1.29 and 1.39), and the distinct-1 is 0.179 to 0.212 and 0.155 to 0.176. By the rule of [Sentiment steering on Qwen3](#sentiment-steering-on-qwen3-c1) (mean perplexity at most 1.5 times the unsteered one and mean distinct-1 at least 0.15), only `logistic` (1.46 times, distinct-1 0.195) passes at +0.3 on the base checkpoint and no finder passes on the post-trained one. The win rates of about 0.92 at 0.3 on the post-trained checkpoint (four of the five finders; `logistic` is 0.743) are therefore scores of text that fails the rule, and are not read as steering that works.
- **Which strengths pass.** On the base checkpoint 0.1 and 0.2 pass in both directions for every finder (ratios at most 1.05 and 1.25, distinct-1 of at least 0.19); among the passing settings at 0.2 `mean_diff` wins most (0.841 [0.813, 0.866]). On the post-trained checkpoint 0.1 passes for every finder (ratios at most 1.12 in both directions, distinct-1 of at least 0.156), and 0.2 does not: at -0.2 the distinct-1 of the five finders is 0.138 to 0.148, below the cutoff (the mean perplexity ratio of a finder is at most 1.40 at +0.2). At 0.1 the finders win 0.593 (`logistic`) to 0.738 (`pca_diff`), against 0.496 for the random direction. In this run the post-trained checkpoint loses fluency at a smaller relative strength than the base one.

Erasure. Projecting a finder's direction out of the activations and fitting a fresh logistic probe on the stored activations of the control layers gives the probe accuracy before and after in the next tables, and the behavioral gap (the score after the more pleasant scenario of a pair minus after the less pleasant one, 100 pairs) is given for the primary and the check judge. The `none` row is the gap without a projection.

`qwen3-1.7b-base`:

| Finder               | Probe before         | Probe after          | Gap primary           | Gap check             |
| -------------------- | -------------------- | -------------------- | --------------------- | --------------------- |
| pca_diff             | 0.720 [0.641, 0.773] | 0.674 [0.595, 0.740] | 0.066 [-0.022, 0.157] | 0.072 [0.034, 0.119]  |
| pca_raw              | 0.720 [0.641, 0.773] | 0.671 [0.617, 0.738] | 0.111 [0.042, 0.205]  | 0.120 [0.099, 0.159]  |
| mean_diff            | 0.720 [0.641, 0.773] | 0.500 [0.500, 0.500] | 0.092 [0.026, 0.128]  | 0.092 [0.055, 0.113]  |
| logistic             | 0.720 [0.641, 0.773] | 0.690 [0.575, 0.776] | 0.082 [0.072, 0.094]  | 0.110 [0.098, 0.130]  |
| kmeans               | 0.720 [0.641, 0.773] | 0.672 [0.617, 0.734] | 0.148 [0.066, 0.254]  | 0.104 [0.059, 0.143]  |
| random               | 0.720 [0.641, 0.773] | 0.720 [0.641, 0.774] | 0.123 [0.087, 0.147]  | 0.105 [0.082, 0.126]  |
| none (no projection) |                      |                      | 0.072 [-0.059, 0.201] | 0.074 [-0.042, 0.189] |

`qwen3-1.7b`:

| Finder               | Probe before         | Probe after          | Gap primary          | Gap check             |
| -------------------- | -------------------- | -------------------- | -------------------- | --------------------- |
| pca_diff             | 0.721 [0.635, 0.772] | 0.643 [0.559, 0.710] | 0.110 [0.059, 0.162] | 0.114 [0.100, 0.125]  |
| pca_raw              | 0.721 [0.635, 0.772] | 0.638 [0.571, 0.693] | 0.198 [0.189, 0.211] | 0.163 [0.127, 0.227]  |
| mean_diff            | 0.721 [0.635, 0.772] | 0.500 [0.500, 0.500] | 0.128 [0.103, 0.147] | 0.094 [0.068, 0.128]  |
| logistic             | 0.721 [0.635, 0.772] | 0.721 [0.631, 0.784] | 0.125 [0.101, 0.141] | 0.130 [0.109, 0.166]  |
| kmeans               | 0.721 [0.635, 0.772] | 0.641 [0.578, 0.712] | 0.171 [0.149, 0.200] | 0.130 [0.087, 0.169]  |
| random               | 0.721 [0.635, 0.772] | 0.721 [0.635, 0.772] | 0.179 [0.150, 0.206] | 0.117 [0.071, 0.145]  |
| none (no projection) |                      |                      | 0.169 [0.064, 0.277] | 0.081 [-0.013, 0.179] |

- **The probe is erased for one finder.** The probe accuracy is 0.720 before the projection on the base checkpoint and 0.721 on the post-trained one. After `mean_diff` it is 0.500, after `logistic` it is 0.690 and 0.721, and after `pca_diff`, `pca_raw` and `kmeans` it is 0.638 to 0.674. After the random direction it is unchanged. The `logistic` finder, which erases the probe partly on `gpt2` (0.563), leaves it at its value before on the post-trained checkpoint.
- **The behavioral gap does not move.** It is 0.072 [-0.059, 0.201] and 0.169 [0.064, 0.277] without a projection, and 0.066 to 0.148 and 0.110 to 0.198 under the projection of each finder or of the random direction, with every interval overlapping the unprojected one; on the check judge the unprojected gaps are 0.074 and 0.081 and the projected ones 0.072 to 0.120 and 0.094 to 0.163. As on `gpt2`, the two operationalizations of erasure disagree: the probe is erased for one finder and the behavioral gap for none.

### Emotion steering (C2)

C2 steers along the logistic reader of each of the six R4 emotions and measures, with the primary emotion judge, the probability of every emotion in the completions (runs `c2/qwen3-1.7b-base/4133e19728fa` and `c2/qwen3-1.7b/1b87ff98a318`, made at `f4d414d` on a rented GPU pod; 100 prompts, 2 samples per prompt, 3 steering seeds, relative strength +0.3). These runs replace `c2/qwen3-1.7b-base/37d187b75a7f` and `c2/qwen3-1.7b/227d782df8ba` (made at `a6e2333` on the laptop GPU) as the reported runs; the earlier runs stay tracked. The configs differ in `n_random` only (absent earlier, 3 here), so these runs add three random arms (unit directions norm-matched to the steering arm, with the same window, coefficients, operator, mode and mask) and the fluency measure. The reading layer moved between the two runs of each checkpoint: it is L20 with steering over layers 19, 20 and 21 on the base checkpoint (L18 earlier) and L18 with layers 17, 18 and 19 on the post-trained one (L16 earlier), so the two checkpoints are not steered at the same depth in either run. The range is the minimum and maximum over the three steering seeds; the change is the own-emotion probability at +0.3 minus the unsteered one. The last column is the range of the own-emotion change under the three random arms.

`qwen3-1.7b-base`:

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.141         | 0.468 [0.434, 0.489]  | +0.327          | 0.327       | 0.707       | +0.380        | sadness -0.058                 | sadness -0.086               | -0.041 to +0.008        |
| sadness   | 0.077         | 0.287 [0.211, 0.366]  | +0.211          | 0.109       | 0.263       | +0.154        | happiness -0.040               | anger -0.056                 | -0.024 to -0.009        |
| anger     | 0.048         | 0.221 [0.218, 0.226]  | +0.173          | 0.075       | 0.445       | +0.370        | happiness -0.127               | disgust +0.366               | -0.009 to +0.007        |
| fear      | 0.069         | 0.344 [0.295, 0.404]  | +0.275          | 0.073       | 0.214       | +0.141        | happiness -0.105               | happiness -0.138             | -0.008 to +0.021        |
| disgust   | 0.059         | 0.197 [0.130, 0.281]  | +0.138          | 0.097       | 0.172       | +0.075        | happiness -0.107               | happiness -0.162             | -0.012 to +0.006        |
| surprise  | 0.091         | 0.270 [0.238, 0.290]  | +0.179          | 0.070       | 0.157       | +0.087        | happiness -0.104               | happiness -0.073             | -0.002 to +0.016        |

`qwen3-1.7b`:

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.116         | 0.363 [0.300, 0.403]  | +0.247          | 0.327       | 0.662       | +0.335        | sadness -0.051                 | sadness -0.093               | -0.038 to -0.016        |
| sadness   | 0.072         | 0.090 [0.069, 0.109]  | +0.018          | 0.124       | 0.105       | -0.019        | happiness -0.013               | surprise -0.042              | -0.029 to -0.011        |
| anger     | 0.035         | 0.107 [0.099, 0.122]  | +0.072          | 0.047       | 0.230       | +0.183        | happiness -0.086               | happiness -0.235             | +0.001 to +0.003        |
| fear      | 0.051         | 0.408 [0.373, 0.431]  | +0.357          | 0.060       | 0.424       | +0.364        | happiness -0.096               | happiness -0.244             | -0.006 to +0.040        |
| disgust   | 0.052         | 0.184 [0.146, 0.244]  | +0.132          | 0.074       | 0.172       | +0.098        | fear +0.120                    | happiness -0.230             | -0.003 to +0.002        |
| surprise  | 0.115         | 0.215 [0.192, 0.247]  | +0.100          | 0.084       | 0.095       | +0.010        | happiness -0.085               | happiness -0.135             | -0.024 to +0.032        |

- **On the base checkpoint all six emotions rise toward their own score on the primary judge**, by +0.138 (disgust) to +0.327 (happiness), with the range over seeds above the unsteered value in every row. The hit rate (the share of rows of the change matrix whose largest entry is on the diagonal) is 1.000 and the specificity (the mean diagonal change minus the mean off-diagonal change) is 0.249; on the check judge they are 1.000 and 0.230. For anger the check judge also moves disgust by +0.366, against +0.370 on the diagonal. The correlation between the judges over the change matrix is 0.833.
- **On the post-trained checkpoint five emotions rise and sadness does not.** Fear (+0.357) and happiness (+0.247) rise most, then disgust (+0.132), surprise (+0.100) and anger (+0.072). The range of sadness ([0.069, 0.109], change +0.018) contains the unsteered value (0.072). The primary hit rate is 1.000 and the specificity 0.177; on the check judge the hit rate is 0.500 (it misses sadness, disgust and surprise) and the specificity 0.187. The correlation between the judges is 0.891, but the diagonals differ: the check judge gives sadness -0.019 and surprise +0.010 where the primary judge gives +0.018 and +0.100.
- **Random arms.** The own-emotion change under a random arm is between -0.041 and +0.021 on the base checkpoint and between -0.038 and +0.040 on the post-trained one, and the largest absolute change of any emotion under any random arm and coefficient is 0.046 and 0.076. On the base checkpoint all six own-emotion changes (+0.138 to +0.327) are above the random-arm maximum. On the post-trained checkpoint happiness, fear, disgust and surprise are above it, and anger (+0.072) and sadness (+0.018) are not, so those two are not separated from an arbitrary edit of the same size.
- **Base against post-trained.** Sadness is steered on the base checkpoint (+0.211) and not on the post-trained one (+0.018); anger is steered more on the base one (+0.173 against +0.072); fear is steered more on the post-trained one (+0.357 against +0.275). The layers differ, and each emotion has one reader per checkpoint, so this is a contrast between two runs.
- **Off-diagonal.** In all 5 non-happiness rows on the base checkpoint and in 4 of the 5 on the post-trained one the largest off-diagonal change on the primary judge is a fall of happiness (-0.040 to -0.127 and -0.013 to -0.096); the exception is disgust on the post-trained checkpoint (fear +0.120). Steering toward an emotion lowers the happiness probability in most rows.

Cost to the text. The table gives the perplexity of held-out neutral text under the steered model and the distinct-1 of the generated text at +0.3, as the mean over the three steering readers of each emotion (with the range over the readers), against the unsteered text and the mean over the three random arms (with the range over the arms).

`qwen3-1.7b-base`:

| Arm                   | Perplexity [range over arms or readers] | Ratio to none | Distinct-1 [range]   |
| --------------------- | --------------------------------------- | ------------- | -------------------- |
| none                  | 18.6                                    | 1.00          | 0.298                |
| random (3 arms, +0.3) | 24.4 [24.0, 24.7]                       | 1.31          | 0.295 [0.289, 0.306] |
| happiness (+0.3)      | 56.4 [54.2, 58.7]                       | 3.03          | 0.216 [0.209, 0.229] |
| sadness (+0.3)        | 51.8 [49.5, 54.6]                       | 2.78          | 0.253 [0.251, 0.255] |
| anger (+0.3)          | 61.9 [59.5, 65.5]                       | 3.32          | 0.209 [0.203, 0.215] |
| fear (+0.3)           | 59.6 [58.2, 61.1]                       | 3.20          | 0.223 [0.207, 0.245] |
| disgust (+0.3)        | 51.7 [49.7, 55.6]                       | 2.77          | 0.224 [0.202, 0.237] |
| surprise (+0.3)       | 56.7 [51.1, 59.8]                       | 3.05          | 0.198 [0.187, 0.207] |

`qwen3-1.7b`:

| Arm                   | Perplexity [range over arms or readers] | Ratio to none | Distinct-1 [range]   |
| --------------------- | --------------------------------------- | ------------- | -------------------- |
| none                  | 33.7                                    | 1.00          | 0.240                |
| random (3 arms, +0.3) | 45.8 [43.2, 47.4]                       | 1.36          | 0.226 [0.215, 0.234] |
| happiness (+0.3)      | 88.0 [84.6, 93.4]                       | 2.61          | 0.193 [0.185, 0.201] |
| sadness (+0.3)        | 159.4 [145.6, 181.4]                    | 4.74          | 0.155 [0.150, 0.159] |
| anger (+0.3)          | 105.6 [97.2, 121.5]                     | 3.14          | 0.186 [0.182, 0.192] |
| fear (+0.3)           | 110.0 [105.9, 112.9]                    | 3.27          | 0.154 [0.147, 0.160] |
| disgust (+0.3)        | 118.4 [99.8, 142.9]                     | 3.52          | 0.175 [0.164, 0.181] |
| surprise (+0.3)       | 104.6 [95.7, 116.9]                     | 3.11          | 0.132 [0.126, 0.138] |

- **+0.3 is not fluent text.** Averaged over the six emotions the perplexity under the emotion arms is 1.11, 1.57 and 3.03 times the unsteered one at +0.1, +0.2 and +0.3 on the base checkpoint and 1.15, 1.72 and 3.40 times on the post-trained one, against 1.03, 1.12 and 1.31 and 1.03, 1.13 and 1.36 for the random arms. By the rule of [Sentiment steering on Qwen3](#sentiment-steering-on-qwen3-c1) (mean perplexity at most 1.5 times the unsteered one and mean distinct-1 at least 0.15) only +0.1 passes on either checkpoint, and every emotion is at 2.61 to 4.74 times at +0.3 (distinct-1 from 0.132 for surprise to 0.253 for sadness). The changes in the first tables are therefore changes of text that fails the rule at +0.3, and a rise in an emotion score at +0.3 is not read as steering that works. The own-emotion probability at +0.1 is in the curves file of each run and is not tabulated here.
- **Not stored.** The probabilities are those of a classifier on short completions; the run does not measure whether a human would read them as the emotion.

#### The 4B checkpoint (C2)

C2 ran on `qwen3-4b` only (run `c2/qwen3-4b/1df7c1f9ff70`, made at `89ea7cb` on a rented GPU pod); `c2-qwen3-4b-base` was
cut when the chain was stopped, so there is no 4B pair. The config is that of the 1.7B runs above (100 prompts, 2 samples
per prompt, 3 steering seeds, relative strength +0.3, three random arms). The reading layer is L25 with steering over
layers 24, 25 and 26 (36 blocks), so it is not the depth of the 1.7B runs (L18 of 28).

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.090         | 0.249 [0.202, 0.294]  | +0.159          | 0.241       | 0.540       | +0.299        | surprise +0.077                | sadness -0.097               | -0.020 to -0.002        |
| sadness   | 0.073         | 0.206 [0.123, 0.268]  | +0.133          | 0.129       | 0.259       | +0.129        | fear +0.039                    | disgust -0.061               | -0.023 to +0.020        |
| anger     | 0.045         | 0.129 [0.098, 0.153]  | +0.084          | 0.066       | 0.143       | +0.076        | happiness -0.050               | happiness -0.122             | -0.014 to -0.005        |
| fear      | 0.072         | 0.249 [0.222, 0.302]  | +0.178          | 0.095       | 0.189       | +0.095        | happiness -0.054               | happiness -0.118             | -0.014 to +0.010        |
| disgust   | 0.051         | 0.074 [0.067, 0.080]  | +0.023          | 0.102       | 0.114       | +0.012        | happiness -0.055               | happiness -0.122             | -0.021 to +0.000        |
| surprise  | 0.099         | 0.348 [0.293, 0.403]  | +0.248          | 0.068       | 0.141       | +0.072        | sadness -0.064                 | happiness +0.264             | +0.003 to +0.018        |

- **All six emotions rise toward their own score on the primary judge**, by +0.023 (disgust) to +0.248 (surprise), with
  the range over seeds above the unsteered value in every row. Surprise (+0.248), fear (+0.178), happiness (+0.159) and
  sadness (+0.133) rise most, then anger (+0.084) and disgust (+0.023). The hit rate is 1.000 and the specificity 0.159;
  on the check judge they are 0.667 (it misses disgust, where anger is the largest entry, and surprise, where happiness
  is) and 0.151. The correlation between the judges over the change matrix is 0.640, below the 1.7B values (0.833 and
  0.891), and the diagonals differ: the check judge gives surprise +0.072 where the primary judge gives +0.248, and
  happiness +0.299 where the primary gives +0.159.
- **Random arms.** The own-emotion change under a random arm is between -0.023 and +0.020, and the largest absolute
  change of any emotion under any random arm and coefficient is 0.052. Five own-emotion changes (+0.084 to +0.248) are
  above it; disgust (+0.023) is not, so it is not separated from an arbitrary edit of the same size.
- **Off-diagonal.** On the primary judge the largest off-diagonal change is a fall of happiness in the anger, fear and
  disgust rows (-0.050 to -0.055), a fall of sadness in the surprise row (-0.064), a rise of fear in the sadness row
  (+0.039) and a rise of surprise in the happiness row (+0.077). On the check judge happiness falls in the same three
  rows (-0.118 to -0.122) and rises by +0.264 in the surprise row, more than the diagonal (+0.072).
- **Sadness.** Sadness is steered on this checkpoint (+0.133), where it is not on the 1.7B post-trained one (+0.018).
  The layers differ and each emotion has one reader per checkpoint, so this is a contrast between two runs.

Cost to the text, with the same layout as above (mean over the three steering readers of each emotion with the range over
the readers, against the unsteered text and the mean over the three random arms):

| Arm                   | Perplexity [range over arms or readers] | Ratio to none | Distinct-1 [range]   |
| --------------------- | --------------------------------------- | ------------- | -------------------- |
| none                  | 31.7                                    | 1.00          | 0.249                |
| random (3 arms, +0.3) | 41.5 [40.7, 42.9]                       | 1.31          | 0.220 [0.216, 0.223] |
| happiness (+0.3)      | 74.7 [71.6, 76.6]                       | 2.36          | 0.156 [0.149, 0.166] |
| sadness (+0.3)        | 99.3 [88.3, 110.5]                      | 3.13          | 0.173 [0.162, 0.184] |
| anger (+0.3)          | 106.2 [104.9, 108.0]                    | 3.35          | 0.161 [0.156, 0.170] |
| fear (+0.3)           | 94.8 [84.6, 102.8]                      | 2.99          | 0.137 [0.121, 0.145] |
| disgust (+0.3)        | 151.4 [146.4, 160.5]                    | 4.78          | 0.139 [0.130, 0.153] |
| surprise (+0.3)       | 99.5 [88.3, 112.5]                      | 3.14          | 0.115 [0.110, 0.121] |

- **+0.3 is not fluent text.** Averaged over the six emotions the perplexity under the emotion arms is 1.14, 1.69 and
  3.29 times the unsteered one at +0.1, +0.2 and +0.3, against 1.02, 1.10 and 1.31 for the random arms (the 1.7B
  post-trained values are 1.15, 1.72 and 3.40). By the rule of [Sentiment steering on
  Qwen3](#sentiment-steering-on-qwen3-c1) (mean perplexity at most 1.5 times the unsteered one and mean distinct-1 at
  least 0.15) only +0.1 passes (mean ratio 1.14, at most 1.44 for one emotion, distinct-1 0.226 or above), and every
  emotion is at 2.36 to 4.78 times at +0.3, with a distinct-1 below 0.15 for fear, disgust and surprise. The changes in
  the first table are therefore changes of text that fails the rule at +0.3, and a rise in an emotion score at +0.3 is
  not read as steering that works.

### LoRRA adapter (LR1) and the TruthfulQA control (C4)

LR1 trains one LoRRA adapter per checkpoint (runs `lr1/qwen3-1.7b-base/95a42d94146c` and
`lr1/qwen3-1.7b/bf86ed5ccde1`: layers 9, 11, 13, 15 and 17, rank 8, learning rate 3e-4, 70 steps, 974,848 trainable
parameters, 2,000 training examples and 64 held-out, with the control prompts "Give a truthful answer." and "Give an
untruthful answer."). The training loss goes from 190.4 to 149.9 on the base checkpoint and from 263.1 to 196.6 on the
post-trained one. The held-out loss of each layer, before and after, and its reduction:

| Layer | qwen3-1.7b-base: before to after | Reduction | qwen3-1.7b: before to after | Reduction |
| ----- | -------------------------------- | --------- | --------------------------- | --------- |
| L9    | 8.95 to 7.25                     | 0.190     | 8.65 to 7.12                | 0.178     |
| L11   | 16.98 to 12.89                   | 0.241     | 19.55 to 14.50              | 0.258     |
| L13   | 28.80 to 22.25                   | 0.228     | 35.71 to 27.36              | 0.234     |
| L15   | 50.18 to 38.56                   | 0.232     | 63.20 to 48.19              | 0.237     |
| L17   | 100.77 to 77.55                  | 0.230     | 134.47 to 102.46            | 0.238     |

The loss grows with the layer, so the layers are compared by the reduction (0.178 to 0.258). LR1 has no honesty
evaluation: it measures the loss of the adapter's own objective on held-out examples and trains the adapter that C4
uses.

C4 asks whether steering or the adapter changes TruthfulQA MC1 (runs `c4/qwen3-1.7b-base/7dd49d64a540` and `c4/qwen3-1.7b/5ebc35ea3be6`, made at `f4d414d` on a rented GPU pod, which read the LR1 adapter of their checkpoint). These runs replace `c4/qwen3-1.7b-base/9cdad6a8b7ce` and `c4/qwen3-1.7b/2840839e453e` (made at `a6e2333` on the laptop GPU) as the reported runs; the earlier runs stay tracked. The configs differ in `n_random` (absent earlier, 3 here) and in the form of the `lorra_run` path (the same adapter, `95a42d94146c`), so these runs add three random arms (unit directions norm-matched to the steering arm, with a coefficient chosen on the train and validation questions like the readers) and the perplexity of neutral held-out text under each steered arm. MC1 is scored on the log-probabilities of the answer choices under the edited model, not on generated text. The 173 test questions are those of R6, the coefficients are chosen on the 644 train and validation questions, and the chance line is 0.228. C4 scores the edited model itself and trains no reader on the test answers, so the R6 text baselines are not its comparison; its control is the unedited row, the two prompting rows and the random arms. The unedited row of the base checkpoint (0.197) is the standard row of R6 (0.197); on the post-trained checkpoint it is 0.289 against 0.295 for R6, one question of 173. The perplexity column is that of held-out neutral text under the edited model; the prompting and `contrast` arms have no steered model, and the readers with coefficient 0 are the unedited model.

`qwen3-1.7b-base`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.197 [0.144, 0.262]       | 0                       | 6.84               |
| prompting +1         | 0.237 [0.180, 0.306]       | +0.040 [+0.006, +0.075] | -                  |
| prompting -1         | 0.225 [0.170, 0.293]       | +0.029 [-0.017, +0.070] | -                  |
| difference (0.2)     | 0.214 [0.159, 0.281]       | +0.017 [-0.040, +0.075] | 17.98              |
| contrast (0.5)       | 0.272 [0.211, 0.342]       | +0.075 [+0.023, +0.127] | -                  |
| LoRRA                | 0.277 [0.216, 0.348]       | +0.081 [+0.035, +0.127] | 6.94               |
| reading (10 readers) | mean 0.209, 0.185 to 0.225 | -0.012 to +0.029        | 8.04 to 64.41      |
| random (3 arms)      | mean 0.224, 0.208 to 0.237 | +0.012 to +0.040        | 8.78 to 9.32       |

`qwen3-1.7b`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.289 [0.227, 0.361]       | 0                       | 8.61               |
| prompting +1         | 0.324 [0.258, 0.397]       | +0.035 [+0.012, +0.064] | -                  |
| prompting -1         | 0.254 [0.195, 0.324]       | -0.035 [-0.081, +0.012] | -                  |
| difference (-0.05)   | 0.295 [0.232, 0.367]       | +0.006 [-0.023, +0.035] | 8.43               |
| contrast (0.5)       | 0.347 [0.280, 0.420]       | +0.058 [+0.000, +0.116] | -                  |
| LoRRA                | 0.341 [0.275, 0.414]       | +0.052 [-0.006, +0.110] | 9.35               |
| reading (10 readers) | mean 0.309, 0.289 to 0.324 | +0.000 to +0.035        | 8.55 to 8.86       |
| random (3 arms)      | mean 0.281, 0.272 to 0.289 | -0.017 to +0.000        | 8.96 to 9.18       |

- **Base checkpoint.** The unedited row (0.197) is below the chance line. The truthful prompt gives +0.040 [+0.006, +0.075] and the untruthful one +0.029 [-0.017, +0.070], so a changed prompt moves MC1 by a similar amount whatever it says. The adapter gives +0.081 [+0.035, +0.127] (14 more questions of 173) and `contrast` gives +0.075 [+0.023, +0.127]; `difference` (+0.017) and the ten readers (-0.012 to +0.029) have intervals that contain zero. The adapter's 0.277 has an interval [0.216, 0.348] that contains the chance line. **The random arms move MC1 as much as the readers do**: +0.012, +0.040 and +0.029, the second with the interval [0.000, +0.087], the same change as the truthful prompt and larger than the best reader and `difference`. The adapter and `contrast` are above every random arm (0.272 and 0.277 against at most 0.237), but their intervals overlap those of the random arms.
- **Post-trained checkpoint.** The unedited row (0.289) is above the chance line. The truthful prompt gives +0.035 [+0.012, +0.064] and the untruthful prompt -0.035 [-0.081, +0.012], in the direction of their meaning, with the second interval reaching zero. The adapter gives +0.052 [-0.006, +0.110] (9 more questions of 173), `contrast` +0.058 [0.000, +0.116], `difference` +0.006 [-0.023, +0.035], and the ten readers 0.000 to +0.035, with every interval containing zero or touching it. The three random arms give -0.017 to 0.000.
- **What is and is not supported.** The truthful prompting row is the only arm whose interval excludes zero on both checkpoints (the lower bounds are +0.006 and +0.012), and on the base checkpoint `contrast` and the adapter also exclude zero; this is among the arms of the tables and with no correction for multiple comparisons. The adapter does not exclude zero on the post-trained checkpoint in this run (it did in the earlier run, +0.064 [+0.012, +0.121]). The intervals of the adapter, of `contrast` and of the prompting rows overlap on both checkpoints. The chosen `contrast` coefficient (0.5) is the largest of its grid on both checkpoints, and the `difference` coefficient of the base checkpoint (+0.2) is the largest of its grid, so the selection was at an edge. The ten readers chose +0.1 (six readers), +0.2 (one) and -0.2 (three) on the base checkpoint, and +0.05 for eight and 0 for two on the post-trained one; the three random arms chose -0.2, +0.2 and -0.2 on the base checkpoint and +0.1 for all three on the post-trained one. The gains of the adapter on the base checkpoint are therefore not separated from what a random direction of the same norm gives at a selected coefficient, and the gain on the post-trained checkpoint is not separated from zero.
- **Cost to the text.** The adapter leaves the perplexity of neutral text near the unedited one: 6.94 against 6.84 on the base checkpoint (1.01 times) and 9.35 against 8.61 on the post-trained one (1.09 times). On the base checkpoint the settings chosen by selection are not all benign: `difference` at +0.2 has a perplexity of 17.98 (2.63 times), the three readers with coefficient -0.2 have 57.53 to 64.41 (8.41 to 9.42 times), and the other seven readers 8.04 to 14.23 (1.18 to 2.08 times); the random arms are at 8.78 to 9.32 (1.28 to 1.36 times). On the post-trained checkpoint the eight readers with a nonzero coefficient are at 8.55 to 8.86 (0.99 to 1.03 times), `difference` at -0.05 at 8.43, and the random arms at 8.96 to 9.18 (1.04 to 1.07 times). The scoring is not generation, so the distinct-1 and the completion fluency of the text are not measured.

#### The 4B pair (LR1)

LR1 on the 4B pair (runs `lr1/qwen3-4b-base/22e472aa6c00` and `lr1/qwen3-4b/f0d81c80e9bc`, made at `89ea7cb` on a rented
GPU pod: layers 11, 13, 15, 17, 19 and 21, rank 8, learning rate 3e-4, 70 steps, batch size 16, 1,720,320 trainable
parameters, 2,000 training examples and 64 held-out, with the same control prompts). The training loss goes from 104.4
to 76.1 on the base checkpoint and from 148.7 to 108.3 on the post-trained one. The held-out loss of each layer, before
and after, and its reduction:

| Layer | qwen3-4b-base: before to after | Reduction | qwen3-4b: before to after | Reduction |
| ----- | ------------------------------ | --------- | ------------------------- | --------- |
| L11   | 5.03 to 4.37                   | 0.132     | 5.49 to 4.46              | 0.187     |
| L13   | 7.08 to 5.74                   | 0.189     | 8.52 to 6.31              | 0.259     |
| L15   | 10.13 to 8.46                  | 0.165     | 12.20 to 9.55             | 0.217     |
| L17   | 14.94 to 12.00                 | 0.196     | 17.69 to 14.19            | 0.198     |
| L19   | 23.18 to 18.49                 | 0.202     | 29.01 to 23.29            | 0.197     |
| L21   | 39.00 to 30.85                 | 0.209     | 52.51 to 41.66            | 0.207     |

The loss grows with the layer, so the layers are compared by the reduction: 0.132 to 0.209 on the base checkpoint (the
smallest at L11, the largest at L21) and 0.187 to 0.259 on the post-trained one (the largest at L13), against 0.178 to
0.258 at 1.7B. LR1 has no honesty evaluation. C4 did not run on the 4B pair, so these adapters are not scored on
TruthfulQA here.

### Lie gallery (C5)

C5 records the share of correct answers without steering and under an honesty-steered and a lie-steered arm on the 1,000 TriviaQA questions of the pool, one sample per question at temperature 0.8, relative strength 0.1 (+0.1 for the honesty arm, -0.1 for the lie arm), steered over layers 8, 11, 13, 16, 19, 21, 24 and 26 (runs `c5/qwen3-1.7b-base/3a14e656df3c` and `c5/qwen3-1.7b/52057b2ddc30`, made at `f4d414d` on a rented GPU pod). These runs replace `c5/qwen3-1.7b-base/fafc346bc266` and `c5/qwen3-1.7b/672f70c5406a` (made at `a6e2333` on the laptop GPU) as the reported runs; the earlier runs stay tracked. The configs differ in `n_random` (absent earlier, 3 here) and in the C2 run they read, which is the C2 run reported above for the same checkpoint. The runs add three random arms (unit directions norm-matched to the steering arm, steered at +0.1 and -0.1) and the fluency measure. The base checkpoint is prompted with 15 question and answer shots; the stored prompt of the post-trained checkpoint is the question alone in the chat template, so the two columns are not a comparison of the models. The random rows are the mean of the three random directions. The perplexity is that of held-out neutral text under the steered model, and the distinct-1 that of the generated answers (about 24 tokens).

| Quantity                                     | qwen3-1.7b-base      | qwen3-1.7b           |
| -------------------------------------------- | -------------------- | -------------------- |
| Correct, no steering                         | 0.344                | 0.237                |
| Correct, honesty-steered                     | 0.305                | 0.216                |
| Correct, lie-steered                         | 0.271                | 0.186                |
| Correct, random +0.1 (mean of 3)             | 0.319                | 0.233                |
| Correct, random -0.1 (mean of 3)             | 0.330                | 0.230                |
| Answered correctly without steering          | 344                  | 237                  |
| Of these, answered wrongly under the lie arm | 90                   | 86                   |
| Flip rate [95% interval]                     | 0.262 [0.218, 0.311] | 0.363 [0.304, 0.426] |
| Perplexity: none                             | 6.8                  | 8.6                  |
| Perplexity: honesty-steered                  | 8.6                  | 10.0                 |
| Perplexity: lie-steered                      | 11.2                 | 13.0                 |
| Perplexity: random (range of 6)              | 7.1 to 7.4           | 9.0 to 9.4           |
| Distinct-1: none                             | 0.124                | 0.231                |
| Distinct-1: honesty-steered                  | 0.114                | 0.227                |
| Distinct-1: lie-steered                      | 0.158                | 0.213                |
| Distinct-1: random (range of 6)              | 0.119 to 0.129       | 0.225 to 0.234       |

- **Both steered arms are below the unsteered one on both checkpoints**, the lie arm further (by 0.073 and 0.051, the honesty arm by 0.039 and 0.021). The lie arm is 0.034 and 0.030 below the honesty arm. The random arms are below the unsteered share too, but less: 0.319 and 0.330 on the base checkpoint (0.025 and 0.014 below) and 0.233 and 0.230 on the post-trained one (0.004 and 0.007 below). The honesty arm is 0.014 to 0.025 below the random arms on the base checkpoint and 0.014 to 0.017 on the post-trained one, and the lie arm 0.048 to 0.059 and 0.044 to 0.047 below. The shares carry no interval; for scale, a share of 1,000 questions near 0.3 has a binomial standard error of about 0.015, and the standard error of a difference of two such shares is about 0.02, so the honesty-arm gap to the random arms is roughly 0.7 to 1.2 standard errors and the lie-arm gap roughly 2 to 3. The earlier runs gave 0.341, 0.315 and 0.258 (base) and 0.252, 0.216 and 0.178 (post-trained) for the three arms, and flip rates of 0.293 and 0.385.
- **Flip rate.** 90 of the 344 questions answered correctly are answered wrongly under the lie arm on the base checkpoint (0.262 [0.218, 0.311]) and 86 of 237 on the post-trained one (0.363 [0.304, 0.426]); the intervals overlap slightly. The honesty arm and the random arms have no stored flip rate, so the flip rate is not separated from the effect of a generic perturbation of the model.
- **Cost to the text.** The perplexity of neutral text under the honesty and lie arms is 1.25 and 1.64 times the unsteered one on the base checkpoint and 1.16 and 1.51 times on the post-trained one, against 1.04 to 1.08 and 1.04 to 1.09 for the random arms, so the lie arm moves the text the most, and the random arms the least. The distinct-1 cutoff of the rule in [Sentiment steering on Qwen3](#sentiment-steering-on-qwen3-c1) (0.15) is not used here: the unsteered answers of the base checkpoint are already at 0.124, and the lie arm of that checkpoint has a higher distinct-1 (0.158) than the unsteered text.
- The run also stores a small gallery of completions (4 unfiltered and up to 4 curated per arm, the curated ones being the first flipped questions), which is a selection and not evidence for a rate. An emotion gallery from the C2 run of the same checkpoint is stored with it and is not read here.

## Position and stimulus-set size on qwen3-1.7b-base and qwen3-1.7b (P2, P3)

P2 ran on the Qwen3 1.7B pair, the base `qwen3-1.7b-base` and the post-trained `qwen3-1.7b`, and P3 ran on the base 1.7B checkpoint only; P2 later ran on `qwen3-0.6b-base`, `qwen3-0.6b`, `gpt2-medium`, `gpt2-large` and `gpt2-xl` (last subsections below). Both use the settings of the `gpt2` runs (logistic finder, 64 training pairs; 10 seeds for P2 and 20 draws per size for P3, the intervals of the [reading conventions](#reading-conventions)). Each is one run: `p2/qwen3-1.7b-base/119f0ac54451` and `p3/qwen3-1.7b-base/61baf30a8a59`, made at `c7d8051` with a clean tree on a cloud GPU (NVIDIA GeForce RTX 4090), and `p2/qwen3-1.7b/51a85fbec631`, made at `20650c5` with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU). The two P2 runs therefore differ in hardware and code commit as well as in checkpoint; the P2 code is the same at both commits (only `cloud.py` differs under `src`). The 0.6B checkpoints did not run P3, and P3 did not run on the post-trained checkpoint. Base and post-trained are two models: their P2 tables are separate and no metric is pooled over them, and the `gpt2` and Qwen3 numbers come from different models, not from a scaling series.

### Token position and a misleading sentence on qwen3-1.7b-base (P2)

The design is that of [P2 on gpt2](#token-position-and-a-misleading-sentence-p2): sentiment (214 report pairs), truth (59) and utility (256) at the positions `last`, `concept` and `mean`, with a `random` and an `against` decoy sentence, and a zero-shot row that scores the model's own verbalized level. The layer is the one chosen on the original prompts.

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L18   | 0.944 [0.939, 0.950] | 0.908 [0.901, 0.914] | 0.785 [0.774, 0.795] |
| sentiment | concept   | L27   | 0.935 [0.927, 0.943] | 0.915 [0.908, 0.922] | 0.865 [0.857, 0.875] |
| sentiment | mean      | L15   | 0.910 [0.900, 0.921] | 0.919 [0.908, 0.929] | 0.914 [0.905, 0.925] |
| sentiment | zero-shot | -     | 0.963 [0.935, 0.986] | 0.762 [0.706, 0.818] | 0.248 [0.192, 0.308] |
| truth     | last      | L18   | 0.968 [0.961, 0.975] | 0.756 [0.742, 0.769] | 0.058 [0.052, 0.064] |
| truth     | concept   | L18   | 0.858 [0.841, 0.876] | 0.888 [0.869, 0.905] | 0.624 [0.600, 0.647] |
| truth     | mean      | L16   | 0.851 [0.829, 0.875] | 0.822 [0.797, 0.853] | 0.817 [0.790, 0.847] |
| truth     | zero-shot | -     | 0.932 [0.864, 0.983] | 0.814 [0.712, 0.915] | 0.424 [0.305, 0.542] |
| utility   | last      | L18   | 0.761 [0.750, 0.773] | 0.709 [0.695, 0.723] | 0.302 [0.249, 0.352] |
| utility   | concept   | L26   | 0.744 [0.731, 0.760] | 0.725 [0.712, 0.737] | 0.330 [0.299, 0.367] |
| utility   | mean      | L16   | 0.705 [0.683, 0.727] | 0.716 [0.693, 0.736] | 0.621 [0.595, 0.646] |
| utility   | zero-shot | -     | 0.773 [0.719, 0.824] | 0.656 [0.598, 0.711] | 0.012 [0.000, 0.027] |

- **Position.** On the original prompts the positions read within 0.035 of each other for sentiment (0.944, 0.935, 0.910) and within 0.055 for utility (0.761, 0.744, 0.705). Truth differs by 0.117: `last` reads 0.968, `concept` 0.858 and `mean` 0.851. On `gpt2` sentiment read higher at `concept` and `mean` than at `last`; here `last` is the highest position on all three tasks.
- **Random decoy.** The reader moves by at most 0.051 at `last` for sentiment (0.944 to 0.908) and utility (0.761 to 0.709), and truth drops by 0.212 (0.968 to 0.756). At `concept` and `mean` no task moves by more than 0.031 (truth at `concept` goes up, 0.858 to 0.888, with overlapping intervals).
- **Against decoy.** At `last` the reader falls on every task: sentiment to 0.785 (a drop of 0.160), utility to 0.302 and truth to 0.058 [0.052, 0.064], far below chance, where the reader ranks the pairs the wrong way round. At `concept` sentiment drops 0.070 (0.865), truth 0.234 (0.624) and utility 0.414 (0.330). At `mean` sentiment (0.914) and truth (0.817) are within 0.034 of the original with overlapping intervals; utility is 0.621 [0.595, 0.646], a drop of 0.084 with an interval that does not overlap the original [0.683, 0.727]. On `gpt2` only utility at `last` collapsed under this decoy; on this checkpoint the drop is large at `last` for all three tasks and at `concept` for truth and utility. The run does not say why.
- **Zero-shot.** The model's own answer drops from 0.963, 0.932 and 0.773 to 0.762, 0.814 and 0.656 under the random decoy and to 0.248, 0.424 and 0.012 under the `against` decoy (for truth the interval, [0.305, 0.542], reaches 0.5). The reader at `mean` stays at 0.914 and 0.817 under the `against` decoy on sentiment and truth, against 0.248 and 0.424 for the model's answer, so on those two tasks the decoy that turns the model's verbalized answer around does not turn that reader; on utility the reader falls to 0.621 and the model's answer to 0.012.

### Token position and a misleading sentence on qwen3-1.7b (P2)

The design, the report splits (214, 59 and 256 pairs) and the layer rule are those of the base checkpoint above.

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L18   | 0.880 [0.866, 0.894] | 0.827 [0.810, 0.844] | 0.722 [0.700, 0.743] |
| sentiment | concept   | L14   | 0.900 [0.892, 0.908] | 0.874 [0.858, 0.890] | 0.865 [0.850, 0.880] |
| sentiment | mean      | L16   | 0.909 [0.900, 0.918] | 0.916 [0.905, 0.926] | 0.910 [0.900, 0.921] |
| sentiment | zero-shot | -     | 0.921 [0.883, 0.958] | 0.631 [0.570, 0.696] | 0.000 [0.000, 0.000] |
| truth     | last      | L18   | 0.934 [0.924, 0.946] | 0.778 [0.763, 0.792] | 0.551 [0.525, 0.576] |
| truth     | concept   | L27   | 0.756 [0.732, 0.778] | 0.617 [0.595, 0.639] | 0.392 [0.334, 0.441] |
| truth     | mean      | L16   | 0.847 [0.831, 0.866] | 0.807 [0.786, 0.829] | 0.807 [0.788, 0.827] |
| truth     | zero-shot | -     | 0.915 [0.847, 0.983] | 0.695 [0.576, 0.814] | 0.254 [0.153, 0.373] |
| utility   | last      | L28   | 0.764 [0.755, 0.773] | 0.714 [0.696, 0.732] | 0.358 [0.294, 0.432] |
| utility   | concept   | L18   | 0.763 [0.755, 0.770] | 0.757 [0.746, 0.768] | 0.604 [0.521, 0.679] |
| utility   | mean      | L16   | 0.695 [0.673, 0.717] | 0.715 [0.690, 0.738] | 0.677 [0.650, 0.701] |
| utility   | zero-shot | -     | 0.695 [0.637, 0.746] | 0.605 [0.547, 0.664] | 0.000 [0.000, 0.000] |

- **Position.** On the original prompts the positions read within 0.029 of each other for sentiment (0.880, 0.900, 0.909) and within 0.069 for utility (0.764, 0.763, 0.695). Truth differs by 0.178: `last` reads 0.934, `concept` 0.756 and `mean` 0.847. `last` is the highest position for truth and, level with `concept`, for utility; for sentiment it is the lowest. Utility at `last` is read at L28, the last block, which is the edge of the layers read.
- **Random decoy.** At `last` the reader moves by 0.054 for sentiment (0.880 to 0.827), 0.156 for truth (0.934 to 0.778) and 0.050 for utility (0.764 to 0.714). At `concept` sentiment moves by 0.026 (0.900 to 0.874) and utility by 0.005, but truth drops by 0.139 (0.756 to 0.617). At `mean` truth drops by 0.041 (0.847 to 0.807, intervals that do not overlap), and sentiment and utility go up (0.909 to 0.916 and 0.695 to 0.715) with overlapping intervals.
- **Against decoy.** At `last` the reader falls on every task: sentiment to 0.722 (a drop of 0.158), truth to 0.551 [0.525, 0.576] (a drop of 0.383, with an interval above chance) and utility to 0.358 [0.294, 0.432], below chance, where the reader ranks the pairs the wrong way round. At `concept` sentiment drops 0.035 (0.865), truth falls to 0.392 [0.334, 0.441], below chance (a drop of 0.364), and utility to 0.604 [0.521, 0.679] (a drop of 0.159). At `mean` sentiment (0.910) and utility (0.677, an interval that overlaps the original) are within 0.019 of the original; truth is 0.807 [0.788, 0.827], a drop of 0.041 with an interval that does not overlap the original [0.831, 0.866]. The run does not say why the position and the task decide where the reader falls below chance.
- **Zero-shot.** The model's own answer drops from 0.921, 0.915 and 0.695 to 0.631, 0.695 and 0.605 under the random decoy and to 0.000, 0.254 and 0.000 under the `against` decoy (the intervals of sentiment and utility are [0.000, 0.000]; for truth the interval is [0.153, 0.373]). The reader at `mean` stays at 0.910, 0.807 and 0.677 under the `against` decoy, so the decoy that turns the model's verbalized answer around on all three tasks does not turn that reader.

### Base against post-trained on P2

The two checkpoints are two models and each has one run, so this compares cells and does not pool them.

- **Original prompts.** At `mean` the two read within 0.011 on every task (sentiment 0.909 against 0.910, truth 0.847 against 0.851, utility 0.695 against 0.705), and utility at `last` and `concept` is within 0.019 (0.764 against 0.761, 0.763 against 0.744). The post-trained point value is lower in seven of the nine cells. In four the intervals do not overlap: sentiment at `last` (0.880 against 0.944) and `concept` (0.900 against 0.935), and truth at `last` (0.934 against 0.968) and `concept` (0.756 against 0.858). In the other three, sentiment, truth and utility at `mean`, the intervals overlap (the differences are 0.001, 0.003 and 0.010); in the remaining two cells, utility at `last` and `concept`, the post-trained reader is higher.
- **Against decoy.** The cells where the reader falls below chance differ. On the base checkpoint truth at `last` is 0.058 [0.052, 0.064], while on the post-trained checkpoint it is 0.551 [0.525, 0.576]; truth at `concept` runs the other way, 0.624 [0.600, 0.647] on the base and 0.392 [0.334, 0.441] on the post-trained checkpoint. Utility at `last` is below chance on both (0.302 [0.249, 0.352] and 0.358 [0.294, 0.432], overlapping intervals), while utility at `concept` is 0.330 on the base and 0.604 on the post-trained checkpoint. Sentiment at `last` is 0.785 and 0.722. The run does not say why.
- **Zero-shot.** The post-trained point values are lower than the base ones on all three tasks, with overlapping intervals under the original prompts (0.921, 0.915 and 0.695 against 0.963, 0.932 and 0.773). Under the `against` decoy the model's own answer is below chance on both checkpoints on all three tasks (0.000, 0.254 and 0.000 against 0.248, 0.424 and 0.012; the base interval for truth, [0.305, 0.542], reaches 0.5).
- **Layers.** The layer chosen on the original prompts differs between the checkpoints in 5 of the 9 reader cells: sentiment at `concept` (L14 against L27) and `mean` (L16 against L15), truth at `concept` (L27 against L18), and utility at `last` (L28 against L18) and `concept` (L18 against L26). A difference in those cells mixes the checkpoint with the layer selected on the select split.
- **Limits.** The intervals are over the 10 training draws and do not include the sampling of the evaluation pairs, which matters most for truth (59 report pairs). The two runs also differ in hardware and in code commit, though not in the P2 code.

### Token position and a misleading sentence on gpt2-medium and the 0.6B checkpoints (P2)

P2 also ran on `gpt2-medium` and on the two 0.6B Qwen3 checkpoints (runs `p2/gpt2-medium/733bd2f76315`, `p2/qwen3-0.6b-base/59a53843dccd` and `p2/qwen3-0.6b/442ebe80ca68`), all made at `fb3b5dd` with a clean tree on one laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU) with batch size 8. The design, the report splits (214, 59 and 256 pairs) and the layer rule are those of the runs above. The P2 experiment code is unchanged since the earlier P2 runs; the differences under `src` are tooling (`chain.py`, `cloud.py`, `cli.py`, `export.py`) and an opt-in refit flag in the shared fit code that P2 does not set. Unlike the 1.7B pair, the 0.6B pair differs in checkpoint only (same hardware and commit). The three tables are separate and no metric is pooled over them.

#### gpt2-medium

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L20   | 0.742 [0.724, 0.758] | 0.724 [0.711, 0.737] | 0.708 [0.696, 0.720] |
| sentiment | concept   | L13   | 0.813 [0.799, 0.825] | 0.826 [0.810, 0.839] | 0.814 [0.791, 0.829] |
| sentiment | mean      | L11   | 0.843 [0.830, 0.857] | 0.850 [0.837, 0.863] | 0.850 [0.837, 0.864] |
| sentiment | zero-shot | -     | 0.603 [0.542, 0.668] | 0.579 [0.514, 0.645] | 0.093 [0.056, 0.136] |
| truth     | last      | L19   | 0.656 [0.625, 0.685] | 0.617 [0.600, 0.639] | 0.593 [0.571, 0.615] |
| truth     | concept   | L20   | 0.729 [0.710, 0.749] | 0.690 [0.680, 0.698] | 0.646 [0.632, 0.661] |
| truth     | mean      | L20   | 0.668 [0.644, 0.692] | 0.681 [0.653, 0.714] | 0.681 [0.653, 0.714] |
| truth     | zero-shot | -     | 0.559 [0.441, 0.695] | 0.559 [0.441, 0.695] | 0.203 [0.102, 0.305] |
| utility   | last      | L15   | 0.685 [0.665, 0.701] | 0.658 [0.642, 0.675] | 0.330 [0.223, 0.449] |
| utility   | concept   | L14   | 0.696 [0.678, 0.712] | 0.706 [0.689, 0.721] | 0.650 [0.606, 0.688] |
| utility   | mean      | L15   | 0.660 [0.643, 0.674] | 0.652 [0.631, 0.668] | 0.674 [0.650, 0.695] |
| utility   | zero-shot | -     | 0.520 [0.457, 0.582] | 0.543 [0.480, 0.609] | 0.020 [0.004, 0.039] |

- **Position.** On the original prompts the positions read 0.742, 0.813 and 0.843 for sentiment (a spread of 0.100), 0.656, 0.729 and 0.668 for truth (0.073) and 0.685, 0.696 and 0.660 for utility (0.036). As on `gpt2`, sentiment reads higher at `concept` and `mean` than at `last`; truth reads highest at `concept` here, where `last` was highest on `gpt2` (0.698).
- **Random decoy.** The reader moves by at most 0.039 in any cell (truth at `last`, 0.656 to 0.617, and at `concept`, 0.729 to 0.690); the intervals of the original and random columns overlap everywhere except truth at `concept`.
- **Against decoy.** At `last` sentiment (0.708) and truth (0.593 [0.571, 0.615]) lose 0.034 and 0.063 and stay above chance, and utility falls to 0.330 [0.223, 0.449], a drop of 0.355 to below chance (0.133 on `gpt2`). At `concept` sentiment is unchanged (0.814), truth drops 0.083 (0.646 [0.632, 0.661]) and utility 0.046 (0.650, with overlapping intervals). At `mean` no task moves by more than 0.014 and the intervals overlap in all three. As on `gpt2`, the collapse at `last` belongs to utility.
- **Zero-shot.** The model's own answer reads 0.603, 0.559 and 0.520 on the original prompts, 0.579, 0.559 and 0.543 under the random decoy and 0.093, 0.203 and 0.020 under the `against` decoy, below chance (the truth intervals, [0.441, 0.695] under the original and the random decoy, include 0.5). The reader at `mean` stays at 0.850, 0.681 and 0.674 under the same decoy.

#### qwen3-0.6b-base

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L18   | 0.952 [0.946, 0.958] | 0.790 [0.776, 0.802] | 0.437 [0.415, 0.457] |
| sentiment | concept   | L28   | 0.927 [0.912, 0.940] | 0.828 [0.818, 0.836] | 0.551 [0.525, 0.577] |
| sentiment | mean      | L16   | 0.876 [0.861, 0.890] | 0.881 [0.869, 0.892] | 0.880 [0.867, 0.892] |
| sentiment | zero-shot | -     | 0.935 [0.897, 0.967] | 0.650 [0.584, 0.710] | 0.023 [0.005, 0.047] |
| truth     | last      | L24   | 0.764 [0.747, 0.781] | 0.712 [0.686, 0.739] | 0.208 [0.163, 0.258] |
| truth     | concept   | L20   | 0.673 [0.656, 0.685] | 0.722 [0.705, 0.744] | 0.629 [0.610, 0.644] |
| truth     | mean      | L15   | 0.751 [0.729, 0.775] | 0.756 [0.741, 0.771] | 0.758 [0.744, 0.773] |
| truth     | zero-shot | -     | 0.864 [0.780, 0.949] | 0.712 [0.593, 0.814] | 0.068 [0.017, 0.136] |
| utility   | last      | L18   | 0.714 [0.705, 0.726] | 0.688 [0.676, 0.699] | 0.375 [0.332, 0.421] |
| utility   | concept   | L22   | 0.731 [0.714, 0.748] | 0.715 [0.701, 0.727] | 0.543 [0.499, 0.586] |
| utility   | mean      | L16   | 0.692 [0.670, 0.711] | 0.695 [0.670, 0.718] | 0.671 [0.651, 0.688] |
| utility   | zero-shot | -     | 0.707 [0.652, 0.762] | 0.633 [0.570, 0.691] | 0.043 [0.020, 0.070] |

- **Position.** On the original prompts sentiment reads 0.952, 0.927 and 0.876 (`last` highest, a spread of 0.076), truth 0.764, 0.673 and 0.751 (`last` highest, 0.092) and utility 0.714, 0.731 and 0.692 (`concept` highest, 0.039).
- **Random decoy.** At `last` the reader falls by 0.162 for sentiment (0.952 to 0.790), 0.053 for truth and 0.026 for utility. At `concept` sentiment falls by 0.099 (0.927 to 0.828) and truth goes up by 0.049 (0.673 to 0.722, with intervals that do not overlap). At `mean` no task moves by more than 0.005.
- **Against decoy.** At `last` the reader falls below chance on all three tasks, with intervals below 0.5: sentiment 0.437 [0.415, 0.457], truth 0.208 [0.163, 0.258] and utility 0.375 [0.332, 0.421]. At `concept` sentiment drops 0.375 (0.551 [0.525, 0.577]), truth 0.044 (0.629) and utility 0.188 (0.543 [0.499, 0.586], an interval that reaches 0.5). At `mean` the three tasks (0.880, 0.758, 0.671) are within 0.021 of the original with overlapping intervals.
- **Zero-shot.** The model's own answer drops from 0.935, 0.864 and 0.707 to 0.650, 0.712 and 0.633 under the random decoy and to 0.023, 0.068 and 0.043 under the `against` decoy. The reader at `mean` stays at 0.880, 0.758 and 0.671 under the `against` decoy, so the decoy that turns the model's verbalized answer around does not turn that reader.

#### qwen3-0.6b

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L18   | 0.860 [0.843, 0.875] | 0.837 [0.825, 0.850] | 0.754 [0.739, 0.768] |
| sentiment | concept   | L14   | 0.798 [0.775, 0.822] | 0.801 [0.777, 0.824] | 0.761 [0.726, 0.794] |
| sentiment | mean      | L14   | 0.875 [0.861, 0.891] | 0.885 [0.873, 0.895] | 0.887 [0.875, 0.897] |
| sentiment | zero-shot | -     | 0.864 [0.818, 0.907] | 0.523 [0.458, 0.593] | 0.000 [0.000, 0.000] |
| truth     | last      | L18   | 0.747 [0.725, 0.771] | 0.783 [0.766, 0.802] | 0.786 [0.758, 0.819] |
| truth     | concept   | L28   | 0.625 [0.586, 0.666] | 0.654 [0.624, 0.685] | 0.659 [0.636, 0.681] |
| truth     | mean      | L16   | 0.761 [0.734, 0.792] | 0.758 [0.732, 0.788] | 0.759 [0.736, 0.788] |
| truth     | zero-shot | -     | 0.661 [0.542, 0.780] | 0.576 [0.441, 0.695] | 0.000 [0.000, 0.000] |
| utility   | last      | L28   | 0.706 [0.691, 0.720] | 0.674 [0.657, 0.690] | 0.296 [0.247, 0.356] |
| utility   | concept   | L22   | 0.709 [0.691, 0.727] | 0.705 [0.690, 0.719] | 0.422 [0.371, 0.475] |
| utility   | mean      | L16   | 0.676 [0.655, 0.694] | 0.685 [0.664, 0.705] | 0.676 [0.654, 0.696] |
| utility   | zero-shot | -     | 0.641 [0.578, 0.699] | 0.633 [0.570, 0.691] | 0.000 [0.000, 0.000] |

- **Position.** On the original prompts sentiment reads 0.860, 0.798 and 0.875 (`concept` lowest, a spread of 0.077), truth 0.747, 0.625 and 0.761 (0.136) and utility 0.706, 0.709 and 0.676 (0.032). Utility at `last` and truth at `concept` are read at L28, the last block, which is the edge of the layers read.
- **Random decoy.** At `last` the reader moves by 0.023 for sentiment (down), 0.036 for truth (up) and 0.032 for utility (down, with intervals that do not overlap); at `concept` and `mean` no task moves by more than 0.029, and all those intervals overlap.
- **Against decoy.** Truth does not collapse: 0.786 [0.758, 0.819] at `last` and 0.659 [0.636, 0.681] at `concept`, both above the original with overlapping intervals. Utility falls below chance at `last` (0.296 [0.247, 0.356], a drop of 0.411) and at `concept` (0.422 [0.371, 0.475], a drop of 0.287). Sentiment drops 0.106 at `last` (0.754) and 0.037 at `concept` (0.761, overlapping). At `mean` the three tasks (0.887, 0.759, 0.676) are within 0.012 of the original.
- **Zero-shot.** The model's own answer drops from 0.864, 0.661 and 0.641 to 0.523, 0.576 and 0.633 under the random decoy and to 0.000 on all three tasks under the `against` decoy (the stored intervals are [0.000, 0.000]). The reader at `mean` stays at 0.887, 0.759 and 0.676.

#### Base against post-trained at 0.6B

The two checkpoints are two models and each has one run, so this compares cells and does not pool them.

- **Original prompts.** The post-trained point value is lower in 8 of the 9 cells; the exception is truth at `mean` (0.761 against 0.751). The intervals do not overlap in two cells, sentiment at `last` (0.860 against 0.952) and `concept` (0.798 against 0.927); in the other seven they overlap.
- **Against decoy.** The cells that fall below chance differ. The base reader is below chance in three cells (sentiment, truth and utility at `last`) and the post-trained reader in two (utility at `last` and `concept`). Truth at `last` is 0.208 [0.163, 0.258] on the base and 0.786 [0.758, 0.819] on the post-trained checkpoint; sentiment at `last` is 0.437 and 0.754. At 1.7B truth at `last` also differs in the same direction (0.058 on the base, 0.551 on the post-trained checkpoint), though there the post-trained value is below its original. Utility at `concept` is 0.543 on the base and 0.422 on the post-trained checkpoint.
- **Zero-shot.** The post-trained point values are lower on all three tasks under the original prompts (0.864, 0.661 and 0.641 against 0.935, 0.864 and 0.707, with overlapping intervals), and under the `against` decoy the model's own answer is below chance on both checkpoints, exactly 0.000 on all three tasks for the post-trained one (0.023, 0.068 and 0.043 on the base).
- **Layers.** The layer chosen on the original prompts differs in 6 of the 9 reader cells: sentiment at `concept` (L28 against L14) and `mean` (L16 against L14), truth at `last` (L24 against L18), `concept` (L20 against L28) and `mean` (L15 against L16), and utility at `last` (L18 against L28). A difference in those cells mixes the checkpoint with the layer selected on the select split.
- **Limits.** The intervals are over the 10 training draws and do not include the sampling of the evaluation pairs, which matters most for truth (59 report pairs).

### Token position and a misleading sentence on gpt2-large and gpt2-xl (P2)

P2 also ran on `gpt2-large` and `gpt2-xl` (runs `p2/gpt2-large/3ea9b00fe28f` and `p2/gpt2-xl/feafacc97cbb`), both made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500) with batch size 8. The design, the report splits (214, 59 and 256 pairs) and the layer rule are those of the runs above, and the P2 experiment code is unchanged since the earlier P2 runs. The two tables are separate and no metric is pooled over them; they differ in checkpoint only (same hardware and commit).

#### gpt2-large

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L32   | 0.807 [0.792, 0.824] | 0.782 [0.763, 0.801] | 0.749 [0.726, 0.769] |
| sentiment | concept   | L26   | 0.847 [0.833, 0.860] | 0.800 [0.785, 0.815] | 0.766 [0.748, 0.783] |
| sentiment | mean      | L19   | 0.893 [0.879, 0.907] | 0.895 [0.883, 0.909] | 0.898 [0.885, 0.910] |
| sentiment | zero-shot | -     | 0.589 [0.523, 0.654] | 0.500 [0.435, 0.565] | 0.000 [0.000, 0.000] |
| truth     | last      | L23   | 0.653 [0.629, 0.678] | 0.646 [0.622, 0.666] | 0.653 [0.632, 0.675] |
| truth     | concept   | L22   | 0.692 [0.669, 0.710] | 0.661 [0.639, 0.681] | 0.658 [0.637, 0.678] |
| truth     | mean      | L34   | 0.695 [0.671, 0.717] | 0.686 [0.663, 0.708] | 0.688 [0.664, 0.710] |
| truth     | zero-shot | -     | 0.559 [0.441, 0.678] | 0.525 [0.390, 0.661] | 0.034 [0.000, 0.085] |
| utility   | last      | L36   | 0.688 [0.668, 0.711] | 0.686 [0.670, 0.700] | 0.331 [0.255, 0.430] |
| utility   | concept   | L25   | 0.705 [0.688, 0.723] | 0.718 [0.705, 0.730] | 0.518 [0.491, 0.544] |
| utility   | mean      | L22   | 0.669 [0.645, 0.689] | 0.669 [0.650, 0.686] | 0.677 [0.655, 0.696] |
| utility   | zero-shot | -     | 0.551 [0.492, 0.609] | 0.555 [0.492, 0.617] | 0.000 [0.000, 0.000] |

- **Position.** On the original prompts the positions read 0.807, 0.847 and 0.893 for sentiment (a spread of 0.086), 0.653, 0.692 and 0.695 for truth (0.042) and 0.688, 0.705 and 0.669 for utility (0.036). Sentiment reads highest at `mean`, as on `gpt2-medium`.
- **Random decoy.** The reader moves by at most 0.047 in any cell (sentiment at `concept`, 0.847 to 0.800, with intervals that do not overlap); at `last` sentiment moves by -0.025, truth by -0.007 and utility by -0.002, and at `mean` no task moves by more than 0.009.
- **Against decoy.** At `last` sentiment (0.749) and truth (0.653) stay above chance, sentiment losing 0.058 and truth nothing, and utility falls to 0.331 [0.255, 0.430], a drop of 0.357 to below chance. At `concept` sentiment drops 0.081 (0.766), truth 0.034 (0.658) and utility 0.187 (0.518 [0.491, 0.544], an interval that reaches 0.5). At `mean` no task moves by more than 0.008 and the intervals overlap in all three. As on the smaller sizes, the collapse at `last` belongs to utility.
- **Zero-shot.** The model's own answer reads 0.589, 0.559 and 0.551 on the original prompts (the truth and utility intervals, [0.441, 0.678] and [0.492, 0.609], include 0.5), 0.500, 0.525 and 0.555 under the random decoy and 0.000, 0.034 and 0.000 under the `against` decoy. The reader at `mean` stays at 0.898, 0.688 and 0.677 under the same decoy.

#### gpt2-xl

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L43   | 0.776 [0.756, 0.794] | 0.732 [0.712, 0.752] | 0.714 [0.692, 0.738] |
| sentiment | concept   | L23   | 0.877 [0.860, 0.892] | 0.890 [0.875, 0.904] | 0.871 [0.855, 0.887] |
| sentiment | mean      | L24   | 0.887 [0.872, 0.902] | 0.892 [0.877, 0.907] | 0.892 [0.877, 0.907] |
| sentiment | zero-shot | -     | 0.589 [0.523, 0.654] | 0.537 [0.472, 0.603] | 0.000 [0.000, 0.000] |
| truth     | last      | L26   | 0.656 [0.646, 0.666] | 0.624 [0.603, 0.644] | 0.631 [0.605, 0.656] |
| truth     | concept   | L37   | 0.703 [0.686, 0.720] | 0.636 [0.622, 0.649] | 0.642 [0.634, 0.653] |
| truth     | mean      | L45   | 0.702 [0.675, 0.727] | 0.703 [0.678, 0.725] | 0.703 [0.678, 0.725] |
| truth     | zero-shot | -     | 0.492 [0.373, 0.610] | 0.475 [0.356, 0.594] | 0.034 [0.000, 0.085] |
| utility   | last      | L33   | 0.686 [0.673, 0.699] | 0.670 [0.654, 0.684] | 0.228 [0.166, 0.297] |
| utility   | concept   | L29   | 0.716 [0.704, 0.730] | 0.729 [0.716, 0.744] | 0.520 [0.487, 0.554] |
| utility   | mean      | L23   | 0.673 [0.645, 0.702] | 0.676 [0.644, 0.706] | 0.680 [0.647, 0.710] |
| utility   | zero-shot | -     | 0.574 [0.512, 0.633] | 0.566 [0.508, 0.625] | 0.004 [0.000, 0.012] |

- **Position.** On the original prompts the positions read 0.776, 0.877 and 0.887 for sentiment (a spread of 0.111), 0.656, 0.703 and 0.702 for truth (0.047) and 0.686, 0.716 and 0.673 for utility (0.043). Sentiment at `last` is lower than at the other two positions by about 0.1, a larger spread than on `gpt2` (0.064), `gpt2-medium` (0.100) and `gpt2-large` (0.086).
- **Random decoy.** The reader moves by at most 0.067 in any cell (truth at `concept`, 0.703 to 0.636, with intervals that do not overlap); at `last` sentiment moves by -0.044 and truth by -0.032 (intervals that barely separate) and utility by -0.016 (overlapping), at `concept` sentiment and utility move up by 0.013, and at `mean` no task moves by more than 0.005.
- **Against decoy.** At `last` sentiment (0.714) and truth (0.631) lose 0.062 and 0.025 and stay above chance, and utility falls to 0.228 [0.166, 0.297], a drop of 0.458 to below chance. At `concept` sentiment is nearly unchanged (0.871), truth drops 0.061 (0.642 [0.634, 0.653]) and utility 0.196 (0.520 [0.487, 0.554], an interval that includes 0.5). At `mean` no task moves by more than 0.007.
- **Zero-shot.** The model's own answer reads 0.589, 0.492 and 0.574 on the original prompts (the truth interval, [0.373, 0.610], includes 0.5), 0.537, 0.475 and 0.566 under the random decoy and 0.000, 0.034 and 0.004 under the `against` decoy. The reader at `mean` stays at 0.892, 0.703 and 0.680.

#### Across the GPT-2 sizes

- **The utility collapse at `last` is present at all four sizes.** The against-decoy utility reader at `last` reads 0.133 (`gpt2`), 0.330 (`gpt2-medium`), 0.331 (`gpt2-large`) and 0.228 (`gpt2-xl`), all below chance; it is not monotone in size, and `gpt2` is the lowest. Sentiment and truth stay above chance at `last` at every size.
- **The reader at `mean` is not turned by the decoy at any size.** The utility reader at `mean` is within 0.014 of its original value under the `against` decoy on all four sizes, and on the two new sizes no task moves by more than 0.009. The decoy that turns the model's own answer to or near 0 (0.000 to 0.034 on sentiment and truth, 0.000 and 0.004 on utility at the two new sizes) does not turn that reader.
- **Limits.** The intervals are over the 10 training draws and do not include the sampling of the evaluation pairs, which matters most for truth (59 report pairs). The layer is chosen on the original prompts per cell, so a difference between sizes in a cell mixes the size with the layer selected.

### Stimulus-set size on qwen3-1.7b-base (P3)

The design is that of [P3 on gpt2](#stimulus-set-size-p3): 2 to 512 pairs, 20 draws per size, the mean accuracy in each cell and in parentheses the share of the 20 draws whose accuracy is below 0.6. The truth set has 172 pairs, so its 256 and 512 cells are skipped, as is the 512 cell of `truth_cities` (452 pairs).

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.755 (0.05) | 0.820 (0.00) | 0.855 (0.00) | 0.901 (0.00) | 0.926 (0.00) | 0.937 (0.00) | 0.958 (0.00) | 0.968 (0.00) | 0.971 (0.00) |
| truth -> truth         | 0.886 (0.00) | 0.886 (0.00) | 0.936 (0.00) | 0.962 (0.00) | 0.972 (0.00) | 0.972 (0.00) | 0.971 (0.00) | -            | -            |
| truth_cities -> truth  | 0.946 (0.00) | 0.957 (0.00) | 0.953 (0.00) | 0.958 (0.00) | 0.958 (0.00) | 0.961 (0.00) | 0.962 (0.00) | 0.963 (0.00) | -            |
| truth_common -> truth  | 0.709 (0.30) | 0.810 (0.10) | 0.878 (0.00) | 0.915 (0.00) | 0.932 (0.00) | 0.931 (0.00) | 0.947 (0.00) | 0.954 (0.00) | 0.953 (0.00) |
| utility -> utility     | 0.651 (0.25) | 0.701 (0.05) | 0.748 (0.00) | 0.747 (0.00) | 0.756 (0.00) | 0.758 (0.00) | 0.749 (0.00) | 0.754 (0.00) | 0.749 (0.00) |

The accuracy of a reader fitted on all the data of the source (the reference, report split) is 0.981 for sentiment (2,048 pairs), 0.966 for truth (172), 0.966 for `truth_cities` (452), 0.949 for `truth_common` (1,296) and 0.770 for utility (2,048). The shuffled-label series is between 0.47 and 0.56 at every size.

- **Sentiment** has one failing draw in 20 at 2 pairs and none from 4 pairs; it reaches 0.937 at 64 pairs and 0.971 [0.969, 0.972] at 512, below the reference of 0.981. **Utility** plateaus early: 0.748 at 8 pairs and 0.749 [0.742, 0.756] at 512, below its reference of 0.770, with no failing draw from 8 pairs.
- **Truth is not the slow case here.** The mean accuracy is 0.886 at 2 pairs with no failing draw, 0.972 at 32 and 64 pairs and 0.971 [0.968, 0.975] at 128, at its reference of 0.966; the source is exhausted at 172 pairs. `truth_cities` is at 0.946 with 2 pairs and 0.963 at 256. `truth_common` needs more: 0.709 at 2 pairs (failure 0.30), 0.878 at 8 and 0.953 [0.947, 0.961] at 512, at its reference of 0.949. On `gpt2` the same truth runs read at most 0.642 at every size; the two models are different, so this is a difference between models on the same data and the run does not say what causes it.
- The stored saturation size is the smallest size whose mean accuracy is within 0.01 of the mean at the largest size, with its bootstrap range: 256 for sentiment (128 to 256), 16 for truth (16 to 32), 4 for `truth_cities` (2 to 16), 128 for `truth_common` (32 to 256) and 8 for utility (8 to 16). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are limits of the set as well as measured plateaus.

## Persona wording and instruction confounds (P5)

P5 asks whether the honesty reader reads honesty or the persona's words. Each condition changes one thing against the paper persona prompt; the reader is trained on the response tokens of true statements as in R5, and scored on plain true against false statements at the token mean, on the 59 `facts` test pairs and on the test splits of five Geometry of Truth subsets (133 `cities`, 142 `neg_cities`, 36 `sp_en_trans`, 187 `larger_than` and 256 `common_claim_true_false` pairs). The conditions are the paper pair ("an honest" against "an untruthful"), two synonym pairs (`synonym_0`: "a truthful" and "a deceitful"; `synonym_1`: "a sincere" and "a lying"), two unrelated pairs (`unrelated_0`: "a tall" and "a short"; `unrelated_1`: "an old" and "a young"), and three controls: `instruction_only` (the paper personas with an empty response), `fact_only` (no persona, plain true against plain false statements) and `random_label` (the paper stimuli with the training labels swapped in exactly half of the pairs). Accuracies are raw: a reader whose first persona aligns with the false statements scores below 0.5. Layers are chosen per condition on the 53 `facts` validation pairs (the `Layer` column) and every set is scored at that layer. The in-template accuracy of the persona conditions is 1.0 by construction, as for R5, and is not reported. The `instruction_only` readers are identical across seeds (every training set holds the same two texts), so that row comes from one reader and carries no interval; any other cell shown without an interval has a stored interval that rounds to the point value. Intervals are over the 10 readers; they do not include the sampling of the evaluation pairs, which matters most for `facts` (59) and `sp_en_trans` (36).

The runs are `p5/gpt2/80b4ae80a6ca`, `p5/qwen3-1.7b-base/a4bd463deeca` and `p5/qwen3-1.7b/7e5e579d102e`, all made at `c7d8051` with a clean tree on a cloud GPU (NVIDIA GeForce RTX 4090). The earlier `gpt2` attempts stopped on a fit that produced a direction of zero norm; these are the runs after the fix. Three more runs, `p5/gpt2-medium/bd9f565baa76`, `p5/qwen3-0.6b-base/b6a702e460e9` and `p5/qwen3-0.6b/fb664c3b4be2`, were made at `fb3b5dd` with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU) with batch size 8 and the same config; the P5 code is unchanged between `c7d8051` and `fb3b5dd`, so these runs differ from the first three in checkpoint and in hardware. Two more runs, `p5/gpt2-large/430d434007c9` and `p5/gpt2-xl/4171a611f734`, were made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500) with batch size 8 and the same config; the P5 code is unchanged since `fb3b5dd` (`git diff --stat fb3b5dd f4d414d -- src` lists no P5 file), so these runs differ from the others in checkpoint and in hardware. Base and post-trained rows are separate tables and are never pooled. The stored gap is the mean of the paper and the two synonym conditions minus the mean of the two unrelated ones, per set.

### gpt2

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L10   | 0.353 [0.327, 0.380] | 0.467 [0.456, 0.477] | 0.478 [0.475, 0.482] | 0.531 [0.517, 0.544] | 0.185 [0.177, 0.194] | 0.533 [0.525, 0.542] |
| synonym_0        | L7    | 0.507 [0.492, 0.520] | 0.508 [0.498, 0.519] | 0.460 [0.451, 0.469] | 0.581 [0.561, 0.600] | 0.329 [0.318, 0.340] | 0.588 [0.579, 0.596] |
| synonym_1        | L7    | 0.542 [0.525, 0.559] | 0.568 [0.556, 0.579] | 0.522 [0.515, 0.528] | 0.558 [0.533, 0.581] | 0.333 [0.325, 0.341] | 0.552 [0.545, 0.559] |
| unrelated_0      | L5    | 0.459 [0.436, 0.481] | 0.531 [0.520, 0.541] | 0.456 [0.445, 0.465] | 0.456 [0.442, 0.469] | 0.670 [0.659, 0.681] | 0.514 [0.507, 0.522] |
| unrelated_1      | L7    | 0.449 [0.439, 0.461] | 0.398 [0.382, 0.413] | 0.527 [0.516, 0.538] | 0.564 [0.531, 0.594] | 0.826 [0.818, 0.833] | 0.380 [0.372, 0.389] |
| instruction_only | L11   | 0.576                | 0.526                | 0.514                | 0.639                | 0.433                | 0.492                |
| fact_only        | L7    | 0.663 [0.641, 0.683] | 0.524 [0.498, 0.547] | 0.507 [0.496, 0.519] | 0.500 [0.469, 0.531] | 0.367 [0.325, 0.414] | 0.593 [0.579, 0.607] |
| random_label     | L5    | 0.529 [0.486, 0.568] | 0.489 [0.457, 0.520] | 0.475 [0.449, 0.503] | 0.481 [0.431, 0.522] | 0.516 [0.448, 0.592] | 0.530 [0.498, 0.557] |

- **The paper persona does not read honesty on `gpt2`.** On `facts` it scores 0.353 [0.327, 0.380], with an interval entirely below 0.5: the reader ranks true and false the wrong way round. The other sets are 0.467 (`cities`), 0.478 (`neg_cities`), 0.531 (`sp_en_trans`), 0.533 (`common_claim`) and, far below chance, 0.185 on `larger_than`.
- **The group means of the synonyms and the unrelated words do not separate on `facts`.** `synonym_0` and `synonym_1` read 0.507 and 0.542, and the unrelated pairs 0.459 and 0.449; the stored gap between the group means is 0.013. Single wordings can still differ: `synonym_1` (0.542 [0.525, 0.559]) and `unrelated_1` (0.449 [0.439, 0.461]) have intervals that do not overlap. The only condition above 0.6 on `facts` is `fact_only`, 0.663 [0.641, 0.683], which reads plain statements with no persona.
- **`larger_than` depends on the persona words.** The paper and synonym readers score 0.185, 0.329 and 0.333, the unrelated readers 0.670 and 0.826 (a gap of -0.465), with intervals that exclude 0.5 in all five, so the sign of the reading on that set follows the words. The other gaps are 0.050 (`cities`), -0.005 (`neg_cities`), 0.047 (`sp_en_trans`) and 0.111 (`common_claim`). The run does not say what the readers of the unrelated words read on `larger_than`.
- **The shuffled-label control reads at chance.** `random_label` has intervals that include 0.5 on all six sets (for example 0.529 [0.486, 0.568] on `facts`). Its stored in-template column, a reader fitted on the half-swapped labels and scored on the real labels of the persona pairs, is 0.602 [0.308, 0.863] at L6.

### qwen3-1.7b-base

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L20   | 0.736 [0.715, 0.754] | 0.889 [0.881, 0.896] | 0.558 [0.547, 0.570] | 0.664 [0.647, 0.678] | 0.734 [0.730, 0.737] | 0.645 [0.640, 0.650] |
| synonym_0        | L20   | 0.695 [0.678, 0.715] | 0.887 [0.876, 0.900] | 0.629 [0.620, 0.637] | 0.647 [0.619, 0.678] | 0.681 [0.671, 0.692] | 0.697 [0.691, 0.705] |
| synonym_1        | L21   | 0.647 [0.625, 0.675] | 0.718 [0.712, 0.724] | 0.460 [0.450, 0.470] | 0.686 [0.661, 0.708] | 0.573 [0.560, 0.585] | 0.664 [0.657, 0.671] |
| unrelated_0      | L17   | 0.600 [0.581, 0.620] | 0.659 [0.649, 0.671] | 0.465 [0.451, 0.480] | 0.592 [0.564, 0.619] | 0.730 [0.714, 0.748] | 0.627 [0.620, 0.634] |
| unrelated_1      | L4    | 0.520 [0.510, 0.529] | 0.518 [0.508, 0.528] | 0.474 [0.465, 0.481] | 0.494 [0.478, 0.511] | 0.669 [0.661, 0.676] | 0.477 [0.473, 0.482] |
| instruction_only | L25   | 0.627                | 0.684                | 0.415                | 0.639                | 0.428                | 0.492                |
| fact_only        | L28   | 0.702 [0.678, 0.719] | 0.544 [0.533, 0.556] | 0.567 [0.558, 0.577] | 0.519 [0.508, 0.533] | 0.693 [0.690, 0.696] | 0.681 [0.668, 0.693] |
| random_label     | L21   | 0.514 [0.446, 0.585] | 0.523 [0.443, 0.605] | 0.477 [0.408, 0.558] | 0.514 [0.456, 0.567] | 0.501 [0.382, 0.617] | 0.520 [0.465, 0.575] |

- **The paper persona reads above chance on this checkpoint**: 0.736 [0.715, 0.754] on `facts`, 0.889 on `cities`, 0.664 on `sp_en_trans`, 0.734 on `larger_than` and 0.645 on `common_claim`; `neg_cities` is close to chance (0.558). `fact_only`, with no persona, reads `facts` at 0.702 [0.678, 0.719] but `cities` only at 0.544, so the persona prompt reads `cities` better than the plain statements do.
- **The persona words matter on most sets, most on `cities`.** The stored gaps are positive on five of the six sets (0.079 to 0.243) and largest on `cities`; the paper and synonym readers score 0.736, 0.695 and 0.647 on `facts` and the unrelated readers 0.600 and 0.520 (stored gap 0.132); on `cities` they score 0.889, 0.887 and 0.718 against 0.659 and 0.518 (gap 0.243). The words do not account for all of it: `unrelated_0` ("a tall", "a short") reads 0.600 on `facts`, above chance with an interval of [0.581, 0.620], and 0.730 on `larger_than`, as high as the paper reader (gap -0.037 on that set). The other gaps are 0.079 (`neg_cities`), 0.123 (`sp_en_trans`) and 0.116 (`common_claim`).
- **The shuffled-label control reads at chance.** `random_label` has intervals that include 0.5 on all six sets (0.514 [0.446, 0.585] on `facts`). Its stored in-template column is 0.676 [0.413, 0.898] at L3.
- `instruction_only` is one reader per cell: 0.627 on `facts` and 0.684 on `cities`. With no interval and 59 `facts` pairs it is not compared with the other rows here.

### qwen3-1.7b

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L20   | 0.744 [0.719, 0.769] | 0.820 [0.805, 0.832] | 0.525 [0.514, 0.538] | 0.731 [0.706, 0.750] | 0.734 [0.722, 0.747] | 0.730 [0.725, 0.735] |
| synonym_0        | L19   | 0.678 [0.664, 0.690] | 0.775 [0.760, 0.790] | 0.440 [0.425, 0.456] | 0.767 [0.736, 0.797] | 0.485 [0.436, 0.529] | 0.676 [0.667, 0.684] |
| synonym_1        | L18   | 0.688 [0.663, 0.714] | 0.822 [0.801, 0.841] | 0.609 [0.597, 0.621] | 0.853 [0.825, 0.875] | 0.769 [0.760, 0.777] | 0.680 [0.673, 0.688] |
| unrelated_0      | L3    | 0.510 [0.508, 0.514] | 0.508 [0.505, 0.512] | 0.485 [0.482, 0.489] | 0.353 [0.339, 0.367] | 0.588 [0.583, 0.592] | 0.529 [0.528, 0.530] |
| unrelated_1      | L5    | 0.492                | 0.533 [0.529, 0.537] | 0.547 [0.544, 0.550] | 0.367 [0.361, 0.375] | 0.600 [0.597, 0.603] | 0.537 [0.536, 0.540] |
| instruction_only | L28   | 0.576                | 0.556                | 0.486                | 0.472                | 0.684                | 0.664                |
| fact_only        | L28   | 0.736 [0.719, 0.754] | 0.608 [0.590, 0.628] | 0.513 [0.501, 0.525] | 0.547 [0.506, 0.586] | 0.661 [0.639, 0.684] | 0.694 [0.687, 0.702] |
| random_label     | L22   | 0.524 [0.488, 0.563] | 0.556 [0.482, 0.625] | 0.505 [0.470, 0.546] | 0.550 [0.514, 0.583] | 0.540 [0.441, 0.637] | 0.495 [0.457, 0.533] |

- **The paper persona reads above chance**: 0.744 [0.719, 0.769] on `facts`, 0.820 on `cities`, 0.731 on `sp_en_trans`, 0.734 on `larger_than` and 0.730 on `common_claim`; `neg_cities` is 0.525 [0.514, 0.538]. `fact_only` reads `facts` at 0.736 [0.719, 0.754] and `common_claim` at 0.694, and the other four sets between 0.513 and 0.661.
- **The persona words matter more than on the base checkpoint.** The paper and synonym readers read 0.744, 0.678 and 0.688 on `facts` against 0.510 and 0.492 for the unrelated pairs (gap 0.203), 0.820, 0.775 and 0.822 on `cities` against 0.508 and 0.533 (0.285), and 0.731, 0.767 and 0.853 on `sp_en_trans` against 0.353 and 0.367 (0.424), where the unrelated readers fall below chance with intervals entirely below 0.5. On `common_claim` the gap is 0.162, with 0.730, 0.676 and 0.680 against 0.529 and 0.537. The sets with the weakest dependence are `neg_cities` (gap 0.009, with readings between 0.440 and 0.609 across the five persona conditions) and `larger_than` (0.069; `synonym_0` reads 0.485 [0.436, 0.529] there while `synonym_1` reads 0.769).
- **The shuffled-label control reads at chance.** `random_label` has intervals that include 0.5 on five of the six sets (0.524 [0.488, 0.563] on `facts`); the exception is `sp_en_trans`, 0.550 [0.514, 0.583]. Its stored in-template column is 0.701 [0.462, 0.894] at L4.
- `instruction_only` is one reader per cell: 0.576 on `facts` and 0.664 on `common_claim`, without an interval.

### gpt2-medium

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L24   | 0.592 [0.578, 0.605] | 0.522 [0.511, 0.531] | 0.485 [0.474, 0.494] | 0.592 [0.553, 0.628] | 0.481 [0.458, 0.501] | 0.467 [0.459, 0.475] |
| synonym_0        | L15   | 0.449 [0.431, 0.466] | 0.502 [0.493, 0.511] | 0.451 [0.444, 0.461] | 0.475 [0.461, 0.492] | 0.295 [0.282, 0.309] | 0.527 [0.517, 0.537] |
| synonym_1        | L11   | 0.500 [0.480, 0.519] | 0.598 [0.585, 0.611] | 0.517 [0.508, 0.525] | 0.467 [0.444, 0.494] | 0.502 [0.472, 0.540] | 0.551 [0.541, 0.560] |
| unrelated_0      | L16   | 0.514 [0.495, 0.529] | 0.510 [0.498, 0.520] | 0.506 [0.500, 0.514] | 0.428 [0.408, 0.447] | 0.519 [0.506, 0.530] | 0.497 [0.490, 0.503] |
| unrelated_1      | L10   | 0.486 [0.466, 0.503] | 0.475 [0.462, 0.490] | 0.477 [0.470, 0.484] | 0.522 [0.514, 0.533] | 0.796 [0.778, 0.818] | 0.412 [0.402, 0.424] |
| instruction_only | L11   | 0.542                | 0.586                | 0.444                | 0.556                | 0.738                | 0.555                |
| fact_only        | L20   | 0.644 [0.619, 0.669] | 0.546 [0.537, 0.556] | 0.519 [0.512, 0.525] | 0.475 [0.436, 0.522] | 0.617 [0.548, 0.678] | 0.632 [0.621, 0.645] |
| random_label     | L9    | 0.536 [0.500, 0.573] | 0.501 [0.476, 0.530] | 0.480 [0.442, 0.516] | 0.556 [0.517, 0.592] | 0.440 [0.374, 0.526] | 0.547 [0.511, 0.578] |

- **The paper persona reads `facts` above chance on `gpt2-medium`, unlike `gpt2`.** It scores 0.592 [0.578, 0.605] (0.353 on `gpt2`). The other sets are 0.522 (`cities`), 0.592 [0.553, 0.628] (`sp_en_trans`) and, with intervals below 0.5, 0.485 (`neg_cities`) and 0.467 (`common_claim`); `larger_than` is 0.481 [0.458, 0.501].
- **The group means of the synonyms and the unrelated words do not separate on `facts`.** `synonym_0` and `synonym_1` read 0.449 and 0.500, the unrelated pairs 0.514 and 0.486; the stored gap is 0.014 (0.013 on `gpt2`). `synonym_0` is below chance on `facts` (0.449 [0.431, 0.466]). `fact_only`, with no persona, is the highest condition on `facts` (0.644 [0.619, 0.669]), above the paper persona.
- **`larger_than` depends on the persona words.** The paper and synonym readers score 0.481, 0.295 [0.282, 0.309] and 0.502, the unrelated readers 0.519 and 0.796 [0.778, 0.818]; the stored gap is -0.231 (-0.465 on `gpt2`), and the two unrelated pairs differ by 0.277 between themselves. The other gaps are 0.048 (`cities`), -0.007 (`neg_cities`), 0.036 (`sp_en_trans`) and 0.060 (`common_claim`). The run does not say what the readers of the unrelated words read on `larger_than`.
- **The shuffled-label control reads at chance on four of the six sets.** `random_label` has intervals that include 0.5 on `facts` (0.536 [0.500, 0.573], a lower bound at 0.500), `cities`, `neg_cities` and `larger_than`; the exceptions are `sp_en_trans` (0.556 [0.517, 0.592]) and `common_claim` (0.547 [0.511, 0.578]). Its stored in-template column is 0.662 [0.420, 0.866] at L24.
- `instruction_only` is one reader per cell: 0.542 on `facts` and 0.738 on `larger_than`, without an interval.

### gpt2-large

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L13   | 0.403 [0.385, 0.425] | 0.459 [0.450, 0.468] | 0.530 [0.518, 0.541] | 0.583 [0.561, 0.603] | 0.364 [0.350, 0.381] | 0.564 [0.559, 0.569] |
| synonym_0        | L3    | 0.590 [0.569, 0.610] | 0.549 [0.538, 0.559] | 0.504 [0.495, 0.513] | 0.533 [0.508, 0.558] | 0.311 [0.288, 0.335] | 0.548 [0.543, 0.553] |
| synonym_1        | L4    | 0.436 [0.417, 0.453] | 0.503 [0.490, 0.517] | 0.489 [0.482, 0.497] | 0.508 [0.486, 0.531] | 0.341 [0.332, 0.350] | 0.519 [0.512, 0.525] |
| unrelated_0      | L4    | 0.595 [0.573, 0.617] | 0.532 [0.519, 0.545] | 0.579 [0.569, 0.589] | 0.519 [0.494, 0.544] | 0.557 [0.529, 0.582] | 0.490 [0.485, 0.495] |
| unrelated_1      | L14   | 0.463 [0.446, 0.480] | 0.456 [0.440, 0.471] | 0.485 [0.475, 0.494] | 0.533 [0.517, 0.550] | 0.706 [0.693, 0.720] | 0.583 [0.575, 0.591] |
| instruction_only | L35   | 0.627                | 0.624                | 0.500                | 0.500                | 0.428                | 0.609                |
| fact_only        | L30   | 0.663 [0.632, 0.693] | 0.541 [0.529, 0.551] | 0.528 [0.523, 0.532] | 0.503 [0.481, 0.528] | 0.539 [0.462, 0.625] | 0.650 [0.635, 0.666] |
| random_label     | L29   | 0.554 [0.503, 0.607] | 0.487 [0.430, 0.538] | 0.513 [0.476, 0.549] | 0.533 [0.483, 0.583] | 0.501 [0.378, 0.618] | 0.507 [0.480, 0.535] |

- **The paper persona does not read honesty on `gpt2-large`.** On `facts` it scores 0.403 [0.385, 0.425], with an interval entirely below 0.5 (0.353 on `gpt2`, 0.592 on `gpt2-medium`). The other sets are 0.459 (`cities`), 0.530 (`neg_cities`), 0.583 (`sp_en_trans`), 0.564 (`common_claim`) and, far below chance, 0.364 on `larger_than`.
- **The group means of the synonyms and the unrelated words do not separate on `facts`.** `synonym_0` and `synonym_1` read 0.590 and 0.436, and the unrelated pairs 0.595 and 0.463; the stored gap between the group means is -0.053, with the unrelated readers above the others. Single wordings differ more than the groups do: `synonym_0` and `synonym_1` are 0.154 apart with intervals that do not overlap. The only condition above 0.6 on `facts` is `fact_only`, 0.663 [0.632, 0.693], which reads plain statements with no persona.
- **`larger_than` depends on the persona words.** The paper and synonym readers score 0.364, 0.311 and 0.341, the unrelated readers 0.557 and 0.706 (a gap of -0.293), with intervals that exclude 0.5 in all five, so the sign of the reading on that set follows the words. The other gaps are 0.010 (`cities`), -0.024 (`neg_cities`), 0.015 (`sp_en_trans`) and 0.007 (`common_claim`). The run does not say what the readers of the unrelated words read on `larger_than`.
- **The shuffled-label control reads at chance on five of the six sets.** `random_label` has intervals that include 0.5 on `cities`, `neg_cities`, `sp_en_trans`, `larger_than` and `common_claim`; on `facts` it is 0.554 [0.503, 0.607], an interval whose lower bound is just above 0.5. Its stored in-template column is 0.654 [0.427, 0.857] at L21.
- `instruction_only` is one reader per cell: 0.627 on `facts`, 0.428 on `larger_than` and exactly 0.500 on `neg_cities` and `sp_en_trans`, without an interval.

### gpt2-xl

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L2    | 0.481 [0.461, 0.498] | 0.505 [0.495, 0.514] | 0.553 [0.542, 0.564] | 0.469 [0.456, 0.481] | 0.379 [0.353, 0.407] | 0.534 [0.525, 0.543] |
| synonym_0        | L43   | 0.485 [0.461, 0.510] | 0.629 [0.620, 0.640] | 0.480 [0.472, 0.487] | 0.592 [0.581, 0.603] | 0.359 [0.348, 0.371] | 0.548 [0.540, 0.556] |
| synonym_1        | L19   | 0.556 [0.546, 0.568] | 0.545 [0.534, 0.558] | 0.480 [0.467, 0.493] | 0.425 [0.406, 0.442] | 0.618 [0.611, 0.625] | 0.532 [0.525, 0.540] |
| unrelated_0      | L6    | 0.547 [0.529, 0.566] | 0.516 [0.505, 0.526] | 0.565 [0.553, 0.577] | 0.525 [0.508, 0.539] | 0.707 [0.695, 0.718] | 0.486 [0.478, 0.495] |
| unrelated_1      | L10   | 0.486 [0.469, 0.505] | 0.453 [0.444, 0.461] | 0.474 [0.466, 0.483] | 0.528 [0.517, 0.539] | 0.705 [0.693, 0.716] | 0.502 [0.498, 0.505] |
| instruction_only | L34   | 0.508                | 0.609                | 0.486                | 0.556                | 0.390                | 0.520                |
| fact_only        | L38   | 0.702 [0.686, 0.720] | 0.626 [0.596, 0.650] | 0.498 [0.483, 0.513] | 0.539 [0.503, 0.572] | 0.578 [0.480, 0.669] | 0.659 [0.649, 0.670] |
| random_label     | L7    | 0.507 [0.461, 0.556] | 0.515 [0.492, 0.536] | 0.489 [0.465, 0.519] | 0.481 [0.447, 0.514] | 0.511 [0.417, 0.596] | 0.502 [0.465, 0.536] |

- **The paper persona sits at or just below chance on `facts` on `gpt2-xl`.** It scores 0.481 [0.461, 0.498], with an interval that stays below 0.5. The other sets are 0.505 (`cities`), 0.553 (`neg_cities`), 0.469 (`sp_en_trans`), 0.534 (`common_claim`) and 0.379 on `larger_than`.
- **The group means of the synonyms and the unrelated words do not separate on `facts`.** `synonym_0` and `synonym_1` read 0.485 and 0.556, and the unrelated pairs 0.547 and 0.486; the stored gap is -0.010. The only condition above 0.6 on `facts` is `fact_only`, 0.702 [0.686, 0.720], the highest `facts` value of the four GPT-2 sizes. The layers chosen on the 53 validation pairs range from L2 (paper) to L43 (`synonym_0`), so a comparison between conditions mixes the wording with a layer picked on a small split.
- **`larger_than` depends on the persona words.** The paper and synonym readers score 0.379, 0.359 and 0.618, the unrelated readers 0.707 and 0.705 (a gap of -0.254), with intervals that exclude 0.5 in all five. The two synonyms are on opposite sides of 0.5 (0.359 and 0.618), 0.259 apart, so the sign of the reading follows the individual wording and not the group. The other gaps are 0.075 (`cities`), -0.015 (`neg_cities`), -0.031 (`sp_en_trans`) and 0.044 (`common_claim`); the `cities` gap comes from `synonym_0` (0.629 [0.620, 0.640]) against unrelated readers at 0.516 and 0.453.
- **The shuffled-label control reads at chance.** `random_label` has intervals that include 0.5 on all six sets (for example 0.507 [0.461, 0.556] on `facts`). Its stored in-template column is 0.663 [0.440, 0.854] at L21.
- `instruction_only` is one reader per cell: 0.508 on `facts`, 0.609 on `cities` and 0.390 on `larger_than`, without an interval.

### qwen3-0.6b-base

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L18   | 0.597 [0.581, 0.614] | 0.658 [0.650, 0.665] | 0.387 [0.381, 0.392] | 0.750 [0.733, 0.767] | 0.316 [0.302, 0.328] | 0.618 [0.611, 0.624] |
| synonym_0        | L21   | 0.744 [0.727, 0.761] | 0.735 [0.723, 0.751] | 0.540 [0.526, 0.556] | 0.606 [0.581, 0.631] | 0.399 [0.377, 0.426] | 0.620 [0.611, 0.630] |
| synonym_1        | L21   | 0.575 [0.558, 0.590] | 0.610 [0.601, 0.617] | 0.432 [0.425, 0.437] | 0.700 [0.678, 0.719] | 0.363 [0.329, 0.396] | 0.630 [0.626, 0.635] |
| unrelated_0      | L17   | 0.541 [0.525, 0.556] | 0.547 [0.529, 0.560] | 0.585 [0.575, 0.595] | 0.497 [0.469, 0.528] | 0.603 [0.593, 0.615] | 0.621 [0.613, 0.630] |
| unrelated_1      | L15   | 0.419 [0.392, 0.447] | 0.414 [0.403, 0.425] | 0.538 [0.532, 0.544] | 0.344 [0.319, 0.369] | 0.463 [0.456, 0.469] | 0.453 [0.443, 0.461] |
| instruction_only | L1    | 0.525                | 0.504                | 0.542                | 0.556                | 0.455                | 0.574                |
| fact_only        | L17   | 0.746 [0.732, 0.759] | 0.753 [0.729, 0.777] | 0.450 [0.432, 0.470] | 0.811 [0.781, 0.839] | 0.432 [0.388, 0.488] | 0.682 [0.671, 0.694] |
| random_label     | L21   | 0.517 [0.429, 0.607] | 0.541 [0.508, 0.569] | 0.476 [0.441, 0.509] | 0.525 [0.428, 0.622] | 0.426 [0.334, 0.512] | 0.518 [0.463, 0.575] |

- **The paper persona reads `facts` above chance and two sets below it.** It scores 0.597 [0.581, 0.614] on `facts`, 0.658 on `cities`, 0.750 on `sp_en_trans` and 0.618 on `common_claim`; `neg_cities` (0.387 [0.381, 0.392]) and `larger_than` (0.316 [0.302, 0.328]) have intervals below 0.5. `fact_only` reads `facts` at 0.746 [0.732, 0.759], and `synonym_0` reads it as high (0.744 [0.727, 0.761]).
- **The persona words matter on `facts`, as on the 1.7B base checkpoint.** The paper and synonym readers read 0.597, 0.744 and 0.575 on `facts` against 0.541 and 0.419 [0.392, 0.447] for the unrelated pairs (stored gap 0.159), 0.658, 0.735 and 0.610 on `cities` against 0.547 and 0.414 (0.187), and 0.750, 0.606 and 0.700 on `sp_en_trans` against 0.497 and 0.344 (0.264). The gaps are negative on `neg_cities` (-0.108) and `larger_than` (-0.174), where `unrelated_0` ("a tall", "a short") reads 0.585 and 0.603 against 0.387 and 0.316 for the paper persona; the other gap is 0.086 (`common_claim`).
- **The shuffled-label control reads at chance on five of the six sets.** `random_label` has intervals that include 0.5 except on `cities` (0.541 [0.508, 0.569]); on `facts` it reads 0.517 [0.429, 0.607]. Its stored in-template column is 0.650 [0.415, 0.853] at L11.
- `instruction_only` is one reader per cell: 0.525 on `facts` and 0.504 on `cities`.

### qwen3-0.6b

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L19   | 0.598 [0.583, 0.619] | 0.693 [0.663, 0.722] | 0.475 [0.456, 0.492] | 0.661 [0.606, 0.714] | 0.720 [0.656, 0.771] | 0.612 [0.599, 0.624] |
| synonym_0        | L25   | 0.575 [0.553, 0.597] | 0.604 [0.598, 0.609] | 0.550 [0.541, 0.559] | 0.492 [0.461, 0.522] | 0.710 [0.702, 0.718] | 0.629 [0.619, 0.643] |
| synonym_1        | L28   | 0.666 [0.649, 0.681] | 0.595 [0.587, 0.602] | 0.489 [0.480, 0.499] | 0.500 [0.489, 0.511] | 0.564 [0.553, 0.575] | 0.632 [0.625, 0.641] |
| unrelated_0      | L18   | 0.654 [0.644, 0.664] | 0.644 [0.617, 0.677] | 0.487 [0.478, 0.494] | 0.444 [0.419, 0.469] | 0.755 [0.716, 0.783] | 0.571 [0.559, 0.588] |
| unrelated_1      | L8    | 0.558 [0.553, 0.563] | 0.541 [0.535, 0.550] | 0.546 [0.537, 0.556] | 0.322 [0.311, 0.333] | 0.696 [0.611, 0.745] | 0.529 [0.520, 0.537] |
| instruction_only | L4    | 0.508                | 0.519                | 0.549                | 0.389                | 0.738                | 0.527                |
| fact_only        | L27   | 0.653 [0.639, 0.666] | 0.543 [0.535, 0.551] | 0.517 [0.508, 0.524] | 0.478 [0.433, 0.531] | 0.736 [0.696, 0.765] | 0.611 [0.595, 0.625] |
| random_label     | L21   | 0.534 [0.498, 0.566] | 0.529 [0.508, 0.552] | 0.494 [0.482, 0.506] | 0.472 [0.403, 0.547] | 0.588 [0.446, 0.703] | 0.523 [0.493, 0.549] |

- **The paper persona reads above chance except on `neg_cities`.** It scores 0.598 [0.583, 0.619] on `facts`, 0.693 on `cities`, 0.661 on `sp_en_trans`, 0.720 on `larger_than` and 0.612 on `common_claim`; `neg_cities` is 0.475 [0.456, 0.492]. `fact_only` reads `facts` at 0.653 [0.639, 0.666] and `larger_than` at 0.736.
- **The persona words do not matter on `facts` at this checkpoint.** The paper and synonym readers read 0.598, 0.575 and 0.666 and the unrelated readers 0.654 [0.644, 0.664] and 0.558 (stored gap 0.007): `unrelated_0` ("a tall", "a short") reads above the paper persona and as high as `fact_only`. On `larger_than` it reads 0.755 [0.716, 0.783] against 0.720 for the paper persona (gap -0.061). The gaps are positive on `cities` (0.038), `sp_en_trans` (0.168) and `common_claim` (0.074) and negative on `neg_cities` (-0.012). On `sp_en_trans` the unrelated readers fall below chance, with intervals below 0.5 (0.444 [0.419, 0.469] and 0.322 [0.311, 0.333]), while the paper persona reads 0.661; the run does not say what those readers read.
- **The shuffled-label control reads at chance on five of the six sets.** `random_label` has intervals that include 0.5 except on `cities` (0.529 [0.508, 0.552]); on `facts` it reads 0.534 [0.498, 0.566]. Its stored in-template column is 0.648 [0.427, 0.833] at L15.
- `instruction_only` is one reader per cell: 0.508 on `facts` and 0.738 on `larger_than`.

### What the eight runs show together

- The paper persona reader scores below chance on `facts` on three of the four GPT-2 sizes (0.353 on `gpt2`, 0.403 on `gpt2-large` and 0.481 on `gpt2-xl`, each with an interval below 0.5) and above it on `gpt2-medium` (0.592) and on every Qwen3 checkpoint: 0.597 on `qwen3-0.6b-base`, 0.598 on `qwen3-0.6b`, 0.736 on `qwen3-1.7b-base` and 0.744 on `qwen3-1.7b`. On the four GPT-2 sizes only `fact_only` reads `facts` above 0.6 (0.663, 0.644, 0.663 and 0.702 from `gpt2` to `gpt2-xl`); on the Qwen3 checkpoints several persona conditions do.
- The stored gap between the paper and synonym readers and the unrelated readers on `facts` is 0.013 (`gpt2`), 0.014 (`gpt2-medium`), -0.053 (`gpt2-large`), -0.010 (`gpt2-xl`), 0.159 (`qwen3-0.6b-base`), 0.007 (`qwen3-0.6b`), 0.132 (`qwen3-1.7b-base`) and 0.203 (`qwen3-1.7b`). A dependence on the persona words shows on the 1.7B pair and on `qwen3-0.6b-base`, where the unrelated readers fall to or near chance on `facts` (0.600 and 0.520 on `qwen3-1.7b-base`, 0.541 and 0.419 on `qwen3-0.6b-base`, 0.510 and 0.492 on `qwen3-1.7b`) while the paper and synonym readers stay at 0.575 or above; it does not show on the four GPT-2 sizes or on `qwen3-0.6b`, where the unrelated readers read `facts` as the others do (on `gpt2-large` the gap is negative: the unrelated readers are the higher ones). The sign of the base to post-trained difference in this gap is not the same at the two sizes (0.159 to 0.007 at 0.6B, 0.132 to 0.203 at 1.7B). Each gap rests on one run per checkpoint and 59 `facts` pairs, and `unrelated_0` on `qwen3-1.7b-base` (0.600 [0.581, 0.620]) is above chance.
- The gap on `larger_than` is negative on all four GPT-2 sizes: -0.465 (`gpt2`), -0.231 (`gpt2-medium`), -0.293 (`gpt2-large`) and -0.254 (`gpt2-xl`); on `gpt2-large` and `gpt2-xl` the intervals of all five persona conditions exclude 0.5. It does not grow or shrink steadily with size.
- Some cells under the unrelated words sit far from 0.5 (`larger_than` on `gpt2`, 0.670 and 0.826, on `gpt2-medium`, 0.796 for `unrelated_1`, on `gpt2-large`, 0.706 for `unrelated_1`, and on `gpt2-xl`, 0.707 and 0.705; `sp_en_trans` on `qwen3-1.7b`, 0.353 and 0.367, on `qwen3-0.6b-base`, 0.344 for `unrelated_1`, and on `qwen3-0.6b`, 0.444 and 0.322); the runs do not say what those readers read.
- Each cell is one run of one model at one layer chosen on 53 pairs and scored on small sets, with intervals over the training draw only. The runs do not test persona wordings beyond the five pairs above. The first three runs were made on a cloud GPU (`c7d8051`), the next three on a laptop GPU (`fb3b5dd`) and the last two on a rented pod GPU (`f4d414d`), with the same config and the same P5 code.

## Template steering on the Qwen3 1.7B pair (P7)

P7 is the protocol of [P7 on gpt2](#template-steering-p7): the sentiment reader is extracted under six templates of P1 and used to steer with the C1 protocol (coefficients -0.3, -0.15, 0, 0.15 and 0.3, 50 prompts, 2 samples per prompt, 2 steering readers per template), over the sweep layers 1, 6, 12, 17, 23 and 28 (28 is the last block). A unit is a template, a sweep layer and a steering reader (72 units). It is readable if the validation accuracy of its reader at that layer is at least 0.6, and inverted if it is readable and its primary dose slope is negative. The reported runs are `p7/qwen3-1.7b-base/16a7a9e4facf` and `p7/qwen3-1.7b/eacde3abd0aa`, both made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500). They replace `p7/qwen3-1.7b-base/947fef33a7c5` and `p7/qwen3-1.7b/7921f3fe5754` (made at `20650c5` on a laptop GPU, NVIDIA GeForce RTX 4070 Laptop GPU), which stay tracked; the configs differ in `n_random` only (absent earlier, 3 here), so these runs add three random directions (one reader each, steered at every sweep layer with the same grid) and the fluency measure (perplexity of held-out neutral text and distinct-1 of the generated text, from `table_effects`). Base and post-trained are two models: the tables are separate and no metric is pooled over them. The counts of inverted units changed between the two runs of each checkpoint, from 9 to 12 on the base and from 16 to 9 on the post-trained checkpoint, so the ordering of the two checkpoints by inversion rate in the earlier runs (0.125 against 0.222) does not hold in these runs (0.167 against 0.125).

### qwen3-1.7b-base

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at steer layer | Dose slope, sweep layers 1 to 23 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ------------------------- | -------------------------------- | -------------- | -------- |
| paper          | L18        | 0.907         | L28         | 1.124                     | 0.061 to 0.765                   | 12             | 0        |
| no_template    | L20        | 0.928         | L28         | 1.291                     | 0.013 to 0.969                   | 12             | 1        |
| no_chat_tags   | L18        | 0.921         | L28         | 1.336                     | -0.064 to 0.718                  | 12             | 3        |
| neutral_frame  | L18        | 0.941         | L28         | 1.388                     | 0.013 to 0.910                   | 12             | 2        |
| scaffold_0     | L27        | 0.957         | L28         | 1.361                     | -0.104 to 1.013                  | 12             | 4        |
| concept_swap_0 | L27        | 0.856         | L28         | 1.334                     | -0.008 to 0.682                  | 12             | 2        |

### qwen3-1.7b

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at steer layer | Dose slope, sweep layers 1 to 23 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ------------------------- | -------------------------------- | -------------- | -------- |
| paper          | L18        | 0.863         | L28         | 0.847                     | 0.040 to 0.617                   | 12             | 1        |
| no_template    | L16        | 0.937         | L17         | 0.733                     | -0.049 to 0.733                  | 12             | 2        |
| no_chat_tags   | L19        | 0.841         | L28         | 0.953                     | -0.013 to 0.555                  | 12             | 2        |
| neutral_frame  | L18        | 0.914         | L28         | 0.682                     | 0.039 to 0.382                   | 12             | 1        |
| scaffold_0     | L18        | 0.918         | L28         | 0.800                     | 0.008 to 0.575                   | 12             | 2        |
| concept_swap_0 | L18        | 0.851         | L28         | 0.721                     | 0.077 to 0.446                   | 12             | 1        |

The read accuracy is the best point of the mean reading curve over the 10 readers, so it is optimistic by the selection. The sweep-layer column is the minimum and maximum of the mean dose slope over layers 1, 6, 12, 17 and 23 (for `no_template` on the post-trained checkpoint it includes the steering layer, L17), read from the curves files.

- **The steering layer is the last block for 11 of the 12 templates.** The steering layer is L28 for all six templates on the base checkpoint and for five on the post-trained one, where `no_template` steers most at L17 (0.733 against 0.649 at L28). The last layer is the edge of what was swept and not an interior optimum. The slope is not monotone before it (`paper` on the post-trained checkpoint: 0.040, 0.106, 0.187, 0.347, 0.617 and 0.847 at layers 1, 6, 12, 17, 23 and 28; on the base checkpoint 0.122, 0.061, 0.162, 0.123, 0.765 and 1.124). The slope at the steering layer is measured on the prompts that chose that layer and is optimistic. It is 1.124 to 1.388 on the base and 0.682 to 0.953 on the post-trained checkpoint; the run does not say what makes the two differ.
- **Reading layers.** The reading layer is L18 to L20 for four of the six templates on the base checkpoint and L27 for `scaffold_0` and `concept_swap_0`; on the post-trained checkpoint it is L16 to L19 for all six. The mean absolute gap between the reading depth and the steering depth (layer divided by 28) is 0.238 on the base and 0.298 on the post-trained checkpoint.
- **Inversion.** Every unit is readable (72 of 72 on both). On the base checkpoint 12 are inverted, a rate of 0.167 [0.098, 0.269]; on the post-trained checkpoint 9, a rate of 0.125 [0.067, 0.221] (Wilson intervals over the 72 units, which overlap). The counts by template are 0, 1, 3, 2, 4 and 2 on the base checkpoint and 1, 2, 2, 1, 2 and 1 on the post-trained one. On `gpt2` the rate was 1 of 72 (0.014), on `gpt2-large` 3 of 72 and on `gpt2-xl` 20 of 72. Every inverted unit is at layer 1, 6, 12 or 17 (base: 2, 7, 2 and 1 units; post-trained: 7 and 2 units at layers 1 and 6) and none is at layer 23 or 28, so in these runs the sign of the slope is positive at the deeper layers.
- **Reading and steering.** The Spearman correlation over the 72 units between reading accuracy and dose slope is 0.762 on the base and 0.773 on the post-trained checkpoint, and between the logit-lens score and dose slope 0.740 and 0.844. The sign of the slope agrees between the two judges for 0.819 of the units on the base checkpoint (59 of 72) and 0.875 on the post-trained one (63 of 72). The sweep layer with the largest logit-lens separation is the steering layer for 5 of 6 templates on both checkpoints (`scaffold_0` has L23 on the base checkpoint and `no_template` L28 on the post-trained one).
- **The correlation is not separated from depth.** Both the reading accuracy and the slope rise with depth over the sweep, so a rank correlation over units that mixes layers can be driven by depth alone. The summary does not give the within-layer correlation.
- **Prompt format.** The mean primary dose slope by steering prompt format (raw, chat and story) is 1.055 to 1.456, 1.151 to 1.432 and 1.161 to 1.497 over the six templates on the base checkpoint. On the post-trained checkpoint it is 0.701 to 0.814 for raw, 0.403 to 0.700 for chat and 0.636 to 1.005 for story, and the chat value is below the raw and story values of the same template for all six templates there.
- **Random directions.** The largest absolute dose slope of a random direction at any sweep layer is 0.447 on the base checkpoint and 0.269 on the post-trained one, both at layer 28, where the template slopes at the steering layer are 1.124 to 1.388 and 0.682 to 0.953. The random arms are three directions with one reader each. Over layers 1 to 17 their slopes range from -0.175 to 0.184 on the base checkpoint, where the inverted units sit (layers 1, 6, 12 and 17); on the post-trained checkpoint the inverted units sit at layers 1 and 6, where the random slopes range from -0.078 to 0.089 (-0.114 to 0.123 over layers 1 to 23). The template means at those layers are within these ranges, but individual inverted units go lower (per-seed slopes down to -0.228 on the base and -0.135 on the post-trained checkpoint, against random minima of -0.175 and -0.114), so a random direction gives small negative slopes of the same order as the inverted units but not all of them; only the slopes at the deepest layers separate the sentiment direction from random ones.
- **Fluency.** The cells are 288 template cells (6 templates, 6 layers, 2 readers and 4 nonzero coefficients) and 72 random-arm rows (360 per run); the counts below are over the template cells. On the base checkpoint 17 cells have a neutral-text perplexity above 1.5 times the unsteered one (1.54 to 1.99), all at layer 28 and at -0.3 (11) or +0.3 (6), and 3 cells at layer 28 and -0.3 have a distinct-1 below 0.15 (lowest 0.114). On the post-trained checkpoint 1 cell is above 1.5 times (`scaffold_0` at layer 28 and -0.3, 1.51) and 1 cell has a distinct-1 below 0.15 (`no_chat_tags` at layer 28 and -0.3, 0.102). The mean over the templates and readers at layer 28 is 1.48 (+0.3) and 1.65 (-0.3) times the unsteered perplexity on the base checkpoint and 1.20 and 1.31 times on the post-trained one, against 1.035 to 1.157 (base) and 1.026 to 1.204 (post-trained, the largest at layer 17 and 23) at the other layers and the same two coefficients. The dose slope at layer 28 is fitted over a grid that includes the degraded cells on the base checkpoint, so it is not a rate of steering within fluent text there; on the post-trained checkpoint it is nearly all fluent.

## Run identity

All reading runs (R1 to R7) except the R4 and R6 runs of `gpt2-medium`, `gpt2-large` and `gpt2-xl` (made at `ca2546c`,
below) and the R4, R6, A1 and A2 runs of the Qwen3 checkpoints (made at `a6e2333`, below) were made at commit
`a39ae61` with a clean tree, on one laptop GPU with 8 GB of memory, with the default config of each experiment (batch
size 8 for the reading runs; the A1 and A2 configs store no batch size). The A1 and A2 runs of `gpt2` were made at
`a39ae61` as well and read the `gpt2` runs of R1, R2, R3, R5 and R7 below. The C1 runs of `gpt2` and `gpt2-xl` were made at `d59a2f7` and those of
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
emotion tasks), and the P2, P3, P4, M1, C2, LR1, C4, P7 and C5 runs of `gpt2` (and the C5 run `155f0f72bc3f` of `gpt2-xl`; the later runs of the larger sizes are described below) were made at `ca2546c` with a clean tree on the
same laptop GPU, with the default config of each experiment (the batch size of each run is in its `config.json`). The
P1 run `078b9d6482a2` differs from the sentiment run `673af340b77d` in the task list only. The C4 run reads the LR1
adapter named in its config. None of the M1, C2, P7, C4 and C5 runs made at `ca2546c` stores the perplexity or the distinct-1 of the
generated text. The `gpt2` P5 run is the rerun after the fix of the zero-norm fit that stopped the earlier attempts; those attempts
are not exported and not reported.

The P2 and P3 runs of `qwen3-1.7b-base` and the P5 runs of `gpt2`, `qwen3-1.7b-base` and `qwen3-1.7b` were made at
`c7d8051` with a clean tree on a cloud GPU (NVIDIA GeForce RTX 4090, torch 2.14.1+cu130, transformers 5.19.0), with the
default config of each experiment and batch size 8 (`config.json` of each run).

The P2 run of `qwen3-1.7b` and the P7 runs of `qwen3-1.7b-base` and `qwen3-1.7b` were made at `20650c5` (an ancestor of the current main; the P2 and P7 code is unchanged since) with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment (batch size 8 for P2, 16 for P7). The time between `started_at` and `finished_at` of the stored `meta.json` is 912 s for `p2/qwen3-1.7b/51a85fbec631`, 3,383 s for `p7/qwen3-1.7b/7921f3fe5754` and 4,112 s for `p7/qwen3-1.7b-base/947fef33a7c5`. None of these Qwen3 P7 runs stores the perplexity or the distinct-1 of the generated text; they are the earlier runs, and the P7 runs of the same checkpoints reported in this document are the later ones made at `f4d414d` (below).

The P9 runs were made on the same laptop GPU with a clean tree: P9a at `ff1177a` (batch size 8, 4 for `gpt2-xl`), P9b
and P9c at `95a8243` (batch size 8, 4 for `gpt2-xl`) and P9d at `a39ae61` (batch size 8, 4 for `gpt2-xl`). The P9 runs of `gpt2`,
`gpt2-medium` and `gpt2-large` (P9a to P9c, and P9d for the two larger ones) were made at `ca2546c` on the same laptop
GPU. Each P9b, P9c and P9d run reads the P9a run of the same model, which is the one in the P9a rows below. The
selectivity columns of the P9d tables of the Qwen3 checkpoints were recomputed offline from the `item_slopes` of the
runs with the current code, not read from the stored `table_selectivity`; the GPT-2 tables are the stored ones, which
agree with a recomputation (every bound within 0.005). Earlier P9 runs exist, made with the four-relation label pool or
before the within-relation headline, the item resampling of the intervals and the dose-slope headline; they are
superseded and are not reported.

The Qwen3 R4, R6, A1, A2, M1, C2, C4, C5 and LR1 runs were made at `a6e2333` with a clean tree on the same laptop GPU,
with the default config of each experiment (batch size 8 for R4 and R6, 16 for M1, C2, C4, C5 and LR1). The A1 and A2
runs read the Qwen3 runs of R1, R2, R3, R5 and R7 below (their hashes are stored in each A1 and A2 summary). The C4 runs
read the LR1 adapter of their checkpoint and the C5 runs read the C2 run of their checkpoint, both named in their
configs. None of these Qwen3 M1, C2, C4 and C5 runs stores the perplexity or the distinct-1 of the generated text; the M1, C2, C4 and C5 runs of the same checkpoints reported in this document are the later ones made at `f4d414d` (below), and the earlier runs stay tracked.

The A1 and A2 runs of `gpt2-medium`, `gpt2-large` and `gpt2-xl`, the P2 runs of `gpt2-medium`, `qwen3-0.6b-base` and `qwen3-0.6b`, the P5 runs of the same three models and the P9b and P9c runs of `qwen3-0.6b` and `qwen3-1.7b` were made at `fb3b5dd` with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment (batch size 8 for P2, P5, P9b and P9c; the A1 and A2 configs store no batch size). The A1 and A2 runs read the R1, R2, R3, R5 and R7 runs of the same size listed below, and the P9b and P9c runs read the P9a run of the same checkpoint. The source files of these experiments are the same at `fb3b5dd` and at `f4d414d` (`git diff --stat fb3b5dd f4d414d -- src` lists only `chain.py` and the C2, C4, C5, M1 and P7 experiments). Against the commits of the committed runs they are compared with, the differences under `src` for these experiments are tooling (`chain.py`, `cloud.py`, `cli.py`, `export.py`), an opt-in refit flag in the shared fit code (default off, set by P4 and P5 only), and a constant moved in the P9 reading code. The P9d code and its abstention patterns also changed, which the P9b and P9c runs do not use.

The 22 runs of chain 14 were made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment on that commit except where noted below (`config.json` of each run): M1, C2 and P7 on `gpt2-large`, `gpt2-xl` and the Qwen3 1.7B pair, C4 on the Qwen3 1.7B pair, C5 on `gpt2-large`, `gpt2-xl` and the Qwen3 1.7B pair, and P2 and P5 on `gpt2-large` and `gpt2-xl`. The batch size is 16 for M1, C2, C4, C5 and P7 and 8 for P2 and P5, and the time between `started_at` and `finished_at` of the stored `meta.json` is 830 s (`c5/qwen3-1.7b/52057b2ddc30`) to 5,541 s (`p7/qwen3-1.7b/eacde3abd0aa`). The config hashes of the 22 runs were recomputed from the config defaults of `0353db7` (the commit this document is written against; two refactors of the C2, C4, C5, M1 and P7 modules sit between `f4d414d` and it) and all 22 agree with the stored ones. The config values that are not defaults are the C4 `lorra_run`, which names the LR1 run it reads, and the C5 `n_pool` (1,000) and `c2_run`, which names the C2 run it reads; the exported configs rewrite those path fields, so the hash of an exported C4 or C5 config differs from its stored hash. Against the earlier tracked runs of the same checkpoints, the configs of the Qwen3 1.7B C2, C4, C5 and P7 runs differ in `n_random` (absent earlier, 3 now) and, for C5, in the C2 run read; the Qwen3 M1 configs differ in `fluency` (off earlier, on now); and the `gpt2-xl` C5 config differs in `n_random` and in `c2_run` (none earlier). The P2 and P5 code is unchanged since `fb3b5dd`. Exporting these runs changed no tracked result.


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
| A1             | ad6abbba8fd0 | d1eeb0585ca6 | 3ed23ece4e63 | 078a72de1cfb |
| A2             | be4e34f21b35 | 6c8366625d7a | b6cf80cfb26f | 46d59dbefe95 |
| P1 (sentiment) | 673af340b77d | not run      | not run      | not run      |
| P1 (all tasks) | 078b9d6482a2 | not run      | not run      | not run      |
| P2             | ee099a704022 | 733bd2f76315 | 3ea9b00fe28f | feafacc97cbb |
| P3             | 676c31488397 | not run      | not run      | not run      |
| P4             | 8c4a73d1e9db | not run      | not run      | not run      |
| P5             | 80b4ae80a6ca | bd9f565baa76 | 430d434007c9 | 4171a611f734 |
| P7             | 6c19128c5192 | not run      | d03910535564 | eb67813e769d |
| C1             | add59c39f574 | 2ed2e53ba7fe | 353f91e1b8dc | de76af1a8a07 |
| M1             | 4e394ed1a72c | not run      | b4995274d1f8 | 3309d5fe55b7 |
| C2             | 85c9a670bf17 | not run      | 4741abe01da5 | aefba1c6274e |
| C4             | 1b571ac45420 | not run      | not run      | not run      |
| C5             | not run      | not run      | 36f392712827 | 0ad95833261d |
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
| R4          | 7a3cbcb17a19    | 48d56adb146e | 7f4457fe5392    | 1ce1c8182cf0 |
| R5          | e2d8b07cd376    | cffc23e593d0 | a5694ff54daa    | 487bd86230f5 |
| R6          | 74d5b69b5f57    | 66616910b862 | 713eccd9a4f3    | 98b2517a3cb1 |
| R7          | ce3e5ce2ec50    | 9cb5abfedb1b | 66d496ff00fa    | b7d78995c5fc |
| A1          | 4e8f8cfa0021    | 0cb04910086e | 769590eef456    | 2326d9ec9b7d |
| A2          | 109c4aa851bf    | 91b62e58953f | 09b991efaf7f    | 1e528ddfbffa |
| P1          | a540a2497ac4    | 31a6ee65d6ab | ef3f8c0d3389    | f4aa9b57824a |
| P2          | 59a53843dccd    | 442ebe80ca68 | 119f0ac54451    | 51a85fbec631 |
| P3          | not run         | not run      | 61baf30a8a59    | not run      |
| P5          | b6a702e460e9    | fb664c3b4be2 | a4bd463deeca    | 7e5e579d102e |
| P7          | not run         | not run      | 16a7a9e4facf    | eacde3abd0aa |
| C1          | 7ae3e2aa34e2    | 832462b7b04e | 0cd5a36d3408    | a91f950ad9f5 |
| M1          | not run         | not run      | e9279a87db07    | 77752112ddbe |
| C2          | not run         | not run      | 4133e19728fa    | 1b87ff98a318 |
| C4          | not run         | not run      | 7dd49d64a540    | 5ebc35ea3be6 |
| C5          | not run         | not run      | 3a14e656df3c    | 52057b2ddc30 |
| LR1         | not run         | not run      | 95a42d94146c    | bf86ed5ccde1 |
| P9a         | 1d9f881153e4    | 10d95352866e | d5d522bfecd4    | f413efcae268 |
| P9b         | 34c0ebff50b0    | 975dbcf95759 | 5ca7821318ef    | 9641ab8235a9 |
| P9c         | 264bf7ca3a01    | 0f8b118c9692 | 0f684b3c4b66    | 7c142334e5fe |
| P9d fewshot | f9a8f827b6b4    | 9c94025d9ad4 | 9543312bcc45    | 42b7a0daf9b9 |
| P9d concept | not run         | f0b3f4eb3988 | not run         | d43b9e297e42 |

Earlier runs that the chain 14 runs replace as the reported runs, and that stay tracked: C5 on `gpt2-xl` `155f0f72bc3f`; on `qwen3-1.7b-base` and `qwen3-1.7b`, in that order, M1 `3fe3a8644289` and `de2cdd3146ac`, C2 `37d187b75a7f` and `227d782df8ba`, C4 `9cdad6a8b7ce` and `2840839e453e`, C5 `fafc346bc266` and `672f70c5406a`, and P7 `947fef33a7c5` and `7921f3fe5754`.

Chain 17, reading, control and LoRRA runs (26 runs). R1, R2, R3, R4, R5 and R7 on `qwen3-4b-base` and `qwen3-4b`, R6
on `qwen3-4b`, A1 and A2 and LR1 on the two 4B checkpoints, C2 on `qwen3-4b`, LR1 on `gpt2-medium`, `gpt2-large` and
`gpt2-xl`, and C4 on the same three GPT-2 sizes were made at `89ea7cb` with a clean tree on a rented GPU pod (NVIDIA RTX
A6000 48 GB, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment on that commit
(`config.json` of each run). The batch size is 8 for R1 to R7, 16 for C2, C4 and LR1, and the A1 and A2 configs store
none. The time between `started_at` and `finished_at` of the stored `meta.json` is 1 s (`a1/qwen3-4b-base/80e2777e6b67`)
to 11,190 s (`r2/qwen3-4b/a6ab527e0544`). The chain log records exit status 0 for every one of the 26 runs. Since
`89ea7cb` the only changes under `src/tomography/experiments/` are the two P10 modules, so the code of these experiments
is that of main. The config hash of each of the 26 runs was recomputed under main (`86c319b`) by loading its stored
`config.json` into the config class of its experiment and hashing it; all 26 agree with the directory names except
the three C4 runs. For those the exported config rewrites the `lorra_run` path (to `results/lr1/<model>/<hash>`), so
the hash of the exported config differs from the stored one; the hash of the unmodified config of the pod run equals
the directory name. A1 and A2 read the R1, R2, R3, R5 and R7 runs of the
same checkpoint (the hashes are stored in each A1 and A2 summary and are those below), and each C4 run reads the LR1 run
of its size. Exporting these runs changed no tracked result. The runs that were cut or never started are not exported
and are listed under [Pending](#pending).

Config hashes of the chain 17 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base | qwen3-4b     |
| ---------- | ------------- | ------------ |
| R1         | 705ebc7745d4  | 15a1ed3eab2b |
| R2         | bb176283725a  | a6ab527e0544 |
| R3         | a6e29d000606  | 039d5c50bb27 |
| R4         | dd2c7d3f2fb2  | 3e8c537bfac8 |
| R5         | d692205ca9bf  | 1630de90cae8 |
| R6         | not exported  | 4e4a60094e47 |
| R7         | 15d7a0c1017f  | 9f28dde7631f |
| A1         | 80e2777e6b67  | 819f34624d6a |
| A2         | 8edf53991111  | 2a18226fa458 |
| C2         | not exported  | 1df7c1f9ff70 |
| LR1        | 22e472aa6c00  | f0d81c80e9bc |

The three larger GPT-2 sizes:

| Experiment | gpt2-medium  | gpt2-large   | gpt2-xl      |
| ---------- | ------------ | ------------ | ------------ |
| LR1        | c6bd21c82ddc | 5f6fdb0ccda5 | f54173153b29 |
| C4         | 88cc5d7f6dba | 43bf8077183e | 321f3da56ea5 |

Other runs: R6 text baselines of `gpt2` computed at `ca2546c` (not a run); P6 `25ee6de331d4`; P8 `e9f9bc04ba3c`.

The files of every run named here are committed under `results/<experiment>/<model>/<hash>/`, copied with
`tomography export`, which keeps the JSON, NPZ and CSV files the plots read and rewrites local paths. The figures are
not committed; `tomography plot all --out results` redraws them from these files.

## Pending

Not in this document yet. No dates are promised.

- A rerun of R6 on `gpt2` that writes the text baselines itself (the three larger GPT-2 runs store them).
- P1 on the truth and emotion tasks on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the Qwen3 checkpoints, and with the
  `pca_diff` finder (the truth and emotion tasks ran on `gpt2` only; the sentiment task ran with the logistic finder on
  `gpt2` and the four Qwen3 checkpoints).
- P3 on the other GPT-2 sizes and on the post-trained and 0.6B Qwen3 checkpoints (it ran on `gpt2` and
  `qwen3-1.7b-base`). P4 on the other GPT-2 sizes and on the Qwen3 checkpoints (it ran on `gpt2` only). P7 on
  `gpt2-medium` and on the two 0.6B Qwen3 checkpoints (it ran on `gpt2`, `gpt2-large`, `gpt2-xl` and the 1.7B pair). M1
  and C2 on `gpt2-medium` and the two 0.6B Qwen3 checkpoints (they ran on `gpt2`, `gpt2-large`, `gpt2-xl` and the 1.7B
  pair). C4 and LR1 on the two 0.6B Qwen3 checkpoints (LR1 ran on the four GPT-2 sizes and the 1.7B and 4B pairs, C4 on
  the four GPT-2 sizes and the 1.7B pair). C5 on `gpt2`, `gpt2-medium` and the two 0.6B Qwen3 checkpoints (it ran on `gpt2-large`, `gpt2-xl` and the 1.7B
  pair).
- The Qwen3 4B pair on a later pod: R6 and C2 on `qwen3-4b-base` (both were cut when chain 17 was stopped and are not
  exported), and C1, C4, C5, M1, P2, P5 and P7 on `qwen3-4b-base` and `qwen3-4b` (not started in chain 17).
- P5, persona wording and instruction confounds for honesty, with persona wordings beyond the five pairs of the run.
- A random-direction arm for C2 and P7 on `gpt2`, a fluency measure (perplexity and distinct-1) for M1, C2 and P7 on
  `gpt2`, and a random-direction arm and the perplexity of neutral text for C4 on `gpt2`. The runs on `gpt2-large` and
  `gpt2-xl` (M1, C2, P7, C5) and on the Qwen3 1.7B pair (M1, C2, P7, C4, C5) have them; the C4 scoring is not
  generation, so no C4 run has a distinct-1.
- Stored but not read in this document: the emotion direction matrix and embedding of the Qwen3 R4 runs, the A1 matrices
  of concept and layer agreement, the emotion gallery of the Qwen3 C5 runs, and the P5 curves over layers and the
  cosine curves between the paper reader and the other readers.
- P9: a hand check of the abstention indicator of P9d (the stored samples carry no human labels), an item-level
  interval for the P9b AUROC (only the interval over the pair draws is stored), and a within-relation increment of
  the probe over the output confidence in P9c (only the pooled increment is stored).
- A5, reading directions against Qwen-Scope SAE decoders.

