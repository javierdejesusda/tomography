# Findings

This document reports the results of this repository so far. It is a reference for the paper, in the style of
[methodology.md](methodology.md) and [related-work.md](related-work.md): it states what each run measured, the number it
produced, and what that number does and does not show. Experiment ids (`r1`, `p6`, `c1` and so on) are those of the
paper mapping in the methodology, which also describes the pipeline choices, the statistics and the run settings that
are not repeated here. Most numbers are read from the files of a finished run (`summary.json`, `table.json` or the
curves files). The derived ones are the P6 tables (recomputed from
the runs), the random-direction cosine of A1, the P9d selectivity columns (recomputed from the stored per-item
slopes), the span of the sizes in decades, the P1 spans, family ranges and transfer ranges (computed from the table
and matrix files of the P1 runs), the means and ranges over readers of the Qwen3 C1 tables (computed from the
per-reader rows), the per-layer dose slopes and judge agreements of the Qwen3 4B C1 pair (recomputed from the stored per-reader sweep curves), the maxima over layers of the random-direction series of R6, the P1 spans and cosine ranges of the
truth and emotion tasks of P1 and the margins, ranks and counts of the P1 follow-up on eight models
(computed from the table and matrix files), the C2 and P7 curve values (read from the curves
files), the P7 counts of inverted units by layer on the Qwen3 1.7B and 4B pairs (recomputed from the per-seed curves), the random and shuffled-label values at the selected layer of R4 on Qwen3, the maxima over layers of the R6
random series on Qwen3, the layer ranges of A1 and the A2 separations on Qwen3 and on the three larger GPT-2 sizes (all computed from the curves files), the
C2 values on Qwen3 (read from the curves and table files), the C4 reader means and ranges on Qwen3 and on the GPT-2 sizes (computed from the
table rows), the fluency means, ratios and counts of cells past the rule and the random-arm ranges
of the runs made at `f4d414d` and of the M1, C2, C5 and P7 runs of the 4B pair (computed from the per-cell rows and the curves files), the P7 counts of inverted
units on the GPT-2 sizes (recomputed from the per-seed curves), the P10b minima and maxima over layers and the counts of cells in P10a and P10b (computed from the table, hedge gap, forced in range, width, diagnostic, hand check, curves, transfer, baseline and people files), and differences or ranges quoted in the text. The P8 tables are read from the P8 run, which aggregates
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
14. [Approximate answers: range against exact quantities
(P10)](#approximate-answers-range-against-exact-quantities-p10) 15. [Qwen3: emotions, TruthfulQA, seed agreement, logit lens, steering and control (R4, R6, A1, A2, M1, C2, C4, C5,
LR1)](#qwen3-emotions-truthfulqa-seed-agreement-logit-lens-steering-and-control-r4-r6-a1-a2-m1-c2-c4-c5-lr1) 16. [Position
and stimulus-set size on qwen3-1.7b-base and qwen3-1.7b (P2,
P3)](#position-and-stimulus-set-size-on-qwen3-17b-base-and-qwen3-17b-p2-p3) 17. [Stimulus-set size and direction finder
across the GPT-2 sizes and Qwen3
(P3, P4)](#stimulus-set-size-and-direction-finder-across-the-gpt-2-sizes-and-qwen3-p3-p4) 18. [Persona wording and
instruction confounds (P5)](#persona-wording-and-instruction-confounds-p5) 19. [Template steering on the Qwen3 1.7B and 4B pairs
(P7)](#template-steering-on-the-qwen3-17b-and-4b-pairs-p7) 20. [Run identity](#run-identity) 21. [Pending](#pending)

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
  gap of 0.146 with non-overlapping intervals. The base value equals that of `qwen3-1.7b-base` (0.905) and the
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
  1.7B values (0.934 and 0.937) within 0.009.

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

- The four plain sets `cities`, `sp_en_trans`, `larger_than` and `smaller_than` round to 1.000 [1.000, 1.000] on
  both checkpoints. On the base checkpoint all 10 readers scored every pair correctly on all four. On `qwen3-4b`
  that holds for `sp_en_trans` and `smaller_than` only: `cities` is 0.999866 and `larger_than` 0.999899, one pair
  scored wrongly by one of the 10 readers in each (1 of 7,480 and 1 of 9,900 reader-pair scores). The negated and
  compound sets are high as well
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
run is the re-run of chain 40 at `bab5b75`, which stores them; the copy it replaced (made at `a39ae61`) did not, and its
other rows are those of the re-run. The text baselines do not depend on the model, so the four runs store the same
values.

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

R6 on the four GPT-2 sizes (runs `r6/gpt2/47e8ec0cf46c`, `r6/gpt2-medium/04fc7ead5f13`, `r6/gpt2-large/f058c0f19396` and
`r6/gpt2-xl/60cc952b4ec5`). The same 173 test questions, the same text baselines and the same layer rules in every
column; the interval of a LAT row is over the 15 readers and the interval of the other rows is a Wilson interval over
the 173 questions, so the two kinds are not comparable in width. Best-layer rows give the layer.

| Row                              | gpt2                     | gpt2-medium              | gpt2-large               | gpt2-xl                  |
| -------------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| standard                         | 0.168 [0.119, 0.230]     | 0.173 [0.124, 0.237]     | 0.179 [0.129, 0.243]     | 0.173 [0.124, 0.237]     |
| heuristic                        | 0.277 [0.216, 0.348]     | 0.266 [0.206, 0.336]     | 0.358 [0.291, 0.432]     | 0.289 [0.227, 0.361]     |
| text_word                        | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     |
| text_char                        | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     |
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

P1 reads the SST-2 sentiment concept through variants of the prompt template, on `gpt2` and the four Qwen3 checkpoints.
The tables of this section are for the sentiment task; `gpt2` also has a second run with the truth and emotion tasks,
reported at the end of the section, where the paper's finder (`pca_diff`) and the truth and emotion tasks on eight models follow. The paper template is `Consider the amount of sentiment in the following:`, the sentence, and `The amount of
sentiment is`. Every variant is fitted on the same 10 draws of 64 SST-2 training pairs as the paper variant (logistic
finder, last token), so a difference between variants belongs to the wording and not to the draw. All are scored on the
same 214 pairs of the SST-2 validation split (the report half; a disjoint half of 214 pairs is the select half). One
layer is used for every variant: the best layer of the paper reader on the select half, L8 on `gpt2` and L18 on the
Qwen3 checkpoints. A variant is not given a layer of its own in the tables below, unlike the template spread of
[P8](#base-against-post-trained-qwen3-p8). The variants:

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

The tables of this section are read from `p1/gpt2/673af340b77d` (`gpt2`), from the logistic runs with the sentiment,
truth and emotion tasks for the Qwen3 0.6B pair (`p1/qwen3-0.6b-base/deabd95764d0` and `p1/qwen3-0.6b/027be280838a`,
made at `bab5b75` on a pod) and from the sentiment-only runs for the Qwen3 1.7B pair (`ef3f8c0d3389` and `f4aa9b57824a`,
made at `a39ae61` on the laptop GPU); only the sentiment rows of the 0.6B runs are used here. The earlier sentiment-only
runs of the 0.6B pair (`a540a2497ac4` and `31a6ee65d6ab`, made at `a39ae61`) stay in the repository and gave report accuracies within 0.012 of these for every variant (at most 0.0084 on `qwen3-0.6b-base` and 0.0117 on `qwen3-0.6b`); the random series moved by up to 0.029 and 0.035, the interval bounds by up to 0.013, `own_report` by up to 0.013, the cosines by up to 0.009, and the selected own layer of 5 and 4 variants. The stored files do not show the cause of the difference.

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
| qwen3-0.6b-base | L18   | 0.952 [0.946, 0.957]          | 0.560  | 0.514    | 0.769       |
| qwen3-0.6b      | L18   | 0.860 [0.843, 0.874]          | 0.550  | 0.504    | 0.631       |
| qwen3-1.7b-base | L18   | 0.943 [0.936, 0.950]          | 0.608  | 0.512    | 0.797       |
| qwen3-1.7b      | L18   | 0.880 [0.866, 0.894]          | 0.553  | 0.500    | 0.654       |

The R1 values of the same models are 0.772, 0.947, 0.857, 0.905 and 0.866 (in the order of the table above). The two
runs score different pairs and select the layer differently (R1 on the pairs it reports), and this document does not
attribute the differences, which are largest for `qwen3-1.7b-base` (0.943 here against 0.905).

Lowest and highest variant among the 16, with the random series of that variant in parentheses, and their difference:

| Model           | Lowest variant: accuracy (random)  | Highest variant: accuracy (random) | Span  |
| --------------- | ---------------------------------- | ---------------------------------- | ----- |
| gpt2            | no_trailing_question 0.752 (0.507) | no_chat_tags 0.843 (0.524)         | 0.090 |
| qwen3-0.6b-base | concept_swap_0 0.765 (0.521)       | paraphrase_2 0.968 (0.593)         | 0.202 |
| qwen3-0.6b      | no_chat_tags 0.777 (0.498)         | neutral_frame 0.966 (0.600)        | 0.190 |
| qwen3-1.7b-base | concept_swap_1 0.839 (0.538)       | paraphrase_0 0.975 (0.669)         | 0.136 |
| qwen3-1.7b      | concept_swap_2 0.781 (0.566)       | paraphrase_2 0.964 (0.586)         | 0.183 |

- **Spread over variants.** The span is 0.090 on `gpt2` and 0.202, 0.190, 0.136 and 0.183 on `qwen3-0.6b-base`,
  `qwen3-0.6b`, `qwen3-1.7b-base` and `qwen3-1.7b`. The paper template is the lowest or the highest variant on none of
  the five models. A span takes the two extremes of 16 cells, so it is larger than the difference between two variants
  picked in advance; the interval of a single variant is between 0.004 and 0.070 wide (upper minus lower bound) over
  the 80 cells. On `gpt2`, all variants except `bare` (0.802) and `no_chat_tags` (0.843) lie between 0.752 and 0.782,
  around the paper value of 0.765.
- **Variants against the paper variant.** Counting the 15 other variants whose interval lies entirely above (below)
  the interval of the paper variant: 2 (0) on `gpt2`, 2 (10) on `qwen3-0.6b-base`, 5 (5) on `qwen3-0.6b`, 3 (6) on
  `qwen3-1.7b-base` and 8 (4) on `qwen3-1.7b`.
- **Random and shuffled series.** The shuffled series is between 0.480 and 0.511 on the 16 cells of `gpt2`, between 0.472 and 0.521 on the 32 cells of the two base checkpoints and between 0.482 and 0.519 on the 32 cells of the two post-trained ones. The random series is not at 0.5 on the Qwen3 checkpoints: it is between 0.493 and 0.674 on the base ones and between 0.495 and 0.649 on the post-trained ones (0.479 to 0.555 on `gpt2`), highest on `qwen3-1.7b-base` with `paraphrase_1` (0.674). The reader is above the random series of the same cell by at least
  0.204 on every one of the 80 cells (the smallest margin is `gpt2` with `no_template`, 0.753 against 0.549), so no
  variant here reads at the level of a random direction.

The `bare` variant, the sentence with nothing around it:

| Model           | Bare accuracy | Random | Shuffled | Cosine with paper | Bare seed cosine |
| --------------- | ------------- | ------ | -------- | ----------------- | ---------------- |
| gpt2            | 0.802         | 0.555  | 0.497    | 0.179             | 0.298            |
| qwen3-0.6b-base | 0.850         | 0.529  | 0.494    | 0.110             | 0.347            |
| qwen3-0.6b      | 0.874         | 0.501  | 0.497    | 0.075             | 0.332            |
| qwen3-1.7b-base | 0.908         | 0.518  | 0.492    | 0.137             | 0.414            |
| qwen3-1.7b      | 0.920         | 0.556  | 0.493    | 0.121             | 0.409            |

The paper reader and the bare reader scored on each other's items:

| Model           | Paper reader on bare items | Bare reader on paper items |
| --------------- | -------------------------- | -------------------------- |
| gpt2            | 0.807                      | 0.686                      |
| qwen3-0.6b-base | 0.850                      | 0.766                      |
| qwen3-0.6b      | 0.768                      | 0.640                      |
| qwen3-1.7b-base | 0.906                      | 0.857                      |
| qwen3-1.7b      | 0.886                      | 0.708                      |

- **Bare accuracy.** The bare reader is at 0.802 on `gpt2` and 0.850, 0.874, 0.908 and 0.920 on the Qwen3 checkpoints, with random series between 0.501 and 0.556. Its interval is above the paper interval on `gpt2` and `qwen3-1.7b`,
  below it on the two base checkpoints, and overlaps it on `qwen3-0.6b` (0.874 against 0.860).
- **Bare direction.** The bare reader's direction has a cosine with the paper reader's of 0.075 to 0.179 on the five
  models, against seed cosines of 0.520 to 0.797 for the paper reader and 0.298 to 0.414 for the bare reader. The two
  readers are close to orthogonal in the residual stream, although both read above the random series. The paper
  reader, scored on bare items, reaches 0.768 to 0.906, and the bare reader, scored on paper items, 0.640 to 0.857;
  the first is above the second on all five models. The random series of the cross-scored cells is not among the
  committed files.

The concept-swap and wording variants, as ranges over the variants of the kind, with the random series and the cosine
with the paper reader over the same variants:

| Model           | Variants         | Accuracy       | Random         | Cosine with paper |
| --------------- | ---------------- | -------------- | -------------- | ----------------- |
| gpt2            | concept_swap (4) | 0.762 to 0.778 | 0.479 to 0.500 | 0.963 to 0.976    |
| gpt2            | paraphrase (3)   | 0.762 to 0.778 | 0.503 to 0.524 | 0.870 to 0.960    |
| gpt2            | scaffold (3)     | 0.765 to 0.782 | 0.506 to 0.522 | 0.762 to 0.776    |
| qwen3-0.6b-base | concept_swap (4) | 0.765 to 0.789 | 0.496 to 0.532 | 0.364 to 0.504    |
| qwen3-0.6b-base | paraphrase (3)   | 0.943 to 0.968 | 0.556 to 0.593 | 0.783 to 0.867    |
| qwen3-0.6b-base | scaffold (3)     | 0.771 to 0.930 | 0.493 to 0.587 | 0.449 to 0.613    |
| qwen3-0.6b      | concept_swap (4) | 0.804 to 0.838 | 0.503 to 0.525 | 0.700 to 0.741    |
| qwen3-0.6b      | paraphrase (3)   | 0.792 to 0.947 | 0.495 to 0.596 | 0.477 to 0.616    |
| qwen3-0.6b      | scaffold (3)     | 0.823 to 0.896 | 0.537 to 0.565 | 0.514 to 0.578    |
| qwen3-1.7b-base | concept_swap (4) | 0.839 to 0.898 | 0.529 to 0.556 | 0.548 to 0.635    |
| qwen3-1.7b-base | paraphrase (3)   | 0.951 to 0.975 | 0.643 to 0.674 | 0.743 to 0.893    |
| qwen3-1.7b-base | scaffold (3)     | 0.908 to 0.943 | 0.556 to 0.586 | 0.563 to 0.606    |
| qwen3-1.7b      | concept_swap (4) | 0.781 to 0.848 | 0.546 to 0.577 | 0.593 to 0.630    |
| qwen3-1.7b      | paraphrase (3)   | 0.922 to 0.964 | 0.585 to 0.587 | 0.629 to 0.775    |
| qwen3-1.7b      | scaffold (3)     | 0.894 to 0.913 | 0.557 to 0.567 | 0.496 to 0.570    |

- **Concept swaps.** On `gpt2` the four concept swaps read at 0.762 to 0.778 (paper 0.765) and their cosine with the
  paper reader is 0.963 to 0.976. On the Qwen3 checkpoints the point estimate of every concept swap is below the paper
  accuracy (0.765 to 0.789 against 0.952 on `qwen3-0.6b-base`, 0.804 to 0.838 against 0.860 on `qwen3-0.6b`, 0.839 to
  0.898 against 0.943 on `qwen3-1.7b-base`, 0.781 to 0.848 against 0.880 on `qwen3-1.7b`); the interval is entirely
  below the paper interval for 4, 2, 4 and 3 of the 4 swaps in that order. Their cosines with the paper reader are 0.364 to 0.741.
- **Wording.** Paraphrases read at 0.762 to 0.778 on `gpt2` and between 0.792 and 0.975 on the Qwen3 checkpoints;
  scaffolds read at 0.765 to 0.782 on `gpt2` and between 0.771 and 0.943. The widest wording spreads are the three paraphrases of `qwen3-0.6b` (0.792 to 0.947) and the three scaffolds of `qwen3-0.6b-base` (0.771 to 0.930).

Transfer between variants. The paper reader scored on the items of each of the 15 other variants, the reader of each
variant scored on the paper items, and the random series of the 16 variants on their own items:

| Model           | Paper reader on variant items | Variant reader on paper items | Random, 16 variants |
| --------------- | ----------------------------- | ----------------------------- | ------------------- |
| gpt2            | 0.750 to 0.811                | 0.686 to 0.770                | 0.479 to 0.555      |
| qwen3-0.6b-base | 0.721 to 0.969                | 0.766 to 0.960                | 0.493 to 0.593      |
| qwen3-0.6b      | 0.762 to 0.942                | 0.640 to 0.874                | 0.495 to 0.649      |
| qwen3-1.7b-base | 0.889 to 0.975                | 0.857 to 0.956                | 0.518 to 0.674      |
| qwen3-1.7b      | 0.762 to 0.943                | 0.708 to 0.890                | 0.509 to 0.620      |

The lowest and the highest cell of the 16 by 16 transfer matrix of the 16 variants every model has, off the diagonal
(the stored matrix is 17 by 17 on `gpt2`, `qwen3-0.6b-base` and `qwen3-1.7b-base`, and the cells of `no_trailing_space`
are left out; reader of the first variant scored on the items of the second):

| Model           | Lowest cell: trained on, scored on | Highest cell: trained on, scored on        |
| --------------- | ---------------------------------- | ------------------------------------------ |
| gpt2            | 0.654: bare, scaffold_2            | 0.869: no_chat_tags, bare                  |
| qwen3-0.6b-base | 0.563: bare, concept_swap_3        | 0.970: paraphrase_1, paraphrase_2          |
| qwen3-0.6b      | 0.568: bare, no_chat_tags          | 0.951: no_trailing_question, neutral_frame |
| qwen3-1.7b-base | 0.700: bare, concept_swap_2        | 0.975: paraphrase_1, paraphrase_0          |
| qwen3-1.7b      | 0.591: bare, concept_swap_3        | 0.959: no_trailing_question, paraphrase_2  |

- **Range.** The paper reader scored on other variants spans 0.750 to 0.811 on `gpt2`, 0.721 to 0.969 on `qwen3-0.6b-base`, 0.762 to 0.942 on `qwen3-0.6b`, 0.889 to 0.975 on `qwen3-1.7b-base` and 0.762 to 0.943 on
  `qwen3-1.7b`. The reader of each variant scored on the paper items spans 0.686 to 0.770, 0.766 to 0.960, 0.640 to 0.874, 0.857 to 0.956 and 0.708 to 0.890.
- **Lowest cell.** On all five models the lowest off-diagonal cell is a reader fitted on `bare`. The cell is no higher
  than the highest random value of the run on `qwen3-0.6b-base` (0.563 against 0.593), `qwen3-0.6b` (0.568 against 0.649) and `qwen3-1.7b` (0.591 against 0.620), and it is higher on `gpt2` (0.654 against 0.555) and
  `qwen3-1.7b-base` (0.700 against 0.674). The random series of the cross-scored cells is not among the committed
  files, so this compares the cell with the random series on the diagonal items.

Every variant of every model, as report accuracy with the random and the shuffled series in parentheses (`not run`
marks a kind that was skipped):

| Variant              | gpt2                 | qwen3-0.6b-base      | qwen3-0.6b           |
| -------------------- | -------------------- | -------------------- | -------------------- |
| paper                | 0.765 (0.497, 0.505) | 0.952 (0.560, 0.514) | 0.860 (0.550, 0.504) |
| no_template          | 0.753 (0.549, 0.497) | 0.865 (0.582, 0.492) | 0.935 (0.649, 0.510) |
| bare                 | 0.802 (0.555, 0.497) | 0.850 (0.529, 0.494) | 0.874 (0.501, 0.497) |
| no_chat_tags         | 0.843 (0.524, 0.480) | 0.953 (0.572, 0.490) | 0.777 (0.498, 0.486) |
| neutral_frame        | 0.766 (0.534, 0.484) | 0.862 (0.562, 0.498) | 0.966 (0.600, 0.486) |
| no_trailing_question | 0.752 (0.507, 0.487) | 0.942 (0.545, 0.489) | 0.907 (0.575, 0.500) |
| no_trailing_space    | 0.771 (0.512, 0.486) | 0.726 (0.499, 0.488) | not run              |
| paraphrase_0         | 0.762 (0.503, 0.493) | 0.963 (0.556, 0.500) | 0.792 (0.513, 0.508) |
| paraphrase_1         | 0.765 (0.521, 0.504) | 0.943 (0.566, 0.516) | 0.836 (0.495, 0.516) |
| paraphrase_2         | 0.778 (0.524, 0.504) | 0.968 (0.593, 0.511) | 0.947 (0.596, 0.516) |
| scaffold_0           | 0.782 (0.506, 0.510) | 0.930 (0.587, 0.521) | 0.896 (0.537, 0.504) |
| scaffold_1           | 0.765 (0.509, 0.488) | 0.771 (0.527, 0.472) | 0.861 (0.565, 0.489) |
| scaffold_2           | 0.780 (0.522, 0.508) | 0.856 (0.493, 0.506) | 0.823 (0.563, 0.500) |
| concept_swap_0       | 0.778 (0.494, 0.501) | 0.765 (0.521, 0.511) | 0.835 (0.507, 0.505) |
| concept_swap_1       | 0.762 (0.500, 0.508) | 0.789 (0.496, 0.493) | 0.804 (0.523, 0.512) |
| concept_swap_2       | 0.771 (0.486, 0.511) | 0.775 (0.532, 0.519) | 0.814 (0.525, 0.510) |
| concept_swap_3       | 0.778 (0.479, 0.502) | 0.765 (0.499, 0.493) | 0.838 (0.503, 0.492) |

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
`bare`). On the Qwen3 checkpoints the accuracy moves by 0.136 to 0.202 across the variants and the paper template is
neither the best nor the worst variant of any model. Each variant changes one aspect of the prompt, but the table does
not say which aspect drives a difference: every cell is one run, the intervals cover the training draw only, and the
214 evaluation pairs are shared by all variants. The near-orthogonality of the bare and paper directions shows that
readers fitted under different prompts differ in direction; it does not show which of them tracks the concept better.
Base and post-trained checkpoints are different models, so the pairs are compared in P8 and the two are not pooled
here.

P1 with the truth and emotion tasks on `gpt2` (run `p1/gpt2/078b9d6482a2`, made after the sentiment run
`p1/gpt2/673af340b77d` with the same settings and the task list as the only difference in the config; its sentiment rows
are identical to the ones above at the printed precision; this run is the re-run of chain 40 at `bab5b75`, which
replaced the copy made at `ca2546c`, see [Run identity](#run-identity)). The tasks are the truth statements of the
Geometry of Truth split and the six emotions of R4, each read against the others. Every variant is read at the one layer
of the paper reader on the select half, and the report half has the number of pairs in parentheses (59 for truth and 36
to 53 for the emotions). Lowest and highest variant over all the variants of the task, including `framing_swap` for the
emotions:

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
  variants is 0.568 (`no_trailing_question`), so the smallest margin of a variant over its random series is 0.061. The
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

### The paper's finder and the truth and emotion tasks on eight models (P1)

The tables above use the logistic finder. The paper reads with the first principal component of the differences between
pairs of training stimuli (`pca_diff`, centered as in the reference code; see the [methodology](methodology.md)). This
subsection adds that finder, and the truth and emotion tasks, for eight models: `gpt2`, `gpt2-medium`, `gpt2-large`,
`gpt2-xl`, `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base` and `qwen3-1.7b`. Sixteen runs, a logistic and a
`pca_diff` run per model, were made with the P1 defaults of their commit (sentiment, truth and the six emotions, 10
draws of 64 training pairs, up to 256 evaluation pairs per split, batch size 8, last token). The four runs of the Qwen3
1.7B pair were made at `f4d414d` on the laptop GPU (`p1/qwen3-1.7b-base/62073cecf6d5` and `a45dc5c93f18`,
`p1/qwen3-1.7b/f3a2d832c9c4` and `7ffb64cce683`, logistic and `pca_diff` in that order). The twelve runs of the other
six models were made at `80baffb` (the two `gpt2-large` runs) and `bab5b75` (the ten others) on rented GPU pods (see
[Run identity](#run-identity)); a run and its partner differ in the finder only. The `gpt2` logistic run
`p1/gpt2/078b9d6482a2` is a re-run of a run of the same hash made earlier at `ca2546c`: it reproduces the earlier copy
in every accuracy and interval of the table file, so only the random series (by at most 0.0017) and the cosines (by at
most 4e-6) moved. The two Qwen3 1.7B logistic runs differ from the sentiment-only runs of the tables above
(`ef3f8c0d3389` and `f4aa9b57824a`) in the task list only, and they are a replication of those runs: none of the cells
of the 17 (base) and 16 (post-trained) sentiment rows of the table file differs, so their logistic rows below are the
rows of the tables above. The logistic runs of the Qwen3 0.6B pair (`p1/qwen3-0.6b-base/deabd95764d0` and
`p1/qwen3-0.6b/027be280838a`) are the runs that the tables above read for these models, so their logistic rows below are
the rows of the tables above. The rules of the section hold: every variant is read at one layer, the best layer of the
paper reader of the run on the select half, so in a `pca_diff` run it is the layer of the `pca_diff` reader, which is
not the layer of the logistic reader (the layers are in the first table). The same kinds were skipped as above
(`no_trailing_space` on the post-trained Qwen3 checkpoints, and `alternate` and `framing_swap` for sentiment). Intervals
are as above: 95 percent bootstrap over the 10 readers, not including the sampling of the 214 evaluation pairs. The
models are separate rows and no metric is pooled over base and post-trained checkpoints. In the paragraphs and bullets
below, "the base checkpoint" and "the post-trained checkpoint" are `qwen3-1.7b-base` and `qwen3-1.7b`; the other six
models have a bullet of their own.

Sentiment, the paper variant:

| Model           | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Seed cosine |
| --------------- | -------- | ----- | ----------------------------- | ------ | -------- | ----------- |
| gpt2            | logistic | L8    | 0.765 [0.744, 0.780]          | 0.497  | 0.505    | 0.520       |
| gpt2            | pca_diff | L11   | 0.512 [0.497, 0.526]          | 0.534  | 0.493    | -0.100      |
| gpt2-medium     | logistic | L20   | 0.742 [0.724, 0.758]          | 0.539  | 0.501    | 0.443       |
| gpt2-medium     | pca_diff | L12   | 0.517 [0.504, 0.529]          | 0.486  | 0.484    | -0.034      |
| gpt2-large      | logistic | L32   | 0.807 [0.792, 0.824]          | 0.505  | 0.501    | 0.504       |
| gpt2-large      | pca_diff | L18   | 0.495 [0.485, 0.505]          | 0.566  | 0.503    | -0.058      |
| gpt2-xl         | logistic | L43   | 0.776 [0.756, 0.794]          | 0.523  | 0.494    | 0.459       |
| gpt2-xl         | pca_diff | L45   | 0.507 [0.493, 0.521]          | 0.558  | 0.503    | -0.055      |
| qwen3-0.6b-base | logistic | L18   | 0.952 [0.946, 0.957]          | 0.560  | 0.514    | 0.769       |
| qwen3-0.6b-base | pca_diff | L20   | 0.665 [0.612, 0.718]          | 0.643  | 0.503    | 0.085       |
| qwen3-0.6b      | logistic | L18   | 0.860 [0.843, 0.874]          | 0.550  | 0.504    | 0.631       |
| qwen3-0.6b      | pca_diff | L10   | 0.564 [0.540, 0.589]          | 0.545  | 0.481    | 0.303       |
| qwen3-1.7b-base | logistic | L18   | 0.943 [0.936, 0.950]          | 0.608  | 0.512    | 0.797       |
| qwen3-1.7b-base | pca_diff | L19   | 0.846 [0.833, 0.857]          | 0.614  | 0.507    | 0.882       |
| qwen3-1.7b      | logistic | L18   | 0.880 [0.866, 0.894]          | 0.553  | 0.500    | 0.654       |
| qwen3-1.7b      | pca_diff | L20   | 0.534 [0.503, 0.562]          | 0.550  | 0.513    | 0.070       |

The `pca_diff` reader of the paper variant is below the logistic reader on both checkpoints, with disjoint intervals:
0.846 [0.833, 0.857] against 0.943 [0.936, 0.950] on the base checkpoint and 0.534 [0.503, 0.562] against 0.880 [0.866,
0.894] on the post-trained one. On the post-trained checkpoint it is not above a random direction: the mean of the
random series (0.550) lies inside the interval of the reader, the mean of the shuffled series is 0.513, and the lower
bound of the interval of the reader is 0.003 above 0.5. Its seed cosine is 0.070: the directions of two draws are close
to orthogonal on average, against 0.654 for the logistic readers. On the base checkpoint the mean of the random series
(0.614) is below the interval of the reader, and the seed cosine is 0.882, above the 0.797 of the logistic readers.

Lowest and highest variant among the 16, with the random series of that variant in parentheses, and their difference:

| Model           | Finder   | Lowest variant: accuracy (random)  | Highest variant: accuracy (random) | Span  |
| --------------- | -------- | ---------------------------------- | ---------------------------------- | ----- |
| gpt2            | logistic | no_trailing_question 0.752 (0.507) | no_chat_tags 0.843 (0.524)         | 0.090 |
| gpt2            | pca_diff | concept_swap_1 0.502 (0.534)       | no_template 0.631 (0.556)          | 0.129 |
| gpt2-medium     | logistic | paraphrase_0 0.729 (0.567)         | no_chat_tags 0.888 (0.495)         | 0.160 |
| gpt2-medium     | pca_diff | concept_swap_2 0.494 (0.508)       | scaffold_1 0.551 (0.528)           | 0.057 |
| gpt2-large      | logistic | concept_swap_1 0.771 (0.518)       | no_chat_tags 0.914 (0.516)         | 0.143 |
| gpt2-large      | pca_diff | scaffold_2 0.487 (0.579)           | no_template 0.657 (0.610)          | 0.170 |
| gpt2-xl         | logistic | paraphrase_0 0.689 (0.501)         | no_template 0.892 (0.550)          | 0.203 |
| gpt2-xl         | pca_diff | concept_swap_3 0.498 (0.543)       | no_template 0.685 (0.607)          | 0.187 |
| qwen3-0.6b-base | logistic | concept_swap_0 0.765 (0.521)       | paraphrase_2 0.968 (0.593)         | 0.202 |
| qwen3-0.6b-base | pca_diff | neutral_frame 0.500 (0.574)        | paraphrase_2 0.970 (0.770)         | 0.469 |
| qwen3-0.6b      | logistic | no_chat_tags 0.777 (0.498)         | neutral_frame 0.966 (0.600)        | 0.190 |
| qwen3-0.6b      | pca_diff | no_chat_tags 0.487 (0.522)         | scaffold_0 0.665 (0.579)           | 0.178 |
| qwen3-1.7b-base | logistic | concept_swap_1 0.839 (0.538)       | paraphrase_0 0.975 (0.669)         | 0.136 |
| qwen3-1.7b-base | pca_diff | neutral_frame 0.509 (0.564)        | paraphrase_2 0.969 (0.698)         | 0.459 |
| qwen3-1.7b      | logistic | concept_swap_2 0.781 (0.566)       | paraphrase_2 0.964 (0.586)         | 0.183 |
| qwen3-1.7b      | pca_diff | concept_swap_0 0.498 (0.533)       | no_template 0.597 (0.599)          | 0.100 |

- **Spread over variants.** With `pca_diff` the span is 0.459 on the base checkpoint (0.136 with logistic) and 0.100 on
  the post-trained one (0.183 with logistic). On the base checkpoint the three paraphrases (`paraphrase_0` to
  `paraphrase_2`) read at 0.958, 0.929, 0.969 and their intervals lie entirely above the paper interval (0.833 to
  0.857); the other 12 variants of the 16 read at 0.509 to 0.607 and lie entirely below it. On the post-trained
  checkpoint all 16 variants lie between 0.498 and 0.597, and none has an interval entirely above or below the paper
  interval. The paper template is the lowest or the highest variant in neither `pca_diff` run. The interval of a single
  `pca_diff` variant is between 0.004 and 0.168 wide on the base checkpoint and between 0.026 and 0.100 on the
  post-trained one, against 0.004 to 0.048 and 0.008 to 0.064 with logistic.

- **Variants against the paper variant.** Counting the 15 other variants whose interval lies entirely above (below) the
  interval of the paper variant: 3 (12) with `pca_diff` and 3 (6) with logistic on the base checkpoint, and 0 (0) and 8
  (4) on the post-trained checkpoint.

- **Random and shuffled series.** The shuffled series ranges from 0.479 to 0.521 on the 32 cells of `gpt2`, from 0.469
  to 0.518 on the 32 cells of `gpt2-medium`, from 0.473 to 0.530 on the 32 cells of `gpt2-large`, from 0.479 to 0.521 on
  the 32 cells of `gpt2-xl`, from 0.472 to 0.521 on the 32 cells of `qwen3-0.6b-base`, from 0.481 to 0.517 on the 32
  cells of `qwen3-0.6b`, from 0.476 to 0.515 on the 32 cells of `qwen3-1.7b-base` and from 0.482 to 0.519 on the 32
  cells of `qwen3-1.7b` (16 variants, both finders). With logistic the reader is above the random series of its cell on
  all 16 variants, by at least 0.274 on the base checkpoint and 0.214 on the post-trained one. With `pca_diff` it is
  not. On the base checkpoint the reader is at or below the random series in 8 of 16 cells (the smallest margin is
  -0.055, `neutral_frame`) and its interval lies entirely above the random value in 4 cells (`paper`, `paraphrase_0`,
  `paraphrase_1`, `paraphrase_2`) and entirely below it in 5. On the post-trained checkpoint the reader is at or below
  the random series in 14 of 16 cells (smallest margin -0.084, `paraphrase_2`), its interval lies entirely below the
  random value in 10 cells and entirely above it in none, and the highest variant, `no_template` (0.597), has a random
  series of 0.599. The random value of a variant has no interval in the committed files, so these comparisons put a
  reader against one number and are not tests.

The `bare` variant, the sentence with nothing around it:

| Model           | Finder   | Bare accuracy | Random | Shuffled | Cosine with paper | Bare seed cosine |
| --------------- | -------- | ------------- | ------ | -------- | ----------------- | ---------------- |
| gpt2            | logistic | 0.802         | 0.555  | 0.497    | 0.179             | 0.298            |
| gpt2            | pca_diff | 0.523         | 0.518  | 0.481    | 0.130             | 0.059            |
| gpt2-medium     | logistic | 0.857         | 0.486  | 0.501    | 0.279             | 0.401            |
| gpt2-medium     | pca_diff | 0.517         | 0.509  | 0.494    | 0.036             | -0.102           |
| gpt2-large      | logistic | 0.885         | 0.504  | 0.500    | 0.274             | 0.416            |
| gpt2-large      | pca_diff | 0.501         | 0.572  | 0.513    | 0.043             | 0.257            |
| gpt2-xl         | logistic | 0.889         | 0.473  | 0.495    | 0.276             | 0.442            |
| gpt2-xl         | pca_diff | 0.519         | 0.548  | 0.485    | -0.026            | 0.066            |
| qwen3-0.6b-base | logistic | 0.850         | 0.529  | 0.494    | 0.110             | 0.347            |
| qwen3-0.6b-base | pca_diff | 0.510         | 0.548  | 0.499    | -0.002            | -0.105           |
| qwen3-0.6b      | logistic | 0.874         | 0.501  | 0.497    | 0.075             | 0.332            |
| qwen3-0.6b      | pca_diff | 0.505         | 0.517  | 0.507    | -0.003            | -0.065           |
| qwen3-1.7b-base | logistic | 0.908         | 0.518  | 0.492    | 0.137             | 0.414            |
| qwen3-1.7b-base | pca_diff | 0.514         | 0.515  | 0.504    | 0.008             | -0.104           |
| qwen3-1.7b      | logistic | 0.920         | 0.556  | 0.493    | 0.121             | 0.409            |
| qwen3-1.7b      | pca_diff | 0.538         | 0.525  | 0.500    | 0.000             | -0.058           |

The paper reader and the bare reader scored on each other's items:

| Model           | Finder   | Paper reader on bare items | Bare reader on paper items |
| --------------- | -------- | -------------------------- | -------------------------- |
| gpt2            | logistic | 0.807                      | 0.686                      |
| gpt2            | pca_diff | 0.515                      | 0.507                      |
| gpt2-medium     | logistic | 0.828                      | 0.743                      |
| gpt2-medium     | pca_diff | 0.514                      | 0.501                      |
| gpt2-large      | logistic | 0.855                      | 0.779                      |
| gpt2-large      | pca_diff | 0.513                      | 0.521                      |
| gpt2-xl         | logistic | 0.736                      | 0.774                      |
| gpt2-xl         | pca_diff | 0.490                      | 0.489                      |
| qwen3-0.6b-base | logistic | 0.850                      | 0.766                      |
| qwen3-0.6b-base | pca_diff | 0.667                      | 0.510                      |
| qwen3-0.6b      | logistic | 0.768                      | 0.640                      |
| qwen3-0.6b      | pca_diff | 0.546                      | 0.488                      |
| qwen3-1.7b-base | logistic | 0.906                      | 0.857                      |
| qwen3-1.7b-base | pca_diff | 0.898                      | 0.510                      |
| qwen3-1.7b      | logistic | 0.886                      | 0.708                      |
| qwen3-1.7b      | pca_diff | 0.574                      | 0.541                      |

- **Bare accuracy.** The bare `pca_diff` reader is at 0.514 on the base checkpoint (random 0.515) and 0.538 on the
  post-trained one (random 0.525), where the bare logistic readers read at 0.908 and 0.920. Its direction has a cosine
  with the `pca_diff` paper reader of 0.008 and 0.000, and the cosine between the bare directions of two draws is -0.104
  and -0.058.

- **Cross-scoring.** The `pca_diff` paper reader scored on bare items reaches 0.898 on the base checkpoint (random 0.515; logistic 0.906, random 0.518) and 0.574 on the post-trained one (random 0.525; logistic 0.886, random 0.556); the bare `pca_diff` reader scored on paper items reaches 0.510 and 0.541. On the base checkpoint the direction fitted under the paper template reads the bare items at a level
  close to that of the logistic paper reader, while the direction fitted on the bare items is at the level of the random
  series.

The concept-swap and wording variants, as ranges over the variants of the kind, with the random series and the cosine
with the paper reader over the same variants:

| Model           | Finder   | Variants         | Accuracy       | Random         | Cosine with paper |
| --------------- | -------- | ---------------- | -------------- | -------------- | ----------------- |
| gpt2            | logistic | concept_swap (4) | 0.762 to 0.778 | 0.479 to 0.500 | 0.963 to 0.976    |
| gpt2            | logistic | paraphrase (3)   | 0.762 to 0.778 | 0.503 to 0.524 | 0.870 to 0.960    |
| gpt2            | logistic | scaffold (3)     | 0.765 to 0.782 | 0.506 to 0.522 | 0.762 to 0.776    |
| gpt2            | pca_diff | concept_swap (4) | 0.502 to 0.515 | 0.512 to 0.534 | 0.524 to 0.920    |
| gpt2            | pca_diff | paraphrase (3)   | 0.504 to 0.527 | 0.486 to 0.534 | 0.600 to 0.752    |
| gpt2            | pca_diff | scaffold (3)     | 0.526 to 0.539 | 0.525 to 0.564 | 0.315 to 0.607    |
| gpt2-medium     | logistic | concept_swap (4) | 0.735 to 0.754 | 0.520 to 0.540 | 0.919 to 0.930    |
| gpt2-medium     | logistic | paraphrase (3)   | 0.729 to 0.790 | 0.567 to 0.577 | 0.822 to 0.875    |
| gpt2-medium     | logistic | scaffold (3)     | 0.800 to 0.863 | 0.574 to 0.603 | 0.698 to 0.730    |
| gpt2-medium     | pca_diff | concept_swap (4) | 0.494 to 0.526 | 0.503 to 0.524 | 0.380 to 0.684    |
| gpt2-medium     | pca_diff | paraphrase (3)   | 0.500 to 0.531 | 0.487 to 0.546 | 0.591 to 0.666    |
| gpt2-medium     | pca_diff | scaffold (3)     | 0.547 to 0.551 | 0.528 to 0.564 | 0.262 to 0.295    |
| gpt2-large      | logistic | concept_swap (4) | 0.771 to 0.794 | 0.511 to 0.523 | 0.866 to 0.917    |
| gpt2-large      | logistic | paraphrase (3)   | 0.794 to 0.845 | 0.512 to 0.524 | 0.812 to 0.911    |
| gpt2-large      | logistic | scaffold (3)     | 0.853 to 0.886 | 0.503 to 0.517 | 0.718 to 0.720    |
| gpt2-large      | pca_diff | concept_swap (4) | 0.500 to 0.520 | 0.532 to 0.571 | 0.379 to 0.704    |
| gpt2-large      | pca_diff | paraphrase (3)   | 0.495 to 0.507 | 0.498 to 0.544 | 0.563 to 0.824    |
| gpt2-large      | pca_diff | scaffold (3)     | 0.487 to 0.507 | 0.527 to 0.579 | 0.434 to 0.599    |
| gpt2-xl         | logistic | concept_swap (4) | 0.738 to 0.783 | 0.490 to 0.516 | 0.846 to 0.888    |
| gpt2-xl         | logistic | paraphrase (3)   | 0.689 to 0.831 | 0.500 to 0.512 | 0.607 to 0.821    |
| gpt2-xl         | logistic | scaffold (3)     | 0.836 to 0.874 | 0.496 to 0.563 | 0.606 to 0.662    |
| gpt2-xl         | pca_diff | concept_swap (4) | 0.498 to 0.527 | 0.493 to 0.554 | 0.051 to 0.282    |
| gpt2-xl         | pca_diff | paraphrase (3)   | 0.500 to 0.524 | 0.500 to 0.556 | 0.267 to 0.869    |
| gpt2-xl         | pca_diff | scaffold (3)     | 0.525 to 0.598 | 0.551 to 0.582 | -0.020 to 0.235   |
| qwen3-0.6b-base | logistic | concept_swap (4) | 0.765 to 0.789 | 0.496 to 0.532 | 0.364 to 0.504    |
| qwen3-0.6b-base | logistic | paraphrase (3)   | 0.943 to 0.968 | 0.556 to 0.593 | 0.783 to 0.867    |
| qwen3-0.6b-base | logistic | scaffold (3)     | 0.771 to 0.930 | 0.493 to 0.587 | 0.449 to 0.613    |
| qwen3-0.6b-base | pca_diff | concept_swap (4) | 0.503 to 0.518 | 0.531 to 0.570 | 0.198 to 0.410    |
| qwen3-0.6b-base | pca_diff | paraphrase (3)   | 0.768 to 0.970 | 0.658 to 0.770 | 0.334 to 0.504    |
| qwen3-0.6b-base | pca_diff | scaffold (3)     | 0.529 to 0.542 | 0.572 to 0.638 | 0.045 to 0.226    |
| qwen3-0.6b      | logistic | concept_swap (4) | 0.804 to 0.838 | 0.503 to 0.525 | 0.700 to 0.741    |
| qwen3-0.6b      | logistic | paraphrase (3)   | 0.792 to 0.947 | 0.495 to 0.596 | 0.477 to 0.616    |
| qwen3-0.6b      | logistic | scaffold (3)     | 0.823 to 0.896 | 0.537 to 0.565 | 0.514 to 0.578    |
| qwen3-0.6b      | pca_diff | concept_swap (4) | 0.518 to 0.534 | 0.529 to 0.550 | 0.399 to 0.540    |
| qwen3-0.6b      | pca_diff | paraphrase (3)   | 0.497 to 0.500 | 0.524 to 0.544 | 0.236 to 0.345    |
| qwen3-0.6b      | pca_diff | scaffold (3)     | 0.570 to 0.665 | 0.515 to 0.579 | 0.310 to 0.408    |
| qwen3-1.7b-base | logistic | concept_swap (4) | 0.839 to 0.898 | 0.529 to 0.556 | 0.548 to 0.635    |
| qwen3-1.7b-base | logistic | paraphrase (3)   | 0.951 to 0.975 | 0.643 to 0.674 | 0.743 to 0.893    |
| qwen3-1.7b-base | logistic | scaffold (3)     | 0.908 to 0.943 | 0.556 to 0.586 | 0.563 to 0.606    |
| qwen3-1.7b-base | pca_diff | concept_swap (4) | 0.516 to 0.567 | 0.543 to 0.591 | 0.092 to 0.132    |
| qwen3-1.7b-base | pca_diff | paraphrase (3)   | 0.929 to 0.969 | 0.643 to 0.733 | 0.704 to 0.924    |
| qwen3-1.7b-base | pca_diff | scaffold (3)     | 0.525 to 0.607 | 0.563 to 0.657 | 0.042 to 0.210    |
| qwen3-1.7b      | logistic | concept_swap (4) | 0.781 to 0.848 | 0.546 to 0.577 | 0.593 to 0.630    |
| qwen3-1.7b      | logistic | paraphrase (3)   | 0.922 to 0.964 | 0.585 to 0.587 | 0.629 to 0.775    |
| qwen3-1.7b      | logistic | scaffold (3)     | 0.894 to 0.913 | 0.557 to 0.567 | 0.496 to 0.570    |
| qwen3-1.7b      | pca_diff | concept_swap (4) | 0.498 to 0.517 | 0.493 to 0.533 | -0.176 to 0.337   |
| qwen3-1.7b      | pca_diff | paraphrase (3)   | 0.536 to 0.546 | 0.545 to 0.629 | 0.121 to 0.756    |
| qwen3-1.7b      | pca_diff | scaffold (3)     | 0.528 to 0.550 | 0.554 to 0.605 | -0.113 to 0.154   |

- **Concept swaps and scaffolds.** With `pca_diff` the four concept swaps read at 0.516 to 0.567 on the base checkpoint
  and 0.498 to 0.517 on the post-trained one (logistic: 0.839 to 0.898 and 0.781 to 0.848), and the three scaffolds at
  0.525 to 0.607 and 0.528 to 0.550 (logistic: 0.908 to 0.943 and 0.894 to 0.913). On the base checkpoint these seven
  variants read between 0.516 and 0.607, below the paper accuracy (0.846), and their random series are 0.543 to 0.657.

- **Paraphrases.** On the base checkpoint the three paraphrases read at 0.929 to 0.969 with `pca_diff` (logistic 0.951
  to 0.975), with a seed cosine of 0.952 to 0.977 and a cosine with the paper reader of 0.704 to 0.924. On the
  post-trained checkpoint they read at 0.536 to 0.546 (logistic 0.922 to 0.964), against random series of 0.545 to
  0.629.

Transfer between variants. The paper reader scored on the items of each of the 15 other variants, the reader of each
variant scored on the paper items, the random series of the 16 variants on their own items, and the lowest and the
highest cell of the whole transfer matrix off the diagonal (16 by 16, and 17 by 17 where `no_trailing_space` was run, so
these two columns can include it while the first three do not; reader of the first variant scored on the items of the
second):

| Model           | Finder   | Paper reader on variant items | Variant reader on paper items | Random, 16 variants | Lowest cell: trained on, scored on       | Highest cell: trained on, scored on        |
| --------------- | -------- | ----------------------------- | ----------------------------- | ------------------- | ---------------------------------------- | ------------------------------------------ |
| gpt2            | logistic | 0.750 to 0.811                | 0.686 to 0.770                | 0.479 to 0.555      | 0.654: bare, scaffold_2                  | 0.869: no_chat_tags, bare                  |
| gpt2            | pca_diff | 0.510 to 0.519                | 0.503 to 0.585                | 0.486 to 0.564      | 0.486: concept_swap_2, no_chat_tags      | 0.628: no_template, neutral_frame          |
| gpt2-medium     | logistic | 0.717 to 0.830                | 0.736 to 0.787                | 0.486 to 0.603      | 0.708: bare, no_trailing_question        | 0.910: no_chat_tags, bare                  |
| gpt2-medium     | pca_diff | 0.493 to 0.519                | 0.498 to 0.540                | 0.486 to 0.564      | 0.448: no_trailing_space, scaffold_0     | 0.580: scaffold_2, scaffold_1              |
| gpt2-large      | logistic | 0.779 to 0.914                | 0.766 to 0.842                | 0.503 to 0.558      | 0.722: bare, neutral_frame               | 0.921: no_chat_tags, bare                  |
| gpt2-large      | pca_diff | 0.469 to 0.517                | 0.494 to 0.568                | 0.498 to 0.610      | 0.465: no_trailing_space, no_template    | 0.643: no_template, no_trailing_question   |
| gpt2-xl         | logistic | 0.730 to 0.829                | 0.742 to 0.810                | 0.473 to 0.563      | 0.605: concept_swap_2, no_trailing_space | 0.883: no_trailing_question, no_template   |
| gpt2-xl         | pca_diff | 0.490 to 0.522                | 0.489 to 0.564                | 0.493 to 0.607      | 0.437: concept_swap_1, scaffold_2        | 0.629: scaffold_1, scaffold_0              |
| qwen3-0.6b-base | logistic | 0.721 to 0.969                | 0.766 to 0.960                | 0.493 to 0.593      | 0.481: bare, no_trailing_space           | 0.970: paraphrase_1, paraphrase_2          |
| qwen3-0.6b-base | pca_diff | 0.563 to 0.841                | 0.497 to 0.940                | 0.531 to 0.770      | 0.415: scaffold_2, paraphrase_2          | 0.970: paraphrase_0, paraphrase_2          |
| qwen3-0.6b      | logistic | 0.762 to 0.942                | 0.640 to 0.874                | 0.495 to 0.649      | 0.568: bare, no_chat_tags                | 0.951: no_trailing_question, neutral_frame |
| qwen3-0.6b      | pca_diff | 0.534 to 0.624                | 0.488 to 0.624                | 0.510 to 0.579      | 0.477: no_chat_tags, scaffold_1          | 0.646: scaffold_0, scaffold_2              |
| qwen3-1.7b-base | logistic | 0.889 to 0.975                | 0.857 to 0.956                | 0.518 to 0.674      | 0.700: bare, concept_swap_2              | 0.975: paraphrase_1, paraphrase_0          |
| qwen3-1.7b-base | pca_diff | 0.748 to 0.963                | 0.510 to 0.929                | 0.515 to 0.733      | 0.480: neutral_frame, scaffold_1         | 0.976: paraphrase_0, no_trailing_question  |
| qwen3-1.7b      | logistic | 0.762 to 0.943                | 0.708 to 0.890                | 0.509 to 0.620      | 0.591: bare, concept_swap_3              | 0.959: no_trailing_question, paraphrase_2  |
| qwen3-1.7b      | pca_diff | 0.485 to 0.577                | 0.465 to 0.547                | 0.493 to 0.629      | 0.464: concept_swap_3, bare              | 0.623: no_template, paraphrase_0           |

- **Range.** With `pca_diff` the paper reader scored on other variants spans 0.748 to 0.963 on the base checkpoint and
  0.485 to 0.577 on the post-trained one, and the reader of each variant scored on the paper items spans 0.510 to 0.929
  and 0.465 to 0.547. On the base checkpoint the lowest cell of the matrix is below 0.5 (0.480: `neutral_frame` scored
  on `scaffold_1`), and on the post-trained checkpoint the 240 off-diagonal cells lie between 0.464 and 0.623.

The other six models of this subsection, sentiment, with the same quantities for every model. Counts and margins are
over the 16 variants every model has (`no_trailing_space` is left out of them):

- **`gpt2`.** The paper variant reads at 0.765 [0.744, 0.780] with logistic (L8, random 0.497, seed cosine 0.520) and at
  0.512 [0.497, 0.526] with `pca_diff` (L11, random 0.534, seed cosine -0.100); the random series of the `pca_diff` run
  lies above the interval of the reader. The span over the 16 variants is 0.090 with logistic (`no_trailing_question`
  0.752 to `no_chat_tags` 0.843) and 0.129 with `pca_diff` (`concept_swap_1` 0.502 to `no_template` 0.631), and the
  paper template is the lowest or the highest variant in neither run. The counts of the other variants whose interval
  lies entirely above (below) the paper interval are 2 (0) with logistic and 1 (0) with `pca_diff`. The logistic reader
  is above the random series of its cell on 16 of 16 variants (smallest margin 0.204, `no_template`); the `pca_diff`
  reader on 6 of 16 (smallest margin -0.042, `no_chat_tags`), with its interval entirely above the random value in 2
  cells and entirely below it in 6. The bare variant reads at 0.802 with logistic (random 0.555) and 0.523 with
  `pca_diff` (random 0.518); the paper reader scored on bare items reaches 0.807 and 0.515, and the bare reader scored
  on paper items 0.686 and 0.507. The paper reader scored on the other variants spans 0.750 to 0.811 with logistic and
  0.510 to 0.519 with `pca_diff`, and the off-diagonal cells of the whole transfer matrix lie between 0.654 and 0.869
  with logistic and between 0.486 and 0.628 with `pca_diff`. The Spearman rank correlation between the 16 accuracies
  under the two finders is -0.077.

- **`gpt2-medium`.** The paper variant reads at 0.742 [0.724, 0.758] with logistic (L20, random 0.539, seed cosine
  0.443) and at 0.517 [0.504, 0.529] with `pca_diff` (L12, random 0.486, seed cosine -0.034); the random series of the
  `pca_diff` run lies below the interval of the reader. The span over the 16 variants is 0.160 with logistic
  (`paraphrase_0` 0.729 to `no_chat_tags` 0.888) and 0.057 with `pca_diff` (`concept_swap_2` 0.494 to `scaffold_1`
  0.551), and the paper template is the lowest or the highest variant in neither run. The counts of the other variants
  whose interval lies entirely above (below) the paper interval are 8 (0) with logistic and 0 (0) with `pca_diff`. The
  logistic reader is above the random series of its cell on 16 of 16 variants (smallest margin 0.162, `paraphrase_0`);
  the `pca_diff` reader on 11 of 16 (smallest margin -0.047, `paraphrase_1`), with its interval entirely above the
  random value in 5 cells and entirely below it in 3. The bare variant reads at 0.857 with logistic (random 0.486) and
  0.517 with `pca_diff` (random 0.509); the paper reader scored on bare items reaches 0.828 and 0.514, and the bare
  reader scored on paper items 0.743 and 0.501. The paper reader scored on the other variants spans 0.717 to 0.830 with
  logistic and 0.493 to 0.519 with `pca_diff`, and the off-diagonal cells of the whole transfer matrix lie between 0.708
  and 0.910 with logistic and between 0.448 and 0.580 with `pca_diff`. The Spearman rank correlation between the 16
  accuracies under the two finders is 0.632.

- **`gpt2-large`.** The paper variant reads at 0.807 [0.792, 0.824] with logistic (L32, random 0.505, seed cosine 0.504)
  and at 0.495 [0.485, 0.505] with `pca_diff` (L18, random 0.566, seed cosine -0.058); the random series of the
  `pca_diff` run lies above the interval of the reader. The span over the 16 variants is 0.143 with logistic
  (`concept_swap_1` 0.771 to `no_chat_tags` 0.914) and 0.170 with `pca_diff` (`scaffold_2` 0.487 to `no_template`
  0.657), and the paper template is the lowest or the highest variant in neither run. The counts of the other variants
  whose interval lies entirely above (below) the paper interval are 8 (0) with logistic and 2 (0) with `pca_diff`. The
  logistic reader is above the random series of its cell on 16 of 16 variants (smallest margin 0.252, `concept_swap_1`);
  the `pca_diff` reader on 3 of 16 (smallest margin -0.092, `scaffold_2`), with its interval entirely above the random
  value in 0 cells and entirely below it in 12. The bare variant reads at 0.885 with logistic (random 0.504) and 0.501
  with `pca_diff` (random 0.572); the paper reader scored on bare items reaches 0.855 and 0.513, and the bare reader
  scored on paper items 0.779 and 0.521. The paper reader scored on the other variants spans 0.779 to 0.914 with
  logistic and 0.469 to 0.517 with `pca_diff`, and the off-diagonal cells of the whole transfer matrix lie between 0.722
  and 0.921 with logistic and between 0.465 and 0.643 with `pca_diff`. The Spearman rank correlation between the 16
  accuracies under the two finders is -0.059.

- **`gpt2-xl`.** The paper variant reads at 0.776 [0.756, 0.794] with logistic (L43, random 0.523, seed cosine 0.459)
  and at 0.507 [0.493, 0.521] with `pca_diff` (L45, random 0.558, seed cosine -0.055); the random series of the
  `pca_diff` run lies above the interval of the reader. The span over the 16 variants is 0.203 with logistic
  (`paraphrase_0` 0.689 to `no_template` 0.892) and 0.187 with `pca_diff` (`concept_swap_3` 0.498 to `no_template`
  0.685), and the paper template is the lowest or the highest variant in neither run. The counts of the other variants
  whose interval lies entirely above (below) the paper interval are 9 (1) with logistic and 2 (0) with `pca_diff`. The
  logistic reader is above the random series of its cell on 16 of 16 variants (smallest margin 0.187, `paraphrase_0`);
  the `pca_diff` reader on 7 of 16 (smallest margin -0.050, `paper`), with its interval entirely above the random value
  in 2 cells and entirely below it in 6. The bare variant reads at 0.889 with logistic (random 0.473) and 0.519 with
  `pca_diff` (random 0.548); the paper reader scored on bare items reaches 0.736 and 0.490, and the bare reader scored
  on paper items 0.774 and 0.489. The paper reader scored on the other variants spans 0.730 to 0.829 with logistic and
  0.490 to 0.522 with `pca_diff`, and the off-diagonal cells of the whole transfer matrix lie between 0.605 and 0.883
  with logistic and between 0.437 and 0.629 with `pca_diff`. The Spearman rank correlation between the 16 accuracies
  under the two finders is 0.488.

- **`qwen3-0.6b-base`.** The paper variant reads at 0.952 [0.946, 0.957] with logistic (L18, random 0.560, seed cosine
  0.769) and at 0.665 [0.612, 0.718] with `pca_diff` (L20, random 0.643, seed cosine 0.085); the random series of the
  `pca_diff` run lies inside the interval of the reader. The span over the 16 variants is 0.202 with logistic
  (`concept_swap_0` 0.765 to `paraphrase_2` 0.968) and 0.469 with `pca_diff` (`neutral_frame` 0.500 to `paraphrase_2`
  0.970), and the paper template is the lowest or the highest variant in neither run. The counts of the other variants
  whose interval lies entirely above (below) the paper interval are 2 (10) with logistic and 2 (11) with `pca_diff`. The
  logistic reader is above the random series of its cell on 16 of 16 variants (smallest margin 0.243, `concept_swap_2`);
  the `pca_diff` reader on 5 of 16 (smallest margin -0.109, `scaffold_0`), with its interval entirely above the random
  value in 4 cells and entirely below it in 9. The bare variant reads at 0.850 with logistic (random 0.529) and 0.510
  with `pca_diff` (random 0.548); the paper reader scored on bare items reaches 0.850 and 0.667, and the bare reader
  scored on paper items 0.766 and 0.510. The paper reader scored on the other variants spans 0.721 to 0.969 with
  logistic and 0.563 to 0.841 with `pca_diff`, and the off-diagonal cells of the whole transfer matrix lie between 0.481
  and 0.970 with logistic and between 0.415 and 0.970 with `pca_diff`. The Spearman rank correlation between the 16
  accuracies under the two finders is 0.739.

- **`qwen3-0.6b`.** The paper variant reads at 0.860 [0.843, 0.874] with logistic (L18, random 0.550, seed cosine 0.631)
  and at 0.564 [0.540, 0.589] with `pca_diff` (L10, random 0.545, seed cosine 0.303); the random series of the
  `pca_diff` run lies inside the interval of the reader. The span over the 16 variants is 0.190 with logistic
  (`no_chat_tags` 0.777 to `neutral_frame` 0.966) and 0.178 with `pca_diff` (`no_chat_tags` 0.487 to `scaffold_0`
  0.665), and the paper template is the lowest or the highest variant in neither run. The counts of the other variants
  whose interval lies entirely above (below) the paper interval are 5 (5) with logistic and 1 (9) with `pca_diff`. The
  logistic reader is above the random series of its cell on 16 of 16 variants (smallest margin 0.260, `scaffold_2`); the
  `pca_diff` reader on 4 of 16 (smallest margin -0.048, `no_template`), with its interval entirely above the random
  value in 2 cells and entirely below it in 7. The bare variant reads at 0.874 with logistic (random 0.501) and 0.505
  with `pca_diff` (random 0.517); the paper reader scored on bare items reaches 0.768 and 0.546, and the bare reader
  scored on paper items 0.640 and 0.488. The paper reader scored on the other variants spans 0.762 to 0.942 with
  logistic and 0.534 to 0.624 with `pca_diff`, and the off-diagonal cells of the whole transfer matrix lie between 0.568
  and 0.951 with logistic and between 0.477 and 0.646 with `pca_diff`. The Spearman rank correlation between the 16
  accuracies under the two finders is 0.115.

Every variant of the eight models and both finders, as report accuracy with its interval and the random and the shuffled
series in parentheses (`not run` marks a kind that was skipped):

`gpt2`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.765 [0.744, 0.780] (0.497, 0.505)                  | 0.512 [0.497, 0.526] (0.534, 0.493)                  |
| no_template          | 0.753 [0.744, 0.762] (0.549, 0.497)                  | 0.631 [0.619, 0.644] (0.556, 0.505)                  |
| bare                 | 0.802 [0.781, 0.823] (0.555, 0.497)                  | 0.523 [0.491, 0.558] (0.518, 0.481)                  |
| no_chat_tags         | 0.843 [0.833, 0.852] (0.524, 0.480)                  | 0.509 [0.499, 0.521] (0.551, 0.485)                  |
| neutral_frame        | 0.766 [0.749, 0.782] (0.534, 0.484)                  | 0.525 [0.482, 0.565] (0.499, 0.514)                  |
| no_trailing_question | 0.752 [0.739, 0.764] (0.507, 0.487)                  | 0.533 [0.495, 0.566] (0.537, 0.479)                  |
| no_trailing_space    | 0.771 [0.755, 0.787] (0.512, 0.486)                  | 0.524 [0.500, 0.547] (0.507, 0.507)                  |
| paraphrase_0         | 0.762 [0.744, 0.774] (0.503, 0.493)                  | 0.504 [0.484, 0.524] (0.502, 0.513)                  |
| paraphrase_1         | 0.765 [0.746, 0.780] (0.521, 0.504)                  | 0.527 [0.500, 0.556] (0.486, 0.521)                  |
| paraphrase_2         | 0.778 [0.757, 0.797] (0.524, 0.504)                  | 0.510 [0.479, 0.542] (0.534, 0.494)                  |
| scaffold_0           | 0.782 [0.767, 0.797] (0.506, 0.510)                  | 0.532 [0.517, 0.546] (0.559, 0.502)                  |
| scaffold_1           | 0.765 [0.753, 0.777] (0.509, 0.488)                  | 0.526 [0.506, 0.546] (0.564, 0.501)                  |
| scaffold_2           | 0.780 [0.764, 0.797] (0.522, 0.508)                  | 0.539 [0.520, 0.558] (0.525, 0.507)                  |
| concept_swap_0       | 0.778 [0.762, 0.790] (0.494, 0.501)                  | 0.511 [0.499, 0.521] (0.527, 0.517)                  |
| concept_swap_1       | 0.762 [0.750, 0.774] (0.500, 0.508)                  | 0.502 [0.482, 0.525] (0.534, 0.486)                  |
| concept_swap_2       | 0.771 [0.758, 0.783] (0.486, 0.511)                  | 0.507 [0.493, 0.521] (0.512, 0.506)                  |
| concept_swap_3       | 0.778 [0.763, 0.789] (0.479, 0.502)                  | 0.515 [0.502, 0.529] (0.521, 0.500)                  |

`gpt2-medium`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.742 [0.724, 0.758] (0.539, 0.501)                  | 0.517 [0.504, 0.529] (0.486, 0.484)                  |
| no_template          | 0.776 [0.760, 0.791] (0.536, 0.502)                  | 0.504 [0.441, 0.563] (0.527, 0.497)                  |
| bare                 | 0.857 [0.843, 0.871] (0.486, 0.501)                  | 0.517 [0.483, 0.550] (0.509, 0.494)                  |
| no_chat_tags         | 0.888 [0.879, 0.898] (0.495, 0.489)                  | 0.518 [0.497, 0.537] (0.507, 0.469)                  |
| neutral_frame        | 0.791 [0.777, 0.806] (0.537, 0.492)                  | 0.521 [0.502, 0.542] (0.518, 0.515)                  |
| no_trailing_question | 0.756 [0.742, 0.770] (0.517, 0.510)                  | 0.526 [0.512, 0.540] (0.507, 0.505)                  |
| no_trailing_space    | 0.773 [0.749, 0.796] (0.500, 0.477)                  | 0.521 [0.513, 0.529] (0.492, 0.476)                  |
| paraphrase_0         | 0.729 [0.710, 0.745] (0.567, 0.501)                  | 0.516 [0.501, 0.529] (0.487, 0.505)                  |
| paraphrase_1         | 0.746 [0.731, 0.761] (0.577, 0.513)                  | 0.500 [0.471, 0.527] (0.546, 0.511)                  |
| paraphrase_2         | 0.790 [0.776, 0.807] (0.576, 0.509)                  | 0.531 [0.511, 0.550] (0.507, 0.500)                  |
| scaffold_0           | 0.863 [0.853, 0.874] (0.603, 0.514)                  | 0.550 [0.501, 0.594] (0.564, 0.492)                  |
| scaffold_1           | 0.833 [0.815, 0.849] (0.575, 0.487)                  | 0.551 [0.499, 0.594] (0.528, 0.489)                  |
| scaffold_2           | 0.800 [0.786, 0.816] (0.574, 0.505)                  | 0.547 [0.529, 0.563] (0.533, 0.513)                  |
| concept_swap_0       | 0.752 [0.741, 0.765] (0.540, 0.504)                  | 0.503 [0.495, 0.512] (0.524, 0.494)                  |
| concept_swap_1       | 0.750 [0.736, 0.762] (0.533, 0.488)                  | 0.526 [0.517, 0.536] (0.507, 0.488)                  |
| concept_swap_2       | 0.735 [0.721, 0.751] (0.520, 0.518)                  | 0.494 [0.483, 0.507] (0.508, 0.490)                  |
| concept_swap_3       | 0.754 [0.743, 0.765] (0.535, 0.489)                  | 0.507 [0.493, 0.521] (0.503, 0.500)                  |

`gpt2-large`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.807 [0.792, 0.824] (0.505, 0.501)                  | 0.495 [0.485, 0.505] (0.566, 0.503)                  |
| no_template          | 0.869 [0.857, 0.879] (0.558, 0.506)                  | 0.657 [0.571, 0.709] (0.610, 0.500)                  |
| bare                 | 0.885 [0.865, 0.902] (0.504, 0.500)                  | 0.501 [0.475, 0.531] (0.572, 0.513)                  |
| no_chat_tags         | 0.914 [0.906, 0.922] (0.516, 0.488)                  | 0.502 [0.482, 0.522] (0.538, 0.473)                  |
| neutral_frame        | 0.816 [0.801, 0.830] (0.537, 0.497)                  | 0.594 [0.505, 0.664] (0.583, 0.502)                  |
| no_trailing_question | 0.836 [0.825, 0.846] (0.529, 0.506)                  | 0.520 [0.450, 0.589] (0.554, 0.502)                  |
| no_trailing_space    | 0.850 [0.842, 0.859] (0.486, 0.503)                  | 0.502 [0.498, 0.507] (0.537, 0.477)                  |
| paraphrase_0         | 0.794 [0.773, 0.812] (0.518, 0.504)                  | 0.495 [0.486, 0.503] (0.524, 0.530)                  |
| paraphrase_1         | 0.824 [0.803, 0.843] (0.512, 0.511)                  | 0.505 [0.490, 0.521] (0.498, 0.523)                  |
| paraphrase_2         | 0.845 [0.832, 0.858] (0.524, 0.503)                  | 0.507 [0.491, 0.522] (0.544, 0.504)                  |
| scaffold_0           | 0.886 [0.874, 0.898] (0.514, 0.515)                  | 0.507 [0.501, 0.514] (0.527, 0.510)                  |
| scaffold_1           | 0.853 [0.843, 0.862] (0.517, 0.477)                  | 0.491 [0.478, 0.502] (0.558, 0.502)                  |
| scaffold_2           | 0.856 [0.842, 0.868] (0.503, 0.506)                  | 0.487 [0.476, 0.499] (0.579, 0.514)                  |
| concept_swap_0       | 0.787 [0.765, 0.807] (0.523, 0.497)                  | 0.500 [0.482, 0.516] (0.571, 0.503)                  |
| concept_swap_1       | 0.771 [0.749, 0.793] (0.518, 0.492)                  | 0.506 [0.495, 0.516] (0.532, 0.514)                  |
| concept_swap_2       | 0.794 [0.777, 0.808] (0.511, 0.508)                  | 0.512 [0.501, 0.526] (0.541, 0.505)                  |
| concept_swap_3       | 0.779 [0.757, 0.799] (0.516, 0.493)                  | 0.520 [0.510, 0.530] (0.565, 0.519)                  |

`gpt2-xl`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.776 [0.756, 0.794] (0.523, 0.494)                  | 0.507 [0.493, 0.521] (0.558, 0.503)                  |
| no_template          | 0.892 [0.886, 0.899] (0.550, 0.513)                  | 0.685 [0.594, 0.743] (0.607, 0.499)                  |
| bare                 | 0.889 [0.874, 0.907] (0.473, 0.495)                  | 0.519 [0.493, 0.539] (0.548, 0.485)                  |
| no_chat_tags         | 0.878 [0.841, 0.908] (0.550, 0.493)                  | 0.502 [0.481, 0.524] (0.515, 0.493)                  |
| neutral_frame        | 0.850 [0.831, 0.867] (0.536, 0.492)                  | 0.573 [0.520, 0.624] (0.518, 0.514)                  |
| no_trailing_question | 0.861 [0.849, 0.875] (0.555, 0.504)                  | 0.576 [0.516, 0.626] (0.563, 0.514)                  |
| no_trailing_space    | 0.788 [0.763, 0.809] (0.525, 0.499)                  | 0.517 [0.492, 0.540] (0.526, 0.501)                  |
| paraphrase_0         | 0.689 [0.661, 0.717] (0.501, 0.479)                  | 0.500 [0.481, 0.520] (0.500, 0.499)                  |
| paraphrase_1         | 0.763 [0.739, 0.785] (0.512, 0.513)                  | 0.509 [0.488, 0.532] (0.556, 0.489)                  |
| paraphrase_2         | 0.831 [0.816, 0.848] (0.500, 0.507)                  | 0.524 [0.503, 0.546] (0.537, 0.500)                  |
| scaffold_0           | 0.874 [0.863, 0.884] (0.496, 0.511)                  | 0.528 [0.488, 0.575] (0.554, 0.515)                  |
| scaffold_1           | 0.852 [0.845, 0.859] (0.563, 0.492)                  | 0.598 [0.560, 0.635] (0.582, 0.507)                  |
| scaffold_2           | 0.836 [0.816, 0.850] (0.528, 0.507)                  | 0.525 [0.507, 0.541] (0.551, 0.515)                  |
| concept_swap_0       | 0.772 [0.750, 0.793] (0.490, 0.518)                  | 0.521 [0.500, 0.541] (0.554, 0.506)                  |
| concept_swap_1       | 0.738 [0.709, 0.764] (0.510, 0.505)                  | 0.515 [0.511, 0.519] (0.493, 0.521)                  |
| concept_swap_2       | 0.754 [0.737, 0.771] (0.497, 0.518)                  | 0.527 [0.509, 0.542] (0.516, 0.507)                  |
| concept_swap_3       | 0.783 [0.766, 0.801] (0.516, 0.501)                  | 0.498 [0.490, 0.505] (0.543, 0.500)                  |

`qwen3-0.6b-base`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.952 [0.946, 0.957] (0.560, 0.514)                  | 0.665 [0.612, 0.718] (0.643, 0.503)                  |
| no_template          | 0.865 [0.846, 0.883] (0.582, 0.492)                  | 0.532 [0.504, 0.557] (0.570, 0.498)                  |
| bare                 | 0.850 [0.824, 0.875] (0.529, 0.494)                  | 0.510 [0.473, 0.552] (0.548, 0.499)                  |
| no_chat_tags         | 0.953 [0.948, 0.958] (0.572, 0.490)                  | 0.718 [0.658, 0.771] (0.618, 0.484)                  |
| neutral_frame        | 0.862 [0.836, 0.885] (0.562, 0.498)                  | 0.500 [0.474, 0.529] (0.574, 0.502)                  |
| no_trailing_question | 0.942 [0.930, 0.952] (0.545, 0.489)                  | 0.529 [0.507, 0.550] (0.600, 0.492)                  |
| no_trailing_space    | 0.726 [0.704, 0.744] (0.499, 0.488)                  | 0.495 [0.483, 0.508] (0.536, 0.516)                  |
| paraphrase_0         | 0.963 [0.959, 0.967] (0.556, 0.500)                  | 0.912 [0.883, 0.935] (0.750, 0.496)                  |
| paraphrase_1         | 0.943 [0.935, 0.950] (0.566, 0.516)                  | 0.768 [0.692, 0.843] (0.658, 0.506)                  |
| paraphrase_2         | 0.968 [0.966, 0.970] (0.593, 0.511)                  | 0.970 [0.964, 0.975] (0.770, 0.514)                  |
| scaffold_0           | 0.930 [0.918, 0.942] (0.587, 0.521)                  | 0.529 [0.495, 0.559] (0.638, 0.509)                  |
| scaffold_1           | 0.771 [0.749, 0.790] (0.527, 0.472)                  | 0.533 [0.512, 0.556] (0.601, 0.483)                  |
| scaffold_2           | 0.856 [0.843, 0.868] (0.493, 0.506)                  | 0.542 [0.526, 0.556] (0.572, 0.515)                  |
| concept_swap_0       | 0.765 [0.747, 0.782] (0.521, 0.511)                  | 0.518 [0.495, 0.540] (0.531, 0.505)                  |
| concept_swap_1       | 0.789 [0.766, 0.813] (0.496, 0.493)                  | 0.516 [0.497, 0.535] (0.551, 0.479)                  |
| concept_swap_2       | 0.775 [0.761, 0.791] (0.532, 0.519)                  | 0.503 [0.484, 0.525] (0.570, 0.506)                  |
| concept_swap_3       | 0.765 [0.748, 0.782] (0.499, 0.493)                  | 0.512 [0.504, 0.521] (0.556, 0.492)                  |

`qwen3-0.6b`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.860 [0.843, 0.874] (0.550, 0.504)                  | 0.564 [0.540, 0.589] (0.545, 0.481)                  |
| no_template          | 0.935 [0.930, 0.939] (0.649, 0.510)                  | 0.508 [0.500, 0.516] (0.556, 0.490)                  |
| bare                 | 0.874 [0.854, 0.893] (0.501, 0.497)                  | 0.505 [0.475, 0.532] (0.517, 0.507)                  |
| no_chat_tags         | 0.777 [0.762, 0.792] (0.498, 0.486)                  | 0.487 [0.471, 0.506] (0.522, 0.500)                  |
| neutral_frame        | 0.966 [0.958, 0.973] (0.600, 0.486)                  | 0.506 [0.496, 0.515] (0.510, 0.499)                  |
| no_trailing_question | 0.907 [0.893, 0.921] (0.575, 0.500)                  | 0.539 [0.486, 0.579] (0.559, 0.505)                  |
| no_trailing_space    | not run                                              | not run                                              |
| paraphrase_0         | 0.792 [0.754, 0.823] (0.513, 0.508)                  | 0.500 [0.494, 0.507] (0.524, 0.504)                  |
| paraphrase_1         | 0.836 [0.811, 0.856] (0.495, 0.516)                  | 0.498 [0.489, 0.506] (0.536, 0.481)                  |
| paraphrase_2         | 0.947 [0.940, 0.953] (0.596, 0.516)                  | 0.497 [0.482, 0.512] (0.544, 0.496)                  |
| scaffold_0           | 0.896 [0.887, 0.905] (0.537, 0.504)                  | 0.665 [0.662, 0.670] (0.579, 0.500)                  |
| scaffold_1           | 0.861 [0.847, 0.875] (0.565, 0.489)                  | 0.570 [0.523, 0.604] (0.515, 0.492)                  |
| scaffold_2           | 0.823 [0.809, 0.836] (0.563, 0.500)                  | 0.626 [0.560, 0.661] (0.574, 0.511)                  |
| concept_swap_0       | 0.835 [0.813, 0.855] (0.507, 0.505)                  | 0.520 [0.510, 0.526] (0.544, 0.517)                  |
| concept_swap_1       | 0.804 [0.780, 0.827] (0.523, 0.512)                  | 0.518 [0.503, 0.531] (0.529, 0.504)                  |
| concept_swap_2       | 0.814 [0.796, 0.833] (0.525, 0.510)                  | 0.527 [0.506, 0.542] (0.550, 0.492)                  |
| concept_swap_3       | 0.838 [0.816, 0.857] (0.503, 0.492)                  | 0.534 [0.515, 0.545] (0.540, 0.490)                  |

`qwen3-1.7b-base`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.943 [0.936, 0.950] (0.608, 0.512)                  | 0.846 [0.833, 0.857] (0.614, 0.507)                  |
| no_template          | 0.928 [0.912, 0.941] (0.536, 0.497)                  | 0.519 [0.507, 0.531] (0.516, 0.484)                  |
| bare                 | 0.908 [0.893, 0.922] (0.518, 0.492)                  | 0.514 [0.432, 0.600] (0.515, 0.504)                  |
| no_chat_tags         | 0.935 [0.926, 0.944] (0.661, 0.492)                  | 0.602 [0.541, 0.652] (0.581, 0.507)                  |
| neutral_frame        | 0.941 [0.923, 0.954] (0.547, 0.495)                  | 0.509 [0.495, 0.522] (0.564, 0.491)                  |
| no_trailing_question | 0.968 [0.962, 0.973] (0.601, 0.487)                  | 0.597 [0.550, 0.640] (0.641, 0.505)                  |
| no_trailing_space    | 0.906 [0.894, 0.917] (0.583, 0.483)                  | 0.611 [0.569, 0.650] (0.628, 0.500)                  |
| paraphrase_0         | 0.975 [0.973, 0.977] (0.669, 0.496)                  | 0.958 [0.955, 0.960] (0.733, 0.495)                  |
| paraphrase_1         | 0.951 [0.949, 0.953] (0.674, 0.515)                  | 0.929 [0.925, 0.933] (0.643, 0.513)                  |
| paraphrase_2         | 0.972 [0.969, 0.974] (0.643, 0.509)                  | 0.969 [0.967, 0.971] (0.698, 0.507)                  |
| scaffold_0           | 0.943 [0.934, 0.951] (0.586, 0.509)                  | 0.607 [0.558, 0.641] (0.657, 0.497)                  |
| scaffold_1           | 0.932 [0.920, 0.943] (0.556, 0.484)                  | 0.525 [0.489, 0.559] (0.563, 0.500)                  |
| scaffold_2           | 0.908 [0.895, 0.921] (0.563, 0.489)                  | 0.563 [0.536, 0.579] (0.601, 0.476)                  |
| concept_swap_0       | 0.860 [0.836, 0.881] (0.529, 0.499)                  | 0.540 [0.460, 0.616] (0.591, 0.490)                  |
| concept_swap_1       | 0.839 [0.814, 0.862] (0.538, 0.490)                  | 0.567 [0.510, 0.608] (0.545, 0.492)                  |
| concept_swap_2       | 0.878 [0.854, 0.898] (0.529, 0.513)                  | 0.563 [0.521, 0.600] (0.543, 0.514)                  |
| concept_swap_3       | 0.898 [0.880, 0.914] (0.556, 0.505)                  | 0.516 [0.484, 0.549] (0.548, 0.496)                  |

`qwen3-1.7b`:

| Variant              | logistic: accuracy [95% interval] (random, shuffled) | pca_diff: accuracy [95% interval] (random, shuffled) |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| paper                | 0.880 [0.866, 0.894] (0.553, 0.500)                  | 0.534 [0.503, 0.562] (0.550, 0.513)                  |
| no_template          | 0.941 [0.933, 0.949] (0.620, 0.503)                  | 0.597 [0.545, 0.644] (0.599, 0.518)                  |
| bare                 | 0.920 [0.912, 0.928] (0.556, 0.493)                  | 0.538 [0.498, 0.582] (0.525, 0.500)                  |
| no_chat_tags         | 0.831 [0.816, 0.848] (0.509, 0.493)                  | 0.515 [0.493, 0.536] (0.536, 0.500)                  |
| neutral_frame        | 0.922 [0.913, 0.930] (0.576, 0.497)                  | 0.519 [0.506, 0.532] (0.588, 0.507)                  |
| no_trailing_question | 0.935 [0.925, 0.945] (0.610, 0.497)                  | 0.526 [0.514, 0.540] (0.552, 0.482)                  |
| no_trailing_space    | not run                                              | not run                                              |
| paraphrase_0         | 0.922 [0.915, 0.929] (0.585, 0.497)                  | 0.536 [0.493, 0.576] (0.545, 0.488)                  |
| paraphrase_1         | 0.939 [0.934, 0.945] (0.587, 0.519)                  | 0.543 [0.514, 0.570] (0.580, 0.490)                  |
| paraphrase_2         | 0.964 [0.959, 0.967] (0.586, 0.512)                  | 0.546 [0.500, 0.590] (0.629, 0.507)                  |
| scaffold_0           | 0.913 [0.897, 0.928] (0.562, 0.507)                  | 0.550 [0.526, 0.575] (0.576, 0.488)                  |
| scaffold_1           | 0.903 [0.888, 0.918] (0.557, 0.482)                  | 0.528 [0.513, 0.543] (0.605, 0.493)                  |
| scaffold_2           | 0.894 [0.878, 0.911] (0.567, 0.489)                  | 0.528 [0.511, 0.542] (0.554, 0.506)                  |
| concept_swap_0       | 0.848 [0.815, 0.879] (0.577, 0.497)                  | 0.498 [0.470, 0.527] (0.533, 0.503)                  |
| concept_swap_1       | 0.793 [0.766, 0.818] (0.546, 0.491)                  | 0.509 [0.493, 0.525] (0.526, 0.484)                  |
| concept_swap_2       | 0.781 [0.754, 0.810] (0.566, 0.512)                  | 0.511 [0.479, 0.544] (0.493, 0.488)                  |
| concept_swap_3       | 0.814 [0.788, 0.838] (0.550, 0.506)                  | 0.517 [0.470, 0.561] (0.528, 0.500)                  |

The same sixteen runs also read the truth statements of the Geometry of Truth split and the six emotions of R4, each
emotion against the others, as the second `gpt2` run of the section does. Each task is read at one layer, the best layer
of the paper reader of the run on the select half, and the report half has the number of pairs in parentheses (59 for
truth and 36 to 53 for the emotions). Lowest and highest variant over all the variants of the task, including
`framing_swap` for the emotions (`no_trailing_space` was run on every model but the post-trained Qwen3 checkpoints):

`gpt2`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant     | Highest variant         | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | ------------------ | ----------------------- | ----- |
| truth (59)          | logistic | L9    | 0.698 [0.681, 0.717]          | 0.517  | 0.495    | scaffold_1 0.608   | paraphrase_0 0.708      | 0.100 |
| truth (59)          | pca_diff | L6    | 0.481 [0.464, 0.502]          | 0.503  | 0.532    | paraphrase_0 0.471 | no_chat_tags 0.592      | 0.120 |
| happiness (38)      | logistic | L12   | 0.850 [0.834, 0.868]          | 0.566  | 0.500    | bare 0.811         | no_template 0.884       | 0.074 |
| happiness (38)      | pca_diff | L12   | 0.605 [0.568, 0.645]          | 0.566  | 0.482    | no_chat_tags 0.505 | scaffold_2 0.832        | 0.326 |
| sadness (44)        | logistic | L12   | 0.930 [0.916, 0.941]          | 0.586  | 0.511    | framing_swap 0.414 | neutral_frame 0.957     | 0.543 |
| sadness (44)        | pca_diff | L11   | 0.845 [0.784, 0.884]          | 0.648  | 0.532    | framing_swap 0.359 | scaffold_1 0.900        | 0.541 |
| anger (45)          | logistic | L9    | 0.947 [0.938, 0.956]          | 0.622  | 0.467    | framing_swap 0.478 | no_chat_tags 0.967      | 0.489 |
| anger (45)          | pca_diff | L12   | 0.849 [0.831, 0.867]          | 0.558  | 0.471    | bare 0.602         | paraphrase_1 0.860      | 0.258 |
| fear (53)           | logistic | L9    | 0.989 [0.975, 1.000]          | 0.562  | 0.509    | framing_swap 0.802 | no_trailing_space 0.998 | 0.196 |
| fear (53)           | pca_diff | L11   | 0.740 [0.704, 0.770]          | 0.534  | 0.494    | no_template 0.628  | framing_swap 0.836      | 0.208 |
| disgust (39)        | logistic | L12   | 1.000 [1.000, 1.000]          | 0.608  | 0.474    | framing_swap 0.818 | paper 1.000             | 0.182 |
| disgust (39)        | pca_diff | L12   | 0.987 [0.974, 0.997]          | 0.608  | 0.477    | framing_swap 0.759 | paraphrase_0 0.997      | 0.238 |
| surprise (36)       | logistic | L10   | 0.933 [0.922, 0.942]          | 0.586  | 0.472    | framing_swap 0.686 | scaffold_1 0.947        | 0.261 |
| surprise (36)       | pca_diff | L11   | 0.728 [0.694, 0.761]          | 0.583  | 0.483    | scaffold_2 0.528   | no_chat_tags 0.839      | 0.311 |

`gpt2-medium`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant     | Highest variant         | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | ------------------ | ----------------------- | ----- |
| truth (59)          | logistic | L19   | 0.656 [0.625, 0.685]          | 0.534  | 0.500    | paraphrase_1 0.605 | bare 0.715              | 0.110 |
| truth (59)          | pca_diff | L1    | 0.517 [0.510, 0.524]          | 0.485  | 0.520    | scaffold_1 0.503   | bare 0.561              | 0.058 |
| happiness (38)      | logistic | L23   | 0.924 [0.908, 0.939]          | 0.600  | 0.489    | paraphrase_0 0.892 | scaffold_0 0.955        | 0.063 |
| happiness (38)      | pca_diff | L18   | 0.811 [0.787, 0.832]          | 0.600  | 0.497    | scaffold_2 0.603   | framing_swap 0.903      | 0.300 |
| sadness (44)        | logistic | L17   | 0.955 [0.943, 0.966]          | 0.591  | 0.509    | framing_swap 0.448 | no_trailing_space 0.984 | 0.536 |
| sadness (44)        | pca_diff | L19   | 0.830 [0.791, 0.866]          | 0.641  | 0.525    | framing_swap 0.336 | scaffold_1 0.891        | 0.555 |
| anger (45)          | logistic | L11   | 0.880 [0.867, 0.893]          | 0.598  | 0.498    | framing_swap 0.438 | no_chat_tags 0.973      | 0.536 |
| anger (45)          | pca_diff | L18   | 0.838 [0.827, 0.851]          | 0.560  | 0.496    | scaffold_0 0.456   | no_chat_tags 0.920      | 0.464 |
| fear (53)           | logistic | L22   | 0.977 [0.970, 0.985]          | 0.589  | 0.502    | framing_swap 0.817 | no_chat_tags 0.991      | 0.174 |
| fear (53)           | pca_diff | L23   | 0.857 [0.845, 0.868]          | 0.577  | 0.509    | scaffold_0 0.662   | no_chat_tags 0.898      | 0.236 |
| disgust (39)        | logistic | L18   | 0.972 [0.967, 0.974]          | 0.613  | 0.472    | framing_swap 0.764 | no_chat_tags 1.000      | 0.236 |
| disgust (39)        | pca_diff | L18   | 0.949 [0.933, 0.964]          | 0.613  | 0.464    | framing_swap 0.754 | no_chat_tags 1.000      | 0.246 |
| surprise (36)       | logistic | L22   | 0.900 [0.886, 0.914]          | 0.586  | 0.472    | framing_swap 0.589 | no_template 0.925       | 0.336 |
| surprise (36)       | pca_diff | L23   | 0.789 [0.744, 0.825]          | 0.569  | 0.472    | no_template 0.542  | no_chat_tags 0.814      | 0.272 |

`gpt2-large`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant     | Highest variant         | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | ------------------ | ----------------------- | ----- |
| truth (59)          | logistic | L23   | 0.653 [0.629, 0.678]          | 0.531  | 0.490    | paraphrase_1 0.615 | bare 0.753              | 0.137 |
| truth (59)          | pca_diff | L2    | 0.534 [0.527, 0.541]          | 0.517  | 0.514    | no_chat_tags 0.495 | concept_swap_0 0.539    | 0.044 |
| happiness (38)      | logistic | L36   | 0.963 [0.953, 0.974]          | 0.532  | 0.476    | scaffold_1 0.908   | paraphrase_1 0.992      | 0.084 |
| happiness (38)      | pca_diff | L32   | 0.874 [0.824, 0.916]          | 0.542  | 0.497    | scaffold_1 0.595   | framing_swap 0.916      | 0.321 |
| sadness (44)        | logistic | L21   | 0.950 [0.934, 0.964]          | 0.575  | 0.505    | framing_swap 0.395 | no_trailing_space 0.986 | 0.591 |
| sadness (44)        | pca_diff | L36   | 0.827 [0.784, 0.861]          | 0.575  | 0.509    | framing_swap 0.484 | no_chat_tags 0.914      | 0.430 |
| anger (45)          | logistic | L14   | 0.953 [0.942, 0.967]          | 0.513  | 0.482    | framing_swap 0.578 | bare 0.978              | 0.400 |
| anger (45)          | pca_diff | L18   | 0.840 [0.818, 0.862]          | 0.598  | 0.471    | scaffold_1 0.551   | no_trailing_space 0.864 | 0.313 |
| fear (53)           | logistic | L27   | 0.998 [0.994, 1.000]          | 0.568  | 0.508    | framing_swap 0.874 | concept_swap_3 1.000    | 0.126 |
| fear (53)           | pca_diff | L35   | 0.932 [0.919, 0.945]          | 0.598  | 0.502    | scaffold_0 0.791   | no_chat_tags 0.975      | 0.185 |
| disgust (39)        | logistic | L23   | 0.982 [0.974, 0.990]          | 0.654  | 0.472    | framing_swap 0.933 | no_chat_tags 0.997      | 0.064 |
| disgust (39)        | pca_diff | L36   | 0.987 [0.979, 0.995]          | 0.718  | 0.477    | scaffold_1 0.892   | scaffold_2 0.990        | 0.097 |
| surprise (36)       | logistic | L21   | 0.925 [0.903, 0.947]          | 0.531  | 0.481    | framing_swap 0.511 | scaffold_2 0.953        | 0.442 |
| surprise (36)       | pca_diff | L6    | 0.772 [0.764, 0.778]          | 0.589  | 0.478    | framing_swap 0.575 | paper 0.772             | 0.197 |

`gpt2-xl`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant       | Highest variant            | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | -------------------- | -------------------------- | ----- |
| truth (59)          | logistic | L26   | 0.656 [0.646, 0.666]          | 0.554  | 0.500    | scaffold_2 0.590     | bare 0.768                 | 0.178 |
| truth (59)          | pca_diff | L2    | 0.488 [0.478, 0.500]          | 0.525  | 0.512    | concept_swap_0 0.480 | bare 0.537                 | 0.058 |
| happiness (38)      | logistic | L27   | 0.921 [0.921, 0.921]          | 0.547  | 0.482    | concept_swap_1 0.895 | no_chat_tags 0.997         | 0.103 |
| happiness (38)      | pca_diff | L43   | 0.839 [0.821, 0.858]          | 0.584  | 0.500    | framing_swap 0.474   | no_trailing_question 0.903 | 0.429 |
| sadness (44)        | logistic | L23   | 0.952 [0.936, 0.966]          | 0.593  | 0.507    | framing_swap 0.439   | concept_swap_0 0.961       | 0.523 |
| sadness (44)        | pca_diff | L42   | 0.793 [0.754, 0.830]          | 0.550  | 0.511    | framing_swap 0.309   | scaffold_1 0.884           | 0.575 |
| anger (45)          | logistic | L31   | 0.958 [0.951, 0.964]          | 0.596  | 0.473    | framing_swap 0.713   | no_chat_tags 1.000         | 0.287 |
| anger (45)          | pca_diff | L22   | 0.749 [0.713, 0.784]          | 0.513  | 0.487    | scaffold_1 0.616     | no_chat_tags 0.962         | 0.347 |
| fear (53)           | logistic | L37   | 1.000 [1.000, 1.000]          | 0.632  | 0.509    | framing_swap 0.915   | paper 1.000                | 0.085 |
| fear (53)           | pca_diff | L44   | 0.987 [0.975, 0.996]          | 0.628  | 0.511    | no_chat_tags 0.528   | concept_swap_3 0.996       | 0.468 |
| disgust (39)        | logistic | L48   | 1.000 [1.000, 1.000]          | 0.664  | 0.474    | bare 0.921           | paper 1.000                | 0.079 |
| disgust (39)        | pca_diff | L46   | 0.997 [0.992, 1.000]          | 0.662  | 0.472    | framing_swap 0.546   | paper 0.997                | 0.451 |
| surprise (36)       | logistic | L27   | 0.894 [0.878, 0.911]          | 0.592  | 0.483    | framing_swap 0.736   | bare 0.956                 | 0.219 |
| surprise (36)       | pca_diff | L40   | 0.722 [0.683, 0.758]          | 0.578  | 0.506    | scaffold_2 0.500     | no_trailing_question 0.758 | 0.258 |

`qwen3-0.6b-base`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant       | Highest variant            | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | -------------------- | -------------------------- | ----- |
| truth (59)          | logistic | L24   | 0.764 [0.749, 0.780]          | 0.527  | 0.514    | concept_swap_3 0.556 | paraphrase_2 0.892         | 0.336 |
| truth (59)          | pca_diff | L28   | 0.588 [0.571, 0.607]          | 0.522  | 0.500    | paraphrase_2 0.469   | bare 0.659                 | 0.190 |
| happiness (38)      | logistic | L18   | 0.974 [0.974, 0.974]          | 0.674  | 0.482    | scaffold_1 0.826     | paper 0.974                | 0.147 |
| happiness (38)      | pca_diff | L19   | 0.958 [0.950, 0.966]          | 0.600  | 0.482    | no_chat_tags 0.545   | paraphrase_0 0.974         | 0.429 |
| sadness (44)        | logistic | L25   | 0.991 [0.984, 0.998]          | 0.575  | 0.509    | framing_swap 0.609   | no_trailing_question 0.998 | 0.389 |
| sadness (44)        | pca_diff | L28   | 0.932 [0.923, 0.941]          | 0.648  | 0.523    | paraphrase_0 0.525   | paper 0.932                | 0.407 |
| anger (45)          | logistic | L19   | 0.978 [0.971, 0.984]          | 0.580  | 0.462    | framing_swap 0.751   | concept_swap_0 0.998       | 0.247 |
| anger (45)          | pca_diff | L27   | 0.651 [0.589, 0.709]          | 0.584  | 0.482    | paraphrase_0 0.547   | bare 0.929                 | 0.382 |
| fear (53)           | logistic | L24   | 1.000 [1.000, 1.000]          | 0.604  | 0.509    | framing_swap 0.908   | paper 1.000                | 0.092 |
| fear (53)           | pca_diff | L16   | 0.898 [0.892, 0.904]          | 0.604  | 0.502    | no_chat_tags 0.600   | bare 0.919                 | 0.319 |
| disgust (39)        | logistic | L16   | 0.995 [0.985, 1.000]          | 0.587  | 0.474    | framing_swap 0.651   | no_trailing_question 1.000 | 0.349 |
| disgust (39)        | pca_diff | L28   | 0.867 [0.854, 0.879]          | 0.651  | 0.510    | paraphrase_0 0.510   | scaffold_1 0.974           | 0.464 |
| surprise (36)       | logistic | L18   | 0.917 [0.903, 0.931]          | 0.594  | 0.500    | framing_swap 0.575   | scaffold_2 0.961           | 0.386 |
| surprise (36)       | pca_diff | L18   | 0.717 [0.694, 0.739]          | 0.594  | 0.433    | no_chat_tags 0.517   | bare 0.908                 | 0.392 |

`qwen3-0.6b`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant     | Highest variant            | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | ------------------ | -------------------------- | ----- |
| truth (59)          | logistic | L18   | 0.744 [0.720, 0.771]          | 0.546  | 0.490    | no_chat_tags 0.624 | neutral_frame 0.885        | 0.261 |
| truth (59)          | pca_diff | L2    | 0.444 [0.432, 0.456]          | 0.493  | 0.475    | paraphrase_1 0.432 | no_chat_tags 0.592         | 0.159 |
| happiness (38)      | logistic | L21   | 0.989 [0.982, 0.997]          | 0.676  | 0.482    | scaffold_1 0.868   | paraphrase_0 0.997         | 0.129 |
| happiness (38)      | pca_diff | L20   | 0.979 [0.974, 0.987]          | 0.668  | 0.482    | no_chat_tags 0.600 | paraphrase_1 0.989         | 0.389 |
| sadness (44)        | logistic | L28   | 0.957 [0.950, 0.964]          | 0.593  | 0.511    | framing_swap 0.636 | no_template 0.966          | 0.330 |
| sadness (44)        | pca_diff | L28   | 0.902 [0.882, 0.923]          | 0.593  | 0.507    | scaffold_2 0.505   | concept_swap_0 0.936       | 0.432 |
| anger (45)          | logistic | L8    | 0.898 [0.882, 0.913]          | 0.542  | 0.502    | framing_swap 0.500 | bare 0.949                 | 0.449 |
| anger (45)          | pca_diff | L24   | 0.907 [0.887, 0.924]          | 0.653  | 0.511    | scaffold_2 0.536   | paraphrase_1 0.927         | 0.391 |
| fear (53)           | logistic | L18   | 0.998 [0.994, 1.000]          | 0.638  | 0.508    | framing_swap 0.906 | paraphrase_1 1.000         | 0.094 |
| fear (53)           | pca_diff | L22   | 0.972 [0.964, 0.979]          | 0.579  | 0.496    | scaffold_2 0.464   | paraphrase_1 0.977         | 0.513 |
| disgust (39)        | logistic | L12   | 0.990 [0.982, 0.997]          | 0.595  | 0.469    | framing_swap 0.687 | paper 0.990                | 0.303 |
| disgust (39)        | pca_diff | L16   | 0.972 [0.967, 0.974]          | 0.669  | 0.472    | no_chat_tags 0.597 | no_trailing_question 0.974 | 0.377 |
| surprise (36)       | logistic | L17   | 0.906 [0.889, 0.922]          | 0.569  | 0.456    | framing_swap 0.444 | no_chat_tags 0.956         | 0.511 |
| surprise (36)       | pca_diff | L25   | 0.836 [0.814, 0.858]          | 0.583  | 0.503    | framing_swap 0.406 | concept_swap_2 0.861       | 0.456 |

`qwen3-1.7b-base`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant       | Highest variant         | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | -------------------- | ----------------------- | ----- |
| truth (59)          | logistic | L18   | 0.969 [0.961, 0.976]          | 0.632  | 0.502    | concept_swap_0 0.776 | paraphrase_2 1.000      | 0.224 |
| truth (59)          | pca_diff | L22   | 0.947 [0.934, 0.958]          | 0.646  | 0.490    | no_chat_tags 0.525   | paraphrase_1 1.000      | 0.475 |
| happiness (38)      | logistic | L19   | 1.000 [1.000, 1.000]          | 0.671  | 0.487    | scaffold_1 0.968     | paper 1.000             | 0.032 |
| happiness (38)      | pca_diff | L19   | 1.000 [1.000, 1.000]          | 0.671  | 0.487    | scaffold_1 0.726     | paper 1.000             | 0.274 |
| sadness (44)        | logistic | L18   | 0.975 [0.968, 0.982]          | 0.625  | 0.516    | framing_swap 0.666   | no_chat_tags 0.998      | 0.332 |
| sadness (44)        | pca_diff | L26   | 0.977 [0.970, 0.984]          | 0.709  | 0.514    | scaffold_1 0.545     | no_chat_tags 0.998      | 0.452 |
| anger (45)          | logistic | L18   | 0.984 [0.978, 0.991]          | 0.618  | 0.464    | framing_swap 0.767   | no_chat_tags 1.000      | 0.233 |
| anger (45)          | pca_diff | L26   | 0.929 [0.918, 0.940]          | 0.553  | 0.480    | paraphrase_0 0.744   | no_chat_tags 0.978      | 0.233 |
| fear (53)           | logistic | L26   | 0.977 [0.972, 0.981]          | 0.655  | 0.502    | framing_swap 0.866   | no_chat_tags 1.000      | 0.134 |
| fear (53)           | pca_diff | L27   | 0.979 [0.975, 0.981]          | 0.617  | 0.500    | scaffold_0 0.481     | concept_swap_3 0.991    | 0.509 |
| disgust (39)        | logistic | L9    | 0.974 [0.974, 0.974]          | 0.687  | 0.474    | framing_swap 0.510   | scaffold_0 0.990        | 0.479 |
| disgust (39)        | pca_diff | L18   | 0.987 [0.979, 0.995]          | 0.669  | 0.462    | framing_swap 0.818   | no_chat_tags 1.000      | 0.182 |
| surprise (36)       | logistic | L21   | 0.964 [0.956, 0.972]          | 0.569  | 0.469    | framing_swap 0.636   | paper 0.964             | 0.328 |
| surprise (36)       | pca_diff | L24   | 0.917 [0.900, 0.931]          | 0.617  | 0.478    | paraphrase_0 0.497   | no_trailing_space 0.933 | 0.436 |

`qwen3-1.7b`:

| Task (report pairs) | Finder   | Layer | Paper accuracy [95% interval] | Random | Shuffled | Lowest variant     | Highest variant      | Span  |
| ------------------- | -------- | ----- | ----------------------------- | ------ | -------- | ------------------ | -------------------- | ----- |
| truth (59)          | logistic | L18   | 0.934 [0.924, 0.946]          | 0.625  | 0.500    | no_chat_tags 0.832 | paraphrase_1 0.985   | 0.153 |
| truth (59)          | pca_diff | L20   | 0.864 [0.836, 0.890]          | 0.615  | 0.495    | no_chat_tags 0.531 | paraphrase_2 0.966   | 0.436 |
| happiness (38)      | logistic | L20   | 0.997 [0.992, 1.000]          | 0.708  | 0.484    | framing_swap 0.905 | paraphrase_0 1.000   | 0.095 |
| happiness (38)      | pca_diff | L23   | 1.000 [1.000, 1.000]          | 0.687  | 0.487    | scaffold_0 0.682   | paper 1.000          | 0.318 |
| sadness (44)        | logistic | L16   | 0.950 [0.941, 0.961]          | 0.564  | 0.518    | framing_swap 0.520 | concept_swap_3 0.982 | 0.461 |
| sadness (44)        | pca_diff | L17   | 0.861 [0.832, 0.889]          | 0.598  | 0.493    | framing_swap 0.491 | bare 0.934           | 0.443 |
| anger (45)          | logistic | L10   | 0.969 [0.962, 0.976]          | 0.591  | 0.484    | framing_swap 0.682 | bare 0.982           | 0.300 |
| anger (45)          | pca_diff | L12   | 0.856 [0.842, 0.867]          | 0.584  | 0.496    | framing_swap 0.573 | bare 0.907           | 0.333 |
| fear (53)           | logistic | L22   | 0.998 [0.994, 1.000]          | 0.640  | 0.511    | scaffold_0 0.943   | no_chat_tags 1.000   | 0.057 |
| fear (53)           | pca_diff | L23   | 0.957 [0.949, 0.962]          | 0.615  | 0.511    | scaffold_0 0.534   | concept_swap_1 0.996 | 0.462 |
| disgust (39)        | logistic | L9    | 0.972 [0.959, 0.985]          | 0.628  | 0.477    | framing_swap 0.667 | paraphrase_0 0.985   | 0.318 |
| disgust (39)        | pca_diff | L12   | 0.962 [0.944, 0.982]          | 0.592  | 0.467    | framing_swap 0.808 | concept_swap_3 0.990 | 0.182 |
| surprise (36)       | logistic | L22   | 0.928 [0.919, 0.936]          | 0.608  | 0.472    | framing_swap 0.656 | paraphrase_1 0.944   | 0.289 |
| surprise (36)       | pca_diff | L24   | 0.822 [0.808, 0.836]          | 0.539  | 0.483    | paraphrase_0 0.561 | concept_swap_0 0.892 | 0.331 |

The interval of the paper accuracy has zero width (the ten readers give the same accuracy on the test pairs) in 10 of
the 112 rows (gpt2 logistic disgust; gpt2-xl logistic happiness; gpt2-xl logistic fear; gpt2-xl logistic disgust;
qwen3-0.6b-base logistic happiness; qwen3-0.6b-base logistic fear; qwen3-1.7b-base logistic happiness; qwen3-1.7b-base
logistic disgust; qwen3-1.7b-base pca_diff happiness; qwen3-1.7b pca_diff happiness); such an interval describes the
training draw only and does not bound the accuracy on other pairs.

- **Truth.** The `pca_diff` paper reader reads the truth statements on both checkpoints: 0.947 [0.934, 0.958] on the
  base checkpoint (random 0.646, shuffled 0.490) against 0.969 [0.961, 0.976] with logistic, and 0.864 [0.836, 0.890] on
  the post-trained one (random 0.615, shuffled 0.495) against 0.934 [0.924, 0.946]. Across the variants the span is
  0.475 on the base checkpoint (0.525 to 1.000) and 0.436 on the post-trained one (0.531 to 0.966) with `pca_diff`,
  against 0.224 and 0.153 with logistic. Of the other variants, 2 lie entirely above and 13 entirely below the paper
  interval with `pca_diff` on the base checkpoint (2 and 12 with logistic), and 5 and 5 on the post-trained one (5 and
  4). With `pca_diff` the smallest margin of a variant over its random series is -0.063 on the base checkpoint
  (`no_chat_tags`, 0.525) and -0.034 on the post-trained one (`no_chat_tags`, 0.531), against 0.234 and 0.263 with
  logistic. The test set is 59 pairs and the intervals cover the training draw only.

- **Emotions.** The paper readers of the six emotions read at 0.964 to 1.000 on the base checkpoint and 0.928 to 0.998
  on the post-trained one with logistic, and at 0.917 to 1.000 and 0.822 to 1.000 with `pca_diff`. The random series of
  these readers are 0.553 to 0.709 on the base checkpoint and 0.539 to 0.708 on the post-trained one (both finders
  together). Outside `framing_swap` the variants of the six emotions read at 0.897 to 1.000 and 0.872 to 1.000 with
  logistic and at 0.481 to 1.000 and 0.514 to 1.000 with `pca_diff`, on the base and the post-trained checkpoint in that
  order.

- **Sentiment against the other tasks.** On the post-trained checkpoint the `pca_diff` paper reader reads truth (0.864)
  and the six emotions (0.822 to 1.000) above their random series but does not read sentiment above its random series
  (0.534, random 0.550). On the base checkpoint it reads sentiment at 0.846, truth at 0.947 and the emotions at 0.917 to
  1.000. The runs do not say why sentiment fails on the post-trained checkpoint.

The truth and emotion tasks on the other six models, with the layers of the paper readers of the truth task and the
quantities of the emotions as for the Qwen3 pair:

- **`gpt2`.** The paper reader reads truth at 0.698 [0.681, 0.717] with logistic (L9, random 0.517, shuffled 0.495) and
  at 0.481 [0.464, 0.502] with `pca_diff` (L6, random 0.503, shuffled 0.532). The span across the variants of truth is
  0.100 with logistic and 0.120 with `pca_diff`; the smallest margin of a variant over its random series is 0.061
  (`scaffold_1`) and -0.039 (`no_template`). The paper readers of the six emotions read at 0.850 to 1.000 with logistic
  and 0.605 to 0.987 with `pca_diff`, against random series of 0.534 to 0.648 (both finders together); outside
  `framing_swap` the variants of the six emotions read at 0.811 to 1.000 and 0.505 to 0.997.

- **`gpt2-medium`.** The paper reader reads truth at 0.656 [0.625, 0.685] with logistic (L19, random 0.534, shuffled
  0.500) and at 0.517 [0.510, 0.524] with `pca_diff` (L1, random 0.485, shuffled 0.520). The span across the variants of
  truth is 0.110 with logistic and 0.058 with `pca_diff`; the smallest margin of a variant over its random series is
  0.032 (`paraphrase_1`) and 0.000 (`neutral_frame`). The paper readers of the six emotions read at 0.880 to 0.977 with
  logistic and 0.789 to 0.949 with `pca_diff`, against random series of 0.560 to 0.641 (both finders together); outside
  `framing_swap` the variants of the six emotions read at 0.860 to 1.000 and 0.456 to 1.000.

- **`gpt2-large`.** The paper reader reads truth at 0.653 [0.629, 0.678] with logistic (L23, random 0.531, shuffled
  0.490) and at 0.534 [0.527, 0.541] with `pca_diff` (L2, random 0.517, shuffled 0.514). The span across the variants of
  truth is 0.137 with logistic and 0.044 with `pca_diff`; the smallest margin of a variant over its random series is
  0.095 (`concept_swap_3`) and -0.039 (`no_chat_tags`). The paper readers of the six emotions read at 0.925 to 0.998
  with logistic and 0.772 to 0.987 with `pca_diff`, against random series of 0.513 to 0.718 (both finders together);
  outside `framing_swap` the variants of the six emotions read at 0.869 to 1.000 and 0.551 to 0.990.

- **`gpt2-xl`.** The paper reader reads truth at 0.656 [0.646, 0.666] with logistic (L26, random 0.554, shuffled 0.500)
  and at 0.488 [0.478, 0.500] with `pca_diff` (L2, random 0.525, shuffled 0.512). The span across the variants of truth
  is 0.178 with logistic and 0.058 with `pca_diff`; the smallest margin of a variant over its random series is 0.085
  (`scaffold_1`) and -0.051 (`no_chat_tags`). The paper readers of the six emotions read at 0.894 to 1.000 with logistic
  and 0.722 to 0.997 with `pca_diff`, against random series of 0.513 to 0.664 (both finders together); outside
  `framing_swap` the variants of the six emotions read at 0.878 to 1.000 and 0.500 to 0.997.

- **`qwen3-0.6b-base`.** The paper reader reads truth at 0.764 [0.749, 0.780] with logistic (L24, random 0.527, shuffled
  0.514) and at 0.588 [0.571, 0.607] with `pca_diff` (L28, random 0.522, shuffled 0.500). The span across the variants
  of truth is 0.336 with logistic and 0.190 with `pca_diff`; the smallest margin of a variant over its random series is
  0.047 (`concept_swap_3`) and -0.097 (`paraphrase_2`). The paper readers of the six emotions read at 0.917 to 1.000
  with logistic and 0.651 to 0.958 with `pca_diff`, against random series of 0.575 to 0.674 (both finders together);
  outside `framing_swap` the variants of the six emotions read at 0.826 to 1.000 and 0.510 to 0.974.

- **`qwen3-0.6b`.** The paper reader reads truth at 0.744 [0.720, 0.771] with logistic (L18, random 0.546, shuffled
  0.490) and at 0.444 [0.432, 0.456] with `pca_diff` (L2, random 0.493, shuffled 0.475). The span across the variants of
  truth is 0.261 with logistic and 0.159 with `pca_diff`; the smallest margin of a variant over its random series is
  0.124 (`no_chat_tags`) and -0.058 (`paraphrase_0`). The paper readers of the six emotions read at 0.898 to 0.998 with
  logistic and 0.836 to 0.979 with `pca_diff`, against random series of 0.542 to 0.676 (both finders together); outside
  `framing_swap` the variants of the six emotions read at 0.831 to 1.000 and 0.464 to 0.989.

The function framing (`framing_swap`) of each emotion, trained at the response position and scored on its own items:

| Model           | Emotion   | Finder   | Function framing accuracy [95% interval] | Random | Shuffled | Cosine with paper | Paper accuracy |
| --------------- | --------- | -------- | ---------------------------------------- | ------ | -------- | ----------------- | -------------- |
| gpt2            | happiness | logistic | 0.829 [0.805, 0.853]                     | 0.508  | 0.495    | 0.372             | 0.850          |
| gpt2            | happiness | pca_diff | 0.755 [0.739, 0.771]                     | 0.508  | 0.511    | 0.024             | 0.605          |
| gpt2            | sadness   | logistic | 0.414 [0.384, 0.445]                     | 0.527  | 0.486    | 0.049             | 0.930          |
| gpt2            | sadness   | pca_diff | 0.359 [0.330, 0.389]                     | 0.425  | 0.468    | -0.120            | 0.845          |
| gpt2            | anger     | logistic | 0.478 [0.453, 0.502]                     | 0.449  | 0.509    | 0.075             | 0.947          |
| gpt2            | anger     | pca_diff | 0.640 [0.624, 0.653]                     | 0.516  | 0.493    | 0.169             | 0.849          |
| gpt2            | fear      | logistic | 0.802 [0.781, 0.825]                     | 0.496  | 0.530    | 0.174             | 0.989          |
| gpt2            | fear      | pca_diff | 0.836 [0.830, 0.842]                     | 0.500  | 0.515    | 0.204             | 0.740          |
| gpt2            | disgust   | logistic | 0.818 [0.792, 0.838]                     | 0.585  | 0.462    | 0.202             | 1.000          |
| gpt2            | disgust   | pca_diff | 0.759 [0.751, 0.767]                     | 0.585  | 0.479    | 0.221             | 0.987          |
| gpt2            | surprise  | logistic | 0.686 [0.667, 0.708]                     | 0.497  | 0.519    | 0.025             | 0.933          |
| gpt2            | surprise  | pca_diff | 0.589 [0.572, 0.603]                     | 0.478  | 0.572    | -0.075            | 0.728          |
| gpt2-medium     | happiness | logistic | 0.947 [0.937, 0.958]                     | 0.558  | 0.487    | 0.288             | 0.924          |
| gpt2-medium     | happiness | pca_diff | 0.903 [0.874, 0.932]                     | 0.616  | 0.495    | 0.123             | 0.811          |
| gpt2-medium     | sadness   | logistic | 0.448 [0.414, 0.484]                     | 0.500  | 0.507    | 0.030             | 0.955          |
| gpt2-medium     | sadness   | pca_diff | 0.336 [0.316, 0.361]                     | 0.477  | 0.500    | -0.084            | 0.830          |
| gpt2-medium     | anger     | logistic | 0.438 [0.416, 0.462]                     | 0.498  | 0.500    | 0.036             | 0.880          |
| gpt2-medium     | anger     | pca_diff | 0.571 [0.551, 0.589]                     | 0.518  | 0.496    | 0.024             | 0.838          |
| gpt2-medium     | fear      | logistic | 0.817 [0.806, 0.828]                     | 0.511  | 0.508    | 0.217             | 0.977          |
| gpt2-medium     | fear      | pca_diff | 0.777 [0.768, 0.787]                     | 0.526  | 0.513    | 0.230             | 0.857          |
| gpt2-medium     | disgust   | logistic | 0.764 [0.751, 0.777]                     | 0.490  | 0.474    | 0.205             | 0.972          |
| gpt2-medium     | disgust   | pca_diff | 0.754 [0.744, 0.764]                     | 0.490  | 0.479    | 0.262             | 0.949          |
| gpt2-medium     | surprise  | logistic | 0.589 [0.558, 0.617]                     | 0.481  | 0.511    | 0.085             | 0.900          |
| gpt2-medium     | surprise  | pca_diff | 0.675 [0.658, 0.692]                     | 0.558  | 0.469    | 0.063             | 0.789          |
| gpt2-large      | happiness | logistic | 0.934 [0.926, 0.942]                     | 0.500  | 0.495    | 0.353             | 0.963          |
| gpt2-large      | happiness | pca_diff | 0.916 [0.908, 0.921]                     | 0.495  | 0.482    | 0.261             | 0.874          |
| gpt2-large      | sadness   | logistic | 0.395 [0.366, 0.425]                     | 0.511  | 0.514    | 0.038             | 0.950          |
| gpt2-large      | sadness   | pca_diff | 0.484 [0.466, 0.502]                     | 0.470  | 0.452    | -0.039            | 0.827          |
| gpt2-large      | anger     | logistic | 0.578 [0.564, 0.589]                     | 0.500  | 0.502    | 0.049             | 0.953          |
| gpt2-large      | anger     | pca_diff | 0.722 [0.709, 0.733]                     | 0.580  | 0.460    | -0.001            | 0.840          |
| gpt2-large      | fear      | logistic | 0.874 [0.864, 0.885]                     | 0.577  | 0.504    | 0.253             | 0.998          |
| gpt2-large      | fear      | pca_diff | 0.919 [0.911, 0.926]                     | 0.553  | 0.511    | 0.296             | 0.932          |
| gpt2-large      | disgust   | logistic | 0.933 [0.921, 0.944]                     | 0.503  | 0.474    | 0.184             | 0.982          |
| gpt2-large      | disgust   | pca_diff | 0.951 [0.936, 0.964]                     | 0.618  | 0.487    | 0.250             | 0.987          |
| gpt2-large      | surprise  | logistic | 0.511 [0.503, 0.519]                     | 0.531  | 0.517    | 0.023             | 0.925          |
| gpt2-large      | surprise  | pca_diff | 0.575 [0.561, 0.586]                     | 0.528  | 0.469    | 0.009             | 0.772          |
| gpt2-xl         | happiness | logistic | 0.921 [0.908, 0.934]                     | 0.484  | 0.487    | 0.299             | 0.921          |
| gpt2-xl         | happiness | pca_diff | 0.474 [0.474, 0.474]                     | 0.474  | 0.518    | -0.210            | 0.839          |
| gpt2-xl         | sadness   | logistic | 0.439 [0.430, 0.450]                     | 0.489  | 0.520    | 0.068             | 0.952          |
| gpt2-xl         | sadness   | pca_diff | 0.309 [0.302, 0.316]                     | 0.468  | 0.477    | -0.334            | 0.793          |
| gpt2-xl         | anger     | logistic | 0.713 [0.700, 0.727]                     | 0.504  | 0.504    | 0.135             | 0.958          |
| gpt2-xl         | anger     | pca_diff | 0.671 [0.660, 0.680]                     | 0.476  | 0.520    | -0.021            | 0.749          |
| gpt2-xl         | fear      | logistic | 0.915 [0.898, 0.932]                     | 0.553  | 0.508    | 0.152             | 1.000          |
| gpt2-xl         | fear      | pca_diff | 0.711 [0.702, 0.719]                     | 0.540  | 0.530    | 0.255             | 0.987          |
| gpt2-xl         | disgust   | logistic | 0.938 [0.926, 0.951]                     | 0.605  | 0.479    | 0.221             | 1.000          |
| gpt2-xl         | disgust   | pca_diff | 0.546 [0.538, 0.554]                     | 0.533  | 0.549    | 0.013             | 0.997          |
| gpt2-xl         | surprise  | logistic | 0.736 [0.725, 0.747]                     | 0.475  | 0.503    | 0.108             | 0.894          |
| gpt2-xl         | surprise  | pca_diff | 0.531 [0.481, 0.578]                     | 0.511  | 0.486    | -0.134            | 0.722          |
| qwen3-0.6b-base | happiness | logistic | 0.924 [0.913, 0.934]                     | 0.616  | 0.484    | 0.338             | 0.974          |
| qwen3-0.6b-base | happiness | pca_diff | 0.921 [0.921, 0.921]                     | 0.532  | 0.482    | 0.388             | 0.958          |
| qwen3-0.6b-base | sadness   | logistic | 0.609 [0.575, 0.643]                     | 0.448  | 0.514    | 0.257             | 0.991          |
| qwen3-0.6b-base | sadness   | pca_diff | 0.586 [0.570, 0.605]                     | 0.611  | 0.527    | 0.009             | 0.932          |
| qwen3-0.6b-base | anger     | logistic | 0.751 [0.731, 0.769]                     | 0.569  | 0.498    | 0.161             | 0.978          |
| qwen3-0.6b-base | anger     | pca_diff | 0.718 [0.702, 0.731]                     | 0.456  | 0.536    | 0.060             | 0.651          |
| qwen3-0.6b-base | fear      | logistic | 0.908 [0.889, 0.926]                     | 0.502  | 0.511    | 0.213             | 1.000          |
| qwen3-0.6b-base | fear      | pca_diff | 0.845 [0.838, 0.853]                     | 0.570  | 0.494    | 0.361             | 0.898          |
| qwen3-0.6b-base | disgust   | logistic | 0.651 [0.638, 0.664]                     | 0.546  | 0.521    | 0.041             | 0.995          |
| qwen3-0.6b-base | disgust   | pca_diff | 0.738 [0.679, 0.790]                     | 0.536  | 0.479    | 0.134             | 0.867          |
| qwen3-0.6b-base | surprise  | logistic | 0.575 [0.547, 0.603]                     | 0.522  | 0.531    | 0.032             | 0.917          |
| qwen3-0.6b-base | surprise  | pca_diff | 0.561 [0.544, 0.578]                     | 0.522  | 0.528    | -0.042            | 0.717          |
| qwen3-0.6b      | happiness | logistic | 0.974 [0.974, 0.974]                     | 0.639  | 0.482    | 0.386             | 0.989          |
| qwen3-0.6b      | happiness | pca_diff | 0.982 [0.974, 0.989]                     | 0.542  | 0.479    | 0.593             | 0.979          |
| qwen3-0.6b      | sadness   | logistic | 0.636 [0.611, 0.657]                     | 0.561  | 0.536    | 0.277             | 0.957          |
| qwen3-0.6b      | sadness   | pca_diff | 0.643 [0.632, 0.655]                     | 0.561  | 0.511    | 0.096             | 0.902          |
| qwen3-0.6b      | anger     | logistic | 0.500 [0.476, 0.524]                     | 0.571  | 0.518    | 0.046             | 0.898          |
| qwen3-0.6b      | anger     | pca_diff | 0.798 [0.791, 0.804]                     | 0.496  | 0.491    | 0.299             | 0.907          |
| qwen3-0.6b      | fear      | logistic | 0.906 [0.900, 0.911]                     | 0.453  | 0.502    | 0.149             | 0.998          |
| qwen3-0.6b      | fear      | pca_diff | 0.883 [0.877, 0.887]                     | 0.592  | 0.506    | 0.331             | 0.972          |
| qwen3-0.6b      | disgust   | logistic | 0.687 [0.669, 0.708]                     | 0.500  | 0.505    | 0.061             | 0.990          |
| qwen3-0.6b      | disgust   | pca_diff | 0.859 [0.846, 0.869]                     | 0.459  | 0.482    | 0.234             | 0.972          |
| qwen3-0.6b      | surprise  | logistic | 0.444 [0.414, 0.472]                     | 0.503  | 0.533    | -0.032            | 0.906          |
| qwen3-0.6b      | surprise  | pca_diff | 0.406 [0.369, 0.442]                     | 0.489  | 0.556    | 0.066             | 0.836          |
| qwen3-1.7b-base | happiness | logistic | 0.974 [0.974, 0.974]                     | 0.608  | 0.487    | 0.358             | 1.000          |
| qwen3-1.7b-base | happiness | pca_diff | 0.987 [0.979, 0.995]                     | 0.608  | 0.489    | 0.533             | 1.000          |
| qwen3-1.7b-base | sadness   | logistic | 0.666 [0.650, 0.682]                     | 0.484  | 0.516    | 0.194             | 0.975          |
| qwen3-1.7b-base | sadness   | pca_diff | 0.780 [0.764, 0.795]                     | 0.532  | 0.475    | 0.192             | 0.977          |
| qwen3-1.7b-base | anger     | logistic | 0.767 [0.742, 0.789]                     | 0.409  | 0.504    | 0.140             | 0.984          |
| qwen3-1.7b-base | anger     | pca_diff | 0.851 [0.844, 0.858]                     | 0.540  | 0.491    | 0.312             | 0.929          |
| qwen3-1.7b-base | fear      | logistic | 0.866 [0.857, 0.875]                     | 0.602  | 0.519    | 0.120             | 0.977          |
| qwen3-1.7b-base | fear      | pca_diff | 0.819 [0.809, 0.828]                     | 0.557  | 0.498    | 0.141             | 0.979          |
| qwen3-1.7b-base | disgust   | logistic | 0.510 [0.497, 0.526]                     | 0.633  | 0.492    | 0.065             | 0.974          |
| qwen3-1.7b-base | disgust   | pca_diff | 0.818 [0.813, 0.821]                     | 0.600  | 0.467    | 0.151             | 0.987          |
| qwen3-1.7b-base | surprise  | logistic | 0.636 [0.619, 0.650]                     | 0.525  | 0.525    | 0.077             | 0.964          |
| qwen3-1.7b-base | surprise  | pca_diff | 0.728 [0.717, 0.739]                     | 0.578  | 0.522    | 0.227             | 0.917          |
| qwen3-1.7b      | happiness | logistic | 0.905 [0.897, 0.913]                     | 0.387  | 0.482    | 0.249             | 0.997          |
| qwen3-1.7b      | happiness | pca_diff | 0.921 [0.921, 0.921]                     | 0.576  | 0.471    | 0.372             | 1.000          |
| qwen3-1.7b      | sadness   | logistic | 0.520 [0.500, 0.541]                     | 0.486  | 0.502    | 0.070             | 0.950          |
| qwen3-1.7b      | sadness   | pca_diff | 0.491 [0.477, 0.502]                     | 0.566  | 0.509    | -0.042            | 0.861          |
| qwen3-1.7b      | anger     | logistic | 0.682 [0.664, 0.698]                     | 0.493  | 0.504    | 0.083             | 0.969          |
| qwen3-1.7b      | anger     | pca_diff | 0.573 [0.556, 0.591]                     | 0.516  | 0.507    | 0.081             | 0.856          |
| qwen3-1.7b      | fear      | logistic | 0.957 [0.951, 0.962]                     | 0.525  | 0.515    | 0.133             | 0.998          |
| qwen3-1.7b      | fear      | pca_diff | 0.949 [0.945, 0.955]                     | 0.566  | 0.508    | 0.219             | 0.957          |
| qwen3-1.7b      | disgust   | logistic | 0.667 [0.636, 0.697]                     | 0.505  | 0.485    | 0.057             | 0.972          |
| qwen3-1.7b      | disgust   | pca_diff | 0.808 [0.795, 0.821]                     | 0.510  | 0.482    | 0.093             | 0.962          |
| qwen3-1.7b      | surprise  | logistic | 0.656 [0.633, 0.681]                     | 0.603  | 0.483    | 0.102             | 0.928          |
| qwen3-1.7b      | surprise  | pca_diff | 0.622 [0.611, 0.633]                     | 0.489  | 0.494    | 0.126             | 0.822          |

- **The function framing.** It is the lowest variant of the emotion in 53 of the 96 (emotion, run) cells, and its cosine
  with the paper reader is between -0.334 and 0.593. Its interval lies entirely above the random series of its cell in
  78 of the 96 cells and entirely below it in 15 (gpt2 logistic sadness; gpt2 pca_diff sadness; gpt2-medium logistic
  sadness; gpt2-medium logistic anger; gpt2-medium pca_diff sadness; gpt2-large logistic sadness; gpt2-large logistic
  surprise; gpt2-xl logistic sadness; gpt2-xl pca_diff sadness; qwen3-0.6b-base pca_diff sadness; qwen3-0.6b logistic
  anger; qwen3-0.6b logistic surprise; qwen3-0.6b pca_diff surprise; qwen3-1.7b-base logistic disgust; qwen3-1.7b
  pca_diff sadness). It includes 0.5 in 7 (gpt2 logistic anger; gpt2-large pca_diff sadness; gpt2-xl pca_diff surprise;
  qwen3-0.6b logistic anger; qwen3-1.7b-base logistic disgust; qwen3-1.7b logistic sadness; qwen3-1.7b pca_diff sadness)
  and lies entirely below 0.5 in 11 (gpt2 logistic sadness; gpt2 pca_diff sadness; gpt2-medium logistic sadness;
  gpt2-medium logistic anger; gpt2-medium pca_diff sadness; gpt2-large logistic sadness; gpt2-xl logistic sadness;
  gpt2-xl pca_diff happiness; gpt2-xl pca_diff sadness; qwen3-0.6b logistic surprise; qwen3-0.6b pca_diff surprise). The
  interval has zero width in 5 (gpt2-xl pca_diff happiness; qwen3-0.6b-base pca_diff happiness; qwen3-0.6b logistic
  happiness; qwen3-1.7b-base logistic happiness; qwen3-1.7b pca_diff happiness).

What the added runs show. How much the template matters depends on the finder. With the logistic finder every variant of
both checkpoints reads above the random series of its cell (by at least 0.214), and the paper template is neither the
best nor the worst variant. With `pca_diff` that is not so. On the post-trained checkpoint no variant reads sentiment
above its random series: the 16 variants lie between 0.498 and 0.597, the reader is at or below its random series in 14
of them, and there is no ordering of the templates to compare with the logistic one. On the base checkpoint the paper
variant reads (0.846) and so do the three paraphrases, but the other 12 variants read at 0.509 to 0.607, and the span
grows from 0.136 with logistic to 0.459. On `gpt2`, [P4](#direction-finder-pairing-and-preprocessing-p4) also found
`pca_diff` at the level of the random direction for sentiment (0.512 against 0.543).

The order of the templates found with the logistic finder holds only in part on the base checkpoint. The highest variant
is a paraphrase under both finders (`paraphrase_0` with logistic, `paraphrase_2` with `pca_diff`). `paraphrase_0` and
`paraphrase_2` lie entirely above the paper interval under both finders, `paraphrase_1` with `pca_diff` only and
`no_trailing_question` with logistic only. The lowest variants differ (`concept_swap_1` with logistic, `neutral_frame`
with `pca_diff`). The Spearman rank correlation between the 16 accuracies under the two finders is 0.647 on the base
checkpoint; on the post-trained one it is 0.770, but the 16 `pca_diff` accuracies there are within 0.1 of chance, so
that value is not read as agreement in the order of the templates. No significance is claimed for either value.

Across the eight models, the `pca_diff` reader of the paper variant has the mean of its random series entirely below its
interval on 2 (`gpt2-medium`, `qwen3-1.7b-base`), inside its interval on 3 (`qwen3-0.6b-base`, `qwen3-0.6b`,
`qwen3-1.7b`) and above it on 3 (`gpt2`, `gpt2-large`, `gpt2-xl`). With the logistic finder the reader is above the
random series of its cell on all 16 variants of every model, the smallest margin being 0.162 (`paraphrase_0` on
`gpt2-medium`). The Spearman rank correlations between the 16 accuracies under the two finders are -0.077, 0.632,
-0.059, 0.488, 0.739, 0.115, 0.647, 0.770 in the order of the tables (gpt2, gpt2-medium, gpt2-large, gpt2-xl,
qwen3-0.6b-base, qwen3-0.6b, qwen3-1.7b-base, qwen3-1.7b); no significance is claimed for them. The 16 `pca_diff`
accuracies of `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` and `qwen3-0.6b` are within 0.131, 0.051, 0.157, 0.185 and
0.165 of 0.5, so their correlations (-0.077, 0.632, -0.059, 0.488 and 0.115) are not read as agreement in the order of
the templates. The paper accuracies are 0.512, 0.517, 0.495, 0.507, 0.665, 0.564, 0.846, 0.534 with `pca_diff` and
0.765, 0.742, 0.807, 0.776, 0.952, 0.860, 0.943, 0.880 with logistic, in the same order.

On the post-trained Qwen3 1.7B checkpoint the failure does not extend to the other tasks. The same finder reads truth
(0.864) and the six emotions (0.822 to 1.000) under the paper template on the post-trained checkpoint, above their
random series, and it moves more across the variants than logistic does on truth too (span 0.475 and 0.436 against 0.224
and 0.153).

What they do not show. The runs do not say why `pca_diff` fails on post-trained sentiment, or why on the base checkpoint
it follows the paraphrases and not the other variants: a near-chance reader may be a direction that tracks something
other than sentiment under that wording, or a layer that does not suit it, and the runs do not separate the two. Every
cell is one run with one layer for all variants of the run, chosen with the paper reader of that run, so a `pca_diff`
row and a logistic row differ in the layer as well as in the finder. The intervals cover the training draw only, and the
evaluation pairs are shared by all variants (214 for sentiment, 59 for truth, 36 to 53 per emotion). Base and
post-trained checkpoints are different models and are not pooled.

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
The shuffled-label series is between 0.48 and 0.52 at every size.

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

- **The paper finder does not read here.** `pca_diff` is at 0.512 for sentiment and 0.481 for truth, not above the
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
- These are one run on `gpt2`; the other GPT-2 sizes and the Qwen3 checkpoints are in [the section on P3 and P4 across the GPT-2 sizes and Qwen3](#stimulus-set-size-and-direction-finder-across-the-gpt-2-sizes-and-qwen3-p3-p4).

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
post-trained checkpoints of Qwen3 0.6B and 1.7B; the 4B pair follows in its own block at the end of the section. The
post-trained checkpoints use the chat template with thinking disabled and the base checkpoints plain tags. Each steering
arm has three readers, so the tables give the mean over the three readers with the range in parentheses, as in the GPT-2
section (single-run arms have no range). Base and post-trained checkpoints are different models: the columns are read
within a pair and never pooled. All four steer at their last block (L28) and over a window of the last three blocks, and
read at L18.

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

#### The 4B pair (C1)

The same C1 procedure on `qwen3-4b-base` and `qwen3-4b` (36 blocks; runs `c1/qwen3-4b-base/72a965626cd7` and
`c1/qwen3-4b/50671febc316`, made at `d09822b` on a rented GPU pod, see [Run identity](#run-identity)): 100 neutral
prompts, the primary and the check judge, three readers per steering arm, the chat template with thinking disabled for
the post-trained checkpoint and plain tags for the base one. The tables give the mean over the three readers with the
range in parentheses, as above, and the two checkpoints stay in separate columns. The base checkpoint steers at its
last block, like the four smaller checkpoints. The post-trained checkpoint does not: its sweep picks L22, and the two
judges agree on the sign of the dose slope at 3 of the 6 swept layers.

| Model         | Reading layer | Steer layer | Window     | Judge agreement |
| ------------- | ------------- | ----------- | ---------- | --------------- |
| qwen3-4b-base | L24           | L36         | 34, 35, 36 | 1.000           |
| qwen3-4b      | L23           | L22         | 21, 22, 23 | 0.500           |

The steer layer is the swept layer with the largest mean dose slope on the primary judge. The dose slope is the
least-squares slope of the score on the coefficient, averaged over the three readers; the table gives it for the primary
judge with the check judge in parentheses (recomputed from the stored per-reader sweep curves with the dose-slope
function of the run, which also gives the two steer layers and the two judge agreements above). The sweep scores a
separate list of 100 prompts, so its unsteered scores (0.59 and 0.65) differ from the 0.64 and 0.63 of the tables below.

| Swept layer | qwen3-4b-base | qwen3-4b       |
| ----------- | ------------- | -------------- |
| L1          | 0.032 (0.072) | 0.006 (-0.018) |
| L8          | 0.111 (0.055) | 0.078 (0.058)  |
| L15         | 0.170 (0.139) | 0.018 (-0.024) |
| L22         | 0.629 (0.661) | 0.123 (0.139)  |
| L29         | 0.444 (0.435) | 0.060 (0.087)  |
| L36         | 0.973 (1.001) | -0.008 (0.123) |

Primary-judge score:

| Arm          | qwen3-4b-base       | qwen3-4b            |
| ------------ | ------------------- | ------------------- |
| none         | 0.64                | 0.63                |
| linear +0.1  | 0.78 (0.73 to 0.81) | 0.67 (0.64 to 0.68) |
| linear +0.2  | 0.91 (0.87 to 0.94) | 0.65 (0.59 to 0.69) |
| linear +0.3  | 0.96 (0.93 to 0.98) | 0.68 (0.63 to 0.76) |
| linear -0.1  | 0.30 (0.14 to 0.43) | 0.58 (0.52 to 0.67) |
| linear -0.2  | 0.05 (0.00 to 0.12) | 0.57 (0.51 to 0.68) |
| linear -0.3  | 0.04 (0.00 to 0.11) | 0.54 (0.42 to 0.74) |
| prompting +1 | 0.86                | 0.83                |
| prompting +2 | 0.90                | 0.88                |
| prompting -1 | 0.39                | 0.40                |
| prompting -2 | 0.37                | 0.47                |

Check-judge score:

| Arm          | qwen3-4b-base       | qwen3-4b            |
| ------------ | ------------------- | ------------------- |
| none         | 0.48                | 0.49                |
| linear +0.1  | 0.67 (0.63 to 0.71) | 0.48 (0.41 to 0.54) |
| linear +0.2  | 0.82 (0.75 to 0.86) | 0.50 (0.43 to 0.59) |
| linear +0.3  | 0.91 (0.83 to 0.97) | 0.53 (0.40 to 0.62) |
| linear -0.1  | 0.16 (0.08 to 0.24) | 0.46 (0.41 to 0.55) |
| linear -0.2  | 0.03 (0.01 to 0.06) | 0.43 (0.34 to 0.57) |
| linear -0.3  | 0.01 (0.01 to 0.01) | 0.43 (0.33 to 0.60) |
| prompting +1 | 0.81                | 0.80                |
| prompting +2 | 0.81                | 0.86                |
| prompting -1 | 0.35                | 0.26                |
| prompting -2 | 0.38                | 0.31                |

Perplexity of neutral text:

| Arm         | qwen3-4b-base           | qwen3-4b              |
| ----------- | ----------------------- | --------------------- |
| none        | 15.9                    | 31.7                  |
| linear +0.1 | 18.7 (18.4 to 19.2)     | 32.6 (29.6 to 35.6)   |
| linear +0.2 | 29.6 (28.0 to 32.5)     | 42.8 (37.5 to 50.3)   |
| linear +0.3 | 65.0 (57.6 to 79.2)     | 69.5 (61.8 to 84.3)   |
| linear -0.1 | 18.6 (18.2 to 19.2)     | 39.5 (34.8 to 46.5)   |
| linear -0.2 | 59.3 (31.7 to 108.0)    | 65.9 (47.1 to 95.7)   |
| linear -0.3 | 1330.7 (99.1 to 3704.7) | 151.5 (80.7 to 270.6) |

Distinct-1 of the generated text:

| Arm          | qwen3-4b-base          | qwen3-4b               |
| ------------ | ---------------------- | ---------------------- |
| none         | 0.283                  | 0.224                  |
| linear +0.1  | 0.290 (0.277 to 0.298) | 0.219 (0.206 to 0.228) |
| linear +0.2  | 0.258 (0.218 to 0.293) | 0.212 (0.198 to 0.220) |
| linear +0.3  | 0.130 (0.093 to 0.175) | 0.178 (0.158 to 0.190) |
| linear -0.1  | 0.214 (0.139 to 0.269) | 0.210 (0.200 to 0.227) |
| linear -0.2  | 0.047 (0.005 to 0.105) | 0.185 (0.168 to 0.210) |
| linear -0.3  | 0.013 (0.001 to 0.034) | 0.140 (0.124 to 0.165) |
| prompting +1 | 0.232                  | 0.182                  |
| prompting +2 | 0.242                  | 0.172                  |
| prompting -1 | 0.227                  | 0.174                  |
| prompting -2 | 0.213                  | 0.159                  |

A coefficient is called fluent by the rule above: the mean perplexity is at most 1.5 times the unsteered one and the mean
distinct-1 is at least 0.15, both over the three steering readers.

- **`qwen3-4b-base`, direction at the fluent coefficients.** +0.1 and -0.1 are fluent (perplexity 18.7 and 18.6 against
  15.9 unsteered, that is 1.18 and 1.17 times; distinct-1 0.290 and 0.214 against 0.283). At -0.1 one reader is at 0.139,
  just under the distinct-1 cutoff, as single readers of the smaller checkpoints are. The primary-judge score goes from
  0.64 without steering to 0.78 (0.73 to 0.81) at +0.1 and 0.30 (0.14 to 0.43) at -0.1. The check judge agrees in
  direction: 0.48 without steering, 0.67 (0.63 to 0.71) at +0.1 and 0.16 (0.08 to 0.24) at -0.1, and every swept layer
  has the same slope sign under both judges (agreement 1.000).
- **`qwen3-4b-base`, against prompting.** At +0.1 the steered score is below both positive prompting arms (0.78 against
  0.86 and 0.90) and the whole range over readers is below them. At -0.1 the mean is below both negative arms (0.30
  against 0.39 and 0.37) but the range over readers reaches both, because the highest reader scores 0.43. At +0.2 the
  score is 0.91 (0.87 to 0.94), level with the stronger positive arm (0.90), but that coefficient is not fluent: the
  perplexity is 29.6 (1.87 times the unsteered one) with a distinct-1 of 0.258. The prompting arms are single runs and
  have no range; their distinct-1 is 0.213 to 0.242, in the band of the steered text at +0.1 and -0.1 (0.214 to 0.290).
- **`qwen3-4b-base`, beyond the fluent coefficients.** At +0.3 the perplexity is 65.0 (4.09 times the unsteered one) and
  the distinct-1 is 0.130 (0.093 to 0.175). The positive side degrades more slowly than on `qwen3-0.6b-base` and
  `qwen3-1.7b-base` (at +0.2 1.87 times against 2.84 and 2.13, at +0.3 4.09 against 10.5 and 5.0). The negative side
  degrades further: at -0.2 the perplexity is 59.3 (31.7 to 108.0) with a distinct-1 of 0.047, and at -0.3 it is 1330.7
  (99.1 to 3704.7, 83.8 times the unsteered one) with a distinct-1 of 0.013 (0.001 to 0.034). That mean is the largest
  perplexity in the Qwen3 C1 tables (the next is 836.2, `qwen3-0.6b` at -0.3) and it is carried by one reader (3704.7
  against 99.1 and 188.3). The scores at -0.2 and -0.3 (0.05 and 0.04, with distinct-1 0.047 and 0.013) and at +0.3 (0.96,
  with perplexity 65.0 and distinct-1 0.130) are scores of degraded text and are not read as steering that works.
- **`qwen3-4b`, the dose response is almost flat.** At the swept layer the primary-judge mean goes from 0.63 without
  steering to 0.67 (0.64 to 0.68), 0.65 (0.59 to 0.69) and 0.68 (0.63 to 0.76) at +0.1, +0.2 and +0.3, and to 0.58
  (0.52 to 0.67), 0.57 (0.51 to 0.68) and 0.54 (0.42 to 0.74) at -0.1, -0.2 and -0.3. The unsteered score is a single run
  with no range; the range over readers contains it at five of the six coefficients (at +0.1 all three readers are
  above it, by 0.002 to 0.046). The check judge gives 0.49 without steering and 0.48, 0.50 and 0.53 at the positive
  coefficients and 0.46, 0.43 and 0.43 at the negative ones. The first reader in fit order moves the score the wrong way
  at all three negative coefficients (0.67, 0.68 and 0.74 at -0.1, -0.2 and -0.3 against 0.63 without steering; check
  judge 0.55, 0.57 and 0.60 against 0.49).
- **`qwen3-4b`, the sweep.** The dose slope on the primary judge is between -0.008 and 0.123 at the six swept layers,
  against 0.973 at L36 on the base checkpoint, and it is larger on the base checkpoint at each of the six layers. At L36,
  the layer steered on the base checkpoint, the slope of the post-trained checkpoint is -0.008 on the primary judge and
  0.123 on the check judge. The three layers where the two judges disagree in sign are L1, L15 and L36, which is why the
  judge agreement is 0.500.
- **`qwen3-4b`, prompting works where steering does not.** The positive prompting arms give 0.83 and 0.88 and the
  negative ones 0.40 and 0.47 on the primary judge (check judge 0.80, 0.86, 0.26 and 0.31). The highest steered reader
  (0.76 at +0.3) is below both positive arms, and the mean at each negative coefficient (0.58, 0.57 and 0.54) is higher
  than both negative arms, so steering does not lower the score as far as prompting does. The distinct-1 of the arms
  (0.159 to 0.182) is in the band of the steered cells (0.140 to 0.219).
- **`qwen3-4b`, the cost.** +0.1, +0.2 and -0.1 are fluent (perplexity 32.6, 42.8 and 39.5 against 31.7 unsteered, that is
  1.03, 1.35 and 1.25 times; distinct-1 0.219, 0.212 and 0.210 against 0.224), and the score hardly moves in them. +0.3,
  -0.2 and -0.3 are not fluent (perplexity 69.5, 65.9 and 151.5, that is 2.20, 2.08 and 4.78 times; distinct-1 0.178,
  0.185 and 0.140). At the same relative coefficient this checkpoint degrades less than the post-trained 0.6B and 1.7B
  checkpoints (at -0.3 a perplexity of 151.5 and a distinct-1 of 0.140 against 836.2 and 546.1 and 0.057 and 0.015; at
  +0.3 69.5 and 0.178 against 350.4 and 130.6 and 0.060 and 0.061), and it also moves the score less, so the smaller side
  effects are not read as cleaner steering.
- **The weakest steering of the six Qwen3 checkpoints.** On the primary judge `qwen3-4b` has the smallest change from the
  unsteered score of the six checkpoints at each of the six coefficients: 0.03, 0.02 and 0.05 upward at +0.1, +0.2 and
  +0.3 and 0.06, 0.07 and 0.09 downward at -0.1, -0.2 and -0.3 (from the unrounded means), against at least 0.12, 0.14
  and 0.14 upward and 0.26, 0.42 and 0.24 downward on each of the other five. It also has the lowest R1 sentiment
  accuracy of the six (0.760 [0.727, 0.792] at L23, in [LAT reading on Qwen3](#lat-reading-on-qwen3)). The two facts are
  stated side by side: these runs do not test whether they are related, nor whether the effect appears at another layer,
  with another number of readers, or beyond the relative coefficient of 0.3.
- **Within the pair.** The base checkpoint is above the post-trained one at +0.1 (0.78 against 0.67) and below it at
  -0.1 (0.30 against 0.58), as in the 0.6B and 1.7B pairs. The ranges over readers do not overlap at either coefficient,
  as in the 1.7B pair and unlike the 0.6B pair. The two checkpoints steer at different layers (L36 and L22), so this
  contrasts each at its own selected layer and not at a common one; the slope table gives the common-layer view.
- **No random control.** As above, C1 has no random-direction arm, so these tables do not show how the sentiment vector
  compares with an arbitrary edit of the same size.

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
- **What is and is not supported.** The only intervals that exclude zero are those of 3 of the 10 readers on
  `gpt2-medium`, 3 of the 10 on `gpt2-large` and none on `gpt2-xl`, among the 18 arms per size compared with the unedited
  model and with no correction
  for multiple comparisons; no row separates the adapter, `difference`,
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
GPT-2 sizes and of the Qwen3 0.6B, 1.7B and 4B checkpoints. P9a, P9b and P9c cover all ten models, and P9d ran on `gpt2-medium`, `gpt2-large`, `gpt2-xl` and the six Qwen3
checkpoints (base and post-trained). The 4B pair was run last, on a rented GPU pod, and has its own subsection under P9b, P9c and P9d. `gpt2` is the one model whose label gate
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
| qwen3-4b-base   | 803 / 7,499 / 1,165 | 440 / 373 / 187  | 96 / 715 / 189  |
| qwen3-4b        | 774 / 7,784 / 909   | 408 / 451 / 141  | 135 / 774 / 91  |
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
| qwen3-4b-base   | 1,632        | 1,377       | 42                | 12           | 201              |
| qwen3-4b        | 1,702        | 1,346       | 48                | 23           | 285              |
| gpt2            | 87           | 70          | 0                 | 2            | 15               |
| gpt2-medium     | 301          | 201         | 0                 | 4            | 96               |
| gpt2-large      | 428          | 304         | 0                 | 4            | 120              |
| gpt2-xl         | 540          | 432         | 0                 | 5            | 103              |

- **Known questions are few.** After the guess control a model has 70 to 1,377 known questions: 70 (`gpt2`), 201
  (`gpt2-medium`), 304 (`gpt2-large`), 393 (`qwen3-0.6b`), 432 (`gpt2-xl`), 525 (`qwen3-0.6b-base`), 860
  (`qwen3-1.7b`), 863 (`qwen3-1.7b-base`), 1,346 (`qwen3-4b`) and 1,377 (`qwen3-4b-base`), against 21,988 to 22,028
  questions. They come from PopQA (69 to 803 per model), TriviaQA (1 to 440) and NQ-open (0 to 135); the real award
  questions give at most 1 on the other eight models and 38 (`qwen3-4b-base`) and 29 (`qwen3-4b`) on the 4B pair, and
  no source other than these three has a known question on `gpt2`, `gpt2-medium` or `gpt2-large`. The count rises with
  size inside the GPT-2 family (70, 201, 304, 432) and inside the Qwen3 base (525, 863, 1,377) and post-trained (393,
  860, 1,346) checkpoints.
- **The guess control matters.** It demotes 17 (`gpt2`) to 356 (`qwen3-4b`) of the known questions of a model, 15 to
  285 of them because their stratum is excluded (`popqa:sport` is excluded by default, and so are the strata
  whose constant-guess accuracy is above 0.10).
- **Base against post-trained.** At 1.7B the two checkpoints have 863 and 860 known questions and at 4B 1,377 and 1,346; at 0.6B the base
  checkpoint has 525 and the post-trained one 393. Labels come from each model's own behavior, so this is a count of
  what each model answers under the labeling recipe, not a comparison on common questions.
- **Gate and cutoff.** Nine of the ten models pass the gate (at least 100 guess-adjusted known questions), so the
  known against unknown contrasts are built for them. `gpt2` does not (70), and the run records
  `known_contrasts = false`. The cutoff year is a setting: 2017 for the GPT-2 sizes and for the two 4B runs (the config
  default, which the 4B runs kept) and 2024 for the other four Qwen3 checkpoints. The cutoff estimated from the labels
  is stored for four runs, 2014 for `qwen3-1.7b-base`, 2007 for `qwen3-1.7b` and 2022 for both 4B checkpoints, and does
  not match the setting of its run. The only results below that use the cutoff are the `pre_vs_post_cutoff` rows of the
  four GPT-2 sizes; the post-cutoff contrasts have too few items for the Qwen3 models, the 4B pair included.

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

#### The Qwen3 4B pair

P9b on `qwen3-4b-base` and `qwen3-4b` (runs `p9b/qwen3-4b-base/91d3fe92e1c9` and `p9b/qwen3-4b/5481dab673d4`, made at `89ea7cb` with a clean tree on a rented GPU pod (NVIDIA RTX A6000), on the labels of the P9a run of the same checkpoint) uses the recipe above: five logistic readers on 64 pairs each, one layer per model chosen on the pooled validation AUROC, which is L22 on `qwen3-4b-base` and L25 on `qwen3-4b`. The post-trained run is a retry at batch size 4: the first attempt, at batch size 8, was killed by the out-of-memory killer of the container and is not reported. In P9b the batch size only sets how many prompts go through one forward pass when the last-token activations are collected; it does not enter the draws of pairs, the seeds, the train and test split or the layer choice, so the retry can differ from a batch size 8 run only through the numerical effects of batching and padding. The batch size is a field of the config, so the retry has its own hash by design. A checkpoint is labeled on its own behavior, so the test counts differ between the base and the post-trained row (177 against 172 known PopQA test questions); the rows are never pooled, and a difference between two rows mixes the checkpoint with the question set. The 1.7B rows are those of the runs above, repeated so that the step in size is next to the 4B rows. Every reader value is next to `text_word`, `text_char` and `popularity`. As on the 1.7B checkpoints, `fictional_synthal` and `pre_vs_post_cutoff` are skipped for lack of items, for different reasons by size (the `skipped` field of each summary): at 1.7B `fictional_synthal` has fewer than 10 validation items on one side and `pre_vs_post_cutoff` fewer than 16 train items on one side, on both checkpoints; at 4B `pre_vs_post_cutoff` has fewer than 10 validation items on one side on both checkpoints, and `fictional_synthal` has fewer than 10 test items on one side on `qwen3-4b-base` and fewer than 10 validation items on one side on `qwen3-4b`.

PopQA known against unknown, within relation (the headline), mean AUROC over the five readers:

| Model           | Contrast         | Test known / unknown | Test pairs | Reader | popularity | text_word | text_char |
| --------------- | ---------------- | -------------------- | ---------- | ------ | ---------- | --------- | --------- |
| qwen3-1.7b-base | obscure_real     | 118 / 1,611          | 10,113     | 0.959  | 0.879      | 0.625     | 0.636     |
| qwen3-1.7b-base | obscure_real_pop | 118 / 1,611          | 10,113     | 0.960  | 0.879      | 0.625     | 0.636     |
| qwen3-1.7b      | obscure_real     | 122 / 1,657          | 10,680     | 0.893  | 0.850      | 0.561     | 0.602     |
| qwen3-1.7b      | obscure_real_pop | 122 / 1,657          | 10,680     | 0.845  | 0.850      | 0.561     | 0.602     |
| qwen3-4b-base   | obscure_real     | 177 / 1,529          | 19,228     | 0.957  | 0.905      | 0.568     | 0.555     |
| qwen3-4b-base   | obscure_real_pop | 177 / 1,529          | 19,228     | 0.960  | 0.905      | 0.568     | 0.555     |
| qwen3-4b        | obscure_real     | 172 / 1,586          | 16,894     | 0.927  | 0.863      | 0.534     | 0.553     |
| qwen3-4b        | obscure_real_pop | 172 / 1,586          | 16,894     | 0.925  | 0.863      | 0.534     | 0.553     |

The same contrasts pooled over relations, with the AUROC of a random direction:

| Model           | Contrast         | Reader | Random direction | popularity | text_word | text_char |
| --------------- | ---------------- | ------ | ---------------- | ---------- | --------- | --------- |
| qwen3-1.7b-base | obscure_real     | 0.961  | 0.648            | 0.865      | 0.873     | 0.873     |
| qwen3-1.7b-base | obscure_real_pop | 0.970  | 0.522            | 0.865      | 0.873     | 0.873     |
| qwen3-1.7b      | obscure_real     | 0.927  | 0.529            | 0.834      | 0.867     | 0.885     |
| qwen3-1.7b      | obscure_real_pop | 0.918  | 0.617            | 0.834      | 0.867     | 0.885     |
| qwen3-4b-base   | obscure_real     | 0.940  | 0.589            | 0.861      | 0.815     | 0.801     |
| qwen3-4b-base   | obscure_real_pop | 0.966  | 0.657            | 0.861      | 0.815     | 0.801     |
| qwen3-4b        | obscure_real     | 0.916  | 0.603            | 0.821      | 0.829     | 0.831     |
| qwen3-4b        | obscure_real_pop | 0.943  | 0.441            | 0.821      | 0.829     | 0.831     |

- **Within relation the 4B reader is above popularity and both text baselines on both checkpoints, by a margin that does not grow with size on the base checkpoints and is not monotone on the post-trained ones.** On `obscure_real` it reads 0.957 and 0.927 against a popularity baseline of 0.905 and 0.863, margins of 0.051 and 0.063; `text_word` and `text_char` are at 0.568 and 0.555 and at 0.534 and 0.553. With the popularity bin matched (`obscure_real_pop`) it reads 0.960 and 0.925, margins of 0.055 and 0.061, where `qwen3-1.7b` was 0.006 below popularity. On the base checkpoints the margin over popularity is 0.120, 0.080 and 0.051 at 0.6B, 1.7B and 4B: the reader reads 0.919, 0.959 and 0.957, and the popularity baseline rises faster (0.799, 0.879, 0.905). On the post-trained checkpoints it is 0.157, 0.043 and 0.063, not monotone. The test known sides hold 87, 118 and 177 (base) and 72, 122 and 172 (post-trained) questions and the stored interval covers the pair draws only, so these margins are not measured differences between sizes.
- **Pooled, the 4B reader is above the text baselines on both checkpoints.** On `obscure_real` it reads 0.940 and 0.916 against 0.815 and 0.801 (base) and 0.829 and 0.831 (post-trained), margins over the better text baseline of 0.125 and 0.085; it is also above popularity (0.861 and 0.821) and a random direction (0.589 and 0.603). On `obscure_real_pop` it reads 0.966 and 0.943, and the random direction of the post-trained run reads 0.441, below 0.5. The pooled text baselines fall with size on the base checkpoints (0.944 and 0.941, 0.873 and 0.873, 0.815 and 0.801 at 0.6B, 1.7B and 4B) and the pooled margin of the reader over the better one rises (0.004, 0.089, 0.125); on the post-trained checkpoints it is 0.021, 0.042 and 0.085. The text classifiers are fitted on the labeled questions of each model, so this is a change in the sets as much as in the reader.

Other contrasts, as reader / `text_word` / `text_char` (`popularity` is not defined for them), with the 1.7B pair repeated. The relational contrasts are within relation:

| Contrast                              | qwen3-1.7b-base       | qwen3-1.7b            | qwen3-4b-base         | qwen3-4b              |
| ------------------------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| fictional_generated (within relation) | 1.000 / 0.851 / 0.992 | 0.993 / 0.845 / 0.989 | 1.000 / 0.891 / 0.998 | 0.999 / 0.887 / 0.996 |
| real_vs_synthal (within relation)     | 0.986 / 0.728 / 1.000 | 0.980 / 0.732 / 1.000 | 0.980 / 0.718 / 1.000 | 0.987 / 0.706 / 1.000 |
| real_vs_generated (within relation)   | 0.986 / 0.954 / 0.999 | 0.968 / 0.952 / 0.999 | 0.985 / 0.953 / 0.999 | 0.983 / 0.951 / 0.999 |
| pre_vs_post_cutoff                    | not read              | not read              | not read              | not read              |

The others, pooled:

| Contrast             | qwen3-1.7b-base       | qwen3-1.7b            | qwen3-4b-base         | qwen3-4b              |
| -------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| triviaqa_known       | 0.883 / 0.693 / 0.671 | 0.840 / 0.664 / 0.588 | 0.852 / 0.627 / 0.592 | 0.827 / 0.606 / 0.600 |
| unknowable_detail    | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| false_premise        | 0.974 / 0.977 / 0.975 | 0.951 / 0.974 / 0.967 | 0.992 / 0.979 / 0.977 | 0.976 / 0.965 / 0.965 |
| kuq_ambiguous        | 0.996 / 0.878 / 0.918 | 0.996 / 0.876 / 0.903 | 0.998 / 0.877 / 0.939 | 0.993 / 0.855 / 0.918 |
| kuq_controversial    | 0.993 / 0.975 / 0.987 | 0.983 / 0.971 / 0.985 | 0.996 / 0.968 / 0.996 | 0.998 / 0.969 / 0.996 |
| kuq_counterfactual   | 1.000 / 1.000 / 1.000 | 0.993 / 1.000 / 1.000 | 0.997 / 0.998 / 0.999 | 0.997 / 0.999 / 0.999 |
| kuq_false_assumption | 0.993 / 0.978 / 0.991 | 0.981 / 0.964 / 0.980 | 0.993 / 0.984 / 0.994 | 0.992 / 0.954 / 0.986 |
| kuq_future_unknown   | 1.000 / 0.992 / 0.986 | 0.999 / 0.990 / 0.981 | 0.999 / 0.991 / 0.989 | 0.998 / 0.993 / 0.988 |
| kuq_unsolved_problem | 0.992 / 0.953 / 0.967 | 0.981 / 0.953 / 0.961 | 0.995 / 0.951 / 0.970 | 0.994 / 0.936 / 0.962 |

- **The reader has a margin of 0.059 or more over the better text baseline on two contrasts, as on the other Qwen3 checkpoints.** `triviaqa_known`: the reader reads 0.852 and 0.827 against a better text baseline of 0.627 and 0.606, margins of 0.226 and 0.222 (0.191 and 0.176 at 1.7B), on test known sides of 87 and 84 questions. `kuq_ambiguous`: margins of 0.059 and 0.075 (0.079 and 0.092 at 1.7B). On every other cross-dataset contrast the largest margin over the better baseline is 0.032 (`kuq_unsolved_problem` on `qwen3-4b`).
- **On `false_premise` the 4B reader is above the better text baseline, where the smaller Qwen3 checkpoints were below it.** It reads 0.992 and 0.976 against 0.979 and 0.965, margins of 0.013 and 0.011; on the four smaller checkpoints the reader was below the better baseline by 0.002 to 0.041. A margin of about 0.01 on 107 and 114 known test questions, with no interval over the test questions stored, is not shown to be a difference.
- **On several contrasts the reader does not exceed the text baselines.** `unknowable_detail` is 1.000 for the reader and both baselines on both checkpoints. The two contrasts between kinds of unknown questions (`real_vs_synthal`, `real_vs_generated`) have `text_char` at 1.000 and 0.999, above the reader within relation (0.980 and 0.987 on `real_vs_synthal`, 0.985 and 0.983 on `real_vs_generated`) and pooled.
- **Elsewhere the margin is small.** `fictional_generated` has the reader above `text_char` within relation by 0.002 and 0.003; four KUQ categories (`kuq_controversial`, `kuq_counterfactual`, `kuq_false_assumption`, `kuq_future_unknown`) are within 0.009 of the better baseline in either direction, and `kuq_unsolved_problem` is above it by 0.025 and 0.032. `pre_vs_post_cutoff` is not read on these checkpoints.
- **Read these cross-dataset rows as a format contrast too.** The better text baseline is at 0.962 or above on every cross-dataset row of the 4B pair except `triviaqa_known` (0.627 and 0.606) and `kuq_ambiguous` (0.939 and 0.918), so a reader between 0.976 and 1.000 on the others (as the tables show them) does not show that it reads knowledge.
- **What this does not show.** As above: popularity and the wording of the question are the only surface controls, each value is one set of five readers on one labeling run per checkpoint, and the question sets differ between models. Whether the reader predicts the errors of the 4B checkpoints better than their output confidence is the question of P9c.

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

#### The Qwen3 4B pair

P9c on `qwen3-4b-base` and `qwen3-4b` (runs `p9c/qwen3-4b-base/ab86a6dc49d2` and `p9c/qwen3-4b/4307bb435ab8`, made at `89ea7cb` with a clean tree on a rented GPU pod (NVIDIA RTX A6000), batch size 8, on the labels of the P9a run of the same checkpoint) has the same predictors and the same sizes of held-out sets (1,937 PopQA, 191 TriviaQA and 210 NQ-open questions) as above. The probe and `lat` read L23 on both checkpoints (chosen on the validation split of P9c). The 1.7B columns are those of the runs above, repeated so that the step in size is next to the 4B columns. A checkpoint answers different questions correctly (correct answers on PopQA: 264, 222, 337 and 297 of 1,937 in the column order below), so columns are separate rows of evidence and are never pooled.

PopQA, within relation, AUROC with its interval:

| Predictor          | qwen3-1.7b-base      | qwen3-1.7b           | qwen3-4b-base        | qwen3-4b             |
| ------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| answer_logprob     | 0.883 [0.857, 0.907] | 0.855 [0.822, 0.887] | 0.914 [0.895, 0.931] | 0.873 [0.847, 0.899] |
| probe              | 0.881 [0.856, 0.905] | 0.853 [0.820, 0.885] | 0.886 [0.858, 0.910] | 0.869 [0.840, 0.895] |
| first_token_prob   | 0.879 [0.853, 0.902] | 0.842 [0.807, 0.875] | 0.910 [0.888, 0.929] | 0.864 [0.838, 0.891] |
| neg_entropy        | 0.879 [0.853, 0.903] | 0.847 [0.812, 0.880] | 0.905 [0.884, 0.923] | 0.869 [0.843, 0.894] |
| mean_token_logprob | 0.862 [0.831, 0.889] | 0.841 [0.807, 0.874] | 0.897 [0.875, 0.918] | 0.870 [0.842, 0.894] |
| consistency        | 0.803 [0.763, 0.841] | 0.807 [0.760, 0.846] | 0.860 [0.830, 0.890] | 0.832 [0.800, 0.861] |
| lat                | 0.848 [0.815, 0.877] | 0.814 [0.773, 0.850] | 0.879 [0.854, 0.902] | 0.859 [0.832, 0.885] |
| p_true             | 0.776 [0.726, 0.821] | 0.772 [0.714, 0.825] | 0.794 [0.757, 0.831] | 0.763 [0.714, 0.811] |
| popularity         | 0.776 [0.736, 0.815] | 0.749 [0.700, 0.795] | 0.813 [0.782, 0.846] | 0.782 [0.745, 0.822] |

PopQA pooled over relations, as AUROC / AURC (lower AURC is better), and the risk at 20 percent coverage (error rate among the answered questions, with its interval):

| Predictor                            | qwen3-1.7b-base      | qwen3-1.7b           | qwen3-4b-base        | qwen3-4b             |
| ------------------------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| probe                                | 0.940 / 0.635        | 0.939 / 0.678        | 0.933 / 0.572        | 0.933 / 0.608        |
| answer_logprob                       | 0.910 / 0.655        | 0.919 / 0.710        | 0.919 / 0.579        | 0.903 / 0.629        |
| neg_entropy                          | 0.899 / 0.661        | 0.906 / 0.715        | 0.917 / 0.583        | 0.910 / 0.629        |
| lat                                  | 0.879 / 0.679        | 0.857 / 0.731        | 0.878 / 0.610        | 0.862 / 0.656        |
| template                             | 0.843 / 0.688        | 0.873 / 0.723        | 0.804 / 0.641        | 0.829 / 0.666        |
| popularity                           | 0.648 / 0.748        | 0.640 / 0.784        | 0.671 / 0.688        | 0.650 / 0.724        |
| error rate (risk at full coverage)   | 0.864                | 0.885                | 0.826                | 0.847                |
| risk at 20% coverage, probe          | 0.456 [0.399, 0.523] | 0.531 [0.477, 0.593] | 0.376 [0.320, 0.430] | 0.418 [0.369, 0.474] |
| risk at 20% coverage, answer_logprob | 0.492 [0.436, 0.546] | 0.552 [0.495, 0.603] | 0.384 [0.327, 0.433] | 0.443 [0.389, 0.497] |

TriviaQA and NQ-open, pooled AUROC with its interval:

| Predictor                | qwen3-1.7b-base      | qwen3-1.7b           | qwen3-4b-base        | qwen3-4b             |
| ------------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| TriviaQA answer_logprob  | 0.819 [0.758, 0.878] | 0.832 [0.771, 0.888] | 0.831 [0.769, 0.882] | 0.731 [0.664, 0.797] |
| TriviaQA neg_entropy     | 0.843 [0.784, 0.894] | 0.798 [0.730, 0.864] | 0.816 [0.753, 0.866] | 0.741 [0.674, 0.806] |
| TriviaQA probe           | 0.811 [0.753, 0.868] | 0.808 [0.743, 0.865] | 0.785 [0.722, 0.846] | 0.789 [0.724, 0.846] |
| TriviaQA lat             | 0.674 [0.601, 0.750] | 0.686 [0.608, 0.756] | 0.766 [0.698, 0.829] | 0.752 [0.682, 0.818] |
| TriviaQA p_true          | 0.772 [0.701, 0.840] | 0.734 [0.657, 0.805] | 0.801 [0.737, 0.861] | 0.816 [0.756, 0.874] |
| TriviaQA correct answers | 78 of 191            | 63 of 191            | 107 of 191           | 99 of 191            |
| NQ-open answer_logprob   | 0.790 [0.697, 0.867] | 0.834 [0.724, 0.923] | 0.714 [0.618, 0.802] | 0.780 [0.698, 0.851] |
| NQ-open neg_entropy      | 0.745 [0.659, 0.831] | 0.750 [0.668, 0.828] | 0.761 [0.689, 0.827] | 0.722 [0.631, 0.804] |
| NQ-open probe            | 0.812 [0.742, 0.878] | 0.879 [0.811, 0.941] | 0.749 [0.672, 0.819] | 0.774 [0.703, 0.845] |
| NQ-open lat              | 0.848 [0.777, 0.913] | 0.861 [0.771, 0.939] | 0.741 [0.666, 0.814] | 0.798 [0.729, 0.871] |
| NQ-open p_true           | 0.694 [0.590, 0.787] | 0.599 [0.458, 0.722] | 0.693 [0.613, 0.770] | 0.781 [0.707, 0.847] |
| NQ-open correct answers  | 38 of 210            | 20 of 210            | 46 of 210            | 38 of 210            |

What `probe` and `lat` add to the output confidence (the logistic stacker of the five confidence columns against the same stacker plus the predictor, as a paired AUROC difference with its interval; pooled AUROCs):

| Predictor                 | qwen3-1.7b-base         | qwen3-1.7b              | qwen3-4b-base           | qwen3-4b                |
| ------------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| PopQA confidence stack    | 0.900                   | 0.932                   | 0.922                   | 0.923                   |
| PopQA + lat               | +0.002 [-0.002, +0.006] | -0.001 [-0.003, +0.002] | -0.001 [-0.002, +0.001] | +0.001 [-0.001, +0.004] |
| PopQA + probe             | +0.044 [+0.028, +0.061] | +0.016 [+0.006, +0.026] | +0.020 [+0.010, +0.031] | +0.016 [+0.009, +0.024] |
| TriviaQA confidence stack | 0.858                   | 0.837                   | 0.847                   | 0.792                   |
| TriviaQA + lat            | -0.006 [-0.018, +0.006] | -0.005 [-0.037, +0.028] | 0.000 [-0.002, +0.002]  | +0.004 [-0.004, +0.012] |
| TriviaQA + probe          | +0.013 [-0.004, +0.029] | +0.020 [-0.017, +0.058] | +0.010 [-0.009, +0.032] | +0.012 [-0.001, +0.026] |
| NQ-open confidence stack  | 0.800                   | 0.866                   | 0.798                   | 0.807                   |
| NQ-open + lat             | +0.047 [+0.015, +0.085] | +0.028 [-0.004, +0.060] | 0.000 [-0.029, +0.033]  | +0.027 [-0.004, +0.060] |
| NQ-open + probe           | +0.036 [+0.007, +0.066] | +0.040 [+0.002, +0.077] | +0.008 [-0.010, +0.027] | +0.013 [-0.013, +0.042] |

- **Within relation `lat` is below the output confidence on both 4B checkpoints, by a gap that is smaller than at 0.6B on both, equal to the 1.7B gap on the base checkpoint and smaller than it on the post-trained one, and its interval is separated from the popularity baseline.** It reads 0.879 [0.854, 0.902] on `qwen3-4b-base` and 0.859 [0.832, 0.885] on `qwen3-4b`, against 0.914 [0.895, 0.931] and 0.873 [0.847, 0.899] for `answer_logprob`, which is the best of the six confidence scores on both. The gap is -0.035 and -0.014 (-0.055 and -0.035 on the 0.6B and 1.7B base checkpoints, -0.086 and -0.041 on the post-trained ones), so on the base checkpoint it is the same at 1.7B and 4B at three decimals and on the post-trained ones it shrinks at each step; the intervals overlap at both and no paired interval is stored. The popularity baseline is 0.813 [0.782, 0.846] and 0.782 [0.745, 0.822]; the interval of `lat` lies above it on both (by 0.008 and 0.010 between the ends), where on the smaller checkpoints the intervals overlapped or touched. Within relation `lat` rises with size on both families (0.787, 0.848, 0.879 on the base checkpoints; 0.757, 0.814, 0.859 on the post-trained ones, at 0.6B, 1.7B and 4B), and so do `answer_logprob` (0.841, 0.883, 0.914; 0.843, 0.855, 0.873) and the popularity baseline (0.696, 0.776, 0.813; 0.637, 0.749, 0.782).
- **The probe is below `answer_logprob` within relation and above it pooled, as before.** Within relation it reads 0.886 [0.858, 0.910] and 0.869 [0.840, 0.895] against 0.914 and 0.873. Pooled it reads 0.933 on both against 0.919 and 0.903, and the AURC is 0.572 and 0.608 against 0.579 and 0.629 (lower is better). The risk at 20 percent coverage is 0.376 [0.320, 0.430] and 0.418 [0.369, 0.474] for the probe against 0.384 [0.327, 0.433] and 0.443 [0.389, 0.497] for `answer_logprob`: each within the other's interval. Added to the confidence stack the probe raises the pooled AUROC by +0.020 [+0.010, +0.031] and +0.016 [+0.009, +0.024], with intervals above zero; the stack itself is 0.922 and 0.923 (0.900 and 0.932 at 1.7B), and the probe's gain on the base checkpoint is smaller at 4B than at 0.6B and 1.7B (+0.044 on both). `lat` adds -0.001 [-0.002, +0.001] and +0.001 [-0.001, +0.004], so it adds nothing that the confidence does not carry on PopQA.
- **Pooled, `lat` is above the accuracy rate of the relation on both 4B checkpoints.** It reads 0.878 and 0.862 against 0.804 and 0.829 for `template`, and below `answer_logprob` (0.919 and 0.903). On the smaller checkpoints only `qwen3-1.7b-base` was above `template`.
- **The 4B checkpoints make fewer errors on PopQA than the 1.7B ones.** The error rate of the greedy answer is 0.826 and 0.847 on the 4B checkpoints, against 0.864 and 0.885 at 1.7B, and the AURC of `answer_logprob` is 0.579 and 0.629 against 0.655 and 0.710.
- **TriviaQA and NQ-open.** The sets hold 107 and 99 correct TriviaQA answers and 46 and 38 correct NQ-open answers, and the intervals are 0.11 to 0.19 wide, so the ordering of predictors there is within the intervals. On TriviaQA `lat` reads 0.766 [0.698, 0.829] and 0.752 [0.682, 0.818] against 0.831 and 0.731 for `answer_logprob`; on NQ-open it reads 0.741 and 0.798 against 0.714 and 0.780. Every increment of `lat` and of the probe over the confidence stack on these two sets has an interval that includes zero (the probe on TriviaQA at `qwen3-4b` has a lower bound of -0.001), where at 1.7B the NQ-open increments of `lat` (+0.047) and the probe (+0.036, +0.040) excluded it. The self-grade `p_true` is the highest predictor on TriviaQA at `qwen3-4b` (0.816 [0.756, 0.874]), with an interval that overlaps that of the probe and `answer_logprob`.
- **What this does not show.** Each cell is one run of one checkpoint, the sets of correct answers are small outside PopQA, and the increments are pooled (no within-relation increment is stored). The result is that `lat` does not predict errors better than the output confidence on the 4B checkpoints either, and that the gap is smaller than at 0.6B in point estimate on both families and smaller than at 1.7B on the post-trained checkpoint only (the same on the base one); it does not say that the reader is useless, since it is label-free and its interval is separated from the popularity baseline within relation.

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

#### The Qwen3 4B pair

P9d on `qwen3-4b-base` and `qwen3-4b` (runs `p9d/qwen3-4b-base/7fe658a89920` and `p9d/qwen3-4b/c7949f5432e6`, made at `89ea7cb` with a clean tree on a rented GPU pod (NVIDIA RTX A6000), batch size 16, on the labels of the P9a run of the same checkpoint) uses the recipe above with the `fewshot` template. `qwen3-4b` has a second run with the `concept` template (`p9d/qwen3-4b/cc4024fe5af1`, made later at `bab5b75` with a clean tree on a rented GPU pod (NVIDIA RTX A5000), batch size 16, on the labels of the same P9a run), compared with the `concept` rows of the smaller post-trained checkpoints; the base checkpoint has the `fewshot` template only, as at the other sizes. The base checkpoint is scored on the shift of the entropy and the post-trained one on abstention, which are different quantities. The stored selectivity tables of these three runs were made by the current code (the interval oriented by the sign of the unknown mean), so they are used as stored. The abstention indicator is still a phrase match whose 100-row sample (`hand_check.csv`) carries no human labels. P9d stores no perplexity or distinct-1 of the steered text, so this section has no fluency measure for the 4B runs; the side effects it shows are the movement of the known items and the false refusals. The 1.7B rows below are those of the runs above, repeated.

Headline cell (unknown side of `obscure_real`), base checkpoints, shift mode: slope of the entropy per unit of relative coefficient, same columns as above.

| Run (steered layer) | Validation AUROC | Direction, entropy      | random0 / 1 / 2, entropy | Min gap bound | Distinguishable |
| ------------------- | ---------------- | ----------------------- | ------------------------ | ------------- | --------------- |
| 1.7b base (L20)     | 0.966            | -5.882 [-6.095, -5.655] | 0.155 / 0.941 / 0.023    | 4.722         | yes             |
| 4b base (L22)       | 0.961            | -5.226 [-5.508, -4.960] | 0.121 / 0.062 / -0.363   | 4.584         | yes             |

Headline cell, post-trained checkpoints, abstention mode, `fewshot` and `concept` templates: slope of the share of abstentions per unit of relative coefficient.

| Run (steered layer)      | Validation AUROC | Direction, abstention   | random0 / 1 / 2, abstention | Min gap bound | Distinguishable |
| ------------------------ | ---------------- | ----------------------- | --------------------------- | ------------- | --------------- |
| 1.7b post, fewshot (L24) | 0.943            | 0.040 [-0.031, 0.114]   | 0.388 / -0.017 / 0.095      | -0.481        | no              |
| 4b post, fewshot (L29)   | 0.913            | -0.648 [-0.779, -0.517] | 0.057 / -0.286 / -0.162     | 0.255         | yes             |
| 1.7b post, concept (L12) | 0.869            | -1.112 [-1.224, -1.002] | -0.348 / 0.483 / 0.174      | 0.514         | yes             |
| 4b post, concept (L24)   | 0.870            | -1.221 [-1.326, -1.119] | 0.171 / 0.050 / -0.324      | 0.764         | yes             |

Selectivity of the 4B runs, same columns as above (the 1.7B and 0.6B rows are in the tables above):

| Run              | Type                | Outcome      | n (k / u) | Known  | Unknown | Gap [95% interval]      | Selective |
| ---------------- | ------------------- | ------------ | --------- | ------ | ------- | ----------------------- | --------- |
| 4b base          | obscure_real        | entropy      | 100 / 100 | -3.253 | -5.226  | 1.972 [1.424, 2.521]    | yes       |
| 4b base          | obscure_real        | gold logprob | 100 / 100 | 0.557  | -3.866  | 3.309 [2.333, 4.329]    | yes       |
| 4b base          | fictional_generated | entropy      | 100 / 100 | -3.253 | -4.363  | 1.110 [0.562, 1.692]    | yes       |
| 4b base          | unknowable_detail   | entropy      | 100 / 60  | -2.870 | -3.158  | 0.288 [-0.163, 0.706]   | no        |
| 4b base          | false_premise       | entropy      | 100 / 83  | -2.870 | -2.461  | -0.410 [-0.851, -0.005] | no        |
| 4b post, fewshot | obscure_real        | abstention   | 100 / 100 | -0.212 | -0.648  | 0.436 [0.279, 0.598]    | yes       |
| 4b post, fewshot | fictional_generated | abstention   | 100 / 100 | -0.212 | -0.576  | 0.364 [0.224, 0.505]    | yes       |
| 4b post, fewshot | unknowable_detail   | abstention   | 100 / 60  | -0.048 | -0.206  | 0.159 [0.036, 0.284]    | yes       |
| 4b post, fewshot | false_premise       | abstention   | 100 / 83  | -0.048 | -0.209  | 0.162 [0.051, 0.279]    | yes       |
| 4b post, concept | obscure_real        | abstention   | 100 / 100 | -0.379 | -1.221  | 0.843 [0.683, 0.991]    | yes       |
| 4b post, concept | fictional_generated | abstention   | 100 / 100 | -0.379 | -1.071  | 0.693 [0.531, 0.860]    | yes       |
| 4b post, concept | unknowable_detail   | abstention   | 100 / 60  | -0.193 | -0.635  | 0.442 [0.253, 0.640]    | yes       |
| 4b post, concept | false_premise       | abstention   | 100 / 83  | -0.193 | -0.602  | 0.410 [0.240, 0.572]    | yes       |

Base checkpoints, the curve behind the slopes: shift of the entropy on `obscure_real` at the two ends of the grid for the direction (95 percent interval) and the range of the three random directions.

| Run       | Side    | Direction at -0.4       | Direction at +0.4       | Random at -0.4 | Random at +0.4 |
| --------- | ------- | ----------------------- | ----------------------- | -------------- | -------------- |
| 1.7b base | known   | +3.861 [+3.630, +4.097] | -0.140 [-0.338, +0.051] | -0.04 to +0.20 | +0.17 to +0.49 |
| 1.7b base | unknown | +1.618 [+1.447, +1.813] | -2.947 [-3.155, -2.715] | -0.35 to +0.07 | +0.09 to +0.39 |
| 4b base   | known   | +2.611 [+2.288, +2.942] | -0.047 [-0.175, +0.095] | +0.17 to +0.30 | +0.22 to +0.25 |
| 4b base   | unknown | +1.356 [+1.186, +1.539] | -2.747 [-2.989, -2.500] | +0.07 to +0.29 | +0.02 to +0.17 |

Post-trained 4B checkpoint, both templates: abstention on the `obscure_real` items at the coefficient chosen on the calibration items (the end of the grid), as the change in the share of abstentions from the unsteered run (95 percent interval). With the `fewshot` template the share of abstentions on the unknown items goes from 0.27 to 0.53 and the false refusals on the known items from 0.05 to 0.19; with the `concept` template the share goes from 0.27 to 0.88 and the false refusals from 0.05 to 0.31.

|                    | 4b post, fewshot     | 4b post, concept     |
| ------------------ | -------------------- | -------------------- |
| Coefficient        | -0.4                 | -0.4                 |
| Direction, unknown | +0.26 [+0.18, +0.35] | +0.61 [+0.51, +0.70] |
| random0, unknown   | -0.07 [-0.13, -0.02] | -0.02 [-0.08, +0.04] |
| random1, unknown   | +0.07 [+0.03, +0.13] | -0.09 [-0.17, -0.01] |
| random2, unknown   | +0.01 [-0.03, +0.05] | +0.11 [+0.06, +0.18] |
| Direction, known   | +0.14 [+0.08, +0.22] | +0.26 [+0.18, +0.35] |

- **Distinguishable on both 4B checkpoints.** The base direction moves the entropy of the unknown questions by -5.226 [-5.508, -4.960] per unit coefficient; the largest absolute slope of a random arm is 0.363 and the smallest gap bound is 4.584. The post-trained direction moves abstention by -0.648 [-0.779, -0.517] against a largest random slope of 0.286, with a smallest gap bound of 0.255. Among the post-trained `fewshot` runs only `qwen3-4b` is distinguishable (smallest gap bounds -0.481 at 1.7B and -0.738 at 0.6B). With the `concept` template the direction moves abstention by -1.221 [-1.326, -1.119] against a largest random slope of 0.324, with a smallest gap bound of 0.764; among the `concept` runs `qwen3-4b` and `qwen3-1.7b` (0.514) are distinguishable and `qwen3-0.6b` (-0.674) is not. Two random arms also move abstention on the unknown `obscure_real` items of `qwen3-4b`: `random1` has a slope of -0.286 [-0.395, -0.181] and `random2` of -0.162 [-0.264, -0.069], both intervals exclude zero, and both slopes are smaller than that of the direction (0.44 and 0.25 of it); `random0` has +0.057 [-0.019, +0.138]. On the other types the unknown-side random arms that exclude zero are `random1` on `fictional_generated` (-0.212 [-0.314, -0.114]) and `random1` and `random2` on `false_premise` (-0.129 [-0.215, -0.055] and -0.092 [-0.164, -0.029]); on `unknowable_detail` the intervals of the random arms reach zero at an end. On the known `obscure_real` items `random1` also excludes zero (-0.057 [-0.119, -0.010], against -0.212 for the direction). In the `concept` run `random0` (+0.171 [+0.067, +0.269]) and `random2` (-0.324 [-0.460, -0.202]) also exclude zero on the unknown `obscure_real` items, at 0.14 and 0.27 of the slope of the direction, and `random1` is +0.050 [-0.038, +0.138]; on the known items no random arm has a slope interval that excludes zero, but that of `random1` starts at 0.000 ([0.000, 0.100]), and `random2` at -0.4 raises the known abstention by +0.04 [+0.01, +0.08], so the random arms are not without effect there.
- **Selective is a separate result.** `qwen3-4b` with the `fewshot` template is selective on all four types (gaps 0.436 [0.279, 0.598] on `obscure_real`, 0.364 on `fictional_generated`, 0.159 [0.036, 0.284] on `unknowable_detail` and 0.162 [0.051, 0.279] on `false_premise`), and with the `concept` template it is selective on all four with larger gaps (0.843 [0.683, 0.991] on `obscure_real`, 0.693 [0.531, 0.860] on `fictional_generated`, 0.442 [0.253, 0.640] on `unknowable_detail` and 0.410 [0.240, 0.572] on `false_premise`). `qwen3-4b-base` is selective on `obscure_real` (entropy gap 1.972 [1.424, 2.521]) and `fictional_generated` (1.110 [0.562, 1.692]) and not on `unknowable_detail` (0.288 [-0.163, 0.706]); on `false_premise` the gap is -0.410 [-0.851, -0.005], an interval below zero, so the entropy of the known items moves more than that of the unknown ones there. The base checkpoints are selective on 4, 2 and 2 of the four types at 0.6B, 1.7B and 4B, which is not a trend in size.
- **The known items move too.** On `qwen3-4b-base` the entropy slope on the known items is -3.253 [-3.721, -2.810], about 62 percent of the unknown-side slope (-5.226). The curve shows why: the entropy of the known items rises by +2.611 at -0.4 and changes by -0.047 at +0.4, while the unknown items rise by +1.356 and fall by -2.747; the random directions move the known items by at most +0.30 at the ends of the grid. On `qwen3-4b` the abstention slope on the known items is -0.212 [-0.310, -0.124], about a third of the unknown-side slope, and at the coefficient -0.4 the false refusals on the known questions rise by +0.14 [+0.08, +0.22], against +0.26 [+0.18, +0.35] on the unknown ones. The random arms move the unknown questions by -0.07, +0.07 and +0.01, so at this coefficient the largest random effect is 0.07 in absolute value against +0.26 for the direction. The coefficient is the end of the grid, so the grid does not show where the effect stops growing. With the `concept` template the abstention slope on the known items is -0.379 [-0.507, -0.267], about 31 percent of the unknown-side slope (-1.221), and at the coefficient -0.4 the false refusals on the known questions rise by +0.26 [+0.18, +0.35] (from 0.05 to 0.31), against +0.61 [+0.51, +0.70] on the unknown ones; the random arms move the unknown questions by -0.02, -0.09 and +0.11, so the largest random effect is 0.11 in absolute value.
- **Gold-answer log-probability (base).** On the known items the slope is 0.557 [0.099, 0.996]; on the unknown items it is -3.866 [-4.774, -2.932]; the gap is 3.309 [2.333, 4.329] (4.290 at 1.7B, 2.861 at 0.6B). The absolute gaps of the direction against the three random arms have lower bounds of 2.830, 2.637 and 2.792, all above zero, so on the 4B base checkpoint the gold-answer log-probability moves distinguishably as well, as on `qwen3-1.7b-base` (lower bounds 3.318 to 3.613 in its stored dose table) and unlike the GPT-2 sizes, where the gaps include zero or clear it only just. Like the entropy, it is defined only on the questions that have a gold answer.
- **Size.** The entropy slopes of the base checkpoints (-4.099, -5.882 and -5.226 at 0.6B, 1.7B and 4B) are in the units of each model's own entropy and are not compared across models. The validation AUROC of the direction at the steered layer is 0.934, 0.966 and 0.961 on the base checkpoints, 0.922, 0.943 and 0.913 on the post-trained `fewshot` ones and 0.795, 0.869 and 0.870 on the post-trained `concept` ones, so it rises from 0.6B to 1.7B in all three series, falls from 1.7B to 4B in two of them (0.966 to 0.961 and 0.943 to 0.913) and is level in the `concept` series (0.869 to 0.870).
- **Template.** On `qwen3-4b` the `concept` direction moves abstention more than the `fewshot` one: the slope on the unknown `obscure_real` items is -1.221 [-1.326, -1.119] against -0.648 [-0.779, -0.517] and the selectivity gap is 0.843 [0.683, 0.991] against 0.436 [0.279, 0.598], with a lower validation AUROC (0.870 at L24 against 0.913 at L29). At the end of the grid the `concept` run raises abstention on the unknown questions by +0.61 and the false refusals on the known ones by +0.26, against +0.26 and +0.14 for `fewshot`. At 1.7B the `concept` slope is also larger than the `fewshot` one (-1.112 against 0.040), and at 0.6B neither run is distinguishable. The two 4B runs differ in the steered layer, the direction and the prompt format (the `concept` template is read in the chat format), so this is a contrast between two runs and not a test of the wording.
- **What this does and does not show.** On the headline outcome the direction moves abstention (post-trained, `fewshot` or `concept`) or entropy (base) more than a random direction of the same norm does, at 4B on both checkpoints and with both templates, and at 1.7B on the base checkpoint and with the `concept` template (the 1.7B post-trained `fewshot` run is not distinguishable). It is not shown to be a control of what the model knows: the known items move too (false refusals +0.14 with `fewshot` and +0.26 with `concept`), two of the three random arms also move abstention on `qwen3-4b` in each run, the effect is largest at the end of the grid, and the abstention indicator is a phrase match whose errors were not measured. The runs are one per cell, and base and post-trained results answer different questions.

## Approximate answers: range against exact quantities (P10)

P10 asks how a model handles a quantity question whose sourced answer is a range, such as the height of a dog breed or the gestation period of a species, against one whose answer is a single value, such as the height of a building or the running time of a film. P10a measures what the model answers: one number, a range or an abstention, with and without the instruction to give one exact number. P10b reads the hidden states of the same questions: it asks whether a direction fitted on the questions of some families separates range questions from exact questions in a family it was not fitted on, and sets every reader beside surface baselines and a random-direction control. The settings below are those of `config.json` of each run; the code is `p10a_quantities.py` and `p10b_reading.py`.

Questions come from Wikidata (CC0), with the original unit and bounds of each statement and at least 5 sitelinks per entity (at least 20 for the films, the floor of their query, and at least 100 for the people). The snapshot dates and the hashes of the queries are in `summary.json` of each run (retrieved 2026-10-17; the people heights 2026-10-18). Three families give 120 range questions and 120 exact questions each, 720 in all:

| Family   | Range side                                                                         | Exact side                                                           |
| -------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| duration | gestation period of a taxon: "How long is the gestation period of the X, in days?" | running time of a film: "How long is the film X, in minutes?"        |
| height   | dog breed: "How tall is a X, in centimetres?"                                      | skyscraper, tower or tower block: "How tall is the X, in metres?"    |
| wingspan | bird species: "What is the wingspan of the X, in centimetres?"                     | aircraft model: "What is the wingspan of the X aircraft, in metres?" |

In all three families the range side is a living thing and the exact side is an artifact, and the wording differs by side (the unit word, and in the height family the article). A model or a reader that separates the sides can therefore be reacting to the kind of answer, to the animacy of the subject or to the wording. Two controls are in the runs. The first is a set of 120 people with one exact height in Wikidata ("How tall is X, in centimetres?"), which are living things with an exact answer: P10a asks them too, and P10b scores them as test questions that never train a reader. The people were drawn with a stricter popularity floor (100 sitelinks, against 5 for the breeds and the buildings), which is set in `datasets/quantities.py` and not in `config.json`. A model or a reader that reacts to a range-valued answer treats them like the buildings; one that reacts to a living subject treats them like the breeds. The comparison is a diagnostic and not a clean contrast, because people heights and breed heights may also differ in how familiar they are to a model. The second control is the set of surface baselines of P10b, below.

Intervals are 95 percent intervals. In P10a a greedy rate has a Wilson interval, and a sampled rate (five samples per question at temperature 1.0) has an item cluster bootstrap interval, 2,000 resamples of the questions with all their samples, replaced by the Wilson interval on the number of questions when the rate is exactly 0 or 1, where every resample agrees. The gap between kinds resamples the questions within each kind. In P10b the AUROC intervals of the transfer, baseline and people files are item bootstrap intervals with 1,000 resamples of the test questions; the intervals of the curves file resample the five draws of training pairs and are not comparable with them. At an AUROC of 1.000 the item bootstrap interval is [1.000, 1.000] because every resample is perfectly separated; with 22 to 30 test questions per side it is not a measure of how certain the value is. Each cell is one run, so no interval includes variation between runs. Rows follow the order of the P9 tables (Qwen3 0.6B, 1.7B, 4B, then the GPT-2 sizes), and a base checkpoint and its post-trained one are separate rows. The runs reported here are P10a and P10b on the ten checkpoints of the other tables, 20 runs in all: `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base`, `qwen3-1.7b`, `qwen3-4b-base`, `qwen3-4b`, `gpt2`, `gpt2-medium`, `gpt2-large` and `gpt2-xl` (the run directories are in the hash tables of [Run identity](#run-identity)). The Qwen3 4B pair was run at `e7295fc` and the other eight checkpoints at `d09822b`, on rented GPU pods; the source of the two commits is the same. P10c, in its own subsection below, adds ten runs on the same ten checkpoints, made at `bab5b75`.

### Behavior (P10a)

P10a asks each of the 840 questions (the 720 and the 120 people) under two conditions. In the free condition the model answers the question; in the forced condition the prompt tells it to give one exact number ("Answer each question with one exact number." before the examples, "Answer with one exact number only." in the chat format). The few-shot format, shared by all models, has six worked examples in two orders: range last (the last example has a range answer) and exact last (the same list reversed), so that only the form of the last example differs. The chat format (the chat template of the tokenizer with thinking disabled, no examples) is used for post-trained checkpoints only. Each question gets one greedy completion and five sampled ones (temperature 1.0) of up to 32 new tokens. A few-shot completion is read up to its first newline and a chat completion from its first non-empty line. An answer is a point (one number), a range (two numbers), an abstention or none; it is also flagged when it marks itself as approximate, and it is correct when the value or the range fits the sourced range within 5 percent. The range share of a group is the share of its answers that are ranges (the hedge share), and the hedge gap is the range share on the range questions minus the range share on the exact questions. In P10a the batch size (4 for `qwen3-4b-base`, 8 for the other nine runs) is the number of prompts per generation call and the sampler is seeded per call, so it changes which sampled completions are drawn but not the questions, the prompts, the parser or the greedy decoding rule, and the batch size 4 run has its own config hash.

Free condition, pooled over the three families (360 range and 360 exact questions): range share on each kind and the hedge gap, greedy and sampled:

| Model           | Prompt               | Greedy: range items  | Greedy: exact items  | Greedy: gap          | Sampled: range items | Sampled: exact items | Sampled: gap         |
| --------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| qwen3-0.6b-base | few-shot, range last | 0.989 [0.972, 0.996] | 0.153 [0.119, 0.194] | 0.836 [0.797, 0.875] | 0.738 [0.714, 0.762] | 0.121 [0.103, 0.141] | 0.617 [0.587, 0.647] |
| qwen3-0.6b-base | few-shot, exact last | 0.989 [0.972, 0.996] | 0.019 [0.009, 0.040] | 0.969 [0.950, 0.986] | 0.637 [0.610, 0.664] | 0.093 [0.077, 0.111] | 0.544 [0.513, 0.575] |
| qwen3-0.6b      | few-shot, range last | 0.675 [0.625, 0.721] | 0.000 [0.000, 0.011] | 0.675 [0.625, 0.720] | 0.398 [0.371, 0.426] | 0.031 [0.023, 0.039] | 0.367 [0.338, 0.396] |
| qwen3-0.6b      | few-shot, exact last | 0.353 [0.305, 0.403] | 0.000 [0.000, 0.011] | 0.353 [0.306, 0.403] | 0.289 [0.264, 0.315] | 0.011 [0.007, 0.016] | 0.278 [0.253, 0.304] |
| qwen3-0.6b      | chat                 | 0.106 [0.078, 0.142] | 0.000 [0.000, 0.011] | 0.106 [0.075, 0.136] | 0.252 [0.226, 0.280] | 0.001 [0.000, 0.002] | 0.252 [0.226, 0.279] |
| qwen3-1.7b-base | few-shot, range last | 0.844 [0.803, 0.878] | 0.000 [0.000, 0.011] | 0.844 [0.808, 0.881] | 0.622 [0.592, 0.652] | 0.032 [0.024, 0.041] | 0.589 [0.559, 0.619] |
| qwen3-1.7b-base | few-shot, exact last | 0.742 [0.694, 0.784] | 0.000 [0.000, 0.011] | 0.742 [0.697, 0.789] | 0.567 [0.536, 0.598] | 0.017 [0.011, 0.023] | 0.550 [0.519, 0.582] |
| qwen3-1.7b      | few-shot, range last | 0.275 [0.231, 0.323] | 0.000 [0.000, 0.011] | 0.275 [0.231, 0.325] | 0.229 [0.193, 0.267] | 0.000 [0.000, 0.011] | 0.229 [0.193, 0.267] |
| qwen3-1.7b      | few-shot, exact last | 0.256 [0.213, 0.303] | 0.000 [0.000, 0.011] | 0.256 [0.211, 0.303] | 0.204 [0.169, 0.242] | 0.000 [0.000, 0.011] | 0.204 [0.169, 0.242] |
| qwen3-1.7b      | chat                 | 0.197 [0.159, 0.241] | 0.000 [0.000, 0.011] | 0.197 [0.158, 0.242] | 0.196 [0.162, 0.232] | 0.000 [0.000, 0.011] | 0.196 [0.162, 0.232] |
| qwen3-4b-base   | few-shot, range last | 0.831 [0.788, 0.866] | 0.000 [0.000, 0.011] | 0.831 [0.792, 0.869] | 0.642 [0.611, 0.669] | 0.007 [0.003, 0.011] | 0.635 [0.604, 0.663] |
| qwen3-4b-base   | few-shot, exact last | 0.856 [0.815, 0.888] | 0.000 [0.000, 0.011] | 0.856 [0.819, 0.892] | 0.662 [0.633, 0.692] | 0.006 [0.002, 0.009] | 0.656 [0.627, 0.686] |
| qwen3-4b        | few-shot, range last | 0.542 [0.490, 0.592] | 0.000 [0.000, 0.011] | 0.542 [0.492, 0.594] | 0.547 [0.507, 0.591] | 0.000 [0.000, 0.011] | 0.547 [0.507, 0.591] |
| qwen3-4b        | few-shot, exact last | 0.489 [0.438, 0.540] | 0.000 [0.000, 0.011] | 0.489 [0.436, 0.542] | 0.515 [0.471, 0.561] | 0.000 [0.000, 0.011] | 0.515 [0.471, 0.561] |
| qwen3-4b        | chat                 | 0.503 [0.451, 0.554] | 0.000 [0.000, 0.011] | 0.503 [0.450, 0.556] | 0.517 [0.467, 0.566] | 0.000 [0.000, 0.011] | 0.517 [0.467, 0.566] |
| gpt2            | few-shot, range last | 1.000 [0.989, 1.000] | 0.978 [0.957, 0.989] | 0.022 [0.008, 0.039] | 0.393 [0.369, 0.417] | 0.337 [0.312, 0.360] | 0.056 [0.023, 0.089] |
| gpt2            | few-shot, exact last | 0.978 [0.957, 0.989] | 0.436 [0.386, 0.488] | 0.542 [0.486, 0.597] | 0.262 [0.239, 0.286] | 0.203 [0.184, 0.223] | 0.059 [0.029, 0.089] |
| gpt2-medium     | few-shot, range last | 0.975 [0.953, 0.987] | 0.347 [0.300, 0.398] | 0.628 [0.572, 0.678] | 0.319 [0.298, 0.341] | 0.113 [0.099, 0.127] | 0.206 [0.181, 0.232] |
| gpt2-medium     | few-shot, exact last | 0.978 [0.957, 0.989] | 0.283 [0.239, 0.332] | 0.694 [0.644, 0.744] | 0.184 [0.165, 0.204] | 0.056 [0.046, 0.067] | 0.128 [0.106, 0.151] |
| gpt2-large      | few-shot, range last | 0.439 [0.389, 0.491] | 0.022 [0.011, 0.043] | 0.417 [0.361, 0.472] | 0.248 [0.226, 0.271] | 0.052 [0.042, 0.063] | 0.197 [0.172, 0.221] |
| gpt2-large      | few-shot, exact last | 0.144 [0.112, 0.185] | 0.000 [0.000, 0.011] | 0.144 [0.108, 0.181] | 0.142 [0.125, 0.159] | 0.018 [0.012, 0.025] | 0.124 [0.104, 0.142] |
| gpt2-xl         | few-shot, range last | 0.589 [0.537, 0.639] | 0.003 [0.000, 0.016] | 0.586 [0.536, 0.636] | 0.322 [0.297, 0.349] | 0.071 [0.059, 0.082] | 0.252 [0.224, 0.281] |
| gpt2-xl         | few-shot, exact last | 0.561 [0.509, 0.611] | 0.006 [0.002, 0.020] | 0.556 [0.508, 0.608] | 0.259 [0.236, 0.282] | 0.049 [0.039, 0.061] | 0.209 [0.183, 0.234] |

- **An exact question gets a point from the Qwen3 checkpoints.** The greedy range share on the exact questions is 0.000 in 13 of the 15 Qwen3 cells; the other two are the `qwen3-0.6b-base` cells (0.153 [0.119, 0.194] with a range last and 0.019 [0.009, 0.040] with an exact last). The sampled range share on the exact questions is 0.121 and 0.093 for `qwen3-0.6b-base` and at most 0.032 in the other thirteen cells.
- **A range question gets a range for part of the answers, and for fewer after post-training.** Greedy, the base checkpoints answer 0.989 and 0.989 (0.6B), 0.844 and 0.742 (1.7B) and 0.831 and 0.856 (4B) of the range questions with a range (range last and exact last), and the post-trained ones 0.675 and 0.353, 0.275 and 0.256, and 0.542 and 0.489 in the same two orders, and 0.106, 0.197 and 0.503 in chat. At each size and in each few-shot order the gap of the base checkpoint is above that of the post-trained one with disjoint intervals (6 of 6 comparisons, for example 0.836 [0.797, 0.875] against 0.675 [0.625, 0.720] for the 0.6B pair with a range last). In the two few-shot orders the post-trained share is not ordered by size (lowest for `qwen3-1.7b`); in chat it rises with size (0.106, 0.197, 0.503).
- **The order of the examples moves the gap of some checkpoints and not of others.** The greedy gap with a range last minus the gap with an exact last is -0.133 (`qwen3-0.6b-base`), +0.322 (`qwen3-0.6b`), +0.103 (`qwen3-1.7b-base`), +0.019 (`qwen3-1.7b`), -0.025 (`qwen3-4b-base`), +0.053 (`qwen3-4b`), -0.519 (`gpt2`), -0.067 (`gpt2-medium`), +0.272 (`gpt2-large`) and +0.031 (`gpt2-xl`). The two gap intervals are disjoint for five of the ten checkpoints (`qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base`, `gpt2` and `gpt2-large`) and overlap for the other five, so the form of the last example does not move the gap of the 4B pair or of `qwen3-1.7b` (+0.019, 0.275 [0.231, 0.325] against 0.256 [0.211, 0.303]) and does move it for `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base` and two of the GPT-2 sizes.
- **The chat format hedges with a marker.** The range share of the chat cell is 0.106 [0.078, 0.142], 0.197 [0.159, 0.241] and 0.503 [0.451, 0.554] for the three post-trained checkpoints. In that cell 348 of the 360 greedy answers to the range questions (0.967) and 161 of the 360 to the exact questions (0.447) mark themselves as approximate for `qwen3-0.6b`, 353 (0.981) and 112 (0.311) for `qwen3-1.7b`, and 310 (0.861) and 10 (0.028) for `qwen3-4b`, against at most 1 of 360 greedy answers in any few-shot cell of any checkpoint (the sampled few-shot answers reach 87 of 1,800, for `qwen3-4b-base` on the range questions with an exact last). The range share does not count these.
- **The GPT-2 checkpoints give ranges on exact questions as well.** `gpt2` answers 0.978 [0.957, 0.989] of the exact questions with a range when the last example is a range and 0.436 [0.386, 0.488] when it is exact (its share on the range questions is 1.000 and 0.978), `gpt2-medium` 0.347 and 0.283, `gpt2-large` 0.022 and 0.000, and `gpt2-xl` 0.003 and 0.006. The greedy gap is 0.022 [0.008, 0.039] and 0.542 [0.486, 0.597] for `gpt2`, 0.628 and 0.694 for `gpt2-medium`, 0.417 and 0.144 for `gpt2-large` and 0.586 and 0.556 for `gpt2-xl`. The files show these numbers and how they depend on the order of the examples; they do not show why a GPT-2 checkpoint answers with a range. For some groups most of their answers are also flagged as being in another unit than the question's (the unit flags in the hand check below).
- **Sampling.** The sampled range share on the range questions is lower than the greedy one in 19 of the 23 cells, by 0.001 to 0.794. It is higher in the `qwen3-0.6b` chat cell (0.252 [0.226, 0.280] against 0.106) and within 0.026 of the greedy one in the three `qwen3-4b` cells (0.547, 0.515 and 0.517 against 0.542, 0.489 and 0.503).
- **Abstentions and non-answers are rare outside `qwen3-0.6b` chat.** Outside the chat answers of `qwen3-0.6b`, no group of `table.csv` has an abstention share above 0.010 or a no-answer share above 0.027. In `qwen3-0.6b` chat, six groups exceed 0.010 for abstention (0.012 to 0.058, the largest on the greedy free answers about people) and eight exceed 0.027 for no answer (0.031 to 0.095, the largest on the forced sampled answers about the dog breeds).

Hedge gap by family, free condition, greedy: range share on the range questions / on the exact questions, and the gap:

| Model           | Prompt               | duration                                | height                                  | wingspan                                |
| --------------- | -------------------- | --------------------------------------- | --------------------------------------- | --------------------------------------- |
| qwen3-0.6b-base | few-shot, range last | 0.992 / 0.000, gap 0.992 [0.975, 1.000] | 0.975 / 0.017, gap 0.958 [0.917, 0.992] | 1.000 / 0.442, gap 0.558 [0.467, 0.642] |
| qwen3-0.6b-base | few-shot, exact last | 1.000 / 0.000, gap 1.000 [1.000, 1.000] | 0.967 / 0.000, gap 0.967 [0.933, 0.992] | 1.000 / 0.058, gap 0.942 [0.892, 0.975] |
| qwen3-0.6b      | few-shot, range last | 0.708 / 0.000, gap 0.708 [0.633, 0.792] | 0.700 / 0.000, gap 0.700 [0.617, 0.775] | 0.617 / 0.000, gap 0.617 [0.533, 0.700] |
| qwen3-0.6b      | few-shot, exact last | 0.000 / 0.000, gap 0.000 [0.000, 0.000] | 0.650 / 0.000, gap 0.650 [0.567, 0.733] | 0.408 / 0.000, gap 0.408 [0.325, 0.500] |
| qwen3-0.6b      | chat                 | 0.050 / 0.000, gap 0.050 [0.017, 0.092] | 0.267 / 0.000, gap 0.267 [0.192, 0.350] | 0.000 / 0.000, gap 0.000 [0.000, 0.000] |
| qwen3-1.7b-base | few-shot, range last | 0.558 / 0.000, gap 0.558 [0.467, 0.650] | 0.992 / 0.000, gap 0.992 [0.975, 1.000] | 0.983 / 0.000, gap 0.983 [0.958, 1.000] |
| qwen3-1.7b-base | few-shot, exact last | 0.342 / 0.000, gap 0.342 [0.258, 0.425] | 0.967 / 0.000, gap 0.967 [0.933, 0.992] | 0.917 / 0.000, gap 0.917 [0.867, 0.958] |
| qwen3-1.7b      | few-shot, range last | 0.000 / 0.000, gap 0.000 [0.000, 0.000] | 0.825 / 0.000, gap 0.825 [0.758, 0.892] | 0.000 / 0.000, gap 0.000 [0.000, 0.000] |
| qwen3-1.7b      | few-shot, exact last | 0.000 / 0.000, gap 0.000 [0.000, 0.000] | 0.767 / 0.000, gap 0.767 [0.692, 0.842] | 0.000 / 0.000, gap 0.000 [0.000, 0.000] |
| qwen3-1.7b      | chat                 | 0.000 / 0.000, gap 0.000 [0.000, 0.000] | 0.592 / 0.000, gap 0.592 [0.500, 0.675] | 0.000 / 0.000, gap 0.000 [0.000, 0.000] |
| qwen3-4b-base   | few-shot, range last | 0.592 / 0.000, gap 0.592 [0.508, 0.675] | 0.983 / 0.000, gap 0.983 [0.958, 1.000] | 0.917 / 0.000, gap 0.917 [0.867, 0.958] |
| qwen3-4b-base   | few-shot, exact last | 0.650 / 0.000, gap 0.650 [0.567, 0.733] | 0.983 / 0.000, gap 0.983 [0.958, 1.000] | 0.933 / 0.000, gap 0.933 [0.883, 0.975] |
| qwen3-4b        | few-shot, range last | 0.267 / 0.000, gap 0.267 [0.192, 0.350] | 0.983 / 0.000, gap 0.983 [0.958, 1.000] | 0.375 / 0.000, gap 0.375 [0.292, 0.467] |
| qwen3-4b        | few-shot, exact last | 0.183 / 0.000, gap 0.183 [0.117, 0.258] | 0.983 / 0.000, gap 0.983 [0.958, 1.000] | 0.300 / 0.000, gap 0.300 [0.217, 0.383] |
| qwen3-4b        | chat                 | 0.133 / 0.000, gap 0.133 [0.075, 0.192] | 0.983 / 0.000, gap 0.983 [0.958, 1.000] | 0.392 / 0.000, gap 0.392 [0.308, 0.483] |
| gpt2            | few-shot, range last | 1.000 / 1.000, gap 0.000 [0.000, 0.000] | 1.000 / 0.933, gap 0.067 [0.025, 0.117] | 1.000 / 1.000, gap 0.000 [0.000, 0.000] |
| gpt2            | few-shot, exact last | 1.000 / 0.025, gap 0.975 [0.942, 1.000] | 0.933 / 0.608, gap 0.325 [0.233, 0.425] | 1.000 / 0.675, gap 0.325 [0.242, 0.417] |
| gpt2-medium     | few-shot, range last | 1.000 / 0.000, gap 1.000 [1.000, 1.000] | 0.925 / 0.108, gap 0.817 [0.742, 0.892] | 1.000 / 0.933, gap 0.067 [0.025, 0.117] |
| gpt2-medium     | few-shot, exact last | 0.983 / 0.000, gap 0.983 [0.958, 1.000] | 0.950 / 0.042, gap 0.908 [0.850, 0.958] | 1.000 / 0.808, gap 0.192 [0.125, 0.267] |
| gpt2-large      | few-shot, range last | 0.767 / 0.000, gap 0.767 [0.692, 0.833] | 0.517 / 0.067, gap 0.450 [0.350, 0.550] | 0.033 / 0.000, gap 0.033 [0.008, 0.067] |
| gpt2-large      | few-shot, exact last | 0.425 / 0.000, gap 0.425 [0.342, 0.517] | 0.008 / 0.000, gap 0.008 [0.000, 0.025] | 0.000 / 0.000, gap 0.000 [0.000, 0.000] |
| gpt2-xl         | few-shot, range last | 0.967 / 0.000, gap 0.967 [0.933, 0.992] | 0.567 / 0.008, gap 0.558 [0.467, 0.650] | 0.233 / 0.000, gap 0.233 [0.158, 0.317] |
| gpt2-xl         | few-shot, exact last | 0.908 / 0.000, gap 0.908 [0.850, 0.958] | 0.633 / 0.017, gap 0.617 [0.525, 0.708] | 0.142 / 0.000, gap 0.142 [0.083, 0.208] |

- **The post-trained checkpoints give ranges for the dog breeds most often.** The greedy range share on the breeds (range last, exact last, chat) is 0.700, 0.650 and 0.267 for `qwen3-0.6b`, 0.825, 0.767 and 0.592 for `qwen3-1.7b`, and 0.983 in all three cells for `qwen3-4b`. The one exception is `qwen3-0.6b` with a range last, where the gestation periods are at 0.708. `qwen3-1.7b` never gives a range for a gestation period or a wingspan (0.000 in the six cells), `qwen3-0.6b` gives 0.000 and 0.408 (duration, wingspan) with an exact last and 0.050 and 0.000 in chat, and `qwen3-4b` gives 0.267, 0.183 and 0.133 for the gestation periods and 0.375, 0.300 and 0.392 for the wingspans.
- **The base checkpoints give ranges in all three families.** The range share on the range questions runs from 0.342 (gestation periods, `qwen3-1.7b-base`, exact last) to 1.000. On the exact questions the base checkpoints give 0.000 in every family and cell except `qwen3-0.6b-base` (0.442 on the aircraft wingspans and 0.017 on the skyscrapers with a range last, 0.058 on the wingspans with an exact last).
- **The GPT-2 sizes differ in the family that hedges.** With a range last, `gpt2-large` gives ranges for 0.767 of the gestation periods, 0.517 of the breeds and 0.033 of the wingspans, and `gpt2-xl` for 0.967, 0.567 and 0.233 (0.425, 0.008 and 0.000 for `gpt2-large` and 0.908, 0.633 and 0.142 for `gpt2-xl` with an exact last), against exact questions at 0.000 to 0.067 in both. For `gpt2` and `gpt2-medium` the exact side is high in some families: 1.000 on the films and the aircraft and 0.933 on the skyscrapers for `gpt2` with a range last, and 0.933 on the aircraft for `gpt2-medium`.
- **A family gap is a domain contrast as well as a kind contrast.** Within a family the two sides also differ in domain (breeds against buildings, birds against aircraft, species against films), so a gap in one family is not the effect of the kind alone; the people rows below separate the two for height.

Forced condition, greedy: range share on each kind, and how often the single number lands inside the sourced range. A point is inside when it lies between the sourced bounds, and inside the band when it lies in the bounds widened by 5 percent, the tolerance of the grader. Exact questions are the reference, because their sourced range is one value. The counts are the points inside over the points given:

| Model           | Prompt               | Range answers, range items | Range answers, exact items | Point inside, range items         | Point inside, exact items        | Point inside the band, range items | Point inside the band, exact items |
| --------------- | -------------------- | -------------------------- | -------------------------- | --------------------------------- | -------------------------------- | ---------------------------------- | ---------------------------------- |
| qwen3-0.6b-base | few-shot, range last | 0.214 [0.175, 0.259]       | 0.000 [0.000, 0.011]       | 0.074 [0.049, 0.111] (21 of 283)  | 0.008 [0.003, 0.024] (3 of 360)  | 0.078 [0.052, 0.115] (22 of 283)   | 0.069 [0.047, 0.101] (25 of 360)   |
| qwen3-0.6b-base | few-shot, exact last | 0.231 [0.190, 0.277]       | 0.000 [0.000, 0.011]       | 0.079 [0.053, 0.117] (22 of 277)  | 0.008 [0.003, 0.024] (3 of 360)  | 0.083 [0.056, 0.122] (23 of 277)   | 0.064 [0.043, 0.094] (23 of 360)   |
| qwen3-0.6b      | few-shot, range last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.058 [0.038, 0.088] (21 of 360)  | 0.006 [0.002, 0.020] (2 of 359)  | 0.061 [0.041, 0.091] (22 of 360)   | 0.103 [0.076, 0.139] (37 of 359)   |
| qwen3-0.6b      | few-shot, exact last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.033 [0.019, 0.057] (12 of 360)  | 0.008 [0.003, 0.024] (3 of 359)  | 0.050 [0.032, 0.078] (18 of 360)   | 0.095 [0.069, 0.129] (34 of 359)   |
| qwen3-0.6b      | chat                 | 0.014 [0.006, 0.032]       | 0.000 [0.000, 0.011]       | 0.066 [0.045, 0.098] (23 of 346)  | 0.006 [0.002, 0.020] (2 of 359)  | 0.081 [0.057, 0.114] (28 of 346)   | 0.097 [0.071, 0.133] (35 of 359)   |
| qwen3-1.7b-base | few-shot, range last | 0.081 [0.057, 0.113]       | 0.000 [0.000, 0.011]       | 0.187 [0.149, 0.233] (62 of 331)  | 0.042 [0.025, 0.068] (15 of 360) | 0.211 [0.171, 0.259] (70 of 331)   | 0.175 [0.139, 0.218] (63 of 360)   |
| qwen3-1.7b-base | few-shot, exact last | 0.067 [0.045, 0.097]       | 0.000 [0.000, 0.011]       | 0.202 [0.163, 0.249] (68 of 336)  | 0.047 [0.030, 0.074] (17 of 360) | 0.238 [0.196, 0.286] (80 of 336)   | 0.181 [0.144, 0.224] (65 of 360)   |
| qwen3-1.7b      | few-shot, range last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.181 [0.144, 0.224] (65 of 360)  | 0.031 [0.017, 0.054] (11 of 360) | 0.219 [0.180, 0.265] (79 of 360)   | 0.147 [0.114, 0.188] (53 of 360)   |
| qwen3-1.7b      | few-shot, exact last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.175 [0.139, 0.218] (63 of 360)  | 0.031 [0.017, 0.054] (11 of 360) | 0.200 [0.162, 0.244] (72 of 360)   | 0.150 [0.117, 0.191] (54 of 360)   |
| qwen3-1.7b      | chat                 | 0.011 [0.004, 0.028]       | 0.000 [0.000, 0.011]       | 0.121 [0.091, 0.160] (43 of 354)  | 0.011 [0.004, 0.028] (4 of 358)  | 0.161 [0.126, 0.203] (57 of 354)   | 0.078 [0.055, 0.111] (28 of 358)   |
| qwen3-4b-base   | few-shot, range last | 0.003 [0.000, 0.016]       | 0.000 [0.000, 0.011]       | 0.334 [0.287, 0.385] (120 of 359) | 0.139 [0.107, 0.178] (50 of 360) | 0.426 [0.376, 0.478] (153 of 359)  | 0.331 [0.284, 0.381] (119 of 360)  |
| qwen3-4b-base   | few-shot, exact last | 0.008 [0.003, 0.024]       | 0.000 [0.000, 0.011]       | 0.336 [0.289, 0.387] (120 of 357) | 0.144 [0.112, 0.185] (52 of 360) | 0.401 [0.351, 0.452] (143 of 357)  | 0.336 [0.289, 0.386] (121 of 360)  |
| qwen3-4b        | few-shot, range last | 0.006 [0.002, 0.020]       | 0.000 [0.000, 0.011]       | 0.268 [0.225, 0.316] (96 of 358)  | 0.081 [0.057, 0.113] (29 of 360) | 0.321 [0.275, 0.371] (115 of 358)  | 0.272 [0.229, 0.320] (98 of 360)   |
| qwen3-4b        | few-shot, exact last | 0.006 [0.002, 0.020]       | 0.000 [0.000, 0.011]       | 0.254 [0.212, 0.302] (91 of 358)  | 0.083 [0.059, 0.116] (30 of 360) | 0.316 [0.270, 0.366] (113 of 358)  | 0.306 [0.260, 0.355] (110 of 360)  |
| qwen3-4b        | chat                 | 0.008 [0.003, 0.024]       | 0.000 [0.000, 0.011]       | 0.246 [0.205, 0.294] (88 of 357)  | 0.044 [0.028, 0.071] (16 of 360) | 0.311 [0.265, 0.361] (111 of 357)  | 0.181 [0.144, 0.224] (65 of 360)   |
| gpt2            | few-shot, range last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011] (0 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.000 [0.000, 0.011] (0 of 360)    | 0.003 [0.000, 0.016] (1 of 360)    |
| gpt2            | few-shot, exact last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011] (0 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.000 [0.000, 0.011] (0 of 360)    | 0.003 [0.000, 0.016] (1 of 360)    |
| gpt2-medium     | few-shot, range last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011] (0 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.000 [0.000, 0.011] (0 of 360)    | 0.000 [0.000, 0.011] (0 of 360)    |
| gpt2-medium     | few-shot, exact last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011] (0 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.000 [0.000, 0.011] (0 of 360)    | 0.000 [0.000, 0.011] (0 of 360)    |
| gpt2-large      | few-shot, range last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.003 [0.000, 0.016] (1 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.003 [0.000, 0.016] (1 of 360)    | 0.000 [0.000, 0.011] (0 of 360)    |
| gpt2-large      | few-shot, exact last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.003 [0.000, 0.016] (1 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.003 [0.000, 0.016] (1 of 360)    | 0.003 [0.000, 0.016] (1 of 360)    |
| gpt2-xl         | few-shot, range last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011] (0 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.003 [0.000, 0.016] (1 of 360)    | 0.000 [0.000, 0.011] (0 of 360)    |
| gpt2-xl         | few-shot, exact last | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011]       | 0.000 [0.000, 0.011] (0 of 360)   | 0.000 [0.000, 0.011] (0 of 360)  | 0.000 [0.000, 0.011] (0 of 360)    | 0.000 [0.000, 0.011] (0 of 360)    |

- **Greedy decoding follows the instruction in the post-trained checkpoints and the larger base ones.** Under the forced prompt the greedy range share on the range questions is 0.000 to 0.014 in the nine post-trained cells, 0.000 in the eight GPT-2 cells, 0.003 and 0.008 for `qwen3-4b-base`, 0.081 and 0.067 for `qwen3-1.7b-base`, and 0.214 [0.175, 0.259] and 0.231 [0.190, 0.277] for `qwen3-0.6b-base`; it is 0.000 on the exact questions in all 23 cells. Sampled, the base Qwen3 checkpoints still give ranges: the forced hedge gaps in `hedge_gap.csv` are 0.186 [0.163, 0.209] and 0.233 [0.207, 0.260] (0.6B), 0.082 [0.066, 0.098] and 0.088 [0.073, 0.103] (1.7B) and 0.102 [0.084, 0.119] and 0.138 [0.119, 0.159] (4B), against 0.000 to 0.108 for the post-trained checkpoints and 0.004 to 0.028 for the GPT-2 sizes.
- **For the Qwen3 checkpoints the forced number is inside the sourced range more often on range questions than on exact questions, and the share rises with size.** Greedy, the points inside the sourced range are 0.033 to 0.079 of the points on the range questions and 0.006 to 0.008 on the exact questions for the 0.6B pair, 0.121 to 0.202 and 0.011 to 0.047 for the 1.7B pair, and 0.246 to 0.336 and 0.044 to 0.144 for the 4B pair. The intervals of the two kinds are disjoint in 14 of the 15 Qwen3 cells (the exception is `qwen3-0.6b` with an exact last, 0.033 [0.019, 0.057] against 0.008 [0.003, 0.024]). For an exact question inside means equal to the sourced value, so this comparison is not matched for width; in the 5 percent band the order of the kinds reverses in three `qwen3-0.6b` cells (0.061 against 0.103, 0.050 against 0.095 and 0.081 against 0.097).
- **The forced GPT-2 numbers are almost never inside the sourced range, and a few values repeat.** The points of the eight GPT-2 cells are inside the sourced range for at most 1 of 360 questions of either kind. In `hand_check.csv` the most frequent value among the 120 forced greedy answers of a cell accounts for 62 and 49 of them for `gpt2` (5.5 with a range last, 6.5 with an exact last), 40 and 45 for `gpt2-medium` (1.5), 37 and 37 for `gpt2-large` (1.5) and 17 and 14 for `gpt2-xl`, against 22 and 22 for `qwen3-0.6b-base`, 43 and 45 for `qwen3-0.6b` (120), 22 and 27 for `qwen3-1.7b-base`, 16 and 19 for `qwen3-1.7b`, 10 and 10 for `qwen3-4b-base` and 6 and 7 for `qwen3-4b`; these counts are over the 120 questions of the file, drawn from seven strata, and not over all 840.
- **Forcing one number lowers the share of range questions answered within the sourced range plus 5 percent for most Qwen3 cells.** The share of correct greedy answers on the range questions is lower in the forced condition than in the free one in 13 of the 15 Qwen3 cells (0.561 and 0.564 against 0.425 and 0.397 for `qwen3-4b-base`, 0.478 against 0.314 for `qwen3-4b` in chat, 0.284 against 0.165 for `qwen3-1.7b` in chat); the two exceptions are `qwen3-0.6b` cells at 0.058 against 0.061 and 0.077 against 0.083, where the share is below 0.09 in both conditions. On the exact questions the two conditions differ by at most 0.083 in these cells. The greedy correct rate of the GPT-2 sizes, pooled over the families, is 0.000 to 0.011 for both kinds in both conditions (`table.csv`, family all).

Range share on the range questions by width of the sourced range, free condition, greedy. The width is the sourced range divided by its midpoint, and the range questions of all three families are cut by rank into three groups of 120 (width 0.100 to 0.141, 0.141 to 0.190 and 0.190 to 1.014):

| Model           | Prompt               | Narrow third         | Middle third         | Wide third           |
| --------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| qwen3-0.6b-base | few-shot, range last | 0.992 [0.954, 0.999] | 0.992 [0.954, 0.999] | 0.983 [0.941, 0.995] |
| qwen3-0.6b-base | few-shot, exact last | 1.000 [0.969, 1.000] | 0.992 [0.954, 0.999] | 0.975 [0.929, 0.991] |
| qwen3-0.6b      | few-shot, range last | 0.667 [0.578, 0.745] | 0.683 [0.596, 0.760] | 0.675 [0.587, 0.752] |
| qwen3-0.6b      | few-shot, exact last | 0.333 [0.255, 0.422] | 0.392 [0.309, 0.481] | 0.333 [0.255, 0.422] |
| qwen3-0.6b      | chat                 | 0.117 [0.071, 0.186] | 0.108 [0.064, 0.177] | 0.092 [0.052, 0.157] |
| qwen3-1.7b-base | few-shot, range last | 0.850 [0.775, 0.903] | 0.833 [0.757, 0.889] | 0.850 [0.775, 0.903] |
| qwen3-1.7b-base | few-shot, exact last | 0.750 [0.666, 0.819] | 0.742 [0.657, 0.812] | 0.733 [0.648, 0.804] |
| qwen3-1.7b      | few-shot, range last | 0.217 [0.152, 0.299] | 0.292 [0.218, 0.378] | 0.317 [0.240, 0.404] |
| qwen3-1.7b      | few-shot, exact last | 0.217 [0.152, 0.299] | 0.275 [0.203, 0.361] | 0.275 [0.203, 0.361] |
| qwen3-1.7b      | chat                 | 0.158 [0.104, 0.234] | 0.242 [0.174, 0.326] | 0.192 [0.131, 0.271] |
| qwen3-4b-base   | few-shot, range last | 0.767 [0.683, 0.833] | 0.842 [0.766, 0.896] | 0.883 [0.814, 0.929] |
| qwen3-4b-base   | few-shot, exact last | 0.817 [0.738, 0.876] | 0.842 [0.766, 0.896] | 0.908 [0.843, 0.948] |
| qwen3-4b        | few-shot, range last | 0.425 [0.340, 0.514] | 0.583 [0.494, 0.668] | 0.617 [0.527, 0.699] |
| qwen3-4b        | few-shot, exact last | 0.375 [0.294, 0.464] | 0.517 [0.428, 0.604] | 0.575 [0.486, 0.660] |
| qwen3-4b        | chat                 | 0.383 [0.301, 0.473] | 0.567 [0.477, 0.652] | 0.558 [0.469, 0.644] |
| gpt2            | few-shot, range last | 1.000 [0.969, 1.000] | 1.000 [0.969, 1.000] | 1.000 [0.969, 1.000] |
| gpt2            | few-shot, exact last | 0.983 [0.941, 0.995] | 0.992 [0.954, 0.999] | 0.958 [0.906, 0.982] |
| gpt2-medium     | few-shot, range last | 0.967 [0.917, 0.987] | 0.975 [0.929, 0.991] | 0.983 [0.941, 0.995] |
| gpt2-medium     | few-shot, exact last | 0.983 [0.941, 0.995] | 0.967 [0.917, 0.987] | 0.983 [0.941, 0.995] |
| gpt2-large      | few-shot, range last | 0.442 [0.356, 0.531] | 0.475 [0.388, 0.564] | 0.400 [0.317, 0.489] |
| gpt2-large      | few-shot, exact last | 0.200 [0.138, 0.280] | 0.150 [0.097, 0.225] | 0.083 [0.046, 0.147] |
| gpt2-xl         | few-shot, range last | 0.525 [0.436, 0.612] | 0.608 [0.519, 0.691] | 0.633 [0.544, 0.714] |
| gpt2-xl         | few-shot, exact last | 0.533 [0.444, 0.620] | 0.583 [0.494, 0.668] | 0.567 [0.477, 0.652] |

- **The range share rises with the width of the sourced range for some checkpoints only.** Between the narrow and the wide third the greedy range share is higher in 12 of the 23 cells, lower in 7 and equal in 4 (`qwen3-0.6b` with an exact last, `qwen3-1.7b-base` with a range last, `gpt2` with a range last and `gpt2-medium` with an exact last). The two intervals are disjoint in two cells, both `qwen3-4b` with a few-shot prompt (0.425 [0.340, 0.514] to 0.617 [0.527, 0.699] with a range last, 0.375 [0.294, 0.464] to 0.575 [0.486, 0.660] with an exact last), and in none in the other direction. The largest rises are `qwen3-4b` (+0.192, +0.200 and +0.175 in chat), `qwen3-4b-base` (+0.117 and +0.092), `gpt2-xl` with a range last (+0.108) and `qwen3-1.7b` with a range last (+0.100); the largest fall is `gpt2-large` with an exact last (0.200 to 0.083). In `qwen3-0.6b-base`, `gpt2` and `gpt2-medium` the range share is 0.958 or higher in every third, so it has little room to rise.
- **Each third mixes families.** The groups cut the range questions of the three families together, so a third also differs in family; the thirds within each family are in `width.csv`.

The people heights, free condition, greedy: the number of range answers on the 120 people (all exact), the 120 dog breeds and the 120 buildings of the height family, and the differences of the range share:

| Model           | Prompt               | People (range answers) | Breeds     | Buildings  | People minus breeds     | People minus buildings  |
| --------------- | -------------------- | ---------------------- | ---------- | ---------- | ----------------------- | ----------------------- |
| qwen3-0.6b-base | few-shot, range last | 0 of 120               | 117 of 120 | 2 of 120   | -0.975 [-1.000, -0.942] | -0.017 [-0.042, 0.000]  |
| qwen3-0.6b-base | few-shot, exact last | 0 of 120               | 116 of 120 | 0 of 120   | -0.967 [-0.992, -0.933] | 0.000 [0.000, 0.000]    |
| qwen3-0.6b      | few-shot, range last | 0 of 120               | 84 of 120  | 0 of 120   | -0.700 [-0.775, -0.608] | 0.000 [0.000, 0.000]    |
| qwen3-0.6b      | few-shot, exact last | 0 of 120               | 78 of 120  | 0 of 120   | -0.650 [-0.733, -0.558] | 0.000 [0.000, 0.000]    |
| qwen3-0.6b      | chat                 | 0 of 120               | 32 of 120  | 0 of 120   | -0.267 [-0.350, -0.183] | 0.000 [0.000, 0.000]    |
| qwen3-1.7b-base | few-shot, range last | 0 of 120               | 119 of 120 | 0 of 120   | -0.992 [-1.000, -0.975] | 0.000 [0.000, 0.000]    |
| qwen3-1.7b-base | few-shot, exact last | 0 of 120               | 116 of 120 | 0 of 120   | -0.967 [-0.992, -0.933] | 0.000 [0.000, 0.000]    |
| qwen3-1.7b      | few-shot, range last | 0 of 120               | 99 of 120  | 0 of 120   | -0.825 [-0.892, -0.758] | 0.000 [0.000, 0.000]    |
| qwen3-1.7b      | few-shot, exact last | 0 of 120               | 92 of 120  | 0 of 120   | -0.767 [-0.842, -0.692] | 0.000 [0.000, 0.000]    |
| qwen3-1.7b      | chat                 | 0 of 120               | 71 of 120  | 0 of 120   | -0.592 [-0.675, -0.500] | 0.000 [0.000, 0.000]    |
| qwen3-4b-base   | few-shot, range last | 0 of 120               | 118 of 120 | 0 of 120   | -0.983 [-1.000, -0.958] | 0.000 [0.000, 0.000]    |
| qwen3-4b-base   | few-shot, exact last | 0 of 120               | 118 of 120 | 0 of 120   | -0.983 [-1.000, -0.958] | 0.000 [0.000, 0.000]    |
| qwen3-4b        | few-shot, range last | 0 of 120               | 118 of 120 | 0 of 120   | -0.983 [-1.000, -0.958] | 0.000 [0.000, 0.000]    |
| qwen3-4b        | few-shot, exact last | 0 of 120               | 118 of 120 | 0 of 120   | -0.983 [-1.000, -0.958] | 0.000 [0.000, 0.000]    |
| qwen3-4b        | chat                 | 0 of 120               | 118 of 120 | 0 of 120   | -0.983 [-1.000, -0.958] | 0.000 [0.000, 0.000]    |
| gpt2            | few-shot, range last | 120 of 120             | 120 of 120 | 112 of 120 | 0.000 [0.000, 0.000]    | 0.067 [0.025, 0.117]    |
| gpt2            | few-shot, exact last | 23 of 120              | 112 of 120 | 73 of 120  | -0.742 [-0.825, -0.658] | -0.417 [-0.525, -0.300] |
| gpt2-medium     | few-shot, range last | 2 of 120               | 111 of 120 | 13 of 120  | -0.908 [-0.958, -0.850] | -0.092 [-0.158, -0.033] |
| gpt2-medium     | few-shot, exact last | 1 of 120               | 114 of 120 | 5 of 120   | -0.942 [-0.983, -0.900] | -0.033 [-0.075, 0.000]  |
| gpt2-large      | few-shot, range last | 0 of 120               | 62 of 120  | 8 of 120   | -0.517 [-0.608, -0.425] | -0.067 [-0.117, -0.025] |
| gpt2-large      | few-shot, exact last | 0 of 120               | 1 of 120   | 0 of 120   | -0.008 [-0.025, 0.000]  | 0.000 [0.000, 0.000]    |
| gpt2-xl         | few-shot, range last | 0 of 120               | 68 of 120  | 1 of 120   | -0.567 [-0.650, -0.475] | -0.008 [-0.025, 0.000]  |
| gpt2-xl         | few-shot, exact last | 0 of 120               | 76 of 120  | 2 of 120   | -0.633 [-0.717, -0.550] | -0.017 [-0.042, 0.000]  |

- **The Qwen3 checkpoints never give a range for a person's height in greedy decoding.** The range answers are 0 of 120 for the people in all 15 Qwen3 cells, against 32 to 119 of 120 for the breeds; for the buildings they are 0 of 120 except 2 of 120 (`qwen3-0.6b-base`, range last). People minus buildings is 0.000 in 14 cells and -0.017 [-0.042, 0.000] in the other, and people minus breeds runs from -0.992 [-1.000, -0.975] (`qwen3-1.7b-base`, range last) to -0.267 [-0.350, -0.183] (`qwen3-0.6b`, chat). The range share follows the sourced answer (a range for breeds, one value for people and buildings) rather than animacy, within the familiarity caveat above.
- **Sampled, the base Qwen3 checkpoints give some ranges for people.** 77 and 53 of 600 sampled answers about people are ranges for `qwen3-0.6b-base` (0.128 and 0.088), 29 and 22 for `qwen3-1.7b-base` (0.048 and 0.037) and 7 and 12 for `qwen3-4b-base` (0.012 and 0.020), against 0.032 and 0.037, 0.003 and 0.003, and 0.002 and 0.002 on the buildings and 0.707 to 0.887 on the breeds. People minus buildings is above 0 in the six cells (0.010 [0.002, 0.020] to 0.097 [0.067, 0.127]). The post-trained checkpoints give 0 to 14 of 600 (0 for `qwen3-1.7b` and `qwen3-4b`).
- **In chat the marker is the hedge, and the people sit between the breeds and the buildings.** The greedy chat answers carry an approximation marker for 110, 115 and 117 of the 120 breeds, 49, 67 and 15 of the 120 people, and 41, 9 and 4 of the 120 buildings (`qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`). The marker is more frequent on the people than on the buildings in all three, so it does not follow the sourced answer alone.
- **`gpt2` does not separate the groups as the sourced answer does.** With a range last it gives a range for all 120 people, all 120 breeds and 112 of the 120 buildings (people minus buildings 0.067 [0.025, 0.117]); with an exact last, for 23 people, 112 breeds and 73 buildings (people minus buildings -0.417 [-0.525, -0.300]). `gpt2-medium` gives 2 and 1 people, 111 and 114 breeds and 13 and 5 buildings; `gpt2-large` 0 and 0, 62 and 1, and 8 and 0; `gpt2-xl` 0 and 0, 68 and 76, and 1 and 2 (range last and exact last).

**The hand check.** Each P10a run writes two files for a person to check the answer parser and the grader. `hand_check.csv` holds 120 questions drawn in turn from the seven strata (three families by two kinds, and the people), 17 or 18 per stratum, with the greedy completion of every cell and its parsed form, value, grade and unit flag. `hand_check_samples.csv` holds 120 sampled completions, stratified the same way by family and kind, with a quarter of the rows (30) given to the completions most likely to be misread (no number, an abstention or a unit flag). The columns for the verdict of the person (`human_form`, `human_correct`, `human_unit_mismatch`) are empty in all 20 files of the ten runs, so no agreement between the parser and a person has been measured and the parser has only its tests. What the runs record is the flags of the parser. A unit mismatch (a number in another unit of the same dimension as the question, such as months for days, or a compound such as "2 hours 22 minutes", whose value is not converted) is flagged for at most 0.070 and 0.067 of the answers of any family and kind group for the 4B pair (base, post-trained), 0.258 and 0.100 for the 1.7B pair, 0.383 and 0.458 for the 0.6B pair, and 0.975, 0.742, 0.992 and 0.967 for `gpt2`, `gpt2-medium`, `gpt2-large` and `gpt2-xl`, so for the GPT-2 sizes nearly every answer of some groups is flagged and the correct rates above reflect it. A completion cut by the 32-token budget is flagged for at most 0.015 of a group for every checkpoint except the chat answers of the post-trained Qwen3 checkpoints (at most 0.270 for 0.6B, 0.138 for 1.7B and 0.080 for 4B, all in sampled chat cells). A flagged compound is graded on one of its numbers: in `hand_check_samples.csv` of `qwen3-4b` the completion "146 (or 2 hours and 26 minutes)" to a question in minutes is read as the point 26, flagged, and graded incorrect, so the correct rates above do not credit an answer that gives the right quantity in another unit. The flagged rows are 19 (`qwen3-4b`) to 30 (`gpt2-xl`) of the 120 rows of `hand_check_samples.csv`, and 2 (both 4B checkpoints) to 61 (`gpt2-large`) of the 120 questions of `hand_check.csv` carry a flag in at least one cell.

### Reading the direction (P10b)

P10b renders each question under two prompts and reads the activations of the last token at every layer from 1 to the last (12, 24, 36 and 48 for the four GPT-2 sizes, 28 for the 0.6B and 1.7B Qwen3 models and 36 for the 4B models; the embedding output is not read). The plain prompt is the bare question. The concept prompt is the concept template of the paper with the word "variability" ("Consider the amount of variability in the following:", the question, "The amount of variability is"), wrapped in the chat format of the checkpoint (the tokenizer's chat template with thinking disabled for a post-trained one, `USER:` and `ASSISTANT:` tags for a base one, which includes the GPT-2 sizes). The questions of each family are split into train, validation and test; the test sets hold 23 range and 30 exact questions for height, 25 and 22 for wingspan, 24 and 23 for duration, and 72 and 75 pooled. A reader is fitted on the train questions of a set of families: each family alone, each pair of families and all three. It pairs a range question with an exact one of the same family and uses one of three direction finders (logistic, pca_diff, mean_diff), each fitted on five draws of pairs, with a random direction as a control that is fitted, oriented and selected like the others. The layer is the one with the highest mean validation AUROC over the five draws on the validation questions of the families of the train set (the first layer on ties), and a higher score means a range answer. The AUROC of a test set is the chance that a range question scores above an exact one. Because the layer and the sign are chosen on data, a random direction does not sit at 0.5: its value in a cell is what a direction with no fitted content reaches under the same selection, and a reader is compared with it, not with 0.5. The surface baselines use no model: `text_word` and `text_char` are logistic regressions on TF-IDF features of the word 1-grams and 2-grams and of the character 2-grams to 5-grams of the question, `length` is the number of characters and `popularity` is the base-10 logarithm of the sitelinks plus 1; each is fitted or oriented on the train questions of the same set of families, so they are the same for every checkpoint.

Train on two families and test on the third, and train on all three and test on the pooled test questions. AUROC with its item bootstrap interval for the logistic reader, the random-direction control of the same cell and the four surface baselines, in the plain and the concept template:

| Model           | Train to test                | Plain: reader        | Plain: random        | Concept: reader      | Concept: random      | text_word            | text_char            | length               | popularity           |
| --------------- | ---------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| qwen3-0.6b-base | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.043 [0.006, 0.098] | 1.000 [1.000, 1.000] | 0.843 [0.717, 0.949] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-0.6b-base | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.520 [0.351, 0.693] | 1.000 [1.000, 1.000] | 0.807 [0.665, 0.926] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-0.6b-base | height+wingspan to duration  | 0.875 [0.754, 0.974] | 0.011 [0.000, 0.043] | 0.973 [0.908, 1.000] | 1.000 [1.000, 1.000] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-0.6b-base | all to the pooled test items | 1.000 [1.000, 1.000] | 0.999 [0.996, 1.000] | 1.000 [1.000, 1.000] | 0.903 [0.851, 0.945] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| qwen3-0.6b      | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.959 [0.883, 1.000] | 0.988 [0.960, 1.000] | 0.865 [0.749, 0.946] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-0.6b      | height+duration to wingspan  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.456 [0.281, 0.636] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-0.6b      | height+wingspan to duration  | 0.826 [0.689, 0.940] | 0.207 [0.087, 0.348] | 0.991 [0.971, 1.000] | 0.960 [0.904, 0.996] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-0.6b      | all to the pooled test items | 1.000 [1.000, 1.000] | 0.932 [0.887, 0.970] | 1.000 [0.999, 1.000] | 0.817 [0.744, 0.879] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| qwen3-1.7b-base | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.929 [0.857, 0.988] | 1.000 [1.000, 1.000] | 0.736 [0.598, 0.865] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-1.7b-base | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.440 [0.281, 0.606] | 1.000 [1.000, 1.000] | 0.655 [0.495, 0.819] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-1.7b-base | height+wingspan to duration  | 0.659 [0.509, 0.811] | 0.841 [0.713, 0.941] | 1.000 [1.000, 1.000] | 0.949 [0.877, 0.996] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-1.7b-base | all to the pooled test items | 1.000 [1.000, 1.000] | 0.960 [0.929, 0.982] | 1.000 [1.000, 1.000] | 0.974 [0.947, 0.994] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| qwen3-1.7b      | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.965 [0.900, 1.000] | 0.867 [0.757, 0.946] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-1.7b      | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.878 [0.768, 0.965] | 0.951 [0.885, 0.993] | 0.645 [0.481, 0.795] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-1.7b      | height+wingspan to duration  | 0.389 [0.217, 0.575] | 0.246 [0.111, 0.404] | 1.000 [1.000, 1.000] | 0.194 [0.087, 0.333] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-1.7b      | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.888 [0.839, 0.932] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| qwen3-4b-base   | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.078 [0.009, 0.170] | 0.980 [0.946, 1.000] | 0.813 [0.684, 0.923] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-4b-base   | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.945 [0.870, 0.991] | 0.933 [0.848, 0.991] | 0.758 [0.594, 0.886] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-4b-base   | height+wingspan to duration  | 0.380 [0.220, 0.547] | 0.996 [0.982, 1.000] | 0.667 [0.495, 0.829] | 0.346 [0.188, 0.515] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-4b-base   | all to the pooled test items | 1.000 [1.000, 1.000] | 0.994 [0.984, 0.999] | 1.000 [1.000, 1.000] | 0.978 [0.955, 0.996] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| qwen3-4b        | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.307 [0.174, 0.462] | 0.991 [0.965, 1.000] | 0.867 [0.757, 0.965] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-4b        | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.998 [0.989, 1.000] | 0.996 [0.984, 1.000] | 0.962 [0.907, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-4b        | height+wingspan to duration  | 0.629 [0.444, 0.798] | 0.987 [0.956, 1.000] | 0.998 [0.989, 1.000] | 0.080 [0.014, 0.164] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-4b        | all to the pooled test items | 1.000 [1.000, 1.000] | 0.949 [0.909, 0.982] | 1.000 [1.000, 1.000] | 0.914 [0.857, 0.958] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| gpt2            | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.725 [0.579, 0.859] | 1.000 [1.000, 1.000] | 0.999 [0.991, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2            | height+duration to wingspan  | 0.998 [0.989, 1.000] | 0.996 [0.984, 1.000] | 0.989 [0.962, 1.000] | 0.356 [0.201, 0.528] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2            | height+wingspan to duration  | 0.199 [0.079, 0.349] | 0.995 [0.978, 1.000] | 0.960 [0.897, 0.996] | 0.995 [0.978, 1.000] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2            | all to the pooled test items | 1.000 [1.000, 1.000] | 0.940 [0.902, 0.971] | 1.000 [1.000, 1.000] | 0.927 [0.882, 0.966] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| gpt2-medium     | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.145 [0.049, 0.264] | 1.000 [1.000, 1.000] | 0.838 [0.718, 0.941] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2-medium     | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.984 [0.940, 1.000] | 0.956 [0.877, 1.000] | 0.935 [0.840, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-medium     | height+wingspan to duration  | 0.924 [0.847, 0.981] | 1.000 [1.000, 1.000] | 0.967 [0.911, 1.000] | 0.967 [0.907, 1.000] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2-medium     | all to the pooled test items | 1.000 [1.000, 1.000] | 0.998 [0.994, 1.000] | 1.000 [1.000, 1.000] | 0.966 [0.937, 0.992] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| gpt2-large      | wingspan+duration to height  | 1.000 [1.000, 1.000] | 0.900 [0.805, 0.968] | 1.000 [1.000, 1.000] | 0.941 [0.865, 0.991] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2-large      | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.956 [0.892, 0.996] | 0.984 [0.942, 1.000] | 0.949 [0.846, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-large      | height+wingspan to duration  | 0.835 [0.703, 0.948] | 0.114 [0.029, 0.226] | 0.989 [0.957, 1.000] | 0.996 [0.984, 1.000] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2-large      | all to the pooled test items | 1.000 [1.000, 1.000] | 0.997 [0.989, 1.000] | 1.000 [1.000, 1.000] | 0.965 [0.926, 0.994] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |
| gpt2-xl         | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.722 [0.582, 0.847] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2-xl         | height+duration to wingspan  | 1.000 [1.000, 1.000] | 0.987 [0.962, 1.000] | 1.000 [1.000, 1.000] | 0.600 [0.434, 0.745] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-xl         | height+wingspan to duration  | 1.000 [1.000, 1.000] | 0.183 [0.069, 0.326] | 0.989 [0.965, 1.000] | 0.926 [0.832, 0.993] | 0.929 [0.827, 1.000] | 0.888 [0.786, 0.964] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2-xl         | all to the pooled test items | 1.000 [1.000, 1.000] | 0.900 [0.853, 0.942] | 1.000 [1.000, 1.000] | 0.948 [0.910, 0.976] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.602 [0.513, 0.695] | 0.597 [0.499, 0.699] |

- **On height, wingspan and the pooled questions the text baselines read 1.000, and so does the logistic reader in 49 of 60 cells.** With height or wingspan held out, and pooled, `text_word` and `text_char` read 1.000 [1.000, 1.000]. The logistic reader reads 1.000 in 49 of those 60 cells (16 of 20 for height, 13 of 20 for wingspan, all 20 pooled), and in none of the 80 cells of this table does its interval lie above that of `text_word`; it lies above that of `text_char` in 6, all on held-out duration.
- **The random control spans most of the range.** Over its 28 cells (seven train sets by four test sets) the random control of a checkpoint and template has a minimum between 0.011 and 0.838 and a maximum of 1.000 in all 20 pairs of checkpoint and template; it reads 1.000 in 154 of the 560 cells of `transfer.csv`. On the pooled questions it is 0.899 or higher in 18 of the 20 pairs (0.817 to 1.000; the three lowest are 0.817, 0.888 and 0.8996). The reader reads 1.000 there, and its interval lies above that of its random control in 16 of the 20 pooled cells (not above it, in the plain template, for `qwen3-0.6b-base`, `qwen3-1.7b`, `gpt2-medium` and `gpt2-large`), in 15 of 20 for height, 11 of 20 for wingspan and 7 of 20 for duration, and below it in 4 of the 80 cells.
- **Held-out duration is the one setting where the baselines are below 1.000 and the readers differ from them.** `text_word` reads 0.929 [0.827, 1.000], `text_char` 0.888 [0.786, 0.964], `length` 0.220 [0.075, 0.394] and `popularity` 0.120 [0.000, 0.271]. In the plain template the logistic reader is above the value of `text_word` for one checkpoint only (`gpt2-xl`, 1.000 [1.000, 1.000]), its interval lies below that of `text_word` for five (`qwen3-1.7b-base`, `qwen3-1.7b`, `qwen3-4b-base`, `qwen3-4b` and `gpt2`, 0.199 to 0.659), and the random control of the cell is at or above the reader for five checkpoints. In the concept template the reader is above the value of `text_word` for nine of the ten checkpoints (0.960 to 1.000; `qwen3-4b-base` is at 0.667 [0.495, 0.829]), its interval is above that of `text_char` for five of them (`qwen3-0.6b`, `qwen3-1.7b-base`, `qwen3-1.7b`, `qwen3-4b` and `gpt2-xl`) and always overlaps that of `text_word`, which reaches 1.000, and the random control of the cell is at or above the reader for four checkpoints and 0.900 or higher for seven. Two concept cells have a random control well below the reader: `qwen3-4b` (0.998 [0.989, 1.000] against 0.080 [0.014, 0.164]) and `qwen3-1.7b` (1.000 [1.000, 1.000] against 0.194 [0.087, 0.333]). This setting is where the value of the logistic reader is above the best baseline: in 10 of the 80 cells, all of them held-out duration. The people heights diagnostic for these readers (fitted on height and wingspan, concept template, `people.csv`) places the 120 people above the test buildings (people against buildings 0.9997 to 1.000 for the ten checkpoints) and nearer the breeds (position 0.844 to 0.991), so it does not support a reading of range against exact for them. The plain `gpt2-xl` reader (height and wingspan) places the people at position 0.899, people against buildings 1.000 and breeds against people 0.830, inside the plain random band of -0.201 to 1.044, so it does not support a range reading either.

Train on one family and test on another, logistic reader, random control and baselines:

| Model           | Train to test        | Plain: reader        | Plain: random        | Concept: reader      | Concept: random      | text_word            | text_char            | length               | popularity           |
| --------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| qwen3-0.6b-base | height to wingspan   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.682 [0.529, 0.835] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-0.6b-base | wingspan to height   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.654 [0.490, 0.824] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| qwen3-0.6b-base | height to duration   | 0.913 [0.796, 1.000] | 0.350 [0.182, 0.511] | 0.960 [0.870, 1.000] | 0.281 [0.128, 0.452] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| qwen3-0.6b-base | wingspan to duration | 0.726 [0.557, 0.867] | 0.962 [0.910, 0.995] | 0.995 [0.978, 1.000] | 0.060 [0.005, 0.133] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-0.6b-base | duration to height   | 0.775 [0.644, 0.892] | 0.745 [0.594, 0.880] | 0.864 [0.741, 0.963] | 0.670 [0.507, 0.832] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-0.6b-base | duration to wingspan | 0.849 [0.726, 0.946] | 0.311 [0.169, 0.469] | 0.965 [0.917, 0.996] | 0.015 [0.000, 0.051] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-0.6b      | height to wingspan   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.527 [0.358, 0.683] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-0.6b      | wingspan to height   | 1.000 [1.000, 1.000] | 0.968 [0.917, 1.000] | 1.000 [1.000, 1.000] | 0.949 [0.881, 0.995] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| qwen3-0.6b      | height to duration   | 0.839 [0.706, 0.951] | 0.600 [0.418, 0.766] | 0.937 [0.859, 0.995] | 0.150 [0.054, 0.278] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| qwen3-0.6b      | wingspan to duration | 0.754 [0.594, 0.891] | 0.181 [0.055, 0.346] | 0.978 [0.938, 1.000] | 0.841 [0.713, 0.946] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-0.6b      | duration to height   | 0.874 [0.770, 0.964] | 0.977 [0.930, 1.000] | 0.861 [0.741, 0.949] | 0.483 [0.321, 0.640] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-0.6b      | duration to wingspan | 0.982 [0.945, 1.000] | 0.904 [0.793, 0.985] | 0.984 [0.950, 1.000] | 0.445 [0.280, 0.627] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-1.7b-base | height to wingspan   | 1.000 [1.000, 1.000] | 0.849 [0.727, 0.947] | 1.000 [1.000, 1.000] | 0.771 [0.635, 0.898] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-1.7b-base | wingspan to height   | 1.000 [1.000, 1.000] | 0.994 [0.977, 1.000] | 1.000 [1.000, 1.000] | 0.730 [0.585, 0.861] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| qwen3-1.7b-base | height to duration   | 0.723 [0.575, 0.862] | 0.944 [0.860, 1.000] | 1.000 [1.000, 1.000] | 0.781 [0.629, 0.910] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| qwen3-1.7b-base | wingspan to duration | 0.366 [0.216, 0.544] | 0.505 [0.325, 0.702] | 1.000 [1.000, 1.000] | 0.987 [0.957, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-1.7b-base | duration to height   | 0.528 [0.359, 0.687] | 0.388 [0.239, 0.545] | 0.981 [0.946, 1.000] | 0.977 [0.926, 1.000] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-1.7b-base | duration to wingspan | 0.369 [0.217, 0.538] | 0.447 [0.272, 0.623] | 0.982 [0.944, 1.000] | 0.740 [0.572, 0.879] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-1.7b      | height to wingspan   | 1.000 [1.000, 1.000] | 0.995 [0.975, 1.000] | 0.989 [0.963, 1.000] | 0.995 [0.977, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-1.7b      | wingspan to height   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 [0.941, 1.000] | 0.971 [0.906, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| qwen3-1.7b      | height to duration   | 0.332 [0.165, 0.500] | 0.889 [0.763, 0.995] | 0.953 [0.855, 1.000] | 1.000 [1.000, 1.000] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| qwen3-1.7b      | wingspan to duration | 0.312 [0.147, 0.485] | 0.437 [0.262, 0.616] | 0.775 [0.625, 0.907] | 0.431 [0.278, 0.596] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-1.7b      | duration to height   | 0.152 [0.053, 0.276] | 0.613 [0.450, 0.759] | 0.807 [0.667, 0.924] | 0.614 [0.456, 0.754] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-1.7b      | duration to wingspan | 0.236 [0.105, 0.383] | 0.971 [0.919, 1.000] | 0.535 [0.355, 0.698] | 0.953 [0.880, 0.996] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-4b-base   | height to wingspan   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.687 [0.519, 0.835] | 0.815 [0.665, 0.923] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-4b-base   | wingspan to height   | 1.000 [1.000, 1.000] | 0.912 [0.809, 0.984] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| qwen3-4b-base   | height to duration   | 0.319 [0.170, 0.480] | 0.996 [0.984, 1.000] | 0.274 [0.123, 0.429] | 0.293 [0.138, 0.446] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| qwen3-4b-base   | wingspan to duration | 0.207 [0.073, 0.376] | 0.033 [0.000, 0.088] | 0.524 [0.353, 0.730] | 0.996 [0.984, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-4b-base   | duration to height   | 0.304 [0.152, 0.490] | 0.019 [0.000, 0.052] | 0.232 [0.118, 0.367] | 0.986 [0.953, 1.000] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-4b-base   | duration to wingspan | 0.231 [0.087, 0.386] | 0.322 [0.172, 0.484] | 0.751 [0.598, 0.898] | 0.984 [0.948, 1.000] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-4b        | height to wingspan   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.340 [0.195, 0.507] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| qwen3-4b        | wingspan to height   | 1.000 [1.000, 1.000] | 0.907 [0.814, 0.975] | 1.000 [1.000, 1.000] | 0.783 [0.658, 0.896] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| qwen3-4b        | height to duration   | 0.572 [0.371, 0.753] | 0.949 [0.836, 1.000] | 0.933 [0.842, 0.993] | 0.774 [0.620, 0.900] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| qwen3-4b        | wingspan to duration | 0.261 [0.102, 0.447] | 0.766 [0.622, 0.886] | 0.940 [0.867, 0.989] | 1.000 [1.000, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| qwen3-4b        | duration to height   | 0.432 [0.259, 0.597] | 0.133 [0.044, 0.246] | 0.743 [0.606, 0.869] | 0.801 [0.664, 0.917] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| qwen3-4b        | duration to wingspan | 0.615 [0.440, 0.776] | 0.902 [0.803, 0.975] | 0.722 [0.563, 0.868] | 0.645 [0.472, 0.790] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2            | height to wingspan   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.965 [0.885, 1.000] | 0.998 [0.989, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2            | wingspan to height   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.961 [0.876, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| gpt2            | height to duration   | 0.505 [0.325, 0.678] | 0.842 [0.712, 0.933] | 0.917 [0.809, 0.993] | 0.904 [0.816, 0.976] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| gpt2            | wingspan to duration | 0.161 [0.057, 0.317] | 0.844 [0.725, 0.944] | 0.819 [0.678, 0.927] | 0.962 [0.879, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2            | duration to height   | 0.422 [0.275, 0.577] | 0.061 [0.009, 0.132] | 0.890 [0.788, 0.967] | 0.800 [0.667, 0.915] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2            | duration to wingspan | 0.255 [0.117, 0.407] | 0.273 [0.117, 0.444] | 0.544 [0.363, 0.714] | 0.711 [0.554, 0.859] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-medium     | height to wingspan   | 0.925 [0.825, 1.000] | 1.000 [1.000, 1.000] | 0.989 [0.962, 1.000] | 0.873 [0.772, 0.955] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-medium     | wingspan to height   | 1.000 [1.000, 1.000] | 0.964 [0.911, 0.998] | 1.000 [1.000, 1.000] | 0.920 [0.833, 0.985] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| gpt2-medium     | height to duration   | 0.937 [0.841, 1.000] | 1.000 [1.000, 1.000] | 0.951 [0.875, 1.000] | 0.909 [0.816, 0.982] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| gpt2-medium     | wingspan to duration | 0.830 [0.705, 0.934] | 0.319 [0.179, 0.482] | 0.966 [0.912, 1.000] | 0.933 [0.831, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2-medium     | duration to height   | 0.961 [0.897, 0.997] | 0.774 [0.622, 0.896] | 0.922 [0.841, 0.981] | 0.864 [0.744, 0.966] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2-medium     | duration to wingspan | 0.682 [0.522, 0.835] | 0.656 [0.502, 0.805] | 0.525 [0.342, 0.695] | 0.916 [0.825, 0.982] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-large      | height to wingspan   | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.945 [0.840, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-large      | wingspan to height   | 1.000 [1.000, 1.000] | 0.693 [0.531, 0.830] | 1.000 [1.000, 1.000] | 0.912 [0.823, 0.983] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| gpt2-large      | height to duration   | 0.906 [0.805, 0.984] | 0.395 [0.243, 0.563] | 0.947 [0.881, 0.993] | 1.000 [1.000, 1.000] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| gpt2-large      | wingspan to duration | 0.529 [0.360, 0.703] | 1.000 [1.000, 1.000] | 0.982 [0.947, 1.000] | 0.998 [0.989, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2-large      | duration to height   | 0.839 [0.720, 0.933] | 0.783 [0.651, 0.895] | 0.962 [0.909, 0.996] | 0.894 [0.801, 0.964] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2-large      | duration to wingspan | 0.384 [0.213, 0.555] | 0.913 [0.813, 0.981] | 0.835 [0.683, 0.967] | 0.465 [0.303, 0.643] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-xl         | height to wingspan   | 1.000 [1.000, 1.000] | 0.991 [0.962, 1.000] | 1.000 [1.000, 1.000] | 0.715 [0.552, 0.850] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |
| gpt2-xl         | wingspan to height   | 1.000 [1.000, 1.000] | 0.990 [0.964, 1.000] | 1.000 [1.000, 1.000] | 0.620 [0.455, 0.768] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.558 [0.400, 0.707] | 0.648 [0.489, 0.789] |
| gpt2-xl         | height to duration   | 0.993 [0.972, 1.000] | 0.625 [0.458, 0.777] | 0.984 [0.952, 1.000] | 0.987 [0.960, 1.000] | 0.924 [0.827, 1.000] | 0.574 [0.399, 0.736] | 0.780 [0.606, 0.925] | 0.880 [0.729, 1.000] |
| gpt2-xl         | wingspan to duration | 0.984 [0.940, 1.000] | 0.263 [0.135, 0.431] | 0.971 [0.916, 1.000] | 0.996 [0.984, 1.000] | 0.960 [0.869, 1.000] | 0.848 [0.727, 0.947] | 0.220 [0.075, 0.394] | 0.120 [0.000, 0.271] |
| gpt2-xl         | duration to height   | 0.987 [0.960, 1.000] | 0.832 [0.701, 0.929] | 0.962 [0.904, 1.000] | 0.846 [0.729, 0.932] | 0.075 [0.000, 0.195] | 0.449 [0.276, 0.612] | 0.442 [0.293, 0.600] | 0.352 [0.211, 0.511] |
| gpt2-xl         | duration to wingspan | 0.953 [0.889, 0.996] | 0.996 [0.983, 1.000] | 0.918 [0.832, 0.982] | 0.420 [0.265, 0.592] | 0.485 [0.417, 0.554] | 0.825 [0.678, 0.929] | 0.444 [0.275, 0.606] | 0.031 [0.000, 0.112] |

- **Height and wingspan carry each other at 1.000, as the text baselines do.** The logistic reader reads 1.000 in 34 of the 40 cells (ten checkpoints, two templates, two directions) with `text_word` and `text_char` at 1.000 [1.000, 1.000] in both directions; the random control reads 1.000 in 11 of the 40.
- **`text_word` transfers to duration but not from it, and the readers are above the baselines in some cells.** To duration `text_word` reads 0.924 [0.827, 1.000] from height and 0.960 [0.869, 1.000] from wingspan, and `text_char` 0.574 [0.399, 0.736] and 0.848 [0.727, 0.947]. From duration to height `text_word` reads 0.075 [0.000, 0.195] and `text_char` 0.449 [0.276, 0.612], and from duration to wingspan 0.485 [0.417, 0.554] and 0.825 [0.678, 0.929]; `length` and `popularity` read 0.442, 0.352, 0.444 and 0.031. Over the 120 single-family cells (ten checkpoints, two templates, six pairs of families) the interval of the logistic reader lies above those of all four baselines in 16 and above that of its random control in 41. It lies above both in 5 cells, all of them fitted on duration: `qwen3-0.6b` in the concept template, to height (0.861 [0.741, 0.949]; random 0.483 [0.321, 0.640]) and to wingspan (0.984 [0.950, 1.000]; random 0.445 [0.280, 0.627]), `qwen3-1.7b-base` in the concept template to wingspan (0.982 [0.944, 1.000]; random 0.740 [0.572, 0.879]), and `gpt2-medium` and `gpt2-xl` in the plain template to height (0.961 [0.897, 0.997] and 0.987 [0.960, 1.000]; random 0.774 [0.622, 0.896] and 0.832 [0.701, 0.929]). These are 5 of 120 comparisons made without a correction for the number of cells, so they mark cells to look at again and not a pattern by themselves. The people heights diagnostic of the four readers behind them (fitted on duration alone, `people.csv`) gives positions of 0.529 (`qwen3-0.6b`), 0.392 (`qwen3-1.7b-base`), 0.319 (`gpt2-medium`) and 0.400 (`gpt2-xl`), breeds against people 0.804 to 0.905 and people against buildings 0.702 to 0.7775, so the breeds read above the people and the people above the buildings, and all four positions are inside the band of the random controls of the same template; the diagnostic neither supports nor excludes a reading of range against exact for these readers.
- **Inverted readers.** The interval of the logistic reader lies entirely below 0.5 in 12 of the 120 single-family cells and in 1 of the 80 cells of the table above; that of the random control in 15 and 11, and that of `text_word` in 20 (duration to height, the same for every checkpoint).

The other two direction finders on the first four cells, next to the random control and the best of the four baselines (the full baseline values are in the two tables above):

| Model           | Train to test                | Plain: pca_diff      | Plain: mean_diff     | Plain: random        | Concept: pca_diff    | Concept: mean_diff   | Concept: random      | Best baseline                  |
| --------------- | ---------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | ------------------------------ |
| qwen3-0.6b-base | wingspan+duration to height  | 0.942 [0.885, 0.987] | 1.000 [1.000, 1.000] | 0.043 [0.006, 0.098] | 1.000 [1.000, 1.000] | 0.997 [0.987, 1.000] | 0.843 [0.717, 0.949] | text_word 1.000 [1.000, 1.000] |
| qwen3-0.6b-base | height+duration to wingspan  | 0.998 [0.989, 1.000] | 1.000 [1.000, 1.000] | 0.520 [0.351, 0.693] | 0.964 [0.904, 0.998] | 1.000 [1.000, 1.000] | 0.807 [0.665, 0.926] | text_word 1.000 [1.000, 1.000] |
| qwen3-0.6b-base | height+wingspan to duration  | 0.853 [0.720, 0.958] | 0.850 [0.720, 0.956] | 0.011 [0.000, 0.043] | 0.984 [0.944, 1.000] | 0.984 [0.944, 1.000] | 1.000 [1.000, 1.000] | text_word 0.929 [0.827, 1.000] |
| qwen3-0.6b-base | all to the pooled test items | 1.000 [0.998, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.996, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.903 [0.851, 0.945] | text_word 1.000 [1.000, 1.000] |
| qwen3-0.6b      | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.959 [0.883, 1.000] | 0.978 [0.940, 1.000] | 0.857 [0.733, 0.945] | 0.865 [0.749, 0.946] | text_word 1.000 [1.000, 1.000] |
| qwen3-0.6b      | height+duration to wingspan  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.967 [0.915, 1.000] | 0.993 [0.971, 1.000] | 0.456 [0.281, 0.636] | text_word 1.000 [1.000, 1.000] |
| qwen3-0.6b      | height+wingspan to duration  | 0.764 [0.613, 0.898] | 0.764 [0.613, 0.898] | 0.207 [0.087, 0.348] | 0.962 [0.904, 0.996] | 0.487 [0.307, 0.682] | 0.960 [0.904, 0.996] | text_word 0.929 [0.827, 1.000] |
| qwen3-0.6b      | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.932 [0.887, 0.970] | 0.947 [0.912, 0.974] | 0.996 [0.988, 1.000] | 0.817 [0.744, 0.879] | text_word 1.000 [1.000, 1.000] |
| qwen3-1.7b-base | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.929 [0.857, 0.988] | 0.993 [0.971, 1.000] | 1.000 [1.000, 1.000] | 0.736 [0.598, 0.865] | text_word 1.000 [1.000, 1.000] |
| qwen3-1.7b-base | height+duration to wingspan  | 0.958 [0.888, 1.000] | 1.000 [1.000, 1.000] | 0.440 [0.281, 0.606] | 0.998 [0.989, 1.000] | 1.000 [1.000, 1.000] | 0.655 [0.495, 0.819] | text_word 1.000 [1.000, 1.000] |
| qwen3-1.7b-base | height+wingspan to duration  | 0.524 [0.357, 0.681] | 0.529 [0.365, 0.689] | 0.841 [0.713, 0.941] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.949 [0.877, 0.996] | text_word 0.929 [0.827, 1.000] |
| qwen3-1.7b-base | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.960 [0.929, 0.982] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.974 [0.947, 0.994] | text_word 1.000 [1.000, 1.000] |
| qwen3-1.7b      | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.943 [0.866, 0.993] | 0.867 [0.757, 0.946] | text_word 1.000 [1.000, 1.000] |
| qwen3-1.7b      | height+duration to wingspan  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.878 [0.768, 0.965] | 0.880 [0.765, 0.966] | 0.671 [0.513, 0.832] | 0.645 [0.481, 0.795] | text_word 1.000 [1.000, 1.000] |
| qwen3-1.7b      | height+wingspan to duration  | 0.313 [0.145, 0.487] | 0.313 [0.145, 0.487] | 0.246 [0.111, 0.404] | 0.996 [0.982, 1.000] | 0.995 [0.978, 1.000] | 0.194 [0.087, 0.333] | text_word 0.929 [0.827, 1.000] |
| qwen3-1.7b      | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.973 [0.948, 0.990] | 0.984 [0.964, 0.996] | 0.888 [0.839, 0.932] | text_word 1.000 [1.000, 1.000] |
| qwen3-4b-base   | wingspan+duration to height  | 0.972 [0.932, 0.997] | 1.000 [1.000, 1.000] | 0.078 [0.009, 0.170] | 0.990 [0.969, 1.000] | 0.910 [0.829, 0.976] | 0.813 [0.684, 0.923] | text_word 1.000 [1.000, 1.000] |
| qwen3-4b-base   | height+duration to wingspan  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.945 [0.870, 0.991] | 0.993 [0.972, 1.000] | 0.995 [0.978, 1.000] | 0.758 [0.594, 0.886] | text_word 1.000 [1.000, 1.000] |
| qwen3-4b-base   | height+wingspan to duration  | 0.187 [0.060, 0.342] | 0.190 [0.064, 0.350] | 0.996 [0.982, 1.000] | 0.826 [0.696, 0.939] | 0.804 [0.670, 0.927] | 0.346 [0.188, 0.515] | text_word 0.929 [0.827, 1.000] |
| qwen3-4b-base   | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.994 [0.984, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.978 [0.955, 0.996] | text_word 1.000 [1.000, 1.000] |
| qwen3-4b        | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.307 [0.174, 0.462] | 0.836 [0.720, 0.933] | 0.862 [0.738, 0.951] | 0.867 [0.757, 0.965] | text_word 1.000 [1.000, 1.000] |
| qwen3-4b        | height+duration to wingspan  | 0.989 [0.957, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.989, 1.000] | 0.889 [0.776, 0.973] | 0.907 [0.810, 0.976] | 0.962 [0.907, 1.000] | text_word 1.000 [1.000, 1.000] |
| qwen3-4b        | height+wingspan to duration  | 0.357 [0.196, 0.549] | 0.370 [0.205, 0.558] | 0.987 [0.956, 1.000] | 0.995 [0.978, 1.000] | 0.995 [0.975, 1.000] | 0.080 [0.014, 0.164] | text_word 0.929 [0.827, 1.000] |
| qwen3-4b        | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.949 [0.909, 0.982] | 0.995 [0.987, 1.000] | 0.999 [0.998, 1.000] | 0.914 [0.857, 0.958] | text_word 1.000 [1.000, 1.000] |
| gpt2            | wingspan+duration to height  | 0.920 [0.841, 0.979] | 1.000 [1.000, 1.000] | 0.725 [0.579, 0.859] | 0.988 [0.957, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.991, 1.000] | text_word 1.000 [1.000, 1.000] |
| gpt2            | height+duration to wingspan  | 0.929 [0.833, 0.993] | 0.976 [0.918, 1.000] | 0.996 [0.984, 1.000] | 0.980 [0.938, 1.000] | 0.998 [0.989, 1.000] | 0.356 [0.201, 0.528] | text_word 1.000 [1.000, 1.000] |
| gpt2            | height+wingspan to duration  | 0.415 [0.244, 0.605] | 0.386 [0.213, 0.580] | 0.995 [0.978, 1.000] | 0.821 [0.683, 0.929] | 0.978 [0.930, 1.000] | 0.995 [0.978, 1.000] | text_word 0.929 [0.827, 1.000] |
| gpt2            | all to the pooled test items | 0.972 [0.947, 0.991] | 0.999 [0.998, 1.000] | 0.940 [0.902, 0.971] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.927 [0.882, 0.966] | text_word 1.000 [1.000, 1.000] |
| gpt2-medium     | wingspan+duration to height  | 0.991 [0.970, 1.000] | 1.000 [1.000, 1.000] | 0.145 [0.049, 0.264] | 0.930 [0.856, 0.983] | 0.986 [0.957, 1.000] | 0.838 [0.718, 0.941] | text_word 1.000 [1.000, 1.000] |
| gpt2-medium     | height+duration to wingspan  | 0.975 [0.931, 1.000] | 0.985 [0.958, 1.000] | 0.984 [0.940, 1.000] | 0.976 [0.933, 1.000] | 0.991 [0.967, 1.000] | 0.935 [0.840, 1.000] | text_word 1.000 [1.000, 1.000] |
| gpt2-medium     | height+wingspan to duration  | 0.967 [0.913, 1.000] | 0.975 [0.924, 1.000] | 1.000 [1.000, 1.000] | 0.993 [0.967, 1.000] | 1.000 [1.000, 1.000] | 0.967 [0.907, 1.000] | text_word 0.929 [0.827, 1.000] |
| gpt2-medium     | all to the pooled test items | 0.999 [0.997, 1.000] | 1.000 [0.998, 1.000] | 0.998 [0.994, 1.000] | 0.989 [0.976, 0.997] | 0.999 [0.997, 1.000] | 0.966 [0.937, 0.992] | text_word 1.000 [1.000, 1.000] |
| gpt2-large      | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.805, 0.968] | 0.991 [0.969, 1.000] | 0.997 [0.987, 1.000] | 0.941 [0.865, 0.991] | text_word 1.000 [1.000, 1.000] |
| gpt2-large      | height+duration to wingspan  | 0.980 [0.939, 1.000] | 0.944 [0.859, 0.998] | 0.956 [0.892, 0.996] | 0.978 [0.930, 1.000] | 0.987 [0.958, 1.000] | 0.949 [0.846, 1.000] | text_word 1.000 [1.000, 1.000] |
| gpt2-large      | height+wingspan to duration  | 0.913 [0.815, 0.985] | 0.895 [0.786, 0.981] | 0.114 [0.029, 0.226] | 0.976 [0.938, 1.000] | 0.980 [0.946, 1.000] | 0.996 [0.984, 1.000] | text_word 0.929 [0.827, 1.000] |
| gpt2-large      | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.989, 1.000] | 0.996 [0.988, 1.000] | 0.998 [0.992, 1.000] | 0.965 [0.926, 0.994] | text_word 1.000 [1.000, 1.000] |
| gpt2-xl         | wingspan+duration to height  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.722 [0.582, 0.847] | text_word 1.000 [1.000, 1.000] |
| gpt2-xl         | height+duration to wingspan  | 0.987 [0.957, 1.000] | 1.000 [1.000, 1.000] | 0.987 [0.962, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.600 [0.434, 0.745] | text_word 1.000 [1.000, 1.000] |
| gpt2-xl         | height+wingspan to duration  | 0.996 [0.984, 1.000] | 0.996 [0.984, 1.000] | 0.183 [0.069, 0.326] | 0.982 [0.948, 1.000] | 0.984 [0.951, 1.000] | 0.926 [0.832, 0.993] | text_word 0.929 [0.827, 1.000] |
| gpt2-xl         | all to the pooled test items | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.853, 0.942] | 0.999 [0.995, 1.000] | 0.999 [0.998, 1.000] | 0.948 [0.910, 0.976] | text_word 1.000 [1.000, 1.000] |

- **pca_diff and mean_diff follow the logistic reader on held-out duration.** In the plain template they read 0.187 to 0.996 (pca_diff) and 0.190 to 0.996 (mean_diff) over the ten checkpoints, above the value of `text_word` (0.929) for two of them each, and in the concept template 0.821 to 1.000 and 0.487 to 1.000, above it for eight each (`qwen3-0.6b` is the low mean_diff value, 0.487 [0.307, 0.682], with a logistic reader at 0.991 [0.971, 1.000]).
- **The same single setting stands out.** Held-out duration in the concept template is where the readers of all three finders are above the value of `text_word` for most checkpoints (nine logistic, eight pca_diff, eight mean_diff of ten). In the other three settings of the table the baselines are at 1.000 and no reader is above them. The random control of the same cell is 0.900 or higher for seven of the ten checkpoints in the concept template, so for those seven a random direction selected the same way also reads high.

Layers and curves. The layer chosen on validation, and the lowest and highest mean test AUROC over all layers (1 to the number of layers of the checkpoint) of the logistic reader and of the random control, in the first four cells:

| Model           | Train to test                | Plain: reader layer | Plain: reader over layers | Plain: random layer | Plain: random over layers | Concept: reader layer | Concept: reader over layers | Concept: random layer | Concept: random over layers |
| --------------- | ---------------------------- | ------------------- | ------------------------- | ------------------- | ------------------------- | --------------------- | --------------------------- | --------------------- | --------------------------- |
| qwen3-0.6b-base | wingspan+duration to height  | L1                  | 0.995 to 1.000            | L19                 | 0.181 to 0.839            | L1                    | 0.988 to 1.000              | L15                   | 0.465 to 0.919              |
| qwen3-0.6b-base | height+duration to wingspan  | L1                  | 0.922 to 1.000            | L19                 | 0.293 to 0.786            | L1                    | 0.849 to 1.000              | L21                   | 0.269 to 0.736              |
| qwen3-0.6b-base | height+wingspan to duration  | L1                  | 0.155 to 1.000            | L4                  | 0.086 to 0.878            | L1                    | 0.510 to 1.000              | L26                   | 0.381 to 0.899              |
| qwen3-0.6b-base | all to the pooled test items | L1                  | 1.000 to 1.000            | L8                  | 0.490 to 0.781            | L1                    | 1.000 to 1.000              | L15                   | 0.565 to 0.766              |
| qwen3-0.6b      | wingspan+duration to height  | L1                  | 0.999 to 1.000            | L17                 | 0.296 to 0.930            | L1                    | 0.846 to 1.000              | L16                   | 0.297 to 0.776              |
| qwen3-0.6b      | height+duration to wingspan  | L1                  | 1.000 to 1.000            | L21                 | 0.208 to 0.927            | L1                    | 0.613 to 1.000              | L20                   | 0.335 to 0.764              |
| qwen3-0.6b      | height+wingspan to duration  | L1                  | 0.166 to 0.996            | L22                 | 0.068 to 0.903            | L1                    | 0.066 to 1.000              | L1                    | 0.235 to 0.872              |
| qwen3-0.6b      | all to the pooled test items | L1                  | 1.000 to 1.000            | L28                 | 0.534 to 0.768            | L1                    | 0.997 to 1.000              | L2                    | 0.541 to 0.684              |
| qwen3-1.7b-base | wingspan+duration to height  | L1                  | 0.997 to 1.000            | L14                 | 0.426 to 0.839            | L1                    | 0.993 to 1.000              | L10                   | 0.490 to 0.872              |
| qwen3-1.7b-base | height+duration to wingspan  | L1                  | 0.961 to 1.000            | L5                  | 0.312 to 0.724            | L1                    | 0.917 to 1.000              | L15                   | 0.356 to 0.863              |
| qwen3-1.7b-base | height+wingspan to duration  | L1                  | 0.489 to 1.000            | L2                  | 0.328 to 0.955            | L1                    | 0.996 to 1.000              | L26                   | 0.253 to 0.939              |
| qwen3-1.7b-base | all to the pooled test items | L1                  | 1.000 to 1.000            | L18                 | 0.582 to 0.764            | L1                    | 1.000 to 1.000              | L24                   | 0.521 to 0.801              |
| qwen3-1.7b      | wingspan+duration to height  | L1                  | 1.000 to 1.000            | L1                  | 0.287 to 0.873            | L1                    | 0.946 to 1.000              | L11                   | 0.304 to 0.994              |
| qwen3-1.7b      | height+duration to wingspan  | L1                  | 0.892 to 1.000            | L28                 | 0.088 to 0.908            | L1                    | 0.617 to 1.000              | L22                   | 0.271 to 0.761              |
| qwen3-1.7b      | height+wingspan to duration  | L1                  | 0.388 to 1.000            | L3                  | 0.216 to 0.758            | L1                    | 0.856 to 1.000              | L14                   | 0.296 to 0.792              |
| qwen3-1.7b      | all to the pooled test items | L1                  | 1.000 to 1.000            | L3                  | 0.562 to 0.783            | L1                    | 0.999 to 1.000              | L27                   | 0.535 to 0.825              |
| qwen3-4b-base   | wingspan+duration to height  | L1                  | 0.739 to 1.000            | L14                 | 0.272 to 0.945            | L1                    | 0.976 to 1.000              | L17                   | 0.172 to 0.811              |
| qwen3-4b-base   | height+duration to wingspan  | L1                  | 0.963 to 1.000            | L11                 | 0.228 to 0.791            | L2                    | 0.905 to 1.000              | L35                   | 0.348 to 0.884              |
| qwen3-4b-base   | height+wingspan to duration  | L1                  | 0.379 to 1.000            | L3                  | 0.212 to 0.903            | L2                    | 0.393 to 1.000              | L1                    | 0.232 to 0.897              |
| qwen3-4b-base   | all to the pooled test items | L1                  | 1.000 to 1.000            | L3                  | 0.524 to 0.751            | L1                    | 1.000 to 1.000              | L16                   | 0.538 to 0.812              |
| qwen3-4b        | wingspan+duration to height  | L1                  | 1.000 to 1.000            | L14                 | 0.254 to 0.830            | L1                    | 0.968 to 1.000              | L25                   | 0.346 to 0.976              |
| qwen3-4b        | height+duration to wingspan  | L1                  | 1.000 to 1.000            | L26                 | 0.354 to 0.955            | L1                    | 0.949 to 1.000              | L35                   | 0.251 to 0.860              |
| qwen3-4b        | height+wingspan to duration  | L1                  | 0.629 to 1.000            | L22                 | 0.141 to 0.833            | L1                    | 0.996 to 1.000              | L6                    | 0.291 to 0.882              |
| qwen3-4b        | all to the pooled test items | L1                  | 1.000 to 1.000            | L20                 | 0.571 to 0.801            | L1                    | 1.000 to 1.000              | L33                   | 0.524 to 0.803              |
| gpt2            | wingspan+duration to height  | L1                  | 1.000 to 1.000            | L10                 | 0.378 to 0.717            | L2                    | 0.998 to 1.000              | L10                   | 0.428 to 0.894              |
| gpt2            | height+duration to wingspan  | L1                  | 0.996 to 1.000            | L3                  | 0.424 to 0.801            | L9                    | 0.910 to 1.000              | L7                    | 0.392 to 0.748              |
| gpt2            | height+wingspan to duration  | L1                  | 0.000 to 0.983            | L12                 | 0.288 to 0.884            | L1                    | 0.912 to 1.000              | L3                    | 0.263 to 0.841              |
| gpt2            | all to the pooled test items | L1                  | 1.000 to 1.000            | L1                  | 0.552 to 0.752            | L2                    | 1.000 to 1.000              | L3                    | 0.601 to 0.732              |
| gpt2-medium     | wingspan+duration to height  | L1                  | 1.000 to 1.000            | L20                 | 0.339 to 0.962            | L1                    | 0.974 to 1.000              | L16                   | 0.423 to 0.761              |
| gpt2-medium     | height+duration to wingspan  | L3                  | 0.903 to 1.000            | L7                  | 0.307 to 0.795            | L2                    | 0.945 to 1.000              | L1                    | 0.243 to 0.832              |
| gpt2-medium     | height+wingspan to duration  | L1                  | 0.925 to 1.000            | L7                  | 0.276 to 0.913            | L1                    | 0.968 to 1.000              | L2                    | 0.232 to 0.916              |
| gpt2-medium     | all to the pooled test items | L1                  | 1.000 to 1.000            | L11                 | 0.603 to 0.859            | L1                    | 0.997 to 1.000              | L6                    | 0.577 to 0.903              |
| gpt2-large      | wingspan+duration to height  | L1                  | 1.000 to 1.000            | L17                 | 0.374 to 0.953            | L1                    | 0.962 to 1.000              | L22                   | 0.537 to 0.846              |
| gpt2-large      | height+duration to wingspan  | L1                  | 0.993 to 1.000            | L19                 | 0.244 to 0.834            | L1                    | 0.951 to 0.999              | L18                   | 0.308 to 0.862              |
| gpt2-large      | height+wingspan to duration  | L1                  | 0.524 to 1.000            | L17                 | 0.172 to 0.955            | L1                    | 0.961 to 1.000              | L17                   | 0.430 to 0.910              |
| gpt2-large      | all to the pooled test items | L1                  | 1.000 to 1.000            | L23                 | 0.554 to 0.839            | L1                    | 0.996 to 1.000              | L28                   | 0.569 to 0.830              |
| gpt2-xl         | wingspan+duration to height  | L1                  | 0.983 to 1.000            | L14                 | 0.250 to 0.941            | L1                    | 0.980 to 1.000              | L11                   | 0.319 to 0.777              |
| gpt2-xl         | height+duration to wingspan  | L1                  | 1.000 to 1.000            | L7                  | 0.243 to 0.793            | L1                    | 0.951 to 1.000              | L26                   | 0.225 to 0.848              |
| gpt2-xl         | height+wingspan to duration  | L1                  | 0.771 to 1.000            | L18                 | 0.208 to 0.928            | L1                    | 0.958 to 1.000              | L4                    | 0.291 to 0.896              |
| gpt2-xl         | all to the pooled test items | L1                  | 1.000 to 1.000            | L31                 | 0.559 to 0.841            | L1                    | 1.000 to 1.000              | L6                    | 0.576 to 0.788              |

- **The chosen layer is the first one for the logistic reader.** It is L1 in 131 of its 140 choices (ten checkpoints, two templates, seven train sets) and in all 14 for seven checkpoints; the exceptions are `qwen3-4b-base` (3 choices, at most L2), `gpt2` (4, at most L9) and `gpt2-medium` (2, at most L3). The pca_diff choices are L1 in 65 of 140 (in all 14 for `gpt2-xl`) and the mean_diff choices in 81, and the random control's layer is spread over the model (L1 in 10 of 140, the median of its 14 choices between layer 4.5 for `gpt2` and layer 22 for `qwen3-4b`). The stored files do not hold the validation AUROC, so ties at the first layer cannot be read from them; the selection takes the first layer on ties.
- **On held-out duration the layer matters.** The mean test AUROC of the logistic reader over the layers has a minimum below 0.5 in 8 of the 20 pairs of checkpoint and template (down to 0.000 for `gpt2` in the plain template) and a maximum of at least 0.983 in all 20, against a maximum of 0.758 to 0.955 for the random direction, which is 0.900 or higher in 9 of the 20; `text_word` reads 0.929 on held-out duration. A maximum over layers is chosen on the test questions, so it is a ceiling and not a result.
- **When the pooled questions are scored the reader curve is flat and the random one is not.** For the pooled setting the mean AUROC of the logistic reader is 0.996 or higher at every layer in both templates for all ten checkpoints, and the random direction runs from 0.490 to 0.903 over the layers; `text_word` and `text_char` read 1.000 on the pooled questions.

The people heights as test questions, readers fitted on all three families. `Breeds against people` is the AUROC of the test breeds against the 120 people: high when the people score lower than the breeds, as a reader of range against exact gives. `People against buildings` is the AUROC of the people against the test buildings: near 0.5 when they score like the buildings, and high when they score above them, as a reader of living against artifact gives. `Position` is the mean score of the people minus that of the buildings, over the mean of the breeds minus that of the buildings: 0 when the people read like buildings (exact) and 1 when they read like breeds (range).

| Model           | Reader    | Plain: layer | Plain: breeds against people | Plain: people against buildings | Plain: position | Concept: layer | Concept: breeds against people | Concept: people against buildings | Concept: position |
| --------------- | --------- | ------------ | ---------------------------- | ------------------------------- | --------------- | -------------- | ------------------------------ | --------------------------------- | ----------------- |
| qwen3-0.6b-base | logistic  | L1           | 0.792 [0.691, 0.876]         | 1.000 [1.000, 1.000]            | 0.882           | L1             | 0.768 [0.666, 0.859]           | 1.000 [1.000, 1.000]              | 0.878             |
| qwen3-0.6b-base | pca_diff  | L10          | 1.000 [1.000, 1.000]         | 0.965 [0.888, 1.000]            | 0.368           | L1             | 0.735 [0.629, 0.841]           | 0.998 [0.994, 1.000]              | 0.839             |
| qwen3-0.6b-base | mean_diff | L3           | 0.926 [0.864, 0.975]         | 1.000 [1.000, 1.000]            | 0.841           | L1             | 0.740 [0.635, 0.838]           | 1.000 [0.999, 1.000]              | 0.855             |
| qwen3-0.6b-base | random    | L8           | 0.985 [0.955, 0.999]         | 0.989 [0.970, 1.000]            | 0.464           | L15            | 0.532 [0.408, 0.646]           | 0.960 [0.923, 0.987]              | 0.920             |
| qwen3-0.6b      | logistic  | L1           | 0.797 [0.688, 0.888]         | 1.000 [1.000, 1.000]            | 0.930           | L1             | 0.790 [0.678, 0.896]           | 1.000 [0.998, 1.000]              | 0.871             |
| qwen3-0.6b      | pca_diff  | L3           | 0.907 [0.850, 0.957]         | 1.000 [1.000, 1.000]            | 0.930           | L1             | 0.602 [0.461, 0.756]           | 0.998 [0.993, 1.000]              | 0.946             |
| qwen3-0.6b      | mean_diff | L3           | 0.927 [0.882, 0.966]         | 1.000 [1.000, 1.000]            | 0.903           | L4             | 0.888 [0.820, 0.941]           | 0.887 [0.804, 0.966]              | 0.566             |
| qwen3-0.6b      | random    | L28          | 0.570 [0.441, 0.693]         | 0.869 [0.803, 0.925]            | 0.871           | L2             | 0.396 [0.266, 0.535]           | 0.981 [0.959, 0.996]              | 1.102             |
| qwen3-1.7b-base | logistic  | L1           | 0.798 [0.720, 0.863]         | 1.000 [1.000, 1.000]            | 0.937           | L1             | 0.873 [0.809, 0.926]           | 1.000 [1.000, 1.000]              | 0.810             |
| qwen3-1.7b-base | pca_diff  | L5           | 0.997 [0.991, 1.000]         | 1.000 [1.000, 1.000]            | 0.854           | L2             | 0.628 [0.518, 0.733]           | 1.000 [1.000, 1.000]              | 0.936             |
| qwen3-1.7b-base | mean_diff | L1           | 0.589 [0.489, 0.683]         | 1.000 [1.000, 1.000]            | 0.990           | L2             | 0.676 [0.568, 0.772]           | 1.000 [1.000, 1.000]              | 0.901             |
| qwen3-1.7b-base | random    | L18          | 1.000 [1.000, 1.000]         | 0.818 [0.718, 0.901]            | 0.248           | L24            | 1.000 [0.998, 1.000]           | 0.667 [0.565, 0.761]              | 0.093             |
| qwen3-1.7b      | logistic  | L1           | 0.803 [0.723, 0.879]         | 1.000 [1.000, 1.000]            | 0.954           | L1             | 0.687 [0.579, 0.789]           | 0.999 [0.996, 1.000]              | 0.914             |
| qwen3-1.7b      | pca_diff  | L4           | 0.760 [0.658, 0.856]         | 1.000 [1.000, 1.000]            | 0.968           | L13            | 0.937 [0.881, 0.983]           | 0.982 [0.955, 1.000]              | 0.613             |
| qwen3-1.7b      | mean_diff | L3           | 0.889 [0.810, 0.948]         | 1.000 [1.000, 1.000]            | 0.941           | L10            | 0.929 [0.885, 0.972]           | 0.965 [0.920, 0.997]              | 0.609             |
| qwen3-1.7b      | random    | L3           | 0.555 [0.402, 0.706]         | 1.000 [1.000, 1.000]            | 0.947           | L27            | 0.859 [0.792, 0.920]           | 1.000 [1.000, 1.000]              | 0.820             |
| qwen3-4b-base   | logistic  | L1           | 0.707 [0.600, 0.805]         | 1.000 [1.000, 1.000]            | 0.941           | L1             | 0.680 [0.557, 0.795]           | 1.000 [1.000, 1.000]              | 0.919             |
| qwen3-4b-base   | pca_diff  | L6           | 0.989 [0.976, 0.998]         | 0.930 [0.871, 0.973]            | 0.434           | L24            | 1.000 [1.000, 1.000]           | 0.994 [0.984, 1.000]              | 0.502             |
| qwen3-4b-base   | mean_diff | L2           | 0.833 [0.745, 0.911]         | 1.000 [1.000, 1.000]            | 0.929           | L4             | 0.998 [0.993, 1.000]           | 0.806 [0.707, 0.895]              | 0.294             |
| qwen3-4b-base   | random    | L3           | 0.177 [0.099, 0.263]         | 1.000 [1.000, 1.000]            | 1.253           | L16            | 0.776 [0.670, 0.872]           | 0.989 [0.971, 0.999]              | 0.759             |
| qwen3-4b        | logistic  | L1           | 0.839 [0.761, 0.907]         | 1.000 [1.000, 1.000]            | 0.884           | L1             | 0.841 [0.767, 0.906]           | 0.987 [0.957, 1.000]              | 0.762             |
| qwen3-4b        | pca_diff  | L3           | 0.920 [0.849, 0.973]         | 1.000 [1.000, 1.000]            | 0.942           | L23            | 0.998 [0.993, 1.000]           | 0.871 [0.798, 0.929]              | 0.250             |
| qwen3-4b        | mean_diff | L1           | 0.791 [0.701, 0.867]         | 1.000 [1.000, 1.000]            | 0.920           | L4             | 0.961 [0.926, 0.987]           | 0.840 [0.731, 0.925]              | 0.361             |
| qwen3-4b        | random    | L20          | 0.981 [0.959, 0.997]         | 0.954 [0.905, 0.994]            | 0.526           | L33            | 0.996 [0.985, 1.000]           | 0.944 [0.901, 0.978]              | 0.356             |
| gpt2            | logistic  | L1           | 0.841 [0.759, 0.912]         | 1.000 [1.000, 1.000]            | 0.889           | L2             | 0.843 [0.770, 0.905]           | 0.998 [0.993, 1.000]              | 0.763             |
| gpt2            | pca_diff  | L4           | 0.874 [0.815, 0.930]         | 0.408 [0.316, 0.502]            | -0.119          | L11            | 0.979 [0.954, 0.997]           | 0.624 [0.507, 0.727]              | 0.189             |
| gpt2            | mean_diff | L3           | 0.850 [0.787, 0.908]         | 1.000 [1.000, 1.000]            | 0.758           | L11            | 0.980 [0.958, 0.997]           | 0.896 [0.824, 0.948]              | 0.393             |
| gpt2            | random    | L1           | 0.855 [0.767, 0.924]         | 0.986 [0.971, 0.997]            | 0.724           | L3             | 0.540 [0.415, 0.669]           | 0.934 [0.871, 0.979]              | 0.874             |
| gpt2-medium     | logistic  | L1           | 0.791 [0.705, 0.872]         | 1.000 [1.000, 1.000]            | 0.859           | L1             | 0.891 [0.821, 0.946]           | 1.000 [1.000, 1.000]              | 0.794             |
| gpt2-medium     | pca_diff  | L5           | 0.874 [0.814, 0.929]         | 0.995 [0.986, 1.000]            | 0.720           | L22            | 0.985 [0.967, 0.998]           | 0.361 [0.244, 0.478]              | -0.188            |
| gpt2-medium     | mean_diff | L4           | 0.881 [0.821, 0.933]         | 1.000 [1.000, 1.000]            | 0.764           | L22            | 0.983 [0.961, 0.998]           | 0.800 [0.686, 0.891]              | 0.286             |
| gpt2-medium     | random    | L11          | 0.998 [0.993, 1.000]         | 0.898 [0.830, 0.957]            | 0.374           | L6             | 0.622 [0.504, 0.745]           | 0.891 [0.787, 0.971]              | 0.890             |
| gpt2-large      | logistic  | L1           | 0.855 [0.783, 0.916]         | 1.000 [1.000, 1.000]            | 0.832           | L1             | 0.854 [0.783, 0.913]           | 1.000 [1.000, 1.000]              | 0.810             |
| gpt2-large      | pca_diff  | L18          | 1.000 [1.000, 1.000]         | 0.524 [0.398, 0.645]            | 0.027           | L14            | 0.843 [0.767, 0.907]           | 0.823 [0.743, 0.891]              | 0.548             |
| gpt2-large      | mean_diff | L4           | 0.927 [0.879, 0.965]         | 1.000 [1.000, 1.000]            | 0.731           | L14            | 0.853 [0.776, 0.915]           | 0.920 [0.868, 0.966]              | 0.635             |
| gpt2-large      | random    | L23          | 0.991 [0.980, 0.999]         | 0.807 [0.717, 0.892]            | 0.289           | L28            | 0.771 [0.643, 0.877]           | 0.853 [0.763, 0.927]              | 0.674             |
| gpt2-xl         | logistic  | L1           | 0.903 [0.840, 0.954]         | 1.000 [1.000, 1.000]            | 0.832           | L1             | 0.852 [0.788, 0.910]           | 1.000 [1.000, 1.000]              | 0.876             |
| gpt2-xl         | pca_diff  | L1           | 0.872 [0.812, 0.928]         | 1.000 [1.000, 1.000]            | 0.824           | L1             | 0.838 [0.772, 0.897]           | 1.000 [1.000, 1.000]              | 0.852             |
| gpt2-xl         | mean_diff | L1           | 0.882 [0.822, 0.935]         | 1.000 [1.000, 1.000]            | 0.807           | L1             | 0.846 [0.782, 0.905]           | 1.000 [1.000, 1.000]              | 0.843             |
| gpt2-xl         | random    | L31          | 0.970 [0.934, 0.995]         | 0.679 [0.570, 0.780]            | 0.185           | L6             | 0.337 [0.219, 0.481]           | 0.952 [0.905, 0.985]              | 1.472             |

The same three numbers for the surface baselines fitted on all three families (the same for every model):

| Baseline   | Breeds against people | People against buildings | Position |
| ---------- | --------------------- | ------------------------ | -------- |
| text_word  | 0.377 [0.278, 0.463]  | 1.000 [1.000, 1.000]     | 1.051    |
| text_char  | 0.759 [0.652, 0.857]  | 1.000 [1.000, 1.000]     | 0.936    |
| length     | 0.565 [0.404, 0.713]  | 0.372 [0.241, 0.500]     | 2.109    |
| popularity | 0.000 [0.000, 0.000]  | 1.000 [1.000, 1.000]     | 13.233   |

- **The logistic readers place the people between the breeds and the buildings in all 20 pairs of checkpoint and template, nearer the breeds by mean score.** People against buildings is 0.987 to 1.000 and the position 0.762 (`qwen3-4b`, concept) to 0.954 (`qwen3-1.7b`, plain), so the people score above the buildings and nearer the breeds; breeds against people is 0.680 to 0.903 (every interval above 0.5), so the people still rank below the breeds. `text_word` and `text_char` fitted on the same questions also place the people above every building (people against buildings 1.000 for both) and at positions 1.051 and 0.936. The people do not score like the buildings, so the diagnostic does not support a reading of range against exact for the logistic reader.
- **pca_diff and mean_diff put the people nearer the buildings in some cells, mostly inside the random band.** Their positions run from -0.188 to 0.990, and 13 of the 40 are below 0.55. The band is the range of the positions of the random control over its seven train sets in the same template; it is wide (-2.380 to 1.543 for `qwen3-4b` in the concept template, 0.372 to 1.079 for `gpt2-medium`), and 34 of the 40 positions are inside it. The six outside are `qwen3-1.7b-base` in the concept template (pca_diff 0.936 and mean_diff 0.901 against a band of -1.942 to 0.530, on the side of the breeds), `qwen3-1.7b` in the concept template (0.613 and 0.609 against 0.624 to 2.462, just below the band and nearer the breeds than the buildings) and `gpt2-medium` in the concept template (pca_diff -0.188 and mean_diff 0.286 against 0.372 to 1.079). In that last pair the pca_diff reader scores the people below the buildings (people against buildings 0.361) and the breeds above the people (0.985), the one pair where a reader places the people on the exact side of every random direction of the template; the logistic reader of the same pair places them with the breeds (position 0.794).
- **The random control places the people above the buildings as well.** Fitted on all three families, its people against buildings is 0.667 to 1.000 over the 20 pairs and its position 0.093 to 1.472, so a high people against buildings is no evidence of a fitted direction, and the positions near 1 of the logistic readers lie inside that range.
- **What the diagnostic says.** In 19 of the 20 pairs of checkpoint and template no reader places the people like the buildings and clearly outside the random band, as a reader of range against exact would; the logistic readers and the text baselines place them above the buildings and nearer the breeds, and so does the random control in many cells. That picture is what a cue shared by the people and the breeds gives, such as a living subject or the wording of the question, and the diagnostic does not separate a fitted direction from a random one. The one exception, `gpt2-medium` in the concept template, is one pair of 20 for two finders.

### Links to knowledge and hedging (P10c)

P10c links the range direction of P10b to two other things in the same model, using only finished runs (the code is `p10c_links.py`; a run reads a P10b, a P9b and a P10a run of the same model, named in its `summary.json`). The reader is the logistic reader of the plain template fitted on all three families (`plain/logistic/all`), and its direction is the one stored for draw 0, fitted on the full train set (the stored item scores are averages over the five draws). P10b chose layer 1 on validation for all ten checkpoints (the first layer wins ties, and the validation AUROC is not stored), so the headline cosine and AUROC of each `summary.json` and the first knowledge table below are layer-1 values. They are the values at the layer that P10b selected and not a finding about where a link lives; the layer profile after that table gives every layer of the CSVs. Intervals are 95 percent intervals with `n_boot` 1,000, and the random bands use `n_random` 200 unit directions, both as in `config.json`.

The knowledge link has two measures per P9 type. The cosine is between the range direction (positive means range) and the direction of the P9 reader of that type (positive means the positive side of its contrast). In a `knowledge` contrast the known questions are on the positive side and the unknown ones on the negative side; in an `unknown_type` contrast the positive side holds the real unknowns (`real_vs_synthal`, `real_vs_generated`) or the pre-cutoff ones (`pre_vs_post_cutoff`). A negative cosine outside the band means the range direction points toward the negative side of that contrast. Its band is the 2.5th to 97.5th percentile of the cosine between the range direction and 200 random unit vectors. The AUROC is that of the stored P9 reader on the 720 P10 questions (360 range, 360 exact; the 120 people are not scored, because the P9 readers have no range side), with the range questions as the positive class, so a value below 0.5 means that the range questions score toward the negative side of the contrast. Its interval resamples the questions, and its band is the 2.5th to 97.5th percentile of the AUROC of 200 random unit directions at the same layer. Those directions are unoriented, so the band is symmetric about 0.5 up to sampling, and it is wide: at layer 1 its lower bound runs from 0.030 (`gpt2`) to 0.150 (`gpt2-medium`) and its upper bound from 0.866 (`gpt2-xl`) to 0.967 (`gpt2`). An AUROC is therefore compared with its band and never with 0.5. A cosine band is a different object: it holds 95 percent of random vectors, so about 5 percent of cells would fall outside it for a P9 direction that behaved like a random vector, but the P9 directions are not random vectors, the layers and the types are not independent, and no test is made. The P9 types and their contrasts are those of [Reading the direction (P9b)](#reading-the-direction-p9b); the Qwen3 checkpoints have 14 types, `gpt2` has 3, `gpt2-medium` 6 and the two larger GPT-2 sizes 15.

Layer-1 knowledge link, per model and P9 type, pooled over the three families (all 720 questions). The last column names the measure whose value lies outside its band (for the AUROC, whose interval lies outside it):

| Model           | P9 type              | Contrast     | Cosine | Cosine band      | AUROC                | Random band    | Outside a band |
| --------------- | -------------------- | ------------ | ------ | ---------------- | -------------------- | -------------- | -------------- |
| qwen3-0.6b-base | fictional_generated  | knowledge    | -0.015 | [-0.058, +0.059] | 0.171 [0.139, 0.203] | [0.057, 0.955] | none           |
| qwen3-0.6b-base | obscure_real         | knowledge    | +0.100 | [-0.058, +0.059] | 0.450 [0.407, 0.492] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | obscure_real_pop     | knowledge    | +0.071 | [-0.058, +0.059] | 0.438 [0.393, 0.483] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | triviaqa_known       | knowledge    | +0.064 | [-0.058, +0.059] | 0.485 [0.442, 0.526] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | unknowable_detail    | knowledge    | +0.017 | [-0.058, +0.059] | 0.226 [0.192, 0.260] | [0.057, 0.955] | none           |
| qwen3-0.6b-base | false_premise        | knowledge    | +0.061 | [-0.058, +0.059] | 0.374 [0.331, 0.414] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | kuq_ambiguous        | knowledge    | +0.116 | [-0.058, +0.059] | 0.521 [0.474, 0.563] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | kuq_controversial    | knowledge    | +0.071 | [-0.058, +0.059] | 0.208 [0.171, 0.245] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | kuq_counterfactual   | knowledge    | +0.013 | [-0.058, +0.059] | 0.290 [0.249, 0.327] | [0.057, 0.955] | none           |
| qwen3-0.6b-base | kuq_false_assumption | knowledge    | +0.066 | [-0.058, +0.059] | 0.272 [0.232, 0.311] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | kuq_future_unknown   | knowledge    | +0.039 | [-0.058, +0.059] | 0.239 [0.201, 0.275] | [0.057, 0.955] | none           |
| qwen3-0.6b-base | kuq_unsolved_problem | knowledge    | +0.071 | [-0.058, +0.059] | 0.216 [0.178, 0.253] | [0.057, 0.955] | cosine         |
| qwen3-0.6b-base | real_vs_synthal      | unknown_type | -0.033 | [-0.058, +0.059] | 0.164 [0.135, 0.195] | [0.057, 0.955] | none           |
| qwen3-0.6b-base | real_vs_generated    | unknown_type | -0.048 | [-0.058, +0.059] | 0.167 [0.137, 0.199] | [0.057, 0.955] | none           |
| qwen3-0.6b      | fictional_generated  | knowledge    | -0.031 | [-0.065, +0.064] | 0.150 [0.119, 0.181] | [0.065, 0.961] | none           |
| qwen3-0.6b      | obscure_real         | knowledge    | +0.009 | [-0.065, +0.064] | 0.063 [0.046, 0.081] | [0.065, 0.961] | none           |
| qwen3-0.6b      | obscure_real_pop     | knowledge    | +0.115 | [-0.065, +0.064] | 0.375 [0.334, 0.416] | [0.065, 0.961] | cosine         |
| qwen3-0.6b      | triviaqa_known       | knowledge    | -0.059 | [-0.065, +0.064] | 0.098 [0.076, 0.123] | [0.065, 0.961] | none           |
| qwen3-0.6b      | unknowable_detail    | knowledge    | -0.029 | [-0.065, +0.064] | 0.140 [0.114, 0.168] | [0.065, 0.961] | none           |
| qwen3-0.6b      | false_premise        | knowledge    | +0.044 | [-0.065, +0.064] | 0.262 [0.224, 0.301] | [0.065, 0.961] | none           |
| qwen3-0.6b      | kuq_ambiguous        | knowledge    | +0.112 | [-0.065, +0.064] | 0.533 [0.488, 0.576] | [0.065, 0.961] | cosine         |
| qwen3-0.6b      | kuq_controversial    | knowledge    | +0.027 | [-0.065, +0.064] | 0.182 [0.151, 0.214] | [0.065, 0.961] | none           |
| qwen3-0.6b      | kuq_counterfactual   | knowledge    | -0.042 | [-0.065, +0.064] | 0.257 [0.220, 0.292] | [0.065, 0.961] | none           |
| qwen3-0.6b      | kuq_false_assumption | knowledge    | +0.030 | [-0.065, +0.064] | 0.183 [0.152, 0.215] | [0.065, 0.961] | none           |
| qwen3-0.6b      | kuq_future_unknown   | knowledge    | -0.008 | [-0.065, +0.064] | 0.106 [0.083, 0.130] | [0.065, 0.961] | none           |
| qwen3-0.6b      | kuq_unsolved_problem | knowledge    | +0.041 | [-0.065, +0.064] | 0.179 [0.147, 0.211] | [0.065, 0.961] | none           |
| qwen3-0.6b      | real_vs_synthal      | unknown_type | -0.037 | [-0.065, +0.064] | 0.208 [0.175, 0.243] | [0.065, 0.961] | none           |
| qwen3-0.6b      | real_vs_generated    | unknown_type | -0.077 | [-0.065, +0.064] | 0.160 [0.129, 0.190] | [0.065, 0.961] | cosine         |
| qwen3-1.7b-base | fictional_generated  | knowledge    | +0.009 | [-0.042, +0.038] | 0.156 [0.127, 0.184] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | obscure_real         | knowledge    | +0.024 | [-0.042, +0.038] | 0.510 [0.466, 0.552] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | obscure_real_pop     | knowledge    | +0.004 | [-0.042, +0.038] | 0.779 [0.745, 0.813] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | triviaqa_known       | knowledge    | +0.028 | [-0.042, +0.038] | 0.384 [0.339, 0.423] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | unknowable_detail    | knowledge    | -0.004 | [-0.042, +0.038] | 0.267 [0.229, 0.305] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | false_premise        | knowledge    | +0.028 | [-0.042, +0.038] | 0.499 [0.456, 0.542] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | kuq_ambiguous        | knowledge    | +0.020 | [-0.042, +0.038] | 0.516 [0.471, 0.560] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | kuq_controversial    | knowledge    | +0.025 | [-0.042, +0.038] | 0.328 [0.288, 0.368] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | kuq_counterfactual   | knowledge    | +0.022 | [-0.042, +0.038] | 0.425 [0.383, 0.468] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | kuq_false_assumption | knowledge    | +0.038 | [-0.042, +0.038] | 0.411 [0.369, 0.454] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | kuq_future_unknown   | knowledge    | +0.017 | [-0.042, +0.038] | 0.198 [0.166, 0.231] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | kuq_unsolved_problem | knowledge    | +0.018 | [-0.042, +0.038] | 0.320 [0.283, 0.360] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | real_vs_synthal      | unknown_type | +0.021 | [-0.042, +0.038] | 0.378 [0.337, 0.420] | [0.106, 0.877] | none           |
| qwen3-1.7b-base | real_vs_generated    | unknown_type | +0.004 | [-0.042, +0.038] | 0.154 [0.125, 0.181] | [0.106, 0.877] | none           |
| qwen3-1.7b      | fictional_generated  | knowledge    | +0.006 | [-0.046, +0.040] | 0.269 [0.231, 0.305] | [0.103, 0.899] | none           |
| qwen3-1.7b      | obscure_real         | knowledge    | +0.015 | [-0.046, +0.040] | 0.303 [0.263, 0.341] | [0.103, 0.899] | none           |
| qwen3-1.7b      | obscure_real_pop     | knowledge    | +0.011 | [-0.046, +0.040] | 0.466 [0.424, 0.508] | [0.103, 0.899] | none           |
| qwen3-1.7b      | triviaqa_known       | knowledge    | +0.023 | [-0.046, +0.040] | 0.302 [0.260, 0.340] | [0.103, 0.899] | none           |
| qwen3-1.7b      | unknowable_detail    | knowledge    | +0.010 | [-0.046, +0.040] | 0.216 [0.183, 0.251] | [0.103, 0.899] | none           |
| qwen3-1.7b      | false_premise        | knowledge    | +0.021 | [-0.046, +0.040] | 0.587 [0.544, 0.627] | [0.103, 0.899] | none           |
| qwen3-1.7b      | kuq_ambiguous        | knowledge    | +0.014 | [-0.046, +0.040] | 0.578 [0.534, 0.621] | [0.103, 0.899] | none           |
| qwen3-1.7b      | kuq_controversial    | knowledge    | -0.012 | [-0.046, +0.040] | 0.379 [0.339, 0.421] | [0.103, 0.899] | none           |
| qwen3-1.7b      | kuq_counterfactual   | knowledge    | -0.003 | [-0.046, +0.040] | 0.448 [0.406, 0.493] | [0.103, 0.899] | none           |
| qwen3-1.7b      | kuq_false_assumption | knowledge    | +0.005 | [-0.046, +0.040] | 0.413 [0.372, 0.455] | [0.103, 0.899] | none           |
| qwen3-1.7b      | kuq_future_unknown   | knowledge    | -0.004 | [-0.046, +0.040] | 0.184 [0.155, 0.216] | [0.103, 0.899] | none           |
| qwen3-1.7b      | kuq_unsolved_problem | knowledge    | -0.010 | [-0.046, +0.040] | 0.343 [0.306, 0.384] | [0.103, 0.899] | none           |
| qwen3-1.7b      | real_vs_synthal      | unknown_type | -0.007 | [-0.046, +0.040] | 0.306 [0.267, 0.345] | [0.103, 0.899] | none           |
| qwen3-1.7b      | real_vs_generated    | unknown_type | +0.009 | [-0.046, +0.040] | 0.227 [0.191, 0.261] | [0.103, 0.899] | none           |
| qwen3-4b-base   | fictional_generated  | knowledge    | -0.019 | [-0.035, +0.037] | 0.237 [0.204, 0.273] | [0.128, 0.910] | none           |
| qwen3-4b-base   | obscure_real         | knowledge    | +0.008 | [-0.035, +0.037] | 0.352 [0.313, 0.395] | [0.128, 0.910] | none           |
| qwen3-4b-base   | obscure_real_pop     | knowledge    | +0.010 | [-0.035, +0.037] | 0.598 [0.555, 0.640] | [0.128, 0.910] | none           |
| qwen3-4b-base   | triviaqa_known       | knowledge    | +0.012 | [-0.035, +0.037] | 0.612 [0.574, 0.650] | [0.128, 0.910] | none           |
| qwen3-4b-base   | unknowable_detail    | knowledge    | +0.000 | [-0.035, +0.037] | 0.175 [0.143, 0.207] | [0.128, 0.910] | none           |
| qwen3-4b-base   | false_premise        | knowledge    | -0.041 | [-0.035, +0.037] | 0.508 [0.466, 0.551] | [0.128, 0.910] | cosine         |
| qwen3-4b-base   | kuq_ambiguous        | knowledge    | -0.046 | [-0.035, +0.037] | 0.452 [0.409, 0.493] | [0.128, 0.910] | cosine         |
| qwen3-4b-base   | kuq_controversial    | knowledge    | -0.042 | [-0.035, +0.037] | 0.342 [0.303, 0.382] | [0.128, 0.910] | cosine         |
| qwen3-4b-base   | kuq_counterfactual   | knowledge    | -0.013 | [-0.035, +0.037] | 0.415 [0.374, 0.455] | [0.128, 0.910] | none           |
| qwen3-4b-base   | kuq_false_assumption | knowledge    | -0.026 | [-0.035, +0.037] | 0.395 [0.354, 0.436] | [0.128, 0.910] | none           |
| qwen3-4b-base   | kuq_future_unknown   | knowledge    | -0.029 | [-0.035, +0.037] | 0.245 [0.210, 0.279] | [0.128, 0.910] | none           |
| qwen3-4b-base   | kuq_unsolved_problem | knowledge    | -0.032 | [-0.035, +0.037] | 0.318 [0.280, 0.356] | [0.128, 0.910] | none           |
| qwen3-4b-base   | real_vs_synthal      | unknown_type | -0.058 | [-0.035, +0.037] | 0.257 [0.222, 0.293] | [0.128, 0.910] | cosine         |
| qwen3-4b-base   | real_vs_generated    | unknown_type | -0.020 | [-0.035, +0.037] | 0.204 [0.175, 0.238] | [0.128, 0.910] | none           |
| qwen3-4b        | fictional_generated  | knowledge    | +0.007 | [-0.032, +0.040] | 0.172 [0.142, 0.203] | [0.124, 0.900] | none           |
| qwen3-4b        | obscure_real         | knowledge    | +0.013 | [-0.032, +0.040] | 0.183 [0.151, 0.217] | [0.124, 0.900] | none           |
| qwen3-4b        | obscure_real_pop     | knowledge    | +0.023 | [-0.032, +0.040] | 0.832 [0.802, 0.863] | [0.124, 0.900] | none           |
| qwen3-4b        | triviaqa_known       | knowledge    | +0.036 | [-0.032, +0.040] | 0.670 [0.629, 0.706] | [0.124, 0.900] | none           |
| qwen3-4b        | unknowable_detail    | knowledge    | +0.007 | [-0.032, +0.040] | 0.167 [0.137, 0.194] | [0.124, 0.900] | none           |
| qwen3-4b        | false_premise        | knowledge    | -0.008 | [-0.032, +0.040] | 0.547 [0.505, 0.589] | [0.124, 0.900] | none           |
| qwen3-4b        | kuq_ambiguous        | knowledge    | -0.035 | [-0.032, +0.040] | 0.490 [0.447, 0.533] | [0.124, 0.900] | cosine         |
| qwen3-4b        | kuq_controversial    | knowledge    | -0.032 | [-0.032, +0.040] | 0.428 [0.388, 0.467] | [0.124, 0.900] | none           |
| qwen3-4b        | kuq_counterfactual   | knowledge    | -0.020 | [-0.032, +0.040] | 0.417 [0.377, 0.457] | [0.124, 0.900] | none           |
| qwen3-4b        | kuq_false_assumption | knowledge    | -0.024 | [-0.032, +0.040] | 0.407 [0.366, 0.447] | [0.124, 0.900] | none           |
| qwen3-4b        | kuq_future_unknown   | knowledge    | -0.025 | [-0.032, +0.040] | 0.183 [0.153, 0.212] | [0.124, 0.900] | none           |
| qwen3-4b        | kuq_unsolved_problem | knowledge    | -0.022 | [-0.032, +0.040] | 0.343 [0.305, 0.381] | [0.124, 0.900] | none           |
| qwen3-4b        | real_vs_synthal      | unknown_type | -0.041 | [-0.032, +0.040] | 0.221 [0.191, 0.255] | [0.124, 0.900] | cosine         |
| qwen3-4b        | real_vs_generated    | unknown_type | -0.007 | [-0.032, +0.040] | 0.167 [0.141, 0.198] | [0.124, 0.900] | none           |
| gpt2            | real_vs_synthal      | unknown_type | +0.000 | [-0.080, +0.072] | 0.233 [0.200, 0.266] | [0.030, 0.967] | none           |
| gpt2            | real_vs_generated    | unknown_type | +0.048 | [-0.080, +0.072] | 0.250 [0.216, 0.285] | [0.030, 0.967] | none           |
| gpt2            | pre_vs_post_cutoff   | unknown_type | -0.114 | [-0.080, +0.072] | 0.178 [0.148, 0.208] | [0.030, 0.967] | cosine         |
| gpt2-medium     | fictional_generated  | knowledge    | -0.077 | [-0.064, +0.054] | 0.610 [0.567, 0.648] | [0.150, 0.884] | cosine         |
| gpt2-medium     | obscure_real         | knowledge    | -0.131 | [-0.064, +0.054] | 0.246 [0.212, 0.282] | [0.150, 0.884] | cosine         |
| gpt2-medium     | obscure_real_pop     | knowledge    | -0.132 | [-0.064, +0.054] | 0.173 [0.142, 0.202] | [0.150, 0.884] | cosine         |
| gpt2-medium     | real_vs_synthal      | unknown_type | +0.042 | [-0.064, +0.054] | 0.670 [0.632, 0.713] | [0.150, 0.884] | none           |
| gpt2-medium     | real_vs_generated    | unknown_type | -0.021 | [-0.064, +0.054] | 0.591 [0.550, 0.631] | [0.150, 0.884] | none           |
| gpt2-medium     | pre_vs_post_cutoff   | unknown_type | +0.022 | [-0.064, +0.054] | 0.842 [0.813, 0.871] | [0.150, 0.884] | none           |
| gpt2-large      | fictional_generated  | knowledge    | -0.046 | [-0.059, +0.051] | 0.395 [0.347, 0.439] | [0.125, 0.917] | none           |
| gpt2-large      | obscure_real         | knowledge    | -0.040 | [-0.059, +0.051] | 0.591 [0.550, 0.636] | [0.125, 0.917] | none           |
| gpt2-large      | obscure_real_pop     | knowledge    | -0.002 | [-0.059, +0.051] | 0.683 [0.643, 0.720] | [0.125, 0.917] | none           |
| gpt2-large      | triviaqa_known       | knowledge    | -0.026 | [-0.059, +0.051] | 0.823 [0.791, 0.855] | [0.125, 0.917] | none           |
| gpt2-large      | unknowable_detail    | knowledge    | -0.028 | [-0.059, +0.051] | 0.612 [0.571, 0.650] | [0.125, 0.917] | none           |
| gpt2-large      | false_premise        | knowledge    | -0.077 | [-0.059, +0.051] | 0.322 [0.281, 0.362] | [0.125, 0.917] | cosine         |
| gpt2-large      | kuq_ambiguous        | knowledge    | -0.037 | [-0.059, +0.051] | 0.331 [0.288, 0.369] | [0.125, 0.917] | none           |
| gpt2-large      | kuq_controversial    | knowledge    | -0.063 | [-0.059, +0.051] | 0.286 [0.244, 0.326] | [0.125, 0.917] | cosine         |
| gpt2-large      | kuq_counterfactual   | knowledge    | -0.086 | [-0.059, +0.051] | 0.430 [0.385, 0.470] | [0.125, 0.917] | cosine         |
| gpt2-large      | kuq_false_assumption | knowledge    | -0.073 | [-0.059, +0.051] | 0.368 [0.326, 0.408] | [0.125, 0.917] | cosine         |
| gpt2-large      | kuq_future_unknown   | knowledge    | -0.021 | [-0.059, +0.051] | 0.488 [0.444, 0.532] | [0.125, 0.917] | none           |
| gpt2-large      | kuq_unsolved_problem | knowledge    | -0.049 | [-0.059, +0.051] | 0.385 [0.344, 0.427] | [0.125, 0.917] | none           |
| gpt2-large      | real_vs_synthal      | unknown_type | -0.012 | [-0.059, +0.051] | 0.412 [0.368, 0.455] | [0.125, 0.917] | none           |
| gpt2-large      | real_vs_generated    | unknown_type | -0.019 | [-0.059, +0.051] | 0.334 [0.293, 0.375] | [0.125, 0.917] | none           |
| gpt2-large      | pre_vs_post_cutoff   | unknown_type | -0.026 | [-0.059, +0.051] | 0.184 [0.154, 0.216] | [0.125, 0.917] | none           |
| gpt2-xl         | fictional_generated  | knowledge    | +0.036 | [-0.045, +0.050] | 0.456 [0.410, 0.498] | [0.128, 0.866] | none           |
| gpt2-xl         | obscure_real         | knowledge    | -0.069 | [-0.045, +0.050] | 0.310 [0.270, 0.349] | [0.128, 0.866] | cosine         |
| gpt2-xl         | obscure_real_pop     | knowledge    | +0.024 | [-0.045, +0.050] | 0.714 [0.676, 0.750] | [0.128, 0.866] | none           |
| gpt2-xl         | triviaqa_known       | knowledge    | +0.010 | [-0.045, +0.050] | 0.637 [0.595, 0.677] | [0.128, 0.866] | none           |
| gpt2-xl         | unknowable_detail    | knowledge    | -0.051 | [-0.045, +0.050] | 0.729 [0.691, 0.767] | [0.128, 0.866] | cosine         |
| gpt2-xl         | false_premise        | knowledge    | +0.018 | [-0.045, +0.050] | 0.575 [0.534, 0.617] | [0.128, 0.866] | none           |
| gpt2-xl         | kuq_ambiguous        | knowledge    | +0.047 | [-0.045, +0.050] | 0.581 [0.538, 0.621] | [0.128, 0.866] | none           |
| gpt2-xl         | kuq_controversial    | knowledge    | +0.007 | [-0.045, +0.050] | 0.590 [0.554, 0.631] | [0.128, 0.866] | none           |
| gpt2-xl         | kuq_counterfactual   | knowledge    | -0.021 | [-0.045, +0.050] | 0.664 [0.623, 0.709] | [0.128, 0.866] | none           |
| gpt2-xl         | kuq_false_assumption | knowledge    | +0.053 | [-0.045, +0.050] | 0.543 [0.501, 0.586] | [0.128, 0.866] | cosine         |
| gpt2-xl         | kuq_future_unknown   | knowledge    | -0.007 | [-0.045, +0.050] | 0.558 [0.518, 0.598] | [0.128, 0.866] | none           |
| gpt2-xl         | kuq_unsolved_problem | knowledge    | +0.047 | [-0.045, +0.050] | 0.624 [0.586, 0.662] | [0.128, 0.866] | none           |
| gpt2-xl         | real_vs_synthal      | unknown_type | +0.058 | [-0.045, +0.050] | 0.748 [0.713, 0.781] | [0.128, 0.866] | cosine         |
| gpt2-xl         | real_vs_generated    | unknown_type | -0.005 | [-0.045, +0.050] | 0.475 [0.433, 0.521] | [0.128, 0.866] | none           |
| gpt2-xl         | pre_vs_post_cutoff   | unknown_type | -0.002 | [-0.045, +0.050] | 0.392 [0.344, 0.438] | [0.128, 0.866] | none           |

- **At layer 1 the range direction is nearly orthogonal to every P9 direction.** The 123 cosines have an absolute value of at most 0.132 (`gpt2-medium`, `obscure_real_pop`, -0.132). 29 of them lie outside their band, 12 above it and 17 below it (Qwen3 base checkpoints 12 of 42, Qwen3 post-trained 5 of 42, GPT-2 sizes 12 of 39). By checkpoint the 29 are 8 (`qwen3-0.6b-base`), 3 (`qwen3-0.6b`), 0 (`qwen3-1.7b-base`), 0 (`qwen3-1.7b`), 4 (`qwen3-4b-base`), 2 (`qwen3-4b`), 1 (`gpt2`), 3 (`gpt2-medium`), 4 (`gpt2-large`) and 4 (`gpt2-xl`).
- **At layer 1 no P9 reader separates the range questions from the exact ones beyond its random band.** The 123 AUROCs run from 0.063 (`qwen3-0.6b`, `obscure_real`) to 0.842 (`gpt2-medium`, `pre_vs_post_cutoff`), and in none of them does the interval lie outside the band. A value that looks large, such as `gpt2-medium` `pre_vs_post_cutoff` at 0.842 [0.813, 0.871] against a band of [0.150, 0.884], or `qwen3-4b` `obscure_real_pop` at 0.832 [0.802, 0.863] against a band of [0.124, 0.900], is inside its band, and so are the values far from 0.5 on the low side (0.063 [0.046, 0.081] for `qwen3-0.6b` `obscure_real`, against [0.065, 0.961]). The same holds for the `knowledge` and the `unknown_type` contrasts and for base and post-trained checkpoints.
- **Within a family the layer-1 readers do no better.** For each of the three families alone (the 120 range and the 120 exact questions of the family, all layer-1 cells of type and family, 369 in all) the interval lies below the band in 9 cells and above it in 1 (`gpt2-large`, duration); pooled over the families it is 0 of 123.
- **The knowledge link cannot tell a range-valued answer from a living subject.** In all three families the range side is a living thing and the exact side an artifact, and the P9 readers score no people, so the P10c files hold no people-heights diagnostic for the knowledge link. The diagnostic that exists is that of P10b (above): for the readers fitted on all three families the people score above the buildings and nearer the breeds in all 20 pairs of checkpoint and template (both templates; the P10c reader is the plain one), the random control also places them above the buildings there (people against buildings 0.667 to 1.000), and the diagnostic does not support a pure range reading of the reader. A cosine or an AUROC between the range direction and a P9 direction cannot be read as a link between "the answer is a range" and "the model does not know".

Layer profile of the knowledge link (a layer list of more than five cells is given as the span from its lowest to its highest layer, with gaps), all layers of `cosine.csv` and `knowledge.csv` (the AUROC of all 720 questions): the cells of type and layer whose cosine lies above or below its band, the cell with the largest absolute cosine and its band, and the cells whose AUROC interval lies below or above the random band of its layer:

| Model           | Cells | Cosine above band | Cosine below band | Largest abs. cosine (type, layer)      | Band there       | AUROC interval below band | AUROC interval above band |
| --------------- | ----- | ----------------- | ----------------- | -------------------------------------- | ---------------- | ------------------------- | ------------------------- |
| qwen3-0.6b-base | 392   | 33                | 47                | +0.175 (triviaqa_known, layer 21)      | [-0.062, +0.074] | 16 (layers 15 to 28)      | 0                         |
| qwen3-0.6b      | 392   | 28                | 73                | -0.160 (obscure_real, layer 5)         | [-0.073, +0.064] | 9 (layers 2 to 28)        | 0                         |
| qwen3-1.7b-base | 392   | 54                | 91                | -0.154 (false_premise, layer 15)       | [-0.044, +0.044] | 7 (layers 12 to 28)       | 0                         |
| qwen3-1.7b      | 392   | 107               | 51                | +0.211 (unknowable_detail, layer 28)   | [-0.041, +0.042] | 3 (layers 5, 12, 26)      | 1 (layers 8)              |
| qwen3-4b-base   | 504   | 18                | 202               | -0.135 (false_premise, layer 26)       | [-0.034, +0.034] | 4 (layers 12, 13, 14, 36) | 0                         |
| qwen3-4b        | 504   | 43                | 141               | -0.161 (real_vs_synthal, layer 7)      | [-0.031, +0.034] | 0                         | 0                         |
| gpt2            | 36    | 1                 | 6                 | -0.147 (real_vs_synthal, layer 2)      | [-0.063, +0.064] | 0                         | 0                         |
| gpt2-medium     | 144   | 13                | 21                | +0.137 (fictional_generated, layer 23) | [-0.056, +0.050] | 5 (layers 2, 5, 15, 16)   | 0                         |
| gpt2-large      | 540   | 13                | 133               | -0.201 (false_premise, layer 18)       | [-0.050, +0.054] | 13 (layers 20 to 30)      | 0                         |
| gpt2-xl         | 720   | 118               | 125               | -0.126 (false_premise, layer 26)       | [-0.048, +0.047] | 5 (layers 20, 28, 31, 32) | 0                         |

- **The cosines stay small at every layer.** Over the 4,016 cells of the ten checkpoints the largest absolute cosine is 0.211 (`qwen3-1.7b`, `unknowable_detail`, layer 28); the largest per checkpoint runs from 0.126 (`gpt2-xl`) to 0.211. 1,318 cells (32.8 percent) lie outside their band, 428 above and 890 below (Qwen3 base checkpoints 445 of 1,288, Qwen3 post-trained 443 of 1,288, GPT-2 sizes 430 of 1,440), 1,096 of 3,288 for the `knowledge` contrasts and 222 of 728 for the `unknown_type` ones. For nine of the ten checkpoints more cells lie below the band than above it (`qwen3-1.7b` is the exception, 107 above and 51 below), so in most outside cells the range direction points toward the negative side of the contrast; `qwen3-4b-base` (18 above, 202 below) and `gpt2-large` (13 above, 133 below) are the extremes.
- **An AUROC interval is outside its random band in 63 of 4,016 cells (27 for the Qwen3 base checkpoints, 13 for the post-trained ones, 23 for the GPT-2 sizes, out of 1,288, 1,288 and 1,440).** In 62 it lies below the band (the range questions score toward the negative side of the contrast more than a random direction does) and in 1 above it (`qwen3-1.7b`, `triviaqa_known`, layer 8: 0.975 [0.964, 0.984] against [0.121, 0.864]). `qwen3-0.6b-base` has 16 of its 392 cells below the band, all in layers 15 to 28, and `gpt2-large` 13 of 540, all in layers 20 to 30; the others have 9, 7, 4, 5 and 5 (`qwen3-0.6b`, `qwen3-1.7b-base`, `qwen3-4b-base`, `gpt2-medium`, `gpt2-xl`), 3 (`qwen3-1.7b`) and none (`qwen3-4b`, `gpt2`). The cells below the band include very low values (0.016 [0.010, 0.023] for `qwen3-0.6b-base` `false_premise` at layer 18, against a band of [0.094, 0.934]). None of the 63 cells is at layer 1.
- **By family the counts are of the same size.** For the three families taken alone (all questions of the family) the interval lies below the band in 13 (duration), 24 (height) and 51 (wingspan) of 4,016 cells and above it in 7, 1 and 1.

The files show that the range direction read at layer 1 is not aligned with any P9 direction and that no P9 reader at that layer ranks the range questions beyond a random direction; they show a small number of later layers where a P9 reader ranks the range questions toward its negative side more than a random direction does. They do not show what a P9 reader responds to in these questions, and with the family confound above they do not separate a range-valued answer from a living subject.

Behavior link. For each test question of the P10b run that the P10a run also holds, the P10b score of the reader (average over the five draws at layer 1) is set against the hedge share of the question, the share of its answers that are ranges, averaged over the shot orders of the few-shot format (the same definition as P10a). The statistic is the Spearman correlation within a kind, with an item bootstrap percentile interval, and `n` is the number of questions of the group (72 range and 75 exact test questions pooled over the families, and the 120 people as their own group of the exact kind). The correlation across the two kinds is not reported: the reader separates the kinds and the hedging differs by kind, so it is high for a reason that is not a link (the module explains it). A group whose scores or hedge shares are all equal has no rank correlation; the table says "undefined (constant)" and the CSV flags it. The few-shot format is read for all ten checkpoints and the chat format for the three post-trained ones, in the free and the forced condition, greedy and sampled (the sampled share is over five samples per question). The per-family rows are in `behavior.csv`.

Within-kind Spearman correlation between the P10b score and the hedge share, test questions (hedge share is the mean over the group):

| Model           | Prompt           | Decoding | Range: hedge share | Range: Spearman [95% interval] | Exact: hedge share | Exact: Spearman [95% interval] | People: hedge share | People: Spearman [95% interval] |
| --------------- | ---------------- | -------- | ------------------ | ------------------------------ | ------------------ | ------------------------------ | ------------------- | ------------------------------- |
| qwen3-0.6b-base | free, few-shot   | greedy   | 0.986              | +0.102 [-0.014, +0.228]        | 0.053              | -0.040 [-0.182, +0.114]        | 0.000               | undefined (constant)            |
| qwen3-0.6b-base | free, few-shot   | sampled  | 0.656              | -0.067 [-0.330, +0.197]        | 0.101              | -0.313 [-0.509, -0.112]        | 0.108               | -0.128 [-0.306, +0.066]         |
| qwen3-0.6b-base | forced, few-shot | greedy   | 0.208              | +0.351 [+0.136, +0.524]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-0.6b-base | forced, few-shot | sampled  | 0.194              | +0.175 [-0.060, +0.394]        | 0.003              | +0.129 [+0.097, +0.247]        | 0.011               | -0.087 [-0.268, +0.082]         |
| qwen3-0.6b      | free, few-shot   | greedy   | 0.521              | +0.256 [+0.020, +0.502]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-0.6b      | free, few-shot   | sampled  | 0.347              | -0.218 [-0.435, +0.008]        | 0.017              | -0.159 [-0.400, +0.119]        | 0.017               | -0.132 [-0.297, +0.036]         |
| qwen3-0.6b      | free, chat       | greedy   | 0.111              | +0.045 [-0.150, +0.248]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-0.6b      | free, chat       | sampled  | 0.236              | -0.309 [-0.506, -0.080]        | 0.000              | undefined (constant)           | 0.010               | -0.087 [-0.169, -0.004]         |
| qwen3-0.6b      | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-0.6b      | forced, few-shot | sampled  | 0.001              | +0.168 [+0.146, +0.327]        | 0.000              | undefined (constant)           | 0.001               | -0.110 [-0.215, -0.091]         |
| qwen3-0.6b      | forced, chat     | greedy   | 0.028              | -0.073 [-0.188, +0.037]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-0.6b      | forced, chat     | sampled  | 0.114              | -0.412 [-0.574, -0.211]        | 0.000              | undefined (constant)           | 0.002               | -0.044 [-0.098, -0.020]         |
| qwen3-1.7b-base | free, few-shot   | greedy   | 0.833              | +0.421 [+0.208, +0.594]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b-base | free, few-shot   | sampled  | 0.610              | +0.597 [+0.422, +0.725]        | 0.028              | -0.273 [-0.440, -0.086]        | 0.043               | -0.128 [-0.294, +0.050]         |
| qwen3-1.7b-base | forced, few-shot | greedy   | 0.083              | +0.212 [+0.064, +0.363]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b-base | forced, few-shot | sampled  | 0.086              | +0.273 [+0.030, +0.460]        | 0.000              | undefined (constant)           | 0.003               | +0.079 [-0.044, +0.215]         |
| qwen3-1.7b      | free, few-shot   | greedy   | 0.312              | +0.195 [-0.016, +0.403]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | free, few-shot   | sampled  | 0.263              | +0.083 [-0.164, +0.314]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | free, chat       | greedy   | 0.222              | +0.143 [-0.029, +0.313]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | free, chat       | sampled  | 0.239              | +0.165 [-0.032, +0.373]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | forced, few-shot | sampled  | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | forced, chat     | greedy   | 0.014              | +0.060 [+0.016, +0.156]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-1.7b      | forced, chat     | sampled  | 0.014              | +0.009 [-0.098, +0.130]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b-base   | free, few-shot   | greedy   | 0.799              | +0.261 [-0.004, +0.474]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b-base   | free, few-shot   | sampled  | 0.633              | +0.191 [-0.050, +0.410]        | 0.009              | -0.085 [-0.273, +0.101]        | 0.016               | -0.124 [-0.280, +0.058]         |
| qwen3-4b-base   | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b-base   | forced, few-shot | sampled  | 0.087              | +0.002 [-0.224, +0.215]        | 0.001              | -0.064 [-0.149, -0.027]        | 0.002               | -0.103 [-0.245, +0.019]         |
| qwen3-4b        | free, few-shot   | greedy   | 0.472              | +0.176 [-0.057, +0.391]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | free, few-shot   | sampled  | 0.519              | +0.211 [-0.024, +0.441]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | free, chat       | greedy   | 0.486              | +0.427 [+0.210, +0.618]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | free, chat       | sampled  | 0.489              | +0.456 [+0.251, +0.632]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | forced, few-shot | sampled  | 0.003              | +0.065 [-0.098, +0.236]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | forced, chat     | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| qwen3-4b        | forced, chat     | sampled  | 0.017              | +0.001 [-0.093, +0.093]        | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| gpt2            | free, few-shot   | greedy   | 0.986              | +0.110 [-0.039, +0.276]        | 0.700              | -0.351 [-0.549, -0.108]        | 0.596               | +0.251 [+0.053, +0.416]         |
| gpt2            | free, few-shot   | sampled  | 0.331              | +0.003 [-0.228, +0.239]        | 0.247              | +0.063 [-0.175, +0.284]        | 0.193               | -0.009 [-0.197, +0.176]         |
| gpt2            | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| gpt2            | forced, few-shot | sampled  | 0.040              | -0.224 [-0.420, +0.003]        | 0.009              | +0.091 [-0.167, +0.346]        | 0.018               | +0.124 [-0.049, +0.286]         |
| gpt2-medium     | free, few-shot   | greedy   | 0.986              | +0.268 [+0.174, +0.430]        | 0.293              | +0.204 [-0.037, +0.422]        | 0.013               | +0.158 [+0.064, +0.282]         |
| gpt2-medium     | free, few-shot   | sampled  | 0.237              | -0.012 [-0.212, +0.206]        | 0.079              | +0.202 [-0.039, +0.427]        | 0.063               | -0.157 [-0.348, +0.042]         |
| gpt2-medium     | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| gpt2-medium     | forced, few-shot | sampled  | 0.039              | -0.087 [-0.309, +0.163]        | 0.012              | +0.154 [-0.068, +0.352]        | 0.020               | +0.022 [-0.146, +0.179]         |
| gpt2-large      | free, few-shot   | greedy   | 0.319              | -0.126 [-0.359, +0.115]        | 0.007              | +0.059 [+0.019, +0.148]        | 0.000               | undefined (constant)            |
| gpt2-large      | free, few-shot   | sampled  | 0.182              | -0.174 [-0.392, +0.064]        | 0.028              | +0.310 [+0.089, +0.507]        | 0.029               | -0.121 [-0.295, +0.050]         |
| gpt2-large      | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| gpt2-large      | forced, few-shot | sampled  | 0.043              | -0.054 [-0.285, +0.175]        | 0.005              | +0.178 [+0.014, +0.332]        | 0.014               | +0.061 [-0.101, +0.220]         |
| gpt2-xl         | free, few-shot   | greedy   | 0.611              | +0.020 [-0.220, +0.261]        | 0.007              | -0.059 [-0.144, -0.021]        | 0.000               | undefined (constant)            |
| gpt2-xl         | free, few-shot   | sampled  | 0.289              | +0.081 [-0.142, +0.308]        | 0.057              | -0.131 [-0.356, +0.106]        | 0.029               | -0.060 [-0.234, +0.115]         |
| gpt2-xl         | forced, few-shot | greedy   | 0.000              | undefined (constant)           | 0.000              | undefined (constant)           | 0.000               | undefined (constant)            |
| gpt2-xl         | forced, few-shot | sampled  | 0.049              | -0.144 [-0.385, +0.098]        | 0.007              | -0.012 [-0.239, +0.268]        | 0.013               | +0.109 [-0.010, +0.223]         |

- **The within-kind link is weak and changes sign.** In the free condition the correlation of the range questions is defined in all 26 cells of the table; its interval lies above 0 in 6 (`qwen3-0.6b` few-shot greedy, +0.256 [+0.020, +0.502]; `qwen3-1.7b-base` few-shot greedy and sampled, +0.421 and +0.597; `qwen3-4b` chat greedy and sampled, +0.427 and +0.456; `gpt2-medium` few-shot greedy, +0.268), below 0 in 1 (`qwen3-0.6b` chat sampled, -0.309 [-0.506, -0.080]) and contains 0 in the other 19. The same checkpoint can change sign with the decoding: `qwen3-0.6b` few-shot is +0.256 [+0.020, +0.502] greedy and -0.218 [-0.435, +0.008] sampled. In the forced condition, where the hedge share of the range questions is 0.208 or lower, the correlation is defined in 16 of 26 cells, with the interval above 0 in 5 and below 0 in 1 (`qwen3-0.6b` chat sampled, -0.412).
- **The exact questions give no consistent sign either.** In the free condition the correlation is defined in 13 of 26 cells, because the others have a hedge share of 0 on every exact question. Its interval lies above 0 in 2 (`gpt2-large`, +0.059 [+0.019, +0.148] greedy and +0.310 [+0.089, +0.507] sampled) and below 0 in 4 (`qwen3-0.6b-base` sampled -0.313, `qwen3-1.7b-base` sampled -0.273, `gpt2` greedy -0.351 [-0.549, -0.108], `gpt2-xl` greedy -0.059 [-0.144, -0.021]). In the forced condition it is defined in 6 of 26 cells.
- **The people heights, the living exact items, show no consistent link.** The correlation within the 120 people is defined in 11 of 26 free cells; its interval lies above 0 for `gpt2` (+0.251 [+0.053, +0.416]) and `gpt2-medium` (+0.158 [+0.064, +0.282]), both few-shot greedy, below 0 for `qwen3-0.6b` chat sampled (-0.087 [-0.169, -0.004]) and contains 0 in the other 8. The hedge share of the people is 0.000 in all nine Qwen3 greedy free cells, as that of the exact questions in eight of the nine (0.053 for `qwen3-0.6b-base`), which is the behavior of the exact kind and not of the breeds; for `gpt2` it is 0.596 against 0.700 for the exact questions and 0.986 for the range questions, and for `gpt2-medium` 0.013 against 0.293 and 0.986. In behavior the people therefore do not support "living thing" over "range-valued answer" for the Qwen3 checkpoints. For the readers the P10b diagnostic does not support a pure range reading (above).
- **`gpt2` and `gpt2-medium` hedge on exact questions in the free condition, the larger two almost never do, and none of the four hedges when the prompt asks for one number.** In the free condition the greedy hedge share of the exact questions is 0.700 (`gpt2`), 0.293 (`gpt2-medium`) and 0.007 (`gpt2-large`, `gpt2-xl`), against 0.986, 0.986, 0.319 and 0.611 on the range questions. In the forced condition the greedy hedge share is 0.000 for both kinds and the people in all four sizes, so the correlation is undefined, and the sampled share is 0.039 to 0.049 on the range questions, 0.005 to 0.012 on the exact ones and 0.013 to 0.020 on the people. The files show these numbers and do not show why a GPT-2 checkpoint answers with a range.
- **What the interval means.** The intervals are percentile bootstrap intervals of one run each, without a correction for the 52 rows and three groups of the table; a cell whose interval excludes 0 is not a finding by itself, and the table is not a list of links. There is no row in which the correlation of the range questions and that of the exact questions are both above 0 with an interval excluding 0.

### What P10 shows

On the Qwen3 checkpoints the greedy range share follows the sourced answer more than the animacy of the subject: an exact question gets a point (the greedy range share on exact questions is 0.000 in 13 of 15 cells), no Qwen3 checkpoint gives a range for a person's height in greedy decoding (0 of 120 in all 15 cells), the instruction to give one number brings the greedy range share on the range questions to 0.014 or less except for `qwen3-0.6b-base` (0.214 and 0.231) and `qwen3-1.7b-base` (0.081 and 0.067), and the forced number lies inside the sourced range more often for range questions than for exact ones (0.033 to 0.336 against 0.006 to 0.144, rising with size; in the 5 percent band the order reverses in three `qwen3-0.6b` cells). Other outputs do not follow the sourced answer alone: the chat approximation marker is more frequent on the people than on the buildings (67 of 120 against 9 for `qwen3-1.7b`, 49 against 41 for `qwen3-0.6b`, 15 against 4 for `qwen3-4b`), as is the sampled range share of the base checkpoints (people minus buildings above 0 in all six cells). The hedge share is not a measure of knowing that an answer is a range. It depends on size, on post-training, on the family (0.983 on the dog breeds for the 4B pair, 0.000 on gestation periods and wingspans for `qwen3-1.7b`) and on the order of the examples (the gap of `qwen3-0.6b` is 0.675 with a range last and 0.353 with an exact last), the chat format hedges with a marker that the range share does not count, and the comparison with the people mixes kind with familiarity. The GPT-2 sizes do not separate the kinds as the sourced answer does (`gpt2` gives a range for 0.978 of the exact questions with a range last and for 0.436 with an exact last), and their forced answers are inside the sourced range for at most 1 of 360 questions of either kind; the files do not say why.

The readers do not separate the kinds beyond the surface baselines and the random control in most cells. The logistic reader reads 1.000 in 49 of the 60 held-out cells on height, wingspan and the pooled questions, where `text_word` and `text_char` read 1.000 and the random control is 0.899 or higher on the pooled questions in 18 of 20 pairs. Held-out duration, where the baselines are lower (0.929 and 0.888), is the one setting where reader values exceed them: the concept-template logistic reader is above their value for nine of ten checkpoints (0.960 to 1.000), with an interval that overlaps that of `text_word`, and the random control is 0.900 or higher in seven of the ten concept cells. In the single-family table 5 of 120 cells, all fitted on duration, have a reader interval above all four baselines and the random control, without a correction for the number of cells. The people heights do not support a range reading for the readers fitted on all three families or on height and wingspan (logistic positions 0.762 to 0.954 for the first, and 0.844 to 0.991 in the concept template for the second, with the people above the test buildings), the pca_diff and mean_diff positions are inside the random band in 34 of 40 cases (of the 13 that are below 0.55, nearer the buildings, 11 are inside it and 2 are below it, both for `gpt2-medium` in the concept template), and for the four readers behind the 5 cells the positions are intermediate (0.319 to 0.529) and inside the band, so the diagnostic neither supports nor excludes a range reading there.

P10c links the range direction to the P9 knowledge readers and to the hedging, and its headline values are those of the layer P10b selected (layer 1 for all ten checkpoints). At that layer the cosine between the range direction and the 123 P9 directions is at most 0.132 in absolute value, 29 of the 123 lie outside the band of random vectors, and the AUROC of no P9 reader on the 720 P10 questions has an interval outside its random-direction band (a wide band: at layer 1 its lower bound is 0.030 to 0.150 and its upper bound 0.866 to 0.967). Over all layers the cosines stay at 0.211 or lower and the AUROC interval is outside its band in 63 of 4,016 cells, 62 of them below it and none at layer 1. The three families put the range side on living things and the P9 readers score no people, so the knowledge link does not separate a range-valued answer from a living thing. The within-kind correlation between the P10b score and the hedge share has an interval above 0 in 6 of the 26 free cells of the range questions and below 0 in 1, and for some checkpoints it changes sign with the decoding; for the exact questions it is above 0 in 2 of the 13 defined free cells and below 0 in 4. `gpt2` and `gpt2-medium` hedge on exact questions in the free condition (greedy hedge share 0.700 and 0.293), `gpt2-large` and `gpt2-xl` almost never do (0.007), and none of the four hedges when the prompt asks for one number (greedy 0.000 for both kinds in the forced condition); the files do not say why.

## Qwen3: emotions, TruthfulQA, seed agreement, logit lens, steering and control (R4, R6, A1, A2, M1, C2, C4, C5, LR1)

Runs on the base and post-trained checkpoints of Qwen3. R4, R6, A1, A2 and LR1 and the first M1, C2, C4 and C5 runs
were made at `a6e2333` with a clean tree on the laptop GPU (hashes under [Run identity](#run-identity)). The M1, C2, C4
and C5 runs reported below are later runs at `f4d414d` on a rented GPU pod, which replace the earlier ones as the
reported runs; the earlier runs stay tracked and each section names both. R4, A1 and A2 ran on all six
checkpoints (the 4B pair in chain 17, at `89ea7cb` on a rented GPU pod) and so did R6 (the `qwen3-4b-base` run of
chain 17 was cut when the chain was stopped and is not exported; the reported one is from chain 20, at `86c319b`).
LR1 and C4 ran on the 0.6B, 1.7B and 4B pairs (the 0.6B runs are from chain 43, at `bab5b75`, and the 4B C4 runs from
chain 21, at `d09822b`), C2 and C5 on the 1.7B and 4B pairs (the `qwen3-4b-base` C2 run and both 4B C5
runs are from chain 23, at `d09822b`), and M1 on the 1.7B and 4B pairs (the 4B M1 runs are from chain 22, at `d09822b`). The 4B results follow the 1.7B text
in a `#### The 4B pair` block of each section, and the 0.6B results of LR1 and C4 follow those in `#### The 0.6B pair` blocks. The post-trained checkpoints use the chat
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

The same questions, split, readers and layer rules (runs `r6/qwen3-4b-base/913301b5547b` and `r6/qwen3-4b/4e4a60094e47`;
15 readers for the LAT rows, 173 test questions, chance line 0.228). The post-trained run is from chain 17 and the base
run from chain 20, made later on another pod after the chain 17 attempt was cut (see [Run identity](#run-identity)). The
two text rows are those of every column above.

| Row                              | qwen3-4b-base            | qwen3-4b                 |
| -------------------------------- | ------------------------ | ------------------------ |
| standard                         | 0.254 [0.195, 0.324]     | 0.318 [0.253, 0.391]     |
| heuristic                        | 0.237 [0.180, 0.306]     | 0.225 [0.170, 0.293]     |
| text_word                        | 0.676 [0.603, 0.742]     | 0.676 [0.603, 0.742]     |
| text_char                        | 0.671 [0.597, 0.736]     | 0.671 [0.597, 0.736]     |
| LAT truthfulqa, validation-layer | 0.751 [0.746, 0.757]     | 0.741 [0.731, 0.750]     |
| LAT truthfulqa, best-layer       | 0.755 [0.748, 0.762] L22 | 0.745 [0.736, 0.754] L22 |
| LAT arc, validation-layer        | 0.434 [0.425, 0.444]     | 0.545 [0.537, 0.553]     |
| LAT arc, best-layer              | 0.436 [0.433, 0.439] L24 | 0.545 [0.536, 0.553] L23 |
| LAT honesty, validation-layer    | 0.511 [0.506, 0.515]     | 0.575 [0.565, 0.584]     |
| LAT honesty, best-layer          | 0.511 [0.506, 0.515] L22 | 0.589 [0.582, 0.595] L36 |

- **The TruthfulQA-source reader is the closest to the text baselines of any checkpoint, and is still not separated from
  them.** At the validation layer it reaches 0.751 [0.746, 0.757] on the base checkpoint and 0.741 [0.731, 0.750] on the
  post-trained one, above the point estimates of `text_word` 0.676 [0.603, 0.742] and `text_char` 0.671 [0.597, 0.736] by
  0.075 and 0.081 (base) and 0.064 and 0.070 (post-trained). These are the two largest margins in the R6 tables (the
  next is `qwen3-1.7b-base`, 0.026 and 0.032). The base interval lies above the upper bounds of both Wilson intervals
  (its lower bound 0.746 against 0.742 and 0.736); the post-trained value is inside the Wilson interval of `text_word`
  (upper bound 0.742) and 0.004 above the upper bound of `text_char` (0.736). The best-layer rows (0.755 and 0.745, both
  at L22) pick the layer on the test questions, are optimistic, and are above both upper bounds. The two kinds of
  interval are not comparable: the reader's interval is over 15 readers and does not carry the sampling of the 173
  questions (a width of about 0.14), so neither checkpoint is separated from the text baselines.
- **Base against post-trained.** The base checkpoint is above the post-trained one at the validation layer (0.751
  against 0.741) and at the best layer (0.755 against 0.745). The intervals over readers overlap ([0.746, 0.757] against
  [0.731, 0.750]), unlike the 0.6B and 1.7B pairs, where the base checkpoint is above with intervals that do not
  overlap. Those intervals do not include the sampling of the 173 questions, so this is a difference between two runs
  and not a result about base against post-trained models in general. The TruthfulQA reader rises with size on both
  kinds of checkpoint: 0.679, 0.703 and 0.751 on the base ones and 0.644, 0.672 and 0.741 on the post-trained ones.
- **The other two sources stay below the text baselines.** The ARC-trained reader is at 0.434 [0.425, 0.444] (base) and
  0.545 [0.537, 0.553] (post-trained) and the honesty-persona reader at 0.511 [0.506, 0.515] and 0.575 [0.565, 0.584] at
  the validation layer. Neither source contains TruthfulQA answers. The ARC reader is 0.237 and 0.126 below `text_char`
  and the honesty reader 0.160 and 0.096 below it. At 4B the post-trained checkpoint is above the base one for both
  sources, the opposite sign to the TruthfulQA source. For ARC the 0.6B pair has the base checkpoint above (0.406
  against 0.367) and the two 1.7B values are within 0.007 of each other (0.400 and 0.407); the honesty reader rises with
  size on the base checkpoints (0.334, 0.443, 0.511) and is highest at 4B on the post-trained ones too (0.346, 0.558,
  0.575).
- **Random and shuffled series.** The highest mean over layers of the random-direction series is 0.425, 0.387 and 0.413
  on the base checkpoint (TruthfulQA, ARC and honesty sources) and 0.480, 0.409 and 0.403 on the post-trained one, and
  the shuffled-label series stays at or below 0.237 and 0.243 (chance 0.228). The best-layer reader is above the highest
  random value in every row, by 0.330, 0.049 and 0.097 on the base checkpoint and by 0.265, 0.136 and 0.185 on the
  post-trained one. On the base checkpoint the ARC row (0.049) is as close to the random maximum as the ARC rows of the
  smaller base checkpoints (0.039 and 0.044) and the honesty row (0.097) is just above the 0.052 and 0.083 of the smaller
  base checkpoints; on the post-trained checkpoint both rows are clearly above it.
- **Standard and heuristic.** The standard log-probability MC1 on the test questions is 0.254 [0.195, 0.324] (base) and
  0.318 [0.253, 0.391] (post-trained) and the heuristic is 0.237 [0.180, 0.306] and 0.225 [0.170, 0.293]. On all 817
  questions the runs record 0.248 [0.220, 0.279] and 0.258 [0.229, 0.289] (base) and 0.332 [0.300, 0.365] and 0.237
  [0.210, 0.268] (post-trained). On the test questions the intervals of both base rows and of the post-trained
  heuristic contain the chance line. The standard row of the post-trained checkpoint is above it as a point estimate,
  with a lower bound of 0.253 (0.300 on all 817 questions). That of the base checkpoint is not separated from it (lower
  bound 0.195 on the test questions, 0.220 on all 817; the base heuristic on all 817 has a lower bound of 0.229, just
  above the line). The post-trained
  checkpoint is above the base one on the standard row, as at 0.6B and 1.7B.

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

#### The 4B pair (M1)

The same config as the 1.7B runs above (six finders, 10 readers per finder, the first 3 used for steering, 250 prompts, one sample per prompt, `fluency` on), on `qwen3-4b-base` and `qwen3-4b` (runs `m1/qwen3-4b-base/5b21f57abf27` and `m1/qwen3-4b/d490d3bf381b`, made in chain 22 at `d09822b` on a rented GPU pod; see [Run identity](#run-identity)). Both are steered at layers 14, 20 and 25 (of 36). There is one run per checkpoint, so the variation between runs that the 1.7B pair showed (above) is not measured here, and the Wilson intervals do not include it. Each win-rate cell has 748 to 750 (prompt, reader) pairs and the test accuracy is on 4,808 pairs, at the layer of each finder.

`qwen3-4b-base`:

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.703 [0.690, 0.717] | L22   | 0.735 [0.703, 0.766] | 0.836 [0.807, 0.860] | 0.911 [0.888, 0.929] | 0.901 [0.878, 0.921] |
| pca_raw   | 0.733 [0.718, 0.750] | L22   | 0.700 [0.666, 0.731] | 0.837 [0.809, 0.862] | 0.925 [0.904, 0.942] | 0.897 [0.874, 0.917] |
| mean_diff | 0.787 [0.773, 0.798] | L21   | 0.716 [0.683, 0.747] | 0.861 [0.835, 0.884] | 0.949 [0.931, 0.963] | 0.912 [0.890, 0.930] |
| logistic  | 0.772 [0.755, 0.784] | L22   | 0.671 [0.637, 0.704] | 0.792 [0.761, 0.819] | 0.869 [0.843, 0.892] | 0.839 [0.811, 0.863] |
| kmeans    | 0.741 [0.727, 0.756] | L22   | 0.731 [0.698, 0.762] | 0.869 [0.843, 0.892] | 0.937 [0.918, 0.953] | 0.924 [0.903, 0.941] |
| random    | 0.618 [0.582, 0.653] | L32   | 0.493 [0.458, 0.529] | 0.528 [0.492, 0.563] | 0.529 [0.493, 0.564] | 0.525 [0.490, 0.561] |

`qwen3-4b`:

| Finder    | Test accuracy        | Layer | Win rate 0.1         | Win rate 0.2         | Win rate 0.3         | Check judge 0.3      |
| --------- | -------------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- |
| pca_diff  | 0.804 [0.802, 0.806] | L24   | 0.811 [0.781, 0.837] | 0.916 [0.894, 0.934] | 0.951 [0.933, 0.964] | 0.905 [0.882, 0.924] |
| pca_raw   | 0.806 [0.805, 0.808] | L21   | 0.807 [0.777, 0.833] | 0.912 [0.890, 0.930] | 0.965 [0.950, 0.976] | 0.928 [0.907, 0.944] |
| mean_diff | 0.807 [0.806, 0.808] | L23   | 0.817 [0.788, 0.843] | 0.916 [0.894, 0.934] | 0.969 [0.954, 0.979] | 0.935 [0.915, 0.950] |
| logistic  | 0.794 [0.786, 0.800] | L21   | 0.750 [0.718, 0.780] | 0.796 [0.766, 0.823] | 0.855 [0.828, 0.878] | 0.831 [0.802, 0.856] |
| kmeans    | 0.807 [0.806, 0.808] | L21   | 0.787 [0.756, 0.814] | 0.917 [0.895, 0.935] | 0.960 [0.943, 0.972] | 0.923 [0.901, 0.940] |
| random    | 0.669 [0.621, 0.713] | L23   | 0.543 [0.508, 0.579] | 0.511 [0.476, 0.547] | 0.460 [0.425, 0.496] | 0.457 [0.422, 0.493] |

- **Reading does not order the finders, and `logistic` steers least.** The test accuracy of the five finders spans 0.703 to 0.787 on the base checkpoint (`mean_diff` highest, `pca_diff` lowest) and 0.794 to 0.807 on the post-trained one (`logistic` lowest). The win rate at 0.3 is highest for `mean_diff` on both checkpoints (0.949 [0.931, 0.963] and 0.969 [0.954, 0.979]) and lowest for `logistic` (0.869 [0.843, 0.892] and 0.855 [0.828, 0.878]). On the base checkpoint `logistic` has the second-highest test accuracy (0.772), so accuracy does not predict its win rate there. The interval of `mean_diff` at 0.3 overlaps those of `pca_raw` and `kmeans` on the base checkpoint and those of `pca_diff`, `pca_raw` and `kmeans` on the post-trained one; it is clear of `pca_diff` (0.911 [0.888, 0.929]) and `logistic` on the base checkpoint and of `logistic` on the post-trained one. The check judge puts `logistic` lowest as well (0.839 [0.811, 0.863] and 0.831 [0.802, 0.856]), but well above the random direction (0.525 and 0.457), unlike the 1.7B post-trained checkpoint, where it was 0.580 [0.544, 0.615] against 0.513 [0.478, 0.549]. Of the four 1.7B and 4B checkpoints, `logistic` has the lowest win rate at 0.3 on three (on the 1.7B base checkpoint `kmeans` and `pca_raw` are lower).
- **The random direction.** Its test accuracy is 0.618 [0.582, 0.653] and 0.669 [0.621, 0.713], 0.085 to 0.169 below the five finders on the base checkpoint and 0.125 to 0.138 below on the post-trained one. On the base checkpoint its win rate includes 0.5 at 0.1, 0.2 and 0.3 on the primary judge (0.493, 0.528, 0.529) and at 0.3 on the check judge (0.525 [0.490, 0.561]). On the post-trained checkpoint it is above 0.5 at 0.1 (0.543 [0.508, 0.579]), includes 0.5 at 0.2 (0.511 [0.476, 0.547]) and is below 0.5 at 0.3 on both judges (0.460 [0.425, 0.496] and 0.457 [0.422, 0.493]). Those two excursions are small (0.04 from 0.5), have opposite signs and come from one run; in the 1.7B pair the earlier run showed excursions of 0.06 to 0.07 that did not reappear in the later run (above), so they are not read as an effect of the random direction. At 0.3 the finders are above the random win rate by 0.341 (`logistic`) to 0.421 (`mean_diff`) on the base checkpoint and by 0.395 (`logistic`) to 0.509 (`mean_diff`) on the post-trained one.
- **Judges and empty completions.** The share of swept settings with the same dose sign under both judges is 0.944 on both checkpoints (0.889 and 0.833 on the 1.7B pair). The share of empty completions over all generations is 0 on both.
- **Against the 1.7B pair.** At the same relative strengths the finders win more at 0.3 on the 4B pair than on the 1.7B pair (base: 0.869 to 0.949 against 0.752 to 0.896; post-trained: 0.855 to 0.969 against 0.743 to 0.921; on the post-trained checkpoint the `pca_diff` intervals overlap, 0.951 [0.933, 0.964] against 0.920 [0.898, 0.937], and the intervals of the other four finders do not). On the base checkpoint the five finders are spread wider at 4B (test accuracy 0.703 to 0.787 against 0.767 to 0.776), with `pca_diff`, `pca_raw` and `kmeans` below the 1.7B base range with non-overlapping intervals. These are one run per checkpoint of each size, so the differences are not tested against variation between runs.

Cost to the text. Unsteered, the neutral text has a perplexity of 15.9 on the base checkpoint and 31.7 on the post-trained one, and the generated text a distinct-1 of 0.217 and 0.175. The tables give the mean over the three steering readers, with the ratio to the unsteered perplexity in parentheses at the largest strength.

`qwen3-4b-base`:

| Finder    | Perplexity +0.1 | Perplexity +0.2 | Perplexity +0.3 | Distinct-1 +0.3 | Perplexity -0.3 | Distinct-1 -0.3 |
| --------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| pca_diff  | 16.4            | 18.3            | 23.1 (1.46x)    | 0.173           | 24.7 (1.55x)    | 0.196           |
| pca_raw   | 16.2            | 18.1            | 22.8 (1.44x)    | 0.174           | 26.2 (1.65x)    | 0.188           |
| mean_diff | 16.7            | 19.0            | 24.3 (1.53x)    | 0.175           | 24.7 (1.55x)    | 0.185           |
| logistic  | 16.8            | 19.5            | 26.7 (1.68x)    | 0.185           | 24.0 (1.51x)    | 0.201           |
| kmeans    | 16.4            | 18.5            | 23.6 (1.49x)    | 0.169           | 25.4 (1.60x)    | 0.180           |
| random    | 16.3            | 17.7            | 21.1 (1.33x)    | 0.212           | 21.1 (1.33x)    | 0.202           |

`qwen3-4b`:

| Finder    | Perplexity +0.1 | Perplexity +0.2 | Perplexity +0.3 | Distinct-1 +0.3 | Perplexity -0.3 | Distinct-1 -0.3 |
| --------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| pca_diff  | 37.7            | 48.2            | 65.3 (2.06x)    | 0.157           | 37.7 (1.19x)    | 0.112           |
| pca_raw   | 35.5            | 41.3            | 50.3 (1.59x)    | 0.163           | 40.8 (1.29x)    | 0.097           |
| mean_diff | 37.8            | 47.4            | 61.6 (1.95x)    | 0.154           | 37.3 (1.18x)    | 0.109           |
| logistic  | 39.6            | 53.6            | 78.3 (2.47x)    | 0.145           | 34.9 (1.10x)    | 0.144           |
| kmeans    | 35.7            | 42.0            | 51.5 (1.63x)    | 0.159           | 40.0 (1.26x)    | 0.101           |
| random    | 31.8            | 33.4            | 37.5 (1.18x)    | 0.155           | 41.2 (1.30x)    | 0.155           |

- **The largest strength costs the text on both checkpoints, in different ways.** At +0.3 the perplexity of the finders is 1.44 to 1.68 times the unsteered one on the base checkpoint and 1.59 to 2.47 times on the post-trained one (the random direction: 1.33 and 1.18), and the distinct-1 is 0.169 to 0.185 and 0.145 to 0.163. At -0.3 the ratio is 1.51 to 1.65 on the base checkpoint and 1.10 to 1.29 on the post-trained one, but the post-trained distinct-1 falls to 0.097 to 0.144 (random: 0.155). By the rule of [Sentiment steering on Qwen3](#sentiment-steering-on-qwen3-c1) (mean perplexity at most 1.5 times the unsteered one and mean distinct-1 at least 0.15), on the base checkpoint `pca_diff` (1.46 times, distinct-1 0.173), `pca_raw` (1.44 times, 0.174) and `kmeans` (1.49 times, 0.169) pass at +0.3 and `mean_diff` (1.53 times) and `logistic` (1.68 times) do not, and all five finders fail at -0.3; on the post-trained checkpoint no finder passes at +0.3 (ratios past 1.5) or at -0.3 (distinct-1 below 0.15). The random direction passes at every setting on both checkpoints. The win rates at 0.3 (0.869 to 0.949 and 0.855 to 0.969) are therefore scores of text that fails the rule in at least one of the two directions, and are not read as steering that works.
- **Which strengths pass.** On the base checkpoint 0.1 and 0.2 pass in both directions for every finder (ratios at most 1.06 and 1.23, distinct-1 of at least 0.192); among the passing settings at 0.2 `kmeans` (0.869 [0.843, 0.892]) and `mean_diff` (0.861 [0.835, 0.884]) win most, with overlapping intervals, and at 0.1 the finders win 0.671 (`logistic`) to 0.735 (`pca_diff`), against 0.493 for the random direction. On the post-trained checkpoint 0.1 passes for every finder (ratios 0.89 to 1.25 over both directions, distinct-1 of at least 0.160) and 0.2 passes for none: `pca_diff` (1.52 times) and `logistic` (1.69 times) fail at +0.2, and at -0.2 the distinct-1 of the other four finders is 0.132 to 0.143 (`logistic` passes there with 0.162). At 0.1 the finders win 0.750 (`logistic`) to 0.817 (`mean_diff`), against 0.543 for the random direction. The boundary is the same as in the 1.7B pair (0.2 on the base checkpoint, 0.1 on the post-trained one).

Erasure. As above, the tables give the probe accuracy before and after projecting a finder's direction out of the activations of the control layers, and the behavioral gap (100 pairs) for the primary and the check judge. The `none` row is the gap without a projection.

`qwen3-4b-base`:

| Finder               | Probe before         | Probe after          | Gap primary           | Gap check             |
| -------------------- | -------------------- | -------------------- | --------------------- | --------------------- |
| pca_diff             | 0.737 [0.621, 0.794] | 0.673 [0.592, 0.723] | 0.077 [0.000, 0.208]  | 0.061 [0.052, 0.074]  |
| pca_raw              | 0.737 [0.621, 0.794] | 0.676 [0.618, 0.718] | 0.091 [0.063, 0.118]  | 0.057 [0.012, 0.119]  |
| mean_diff            | 0.737 [0.621, 0.794] | 0.500 [0.500, 0.500] | 0.062 [0.038, 0.107]  | 0.102 [0.077, 0.132]  |
| logistic             | 0.737 [0.621, 0.794] | 0.713 [0.613, 0.798] | 0.089 [0.051, 0.150]  | 0.100 [0.026, 0.168]  |
| kmeans               | 0.737 [0.621, 0.794] | 0.671 [0.630, 0.707] | 0.037 [-0.004, 0.090] | 0.037 [0.004, 0.089]  |
| random               | 0.737 [0.621, 0.794] | 0.737 [0.621, 0.795] | 0.065 [0.050, 0.078]  | 0.060 [0.032, 0.091]  |
| none (no projection) |                      |                      | 0.025 [-0.102, 0.165] | 0.043 [-0.071, 0.162] |

`qwen3-4b`:

| Finder               | Probe before         | Probe after          | Gap primary           | Gap check             |
| -------------------- | -------------------- | -------------------- | --------------------- | --------------------- |
| pca_diff             | 0.757 [0.663, 0.816] | 0.642 [0.530, 0.710] | 0.165 [0.141, 0.200]  | 0.123 [0.116, 0.126]  |
| pca_raw              | 0.757 [0.663, 0.816] | 0.634 [0.551, 0.697] | 0.250 [0.203, 0.291]  | 0.195 [0.140, 0.251]  |
| mean_diff            | 0.757 [0.663, 0.816] | 0.500 [0.500, 0.500] | 0.176 [0.156, 0.199]  | 0.137 [0.115, 0.158]  |
| logistic             | 0.757 [0.663, 0.816] | 0.749 [0.644, 0.808] | 0.178 [0.127, 0.212]  | 0.109 [0.077, 0.141]  |
| kmeans               | 0.757 [0.663, 0.816] | 0.643 [0.585, 0.699] | 0.201 [0.155, 0.254]  | 0.167 [0.144, 0.190]  |
| random               | 0.757 [0.663, 0.816] | 0.757 [0.663, 0.817] | 0.089 [0.034, 0.130]  | 0.089 [-0.017, 0.147] |
| none (no projection) |                      |                      | 0.087 [-0.032, 0.207] | 0.101 [-0.019, 0.211] |

- **The probe is erased for one finder.** The probe accuracy is 0.737 [0.621, 0.794] before the projection on the base checkpoint and 0.757 [0.663, 0.816] on the post-trained one. After `mean_diff` it is 0.500 on both, the only interval clear of the one before. After `logistic` it is 0.713 and 0.749, and after `pca_diff`, `pca_raw` and `kmeans` it is 0.671 to 0.676 and 0.634 to 0.643, falls of 0.062 to 0.066 and 0.114 to 0.122 whose intervals overlap the ones before. After the random direction it is unchanged. As on the 1.7B pair, `logistic` leaves the probe close to its value before.
- **The behavioral gap does not fall.** On the base checkpoint it is 0.025 [-0.102, 0.165] without a projection and 0.037 to 0.091 under the projection of a finder (0.065 for the random direction), every interval overlapping the unprojected one; on the check judge it is 0.043 without a projection and 0.037 to 0.102 under one. On the post-trained checkpoint it is 0.087 [-0.032, 0.207] without a projection and 0.165 to 0.250 under the projection of a finder (0.089 for the random direction), every interval overlapping the unprojected one, and the intervals of four finders (`pca_diff`, `pca_raw`, `mean_diff` and `kmeans`) do not overlap that of the random direction ([0.034, 0.130]); on the check judge it is 0.101 without a projection and 0.109 to 0.195 under one. The data show no fall of the gap under a projection on either checkpoint, and on the post-trained one the point values are above the unprojected one. As on the 1.7B pair, the two operationalizations of erasure disagree: the probe is erased for one finder and the behavioral gap for none.

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

#### The 4B pair (C2)

The same config as the 1.7B runs above (100 prompts, 2 samples per prompt, 3 steering seeds, relative strength +0.3, three
random arms), on `qwen3-4b-base` and `qwen3-4b` (36 blocks; runs `c2/qwen3-4b-base/984485501f3e` and
`c2/qwen3-4b/1df7c1f9ff70`). The post-trained run was made in chain 17 at `89ea7cb` and the base run in chain 23 at
`d09822b`, after the chain 17 attempt was cut (see [Run identity](#run-identity)). Both read at L25 and steer over
layers 24, 25 and 26, so, unlike the 1.7B pair (L20 and L18 of 28), the two checkpoints are steered at the same depth;
it is not the depth of the 1.7B runs.

`qwen3-4b-base`:

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.105         | 0.456 [0.422, 0.483]  | +0.351          | 0.280       | 0.743       | +0.462        | sadness -0.059                 | sadness -0.091               | -0.016 to +0.016        |
| sadness   | 0.079         | 0.179 [0.137, 0.212]  | +0.100          | 0.117       | 0.165       | +0.048        | happiness +0.040               | happiness +0.146             | -0.020 to +0.031        |
| anger     | 0.055         | 0.146 [0.133, 0.159]  | +0.091          | 0.065       | 0.191       | +0.126        | happiness -0.071               | happiness -0.140             | -0.017 to +0.007        |
| fear      | 0.075         | 0.259 [0.243, 0.285]  | +0.184          | 0.088       | 0.169       | +0.081        | happiness -0.075               | happiness -0.130             | -0.011 to +0.006        |
| disgust   | 0.048         | 0.187 [0.136, 0.213]  | +0.140          | 0.085       | 0.129       | +0.044        | happiness -0.070               | happiness -0.129             | -0.008 to +0.014        |
| surprise  | 0.097         | 0.219 [0.185, 0.251]  | +0.122          | 0.062       | 0.105       | +0.043        | happiness -0.061               | sadness -0.065               | -0.008 to +0.001        |

`qwen3-4b`:

| Emotion   | Primary: none | Primary: +0.3 [range] | Primary: change | Check: none | Check: +0.3 | Check: change | Largest off-diagonal (primary) | Largest off-diagonal (check) | Random arms: own change |
| --------- | ------------- | --------------------- | --------------- | ----------- | ----------- | ------------- | ------------------------------ | ---------------------------- | ----------------------- |
| happiness | 0.090         | 0.249 [0.202, 0.294]  | +0.159          | 0.241       | 0.540       | +0.299        | surprise +0.077                | sadness -0.097               | -0.020 to -0.002        |
| sadness   | 0.073         | 0.206 [0.123, 0.268]  | +0.133          | 0.129       | 0.259       | +0.129        | fear +0.039                    | disgust -0.061               | -0.023 to +0.020        |
| anger     | 0.045         | 0.129 [0.098, 0.153]  | +0.084          | 0.066       | 0.143       | +0.076        | happiness -0.050               | happiness -0.122             | -0.014 to -0.005        |
| fear      | 0.072         | 0.249 [0.222, 0.302]  | +0.178          | 0.095       | 0.189       | +0.095        | happiness -0.054               | happiness -0.118             | -0.014 to +0.010        |
| disgust   | 0.051         | 0.074 [0.067, 0.080]  | +0.023          | 0.102       | 0.114       | +0.012        | happiness -0.055               | happiness -0.122             | -0.021 to +0.000        |
| surprise  | 0.099         | 0.348 [0.293, 0.403]  | +0.248          | 0.068       | 0.141       | +0.072        | sadness -0.064                 | happiness +0.264             | +0.003 to +0.018        |

- **All six emotions rise toward their own score on the primary judge on both checkpoints**, by +0.091 (anger) to +0.351
  (happiness) on the base checkpoint and by +0.023 (disgust) to +0.248 (surprise) on the post-trained one, with the range
  over seeds above the unsteered value in every row. On the base checkpoint happiness (+0.351), fear (+0.184), disgust
  (+0.140), surprise (+0.122) and sadness (+0.100) rise most, then anger (+0.091). On the post-trained one surprise
  (+0.248), fear (+0.178), happiness (+0.159) and sadness (+0.133) rise most, then anger (+0.084) and disgust (+0.023).
  The primary hit rate is 1.000 on both and the specificity is 0.195 (base) and 0.159 (post-trained). On the check judge
  the hit rate is 0.833 on the base checkpoint (it misses sadness, where happiness is the largest entry, +0.146 against
  +0.048 on the diagonal) and 0.667 on the post-trained one (it misses disgust, where anger is the largest entry, and
  surprise, where happiness is); the specificity is 0.168 and 0.151. The correlation between the judges over the change
  matrix is 0.877 (base), in the range of the 1.7B values (0.833 and 0.891), and 0.640 (post-trained), below it; on the
  post-trained checkpoint the diagonals differ: the check judge gives surprise +0.072 where the primary judge gives
  +0.248, and happiness +0.299 where the primary gives +0.159.
- **Random arms.** The own-emotion change under a random arm is between -0.020 and +0.031 on the base checkpoint and
  between -0.023 and +0.020 on the post-trained one, and the largest absolute change of any emotion under any random arm,
  coefficient and judge is 0.091 and 0.052 (the largest of the C2 runs on Qwen3; the 1.7B values are 0.046 and 0.076). On the base checkpoint happiness (+0.351), fear (+0.184), disgust (+0.140) and surprise (+0.122) are above the
  random maximum; sadness (+0.100) is above it by 0.009 and anger (+0.091, that is 0.0912 against 0.0908) is level with
  it, so those two are not separated from an arbitrary edit of the same size. On the post-trained checkpoint five
  own-emotion changes (+0.084 to +0.248) are above it and disgust (+0.023) is not.
- **Base against post-trained.** Happiness (+0.351 against +0.159) and disgust (+0.140 against +0.023) are steered more on
  the base checkpoint, surprise (+0.122 against +0.248) more on the post-trained one,
  sadness (+0.100 against +0.133) is not separated, the ranges over seeds of the change overlapping ([+0.058, +0.133]
  and [+0.050, +0.195]), and anger (+0.091 against +0.084) and fear (+0.184 against +0.178) are within 0.007 of each other. There is no common
  sign. Each emotion has one reader per checkpoint and each checkpoint one run, so this is a contrast between two runs.
- **Off-diagonal.** On the base checkpoint the largest off-diagonal change on the primary judge is a fall of happiness in
  the anger, fear, disgust and surprise rows (-0.061 to -0.075), a rise of happiness in the sadness row (+0.040) and a
  fall of sadness in the happiness row (-0.059). On the check judge happiness falls in the anger, fear and disgust rows
  (-0.129 to -0.140), rises by +0.146 in the sadness row (more than the diagonal, +0.048), and sadness falls in the
  happiness and surprise rows (-0.091 and -0.065). On the post-trained checkpoint, on the primary judge the largest
  off-diagonal change is a fall of happiness in the anger, fear and disgust rows (-0.050 to -0.055), a fall of sadness in
  the surprise row (-0.064), a rise of fear in the sadness row (+0.039) and a rise of surprise in the happiness row
  (+0.077). On the check judge happiness falls in the same three rows (-0.118 to -0.122) and rises by +0.264 in the
  surprise row, more than the diagonal (+0.072).

Cost to the text. The tables give the perplexity of held-out neutral text under the steered model and the distinct-1 of
the generated text at +0.3, as the mean over the three steering readers of each emotion (with the range over the
readers), against the unsteered text and the mean over the three random arms (with the range over the arms).

`qwen3-4b-base`:

| Arm                   | Perplexity [range over arms or readers] | Ratio to none | Distinct-1 [range]   |
| --------------------- | --------------------------------------- | ------------- | -------------------- |
| none                  | 15.9                                    | 1.00          | 0.319                |
| random (3 arms, +0.3) | 21.3 [20.8, 21.9]                       | 1.34          | 0.281 [0.276, 0.286] |
| happiness (+0.3)      | 40.6 [38.4, 42.4]                       | 2.56          | 0.272 [0.261, 0.280] |
| sadness (+0.3)        | 60.2 [57.3, 62.8]                       | 3.79          | 0.226 [0.210, 0.248] |
| anger (+0.3)          | 65.4 [64.5, 66.8]                       | 4.12          | 0.254 [0.249, 0.256] |
| fear (+0.3)           | 68.8 [64.4, 74.1]                       | 4.33          | 0.211 [0.207, 0.216] |
| disgust (+0.3)        | 75.5 [73.0, 77.2]                       | 4.75          | 0.210 [0.197, 0.217] |
| surprise (+0.3)       | 57.8 [53.6, 61.3]                       | 3.64          | 0.204 [0.186, 0.213] |

`qwen3-4b`:

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

- **+0.3 is not fluent text.** Averaged over the six emotions the perplexity under the emotion arms is 1.13, 1.74 and
  3.86 times the unsteered one at +0.1, +0.2 and +0.3 on the base checkpoint and 1.14, 1.69 and 3.29 times on the
  post-trained one, against 1.03, 1.12 and 1.34 and 1.02, 1.10 and 1.31 for the random arms. By the rule of [Sentiment
  steering on Qwen3](#sentiment-steering-on-qwen3-c1) (mean perplexity at most 1.5 times the unsteered one and mean
  distinct-1 at least 0.15) +0.1 passes on both checkpoints (mean distinct-1 0.305 and 0.239 against 0.319 and 0.249
  unsteered), as does -0.1 (1.11 and 1.14 times, distinct-1 0.302 and 0.241), and +0.2 does not (1.74 and 1.69 times the unsteered perplexity, distinct-1 0.274 and
  0.200). At +0.3 every emotion is at 2.56 to 4.75 times the unsteered perplexity on the base checkpoint (distinct-1 from
  0.204 for surprise to 0.272 for happiness) and at 2.36 to 4.78 times on the post-trained one (distinct-1 from 0.115 for
  surprise to 0.173 for sadness; the mean over the six emotions is 0.147, below the cutoff, and three emotions are below
  0.15). The changes in the first tables are therefore changes of text that fails the rule at +0.3, and a rise in an
  emotion score at +0.3 is not read as steering that works. The own-emotion probability at +0.1 is in the curves file of
  each run and is not tabulated here.

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
smallest at L11, the largest at L21) and 0.187 to 0.259 on the post-trained one (the largest at L13), against 0.190 to 0.241
(base) and 0.178 to 0.258 (post-trained) at 1.7B. LR1 has no honesty evaluation; C4 scores these adapters on TruthfulQA below.

#### The 4B pair (C4)

C4 on the 4B pair (runs `c4/qwen3-4b-base/dd918686b022` and `c4/qwen3-4b/a079b7a25b4f`, made at `d09822b` on a rented
GPU pod, see [Run identity](#run-identity)), which read the LR1 adapter of their checkpoint (the two LR1 runs above). The
config is that of the 1.7B runs (10 readers, 3 random arms, 173 test questions, 644 train and validation questions,
chance line 0.228), with batch size 16, and there is no earlier 4B C4 run. The unedited row of the base checkpoint
(0.249, 43 of 173 questions) is one question below the standard row of R6 (0.254, 44), and that of the post-trained
checkpoint (0.324, 56) is one above it (0.318, 55). The perplexity column is that of held-out neutral text under the
edited model; the prompting and `contrast` arms have no steered model, and the readers with coefficient 0 are the
unedited model.

`qwen3-4b-base`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.249 [0.190, 0.318]       | 0                       | 6.37               |
| prompting +1         | 0.243 [0.185, 0.312]       | -0.006 [-0.035, +0.023] | -                  |
| prompting -1         | 0.197 [0.144, 0.262]       | -0.052 [-0.104, -0.006] | -                  |
| difference (0.2)     | 0.266 [0.206, 0.336]       | +0.017 [-0.058, +0.092] | 20.05              |
| contrast (0.5)       | 0.382 [0.312, 0.456]       | +0.133 [+0.075, +0.197] | -                  |
| LoRRA                | 0.301 [0.237, 0.373]       | +0.052 [+0.000, +0.110] | 6.61               |
| reading (10 readers) | mean 0.244, 0.220 to 0.249 | -0.029 to +0.000        | 6.37 to 14.35      |
| random (3 arms)      | mean 0.250, 0.243 to 0.254 | -0.006 to +0.006        | 6.73 to 8.81       |

`qwen3-4b`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.324 [0.258, 0.397]       | 0                       | 7.96               |
| prompting +1         | 0.295 [0.232, 0.367]       | -0.029 [-0.069, +0.006] | -                  |
| prompting -1         | 0.237 [0.180, 0.306]       | -0.087 [-0.145, -0.029] | -                  |
| difference (-0.05)   | 0.277 [0.216, 0.348]       | -0.046 [-0.087, -0.012] | 8.99               |
| contrast (0.5)       | 0.376 [0.307, 0.450]       | +0.052 [-0.012, +0.110] | -                  |
| LoRRA                | 0.329 [0.264, 0.403]       | +0.006 [-0.046, +0.058] | 9.14               |
| reading (10 readers) | mean 0.324, 0.295 to 0.347 | -0.029 to +0.023        | 8.23 to 9.81       |
| random (3 arms)      | mean 0.316, 0.306 to 0.329 | -0.017 to +0.006        | 8.01 to 8.39       |

- **Base checkpoint.** The unedited row (0.249 [0.190, 0.318]) has an interval that contains the chance line. The
  untruthful prompt lowers MC1 by 0.052 [-0.104, -0.006] (9 questions of 173) and the truthful prompt changes it by -0.006
  [-0.035, +0.023], so the prompt that asks for truth does not raise it, unlike at 1.7B. `contrast` gives +0.133 [+0.075,
  +0.197] (23 more questions) and reaches 0.382 [0.312, 0.456], an interval above the chance line. The adapter gives
  +0.052 [+0.000, +0.110] (9 more questions; the lower bound is zero) and 0.301 [0.237, 0.373]. `difference` (+0.017) has
  an interval that contains zero. Seven of the ten readers chose coefficient 0 (the unedited model, so their change is 0
  by construction) and three chose +0.2, with changes of -2, -1 and -5 questions; the ten readers give -0.029 to +0.000.
  The three random arms give -0.006 to +0.006, each with an interval that contains zero. `contrast` and the adapter are
  above every random arm (0.382 and 0.301 against at most 0.254), and only `contrast` has a change whose interval
  excludes zero on the positive side, but the test MC1 intervals of both overlap those of the random arms (the highest
  random upper bound is 0.324).
- **Post-trained checkpoint.** The unedited row (0.324 [0.258, 0.397]) has a lower bound above the chance line. The
  truthful prompt gives -0.029 [-0.069, +0.006] and the untruthful one -0.087 [-0.145, -0.029] (15 questions fewer), so
  only the second excludes zero and no prompt raises MC1. `difference` at -0.05 gives -0.046 [-0.087, -0.012] (8
  questions fewer), an interval below zero. `contrast` gives +0.052 [-0.012, +0.110] (9 more), the adapter +0.006 [-0.046,
  +0.058] (1 more), the ten readers -0.029 to +0.023 and the three random arms -0.017 to +0.006, all with intervals that
  contain zero.
- **What is and is not supported.** The untruthful prompt lowers MC1 on both checkpoints (the intervals exclude zero)
  and `difference` lowers it on the post-trained one; `contrast` raises it on the base one. No arm raises MC1 on the
  post-trained checkpoint, and the adapter's interval on the base checkpoint reaches zero. This is among the arms of the
  tables and with no correction for multiple comparisons. The `contrast` coefficient (0.5) is the largest of its grid on
  both checkpoints, and the `difference` coefficient of the base checkpoint (+0.2) and that of its three nonzero readers
  (+0.2) are the largest of their grid, so the selection was at an edge. The post-trained readers chose +0.1 (seven) and
  +0.05 (three); the random arms chose -0.1, -0.1 and +0.2 on the base checkpoint and -0.1, +0.05 and -0.1 on the
  post-trained one. The `contrast` gain on the base checkpoint (+0.133) is larger than the change of any random arm (at
  most +0.006); it is one run on one set of 173 questions.
- **Cost to the text.** The adapter leaves the perplexity of neutral text near the unedited one on the base checkpoint,
  6.61 against 6.37 (1.04 times), and raises it on the post-trained one, 9.14 against 7.96 (1.15 times; 1.09 times at
  1.7B). On the base checkpoint the settings chosen by selection are not all benign: `difference` at +0.2 has a
  perplexity of 20.05 (3.15 times), the three readers with coefficient +0.2 have 12.95 to 14.35 (2.03 to 2.25 times),
  and the random arms 6.73 to 8.81 (1.06 to 1.38 times). On the post-trained checkpoint the ten readers are at 8.23 to
  9.81 (1.03 to 1.23 times), `difference` at -0.05 at 8.99 (1.13 times) and the random arms at 8.01 to 8.39 (1.01 to 1.05
  times). The scoring is not generation, so the distinct-1 and the completion fluency of the text are not measured.

#### The 0.6B pair (LR1)

LR1 on the 0.6B pair (runs `lr1/qwen3-0.6b-base/03ba155a9fe7` and `lr1/qwen3-0.6b/74d01330d28c`, made at `bab5b75` on a
rented GPU pod: layers 9, 11, 13, 15 and 17, rank 8, learning rate 3e-4, 70 steps, batch size 16, 696,320 trainable
parameters, 2,000 training examples and 64 held-out, with the same control prompts). The training loss goes from 43.3 to
34.4 on the base checkpoint and from 65.0 to 48.9 on the post-trained one. The held-out loss of each layer, before and
after, and its reduction:

| Layer | qwen3-0.6b-base: before to after | Reduction | qwen3-0.6b: before to after | Reduction |
| ----- | -------------------------------- | --------- | --------------------------- | --------- |
| L9    | 3.62 to 2.85                     | 0.211     | 2.68 to 2.25                | 0.162     |
| L11   | 5.99 to 4.62                     | 0.228     | 5.92 to 4.29                | 0.275     |
| L13   | 8.58 to 6.86                     | 0.200     | 10.09 to 7.47               | 0.260     |
| L15   | 10.71 to 8.37                    | 0.218     | 16.72 to 12.29              | 0.265     |
| L17   | 19.02 to 14.42                   | 0.242     | 31.58 to 23.22              | 0.265     |

The loss grows with the layer, so the layers are compared by the reduction: 0.200 to 0.242 on the base checkpoint (the
smallest at L13, the largest at L17) and 0.162 to 0.275 on the post-trained one (the smallest at L9, the largest at
L11), against 0.190 to 0.241 (base) and 0.178 to 0.258
(post-trained) at 1.7B. LR1 has no honesty evaluation; C4 scores these adapters on TruthfulQA below.

#### The 0.6B pair (C4)

C4 on the 0.6B pair (runs `c4/qwen3-0.6b-base/b08c4e236dcf` and `c4/qwen3-0.6b/4d4ff337ade5`, made at `bab5b75` on a
rented GPU pod, see [Run identity](#run-identity)), which read the LR1 adapter of their checkpoint (the two LR1 runs
above). The config is that of the 1.7B and 4B runs (10 readers, 3 random arms, 173 test questions, 644 train and
validation questions, chance line 0.228), with batch size 16, and there is no earlier 0.6B C4 run. The unedited row of
the base checkpoint (0.237, 41 of 173 questions) is one question above the standard row of R6 (0.231, 40), and that of
the post-trained checkpoint (0.318, 55) is one below it (0.324, 56). The perplexity column is that of held-out neutral
text under the edited model; the prompting and `contrast` arms have no steered model, and the readers with coefficient
0 are the unedited model.

`qwen3-0.6b-base`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.237 [0.180, 0.306]       | 0                       | 8.14               |
| prompting +1         | 0.243 [0.185, 0.312]       | +0.006 [-0.029, +0.040] | -                  |
| prompting -1         | 0.243 [0.185, 0.312]       | +0.006 [-0.029, +0.046] | -                  |
| difference (0.05)    | 0.208 [0.154, 0.275]       | -0.029 [-0.069, +0.006] | 8.93               |
| contrast (0.25)      | 0.202 [0.149, 0.268]       | -0.035 [-0.075, +0.006] | -                  |
| LoRRA                | 0.231 [0.175, 0.299]       | -0.006 [-0.052, +0.040] | 8.34               |
| reading (10 readers) | mean 0.222, 0.197 to 0.237 | -0.040 to +0.000        | 8.14 to 136.42     |
| random (3 arms)      | mean 0.227, 0.202 to 0.243 | -0.035 to +0.006        | 13.26 to 13.39     |

`qwen3-0.6b`:

| Arm                  | Test MC1                   | Change from none        | Neutral perplexity |
| -------------------- | -------------------------- | ----------------------- | ------------------ |
| none                 | 0.318 [0.253, 0.391]       | 0                       | 10.55              |
| prompting +1         | 0.306 [0.242, 0.379]       | -0.012 [-0.046, +0.023] | -                  |
| prompting -1         | 0.243 [0.185, 0.312]       | -0.075 [-0.121, -0.029] | -                  |
| difference (0.05)    | 0.324 [0.258, 0.397]       | +0.006 [-0.023, +0.035] | 12.33              |
| contrast (0.25)      | 0.376 [0.307, 0.450]       | +0.058 [+0.006, +0.110] | -                  |
| LoRRA                | 0.364 [0.296, 0.438]       | +0.046 [-0.012, +0.104] | 11.66              |
| reading (10 readers) | mean 0.310, 0.295 to 0.318 | -0.023 to +0.000        | 10.55 to 12.25     |
| random (3 arms)      | mean 0.318, 0.318 to 0.318 | +0.000 to +0.000        | 10.78 to 16.32     |

- **Base checkpoint.** The unedited row (0.237 [0.180, 0.306]) has an interval that contains the chance line. The
  truthful and the untruthful prompt both change MC1 by +0.006 (1 question each of 173), with intervals that contain zero, so
  a changed prompt moves MC1 by the same amount whatever it says. No arm raises MC1: `difference` gives -0.029 [-0.069,
  +0.006] (5 fewer questions), `contrast` -0.035 [-0.075, +0.006] (6 fewer) and the adapter -0.006 [-0.052, +0.040] (1
  fewer), all with intervals that contain zero. Two of the ten readers chose coefficient 0 (the unedited model, so their
  change is 0 by construction), seven chose -0.1 and one chose -0.2, with changes of 2 to 7 questions fewer; the ten
  readers give -0.040 to +0.000, each with an interval that contains zero. The three random arms give -0.035 to +0.006,
  each with an interval that contains zero.
- **Post-trained checkpoint.** The unedited row (0.318 [0.253, 0.391]) has a lower bound above the chance line. The
  truthful prompt gives -0.012 [-0.046, +0.023] and the untruthful one -0.075 [-0.121, -0.029] (13 questions fewer), so
  only the second excludes zero and no prompt raises MC1. `contrast` gives +0.058 [+0.006, +0.110] (10 more questions)
  and reaches 0.376 [0.307, 0.450], the only arm of the table with a change whose interval excludes zero on the positive
  side (its lower bound is +0.006). The adapter gives +0.046 [-0.012, +0.104] (8 more) and 0.364 [0.296, 0.438], and
  `difference` +0.006 [-0.023, +0.035] (1 more), both with intervals that contain zero. Nine readers chose -0.05 and one
  chose 0; the ten readers give -0.023 to +0.000, each interval containing zero or (for one reader) ending at 0.000.
  The three random arms give 0.318 each, the same as the unedited row.
- **What is and is not supported.** The untruthful prompt lowers MC1 on the post-trained checkpoint (the interval
  excludes zero) and `contrast` raises it there by a change whose interval barely excludes zero; on the base checkpoint
  no arm has a change whose interval excludes zero. This is among the arms of the tables and with no correction for
  multiple comparisons. The `contrast` gain on the post-trained checkpoint (+0.058) is larger than the change of any
  random arm (0.000), but its test MC1 interval [0.307, 0.450] overlaps those of the random arms (the highest upper
  bound is 0.391) and that of the unedited row, and the adapter (+0.046) is not separated from zero. It is one run on one
  set of 173 questions. The `contrast` coefficient (0.25) and the `difference` coefficient (+0.05) are inside their
  grids on both checkpoints (the `contrast` grid goes to 0.5 and the coefficient grid to 0.2). On the base checkpoint
  one reader (-0.2) and all three random arms (-0.2, +0.2 and +0.2) were selected at the edge of the coefficient grid
  (the other readers chose -0.1 for seven and 0 for two); on the post-trained one a random arm (-0.2) was, the other two
  random arms chose -0.05 and +0.1, and the readers chose -0.05 for nine and 0 for one.
- **Cost to the text.** The adapter leaves the perplexity of neutral text near the unedited one on the base
  checkpoint, 8.34 against 8.14 (1.02 times), and raises it on the post-trained one, 11.66 against 10.55 (1.10 times;
  1.09 times at 1.7B and 1.15 times at 4B). On the base checkpoint the settings chosen by selection are not all benign:
  `difference` at +0.05 has a perplexity of 8.93 (1.10 times), the seven readers with coefficient -0.1 have 13.35 to
  14.08 (1.64 to 1.73 times), the reader with -0.2 has 136.42 (16.75 times), and the random arms 13.26 to 13.39 (1.63
  to 1.64 times). On the post-trained checkpoint the nine readers with a nonzero coefficient are at 11.74 to 12.25
  (1.11 to 1.16 times), `difference` at +0.05 at 12.33 (1.17 times) and the random arms at 10.78 to 16.32 (1.02 to 1.55
  times). The scoring is not generation, so the distinct-1 and the completion fluency of the text are not measured.

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

#### The 4B pair (C5)

The same design on `qwen3-4b-base` and `qwen3-4b` (36 blocks; runs `c5/qwen3-4b-base/3d68f877eed5` and
`c5/qwen3-4b/fd207b9d8079`, made at `d09822b` on a rented GPU pod; see [Run identity](#run-identity)): 1,000 TriviaQA
questions, one sample per question at temperature 0.8, relative strength 0.1, steered over layers 10, 13, 17, 20, 23, 27,
30 and 34, three random arms steered at +0.1 and -0.1. The base checkpoint is prompted with 15 question and answer shots
and the post-trained one with the question alone in the chat template, so the two columns are not a comparison of the
models. Each run reads the C2 run of its checkpoint: `c2/qwen3-4b-base/984485501f3e` and `c2/qwen3-4b/1df7c1f9ff70`.
A share is taken over the non-empty answers. The base checkpoint returned one empty answer unsteered, 15 under the lie
arm, none under the honesty arm and one under three of the six random arms, so its unsteered share is 473 of 999 and its
lie-arm share 316 of 985; the post-trained checkpoint returned none. The random rows are the mean of the three random directions.

| Quantity                                     | qwen3-4b-base        | qwen3-4b             |
| -------------------------------------------- | -------------------- | -------------------- |
| Correct, no steering                         | 0.473                | 0.336                |
| Correct, honesty-steered                     | 0.424                | 0.299                |
| Correct, lie-steered                         | 0.321                | 0.324                |
| Correct, random +0.1 (mean of 3)             | 0.455                | 0.328                |
| Correct, random -0.1 (mean of 3)             | 0.452                | 0.329                |
| Answered correctly without steering          | 473                  | 336                  |
| Of these, answered wrongly under the lie arm | 170                  | 54                   |
| Flip rate [95% interval]                     | 0.359 [0.317, 0.404] | 0.161 [0.125, 0.204] |
| Perplexity: none                             | 6.4                  | 8.0                  |
| Perplexity: honesty-steered                  | 7.3                  | 9.8                  |
| Perplexity: lie-steered                      | 8.7                  | 9.6                  |
| Perplexity: random (range of 6)              | 6.7 to 6.8           | 8.0 to 8.5           |
| Distinct-1: none                             | 0.137                | 0.238                |
| Distinct-1: honesty-steered                  | 0.130                | 0.227                |
| Distinct-1: lie-steered                      | 0.162                | 0.237                |
| Distinct-1: random (range of 6)              | 0.139 to 0.148       | 0.232 to 0.237       |

- **`qwen3-4b-base`: both steered arms are below the unsteered one, the lie arm far below.** The honesty arm is 0.049
  below (0.424 against 0.473) and the lie arm 0.152 below (0.321), 0.103 below the honesty arm. The random arms are 0.018
  and 0.022 below the unsteered share (0.455 and 0.452). The honesty arm is 0.031 and 0.028 below the random arms and the
  lie arm 0.134 and 0.131 below them. The shares carry no interval; for scale, a share of 1,000 questions near 0.45 has a
  binomial standard error of about 0.016, and the standard error of a difference of two such shares is about 0.022, so
  the honesty-arm gap to the random arms is roughly 1.3 to 1.4 standard errors and the lie-arm gap roughly 6. That is a
  larger lie-arm effect than at 1.7B (0.073 below the unsteered share on the base checkpoint, 0.048 to 0.059 below the
  random arms).
- **`qwen3-4b`: the lie arm is not separated from the random arms.** The lie arm is 0.324, 0.012 below the unsteered
  share (0.336) and 0.004 and 0.005 below the random arms (0.328 and 0.329), well inside the scale above (a share near
  0.33 has a standard error of about 0.015, a difference about 0.021). The honesty arm is lower (0.299, 0.037 below the
  unsteered share and 0.029 and 0.030 below the random arms, roughly 1.4 standard errors), so the order of the two
  steered arms is the reverse of the other five checkpoints, where the lie arm is the lowest.
- **Flip rate.** 170 of the 473 questions answered correctly are answered wrongly under the lie arm on the base
  checkpoint (0.359 [0.317, 0.404]) and 54 of 336 on the post-trained one (0.161 [0.125, 0.204]). The post-trained
  interval is below the intervals of both 1.7B checkpoints (0.262 [0.218, 0.311] and 0.363 [0.304, 0.426]); the base
  interval is above that of `qwen3-1.7b-base` (lower bound 0.317 against an upper bound of 0.311) and overlaps that of
  `qwen3-1.7b`. A question that gets an empty completion under the lie arm is not counted as flipped. The honesty arm
  and the random arms have no stored flip rate, so the flip rate is not separated from the effect of a generic
  perturbation of the model.
- **Cost to the text.** The perplexity of neutral text under the honesty and lie arms is 1.15 and 1.37 times the
  unsteered one on the base checkpoint and 1.23 and 1.21 times on the post-trained one, against 1.06 to 1.07 and 1.01 to
  1.07 for the random arms. The lie arm moves the base text the most; on the post-trained checkpoint the two arms move it
  by the same amount (9.8 and 9.6), more than the random arms. The distinct-1 cutoff of the rule in [Sentiment steering
  on Qwen3](#sentiment-steering-on-qwen3-c1) (0.15) is not used here: the unsteered answers of the base checkpoint are
  already at 0.137, and the lie arm of that checkpoint has a higher distinct-1 (0.162) than the unsteered text, with
  shorter answers (19.8 tokens on average against 21.6) and the 15 empty completions.
- The run also stores a small gallery of completions and the emotion gallery of the C2 run of the same checkpoint, which
  are a selection and are not read here.

## Position and stimulus-set size on qwen3-1.7b-base and qwen3-1.7b (P2, P3)

P2 ran on the Qwen3 1.7B pair, the base `qwen3-1.7b-base` and the post-trained `qwen3-1.7b`, and P3 ran here on the base 1.7B checkpoint only; P2 later ran on `qwen3-0.6b-base`, `qwen3-0.6b`, `gpt2-medium`, `gpt2-large` and `gpt2-xl`, and on the Qwen3 4B pair (last subsections below). Both use the settings of the `gpt2` runs (logistic finder, 64 training pairs; 10 seeds for P2 and 20 draws per size for P3, the intervals of the [reading conventions](#reading-conventions)). Each is one run: `p2/qwen3-1.7b-base/119f0ac54451` and `p3/qwen3-1.7b-base/61baf30a8a59`, made at `c7d8051` with a clean tree on a cloud GPU (NVIDIA GeForce RTX 4090), and `p2/qwen3-1.7b/51a85fbec631`, made at `20650c5` with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU). The two P2 runs therefore differ in hardware and code commit as well as in checkpoint; the P2 code is the same at both commits (only `cloud.py` differs under `src`). P3 on the 0.6B checkpoints, on the post-trained 1.7B checkpoint and on the 4B pair is in [the section on P3 and P4 across the GPT-2 sizes and Qwen3](#stimulus-set-size-and-direction-finder-across-the-gpt-2-sizes-and-qwen3-p3-p4). Base and post-trained are two models: their P2 tables are separate and no metric is pooled over them, and the `gpt2` and Qwen3 numbers come from different models, not from a scaling series.

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

### Token position and a misleading sentence on the Qwen3 4B pair (P2)

P2 also ran on the 4B pair (runs `p2/qwen3-4b-base/64228009c695` and `p2/qwen3-4b/bf265bb382ea`; 36 blocks), both made at
`86c319b` with a clean tree on one rented GPU pod (NVIDIA RTX A4000) with batch size 8 (see [Run identity](#run-identity)).
The design, the report splits (214, 59 and 256 pairs) and the layer rule are those of the runs above. Like the 0.6B pair,
and unlike the 1.7B pair, the two runs differ in checkpoint only (same hardware and commit). The two tables are separate
and no metric is pooled over them.

#### qwen3-4b-base

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L24   | 0.901 [0.886, 0.915] | 0.958 [0.951, 0.964] | 0.957 [0.953, 0.961] |
| sentiment | concept   | L20   | 0.905 [0.893, 0.918] | 0.866 [0.839, 0.886] | 0.800 [0.771, 0.826] |
| sentiment | mean      | L20   | 0.911 [0.901, 0.921] | 0.907 [0.898, 0.918] | 0.904 [0.893, 0.915] |
| sentiment | zero-shot | -     | 0.963 [0.935, 0.986] | 0.935 [0.897, 0.963] | 0.827 [0.776, 0.874] |
| truth     | last      | L23   | 1.000 [1.000, 1.000] | 0.997 [0.992, 1.000] | 0.983 [0.975, 0.992] |
| truth     | concept   | L24   | 0.936 [0.927, 0.942] | 0.983 [0.978, 0.988] | 0.936 [0.912, 0.954] |
| truth     | mean      | L22   | 0.897 [0.873, 0.919] | 0.892 [0.868, 0.915] | 0.880 [0.854, 0.903] |
| truth     | zero-shot | -     | 1.000 [1.000, 1.000] | 0.983 [0.949, 1.000] | 0.932 [0.864, 0.983] |
| utility   | last      | L21   | 0.781 [0.764, 0.795] | 0.764 [0.753, 0.775] | 0.568 [0.511, 0.627] |
| utility   | concept   | L20   | 0.802 [0.786, 0.820] | 0.780 [0.770, 0.790] | 0.692 [0.630, 0.747] |
| utility   | mean      | L20   | 0.708 [0.684, 0.730] | 0.723 [0.696, 0.747] | 0.647 [0.623, 0.670] |
| utility   | zero-shot | -     | 0.773 [0.719, 0.820] | 0.727 [0.672, 0.781] | 0.230 [0.180, 0.285] |

- **Position.** On the original prompts the positions read within 0.010 of each other for sentiment (0.901, 0.905,
  0.911), and truth and utility spread by 0.103 and 0.094 (truth 1.000, 0.936, 0.897; utility 0.781, 0.802, 0.708). Truth
  at `last` is 1.000 [1.000, 1.000]: all 10 readers score all 59 report pairs correctly. `last` is the highest position
  for truth only; for sentiment `mean` is the highest, with overlapping intervals, and for utility `concept` is (0.802
  against 0.781 at `last`).
- **Random decoy.** At `last` the reader rises by 0.057 for sentiment (0.901 to 0.958, intervals that do not overlap) and
  moves by -0.003 for truth (1.000 to 0.997) and -0.016 for utility (0.781 to 0.764). At `concept` sentiment moves by
  -0.038 (0.905 to 0.866), truth by +0.047 (0.936 to 0.983) and utility by -0.022 (0.802 to 0.780). At `mean` no task
  moves by more than 0.015.
- **Against decoy.** At `last` sentiment rises to 0.957 [0.953, 0.961] (+0.057), as under the random decoy, truth is
  0.983 (-0.017) and utility falls to 0.568 [0.511, 0.627] (-0.213). At `concept` sentiment drops 0.104 (0.800), truth is
  unchanged (0.936) and utility drops 0.110 (0.692). At `mean` sentiment (0.904) and truth (0.880) are within 0.017 of the
  original with overlapping intervals; utility is 0.647 [0.623, 0.670], a drop of 0.061 with an interval that does not
  overlap the original [0.684, 0.730]. No cell is below chance: the lowest is utility at `last` (0.568), with a lower
  bound of 0.511. Under this decoy two or three of the nine cells are below chance on each of the 0.6B and 1.7B
  checkpoints. The run does not say why a decoy sentence raises the reader at `last` for sentiment.
- **Zero-shot.** The model's own answer goes from 0.963, 1.000 and 0.773 to 0.935, 0.983 and 0.727 under the random decoy
  and to 0.827, 0.932 and 0.230 under the `against` decoy. Only utility turns around (0.230 [0.180, 0.285], below chance);
  sentiment and truth stay high, where the 1.7B base checkpoint fell to 0.248 and 0.424. The reader on utility stays above
  chance under the same decoy (0.568 at `last`, 0.692 at `concept`, 0.647 at `mean`), so the decoy that turns the model's
  verbalized answer on utility does not turn that reader.

#### qwen3-4b

| Task      | Position  | Layer | Original             | Random decoy         | Against decoy        |
| --------- | --------- | ----- | -------------------- | -------------------- | -------------------- |
| sentiment | last      | L23   | 0.753 [0.726, 0.782] | 0.754 [0.722, 0.783] | 0.710 [0.673, 0.747] |
| sentiment | concept   | L20   | 0.891 [0.880, 0.902] | 0.843 [0.829, 0.857] | 0.760 [0.740, 0.785] |
| sentiment | mean      | L15   | 0.893 [0.880, 0.904] | 0.905 [0.893, 0.915] | 0.903 [0.891, 0.914] |
| sentiment | zero-shot | -     | 0.935 [0.902, 0.967] | 0.907 [0.864, 0.944] | 0.799 [0.743, 0.850] |
| truth     | last      | L23   | 0.968 [0.961, 0.975] | 0.927 [0.919, 0.936] | 0.808 [0.800, 0.819] |
| truth     | concept   | L29   | 0.946 [0.936, 0.958] | 0.931 [0.919, 0.942] | 0.853 [0.836, 0.869] |
| truth     | mean      | L22   | 0.908 [0.883, 0.932] | 0.898 [0.881, 0.915] | 0.895 [0.880, 0.910] |
| truth     | zero-shot | -     | 0.932 [0.864, 0.983] | 0.881 [0.797, 0.949] | 0.407 [0.271, 0.525] |
| utility   | last      | L21   | 0.785 [0.774, 0.794] | 0.744 [0.719, 0.768] | 0.641 [0.565, 0.720] |
| utility   | concept   | L19   | 0.766 [0.753, 0.779] | 0.765 [0.755, 0.776] | 0.650 [0.609, 0.692] |
| utility   | mean      | L21   | 0.702 [0.678, 0.724] | 0.713 [0.693, 0.732] | 0.622 [0.606, 0.641] |
| utility   | zero-shot | -     | 0.770 [0.715, 0.820] | 0.699 [0.641, 0.754] | 0.195 [0.152, 0.246] |

- **Position.** On the original prompts the positions `concept` and `mean` read within 0.002 of each other for
  sentiment (0.891, 0.893), but `last` is 0.138 and 0.140 lower (0.753 [0.726, 0.782]; the R1 sentiment reader of this checkpoint
  is 0.760 at L23, in [LAT reading on Qwen3](#lat-reading-on-qwen3)). Truth spreads by 0.059 (0.968, 0.946, 0.908) and
  utility by 0.083 (0.785, 0.766, 0.702). `last` is the highest position for truth and utility (for utility within 0.019
  of `concept`) and the lowest for sentiment, as on `qwen3-1.7b`.
- **Random decoy.** At `last` the reader moves by +0.001 for sentiment (0.753 to 0.754), -0.041 for truth (0.968 to 0.927)
  and -0.041 for utility (0.785 to 0.744). At `concept` sentiment moves by -0.048 (0.891 to 0.843), truth by -0.015 and
  utility by -0.001. At `mean` no task moves by more than 0.013.
- **Against decoy.** At `last` the reader falls on every task: sentiment to 0.710 [0.673, 0.747] (-0.043), truth to 0.808
  (-0.159) and utility to 0.641 [0.565, 0.720] (-0.143). At `concept` sentiment drops 0.131 (0.760), truth 0.093 (0.853)
  and utility 0.116 (0.650). At `mean` sentiment (0.903) and truth (0.895) are within 0.014 of the original with
  overlapping intervals; utility is 0.622 [0.606, 0.641], a drop of 0.080 with an interval that does not overlap the
  original [0.678, 0.724]. No cell is below chance; the lowest is utility at `mean` (0.622).
- **Zero-shot.** The model's own answer goes from 0.935, 0.932 and 0.770 to 0.907, 0.881 and 0.699 under the random decoy
  and to 0.799, 0.407 and 0.195 under the `against` decoy (for truth the interval, [0.271, 0.525], reaches 0.5; for
  utility it is [0.152, 0.246]). The reader at `mean` stays at 0.903, 0.895 and 0.622 under the `against` decoy, so on
  truth and utility the decoy that turns the model's verbalized answer around does not turn that reader.

#### Base against post-trained at 4B

The two checkpoints are two models and each has one run, so this compares cells and does not pool them.

- **Original prompts.** The post-trained point value is lower in six of the nine cells. In three the intervals do not
  overlap: sentiment at `last` (0.753 against 0.901), truth at `last` (0.968 against 1.000) and utility at `concept`
  (0.766 against 0.802). In the other six the intervals overlap and the differences are at most 0.019; the post-trained
  reader is higher in three of them (truth at `concept` and `mean`, utility at `last`, by 0.010, 0.012 and 0.004).
- **Against decoy.** Neither checkpoint has a reader cell below chance (the lowest are 0.568 and 0.622). The largest
  differences are at `last`: sentiment 0.957 on the base checkpoint against 0.710 on the post-trained one (the base
  reader rises under the decoy, the post-trained one falls) and truth 0.983 against 0.808; utility at `last` runs the
  other way, 0.568 on the base and 0.641 on the post-trained checkpoint.
- **Zero-shot.** The post-trained point value is lower than the base one in all nine task and condition cells. The
  largest gap is truth under the `against` decoy (0.932 on the base checkpoint, 0.407 on the post-trained one, intervals
  that do not overlap); the intervals also fail to overlap for truth under the original prompts (1.000 against 0.932
  [0.864, 0.983]).
- **Layers.** The layer chosen on the original prompts differs between the checkpoints in 5 of the 9 reader cells:
  sentiment at `last` (L24 against L23) and `mean` (L20 against L15), truth at `concept` (L24 against L29), and utility
  at `concept` (L20 against L19) and `mean` (L20 against L21). A difference in those cells mixes the checkpoint with the
  layer selected on the select split.
- **Limits.** The intervals are over the 10 training draws and do not include the sampling of the evaluation pairs, which
  matters most for truth (59 report pairs, and 1.000 [1.000, 1.000] on the base checkpoint at `last`).

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

## Stimulus-set size and direction finder across the GPT-2 sizes and Qwen3 (P3, P4)

P3 and P4 are described under [gpt2](#position-sample-size-and-direction-finder-on-gpt2-p2-p3-p4), where [P3](#stimulus-set-size-p3) and [P4](#direction-finder-pairing-and-preprocessing-p4) ran on `gpt2` only; P3 on `qwen3-1.7b-base` is reported [with P2](#stimulus-set-size-on-qwen3-17b-base-p3). This section reports the other runs, all with the settings of the `gpt2` runs (P3 with the logistic finder and 20 draws per size from 2 to 512 pairs, P4 with the six finders, 64 training pairs and 10 seeds, the intervals of the [reading conventions](#reading-conventions)): P3 on `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b-base` and `qwen3-4b`, and P4 on `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base`, `qwen3-1.7b`, `qwen3-4b-base` and `qwen3-4b`. Each is one run, at the default config of its experiment (batch size 8): `p3/gpt2-medium/b80a5a3c0faf`, `p3/gpt2-large/327dd31a1de6`, `p3/gpt2-xl/12ee53fd16fe`, `p3/qwen3-0.6b-base/776d454d5ca2`, `p3/qwen3-0.6b/b38783812e71`, `p3/qwen3-1.7b/72edcdd9158e`, `p3/qwen3-4b-base/44efe2f36376`, `p3/qwen3-4b/3f8cb6317de0`, `p4/gpt2-medium/9a2f3f02568b`, `p4/gpt2-large/f4e8481cbbbf`, `p4/gpt2-xl/427a6b2f9da4`, `p4/qwen3-0.6b-base/3b2677205f7d`, `p4/qwen3-0.6b/e40fa46cf517`, `p4/qwen3-1.7b-base/b772e1cc4db9`, `p4/qwen3-1.7b/607e86d9c7f6`, `p4/qwen3-4b-base/346291dfa5d1` and `p4/qwen3-4b/36371d16b354`. The `gpt2-medium` runs were made at `fb3b5dd` with a clean tree on CPU, and the others at `80baffb` with a clean tree on rented GPU pods (NVIDIA GeForce RTX 3090, except `p4/qwen3-1.7b-base`, which ran on an NVIDIA L4); the four 4B runs were made later, at `bab5b75` with a clean tree on one rented pod (NVIDIA RTX A5000, two jobs at a time, batch size 8); the config hashes of all of them agree with the hashes recomputed under the current code (see [Run identity](#run-identity)). Base and post-trained checkpoints are two models: their tables are separate and no metric is pooled over them, and the `gpt2` and Qwen3 numbers come from different models, not from a scaling series.

### Stimulus-set size (P3)

The design is that of [P3 on gpt2](#stimulus-set-size-p3): 2 to 512 pairs, 20 draws per size, the mean accuracy in each cell and in parentheses the share of the 20 draws whose accuracy is below 0.6. The truth set has 172 pairs, so its 256 and 512 cells are skipped, as is the 512 cell of `truth_cities` (452 pairs). `truth_cities` and `truth_common` fit the reader on those subsets of the truth statements and score it on the truth target. The intervals quoted in the text are over the 20 draws; they do not include the sampling of the evaluation pairs, which matters most for the truth target (59 report pairs). Each table is followed by the accuracy of a reader fitted on all the data of the source (the reference, report split) and the range of the shuffled-label series.

#### gpt2-medium

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.523 (1.00) | 0.564 (0.80) | 0.588 (0.70) | 0.626 (0.35) | 0.684 (0.10) | 0.754 (0.00) | 0.797 (0.00) | 0.837 (0.00) | 0.869 (0.00) |
| truth -> truth         | 0.543 (0.75) | 0.567 (0.65) | 0.583 (0.55) | 0.602 (0.50) | 0.624 (0.35) | 0.651 (0.10) | 0.654 (0.00) | -            | -            |
| truth_cities -> truth  | 0.550 (0.70) | 0.569 (0.70) | 0.563 (0.70) | 0.610 (0.45) | 0.683 (0.10) | 0.714 (0.00) | 0.730 (0.00) | 0.736 (0.00) | -            |
| truth_common -> truth  | 0.551 (0.75) | 0.503 (0.85) | 0.564 (0.75) | 0.617 (0.40) | 0.621 (0.25) | 0.631 (0.30) | 0.664 (0.10) | 0.691 (0.05) | 0.717 (0.00) |
| utility -> utility     | 0.522 (0.65) | 0.508 (0.85) | 0.602 (0.40) | 0.598 (0.40) | 0.637 (0.15) | 0.684 (0.00) | 0.695 (0.00) | 0.709 (0.00) | 0.710 (0.00) |

The reference is 0.911 for sentiment (2,048 pairs), 0.678 for truth (172), 0.763 for `truth_cities` (452), 0.729 for `truth_common` (1,296) and 0.719 for utility (2,048). The shuffled-label series is between 0.47 and 0.53 at every size.

- **Sentiment** has no failing draw from 64 pairs (0.754 [0.736, 0.771]; at 32 pairs the mean is 0.684 with 0.10 failing) and still rises at 512 (0.869 [0.861, 0.878]), below the reference of 0.911. **Utility** has no failing draw from 64 pairs (0.684) and gains 0.027 between 64 and 512 (0.710 [0.701, 0.718]), against 0.719 for the full data.
- **Truth** has failing draws up to 64 pairs (0.35 at 32 pairs, 0.10 at 64) and none at 128, where the mean is 0.654 [0.644, 0.664], below its reference of 0.678; the source is exhausted at 172 pairs. `truth_cities` has no failing draw from 64 pairs and reaches 0.736 [0.725, 0.746] at 256, below its reference of 0.763. `truth_common` is at 0.631 at 64 pairs (failure 0.30) and 0.717 [0.700, 0.733] at 512, with no failing draw, near its reference of 0.729 and still rising (it gains 0.026 from 256 to 512).
- The stored saturation size is the smallest size whose mean accuracy is within 0.01 of the mean at the largest size, with its bootstrap range: 512 for sentiment (512 to 512), 64 for truth (32 to 128), 128 for `truth_cities` (64 to 256), 512 for `truth_common` (256 to 512) and 256 for utility (64 to 512). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set and do not show a plateau beyond it.

#### gpt2-large

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.514 (1.00) | 0.538 (0.85) | 0.593 (0.50) | 0.648 (0.25) | 0.721 (0.00) | 0.805 (0.00) | 0.855 (0.00) | 0.880 (0.00) | 0.899 (0.00) |
| truth -> truth         | 0.542 (0.80) | 0.551 (0.80) | 0.564 (0.65) | 0.575 (0.65) | 0.591 (0.60) | 0.639 (0.25) | 0.676 (0.00) | -            | -            |
| truth_cities -> truth  | 0.503 (1.00) | 0.500 (1.00) | 0.484 (1.00) | 0.469 (1.00) | 0.478 (1.00) | 0.480 (1.00) | 0.475 (1.00) | 0.491 (1.00) | -            |
| truth_common -> truth  | 0.509 (0.95) | 0.480 (0.95) | 0.518 (0.80) | 0.553 (0.70) | 0.571 (0.65) | 0.576 (0.55) | 0.619 (0.30) | 0.654 (0.05) | 0.674 (0.05) |
| utility -> utility     | 0.533 (0.70) | 0.527 (0.75) | 0.579 (0.60) | 0.593 (0.50) | 0.631 (0.15) | 0.674 (0.00) | 0.686 (0.00) | 0.704 (0.00) | 0.711 (0.00) |

The reference is 0.935 for sentiment (2,048 pairs), 0.695 for truth (172), 0.508 for `truth_cities` (452), 0.661 for `truth_common` (1,296) and 0.719 for utility (2,048). The shuffled-label series is between 0.47 and 0.54 at every size.

- **Sentiment** has no failing draw from 32 pairs (0.721 [0.705, 0.736]) and reaches 0.899 [0.894, 0.904] at 512, below the reference of 0.935; it gains 0.019 between 256 and 512 (0.880 to 0.899). **Utility** has no failing draw from 64 pairs (0.674) and gains 0.037 between 64 and 512 (0.711 [0.701, 0.721]), against 0.719 for the full data.
- **Truth** has a failure rate of 0.25 at 64 pairs (0.639) and none at 128, where the mean is 0.676 [0.667, 0.685], below its reference of 0.695; the source is exhausted at 172 pairs. `truth_common` is at 0.619 at 128 pairs (failure 0.30) and 0.674 [0.659, 0.688] at 512 (failure 0.05), above its reference of 0.661.
- **`truth_cities` is at chance on the truth target.** The mean accuracy is between 0.469 and 0.503 at every size, and all 20 draws are below 0.6 at every size (failure 1.00). The reference reader fitted on all 452 `truth_cities` pairs scores 0.508 on the report split, and 0.642 on the select split, which chose layer 11. On `gpt2` and `gpt2-medium` the same row reads 0.638 and 0.736 at 256 pairs. The run does not say why.
- The stored saturation size is 512 for sentiment (512 to 512), 128 for truth (128 to 128), 2 for `truth_cities` (2 to 4), 512 for `truth_common` (256 to 512) and 256 for utility (256 to 512). The `truth_cities` value belongs to a series that is at chance at every size and marks no plateau of a reader. For truth the largest size is the end of a small set (128 pairs), so its saturation size is measured against the end of the set.

#### gpt2-xl

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.533 (1.00) | 0.554 (0.90) | 0.583 (0.60) | 0.638 (0.25) | 0.728 (0.00) | 0.800 (0.00) | 0.853 (0.00) | 0.898 (0.00) | 0.931 (0.00) |
| truth -> truth         | 0.525 (0.90) | 0.559 (0.75) | 0.566 (0.85) | 0.578 (0.60) | 0.617 (0.35) | 0.636 (0.10) | 0.662 (0.00) | -            | -            |
| truth_cities -> truth  | 0.527 (1.00) | 0.502 (1.00) | 0.547 (0.95) | 0.563 (0.75) | 0.607 (0.40) | 0.609 (0.35) | 0.620 (0.30) | 0.607 (0.30) | -            |
| truth_common -> truth  | 0.512 (0.95) | 0.497 (1.00) | 0.505 (0.90) | 0.530 (0.90) | 0.560 (0.60) | 0.576 (0.75) | 0.616 (0.40) | 0.656 (0.05) | 0.677 (0.00) |
| utility -> utility     | 0.543 (0.70) | 0.534 (0.85) | 0.612 (0.45) | 0.608 (0.55) | 0.651 (0.05) | 0.679 (0.00) | 0.688 (0.00) | 0.704 (0.00) | 0.716 (0.00) |

The reference is 0.953 for sentiment (2,048 pairs), 0.661 for truth (172), 0.644 for `truth_cities` (452), 0.661 for `truth_common` (1,296) and 0.727 for utility (2,048). The shuffled-label series is between 0.46 and 0.54 at every size.

- **Sentiment** has no failing draw from 32 pairs (0.728 [0.710, 0.747]) and reaches 0.931 [0.926, 0.936] at 512, below the reference of 0.953; it gains 0.033 between 256 and 512 (0.898 to 0.931). **Utility** has no failing draw from 64 pairs (0.679) and gains 0.037 between 64 and 512 (0.716 [0.708, 0.724]), against 0.727 for the full data.
- **Truth** has a failure rate of 0.35 at 32 pairs and 0.10 at 64 (0.636) and none at 128, where the mean is 0.662 [0.653, 0.670], at its reference of 0.661; the source is exhausted at 172 pairs. `truth_cities` has no size without failing draws: the failure rate is 0.30 at 128 and at 256 pairs (0.620 [0.608, 0.633] and 0.607 [0.598, 0.615]), against a reference of 0.644. `truth_common` is at 0.576 at 64 pairs (failure 0.75), 0.656 at 256 (0.05) and 0.677 [0.664, 0.690] at 512 with no failing draw, above its reference of 0.661 and still rising.
- The stored saturation size is 512 for sentiment (512 to 512), 128 for truth (128 to 128), 32 for `truth_cities` (32 to 128), 512 for `truth_common` (256 to 512) and 512 for utility (256 to 512). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set.

#### qwen3-0.6b-base

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.653 (0.20) | 0.725 (0.05) | 0.790 (0.00) | 0.833 (0.00) | 0.903 (0.00) | 0.919 (0.00) | 0.935 (0.00) | 0.942 (0.00) | 0.944 (0.00) |
| truth -> truth         | 0.575 (0.60) | 0.602 (0.40) | 0.667 (0.20) | 0.744 (0.05) | 0.786 (0.00) | 0.814 (0.00) | 0.830 (0.00) | -            | -            |
| truth_cities -> truth  | 0.677 (0.20) | 0.655 (0.25) | 0.692 (0.00) | 0.731 (0.05) | 0.743 (0.00) | 0.762 (0.00) | 0.759 (0.00) | 0.773 (0.00) | -            |
| truth_common -> truth  | 0.560 (0.75) | 0.546 (0.80) | 0.574 (0.65) | 0.629 (0.25) | 0.671 (0.15) | 0.714 (0.05) | 0.748 (0.05) | 0.754 (0.00) | 0.762 (0.00) |
| utility -> utility     | 0.616 (0.30) | 0.614 (0.40) | 0.680 (0.10) | 0.686 (0.00) | 0.704 (0.00) | 0.711 (0.00) | 0.714 (0.00) | 0.715 (0.00) | 0.707 (0.00) |

The reference is 0.949 for sentiment (2,048 pairs), 0.831 for truth (172), 0.780 for `truth_cities` (452), 0.780 for `truth_common` (1,296) and 0.715 for utility (2,048). The shuffled-label series is between 0.47 and 0.55 at every size.

- **Sentiment** is at 0.653 [0.622, 0.682] with 2 pairs (failure 0.20), has no failing draw from 8 pairs (0.790) and reaches 0.944 [0.941, 0.948] at 512, just below the reference of 0.949 (0.919 at 64 pairs). **Utility** plateaus early: 0.680 at 8 pairs (failure 0.10), 0.686 at 16 pairs with no failing draw, 0.711 [0.703, 0.719] at 64 and 0.707 [0.698, 0.715] at 512, below its reference of 0.715; from 64 pairs on the mean stays between 0.707 and 0.715.
- **Truth** is at 0.575 with 2 pairs (failure 0.60), 0.744 at 16 pairs (0.05) and has no failing draw from 32 pairs (0.786); it reaches 0.830 [0.820, 0.839] at 128, at its reference of 0.831, and the source is exhausted at 172 pairs. `truth_cities` is at 0.677 [0.648, 0.707] with 2 pairs (failure 0.20), has no failing draw from 32 pairs (0.743) and reaches 0.773 [0.768, 0.778] at 256, below its reference of 0.780. `truth_common` needs more: 0.560 at 2 pairs (failure 0.75), 0.714 at 64 (0.05), no failing draw from 256 pairs (0.754) and 0.762 [0.753, 0.773] at 512, below its reference of 0.780.
- The stored saturation size is 128 for sentiment (128 to 256), 128 for truth (64 to 128), 256 for `truth_cities` (64 to 256), 256 for `truth_common` (128 to 512) and 32 for utility (8 to 64). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set.

#### qwen3-0.6b

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.593 (0.50) | 0.606 (0.55) | 0.666 (0.25) | 0.722 (0.05) | 0.793 (0.00) | 0.834 (0.00) | 0.868 (0.00) | 0.889 (0.00) | 0.913 (0.00) |
| truth -> truth         | 0.530 (0.95) | 0.538 (0.80) | 0.591 (0.65) | 0.648 (0.20) | 0.712 (0.05) | 0.780 (0.00) | 0.838 (0.00) | -            | -            |
| truth_cities -> truth  | 0.585 (0.60) | 0.603 (0.60) | 0.651 (0.25) | 0.680 (0.05) | 0.717 (0.00) | 0.751 (0.00) | 0.742 (0.00) | 0.750 (0.00) | -            |
| truth_common -> truth  | 0.525 (0.95) | 0.521 (0.95) | 0.531 (0.90) | 0.559 (0.75) | 0.565 (0.75) | 0.626 (0.25) | 0.645 (0.15) | 0.681 (0.00) | 0.699 (0.00) |
| utility -> utility     | 0.644 (0.20) | 0.629 (0.25) | 0.703 (0.00) | 0.703 (0.00) | 0.708 (0.00) | 0.706 (0.00) | 0.712 (0.00) | 0.720 (0.00) | 0.715 (0.00) |

The reference is 0.930 for sentiment (2,048 pairs), 0.847 for truth (172), 0.712 for `truth_cities` (452), 0.763 for `truth_common` (1,296) and 0.738 for utility (2,048). The shuffled-label series is between 0.48 and 0.54 at every size.

- **Sentiment** is at 0.593 [0.544, 0.640] with 2 pairs (failure 0.50), has no failing draw from 32 pairs (0.793) and reaches 0.913 [0.908, 0.917] at 512, below the reference of 0.930; it gains 0.024 between 256 and 512 (0.889 to 0.913). **Utility** has no failing draw from 8 pairs (0.703) and the mean stays between 0.703 and 0.720 from 8 pairs on (0.715 [0.704, 0.726] at 512), below its reference of 0.738.
- **Truth** is at 0.530 with 2 pairs (failure 0.95), 0.712 at 32 pairs (0.05) and has no failing draw from 64 pairs (0.780); it reaches 0.838 [0.826, 0.851] at 128, below its reference of 0.847. `truth_cities` has no failing draw from 32 pairs (0.717) and stays between 0.742 and 0.751 from 64 pairs on (0.750 [0.742, 0.759] at 256), above its reference of 0.712. `truth_common` is slow: 0.626 at 64 pairs (failure 0.25), no failing draw from 256 pairs (0.681) and 0.699 [0.683, 0.715] at 512, still rising and below its reference of 0.763.
- The stored saturation size is 512 for sentiment (512 to 512), 128 for truth (128 to 128), 64 for `truth_cities` (64 to 256), 512 for `truth_common` (256 to 512) and 32 for utility (8 to 256). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set.

#### qwen3-1.7b-base

Reported above with P2, in [Stimulus-set size on qwen3-1.7b-base (P3)](#stimulus-set-size-on-qwen3-17b-base-p3).

#### qwen3-1.7b

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.586 (0.50) | 0.626 (0.30) | 0.692 (0.15) | 0.785 (0.00) | 0.840 (0.00) | 0.888 (0.00) | 0.907 (0.00) | 0.934 (0.00) | 0.946 (0.00) |
| truth -> truth         | 0.836 (0.00) | 0.840 (0.05) | 0.881 (0.00) | 0.925 (0.00) | 0.929 (0.00) | 0.920 (0.00) | 0.932 (0.00) | -            | -            |
| truth_cities -> truth  | 0.830 (0.00) | 0.818 (0.00) | 0.844 (0.00) | 0.881 (0.00) | 0.876 (0.00) | 0.897 (0.00) | 0.891 (0.00) | 0.897 (0.00) | -            |
| truth_common -> truth  | 0.669 (0.35) | 0.672 (0.20) | 0.755 (0.05) | 0.786 (0.00) | 0.841 (0.00) | 0.859 (0.00) | 0.904 (0.00) | 0.893 (0.00) | 0.900 (0.00) |
| utility -> utility     | 0.665 (0.20) | 0.707 (0.05) | 0.749 (0.00) | 0.742 (0.00) | 0.760 (0.00) | 0.762 (0.00) | 0.762 (0.00) | 0.756 (0.00) | 0.750 (0.00) |

The reference is 0.972 for sentiment (2,048 pairs), 0.932 for truth (172), 0.881 for `truth_cities` (452), 0.915 for `truth_common` (1,296) and 0.738 for utility (2,048). The shuffled-label series is between 0.48 and 0.55 at every size.

- **Sentiment** is at 0.586 [0.557, 0.612] with 2 pairs (failure 0.50), has no failing draw from 16 pairs (0.785) and reaches 0.946 [0.943, 0.950] at 512, below the reference of 0.972; it gains 0.012 between 256 and 512 (0.934 to 0.946). **Utility** has no failing draw from 8 pairs (0.749) and the mean stays between 0.742 and 0.762 from 8 pairs on (0.750 [0.744, 0.755] at 512), above its reference of 0.738.
- **Truth** reads from 2 pairs: 0.836 [0.810, 0.858] with no failing draw (one draw in 20 fails at 4 pairs), 0.925 at 16 pairs and 0.932 [0.929, 0.936] at 128, at its reference of 0.932; the source is exhausted at 172 pairs. `truth_cities` has no failing draw at any size (0.830 at 2 pairs, 0.897 [0.889, 0.903] at 256, reference 0.881). `truth_common` is at 0.669 [0.614, 0.722] with 2 pairs (failure 0.35), has no failing draw from 16 pairs (0.786), and reads 0.904 [0.891, 0.917] at 128 and 0.900 [0.889, 0.910] at 512, below its reference of 0.915.
- The stored saturation size is 512 for sentiment (256 to 512), 16 for truth (16 to 128), 64 for `truth_cities` (16 to 64), 128 for `truth_common` (128 to 512) and 8 for utility (8 to 32). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set.

#### qwen3-4b-base

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.590 (0.65) | 0.634 (0.25) | 0.696 (0.05) | 0.754 (0.00) | 0.860 (0.00) | 0.913 (0.00) | 0.942 (0.00) | 0.957 (0.00) | 0.962 (0.00) |
| truth -> truth         | 0.972 (0.00) | 0.985 (0.00) | 0.993 (0.00) | 0.997 (0.00) | 0.999 (0.00) | 0.999 (0.00) | 1.000 (0.00) | -            | -            |
| truth_cities -> truth  | 0.984 (0.00) | 0.990 (0.00) | 0.987 (0.00) | 0.994 (0.00) | 0.998 (0.00) | 0.997 (0.00) | 1.000 (0.00) | 1.000 (0.00) | -            |
| truth_common -> truth  | 0.860 (0.10) | 0.906 (0.05) | 0.946 (0.00) | 0.964 (0.00) | 0.969 (0.00) | 0.968 (0.00) | 0.975 (0.00) | 0.981 (0.00) | 0.984 (0.00) |
| utility -> utility     | 0.665 (0.20) | 0.679 (0.15) | 0.732 (0.00) | 0.761 (0.00) | 0.770 (0.00) | 0.776 (0.00) | 0.772 (0.00) | 0.777 (0.00) | 0.777 (0.00) |

The reference is 0.972 for sentiment (2,048 pairs), 1.000 for truth (172), 1.000 for `truth_cities` (452), 1.000 for `truth_common` (1,296) and 0.781 for utility (2,048). The shuffled-label series is between 0.47 and 0.55 at every size.

- **Sentiment** is at 0.590 [0.558, 0.624] with 2 pairs (failure 0.65), has no failing draw from 16 pairs (0.754) and reaches 0.962 [0.959, 0.964] at 512, below the reference of 0.972; it gains 0.005 between 256 and 512 (0.957 to 0.962). **Utility** has no failing draw from 8 pairs (0.732) and the mean stays between 0.732 and 0.777 from 8 pairs on (0.777 [0.770, 0.783] at 512), below its reference of 0.781.
- **Truth** reads from 2 pairs: 0.972 [0.963, 0.981] with no failing draw at any size, 0.997 at 16 pairs and 1.000 at 128, at its reference of 1.000; the source is exhausted at 172 pairs. `truth_cities` has no failing draw at any size (0.984 [0.976, 0.991] at 2 pairs, 1.000 [1.000, 1.000] at 256, reference 1.000). `truth_common` is at 0.860 [0.793, 0.917] with 2 pairs (failure 0.10), has no failing draw from 8 pairs (0.946), and reads 0.975 [0.972, 0.979] at 128 and 0.984 [0.980, 0.988] at 512, below its reference of 1.000.
- The stored saturation size is 256 for sentiment (256 to 256), 8 for truth (4 to 16), 16 for `truth_cities` (2 to 16), 128 for `truth_common` (32 to 256) and 32 for utility (16 to 128). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set.

#### qwen3-4b

| Source -> target       | 2            | 4            | 8            | 16           | 32           | 64           | 128          | 256          | 512          |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sentiment -> sentiment | 0.535 (0.95) | 0.544 (1.00) | 0.579 (0.75) | 0.632 (0.20) | 0.685 (0.00) | 0.781 (0.00) | 0.843 (0.00) | 0.912 (0.00) | 0.939 (0.00) |
| truth -> truth         | 0.586 (0.55) | 0.659 (0.30) | 0.692 (0.10) | 0.800 (0.00) | 0.881 (0.00) | 0.931 (0.00) | 0.965 (0.00) | -            | -            |
| truth_cities -> truth  | 0.864 (0.00) | 0.851 (0.00) | 0.853 (0.00) | 0.892 (0.00) | 0.893 (0.00) | 0.893 (0.00) | 0.899 (0.00) | 0.911 (0.00) | -            |
| truth_common -> truth  | 0.742 (0.00) | 0.752 (0.10) | 0.808 (0.00) | 0.834 (0.00) | 0.861 (0.00) | 0.886 (0.00) | 0.924 (0.00) | 0.945 (0.00) | 0.950 (0.00) |
| utility -> utility     | 0.692 (0.15) | 0.710 (0.05) | 0.762 (0.00) | 0.756 (0.00) | 0.760 (0.00) | 0.760 (0.00) | 0.762 (0.00) | 0.766 (0.00) | 0.767 (0.00) |

The reference is 0.958 for sentiment (2,048 pairs), 1.000 for truth (172), 0.915 for `truth_cities` (452), 0.949 for `truth_common` (1,296) and 0.746 for utility (2,048). The shuffled-label series is between 0.48 and 0.54 at every size.

- **Sentiment** is at 0.535 [0.518, 0.553] with 2 pairs (failure 0.95), has no failing draw from 32 pairs (0.685) and reaches 0.939 [0.934, 0.943] at 512, below the reference of 0.958; it gains 0.026 between 256 and 512 (0.912 to 0.939). **Utility** has no failing draw from 8 pairs (0.762) and the mean stays between 0.756 and 0.767 from 8 pairs on (0.767 [0.760, 0.774] at 512), above its reference of 0.746.
- **Truth** is at 0.586 [0.558, 0.613] with 2 pairs (failure 0.55), 0.692 at 8 pairs (0.10), has no failing draw from 16 pairs (0.800) and reaches 0.965 [0.957, 0.974] at 128, 0.035 below its reference of 1.000; the source is exhausted at 172 pairs. `truth_cities` has no failing draw at any size (0.864 [0.843, 0.886] at 2 pairs, 0.911 [0.908, 0.914] at 256, reference 0.915). `truth_common` is at 0.742 [0.714, 0.768] with 2 pairs (no failing draw; one draw in 10 fails at 4 pairs), has no failing draw from 8 pairs (0.808), and reads 0.924 [0.909, 0.936] at 128 and 0.950 [0.947, 0.953] at 512, at its reference of 0.949.
- The stored saturation size is 512 for sentiment (512 to 512), 128 for truth (128 to 128), 256 for `truth_cities` (32 to 256), 256 for `truth_common` (256 to 512) and 8 for utility (8 to 256). For truth and `truth_cities` the largest size is the end of a small set (128 and 256 pairs), so their saturation sizes are measured against the end of the set.

### Direction finder, pairing and preprocessing (P4)

The design is that of [P4 on gpt2](#direction-finder-pairing-and-preprocessing-p4): the six finders `pca_diff` (the paper's finder and the P4 reference), `pca_raw`, `mean_diff`, `logistic`, `kmeans` and a `random` direction, the pairings `matched` (utility only), `label_contrastive` and `random`, and the preprocessing `center`, `unit` and `none`. Each cell is the report-split accuracy at the layer chosen on the select split for that cell, with the interval over the 10 seeds; the `random` direction is scored at its own selected layer like the others. The principal-component figures are at the layer of the reference cell (`pca_diff`, the pairing the task has, centered), which is the layer named in the text. Shuffled-label accuracy is given per run. "Above the random direction" means that the interval of the finder lies above the interval of the random direction in the same table row.

#### gpt2-medium

Shuffled-label accuracy is between 0.46 and 0.51.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.517 [0.504, 0.529] | 0.509 [0.499, 0.519] | 0.689 [0.667, 0.713] | 0.742 [0.723, 0.758] | 0.506 [0.490, 0.522] | 0.557 [0.515, 0.593] |
| truth (label_contrastive)     | 0.517 [0.510, 0.524] | 0.473 [0.453, 0.490] | 0.615 [0.600, 0.632] | 0.656 [0.625, 0.686] | 0.514 [0.508, 0.522] | 0.536 [0.490, 0.571] |
| utility (matched)             | 0.618 [0.559, 0.652] | 0.636 [0.627, 0.644] | 0.652 [0.643, 0.660] | 0.685 [0.665, 0.702] | 0.632 [0.622, 0.641] | 0.553 [0.520, 0.584] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.517    | 0.509   | 0.689     | 0.742    | 0.506  | 0.557  |
| sentiment | random            | 0.522    | 0.490   | 0.693     | 0.750    | 0.497  | 0.570  |
| truth     | label_contrastive | 0.517    | 0.473   | 0.615     | 0.656    | 0.514  | 0.536  |
| truth     | random            | 0.473    | 0.469   | 0.644     | 0.685    | 0.503  | 0.554  |
| utility   | matched           | 0.618    | 0.636   | 0.652     | 0.685    | 0.632  | 0.553  |
| utility   | label_contrastive | 0.585    | 0.580   | 0.658     | 0.658    | 0.579  | 0.553  |
| utility   | random            | 0.594    | 0.598   | 0.650     | 0.655    | 0.602  | 0.570  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.517 [0.504, 0.529] | 0.515 [0.503, 0.524] | 0.509 [0.499, 0.518] |
| sentiment | pca_raw   | 0.509 [0.499, 0.519] | 0.509 [0.497, 0.520] | 0.554 [0.552, 0.556] |
| sentiment | mean_diff | 0.689 [0.667, 0.713] | 0.692 [0.668, 0.717] | 0.689 [0.666, 0.713] |
| sentiment | logistic  | 0.742 [0.723, 0.758] | 0.741 [0.722, 0.757] | 0.691 [0.668, 0.714] |
| truth     | pca_diff  | 0.517 [0.510, 0.524] | 0.437 [0.417, 0.463] | 0.488 [0.473, 0.500] |
| truth     | pca_raw   | 0.473 [0.453, 0.490] | 0.469 [0.451, 0.486] | 0.605 [0.600, 0.610] |
| truth     | mean_diff | 0.615 [0.600, 0.632] | 0.612 [0.595, 0.632] | 0.615 [0.600, 0.631] |
| truth     | logistic  | 0.656 [0.625, 0.686] | 0.661 [0.634, 0.688] | 0.619 [0.602, 0.636] |
| utility   | pca_diff  | 0.618 [0.559, 0.652] | 0.641 [0.631, 0.649] | 0.615 [0.556, 0.649] |
| utility   | pca_raw   | 0.636 [0.627, 0.644] | 0.635 [0.626, 0.645] | 0.503 [0.495, 0.511] |
| utility   | mean_diff | 0.652 [0.643, 0.660] | 0.660 [0.654, 0.666] | 0.652 [0.644, 0.660] |
| utility   | logistic  | 0.685 [0.665, 0.702] | 0.684 [0.665, 0.699] | 0.655 [0.648, 0.663] |

- **The paper finder is not above the random direction.** `pca_diff` is at 0.517 [0.504, 0.529] for sentiment and 0.517 [0.510, 0.524] for truth, against 0.557 and 0.536 for the random direction, and at 0.618 [0.559, 0.652] for utility against 0.553 [0.520, 0.584]; on every task its interval overlaps that of the random direction. The first principal component explains 19 percent (sentiment, L12), 58 percent (truth, L1) and 26 percent (utility, matched, L24) of the variance of the pair differences, and the component chosen by cross-validation among the first eight reads 0.546 [0.527, 0.567] (sentiment), 0.532 [0.503, 0.564] (truth) and 0.569 [0.515, 0.619] (utility, matched).
- **The supervised finders read, and on utility so do `pca_raw` and `kmeans`.** `logistic` is the highest on all three tasks (0.742, 0.656, 0.685) and `mean_diff` is second (0.689, 0.615, 0.652), both above the random direction. On utility `pca_raw` (0.636 [0.627, 0.644]) and `kmeans` (0.632 [0.622, 0.641]) are above the random direction as well, which they are not on `gpt2` (0.555 and 0.557 against 0.526).
- **Pairing.** The best pairing for `logistic` is `random` for sentiment (0.750 against 0.742 for `label_contrastive`) and truth (0.685 against 0.656) and `matched` for utility (0.685 against 0.658 for `label_contrastive` and 0.655); the differences between pairings are at most 0.030. For `mean_diff` on truth the `random` pairing is higher (0.644 against 0.615).
- **Preprocessing.** `logistic` loses with no preprocessing on all three tasks (sentiment 0.742, 0.741, 0.691; truth 0.656, 0.661, 0.619 (overlapping intervals); utility 0.685, 0.684, 0.655), and `unit` is level with `center`. `pca_raw` without preprocessing is higher than centered on sentiment (0.554 against 0.509) and truth (0.605 against 0.473) and lower on utility (0.503 against 0.636). The two higher cells have intervals of width 0.004 and 0.010 ([0.552, 0.556] and [0.600, 0.610]): the ten readers give nearly the same value, so they are not read as reader results. `pca_diff` on truth with `unit` falls to 0.437 [0.417, 0.463], below chance.

#### gpt2-large

Shuffled-label accuracy is between 0.46 and 0.54.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.495 [0.486, 0.505] | 0.501 [0.498, 0.504] | 0.719 [0.681, 0.756] | 0.807 [0.792, 0.823] | 0.501 [0.498, 0.505] | 0.525 [0.486, 0.561] |
| truth (label_contrastive)     | 0.534 [0.527, 0.541] | 0.541 [0.537, 0.542] | 0.546 [0.525, 0.568] | 0.653 [0.629, 0.680] | 0.539 [0.532, 0.546] | 0.534 [0.507, 0.561] |
| utility (matched)             | 0.571 [0.535, 0.592] | 0.611 [0.602, 0.618] | 0.658 [0.636, 0.681] | 0.688 [0.666, 0.710] | 0.614 [0.606, 0.620] | 0.546 [0.513, 0.575] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.495    | 0.501   | 0.719     | 0.807    | 0.501  | 0.525  |
| sentiment | random            | 0.497    | 0.498   | 0.725     | 0.816    | 0.498  | 0.543  |
| truth     | label_contrastive | 0.534    | 0.541   | 0.546     | 0.653    | 0.539  | 0.534  |
| truth     | random            | 0.492    | 0.498   | 0.551     | 0.664    | 0.493  | 0.502  |
| utility   | matched           | 0.571    | 0.611   | 0.658     | 0.688    | 0.614  | 0.546  |
| utility   | label_contrastive | 0.547    | 0.549   | 0.623     | 0.642    | 0.547  | 0.532  |
| utility   | random            | 0.558    | 0.548   | 0.643     | 0.651    | 0.569  | 0.539  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.495 [0.486, 0.505] | 0.501 [0.496, 0.506] | 0.505 [0.500, 0.509] |
| sentiment | pca_raw   | 0.501 [0.498, 0.504] | 0.500 [0.500, 0.500] | 0.613 [0.555, 0.643] |
| sentiment | mean_diff | 0.719 [0.681, 0.756] | 0.721 [0.680, 0.761] | 0.719 [0.680, 0.755] |
| sentiment | logistic  | 0.807 [0.792, 0.823] | 0.809 [0.792, 0.828] | 0.724 [0.684, 0.764] |
| truth     | pca_diff  | 0.534 [0.527, 0.541] | 0.536 [0.529, 0.541] | 0.532 [0.525, 0.539] |
| truth     | pca_raw   | 0.541 [0.537, 0.542] | 0.539 [0.534, 0.542] | 0.492 [0.492, 0.492] |
| truth     | mean_diff | 0.546 [0.525, 0.568] | 0.541 [0.524, 0.561] | 0.546 [0.527, 0.566] |
| truth     | logistic  | 0.653 [0.629, 0.680] | 0.610 [0.583, 0.634] | 0.544 [0.524, 0.564] |
| utility   | pca_diff  | 0.571 [0.535, 0.592] | 0.557 [0.506, 0.593] | 0.573 [0.536, 0.594] |
| utility   | pca_raw   | 0.611 [0.602, 0.618] | 0.612 [0.604, 0.618] | 0.516 [0.509, 0.521] |
| utility   | mean_diff | 0.658 [0.636, 0.681] | 0.656 [0.635, 0.679] | 0.658 [0.636, 0.680] |
| utility   | logistic  | 0.688 [0.666, 0.710] | 0.689 [0.667, 0.711] | 0.673 [0.650, 0.695] |

- **The paper finder is not above the random direction.** `pca_diff` is at 0.495 [0.486, 0.505] for sentiment, 0.534 [0.527, 0.541] for truth and 0.571 [0.535, 0.592] for utility, against 0.525, 0.534 and 0.546 for the random direction, and its interval overlaps that of the random direction on every task. The first principal component explains 27 percent (sentiment, L18), 34 percent (truth, L2) and 22 percent (utility, matched, L36) of the variance of the pair differences, and the component chosen by cross-validation among the first eight reads 0.536 [0.496, 0.583] (sentiment), 0.553 [0.527, 0.588] (truth) and 0.599 [0.576, 0.623] (utility, matched); only the utility interval lies above that of the random direction (0.546 [0.513, 0.575]), by less than 0.001.
- **The supervised finders read, and on utility so do `pca_raw` and `kmeans`.** `logistic` is the highest on all three tasks (0.807, 0.653, 0.688) and `mean_diff` is second on sentiment (0.719) and utility (0.658). On truth only `logistic` is above the random direction: `mean_diff` is at 0.546 [0.525, 0.568] against 0.534 [0.507, 0.561]. On utility `pca_raw` (0.611 [0.602, 0.618]) and `kmeans` (0.614 [0.606, 0.620]) are above the random direction as well.
- **Pairing.** The best pairing for `logistic` is `random` for sentiment (0.816 against 0.807 for `label_contrastive`) and truth (0.664 against 0.653) and `matched` for utility (0.688 against 0.651 for `random` and 0.642); the differences are at most 0.046. On utility `pca_raw` and `kmeans` are higher with the `matched` pairing (0.611 and 0.614) than with the others (0.549 and 0.547 for `label_contrastive`, 0.548 and 0.569 for `random`).
- **Preprocessing.** `logistic` loses with no preprocessing on sentiment (0.807, 0.809, 0.724), truth (0.653, 0.610, 0.544) and, by 0.016 (overlapping intervals), utility (0.688, 0.689, 0.673); on truth `unit` is also lower than `center`. `pca_raw` is higher without preprocessing on sentiment (0.613 [0.555, 0.643] against 0.501) and lower on utility (0.516 against 0.611). Two `pca_raw` cells have intervals of zero width, sentiment with `unit` (0.500) and truth with no preprocessing (0.492): the ten readers give the same value, so they are not read as reader results.

#### gpt2-xl

Shuffled-label accuracy is between 0.46 and 0.54.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.507 [0.493, 0.521] | 0.506 [0.491, 0.520] | 0.664 [0.646, 0.684] | 0.776 [0.756, 0.794] | 0.500 [0.485, 0.516] | 0.545 [0.508, 0.586] |
| truth (label_contrastive)     | 0.488 [0.478, 0.500] | 0.486 [0.478, 0.495] | 0.514 [0.497, 0.536] | 0.656 [0.644, 0.666] | 0.480 [0.469, 0.488] | 0.503 [0.468, 0.537] |
| utility (matched)             | 0.576 [0.539, 0.598] | 0.597 [0.594, 0.600] | 0.640 [0.628, 0.653] | 0.686 [0.672, 0.701] | 0.582 [0.579, 0.586] | 0.525 [0.492, 0.552] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.507    | 0.506   | 0.664     | 0.776    | 0.500  | 0.545  |
| sentiment | random            | 0.497    | 0.513   | 0.678     | 0.771    | 0.503  | 0.541  |
| truth     | label_contrastive | 0.488    | 0.486   | 0.514     | 0.656    | 0.480  | 0.503  |
| truth     | random            | 0.478    | 0.485   | 0.602     | 0.649    | 0.454  | 0.541  |
| utility   | matched           | 0.576    | 0.597   | 0.640     | 0.686    | 0.582  | 0.525  |
| utility   | label_contrastive | 0.539    | 0.562   | 0.626     | 0.653    | 0.552  | 0.528  |
| utility   | random            | 0.563    | 0.566   | 0.632     | 0.653    | 0.572  | 0.539  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.507 [0.493, 0.521] | 0.507 [0.495, 0.520] | 0.506 [0.488, 0.522] |
| sentiment | pca_raw   | 0.506 [0.491, 0.520] | 0.498 [0.493, 0.504] | 0.552 [0.501, 0.586] |
| sentiment | mean_diff | 0.664 [0.646, 0.684] | 0.683 [0.657, 0.714] | 0.664 [0.645, 0.683] |
| sentiment | logistic  | 0.776 [0.756, 0.794] | 0.779 [0.758, 0.797] | 0.668 [0.649, 0.692] |
| truth     | pca_diff  | 0.488 [0.478, 0.500] | 0.485 [0.475, 0.497] | 0.488 [0.478, 0.498] |
| truth     | pca_raw   | 0.486 [0.478, 0.495] | 0.488 [0.481, 0.495] | 0.471 [0.461, 0.485] |
| truth     | mean_diff | 0.514 [0.497, 0.536] | 0.561 [0.544, 0.578] | 0.514 [0.497, 0.532] |
| truth     | logistic  | 0.656 [0.644, 0.666] | 0.659 [0.646, 0.671] | 0.563 [0.547, 0.578] |
| utility   | pca_diff  | 0.576 [0.539, 0.598] | 0.577 [0.534, 0.600] | 0.577 [0.537, 0.599] |
| utility   | pca_raw   | 0.597 [0.594, 0.600] | 0.596 [0.593, 0.600] | 0.563 [0.560, 0.566] |
| utility   | mean_diff | 0.640 [0.628, 0.653] | 0.641 [0.628, 0.657] | 0.640 [0.628, 0.654] |
| utility   | logistic  | 0.686 [0.672, 0.701] | 0.690 [0.677, 0.700] | 0.648 [0.636, 0.661] |

- **The paper finder is not above the random direction.** `pca_diff` is at 0.507 [0.493, 0.521] for sentiment, 0.488 [0.478, 0.500] for truth and 0.576 [0.539, 0.598] for utility, against 0.545, 0.503 and 0.525 for the random direction, and its interval overlaps that of the random direction on every task. The first principal component explains 25 percent (sentiment, L45), 20 percent (truth, L2) and 32 percent (utility, matched, L26) of the variance of the pair differences, and the component chosen by cross-validation among the first eight reads 0.595 [0.546, 0.646] (sentiment), 0.493 [0.481, 0.505] (truth) and 0.609 [0.580, 0.637] (utility, matched); only the utility value has an interval above that of the random direction.
- **The supervised finders read.** `logistic` is the highest on all three tasks (0.776, 0.656, 0.686) and `mean_diff` is second on sentiment (0.664) and utility (0.640). On truth `mean_diff` is at 0.514 [0.497, 0.536], an interval that overlaps the random direction (0.503 [0.468, 0.537]). On utility `pca_raw` (0.597 [0.594, 0.600]) and `kmeans` (0.582 [0.579, 0.586]) are above the random direction.
- **Pairing.** The best pairing for `logistic` is `label_contrastive` for sentiment (0.776 against 0.771 for `random`) and truth (0.656 against 0.649) and `matched` for utility (0.686 against 0.653 for both others); the differences are at most 0.034, and 0.007 or less on sentiment and truth. For `mean_diff` on truth the `random` pairing is higher (0.602 against 0.514).
- **Preprocessing.** `logistic` loses with no preprocessing on all three tasks (sentiment 0.776, 0.779, 0.668; truth 0.656, 0.659, 0.563; utility 0.686, 0.690, 0.648), and `unit` is level with `center`. `mean_diff` on truth is higher with `unit` (0.561 [0.544, 0.578] against 0.514 [0.497, 0.536]). No cell has an interval of zero width.

#### qwen3-0.6b-base

Shuffled-label accuracy is between 0.45 and 0.51.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.665 [0.613, 0.722] | 0.557 [0.509, 0.601] | 0.937 [0.927, 0.947] | 0.952 [0.946, 0.958] | 0.576 [0.538, 0.615] | 0.658 [0.619, 0.700] |
| truth (label_contrastive)     | 0.588 [0.573, 0.607] | 0.547 [0.517, 0.578] | 0.688 [0.664, 0.712] | 0.764 [0.749, 0.778] | 0.532 [0.505, 0.559] | 0.549 [0.524, 0.578] |
| utility (matched)             | 0.727 [0.722, 0.731] | 0.737 [0.727, 0.745] | 0.734 [0.727, 0.741] | 0.709 [0.698, 0.720] | 0.740 [0.733, 0.747] | 0.604 [0.567, 0.638] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.665    | 0.557   | 0.937     | 0.952    | 0.576  | 0.658  |
| sentiment | random            | 0.550    | 0.539   | 0.935     | 0.950    | 0.541  | 0.658  |
| truth     | label_contrastive | 0.588    | 0.547   | 0.688     | 0.764    | 0.532  | 0.549  |
| truth     | random            | 0.568    | 0.554   | 0.705     | 0.781    | 0.541  | 0.517  |
| utility   | matched           | 0.727    | 0.737   | 0.734     | 0.709    | 0.740  | 0.604  |
| utility   | label_contrastive | 0.736    | 0.741   | 0.731     | 0.709    | 0.716  | 0.636  |
| utility   | random            | 0.723    | 0.726   | 0.732     | 0.691    | 0.729  | 0.627  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.665 [0.613, 0.722] | 0.713 [0.658, 0.770] | 0.667 [0.611, 0.723] |
| sentiment | pca_raw   | 0.557 [0.509, 0.601] | 0.556 [0.512, 0.597] | 0.769 [0.765, 0.773] |
| sentiment | mean_diff | 0.937 [0.927, 0.947] | 0.940 [0.930, 0.949] | 0.937 [0.928, 0.947] |
| sentiment | logistic  | 0.952 [0.946, 0.958] | 0.953 [0.947, 0.959] | 0.940 [0.931, 0.949] |
| truth     | pca_diff  | 0.588 [0.573, 0.607] | 0.563 [0.551, 0.575] | 0.583 [0.569, 0.597] |
| truth     | pca_raw   | 0.547 [0.517, 0.578] | 0.512 [0.493, 0.532] | 0.583 [0.578, 0.588] |
| truth     | mean_diff | 0.688 [0.664, 0.712] | 0.685 [0.658, 0.712] | 0.688 [0.663, 0.715] |
| truth     | logistic  | 0.764 [0.749, 0.778] | 0.768 [0.753, 0.783] | 0.697 [0.675, 0.720] |
| utility   | pca_diff  | 0.727 [0.722, 0.731] | 0.725 [0.717, 0.732] | 0.727 [0.723, 0.732] |
| utility   | pca_raw   | 0.737 [0.727, 0.745] | 0.736 [0.727, 0.743] | 0.597 [0.589, 0.605] |
| utility   | mean_diff | 0.734 [0.727, 0.741] | 0.733 [0.727, 0.741] | 0.734 [0.727, 0.741] |
| utility   | logistic  | 0.709 [0.698, 0.720] | 0.708 [0.696, 0.718] | 0.733 [0.727, 0.739] |

- **The paper finder is not above the random direction on sentiment and truth, and reads on utility.** The random direction itself reads 0.658 [0.619, 0.700] on sentiment here, and `pca_diff` is at 0.665 [0.613, 0.722]; on truth `pca_diff` is at 0.588 [0.573, 0.607] against 0.549 [0.524, 0.578], with overlapping intervals; on utility it is at 0.727 [0.722, 0.731] against 0.604 [0.567, 0.638]. `pca_raw` and `kmeans` are below the random direction on sentiment (0.557 and 0.576, intervals below that of the random direction). The first principal component explains 22 percent (sentiment, L20), 16 percent (truth, L28) and 39 percent (utility, matched, L25) of the variance of the pair differences. The component chosen by cross-validation among the first eight reads 0.818 [0.744, 0.873] for sentiment against 0.665 for the first component, and the component chosen most often across the 10 seeds is the second; the other values are 0.583 [0.561, 0.605] (truth) and 0.714 [0.688, 0.730] (utility, matched).
- **The supervised finders lead on sentiment and truth, and trail on utility.** `logistic` is the highest on sentiment (0.952 [0.946, 0.958]) and truth (0.764) and `mean_diff` is second (0.937, 0.688). On utility the order is reversed: `kmeans` (0.740), `pca_raw` (0.737), `mean_diff` (0.734) and `pca_diff` (0.727) have intervals above that of `logistic` (0.709 [0.698, 0.720]), and all five are above the random direction.
- **Pairing.** The differences between pairings for `logistic` are at most 0.018 (sentiment 0.952 and 0.950, truth 0.764 and 0.781 for `random`, utility 0.709, 0.709 and 0.691). `pca_diff` on sentiment depends on the pairing more than any other finder and task: 0.665 with `label_contrastive` and 0.550 with `random`.
- **Preprocessing.** `logistic` loses with no preprocessing on sentiment (0.952, 0.953, 0.940; overlapping intervals) and truth (0.764, 0.768, 0.697) and gains on utility (0.709, 0.708, 0.733). `pca_raw` on sentiment is 0.769 [0.765, 0.773] without preprocessing against 0.557 centered, an interval of width 0.008 (the ten readers give nearly the same value, so it is not read as a reader result), and on utility it falls to 0.597 without preprocessing (0.737 centered). `pca_diff` on sentiment is 0.713 [0.658, 0.770] with `unit` and 0.665 [0.613, 0.722] centered, with overlapping intervals.

#### qwen3-0.6b

Shuffled-label accuracy is between 0.47 and 0.53.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.564 [0.541, 0.588] | 0.574 [0.549, 0.597] | 0.750 [0.726, 0.770] | 0.860 [0.844, 0.875] | 0.551 [0.481, 0.611] | 0.594 [0.557, 0.637] |
| truth (label_contrastive)     | 0.444 [0.432, 0.456] | 0.434 [0.420, 0.446] | 0.600 [0.578, 0.629] | 0.744 [0.720, 0.770] | 0.449 [0.441, 0.458] | 0.558 [0.514, 0.605] |
| utility (matched)             | 0.718 [0.713, 0.724] | 0.720 [0.716, 0.722] | 0.720 [0.715, 0.725] | 0.711 [0.696, 0.725] | 0.712 [0.705, 0.718] | 0.614 [0.560, 0.662] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.564    | 0.574   | 0.750     | 0.860    | 0.551  | 0.594  |
| sentiment | random            | 0.576    | 0.568   | 0.783     | 0.873    | 0.550  | 0.588  |
| truth     | label_contrastive | 0.444    | 0.434   | 0.600     | 0.744    | 0.449  | 0.558  |
| truth     | random            | 0.456    | 0.498   | 0.531     | 0.786    | 0.551  | 0.561  |
| utility   | matched           | 0.718    | 0.720   | 0.720     | 0.711    | 0.712  | 0.614  |
| utility   | label_contrastive | 0.720    | 0.717   | 0.729     | 0.716    | 0.722  | 0.614  |
| utility   | random            | 0.717    | 0.718   | 0.720     | 0.696    | 0.722  | 0.614  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.564 [0.541, 0.588] | 0.553 [0.498, 0.603] | 0.569 [0.547, 0.593] |
| sentiment | pca_raw   | 0.574 [0.549, 0.597] | 0.574 [0.550, 0.596] | 0.648 [0.569, 0.690] |
| sentiment | mean_diff | 0.750 [0.726, 0.770] | 0.751 [0.728, 0.773] | 0.750 [0.727, 0.771] |
| sentiment | logistic  | 0.860 [0.844, 0.875] | 0.860 [0.842, 0.876] | 0.759 [0.737, 0.781] |
| truth     | pca_diff  | 0.444 [0.432, 0.456] | 0.483 [0.466, 0.502] | 0.451 [0.437, 0.464] |
| truth     | pca_raw   | 0.434 [0.420, 0.446] | 0.463 [0.454, 0.471] | 0.690 [0.683, 0.697] |
| truth     | mean_diff | 0.600 [0.578, 0.629] | 0.519 [0.508, 0.527] | 0.600 [0.578, 0.627] |
| truth     | logistic  | 0.744 [0.720, 0.770] | 0.744 [0.720, 0.771] | 0.608 [0.585, 0.637] |
| utility   | pca_diff  | 0.718 [0.713, 0.724] | 0.720 [0.714, 0.727] | 0.719 [0.713, 0.725] |
| utility   | pca_raw   | 0.720 [0.716, 0.722] | 0.716 [0.714, 0.718] | 0.688 [0.684, 0.692] |
| utility   | mean_diff | 0.720 [0.715, 0.725] | 0.722 [0.716, 0.727] | 0.720 [0.714, 0.725] |
| utility   | logistic  | 0.711 [0.696, 0.725] | 0.711 [0.694, 0.726] | 0.725 [0.717, 0.731] |

- **The paper finder is not above the random direction on sentiment and truth, and reads on utility.** `pca_diff` is at 0.564 [0.541, 0.588] on sentiment against 0.594 [0.557, 0.637] for the random direction, and at 0.444 [0.432, 0.456] on truth, below chance and below the random direction (0.558 [0.514, 0.605]); `pca_raw` (0.434) and `kmeans` (0.449) are also below chance on truth. On utility `pca_diff` is at 0.718 [0.713, 0.724] against 0.614 [0.560, 0.662]. The first principal component explains 30 percent (sentiment, L10), 28 percent (truth, L2) and 44 percent (utility, matched, L28) of the variance of the pair differences, and the component chosen by cross-validation among the first eight reads 0.556 [0.519, 0.591] (sentiment), 0.471 [0.444, 0.512] (truth) and 0.718 [0.713, 0.724] (utility, matched).
- **The supervised finders lead on sentiment and truth, and the finders are level on utility.** `logistic` is the highest on sentiment (0.860) and truth (0.744) and `mean_diff` is second on sentiment (0.750); on truth `mean_diff` is at 0.600 [0.578, 0.629], an interval that overlaps the random direction. On utility the five finders read between 0.711 and 0.720, all above the random direction, and `logistic` has the lowest point value (0.711).
- **Pairing.** The best pairing for `logistic` is `random` for sentiment (0.873 against 0.860 for `label_contrastive`) and truth (0.786 against 0.744) and `label_contrastive` for utility (0.716 against 0.711 for `matched` and 0.696); the differences are at most 0.042. For `mean_diff` on truth `label_contrastive` is higher (0.600 against 0.531), the reverse of `gpt2`. `kmeans` on truth depends on the pairing more than any other finder and task: 0.449 with `label_contrastive` and 0.551 with `random`.
- **Preprocessing.** `logistic` loses with no preprocessing on sentiment (0.860, 0.860, 0.759) and truth (0.744, 0.744, 0.608) and gains on utility (0.711, 0.711, 0.725; overlapping intervals). `pca_raw` on truth is 0.690 [0.683, 0.697] without preprocessing against 0.434 centered, an interval of width 0.014 (the ten readers give nearly the same value, so it is not read as a reader result). `mean_diff` on truth is lower with `unit` (0.519 [0.508, 0.527]) than centered (0.600).

#### qwen3-1.7b-base

Shuffled-label accuracy is between 0.45 and 0.52.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.846 [0.832, 0.857] | 0.837 [0.827, 0.847] | 0.893 [0.881, 0.905] | 0.943 [0.937, 0.948] | 0.847 [0.839, 0.855] | 0.676 [0.632, 0.720] |
| truth (label_contrastive)     | 0.959 [0.949, 0.966] | 0.944 [0.932, 0.956] | 0.963 [0.956, 0.966] | 0.971 [0.964, 0.978] | 0.951 [0.941, 0.961] | 0.724 [0.646, 0.795] |
| utility (matched)             | 0.799 [0.793, 0.805] | 0.793 [0.782, 0.804] | 0.791 [0.787, 0.796] | 0.761 [0.748, 0.774] | 0.788 [0.783, 0.793] | 0.649 [0.620, 0.684] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.846    | 0.837   | 0.893     | 0.943    | 0.847  | 0.676  |
| sentiment | random            | 0.836    | 0.834   | 0.900     | 0.943    | 0.840  | 0.697  |
| truth     | label_contrastive | 0.959    | 0.944   | 0.963     | 0.971    | 0.951  | 0.724  |
| truth     | random            | 0.956    | 0.958   | 0.966     | 0.971    | 0.929  | 0.724  |
| utility   | matched           | 0.799    | 0.793   | 0.791     | 0.761    | 0.788  | 0.649  |
| utility   | label_contrastive | 0.797    | 0.800   | 0.788     | 0.758    | 0.789  | 0.649  |
| utility   | random            | 0.792    | 0.786   | 0.783     | 0.735    | 0.789  | 0.612  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.846 [0.832, 0.857] | 0.857 [0.844, 0.868] | 0.857 [0.848, 0.865] |
| sentiment | pca_raw   | 0.837 [0.827, 0.847] | 0.843 [0.836, 0.851] | 0.649 [0.644, 0.652] |
| sentiment | mean_diff | 0.893 [0.881, 0.905] | 0.894 [0.881, 0.906] | 0.893 [0.881, 0.905] |
| sentiment | logistic  | 0.943 [0.937, 0.948] | 0.944 [0.939, 0.949] | 0.901 [0.893, 0.911] |
| truth     | pca_diff  | 0.959 [0.949, 0.966] | 0.963 [0.958, 0.966] | 0.959 [0.949, 0.966] |
| truth     | pca_raw   | 0.944 [0.932, 0.956] | 0.946 [0.932, 0.959] | 0.612 [0.610, 0.615] |
| truth     | mean_diff | 0.963 [0.956, 0.966] | 0.963 [0.956, 0.966] | 0.963 [0.956, 0.966] |
| truth     | logistic  | 0.971 [0.964, 0.978] | 0.971 [0.964, 0.978] | 0.964 [0.961, 0.966] |
| utility   | pca_diff  | 0.799 [0.793, 0.805] | 0.795 [0.789, 0.801] | 0.802 [0.795, 0.808] |
| utility   | pca_raw   | 0.793 [0.782, 0.804] | 0.792 [0.780, 0.804] | 0.612 [0.599, 0.625] |
| utility   | mean_diff | 0.791 [0.787, 0.796] | 0.806 [0.800, 0.812] | 0.791 [0.787, 0.796] |
| utility   | logistic  | 0.761 [0.748, 0.774] | 0.762 [0.749, 0.776] | 0.797 [0.792, 0.801] |

- **The paper finder reads on all three tasks.** `pca_diff` is at 0.846 [0.832, 0.857] for sentiment, 0.959 [0.949, 0.966] for truth and 0.799 [0.793, 0.805] for utility, against 0.676 [0.632, 0.720], 0.724 [0.646, 0.795] and 0.649 [0.620, 0.684] for the random direction, and its interval lies above that of the random direction on every task. The first principal component explains 25 percent (sentiment, L19), 45 percent (truth, L22) and 46 percent (utility, matched, L26) of the variance of the pair differences. The component chosen by cross-validation among the first eight is, most often across the 10 seeds, the first on all three tasks, and it reads 0.846 [0.833, 0.857] (sentiment), 0.959 [0.949, 0.966] (truth) and 0.799 [0.793, 0.805] (utility, matched), the same value as the first component to three decimals.
- **All five finders are above the random direction on every task, and the order depends on the task.** On sentiment the order is `logistic` (0.943 [0.937, 0.948]), `mean_diff` (0.893 [0.881, 0.905]), `kmeans` (0.847), `pca_diff` (0.846) and `pca_raw` (0.837), and the interval of each of the first two lies above those of all that follow. On truth the five read between 0.944 and 0.971: `logistic` is at 0.971 [0.964, 0.978], with an interval above those of `kmeans` (0.951) and `pca_raw` (0.944) and overlapping those of `mean_diff` (0.963 [0.956, 0.966]) and `pca_diff`. On utility `logistic` has the lowest point value of the five, as on the other three Qwen3 checkpoints, and here its interval is below those of the other four: `pca_diff` (0.799), `pca_raw` (0.793), `mean_diff` (0.791) and `kmeans` (0.788) against 0.761 [0.748, 0.774].
- **Pairing.** The differences between pairings for `logistic` are at most 0.026 (sentiment 0.943 with both `label_contrastive` and `random`, truth 0.971 with both, utility 0.761, 0.758 and 0.735 for `matched`, `label_contrastive` and `random`). None of the other four finders changes by more than that: the next largest difference is `kmeans` on truth (0.951 with `label_contrastive` and 0.929 with `random`, 0.022). On utility `pca_raw` is at 0.793 with `matched` and 0.800 with `label_contrastive`.
- **Preprocessing.** `logistic` is lower with no preprocessing on sentiment (0.943, 0.944, 0.901 [0.893, 0.911]) and, with overlapping intervals, on truth (0.971, 0.971, 0.964 [0.961, 0.966]), and higher on utility (0.761, 0.762, 0.797 [0.792, 0.801]). `pca_raw` is lower without preprocessing on all three tasks (sentiment 0.649 [0.644, 0.652] against 0.837, truth 0.612 [0.610, 0.615] against 0.944, utility 0.612 [0.599, 0.625] against 0.793); the first two intervals have widths 0.008 and 0.005 (the ten readers give nearly the same value, so they are not read as reader results). `pca_diff` changes by at most 0.011 across the three on every task, and `mean_diff` on utility is higher with `unit` (0.806 [0.800, 0.812] against 0.791 [0.787, 0.796]). Three cells with the `random` pairing on truth have intervals of zero width, at 0.966 in all ten seeds: `mean_diff` centered and with no preprocessing, and `logistic` with no preprocessing.

#### qwen3-1.7b

Shuffled-label accuracy is between 0.46 and 0.52.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.531 [0.504, 0.557] | 0.521 [0.493, 0.546] | 0.750 [0.729, 0.771] | 0.877 [0.865, 0.889] | 0.523 [0.493, 0.547] | 0.585 [0.547, 0.621] |
| truth (label_contrastive)     | 0.869 [0.842, 0.893] | 0.781 [0.759, 0.802] | 0.912 [0.903, 0.924] | 0.932 [0.920, 0.944] | 0.808 [0.771, 0.846] | 0.639 [0.573, 0.715] |
| utility (matched)             | 0.780 [0.777, 0.784] | 0.775 [0.772, 0.779] | 0.784 [0.782, 0.786] | 0.760 [0.750, 0.771] | 0.779 [0.775, 0.783] | 0.627 [0.566, 0.682] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.531    | 0.521   | 0.750     | 0.877    | 0.523  | 0.585  |
| sentiment | random            | 0.507    | 0.513   | 0.815     | 0.886    | 0.504  | 0.580  |
| truth     | label_contrastive | 0.869    | 0.781   | 0.912     | 0.932    | 0.808  | 0.639  |
| truth     | random            | 0.763    | 0.792   | 0.915     | 0.914    | 0.827  | 0.636  |
| utility   | matched           | 0.780    | 0.775   | 0.784     | 0.760    | 0.779  | 0.627  |
| utility   | label_contrastive | 0.780    | 0.774   | 0.780     | 0.755    | 0.777  | 0.627  |
| utility   | random            | 0.774    | 0.777   | 0.772     | 0.739    | 0.778  | 0.616  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.531 [0.504, 0.557] | 0.529 [0.518, 0.540] | 0.532 [0.505, 0.553] |
| sentiment | pca_raw   | 0.521 [0.493, 0.546] | 0.521 [0.495, 0.544] | 0.577 [0.539, 0.596] |
| sentiment | mean_diff | 0.750 [0.729, 0.771] | 0.766 [0.736, 0.795] | 0.750 [0.728, 0.771] |
| sentiment | logistic  | 0.877 [0.865, 0.889] | 0.876 [0.861, 0.889] | 0.791 [0.766, 0.813] |
| truth     | pca_diff  | 0.869 [0.842, 0.893] | 0.876 [0.856, 0.893] | 0.871 [0.846, 0.893] |
| truth     | pca_raw   | 0.781 [0.759, 0.802] | 0.792 [0.775, 0.808] | 0.612 [0.610, 0.615] |
| truth     | mean_diff | 0.912 [0.903, 0.924] | 0.910 [0.903, 0.919] | 0.912 [0.903, 0.922] |
| truth     | logistic  | 0.932 [0.920, 0.944] | 0.934 [0.920, 0.947] | 0.912 [0.902, 0.922] |
| utility   | pca_diff  | 0.780 [0.777, 0.784] | 0.783 [0.779, 0.787] | 0.780 [0.777, 0.784] |
| utility   | pca_raw   | 0.775 [0.772, 0.779] | 0.775 [0.772, 0.779] | 0.639 [0.627, 0.650] |
| utility   | mean_diff | 0.784 [0.782, 0.786] | 0.784 [0.782, 0.786] | 0.784 [0.782, 0.786] |
| utility   | logistic  | 0.760 [0.750, 0.771] | 0.761 [0.751, 0.772] | 0.788 [0.780, 0.796] |

- **The paper finder reads truth and utility, and not sentiment.** On sentiment `pca_diff` is at 0.531 [0.504, 0.557] against 0.585 [0.547, 0.621] for the random direction, with overlapping intervals. On truth it is at 0.869 [0.842, 0.893] against 0.639 [0.573, 0.715], and on utility at 0.780 [0.777, 0.784] against 0.627 [0.566, 0.682]. The first principal component explains 33 percent (sentiment), 30 percent (truth) and 39 percent (utility, matched) of the variance of the pair differences, at L20 for all three. The component chosen by cross-validation among the first eight reads 0.712 [0.673, 0.744] for sentiment, against 0.531 for the first component, and the component chosen most often across the 10 seeds is the second; for truth and utility it is the first component (0.869 and 0.780).
- **The finders are ordered differently by task.** On sentiment only `logistic` (0.877) and `mean_diff` (0.750) are above the random direction. On truth all five finders are, in the order `logistic` 0.932 [0.920, 0.944], `mean_diff` 0.912, `pca_diff` 0.869, `kmeans` 0.808 and `pca_raw` 0.781; the interval of `pca_diff` lies below that of `logistic`. On utility `logistic` is the lowest of the five, as on `qwen3-0.6b-base`: `mean_diff` (0.784), `pca_diff` (0.780), `kmeans` (0.779) and `pca_raw` (0.775) have intervals above that of `logistic` (0.760 [0.750, 0.771]), and all five are above the random direction.
- **Pairing.** The differences between pairings for `logistic` are at most 0.021 (sentiment 0.877 and 0.886 for `random`, truth 0.932 and 0.914, utility 0.760, 0.755 and 0.739). `pca_diff` on truth depends on the pairing more than any other finder and task: 0.869 with `label_contrastive` and 0.763 with `random`. `mean_diff` on sentiment is 0.750 with `label_contrastive` and 0.815 with `random`.
- **Preprocessing.** `logistic` loses with no preprocessing on sentiment (0.877, 0.876, 0.791) and truth (0.932, 0.934, 0.912; overlapping intervals) and gains on utility (0.760, 0.761, 0.788 [0.780, 0.796]). `pca_diff` changes by at most 0.007 across the three on every task. `pca_raw` falls without preprocessing on truth (0.612 [0.610, 0.615] against 0.781, an interval of width 0.005) and on utility (0.639 against 0.775).

#### qwen3-4b-base

Shuffled-label accuracy is between 0.47 and 0.51.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.529 [0.490, 0.564] | 0.539 [0.506, 0.563] | 0.792 [0.774, 0.808] | 0.900 [0.886, 0.913] | 0.528 [0.492, 0.562] | 0.577 [0.540, 0.614] |
| truth (label_contrastive)     | 0.990 [0.985, 0.995] | 0.980 [0.971, 0.990] | 0.995 [0.990, 1.000] | 0.998 [0.995, 1.000] | 0.995 [0.990, 1.000] | 0.859 [0.783, 0.924] |
| utility (matched)             | 0.674 [0.669, 0.679] | 0.724 [0.701, 0.750] | 0.786 [0.775, 0.796] | 0.780 [0.763, 0.795] | 0.745 [0.722, 0.767] | 0.581 [0.545, 0.618] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.529    | 0.539   | 0.792     | 0.900    | 0.528  | 0.577  |
| sentiment | random            | 0.500    | 0.484   | 0.807     | 0.923    | 0.497  | 0.586  |
| truth     | label_contrastive | 0.990    | 0.980   | 0.995     | 0.998    | 0.995  | 0.859  |
| truth     | random            | 0.976    | 0.980   | 0.997     | 1.000    | 0.993  | 0.859  |
| utility   | matched           | 0.674    | 0.724   | 0.786     | 0.780    | 0.745  | 0.581  |
| utility   | label_contrastive | 0.756    | 0.732   | 0.797     | 0.759    | 0.768  | 0.595  |
| utility   | random            | 0.731    | 0.738   | 0.780     | 0.742    | 0.754  | 0.582  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.529 [0.490, 0.564] | 0.536 [0.497, 0.571] | 0.529 [0.488, 0.562] |
| sentiment | pca_raw   | 0.539 [0.506, 0.563] | 0.537 [0.503, 0.561] | 0.514 [0.500, 0.525] |
| sentiment | mean_diff | 0.792 [0.774, 0.808] | 0.792 [0.773, 0.808] | 0.792 [0.774, 0.808] |
| sentiment | logistic  | 0.900 [0.886, 0.913] | 0.901 [0.888, 0.914] | 0.808 [0.790, 0.824] |
| truth     | pca_diff  | 0.990 [0.985, 0.995] | 0.993 [0.988, 0.998] | 0.990 [0.985, 0.995] |
| truth     | pca_raw   | 0.980 [0.971, 0.990] | 0.983 [0.973, 0.993] | 0.919 [0.915, 0.925] |
| truth     | mean_diff | 0.995 [0.990, 1.000] | 0.995 [0.990, 1.000] | 0.995 [0.990, 1.000] |
| truth     | logistic  | 0.998 [0.995, 1.000] | 0.998 [0.995, 1.000] | 0.997 [0.992, 1.000] |
| utility   | pca_diff  | 0.674 [0.669, 0.679] | 0.675 [0.670, 0.680] | 0.675 [0.671, 0.680] |
| utility   | pca_raw   | 0.724 [0.701, 0.750] | 0.726 [0.703, 0.753] | 0.620 [0.616, 0.624] |
| utility   | mean_diff | 0.786 [0.775, 0.796] | 0.787 [0.778, 0.796] | 0.786 [0.774, 0.795] |
| utility   | logistic  | 0.780 [0.763, 0.795] | 0.779 [0.763, 0.794] | 0.789 [0.777, 0.799] |

- **The paper finder reads truth and utility, and not sentiment.** On sentiment `pca_diff` is at 0.529 [0.490, 0.564] against 0.577 [0.540, 0.614] for the random direction, with overlapping intervals. On truth it is at 0.990 [0.985, 0.995] against 0.859 [0.783, 0.924], and on utility at 0.674 [0.669, 0.679] against 0.581 [0.545, 0.618]. The first principal component explains 23 percent (sentiment, at L36), 59 percent (truth, at L23) and 35 percent (utility, matched, at L32) of the variance of the pair differences. The component chosen by cross-validation among the first eight reads 0.608 [0.571, 0.645] for sentiment, against 0.529 for the first component, with an interval that still overlaps that of the random direction, and the component chosen most often across the 10 seeds is the third; for truth and utility it is the first component (0.990 and 0.659 [0.636, 0.677]).
- **The finders are ordered differently by task.** On sentiment only `logistic` (0.900) and `mean_diff` (0.792) are above the random direction. On truth all five finders are, within 0.018 of each other: `logistic` 0.998 [0.995, 1.000], `mean_diff` 0.995, `kmeans` 0.995, `pca_diff` 0.990 and `pca_raw` 0.980; the random direction itself reads 0.859. On utility `pca_diff` is the lowest of the five (0.674) and its interval lies below those of the other four: `mean_diff` 0.786 [0.775, 0.796], `logistic` 0.780 [0.763, 0.795], `kmeans` 0.745 [0.722, 0.767] and `pca_raw` 0.724 [0.701, 0.750]; all five are above the random direction. Unlike on the smaller Qwen3 checkpoints, `logistic` is not the lowest finder on utility here.
- **Pairing.** The differences between pairings for `logistic` are at most 0.038 (sentiment 0.900 and 0.923 for `random`, truth 0.998 and 1.000, utility 0.780, 0.759 and 0.742). `pca_diff` on utility depends on the pairing more than any other finder and task: 0.674 with `matched`, 0.756 with `label_contrastive` and 0.731 with `random`. `pca_raw` on sentiment is 0.539 with `label_contrastive` and 0.484 with `random`.
- **Preprocessing.** `logistic` loses with no preprocessing on sentiment (0.900, 0.901, 0.808), is level on truth (0.998, 0.998, 0.997) and gains on utility (0.780, 0.779, 0.789 [0.777, 0.799]; overlapping intervals). `pca_diff` changes by at most 0.007 across the three on every task. `pca_raw` falls without preprocessing on truth (0.919 [0.915, 0.925] against 0.980) and on utility (0.620 [0.616, 0.624] against 0.724).

#### qwen3-4b

Shuffled-label accuracy is between 0.47 and 0.52.

Centered preprocessing, the pairing the task has:

| Task (pairing)                | pca_diff             | pca_raw              | mean_diff            | logistic             | kmeans               | random               |
| ----------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| sentiment (label_contrastive) | 0.511 [0.467, 0.553] | 0.505 [0.485, 0.524] | 0.614 [0.586, 0.645] | 0.757 [0.730, 0.791] | 0.509 [0.486, 0.531] | 0.525 [0.490, 0.560] |
| truth (label_contrastive)     | 0.931 [0.922, 0.939] | 0.888 [0.868, 0.910] | 0.941 [0.932, 0.949] | 0.969 [0.959, 0.978] | 0.934 [0.922, 0.944] | 0.722 [0.642, 0.795] |
| utility (matched)             | 0.774 [0.770, 0.778] | 0.776 [0.772, 0.780] | 0.782 [0.777, 0.787] | 0.785 [0.773, 0.797] | 0.774 [0.770, 0.779] | 0.655 [0.615, 0.693] |

By pairing, centered preprocessing:

| Task      | Pairing           | pca_diff | pca_raw | mean_diff | logistic | kmeans | random |
| --------- | ----------------- | -------- | ------- | --------- | -------- | ------ | ------ |
| sentiment | label_contrastive | 0.511    | 0.505   | 0.614     | 0.757    | 0.509  | 0.525  |
| sentiment | random            | 0.486    | 0.485   | 0.591     | 0.771    | 0.497  | 0.561  |
| truth     | label_contrastive | 0.931    | 0.888   | 0.941     | 0.969    | 0.934  | 0.722  |
| truth     | random            | 0.847    | 0.878   | 0.958     | 0.966    | 0.929  | 0.678  |
| utility   | matched           | 0.774    | 0.776   | 0.782     | 0.785    | 0.774  | 0.655  |
| utility   | label_contrastive | 0.779    | 0.773   | 0.784     | 0.759    | 0.775  | 0.655  |
| utility   | random            | 0.783    | 0.779   | 0.782     | 0.752    | 0.773  | 0.655  |

By preprocessing, with the pairing above:

| Task      | Finder    | center               | unit                 | none                 |
| --------- | --------- | -------------------- | -------------------- | -------------------- |
| sentiment | pca_diff  | 0.511 [0.467, 0.553] | 0.506 [0.488, 0.522] | 0.511 [0.466, 0.553] |
| sentiment | pca_raw   | 0.505 [0.485, 0.524] | 0.513 [0.468, 0.555] | 0.529 [0.503, 0.550] |
| sentiment | mean_diff | 0.614 [0.586, 0.645] | 0.611 [0.583, 0.641] | 0.614 [0.585, 0.645] |
| sentiment | logistic  | 0.757 [0.730, 0.791] | 0.761 [0.734, 0.795] | 0.613 [0.583, 0.644] |
| truth     | pca_diff  | 0.931 [0.922, 0.939] | 0.925 [0.917, 0.934] | 0.932 [0.924, 0.939] |
| truth     | pca_raw   | 0.888 [0.868, 0.910] | 0.900 [0.881, 0.919] | 0.537 [0.531, 0.544] |
| truth     | mean_diff | 0.941 [0.932, 0.949] | 0.953 [0.942, 0.963] | 0.941 [0.932, 0.949] |
| truth     | logistic  | 0.969 [0.959, 0.978] | 0.969 [0.959, 0.978] | 0.944 [0.937, 0.951] |
| utility   | pca_diff  | 0.774 [0.770, 0.778] | 0.778 [0.772, 0.784] | 0.775 [0.772, 0.779] |
| utility   | pca_raw   | 0.776 [0.772, 0.780] | 0.777 [0.773, 0.780] | 0.664 [0.656, 0.672] |
| utility   | mean_diff | 0.782 [0.777, 0.787] | 0.780 [0.774, 0.786] | 0.782 [0.777, 0.788] |
| utility   | logistic  | 0.785 [0.773, 0.797] | 0.787 [0.774, 0.799] | 0.785 [0.778, 0.792] |

- **The paper finder reads truth and utility, and not sentiment.** On sentiment `pca_diff` is at 0.511 [0.467, 0.553] against 0.525 [0.490, 0.560] for the random direction, with overlapping intervals. On truth it is at 0.931 [0.922, 0.939] against 0.722 [0.642, 0.795], and on utility at 0.774 [0.770, 0.778] against 0.655 [0.615, 0.693]. The first principal component explains 28 percent (sentiment, at L9), 40 percent (truth, at L24) and 46 percent (utility, matched, at L24) of the variance of the pair differences. The component chosen by cross-validation among the first eight reads 0.564 [0.528, 0.592] for sentiment, against 0.511 for the first component, with an interval that still overlaps that of the random direction, and the component chosen most often across the 10 seeds is the second; for truth and utility it is the first component (0.931 and 0.774).
- **The finders are ordered differently by task.** On sentiment only `logistic` (0.757) and `mean_diff` (0.614) are above the random direction. On truth all five finders are, in the order `logistic` 0.969 [0.959, 0.978], `mean_diff` 0.941, `kmeans` 0.934, `pca_diff` 0.931 and `pca_raw` 0.888 [0.868, 0.910]; the interval of `pca_diff` lies below that of `logistic`. On utility the five are within 0.011 of each other (`logistic` 0.785 [0.773, 0.797], `mean_diff` 0.782, `pca_raw` 0.776, `pca_diff` 0.774 and `kmeans` 0.774) and all are above the random direction (0.655 [0.615, 0.693]).
- **Pairing.** The differences between pairings for `logistic` are at most 0.033 (utility 0.785, 0.759 and 0.752; sentiment 0.757 and 0.771 for `random`, truth 0.969 and 0.966). `pca_diff` on truth depends on the pairing more than any other finder and task: 0.931 with `label_contrastive` and 0.847 with `random`. `mean_diff` on truth is 0.941 with `label_contrastive` and 0.958 with `random`.
- **Preprocessing.** `logistic` loses with no preprocessing on sentiment (0.757, 0.761, 0.613) and on truth (0.969, 0.969, 0.944 [0.937, 0.951]; the interval of the centered cell, [0.959, 0.978], does not overlap it), and is level on utility (0.785, 0.787, 0.785). `pca_diff` changes by at most 0.007 across the three on every task. `pca_raw` falls without preprocessing on truth (0.537 [0.531, 0.544] against 0.888, an interval of width 0.013) and on utility (0.664 against 0.776).

### What P3 and P4 show across sizes

Each model has one run, and base and post-trained checkpoints are compared cell by cell and not pooled.

**P3.** The table gives, per model, the smallest size from which none of the 20 draws is below 0.6 at that size or at any larger one (`none` when the largest size still has failing draws; for truth the largest size is 128 pairs, the end of the source). The `gpt2` row is the run reported [above](#stimulus-set-size-p3) and the `qwen3-1.7b-base` row the run reported [with P2](#stimulus-set-size-on-qwen3-17b-base-p3).

| Model           | sentiment | truth | truth_cities | truth_common | utility |
| --------------- | --------- | ----- | ------------ | ------------ | ------- |
| gpt2            | 32        | none  | 256          | none         | 256     |
| gpt2-medium     | 64        | 128   | 64           | 512          | 64      |
| gpt2-large      | 32        | 128   | none         | none         | 64      |
| gpt2-xl         | 32        | 128   | none         | 512          | 64      |
| qwen3-0.6b-base | 8         | 32    | 32           | 256          | 16      |
| qwen3-0.6b      | 32        | 64    | 32           | 256          | 8       |
| qwen3-1.7b-base | 4         | 2     | 2            | 8            | 8       |
| qwen3-1.7b      | 16        | 8     | 2            | 16           | 8       |
| qwen3-4b-base   | 16        | 2     | 2            | 8            | 8       |
| qwen3-4b        | 32        | 16    | 2            | 8            | 8       |

- **Sentiment and utility.** The GPT-2 readers need 32 to 64 pairs for sentiment and 64 pairs (256 on `gpt2`) for utility before no draw fails; the Qwen3 readers need 4 to 32 pairs for sentiment (16 on `qwen3-4b-base` and 32 on `qwen3-4b`) and 8 to 16 for utility (8 on both 4B checkpoints). At the largest size the sentiment mean is below the full-data reference on all ten models (by 0.004 to 0.042), and its stored saturation size is 512 on seven of them (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-0.6b`, `qwen3-1.7b` and `qwen3-4b`), so on those seven the sentiment curve has not flattened by 512 pairs. Utility flattens earlier on Qwen3 (stored saturation size 32 on the 0.6B pair, 8 on the 1.7B pair, 32 on `qwen3-4b-base` and 8 on `qwen3-4b`, and the mean moves by between -0.012 and 0.009 from 64 to 512 pairs) than on the GPT-2 sizes (256 on three and 512 on `gpt2-xl`, with a gain of 0.015 to 0.037 from 64 to 512 pairs).
- **Truth.** On the four GPT-2 sizes the truth reader has failing draws at every size below 128 pairs; at 128 pairs, the end of the source, the mean is 0.642 (`gpt2`), 0.654, 0.676 and 0.662, within 0.024 of the references (0.644, 0.678, 0.695 and 0.661). On Qwen3 no draw fails from 32 pairs on `qwen3-0.6b-base`, from 64 on `qwen3-0.6b`, from 8 on `qwen3-1.7b`, from 2 on `qwen3-1.7b-base` and `qwen3-4b-base` and from 16 on `qwen3-4b`, and the 128-pair means (0.830, 0.838, 0.932, 0.971, 1.000 and 0.965) are within 0.009 of their references on the first five and 0.035 below the reference of `qwen3-4b` (1.000). On nine of the ten models the truth mean at 128 pairs is within 0.024 of its reference.
- **The truth subsets.** `truth_cities` has no failing draw from 2 pairs on both 1.7B and both 4B checkpoints, but it still has a failure rate of 0.30 at 256 pairs on `gpt2-xl`, and on `gpt2-large` it is at chance on the truth target at every size (0.469 to 0.503, reference 0.508). `truth_common` reaches no failing draw only at 256 or 512 pairs on `gpt2-medium`, `gpt2-xl` and the 0.6B pair, never on `gpt2` and `gpt2-large`, at 8 and 16 pairs on the two 1.7B checkpoints and at 8 pairs on both 4B checkpoints.
- **Base against post-trained.** At 0.6B, 1.7B and 4B the post-trained point value is below the base one in all 9 sentiment cells, all 9 `truth_common` cells and all 8 `truth_cities` cells, and in 6 of 7 (0.6B) and 7 of 7 (1.7B and 4B) truth cells; the intervals over the draws are separate in 8 (0.6B) and 9 (1.7B and 4B) of the 9 sentiment cells, and at 4B in all 33 cells of sentiment, truth, `truth_cities` and `truth_common`. The post-trained 4B truth reader is 0.386 below the base one at 2 pairs and 0.035 below it at 128. On utility the two are level at 0.6B and 1.7B: the differences (post-trained minus base) are between -0.005 and 0.028 and no interval is separate. At 4B the post-trained reader is higher at 2, 4 and 8 pairs (by 0.027, 0.031 and 0.030, with a separate interval only at 8) and lower from 16 pairs on (by 0.005 to 0.016, with no separate interval). The base 1.7B run was made at `c7d8051` on a different card (NVIDIA GeForce RTX 4090) than the post-trained one, as in the P2 comparison above; the two 4B runs were made at the same commit on the same pod card.

**P4.** The table gives, for each model, the accuracy of the paper finder (`pca_diff`), of `logistic` and of the random direction in the reference cell of each task (centered, with the pairing the task has). The `gpt2` row is the run reported [above](#direction-finder-pairing-and-preprocessing-p4). A `+` after a value means that its interval lies above the interval of the random direction in the same task, a `~` that the two overlap and a `-` that it lies below. The intervals are over the 10 seeds and do not include the sampling of the evaluation pairs.

| Model           | sentiment random | sentiment pca_diff | sentiment logistic | truth random | truth pca_diff | truth logistic | utility random | utility pca_diff | utility logistic |
| --------------- | ---------------- | ------------------ | ------------------ | ------------ | -------------- | -------------- | -------------- | ---------------- | ---------------- |
| gpt2            | 0.543            | 0.512 ~            | 0.765 +            | 0.510        | 0.481 ~        | 0.698 +        | 0.526          | 0.551 ~          | 0.644 +          |
| gpt2-medium     | 0.557            | 0.517 ~            | 0.742 +            | 0.536        | 0.517 ~        | 0.656 +        | 0.553          | 0.618 ~          | 0.685 +          |
| gpt2-large      | 0.525            | 0.495 ~            | 0.807 +            | 0.534        | 0.534 ~        | 0.653 +        | 0.546          | 0.571 ~          | 0.688 +          |
| gpt2-xl         | 0.545            | 0.507 ~            | 0.776 +            | 0.503        | 0.488 ~        | 0.656 +        | 0.525          | 0.576 ~          | 0.686 +          |
| qwen3-0.6b-base | 0.658            | 0.665 ~            | 0.952 +            | 0.549        | 0.588 ~        | 0.764 +        | 0.604          | 0.727 +          | 0.709 +          |
| qwen3-0.6b      | 0.594            | 0.564 ~            | 0.860 +            | 0.558        | 0.444 -        | 0.744 +        | 0.614          | 0.718 +          | 0.711 +          |
| qwen3-1.7b-base | 0.676            | 0.846 +            | 0.943 +            | 0.724        | 0.959 +        | 0.971 +        | 0.649          | 0.799 +          | 0.761 +          |
| qwen3-1.7b      | 0.585            | 0.531 ~            | 0.877 +            | 0.639        | 0.869 +        | 0.932 +        | 0.627          | 0.780 +          | 0.760 +          |
| qwen3-4b-base   | 0.577            | 0.529 ~            | 0.900 +            | 0.859        | 0.990 +        | 0.998 +        | 0.581          | 0.674 +          | 0.780 +          |
| qwen3-4b        | 0.525            | 0.511 ~            | 0.757 +            | 0.722        | 0.931 +        | 0.969 +        | 0.655          | 0.774 +          | 0.785 +          |

- **Finders.** `logistic` has the highest point value on sentiment and truth on all ten models, and on utility on the four GPT-2 sizes and on `qwen3-4b`. On utility on the other four Qwen3 checkpoints it has the lowest point value of the five finders on all four: its interval is below those of the other four on `qwen3-0.6b-base`, `qwen3-1.7b-base` and `qwen3-1.7b`, and on `qwen3-0.6b` the five are level (0.711 to 0.720). On `qwen3-4b-base` it is second (0.780, against 0.786 for `mean_diff`), and on `qwen3-4b` the five are within 0.011 (0.774 to 0.785). `mean_diff` is above the random direction on sentiment and utility on all ten models and on truth on six (`gpt2-medium`, `qwen3-0.6b-base`, `qwen3-1.7b-base`, `qwen3-1.7b`, `qwen3-4b-base` and `qwen3-4b`). `pca_raw` and `kmeans` are above it on utility on nine models (all but `gpt2`), on sentiment only on `qwen3-1.7b-base` and on truth only on `qwen3-1.7b-base`, `qwen3-1.7b` and the 4B pair.
- **The paper finder against the random direction.** The interval of `pca_diff` lies above that of the random direction on sentiment on one model (`qwen3-1.7b-base`), on truth on four (`qwen3-1.7b-base`, `qwen3-1.7b` and the 4B pair) and on utility on six (the six Qwen3 checkpoints), and on none of the four GPT-2 sizes on any task; on truth on `qwen3-0.6b` it lies below (0.444 [0.432, 0.456] against 0.558 [0.514, 0.605]). Choosing the component by cross-validation among the first eight adds four cells where the first component is not above the random direction: sentiment on `qwen3-0.6b-base` (0.818) and `qwen3-1.7b` (0.712), where the component chosen most often is the second, and utility on `gpt2-large` (0.599) and `gpt2-xl` (0.609); on the 4B pair it adds none (sentiment 0.608 on the base checkpoint and 0.564 on the post-trained one, both overlapping the random direction).
- **The random direction.** It reads between 0.503 and 0.557 on the GPT-2 sizes and between 0.525 and 0.859 on the Qwen3 checkpoints. It is the highest of the ten models on `qwen3-1.7b-base` for sentiment (0.676), on `qwen3-4b-base` for truth (0.859 [0.783, 0.924]) and on `qwen3-4b` for utility (0.655). The shuffled-label accuracy is between 0.45 and 0.54 in all ten runs.
- **Pairing.** For `logistic` the pairings differ by at most 0.046 on any task and model (utility on `gpt2-large`). On utility the best pairing for `logistic` is `matched` on eight models and `label_contrastive` on the 0.6B pair, where it exceeds `matched` by 0.005 or less; on sentiment and truth the best pairing changes from model to model (differences of at most 0.023 and 0.042). On utility `pca_raw` has a higher point value with `matched` than with the other two pairings on the four GPT-2 sizes (by 0.011 to 0.062), and the intervals are separate only on `gpt2-large` (against both) and on `gpt2-xl` (against `random`); on the Qwen3 checkpoints the point value with `matched` is within 0.007 of the best other pairing, except on `qwen3-4b-base`, where it is 0.014 below it. On `qwen3-4b-base` `pca_diff` on utility is 0.674 with `matched` against 0.756 with `label_contrastive`.
- **Preprocessing.** `logistic` has a lower point value with no preprocessing than centered on sentiment and truth on all ten models (by 0.012 to 0.145 on sentiment and 0.002 to 0.161 on truth) and on utility on the four GPT-2 sizes (by 0.016 to 0.038), and a higher one on utility on the five Qwen3 checkpoints other than `qwen3-4b` (by 0.009 to 0.036); on `qwen3-4b` the two are level on utility (0.785 for both). `unit` is within 0.006 of centered in 27 of the 30 `logistic` cells; it is lower on sentiment on `gpt2` (by 0.020) and on truth on `gpt2` (0.041) and `gpt2-large` (0.042).
- **Base against post-trained.** Over the five finders and the three tasks, the base checkpoint has the higher point value in 13 of 15 cells at 0.6B, in 15 of 15 at 1.7B and in 11 of 15 at 4B. Where the intervals are separate (10 cells at 0.6B, 14 at 1.7B and 10 at 4B) the base checkpoint is the higher one at 0.6B and 1.7B, and the post-trained checkpoint is never above. At 4B the base checkpoint is the higher one in 7 of the 10 separate cells (sentiment `mean_diff` and `logistic`, and all five finders on truth) and the post-trained checkpoint in 3 (`pca_diff`, `pca_raw` and `kmeans` on utility, by 0.100, 0.052 and 0.030). The largest differences are on sentiment at 1.7B for `pca_diff`, `pca_raw` and `kmeans` (0.314 to 0.324); on utility the differences (base minus post-trained) are between -0.002 and 0.029 at 0.6B and 1.7B and between -0.100 and 0.004 at 4B. The random direction is also higher on the base checkpoint on sentiment at the three sizes (0.658 against 0.594, 0.676 against 0.585 and 0.577 against 0.525), with separate intervals at 1.7B only; at 4B it is higher on the post-trained checkpoint on utility (0.655 against 0.581) and on the base checkpoint on truth (0.859 against 0.722), with overlapping intervals. The base 1.7B run was made on an NVIDIA L4 and the post-trained 1.7B run on an RTX 3090; the two 4B runs were made at the same commit on the same pod card.

## Persona wording and instruction confounds (P5)

P5 asks whether the honesty reader reads honesty or the persona's words. Each condition changes one thing against the paper persona prompt; the reader is trained on the response tokens of true statements as in R5, and scored on plain true against false statements at the token mean, on the 59 `facts` test pairs and on the test splits of five Geometry of Truth subsets (133 `cities`, 142 `neg_cities`, 36 `sp_en_trans`, 187 `larger_than` and 256 `common_claim_true_false` pairs). The conditions are the paper pair ("an honest" against "an untruthful"), two synonym pairs (`synonym_0`: "a truthful" and "a deceitful"; `synonym_1`: "a sincere" and "a lying"), two unrelated pairs (`unrelated_0`: "a tall" and "a short"; `unrelated_1`: "an old" and "a young"), and three controls: `instruction_only` (the paper personas with an empty response), `fact_only` (no persona, plain true against plain false statements) and `random_label` (the paper stimuli with the training labels swapped in exactly half of the pairs). Accuracies are raw: a reader whose first persona aligns with the false statements scores below 0.5. Layers are chosen per condition on the 53 `facts` validation pairs (the `Layer` column) and every set is scored at that layer. The in-template accuracy of the persona conditions is 1.0 by construction, as for R5, and is not reported. The `instruction_only` readers are identical across seeds (every training set holds the same two texts), so that row comes from one reader and carries no interval; any other cell shown without an interval has a stored interval that rounds to the point value. Intervals are over the 10 readers; they do not include the sampling of the evaluation pairs, which matters most for `facts` (59) and `sp_en_trans` (36).

The runs are `p5/gpt2/80b4ae80a6ca`, `p5/qwen3-1.7b-base/a4bd463deeca` and `p5/qwen3-1.7b/7e5e579d102e`, all made at `c7d8051` with a clean tree on a cloud GPU (NVIDIA GeForce RTX 4090). The earlier `gpt2` attempts stopped on a fit that produced a direction of zero norm; these are the runs after the fix. Three more runs, `p5/gpt2-medium/bd9f565baa76`, `p5/qwen3-0.6b-base/b6a702e460e9` and `p5/qwen3-0.6b/fb664c3b4be2`, were made at `fb3b5dd` with a clean tree on a laptop GPU (NVIDIA GeForce RTX 4070 Laptop GPU) with batch size 8 and the same config; the P5 code is unchanged between `c7d8051` and `fb3b5dd`, so these runs differ from the first three in checkpoint and in hardware. Two more runs, `p5/gpt2-large/430d434007c9` and `p5/gpt2-xl/4171a611f734`, were made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500) with batch size 8 and the same config; the P5 code is unchanged since `fb3b5dd` (`git diff --stat fb3b5dd f4d414d -- src` lists no P5 file), so these runs differ from the others in checkpoint and in hardware. Two more runs, `p5/qwen3-4b-base/9c3330f88a67` and `p5/qwen3-4b/7ac5d1bc7189`, were made at `86c319b` with a clean tree on another rented GPU pod (NVIDIA RTX A4000) with batch size 8 and the same config; `git diff --stat f4d414d 86c319b -- src` lists no P5 file (the only change to a module P5 imports adds the two 4B checkpoints to the model registry), so these runs differ from the others in checkpoint and in hardware. Base and post-trained rows are separate tables and are never pooled. The stored gap is the mean of the paper and the two synonym conditions minus the mean of the two unrelated ones, per set.

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

### qwen3-4b-base

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L30   | 0.585 [0.559, 0.607] | 0.787 [0.775, 0.800] | 0.489 [0.473, 0.504] | 0.744 [0.733, 0.758] | 0.651 [0.614, 0.686] | 0.613 [0.603, 0.623] |
| synonym_0        | L30   | 0.636 [0.615, 0.654] | 0.637 [0.629, 0.644] | 0.348 [0.333, 0.363] | 0.578 [0.555, 0.603] | 0.639 [0.630, 0.648] | 0.581 [0.576, 0.586] |
| synonym_1        | L36   | 0.603 [0.581, 0.625] | 0.744 [0.735, 0.751] | 0.568 [0.552, 0.586] | 0.592 [0.575, 0.608] | 0.890 [0.885, 0.895] | 0.604 [0.595, 0.613] |
| unrelated_0      | L21   | 0.592 [0.578, 0.607] | 0.699 [0.684, 0.717] | 0.713 [0.695, 0.729] | 0.278 [0.247, 0.308] | 0.444 [0.437, 0.450] | 0.501 [0.497, 0.504] |
| unrelated_1      | L7    | 0.431 [0.417, 0.446] | 0.562 [0.552, 0.573] | 0.486 [0.479, 0.493] | 0.675 [0.658, 0.694] | 0.566 [0.563, 0.570] | 0.497 [0.491, 0.504] |
| instruction_only | L14   | 0.424                | 0.556                | 0.479                | 0.750                | 0.594                | 0.547                |
| fact_only        | L36   | 0.705 [0.692, 0.717] | 0.555 [0.532, 0.580] | 0.626 [0.611, 0.642] | 0.583 [0.547, 0.628] | 0.806 [0.776, 0.828] | 0.659 [0.649, 0.670] |
| random_label     | L10   | 0.495 [0.442, 0.546] | 0.502 [0.460, 0.541] | 0.485 [0.458, 0.511] | 0.572 [0.506, 0.639] | 0.544 [0.448, 0.638] | 0.501 [0.487, 0.513] |

- **The paper persona reads above chance on five of the six sets.** It scores 0.585 [0.559, 0.607] on `facts`, 0.787 [0.775, 0.800] on `cities`, 0.744 on `sp_en_trans`, 0.651 on `larger_than` and 0.613 on `common_claim`; `neg_cities` is 0.489 [0.473, 0.504], an interval that includes 0.5. `fact_only`, with no persona, reads `facts` at 0.705 [0.692, 0.717], 0.120 above the paper persona, and `larger_than` at 0.806, but `cities` only at 0.555 against 0.787 for the paper persona.
- **The persona words matter on five of the six sets and `neg_cities` goes the other way.** The stored gaps are 0.097 (`facts`), 0.092 (`cities`), 0.162 (`sp_en_trans`), 0.222 (`larger_than`) and 0.100 (`common_claim`), and -0.131 on `neg_cities`. The words do not account for all of it. On `facts` the paper and synonym readers score 0.585, 0.636 and 0.603 against 0.592 [0.578, 0.607] for `unrelated_0` ("a tall", "a short") and 0.431 [0.417, 0.446] for `unrelated_1` ("an old", "a young"): the gap rests on the second unrelated pair, and `unrelated_0` reads `facts` inside the range of the paper and synonym readers. On `neg_cities`, `unrelated_0` reads 0.713 [0.695, 0.729], the highest of the five persona conditions, and `synonym_0` ("a truthful", "a deceitful") reads 0.348 [0.333, 0.363], below chance. On `sp_en_trans`, `unrelated_0` reads 0.278 [0.247, 0.308] against 0.744 for the paper persona, while `unrelated_1` reads 0.675. The run does not say what the readers of the unrelated words read.
- **The shuffled-label control reads at chance on five of the six sets.** `random_label` has intervals that include 0.5 except on `sp_en_trans` (0.572 [0.506, 0.639], a lower bound 0.006 above 0.5); on `facts` it reads 0.495 [0.442, 0.546]. Its stored in-template column is 0.685 [0.414, 0.899] at L3.
- `instruction_only` is one reader per cell: 0.424 on `facts` and 0.750 on `sp_en_trans` (36 pairs), without an interval.

### qwen3-4b

| Condition        | Layer | facts                | cities               | neg_cities           | sp_en_trans          | larger_than          | common_claim         |
| ---------------- | ----- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| paper            | L32   | 0.761 [0.739, 0.780] | 0.717 [0.701, 0.732] | 0.633 [0.623, 0.644] | 0.714 [0.703, 0.728] | 0.933 [0.864, 0.987] | 0.696 [0.684, 0.707] |
| synonym_0        | L30   | 0.539 [0.517, 0.563] | 0.618 [0.605, 0.633] | 0.557 [0.549, 0.565] | 0.564 [0.550, 0.575] | 0.640 [0.626, 0.656] | 0.631 [0.619, 0.644] |
| synonym_1        | L32   | 0.705 [0.692, 0.719] | 0.673 [0.660, 0.686] | 0.582 [0.570, 0.594] | 0.619 [0.603, 0.636] | 0.935 [0.866, 0.990] | 0.702 [0.693, 0.709] |
| unrelated_0      | L24   | 0.514 [0.500, 0.529] | 0.493 [0.478, 0.507] | 0.613 [0.598, 0.627] | 0.306 [0.283, 0.328] | 0.573 [0.561, 0.589] | 0.604 [0.598, 0.610] |
| unrelated_1      | L35   | 0.539 [0.520, 0.559] | 0.738 [0.726, 0.751] | 0.563 [0.558, 0.568] | 0.381 [0.364, 0.400] | 0.706 [0.681, 0.739] | 0.499 [0.486, 0.511] |
| instruction_only | L15   | 0.458                | 0.526                | 0.465                | 0.278                | 0.578                | 0.492                |
| fact_only        | L36   | 0.722 [0.697, 0.749] | 0.690 [0.623, 0.771] | 0.666 [0.644, 0.688] | 0.753 [0.719, 0.789] | 0.850 [0.807, 0.885] | 0.707 [0.679, 0.732] |
| random_label     | L21   | 0.488 [0.463, 0.517] | 0.545 [0.487, 0.618] | 0.525 [0.470, 0.574] | 0.419 [0.339, 0.519] | 0.516 [0.485, 0.550] | 0.532 [0.518, 0.547] |

- **The paper persona reads above chance on all six sets**, with every interval above 0.5: 0.761 [0.739, 0.780] on `facts`, 0.717 on `cities`, 0.633 [0.623, 0.644] on `neg_cities`, 0.714 on `sp_en_trans`, 0.933 [0.864, 0.987] on `larger_than` and 0.696 on `common_claim`. The `larger_than` interval is the widest of the row (0.123 against at most 0.041 on the other sets), so the ten readers differ more there. `fact_only` reads `facts` at 0.722 [0.697, 0.749] and the six sets between 0.666 and 0.850.
- **The persona words matter on four sets and hardly on the other two.** The stored gaps are 0.289 (`sp_en_trans`), 0.196 (`larger_than`), 0.142 (`facts`), 0.125 (`common_claim`), 0.053 (`cities`) and 0.003 (`neg_cities`). On `sp_en_trans` the paper and synonym readers score 0.714, 0.564 and 0.619 against 0.306 [0.283, 0.328] and 0.381 [0.364, 0.400], so the unrelated readers fall below chance with intervals entirely below 0.5. On `facts` they score 0.761, 0.539 and 0.705 against 0.514 [0.500, 0.529] and 0.539 [0.520, 0.559]; `synonym_0` ("a truthful", "a deceitful") reads as low as `unrelated_1` (0.539 for both), so the wording inside the honesty group matters as much as the group. On `cities`, `unrelated_1` reads 0.738 [0.726, 0.751], above all three honesty readers (0.717, 0.618 and 0.673; the paper interval, [0.701, 0.732], overlaps it), and on `neg_cities` all five persona conditions read between 0.557 and 0.633.
- **The shuffled-label control reads at chance on five of the six sets.** `random_label` has intervals that include 0.5 except on `common_claim` (0.532 [0.518, 0.547]); on `facts` it reads 0.488 [0.463, 0.517]. Its stored in-template column is 0.659 [0.421, 0.868] at L6.
- `instruction_only` is one reader per cell: 0.458 on `facts` and 0.278 on `sp_en_trans`, without an interval; the second is as low as the unrelated readers on that set (0.306 and 0.381).

### What the ten runs show together

- The paper persona reader scores below chance on `facts` on three of the four GPT-2 sizes (0.353 on `gpt2`, 0.403 on `gpt2-large` and 0.481 on `gpt2-xl`, each with an interval below 0.5) and above it on `gpt2-medium` (0.592) and on every Qwen3 checkpoint: 0.597 on `qwen3-0.6b-base`, 0.598 on `qwen3-0.6b`, 0.736 on `qwen3-1.7b-base`, 0.744 on `qwen3-1.7b`, 0.585 on `qwen3-4b-base` and 0.761 on `qwen3-4b`. The two checkpoints of a size read `facts` within 0.008 of each other at 0.6B and 1.7B and 0.176 apart at 4B (0.585 [0.559, 0.607] against 0.761 [0.739, 0.780], intervals that do not overlap), each from one run. On the four GPT-2 sizes only `fact_only` reads `facts` above 0.6 (0.663, 0.644, 0.663 and 0.702 from `gpt2` to `gpt2-xl`); on the Qwen3 checkpoints several persona conditions do.
- The stored gap between the paper and synonym readers and the unrelated readers on `facts` is 0.013 (`gpt2`), 0.014 (`gpt2-medium`), -0.053 (`gpt2-large`), -0.010 (`gpt2-xl`), 0.159 (`qwen3-0.6b-base`), 0.007 (`qwen3-0.6b`), 0.132 (`qwen3-1.7b-base`), 0.203 (`qwen3-1.7b`), 0.097 (`qwen3-4b-base`) and 0.142 (`qwen3-4b`). A dependence on the persona words shows on the 1.7B pair and on `qwen3-0.6b-base`, where the unrelated readers fall to or near chance on `facts` (0.600 and 0.520 on `qwen3-1.7b-base`, 0.541 and 0.419 on `qwen3-0.6b-base`, 0.510 and 0.492 on `qwen3-1.7b`) while the paper and synonym readers stay at 0.575 or above; it does not show on the four GPT-2 sizes or on `qwen3-0.6b`, where the unrelated readers read `facts` as the others do (on `gpt2-large` the gap is negative: the unrelated readers are the higher ones). The 4B pair is a weaker case. On `qwen3-4b-base` the gap rests on one unrelated reader: `unrelated_1` reads 0.431 [0.417, 0.446], below chance, while `unrelated_0` reads 0.592 [0.578, 0.607], inside the range of the paper and synonym readers (0.585 to 0.636). On `qwen3-4b` the unrelated readers read 0.514 [0.500, 0.529] and 0.539 [0.520, 0.559], near chance, but so does `synonym_0` (0.539 [0.517, 0.563]) while the paper and `synonym_1` readers read 0.761 and 0.705. The sign of the base to post-trained difference in this gap is not the same at every size: it falls from the base to the post-trained checkpoint at 0.6B (0.159 to 0.007) and rises at 1.7B (0.132 to 0.203) and 4B (0.097 to 0.142). Each gap rests on one run per checkpoint and 59 `facts` pairs, and `unrelated_0` is above chance on `qwen3-1.7b-base` (0.600 [0.581, 0.620]) and on `qwen3-4b-base` (0.592 [0.578, 0.607]).
- The gap on `larger_than` is negative on all four GPT-2 sizes: -0.465 (`gpt2`), -0.231 (`gpt2-medium`), -0.293 (`gpt2-large`) and -0.254 (`gpt2-xl`); on `gpt2-large` and `gpt2-xl` the intervals of all five persona conditions exclude 0.5. It does not grow or shrink steadily with size. On the Qwen3 checkpoints the gap is -0.174 and -0.061 on the 0.6B pair and -0.037 on `qwen3-1.7b-base`, then positive on `qwen3-1.7b` (0.069) and on the 4B pair (0.222 on `qwen3-4b-base` and 0.196 on `qwen3-4b`, the two largest of the ten runs); the runs do not say why the sign changes.
- Some cells under the unrelated words sit far from 0.5 (`larger_than` on `gpt2`, 0.670 and 0.826, on `gpt2-medium`, 0.796 for `unrelated_1`, on `gpt2-large`, 0.706 for `unrelated_1`, and on `gpt2-xl`, 0.707 and 0.705; `sp_en_trans` on `qwen3-1.7b`, 0.353 and 0.367, on `qwen3-0.6b-base`, 0.344 for `unrelated_1`, and on `qwen3-0.6b`, 0.444 and 0.322; and on the 4B pair, `sp_en_trans` on `qwen3-4b-base`, 0.278 for `unrelated_0` and 0.675 for `unrelated_1`, and on `qwen3-4b`, 0.306 and 0.381, with `neg_cities` on `qwen3-4b-base`, 0.713 for `unrelated_0`, and `cities` on `qwen3-4b`, 0.738 for `unrelated_1`); the runs do not say what those readers read.
- Each cell is one run of one model at one layer chosen on 53 pairs and scored on small sets, with intervals over the training draw only. The runs do not test persona wordings beyond the five pairs above. The first three runs were made on a cloud GPU (`c7d8051`), the next three on a laptop GPU (`fb3b5dd`), the next two on a rented pod GPU (`f4d414d`) and the 4B pair on another rented pod GPU (`86c319b`), with the same config and the same P5 code.

## Template steering on the Qwen3 1.7B and 4B pairs (P7)

P7 is the protocol of [P7 on gpt2](#template-steering-p7): the sentiment reader is extracted under six templates of P1 and used to steer with the C1 protocol (coefficients -0.3, -0.15, 0, 0.15 and 0.3, 50 prompts, 2 samples per prompt, 2 steering readers per template), over the sweep layers 1, 6, 12, 17, 23 and 28 on the 1.7B pair (28 is the last block; the 4B pair follows at the end of the section). A unit is a template, a sweep layer and a steering reader (72 units). It is readable if the validation accuracy of its reader at that layer is at least 0.6, and inverted if it is readable and its primary dose slope is negative. The reported runs are `p7/qwen3-1.7b-base/16a7a9e4facf` and `p7/qwen3-1.7b/eacde3abd0aa`, both made at `f4d414d` with a clean tree on a rented GPU pod (NVIDIA RTX A4500). They replace `p7/qwen3-1.7b-base/947fef33a7c5` and `p7/qwen3-1.7b/7921f3fe5754` (made at `20650c5` on a laptop GPU, NVIDIA GeForce RTX 4070 Laptop GPU), which stay tracked; the configs differ in `n_random` only (absent earlier, 3 here), so these runs add three random directions (one reader each, steered at every sweep layer with the same grid) and the fluency measure (perplexity of held-out neutral text and distinct-1 of the generated text, from `table_effects`). Base and post-trained are two models: the tables are separate and no metric is pooled over them. The counts of inverted units changed between the two runs of each checkpoint, from 9 to 12 on the base and from 16 to 9 on the post-trained checkpoint, so the ordering of the two checkpoints by inversion rate in the earlier runs (0.125 against 0.222) does not hold in these runs (0.167 against 0.125).

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

### The 4B pair (P7)

The same protocol ran on `qwen3-4b-base` and `qwen3-4b` (runs `p7/qwen3-4b-base/7d0197f7c9bf` and
`p7/qwen3-4b/be6b6b4f3b93`), both made at `86c319b` with a clean tree on one rented GPU pod (NVIDIA RTX A4000) with
batch size 16 (see [Run identity](#run-identity)). The config is that of the 1.7B runs above (50 prompts, 2 samples per
prompt, 2 steering readers per template, 3 random directions, the same coefficients and threshold) and differs in the
checkpoint only. The model has 36 blocks, so the sweep layers are 1, 8, 15, 22, 29 and 36 (36 is the last block). A unit
is again a template, a sweep layer and a steering reader (72 units). Base and post-trained are two models: the tables
are separate and no metric is pooled over them.

#### qwen3-4b-base

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at steer layer | Dose slope, sweep layers 1 to 29 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ------------------------- | -------------------------------- | -------------- | -------- |
| paper          | L24        | 0.905         | L36         | 1.045                     | -0.070 to 0.518                  | 11             | 1        |
| no_template    | L23        | 0.790         | L36         | 0.897                     | 0.041 to 0.678                   | 11             | 2        |
| no_chat_tags   | L27        | 0.855         | L36         | 0.936                     | -0.029 to 0.697                  | 12             | 1        |
| neutral_frame  | L21        | 0.811         | L36         | 0.899                     | -0.064 to 0.493                  | 11             | 1        |
| scaffold_0     | L35        | 0.953         | L36         | 1.039                     | 0.040 to 0.598                   | 11             | 0        |
| concept_swap_0 | L24        | 0.814         | L36         | 0.819                     | -0.078 to 0.539                  | 11             | 1        |

#### qwen3-4b

| Template       | Read layer | Read accuracy | Steer layer | Dose slope at steer layer | Dose slope, sweep layers 1 to 29 | Readable units | Inverted |
| -------------- | ---------- | ------------- | ----------- | ------------------------- | -------------------------------- | -------------- | -------- |
| paper          | L23        | 0.763         | L22         | 0.073                     | -0.082 to 0.073                  | 12             | 8        |
| no_template    | L20        | 0.900         | L22         | 0.622                     | -0.007 to 0.622                  | 12             | 2        |
| no_chat_tags   | L24        | 0.777         | L36         | 0.418                     | -0.102 to 0.251                  | 12             | 5        |
| neutral_frame  | L20        | 0.906         | L22         | 0.525                     | -0.090 to 0.525                  | 12             | 5        |
| scaffold_0     | L24        | 0.919         | L22         | 0.766                     | 0.038 to 0.766                   | 12             | 1        |
| concept_swap_0 | L24        | 0.768         | L22         | 0.274                     | -0.076 to 0.274                  | 12             | 4        |

The read accuracy is the best point of the mean reading curve over the 10 readers, so it is optimistic by the selection.
The sweep-layer column is the minimum and maximum of the mean dose slope over layers 1, 8, 15, 22 and 29, read from the
curves files; on the post-trained checkpoint it includes the steering layer (L22) for five templates. On the base
checkpoint 67 of the 72 units are readable: the five others are the first steering reader of the five templates other
than `no_chat_tags` at layer 1, with a validation accuracy of 0.547 to 0.586.

- **The steering layer.** On the base checkpoint it is the last block (L36) for all six templates, with slopes of 0.819
  to 1.045 there; the slope is not monotone before it (`paper`: -0.070, 0.133, 0.263, 0.518, 0.473 and 1.045 at layers
  1, 8, 15, 22, 29 and 36). On the post-trained checkpoint it is L22 for five templates, with slopes from 0.073
  (`paper`) to 0.766 (`scaffold_0`), and L36 for `no_chat_tags` (0.418); the C1 sweep of this checkpoint also picks L22
  ([The 4B pair (C1)](#the-4b-pair-c1)). For the five templates that steer at L22 the
  mean slope at L36 is lower (`paper` 0.030 against 0.073, `no_template` 0.318 against 0.622, `neutral_frame` 0.154
  against 0.525, `scaffold_0` 0.528 against 0.766, `concept_swap_0` 0.040 against 0.274), so the last block is not where
  their slope is largest. The steering layer is the last block for 6 of 6 templates on each base checkpoint (1.7B and 4B), for 5 of 6 on
  `qwen3-1.7b` and for 1 of 6 on `qwen3-4b`. The slope at the steering layer is measured on the prompts that chose that layer and is optimistic.
- **The `paper` template barely steers on the post-trained checkpoint.** Its mean dose slope is between -0.082 and 0.073
  over the six sweep layers, and its two steering readers differ in sign at four of them (layers 15, 22, 29 and 36); at
  its steering layer (L22) the slopes are -0.094 and 0.239, whose mean is 0.073. Eight of its 12 units are inverted,
  against 1 to 5 for each of the other templates, and its transfer slopes by prompt format are 0.120 (raw), 0.024 (chat)
  and 0.144 (story). Its read accuracy (0.763) is the lowest of the six on this checkpoint, but all of its 12 units are
  readable.
- **Reading layers.** The reading layer is L21 to L27 for five templates on the base checkpoint and L35 for
  `scaffold_0` (0.953, the highest read accuracy of the six); on the post-trained checkpoint it is L20 to L24 for all
  six. The read accuracy ranges from 0.790 to 0.953 on the base and from 0.763 to 0.919 on the post-trained checkpoint
  (three templates at 0.763 to 0.777 and three at 0.900 to 0.919). The mean absolute gap between the reading depth and
  the steering depth (layer divided by 36) is 0.287 on the base and 0.097 on the post-trained checkpoint, against 0.238
  and 0.298 on the 1.7B pair.
- **Inversion.** On the base checkpoint 6 of the 67 readable units are inverted, a rate of 0.090 [0.042, 0.182]; on the
  post-trained checkpoint 25 of 72, a rate of 0.347 [0.248, 0.462] (Wilson intervals, which do not overlap: 0.182
  against 0.248). The order of the two checkpoints is the reverse of the 1.7B pair (0.167 and 0.125, intervals that
  overlap). The counts by template are 1, 2, 1, 1, 0 and 1 on the base checkpoint and 8, 2, 5, 5, 1 and 4 on the
  post-trained one; for comparison, `gpt2-xl` has 20 of 72. Three of the five units of the base checkpoint that are not
  readable have a negative slope (-0.005, -0.126 and -0.006) and are not counted. Every inverted unit of the base
  checkpoint is at layer 1, 8 or 15 (4, 1 and 1 units). On the post-trained checkpoint the units are at layers 1, 8, 15,
  22, 29 and 36 (9, 8, 4, 1, 1 and 2), so 4 of the 25 are at the three deepest layers: `paper` at layers 22, 29 and 36
  (-0.094, -0.003 and -0.035) and `concept_swap_0` at layer 36 (-0.199). On the 1.7B pair no inverted unit is at the two
  deepest sweep layers.
- **Reading and steering.** The Spearman correlation over the 72 units between reading accuracy and dose slope is 0.721
  on the base and 0.650 on the post-trained checkpoint, and between the logit-lens score and dose slope 0.767 and 0.670.
  The sign of the slope agrees between the two judges for 0.917 of the units on the base checkpoint (66 of 72) and 0.750
  on the post-trained one (54 of 72). The sweep layer with the largest logit-lens separation is the steering layer for
  all six templates on the base checkpoint (L36) and for one on the post-trained one (`no_chat_tags`, L36); for the other
  five it is L29 (`no_template`, `neutral_frame`, `concept_swap_0`) or L36 (`paper`, `scaffold_0`) while the steering
  layer is L22. As on the 1.7B pair, both the reading accuracy and the slope vary with depth over the sweep, so a rank
  correlation over units that mixes layers can be driven by depth alone, and the summary does not give the within-layer
  correlation.
- **Prompt format.** The mean primary dose slope by steering prompt format (raw, chat and story) is 0.732 to 1.020,
  0.962 to 1.138 and 0.957 to 1.224 over the six templates on the base checkpoint, where raw is the lowest of the three
  formats for every template. On the post-trained checkpoint it is 0.120 to 0.868 for raw, 0.024 to 0.795 for chat and
  0.144 to 0.785 for story; chat is the lowest format for four templates (`paper`, `no_template`, `no_chat_tags` and
  `neutral_frame`) and story for the other two (`scaffold_0` and `concept_swap_0`). Only `scaffold_0` and `no_template`
  stay above 0.5 in all three formats (0.868, 0.795 and 0.785; 0.641, 0.535 and 0.652).
- **Random directions.** The largest absolute dose slope of a random direction at any sweep layer is 0.224 on the base
  checkpoint (layer 36, where the template slopes at the steering layer are 0.819 to 1.045) and 0.313 on the
  post-trained one (layer 22, where the template slopes are 0.073 to 0.766). The random arms are three directions with
  one reader each. On the base checkpoint the random slopes at layers 1, 8 and 15, where the inverted units sit, range
  from -0.061 to 0.107, and the inverted units go lower (down to -0.149), so a random direction gives small slopes of the
  same order as the inverted units but not all of them. On the post-trained checkpoint the random slopes range from
  -0.313 to 0.132 over the six layers (two are below -0.2: -0.313 at layer 22 and -0.241 at layer 29) and the inverted
  units from -0.199 to -0.002, so every inverted unit lies inside the range of the random slopes. The steering-layer
  slopes of two templates are below the largest random magnitude there (`paper`, 0.073, and `concept_swap_0`, 0.274);
  the other four (0.418 to 0.766) are above it, as are all six on the base checkpoint.
- **Fluency.** The cells are 288 template cells and 72 random-arm rows (360 per run); the counts below are over the
  template cells. On the base checkpoint no cell has a neutral-text perplexity above 1.5 times the unsteered one and 1
  cell has a distinct-1 below 0.15 (`paper` at layer 36 and -0.3, reader 1, 0.133; unsteered 0.348). The mean over the
  templates and readers at layer 36 is 1.305 (+0.3) and 1.340 (-0.3) times the unsteered perplexity, against 1.006 to
  1.169 at the other layers and the same two coefficients. On the post-trained checkpoint 4 cells are above 1.5 times
  (1.54 to 1.80), all at -0.3: `paper` (reader 1) and `no_chat_tags` (reader 0) at layer 22 (1.57 and 1.80) and
  `neutral_frame` (both readers) at layer 36 (1.66 and 1.54); none has a distinct-1 below 0.15 (lowest 0.201; unsteered
  0.269). The mean at layer 22 is 1.152 (+0.3) and 1.292 (-0.3) times and at layer 36 1.113 and 1.307 times, against
  1.013 to 1.152 at layers 1, 8, 15 and 29. Of the four flagged cells only `paper` at layer 22 sits at its template's
  steering layer. The largest random-arm ratio is 1.25 (base) and 1.26 (post-trained) and the lowest random-arm
  distinct-1 is 0.308 and 0.228.
- **Base against post-trained.** The slope at the steering layer is 0.819 to 1.045 on the base checkpoint and 0.073 to
  0.766 on the post-trained one, whose steering layer is an interior layer for five of six templates. The inversion rate
  is higher on the post-trained checkpoint (0.347 against 0.090, intervals that do not overlap) and the two judges agree
  on fewer units (0.750 against 0.917). Each checkpoint has one run, and the runs do not say why the post-trained
  checkpoint steers less under the same protocol.

## Run identity

All reading runs (R1 to R7) except the R4 and R6 runs of `gpt2-medium`, `gpt2-large` and `gpt2-xl` (made at `ca2546c`,
below) the R6 run of `gpt2` (re-run in chain 40 at `bab5b75`, below) and the R4, R6, A1 and A2 runs of the Qwen3
checkpoints (made at `a6e2333`, below) were made at commit `a39ae61` with a clean tree, on one laptop GPU with 8 GB of
memory, with the default config of each experiment (batch size 8 for the reading runs; the A1 and A2 configs store no
batch size). The A1 and A2 runs of `gpt2` were made at `a39ae61` as well and read the `gpt2` runs of R1, R2, R3, R5 and
R7 below. The C1 runs of `gpt2` and `gpt2-xl` were made at `d59a2f7` and those of `gpt2-medium` and `gpt2-large` at
`a39ae61`; the C1 and reading source code is unchanged between them (`git diff --stat` over `src` lists only
`p9d_steering.py` and `p9/behavior.py`). The C1 batch size is 16 for `gpt2` and `gpt2-medium` and 4 for `gpt2-large` and
`gpt2-xl`. The Qwen3 C1 runs were made at `a39ae61` with a clean tree and batch size 16 on all four checkpoints. The P1
runs of the five models were made at `a39ae61` with a clean tree, with the sentiment task only, 10 seeds of 64 training
pairs, up to 256 evaluation pairs per split (214 available) and batch size 8; the sentiment-only runs of the Qwen3 0.6B
pair stay committed and are superseded by the 3-task logistic runs of chain 40 (below). The P6 tables were recomputed on
the CPU from the finished runs at commit `ca2546c` (no model is loaded). The P8 run was recomputed on the CPU at commit
`beb5801` (clean tree, no model is loaded), after the runs it reads. The clean tree refers to the code: P8 read the run
folders from the local output root, and the runs it reads are committed under `results/`. Its config names the five P1 runs as extra runs (for the Qwen3 0.6B pair, the sentiment-only runs `a540a2497ac4` and `31a6ee65d6ab`, so the P8 numbers read those runs and not the three-task runs). Library versions and the device are in each run's `meta.json`.

The R4 and R6 runs of `gpt2-medium`, `gpt2-large` and `gpt2-xl`, and the P2, P3, P4, M1, C2, LR1, C4, P7 and C5 runs of
`gpt2` (and the C5 run `155f0f72bc3f` of `gpt2-xl`; the later runs of the larger sizes are described below) were made at
`ca2546c` with a clean tree on the same laptop GPU, with the default config of each experiment (the batch size of each
run is in its `config.json`). The P1 run `078b9d6482a2` of `gpt2` (a copy made at `ca2546c`, replaced by the re-run of
chain 40 below) differs from the sentiment run `673af340b77d` in the task list only. The C4 run reads the LR1 adapter
named in its config. None of the M1, C2, P7, C4 and C5 runs made at `ca2546c` stores the perplexity or the distinct-1 of
the generated text. The `gpt2` P5 run is the rerun after the fix of the zero-norm fit that stopped the earlier attempts;
those attempts are not exported and not reported.

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

The 8 runs of chain 17 that cover P9 on the Qwen3 4B pair (P9a to P9d on `qwen3-4b-base` and `qwen3-4b`) were made at `89ea7cb` with a clean tree on a rented GPU pod (NVIDIA RTX A6000 48 GB, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment on that commit except one field (`config.json` of each run): the batch size is 8 for P9a, P9b and P9c and 16 for P9d, and `p9b/qwen3-4b/5481dab673d4` ran at batch size 4 after the batch size 8 attempt was killed by the out-of-memory killer of the container. The batch size is a config field, so the retry has its own hash; the default P9b config for `qwen3-4b` would have the hash `ecbd8e8a1b1b`, which is the unfinished attempt and is not exported. The P9a configs keep the default `cutoff_year` (2017), where the other Qwen3 P9a runs set 2024. The time between `started_at` and `finished_at` of the stored `meta.json` is 972 s (`p9d/qwen3-4b-base/7fe658a89920`) to 36,819 s (`p9b/qwen3-4b-base/91d3fe92e1c9`). The config hashes of the eight runs were recomputed from the original `config.json` of each run with the code of `86c319b` (since `89ea7cb` only P10 files changed under `src/tomography/experiments/`) and all eight agree with the stored ones; the exported configs rewrite the `labels_run` path of P9b, P9c and P9d, so the hash of an exported config of those differs from its stored hash. Exporting these runs changed no tracked result.

The 13 runs of [the P3 and P4 section across the GPT-2 sizes and Qwen3](#stimulus-set-size-and-direction-finder-across-the-gpt-2-sizes-and-qwen3-p3-p4) (P3 on `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-0.6b-base`, `qwen3-0.6b` and `qwen3-1.7b`; P4 on `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base` and `qwen3-1.7b`) were made with a clean tree and the default config of each experiment on its commit, with batch size 8 (`config.json` of each run). The two `gpt2-medium` runs were made at `fb3b5dd` on a mini PC without a GPU (an AMD Ryzen 7 7730U CPU, which `meta.json` does not store; `device` is `cpu`, torch 2.14.1+cpu, transformers 5.19.0); the time between `started_at` and `finished_at` of the stored `meta.json` is 42,657 s for P3 (`p3/gpt2-medium/b80a5a3c0faf`) and 3,008 s for P4 (`p4/gpt2-medium/9a2f3f02568b`). The other 11 runs were made at `80baffb` on rented GPU pods (torch 2.14.1+cu130, transformers 5.19.0): ten on an NVIDIA GeForce RTX 3090 and `p4/qwen3-1.7b-base/b772e1cc4db9` on an NVIDIA L4. Their times are 518 s (`p4/qwen3-0.6b/e40fa46cf517`) to 1,464 s (`p4/qwen3-1.7b/607e86d9c7f6`). The config hashes of the 13 runs were recomputed from the stored `config.json` of each run with the code of `0a66255` and all 13 agree with the stored ones, and each stored config equals the default config of its experiment and model. Between `fb3b5dd` and `0a66255` the source files that P3 and P4 import (the import closure of `p3_size.py` and `p4_pairing.py`) differ only in `model.py` (12 added lines, two Qwen3 4B entries of the model registry). The `gpt2` P4 run was made at `ca2546c`, before P4 set the opt-in refit flag of the shared fit code: the flag repeats a logistic fit with a tighter tolerance only when the solver returns all-zero coefficients, which earlier stopped a run with a zero-norm error, so it changes no fit of a run that finished without it. The `gpt2` P3 run and the P3 run of `qwen3-1.7b-base` do not use the flag. Exporting these runs changed no tracked result.

Config hashes (12 hex digits), P9 on the Qwen3 4B pair:

| Experiment  | qwen3-4b-base | qwen3-4b                    |
| ----------- | ------------- | --------------------------- |
| P9a         | 25d7cc240fef  | 7bd160f5a7e0                |
| P9b         | 91d3fe92e1c9  | 5481dab673d4 (batch size 4) |
| P9c         | ab86a6dc49d2  | 4307bb435ab8                |
| P9d fewshot | 7fe658a89920  | c7949f5432e6                |
| P9d concept | not run       | cc4024fe5af1 (chain 39)     |


The 4 runs of chain 19 that cover P10 on the Qwen3 4B pair (P10a and P10b on `qwen3-4b-base` and `qwen3-4b`) were made at `e7295fc` (an ancestor of the current main; `git diff --stat e7295fc 0a66255 -- src` lists only `src/tomography/cloud.py`, so the P10 code is that of main) with a clean tree on a rented GPU pod (NVIDIA RTX A4500 20 GB, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment on that commit except two fields (`config.json` of each run): `offline` is true, so the questions are read from the cache, and the batch size of `p10a/qwen3-4b-base/94faa23c130d` is 4. The batch size is 8 for the other three runs. The batch size 4 run is the retry of the batch size 8 attempt of P10a on `qwen3-4b-base`, which ended with exit status 1 (out of GPU memory beside another 4B job) and is not exported; the batch size is a config field, so the retry has its own hash, and the config of the attempt would have the hash `67c1e4eacad2`. The time between `started_at` and `finished_at` of the stored `meta.json` is 138 s (`p10b/qwen3-4b/26f6e1eb9116`) to 3,540 s (`p10a/qwen3-4b/9bd03306f0a4`); two jobs shared the GPU for part of the time. The chain log records exit status 0 for the four exported runs. The config hash of each run was recomputed under main (`0a66255`) by loading its stored `config.json` into the config class of its experiment and hashing it, and all four agree with the directory names; the exported configs equal the stored ones. The export keeps the CSV and summary files of these runs and not `items.json` (the completions and the per-question scores) or `readers.npz`, so the tables of the P10 section are read from the committed files. Exporting these runs changed no tracked result.

Config hashes (12 hex digits), P10 on the Qwen3 4B pair:

| Experiment | qwen3-4b-base               | qwen3-4b     |
| ---------- | --------------------------- | ------------ |
| P10a       | 94faa23c130d (batch size 4) | 9bd03306f0a4 |
| P10b       | 8295e1190b5c                | 26f6e1eb9116 |

The 16 runs of chains 25 and 26 that cover P10 on the GPT-2 sizes (chain 25: P10a and P10b on `gpt2`, `gpt2-medium`, `gpt2-large` and `gpt2-xl`, on a Secure Cloud pod) and on the Qwen3 0.6B and 1.7B pairs (chain 26: P10a and P10b on `qwen3-0.6b-base`, `qwen3-0.6b`, `qwen3-1.7b-base` and `qwen3-1.7b`, on a Community Cloud pod) were made at `d09822b` (an ancestor of the current main; `git diff --stat e7295fc d09822b -- src` is empty and `git diff --stat d09822b 0a66255 -- src` lists only `src/tomography/cloud.py`, so the P10 code is that of chain 19 and of main) with a clean tree on two rented GPU pods (NVIDIA GeForce RTX 3090 24 GB, torch 2.14.1+cu130, transformers 5.19.0), with the default config of each experiment on that commit except one field (`config.json` of each run): `offline` is true, so the questions are read from the cache. The batch size is 8 for all 16 runs. The time between `started_at` and `finished_at` of the stored `meta.json` is 100 s (`p10b/gpt2/e04b618425f5`) to 1,718 s (`p10a/qwen3-1.7b/ed9c40d94e25`); each pod ran two jobs at a time. The chain logs record exit status 0 for every one of the 16 runs and end with `CHAIN_DONE failed=0`. The config hash of each of the 20 P10 runs, those of chain 19 included, was recomputed under main (`0a66255`) by loading its stored `config.json` into the config class of its experiment and hashing it, and all 20 agree with the directory names; the exported configs equal the stored ones. The export keeps the same files as for chain 19. Exporting these runs changed no tracked result.

Config hashes (12 hex digits), P10 on the GPT-2 sizes and the Qwen3 0.6B and 1.7B pairs:

| Checkpoint      | P10a         | P10b         |
| --------------- | ------------ | ------------ |
| gpt2            | 5af932e9547d | e04b618425f5 |
| gpt2-medium     | 989f13a6c7a0 | 82459bd192fa |
| gpt2-large      | d03b40bf6194 | f0782e792903 |
| gpt2-xl         | 6d6b168a8749 | 077222ed6502 |
| qwen3-0.6b-base | 5ecb5a5fcdea | 09ceada2ace4 |
| qwen3-0.6b      | aa3e0544c2a8 | 8d0a0478fe82 |
| qwen3-1.7b-base | 75a098424d14 | 9756eed4339d |
| qwen3-1.7b      | ed9c40d94e25 | fd69e18dc1cc |

The four P1 runs on the Qwen3 1.7B pair of the subsection on the paper's finder (logistic and `pca_diff`, each with the
sentiment, truth and emotion tasks) were made at `f4d414d` with a clean tree on the laptop GPU (NVIDIA GeForce RTX 4070
Laptop GPU, torch 2.14.1+cu130, transformers 5.19.0), from one queue, with the default P1 config of that commit (10
seeds of 64 training pairs, 256 evaluation pairs per split, batch size 8, last token; `config.json` of each run) and
`method=pca_diff` for the two `pca_diff` runs. The time between `started_at` and `finished_at` of the stored `meta.json`
is 3,340 s (`p1/qwen3-1.7b/f3a2d832c9c4`) to 5,059 s (`p1/qwen3-1.7b-base/a45dc5c93f18`), and the queue log records exit
status 0 for each. The queue started the logistic run on `qwen3-1.7b-base` twice (the first start has no end in the
log); the exported run is the second start, and its folder holds only files written after it. The earlier attempt is not
exported. The config hash of each run was recomputed under main by loading its stored `config.json` into the P1 config
class and hashing it; all four agree with the directory names. The two logistic runs differ from the sentiment-only runs
`ef3f8c0d3389` and `f4aa9b57824a` (made at `a39ae61`) in the task list only, and their sentiment rows are identical to
those runs; the code of P1, the sweep and the readers differs between `a39ae61` and `f4d414d` only by an option
(`refit_at_zero`) that is off by default. Exporting these runs changed no tracked result.

The two P1 runs on `gpt2-large` (logistic `p1/gpt2-large/4920a2c4975e` on pod 36 and `pca_diff`
`p1/gpt2-large/7e5f079053ef` on pod 37) were made at `80baffb` with a clean tree on two rented GPU pods (Secure Cloud
NVIDIA GeForce RTX 4090, torch 2.14.1+cu130, transformers 5.19.0, Linux), with the default P1 config of that commit
(sentiment, truth and the six emotions, 10 seeds of 64 training pairs, 256 evaluation pairs per split, batch size 8,
last token; `config.json` of each run) and `method=pca_diff` for the second; the two configs differ in `method` only.
The time between `started_at` and `finished_at` of the stored `meta.json` is 2,107 s for the logistic run and 3,671 s
for the `pca_diff` run. The P1 code path (the import closure of `p1_templates.py`) is identical at `f4d414d` and
`80baffb`, except that `model.py` gained two Qwen3 4B entries of the model registry (12 added lines). The config hash of
each run was recomputed under main by loading its stored `config.json` into the P1 config class and hashing it, and both
agree with the directory names. Exporting these runs changed no tracked result.

Chain 40 (Community 3090): P1 on gpt2, gpt2-medium and the Qwen3 0.6B pair, R6 on gpt2. The nine runs were made at
`bab5b75` with a clean tree on one rented GPU pod (Community Cloud NVIDIA GeForce RTX 3090, torch 2.14.1+cu130,
transformers 5.19.0, Linux), with the default config of each experiment on that commit (P1: sentiment, truth and the six
emotions, 10 seeds of 64 training pairs, 256 evaluation pairs per split, batch size 8, last token, `method=logistic` or
`pca_diff`; R6: batch size 8; `config.json` of each run). The chain log records exit status 0 for every job and ends
with `CHAIN_DONE failed=0`; the pod ran two jobs at a time, so the times overlap. The time between `started_at` and
`finished_at` of the stored `meta.json` is 589 s (`p1/gpt2/078b9d6482a2`) to 2,008 s (`p1/qwen3-0.6b-base/61b62dc50aaa`)
for P1 and 158 s for R6. Seven of the runs are new: P1 on `gpt2-medium` (`ac7c6af28fc3` logistic and `bc97021c8520`
`pca_diff`), on `qwen3-0.6b-base` (`deabd95764d0` and `61b62dc50aaa`), on `qwen3-0.6b` (`027be280838a` and
`820718d9b28a`) and the `pca_diff` run on `gpt2` (`cce69c9048fd`). The other two have the config hash of a committed run
and replace it: the P1 run of `gpt2` `078b9d6482a2` (sentiment, truth and emotion tasks; the committed copy was made at
`ca2546c`) and the R6 run of `gpt2` `47e8ec0cf46c` (the committed copy was made at `a39ae61`). The re-run of P1 agrees
with the committed copy in every accuracy, interval and layer of `summary.json` and `table.json`; only the random-direction series (by at most 0.0017) and the cosines (by at most 4e-6) moved, one stored reader accuracy of a curves file moved by 0.0023 (one test item of 44 for one seed, in `curves_emotion_sadness.npz` and its wording copy, at a layer other than the selected one), and one printed number changed: the
random series of the truth variant `scaffold_1` reads 0.547 instead of 0.546, and with it the smallest margin of a truth
variant over its random series reads 0.061 instead of 0.063. The re-run of R6 agrees with the committed copy in every
stored row and layer and adds the two text baselines that the committed copy did not store (`text_word` 0.676 [0.603,
0.742] and `text_char` 0.671 [0.597, 0.736]), which are the values this document had computed with the same estimator.
The config hash of each run was recomputed under main by loading its stored `config.json` into the config class of its
experiment and hashing it, and all nine agree with the directory names. Between `a39ae61` or `ca2546c` and `bab5b75` the
source files that P1 imports differ only in an option of the fit code that P1 does not use (`refit_at_zero`, off by
default) and in the 12 lines of `model.py` (two Qwen3 4B entries of the model registry); the source of R6 differs from `a39ae61` by the same option, the 12 lines of `model.py`, and the code that adds the text baselines (with the text analyzer constants of `p9/risk.py`). Three cells of the R6 curve files (`curves_arc.npz` and `curves_honesty.npz`) moved in the `gpt2` re-run, by at most 0.0004 in the mean (one test item in one seed). The runs do not say what moved the random series of the `gpt2` run, or the sentiment rows of the 0.6B runs against the sentiment-only runs of `a39ae61` (`a540a2497ac4` and `31a6ee65d6ab`: report accuracy within 0.012 per variant, the random series by up to 0.035, the interval bounds by up to 0.013, `own_report` by up to 0.013, the cosines by up to 0.009, and the selected own layer of 4 or 5 variants): the code on the P1 path differs only by the unused option and the registry entries, and device, library build and solver are not separated by the files. Exporting these runs changed no tracked result
apart from the two replaced runs.

Config hashes (12 hex digits), the P1 follow-up:

| Experiment               | gpt2         | gpt2-medium  | gpt2-large   | gpt2-xl      | qwen3-0.6b-base | qwen3-0.6b   | qwen3-1.7b-base | qwen3-1.7b   |
| ------------------------ | ------------ | ------------ | ------------ | ------------ | --------------- | ------------ | --------------- | ------------ |
| P1 (all tasks, logistic) | 078b9d6482a2 | ac7c6af28fc3 | 4920a2c4975e | 006a40a28414 | deabd95764d0    | 027be280838a | 62073cecf6d5    | f3a2d832c9c4 |
| P1 (all tasks, pca_diff) | cce69c9048fd | bc97021c8520 | 7e5f079053ef | d11b2daa837b | 61b62dc50aaa    | 820718d9b28a | a45dc5c93f18    | 7ffb64cce683 |

The rows `P1 (sentiment)` and `P1 (all tasks)` of the GPT-2 table below, and the row `P1` of the Qwen3 tables, are the
sentiment-only runs and the `gpt2` logistic run; the logistic and `pca_diff` runs of every model with all three tasks
are in the table above.

Chain 41 (P1 on gpt2-xl): the logistic run `p1/gpt2-xl/006a40a28414` and the `pca_diff` run `p1/gpt2-xl/d11b2daa837b`
were made at `bab5b75` with a clean tree on one rented GPU pod (NVIDIA GeForce RTX 3090, torch 2.14.1+cu130,
transformers 5.19.0, Linux), with the default P1 config of that commit (sentiment, truth and the six emotions, 10 seeds
of 64 training pairs, 256 evaluation pairs per split, batch size 8, last token; `config.json` of each run); the two
configs differ in `method` only. The time between `started_at` and `finished_at` of the stored `meta.json` is 2,948 s
for the logistic run and 2,440 s for the `pca_diff` run; the pod ran M1 jobs of the same chain beside them, so the times
overlap those of other jobs. The chain log records exit status 0 for each of its five jobs and ends with `CHAIN_DONE
failed=0`. The config hash of each run was recomputed under main by loading the config of the pod copy (identical to the
exported one) into the P1 config class and hashing it, and both agree with the directory names. The source files that P1
imports are those of main at `bab5b75`: `git diff --stat bab5b75 HEAD -- src` is empty. Exporting these runs changed no
tracked result.

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
| P1 (all tasks) | 078b9d6482a2 | ac7c6af28fc3 | 4920a2c4975e | 006a40a28414 |
| P2             | ee099a704022 | 733bd2f76315 | 3ea9b00fe28f | feafacc97cbb |
| P3             | 676c31488397 | b80a5a3c0faf | 327dd31a1de6 | 12ee53fd16fe |
| P4             | 8c4a73d1e9db | 9a2f3f02568b | f4e8481cbbbf | 427a6b2f9da4 |
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
| P3          | 776d454d5ca2    | b38783812e71 | 61baf30a8a59    | 72edcdd9158e |
| P4          | 3b2677205f7d    | e40fa46cf517 | b772e1cc4db9    | 607e86d9c7f6 |
| P5          | b6a702e460e9    | fb664c3b4be2 | a4bd463deeca    | 7e5e579d102e |
| P7          | not run         | not run      | 16a7a9e4facf    | eacde3abd0aa |
| C1          | 7ae3e2aa34e2    | 832462b7b04e | 0cd5a36d3408    | a91f950ad9f5 |
| M1          | not run         | not run      | e9279a87db07    | 77752112ddbe |
| C2          | not run         | not run      | 4133e19728fa    | 1b87ff98a318 |
| C4          | b08c4e236dcf    | 4d4ff337ade5 | 7dd49d64a540    | 5ebc35ea3be6 |
| C5          | not run         | not run      | 3a14e656df3c    | 52057b2ddc30 |
| LR1         | 03ba155a9fe7    | 74d01330d28c | 95a42d94146c    | bf86ed5ccde1 |
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
of its size. Exporting these runs changed no tracked result. The runs that were cut or never started are not exported; those of the 4B pair were made later, in chains 20 to 24 below.

Config hashes of the chain 17 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base      | qwen3-4b     |
| ---------- | ------------------ | ------------ |
| R1         | 705ebc7745d4       | 15a1ed3eab2b |
| R2         | bb176283725a       | a6ab527e0544 |
| R3         | a6e29d000606       | 039d5c50bb27 |
| R4         | dd2c7d3f2fb2       | 3e8c537bfac8 |
| R5         | d692205ca9bf       | 1630de90cae8 |
| R6         | cut (see chain 20) | 4e4a60094e47 |
| R7         | 15d7a0c1017f       | 9f28dde7631f |
| A1         | 80e2777e6b67       | 819f34624d6a |
| A2         | 8edf53991111       | 2a18226fa458 |
| C2         | cut (see chain 23) | 1df7c1f9ff70 |
| LR1        | 22e472aa6c00       | f0d81c80e9bc |

The three larger GPT-2 sizes:

| Experiment | gpt2-medium  | gpt2-large   | gpt2-xl      |
| ---------- | ------------ | ------------ | ------------ |
| LR1        | c6bd21c82ddc | 5f6fdb0ccda5 | f54173153b29 |
| C4         | 88cc5d7f6dba | 43bf8077183e | 321f3da56ea5 |

Chain 20, R6, P2, P5 and P7 on the Qwen3 4B pair (7 runs). `r6/qwen3-4b-base/913301b5547b`,
`p2/qwen3-4b-base/64228009c695`, `p2/qwen3-4b/bf265bb382ea`, `p5/qwen3-4b-base/9c3330f88a67`,
`p5/qwen3-4b/7ac5d1bc7189`, `p7/qwen3-4b-base/7d0197f7c9bf` and `p7/qwen3-4b/be6b6b4f3b93` were made at `86c319b` with a
clean tree on a rented GPU pod (NVIDIA RTX A4000, torch 2.14.1+cu130, transformers 5.19.0), with the default config of
each experiment on that commit (`config.json` of each run). The batch size is 8 for R6, P2 and P5 and 16 for P7. The
time between `started_at` and `finished_at` of the stored `meta.json` is 1,147 s for R6, 1,336 s and 1,421 s for P2, 583
s and 568 s for P5 and 6,235 s and 6,168 s for P7 (base checkpoint first). The chain log records exit status 0 for each
of the seven runs; it then records the interruption of the next job, C2 on `qwen3-4b-base`, and ends there. That
interrupted run is not exported (the reported C2 run of that checkpoint is the one of chain 23, below), and the
remaining jobs of the chain did not start. Since `86c319b` the only change under `src` is in `cloud.py`
(pod tooling), so the code of these experiments is that of main. The config hash of each run was recomputed under main
by loading its stored `config.json` into the config class of its experiment and hashing it; all seven agree with the
directory names, and each also equals the hash of the default config of its checkpoint. The R6 run of `qwen3-4b`
(`4e4a60094e47`) is from chain 17. Exporting these runs changed no tracked result.

Config hashes of the chain 20 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base | qwen3-4b                |
| ---------- | ------------- | ----------------------- |
| R6         | 913301b5547b  | in chain 17 (see above) |
| P2         | 64228009c695  | bf265bb382ea            |
| P5         | 9c3330f88a67  | 7ac5d1bc7189            |
| P7         | 7d0197f7c9bf  | be6b6b4f3b93            |

Chain 21, C4 on the Qwen3 4B pair (2 runs). `c4/qwen3-4b-base/dd918686b022` and `c4/qwen3-4b/a079b7a25b4f` were made at
`d09822b` with a clean tree on a rented GPU pod (NVIDIA GeForce RTX 3090, torch 2.14.1+cu130, transformers 5.19.0), with
the default config of the experiment on that commit (`config.json` of each run) and batch size 16, except the
`lorra_run` field, which names the LR1 run of the checkpoint (`lr1/qwen3-4b-base/22e472aa6c00` and
`lr1/qwen3-4b/f0d81c80e9bc`, chain 17, above), uploaded to the pod. The time between `started_at` and `finished_at` of
the stored `meta.json` is 2,907 s for the base checkpoint and 3,279 s for the post-trained one. The chain log records
exit status 0 for both runs. Since `d09822b` the only change under `src` is in `cloud.py` (pod tooling), so the code of
this experiment is that of main. The config hash of each run was recomputed under main by loading its stored
`config.json` into the C4 config class and hashing it, and both agree with the directory names; they differ from the
hash of the default C4 config of their checkpoint only through `lorra_run`. The exported config rewrites `lorra_run` to
`results/lr1/<model>/<hash>`, so the hash of an exported config is not its directory name. Exporting these runs changed
no tracked result (the two LR1 runs were exported again with them and are unchanged).

Config hashes of the chain 21 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base | qwen3-4b     |
| ---------- | ------------- | ------------ |
| C4         | dd918686b022  | a079b7a25b4f |

Chain 22, M1 on the Qwen3 4B pair (2 runs). `m1/qwen3-4b-base/5b21f57abf27` and `m1/qwen3-4b/d490d3bf381b` were made at
`d09822b` with a clean tree on a rented GPU pod (NVIDIA GeForce RTX 3090, torch 2.14.1+cu130, transformers 5.19.0), with
the default config of the experiment on that commit (`config.json` of each run) and batch size 16. The time between
`started_at` and `finished_at` of the stored `meta.json` is 3,955 s for the base checkpoint and 3,924 s for the
post-trained one. The chain log records exit status 0 for both runs. Since `d09822b` the only change under `src` is in
`cloud.py` (pod tooling), so the code of this experiment is that of main. The config hash of each run was recomputed
under main by loading its stored `config.json` into the M1 config class and hashing it, and both agree with the
directory names and with the hash of the default M1 config of their checkpoint. Exporting these runs changed no tracked
result.

Config hashes of the chain 22 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base | qwen3-4b     |
| ---------- | ------------- | ------------ |
| M1         | 5b21f57abf27  | d490d3bf381b |

Chain 23, C2 and C5 on the Qwen3 4B pair (3 runs). `c2/qwen3-4b-base/984485501f3e`, `c5/qwen3-4b-base/3d68f877eed5` and
`c5/qwen3-4b/fd207b9d8079` were made at `d09822b` with a clean tree on a rented GPU pod (NVIDIA GeForce RTX 3090, torch
2.14.1+cu130, transformers 5.19.0), with the default config of each experiment on that commit (`config.json` of each
run) and batch size 16, except two fields of the C5 runs: `n_pool` is 1,000 (default 200) and `c2_run` names the C2 run
of the checkpoint. The time between `started_at` and `finished_at` of the stored `meta.json` is 2,185 s for C2 and 1,412
s and 755 s for C5 (base checkpoint first). The chain log records exit status 0 for all three runs. The C5 run of each
checkpoint reads the C2 run of that checkpoint: `c2/qwen3-4b-base/984485501f3e`, made in this chain, and
`c2/qwen3-4b/1df7c1f9ff70` (chain 17, above), which was uploaded to the pod. The exported config of
`c5/qwen3-4b/fd207b9d8079` names that C2 run as `results/c2/qwen3-4b/1df7c1f9ff70`. The run was first exported without
the C2 run in the copy root, which wrote `<external>/1df7c1f9ff70` (the exporter rewrites a path only for runs inside
the exported roots), and was exported again with it; the C2 run was exported along with it and is unchanged. The C2 run of
`qwen3-4b-base` has the same config hash as a C2 run of the same checkpoint that was cut in chain 17 and is not
exported; the run reported here is the finished one of this chain. Since `d09822b` the only change under `src` is in
`cloud.py` (pod tooling), so the code of these experiments is that of main. The config hash of each run was recomputed
under main by loading its stored `config.json` into the config class of its experiment and hashing it. The C2 run
agrees with its directory name and with the hash of the default C2 config of its checkpoint. For the two C5 runs the
hash of the unmodified pod config equals the directory name, while the exported config differs from it in the `c2_run`
path, so the hash of an exported C5 config is not its directory name. Exporting these runs changed no tracked result.

Config hashes of the chain 23 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base | qwen3-4b                |
| ---------- | ------------- | ----------------------- |
| C2         | 984485501f3e  | in chain 17 (see above) |
| C5         | 3d68f877eed5  | fd207b9d8079            |

Chain 24, C1 on the Qwen3 4B pair (2 runs). `c1/qwen3-4b-base/72a965626cd7` and `c1/qwen3-4b/50671febc316` were made at
`d09822b` with a clean tree on a rented GPU pod (NVIDIA GeForce RTX 3090, torch 2.14.1+cu130, transformers 5.19.0), with
the default config of the experiment on that commit (`config.json` of each run) and batch size 16, as for the other
Qwen3 C1 runs. The time between `started_at` and `finished_at` of the stored `meta.json` is 1,976 s for the base
checkpoint and 1,844 s for the post-trained one. The chain log records exit status 0 for both runs. Since `d09822b` the
only change under `src` is in `cloud.py` (pod tooling), so the code of this experiment is that of main. The config hash
of each run was recomputed under main by loading its stored `config.json` into the C1 config class and hashing it, and
both agree with the directory names; both also equal the hash of the default C1 config of their checkpoint. Exporting
these runs changed no tracked result.

Config hashes of the chain 24 runs (12 hex digits), the Qwen3 4B pair:

| Experiment | qwen3-4b-base | qwen3-4b     |
| ---------- | ------------- | ------------ |
| C1         | 72a965626cd7  | 50671febc316 |

Chain 38, P10c on all ten checkpoints (10 runs). The runs `p10c/<model>/<hash>` of [Links to knowledge and hedging (P10c)](#links-to-knowledge-and-hedging-p10c) were made at `bab5b75` (the merge of pull request 199, an ancestor of the current main; `git diff --stat bab5b75 HEAD -- src` is empty, so the P10c code is that of main) with a clean tree on one rented GPU pod (Secure Cloud, NVIDIA RTX A5000, torch 2.14.1+cu130, transformers 5.19.0), with the default config of the experiment on that commit except the three run paths (`p10b_run`, `knowledge_run` and `behavior_run`, relative paths of the full P10b, P9b and P10a run directories uploaded to the pod, which are part of the config hash). The batch size is 8, `n_boot` 1,000, `n_random` 200 and `offline` true. The time between `started_at` and `finished_at` of the stored `meta.json` is 39 s (`gpt2`) to 599 s (`gpt2-xl`); the pod ran three jobs at a time on 2026-10-18, and its lane log records exit status 0 for every one of the ten runs and ends with `P10C_DONE`. The 30 input run directories named in the configs (`results/p10b/...`, `results/p9b/...` and `results/p10a/...`) all exist on main as exports; the full directories that the runs read hold `readers.npz` and `items.json`, which the export drops, so the runs cannot be rerun from the committed files alone. The config hash of each run was recomputed under main by loading its stored `config.json` into the config class and hashing it, and all ten agree with the directory names; the exported configs equal the stored ones. Exporting these runs changed no tracked result.

Config hashes (12 hex digits), P10c:

| Checkpoint      | P10c         |
| --------------- | ------------ |
| qwen3-0.6b-base | 34b9801a5197 |
| qwen3-0.6b      | c78c96b2a2e5 |
| qwen3-1.7b-base | 179a832720ce |
| qwen3-1.7b      | 2a658b6f4424 |
| qwen3-4b-base   | 490728d9b67c |
| qwen3-4b        | 6fff402ec3db |
| gpt2            | 1c68fce5b1cb |
| gpt2-medium     | e4f6c879a3e9 |
| gpt2-large      | cee50698ab52 |
| gpt2-xl         | aec7dcc56451 |

Chain 39 (Secure A5000): P3, P4 and P9d concept on the Qwen3 4B pair (5 runs). `p3/qwen3-4b-base/44efe2f36376`, `p3/qwen3-4b/3f8cb6317de0`, `p4/qwen3-4b-base/346291dfa5d1`, `p4/qwen3-4b/36371d16b354` and `p9d/qwen3-4b/cc4024fe5af1` (the `concept` template) were made at `bab5b75` (the merge of pull request 199, an ancestor of the current main; `git diff --stat bab5b75 HEAD -- src` is empty, so their code is that of main) with a clean tree on one rented GPU pod (NVIDIA RTX A5000, torch 2.14.1+cu130, transformers 5.19.0; the launcher log outside the repository names it a Secure Cloud pod). The P3 and P4 runs use the default config of their experiment and model; the P9d run uses the default config of P9d for `qwen3-4b` with two fields set (`config.json`): `template` is `concept`, and `labels_run` is the path of the P9a run `p9a/qwen3-4b/7bd160f5a7e0` on the pod, which is part of the config hash (the export rewrites it, so the exported config of that run hashes differently from its stored one). The batch size is 8 for P3 and P4 and 16 for P9d. The source files that these experiments import differ from those of the earlier runs of the same experiments only by lines added to `cloud.py`, `chain.py` and the P10 files (`git diff --stat 80baffb bab5b75 -- src` lists only `cloud.py` and `p10c_links.py`; `git diff --stat 89ea7cb bab5b75 -- src` lists only added lines). The time between `started_at` and `finished_at` of the stored `meta.json` is 2,909 s (`p3/qwen3-4b-base/44efe2f36376`) to 3,025 s (`p3/qwen3-4b/3f8cb6317de0`) for P3, 1,369 s (base) and 1,461 s (post-trained) for P4, and 993 s for P9d; the pod ran two jobs at a time on 2026-10-18, and the chain log of the launcher, which is outside the repository, records exit status 0 for every one of the five runs and ends with `CHAIN_DONE failed=0`. The config hash of each run was recomputed under main by loading its stored `config.json` into the config class of its experiment and hashing it, and all five agree with the directory names. The four P3 and P4 hashes reproduce from the committed `config.json` files; the P9d hash was checked from the original pod config, because the exported `config.json` rewrites `labels_run` and hashes to `f2e05309877a`. Exporting these runs changed no tracked result.

Config hashes (12 hex digits), chain 39:

| Experiment  | qwen3-4b-base | qwen3-4b     |
| ----------- | ------------- | ------------ |
| P3          | 44efe2f36376  | 3f8cb6317de0 |
| P4          | 346291dfa5d1  | 36371d16b354 |
| P9d concept | not run       | cc4024fe5af1 |

Chain 43 (Secure A5000): LR1 and C4 on the Qwen3 0.6B pair (4 runs). `lr1/qwen3-0.6b-base/03ba155a9fe7`,
`lr1/qwen3-0.6b/74d01330d28c`, `c4/qwen3-0.6b-base/b08c4e236dcf` and `c4/qwen3-0.6b/4d4ff337ade5` were made at `bab5b75`
(the merge of pull request 199, an ancestor of the current main; `git diff --stat bab5b75 HEAD -- src` is empty, so
their code is that of main) with a clean tree on one rented GPU pod (NVIDIA RTX A5000, torch 2.14.1+cu130, transformers
5.19.0), with the default config of the experiment and model (`config.json` of each run) and batch size 16, except the
`lorra_run` field of the C4 runs. The pod config of each C4 run records `lorra_run` as the path of the LR1 run of the
same checkpoint in the same chain, `/workspace/chains/../out/chain43/lr1/<model>/<hash>`; the export rewrites it to `results/lr1/qwen3-0.6b-base/03ba155a9fe7`
and `results/lr1/qwen3-0.6b/74d01330d28c`, so the stored `lorra_run` of each C4 run names the committed LR1 run of
its checkpoint. The time between `started_at` and `finished_at` of the stored `meta.json` is 126 s (base) and 124 s
(post-trained) for LR1, and 1,225 s (base) and 1,497 s (post-trained) for C4, and the four runs ran back to back on
2026-10-18 (LR1 base 22:01:23 to 22:03:29, LR1 post-trained 22:04:13 to 22:06:17, C4 base 22:06:59 to 22:27:24, C4
post-trained 22:28:04 to 22:53:01 UTC).
The config hash of each run was recomputed by loading its original pod `config.json` into the config class of its
experiment and hashing it, and all four agree with the directory names; the two LR1 configs equal the default LR1
config of their checkpoint, and the C4 configs differ from the default C4 config only through `lorra_run`. The two LR1
hashes also reproduce from the committed `config.json`; the exported config of a C4 run hashes differently because of
the rewritten `lorra_run`, as for the chain 14, 17 and 21 C4 runs. Exporting these runs changed no tracked result.

Config hashes (12 hex digits), chain 43:

| Experiment | qwen3-0.6b-base | qwen3-0.6b   |
| ---------- | --------------- | ------------ |
| LR1        | 03ba155a9fe7    | 74d01330d28c |
| C4         | b08c4e236dcf    | 4d4ff337ade5 |

Other runs: P6 `25ee6de331d4`; P8 `e9f9bc04ba3c`.

The files of every run named here are committed under `results/<experiment>/<model>/<hash>/`, copied with
`tomography export`, which keeps the JSON, NPZ and CSV files the plots read and rewrites local paths. The figures are
not committed; `tomography plot all --out results` redraws them from these files.

## Pending

Not in this document yet. No dates are promised.

- P1 on the Qwen3 4B pair with both finders and the sentiment, truth and emotion tasks (it ran on the eight models of
  the P1 subsection).
- P7 on `gpt2-medium` and on the two 0.6B Qwen3 checkpoints (it ran on `gpt2`, `gpt2-large`, `gpt2-xl` and the 1.7B and 4B pairs). M1
  and C2 on `gpt2-medium` and the two 0.6B Qwen3 checkpoints (they ran on `gpt2`, `gpt2-large`, `gpt2-xl` and the 1.7B
  and 4B pairs). C5 on `gpt2`, `gpt2-medium` and the two 0.6B Qwen3 checkpoints (it ran on `gpt2-large`, `gpt2-xl` and the 1.7B and 4B
  pairs).
- P5, persona wording and instruction confounds for honesty, with persona wordings beyond the five pairs of the run.
- A random-direction arm for C2 and P7 on `gpt2`, a fluency measure (perplexity and distinct-1) for M1, C2 and P7 on
  `gpt2`, and a random-direction arm and the perplexity of neutral text for C4 on `gpt2`. The runs on `gpt2-large` and
  `gpt2-xl` (M1, C2, P7, C5), the C4 runs on `gpt2-medium`, `gpt2-large` and `gpt2-xl`, the runs on the Qwen3
  1.7B pair (M1, C2, P7, C4, C5), the runs on the Qwen3 4B pair (M1, C2, C4, C5, P7) and the C4 runs on the Qwen3 0.6B pair have them; the C4 scoring is not generation, so no C4 run has a distinct-1.
- Stored but not read in this document: the emotion direction matrix and embedding of the Qwen3 R4 runs, the A1 matrices
  of concept and layer agreement, the emotion gallery of the Qwen3 C5 runs, and the P5 curves over layers and the
  cosine curves between the paper reader and the other readers.
- P9: a hand check of the abstention indicator of P9d (the stored samples carry no human labels), an item-level
  interval for the P9b AUROC (only the interval over the pair draws is stored), and a within-relation increment of
  the probe over the output confidence in P9c (only the pooled increment is stored).
- P10: a hand check of the answer parser (the `human_*` columns of `hand_check.csv` and `hand_check_samples.csv` are empty in every P10a run), and P10d (steering toward hedging); P10d has not run. The animacy and wording confound is controlled by the people heights for the height family only, with no control of the same kind for gestation periods and wingspans, and P10 ran on one set of questions and one seed.
- A5, reading directions against Qwen-Scope SAE decoders.

