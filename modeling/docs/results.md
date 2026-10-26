# First baseline: actual development results

Run: `modeling/artifacts/baseline-v1/`. Source manifest:
`2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`.

Only the 3,716 training and 467 validation statements were loaded for modeling.
Test identity/count and split integrity were checked structurally; no test target
frequencies, lengths, examples or prediction metrics were used for decisions.
The existing 465 test records remain reserved for a separately invoked final run.

## Training-only readiness findings

- 35 observed tag types; 29 meet the fixed 30-positive/30-negative training rule.
- No empty/malformed training targets or unsupported tag names were found.
- Common tags are greedy (1,369), math (1,249), implementation (1,057), DP (885)
  and data structures (757). DSU has 147 and binary search 446 assignments.
- Training statement lengths: median 369 whitespace tokens, P95 634, maximum
  1,258. The corresponding character counts are median 2,023.5, P95 3,504.5,
  maximum 6,392. Complete training distributions and co-occurrence counts are
  in `artifacts/audit/label-audit.json`.
- 597 training statements (16.1%) exceed 512 whitespace tokens; 3 exceed 1,024.
  These are proxies only. Math-heavy subword tokenization may behave differently;
  exact encoder truncation must be measured with the chosen tokenizer using
  training data. No statement truncation is applied by this TF-IDF baseline.

## Configuration and selection

Word TF-IDF, unigrams/bigrams, min_df=2, max_df=0.98, maximum 40,000 features,
sublinear term frequency and L2 normalization. Single-character word tokens are
retained so variables are not discarded mechanically. No stop-word list, label
text, editorials, code, IDs or metadata is added to feature text. All vocabulary
and IDF fitting uses training statements only; all matrices remain sparse CSR.

One-vs-rest logistic regression uses liblinear, seed 42, tolerance 1e-4 and
max_iter=1,000. Validation compared C in {1,3,10} and class_weight in {None,balanced}.
The selected candidate was **C=3, class_weight=balanced**. Balanced class weights
are computed by each binary estimator from its training assignments only.

Each candidate selected a global threshold on the fixed grid 0.10–0.80, maximizing
validation micro F1 (macro F1, proximity to 0.5 and deterministic ordering break
ties). After candidate selection, supported labels individually maximize
validation F1 on the same grid. Labels with fewer than five validation positives
or negatives use the selected global threshold. The exact thresholds, fallback
reasons and grid results are saved in `run.json`.

## Validation performance

| Measure | Actual value |
|---|---:|
| Micro F1 | **0.549974** |
| Macro F1 | **0.474177** |
| DSU precision / recall / F1 | 0.307692 / 0.444444 / **0.363636** |
| DSU validation support | 27 |
| DSU threshold | 0.30 |
| Binary-search precision / recall / F1 | 0.344538 / 0.554054 / **0.424870** |
| Binary-search validation support | 74 |
| Binary-search threshold | 0.45 |
| Full development run | 24.86 seconds |
| Initial TF-IDF fit/cache write | 3.18 seconds |
| Selected estimator fit | 3.62 seconds |
| Batch inference, 467 statements | 0.208 seconds |
| Batch mean per statement | 0.446 milliseconds |

Timings are from the actual run on this machine and are not latency guarantees.
The per-statement figure is a batch mean, not cold-start CLI latency. Full per-
label precision/recall/F1/support is in `validation-report.md` inside the run.
FFT, flows, graph matchings and matrices have F1=0 on their very small validation
supports (3–4 each). Macro performance exposes these weak labels rather than
silently dropping them from evaluation.

## Principal observed-tag errors

DSU has **27 false positives and 15 false negatives** against observed tags.
High-scoring untagged examples include `codeforces:1805:D` A Wide, Wide Graph
(score 0.870), `codeforces:1915:G` Bicycles (0.740), and `codeforces:1817:B`
Fish Graph (0.729). Their graph vocabulary suggests a lexical bias. Whether DSU
is actually unsuitable is unreviewed; platform-tag absence is not definitive.

DSU misses include `codeforces:1922:D` Berserk Monsters (0.031),
`codeforces:1895:E` Infinite Card Game (0.059), and `codeforces:1845:D`
Rating System (0.063). These examples express tasks through monsters/cards/
ratings rather than explicit component language, consistent with weak semantic
generalization by a bag-of-words baseline. This is an interpretation of observed
errors, not a completed editorial-based review.

Binary search has **78 false positives and 33 false negatives**. Untagged examples
include `codeforces:1921:D` Very Different Array (0.887),
`codeforces:1903:D2` Maximum And Queries (0.797), and `codeforces:1881:F`
Minimum Maximum Distance (0.781). Optimization/array wording can trigger the
label without checking an appropriate monotone predicate.

Misses include `codeforces:1861:F` Four Suits (0.117),
`codeforces:1846:E2` Rudolf and Snowflakes (0.135), and
`codeforces:1805:C` Place for a Selfie (0.137): the relevant structure is obscured
by card, geometric or mathematical surface descriptions. Again, these are
unreviewed interpretations, not new positive/negative annotations.

Every label has saved FP/FN counts and representative examples in
`validation-errors.json`. Scores are independent, **uncalibrated** logistic
scores. They are not calibrated confidence values and need not sum to one.

The same validation set selected hyperparameters and thresholds. These results
are therefore development estimates and may be optimistic. No final test score
is reported. The run and threshold configuration are frozen before any test use.

## Recommended next experiment

Compare the frozen word baseline against a word-plus-character TF-IDF variant
with training-only cross-validation for C/class weighting, followed by one
validation threshold choice. This is an inexpensive way to test sensitivity to
mathematical notation and lexical variants. Keep the final test untouched until
the experimental choice is frozen. Follow that with human review of the DSU and
binary-search contrast/error samples before using generated fine labels.

Hardware inspection found 16 logical CPUs, about 15.26 GiB system RAM and an
RTX 3050 Laptop GPU with 4 GiB VRAM (driver 581.80). Only about 2.14 GiB host RAM
was free at inspection. This milestone used one CPU worker and one numerical
thread. No neural library/model was downloaded or trained. Future encoder work
should measure training-token lengths and memory using small batches first;
the current inspection does not justify a large neural fine-tuning job.
