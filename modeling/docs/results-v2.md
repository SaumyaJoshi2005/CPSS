# Classifier audit and representation comparison — 26 October 2026

All numbers below are actual runs on the pinned frozen corpus, using the same
29 original-tag targets. The final 465-problem test split remains reserved.
Neither new representation meets the rule written before these experiments.
Keep baseline-v1 as the existing reference; no default or frozen artifact was
replaced. This is a negative experiment result, not evidence of a broken model.

## Audit findings

No historical prediction, alignment, feature-leakage or evaluation bug was
confirmed. The audit reproduced baseline-v1's saved micro/macro and all 29
per-label metrics and FP/FN counts exactly. It independently counted training
document frequencies to reproduce the persisted IDF, checked estimator
regularization/class-weight parameters against saved configuration, matched
training/validation row IDs and target column order, checked sparse dimensions,
and compared single-statement inference with batch scores. Source inspection
confirmed that only `Record.statement` crosses the classifier feature boundary.
The existing corpus verifier and split/alias/duplicate checks are reused.

Baseline-v1's known limitation remains: validation selected both hyperparameters
and thresholds. Its stronger validation score does not establish superior unseen
generalization against a protocol selected entirely from training data. Tiny
validation supports, missing upstream tags and zero label F1 are not by themselves
implementation errors. No original tags were corrected or reinterpreted in the
reported metrics.

The historical directory's complete file inventory, including its existing
validation-evaluation file, is hashed in `frozen-plan.json`. Every experiment
entry point checks that inventory. The audit writes elsewhere and never invokes
the historical evaluation command, which would create a file in that directory.

## Fixed selection protocol

[The plan](experiment-plan-v2.md) and executable
[configuration](../config/experiment-v2.json) preceded model fitting. The frozen
plan digest is `a71605a07919c6a6979399a74becda05e2942c2e576ad5a08824fd4a14aae947`.
Source manifest:
`2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`.

The 3,716 training rows form 748 transitive contest/duplicate/alias groups.
Three seed-42 GroupKFold held-out partitions contain 1,237, 1,240 and 1,239 rows.
Each lexical vectorizer is fitted independently inside its fold; class weights
use that fold's training subset. There were no constant/absent training labels
in these actual folds. No validation information determines these groups.

Four candidates per representation compare C={1,3} and class_weight={None,
balanced}. Selection maximizes pooled training OOF macro F1 at fixed 0.5, then
micro F1, lower C and no weighting. Thresholds are chosen from 0.10–0.80 in 0.05
steps using the selected candidate's training OOF predictions. A label requires
30 positives, 30 negatives and two positives in every held-out fold for its own
F1 threshold; otherwise it uses the OOF micro-F1 global threshold. All 29 actual
labels met that predeclared support rule. Tests cover the fallback branch.

OOF predictions are used for both candidate and threshold selection; this is not
nested CV and the selected OOF score is not an unbiased performance estimate.
Validation is the common development comparison. It has also been exposed in
the earlier milestone, so no result here is a fresh final-test estimate. The
plan was not revised after seeing validation results.

## Actual validation results

| Run | Micro F1 | Macro F1 | DSU F1 | Binary-search F1 |
|---|---:|---:|---:|---:|
| Historical baseline-v1, validation-selected | 0.549974 | 0.474177 | 0.363636 | 0.424870 |
| A: word TF-IDF, training OOF selection | 0.508303 | 0.455723 | 0.274510 | 0.372414 |
| B: word + character TF-IDF, same protocol | 0.502842 | 0.455898 | 0.344828 | 0.384058 |
| C: frozen quantized MiniLM + logistic regression | 0.474525 | 0.385231 | 0.218750 | 0.332075 |

| Run | DSU precision / recall | DSU FP / FN | Binary-search precision / recall | Binary-search FP / FN |
|---|---:|---:|---:|---:|
| Historical v1 | 0.307692 / 0.444444 | 27 / 15 | 0.344538 / 0.554054 | 78 / 33 |
| A | 0.291667 / 0.259259 | 17 / 20 | 0.250000 / 0.729730 | 162 / 20 |
| B | 0.322581 / 0.370370 | 21 / 17 | 0.262376 / 0.716216 | 149 / 21 |
| C | 0.189189 / 0.259259 | 30 / 20 | 0.230366 / 0.594595 | 147 / 30 |

DSU support is 27 and binary-search support is 74 throughout. Compared with A,
B reduces binary-search FP by 13 while adding one FN; DSU adds four FP and
removes three FN. C adds 13 DSU FP with unchanged FN, and removes 15 binary-search
FP while adding ten FN. None of these errors were used to retune this plan.

B improves F1 on 12 labels, regresses on 12 and ties on five versus A. Its largest
gains are DSU (+0.070318), math (+0.020222), number theory (+0.013773), shortest
paths (+0.012097) and binary search (+0.011644). Its largest losses are FFT
(-0.048583, validation support three), hashing (-0.022727), trees (-0.017073),
combinatorics (-0.016659) and sortings (-0.013861). The macro gain is only
0.000176 while micro declines 0.005461. Character features address spelling,
lexical and notation variation; these results do not establish algorithmic
understanding.

C improves four labels, regresses on 24 and ties on one. Trees (+0.041573) and
geometry (+0.031884) improve, while probabilities (-0.355556, support five),
shortest paths (-0.272436), flows (-0.180451) and combinatorics (-0.167549) decline.
This one general-text encoder, quantization and pooling policy are not evidence
that all frozen encoders are unsuitable for this task.

The generated [comparison report](../artifacts/comparison-v2/comparison.md)
contains precision, recall, F1, support, FP/FN and changes for every label/run.
Machine-readable scores, thresholds, candidate OOF results and exact error
examples are saved separately in each run. Low OOF thresholds can yield many
validation positives: B's graph-matchings threshold 0.10 predicts 150 positives
against only three observed positives. The stated support rule does not make
rare-label threshold estimates reliable under a changing problem distribution.

## Configuration, resources and timings

A and B select C=1, balanced weights. C selects C=3, balanced weights. A/B global
thresholds are 0.45; C's is 0.55. DSU thresholds are 0.55/0.45/0.75 for A/B/C;
binary-search thresholds are 0.40/0.40/0.50. All 29 label thresholds and their
training-fold supports are frozen in each `run.json`.

The host had 16 logical CPUs, 15.26 GiB RAM and about 2.70 GiB free at plan freeze.
The RTX 3050 Laptop GPU had 4,096 MiB total and 3,729 MiB free. CPU ONNX with two
threads and at most eight chunks per batch was sufficient. No GPU allocation,
encoder weight update, large training framework or fine-tuning was required.

| Run | Training feature shape | Selection + representation | Full final head fit | Total | Eight-statement end-to-end probe |
|---|---|---:|---:|---:|---:|
| A | 3,716 × 40,000 CSR | 32.12 s | 2.87 s | 38.58 s | 0.0107 s |
| B | 3,716 × 80,000 CSR | 195.88 s | 20.51 s | 232.55 s | 0.0285 s |
| C | 3,716 × 384 float32 | 478.42 s | 5.68 s | 484.99 s | 0.8357 s |

Validation head-only inference on cached features took 0.0156/0.0641/0.0167 s;
those numbers exclude representation cost and are not end-to-end latency.
The bounded probes exclude process startup/model loading and are descriptive,
not controlled throughput benchmarks. Downloads are outside the recorded C run.
An independent full validation evaluation through each reloaded saved pipeline
reproduced all metrics and error counts: end-to-end inference on 467 statements
took 0.250 s for A, 1.108 s for B and 56.216 s for C, without refitting.
Embedding computation for 4,183 train/validation rows took 445.29 s; its array
uses approximately 6.13 MiB. Sampled final process RSS was about 260/351/479 MiB;
process-lifetime peaks were about 788/792/788 MiB. Those peaks include verification
and other work in the process and are not isolated representation peaks. The
embedding run and lexical work overlapped, so wall-clock comparisons are rough.

C pins [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41),
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, quantized ONNX asset
`onnx/model_quint8_avx2.onnx`, SHA-256
`b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21`.
The source model declares Apache-2.0. The tokenizer audit was written before
embedding: median 671 content tokens, p95 1,252, p99 1,604.5, maximum 2,718;
3,633/3,716 exceed one 254-content-token chunk. This supports using the plan's
overlapping-chunk policy rather than dropping most statements' suffixes.

There is no truncation: chunks have up to 254 content tokens plus CLS/SEP and
32-token overlap; attention-mask token means are combined by newly covered token
counts and L2 normalized. Mean chunks per training row is 3.557; maximum 13.
This pooling is an approximation that can dilute important observations. It
does not preserve all cross-chunk interactions. Quantization batches are kept
within each document so batch membership cannot change a statement's embedding.
The real eight-statement cached/pipeline probe and saved-model CLI inference
passed; synthetic tests specifically guard that batch boundary.

The frozen encoder assets and hashes live in `modeling/.cache/encoders/` and are
required by its saved pipeline. They are not embedded in the joblib file. Keep
that cache for offline inference; `prepare-encoder` restores the exact assets
when networking is available. Recreate the pinned cache location after moving
the workspace; the current run records its absolute location. Lexical models
need no encoder assets. All caches/models stay ignored by source control.

## Evidence-based review

[The review package](../artifacts/comparison-v2/error-review-v1.md) contains 48
label/case pairs from 45 validation problems: six FP, six FN, six TP and six TN
for each focus label. IDs were selected by a fixed seeded hash, independently
of new experiment outcomes. Every case records original tags, both focus-label
scores/decisions, statement/editorial versions, accepted-source identities,
exact matching evidence quotes with offsets, and the automated review status.
Source quote/version checks and a JSON Schema guard the saved package.

Automated proposals: six corroborated model-error candidates, one supported
alternative-approach candidate, zero potentially-incomplete-tagging candidates,
and 41 unresolved cases. **Zero are human-reviewed.** Correct observed-tag
decisions may remain semantically unresolved under the four requested outcome
categories. Unresolved does not mean the classifier is correct or incorrect
about suitability. Rules inspect complete selected source strings; they are
not a full semantic proof, execution check or annotation adjudication.

The following interpretations were made by the coding agent from local source
evidence and remain automated, not human-approved:

- **DSU false positives, 1811:F and 1814:E:** both describe vertices, edges and
  connectivity-like operations. The former editorial uses degree checks and
  BFS/DFS to validate a flower graph; the latter derives dynamic programming
  maintained by segment-tree matrices for edge-weight updates. These illustrate
  how graph vocabulary alone fails to identify the maintained invariant.
  Their suitability outcomes remain unresolved; neither editorial rules out
  every possible DSU-assisted solution.
- **DSU false negatives, 1797:E and 1847:D:** editorials explicitly discuss DSU
  for finding the next non-one array element and for building an ordering from
  string ranges, respectively. These are non-obvious successor/skipping uses
  whose statements do not read like simple component-merging tasks.
- **DSU false negative, 1872:F:** the editorial handles functional-graph cycles
  by peeling vertices and ordering cycles; the observed DSU tag remains a miss
  in original-tag evaluation, but the retained evidence does not adjudicate
  the DSU approach. It stays unresolved, rather than being relabeled.
- **Binary-search false positives, 1814:C and 1917:C:** a story about searching
  boxes is solved by sorting and greedy assignment; an array maximization
  problem is solved by a bounded enumeration of the first reset day. The agent
  interpretation is lexical/optimization confusion, not proof that binary
  search is impossible. Both suitability outcomes stay unresolved.
- **Binary-search false negatives, 1856:C and 1861:F:** the editorial explicitly
  derives search on an answer for local array operations and monotone card
  assignment feasibility; the latter uses a flow feasibility model. The missing
  operation is an inference over the task, not an explicit statement keyword.
- **Binary-search false negative, 1913:F:** an accepted source searches a
  palindrome radius with substring-comparison predicates. The string-editing
  statement does not explicitly advertise that implementation technique.
- **Binary-search false positive, 1822:G2:** retained accepted sources use
  `lower_bound`/`upper_bound` for frequency lookup in a number-theoretic
  enumeration. This supports an alternative implementation candidate; it does
  not establish that the upstream tag is wrong or that binary search is the
  central algorithm. No metric or original label is changed.
- **Correct controls, 1851:G and 1903:F:** the first editorial explicitly uses
  sorted offline connectivity with DSU; the second pairs binary search with
  2-SAT. They show why one task can support several families and techniques.

The first source scan exposed a pitfall in review automation: a reusable macro
can mention binary search without being executed, and a set's member
`lower_bound` is not an array binary-search proof. The scanner now excludes
comments, preprocessor templates and member-bound lookups. Regression tests
cover these exclusions. This was a correction to the new review tooling, not
a confirmed baseline classifier bug. Arbitrary unused functions and two-pointer
loops can still trigger weak signals, so human adjudication is essential.

## Verification and recommendation

The final check record is [checks-v2.json](../artifacts/comparison-v2/checks-v2.json).
It distinguishes synthetic regression tests from actual-corpus runs and CLI
smokes. The existing MongoDB integration suite uses disposable fixture databases;
modeling consumes the export and never connects to active `cpss`.
The combined suite passed **90 tests** in 12.82 s (including 13 new experiment
tests), `pip check` reported no broken requirements, and the forced authoritative
export verifier passed. The baseline audit, completed-run reuse, source-evidence
validation, saved-pipeline full-validation reproduction and unseen-statement CLI
smokes all ran successfully. No real final-test evaluation was invoked.

Keep the existing baseline reference. B is an interesting DSU-oriented
representation candidate, but the full-label micro regression and tiny macro
gain fail the declared adoption rule. C is slower and worse on the principal
metrics. Neither warrants adoption based on these results. No statistical
significance or calibrated confidence claim is made.

Next, have a human adjudicate the source-backed DSU/binary-search review queue,
keeping suitability annotations separate from upstream tags. Use the existing
100-problem training context pilot to draft and review objective, structure,
operation and applicability fields with exact evidence. For a later classifier
experiment, predeclare stronger rare-label threshold support/shrinkage and
group-aware nested or calibration partitions using training data only. For
retrieval, begin a small judged query set and BM25 statement baseline before
adding context or dense retrieval. Keep the final test unopened until the
configuration and evaluation question are deliberately frozen.

Implementation references:
[GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html),
[ONNX Runtime CPU sessions](https://onnxruntime.ai/docs/api/python/api_summary.html).
