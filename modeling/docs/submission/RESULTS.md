# Actual delivery results

## Coverage and artifacts

Both views index the same 3,716 training problems from the 4,648-problem frozen
corpus. No validation/test problem is a search candidate. Metadata, original
statements, editorial evidence and labels are not modified. No newly generated
reasoning, suitability labels or human-reviewed annotations are claimed.

| Artifact | Location under `modeling/` |
|---|---|
| Search configuration | `config/search-v1.json` |
| Fixed 24-query set | `config/retrieval-queries-v1.json` |
| Active index pointers | `artifacts/search-v1/active.json` |
| Versioned sparse/dense indexes | `artifacts/search-v1/indexes/` |
| Frozen pooled rankings | `artifacts/retrieval-evaluation-v1/pool.json` |
| Blinded review sheet | `artifacts/retrieval-evaluation-v1/review-sheet.csv` |
| Current judgment-based evaluation | `artifacts/retrieval-evaluation-v1/retrieval-metrics.json` |
| Detailed latency samples | `artifacts/retrieval-evaluation-v1/latency.json` |
| Delivery verification record | `artifacts/submission-v1/checks.json` |

Statement index promotion took 4.05 s, including existing training-vector reuse.
Editorial index promotion took 649.56 s, including 642.92 s of dense encoding.
Their total persisted sizes are approximately 10.96/12.07 MiB, respectively;
source corpus text is not copied into the index artifacts. Both record exact
view hashes, canonical row order, source manifest, full configuration, model
revision, runtime versions and file hashes. Writes use temporary directories
and atomic promotion. A repeated indexing command successfully reused both.

## Retrieval relevance

The review pool covers 24 fixed development queries and 814 unique query–problem
pairs, deduplicated across BM25/dense/hybrid and statement/editorial views.
Top-ten pooling is consistent with the requested @10 metrics. There are
**0 judged pairs and 0 fully judged queries** in the actual corpus review.

| Comparison | NDCG@10 | MRR@10 |
|---|---|---|
| Statement BM25 / dense / hybrid | Unavailable | Unavailable |
| Editorial BM25 / dense / hybrid | Unavailable | Unavailable |

Unavailable is not zero and is not an estimated benchmark score. No tag-overlap
heuristic was substituted for relevance. The pool and candidate examples are
provisional, unjudged outputs. No method/view effectiveness winner is claimed.
Classifier F1 numbers in the earlier reports evaluate observed-tag prediction,
not retrieval relevance.

Once reviewed, NDCG@10 uses gain `2^grade-1` and a judged-pool ideal ranking.
MRR@10 considers grade 2 directly relevant. A query is included only after all
of its pooled pairs are judged, and identical complete queries are compared
for every method/view. The review UI and CSV hide originating method, score,
rank and view; the raw pool preserves this information for evaluation.

## Latency protocol and observations

One process loaded both verified indexes and their original source contexts:
**8.891 s**. It ran 24 queries, one warm-up pass, then three measured passes for
each method/view: **72 samples per configuration, 432 measured searches**.
Query vectors were not cached. Boosts were off; candidate limit was 100 per
retriever. Timings include parsing, query encoding, ranking and source snippet
selection; they exclude loading and HTTP/browser overhead.

| View / method | Mean | Median | P95 |
|---|---:|---:|---:|
| Statement / BM25 | 9.27 ms | 9.01 ms | 11.28 ms |
| Statement / dense | 13.61 ms | 13.55 ms | 14.96 ms |
| Statement / hybrid | 15.35 ms | 15.15 ms | 16.99 ms |
| Editorial / BM25 | 11.40 ms | 11.52 ms | 13.13 ms |
| Editorial / dense | 14.55 ms | 14.55 ms | 16.30 ms |
| Editorial / hybrid | 17.53 ms | 17.65 ms | 19.56 ms |

The first dense call, including lazy ONNX initialization, took 315.47 ms.
Later configurations reused the encoder. These are actual measurements on a
16-logical-CPU Windows host with 15.26 GiB RAM, CPU encoder inference using two
threads, and other processes active. Pooling and server startup were also
running during portions of the measurement; no dedicated-host or production
latency claim is made. Sampled latency-process RSS after measurement was about
279 MiB; its 692 MiB lifetime peak includes verification and loading. The GPU
was inspected but not used. No training weights were updated.

## Checks and remaining limitation

The combined suite passed **104 tests** in 14.32 seconds: the prior 90 tests plus
14 new retrieval cases. These cover BM25 against a hand calculation, RRF,
deduplication, stable ties, filters before candidate limits, source/metadata
boundaries, cache invalidation and corruption, soft boosts without gating,
synthetic graded metrics, blinded review/attestation and local HTTP behavior.
Existing MongoDB integration tests use disposable databases, not active `cpss`.

`pip check` passed. All three real CLI/HTTP retrieval methods worked. Optional
tag/classifier boosts were smoke-tested separately from the unboosted benchmark.
The shipped client script passed actual-server integration for the three demo
queries, empty results, invalid rating constraints and empty input. Both client
scripts passed JavaScript syntax checks. These checks do not establish browser
layout quality. Visual browser verification and screenshots remain blocked
because the automation tool could not verify the current browser URL and
stopped; no bypass or successful visual check is claimed.

The frozen baseline's full file inventory remains unchanged. The reserved
classifier test has not been evaluated, and retrieval has used only training
candidates. No further classifier tuning, data collection or bulk annotation
was performed during the delivery sprint.
