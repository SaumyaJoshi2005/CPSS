# Open R1 corpus — 26 October 2026

Open R1 is the primary corpus provider. Codeforces remains the platform identity
(`codeforces:<contest_id>:<index>`), with direct page/API collection available as
a secondary adapter. There is no longer a 2,000-problem cap.

## Validated collection

| Component | Count |
|---|---:|
| Upstream problem rows inspected | 10,024 |
| Upstream selected accepted submissions inspected | 42,467 |
| Complete qualifying problems | **4,648** |
| Complete statement projections | **4,648** |
| Version-specific editorial associations | **4,648** |
| Retained accepted solution sources | **13,786** |
| Complete public input/output example pairs | **8,443** |
| Excluded problem rows | **5,376** |
| Immutable document versions | 23,082 |
| Content-addressed raw row blobs | 18,434 |

Coverage is **31 January 2015 through 11 February 2025** among admitted problems.
There are 2,736 CF-scored and 1,912 ICPC-scored problems, including ordinary
Educational rounds. Scoring system does not determine regular-contest status.

## Pinned sources

- [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces), revision
  `fbe3f6e903ee854eec2e69e9d96d0306cde59baf`, `default` data files, train and test.
- [open-r1/codeforces-submissions](https://huggingface.co/datasets/open-r1/codeforces-submissions), revision
  `07e7bf31f284b0e90e02dfc7ab675d5821e1c068`, `selected_accepted` data file.

The snapshot is in `data/openr1/`. `snapshot.json` records repository revisions,
upstream Parquet file identities, selected columns, row counts and hashes of the
exact locally retained JSONL projections. Full Parquet hashes are marked as
upstream-declared rather than falsely claimed to have been verified locally.
The upstream README and license declarations are saved with hashes. Open R1's
card states CC-BY-4.0 while its problem README body names ODC-By; both declarations
are retained without inventing a resolution.

Only relevant problem fields and the selected accepted submission subset were
read. Full private tests, the complete multi-million-row submission archive,
generated checkers/test archives and generated reasoning traces were not imported.

## Completeness policy

Every admitted problem requires matching stable IDs; contest name, date and
scoring type; title; rating; nonempty original tags; positive time/memory limits;
description; input and output formats; complete public examples; a non-placeholder
editorial; and at least one accepted source with language and submission identity.
Notes are preserved when supplied. Interactive and special-format contests and
unresolved images are excluded. No content or missing metadata was fabricated.

The statement combines title, limits, description, input/output formats, every
example and any note, while immutable sections preserve their individual values.
Constraints and mathematical notation remain in their upstream description/input
sections. Original row bytes are retained as hashed canonical JSON projections.
`problems.corpus` exposes structured examples, limits, aliases, source revision,
upstream split, accepted submission IDs and upstream verification flags.

Accepted code is joined through the canonical problem ID or an explicit alias,
checked for verdict `OK` and deduplicated by source content. Up to three distinct
sources per problem are retained. The original `og_source` is preferred when
available. Actual programming-language and submission metadata are preserved;
author handles absent upstream are not fabricated.

"Complete" is a structural and identity guarantee for the required corpus
components, not a claim of exhaustive manual review or local execution. Code
was not compiled/run locally. Some equations were OCR-processed upstream and
still warrant sample review. Only 579 admitted rows carry the upstream flag
`official_tests_complete=true`; private test bodies were not downloaded even for
those rows. This corpus must not be described as having complete private judge tests.

## Exclusions

Reasons overlap; their counts therefore do not sum to the 5,376 excluded rows:

| Reason | Rows |
|---|---:|
| Missing editorial | 3,928 |
| Outside regular round scope | 2,296 |
| Special contest | 663 |
| Missing output format | 282 |
| Missing rating | 257 |
| Interactive problem | 226 |
| Missing input format | 162 |
| Missing tags | 154 |
| Incomplete/missing examples | 143 |
| No accepted code source after identity/alias join | 134 |
| Missing description | 43 |
| Missing time limit | 30 |
| Missing memory limit | 30 |
| Invalid identity | 4 |
| Unresolved image | 3 |

The full row-level report is
`data/openr1/import-report-cpss_openr1_stage_full_20261026.json`.
The import run ID is `c36f0ab9-5156-580d-b0f3-f2101199d2a5`.

## Exports

Use **`exports/openr1-complete-2026-10-26-v2/`**:

- `statement-only.jsonl`: 4,648 records; classifier features are only `input.statement`.
- `editorial-enriched.jsonl`: 4,648 records, including pinned accepted source code.
- `manifest.json`, `manifest.sha256`, `versions.jsonl`, and `blobs/`: metadata,
  frozen label decisions, exact content versions, provenance and recoverable bytes.

Manifest SHA-256:
`2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`.

| Split | Problems |
|---|---:|
| Train | 3,716 |
| Validation | 467 |
| Test | 465 |

Contests, explicit duplicate groups, exact clean-text duplicates and transitive
Open R1 aliases stay together. Title heuristics are restricted to explicit
easy/hard variants on the same date; generic shared names are not duplicate
evidence. Chronological intervals are respected, and all 222 admitted upstream
test rows remain held out. The initial export without the `-v2` suffix is
superseded because its overly broad title grouping produced unsuitable splits.

## Validation and activation

The 20-problem pilot imported 60 accepted sources and passed a completeness
audit. The full staging audit checked all 4,648 admitted bundles with no issues.
The final suite passed **61 tests**: 40 offline and 21 actual MongoDB integration
tests. Tests cover missing components, alias code joins, preserved originals,
idempotency, corrupt snapshots, export content, upstream holdout separation,
transitive duplicates, generic title collisions, archive/promotion and rejection
of an incomplete staging database, alongside the existing failure tests.

An actual query explain motivated the `problem_sources` index: finding the three
accepted sources for one problem examined 3 documents with the index, versus
23,082 without it. `pip check` passed, and `requirements.lock` pins the tested
dependencies.

The active database is `cpss`. Promotion archives the prior collections in
`cpss_backup_before_openr1_20261026`, preserving the previous 6,000 metadata
candidates, 6,004 jobs and the earlier direct editorial/code snapshots. Those
incomplete records are removed from the active corpus. No fixture database is
mixed into the active corpus. The exact recoverable cutover plan/status is
`data/activation-cpss_openr1_stage_full_20261026.json`.

Promotion completed successfully. The post-promotion completeness audit passed
for all 4,648 active problems, and reconciliation found no issues. Additional
cleanup checks found **0 incomplete active problems**, **0 sources without
dataset provenance**, and **0 pending/failed legacy jobs**. The backup still
contains all 6,000 previous metadata records. Machine-readable verification is
saved in `data/openr1/active-validation.json`.

Commands to inspect the resulting corpus:

```powershell
.\.venv\Scripts\python.exe -m cpss doctor
.\.venv\Scripts\python.exe -m cpss audit-complete
.\.venv\Scripts\python.exe -m cpss reconcile
.\.venv\Scripts\python.exe -m cpss verify-export exports/openr1-complete-2026-10-26-v2
```

No retrieval models, classifiers, frontend, generated summaries or new inferred
labels were added.
