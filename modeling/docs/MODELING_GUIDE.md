# Classifier and context modeling guide

This guide preserves the completed classifier milestones and their commands. Code/configuration paths in prose are relative to `modeling/`; run every command from the project root. Historical test totals and scope describe their respective milestones. Start with [the modeling README](../README.md) for the current application and documentation map.

## Preserved classifier milestones

**Follow-up classifier experiments:** the implementation audit reproduced the
frozen baseline with no confirmed prediction/metric bug. Training-only grouped
OOF selection and all three representation comparisons are complete. Neither
word+character nor frozen MiniLM meets the predeclared adoption rule. See
[actual comparison, source review and limitations](results-v2.md) and the
[fixed experiment plan](experiment-plan-v2.md). Baseline-v1 and test remain
unchanged.

Run the follow-up commands from the project root:

```powershell
# Optional dependencies for the frozen CPU encoder, using the existing .venv.
.\.venv\Scripts\python.exe -m pip install -r modeling/requirements-encoder.lock
.\.venv\Scripts\python.exe -m modeling audit-baseline
.\.venv\Scripts\python.exe -m modeling review-errors
# Completed runs are verified/reused, not overwritten or refitted.
.\.venv\Scripts\python.exe -m modeling experiment --representation all
.\.venv\Scripts\python.exe -m modeling compare
.\.venv\Scripts\python.exe -m modeling predict --run modeling/artifacts/comparison-v2/word_char --statement "Maintain connectivity as undirected edges are inserted."
# Restore pinned encoder assets if needed; otherwise all runs use local caches.
.\.venv\Scripts\python.exe -m modeling prepare-encoder
```

For an independent rerun, copy `config/experiment-v2.json`, change **only** its
`output` to a new directory under `modeling/artifacts/`, and pass `--plan` to
`experiment`. The new directory receives its own frozen plan, fold receipt and
complete inference bundles. Existing completed directories are never silently
retrained. Lexical caches reuse only matching fold/config/input hashes; encoder
caches reuse only pinned assets/config/input hashes. Repeating `compare` reads
saved validation metrics; it does not make new predictions or expose test.

New modules: `implementation_audit.py`, `grouping.py`, `representations.py`,
`encoder.py`, `experiment.py`, `error_review.py` and `resources.py`. All follow-up
source, schemas, tests, configuration and generated output are under `modeling/`.
No additional change outside this folder was needed for this milestone.

This folder contains the modeling-readiness audit, versioned algorithm vocabulary,
statement-only TF-IDF + one-vs-rest logistic-regression baseline, evaluation and
inference, and preparation for later context generation/review. It consumes the
existing frozen export; it does not contact MongoDB, recollect data, modify corpus
records, or create new splits. It uses the existing repository `.venv`.

**Actual validation:** micro F1 **0.549974**, macro F1 **0.474177**, 29 modeled tags.
DSU F1 **0.363636**; binary-search F1 **0.424870**. These are development results
against observed upstream tags, not exhaustive algorithm suitability judgments.
The real test split has **not** been evaluated.

See [actual results and error analysis](results.md),
[label specification and case studies](label-specification.md), and
[context generation/review instructions](generation-and-review.md).

## Setup and commands

Run from the main project folder in PowerShell. Existing corpus dependencies and
the installed `cpss` package are prerequisites. Modeling adds no nested environment,
Git repository, service or experiment tracking system. The optional encoder
uses the separately pinned existing model cache.

```powershell
# Dependencies were installed into the existing .venv during implementation.
.\.venv\Scripts\python.exe -m pip install -r modeling/requirements.lock

# Existing corpus verifier, followed by modeling identity/split checks.
.\.venv\Scripts\python.exe -m modeling verify

# Training-only frequencies, imbalance, co-occurrence, lengths and examples.
.\.venv\Scripts\python.exe -m modeling audit

# Train/select on the preserved train/validation splits; no test evaluation.
# baseline-v1 already exists; choose another name only for a deliberate new run.
.\.venv\Scripts\python.exe -m modeling train --run-name baseline-v2

# Reproduce historical validation metrics without writing into baseline-v1.
.\.venv\Scripts\python.exe -m modeling audit-baseline

# Predict from an unseen statement string or UTF-8 text file.
.\.venv\Scripts\python.exe -m modeling predict --run modeling/artifacts/baseline-v1 --statement "Maintain connectivity while undirected edges are added."
.\.venv\Scripts\python.exe -m modeling predict --run modeling/artifacts/baseline-v1 --statement-file path/to/unseen-statement.txt

# Deterministic training-only references and empty, unreviewed context drafts.
.\.venv\Scripts\python.exe -m modeling prepare-contexts
.\.venv\Scripts\python.exe -m modeling context-input --problem-id codeforces:1156:E

# Validate an illustration with its synthetic evidence.
.\.venv\Scripts\python.exe -m modeling validate-context --file modeling/examples/problem-dsu.json --illustrative-sources modeling/examples/synthetic-sources.json
```

The global option `--config PATH` precedes the command:

```powershell
.\.venv\Scripts\python.exe -m modeling --config modeling/config/baseline.json audit
```

Paths in configuration are relative to the repository root, not the shell's
working directory. `--run-name` must be one directory name; existing runs are
never silently overwritten. Commands fail explicitly on invalid inputs.

## Frozen corpus and split isolation

Default source: `exports/openr1-complete-2026-10-26-v2/`; expected manifest digest:
`2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`.
Counts remain 3,716 train, 467 validation, 465 test. Canonical IDs, contest groups,
exact duplicates, explicit duplicate groups, aliases and upstream holdouts are
checked against the existing manifest without recalculating assignments.

`data.Corpus.verified` reuses `cpss.export.verify_export`. A verification receipt
avoids rehashing 18,434 unchanged blobs on every command. The expected manifest
hash is checked every time; size/mtime signatures of all pinned files invalidate
the receipt if anything changes. `verify --force` reruns the complete verifier.

The loader exposes selected splits through typed `Record` values. Development
loads only train and validation text/targets; test rows are skipped before their
targets/text are inspected or summarized. Test IDs/counts/group membership are
used only for structural integrity. The audit computes label selection, length
statistics and representative examples from training records only.

## Labels and baseline

`config/taxonomy-v1.json` separates original tags, aliases, explicit broad-family
projections and future fine vocabulary. DSU has multiple parents. Generic tags
never create Dijkstra/Bellman–Ford or binary-search-on-answer annotations.
Unknown/unreviewed remains explicit. There are no newly approved corpus labels.

`config/baseline.json` configures support thresholds, sparse features,
regularization/class weighting, seed and validation threshold grid. Labels need
at least 30 training positives and negatives. Six low-support tags are retained
in reports but excluded from the initial 29-label classifier. Absent, unsupported
and constant labels have explicit exclusion reasons. Unmodeled validation tags
are reported without changing the training-derived vocabulary.

Features come exclusively from `Record.statement`, sourced from `input.statement`.
Original tags, broad mappings, editorials, code, IDs and metadata never enter the
feature matrix. TF-IDF vocabulary/IDF and class weights use training only. No
learned feature selection or normalization is fitted on validation/test data.
TF-IDF uses all statement text without truncation; no corpus copy is created.

Six regularization/weighting candidates are selected by validation micro F1 at
validation-selected global thresholds, then per-label validation F1 thresholds
are fitted when support is sufficient. The exact tie-breaking, fallback reasons,
configuration, candidate metrics, label order and timings are frozen in the run.
This double use of validation can be optimistic; it is documented as development
performance. Scores are **uncalibrated independent logistic scores**, not
confidence guarantees or an exhaustive algorithm suitability distribution.

## Artifacts and caching

Generated artifacts/caches are ignored by `modeling/.gitignore`:

```text
modeling/
  config/                  baseline configuration and shared vocabulary
  schemas/                 problem/query context JSON Schema
  docs/                    actual findings, case studies and review instructions
  examples/                three synthetic, unreviewed context illustrations
  tests/                   synthetic fixtures and isolation/persistence checks
  artifacts/
    audit/                 JSON/Markdown training-only readiness report
    baseline-v1/           pipeline.joblib, run.json, metrics/errors and file hashes
    context-pilot/         100 pinned training references and pending JSONL drafts
  .cache/                  export verification and training TF-IDF/CSR cache
```

The complete sklearn inference pipeline, ordered labels, thresholds, configuration,
taxonomy and provenance are saved together. Runs record the corpus manifest,
configuration/taxonomy digests, dependency/model versions, seed, fit IDs and
selection IDs. Artifact file hashes are checked before loading. Dependency
versions must match the frozen run; use `modeling/requirements.lock` alongside
the existing root environment. Joblib artifacts must come from a trusted local
run: pickle-based model files are not safe inputs from untrusted sources.

Feature caches contain a fitted vectorizer and sparse CSR matrix, keyed by actual
training statement hashes/IDs, feature configuration and dependency/feature
versions. Cache integrity is checked before reuse. No validation vocabulary is
learned, and no corpus text/editorial archive is copied into the cache. Short
physical cache/temporary names avoid Windows path-length failures; full digests
remain in metadata. Final model directories are promoted by atomic rename.

## Final test evaluation (reserved; not executed)

Only after deliberately freezing the experiment:

```powershell
.\.venv\Scripts\python.exe -m modeling evaluate --run modeling/artifacts/baseline-v1 --split test --final
```

This loads test targets/text only for the separately requested evaluation, uses
the already-frozen labels and thresholds, and saves `test-evaluation.json`. An
exclusive claim prevents accidentally evaluating the same run again. If that
process is interrupted, inspect the claim/report before making an explicit
recovery decision; the command does not silently repeat a partial test exposure.
`run.json` records that no test had been evaluated at training/freeze time; the
separate claim/report records any later final evaluation. Do not use test errors
to tune another configuration and then present that as untouched final testing.

## Context pilot

`prepare-contexts` deterministically samples 100 training problems, beginning
with DSU/binary-search quotas and covering every selected label. Current coverage
includes 13 DSU-tagged and 14 binary-search-tagged problems. The sample contains
IDs, exact statement/editorial versions, hashes and observed tag metadata only.
Pending drafts have no generated objective, approach or observation annotations.
Their status is `pending_generation`; zero records are human-reviewed.

Problem-context schema: abstract objective, structures, operations, approaches,
observations, applicability, unknowns, provenance, exact source evidence and
review state. Query-context schema separates original text, explicit requirements,
inferred approaches, exclusions and unknowns. Both use the same controlled
algorithm IDs. Model-generated drafts must identify model/prompt revisions.

The review CLI validates source quotes and versions and requires explicit
reviewer/decision information. It does not write annotations to MongoDB. Passing
schema validation is not a semantic/human review. Bulk context generation and neural fine-tuning remain outside the initial modeling milestone. Retrieval and the local interface were subsequently implemented; see the [current runbook](submission/RUNBOOK.md).

## Checks and integration

```powershell
.\.venv\Scripts\python.exe -m pytest modeling/tests -q
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest tests modeling/tests -q
.\.venv\Scripts\python.exe -m pip check
```

The modeling tests use synthetic data, including synthetic held-out records for
the single-use final-evaluation guard. They never evaluate real test content.
Existing MongoDB integration tests use disposable databases, not active `cpss`.

All new modeling work is under this folder. The sole code change elsewhere is
`tools/fetch_wheels.py`: its existing PyPI fallback now skips invalid legacy
version strings (encountered while installing joblib). No corpus implementation,
active database, authoritative export, root dependency configuration or splits
were replaced.

Primary library references consulted for the implementation:
[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
[logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[one-vs-rest](https://scikit-learn.org/stable/modules/multiclass.html),
[model persistence](https://scikit-learn.org/stable/model_persistence.html).
