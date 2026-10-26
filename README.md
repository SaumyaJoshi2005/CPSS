# CPSS - Competitive Programming Semantic Search

**Algorithm-Aware Semantic Retrieval of Competitive Programming Problems Using Hybrid NLP Models**

A local application for finding competitive-programming problems from a named
algorithm or a natural-language description. CPSS combines BM25, frozen MiniLM
embeddings and reciprocal rank fusion, with original source evidence for results.

| Component | Current state |
|---|---|
| Complete frozen corpus | 4,648 Open R1 / Codeforces problems |
| Searchable demonstration | 3,716 training problems; statement and editorial views |
| Classifier | Preserved 29-label TF-IDF + logistic-regression baseline |
| Retrieval evaluation | 24 queries, 814 pooled pairs; awaiting human judgments |
| Reserved classifier test | 465 problems; not evaluated |

## Run the application

Run PowerShell from this project folder using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m modeling index
.\.venv\Scripts\python.exe -m modeling serve
```

Open **http://127.0.0.1:8765**. Completed indexes are verified and reused. Search
supports BM25, dense and hybrid retrieval; rating filters; original tags;
Codeforces links; and exact statement/editorial excerpts. The application uses
the frozen export and does not require a running MongoDB server.

```powershell
.\.venv\Scripts\python.exe -m modeling search --method hybrid --view editorial --query "DSU with connectivity queries." --limit 3
```

For a new environment, dependency installation, restoring the existing encoder
assets, or the BM25-only fallback, follow the [runbook](modeling/docs/submission/RUNBOOK.md).

For the separate read-only public runtime, see [Replit deployment](modeling/deployment/REPLIT.md)
and [GitHub preparation](modeling/deployment/GITHUB.md). The verified serving
bundle runs without MongoDB or the full corpus archive; local review remains available.

## Project layout

```text
NLP_mini_project/
  src/cpss/          Corpus collection, validation, versioning and export
  modeling/         Classifier, context preparation, retrieval and local application
  config/           Corpus configuration and database schemas
  tests/            Corpus fixtures and tests
  tools/            Existing environment/bootstrap utilities
  docs/             Detailed corpus operations reference
  data/             Local source snapshots and collection audit records
  exports/          Frozen corpus exports
  .venv/            Existing Python environment
```

Generated models, indexes, relevance pools and modeling caches stay under
`modeling/artifacts/` and `modeling/.cache/`. These local assets are ignored by
source control and retained for reproducibility and offline operation.

## Documentation

- [Modeling and search](modeling/README.md): commands, source layout and current status.
- [Submission overview and architecture](modeling/docs/submission/PROJECT_REPORT.md).
- [Installation, indexing, review and launch](modeling/docs/submission/RUNBOOK.md).
- [Three-query demonstration](modeling/docs/submission/DEMO.md).
- [Actual retrieval measurements](modeling/docs/submission/RESULTS.md).
- [Corpus provenance and activation](OPENR1_COLLECTION.md).
- [Detailed corpus operations](docs/corpus-workflow.md), including the optional legacy adapter.

`LIVE_COLLECTION.md` and `VALIDATION.md` preserve historical collection evidence;
they are not the current application entry point.

## Verification

```powershell
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest tests modeling/tests -q
.\.venv\Scripts\python.exe -m pip check
```

The submission run passed 104 tests, including disposable MongoDB integration
tests. Detailed check records remain with the corresponding artifacts. Without
the MongoDB opt-in, those integration tests are skipped rather than counted as
passed.

The frozen export remains `exports/openr1-complete-2026-10-26-v2/`; its source
manifest SHA-256 is
`2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`.
Original splits are 3,716 train / 467 validation / 465 test. Missing upstream
tags do not prove an approach unsuitable. Retrieval relevance metrics remain
unavailable until people judge the candidate pool; classifier F1 measures a
different task. See the linked reports for development limitations and the
blocked browser visual check.

