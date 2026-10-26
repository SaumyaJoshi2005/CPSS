# CPSS modeling and search

For the portable read-only runtime, Replit setup and GitHub handoff, see
[deployment/README.md](deployment/README.md). Existing local models and splits
are preserved.

This package contains statement-only classifiers, source-backed context
preparation, retrieval, a local search interface and relevance evaluation.
It consumes the existing frozen export using the project `.venv`.

## Quick start

Run from the project root, not from this subfolder:

```powershell
.\.venv\Scripts\python.exe -m modeling index
.\.venv\Scripts\python.exe -m modeling serve
```

Open **http://127.0.0.1:8765**. If that port is occupied, use the existing instance
or `serve --port 8766`. The application searches **3,716 training problems** from
the **4,648-problem complete corpus**. Existing indexes are verified and reused.
Dense/hybrid search uses the existing pinned MiniLM cache; `index --lexical-only`
provides the BM25 fallback. Installation and cache restoration are documented
in the [runbook](docs/submission/RUNBOOK.md).

## Commands

| Task | Command after `.\.venv\Scripts\python.exe -m modeling` |
|---|---|
| Verify frozen corpus and splits | `verify` |
| Build or reuse both search views | `index` |
| Search from the terminal | `search --method hybrid --view editorial --query "DSU with connectivity queries."` |
| Launch the application | `serve` |
| Prepare or reuse blinded judgments pool | `retrieval-pool` |
| Evaluate actual relevance judgments | `retrieval-evaluate` |
| Measure warm search latency | `retrieval-latency` |
| Reproduce historical classifier metrics | `audit-baseline` |
| Inspect training labels and lengths | `audit` |
| Predict labels for a new statement | `predict --run modeling/artifacts/baseline-v1 --statement "Maintain connectivity as edges are inserted."` |
| Prepare unreviewed training context references | `prepare-contexts` |

Use `--help` after any command for its options. The global `--config PATH`
precedes a command. Search commands also accept `--search-config PATH` after
the command. Configuration paths resolve against the project root.

## Source and generated files

```text
modeling/
  application/        Local HTTP server and readable HTML/CSS/JavaScript clients
  retrieval/          Loading/context views, BM25, dense indexes, fusion and review
  config/             Frozen experiment, vocabulary, search and query settings
  schemas/            Context and automated error-review schemas
  tests/              Synthetic regression fixtures and client integration check
  examples/           Explicitly synthetic context illustrations
  docs/               Guides, historical findings and submission materials
  *.py                Existing modeling modules and stable CLI entry points
  artifacts/          Saved runs, indexes, reports and review pools (local/ignored)
  .cache/             Pinned encoder assets and preprocessing caches (local/ignored)
```

The classifier modules retain their existing import paths because saved
pipelines depend on them. The [development guide](docs/DEVELOPMENT.md) maps these
modules and explains the source/artifact boundary. Source files use UTF-8 and
the repository `.editorconfig` conventions.

## Current evidence

| Result | Recorded status |
|---|---|
| Historical classifier validation micro / macro F1 | 0.549974 / 0.474177; 29 labels |
| Follow-up word / word+character / MiniLM comparison | Neither alternative met the adoption rule |
| Relevance pool | 24 queries, 814 query-problem pairs |
| Human relevance judgments | Zero at submission snapshot; NDCG/MRR unavailable |
| Warm mean search time | Approximately 9-18 ms, with loading measured separately |
| Submission verification | 104 tests; CLI, HTTP and client-script checks passed |
| Browser visual verification | Blocked by tool URL verification; no screenshot claim |
| Final classifier test split | Reserved; not evaluated |

These are recorded development results, not calibrated confidence or exhaustive
algorithm-suitability judgments. Historical classifier validation selected both
hyperparameters and thresholds. Its results are not protocol-matched to the
later grouped-CV experiments. Current review status is read by
`retrieval-evaluate`; the table describes the saved submission snapshot.

## Documentation

Start with the [documentation index](docs/README.md), or go directly to:

- [Application runbook](docs/submission/RUNBOOK.md) and [three-query demo](docs/submission/DEMO.md).
- [Project architecture](docs/submission/PROJECT_REPORT.md) and [retrieval results](docs/submission/RESULTS.md).
- [Detailed classifier/context commands](docs/MODELING_GUIDE.md).
- [Classifier results](docs/results.md), [controlled comparison](docs/results-v2.md) and [fixed plan](docs/experiment-plan-v2.md).
- [Label specification](docs/label-specification.md) and [context generation/review](docs/generation-and-review.md).

## Checks

```powershell
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest tests modeling/tests -q
.\.venv\Scripts\python.exe -m pip check
# Optional client check against an already running local application:
node modeling/tests/ui-smoke.cjs
```

MongoDB integration tests use disposable databases. The real corpus, immutable
exports, historical model directories and reserved test content are preserved.
Keep downloaded encoder assets for offline dense search. Historical manifests
retain their original source hashes; later maintenance has its own verification
record rather than rewriting earlier evidence.
