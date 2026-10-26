# Development guide

Run commands from the repository root. Use the existing `.venv`; dependencies
are pinned in the root and modeling requirement files. No extra service or
frontend build step is needed for the local application.

## Source boundaries

| Responsibility | Source |
|---|---|
| CLI entry point | `modeling/__main__.py`, `modeling/cli.py` |
| Shared configuration, hashes and atomic JSON writes | `modeling/common.py` |
| Verified corpus reader and typed records | `modeling/data.py` |
| Vocabulary and label audit | `modeling/labels.py`, `modeling/audit.py` |
| Classifier features, fitting, metrics and inference | `features.py`, `training.py`, `metrics.py`, `inference.py`, `evaluation.py` |
| Controlled experiments and review | `grouping.py`, `representations.py`, `experiment.py`, `implementation_audit.py`, `error_review.py` |
| Frozen encoder and host measurements | `modeling/encoder.py`, `modeling/resources.py` |
| Context pilot and schema validation | `modeling/contexts.py`, `modeling/schemas/` |
| Retrieval configuration and problem/query views | `modeling/retrieval/configuration.py`, `modeling/retrieval/contexts.py` |
| Sparse/dense indexes and ranking | `modeling/retrieval/lexical.py`, `indexing.py`, `ranking.py`, `search.py` |
| Relevance judgments and evaluation | `modeling/retrieval/judgments.py`, `evaluation.py` |
| Local presentation | `modeling/application/server.py`, `modeling/application/static/` |

Unqualified module filenames in the classifier rows live directly under
`modeling/`; retrieval filenames live under `modeling/retrieval/`. The existing
boundaries separate data, representation, selection, ranking and presentation.
Browser clients use named request/render/event functions and text nodes for
original source content. They do not implement a second ranking pipeline.

## Compatibility and reproducibility

The package's existing module names are intentionally stable. Saved joblib
pipelines can reference Python classes by import path, particularly
`modeling.encoder.FrozenEncoder`. Moving modules or model caches is therefore
an artifact migration, not cosmetic folder organization. Source cleanup keeps
those paths and all CLI commands intact.

Configuration is versioned under `modeling/config/`. Source export identities,
taxonomy, dependency versions, model revisions and relevant settings are stored
with generated results. Index and preprocessing caches are keyed by their
inputs and settings. A change to model inputs belongs in a deliberate new run;
formatting or documentation cleanup does not retrain existing runs.

## Local assets

| Location | Ownership and use |
|---|---|
| `exports/openr1-complete-2026-10-26-v2/` | Authoritative immutable corpus export |
| `data/` | Collection snapshots, provenance and activation records |
| `modeling/artifacts/baseline-v1/` | Historical classifier; preserve its files |
| `modeling/artifacts/comparison-v2/` | Frozen representation comparison and source review |
| `modeling/artifacts/search-v1/` | Current sparse/dense indexes and active pointers |
| `modeling/artifacts/retrieval-evaluation-v1/` | Frozen pools, original human judgments when supplied, and evaluation outputs |
| `modeling/artifacts/context-pilot/` | Pinned training references and pending drafts |
| `modeling/artifacts/submission-v1/` | Historical submission manifest and checks |
| `modeling/artifacts/maintenance/` | Subsequent cleanup snapshots and verification |
| `modeling/.cache/encoders/` | Downloaded pinned model assets, required for offline dense inference |
| `modeling/.cache/features/`, `cv-features/`, `embeddings/` | Reusable verified representations |

Generated artifacts and downloaded assets are local/ignored, but are not
disposable project clutter. Human judgments and pinned source snapshots are
research evidence. Standard interpreter/test scratch directories are separate
from these assets and hidden by the workspace Explorer settings.

Historical submission manifests keep the original source hashes from that
delivery. Subsequent source cleanup has its own maintenance report. A historical
source hash mismatch after an explicitly documented edit does not imply a
change to the frozen model or index bytes.

## Formatting and checks

`.editorconfig` defines UTF-8, LF line endings, four-space Python indentation and
two-space web/config indentation. Workspace settings hide interpreter/test
scratch folders and exclude large generated data from ordinary editor searches;
they do not remove the underlying files.

```powershell
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest tests modeling/tests -q
.\.venv\Scripts\python.exe -m pip check
node --check modeling/application/static/app.js
node --check modeling/application/static/review.js
# Against an already running application:
node modeling/tests/ui-smoke.cjs
```

Do not use the real final classifier test as a routine cleanup check. Existing
tests exercise that boundary on synthetic fixtures. Integration tests use
disposable MongoDB databases. Real search checks consume only training problems.
