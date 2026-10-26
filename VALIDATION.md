# Execution report — 26 October 2026

Implementation and validation used the VS Code workspace:
`C:\New folder (2)\Proj\NLP_mini_project`.
No implementation was placed in the originally requested Desktop folder.

**Current primary corpus:** Open R1 now supplies 4,648 complete qualifying
problems, 13,786 accepted code sources and 8,443 public example pairs. The
current suite passed 61 tests. See `OPENR1_COLLECTION.md` for source revisions,
exclusions, exports and cleanup details. The direct-collection results below are
historical.

**Earlier update:** the initial implementation results below are historical.
Subsequent live collection stored 4 editorials, 33 approved problem sections and
32 current embedded code blocks. Statement access still returns HTTP 403. The
updated test suite passed 42 tests. See `LIVE_COLLECTION.md` for the current
counts, artifacts and editorial-only collection command.

## Environment

| Check | Actual result |
|---|---|
| Python | 3.13.1; project-local `.venv` installed |
| MongoDB service | Already running; anonymous loopback ping succeeded |
| MongoDB version | 6.0.29 |
| mongosh | 2.6.0; ping succeeded |
| Docker | Client 29.4.1 installed; engine stopped/unavailable; not required |
| Dependency compatibility | `pip check` passed; exact versions in `requirements.lock` |
| Credentials | `token.txt` was never read; no credentials added or logged |

## Tests actually run

**37 tests passed: 22 offline tests and 15 actual MongoDB integration tests.**
The integration tests used uniquely named disposable `cpss_test_<uuid>` databases
on the running local MongoDB service; each was removed after its test. They were
not simulated with mongomock. No Codeforces network requests were used by tests.

Coverage includes deterministic extraction, formulas/MathJax/Unicode, sample
line boundaries and code indentation; HTTP 200 challenge/login rejection;
API FAILED and redirect rejection; shared limiter spacing, caching and
Retry-After; robots enforcement and denied-origin circuit breaking; actual
server validators for malformed types/readiness/hash/confidence; actual unique
indexes; duplicate imports; changed raw/clean content and processor settings;
stale observation guards; crashes after blob/version/source writes; projection
repair; inline/GridFS routing and interrupted upload recovery; incorrect
editorial/curated mappings; taxonomy cycles; preserved reviewed annotations;
retry replay of original bytes; resume skipping blocked jobs; reprocessing;
exported code references; manifest hashes/tamper detection; and contest/duplicate
split separation with chronological ordering.

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not mongo" -q
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

The initial server test caught MongoDB's rejection of an empty JSON Schema
`required` array; the shared schema builder was corrected. All final tests passed.

## Offline CLI workflow

The real CLI was also exercised end to end in the separate
`cpss_fixture_demo` database, with **synthetic** content only:

| Item | Count/result |
|---|---|
| Run | `5066e2bb-f7a3-4893-a7c8-a19065728a55` |
| Statements ready | 4 |
| Approved editorial sections | 4, from 2 shared editorials |
| Problems with embedded code | 1 |
| Failed jobs | 0 |
| Reconciliation | Passed, no issues |
| Export | `exports/fixture-validation` |
| Export verification | Passed |
| Manifest SHA-256 | `9051816321a99b65cc6b0c5f06d5589ceb84118f993f38bf1a65e6d0036356eb` |

The tiny fixture selection has four training records and empty validation/test
splits. This is expected from the whole-contest/variant grouping and is not a
usable evaluation corpus. Synthetic fixtures are excluded from the live `cpss`
database and from the live counts below.

## Live Codeforces execution

Official API contracts, terms, robots policy and dependency documentation were
checked before collection. Both catalogue API requests succeeded and their exact
response bodies were retained. The current robots response was fetched and
cached before the statement request.

| Item | Actual result |
|---|---|
| Database | `cpss` |
| Target | 2,000 problems with valid statements and approved editorial sections |
| Run | `9f49ed68-7108-49d4-a41b-d53a7eff5100` |
| Metadata candidates discovered | **6,000** |
| Latest selected contest | Codeforces Round 1129 (Div. 2), contest 2277 |
| Latest contest timestamp | 24 October 2026, 11:05 UTC |
| Statement jobs attempted | 1 |
| Statement jobs blocked | 1, `http_403_blocked` |
| Remaining statement jobs | 5,999 pending |
| Valid live statements collected | **0** |
| Valid live editorial sections collected | **0** |
| Live solution code collected | **0** |
| Run state | Partial, resumable |
| Database reconciliation | Passed, no issues |

The planned 20-problem live pilot stopped at the first denied statement page.
The circuit breaker prevented further page requests in that batch. **The pilot
did not pass and the 2,000-problem corpus has not been collected.** No challenge
or forbidden response was published as problem text. No live corpus export was
created, since there are no valid live statements.

## Remaining limitations and next execution

- Direct statement-page access from this environment returned HTTP 403. A
  working permitted page-access environment or independently acquired permitted
  HTML imports are needed. Do not bypass access controls or repeatedly retry
  blocked pages. Ordinary resume skips blocked jobs; explicit `retry` is available
  only after the access condition has been resolved.
- Live editorial extraction and solution-code collection remain unverified.
  Offline/API-envelope fixtures establish implementation behavior, not live
  availability. Public submission metadata is not a source-code API.
- The target counts only statements with approved per-problem editorial
  associations. A larger candidate pool or a curated manifest may be needed if
  explicit links are missing. Missing editorials do not discard good statements.
- Generic semantic duplicate detection is not implemented. Contest grouping,
  identical clean hashes, normalized variant titles and curated duplicate groups
  are supported. Manual review is still needed for other variants.
- HTML images without alt text produce a warning and remain recoverable in raw
  bytes; no OCR or image interpretation is generated. Oversized cleaned content
  requires explicit future chunking. Incomplete GridFS uploads without age
  metadata are reported; re-uploading the exact bytes repairs that content ID.
- Integration exercised a standalone MongoDB server, not Atlas, transactions,
  multiple workers or Docker. GridFS routing used a deliberately small threshold
  in tests; owned-document size rejection was separately tested.
- No retrieval models, classifiers, generated summaries, frontend or fine-label
  inference were implemented. Reviewed label import/projection infrastructure is
  available for later curated decisions.

After resolving permitted access, explicitly retry the blocked job and run a
20-problem pilot before expanding:

```powershell
.\.venv\Scripts\python.exe -m cpss retry --run-id 9f49ed68-7108-49d4-a41b-d53a7eff5100 --limit 20 --allow-pages
.\.venv\Scripts\python.exe -m cpss reconcile
# Only when the pilot has 20 editorial-ready problems and reconciliation passes:
.\.venv\Scripts\python.exe -m cpss resume --run-id 9f49ed68-7108-49d4-a41b-d53a7eff5100 --limit 1980 --allow-pages
```

To continue using permitted imported HTML instead, use the same run ID with
`--import-dir PATH` and, if needed, `--editorial-manifest PATH/editorials.json`.
The file layout is documented in `README.md`. The existing metadata jobs and
source identities can be reused without a second discovery run.
