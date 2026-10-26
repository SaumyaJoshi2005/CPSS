# Corpus collection and operations reference

This preserves the earlier corpus README, including historical direct-collection procedures. Run commands from the project root. Collection is already complete; the search application consumes the frozen export. For current project setup, start with [the project README](../README.md).

**Primary corpus source: `open-r1/codeforces` plus
`open-r1/codeforces-submissions` (`selected_accepted`).** The active workflow
imports every structurally complete ordinary round problem from pinned Open R1
snapshots. The original 2,000-problem target is no longer a limit. The direct
Codeforces collector below remains an optional secondary adapter.

```powershell
# Download only problem fields and selected accepted submissions; no private-test archive.
.\.venv\Scripts\python.exe -m cpss prepare-openr1 --output data/openr1

# A separate staging database keeps incomplete writes out of the active corpus.
.\.venv\Scripts\python.exe -m cpss --database cpss_openr1_stage_full_20261026 import-openr1 --snapshot data/openr1
.\.venv\Scripts\python.exe -m cpss --database cpss_openr1_stage_full_20261026 audit-complete

# Archives the previous active collections, then promotes the audited staging corpus.
# This was already executed for the current corpus; use a NEW staging name for a future snapshot.
.\.venv\Scripts\python.exe -m cpss activate-openr1 --staging-database cpss_openr1_stage_full_20261026
.\.venv\Scripts\python.exe -m cpss audit-complete
.\.venv\Scripts\python.exe -m cpss reconcile
```

`prepare-openr1` pins both repository revisions, saves upstream README/license
declarations, and reads selected Parquet columns over HTTP ranges. Exact projected
row bytes are saved and hashed; the manifest distinguishes these verified local
hashes from upstream-declared full-Parquet LFS hashes. It resumes completed files
without changing their revisions. The accepted-code subset is used rather than
the much larger full submission or generated reasoning datasets.

Admission requires valid matching IDs, contest name/type/date, title, rating,
nonempty original tags, positive time/memory limits, nonempty description and
input/output sections, at least one complete example pair, a non-placeholder
editorial and at least one accepted contestant source. Up to three distinct code
sources are retained per problem, preferring `og_source` when available. Alias
IDs can supply accepted code and are grouped transitively during export.
Interactive and special-format contests, unresolved images and incomplete rows
are excluded. Educational rounds are eligible even when their scoring type is
ICPC; scoring type does not determine whether a round is a regular contest.

Each accepted problem has structured examples, resource limits, aliases,
provider/revision/split, upstream execution/test-completeness flags, and accepted
submission IDs in `problems.corpus`. Full statement sections are in immutable
versions and the current complete text remains in `problems.statement`.
Statement/editorial versions retain the source problem row; solution versions
retain the source submission row, language, verdict and available provenance.
`reprocess` recognizes these structured source records.

Re-run the same `import-openr1` staging command to resume. Rejected rows and
reasons are in `data/openr1/import-report-<database>.json`; rejected rows never
become active problem metadata. A pilot can use `--limit 20` with a separate
staging database. Upstream `test` rows stay in the test split of exports.

`activate-openr1` runs completeness/reference audits before touching the active
database. It archives previous collections into a named `cpss_backup_*` database
using non-overwriting collection renames and records each step in
`data/activation-<staging>.json`. Renames are atomic per collection, not across
the entire corpus; rerunning the command resumes an interrupted cutover.
Existing annotation decisions require an explicit merge before promotion.
The previous metadata-only candidates and blocked jobs are removed from the
active corpus while their historical editorial content remains recoverable in
the backup database and prior local artifacts.

"Complete" means all required corpus components passed structural and identity
checks. It does not certify every upstream equation or editorial manually, and
does not mean all private judge tests were recovered. Solution verdicts are
upstream accepted verdicts; this importer does not execute downloaded code.
Notes are retained when present rather than fabricated when absent. Open R1's
snapshot coverage ends in early 2025; this is not a live 2026 corpus.

See [the collection record](../OPENR1_COLLECTION.md) for the final import, exclusion and cleanup counts.

## Optional legacy direct Codeforces collection

Python + MongoDB ingestion for **Algorithm-Aware Semantic Retrieval of Competitive
Programming Problems Using Hybrid NLP Models**. This application collects,
validates, versions, reconciles and exports corpus data. Retrieval, classifiers,
generated summaries and a frontend are outside its scope.

The original direct-collection target was **2,000 recent ordinary Codeforces problems with validated
editorial sections**. Statements without editorials remain usable in storage;
they do not count toward that target. Editorials do not guarantee public solution
code. Embedded editorial code is optional, and missing code never blocks a statement.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --progress-bar off -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m cpss doctor
.\.venv\Scripts\python.exe -m cpss init-db
```

MongoDB is read from `CPSS_MONGODB_URI`, default `mongodb://127.0.0.1:27017/`,
and database `CPSS_DATABASE`, default `cpss`. Copy `.env.example` to `.env` only
if custom settings are needed. Never commit secrets or use `token.txt`; the
application refuses to import a file with that name. Errors report sanitized codes,
never driver/HTTP exception messages or connection URIs.

The inspected machine already runs MongoDB **6.0.29**, with `mongosh` **2.6.0**.
Docker's client is installed, but its engine was stopped. Docker is unnecessary
for this setup. The application does not change existing service configuration.
If using a different machine without MongoDB, an optional isolated setup is:

```powershell
docker run --name cpss-mongo -p 127.0.0.1:27017:27017 -v cpss-mongo-data:/data/db -d mongo:8.0
```

Do not run that command on an occupied port. No replica set is required.

`requirements.lock` records the versions actually validated on Python 3.13.
If pip stalls during remote metadata resolution, `tools/fetch_wheels.py` is a
fallback that downloads compatible wheels from official PyPI, verifies their
published SHA-256 hashes and installs them locally without dependency resolution.
Standard pip setup is preferred.

## Offline fixtures and tests

All supplied problems/editorials are **synthetic**, explicitly marked in their
content. Keep them out of the real `cpss` database:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not mongo" -q
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest -m mongo -q

.\.venv\Scripts\python.exe -m cpss --database cpss_fixture_demo init-db
.\.venv\Scripts\python.exe -m cpss --database cpss_fixture_demo discover --offline-catalog tests/fixtures --limit 4 --candidate-limit 4
# Replace RUN_ID with the returned ID:
.\.venv\Scripts\python.exe -m cpss --database cpss_fixture_demo ingest --run-id RUN_ID --limit 4 --import-dir tests/fixtures --editorial-manifest tests/fixtures/editorial-manifest.json
.\.venv\Scripts\python.exe -m cpss --database cpss_fixture_demo reconcile
.\.venv\Scripts\python.exe -m cpss --database cpss_fixture_demo export --run-id RUN_ID --output exports/fixture-demo
.\.venv\Scripts\python.exe -m cpss verify-export exports/fixture-demo
```

MongoDB integration tests create unique `cpss_test_<uuid>` databases and delete
only those exact disposable databases. If `CPSS_TEST_MONGO` is unset, they are
reported as **skipped**, not passed. `CPSS_TEST_MONGODB_URI` optionally selects
another test server. Offline tests use local HTML and mocked HTTP; they do not
establish live Codeforces access.

## Live discovery, pilot and expansion

```powershell
# Official contest.list + problemset.problems, once each (or fresh cached bytes).
# Discovers up to 6,000 candidates to allow for missing editorials.
.\.venv\Scripts\python.exe -m cpss discover --limit 2000

# First validate a small live pilot, using the returned run ID.
.\.venv\Scripts\python.exe -m cpss ingest --run-id RUN_ID --limit 20 --allow-pages
.\.venv\Scripts\python.exe -m cpss reconcile

# Only after a successful pilot and reconciliation:
.\.venv\Scripts\python.exe -m cpss resume --run-id RUN_ID --limit 1980 --allow-pages
.\.venv\Scripts\python.exe -m cpss export --run-id RUN_ID --output exports/cpss-2000
.\.venv\Scripts\python.exe -m cpss verify-export exports/cpss-2000
```

Use `--order oldest`, `--since 2024-01-01`, `--until 2026-01-01` (exclusive),
or repeated `--contest-id` overrides to configure selection. Dates come from
actual `startTimeSeconds`, never contest ID order. Default selection includes
finished standard Codeforces/Educational/Global rounds with `type=CF`, excludes
Gym/group contests and named team, Kotlin, April Fools, marathon and experimental
events. `--contest-id` overrides the name filter but not finished/programming/CF
checks. Catalogue discovery is metadata, not successful statement collection.

The `--limit` on ingestion means **additional editorial-ready problems**, bounded
by at most three times that number of attempted jobs. Missing editorials may
therefore require another batch or a larger candidate pool. Every batch returns
actual statement/editorial/code counts and job errors. One OS worker lock prevents
simultaneous CLI workers sharing the cache directory. Use the same cache for all
commands that contact Codeforces.

Editorial links come only from explicit Tutorial/Editorial anchors in the statement
page or a curated manifest. The robots policy disallows `/contest` page crawling,
so the collector uses `/problemset/problem/...` and does **not** scrape contest
pages for links. No invented problem-to-editorial endpoint or public submission
source endpoint is used.

To supply independently checked editorial identities:

Editorials can also be collected independently while statement pages are
unavailable. This command uses only the official API and a curated manifest;
it creates resumable jobs for shared editorials and skips completed jobs:

```powershell
.\.venv\Scripts\python.exe -m cpss collect-editorials --run-id RUN_ID --editorial-manifest config/live-editorials.json --limit 20
```

Here `--limit` bounds editorial **documents**, each of which may cover several
problems. Use `--retry` explicitly for failed jobs after resolving their cause.
`editorial_sections_ready` counts problems with approved editorial sections;
`editorial_ready` continues to count only those that also have ready statements.
An editorial-only batch does not complete the paired corpus target.

The initial real batch is described in `LIVE_COLLECTION.md`; its raw responses,
readable sections, code and hash manifest are in `data/live-editorials-2026-10-26`.
Shared headings may explicitly link both division aliases of the same problem.
Only the linked identities present in the discovered catalogue receive those
sections. Unfinished "Tutorial is loading" sections are excluded from approval.

Manifest format:

```json
{
  "entries": [
    {
      "contest_id": 1234,
      "blog_id": 56789,
      "problem_ids": ["codeforces:1234:A"],
      "anchors": {"problem-a-heading-id": "codeforces:1234:A"}
    }
  ]
}
```

`problem_ids` and `anchors` are optional. A manifest supplies candidates, not
proof: the blog ID is verified and each section must map to the relevant problem.
Only unique explicit problem links or checked DOM anchors are approved
automatically. Index-only headings stay proposed. The full contest editorial is
stored once and never copied as every problem's solution. Use
`--editorial-manifest path.json` on every applicable ingest/resume/retry invocation.

Imported content uses `contests.json`, `problems.json`,
`statements/<contest_id>/<index>.html`, and `editorials/<blog_id>.json`. Editorial
imports retain the official `blogEntry.view` JSON envelope. Imported statements
can carry explicit tutorial links. Use `--import-dir` to process permitted local
content without contacting Codeforces.

## Access and operational rules

Contracts and access references were inspected on **26 October 2026**:

- [API introduction](https://codeforces.com/apiHelp): one request per two seconds;
  this application enforces at least **2.2 seconds between API request starts**.
- [API methods](https://codeforces.com/apiHelp/methods): `contest.list`,
  `problemset.problems`, known-ID `blogEntry.view`. `contest.status` source access
  requires contest-manager permissions; it is not used here.
- [Problem and BlogEntry objects](https://codeforces.com/apiHelp/objects): full
  statements are not part of Problem metadata; blog content is available in full
  BlogEntry responses.
- [Terms](https://codeforces.com/terms) and [robots policy](https://codeforces.com/robots.txt):
  page collection requires `--allow-pages`, checks the current robots policy, uses
  an identified CPSS collector agent and has a separate **5-second** limiter.
  Recheck access rules before future collection. Content is retained for research
  with provenance and attribution; this project grants no redistribution rights.
- [PyMongo compatibility](https://www.mongodb.com/docs/drivers/compatibility/),
  [GridFS](https://www.mongodb.com/docs/manual/core/gridfs/), and
  [schema validation](https://www.mongodb.com/docs/manual/core/schema-validation/)
  informed the storage implementation. Actual tests use the running server.

The client caches successful responses (API: one hour; pages/robots: one day).
`--refresh` explicitly fetches again. Cached versions retain their original
retrieval timestamp. API and page rate timestamps persist across CLI commands.
Requests have 10-second connect / 30-second operation timeouts, a 64 MiB response
cap, at most three attempts for transient failures, capped jittered backoff and
`Retry-After` handling. Delays beyond 60 seconds are deferred to job retry state.
Redirects are bounded and checked against HTTPS and the exact allowed host.

HTTP 401/403, robot denial, challenge/login HTML (including HTTP 200), invalid API
envelopes and incomplete statements cannot become corpus versions. A denied
statement origin stops the batch. Do not bypass challenges, rotate identities or
use authenticated browser sessions to circumvent access restrictions.

## Storage, refresh and recovery

The nine owned collections are `problems`, `source_documents`, `blobs`,
`document_versions`, `editorial_links`, `taxonomy`, `annotations`,
`ingestion_jobs`, and `ingestion_runs`. GridFS adds its standard `fs.files` and
`fs.chunks` collections only when needed. See `src/cpss/schemas.py` for shared
application/MongoDB schemas and exact index definitions.

- Problem identity is `codeforces:<contestId>:<index>`, independent of content.
- Raw SHA-256 hashes the **decompressed response bytes actually retained** by
  HTTPX. HTML is interpreted as strict UTF-8. No body re-encoding precedes hashing.
- Clean SHA-256 hashes deterministic `clean_text.encode('utf-8')`.
- Version identity hashes canonical JSON with `source_document_id`,
  `raw_sha256`, and `processor_fingerprint`. The fingerprint includes parser,
  cleaner, Beautiful Soup version, language and output-affecting configuration.
- Raw bytes and clean versions are immutable. Different raw HTML may have the
  same clean hash; the clean-hash index is deliberately nonunique. No embeddings
  are built by this project.
- Every document has `schema_version=1`; timestamps are UTC BSON Dates. Unknown
  ratings are omitted. Current complete statement text is directly in `problems`.
- Raw bodies above 8 MiB route to GridFS. Every owned document receives an
  explicit encoded-BSON size check with a 15 MiB ceiling. Oversized clean
  documents/links are rejected for explicit future chunking, never truncated.
- Write order is blob → source identity → immutable version → source pointer →
  atomic statement projection → job success. Per-source observation sequences
  prevent old retries from replacing newer published state.
- Failures retain good statement text and human label decisions. Statement and
  editorial jobs are independent. Historical annotations remain tied to exact
  evidence versions; exports flag whether that evidence is still current.

```powershell
.\.venv\Scripts\python.exe -m cpss reconcile
.\.venv\Scripts\python.exe -m cpss reconcile --repair
.\.venv\Scripts\python.exe -m cpss resume --run-id RUN_ID --limit 20 --allow-pages
# Explicit retry after resolving the failure; never an automatic block bypass:
.\.venv\Scripts\python.exe -m cpss retry --run-id RUN_ID --limit 20 --allow-pages
.\.venv\Scripts\python.exe -m cpss reprocess
.\.venv\Scripts\python.exe -m cpss reprocess --source-id cf:statement:1234:A:en
```

Reconciliation checks source/version/blob references, editorial sections,
annotation references, interrupted jobs, current statement projections and
abandoned GridFS files. `--repair` rebuilds projections, makes interrupted jobs
retryable and removes unreferenced **complete** uploads older than an hour.
Incomplete GridFS chunk sets have no reliable age metadata: they are reported
for explicit investigation rather than blindly deleted. Back up MongoDB and
GridFS together before maintenance.

## Taxonomy and annotations

No fine labels or summaries are generated. Original tags stay untouched.
Taxonomy records support multiple parents and are cycle checked. Versioned IDs
(for example `v1:graphs`) avoid rewriting old vocabulary entries.

`import-taxonomy labels.json` imports an array matching the taxonomy schema;
`annotate proposals.json` imports decisions with `problem_id`, `label_id`,
`taxonomy_version`, `origin`, `producer_version`, version-specific `evidence`,
`confidence_policy`, `status` and `reviewer`. Evidence includes `version_id`, a
section key as `locator`, and a short exact `span`. The application verifies
ownership/association of evidence and taxonomy references.

Origins distinguish `platform_map`, `editorial`, `human`, and `model`.
Deterministic maps should document their policy and omit model probability.
`review ANNOTATION_ID --status approved --reviewer NAME` changes a decision and
atomically rebuilds the bounded approved-label projection. Reimporting an
existing proposal does not overwrite review decisions. Fine-label gold subsets
must use reviewed evidence; a missing platform tag is not a negative label.

Optional permitted source files can be imported without requesting submission
pages:

```powershell
.\.venv\Scripts\python.exe -m cpss import-solution solution.cpp --problem-id codeforces:1234:A --url https://codeforces.com/contest/1234/submission/56789 --language cpp --attribution AUTHOR --submission-id 56789
```

`--editorial-version-id` can tie the import to an existing approved association.
The URL and submission ID are provenance only; the command never fetches them.

## Exports and evaluation boundaries

Exports create a new immutable output directory and refuse to overwrite one.
They contain:

- `statement-only.jsonl`: classifier features are **only `input.statement`**.
  Tags, labels, rating and other metadata remain outside `input` as frozen
  targets/audit metadata. Do not feed the entire JSON record to a classifier.
- `editorial-enriched.jsonl`: a deliberate separate experimental input with
  statement plus the approved, version-pinned editorial **section**.
- `manifest.json` and `manifest.sha256`: exact problem/version/hash references,
  metadata, editorial association snapshots, annotation decisions and label
  values, taxonomy versions, configuration, counts, split assignments and file hashes.
- `versions.jsonl` and `blobs/<sha256>`: pinned clean records and original bytes
  for independent audit and restoration of content without network access.

Splits group contests, identical clean text, normalized easy/hard variant titles
and explicit duplicate groups. Supply additional known groups with
`--duplicate-groups groups.json` (an array of problem-ID arrays). Cross-contest
duplicate date intervals are merged to keep train/validation/test chronology
strict. This can reduce split balance; a very small fixture set may have empty
splits. Default target fractions are 0.8/0.1/0.1, adjustable with `--fractions`.
Unknown semantic duplicates still require manual curation. Relevance judgments
for retrieval must be independently assessed in the later retrieval stage.

## Non-destructive migrations and query plans

`init-db` creates absent owned collections and indexes, but refuses to replace a
different existing validator. `migrate` reports incompatible records without
writing. `migrate --apply` requires all records already to conform before applying
validators/indexes. Future schema transformations should write a new collection,
validate it, compare counts/references and back it up before an explicit cutover;
never silently discard fields or rewrite immutable processing versions.

Use workload-specific explains before adding indexes. For example in `mongosh`:

```javascript
use cpss
db.problems.find({"quality.statement":"ready",rating:{$gte:1200,$lte:1800}})
  .hint("ready_rating").explain("executionStats")
db.problems.find({"approved_labels.broad":"v1:graphs",rating:{$gte:1200}})
  .hint("broad_rating").explain("executionStats")
db.document_versions.find({clean_sha256:"<real-sha256>"})
  .hint("clean_hash_lookup").explain("executionStats")
```

No raw text/body indexes or MongoDB hashed indexes are needed. Broad and technique
arrays use separate compound indexes; they are not compounded together.

See [the validation record](../VALIDATION.md) for actual execution results and live collection counts.
