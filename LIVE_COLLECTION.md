# Live collection update — 26 October 2026

**Historical direct-collection report.** The active `cpss` database has since
been replaced by the complete Open R1 corpus. The data described below is
preserved in `cpss_backup_before_openr1_20261026` and the original local artifacts.
See `OPENR1_COLLECTION.md` for current counts and exports.

The official [API introduction](https://codeforces.com/apiHelp),
[methods](https://codeforces.com/apiHelp/methods) and
[objects](https://codeforces.com/apiHelp/objects) were checked again. Collection
uses anonymous public requests with `lang=en` and at least 2.2 seconds between
API request starts. No API key or token file was used.

**Real content is now stored in MongoDB database `cpss`:**

| Item | Count |
|---|---:|
| Metadata candidates | 6,000 |
| Shared editorial documents | 4 |
| Problems with approved editorial sections | 33 |
| Current embedded code blocks | 32 |
| Problems with solution code | 31 |
| Ready statements | 0 |
| Ready statement + editorial pairs | 0 |

The four shared editorials are:

| Round | Blog ID | Approved problem sections |
|---|---:|---:|
| Codeforces Round 1129 | [158113](https://codeforces.com/blog/entry/158113) | 7 |
| WorldQuant / Codeforces Round 1128 | [157917](https://codeforces.com/blog/entry/157917) | 10 |
| DeepInfra / Codeforces Round 1126 | [157590](https://codeforces.com/blog/entry/157590) | 8 |
| Codeforces Round 1124, Div. 1 and Div. 2 | [157140](https://codeforces.com/blog/entry/157140) | 8 |

Editorial identities came from explicit official URLs found through web search.
Actual content came from `blogEntry.view`, not search snippets. Every approved
association is tied to an exact processed version and an explicit problem link.
The Round 1127 candidate (blog 157634) contained unfinished tutorial placeholders
and was excluded from verified sections.

The actual HTML exposed shared division headings. The extractor was updated to
recognize their boundaries, assign the shared explanation only to explicitly
linked selected identities, and leave ordinary prose cross-references inside
their correct section. The processor revision changed, preserving old versions.

Local inspection artifacts are in **`data/live-editorials-2026-10-26/`**:

- `raw/<blog_id>.json`: four original API response bodies.
- `editorial-sections.jsonl`: 33 approved current per-problem sections.
- `solution-code.jsonl`: 32 code blocks with problem, parent editorial version,
  code version, hash, available language and attribution.
- `manifest.json` and `manifest.sha256`: pinned source/version/hash references
  and actual collection counts.

Manifest digest:
`65c460d4d6cf996baed5b6a1a0a0645be3ba76363c5bfcb8e5405585a77106a5`.
These are partial editorial inspection artifacts, not a statement classification
export. Historical code versions are retained in MongoDB; the counts above refer
to code attached to the current editorial versions.

The independently resumable command is:

```powershell
.\.venv\Scripts\python.exe -m cpss collect-editorials --run-id 9f49ed68-7108-49d4-a41b-d53a7eff5100 --editorial-manifest config/live-editorials.json --limit 20
```

Four editorial jobs completed successfully. Repeating that command skips those
jobs. Adding verified identities to the manifest enables subsequent batches.
Statement jobs remain independent.

**The statement retry still returned HTTP 403**, and the batch stopped after
that denied request. The API's Problem object does not contain full statements;
API metadata cannot fill this gap. A permitted working page-access environment
or permitted HTML imports are required to collect statements. The 20-pair pilot
has not passed and the 2,000-pair corpus remains unfinished.

Final validation for this update: **42 tests passed** (26 offline and 16 real
MongoDB integration tests), including new shared-heading, missing-alias,
unfinished-editorial and editorial-only job/idempotency checks. Database
reconciliation passed with no issues. No statement access control was bypassed.
