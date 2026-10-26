# Deployment verification — 27 October 2026

This is actual local verification, not a Replit deployment or a new relevance
evaluation. No classifier training, ranking tuning or test-split evaluation was
performed. All **112 protected files** in the pre-change artifact snapshot were
rehash-checked and unchanged, including frozen classifier runs, original search
indexes, encoder assets, source manifests and experiment configurations.

## Final release

- Archive: `modeling/artifacts/runtime/bundle-facf415e48fc9777c9ae.zip`.
- Archive size: **37,882,870 bytes (36.13 MiB)**.
- Payload size excluding its manifest: **57,097,373 bytes (54.45 MiB)**; 21 files.
- Archive SHA-256:
  `cfdad1c91a1f81e23916862466593952d76db7ec61ed7d4e1e8e0b2e84fc9258`.
- Manifest SHA-256:
  `facf415e48fc9777c9aee6af1c3678161ca5eece8b1fa031c2c608d69e9db722`.
- Source corpus manifest:
  `2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`.
- Searchable: **3,716 unique ordered training IDs**; corpus counts remain
  3,716 train / 467 validation / 465 test. No held-out documents were packaged.
- Data content identity:
  `76f473fd08f2b0631530c667c0945bf1e31ecd9c8a248576b7c45e9b118c1899`.

Packaging the final release twice produced the same archive SHA-256. Its
matching source checkout has 32 pinned serving-source/dependency files plus
three UI hashes. The format is `cpss-serving-bundle-v1`; existing index, context,
tokenizer and encoder versions are preserved in the manifest. Model revision:
`sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The CPU quantized ONNX file, chunking/pooling and all ranking parameters remain
unchanged. The optional classifier is omitted only from the public runtime.

## Portability and measurements

The same Windows-built archive was unpacked to different runtime paths on
Windows and Ubuntu, without rebuilding embeddings or indexes. Both passed all
18 top-10 identity/score probes before reporting ready. Exact package versions
remain required; only the data-only format supports the explicitly checked
Python minor-version/platform boundary. Original pickle/cache validators were
not weakened. UTF-8/LF source identity tolerates Git checkout line endings.

Latest fresh-process measurements, using already-present disk assets:

| Observation | Windows 11 / Python 3.13.1 | Ubuntu WSL2 / Python 3.12.3 |
|---|---:|---:|
| Process start to liveness | 0.560 s | 0.429 s |
| Process start to verified readiness | 3.063 s | 11.457 s |
| Verification/index/encoder probes after imports | 0.999 s | 4.996 s |
| RSS at readiness | 231.0 MiB | 243.8 MiB |
| Peak RSS by readiness | 235.4 MiB | 243.8 MiB |
| Statement BM25 mean HTTP latency | 15.9 ms | 119.6 ms |
| Statement dense mean HTTP latency | 21.1 ms | 116.3 ms |
| Statement hybrid mean HTTP latency | 22.7 ms | 119.3 ms |
| Editorial BM25 mean HTTP latency | 25.3 ms | 110.2 ms |
| Editorial dense mean HTTP latency | 24.8 ms | 111.3 ms |
| Editorial hybrid mean HTTP latency | 28.6 ms | 122.4 ms |

Each method/view has nine timed requests: three fixed development queries,
three repeats, including HTTP overhead. Encoder initialization already completed
in readiness probes. Polling resolution was 50 ms. These are process cold starts,
not flushed-disk cold starts. WSL used the Windows-mounted project filesystem;
OS, Python, filesystem and HTTP scheduling differ. Earlier local readiness
observations were 4.223 s Windows and 14.647 s WSL. These small serial samples
are not throughput, load, percentile, relevance or actual Replit benchmarks.
Concurrency remains serialized by the original search lock.

## Checks actually run

- Full Windows suite: **133 passed** (104 existing + 29 deployment tests), with
  `CPSS_TEST_MONGO=1`. Integration tests used disposable MongoDB databases.
- New deployment tests on actual Linux: **29 passed**. Test-only pytest 9.1.1
  was installed into an isolated ignored Linux environment, not the Windows one.
- `pip check`: passed in the Windows environment and isolated Linux serving
  environment. The serving lock contains 34 exact packages; Waitress 3.0.2 is the
  only added package in the existing Windows environment.
- HTTP smoke on both platforms: BM25/dense/hybrid, both views, rating bounds,
  tag boosts, empty results, empty/invalid queries, forbidden classifier boost,
  source links, static files, liveness/readiness, wrong-origin rejection and
  disabled review/judgment writes passed.
- Actual HTTP plus shipped JavaScript DOM-adapter checks passed against both
  original localhost app and public runtime, including hidden public controls,
  loading/busy state, reset, source escaping and errors. Node syntax checks
  passed. This is not browser rendering or visual QA.
- Explicit `prepare --no-install` passed against the unpacked runtime; it
  performed no model download or corpus embedding.
- `.replit` parsed successfully as TOML. GitHub candidate preflight passed;
  generated assets/private files are ignored and were not read by that check.
- GitHub Actions configuration was prepared, **not executed remotely**.

Focused tests cover host/port defaults, explicit deployment, HTTPS origins,
read-only routes, body limits, liveness/loading/ready/failure/timeouts, single
initialization, sparse round trips, reproducible archives, document ordering,
corrupt/missing payloads, dependency/source contracts, path traversal and source
line endings. The original prediction/retrieval tests still pass.

Machine-readable measurements and the preservation receipt are under ignored
`modeling/artifacts/deployment-validation/`; the pre-change snapshot is under
`modeling/artifacts/maintenance/deployment-20261027/`. Historical evaluation
manifests were preserved; they describe their historical source snapshot and
are not rewritten to pretend this deployment source existed at that time.

## Remaining platform checks

No Replit account/project or URL was available. Preview, Replit machine CPU/RAM,
production cold starts and publication remain to be checked using [REPLIT.md](REPLIT.md).
The local Git repository is ready on `main`; no commit, remote or push was made.
Publishing and CPSS's own source-license choice remain with the owner. Upstream
notices and Apache 2.0 text accompany the archive.
