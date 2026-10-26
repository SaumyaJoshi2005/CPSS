# Portable CPSS runtime

This directory adds a separate read-only Waitress application for Replit. It
serves the same **3,716 training problems** with BM25, frozen CPU MiniLM, hybrid
RRF, both views, rating filters and original-tag boosts. Public review writes and
classifier boosts are disabled. The original `python -m modeling serve` at
`http://127.0.0.1:8765`, including classifier and local review, remains available.

See [Replit setup](REPLIT.md), [GitHub handoff](GITHUB.md),
[actual verification](VALIDATION.md) and [upstream notices](THIRD_PARTY_NOTICES.md).
No remote repository, release, Replit project or paid resource has been created.

## Package and test on Windows

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r modeling/deployment/requirements.lock
.\.venv\Scripts\python.exe -m modeling.deployment package
$release = Get-Content modeling/artifacts/runtime/bundle-release.json | ConvertFrom-Json
.\.venv\Scripts\python.exe -m modeling.deployment unpack --archive $release.archive --sha256 $release.sha256 --output runtime
.\.venv\Scripts\python.exe -m modeling.deployment verify --bundle runtime
.\.venv\Scripts\python.exe -m modeling.deployment measure --bundle runtime --port 8766 --output modeling/artifacts/deployment-validation/new-measurement.json
```

Packaging reads the existing verified export/indexes. It does not make corpus
embeddings, fit classifiers, change splits or write MongoDB. Missing original
indexes must be restored using the existing local runbook. Bundles are
content-addressed directories with deterministic ZIP ordering/timestamps/modes;
identical producer inputs produce identical bytes. `bundle-release.json` records
the archive name, SHA-256 and manifest digest. Existing bundles are preserved.

The archive contains training documents/provenance, sparse BM25 and dense arrays,
pinned encoder/tokenizer, taxonomy, search configuration, UI, notices, dependency
contracts and 18 ranking probes. It excludes accepted code, validation/test
documents, full raw corpus, judgments, classifier pickles and environments.
Application source is delivered through its matching Git checkout and verified
using normalized UTF-8/LF hashes, so Git line endings are portable. Repackage
after serving-source/UI changes. Do not edit manifests to bypass mismatches.

## Install and run on Linux

Use Python **3.12 or 3.13**, x86-64 CPU with AVX2 for the selected quantized model.
Both Linux 3.12.3 and Windows 3.13.1 were tested with the same archive.

```bash
python -m pip install --only-binary=:all: -r modeling/deployment/requirements.lock
python -m pip check
python -m modeling.deployment unpack --archive /path/to/bundle-HASH.zip --sha256 ACTUAL_ARCHIVE_SHA256 --output runtime
python -m modeling.deployment verify --bundle runtime
export CPSS_MODE=deployment
export CPSS_PORT=8080
export CPSS_PUBLIC_ORIGIN=https://YOUR_ACTUAL_HOST
export CPSS_RUNTIME_DIR=runtime
python -m modeling.deployment serve
```

In another shell, check `curl -f http://127.0.0.1:8080/healthz` and
`curl -f http://127.0.0.1:8080/readyz`, then:

```bash
python -m modeling.deployment smoke --url http://127.0.0.1:8080 --origin https://YOUR_ACTUAL_HOST
```

Serving needs neither an editable corpus-package installation nor MongoDB.
The bootstrap exposes the checkout's pure `src/cpss` helpers. Deployment pins the
existing numerical, ONNX and tokenizer versions and adds Waitress 3.0.2; it omits
ingestion/database packages, training frameworks and GPU dependencies.

## Environment and behavior

| Variable | Meaning |
|---|---|
| `CPSS_MODE` | Default `local` binds 127.0.0.1; explicit `deployment` binds 0.0.0.0 |
| `PORT` / `CPSS_PORT` | Integer 1–65535; `PORT` wins; default 8765 local / 8080 deployment |
| `CPSS_PUBLIC_ORIGIN` | One exact browser origin; mandatory HTTPS in deployment |
| `CPSS_RUNTIME_DIR` | Default repository-relative `runtime/`; configurable absolute/relative directory |
| `CPSS_INIT_TIMEOUT` | 5–300 seconds, default 120; late results never become ready |
| `CPSS_FRAME_ORIGINS` | Explicit HTTPS frame parents, e.g. `https://replit.com` for Preview |
| `CPSS_BUNDLE_URL` | Optional public HTTPS archive URL, fetched only by explicit build `prepare` |
| `CPSS_BUNDLE_SHA256` | Independently recorded archive digest, required with URL |

See [.env.example](.env.example); set real environment variables rather than
expecting that example file to be loaded automatically. For a local browser at
HTTP port 8080, use `CPSS_MODE=local` and the matching HTTP origin. Deployment
mode intentionally rejects HTTP/mismatched browser origins. Origins are never
derived from Host/forwarded headers; no wildcard CORS or proxy trust is enabled.
TLS terminates at Replit. CSP permits shipped scripts/styles and configured
frame parents. Search bodies are limited to 16 KiB, with existing option checks.
Public errors omit tracebacks and filesystem paths. Public corpus/admin routes
are not exposed.

The listener and UI start before numerical initialization. `/healthz` means
alive; only `/readyz` 200 means verified search is available. Until then, search
returns 503 and the UI shows loading/retries. A single background initialization
checks payload hashes, package/model identities, ordered IDs, shapes/norms and
18 top-10 ranking probes; the encoder is warmed before readiness. Normal startup
is offline, with no downloads or corpus embedding. Timeout/failure remains
unready; a daemon initializer cannot be forcibly killed, and late results are
discarded. Restart after repairing assets. Logs show safe state/timing/memory
information. Four HTTP threads retain the existing global search lock and
encoder limits; no throughput/concurrency optimization is claimed.

## Verification and rollback

```powershell
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest tests modeling/tests modeling/deployment/tests -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m modeling.deployment github-check
```

`measure` starts a fresh server child, records liveness/readiness and memory,
runs HTTP checks and stops only that child. It does not flush OS disk caches or
simulate production load. All generated outputs stay ignored.

Rollback: stop the separate deployment server and run the original
`python -m modeling serve` in the existing Windows environment at port 8765.
Its corpus, cache, classifier and local judgment endpoints are preserved. For
a public rollback restore the matching source revision and previous verified
archive together, using a new runtime directory. Unpacking refuses to overwrite
a different existing bundle.
