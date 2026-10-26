# CPSS submission runbook

Run PowerShell from `C:\New folder (2)\Proj\NLP_mini_project`.
The local application is currently available at **http://127.0.0.1:8765**.
Use Ctrl+C in its launch terminal to stop it. Do not start a second process on
that port; use `--port 8766` if the first process is still running.

## Environment and indexing

Use the existing Python 3.13 `.venv`. The corpus package and root dependencies
are already installed. No new package was needed for the HTTP application,
BM25 or fusion. Dense search uses the existing pinned ONNX/tokenizers runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r modeling/requirements.lock -r modeling/requirements-encoder.lock
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m modeling verify
.\.venv\Scripts\python.exe -m modeling index
```

`index` verifies/reuses both completed indexes. It does not retrain classifiers
or download another encoder. The existing MiniLM assets must remain in
`modeling/.cache/encoders/97127f7df3429a24/`; their hashes and runtime versions are
checked. Preserve that folder for an offline handoff. Statements reuse matching
training vectors from the older cache when available; otherwise they are
embedded locally. The editorial-enriched index took about 10.8 minutes on this
machine. Both views search the same **3,716 training problems**.

If the encoder assets/runtime are unavailable, the working fallback is:

```powershell
.\.venv\Scripts\python.exe -m modeling index --lexical-only
.\.venv\Scripts\python.exe -m modeling search --method bm25 --view editorial --query "DSU with connectivity queries."
```

This selects lexical-only indexes; it retains existing dense artifacts. Run
ordinary `index` again to reactivate dense search after restoring the pinned
cache. A fresh machine without the model cache cannot run dense/hybrid offline.
See the earlier `prepare-encoder` command only if deliberately restoring the
same pinned model with networking available.

## Search and launch

```powershell
.\.venv\Scripts\python.exe -m modeling search --method hybrid --view editorial --query "DSU with connectivity queries." --limit 10 --candidates 100
.\.venv\Scripts\python.exe -m modeling search --method dense --view statement --query "Maintain connected groups as new connections are added."
.\.venv\Scripts\python.exe -m modeling search --method hybrid --view editorial --rating-min 1200 --rating-max 2000 --query "Binary search on the answer with a greedy feasibility check."
.\.venv\Scripts\python.exe -m modeling serve
```

Open `http://127.0.0.1:8765`. Select the method and context view, try the three
example buttons, and inspect source links, original tags and exact snippets.
The header distinguishes searchable coverage from the complete corpus size.
The measured result time excludes index loading and browser/HTTP overhead.

Rating syntax in queries also supports `rating >= 1200`, `rating <= 2000` and
`rating 1200-2000`. Text and form filters are intersected; conflicting bounds
produce a clear error. `without dp` filters the observed original tag, not every
possible solution. Unsupported fine-grained exclusions are rejected rather than
approximated by a broader tag. Alias parsing is deterministic; it is not a
trained query-understanding model.

CLI `--tag-boost` and `--classifier-boost` correspond to the UI's optional ranking
options. Both default off and were off in the benchmark. Classifier boosts use
uncalibrated statement-trained query scores; they never remove candidates.

## Development evaluation and human review

```powershell
.\.venv\Scripts\python.exe -m modeling retrieval-pool
.\.venv\Scripts\python.exe -m modeling retrieval-evaluate
.\.venv\Scripts\python.exe -m modeling retrieval-latency
```

The existing pool contains 24 queries (eight named, eight unnamed, eight
combined) and **814 distinct query–problem pairs**, pooled from top ten results
for three methods and two views. Repeating `retrieval-pool` reuses the frozen
pool after provenance checks. It does not overwrite judgments.

Open **http://127.0.0.1:8765/review**. Supply your reviewer identifier, read the
query and original evidence, enter a rationale, personally attest to the review
and choose grade 2/1/0. This is the human reviewer workflow; automated scripts
must not submit human attestations. The UI and CSV hide originating methods,
ranks, scores and view. The raw `pool.json` retains rankings for reproducibility,
so this is reviewer-interface blinding, not cryptographic blinding.

Alternatively, fill **grade** and **rationale** in
`modeling/artifacts/retrieval-evaluation-v1/review-sheet.csv`. After a person has
actually reviewed those rows:

```powershell
.\.venv\Scripts\python.exe -m modeling import-judgments --file modeling/artifacts/retrieval-evaluation-v1/review-sheet.csv --reviewer "YOUR_NAME" --human-attested
.\.venv\Scripts\python.exe -m modeling retrieval-evaluate
```

Judgments are append-only individual JSON files; corrections preserve older
decisions and evaluation uses the latest decision per pair. No labels are
written to the corpus or MongoDB. The current real review has **zero human
judgments**; NDCG/MRR are therefore unavailable, not zero.

The evaluator includes a query only after its complete pooled set is judged.
The same complete-query subset is used for all methods/views. NDCG@10 uses gains
`2^grade - 1`; MRR@10 requires grade 2. The ideal ranking is within the judged
pool, not all problems. Unknown documents are never silently treated as grade
zero. Recall is omitted because exhaustive corpus relevance is unknown.

To change queries/settings, use a copy of `search-v1.json` with a new
`evaluation_path`, and pass `--search-config PATH` to commands. Representation or
configuration changes require compatible rebuilt indexes. Do not edit a frozen
pool to make results look better.

## Checks and handoff

```powershell
$env:CPSS_TEST_MONGO = '1'
.\.venv\Scripts\python.exe -m pytest tests modeling/tests -q
.\.venv\Scripts\python.exe -m pip check
# Optional developer client checks; Node is not an application dependency.
node --check modeling/application/static/app.js
node --check modeling/application/static/review.js
node modeling/tests/ui-smoke.cjs
```

Actual verification: **104 tests passed**, including disposable MongoDB
integration tests; dependency checks passed. Local HTTP and shipped-client
script integration exercised all three demo queries, empty results, invalid
ratings and empty input. This client check uses a DOM adapter and does not
claim browser layout verification. Browser automation was stopped by its URL
verification safeguard; no screenshot or successful visual review is claimed.

Keep source files, configs, docs, frozen export, model cache and generated search
indexes for the local demonstration. Generated indexes/caches remain ignored by
source control; no corpus copy or nested environment/repository was created.
The final classifier test split remains reserved. Do not run `evaluate --split
test --final` as a submission rehearsal.
