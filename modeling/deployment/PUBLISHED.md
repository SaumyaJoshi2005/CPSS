# Publication status — 27 October 2026

- Source repository: https://github.com/SaumyaJoshi2005/CPSS
- Runtime release: https://github.com/SaumyaJoshi2005/CPSS/releases/tag/v1.0.0-demo
- Replit workspace: https://replit.com/@saumyajoshi2008/CPSS
- Searchable coverage: **3,716 training problems** from the frozen 4,648-problem corpus.
- Replit Preview: **verified**, including real browser search.
- Production: **publishing in progress; the stable public URL is not verified yet**.

## Verified in this publication session

The uploaded ZIP is 37,882,870 bytes (36.13 MiB), with SHA-256:
`cfdad1c91a1f81e23916862466593952d76db7ec61ed7d4e1e8e0b2e84fc9258`.

Replit verified all payload/source hashes, dependency versions and saved ranking
probes. The Preview HTTP smoke check passed BM25/dense/hybrid, both views,
ratings/tag boosts, invalid inputs, origin validation, read-only mode and static
assets. Three fixed queries with three repeats per method/view produced warm
mean HTTP times of approximately 9–16 ms; this is a small development sample,
not a load test or relevance score. Index initialization in that smoke check was
2.232 seconds. Browser search also returned actual ranked Codeforces links and
editorial excerpts.

A clean second environment using Nix Python 3.12.11 also passed runtime and
ranking verification after native-library configuration. GitHub Actions passed
the pinned offline test workflow on Python 3.12/3.13, dependency checks,
JavaScript syntax checks and publication preflight. Corpus collection,
classifier training and reserved-test evaluation were not rerun.

## Hosting configuration

Replit's importer selected the Python 3.11 tooling module, so `.replit` also
provisions Python 3.12 through Nix. The server always uses a supported 3.12
interpreter. `replit.nix` supplies the C++ runtime and zlib needed by numerical
wheels. `PIP_USER=false` keeps installation inside the virtual environment.

Preview uses `.venv/bin/python` and the **CPSS server** Run workflow.
Production build creates `.deploy-venv`, installs exact serving pins, downloads
the fixed release ZIP, checks SHA-256 and validates rankings. Production startup
uses `.deploy-venv/bin/python -m modeling.deployment serve`, with no downloads.

Set **CPSS_PUBLIC_ORIGIN** to the exact assigned Preview origin for development.
Set a separate deployment value to the exact public HTTPS origin before
publishing. Wildcard origins and forwarded-host inference remain disabled.
No MongoDB, new model training, corpus expansion or paid add-on is needed.

Use the current **main** branch for the refined hosting configuration.
The v1.0.0-demo asset retains its original serving-source provenance; subsequent
changes are deployment configuration/documentation, not ranking or model changes.
