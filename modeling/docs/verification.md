# Verification executed — 26 October 2026

The authoritative frozen export passed the existing command:

```powershell
.\.venv\Scripts\python.exe -m cpss verify-export exports/openr1-complete-2026-10-26-v2
```

The modeling loader also invoked that existing verifier and saved an integrity
receipt. Modeling verification subsequently passed with unchanged manifest
`2572bdbe3448a230132ce759722aa5353d4f5bb57dc204597a0447e2a82cdc93`
and split counts 3,716 / 467 / 465. The corpus and MongoDB were not modified.

Actual commands/checks completed:

| Check | Result |
|---|---|
| Training-only label audit on the frozen corpus | Passed; 29 selected labels, no malformed/empty/unsupported training targets |
| Real baseline training and validation | Completed; micro F1 0.549974, macro F1 0.474177 |
| Frozen-model validation evaluation | Reproduced the saved metrics; no threshold refitting |
| Unseen synthetic statement CLI prediction | Returned scores and selected labels successfully |
| Deterministic context pilot preparation | 100 training references; 13 DSU, 14 binary-search assignments; no generation/review performed |
| Context illustration validation | Two synthetic problem drafts and one query draft passed; all remain unreviewed |
| New modeling tests | 16 passed |
| Existing corpus + modeling tests | **77 passed** (61 existing + 16 new) |
| `pip check` | Passed; no broken requirements |
| Real final test evaluation | **Not run**; no test-evaluation claim/report exists in the real run |

Combined tests were run with `CPSS_TEST_MONGO=1`; existing database integration
checks used their uniquely named disposable databases. The real test split was
not evaluated: only a synthetic unit fixture exercised the final-evaluation guard.

The new tests cover taxonomy aliases/multiple parents/unreviewed fine labels;
invalid schemas/labels/configurations; training-derived label selection;
unsupported/absent labels; sparse statement-only features and validation-vocabulary
isolation; feature/cache invalidation and corruption; validation threshold fallback;
fitting split/identity guards; stable label order; in-memory/save/load inference
consistency; single-label score shape; protected final evaluation; deterministic
training-only context sampling; query span and problem source-evidence validation;
review-status guards; frozen export verification/cache invalidation; and invalid
inference inputs.

A Windows path-length failure surfaced in the first new test pass. Cache directory
and atomic temporary names were shortened while full digests stayed in metadata;
the final test suite passed. This did not alter the baseline's text features or
fitted parameters. The existing wheel-bootstrap helper was also corrected to
skip invalid legacy PyPI version strings encountered in joblib's release history.

Network access was used only for primary library documentation and missing Python
dependencies. Modeling, corpus audit, training and validation ran offline. No
neural model, retrieval model, frontend or bulk context generation was executed.
