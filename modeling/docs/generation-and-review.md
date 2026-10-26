# Context pilot instructions v1

This milestone prepares inputs and a review contract; it does not run bulk
generation. The 100-problem sample is training-only. Neither validation nor test
problems may enter context generation, prompt iteration or annotation review.
The shared algorithm vocabulary is `config/taxonomy-v1.json`.

## Draft problem-context generation instruction

Given the pinned statement and its problem-specific editorial evidence, emit one
JSON object matching `schemas/context-v1.schema.json`, problem-context-v1 branch.
Keep `status=draft`, `review.decision=pending`, and reviewer/timestamp null. Record
the actual model revision, prompt version and source manifest. A model output is
never human-reviewed merely because it passes schema validation.

Abstract the underlying objective, structures, operations and useful observations.
Preserve applicability conditions: insertion-only connectivity differs from
deletions; binary search needs an ordered boundary or a monotone predicate. Use
only controlled algorithm IDs. Generic source tags do not establish particular
fine algorithms. Keep alternative valid approaches separate; do not force one
exclusive algorithm or family. An absent tag is not evidence of impossibility.

Each supported approach needs exact source evidence (statement/editorial kind,
version ID, locator, verbatim quote). Key observations must also cite evidence.
Use `candidate` for plausible but insufficiently evidenced approaches, and list
unresolved details under unknowns. Do not fabricate asymptotic bounds, constraints,
proofs, source quotes, alternative approaches, negative labels or review decisions.
Never include whole code listings as the abstract objective. No generated summary
or proposed label is written back to the authoritative corpus in this milestone.

## Draft query-context generation instruction

Preserve the original query exactly. Separate explicit user requirements from
inferred approaches. An inference needs a rationale and verbatim spans from the
original query; do not upgrade it into a hard requirement. Record explicit
exclusions and missing facts. In particular, do not infer insertion-only updates,
nonnegative weights, sortedness or monotonicity if these are unknown. Use the same
algorithm IDs as problem contexts. Return a draft, not an approved annotation.

## Review procedure

1. Inspect `context-pilot/sample.json` and fetch an individual evidence input using
   `python -m modeling context-input --problem-id codeforces:...`.
2. Generate/edit one draft outside this milestone, recording model/prompt provenance.
3. Run `python -m modeling validate-context --file draft.json`. The validator checks
   schema, vocabulary, training-only ownership and exact quote/version matches.
4. A human checks objective fidelity, conditions, evidence relevance, alternative
   approaches, unknowns and any unsupported fine labels. Schema validity alone
   does not establish semantic correctness.
5. Explicitly record a decision with `review-context --file draft.json --reviewer
   NAME --decision approved --output reviewed.json` (or `rejected`). Keep the draft.
   Review files remain under modeling artifacts; they do not automatically become
   gold labels or mutate MongoDB.

Illustrations in `examples/` are synthetic drafts only. Their synthetic evidence
is supplied separately, they do not identify real corpus annotations, and the
validator prevents promoting them to reviewed corpus annotations.
