# Fixed experiment plan v2

This plan was written before any new representation was fitted or its validation
metrics computed. `config/experiment-v2.json` is the executable plan; a hashed,
timestamped copy is frozen under the comparison artifacts before training.
Baseline-v1 and the corpus/splits remain unchanged. Its validation-selected
configuration is a historical reference, not a protocol-matched control.

Compare A: word TF-IDF; B: word plus character-within-word TF-IDF (3–5 grams,
weight 0.5); and C: frozen MiniLM embeddings if the model/runtime are available.
Use exactly the historical 29 labels. Keep the existing word preprocessing and
40k feature cap; add a separate 40k character cap. Inputs are statement text only.
Character features test lexical/notation variation, not algorithm understanding.

Three shuffled GroupKFold folds use seed 42. Groups are transitive closures of
contests, aliases, exact clean hashes, explicit duplicate groups and explicit
same-date variants. Every lexical vectorizer is fitted separately on its fold's
training partition. Class weights use that partition only. A frozen encoder is
a deterministic per-document transformation and can be cached across folds;
there is no fitted corpus vocabulary or supervised encoder update.

For all representations, compare C={1,3} and class_weight={None,balanced} using
pooled training out-of-fold macro F1 at threshold 0.5. Break ties by micro F1,
lower C, then no class weighting. After selection, obtain a global threshold
from pooled OOF micro F1. Tune a label threshold only with >=30 training positives,
>=30 negatives and >=2 positives in every held-out fold; otherwise use the global
threshold. No validation target is used for candidate or threshold selection.
OOF model selection is not nested CV; untouched validation is the development
comparison, and the real final test remains reserved.

MiniLM is pinned to the exact revision/quantized ONNX file in the plan. Inspect
current host/GPU memory. Use CPU ONNX inference, batch 8 and two threads. Measure
actual pretrained-tokenizer lengths on training statements before embedding.
The fixed long-input rule retains every content token in 256-token overlapping
chunks, attention-mask mean pools token representations, averages chunks by new
content-token coverage, then L2 normalizes. Inputs exceeding 64 chunks fail instead
of silently truncating. Do not update encoder weights. Record actual length,
chunk, timing and memory observations. A download/resource failure is reported,
and must not prevent A/B.

Review 48 deterministic label/case pairs: six each of FP/FN/TP/TN for DSU and
binary search. Inspect their validation statements, corresponding editorials and
all supplied accepted sources. Save exact evidence references/quotes separately
from original labels. Rules can propose model-error corroboration, an alternative
approach or incomplete-tag candidates; ambiguous and correct-prediction semantic
reviews can remain unresolved. Every such review is automated, not human-approved.
The source review does not alter features, labels, thresholds or this plan.

An alternative merits development adoption only if neither aggregate F1 regresses
against the new word control, at least one improves by 0.005, and neither case
label loses more than 0.02 F1. This is a descriptive rule, not a significance
claim. Do not automatically replace defaults or touch baseline-v1. Report all
regressions and results regardless of this rule. Validation must not trigger a
second search or a revised plan.
