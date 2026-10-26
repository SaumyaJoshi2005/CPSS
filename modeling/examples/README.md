# Synthetic illustrations only

The two problem contexts and one query context are handcrafted illustrations,
not generated corpus annotations or model predictions. All are unreviewed drafts.
`synthetic-sources.json` contains their invented, explicitly synthetic evidence;
its hashes identify those strings, not Codeforces documents. The validator rejects
promoting an illustration to a reviewed corpus annotation.

Reproduce with `python -m modeling.examples.make_illustrations` from the repository
root. Validate each with `python -m modeling validate-context --file PATH
--illustrative-sources modeling/examples/synthetic-sources.json`.
