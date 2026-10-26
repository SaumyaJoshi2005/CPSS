# Label specification: cpss-algorithms-v1

The executable specification is `config/taxonomy-v1.json`. The initial classifier
predicts normalized **observed original Codeforces tags**. It does not predict
exhaustive algorithm suitability or a human-verified solution method.

1. Original upstream tags remain untouched in the frozen corpus.
2. Normalization case-folds/normalizes whitespace and applies explicit aliases.
   For example, `union-find` maps to `dsu`; `binary-search` maps to `binary search`.
   The actual corpus has the canonical spellings; no observed spelling was changed
   merely to create a more attractive label distribution.
3. Broad family projections come from explicit configuration, never learned
   associations. DSU has both graph and data-structure parents. Trees, hashing
   and other techniques can also have multiple family parents. These projections
   are not separate human-approved annotations.
4. Future fine IDs such as `algorithm:dijkstra`, `algorithm:bellman_ford` and
   `pattern:binary_search_on_answer` are vocabulary entries with **unreviewed**
   status. A `shortest paths` or `binary search` tag does not assign those fine IDs.
   No problem received a new fine label during this milestone.

The baseline selects supported tags using **training data only**: at least 30
positive and 30 negative training assignments. This selects 29 labels. DSU has
147 positives and binary search 446 among the 3,716 training records. All 35
observed training tags are recognized by the specification; none are unsupported.
Rare labels remain in the audit and corpus but are excluded explicitly from the
classifier: ternary search (22), meet-in-the-middle (21), 2-sat (17), Chinese
remainder theorem (9), expression parsing (4), and schedules (2).

## DSU case study

**Definition.** Disjoint-set union maintains a partition of elements under merges
and representative/component queries. Typical clues include connecting groups,
merging equivalence classes, maintaining component counts or sizes, and answering
connectivity queries while adding edges. Ordinary union-find supports merges;
arbitrary online deletions need additional machinery and must not be assumed.

**Related techniques.** BFS/DFS can solve static component questions. Kruskal uses
DSU within a greedy graph algorithm. Offline ordering, reverse processing and
rollback DSU handle particular dynamic variants under extra conditions. Small-to-
large merging and “DSU on tree” terminology should not be silently equated with
ordinary union-find. Those fine distinctions require evidence/review.

**Common confusions.** A graph, tree or grid alone does not establish DSU. Paths,
distances, traversal order and reachability with direction can require other
methods. Conversely, DSU-tagged tasks can be expressed as permutation/range or
grouping problems without graph vocabulary. A missing tag remains unobserved,
not a proof that DSU cannot apply.

**Actual training positives (observed tags, not reviewed exclusive methods):**

- `codeforces:1156:E`, Special Segments of Permutation — observed DSU alongside
  divide and conquer, data structures and two pointers; useful non-graph wording.
- `codeforces:1365:D`, Solve The Maze — DSU co-occurs with traversal/graph tags;
  multiple supported implementations may be possible.
- `codeforces:1344:B`, Monopole Magnets — another grid/structure example with a
  DSU assignment and other graph/traversal tags.

**Candidate training contrasts, still unreviewed:** `codeforces:1000:G` Two-Paths
(tree/DP), `codeforces:917:D` Stranger Trees (trees/math/matrices), and
`codeforces:1741:G` Kirill and Company (graphs/shortest paths). These were selected
deterministically for related tags without an observed DSU tag. They are not
certified algorithm negatives. Their source versions/excerpts are in the audit.

## Binary-search case study

**Definition.** Binary search locates a boundary in an ordered domain using a
monotone predicate or searches ordered values. Structural clues include a
feasible/infeasible threshold, an ordered transition and a check procedure whose
answer remains monotone as a candidate bound changes. Sorted input is useful but
not necessary for binary search on a numerical answer.

**Related techniques.** Sorting/lower-bound queries, two pointers, prefix sums,
greedy feasibility checks and parametric optimization can accompany binary
search. Ternary search relies on an appropriate unimodal objective, a different
condition. The generic tag does not determine which of these patterns is used.

**Common confusions.** “Minimum,” “maximum,” a large bound or an array alone do not
prove a monotone predicate. Some optimization tasks admit direct formulas, greedy
or two-pointer solutions. Binary search may be hidden inside geometric/algebraic
or probabilistic tasks. Monotonicity and boundary conditions need evidence.

**Actual training positives (observed tags):**

- `codeforces:1418:E`, Expected Damage — binary search with combinatorics and
  probabilities; wording differs from an explicit search task.
- `codeforces:1129:C`, Morse Code — binary search with strings, hashing, DP and
  suffix-structure tags; illustrates multi-label overlap.
- `codeforces:1301:B`, Motarack's Birthday — binary search, greedy and ternary
  search assignments; a useful case for reviewing alternative approaches.

**Candidate training contrasts, still unreviewed:** `codeforces:1397:A` Juggling
Letters, `codeforces:1691:A` Beat The Odds, `codeforces:1552:A` Subsequence
Permutation and `codeforces:1400:D` Zigzags. They have related greedy/sorting/
two-pointer assignments without an observed binary-search tag. Their algorithm
suitability has not been exhaustively assessed.

Definitions and contrasts guide later evidence review; they do not create gold
annotations or replace original supervision. Validation errors are discussed
separately in `results.md` and did not alter this run's label support rule.
