# Three-minute CPSS demonstration

Start from the project root:

```powershell
.\.venv\Scripts\python.exe -m modeling index
.\.venv\Scripts\python.exe -m modeling serve
```

Open **http://127.0.0.1:8765**. If the already-running server occupies this port,
use that instance or launch with `--port 8766`.

1. **Named technique — 40 seconds.** Click **Name a technique**. Use Hybrid
   and Editorial-enriched. The actual unboosted top candidates include
   `codeforces:1217:F` (Forced Online Queries Problem), `codeforces:938:G`
   (Shortest Path Queries) and `codeforces:813:F` (Bipartite Checking).
   Show the original tags and the editorial excerpt discussing rebuilding a
   DSU for blocks of queries. Open a Codeforces source link if desired.
   Explain that a ranked candidate is not a human relevance judgment.
2. **Unnamed approach — 40 seconds.** Click **Describe the task**:
   “Maintain connected groups as new connections are added.” The parser leaves
   this as natural language without assigning an inferred DSU label. Actual
   editorial-enriched hybrid candidates begin with `codeforces:1023:F`
   (Mobile Phone Network), `codeforces:888:F` (Connecting Vertices), and
   `codeforces:1373:F` (Network Coverage). Inspect the stored evidence and
   acknowledge that suitability still needs review. Switch to Statement only
   or BM25 to show a different ranking on the same searchable collection.
3. **Combined techniques and filters — 45 seconds.** Click **Combine ideas**.
   Actual unfiltered editorial-enriched hybrid candidates begin with
   `codeforces:1370:D` (Odd-Even Subsequence, rating 2000), `codeforces:1680:C`
   (Binary String, 1600) and `codeforces:1760:F` (Quests, 1500). Show the first
   editorial's search-and-feasibility excerpt. Set ratings to 1200–2000 and
   search again. Highlight that filters are explicit constraints and that
   scores are not confidence percentages.
4. **Evaluation honesty — 35 seconds.** Open **Relevance review**. Show that
   methods/ranks/scores are hidden and that statements/editorials support
   grades 2/1/0. The real queue starts at **0/814 reviewed**. Do not submit an
   artificial human judgment for the demonstration. Explain that NDCG/MRR are
   unavailable until people actually review the pairs. Mention 104 passing
   tests and the untouched classifier test split.

The three queries are also example buttons in the app. The IDs above are
observed outputs under the frozen search-v1 configuration, not gold relevance
annotations or a guarantee that every result meets every query requirement.
Keep both optional boosts off to reproduce the fixed pool.

Optional CLI demonstration:

```powershell
.\.venv\Scripts\python.exe -m modeling search --query "DSU with connectivity queries." --method hybrid --view editorial --limit 3
.\.venv\Scripts\python.exe -m modeling search --query "Maintain connected groups as new connections are added." --method dense --view statement --limit 3
.\.venv\Scripts\python.exe -m modeling search --query "Binary search on the answer with a greedy feasibility check." --method hybrid --view editorial --rating-min 1200 --rating-max 2000 --limit 3
```

Suggested screenshots for a manual submission capture: the initial page showing
3,716/4,648 coverage; a populated hybrid search with evidence and filters; and
the blinded review page showing zero reviewed pairs. No screenshots are supplied
or claimed to be visually verified: automated browser access was stopped by the
tool's URL-confidence safeguard. The HTTP and client-script checks passed.
