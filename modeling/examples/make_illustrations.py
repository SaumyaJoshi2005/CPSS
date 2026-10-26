"""Write three handcrafted synthetic drafts; never annotate corpus problems."""
from pathlib import Path

from cpss.core import sha256
from modeling.common import atomic_json
from modeling.contexts import pending_problem, validate_context
from modeling.labels import Taxonomy


def build():
    root = Path(__file__).parent
    taxonomy = Taxonomy.load(root.parent / "config" / "taxonomy-v1.json")
    sources = {}
    definitions = [
        ("illustrative:connectivity", "dsu", "Maintain components under insertion of undirected edges.",
         "There are n fixed vertices. Add undirected edges and answer whether two vertices are connected. There are no deletions.",
         "Keep a representative for each component and merge representatives when an edge is inserted.",
         ["fixed vertices", "undirected graph", "partition"], ["insert edge", "query component membership"],
         ["Undirected connectivity", "Insertions only; arbitrary deletions are outside ordinary DSU"]),
        ("illustrative:ordered-boundary", "binary search", "Find the first array position meeting an ordered threshold.",
         "Given a nondecreasing array a and value x, find the first index i with a[i] >= x, or report none.",
         "The predicate a[i] >= x is false before a boundary and true from that boundary onward.",
         ["ordered array"], ["compare candidate value to x", "narrow an index interval"],
         ["Array is nondecreasing", "The predicate is monotone in the index"]),
    ]
    for pid, tag, objective, statement, editorial, structures, operations, conditions in definitions:
        sources[pid] = {"statement": {"version_id": sha256(statement.encode()), "text": statement},
                        "editorial": {"version_id": sha256(editorial.encode()), "text": editorial}}
        document = pending_problem(pid, taxonomy, "0" * 64)
        document.update(status="draft", abstract_objective=objective, structures=structures, operations=operations,
                        applicability_conditions=conditions, unknowns=["Input-size bounds are unspecified in this illustration."])
        document["provenance"].update(kind="illustrative", source_manifest_sha256=None, generation_method="illustrative", prompt_version="illustrative-v1")
        evidence = {"source_kind": "editorial", "version_id": sources[pid]["editorial"]["version_id"], "locator": "whole_text", "quote": editorial}
        document["supported_approaches"] = [{"algorithm_id": taxonomy.spec["tags"][tag]["algorithm_id"],
            "status": "candidate", "evidence": [evidence], "applicability_conditions": conditions}]
        document["key_observations"] = [{"text": conditions[0], "evidence": [evidence]}]
        validate_context(document, taxonomy, illustrative_sources=sources)
        atomic_json(root / ("problem-dsu.json" if tag == "dsu" else "problem-binary-search.json"), document)
    query = {"schema_version": "query-context-v1", "taxonomy_version": taxonomy.spec["version"], "status": "draft",
        "original_text": "Find problems where roads are added over time and I must answer whether two towns are connected. I do not want edge removals.",
        "explicit_requirements": ["Road additions over time", "Connectivity queries", "Exclude edge removals"],
        "inferred_approaches": [{"algorithm_id": "technique:dsu", "certainty": "inferred",
            "rationale": "DSU is a candidate if the roads represent undirected connectivity; direction has not been specified.",
            "evidence_spans": ["roads are added over time", "whether two towns are connected"]}],
        "exclusions": ["edge removals"], "unknowns": ["Are roads directed or undirected?", "What are the graph and query sizes?"],
        "provenance": {"kind": "illustrative", "source_manifest_sha256": None, "generation_method": "illustrative", "model_revision": None, "prompt_version": "illustrative-v1"},
        "review": {"decision": "pending", "reviewer": None, "reviewed_at": None}}
    validate_context(query, taxonomy)
    atomic_json(root / "query-connectivity.json", query)
    atomic_json(root / "synthetic-sources.json", sources)


if __name__ == "__main__":
    build()
