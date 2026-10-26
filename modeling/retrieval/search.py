from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict
from typing import Any, Protocol

import numpy as np
from threadpoolctl import threadpool_limits

from ..common import Config, ModelingError, read_json
from ..data import Corpus
from ..inference import load_bundle, predict
from ..labels import Taxonomy
from .configuration import SearchConfig
from .contexts import Document, QueryContext, load_documents, parse_query
from .indexing import SearchIndex, index_key, load_index, local_encoder
from .lexical import tokens
from .ranking import boosted_score, ranked_ids, reciprocal_rank_fusion


class CorpusIdentity(Protocol):
    manifest_sha256: str
    split_counts: dict[str, int]


def evidence(document: Document, view: str, query: QueryContext, bm25_configuration: dict) -> dict:
    query_terms = set(tokens(query.search_text, bm25_configuration))
    choices = [("statement", document.statement, document.provenance["statement_version_id"])]
    if view == "editorial":
        choices += [("editorial", document.editorial, document.provenance["editorial_version_id"])]
    spans = []
    for kind, text, version in choices:
        # Score bounded original sentence/line windows, keeping exact offsets.
        for match in re.finditer(r"[^\n.!?]+[.!?]?", text):
            start, end = match.span()
            if end - start < 20:
                continue
            for offset in range(start, end, 280):
                stop = min(offset + 320, end)
                quote = text[offset:stop]
                overlap = query_terms & set(tokens(quote, bm25_configuration))
                spans.append((len(overlap), kind, offset, stop, quote, version, sorted(overlap)))
    best = max(spans, key=lambda v: (v[0], v[1] == "editorial", -v[2]), default=None)
    tag_matches = sorted(set(a["original_tag_proxy"] for a in query.explicit_algorithms) & set(document.original_tags))
    if best is None:
        quote, kind, start, end, version, matched = document.statement[:280], "statement", 0, min(280, len(document.statement)), document.provenance["statement_version_id"], []
    else:
        _, kind, start, end, quote, version, matched = best
    return {"source_kind": kind, "quote": quote, "start": start, "end": end, "version_id": version,
            "matched_terms": matched, "original_tag_overlap": tag_matches,
            "semantics": "Exact source excerpt and lexical/tag overlap; not generated reasoning or proof of suitability"}


class SearchEngine:
    def __init__(self, settings: SearchConfig, corpus_settings: Config, corpus: CorpusIdentity,
                 documents: list[Document], indexes: dict[str, SearchIndex], encoder=None):
        self.settings, self.corpus_settings, self.corpus = settings, corpus_settings, corpus
        self.documents, self.indexes, self.encoder = documents, indexes, encoder
        self.by_id = {d.problem_id: d for d in documents}
        self.ids = [d.problem_id for d in documents]
        self.positions = {pid: i for i, pid in enumerate(self.ids)}
        self.taxonomy = Taxonomy.load(corpus_settings.path("taxonomy_path"))
        self.classifier = None
        self._lock = threading.RLock()
        self.load_seconds = 0.0

    @classmethod
    def load(cls, settings: SearchConfig, corpus_settings: Config) -> "SearchEngine":
        started = time.perf_counter()
        corpus = Corpus.verified(corpus_settings)
        documents = load_documents(corpus)
        active_path = settings.path("index_path") / "active.json"
        if not active_path.exists():
            raise ModelingError("Search indexes are missing. Run python -m modeling index first.")
        active = read_json(active_path)
        if active["source_manifest_sha256"] != corpus.manifest_sha256 or active["configuration_sha256"] != settings.fingerprint:
            raise ModelingError("Search source/configuration changed; rebuild the search indexes")
        encoder, assets = (local_encoder(settings, corpus_settings) if any(v["dense_available"] for v in active["views"].values()) else (None, None))
        indexes = {}
        for view, receipt in active["views"].items():
            key = index_key(documents, view, settings, corpus.manifest_sha256, assets if receipt["dense_available"] else None)
            if receipt["key"] != key:
                raise ModelingError("Search document content/configuration changed; rebuild indexes")
            directory = (settings.path("index_path") / receipt["directory"]).resolve()
            if not directory.is_relative_to(settings.path("index_path").resolve()):
                raise ModelingError("Invalid search index location")
            indexes[view] = load_index(directory, key)
            if indexes[view].metadata["problem_ids"] != [d.problem_id for d in documents]:
                raise ModelingError("Search document/vector ordering mismatch")
        engine = cls(settings, corpus_settings, corpus, documents, indexes, encoder)
        engine.load_seconds = time.perf_counter() - started
        return engine

    def info(self) -> dict:
        return {"searchable_count": len(self.documents), "complete_corpus_count": sum(self.corpus.split_counts.values()),
                "scope": "Preserved training split", "views": {v: {"dense_available": i.dense is not None} for v, i in self.indexes.items()},
                "source_manifest_sha256": self.corpus.manifest_sha256, "configuration_sha256": self.settings.fingerprint,
                "index_loading_seconds": self.load_seconds, "original_tags_are_not_exhaustive_suitability": True}

    def search(self, query: str, method="hybrid", view="editorial", limit=None, candidates=None,
               rating_min=None, rating_max=None, tag_boost=False, classifier_boost=False) -> dict[str, Any]:
        with self._lock, threadpool_limits(limits=1):
            return self._search(query, method, view, limit, candidates, rating_min, rating_max, tag_boost, classifier_boost)

    def _search(self, query, method, view, limit, candidates, rating_min, rating_max, tag_boost, classifier_boost):
        started = time.perf_counter()
        if method not in {"bm25", "dense", "hybrid"} or view not in self.indexes:
            raise ModelingError("Choose an available retrieval method and context view")
        limit = self.settings.values["result_limit"] if limit is None else limit
        candidates = self.settings.values["candidate_limit"] if candidates is None else candidates
        if type(limit) is not int or type(candidates) is not int or not 1 <= limit <= candidates <= 500:
            raise ModelingError("Result limit must be between 1 and candidate limit (maximum 500)")
        if type(tag_boost) is not bool or type(classifier_boost) is not bool:
            raise ModelingError("Boost options must be boolean")
        context = parse_query(query, self.settings, self.taxonomy, rating_min, rating_max)
        if not context.search_text.strip():
            raise ModelingError("Include a task or approach as well as filters")
        index = self.indexes[view]
        if method != "bm25" and (index.dense is None or self.encoder is None):
            raise ModelingError("Dense retrieval is unavailable in this index; choose BM25 or build dense indexes")
        eligible = np.asarray([(context.rating_min is None or d.rating >= context.rating_min)
                               and (context.rating_max is None or d.rating <= context.rating_max)
                               and not set(context.excluded_tags) & set(d.original_tags) for d in self.documents])
        lexical_scores = np.zeros(len(self.ids))
        dense_scores = np.zeros(len(self.ids))
        lexical_ranking, dense_ranking = [], []
        if method in {"bm25", "hybrid"}:
            lexical_scores = index.lexical.scores(context.search_text)
            lexical_ranking = ranked_ids(lexical_scores, self.ids, eligible, candidates, positive_only=True)
        if method in {"dense", "hybrid"} and eligible.any():
            vector = self.encoder.transform([context.search_text])[0]
            dense_scores = np.asarray(index.dense @ vector).ravel()
            dense_ranking = ranked_ids(dense_scores, self.ids, eligible, candidates)
        if method == "hybrid":
            base = reciprocal_rank_fusion([lexical_ranking, dense_ranking], self.settings.values["rrf_k"])
            semantics = "Reciprocal rank fusion score; not a probability or calibrated confidence"
        else:
            order = lexical_ranking if method == "bm25" else dense_ranking
            values = lexical_scores if method == "bm25" else dense_scores
            base = {pid: float(values[self.positions[pid]]) for pid in order}
            semantics = "BM25 lexical relevance score" if method == "bm25" else "Cosine similarity to a frozen query embedding; not calibrated confidence"
        classifier_prediction, model_scores = None, {}
        if classifier_boost and base:
            if self.classifier is None:
                self.classifier = load_bundle(self.settings.path("classifier_run"))
            classifier_prediction = predict(self.classifier, context.original_text)
            model_scores = classifier_prediction["label_scores"]
        explicit_tags = {a["original_tag_proxy"] for a in context.explicit_algorithms}
        ranked = []
        for pid, score in base.items():
            document = self.by_id[pid]
            matches = explicit_tags & set(document.original_tags)
            model_signal = max((model_scores.get(t, 0) for t in document.original_tags), default=0)
            boosted = boosted_score(score, len(matches) / max(1, len(explicit_tags)), model_signal,
                self.settings.values["explicit_tag_boost"] if tag_boost else 0,
                self.settings.values["classifier_boost"] if classifier_boost else 0)
            ranked.append((pid, boosted, score, model_signal))
        ranked.sort(key=lambda r: (-r[1], r[0]))
        results = []
        for pid, final_score, base_score, model_signal in ranked[:limit]:
            d, position = self.by_id[pid], self.positions[pid]
            results.append({"rank": len(results) + 1, "problem_id": pid, "title": d.title, "url": d.url, "rating": d.rating,
                "original_tags": list(d.original_tags), "score": final_score, "base_retrieval_score": base_score,
                "bm25_score": float(lexical_scores[position]) if method != "dense" else None,
                "dense_cosine": float(dense_scores[position]) if method != "bm25" else None,
                "explicit_tag_boost_applied": tag_boost, "classifier_boost_applied": classifier_boost,
                "classifier_signal": model_signal if classifier_boost else None,
                "evidence": evidence(d, view, context, self.settings.values["bm25"]), "provenance": d.provenance})
        return {"query_context": asdict(context), "method": method, "view": view, "results": results,
            "searchable_count": len(self.documents), "complete_corpus_count": sum(self.corpus.split_counts.values()),
            "eligible_count": int(eligible.sum()), "candidate_count": len(base), "candidate_limit": candidates,
            "query_seconds": time.perf_counter() - started, "score_semantics": semantics,
            "classifier_prediction": classifier_prediction, "index_key": index.key,
            "source_manifest_sha256": self.corpus.manifest_sha256, "test_used": False}
