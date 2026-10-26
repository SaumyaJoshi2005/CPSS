from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse

from ..common import Config, ModelingError, atomic_json, dependency_versions, hash_file, identity, read_json
from ..data import Corpus
from ..encoder import ENCODER_VERSION, FrozenEncoder, runtime_versions, token_chunks
from ..resources import memory_observation
from .configuration import SearchConfig
from .contexts import CONTEXT_VERSION, Document, View, load_documents
from .lexical import BM25Index, TOKENIZER_VERSION

INDEX_VERSION = "cpss-bm25-dense-rrf-index-v1"


def local_encoder(settings: SearchConfig, corpus_settings: Config) -> tuple[FrozenEncoder, dict]:
    spec = read_json(settings.path("encoder_plan"))["encoder"]
    directory = corpus_settings.path("cache_path") / "encoders" / identity(spec)[:16]
    receipt = directory / "assets.json"
    if not receipt.exists():
        raise ModelingError("Existing MiniLM assets unavailable; build with --lexical-only for BM25. No new model is downloaded.")
    assets = read_json(receipt)
    if (assets["spec"] != spec or assets["encoder_version"] != ENCODER_VERSION
            or assets["dependencies"] != runtime_versions()
            or any(not (directory / n).is_file() or hash_file(directory / n) != h for n, h in assets["files"].items())):
        raise ModelingError("Existing frozen encoder assets failed configuration/hash/version checks")
    return FrozenEncoder(str(directory.resolve()), spec), {"directory": str(directory.resolve()), **assets}


def index_key(documents: list[Document], view: View, settings: SearchConfig, manifest: str, assets: dict | None) -> str:
    return identity({"index_version": INDEX_VERSION, "context_version": CONTEXT_VERSION, "tokenizer_version": TOKENIZER_VERSION,
        "source_manifest_sha256": manifest, "configuration": settings.values, "versions": dependency_versions(),
        "view": view, "rows": [d.signature(view) for d in documents], "encoder": assets})


def existing_statement_vectors(documents: list[Document], corpus: Corpus, config: Config, assets: dict):
    """Select training identities only from the existing verified encoder cache."""
    for receipt in sorted((config.path("cache_path") / "embeddings").glob("*/metadata.json")):
        metadata = read_json(receipt)
        if metadata.get("assets") != assets:
            continue
        rows = metadata["rows"]
        if len(set(rows)) != len(rows) or not all(p in corpus.structural for p in rows):
            continue
        expected = identity({"assets": assets, "version": ENCODER_VERSION,
                             "rows": [(p, corpus.structural[p]["clean_sha256"]) for p in rows]})
        positions = {p: i for i, p in enumerate(rows)}
        file = receipt.parent / "embeddings.npy"
        if metadata["key"] != expected or not all(d.problem_id in positions for d in documents):
            continue
        if hash_file(file) != metadata["sha256"]:
            raise ModelingError("Existing statement embedding cache failed hash verification")
        matrix = np.load(file, mmap_mode="r")
        return np.asarray(matrix[[positions[d.problem_id] for d in documents]], dtype=np.float32), str(receipt)
    return None, None


@dataclass
class SearchIndex:
    view: View
    key: str
    lexical: BM25Index
    dense: np.ndarray | None
    metadata: dict


def load_index(directory: Path, expected_key: str) -> SearchIndex:
    manifest = read_json(directory / "index.json")
    if manifest["key"] != expected_key or manifest["version"] != INDEX_VERSION:
        raise ModelingError("Search index configuration/source identity changed; rebuild the indexes")
    if manifest["dependency_versions"] != dependency_versions():
        raise ModelingError("Search index dependency versions changed; use pinned dependencies")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or hash_file(directory / name) != digest:
            raise ModelingError("Search index file integrity failure")
    vocabulary = read_json(directory / "vocabulary.json")
    weights = sparse.load_npz(directory / "bm25.npz").tocsr()
    dense = np.load(directory / "dense.npy", mmap_mode="r") if manifest["dense_available"] else None
    rows = len(manifest["problem_ids"])
    if weights.shape != (rows, len(vocabulary)) or (dense is not None and dense.shape != (rows, 384)):
        raise ModelingError("Search index matrix/identity dimensions disagree")
    return SearchIndex(manifest["view"], expected_key, BM25Index(vocabulary, weights, manifest["bm25"]), dense, manifest)


def build_view(documents: list[Document], corpus: Corpus, settings: SearchConfig, config: Config,
               view: View, include_dense: bool, progress: Callable[[dict], None] | None = None) -> dict:
    if len({d.problem_id for d in documents}) != len(documents) or any(d.provenance["split"] != "train" for d in documents):
        raise ModelingError("Search indexing requires unique canonical training identities")
    encoder, assets = local_encoder(settings, config) if include_dense else (None, None)
    key = index_key(documents, view, settings, corpus.manifest_sha256, assets)
    directory = settings.path("index_path") / "indexes" / key[:20]
    if directory.exists():
        index = load_index(directory, key)
        return {"key": key, "directory": str(directory.resolve()), "reused": True, "metadata": index.metadata}
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + ".tmp-" + uuid.uuid4().hex[:6])
    temporary.mkdir()
    started, before = time.perf_counter(), memory_observation()
    texts = [d.text(view) for d in documents]
    lexical = BM25Index.build(texts, settings.values["bm25"])
    sparse.save_npz(temporary / "bm25.npz", lexical.weights)
    atomic_json(temporary / "vocabulary.json", lexical.vocabulary)
    lexical_seconds = time.perf_counter() - started
    lengths_report, reuse_source, dense_seconds = None, None, 0.0
    if encoder is not None:
        lengths = [len(encoder.token_ids(t)) for t in texts]
        chunks = []
        for document, length in zip(documents, lengths):
            try:
                chunks.append(len(token_chunks(list(range(length)), encoder.spec["maximum_tokens"], encoder.spec["overlap_tokens"], encoder.spec["maximum_chunks"])))
            except ModelingError as exc:
                raise ModelingError(f"{document.problem_id} exceeds the declared {view} chunk bound; no text was truncated") from exc
        lengths_report = {"records": len(texts), "split": "train", "tokenizer_revision": encoder.spec["revision"],
            "content_token_percentiles": {str(p): float(np.percentile(lengths, p)) for p in [0, 50, 95, 99, 100]},
            "more_than_one_chunk": sum(n > encoder.spec["maximum_tokens"] - 2 for n in lengths),
            "maximum_chunks": max(chunks), "mean_chunks": float(np.mean(chunks)),
            "policy": "All content tokens retained; overlapping 254-content-token chunks, 32 overlap, newly covered-token weighted mean then L2; one vector per canonical problem"}
        atomic_json(temporary / "tokenizer-audit.json", lengths_report)  # Before encoding.
        if progress:
            progress({"view": view, "tokenizer_audit": lengths_report})
        tick = time.perf_counter()
        reused = None
        if view == "statement":
            reused, reuse_source = existing_statement_vectors(documents, corpus, config, assets)
        if reused is not None:
            np.save(temporary / "dense.npy", reused)
        else:
            matrix = np.lib.format.open_memmap(temporary / "dense.npy", mode="w+", dtype=np.float32, shape=(len(documents), 384))
            for first in range(0, len(documents), 16):
                matrix[first:first + 16] = encoder.transform(texts[first:first + 16])
                matrix.flush()
                if progress and first % 128 == 0:
                    progress({"view": view, "embedded": min(first + 16, len(documents)), "total": len(documents)})
            del matrix
        dense_seconds = time.perf_counter() - tick
        matrix = np.load(temporary / "dense.npy", mmap_mode="r")
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ModelingError("Dense problem vectors are not finite unit vectors")
        del matrix
    files = {p.name: hash_file(p) for p in temporary.iterdir() if p.is_file()}
    metadata = {"version": INDEX_VERSION, "key": key, "source_manifest_sha256": corpus.manifest_sha256,
        "configuration_sha256": settings.fingerprint, "configuration": settings.values, "view": view, "split": "train",
        "problem_ids": [d.problem_id for d in documents], "document_signatures": [d.signature(view) for d in documents],
        "searchable_count": len(documents), "complete_corpus_count": sum(corpus.split_counts.values()),
        "dense_available": include_dense, "encoder": assets, "tokenizer_audit": lengths_report,
        "statement_vector_reuse": reuse_source, "bm25": lexical.configuration, "dependency_versions": dependency_versions(),
        "timings": {"bm25_build_seconds": lexical_seconds, "dense_build_or_reuse_seconds": dense_seconds, "total_seconds": time.perf_counter() - started},
        "memory": {"before": before, "after": memory_observation(), "peak_semantics": "process-lifetime peak, not isolated index peak"},
        "files": files, "test_used": False, "corpus_text_copied": False}
    atomic_json(temporary / "index.json", metadata)
    temporary.rename(directory)
    return {"key": key, "directory": str(directory.resolve()), "reused": False, "metadata": metadata}


def build_indexes(settings: SearchConfig, config: Config, corpus: Corpus, include_dense=True, progress=None) -> dict:
    documents = load_documents(corpus)
    active = {"source_manifest_sha256": corpus.manifest_sha256, "configuration_sha256": settings.fingerprint, "views": {}}
    for view in settings.values["views"]:
        result = build_view(documents, corpus, settings, config, view, include_dense, progress)
        active["views"][view] = {"key": result["key"], "directory": str(Path("indexes") / result["key"][:20]),
                                  "searchable_count": len(documents), "dense_available": include_dense}
        # An interrupted editorial build still leaves the completed statement
        # view usable. Each index promotion and pointer write is atomic.
        atomic_json(settings.path("index_path") / "active.json", active)
        if progress:
            progress({"indexed_view": view, "reused": result["reused"], "count": len(documents), "timings": result["metadata"]["timings"]})
    return active
