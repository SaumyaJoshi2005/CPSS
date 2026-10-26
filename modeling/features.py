from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import ModelingError, atomic_json, dependency_versions, hash_file, identity, read_json
from .data import Record

FEATURE_VERSION = "statement-tfidf-v1"


def texts(records: list[Record]) -> list[str]:
    """The sole classifier feature boundary: no dictionaries or metadata accepted."""
    if not records or any(not isinstance(r.statement, str) or not r.statement.strip() for r in records):
        raise ModelingError("Feature construction needs nonempty statement text")
    return [r.statement for r in records]


def feature_key(records: list[Record], config: dict[str, Any]) -> str:
    from cpss.core import sha256
    return identity({"feature_version": FEATURE_VERSION, "configuration": config, "versions": dependency_versions(),
                     "training_rows": [(r.problem_id, sha256(r.statement.encode("utf-8"))) for r in records]})


def fit_features(records: list[Record], config: dict[str, Any], cache: Path):
    if any(r.split != "train" for r in records):
        raise ModelingError("TF-IDF may only be fitted on training statements")
    key = feature_key(records, config)
    directory = cache / "features" / key[:24]
    receipt = directory / "metadata.json"
    start = time.perf_counter()
    if receipt.exists():
        metadata = read_json(receipt)
        if (metadata.get("key") == key and all((directory / name).exists()
                and hash_file(directory / name) == digest for name, digest in metadata["files"].items())):
            vectorizer = joblib.load(directory / "vectorizer.joblib")
            matrix = sparse.load_npz(directory / "train.npz").tocsr()
            return vectorizer, matrix, {"cache_key": key, "cache_hit": True, "seconds": time.perf_counter() - start}
        raise ModelingError("Feature cache failed integrity verification; remove that cache entry and retry")
    kwargs = {**config, "ngram_range": tuple(config["ngram_range"]), "dtype": np.float64}
    vectorizer = TfidfVectorizer(**kwargs)
    try:
        matrix = vectorizer.fit_transform(texts(records)).tocsr()
    except ValueError as exc:
        raise ModelingError("TF-IDF configuration produced an empty or invalid vocabulary") from exc
    if not sparse.isspmatrix_csr(matrix):
        raise ModelingError("Training features must remain CSR sparse")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(key[:24] + ".building-" + uuid.uuid4().hex[:8])
    staging.mkdir()
    joblib.dump(vectorizer, staging / "vectorizer.joblib", compress=3)
    sparse.save_npz(staging / "train.npz", matrix)
    atomic_json(staging / "metadata.json", {"key": key, "fit_split": "train", "rows": len(records),
        "training_ids": [r.problem_id for r in records], "shape": list(matrix.shape), "nonzero": matrix.nnz,
        "files": {name: hash_file(staging / name) for name in ("vectorizer.joblib", "train.npz")}})
    staging.rename(directory)
    return vectorizer, matrix, {"cache_key": key, "cache_hit": False, "seconds": time.perf_counter() - start}
