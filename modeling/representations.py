from __future__ import annotations

import uuid
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .common import ModelingError, atomic_json, dependency_versions, hash_file, identity, read_json
from .data import Record
from .features import texts

REPRESENTATION_VERSION = "word-char-fold-local-v2"


def lexical_transformer(kind: str, plan: dict):
    def vectorizer(configuration):
        return TfidfVectorizer(**{**configuration, "ngram_range": tuple(configuration["ngram_range"]), "dtype": np.float64})
    if kind == "word":
        return vectorizer(plan["word"])
    if kind == "word_char":
        return FeatureUnion([("word", vectorizer(plan["word"])), ("character", vectorizer(plan["character"]))],
                            transformer_weights={"word": 1.0, "character": plan["character_weight"]}, n_jobs=1)
    raise ModelingError("Unknown lexical representation")


def fold_features(training: list[Record], held: list[Record], kind: str, plan: dict, cache: Path):
    if not training or not held or any(r.split != "train" for r in [*training, *held]):
        raise ModelingError("CV preprocessing accepts original training records only")
    if {r.problem_id for r in training} & {r.problem_id for r in held}:
        raise ModelingError("Fold preprocessing partitions overlap")
    return cached_lexical(training, held, kind, plan, cache)


def cached_lexical(training: list[Record], other: list[Record], kind: str, plan: dict, cache: Path):
    if any(r.split != "train" for r in training):
        raise ModelingError("Lexical vocabulary may be fitted only on training records")
    key = identity({"version": REPRESENTATION_VERSION, "kind": kind, "word": plan["word"], "character": plan["character"],
        "character_weight": plan["character_weight"], "versions": dependency_versions(),
        "fit": [(r.problem_id, r.clean_sha256) for r in training], "transform": [(r.problem_id, r.clean_sha256) for r in other]})
    directory = cache / "cv-features" / key[:20]
    receipt = directory / "metadata.json"
    if receipt.exists():
        meta = read_json(receipt)
        if meta["key"] != key or any(hash_file(directory / n) != h for n, h in meta["files"].items()):
            raise ModelingError("CV feature cache integrity failure")
        return joblib.load(directory / "transformer.joblib"), sparse.load_npz(directory / "fit.npz").tocsr(), sparse.load_npz(directory / "other.npz").tocsr(), key
    transformer = lexical_transformer(kind, plan)
    x_fit = transformer.fit_transform(texts(training)).tocsr()
    x_other = transformer.transform(texts(other)).tocsr()
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + ".tmp-" + uuid.uuid4().hex[:6])
    temporary.mkdir()
    joblib.dump(transformer, temporary / "transformer.joblib", compress=3)
    sparse.save_npz(temporary / "fit.npz", x_fit)
    sparse.save_npz(temporary / "other.npz", x_other)
    atomic_json(temporary / "metadata.json", {"key": key, "fit_ids": [r.problem_id for r in training],
        "transformed_ids": [r.problem_id for r in other], "fit_split": "train", "matrix_format": "csr",
        "files": {name: hash_file(temporary / name) for name in ["transformer.joblib", "fit.npz", "other.npz"]}})
    temporary.rename(directory)
    return transformer, x_fit, x_other, key
