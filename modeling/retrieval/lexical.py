from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..common import ModelingError

TOKENIZER_VERSION = "unicode-word-bm25-v1"


def tokens(text: str, configuration: dict) -> list[str]:
    stopwords = set(configuration["stopwords"])
    return [w for w in re.findall(r"[^\W_]+", text.casefold())
            if len(w) >= configuration["minimum_token_length"] and w not in stopwords]


@dataclass
class BM25Index:
    vocabulary: dict[str, int]
    weights: sparse.csr_matrix
    configuration: dict

    @classmethod
    def build(cls, texts: list[str], configuration: dict) -> "BM25Index":
        if not texts:
            raise ModelingError("Cannot index an empty problem collection")
        counts = [Counter(tokens(t, configuration)) for t in texts]
        vocabulary = {term: i for i, term in enumerate(sorted({t for c in counts for t in c}))}
        if not vocabulary:
            raise ModelingError("No indexable lexical terms")
        rows, columns, values = [], [], []
        lengths = np.asarray([sum(c.values()) for c in counts], dtype=np.float32)
        for row, counter in enumerate(counts):
            for term, count in counter.items():
                rows.append(row); columns.append(vocabulary[term]); values.append(count)
        tf = sparse.csr_matrix((np.asarray(values, np.float32), (rows, columns)), shape=(len(texts), len(vocabulary)))
        df = np.asarray(tf.getnnz(axis=0)).ravel()
        idf = np.log1p((len(texts) - df + .5) / (df + .5))
        k1, b = configuration["k1"], configuration["b"]
        denominator = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1))
        weight = tf.copy()
        weight.data = (tf.data * (k1 + 1) / (tf.data + np.repeat(denominator, np.diff(tf.indptr))) * idf[tf.indices]).astype(np.float32)
        return cls(vocabulary, weight, configuration)

    def scores(self, query: str) -> np.ndarray:
        counts = Counter(tokens(query, self.configuration))
        selected = [(self.vocabulary[t], count) for t, count in counts.items() if t in self.vocabulary]
        if not selected:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        ids, multiplicities = zip(*selected)
        return np.asarray(self.weights[:, list(ids)] @ np.asarray(multiplicities, dtype=np.float32)).ravel()
