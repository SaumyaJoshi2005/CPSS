"""Frozen, pinned CPU ONNX MiniLM inference; no training framework or weight updates."""
from __future__ import annotations

import json
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .common import ModelingError, atomic_json, hash_file, identity, read_json
from .data import Record
from .features import texts
from .resources import memory_observation

ENCODER_VERSION = "frozen-minilm-overlap-mean-v1"


def runtime_versions() -> dict[str, str]:
    try:
        return {name: version(name) for name in ["onnxruntime", "tokenizers"]}
    except PackageNotFoundError as exc:
        raise ModelingError("Frozen encoder dependencies are unavailable; install modeling/requirements-encoder.lock") from exc


def prepare_encoder(spec: dict, cache: Path, progress=None) -> dict:
    dependencies = runtime_versions()
    directory = cache / "encoders" / identity(spec)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    files = [spec["file"], "tokenizer.json", "tokenizer_config.json", "config.json", "sentence_bert_config.json", "1_Pooling/config.json"]
    hashes = {}
    for name in files:
        local = directory / name
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{name}"
            temporary = local.with_suffix(local.suffix + ".tmp")
            length = 0
            try:
                with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        length += len(block)
                        if length > 150 * 1024 * 1024:
                            raise ModelingError("Encoder asset exceeds the bounded download size")
                        output.write(block)
                temporary.replace(local)
            except OSError as exc:
                raise ModelingError(f"Encoder download failed for {name}: {type(exc).__name__}") from exc
            if progress:
                progress({"downloaded_encoder_asset": name, "bytes": length})
        hashes[name] = hash_file(local)
    metadata = read_json(directory / "sentence_bert_config.json")
    if spec["maximum_tokens"] > metadata["max_seq_length"]:
        raise ModelingError("Configured chunk length exceeds the pinned sentence encoder limit")
    manifest = {"encoder_version": ENCODER_VERSION, "spec": spec, "files": hashes,
                "dependencies": dependencies,
                "weights_updated": False, "execution_provider": "CPUExecutionProvider"}
    receipt = directory / "assets.json"
    if receipt.exists() and read_json(receipt) != manifest:
        raise ModelingError("Pinned encoder assets or runtime changed")
    atomic_json(receipt, manifest)
    return {"directory": str(directory.resolve()), **manifest}


def token_chunks(ids: list[int], maximum: int, overlap: int, maximum_chunks: int):
    capacity = maximum - 2
    if capacity <= 0 or overlap < 0 or overlap >= capacity:
        raise ModelingError("Invalid encoder chunk/overlap policy")
    if not ids:
        raise ModelingError("Empty tokenized statement")
    result = []
    start, covered = 0, 0
    while start < len(ids):
        end = min(start + capacity, len(ids))
        result.append((ids[start:end], end - max(start, covered)))
        covered = end
        if len(result) > maximum_chunks:
            raise ModelingError("Statement exceeds the configured chunk bound; no silent truncation")
        if end == len(ids):
            break
        start = end - overlap
    return result


def attention_mean(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or np.any(mask.sum(axis=1) == 0):
        raise ModelingError("Invalid encoder hidden-state/attention dimensions")
    return (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)


class FrozenEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, asset_directory: str, spec: dict):
        self.asset_directory = asset_directory
        self.spec = spec

    def fit(self, X, y=None):
        # Fixed pretrained transformation; no fitted corpus statistics or weights.
        return self

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in {"_session", "_tokenizer"}}

    def _load(self):
        if hasattr(self, "_session"):
            return
        dependencies = runtime_versions()
        import onnxruntime as ort
        from tokenizers import Tokenizer
        directory = Path(self.asset_directory)
        receipt = read_json(directory / "assets.json")
        if receipt["spec"] != self.spec or any(hash_file(directory / n) != h for n, h in receipt["files"].items()):
            raise ModelingError("Frozen encoder asset integrity failure")
        if receipt["dependencies"] != dependencies:
            raise ModelingError("Encoder runtime versions differ from pinned assets")
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.spec["cpu_threads"]
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(directory / self.spec["file"]), sess_options=options, providers=["CPUExecutionProvider"])

    def token_ids(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text.strip():
            raise ModelingError("Frozen encoder requires nonempty statement strings")
        if not hasattr(self, "_tokenizer"):
            # Tokenizer-only audit need not allocate the model session.
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(Path(self.asset_directory) / "tokenizer.json"))
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def transform(self, X):
        self._load()
        cls, sep, pad = [self._tokenizer.token_to_id(v) for v in ["[CLS]", "[SEP]", "[PAD]"]]
        if any(v is None for v in [cls, sep, pad]):
            raise ModelingError("Pinned tokenizer is missing required special tokens")
        embeddings = np.zeros((len(X), 384), dtype=np.float32)
        pending = []
        def flush():
            if not pending:
                return
            length = max(len(chunk) + 2 for _, chunk, _ in pending)
            ids = np.full((len(pending), length), pad, dtype=np.int64)
            mask = np.zeros_like(ids)
            for j, (_, chunk, _) in enumerate(pending):
                sequence = [cls, *chunk, sep]
                ids[j, :len(sequence)] = sequence
                mask[j, :len(sequence)] = 1
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            inputs = {i.name: inputs[i.name] for i in self._session.get_inputs()}
            hidden = self._session.run(None, inputs)[0]
            pooled = attention_mean(hidden, mask)
            if pooled.shape != (len(pending), 384):
                raise ModelingError("Unexpected pinned MiniLM output dimension")
            for (row, _, weight), vector in zip(pending, pooled):
                embeddings[row] += vector.astype(np.float32) * weight
            pending.clear()
        for row, text in enumerate(X):
            for chunk, weight in token_chunks(self.token_ids(text), self.spec["maximum_tokens"], self.spec["overlap_tokens"], self.spec["maximum_chunks"]):
                pending.append((row, chunk, weight))
                if len(pending) >= self.spec["batch_size"]:
                    flush()
            # Keep dynamic-quantization batch context fixed per document, so
            # cached features and one-statement inference are identical.
            flush()
        flush()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or np.any(norms <= 0):
            raise ModelingError("Nonfinite or empty frozen embedding")
        return embeddings / norms


def prepare_embeddings(records: list[Record], spec: dict, cache: Path, output: Path, progress=None):
    if any(r.split not in {"train", "validation"} for r in records):
        raise ModelingError("Embedding development must not read test statements")
    assets = prepare_encoder(spec, cache, progress)
    encoder = FrozenEncoder(assets["directory"], spec)
    training = [r for r in records if r.split == "train"]
    lengths = [len(encoder.token_ids(r.statement)) for r in training]
    chunks = [len(token_chunks(list(range(n)), spec["maximum_tokens"], spec["overlap_tokens"], spec["maximum_chunks"])) for n in lengths]
    report = {"split": "train", "records": len(training), "tokenizer_revision": spec["revision"],
        "token_count_excludes_special_tokens": True, "length_percentiles": {str(p): float(np.percentile(lengths, p)) for p in [0, 50, 90, 95, 99, 100]},
        "exceed_single_chunk": sum(n > spec["maximum_tokens"] - 2 for n in lengths),
        "chunk_count_maximum": max(chunks), "chunk_count_mean": float(np.mean(chunks)),
        "resolved_policy": "All content tokens retained in overlapping chunks; no truncation; per-document L2 normalized weighted mean",
        "spec": spec, "assets": assets, "memory_before_encoding": memory_observation()}
    atomic_json(output / "encoder-length-audit.json", report)  # Before any model embedding computation.
    key = identity({"assets": assets, "version": ENCODER_VERSION, "rows": [(r.problem_id, r.clean_sha256) for r in records]})
    directory = cache / "embeddings" / key[:20]
    directory.mkdir(parents=True, exist_ok=True)
    matrix_file = directory / "embeddings.npy"
    receipt_file = directory / "metadata.json"
    if receipt_file.exists():
        receipt = read_json(receipt_file)
        if receipt["key"] != key or hash_file(matrix_file) != receipt["sha256"]:
            raise ModelingError("Embedding cache integrity failure")
        matrix = np.load(matrix_file, mmap_mode="r")
        return encoder, matrix, receipt
    start = time.perf_counter()
    matrix = np.lib.format.open_memmap(directory / "embeddings.partial.npy", mode="w+", dtype=np.float32, shape=(len(records), 384))
    for first in range(0, len(records), 32):
        batch = records[first:first + 32]
        matrix[first:first + len(batch)] = encoder.transform(texts(batch))
        matrix.flush()
        if progress and (first // 32) % 8 == 0:
            progress({"embedded_statements": first + len(batch), "total": len(records)})
    del matrix
    (directory / "embeddings.partial.npy").replace(matrix_file)
    receipt = {"key": key, "sha256": hash_file(matrix_file), "shape": [len(records), 384], "dtype": "float32",
        "rows": [r.problem_id for r in records], "seconds": time.perf_counter() - start,
        "weights_updated": False, "memory_after_encoding": memory_observation(), "assets": assets}
    atomic_json(receipt_file, receipt)
    atomic_json(output / "embedding-performance.json", receipt)
    return encoder, np.load(matrix_file, mmap_mode="r"), receipt
