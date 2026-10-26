from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import sparse

from ..common import Config, ModelingError, ROOT, atomic_json, hash_file, identity, read_json
from ..encoder import ENCODER_VERSION, FrozenEncoder
from ..labels import Taxonomy
from ..retrieval.configuration import SearchConfig
from ..retrieval.contexts import CONTEXT_VERSION, Document
from ..retrieval.indexing import INDEX_VERSION, SearchIndex
from ..retrieval.lexical import BM25Index, TOKENIZER_VERSION
from ..retrieval.search import SearchEngine
from .compatibility import actual_dependencies, producer, verify_runtime

FORMAT = "cpss-serving-bundle-v1"
MAX_TOTAL_BYTES = 512 * 1024 * 1024
PROBE_QUERIES = (
    "DSU with connectivity queries.",
    "Maintain connected groups as new connections are added.",
    "Binary search on the answer with a greedy feasibility check.",
)


def safe_path(root: Path, name: str) -> Path:
    item = PurePosixPath(name)
    if not name or item.as_posix() != name or item.is_absolute() or ".." in item.parts or "\\" in name or ":" in name:
        raise ModelingError("Invalid portable bundle member path")
    path = (root / Path(*item.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ModelingError("Bundle member escapes its root")
    return path


def serving_sources() -> list[Path]:
    paths = [ROOT / "src/cpss" / n for n in ["__init__.py", "core.py", "export.py"]]
    paths += [ROOT / "modeling" / n for n in ["__init__.py", "common.py", "data.py", "encoder.py", "features.py", "labels.py", "metrics.py", "inference.py", "resources.py"]]
    paths += list((ROOT / "modeling/retrieval").glob("*.py"))
    paths += list((ROOT / "modeling/deployment").glob("*.py"))
    paths += [ROOT / "modeling/deployment/requirements.lock"]
    return sorted(paths)


def source_hash(path: Path) -> str:
    """Git checkout line endings are storage details, not source identity."""
    from cpss.core import sha256
    return sha256(path.read_text(encoding="utf8").replace("\r\n", "\n").encode("utf8"))


def probe_results(engine: SearchEngine) -> list[dict]:
    probes = []
    for view in ["statement", "editorial"]:
        for method in ["bm25", "dense", "hybrid"]:
            for query in PROBE_QUERIES:
                result = engine.search(query, method=method, view=view, limit=10, tag_boost=False, classifier_boost=False)
                probes.append({"query": query, "method": method, "view": view,
                    "problem_ids": [r["problem_id"] for r in result["results"]],
                    "scores": [r["score"] for r in result["results"]]})
    return probes


def content_identity(manifest: dict) -> str:
    return identity({k: manifest[k] for k in ["source_manifest_sha256", "problem_ids", "signatures", "configuration_sha256", "representations", "model_identity"]})


def write_bundle(engine: SearchEngine, output: Path) -> dict:
    """Package data-only representations from an already verified local engine."""
    if set(engine.indexes) != {"statement", "editorial"} or any(i.dense is None for i in engine.indexes.values()):
        raise ModelingError("The public bundle requires both views and all three retrieval methods")
    if len(set(engine.ids)) != len(engine.ids) or any(d.provenance["split"] != "train" for d in engine.documents):
        raise ModelingError("Only unique preserved training identities may be packaged")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (".building-" + uuid.uuid4().hex[:10])
    temporary.mkdir()
    with (temporary / "documents.jsonl").open("w", encoding="utf8", newline="\n") as stream:
        for document in engine.documents:
            stream.write(json.dumps(asdict(document), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    atomic_json(temporary / "taxonomy.json", engine.taxonomy.spec)
    atomic_json(temporary / "search.json", engine.settings.values)
    assets = engine.indexes["statement"].metadata["encoder"]
    if assets != engine.indexes["editorial"].metadata["encoder"]:
        raise ModelingError("Context views use different model identities")
    source = Path(assets["directory"])
    encoder_files = [*assets["files"], "assets.json"]
    for name in encoder_files:
        src = safe_path(source, name)
        if name != "assets.json" and hash_file(src) != assets["files"][name]:
            raise ModelingError("Encoder asset changed before packaging")
        target = safe_path(temporary / "encoder", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    # The original receipt contains only spec, hashes and runtime versions, not
    # its absolute directory. FrozenEncoder will verify it at its new location.
    receipt = read_json(temporary / "encoder/assets.json")
    model = {k: receipt[k] for k in ["encoder_version", "spec", "files"]}
    indexes = {}
    for view, index in engine.indexes.items():
        folder = temporary / "indexes" / view
        folder.mkdir(parents=True)
        sparse.save_npz(folder / "bm25.npz", index.lexical.weights)
        np.save(folder / "dense.npy", np.asarray(index.dense, dtype=np.float32))
        atomic_json(folder / "vocabulary.json", index.lexical.vocabulary)
        indexes[view] = {"source_index_key": index.key, "shape": list(index.lexical.weights.shape),
            "dense_shape": list(index.dense.shape), "problem_order_sha256": identity(engine.ids)}
    ui = temporary / "ui"
    ui.mkdir()
    for name in ["index.html", "app.js", "style.css"]:
        shutil.copyfile(ROOT / "modeling/application/static" / name, ui / name)
    notices = ROOT / "modeling/deployment/THIRD_PARTY_NOTICES.md"
    if notices.exists():
        shutil.copyfile(notices, temporary / "THIRD_PARTY_NOTICES.md")
    probes = probe_results(engine)
    atomic_json(temporary / "ranking-probes.json", probes)
    files = {p.relative_to(temporary).as_posix(): {"sha256": hash_file(p), "bytes": p.stat().st_size}
             for p in sorted(temporary.rglob("*")) if p.is_file()}
    manifest = {"format": FORMAT, "source_manifest_sha256": engine.corpus.manifest_sha256,
        "split_counts": engine.corpus.split_counts, "split": "train", "problem_ids": engine.ids, "searchable_count": len(engine.ids),
        "signatures": {v: identity([d.signature(v) for d in engine.documents]) for v in indexes},
        "configuration_sha256": engine.settings.fingerprint, "representations": {"index": INDEX_VERSION, "context": CONTEXT_VERSION, "tokenizer": TOKENIZER_VERSION},
        "model_identity": model, "indexes": indexes, "files": files, "producer": producer(),
        "runtime_contract": {"python": ["3.12", "3.13"], "dependencies": actual_dependencies(),
            "array_formats": ["numpy-npy-v1-or-v2", "scipy-csr-npz"],
            "compatibility": "Data-only numeric arrays and fixed ONNX weights; exact package versions plus mandatory ranking probes; no pickle"},
        "features": {"review": False, "classifier_boost": False, "tag_boost": True},
        "source_normalization": "utf8-lf",
        "source_files": {p.relative_to(ROOT).as_posix(): source_hash(p) for p in serving_sources()},
        "ui_source_files": {"modeling/application/static/" + n: source_hash(ui / n) for n in ["index.html", "app.js", "style.css"]},
        "not_included": ["validation/test documents", "private tests", "accepted code", "raw corpus blobs", "classifier pickle", "judgments", "credentials", "virtual environments"]}
    manifest["content_id"] = content_identity(manifest)
    atomic_json(temporary / "manifest.json", manifest)
    digest = hash_file(temporary / "manifest.json")
    destination = output / ("bundle-" + digest[:20])
    if destination.exists():
        verify_files(destination)
        if not temporary.resolve().is_relative_to(output.resolve()) or not temporary.name.startswith(".building-"):
            raise ModelingError("Refusing unsafe temporary-directory cleanup")
        shutil.rmtree(temporary)  # Only this invocation's known temporary directory.
    else:
        temporary.rename(destination)
    archive = output / (destination.name + ".zip")
    if not archive.exists():
        pending = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for p in sorted(destination.rglob("*")):
                if p.is_file():
                    info = zipfile.ZipInfo(p.relative_to(destination).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    zipped.writestr(info, p.read_bytes(), compresslevel=6)
        pending.replace(archive)
    result = {"bundle": str(destination.resolve()), "archive": str(archive.resolve()), "sha256": hash_file(archive),
        "manifest_sha256": digest, "archive_bytes": archive.stat().st_size, "uncompressed_bytes": sum(v["bytes"] for v in files.values()),
        "searchable_count": len(engine.ids), "content_id": manifest["content_id"]}
    atomic_json(output / "bundle-release.json", result)
    archive.with_suffix(".zip.sha256").write_text(result["sha256"] + "  " + archive.name + "\n", encoding="ascii")
    return result


def verify_files(directory: Path) -> dict:
    try:
        manifest = read_json(directory / "manifest.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelingError("Runtime manifest is missing or malformed; unpack a verified serving archive") from exc
    if manifest.get("format") != FORMAT or manifest.get("content_id") != content_identity(manifest):
        raise ModelingError("Unsupported or inconsistent serving-bundle identity")
    total = 0
    for name, details in manifest["files"].items():
        path = safe_path(directory, name)
        total += details["bytes"]
        if total > MAX_TOTAL_BYTES or not path.is_file() or path.stat().st_size != details["bytes"] or hash_file(path) != details["sha256"]:
            raise ModelingError("Serving-bundle asset size/hash verification failed")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ModelingError("Bundle contains missing or unlisted assets")
    return manifest


@dataclass(frozen=True)
class RuntimeCorpus:
    manifest_sha256: str
    split_counts: dict[str, int]


class PortableSearchEngine(SearchEngine):
    def search(self, *args, **kwargs):
        if kwargs.get("classifier_boost"):
            raise ModelingError("Classifier boost is not included in this public runtime")
        return super().search(*args, **kwargs)


def load_runtime(directory: Path, *, expected_source: str | None = None, expected_counts: dict | None = None) -> PortableSearchEngine:
    started = time.perf_counter()
    manifest = verify_files(directory)
    baseline = Config.load()
    expected_source = expected_source or baseline.values["expected_manifest_sha256"]
    expected_counts = expected_counts or baseline.values["expected_split_counts"]
    if manifest["source_manifest_sha256"] != expected_source or manifest["split_counts"] != expected_counts or manifest["split"] != "train":
        raise ModelingError("Serving bundle does not match the pinned corpus and preserved counts")
    verify_runtime(manifest["runtime_contract"])
    if manifest["representations"] != {"index": INDEX_VERSION, "context": CONTEXT_VERSION, "tokenizer": TOKENIZER_VERSION}:
        raise ModelingError("Serving representation versions differ from this application")
    if manifest.get("source_normalization") != "utf8-lf":
        raise ModelingError("Unsupported serving-source normalization")
    for name, digest in {**manifest["source_files"], **manifest["ui_source_files"]}.items():
        if source_hash(safe_path(ROOT, name)) != digest:
            raise ModelingError("Serving source/UI differs from the packaged release; package a new bundle")
    settings = SearchConfig.load(directory / "search.json", project_root=directory)
    if settings.fingerprint != manifest["configuration_sha256"]:
        raise ModelingError("Serving search configuration changed")
    documents = []
    with (directory / "documents.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            value = json.loads(line)
            value["original_tags"] = tuple(value["original_tags"])
            documents.append(Document(**value))
    ids = [d.problem_id for d in documents]
    if ids != manifest["problem_ids"] or len(set(ids)) != len(ids) or len(ids) != expected_counts["train"] or len(ids) != manifest["searchable_count"]:
        raise ModelingError("Serving document identity/order/count mismatch")
    for view in ["statement", "editorial"]:
        if identity([d.signature(view) for d in documents]) != manifest["signatures"][view]:
            raise ModelingError("Serving statement/editorial source signatures disagree")
    Taxonomy.load(directory / "taxonomy.json")
    indexes = {}
    for view, details in manifest["indexes"].items():
        folder = directory / "indexes" / view
        if view not in {"statement", "editorial"} or details["problem_order_sha256"] != identity(ids):
            raise ModelingError("Serving representation ordering mismatch")
        weights = sparse.load_npz(folder / "bm25.npz").tocsr()
        vocabulary = read_json(folder / "vocabulary.json")
        vectors = np.load(folder / "dense.npy", mmap_mode="r", allow_pickle=False)
        if (weights.shape != (len(ids), len(vocabulary)) or list(weights.shape) != details["shape"]
                or vectors.shape != (len(ids), 384) or list(vectors.shape) != details["dense_shape"]
                or not np.isfinite(weights.data).all() or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
                or set(vocabulary.values()) != set(range(len(vocabulary)))):
            raise ModelingError("Serving sparse/dense representation validation failed")
        indexes[view] = SearchIndex(view, details["source_index_key"], BM25Index(vocabulary, weights, settings.values["bm25"]), vectors, details)
    if set(indexes) != {"statement", "editorial"}:
        raise ModelingError("Both context views are required")
    assets = read_json(directory / "encoder/assets.json")
    if {k: assets[k] for k in ["encoder_version", "spec", "files"]} != manifest["model_identity"] or assets["encoder_version"] != ENCODER_VERSION:
        raise ModelingError("Frozen encoder model identity changed")
    encoder = FrozenEncoder(str((directory / "encoder").resolve()), assets["spec"])
    engine = PortableSearchEngine(settings, Config({"taxonomy_path": "taxonomy.json"}, directory), RuntimeCorpus(expected_source, expected_counts), documents, indexes, encoder)
    expected_probes = read_json(directory / "ranking-probes.json")
    actual_probes = probe_results(engine)  # Initializes the encoder before readiness.
    if len(expected_probes) != len(actual_probes):
        raise ModelingError("Incomplete serving ranking probes")
    for expected, actual in zip(expected_probes, actual_probes):
        if ({k: expected[k] for k in ["query", "method", "view", "problem_ids"]} != {k: actual[k] for k in ["query", "method", "view", "problem_ids"]}
                or not np.allclose(expected["scores"], actual["scores"], rtol=1e-5, atol=2e-6)):
            raise ModelingError("Cross-runtime ranking consistency probe failed; serving is not ready")
    engine.load_seconds = time.perf_counter() - started
    return engine


def unpack_bundle(archive: Path, expected_sha256: str, destination: Path) -> dict:
    if len(expected_sha256) != 64 or hash_file(archive) != expected_sha256.lower():
        raise ModelingError("Runtime archive SHA-256 mismatch")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpss-unpack-", dir=destination.parent) as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if len(members) > 128 or len({m.filename for m in members}) != len(members) or sum(m.file_size for m in members) > MAX_TOTAL_BYTES:
                raise ModelingError("Runtime archive limits/identities are invalid")
            for member in members:
                if member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ModelingError("Runtime archive directories/symlinks are not supported")
                path = safe_path(root, member.filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
        manifest = verify_files(root)
        if destination.exists():
            verify_files(destination)
            if hash_file(destination / "manifest.json") != hash_file(root / "manifest.json"):
                raise ModelingError("An existing different runtime is preserved; select a new runtime directory")
        else:
            root.rename(destination)
    return {"runtime_dir": str(destination), "content_id": manifest["content_id"], "searchable_count": manifest["searchable_count"]}
