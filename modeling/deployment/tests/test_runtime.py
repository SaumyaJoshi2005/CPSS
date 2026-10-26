from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from modeling.deployment.app import Application, SearchState
from modeling.deployment.settings import DeploymentSettings, canonical_origin
from modeling.deployment import bundle
from modeling.common import Config, ModelingError, ROOT, atomic_json, hash_file, identity, read_json
from modeling.encoder import ENCODER_VERSION
from modeling.retrieval.configuration import SearchConfig
from modeling.retrieval.contexts import Document
from modeling.retrieval.indexing import SearchIndex
from modeling.retrieval.lexical import BM25Index
from modeling.retrieval.search import SearchEngine


class Encoder:
    def __init__(self, *args):
        pass

    def transform(self, texts):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result


@pytest.fixture
def engine(tmp_path):
    settings = SearchConfig.load()
    documents = [Document(f"codeforces:{i}:A", title, rating, tags, statement, editorial,
        {"split": "train", "statement_version_id": f"s{i}", "editorial_version_id": f"e{i}"})
        for i, title, rating, tags, statement, editorial in [
            (1, "Components", 1200, ("dsu",), "Maintain connectivity as edges are added.", "DSU joins components."),
            (2, "Limit", 1800, ("binary search",), "Find the maximum feasible budget.", "Binary search on the answer with a greedy check."),
            (3, "Count", 900, ("dp",), "Count arrangements of a sequence.", "Dynamic programming counts overlapping subproblems.")]]
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "model.onnx").write_bytes(b"synthetic: never executed")
    receipt = {"encoder_version": ENCODER_VERSION, "spec": {"revision": "synthetic"},
               "files": {"model.onnx": hash_file(assets / "model.onnx")}}
    atomic_json(assets / "assets.json", receipt)
    metadata = {**receipt, "directory": str(assets)}
    vectors = Encoder().transform(documents)
    indexes = {view: SearchIndex(view, "synthetic-" + view,
        BM25Index.build([d.text(view) for d in documents], settings.values["bm25"]), vectors,
        {"encoder": metadata}) for view in ["statement", "editorial"]}
    return SearchEngine(settings, Config.load(), bundle.RuntimeCorpus("a" * 64,
        {"train": 3, "validation": 1, "test": 1}), documents, indexes, Encoder())


def invoke(app, path="/api/search", payload=None, origin="https://demo.example", method=None, **extra):
    body = json.dumps(payload).encode() if payload is not None else b""
    env = {"PATH_INFO": path, "REQUEST_METHOD": method or ("POST" if payload is not None else "GET"),
           "CONTENT_TYPE": "application/json", "CONTENT_LENGTH": str(len(body)), "wsgi.input": io.BytesIO(body),
           "HTTP_HOST": "untrusted.invalid", "HTTP_X_FORWARDED_HOST": "untrusted.invalid", **extra}
    if origin is not None:
        env["HTTP_ORIGIN"] = origin
    response = {}
    def start(status, headers):
        response.update(status=int(status.split()[0]), headers=dict(headers))
    response["body"] = b"".join(app(env, start))
    return response


def wait_ready(state):
    deadline = time.monotonic() + 3
    while state.snapshot()[0] == "loading" and time.monotonic() < deadline:
        time.sleep(.01)
    assert state.snapshot()[0] == "ready"


def test_settings_preserve_local_and_require_explicit_public_origin():
    local = DeploymentSettings.from_env({})
    assert (local.host, local.port) == ("127.0.0.1", 8765)
    assert local.accepts_origin(None) and local.accepts_origin("http://127.0.0.1:8765")
    public = DeploymentSettings.from_env({"CPSS_MODE": "deployment", "PORT": "8081", "CPSS_PORT": "8082",
        "CPSS_PUBLIC_ORIGIN": "https://Demo.Example:443/", "CPSS_FRAME_ORIGINS": "https://replit.com"})
    assert (public.host, public.port, public.public_origin) == ("0.0.0.0", 8081, "https://demo.example")
    assert public.accepts_origin("https://demo.example") and not public.accepts_origin(None)
    assert not public.accepts_origin("http://demo.example") and not public.accepts_origin("https://attacker.invalid")
    assert public.frame_origins == ("https://replit.com",)


@pytest.mark.parametrize("updates", [{"CPSS_MODE": "public"}, {"PORT": "0"}, {"PORT": "65536"}, {"PORT": "abc"},
    {"CPSS_INIT_TIMEOUT": "1"}, {"CPSS_MODE": "deployment"},
    {"CPSS_MODE": "deployment", "CPSS_PUBLIC_ORIGIN": "http://demo.example"}])
def test_settings_invalid(updates):
    with pytest.raises(ValueError):
        DeploymentSettings.from_env(updates)


@pytest.mark.parametrize("origin", ["*", "https://*.example", "https://user:password@demo.example", "https://demo.example/path",
    "https://demo.example?x=1", "https://demo.example#x", "https://a,https://b", "https://demo.example\r\nx:1"])
def test_origin_rejects_unsafe_values(origin):
    with pytest.raises(ValueError):
        canonical_origin(origin)


def test_loading_readiness_single_initialization_timeout_and_failure(engine):
    gate, calls = threading.Event(), []
    def load():
        calls.append(1)
        gate.wait(2)
        return engine
    app = Application(DeploymentSettings(), load)
    assert invoke(app, "/healthz")["status"] == 200
    assert invoke(app, "/")["status"] == 200
    assert invoke(app, "/readyz")["status"] == 503
    assert invoke(app, payload={"query": "DSU"}, origin=None)["status"] == 503
    app.state.start(); app.state.start()
    gate.set(); wait_ready(app.state)
    assert calls == [1] and invoke(app, "/readyz")["status"] == 200
    expired = SearchState(lambda: gate.wait(2), .02)
    gate.clear(); expired.start()
    time.sleep(.05)
    assert expired.snapshot()[0:3:2] == ("failed", "initialization_timeout")
    gate.set()
    def fail():
        raise RuntimeError("private filesystem path and secret")
    failed = Application(DeploymentSettings(), fail)
    failed.state.start()
    time.sleep(.05)
    response = invoke(failed, "/readyz")
    assert response["status"] == 503 and b"private" not in response["body"]


def test_public_api_is_readonly_bounded_and_origin_checked(engine):
    app = Application(DeploymentSettings.from_env({"CPSS_MODE": "deployment", "CPSS_PUBLIC_ORIGIN": "https://demo.example"}), lambda: engine)
    app.state.start(); wait_ready(app.state)
    info = json.loads(invoke(app, "/api/info")["body"])
    assert info["features"] == {"review": False, "classifier_boost": False, "tag_boost": True}
    assert invoke(app, "/api/judgments", {"grade": 2})["status"] == 403
    assert invoke(app, "/api/review")["status"] == 403
    assert invoke(app, "/review")["status"] == 403
    assert invoke(app, payload={"query": "DSU"}, origin=None)["status"] == 403
    assert invoke(app, payload={"query": "DSU"}, origin="https://evil.invalid")["status"] == 403
    for payload in [{"query": ""}, {"query": "DSU", "classifier_boost": True}, {"query": "DSU", "unknown": True},
                    {"query": "DSU", "rating_min": 2000, "rating_max": 1000}]:
        assert invoke(app, payload=payload)["status"] == 400
    assert invoke(app, payload={"query": "DSU"}, CONTENT_LENGTH="20000")["status"] == 400
    response = invoke(app, payload={"query": "DSU", "method": "hybrid", "tag_boost": True})
    assert response["status"] == 200
    assert "Access-Control-Allow-Origin" not in response["headers"]
    assert "frame-ancestors 'none'" in response["headers"]["Content-Security-Policy"]
    assert invoke(app, "/app.js", method="HEAD")["body"] == b""
    assert invoke(app, "/../token.txt")["status"] == 404


@pytest.fixture
def release(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "FrozenEncoder", Encoder)
    return bundle.write_bundle(engine, tmp_path / "release")


def load_synthetic(path):
    return bundle.load_runtime(Path(path), expected_source="a" * 64, expected_counts={"train": 3, "validation": 1, "test": 1})


def test_bundle_roundtrip_ranking_order_sparse_and_reproducible(engine, release, tmp_path):
    restored = load_synthetic(release["bundle"])
    assert restored.ids == engine.ids and bundle.probe_results(restored) == bundle.probe_results(engine)
    assert all(i.lexical.weights.format == "csr" for i in restored.indexes.values())
    assert bundle.write_bundle(engine, tmp_path / "release")["sha256"] == release["sha256"]
    target = tmp_path / "unpacked"
    bundle.unpack_bundle(Path(release["archive"]), release["sha256"], target)
    assert load_synthetic(target).ids == engine.ids
    assert bundle.unpack_bundle(Path(release["archive"]), release["sha256"], target)["searchable_count"] == 3
    assert not any("directory" in v for v in read_json(target / "manifest.json")["model_identity"])


def test_bundle_corruption_missing_and_identity_order(release):
    directory = Path(release["bundle"])
    manifest = read_json(directory / "manifest.json")
    manifest["problem_ids"].reverse()
    manifest["content_id"] = bundle.content_identity(manifest)
    atomic_json(directory / "manifest.json", manifest)
    with pytest.raises(ModelingError, match="order"):
        load_synthetic(directory)
    (directory / "documents.jsonl").write_text("corrupt", encoding="utf8")
    with pytest.raises(ModelingError, match="hash"):
        bundle.verify_files(directory)
    (directory / "documents.jsonl").unlink()
    with pytest.raises(ModelingError, match="hash"):
        bundle.verify_files(directory)


def test_dependency_source_and_ui_contracts_fail_closed(release):
    directory = Path(release["bundle"])
    manifest = read_json(directory / "manifest.json")
    original = manifest["runtime_contract"]["dependencies"]["numpy"]
    manifest["runtime_contract"]["dependencies"]["numpy"] = "0"
    atomic_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dependency"):
        load_synthetic(directory)
    manifest["runtime_contract"]["dependencies"]["numpy"] = original
    manifest["source_files"]["modeling/deployment/app.py"] = "0" * 64
    atomic_json(directory / "manifest.json", manifest)
    with pytest.raises(ModelingError, match="source/UI"):
        load_synthetic(directory)


def test_source_line_endings_are_not_model_identity(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"print('same')\r\n")
    b.write_bytes(b"print('same')\n")
    assert bundle.source_hash(a) == bundle.source_hash(b)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/drive", "a/../b", "a//b"])
def test_archive_paths_cannot_escape(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, "bad")
    with pytest.raises(ModelingError, match="path|escapes"):
        bundle.unpack_bundle(archive, hash_file(archive), tmp_path / "out")


def test_backslash_member_rejected_before_platform_path_normalization(tmp_path):
    with pytest.raises(ModelingError, match="path"):
        bundle.safe_path(tmp_path, "a\\b")


def test_archive_requires_independent_digest(release, tmp_path):
    with pytest.raises(ModelingError, match="SHA-256"):
        bundle.unpack_bundle(Path(release["archive"]), "0" * 64, tmp_path / "out")
