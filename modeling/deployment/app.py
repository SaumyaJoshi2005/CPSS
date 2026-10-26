"""Small production WSGI adapter; initialization and search remain separate."""
from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Callable

from .settings import DeploymentSettings, ROOT

LOG = logging.getLogger("cpss.deployment")
FEATURES = {"review": False, "classifier_boost": False, "tag_boost": True}
STATIC = ROOT / "modeling/application/static"


class SearchState:
    def __init__(self, loader: Callable, timeout: float):
        self.loader, self.timeout = loader, timeout
        self.status, self.engine, self.error = "loading", None, None
        self._lock = threading.Lock()
        self._started = False
        self.started_at = None

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self.started_at = time.monotonic()
        LOG.info("Search initialization started")
        threading.Thread(target=self._initialize, name="cpss-initialize", daemon=True).start()
        timer = threading.Timer(self.timeout, self._expire)
        timer.daemon = True
        timer.start()

    def _expire(self):
        with self._lock:
            if self.status == "loading":
                self.status, self.error = "failed", "initialization_timeout"
                LOG.error("Search initialization deadline exceeded")

    def _initialize(self):
        try:
            engine = self.loader()
            from .measure import memory
            LOG.info("CPSS_OBSERVATION %s", json.dumps(memory(), sort_keys=True))
            with self._lock:
                if self.status != "loading" or time.monotonic() - self.started_at >= self.timeout:
                    self.status, self.error = "failed", "initialization_timeout"
                    return
                self.engine, self.status = engine, "ready"
            LOG.info("Search ready: %s problems, %.3f seconds", len(engine.documents), time.monotonic() - self.started_at)
        except Exception as exc:
            with self._lock:
                self.status, self.error = "failed", "initialization_failed"
            # Exception text can contain private filesystem paths. Keep those
            # out of public errors and ordinary deployment logs.
            LOG.error("Search initialization failed (%s)", type(exc).__name__)

    def snapshot(self):
        with self._lock:
            return self.status, self.engine, self.error


class Application:
    def __init__(self, settings: DeploymentSettings, loader: Callable | None = None):
        self.settings = settings
        if loader is None:
            def loader():
                from .bundle import load_runtime
                return load_runtime(settings.runtime_dir)
        self.state = SearchState(loader, settings.initialization_timeout)

    def __call__(self, environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET")
        try:
            status, payload, kind = self.handle(environ)
        except Exception as exc:
            LOG.error("Request failed (%s)", type(exc).__name__)
            status, payload, kind = 500, {"error": "Search unavailable"}, "application/json"
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode() if isinstance(payload, dict) else payload
        ancestors = "'self' " + " ".join(self.settings.frame_origins) if self.settings.frame_origins else "'none'"
        headers = [("Content-Type", kind + "; charset=utf-8"), ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "strict-origin-when-cross-origin"),
            ("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors " + ancestors)]
        if status == 503:
            headers.append(("Retry-After", "2"))
        if status == 405:
            headers.append(("Allow", "GET, HEAD, POST"))
        start_response(str(status) + " " + HTTPStatus(status).phrase, headers)
        return [] if method == "HEAD" else [body]

    def handle(self, env):
        path = env.get("PATH_INFO", "/")
        method = env.get("REQUEST_METHOD", "GET")
        if method not in {"GET", "HEAD", "POST"}:
            return 405, {"error": "Method not allowed"}, "application/json"
        status, engine, error = self.state.snapshot()
        if method in {"GET", "HEAD"}:
            if path == "/healthz":
                return 200, {"status": "alive"}, "application/json"
            if path == "/readyz":
                return (200 if status == "ready" else 503), {"status": status, "error": error}, "application/json"
            if path in {"/review", "/api/review"}:
                return 403, {"error": "This demonstration is read-only"}, "application/json"
            if path == "/api/info":
                if status != "ready":
                    return 503, {"status": status, "error": error or "Search is loading", "features": FEATURES}, "application/json"
                return 200, {**engine.info(), "features": FEATURES, "read_only": True}, "application/json"
            files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "application/javascript"), "/style.css": ("style.css", "text/css")}
            if path in files:
                name, kind = files[path]
                return 200, (STATIC / name).read_bytes(), kind
            return 404, {"error": "Not found"}, "application/json"
        if path == "/api/judgments":
            return 403, {"error": "Judgment writes are disabled in this read-only demonstration"}, "application/json"
        if path != "/api/search":
            return 404, {"error": "Not found"}, "application/json"
        if not self.settings.accepts_origin(env.get("HTTP_ORIGIN")):
            return 403, {"error": "Origin not allowed"}, "application/json"
        if status != "ready":
            return 503, {"status": status, "error": error or "Search is loading"}, "application/json"
        try:
            length = env.get("CONTENT_LENGTH", "")
            if not length.isdecimal() or not 0 < int(length) <= 16384:
                raise ValueError("Invalid body length")
            if env.get("CONTENT_TYPE", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ValueError("Expected JSON")
            payload = json.loads(env["wsgi.input"].read(int(length)))
            allowed = {"query", "method", "view", "limit", "candidates", "rating_min", "rating_max", "tag_boost", "classifier_boost"}
            if not isinstance(payload, dict) or not set(payload) <= allowed:
                raise ValueError("Unsupported options")
            if payload.get("classifier_boost") is not False and "classifier_boost" in payload:
                return 400, {"error": "Classifier boost is not included in this public runtime"}, "application/json"
            return 200, engine.search(**payload), "application/json"
        except (ValueError, KeyError, TypeError):
            return 400, {"error": "Invalid search request. Check the query, rating bounds and search options."}, "application/json"


def serve(settings: DeploymentSettings):
    from waitress import create_server
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = Application(settings)
    # Bind before starting any numerical imports, hashing or encoder work.
    server = create_server(app, host=settings.host, port=settings.port, threads=4,
        connection_limit=64, max_request_body_size=16384, channel_timeout=30,
        clear_untrusted_proxy_headers=True, expose_tracebacks=False)
    LOG.info("HTTP listening on %s:%s; mode=%s; read-only", settings.host, settings.port, settings.mode)
    app.state.start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
