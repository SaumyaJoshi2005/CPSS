from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..common import Config, ModelingError
from ..retrieval.configuration import SearchConfig
from ..retrieval.search import SearchEngine

STATIC = Path(__file__).parent / "static"


def make_server(engine: SearchEngine, host="127.0.0.1", port=8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ModelingError("The submission application binds to localhost only")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print("CPSS HTTP: " + format % args, file=sys.stderr)

        def send(self, status: int, content: bytes, kind="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def json_response(self, status, value):
            self.send(status, json.dumps(value, ensure_ascii=True, allow_nan=False).encode())

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/info":
                return self.json_response(200, engine.info())
            if path == "/api/review":
                try:
                    from ..retrieval.judgments import review_queue
                    return self.json_response(200, review_queue(engine))
                except (OSError, ValueError, KeyError) as exc:
                    return self.json_response(400, {"error": str(exc)})
            files = {"/": ("index.html", "text/html; charset=utf-8"), "/review": ("review.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "application/javascript; charset=utf-8"), "/review.js": ("review.js", "application/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in files:
                return self.json_response(404, {"error": "Not found"})
            name, kind = files[path]
            self.send(200, (STATIC / name).read_bytes(), kind)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                expected = f"http://{self.headers.get('Host')}"
                if origin and origin != expected:
                    raise ModelingError("Cross-origin requests are not supported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ModelingError("Request body must be between 1 and 16384 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ModelingError("Request must be a JSON object")
                path = urlparse(self.path).path
                if path == "/api/search":
                    allowed = {"query", "method", "view", "limit", "candidates", "rating_min", "rating_max", "tag_boost", "classifier_boost"}
                    if not set(payload) <= allowed:
                        raise ModelingError("Unsupported search option")
                    return self.json_response(200, engine.search(**payload))
                if path == "/api/judgments":
                    from ..retrieval.judgments import record_judgment
                    return self.json_response(200, record_judgment(engine, payload))
                return self.json_response(404, {"error": "Not found"})
            except (ModelingError, ValueError, KeyError, TypeError, OSError) as exc:
                self.json_response(400, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), Handler)


def serve(settings: SearchConfig, config: Config, port: int | None = None):
    engine = SearchEngine.load(settings, config)
    host = settings.values["application"]["host"]
    server = make_server(engine, host, settings.values["application"]["port"] if port is None else port)
    print(f"CPSS ready: http://{host}:{server.server_port} — {len(engine.documents)} training problems", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
