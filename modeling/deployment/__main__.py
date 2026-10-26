from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .settings import DeploymentSettings, ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m modeling.deployment")
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package", help="Create a verified, portable serving bundle from existing local indexes")
    package.add_argument("--output", type=Path, default=ROOT / "modeling/artifacts/runtime")
    verify = sub.add_parser("verify", help="Verify a runtime and execute its frozen ranking probes")
    verify.add_argument("--bundle", type=Path, required=True)
    unpack = sub.add_parser("unpack")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--sha256", required=True)
    unpack.add_argument("--output", type=Path, default=ROOT / "runtime")
    sub.add_parser("serve", help="Offline, read-only Waitress server; local binding unless deployment mode is explicit")
    sub.add_parser("github-check", help="Check Git candidates for generated assets, credentials and large source files")
    build = sub.add_parser("prepare", help="Explicit build-time dependency setup and optional checked release download")
    build.add_argument("--no-install", action="store_true")
    check = sub.add_parser("smoke", help="HTTP health, readiness and retrieval checks against a running runtime")
    check.add_argument("--url", default="http://127.0.0.1:8080")
    check.add_argument("--origin", required=True)
    check.add_argument("--output", type=Path)
    measurement = sub.add_parser("measure", help="Measure a fresh server, smoke-test it, then stop that child process")
    measurement.add_argument("--bundle", type=Path, required=True)
    measurement.add_argument("--port", type=int, default=8766)
    measurement.add_argument("--origin", default="https://demo.example")
    measurement.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from .app import serve
            serve(DeploymentSettings.from_env())
            return 0
        if args.command == "prepare":
            from .prepare import prepare
            result = prepare(install=not args.no_install)
        elif args.command == "github-check":
            from .github import check_github
            result = check_github()
        elif args.command == "package":
            from ..common import Config
            from ..retrieval.configuration import SearchConfig
            from ..retrieval.search import SearchEngine
            from .bundle import write_bundle
            result = write_bundle(SearchEngine.load(SearchConfig.load(), Config.load()), args.output)
        elif args.command == "verify":
            from .bundle import load_runtime
            engine = load_runtime(args.bundle.resolve())
            result = {"verified": True, "searchable_count": len(engine.documents), "ranking_probes_passed": True, "initialization_seconds": engine.load_seconds}
        elif args.command == "unpack":
            from .bundle import unpack_bundle
            result = unpack_bundle(args.archive, args.sha256, args.output)
        elif args.command == "measure":
            from .measure import measure
            result = measure(args.bundle, args.port, args.origin, args.output)
        else:
            from .smoke import smoke
            result = smoke(args.url, args.origin, args.output)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as exc:
        # CLI diagnostics name the failure class without echoing a download URL,
        # environment secrets, driver text or private paths.
        print(json.dumps({"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
