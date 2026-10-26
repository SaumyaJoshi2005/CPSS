"""Measure a fresh child process and stop only that process after HTTP checks."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from .settings import ROOT
from .smoke import call, smoke


def memory() -> dict:
    from ..resources import memory_observation
    result = memory_observation()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["process_rss_bytes"] = int(line.split()[1]) * 1024
    return result


def measure(bundle: Path, port: int, origin: str, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    env = {**os.environ, "CPSS_MODE": "deployment", "PORT": str(port),
           "CPSS_PUBLIC_ORIGIN": origin, "CPSS_RUNTIME_DIR": str(bundle.resolve())}
    url = f"http://127.0.0.1:{port}"
    # A preexisting listener must never be mistaken for this child's readiness.
    import socket
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    with log.open("w", encoding="utf8") as stream:
        started = time.perf_counter()
        process = subprocess.Popen([sys.executable, "-B", "-m", "modeling.deployment", "serve"],
            cwd=ROOT, env=env, stdout=stream, stderr=stream,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        live = ready = None
        try:
            while time.perf_counter() - started < 130:
                if process.poll() is not None:
                    raise RuntimeError("Measured server exited before readiness; inspect the private measurement log")
                try:
                    if call(url + "/healthz")[0] == 200 and live is None:
                        live = time.perf_counter() - started
                    code, state = call(url + "/readyz")
                    if code == 200:
                        ready = time.perf_counter() - started
                        break
                    if state.get("status") == "failed":
                        raise RuntimeError("Measured initialization failed; inspect the private measurement log")
                except (urllib.error.URLError, ConnectionError):
                    pass
                time.sleep(.05)
            if ready is None:
                raise RuntimeError("Measured server did not become ready within 130 seconds")
            report = smoke(url, origin)
            report.update(platform=platform.platform(), cold_process_to_liveness_seconds=live,
                          cold_process_to_readiness_seconds=ready,
                          startup_protocol="Fresh interpreter/process; existing local disk assets; OS disk caches not flushed; polling at 50 ms; no startup download or corpus embedding")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
    observations = [line.split("CPSS_OBSERVATION ", 1)[1] for line in log.read_text("utf8").splitlines() if "CPSS_OBSERVATION " in line]
    report["ready_process_memory"] = json.loads(observations[-1]) if observations else None
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return report
