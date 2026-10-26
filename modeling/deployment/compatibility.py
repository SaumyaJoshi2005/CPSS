from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def pinned_dependencies() -> dict[str, str]:
    entries = Path(__file__).with_name("requirements.lock").read_text("utf8").splitlines()
    return dict(line.strip().split("==", 1) for line in entries if line.strip() and not line.startswith("#"))


def actual_dependencies() -> dict[str, str]:
    result = {}
    for name in pinned_dependencies():
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def verify_runtime(contract: dict) -> None:
    if contract.get("python") != ["3.12", "3.13"] or sys.version_info[:2] not in {(3, 12), (3, 13)}:
        raise ValueError("Portable serving supports Python 3.12 or 3.13")
    if contract.get("dependencies") != actual_dependencies():
        raise ValueError("Runtime dependency versions differ; install modeling/deployment/requirements.lock")
    if contract.get("array_formats") != ["numpy-npy-v1-or-v2", "scipy-csr-npz"]:
        raise ValueError("Unsupported portable numeric representation")


def producer() -> dict:
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "dependencies": actual_dependencies()}
