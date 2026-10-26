"""Explicit build-time installation/retrieval. Never called by server startup."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .settings import DeploymentSettings, ROOT


def prepare(*, install: bool = True) -> dict:
    if install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
                        "-r", str(Path(__file__).with_name("requirements.lock"))], check=True)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    from .bundle import load_runtime, unpack_bundle
    settings = DeploymentSettings.from_env()
    url, digest = os.getenv("CPSS_BUNDLE_URL"), os.getenv("CPSS_BUNDLE_SHA256")
    if url:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise ValueError("CPSS_BUNDLE_URL must be an explicit public HTTPS release URL")
        if not digest or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            raise ValueError("Bundle retrieval requires the independently recorded CPSS_BUNDLE_SHA256")
        with tempfile.TemporaryDirectory(prefix="cpss-download-") as temporary:
            archive = Path(temporary) / "runtime.zip"
            count, checksum = 0, hashlib.sha256()
            with urllib.request.urlopen(url, timeout=45) as response, archive.open("wb") as output:
                if not response.geturl().startswith("https://"):
                    raise ValueError("Bundle download redirected away from HTTPS")
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > 256 * 1024 * 1024:
                        raise ValueError("Bundle archive exceeds the bounded download size")
                    checksum.update(block); output.write(block)
            if checksum.hexdigest() != digest.lower():
                raise ValueError("Downloaded bundle does not match its expected SHA-256")
            unpack_bundle(archive, digest, settings.runtime_dir)
    engine = load_runtime(settings.runtime_dir)
    return {"prepared": True, "searchable_count": len(engine.documents), "runtime_dir": str(settings.runtime_dir),
            "ranking_probes_passed": True, "downloaded_at_startup": False}
