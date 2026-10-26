"""Bootstrap fallback for pip metadata resolution issues; downloads from official PyPI.

Normal setup uses pip. This helper obtains compatible wheels and verifies PyPI's
SHA-256 digests before a local --no-deps installation. No credentials are used.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import InvalidVersion, Version


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


def main():
    root = Path(".cache/wheels")
    root.mkdir(parents=True, exist_ok=True)
    queue = [Requirement(r) for r in sys.argv[1:]]
    selected, filenames = {}, []
    tags = set(sys_tags())
    while queue:
        req = queue.pop(0)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(req.name)
        if name in selected:
            if selected[name] not in req.specifier:
                raise RuntimeError("conflicting requirements; use standard pip")
            continue
        listing = fetch_json(f"https://pypi.org/pypi/{name}/json")
        versions = []
        for release in listing["releases"]:
            try:
                candidate = Version(release)
            except InvalidVersion:
                continue  # Some long-lived packages retain pre-PEP-440 releases.
            if not candidate.is_prerelease and candidate in req.specifier:
                versions.append(candidate)
        versions.sort(reverse=True)
        chosen = None
        for version in versions:
            metadata = fetch_json(f"https://pypi.org/pypi/{name}/{version}/json")
            if not SpecifierSet(metadata["info"].get("requires_python") or "").contains(".".join(map(str, sys.version_info[:3]))):
                continue
            for item in metadata["urls"]:
                if item["filename"].endswith(".whl") and not item.get("yanked") and parse_wheel_filename(item["filename"])[3] & tags:
                    chosen = item
                    break
            if chosen:
                break
        if not chosen:
            raise RuntimeError(f"no compatible wheel for {name}")
        path = root / chosen["filename"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != chosen["digests"]["sha256"]:
            with urllib.request.urlopen(chosen["url"], timeout=30) as response:
                body = response.read()
            if hashlib.sha256(body).hexdigest() != chosen["digests"]["sha256"]:
                raise RuntimeError("wheel digest mismatch")
            path.write_bytes(body)
        print(f"{name}=={version}", flush=True)
        selected[name] = version
        filenames.append(str(path))
        queue.extend(Requirement(r) for r in metadata["info"].get("requires_dist") or [])
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--disable-pip-version-check", "--progress-bar", "off", *filenames], check=True)


if __name__ == "__main__":
    main()
