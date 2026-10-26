"""Check only Git's publishable candidate set; ignored credentials are not read."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .settings import ROOT


def check_github(root: Path = ROOT) -> dict:
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True)
    names = sorted(set(result.stdout.decode("utf8").rstrip("\0").split("\0")) - {""})
    failures, total = [], 0
    sensitive = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})")
    for name in names:
        path = root / name
        # Do not read anything that is a known credential/artifact or resolves
        # outside the checkout, even if it was accidentally force-added.
        if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                or (path.name.startswith(".env") and path.name != ".env.example")
                or path.name == "token.txt" or path.suffix in {".pem", ".key", ".p12", ".pfx", ".onnx", ".joblib", ".npy", ".npz", ".zip"}
                or name.split("/")[0] in {"data", "exports", "runtime", ".venv", ".cache"}
                or name.startswith(("modeling/artifacts/", "modeling/.cache/"))):
            failures.append({"path": name, "reason": "credential, generated asset or external link"})
            continue
        size = path.stat().st_size
        total += size
        if size > 10 * 1024 * 1024:
            failures.append({"path": name, "reason": "source file exceeds 10 MiB"})
            continue
        try:
            text = path.read_text("utf8")
        except UnicodeError:
            continue
        if sensitive.search(text):
            failures.append({"path": name, "reason": "credential-like content; inspect privately"})
    if failures:
        raise ValueError("GitHub candidate check failed: " + str(failures))
    return {"passed": True, "candidate_files": len(names), "source_bytes": total,
        "ignored_credentials_read": False, "scope": "Current Git candidate set only; heuristic credential checks are not a security guarantee"}
