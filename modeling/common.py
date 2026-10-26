from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any

from cpss.core import canonical, identity, sha256

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "baseline.json"


class ModelingError(ValueError):
    """A user-facing modeling input or artifact contract failure."""


def read_json(path: Path) -> Any:
    if path.name.casefold() == "token.txt":
        raise ModelingError("Credential files cannot be modeling inputs")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".tmp-" + uuid.uuid4().hex[:12])
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


def hash_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("scikit-learn", "numpy", "scipy", "joblib", "jsonschema"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


@dataclass(frozen=True)
class Config:
    values: dict[str, Any]
    project_root: Path = ROOT

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "Config":
        values = read_json(path)
        if values.get("schema_version") != 1:
            raise ModelingError("Unsupported modeling configuration version")
        if values["thresholds"]["mode"] not in {"global", "per_label"}:
            raise ModelingError("Invalid threshold mode")
        if not values["thresholds"]["grid"] or any(not 0 < x < 1 for x in values["thresholds"]["grid"]):
            raise ModelingError("Threshold grid must contain values strictly between zero and one")
        if min(values["labels"].values()) < 1:
            raise ModelingError("Training support thresholds must be positive")
        if any(x <= 0 for x in values["classifier"]["C_candidates"]):
            raise ModelingError("Regularization parameters must be positive")
        return cls(values)

    def path(self, key: str) -> Path:
        value = Path(self.values[key])
        return value if value.is_absolute() else self.project_root / value

    @property
    def fingerprint(self) -> str:
        return identity(self.values)
