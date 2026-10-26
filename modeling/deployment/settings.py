from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def canonical_origin(value: str, *, https_required: bool = False) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid configured origin") from exc
    if (parts.scheme not in ({"https"} if https_required else {"http", "https"})
            or not parts.hostname or parts.username or parts.password
            or parts.path not in {"", "/"} or parts.query or parts.fragment
            or any(c in value for c in ["*", ",", " ", "\r", "\n", "\\"])):
        raise ValueError("Origin must be one explicit HTTP(S) origin without credentials, wildcards or paths")
    host = parts.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    suffix = "" if port is None or port == {"http": 80, "https": 443}[parts.scheme] else ":" + str(port)
    return parts.scheme + "://" + host + suffix


@dataclass(frozen=True)
class DeploymentSettings:
    mode: str = "local"
    port: int = 8765
    public_origin: str | None = None
    runtime_dir: Path = ROOT / "runtime"
    frame_origins: tuple[str, ...] = ()
    initialization_timeout: int = 120

    @property
    def host(self) -> str:
        return "0.0.0.0" if self.mode == "deployment" else "127.0.0.1"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DeploymentSettings":
        env = os.environ if environ is None else environ
        mode = env.get("CPSS_MODE", "local")
        if mode not in {"local", "deployment"}:
            raise ValueError("CPSS_MODE must be local or deployment")
        raw_port = env.get("PORT", env.get("CPSS_PORT", "8080" if mode == "deployment" else "8765"))
        if not raw_port.isdecimal() or not 1 <= int(raw_port) <= 65535:
            raise ValueError("PORT/CPSS_PORT must be an integer from 1 to 65535")
        raw_timeout = env.get("CPSS_INIT_TIMEOUT", "120")
        if not raw_timeout.isdecimal() or not 5 <= int(raw_timeout) <= 300:
            raise ValueError("CPSS_INIT_TIMEOUT must be an integer from 5 to 300 seconds")
        origin = env.get("CPSS_PUBLIC_ORIGIN")
        if mode == "deployment" and not origin:
            raise ValueError("Deployment requires CPSS_PUBLIC_ORIGIN set to the exact HTTPS origin")
        origin = canonical_origin(origin, https_required=mode == "deployment") if origin else None
        frames = tuple(canonical_origin(v.strip(), https_required=True) for v in env.get("CPSS_FRAME_ORIGINS", "").split(",") if v.strip())
        path = Path(env.get("CPSS_RUNTIME_DIR", str(ROOT / "runtime"))).expanduser()
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        return cls(mode, int(raw_port), origin, path, frames, int(raw_timeout))

    def accepts_origin(self, value: str | None) -> bool:
        if value is None:
            return self.mode == "local"
        try:
            actual = canonical_origin(value)
        except ValueError:
            return False
        expected = self.public_origin or f"http://127.0.0.1:{self.port}"
        return actual == expected
