from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


def utcnow():
    return datetime.now(timezone.utc)


def canonical(value) -> bytes:
    def default(obj):
        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc).isoformat()
        raise TypeError(type(obj).__name__)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=default).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identity(value) -> str:
    return sha256(canonical(value))


def problem_id(contest_id: int, index: str) -> str:
    import re
    if type(contest_id) is not int or contest_id <= 0 or not re.fullmatch(r"[A-Z][0-9]*", index):
        raise ValueError("invalid Codeforces identity")
    return f"codeforces:{contest_id}:{index}"


class CorpusError(Exception):
    def __init__(self, code: str, retryable: bool = False):
        self.code, self.retryable = code, retryable
        super().__init__(code)


def safe_error(exc: Exception) -> str:
    # Do not log exception text: drivers/HTTP errors may contain connection secrets.
    return exc.code if isinstance(exc, CorpusError) else type(exc).__name__


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname != "codeforces.com" or parts.username
            or parts.password or parts.port not in (None, 443) or parts.fragment):
        raise CorpusError("url_not_allowed")
    return url


@dataclass(frozen=True)
class Settings:
    uri: str = "mongodb://127.0.0.1:27017/"
    database: str = "cpss"
    cache_dir: Path = Path(".cache")
    api_interval: float = 2.2
    page_interval: float = 5.0
    gridfs_threshold: int = 8 * 1024 * 1024

    @classmethod
    def load(cls):
        load_dotenv(override=False)
        return cls(
            uri=os.getenv("CPSS_MONGODB_URI", cls.uri),
            database=os.getenv("CPSS_DATABASE", cls.database),
            cache_dir=Path(os.getenv("CPSS_CACHE_DIR", ".cache")),
            api_interval=max(2.2, float(os.getenv("CPSS_API_INTERVAL", "2.2"))),
            page_interval=max(5.0, float(os.getenv("CPSS_PAGE_INTERVAL", "5"))),
            gridfs_threshold=int(os.getenv("CPSS_GRIDFS_THRESHOLD", "8388608")),
        )


class WorkerLock:
    """OS advisory lock. Automatically released after a crash; never steal a live lock."""
    def __init__(self, directory: Path):
        self.path = directory / "worker.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0)
        if not self.file.read(1):
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise CorpusError("another_worker_is_running") from None
        return self

    def __exit__(self, *args):
        self.file.close()

