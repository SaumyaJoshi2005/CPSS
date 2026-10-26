from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .core import CorpusError, Settings, canonical, identity, safe_url, sha256

USER_AGENT = "CPSSCorpus/0.1 (academic corpus collector; single worker)"
MAX_RESPONSE = 64 * 1024 * 1024


class Limiter:
    def __init__(self, interval, clock=time.monotonic, sleep=time.sleep):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.last = None

    def wait(self):
        now = self.clock()
        if self.last is not None:
            delay = self.interval - (now - self.last)
            if delay > 0:
                self.sleep(delay)
        self.last = self.clock()


@dataclass
class Fetched:
    body: bytes
    url: str
    retrieved_at: datetime
    status_code: int = 200
    cached: bool = False

    def provenance(self, method):
        return {"url": self.url, "retrieved_at": self.retrieved_at,
                "status_code": self.status_code, "method": method}


class Transport:
    def __init__(self, settings: Settings, allow_pages=False, client=None, sleep=time.sleep):
        self.settings, self.allow_pages, self.sleep = settings, allow_pages, sleep
        self.cache = settings.cache_dir / "http"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.api_limiter = Limiter(max(2.2, settings.api_interval), sleep=sleep)
        self.page_limiter = Limiter(max(5.0, settings.page_interval), sleep=sleep)
        self.client = client or httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=False,
                                            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
        self.robots = None
        self.blocked = {True: False, False: False}

    def close(self):
        self.client.close()

    def _read_cache(self, key, ttl):
        header, bodyfile = self.cache / f"{key}.json", self.cache / f"{key}.body"
        if not header.exists() or not bodyfile.exists():
            return None
        try:
            meta = json.loads(header.read_text("utf-8"))
            if time.time() - meta["stored_at"] > ttl:
                return None
            body = bodyfile.read_bytes()
            if sha256(body) != meta["sha256"]:
                return None
            return Fetched(body, meta["url"], datetime.fromisoformat(meta["retrieved_at"]), cached=True)
        except (ValueError, OSError, KeyError):
            return None

    def _write_cache(self, key, fetched):
        bodyfile = self.cache / f"{key}.body"
        tmpbody = bodyfile.with_suffix(".body.tmp")
        tmpbody.write_bytes(fetched.body)
        tmpbody.replace(bodyfile)
        header = self.cache / f"{key}.json"
        tmpheader = header.with_suffix(".json.tmp")
        tmpheader.write_bytes(canonical({"url": fetched.url, "stored_at": time.time(),
            "retrieved_at": fetched.retrieved_at.isoformat(), "sha256": sha256(fetched.body)}))
        tmpheader.replace(header)

    def _wait(self, api):
        limiter = self.api_limiter if api else self.page_limiter
        # Persist last request start across separate CLI runs (which use the worker lock).
        state = self.cache / ("api-last-start" if api else "page-last-start")
        if state.exists():
            try:
                delay = limiter.interval - (time.time() - float(state.read_text()))
                if delay > 0:
                    self.sleep(min(delay, limiter.interval))
            except (ValueError, OSError):
                pass
        limiter.wait()
        state.write_text(str(time.time()))

    def _request(self, url, api):
        safe_url(url)
        for redirect in range(4):
            self._wait(api)
            try:
                with self.client.stream("GET", url) as response:
                    if response.is_redirect:
                        if redirect == 3 or "location" not in response.headers:
                            raise CorpusError("redirect_limit")
                        target = safe_url(urljoin(url, response.headers["location"]))
                        if api and not urlsplit(target).path.startswith("/api/"):
                            raise CorpusError("api_redirect_not_allowed")
                        if not api and self.robots and not self.robots.can_fetch(USER_AGENT, target):
                            raise CorpusError("robots_disallowed")
                        url = target
                        continue
                    chunks, size = [], 0
                    for data in response.iter_bytes():
                        size += len(data)
                        if size > MAX_RESPONSE:
                            raise CorpusError("response_too_large")
                        chunks.append(data)
                    return response.status_code, dict(response.headers), b"".join(chunks), url
            except httpx.RequestError:
                raise CorpusError("network_failure", retryable=True) from None
        raise CorpusError("redirect_limit")

    def fetch(self, url, api=False, refresh=False):
        safe_url(url)
        if self.blocked[api]:
            raise CorpusError("origin_blocked_for_this_batch")
        if not api and not self.allow_pages:
            raise CorpusError("page_adapter_requires_opt_in")
        if not api and urlsplit(url).path != "/robots.txt":
            self.check_robots(url)
        key = identity({"url": url, "language": "en", "agent": USER_AGENT})
        cached = None if refresh else self._read_cache(key, 3600 if api else 86400)
        if cached:
            return cached
        for attempt in range(3):
            try:
                status, headers, body, final_url = self._request(url, api)
                if status in {401, 403}:
                    self.blocked[api] = True
                    raise CorpusError(f"http_{status}_blocked")
                if status == 429 or status in {500, 502, 503, 504}:
                    delay = self.retry_after(headers.get("retry-after"))
                    if delay > 60:
                        raise CorpusError("retry_after_exceeds_inline_budget", retryable=True)
                    if attempt < 2:
                        self.sleep(max(delay, min(30, 2 ** (attempt + 1) + random.random())))
                        continue
                    raise CorpusError(f"http_{status}", retryable=True)
                if status != 200:
                    raise CorpusError(f"http_{status}")
                if api:
                    try:
                        payload = json.loads(body)
                    except (ValueError, UnicodeError):
                        raise CorpusError("api_non_json_or_challenge") from None
                    if payload.get("status") != "OK":
                        if "call limit" in str(payload.get("comment", "")).lower() and attempt < 2:
                            self.sleep(5)
                            continue
                        raise CorpusError("api_failed")
                    if "result" not in payload:
                        raise CorpusError("api_missing_result")
                elif urlsplit(final_url).path != "/robots.txt":
                    from .parsers import soup_checked
                    try:
                        soup_checked(body)
                    except CorpusError as exc:
                        if "blocked" in exc.code:
                            self.blocked[api] = True
                        raise
                fetched = Fetched(body, final_url, datetime.now(timezone.utc))
                self._write_cache(key, fetched)
                return fetched
            except CorpusError as exc:
                if exc.retryable and attempt < 2 and exc.code == "network_failure":
                    self.sleep(min(30, 2 ** (attempt + 1) + random.random()))
                    continue
                raise
        raise CorpusError("retry_budget_exhausted", retryable=True)

    @staticmethod
    def retry_after(value):
        if not value:
            return 0
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                return 0

    def check_robots(self, url):
        if self.robots is None:
            response = self.fetch("https://codeforces.com/robots.txt")
            try:
                text = response.body.decode("utf-8")
            except UnicodeError:
                raise CorpusError("invalid_robots_response") from None
            if "user-agent:" not in text.lower() or "<html" in text.lower():
                raise CorpusError("invalid_robots_response")
            self.robots = RobotFileParser()
            self.robots.parse(text.splitlines())
        if not self.robots.can_fetch(USER_AGENT, url):
            raise CorpusError("robots_disallowed")

    def api(self, method, refresh=False, **params):
        if method not in {"contest.list", "problemset.problems", "blogEntry.view"}:
            raise CorpusError("api_method_not_in_scope")
        url = "https://codeforces.com/api/" + method + "?" + urlencode({"lang": "en", **params})
        return self.fetch(url, api=True, refresh=refresh)
