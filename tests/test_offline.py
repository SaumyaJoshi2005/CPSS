import copy
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from cpss.core import CorpusError, Settings, canonical, identity, safe_error, safe_url, sha256, utcnow
from cpss.export import assign_splits
from cpss.http import Limiter, Transport
from cpss.parsers import editorial, editorial_candidates, statement
from cpss.pipeline import manifest_entries, read_local, select_metadata
from cpss.schemas import validate


def test_statement_preserves_math_examples_unicode(fixtures):
    parsed = statement((fixtures / "statements/99001/A.html").read_bytes(), "A")
    assert "$$$1 \\le a,b \\le 10^9$$$" in parsed.clean_text
    assert "α ≤ β" in parsed.clean_text
    assert "2 3" in parsed.clean_text and "5" in parsed.clean_text
    assert "Tags must not leak" not in parsed.clean_text
    assert parsed.clean_text == statement((fixtures / "statements/99001/A.html").read_bytes(), "A").clean_text


@pytest.mark.parametrize("raw", [b"<html>not a statement</html>",
    b"<title>Enter - Codeforces</title>", b"<html><title>Just a moment...</title></html>",
    b"<div class='problem-statement'>fake</div><script>cf-chl-test</script>"])
def test_error_responses_rejected(raw):
    with pytest.raises(CorpusError):
        statement(raw)


def test_blocked_and_identity_mismatch(fixtures):
    with pytest.raises(CorpusError, match="blocked"):
        statement((fixtures / "blocked.html").read_bytes())
    with pytest.raises(CorpusError, match="identity"):
        statement((fixtures / "statements/99001/A.html").read_bytes(), "B")


def test_editorial_sections_and_indentation(fixtures):
    result = editorial((fixtures / "editorials/799001.json").read_bytes(), ["codeforces:99001:A", "codeforces:99001:B"])
    sections = {s["key"]: s["text"] for s in result.sections}
    assert "Dijkstra" not in sections["codeforces:99001:A"]
    assert "Dijkstra" in sections["codeforces:99001:B"]
    assert "    print(a + b)" in result.codes["codeforces:99001:A"][0]
    assert result.mappings["codeforces:99001:A"] == ("explicit_link", 1.0, "approved")


def test_wrong_editorial_and_heading_mapping(fixtures):
    raw = (fixtures / "editorials/799001.json").read_bytes()
    result = editorial(raw, ["codeforces:42:A"])
    assert not result.mappings
    raw = json.dumps({"status": "OK", "result": {"content": "<h2>A. Title</h2><p>Ambiguous technique.</p>"}}).encode()
    result = editorial(raw, ["codeforces:42:A"])
    assert result.mappings["codeforces:42:A"][2] == "proposed"
    assert not editorial(raw, ["codeforces:42:A", "codeforces:43:A"]).mappings


def test_editorial_candidate_host_validation():
    raw = b"<a href='/blog/entry/1'>Tutorial</a><a href='https://evil.example/blog/entry/2'>Editorial</a>"
    assert editorial_candidates(raw, "https://codeforces.com/problemset/problem/1/A") == [1]


def test_selection_dates_types_and_missing_rating(fixtures):
    contests = json.loads((fixtures / "contests.json").read_bytes())["result"]
    problems = json.loads((fixtures / "problems.json").read_bytes())["result"]["problems"]
    contests[0]["id"], problems[0]["contestId"], problems[1]["contestId"] = 99999, 99999, 99999
    selected = select_metadata(contests, problems, 4)
    assert selected[0][0]["contestId"] == 99002  # date, not the larger ID
    assert "rating" not in selected[-1][0]
    contests[1]["phase"] = "CODING"
    assert len(select_metadata(contests, problems, 10)) == 2
    problems[0]["type"] = "QUESTION"
    assert len(select_metadata(contests, problems, 10)) == 1


def test_global_spacing():
    now, starts = [0.0], []
    def sleep(delay):
        now[0] += delay
    limiter = Limiter(2.2, clock=lambda: now[0], sleep=sleep)
    for _ in range(4):
        limiter.wait()
        starts.append(now[0])
    assert all(b - a >= 2.199999 for a, b in zip(starts, starts[1:]))


def test_api_failed_cache_redirect_and_retries(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"status": "FAILED", "comment": "secret must not be logged"})
    transport = Transport(Settings(cache_dir=tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None)
    with pytest.raises(CorpusError, match="api_failed"):
        transport.api("contest.list")
    assert not list((tmp_path / "http").glob("*.body"))
    transport.close()
    def redirect(request):
        return httpx.Response(302, headers={"location": "https://evil.example/steal"})
    transport = Transport(Settings(cache_dir=tmp_path), client=httpx.Client(transport=httpx.MockTransport(redirect)), sleep=lambda _: None)
    with pytest.raises(CorpusError, match="url_not_allowed"):
        transport.api("contest.list")
    transport.close()


def test_retry_after_and_cache(tmp_path):
    calls, sleeps = [], []
    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "7"}) if len(calls) == 1 else httpx.Response(200, json={"status": "OK", "result": []})
    transport = Transport(Settings(cache_dir=tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleeps.append)
    first = transport.api("contest.list")
    second = transport.api("contest.list")
    assert 7 in sleeps and len(calls) == 2 and second.cached and first.body == second.body
    transport.close()


def test_robots_and_opt_in(tmp_path):
    def handler(request):
        return httpx.Response(200, text="User-agent: *\nDisallow: /contest\n")
    transport = Transport(Settings(cache_dir=tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None)
    with pytest.raises(CorpusError, match="opt_in"):
        transport.fetch("https://codeforces.com/problemset/problem/1/A")
    transport.allow_pages = True
    with pytest.raises(CorpusError, match="robots_disallowed"):
        transport.fetch("https://codeforces.com/contest/1")
    transport.close()


def test_secret_sanitization_and_urls(tmp_path):
    assert safe_error(ValueError("mongodb://user:password@host")) == "ValueError"
    for url in ("http://codeforces.com", "https://codeforces.com.evil.example", "https://user:pw@codeforces.com"):
        with pytest.raises(CorpusError):
            safe_url(url)
    with pytest.raises(CorpusError, match="credential_file_forbidden"):
        read_local(tmp_path / "token.txt")


def test_schema_types_size_and_enum():
    doc = {"_id": "0" * 64, "schema_version": 1, "byte_length": 3, "storage": "inline", "body": b"abc", "created_at": utcnow()}
    validate("blobs", doc)
    for change in ({"storage": "unknown"}, {"byte_length": True}, {"_id": "BAD"}, {"created_at": "today"}):
        with pytest.raises(CorpusError):
            validate("blobs", {**doc, **change})
    with pytest.raises(CorpusError, match="oversize"):
        validate("blobs", {**doc, "body": b"x" * (16 * 1024 * 1024)})


def test_split_groups_and_strict_chronology():
    problems = []
    for i in range(12):
        problems.append({"_id": f"p{i}", "contest_id": i // 2, "title": f"Title {i}",
                         "contest": {"started_at": datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i // 2)},
                         "statement": {"clean_sha256": sha256(str(i).encode())}})
    mapping = assign_splits(problems, [["p0", "p6"]], (0.5, 0.25, 0.25))
    assert mapping["p0"] == mapping["p6"]
    for contest in {p["contest_id"] for p in problems}:
        assert len({mapping[p["_id"]] for p in problems if p["contest_id"] == contest}) == 1
    ranks = {"train": 0, "validation": 1, "test": 2}
    assert [ranks[mapping[p["_id"]]] for p in problems] == sorted(ranks[mapping[p["_id"]]] for p in problems)


def test_mathjax_and_sample_line_boundaries(fixtures):
    raw = (fixtures / "statements/99001/A.html").read_bytes()
    raw = raw.replace(b"$$$a+b$$$", b'<script type="math/tex">a+b</script>')
    raw = raw.replace(b"2 3\n", b'<div class="test-example-line">2 3</div><div class="test-example-line">4 5</div>')
    text = statement(raw, "A").clean_text
    assert "$a+b$" in text and "2 3\n4 5" in text


def test_editorial_subheadings_do_not_lose_code():
    raw = json.dumps({"status": "OK", "result": {"content":
        "<p><a href='/contest/42/problem/A'>A. Example</a></p><h3>Solution</h3>"
        "<p>Use a queue.</p><pre><code class='language-python'>if True:\n    print(1)</code></pre>"
        "<h2><a href='/contest/43/problem/A'>Other contest</a></h2><p>Do not attach this.</p>"}}).encode()
    result = editorial(raw, ["codeforces:42:A"])
    section = next(s for s in result.sections if s["key"] == "codeforces:42:A")
    assert "Use a queue" in section["text"] and "Do not attach" not in section["text"]
    assert result.code_languages["codeforces:42:A"] == ["python"]


def test_curated_mapping_cannot_override_conflicting_explicit_link():
    raw = json.dumps({"status": "OK", "result": {"content":
        "<h2 id='a'><a href='/contest/42/problem/A'>A. Example</a></h2><p>Explanation.</p>"}}).encode()
    with pytest.raises(CorpusError, match="curated_explicit_link_conflict"):
        editorial(raw, ["codeforces:42:A", "codeforces:42:B"], {"a": "codeforces:42:B"})


def test_denied_origin_stops_requests(tmp_path):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(403, text="Forbidden")
    transport = Transport(Settings(cache_dir=tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None)
    with pytest.raises(CorpusError, match="http_403_blocked"):
        transport.api("contest.list")
    with pytest.raises(CorpusError, match="origin_blocked"):
        transport.api("problemset.problems")
    assert len(calls) == 1
    transport.close()


def test_manifest_rejects_wrong_contest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": [{"contest_id": 1, "blog_id": 2, "problem_ids": ["codeforces:3:A"]}]}))
    with pytest.raises(CorpusError, match="manifest_contest_problem_mismatch"):
        manifest_entries(path)


def test_shared_editorial_headers_do_not_leak_into_previous_problem():
    content = ("<p><a href='/contest/41/problem/A'>First</a></p><p>Only first.</p>"
               "<p><a href='/contest/41/problem/B'>Shared easy</a> / <a href='/contest/42/problem/A'>Shared hard</a></p>"
               "<p>Shared explanation.</p><pre>if True:\n    print(2)</pre>"
               "<p>I also used <a href='/contest/40/problem/A'>this older problem</a> as inspiration.</p>"
               "<p><a href='/contest/42/problem/B'>Last</a></p><p>Only last.</p>")
    raw = json.dumps({"status": "OK", "result": {"content": content}}).encode()
    parsed = editorial(raw, ["codeforces:41:A", "codeforces:41:B", "codeforces:42:A", "codeforces:42:B"])
    sections = {s["key"]: s["text"] for s in parsed.sections}
    assert "Shared explanation" not in sections["codeforces:41:A"]
    assert sections["codeforces:41:B"] == sections["codeforces:42:A"]
    assert "Only last" not in sections["codeforces:41:B"]
    assert parsed.codes["codeforces:41:B"] == parsed.codes["codeforces:42:A"]
    assert all(m[2] == "approved" for m in parsed.mappings.values())


def test_unknown_shared_problem_is_boundary_not_previous_explanation():
    content = ("<p><a href='/contest/41/problem/A'>First</a></p><p>Only first.</p>"
               "<p><a href='/contest/41/problem/B'>B</a> / <a href='/contest/42/problem/A'>Other</a></p>"
               "<p>Do not attach to first.</p>")
    raw = json.dumps({"status": "OK", "result": {"content": content}}).encode()
    parsed = editorial(raw, ["codeforces:41:A"])
    assert "Do not attach" not in next(s["text"] for s in parsed.sections if s["key"] == "codeforces:41:A")


def test_unfinished_editorial_section_is_not_approved():
    content = "<p><a href='/contest/41/problem/A'>First</a></p><p>Tutorial is loading...</p>"
    raw = json.dumps({"status": "OK", "result": {"content": content}}).encode()
    parsed = editorial(raw, ["codeforces:41:A"])
    assert not parsed.mappings and "editorial_section_unavailable" in parsed.warnings


def test_shared_editorial_keeps_selected_canonical_alias():
    # problemset.problems can list only one of the two division aliases.
    content = ("<p><a href='/contest/41/problem/B'>Div 2</a> / <a href='/contest/42/problem/A'>Div 1</a></p>"
               "<p>One explanation explicitly covers both aliases.</p>")
    raw = json.dumps({"status": "OK", "result": {"content": content}}).encode()
    parsed = editorial(raw, ["codeforces:42:A"])
    assert parsed.mappings == {"codeforces:42:A": ("explicit_link", 1.0, "approved")}
    assert [s["key"] for s in parsed.sections] == ["codeforces:42:A"]
