from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag

from .core import CorpusError, identity, problem_id

EXTRACTOR_VERSION = "cf-html-3"
CLEANER_VERSION = "whitespace-1"
BLOCKS = {"div", "p", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote"}
PROBLEM_URL = re.compile(r"^/(?:problemset/problem/(\d+)/(\w+)|contest/(\d+)/problem/(\w+))/?$")


@dataclass
class Parsed:
    sections: list[dict]
    warnings: list[str] = field(default_factory=list)
    mappings: dict[str, tuple[str, float, str]] = field(default_factory=dict)
    codes: dict[str, list[str]] = field(default_factory=dict)
    code_languages: dict[str, list[str | None]] = field(default_factory=dict)

    @property
    def clean_text(self):
        return "\n\n".join(s["text"] for s in self.sections)


def processor_fingerprint(kind: str, config: dict | None = None):
    import bs4
    return identity({"extractor": EXTRACTOR_VERSION, "cleaner": CLEANER_VERSION,
                     "parser": "html.parser", "beautifulsoup": bs4.__version__,
                     "kind": kind, "language": "en", "config": config or {}})


def soup_checked(raw: bytes):
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeError:
        raise CorpusError("invalid_utf8") from None
    soup = BeautifulSoup(decoded, "html.parser")
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    if (any(s in decoded.lower() for s in ("cf-chl-", "challenge-platform", "ddos-guard", "checking your browser"))
            or any(s in title for s in ("just a moment", "access denied", "sign in", "login", "enter - codeforces", "error"))
            or soup.select_one("form[action*='/enter'], #challenge-form")):
        raise CorpusError("blocked_or_login_html")
    return soup


def render(node) -> str:
    """HTML text with block boundaries; do not collapse math or preformatted whitespace."""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag):
        return ""
    if node.name == "script" and str(node.get("type", "")).startswith("math/tex"):
        delimiter = "$$" if "mode=display" in node.get("type", "") else "$"
        return delimiter + node.get_text() + delimiter
    if node.name in {"script", "style", "noscript"}:
        return ""
    if node.name == "br":
        return "\n"
    if node.name == "img":
        return node.get("alt", "")
    if node.name == "math":
        annotation = node.find("annotation", attrs={"encoding": "application/x-tex"})
        if annotation:
            return "$" + annotation.get_text() + "$"
    if node.name == "pre":
        copy = BeautifulSoup(str(node), "html.parser").pre
        for br in copy.find_all("br"):
            br.replace_with("\n")
        for line in copy.select(".test-example-line"):
            line.insert_after("\n")
        return "\n" + copy.get_text().replace("\r\n", "\n").strip("\n") + "\n"
    result = "".join(render(child) for child in node.children)
    if node.name in {"td", "th"}:
        return result + "\t"
    return "\n" + result + "\n" if node.name in BLOCKS else result


def clean(node) -> str:
    # Strip outer blank lines only; indentation inside code/examples is meaningful.
    text = render(node).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    result, blank = [], False
    for line in lines:
        if not line.strip():
            if not blank:
                result.append("")
            blank = True
        else:
            result.append(line.rstrip())
            blank = False
    return "\n".join(result).strip("\n ")


def statement(raw: bytes, expected_index: str | None = None) -> Parsed:
    soup = soup_checked(raw)
    roots = soup.select(".problem-statement")
    if len(roots) != 1:
        raise CorpusError("missing_or_ambiguous_statement")
    root = roots[0]
    title = root.select_one(".header .title")
    if title is None or not title.get_text(strip=True):
        raise CorpusError("missing_statement_title")
    if expected_index and not re.match(r"^" + re.escape(expected_index) + r"[.\s]", title.get_text(strip=True)):
        raise CorpusError("statement_identity_mismatch")
    if not root.select_one(".input-specification") or not root.select_one(".output-specification"):
        raise CorpusError("incomplete_statement")
    warnings = []
    if any(not img.get("alt") for img in root.find_all("img")):
        warnings.append("image_requires_raw_reference")
    sections = []
    counts = {}
    for child in root.children:
        if not isinstance(child, Tag) or child.name in {"script", "style"}:
            continue
        value = clean(child)
        if not value:
            continue
        classes = child.get("class", [])
        key = next((c for c in classes if c in {"header", "input-specification", "output-specification",
                                              "sample-tests", "note"}), "description")
        counts[key] = counts.get(key, 0) + 1
        sections.append({"key": f"{key}:{counts[key]}", "text": value})
    if len("".join(s["text"] for s in sections)) < 80:
        raise CorpusError("statement_too_short")
    return Parsed(sections, warnings)


def editorial_candidates(raw: bytes, base_url: str) -> list[int]:
    soup = soup_checked(raw)
    ids = set()
    for a in soup.find_all("a", href=True):
        if not re.search(r"tutorial|editorial", a.get_text(" ", strip=True), re.I):
            continue
        url = urlsplit(urljoin(base_url, a["href"]))
        match = re.fullmatch(r"/blog/entry/(\d+)/?", url.path)
        if url.scheme == "https" and url.hostname == "codeforces.com" and not url.username and match:
            ids.add(int(match[1]))
    return sorted(ids)


def editorial(raw: bytes, expected_problems: list[str], curated: dict | None = None) -> Parsed:
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        raise CorpusError("invalid_editorial_api_json") from None
    if payload.get("status") != "OK" or not isinstance(payload.get("result", {}).get("content"), str):
        raise CorpusError("failed_editorial_api_response")
    soup = soup_checked(payload["result"]["content"].encode("utf-8"))
    root = soup.select_one(".ttypography") or soup
    expected = set(expected_problems)
    markers, mappings, shared = {}, {}, {}
    curated = curated or {}
    for heading in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        is_heading = heading.name.startswith("h")
        if not is_heading and not heading.find(["strong", "b"]):
            # A shared Div. 1/Div. 2 title can contain several problem anchors.
            # Ordinary prose cross-references are not section boundaries.
            copy = BeautifulSoup(str(heading), "html.parser")
            anchors = copy.find_all("a", href=True)
            for anchor in anchors:
                anchor.decompose()
            remainder = copy.get_text(" ", strip=True)
            if not anchors or re.search(r"[\w]", remainder):
                continue
        found = set()
        for a in heading.find_all("a", href=True):
            target = urlsplit(urljoin("https://codeforces.com", a["href"]))
            match = PROBLEM_URL.fullmatch(target.path)
            if target.hostname == "codeforces.com" and match:
                found.add(problem_id(int(match[1] or match[3]), match[2] or match[4]))
        selected = found & expected
        if selected:
            key = next(iter(selected)) if len(selected) == 1 else "shared:" + identity(sorted(selected))
            markers[id(heading)] = key
            if len(selected) > 1:
                shared[key] = sorted(selected)
            for pid in selected:
                mappings[pid] = ("explicit_link", 1.0, "approved")
        elif found:
            # A heading linking another problem must never be guessed as a selected problem.
            markers[id(heading)] = "unmapped:" + str(len(markers))
        elif is_heading:
            heading_text = heading.get_text(" ", strip=True)
            match = re.match(r"^([A-Z][0-9]*)[.\s:–-]", heading_text)
            matches = [pid for pid in expected if match and pid.rsplit(":", 1)[1] == match[1]]
            if len(matches) == 1:
                pid = matches[0]
                markers[id(heading)] = pid
                mappings[pid] = ("heading", 0.6, "proposed")
            elif match:
                markers[id(heading)] = "unmapped:" + str(len(markers))
        if heading.get("id") in curated:
            pid = curated[heading["id"]]
            if pid not in expected:
                raise CorpusError("curated_problem_not_selected")
            if found and found != {pid}:
                raise CorpusError("curated_explicit_link_conflict")
            markers[id(heading)] = pid
            mappings[pid] = ("curated", 1.0, "approved")
    pieces, codes, code_languages = {}, {}, {}
    active = "introduction"
    seen = set()

    def walk(node):
        nonlocal active
        if isinstance(node, Tag) and id(node) in markers:
            active = markers[id(node)]
            if active in seen:
                raise CorpusError("duplicate_editorial_section")
            seen.add(active)
            pieces.setdefault(active, []).append(render(node))
            return
        if isinstance(node, Tag) and node.name == "pre":
            code = render(node).strip("\n")
            codes.setdefault(active, []).append(code)
            language = None
            for code_node in [node, *node.find_all("code")]:
                classes = " ".join(code_node.get("class", []))
                language_match = re.search(r"(?:language-|lang-)([\w+#-]+)", classes)
                if language_match:
                    language = language_match[1]
                    break
            code_languages.setdefault(active, []).append(language)
            pieces.setdefault(active, []).append("\n" + code + "\n")
            return
        if isinstance(node, Tag):
            if node.name in {"script", "style", "noscript"}:
                return
            if node.name in BLOCKS:
                pieces.setdefault(active, []).append("\n")
            if node.name in {"br", "img"}:
                pieces.setdefault(active, []).append(render(node))
            else:
                for child in node.children:
                    walk(child)
            if node.name in BLOCKS:
                pieces.setdefault(active, []).append("\n")
        elif isinstance(node, NavigableString):
            pieces.setdefault(active, []).append(render(node))
    walk(root)
    sections = []
    section_keys = set()
    for key, values in pieces.items():
        value = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "".join(values)).strip("\n ")
        if value:
            for section_key in shared.get(key, [key]):
                if section_key in section_keys:
                    raise CorpusError("duplicate_editorial_section")
                section_keys.add(section_key)
                sections.append({"key": section_key, "text": value})
                if key in shared:
                    codes[section_key] = list(codes.get(key, []))
                    code_languages[section_key] = list(code_languages.get(key, []))
                if section_key in mappings and re.search(r"tutorial is loading|editorial (?:is )?coming soon", value, re.I):
                    mappings.pop(section_key)
    if not sections:
        raise CorpusError("empty_editorial")
    warnings = ["editorial_section_unavailable"] if any(
        s["key"] in expected and s["key"] not in mappings for s in sections) else []
    return Parsed(sections, warnings, mappings, codes, code_languages)


def solution(raw: bytes) -> Parsed:
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeError:
        raise CorpusError("invalid_solution_utf8") from None
    if not text.strip() or "<html" in text.lower():
        raise CorpusError("invalid_solution_code")
    return Parsed([{"key": "code", "text": text}])
