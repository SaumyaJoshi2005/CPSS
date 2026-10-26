from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from cpss.core import sha256

from ..common import ModelingError, identity
from ..data import Corpus
from ..labels import Taxonomy
from .configuration import SearchConfig

View = Literal["statement", "editorial"]
CONTEXT_VERSION = "original-evidence-context-v1"


@dataclass(frozen=True)
class Document:
    problem_id: str
    title: str
    rating: int
    original_tags: tuple[str, ...]
    statement: str
    editorial: str
    provenance: dict[str, Any]

    def __post_init__(self):
        if (not re.fullmatch(r"codeforces:[1-9][0-9]*:[A-Z][0-9]*", self.problem_id)
                or type(self.rating) is not int or not 0 <= self.rating <= 5000
                or not self.title.strip() or not self.statement.strip() or not self.editorial.strip()
                or self.provenance.get("split") != "train"):
            raise ModelingError("Search documents require canonical training identities and complete original context")

    @property
    def url(self) -> str:
        _, contest, index = self.problem_id.split(":")
        return f"https://codeforces.com/problemset/problem/{contest}/{index}"

    def text(self, view: View) -> str:
        if view == "statement":
            return self.statement
        if view == "editorial":
            return (self.statement + "\n\nOriginal upstream tags: " + "; ".join(self.original_tags)
                    + "\n\nOriginal editorial:\n" + self.editorial)
        raise ModelingError("Unknown problem-context view")

    def signature(self, view: View) -> dict:
        return {"problem_id": self.problem_id, "view_sha256": sha256(self.text(view).encode()),
                "metadata": identity([self.title, self.rating, self.original_tags]), "provenance": self.provenance}


def load_documents(corpus: Corpus) -> list[Document]:
    """Join only training rows; do not inspect held-out text/labels for search."""
    training = corpus.records(["train"])
    wanted = {r.problem_id: r for r in training}
    result = {}
    with (corpus.directory / "editorial-enriched.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            pid = row["problem_id"]
            if pid not in wanted:
                continue
            r, metadata = wanted[pid], row["metadata"]
            pin = corpus.structural[pid]
            if (pid in result or row["split"] != "train" or row["statement_version_id"] != r.version_id
                    or row["editorial_version_id"] != pin["editorial_version_id"] or row["input"]["statement"] != r.statement
                    or row["target"]["original_tags"] != list(r.original_tags)):
                raise ModelingError("Training problem-context identity/source mismatch")
            editorial = row["input"]["editorial"]
            rating = metadata.get("rating")
            if not isinstance(editorial, str) or not editorial.strip() or type(rating) is not int or rating < 0:
                raise ModelingError("Search context is missing its original editorial or rating")
            result[pid] = Document(pid, r.title, rating, r.original_tags, r.statement, editorial,
                {"source_manifest_sha256": corpus.manifest_sha256, "statement_version_id": r.version_id,
                 "statement_clean_sha256": r.clean_sha256, "editorial_version_id": row["editorial_version_id"],
                 "editorial_sha256": sha256(editorial.encode()), "editorial_link_id": row["editorial_link_id"],
                 "provider": metadata["corpus"]["repository"], "provider_revision": metadata["corpus"]["revision"],
                 "upstream_row_id": metadata["corpus"]["row_id"], "split": "train", "context_version": CONTEXT_VERSION,
                 "annotation_status": "original_evidence_not_newly_verified_annotations"})
    if set(result) != set(wanted):
        raise ModelingError("Search context coverage differs from the complete training split")
    return [result[r.problem_id] for r in training]


@dataclass(frozen=True)
class QueryContext:
    original_text: str
    search_text: str
    explicit_algorithms: tuple[dict[str, str], ...]
    excluded_tags: tuple[str, ...]
    rating_min: int | None
    rating_max: int | None
    inferred_approaches: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    interpretation_method: str = "deterministic_alias_and_filter_parser_v1"


def parse_query(text: str, settings: SearchConfig, taxonomy: Taxonomy,
                rating_min: int | None = None, rating_max: int | None = None) -> QueryContext:
    if not isinstance(text, str) or not text.strip():
        raise ModelingError("Enter a nonempty search description")
    if len(text) > settings.values["maximum_query_characters"]:
        raise ModelingError("Search description exceeds the configured character limit")
    search_text = text.casefold()
    parsed_min, parsed_max = [], []
    def rating_filter(match):
        operator, number = match.group(1), int(match.group(2))
        if operator in {">=", ">"}:
            parsed_min.append(number + (operator == ">"))
        elif operator in {"<=", "<"}:
            parsed_max.append(number - (operator == "<"))
        else:
            parsed_min.append(number); parsed_max.append(number)
        return " "
    search_text = re.sub(r"\brating\s*(>=|<=|>|<|=|:)\s*(\d+)\b", rating_filter, search_text)
    def rating_range(match):
        parsed_min.append(int(match.group(1))); parsed_max.append(int(match.group(2))); return " "
    search_text = re.sub(r"\brating\s+(?:between\s+)?(\d+)\s*(?:-|to|and)\s*(\d+)\b", rating_range, search_text)
    if re.search(r"\brating\b\s*(?:[<>=:]|between\b|\d)", search_text):
        raise ModelingError("Use rating >= 1200, rating <= 2000, or rating 1200-2000")
    for value in [rating_min, rating_max, *parsed_min, *parsed_max]:
        if value is not None and (type(value) is not int or not 0 <= value <= 5000):
            raise ModelingError("Rating bounds must be whole numbers between 0 and 5000")
    low = max([v for v in [rating_min, *parsed_min] if v is not None], default=None)
    high = min([v for v in [rating_max, *parsed_max] if v is not None], default=None)
    if low is not None and high is not None and low > high:
        raise ModelingError("Rating constraints conflict: minimum exceeds maximum")
    aliases = {name: name for name in taxonomy.spec["tags"]}
    aliases.update(taxonomy.spec["aliases"])
    aliases.update(settings.values["query_aliases"])
    fine = settings.values["query_fine_names"]
    names = sorted(set(aliases) | set(fine), key=lambda n: (-len(n), n))
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)", re.I)
    explicit, exclusions, expansions = [], set(), []
    for match in pattern.finditer(search_text):
        name = match.group().casefold()
        tag = fine[name]["tag_proxy"] if name in fine else aliases[name]
        if re.search(r"(?:\bwithout|\bexclude|\bexcluding|\bnot|\bno)\s+(?:using\s+)?$", search_text[max(0, match.start() - 30):match.start()]):
            if name in fine or (name not in taxonomy.spec["tags"] and name not in taxonomy.spec["aliases"] and tag != "dsu"):
                raise ModelingError("Cannot enforce absence of a fine-grained algorithm from broad original tags; supported exclusions name original tags or their exact aliases")
            exclusions.add(tag)
            continue
        algorithm_id = fine[name]["algorithm_id"] if name in fine else taxonomy.spec["tags"][tag]["algorithm_id"]
        explicit.append({"named_text": match.group(), "algorithm_id": algorithm_id, "original_tag_proxy": tag,
                         "mapping_semantics": "Original-tag proxy; not a verified fine-grained problem annotation"})
        expansions.append(tag)
    # Remove excluded phrases from the ranking text; enforce their observed-tag
    # exclusion separately. Negation must not accidentally promote a technique.
    search_text = re.sub(r"\b(?:without|exclude|excluding|not|no)\s+(?:using\s+)?(" + "|".join(re.escape(n) for n in names) + r")(?!\w)", " ", search_text)
    unique = {x["algorithm_id"]: x for x in explicit}
    return QueryContext(text, " ".join((search_text + " " + " ".join(dict.fromkeys(expansions))).split()),
        tuple(unique.values()), tuple(sorted(exclusions)), low, high, (),
        ("No implicit algorithm assignment is made for unnamed descriptions.",) if not unique else ())
