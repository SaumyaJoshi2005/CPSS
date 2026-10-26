from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import Config, ModelingError, ROOT, identity, read_json

DEFAULT_SEARCH_CONFIG = ROOT / "modeling/config/search-v1.json"


@dataclass(frozen=True)
class SearchConfig:
    values: dict[str, Any]
    project_root: Path = ROOT

    @classmethod
    def load(cls, path: Path = DEFAULT_SEARCH_CONFIG, project_root: Path = ROOT) -> "SearchConfig":
        values = read_json(path)
        if values.get("version") != "cpss-search-v1" or values.get("split") != "train":
            raise ModelingError("Search development is restricted to the preserved training split")
        if set(values.get("views", [])) != {"statement", "editorial"}:
            raise ModelingError("Search needs statement and editorial views")
        if not 1 <= values.get("result_limit", 0) <= values.get("candidate_limit", 0) <= 500:
            raise ModelingError("Search result/candidate limits must lie in 1..500")
        if values.get("rrf_k", 0) < 1 or not 0 <= values.get("explicit_tag_boost", -1) <= 1 or not 0 <= values.get("classifier_boost", -1) <= 1:
            raise ModelingError("Invalid fusion/boost parameters")
        bm25 = values.get("bm25", {})
        if bm25.get("k1", 0) <= 0 or not 0 <= bm25.get("b", -1) <= 1:
            raise ModelingError("Invalid BM25 parameters")
        if values.get("maximum_query_characters", 0) < 1:
            raise ModelingError("Query length limit must be positive")
        latency = values.get("latency", {})
        if latency.get("warmups", -1) < 0 or latency.get("repeats", 0) < 1:
            raise ModelingError("Latency needs nonnegative warmups and positive repeats")
        if values.get("evaluation", {}).get("cutoff") != 10 or values["evaluation"].get("relevance_threshold") not in {1, 2}:
            raise ModelingError("Submission evaluation reports @10 with a declared relevance threshold of 1 or 2")
        for key in ["index_path", "evaluation_path"]:
            path_value = (project_root / values[key]).resolve()
            if not path_value.is_relative_to((project_root / "modeling").resolve()):
                raise ModelingError("Generated search artifacts must remain under modeling/")
        return cls(values, project_root)

    def path(self, key: str) -> Path:
        return self.project_root / self.values[key]

    @property
    def fingerprint(self) -> str:
        return identity(self.values)


def corpus_config() -> Config:
    return Config.load()
