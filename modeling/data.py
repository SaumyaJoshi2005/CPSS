from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from cpss.export import verify_export
from cpss.core import sha256, identity

from .common import Config, ModelingError, atomic_json, hash_file, read_json

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class Record:
    problem_id: str
    split: Split
    statement: str
    original_tags: tuple[str, ...]
    version_id: str
    clean_sha256: str
    title: str = ""


@dataclass(frozen=True)
class Corpus:
    directory: Path
    manifest_sha256: str
    structural: dict[str, dict[str, Any]]
    split_counts: dict[str, int]

    @classmethod
    def verified(cls, config: Config, force: bool = False) -> "Corpus":
        directory = config.path("export_path")
        manifest_file = directory / "manifest.json"
        expected = config.values["expected_manifest_sha256"]
        if hash_file(manifest_file) != expected:
            raise ModelingError("The export manifest does not match the configured frozen corpus")
        manifest = read_json(manifest_file)
        # Stat signatures avoid repeatedly hashing 18k immutable blobs. Any file
        # size/mtime change forces the existing full verifier to run again.
        paths = [manifest_file, directory / "manifest.sha256"]
        paths += [directory / filename for filename in manifest["files"]]
        paths += [directory / "blobs" / digest for digest in manifest["blobs"]]
        try:
            signature = identity([(str(p.relative_to(directory)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
        except OSError as exc:
            raise ModelingError("A pinned export component is missing") from exc
        receipt_path = config.path("cache_path") / "export-verification.json"
        receipt = read_json(receipt_path) if receipt_path.exists() else {}
        if force or receipt.get("manifest_sha256") != expected or receipt.get("stat_signature") != signature:
            try:
                result = verify_export(directory)  # Reuse authoritative corpus infrastructure.
            except Exception as exc:
                raise ModelingError("Frozen export verification failed") from exc
            if not result.get("verified"):
                raise ModelingError("Existing corpus verification failed")
            atomic_json(receipt_path, {"manifest_sha256": expected, "stat_signature": signature,
                                      "verifier": "cpss.export.verify_export", "result": result})
        structural: dict[str, dict[str, Any]] = {}
        split_counts: Counter[str] = Counter()
        contests: dict[int, set[str]] = defaultdict(set)
        hashes: dict[str, set[str]] = defaultdict(set)
        for row in manifest["records"]:
            pid, split = row["problem_id"], row["split"]
            metadata = row["metadata"]
            if pid in structural or split not in {"train", "validation", "test"}:
                raise ModelingError("Duplicate identity or invalid frozen split")
            match = re.fullmatch(r"codeforces:([1-9][0-9]*):([A-Z][0-9]*)", pid)
            if not match or int(match[1]) != metadata["contest_id"] or match[2] != metadata["problem_index"]:
                raise ModelingError("Canonical identity disagrees with manifest metadata")
            source = metadata.get("corpus", {})
            if source.get("source_split") == "test" and split != "test":
                raise ModelingError("Upstream holdout moved out of test")
            structural[pid] = {"split": split, "version_id": row["statement_version_id"],
                "clean_sha256": row["clean_sha256"], "contest_id": metadata["contest_id"],
                "aliases": source.get("aliases", []), "upstream_split": source.get("source_split"),
                "editorial_link_id": (row.get("editorial_link") or {}).get("_id"),
                "editorial_version_id": (row.get("editorial_link") or {}).get("editorial_version_id")}
            split_counts[split] += 1
            contests[metadata["contest_id"]].add(split)
            hashes[row["clean_sha256"]].add(split)
        if any(len(v) != 1 for v in [*contests.values(), *hashes.values()]):
            raise ModelingError("Contest or exact duplicate crosses frozen splits")
        for pid, info in structural.items():
            for alias in info["aliases"]:
                alias_pid = "codeforces:" + alias.replace("/", ":")
                if alias_pid in structural and structural[alias_pid]["split"] != info["split"]:
                    raise ModelingError("Known aliases cross frozen splits")
        for group in manifest["configuration"].get("duplicate_groups", []):
            if len({structural[p]["split"] for p in group if p in structural}) > 1:
                raise ModelingError("Explicit duplicate group crosses frozen splits")
        if dict(split_counts) != config.values["expected_split_counts"]:
            raise ModelingError("Unexpected split counts")
        return cls(directory, expected, structural, dict(split_counts))

    def records(self, splits: Iterable[Split], *, final_test: bool = False) -> list[Record]:
        selected = set(splits)
        if not selected <= {"train", "validation", "test"} or not selected:
            raise ModelingError("Invalid requested split")
        if "test" in selected and not final_test:
            raise ModelingError("Test content requires an explicit final-evaluation invocation")
        result, ids = [], set()
        with (self.directory / "statement-only.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                pid = row["problem_id"]
                if pid in ids or pid not in self.structural or row["split"] != self.structural[pid]["split"]:
                    raise ModelingError("Statement view identities disagree with frozen manifest")
                ids.add(pid)
                if row["split"] not in selected:
                    continue  # No test targets/text validation or statistics in development.
                features = row.get("input")
                if not isinstance(features, dict) or set(features) != {"statement"}:
                    raise ModelingError("Classifier input must contain only input.statement")
                statement = features["statement"]
                tags = row.get("target", {}).get("original_tags")
                if not isinstance(statement, str) or not statement.strip():
                    raise ModelingError("Empty or malformed statement")
                if not isinstance(tags, list) or any(not isinstance(t, str) or not t.strip() for t in tags):
                    raise ModelingError("Malformed original tag assignments")
                info = self.structural[pid]
                if (row["statement_version_id"] != info["version_id"] or row["clean_sha256"] != info["clean_sha256"]
                        or sha256(statement.encode("utf-8")) != info["clean_sha256"]):
                    raise ModelingError("Statement text/version does not match the pinned clean hash")
                result.append(Record(pid, row["split"], statement, tuple(tags), info["version_id"],
                                     info["clean_sha256"], row["metadata"]["title"]))
        if ids != set(self.structural):
            raise ModelingError("Statement view does not cover all manifest identities")
        return result

    def context_sources(self, problem_id: str) -> dict[str, Any]:
        info = self.structural.get(problem_id)
        if not info or info["split"] != "train":
            raise ModelingError("Context generation/review is restricted to training identities")
        with (self.directory / "editorial-enriched.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["problem_id"] == problem_id:
                    if (row["split"] != "train" or row["statement_version_id"] != info["version_id"]
                            or row["editorial_version_id"] != info["editorial_version_id"]):
                        raise ModelingError("Context evidence identity mismatch")
                    return {"problem_id": problem_id, "source_manifest_sha256": self.manifest_sha256,
                        "statement": {"version_id": info["version_id"], "text": row["input"]["statement"]},
                        "editorial": {"version_id": row["editorial_version_id"], "text": row["input"]["editorial"]}}
        raise ModelingError("No editorial context evidence for this training problem")
