from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cpss.core import sha256
from modeling.common import Config, DEFAULT_CONFIG, read_json
from modeling.data import Record
from modeling.labels import Taxonomy


@pytest.fixture
def small_config(tmp_path):
    values = copy.deepcopy(read_json(DEFAULT_CONFIG))
    values["taxonomy_path"] = str(Path(__file__).parents[1] / "config" / "taxonomy-v1.json")
    values["cache_path"] = str(tmp_path / "cache")
    values["artifacts_path"] = str(tmp_path / "artifacts")
    values["labels"] = {"min_train_positive": 2, "min_train_negative": 2}
    values["features"].update(min_df=1, max_df=1.0, max_features=200, ngram_range=[1, 1])
    values["classifier"].update(C_candidates=[1.0], class_weight_candidates=[None])
    values["thresholds"].update(grid=[0.25, 0.5, 0.75], min_validation_positive=1, min_validation_negative=1)
    values["context_pilot"].update(size=6, case_study_minimum=1)
    return Config(values)


@pytest.fixture
def taxonomy(small_config):
    return Taxonomy.load(small_config.path("taxonomy_path"))


def record(index, split="train", label="dsu", suffix=""):
    statement = ({"dsu": "Merge components and determine connectivity between vertices.",
                  "binary search": "Search a sorted sequence for the monotone feasible threshold.",
                  "math": "Calculate the number of integers and their arithmetic sum."}[label] + " " + suffix)
    return Record(f"codeforces:{100 + index}:A", split, statement, (label,),
                  sha256(f"version-{index}".encode()), sha256(statement.encode()), f"Metadata-only title {index}")


@pytest.fixture
def records():
    train = [record(i, label=["dsu", "binary search", "math"][i % 3]) for i in range(18)]
    validation = [record(i + 30, "validation", ["dsu", "binary search", "math"][i % 3], "validationonlytoken") for i in range(6)]
    return train, validation
