import copy
import json
from datetime import datetime, timezone

import pytest

from cpss.core import CorpusError, canonical, sha256
from cpss.export import assign_splits, export_corpus, verify_export
from cpss.openr1 import (POLICY, audit_complete, completeness, import_snapshot,
                         statement_parsed, write_json)


def complete_row(index="A"):
    return {"id": f"99001/{index}", "aliases": [], "contest_id": "99001", "contest_name": "Codeforces Round 99991 (fixture)",
            "contest_type": "CF", "contest_start": 1735689600, "contest_start_year": 2025, "index": index,
            "time_limit": 2.0, "memory_limit": 256.0, "title": "Fixture " + index,
            "description": "Given integers $$$a$$$ and $$$b$$$, calculate their sum. Constraints: $$$1 \\le a,b \\le 10^9$$$.",
            "input_format": "Two integers a and b, separated by one space.", "output_format": "Print their sum.",
            "interaction_format": None, "note": "Addition is sufficient.", "examples": [{"input": "2 3\n", "output": "5\n"}],
            "editorial": "Read both integers and add them. The sum is the required answer. The time complexity is O(1).",
            "rating": 800, "tags": ["math"], "input_mode": "stdio", "official_tests_complete": False,
            "executable": True, "testset_size": 20, "generated_tests": 10}


def snapshot(path):
    rows = [complete_row(), complete_row("B"), {**complete_row("C"), "editorial": None}]
    rows[1]["aliases"] = ["99002/A"]
    code = [{"submission_id": str(100 + i), "source": "int main(){return 0;}\n", "og_source": "int main() {\n    return 0;\n}\n",
             "contestId": contest, "problem_index": index, "programmingLanguage": "GNU C++17", "verdict": "OK",
             "problem_id": contest + "/" + index} for i, (contest, index) in enumerate((("99001", "A"), ("99002", "A")))]
    manifest = {"policy": POLICY, "repositories": {"open-r1/codeforces": {"revision": "a" * 40},
                "open-r1/codeforces-submissions": {"revision": "b" * 40}}, "files": []}
    for repo, filename, records in (("open-r1/codeforces", "problems.jsonl", rows),
                                    ("open-r1/codeforces-submissions", "solutions.jsonl", code)):
        body = b"".join(canonical(r) + b"\n" for r in records)
        (path / filename).write_bytes(body)
        manifest["files"].append({"repository": repo, "revision": "a" * 40 if repo.endswith("codeforces") else "b" * 40,
            "file": "data/train-00000.parquet", "split": "train", "projection_file": filename,
            "projection_sha256": sha256(body), "rows": len(records)})
    write_json(path / "snapshot.json", manifest)
    return path


def test_structured_statement_preserves_all_sections():
    row = complete_row()
    assert not completeness(row)
    parsed = statement_parsed(row)
    assert {s["key"] for s in parsed.sections} == {"title", "limits", "description", "input", "output", "example:0:input", "example:0:output", "note"}
    assert "$$$1 \\le a,b \\le 10^9$$$" in parsed.clean_text
    assert "2 3\n" in parsed.clean_text
    assert "O(1)" not in parsed.clean_text and "math" not in parsed.clean_text


@pytest.mark.parametrize("field", ["editorial", "description", "input_format", "output_format", "examples", "rating", "tags", "time_limit", "memory_limit"])
def test_incomplete_rows_are_rejected(field):
    row = complete_row()
    row[field] = None
    assert completeness(row)


def test_placeholder_and_interactive_rejection():
    assert "placeholder_or_challenge" in completeness({**complete_row(), "editorial": "Tutorial is loading..."})
    assert "interactive_problem" in completeness({**complete_row(), "interaction_format": "Ask the judge."})
    assert not completeness({**complete_row(), "contest_name": "Educational Codeforces Round 172", "contest_type": "ICPC"})


@pytest.mark.mongo
def test_import_is_complete_idempotent_and_preserves_alias_code(repo, tmp_path):
    root = snapshot(tmp_path)
    result = import_snapshot(repo, root)
    assert result["counts"]["complete_problems"] == 2
    assert repo.db.problems.count_documents({}) == 2
    assert not repo.db.problems.find_one({"_id": "codeforces:99001:C"})
    assert audit_complete(repo)["ok"]
    before = repo.db.document_versions.count_documents({})
    again = import_snapshot(repo, root)
    assert again["run_id"] == result["run_id"]
    assert repo.db.document_versions.count_documents({}) == before
    problem = repo.db.problems.find_one({"_id": "codeforces:99001:B"})
    assert problem["corpus"]["aliases"] == ["99002/A"]
    source = repo.db.source_documents.find_one({"problem_id": problem["_id"], "kind": "solution"})
    version = repo.db.document_versions.find_one({"_id": source["current_version_id"]})
    assert "    return 0;" in version["clean_text"]
    assert not repo.reconcile()["issues"]


@pytest.mark.mongo
def test_missing_code_and_corrupt_snapshot_never_admitted(repo, tmp_path):
    root = snapshot(tmp_path)
    manifest = json.loads((root / "snapshot.json").read_bytes())
    code_path = root / "solutions.jsonl"
    code_path.write_bytes(b"")
    manifest["files"][1]["projection_sha256"] = sha256(b"")
    write_json(root / "snapshot.json", manifest)
    assert import_snapshot(repo, root)["counts"]["complete_problems"] == 0
    assert repo.db.problems.count_documents({}) == 0
    code_path.write_bytes(b"corrupt")
    with pytest.raises(CorpusError, match="snapshot_integrity_failure"):
        import_snapshot(repo, root)


@pytest.mark.mongo
def test_openr1_export_contains_examples_metadata_code_and_raw_provenance(repo, tmp_path):
    root = snapshot(tmp_path)
    result = import_snapshot(repo, root)
    out = root / "export"
    export_corpus(repo, out, result["run_id"])
    assert verify_export(out)["verified"]
    rows = [json.loads(line) for line in (out / "editorial-enriched.jsonl").read_text("utf-8").splitlines()]
    assert all(r["metadata"]["corpus"]["examples"] and r["solutions"] for r in rows)
    assert all(set(r["input"]) == {"statement", "editorial"} for r in rows)


def test_upstream_holdout_never_becomes_training_data():
    problems = [{"_id": str(i), "contest_id": i, "title": f"Distinct {i}",
                 "contest": {"started_at": datetime(2025, 1, i + 1, tzinfo=timezone.utc)},
                 "statement": {"clean_sha256": sha256(str(i).encode())},
                 "corpus": {"source_split": "test" if i == 3 else "train"}} for i in range(10)]
    result = assign_splits(problems)
    assert all(result[str(i)] == "test" for i in range(3, 10))


def test_upstream_aliases_form_transitive_split_groups():
    problems = [{"_id": f"codeforces:{i + 1}:A", "contest_id": i + 1, "title": f"Distinct {i}",
                 "contest": {"started_at": datetime(2025, 1, i + 1, tzinfo=timezone.utc)},
                 "statement": {"clean_sha256": sha256(str(i).encode())},
                 "corpus": {"source_split": "train", "aliases": [f"{i + 2}/A"] if i < 2 else []}} for i in range(4)]
    result = assign_splits(problems, fractions=(0.25, 0.25, 0.5))
    assert len({result[f"codeforces:{i}:A"] for i in (1, 2, 3)}) == 1


def test_generic_repeated_titles_are_not_duplicate_evidence():
    problems = [{"_id": str(i), "contest_id": i, "title": "Game",
                 "contest": {"started_at": datetime(2025, 1, i + 1, tzinfo=timezone.utc)},
                 "statement": {"clean_sha256": sha256(str(i).encode())}} for i in range(10)]
    result = assign_splits(problems, fractions=(0.6, 0.2, 0.2))
    assert {result[str(i)] for i in range(10)} == {"train", "validation", "test"}


@pytest.mark.mongo
def test_activation_archives_previous_metadata_without_dangling_records(repo, tmp_path, monkeypatch):
    import uuid
    from dataclasses import replace
    from cpss.repository import Repository
    from cpss.openr1 import activate
    monkeypatch.chdir(tmp_path)
    stage_name = "cpss_openr1_stage_test_" + uuid.uuid4().hex
    backup_name = "cpss_backup_test_" + uuid.uuid4().hex
    stage = Repository(replace(repo.settings, database=stage_name))
    try:
        stage.bootstrap()
        import_snapshot(stage, snapshot(tmp_path))
        repo.metadata({"contestId": 99009, "index": "A", "name": "Old metadata only", "tags": []},
                      {"name": "Old fixture", "started_at": datetime(2025, 1, 1, tzinfo=timezone.utc)})
        result = activate(repo, stage_name, backup_name)
        assert result["complete_problems"] == 2
        assert repo.client[backup_name].problems.count_documents({}) == 1
        assert not repo.db.problems.find_one({"_id": "codeforces:99009:A"})
        assert audit_complete(repo)["ok"] and not repo.reconcile()["issues"]
        assert activate(repo, stage_name, backup_name)["status"] == "already_completed"
    finally:
        stage.close()
        repo.client.drop_database(stage_name)
        repo.client.drop_database(backup_name)


@pytest.mark.mongo
def test_activation_refuses_incomplete_staging(repo, tmp_path, monkeypatch):
    import uuid
    from dataclasses import replace
    from cpss.repository import Repository
    from cpss.openr1 import activate
    monkeypatch.chdir(tmp_path)
    stage_name = "cpss_openr1_stage_test_" + uuid.uuid4().hex
    stage = Repository(replace(repo.settings, database=stage_name))
    try:
        stage.bootstrap()
        with pytest.raises(CorpusError, match="completeness_audit_failed"):
            activate(repo, stage_name)
        assert repo.db.list_collection_names()
    finally:
        stage.close()
        repo.client.drop_database(stage_name)
