import json
from dataclasses import replace

import pytest
from pymongo.errors import DuplicateKeyError, WriteError

from cpss import parsers
from cpss.core import CorpusError, sha256, utcnow
from cpss.export import export_corpus, verify_export
from cpss.pipeline import Pipeline

pytestmark = pytest.mark.mongo


def seed(repo, fixtures):
    pipeline = Pipeline(repo)
    run = pipeline.discover(target=4, candidate_limit=4, catalog_dir=fixtures)
    result = pipeline.run(run["run_id"], limit=4, import_dir=fixtures, manifest=fixtures / "editorial-manifest.json")
    assert not result["errors"]
    assert result["counts"]["statement_ready"] == 4
    assert result["counts"]["editorial_ready"] == 4
    return run["run_id"]


def test_real_validators_indexes_and_duplicate_import(repo, fixtures):
    seed(repo, fixtures)
    before = {c: repo.db[c].count_documents({}) for c in ("problems", "document_versions", "blobs", "editorial_links")}
    seed(repo, fixtures)
    assert before == {c: repo.db[c].count_documents({}) for c in before}
    assert "rating" not in repo.db.problems.find_one({"_id": "codeforces:99001:B"})
    malformed = repo.db.problems.find_one()
    malformed["_id"], malformed["rating"] = "bad", "not an integer"
    with pytest.raises(WriteError) as caught:
        repo.db.problems.insert_one(malformed)
    assert caught.value.code == 121
    duplicate = repo.db.problems.find_one()
    duplicate["_id"] = "codeforces:99999:A"
    with pytest.raises(DuplicateKeyError):
        repo.db.problems.insert_one(duplicate)
    assert repo.db.problems.index_information()["broad_rating"]["key"][0][0] == "approved_labels.broad"
    assert not repo.reconcile()["issues"]


def test_versions_processor_changes_and_stale_guards(repo, fixtures):
    seed(repo, fixtures)
    sid = "cf:statement:99001:A:en"
    raw = (fixtures / "statements/99001/A.html").read_bytes()
    provenance = {"url": "https://codeforces.com/problemset/problem/99001/A", "retrieved_at": utcnow(), "method": "import", "status_code": 200}
    older, newer = repo.observe(sid), repo.observe(sid)
    changed = raw.replace(b"Print the sum.", b"Print the sum as an integer.")
    newvid, _ = repo.ingest(sid, newer, changed, parsers.statement(changed, "A"), provenance)
    oldvid, published = repo.ingest(sid, older, raw, parsers.statement(raw, "A"), provenance)
    assert not published and repo.db.problems.find_one({"_id": "codeforces:99001:A"})["statement"]["version_id"] == newvid
    equivalent = changed.replace(b"<nav>", b"<nav class='new-menu'>")
    vid2, _ = repo.ingest(sid, repo.observe(sid), equivalent, parsers.statement(equivalent, "A"), provenance)
    assert newvid != vid2
    assert repo.db.document_versions.find_one({"_id": newvid})["clean_sha256"] == repo.db.document_versions.find_one({"_id": vid2})["clean_sha256"]
    vid3, _ = repo.ingest(sid, repo.observe(sid), equivalent, parsers.statement(equivalent, "A"), provenance, config={"new_parser_option": True})
    assert vid3 != vid2


@pytest.mark.parametrize("stage", ["blob", "version", "source"])
def test_crash_recovery(repo, fixtures, stage):
    seed(repo, fixtures)
    sid = "cf:statement:99001:A:en"
    raw = (fixtures / "statements/99001/A.html").read_bytes().replace(b"Print the sum.", b"Print the resulting total.")
    seq = repo.observe(sid)
    provenance = {"url": "https://codeforces.com/problemset/problem/99001/A", "retrieved_at": utcnow(), "method": "import", "status_code": 200}
    with pytest.raises(CorpusError, match="simulated_crash"):
        repo.ingest(sid, seq, raw, parsers.statement(raw, "A"), provenance, crash_after=stage)
    repo.ingest(sid, seq, raw, parsers.statement(raw, "A"), provenance)
    assert not repo.reconcile()["issues"]
    assert "resulting total" in repo.db.problems.find_one({"_id": "codeforces:99001:A"})["statement"]["clean_text"]


def test_reconcile_repairs_projection_and_interrupted_job(repo, fixtures):
    run = seed(repo, fixtures)
    repo.db.problems.update_one({"_id": "codeforces:99001:A"}, {"$unset": {"statement": ""}, "$set": {"quality.statement": "missing"}})
    repo.db.ingestion_jobs.update_one({"run_id": run}, {"$set": {"status": "running"}})
    report = repo.reconcile()
    assert {i["code"] for i in report["issues"]} == {"stale_statement_projection", "interrupted_job"}
    repo.reconcile(repair=True)
    assert not repo.reconcile()["issues"]


def test_gridfs_routing_and_content_verification(repo):
    repo.settings = replace(repo.settings, gridfs_threshold=16)
    small = repo.put_blob(b"short")
    large = repo.put_blob(b"a" * 100)
    assert repo.db.blobs.find_one({"_id": small})["storage"] == "inline"
    assert repo.db.blobs.find_one({"_id": large})["storage"] == "gridfs"
    assert repo.get_blob(large) == b"a" * 100
    assert repo.put_blob(b"a" * 100) == large


def test_reviewed_labels_and_good_content_survive_failures(repo, fixtures):
    seed(repo, fixtures)
    repo.put_taxonomy([{"_id": "v1:math", "schema_version": 1, "taxonomy_version": "v1", "display_name": "Math",
                        "kind": "broad", "parent_ids": [], "aliases": [], "active": True}])
    problem = repo.db.problems.find_one({"_id": "codeforces:99001:A"})
    version = repo.db.document_versions.find_one({"_id": problem["statement"]["version_id"]})
    record = {"problem_id": problem["_id"], "label_id": "v1:math", "taxonomy_version": "v1", "origin": "human",
              "producer_version": "review-1", "evidence": {"version_id": version["_id"], "locator": "description:1", "span": "compute"},
              "confidence_policy": "human review; no model probability", "status": "approved", "reviewer": "fixture-reviewer"}
    aid = repo.annotate(record)
    assert repo.annotate(record) == aid
    sid = problem["statement"]["source_document_id"]
    repo.failure(sid, repo.observe(sid), "blocked_or_login_html")
    seed(repo, fixtures)
    refreshed = repo.db.problems.find_one({"_id": problem["_id"]})
    assert refreshed["statement"]["version_id"] == problem["statement"]["version_id"]
    assert refreshed["approved_labels"]["annotation_ids"] == [aid]
    assert repo.db.annotations.count_documents({}) == 1


def test_taxonomy_rejects_cycle_and_wrong_editorial_link(repo, fixtures):
    seed(repo, fixtures)
    records = [{"_id": name, "schema_version": 1, "taxonomy_version": "v1", "display_name": name,
                "kind": "broad", "parent_ids": [other], "aliases": [], "active": True} for name, other in (("a", "b"), ("b", "a"))]
    with pytest.raises(CorpusError, match="taxonomy_cycle"):
        repo.put_taxonomy(records)
    editorial = repo.db.source_documents.find_one({"_id": "cf:editorial:799001:en"})
    with pytest.raises(CorpusError, match="editorial_section_mismatch"):
        repo.link_editorial("codeforces:99001:A", editorial["current_version_id"], "codeforces:99001:B", ("explicit_link", 1.0, "approved"))


def test_export_pins_hashes_labels_and_prevents_input_leakage(repo, fixtures, tmp_path):
    run = seed(repo, fixtures)
    output = tmp_path / "snapshot"
    report = export_corpus(repo, output, run)
    assert report["counts"]["problems"] == 4
    assert verify_export(output)["verified"]
    statements = [json.loads(line) for line in (output / "statement-only.jsonl").read_text("utf-8").splitlines()]
    assert all(set(row["input"]) == {"statement"} for row in statements)
    assert all("editorial" not in row["input"] for row in statements)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert len(manifest["records"]) == 4 and all(p["editorial_link"] for p in manifest["records"])
    with pytest.raises(CorpusError, match="destination_exists"):
        export_corpus(repo, output, run)
    (output / "statement-only.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(CorpusError, match="file_mismatch"):
        verify_export(output)


def test_real_server_rejects_readiness_confidence_and_hash_errors(repo, fixtures):
    seed(repo, fixtures)
    problem = repo.db.problems.find_one()
    with pytest.raises(WriteError):
        repo.db.problems.update_one({"_id": problem["_id"]}, {"$unset": {"statement": ""}})
    link = repo.db.editorial_links.find_one()
    with pytest.raises(WriteError):
        repo.db.editorial_links.update_one({"_id": link["_id"]}, {"$set": {"confidence": 1.1}})
    version = repo.db.document_versions.find_one()
    with pytest.raises(WriteError):
        repo.db.document_versions.update_one({"_id": version["_id"]}, {"$set": {"raw_sha256": "NOT-A-HASH"}})


def test_incomplete_gridfs_upload_recovers_from_exact_bytes(repo):
    from bson import Binary
    repo.settings = replace(repo.settings, gridfs_threshold=16)
    body = b"payload" * 100
    digest = sha256(body)
    repo.db.fs.chunks.insert_one({"files_id": digest, "n": 0, "data": Binary(body[:10])})
    assert any(i["code"] == "incomplete_gridfs_upload" for i in repo.reconcile()["issues"])
    assert repo.put_blob(body) == digest
    assert repo.get_blob(digest) == body
    assert not repo.reconcile()["issues"]


def test_reprocess_reuses_unchanged_versions_and_pins_code(repo, fixtures, tmp_path):
    run = seed(repo, fixtures)
    before = repo.db.document_versions.count_documents({})
    assert Pipeline(repo).reprocess()["reprocessed_sources"] == 7
    assert repo.db.document_versions.count_documents({}) == before
    output = tmp_path / "with-code"
    export_corpus(repo, output, run)
    rows = [json.loads(line) for line in (output / "editorial-enriched.jsonl").read_text("utf-8").splitlines()]
    row = next(row for row in rows if row["problem_id"] == "codeforces:99001:A")
    assert len(row["solutions"]) == 1 and "    print(a + b)" in row["solutions"][0]["code"]
    assert verify_export(output)["verified"]


def test_resume_does_not_retry_blocked_jobs(repo, fixtures):
    run = Pipeline(repo).discover(target=4, candidate_limit=4, catalog_dir=fixtures)
    repo.db.ingestion_jobs.update_many({"run_id": run["run_id"]}, {"$set": {"status": "blocked", "last_error": "http_403_blocked"}})
    result = Pipeline(repo).run(run["run_id"], limit=4, import_dir=fixtures)
    assert result["processed_jobs"] == 0 and result["counts"]["statement_ready"] == 0


def test_retry_replays_checkpoint_bytes_and_sequence(repo, fixtures):
    run = Pipeline(repo).discover(target=4, candidate_limit=4, catalog_dir=fixtures)
    job = repo.db.ingestion_jobs.find_one({"run_id": run["run_id"], "entity_id": "codeforces:99001:A"})
    pipeline = Pipeline(repo)
    pipeline.process_statement(job, fixtures)
    replay_job = repo.db.ingestion_jobs.find_one({"_id": job["_id"]})
    source = repo.db.source_documents.find_one({"_id": replay_job["checkpoint"]["source_document_id"]})
    original = source["current_version_id"]
    # No local file or live transport is needed after bytes are checkpointed.
    pipeline.process_statement(replay_job, import_dir=fixtures / "does-not-exist")
    current = repo.db.source_documents.find_one({"_id": source["_id"]})
    assert current["current_version_id"] == original
    assert current["published_seq"] == source["published_seq"]


def test_editorials_can_be_collected_with_all_statements_blocked(repo, fixtures):
    from cpss.pipeline import imported
    class FixtureAPI:
        def __init__(self):
            self.calls = []
        def api(self, method, **kwargs):
            self.calls.append(kwargs["blogEntryId"])
            blog_id = kwargs["blogEntryId"]
            return imported(fixtures / "editorials" / f"{blog_id}.json", f"https://codeforces.com/api/blogEntry.view?blogEntryId={blog_id}")
    run = Pipeline(repo).discover(target=4, candidate_limit=4, catalog_dir=fixtures)
    repo.db.ingestion_jobs.update_many({"run_id": run["run_id"]}, {"$set": {"status": "blocked", "last_error": "http_403_blocked"}})
    api = FixtureAPI()
    pipeline = Pipeline(repo, api)
    report = pipeline.collect_editorials(run["run_id"], fixtures / "editorial-manifest.json")
    assert not report["errors"] and report["processed_editorials"] == 2
    assert report["counts"]["editorial_sections_ready"] == 4
    assert report["counts"]["statement_ready"] == report["counts"]["editorial_ready"] == 0
    assert len(api.calls) == 2
    repeated = pipeline.collect_editorials(run["run_id"], fixtures / "editorial-manifest.json")
    assert repeated["processed_editorials"] == 0 and len(api.calls) == 2
    assert repo.db.ingestion_jobs.count_documents({"run_id": run["run_id"], "stage": "editorial", "status": "succeeded"}) == 2
    assert not repo.reconcile()["issues"]
