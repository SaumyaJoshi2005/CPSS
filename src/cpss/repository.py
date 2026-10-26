from __future__ import annotations

from datetime import timedelta

from bson import BSON, Binary
from gridfs import GridFS
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from .core import CorpusError, Settings, identity, problem_id, sha256, utcnow
from .parsers import CLEANER_VERSION, EXTRACTOR_VERSION, Parsed, processor_fingerprint
from .schemas import INDEXES, MAX_BSON, SCHEMAS, validate


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = MongoClient(settings.uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        self.db = self.client[settings.database]
        self.fs = GridFS(self.db)

    def close(self):
        self.client.close()

    def bootstrap(self):
        """Create missing collections only; changing existing validators is an explicit migration."""
        existing = set(self.db.list_collection_names())
        for name, schema in SCHEMAS.items():
            if name not in existing:
                self.db.create_collection(name, validator={"$jsonSchema": schema},
                                          validationLevel="strict", validationAction="error")
            else:
                options = self.db[name].options()
                if options.get("validator") != {"$jsonSchema": schema}:
                    raise CorpusError("existing_schema_differs_run_migrate_check")
            for keys, label, unique in INDEXES.get(name, []):
                self.db[name].create_index(keys, name=label, unique=unique)
        return {"collections": len(SCHEMAS), "schema_version": 1}

    def migration_check(self, apply=False):
        invalid = []
        for name in SCHEMAS:
            for record in self.db[name].find():
                try:
                    validate(name, record)
                except CorpusError:
                    invalid.append({"collection": name, "id": record["_id"]})
        if apply and invalid:
            raise CorpusError("migration_requires_manual_data_repair")
        if apply:
            for name, schema in SCHEMAS.items():
                if name not in self.db.list_collection_names():
                    self.db.create_collection(name)
                self.db.command("collMod", name, validator={"$jsonSchema": schema},
                                validationLevel="strict", validationAction="error")
            self.bootstrap()
        return {"invalid_documents": invalid, "applied": apply}

    def immutable(self, collection, record):
        validate(collection, record)
        try:
            self.db[collection].insert_one(record)
        except DuplicateKeyError:
            existing = self.db[collection].find_one({"_id": record["_id"]})
            if existing is None:
                raise CorpusError("natural_key_conflict") from None
            # creation/provenance can differ on an equivalent retry; content cannot.
            ignored = {"created_at", "provenance"}
            if any(existing.get(k, {} if k == "processor_config" else None) != v
                   for k, v in record.items() if k not in ignored):
                raise CorpusError("immutable_record_conflict") from None
        return record["_id"]

    def put_blob(self, body: bytes):
        digest = sha256(body)
        if self.db.blobs.find_one({"_id": digest}):
            return digest
        record = {"_id": digest, "schema_version": 1, "byte_length": len(body), "created_at": utcnow()}
        inline = {**record, "storage": "inline", "body": Binary(body)}
        if len(body) <= self.settings.gridfs_threshold and len(BSON.encode(inline)) <= MAX_BSON:
            return self.immutable("blobs", inline)
        # Deterministic GridFS identity; a complete file is checked before publishing its reference.
        if not self.fs.exists(digest):
            # Only one worker is supported. An interrupted upload with this exact
            # content identity can safely be replaced from the bytes now in hand.
            self.db.fs.chunks.delete_many({"files_id": digest})
            self.fs.put(body, _id=digest, filename=digest, metadata={"sha256": digest})
        stored = self.fs.get(digest).read()
        if len(stored) != len(body) or sha256(stored) != digest:
            raise CorpusError("gridfs_verification_failed")
        return self.immutable("blobs", {**record, "storage": "gridfs", "gridfs_id": digest})

    def get_blob(self, digest):
        record = self.db.blobs.find_one({"_id": digest})
        if not record:
            raise CorpusError("missing_blob")
        data = bytes(record["body"]) if record["storage"] == "inline" else self.fs.get(record["gridfs_id"]).read()
        if sha256(data) != digest or len(data) != record["byte_length"]:
            raise CorpusError("blob_integrity_failure")
        return data

    def metadata(self, item: dict, contest: dict):
        pid = problem_id(item["contestId"], item["index"])
        now = utcnow()
        tags = list(item.get("tags", []))
        metadata = {"source": "codeforces", "contest_id": item["contestId"], "problem_index": item["index"],
                    "title": item["name"], "canonical_url": f"https://codeforces.com/problemset/problem/{item['contestId']}/{item['index']}",
                    "contest": {"name": contest["name"], "started_at": contest["started_at"], "phase": "FINISHED"},
                    "original_tags": tags}
        if "rating" in item:
            metadata["rating"] = item["rating"]
        metadata["metadata_fingerprint"] = identity(metadata)
        existing = self.db.problems.find_one({"_id": pid})
        initial = {"_id": pid, "schema_version": 1, "approved_labels": {
            "taxonomy_version": "v1", "broad": [], "techniques": [], "annotation_ids": [], "projection_revision": 0},
            "primary_editorial_link_id": None, "quality": {"statement": "missing", "editorial": "missing",
            "solutions": "not_requested", "labels": "unreviewed", "issue_codes": []}, "created_at": now}
        record = {**(existing or initial), **metadata, "metadata_checked_at": now, "updated_at": now}
        if "rating" not in item:
            record.pop("rating", None)
        validate("problems", record)
        update = {"$set": {**metadata, "metadata_checked_at": now, "updated_at": now},
                  "$setOnInsert": {k: v for k, v in initial.items() if k != "_id"}}
        if "rating" not in item:
            update["$unset"] = {"rating": ""}
        self.db.problems.update_one({"_id": pid}, update, upsert=True)
        return pid

    def ensure_source(self, sid, kind, url, pid=None, blog_id=None, attribution="", **optional):
        record = {"_id": sid, "schema_version": 1, "kind": kind, "canonical_url": url, "language": "en",
                  "attribution": attribution, "observation_seq": 0, "published_seq": 0, "created_at": utcnow()}
        if pid:
            if not self.db.problems.find_one({"_id": pid}):
                raise CorpusError("unknown_problem")
            record["problem_id"] = pid
        if blog_id is not None:
            record["blog_id"] = blog_id
        record.update(optional)
        validate("source_documents", record)
        self.db.source_documents.update_one({"_id": sid}, {"$setOnInsert": record}, upsert=True)
        return sid

    def observe(self, sid):
        source = self.db.source_documents.find_one_and_update({"_id": sid},
            {"$inc": {"observation_seq": 1}, "$set": {"last_checked_at": utcnow()}}, return_document=ReturnDocument.AFTER)
        if not source:
            raise CorpusError("unknown_source")
        return source["observation_seq"]

    def failure(self, sid, seq, code):
        self.db.source_documents.update_one({"_id": sid, "observation_seq": seq},
            {"$set": {"last_error": code[:128], "last_checked_at": utcnow(), "stale": True}})
        source = self.db.source_documents.find_one({"_id": sid})
        if source and source.get("problem_id") and source["observation_seq"] == seq:
            self.db.problems.update_one({"_id": source["problem_id"]}, [{"$set": {
                "quality.issue_codes": {"$slice": [{"$setUnion": ["$quality.issue_codes", [code[:128]]]}, -256]}}}])

    def ingest(self, sid: str, seq: int, raw: bytes, parsed: Parsed, provenance: dict,
               config: dict | None = None, crash_after: str | None = None):
        source = self.db.source_documents.find_one({"_id": sid})
        if not source:
            raise CorpusError("unknown_source")
        digest = self.put_blob(raw)
        if crash_after == "blob":
            raise CorpusError("simulated_crash")
        fingerprint = processor_fingerprint(source["kind"], config)
        vid = identity({"source_document_id": sid, "raw_sha256": digest, "processor_fingerprint": fingerprint})
        version = {"_id": vid, "schema_version": 1, "source_document_id": sid, "raw_sha256": digest,
                   "processor_fingerprint": fingerprint, "extractor_version": EXTRACTOR_VERSION,
                   "processor_config": config or {},
                   "cleaner_version": CLEANER_VERSION, "language": "en", "sections": parsed.sections,
                   "clean_text": parsed.clean_text, "clean_sha256": sha256(parsed.clean_text.encode("utf-8")),
                   "created_at": utcnow(), "provenance": provenance, "warnings": parsed.warnings}
        self.immutable("document_versions", version)
        if crash_after == "version":
            raise CorpusError("simulated_crash")
        published = self.db.source_documents.update_one({"_id": sid, "published_seq": {"$lt": seq}},
            {"$set": {"current_version_id": vid, "current_raw_sha256": digest, "published_seq": seq,
                      "last_success_at": utcnow(), "stale": False}, "$unset": {"last_error": ""}})
        if crash_after == "source":
            raise CorpusError("simulated_crash")
        current = self.db.source_documents.find_one({"_id": sid})
        if current.get("current_version_id") == vid and current["published_seq"] == seq:
            if source["kind"] == "statement":
                self.project_statement(current, version)
            if source["kind"] == "solution" and source.get("problem_id"):
                self.db.problems.update_one({"_id": source["problem_id"]}, {"$set": {"quality.solutions": "ready"}})
        return vid, bool(published.modified_count)

    def project_statement(self, source, version):
        pid, seq = source["problem_id"], source["published_seq"]
        projection = {"source_document_id": source["_id"], "version_id": version["_id"],
                      "clean_sha256": version["clean_sha256"], "clean_text": version["clean_text"],
                      "language": "en", "observation_seq": seq}
        self.db.problems.update_one({"_id": pid, "$or": [
            {"statement.observation_seq": {"$exists": False}}, {"statement.observation_seq": {"$lte": seq}}]},
            {"$set": {"statement": projection, "quality.statement": "ready", "updated_at": utcnow()}})

    def link_editorial(self, pid, vid, section_key, mapping):
        version = self.db.document_versions.find_one({"_id": vid})
        problem = self.db.problems.find_one({"_id": pid})
        if not version or not problem:
            raise CorpusError("missing_editorial_reference")
        source = self.db.source_documents.find_one({"_id": version["source_document_id"], "kind": "editorial"})
        section = next((s for s in version["sections"] if s["key"] == section_key), None)
        if not source or not section or section_key != pid:
            raise CorpusError("editorial_section_mismatch")
        method, confidence, status = mapping
        lid = identity({"problem_id": pid, "editorial_version_id": vid, "section_key": section_key})
        record = {"_id": lid, "schema_version": 1, "problem_id": pid,
                  "editorial_source_document_id": source["_id"], "editorial_version_id": vid,
                  "section_key": section_key, "section_text": section["text"], "mapping_method": method,
                  "confidence": confidence, "review_status": status, "created_at": utcnow()}
        validate("editorial_links", record)
        # Preserve a review decision if an identical association is imported again.
        self.db.editorial_links.update_one({"_id": lid}, {"$setOnInsert": record}, upsert=True)
        stored = self.db.editorial_links.find_one({"_id": lid})
        if source.get("current_version_id") == vid:
            if stored["review_status"] == "approved":
                self.db.problems.update_one({"_id": pid}, {"$set": {
                    "primary_editorial_link_id": lid, "quality.editorial": "ready"}})
            elif not problem.get("primary_editorial_link_id"):
                self.db.problems.update_one({"_id": pid}, {"$set": {"quality.editorial": "candidate"}})
        return lid

    def put_taxonomy(self, records):
        combined = {d["_id"]: d for d in self.db.taxonomy.find()}
        for record in records:
            validate("taxonomy", record)
            old = combined.get(record["_id"])
            if old and old != record:
                raise CorpusError("taxonomy_revision_requires_new_ids")
            combined[record["_id"]] = record
        visiting, visited = set(), set()
        def visit(key):
            if key in visiting:
                raise CorpusError("taxonomy_cycle")
            if key in visited:
                return
            if key not in combined:
                raise CorpusError("unknown_taxonomy_parent")
            visiting.add(key)
            for parent in combined[key]["parent_ids"]:
                if parent not in combined:
                    raise CorpusError("unknown_taxonomy_parent")
                if combined[parent]["taxonomy_version"] != combined[key]["taxonomy_version"]:
                    raise CorpusError("taxonomy_parent_version_mismatch")
                visit(parent)
            visiting.remove(key)
            visited.add(key)
        for key in combined:
            visit(key)
        for record in records:
            self.immutable("taxonomy", record)

    def annotate(self, record):
        evidence = record["evidence"]
        if not self.db.problems.find_one({"_id": record["problem_id"]}):
            raise CorpusError("unknown_problem")
        label = self.db.taxonomy.find_one({"_id": record["label_id"], "taxonomy_version": record["taxonomy_version"]})
        version = self.db.document_versions.find_one({"_id": evidence["version_id"]})
        if not label or not version:
            raise CorpusError("invalid_annotation_reference")
        section = next((s for s in version["sections"] if s["key"] == evidence["locator"]), None)
        if not section or evidence["span"] not in section["text"]:
            raise CorpusError("invalid_annotation_evidence_span")
        source = self.db.source_documents.find_one({"_id": version["source_document_id"]})
        associated = source.get("problem_id") == record["problem_id"] or self.db.editorial_links.find_one({
            "problem_id": record["problem_id"], "editorial_version_id": version["_id"], "section_key": evidence["locator"]})
        if not associated:
            raise CorpusError("annotation_evidence_problem_mismatch")
        key = {k: record[k] for k in ("problem_id", "label_id", "taxonomy_version", "origin", "producer_version")}
        key["evidence_fingerprint"] = identity(evidence)
        now = utcnow()
        complete = {**record, "_id": identity(key), "schema_version": 1, "created_at": now, "updated_at": now}
        validate("annotations", complete)
        self.db.annotations.update_one({"_id": complete["_id"]}, {"$setOnInsert": complete}, upsert=True)
        self.rebuild_labels(record["problem_id"])
        return complete["_id"]

    def rebuild_labels(self, pid, taxonomy_version="v1"):
        annotations = list(self.db.annotations.find({"problem_id": pid, "status": "approved", "taxonomy_version": taxonomy_version}))
        broad, techniques = set(), set()
        for annotation in annotations:
            label = self.db.taxonomy.find_one({"_id": annotation["label_id"]})
            if not label:
                raise CorpusError("missing_label")
            (broad if label["kind"] == "broad" else techniques).add(label["_id"])
        old = self.db.problems.find_one({"_id": pid})
        projection = {"taxonomy_version": taxonomy_version, "broad": sorted(broad), "techniques": sorted(techniques),
                      "annotation_ids": sorted(a["_id"] for a in annotations),
                      "projection_revision": old["approved_labels"]["projection_revision"] + 1}
        # Bound and validate before atomic replacement of all projection fields.
        validate("problems", {**old, "approved_labels": projection})
        self.db.problems.update_one({"_id": pid}, {"$set": {"approved_labels": projection,
            "quality.labels": "reviewed" if annotations else "unreviewed"}})

    def reconcile(self, repair=False):
        issues, repaired = [], []
        for source in self.db.source_documents.find({"current_version_id": {"$exists": True}}):
            version = self.db.document_versions.find_one({"_id": source["current_version_id"]})
            if not version or version["source_document_id"] != source["_id"] or version["raw_sha256"] != source.get("current_raw_sha256"):
                issues.append({"code": "dangling_source_version", "id": source["_id"]})
                continue
            if source["kind"] == "statement":
                problem = self.db.problems.find_one({"_id": source.get("problem_id")})
                if not problem:
                    issues.append({"code": "dangling_problem", "id": source["_id"]})
                elif (problem.get("statement", {}).get("version_id") != version["_id"]
                      or problem.get("statement", {}).get("observation_seq") != source["published_seq"]
                      or problem.get("statement", {}).get("clean_text") != version["clean_text"]):
                    if repair:
                        self.project_statement(source, version)
                        repaired.append(source["_id"])
                    else:
                        issues.append({"code": "stale_statement_projection", "id": source["_id"]})
        for version in self.db.document_versions.find():
            expected_id = identity({"source_document_id": version["source_document_id"],
                                    "raw_sha256": version["raw_sha256"],
                                    "processor_fingerprint": version["processor_fingerprint"]})
            if version["_id"] != expected_id or sha256(version["clean_text"].encode("utf-8")) != version["clean_sha256"]:
                issues.append({"code": "version_integrity_failure", "id": version["_id"]})
            if not self.db.source_documents.find_one({"_id": version["source_document_id"]}):
                issues.append({"code": "dangling_version_source", "id": version["_id"]})
            try:
                self.get_blob(version["raw_sha256"])
            except Exception:
                issues.append({"code": "missing_or_corrupt_blob", "id": version["_id"]})
        for link in self.db.editorial_links.find():
            version = self.db.document_versions.find_one({"_id": link["editorial_version_id"]})
            section = next((s for s in (version or {}).get("sections", []) if s["key"] == link["section_key"]), None)
            if (not section or section["text"] != link["section_text"]
                    or version["source_document_id"] != link["editorial_source_document_id"]
                    or not self.db.problems.find_one({"_id": link["problem_id"]})):
                issues.append({"code": "invalid_editorial_link", "id": link["_id"]})
        for annotation in self.db.annotations.find():
            if (not self.db.problems.find_one({"_id": annotation["problem_id"]})
                    or not self.db.taxonomy.find_one({"_id": annotation["label_id"], "taxonomy_version": annotation["taxonomy_version"]})
                    or not self.db.document_versions.find_one({"_id": annotation["evidence"]["version_id"]})):
                issues.append({"code": "dangling_annotation", "id": annotation["_id"]})
        for problem in self.db.problems.find():
            projection = problem.get("statement")
            if projection and (not self.db.source_documents.find_one({"_id": projection["source_document_id"]})
                               or not self.db.document_versions.find_one({"_id": projection["version_id"]})):
                issues.append({"code": "dangling_statement_projection", "id": problem["_id"]})
            primary = problem.get("primary_editorial_link_id")
            if primary and not self.db.editorial_links.find_one({"_id": primary, "problem_id": problem["_id"], "review_status": "approved"}):
                issues.append({"code": "invalid_primary_editorial", "id": problem["_id"]})
            for aid in problem["approved_labels"]["annotation_ids"]:
                if not self.db.annotations.find_one({"_id": aid, "problem_id": problem["_id"], "status": "approved"}):
                    issues.append({"code": "stale_approved_labels", "id": problem["_id"]})
        for job in self.db.ingestion_jobs.find({"status": "running"}):
            if repair:
                self.db.ingestion_jobs.update_one({"_id": job["_id"]}, {"$set": {"status": "retry", "updated_at": utcnow()}})
                repaired.append(job["_id"])
            else:
                issues.append({"code": "interrupted_job", "id": job["_id"]})
        # Only remove abandoned uploads after an hour; no valid blob reference is removed.
        cutoff = utcnow() - timedelta(hours=1)
        for file in self.db.fs.files.find({"uploadDate": {"$lt": cutoff}}):
            if not self.db.blobs.find_one({"gridfs_id": file["_id"]}):
                if repair:
                    self.fs.delete(file["_id"])
                    repaired.append(str(file["_id"]))
                else:
                    issues.append({"code": "orphan_gridfs_file", "id": str(file["_id"])})
        for chunk in self.db.fs.chunks.aggregate([{"$group": {"_id": "$files_id"}}]):
            if not self.db.fs.files.find_one({"_id": chunk["_id"]}):
                # Incomplete GridFS uploads have no uploadDate. Report, don't delete blindly.
                issues.append({"code": "incomplete_gridfs_upload", "id": str(chunk["_id"])})
        return {"issues": issues, "repaired": repaired}
