from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import parsers
from .core import CorpusError, canonical, identity, safe_error, utcnow
from .http import Fetched, Transport
from .repository import Repository
from .schemas import validate


def read_local(path: Path) -> bytes:
    if path.name.lower() == "token.txt":
        raise CorpusError("credential_file_forbidden")
    return path.read_bytes()


def imported(path: Path, url: str):
    return Fetched(read_local(path), url, utcnow())


def manifest_entries(path: Path | None):
    if path is None:
        return []
    content = json.loads(read_local(path))
    entries = content["entries"]
    if not isinstance(entries, list):
        raise CorpusError("invalid_editorial_manifest")
    for item in entries:
        if type(item.get("blog_id")) is not int or item["blog_id"] <= 0:
            raise CorpusError("invalid_manifest_blog_id")
        if type(item.get("contest_id")) is not int:
            raise CorpusError("invalid_manifest_contest_id")
        for pid in item.get("problem_ids", []):
            if not pid.startswith(f"codeforces:{item['contest_id']}:"):
                raise CorpusError("manifest_contest_problem_mismatch")
    return entries


def select_metadata(contests, problems, limit, order="newest", since=None, until=None, contest_ids=None):
    if limit <= 0 or order not in {"newest", "oldest"}:
        raise CorpusError("invalid_selection")
    now = utcnow().timestamp()
    allowed = {}
    for contest in contests:
        stamp = contest.get("startTimeSeconds")
        if (contest.get("phase") != "FINISHED" or not isinstance(stamp, int) or stamp > now
                or contest.get("type") != "CF" or contest.get("id", 100000) >= 100000):
            continue
        if contest_ids and contest["id"] not in contest_ids:
            continue
        name = contest.get("name", "")
        # API gym=false excludes Gym; scoring type alone does not identify standard rounds.
        if not contest_ids and (not re.search(r"(?:Codeforces|Educational).*\bRound\b", name, re.I)
                                or re.search(r"April Fools|Kotlin|Marathon|Teams?|Experimental", name, re.I)):
            continue
        started = datetime.fromtimestamp(stamp, timezone.utc)
        if since and started < since or until and started >= until:
            continue
        allowed[contest["id"]] = {**contest, "started_at": started}
    selected = [p for p in problems if p.get("contestId") in allowed
                and p.get("type") == "PROGRAMMING" and p.get("index") and p.get("name")]
    # Stable tie breaks do not substitute IDs for dates.
    selected.sort(key=lambda p: (allowed[p["contestId"]]["startTimeSeconds"], p["contestId"]),
                  reverse=order == "newest")
    # Keep the ordinary A..H ordering within each contest.
    result = []
    for contest_id in dict.fromkeys(p["contestId"] for p in selected):
        result.extend(sorted((p for p in selected if p["contestId"] == contest_id), key=lambda p: p["index"]))
    return [(p, allowed[p["contestId"]]) for p in result[:limit]]


class Pipeline:
    def __init__(self, repo: Repository, transport: Transport | None = None):
        self.repo, self.http = repo, transport

    def collect_editorials(self, run_id, manifest, limit=20, retry=False, refresh=False, progress=None):
        """Collect known-ID editorials independently of statement-page availability."""
        run = self.repo.db.ingestion_runs.find_one({"_id": run_id})
        if not run:
            raise CorpusError("unknown_run")
        if limit <= 0:
            raise CorpusError("invalid_limit")
        entries = manifest_entries(manifest)
        run_pids = set(self.repo.db.ingestion_jobs.distinct("entity_id", {"run_id": run_id, "stage": "statement"}))
        grouped = {}
        for entry in entries:
            related = list(self.repo.db.problems.find({"contest_id": entry["contest_id"]}, {"_id": 1}))
            if not any(p["_id"] in run_pids for p in related):
                raise CorpusError("editorial_contest_outside_run")
            grouped.setdefault(entry["blog_id"], []).append(entry)
        attempted, errors = 0, []
        for blog_id, related in grouped.items():
            sid = f"cf:editorial:{blog_id}:en"
            jid = identity({"run_id": run_id, "stage": "editorial", "entity_id": sid})
            job = {"_id": jid, "schema_version": 1, "run_id": run_id, "stage": "editorial", "entity_id": sid,
                   "status": "pending", "attempts": 0, "next_attempt_at": utcnow(),
                   "checkpoint": {"source_document_id": sid, "blog_ids": [blog_id]}, "updated_at": utcnow()}
            validate("ingestion_jobs", job)
            self.repo.db.ingestion_jobs.update_one({"_id": jid}, {"$setOnInsert": job}, upsert=True)
            job = self.repo.db.ingestion_jobs.find_one({"_id": jid})
            if job["status"] == "succeeded" or (job["status"] in {"blocked", "failed"} and not retry):
                continue
            if not retry and job["next_attempt_at"] > utcnow():
                continue
            if attempted >= limit or job["attempts"] >= 5:
                continue
            attempted += 1
            self.repo.db.ingestion_jobs.update_one({"_id": jid}, {"$set": {"status": "running", "updated_at": utcnow()},
                                                                  "$inc": {"attempts": 1}})
            try:
                vid = self.process_editorial(blog_id, related[0]["contest_id"], None, entries, refresh)
                approved = self.repo.db.editorial_links.count_documents({"editorial_version_id": vid, "review_status": "approved"})
                if not approved:
                    raise CorpusError("no_approved_editorial_sections")
                self.repo.db.ingestion_jobs.update_one({"_id": jid}, {"$set": {
                    "status": "succeeded", "checkpoint.completed": True, "updated_at": utcnow()}, "$unset": {"last_error": ""}})
                if progress:
                    progress({"blog_id": blog_id, "approved_sections": approved})
            except Exception as exc:
                code = safe_error(exc)
                blocked = any(s in code for s in ("blocked", "challenge", "robots"))
                status = "blocked" if blocked else "retry" if isinstance(exc, CorpusError) and exc.retryable else "failed"
                self.repo.db.ingestion_jobs.update_one({"_id": jid}, {"$set": {
                    "status": status, "last_error": code, "updated_at": utcnow(), "next_attempt_at": utcnow() + timedelta(minutes=5)}})
                errors.append({"blog_id": blog_id, "code": code})
                if blocked or code == "network_failure":
                    break
        counts = self.run_counts(run_id)
        self.repo.db.ingestion_runs.update_one({"_id": run_id}, {"$set": {"counts": counts}})
        return {"run_id": run_id, "processed_editorials": attempted, "counts": counts, "errors": errors}

    def discover(self, target=2000, candidate_limit=None, catalog_dir=None, order="newest",
                 since=None, until=None, contest_ids=None, refresh=False):
        candidate_limit = candidate_limit or target * 3
        if candidate_limit < target:
            raise CorpusError("candidate_limit_below_target")
        config = {"target": target, "candidate_limit": candidate_limit, "order": order,
                  "since": since.isoformat() if since else None, "until": until.isoformat() if until else None,
                  "contest_ids": sorted(contest_ids or []), "editorials_required_for_target": True,
                  "mode": "import" if catalog_dir else "live", "worker_count": 1}
        run_id = str(uuid.uuid4())
        run = {"_id": run_id, "schema_version": 1, "started_at": utcnow(), "status": "running",
               "config_fingerprint": identity(config), "config": config, "counts": {}, "export_refs": []}
        self.repo.immutable("ingestion_runs", run)
        try:
            catalogues = []
            for method, filename in (("contest.list", "contests.json"), ("problemset.problems", "problems.json")):
                if catalog_dir:
                    response = imported(Path(catalog_dir) / filename, "https://codeforces.com/api/" + method)
                else:
                    if not self.http:
                        raise CorpusError("live_transport_required")
                    response = self.http.api(method, refresh=refresh, **({"gym": "false"} if method == "contest.list" else {}))
                payload = json.loads(response.body)
                if payload.get("status") != "OK":
                    raise CorpusError("invalid_catalogue")
                result = payload["result"]
                sid = f"cf:catalog:{method}:en"
                self.repo.ensure_source(sid, "metadata_catalog", response.url)
                seq = self.repo.observe(sid)
                parsed = parsers.Parsed([{"key": "catalogue", "text": canonical(result).decode("utf-8")}])
                self.repo.ingest(sid, seq, response.body, parsed, response.provenance("import" if catalog_dir else "official_api"))
                catalogues.append(result)
            selection = select_metadata(catalogues[0], catalogues[1]["problems"], candidate_limit,
                                        order, since, until, contest_ids)
            for item, contest in selection:
                pid = self.repo.metadata(item, contest)
                job_id = identity({"run_id": run_id, "stage": "statement", "entity_id": pid})
                self.repo.immutable("ingestion_jobs", {"_id": job_id, "schema_version": 1, "run_id": run_id,
                    "stage": "statement", "entity_id": pid, "status": "pending", "attempts": 0,
                    "next_attempt_at": utcnow(), "checkpoint": {}, "updated_at": utcnow()})
            self.repo.db.ingestion_runs.update_one({"_id": run_id}, {"$set": {"counts.discovered": len(selection)}})
            return {"run_id": run_id, "discovered": len(selection), "target": target}
        except Exception:
            self.repo.db.ingestion_runs.update_one({"_id": run_id}, {"$set": {"status": "failed", "completed_at": utcnow()}})
            raise

    def process_statement(self, job, import_dir=None, entries=None, refresh=False):
        problem = self.repo.db.problems.find_one({"_id": job["entity_id"]})
        if not problem:
            raise CorpusError("unknown_problem")
        sid = f"cf:statement:{problem['contest_id']}:{problem['problem_index']}:en"
        self.repo.ensure_source(sid, "statement", problem["canonical_url"], problem["_id"])
        checkpoint = job["checkpoint"]
        # Keep the observation sequence on retry; allocate a new sequence only for a new job.
        seq = checkpoint.get("observation_seq") or self.repo.observe(sid)
        self.repo.db.ingestion_jobs.update_one({"_id": job["_id"]}, {"$set": {
            "checkpoint.source_document_id": sid, "checkpoint.observation_seq": seq}})
        try:
            method = checkpoint.get("method", "import" if import_dir else "statement_page")
            if checkpoint.get("raw_sha256"):
                fetched = Fetched(self.repo.get_blob(checkpoint["raw_sha256"]), checkpoint["url"], checkpoint["retrieved_at"])
            elif import_dir:
                path = Path(import_dir) / "statements" / str(problem["contest_id"]) / f"{problem['problem_index']}.html"
                fetched = imported(path, problem["canonical_url"])
            else:
                fetched = self.http.fetch(problem["canonical_url"] + "?locale=en", refresh=refresh)
            raw_digest = self.repo.put_blob(fetched.body)
            self.repo.db.ingestion_jobs.update_one({"_id": job["_id"]}, {"$set": {
                "checkpoint.raw_sha256": raw_digest, "checkpoint.url": fetched.url,
                "checkpoint.retrieved_at": fetched.retrieved_at, "checkpoint.method": method}})
            parsed = parsers.statement(fetched.body, problem["problem_index"])
            self.repo.ingest(sid, seq, fetched.body, parsed, fetched.provenance(method))
        except Exception as exc:
            self.repo.failure(sid, seq, safe_error(exc))
            raise
        blog_ids = {e["blog_id"] for e in entries or [] if e["contest_id"] == problem["contest_id"]
                    and (not e.get("problem_ids") or problem["_id"] in e["problem_ids"])}
        blog_ids.update(parsers.editorial_candidates(fetched.body, problem["canonical_url"]))
        editorial_errors = []
        for blog_id in sorted(blog_ids):
            try:
                self.process_editorial(blog_id, problem["contest_id"], import_dir, entries or [], refresh)
            except Exception as exc:
                editorial_errors.append(safe_error(exc))
        if editorial_errors:
            # An independent resumable editorial job preserves successful statement ingestion.
            entity = problem["_id"]
            eid = identity({"run_id": job["run_id"], "stage": "editorial", "entity_id": entity})
            retry_job = {"_id": eid, "schema_version": 1, "run_id": job["run_id"], "stage": "editorial",
                         "entity_id": entity, "status": "retry", "attempts": 0,
                         "next_attempt_at": utcnow() + timedelta(minutes=5),
                         "checkpoint": {"blog_ids": sorted(blog_ids)}, "last_error": editorial_errors[0], "updated_at": utcnow()}
            validate("ingestion_jobs", retry_job)
            self.repo.db.ingestion_jobs.update_one({"_id": eid}, {"$setOnInsert": retry_job}, upsert=True)
        return {"statement": True, "editorial_errors": editorial_errors}

    def process_editorial(self, blog_id, contest_id, import_dir, entries, refresh=False):
        url = f"https://codeforces.com/blog/entry/{blog_id}"
        sid = f"cf:editorial:{blog_id}:en"
        self.repo.ensure_source(sid, "editorial", url, blog_id=blog_id)
        seq = self.repo.observe(sid)
        try:
            response = (imported(Path(import_dir) / "editorials" / f"{blog_id}.json", url) if import_dir
                        else self.http.api("blogEntry.view", blogEntryId=blog_id, refresh=refresh))
            payload = json.loads(response.body)
            if payload.get("result", {}).get("id") != blog_id:
                raise CorpusError("editorial_blog_identity_mismatch")
            # Include all discovered identities in linked contests, not only the requesting problem.
            related_contests = {contest_id} | {e["contest_id"] for e in entries if e["blog_id"] == blog_id}
            expected = sorted(p["_id"] for p in self.repo.db.problems.find({"contest_id": {"$in": list(related_contests)}}))
            curated = {}
            for entry in entries:
                if entry["blog_id"] == blog_id:
                    curated.update(entry.get("anchors", {}))
            parsed = parsers.editorial(response.body, expected, curated)
            config = {"expected_problems": expected, "curated_anchors": curated}
            provenance = response.provenance("import" if import_dir else "official_api")
            provenance["attribution"] = payload["result"].get("authorHandle", "")
            vid, _ = self.repo.ingest(sid, seq, response.body, parsed, provenance, config=config)
            for pid, mapping in parsed.mappings.items():
                self.repo.link_editorial(pid, vid, pid, mapping)
                if mapping[2] == "approved":
                    for number, code in enumerate(parsed.codes.get(pid, [])):
                        code_sid = f"cf:solution:{blog_id}:{pid}:{vid}:{number}:en"
                        self.repo.ensure_source(code_sid, "solution", url, pid=pid,
                                                attribution=provenance["attribution"], parent_version_id=vid,
                                                collection_method="editorial_embedded_code",
                                                **({"programming_language": parsed.code_languages[pid][number]}
                                                   if parsed.code_languages.get(pid, [])[number] else {}))
                        code_seq = self.repo.observe(code_sid)
                        self.repo.ingest(code_sid, code_seq, code.encode("utf-8"), parsers.solution(code.encode("utf-8")),
                                         {**provenance, "method": "editorial_embedded_code"},
                                         config={"editorial_version_id": vid, "section_key": pid, "block_index": number})
            return vid
        except Exception as exc:
            self.repo.failure(sid, seq, safe_error(exc))
            raise

    def run(self, run_id, limit=20, import_dir=None, manifest=None, retry=False, refresh=False, progress=None):
        run = self.repo.db.ingestion_runs.find_one({"_id": run_id})
        if not run:
            raise CorpusError("unknown_run")
        if limit <= 0:
            raise CorpusError("invalid_limit")
        entries = manifest_entries(manifest)
        statuses = ["pending", "retry"] + (["failed", "blocked"] if retry else [])
        jobs = list(self.repo.db.ingestion_jobs.find({"run_id": run_id, "status": {"$in": statuses},
                                                    "entity_id": {"$regex": "^codeforces:"}}))
        if retry:
            jobs += list(self.repo.db.ingestion_jobs.find({"run_id": run_id, "status": "running",
                                                         "entity_id": {"$regex": "^codeforces:"}}))
        jobs.sort(key=lambda j: (self.repo.db.problems.find_one({"_id": j["entity_id"]})["contest"]["started_at"],
                                 j["entity_id"]), reverse=run["config"]["order"] == "newest")
        processed, errors = 0, []
        baseline = self.run_counts(run_id)["editorial_ready"]
        for job in jobs:
            # Editorial page/API denial also closes its adapter for this batch.
            if self.http and any(self.http.blocked.values()):
                break
            if self.run_counts(run_id)["editorial_ready"] - baseline >= limit or processed >= limit * 3:
                break
            if not retry and job["next_attempt_at"] > utcnow():
                continue
            if job["attempts"] >= 5:
                continue
            processed += 1
            self.repo.db.ingestion_jobs.update_one({"_id": job["_id"]}, {
                "$set": {"status": "running", "updated_at": utcnow()}, "$inc": {"attempts": 1}})
            try:
                if job["stage"] == "statement":
                    self.process_statement(job, import_dir, entries, refresh)
                elif job["stage"] == "editorial":
                    problem = self.repo.db.problems.find_one({"_id": job["entity_id"]})
                    for bid in job["checkpoint"].get("blog_ids", []):
                        self.process_editorial(bid, problem["contest_id"], import_dir, entries, refresh)
                else:
                    raise CorpusError("unsupported_job_stage")
                self.repo.db.ingestion_jobs.update_one({"_id": job["_id"]}, {"$set": {
                    "status": "succeeded", "updated_at": utcnow(), "checkpoint.completed": True}, "$unset": {"last_error": ""}})
            except Exception as exc:
                code = safe_error(exc)
                blocked = any(s in code for s in ("blocked", "challenge", "robots", "opt_in"))
                status = "blocked" if blocked else "retry" if isinstance(exc, CorpusError) and exc.retryable else "failed"
                self.repo.db.ingestion_jobs.update_one({"_id": job["_id"]}, {"$set": {
                    "status": status, "last_error": code, "updated_at": utcnow(),
                    "next_attempt_at": utcnow() + timedelta(minutes=5)}})
                errors.append({"job_id": job["_id"], "code": code})
                if blocked or code == "network_failure":
                    break  # Circuit breaker: don't probe thousands of pages on a denied origin.
            if progress and processed % 10 == 0:
                progress({"processed_jobs": processed, "counts": self.run_counts(run_id)})
        counts = self.run_counts(run_id)
        state = "completed" if counts["editorial_ready"] >= run["config"]["target"] else "partial"
        self.repo.db.ingestion_runs.update_one({"_id": run_id}, {"$set": {
            "counts": counts, "status": state, "completed_at": utcnow()}})
        return {"run_id": run_id, "status": state, "processed_jobs": processed, "counts": counts, "errors": errors}

    def run_counts(self, run_id):
        ids = self.repo.db.ingestion_jobs.distinct("entity_id", {"run_id": run_id, "stage": "statement"})
        base = {"_id": {"$in": ids}}
        return {"discovered": len(ids),
                "statement_ready": self.repo.db.problems.count_documents({**base, "quality.statement": "ready"}),
                "editorial_sections_ready": self.repo.db.problems.count_documents({**base, "quality.editorial": "ready"}),
                "editorial_ready": self.repo.db.problems.count_documents({**base, "quality.statement": "ready", "quality.editorial": "ready"}),
                "solutions_ready": self.repo.db.problems.count_documents({**base, "quality.solutions": "ready"}),
                "failed_jobs": self.repo.db.ingestion_jobs.count_documents({"run_id": run_id, "status": {"$in": ["failed", "blocked"]}})}

    def reprocess(self, source_id=None):
        query = {"current_version_id": {"$exists": True}}
        if source_id:
            query["_id"] = source_id
        count = 0
        for source in self.repo.db.source_documents.find(query):
            if source["kind"] == "metadata_catalog":
                continue
            raw = self.repo.get_blob(source["current_raw_sha256"])
            old = self.repo.db.document_versions.find_one({"_id": source["current_version_id"]})
            config = old.get("processor_config", {})
            if source.get("dataset", {}).get("repository", "").startswith("open-r1/"):
                from .openr1 import code_text, statement_parsed
                row = json.loads(raw)
                if source["kind"] == "statement":
                    parsed = statement_parsed(row)
                elif source["kind"] == "editorial":
                    parsed = parsers.Parsed([{"key": source["problem_id"], "text": row["editorial"]}])
                else:
                    parsed = parsers.solution(code_text(row).encode("utf-8"))
            elif source["kind"] == "statement":
                parsed = parsers.statement(raw, source["problem_id"].rsplit(":", 1)[1])
            elif source["kind"] == "editorial":
                expected = config.get("expected_problems") or sorted({s["key"] for v in self.repo.db.document_versions.find({"source_document_id": source["_id"]})
                                   for s in v["sections"] if s["key"].startswith("codeforces:")})
                curated = config.get("curated_anchors", {})
                parsed = parsers.editorial(raw, expected, curated)
                config = {"expected_problems": expected, "curated_anchors": curated}
            else:
                parsed = parsers.solution(raw)
            seq = self.repo.observe(source["_id"])
            vid, _ = self.repo.ingest(source["_id"], seq, raw, parsed, old["provenance"], config=config)
            if source["kind"] == "editorial":
                if source.get("dataset"):
                    self.repo.link_editorial(source["problem_id"], vid, source["problem_id"], ("dataset_id", 1.0, "approved"))
                else:
                    for pid, mapping in parsed.mappings.items():
                        self.repo.link_editorial(pid, vid, pid, mapping)
            count += 1
        return {"reprocessed_sources": count}
