from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .core import CorpusError, Settings, WorkerLock, canonical, safe_error
from .export import export_corpus, verify_export
from .http import Transport
from .pipeline import Pipeline, read_local
from .repository import Repository


def date(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError("use an ISO-8601 UTC date") from None


def parser():
    result = argparse.ArgumentParser(prog="cpss", description="CPSS corpus ingestion; no retrieval or classifiers")
    result.add_argument("--database", help="override CPSS_DATABASE")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("init-db")
    prepare = commands.add_parser("prepare-openr1")
    prepare.add_argument("--output", type=Path, default=Path("data/openr1"))
    importer = commands.add_parser("import-openr1")
    importer.add_argument("--snapshot", type=Path, default=Path("data/openr1"))
    importer.add_argument("--limit", type=int, help="optional pilot limit; default imports all complete problems")
    commands.add_parser("audit-complete")
    activate = commands.add_parser("activate-openr1")
    activate.add_argument("--staging-database", required=True)
    activate.add_argument("--backup-database")
    migration = commands.add_parser("migrate")
    migration.add_argument("--apply", action="store_true", help="apply only if every existing record already conforms")
    discovery = commands.add_parser("discover")
    discovery.add_argument("--limit", type=int, default=2000, help="target of statement + editorial ready problems")
    discovery.add_argument("--candidate-limit", type=int)
    discovery.add_argument("--offline-catalog", type=Path)
    discovery.add_argument("--order", choices=["newest", "oldest"], default="newest")
    discovery.add_argument("--since", type=date)
    discovery.add_argument("--until", type=date, help="exclusive upper date")
    discovery.add_argument("--contest-id", type=int, action="append")
    discovery.add_argument("--refresh", action="store_true")
    editorials = commands.add_parser("collect-editorials")
    editorials.add_argument("--run-id", required=True)
    editorials.add_argument("--editorial-manifest", type=Path, required=True)
    editorials.add_argument("--limit", type=int, default=20, help="maximum shared editorial documents to process")
    editorials.add_argument("--retry", action="store_true", help="explicitly retry failed/blocked editorial jobs")
    editorials.add_argument("--refresh", action="store_true")
    for name in ("ingest", "resume", "retry"):
        p = commands.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--limit", type=int, default=20, help="additional editorial-ready target; at most 3x this many jobs")
        p.add_argument("--import-dir", type=Path)
        p.add_argument("--editorial-manifest", type=Path)
        p.add_argument("--allow-pages", action="store_true")
        p.add_argument("--refresh", action="store_true")
    reprocess = commands.add_parser("reprocess")
    reprocess.add_argument("--source-id")
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--repair", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--run-id")
    export.add_argument("--include-missing-editorials", action="store_true")
    export.add_argument("--duplicate-groups", type=Path, help="JSON array of arrays of problem IDs")
    export.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    verify = commands.add_parser("verify-export")
    verify.add_argument("path", type=Path)
    taxonomy = commands.add_parser("import-taxonomy")
    taxonomy.add_argument("path", type=Path)
    annotate = commands.add_parser("annotate")
    annotate.add_argument("path", type=Path, help="JSON array of annotation proposals/decisions")
    review = commands.add_parser("review")
    review.add_argument("annotation_id")
    review.add_argument("--status", required=True, choices=["approved", "rejected", "proposed"])
    review.add_argument("--reviewer", required=True)
    solution = commands.add_parser("import-solution")
    solution.add_argument("path", type=Path)
    solution.add_argument("--problem-id", required=True)
    solution.add_argument("--url", required=True, help="original Codeforces source URL; never fetched")
    solution.add_argument("--language", required=True)
    solution.add_argument("--attribution", default="")
    solution.add_argument("--editorial-version-id")
    solution.add_argument("--submission-id", type=int)
    return result


def execute(args, settings, repo):
    command = args.command
    if command == "doctor":
        repo.client.admin.command("ping")
        latest = repo.db.ingestion_runs.find_one({}, {"config": 1}, sort=[("started_at", -1)]) or {}
        return {"mongodb": "reachable", "server_version": repo.client.server_info()["version"],
                "database": settings.database, "python": sys.version.split()[0], "credentials_logged": False,
                "primary_source": latest.get("config", {}).get("provider", "codeforces_direct"),
                "complete_problems": repo.db.problems.count_documents({"corpus.complete": True})}
    if command == "init-db":
        return repo.bootstrap()
    if command == "prepare-openr1":
        from .openr1 import prepare
        return prepare(args.output, progress=lambda state: print(canonical(state).decode(), file=sys.stderr, flush=True))
    if command == "import-openr1":
        from .openr1 import import_snapshot
        if not settings.database.startswith("cpss_openr1_stage_"):
            raise CorpusError("openr1_import_requires_staging_database")
        repo.bootstrap()
        return import_snapshot(repo, args.snapshot, args.limit,
            progress=lambda state: print(canonical(state).decode(), file=sys.stderr, flush=True))
    if command == "audit-complete":
        from .openr1 import audit_complete
        return audit_complete(repo)
    if command == "activate-openr1":
        from .openr1 import activate
        return activate(repo, args.staging_database, args.backup_database)
    if command == "migrate":
        return repo.migration_check(args.apply)
    if command == "reconcile":
        report = repo.reconcile(args.repair)
        report["ok"] = not report["issues"]
        return report
    if command == "reprocess":
        return Pipeline(repo).reprocess(args.source_id)
    if command == "export":
        groups = json.loads(read_local(args.duplicate_groups)) if args.duplicate_groups else None
        return export_corpus(repo, args.output, args.run_id, not args.include_missing_editorials, groups, args.fractions)
    if command == "import-taxonomy":
        records = json.loads(read_local(args.path))
        repo.put_taxonomy(records)
        return {"imported": len(records)}
    if command == "annotate":
        return {"annotation_ids": [repo.annotate(record) for record in json.loads(read_local(args.path))]}
    if command == "import-solution":
        from . import parsers
        from .core import identity, safe_url, utcnow
        safe_url(args.url)
        optional = {"programming_language": args.language, "collection_method": "permitted_local_import"}
        if args.editorial_version_id:
            if not repo.db.editorial_links.find_one({"problem_id": args.problem_id,
                    "editorial_version_id": args.editorial_version_id, "review_status": "approved"}):
                raise CorpusError("solution_editorial_reference_mismatch")
            optional["parent_version_id"] = args.editorial_version_id
        if args.submission_id is not None:
            optional.update(submission_id=args.submission_id, submission_url=args.url)
        sid = "cf:solution:import:" + identity({"problem_id": args.problem_id, "url": args.url, "language": args.language})
        repo.ensure_source(sid, "solution", args.url, pid=args.problem_id, attribution=args.attribution, **optional)
        raw = read_local(args.path)
        seq = repo.observe(sid)
        vid, _ = repo.ingest(sid, seq, raw, parsers.solution(raw), {"url": args.url,
            "retrieved_at": utcnow(), "method": "permitted_local_import", "status_code": 200,
            "attribution": args.attribution})
        return {"source_document_id": sid, "version_id": vid}
    if command == "review":
        from .core import utcnow
        from .schemas import validate
        record = repo.db.annotations.find_one({"_id": args.annotation_id})
        if not record:
            raise CorpusError("unknown_annotation")
        record.update(status=args.status, reviewer=args.reviewer, updated_at=utcnow())
        validate("annotations", record)
        repo.db.annotations.replace_one({"_id": record["_id"]}, record)
        repo.rebuild_labels(record["problem_id"], record["taxonomy_version"])
        return {"reviewed": record["_id"], "status": record["status"]}
    transport = Transport(settings, allow_pages=getattr(args, "allow_pages", False))
    try:
        pipeline = Pipeline(repo, transport)
        if command == "discover":
            return pipeline.discover(args.limit, args.candidate_limit, args.offline_catalog, args.order,
                                     args.since, args.until, args.contest_id, args.refresh)
        if command == "collect-editorials":
            return pipeline.collect_editorials(args.run_id, args.editorial_manifest, args.limit, args.retry, args.refresh,
                progress=lambda state: print(canonical(state).decode("utf-8"), file=sys.stderr, flush=True))
        if command == "resume":
            repo.reconcile(repair=True)
        return pipeline.run(args.run_id, args.limit, args.import_dir, args.editorial_manifest,
                            retry=command == "retry", refresh=args.refresh,
                            progress=lambda state: print(canonical(state).decode("utf-8"), file=sys.stderr, flush=True))
    finally:
        transport.close()


def main(argv=None):
    args = parser().parse_args(argv)
    settings = Settings.load()
    if args.database:
        from dataclasses import replace
        settings = replace(settings, database=args.database)
    repo = None
    try:
        if args.command == "verify-export":
            report = verify_export(args.path)
        else:
            repo = Repository(settings)
            with WorkerLock(settings.cache_dir):
                report = execute(args, settings, repo)
        print(canonical(report).decode("utf-8"))
        if report.get("errors") or report.get("issues") or report.get("ok") is False:
            return 2
        return 0
    except KeyboardInterrupt:
        print('{"error":"interrupted_resume_available"}', file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"error": safe_error(exc)}), file=sys.stderr)
        return 1
    finally:
        if repo:
            repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
