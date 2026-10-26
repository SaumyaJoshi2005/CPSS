from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cpss.core import canonical, utcnow

from .common import Config, DEFAULT_CONFIG, ModelingError, atomic_json, read_json
from .data import Corpus


def parser():
    p = argparse.ArgumentParser(
        prog="python -m modeling",
        description="CPSS modeling and semantic search over the frozen corpus",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = p.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--force", action="store_true")
    commands.add_parser("audit")
    train = commands.add_parser("train")
    train.add_argument("--run-name")
    from .retrieval.configuration import DEFAULT_SEARCH_CONFIG
    for name in ("index", "search", "serve", "retrieval-pool", "retrieval-evaluate", "retrieval-latency", "import-judgments"):
        search_command = commands.add_parser(name)
        search_command.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
        if name == "index":
            search_command.add_argument("--lexical-only", action="store_true")
        elif name == "serve":
            search_command.add_argument("--port", type=int)
        elif name == "search":
            search_command.add_argument("--query", required=True)
            search_command.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
            search_command.add_argument("--view", choices=["statement", "editorial"], default="editorial")
            search_command.add_argument("--limit", type=int)
            search_command.add_argument("--candidates", type=int)
            search_command.add_argument("--rating-min", type=int)
            search_command.add_argument("--rating-max", type=int)
            search_command.add_argument("--tag-boost", action="store_true")
            search_command.add_argument("--classifier-boost", action="store_true")
        elif name == "import-judgments":
            search_command.add_argument("--file", type=Path, required=True)
            search_command.add_argument("--reviewer", required=True)
            search_command.add_argument("--human-attested", action="store_true")
    for name in ("audit-baseline", "review-errors", "prepare-encoder", "experiment", "compare"):
        experiment = commands.add_parser(name)
        experiment.add_argument("--plan", type=Path, default=Path(__file__).parent / "config" / "experiment-v2.json")
        if name == "experiment":
            experiment.add_argument("--representation", choices=["word", "word_char", "embedding", "all"], default="all")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--final", action="store_true")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--run", type=Path, required=True)
    inputs = prediction.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--statement")
    inputs.add_argument("--statement-file", type=Path)
    commands.add_parser("prepare-contexts")
    evidence = commands.add_parser("context-input")
    evidence.add_argument("--problem-id", required=True)
    for name in ("validate-context", "review-context"):
        v = commands.add_parser(name)
        v.add_argument("--file", type=Path, required=True)
        v.add_argument("--illustrative-sources", type=Path)
        if name == "review-context":
            v.add_argument("--reviewer", required=True)
            v.add_argument("--decision", choices=["approved", "rejected"], required=True)
            v.add_argument("--output", type=Path, required=True)
    return p


def execute(args):
    config = Config.load(args.config)
    if args.command in {"index", "search", "serve", "retrieval-pool", "retrieval-evaluate", "retrieval-latency", "import-judgments"}:
        from .retrieval.configuration import SearchConfig
        from .retrieval.search import SearchEngine
        settings = SearchConfig.load(args.search_config)
        def progress(value):
            print(canonical(value).decode("utf8"), file=sys.stderr, flush=True)
        if args.command == "index":
            from .retrieval.indexing import build_indexes
            return build_indexes(settings, config, Corpus.verified(config), not args.lexical_only, progress)
        if args.command == "serve":
            from .application.server import serve
            serve(settings, config, args.port)
            return {"server_stopped": True}
        engine = SearchEngine.load(settings, config)
        if args.command == "search":
            return engine.search(args.query, method=args.method, view=args.view, limit=args.limit, candidates=args.candidates,
                                 rating_min=args.rating_min, rating_max=args.rating_max, tag_boost=args.tag_boost, classifier_boost=args.classifier_boost)
        if args.command == "retrieval-pool":
            from .retrieval.judgments import pool_candidates
            return pool_candidates(engine, progress)
        if args.command == "retrieval-evaluate":
            from .retrieval.evaluation import evaluate_retrieval
            return evaluate_retrieval(engine)
        if args.command == "retrieval-latency":
            from .retrieval.evaluation import measure_latency
            return measure_latency(engine, progress)
        from .retrieval.judgments import import_sheet
        return import_sheet(engine, args.file, args.reviewer, args.human_attested)
    if args.command == "predict":
        from .inference import load_bundle, predict
        if args.statement_file and args.statement_file.name.casefold() == "token.txt":
            raise ModelingError("Credential files cannot be inference inputs")
        text = args.statement_file.read_text("utf-8") if args.statement_file else args.statement
        return predict(load_bundle(args.run), text)
    corpus = Corpus.verified(config, force=getattr(args, "force", False))
    if args.command == "verify":
        return {"verified": True, "source_manifest_sha256": corpus.manifest_sha256, "split_counts": corpus.split_counts}
    if args.command == "audit":
        from .audit import audit_labels
        return audit_labels(config, corpus)
    if args.command == "train":
        from .training import train_run
        return train_run(config, corpus, args.run_name)
    if args.command == "audit-baseline":
        from .implementation_audit import audit_baseline
        return audit_baseline(config, corpus, args.plan)
    if args.command == "review-errors":
        from .error_review import review_errors
        return review_errors(config, corpus, args.plan)
    if args.command == "prepare-encoder":
        from .encoder import prepare_encoder
        from .implementation_audit import freeze_plan
        plan, _, _ = freeze_plan(config, corpus, args.plan)
        return prepare_encoder(plan["encoder"], config.path("cache_path"))
    if args.command == "experiment":
        from .experiment import run_representation
        names = ["word", "word_char", "embedding"] if args.representation == "all" else [args.representation]
        def progress(value):
            print(canonical(value).decode("utf-8"), file=sys.stderr, flush=True)
        return {name: run_representation(config, corpus, args.plan, name, progress) for name in names}
    if args.command == "compare":
        from .experiment import summarize_comparison
        return summarize_comparison(config, corpus, args.plan)
    if args.command == "evaluate":
        from .evaluation import evaluate_run
        return evaluate_run(config, corpus, args.run, args.split, args.final)
    if args.command == "prepare-contexts":
        from .contexts import prepare_pilot
        return prepare_pilot(config, corpus)
    if args.command == "context-input":
        return corpus.context_sources(args.problem_id)
    from .contexts import validate_context
    from .labels import Taxonomy
    taxonomy = Taxonomy.load(config.path("taxonomy_path"))
    document = read_json(args.file)
    sources = read_json(args.illustrative_sources) if args.illustrative_sources else None
    validate_context(document, taxonomy, corpus, sources)
    if args.command == "validate-context":
        return {"valid": True, "status": document["status"], "human_reviewed": document["status"] == "human_reviewed"}
    if not args.reviewer.strip() or args.output.exists():
        raise ModelingError("Review requires a reviewer and a new output path")
    document["status"] = "human_reviewed" if args.decision == "approved" else "rejected"
    document["review"] = {"decision": args.decision, "reviewer": args.reviewer, "reviewed_at": utcnow().isoformat()}
    validate_context(document, taxonomy, corpus, sources)
    atomic_json(args.output, document)
    return {"review_file": str(args.output.resolve()), "status": document["status"]}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
        print(canonical(result).decode("utf-8"))
        return 0
    except (ModelingError, OSError, KeyError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
