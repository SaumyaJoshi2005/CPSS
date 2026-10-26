"""One schema shared by application validation and strict MongoDB validators."""
from __future__ import annotations

from datetime import datetime

from bson import BSON, Binary, ObjectId
from jsonschema import Draft202012Validator, ValidationError, validators

from .core import CorpusError

MAX_BSON = 15 * 1024 * 1024  # Conservative headroom below the server's 16 MiB limit.


def obj(properties, required=(), **extra):
    result = {"bsonType": "object", "properties": properties,
              "additionalProperties": False, **extra}
    if required:
        result["required"] = list(required)
    return result


def string(**kw):
    return {"bsonType": "string", **kw}


def enum(*values):
    return {"enum": list(values)}


def array(items, maximum=256):
    return {"bsonType": "array", "items": items, "maxItems": maximum}


TEXT = string(minLength=1)
HASH = string(pattern="^[a-f0-9]{64}$")
DATE = {"bsonType": "date"}
INT = {"bsonType": "int", "minimum": 0}
NUMBER = {"bsonType": ["double", "int"], "minimum": 0, "maximum": 1}
STRINGS = array(TEXT)
QUALITY = obj({"statement": enum("missing", "ready"),
               "editorial": enum("missing", "candidate", "ready"),
               "solutions": enum("not_requested", "missing", "ready"),
               "labels": enum("unreviewed", "reviewed"), "issue_codes": STRINGS},
              ("statement", "editorial", "solutions", "labels", "issue_codes"))
STATEMENT = obj({"source_document_id": TEXT, "version_id": HASH, "clean_sha256": HASH,
                 "clean_text": TEXT, "language": enum("en"), "observation_seq": INT},
                ("source_document_id", "version_id", "clean_sha256", "clean_text", "language", "observation_seq"))
LABELS = obj({"taxonomy_version": TEXT, "broad": STRINGS, "techniques": STRINGS,
              "annotation_ids": array(HASH), "projection_revision": INT},
             ("taxonomy_version", "broad", "techniques", "annotation_ids", "projection_revision"))
PROVENANCE = obj({"url": TEXT, "retrieved_at": DATE, "method": TEXT,
                  "status_code": INT, "attribution": string(), "import_path": string()},
                 ("url", "retrieved_at", "method", "status_code"))
SECTION = obj({"key": TEXT, "text": TEXT}, ("key", "text"))
EVIDENCE = obj({"version_id": HASH, "locator": TEXT, "span": string(maxLength=1000)},
               ("version_id", "locator", "span"))


def document(properties, required):
    return obj({"_id": TEXT, "schema_version": enum(1), **properties},
               ("_id", "schema_version", *required))


SCHEMAS = {
    "problems": document({
        "source": enum("codeforces"), "contest_id": INT, "problem_index": TEXT,
        "title": TEXT, "canonical_url": TEXT,
        "contest": obj({"name": TEXT, "started_at": DATE, "phase": enum("FINISHED")},
                       ("name", "started_at", "phase")),
        "original_tags": STRINGS, "rating": INT, "statement": STATEMENT,
        "approved_labels": LABELS, "primary_editorial_link_id": {"bsonType": ["string", "null"]},
        "quality": QUALITY, "metadata_fingerprint": HASH, "metadata_checked_at": DATE,
        "created_at": DATE, "updated_at": DATE, "duplicate_group": TEXT,
        "corpus": obj({"provider": enum("open-r1"), "repository": TEXT, "revision": string(pattern="^[a-f0-9]{40}$"),
                       "source_split": enum("train", "test"), "row_id": TEXT, "aliases": STRINGS,
                       "contest_type": enum("CF", "ICPC", "IOI"),
                       "time_limit_seconds": {"bsonType": ["double", "int"], "minimum": 0},
                       "memory_limit_mb": {"bsonType": ["double", "int"], "minimum": 0},
                       "input_mode": enum("stdio", "file"),
                       "examples": array(obj({"input": string(), "output": TEXT}, ("input", "output")), 256),
                       "complete": enum(True), "accepted_solution_ids": STRINGS,
                       "official_tests_complete": {"bsonType": "bool"}, "upstream_executable": {"bsonType": "bool"},
                       "testset_size": INT},
                      ("provider", "repository", "revision", "source_split", "row_id", "aliases",
                       "time_limit_seconds", "memory_limit_mb", "input_mode", "examples", "complete", "accepted_solution_ids")),
    }, ("source", "contest_id", "problem_index", "title", "canonical_url", "contest", "original_tags",
        "approved_labels", "primary_editorial_link_id", "quality", "metadata_fingerprint",
        "metadata_checked_at", "created_at", "updated_at")),
    "source_documents": document({
        "kind": enum("metadata_catalog", "statement", "editorial", "solution"),
        "canonical_url": TEXT, "language": enum("en"), "problem_id": TEXT, "blog_id": INT,
        "attribution": string(), "observation_seq": INT, "published_seq": INT,
        "parent_version_id": HASH, "programming_language": TEXT,
        "submission_url": TEXT, "submission_id": INT, "collection_method": TEXT,
        "dataset": obj({"repository": TEXT, "revision": string(pattern="^[a-f0-9]{40}$"), "split": TEXT, "file": TEXT, "row_id": TEXT},
                       ("repository", "revision", "split", "file", "row_id")),
        "current_version_id": HASH, "current_raw_sha256": HASH,
        "last_checked_at": DATE, "last_success_at": DATE,
        "last_error": string(maxLength=128), "stale": {"bsonType": "bool"},
        "created_at": DATE,
    }, ("kind", "canonical_url", "language", "observation_seq", "published_seq", "created_at")),
    "blobs": document({
        "byte_length": INT, "storage": enum("inline", "gridfs"), "body": {"bsonType": "binData"},
        "gridfs_id": HASH, "created_at": DATE,
    }, ("byte_length", "storage", "created_at")),
    "document_versions": document({
        "source_document_id": TEXT, "raw_sha256": HASH, "processor_fingerprint": HASH,
        "processor_config": {"bsonType": "object"},
        "extractor_version": TEXT, "cleaner_version": TEXT, "language": enum("en"),
        "sections": array(SECTION, 4096), "clean_text": TEXT, "clean_sha256": HASH,
        "created_at": DATE, "provenance": PROVENANCE, "warnings": STRINGS,
    }, ("source_document_id", "raw_sha256", "processor_fingerprint", "extractor_version", "cleaner_version",
        "language", "sections", "clean_text", "clean_sha256", "created_at", "provenance", "warnings")),
    "editorial_links": document({
        "problem_id": TEXT, "editorial_source_document_id": TEXT, "editorial_version_id": HASH,
        "section_key": TEXT, "section_text": TEXT, "mapping_method": enum("explicit_link", "curated", "heading", "dataset_id"),
        "confidence": NUMBER, "review_status": enum("proposed", "approved", "rejected"), "created_at": DATE,
    }, ("problem_id", "editorial_source_document_id", "editorial_version_id", "section_key", "section_text",
        "mapping_method", "confidence", "review_status", "created_at")),
    "taxonomy": document({
        "taxonomy_version": TEXT, "display_name": TEXT, "kind": enum("broad", "technique", "pattern"),
        "parent_ids": STRINGS, "aliases": STRINGS, "active": {"bsonType": "bool"},
    }, ("taxonomy_version", "display_name", "kind", "parent_ids", "aliases", "active")),
    "annotations": document({
        "problem_id": TEXT, "label_id": TEXT, "taxonomy_version": TEXT,
        "origin": enum("platform_map", "editorial", "human", "model"), "producer_version": TEXT,
        "evidence": EVIDENCE, "confidence": NUMBER, "confidence_policy": TEXT,
        "status": enum("proposed", "approved", "rejected"), "reviewer": string(maxLength=256),
        "created_at": DATE, "updated_at": DATE,
    }, ("problem_id", "label_id", "taxonomy_version", "origin", "producer_version", "evidence",
        "confidence_policy", "status", "reviewer", "created_at", "updated_at")),
    "ingestion_jobs": document({
        "run_id": TEXT, "stage": enum("statement", "editorial", "solution"), "entity_id": TEXT,
        "status": enum("pending", "running", "retry", "blocked", "failed", "succeeded"),
        "attempts": INT, "next_attempt_at": DATE,
        "checkpoint": obj({"observation_seq": INT, "source_document_id": TEXT, "blog_ids": array(INT, 64),
                           "raw_sha256": HASH, "retrieved_at": DATE, "url": TEXT, "method": TEXT,
                           "completed": {"bsonType": "bool"}}, ()),
        "last_error": string(maxLength=128), "updated_at": DATE,
    }, ("run_id", "stage", "entity_id", "status", "attempts", "next_attempt_at", "checkpoint", "updated_at")),
    "ingestion_runs": document({
        "started_at": DATE, "completed_at": DATE, "status": enum("running", "completed", "partial", "failed"),
        "config_fingerprint": HASH, "config": {"bsonType": "object"},
        "counts": {"bsonType": "object"}, "export_refs": array(TEXT, 64),
    }, ("started_at", "status", "config_fingerprint", "config", "counts", "export_refs")),
}

SCHEMAS["blobs"]["properties"]["_id"] = HASH
for _name in ("document_versions", "editorial_links", "annotations", "ingestion_jobs"):
    SCHEMAS[_name]["properties"]["_id"] = HASH
SCHEMAS["problems"]["properties"]["_id"] = string(pattern="^codeforces:[1-9][0-9]*:[A-Z][0-9]*$")
SCHEMAS["problems"]["properties"]["problem_index"] = string(pattern="^[A-Z][0-9]*$")
SCHEMAS["problems"]["properties"]["contest_id"] = {"bsonType": "int", "minimum": 1}
SCHEMAS["blobs"]["oneOf"] = [
    {"properties": {"storage": enum("inline")}, "required": ["body"], "not": {"required": ["gridfs_id"]}},
    {"properties": {"storage": enum("gridfs")}, "required": ["gridfs_id"], "not": {"required": ["body"]}},
]
SCHEMAS["problems"]["anyOf"] = [
    {"properties": {"quality": {"properties": {"statement": enum("ready")}}}, "required": ["statement"]},
    {"properties": {"quality": {"properties": {"statement": enum("missing")}}}, "not": {"required": ["statement"]}},
]


def bson_type(validator, expected, instance, schema):
    types = expected if isinstance(expected, list) else [expected]
    matches = {
        "object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str), "int": lambda x: type(x) is int and -(2**31) <= x < 2**31,
        "double": lambda x: type(x) is float, "date": lambda x: isinstance(x, datetime),
        "binData": lambda x: isinstance(x, (bytes, Binary)), "objectId": lambda x: isinstance(x, ObjectId),
        "null": lambda x: x is None, "bool": lambda x: type(x) is bool,
    }
    if not any(matches[t](instance) for t in types):
        yield ValidationError("invalid BSON type")


AppValidator = validators.extend(Draft202012Validator, {"bsonType": bson_type})


def validate(collection: str, record: dict):
    try:
        AppValidator(SCHEMAS[collection]).validate(record)
    except ValidationError:
        raise CorpusError(f"invalid_{collection}_document") from None
    if len(BSON.encode(record)) > MAX_BSON:
        raise CorpusError("oversize_requires_chunking")
    return record


INDEXES = {
    "source_documents": [([("problem_id", 1), ("kind", 1), ("parent_version_id", 1)], "problem_sources", False)],
    "problems": [([("source", 1), ("contest_id", 1), ("problem_index", 1)], "problem_identity", True),
                 ([("quality.statement", 1), ("rating", 1)], "ready_rating", False),
                 ([("approved_labels.broad", 1), ("rating", 1)], "broad_rating", False),
                 ([("approved_labels.techniques", 1), ("rating", 1)], "technique_rating", False),
                 ([("contest.started_at", -1), ("contest_id", 1)], "chronological_export", False)],
    "document_versions": [([("source_document_id", 1), ("raw_sha256", 1), ("processor_fingerprint", 1)], "version_identity", True),
                          ([("clean_sha256", 1)], "clean_hash_lookup", False)],
    "editorial_links": [([("problem_id", 1), ("editorial_version_id", 1), ("section_key", 1)], "editorial_section_identity", True)],
    "annotations": [([("problem_id", 1), ("status", 1), ("taxonomy_version", 1)], "problem_annotation_review", False)],
    "ingestion_jobs": [([("run_id", 1), ("status", 1), ("next_attempt_at", 1)], "runnable_jobs", False)],
}
