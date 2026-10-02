"""Evidence agent tools: ingest an uploaded PDF and save the criteria / template configs.

Both are deterministic. ``ingest_evidence_source`` only accepts a file the upload route
already placed under data_dir/evidence (content-addressed) and re-checks its sha256, so a
model-issued call can never pull an arbitrary server file into the ledger.
``save_evidence_config`` validates and normalises the body before hashing it; the hash is
what screens and records cite, so history stays interpretable after a re-save.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.core.pharmstate import PharmState
from app.evidence import ingest
from app.evidence.models import CONFIG_KINDS, Source, now_iso
from app.evidence.store import EvidenceStore, get_store
from app.tools.base import Tool, ToolContext, ToolResult

AGENT = "evidence"
MAX_CRITERIA, MAX_CRITERION_TEXT, MAX_ID, MAX_KEYWORDS, MAX_KEYWORD_LEN = 100, 500, 64, 50, 60
MAX_FIELDS, MAX_ALIASES, MAX_DESCRIPTION, MAX_UNIT = 60, 20, 500, 40
CRITERION_KINDS, FIELD_TYPES, DECIMALS = ("inclusion", "exclusion"), ("number", "integer", "text"), ("auto", ".", ",")
FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
#: Keys of a source row exposed to clients (server paths stay server-side).
SOURCE_PUBLIC_KEYS = ("id", "session_id", "sha256", "filename", "pages", "title", "metadata", "text_chars",
                      "parse_status", "decimal_style", "licence_note", "uploaded_by", "uploaded_at")


def source_row(src: Source) -> dict[str, Any]:
    d = asdict(src)
    return {k: d[k] for k in SOURCE_PUBLIC_KEYS}


def summarise(store: EvidenceStore, session_id: str) -> dict[str, Any]:
    """The PharmState mirror: counts and config hashes only; the DB is the ledger."""
    counts = store.counts(session_id)
    statuses = [s.parse_status for s in store.list_sources(session_id)]
    crit, tmpl = store.get_config(session_id, "criteria"), store.get_config(session_id, "template")
    datasets = store.list_datasets(session_id)
    return {"sources": counts["sources"], "parsed": statuses.count("parsed"),
            "no_text": statuses.count("no_text"), "failed": statuses.count("failed"),
            "screens": counts["screens"], "records": counts["records"], "datasets": counts["datasets"],
            "criteria_hash": crit["body_hash"] if crit else None,
            "template_hash": tmpl["body_hash"] if tmpl else None,
            "wide_path": datasets[-1].wide_path if datasets else None, "updated_at": now_iso()}


# ------------------------------------------------------------------ ingest ---

def _str(args: dict[str, Any], key: str, *, required: bool = False, max_len: int = 1000) -> str | None:
    val = args.get(key)
    if val is None or val == "":
        if required:
            raise ValueError(f"'{key}' is required")
        return None
    if not isinstance(val, str):
        raise ValueError(f"'{key}' must be a string")
    return val[:max_len]


def ingest_evidence_source(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    path = Path(_str(args, "path", required=True)).resolve()
    expected = (_str(args, "sha256", required=True) or "").lower()
    ev_dir = ingest.evidence_dir(ctx.data_dir).resolve()
    if not path.is_relative_to(ev_dir) or path.suffix.lower() != ".pdf":
        raise ValueError("path must be a PDF inside the session's evidence directory")
    if not path.is_file():
        raise ValueError("source file missing")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("sha256 does not match the file on disk")
    store = get_store()
    src = ingest.ingest_pdf(data, _str(args, "filename") or path.name, state.session_id,
                            _str(args, "actor") or "anonymous", data_dir=ctx.data_dir, store=store,
                            licence_note=_str(args, "licence_note", max_len=500))
    row = source_row(src)
    return ToolResult(
        summary=(f"Ingested {src.filename}: {src.pages} page(s), {src.text_chars} text chars, "
                 f"parse {src.parse_status}, source {src.id}."),
        writes={"evidence_summary": summarise(store, state.session_id)},
        result={"source": row, "duplicate": False},
        action=f"ingest_evidence_source({src.id}, sha256={src.sha256[:12]}, {src.parse_status})")


# ------------------------------------------------------------------ config ---

def _bool(item: dict[str, Any], key: str, default: bool, where: str) -> bool:
    val = item.get(key, default)
    if not isinstance(val, bool):
        raise ValueError(f"{where}: '{key}' must be true or false")
    return val


def _str_list(item: dict[str, Any], key: str, where: str, max_items: int, max_len: int) -> list[str]:
    vals = item.get(key) or []
    if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
        raise ValueError(f"{where}: '{key}' must be a list of strings")
    if len(vals) > max_items:
        raise ValueError(f"{where}: at most {max_items} '{key}'")
    cleaned = [v.strip()[:max_len] for v in vals if v.strip()]
    return list(dict.fromkeys(cleaned))


def _number_or_none(item: dict[str, Any], key: str, where: str) -> float | None:
    val = item.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{where}: '{key}' must be a number")
    return float(val)


def validate_criteria(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, list) or not body:
        raise ValueError("criteria must be a non-empty list")
    if len(body) > MAX_CRITERIA:
        raise ValueError(f"at most {MAX_CRITERIA} criteria")
    out, seen = [], set()
    for i, item in enumerate(body):
        where = f"criteria[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: must be an object")
        cid = item.get("id")
        if not isinstance(cid, str) or not cid.strip() or len(cid) > MAX_ID:
            raise ValueError(f"{where}: 'id' must be a non-empty string (<= {MAX_ID} chars)")
        if cid in seen:
            raise ValueError(f"{where}: duplicate criterion id '{cid}'")
        seen.add(cid)
        text = item.get("text")
        if not isinstance(text, str) or not (1 <= len(text.strip()) <= MAX_CRITERION_TEXT):
            raise ValueError(f"{where}: 'text' must be 1..{MAX_CRITERION_TEXT} characters")
        kind = item.get("kind")
        if kind not in CRITERION_KINDS:
            raise ValueError(f"{where}: 'kind' must be one of {list(CRITERION_KINDS)}")
        out.append({"id": cid, "text": text.strip(), "kind": kind,
                    "required": _bool(item, "required", False, where),
                    "keywords": _str_list(item, "keywords", where, MAX_KEYWORDS, MAX_KEYWORD_LEN)})
    return out


def _validate_field(item: Any, where: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{where}: must be an object")
    name = item.get("name")
    if not isinstance(name, str) or not FIELD_NAME_RE.match(name):
        raise ValueError(f"{where}: 'name' must match {FIELD_NAME_RE.pattern}")
    ftype = item.get("type", "number")
    if ftype not in FIELD_TYPES:
        raise ValueError(f"{where}: 'type' must be one of {list(FIELD_TYPES)}")
    decimal = item.get("decimal", "auto")
    if decimal not in DECIMALS:
        raise ValueError(f"{where}: 'decimal' must be one of {list(DECIMALS)}")
    unit = item.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError(f"{where}: 'unit' must be a string")
    description = item.get("description") or ""
    if not isinstance(description, str):
        raise ValueError(f"{where}: 'description' must be a string")
    lo, hi = _number_or_none(item, "min", where), _number_or_none(item, "max", where)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"{where}: 'min' must be <= 'max'")
    return {"name": name, "type": ftype, "unit": (unit or "").strip()[:MAX_UNIT] or None,
            "description": description.strip()[:MAX_DESCRIPTION],
            "required": _bool(item, "required", False, where), "min": lo, "max": hi,
            "aliases": _str_list(item, "aliases", where, MAX_ALIASES, MAX_KEYWORD_LEN), "decimal": decimal}


def validate_template(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or not isinstance(body.get("fields"), list) or not body["fields"]:
        raise ValueError("template must be an object with a non-empty 'fields' list")
    fields = body["fields"]
    if len(fields) > MAX_FIELDS:
        raise ValueError(f"at most {MAX_FIELDS} template fields")
    out = [_validate_field(f, f"fields[{i}]") for i, f in enumerate(fields)]
    names = [f["name"] for f in out]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate field names: {dupes}")
    return {"fields": out}


def save_evidence_config(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    kind = args.get("kind")
    if kind not in CONFIG_KINDS:
        raise ValueError(f"'kind' must be one of {list(CONFIG_KINDS)}")
    body = validate_criteria(args.get("body")) if kind == "criteria" else validate_template(args.get("body"))
    store = get_store()
    h = store.save_config(state.session_id, kind, body, _str(args, "actor") or "anonymous")
    n = len(body) if kind == "criteria" else len(body["fields"])
    noun = "criteria" if kind == "criteria" else "template field(s)"
    return ToolResult(summary=f"Saved {n} {noun}; {kind} hash {h[:12]}.",
                      writes={"evidence_summary": summarise(store, state.session_id)},
                      result={"kind": kind, "body_hash": h, "n_items": n, "body": body},
                      action=f"save_evidence_config({kind}, hash={h[:12]}, n={n})")


TOOLS = [
    Tool(name="ingest_evidence_source", agent=AGENT,
         description=("Parse an uploaded PDF already stored under the evidence directory and record it as "
                      "an evidence source (idempotent per sha256). Used by the upload endpoint."),
         input_schema={"type": "object",
                       "properties": {"path": {"type": "string"}, "sha256": {"type": "string"},
                                      "filename": {"type": "string"}, "licence_note": {"type": "string"},
                                      "actor": {"type": "string"}},
                       "required": ["path", "sha256"]},
         run=ingest_evidence_source),
    Tool(name="save_evidence_config", agent=AGENT,
         description=("Validate and save the screening criteria (kind 'criteria': list of {id, text, kind "
                      "inclusion|exclusion, required, keywords}) or the extraction template (kind 'template': "
                      "{fields:[{name, type number|integer|text, unit, description, required, min, max, "
                      "aliases, decimal}]}). Returns the body hash that screens and records cite."),
         input_schema={"type": "object",
                       "properties": {"kind": {"type": "string", "enum": list(CONFIG_KINDS)},
                                      "body": {}, "actor": {"type": "string"}},
                       "required": ["kind", "body"]},
         run=save_evidence_config),
]
