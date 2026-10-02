"""Evidence agent tool: export_evidence_dataset — the APPROVED records as two CSV files.

Long: one row per approved record with its locator and provenance (record_hash, source sha256);
'value' is the page's literal, 'value_num' the same number in canonical point-decimal form and
'decimal_style' the page's convention, so '3,412' (3412) and '18,2' (18.2) cannot be confused.
Wide: one row per source x group; <field> is the canonical number (text for a text field),
<field>_literal the page literal and <field>_unit the unit.
Only status='approved' rows of publications included under the current criteria leave the ledger
(approved rows of an excluded publication are left out and listed), so every exported number was
found literally on its cited page by the verifier and then accepted by a human. Row and column order
are deterministic, so an identical approved set yields byte-identical files (same sha256, same
dataset id); file names carry a sha256 prefix and are created without replacing, so no export
overwrites another's file. Cells that start with = + - @ (or tab / CR) get a leading apostrophe
(spreadsheet formula guard). The ledger row names the authenticated caller (ToolContext.actor).
"""
from __future__ import annotations

import csv
import math
import os
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.pharmstate import PharmState
from app.core.provenance import file_sha256
from app.evidence import ingest
from app.evidence.config_tools import summarise
from app.evidence.inclusion import current_screens, exclusion_reason
from app.evidence.models import Dataset, Record, Source
from app.evidence.store import EvidenceStore, get_store
from app.tools.base import Tool, ToolContext, ToolResult

__all__ = ["LONG_COLUMNS", "WIDE_FIXED", "csv_cell", "num_text", "build_long", "build_wide", "write_csv",
           "export_evidence_dataset", "TOOLS"]

AGENT = "evidence"
LONG_COLUMNS = ("record_id", "source_id", "source_sha256", "filename", "title", "field", "group", "value",
                "value_num", "decimal_style", "unit", "page", "kind", "snippet", "literal", "extractor",
                "verify_status", "flags", "record_hash", "decided_by", "decided_at")
WIDE_FIXED = ("source_id", "source_sha256", "filename", "group")
FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")   # OWASP CSV-injection leaders
MULTI_SEP = " | "          # several approved values for one (source, group, field) in the wide file


def csv_cell(value: Any) -> str:
    """Stringify a cell; a leading formula character gets an apostrophe so no spreadsheet runs it."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return f"'{text}" if text.startswith(FORMULA_LEADERS) else text


def num_text(value: float | None) -> str:
    """Canonical point-decimal text of a verified number ('3412', '18.2', '-20'); '' when absent.
    Written without the formula guard: a bare number cannot be a spreadsheet formula."""
    if value is None or not math.isfinite(float(value)):
        return ""
    v = float(value)
    return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


def exports_dir(data_dir: str | Path | None) -> Path:
    d = ingest.evidence_dir(data_dir) / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- builders ---

def _approved(records: list[Record]) -> list[Record]:
    return sorted((r for r in records if r.status == "approved"),
                  key=lambda r: (r.source_id, r.group_label, r.field, r.record_hash))


def _source_key(sources: dict[str, Source]):
    return lambda r: ((sources[r.source_id].filename if r.source_id in sources else ""), r.source_id,
                      r.group_label, r.field, r.record_hash)


def build_long(records: list[Record], sources: dict[str, Source]) -> list[dict[str, str]]:
    """Approved records only, ordered by (filename, source, group, field, record_hash)."""
    rows = []
    for r in sorted(_approved(records), key=_source_key(sources)):
        src, loc = sources.get(r.source_id), r.locator or {}
        row = {"record_id": r.id, "source_id": r.source_id,
               "source_sha256": src.sha256 if src else "", "filename": src.filename if src else "",
               "title": (src.title if src else None) or "", "field": r.field, "group": r.group_label,
               "value": r.value, "decimal_style": loc.get("decimal_style") or (src.decimal_style if src else "."),
               "unit": r.unit, "page": loc.get("page"), "kind": loc.get("kind"),
               "snippet": loc.get("snippet"), "literal": loc.get("literal"), "extractor": r.extractor,
               "verify_status": r.verify_status, "flags": ";".join(f.get("code", "") for f in r.flags),
               "record_hash": r.record_hash, "decided_by": r.decided_by, "decided_at": r.decided_at}
        rows.append({k: num_text(r.value_num) if k == "value_num" else csv_cell(row[k]) for k in LONG_COLUMNS})
    return rows


def wide_columns(records: list[Record], template: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """Template field order first, then (sorted) fields present in approved records but not in the template."""
    names = [str(f.get("name")) for f in (template or {}).get("fields") or [] if f.get("name")]
    extra = sorted({r.field for r in records} - set(names))
    fields = list(dict.fromkeys(names + extra))
    return fields, list(WIDE_FIXED) + [c for n in fields for c in (n, f"{n}_literal", f"{n}_unit")]


def _wide_value(values: list[str]) -> str:
    """One number stays bare (it cannot be a formula); several are joined and guarded like any text."""
    return values[0] if len(values) == 1 and values[0] == num_text(_as_float(values[0])) else \
        csv_cell(MULTI_SEP.join(values))


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def build_wide(records: list[Record], sources: dict[str, Source],
               template: dict[str, Any] | None) -> tuple[list[dict[str, str]], list[str]]:
    """One row per (source, group); several approved values for one field are joined with ' | '."""
    approved = sorted(_approved(records), key=_source_key(sources))
    fields, columns = wide_columns(approved, template)
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for r in approved:
        cell = groups.setdefault((r.source_id, r.group_label), {})
        cell.setdefault(r.field, []).append(num_text(r.value_num) if r.value_num is not None else r.value)
        cell.setdefault(f"{r.field}_literal", []).append(r.value)
        cell.setdefault(f"{r.field}_unit", []).append(r.unit or "")
    rows = []
    for (src_id, group), cells in groups.items():
        src = sources.get(src_id)
        row = {"source_id": src_id, "source_sha256": src.sha256 if src else "",
               "filename": src.filename if src else "", "group": group}
        for f in fields:
            row[f"{f}_literal"] = MULTI_SEP.join(cells.get(f"{f}_literal", []))
            row[f"{f}_unit"] = MULTI_SEP.join(dict.fromkeys(cells.get(f"{f}_unit", [])))
        out = {k: csv_cell(row[k]) for k in columns if k in row}
        out.update({f: _wide_value(cells[f]) if cells.get(f) else "" for f in fields})
        rows.append({k: out[k] for k in columns})
    return rows, columns


def write_csv(path: Path, columns: list[str] | tuple[str, ...], rows: list[dict[str, str]]) -> str:
    """Write atomically (tmp + rename) and return the file's sha256."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return file_sha256(path)


def write_named_csv(out_dir: Path, stem: str, columns: list[str] | tuple[str, ...], rows: list[dict[str, str]],
                    *, wide: bool = False) -> tuple[Path, str]:
    """Write to a private temp name, then publish without ever replacing another export's file (two
    exports in one second, two tabs): the long file as <stem>_<sha256[:12]>.csv, the wide one beside
    it as <long stem>_wide.csv, or <long stem>_<its sha256[:12]>_wide.csv when that name holds other
    bytes (same long content, another template). An existing file of the chosen name holds the same
    content and is kept; the temp file is dropped."""
    tmp = out_dir / f".{stem}.{uuid.uuid4().hex}.tmp"
    sha = write_csv(tmp, columns, rows)
    names = [f"{stem}_wide.csv", f"{stem}_{sha[:12]}_wide.csv"] if wide else [f"{stem}_{sha[:12]}.csv"]
    names.append(f"{stem}_{sha}{'_wide' if wide else ''}.csv")
    try:
        for name in names:
            path = out_dir / name
            if _publish(tmp, path) or file_sha256(path) == sha:
                return path, sha
        raise ValueError(f"could not name the export file {names[-1]}")
    finally:
        tmp.unlink(missing_ok=True)


def _publish(tmp: Path, path: Path) -> bool:
    """Create *path* with *tmp*'s bytes atomically, never replacing a file; False when *path* exists."""
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:                              # no hard links on this filesystem
        if path.exists():
            return False
        os.replace(tmp, path)
        return True


# -------------------------------------------------------------------- tool ---

def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _dedupe_files(ds: Dataset, long_path: Path, wide_path: Path) -> None:
    """An identical approved set was exported before (same long sha256 -> same ledger row): the
    ledger's paths win. A new file replaces a missing ledger file in the same exports dir only when
    its sha256 equals the ledger's (the wide file also depends on the template), else it is dropped."""
    pairs = ((long_path, Path(ds.long_path), ds.long_sha256), (wide_path, Path(ds.wide_path), ds.wide_sha256))
    for new, old, sha in pairs:
        if new == old:
            continue
        if (not old.is_file() and old.parent.resolve() == new.parent.resolve()
                and file_sha256(new) == sha):
            os.replace(new, old)
        else:
            new.unlink(missing_ok=True)


def _split_included(store: EvidenceStore, sid: str, approved: list[Record]
                    ) -> tuple[list[Record], list[dict[str, str]]]:
    """(approved records of sources included under the current criteria, [{record_id, source_id,
    reason}] for the approved records of sources that are not, or that did not pass verification:
    a hand-edited ledger never ships a number the verifier did not find)."""
    screens = current_screens(store, sid)
    keep, left_out = [], []
    for r in approved:
        why = (f"verify_status is {r.verify_status!r}, not 'pass'" if r.verify_status != "pass"
               else exclusion_reason(screens, r.source_id))
        if why is None:
            keep.append(r)
        else:
            left_out.append({"record_id": r.id, "source_id": r.source_id, "reason": why})
    return keep, left_out


def export_evidence_dataset(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store: EvidenceStore = get_store()
    sid = state.session_id
    approved = _approved(store.list_records(sid, status="approved"))
    if not approved:
        raise ValueError("no approved records to export — approve extracted records in the Evidence workspace first")
    approved, left_out = _split_included(store, sid, approved)
    if not approved:
        raise ValueError(f"every approved record ({len(left_out)}) comes from a publication that is not currently "
                         "included under the current criteria; screen and include it first")
    actor = ctx.actor or "anonymous"               # the authenticated caller; args['actor'] is ignored
    sources = {s.id: s for s in store.list_sources(sid)}
    tmpl = store.get_config(sid, "template")
    template, thash = (tmpl["body"], tmpl["body_hash"]) if tmpl else (None, "")
    long_rows = build_long(approved, sources)
    wide_rows, wide_cols = build_wide(approved, sources, template)
    out_dir = exports_dir(ctx.data_dir)
    stem = f"evidence_{sid}_{_stamp()}"
    long_path, long_sha = write_named_csv(out_dir, stem, LONG_COLUMNS, long_rows)
    wide_path, wide_sha = write_named_csv(out_dir, long_path.stem, wide_cols, wide_rows, wide=True)
    hashes = sorted(r.record_hash for r in approved)
    n_sources = len({r.source_id for r in approved})
    ds = store.add_dataset(sid, thash, str(long_path), str(wide_path), long_sha, wide_sha, hashes,
                           n_records=len(approved), n_sources=n_sources, exported_by=actor)
    _dedupe_files(ds, long_path, wide_path)
    dropped = (f" {len(left_out)} approved record(s) of publications not included under the current criteria "
               "were left out." if left_out else "")
    return ToolResult(
        summary=(f"Exported {len(approved)} approved record(s) from {n_sources} publication(s): "
                 f"{Path(ds.long_path).name} (long) and {Path(ds.wide_path).name} (wide), dataset {ds.id}."
                 + dropped),
        writes={"evidence_dataset_path": ds.long_path, "evidence_summary": summarise(store, sid)},
        result={"dataset": asdict(ds), "long_path": ds.long_path, "wide_path": ds.wide_path,
                "long_sha256": ds.long_sha256, "wide_sha256": ds.wide_sha256, "record_hashes": hashes,
                "n_records": len(approved), "n_sources": n_sources, "template_hash": thash,
                "long_columns": list(LONG_COLUMNS), "wide_columns": wide_cols, "left_out": left_out},
        action=(f"exported evidence dataset {ds.id}: {len(approved)} approved records from {n_sources} sources "
                f"(long sha256 {ds.long_sha256[:12]}, wide sha256 {ds.wide_sha256[:12]})"
                + (f"; {len(left_out)} left out (not included)" if left_out else "")))


TOOLS = [
    Tool(name="export_evidence_dataset", agent=AGENT,
         description=("Export the APPROVED evidence records as CSV: a long file (one row per record with its "
                      "page locator, literal, canonical number, decimal style and record hash) and a wide file "
                      "(one row per source and group: per template field the number, its page literal and its "
                      "unit). Proposed or rejected records are never exported. Requires at least one approved "
                      "record."),
         input_schema={"type": "object", "properties": {}},
         run=export_evidence_dataset),
]
