"""Scripted QC flags for proposed records (deterministic, origin 'script') and the add-only merge
of flags an LLM may propose (origin 'llm'). Flags never change a value; they are shown to the
human who decides. Shape: {code, severity warn|error, message, field, group, origin}.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import numbers_in

MAX_CODE_CHARS, MAX_MESSAGE_CHARS = 40, 300
_SE_RE = re.compile(r"(^|[_\s])(se|sd|sem|cv|rse)($|[_\s%])")
_LOWER_RE = re.compile(r"^(?P<stem>.*?)[_\s]?(lower|low|lo|lcl)$")
_UPPER_RE = re.compile(r"^(?P<stem>.*?)[_\s]?(upper|high|hi|ucl)$")
_UNIT_SYNONYMS = ((re.compile(r"\b(hrs?|hours?)\b"), "h"), (re.compile(r"\b(liters?|litres?)\b"), "l"),
                  (re.compile(r"\bmcg\b"), "ug"), (re.compile(r"\b(seconds?|sec)\b"), "s"))
_UNIT_STRIP = re.compile(r"[\s·*×.]+")
_NUMERIC_TYPES = ("number", "integer")

Key = tuple[str, str]            # (source_id, group)


def _flag(code: str, severity: str, message: str, field: str, group: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "field": field, "group": group,
            "origin": "script"}


def normalise_unit(unit: str | None) -> str:
    """Comparison form: NFKC, casefold, µ/μ -> 'u', hr -> h, separators (space · * × .) removed."""
    u = unicodedata.normalize("NFKC", unit or "").casefold().replace("μ", "u").replace("µ", "u")
    for pattern, rep in _UNIT_SYNONYMS:
        u = pattern.sub(rep, u)
    return _UNIT_STRIP.sub("", u)


def _field_index(template: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """casefolded name or alias -> field spec (the spec's own 'name' is the canonical field name)."""
    index: dict[str, dict[str, Any]] = {}
    for spec in (template or {}).get("fields") or []:
        name = str(spec.get("name") or "")
        if not name:
            continue
        for key in [name, *(spec.get("aliases") or [])]:
            index.setdefault(str(key).casefold(), spec)
    return index


def _canon(rec: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    spec = index.get(str(rec.get("field") or "").casefold())
    return str(spec["name"]) if spec else str(rec.get("field") or "")


def group_key(group: Any) -> str:
    """Comparison form of a group label: NFKC, casefold, whitespace collapsed and trimmed."""
    return " ".join(unicodedata.normalize("NFKC", str(group or "")).casefold().split())


def _key(rec: dict[str, Any]) -> Key:
    return str(rec.get("source_id") or ""), group_key(rec.get("group_label"))


def _is_int(x: float | None) -> bool:
    return x is not None and float(x).is_integer()


def _field_flags(rec: dict[str, Any], spec: dict[str, Any], field: str, group: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    ftype, num, value = spec.get("type") or "number", rec.get("value_num"), str(rec.get("value") or "")
    if ftype in _NUMERIC_TYPES:
        if num is None:
            flags.append(_flag(M.FLAG_NON_NUMERIC, "error", f"{field}: value {value!r} is not a number", field, group))
        if (ftype == "integer" or field.casefold() == "n") and num is not None and not _is_int(num):
            flags.append(_flag(M.FLAG_NON_INTEGER_N, "error", f"{field}: {value} is not an integer count",
                               field, group))
        lo, hi = spec.get("min"), spec.get("max")
        if num is not None and lo is not None and num < lo:
            flags.append(_flag(M.FLAG_OUT_OF_RANGE, "warn", f"{field}: {value} below min {lo}", field, group))
        if num is not None and hi is not None and num > hi:
            flags.append(_flag(M.FLAG_OUT_OF_RANGE, "warn", f"{field}: {value} above max {hi}", field, group))
        style = (rec.get("locator") or {}).get("decimal_style") or "."
        if len(numbers_in(value, style)) > 1:
            flags.append(_flag(M.FLAG_COMPOSITE_CELL, "warn",
                               f"{field}: value {value!r} holds more than one number; keep the estimate only",
                               field, group))
        if _SE_RE.search(field.casefold()) and num is not None and num <= 0:
            flags.append(_flag(M.FLAG_SE_NONPOSITIVE, "warn", f"{field}: dispersion {value} must be > 0", field, group))
    want = spec.get("unit")
    if want:
        have = rec.get("unit")
        if not have:
            flags.append(_flag(M.FLAG_UNIT_MISSING, "warn", f"{field}: no unit (template expects {want})",
                               field, group))
        elif normalise_unit(have) != normalise_unit(want):
            flags.append(_flag(M.FLAG_UNIT_MISMATCH, "warn", f"{field}: unit {have!r} differs from template {want!r}",
                               field, group))
    return flags


def _ci_flag(rec: dict[str, Any], field: str, group: str, peers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """ci_inverted on both bounds when <stem>_lower > <stem>_upper within the same source x group."""
    lower, upper = _LOWER_RE.match(field.casefold()), _UPPER_RE.match(field.casefold())
    if not (lower or upper) or rec.get("value_num") is None:
        return []
    stem = (lower or upper).group("stem")
    partner_re = _UPPER_RE if lower else _LOWER_RE
    for name, other in peers.items():
        m = partner_re.match(name.casefold())
        if m and m.group("stem") == stem and other.get("value_num") is not None:
            lo, hi = (rec, other) if lower else (other, rec)
            if lo["value_num"] > hi["value_num"]:
                msg = f"{_canon_name(lo)} {lo.get('value')} > {_canon_name(hi)} {hi.get('value')}"
                return [_flag(M.FLAG_CI_INVERTED, "error", msg, field, group)]
    return []


def _canon_name(rec: dict[str, Any]) -> str:
    return str(rec.get("_canon") or rec.get("field") or "")


def _record_flags(rec: dict[str, Any], field: str, group: str, index: dict[str, dict[str, Any]],
                  present: set[str], approved: dict[tuple[str, str, str], dict[str, Any]],
                  peers: dict[str, dict[str, Any]], required: list[str],
                  batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if rec.get("verify_status") == "fail":
        flags.append(_flag(M.FLAG_VERIFY_FAIL, "error",
                           f"{field}: verification failed ({rec.get('verify_reason') or 'unknown'})", field, group))
    spec = index.get(str(rec.get("field") or "").casefold())
    if spec:
        flags.extend(_field_flags(rec, spec, field, group))
    prior = approved.get((rec.get("source_id") or "", field.casefold(), group_key(group)))
    if prior is not None and not same_value(prior, rec):
        flags.append(_flag(M.FLAG_DUPLICATE, "warn",
                           f"{field}: an approved record already holds {prior.get('value')!r}"
                           f" {prior.get('unit') or ''}".rstrip() + f" (proposed {rec.get('value')!r})", field, group))
    rivals = [o for o in batch if o is not rec and not same_value(o, rec)]
    if rivals:
        flags.append(_flag(M.FLAG_DUPLICATE, "warn",
                           f"{field}: conflicting values proposed for this source/group (this or an earlier run): "
                           + ", ".join(repr(str(o.get("value"))) for o in [rec, *rivals][:4]), field, group))
    flags.extend(_ci_flag(rec, field, group, peers))
    cell = (rec.get("locator") or {}).get("cell") or {}
    if (rec.get("locator") or {}).get("kind") == "table" and cell.get("strategy") == "layout":
        flags.append(_flag(M.FLAG_TABLE_LOW_CONFIDENCE, "warn",
                           f"{field}: read from an unruled (layout-recovered) table; check the cell", field, group))
    for name in required:
        if name not in present:
            flags.append(_flag(M.FLAG_REQUIRED_MISSING, "warn",
                               f"required field {name} has no record for this source/group", name, group))
    return flags


def same_value(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("value_num") is not None and b.get("value_num") is not None:
        return float(a["value_num"]) == float(b["value_num"])
    return str(a.get("value") or "").strip() == str(b.get("value") or "").strip()


def slot_key(rec: Any, template: dict[str, Any]) -> tuple[str, str, str]:
    """(source, canonical field casefolded, normalised group): the slot one value belongs to."""
    r = M.as_mapping(rec)
    return str(r.get("source_id") or ""), _canon(r, _field_index(template)).casefold(), group_key(r.get("group_label"))


def scripted_flags(records: list[Any], template: dict[str, Any], approved_records: list[Any],
                   live: list[Any] | None = None) -> list[list[dict]]:
    """One flag list per input record (same order). *live* holds other still-proposed records (an
    earlier run's) that a record must not contradict; they are compared, never flagged here. Inputs
    are read only; nothing is mutated."""
    index = _field_index(template)
    required = [str(f["name"]) for f in (template or {}).get("fields") or [] if f.get("name") and f.get("required")]
    recs = [{**M.as_mapping(r)} for r in records]
    approved_list = [{**M.as_mapping(r)} for r in approved_records or []]
    live_list = [{**M.as_mapping(r)} for r in live or []]
    for r in recs + approved_list + live_list:
        r["_canon"] = _canon(r, index)
    present: dict[Key, set[str]] = defaultdict(set)
    by_source: dict[str, set[str]] = defaultdict(set)
    peers: dict[Key, dict[str, dict[str, Any]]] = defaultdict(dict)
    batch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs + approved_list:
        present[_key(r)].add(r["_canon"])
        by_source[_key(r)[0]].add(r["_canon"])
        peers[_key(r)].setdefault(r["_canon"], r)
    for r in recs + live_list:
        batch[(*_key(r), r["_canon"].casefold())].append(r)
    approved = {(str(r.get("source_id") or ""), r["_canon"].casefold(), _key(r)[1]): r for r in approved_list}
    # an ungrouped record ('' group: a text-rule n or dose) is checked against its whole source
    return [_record_flags(r, r["_canon"], str(r.get("group_label") or ""), index,
                          by_source[_key(r)[0]] if not _key(r)[1] else present[_key(r)], approved,
                          peers[_key(r)], required, batch[(*_key(r), r["_canon"].casefold())])
            for r in recs]


def merge_llm_flags(existing: list[dict[str, Any]] | None, proposed: list[Any] | None) -> list[dict[str, Any]]:
    """Scripted flags are kept verbatim; well-formed LLM flags are appended with origin 'llm' (add-only)."""
    merged = [dict(f) for f in existing or []]
    seen = {(f.get("code"), f.get("field"), f.get("group")) for f in merged}
    for p in proposed or []:
        if not isinstance(p, dict):
            continue
        code = str(p.get("code") or "").strip()[:MAX_CODE_CHARS]
        message = str(p.get("message") or "").strip()[:MAX_MESSAGE_CHARS]
        severity = p.get("severity")
        if not code or not message or severity not in M.FLAG_SEVERITIES:
            continue
        key = (code, str(p.get("field") or ""), str(p.get("group") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"code": code, "severity": severity, "message": message, "field": key[1], "group": key[2],
                       "origin": "llm"})
    return merged
