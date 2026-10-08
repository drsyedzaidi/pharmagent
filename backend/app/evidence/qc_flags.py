"""Scripted QC flags for proposed records (deterministic, origin 'script') and the add-only merge
of flags an LLM may propose (origin 'llm'). Flags never change a value; they are shown to the
human who decides. Shape: {code, severity warn|error, message, field, group, origin}.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import numbers_in
from app.evidence.numgrammar.alphabet import OUTSIDE_RE, WORD_RE
from app.evidence.numgrammar.equivalents import THIN, casefold_letters, fold
from app.evidence.unit_norm import normalise_unit, same_unit

__all__ = ["normalise_unit", "same_value", "scripted_flags", "merge_llm_flags", "slot_key", "group_key",
           "is_interval_field", "label_key"]

MAX_CODE_CHARS, MAX_MESSAGE_CHARS = 40, 300
_CODE_RE = re.compile(r"[a-z0-9_]{1,40}")          # a model flag code is an identifier, never text
_SE_RE = re.compile(r"(^|[^a-z])(se|sd|sem|cv|rse)($|[^a-z])")          # 'cmax_se', '%cv', 'sd(cl)'
# A bound names its interval by a suffix ('cl_lower', 'cl_ucl') or a separated prefix ('Lower 95% CI').
_BOUNDS = {"lower": ("lower", "low", "lo", "lcl"), "upper": ("upper", "high", "hi", "ucl")}
_BOUND_RES = {side: (re.compile(r"^(?P<stem>.*?)[_\s]?(" + "|".join(words) + r")$"),
                     re.compile(r"^(" + "|".join(words) + r")[_\s](?P<stem>.+)$")) for side, words in _BOUNDS.items()}
_RANGE_NAME = re.compile(r"(^|[_\s])(range|min|max|minimum|maximum)($|[_\s])")
_NUMERIC_TYPES = ("number", "integer")

Key = tuple[str, str]            # (source_id, group)


def _flag(code: str, severity: str, message: str, field: str, group: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "field": field, "group": group,
            "origin": "script"}


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


def label_key(group: Any) -> str:
    """The EXACT comparison form of a group label (r7 writer 3): the reader's equivalents (``fold``), the case of the
    letters the alphabet lists (``CASE_PAIRS``), runs of the GAP set as one space, trimmed. Nothing else is equated or
    dropped: 'Arm 1\u00b2', 'Arm \u2461', 'Arm \uff11\uff12' and 'Arm 1<U+200B>2' are four groups, never 'Arm 12'. It
    decides which fields a group holds (``required_missing``), so a look-alike or invisible character can only ADD a
    flag."""
    return " ".join(WORD_RE.findall(casefold_letters(fold(str(group or "")))))


def group_key(group: Any) -> str:
    """The MERGING comparison form of a group label (duplicates, conflicting values, interval peers, slots): the exact
    form with the thin space read as a gap and every character outside the readable alphabet dropped (the derived
    allow-list of the reader, ``OUTSIDE_RE``; no NFKC and no Unicode category), so 'Healthy<U+200B>' is 'healthy' and an
    inserted invisible character never dodges the ``duplicate`` flag; 'Arm 1\u00b2' (a readable superscript) stays apart
    from 'Arm 12'. Merging only ever adds a flag here."""
    text = OUTSIDE_RE.sub("", casefold_letters(fold(str(group or ""))).replace(THIN, " "))
    return " ".join(WORD_RE.findall(text))


def _key(rec: dict[str, Any]) -> Key:
    return str(rec.get("source_id") or ""), group_key(rec.get("group_label"))


def _label(rec: dict[str, Any]) -> Key:
    return str(rec.get("source_id") or ""), label_key(rec.get("group_label"))


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


def is_interval_field(name: str) -> bool:
    """True for a field that names one end of an interval or a range ('cl_lower', 'Upper 95% CI',
    'cmax_min', 'tmax_range'): it may cite one end of a printed range."""
    return _bound(name) is not None or bool(_RANGE_NAME.search(name.casefold()))


def _bound(name: str) -> tuple[str, str] | None:
    """('lower' | 'upper', interval stem) for a bound's field name, else None."""
    key = name.casefold()
    for side, (suffix, prefix) in _BOUND_RES.items():
        m = suffix.match(key) or prefix.match(key)
        if m:
            return side, m.group("stem").strip(" _")
    return None


def _ci_flag(rec: dict[str, Any], field: str, group: str, peers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """ci_inverted on both bounds when lower > upper of one interval within the same source x group;
    estimate_outside_ci on a point estimate named like its interval's stem ('cl' for 'cl_lower')."""
    if rec.get("value_num") is None:
        return []
    mine = _bound(field)
    bounds = {}
    for name, other in peers.items():
        b = _bound(name)
        if b is not None and other.get("value_num") is not None:
            bounds.setdefault(b[1], {})[b[0]] = other
    if mine is None:
        pair = bounds.get(field.casefold().strip(" _"), {})
        lo, hi = pair.get("lower"), pair.get("upper")
        if lo is None or hi is None or lo["value_num"] > hi["value_num"]:
            return []
        if not lo["value_num"] <= rec["value_num"] <= hi["value_num"]:
            msg = f"{field} {rec.get('value')} lies outside its interval {lo.get('value')} to {hi.get('value')}"
            return [_flag(M.FLAG_ESTIMATE_OUTSIDE_CI, "error", msg, field, group)]
        return []
    side, stem = mine
    partner = bounds.get(stem, {}).get("upper" if side == "lower" else "lower")
    if partner is None:
        return []
    lo, hi = (rec, partner) if side == "lower" else (partner, rec)
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
        detail = str(rec.get("verify_detail") or "")        # the server-written sentence of a numeric-grammar failure
        flags.append(_flag(M.FLAG_VERIFY_FAIL, "error", f"{field}: verification failed "
                           f"({rec.get('verify_reason') or 'unknown'})" + (f": {detail}" if detail else ""),
                           field, group))
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
    """One claim for a slot: equal numbers (or value text) AND the same unit -- 2.31 L/h and 2.31 mL/min
    are a conflict, never a repeat."""
    if not same_unit(a.get("unit"), b.get("unit")):
        return False
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
    present: dict[Key, set[str]] = defaultdict(set)          # by the EXACT label: a merged label never hides a field
    by_source: dict[str, set[str]] = defaultdict(set)
    peers: dict[Key, dict[str, dict[str, Any]]] = defaultdict(dict)
    batch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs + approved_list:
        present[_label(r)].add(r["_canon"])
        by_source[_key(r)[0]].add(r["_canon"])
        peers[_key(r)].setdefault(r["_canon"], r)
    for r in recs + live_list:
        batch[(*_key(r), r["_canon"].casefold())].append(r)
    approved = {(str(r.get("source_id") or ""), r["_canon"].casefold(), _key(r)[1]): r for r in approved_list}
    # an ungrouped record ('' group: a text-rule n or dose) is checked against its whole source
    return [_record_flags(r, r["_canon"], str(r.get("group_label") or ""), index,
                          by_source[_key(r)[0]] if not _label(r)[1] else present[_label(r)], approved,
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
        if not _CODE_RE.fullmatch(code) or not message or severity not in M.FLAG_SEVERITIES:
            continue
        key = (code, str(p.get("field") or ""), str(p.get("group") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"code": code, "severity": severity, "message": message, "field": key[1], "group": key[2],
                       "origin": "llm"})
    return merged
