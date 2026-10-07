"""Literal verification and scripted checks of a batch of proposals (the path every model and every keyless
proposal takes): ``check_proposals`` verifies each against the parsed page, then adds the scripted QC flags, the
instruction-text flag and the page checks (unit, hidden text, qualifier, scale, style, glyphs). Pure: it reads the
document it is given and returns new dicts. One request cache (``read_cache``) lives for the whole batch, so a page
is normalised and tokenised once however many records cite it."""
from __future__ import annotations

import re
from typing import Any

from app.evidence import models as M
from app.evidence.earlier_decisions import earlier_flags
from app.evidence.extract_rules import field_index, norm_key, value_decimal
from app.evidence.normalize import normalise, numbers_in
from app.evidence.numgrammar import claim_decimal
from app.evidence.qc_flags import scripted_flags
from app.evidence.read_cache import read_cache
from app.evidence.record_checks import page_checks
from app.evidence.value_checks import cell_numbers, document_evidence
from app.evidence.verify import verify

_INSTRUCTION_RE = re.compile(
    r"\b(ignore|disregard|override|forget)\b[^.\n]{0,60}\b(instructions?|rules?|prompts?|guidelines?)\b"
    r"|\bapprove (everything|all|every)\b|\bsystem prompt\b|\b(you are an? |as an ai\b|automated readers?)",
    re.IGNORECASE)


def _suspicious(p: dict[str, Any]) -> list[dict[str, Any]]:
    text = f"{p.get('value') or ''} {p['locator'].get('snippet') or ''}"
    if not _INSTRUCTION_RE.search(text):
        return []
    return [{"code": M.FLAG_SUSPICIOUS_INSTRUCTION, "severity": "warn",
             "message": f"{p['field']}: the value or its snippet reads like an instruction to an automated "
                        "reader; it is document text, not a finding", "field": p["field"],
             "group": p.get("group_label") or "", "origin": "script"}]


def _composite(p: dict[str, Any], ftype: str, doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    """A numeric table cell holding more than one number ('3,412 (812)'), counted in the CITED CELL
    (the snippet may quote one of them): the reviewer must confirm which number it is."""
    loc = p["locator"]
    if ftype == "text" or loc.get("kind") != "table" or any(f["code"] == M.FLAG_COMPOSITE_CELL for f in p["flags"]):
        return []
    count = cell_numbers(p, doc)
    if (len(numbers_in(str(loc.get("snippet") or ""), loc.get("decimal_style") or ".")) if count is None
            else count) <= 1:
        return []
    return [{"code": M.FLAG_COMPOSITE_CELL, "severity": "warn",
             "message": f"{p['field']}: cell {loc.get('snippet')!r} holds more than one number; "
                        f"{p['value']!r} was read as the estimate", "field": p["field"],
             "group": p.get("group_label") or "", "origin": "script"}]


def _field_type(p: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    spec = index.get(norm_key(p.get("field")))
    if spec is not None:
        return str(spec.get("type") or "number")
    return "text" if p.get("value_num") is None else "number"


def _page_reading(p: dict[str, Any], number: float | None, literal: str | None) -> dict[str, Any]:
    """value_num := the matched page token's value (never an independent re-parse of the proposal);
    a proposal whose own reading disagrees ('1,234' on a decimal-comma page that prints 1234) takes
    the page literal as its value, so value and value_num always describe the same printed number."""
    if number is None:
        return p
    style = (p.get("locator") or {}).get("decimal_style") or "."
    own = value_decimal(str(p.get("value") or ""), style)
    page = claim_decimal(literal or "", style)
    same = own is not None and (own == page if page is not None else float(own) == number)   # exact, no tolerance
    return {**p, "value_num": number} if same else {**p, "value": literal or p.get("value"), "value_num": number}


def _verify_one(p: dict[str, Any], doc: dict[str, Any], ftype: str) -> dict[str, Any]:
    res = verify(p, doc, field_type=ftype)
    status, reason, loc = res.status, res.reason, dict(res.locator or p["locator"])
    if status == "pass" and ftype == "text" and normalise(str(p.get("value") or ""))[0] not in normalise(
            str(loc.get("snippet") or ""))[0]:
        status, reason, loc = "fail", M.REASON_TEXT_NOT_IN_SNIPPET, {**loc, "span": None, "literal": None}
    base = _page_reading(p, res.number, res.literal) if status == "pass" else p
    if ftype == "text":                          # a text field never carries a number of its own
        base = {**base, "value_num": None}
    return {**base, "locator": loc, "verify_status": status, "verify_reason": reason, "verify_detail": res.detail}


def check_proposals(proposals: list[dict[str, Any]], doc: dict[str, Any] | None, template: dict[str, Any],
                    approved: list[Any] | None = None, live: list[Any] | None = None,
                    decided: list[Any] | None = None, template_hash: str = "") -> list[dict[str, Any]]:
    """Pure: literal verification of every proposal (all fail 'source_changed' when *doc* is None),
    then scripted QC flags (conflicts against *approved* and the *live* proposals of earlier runs), the
    instruction-text flag and the 'decided_earlier' flag (a claim already approved / rejected under another
    template hash, from *decided*). Returns new dicts; inputs are untouched."""
    index = field_index(template)
    with read_cache():
        if doc is None:
            checked = [{**p, "locator": {**p["locator"], "span": None, "literal": None}, "verify_status": "fail",
                        "verify_reason": M.REASON_SOURCE_CHANGED} for p in proposals]
        else:
            checked = [_verify_one(p, doc, _field_type(p, index)) for p in proposals]
        flags = scripted_flags(checked, template, list(approved or []), list(live or []))
        counted = {**doc, "decimal_evidence": document_evidence(doc)} if doc is not None else None
        earlier = earlier_flags(checked, list(decided or []), template, template_hash)
        out = [{**p, "flags": fl + _suspicious(p) + page_checks(p, counted, index.get(norm_key(p.get("field")))) + ea}
               for p, fl, ea in zip(checked, flags, earlier, strict=True)]
        return [{**p, "flags": p["flags"] + _composite(p, _field_type(p, index), doc)} for p in out]
