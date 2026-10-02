"""Checks on a VERIFIED proposal that need its parsed page: the proposed unit against the unit the
source prints for that number, and a flag when the cited page had invisible text removed.

Source unit, in order: the unit token right after the number in the snippet / cell ('100 mcg',
'45%'), else for a table cell the column header's bracketed unit, the row label's bracketed unit or
the row's 'Unit' column. A recognisable source unit that differs from the proposed one is an error
flag (a 1000x dose error otherwise passes); no source unit at all is a warning (unit_unverified).
"""
from __future__ import annotations

import re
from functools import cache
from typing import Any

from app.evidence import models as M
from app.evidence.extract_rules import UNIT_HEADERS, norm_key, split_unit
from app.evidence.normalize import normalise
from app.evidence.qc_flags import normalise_unit

_UNIT_TOKEN_RE = re.compile(r"^\s*(%|[^\W\d_][\w/·*.^\-]*(?:\s+[^\W\d_]{1,3}(?:/[\w.^\-]+)+)?)")
_BASES = ("kg", "g", "mg", "ug", "ng", "pg", "mol", "mmol", "umol", "nmol", "pmol", "l", "ml", "dl", "ul",
          "h", "min", "s", "d", "wk", "iu", "miu", "u", "%", "kda", "m2", "cells", "cell", "10^6", "10^9",
          "bpm", "mmhg", "x", "fold")


@cache
def _segmentable(piece: str) -> bool:
    """True when *piece* is a concatenation of base unit symbols ('ngh' = 'ng' + 'h'; 'healthy' is not)."""
    if not piece:
        return True
    return any(piece.startswith(b) and _segmentable(piece[len(b):]) for b in _BASES)


def recognised_unit(token: str | None) -> str | None:
    """normalise_unit(token) when every '/'-separated piece is made of known unit symbols, else None."""
    if not token:
        return None
    unit = normalise_unit(token).rstrip(".,;:")
    pieces = [p for p in re.split(r"[/]", unit) if p]
    return unit if pieces and all(_segmentable(p.replace("-1", "").replace("^-1", "")) for p in pieces) else None


def _after_literal(snippet: str, literal: str | None) -> str | None:
    """Normalised snippet text right after the cited number, else None."""
    if not literal:
        return None
    ns, nl = normalise(snippet)[0], normalise(literal)[0]
    j = ns.find(nl)
    return None if j < 0 or not nl else ns[j + len(nl):]


def _adjacent_unit(rest: str | None) -> str | None:
    m = _UNIT_TOKEN_RE.match(rest or "")
    return recognised_unit(m.group(1)) if m else None


def _table_unit(loc: dict[str, Any], page: dict[str, Any]) -> str | None:
    cell = loc.get("cell") or {}
    try:
        rows = (page.get("tables") or [])[cell["table"]]["rows"]
        row, header = rows[cell["row"]], rows[0]
    except (KeyError, IndexError, TypeError):
        return None
    col = cell.get("col")
    candidates = [split_unit(header[col])[1] if isinstance(col, int) and col < len(header) else None,
                  split_unit(row[0])[1] if row else None]
    candidates += [row[i] for i, h in enumerate(header) if norm_key(h) in UNIT_HEADERS and i < len(row)]
    return next((u for u in (recognised_unit(c) for c in candidates) if u), None)


def source_unit(p: dict[str, Any], page: dict[str, Any]) -> str | None:
    loc = p.get("locator") or {}
    unit = _adjacent_unit(_after_literal(str(loc.get("snippet") or ""), loc.get("literal")))
    if unit is None and loc.get("kind") == "table":
        unit = _table_unit(loc, page)
    return unit


def _flag(code: str, severity: str, message: str, p: dict[str, Any]) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "field": p["field"],
            "group": p.get("group_label") or "", "origin": "script"}


def unit_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    """For a passed numeric proposal with a unit: mismatch (error) or unverified (warn)."""
    proposed = p.get("unit")
    if p.get("verify_status") != "pass" or not proposed or p.get("value_num") is None:
        return []
    have = source_unit(p, page)
    if have is None:
        return [_flag(M.FLAG_UNIT_UNVERIFIED, "warn", f"{p['field']}: unit {proposed!r} is not printed next to "
                      "the cited number or in its table header; check it on the page", p)]
    want = normalise_unit(proposed)
    if want == have or want.startswith(have + "/") or (len(have) >= 2 and want.startswith(have)):
        return []
    return [_flag(M.FLAG_UNIT_SOURCE_MISMATCH, "error", f"{p['field']}: the source prints {have!r} for this "
                  f"number, the record says {proposed!r}", p)]


def hidden_text_flags(p: dict[str, Any], page: dict[str, Any] | None) -> list[dict[str, Any]]:
    n = int((page or {}).get("hidden_chars") or 0)
    if n <= 0:
        return []
    return [_flag(M.FLAG_HIDDEN_TEXT, "warn", f"page {(page or {}).get('n')} holds {n} invisible or off-page "
                  "characters that were removed before verification; compare with the PDF", p)]


def page_checks(p: dict[str, Any], doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Unit and hidden-text flags for one checked proposal (none when the page is unavailable)."""
    pages = (doc or {}).get("pages") or []
    n = (p.get("locator") or {}).get("page")
    page = pages[n - 1] if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(pages) else None
    if page is None:
        return []
    return unit_flags(p, page) + hidden_text_flags(p, page)
