"""Deterministic (keyless) extraction rules: header/alias-matched table cells plus two text
patterns ('n = 12', '150 mg ... dose'). Every proposal cites a table cell or a text window the
verifier can find literally; nothing here computes, converts or rounds a number.

Table orientation: 'wide' when a header cell names a template field (columns are fields, rows
are groups labelled by their first cell); otherwise 'long' (the first column holds parameter
names; the other columns are groups, a 'Unit' column supplies units, and 'Value' / 'Estimate'
-like headers mean the unlabelled group ''). Dispersion columns (SD, RSE, CI, ...) are never
read as values. Headers and row labels that match no field are reported as unmatched_headers
so the user can add aliases to the template instead of guessing.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from app.evidence.numgrammar import GAP, find_dose, find_n_lead, parse_claim, tokens

MAX_SNIPPET, MAX_VALUE, MAX_UNMATCHED, WINDOW_HALF = 300, 120, 50, 140
VALUE_HEADERS = frozenset({"value", "values", "estimate", "estimates", "mean", "result", "results",
                           "pointestimate", "geometricmean", "median", "typicalvalue", "all", "total", "overall"})
UNIT_HEADERS = frozenset({"unit", "units"})
DISPERSION_HEADERS = frozenset({"sd", "se", "sem", "rse", "cv", "cv%", "%cv", "rse%", "%rse", "ci", "95%ci",
                                "90%ci", "range", "iqr", "min", "max", "p", "pvalue", "n", "shrinkage", "iiv"})
_KEY_STRIP = re.compile(r"[\s_\-,.]+")
_PAREN_UNIT_RE = re.compile(r"^(?P<label>.*?)\s*[(\[](?P<unit>[^()\[\]]{1,40})[)\]]\s*$")
_TRAIL_UNIT_RE = re.compile(r"^\s*(?P<unit>[A-Za-z%µμ][\w%/·^.*\-]{0,24})\s*$")
_DOSE_CTX = re.compile(r"\bdos(?:e|es|ed|ing|age)\b|\badminist", re.IGNORECASE)


def _n_hits(text: str) -> list[tuple]:
    """'n = 12': the snippet window starts at the ``n``."""
    return [(s, e, n, None, lead, e) for lead, s, e, n in find_n_lead(text)]


def _dose_hits(text: str) -> list[tuple]:
    """'150 mg': the snippet window runs from the number to the end of its unit word."""
    out = []
    for s, e, value, unit in find_dose(text):
        j = e
        while j < len(text) and text[j] in GAP:
            j += 1
        out.append((s, e, value, unit, s, j + len(unit)))
    return out


# (field, finder of (start, end, value, unit, window start, window end) over a text, context the snippet must
# match): numbers are read by numgrammar (find_n_lead: 'n = 12'; find_dose: a number followed by a dose unit word),
# never by a pattern here
TEXT_RULES = (("n", _n_hits, None), ("dose", _dose_hits, _DOSE_CTX))


# ----------------------------------------------------------------- matching ---

def norm_key(text: Any) -> str:
    """Comparison key for headers, row labels and field names/aliases."""
    t = unicodedata.normalize("NFKC", str(text or "")).replace("∞", "inf").replace("µ", "μ").casefold()
    return _KEY_STRIP.sub("", t)


def field_index(template: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """norm_key(name or alias) -> field spec; the spec's 'name' is the canonical field."""
    index: dict[str, dict[str, Any]] = {}
    for spec in (template or {}).get("fields") or []:
        name = str(spec.get("name") or "")
        if not name:
            continue
        for key in [name, *(spec.get("aliases") or [])]:
            k = norm_key(key)
            if k:
                index.setdefault(k, spec)
    return index


def split_unit(label: str) -> tuple[str, str | None]:
    """'Cmax (ng/mL)' -> ('Cmax', 'ng/mL'); a label without a bracketed unit -> (label, None)."""
    text = " ".join(str(label or "").split())
    m = _PAREN_UNIT_RE.match(text)
    return (m.group("label").strip(), m.group("unit").strip() or None) if m else (text, None)


def match_field(text: str, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    label, _ = split_unit(text)
    return index.get(norm_key(label)) or index.get(norm_key(text))


def decimal_style(spec: dict[str, Any] | None, doc: dict[str, Any]) -> str:
    forced = (spec or {}).get("decimal")
    return forced if forced in (".", ",") else (doc.get("decimal_style") or ".")


def value_number(value: str, style: str) -> float | None:
    """Numeric reading of a proposed value in the page's style, else point-decimal, else None."""
    num = parse_claim(value, style)
    return num if num is not None or style == "." else parse_claim(value, ".")


def leading_number(cell: str, style: str) -> tuple[str, float, str] | None:
    """(literal as printed, value, remainder) for the first number in a cell, else None."""
    first = tokens(cell, style, mode="claim")[:1]
    if not first or first[0].value is None:              # '4.5 x 10' (exponent lost): never a later number
        return None
    return cell[first[0].start:first[0].end], first[0].value, cell[first[0].end:]


def trailing_unit(rest: str) -> str | None:
    m = _TRAIL_UNIT_RE.match(rest)
    return m.group("unit") if m else None


# ---------------------------------------------------------------- proposals ---

def _proposal(spec: dict[str, Any], *, source_id: str, page: int, kind: str, snippet: str, value: str,
              value_num: float | None, unit: str | None, group: str, style: str,
              cell: dict[str, Any] | None = None) -> dict[str, Any]:
    locator = {"page": page, "kind": kind, "snippet": snippet[:MAX_SNIPPET], "span": None, "literal": None,
               "cell": cell, "column": None, "decimal_style": style}
    return {"source_id": source_id, "field": str(spec["name"]), "group_label": group[:MAX_VALUE],
            "value": value[:MAX_VALUE], "value_num": value_num, "unit": (unit or None), "locator": locator,
            "extractor": "table"}


def cell_proposal(spec: dict[str, Any], cell_text: str, *, source_id: str, page: int, style: str,
                  cell: dict[str, Any], group: str, unit: str | None) -> dict[str, Any] | None:
    text = str(cell_text or "").strip()
    if not text or len(text) > MAX_SNIPPET:
        return None
    value, value_num = text, None
    if spec.get("type") in ("number", "integer"):
        found = leading_number(text, style)
        if found is not None:
            value, value_num, rest = found
            unit = unit or trailing_unit(rest)
    return _proposal(spec, source_id=source_id, page=page, kind="table", snippet=text, value=value,
                     value_num=value_num, unit=unit, group=group, style=style, cell=cell)


def _cell_ref(table: dict[str, Any], row: int, col: int, header: str, row_label: str) -> dict[str, Any]:
    return {"table": int(table.get("index") or 0), "row": row, "col": col, "header": header.strip(),
            "row_label": row_label.strip()}


def _wide(rows: list[list[str]], matched: dict[int, dict[str, Any]], table: dict[str, Any], page: int,
          doc: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
    out = []
    header = rows[0]
    for ri, row in enumerate(rows[1:], 1):
        row_label = row[0].strip() if row else ""
        for ci, spec in matched.items():
            if ci >= len(row) or not row[ci].strip():
                continue
            _, unit = split_unit(header[ci])
            p = cell_proposal(spec, row[ci], source_id=source_id, page=page, style=decimal_style(spec, doc),
                              cell=_cell_ref(table, ri, ci, header[ci], row_label), group=row_label, unit=unit)
            if p is not None:
                out.append(p)
    return out


def _long_columns(header: list[str]) -> tuple[int | None, list[tuple[int, str]], list[str]]:
    """(unit column, [(value column, group)], skipped headers) for a parameter-rows table."""
    unit_col = next((ci for ci, h in enumerate(header[1:], 1) if norm_key(h) in UNIT_HEADERS), None)
    value_cols, skipped = [], []
    for ci, h in enumerate(header[1:], 1):
        if ci == unit_col or not h.strip():
            continue
        label, _ = split_unit(h)
        if norm_key(label) in DISPERSION_HEADERS:
            skipped.append(h.strip())
            continue
        value_cols.append((ci, "" if norm_key(h) in VALUE_HEADERS else h.strip()))
    return unit_col, value_cols, skipped


def _long(rows: list[list[str]], index: dict[str, dict[str, Any]], table: dict[str, Any], page: int,
          doc: dict[str, Any], source_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    header = rows[0]
    unit_col, value_cols, unmatched = _long_columns(header)
    out = []
    for ri, row in enumerate(rows[1:], 1):
        label = row[0] if row else ""
        spec = match_field(label, index) if label.strip() else None
        if spec is None:
            if label.strip():
                unmatched.append(label.strip())
            continue
        _, unit = split_unit(label)
        if unit_col is not None and unit_col < len(row) and row[unit_col].strip():
            unit = row[unit_col].strip()
        for ci, group in value_cols:
            if ci >= len(row):
                continue
            p = cell_proposal(spec, row[ci], source_id=source_id, page=page, style=decimal_style(spec, doc),
                              cell=_cell_ref(table, ri, ci, header[ci], label), group=group, unit=unit)
            if p is not None:
                out.append(p)
    return out, unmatched


def table_extract(doc: dict[str, Any], template: dict[str, Any], source_id: str) -> dict[str, Any]:
    """{'proposals': [...], 'unmatched_headers': [...]} over every parsed table of the document."""
    index = field_index(template)
    proposals: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for page in doc.get("pages") or []:
        for table in page.get("tables") or []:
            rows = [[str(c or "") for c in r] for r in (table.get("rows") or [])
                    if any(str(c or "").strip() for c in r)]
            if len(rows) < 2:
                continue
            header = rows[0]
            matched = {ci: spec for ci, h in enumerate(header) if h.strip() and (spec := match_field(h, index))}
            if matched:
                proposals.extend(_wide(rows, matched, table, int(page["n"]), doc, source_id))
                unmatched.extend(h.strip() for ci, h in enumerate(header) if h.strip() and ci not in matched)
            else:
                got, missed = _long(rows, index, table, int(page["n"]), doc, source_id)
                proposals.extend(got)
                unmatched.extend(missed)
    return {"proposals": proposals, "unmatched_headers": list(dict.fromkeys(unmatched))[:MAX_UNMATCHED]}


# --------------------------------------------------------------- text rules ---

def _window(text: str, start: int, end: int) -> str:
    """The line(s) holding text[start:end], trimmed around the match to MAX_SNIPPET chars."""
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    le = len(text) if le < 0 else le
    if le - ls > MAX_SNIPPET:
        ls = max(ls, min(start - WINDOW_HALF, le - MAX_SNIPPET))
        le = min(le, ls + MAX_SNIPPET)
    return text[ls:le].strip()


def text_extract(doc: dict[str, Any], template: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
    """First 'n = <int>' and first '<number> mg ... dose' per source, when the template names them."""
    index = field_index(template)
    out = []
    for key, finder, context in TEXT_RULES:
        spec = index.get(key)
        if spec is None:
            continue
        hit = _first_text_hit(doc, finder, context)
        if hit is None:
            continue
        page_n, column, num, unit, snippet = hit
        style = decimal_style(spec, doc)
        p = _proposal(spec, source_id=source_id, page=page_n, kind="text", snippet=snippet,
                      value=num, value_num=value_number(num, style), unit=unit, group="", style=style)
        out.append({**p, "locator": {**p["locator"], "column": column}})
    return out


def _reading_texts(page: dict[str, Any]) -> list[tuple[int | None, str]]:
    """(column, text) in reading order: each column of a two-column page first (its merged page-text
    lines join the columns), then the page text (lines spanning both columns, single-column pages)."""
    columns = [c for c in (page.get("columns") or []) if isinstance(c, str)]
    return list(enumerate(columns)) + [(None, page.get("text") or "")]


def _first_text_hit(doc: dict[str, Any], finder: Callable[[str], list[tuple]], context: re.Pattern[str] | None):
    """(page, column, literal, unit word or None, snippet) of the first hit of *finder* whose line passes *context*."""
    for page in doc.get("pages") or []:
        for column, text in _reading_texts(page):
            for start, end, _value, unit, w0, w1 in finder(text):
                snippet = _window(text, w0, w1)
                if context is None or context.search(snippet):
                    return int(page["n"]), column, text[start:end], unit, snippet
    return None


def _merge_text(table_props: list[dict[str, Any]], text_props: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Text-rule proposals that add something: one whose field and number a table cell already holds
    is dropped (the same study arm must not count twice); when the tables name exactly one group,
    the text value belongs to it."""
    held = {(p["field"], p["value_num"]) for p in table_props}
    groups = {p["group_label"] for p in table_props if p["group_label"]}
    only = next(iter(groups)) if len(groups) == 1 else ""
    return [{**p, "group_label": only} for p in text_props if (p["field"], p["value_num"]) not in held]


def rule_extract(doc: dict[str, Any], template: dict[str, Any], source_id: str) -> dict[str, Any]:
    """Tables first, then the text rules; the deterministic fallback every keyless run uses."""
    tables = table_extract(doc, template, source_id)
    return {"proposals": tables["proposals"] + _merge_text(tables["proposals"], text_extract(doc, template, source_id)),
            "unmatched_headers": tables["unmatched_headers"]}
