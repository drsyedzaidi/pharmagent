"""Shared helpers of the r6 round-1 regressions: a one-page document and the end-to-end path every proposal takes
(``check_proposals``: verify, then the value, unit and glyph checks), reduced to the fields a regression asserts."""
from __future__ import annotations

from typing import Any, NamedTuple

from app.evidence import models as M
from app.evidence.proposal_check import check_proposals

TEMPLATE = {"fields": [{"name": "Cmax", "type": "number"}, {"name": "AUC", "type": "number"},
                       {"name": "Cell dose", "type": "text"}, {"name": "dose", "type": "number"}]}
COMMA_PAGE = "\n".join(["Werte 3,4 und 5,6 und 7,8"] * 3) + "\n"        # seven decimal-comma numbers: a comma document
POINT_PAGE = "Werte 1.1 und 2.2 und 3.3 und 4.4\n"                       # decimal-point evidence


class Outcome(NamedTuple):
    status: str
    reason: str
    value_num: float | None
    value: str
    flags: list[str]
    detail: str = ""


def doc(text: str, style: str = ".", tables: list | None = None, columns: list | None = None) -> dict[str, Any]:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": columns or [], "tables": tables or []}]}


def run(page: str, snippet: str, value: str, *, style: str = ".", field: str = "Cmax", unit: str | None = None,
        document: dict[str, Any] | None = None, kind: str = "text", cell: dict[str, int] | None = None) -> Outcome:
    loc = M.Locator(page=1, kind=kind, snippet=snippet, cell=cell, decimal_style=style).to_json()
    p = {"source_id": "S1", "field": field, "group_label": "", "value": value, "value_num": None, "unit": unit,
         "locator": loc, "extractor": "llm"}
    r = check_proposals([p], document or doc(page, style), TEMPLATE)[0]
    verify_msg = next((f["message"] for f in r["flags"] if f["code"] == M.FLAG_VERIFY_FAIL), "")
    return Outcome(r["verify_status"], r["verify_reason"], r["value_num"], r["value"],
                   [f["code"] for f in r["flags"]], verify_msg)


def cell(rows: list[list[str]], r: int, c: int, value: str, *, snippet: str | None = None, style: str = ".",
         unit: str | None = None, field: str = "Cmax", page: str = "") -> Outcome:
    """A table-cell record: the cited cell is rows[r][c]."""
    return run(page, value if snippet is None else snippet, value, style=style, field=field, unit=unit,
               document=doc(page or "x", style, tables=[{"rows": rows}]), kind="table",
               cell={"table": 0, "row": r, "col": c})
