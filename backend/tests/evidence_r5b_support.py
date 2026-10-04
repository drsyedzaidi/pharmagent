"""Shared helpers of the r5b number-reading regressions: a one-page document and a verify() call."""
from __future__ import annotations

from app.evidence import models as M
from app.evidence.verify import verify


def doc(text: str, small: str = "", tables: list | None = None, style: str = ".") -> dict:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": small, "columns": [], "tables": tables or []}]}


def table_doc(cell: str, small: str = "") -> dict:
    return doc(f"row\n{cell}", small, [{"rows": [["Label", cell]], "strategy": "lines"}])


def check(value: str, snippet: str, page_text: str, small: str = "", cell: bool = False) -> M.VerifyResult:
    """verify() of a text record on *page_text*, or of a table-cell record when *cell* (the page text
    is then the cell text itself)."""
    if cell:
        loc = M.Locator(page=1, kind="table", snippet=snippet, cell={"table": 0, "row": 0, "col": 1},
                        decimal_style=".").to_json()
        return verify({"value": value, "locator": loc}, table_doc(page_text, small))
    loc = M.Locator(page=1, kind="text", snippet=snippet, decimal_style=".").to_json()
    return verify({"value": value, "locator": loc}, doc(page_text, small))
