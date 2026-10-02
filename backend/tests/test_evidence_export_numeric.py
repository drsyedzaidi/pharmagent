"""Exported numbers are unambiguous: the long file carries value_num (canonical point-decimal) and the
page's decimal_style beside the literal; the wide file's <field> column is the number and the page's
literal moves to <field>_literal. '3,412' (thousands, point page) and '18,2' (decimal comma page)
export as 3412 and 18.2."""
from __future__ import annotations

from app.evidence.export_tools import LONG_COLUMNS, build_long, build_wide, num_text
from app.evidence.models import Record, Source


def _src(sid: str, name: str, style: str) -> Source:
    return Source(id=sid, session_id="s", sha256=sid * 4, filename=name, pdf_path="", parse_path="", pages=1,
                  title=None, metadata={}, text_chars=1, parse_status="parsed", decimal_style=style,
                  licence_note=None, uploaded_by="a", uploaded_at="t")


def _rec(rid: str, src: str, field: str, value: str, num: float | None, style: str, group: str = "") -> Record:
    return Record(id=rid, session_id="s", source_id=src, template_hash="t", field=field, group_label=group,
                  value=value, value_num=num, unit="ng/mL", extractor="table", verify_status="pass", verify_reason="",
                  locator={"page": 1, "kind": "table", "snippet": value, "literal": value, "decimal_style": style},
                  flags=[], status="approved", decided_by="alice", decided_at="t", decision_reason="ok",
                  record_hash=rid * 4, created_at="t")


SOURCES = {"A": _src("A", "a.pdf", "."), "C": _src("C", "c.pdf", ",")}
RECORDS = [_rec("r1", "A", "cmax", "3,412", 3412.0, "."), _rec("r2", "C", "cmax", "18,2", 18.2, ","),
           _rec("r3", "A", "change", "−20", -20.0, "."), _rec("r4", "A", "species", "human", None, ".")]


def test_long_rows_carry_value_num_and_decimal_style():
    assert LONG_COLUMNS[LONG_COLUMNS.index("value") + 1:][:2] == ("value_num", "decimal_style")
    rows = {r["record_id"]: r for r in build_long(RECORDS, SOURCES)}
    assert (rows["r1"]["value"], rows["r1"]["value_num"], rows["r1"]["decimal_style"]) == ("3,412", "3412", ".")
    assert (rows["r2"]["value"], rows["r2"]["value_num"], rows["r2"]["decimal_style"]) == ("18,2", "18.2", ",")
    assert rows["r3"]["value_num"] == "-20" and rows["r4"]["value_num"] == ""


def test_wide_field_column_is_the_number_and_the_literal_moves():
    rows, cols = build_wide(RECORDS, SOURCES, {"fields": [{"name": "cmax"}, {"name": "change"}]})
    assert cols[4:7] == ["cmax", "cmax_literal", "cmax_unit"]
    by_src = {r["source_id"]: r for r in rows}
    assert (by_src["A"]["cmax"], by_src["A"]["cmax_literal"]) == ("3412", "3,412")
    assert (by_src["C"]["cmax"], by_src["C"]["cmax_literal"]) == ("18.2", "18,2")
    assert by_src["A"]["change"] == "-20" and by_src["A"]["species"] == "human"
    assert float(by_src["A"]["cmax"]) == 3412 and float(by_src["C"]["cmax"]) == 18.2


def test_num_text_is_canonical():
    assert [num_text(v) for v in (3412.0, 18.2, -20.0, 0.000123, None, float("nan"))] == \
        ["3412", "18.2", "-20", "0.000123", "", ""]
