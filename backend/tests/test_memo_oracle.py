"""Independent oracle for the briefing memo: every rendered number must be the
rendering of the state value at its own key.

The trace check (memo_untraced_numbers) is membership only: a number anywhere in
the value table passes, so a builder that prints shrinkage in the 'Typical value'
column, or the upper CI in the GMR column, is invisible to it. Here each
TracedValue's key is resolved against the fixture state independently of the
builders, and its text must equal that state value rounded the way the text is
written. Hard-coded rows pin the column order of the main tables.
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import pytest

from app.compute.memo_build import build_memo
from app.compute.memo_model import Table
from tests.memo_fixtures import full_state, full_trail

# keys whose value is DERIVED in the template (a count, a percentage of a level, ...):
# checked by the explicit rows below instead of the generic resolver
DERIVED = re.compile(r"(n_dose_levels|dose_range|qc\.n_checks|qc\.n_flagged|n_covariate_effects|"
                     r"scm_results\.n_selected|ci_level_pct|\.converged\.|nlme_results\.covariate_rse_pct|"
                     r"\.n_ok\.)")


def _nca(state: dict, rest: list[str]) -> Any:
    group = next(g for g in state["nca_summary"]["descriptive"] if g["group"] == "all")
    p = next(x for x in group["parameters"] if x["parameter"] == rest[0])
    return p[rest[1]]


_BOUND_NAMES = {"lower": ("lo", "ci_lo", "ci_lower", 0), "upper": ("hi", "ci_hi", "ci_upper", 1)}


def _bound(node: Any, parent: Any, last: str, end: str) -> Any:
    """``<key>.lower`` / ``.upper`` of a span: the matching bound on, or beside, the spanned value."""
    *names, index = _BOUND_NAMES[end]
    if isinstance(node, list):
        return node[index]
    for host in (node, parent):
        if not isinstance(host, dict):
            continue
        for name in (*names, f"{last}_{names[0]}"):
            if name in host:
                return host[name]
    raise KeyError(f"no {end} bound for {last}")


def resolve(state: dict, key: str) -> Any:
    """The state value a memo key names, found WITHOUT the memo builders."""
    head, *rest = key.split(".")
    if head == "nca_summary" and rest[0] not in ("n_subjects", "blq"):
        return _nca(state, rest)
    params = (((state.get(head) or {}).get("best") or {}).get("population") or {}).get("parameters") or {}
    if head == "pk_model_results" and rest[0] == "best" and rest[1] in params:
        return params[rest[1]][rest[2]]
    if head == "diagnostics_results":
        where, prefix = {"iwres": ("residuals", "iwres_"), "cwres": ("cwres", "cwres_"), "npd": ("npde", "")}[rest[0]]
        node = state[head][where]["summary"]
        name = {"pct_outside": "pct_outside_1_96"}.get(rest[1], rest[1])
        return node["n"] if name == "n" else node.get(prefix + name, node.get(name))
    rest = {"forest_results": ["rows"], "indirect_results": ["comparisons"]}.get(head, []) + rest
    node: Any = state[head]
    parent: Any = None
    for i, part in enumerate(rest):
        if part in ("lower", "upper") and i == len(rest) - 1:
            return _bound(node, parent, rest[i - 1], part)
        if isinstance(node, list):
            parent, node = node, node[int(part)]
        elif isinstance(node, dict) and part in node:
            parent, node = node, node[part]
        elif part in ("ci", "estimate"):
            continue                 # a span over bounds held on this node / the scalar itself
        else:
            raise KeyError(f"{key}: no '{part}'")
    return node


def _matches(text: str, value: Any) -> bool:
    """``text`` is ``value`` rounded at the precision ``text`` is written with
    (fixed decimals, or significant digits for a ``%g`` rendering such as 1.21e-09)."""
    v = float(value)
    if "e" not in text.lower():
        places = len(text.split(".")[1]) if "." in text else 0
        if Decimal(text) == Decimal(f"{v:.{places}f}"):
            return True
    mantissa = text.lower().split("e")[0].lstrip("-").replace(".", "").lstrip("0")
    return float(text) == float(f"{v:.{max(len(mantissa), 1)}g}")


def test_every_number_in_the_memo_renders_the_state_value_at_its_own_key():
    state = full_state()
    dump = state.model_dump()
    memo = build_memo(state, full_trail())
    checked = 0
    for v in memo.values:
        if v.kind == "literal" or DERIVED.search(v.key):
            continue
        value = resolve(dump, v.key)
        assert _matches(v.text, value), f"{v.key}: memo prints {v.text}, state holds {value!r}"
        checked += 1
    assert checked >= 120                       # nearly every number is independently checked


def _rows(memo, first_header: str) -> list[tuple[str, ...]]:
    return [r for b in memo.blocks() if isinstance(b, Table) and b.headers[0] == first_header for r in b.rows]


def test_population_forest_nca_and_diagnostics_rows_are_pinned_column_by_column():
    memo = build_memo(full_state(), full_trail())
    assert ("CL", "2.710 [#6]", "8.2 [#6]", "28.5 [#6]", "12.4 [#6]") in _rows(memo, "Parameter")
    assert ("KA", "1.230 [#6]", "15.6 [#6]", "-", "-") in _rows(memo, "Parameter")
    forest = [r for r in _rows(memo, "Parameter") if len(r) == 6]
    assert forest[0] == ("CL", "CRCL", "5th percentile (45 mL/min) [#8]", "0.840 [#8]",
                         "0.710 to 0.990 [#8]", "delta")
    assert ("CL/F", "12 [#3]", "2.71 [#3]", "28.9 [#3]", "2.90 [#3]", "1.49 [#3]", "3.91 [#3]") \
        in _rows(memo, "Parameter")
    metrics = {r[0]: r[1:] for r in _rows(memo, "Metric")}
    assert metrics["CWRES mean"] == ("0.031 [#10]", "120 [#10]")
    assert metrics["npd beyond the nominal limits (%)"] == ("5.8 [#10]", "120 [#10]")
    assert metrics["R² of IPRED (log scale)"] == ("0.967 [#9]", "120 [#9]")


def test_er_and_dose_selection_rows_are_pinned_column_by_column():
    memo = build_memo(full_state(), full_trail())
    fits = {r[0]: r[1:] for r in _rows(memo, "Fit")}
    assert fits["efficacy"] == ("logistic", "AUC", "150 [#13]", "72 [#13]", "OR per SD", "16.901 [#13]",
                                "6.792 to 42.056 [#13]", "95% [#13]", "1.21e-09 [#13]")
    assert fits["pfs"][5:7] == ("2.104 [#15]", "1.703 to 2.599 [#15]")
    doses = {r[0]: r[1:] for r in _rows(memo, "Dose")}
    assert doses["200 [#17]"] == ("0.954 [#17]", "0.529 [#17]", "0.425 [#17]", "0.290 to 0.547 [#17]",
                                  "0.90 [#17]", "yes")


@pytest.mark.parametrize("key", ["nlme_results.theta.CL", "nlme_results.theta.V", "nlme_results.ofv"])
def test_missing_values_are_dashes_and_never_recorded(key):
    state = full_state()
    nlme = {**state.nlme_results, "theta": {"CL": None, "V": float("nan")}, "ofv": None}
    memo = build_memo(state.model_copy(update={"nlme_results": nlme}), full_trail())
    assert key not in {v.key for v in memo.values}
    rows = {r[0]: r for r in _rows(memo, "Parameter") if len(r) == 5}
    assert rows["CL"][1] == "-" and rows["V"][1] == "-"
    assert rows["CL"][2] == "8.2 [#6]"                       # the rest of the row still renders
    assert "OFV" not in next(b.text for s in memo.sections if s.key == "population" for b in s.blocks[:1])
