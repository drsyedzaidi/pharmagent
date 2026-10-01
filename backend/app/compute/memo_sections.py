"""Briefing-memo sections, part 1: dataset, NCA, QC, structural model comparison.

Each builder reads ONLY PharmState values and returns a ``Section`` (or ``None``
when that analysis has not been run). Every number goes through a ``Cited`` view,
so it is tagged with the audit entry that produced it and recorded in the value
table. The wording written here contains no digits: a digit in template text would
be flagged by the trace check.
"""
from __future__ import annotations

from typing import Any

from app.compute.memo import AuditTrace
from app.compute.memo_model import (
    MISSING,
    Block,
    Bullet,
    Cited,
    Para,
    Recorder,
    Section,
    Table,
    is_number,
)
from app.core.pharmstate import PharmState

# (nca_summary parameter key, display label, decimals)
NCA_PARAMS: tuple[tuple[str, str, int], ...] = (
    ("Cmax", "Cmax", 2), ("Tmax", "Tmax", 2), ("AUC_last", "AUClast", 1),
    ("AUC_inf", "AUCinf", 1), ("t_half", "t½", 2), ("CL_F", "CL/F", 2), ("Vz_F", "Vz/F", 1),
)


def _rows(*pairs: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """Label/value rows, dropping any whose value is missing."""
    return tuple((label, cell) for label, cell in pairs if cell != MISSING)


def _table_or_none(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> tuple[Block, ...]:
    return (Table(headers, rows),) if rows else ()


# ── dataset ────────────────────────────────────────────────────────────────────
def _dose_rows(cm: Cited, doses: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(doses, (list, tuple)) and doses and all(is_number(d) for d in doses):
        return _rows(("Distinct dose levels", cm.count("dataset_metadata.n_dose_levels", len(doses))),
                     ("Dose range", cm.span("dataset_metadata.dose_range", min(doses), max(doses))))
    if doses:
        text = doses if isinstance(doses, str) else ", ".join(str(d) for d in doses)
        return _rows(("Dose levels", cm.lit("dataset_metadata.dose_levels", text)))
    return ()


def dataset(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    meta = state.dataset_metadata
    if not meta:
        return None
    cm = rec.at(trace.index_for("dataset_metadata"))
    tr = meta.get("time_range")
    span = (cm.span("dataset_metadata.time_range", tr[0], tr[1])
            if isinstance(tr, (list, tuple)) and len(tr) == 2 else MISSING)
    rows = (_rows(("Subjects", cm.count("dataset_metadata.n_subjects", meta.get("n_subjects"))),
                  ("Records", cm.count("dataset_metadata.n_records", meta.get("n_records"))))
            + _dose_rows(cm, meta.get("dose_levels"))
            + _rows(("Observation time range", span)))
    blocks: list[Block] = []
    dq = state.data_quality
    if dq:
        cd = rec.at(trace.index_for("data_quality"))
        rows += _rows(
            ("BLQ observations (%)", cd.num("data_quality.blq_pct", dq.get("blq_pct"))),
            ("Missing data (%)", cd.num("data_quality.total_missing_pct", dq.get("total_missing_pct"))),
            ("Sparse subjects", cd.count("data_quality.n_sparse_subjects", dq.get("n_sparse_subjects"))))
        blocks += [Bullet("Data-quality flag: " + cd.lit("data_quality.flag", f))
                   for f in (dq.get("quality_flags") or [])]
    blocks = [*_table_or_none(("Item", "Value"), rows), *blocks]
    if not blocks:
        return None
    n_sub, n_rec = (cm.count("dataset_metadata.n_subjects", meta.get("n_subjects")),
                    cm.count("dataset_metadata.n_records", meta.get("n_records")))
    headline = (f"Dataset: {n_sub} subjects, {n_rec} records."
                if MISSING not in (n_sub, n_rec) else "Dataset loaded.")
    return Section("dataset", "Dataset", tuple(blocks), headline)


# ── NCA ────────────────────────────────────────────────────────────────────────
def _gm_clause(label: str, p: dict[str, Any], c: Cited, decimals: int, key: str) -> str:
    gm = c.num(f"nca_summary.{key}.geomean", p.get("geomean"), decimals)
    if gm == MISSING:
        return ""
    cv = c.num(f"nca_summary.{key}.geocv_pct", p.get("geocv_pct"), 1, unit="%")
    return f"{label} geometric mean {gm}" + (f" (gCV {cv})" if cv != MISSING else "")


def nca(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    summ = state.nca_summary
    if not summ:
        return None
    c = rec.at(trace.index_for("nca_summary"))
    group = next((g for g in summ.get("descriptive") or [] if g.get("group") == "all"), None) or {}
    by_param = {p.get("parameter"): p for p in group.get("parameters") or []}
    n_sub = c.count("nca_summary.n_subjects", summ.get("n_subjects") or group.get("n"))
    intro = "Non-compartmental analysis" + (f" of {n_sub} subjects" if n_sub != MISSING else "")
    if summ.get("route"):
        intro += f", {c.lit('nca_summary.route', summ['route'])} dosing"
    n_blq = c.count("nca_summary.blq.n_below_loq", (summ.get("blq") or {}).get("n_below_loq"))
    if n_blq != MISSING:
        intro += f"; {n_blq} records below the quantification limit"
    blocks: list[Block] = [Para(intro + ".")]
    rows = []
    for key, label, dec in NCA_PARAMS:
        p = by_param.get(key)
        if not p:
            continue
        pre = f"nca_summary.{key}"
        rows.append((label, c.count(f"{pre}.n", p.get("n")),
                     c.num(f"{pre}.geomean", p.get("geomean"), dec),
                     c.num(f"{pre}.geocv_pct", p.get("geocv_pct"), 1),
                     c.num(f"{pre}.median", p.get("median"), dec),
                     c.num(f"{pre}.min", p.get("min"), dec),
                     c.num(f"{pre}.max", p.get("max"), dec)))
    if rows:
        blocks.append(Table(("Parameter", "n", "Geometric mean", "gCV %", "Median", "Min", "Max"),
                            tuple(rows)))
    else:
        blocks.append(Para("No pooled summary statistics are available for the NCA parameters."))
    clauses = [_gm_clause(lbl, by_param[k], c, d, k) for k, lbl, d in NCA_PARAMS
               if k in ("CL_F", "AUC_inf") and k in by_param]
    clauses = [x for x in clauses if x]
    headline = "NCA: " + "; ".join(clauses) + "." if clauses else "NCA summary computed."
    return Section("nca", "Non-compartmental analysis", tuple(blocks), headline)


# ── QC ─────────────────────────────────────────────────────────────────────────
def qc(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    if not state.qc_verdict:
        return None
    c = rec.at(trace.index_for("qc"))
    verdict = c.lit("qc.verdict", state.qc_verdict)
    checks = state.qc_checklist or []
    flagged = [x for x in checks if str(x.get("status", "")).upper() != "PASS"]
    blocks: list[Block] = [Para(f"QC verdict: {verdict}.")]
    headline = f"QC verdict: {verdict}."
    if checks:
        n_checks, n_flag = c.count("qc.n_checks", len(checks)), c.count("qc.n_flagged", len(flagged))
        blocks.append(Para(f"{n_flag} of {n_checks} checks were not passed."))
        headline = f"QC verdict: {verdict}; {n_flag} of {n_checks} checks not passed."
    blocks += [Bullet(c.lit("qc.failing_check",
                            f"[{x.get('status')}] {x.get('check')}: {x.get('detail')}"))
               for x in flagged]
    blocks += [Bullet(c.lit("qc.issue", f"[{i.get('severity')}] {i.get('issue')}"))
               for i in state.qc_issues or []]
    return Section("qc", "Quality control", tuple(blocks), headline)


# ── structural model comparison ────────────────────────────────────────────────
def _ranking_table(pm: dict[str, Any], c: Cited) -> tuple[Block, ...]:
    rows = []
    for i, r in enumerate(pm.get("ranking") or []):
        pre = f"pk_model_results.ranking.{i}"
        rows.append((c.lit(f"{pre}.label", r.get("label") or r.get("model_key")),
                     c.frac(f"{pre}.converged", r.get("n_converged"), r.get("n_subjects")),
                     c.num(f"{pre}.mean_aic", r.get("mean_aic"), 2),
                     c.num(f"{pre}.total_aic", r.get("total_aic"), 2)))
    return _table_or_none(("Model", "Converged", "Mean AIC", "Total AIC"), tuple(rows))


def structural(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    pm = state.pk_model_results
    if not pm or pm.get("status") != "ok":
        return None
    c = rec.at(trace.index_for("pk_model_results"))
    best = pm if pm.get("mode") == "fit" else (pm.get("best") or {})
    label = c.lit("pk_model_results.best.label", best.get("label") or pm.get("best_model"))
    aic = c.num("pk_model_results.best.mean_aic", best.get("mean_aic"), 2)
    blocks: list[Block] = [*_ranking_table(pm, c)]
    blocks.append(Para(f"Best model by AIC: {label}" + (f", mean AIC {aic}" if aic != MISSING else "")
                       + ". Two-stage population summary:"))
    params = (best.get("population") or {}).get("parameters") or {}
    prow = tuple((name, c.num(f"pk_model_results.best.{name}.typical_value", v.get("typical_value"), 3),
                  c.num(f"pk_model_results.best.{name}.iiv_cv_pct", v.get("iiv_cv_pct"), 1),
                  c.count(f"pk_model_results.best.{name}.n", v.get("n")))
                 for name, v in params.items())
    blocks += _table_or_none(("Parameter", "Typical value", "IIV CV %", "n"), prow)
    headline = f"Structural model: {label} preferred by AIC" + (f" (mean AIC {aic})." if aic != MISSING else ".")
    return Section("structural", "Structural model comparison", tuple(blocks), headline)

