"""Report Agent tools: generate a regulatory-style DOCX from PharmState."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document

from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

METHODS_NCA = (
    "Non-compartmental analysis was performed using the linear-up/log-down "
    "trapezoidal rule. The terminal elimination rate constant (lambda_z) was "
    "estimated by log-linear regression over the terminal phase, selecting the "
    "window that maximized the adjusted R-squared. AUC was extrapolated to "
    "infinity as Clast/lambda_z. Clearance (CL/F) and volume (Vz/F) were "
    "derived from dose and AUCinf. All computations were performed by "
    "deterministic numerical routines; results are recorded in a SHA-256 "
    "hash-chained audit trail."
)


def _fmt(v: Any, d: int = 2) -> str:
    """Format a numeric cell to d decimals; '-' for missing."""
    if v is None:
        return "-"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def _subject_key(r: dict[str, Any]) -> tuple[int, Any]:
    """Sort subjects numerically when possible, else lexically."""
    s = r.get("subject")
    try:
        return (0, int(float(s)))
    except (TypeError, ValueError):
        return (1, str(s))


# (display header, nca_parameters key, decimals)
_SUBJECT_COLS: list[tuple[str, str, int]] = [
    ("ID", "subject", 0),
    ("Dose", "dose", 0),
    ("Cmax", "Cmax", 2),
    ("Tmax", "Tmax", 2),
    ("AUClast", "AUC_last", 1),
    ("AUCinf", "AUC_inf", 1),
    ("t½", "t_half", 1),
    ("CL/F", "CL_F", 2),
    ("Vz/F", "Vz_F", 1),
    ("λz adjR²", "lambda_z_r2_adj", 3),
    ("λz n", "lambda_z_n_points", 0),
    ("%extrap", "pct_AUC_extrap", 1),
]


def _subject_table(doc: Document, params: list[dict[str, Any]]) -> None:
    """Primary results: one row per subject. Robust for weight-based dosing
    where each subject has a near-unique dose (no meaningful pooling)."""
    if not params:
        doc.add_paragraph("No per-subject NCA parameters available.")
        return
    rows = sorted(params, key=_subject_key)
    headers = [h for h, _, _ in _SUBJECT_COLS]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for r in rows:
        cells = table.add_row().cells
        for i, (_h, key, dec) in enumerate(_SUBJECT_COLS):
            cells[i].text = _fmt(r.get(key), dec)


def _dose_grouped_meaningfully(summary: dict[str, Any]) -> bool:
    """Only pool by dose when there are genuinely few levels with replicates —
    otherwise (e.g. mg/kg weight-based dosing) every subject is its own 'group'
    and the geomean/CV table is noise. Mirrors the frontend heuristic."""
    rows = summary.get("by_dose", [])
    n_subjects = summary.get("n_subjects", 0) or 0
    if not rows or not n_subjects:
        return False
    few_levels = len(rows) <= max(3, n_subjects // 3)
    has_replicates = any((r.get("n") or 0) >= 3 for r in rows)
    return few_levels and has_replicates


def _summary_table(doc: Document, summary: dict[str, Any]) -> None:
    rows = summary.get("by_dose", [])
    if not rows:
        return
    cols = ["dose", "n", "Cmax_geomean", "Cmax_geocv_pct",
            "AUC_inf_geomean", "AUC_inf_geocv_pct", "t_half_median"]
    headers = ["Dose", "n", "Cmax GM", "Cmax gCV%", "AUCinf GM", "AUCinf gCV%", "t½ median"]
    decimals = {"dose": 0, "n": 0, "Cmax_geomean": 2, "Cmax_geocv_pct": 1,
                "AUC_inf_geomean": 1, "AUC_inf_geocv_pct": 1, "t_half_median": 1}
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for r in rows:
        cells = table.add_row().cells
        for i, c in enumerate(cols):
            cells[i].text = _fmt(r.get(c), decimals[c])


def _descriptive_table(doc: Document, params: list[dict[str, Any]]) -> None:
    def f(v: Any, d: int = 3) -> str:
        return "–" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))
    rows = [[p.get("parameter"), p.get("n"), f(p.get("mean")), f(p.get("sd")), f(p.get("cv_pct"), 1),
             f(p.get("median")), f(p.get("min")), f(p.get("max")), f(p.get("geomean")), f(p.get("geocv_pct"), 1)]
            for p in params]
    _kv_table(doc, ["Parameter", "N", "Mean", "SD", "CV%", "Median", "Min", "Max", "GM", "gCV%"], rows)


def _kv_table(doc: Document, headers: list[str], rows: list[list[Any]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = str(h)
    for r in rows:
        cells = table.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = _fmt(v) if isinstance(v, (int, float)) else str(v)


def _extra_sections(doc: Document, state: PharmState,
                    audit_trail: tuple[dict[str, Any], ...] = ()) -> None:
    """Append sections for any further analyses present in state."""
    n = 4

    be = state.be_results
    if be and be.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Bioequivalence", level=1)
        doc.add_paragraph(f"{be['design']}: {be.get('test_level')} vs {be.get('reference_level')}; "
                          f"limits {be['limits'][0]}–{be['limits'][1]}%. Overall: "
                          f"{'BIOEQUIVALENT' if be.get('bioequivalent') else 'NOT bioequivalent'}.")
        _kv_table(doc, ["Parameter", "GMR %", "90% CI", "Within"],
                  [[k, v.get("gmr_pct"), f"{_fmt(v.get('ci_lower_pct'))}–{_fmt(v.get('ci_upper_pct'))}",
                    "yes" if v.get("within_limits") else "no"] for k, v in (be.get("parameters") or {}).items()])

    dp = state.dose_prop_results
    if dp and dp.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Dose proportionality", level=1)
        doc.add_paragraph(f"Power model; {'proportional' if dp.get('proportional') else 'not proportional'}.")
        _kv_table(doc, ["Parameter", "Slope", "90% CI", "Proportional"],
                  [[k, v.get("slope"), f"{_fmt(v.get('slope_ci_lower'))}–{_fmt(v.get('slope_ci_upper'))}",
                    "yes" if v.get("proportional") else "no"] for k, v in (dp.get("parameters") or {}).items()])

    pm = state.pk_model_results
    if pm and pm.get("status") == "ok":
        n += 1
        best = pm if pm.get("mode") == "fit" else (pm.get("best") or {})
        doc.add_heading(f"{n}. Structural PK model", level=1)
        doc.add_paragraph(f"Model: {best.get('label')} — {best.get('n_converged')}/{best.get('n_subjects')} "
                          f"converged, mean AIC {best.get('mean_aic')}. Two-stage population:")
        pop = (best.get("population") or {}).get("parameters") or {}
        _kv_table(doc, ["Parameter", "Typical (GM)", "IIV CV%", "n"],
                  [[k, v.get("typical_value"), v.get("iiv_cv_pct"), v.get("n")] for k, v in pop.items()])

    pk = state.poppk_results
    if pk and pk.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Population PK (two-stage)", level=1)
        _kv_table(doc, ["Parameter", "Typical (GM)", "IIV CV%", "n"],
                  [[k, v.get("typical_value"), v.get("iiv_cv_pct"), v.get("n")]
                   for k, v in (pk.get("parameters") or {}).items()])

    nl = state.nlme_results
    if nl and nl.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Population PK — mixed-effects ({nl.get('method')})", level=1)
        cond = nl.get("condition_number")
        cond_str = f"; condition number {_fmt(cond, 1)}" if cond is not None else ""
        doc.add_paragraph(f"Model: {nl.get('label')}; IIV on {nl.get('iiv_params')}; "
                          f"{nl.get('error_model')} error; OFV {_fmt(nl.get('ofv'), 1)}{cond_str}; "
                          f"{'converged' if nl.get('converged') else 'did not converge'}.")
        omega, shr = nl.get("omega_cv_pct") or {}, nl.get("shrinkage_pct") or {}
        rse, ormse = nl.get("theta_rse_pct") or {}, nl.get("omega_rse_pct") or {}
        _kv_table(doc, ["Parameter", "Typical (θ)", "RSE%", "IIV CV%", "IIV RSE%", "η-shrinkage%"],
                  [[p, v, rse.get(p, "—"), omega.get(p, "—"), ormse.get(p, "—"), shr.get(p, "—")]
                   for p, v in (nl.get("theta") or {}).items()])
        ce = nl.get("covariate_effects") or []
        if ce:
            doc.add_paragraph("Covariate effects: " + "; ".join(
                f"{e['param']} ~ {e['description']}"
                + (f" ({_fmt(e['rse_pct'], 1)}% RSE)"
                   if isinstance(e.get("rse_pct"), (int, float)) else "")
                for e in ce))
        if nl.get("cov_note"):
            doc.add_paragraph(str(nl.get("cov_note")))

    sc = state.scm_results
    if sc and sc.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Covariate model selection (SCM)", level=1)
        sel = sc.get("selected") or []
        sel_str = ", ".join(f"{s['param']}~{s['covariate']} ({s['kind']})"
                            for s in sel) or "none"
        doc.add_paragraph(
            f"Stepwise covariate modeling on {sc.get('label')} "
            f"(forward p<{sc.get('forward_p')}, backward p<{sc.get('backward_p')}): "
            f"{sc.get('n_candidates')} candidate(s) tested; selected {sel_str}; "
            f"OFV {_fmt(sc.get('base_ofv'), 1)} -> {_fmt(sc.get('final_ofv'), 1)}.")
        steps = sc.get("steps") or []
        if steps:
            _kv_table(doc, ["Phase", "Effect", "ΔOFV", "χ² crit", "df", "Decision"],
                      [[s["phase"], s["effect"], _fmt(s["delta_ofv"], 2),
                        _fmt(s["crit"], 2), s["df"], s["decision"]] for s in steps])

    dg = state.diagnostics_results
    if dg and dg.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Residual diagnostics", level=1)
        iw = (dg.get("residuals") or {}).get("summary") or {}
        cw, npd_ = dg.get("cwres") or {}, dg.get("npde") or {}
        cws, npds = cw.get("summary") or {}, npd_.get("summary") or {}
        prov = dg.get("nlme_provenance")
        doc.add_paragraph(
            f"Model: {dg.get('label')}. Two-stage IWRES over n = {iw.get('n')} observations. "
            + (f"CWRES and prediction-discrepancy diagnostics are calibrated from the "
               f"mixed-effects fit of {prov} (single provenance)."
               if prov else
               "No converged NLME fit of this model was available, so CWRES and "
               "prediction-discrepancy diagnostics were not computed."))
        rows: list[list[Any]] = [["IWRES (two-stage)", _fmt(iw.get("mean"), 3),
                                  _fmt(iw.get("sd"), 3), iw.get("n"), "computed"]]
        rows.append(["CWRES", _fmt(cws.get("cwres_mean"), 3), _fmt(cws.get("cwres_sd"), 3),
                     cws.get("n", "-"), cw.get("status", "ok")])
        rows.append(["npd", _fmt(npds.get("mean"), 3), _fmt(npds.get("sd"), 3),
                     npds.get("n", "-"), npd_.get("status", "ok")])
        _kv_table(doc, ["Residual", "Mean", "SD", "n", "Status"], rows)
        if "pct_outside_1_96" in npds:
            doc.add_paragraph(f"npd outside ±1.96: {_fmt(npds.get('pct_outside_1_96'), 1)}% "
                              "(≈5% expected under a correctly specified model).")
        for key, blob in (("CWRES", cw), ("npd", npd_)):
            if blob.get("status") not in (None, "ok") and blob.get("message"):
                doc.add_paragraph(f"{key}: {blob['message']}")

    fr = state.forest_results
    if fr and fr.get("status") == "ok":
        n += 1
        doc.add_heading(f"{n}. Covariate effects (forest)", level=1)
        doc.add_paragraph(
            f"Covariate effects from the {fr.get('source')} fit of {fr.get('label')}, "
            f"evaluated at the {fr.get('percentiles')} percentiles of the current dataset; "
            f"geometric mean ratios with {_fmt(100 * float(fr.get('ci_level') or 0), 0)}% CI.")
        _kv_table(doc, ["Parameter", "Covariate", "Evaluated at", "GMR", "CI", "CI source"],
                  [[r["param"], r["covariate"], r["eval_label"], _fmt(r.get("gmr"), 3),
                    (f"{_fmt(r.get('ci_lo'), 3)}–{_fmt(r.get('ci_hi'), 3)}"
                     if r.get("ci_lo") is not None else "-"),
                    r.get("ci_source")] for r in (fr.get("rows") or [])])
        for note in (fr.get("notes") or []):
            doc.add_paragraph(str(note))

    vp = state.vpc_results
    if vp and vp.get("status") == "ok":
        n += 1
        g = vp.get("gof") or {}
        doc.add_heading(f"{n}. Model evaluation (GOF / VPC)", level=1)
        doc.add_paragraph(f"Goodness-of-fit (log scale): R²(IPRED) = {_fmt(g.get('r2_log_ipred'), 3)}, "
                          f"RMSE = {_fmt(g.get('rmse_log_ipred'), 3)}, n = {g.get('n')}. "
                          f"A 5/50/95 VPC band was generated at the {vp.get('vpc_dose')} dose level.")

    if state.protocol_sim:
        n += 1
        _protocol_sim_section(doc, state, n, audit_trail=audit_trail)


def _find_gate_approval(audit_trail: tuple[dict[str, Any], ...],
                        gate_label: str = "Adversarial review") -> dict[str, Any] | None:
    """Find the human_review approval for a specific gate in the audit trail.

    Matches audit entries whose ``inputs.review.label`` contains *gate_label*
    (case-insensitive).  This prevents attributing an earlier gate's approval
    (e.g. the plan gate or special-population gate) to the adversarial-review
    gate.  Returns a dict with ``actor``, ``timestamp`` and ``reason`` keys,
    or None if no matching approval is found.  Searches backwards so the most
    recent matching approval wins.
    """
    needle = gate_label.lower()
    for entry in reversed(audit_trail):
        if (entry.get("tool") == "human_review"
                and isinstance(entry.get("action"), str)
                and entry["action"].startswith("approved")):
            review = (entry.get("inputs") or {}).get("review") or {}
            label = (review.get("label") or "").lower()
            if needle in label:
                return {
                    "actor": entry.get("actor") or "unknown",
                    "timestamp": entry.get("timestamp") or "unknown",
                    "reason": entry.get("reason") or "(no reason recorded)",
                }
    return None


def _protocol_sim_section(doc: Document, state: PharmState, section_num: int = 5,
                          audit_trail: tuple[dict[str, Any], ...] = ()) -> dict[str, Any]:
    """Render the 'Protocol simulation' section strictly from ``state.protocol_sim``
    and ``state.review_results`` without inventing or transforming any numbers."""
    ps = state.protocol_sim or {}
    source_fit = ps.get("source_fit") or {}
    plan = ps.get("plan") or {}
    pta = ps.get("pta") or {}
    sp = ps.get("special_populations") or {}
    unc = ps.get("uncertainty") or {}
    prov = ps.get("provenance") or {}
    rev = state.review_results or {}

    skipped_steps: list[str] = []
    md_lines: list[str] = [f"## {section_num}. Protocol simulation"]

    doc.add_heading(f"{section_num}. Protocol simulation", level=1)

    # 1. Approved simulation plan
    doc.add_heading(f"{section_num}.1 Approved simulation plan", level=2)
    md_lines.append(f"### {section_num}.1 Approved simulation plan")
    if plan:
        tw = plan.get("target_window") or {}
        tw_prov = tw.get("provenance") or {}
        plan_summary = (
            f"Virtual cohort: n_subjects = {plan.get('n_subjects')}, "
            f"replicates = {plan.get('replicates')}, seed = {plan.get('seed')}, "
            f"uncertainty_method = {plan.get('uncertainty_method')}. "
            f"Target window on {tw.get('metric', 'auc')}: "
            f"[{_fmt(tw.get('low'), 3)}, {_fmt(tw.get('high'), 3)}] "
            f"(cmax_max = {_fmt(tw.get('cmax_max'), 3)}, "
            f"ctrough_min = {_fmt(tw.get('ctrough_min'), 3)}; "
            f"source: {tw_prov.get('source', 'n/a')} — {tw_prov.get('citation', 'n/a')})."
        )
        doc.add_paragraph(plan_summary)
        md_lines.append(plan_summary)

        band_rows = [
            [
                b.get("name"),
                f"{b.get('wt_kg', ['-', '-'])[0]}–{b.get('wt_kg', ['-', '-'])[1]}",
                f"{b.get('age_yr', ['-', '-'])[0]}–{b.get('age_yr', ['-', '-'])[1]}",
                f"{b.get('crcl_ml_min', ['-', '-'])[0]}–{b.get('crcl_ml_min', ['-', '-'])[1]}",
            ]
            for b in (plan.get("bands") or [])
        ]
        if band_rows:
            _kv_table(doc, ["Band", "Weight (kg)", "Age (yr)", "CrCl (mL/min)"], band_rows)
            md_lines.append("| Band | Weight (kg) | Age (yr) | CrCl (mL/min) |")
            md_lines.append("| --- | --- | --- | --- |")
            for br in band_rows:
                md_lines.append(f"| {br[0]} | {br[1]} | {br[2]} | {br[3]} |")

        reg_rows = [
            [
                r.get("label"),
                _fmt(r.get("dose_mg"), 1),
                _fmt(r.get("tau_h"), 1),
                r.get("route", "oral"),
                r.get("n_doses", 5),
            ]
            for r in (plan.get("regimens") or [])
        ]
        if reg_rows:
            _kv_table(doc, ["Regimen", "Dose (mg)", "Interval (h)", "Route", "Doses"], reg_rows)
            md_lines.append("| Regimen | Dose (mg) | Interval (h) | Route | Doses |")
            md_lines.append("| --- | --- | --- | --- | --- |")
            for rr in reg_rows:
                md_lines.append(f"| {rr[0]} | {rr[1]} | {rr[2]} | {rr[3]} | {rr[4]} |")
    else:
        skipped_steps.append("propose_simulation_plan")
        msg = "Skipped step: propose_simulation_plan (no simulation plan recorded)."
        doc.add_paragraph(msg)
        md_lines.append(msg)

    # Check optional dose_sweep and special_populations steps
    if not (ps.get("dose_sweep") or (state.dose_sweep_results or {}).get("status") == "ok"):
        skipped_steps.append("run_dose_sweep")
        msg = "Skipped step: run_dose_sweep."
        doc.add_paragraph(msg)
        md_lines.append(msg)

    if sp and sp.get("status") == "ok":
        sp_msg = (
            f"Special-population simulation completed for plan bands: "
            f"{sp.get('bands_simulated')} (seed = {sp.get('seed')})."
        )
        doc.add_paragraph(sp_msg)
        md_lines.append(sp_msg)
    else:
        skipped_steps.append("simulate_special_populations")
        msg = "Skipped step: simulate_special_populations."
        doc.add_paragraph(msg)
        md_lines.append(msg)

    # 2. PTA per band per regimen with MC SE and parameter-uncertainty interval
    doc.add_heading(f"{section_num}.2 PTA by band and regimen", level=2)
    md_lines.append(f"### {section_num}.2 PTA by band and regimen")
    pta_rows = pta.get("rows") or []
    if pta_rows:
        if not (unc and unc.get("status") == "ok"):
            skipped_steps.append("propagate_parameter_uncertainty")
            unc_msg = "Skipped step: propagate_parameter_uncertainty (parameter-uncertainty intervals not computed)."
            if unc and unc.get("status") == "unverifiable":
                unc_msg = f"Parameter-uncertainty intervals not computed: {unc.get('reason')}."
            doc.add_paragraph(unc_msg)
            md_lines.append(unc_msg)
        elif unc.get("approximation"):
            unc_msg = (
                "Parameter-uncertainty intervals are an asymptotic log-normal approximation "
                "from the fit's theta RSE%, not a bootstrap, SIR or profile result "
                f"(requested: {unc.get('requested_method')})."
            )
            doc.add_paragraph(unc_msg)
            md_lines.append(unc_msg)
        tbl_rows = []
        pta_cols = [
            "Band", "Regimen", "Dose (mg)", "PTA", "MC SE",
            "95% Param CI", "AUC Median", "Cmax Median", "Ctrough Median",
        ]
        md_lines.append("| " + " | ".join(pta_cols) + " |")
        md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in pta_rows:
            ci_str = (
                f"{_fmt(r.get('pta_ci_lo'), 4)}–{_fmt(r.get('pta_ci_hi'), 4)}"
                if r.get("pta_ci_lo") is not None and r.get("pta_ci_hi") is not None
                else "-"
            )
            auc_med = _fmt((r.get("auc") or {}).get("median"), 3)
            cmax_med = _fmt((r.get("cmax") or {}).get("median"), 3)
            ctr_med = _fmt((r.get("ctrough") or {}).get("median"), 4)
            row_cells = [
                r.get("band"),
                r.get("regimen"),
                _fmt(r.get("dose_mg"), 1),
                _fmt(r.get("pta"), 4),
                _fmt(r.get("mc_se"), 4),
                ci_str,
                auc_med,
                cmax_med,
                ctr_med,
            ]
            tbl_rows.append(row_cells)
            md_lines.append(" | ".join(["", *[str(c) for c in row_cells], ""]).strip())
        _kv_table(
            doc,
            pta_cols,
            tbl_rows,
        )
    else:
        skipped_steps.append("simulate_trial_pta")
        msg = "Skipped step: simulate_trial_pta (no PTA rows available)."
        doc.add_paragraph(msg)
        md_lines.append(msg)

    # 3. Recommended regimen per band
    doc.add_heading(f"{section_num}.3 Recommended regimen per band", level=2)
    md_lines.append(f"### {section_num}.3 Recommended regimen per band")
    recs = ps.get("recommendations") or pta.get("recommendations") or []
    if recs:
        rec_rows = []
        md_lines.append("| Band | Recommended Regimen | PTA | MC SE | 95% Param CI | Driving Metric |")
        md_lines.append("| --- | --- | --- | --- | --- | --- |")
        for rec in recs:
            ci_str = (
                f"{_fmt(rec.get('pta_ci_lo'), 4)}–{_fmt(rec.get('pta_ci_hi'), 4)}"
                if rec.get("pta_ci_lo") is not None and rec.get("pta_ci_hi") is not None
                else "-"
            )
            r_cells = [
                rec.get("band"),
                rec.get("recommended_regimen") or "none",
                _fmt(rec.get("pta"), 4),
                _fmt(rec.get("mc_se"), 4),
                ci_str,
                rec.get("driving_metric", "-"),
            ]
            rec_rows.append(r_cells)
            md_lines.append(" | ".join(["", *[str(c) for c in r_cells], ""]).strip())
        _kv_table(
            doc,
            ["Band", "Recommended Regimen", "PTA", "MC SE", "95% Param CI", "Driving Metric"],
            rec_rows,
        )
    else:
        doc.add_paragraph("No regimen recommendations recorded.")
        md_lines.append("No regimen recommendations recorded.")

    # 4. Adversarial review findings (HIGH / CRITICAL) and disposition
    doc.add_heading(f"{section_num}.4 Adversarial review findings and disposition", level=2)
    md_lines.append(f"### {section_num}.4 Adversarial review findings and disposition")
    high_crit = [
        f for f in (rev.get("findings") or [])
        if f.get("severity") in ("HIGH", "CRITICAL")
    ]
    if high_crit:
        f_rows = []
        md_lines.append("| ID | Severity | Target | Evidence | Disposition |")
        md_lines.append("| --- | --- | --- | --- | --- |")
        for f in high_crit:
            if f.get("resolved"):
                disp = "Resolved"
            elif f.get("disposition"):
                disp = str(f["disposition"])
            else:
                # Look for the human_review approval that followed the adversarial
                # review gate in the audit trail.
                gate_approval = _find_gate_approval(audit_trail)
                if gate_approval:
                    disp = (
                        f"Unresolved; gate approved by {gate_approval['actor']} "
                        f"on {gate_approval['timestamp']}: {gate_approval['reason']}"
                    )
                else:
                    disp = "Unresolved; no gate decision recorded"
            f_cells = [
                f.get("id"),
                f.get("severity"),
                f.get("target"),
                f.get("evidence"),
                disp,
            ]
            f_rows.append(f_cells)
            md_lines.append(" | ".join(["", *[str(c) for c in f_cells], ""]).strip())
        _kv_table(doc, ["ID", "Severity", "Target", "Evidence", "Disposition"], f_rows)
    else:
        clean_msg = f"Zero HIGH or CRITICAL findings (goal_met = {rev.get('goal_met', True)})."
        doc.add_paragraph(clean_msg)
        md_lines.append(clean_msg)

    # 5. Provenance block
    doc.add_heading(f"{section_num}.5 Simulation provenance", level=2)
    md_lines.append(f"### {section_num}.5 Simulation provenance")
    unc_method = unc.get("method") or "n/a"
    unc_requested = unc.get("requested_method") or plan.get("uncertainty_method") or "n/a"
    unc_approx = bool(unc.get("approximation")) if unc else None
    prov_rows = [
        ["source_fit_id", prov.get("source_fit_id") or source_fit.get("fit_id") or "n/a"],
        ["review_hash", prov.get("review_hash") or "n/a"],
        ["plan_hash", prov.get("plan_hash") or "n/a"],
        ["seed", prov.get("seed") if prov.get("seed") is not None else plan.get("seed", "n/a")],
        ["uncertainty_method", unc_method],
        ["uncertainty_requested", unc_requested],
        ["uncertainty_approximation",
         "n/a" if unc_approx is None else ("yes" if unc_approx else "no")],
    ]
    _kv_table(doc, ["Field", "Value"], prov_rows)
    md_lines.append("| Field | Value |")
    md_lines.append("| --- | --- |")
    for pr in prov_rows:
        md_lines.append(f"| {pr[0]} | {pr[1]} |")

    return {
        "source_fit_id": prov.get("source_fit_id") or source_fit.get("fit_id"),
        "review_hash": prov.get("review_hash"),
        "plan_hash": prov.get("plan_hash"),
        "seed": prov.get("seed") if prov.get("seed") is not None else plan.get("seed"),
        "uncertainty_method": unc_method,
        "uncertainty_requested": unc_requested,
        "uncertainty_approximation": unc_approx,
        "n_pta_rows": len(pta_rows),
        "recommendations": recs,
        "high_critical_findings": high_crit,
        "skipped_steps": skipped_steps,
        "rendered_markdown": "\n".join(md_lines),
    }


def _reproducibility_section(doc: Document, state: PharmState) -> None:
    """Software/platform provenance, dataset integrity hash, and seed — so the
    report's results can be reproduced from the record alone (ALCOA+)."""
    from app.core.provenance import run_provenance_report_lines
    meta = state.dataset_metadata or {}
    doc.add_heading("Software & reproducibility", level=1)
    for line in run_provenance_report_lines(state):    # recorded at session creation; never the current server
        doc.add_paragraph(line)
    sha = meta.get("dataset_sha256")
    if sha and sha != "n/a":
        doc.add_paragraph(f"Source dataset SHA-256: {sha}")
    doc.add_paragraph(
        "Stochastic procedures (SAEM E-step, VPC/pcVPC, simulation-based NPDE) use "
        "a fixed RNG seed (20250614) so results are bit-reproducible; FOCE-I is "
        "deterministic. All computations are recorded in a SHA-256 hash-chained, "
        "identity-stamped audit trail (verify via GET /sessions/{id}/audit).")


def generate_report(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    from app.core.provenance import run_provenance_report_lines
    run_provenance_report_lines(state)                  # refuse a quarantined (mismatched) session before any work
    title = args.get("title", "Pharmacokinetic Analysis Report")
    doc = Document()
    doc.add_heading(title, level=0)

    meta = state.dataset_metadata or {}
    doc.add_heading("1. Dataset", level=1)
    doc.add_paragraph(
        f"Dataset {meta.get('dataset_id', 'n/a')}: {meta.get('n_records', '?')} records, "
        f"{meta.get('n_subjects', '?')} subjects, dose levels "
        f"{meta.get('dose_levels', 'n/a')}.")

    doc.add_heading("2. Methods", level=1)
    doc.add_paragraph(METHODS_NCA)

    doc.add_heading("3. Results", level=1)
    params = state.nca_parameters or []
    doc.add_heading("3.1 Per-subject NCA parameters", level=2)
    doc.add_paragraph(
        f"Non-compartmental parameters for all {len(params)} subjects. "
        "λz adjR² and λz n report the terminal-phase fit quality and the number "
        "of points used; %extrap is the fraction of AUCinf obtained by "
        "extrapolation beyond the last measured concentration.")
    _subject_table(doc, params)

    descriptive = (state.nca_summary or {}).get("descriptive") or []
    if descriptive:
        doc.add_heading("3.2 Summary statistics", level=2)
        doc.add_paragraph(
            "Descriptive statistics per parameter: N = subjects with a valid value; "
            "SD uses n−1; CV% = SD/mean; GM = geometric mean; gCV% = geometric CV "
            "(√(exp(s²_log) − 1)).")
        grouped = _dose_grouped_meaningfully(state.nca_summary)
        for g in descriptive:
            if g.get("group") != "all" and not grouped:
                continue
            doc.add_paragraph(f"{g.get('label')} (n = {g.get('n')})")
            _descriptive_table(doc, g.get("parameters") or [])

    if state.nca_summary and _dose_grouped_meaningfully(state.nca_summary):
        doc.add_heading("3.3 Dose-group summary", level=2)
        doc.add_paragraph(
            "Geometric mean (GM) and geometric CV% by dose level; t½ as median.")
        _summary_table(doc, state.nca_summary)

    doc.add_heading("4. Quality Control", level=1)
    doc.add_paragraph(f"QC verdict: {state.qc_verdict or 'not run'}.")
    for c in (state.qc_checklist or []):
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"[{c['status']}] {c['check']}: ").bold = True
        p.add_run(c["detail"])

    _extra_sections(doc, state, audit_trail=ctx.audit_trail)
    _reproducibility_section(doc, state)

    out_dir = Path(ctx.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = str(out_dir / f"report_{meta.get('dataset_id', 'analysis')}.docx")
    doc.save(path)

    sections: dict[str, Any] = {
        "dataset": meta.get("dataset_id"),
        "methods": "NCA (linear-up/log-down, best-fit lambda_z)",
        "qc_verdict": state.qc_verdict,
    }
    if state.protocol_sim:
        sections["protocol_simulation"] = _protocol_sim_section(Document(), state, 5,
                                                                audit_trail=ctx.audit_trail)
    return ToolResult(
        summary=f"Generated DOCX report at {path}.",
        action=f"generate_report -> {path}",
        writes={"report_path": path, "report_sections": sections},
        result={"report_path": path, "sections": sections},
    )


TOOLS = [
    Tool("generate_report",
         "Generate an FDA/EMA-style DOCX report (dataset, methods, per-subject "
         "NCA table, optional dose-group summary, QC) from the current state.",
         "report",
         {"type": "object",
          "properties": {"title": {"type": "string"}}, "required": []},
         generate_report,
         uses_audit_trail=True),
]
