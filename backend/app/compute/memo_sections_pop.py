"""Briefing-memo sections, part 2: population fit, covariates, diagnostics, indirect comparison.

Same contract as ``memo_sections``: PharmState values only, every number tagged with
the audit entry that produced it, template wording free of digits.
"""
from __future__ import annotations

from typing import Any

from app.compute.memo import AuditTrace
from app.compute.memo_model import (
    MISSING,
    Block,
    Bullet,
    Cited,
    Heading,
    Para,
    Recorder,
    Section,
    Table,
)
from app.compute.textguard import reject_numerals
from app.core.pharmstate import PharmState


def _ok(blob: Any) -> dict[str, Any] | None:
    return blob if isinstance(blob, dict) and blob.get("status") == "ok" else None


def _opt(prefix: str, cell: str) -> str:
    return f"{prefix}{cell}" if cell != MISSING else ""


# ── population fit ─────────────────────────────────────────────────────────────
def population(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    nl = _ok(state.nlme_results)
    if not nl:
        return None
    c = rec.at(trace.index_for("nlme_results"))
    label, method = (c.lit("nlme_results.label", nl.get("label")),
                     c.lit("nlme_results.method", nl.get("method")))
    ofv = c.num("nlme_results.ofv", nl.get("ofv"), 1)
    verdict = {True: "converged", False: "did not converge"}.get(nl.get("converged"),
                                                                "convergence not reported")
    iiv = c.lit("nlme_results.iiv_params", ", ".join(nl.get("iiv_params") or []))
    text = (f"Mixed-effects fit ({method}) of {label}"
            + _opt("; ", c.lit("nlme_results.error_model", nl.get("error_model")))
            + (" residual error" if nl.get("error_model") else "")
            + _opt("; IIV on ", iiv)
            + _opt("; OFV ", ofv)
            + _opt("; condition number ", c.num("nlme_results.condition_number",
                                                nl.get("condition_number"), 1))
            + f"; {verdict}.")
    rows = []
    for name, value in (nl.get("theta") or {}).items():
        rows.append((c.lit("nlme_results.theta_name", name),
                     c.num(f"nlme_results.theta.{name}", value, 3),
                     c.num(f"nlme_results.theta_rse_pct.{name}", (nl.get("theta_rse_pct") or {}).get(name), 1),
                     c.num(f"nlme_results.omega_cv_pct.{name}", (nl.get("omega_cv_pct") or {}).get(name), 1),
                     c.num(f"nlme_results.shrinkage_pct.{name}", (nl.get("shrinkage_pct") or {}).get(name), 1)))
    blocks: list[Block] = [Para(text)]
    if rows:
        blocks.append(Table(("Parameter", "Typical value", "RSE %", "IIV CV %", "Shrinkage %"), tuple(rows)))
    headline = f"Population fit: {label} by {method}" + _opt(", OFV ", ofv) + f", {verdict}."
    return Section("population", "Population fit", tuple(blocks), headline)


# ── covariates ─────────────────────────────────────────────────────────────────
def _nlme_effects(nl: dict[str, Any], c: Cited) -> list[Block]:
    out: list[Block] = []
    for e in nl.get("covariate_effects") or []:
        rse = c.num("nlme_results.covariate_rse_pct", e.get("rse_pct"), 1, unit="%")
        out.append(Bullet(f"{c.lit('nlme_results.covariate_param', e.get('param'))}: "
                          f"{c.lit('nlme_results.covariate_effect', e.get('description'))}"
                          + (f" (RSE {rse})" if rse != MISSING else "")))
    return out


def _scm(scm: dict[str, Any], c: Cited) -> Para:
    selected = ", ".join(f"{s.get('param')}~{s.get('covariate')} ({s.get('kind')})"
                         for s in scm.get("selected") or []) or "none"
    cand = c.count("scm_results.n_candidates", scm.get("n_candidates"))
    fp, bp = (c.sig("scm_results.forward_p", scm.get("forward_p")),
              c.sig("scm_results.backward_p", scm.get("backward_p")))
    base, final = (c.num("scm_results.base_ofv", scm.get("base_ofv"), 1),
                   c.num("scm_results.final_ofv", scm.get("final_ofv"), 1))
    return Para("Stepwise covariate modelling"
                + (f": {cand} candidate effects tested" if cand != MISSING else "")
                + f"; selected {c.lit('scm_results.selected', selected)}"
                + (f" (forward p below {fp}, backward p below {bp})" if MISSING not in (fp, bp) else "")
                + (f"; OFV {base} to {final}" if MISSING not in (base, final) else "") + ".")


def _forest(fr: dict[str, Any], c: Cited) -> list[Block]:
    level = c.level("forest_results.ci_level_pct", fr.get("ci_level"))
    intro = ("Covariate effects as geometric mean ratios"
             + (f" with {level} confidence intervals" if level != MISSING else "")
             + f" from the {c.lit('forest_results.source', fr.get('source'))} fit.")
    rows = []
    for i, r in enumerate(fr.get("rows") or []):
        pre = f"forest_results.{i}"
        rows.append((c.lit(f"{pre}.param", r.get("param")), c.lit(f"{pre}.covariate", r.get("covariate")),
                     c.lit(f"{pre}.eval_label", r.get("eval_label")), c.num(f"{pre}.gmr", r.get("gmr"), 3),
                     c.span(f"{pre}.ci", r.get("ci_lo"), r.get("ci_hi"), 3),
                     c.lit(f"{pre}.ci_source", r.get("ci_source"))))
    table = Table(("Parameter", "Covariate", "Evaluated at", "GMR", "CI", "CI source"), tuple(rows))
    return [Para(intro), table] if rows else [Para(intro)]


def covariates(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    nl, scm, fr = _ok(state.nlme_results), _ok(state.scm_results), _ok(state.forest_results)
    blocks: list[Block] = []
    heads: list[str] = []
    effects = _nlme_effects(nl, rec.at(trace.index_for("nlme_results"))) if nl else []
    if effects:
        blocks += [Para("Covariate effects in the population model:"), *effects]
        n_eff = rec.at(trace.index_for("nlme_results")).count("nlme_results.n_covariate_effects", len(effects))
        heads.append(f"{n_eff} covariate effect(s) in the population model")
    if scm:
        cs = rec.at(trace.index_for("scm_results"))
        blocks.append(_scm(scm, cs))
        heads.append(f"SCM selected {cs.count('scm_results.n_selected', len(scm.get('selected') or []))} effect(s)")
    if fr:
        blocks += _forest(fr, rec.at(trace.index_for("forest_results")))
    if not blocks:
        return None
    headline = "Covariates: " + "; ".join(heads) + "." if heads else "Covariate effects evaluated by forest plot."
    return Section("covariates", "Covariate effects", tuple(blocks), headline)


# ── diagnostics ────────────────────────────────────────────────────────────────
def _diag_rows(dg: dict[str, Any], c: Cited) -> list[tuple[str, str, str]]:
    iw = (dg.get("residuals") or {}).get("summary") or {}
    rows = [("IWRES mean", c.num("diagnostics_results.iwres.mean", iw.get("iwres_mean", iw.get("mean")), 3),
             c.count("diagnostics_results.iwres.n", iw.get("n"))),
            ("IWRES SD", c.num("diagnostics_results.iwres.sd", iw.get("iwres_sd", iw.get("sd")), 3),
             c.count("diagnostics_results.iwres.n", iw.get("n")))]
    cw = _ok(dg.get("cwres")) and (dg["cwres"].get("summary") or {})
    if cw:
        n = c.count("diagnostics_results.cwres.n", cw.get("n"))
        rows += [("CWRES mean", c.num("diagnostics_results.cwres.mean", cw.get("cwres_mean"), 3), n),
                 ("CWRES SD", c.num("diagnostics_results.cwres.sd", cw.get("cwres_sd"), 3), n)]
    npd = _ok(dg.get("npde")) and (dg["npde"].get("summary") or {})
    if npd:
        n = c.count("diagnostics_results.npd.n", npd.get("n"))
        rows += [("npd mean", c.num("diagnostics_results.npd.mean", npd.get("mean"), 3), n),
                 ("npd SD", c.num("diagnostics_results.npd.sd", npd.get("sd"), 3), n),
                 ("npd beyond the nominal limits (%)",
                  c.num("diagnostics_results.npd.pct_outside", npd.get("pct_outside_1_96"), 1), n)]
    return rows


def diagnostics(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    vp, dg = _ok(state.vpc_results), _ok(state.diagnostics_results)
    if not (vp or dg):
        return None
    rows: list[tuple[str, str, str]] = []
    head: list[str] = []
    if vp:
        cv = rec.at(trace.index_for("vpc_results"))
        g = vp.get("gof") or {}
        r2 = cv.num("vpc_results.gof.r2_log_ipred", g.get("r2_log_ipred"), 3)
        n = cv.count("vpc_results.gof.n", g.get("n"))
        rows += [("R\u00b2 of IPRED (log scale)", r2, n),
                 ("RMSE of IPRED (log scale)", cv.num("vpc_results.gof.rmse_log_ipred",
                                                      g.get("rmse_log_ipred"), 3), n)]
        head.append(f"goodness of fit R\u00b2 {r2}")
    blocks: list[Block] = []
    if dg:
        rows += _diag_rows(dg, rec.at(trace.index_for("diagnostics_results")))
        missing = [name for name, key in (("CWRES", "cwres"), ("npd", "npde")) if not _ok(dg.get(key))]
        if missing:
            blocks.append(Para(f"{' and '.join(missing)} not computed: a converged mixed-effects fit is required."))
    kept = tuple(r for r in rows if r[1] != MISSING)
    blocks = [*((Table(("Metric", "Value", "n"), kept),) if kept else ()), *blocks]
    return Section("diagnostics", "Model diagnostics", tuple(blocks),
                   "Diagnostics: " + (", ".join(head) if head else "residual summaries computed") + ".")


# ── indirect comparison ────────────────────────────────────────────────────────
def _indirect_rows(i: int, comp: dict[str, Any], c: Cited) -> list[tuple[str, ...]]:
    """The computed indirect row, then the two direct legs. The legs are the user's
    INPUTS (copied from the literature, typed into the tool call), recorded as kind
    ``input`` and marked user-supplied; only an SE derived from a CI is computed."""
    t = comp.get("treatments") or {}
    pre = f"indirect_results.{i}"
    rows: list[tuple[str, ...]] = [(
        "Indirect", "computed", c.num(f"{pre}.estimate", comp.get("estimate"), 3),
        c.span(f"{pre}.ci", comp.get("ci_lower"), comp.get("ci_upper"), 3),
        c.level(f"{pre}.ci_level_pct", comp.get("ci_level")),
        c.num(f"{pre}.se", comp.get("se"), 3), c.num(f"{pre}.z", comp.get("z"), 2),
        c.sig(f"{pre}.p_value", comp.get("p_value")))]
    given = c.inputs()
    for leg, a, b in (("ab", "A", "B"), ("cb", "C", "B")):
        d = (comp.get("inputs") or {}).get(leg) or {}
        from_ci = d.get("se_source") == "from_ci"
        ci = given.span(f"{pre}.inputs.{leg}.ci", d.get("ci_lower"), d.get("ci_upper"), 3) if from_ci else MISSING
        level = given.level(f"{pre}.inputs.{leg}.ci_level_pct", d.get("ci_level")) if from_ci else MISSING
        if from_ci and d.get("ci_level_source") == "default" and level != MISSING:
            level += " assumed"
        se = (c if from_ci else given).num(f"{pre}.inputs.{leg}.se", d.get("se"), 3)
        rows.append((f"Direct {c.lit(f'{pre}.{leg}.treatments', t.get(a, a) + ' vs ' + t.get(b, b))}",
                     "user-supplied" + (", SE derived from its CI" if from_ci else ""),
                     given.num(f"{pre}.inputs.{leg}.estimate", d.get("estimate"), 3), ci, level,
                     se, MISSING, MISSING))
    return rows


def _check_labels(comp: dict[str, Any]) -> None:
    """Treatment names are model-supplied free text: they may carry no numeral."""
    names = [comp.get("contrast"), *((comp.get("treatments") or {}).values())]
    for name in names:
        reject_numerals(str(name or ""), "treatment label")


def indirect(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    comps = [c for c in (state.indirect_results or {}).get("comparisons") or [] if _ok(c)]
    if not comps:
        return None
    for comp in comps:
        _check_labels(comp)
    blocks: list[Block] = []
    heads: list[str] = []
    for i, comp in enumerate(comps):
        c = rec.at(trace.by_output("indirect_comparison", comp))
        pre = f"indirect_results.{i}"
        contrast, scale = c.lit(f"{pre}.contrast", comp.get("contrast")), c.lit(f"{pre}.scale", comp.get("scale"))
        analysis_scale = c.lit(f"{pre}.analysis_scale", comp.get("analysis_scale"))
        blocks += [Heading(f"{contrast} ({scale})", 2),
                   Table(("Comparison", "Source", "Estimate", "CI", "CI level", "SE", "z", "p"),
                         tuple(_indirect_rows(i, comp, c))),
                   Para("The CI level of the indirect row is the level of the computed interval; each "
                        "direct row shows the level of its own published interval (marked assumed when "
                        "none was given). Direct rows are user-supplied inputs, not computed results. "
                        f"Standard errors are on the {analysis_scale} scale. "
                        + c.lit(f"{pre}.assumptions", comp.get("assumptions")))]
        heads.append(f"{contrast}: {scale} {c.num(f'{pre}.estimate', comp.get('estimate'), 3)} "
                     f"(CI {c.span(f'{pre}.ci', comp.get('ci_lower'), comp.get('ci_upper'), 3)}), "
                     f"p {c.sig(f'{pre}.p_value', comp.get('p_value'))}")
    return Section("indirect", "Indirect comparison", tuple(blocks), "Indirect comparison. " + "; ".join(heads) + ".")
