"""Briefing-memo sections, part 3: exposure-response fits and the dose-selection decision.

Same contract as ``memo_sections``: PharmState values only, every number tagged with
the audit entry that produced it, template wording free of digits. Fit labels and
exposure labels are free text from a tool call, so one that carries a numeral is
withheld rather than printed (nothing numeric may enter the memo except through the
traced value table).

Fits are stored side by side under one state field but each comes from its own run, so
each fit (and its bootstrap, and the dose selection) is resolved to the audit entry
whose output it IS, never to the newest E-R entry. ``current_er`` withholds, one by one,
fits computed on a previously loaded dataset, and a dose selection computed from such a
fit even when the selection itself ran after the reload.
"""
from __future__ import annotations

from typing import Any

from app.compute.er_common import bootstrap_output, fit_core
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
)
from app.compute.textguard import has_numeral
from app.core.pharmstate import PharmState

LABEL_WITHHELD = "label withheld (contains digits)"
SELECTION_STATUSES = ("ok", "no_feasible_dose")


def _label(c: Cited, key: str, text: Any) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return MISSING
    return LABEL_WITHHELD if has_numeral(clean) else c.lit(key, clean)


# ── provenance: which run produced each stored result ──────────────────────────
def fit_index(trace: AuditTrace, fit: dict[str, Any]) -> int | None:
    return trace.by_output("fit_exposure_response", fit_core(fit))


def selection_index(trace: AuditTrace, ds: dict[str, Any]) -> int | None:
    return trace.by_output("select_optimal_dose", ds)


def _other_dataset(dataset_id: Any, state: PharmState) -> bool:
    return bool(dataset_id) and bool(state.dataset_id) and dataset_id != state.dataset_id


def _predates_load(index: int | None, field: str, trace: AuditTrace) -> bool:
    """``index`` is before the latest load. When the run cannot be matched (state edited
    since), the field's newest producing entry dates it: if even that predates the load,
    so does every result of the field."""
    return trace.before_latest_load(index if index is not None else trace.index_for(field))


def _fit_is_stale(fit: dict[str, Any], state: PharmState, trace: AuditTrace) -> bool:
    return (_predates_load(fit_index(trace, fit), "er_results", trace)
            or _other_dataset(fit.get("dataset_id"), state))


def _selection_is_stale(ds: dict[str, Any], state: PharmState, trace: AuditTrace) -> bool:
    """Its own run predates the reload, or a fit it was computed from does (or was
    computed on another dataset) -- whether or not that fit is still in state."""
    if _predates_load(selection_index(trace, ds), "dose_selection_results", trace):
        return True
    return any(trace.before_latest_load(trace.by_digest("fit_exposure_response", str(f.get("fit_sha256"))))
               or _other_dataset(f.get("dataset_id"), state)
               for f in (ds.get("input_fits") or {}).values() if isinstance(f, dict))


def _fit_name(label: str) -> str:
    return "Exposure-response fit " + ("(label withheld: contains digits)" if has_numeral(label) else label)


def current_er(state: PharmState, trace: AuditTrace) -> tuple[PharmState, tuple[tuple[str, str], ...]]:
    """``state`` without the E-R fits and dose selection computed on a previously loaded
    dataset, and (key, name) of each one withheld."""
    er = state.er_results or {}
    fits = er.get("fits") or {}
    stale = [k for k, f in fits.items() if isinstance(f, dict) and _fit_is_stale(f, state, trace)]
    ds = state.dose_selection_results
    ds_stale = isinstance(ds, dict) and _selection_is_stale(ds, state, trace)
    withheld = [(f"er_results.fits.{k}", _fit_name(k)) for k in stale]
    withheld += [("dose_selection_results", "Dose selection")] if ds_stale else []
    if not withheld:
        return state, ()
    kept = {k: f for k, f in fits.items() if k not in stale}
    update: dict[str, Any] = {"er_results": {**er, "fits": kept} if kept else None}
    if ds_stale:
        update["dose_selection_results"] = None
    return state.model_copy(update=update), tuple(withheld)


# ── exposure-response fits ─────────────────────────────────────────────────────
def _effect(fit: dict[str, Any], pre: str) -> tuple[str, str, dict[str, Any], str, Any]:
    """(effect name, state key prefix, {estimate, lo, hi}, p key, p value) of the exposure term."""
    if fit.get("model") == "logistic":
        o = fit.get("or_per_sd") or {}
        p = ((fit.get("coef") or {}).get("slope") or {}).get("p")
        return "OR per SD", f"{pre}.or_per_sd", o, f"{pre}.coef.slope.p", p
    cov = ((fit.get("cox") or {}).get("covariates") or [{}])[0]
    est = {"estimate": cov.get("hr_per_sd"), "lo": cov.get("hr_per_sd_lo"), "hi": cov.get("hr_per_sd_hi")}
    return "HR per SD", f"{pre}.cox.covariates.0.hr_per_sd", est, f"{pre}.cox.covariates.0.p", cov.get("p")


def _fit_row(label: str, fit: dict[str, Any], c: Cited) -> tuple[tuple[str, ...], str]:
    pre = f"er_results.fits.{label}"
    name, key, est, p_key, p = _effect(fit, pre)
    shown = _label(c, f"{pre}.label", label)
    exposure = _label(c, f"{pre}.exposure_label",
                      fit.get("exposure_label") or (fit.get("data") or {}).get("exposure_label"))
    value = c.num(f"{key}.estimate", est.get("estimate"), 3)
    row = (shown, c.lit(f"{pre}.model", fit.get("model")), exposure,
           c.count(f"{pre}.n", fit.get("n")), c.count(f"{pre}.n_events", fit.get("n_events")),
           name, value, c.span(key, est.get("lo"), est.get("hi"), 3),
           c.level(f"{pre}.ci_level_pct", fit.get("ci_level")), c.sig(p_key, p))
    head = f"{shown} {name} {value}" if value != MISSING else ""
    return row, head


def _bootstrap_bullet(label: str, fit: dict[str, Any], trace: AuditTrace, rec: Recorder) -> Bullet | None:
    boot = fit.get("bootstrap")
    if not isinstance(boot, dict) or boot.get("status") != "ok":
        return None
    c = rec.at(trace.by_output("bootstrap_exposure_response", bootstrap_output(label, boot)))
    pre = f"er_results.fits.{label}.bootstrap"
    eff = "or_per_sd" if fit.get("model") == "logistic" else "hr_per_sd"
    iv = boot.get(eff) or {}
    span = c.span(f"{pre}.{eff}", iv.get("lo"), iv.get("hi"), 3)
    if span == MISSING:
        return None
    return Bullet(f"{_label(c, f'er_results.fits.{label}.label', label)}: bootstrap percentile interval "
                  f"{span} from {c.frac(f'{pre}.n_ok', boot.get('n_ok'), boot.get('n_completed'))} "
                  f"usable replicates, seed {c.count(f'{pre}.seed', boot.get('seed'))}.")


def exposure_response(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    fits = {k: v for k, v in ((state.er_results or {}).get("fits") or {}).items()
            if isinstance(v, dict) and v.get("status") == "ok"}
    if not fits:
        return None
    rows, heads, boots = [], [], []
    for label in sorted(fits):
        row, head = _fit_row(label, fits[label], rec.at(fit_index(trace, fits[label])))
        rows.append(row)
        heads += [head] if head else []
        bullet = _bootstrap_bullet(label, fits[label], trace, rec)
        boots += [bullet] if bullet else []
    blocks: list[Block] = [
        Para("Exposure-response fits (logistic regression for binary endpoints, Cox proportional "
             "hazards for time to event); effects are per standard deviation of exposure with Wald "
             "confidence intervals."),
        Table(("Fit", "Model", "Exposure", "n", "Events", "Effect", "Estimate", "CI", "CI level", "p"),
              tuple(rows)),
        *boots]
    headline = "Exposure-response: " + ("; ".join(heads) if heads else "fits computed") + "."
    return Section("er", "Exposure-response", tuple(blocks), headline)


# ── dose selection ─────────────────────────────────────────────────────────────
def _dose_rows(ds: dict[str, Any], c: Cited) -> tuple[tuple[str, ...], ...]:
    rows = []
    for i, d in enumerate(ds.get("doses") or []):
        pre = f"dose_selection_results.doses.{i}"
        rows.append((c.sig(f"{pre}.dose", d.get("dose"), 6),
                     c.num(f"{pre}.p_eff", d.get("p_eff"), 3), c.num(f"{pre}.p_tox", d.get("p_tox"), 3),
                     c.num(f"{pre}.utility", d.get("utility"), 3),
                     c.span(f"{pre}.utility", d.get("utility_lo"), d.get("utility_hi"), 3),
                     c.num(f"{pre}.prob_optimal", d.get("prob_optimal"), 2),
                     {True: "yes", False: "no"}.get(d.get("feasible"), MISSING)))
    return tuple(rows)


def _selection_para(ds: dict[str, Any], c: Cited) -> tuple[Para, str]:
    s = ds.get("selected") or {}
    if ds.get("status") != "ok" or not s:
        msg = c.lit("dose_selection_results.message", ds.get("message") or "no dose met the constraints")
        return Para(f"No dose was selected: {msg}"), "Dose selection: no feasible dose."
    dose = c.sig("dose_selection_results.selected.dose", s.get("dose"), 6)
    text = (f"Selected dose {dose}: utility {c.num('dose_selection_results.selected.utility', s.get('utility'), 3)}, "
            f"probability of efficacy {c.num('dose_selection_results.selected.p_eff', s.get('p_eff'), 3)}, "
            f"probability of toxicity {c.num('dose_selection_results.selected.p_tox', s.get('p_tox'), 3)}, "
            f"probability that it is the optimal dose "
            f"{c.num('dose_selection_results.selected.prob_optimal', s.get('prob_optimal'), 2)}.")
    return Para(text), f"Dose selection: dose {dose} selected."


def dose_selection(state: PharmState, trace: AuditTrace, rec: Recorder) -> Section | None:
    ds = state.dose_selection_results
    if not isinstance(ds, dict) or ds.get("status") not in SELECTION_STATUSES:
        return None
    c = rec.at(selection_index(trace, ds))
    u = ds.get("utility") or {}
    unc = ds.get("uncertainty") or {}
    cap = c.num("dose_selection_results.toxicity_cap", ds.get("toxicity_cap"), 3)
    intro = (f"Utility {c.lit('dose_selection_results.utility.formula', u.get('formula'))} with weight w "
             f"{c.num('dose_selection_results.utility.w', u.get('w'), 3)} "
             f"({c.lit('dose_selection_results.utility.w_source', u.get('w_source'))})"
             + (f"; toxicity cap on the probability of toxicity {cap}" if cap != MISSING else "")
             + f"; uncertainty from {c.count('dose_selection_results.uncertainty.n_draws', unc.get('n_draws'))} "
               f"draws, seed {c.count('dose_selection_results.uncertainty.seed', unc.get('seed'))}.")
    selected, headline = _selection_para(ds, c)
    rows = _dose_rows(ds, c)
    blocks: list[Block] = [Para(intro), selected]
    if rows:
        blocks.append(Table(("Dose", "P efficacy", "P toxicity", "Utility", "Utility interval",
                             "P optimal", "Feasible"), rows))
    blocks += [Bullet("Warning: " + c.lit("dose_selection_results.warning", w)) for w in ds.get("warnings") or []]
    return Section("dose_selection", "Dose selection", tuple(blocks), headline)
