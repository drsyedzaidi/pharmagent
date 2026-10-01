"""Exposure-response and dose-selection tools (agent ``er_dose``).

* ``fit_exposure_response`` -- CHEAP. Logistic E-R for a binary endpoint, or
  Kaplan-Meier + log-rank + Cox PH for time-to-event, on explicit dataset
  columns (exposure from the dataset, or from the per-subject NCA results in
  state). Stored under ``state.er_results["fits"][label]``; a refused fit
  (separation, too few events, ...) writes NOTHING, so it can never clobber a
  good earlier fit.
* ``bootstrap_exposure_response`` -- ``expensive=True, proposable=True`` with the
  exact contract of ``run_bootstrap`` (see bootstrap_tools): the registry
  refuses it on the synchronous chat path, the agent loop turns the refusal
  into a PROPOSAL (``state.pending_tool``), and it runs only after the human
  approves, as a job via ``Orchestrator.run_tool``, with ``confirm=True``
  unconditionally required. It re-reads the dataset, proves by fingerprint that
  it is resampling the data the stored fit used, and attaches the seeded result
  to that fit. Pre-run rejections never overwrite a completed bootstrap.
* ``select_optimal_dose`` -- CHEAP. Optimus-style utility selection from a
  stored efficacy and toxicity logistic fit; the utility weight must be
  declared by the user (see ``compute.optimus``).
"""
from __future__ import annotations

from typing import Any

from app.compute.er_bootstrap import bootstrap_cox, bootstrap_logistic
from app.compute.er_common import DEFAULT_SEED, MAX_N_BOOT, ErRefusal, bootstrap_output, fit_core
from app.compute.optimus import select_dose
from app.core.audit import hash_payload
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult, require_dataset
from app.tools.er_data import ENDPOINTS, build_analysis
from app.tools.er_fit import fit_entry, history_record, summarize

AGENT = "er_dose"
HISTORY_MAX = 20
FITS_MAX = 20                   # stored fits (by label); oldest evicted
DRAWS_STORED_MAX = 500          # replicate coefficient draws kept in state for dose selection
_STORED_STATUSES = {"ok", "no_feasible_dose"}


def _with_entry(state: PharmState, entry: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Immutable update of ``er_results`` + bounded ``er_history`` for one entry."""
    prior = (state.er_results or {}).get("fits") or {}
    # recency order: a refit moves to the end; the oldest fits are evicted beyond FITS_MAX
    fits = {**{k: v for k, v in prior.items() if k != entry["label"]}, entry["label"]: entry}
    fits = dict(list(fits.items())[-FITS_MAX:])
    return {"er_results": {"fits": fits, "last_label": entry["label"]},
            "er_history": [*(state.er_history or []), record][-HISTORY_MAX:]}


def _status_result(summary: str, action: str, status: str, message: str) -> ToolResult:
    return ToolResult(summary=summary, action=action, writes={}, result={"status": status, "message": message})


def _default_label(args: dict[str, Any]) -> str:
    return str(args.get("label") or args.get("response_column") or args.get("event_column") or "fit")[:64]


# ── fit_exposure_response ────────────────────────────────────────────────────
def fit_exposure_response(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dataset_id, df = require_dataset(ctx, state, args)
    label = _default_label(args)
    an = build_analysis(df, state, args)
    entry = fit_entry(an, args, label)
    if entry["status"] != "ok":
        return _status_result(f"Exposure-response fit refused ({entry['status']}): {entry['message']}",
                              f"fit_exposure_response({entry['status']})", entry["status"], entry["message"])
    entry = {**entry, "dataset_id": dataset_id}        # the dataset this fit was computed on
    writes = _with_entry(state, entry, history_record(entry))
    return ToolResult(summary=summarize(entry), action=f"fit_exposure_response({label})",
                      writes=writes, result=entry)


# ── bootstrap_exposure_response ──────────────────────────────────────────────
def _thin(draws: list[list[float]]) -> list[list[float]]:
    if len(draws) <= DRAWS_STORED_MAX:
        return draws
    step = len(draws) / DRAWS_STORED_MAX
    return [draws[int(i * step)] for i in range(DRAWS_STORED_MAX)]


def bootstrap_exposure_response(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not args.get("confirm"):
        return _status_result(
            "Bootstrap requires confirm=true.", "bootstrap_exposure_response(confirm_required)",
            "confirm_required",
            "A bootstrap refits the exposure-response model on hundreds of resampled datasets while holding "
            "the session lock. Pass confirm=true to run it.")
    fits = (state.er_results or {}).get("fits") or {}
    label = str(args.get("label") or (state.er_results or {}).get("last_label") or "")
    entry = fits.get(label)
    if entry is None:
        return _status_result(
            "Bootstrap skipped: no stored exposure-response fit.", "bootstrap_exposure_response(no_fit)",
            "no_fit", f"Run fit_exposure_response first (label {label!r}; stored labels: "
                      f"{', '.join(fits) or 'none'}).")
    try:
        _, df = require_dataset(ctx, state, args)
    except ValueError as exc:
        return _status_result("Bootstrap skipped: no dataset.", "bootstrap_exposure_response(no_dataset)",
                              "no_dataset", str(exc))
    an = build_analysis(df, state, {**entry["spec"], "stratify_by": args.get("stratify_by")})
    if an.fingerprint != entry["input_fingerprint"]:
        return _status_result(
            "Bootstrap skipped: the data changed since the fit.", "bootstrap_exposure_response(data_changed)",
            "data_changed", "The analysis data no longer match the stored fit (fingerprint differs); refit "
                            "with fit_exposure_response before bootstrapping.")
    n_boot = max(1, min(int(args.get("n_boot", 500)), MAX_N_BOOT))
    seed = int(args.get("seed", DEFAULT_SEED))
    if entry["model"] == "logistic":
        out = bootstrap_logistic(an.exposure, an.response, n_boot=n_boot, seed=seed,
                                 ci_level=entry["ci_level"], strata=an.strata)
        if out["status"] == "ok":
            out = {**out, "n_draws_available": len(out["draws"]), "draws": _thin(out["draws"])}
    else:
        covs = {entry["data"]["exposure_label"]: an.exposure, **an.covariates}
        out = bootstrap_cox(an.time, an.event, covs, ties=entry["cox"]["ties"], n_boot=n_boot, seed=seed,
                            ci_level=entry["ci_level"], strata=an.strata)
    if out["status"] != "ok":
        return _status_result(f"Bootstrap incomplete: {out.get('message', out['status'])}",
                              f"bootstrap_exposure_response({out['status']})", out["status"],
                              out.get("message", out["status"]))
    updated = {**entry, "bootstrap": out}
    rec = {"kind": "bootstrap", "label": label, "seed": seed, "n_ok": out["n_ok"], "n_failed": out["n_failed"]}
    key = "or_per_sd" if entry["model"] == "logistic" else "hr_per_sd"
    iv = out[key]
    return ToolResult(
        summary=(f"Bootstrap of '{label}': {out['n_ok']}/{out['n_completed']} replicates usable, seed {seed}; "
                 f"{key.replace('_', ' ')} {iv['estimate']:.3g} (percentile CI {iv['lo']:.3g}-{iv['hi']:.3g})."),
        action=f"bootstrap_exposure_response({label})", writes=_with_entry(state, updated, rec),
        result=bootstrap_output(label, out))


# ── select_optimal_dose ──────────────────────────────────────────────────────
def _dp_mapping(state: PharmState, fit: dict[str, Any], param: str | None) -> dict[str, Any]:
    dp = state.dose_prop_results or {}
    key = param or fit.get("exposure_label")
    rec = ((dp.get("parameters") or {}).get(key) or {}) if dp.get("status") == "ok" else {}
    if rec.get("intercept") is None or rec.get("slope") is None:
        have = sorted(dp.get("parameters") or {}) if dp.get("status") == "ok" else []
        raise ErRefusal("invalid_input",
                        f"no usable dose-proportionality fit for {key!r} (available: {', '.join(have) or 'none'}; "
                        "run run_dose_proportionality, or pass dp_parameter or an explicit exposure_mapping)")
    return {"type": "dose_proportionality", "parameter": key, "intercept": rec["intercept"],
            "slope": rec["slope"], "dose_levels": dp.get("dose_levels") or []}


def _mapping(state: PharmState, args: dict[str, Any], eff: dict[str, Any], tox: dict[str, Any]) -> dict[str, Any]:
    if args.get("exposure_mapping"):
        return args["exposure_mapping"]
    if args.get("exposure_from") != "dose_proportionality":
        raise ValueError("exposure_mapping (linear / power / by_dose) or exposure_from='dose_proportionality' "
                         "is required")
    param = args.get("dp_parameter")
    me, mt = _dp_mapping(state, eff, param), _dp_mapping(state, tox, param)
    return me if me["parameter"] == mt["parameter"] else {"efficacy": me, "toxicity": mt}


def _input_fit(fit: dict[str, Any]) -> dict[str, Any]:
    """Provenance of a fit a selection used: the hash of its audited output (so the memo
    can find the run that produced it) and the dataset it was computed on."""
    return {"label": fit.get("label"), "fit_sha256": hash_payload(fit_core(fit)),
            "dataset_id": fit.get("dataset_id")}


def select_optimal_dose(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    fits = (state.er_results or {}).get("fits") or {}
    labels = (str(args.get("efficacy_label") or "efficacy"), str(args.get("toxicity_label") or "toxicity"))
    missing = [lab for lab in labels if lab not in fits]
    if missing:
        return _status_result(
            "Dose selection skipped: missing E-R fit.", "select_optimal_dose(no_fit)", "no_fit",
            f"No stored fit for {', '.join(map(repr, missing))} (stored: {', '.join(fits) or 'none'}). Fit the "
            "efficacy and toxicity endpoints with fit_exposure_response, labelled to match "
            "efficacy_label / toxicity_label.")
    eff, tox = fits[labels[0]], fits[labels[1]]
    mapping: dict[str, Any] = {}
    if eff.get("model") == tox.get("model") == "logistic":     # else select_dose refuses with a clear status
        try:
            mapping = _mapping(state, args, eff, tox)
        except ErRefusal as exc:
            return _status_result(f"Dose selection skipped: {exc.message}", "select_optimal_dose(invalid_input)",
                                  exc.status, exc.message)
    out = select_dose(
        doses=args.get("doses") or [], exposure_mapping=mapping, efficacy_fit=eff, toxicity_fit=tox,
        utility_weight=args.get("utility_weight"), utility=args.get("utility"),
        toxicity_cap=args.get("toxicity_cap"), n_draws=int(args.get("n_draws", 2000)),
        seed=int(args.get("seed", DEFAULT_SEED)), ci_level=float(args.get("ci_level", 0.95)))
    out = {**out, "input_fits": {role: _input_fit(fit) for role, fit in (("efficacy", eff), ("toxicity", tox))}}
    if out["status"] not in _STORED_STATUSES:
        return _status_result(f"Dose selection refused ({out['status']}): {out['message']}",
                              f"select_optimal_dose({out['status']})", out["status"], out["message"])
    w = out["utility"]
    note = " [default w acknowledged]" if w["w_source"] == "default_acknowledged" else ""
    if out["status"] == "no_feasible_dose":
        summary = f"Dose selection: {out['message']}"
    else:
        s = out["selected"]
        summary = (f"Dose selection (U = P_eff - w*P_tox, w={w['w']:g}{note}): selected dose {s['dose']:g} "
                   f"(U={s['utility']:.3f}, P_eff={s['p_eff']:.3f}, P_tox={s['p_tox']:.3f}); probability it is "
                   f"optimal {100 * s['prob_optimal']:.0f}% over {out['uncertainty']['n_draws']} draws "
                   f"(seed {out['uncertainty']['seed']})." + (f" {len(out['warnings'])} warning(s)."
                                                               if out["warnings"] else ""))
    rec = {"kind": "dose_selection", "status": out["status"], "w": w["w"], "w_source": w["w_source"],
           "toxicity_cap": out["toxicity_cap"], "seed": out["uncertainty"]["seed"],
           "selected_dose": (out.get("selected") or {}).get("dose")}
    return ToolResult(summary=summary, action=f"select_optimal_dose({labels[0]}/{labels[1]})",
                      writes={"dose_selection_results": out,
                              "er_history": [*(state.er_history or []), rec][-HISTORY_MAX:]},
                      result=out)


# ── registry ─────────────────────────────────────────────────────────────────
_COLUMNS = {
    "endpoint": {"type": "string", "enum": list(ENDPOINTS)},
    "exposure_column": {"type": "string", "description": "per-subject exposure column (AUC, Cavg, Cmax ...)"},
    "exposure_source": {"type": "string", "enum": ["dataset", "nca"]},
    "exposure_metric": {"type": "string", "description": "NCA parameter when exposure comes from the NCA "
                                                         "results (AUC_inf, AUC_last, AUC_tau, Cmax, Cavg)"},
    "exposure_label": {"type": "string"},
    "response_column": {"type": "string", "description": "binary 0/1 response (endpoint=binary)"},
    "time_column": {"type": "string"}, "event_column": {"type": "string", "description": "1 = event, 0 = censored"},
    "covariate_columns": {"type": "array", "items": {"type": "string"},
                          "description": "extra Cox covariates (time_to_event only)"},
    "subject_column": {"type": "string"}, "label": {"type": "string"},
}

TOOLS = [
    Tool("fit_exposure_response",
         "Exposure-response analysis on per-subject data. endpoint=binary: logistic regression of a 0/1 "
         "response (efficacy or adverse event) on an exposure metric -- odds ratio per unit and per SD, "
         "Wald CIs, predicted-probability curve with a CI band. endpoint=time_to_event: Kaplan-Meier "
         "(Greenwood) with medians, log-rank across exposure quartiles, and Cox PH hazard ratio per unit "
         "and per SD (Efron ties). Validated against R glm / survival. Refuses (never guesses) on perfect "
         "separation, too few events, or zero-variance exposure. Use label='efficacy' / 'toxicity' so "
         "select_optimal_dose finds the fits.",
         AGENT,
         {"type": "object", "properties": {**_COLUMNS, "ties": {"type": "string", "enum": ["efron", "breslow"]},
                                           "ci_level": {"type": "number"},
                                           "exposure_groups": {"type": "integer", "minimum": 0, "maximum": 5}},
          "required": ["endpoint"]},
         fit_exposure_response),
    Tool("bootstrap_exposure_response",
         "Seeded subject-level bootstrap of a stored exposure-response fit (percentile intervals for the "
         "odds / hazard ratio and the probability band; replicate draws feed dose selection). The seed is "
         "an input and is echoed. Refits hundreds of resamples, so `confirm=true` is required and the "
         "chat path can only PROPOSE it for human approval. Optionally stratify by a per-subject column "
         "(e.g. DOSE) so every dose group keeps its size.",
         AGENT,
         {"type": "object",
          "properties": {"confirm": {"type": "boolean"}, "label": {"type": "string"},
                         "n_boot": {"type": "integer", "minimum": 1, "maximum": MAX_N_BOOT},
                         "seed": {"type": "integer"}, "stratify_by": {"type": "string"}},
          "required": ["confirm"]},
         bootstrap_exposure_response, expensive=True, proposable=True),
    Tool("select_optimal_dose",
         "Project Optimus-style dose selection: from stored efficacy and toxicity logistic E-R fits, an "
         "exposure-per-dose mapping (linear/power scaling, per-dose values, or the dose-proportionality "
         "results in state) and a user-DECLARED utility U(d) = P_eff(d) - w*P_tox(d), report P_eff, P_tox "
         "and utility per dose, the selected dose, the probability each dose is optimal and a credible "
         "band on utility (E-R bootstrap draws or seeded asymptotic draws). `utility_weight` (w) is "
         "REQUIRED -- or acknowledge the default explicitly with utility='default_w=1'. Optional "
         "toxicity_cap on P_tox. Warns when the selected dose sits at the edge of the grid.",
         AGENT,
         {"type": "object",
          "properties": {"doses": {"type": "array", "items": {"type": "number"}},
                         "exposure_mapping": {"type": "object"},
                         "exposure_from": {"type": "string", "enum": ["dose_proportionality"]},
                         "dp_parameter": {"type": "string"},
                         "efficacy_label": {"type": "string"}, "toxicity_label": {"type": "string"},
                         "utility_weight": {"type": "number", "minimum": 0},
                         "utility": {"type": "string", "enum": ["default_w=1"]},
                         "toxicity_cap": {"type": "number"}, "n_draws": {"type": "integer"},
                         "seed": {"type": "integer"}, "ci_level": {"type": "number"}},
          "required": ["doses"]},
         select_optimal_dose),
]
