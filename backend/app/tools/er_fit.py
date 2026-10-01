"""Build the stored exposure-response entries (logistic, and Kaplan-Meier + Cox).

An entry is exactly what ``state.er_results["fits"][label]`` holds: the compute
result plus the re-runnable ``spec`` and the ``input_fingerprint`` of the
analysis arrays. A refused fit is returned as the bare refusal dict (it has no
``label``); callers treat any ``status != "ok"`` as "write nothing".
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.compute.er_common import sig
from app.compute.er_models import fit_logistic_er, probability_curve
from app.compute.er_survival import cox_ph, kaplan_meier, logrank_test
from app.compute.levels import level_percent
from app.tools.er_data import Analysis, spec_from_args

_CURVE_POINTS = 25
_KM_ROWS_CAP = 300
_GROUP_MIN_PER = 8                # subjects per exposure group needed to attempt KM-by-group


def _event_rows(km: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Event-time rows only, thinned evenly to ``_KM_ROWS_CAP`` (last row kept)."""
    rows = [r for r in km["table"] if r["n_event"] > 0]
    if len(rows) <= _KM_ROWS_CAP:
        return rows, False
    idx = sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, _KM_ROWS_CAP)})
    return [rows[i] for i in idx], True


def _km_summary(km: dict[str, Any]) -> dict[str, Any]:
    rows, thinned = _event_rows(km)
    return {"n": km["n"], "n_events": km["n_events"], "n_censored": km["n_censored"],
            "conf_type": km["conf_type"], "ci_level": km["ci_level"], "median": km["median"],
            "table": rows, "table_thinned": thinned}


def _group_ids(x: np.ndarray, n_groups: int) -> tuple[np.ndarray, int]:
    """Quantile bins that actually exist: right-closed like ``pd.qcut`` (a value on an
    edge goes to the lower bin), duplicate edges dropped, empty bins removed and the
    rest renumbered from 0 = lowest exposure. Returns (group id per value, n groups)."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_groups + 1)[1:-1]))
    raw = np.searchsorted(edges, x, side="left")
    present = np.unique(raw)
    return np.searchsorted(present, raw), int(present.size)


def _exposure_groups(an: Analysis, n_groups: int, ci_level: float) -> dict[str, Any] | None:
    t = np.array([np.nan if v is None else v for v in an.time], dtype=float)
    e = np.array([np.nan if v is None else v for v in an.event], dtype=float)
    x = np.array([np.nan if v is None else v for v in an.exposure], dtype=float)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(x)
    if n_groups < 2 or ok.sum() < _GROUP_MIN_PER * n_groups:
        return None
    t, e, x = t[ok], e[ok], x[ok]
    gid, k = _group_ids(x, n_groups)
    if k < 2:
        return None
    names = [f"Q{i + 1}" + (" (lowest)" if i == 0 else " (highest)" if i == k - 1 else "")
             for i in range(k)]
    groups = []
    for i in range(k):
        m = gid == i
        km = kaplan_meier(t[m].tolist(), e[m].tolist(), ci_level=ci_level)
        if km["status"] != "ok":
            continue
        groups.append({"group": names[i], "exposure_min": sig(float(x[m].min())),
                       "exposure_max": sig(float(x[m].max())), **_km_summary(km)})
    lr = logrank_test(t.tolist(), e.tolist(), [names[g] for g in gid])
    out = {"n_groups": k, "requested_groups": n_groups, "groups": groups,
           "logrank": {k_: lr.get(k_) for k_ in ("status", "chisq", "df", "p", "message") if k_ in lr}}
    if k < n_groups:
        out["note"] = (f"exposure has tied values on the quantile edges: {n_groups} groups were requested "
                       f"but only {k} distinct exposure groups exist")
    return out


def fit_entry(an: Analysis, args: dict[str, Any], label: str) -> dict[str, Any]:
    """Fit the E-R model for ``an``; the stored entry, or a refusal dict."""
    ci = float(args.get("ci_level", 0.95))
    exposure_label = an.meta["exposure_label"]
    common = {"label": label, "endpoint": an.endpoint, "spec": spec_from_args(args, label),
              "input_fingerprint": an.fingerprint, "data": an.meta}
    if an.endpoint == "binary":
        fit = fit_logistic_er(an.exposure, an.response, ci_level=ci, exposure_label=exposure_label)
        if fit["status"] != "ok":
            return fit
        return {**fit, **common, "curve": probability_curve(fit, n_points=_CURVE_POINTS)}
    ties = args.get("ties", "efron")
    cox = cox_ph(an.time, an.event, {exposure_label: an.exposure, **an.covariates}, ties=ties, ci_level=ci)
    if cox["status"] != "ok":
        return cox
    km = kaplan_meier(an.time, an.event, ci_level=ci)
    n_groups = int(args.get("exposure_groups", 4))
    entry = {**common, "status": "ok", "model": "cox", "ci_level": ci, "n": cox["n"],
             "n_events": cox["n_events"], "exposure_range": cox["exposure_range"], "cox": cox,
             "km": _km_summary(km), "warnings": list(cox["warnings"])}
    groups = _exposure_groups(an, n_groups, ci)
    if groups is not None:
        entry["km_by_exposure_group"] = groups
    return entry


def summarize(entry: dict[str, Any]) -> str:
    """One human/LLM-facing line for a successful entry."""
    label, n = entry["label"], entry["n"]
    if entry["model"] == "logistic":
        o = entry["or_per_sd"]
        return (f"Logistic E-R '{label}' on {entry['exposure_label']}: OR per SD {o['estimate']:.3g} "
                f"({level_percent(entry['ci_level'])}% CI {o['lo']:.3g}-{o['hi']:.3g}), n={n}, "
                f"{entry['n_events']} events, slope p={entry['coef']['slope']['p']:.3g}.")
    c = entry["cox"]["covariates"][0]
    med = entry["km"]["median"]["estimate"]
    return (f"Cox E-R '{label}' on {c['name']}: HR per SD {c['hr_per_sd']:.3g} "
            f"({level_percent(entry['ci_level'])}% CI {c['hr_per_sd_lo']:.3g}-{c['hr_per_sd_hi']:.3g}), "
            f"n={n}, {entry['n_events']} events, {entry['cox']['ties']} ties; "
            f"KM median {'not reached' if med is None else f'{med:.3g}'}.")


def history_record(entry: dict[str, Any]) -> dict[str, Any]:
    """Compact, bounded-size record of a fit for ``state.er_history``."""
    rec = {"kind": "fit", "label": entry["label"], "endpoint": entry["endpoint"], "model": entry["model"],
           "n": entry["n"], "n_events": entry["n_events"], "input_fingerprint": entry["input_fingerprint"]}
    if entry["model"] == "logistic":
        rec["or_per_sd"] = entry["or_per_sd"]["estimate"]
    else:
        rec["hr_per_sd"] = entry["cox"]["covariates"][0]["hr_per_sd"]
    return rec
