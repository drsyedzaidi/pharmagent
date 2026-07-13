"""Multiple-dose / steady-state dataset extraction.

Turns a NONMEM-style record set with repeated dosing (ADDL / interdose
interval) into one steady-state dosing-interval profile per subject: the rich
sampling after the LAST dose, expressed as time-after-dose over [0, tau].

Used by the steady-state NCA and compartmental tools. Pure functions; no
agent/LLM dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _num(v: Any) -> float | None:
    """Coerce a cell to float; '.', '', NA -> None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip()
    if s in {"", ".", "NA", "na", "NaN", "nan"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass(frozen=True)
class SSProfile:
    subject: Any
    tad: tuple[float, ...]      # time after last dose, ascending, within [0, tau]
    conc: tuple[float, ...]     # concentrations aligned to tad
    dose: float                 # dose amount per administration
    tau: float                  # interdose interval
    n_doses: int                # total doses (explicit + ADDL-expanded)
    c0_source: str              # "measured" | "ctau_assumed" | "none"


def _dose_times(rows: list[dict], *, time_col: str, amt_col: str,
                ii_col: str | None, addl_col: str | None) -> tuple[list[float], float | None, float | None]:
    """Expand ADDL/II into the full list of dose times; return (times, dose, tau)."""
    times: list[float] = []
    dose_amt: float | None = None
    tau: float | None = None
    for r in rows:
        amt = _num(r.get(amt_col))
        t = _num(r.get(time_col))
        if amt is None or amt <= 0 or t is None:
            continue
        dose_amt = amt  # assume constant regimen; last seen wins
        ii = _num(r.get(ii_col)) if ii_col else None
        addl = _num(r.get(addl_col)) if addl_col else None
        n_extra = int(addl) if addl is not None and addl > 0 else 0
        step = ii if (ii is not None and ii > 0) else 0.0
        if step > 0:
            tau = step
        times.append(t)
        for k in range(1, n_extra + 1):
            times.append(t + k * step)
    return sorted(times), dose_amt, tau


def dose_events(rows: list[dict], *, time_col: str, amt_col: str,
                ii_col: str | None, addl_col: str | None) -> list[dict]:
    """Full list of dose administrations [{time, amt}] (ADDL/II expanded)."""
    times, dose, _tau = _dose_times(rows, time_col=time_col, amt_col=amt_col,
                                    ii_col=ii_col, addl_col=addl_col)
    if dose is None:
        return []
    return [{"time": t, "amt": dose} for t in times]


def is_multiple_dose(records: list[dict], *, time_col: str, amt_col: str,
                     ii_col: str | None, addl_col: str | None,
                     id_col: str) -> bool:
    """True if any subject has >1 dose administration (explicit or via ADDL)."""
    by_subj: dict[Any, list[dict]] = {}
    for r in records:
        by_subj.setdefault(r.get(id_col), []).append(r)
    for rows in by_subj.values():
        times, _dose, tau = _dose_times(rows, time_col=time_col, amt_col=amt_col,
                                        ii_col=ii_col, addl_col=addl_col)
        if len(times) > 1 and tau:
            return True
    return False


def extract_ss_intervals(records: list[dict], *, id_col: str, time_col: str,
                         dv_col: str, amt_col: str, ii_col: str | None = None,
                         addl_col: str | None = None,
                         tol_frac: float = 0.02) -> dict[Any, SSProfile]:
    """Build one steady-state interval profile per subject (last dose interval).

    For each subject: expand the dose schedule, take the last dose, and collect
    measured concentrations with time-after-dose in [0, tau]. If no sample is
    taken exactly at tad=0, the trough is set to C(tau) (the steady-state
    assumption C_ss(0) = C_ss(tau)).
    """
    by_subj: dict[Any, list[dict]] = {}
    for r in records:
        by_subj.setdefault(r.get(id_col), []).append(r)

    out: dict[Any, SSProfile] = {}
    for sid, rows in by_subj.items():
        times, dose, tau = _dose_times(rows, time_col=time_col, amt_col=amt_col,
                                       ii_col=ii_col, addl_col=addl_col)
        if not times or dose is None or not tau:
            continue
        last_dose = times[-1]
        tol = tol_frac * tau

        # measured observations in the last interval, as time-after-dose
        pts: list[tuple[float, float]] = []
        for r in rows:
            t = _num(r.get(time_col))
            dv = _num(r.get(dv_col))
            if t is None or dv is None:
                continue
            tad = t - last_dose
            if -tol <= tad <= tau + tol:
                pts.append((max(tad, 0.0), dv))
        if len(pts) < 3:
            continue
        pts.sort(key=lambda x: x[0])
        # dedup identical tad (keep first)
        seen: set[float] = set()
        uniq = [(t, c) for t, c in pts if not (t in seen or seen.add(t))]
        tad = [t for t, _ in uniq]
        conc = [c for _, c in uniq]

        # establish C(0). Only an essentially-at-dose sample counts as the
        # measured pre-dose trough; a first sample partway up the absorption
        # limb (e.g. 0.25 h) is NOT the trough, so anchor C(0) = C(tau)
        # (steady-state assumption C_ss(0) = C_ss(tau)).
        PRE_DOSE_TOL = 1e-3
        c0_source = "none"
        if tad[0] <= PRE_DOSE_TOL:
            c0_source = "measured"
        else:
            ctau = next((c for t, c in zip(reversed(tad), reversed(conc))
                         if t >= tau - max(tol, 0.1 * tau)), None)
            if ctau is not None:
                tad = [0.0] + tad
                conc = [ctau] + conc
                c0_source = "ctau_assumed"

        out[sid] = SSProfile(
            subject=sid, tad=tuple(tad), conc=tuple(conc), dose=float(dose),
            tau=float(tau), n_doses=len(times), c0_source=c0_source,
        )
    return out
