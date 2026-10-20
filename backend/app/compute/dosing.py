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

import numpy as np


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
    ctau_source: str = "measured"   # "measured" | "c0_assumed" (closed with C(tau) = C(0); require_complete_interval)


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
                ii_col: str | None, addl_col: str | None,
                rate_col: str | None = None,
                cmt_col: str | None = None) -> list[dict]:
    """Full list of dose administrations (ADDL/II expanded).

    Each event is ``{"time", "amt"}`` plus, when the dataset supplies them:

    * ``rate`` — a positive RATE makes the dose a zero-order infusion rather
      than a bolus. Note this disables the matrix-exponential fast path in
      ``pk_simulate.simulate``, so infusion datasets integrate more slowly.
    * ``cmt`` — the dosing compartment, converted from the dataset's **1-based**
      NONMEM ``CMT`` to the **0-based** index ``simulate`` uses. Without the
      conversion a ``CMT=1`` depot dose would enter compartment index 1 — the
      central compartment of an oral model — silently turning an extravascular
      dose into an IV bolus. Values below 1 are ignored rather than converted,
      since ``CMT=0`` would become ``-1`` and dose the LAST compartment.

    As in NONMEM, a dataset's CMT numbering is written for a specific model;
    an index past the model's compartment count is a data/model mismatch and
    ``simulate`` will raise rather than silently redirect the dose.

    Unlike ``_dose_times`` (which collapses a subject to one representative
    amount for steady-state work), each record keeps its OWN amount, so a
    within-subject dose change is preserved instead of being overwritten by the
    last row.
    """
    events: list[dict] = []
    for r in rows:
        amt = _num(r.get(amt_col))
        t = _num(r.get(time_col))
        if amt is None or amt <= 0 or t is None:
            continue
        ii = _num(r.get(ii_col)) if ii_col else None
        addl = _num(r.get(addl_col)) if addl_col else None
        n_extra = int(addl) if addl is not None and addl > 0 else 0
        step = ii if (ii is not None and ii > 0) else 0.0

        extra: dict[str, Any] = {}
        rate = _num(r.get(rate_col)) if rate_col else None
        if rate is not None and rate > 0:
            extra["rate"] = rate
        cmt = _num(r.get(cmt_col)) if cmt_col else None
        if cmt is not None and cmt >= 1:
            extra["cmt"] = int(cmt) - 1          # NONMEM 1-based -> 0-based

        for k in range(n_extra + 1):
            events.append({"time": t + k * step, "amt": amt, **extra})
    return sorted(events, key=lambda e: e["time"])


def time_after_dose(obs_t, doses: list[dict]) -> list[float | None]:
    """Time-after-dose (TAD) for each observation: elapsed time since the most
    recent PRIOR dose administration, not the dose amount or column semantics.

    Unlike ``extract_ss_intervals`` (last-interval-only, for steady-state
    profiles), this is defined at every observation against whichever dose
    preceded it — the diagnostic-plot convention (pmplots' TAD, NONMEM's
    ``$INPUT TAD``), useful across an entire multiple-dose record, not just
    the final interval.

    ``doses``: ``[{"time": ..., "amt": ...}, ...]`` — the same shape consumed
    by ``app.compute.pk_simulate.simulate`` (typically ``dose_events()``
    output, or a subject's own ``doses`` list). Only ``"time"`` is read here.
    A dose record's ``amt`` is trusted to already be a real administration:
    this function does not itself filter by amount, so passing raw dataset
    rows (which may repeat AMT on observation rows) rather than expanded
    dose events would inject spurious dose times and understate TAD.

    Returns one value per ``obs_t``, ``None`` for an observation at or before
    every dose (nothing yet administered) rather than a negative TAD.
    """
    t = np.asarray(obs_t, dtype=float)
    dose_times = sorted(float(d["time"]) for d in doses if d.get("time") is not None)
    if t.size == 0 or not dose_times:
        return [None] * t.size
    dt = np.asarray(dose_times, dtype=float)
    idx = np.searchsorted(dt, t, side="right") - 1
    out: list[float | None] = []
    for i, ti in zip(idx, t):
        out.append(None if i < 0 else float(ti - dt[i]))
    return out


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


def _m1_in_interval(rows: list[tuple[float, bool, float | None]]) -> list[tuple[float, float]]:
    """Rule M1 for one interval's (tad, below-LOQ, dv) records: flagged records before the first quantifiable one
    (unflagged with a finite DV; +-inf is not finite) become a 0 baseline, later flagged records are dropped;
    unflagged records without a finite DV carry no measurement and are dropped. With nothing quantifiable, all flagged
    records stay as zeros."""
    rows = sorted(rows, key=lambda r: r[0])
    def finite(dv: float | None) -> bool:
        return dv is not None and math.isfinite(dv)

    first_q = next((i for i, (_, blq, dv) in enumerate(rows) if not blq and finite(dv)), len(rows))
    out: list[tuple[float, float]] = []
    for i, (tad, blq, dv) in enumerate(rows):
        if blq:
            if i < first_q:
                out.append((tad, 0.0))
        elif finite(dv):
            out.append((tad, dv))
    return out


def extract_ss_intervals(records: list[dict], *, id_col: str, time_col: str,
                         dv_col: str, amt_col: str, ii_col: str | None = None,
                         addl_col: str | None = None, blq_col: str | None = None,
                         tol_frac: float = 0.02, require_complete_interval: bool = False) -> dict[Any, SSProfile]:
    """Per-subject last-interval profiles; subjects that cannot form one are left out (see
    ``extract_ss_intervals_checked`` for the reasons)."""
    profiles, _skipped = extract_ss_intervals_checked(
        records, id_col=id_col, time_col=time_col, dv_col=dv_col, amt_col=amt_col, ii_col=ii_col,
        addl_col=addl_col, blq_col=blq_col, tol_frac=tol_frac, require_complete_interval=require_complete_interval)
    return profiles


def extract_ss_intervals_checked(records: list[dict], *, id_col: str, time_col: str,
                                 dv_col: str, amt_col: str, ii_col: str | None = None,
                                 addl_col: str | None = None, blq_col: str | None = None,
                                 tol_frac: float = 0.02, require_complete_interval: bool = False,
                                 ) -> tuple[dict[Any, SSProfile], dict[Any, str]]:
    """Build one steady-state interval profile per subject (last dose interval).

    For each subject: expand the dose schedule, take the last dose, and collect
    measured concentrations with time-after-dose in [0, tau]. If no sample is
    taken exactly at tad=0, the trough is set to C(tau) (the steady-state
    assumption C_ss(0) = C_ss(tau)).

    With ``blq_col`` (1 = below LOQ), rule M1 is applied WITHIN that last interval only: flagged records before the
    interval's first quantifiable value (unflagged, finite time and DV) are a 0 baseline, later flagged records are
    excluded. Samples from earlier intervals play no part in that classification. Dose rows (AMT > 0) only build
    the schedule and are never concentrations. If the sample at tau is excluded, the interval is closed with the
    steady-state convention C(tau) = C(0).

    With ``require_complete_interval`` (NCA), with or without ``blq_col``: an accepted sample within the tau tolerance
    is placed at exactly tau; if C(tau) is absent but C(0) is known, the interval is closed with C(tau) = C(0)
    (``ctau_source="c0_assumed"``); a subject with neither end is skipped with that reason. Without it (legacy callers
    such as the compartmental tools) profiles are returned as sampled.

    Returns ``(profiles, skipped)``: ``skipped`` maps each subject that could not form a complete interval to why.
    """
    by_subj: dict[Any, list[dict]] = {}
    for r in records:
        by_subj.setdefault(r.get(id_col), []).append(r)

    out: dict[Any, SSProfile] = {}
    skipped: dict[Any, str] = {}
    for sid, rows in by_subj.items():
        times, dose, tau = _dose_times(rows, time_col=time_col, amt_col=amt_col,
                                       ii_col=ii_col, addl_col=addl_col)
        if not times or dose is None or not tau:
            skipped[sid] = "no repeated-dose schedule (dose amount and interdose interval)"
            continue
        last_dose = times[-1]
        tol = tol_frac * tau
        PRE_DOSE_TOL = 1e-3                     # a sample this close to the dose is the measured pre-dose trough
        strict = blq_col is not None or require_complete_interval

        # measured observations in the last interval, as time-after-dose
        pts: list[tuple[float, float]] = []
        flagged: list[tuple[float, bool, float | None]] = []   # (tad, below-LOQ, dv); used when blq_col is given
        for r in rows:
            t = _num(r.get(time_col))
            dv = _num(r.get(dv_col))
            if strict and dv is not None and not math.isfinite(dv):
                dv = None                       # +-inf is not a measurement
            is_blq = blq_col is not None and _num(r.get(blq_col)) == 1
            if t is None or (dv is None and not is_blq):
                continue
            if strict and (_num(r.get(amt_col)) or 0) > 0:
                continue                        # a dose event builds the schedule; it is never a concentration
            tad = t - last_dose
            if -tol <= tad <= tau + tol:
                near_tau = require_complete_interval and tad >= tau - tol   # within tolerance: this IS the tau sample
                near_zero = require_complete_interval and tad <= PRE_DOSE_TOL  # ... and this IS the tad-0 sample
                tad = float(tau) if near_tau else 0.0 if near_zero else max(tad, 0.0)
                if blq_col is None:
                    pts.append((tad, dv))
                else:
                    flagged.append((tad, is_blq, dv))
        if blq_col is not None:
            pts = _m1_in_interval(flagged)
        if len(pts) < 3:
            skipped[sid] = f"fewer than 3 usable samples in the final dosing interval ({len(pts)})"
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

        # close the interval: if the sample at tau was excluded (BLQ path), use the steady-state periodic
        # convention C(tau) = C(0) instead of integrating a partial interval and calling it AUC_tau
        ctau_source = "measured"
        if require_complete_interval and tad[-1] < tau:
            if tad[0] > PRE_DOSE_TOL:
                skipped[sid] = "neither C(0) nor C(tau) is available, so no complete [0, tau] interval can be formed"
                continue
            tad = tad + [float(tau)]
            conc = conc + [conc[0]]
            ctau_source = "c0_assumed"

        out[sid] = SSProfile(
            subject=sid, tad=tuple(tad), conc=tuple(conc), dose=float(dose),
            tau=float(tau), n_doses=len(times), c0_source=c0_source, ctau_source=ctau_source,
        )
    return out, skipped
