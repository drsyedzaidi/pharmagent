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


_BLQ_YES = {"1", "Y", "YES", "TRUE"}
_BLQ_NO = {"0", "N", "NO", "FALSE"}
_BLANK = {"", ".", "NA", "NAN"}


def parse_blq_flag(v: Any) -> bool:
    """Below-LOQ flag cell -> True (flagged) / False (not flagged).

    Accepted: booleans; 1/0 as numbers or text ("1.0" too); Y/N, YES/NO, TRUE/FALSE in any case (CDISC ADPC flags are
    "Y" or blank); a blank or missing cell is not flagged. Anything else raises ValueError, so an unknown code never
    silently means "not BLQ"."""
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
    else:
        s = str(v).strip().upper()
        if s in _BLANK or s in _BLQ_NO:
            return False
        if s in _BLQ_YES:
            return True
        num = _num(s)
        f = float("inf") if num is None else num                # not a number: falls through to the error
    if math.isnan(f):
        return False
    if f in (0.0, 1.0):
        return f == 1.0
    raise ValueError(f"unrecognised below-LOQ flag value {v!r}: use 1/0, Y/N, YES/NO or TRUE/FALSE "
                     "(blank = not flagged)")


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


PRE_DOSE_TOL = 1e-3          # complete-interval mode: a sample this soon after the dose is still the pre-dose trough
_SAME_TIME = 1e-9            # times closer than this are one time point
_Rec = tuple[float, bool, float | None]      # (time after dose, below-LOQ flag, dv)


def _m1_in_interval(rows: list[_Rec]) -> list[tuple[float, float]]:
    """Rule M1 for one interval's (tad, below-LOQ, dv) records: flagged records before the first quantifiable one
    become a 0 baseline, later flagged records are dropped; unflagged records without a finite DV carry no measurement
    and are dropped. Quantifiable means unflagged with a finite, POSITIVE DV (+-inf is not finite): an unflagged 0 is a
    measured value and is kept as an observed zero, but it is not a quantifiable concentration, so it never ends the
    leading-BLQ phase. With nothing quantifiable, all flagged records stay as zeros."""
    rows = sorted(rows, key=lambda r: r[0])

    def finite(dv: float | None) -> bool:
        return dv is not None and math.isfinite(dv)

    first_q = next((i for i, (_, blq, dv) in enumerate(rows) if not blq and finite(dv) and dv > 0), len(rows))
    out: list[tuple[float, float]] = []
    for i, (tad, blq, dv) in enumerate(rows):
        if blq:
            if i < first_q:
                out.append((tad, 0.0))
        elif finite(dv):
            out.append((tad, dv))
    return out


def _interval_records(rows: list[dict], *, time_col: str, dv_col: str, amt_col: str, blq_col: str | None,
                      strict: bool, last_dose: float, tau: float, tol: float) -> list[_Rec]:
    """(time after the last dose, below-LOQ, dv) for the observations within [-tol, tau + tol] of the last dose."""
    recs: list[_Rec] = []
    for r in rows:
        t = _num(r.get(time_col))
        dv = _num(r.get(dv_col))
        if strict and dv is not None and not math.isfinite(dv):
            dv = None                       # +-inf is not a measurement
        is_blq = blq_col is not None and parse_blq_flag(r.get(blq_col))
        if t is None or (dv is None and not is_blq):
            continue
        if strict and (_num(r.get(amt_col)) or 0) > 0:
            continue                        # a dose event builds the schedule; it is never a concentration
        tad = t - last_dose
        if -tol <= tad <= tau + tol:
            recs.append((tad, is_blq, dv))
    return recs


def _place_endpoints(recs: list[_Rec], tau: float, tol: float) -> list[_Rec]:
    """Complete-interval mode: the sample closest to each interval end is placed AT that end. The tad-0 end takes
    samples from tol before the dose (the pre-dose trough) to PRE_DOSE_TOL after it; the tau end takes samples within
    tol of tau. Ties go to the earlier sample, so the choice never depends on row order. Every other sample keeps its
    own time if that lies inside (0, tau) and is dropped otherwise: nothing before the dose or after tau is
    integrated."""
    ends: dict[float, float] = {}
    for end, lo, hi in ((0.0, -tol, PRE_DOSE_TOL), (float(tau), tau - tol, tau + tol)):
        near = [t for t, _, _ in recs if lo <= t <= hi]
        if near:
            ends[end] = min(near, key=lambda t, e=end: (round(abs(t - e), 9), t))
    placed: list[_Rec] = []
    for t, blq, dv in recs:
        end = next((e for e, best in ends.items() if abs(t - best) <= _SAME_TIME), None)
        if end is not None:
            placed.append((end, blq, dv))
        elif 0.0 < t < tau:
            placed.append((t, blq, dv))
    return placed


def _agree(a: _Rec, b: _Rec) -> bool:
    """Two records at one time agree when both are BLQ-flagged, or both carry the same concentration (1 part in 1e6)."""
    if a[1] or b[1]:
        return a[1] == b[1]
    return a[2] is not None and b[2] is not None and math.isclose(a[2], b[2], rel_tol=1e-6, abs_tol=1e-12)


def _unique_times(recs: list[_Rec]) -> tuple[list[_Rec], str | None]:
    """One record per time point. Agreeing duplicates collapse to one; conflicting ones reject the subject, because
    keeping either would depend on the order of the rows in the file."""
    out: list[_Rec] = []
    for rec in sorted(recs, key=lambda r: (r[0], r[1], -math.inf if r[2] is None else r[2])):   # total order
        if out and abs(rec[0] - out[-1][0]) <= _SAME_TIME:
            if not _agree(out[-1], rec):
                def show(r: _Rec) -> str:
                    return "BLQ" if r[1] else f"{r[2]:g}"
                return out, (f"conflicting duplicate concentrations at time after dose {rec[0]:g} "
                             f"({show(out[-1])} vs {show(rec)}); the interval cannot be formed without choosing one")
            continue
        out.append(rec)
    return out, None


def _close_interval(pts: list[tuple[float, float]], tau: float) -> tuple[list[float], list[float], str, str] | str:
    """Complete-interval mode: both ends need their own support, a sample placed at tad 0 (measured, or a BLQ record
    normalised to a 0 baseline) and one at tau. Where exactly one end has it, steady-state periodicity is assumed for
    the other end only: C(0) = C(tau) or C(tau) = C(0). One sample never supplies both ends. The assumed closing
    segment must be integrable by linear-up/log-down: a decline to an assumed C(tau) of 0 has no log-down area, so it
    rejects the subject rather than drawing a straight line across excluded BLQ records. Returns (tad, conc,
    c0_source, ctau_source), or the reason no complete [0, tau] interval can be formed."""
    has_c0, has_ctau = pts[0][0] == 0.0, pts[-1][0] == tau
    if not (has_c0 or has_ctau):
        return ("neither C(0) nor C(tau) is available (no sample within the tolerance of either interval end), "
                "so no complete [0, tau] interval can be formed")
    if not has_c0:
        pts = [(0.0, pts[-1][1])] + pts
    if not has_ctau:
        (t_last, c_last), c0 = pts[-1], pts[0][1]
        if c0 < c_last and c0 <= 0:
            return (f"C(tau) = C(0) = {c0:g} is assumed for the missing tau sample, but the decline to it from "
                    f"{c_last:g} at time after dose {t_last:g} cannot be integrated by linear-up/log-down "
                    "(log-down needs a positive end)")
        pts = pts + [(float(tau), c0)]
    return ([t for t, _ in pts], [c for _, c in pts],
            "measured" if has_c0 else "ctau_assumed", "measured" if has_ctau else "c0_assumed")


def _legacy_anchor(pts: list[tuple[float, float]], tau: float, tol: float) -> tuple[list[float], list[float], str]:
    """Profiles as sampled (callers without require_complete_interval): keep the first record at a repeated time; if no
    sample is essentially at the dose, anchor C(0) to a sample in the last 10% of tau."""
    pts = sorted(pts, key=lambda x: x[0])
    seen: set[float] = set()
    uniq = [(t, c) for t, c in pts if not (t in seen or seen.add(t))]
    tad = [t for t, _ in uniq]
    conc = [c for _, c in uniq]
    c0_source = "none"
    if tad[0] <= PRE_DOSE_TOL:
        c0_source = "measured"
    else:
        ctau = next((c for t, c in zip(reversed(tad), reversed(conc)) if t >= tau - max(tol, 0.1 * tau)), None)
        if ctau is not None:
            tad = [0.0] + tad
            conc = [ctau] + conc
            c0_source = "ctau_assumed"
    return tad, conc, c0_source


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

    For each subject: expand the dose schedule, take the last dose, and collect measured concentrations with
    time-after-dose in [0, tau].

    With ``blq_col`` (flag values as ``parse_blq_flag`` reads them), rule M1 is applied WITHIN that last interval only:
    flagged records before the interval's first quantifiable value (unflagged, finite and positive DV; an unflagged 0
    is kept as a measured zero but does not end this phase) are a 0 baseline, later flagged records are excluded.
    Samples from earlier intervals play no part in that classification. Dose rows (AMT > 0) only build the schedule
    and are never concentrations.

    With ``require_complete_interval`` (NCA), with or without ``blq_col``: the sample closest to each interval end is
    placed exactly at it (``_place_endpoints``), conflicting duplicates reject the subject, and the interval is closed
    as ``_close_interval`` describes (each end needs its own sample; at most one end is assumed by periodicity). A
    subject that cannot form a complete interval is skipped with the reason. Without it (legacy callers such as the
    compartmental tools) profiles are returned as sampled, with C(0) anchored to a late sample when none is at the dose.

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
        tol = tol_frac * tau
        recs = _interval_records(rows, time_col=time_col, dv_col=dv_col, amt_col=amt_col, blq_col=blq_col,
                                 strict=blq_col is not None or require_complete_interval,
                                 last_dose=times[-1], tau=tau, tol=tol)
        if require_complete_interval:
            recs, conflict = _unique_times(_place_endpoints(recs, tau, tol))
            if conflict:
                skipped[sid] = conflict
                continue
        else:
            recs = [(max(t, 0.0), blq, dv) for t, blq, dv in recs]
        pts = _m1_in_interval(recs) if blq_col is not None else [(t, dv) for t, _, dv in recs]
        if len(pts) < 3:
            skipped[sid] = f"fewer than 3 usable samples in the final dosing interval ({len(pts)})"
            continue
        if require_complete_interval:
            closed = _close_interval(sorted(pts, key=lambda x: x[0]), tau)
            if isinstance(closed, str):
                skipped[sid] = closed
                continue
            tad, conc, c0_source, ctau_source = closed
        else:
            (tad, conc, c0_source), ctau_source = _legacy_anchor(pts, tau, tol), "measured"

        out[sid] = SSProfile(
            subject=sid, tad=tuple(tad), conc=tuple(conc), dose=float(dose),
            tau=float(tau), n_doses=len(times), c0_source=c0_source, ctau_source=ctau_source,
        )
    return out, skipped
