"""Quality-control rules and the subject-level disposition table for average bioequivalence.

Pure functions over the contract's indexed records; no I/O. Each finding names its rule, subject, treatment, period,
the action taken (``excluded`` or ``flagged``), the numbers behind it and the guidance it comes from.

Rules (guidance texts read 2026-10-23 and 2026-10-26):
- ``predose_gt_5pct_cmax``: a pre-dose concentration above 5 % of the subject's Cmax in that period drops the subject of
  the affected period. FDA, Bioequivalence Studies With Pharmacokinetic Endpoints for Drugs Submitted Under an ANDA
  (May 2026), "Subjects with pre-dose concentrations"; ICH M13A (EMA/CHMP/ICH/953493/2022) sec. 2.2.3.3; EMA
  CPMP/EWP/QWP/1401/98 Rev. 1/Corr**, "Carry-over effects" (which does not apply to endogenous drugs, hence the
  justified ``flag_only`` policy). A pre-dose of exactly 5 % keeps the data; values within 1e-12 relative of the
  boundary are treated as equal because the boundary cannot be represented exactly in floating point.
- ``low_exposure_lt_5pct_gmean``: a period whose AUC is under 5 % of the geometric mean AUC of that product (computed
  without the subject) is flagged, never excluded: the guidance accepts such an exclusion only in exceptional cases
  (M13A sec. 2.2.1.1).
- ``auc_coverage_lt_80pct`` and ``auc_coverage_lt_80pct_in_gt_20pct_of_records``: AUC(0-t) should cover at least 80 % of
  AUC(0-inf); when it does not in more than 20 % of observations the validity of the study may need to be discussed
  (M13A sec. 2.2.2.2; EMA sec. 4.1.4). Subjects are not excluded for it.
- ``cmax_at_first_sample``: Cmax at the first post-dose sample is to be discussed (M13A sec. 2.1.8.1; FDA "First Point
  Cmax").
- ``no_predose_sample`` and ``invalid_predose_value``: a missing or unusable pre-dose sample cannot be checked against
  the 5 % rule; it is reported, not guessed.
"""
from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

PREDOSE_FRACTION = Decimal("0.05")
LOW_EXPOSURE_FRACTION = 0.05
COVERAGE_FLOOR_PCT = 80.0
COVERAGE_SHARE = 0.20
ABSENT = object()                       # the record carries no ``predose`` key at all: nothing to check

BASIS = {
    "predose_gt_5pct_cmax": (
        "FDA ANDA PK BE guidance (May 2026), 'Subjects with pre-dose concentrations': a pre-dose value above 5 percent "
        "of Cmax means the subject of the affected period is dropped; ICH M13A sec. 2.2.3.3; EMA CPMP/EWP/QWP/1401/98 "
        "Rev. 1/Corr** 'Carry-over effects' (not for endogenous drugs)"),
    "multiple_predose_samples": (
        "More than one sample at or before time 0: the 5 percent rule uses the highest, because any pre-dose "
        "concentration above 5 percent of Cmax signals carry-over (FDA ANDA PK BE guidance; ICH M13A sec. 2.2.3.3)"),
    "low_exposure_check_skipped": (
        "Fewer than 3 records of the product remain after the records the analysis discards are left out, so the "
        "leave-one-out geometric mean of the very-low-exposure check (ICH M13A sec. 2.2.1.1) cannot be formed"),
    "no_predose_sample": "A pre-dose sample is expected (FDA ANDA PK BE guidance; ICH M13A sampling schedule)",
    "invalid_predose_value": "A pre-dose value must be a finite non-negative number to be checked against the 5 % rule",
    "low_exposure_lt_5pct_gmean": (
        "ICH M13A sec. 2.2.1.1: very low exposure (AUC under 5 % of the product's geometric mean AUC without the "
        "subject) is an exception accepted only in exceptional cases; reported, never applied automatically"),
    "auc_coverage_lt_80pct": (
        "ICH M13A sec. 2.2.2.2 / EMA sec. 4.1.4: AUC(0-t) should cover at least 80 % of AUC(0-inf)"),
    "auc_coverage_lt_80pct_in_gt_20pct_of_records": (
        "ICH M13A sec. 2.2.2.2 / EMA sec. 4.1.4: coverage under 80 % in more than 20 % of observations means the "
        "validity of the study may need to be discussed; subjects are not excluded for it"),
    "cmax_at_first_sample": (
        "ICH M13A sec. 2.1.8.1 and FDA 'First Point Cmax': the possible missed Cmax is to be discussed"),
}


def _num(x: Any) -> float | None:
    """x as a finite float when it is a real, non-boolean number; otherwise None."""
    if isinstance(x, (bool,)) or not isinstance(x, numbers.Real):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _finding(rule: str, action: str, *, scope: str = "record", subject: Any = None, treatment: Any = None,
             period: Any = None, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"scope": scope, "rule": rule, "subject": subject, "treatment": treatment, "period": period,
            "action": action, "detail": detail or {}, "basis": BASIS[rule]}


_ASCII_DIGITS = re.compile(r"[0-9]+")


def natural_key(x: Any) -> list[tuple[int, int, str]]:
    """Total sort key that orders S2 before S10: runs of ASCII digits compare by value (by length, then text, so no
    integer conversion and no digit-count limit), everything else (non-ASCII digits and superscripts included)
    compares as text, and the full text is the last element so two distinct ids never tie."""
    text = str(x)
    key: list[tuple[int, int, str]] = []
    pos = 0
    for m in _ASCII_DIGITS.finditer(text):
        if m.start() > pos:
            key.append((1, 0, text[pos:m.start()]))
        digits = m.group().lstrip("0")
        key.append((0, len(digits), digits))
        pos = m.end()
    if pos < len(text):
        key.append((1, 0, text[pos:]))
    key.append((2, 0, text))
    return key


def _above_predose_limit(predose: float, cmax: float) -> bool:
    limit = float(PREDOSE_FRACTION) * cmax
    if math.isclose(predose, limit, rel_tol=1e-12, abs_tol=0.0):
        return False
    return predose > limit


def run_qc(by: Mapping[Any, Mapping[Any, Mapping[str, Any]]], extras: Mapping[tuple, Mapping[str, Any]],
           periods: Mapping[tuple, Any], test: Any, reference: Any, *, policy: str,
           prespecified: frozenset | set = frozenset()
           ) -> tuple[list[dict[str, Any]], dict[tuple, dict[str, Any]]]:
    """(findings, excluded) where ``excluded`` maps (treatment, subject) to the finding that excludes that record.

    ``prespecified`` are subjects the analysis plan already excludes: a pre-dose failure on their records is still
    reported but does not count as a QC exclusion (the plan came first), and records the analysis discards, by QC or
    by the plan, do not feed the study-level flag rules, which describe the population that is analysed."""
    findings: list[dict[str, Any]] = []
    excluded: dict[tuple, dict[str, Any]] = {}
    action_predose = "excluded" if policy == "profile_rule" else "flagged"

    for lv in (test, reference):
        for s in sorted(by[lv], key=str):
            ex = extras.get((lv, s), {})
            period = periods.get((lv, s))
            pre = ex.get("predose", ABSENT)
            if pre is None:
                findings.append(_finding("no_predose_sample", "flagged", subject=s, treatment=lv, period=period))
            elif pre is not ABSENT:
                value = _num(pre)
                if value is None or value < 0:
                    findings.append(_finding("invalid_predose_value", "flagged", subject=s, treatment=lv,
                                             period=period, detail={"value": repr(pre)}))
                else:
                    cmax = _num(by[lv][s].get("Cmax"))
                    if cmax is not None and cmax > 0 and _above_predose_limit(value, cmax):
                        superseded = s in prespecified
                        f = _finding("predose_gt_5pct_cmax", "flagged" if superseded else action_predose, subject=s,
                                     treatment=lv, period=period,
                                     detail={"predose": value, "cmax": cmax, "ratio": value / cmax,
                                             "limit_fraction": float(PREDOSE_FRACTION),
                                             **({"predose_time": ex["predose_time"]} if "predose_time" in ex else {}),
                                             **({"superseded_by": "prespecified_exclusion",
                                                 "would_exclude": action_predose == "excluded"}
                                                if superseded else {})})
                        findings.append(f)
                        if action_predose == "excluded" and not superseded:
                            excluded[(lv, s)] = f
            if (ex.get("predose_samples") or 0) > 1:
                multi = {"n": ex["predose_samples"], "used": "highest"}
                if "predose_time" in ex:
                    multi["time_used"] = ex["predose_time"]
                findings.append(_finding("multiple_predose_samples", "flagged", subject=s, treatment=lv,
                                         period=period, detail=multi))
            if ex.get("cmax_first_point") is True:
                findings.append(_finding("cmax_at_first_sample", "flagged", subject=s, treatment=lv, period=period))

    # the analysis drops a QC-excluded subject from both arms of a crossover (no evaluable counterpart), and a
    # prespecified exclusion applies to the subject everywhere: both leave the study-level rules together
    dropped_subjects = set(prespecified) | {s for (_, s) in excluded}
    discarded = {(lv, s) for lv in (test, reference) for s in dropped_subjects}
    for lv in (test, reference):                       # very low exposure, per product, leave-one-out
        aucs = {s: v for s, p in by[lv].items()
                if (lv, s) not in discarded and (v := _num(p.get("AUC_last"))) is not None and v > 0}
        n_discarded = sum(1 for s, p in by[lv].items() if (lv, s) in discarded and _num(p.get("AUC_last")))
        if len(aucs) < 3 and n_discarded:
            findings.append(_finding("low_exposure_check_skipped", "flagged", scope="study", treatment=lv,
                                     detail={"product": lv, "n_remaining": len(aucs), "n_discarded": n_discarded}))
        if len(aucs) >= 3:
            logs = {s: math.log(v) for s, v in aucs.items()}
            total = sum(logs.values())
            for s in sorted(aucs, key=str):
                gm = math.exp((total - logs[s]) / (len(aucs) - 1))
                ratio = aucs[s] / gm
                if ratio < LOW_EXPOSURE_FRACTION:
                    findings.append(_finding("low_exposure_lt_5pct_gmean", "flagged", subject=s, treatment=lv,
                                             period=periods.get((lv, s)),
                                             detail={"auc_last": aucs[s], "geo_mean_others": gm, "ratio": ratio}))

    n_records = n_below = 0                            # AUC(0-t) coverage of AUC(0-inf)
    for lv in (test, reference):
        for s in sorted(by[lv], key=str):
            if (lv, s) in discarded:
                continue
            a, b = _num(by[lv][s].get("AUC_last")), _num(by[lv][s].get("AUC_inf"))
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            n_records += 1
            coverage = 100.0 * a / b
            if coverage < COVERAGE_FLOOR_PCT:
                n_below += 1
                findings.append(_finding("auc_coverage_lt_80pct", "flagged", subject=s, treatment=lv,
                                         period=periods.get((lv, s)), detail={"coverage_pct": coverage}))
    if n_records and n_below / n_records > COVERAGE_SHARE:
        findings.append(_finding("auc_coverage_lt_80pct_in_gt_20pct_of_records", "flagged", scope="study",
                                 detail={"n_records": n_records, "n_below": n_below, "fraction": n_below / n_records}))
    return findings, excluded


def summarise(findings: list[dict[str, Any]]) -> dict[str, Any]:
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    return {"excluded": sum(f["action"] == "excluded" for f in findings),
            "flagged": sum(f["action"] == "flagged" for f in findings), "by_rule": by_rule}


def build_dispositions(by: Mapping[Any, Mapping[Any, Mapping[str, Any]]], periods: Mapping[tuple, Any],
                       design: str, test: Any, reference: Any, layout: Mapping[Any, bool] | None,
                       analysed: Mapping[str, set], ledger: list[dict[str, Any]], findings: list[dict[str, Any]]
                       ) -> list[dict[str, Any]]:
    """One row per subject: sequence or arm, periods, per-parameter status with the reasons, QC flags, overall state."""
    reasons: dict[tuple, list[str]] = {}
    for e in ledger:
        bucket = reasons.setdefault((e["subject"], e["parameter"]), [])
        if e["reason"] not in bucket:
            bucket.append(e["reason"])
    flags: dict[Any, list[str]] = {}
    for f in findings:
        if f["scope"] == "record" and f["subject"] is not None and f["rule"] not in flags.get(f["subject"], []):
            flags.setdefault(f["subject"], []).append(f["rule"])

    names = sorted(analysed)
    subjects = sorted(set(by[test]) | set(by[reference]), key=natural_key)
    rows = []
    for s in subjects:
        params = {}
        for n in names:
            if s in analysed[n]:
                params[n] = {"status": "analysed", "reasons": []}
            else:
                params[n] = {"status": "excluded", "reasons": reasons.get((s, n), []) + [
                    r for r in reasons.get((s, "*"), []) if r not in reasons.get((s, n), [])]}
        n_ok = sum(p["status"] == "analysed" for p in params.values())
        row: dict[str, Any] = {
            "subject": s, "parameters": params, "qc_flags": flags.get(s, []),
            "disposition": "analysed" if n_ok == len(params) else "excluded" if n_ok == 0 else "partial"}
        if design == "crossover":
            row["sequence"] = ("TR" if layout[s] else "RT") if layout and s in layout else None
            row["arm"] = None
            row["periods"] = {lv: periods.get((lv, s)) for lv in (test, reference) if s in by[lv]}
        else:
            row["arm"] = test if s in by[test] else reference
            row["sequence"] = None
            row["periods"] = {}
        rows.append(row)
    return rows
