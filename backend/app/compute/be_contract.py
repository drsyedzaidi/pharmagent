"""Average-bioequivalence safety contract.

Wraps the log-scale estimator in ``app.compute.bioequivalence`` with the checks a regulatory analysis needs before a
pass/fail verdict can be trusted:

- the design, the test and reference levels and the regulatory profile are explicit arguments; nothing is inferred
  and a crossover is never re-analysed as parallel because a subject is incomplete;
- every value left out of an estimate is recorded in an exclusion ledger with its subject, treatment, parameter,
  reason, original value and whether the exclusion was pre-specified;
- a parameter cannot pass with fewer evaluable subjects than the profile requires (crossover: subjects with both
  products; parallel: per arm);
- the confidence bounds are rounded with decimal arithmetic where the profile's source states a rounding rule, and
  the verdict uses the rounded bounds;
- limits other than the profile's need a written justification; replicate and multi-group data are refused.

Profiles and their sources (texts downloaded and read 2026-10-23):
- ``fda_2026``: FDA, Statistical Approaches to Establishing Bioequivalence, final guidance, May 2026 (sec. II.A,
  footnote 6, rounding; sec. II.B.2, missing data).
- ``ich_m13a``: ICH M13A, EMA/CHMP/ICH/953493/2022, Step 5, 25 July 2024 (sec. 2.2.3.1, at least 12 evaluable
  subjects; sec. 2.2.4, 80.00-125.00%). The text states no rounding rule, so bounds are compared unrounded.
- ``ema_2010``: EMA, CPMP/EWP/QWP/1401/98 Rev. 1/Corr** (sec. 4.1.8, rounding to two decimals; at least 12
  evaluable subjects).

Statistical model: this slice keeps the existing estimator (paired log differences for a crossover, pooled two-sample
log means for a parallel design). It has no period or sequence terms; the profile-specific models are a separate
change, and every result says the model is interim.
"""
from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from numpy import bool_ as np_bool

from app.compute.bioequivalence import MODEL_2X2, be_one_parameter

DESIGNS = ("crossover", "parallel")
INTERIM_MODEL = ("interim: paired log differences (crossover) or pooled two-sample log means (parallel); no period "
                 "or sequence terms. The profile-specific model is pending.")
_TWO_DP = Decimal("0.01")
_log = logging.getLogger(__name__)


class BEContractError(ValueError):
    """The request cannot be analysed under the contract; the message says why."""


@dataclass(frozen=True)
class Profile:
    name: str
    lower: Decimal
    upper: Decimal
    ci_percent: int
    alpha: float
    min_evaluable: int
    source: str
    rounding_source: str | None


PROFILES: dict[str, Profile] = {
    "fda_2026": Profile(
        "fda_2026", Decimal("80.00"), Decimal("125.00"), 90, 0.10, 12,
        "FDA, Statistical Approaches to Establishing Bioequivalence (May 2026)",
        "FDA May 2026, sec. II.A footnote 6: rounded CI at least 80.00% and not more than 125.00%"),
    "ich_m13a": Profile(
        "ich_m13a", Decimal("80.00"), Decimal("125.00"), 90, 0.10, 12,
        "ICH M13A, EMA/CHMP/ICH/953493/2022 (Step 5, 25 July 2024)",
        None),
    "ema_2010": Profile(
        "ema_2010", Decimal("80.00"), Decimal("125.00"), 90, 0.10, 12,
        "EMA, CPMP/EWP/QWP/1401/98 Rev. 1/Corr**",
        "EMA 2010, sec. 4.1.8: bounds compared when rounded to two decimal places"),
}


def regulatory_round(value: float) -> Decimal:
    """Round a computed bound to two decimals: ``Decimal(str(value))`` quantized with ROUND_HALF_UP.

    Binary rounding misreads boundary values: ``round(1.005, 2)`` is 1.0 because the float is 1.00499...; here the
    computed value's decimal string 1.005 rounds to 1.01. Half up for exact .xx5 ties is a PharmAgent implementation
    convention: neither the FDA nor the EMA text defines tie handling."""
    return Decimal(str(float(value))).quantize(_TWO_DP, rounding=ROUND_HALF_UP)


def _estimate(test_vals: list[float], ref_vals: list[float], *, paired: bool, alpha: float,
              design_cols: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    r = be_one_parameter(test_vals, ref_vals, paired=paired, alpha=alpha, **(design_cols or {}))
    return {k: r.get(k) for k in ("gmr_pct", "ci_lower_pct", "ci_upper_pct", "cv_intra_pct",
                                  "ci_lower_raw", "ci_upper_raw", "df")}


def _real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np_bool))


def _limit(x) -> Decimal:
    if not _real(x) or not math.isfinite(float(x)):
        raise BEContractError(f"limits must be finite numbers; got {x!r}")
    d = Decimal(repr(float(x)))
    if d != d.quantize(_TWO_DP):
        raise BEContractError(f"limits are stated to two decimals; {x!r} has more")
    return d.quantize(_TWO_DP)


def _validate(records, design, profile, test, reference, limits, limits_justification):
    missing = [k for k, v in (("design", design), ("profile", profile), ("test", test),
                              ("reference", reference)) if v in (None, "")]
    if missing:
        raise BEContractError(f"missing required argument(s): {', '.join(missing)}")
    if design not in DESIGNS:
        raise BEContractError(f"design must be one of {DESIGNS}; got {design!r}")
    if profile not in PROFILES:
        raise BEContractError(f"profile must be one of {sorted(PROFILES)}; got {profile!r}")
    if test == reference:
        raise BEContractError("test and reference must be different treatment levels")
    levels = {r["treatment"] for r in records}
    for name, lv in (("test", test), ("reference", reference)):
        if lv not in levels:
            raise BEContractError(f"{name} level {lv!r} is not in the data (levels: {sorted(map(str, levels))})")
    prof = PROFILES[profile]
    lo, hi = prof.lower, prof.upper
    if limits is not None:
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise BEContractError(f"limits must be [lower, upper]; got {limits!r}")
        lo, hi = (_limit(x) for x in limits)
        if (lo, hi) != (prof.lower, prof.upper) and not (limits_justification or "").strip():
            raise BEContractError("custom limits need a written justification (limits_justification)")
        if not lo < 100 < hi:
            raise BEContractError(f"limits must bracket 100%; got {lo}-{hi}")
    return prof, lo, hi


def _index(records, design, test, reference):
    """{treatment: {subject: params}} for the two compared levels; refuses replicate and cross-arm subjects."""
    by: dict[Any, dict[Any, dict]] = {test: {}, reference: {}}
    for r in records:
        if r["treatment"] not in by:
            continue
        arm = by[r["treatment"]]
        if r["subject"] in arm:
            raise BEContractError(f"subject {r['subject']!r} has more than one {r['treatment']!r} record: replicate "
                                  "designs are not supported")
        arm[r["subject"]] = dict(r["params"])
    if design == "parallel":
        both = set(by[test]) & set(by[reference])
        if both:
            raise BEContractError(f"parallel design but subject(s) {sorted(map(str, both))} received both products")
    return by


def _entry(subject, treatment, parameter, reason, value, prespecified):
    return {"subject": subject, "treatment": treatment, "parameter": parameter, "reason": reason,
            "original_value": value, "prespecified": prespecified}


def _value_problem(v) -> str | None:
    if v is None:
        return "missing_value"
    if not _real(v):
        return "invalid_type"
    if math.isnan(float(v)):
        return "missing_value"
    if math.isinf(float(v)):
        return "non_finite_value"
    return None if v > 0 else "non_positive_value"


def _verdict(est, prof, lo, hi, enough):
    lower_raw = est.get("ci_lower_raw", est["ci_lower_pct"])
    upper_raw = est.get("ci_upper_raw", est["ci_upper_pct"])
    if lower_raw is None or upper_raw is None:
        return {"within_limits": False, "acceptance": "not_estimable",
                "ci_lower_rounded": None, "ci_upper_rounded": None}
    if prof.rounding_source:
        cl, cu = regulatory_round(lower_raw), regulatory_round(upper_raw)
        inside = cl >= lo and cu <= hi
        rounded = (str(cl), str(cu))
    else:
        inside = Decimal(repr(float(lower_raw))) >= lo and Decimal(repr(float(upper_raw))) <= hi
        rounded = (None, None)
    acceptance = ("insufficient_evaluable_subjects" if not enough
                  else "within_limits" if inside else "outside_limits")
    return {"within_limits": bool(inside and enough), "acceptance": acceptance,
            "ci_lower_rounded": rounded[0], "ci_upper_rounded": rounded[1]}


def _one_parameter(name, by, design, test, reference, excluded_subjects, prof, lo, hi, ledger, layout=None):
    t_arm, r_arm = by[test], by[reference]
    usable = {lv: {} for lv in (test, reference)}
    for lv, arm in ((test, t_arm), (reference, r_arm)):
        for s, params in arm.items():
            v = params.get(name)
            if s in excluded_subjects:
                ledger.append(_entry(s, lv, name, excluded_subjects[s], v, True))
                continue
            problem = _value_problem(v)
            if problem:
                ledger.append(_entry(s, lv, name, problem, v, False))
                continue
            usable[lv][s] = float(v)
    if design == "crossover":
        common = sorted(set(usable[test]) & set(usable[reference]), key=str)
        for lv, other in ((test, reference), (reference, test)):
            for s in sorted(set(usable[lv]) - set(usable[other]), key=str):
                ledger.append(_entry(s, lv, name, "no_evaluable_counterpart", usable[lv][s], False))
        design_cols = None
        if layout is not None:
            for s in [s for s in common if s not in layout]:
                ledger.append(_entry(s, test, name, "no_period_or_sequence", usable[test][s], False))
            common = [s for s in common if s in layout]
            design_cols = {"sequences": [layout[s][0] for s in common],
                           "test_periods": [layout[s][1] for s in common],
                           "ref_periods": [layout[s][2] for s in common]}
        tv, rv = [usable[test][s] for s in common], [usable[reference][s] for s in common]
        n_t = n_r = len(common)
        enough = len(common) >= prof.min_evaluable
    else:
        tv, rv = list(usable[test].values()), list(usable[reference].values())
        n_t, n_r = len(tv), len(rv)
        enough = min(n_t, n_r) >= prof.min_evaluable
        design_cols = None
    est = _estimate(tv, rv, paired=design == "crossover", alpha=prof.alpha, design_cols=design_cols)
    shown = {k: est.get(k) for k in ("gmr_pct", "ci_lower_pct", "ci_upper_pct", "cv_intra_pct",
                                     "ci_lower_raw", "ci_upper_raw", "df")}
    return {**shown, **_verdict(est, prof, lo, hi, enough),
            "n_evaluable": n_t if design == "crossover" else n_t + n_r, "n_test": n_t, "n_reference": n_r,
            "min_evaluable": prof.min_evaluable}


def assess_average_be(records: list[dict], *, design: str | None, profile: str | None, test: Any, reference: Any,
                      prespecified_exclusions: list[dict] | None = None, limits: tuple[float, float] | None = None,
                      limits_justification: str | None = None,
                      layout: dict[Any, tuple[Any, Any, Any]] | None = None) -> dict[str, Any]:
    """Average BE for every parameter in ``records`` (``[{subject, treatment, params: {name: value}}]``).

    ``layout`` ({subject: (sequence, test period, reference period)}) selects the 2x2 crossover model with
    sequence, subject(sequence), period and treatment terms; without it a crossover is the interim paired model.

    Any malformed input is a BEContractError, never another exception."""
    try:
        return _assess(records, design, profile, test, reference, prespecified_exclusions, limits,
                       limits_justification, layout)
    except BEContractError:
        raise
    except (TypeError, ValueError, ArithmeticError, InvalidOperation, KeyError, AttributeError) as exc:
        # keep the traceback: a genuine estimator bug must stay distinguishable from bad input
        _log.warning("average BE input rejected after %s", type(exc).__name__, exc_info=True)
        raise BEContractError(f"invalid input: {type(exc).__name__}: {exc}") from exc


def _exclusion_map(prespecified_exclusions, subjects) -> dict[Any, str]:
    if prespecified_exclusions is None:
        return {}
    if not isinstance(prespecified_exclusions, (list, tuple)):
        raise BEContractError("prespecified_exclusions must be a list of {subject, reason}")
    excluded: dict[Any, str] = {}
    for ex in prespecified_exclusions:
        if not isinstance(ex, dict):
            raise BEContractError(f"each pre-specified exclusion must be {{subject, reason}}; got {ex!r}")
        s, reason = ex.get("subject"), ex.get("reason")
        try:
            known = s in subjects
        except TypeError:
            raise BEContractError(f"pre-specified exclusion subject {s!r} is not a valid id") from None
        if not known:
            raise BEContractError(f"pre-specified exclusion names unknown subject {s!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise BEContractError(f"pre-specified exclusion of {s!r} has no reason")
        if s in excluded:
            raise BEContractError(f"subject {s!r} is pre-specified for exclusion more than once")
        excluded[s] = reason.strip()
    return excluded


def _assess(records, design, profile, test, reference, prespecified_exclusions, limits, limits_justification,
            layout=None):
    prof, lo, hi = _validate(records, design, profile, test, reference, limits, limits_justification)
    by = _index(records, design, test, reference)
    subjects = set(by[test]) | set(by[reference])
    excluded = _exclusion_map(prespecified_exclusions, subjects)
    names = sorted({k for arm in by.values() for p in arm.values() for k in p})
    ledger: list[dict] = []
    layout = layout if design == "crossover" else None
    params = {n: _one_parameter(n, by, design, test, reference, excluded, prof, lo, hi, ledger, layout)
              for n in names}
    custom = (lo, hi) != (prof.lower, prof.upper)
    return {
        "design": design, "profile": prof.name, "profile_source": prof.source,
        "test_level": test, "reference_level": reference,
        "ci_percent": prof.ci_percent, "alpha": prof.alpha, "limits": [str(lo), str(hi)],
        "limits_justification": limits_justification if custom else None,
        "rounding": (f"two decimals ({prof.rounding_source}); exact .xx5 ties round half up, a PharmAgent "
                     "implementation convention (the guidance does not define tie handling)" if prof.rounding_source
                     else "none (the profile's source states no rounding rule)"),
        "rounding_mode": "ROUND_HALF_UP" if prof.rounding_source else None,
        "model": MODEL_2X2 if layout is not None else INTERIM_MODEL,
        "ignored_treatments": sorted({r["treatment"] for r in records} - {test, reference}, key=str),
        "parameters": params, "exclusions": ledger,
        "bioequivalent": bool(params) and all(p["within_limits"] for p in params.values()),
    }


def report_disclosure(res: dict[str, Any]) -> str:
    """One sentence a report prints next to any BE verdict: profile, rounding rule, model status, exclusions."""
    n = len(res.get("exclusions") or [])
    custom = f" Custom limits, justified as: {res['limits_justification']}." if res.get("limits_justification") else ""
    return (f"Profile {res.get('profile')} ({res.get('profile_source')}); rounding: {res.get('rounding')}. "
            f"Model: {res.get('model')} {n} excluded value(s) are listed in the exclusion ledger.{custom}")


def report_ci(v: dict[str, Any]) -> str:
    """The CI the verdict used: the rounded bounds where the profile rounds, else the unrounded ones."""
    if v.get("ci_lower_rounded") is not None:
        return f"{v['ci_lower_rounded']}–{v['ci_upper_rounded']}"
    lo = v.get("ci_lower_raw", v.get("ci_lower_pct"))
    hi = v.get("ci_upper_raw", v.get("ci_upper_pct"))
    return "n/a" if lo is None else f"{lo!r}–{hi!r} (unrounded, as compared)"
