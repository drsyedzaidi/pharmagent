"""Turn harvested PK-DB answer keys into graded FIH tasks.

This is the *answer-key* half of the real-drug category. It consumes
:class:`~pharmacometricsbench.pkdb.loader.PKParameter` records (real, cited
drug-level PK measurements) and builds tasks whose ground truth is a
unit-harmonised, study-pooled literature value, graded within 2-fold — the
first-in-human accuracy criterion (Käser et al., Mol Pharm 2026).

Honesty contract
----------------
* Ground truth is the **pooled real observation**, never a fabricated number.
* A value is used only if its unit can be converted to the parameter's canonical
  unit by an *exact, dimensioned* factor (no body-weight guesses). Un-convertible
  units are dropped and counted — never coerced.
* Provenance (every contributing study sid + the raw value/unit) rides in
  ``meta`` and is not needed to answer the task.
* Anonymous harvests carry no PK parameters, so :func:`build_fih_pk_tasks` returns
  ``[]``. The category activates automatically once a token unlocks the outputs;
  it never invents tasks to fill the gap.

This category is *knowledge-grounded* (there is no tool that derives a drug's
population PK from the visible prompt), which is why it grades at 2-fold rather
than the tight tolerances of the deterministic tool-grounded categories.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from ..spec import Target, Task

if TYPE_CHECKING:  # avoid a runtime import cycle
    from .loader import DrugHarvest, PKParameter

# Canonical unit per parameter + the exact factors that convert into it.
# Only absolute (non weight/BSA-normalised) units are listed — anything else is
# intentionally absent so it gets dropped rather than silently mis-scaled.
_CANONICAL: dict[str, tuple[str, dict[str, float]]] = {
    "clearance": ("liter / hour", {
        "liter / hour": 1.0,
        "milliliter / minute": 0.06,      # mL/min -> L/h  (*60/1000)
        "liter / minute": 60.0,
        "milliliter / hour": 1e-3,
    }),
    "clearance/bioavailability": ("liter / hour", {
        "liter / hour": 1.0, "milliliter / minute": 0.06,
        "liter / minute": 60.0, "milliliter / hour": 1e-3,
    }),
    "auc_inf": ("milligram * hour / liter", {
        "milligram * hour / liter": 1.0,
        "gram * hour / liter": 1000.0,
        "microgram * hour / milliliter": 1.0,   # µg·h/mL == mg·h/L
        "nanogram * hour / milliliter": 1e-3,
    }),
    "cmax": ("milligram / liter", {
        "milligram / liter": 1.0,
        "microgram / milliliter": 1.0,          # µg/mL == mg/L
        "nanogram / milliliter": 1e-3,
        "gram / liter": 1000.0,
    }),
    "thalf": ("hour", {"hour": 1.0, "minute": 1.0 / 60.0, "day": 24.0}),
    "vd": ("liter", {"liter": 1.0, "milliliter": 1e-3}),
    "vd/bioavailability": ("liter", {"liter": 1.0, "milliliter": 1e-3}),
}


def harmonize(value: float, unit: str, measurement_type: str) -> float | None:
    """Convert ``value`` to the parameter's canonical unit, or ``None`` if the
    unit is unknown / not exactly convertible (weight-normalised, etc.)."""
    spec = _CANONICAL.get(measurement_type)
    if spec is None:
        return None
    _, factors = spec
    factor = factors.get((unit or "").strip().lower())
    if factor is None:
        return None
    return value * factor


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# Parameters that describe *oral/apparent* exposure — for these, records from a
# known non-oral route must not be pooled in. Route-independent parameters
# (t1/2) and explicitly-systemic ones (CL, V) are left to all reported routes.
_ORAL_ONLY_TYPES = {
    "clearance/bioavailability", "vd/bioavailability", "auc_inf", "cmax",
}

# Dose-DEPENDENT parameters: AUC and Cmax scale with dose (≈proportionally for
# linear PK), so pooling records taken at different doses yields a number that
# corresponds to no real dose and would false-fail a correct agent. They are
# harvested as raw PKParameters but NOT turned into pooled answer-key tasks until
# the authenticated schema links each value to its dose (then: dose-stratify or
# dose-normalise). Only dose-INDEPENDENT parameters (CL, CL/F, t1/2, V, V/F) seed
# FIH tasks.
_DOSE_DEPENDENT_TYPES = {"auc_inf", "auc_end", "cmax"}


def pool_parameter(
    params: list[PKParameter], measurement_type: str,
) -> tuple[float | None, list[PKParameter], int]:
    """Pool one parameter across **studies** as a unit-harmonised geometric mean.

    Pools per-study first (geomean within a study), then geomean across studies,
    so a multi-arm study is not upweighted relative to a single-value study.
    Returns ``(pooled_value_or_None, used_records, n_dropped)``. Only positive,
    finite, unit-convertible values from an admissible route contribute; the
    pooled value is ``None`` when nothing survives.
    """
    per_study: dict[str, list[float]] = defaultdict(list)
    used: list[PKParameter] = []
    dropped = 0
    for p in params:
        if p.measurement_type != measurement_type:
            continue
        # Route gate: for oral/apparent exposure, drop records from a KNOWN
        # non-oral route (unknown route is kept — the export often omits it).
        if measurement_type in _ORAL_ONLY_TYPES and p.route not in ("", "oral"):
            dropped += 1
            continue
        h = harmonize(p.value, p.unit, measurement_type)
        if h is None or not math.isfinite(h) or h <= 0.0:
            dropped += 1
            continue
        per_study[p.reference.study_sid].append(h)
        used.append(p)
    if not per_study:
        return None, [], dropped
    study_values = [_geomean(vs) for vs in per_study.values()]
    return _geomean(study_values), used, dropped


def build_fih_pk_tasks(
    harvests: list[DrugHarvest], *, min_studies: int = 2,
) -> list[Task]:
    """Build 2-fold-graded FIH tasks from harvested answer keys.

    One task per (drug, parameter) that has ≥ ``min_studies`` distinct
    open-licence source studies after unit harmonisation. Returns ``[]`` when no
    harvest carries PK parameters (the anonymous case).
    """
    tasks: list[Task] = []
    for h in harvests:
        by_type: dict[str, list[PKParameter]] = defaultdict(list)
        for p in h.pk_parameters:
            by_type[p.measurement_type].append(p)
        for mt, params in sorted(by_type.items()):
            if mt in _DOSE_DEPENDENT_TYPES:
                continue  # dose-dependent — cannot pool dose-blind (see note above)
            pooled, used, dropped = pool_parameter(params, mt)
            if pooled is None:
                continue
            source_sids = sorted({p.reference.study_sid for p in used})
            if len(source_sids) < min_studies:
                continue
            canonical_unit = _CANONICAL[mt][0]
            tasks.append(_make_task(h.drug, mt, pooled, canonical_unit,
                                    used, source_sids, dropped))
    return tasks


def _make_task(drug: str, mt: str, pooled: float, unit: str,
               used: list[PKParameter], source_sids: list[str],
               dropped: int) -> Task:
    # CL vs CL/F (and V vs V/F) are kept distinct — collapsing systemic and
    # apparent clearance would mis-state the answer key.
    label = {
        "clearance": "systemic clearance (CL)",
        "clearance/bioavailability": "apparent oral clearance (CL/F)",
        "auc_inf": "oral AUC extrapolated to infinity (AUC0-inf)",
        "cmax": "oral peak plasma concentration (Cmax)",
        "thalf": "terminal half-life (t1/2)",
        "vd": "volume of distribution (V)",
        "vd/bioavailability": "apparent volume of distribution (V/F)",
    }.get(mt, mt)
    # Only the oral/apparent parameters assert an oral route in the prompt.
    route = "oral" if mt in _ORAL_ONLY_TYPES else "as reported"
    return Task(
        task_id=f"fih-{drug}-{mt.replace('/', '_')}",
        category="fih_pk",
        prompt=(
            f"Estimate the population-typical {label} of {drug} in healthy adults. "
            f"Report a single value in {unit}. "
            "Accuracy is judged within 2-fold of the pooled literature value."
        ),
        dataset={"drug": drug, "parameter": mt, "unit": unit, "route": route,
                 "population": "healthy adults"},
        targets=[Target(mt, round(pooled, 6), {"type": "twofold"}, unit)],
        oracle="pkdb.pooled_literature_value",
        meta={
            "source": "PK-DB (pk-db.com), open-licence studies only",
            "n_studies": len(source_sids),
            "source_study_sids": source_sids,
            "n_dropped": dropped,
            "raw_values": [{"value": p.value, "unit": p.unit,
                            "route": p.route, "study": p.reference.study_sid}
                           for p in used],
            "pooling": "per-study geometric mean, then geometric mean across studies",
        },
    )
