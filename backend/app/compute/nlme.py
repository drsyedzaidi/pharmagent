"""True nonlinear mixed-effects (NLME) estimation for the PK model library.

This module implements two population estimators that share the same lognormal
between-subject variability (IIV) model and residual-error model:

  * **FOCE-I** — first-order conditional estimation *with interaction*. Each
    subject's empirical Bayes estimate (EBE) is found by minimizing the
    individual conditional objective (the inner problem), and the marginal
    likelihood is approximated by Laplace's method at that conditional mode. The
    population objective function value (OFV) is ``-2 * log L_marginal`` summed
    over subjects, minimized over the transformed population parameters (the
    outer problem).
  * **SAEM** — stochastic approximation expectation-maximization. The E-step
    samples each subject's random effects from their conditional posterior with
    a random-walk Metropolis kernel; the M-step updates Omega and the residual
    variance by Robbins-Monro stochastic approximation and refits the typical
    structural values by weighted least squares holding the sampled etas fixed.
    For comparability, the reported ``ofv`` is the FOCE/Laplace OFV evaluated at
    the final SAEM estimates.

Population model (lognormal IIV):

    param_p_i = theta_p * exp(eta_{p,i})   for p in iiv_params
    param_p_i = theta_p                     otherwise
    eta_i ~ N(0, Omega),  Omega = diag(omega2_p) by default

Omega is diagonal unless a *block* is requested, which makes a named subset of
the IIV parameters correlated (NONMEM's ``$OMEGA BLOCK``). A block is carried
in the outer vector as a Cholesky factor so it is positive-definite for any
parameter value; see "correlated IIV" below. The diagonal path is untouched by
that machinery -- every affected expression branches on ``spec.omega_block is
None`` and keeps its original scalar arithmetic, because the "equivalent"
matrix rewrite is not bitwise-identical.

Residual error on concentration f_ij = simulate(...)["cp"]:

    proportional : Var = (sigma_prop * f_ij) ** 2
    additive     : Var = sigma_add ** 2
    combined     : Var = sigma_add ** 2 + (sigma_prop * f_ij) ** 2

The structural model is *not* re-implemented here: predictions come from
``app.compute.pk_simulate.simulate`` over the model defined in
``app.compute.pk_models``. The module is pure, deterministic Python relying only
on ``numpy``, ``scipy`` and the two named ``app.compute`` imports. No file I/O,
network access, or agent imports.

An optional covariate model lets structural typical values depend on subject
covariates (power/linear/exponential for continuous, categorical for discrete);
``scm`` runs stepwise covariate modeling (forward selection + backward
elimination) on top of the FOCE-I OFV.

Public API:
    population_fit, focei_fit, saem_fit, scm, map_estimate, posthoc_residuals
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    # ProcessPoolExecutor transitively imports _multiprocessing, which the
    # Pyodide/WASM build removes; import it lazily where the SCM pool is built
    # (never reached single-threaded) so this module imports in the browser.
    from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2, norm

from app.compute.dosing import time_after_dose
from app.compute.pk_models import PKModel, get_model
from app.compute.pk_simulate import simulate

# ── numerical guards / fixed constants ───────────────────────────────────────
_EPS = 1e-12                 # floor for predictions / variances (avoid log 0)
_VAR_FLOOR = 1e-10           # hard floor on residual variance
_OMEGA_FLOOR = 1e-6          # hard floor on a diagonal Omega element (variance)
_SIGMA_FLOOR = 1e-4          # hard floor on a residual-error sigma
_BIG = 1e10                  # objective value returned on any failure
_HESS_STEP = 1e-4            # base step for the numerical (Gauss-Newton) Hessian
_COV_STEP = 5e-2            # larger step for the OFV Hessian -> parameter covariance
_COND_RED_FLAG = 1e3        # condition number above this signals over-parameterization
_RSE_CAP = 1e3             # RSE% above this -> report None (parameter unidentified)
_MIN_OBS_PER_SUBJECT = 1     # subjects with fewer usable obs are skipped
_SAEM_SEED_ITER = 100        # SAEM burn-in length when seeding FOCE-I (basin-finding only)
_AUTO_TOL_ABS = 1.0          # OFV units: two starts closer than this agree (same basin)
_AUTO_TOL_REL = 1e-3         # ...or within this fraction of |OFV|, whichever is larger
_AUTO_MAX_STARTS = 6         # cap on SAEM-seeded starts tried by method="auto"
# Inner budget for one constrained re-optimization during likelihood profiling.
# Deliberately tight: profiling makes tens of these calls per parameter and each
# objective evaluation is a full population Laplace pass. The inner solve only
# has to track a nearby optimum from a warm start, not find one cold.
_PROFILE_INNER_ITER = 8
_PROFILE_INNER_FEV = 30
_ROUND_DP = 6                # decimal places for all CWRES/IWRES payload floats

# CWRES (Hooker, Staatz & Karlsson 2007, Pharm Res 24:2187-2197): finite-
# difference step for G = df/deta at the conditional mode. Far above the 1e-7
# rounding granularity of `_make_predictor_cache`'s memo key, so `_jacobian`
# calls `_predict` directly rather than through that cache -- a step below the
# cache's rounding would silently collide eta+h with eta-h and return a zero
# Jacobian column, degrading CWRES to IWRES without any error.
_CWRES_FD_STEP = 1e-3
# Below this multiple of the prediction/variance floor (_EPS, _VAR_FLOOR), a
# row is a numerically degenerate (under/overflowed) simulation, not a real
# observation, and is dropped before G/Cov are formed.
_CWRES_FLOOR_MULT = 10.0
# Eigenvalues of Cov = G*Omega*G' + diag(R) are floored RELATIVE to the
# largest eigenvalue (not an absolute constant): this is scale-invariant
# across concentration units, unlike an absolute floor.
_CWRES_EIG_REL_FLOOR = 1e-12


# ─────────────────────────── subject preprocessing ──────────────────────────

class _Subject:
    """Validated, simulation-ready view of one subject's data.

    Attributes:
        sid: subject identifier (passed through unchanged).
        doses: dosing records forwarded to ``simulate``.
        t: strictly increasing observation times (1-D float array).
        c: observed concentrations aligned to ``t`` (1-D float array).
        wt: body weight used for allometric scaling.
        cov: baseline covariate values (name -> number or category string),
            used by the covariate model. Empty when no covariates are supplied.
        blq: boolean mask of below-quantification-limit observations (M3); ``lloq``
            is the limit. Empty/all-False unless an LLOQ is supplied.
        usable: whether the subject contributes to the likelihood.
    """

    __slots__ = ("sid", "doses", "t", "c", "wt", "cov", "blq", "lloq", "usable")

    def __init__(self, raw: dict) -> None:
        self.sid = raw.get("subject")
        self.doses = list(raw.get("doses") or [])
        t = np.asarray(raw.get("obs_t"), dtype=float)
        c = np.asarray(raw.get("obs_c"), dtype=float)
        lloq = raw.get("lloq")
        finite = np.isfinite(t) & np.isfinite(c)
        if lloq is not None:
            # M3 BLQ handling: keep BLQ records (flagged or below LLOQ) alongside
            # quantified observations; they contribute a censored-likelihood term.
            blq_in = raw.get("obs_blq")
            blq = (np.asarray(blq_in, dtype=bool) if blq_in is not None
                   else (c < float(lloq)))
            keep = finite & (blq | (c > 0.0))
            t, c, blq = t[keep], c[keep], blq[keep]
            order = np.argsort(t)
            self.t, self.c, self.blq = t[order], c[order], blq[order]
            self.lloq = float(lloq)
        else:
            # Default (validated) path: drop non-positive concentrations, no BLQ.
            mask = finite & (c > 0.0)
            t, c = t[mask], c[mask]
            order = np.argsort(t)
            self.t, self.c = t[order], c[order]
            self.blq = np.zeros(self.t.size, dtype=bool)
            self.lloq = None
        self.wt = float(raw.get("wt", 70.0) or 70.0)
        self.cov = dict(raw.get("cov") or {})
        self.usable = (self.t.size >= _MIN_OBS_PER_SUBJECT) and bool(self.doses)


def _prepare_subjects(subjects: list[dict]) -> list[_Subject]:
    """Wrap raw subject dicts; skip those too sparse to contribute."""
    return [s for s in (_Subject(r) for r in subjects) if s.usable]


# ───────────────────────── population-parameter packing ──────────────────────
#
# The outer optimizer works on an unconstrained vector ``x`` built as
#   [ log(theta_p) for p in all model params ]
#   [ log(omega2_p) for p in iiv_params ]
#   [ log(sigma_prop) ]   (if the error model has a proportional component)
#   [ log(sigma_add)  ]   (if the error model has an additive component)
# This keeps every estimated quantity strictly positive.

def _error_components(error_model: str) -> tuple[bool, bool]:
    """Return (has_prop, has_add) for the named residual-error model."""
    em = error_model.lower()
    if em == "proportional":
        return True, False
    if em == "additive":
        return False, True
    if em == "combined":
        return True, True
    raise ValueError(f"unknown error_model: {error_model!r}")


# ───────────────────────────── covariate model ──────────────────────────────
#
# A covariate effect multiplies a structural typical value by a subject-specific
# factor before IIV is applied:
#
#   power        : theta_p * (cov/center) ** beta
#   linear       : theta_p * (1 + beta*(cov - center))
#   exponential  : theta_p * exp(beta*(cov - center))
#   categorical  : theta_p * exp(beta_level)   (reference level -> factor 1)
#
# The coefficients ``beta`` are estimated on the natural (unconstrained) scale —
# they may be negative — so they are NOT log-transformed in the parameter vector.

_CONT_KINDS = ("power", "linear", "exponential")


def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class _CovEffect:
    """One parameter-covariate relationship (continuous or categorical)."""

    __slots__ = ("param", "covariate", "kind", "center", "levels", "n_coef")

    def __init__(self, param: str, covariate: str, kind: str,
                 center: float = 0.0, levels: tuple[str, ...] = ()) -> None:
        self.param = param
        self.covariate = covariate
        self.kind = kind
        self.center = float(center)
        self.levels = tuple(levels)              # non-reference categorical levels
        self.n_coef = len(self.levels) if kind == "categorical" else 1

    def factor(self, coefs: np.ndarray, value: Any) -> float:
        """Multiplicative factor on the typical parameter for one subject."""
        if value is None:
            return 1.0
        if self.kind == "categorical":
            key = str(value)
            for i, lv in enumerate(self.levels):
                if key == lv:
                    return math.exp(float(coefs[i]))
            return 1.0                            # reference / unseen level
        if not _is_num(value):
            return 1.0
        v = float(value)
        b = float(coefs[0])
        if self.kind == "power":
            return (max(v, _EPS) / max(self.center, _EPS)) ** b
        if self.kind == "linear":
            return max(1.0 + b * (v - self.center), _EPS)
        if self.kind == "exponential":
            return math.exp(b * (v - self.center))
        return 1.0

    @property
    def key(self) -> str:
        return f"{self.param}~{self.covariate}"

    def describe(self, coefs: np.ndarray) -> str:
        """Human-readable effect summary given the fitted coefficient(s)."""
        if self.kind == "categorical":
            parts = [f"{lv}: {math.exp(float(coefs[i])):.3g}x"
                     for i, lv in enumerate(self.levels)]
            return f"{self.covariate} (ref vs {', '.join(parts)})"
        b = float(coefs[0])
        if self.kind == "power":
            return f"{self.covariate}^{b:.3g} (center {self.center:.3g})"
        if self.kind == "linear":
            return f"1+{b:.3g}*({self.covariate}-{self.center:.3g})"
        return f"exp({b:.3g}*({self.covariate}-{self.center:.3g}))"


def _build_cov_effects(model_spec: list[dict] | None,
                       subjects: list[_Subject]) -> list[_CovEffect]:
    """Resolve a covariate-model spec into _CovEffects, computing centers and
    categorical levels from the (prepared) subject covariates."""
    effects: list[_CovEffect] = []
    for eff in model_spec or []:
        param = eff.get("param")
        cov = eff.get("covariate")
        if not param or not cov:
            continue
        kind = (eff.get("kind") or "power").lower()
        if kind == "categorical":
            vals = [str(s.cov.get(cov)) for s in subjects if s.cov.get(cov) is not None]
            if not vals:
                continue
            uniq = sorted(set(vals))
            ref = max(uniq, key=vals.count)      # reference = most frequent level
            levels = tuple(u for u in uniq if u != ref)
            if not levels:
                continue
            effects.append(_CovEffect(param, cov, "categorical", levels=levels))
        else:
            if kind not in _CONT_KINDS:
                kind = "power"
            center = eff.get("center")
            if center is None:
                nums = [float(s.cov[cov]) for s in subjects
                        if s.cov.get(cov) is not None and _is_num(s.cov.get(cov))]
                center = float(np.median(nums)) if nums else 1.0
            effects.append(_CovEffect(param, cov, kind, center=center))
    return effects


# ── correlated IIV (block Omega) ─────────────────────────────────────────────
# Omega is diagonal by default. A *block* makes a named subset of the IIV
# parameters correlated (the NONMEM `$OMEGA BLOCK` construct), which matters
# whenever two random effects genuinely move together -- e.g. CL and Vc, where
# a published 2-cmt oral reference estimates r = 0.54.
#
# The block is carried in the outer vector as a CHOLESKY FACTOR, never as the
# covariance directly: Omega = L L' with L lower-triangular is positive-definite
# for ANY value of the free parameters, so the optimizer cannot wander to an
# invalid Omega and no post-hoc repair is ever needed. The free parameters are
# the log-diagonal of L (positivity, and uniqueness of the factorization) and
# its strictly-lower entries raw (they may be negative).
#
# Segment layout, appended in place of the block members' log-variances:
#     [log omega2_p for each NON-block param, in iiv order]
#     [log L_00 ... log L_(k-1)(k-1)]          k log-diagonal entries
#     [L_10, L_20, L_21, ...]                  k(k-1)/2 strictly-lower, row-major
#
# With every strictly-lower entry at 0 this reproduces diag(omega2) EXACTLY, so
# a block fit can be seeded neutrally from a converged diagonal fit.

def _resolve_block(iiv_params: list[str],
                   omega_block: list[str] | tuple[str, ...] | None
                   ) -> tuple[int, ...] | None:
    """Map block member names to ASCENDING indices into ``iiv_params``.

    Ascending order is required, not cosmetic: the segment is decoded by
    writing the free parameters straight into a lower-triangular L, which is
    only a Cholesky factor of the intended Omega under that ordering.

    Returns None for "no block" (unset, or fewer than 2 usable members, which
    carries no correlation and must stay on the diagonal path).
    """
    if not omega_block:
        return None
    idx: set[int] = set()
    for name in omega_block:
        if name not in iiv_params:
            raise ValueError(
                f"omega_block member {name!r} is not an IIV parameter "
                f"(have {iiv_params})")
        idx.add(iiv_params.index(name))
    if len(idx) < 2:
        return None
    return tuple(sorted(idx))


def _omega_layout(n_omega: int, block_idx: tuple[int, ...] | None
                  ) -> tuple[int, tuple[int, ...]]:
    """``(n_omega_par, omega_slot)`` for an Omega segment.

    ``omega_slot[j]`` is the offset, WITHIN the Omega segment, of parameter
    ``j``'s own log-variance -- or -1 when ``j`` is a block member, whose
    marginal variance is a function of several Cholesky entries rather than a
    single slot. Consumers that index the segment per-parameter must go through
    this map: with a non-trailing block (say a block on [CL, KA] of
    [CL, V, KA]) the naive ``enumerate(iiv_params)`` offset silently reads a
    different parameter's slot and mislabels its %CV / %RSE.

    Diagonal case returns ``(n_omega, (0, 1, ..., n_omega-1))`` -- the identity
    map, so existing literal indexing is unchanged.
    """
    if block_idx is None:
        return n_omega, tuple(range(n_omega))
    k = len(block_idx)
    n_par = (n_omega - k) + k + k * (k - 1) // 2
    slot: list[int] = []
    nxt = 0
    for j in range(n_omega):
        if j in block_idx:
            slot.append(-1)
        else:
            slot.append(nxt)
            nxt += 1
    return n_par, tuple(slot)


def _chol_from_seg(seg: np.ndarray, k: int) -> np.ndarray:
    """Lower-triangular L from ``[log diag] + [strictly-lower, row-major]``."""
    L = np.zeros((k, k), dtype=float)
    L[np.diag_indices(k)] = np.exp(np.asarray(seg[:k], dtype=float))
    if k > 1:
        L[np.tril_indices(k, -1)] = np.asarray(seg[k:], dtype=float)
    return L


def _seg_from_chol(L: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_chol_from_seg`."""
    k = L.shape[0]
    diag = np.log(np.maximum(np.diag(L), _EPS))
    if k == 1:
        return np.asarray(diag, dtype=float)
    return np.concatenate([diag, L[np.tril_indices(k, -1)]])


def _omega_full_from_seg(spec: _PopSpec, seg: np.ndarray) -> np.ndarray:
    """Full ``n_omega x n_omega`` Omega from its packed segment.

    Block-diagonal by construction: independent variances on the diagonal for
    non-block parameters, the reconstructed ``L L'`` for the block members.
    """
    n = spec.n_omega
    om = np.zeros((n, n), dtype=float)
    block = spec.block_idx or ()
    for j in range(n):
        s = spec.omega_slot[j]
        if s >= 0:
            om[j, j] = max(math.exp(float(seg[s])), _OMEGA_FLOOR)
    if block:
        k = len(block)
        off = n - k                       # non-block log-variances precede it
        L = _chol_from_seg(np.asarray(seg[off:], dtype=float), k)
        blk = L @ L.T
        for a, ja in enumerate(block):
            for b, jb in enumerate(block):
                om[ja, jb] = float(blk[a, b])
    return om


def _seg_from_omega_full(spec: _PopSpec, om: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_omega_full_from_seg` (encode Omega to its segment)."""
    n = spec.n_omega
    block = spec.block_idx or ()
    parts: list[float] = []
    for j in range(n):
        if spec.omega_slot[j] >= 0:
            parts.append(math.log(max(float(om[j, j]), _OMEGA_FLOOR)))
    if block:
        sub = np.asarray([[float(om[a, b]) for b in block] for a in block], dtype=float)
        sub = 0.5 * (sub + sub.T)                       # enforce exact symmetry
        sub[np.diag_indices(len(block))] = np.maximum(
            np.diag(sub), _OMEGA_FLOOR)
        parts.extend(_seg_from_chol(np.linalg.cholesky(sub)).tolist())
    return np.asarray(parts, dtype=float)


def _omega_delta_se(spec: _PopSpec, seg: np.ndarray, cov_seg: np.ndarray,
                    *, step: float = 1e-5
                    ) -> tuple[dict[str, float], dict[str, float]]:
    """Delta-method standard errors for quantities derived from a block Omega.

    Every other population parameter in this module is log-linked, which makes
    the relative SE on the natural scale equal to the SE on the estimation
    scale -- the ``RSE% = 100*SE`` shortcut used above. That shortcut does NOT
    apply here: the packed block parameters are Cholesky entries, a MIX of
    log-scale (the diagonal of L) and raw-scale (its strictly-lower entries),
    and each reported quantity is a nonlinear function of several of them. So
    each needs its own gradient, and

        Var(g) = grad(g)' Cov(phi) grad(g).

    Gradients are central finite differences on the same decode the fit uses,
    so they cannot drift from it.

    Returns ``(var_se, corr_se)``: the SE of each block member's MARGINAL
    variance, and the ABSOLUTE SE of each pairwise correlation. The correlation
    deliberately gets an absolute SE rather than an RSE%, because r may sit
    arbitrarily close to zero -- exactly the null case the block is tested
    against -- where a relative error is meaningless and a CI is what a reader
    actually needs.
    """
    seg = np.asarray(seg, dtype=float)
    blk = spec.block_idx or ()

    def quantities(s: np.ndarray) -> np.ndarray:
        om = _omega_full_from_seg(spec, s)
        corr = _block_corr(om)
        vals = [float(om[j, j]) for j in blk]
        vals += [float(corr[a, b])
                 for i, a in enumerate(blk) for b in blk[i + 1:]]
        return np.asarray(vals, dtype=float)

    base = quantities(seg)
    jac = np.zeros((base.size, seg.size), dtype=float)
    for k in range(seg.size):
        h = step * max(abs(float(seg[k])), 1.0)
        sp, sm = seg.copy(), seg.copy()
        sp[k] += h
        sm[k] -= h
        jac[:, k] = (quantities(sp) - quantities(sm)) / (2.0 * h)

    var_g = np.einsum("ij,jk,ik->i", jac, cov_seg, jac)
    se_g = np.sqrt(np.maximum(var_g, 0.0))

    var_se = {spec.iiv_params[j]: float(se_g[i]) for i, j in enumerate(blk)}
    corr_se: dict[str, float] = {}
    pos = len(blk)
    for i, a in enumerate(blk):
        for b in blk[i + 1:]:
            corr_se[f"{spec.iiv_params[a]}~{spec.iiv_params[b]}"] = float(se_g[pos])
            pos += 1
    return var_se, corr_se


def _project_block(spec: _PopSpec, om: np.ndarray) -> np.ndarray:
    """Impose the spec's Omega structure on an unstructured covariance.

    The SAEM M-step's empirical second moment ``E'E/n`` is dense, but only the
    covariances *within* the declared block are free parameters. Everything
    outside it is structurally zero and must be projected away each iteration,
    or the model silently drifts to a full Omega that the packed vector cannot
    represent and the reported degrees of freedom do not account for.
    """
    n = spec.n_omega
    out = np.zeros((n, n), dtype=float)
    out[np.diag_indices(n)] = np.maximum(np.diag(om), _OMEGA_FLOOR)
    for a in (spec.block_idx or ()):
        for b in (spec.block_idx or ()):
            if a != b:
                out[a, b] = float(om[a, b])
    return out


def _shrink_to_pd(om: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """Nudge a covariance back inside the PD cone if sampling noise left it out.

    The empirical second moment of a finite eta sample can be singular (or
    indefinite after Robbins-Monro mixing) when two etas are nearly collinear
    -- exactly the regime a correlation block invites. Shrinking toward its own
    diagonal is the minimal repair that preserves the variances, and the
    subsequent Cholesky encode would otherwise raise.
    """
    sym = 0.5 * (om + om.T)
    if np.all(np.linalg.eigvalsh(sym) > floor):
        return sym
    d = np.diag(np.diag(sym))
    for w in (0.05, 0.1, 0.25, 0.5, 0.9):          # increasing shrinkage
        cand = (1.0 - w) * sym + w * d
        if np.all(np.linalg.eigvalsh(cand) > floor):
            return cand
    return d


class _OmegaPrior(NamedTuple):
    """Everything the conditional objective needs from a non-diagonal Omega.

    Carried as a pair so the Cholesky factorization is done ONCE per objective
    evaluation rather than once per subject per inner iteration (the inner
    Nelder-Mead runs up to 40 iterations for each of n_subjects).

    ``prec`` is Omega^-1 and ``logdet`` is log|Omega|. A value of None anywhere
    downstream means "diagonal", and every consumer then takes its original
    scalar branch untouched.
    """
    prec: np.ndarray
    logdet: float


def _omega_prior(om: np.ndarray) -> _OmegaPrior:
    """Precision and log-determinant of a covariance matrix, via Cholesky.

    Uses the Cholesky factor for both quantities so they are mutually
    consistent and the determinant cannot go negative through round-off:
    log|Omega| = 2*sum(log diag(L)).
    """
    L = np.linalg.cholesky(0.5 * (om + om.T))
    logdet = 2.0 * float(np.sum(np.log(np.maximum(np.diag(L), _EPS))))
    Li = np.linalg.inv(L)
    return _OmegaPrior(prec=Li.T @ Li, logdet=logdet)


def _block_corr(om: np.ndarray) -> np.ndarray:
    """Correlation matrix of a covariance matrix (0 where a variance is 0)."""
    d = np.sqrt(np.maximum(np.diag(om), 0.0))
    safe = np.where(d > 0.0, d, 1.0)
    corr = om / np.outer(safe, safe)
    corr[d <= 0.0, :] = 0.0
    corr[:, d <= 0.0] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


class _PopSpec:
    """Static layout describing how a parameter vector maps to named values."""

    __slots__ = ("model", "param_names", "iiv_params", "error_model",
                 "has_prop", "has_add", "n_theta", "n_omega",
                 "cov_effects", "n_cov",
                 "omega_block", "block_idx", "n_omega_par", "omega_slot")

    def __init__(self, model: PKModel, iiv_params: list[str],
                 error_model: str,
                 cov_effects: list[_CovEffect] | None = None,
                 *, omega_block: list[str] | tuple[str, ...] | None = None) -> None:
        self.model = model
        self.param_names: tuple[str, ...] = tuple(model.params)
        self.iiv_params: list[str] = list(iiv_params)
        self.error_model = error_model.lower()
        self.has_prop, self.has_add = _error_components(error_model)
        self.n_theta = len(self.param_names)
        self.n_omega = len(self.iiv_params)
        self.cov_effects: list[_CovEffect] = list(cov_effects or [])
        self.n_cov = sum(e.n_coef for e in self.cov_effects)
        self.block_idx = _resolve_block(self.iiv_params, omega_block)
        self.omega_block: tuple[str, ...] | None = (
            tuple(self.iiv_params[j] for j in self.block_idx)
            if self.block_idx else None)
        self.n_omega_par, self.omega_slot = _omega_layout(
            self.n_omega, self.block_idx)


def _apply_cov(spec: _PopSpec, theta: dict[str, float], cov_coefs: np.ndarray,
               subj: _Subject) -> dict[str, float]:
    """Subject-specific typical values: theta with covariate factors applied
    (before IIV). Returns ``theta`` unchanged when there is no covariate model,
    so the no-covariate path is byte-identical to the original."""
    if not spec.cov_effects:
        return theta
    p = dict(theta)
    i = 0
    for eff in spec.cov_effects:
        coefs = cov_coefs[i:i + eff.n_coef]
        if eff.param in p:
            p[eff.param] = p[eff.param] * eff.factor(coefs, subj.cov.get(eff.covariate))
        i += eff.n_coef
    return p


def _pack(spec: _PopSpec, theta: dict[str, float], cov_coefs: np.ndarray,
          omega2: dict[str, float], sigma_prop: float,
          sigma_add: float, *, omega_matrix: np.ndarray | None = None) -> np.ndarray:
    """Encode named population values into the unconstrained outer vector.

    Layout: [log theta] [covariate coefs (raw)] [Omega segment] [log sigma_*].
    Covariate coefficients are stored raw (not logged) since they may be
    negative. The Omega segment is ``[log omega2]`` for a diagonal spec, or the
    Cholesky layout documented above when ``spec.omega_block`` is set.

    ``omega_matrix`` is REQUIRED for a block spec and raises otherwise: the
    off-diagonals cannot be recovered from ``omega2`` (marginal variances)
    alone, and silently falling back to a diagonal encode here would corrupt
    the point the Hessian is taken around on the standard-error path.
    """
    parts: list[float] = [math.log(max(theta[p], _EPS)) for p in spec.param_names]
    parts += [float(c) for c in (cov_coefs if cov_coefs is not None else ())]
    if spec.omega_block is None:
        parts += [math.log(max(omega2[p], _OMEGA_FLOOR)) for p in spec.iiv_params]
    else:
        if omega_matrix is None:
            raise ValueError(
                "omega_matrix is required to pack a spec with omega_block="
                f"{spec.omega_block}; marginal variances cannot encode the "
                "off-diagonals")
        parts += [float(v) for v in _seg_from_omega_full(spec, omega_matrix)]
    if spec.has_prop:
        parts.append(math.log(max(sigma_prop, _SIGMA_FLOOR)))
    if spec.has_add:
        parts.append(math.log(max(sigma_add, _SIGMA_FLOOR)))
    return np.asarray(parts, dtype=float)


def _unpack(spec: _PopSpec, x: np.ndarray
            ) -> tuple[dict[str, float], np.ndarray, dict[str, float], float, float]:
    """Decode the outer vector into (theta, cov_coefs, omega2, sigma_prop,
    sigma_add)."""
    i = 0
    theta = {p: math.exp(x[i + k]) for k, p in enumerate(spec.param_names)}
    i += spec.n_theta
    cov_coefs = np.asarray(x[i:i + spec.n_cov], dtype=float)
    i += spec.n_cov
    if spec.omega_block is None:
        omega2 = {p: max(math.exp(x[i + k]), _OMEGA_FLOOR)
                  for k, p in enumerate(spec.iiv_params)}
        i += spec.n_omega
    else:
        # Marginal variances = diag(Omega), so every diagonal consumer of
        # omega2 (%CV, shrinkage, reporting) keeps working unchanged; callers
        # that need the off-diagonals use _omega_matrix(spec, x).
        _om = _omega_full_from_seg(spec, x[i:i + spec.n_omega_par])
        omega2 = {p: max(float(_om[k, k]), _OMEGA_FLOOR)
                  for k, p in enumerate(spec.iiv_params)}
        i += spec.n_omega_par
    sigma_prop = math.exp(x[i]) if spec.has_prop else 0.0
    if spec.has_prop:
        i += 1
    sigma_add = math.exp(x[i]) if spec.has_add else 0.0
    return theta, cov_coefs, omega2, max(sigma_prop, 0.0), max(sigma_add, 0.0)


def _omega_matrix(spec: _PopSpec, x: np.ndarray) -> np.ndarray | None:
    """Full Omega from an outer vector, or None for a diagonal spec.

    Returning None (rather than ``np.diag(omega2)``) for the diagonal case is
    deliberate: it keeps the diagonal hot path on its original scalar
    arithmetic instead of silently routing it through matrix algebra, which is
    NOT bitwise-equivalent.
    """
    if spec.omega_block is None:
        return None
    i = spec.n_theta + spec.n_cov
    return _omega_full_from_seg(spec, x[i:i + spec.n_omega_par])


# ───────────────────────────── prediction helpers ───────────────────────────

def _individual_params(spec: _PopSpec, theta: dict[str, float],
                       eta: np.ndarray) -> dict[str, float]:
    """Realize a subject's structural parameters from theta and its eta.

    ``eta`` is ordered to match ``spec.iiv_params``; non-IIV params take the
    typical value unchanged.
    """
    p = dict(theta)
    for k, name in enumerate(spec.iiv_params):
        p[name] = theta[name] * math.exp(float(eta[k]))
    return p


def _predict(spec: _PopSpec, subj: _Subject, theta: dict[str, float],
             eta: np.ndarray) -> np.ndarray:
    """Predicted concentrations for one subject at its observation times.

    Returns an array floored at ``_EPS``; on simulator failure returns NaNs so
    callers can penalize the objective instead of crashing.
    """
    p = _individual_params(spec, theta, eta)
    try:
        cp = simulate(spec.model, p, subj.doses, subj.t, wt=subj.wt)["cp"]
    except Exception:
        return np.full(subj.t.size, np.nan)
    cp = np.asarray(cp, dtype=float)
    if not np.all(np.isfinite(cp)):
        return np.full(subj.t.size, np.nan)
    return np.maximum(cp, _EPS)


def _residual_variance(spec: _PopSpec, f: np.ndarray, sigma_prop: float,
                       sigma_add: float) -> np.ndarray:
    """Per-observation residual variance under the configured error model."""
    var = np.zeros_like(f)
    if spec.has_add:
        var = var + sigma_add ** 2
    if spec.has_prop:
        var = var + (sigma_prop * f) ** 2
    return np.maximum(var, _VAR_FLOOR)


# ───────────────────── post-hoc residual diagnostics (CWRES) ─────────────────
#
# Conditional weighted residuals (Hooker, Staatz & Karlsson 2007): a linear
# (FOCE) approximation to the marginal distribution of y_i around the
# conditional mode eta_hat, evaluated at an ALREADY-FITTED population result
# (theta, Omega, sigma, eta_hat) -- never a new fit. Given
#
#   G_i        = df_i/deta at eta_hat                    (n_obs x n_omega)
#   E_FOCE(y_i)  = f_i(eta_hat) - G_i @ eta_hat
#   Cov(y_i)     = G_i @ Omega @ G_i.T + diag(R_i)
#
# CWRES_i = Cov(y_i)^{-1/2} (y_i - E_FOCE(y_i)). R_i (the residual-error
# variance) is evaluated at f_i(eta_hat) when `interaction=True` (matching
# NONMEM's FOCE-I / METH=1 INTER) or at f_i(eta=0) when False (the literal
# Hooker 2007 "FOCE" without interaction).

def _sid(x: Any) -> str:
    """Coerce a subject id to a stable join key.

    Subject ids reach this module from two different paths that do not agree
    on type: `_build_subjects` groups a DataFrame by the ID column (pandas
    gives back numpy scalars, e.g. `np.int64`), while a fitted result that has
    been persisted and reloaded has been through `json.dumps(..., default=str)`
    (numpy ints are NOT a subclass of `int`, so they fall through the default
    string-encoding path). `str()` on an int, `np.int64`, or a `str` all agree
    on the same decimal text, so it is a safe, order-independent join key
    regardless of which path a given sid came from.
    """
    return str(x)


def _jacobian(spec: _PopSpec, subj: _Subject, theta_i: dict[str, float],
             eta_hat: np.ndarray, *, step: float = _CWRES_FD_STEP) -> np.ndarray:
    """Central finite-difference G = df/deta at eta_hat (n_obs x n_omega).

    Calls `_predict` directly (never `_make_predictor_cache`): that cache
    rounds eta to 7 decimal places for memoization, and `step` here is fixed
    well above that granularity so an FD step never collides with the cache
    key by construction, but reaching for the cache anyway would be a latent
    footgun for any future change to `step`.
    """
    n_obs = subj.t.size
    n_eta = eta_hat.size
    g = np.zeros((n_obs, n_eta), dtype=float)
    for k in range(n_eta):
        eta_p = eta_hat.copy(); eta_p[k] += step
        eta_m = eta_hat.copy(); eta_m[k] -= step
        f_p = _predict(spec, subj, theta_i, eta_p)
        f_m = _predict(spec, subj, theta_i, eta_m)
        g[:, k] = (f_p - f_m) / (2.0 * step)
    return g


def _whiten(resid: np.ndarray, g: np.ndarray, omega2_vec: np.ndarray,
           r_var: np.ndarray) -> tuple[np.ndarray, bool]:
    """Symmetric (eigendecomposition) inverse square root of
    ``Cov = G @ diag(omega2) @ G.T + diag(r_var)``, applied to ``resid``.

    The symmetric ("Loewdin") root is used rather than a Cholesky root because
    it is independent of observation order within a subject, and because it
    was verified empirically against a real NONMEM 7.5.0 FOCE-I+INTER fit (the
    IU PopPK course 2-compartment reference, 120 subjects / 1943 observations,
    ``$TABLE ... CWRES``): the symmetric root reproduces NONMEM's own CWRES
    column at correlation 1.000000 / rmse 0.00048, whereas a Cholesky root on
    the identical inputs gives correlation only 0.894 for subjects with more
    than one observation (Cholesky and the symmetric root coincide only when
    Cov is diagonal, i.e. exactly one observation per subject -- multi-dose
    subjects here have up to 12). Neither Hooker (2007) nor Nguyen (2017)
    specifies which root to use; this choice is grounded in matching the tool
    this feature is meant to be comparable with, not a documentation reading.

    Small or negative eigenvalues (numerical noise on a near-singular Cov,
    e.g. nearly collinear Jacobian columns) are floored RELATIVE to the
    largest eigenvalue -- scale-invariant across concentration units, unlike
    an absolute floor. Returns whether the floor engaged, so the caller can
    exclude the affected subject from pooled statistics rather than report a
    residual against a covariance that was not really positive definite.
    """
    cov = g @ np.diag(omega2_vec) @ g.T + np.diag(r_var)
    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    floor = max(float(w.max()), 0.0) * _CWRES_EIG_REL_FLOOR
    floor = max(floor, _VAR_FLOOR)
    used_fallback = bool(np.any(w < floor))
    w_c = np.maximum(w, floor)
    inv_sqrt = v @ np.diag(w_c ** -0.5) @ v.T
    return inv_sqrt @ resid, used_fallback


def _cwres_subject(spec: _PopSpec, subj: _Subject, theta_i: dict[str, float],
                   eta_hat: np.ndarray, omega2_vec: np.ndarray,
                   sigma_prop: float, sigma_add: float, *,
                   interaction: bool) -> dict[str, Any] | None:
    """CWRES/IWRES for one subject at a FIXED eta_hat (never re-optimized).

    BLQ (censored) rows and rows where the prediction or its variance sits at
    the numerical floor (an under/overflowed simulation -- see `_predict`,
    `_residual_variance`) are dropped before G/Cov are formed, so a single
    degenerate row cannot blow up the whole subject's CWRES. Returns `None`
    when no rows survive, the Jacobian is non-finite, or the structural
    prediction failed outright (never raises).
    """
    keep_obs = ~subj.blq if subj.lloq is not None else np.ones(subj.t.size, dtype=bool)
    n_blq_dropped = int(np.count_nonzero(~keep_obs))

    f_hat_full = _predict(spec, subj, theta_i, eta_hat)
    if interaction:
        f_var_full = f_hat_full
    else:
        f_var_full = _predict(spec, subj, theta_i, np.zeros_like(eta_hat))
    if not (np.all(np.isfinite(f_hat_full)) and np.all(np.isfinite(f_var_full))):
        return None
    r_var_full = _residual_variance(spec, f_var_full, sigma_prop, sigma_add)

    floor_ok = ((f_hat_full > _EPS * _CWRES_FLOOR_MULT)
                & (r_var_full > _VAR_FLOOR * _CWRES_FLOOR_MULT))
    keep = keep_obs & floor_ok
    n_floored_dropped = int(np.count_nonzero(keep_obs & ~floor_ok))
    if int(np.count_nonzero(keep)) == 0:
        return None

    g_full = _jacobian(spec, subj, theta_i, eta_hat)
    if not np.all(np.isfinite(g_full)):
        return None

    t = subj.t[keep]
    y = subj.c[keep]
    f_hat = f_hat_full[keep]
    r_var = r_var_full[keep]
    g = g_full[keep, :]

    e_foce = f_hat - g @ eta_hat
    resid = y - e_foce
    cwres, used_fallback = _whiten(resid, g, omega2_vec, r_var)
    iwres_var = _residual_variance(spec, f_hat, sigma_prop, sigma_add)
    iwres = (y - f_hat) / np.sqrt(iwres_var)

    return {
        "time": t, "obs": y, "ipred": f_hat, "cpred": e_foce,
        "cwres": cwres, "iwres": iwres, "g": g, "cov_fallback": used_fallback,
        "n_blq_dropped": n_blq_dropped, "n_floored_dropped": n_floored_dropped,
    }


# ─────────────────────────── inner (conditional) problem ─────────────────────

def _make_predictor_cache(spec: _PopSpec, subj: _Subject,
                          theta: dict[str, float]) -> Callable[[np.ndarray], np.ndarray]:
    """Memoized predictor f(eta) for one subject at fixed theta.

    The conditional optimizer evaluates the same eta repeatedly (line searches,
    finite-difference Hessians); caching on rounded eta avoids redundant ODE
    integrations without affecting the result.
    """
    cache: dict[tuple[int, ...], np.ndarray] = {}

    def predict(eta: np.ndarray) -> np.ndarray:
        key = tuple(round(float(v), 7) for v in eta)
        hit = cache.get(key)
        if hit is None:
            hit = _predict(spec, subj, theta, eta)
            cache[key] = hit
        return hit

    return predict


def _ind_obj(eta: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
             subj: _Subject, spec: _PopSpec, omega2_vec: np.ndarray,
             sigma_prop: float, sigma_add: float,
             prior: _OmegaPrior | None = None) -> float:
    """Individual conditional objective (the quantity minimized for the EBE).

    ind_obj(eta) = sum_j[(y-f)^2 / Var + log(2*pi*Var)] + eta' Omega^-1 eta

    ``prior`` supplies Omega^-1 when Omega is not diagonal; when it is None the
    penalty keeps its original elementwise form (which is NOT bitwise-equal to
    the matrix expression, so the branch is required, not cosmetic).

    The ``log(2*pi*Var)`` term carries the interaction (Var depends on f, which
    depends on eta). When the subject has below-quantification-limit records
    (``subj.lloq`` set), those contribute the M3 censored term
    ``-2*log P(Y < LLOQ) = -2*log Phi((LLOQ - f)/sqrt(Var))`` instead of being
    dropped. Returns ``_BIG`` if the simulation failed.
    """
    f = predict(eta)
    if not np.all(np.isfinite(f)):
        return _BIG
    var = _residual_variance(spec, f, sigma_prop, sigma_add)
    blq = subj.blq
    if subj.lloq is not None and blq.any():
        obs = ~blq
        resid = subj.c[obs] - f[obs]
        data_term = float(np.sum(resid ** 2 / var[obs] + np.log(2.0 * math.pi * var[obs])))
        # M3: BLQ records contribute the probability of being below the LLOQ.
        z = (subj.lloq - f[blq]) / np.sqrt(var[blq])
        p_blq = np.clip(norm.cdf(z), 1e-12, 1.0)
        data_term += float(np.sum(-2.0 * np.log(p_blq)))
    else:
        resid = subj.c - f
        data_term = float(np.sum(resid ** 2 / var + np.log(2.0 * math.pi * var)))
    if prior is None:
        penalty = float(np.sum(eta ** 2 / omega2_vec))
    else:
        penalty = float(eta @ prior.prec @ eta)
    val = data_term + penalty
    return val if math.isfinite(val) else _BIG


def _numeric_hessian(fun: Callable[[np.ndarray], float], x: np.ndarray,
                     step: float = _HESS_STEP) -> np.ndarray:
    """Central-difference Hessian of a scalar function at ``x``.

    Used both for the Laplace approximation (H_i is the Hessian of ``0.5*ind_obj``
    at the conditional mode, default ``step``) and for the population-parameter
    covariance (Hessian of the OFV at the optimum, larger ``step`` to ride over
    the inner solver's numerical noise). Steps scale with |x| for conditioning.
    """
    d = x.size
    H = np.zeros((d, d), dtype=float)
    h = step * (1.0 + np.abs(x))
    f0 = fun(x)
    for i in range(d):
        xi = x.copy()
        xi[i] = x[i] + h[i]
        fpi = fun(xi)
        xi[i] = x[i] - h[i]
        fmi = fun(xi)
        H[i, i] = (fpi - 2.0 * f0 + fmi) / (h[i] ** 2)
    for i in range(d):
        for j in range(i + 1, d):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            val = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = val
    return H


def _conditional_mode(spec: _PopSpec, subj: _Subject, theta: dict[str, float],
                      omega2_vec: np.ndarray, sigma_prop: float,
                      sigma_add: float, eta0: np.ndarray | None = None,
                      prior: _OmegaPrior | None = None
                      ) -> tuple[np.ndarray, float, Callable[[np.ndarray], np.ndarray]]:
    """Find a subject's conditional mode eta_hat (the inner minimization).

    Returns (eta_hat, ind_obj_at_mode, predict) where ``predict`` is the
    memoized predictor (reused by the caller for the Laplace Hessian). Uses
    Nelder-Mead from eta0 (default 0); robust to the non-smoothness introduced
    by the ODE solver tolerances.
    """
    d = spec.n_omega
    if eta0 is None:
        eta0 = np.zeros(d, dtype=float)
    predict = _make_predictor_cache(spec, subj, theta)

    def obj(eta: np.ndarray) -> float:
        return _ind_obj(eta, predict, subj, spec, omega2_vec,
                        sigma_prop, sigma_add, prior)

    # Inner solves are warm-started across outer iterations, so a modest
    # iteration cap is sufficient and keeps the population OFV affordable.
    res = minimize(obj, eta0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 40})
    eta_hat = np.asarray(res.x, dtype=float)
    return eta_hat, float(obj(eta_hat)), predict


def _laplace_subject(spec: _PopSpec, subj: _Subject, theta: dict[str, float],
                     omega2_vec: np.ndarray, sigma_prop: float,
                     sigma_add: float, eta0: np.ndarray | None = None,
                     prior: _OmegaPrior | None = None
                     ) -> tuple[float, np.ndarray]:
    """Subject -2*log marginal likelihood by Laplace at the conditional mode.

    -2LL_i = ind_obj(eta_hat) + log|Omega| + log|det(H_i)|

    where H_i is the Hessian of ``0.5*ind_obj`` at the mode. The (2*pi)^(d/2)
    factor from Laplace's integral cancels the matching constant in the Gaussian
    prior's normalizer, so no leftover d*log(2*pi) term remains (verified against
    direct numerical integration of the marginal). Keeping the cancellation
    explicit also makes the OFV comparable across models with differing IIV
    dimension. Returns (-2LL_i, eta_hat). Falls back to a stable surrogate when
    the Hessian is not positive definite. The mode-finding predictor cache is
    reused for the Hessian's finite differences.
    """
    eta_hat, obj_at_mode, predict = _conditional_mode(
        spec, subj, theta, omega2_vec, sigma_prop, sigma_add, eta0, prior)

    # NOTE: `prior` must reach BOTH _ind_obj consumers in this function. The
    # mode above comes from the block prior; if this closure were left on the
    # diagonal penalty the Hessian would carry prior curvature diag(1/omega2)
    # while the mode came from Omega^-1, making log_det_H -- and therefore the
    # OFV, AIC/BIC, every SCM delta-OFV decision and _auto_fit's basin
    # arbitration -- wrong for block fits only. The correlation itself would
    # still come out right (Omega enters the M-step through the sampled etas),
    # so recovery tests would pass while the number that justifies the feature
    # was silently corrupt.
    def half_obj(eta: np.ndarray) -> float:
        return 0.5 * _ind_obj(eta, predict, subj, spec, omega2_vec,
                              sigma_prop, sigma_add, prior)

    H = _numeric_hessian(half_obj, eta_hat)
    if prior is None:
        log_det_omega = float(np.sum(np.log(omega2_vec)))
    else:
        log_det_omega = prior.logdet

    # Symmetrize and take the (clipped) eigenvalues for a robust log|det H|.
    Hs = 0.5 * (H + H.T)
    eigvals = np.linalg.eigvalsh(Hs)
    eigvals = np.clip(eigvals, _VAR_FLOOR, None)
    log_det_H = float(np.sum(np.log(eigvals)))

    m2ll = obj_at_mode + log_det_omega + log_det_H
    if not math.isfinite(m2ll):
        return _BIG, eta_hat
    return m2ll, eta_hat


def _population_ofv(spec: _PopSpec, subjects: list[_Subject],
                    theta: dict[str, float], cov_coefs: np.ndarray,
                    omega2: dict[str, float], sigma_prop: float, sigma_add: float,
                    eta_inits: list[np.ndarray] | None = None,
                    omega_matrix: np.ndarray | None = None
                    ) -> tuple[float, list[np.ndarray]]:
    """FOCE/Laplace population OFV = sum_i(-2LL_i), with per-subject EBEs.

    Each subject's typical values are covariate-adjusted (``_apply_cov``) before
    the conditional problem. Returns (ofv, eta_hats); ``eta_inits`` warm-starts
    each inner problem. ``omega_matrix`` (block specs only) switches the prior
    from diag(omega2) to the full Omega; its factorization is hoisted here so it
    happens once per OFV evaluation rather than once per inner iteration.
    """
    omega2_vec = np.array([omega2[p] for p in spec.iiv_params], dtype=float)
    omega2_vec = np.maximum(omega2_vec, _OMEGA_FLOOR)
    prior = _omega_prior(omega_matrix) if omega_matrix is not None else None
    total = 0.0
    eta_hats: list[np.ndarray] = []
    for idx, subj in enumerate(subjects):
        eta0 = eta_inits[idx] if eta_inits is not None else None
        theta_i = _apply_cov(spec, theta, cov_coefs, subj)
        m2ll, eta_hat = _laplace_subject(
            spec, subj, theta_i, omega2_vec, sigma_prop, sigma_add, eta0, prior)
        total += m2ll
        eta_hats.append(eta_hat)
    if not math.isfinite(total):
        total = _BIG
    return total, eta_hats


# ─────────────────────────────── initial guess ──────────────────────────────

def _terminal_ke(t: np.ndarray, c: np.ndarray) -> float:
    """Crude terminal-slope estimate of the elimination rate constant."""
    pos = c > 0
    tt, cc = t[pos], c[pos]
    if tt.size >= 2:
        n = min(3, tt.size)
        slope = np.polyfit(tt[-n:], np.log(cc[-n:]), 1)[0]
        if slope < 0 and math.isfinite(slope):
            return float(-slope)
    return 0.2


def _initial_theta(spec: _PopSpec, subjects: list[_Subject]) -> dict[str, float]:
    """Data-informed typical values, defaulting to the model defaults.

    Uses dose / Cmax for the central volume and ke*V for clearance, averaged
    (geometric mean) across subjects; other parameters keep model defaults.
    """
    model = spec.model
    theta = dict(model.defaults)
    vkey = "V" if "V" in theta else ("VC" if "VC" in theta else None)
    v_est: list[float] = []
    cl_est: list[float] = []
    for subj in subjects:
        if subj.c.size == 0 or not subj.doses:
            continue
        dose = float(subj.doses[-1]["amt"])
        cmax = float(np.max(subj.c))
        ke = _terminal_ke(subj.t, subj.c)
        if vkey and cmax > 0:
            v = max(dose / cmax, 1e-3)
            v_est.append(v)
            if "CL" in theta:
                cl_est.append(max(ke * v, 1e-3))
    if vkey and v_est:
        theta[vkey] = float(np.exp(np.mean(np.log(v_est))))
    if "CL" in theta and cl_est:
        theta["CL"] = float(np.exp(np.mean(np.log(cl_est))))
    return theta


# ─────────────────────────────── result assembly ────────────────────────────

def _cv_pct(omega2: float) -> float:
    """Lognormal IIV expressed as %CV = 100*sqrt(exp(omega2)-1)."""
    return 100.0 * math.sqrt(max(math.exp(omega2) - 1.0, 0.0))


def _shrinkage_pct(eta_hats: list[np.ndarray], omega2_vec: np.ndarray,
                   iiv_params: list[str]) -> dict[str, float]:
    """Eta-shrinkage per IIV parameter = 100*(1 - sd(eta_hat)/sqrt(omega2))."""
    if not eta_hats:
        return {p: 100.0 for p in iiv_params}
    E = np.vstack(eta_hats)  # (n_subjects, n_iiv)
    out: dict[str, float] = {}
    for k, p in enumerate(iiv_params):
        sd_eta = float(np.std(E[:, k], ddof=1)) if E.shape[0] > 1 else 0.0
        denom = math.sqrt(max(omega2_vec[k], _OMEGA_FLOOR))
        out[p] = round(100.0 * (1.0 - sd_eta / denom), 4) if denom > 0 else 100.0
    return out


def _individual_records(spec: _PopSpec, subjects: list[_Subject],
                        theta: dict[str, float], cov_coefs: np.ndarray,
                        eta_hats: list[np.ndarray]) -> list[dict[str, Any]]:
    """Per-subject EBE record: eta map and realized individual parameters
    (covariate-adjusted typical values with IIV applied)."""
    records: list[dict[str, Any]] = []
    for subj, eta in zip(subjects, eta_hats):
        eta_map = {p: round(float(eta[k]), 6)
                   for k, p in enumerate(spec.iiv_params)}
        theta_i = _apply_cov(spec, theta, cov_coefs, subj)
        p_ind = _individual_params(spec, theta_i, eta)
        params = {name: round(float(p_ind[name]), 6)
                  for name in spec.param_names}
        records.append({"subject": subj.sid, "eta": eta_map, "params": params})
    return records


def _empty_uncertainty(note: str = "") -> dict[str, Any]:
    """Uncertainty payload when standard errors are unavailable."""
    return {"theta_rse_pct": {}, "omega_rse_pct": {},
            "sigma_rse_pct": {"prop": None, "add": None},
            "cov_rse_pct": [], "condition_number": None, "cov_note": note,
            # Block-Omega only; present unconditionally so _assemble can read
            # it by subscript on every path, including non-converged fits.
            "omega_corr_se": {}}


def _parameter_uncertainty(spec: _PopSpec, ofv_at: Callable[[np.ndarray], float],
                           x_hat: np.ndarray, *, step: float = _COV_STEP
                           ) -> dict[str, Any]:
    """Asymptotic parameter uncertainty from the OFV Hessian at the optimum.

    The outer objective is OFV = -2*log L, so the observed Fisher information is
    ``I = 0.5 * Hess(OFV)`` and the estimate covariance is
    ``Cov(x_hat) = I^-1 = 2 * Hess(OFV)^-1`` on the log-estimation scale. Because
    every population parameter is log-linked (``p = exp(x)``), the delta method
    makes the relative SE on the natural scale equal to the SE on the log scale:
    ``RSE(p) = SE(p)/p = SE(x) = sqrt(Cov_ii)``. Hence RSE% = 100*sqrt(Cov_ii)
    holds directly for theta, sigma, and the Omega variance elements (the latter
    reported as the RSE of the variance, NONMEM convention). Covariate
    coefficients are estimated on the raw (un-logged) scale, so their RSE% is the
    usual ``100*SE/|coef|``.

    The condition number is the ratio of the largest to smallest eigenvalue of
    the correlation matrix of the estimates — a standard identifiability /
    over-parameterization diagnostic (> ~1000 is a red flag).

    Fails soft: returns empty RSEs with an explanatory ``cov_note`` when the
    information matrix is not finite / not positive-definite / singular, which is
    itself the meaningful signal that the model is over-parameterized for the
    data. A diagnostic condition number is still reported when obtainable.
    """
    try:
        H = _numeric_hessian(ofv_at, x_hat, step=step)
    except Exception:
        return _empty_uncertainty("covariance step failed; standard errors unavailable")
    if not np.all(np.isfinite(H)):
        return _empty_uncertainty("non-finite information matrix; standard errors unavailable")
    Hs = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(Hs)
    eig_max = float(np.max(eigvals)) if eigvals.size else 0.0
    if not math.isfinite(eig_max) or eig_max <= 0.0:
        return _empty_uncertainty(
            "information matrix not positive (fit not at a minimum); "
            "standard errors unavailable")
    # A *substantially* negative eigenvalue means the point is not a local
    # minimum (SEs would be meaningless). A merely near-zero/slightly-negative
    # one is finite-difference noise on an ill-conditioned (collinear) Hessian
    # and is floored — the affected directions then get very large SEs, which we
    # suppress per-parameter (report None) rather than discarding all SEs.
    if np.any(eigvals < -1e-2 * eig_max):
        out = _empty_uncertainty(
            "information matrix indefinite (fit not at a true minimum); "
            "standard errors unreliable")
        amin = float(np.min(np.abs(eigvals)))
        if amin > 0:
            out["condition_number"] = round(eig_max / amin, 1)
        return out
    floor = eig_max * 1e-10
    n_floored = int(np.sum(eigvals < floor))
    eig_use = np.clip(eigvals, floor, None)
    cov = 2.0 * (eigvecs @ np.diag(1.0 / eig_use) @ eigvecs.T)
    near_singular = bool(n_floored)
    diag = np.diag(cov)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
        return _empty_uncertainty("invalid covariance diagonal; standard errors unreliable")

    se = np.sqrt(diag)
    dinv = 1.0 / se
    corr = cov * np.outer(dinv, dinv)
    reig = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    rmin = float(np.min(reig))
    cond = round(float(np.max(reig) / rmin), 1) if rmin > 0 else None

    suppressed = {"v": False}

    def _rse(value: float) -> float | None:
        """RSE% capped: values above the cap mean the parameter is effectively
        unidentified (near-singular direction) -> report None, not a giant number."""
        if not math.isfinite(value) or value > _RSE_CAP:
            suppressed["v"] = True
            return None
        return round(value, 2)

    rse = 100.0 * se
    out_theta = {p: _rse(float(rse[k])) for k, p in enumerate(spec.param_names)}
    i = spec.n_theta
    # Covariate coefficients are raw-scale: RSE% = 100*SE/|coef|.
    cov_rse: list[float | None] = []
    for k in range(spec.n_cov):
        coef = float(x_hat[i + k])
        cov_rse.append(_rse(100.0 * float(se[i + k]) / abs(coef))
                       if abs(coef) > 1e-8 else None)
    i += spec.n_cov
    omega_corr_se: dict[str, float | None] = {}
    if spec.omega_block is None:
        out_omega = {p: _rse(float(rse[i + k])) for k, p in enumerate(spec.iiv_params)}
        i += spec.n_omega
    else:
        # Block members' marginal variances are functions of several Cholesky
        # entries, so they have no single slot to read: they need the delta
        # method. Non-block members still have their own log-variance slot and
        # keep the ordinary log-scale shortcut.
        seg = np.asarray(x_hat[i:i + spec.n_omega_par], dtype=float)
        cov_seg = cov[i:i + spec.n_omega_par, i:i + spec.n_omega_par]
        var_se, corr_se = _omega_delta_se(spec, seg, cov_seg)
        om_hat = _omega_full_from_seg(spec, seg)
        out_omega = {}
        for k, p in enumerate(spec.iiv_params):
            if spec.omega_slot[k] >= 0:
                out_omega[p] = _rse(float(rse[i + spec.omega_slot[k]]))
            else:
                v = float(om_hat[k, k])
                out_omega[p] = (_rse(100.0 * var_se[p] / v) if v > 0.0 else None)
        omega_corr_se = {k: (round(v, 6) if math.isfinite(v) else None)
                         for k, v in corr_se.items()}
        i += spec.n_omega_par
    sig = {"prop": None, "add": None}
    if spec.has_prop:
        sig["prop"] = _rse(float(rse[i]))
        i += 1
    if spec.has_add:
        sig["add"] = _rse(float(rse[i]))
        i += 1
    if near_singular or suppressed["v"]:
        note = ("information matrix near-singular — some standard errors are "
                "unavailable (model likely over-parameterized for this data)")
    elif cond is not None and cond > _COND_RED_FLAG:
        note = "condition number high — parameters may be poorly identified"
    else:
        note = ""
    return {"theta_rse_pct": out_theta, "omega_rse_pct": out_omega,
            "sigma_rse_pct": sig, "cov_rse_pct": cov_rse,
            "condition_number": cond, "cov_note": note,
            "omega_corr_se": omega_corr_se}


def _post_fit_uncertainty(spec: _PopSpec, subjects: list[_Subject],
                          theta: dict[str, float], cov_coefs: np.ndarray,
                          omega2: dict[str, float], sigma_prop: float,
                          sigma_add: float, eta_hats: list[np.ndarray], *,
                          enabled: bool, converged: bool,
                          omega_matrix: np.ndarray | None = None
                          ) -> dict[str, Any] | None:
    """Compute asymptotic uncertainty at the final estimates (FOCE-I & SAEM).

    Builds a clean OFV closure warm-started from the converged EBEs (so each
    perturbed Laplace pass starts at its mode and stays consistent), then defers
    to :func:`_parameter_uncertainty`. Returns ``None`` when disabled or when the
    fit did not converge (uncertainty at a non-optimum is meaningless).
    """
    if not (enabled and converged and subjects):
        return None
    final_etas = list(eta_hats)

    def ofv_at(xv: np.ndarray) -> float:
        th, cc, om, sp, sa = _unpack(spec, xv)
        # The perturbed Omega must be rebuilt from the perturbed vector, or the
        # Hessian would be taken with the off-diagonals held fixed and every
        # block standard error would be wrong.
        val, _ = _population_ofv(spec, subjects, th, cc, om, sp, sa, final_etas,
                                 _omega_matrix(spec, xv))
        return val

    x_hat = _pack(spec, theta, cov_coefs, omega2, sigma_prop, sigma_add,
                  omega_matrix=omega_matrix)
    return _parameter_uncertainty(spec, ofv_at, x_hat)


def _covariate_records(spec: _PopSpec, cov_coefs: np.ndarray,
                       cov_rse: list) -> list[dict[str, Any]]:
    """Public per-effect covariate summary (coefficient, RSE%, description)."""
    out: list[dict[str, Any]] = []
    ci = 0
    for eff in spec.cov_effects:
        coefs = np.asarray(cov_coefs[ci:ci + eff.n_coef], dtype=float)
        rse_slice = (cov_rse[ci:ci + eff.n_coef] if cov_rse
                     else [None] * eff.n_coef)
        if eff.kind == "categorical":
            coefficient: Any = {lv: round(float(coefs[k]), 6)
                                for k, lv in enumerate(eff.levels)}
            rse: Any = {lv: rse_slice[k] for k, lv in enumerate(eff.levels)}
        else:
            coefficient = round(float(coefs[0]), 6)
            rse = rse_slice[0] if rse_slice else None
        out.append({
            "param": eff.param, "covariate": eff.covariate, "kind": eff.kind,
            "center": (None if eff.kind == "categorical" else round(eff.center, 6)),
            "levels": (list(eff.levels) if eff.kind == "categorical" else None),
            "coefficient": coefficient, "rse_pct": rse,
            "description": eff.describe(coefs),
        })
        ci += eff.n_coef
    return out


def _assemble(spec: _PopSpec, method_label: str, model_key: str,
              theta: dict[str, float], cov_coefs: np.ndarray,
              omega2: dict[str, float],
              sigma_prop: float, sigma_add: float, ofv: float,
              eta_hats: list[np.ndarray], subjects: list[_Subject],
              n_obs: int, converged: bool, iterations: int,
              uncertainty: dict[str, Any] | None = None,
              omega_matrix: np.ndarray | None = None) -> dict[str, Any]:
    """Build the public result dict shared by FOCE-I and SAEM."""
    omega2_vec = np.array([omega2[p] for p in spec.iiv_params], dtype=float)
    sigma = {
        "prop": round(float(sigma_prop), 6) if spec.has_prop else None,
        "add": round(float(sigma_add), 6) if spec.has_add else None,
    }
    unc = uncertainty or _empty_uncertainty()
    return {
        "method": method_label,
        "model_key": model_key,
        "label": spec.model.label,
        "iiv_params": list(spec.iiv_params),
        "error_model": spec.error_model,
        "theta": {name: round(float(theta[name]), 6)
                  for name in spec.param_names},
        "theta_rse_pct": unc["theta_rse_pct"],
        "omega_cv_pct": {p: round(_cv_pct(omega2[p]), 4)
                         for p in spec.iiv_params},
        "omega_rse_pct": unc["omega_rse_pct"],
        "sigma": sigma,
        "sigma_rse_pct": unc["sigma_rse_pct"],
        "covariate_effects": _covariate_records(spec, cov_coefs,
                                                unc.get("cov_rse_pct") or []),
        "ofv": round(float(ofv), 4) if math.isfinite(ofv) else float(ofv),
        "condition_number": unc["condition_number"],
        "cov_note": unc["cov_note"],
        "shrinkage_pct": _shrinkage_pct(eta_hats, omega2_vec, spec.iiv_params),
        "n_subjects": len(subjects),
        "n_obs": int(n_obs),
        "n_blq": int(sum(int(s.blq.sum()) for s in subjects)),
        "converged": bool(converged),
        "individual": _individual_records(spec, subjects, theta, cov_coefs, eta_hats),
        "iterations": int(iterations),
        # Block-Omega keys. Emitted only when a block was actually fitted, so
        # every diagonal payload keeps its exact previous key set.
        **({} if (spec.omega_block is None or omega_matrix is None) else {
            "omega_block": list(spec.omega_block),
            "omega_matrix": [[round(float(v), 8) for v in row]
                             for row in np.asarray(omega_matrix, dtype=float)],
            "omega_corr": [[round(float(v), 6) for v in row]
                           for row in _block_corr(np.asarray(omega_matrix, dtype=float))],
            "omega_block_corr": {
                f"{spec.iiv_params[a]}~{spec.iiv_params[b]}": round(
                    float(_block_corr(np.asarray(omega_matrix, dtype=float))[a, b]), 6)
                for i, a in enumerate(spec.block_idx or ())
                for b in (spec.block_idx or ())[i + 1:]},
            # Absolute SE of each correlation (delta method). Absolute rather
            # than RSE% because r may sit near zero, where a relative error is
            # meaningless; this is what a Wald CI for r needs.
            "omega_corr_se": unc["omega_corr_se"],
        }),
    }


def _resolve_iiv(model: PKModel, iiv_params: list[str] | None) -> list[str]:
    """Choose IIV parameters: requested ∩ model params, else CL/V, else first 2."""
    model_params = list(model.params)
    if iiv_params:
        chosen = [p for p in iiv_params if p in model_params]
        if chosen:
            return chosen
    default = [p for p in ("CL", "V") if p in model_params]
    if default:
        return default
    return model_params[:2]


# ──────────────────────────────── FOCE-I ─────────────────────────────────────

def focei_fit(model_key: str, subjects: list[dict], *,
              iiv_params: list[str], error_model: str,
              max_iter: int = 200, compute_uncertainty: bool = True,
              covariate_model: list[dict] | None = None,
              init: dict[str, Any] | None = None,
              omega_block: list[str] | None = None) -> dict[str, Any]:
    """Fit a population PK model by FOCE-I (Laplace conditional estimation).

    Inner problem: per-subject conditional modes (EBEs). Outer problem: minimize
    the summed Laplace -2LL over [log theta, covariate coefs, log omega2,
    log sigma_*] with Powell. See the module docstring for the full spec.

    Args:
        model_key: key into the PK model registry.
        subjects: list of subject dicts (subject, doses, obs_t, obs_c, wt, cov).
        iiv_params: structural parameters carrying between-subject variability.
        error_model: "proportional", "additive", or "combined".
        max_iter: cap on outer-optimizer iterations.
        compute_uncertainty: compute the asymptotic covariance after convergence.
        covariate_model: optional list of covariate-effect specs, each
            {"param", "covariate", "kind"(power|linear|exponential|categorical),
            "center"?}. Coefficients are estimated jointly with the structural
            parameters.
        init: optional warm-start {"theta": dict, "omega2": dict, "sigma_prop",
            "sigma_add", "cov_coefs": list}. Used as the optimizer's starting
            point instead of the cold data-derived defaults — set by ``scm`` to
            seed each candidate fit from the incumbent model (adding a covariate
            barely moves the structural optimum), cutting Powell evaluations.

    Returns:
        The standard NLME result dict (see module docstring / population_fit).
    """
    model = get_model(model_key)
    iiv = _resolve_iiv(model, iiv_params)
    prepared = _prepare_subjects(subjects)
    cov_effects = _build_cov_effects(covariate_model, prepared)
    spec = _PopSpec(model, iiv, error_model, cov_effects, omega_block=omega_block)
    n_obs = int(sum(s.t.size for s in prepared))
    cov0 = np.zeros(spec.n_cov, dtype=float)

    # Degenerate guard: nothing usable -> return defaults, not converged.
    if not prepared:
        theta0 = dict(model.defaults)
        omega2 = {p: 0.09 for p in iiv}
        return _assemble(spec, "FOCE-I", model_key, theta0, cov0, omega2,
                         0.1 if spec.has_prop else 0.0,
                         0.1 if spec.has_add else 0.0,
                         _BIG, [], prepared, 0, False, 0,
                         omega_matrix=(np.diag([omega2[p] for p in spec.iiv_params])
                                       if spec.omega_block else None))

    if init:
        theta0 = {**dict(model.defaults), **(init.get("theta") or {})}
        omega2_0 = {p: float((init.get("omega2") or {}).get(p, 0.09)) for p in iiv}
        sigma_prop0 = float(init.get("sigma_prop", 0.15)) if spec.has_prop else 0.0
        sigma_add0 = float(init.get("sigma_add", 0.5)) if spec.has_add else 0.0
        ic = np.asarray(init.get("cov_coefs") or [], dtype=float)
        if ic.size == spec.n_cov:
            cov0 = ic
    else:
        theta0 = _initial_theta(spec, prepared)
        omega2_0 = {p: 0.09 for p in iiv}            # ~30 %CV start
        sigma_prop0 = 0.15 if spec.has_prop else 0.0
        sigma_add0 = 0.5 if spec.has_add else 0.0
    # A block spec starts NEUTRAL: diag(omega2_0) encodes to zero off-diagonal
    # Cholesky entries, so the block fit begins at exactly the diagonal model
    # and any OFV gain it reports is attributable to the correlation alone.
    # A warm start that already carries a fitted Omega keeps its off-diagonals.
    omega_m0 = None
    if spec.omega_block is not None:
        warm_om = (init or {}).get("omega_matrix")
        omega_m0 = (np.asarray(warm_om, dtype=float) if warm_om is not None
                    else np.diag([omega2_0[p] for p in spec.iiv_params]))
    x0 = _pack(spec, theta0, cov0, omega2_0, sigma_prop0, sigma_add0,
               omega_matrix=omega_m0)

    # Warm-start cache: reuse the previous EBEs to seed each inner solve.
    warm: dict[str, list[np.ndarray]] = {"eta": None}
    eval_count = {"n": 0}
    start_ofv = {"v": None}

    def outer_obj(x: np.ndarray) -> float:
        theta, cc, omega2, s_prop, s_add = _unpack(spec, x)
        # For a block, Powell is moving the Cholesky entries themselves, so the
        # prior has to be rebuilt from the CURRENT x on every evaluation --
        # reusing a fixed Omega would hold the correlation constant and the
        # off-diagonals would never actually be estimated.
        ofv, eta_hats = _population_ofv(
            spec, prepared, theta, cc, omega2, s_prop, s_add, warm["eta"],
            _omega_matrix(spec, x))
        warm["eta"] = eta_hats
        if start_ofv["v"] is None:
            start_ofv["v"] = ofv
        eval_count["n"] += 1
        return ofv

    # ``maxfev`` bounds the total objective evaluations (each is a full
    # population Laplace pass), keeping wall-time predictable on large cohorts.
    # A handful of Powell sweeps suffice because the inner EBEs are warm-started.
    n_par = (spec.n_theta + spec.n_cov + spec.n_omega_par
             + int(spec.has_prop) + int(spec.has_add))
    res = minimize(outer_obj, x0, method="Powell",
                   options={"maxiter": max_iter,
                            "maxfev": min(max(12 * n_par, 24), 25 * max_iter),
                            "xtol": 1e-3, "ftol": 1e-3})

    x_final = np.asarray(res.x, dtype=float)
    theta, cc, omega2, s_prop, s_add = _unpack(spec, x_final)
    omega_m = _omega_matrix(spec, x_final)
    final_ofv, eta_hats = _population_ofv(
        spec, prepared, theta, cc, omega2, s_prop, s_add, warm["eta"], omega_m)
    # Converged if the optimizer reports success, OR it exhausted its evaluation
    # budget at a finite OFV that improved on the starting value (a stabilized
    # optimum that simply did not trip Powell's strict tolerance test).
    improved = (start_ofv["v"] is not None
                and math.isfinite(start_ofv["v"])
                and final_ofv <= start_ofv["v"] + 1e-6)
    converged = (bool(res.success) or improved) \
        and math.isfinite(final_ofv) and final_ofv < _BIG

    uncertainty = _post_fit_uncertainty(
        spec, prepared, theta, cc, omega2, s_prop, s_add, eta_hats,
        enabled=compute_uncertainty, converged=converged, omega_matrix=omega_m)

    return _assemble(spec, "FOCE-I", model_key, theta, cc, omega2, s_prop, s_add,
                     final_ofv, eta_hats, prepared, n_obs, converged,
                     int(res.nit) if hasattr(res, "nit") else eval_count["n"],
                     omega_matrix=omega_m,
                     uncertainty=uncertainty)


# ──────────────────────────────── SAEM ───────────────────────────────────────

def _combined_sigma_mle(resid: np.ndarray, f: np.ndarray,
                        var_prop0: float, var_add0: float) -> tuple[float, float]:
    """Joint MLE of (var_prop, var_add) for the *combined* residual-error model.

    Given residuals ``r = y - f`` and predictions ``f`` (BLQ already excluded),
    the two variance components maximize the residual log-likelihood

        -0.5 * sum_ij [ r_ij^2 / V_ij + log(V_ij) ],   V_ij = var_add + var_prop*f_ij^2

    which has **no closed form** — the additive and proportional pieces must be
    estimated *together* because each observation's variance depends on both.
    Estimating them independently (``var_add = mean(r^2)``,
    ``var_prop = mean((r/f)^2)``) double-counts the residual: on data spanning a
    wide concentration range the additive fit is dominated by the large absolute
    residuals of the high-concentration samples (whose scatter is really
    proportional), inflating ``sigma_add`` to values that can exceed most
    observed concentrations.

    Solved as a 2-D minimization over ``[log var_add, log var_prop]`` (keeping
    both components positive), warm-started from the current estimates. Returns
    ``(var_prop, var_add)`` floored at ``_SIGMA_FLOOR**2``; on solver failure the
    warm-start values are returned unchanged so Robbins-Monro simply holds.
    """
    f2 = f ** 2
    r2 = resid ** 2
    floor = _SIGMA_FLOOR ** 2
    a0 = max(float(var_add0), floor)
    b0 = max(float(var_prop0), floor)

    def neg2ll(log_ab: np.ndarray) -> float:
        a = math.exp(float(log_ab[0]))
        b = math.exp(float(log_ab[1]))
        var = np.maximum(a + b * f2, _VAR_FLOOR)
        return float(np.sum(r2 / var + np.log(var)))

    try:
        sol = minimize(neg2ll, np.array([math.log(a0), math.log(b0)]),
                       method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 400})
        a = math.exp(float(sol.x[0]))
        b = math.exp(float(sol.x[1]))
        if math.isfinite(a) and math.isfinite(b):
            return max(b, floor), max(a, floor)
    except Exception:
        pass
    return b0, a0


def _saem_sigma_targets(spec: _PopSpec, subjects: list[_Subject],
                        theta: dict[str, float], cov_coefs: np.ndarray,
                        etas: list[np.ndarray], sigma_prop: float, sigma_add: float
                        ) -> tuple[float, float, int]:
    """Robbins-Monro *targets* for the residual variance component(s).

    Returns ``(var_prop_target, var_add_target, n_obs)`` — the variances the
    M-step drives ``sigma_prop**2`` / ``sigma_add**2`` toward at the current
    ``(theta, etas)``. Each component's target is its maximum-likelihood value
    under the configured error model:

      * proportional only : ``var_prop = mean((y/f - 1)^2)``      (var_add unused)
      * additive only      : ``var_add  = mean((y - f)^2)``        (var_prop unused)
      * combined           : ``(var_prop, var_add)`` from the joint MLE
        (:func:`_combined_sigma_mle`) — NOT the two single-component formulas,
        which each attribute the whole residual to one component and blow up
        ``sigma_add`` on wide-range concentration data.

    BLQ records are censored, not observed, so they are excluded from the
    residual-variance estimate (SAEM point-estimate approximation for M3).
    """
    resid_chunks: list[np.ndarray] = []
    f_chunks: list[np.ndarray] = []
    for subj, eta in zip(subjects, etas):
        theta_i = _apply_cov(spec, theta, cov_coefs, subj)
        f = _predict(spec, subj, theta_i, eta)
        if not np.all(np.isfinite(f)):
            continue
        keep = ~subj.blq if subj.lloq is not None else slice(None)
        c, fk = subj.c[keep], np.maximum(f[keep], _EPS)
        resid_chunks.append(c - fk)
        f_chunks.append(fk)
    if not f_chunks:
        return sigma_prop ** 2, sigma_add ** 2, 0
    resid = np.concatenate(resid_chunks)
    f = np.concatenate(f_chunks)
    n = int(resid.size)
    if n == 0:
        return sigma_prop ** 2, sigma_add ** 2, 0

    if spec.has_prop and spec.has_add:
        var_prop, var_add = _combined_sigma_mle(
            resid, f, sigma_prop ** 2, sigma_add ** 2)
        return var_prop, var_add, n
    if spec.has_prop:
        return float(np.mean((resid / f) ** 2)), sigma_add ** 2, n
    # additive only
    return sigma_prop ** 2, float(np.mean(resid ** 2)), n


def _saem_estep(spec: _PopSpec, subjects: list[_Subject],
                theta: dict[str, float], cov_coefs: np.ndarray,
                etas: list[np.ndarray],
                omega2_vec: np.ndarray, sigma_prop: float, sigma_add: float,
                rng: np.random.Generator, n_walk: int, scale: float,
                omega_matrix: np.ndarray | None = None
                ) -> list[np.ndarray]:
    """One E-step: random-walk Metropolis update of every subject's eta.

    The target is the unnormalized conditional posterior
    ``p(eta|y) ∝ exp(-0.5 * ind_obj(eta))``. ``n_walk`` proposals per subject
    are drawn from N(0, (scale^2) * Omega) -- diag(Omega) by default, the full
    Omega for a block spec, so the proposal follows the correlation instead of
    fighting it. Both forms consume exactly ``cur.size`` normal draws per
    proposal, so the diagonal path's RNG stream is bit-for-bit unchanged.
    """
    new_etas: list[np.ndarray] = []
    prop_sd = scale * np.sqrt(omega2_vec)
    prior = _omega_prior(omega_matrix) if omega_matrix is not None else None
    prop_chol = (scale * np.linalg.cholesky(0.5 * (omega_matrix + omega_matrix.T))
                 if omega_matrix is not None else None)
    for subj, eta in zip(subjects, etas):
        theta_i = _apply_cov(spec, theta, cov_coefs, subj)
        predict = _make_predictor_cache(spec, subj, theta_i)
        cur = eta.copy()
        cur_obj = _ind_obj(cur, predict, subj, spec, omega2_vec,
                           sigma_prop, sigma_add, prior)
        for _ in range(n_walk):
            z = rng.normal(0.0, 1.0, size=cur.size)
            step = z * prop_sd if prop_chol is None else prop_chol @ z
            cand = cur + step
            cand_obj = _ind_obj(cand, predict, subj, spec, omega2_vec,
                                sigma_prop, sigma_add, prior)
            # Metropolis acceptance on -0.5*ind_obj (log target).
            log_accept = -0.5 * (cand_obj - cur_obj)
            if math.log(rng.random() + _EPS) < log_accept:
                cur, cur_obj = cand, cand_obj
        new_etas.append(cur)
    return new_etas


def _saem_update_theta(spec: _PopSpec, subjects: list[_Subject],
                       theta: dict[str, float], cov_coefs: np.ndarray,
                       etas: list[np.ndarray], sigma_prop: float, sigma_add: float
                       ) -> tuple[dict[str, float], np.ndarray]:
    """Refit typical structural values AND covariate coefficients, etas fixed.

    For a purely **proportional** error model the typical values are fit on the
    log scale (``log y - log f``), which is the proper maximum-likelihood
    objective for lognormal/proportional residuals and avoids the downward bias
    of weighting plain residuals by ``1/f``. For models with an additive
    component the step minimizes the variance-weighted residual sum of squares
    ``sum_ij (y-f)^2 / Var_ij``. The optimization is a Gauss-Newton step (scipy
    ``least_squares``) over [log-typical values, raw covariate coefs]; Omega and
    sigma are updated by Robbins-Monro in the caller. Returns (theta, cov_coefs).

    For the **combined** error model the weights ``1/Var_ij`` are *frozen* at the
    incoming theta's predictions rather than recomputed from each trial
    prediction. Recomputing them makes ``Var`` depend on the parameters being
    optimized, so plain weighted least squares silently drops the ``log|Var|``
    term of the true likelihood and rewards inflating ``f`` (bigger ``f`` ->
    bigger ``Var`` -> smaller weighted residual), biasing the typical values.
    Freezing the weights turns the step into a one-step IRLS/GLS update whose
    fixed point — reached across SAEM iterations, which re-weight every pass —
    solves the unbiased estimating equation ``sum_ij g_ij (y-f)_ij / Var_ij = 0``.
    (Proportional-only uses the exact log-scale objective; additive-only has
    ``Var`` constant in ``f`` so freezing changes nothing.)
    """
    names = spec.param_names
    n_theta = spec.n_theta
    x0 = np.concatenate([
        np.array([math.log(max(theta[n], _EPS)) for n in names], dtype=float),
        np.asarray(cov_coefs, dtype=float),
    ])
    log_scale = spec.has_prop and not spec.has_add
    freeze_weights = spec.has_prop and spec.has_add

    # Combined error: residual-error SD evaluated once at the entry theta/etas,
    # per subject (BLQ excluded), so the Gauss-Newton weights stay fixed while
    # the parameters move. See docstring for why recomputing them biases theta.
    frozen_sd: list[np.ndarray] = []
    if freeze_weights:
        for subj, eta in zip(subjects, etas):
            theta_i = _apply_cov(spec, theta, cov_coefs, subj)
            f0 = _predict(spec, subj, theta_i, eta)
            keep = ~subj.blq if subj.lloq is not None else slice(None)
            base = f0[keep] if np.all(np.isfinite(f0)) else subj.c[keep]
            frozen_sd.append(
                np.sqrt(_residual_variance(spec, base, sigma_prop, sigma_add)))

    def residuals(x: np.ndarray) -> np.ndarray:
        th = {n: math.exp(x[k]) for k, n in enumerate(names)}
        cc = x[n_theta:]
        chunks: list[np.ndarray] = []
        for idx, (subj, eta) in enumerate(zip(subjects, etas)):
            th_i = _apply_cov(spec, th, cc, subj)
            f = _predict(spec, subj, th_i, eta)
            if not np.all(np.isfinite(f)):
                chunks.append(np.full(subj.t.size, 1e3))
                continue
            keep = ~subj.blq if subj.lloq is not None else slice(None)
            c, fk = subj.c[keep], f[keep]      # exclude censored (BLQ) records
            if log_scale:
                chunks.append(np.log(np.maximum(fk, _EPS)) - np.log(c))
            elif freeze_weights:
                chunks.append((c - fk) / frozen_sd[idx])
            else:
                var = _residual_variance(spec, fk, sigma_prop, sigma_add)
                chunks.append((c - fk) / np.sqrt(var))
        return np.concatenate(chunks) if chunks else np.zeros(1)

    try:
        sol = least_squares(residuals, x0, method="lm", max_nfev=200)
        new = {n: float(math.exp(v)) for n, v in zip(names, sol.x[:n_theta])}
        new_cc = np.asarray(sol.x[n_theta:], dtype=float)
        if (all(math.isfinite(v) and v > 0 for v in new.values())
                and np.all(np.isfinite(new_cc))):
            return new, new_cc
    except Exception:
        pass
    return theta, cov_coefs


def saem_fit(model_key: str, subjects: list[dict], *,
             iiv_params: list[str], error_model: str,
             max_iter: int = 300, seed: int = 20250614,
             compute_uncertainty: bool = True,
             covariate_model: list[dict] | None = None,
             omega_block: list[str] | None = None) -> dict[str, Any]:
    """Fit a population PK model by SAEM (stochastic approximation EM).

    Exploratory burn-in (gain = 1) for the first ~60% of iterations, then a
    smoothing phase with Robbins-Monro gain ``gamma_k = 1/(k - K1)``. Each
    iteration runs a Metropolis E-step (sampling eta), a Gauss-Newton theta
    M-step, and stochastic-approximation updates of Omega and the residual
    variance. The reported ``ofv`` is the FOCE/Laplace OFV evaluated at the
    final estimates so it is comparable to ``focei_fit``.

    Determinism: identical ``seed`` and inputs produce identical estimates.

    Args, Returns: see module docstring / population_fit.
    """
    model = get_model(model_key)
    iiv = _resolve_iiv(model, iiv_params)
    prepared = _prepare_subjects(subjects)
    cov_effects = _build_cov_effects(covariate_model, prepared)
    spec = _PopSpec(model, iiv, error_model, cov_effects, omega_block=omega_block)
    n_obs = int(sum(s.t.size for s in prepared))
    rng = np.random.default_rng(seed)
    cov_coefs = np.zeros(spec.n_cov, dtype=float)

    if not prepared:
        theta0 = dict(model.defaults)
        omega2 = {p: 0.09 for p in iiv}
        return _assemble(spec, "SAEM", model_key, theta0, cov_coefs, omega2,
                         0.1 if spec.has_prop else 0.0,
                         0.1 if spec.has_add else 0.0,
                         _BIG, [], prepared, 0, False, 0,
                         omega_matrix=(np.diag([omega2[p] for p in spec.iiv_params])
                                       if spec.omega_block else None))

    # ── initialization ──
    theta = _initial_theta(spec, prepared)
    omega2_vec = np.full(spec.n_omega, 0.09, dtype=float)   # ~30 %CV
    # A block starts NEUTRAL (zero off-diagonals) so it begins at exactly the
    # diagonal model and any OFV gain is attributable to the correlation.
    omega_m = (np.diag(omega2_vec.copy()) if spec.omega_block else None)
    sigma_prop = 0.15 if spec.has_prop else 0.0
    sigma_add = 0.5 if spec.has_add else 0.0
    etas = [np.zeros(spec.n_omega, dtype=float) for _ in prepared]

    k1 = max(int(round(0.6 * max_iter)), 1)                 # burn-in length
    n_walk = 2                                              # Metropolis steps/iter
    walk_scale = 0.4                                        # proposal scale

    prev_vec = _state_vector(theta, cov_coefs, omega2_vec, sigma_prop, sigma_add,
                             spec, omega_m)
    iterations_run = 0
    converged = False
    stable_streak = 0

    for k in range(1, max_iter + 1):
        iterations_run = k
        gamma = 1.0 if k <= k1 else 1.0 / (k - k1 + 1)

        # E-step: sample etas from their conditional posteriors.
        etas = _saem_estep(spec, prepared, theta, cov_coefs, etas, omega2_vec,
                           sigma_prop, sigma_add, rng, n_walk, walk_scale, omega_m)

        # Identifiability constraint E[eta] = 0: fold the sampled eta mean into
        # the typical values (theta_p *= exp(mean_eta_p)) and re-center the etas.
        # Without this, theta and the random-effect mean are confounded and the
        # typical values drift by exp(mean_eta).
        E = np.vstack(etas)
        eta_mean = np.mean(E, axis=0)
        for kk, name in enumerate(spec.iiv_params):
            theta[name] = theta[name] * math.exp(float(eta_mean[kk]))
        etas = [e - eta_mean for e in etas]

        # M-step (theta + covariate coefs): Gauss-Newton fit, centered etas fixed.
        theta, cov_coefs = _saem_update_theta(spec, prepared, theta, cov_coefs,
                                              etas, sigma_prop, sigma_add)

        # M-step (Omega): Robbins-Monro on the (centered) empirical 2nd moment.
        # For a block the moment is the FULL matrix E'E/n -- the closed-form
        # MLE of Omega given the sampled etas -- projected back onto the
        # declared block structure. This is the one place a correlation can
        # enter the model, and it needs no optimizer.
        E = np.vstack(etas)
        if omega_m is None:
            emp_omega2 = np.maximum(np.mean(E ** 2, axis=0), _OMEGA_FLOOR)
            omega2_vec = (1.0 - gamma) * omega2_vec + gamma * emp_omega2
            omega2_vec = np.maximum(omega2_vec, _OMEGA_FLOOR)
        else:
            emp = _project_block(spec, (E.T @ E) / float(E.shape[0]))
            omega_m = _shrink_to_pd(
                _project_block(spec, (1.0 - gamma) * omega_m + gamma * emp))
            omega2_vec = np.maximum(np.diag(omega_m), _OMEGA_FLOOR)

        # M-step (sigma): Robbins-Monro toward the ML variance target(s). For a
        # combined error model prop/add are estimated jointly (a single obs's
        # variance depends on both); estimating each alone double-counts the
        # residual and inflates sigma_add. See _saem_sigma_targets.
        var_prop_t, var_add_t, n_used = _saem_sigma_targets(
            spec, prepared, theta, cov_coefs, etas, sigma_prop, sigma_add)
        if n_used > 0:
            if spec.has_prop:
                new_var = max(var_prop_t, _SIGMA_FLOOR ** 2)
                sigma_prop = math.sqrt(
                    (1.0 - gamma) * sigma_prop ** 2 + gamma * new_var)
            if spec.has_add:
                new_var = max(var_add_t, _SIGMA_FLOOR ** 2)
                sigma_add = math.sqrt(
                    (1.0 - gamma) * sigma_add ** 2 + gamma * new_var)

        # Convergence: small relative change in the smoothing phase.
        cur_vec = _state_vector(theta, cov_coefs, omega2_vec, sigma_prop, sigma_add,
                                spec, omega_m)
        if k > k1:
            rel = float(np.max(np.abs(cur_vec - prev_vec)
                               / (np.abs(prev_vec) + 1e-8)))
            stable_streak = stable_streak + 1 if rel < 1e-3 else 0
            if stable_streak >= 3:
                converged = True
                break
        prev_vec = cur_vec

    omega2 = {p: float(omega2_vec[k]) for k, p in enumerate(spec.iiv_params)}

    # Final E-step -> conditional-mode EBEs and a comparable Laplace OFV.
    final_ofv, eta_hats = _population_ofv(
        spec, prepared, theta, cov_coefs, omega2, sigma_prop, sigma_add, etas,
        omega_m)
    converged = converged or (math.isfinite(final_ofv) and final_ofv < _BIG)

    uncertainty = _post_fit_uncertainty(
        spec, prepared, theta, cov_coefs, omega2, sigma_prop, sigma_add, eta_hats,
        enabled=compute_uncertainty, converged=converged, omega_matrix=omega_m)

    return _assemble(spec, "SAEM", model_key, theta, cov_coefs, omega2,
                     sigma_prop, sigma_add, final_ofv, eta_hats, prepared, n_obs,
                     converged, iterations_run, uncertainty=uncertainty,
                     omega_matrix=omega_m)


def _state_vector(theta: dict[str, float], cov_coefs: np.ndarray,
                  omega2_vec: np.ndarray, sigma_prop: float, sigma_add: float,
                  spec: _PopSpec, omega_matrix: np.ndarray | None = None
                  ) -> np.ndarray:
    """Flatten the current SAEM estimates into a vector for change tracking.

    A block contributes its CORRELATIONS, not its raw covariances. The caller's
    convergence test is purely relative -- ``|cur-prev| / (|prev| + 1e-8)`` --
    and a covariance legitimately sits near zero under the null of no
    correlation, which collapses that denominator and makes the relative change
    explode forever. The run would then never satisfy the stability criterion,
    burn every iteration, and STILL be reported as converged (the caller falls
    back to a finite-OFV check), so the failure would be invisible. A
    correlation is bounded in [-1, 1] and moves on the same relative scale as
    the variances already in this vector.
    """
    parts = [theta[p] for p in spec.param_names]
    parts += list(cov_coefs)
    parts += list(omega2_vec)
    if omega_matrix is not None and spec.block_idx:
        corr = _block_corr(omega_matrix)
        blk = spec.block_idx
        parts += [float(corr[a, b])
                  for i, a in enumerate(blk) for b in blk[i + 1:]]
    if spec.has_prop:
        parts.append(sigma_prop)
    if spec.has_add:
        parts.append(sigma_add)
    return np.asarray(parts, dtype=float)


# ──────────────────────────────── dispatch ───────────────────────────────────

def population_fit(model_key: str, subjects: list[dict], *,
                   method: str = "focei", iiv_params: list[str] | None = None,
                   error_model: str = "proportional", max_iter: int = 200,
                   seed: int = 20250614, compute_uncertainty: bool = True,
                   covariate_model: list[dict] | None = None,
                   omega_block: list[str] | None = None) -> dict[str, Any]:
    """Estimate a population PK model by the requested NLME method.

    Args:
        model_key: key into the PK model registry (``app.compute.pk_models``).
        subjects: list of {"subject", "doses":[{time,amt}], "obs_t", "obs_c",
            "wt", "cov"}; sparse subjects are skipped gracefully.
        method: "focei" (FOCE-I / Laplace), "saem" (stochastic approx. EM), or
            "focei_saem" (FOCE-I started from a short SAEM burn-in). Prefer
            "focei_saem" on harder models — many structural parameters, several
            IIV terms, covariates, or allometric scaling switched off — where
            cold-start Powell can settle in the wrong basin; see
            ``_focei_saem_fit``.
        iiv_params: structural parameters carrying IIV; default ["CL","V"] ∩
            model params, with a first-two-params fallback.
        error_model: "proportional" (default), "additive", or "combined".
        max_iter: iteration cap forwarded to the chosen estimator (for
            "focei_saem" this caps the FOCE-I stage; the SAEM seed is capped
            separately at ``_SAEM_SEED_ITER``).
        seed: RNG seed for the stochastic stage ("saem", and the seed run of
            "focei_saem"); plain FOCE-I is deterministic regardless.
        compute_uncertainty: compute asymptotic SE/RSE% after convergence.
        covariate_model: optional covariate-effect specs (see focei_fit).

    Returns:
        The standard NLME result dict documented in the module header.

    Raises:
        KeyError: if ``model_key`` is unknown.
        ValueError: if ``method`` or ``error_model`` is unrecognized.
    """
    model = get_model(model_key)
    iiv = _resolve_iiv(model, iiv_params)
    m = method.lower().replace("-", "_")
    if m == "focei":
        return focei_fit(model_key, subjects, iiv_params=iiv,
                         error_model=error_model, max_iter=max_iter,
                         compute_uncertainty=compute_uncertainty,
                         covariate_model=covariate_model,
                         omega_block=omega_block)
    if m == "saem":
        return saem_fit(model_key, subjects, iiv_params=iiv,
                        error_model=error_model, max_iter=max(max_iter, 1),
                        seed=seed, compute_uncertainty=compute_uncertainty,
                        covariate_model=covariate_model,
                        omega_block=omega_block)
    if m == "focei_saem":
        return _focei_saem_fit(model_key, subjects, iiv_params=iiv,
                               error_model=error_model, max_iter=max_iter,
                               seed=seed,
                               compute_uncertainty=compute_uncertainty,
                               covariate_model=covariate_model,
                               omega_block=omega_block)
    if m == "auto":
        return _auto_fit(model_key, subjects, iiv_params=iiv,
                         error_model=error_model, max_iter=max_iter, seed=seed,
                         compute_uncertainty=compute_uncertainty,
                         covariate_model=covariate_model,
                         omega_block=omega_block)
    raise ValueError(
        f"unknown method: {method!r} "
        "(expected 'focei', 'saem', 'focei_saem' or 'auto')")


def _auto_fit(model_key: str, subjects: list[dict], *,
              iiv_params: list[str], error_model: str, max_iter: int, seed: int,
              compute_uncertainty: bool,
              covariate_model: list[dict] | None,
              omega_block: list[str] | None = None) -> dict[str, Any]:
    """Escalating FOCE-I: probe for multimodality, multi-start only if found.

    Neither plain FOCE-I nor a single SAEM-seeded fit is reliable on a
    multimodal surface. Measured on the IU PopPK Week-8 case (2-cmt oral, IIV
    KA/CL/VC, allometry off), against a reference Vc of 64.6: cold FOCE-I
    converges cleanly to Vc 33.6, and SAEM seeding lands in the wrong basin in
    3 of 5 seeds (Vc 82-90). Neither failure announces itself — both report
    ``converged=True``.

    What IS reliable is the objective itself: the OFVs separate the basins
    unambiguously (23832 at the true optimum, 24443 cold, 26400+ for a bad
    seed). Every candidate here is a fully converged FOCE-I fit of the same
    model on the same data, so their OFVs are directly comparable and the
    lowest one wins.

    Strategy, cheapest-first:

    1. Cold FOCE-I.
    2. One SAEM-seeded probe — an independent start.
    3. If the two agree on OFV (within ``_AUTO_TOL_ABS`` / ``_AUTO_TOL_REL``),
       two independent starts found the same optimum: accept the better of the
       two. Cost: 2 fits.
    4. If they disagree, the surface is multimodal: run further SAEM-seeded
       starts (up to ``_AUTO_MAX_STARTS`` total) and return the global minimum
       OFV over every candidate, cold included.

    Agreement between two starts is evidence of a unique basin, not proof — two
    starts can in principle converge to the same wrong optimum. The guarantee
    this method does make is weaker but exact: the result is never worse (by
    OFV) than any candidate it evaluated, and never worse than plain FOCE-I.

    Deterministic: the cold fit is deterministic and every seeded start uses a
    seed derived from ``seed``, so identical inputs give identical estimates.
    """
    kw = dict(iiv_params=iiv_params, error_model=error_model,
              max_iter=max_iter, covariate_model=covariate_model,
              omega_block=omega_block)

    cold = focei_fit(model_key, subjects, compute_uncertainty=compute_uncertainty,
                     **kw)
    probe = _focei_saem_fit(model_key, subjects, seed=seed,
                            compute_uncertainty=compute_uncertainty, **kw)
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("cold", cold), (f"saem_seed:{seed}", probe)]

    def _ofv(r: dict[str, Any]) -> float:
        v = r.get("ofv")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else _BIG

    cold_ofv, probe_ofv = _ofv(cold), _ofv(probe)
    tol = max(_AUTO_TOL_ABS, _AUTO_TOL_REL * min(abs(cold_ofv), abs(probe_ofv)))
    both_converged = bool(cold.get("converged")) and bool(probe.get("converged"))
    disagree = abs(cold_ofv - probe_ofv) > tol
    # Escalate on EITHER signal. Disagreement means multiple basins; a start
    # that stopped on the iteration cap has not identified any optimum, and its
    # OFV gap would otherwise be read as multimodality when it is really
    # under-convergence. Both are reasons to spend more starts.
    escalated = disagree or not both_converged

    if escalated:
        # Multimodal: spend more starts. Seeds are derived from `seed` so the
        # whole escalation stays reproducible.
        for i in range(1, _AUTO_MAX_STARTS):
            s = seed + 7919 * i          # arbitrary large stride -> distinct streams
            candidates.append((f"saem_seed:{s}", _focei_saem_fit(
                model_key, subjects, seed=s,
                compute_uncertainty=compute_uncertainty, **kw)))

    winner_name, winner = min(candidates, key=lambda kv: _ofv(kv[1]))
    if escalated:
        reason = ("starts disagreed on OFV" if disagree
                  else "a start hit the iteration cap without converging")
    else:
        reason = "two independent starts converged and agreed"
    winner["method"] = f"FOCE-I (auto: {winner_name})"
    winner["auto"] = {
        "escalated": escalated,
        "reason": reason,
        "tol": round(tol, 6),
        "n_candidates": len(candidates),
        "winner": winner_name,
        "candidate_ofv": {name: (None if _ofv(r) >= _BIG else _ofv(r))
                          for name, r in candidates},
    }
    return winner


def _focei_saem_fit(model_key: str, subjects: list[dict], *,
                    iiv_params: list[str], error_model: str, max_iter: int,
                    seed: int, compute_uncertainty: bool,
                    covariate_model: list[dict] | None,
                    omega_block: list[str] | None = None) -> dict[str, Any]:
    """FOCE-I started from a short SAEM burn-in ("SAEM-seeded FOCE-I").

    FOCE-I's outer problem is minimized by Powell, a local derivative-free
    method: on a rough or multimodal surface it converges to whichever basin
    the cold, data-derived starting values happen to fall in. On harder models
    (many structural parameters + several IIV terms + covariates, especially
    when allometric scaling is switched off so WT no longer anchors CL/V) that
    basin can be the wrong one, and simply raising ``max_iter`` does not help —
    it can even land in a *worse* local minimum, since more iterations only
    search the same basin harder.

    SAEM's stochastic E-step explores the parameter space instead of descending
    it, so it is far less sensitive to starting values. Running a short SAEM
    first and handing its estimates to FOCE-I as ``init`` combines SAEM's
    basin-finding with FOCE-I's sharper terminal convergence and its exact
    Laplace OFV / asymptotic standard errors.

    The seed run is deliberately cheap and throwaway: capped at
    ``_SAEM_SEED_ITER`` iterations with no uncertainty computation. It only has
    to identify the right basin, not converge. If it fails to produce a usable
    warm start the fit degrades gracefully to an ordinary cold-start FOCE-I,
    and the returned ``method`` says so rather than claiming a seed that did
    not happen.

    Determinism is preserved: the SAEM stage is driven by ``seed`` and FOCE-I
    is deterministic, so identical inputs give identical estimates.
    """
    seed_iter = max(1, min(int(max_iter), _SAEM_SEED_ITER))
    seed_res = saem_fit(model_key, subjects, iiv_params=iiv_params,
                        error_model=error_model, max_iter=seed_iter,
                        seed=seed, compute_uncertainty=False,
                        covariate_model=covariate_model,
                        omega_block=omega_block)

    # A seed is only usable if SAEM actually produced a fit: a warm-start dict
    # AND a finite objective below the failure sentinel. (A degenerate seed —
    # empty/all-BLQ data — still returns default theta with ofv == _BIG, which
    # must NOT be dressed up as a warm start.) A non-converged but finite seed
    # IS usable: 100 SAEM iterations are meant to find the basin, not converge.
    init = _warm_init(seed_res)
    seed_ofv = seed_res.get("ofv")
    seed_usable = (init is not None and isinstance(seed_ofv, (int, float))
                   and math.isfinite(seed_ofv) and seed_ofv < _BIG)

    res = focei_fit(model_key, subjects, iiv_params=iiv_params,
                    error_model=error_model, max_iter=max_iter,
                    compute_uncertainty=compute_uncertainty,
                    covariate_model=covariate_model,
                    init=init if seed_usable else None,
                    omega_block=omega_block)

    if not seed_usable:
        # Degrade honestly to a cold start rather than claim a seed.
        res["seeded_by"] = None
        return res
    res["method"] = "FOCE-I (SAEM-seeded)"
    res["seeded_by"] = {"method": "SAEM", "iterations": seed_iter,
                        "ofv": seed_ofv,
                        "converged": seed_res.get("converged")}
    return res


# ───────────────────────── stepwise covariate modeling ───────────────────────

def _candidate_key(cand: dict) -> str:
    return f"{cand.get('param')}~{cand.get('covariate')}"


def _warm_init(result: dict | None, n_new_coefs: int = 0) -> dict | None:
    """Build a focei_fit ``init`` warm-start from an incumbent fit result.

    Carries the incumbent theta/Omega/sigma and its covariate coefficients,
    appending ``n_new_coefs`` zeros for a candidate effect being trialled.
    """
    if not result or result.get("status") not in (None, "ok") and "theta" not in result:
        return None
    if "theta" not in result:
        return None
    _, inc_coefs = _cov_effects_from_records(result.get("covariate_effects"))
    cov_coefs = (np.concatenate([inc_coefs, np.zeros(n_new_coefs)])
                 if n_new_coefs else inc_coefs)
    sig = result.get("sigma") or {}
    out = {
        "theta": dict(result.get("theta") or {}),
        "omega2": {p: cv_pct_to_omega2(v)
                   for p, v in (result.get("omega_cv_pct") or {}).items()},
        "sigma_prop": float(sig.get("prop") or 0.15),
        "sigma_add": float(sig.get("add") or 0.5),
        "cov_coefs": [float(c) for c in cov_coefs],
    }
    # omega_cv_pct carries only MARGINAL variances, so a block incumbent would
    # otherwise be warm-started as if it were uncorrelated -- throwing away the
    # correlation the seed run existed to find.
    if result.get("omega_matrix") is not None:
        out["omega_matrix"] = [list(map(float, row))
                               for row in result["omega_matrix"]]
    return out


def _scm_max_workers(n_tasks: int) -> int:
    return max(1, min(n_tasks, (os.cpu_count() or 2) - 1))


def _fit_batch(model_key: str, subjects: list[dict], iiv: list[str],
               error_model: str, max_iter: int,
               trials: list[tuple[list[dict], dict | None]],
               pool: ProcessPoolExecutor | None) -> list[dict]:
    """Fit a set of (covariate_model, init) trials, in parallel processes when a
    pool is given (FOCE-I is deterministic, so order/parallelism is irrelevant to
    the result), else serially."""
    def _one(cov_model, init):
        return focei_fit(model_key, subjects, iiv_params=iiv,
                         error_model=error_model, max_iter=max_iter,
                         compute_uncertainty=False, covariate_model=cov_model, init=init)
    if pool is not None and len(trials) > 1:
        futs = [pool.submit(focei_fit, model_key, subjects, iiv_params=iiv,
                            error_model=error_model, max_iter=max_iter,
                            compute_uncertainty=False, covariate_model=cm, init=ini)
                for cm, ini in trials]
        return [f.result() for f in futs]
    return [_one(cm, ini) for cm, ini in trials]


def scm(model_key: str, subjects: list[dict], *, candidates: list[dict],
        iiv_params: list[str] | None = None, error_model: str = "proportional",
        forward_p: float = 0.05, backward_p: float = 0.01,
        max_iter: int = 25, parallel: bool = True) -> dict[str, Any]:
    """Stepwise covariate modeling (forward selection + backward elimination).

    Reuses FOCE-I OFVs (the Laplace -2*log L, comparable across nested models).
    Forward: at each step fit the base model plus each not-yet-included candidate
    and add the one with the largest drop in OFV that exceeds the chi-square
    critical value at ``forward_p`` (df = #coefficients the effect adds). Repeat
    until no candidate is significant. Backward: from the forward model, remove
    any effect whose deletion raises OFV by *less* than the chi-square critical
    value at the stricter ``backward_p`` (i.e., not justified), least-justified
    first, until all remaining effects are significant.

    Args:
        candidates: list of effect specs {"param","covariate","kind"?,"center"?}.
        forward_p / backward_p: significance levels (default 0.05 / 0.01).
        max_iter: outer-iteration cap for each intermediate FOCE-I fit (search
            fits skip the covariance pass; only the final model gets RSE%).

    Returns:
        Dict with base/final OFV, the ordered step log, the selected effects,
        and the final fitted NLME result (with ``covariate_effects`` + RSE%).
    """
    model = get_model(model_key)
    label = model.label
    prepared = _prepare_subjects(subjects)

    # df per candidate (resolved against the actual data), de-duplicated.
    cand_df: dict[str, int] = {}
    uniq: list[dict] = []
    seen: set[str] = set()
    for cand in candidates or []:
        key = _candidate_key(cand)
        if key in seen:
            continue
        df = sum(e.n_coef for e in _build_cov_effects([cand], prepared))
        if df <= 0:
            continue
        seen.add(key)
        cand_df[key] = df
        uniq.append(cand)

    iiv = iiv_params or ["CL", "V"]

    def fit_one(cov_model: list[dict], *, uncertainty: bool = False,
                init: dict | None = None) -> dict[str, Any]:
        return focei_fit(model_key, subjects, iiv_params=iiv,
                         error_model=error_model, max_iter=max_iter,
                         compute_uncertainty=uncertainty, covariate_model=cov_model,
                         init=init)

    base = fit_one([])
    base_ofv = float(base.get("ofv", _BIG))
    steps: list[dict[str, Any]] = []

    if not uniq or not math.isfinite(base_ofv) or base_ofv >= _BIG:
        final = fit_one([], uncertainty=True, init=_warm_init(base))
        return {"status": "ok", "model_key": model_key, "label": label,
                "base_ofv": round(base_ofv, 4) if math.isfinite(base_ofv) else None,
                "final_ofv": final.get("ofv"), "forward_p": forward_p,
                "backward_p": backward_p, "selected": [], "steps": steps,
                "n_candidates": len(uniq), "final": final,
                "note": "no usable covariate candidates" if not uniq else
                        "base model did not converge"}

    # One process pool for the whole run (FOCE-I is deterministic, so parallel
    # candidate fits are order-independent). Each fit is seconds, dwarfing the
    # one-time worker startup. Falls back to serial on a single core / if disabled.
    if parallel and (os.cpu_count() or 1) > 1:
        from concurrent.futures import ProcessPoolExecutor  # lazy: single-core/WASM skips this
        pool = ProcessPoolExecutor(max_workers=_scm_max_workers(len(uniq)))
    else:
        pool = None
    try:
        # ── forward selection ── (warm-start every candidate from the incumbent)
        included: list[dict] = []
        current_ofv = base_ofv
        incumbent = base
        remaining = list(uniq)
        while remaining:
            trials = [(included + [c], _warm_init(incumbent, cand_df[_candidate_key(c)]))
                      for c in remaining]
            fits = _fit_batch(model_key, subjects, iiv, error_model, max_iter, trials, pool)
            best = None
            for cand, res in zip(remaining, fits):
                if not res.get("converged") or not math.isfinite(res.get("ofv", _BIG)):
                    continue
                dofv = current_ofv - float(res["ofv"])
                df = cand_df[_candidate_key(cand)]
                crit = float(chi2.isf(forward_p, df))
                if dofv > crit and (best is None or dofv > best["delta_ofv"]):
                    best = {"cand": cand, "fit": res, "ofv": float(res["ofv"]),
                            "delta_ofv": dofv, "df": df, "crit": crit}
            if best is None:
                break
            cand = best["cand"]
            included.append(cand)
            remaining = [c for c in remaining if _candidate_key(c) != _candidate_key(cand)]
            incumbent = best["fit"]
            current_ofv = best["ofv"]
            steps.append({"phase": "forward", "effect": _candidate_key(cand),
                          "delta_ofv": round(best["delta_ofv"], 3), "df": best["df"],
                          "crit": round(best["crit"], 3), "p": forward_p,
                          "ofv": round(best["ofv"], 3), "decision": "added"})

        # ── backward elimination (stricter p) ──
        while len(included) >= 1:
            full = fit_one(included, init=_warm_init(incumbent))
            full_ofv = float(full.get("ofv", _BIG))
            warm_full = _warm_init(full)
            trials = [([e for e in included if _candidate_key(e) != _candidate_key(eff)],
                       warm_full) for eff in included]
            fits = _fit_batch(model_key, subjects, iiv, error_model, max_iter, trials, pool)
            worst = None
            for eff, res in zip(included, fits):
                if not res.get("converged") or not math.isfinite(res.get("ofv", _BIG)):
                    continue
                dofv = float(res["ofv"]) - full_ofv     # OFV rise from removal
                df = cand_df[_candidate_key(eff)]
                crit = float(chi2.isf(backward_p, df))
                if dofv < crit and (worst is None or dofv < worst["delta_ofv"]):
                    worst = {"eff": eff, "fit": res, "delta_ofv": dofv, "df": df,
                             "crit": crit, "ofv": float(res["ofv"])}
            if worst is None:
                break
            eff = worst["eff"]
            included = [e for e in included if _candidate_key(e) != _candidate_key(eff)]
            incumbent = worst["fit"]
            steps.append({"phase": "backward", "effect": _candidate_key(eff),
                          "delta_ofv": round(worst["delta_ofv"], 3), "df": worst["df"],
                          "crit": round(worst["crit"], 3), "p": backward_p,
                          "ofv": round(worst["ofv"], 3), "decision": "removed"})
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # ── final model with full uncertainty (warm-started from the incumbent) ──
    final = fit_one(included, uncertainty=True, init=_warm_init(incumbent))
    selected = [{"param": c.get("param"), "covariate": c.get("covariate"),
                 "kind": (c.get("kind") or "power").lower()} for c in included]
    return {"status": "ok", "model_key": model_key, "label": label,
            "base_ofv": round(base_ofv, 4), "final_ofv": final.get("ofv"),
            "forward_p": forward_p, "backward_p": backward_p,
            "selected": selected, "steps": steps,
            "n_candidates": len(uniq), "final": final}


# ───────────────────── empirical-Bayes (MAP) forecasting ─────────────────────

def cv_pct_to_omega2(cv_pct: float) -> float:
    """Inverse of :func:`_cv_pct`: lognormal %CV -> variance omega2."""
    return math.log(1.0 + (float(cv_pct) / 100.0) ** 2)


def _cov_effects_from_records(records: list[dict] | None
                              ) -> tuple[list[_CovEffect], np.ndarray]:
    """Rebuild _CovEffects + the flat coefficient vector from the public
    ``covariate_effects`` records stored on a fitted result (centers/levels are
    already resolved, so no data is needed)."""
    effects: list[_CovEffect] = []
    coefs: list[float] = []
    for r in records or []:
        kind = r.get("kind", "power")
        if kind == "categorical":
            levels = tuple(r.get("levels") or [])
            effects.append(_CovEffect(r["param"], r["covariate"], "categorical",
                                      levels=levels))
            cf = r.get("coefficient") or {}
            coefs.extend(float(cf.get(lv, 0.0)) for lv in levels)
        else:
            effects.append(_CovEffect(r["param"], r["covariate"], kind,
                                      center=float(r.get("center") or 0.0)))
            coefs.append(float(r.get("coefficient") or 0.0))
    return effects, np.asarray(coefs, dtype=float)


def map_estimate(model_key: str, *, theta: dict[str, float],
                 omega2: dict[str, float], sigma_prop: float, sigma_add: float,
                 iiv_params: list[str], obs_t, obs_c, doses: list[dict],
                 wt: float = 70.0, cov: dict | None = None,
                 covariate_effects: list[dict] | None = None,
                 error_model: str = "proportional") -> dict[str, Any]:
    """Maximum-a-posteriori (empirical-Bayes) estimate of a NEW patient's random
    effects from sparse observations, given fitted population parameters.

    Maximizes the conditional posterior p(eta|y) ∝ exp(-0.5*ind_obj) over the
    patient's eta (the same conditional-mode solver used inside FOCE-I), with the
    population Omega as the prior. Returns the patient's individual parameters
    (covariate-adjusted typical values × exp(eta_MAP)).
    """
    model = get_model(model_key)
    iiv = _resolve_iiv(model, iiv_params)
    cov_effects, cov_coefs = _cov_effects_from_records(covariate_effects)
    spec = _PopSpec(model, iiv, error_model, cov_effects)
    subj = _Subject({"subject": "NEW", "doses": doses, "obs_t": obs_t,
                     "obs_c": obs_c, "wt": wt, "cov": cov or {}})
    full_theta = {**model.defaults, **theta}
    theta_i = _apply_cov(spec, full_theta, cov_coefs, subj)
    # A param in iiv_params but absent from omega2 gets the floor variance (its
    # eta prior is effectively fixed at 0 -> MAP returns the typical value for it
    # while still fitting the others). Tolerant lookup mirrors the floor pattern
    # used elsewhere and avoids a KeyError when a caller passes a partial omega.
    omega2_vec = np.array([max(omega2.get(p, _OMEGA_FLOOR), _OMEGA_FLOOR)
                           for p in spec.iiv_params], dtype=float)
    if subj.t.size == 0:                          # no levels -> fall back to typical
        eta_hat = np.zeros(spec.n_omega, dtype=float)
        obj = float("nan")
    else:
        eta_hat, obj, _ = _conditional_mode(
            spec, subj, theta_i, omega2_vec, sigma_prop, sigma_add)
    p_ind = _individual_params(spec, theta_i, eta_hat)
    return {
        "eta": {p: round(float(eta_hat[k]), 6) for k, p in enumerate(spec.iiv_params)},
        "individual_params": {n: round(float(p_ind[n]), 6) for n in spec.param_names},
        "typical_params": {n: round(float(theta_i[n]), 6) for n in spec.param_names},
        "n_obs": int(subj.t.size),
        "objective": None if not math.isfinite(obj) else round(obj, 4),
    }


def posthoc_residuals(model_key: str, subjects: list[dict], *,
                      theta: dict[str, float], omega2: dict[str, float],
                      sigma_prop: float, sigma_add: float,
                      iiv_params: list[str], error_model: str = "proportional",
                      covariate_effects: list[dict] | None = None,
                      etas: dict[Any, dict[str, float]] | None = None,
                      interaction: bool = True) -> dict[str, Any]:
    """Conditional weighted residuals (CWRES; Hooker, Staatz & Karlsson 2007)
    and standardized IWRES from an ALREADY-FITTED population result.

    Never runs a new fit and never re-optimizes a subject's conditional mode
    when its empirical Bayes estimate (EBE) is available in ``etas`` (keyed by
    subject id, joined via `_sid` so the join is robust to a persisted result's
    ids having round-tripped through JSON). A subject missing from ``etas``
    falls back to solving its conditional mode here (the same inner solver
    `population_fit` uses) -- this keeps the function usable even without a
    stored fit, at the cost of one Nelder-Mead solve per such subject.

    Parameters
    ----------
    model_key, iiv_params, error_model, covariate_effects:
        As reported by a converged `population_fit` result (`model_key`,
        `iiv_params`, `error_model`, `covariate_effects`).
    theta, omega2, sigma_prop, sigma_add:
        The fitted population parameters (`theta`, `omega_cv_pct` converted to
        variance via `cv_pct_to_omega2`, `sigma["prop"]`, `sigma["add"]`).
    etas:
        ``{subject_id: {iiv_param: eta_value}}``, typically a fitted result's
        ``individual`` records reshaped to ``{r["subject"]: r["eta"] for r in
        individual}``. May be ``None``/incomplete; missing subjects are
        resolved locally.
    interaction:
        ``True`` (default): the residual-error variance in CWRES is evaluated
        at the conditional-mode prediction (NONMEM's FOCE-I / METH=1 INTER;
        empirically verified against the IU PopPK course NONMEM 7.5.0
        reference to correlation 1.000000). ``False``: evaluated at the
        population prediction (eta=0) -- the literal Hooker (2007) "FOCE"
        variant without interaction.

    Returns
    -------
    dict with parallel arrays ``"time"``, ``"obs"``, ``"ipred"``, ``"cpred"``
    (the FOCE-linearized expectation E_FOCE(y), Hooker's "CPRED"), ``"cwres"``,
    ``"iwres"`` (standardized, R^-1/2 weighted -- NOT the unweighted log
    residual `fit_residuals` returns), ``"tad"``, plus ``"skipped_subjects"``
    and ``"cov_fallback_subjects"`` (subject ids, as strings) and a
    ``"summary"`` dict of counts and moments. Subjects whose whitening
    covariance required the relative eigenvalue floor (`_whiten`) are excluded
    from the pooled arrays and counted in ``cov_fallback_n`` rather than
    reported against a covariance that was not really positive definite.
    """
    model = get_model(model_key)
    iiv = _resolve_iiv(model, iiv_params)
    cov_effects, cov_coefs = _cov_effects_from_records(covariate_effects)
    spec = _PopSpec(model, iiv, error_model, cov_effects)
    omega2_vec = np.array([max(omega2.get(p, _OMEGA_FLOOR), _OMEGA_FLOOR)
                           for p in spec.iiv_params], dtype=float)
    full_theta = {**model.defaults, **theta}

    eta_by_sid = {_sid(k): v for k, v in (etas or {}).items()}
    subs = _prepare_subjects(subjects)

    time: list[float] = []
    obs: list[float] = []
    ipred: list[float] = []
    cpred: list[float] = []
    cwres: list[float] = []
    iwres: list[float] = []
    tad: list[float | None] = []

    n_etas_reused = 0
    n_etas_resolved = 0
    n_blq_dropped = 0
    n_floored_dropped = 0
    cov_fallback_n = 0
    cov_fallback_subjects: list[Any] = []
    skipped_subjects: list[Any] = []

    for subj in subs:
        theta_i = _apply_cov(spec, full_theta, cov_coefs, subj)
        eta_map = eta_by_sid.get(_sid(subj.sid))
        if eta_map is not None:
            eta_hat = np.array([float(eta_map.get(p, 0.0)) for p in spec.iiv_params],
                               dtype=float)
            n_etas_reused += 1
        else:
            eta_hat, _obj, _pred = _conditional_mode(
                spec, subj, theta_i, omega2_vec, sigma_prop, sigma_add)
            n_etas_resolved += 1

        out = _cwres_subject(spec, subj, theta_i, eta_hat, omega2_vec,
                             sigma_prop, sigma_add, interaction=interaction)
        if out is None:
            skipped_subjects.append(subj.sid)
            continue
        n_blq_dropped += out["n_blq_dropped"]
        n_floored_dropped += out["n_floored_dropped"]
        if out["cov_fallback"]:
            cov_fallback_n += 1
            cov_fallback_subjects.append(subj.sid)
            continue

        tad_sub = time_after_dose(out["time"], subj.doses)
        time.extend(float(v) for v in out["time"])
        obs.extend(float(v) for v in out["obs"])
        ipred.extend(float(v) for v in out["ipred"])
        cpred.extend(float(v) for v in out["cpred"])
        cwres.extend(float(v) for v in out["cwres"])
        iwres.extend(float(v) for v in out["iwres"])
        tad.extend(tad_sub)

    n = len(cwres)
    cwres_arr = np.asarray(cwres, dtype=float)
    iwres_arr = np.asarray(iwres, dtype=float)
    # ε-shrinkage (Karlsson & Savic 2007): how much of the assumed residual
    # variability is "eaten" by conditioning on the individual data.
    eps_shrinkage = (100.0 * (1.0 - float(np.std(iwres_arr))) if n else None)

    summary = {
        "n": n,
        "n_subjects_used": len(subs) - len(skipped_subjects) - cov_fallback_n,
        "n_subjects_skipped": len(skipped_subjects),
        "n_blq_dropped": n_blq_dropped,
        "n_floored_dropped": n_floored_dropped,
        "n_tad_null": sum(1 for v in tad if v is None),
        "cov_fallback_n": cov_fallback_n,
        "n_etas_reused": n_etas_reused,
        "n_etas_resolved": n_etas_resolved,
        "cwres_mean": round(float(np.mean(cwres_arr)), _ROUND_DP) if n else None,
        "cwres_sd": round(float(np.std(cwres_arr)), _ROUND_DP) if n else None,
        "cwres_pct_outside_1_96": (
            round(100.0 * float(np.count_nonzero(np.abs(cwres_arr) > 1.96)) / n, _ROUND_DP)
            if n else None),
        "eps_shrinkage_pct": round(eps_shrinkage, _ROUND_DP) if eps_shrinkage is not None else None,
        "interaction": bool(interaction),
        "cwres_variant": "focei" if interaction else "foce",
    }
    return {
        "time": [round(v, _ROUND_DP) for v in time],
        "obs": [round(v, _ROUND_DP) for v in obs],
        "ipred": [round(v, _ROUND_DP) for v in ipred],
        "cpred": [round(v, _ROUND_DP) for v in cpred],
        "cwres": [round(v, _ROUND_DP) for v in cwres],
        "iwres": [round(v, _ROUND_DP) for v in iwres],
        "tad": [None if v is None else round(v, _ROUND_DP) for v in tad],
        "skipped_subjects": [str(s) for s in skipped_subjects],
        "cov_fallback_subjects": [str(s) for s in cov_fallback_subjects],
        "summary": summary,
    }


def simulate_replicate_subject(model_key: str, theta: dict[str, float],
                               omega2: dict[str, float], sigma_prop: float, sigma_add: float,
                               iiv_params: list[str], error_model: str,
                               doses: list[dict], obs_t: Any, wt: float,
                               rng: np.random.Generator, *,
                               lloq: float | None = None,
                               max_resample: int = 20) -> dict[str, Any]:
    """Simulate one subject's observations under the population model: draw
    eta, realize the individual prediction, and add a residual-error draw --
    for SIMULATION-ESTIMATION replicate generation (`app.compute.simest`),
    never for the estimators themselves (which condition on observed data,
    never simulate it).

    Residual draws are resampled (bounded, ``max_resample`` attempts) so the
    simulated concentration stays positive under the SAME error model the
    downstream estimator will assume -- an unbounded draw can go negative
    under combined/additive error, which is not a valid concentration and
    would otherwise inject a high-leverage outlier into the replicate.
    ``lloq`` (if given) censors resampled-positive draws below it into an M3
    BLQ flag, matching ``_Subject``'s own censoring convention, rather than
    dropping them.

    No covariate model: this function realizes ``theta`` unmodified (plus
    IIV) at every call. Simulation-estimation with covariate effects is out
    of scope (see ``app.compute.simest.run_simest``); callers needing a
    covariate-adjusted typical value must apply ``_apply_cov`` before calling.

    Returns ``{"obs_t", "obs_c", "eta", "obs_blq", "lloq", "n_resampled",
    "n_negative_draws", "ok"}``. ``ok=False`` (with empty arrays) signals a
    failed structural simulation (e.g. a numerically pathological draw) --
    the caller should treat that replicate-subject as unusable, not crash.
    """
    model = get_model(model_key)
    full_theta = {**model.defaults, **theta}
    spec = _PopSpec(model, list(iiv_params), error_model)
    subj = _Subject({"subject": "SIM", "doses": doses, "obs_t": obs_t,
                     "obs_c": [1.0] * len(obs_t), "wt": wt})
    if subj.t.size == 0:
        return {"obs_t": [], "obs_c": [], "eta": {}, "obs_blq": None, "lloq": lloq,
                "n_resampled": 0, "n_negative_draws": 0, "ok": False}

    omega2_vec = np.array([max(omega2.get(p, _OMEGA_FLOOR), _OMEGA_FLOOR) for p in spec.iiv_params],
                          dtype=float)
    eta = (rng.normal(0.0, np.sqrt(omega2_vec)) if omega2_vec.size
          else np.zeros(0, dtype=float))
    f = _predict(spec, subj, full_theta, eta)
    if not np.all(np.isfinite(f)):
        return {"obs_t": [], "obs_c": [], "eta": {}, "obs_blq": None, "lloq": lloq,
                "n_resampled": 0, "n_negative_draws": 0, "ok": False}

    sd = np.sqrt(_residual_variance(spec, f, sigma_prop, sigma_add))
    y = np.array(f, dtype=float)
    blq = np.zeros(f.size, dtype=bool)
    n_resampled = 0
    n_negative_draws = 0
    for j in range(f.size):
        draw = None
        for attempt in range(max_resample):
            candidate = float(f[j] + sd[j] * rng.standard_normal())
            if candidate > 0.0:
                draw = candidate
                if attempt > 0:
                    n_resampled += 1
                break
            n_negative_draws += 1
        y[j] = draw if draw is not None else float(f[j])  # exhausted -> noiseless fallback
        if lloq is not None and y[j] < lloq:
            blq[j] = True

    eta_map = {p: float(eta[k]) for k, p in enumerate(spec.iiv_params)}
    return {
        "obs_t": [float(t) for t in subj.t], "obs_c": [float(v) for v in y],
        "eta": eta_map, "obs_blq": (blq.tolist() if lloq is not None else None),
        "lloq": lloq, "n_resampled": n_resampled, "n_negative_draws": n_negative_draws,
        "ok": True,
    }


def sir_inputs(model_key: str, subjects: list[dict], nlme_result: dict[str, Any],
               *, step: float = _COV_STEP) -> dict[str, Any] | None:
    """Assemble what :func:`app.compute.sir.run_sir` needs from a fitted result.

    SIR samples on the PACKED (estimation) scale and decodes afterwards, which
    is what lets it produce asymmetric natural-scale intervals. It therefore
    needs the packed point estimate, the packed covariance used as the proposal,
    the objective there, and closures to evaluate and decode.

    The covariance is recomputed here rather than carried on the fit result: it
    is an n x n matrix that would bloat every payload and the audit hash, and
    almost no consumer wants it. Cost is one numeric Hessian (~2*n_par^2
    objective passes) -- large next to nothing, negligible next to the hundreds
    of full refits a bootstrap needs.

    Returns None when the fit cannot support SIR (no result, or a degenerate
    information matrix), rather than a half-built input set.
    """
    if not nlme_result or nlme_result.get("status") not in (None, "ok"):
        return None
    if "theta" not in nlme_result:
        return None

    model = get_model(model_key)
    iiv = list(nlme_result.get("iiv_params") or [])
    prepared = _prepare_subjects(subjects)
    if not prepared or not iiv:
        return None

    cov_effects, cov_coefs = _cov_effects_from_records(
        nlme_result.get("covariate_effects"))
    spec = _PopSpec(model, iiv, nlme_result.get("error_model", "proportional"),
                    cov_effects,
                    omega_block=nlme_result.get("omega_block"))

    theta = {**dict(model.defaults), **(nlme_result.get("theta") or {})}
    omega2 = {p: cv_pct_to_omega2(v)
              for p, v in (nlme_result.get("omega_cv_pct") or {}).items()}
    for p in spec.iiv_params:                      # every IIV term must be present
        omega2.setdefault(p, 0.09)
    sig = nlme_result.get("sigma") or {}
    s_prop = float(sig.get("prop") or 0.0)
    s_add = float(sig.get("add") or 0.0)
    om_mat = (np.asarray(nlme_result["omega_matrix"], dtype=float)
              if nlme_result.get("omega_matrix") is not None else None)

    x_hat = _pack(spec, theta, cov_coefs, omega2, s_prop, s_add, omega_matrix=om_mat)

    # Warm-start the inner EBEs once, then reuse them so every perturbed pass
    # starts at its mode -- the same discipline _post_fit_uncertainty uses, and
    # what keeps the objective surface smooth enough to sample against.
    ofv_hat, eta_hats = _population_ofv(
        spec, prepared, theta, cov_coefs, omega2, s_prop, s_add, None, om_mat)
    if not math.isfinite(ofv_hat) or ofv_hat >= _BIG:
        return None

    def ofv_fn(xv: np.ndarray) -> float:
        th, cc, om, sp, sa = _unpack(spec, np.asarray(xv, dtype=float))
        val, _ = _population_ofv(spec, prepared, th, cc, om, sp, sa, eta_hats,
                                 _omega_matrix(spec, np.asarray(xv, dtype=float)))
        return val

    def decode_fn(xv: np.ndarray) -> dict[str, float]:
        th, _cc, om, sp, sa = _unpack(spec, np.asarray(xv, dtype=float))
        out = {p: float(th[p]) for p in spec.param_names}
        out.update({f"omega_{p}": _cv_pct(om[p]) for p in spec.iiv_params})
        if spec.has_prop:
            out["sigma_prop"] = float(sp)
        if spec.has_add:
            out["sigma_add"] = float(sa)
        return out

    try:
        H = _numeric_hessian(ofv_fn, x_hat, step=step)
    except Exception:
        return None
    if not np.all(np.isfinite(H)):
        return None
    Hs = 0.5 * (H + H.T)
    eig, vec = np.linalg.eigh(Hs)
    emax = float(np.max(eig)) if eig.size else 0.0
    if not math.isfinite(emax) or emax <= 0.0:
        return None
    # OFV = -2 log L, so Cov = 2 * Hess^-1. Floor the eigenvalues so a
    # near-singular direction yields a wide (not infinite) proposal — SIR can
    # down-weight an over-wide proposal, which is the safer failure here.
    floor = emax * 1e-10
    n_floored = int(np.sum(eig < floor))
    cov = 2.0 * (vec @ np.diag(1.0 / np.clip(eig, floor, None)) @ vec.T)
    # Condition number of the CORRELATION matrix of the estimates -- the same
    # quantity _parameter_uncertainty reports, so this flag is comparable with
    # the fit's own verdict and _COND_RED_FLAG is the threshold it was
    # calibrated for. (The Hessian's own condition number is a different scale
    # entirely and would make the flag meaningless.)
    dg = np.diag(cov)
    cond = None
    if np.all(np.isfinite(dg)) and np.all(dg > 0.0):
        dinv = 1.0 / np.sqrt(dg)
        corr = cov * np.outer(dinv, dinv)
        reig = np.linalg.eigvalsh(0.5 * (corr + corr.T))
        rmin = float(np.min(reig))
        cond = (float(np.max(reig)) / rmin) if rmin > 0 else None

    # Flooring makes a degenerate direction WIDE rather than infinite, which is
    # the safe direction for a proposal (SIR can down-weight an over-wide
    # proposal; it cannot invent samples in a region never visited). But the
    # result then LOOKS well-conditioned when the fit itself refused to stand
    # behind it -- `_parameter_uncertainty` suppresses those SEs entirely. The
    # caller is told, so it can say so rather than reporting confident
    # intervals built on a regularized singularity.
    # The FIT's own verdict is authoritative and is ORed in. Its Hessian is
    # taken at the converged EBEs and at the unrounded internal estimates,
    # whereas this one is rebuilt from the PUBLISHED (rounded) values with
    # re-optimized EBEs, so the two can disagree sharply -- observed 1.4e9 vs
    # 21.7 on an over-parameterized fit. Letting the rosier recomputation win
    # would silently overturn a refusal the fit was right to make.
    fit_flagged = (nlme_result.get("condition_number") is not None
                   and nlme_result["condition_number"] > _COND_RED_FLAG) \
        or bool(nlme_result.get("cov_note")) \
        or any(v is None for v in (nlme_result.get("theta_rse_pct") or {}).values())
    return {"x_hat": x_hat, "cov": cov, "ofv_hat": float(ofv_hat),
            "ofv_fn": ofv_fn, "decode_fn": decode_fn,
            "n_par": int(x_hat.size),
            "near_singular": (bool(n_floored)
                              or (cond is not None and cond > _COND_RED_FLAG)
                              or fit_flagged),
            "fit_flagged_uncertainty": fit_flagged,
            "n_floored_directions": n_floored,
            "condition_number": round(cond, 1) if cond and math.isfinite(cond) else None,
            "fit_condition_number": nlme_result.get("condition_number")}


def profile_ofv_factory(model_key: str, subjects: list[dict],
                        nlme_result: dict[str, Any]) -> dict[str, Any] | None:
    """Build the constrained-reoptimization closure log-likelihood profiling needs.

    Profiling fixes ONE parameter and lets every other re-optimize. Rather than
    teaching the fitter to hold a parameter (which would mean editing the
    validated estimator), this fixes one coordinate of the PACKED vector and
    minimizes the same objective over the remaining coordinates. The objective
    is identical to the one the fit minimized, so the resulting dOFV is
    directly comparable to the fit's own OFV — which is the entire premise of
    the chi-square cut-off.

    Only the log-linked structural THETAs are offered. The packed vector also
    holds omega and sigma, but profiling a variance means re-optimizing across
    a boundary-constrained parameter whose chi-square reference is not 1 df,
    and reporting it as though it were would be wrong.

    Returns ``{"profile_ofv_fn", "estimates", "ofv_hat", "initial_step"}``, or
    None when the fit cannot support it.
    """
    inp = sir_inputs(model_key, subjects, nlme_result)
    if inp is None:
        return None

    model = get_model(model_key)
    names = list(model.params)
    x_hat = np.asarray(inp["x_hat"], dtype=float)
    ofv_fn = inp["ofv_fn"]
    idx = {p: k for k, p in enumerate(names)}          # thetas lead the vector

    # Warm-start cache, per profiled parameter. Consecutive profile points sit
    # close together, so re-optimizing from the PREVIOUS point's solution
    # converges in a handful of iterations instead of restarting from the
    # global estimate every time. Without this a single parameter costs
    # thousands of population Laplace passes and is unusable in practice.
    warm: dict[str, np.ndarray] = {}

    def profile_ofv_fn(param: str, value: float) -> float:
        """Fix `param` at `value` and re-optimize every other parameter."""
        j = idx.get(param)
        if j is None or value <= 0.0:
            return float(_BIG)
        free = [k for k in range(x_hat.size) if k != j]
        fixed_log = math.log(max(float(value), _EPS))

        def obj(free_vals: np.ndarray) -> float:
            xv = x_hat.copy()
            xv[free] = free_vals
            xv[j] = fixed_log
            return float(ofv_fn(xv))

        if not free:
            return obj(np.array([]))
        start = warm.get(param)
        if start is None or start.size != len(free):
            start = x_hat[free]
        res = minimize(obj, start, method="Powell",
                       options={"maxiter": _PROFILE_INNER_ITER,
                                "maxfev": _PROFILE_INNER_FEV,
                                "xtol": 1e-2, "ftol": 1e-2})
        warm[param] = np.asarray(res.x, dtype=float)
        return float(res.fun)

    theta = {p: float(v) for p, v in (nlme_result.get("theta") or {}).items()
             if p in idx}
    # Asymptotic SE on the natural scale is the natural first step: it is
    # roughly where the crossing sits when the likelihood is near-quadratic.
    rse = nlme_result.get("theta_rse_pct") or {}
    step = {p: (abs(theta[p]) * float(rse[p]) / 100.0)
            for p in theta if rse.get(p) and math.isfinite(float(rse[p]))}

    return {"profile_ofv_fn": profile_ofv_fn, "estimates": theta,
            "ofv_hat": float(inp["ofv_hat"]), "initial_step": step,
            "near_singular": inp.get("near_singular", False),
            "fit_condition_number": inp.get("fit_condition_number")}
