"""NONMEM / mrgsolve control-stream export from a fitted NLME result."""
from __future__ import annotations

from app.compute.nmexport import build_mrgsolve, build_nonmem


def _fit(model_key="oral_1cmt", theta=None, iiv=None, cov=None):
    return {
        "model_key": model_key, "label": model_key,
        "theta": theta or {"CL": 5.0, "V": 50.0, "KA": 1.0},
        "omega_cv_pct": {"CL": 30.0, "V": 20.0},
        "sigma": {"prop": 0.1, "add": None},
        "iiv_params": iiv or ["CL", "V"], "error_model": "proportional",
        "covariate_effects": cov or [],
    }


def test_nonmem_oral_1cmt_advan2_trans2():
    ctl = build_nonmem(_fit())
    assert "$SUBROUTINE ADVAN2 TRANS2" in ctl
    assert "TVCL = THETA(1)" in ctl and "CL = TVCL * EXP(ETA(1))" in ctl
    assert "S2 = V" in ctl
    assert "$THETA" in ctl and "$OMEGA" in ctl and "$SIGMA" in ctl
    assert "$ESTIMATION METHOD=1 INTERACTION" in ctl
    # proportional sigma variance = 0.1**2
    assert "0.01" in ctl


def test_nonmem_covariate_written_as_power_theta():
    ctl = build_nonmem(_fit(cov=[{"param": "CL", "covariate": "CRCL",
                                  "kind": "power", "center": 100.0, "coefficient": 0.75}]))
    assert "(CRCL/100)**THETA(4)" in ctl
    assert "CRCL" in ctl.splitlines()[1]      # listed in $INPUT


def test_nonmem_iv_2cmt_advan3_trans4():
    ctl = build_nonmem(_fit("iv_2cmt", theta={"CL": 5.0, "VC": 30.0, "Q": 10.0, "VP": 80.0},
                            iiv=["CL"]))
    assert "$SUBROUTINE ADVAN3 TRANS4" in ctl
    assert "V1" in ctl and "V2" in ctl and "S1 = V1" in ctl


def test_mrgsolve_oral_1cmt_has_ode_and_blocks():
    cpp = build_mrgsolve(_fit())
    assert "$ODE" in cpp and "dxdt_DEPOT = -KA*DEPOT" in cpp
    assert "$PARAM" in cpp and "$OMEGA" in cpp and "$SIGMA" in cpp
    assert "double CL = TVCL" in cpp and "exp(ECL)" in cpp


def test_mrgsolve_lag_on_dosing_compartment():
    """oral_1cmt_lag doses into DEPOT, so the absorption lag must be ALAG_DEPOT
    (ALAG_CENT would attach it to a compartment that receives no dose)."""
    cpp = build_mrgsolve(_fit("oral_1cmt_lag",
                              theta={"CL": 5.0, "V": 50.0, "KA": 1.0, "ALAG": 0.3}))
    assert "ALAG_DEPOT = ALAG" in cpp
    assert "ALAG_CENT" not in cpp


def test_nonmem_lag_uses_alag1_on_depot():
    """NONMEM ADVAN2 depot is compartment 1 -> ALAG1 (unchanged, correct)."""
    ctl = build_nonmem(_fit("oral_1cmt_lag",
                            theta={"CL": 5.0, "V": 50.0, "KA": 1.0, "ALAG": 0.3}))
    assert "ALAG1 = THETA(4)" in ctl


def test_unsupported_model_returns_none():
    assert build_nonmem(_fit("oral_1cmt_transit")) is None
    assert build_mrgsolve(_fit("iv_1cmt_mm")) is None


# ── correlated (block) Omega export ─────────────────────────────────────────
# A block model exported as a diagonal Omega is a silent misrepresentation: the
# control stream would look valid and fit, just without the correlation the
# model was built to carry.

_BLOCK_NL = {
    "model_key": "oral_1cmt", "label": "1-cmt oral",
    "iiv_params": ["CL", "V", "KA"],
    "theta": {"CL": 3.0, "V": 60.0, "KA": 1.5},
    "omega_cv_pct": {"CL": 31.0, "V": 26.0, "KA": 40.0},
    "sigma": {"prop": 0.2, "add": 2.0}, "error_model": "combined",
    # NON-contiguous on purpose: CL and KA, with V between them.
    "omega_block": ["CL", "KA"],
    "omega_matrix": [[0.093, 0.0, 0.042],
                     [0.0, 0.065, 0.0],
                     [0.042, 0.0, 0.15]],
}


def _lines(text):
    return [ln.rstrip() for ln in text.splitlines()]


def test_nonmem_emits_omega_block_with_the_lower_triangle():
    out = _lines(build_nonmem(_BLOCK_NL))
    assert "$OMEGA BLOCK(2)" in out
    i = out.index("$OMEGA BLOCK(2)")
    assert out[i + 1].split(";")[0].split() == ["0.093"]
    assert out[i + 2].split(";")[0].split() == ["0.042", "0.15"]


def test_nonmem_renumbers_etas_so_the_block_is_contiguous():
    """$OMEGA BLOCK(n) describes n CONSECUTIVE etas, so a block on CL and KA
    of [CL, V, KA] is only writable if the etas are re-ordered. $PK must agree
    with that re-ordering or the covariance lands on the wrong pair."""
    out = _lines(build_nonmem(_BLOCK_NL))
    pk = {ln.split("=")[0].strip(): ln for ln in out if "EXP(ETA(" in ln}
    assert "ETA(1)" in pk["CL"]
    assert "ETA(2)" in pk["KA"]      # block member, moved next to CL
    assert "ETA(3)" in pk["V"]       # non-block, pushed after


def test_nonmem_writes_remaining_etas_as_a_separate_diagonal():
    out = _lines(build_nonmem(_BLOCK_NL))
    assert out.count("$OMEGA") == 1                  # the plain one for V
    tail = out[out.index("$OMEGA") + 1]
    assert tail.split(";")[0].strip() == "0.065"     # V, from omega_matrix


def test_mrgsolve_emits_a_block_omega():
    out = _lines(build_mrgsolve(_BLOCK_NL))
    hdr = next(ln for ln in out if ln.startswith("$OMEGA @block"))
    assert "ECL" in hdr and "EKA" in hdr
    tri = out[out.index(hdr) + 1].split()
    assert tri == ["0.093", "0.042", "0.15"]         # lower triangle, row-major


def test_block_diagonal_entries_come_from_the_matrix_not_a_rounded_cv():
    """Mixing exact covariances with %CV-derived variances in one Omega would
    make the exported matrix inconsistent with the fitted one."""
    out = build_nonmem(_BLOCK_NL)
    assert "0.065" in out                             # exact V variance
    assert "0.0654131" not in out                     # the %CV-derived value


def test_a_single_member_block_falls_back_to_diagonal():
    nl = dict(_BLOCK_NL, omega_block=["CL"])
    out = _lines(build_nonmem(nl))
    assert not any(ln.startswith("$OMEGA BLOCK") for ln in out)


def test_block_without_a_matrix_falls_back_to_diagonal():
    """omega_block alone cannot be written -- the covariances live in the
    matrix, and guessing them would be fabrication."""
    nl = {k: v for k, v in _BLOCK_NL.items() if k != "omega_matrix"}
    out = _lines(build_nonmem(nl))
    assert not any(ln.startswith("$OMEGA BLOCK") for ln in out)


def test_diagonal_export_is_unchanged_by_the_block_feature():
    nl = {k: v for k, v in _BLOCK_NL.items()
          if k not in ("omega_block", "omega_matrix")}
    out = _lines(build_nonmem(nl))
    i = out.index("$OMEGA")
    assert [ln.split(";")[0].strip() for ln in out[i + 1:i + 4]] == \
        ["0.0917584", "0.0654131", "0.14842"]        # all %CV-derived, in order
    pk = {ln.split("=")[0].strip(): ln for ln in out if "EXP(ETA(" in ln}
    assert "ETA(1)" in pk["CL"] and "ETA(2)" in pk["V"] and "ETA(3)" in pk["KA"]
