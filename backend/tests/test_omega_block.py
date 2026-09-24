"""P0: the correlated-Omega (block) layout, encode/decode, and inertness.

All analytic -- no fit is run. These tests exist to make the two silent-failure
modes loud:
  * a scale/transpose error in the Cholesky encode/decode that a round-trip
    alone cannot see (round-trips are blind to inverse-consistent errors), and
  * per-parameter index mislabelling when the block is NOT the trailing subset
    of iiv_params.
"""
import math

import numpy as np
import pytest

from app.compute.nlme import (
    _block_corr,
    _chol_from_seg,
    _omega_full_from_seg,
    _omega_layout,
    _omega_matrix,
    _pack,
    _PopSpec,
    _resolve_block,
    _seg_from_chol,
    _seg_from_omega_full,
    _unpack,
)
from app.compute.pk_models import get_model

MODEL = get_model("oral_1cmt")
THETA = {"CL": 3.0, "V": 60.0, "KA": 1.5}
# The published 2-cmt oral reference block (CL, Vc): r = 0.042/sqrt(.093*.065).
REF_OMEGA = np.array([[0.093, 0.042], [0.042, 0.065]])
REF_R = 0.042 / math.sqrt(0.093 * 0.065)


def _spec(iiv, block=None):
    return _PopSpec(MODEL, iiv, "proportional", omega_block=block)


# ── layout ──────────────────────────────────────────────────────────────────

def test_diagonal_layout_is_the_identity_map():
    """No block => n_omega_par == n_omega and slot j == j, so every existing
    literal `x[i + k] for k, p in enumerate(iiv_params)` stays correct."""
    s = _spec(["CL", "V"])
    assert s.omega_block is None and s.block_idx is None
    assert s.n_omega_par == s.n_omega == 2
    assert s.omega_slot == (0, 1)


def test_non_trailing_block_layout_maps_slots_correctly():
    """The hazard case: block on [CL, KA] of [CL, V, KA]. V's log-variance is
    the ONLY marginal slot; naive enumerate() indexing would read it as CL's."""
    s = _spec(["CL", "V", "KA"], ["KA", "CL"])
    assert s.block_idx == (0, 2), "block indices must be ascending"
    assert s.omega_block == ("CL", "KA")
    assert s.n_omega_par == 4          # V log-var + 2 log-diag + 1 off-diagonal
    assert s.omega_slot == (-1, 0, -1)


def test_block_member_order_is_normalised_ascending():
    """Declared order must not matter; the decode only reconstructs a Cholesky
    factor under ascending index order."""
    a = _spec(["CL", "V", "KA"], ["KA", "CL"])
    b = _spec(["CL", "V", "KA"], ["CL", "KA"])
    assert a.block_idx == b.block_idx == (0, 2)


def test_block_of_fewer_than_two_members_stays_diagonal():
    s = _spec(["CL", "V"], ["CL"])
    assert s.omega_block is None and s.n_omega_par == 2


def test_unknown_block_member_raises():
    with pytest.raises(ValueError, match="not an IIV parameter"):
        _spec(["CL", "V"], ["CL", "NOPE"])


@pytest.mark.parametrize("n,block,expect_par", [
    (2, None, 2), (3, None, 3),
    (2, (0, 1), 3),        # 2 log-diag + 1 off-diagonal
    (3, (0, 1), 4),        # 1 marginal + 2 log-diag + 1 off-diagonal
    (3, (0, 1, 2), 6),     # 3 log-diag + 3 off-diagonal
])
def test_parameter_count(n, block, expect_par):
    assert _omega_layout(n, block)[0] == expect_par


# ── encode / decode ─────────────────────────────────────────────────────────

def test_chol_segment_has_the_documented_literal_layout():
    """Absolute check, not a round-trip: a round-trip is blind to any error
    that is inverse-consistent between encode and decode (e.g. storing 2*lam)."""
    seg = np.array([math.log(2.0), math.log(3.0), 0.5])   # [log d0, log d1, lam10]
    L = _chol_from_seg(seg, 2)
    np.testing.assert_allclose(L, [[2.0, 0.0], [0.5, 3.0]], rtol=0, atol=1e-15)
    # ...and the covariance it implies, computed by hand:
    #   L L' = [[4, 1], [1, 0.25 + 9]]
    np.testing.assert_allclose(L @ L.T, [[4.0, 1.0], [1.0, 9.25]], rtol=0, atol=1e-14)


def test_seg_from_chol_inverts_chol_from_seg():
    seg = np.array([math.log(2.0), math.log(3.0), 0.5])
    np.testing.assert_allclose(_seg_from_chol(_chol_from_seg(seg, 2)), seg,
                               rtol=0, atol=1e-14)


def test_reference_omega_round_trips_and_recovers_its_correlation():
    s = _spec(["CL", "V"], ["CL", "V"])
    seg = _seg_from_omega_full(s, REF_OMEGA)
    back = _omega_full_from_seg(s, seg)
    np.testing.assert_allclose(back, REF_OMEGA, rtol=0, atol=1e-12)
    assert _block_corr(back)[0, 1] == pytest.approx(REF_R, abs=1e-9)


def test_neutral_init_reproduces_the_diagonal_exactly():
    """Seeding a block fit from a diagonal one must start at EXACTLY the
    diagonal model, so any OFV gain is attributable to the correlation."""
    s = _spec(["CL", "V"], ["CL", "V"])
    om = np.diag([0.093, 0.065])
    back = _omega_full_from_seg(s, _seg_from_omega_full(s, om))
    np.testing.assert_allclose(back, om, rtol=0, atol=1e-15)
    assert back[0, 1] == 0.0


def test_decoded_omega_is_positive_definite_for_arbitrary_parameters():
    """PD by construction: no parameter value can produce an invalid Omega."""
    s = _spec(["CL", "V", "KA"], ["CL", "V", "KA"])
    rng = np.random.default_rng(3)
    for _ in range(200):
        seg = rng.normal(0.0, 3.0, s.n_omega_par)      # wild, unconstrained
        om = _omega_full_from_seg(s, seg)
        assert np.all(np.linalg.eigvalsh(om) > 0.0)
        np.testing.assert_allclose(om, om.T, rtol=0, atol=1e-14)


def test_non_trailing_block_places_values_at_the_right_indices():
    s = _spec(["CL", "V", "KA"], ["CL", "KA"])
    om = np.diag([0.093, 0.050, 0.065])
    om[0, 2] = om[2, 0] = 0.042                        # CL-KA covariance
    back = _omega_full_from_seg(s, _seg_from_omega_full(s, om))
    np.testing.assert_allclose(back, om, rtol=0, atol=1e-12)
    assert back[0, 1] == 0.0 and back[1, 2] == 0.0     # V stays uncorrelated


# ── pack / unpack ───────────────────────────────────────────────────────────

def test_pack_requires_the_matrix_for_a_block_spec():
    """Marginal variances cannot encode off-diagonals; a silent diagonal
    fallback here would corrupt the point the Hessian is taken around."""
    s = _spec(["CL", "V"], ["CL", "V"])
    with pytest.raises(ValueError, match="omega_matrix is required"):
        _pack(s, THETA, np.array([]), {"CL": 0.093, "V": 0.065}, 0.2, 0.0)


def test_pack_unpack_recovers_marginals_and_matrix():
    s = _spec(["CL", "V"], ["CL", "V"])
    x = _pack(s, THETA, np.array([]), {"CL": 0.093, "V": 0.065}, 0.2, 0.0,
              omega_matrix=REF_OMEGA)
    assert x.size == s.n_theta + s.n_omega_par + 1
    theta, _cc, omega2, sp, _sa = _unpack(s, x)
    # unpack reports MARGINAL variances, so diagonal consumers keep working
    assert omega2["CL"] == pytest.approx(0.093, abs=1e-12)
    assert omega2["V"] == pytest.approx(0.065, abs=1e-12)
    assert theta["CL"] == pytest.approx(3.0, rel=1e-12)
    assert sp == pytest.approx(0.2, rel=1e-12)
    np.testing.assert_allclose(_omega_matrix(s, x), REF_OMEGA, rtol=0, atol=1e-12)


def test_omega_matrix_is_none_on_the_diagonal_path():
    """None, not diag(omega2): the diagonal hot path must keep its scalar
    arithmetic rather than being routed through matrix algebra."""
    s = _spec(["CL", "V"])
    x = _pack(s, THETA, np.array([]), {"CL": 0.09, "V": 0.05}, 0.2, 0.0)
    assert _omega_matrix(s, x) is None


def test_diagonal_pack_unpack_is_unchanged():
    """The inertness guarantee, stated as a test."""
    s = _spec(["CL", "V"])
    omega2 = {"CL": 0.09, "V": 0.05}
    x = _pack(s, THETA, np.array([]), omega2, 0.2, 0.0)
    expect = [math.log(THETA[p]) for p in s.param_names] + \
             [math.log(0.09), math.log(0.05), math.log(0.2)]
    np.testing.assert_allclose(x, expect, rtol=0, atol=0.0)


# ── correlation reporting ───────────────────────────────────────────────────

def test_block_corr_matches_the_hand_computed_reference():
    corr = _block_corr(REF_OMEGA)
    assert corr[0, 1] == pytest.approx(REF_R, abs=1e-12)
    assert corr[0, 0] == 1.0 and corr[1, 1] == 1.0
    np.testing.assert_allclose(corr, corr.T, rtol=0, atol=1e-15)


def test_block_corr_handles_a_zero_variance_without_dividing_by_zero():
    corr = _block_corr(np.array([[0.0, 0.0], [0.0, 0.09]]))
    assert np.all(np.isfinite(corr))
    assert corr[0, 1] == 0.0


def test_resolve_block_returns_none_when_unset():
    assert _resolve_block(["CL", "V"], None) is None
    assert _resolve_block(["CL", "V"], []) is None
