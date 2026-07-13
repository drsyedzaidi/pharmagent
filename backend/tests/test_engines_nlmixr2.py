"""Live nlmixr2 adapter test — SKIPPED automatically when R/nlmixr2 is absent.

This is the end-to-end proof that an external R engine plugs into the same
scoring path. It runs a real nlmixr2 fit (~20s incl. model compilation), so it is
isolated from the fast suite and self-skips on any machine without the toolchain.
"""
import numpy as np
import pytest

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.engines import CandidateSpec, Nlmixr2Adapter

_AVAILABLE = Nlmixr2Adapter().available()
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="R/nlmixr2 not installed")


def _subjects(n=8):
    rng = np.random.default_rng(7)
    m = get_model("oral_1cmt")
    base = dict(m.defaults)
    times = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 24], dtype=float)
    out = []
    for sid in range(1, n + 1):
        p = {k: base[k] * float(np.exp(rng.normal(0, 0.22))) for k in base}
        cp = simulate(m, p, [{"time": 0.0, "amt": 100.0}], times)["cp"]
        out.append({"subject": f"S{sid}", "doses": [{"time": 0.0, "amt": 100.0}],
                    "obs_t": times.tolist(),
                    "obs_c": (cp * np.exp(rng.normal(0, 0.06, size=cp.size))).tolist(),
                    "wt": 70.0})
    return out


def test_nlmixr2_recovers_params_and_scores():
    r = Nlmixr2Adapter().fit(CandidateSpec("oral_1cmt", iiv_params=["CL", "V"]), _subjects())
    assert r.status == "ok", r.message
    assert r.engine == "nlmixr2_focei"
    # truth CL=5, V=50 -> recovery within 25%
    assert 3.75 < r.params["CL"] < 6.25
    assert 37.5 < r.params["V"] < 62.5
    # scored on our footing
    assert r.pred_rmse is not None and r.pred_rmse < 0.2
    assert r.vpc_coverage90 is not None
    # native OFV present but flagged within-engine only (not derived to AIC/BIC)
    assert r.ofv is not None and r.aic is None


def test_nlmixr2_unsupported_model_fails_cleanly():
    r = Nlmixr2Adapter().fit(CandidateSpec("oral_2cmt"), _subjects(3))
    assert r.status == "failed" and "supports" in r.message
