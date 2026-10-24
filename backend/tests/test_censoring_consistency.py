"""Censoring (CENS / BLQ) handling is explicit and consistent across every
`_build_subjects` caller (PMxbench gap G5).

No real NLME/SCM fit runs here: the estimators are monkeypatched with fakes
that wrap the subjects in the real `nlme._Subject`, so `n_blq` is exactly what
a real fit would count on the subjects the tool hands over.
"""
import numpy as np
import pandas as pd

import app.compute.nlme as nlme
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.pkmodel_tools import run_nlme, run_scm

MODEL_KEY = "oral_1cmt"
LLOQ = 0.05
N_POSTDOSE_CENS = 2


def _dataset() -> pd.DataFrame:
    """NONMEM convention: CENS=1 rows carry the LLOQ in DV. Two post-dose
    censored records plus one predose censored record (t=0, never fitted)."""
    rows = []
    for sid in range(1, 6):
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "CENS": 0,
                     "AGE": 30 + 5 * sid})
        if sid == 3:
            rows.append({"ID": sid, "TIME": 0.0, "DV": LLOQ, "AMT": np.nan, "CENS": 1,
                         "AGE": 30 + 5 * sid})
        for t, c in zip([0.5, 1, 2, 4, 8, 12, 24], [0.8, 1.4, 1.2, 0.9, 0.5, 0.2, 0.1]):
            dv, cens = c * (1 + 0.03 * sid), 0
            if sid in (1, 2) and t == 24:
                dv, cens = LLOQ, 1
            rows.append({"ID": sid, "TIME": t, "DV": dv, "AMT": np.nan, "CENS": cens,
                         "AGE": 30 + 5 * sid})
    return pd.DataFrame(rows)


ROLES = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT", "CENS": "CENS"}


def _loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": ROLES})
    return state, ctx


def _n_blq(subjects: list[dict]) -> int:
    return int(sum(int(s.blq.sum()) for s in nlme._prepare_subjects(subjects)))


def test_run_scm_uses_m3_like_the_base_nlme_fit(monkeypatch):
    seen: dict = {}

    def fake_population_fit(model_key, subjects, **kw):
        return {"method": "FOCE-I", "label": "1-cmt oral", "ofv": 1.0,
                "n_subjects": len(subjects), "iiv_params": ["CL", "V"],
                "converged": True, "theta": {}, "omega_cv_pct": {}, "sigma": {},
                "n_blq": _n_blq(subjects)}

    def fake_scm(model_key, subjects, **kw):
        seen["subjects"] = subjects
        return {"status": "ok", "model_key": model_key, "label": "1-cmt oral",
                "base_ofv": 1.0, "final_ofv": 1.0, "selected": [], "steps": [],
                "n_candidates": 1, "final": {"n_blq": _n_blq(subjects)}}

    monkeypatch.setattr(nlme, "population_fit", fake_population_fit)
    monkeypatch.setattr(nlme, "scm", fake_scm)
    state, ctx = _loaded()
    base = run_nlme(state, ctx, {"model_key": MODEL_KEY}).writes["nlme_results"]
    res = run_scm(state, ctx, {"model_key": MODEL_KEY,
                               "candidates": [{"param": "CL", "covariate": "AGE"}]})
    final = res.writes["scm_results"]["final"]
    assert base["n_blq"] == N_POSTDOSE_CENS
    assert final["n_blq"] == base["n_blq"]
    # the LLOQ is carried as the censoring limit, not as a quantified value
    assert {s.get("lloq") for s in seen["subjects"] if s.get("obs_blq")
            and any(s["obs_blq"])} == {LLOQ}
