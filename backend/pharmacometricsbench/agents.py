"""Reference agents.

An *agent* here is any callable ``(Task) -> dict[str, value]`` returning a
prediction for each target name. Three references ship with v0:

* ``oracle_agent`` — calls the validated compute tools. It reproduces ground
  truth exactly and therefore scores 1.0. It is both the top-of-leaderboard
  reference and the harness self-test (if it ever drops below 1.0, a task is
  malformed).

* ``naive_agent`` — a plausible-but-wrong "eyeballing" agent that free-hands
  numbers without the tools (linear-trapezoid AUC, arithmetic-mean ratio, a
  two-point slope). It exists to prove the benchmark discriminates good process
  from bad — it should score well below 1.0.

* ``llm_agent`` (in ``llm.py``) — routes the task through a prompt → text →
  parse pipeline. Keyless by default via ``MockLLM``; drop in a real client to
  score an actual model.

The prediction bodies are exposed as ``oracle_predict`` / ``naive_predict`` so a
keyless MockLLM can reuse the exact same math without duplicating it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.compute.bioequivalence import be_one_parameter
from app.compute.compartmental import fit_one_subject
from app.compute.dose_proportionality import power_model
from app.compute.nca import Profile, nca_subject

from .spec import Task


# ── Oracle: uses the validated tools (should score 1.0) ────────────────────
def oracle_predict(category: str, d: dict[str, Any]) -> dict[str, Any]:
    if category == "nca":
        r = nca_subject(Profile("S", np.array(d["time"]), np.array(d["conc"]), d["dose"]))
        return {"Cmax": r["Cmax"], "AUC_inf": r["AUC_inf"], "t_half": r["t_half"]}
    if category == "be":
        r = be_one_parameter(d["test"], d["ref"], paired=True)
        return {"gmr_pct": r["gmr_pct"], "ci_lower_pct": r["ci_lower_pct"],
                "ci_upper_pct": r["ci_upper_pct"], "within_limits": r["within_limits"]}
    if category == "dp":
        r = power_model(d["doses"], d["values"])
        return {"slope": r["slope"], "proportional": r["proportional"]}
    if category == "compartmental":
        r = fit_one_subject(d["time"], d["conc"], d["dose"], model="1cmt")
        p = r.get("params", {})
        return {"CL": p.get("CL"), "V": p.get("V"), "ka": p.get("ka")}
    if category == "exposure":
        from .generators import ss_exposure  # shares the ground-truth simulator
        e = ss_exposure(d["model"], d["params"], d["dose"], d["tau"], d["n_doses"])
        return {"Cmax_ss": e["Cmax_ss"], "AUC_tau": e["AUC_tau"]}
    return {}


# ── Naive: free-hands the numbers without tools (should score < 1.0) ───────
def naive_predict(category: str, d: dict[str, Any]) -> dict[str, Any]:
    if category == "nca":
        t = np.array(d["time"], dtype=float)
        c = np.array(d["conc"], dtype=float)
        cmax = float(c.max())
        auc_linear = float(np.trapezoid(c, t))          # linear trapezoid, no tail extrap
        # crude t1/2 from the last two points
        if c[-1] > 0 and c[-2] > c[-1]:
            k = math.log(c[-2] / c[-1]) / (t[-1] - t[-2])
            thalf = math.log(2) / k
        else:
            thalf = float("nan")
        return {"Cmax": cmax, "AUC_inf": auc_linear, "t_half": thalf}
    if category == "be":
        test = np.array(d["test"], dtype=float)
        ref = np.array(d["ref"], dtype=float)
        gmr = float((test / ref).mean() * 100.0)        # arithmetic mean ratio (wrong)
        return {"gmr_pct": gmr, "ci_lower_pct": gmr * 0.9, "ci_upper_pct": gmr * 1.1,
                "within_limits": 80.0 <= gmr <= 125.0}
    if category == "dp":
        doses = np.array(d["doses"], dtype=float)
        vals = np.array(d["values"], dtype=float)
        # naive slope from the two extreme doses only
        slope = float((math.log(vals.max()) - math.log(vals.min())) /
                      (math.log(doses.max()) - math.log(doses.min())))
        return {"slope": slope, "proportional": abs(slope - 1.0) < 0.05}
    if category == "compartmental":
        # guesses CL from dose/AUC_linear, ignores absorption entirely
        t = np.array(d["time"], dtype=float)
        c = np.array(d["conc"], dtype=float)
        auc = float(np.trapezoid(c, t))
        cl = d["dose"] / auc if auc > 0 else float("nan")
        return {"CL": cl, "V": cl * 5.0, "ka": 1.0}
    if category == "exposure":
        # closed-form without a simulator: AUC_tau at steady state = Dose/CL (exact
        # for a linear model), but Cmax_ss is crudely taken as the average
        # concentration (ignores the peak-to-trough swing), so it under-predicts.
        cl = d["params"]["CL"]
        auc_tau = d["dose"] / cl
        cmax = auc_tau / d["tau"]
        return {"Cmax_ss": cmax, "AUC_tau": auc_tau}
    return {}


def oracle_agent(task: Task) -> dict:
    return oracle_predict(task.category, task.dataset)


def naive_agent(task: Task) -> dict:
    return naive_predict(task.category, task.dataset)


def _build_agents() -> dict:
    agents = {"oracle": oracle_agent, "naive": naive_agent}
    # llm imports from this module, so wire it lazily to avoid a circular import.
    from .llm import llm_agent, make_model_agent
    agents["llm"] = llm_agent
    # Permanent, model-pinned reference agents (the figure's rows). Require a key
    # when run; keyless import is fine because make_model_agent builds lazily.
    agents["llm-opus"] = make_model_agent("claude-opus-4-8")
    agents["llm-haiku"] = make_model_agent("claude-haiku-4-5")
    # Tool-USING agents: the model calls the validated compute tools by
    # function-calling (the direct test of the tool-grounding thesis).
    from .tool_agent import make_tool_agent
    agents["llm-opus-tools"] = make_tool_agent("claude-opus-4-8")
    agents["llm-haiku-tools"] = make_tool_agent("claude-haiku-4-5")
    return agents


AGENTS = _build_agents()
