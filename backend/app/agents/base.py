"""Agent base.

An agent owns a system prompt, a set of bound tools (by ownership in the
registry), and a bounded tool-use loop. It mutates state ONLY by executing
tools through the registry (which enforces audit + write-access). Agents never
call each other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.audit import AuditChain
from app.core.finding_ids import finding_kind
from app.core.pharmstate import PharmState
from app.tools.base import ExpensiveToolError, ToolContext, ToolRegistry

MAX_TOOL_STEPS = 6


def _n(items) -> int:
    return len(items or [])


def idle_hint(agent: str, state: PharmState) -> str:
    """One line for a turn that selected no tool: what IS loaded and what the
    user can ask for next. Never raw rows — counts and ids only.

    A bare "no action taken" read as "the app is broken" to a desktop user who
    typed "load dataset" with nothing uploaded; the hint has to name the
    prerequisite (upload / path) or the fact that the result already exists.
    """
    loaded = state.dataset_id is not None
    n_subj = (state.dataset_metadata or {}).get("n_subjects")
    subj = f", {n_subj} subjects" if n_subj is not None else ""
    n_nca = _n(state.nca_parameters)
    where = (f"dataset {state.dataset_id}{subj} is loaded" if loaded
             else "no dataset is loaded — use 'Click to upload' (or drag a CSV in), "
                  "or tell me the CSV path")
    if agent == "statistician":
        return (f"[statistician] {where}. Ask which test fits your data, or for a Bucher indirect "
                "comparison give both direct effects (A vs B and C vs B: estimate with SE, or CI and its "
                "level) and the scale (HR, OR, RR, MD, RD or SMD).")
    if agent == "data_manager":
        if not loaded:
            return f"[data_manager] {where}."
        profiled = state.data_quality is not None
        return (f"[data_manager] {where}"
                f"{' and profiled' if profiled else ''}. Next: compute NCA, fit a PK model, "
                "run the QC review, or ask for the report.")
    if agent == "nca":
        if not loaded:
            return f"[nca] {where}; NCA needs a dataset first."
        if n_nca:
            return (f"[nca] NCA already computed for {n_nca} subjects on {state.dataset_id}. "
                    "Next: QC review, bioequivalence, dose proportionality, or the report.")
        return f"[nca] {where}; ask me to compute NCA."
    if agent == "er_dose":
        if not loaded:
            return (f"[er_dose] {where}; exposure-response needs per-subject data (one row per subject "
                    "with a response, or time and event) and an exposure column or NCA results.")
        fits = sorted((state.er_results or {}).get("fits") or {})
        have_fits = (f"stored fits: {', '.join(fits)}" if fits else "no exposure-response fit yet")
        return (f"[er_dose] {where}; {have_fits}. Name the exposure column and the response column "
                "(or time and event columns), e.g. 'logistic E-R of AE vs exposure'. With fits labelled "
                "'efficacy' and 'toxicity' I can select a dose once you declare the utility weight w.")
    done = []
    if n_nca:
        done.append(f"NCA {n_nca} subjects")
    if state.pk_model_results:
        done.append("PK model fit")
    if (state.nlme_results or {}).get("status") == "ok":
        done.append("NLME fit")
    if state.qc_verdict:
        done.append(f"QC {state.qc_verdict}")
    if state.report_path:
        done.append("report")
    have = f" Done so far: {', '.join(done)}." if done else ""
    return f"[{agent}] nothing to run for that request — {where}.{have}"


@dataclass
class AgentResult:
    state: PharmState
    messages: list[str] = field(default_factory=list)   # human-facing log
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # A proposable expensive tool the LLM selected: {tool, agent, args}. Not
    # executed here; the orchestrator records it as state.pending_tool.
    proposal: dict[str, Any] | None = None


@dataclass
class Agent:
    name: str
    system_prompt: str

    def _state_summary(self, state: PharmState,
                       observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Compact, privacy-safe view of state for LLM tool selection.

        Exposes aggregate diagnostic summaries (never raw patient rows) and the
        current turn's execution observations so the agent can reason about
        scientific state, convergence, and quality findings.
        """
        meta = state.dataset_metadata or {}
        nca_params = state.nca_parameters or []
        pm = state.pk_model_results or {}
        nl = state.nlme_results or {}
        rev = state.review_results or {}

        nca_diag = None
        if nca_params:
            nca_diag = {
                "n_subjects": len(nca_params),
                "missing_lambda_z": sum(1 for p in nca_params if p.get("lambda_z") is None),
                "high_extrap_gt20pct": sum(
                    1 for p in nca_params
                    if (p.get("pct_AUC_extrap") or 0.0) > 20.0
                ),
            }

        pk_diag = None
        if pm.get("status") == "ok":
            best = (pm.get("best") or {}) if pm.get("mode") == "compare" else pm
            ranking = pm.get("ranking") or []
            aic_delta = None
            if len(ranking) >= 2 and ranking[0].get("aic") is not None and ranking[1].get("aic") is not None:
                aic_delta = round(abs(float(ranking[1]["aic"]) - float(ranking[0]["aic"])), 3)
            pk_diag = {
                "mode": pm.get("mode", "single"),
                "best_model": best.get("model_key") or best.get("label"),
                "n_subjects": best.get("n_subjects"),
                "n_converged": best.get("n_converged"),
                "aic_delta_runner_up": aic_delta,
            }

        nlme_diag = None
        if nl.get("status") == "ok":
            sh = list((nl.get("shrinkage_pct") or {}).values())
            rse = list((nl.get("theta_rse_pct") or {}).values())
            nlme_diag = {
                "converged": nl.get("converged"),
                "condition_number": nl.get("condition_number"),
                "max_shrinkage_pct": max((v for v in sh if v is not None), default=None),
                "max_theta_rse_pct": max((v for v in rse if v is not None), default=None),
            }

        rev_diag = None
        if rev:
            rev_diag = {
                "status": rev.get("status"),
                "goal_met": rev.get("goal_met"),
                "counts": rev.get("counts"),
                # Per-subject finding ids end in the subject id (recompute-cmax-1771) and this
                # summary reaches the chat and planner LLMs: keep the kind, drop the subject.
                "blocking_ids": sorted({
                    finding_kind(str(f.get("id"))) for f in (rev.get("findings") or [])
                    if not f.get("resolved") and f.get("severity") in ("CRITICAL", "HIGH")
                }),
            }

        return {
            "dataset_id": state.dataset_id,
            "dataset_loaded": state.dataset_id is not None,
            "dataset_profiled": state.data_quality is not None,
            "spaghetti_plotted": state.spaghetti_data is not None,
            "has_metadata": state.dataset_metadata is not None,
            "n_subjects": meta.get("n_subjects"),
            "columns": meta.get("columns"),
            "detected_roles": meta.get("detected_roles"),
            "data_quality": (state.data_quality or {}).get("quality_flags"),
            "nca_parameters": "present" if state.nca_parameters else None,
            "nca_diagnostics": nca_diag,
            "be_results": "present" if state.be_results else None,
            "dose_prop_results": "present" if state.dose_prop_results else None,
            "compartmental_results": "present" if state.compartmental_results else None,
            "poppk_results": "present" if state.poppk_results else None,
            "pk_model_results": "present" if state.pk_model_results else None,
            "pk_model_diagnostics": pk_diag,
            "nlme_results": "present" if nl.get("status") == "ok" else None,
            "nlme_diagnostics": nlme_diag,
            "simest_results": "present" if state.simest_results else None,
            "stats_advice": "present" if state.stats_advice else None,
            "er_fits": sorted((state.er_results or {}).get("fits") or {}) or None,
            "dose_selection": "present" if state.dose_selection_results else None,
            "last_calculation": (state.clinpharm_results or {}).get("tool"),
            "pending_tool": (state.pending_tool or {}).get("tool"),
            "qc_verdict": state.qc_verdict,
            "lambda_z_review_status": (state.lambda_z_review or {}).get("status"),
            "poppk_qc_verdict": state.poppk_qc_verdict,
            "review_diagnostics": rev_diag,
            "report_path": state.report_path,
            "memo_path": state.memo_path,
            "turn_observations": list(observations) if observations else None,
        }

    def run_turn(self, *, state: PharmState, message: str, llm, registry: ToolRegistry,
                 ctx: ToolContext, audit: AuditChain, clock, actor: str = "",
                 prior_observations: list[dict[str, Any]] | None = None) -> AgentResult:
        tools = registry.for_agent(self.name)
        result = AgentResult(state=state)
        observations: list[dict[str, Any]] = list(prior_observations or [])
        last_call: tuple[str, str] | None = None
        for _ in range(MAX_TOOL_STEPS):
            choice = llm.select_tool(
                self.name, message, tools,
                self._state_summary(result.state, observations),
            )
            if not choice:
                break
            # Every tool is deterministic, so an identical repeat cannot
            # produce anything new: stop instead of burning MAX_TOOL_STEPS.
            this_call = (choice["name"], json.dumps(choice.get("input") or {}, sort_keys=True,
                                                    default=str))
            if this_call == last_call:
                result.messages.append(
                    f"{choice['name']} already ran with these arguments this turn; "
                    "stopping (the result is above).")
                result.tool_calls.append({"tool": choice["name"], "stopped": "repeat"})
                break
            last_call = this_call
            try:
                # allow_expensive is NOT passed: a chat turn runs synchronously, so a
                # long-running fit here would block the worker AND bypass the job
                # queue. The registry refuses it (ExpensiveToolError) — enforcing
                # there, not on `tools`, also covers a tool the LLM names that is not
                # in this agent's own list.
                call_args = dict(choice.get("input") or {})
                new_state, tool_res = registry.execute(
                    choice["name"], state=result.state, ctx=ctx,
                    args=call_args, audit=audit, timestamp=clock(),
                    actor=actor, caller_agent=self.name,
                )
            except ExpensiveToolError as exc:
                tool = registry.get(choice["name"])
                if tool.proposable and tool.agent == self.name:
                    # Reachable from the agent, never run by it: the refused call
                    # is handed back as a PROPOSAL. The human's approval is the
                    # tool's `confirm` (an LLM-supplied confirm is dropped), and
                    # the orchestrator submits the approved call as a bounded
                    # background job. Nothing is computed on this turn.
                    args = {k: v for k, v in (choice.get("input") or {}).items()
                            if k != "confirm"}
                    result.proposal = {"tool": tool.name, "agent": self.name, "args": args}
                    result.messages.append(
                        f"Proposed {tool.name} — a long-running fit (minutes to tens of "
                        "minutes). Nothing has been computed. Approve it to run as a "
                        "background job, or reject it.")
                    result.tool_calls.append({"tool": tool.name, "proposed": "expensive",
                                              "args": args})
                    break
                result.messages.append(f"{exc} Ask for it from that panel instead.")
                result.tool_calls.append({"tool": choice["name"], "skipped": "expensive"})
                break
            except Exception as exc:
                # A tool failed (e.g. NCA requested before a dataset is loaded).
                # Surface it as a chat message rather than 500-ing the whole turn.
                result.messages.append(f"Could not run {choice['name']}: {exc}")
                result.tool_calls.append({"tool": choice["name"], "error": str(exc)})
                observations.append({"agent": self.name, "tool": choice["name"],
                                     "status": "error", "error": str(exc)})
                break
            result.state = new_state
            result.messages.append(tool_res.summary)
            result.tool_calls.append({"tool": choice["name"], "agent": self.name,
                                      "args": call_args, "summary": tool_res.summary})
            observations.append({"agent": self.name, "tool": choice["name"],
                                 "status": "ok", "summary": tool_res.summary})
        if not result.messages:
            result.messages.append(idle_hint(self.name, result.state))
        return result
