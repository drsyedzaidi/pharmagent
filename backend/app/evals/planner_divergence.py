"""Replicate-divergence harness for the goal planner. No fit is ever run.

Method (re-expressed from Pritchard-Bell et al., CPT Pharmacometrics Syst
Pharmacol 2026, doi:10.1002/psp4.70317): run the same agentic leg several times
on the same input and compare the runs decision by decision. Where replicates
agree, the rules steering the planner are adequate for that decision; where
they disagree, a rule is missing or the decision is a genuine judgment branch.
The harness only REPORTS this: turning a divergence into a policy or mikebrain
rule is an owner decision (docs/planner_divergence.md).

Each replicate runs the ``poppk_goal`` leg through the real
``Orchestrator.start_workflow`` in a fresh in-memory session. ``allow_expensive``
is never passed, so it keeps its default (False): the leg ends at its FIRST
gated proposal (app.core.policy gates every long-running fit and every
scientific model choice), and were the policy ever to admit an expensive tool
autonomously, ``ToolRegistry.execute`` still raises ExpensiveToolError and the
leg ends ``awaiting_job``. Every execution's ``allow_expensive`` flag is observed
at the registry, and each replicate's audit chain is scanned afterwards; either
showing an expensive execution raises HarnessGuaranteeError instead of a report.

    python -m app.evals.planner_divergence --dataset sample_data/oral_pk.csv \\
        --replicates 5 --provider local --model qwen3:8b --out divergence_out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.core import policy
from app.core.llm import LLM, LOCAL_REASONING_EFFORT, MockLLM, OpenAICompatLLM
from app.core.llm_config import DEFAULT_MODELS, build_llm, choice_from_settings
from app.core.orchestrator import Orchestrator, Session
from app.core.provenance import file_sha256
from app.core.store import SessionStore
from app.evals import divergence_report as dr
from app.tools.base import ExpensiveToolError, Tool, ToolRegistry
from app.tools.builtins import default_registry
from app.tools.data_tools import _safe_path
from app.workflows import get_workflow

WORKFLOW = "poppk_goal"
DEFAULT_REPLICATES = 5
MAX_REPLICATES = 50         # local models are slow: bound an accidental huge run
MAX_TIMEOUT_S = 3600.0      # per planner call; a 31B model under CPU load can need minutes
PROVIDERS = ("mock", "local", "configured")
HOSTED_PROVIDERS = frozenset({"openai", "anthropic"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
JSON_NAME = "planner_divergence.json"
MD_NAME = "planner_divergence.md"
PARTIAL_NAME = "replicates.jsonl"           # one line per finished replicate
FAILED_NAME = "planner_divergence.FAILED.txt"
DATASET_TOKEN = "<dataset>"                 # the dataset path never lands in the report
DETAIL_CHARS = 500

EXIT_OK, EXIT_SELF_TEST, EXIT_USAGE, EXIT_INCOMPLETE, EXIT_GUARANTEE, EXIT_POLICY = 0, 1, 2, 3, 4, 5

# The leg's end, from the orchestrator status and (for "stopped") the last planner
# action on the audit chain, whose shape is fixed by Orchestrator._advance_goal.
STATUS_STOP = {"awaiting_review": "gated_proposal", "awaiting_job": "expensive_refused",
               "complete": "planner_complete", "error": "harness_error"}
ACTION_STOP = {"plan_error": "planner_error", "tool_error": "tool_error", "stop": "budget",
               "plan": "silent_stop"}
END_BY_STOP = {"planner_complete": dr.END_STOP, "silent_stop": dr.END_SILENT,
               "planner_error": dr.END_PLANNER_ERROR, "harness_error": dr.END_HARNESS_ERROR}


class HarnessGuaranteeError(RuntimeError):
    """An expensive tool executed, or an execution was allowed to be expensive.
    ``record`` is the offending replicate's (redacted) record, when one exists."""

    def __init__(self, message: str, record: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class LlmSpec:
    """The planner LLM each replicate gets. ``seed`` is a base: replicate i samples
    with ``seed + i``, so the run is reproducible yet the replicates are independent
    draws (one shared seed would make every sampled replicate identical)."""
    provider: str
    model: str
    base_url: str | None = None
    temperature: float | None = None
    seed: int | None = None
    timeout_s: float | None = None      # per planner call; None keeps the client's own

    def sampling(self, index: int) -> dict[str, Any] | None:
        if self.provider != "local":
            return None
        return {"temperature": self.temperature,
                "seed": None if self.seed is None else self.seed + index}

    def public(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "seed_base": self.seed, "timeout_s": self.timeout_s}

    def build(self, index: int) -> LLM:
        if self.provider == "mock":
            return MockLLM()
        if self.provider == "local":
            s = self.sampling(index) or {}
            llm = OpenAICompatLLM(base_url=self.base_url or settings.DEFAULT_OLLAMA_URL,
                                  model=self.model, api_key=None,
                                  reasoning_effort=LOCAL_REASONING_EFFORT,
                                  temperature=s.get("temperature"), seed=s.get("seed"))
            if self.timeout_s is not None:
                llm.TIMEOUT_S = self.timeout_s      # this instance only; the class default is untouched
            return llm
        if self.provider == "configured":
            return build_llm(choice_from_settings())
        raise ValueError(f"unknown provider {self.provider!r}; one of {', '.join(PROVIDERS)}")


class _RecordingPlanner:
    """Transparent proxy over a replicate's LLM: returns exactly what the model
    returned (or re-raises), keeping each answer for the report."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.answers: list[dict[str, Any]] = []
        self.planner_s = 0.0

    def plan(self, goal: str, state_summary: dict[str, Any], observations: list[dict[str, Any]],
             history: list[dict[str, Any]], tools: list[Tool]) -> dict[str, Any] | None:
        t0 = time.perf_counter()
        try:
            raw = self._llm.plan(goal, state_summary, observations, history, tools)
        except Exception as exc:
            self.answers.append({"error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            self.planner_s += time.perf_counter() - t0
        self.answers.append(dict(raw) if isinstance(raw, dict)
                            else {} if raw is None else {"malformed": type(raw).__name__})
        return raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


class _WatchedRegistry:
    """Delegates to a fresh registry and records the ``allow_expensive`` flag every
    execution actually received at the choke point. A call that arrives with the
    flag set is recorded and REFUSED before it reaches the registry, so a harness
    regression is caught before a fit runs, not after."""

    def __init__(self, inner: ToolRegistry) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def execute(self, name: str, **kwargs: Any) -> Any:
        flagged = bool(kwargs.get("allow_expensive", False))
        self.calls.append({"tool": name, "allow_expensive": flagged})
        if flagged:
            raise HarnessGuaranteeError(f"refused {name}: the harness never allows expensive tools")
        return self._inner.execute(name, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# -- one replicate -----------------------------------------------------------
def _executed_tools(registry: ToolRegistry | _WatchedRegistry, sess: Session) -> list[str]:
    """Tools that EXECUTED, in order: ``registry.execute`` appends an entry under the
    tool's own agent only after the tool ran (planner proposals are agent "planner")."""
    ran = []
    for entry in sess.audit.entries:
        try:
            tool = registry.get(entry.tool)
        except KeyError:
            continue
        if entry.agent == tool.agent:
            ran.append(entry.tool)
    return ran


def _stop_category(status: str, sess: Session, history: Sequence[Mapping[str, Any]]) -> str:
    if status in STATUS_STOP:
        return STATUS_STOP[status]
    last = next((e.action for e in reversed(sess.audit.entries) if e.agent == "planner"), None)
    if last == "policy_rejected" and history:
        return "budget" if str(history[-1].get("reason", "")).startswith("budget:") else "repeat"
    return ACTION_STOP.get(last or "", "stopped")


def _decisions(history: Sequence[Mapping[str, Any]], stop: str) -> list[dict[str, Any]]:
    out = []
    for i, h in enumerate(history):
        last = i == len(history) - 1
        if not h.get("allow"):
            outcome = "rejected"
        elif h.get("mode") == policy.MODE_GATED:
            outcome = "gated"
        elif last and stop in ("tool_error", "dataset_error"):
            outcome = "tool_error"
        elif last and stop == "expensive_refused":
            outcome = "refused_expensive"
        else:
            outcome = "ran"
        out.append({"step": h.get("step"), "tool": h.get("tool"), "args": dict(h.get("args") or {}),
                    "rationale": str(h.get("rationale") or ""), "mode": h.get("mode"),
                    "reason": str(h.get("reason") or ""), "outcome": outcome})
    return out


def _end_text(stop: str, detail: str, recorder: _RecordingPlanner) -> str:
    if stop in ("planner_complete", "silent_stop") and recorder.answers:
        last = recorder.answers[-1]
        return str(last.get("rationale") or last.get("stop_reason") or "")
    return detail


def _redact(value: Any, path: str) -> Any:
    if isinstance(value, str):
        return value.replace(path, DATASET_TOKEN)
    if isinstance(value, dict):
        return {k: _redact(v, path) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, path) for v in value]
    return value


def _replicate_record(index: int, out: Mapping[str, Any], sess: Session, *,
                      recorder: _RecordingPlanner, registry: _WatchedRegistry, spec: LlmSpec,
                      elapsed: float) -> dict[str, Any]:
    status = str(out.get("status"))
    history = [dict(h) for h in (sess.state.planner_history or [])]
    stop = _stop_category(status, sess, history)
    if stop == "tool_error" and history and history[-1].get("tool") == "load_dataset":
        stop = "dataset_error"          # the harness's input failed, not a planner decision
    proposal = out.get("proposal") if status == "awaiting_review" else None
    gated = ({"tool": proposal["tool"], "args": dict(proposal.get("args") or {}),
              "rationale": str(proposal.get("rationale") or ""),
              "gate_reason": str(proposal.get("gate_reason") or "")} if proposal else None)
    detail = str(out.get("stop_reason") or out.get("message")
                 or (gated["gate_reason"] if gated else ""))[:DETAIL_CHARS]
    decisions = _decisions(history, stop)
    if stop in END_BY_STOP:
        decisions.append({"step": len(history), "tool": END_BY_STOP[stop], "args": {},
                          "rationale": _end_text(stop, detail, recorder), "mode": None,
                          "reason": detail, "outcome": "ended"})
    executed = _executed_tools(registry, sess)
    return {
        "replicate": index, "provider": spec.provider, "model": spec.model,
        "sampling": spec.sampling(index),
        "elapsed_s": round(elapsed, 3), "planner_s": round(recorder.planner_s, 3),
        "status": status, "stop_reason": stop, "stop_detail": detail,
        "autonomous_tools": executed, "gated_proposal": gated,
        "rejected": [{"step": d["step"], "tool": d["tool"], "args": d["args"],
                      "rationale": d["rationale"], "reason": d["reason"]}
                     for d in decisions if d["outcome"] == "rejected"],
        "decisions": decisions,
        "execute_calls": list(registry.calls),
        "audit_entries_scanned": len(sess.audit.entries),
        "expensive_executed": [t for t in executed if registry.get(t).expensive],
    }


def run_replicate(index: int, llm: LLM, dataset: str, *, spec: LlmSpec,
                  workflow: str = WORKFLOW) -> dict[str, Any]:
    """One fresh in-memory session, one goal leg, stopped at its first gate."""
    recorder = _RecordingPlanner(llm)
    registry = _WatchedRegistry(default_registry())
    orch = Orchestrator(llm=recorder, registry=registry, store=SessionStore(":memory:"))
    sess = orch.create_session()
    t0 = time.perf_counter()
    try:
        # allow_expensive deliberately NOT passed: its default (False) is what makes
        # ToolRegistry.execute refuse every long-running fit.
        out = orch.start_workflow(sess.id, workflow, {"path": dataset})
    except ExpensiveToolError as exc:
        out = {"status": "awaiting_job", "message": str(exc)}
    except Exception as exc:  # recorded as this replicate's (invalid) stop, never swallowed
        out = {"status": "error", "stop_reason": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - t0
    record = _replicate_record(index, out, sess, recorder=recorder, registry=registry,
                               spec=spec, elapsed=elapsed)
    flagged = [c["tool"] for c in record["execute_calls"] if c["allow_expensive"]]
    if record["expensive_executed"] or flagged:
        raise HarnessGuaranteeError(
            f"replicate {index}: expensive tool(s) executed {record['expensive_executed']}; "
            f"calls with allow_expensive=True {flagged}", record=_redact(record, dataset))
    return _redact(record, dataset)


# -- the run -----------------------------------------------------------------
def prepare(dataset: str, replicates: int, workflow: str = WORKFLOW) -> str:
    """Validate the inputs before any LLM is built; return the confined dataset path."""
    if not dr.MIN_REPLICATES <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be {dr.MIN_REPLICATES}..{MAX_REPLICATES}, got {replicates}")
    if get_workflow(workflow).get("kind") != "goal":
        raise ValueError(f"{workflow} is not a goal-driven workflow")
    return str(_safe_path(dataset))     # load_dataset's own confinement


def run_divergence(dataset: str, *, spec: LlmSpec, replicates: int = DEFAULT_REPLICATES,
                   workflow: str = WORKFLOW, llm_factory: Callable[[int], LLM] | None = None,
                   on_replicate: Callable[[dict[str, Any]], None] | None = None,
                   ) -> dict[str, Any]:
    """Run ``replicates`` independent goal legs and return the divergence report."""
    path = prepare(dataset, replicates, workflow)
    sha_before = file_sha256(path)
    factory = llm_factory or spec.build
    reps = []
    for i in range(replicates):
        rep = run_replicate(i, factory(i), path, spec=spec, workflow=workflow)
        reps.append(rep)
        if on_replicate is not None:
            on_replicate(rep)
    dataset_info = {"name": Path(path).name, "sha256_before": sha_before,
                    "sha256_after": file_sha256(path)}
    return dr.build_report(reps, llm=spec.public(), dataset=dataset_info, workflow=workflow)


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / JSON_NAME, out_dir / MD_NAME
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(dr.render_markdown(report), encoding="utf-8")
    return json_path, md_path


# -- CLI -----------------------------------------------------------------------
def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m app.evals.planner_divergence",
        description="Replicate-divergence harness for the poppk_goal planner (never runs a fit).")
    ap.add_argument("--dataset", required=True,
                    help="CSV under the PharmAgent data dir or backend/sample_data")
    ap.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    ap.add_argument("--provider", choices=PROVIDERS, default="local",
                    help="mock (harness self-test), local (Ollama/LM Studio on loopback), or "
                         "configured (the app's environment configuration)")
    ap.add_argument("--model", help=f"local model (default {DEFAULT_MODELS['local']})")
    ap.add_argument("--base-url", help=f"local server (default {settings.DEFAULT_OLLAMA_URL})")
    ap.add_argument("--temperature", type=float, help="local only (default: the app's own, 0)")
    ap.add_argument("--seed", type=int, help="local, with --temperature > 0: replicate i uses seed + i")
    ap.add_argument("--timeout", type=float,
                    help="local only: seconds per planner call (default: the client's own, 120)")
    ap.add_argument("--allow-hosted", action="store_true",
                    help="configured only: confirm sending state summaries to a hosted API")
    ap.add_argument("--out", required=True, help="output directory for the JSON + markdown report")
    return ap.parse_args(argv)


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.hostname in LOOPBACK_HOSTS


def spec_from_args(ns: argparse.Namespace) -> LlmSpec:
    if ns.provider == "local":
        url = ns.base_url or settings.DEFAULT_OLLAMA_URL
        if not _is_loopback_url(url):
            raise ValueError(f"--provider local needs a loopback server (127.0.0.1, localhost, ::1), "
                             f"got {url}; a hosted endpoint is --provider configured --allow-hosted")
        if ns.temperature is not None and not 0 <= ns.temperature <= 2:
            raise ValueError(f"--temperature must be within 0..2, got {ns.temperature}")
        if ns.seed is not None and not ns.temperature:
            raise ValueError("--seed only changes anything with --temperature > 0")
        if ns.timeout is not None and not 0 < ns.timeout <= MAX_TIMEOUT_S:
            raise ValueError(f"--timeout must be within 0..{MAX_TIMEOUT_S:g} s, got {ns.timeout}")
        return LlmSpec("local", ns.model or DEFAULT_MODELS["local"], url, ns.temperature, ns.seed,
                       ns.timeout)
    if any(v is not None for v in (ns.model, ns.base_url, ns.temperature, ns.seed, ns.timeout)):
        raise ValueError("--model, --base-url, --temperature, --seed and --timeout apply to "
                         "--provider local only")
    if ns.provider == "mock":
        return LlmSpec("mock", "mock")
    choice = choice_from_settings()
    if choice.provider == "mock":
        raise ValueError("--provider configured resolves to the mock (no LLM is configured); "
                         "use --provider mock for the self-test")
    if choice.provider in HOSTED_PROVIDERS and not ns.allow_hosted:
        raise ValueError(f"--provider configured resolves to the hosted {choice.provider} API, which "
                         "receives the planner's state summaries; pass --allow-hosted to confirm")
    return LlmSpec("configured", f"{choice.provider}:{choice.model}", choice.effective_base_url)


def _progress(rep: Mapping[str, Any]) -> None:
    gp = rep["gated_proposal"]
    what = dr.call_text(gp["tool"], gp["args"]) if gp else rep["stop_detail"]
    print(f"replicate {rep['replicate']}: {rep['stop_reason']} {dr.plain(what, 120)} "
          f"({rep['elapsed_s']} s)", file=sys.stderr)


def _self_test_failed(report: Mapping[str, Any]) -> bool:
    return (report["verdict"] != dr.ALL_CONVERGED
            or any(r["stop_reason"] != "gated_proposal" for r in report["replicates"]))


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse(argv)
    try:
        spec = spec_from_args(ns)
        prepare(ns.dataset, ns.replicates)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in (JSON_NAME, MD_NAME, FAILED_NAME):
        (out / stale).unlink(missing_ok=True)
    partial = out / PARTIAL_NAME
    partial.write_text("", encoding="utf-8")

    def on_replicate(rep: dict[str, Any]) -> None:
        _progress(rep)
        with partial.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rep, default=str) + "\n")

    try:
        report = run_divergence(ns.dataset, spec=spec, replicates=ns.replicates,
                                on_replicate=on_replicate)
    except HarnessGuaranteeError as exc:
        (out / FAILED_NAME).write_text(
            f"{exc}\n\n{json.dumps(exc.record, indent=2, default=str)}\n", encoding="utf-8")
        print(f"GUARANTEE BREACH: {exc}", file=sys.stderr)
        return EXIT_GUARANTEE
    json_path, md_path = write_report(report, out)
    first = report["first_divergence_step"]
    print(report["verdict"] + (f" (first divergence at step {first})" if first is not None else "")
          + f"\n{json_path}\n{md_path}")
    if any(r["stop_reason"] == "expensive_refused" for r in report["replicates"]):
        print("POLICY REGRESSION: the policy admitted an expensive tool without a gate; the "
              "registry refused it (see the report)", file=sys.stderr)
        return EXIT_POLICY
    if spec.provider == "mock" and _self_test_failed(report):
        print("self-test FAILED: the deterministic mock planner did not give identical replicates "
              "that all stop at the same gate", file=sys.stderr)
        return EXIT_SELF_TEST
    if report["verdict"] == dr.INCOMPLETE:
        for reason in report["incomplete_reasons"]:
            print(f"incomplete: {reason}", file=sys.stderr)
        return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
