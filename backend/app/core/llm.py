"""LLM client.

Three responsibilities only — everything quantitative is a tool:
  1. classify(message, options)  -> route to an agent
  2. select_tool(agent, ...)     -> pick the next tool (or None to stop)
  3. plan(goal, ...)             -> propose the next tool of a goal-driven
                                    workflow leg (a deterministic policy then
                                    admits, gates or rejects the proposal)

`RealLLM` uses Anthropic Claude (tool-use). `OpenAICompatLLM` talks to any
OpenAI-compatible chat-completions server with function calling — a FREE,
LOCAL model through Ollama or LM Studio, or a hosted free tier (OpenRouter,
Groq). `MockLLM` is deterministic and keyless, so the platform and the test
suite run with no external calls. The active client is chosen by
configuration (see app.config.Settings.llm_provider_effective).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import requests

from app.config import settings
from app.tools.base import Tool

log = logging.getLogger("pharmagent")


class LLM(Protocol):
    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str: ...
    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None: ...
    def plan(self, goal: str, state_summary: dict[str, Any], observations: list[dict[str, Any]],
             history: list[dict[str, Any]], tools: list[Tool]) -> dict[str, Any] | None: ...
    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str: ...


_PATH_RE = re.compile(r"[\w./~-]+\.(?:csv|xpt|sas7bdat)", re.IGNORECASE)


_KE_RE = re.compile(
    r"(?:\bke|(?:elimination\s+)?rate\s+constant)\s*(?:is|=|of)?\s*([0-9]*\.?[0-9]+)"
)
_HL_RE = re.compile(r"half[- ]life\s*(?:is|=|of)?\s*([0-9]*\.?[0-9]+)")


def _mock_numbers(low: str, tool: str) -> dict[str, Any]:
    """Best-effort numeric extraction for the keyless mock (half-life / ke only;
    everything else is left for the tool to ask for)."""
    if tool in ("calc_half_life", "calc_accumulation"):
        m = _KE_RE.search(low)
        if m:
            return {"ke": float(m.group(1))}
        m = _HL_RE.search(low)
        if m:
            return {"half_life": float(m.group(1))}
    return {}


_ER_SELECT = ("optimal dose", "dose selection", "select the dose", "optimus", "utility")
_ER_FIT = ("logistic", "odds ratio", "hazard ratio", "cox", "kaplan", "exposure-response",
           "exposure response")
_ER_TTE = ("hazard", "cox", "kaplan", "survival", "time to event", "time-to-event")


def _mock_er_dose(low: str, s: dict[str, Any]) -> dict[str, Any] | None:
    """Keyless exposure-response choices. The mock cannot read column names, the
    doses or the utility weight from prose: the tool's own refusal names what is
    missing. The bootstrap is only ever PROPOSED (expensive+proposable); the mock
    never confirms on the human's behalf."""
    if any(k in low for k in _ER_SELECT):
        return None if s.get("dose_selection") else {"name": "select_optimal_dose", "input": {"doses": []}}
    if "bootstrap" in low:
        return {"name": "bootstrap_exposure_response", "input": {}}
    if any(k in low for k in _ER_FIT):
        if s.get("er_fits"):
            return None
        endpoint = "time_to_event" if any(k in low for k in _ER_TTE) else "binary"
        return {"name": "fit_exposure_response", "input": {"endpoint": endpoint}}
    return None


PLAN_STOP = {"tool": None, "args": {}}
_INLINE_CALL_RE = re.compile(
    r"(?:^|\W)(?:call:(?P<call>[A-Za-z_]\w*)\s*\{|(?P<fn>[A-Za-z_]\w*)\s*\()"
)


def _plan_stop(reason: str, rationale: str = "") -> dict[str, Any]:
    """A planner answer that selects no tool: the loop ends with ``stop_reason``.
    ``rationale`` is left as the model gave it — an EMPTY rationale (no text, or
    only unknown tool names) is how the orchestrator tells a silent stop from a
    reasoned "goal met"."""
    return {**PLAN_STOP, "rationale": rationale, "stop_reason": reason}


def _plain_text_tool_call(text: str, known: set[str]) -> str | None:
    """First known tool name written as a call in plain text (``tool(...)`` or
    ``call:tool{...}``), or None.

    Found by the replicate-divergence harness: ``gemma4:31b`` wrote ``fit_pk_model(...)``
    in a fenced Python block or ``call:fit_pk_model{...}`` instead of emitting a
    structured tool call in 3 of 5 replicates. Without this check, a plain-text
    reply with a rationale is mistaken for a reasoned "goal met" (``complete``).
    Bare mentions in prose (e.g. "after load_dataset has run") have no ``(`` or
    ``call:...{`` and are not matched.
    """
    for m in _INLINE_CALL_RE.finditer(text or ""):
        name = m.group("call") or m.group("fn")
        if name in known:
            return name
    return None


def _plan_no_tool(rationale: str, known: set[str]) -> dict[str, Any]:
    called = _plain_text_tool_call(rationale, known)
    if called is not None:
        return _plan_stop(
            f"planner wrote a tool call ({called}) in plain text instead of a structured tool call",
            rationale="")
    return _plan_stop(rationale or "planner selected no tool", rationale)


def _planner_system(goal: str, state_summary: dict[str, Any], observations: list[dict[str, Any]],
                    history: list[dict[str, Any]]) -> str:
    """System text shared by the hosted planners (Anthropic / OpenAI-compatible)."""
    from app.compute.pk_models import REGISTRY
    from app.core.policy import PREREQUISITES
    valid_models = ", ".join(sorted(REGISTRY))
    prereqs = ", ".join(f"{t} requires {' & '.join(req)}" for t, req in PREREQUISITES.items())
    return (
        "You are the planner of a governed pharmacometrics workflow. Propose ONE next tool "
        "call that makes progress toward the goal, or answer in plain text when the goal "
        "is met or no admissible step remains. Always use a structured tool call (never "
        "write code blocks or tool calls in plain text). A deterministic policy then admits "
        "the call, parks it for human approval (every long-running fit and every scientific "
        "model choice is gated), or rejects it; you never execute anything yourself. Never "
        "repeat a call with identical arguments: every tool is deterministic. Give a "
        "one-sentence rationale grounded in the diagnostics before the call.\n"
        "When State shows dataset_id is None and dataset_configured is True, a dataset path "
        "is already configured for this workflow — call load_dataset (the orchestrator "
        "injects the configured path automatically).\n"
        f"Valid PK model library keys (for model_key / models): {valid_models}\n"
        f"Policy prerequisites: {prereqs}\n"
        f"Goal: {goal}\nState: {state_summary}\nObservations this leg: {observations}\n"
        f"Prior proposals and policy decisions: {history}")


# State-summary keys that prove a poppk_full template step has already run,
# used by the keyless mock to walk the template without re-running a step.
_TEMPLATE_DONE_KEYS: dict[str, str] = {
    "load_dataset": "dataset_id",
    "profile_pk_dataset": "data_quality",
    "fit_pk_model": "pk_model_results",
    "run_nlme": "nlme_results",
    "generate_report": "report_path",
}


class MockLLM:
    """Deterministic, keyless. Drives the core NCA flow heuristically."""

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        return options[0] if options else "data_manager"

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        s = state_summary
        if agent == "data_manager":
            if not s.get("dataset_id"):
                m = _PATH_RE.search(message or "")
                return {"name": "load_dataset", "input": {"path": m.group(0)}} if m else None
            # `data_quality` in the summary is the quality_flags LIST: an empty
            # list means "profiled, nothing flagged", not "not profiled" — a
            # falsy check here re-profiled clean datasets MAX_TOOL_STEPS times.
            if s.get("data_quality") is None:
                return {"name": "profile_pk_dataset", "input": {}}
            return None
        if agent == "nca":
            return None if s.get("nca_parameters") else {"name": "compute_nca", "input": {}}
        if agent == "be":
            return None if s.get("be_results") else {"name": "run_bioequivalence", "input": {}}
        if agent == "dose_prop":
            return None if s.get("dose_prop_results") else {"name": "run_dose_proportionality", "input": {}}
        if agent == "compartmental":
            return None if s.get("compartmental_results") else {"name": "fit_compartmental", "input": {}}
        if agent == "poppk":
            return None if s.get("poppk_results") else {"name": "run_poppk", "input": {}}
        if agent == "modeler":
            return None if s.get("pk_model_results") else {"name": "fit_pk_model", "input": {"compare": True}}
        if agent == "reviewer":
            rev = s.get("review_diagnostics") or {}
            return None if rev.get("status") else {"name": "adversarial_review", "input": {}}
        if agent == "qc":
            return None if s.get("qc_verdict") else {"name": "run_qc", "input": {}}
        if agent == "report":
            low = (message or "").lower()
            if "memo" in low or "briefing" in low:
                return None if s.get("memo_path") else {"name": "build_briefing_memo", "input": {}}
            return None if s.get("report_path") else {"name": "generate_report", "input": {}}
        if agent == "statistician":
            low = (message or "").lower()
            # An indirect comparison needs the two direct effects, which the keyless
            # mock cannot read from prose: stop with the idle hint naming the inputs.
            if any(k in low for k in ("indirect", "bucher", "common comparator")):
                return None
            return None if s.get("stats_advice") else {"name": "recommend_statistics", "input": {}}
        if agent == "er_dose":
            return _mock_er_dose((message or "").lower(), s)
        if agent == "clinpharm":
            low = (message or "").lower()
            # keyword → calculator; the keyless mock passes the numbers it can find
            # as a best effort and the tool names any input still missing.
            table = (("calc_renal_function", ("creatinine", "crcl", "egfr", "ckd", "cockcroft", "renal")),
                     ("calc_be_sample_size", ("sample size", "how many subjects")),
                     ("calc_dose_regimen", ("loading dose", "maintenance dose")),
                     ("calc_accumulation", ("accumulation", "steady state", "steady-state")),
                     ("calc_allometric", ("allometric", "allometry", "body weight scaling")),
                     ("convert_concentration", ("micromolar", "µm", "umol", "mg/l", "molar mass",
                                                "molecular weight")),
                     ("quick_one_compartment", ("compartment profile", "simulate a one")),
                     ("calc_half_life", ("half-life", "half life", "rate constant", " ke ", "ke is",
                                         "elimination rate")))
            for name, keys in table:
                if any(k in low for k in keys):
                    if s.get("last_calculation") == name:
                        return None            # already answered this turn
                    return {"name": name, "input": _mock_numbers(low, name)}
            return None
        if agent == "simulator":
            low = (message or "").lower()
            # Every choice here is a PROPOSAL (expensive+proposable tools): the
            # keyless mock never confirms on the human's behalf and cannot
            # compose a `design`/`params` from prose -- the human fills/approves.
            if any(k in low for k in ("simulation-estimation", "simulation estimation",
                                      "simest", "sim-est")):
                return {"name": "run_simest", "input": {}}
            if "bootstrap" in low:
                return {"name": "run_bootstrap", "input": {}}
            if any(k in low for k in ("importance resampling", "sampling importance", " sir")):
                return {"name": "run_sir", "input": {}}
            if any(k in low for k in ("likelihood profil", "profile likelihood")):
                return {"name": "run_profile", "input": {}}
            return None
        return None

    def plan(self, goal: str, state_summary: dict[str, Any], observations: list[dict[str, Any]],
             history: list[dict[str, Any]], tools: list[Tool]) -> dict[str, Any] | None:
        """Walk the ``poppk_full`` template order, restricted to the tools offered.

        The keyless mock cannot read diagnostics, so a goal-driven leg under it
        degrades to today's rails: each call proposes the first template step
        that has neither left its mark in state nor already been admitted by the
        policy (``history``), with the template's own arguments. Once every step
        is done it selects no tool, with a stop reason — never silently nothing.
        """
        from app.workflows import get_workflow  # local: keep the import graph light
        roster = {t.name for t in tools}
        admitted = {h.get("tool") for h in history if h.get("allow")}
        for step in get_workflow("poppk_full")["steps"]:
            tool = step["tool"]
            if tool not in roster or tool in admitted:
                continue
            marker = _TEMPLATE_DONE_KEYS.get(tool)
            if marker and state_summary.get(marker) is not None:
                continue
            return {"tool": tool, "args": dict(step.get("args", {})),
                    "rationale": f"poppk_full template order: {step.get('label', tool)}",
                    "stop_reason": None}
        return _plan_stop("every poppk_full template step offered to the planner has run",
                          rationale="poppk_full template exhausted")

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        """Keyless: the mikebrain critic never claims evidence it did not read, so
        the mock abstains — tests exercise the deterministic grounding path."""
        return json.dumps({"verdict": "no_evidence",
                           "rationale": "mock LLM: no literature was consulted", "citations": []})


class RealLLM:
    """Anthropic Claude client."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        import anthropic
        self.model = model or settings.model
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        roster = "\n".join(f"- {o}: {descriptions.get(o, '')}" for o in options)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=64,
            system=("You route a pharmacometrics request to ONE specialist agent. "
                    "Reply with only the agent key, nothing else.\n" + roster),
            messages=[{"role": "user", "content": message}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
        for o in options:
            if o in text:
                return o
        return options[0] if options else "data_manager"

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        from app.agents.definitions import AGENTS
        domain_prompt = AGENTS[agent].system_prompt if agent in AGENTS else ""
        system_text = (f"You are the {agent} agent in a pharmacometrics platform. "
                       f"{domain_prompt}\n" if domain_prompt else
                       f"You are the {agent} agent in a pharmacometrics platform. ")
        system_text += ("Use a tool to make progress, or answer directly if the task is done. "
                        f"Current state: {state_summary}")
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=system_text,
            tools=[t.to_anthropic() for t in tools],
            messages=[{"role": "user", "content": message}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return {"name": block.name, "input": block.input}
        return None

    def plan(self, goal: str, state_summary: dict[str, Any], observations: list[dict[str, Any]],
             history: list[dict[str, Any]], tools: list[Tool]) -> dict[str, Any] | None:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=_planner_system(goal, state_summary, observations, history),
            tools=[t.to_anthropic() for t in tools],
            messages=[{"role": "user", "content": "Propose the next tool call, or say why to stop."}],
        )
        rationale = " ".join(b.text for b in resp.content if b.type == "text").strip()
        known = {t.name for t in tools}
        for block in resp.content:
            if block.type == "tool_use" and block.name in known:
                return {"tool": block.name, "args": dict(block.input or {}),
                        "rationale": rationale, "stop_reason": None}
        return _plan_no_tool(rationale, known)

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        """Plain text answer (no tools) — used by the mikebrain critic."""
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if b.type == "text")


LOCAL_REASONING_EFFORT = "none"


def is_loopback(url: str) -> bool:
    """True for a server on this machine (Ollama, LM Studio), false for hosted APIs."""
    return "127.0.0.1" in url or "localhost" in url or "[::1]" in url


class OpenAICompatLLM:
    """OpenAI chat-completions client (function calling) over plain HTTP.

    Works unchanged against Ollama (``http://127.0.0.1:11434/v1``, no key),
    LM Studio (``http://127.0.0.1:1234/v1``), OpenRouter, Groq, vLLM, or the
    OpenAI API itself. The model must support tool calling (Ollama: gemma4, qwen3, qwen2.5,
    llama3.1/3.2, mistral-nemo, ...); a model that only answers in prose
    simply selects no tool (the turn ends with an idle hint).

    Only ``requests.post`` is used, so tests fake it without a server.
    """

    TIMEOUT_S = 120.0   # local 7B models on CPU can take a while per call
    # The mikebrain critic call carries ~6 excerpts and asks for grounded JSON; a
    # local 31B model that has to load first exceeded 120 s (measured 2026-10-23,
    # gemma4:31b cold start), while warm calls took 75-116 s.
    COMPLETE_TIMEOUT_S = 300.0

    def __init__(self, *, base_url: str, model: str, api_key: str | None,
                 reasoning_effort: str | None = None, temperature: float | None = None,
                 seed: int | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # Sampling overrides for evaluation harnesses (app.evals.planner_divergence)
        # only. None keeps the app's request exactly as before: temperature 0, no seed.
        self.temperature = temperature
        self.seed = seed
        # Sent on ROUTING calls only. "none" switches off hidden reasoning in
        # local thinking models (qwen3 on Ollama): with classify's 32-token
        # budget they spend everything thinking and return empty content. Tool
        # selection keeps the model's default because qwen3 picks the right
        # calculator 6/6 with thinking on but 4/6 with it off (measured
        # 2026-09-30). Never set for hosted OpenAI: it rejects the field for
        # non-reasoning models.
        self.reasoning_effort = reasoning_effort

    # -- transport ----------------------------------------------------------
    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
              max_tokens: int = 512, quick: bool = False, timeout: float | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
        if quick and self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                             timeout=timeout or self.TIMEOUT_S)
        if resp.status_code >= 400:
            raise RuntimeError(self._http_error(resp.status_code))
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM server {self.base_url} returned no choices: {data}")
        return choices[0].get("message") or {}

    def _http_error(self, status: int) -> str:
        """A message that says what to fix, not just the status code."""
        where = f"{self.base_url} (model {self.model!r})"
        if status in (401, 403):
            return f"authentication failed at {where}: check the API key"
        if status == 404:
            return (f"model {self.model!r} not found at {self.base_url}: pull/load it "
                    "(Ollama: `ollama pull <model>`) or pick another name")
        if status == 429:
            return f"rate limit / quota exceeded at {where}"
        if status >= 500:
            return f"LLM server error HTTP {status} at {where}"
        return f"HTTP {status} from {where}; does the model support tool calling?"

    # -- LLM protocol -------------------------------------------------------
    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        roster = "\n".join(f"- {o}: {descriptions.get(o, '')}" for o in options)
        msg = self._chat([
            {"role": "system", "content": (
                "You route a pharmacometrics request to ONE specialist agent. "
                "Reply with only the agent key, nothing else.\n" + roster)},
            {"role": "user", "content": message},
        ], max_tokens=32, quick=True)
        text = (msg.get("content") or "").strip().lower()
        for o in options:
            if o in text:
                return o
        fallback = options[0] if options else "data_manager"
        log.warning("llm_classify_no_match",
                    extra={"model": self.model, "reply": text[:80], "fallback": fallback})
        return fallback

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        from app.agents.definitions import AGENTS
        domain_prompt = AGENTS[agent].system_prompt if agent in AGENTS else ""
        system_text = (f"You are the {agent} agent in a pharmacometrics platform. "
                       f"{domain_prompt}\n" if domain_prompt else
                       f"You are the {agent} agent in a pharmacometrics platform. ")
        system_text += ("Call a tool to make progress, or answer in plain text if the task is "
                        f"done. Current state: {state_summary}")
        msg = self._chat([
            {"role": "system", "content": system_text},
            {"role": "user", "content": message},
        ], tools=[t.to_openai() for t in tools], max_tokens=settings.max_tokens)
        known = {t.name for t in tools}
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in known:
                log.warning("llm_unknown_tool", extra={"tool": name, "agent": agent})
                continue
            raw = fn.get("arguments") or {}
            if isinstance(raw, str):
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    log.warning("llm_bad_tool_arguments", extra={"tool": name})
                    args = {}
            else:
                args = dict(raw)
            return {"name": name, "input": args if isinstance(args, dict) else {}}
        return None

    def plan(self, goal: str, state_summary: dict[str, Any], observations: list[dict[str, Any]],
             history: list[dict[str, Any]], tools: list[Tool]) -> dict[str, Any] | None:
        msg = self._chat([
            {"role": "system", "content": _planner_system(goal, state_summary, observations, history)},
            {"role": "user", "content": "Propose the next tool call, or say why to stop."},
        ], tools=[t.to_openai() for t in tools], max_tokens=settings.max_tokens)
        rationale = (msg.get("content") or "").strip()
        known = {t.name for t in tools}
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in known:
                log.warning("llm_unknown_tool", extra={"tool": name, "agent": "planner"})
                continue
            raw = fn.get("arguments") or {}
            if isinstance(raw, str):
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    log.warning("llm_bad_tool_arguments", extra={"tool": name})
                    args = {}
            else:
                args = dict(raw)
            return {"tool": name, "args": args if isinstance(args, dict) else {},
                    "rationale": rationale, "stop_reason": None}
        return _plan_no_tool(rationale, known)

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        """Plain text answer (no tools) — used by the mikebrain critic. Thinking stays
        at the model's default (see reasoning_effort): a grounded verdict benefits
        from it and the token budget here is far above classify's."""
        msg = self._chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}], max_tokens=max_tokens,
                         timeout=self.COMPLETE_TIMEOUT_S)
        return msg.get("content") or ""


def get_llm() -> LLM:
    provider = settings.llm_provider_effective
    if provider == "anthropic":
        return RealLLM()
    if provider == "openai":
        url = settings.llm_base_url or settings.DEFAULT_OLLAMA_URL
        return OpenAICompatLLM(base_url=url, model=settings.model, api_key=settings.llm_api_key,
                               reasoning_effort=LOCAL_REASONING_EFFORT if is_loopback(url) else None)
    return MockLLM()
