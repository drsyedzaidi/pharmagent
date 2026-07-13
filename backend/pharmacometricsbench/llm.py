"""LLM adapter — route a task through prompt → text → parse → grade.

The real value of this module is the *plumbing*, all of which is testable
keyless:

1. ``build_prompt(task)`` turns a Task into an instruction + the dataset, and
   tells the model to answer as a JSON object with the exact target keys.
2. A client returns free text (``LLMClient.complete``).
3. ``parse_answer(text, task)`` robustly extracts the JSON answer out of prose /
   code fences and coerces booleans — the fiddly part every real LLM run needs.

``MockLLM`` is a keyless stand-in: it reads the dataset embedded in the prompt,
computes an answer (``naive`` strategy = the tool-free floor; ``oracle`` strategy
= tool-quality, used to prove the parse round-trip is lossless), and renders it
as prose + a fenced JSON block — exactly what a real model would emit. Swapping
in a real client is a one-line change (see ``default_client``); no other code
moves.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from typing import Any, Protocol

from .agents import naive_predict, oracle_predict
from .spec import Task

# The exact answer keys the model must return, per category.
TARGET_KEYS: dict[str, list[str]] = {
    "nca": ["Cmax", "AUC_inf", "t_half"],
    "be": ["gmr_pct", "ci_lower_pct", "ci_upper_pct", "within_limits"],
    "dp": ["slope", "proportional"],
    "compartmental": ["CL", "V", "ka"],
    "exposure": ["Cmax_ss", "AUC_tau"],
}

_DATA_MARKER = "DATA_JSON:"


# ── Prompt construction ────────────────────────────────────────────────────
def build_prompt(task: Task) -> str:
    keys = TARGET_KEYS.get(task.category, [t.name for t in task.targets])
    return (
        "You are a pharmacometrics analyst. Solve the task below and return your "
        "answer as a single JSON object inside a ```json code fence.\n"
        f"The JSON must contain exactly these keys: {', '.join(keys)}. "
        "Use numbers for quantities and true/false for verdicts.\n"
        "Keep any working brief; the LAST thing in your response must be the "
        "```json answer block.\n\n"
        f"TASK: {task.prompt}\n\n"
        f"CATEGORY: {task.category}\n"
        f"{_DATA_MARKER} {json.dumps(task.dataset)}\n"
    )


# ── Answer parsing (the part a real LLM run actually needs) ────────────────
def parse_answer(text: str, task: Task) -> dict[str, Any]:
    """Extract {target_name: value} from free-form model text."""
    obj = _extract_json_object(text)
    if obj is None:
        return {}
    out: dict[str, Any] = {}
    for tgt in task.targets:
        if tgt.name not in obj:
            continue
        val = obj[tgt.name]
        if tgt.tol["type"] == "exact":
            out[tgt.name] = _coerce_bool(val)
        else:
            out[tgt.name] = _coerce_float(val)
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    # A model may show a worked example (with its own braces) before committing
    # the final answer, so prefer the LAST ```json fence, then the last bare
    # {...}. First-match would grab an intermediate object.
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    bare = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    candidates = list(reversed(fences)) + list(reversed(bare))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _coerce_bool(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "pass"}:
            return True
        if s in {"false", "no", "fail"}:
            return False
    return v


# ── Clients ────────────────────────────────────────────────────────────────
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockLLM:
    """Keyless stand-in. Reads the embedded dataset, computes an answer with the
    chosen strategy, and emits realistic prose + a fenced JSON block."""

    def __init__(self, strategy: str = "naive") -> None:
        self.strategy = strategy

    def complete(self, prompt: str) -> str:
        category, dataset = self._read_prompt(prompt)
        predict = oracle_predict if self.strategy == "oracle" else naive_predict
        answer = predict(category, dataset)
        answer = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in answer.items()}
        body = json.dumps(answer)
        return (
            f"Here is my {category.upper()} analysis. Based on the profile I "
            f"estimate the following values.\n\n```json\n{body}\n```\n"
        )

    @staticmethod
    def _read_prompt(prompt: str) -> tuple[str, dict[str, Any]]:
        cat = re.search(r"CATEGORY:\s*(\w+)", prompt)
        data = re.search(rf"{re.escape(_DATA_MARKER)}\s*(\{{.*\}})", prompt, re.DOTALL)
        category = cat.group(1) if cat else ""
        dataset = json.loads(data.group(1)) if data else {}
        return category, dataset


class AnthropicLLM:
    """Real Claude client — the one line that turns a keyless self-test into an
    actual model score. ``complete(prompt)`` calls the Messages API and returns
    the text; the rest of the pipeline (build_prompt → parse_answer → grade) is
    unchanged. Model defaults to Haiku 4.5 (cheapest — ~$0.11 for a 30-task run);
    override with ``PMBENCH_LLM_MODEL`` (e.g. ``claude-opus-4-8`` for the headline)."""

    def __init__(self, model: str | None = None, max_tokens: int = 4096) -> None:
        import anthropic  # lazy: keyless paths never import the SDK
        self._client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("PMBENCH_LLM_MODEL", "claude-haiku-4-5")
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


def default_client() -> LLMClient:
    """Keyless by default (MockLLM), so tests and CI need no API key. When
    ``ANTHROPIC_API_KEY`` is set, score a **real** model instead — no other code
    moves. Set ``PMBENCH_LLM_MODEL`` to pick the model (default: Haiku 4.5)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception:  # SDK missing / construction failed → stay keyless
            pass
    return MockLLM(strategy="naive")


# ── Agent factory ──────────────────────────────────────────────────────────
def make_llm_agent(client: LLMClient) -> Callable[[Task], dict]:
    def llm_agent(task: Task) -> dict:
        return parse_answer(client.complete(build_prompt(task)), task)
    return llm_agent


def make_model_agent(model: str) -> Callable[[Task], dict]:
    """A reference LLM agent pinned to a specific model (e.g. Opus 4.8). Unlike the
    generic ``llm`` agent, this NEVER falls back to the keyless MockLLM — if no key
    is set it raises, so a keyless run can't masquerade as a real model score. The
    client is built lazily so the agent registry imports without a key."""
    box: list[LLMClient] = []

    def agent(task: Task) -> dict:
        if not box:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    f"reference agent for {model} needs ANTHROPIC_API_KEY "
                    "(it scores a real model — no keyless fallback)")
            box.append(AnthropicLLM(model=model))
        return parse_answer(box[0].complete(build_prompt(task)), task)

    return agent


llm_agent = make_llm_agent(default_client())
