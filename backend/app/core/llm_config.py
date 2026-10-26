"""Runtime LLM provider switching.

The user picks, from the UI, which model answers routing/tool-selection:

    mock       deterministic, keyless
    local      a local OpenAI-compatible server — Ollama by default — no key
    openai     the OpenAI API ("ChatGPT"): gpt-4o etc., needs an API key
    anthropic  Claude via the Anthropic SDK, needs an API key

Secrets policy: an API key entered in the UI lives in this process's memory
only. It is never written to disk, never returned by any endpoint (only a
``has_key`` flag), and never logged. The non-secret choice (provider, model,
base URL) is persisted to ``<data_dir>/llm_settings.json`` so a desktop
relaunch remembers it; keys come back from the environment or are re-entered.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from app.config import settings
from app.core.llm import LLM, LOCAL_REASONING_EFFORT, MockLLM, OpenAICompatLLM, RealLLM

log = logging.getLogger("pharmagent")

PROVIDERS: tuple[str, ...] = ("mock", "local", "openai", "anthropic")
OPENAI_API_URL = "https://api.openai.com/v1"
DEFAULT_MODELS: dict[str, str] = {
    "mock": "mock",
    "local": "gemma4:31b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-4-8",
}
SETTINGS_FILE = "llm_settings.json"
TEST_TIMEOUT_S = 60.0
ALLOWLIST_ENV = "PHARMAGENT_LLM_BASE_URL_ALLOWLIST"   # comma-separated scheme://host[:port]
_DEFAULT_PORTS = {"http": 80, "https": 443}
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_OLLAMA_PORT = 11434


def _origin(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port) with case and default ports normalised; None when the
    URL is not a plain http(s) URL with a host and no embedded credentials."""
    try:
        parts = urlsplit(url.strip())
        scheme, host = parts.scheme.lower(), (parts.hostname or "").lower().rstrip(".")
        port = parts.port or _DEFAULT_PORTS.get(scheme)
    except ValueError:
        return None
    if scheme not in _DEFAULT_PORTS or not host or port is None or parts.username or parts.password:
        return None
    return scheme, host, port


def same_origin(a: str | None, b: str | None) -> bool:
    oa, ob = (_origin(a) if a else None), (_origin(b) if b else None)
    return oa is not None and oa == ob


def _resolve_host(host: str) -> list[str]:
    return [i[4][0] for i in socket.getaddrinfo(host, None)]


def _allowed_origins() -> set[tuple[str, str, int]]:
    urls = [settings.DEFAULT_OLLAMA_URL, OPENAI_API_URL, settings.llm_base_url or ""]
    urls += os.environ.get(ALLOWLIST_ENV, "").split(",")
    urls += [f"http://{h if h != '::1' else '[::1]'}:{_OLLAMA_PORT}" for h in _LOOPBACK_HOSTS]
    return {o for o in (_origin(u) for u in urls if u.strip()) if o}


def check_base_url(url: str) -> None:
    """Raise ValueError unless ``url`` is allowlisted (scheme+host[:port]) or a
    public address. Private, loopback and link-local targets need an allowlist
    entry, so PUT /api/llm cannot be used to probe the host's own network."""
    origin = _origin(url)
    if origin is None:
        raise ValueError("base_url must be an http(s) URL with a host and no credentials")
    if origin in _allowed_origins():
        return
    host = origin[1]
    if host == "localhost" or host.endswith(".localhost"):
        addrs = ["127.0.0.1"]
    else:
        try:
            ipaddress.ip_address(host)
            addrs = [host]
        except ValueError:
            try:
                addrs = _resolve_host(host)
            except OSError as exc:
                raise ValueError(f"base_url host {host!r} did not resolve: {exc}") from exc
    if not addrs or not all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs):
        raise ValueError(f"base_url {origin[0]}://{host}:{origin[2]} is a private, loopback or "
                         f"link-local target; add it to {ALLOWLIST_ENV} to permit it")


@dataclass(frozen=True)
class LlmChoice:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None          # memory only — see module docstring

    def public(self) -> dict[str, Any]:
        """Client-facing view: never the key."""
        return {"provider": self.provider, "model": self.model,
                "base_url": self.effective_base_url, "has_key": bool(self.api_key)}

    @property
    def effective_base_url(self) -> str | None:
        if self.provider == "local":
            return self.base_url or settings.DEFAULT_OLLAMA_URL
        if self.provider == "openai":
            return self.base_url or OPENAI_API_URL
        return None

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; one of {', '.join(PROVIDERS)}")
        if not self.model and self.provider != "mock":
            raise ValueError("model is required")
        if self.provider in ("openai", "anthropic") and not self.api_key:
            raise ValueError(f"{self.provider} needs an API key")


def choice_from_settings() -> LlmChoice:
    """The choice implied by the process environment / config (startup default)."""
    p = settings.llm_provider_effective
    if p == "anthropic":
        return LlmChoice("anthropic", settings.model, None, settings.anthropic_api_key)
    if p == "openai":
        url = settings.llm_base_url or settings.DEFAULT_OLLAMA_URL
        hosted = "127.0.0.1" not in url and "localhost" not in url
        return LlmChoice("openai" if hosted else "local", settings.model, url, settings.llm_api_key)
    return LlmChoice("mock", "mock")


def build_llm(choice: LlmChoice) -> LLM:
    choice.validate()
    if choice.provider == "mock":
        return MockLLM()
    if choice.provider == "anthropic":
        return RealLLM(api_key=choice.api_key, model=choice.model)
    return OpenAICompatLLM(base_url=choice.effective_base_url or OPENAI_API_URL,
                           model=choice.model, api_key=choice.api_key,
                           reasoning_effort=LOCAL_REASONING_EFFORT if choice.provider == "local" else None)


def probe(llm: LLM) -> dict[str, Any]:
    """One cheap real call. Returns {ok, detail}; never raises."""
    try:
        out = llm.classify("compute NCA AUC and Cmax", ["nca", "data_manager"],
                           {"nca": "non-compartmental analysis", "data_manager": "load data"})
        return {"ok": True, "detail": f"model answered; routed a test request to '{out}'"}
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
        return {"ok": False, "detail": str(exc)[:400]}


def ollama_models(base_url: str | None) -> list[str]:
    """Model names an Ollama server has pulled (``/api/tags``); [] on any error."""
    url = (base_url or settings.DEFAULT_OLLAMA_URL).rstrip("/")
    root = url[:-3] if url.endswith("/v1") else url
    try:
        r = requests.get(f"{root}/api/tags", timeout=3)
        if r.status_code != 200:
            return []
        names = (m.get("name", "") for m in (r.json().get("models") or []))
        # embedding models (nomic-embed-text, mxbai-embed-large, ...) cannot chat
        return sorted(n for n in names if n and "embed" not in n.lower())
    except (requests.RequestException, ValueError):
        return []


class LlmManager:
    """Owns the active choice; applies it to the orchestrator and settings."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / SETTINGS_FILE
        self.choice = self._load() or choice_from_settings()

    # -- persistence (non-secret) ------------------------------------------
    def _load(self) -> LlmChoice | None:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        provider = raw.get("provider")
        if provider not in PROVIDERS:
            return None
        env = choice_from_settings()
        # keys are never on disk: reuse the environment's key when the provider matches
        key = env.api_key if env.provider == provider else None
        return LlmChoice(provider, raw.get("model") or DEFAULT_MODELS[provider],
                         raw.get("base_url") or None, key)

    def _save(self, choice: LlmChoice) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"provider": choice.provider, "model": choice.model, "base_url": choice.base_url},
                indent=1))
        except OSError as exc:
            log.warning("llm_settings_not_saved", extra={"error": str(exc)})

    def activate(self, orch: Any) -> None:
        """Make the loaded (persisted) choice live at startup. A choice that
        cannot be built now — e.g. ChatGPT whose key is not in this process —
        falls back to the environment default rather than failing startup."""
        try:
            self.apply(self.choice, orch)
        except (ValueError, ImportError) as exc:
            log.warning("llm_persisted_choice_unavailable",
                        extra={"provider": self.choice.provider, "error": str(exc)})
            self.choice = choice_from_settings()

    # -- API ---------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "current": self.choice.public(),
            "providers": list(PROVIDERS),
            "defaults": DEFAULT_MODELS,
            "local_models": ollama_models(self.choice.base_url if self.choice.provider == "local" else None),
            "label": settings.llm_label,
        }

    def resolve(self, body: dict[str, Any]) -> LlmChoice:
        """Merge a request body onto the current choice (a blank key keeps the
        key already held for that provider, so re-applying a model change does
        not force re-typing the secret)."""
        provider = body.get("provider") or self.choice.provider
        model = (body.get("model") or "").strip() or (
            self.choice.model if provider == self.choice.provider else DEFAULT_MODELS.get(provider, ""))
        base_url = (body.get("base_url") or "").strip() or None
        if base_url and provider in ("local", "openai"):
            check_base_url(base_url)
        target = LlmChoice(provider, model, base_url).effective_base_url

        def _may_reuse(held: LlmChoice) -> bool:
            # a held key goes only to the URL it was configured for (F-02)
            return held.provider == provider and (
                provider == "anthropic" or same_origin(target, held.effective_base_url))

        key = (body.get("api_key") or "").strip() or None
        if not key and _may_reuse(self.choice):
            key = self.choice.api_key
        if not key:
            env = choice_from_settings()
            if _may_reuse(env):
                key = env.api_key
        choice = LlmChoice(provider, model or DEFAULT_MODELS.get(provider, ""), base_url, key)
        choice.validate()
        return choice

    def apply(self, choice: LlmChoice, orch: Any) -> None:
        """Swap the live LLM and mirror the choice into settings (health label)."""
        llm = build_llm(choice)
        orch.set_llm(llm)
        if choice.provider == "mock":
            settings.llm_provider = "mock"
        elif choice.provider == "anthropic":
            settings.llm_provider = "anthropic"
            settings.anthropic_api_key = choice.api_key
            settings.model = choice.model
        else:
            settings.llm_provider = "openai"
            settings.llm_base_url = choice.effective_base_url
            settings.llm_api_key = choice.api_key
            settings.model = choice.model
        self.choice = choice
        self._save(choice)
        log.info("llm_switched", extra={"provider": choice.provider, "model": choice.model})
