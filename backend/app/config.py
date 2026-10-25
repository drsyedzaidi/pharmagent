"""Application configuration.

Loaded from environment variables (or a .env file). The LLM falls back to a
deterministic mock when no API key is present, so the platform — and the test
suite — runs without external calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_user_tokens(raw: str | None) -> dict[str, str]:
    """Parse ``user=token,user2=token2`` into {token: user}. Rejects blanks, duplicates and reserved names."""
    out: dict[str, str] = {}
    users: set[str] = set()
    for item in (raw or "").split(","):
        if not item.strip():
            continue
        user, sep, token = item.partition("=")
        user, token = user.strip(), token.strip()
        if not sep or not user or not token:
            raise ValueError("api_tokens entries must look like user=token")
        if user.lower() == "anonymous" or user.lower().startswith("token:"):
            raise ValueError(f"api_tokens: reserved user name {user!r}")
        if user in users or token in out:
            raise ValueError("api_tokens: duplicate user or token")
        users.add(user)
        out[token] = user
    return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHARMAGENT_", env_file=".env", extra="ignore")

    # LLM. Three providers:
    #   anthropic  Claude via the Anthropic SDK (PHARMAGENT_ANTHROPIC_API_KEY)
    #   openai     any OpenAI-compatible chat-completions server with tool calling:
    #              Ollama (free, local, default URL below), LM Studio, OpenRouter,
    #              Groq, vLLM ... (PHARMAGENT_LLM_BASE_URL, optional PHARMAGENT_LLM_API_KEY)
    #   mock       deterministic, keyless (tests / demo)
    # `auto` (default) = anthropic if a key is set, else openai if a base URL is
    # set, else mock. `model` names the model for whichever provider is active.
    llm_provider: Literal["auto", "anthropic", "openai", "mock"] = "auto"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-8"
    max_tokens: int = 2048
    use_mock_llm: bool = False  # legacy switch; same as llm_provider="mock"

    DEFAULT_OLLAMA_URL: str = "http://127.0.0.1:11434/v1"

    # Storage
    data_dir: Path = Path("data")
    db_path: Path = Path("pharmagent.db")   # SQLite persistence file

    # Auth — when set, all /api/sessions endpoints require this bearer token.
    # Unset (default) = open dev mode.
    api_token: str | None = None
    # Optional per-user tokens ("alice=tok1,bob=tok2"): each user is its own actor/owner, so session and skill
    # ownership isolate users. Independent of api_token (both may be set); unset = no per-user auth.
    api_tokens: str | None = None
    # Datasets kept in memory across all sessions (LRU); reloaded from disk on demand.
    max_loaded_datasets: int = 8
    cors_origins: list[str] = ["*"]   # tighten in production (e.g. your UI origin)

    # Audit authenticity. ``off`` is explicitly hash-only development mode.
    # Authenticated deployments must use ``enforce`` with a distinct 256-bit+
    # HMAC keyring and an anchor outside the primary DB writer's trust boundary.
    audit_mode: Literal["off", "enforce"] = "off"
    audit_keys_json: SecretStr | None = None
    audit_active_key_id: str | None = None
    audit_anchor_path: Path | None = None

    # Behaviour
    app_name: str = "PharmAgent"
    org_name: str = "PmatricsAI"

    @field_validator("api_tokens")
    @classmethod
    def _check_api_tokens(cls, v: str | None) -> str | None:
        if v:
            parse_user_tokens(v)       # fail at startup on a malformed value; the message never echoes a token
        return v or None

    @property
    def user_tokens(self) -> dict[str, str]:
        """{token: user} parsed from ``api_tokens``."""
        return parse_user_tokens(self.api_tokens)

    @property
    def llm_provider_effective(self) -> Literal["anthropic", "openai", "mock"]:
        if self.use_mock_llm or self.llm_provider == "mock":
            return "mock"
        if self.llm_provider == "anthropic":
            return "anthropic" if self.anthropic_api_key else "mock"
        if self.llm_provider == "openai":
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        if self.llm_base_url:
            return "openai"
        return "mock"

    @property
    def llm_is_mock(self) -> bool:
        return self.llm_provider_effective == "mock"

    @property
    def llm_label(self) -> str:
        """Human-readable active provider for /api/health and the UI badge."""
        p = self.llm_provider_effective
        if p == "mock":
            return "mock"
        if p == "anthropic":
            return f"anthropic:{self.model}"
        return f"openai:{self.model}@{self.llm_base_url or self.DEFAULT_OLLAMA_URL}"

    @property
    def allowed_data_dirs(self) -> list[Path]:
        """Roots a dataset path may be read from (anti path-traversal)."""
        here = Path(__file__).resolve().parent.parent  # backend/
        return [self.data_dir.resolve(), (here / "sample_data").resolve()]


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
