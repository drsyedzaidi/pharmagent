"""Application configuration.

Loaded from environment variables (or a .env file). The LLM falls back to a
deterministic mock when no API key is present, so the platform — and the test
suite — runs without external calls.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHARMAGENT_", env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-8"
    max_tokens: int = 2048
    use_mock_llm: bool = False  # forced on automatically when no api key

    # Storage
    data_dir: Path = Path("data")
    db_path: Path = Path("pharmagent.db")   # SQLite persistence file

    # Auth — when set, all /api/sessions endpoints require this bearer token.
    # Unset (default) = open dev mode.
    api_token: str | None = None
    cors_origins: list[str] = ["*"]   # tighten in production (e.g. your UI origin)

    # Behaviour
    app_name: str = "PharmAgent"
    org_name: str = "PmatricsAI"

    @property
    def llm_is_mock(self) -> bool:
        return self.use_mock_llm or not self.anthropic_api_key

    @property
    def allowed_data_dirs(self) -> list[Path]:
        """Roots a dataset path may be read from (anti path-traversal)."""
        here = Path(__file__).resolve().parent.parent  # backend/
        return [self.data_dir.resolve(), (here / "sample_data").resolve()]


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
