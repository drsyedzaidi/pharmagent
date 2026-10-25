"""Ensure the backend package root is importable as `app` during tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _gate_critic_off_unless_a_test_enables_it(monkeypatch):
    """PHARMAGENT_MIKEBRAIN_GATE_CRITIC / a .env must not turn the suite's planner gates into real LLM calls."""
    from app.config import settings
    monkeypatch.setattr(settings, "mikebrain_gate_critic", False)
