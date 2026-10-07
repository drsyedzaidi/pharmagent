"""Ensure the backend package root is importable as `app` during tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "exhaustive: the r6 every-code-point tokenizer sweep (set R6_EXHAUSTIVE=1)")
