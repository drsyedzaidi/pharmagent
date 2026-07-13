"""Ensure the backend package root is importable as `app` during tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
