"""Confidence-level text shared by tool summaries and the briefing memo.

A level is printed as a percentage at the precision it was given: 0.95 -> '95',
0.975 -> '97.5' (never rounded to '98'), 0.9999 -> '99.99'.
"""
from __future__ import annotations

#: decimals kept for a non-integral percentage (trailing zeros are dropped)
LEVEL_DECIMALS = 2


def level_percent(level: float) -> str:
    """``100 * level`` without a % sign: whole when integral, else up to two decimals."""
    text = f"{100 * float(level):.{LEVEL_DECIMALS}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text
