"""Guards for model-supplied free text that a deterministic document will print.

The rule behind the briefing memo is that a language model never writes a number.
Free text a model can choose (a memo title, a treatment name) is therefore refused
if it carries any numeral, at the tool boundary and again when the memo is rendered.
Number words ("ninety") are not numerals and cannot be caught here.
"""
from __future__ import annotations

import unicodedata


def has_numeral(text: str) -> bool:
    """True if ``text`` holds any numeral character: any Unicode N* category, i.e. ASCII
    and fullwidth digits, superscripts, circled numbers, vulgar fractions, roman numerals."""
    return any(unicodedata.category(ch).startswith("N") for ch in text)


def reject_numerals(text: str, what: str) -> str:
    """``text`` unchanged, or ValueError naming ``what`` when it carries a numeral."""
    if has_numeral(text):
        raise ValueError(f"{what} must not contain digits: numbers come only from computed results")
    return text
