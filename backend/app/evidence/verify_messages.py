"""Reviewer sentences of a numeric-grammar failure (``unresolved_context`` / ``unresolved_shape``).

Fixed server-written text, one template per family, built from the first offending code point or shape: no
exception text, no path, no page content beyond the cited literal. ``unicodedata.name`` is used here ONLY to put a
name next to a code point for a human (``U+200B ZERO WIDTH SPACE``); it never classifies a character. The reader
(``numgrammar/``) imports no ``unicodedata`` at all: it is a membership test on literal sets.
"""
from __future__ import annotations

import unicodedata

from app.evidence.numgrammar import Tok

COMPARE = "Compare the value with the page image."
SHAPE_WHY = {
    "too_many_digits": "more than 40 digits", "leading_zero": "a leading zero", "middot": "a raised dot used as a "
    "decimal point on a page that does not use it", "groups": "digit groups that are not groups of three",
    "ambiguous_tail": "an ambiguous digit-group tail", "separators": "an unreadable mix of separators",
    "exponent_range": "an exponent out of range", "ten_after_multiplier": "a bare ten after a multiplier (a raised "
    "exponent the text layer cannot tell from digits)", "multiplier_operand": "a multiplier without a number",
    "detached_exponent": "an exponent detached from its number", "spaced_exponent": "a spaced exponent",
    "mixed_number": "a mixed number", "fraction": "a fraction", "glued_letters": "letters glued to the digits",
    "glued_symbol": "a symbol glued to the digits", "bad_start": "a number that starts in an unreadable way",
    "bare_superscript_power": "a raised power of ten without a multiplier"}


def named(label: str) -> str:
    """``U+200B`` -> ``U+200B ZERO WIDTH SPACE``; the marker ``(cid:3)`` -> ``an unmapped glyph marker (cid:3)``."""
    if label.startswith("(cid:"):
        return f"an unmapped glyph marker {label}"
    name = unicodedata.name(chr(int(label[2:], 16)), "")
    return f"{label} {name}" if name else f"{label} (no assigned name)"


def context_sentence(literal: str, offender: str, where: str) -> str:
    """The cited number *literal* touches a character outside the allowed alphabet, at *where*."""
    return (f"the cited number {literal!r} touches {named(offender)} ({where}), a character outside the allowed "
            f"number alphabet; the text layer cannot be read as a number here. {COMPARE}")


def shape_sentence(expr: str, why: str) -> str:
    """The cited expression *expr* is written in an order of allowed characters the grammar rejects."""
    return (f"the cited number {expr!r} is written in a form the reader does not accept "
            f"({SHAPE_WHY.get(why, why.replace('_', ' '))}); read it on the page image.")


def unreadable_sentence(what: str, offender: str) -> str:
    """The *what* ('snippet' or 'value') of a proposal holds a character outside the allowed alphabet."""
    return (f"the {what} holds {named(offender)}, a character outside the allowed number alphabet; the text layer "
            f"cannot be read as a number here. {COMPARE}")


def tok_sentence(tok: Tok, text: str) -> str:
    """The reviewer sentence of an unresolved expression of *text* ('' for a resolved token)."""
    literal = text[tok.start:tok.end]
    if tok.reason == "unresolved_context":
        return context_sentence(literal, tok.offender, tok.where)
    return shape_sentence(literal, tok.why) if tok.value is None else ""
