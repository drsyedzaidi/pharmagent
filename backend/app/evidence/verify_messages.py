"""Reviewer sentences of a numeric-grammar failure (``unresolved_context`` / ``unresolved_shape``).

Fixed server-written text, one template per family, built from the first offending code point or shape: no
exception text, no path, no page content beyond the cited literal. ``unicodedata.name`` is used here ONLY to put a
name next to a code point for a human (``U+200B ZERO WIDTH SPACE``); it never classifies a character. The reader
(``numgrammar/``) imports no ``unicodedata`` at all: it is a membership test on literal sets.
"""
from __future__ import annotations

import unicodedata

from app.evidence.numgrammar import Tok
from app.evidence.numgrammar.alphabet import CLOSERS_BR, GROUP_ONLY, TRAILING

_GLUE_MARKS = TRAILING - CLOSERS_BR - GROUP_ONLY    # , . ; : glued between a name and its digits ('Cmax:12')

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
    "bare_superscript_power": "a raised power of ten without a multiplier",
    "citation": "a bracketed reference citation, which is not a value",
    "citation_list": "a pair of reference numbers in square brackets, never a decimal",
    "split_unit": "a unit broken by a space inside it, as in '10m L': the text layer split the unit",
    "label": "digits inside a label word, which are part of a name, never a value",
    "neighbour": "a word beside the number that belongs to no accepted form: a sign, multiplier, exponent letter, "
    "qualifier, percent or number word the reader cannot place, which may change the value"}


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


def label_sentence(literal: str, label: str) -> str:
    """The cited number *literal* is a digit run of the label word *label* (r7, owner decision 3). When a mark glues it
    to the name ('Cmax:12', 'Cmax,12', 'p<.05': r7 round 2) the sentence says so: it may be a value written without a
    space, which the reader cannot tell from digits of a name."""
    at = label.find(literal)
    if at > 0 and label[at - 1] in _GLUE_MARKS:
        return (f"the cited number {literal!r} is glued to the name {label!r} by {label[at - 1]!r} with no space: a "
                f"value written after a name cannot be told from digits inside a name (AUC0-24, CYP2D6, t1/2), so it "
                f"is not read as a number here; read it on the page image.")
    return (f"the cited number {literal!r} is part of the label {label!r}: digits inside a word led by a letter "
            f"(AUC0-24, CYP2D6, t1/2) are a name, never a value; cite a number printed on its own.")


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
