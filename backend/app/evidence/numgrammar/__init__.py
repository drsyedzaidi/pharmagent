"""The numeric grammar: a number is readable only when it matches a small explicit grammar made of an explicit
allowed alphabet; every other character inside a numeric expression, or within one visual separator of it, makes the
expression unresolved (see docs/evidence-pipeline.md, "How numbers are matched")."""
from __future__ import annotations

from app.evidence.numgrammar.alphabet import CID_MARKER, CONTEXT, GAP, INSIDE, NUMBER_CHARS, cls
from app.evidence.numgrammar.api import (
    claim_decimal,
    digit_group,
    digits_of,
    find_n_lead,
    is_data_cell,
    number_spans,
    numbers_in,
    parse_claim,
    unreadable,
    words_after,
    words_before,
)
from app.evidence.numgrammar.equivalents import EQUIV
from app.evidence.numgrammar.tok import Tok
from app.evidence.numgrammar.tokenizer import tokens

__all__ = ["CID_MARKER", "CONTEXT", "EQUIV", "GAP", "INSIDE", "NUMBER_CHARS", "Tok", "claim_decimal", "cls",
           "digit_group", "digits_of", "find_n_lead", "is_data_cell", "number_spans", "numbers_in", "parse_claim",
           "tokens", "unreadable", "words_after", "words_before"]
