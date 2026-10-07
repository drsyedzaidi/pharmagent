"""Helpers that only the tests call. They used to live in ``app/evidence`` (``verify.find_number_in_snippet``,
``numgrammar.find_n`` / ``token_at`` / ``words_around``) with no caller in the application (r6 round-2 audit); they are
thin readings over ``tokens`` and the word scanner, and keep the tests that exercised them meaningful.
"""
from __future__ import annotations

from app.evidence import models as M
from app.evidence.numgrammar import Tok, find_n_lead, tokens
from app.evidence.numgrammar.api import _next_word, _prev_word, _word_bounds
from app.evidence.numgrammar.equivalents import fold
from app.evidence.verify import _number_check


def token_at(text: str, a: int, b: int, style: str = ".", *, mode: str = "page") -> Tok | None:
    """The token or unresolved expression whose span is exactly text[a:b], else None."""
    return next((t for t in tokens(text, style, mode=mode) if (t.start, t.end) == (a, b)), None)


def words_around(text: str, start: int, end: int) -> tuple[str, str]:
    """(W-, W+): the whole words one gap run before and after the word that holds text[start:end] ('' at an edge).
    A run of gap characters of any length and mix is ONE separator, so padding never widens the window."""
    t = fold(text)
    a, _ = _word_bounds(t, start)
    _, b = _word_bounds(t, max(end - 1, start))
    prev, nxt = _prev_word(t, a), _next_word(t, b)
    return (t[prev[0]:prev[1]] if prev else "", t[nxt[0]:nxt[1]] if nxt else "")


def find_n(text: str) -> list[tuple[int, int, int]]:
    """(start, end, n) of every ``n = <integer of 1-6 digits>``: ``n = 12``, ``n=12``, ``(n=12)``, ``N=12``."""
    return [(s, e, n) for _, s, e, n in find_n_lead(text)]


def find_number_in_snippet(value: str, snippet: str, decimal_style: str, integer: bool) -> str | None:
    """The literal token in *snippet* equal to *value* (parsed with *decimal_style*), else None."""
    span, reason, _ = _number_check(value, snippet, decimal_style, integer)
    return snippet[span[0]:span[1]] if span is not None and reason == M.REASON_OK else None
