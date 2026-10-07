"""A cache that lives for ONE request: the normalised page and the tokens of a page or a cell, shared by every
record that cites it.

A table page with hundreds of records read the same page text again for every record (a normalisation and a
tokenisation each, twice). ``check_proposals`` (one extraction or one re-verification) opens ``read_cache()``; every
``cached()`` call inside it reads and fills one dict, which dies with the request. Outside a request nothing is cached,
so a bare ``verify()`` call behaves exactly as before. The cache is bounded by the characters it holds, never by the
number of entries: a dense page is cached once, a document of many pages never holds more than ``MAX_WEIGHT``
characters of them (the dict is emptied when a new entry would pass the budget).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, NamedTuple

from app.evidence.numgrammar import Tok, tokens

MAX_WEIGHT = 800_000            # characters of text whose readings one request keeps (~100 MB of tokens at worst)
_ACTIVE: ContextVar[list | None] = ContextVar("evidence_read_cache", default=None)


@contextmanager
def read_cache() -> Iterator[None]:
    """Open the cache of one request (re-entrant: an inner call shares the outer cache)."""
    if _ACTIVE.get() is not None:
        yield
        return
    token = _ACTIVE.set([{}, 0])
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def cached[T](key: tuple[Any, ...], make: Callable[[], T], weight: int) -> T:
    """``make()`` once per *key* while a request cache is open (*weight*: characters of text it was made from)."""
    state = _ACTIVE.get()
    if state is None:
        return make()
    store, held = state
    if key in store:
        return store[key]
    value = make()
    if weight + held > MAX_WEIGHT:
        store.clear()
        held = 0
    store[key] = value
    state[1] = held + weight
    return value


class Reading(NamedTuple):
    """The tokens of one text and their start offsets (for a bisect to the token that holds an offset)."""
    toks: tuple[Tok, ...]
    starts: tuple[int, ...]


def read(text: str, style: str, mode: str) -> Reading:
    """``tokens(text, style, mode=mode)`` of a page text, column or cell, once per request."""
    def make() -> Reading:
        toks = tuple(tokens(text, style, mode=mode))
        return Reading(toks, tuple(t.start for t in toks))
    return cached(("tokens", text, style, mode), make, len(text))
