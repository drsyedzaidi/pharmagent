"""The text a reader sees next to a token: a window of VISIBLE characters read from the raw text.

A qualifier ('<', 'below', '≤') or range bound must be found whatever invisible characters, combining
marks or long whitespace lie between it and the number, so the window is counted in visible characters
(whitespace runs count one) and not in raw ones. Unreadable characters (private use, noncharacter,
replacement, lone surrogate: a text layer's unmapped glyph) are skipped like invisible ones. Walking is
capped at MAX_STEPS raw characters: when the cap is hit before the window is full the caller must treat
the context as unreadable (fail closed) rather than as empty.
"""
from __future__ import annotations

import unicodedata

from app.evidence.invisible import kind

MAX_STEPS = 2048


def ignorable(ch: str) -> bool:
    """True for a character that draws nothing here or cannot be read: invisible, combining, or unreadable."""
    if ch < "\x80":
        return kind(ch) != 0
    cp = ord(ch)
    return (kind(ch) != 0 or ch in "￼�" or 0x80 <= cp <= 0x9F or unicodedata.category(ch) in ("Co", "Cn", "Cs"))


def _walk(text: str, indexes: range, limit: int) -> tuple[list[str], bool]:
    out: list[str] = []
    for steps, i in enumerate(indexes, 1):
        if steps > MAX_STEPS:
            return out, True
        ch = text[i]
        if ignorable(ch):
            continue
        if ch.isspace():
            if out and out[-1] == " ":
                continue
            ch = " "
        out.append(ch)
        if len(out) >= limit:
            break
    return out, False


def visible_before(text: str, pos: int, limit: int) -> tuple[str, bool]:
    """(the last *limit* visible characters before text[pos], whether the walk hit MAX_STEPS first)."""
    out, capped = _walk(text, range(pos - 1, -1, -1), limit)
    return "".join(reversed(out)), capped


def visible_after(text: str, pos: int, limit: int) -> tuple[str, bool]:
    """(the first *limit* visible characters from text[pos], whether the walk hit MAX_STEPS first)."""
    out, capped = _walk(text, range(pos, len(text)), limit)
    return "".join(out), capped
