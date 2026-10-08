"""Deterministic keyword screening (the keyless path and the fallback when an LLM answer is
invalid). Terms come ONLY from the saved criteria (their keywords, else their text minus
stopwords), never from the document, so text inside a PDF cannot add a matching term.
Matches are whole-token (suffix-stripped) against the normalised page text and reported as
raw-offset locators, exactly like verified quotes. Keywords are alternatives (any one meets the
criterion); text-derived words must mostly co-occur within each 'or'-separated part. Each page is
normalised and tokenised once per screen, not once per term.
"""
from __future__ import annotations

import re
from typing import Any

from app.evidence.normalize import normalise
from app.evidence.numgrammar.alphabet import DIGITS, LETTERS, char_class

STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or that the this to was were with
which who whom whose will would should could may might must can not no nor any all each either both
than then there their they them these those such via per within without about above below between
study studies report reports reported reporting describe describes described include includes included
exclude excludes excluded paper papers article articles publication publications data value values
""".split())
# keyword tokens (never a value): runs of the alphabet's letters, runs of ASCII digits (r7 round 2: built from the
# reader's lists; a Unicode word class made every non-ASCII digit a letter); any other character separates tokens
_LETTER = char_class(LETTERS, negate=False)
_ALNUM = char_class(LETTERS | DIGITS, negate=False)
_TOKEN_RE = re.compile(_LETTER + "+|[0-9]+")
_TERM_RE = re.compile(_LETTER + _ALNUM + "*(?:-" + _ALNUM + "+)*")
_OR_RE = re.compile("(?<!" + _LETTER + ")or(?!" + _LETTER + ")|;", re.IGNORECASE)
MAX_SNIPPET, MAX_EVIDENCE_PER_CRITERION, MIN_TERM_LEN = 300, 6, 2


def stem(word: str) -> str:
    """Minimal plural stripping: ies->y, sses->ss, trailing s (never ss). 'rats'->'rat', 'rate'->'rate'."""
    w = word.casefold()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def stem_phrase(text: str) -> tuple[str, ...]:
    """Token stems of a term or a raw text slice ('In\\nVitro' == 'in vitro', 'Rats' == 'rat')."""
    return tuple(stem(t) for t in _TOKEN_RE.findall(normalise(text)[0]))


def criterion_terms(criterion: dict[str, Any]) -> list[str]:
    """The criterion's keywords, else the content words of its text (stopwords dropped)."""
    keywords = [k.strip() for k in (criterion.get("keywords") or []) if k and k.strip()]
    if keywords:
        return list(dict.fromkeys(keywords))
    words = [w.casefold() for w in _TERM_RE.findall(criterion.get("text") or "")]
    return list(dict.fromkeys(w for w in words if len(w) >= MIN_TERM_LEN and w not in STOPWORDS))


def _tokens(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """(stems, raw [start, end) spans) of the tokens of *text* after normalisation."""
    norm, idx = normalise(text)
    stems, spans = [], []
    for m in _TOKEN_RE.finditer(norm):
        stems.append(stem(m.group()))
        spans.append((idx[m.start()], idx[m.end() - 1] + 1))
    return stems, spans


def _snippet(text: str, start: int, end: int) -> str:
    """The raw line holding the match, trimmed to MAX_SNIPPET around it."""
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    le = len(text) if le < 0 else le
    if le - ls <= MAX_SNIPPET:
        return text[ls:le]
    half = (MAX_SNIPPET - (end - start)) // 2
    return text[max(ls, start - half):min(le, end + half)]


def _find(want: tuple[str, ...], stems: list[str], spans: list[tuple[int, int]]) -> list[int] | None:
    n = len(want)
    for i in range(len(stems) - n + 1):
        if tuple(stems[i:i + n]) == want:
            return [spans[i][0], spans[i + n - 1][1]]
    return None


Indexed = list[tuple[int, int | None, str, list[str], list[tuple[int, int]]]]   # page, column, text, stems, spans


def index_pages(pages: list[dict[str, Any]]) -> Indexed:
    """Every page text (then each column text) normalised and tokenised ONCE, reused for every term."""
    out: Indexed = []
    for page in pages:
        texts = [(None, page.get("text") or "")] + list(enumerate(page.get("columns") or []))
        for column, text in texts:
            out.append((page["n"], column, text or "", *_tokens(text or "")))
    return out


def _term_hit(term: str, indexed: Indexed) -> dict[str, Any] | None:
    """Locator for the first page (page text before its columns) holding *term*."""
    want = stem_phrase(term)
    if not want:
        return None
    for page_n, column, text, stems, spans in indexed:
        span = _find(want, stems, spans)
        if span is not None:
            return {"page": page_n, "term": term, "snippet": _snippet(text, *span), "span": span, "column": column}
    return None


def criterion_alternatives(criterion: dict[str, Any]) -> tuple[list[list[str]], bool]:
    """(alternatives, from_keywords). Keywords are alternatives (any one matches). Without keywords the
    criterion TEXT is split on 'or'; each part's content words must MOSTLY co-occur (a majority), so
    'human adult PK study' is not met by 'adult' alone, while 'in-vitro or animal' is met by either."""
    if [k for k in (criterion.get("keywords") or []) if k and k.strip()]:
        return [[t] for t in criterion_terms(criterion)], True
    parts = _OR_RE.split(criterion.get("text") or "")
    alts = [criterion_terms({"text": part}) for part in parts]
    return [a for a in alts if a], False


def needed(n_terms: int) -> int:
    return n_terms // 2 + 1


def _criterion_entry(criterion: dict[str, Any], indexed: Indexed) -> tuple[dict[str, Any], float]:
    """(entry, credit): credit is the best alternative's share of matched words (1.0 for a keyword)."""
    alts, from_keywords = criterion_alternatives(criterion)
    base = {"criterion_id": criterion["id"], "kind": criterion["kind"], "required": bool(criterion.get("required"))}
    if from_keywords:                                # any keyword meets it; every keyword found is evidence
        hits = [h for h in (_term_hit(alt[0], indexed) for alt in alts) if h is not None]
        return {**base, "met": bool(hits), "evidence": hits[:MAX_EVIDENCE_PER_CRITERION], "note": ""}, float(bool(hits))
    best: tuple[bool, float, list[dict[str, Any]], int, int] = (False, 0.0, [], 0, 0)
    for alt in alts:
        hits = [h for h in (_term_hit(t, indexed) for t in alt) if h is not None]
        cand = (len(hits) >= needed(len(alt)), len(hits) / len(alt), hits, len(hits), len(alt))
        if cand[:2] > best[:2]:
            best = cand
    met, share, hits, k, n = best
    note = f"{k} of {n} criterion words found (needs {needed(n)})" if n > 1 else ""
    return {**base, "met": met, "evidence": hits[:MAX_EVIDENCE_PER_CRITERION], "note": note}, share


def _describe(entry: dict[str, Any]) -> str:
    cid, met = entry["criterion_id"], entry["met"]
    where = ", ".join(f"{ev['term']} (page {ev['page']})" for ev in entry["evidence"])
    note = f" [{entry['note']}]" if entry["note"] else ""
    if not met:
        return f"{cid} not met" + (f" — only {where}" if where else "") + note
    return f"{cid} met — {where}{note}"


def _verdict(entries: list[dict[str, Any]]) -> tuple[str, float, str]:
    """(verdict, score, why). Exclusion fires -> exclude 0.0; all required inclusion met -> include
    with score = fraction of inclusion criteria met; nothing matched -> exclude; else unsure."""
    excl = [e["criterion_id"] for e in entries if e["kind"] == "exclusion" and e["met"]]
    incl = [e for e in entries if e["kind"] == "inclusion"]
    n_met = sum(1 for e in incl if e["met"])
    fraction = n_met / len(incl) if incl else 1.0
    if excl:
        return "exclude", 0.0, f"exclusion criterion {', '.join(excl)} matched"
    unmet_required = [e["criterion_id"] for e in incl if e["required"] and not e["met"]]
    if not unmet_required:
        why = f"{n_met}/{len(incl)} inclusion criteria met, no exclusion criterion matched"
        return "include", round(fraction, 4), why
    if n_met == 0:
        return "exclude", 0.0, "no inclusion criterion matched"
    return "unsure", round(fraction, 4), f"required criterion {', '.join(unmet_required)} not met"


def keyword_screen(criteria: list[dict[str, Any]], pages: list[dict[str, Any]]) -> dict[str, Any]:
    """{verdict, score, reasoning, criteria:[{criterion_id, kind, required, met, evidence, note}]}."""
    indexed = index_pages(pages)
    scored = [_criterion_entry(c, indexed) for c in criteria]
    entries = [e for e, _ in scored]
    verdict, score, why = _verdict(entries)
    incl = [credit for e, credit in scored if e["kind"] == "inclusion"]
    if verdict != "exclude" and incl:                # partial word matches lower the score
        score = round(min(score, sum(incl) / len(incl)), 4)
    reasoning = ("Keyword screen (deterministic; keywords are alternatives, a criterion without keywords "
                 "needs most of its words): " + "; ".join(_describe(e) for e in entries)
                 + f". Verdict {verdict}: {why}.")
    return {"verdict": verdict, "score": score, "reasoning": reasoning, "criteria": entries}
