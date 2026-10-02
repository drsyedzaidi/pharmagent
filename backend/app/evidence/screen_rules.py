"""Deterministic keyword screening (the keyless path and the fallback when an LLM answer is
invalid). Terms come ONLY from the saved criteria (their keywords, else their text minus
stopwords), never from the document, so text inside a PDF cannot add a matching term.
Matches are whole-token (suffix-stripped) against the normalised page text and reported as
raw-offset locators, exactly like verified quotes.
"""
from __future__ import annotations

import re
from typing import Any

from app.evidence.normalize import normalise

STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or that the this to was were with
which who whom whose will would should could may might must can not no nor any all each either both
than then there their they them these those such via per within without about above below between
study studies report reports reported reporting describe describes described include includes included
exclude excludes excluded paper papers article articles publication publications data value values
""".split())
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_TERM_RE = re.compile(r"[^\W\d_][^\W_]*(?:-[^\W_]+)*", re.UNICODE)
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


def find_term(term: str, text: str) -> list[int] | None:
    """Raw [start, end) of the first whole-token match of *term* (phrases allowed) in *text*."""
    want = stem_phrase(term)
    if not want:
        return None
    stems, spans = _tokens(text)
    n = len(want)
    for i in range(len(stems) - n + 1):
        if tuple(stems[i:i + n]) == want:
            return [spans[i][0], spans[i + n - 1][1]]
    return None


def _page_hits(term: str, page: dict[str, Any]) -> dict[str, Any] | None:
    """Locator for *term* on a page: page text first, then each column text (column set)."""
    text = page.get("text") or ""
    span = find_term(term, text)
    if span is not None:
        return {"page": page["n"], "term": term, "snippet": _snippet(text, *span), "span": span, "column": None}
    for ci, col in enumerate(page.get("columns") or []):
        span = find_term(term, col or "")
        if span is not None:
            return {"page": page["n"], "term": term, "snippet": _snippet(col, *span), "span": span, "column": ci}
    return None


def _criterion_evidence(criterion: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in criterion_terms(criterion):
        for page in pages:
            hit = _page_hits(term, page)
            if hit is not None:
                out.append(hit)
                break                                   # first page per term is enough
        if len(out) >= MAX_EVIDENCE_PER_CRITERION:
            break
    return out


def _describe(entry: dict[str, Any]) -> str:
    cid, met = entry["criterion_id"], entry["met"]
    if not met:
        return f"{cid} not met"
    where = ", ".join(f"{ev['term']} (page {ev['page']})" for ev in entry["evidence"])
    return f"{cid} met — {where}"


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
    entries = []
    for c in criteria:
        evidence = _criterion_evidence(c, pages)
        entries.append({"criterion_id": c["id"], "kind": c["kind"], "required": bool(c.get("required")),
                        "met": bool(evidence), "evidence": evidence, "note": ""})
    verdict, score, why = _verdict(entries)
    reasoning = ("Keyword screen (deterministic): " + "; ".join(_describe(e) for e in entries)
                 + f". Verdict {verdict}: {why}.")
    return {"verdict": verdict, "score": score, "reasoning": reasoning, "criteria": entries}
