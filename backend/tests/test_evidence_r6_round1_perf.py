"""r6 round-1 regressions on cost: work that was quadratic in the page (or in the records of a page), the cache that kept
27 MB per dense page, and the NFKC expansion that turned one glyph into eighteen characters. Each timing limit is
orders of magnitude above the fixed cost and below the old one, scaled while a tracer is installed (evidence_timing);
the structural assertions (a call count) are what a tracer cannot distort."""
from __future__ import annotations

import random
import time

from app.evidence import dose_rule
from app.evidence import normalize as N
from app.evidence import read_cache as RC
from app.evidence import value_checks as VC
from app.evidence import verify_columns as VCOL
from app.evidence.dose_rule import find_dose
from app.evidence.extract_rules import rule_extract
from app.evidence.proposal_check import check_proposals
from app.evidence.verify import verify
from tests.evidence_r6_round1_support import doc, run
from tests.evidence_timing import limit


def test_find_dose_folds_the_text_once_not_once_per_number(monkeypatch):
    calls = []
    real = dose_rule.fold
    monkeypatch.setattr(dose_rule, "fold", lambda text, **kw: (calls.append(len(text)), real(text, **kw))[1])
    text = "1 mg " * 20_000
    t0 = time.perf_counter()
    out = find_dose(text)
    assert len(out) == 20_000
    assert len(calls) == 1 and calls[0] == len(text)
    assert time.perf_counter() - t0 < limit(3.0)             # 100 000 characters took 6 s when it folded per number


def test_a_dense_page_costs_a_dose_rule_about_a_second(monkeypatch):
    template = {"fields": [{"name": "dose", "type": "number"}]}
    page = doc("1 " * 100_000)                                # 200 000 glyphs: the per-page cap
    t0 = time.perf_counter()
    assert rule_extract(page, template, "S1")["proposals"] == []
    assert time.perf_counter() - t0 < limit(8.0)             # 14 s before


def test_two_column_verification_reads_the_columns_once_however_many_times_the_snippet_occurs(monkeypatch):
    built = []
    real = VCOL.ColumnLines.__init__
    monkeypatch.setattr(VCOL.ColumnLines, "__init__", lambda self, cols: (built.append(1), real(self, cols))[1])
    cols = ["AUC was 12.5 mg\n" * 1500, "AUC was 12.5 mg\n" * 1500]
    page = {"pages": [{"n": 1, "text": "\n".join(cols), "columns": cols, "tables": []}], "decimal_style": "."}
    t0 = time.perf_counter()
    res = verify({"field": "x", "value": "12.5", "locator": {"page": 1, "kind": "text", "snippet": "AUC was 12.5 mg"}}, page)
    assert res.status == "pass" and len(built) == 1
    assert time.perf_counter() - t0 < limit(3.0)             # 58 s for 1000 occurrences before


def _table_doc(n: int):
    rnd = random.Random(3)
    cols = ["Subject", "Cmax", "AUC", "CL", "Vd", "t12", "Tmax"]
    rows = [cols] + [[f"S{i}"] + [f"{rnd.uniform(1, 999):.1f}" for _ in range(6)] for i in range(n)]
    template = {"fields": [{"name": c, "type": "number"} for c in cols[1:]]}
    text = "\n".join(" ".join(r) for r in rows)
    page = {"sha256": "x" * 64, "decimal_style": ".", "pages": [{"n": 1, "text": text, "small_text": "", "columns": [],
                                                                  "tables": [{"index": 0, "rows": rows, "strategy": "lines"}]}]}
    return page, template, text


def test_a_table_page_is_normalised_and_tokenised_once_per_request_not_once_per_record(monkeypatch):
    page, template, text = _table_doc(60)
    props = rule_extract(page, template, "S1")["proposals"]
    assert len(props) == 360
    seen = {"normalise": 0, "tokens": 0}
    real_norm, real_tokens = N._normalise, RC.tokens

    def count_norm(t):
        seen["normalise"] += t == text
        return real_norm(t)

    def count_tokens(t, style, mode="page", **kw):
        seen["tokens"] += t == text
        return real_tokens(t, style, mode=mode, **kw)

    monkeypatch.setattr(N, "_normalise", count_norm)
    monkeypatch.setattr(RC, "tokens", count_tokens)
    t0 = time.perf_counter()
    out = check_proposals(props, page, template)
    assert all(r["verify_status"] == "pass" for r in out)
    assert seen["normalise"] == 1 and seen["tokens"] <= 2           # 720 reads of the page text before
    assert time.perf_counter() - t0 < limit(5.0)


def test_the_cost_per_record_of_a_table_page_does_not_grow_with_the_number_of_rows():
    def per_record(n):
        page, template, _ = _table_doc(n)
        props = rule_extract(page, template, "S1")["proposals"]
        t0 = time.perf_counter()
        check_proposals(props, page, template)
        return (time.perf_counter() - t0) / len(props)

    small, large = per_record(40), per_record(320)
    assert large < max(small * 5, limit(0.004))                        # 20x the rows, not 20x the cost per record


def test_the_request_cache_dies_with_the_request_and_is_off_outside_one():
    made = []
    assert RC.cached(("k", "t"), lambda: made.append(1) or object(), 10) is not RC.cached(("k", "t"), lambda: object(), 10)
    with RC.read_cache():
        a = RC.cached(("k", "t"), lambda: made.append(1) or object(), 10)
        with RC.read_cache():                                           # re-entrant: shares the outer cache
            assert RC.cached(("k", "t"), lambda: object(), 10) is a
        assert RC.cached(("k", "t"), lambda: object(), 10) is a
    assert RC._ACTIVE.get() is None
    page, template, _ = _table_doc(5)
    check_proposals(rule_extract(page, template, "S1")["proposals"], page, template)
    assert RC._ACTIVE.get() is None


def test_the_request_cache_is_bounded_by_the_characters_it_holds_not_by_the_entries():
    with RC.read_cache():
        keep = RC.cached(("big", 1), lambda: object(), RC.MAX_WEIGHT - 10)
        assert RC.cached(("big", 1), lambda: object(), 1) is keep
        RC.cached(("big", 2), lambda: object(), 100)                    # would pass the budget: the cache is emptied first
        assert RC.cached(("big", 1), lambda: "again", 5) == "again"


def test_the_style_cache_of_old_value_checks_is_gone():
    assert not hasattr(VC, "_read") and not hasattr(VC, "lru_cache")


def test_a_character_that_expands_eighteen_times_does_not_balloon_the_locator():
    page = "ﷺ" * 100_000 + " Cmax was 12.5 ng/mL"
    t0 = time.perf_counter()
    norm, m = N.normalise(page)
    assert len(norm) == len(m) <= len(page) and norm.endswith("cmax was 12.5 ng/ml")
    res = run(page, "Cmax was 12.5 ng/mL", "12.5")
    assert (res.status, res.reason) == ("fail", "unresolved_context")   # the character is outside the alphabet and on the line
    assert time.perf_counter() - t0 < limit(6.0)                        # 3 s and 470 MB for ONE record before


def test_the_locator_keeps_the_numeric_alphabet_apart_and_only_spells_the_listed_ligatures():
    """Replaces test_nfkc_keeps_the_numeric_alphabet_apart_and_still_folds_ligatures (r6 round 2: the locator has no
    NFKC). Old assertion: nfkc_char left these eight characters alone, turned ﬁ into fi, ™ into TM, kept ﷺ whole and
    left a no-break space to the equivalents table. New: NOTHING but the equivalents, the seven ligatures and the
    lower case of the alphabet's letters is rewritten, so ™ and every other compatibility form stays itself too."""
    for ch in ("⁰", "¹", "₂", "１", "½", "ⓔ", "．", "Ⅹ", "™", "ﷺ", "㎎", "№", "ℓ"):
        assert N.normalise(f"a{ch}b")[0] == f"a{ch}b", hex(ord(ch))
    assert N.normalise("ﬁ")[0] == "fi" and N.normalise("ﬃ")[0] == "ffi"
    assert N.normalise("a\u00a0b")[0] == "a b"                     # an equivalent: the table handles it
