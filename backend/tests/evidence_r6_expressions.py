"""The canonical expressions of the r6 exhaustive tests (one sentence, page fragment or table cell each) and the
helpers that read a variant of them: ``E[:b] + c + E[b:]`` for a boundary b and one code point c.

``|`` in a template marks a boundary: the expressions give about 300 (expression, boundary) pairs. The oracle that
judges the variants is in ``evidence_r6_oracle.py``.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from app.evidence.numgrammar import Tok, tokens

SEP = "\nok ok\n"
REF_OUTSIDE = "\ufffd"
REF_FORMER_LINE = "\u2014"         # EM DASH: a character of the deleted LINE list, outside since r7 (the derived rule)
CID_PANEL = ("(cid:1)", "(cid:2879)")


# ------------------------------------------------------------------- the expressions ---
class Expr(NamedTuple):
    name: str
    mode: str          # page | cell
    style: str
    text: str          # the expression
    cuts: tuple[int, ...]   # boundaries to test

    @property
    def pairs(self) -> list[tuple[str, int]]:
        return [(self.name, b) for b in self.cuts]


_TEMPLATES = [
    ("n_eq_12", "page", ".", "|n |= |1|2|"),
    ("cmax_dot", "page", ".", "|Cmax |was |1|2|.|5| |ng/mL|"),
    ("cmax_comma", "page", ",", "|Cmax |var |1|2|,|5| |ng/mL|"),
    ("auc_thousands", "page", ".", "|AUC |1|,|2|3|4| |ng.h/mL|"),
    ("auc_thin", "page", ".", "|AUC |1|8|\u2009|9|0|0| |ng|"),
    ("neg_hyphen", "page", ".", "|change |was |-|2|0| |mg|"),
    ("neg_minus", "page", ".", "|change |was |−|2|0| |mg|"),
    ("neg_endash", "page", ".", "|change |was |–|2|0| |mg|"),
    ("percent", "page", ".", "|response |in |4|5|%| |of|"),
    ("e_notation", "page", ".", "|k |was |1|.|2|e|-|3| |per h|"),
    ("sci_x_caret", "page", ".", "|Cmax |1.2| |x| |1|0|^|3| |cells|"),
    ("sci_times_caret_neg", "page", ".", "|Cmax |1.2| |×| |10|^|-|3| |cells|"),
    ("sci_times_sup", "page", ".", "|Cmax |1.2| |×| |10|⁻|³| |cells|"),
    ("bare_power", "page", ".", "|1|0|^|6| |cells|"),
    ("range", "page", ".", "|range |1|.|95|-|2|.|74| |L/h|"),
    ("pm", "page", ".", "|CL |5.3| |±| |0.4| |L/h|"),
    ("qual_glued", "page", ".", "|conc |<|0|.|5| |ng/mL|"),
    ("qual_spaced", "page", ".", "|conc |<| |0.5| |ng/mL|"),
    ("tilde_operator", "page", ".", "|∼|45|%| |of dose|"),
    ("dose_mg", "page", ".", "|Dose |150| |mg| |daily|"),
    ("lancet", "page", ".", "|Lancet |1|2|·|4| |units|"),
    ("degree", "page", ".", "|stored at |3|7|◦|C|"),
    ("sbp_table", "page", ".", "|SystolicBP |–|11.8|*| |–|14.2|***|"),
    ("mixed_number", "page", ".", "|12| |3|/|4|"),
    ("pua_inside", "page", ".", "|12|.|\ue000|5|"),
    ("braille_qualifier", "page", ".", "|<|\u2800|12|"),
    ("ten_dose", "page", ".", "|2 |x| |10| |mg|"),
    ("mult_daily", "page", ".", "|2 |×| |daily|"),
    ("detached", "page", ".", "|10| |^| |3|"),
    ("lookalike_ten", "page", ".", "|1.2 |×| |l0|³|"),
    ("cid_inside", "page", ".", "|31.1|(cid:1)|0.7|"),
    ("cid_lead", "page", ".", "|(cid:3)|85|%|"),
    ("glued_unit", "page", ".", "|150|mg|"),
    ("plus_glued", "page", ".", "|65|+| |years|"),
    ("cell_thousands", "cell", ".", "|3|,|412| |(|812|)|"),
    ("cell_12_145", "cell", ".", "|12| |145|"),
    ("cell_bad_groups", "cell", ".", "|1|,|23|,|456|"),
    ("cell_pm_groups", "cell", ".", "|18\u2009900| |±| |4\u2009100|"),
    ("cell_lancet", "cell", ".", "|12|·|4|"),
    ("cell_stars", "cell", ".", "|−|14.2|***|"),
    ("line_n_eq_12", "page", ".", "|fill\nok ok\nn = 12\nok ok\nfill|"),
    ("line_cmax", "page", ".", "|fill\nok ok\nCmax was 12.5 ng/mL\nok ok\nfill|"),
    ("line_sci", "page", ".", "|fill\nok ok\nCmax 1.2 x 10^3 cells\nok ok\nfill|"),
    # r7 (decision 5): the derived line rule. Every boundary two or more words from the number is on its line: each of the
    # 1.1 M code points there is either listed (the number keeps its reading) or outside (every number of the line is
    # unresolved, exactly like the reference outside character)
    ("line_far", "page", ".", "|AUC |was |measured |at |12.5 |mg/L |in |all |subjects|"),
    # r7 (decisions 6, 7, 8): the ratio production, an unmistakable citation and the chi-square form, swept like the rest
    ("ratio_glued", "page", ".", "|n |12|/|15|"),
    ("citation", "page", ".", "|efflux |[46]| |12 mg|"),
    ("chi_square", "page", ".", "|LRT |\u03c7\u00b2| = |3.84|"),
    # r7 writer 2 (decision 2): the unit production, every code point at every boundary of a glued unit (a chain with
    # denominators, a plain exponent, the 'per' connector, a unit the next word continues, a unit closed by a bracket)
    ("unit_chain", "page", ".", "|dose |2|.|5|m|g|/|k|g|/|h| |iv|"),
    ("unit_exponent", "page", ".", "|BSA |1|.|73|m|2| |ok|"),
    ("unit_per", "page", ".", "|took |5|m|g|p|e|r|k|g| |ok|"),
    ("unit_split", "page", ".", "|took |10|m| |L| |daily|"),
    ("unit_cell", "cell", ".", "|(|20|m|g|)|"),
    # (decision 3): label words whose digits must never act on a neighbour (a range end, a mixed number, a citation)
    ("label_range", "page", ".", "|AUC|0|-|24| |-|0.5| |h|"),
    ("label_ratio", "page", ".", "|by |CYP|2|D|6| |3|/|4| |of|"),
    ("label_cite", "page", ".", "|CYP|2|D|6| |[38]| |20 |trials|"),
]


def _build(name: str, mode: str, style: str, template: str) -> Expr:
    parts = template.split("|")
    cuts, pos = [], 0
    for part in parts[:-1]:
        pos += len(part)
        cuts.append(pos)
    return Expr(name, mode, style, "".join(parts), tuple(cuts))


EXPRESSIONS: list[Expr] = [_build(*t) for t in _TEMPLATES]
PAIRS: list[tuple[int, int]] = [(i, b) for i, e in enumerate(EXPRESSIONS) for b in e.cuts]



def sig(toks: Iterable[Tok]) -> tuple:
    """Structure of a token list: spans, shapes and flags, never values."""
    return tuple((t.start, t.end, t.shape, t.qualified, t.ranged) for t in toks)


def full(toks: Iterable[Tok]) -> tuple:
    return tuple((t.start, t.end, t.value, t.shape, t.parts, t.qualified, t.ranged, t.canonical, t.decimal)
                 for t in toks)


def golden_text(toks: Iterable[Tok]) -> str:
    return ";".join(f"{t.start}-{t.end}:{t.shape}{'q' if t.qualified else ''}{'r' if t.ranged else ''}"
                    f"{'=' + repr(t.value) if t.value is not None else ''}" for t in toks)


def variants(e: Expr, b: int, ins: str) -> str:
    return e.text[:b] + ins + e.text[b:]


def run(e: Expr, text: str) -> list[Tok]:
    return tokens(text, e.style, mode=e.mode)
