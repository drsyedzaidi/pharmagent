"""r6 numeric grammar: invariants on random text (the tokenizer never raises, every expression is well formed,
gap padding changes nothing, a hostile character two lines away changes nothing)."""
from __future__ import annotations

import pytest

from app.evidence.numgrammar import tokens

# ---------------------------------------------------------------- invariants on random text ---
_ALPHABET = (list("0123456789") * 6 + list(".,'-+ \t\n\r") * 3 + list("eExX*^%()[]/=:;<>~") * 2
             + [" ", "·", "×", "±", "−", "–", "¹", "²", "⁻", "‰",
                "≤", "°", "m", "g", "L", "n", "μ", "≈", "10", "10^3", "x10", "E-03"])
_HOSTILE = ["†", "​", "�", " ", "(cid:1)", "(cid:23)", "", "®", "\ud800"]


def _random_texts(n: int, seed: int, extra=()):
    import random
    rnd = random.Random(seed)
    alphabet = _ALPHABET + list(extra)
    return ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 40))) for _ in range(n)]


@pytest.mark.parametrize("mode", ["page", "claim", "cell"])
@pytest.mark.parametrize("style", [".", ","])
def test_random_text_never_raises_and_every_expression_is_well_formed(mode, style):
    import math
    shapes = {"int", "decimal", "foreign_decimal", "style_dependent", "grouped", "grouped_decimal", "middot", "sci",
              "bare_power", "ratio"}                                    # r7: the explicit ratio production
    for s in _random_texts(1500, 7, _HOSTILE):
        end = -1
        for t in tokens(s, style, mode=mode):
            assert 0 <= t.start < t.end <= len(s) and t.start >= end, (s, t)
            end = t.end
            if t.value is not None:
                assert math.isfinite(t.value) and t.canonical and t.shape in shapes, (s, t)
            else:
                assert t.shape.startswith(("unresolved_context", "unresolved_shape:")) and t.detail, (s, t)


def test_a_run_of_gap_characters_of_any_length_and_mix_never_changes_the_reading():
    import random
    import re
    rnd = random.Random(11)
    for s in _random_texts(1500, 3):
        want = [(t.value, t.shape, t.qualified, t.ranged) for t in tokens(s)]
        padded = re.sub("[ \t\n\r]+", lambda m: "".join(rnd.choice(" \t\n\r") for _ in range(rnd.randint(1, 6))), s)
        got = [(t.value, t.shape, t.qualified, t.ranged) for t in tokens(padded)]
        assert got == want, (s, padded)


def test_an_unread_character_two_lines_away_from_every_number_changes_nothing():
    import random
    rnd = random.Random(5)
    for s in _random_texts(800, 9):
        hostile = rnd.choice(_HOSTILE)
        base = tokens(s + "\nok ok\n12 mg")
        assert tokens(s + "\nok ok\n12 mg\nok ok\n" + hostile) == base, (s, hostile)
