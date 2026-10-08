"""The hand-written WORD lists of the flag checks and of the keyless dose rule (r7 round 2: moved out of the number
reader).

They are lists of whole words, never characters, and the reader never reads them: ``tokens`` decides what a number is
from the four permitted character lists alone. A word list may only RAISE a flag or propose a record for a human to
verify; it can never make a number readable. Every list is matched with ``numgrammar.api.loose_in`` (a look-alike
spelling flags like the plain spelling), except the dose units (an exact unit word after a number).
"""
from __future__ import annotations

# value_checks.censored_flags: a word BEFORE the token that makes it a bound or an approximation ('below 12.4')
QUALIFIER_WORDS = frozenset({"below", "above", "under", "over", "approximately", "approx", "about", "circa", "ca",
                             "blq", "bloq", "lloq", "uloq"})
# the only look-backs: 'less than', 'more than', 'greater than', 'up to', 'at least', 'at most' and '... or equal to'
BOUND_WORDS = frozenset({"than", "to", "least", "most"})
BOUND_LEADS = frozenset({"less", "more", "greater", "up", "at"})
TO, OR, EQUAL = frozenset({"to"}), frozenset({"or"}), frozenset({"equal"})

# scale_shapes (a header, row label or unit: '(thousands)') and value_checks.scaled_flags; right AFTER the number the
# abbreviations count too ('1.5 bn', '1.5 lakh': never read in a header, where 'Mn' is manganese), and SCALE_SYMBOLS are
# matched exactly, case kept ('1.5 M', '1.5 k'; never 'mM' or 'm')
SCALE_WORDS = frozenset({"thousand", "thousands", "million", "millions", "billion", "billions", "trillion",
                         "trillions"})
AFTER_SCALE_WORDS = SCALE_WORDS | frozenset({"bn", "mn", "mio", "lakh", "lakhs", "crore", "crores"})
SCALE_SYMBOLS = frozenset({"k", "K", "M"})

# extract_rules (the keyless dose rule): the unit word after a number that makes it a dose candidate
DOSE_UNITS = frozenset({"mg", "μg", "ug", "g", "mg/kg", "μg/kg"})

__all__ = ["AFTER_SCALE_WORDS", "BOUND_LEADS", "BOUND_WORDS", "DOSE_UNITS", "EQUAL", "OR", "QUALIFIER_WORDS",
           "SCALE_SYMBOLS", "SCALE_WORDS", "TO"]
