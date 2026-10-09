"""The hand-written WORD lists of the flag checks and of the keyless dose rule (r7 round 2: moved out of the number
reader).

They are lists of whole words, never characters, and the reader never reads them: ``tokens`` decides what a number is
from the four permitted character lists alone. A word list may only RAISE a flag or propose a record for a human to
verify; it can never make a number readable. Every list is matched with ``word_match.loose_in`` (a look-alike
spelling flags like the plain spelling), except the dose units (an exact unit word after a number).
"""
from __future__ import annotations

# value_checks.censored_flags: a word BEFORE the token that makes it a bound or an approximation ('below 12.4')
# r7 round 3 adds around 8, roughly 6, nearly 4 and within 11 (the word right before a resolved number of the 60-paper
# design corpus, r7/r3/census_after.py) and the pharmacopoeial NMT / NLT (not more / less than: 0 in the corpus)
# r7 round 4 (the round-4 audit) adds the bound words of dosing text and the operator abbreviations of SAS / CDISC
# listings. Corpus counts (the word right before a resolved number of the 60-paper design corpus, r7/r4/census_r4.py):
# maximum 13, minimum 12, max 5, exceed 2, exceeds 1, beyond 2, min 1, le 2 and gt 2 (a broken 'T ab le 2' and the R
# package 'gt': two false flags), exceeding / exceeded / almost / upwards / ge / lt / gte / lte 0 (standard spellings).
# The HTML entities '&gt;' / '&lt;' occur nowhere in the parse caches and are not listed.
QUALIFIER_WORDS = frozenset({"below", "above", "under", "over", "approximately", "approx", "about", "circa", "ca",
                             "blq", "bloq", "lloq", "uloq", "around", "roughly", "nearly", "within", "nmt", "nlt",
                             "exceed", "exceeds", "exceeding", "exceeded", "maximum", "max", "minimum", "min",
                             "upwards", "almost", "beyond", "ge", "gt", "le", "lt", "gte", "lte"})
# the look-backs: 'less than', 'more than', 'greater than', 'up to', 'at least', 'at most' and '... or equal to'; r7
# round 3: any comparative before 'than' (a letter word ending in COMPARATIVE_END: 'fewer than', 'lower than'); r7 round
# 4: 'in excess of' 2 in the corpus, 'upwards of', 'a minimum of', 'a maximum of'
BOUND_WORDS = frozenset({"than", "to", "least", "most", "of"})
BOUND_LEADS = frozenset({"less", "more", "greater", "up", "at", "excess", "upwards", "minimum", "maximum"})
# the range words between two numbers (value_ranges.py): 'to' 173 and r7 round 4 'through' 8 between two resolved
# numbers of the corpus; 'thru', 'until', 'till' 0 (standard spellings of dosing schedules: 'days 1 through 5')
TO = frozenset({"to", "through", "thru", "until", "till"})
OR, EQUAL = frozenset({"or"}), frozenset({"equal"})
# r7 round 4: a sign spelled out before the number ('minus 12%', 'negative 1.5'; corpus: plus 3, minus 0, negative 0)
SIGN_WORDS = frozenset({"minus", "negative", "plus"})
THAN, AND = frozenset({"than"}), frozenset({"and"})
COMPARATIVE_END = "er"
BETWEEN = frozenset({"between"})                      # 'between 12 and 24': both numbers are range ends (61 in corpus)
# r7 round 3: a bound printed AFTER the number, behind 'or' / 'and' ('0.5 or less', '18 or older', '12 and above';
# corpus: 'or higher' 4, 'or more' 3, 'or fewer' 1, 'and earlier' 1)
# r7 round 4: 'longer' and 'shorter', the duration comparatives beside 'earlier' and 'later' (0 in the corpus; the audit
# printed '12 h or longer')
AFTER_BOUNDS = frozenset({"less", "more", "fewer", "greater", "lower", "higher", "older", "younger", "above", "below",
                          "over", "under", "earlier", "later", "longer", "shorter"})

# scale_shapes (a header, row label or unit: '(thousands)') and value_checks.scaled_flags; right AFTER the number the
# abbreviations count too ('1.5 bn', '1.5 lakh': never read in a header, where 'Mn' is manganese), and SCALE_SYMBOLS are
# matched exactly, case kept ('1.5 M', '1.5 k'; never 'mM' or 'm')
# r7 round 3 (the round-3 audit): 'hundred(s)' (a header '(hundreds)'), and after the number the abbreviations
# 'mln', 'mil', 'bil', 'bln', 'lac(s)' and the symbols 'B' and 'MM'. Corpus counts (60-paper design corpus, the word
# right after a resolved number of the page text, r7/r3/census_after.py): every one 0 ('million' 2, 'K' 10; 'mm' 53 is
# millimetres, never read: SCALE_SYMBOLS keep their case). They are the standard spellings of the scales the audit
# printed right after a value ('1.5 mln patients', 'sales of 1.5 B'): a word list only ever raises a flag.
SCALE_WORDS = frozenset({"hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion", "billions",
                         "trillion", "trillions"})
AFTER_SCALE_WORDS = SCALE_WORDS | frozenset({"bn", "mn", "mio", "lakh", "lakhs", "crore", "crores", "mln", "mil", "bil",
                                             "bln", "lac", "lacs"})
SCALE_SYMBOLS = frozenset({"k", "K", "M", "B", "MM"})

# extract_rules (the header keys of a table, r7 round 4: moved here from extract_rules.py, word lists of header WORDS)
VALUE_HEADERS = frozenset({"value", "values", "estimate", "estimates", "mean", "result", "results",
                           "pointestimate", "geometricmean", "median", "typicalvalue", "all", "total", "overall"})
UNIT_HEADERS = frozenset({"unit", "units"})
DISPERSION_HEADERS = frozenset({"sd", "se", "sem", "rse", "cv", "cv%", "%cv", "rse%", "%rse", "ci", "95%ci",
                                "90%ci", "range", "iqr", "min", "max", "p", "pvalue", "n", "shrinkage", "iiv"})

# dose_rule (the keyless dose rule): the unit word after a number that makes it a dose candidate
DOSE_UNITS = frozenset({"mg", "μg", "ug", "g", "mg/kg", "μg/kg"})

__all__ = ["AFTER_BOUNDS", "AFTER_SCALE_WORDS", "AND", "BETWEEN", "BOUND_LEADS", "BOUND_WORDS", "COMPARATIVE_END",
           "DISPERSION_HEADERS", "DOSE_UNITS", "EQUAL", "OR", "QUALIFIER_WORDS", "SCALE_SYMBOLS", "SCALE_WORDS",
           "SIGN_WORDS", "THAN", "TO", "UNIT_HEADERS", "VALUE_HEADERS"]
