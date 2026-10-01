"""Briefing-memo trace check: no number may appear in the memo unless a state value produced it.

The memo is a TEMPLATE filled from PharmState. A language model never writes a
number. This module is the deterministic guard that makes that checkable:

* every numeric value the memo prints is recorded in a *value table*
  (``TracedValue``: the PharmState path, the text exactly as rendered, and the
  audit-entry index of the tool run that produced it);
* ``memo_untraced_numbers(memo_text, value_table)`` extracts every numeric token
  from the generated memo and returns those the table does not contain.

Exact semantics (also what the tests pin):

1. *Tags.* A trace tag is exactly ``[#<ASCII digits>]`` or ``[#?]``. Tags are
   replaced by a space before scanning (so neighbours cannot fuse); look-alikes
   such as ``[# 12]``, ``[#12.5]`` or fullwidth digits are not tags, so a number
   cannot be hidden in one. ``memo_bad_trace_tags`` flags tags whose index is
   not an existing audit entry.
2. *Literals.* Table entries of kind ``literal`` (and non-numeric plain strings)
   are verbatim strings the deterministic tools emitted -- a QC check detail, a
   model label. Each is masked out of the memo before scanning, so digits inside
   it are not counted. Masking is exact and token-bounded: the literal must
   contain a letter (punctuation-only literals mask nothing), and it matches only
   where no word character touches either end, so it cannot eat part of a number
   or of a longer word. A numeric-only literal is treated as a number instead.
3. *Tokens.* A numeric token is a standalone number: optional sign, digits with an
   optional fraction (or a leading-dot fraction), optional exponent, plus any
   malformed continuation (``1.2.3``, ``1_000``) kept in the same token so it
   cannot match a value. It must not directly follow a word character, digit,
   underscore or dot, so identifiers (``ds_978f9ca8``, ``AUC0``, ``theta1``) are
   not numbers while ``10mg`` yields ``10``. The Unicode minus and its
   look-alikes (U+2212, U+FE63, U+FF0D) are read as ``-``; an en dash or a hyphen
   between two numbers is a separator, not a sign. Number WORDS ("ninety") are
   not numerals and are out of scope: free text from a model is therefore kept
   out of the memo at its sources (title and treatment labels reject any numeral,
   study metadata with digits is withheld).
4. *Membership.* A token is traced iff it is numerically EQUAL (``Decimal``) to the
   rendered text of some numeric table entry. ``8.650`` matches ``8.65``;
   ``8.6`` does not (no rounding tolerance -- a bare ``1`` must not be excused by
   ``0.7``); the sign matters.

What this does and does not prove: it proves the memo contains no number that is
absent from the state-derived table (an invented or stale figure is caught). It
is membership, not per-cell provenance; the per-cell audit tag carries the
specific source, and it names the LATEST audited run of the producing tool (some
tools audit only part of what they write to state). Digits in the template's own
wording would be flagged, so the template is written without any.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.audit import hash_payload

TAG_RE = re.compile(r"\[#(?:[0-9]+|\?)\]")
_TAG_INDEX_RE = re.compile(r"\[#([0-9]+)\]")
NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:[_.]\d[\w.]*)?")
_MINUS_LOOKALIKES = str.maketrans({"\u2212": "-", "\ufe63": "-", "\uff0d": "-"})

#: Audit entries by these agents are never a result source: the supervisor logs
#: propose/approve/reject for a tool NAME, and system logs session bookkeeping.
NON_SOURCE_AGENTS = frozenset({"supervisor", "system"})

#: PharmState section -> the tools whose audited run produces it (latest run wins).
SOURCE_TOOLS: dict[str, tuple[str, ...]] = {
    "dataset_metadata": ("load_dataset",),
    "data_quality": ("profile_pk_dataset",),
    "nca_summary": ("compute_nca",),
    "qc": ("run_qc",),
    "pk_model_results": ("fit_pk_model",),
    "nlme_results": ("run_nlme",),
    "scm_results": ("run_scm",),
    "forest_results": ("run_covariate_forest",),
    "vpc_results": ("run_vpc",),
    "diagnostics_results": ("run_diagnostics",),
    "indirect_results": ("indirect_comparison",),
    "er_results": ("fit_exposure_response", "bootstrap_exposure_response"),
    "dose_selection_results": ("select_optimal_dose",),
}

#: Sources computed on the loaded dataset: a run older than the latest load_dataset
#: entry was computed on a dataset that is no longer the current one.
DATASET_BOUND: tuple[str, ...] = tuple(
    k for k in SOURCE_TOOLS if k not in ("dataset_metadata", "indirect_results"))


class MemoTraceError(ValueError):
    """The memo contains numbers (or trace tags) that state does not account for."""


@dataclass(frozen=True)
class TracedValue:
    """One value the memo prints, and where it came from."""
    key: str                      # PharmState path, e.g. "nca_summary.CL_F.geomean"
    text: str                     # exactly as rendered in the memo (no tag)
    audit_index: int | None       # audit entry that produced it; None = not found
    kind: str = "number"          # "number" (computed) | "input" (user-supplied) | "literal"


# ── tokenisation and membership ────────────────────────────────────────────────
def strip_tags(text: str) -> str:
    """Remove trace tags, leaving a space so the text on either side cannot fuse."""
    return TAG_RE.sub(" ", text or "")


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _mask_literals(text: str, literals: Iterable[str]) -> str:
    """Blank out each literal where it stands as a whole token (see module docstring)."""
    usable = sorted({lit for lit in literals if lit and _has_letter(lit)}, key=len, reverse=True)
    if not usable:
        return text
    pattern = "|".join(r"(?<!\w)" + re.escape(lit) + r"(?!\w)" for lit in usable)
    return re.sub(pattern, " ", text)


def extract_numbers(text: str, literals: Iterable[str] = ()) -> list[str]:
    """Numeric tokens of ``text`` in order, after masking literals and removing tags."""
    body = strip_tags(_mask_literals((text or "").translate(_MINUS_LOOKALIKES), literals))
    return [m.group(0) for m in NUMBER_RE.finditer(body)]


def _decimal(text: str) -> Decimal | None:
    try:
        d = Decimal(text.strip().translate(_MINUS_LOOKALIKES))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _split_table(value_table: Iterable[Any]) -> tuple[set[Decimal], list[str]]:
    """(numeric values, literal strings) from a mixed value table."""
    numbers: set[Decimal] = set()
    literals: list[str] = []
    for item in value_table:
        if isinstance(item, TracedValue):
            text, is_literal = item.text, item.kind == "literal"
        elif isinstance(item, bool):
            continue
        elif isinstance(item, (int, float)):
            text, is_literal = repr(item), False
        elif isinstance(item, str):
            text, is_literal = item, False
        else:
            continue
        dec = _decimal(text)
        if dec is not None and not (is_literal and _has_letter(text)):
            numbers.add(dec)           # a number, or a numeric-only literal
        elif is_literal or isinstance(item, str):
            literals.append(text)
    return numbers, literals


def memo_untraced_numbers(memo_text: str, value_table: Iterable[Any]) -> list[str]:
    """Numeric tokens in ``memo_text`` that the value table does not contain.

    Returned in order of first appearance, de-duplicated. Empty means every
    number in the memo is traced to a state value. See the module docstring for
    the exact tag, literal, token and equality rules.
    """
    numbers, literals = _split_table(value_table)
    untraced: list[str] = []
    for token in extract_numbers(memo_text, literals):
        dec = _decimal(token)
        if (dec is None or dec not in numbers) and token not in untraced:
            untraced.append(token)
    return untraced


def memo_bad_trace_tags(memo_text: str, n_audit_entries: int) -> list[str]:
    """Trace tags whose index is not an existing audit entry (index >= n_audit_entries)."""
    bad: list[str] = []
    for m in _TAG_INDEX_RE.finditer(memo_text or ""):
        if int(m.group(1)) >= n_audit_entries and m.group(0) not in bad:
            bad.append(m.group(0))
    return bad


# ── audit lookup ───────────────────────────────────────────────────────────────
class AuditTrace:
    """Read-only view of the audit entries that existed when the memo was built."""

    def __init__(self, entries: Iterable[Mapping[str, Any]] = ()) -> None:
        self._entries = tuple(dict(e) for e in entries)

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    def _sources(self):
        return (e for e in reversed(self._entries) if e.get("agent") not in NON_SOURCE_AGENTS)

    def latest(self, tools: Iterable[str]) -> int | None:
        """Index of the most recent result-producing entry for any of ``tools``."""
        wanted = set(tools)
        return next((int(e["index"]) for e in self._sources() if e.get("tool") in wanted), None)

    def by_output(self, tool: str, payload: Any) -> int | None:
        """Index of the entry of ``tool`` whose audited output is exactly ``payload``."""
        return self.by_digest(tool, hash_payload(payload))

    def by_digest(self, tool: str, digest: str) -> int | None:
        """Index of the latest entry of ``tool`` whose audited output hash is ``digest``."""
        return next((int(e["index"]) for e in self._sources()
                     if e.get("tool") == tool and e.get("outputs_hash") == digest), None)

    def before_latest_load(self, index: int | None) -> bool:
        """True when entry ``index`` is older than the latest load_dataset entry, i.e. that
        run used a previously loaded dataset. An unknown index cannot be dated: False."""
        load = self.latest(("load_dataset",))
        return index is not None and load is not None and index < load

    def index_for(self, field: str) -> int | None:
        return self.latest(SOURCE_TOOLS.get(field, ()))

    def predates_dataset(self, field: str) -> bool:
        """True when ``field``'s latest producing run is older than the latest
        load_dataset entry, i.e. it was computed on a previously loaded dataset."""
        return field in DATASET_BOUND and self.before_latest_load(self.index_for(field))
