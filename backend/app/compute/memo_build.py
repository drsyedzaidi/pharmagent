"""Assemble the briefing memo from PharmState (deterministic, no language model).

The memo is a TEMPLATE: the sections that appear are chosen from which analyses
exist in state; every missing one is omitted with a one-line 'not run' note, and a
result whose producing run predates the latest load_dataset (computed on another
dataset) is withheld with a note rather than mixed in. No
text is generated from a model, and every number is rendered through a recorder
that tags it with its audit entry and records it in the value table, which is
what ``memo_untraced_numbers`` later checks the finished memo against.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.compute.memo import DATASET_BOUND, AuditTrace, TracedValue
from app.compute.memo_model import (
    Block,
    Bullet,
    Heading,
    Para,
    Recorder,
    Section,
    blocks_text,
    trace_tag,
)
from app.compute.memo_sections import dataset, nca, qc, structural
from app.compute.memo_sections_er import current_er, dose_selection, exposure_response
from app.compute.memo_sections_pop import covariates, diagnostics, indirect, population
from app.compute.textguard import has_numeral, reject_numerals
from app.core.pharmstate import PharmState

DEFAULT_TITLE = "Briefing memo"
MAX_TITLE = 120
METHOD_NOTE = (
    "Every number in this memo is copied from computed analysis results. The tag after a "
    "number names the audit-trail entry of the tool run that produced it; an unknown tag "
    "means no matching entry was found. No number was written by a language model.")
METHOD_NOTE_WITH_INPUTS = (
    "Every number in this memo is copied from analysis state. Numbers in rows marked "
    "user-supplied are inputs as they were entered (for example effect estimates copied "
    "from the literature): they were not computed here, may have been transcribed by the "
    "chat assistant, and must be checked against their source. Every other number is a "
    "computed result. The tag after a number names the audit-trail entry of the tool run "
    "that recorded it; an unknown tag means no matching entry was found.")

Builder = Callable[[PharmState, AuditTrace, Recorder], "Section | None"]
BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("dataset", dataset), ("nca", nca), ("qc", qc), ("structural", structural),
    ("population", population), ("covariates", covariates), ("diagnostics", diagnostics),
    ("indirect", indirect), ("er", exposure_response), ("dose_selection", dose_selection),
)
SECTION_TITLES: dict[str, str] = {
    "dataset": "Dataset", "nca": "Non-compartmental analysis", "qc": "Quality control",
    "structural": "Structural model comparison", "population": "Population fit",
    "covariates": "Covariate effects", "diagnostics": "Model diagnostics",
    "indirect": "Indirect comparison", "er": "Exposure-response", "dose_selection": "Dose selection",
}
#: Source key (memo.SOURCE_TOOLS) -> (PharmState fields it fills, reader-facing name).
SOURCE_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "data_quality": (("data_quality",), "Data-quality profile"),
    "nca_summary": (("nca_summary",), "Non-compartmental analysis"),
    "qc": (("qc_verdict", "qc_issues", "qc_checklist"), "Quality control"),
    "pk_model_results": (("pk_model_results",), "Structural model comparison"),
    "nlme_results": (("nlme_results",), "Population fit"),
    "scm_results": (("scm_results",), "Stepwise covariate modelling"),
    "forest_results": (("forest_results",), "Covariate forest"),
    "vpc_results": (("vpc_results",), "Goodness of fit"),
    "diagnostics_results": (("diagnostics_results",), "Residual diagnostics"),
    "er_results": (("er_results",), "Exposure-response fits"),
    "dose_selection_results": (("dose_selection_results",), "Dose selection"),
}
#: Resolved run by run in ``memo_sections_er.current_er`` (several stored results per field,
#: each from its own audit entry), not by the field's latest producing entry.
PER_RUN_SOURCES = frozenset({"er_results", "dose_selection_results"})
#: Memo section -> the source keys it reads (a section withheld for staleness is not "not run").
SECTION_SOURCES: dict[str, tuple[str, ...]] = {
    "dataset": ("data_quality",), "nca": ("nca_summary",), "qc": ("qc",),
    "structural": ("pk_model_results",), "population": ("nlme_results",),
    "covariates": ("nlme_results", "scm_results", "forest_results"),
    "diagnostics": ("vpc_results", "diagnostics_results"), "indirect": (),
    "er": ("er_results",), "dose_selection": ("dose_selection_results",),
}


@dataclass(frozen=True)
class Memo:
    title: str
    study: tuple[Block, ...]
    sections: tuple[Section, ...]
    omitted: tuple[tuple[str, str], ...]      # (key, title) of sections not run
    values: tuple[TracedValue, ...]
    footer: tuple[Block, ...] = ()
    withheld: tuple[tuple[str, str], ...] = ()   # (source key, name) computed on an earlier dataset

    @property
    def headlines(self) -> tuple[str, ...]:
        return tuple(s.headline for s in self.sections if s.headline)

    def blocks(self) -> tuple[Block, ...]:
        has_inputs = any(v.kind == "input" for v in self.values)
        out: list[Block] = [Heading(self.title, 0), *self.study,
                            Para(METHOD_NOTE_WITH_INPUTS if has_inputs else METHOD_NOTE)]
        if self.headlines:
            out += [Heading("Headlines"), *(Bullet(h) for h in self.headlines)]
        for sec in self.sections:
            out += [Heading(sec.title), *sec.blocks]
        if self.withheld:
            out += [Heading("Results withheld"),
                    *(Bullet(f"{name}: computed on a previously loaded dataset, not on the current "
                             "one; not shown. Re-run it on the current dataset.") for _, name in self.withheld)]
        if self.omitted:
            out += [Heading("Sections not run"), *(Bullet(f"{t}: not run.") for _, t in self.omitted)]
        return (*out, *self.footer)

    def text(self) -> str:
        return blocks_text(self.blocks())


def _study_block(state: PharmState) -> tuple[Block, ...]:
    """Compound / sponsor / study / indication, shown only when they carry no numeral.

    These strings are human-entered (and can arrive through a chat tool call), so a
    value with digits is withheld rather than printed: nothing numeric may enter the
    memo except through the traced value table."""
    info = state.study_info
    if not info:
        return ()
    shown: list[Block] = []
    withheld = False
    for label, value in (("Compound", info.drug_name), ("Sponsor", info.sponsor),
                         ("Study", info.study_id), ("Indication", info.indication)):
        text = " ".join(str(value or "").split())
        if not text:
            continue
        if has_numeral(text):
            withheld = True
        else:
            shown.append(Para(f"{label}: {text}"))
    if withheld:
        shown.append(Para("Study metadata containing digits is not shown."))
    return tuple(shown)


def _check_title(title: object) -> str:
    clean = " ".join(str(title or "").split())[:MAX_TITLE] or DEFAULT_TITLE
    return reject_numerals(clean, "title")


def _render(key: str, fn: Builder, state: PharmState, trace: AuditTrace,
            rec: Recorder) -> Section | None:
    """Run one section builder; any failure on odd state is a clear ValueError (fail closed)."""
    try:
        return fn(state, trace, rec)
    except (AttributeError, TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError(
            f"cannot render the '{key}' section from the current state: "
            f"{type(exc).__name__}: {exc}") from exc


def _current_only(state: PharmState, trace: AuditTrace) -> tuple[PharmState, tuple[tuple[str, str], ...]]:
    """``state`` without results computed before the latest load_dataset (a memo must not
    mix a previous dataset's analyses with the current one), and what was withheld.
    Dating needs the audit trail; without one only an E-R fit stamped with another
    dataset_id than the current one (or a selection computed from one) is withheld."""
    stale = [k for k in DATASET_BOUND if k in SOURCE_FIELDS and k not in PER_RUN_SOURCES
             and trace.predates_dataset(k) and any(getattr(state, f) for f in SOURCE_FIELDS[k][0])]
    cleared = {f: None for k in stale for f in SOURCE_FIELDS[k][0]}
    state, er_withheld = current_er(state.model_copy(update=cleared) if cleared else state, trace)
    return state, (*((k, SOURCE_FIELDS[k][1]) for k in stale), *er_withheld)


def build_memo(state: PharmState, trace: AuditTrace, *, title: object = DEFAULT_TITLE,
               own_index: int | None = None) -> Memo:
    """The memo for ``state``. ``trace`` supplies audit indices (empty -> unknown tags);
    ``own_index`` is the audit entry this memo's own build will be recorded as."""
    clean = _check_title(title)
    rec = Recorder()
    study = _study_block(state)
    state, withheld = _current_only(state, trace)
    gone = {k.split(".")[0] for k, _ in withheld}           # 'er_results.fits.x' -> 'er_results'
    built = [(key, _render(key, fn, state, trace, rec)) for key, fn in BUILDERS]
    footer: tuple[Block, ...] = (
        (Para("Built by audit entry" + trace_tag(own_index)),) if own_index is not None else ())
    return Memo(
        title=clean, study=study,
        sections=tuple(sec for _, sec in built if sec is not None),
        omitted=tuple((key, SECTION_TITLES[key]) for key, sec in built
                      if sec is None and not gone.intersection(SECTION_SOURCES[key])),
        values=rec.values, footer=footer, withheld=withheld)
