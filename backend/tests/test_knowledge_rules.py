"""mikebrain M2: the owner-approved rule catalog.

Keyless and offline. Properties pinned here:
  * a rule enters only as ``pending_owner_review`` and a pending rule has NO effect, to the byte;
  * a rule-derived finding can never be CRITICAL/HIGH, so it can never change ``goal_met``;
  * the predicate vocabulary is closed and evaluation is a pure function of (state, catalog);
  * evidence is bound by hash (no passage text in the catalog) and re-verifiable;
  * the end-to-end flow propose -> owner approves -> deterministic finding works, and not before.
"""
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.compute import adversarial
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.knowledge import cli
from app.knowledge import manifest as mf
from app.knowledge.extract import extract_text
from app.knowledge.index import load_index
from app.knowledge.rules import catalog_io as cio
from app.knowledge.rules.catalog_io import RuleCatalogRecord, load_rule_catalog
from app.knowledge.rules.evaluate import ORIGIN, apply_default_catalog, evaluate
from app.knowledge.rules.evidence import EvidenceError, evidence_for_chunks, verify_ref
from app.knowledge.rules.schema import ALLOWED_PATHS, Condition, Rule, RuleCatalog
from app.knowledge.rules_cli import parse_when
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.review_tools import TOOLS as REVIEW_TOOLS

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"
NOTES = sorted(FIXTURES.glob("*.txt"))
SHA_A, SHA_B = "a" * 64, "b" * 64
PENDING = {"status": "pending_owner_review"}
APPROVED = {"status": "approved", "by": "Dr. Owner Example", "date": "2026-10-24"}


# ── builders ────────────────────────────────────────────────────────────────────

def _evidence(**over) -> dict:
    base = dict(doc_sha=SHA_A, document="Synthetic note", year=2026, source_type="paper",
                start=0, end=100, text_sha256=SHA_B)
    return {**base, **over}


def _rule(**over) -> dict:
    base = dict(
        id="mb.nlme.high_shrinkage_ebe_screening", version="1.0.0", title="High eta-shrinkage",
        when=[dict(path="nlme_results.shrinkage_pct", quantifier="any", op="gt", value=30)],
        severity="MEDIUM", target="NLME covariate screening on EBEs",
        claim="EBE-based covariate screening is reported as informative",
        rationale="above ~30 % eta-shrinkage the EBEs are drawn toward the population mean",
        suggested_action="screen covariates inside the estimation instead of post hoc on the etas",
        evidence=[_evidence()], known_limits=["threshold is a convention"],
        proposed_by="owner", proposed_on="2026-10-24", approval=PENDING)
    return {**base, **over}


def _catalog(*rules: dict) -> RuleCatalog:
    return RuleCatalog.model_validate({"catalog_version": "0.1.0", "rules": list(rules)})


def _record(*rules: dict) -> RuleCatalogRecord:
    return RuleCatalogRecord(_catalog(*rules), "c" * 64)


def _state(**nlme) -> dict:
    block = {"status": "ok", "converged": True, "condition_number": 12.0,
             "shrinkage_pct": {"V": 45.0, "CL": 10.0}, "theta_rse_pct": {"CL": 9.0}, **nlme}
    return PharmState(session_id="s", nlme_results=block).model_dump()


# ── schema ──────────────────────────────────────────────────────────────────────

def test_valid_rule_loads_and_is_frozen():
    rule = Rule.model_validate(_rule())
    assert rule.approval.status == "pending_owner_review" and rule.when[0].quantifier == "any"
    with pytest.raises(ValidationError):
        rule.severity = "LOW"  # type: ignore[misc]


def test_approval_needs_who_and_when_and_pending_forbids_them():
    Rule.model_validate(_rule(approval=APPROVED))
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(approval={"status": "approved"}))
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(approval={"status": "pending_owner_review", "by": "Dr. Owner Example"}))
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(approval={**APPROVED, "date": "2026-13-45"}))


@pytest.mark.parametrize("who", ["Claude", "mikebrain-bot", "PharmAgent Loop Runner", "LLM reviewer", "Copilot"])
def test_automated_identity_cannot_approve(who):
    with pytest.raises(ValidationError, match="automated identity"):
        Rule.model_validate(_rule(approval={**APPROVED, "by": who}))


def test_ordinary_names_containing_those_letters_are_fine():
    for who in ("Dr. Abbott", "Sahab Zaidi", "Carlo Gentile"):
        Rule.model_validate(_rule(approval={**APPROVED, "by": who}))


@pytest.mark.parametrize("severity", ["HIGH", "CRITICAL", "high", "", "INFO"])
def test_rule_severity_cannot_block(severity):
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(severity=severity))


@pytest.mark.parametrize("cond", [
    dict(path="dataset_metadata.rows", op="gt", value=1),                                # not an allowed path
    dict(path="nlme_results.shrinkage_pct", op="gt", value=30),                          # map without quantifier
    dict(path="nlme_results.condition_number", op="gt", value=3, quantifier="any"),      # quantifier on a scalar
    dict(path="nlme_results.shrinkage_pct", op="eq", value=30, quantifier="any"),        # eq on a map
    dict(path="nlme_results.converged", op="gt", value=True),                            # ordering on a bool
    dict(path="nlme_results.converged", op="eq", value="no"),                           # wrong value type
    dict(path="pk_model_results.mode", op="gt", value="compare"),                       # ordering on a string
    dict(path="nlme_results.condition_number", op="gt", value=True),                    # bool is not a number
    dict(path="nlme_results.condition_number", op="gt", value=float("nan")),
    dict(path="nlme_results.condition_number", op="gt", value=float("inf")),
    dict(path="nlme_results.condition_number", op="matches", value=1),                  # unknown operator
])
def test_closed_predicate_vocabulary(cond):
    with pytest.raises(ValidationError):
        Condition.model_validate(cond)


def test_every_allowed_path_has_a_valid_condition():
    samples = {"num": dict(op="gt", value=1), "bool": dict(op="eq", value=False), "str": dict(op="eq", value="x"),
               "map_num": dict(op="ge", value=1, quantifier="all")}
    for path, kind in ALLOWED_PATHS.items():
        Condition.model_validate({"path": path, **samples[kind]})


@pytest.mark.parametrize("bad", [
    dict(id="nlme.no_prefix"), dict(id="mb.Upper"), dict(version="1.0"), dict(title=""), dict(evidence=[]),
    dict(when=[]), dict(proposed_on="24/10/2026"), dict(unexpected="field"),
    dict(when=[dict(path="nlme_results.converged", op="eq", value=False)] * 5),
    dict(evidence=[_evidence(end=0)]), dict(evidence=[_evidence(start=50, end=10)]),
    dict(evidence=[_evidence(doc_sha="XYZ")]), dict(evidence=[_evidence(text_sha256="B" * 64)]),
    dict(evidence=[_evidence(source_type="blog")]), dict(evidence=[_evidence(year=1500)]),
])
def test_rule_validation_rejects_malformed_entries(bad):
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(**bad))


def test_catalog_rejects_duplicate_ids_and_extra_keys():
    with pytest.raises(ValidationError, match="duplicate rule ids"):
        _catalog(_rule(), _rule())
    with pytest.raises(ValidationError):
        RuleCatalog.model_validate({"catalog_version": "0.1.0", "rules": [], "extra": 1})


# ── catalog file ────────────────────────────────────────────────────────────────

def test_shipped_catalog_is_valid_and_the_suite_never_reads_it_by_accident():
    rec = load_rule_catalog(cio.SHIPPED_CATALOG)                 # the real file, whatever the owner has put in it
    assert len(rec.catalog_sha256) == 64
    assert cio.DEFAULT_CATALOG != cio.SHIPPED_CATALOG            # conftest isolates every test from it
    assert cached_catalog_is_empty()


def cached_catalog_is_empty() -> bool:
    return cio.cached_rule_catalog().catalog.rules == ()


def test_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text('{"catalog_version": "0.1.0", "rules": [], "rules": []}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_rule_catalog(p)


def test_local_digest_recipe_equals_the_conventions_recipe():
    """The rules package defines its own digest (it may not import the conventions package); the two must agree."""
    from app.conventions.catalog_io import canonical_json_sha256 as conventions_digest
    sample = {"b": [1, 2, {"z": "é"}], "a": None, "catalog_version": "0.1.0"}
    assert cio.canonical_json_sha256(sample) == conventions_digest(sample)


def test_digest_ignores_formatting_but_not_content(tmp_path: Path):
    raw = {"catalog_version": "0.1.0", "rules": [_rule()]}
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(raw, indent=2))
    b.write_text(json.dumps(raw, separators=(",", ":")))
    c.write_text(json.dumps({**raw, "catalog_version": "0.1.1"}))
    assert load_rule_catalog(a).catalog_sha256 == load_rule_catalog(b).catalog_sha256
    assert load_rule_catalog(c).catalog_sha256 != load_rule_catalog(a).catalog_sha256


# ── evaluation ──────────────────────────────────────────────────────────────────

def test_approved_rule_fires_with_a_complete_deterministic_finding():
    rec = _record(_rule(approval=APPROVED))
    rep = evaluate(_state(), rec)
    assert len(rep.findings) == 1 and rep.info["n_fired"] == 1 and rep.info["fired"] == ["mb.nlme.high_shrinkage_ebe_screening"]
    f = rep.findings[0]
    assert f["severity"] == "MEDIUM" and f["resolved"] is False and f["origin"] == ORIGIN
    assert f["id"] == f["rule_id"] and f["rule_version"] == "1.0.0"
    assert f["catalog_version"] == "0.1.0" and f["catalog_sha256"] == "c" * 64
    assert f["approved_by"] == "Dr. Owner Example" and f["approved_on"] == "2026-10-24"
    assert "V=45" in f["evidence"] and "CL=10" not in f["evidence"]            # only the parameters that matched
    assert f["literature"] == [{"document": "Synthetic note", "year": 2026, "doc_sha": SHA_A, "text_sha256": SHA_B}]
    assert {"id", "severity", "target", "claim", "evidence", "suggested_action", "resolved"} <= set(f)


def test_pending_rule_has_no_effect_and_is_counted():
    rep = evaluate(_state(), _record(_rule(approval=PENDING)))
    assert rep.findings == () and rep.info is None                              # invisible while nothing is approved
    rep = evaluate(_state(), _record(_rule(approval=PENDING),
                                     _rule(id="mb.nlme.other", approval=APPROVED, when=[
                                         dict(path="nlme_results.condition_number", op="gt", value=1000)])))
    assert rep.findings == () and rep.info["n_approved"] == 1 and rep.info["n_pending_ignored"] == 1


@pytest.mark.parametrize("quantifier, shrink, fires", [
    ("any", {"V": 45.0, "CL": 10.0}, True), ("all", {"V": 45.0, "CL": 10.0}, False),
    ("all", {"V": 45.0, "CL": 31.0}, True), ("any", {"V": 30.0}, False),              # strict >
    ("any", {}, False), ("any", None, False), ("all", {"V": None}, False),
    ("any", {"V": "45"}, False), ("any", {"V": True}, False), ("all", {"V": 45.0, "CL": None}, False),
])
def test_map_quantifiers_and_non_numeric_values(quantifier, shrink, fires):
    cond = dict(path="nlme_results.shrinkage_pct", quantifier=quantifier, op="gt", value=30)
    rep = evaluate(_state(shrinkage_pct=shrink), _record(_rule(approval=APPROVED, when=[cond])))
    assert bool(rep.findings) is fires


@pytest.mark.parametrize("op, value, fires", [("gt", 30, False), ("ge", 30, True), ("lt", 31, True), ("le", 29, False),
                                              ("eq", 30, True), ("ne", 30, False)])
def test_scalar_operators(op, value, fires):
    cond = dict(path="nlme_results.condition_number", op=op, value=value)
    rep = evaluate(_state(condition_number=30.0), _record(_rule(approval=APPROVED, when=[cond])))
    assert bool(rep.findings) is fires


def test_missing_or_non_numeric_scalars_never_fire():
    cond = [dict(path="nlme_results.condition_number", op="lt", value=1e9)]
    rule = _rule(approval=APPROVED, when=cond)
    for bad in (None, True, "12", float("nan")):
        assert evaluate(_state(condition_number=bad), _record(rule)).findings == ()
    assert evaluate(PharmState(session_id="s").model_dump(), _record(rule)).findings == ()


def test_bool_and_string_conditions_and_conjunction():
    rule = _rule(approval=APPROVED, when=[dict(path="nlme_results.converged", op="eq", value=False),
                                          dict(path="nlme_results.condition_number", op="gt", value=1000)])
    assert evaluate(_state(converged=False, condition_number=5000.0), _record(rule)).findings
    assert evaluate(_state(converged=False, condition_number=10.0), _record(rule)).findings == ()   # AND
    assert evaluate(_state(converged=None, condition_number=5000.0), _record(rule)).findings == ()
    mode = _rule(approval=APPROVED, when=[dict(path="pk_model_results.mode", op="eq", value="compare")])
    st = PharmState(session_id="s", pk_model_results={"status": "ok", "mode": "compare"}).model_dump()
    assert evaluate(st, _record(mode)).findings
    st = PharmState(session_id="s", pk_model_results={"status": "ok", "mode": "single"}).model_dump()
    assert evaluate(st, _record(mode)).findings == ()


def test_results_block_must_report_status_ok():
    rec = _record(_rule(approval=APPROVED))
    assert evaluate(_state(status="failed"), rec).findings == ()
    assert evaluate(_state(status=None), rec).findings == ()


def test_evaluation_is_pure_ordered_and_does_not_mutate_state():
    rec = _record(_rule(id="mb.nlme.b_rule", approval=APPROVED),
                  _rule(id="mb.nlme.a_rule", approval=APPROVED))
    state = _state()
    before = copy.deepcopy(state)
    first, second = evaluate(state, rec), evaluate(state, rec)
    assert first == second and state == before
    assert [f["id"] for f in first.findings] == ["mb.nlme.b_rule", "mb.nlme.a_rule"]     # catalog order


def test_invalid_catalog_degrades_loudly_instead_of_raising(tmp_path: Path, monkeypatch):
    bad = tmp_path / "catalog.json"
    bad.write_text("{not json")
    monkeypatch.setattr(cio, "DEFAULT_CATALOG", bad)
    cio.cached_rule_catalog.cache_clear()
    try:
        findings, info = apply_default_catalog(_state())
    finally:
        cio.cached_rule_catalog.cache_clear()
    assert findings == [] and info["status"] == "invalid" and info["error"]


# ── the hard floor refuses blocking extras ──────────────────────────────────────

def _extra(sev: str) -> dict:
    return {"id": "x", "severity": sev, "target": "t", "claim": "c", "evidence": "e", "suggested_action": "a",
            "resolved": False}


@pytest.mark.parametrize("sev", ["CRITICAL", "HIGH", "", None, "INFO"])
def test_engine_refuses_blocking_or_unknown_extra_severity(sev):
    with pytest.raises(ValueError, match="only"):
        adversarial.review(_state(), None, None, extra_findings=[_extra(sev)])


def test_engine_merges_non_blocking_extras_without_touching_the_verdict():
    base = adversarial.review(_state(), None, None)
    with_extra = adversarial.review(_state(), None, None, extra_findings=[_extra("MEDIUM"), _extra("LOW")])
    assert with_extra["counts"]["MEDIUM"] == base["counts"]["MEDIUM"] + 1
    assert with_extra["counts"]["LOW"] == base["counts"]["LOW"] + 1
    for key in ("goal_met", "status", "checked", "nothing_checked", "unverifiable"):
        assert with_extra[key] == base[key], key
    assert with_extra["n_findings"] == base["n_findings"] + 2
    sev = [f["severity"] for f in with_extra["findings"]]
    assert sev == sorted(sev, key=("CRITICAL", "HIGH", "MEDIUM", "LOW").index)
    extra = _extra("LOW")
    adversarial.review(_state(), None, None, extra_findings=[extra])
    assert extra == _extra("LOW")                                                   # caller's dict untouched


# ── review tool integration ─────────────────────────────────────────────────────

@pytest.fixture
def catalog_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "rules_catalog.json"
    monkeypatch.setattr(cio, "DEFAULT_CATALOG", path)
    cio.cached_rule_catalog.cache_clear()

    def write(*rules: dict) -> Path:
        path.write_text(json.dumps({"catalog_version": "0.1.0", "rules": list(rules)}))
        cio.cached_rule_catalog.cache_clear()
        return path
    write()
    yield write
    cio.cached_rule_catalog.cache_clear()


def _review(state: PharmState, audit: AuditChain | None = None):
    audit = audit if audit is not None else AuditChain()
    return default_registry().execute("adversarial_review", state=state, ctx=ToolContext(), args={}, audit=audit,
                                      timestamp="2026-10-24T00:00:00Z", actor="tester"), audit


def _clean_state() -> PharmState:
    return PharmState(session_id="s", nlme_results={"status": "ok", "converged": True, "condition_number": 12.0,
                                                    "shrinkage_pct": {"V": 45.0}, "theta_rse_pct": {"CL": 9.0}})


def test_review_tool_schema_is_unchanged_by_rules():
    t = REVIEW_TOOLS[0]
    assert set(t.input_schema["properties"]) == {"goal"} and "rule" not in t.description.lower()


def test_pending_only_catalog_is_byte_identical_to_no_catalog(catalog_file):
    catalog_file()
    (_, none), _ = _review(_clean_state())
    catalog_file(_rule(approval=PENDING))
    (_, pending), _ = _review(_clean_state())
    assert pending.result == none.result and pending.summary == none.summary
    assert "rule_catalog" not in pending.result
    # and both equal the pre-M2 path: the engine called directly, with no catalog in sight
    assert pending.result == adversarial.review(_clean_state().model_dump(), None, {})


def test_approved_rule_adds_a_finding_but_never_changes_the_verdict(catalog_file):
    catalog_file()
    (_, base), _ = _review(_clean_state())
    catalog_file(_rule(approval=APPROVED))
    (st, got), audit = _review(_clean_state())
    r = got.result
    assert base.result["goal_met"] is True and r["goal_met"] is True and r["status"] == base.result["status"] == "GOAL MET"
    assert r["counts"]["MEDIUM"] == base.result["counts"]["MEDIUM"] + 1
    assert r["counts"]["CRITICAL"] == base.result["counts"]["CRITICAL"] and r["counts"]["HIGH"] == base.result["counts"]["HIGH"]
    fired = [f for f in r["findings"] if f.get("origin") == ORIGIN]
    assert [f["id"] for f in fired] == ["mb.nlme.high_shrinkage_ebe_screening"]
    assert [f for f in r["findings"] if f.get("origin") != ORIGIN] == base.result["findings"]
    assert r["rule_catalog"]["status"] == "ok" and r["rule_catalog"]["n_fired"] == 1
    assert "mikebrain rules: 1 fired" in got.summary
    assert st.review_results == r and audit.verify() and audit.entries[-1].tool == "adversarial_review"


def test_approved_rule_alongside_a_blocking_finding_keeps_the_block(catalog_file):
    catalog_file(_rule(approval=APPROVED))
    failing = PharmState(session_id="s", nlme_results={"status": "ok", "converged": False, "condition_number": 5000.0,
                                                       "shrinkage_pct": {"V": 45.0}, "theta_rse_pct": {"CL": 9.0}})
    (_, got), _ = _review(failing)
    assert got.result["goal_met"] is False and got.result["status"] == "FINDINGS BLOCK GOAL"
    assert got.result["counts"]["HIGH"] == 2 and got.result["counts"]["MEDIUM"] >= 1


def test_broken_catalog_is_recorded_and_the_review_is_otherwise_unchanged(catalog_file, tmp_path: Path):
    catalog_file()
    (_, base), _ = _review(_clean_state())
    broken = tmp_path / "rules_catalog.json"
    broken.write_text('{"catalog_version": "0.1.0", "rules": [{"id": "nope"}]}')
    cio.cached_rule_catalog.cache_clear()
    (_, got), audit = _review(_clean_state())
    assert got.result["rule_catalog"]["status"] == "invalid"
    assert {k: v for k, v in got.result.items() if k != "rule_catalog"} == base.result
    assert "SKIPPED (invalid)" in got.summary and audit.verify()


# ── review follow-ups: hostile input, immutability, approver and text guards ─────

def test_huge_integers_are_rejected_not_an_overflow():
    big = 10 ** 400
    with pytest.raises(ValidationError, match="finite number"):
        Condition.model_validate(dict(path="nlme_results.condition_number", op="gt", value=big))
    with pytest.raises(ValidationError, match="finite number"):
        Condition.model_validate(dict(path="nlme_results.shrinkage_pct", quantifier="any", op="gt", value=-big))


def test_evaluation_survives_hostile_state_values():
    rule = _rule(approval=APPROVED, when=[dict(path="nlme_results.condition_number", op="gt", value=1)])
    assert evaluate(_state(condition_number=10 ** 400), _record(rule)).findings == ()        # not finite as a float
    mixed = _rule(approval=APPROVED, when=[dict(path="nlme_results.shrinkage_pct", quantifier="any", op="gt", value=1)])
    # keys 1 and "1" collapse to the same text; the sort must not compare None with a float
    assert evaluate(_state(shrinkage_pct={1: None, "1": 5.0}), _record(mixed)).findings


def test_large_values_are_printed_without_losing_the_difference():
    rule = _rule(approval=APPROVED, when=[dict(path="nlme_results.condition_number", op="gt", value=1234566)])
    (finding,) = evaluate(_state(condition_number=1234567.0), _record(rule)).findings
    assert "1234567" in finding["evidence"] and "1234566" in finding["evidence"]


def test_catalog_failures_never_escape_and_never_leak_a_path(tmp_path: Path, monkeypatch):
    for text in ('{"catalog_version": "0.1.0", "rules": [' + _huge_rule_json() + ']}',
                 "[" * 20000 + "]" * 20000,                                   # RecursionError territory
                 '{"catalog_version": "0.1.0", "rules": [{"id": "nope"}]}'):
        bad = tmp_path / "catalog.json"
        bad.write_text(text)
        monkeypatch.setattr(cio, "DEFAULT_CATALOG", bad)
        cio.cached_rule_catalog.cache_clear()
        findings, info = apply_default_catalog(_state())
        assert findings == [] and info["status"] == "invalid"
        assert str(tmp_path) not in info["error"] and len(info["error"]) < 200       # generic message only


def _huge_rule_json() -> str:
    r = _rule()
    r["when"] = [dict(path="nlme_results.condition_number", op="gt", value=0)]
    return json.dumps(r).replace('"value": 0', '"value": ' + "9" * 400)


def test_an_unexpected_evaluation_error_skips_rules_and_says_so(monkeypatch):
    import app.knowledge.rules.evaluate as ev

    def boom(*a, **k):
        raise RuntimeError("/Users/someone/secret/path exploded")
    monkeypatch.setattr(ev, "evaluate", boom)
    findings, info = ev.apply_default_catalog(_state())
    assert findings == [] and info["status"] == "error" and "secret" not in json.dumps(info)


def test_a_loaded_catalog_cannot_be_mutated_in_place():
    rec = _record(_rule(approval=APPROVED))
    rule = rec.catalog.rules[0]
    assert isinstance(rec.catalog.rules, tuple) and isinstance(rule.when, tuple) and isinstance(rule.evidence, tuple)
    for target, op in ((rec.catalog.rules, "clear"), (rule.when, "append"), (rule.evidence, "append"),
                       (rule.known_limits, "append")):
        assert not hasattr(target, op)
    with pytest.raises(ValidationError):
        rule.when[0].value = 0                                                # type: ignore[misc]


@pytest.mark.parametrize("who", ["ClaudeBot".replace("Bot", " Bot"), "\uff23laude", "Cl\u200baude", "ChatGPT",
                                 "dependabot", "Gemini"])
def test_approver_check_folds_width_and_invisible_characters(who):
    with pytest.raises(ValidationError, match="automated identity"):
        Rule.model_validate(_rule(approval={**APPROVED, "by": who}))


@pytest.mark.parametrize("by", ["  ", "1234", "-"])
def test_approver_must_name_a_person(by):
    with pytest.raises(ValidationError, match="name a person"):
        Rule.model_validate(_rule(approval={**APPROVED, "by": by}))


def test_a_rule_cannot_be_approved_before_it_was_proposed():
    with pytest.raises(ValidationError, match="before it was proposed"):
        Rule.model_validate(_rule(proposed_on="2026-10-24", approval={**APPROVED, "date": "2026-10-23"}))


@pytest.mark.parametrize("field, limit", [("title", 200), ("claim", 400), ("rationale", 800),
                                          ("suggested_action", 400), ("target", 200)])
def test_free_text_is_size_capped(field, limit):
    Rule.model_validate(_rule(**{field: "x" * limit}))
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(**{field: "x" * (limit + 1)}))
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(known_limits=["y" * 301]))


def test_evidence_stores_no_section_heading():
    with pytest.raises(ValidationError):
        Rule.model_validate(_rule(evidence=[{**_evidence(), "section": "2 Why it matters"}]))


def test_desktop_bundle_ships_every_app_catalog():
    """The packaged app reads catalog.json beside the code; PyInstaller only bundles what the spec lists."""
    backend = Path(__file__).parent.parent
    spec = (backend.parent / "desktop" / "PharmAgent.spec").read_text()
    found = sorted((backend / "app").rglob("catalog.json"))
    assert backend / "app" / "knowledge" / "rules" / "catalog.json" in found
    for path in found:
        needle = " / ".join(f'"{part}"' for part in path.relative_to(backend).parts)
        assert needle in spec, f"{path.relative_to(backend)} is not bundled by desktop/PharmAgent.spec"


# ── evidence binding ────────────────────────────────────────────────────────────

@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    for p in NOTES:
        mf.add_document(root, mf.DocumentRecord(str(p), mf.file_sha256(p), p.stem, 2026, "paper", "t",
                                                datetime.now(UTC).isoformat()), extract_text(p))
    return root


def _chunk_id(root: Path, needle: str) -> str:
    return next(c.chunk_id for c in load_index(root).chunks if needle in " ".join(c.text.split()))


def test_evidence_is_a_hash_pointer_with_no_passage_text(corpus: Path):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    (ref,) = evidence_for_chunks(corpus, [cid, cid])                       # duplicates collapse
    chunk = load_index(corpus).chunk(cid)
    assert (ref.start, ref.end, ref.text_sha256) == (chunk.start, chunk.end, chunk.chunk_hash)
    assert ref.document == "shrinkage_note" and ref.year == 2026 and ref.source_type == "paper"
    assert "uninformative" not in json.dumps(ref.model_dump())              # nothing from the passage
    assert verify_ref(corpus, ref) == "present"
    with pytest.raises(EvidenceError, match="unknown chunk id"):
        evidence_for_chunks(corpus, ["ffffffffffff:99"])


def test_verify_detects_every_way_the_binding_can_break(corpus: Path):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    (ref,) = evidence_for_chunks(corpus, [cid])
    text_file = mf.text_path(corpus, ref.doc_sha)
    original = text_file.read_text()
    # same-length edit INSIDE the cited span (the same word also occurs earlier in the document, outside it)
    inside = original[ref.start:ref.end]
    assert "uninformative" in inside
    text_file.write_text(original[:ref.start] + inside.replace("uninformative", "informativeXX", 1) + original[ref.end:])
    assert verify_ref(corpus, ref) == "text_changed"
    text_file.write_text(original.replace("uninformative", "informativeXX", 1))     # edit outside the span: binding holds
    assert original.index("uninformative") < ref.start and verify_ref(corpus, ref) == "present"
    text_file.write_text(original[: ref.end - 5])
    assert verify_ref(corpus, ref) == "span_out_of_range"
    text_file.unlink()
    assert verify_ref(corpus, ref) == "missing_text"
    text_file.write_text(original)
    assert verify_ref(corpus, ref) == "present"
    mf.remove_document(corpus, ref.doc_sha)
    assert verify_ref(corpus, ref) == "missing_document"
    for p in NOTES:
        sha = mf.file_sha256(p)
        if mf.load_manifest(corpus).get(sha):
            mf.remove_document(corpus, sha)
    assert verify_ref(corpus, ref) == "corpus_empty"


# ── CLI + the whole flow ────────────────────────────────────────────────────────

def _add_args(cat: Path, root: Path, cid: str, **over) -> list[str]:
    args = ["--root", str(root), "rules", "--catalog", str(cat), "add", "--id", "mb.nlme.high_shrinkage_ebe_screening",
            "--title", "High eta-shrinkage", "--severity", "MEDIUM",
            "--when", "nlme_results.shrinkage_pct any gt 30", "--target", "NLME covariate screening on EBEs",
            "--claim", "EBE-based covariate screening is reported as informative",
            "--rationale", "above ~30 % eta-shrinkage the EBEs collapse toward the population mean",
            "--action", "screen covariates inside the estimation", "--chunk", cid, "--limit", "30 % is a convention",
            "--by", "owner"]
    return args


@pytest.fixture
def empty_cat(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps({"catalog_version": "0.1.0", "rules": []}))
    return p


def test_parse_when_forms():
    assert parse_when("nlme_results.shrinkage_pct any gt 30").quantifier == "any"
    assert parse_when("nlme_results.converged eq false").value is False
    assert parse_when("pk_model_results.mode eq compare").value == "compare"
    with pytest.raises(ValueError, match="expected"):
        parse_when("nlme_results.shrinkage_pct")
    with pytest.raises(ValidationError):
        parse_when("dataset_metadata.rows gt 3")


def test_cli_add_writes_a_pending_rule_with_hash_evidence_and_no_passage_text(corpus, empty_cat, capsys):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    before = load_rule_catalog(empty_cat)
    assert cli.main(_add_args(empty_cat, corpus, cid)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["approval"] == "pending_owner_review" and out["catalog_version"] == "0.1.1"
    after = load_rule_catalog(empty_cat)
    assert after.catalog_sha256 == out["catalog_sha256"] != before.catalog_sha256
    (rule,) = after.catalog.rules
    assert rule.approval.status == "pending_owner_review" and rule.approval.by is None and rule.approval.date is None
    assert rule.proposed_by == "owner" and rule.version == "1.0.0" and len(rule.evidence) == 1
    text = empty_cat.read_text()
    for phrase in ("drawn toward the population mean", "lose their power to reveal", "uninformative"):
        assert phrase not in text                                          # copyrighted-text guard
    assert evaluate(_state(), load_rule_catalog(empty_cat)).info is None   # pending: still inert


def test_cli_refuses_text_copied_from_the_cited_passage(corpus, empty_cat, capsys):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    passage = " ".join(load_index(corpus).chunk(cid).text.split())
    snapshot = empty_cat.read_text()
    args = _add_args(empty_cat, corpus, cid)
    args[args.index("--rationale") + 1] = "Per the note: " + passage[40:140]            # 100 chars, verbatim
    assert cli.main(args) == 2
    assert "own words" in capsys.readouterr().err and empty_cat.read_text() == snapshot
    args[args.index("--rationale") + 1] = passage[40:70]                              # 30 chars: below the threshold
    assert cli.main(args) == 0


def test_cli_stores_no_section_heading_and_reports_huge_numbers_as_errors(corpus, empty_cat, capsys):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    bad = _add_args(empty_cat, corpus, cid)
    bad[bad.index("--when") + 1] = "nlme_results.condition_number gt " + "9" * 400
    assert cli.main(bad) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err
    assert cli.main(_add_args(empty_cat, corpus, cid)) == 0
    assert "Why it matters" not in empty_cat.read_text() and '"section"' not in empty_cat.read_text()


def test_concurrent_edit_during_add_is_detected_and_leaves_no_litter(empty_cat, monkeypatch):
    real_mkstemp = cio.tempfile.mkstemp

    def mkstemp_then_someone_else_writes(*a, **k):
        fd, name = real_mkstemp(*a, **k)
        empty_cat.write_text(json.dumps({"catalog_version": "0.1.0", "rules": []}, indent=4))   # another writer
        return fd, name
    monkeypatch.setattr(cio.tempfile, "mkstemp", mkstemp_then_someone_else_writes)
    with pytest.raises(ValueError, match="changed while the rule was being added"):
        cio.append_pending_rule(Rule.model_validate(_rule()), empty_cat)
    assert load_rule_catalog(empty_cat).catalog.rules == ()
    assert [p.name for p in empty_cat.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_cli_failures_leave_the_catalog_untouched(corpus, empty_cat, capsys):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    snapshot = empty_cat.read_text()
    assert cli.main(_add_args(empty_cat, corpus, "ffffffffffff:99")) == 2            # unknown chunk
    bad_when = _add_args(empty_cat, corpus, cid)
    bad_when[bad_when.index("--when") + 1] = "dataset_metadata.rows gt 3"
    assert cli.main(bad_when) == 2                                                   # path outside the vocabulary
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                                  # argparse: HIGH is not a choice
        cli.main([*_add_args(empty_cat, corpus, cid), "--severity", "HIGH"])
    assert empty_cat.read_text() == snapshot
    assert cli.main(_add_args(empty_cat, corpus, cid)) == 0
    assert cli.main(_add_args(empty_cat, corpus, cid)) == 2                          # duplicate id
    assert len(load_rule_catalog(empty_cat).catalog.rules) == 1


def test_the_only_writer_refuses_to_approve(empty_cat):
    snapshot = empty_cat.read_text()
    with pytest.raises(ValueError, match="pending_owner_review"):
        cio.append_pending_rule(Rule.model_validate(_rule(approval=APPROVED)), empty_cat)
    assert empty_cat.read_text() == snapshot and load_rule_catalog(empty_cat).catalog.rules == ()


def test_cli_list_and_verify(corpus, empty_cat, capsys):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    assert cli.main(_add_args(empty_cat, corpus, cid)) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(corpus), "rules", "--catalog", str(empty_cat), "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["rules"][0]["approval"] == "pending_owner_review" and listed["rules"][0]["n_evidence"] == 1
    assert cli.main(["--root", str(corpus), "rules", "--catalog", str(empty_cat), "verify"]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == 0
    mf.remove_document(corpus, load_rule_catalog(empty_cat).catalog.rules[0].evidence[0].doc_sha)
    assert cli.main(["--root", str(corpus), "rules", "--catalog", str(empty_cat), "verify"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["problems"] == 1 and rep["rules"][0]["evidence"][0]["status"] == "missing_document"


def test_end_to_end_propose_then_owner_approves_then_it_fires(corpus, empty_cat, catalog_file, monkeypatch):
    cid = _chunk_id(corpus, "Covariate screening on shrunken EBEs")
    assert cli.main(_add_args(empty_cat, corpus, cid)) == 0
    monkeypatch.setattr(cio, "DEFAULT_CATALOG", empty_cat)
    cio.cached_rule_catalog.cache_clear()
    (_, proposed), _ = _review(_clean_state())
    base_medium = proposed.result["counts"]["MEDIUM"]                    # the engine's own shrinkage check
    assert "rule_catalog" not in proposed.result and not any(f.get("origin") == ORIGIN
                                                             for f in proposed.result["findings"])   # pending: nothing

    raw = json.loads(empty_cat.read_text())                                  # the owner's reviewed edit
    raw["rules"][0]["approval"] = {"status": "approved", "by": "Dr. Owner Example", "date": "2026-10-24"}
    empty_cat.write_text(json.dumps(raw, indent=2))
    cio.cached_rule_catalog.cache_clear()
    (_, approved), audit = _review(_clean_state())
    r = approved.result
    assert r["goal_met"] is True and r["counts"]["MEDIUM"] == base_medium + 1 and r["rule_catalog"]["n_fired"] == 1
    finding = next(f for f in r["findings"] if f.get("origin") == ORIGIN)
    assert finding["literature"][0]["document"] == "shrinkage_note"
    assert finding["catalog_sha256"] == r["rule_catalog"]["catalog_sha256"] == load_rule_catalog(empty_cat).catalog_sha256
    assert audit.verify()


def test_rule_authoring_has_no_llm_path():
    """The catalog writer is reachable from exactly two modules — the CLI that implements `rules add` and
    its definition — so no tool, agent, LLM provider or API route can mint a rule."""
    app_dir = Path(__file__).parent.parent / "app"
    users = {str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
             if "append_pending_rule" in p.read_text(encoding="utf-8")}
    assert users == {"knowledge/rules/catalog_io.py", "knowledge/rules_cli.py"}
    import app.tools.knowledge_tools as kt
    src = Path(kt.__file__).read_text()
    for forbidden in ("catalog_io", "rules_cli", "knowledge.rules"):
        assert forbidden not in src, forbidden
