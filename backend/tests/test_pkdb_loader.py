"""PK-DB loader + FIH answer-key tests — all offline against a real fixture.

The fixture ``pkdb/fixtures/caffeine_filter.zip`` is a trimmed but *real* PK-DB
``/filter`` download for caffeine (2026-07-12), so these tests exercise the
actual export schema without hitting the network. The live path is covered by a
single opt-in smoke test, skipped unless ``PKDB_LIVE=1``.
"""
from __future__ import annotations

import csv
import io
import os
import zipfile

import pytest

from pharmacometricsbench.grading import within_tolerance
from pharmacometricsbench.pkdb.fih_tasks import (
    build_fih_pk_tasks,
    harmonize,
    pool_parameter,
)
from pharmacometricsbench.pkdb.loader import (
    DrugHarvest,
    PKDBClient,
    PKParameter,
    Reference,
)

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "pharmacometricsbench", "pkdb",
    "fixtures", "caffeine_filter.zip",
)


@pytest.fixture(scope="module")
def caffeine_zip() -> zipfile.ZipFile:
    return zipfile.ZipFile(_FIXTURE)


def _make_zip(studies, interventions, rows, rows_file="individuals.csv") -> zipfile.ZipFile:
    """Build an in-memory PK-DB-shaped ZIP so the *populated* answer-key path
    (which is empty in the anonymous fixture) can be exercised offline.
    ``rows_file`` chooses which table the measurement rows land in."""
    def csv_bytes(records, cols):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({c: r.get(c, "") for c in cols})
        return buf.getvalue()

    st_cols = ["sid", "name", "licence", "access"]
    iv_cols = ["study_sid", "study_name", "substance", "substance_label",
               "value", "mean", "median", "unit", "route", "form"]
    row_cols = ["study_sid", "study_name", "measurement_type", "substance",
                "value", "mean", "median", "unit", "tissue", "route", "count"]
    tables = {"individuals.csv": [], "groups.csv": [], "outputs.csv": []}
    tables[rows_file] = rows
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("studies.csv", csv_bytes(studies, st_cols))
        z.writestr("interventions.csv", csv_bytes(interventions, iv_cols))
        for name, recs in tables.items():
            z.writestr(name, csv_bytes(recs, row_cols))
    bio.seek(0)
    return zipfile.ZipFile(bio)


# A caffeine intervention pins the drug's sid ('caf') so the substance filter
# has something to resolve; used by the populated-path tests below.
_CAF_IV = {"study_sid": "PKDB01", "study_name": "S1", "substance": "caf",
           "substance_label": "caffeine (137X)", "value": "0.2", "unit": "gram",
           "route": "oral", "form": "tablet"}
_CAF_STUDY = {"sid": "PKDB01", "name": "S1", "licence": "open", "access": "public"}
_CAF_STUDY2 = {"sid": "PKDB02", "name": "S2", "licence": "open", "access": "public"}


# ── Loader: parsing real export ─────────────────────────────────────────────
def test_harvest_dosing_is_real_and_cited(caffeine_zip):
    h = PKDBClient(open_only=True).harvest_from_zip("caffeine", caffeine_zip)
    assert h.dosing, "expected real caffeine dosing records"
    d = h.dosing[0]
    assert d.substance == "caf"          # the caffeine substance sid
    assert d.route == "oral"
    assert d.dose > 0
    # provenance is mandatory, never blank
    assert d.reference.study_sid.startswith("PKDB")
    assert d.reference.study_name


def test_licence_gate_open_only(caffeine_zip):
    """PK-DB caffeine is a mix of open/closed studies; the gate must respect it."""
    open_h = PKDBClient(open_only=True).harvest_from_zip("caffeine", caffeine_zip)
    all_h = PKDBClient(open_only=False).harvest_from_zip("caffeine", caffeine_zip)
    assert len(all_h.dosing) > len(open_h.dosing)      # closed studies exist
    assert all(r.licence == "open" for r in open_h.studies)
    # every open-only dosing came from an open (or, for uncatalogued, unknown) study
    assert all(d.reference.licence in ("open", "unknown") for d in open_h.dosing)


def test_no_pk_parameters_anonymously(caffeine_zip):
    """The core finding: anonymous export exposes no drug-level PK answer keys."""
    h = PKDBClient().harvest_from_zip("caffeine", caffeine_zip)
    assert h.pk_parameters == []
    assert any("PKDB_API_TOKEN" in s or "auth-gated" in s for s in h.skipped)


def test_creatinine_clearance_excluded_on_populated_path():
    """The core exclusion, exercised on a POPULATED answer-key path: a caffeine
    clearance row is kept; a creatinine clearance row (covariate) is dropped."""
    rows = [
        {"study_sid": "PKDB01", "measurement_type": "clearance", "substance": "caf",
         "value": "6.0", "unit": "liter / hour", "tissue": "plasma", "count": "10"},
        {"study_sid": "PKDB01", "measurement_type": "clearance", "substance": "creatinine",
         "value": "5.4", "unit": "liter / hour", "tissue": "serum", "count": "10"},
    ]
    zf = _make_zip([_CAF_STUDY], [_CAF_IV], rows)
    h = PKDBClient().harvest_from_zip("caffeine", zf)
    subs = {p.substance for p in h.pk_parameters}
    assert subs == {"caf"}                      # caffeine kept, creatinine excluded
    assert all(p.reference.study_sid for p in h.pk_parameters)   # provenance present


def test_outputs_csv_is_parsed():
    """The dedicated outputs.csv table (empty anonymously, populated once a token
    unlocks it) must be read — a PK param living only there is harvested."""
    rows = [
        {"study_sid": "PKDB01", "measurement_type": "auc_inf", "substance": "caf",
         "value": "40.0", "unit": "milligram * hour / liter", "tissue": "plasma",
         "route": "oral", "count": "12"},
    ]
    zf = _make_zip([_CAF_STUDY], [_CAF_IV], rows, rows_file="outputs.csv")
    h = PKDBClient().harvest_from_zip("caffeine", zf)
    assert len(h.pk_parameters) == 1
    p = h.pk_parameters[0]
    assert p.measurement_type == "auc_inf" and p.value == 40.0 and p.route == "oral"


def test_empty_drug_sids_fails_closed():
    """If the drug's sid cannot be resolved, harvest emits NOTHING (never falls
    open to keep every substance) — the guard against covariate leakage."""
    # No intervention names the drug, so _drug_substance_sids -> empty set.
    rows = [
        {"study_sid": "PKDB01", "measurement_type": "clearance", "substance": "creatinine",
         "value": "5.4", "unit": "liter / hour", "tissue": "serum", "count": "10"},
    ]
    ivs = [{"study_sid": "PKDB01", "study_name": "S1", "substance": "xyz",
            "substance_label": "something-else (000)", "value": "1", "unit": "gram",
            "route": "oral"}]
    h = PKDBClient().harvest_from_zip("caffeine", _make_zip([_CAF_STUDY], ivs, rows))
    assert h.pk_parameters == []
    assert h.dosing == []
    assert any("fail-closed" in s for s in h.skipped)


def test_substring_collision_excluded_salt_kept():
    """'codeine' captures its salt forms but NOT the congener 'dihydrocodeine'."""
    sids = PKDBClient._drug_substance_sids("codeine", [
        {"substance": "cod", "substance_label": "codeine (...)"},
        {"substance": "cod-p", "substance_label": "codeine phosphate (...)"},   # salt: keep
        {"substance": "dhc", "substance_label": "dihydrocodeine (...)"},        # congener: drop
    ])
    assert sids == {"cod", "cod-p"}


def test_synonym_resolution():
    """A well-known INN/USAN synonym still resolves (acetaminophen -> paracetamol)."""
    sids = PKDBClient._drug_substance_sids("acetaminophen", [
        {"substance": "apap", "substance_label": "paracetamol (N02BE01)"},
    ])
    assert sids == {"apap"}


def test_pk_param_whitelist_tissue_and_licence_gates():
    """harvest_pk_parameters: non-whitelisted type dropped, urine tissue dropped,
    closed-licence study dropped-with-reason, unparseable value skip-logged."""
    rows = [
        {"study_sid": "PKDB01", "measurement_type": "clearance", "substance": "caf",
         "value": "6.0", "unit": "liter / hour", "tissue": "plasma", "count": "10"},
        {"study_sid": "PKDB01", "measurement_type": "weight", "substance": "caf",
         "value": "70", "unit": "kg", "tissue": "plasma"},                 # not a PK type
        {"study_sid": "PKDB01", "measurement_type": "clearance", "substance": "caf",
         "value": "9.0", "unit": "liter / hour", "tissue": "urine"},        # wrong tissue
        {"study_sid": "PKDB09", "measurement_type": "clearance", "substance": "caf",
         "value": "7.0", "unit": "liter / hour", "tissue": "plasma"},       # closed study
        {"study_sid": "PKDB01", "measurement_type": "cmax", "substance": "caf",
         "value": "", "unit": "milligram / liter", "tissue": "plasma"},     # no value
    ]
    studies = [_CAF_STUDY, {"sid": "PKDB09", "name": "S9", "licence": "closed", "access": "restricted"}]
    h = PKDBClient(open_only=True).harvest_from_zip("caffeine", _make_zip(studies, [_CAF_IV], rows))
    kept = h.pk_parameters
    assert len(kept) == 1 and kept[0].measurement_type == "clearance"
    assert kept[0].value == 6.0 and kept[0].reference.study_sid == "PKDB01"
    assert any("not open" in s for s in h.skipped)          # closed dropped w/ reason
    assert any("no numeric value" in s for s in h.skipped)  # blank value skip-logged


def test_no_silent_drops(caffeine_zip):
    """Closed-licence studies are dropped *with a logged reason*, not silently."""
    h = PKDBClient(open_only=True).harvest_from_zip("caffeine", caffeine_zip)
    assert any("not open" in s for s in h.skipped)


def test_client_token_from_env(monkeypatch):
    monkeypatch.setenv("PKDB_API_TOKEN", "abc123")
    assert PKDBClient().authenticated
    monkeypatch.delenv("PKDB_API_TOKEN", raising=False)
    assert not PKDBClient(token=None).authenticated


# ── Unit harmonisation ──────────────────────────────────────────────────────
def test_harmonize_clearance_units():
    assert harmonize(100.0, "milliliter / minute", "clearance") == pytest.approx(6.0)
    assert harmonize(5.0, "liter / hour", "clearance") == pytest.approx(5.0)
    assert harmonize(1.0, "liter / minute", "clearance") == pytest.approx(60.0)


def test_harmonize_concentration_and_auc_units():
    assert harmonize(2.0, "microgram / milliliter", "cmax") == pytest.approx(2.0)
    assert harmonize(500.0, "nanogram / milliliter", "cmax") == pytest.approx(0.5)
    assert harmonize(3.0, "microgram * hour / milliliter", "auc_inf") == pytest.approx(3.0)


def test_harmonize_rejects_weight_normalised_units():
    """No body-weight guessing: mL/min/kg is un-convertible -> None, not coerced."""
    assert harmonize(3.0, "milliliter / minute / kilogram", "clearance") is None
    assert harmonize(3.0, "unknown-unit", "clearance") is None


def test_every_canonical_unit_factor_is_pinned():
    """Pin EVERY factor in _CANONICAL so a mis-scaled edit (e.g. vd mL 1e-3->1e3,
    thalf minute 1/60->60) fails a test instead of shipping a bad answer key."""
    expected = {
        ("clearance", "liter / hour"): 1.0,
        ("clearance", "milliliter / minute"): 0.06,
        ("clearance", "liter / minute"): 60.0,
        ("clearance", "milliliter / hour"): 1e-3,
        ("auc_inf", "milligram * hour / liter"): 1.0,
        ("auc_inf", "gram * hour / liter"): 1000.0,
        ("auc_inf", "microgram * hour / milliliter"): 1.0,
        ("auc_inf", "nanogram * hour / milliliter"): 1e-3,
        ("cmax", "milligram / liter"): 1.0,
        ("cmax", "microgram / milliliter"): 1.0,
        ("cmax", "nanogram / milliliter"): 1e-3,
        ("cmax", "gram / liter"): 1000.0,
        ("thalf", "hour"): 1.0,
        ("thalf", "minute"): 1.0 / 60.0,
        ("thalf", "day"): 24.0,
        ("vd", "liter"): 1.0,
        ("vd", "milliliter"): 1e-3,
    }
    for (mt, unit), factor in expected.items():
        assert harmonize(7.0, unit, mt) == pytest.approx(7.0 * factor), f"{mt}/{unit}"
    # and the /F variants share the absolute-unit factors
    assert harmonize(6.0, "milliliter / minute", "clearance/bioavailability") == pytest.approx(0.36)
    assert harmonize(50.0, "milliliter", "vd/bioavailability") == pytest.approx(0.05)


# ── Pooling + task building ─────────────────────────────────────────────────
def _pk(value: float, unit: str, sid: str, mt: str = "clearance") -> PKParameter:
    ref = Reference(sid, f"Study{sid}", "open", "public")
    return PKParameter("caf", mt, value, unit, "plasma", 10, ref)


def test_pool_parameter_geomean_and_unit_drop():
    params = [
        _pk(6.0, "liter / hour", "PKDB1"),
        _pk(100.0, "milliliter / minute", "PKDB2"),      # -> 6.0 L/h
        _pk(3.0, "milliliter / minute / kilogram", "PKDB3"),  # dropped (no weight)
    ]
    pooled, used, dropped = pool_parameter(params, "clearance")
    assert pooled == pytest.approx(6.0)      # geomean(6, 6)
    assert len(used) == 2
    assert dropped == 1


def test_route_filter_drops_nonoral_for_oral_params():
    """For an oral-exposure parameter (auc_inf), a KNOWN iv record is dropped;
    unknown-route records are kept."""
    params = [
        _pk(100.0, "milligram * hour / liter", "PKDB1", mt="auc_inf"),          # route "" kept
        PKParameter("caf", "auc_inf", 999.0, "milligram * hour / liter",
                    "plasma", 8, Reference("PKDB2", "S2", "open", "public"), route="iv"),
    ]
    pooled, used, dropped = pool_parameter(params, "auc_inf")
    assert pooled == pytest.approx(100.0)     # only the oral/unknown record
    assert dropped == 1 and len(used) == 1


def test_per_study_pooling_not_upweighted():
    """A 2-arm study must not outweigh a 1-value study: per-study geomean first."""
    params = [
        _pk(4.0, "liter / hour", "PKDB1"),   # study 1, arm A
        _pk(9.0, "liter / hour", "PKDB1"),   # study 1, arm B  -> study1 geomean = 6
        _pk(24.0, "liter / hour", "PKDB2"),  # study 2         -> study2 = 24
    ]
    pooled, used, _ = pool_parameter(params, "clearance")
    assert pooled == pytest.approx((6.0 * 24.0) ** 0.5)   # geomean(6, 24), NOT geomean(4,9,24)
    assert len(used) == 3


def test_dose_dependent_params_excluded_from_tasks():
    """Cmax/AUC are dose-dependent: no pooled task is built from them (would be an
    ill-defined dose-blind ground truth). Dose-independent CL still builds one."""
    h = DrugHarvest(drug="caffeine", pk_parameters=[
        _pk(2.0, "milligram / liter", "PKDB1", mt="cmax"),
        _pk(4.0, "milligram / liter", "PKDB2", mt="cmax"),
        _pk(5.0, "liter / hour", "PKDB1", mt="clearance"),
        _pk(6.0, "liter / hour", "PKDB2", mt="clearance"),
    ])
    tasks = build_fih_pk_tasks([h], min_studies=2)
    mts = {t.targets[0].name for t in tasks}
    assert "cmax" not in mts and "clearance" in mts


def test_missing_study_sid_skipped_no_blank_provenance():
    """A PK row without a study_sid is refused (no blank provenance, no pool-bucket
    collapse), and the drop is logged."""
    rows = [
        {"study_sid": "", "measurement_type": "clearance", "substance": "caf",
         "value": "6.0", "unit": "liter / hour", "tissue": "plasma"},
        {"study_sid": "PKDB01", "measurement_type": "clearance", "substance": "caf",
         "value": "5.5", "unit": "liter / hour", "tissue": "plasma"},
    ]
    h = PKDBClient().harvest_from_zip("caffeine", _make_zip([_CAF_STUDY], [_CAF_IV], rows))
    assert all(p.reference.study_sid for p in h.pk_parameters)   # no blank provenance
    assert len(h.pk_parameters) == 1
    assert any("missing study_sid" in s for s in h.skipped)


def test_build_fih_tasks_empty_when_no_parameters():
    """Anonymous harvest -> no PK params -> no tasks (never fabricated)."""
    h = DrugHarvest(drug="caffeine")
    assert build_fih_pk_tasks([h]) == []


def test_build_fih_tasks_from_real_answer_keys():
    h = DrugHarvest(drug="caffeine", pk_parameters=[
        _pk(5.5, "liter / hour", "PKDB1"),
        _pk(6.5, "liter / hour", "PKDB2"),
    ])
    tasks = build_fih_pk_tasks([h], min_studies=2)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.category == "fih_pk"
    assert t.targets[0].tol["type"] == "twofold"
    # ground truth is the pooled real value, provenance carries the sources
    assert t.targets[0].value == pytest.approx((5.5 * 6.5) ** 0.5, rel=1e-6)
    assert set(t.meta["source_study_sids"]) == {"PKDB1", "PKDB2"}
    # provenance is non-empty on every contributing record
    assert t.meta["raw_values"] and all(rv["study"] for rv in t.meta["raw_values"])
    # the answer must NOT be leaked into what the agent sees — by key OR by value
    assert "value" not in t.dataset and t.targets[0].name not in t.dataset
    assert not any(v == t.targets[0].value for v in t.dataset.values())


def test_min_studies_gate():
    h = DrugHarvest(drug="caffeine", pk_parameters=[_pk(6.0, "liter / hour", "PKDB1")])
    assert build_fih_pk_tasks([h], min_studies=2) == []      # only 1 study
    assert build_fih_pk_tasks([h], min_studies=1)            # relaxed -> a task


def test_fih_task_twofold_grading():
    """A prediction within 2-fold passes; outside fails."""
    h = DrugHarvest(drug="caffeine", pk_parameters=[
        _pk(6.0, "liter / hour", "PKDB1"), _pk(6.0, "liter / hour", "PKDB2")])
    tgt = build_fih_pk_tasks([h], min_studies=2)[0].targets[0]
    assert within_tolerance(11.0, tgt)      # ~1.8x -> within
    assert within_tolerance(3.5, tgt)       # ~0.58x -> within
    assert not within_tolerance(13.0, tgt)  # >2x -> out
    assert not within_tolerance(2.5, tgt)   # <0.5x -> out


# ── Optional live smoke test ────────────────────────────────────────────────
@pytest.mark.skipif(os.environ.get("PKDB_LIVE") != "1", reason="set PKDB_LIVE=1")
def test_live_caffeine_harvest():
    h = PKDBClient(verify_tls=False).harvest_drug("caffeine")
    assert h.dosing, "live PK-DB should return caffeine dosing"
    assert all(d.reference.study_sid for d in h.dosing)
