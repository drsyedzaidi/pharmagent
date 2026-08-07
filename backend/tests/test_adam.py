"""ADaM -> NONMEM adapter.

The fixtures mirror the shapes that matter in a real deliverable: a mixed
IV/SC dataset, a 2 h infusion, a repeat-dose subject, a BLQ record, and a
placebo subject with no dose.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.compute.adam import build_analysis_dataset


def _adsl():
    return pd.DataFrame([
        {"USUBJID": "S-01", "STUDYID": "ST1", "PKFL": "Y", "WEIGHTBL": 70, "AGE": 40,
         "SEX": "M", "EGFRBL": 90, "ALBBL": 4.0, "POPULATION": "healthy"},
        {"USUBJID": "S-02", "STUDYID": "ST1", "PKFL": "Y", "WEIGHTBL": 60, "AGE": 55,
         "SEX": "F", "EGFRBL": 70, "ALBBL": 3.5, "POPULATION": "disease"},
        # placebo: flagged but never dosed -> must be dropped, not crash
        {"USUBJID": "S-03", "STUDYID": "ST1", "PKFL": "Y", "WEIGHTBL": 80, "AGE": 33,
         "SEX": "M", "EGFRBL": 95, "ALBBL": 4.2, "POPULATION": "healthy"},
    ])


def _adex():
    return pd.DataFrame([
        # S-01: 2 h IV infusion
        {"USUBJID": "S-01", "EXDOSE": 100, "EXROUTE": "IV",
         "EXSTDTC": "2023-01-01T08:00", "EXENDTC": "2023-01-01T10:00"},
        # S-02: two SC doses, 14 days apart, instantaneous
        {"USUBJID": "S-02", "EXDOSE": 200, "EXROUTE": "SC",
         "EXSTDTC": "2023-01-01T08:00", "EXENDTC": "2023-01-01T08:00"},
        {"USUBJID": "S-02", "EXDOSE": 200, "EXROUTE": "SC",
         "EXSTDTC": "2023-01-15T08:00", "EXENDTC": "2023-01-15T08:00"},
        {"USUBJID": "S-03", "EXDOSE": 0, "EXROUTE": "SC",          # placebo
         "EXSTDTC": "2023-01-01T08:00", "EXENDTC": "2023-01-01T08:00"},
    ])


def _adpc():
    return pd.DataFrame([
        {"USUBJID": "S-01", "PCTESTCD": "DRUG", "PCORRES": "BLQ", "PCSTRESN": None,
         "PCDTC": "2023-01-01T08:00"},
        {"USUBJID": "S-01", "PCTESTCD": "DRUG", "PCORRES": "12.5", "PCSTRESN": 12.5,
         "PCDTC": "2023-01-01T20:00"},                       # +12 h
        {"USUBJID": "S-02", "PCTESTCD": "DRUG", "PCORRES": "3.2", "PCSTRESN": 3.2,
         "PCDTC": "2023-01-02T08:00"},                       # +24 h
        {"USUBJID": "S-02", "PCTESTCD": "DRUG", "PCORRES": "8.0", "PCSTRESN": 8.0,
         "PCDTC": "2023-01-16T08:00"},                       # +360 h (after dose 2)
        {"USUBJID": "S-03", "PCTESTCD": "DRUG", "PCORRES": "0.0", "PCSTRESN": 0.0,
         "PCDTC": "2023-01-02T08:00"},
    ])


def _build(**kw):
    return build_analysis_dataset(_adsl(), _adex(), _adpc(), **kw)


def test_shapes_routes_and_placebo_exclusion():
    r = _build()
    assert r.meta["n_subjects"] == 2                  # S-03 dropped: never dosed
    assert set(r.id_map["ID"]) == {1, 2}              # sequential from 1
    doses = r.df[r.df.EVID == 1]
    assert len(doses) == 3                            # 1 IV + 2 SC
    # route -> compartment: IV central (2), SC depot (1)
    assert list(r.df[(r.df.EVID == 1) & (r.df.RATE > 0)].CMT) == [2]
    assert set(doses[doses.RATE == 0].CMT) == {1}


def test_infusion_becomes_a_rate_not_a_bolus():
    r = _build()                                      # hours
    iv = r.df[(r.df.EVID == 1) & (r.df.CMT == 2)].iloc[0]
    assert iv.RATE == pytest.approx(100 / 2.0)        # 100 mg over 2 h
    assert any("infusion" in i for i in r.issues)


def test_rate_follows_the_time_unit():
    """RATE is amount per TIME unit — switching to days must rescale it, or the
    infusion silently delivers 24x too little drug."""
    hr = _build(time_unit="hour").df
    dy = _build(time_unit="day").df
    r_h = hr[(hr.EVID == 1) & (hr.RATE > 0)].RATE.iloc[0]
    r_d = dy[(dy.EVID == 1) & (dy.RATE > 0)].RATE.iloc[0]
    assert r_d == pytest.approx(r_h * 24)


def test_time_is_relative_to_each_subjects_first_dose():
    r = _build()
    d = r.df
    s2 = d[(d.ID == 2)].sort_values("TIME")
    assert list(s2[s2.EVID == 1].TIME) == [0.0, pytest.approx(336.0)]   # 14 d apart
    assert s2[s2.EVID == 0].TIME.max() == pytest.approx(360.0)
    # day mode is the same clock, divided
    assert _build(time_unit="day").df.TIME.max() == pytest.approx(15.0)


def test_blq_uses_m3_coding_and_lloq_is_reported_as_inferred():
    r = _build()
    blq = r.df[r.df.BLQ == 1]
    assert len(blq) == 1
    assert blq.iloc[0].DV == 0.0 and blq.iloc[0].MDV == 1
    assert r.meta["lloq"] == pytest.approx(3.2)        # smallest quantified
    assert any("LLOQ not stated" in i for i in r.issues)
    # an explicit LLOQ suppresses the inference notice
    assert not any("LLOQ not stated" in i for i in _build(lloq=1.0).issues)


def test_dose_sorts_before_an_observation_at_the_same_time():
    r = _build()
    first = r.df[r.df.ID == 1].iloc[0]
    assert first.EVID == 1 and first.TIME == 0.0       # dose precedes the t=0 sample


def test_covariates_are_curated_and_encoded_numerically():
    r = _build()
    assert {"WT", "AGE", "EGFR", "ALB", "SEXN", "DISEASE"} <= set(r.df.columns)
    assert r.df[r.df.ID == 1].SEXN.iloc[0] == 1        # M
    assert r.df[r.df.ID == 2].DISEASE.iloc[0] == 1     # disease population
    assert "USUBJID" not in r.df.columns               # would become a covariate
    assert "USUBJID" in _build(keep_usubjid=True).df.columns


def test_roles_are_detectable_by_the_schema_extractor():
    """The whole point is that the output drops into the existing pipeline."""
    from app.core.schema_extractor import detect_roles
    roles = detect_roles(list(_build().df.columns))
    assert {"ID", "TIME", "DV", "AMT", "EVID"} <= set(roles.values())


def test_multiple_analytes_must_be_disambiguated():
    pc = _adpc()
    pc.loc[pc.index[-1], "PCTESTCD"] = "METABOLITE"
    with pytest.raises(ValueError, match="analyte"):
        build_analysis_dataset(_adsl(), _adex(), pc)
    r = build_analysis_dataset(_adsl(), _adex(), pc, analyte="DRUG")
    assert r.meta["analyte"] == "DRUG"


def test_no_modelable_subject_raises_rather_than_returning_an_empty_frame():
    ex = _adex()
    ex["EXDOSE"] = 0
    with pytest.raises(ValueError, match="no subject"):
        build_analysis_dataset(_adsl(), ex, _adpc())


def test_bad_time_unit_is_rejected():
    with pytest.raises(ValueError, match="time_unit"):
        _build(time_unit="week")
