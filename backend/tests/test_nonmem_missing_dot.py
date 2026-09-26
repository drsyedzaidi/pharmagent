"""A NONMEM-style "." missing marker must not turn a numeric column into text.

Seen on a real upload: Conc(mcg/L) had 55/241 cells equal to "." -> pandas
read the column as str -> flexplot typed it categorical and refused
"outcome must be continuous", while NCA silently coerced and dropped the rows.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.tools.data_tools import NONMEM_NA_VALUES, _read


@pytest.fixture
def dotted_csv(tmp_path, monkeypatch):
    p = tmp_path / "dotted.csv"
    p.write_text("ID,TIME,DV,AMT,CENS\n1,0,.,100,.\n1,1,12.5,.,0\n1,2,9.1,.,0\n1,4,6.2,.,0\n2,0,.,100,.\n2,1,15,.,0\n2,2,11.3,.,0\n2,4,7.7,.,0\n")
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return p


def test_dot_is_the_nonmem_missing_marker():
    assert "." in NONMEM_NA_VALUES


def test_read_treats_dot_as_missing_so_dv_stays_numeric(dotted_csv):
    df = _read(str(dotted_csv))
    assert pd.api.types.is_numeric_dtype(df["DV"])
    assert pd.api.types.is_numeric_dtype(df["AMT"])
    assert df["DV"].isna().sum() == 2 and df["AMT"].isna().sum() == 6
    assert df["DV"].max() == 15


def test_flexplot_types_a_dotted_dv_as_continuous(dotted_csv):
    from app.compute.flexplot import _classify_variable
    df = _read(str(dotted_csv))
    assert _classify_variable(df["DV"]) == "continuous"
