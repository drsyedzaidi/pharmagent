"""CDISC ADaM-style export: ADPP/ADPC structure, zip package, define.xml validity."""
import io
import time
import xml.etree.ElementTree as ET
import zipfile
from types import SimpleNamespace

import pandas as pd

from app.core.cdisc import build_adpp, build_package
from app.core.pharmstate import PharmState


def _state():
    return PharmState(
        nca_parameters=[
            {"subject": 1, "dose": 100, "Cmax": 10.0, "Tmax": 1.0,
             "AUC_last": 40.0, "AUC_inf": 50.0, "CL_F": 2.0, "Vz_F": 20.0, "t_half": 7.0},
            {"subject": 2, "dose": 100, "Cmax": 12.0, "Tmax": 1.5,
             "AUC_last": 44.0, "AUC_inf": 55.0, "CL_F": 1.8, "Vz_F": 22.0, "t_half": 8.0},
        ],
        dataset_metadata={"dataset_id": "ABC",
                          "detected_roles": {"ID": "ID", "TIME": "TIME", "DV": "DV"}},
    )


def test_adpp_long_bds_with_standard_paramcds():
    rows, cols = build_adpp(_state())
    pcds = {r["PARAMCD"] for r in rows}
    assert {"CMAX", "TMAX", "AUCLST", "AUCIFO", "CLFO", "VZFO", "LAMZHL"} <= pcds
    assert all(r["USUBJID"].startswith("ABC-") for r in rows)
    assert "AVAL" in cols and "PARAMCD" in cols


def test_package_zip_contents_and_define_valid():
    df = pd.DataFrame({"ID": [1, 1, 2], "TIME": [1.0, 2.0, 1.0], "DV": [5.0, 3.0, 6.0]})
    body = build_package(_state(), df, {"ID": "ID", "TIME": "TIME", "DV": "DV"})
    z = zipfile.ZipFile(io.BytesIO(body))
    names = set(z.namelist())
    assert {"ADPP.csv", "ADPC.csv", "define.xml", "README.txt"} <= names
    # define.xml is well-formed and CDISC ODM-rooted
    root = ET.fromstring(z.read("define.xml"))
    assert root.tag.endswith("ODM")
    # ADPC has header + 3 concentration rows
    adpc = z.read("ADPC.csv").decode().strip().splitlines()
    assert len(adpc) == 4 and adpc[0].startswith("STUDYID")
    # ADPP non-empty
    assert len(z.read("ADPP.csv").decode().strip().splitlines()) > 1


def test_package_without_dataframe():
    # No raw concentrations available -> ADPC empty header only, still valid zip
    body = build_package(_state(), None, {})
    z = zipfile.ZipFile(io.BytesIO(body))
    assert z.read("ADPC.csv").decode().strip().count("\n") == 0  # header only
    ET.fromstring(z.read("define.xml"))


def _zip_of_one_member() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x", b"")
    return buf.getvalue()


def test_package_bytes_do_not_depend_on_the_wall_clock(monkeypatch):
    # The export's sha256 is recorded in a separately requested manifest, so two builds of
    # the same session must be byte-identical even when a 2 s DOS-time tick falls between them.
    df = pd.DataFrame({"ID": [1, 1, 2], "TIME": [1.0, 2.0, 1.0], "DV": [5.0, 3.0, 6.0]})
    roles = {"ID": "ID", "TIME": "TIME", "DV": "DV"}
    now = [1_700_000_000.0]
    monkeypatch.setattr(zipfile, "time", SimpleNamespace(time=lambda: now[0], localtime=time.localtime))
    first, control = build_package(_state(), df, roles), _zip_of_one_member()
    now[0] += 2.0
    assert _zip_of_one_member() != control                 # the fake clock does reach zipfile
    assert build_package(_state(), df, roles) == first


def test_package_members_keep_order_compression_and_permissions():
    z = zipfile.ZipFile(io.BytesIO(build_package(_state(), None, {})))
    assert z.namelist() == ["ADPP.csv", "ADPC.csv", "define.xml", "README.txt"]
    for info in z.infolist():
        assert info.compress_type == zipfile.ZIP_DEFLATED, info.filename
        assert info.external_attr == 0o600 << 16, info.filename
