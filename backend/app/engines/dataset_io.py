"""Serialize the app's in-memory ``subjects`` list to a NONMEM-style dataset.

External engines (nlmixr2, FeRx, Monolix, NONMEM) consume a rectangular
ID/TIME/DV/AMT/EVID/CMT table, not the app's per-subject dicts. This is the
converter the mapping blueprint flagged as missing (OQ-4). Column conventions
follow the standard nlmixr2 layout:

  * dose rows  -> EVID=1, AMT=amt, DV=".",  CMT=depot(1) for oral / central(1) for IV
  * obs rows   -> EVID=0, AMT=".", DV=conc, CMT=central(2 oral / 1 IV)

For an oral (first-order absorption) model the ODE has depot(1)+central(2); for
IV there is a single central compartment (1). Rows are ordered by TIME with dose
events before observations at the same time.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

_COLUMNS = ["ID", "TIME", "DV", "AMT", "EVID", "CMT", "WT"]


def subjects_to_records(subjects: list[dict], *, is_iv: bool) -> list[dict[str, Any]]:
    """Flatten subjects into NONMEM-style event records."""
    dose_cmt = 1
    obs_cmt = 1 if is_iv else 2
    rows: list[dict[str, Any]] = []
    for s in subjects:
        sid = s["subject"]
        wt = float(s.get("wt", 70.0))
        for d in s.get("doses", []):
            rows.append({"ID": sid, "TIME": float(d["time"]), "DV": ".",
                         "AMT": float(d["amt"]), "EVID": 1, "CMT": dose_cmt, "WT": wt})
        for t, c in zip(s.get("obs_t", []), s.get("obs_c", [])):
            rows.append({"ID": sid, "TIME": float(t), "DV": float(c),
                         "AMT": ".", "EVID": 0, "CMT": obs_cmt, "WT": wt})
    # stable sort: by (ID, TIME) with dose (EVID=1) before obs (EVID=0) at same time
    rows.sort(key=lambda r: (str(r["ID"]), r["TIME"], -int(r["EVID"])))
    return rows


def subjects_to_frame(subjects: list[dict], *, is_iv: bool) -> pd.DataFrame:
    return pd.DataFrame(subjects_to_records(subjects, is_iv=is_iv), columns=_COLUMNS)


def write_nonmem_csv(subjects: list[dict], path: str, *, is_iv: bool) -> str:
    """Write the NONMEM-style CSV and return the path."""
    subjects_to_frame(subjects, is_iv=is_iv).to_csv(path, index=False)
    return path
