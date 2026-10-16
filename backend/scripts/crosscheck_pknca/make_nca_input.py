#!/usr/bin/env python3
"""Build the ONE table both engines analyse: first dosing interval of the pharmaverse adppk fixture.

    python3 make_nca_input.py [adppk_subset.csv] [out.csv]

Rules (written down so the comparison is reproducible, not tuned):
  * first dose only: keep the dose record at nominal time 0, drop the doses at 24 h and 48 h
    (the concentrations are a single-dose profile; later BLQ values show no accumulation);
  * observations: EVID == 0, nominal time NFRLT <= 24 h;
  * a value that is blank in the source is a MISSING observation and is dropped (not turned into zero);
  * a below-LOQ value that the source codes as 0 (the pre-dose sample) is kept as 0 and flagged BLQ = 1;
  * time is the NOMINAL time NFRLT; the concentration is DV, unchanged.
Output columns: ID, TIME, DV, AMT, EVID, BLQ (NONMEM-style, so PharmAgent detects every role by name).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE.parents[1] / "tests" / "fixtures" / "pharmaverse" / "adppk_subset.csv"
WINDOW_H = 24.0
COLUMNS = ["ID", "TIME", "DV", "AMT", "EVID", "BLQ"]


def build(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        time = float(r["NFRLT"])
        if r["EVID"] == "1":
            if time == 0.0:
                out.append({"ID": r["USUBJID"], "TIME": "0", "DV": "", "AMT": r["AMT"], "EVID": "1", "BLQ": "0"})
            continue
        if time > WINDOW_H or r["DV"] in ("", "NA"):
            continue
        out.append({"ID": r["USUBJID"], "TIME": r["NFRLT"], "DV": r["DV"], "AMT": "0", "EVID": "0",
                    "BLQ": "1" if r["BLQFL"] == "Y" else "0"})
    # dose record first at a shared time, then by time: stable, NONMEM-like
    out.sort(key=lambda x: (x["ID"], float(x["TIME"]), -int(x["EVID"])))
    return out


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else DEFAULT_IN
    dst = Path(argv[2]) if len(argv) > 2 else HERE / "nca_input.csv"
    with src.open(encoding="utf-8", newline="") as fh:
        rows = build(list(csv.DictReader(fh)))
    with dst.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len({r['ID'] for r in rows})} subjects, {sum(r['EVID'] == '0' for r in rows)} observations -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
