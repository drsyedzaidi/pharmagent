"""Generate a deterministic NONMEM-style oral PK dataset for demos/tests.

12 subjects, two dose groups (100, 300 mg), 1-compartment oral PK with
lognormal IIV on CL and V and proportional residual error. Seeded.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

random.seed(2025)

TIMES = [0.0, 0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48]
DOSES = [100.0, 300.0]
KA = 1.1            # 1/h
TVCL = 5.0         # L/h
TVV = 50.0         # L
OM_CL = 0.09       # variance (~30% CV)
OM_V = 0.04
PROP_ERR = 0.10

out = Path(__file__).parent / "oral_pk.csv"
rows = []
sid = 0
for dose in DOSES:
    for _ in range(6):
        sid += 1
        cl = TVCL * math.exp(random.gauss(0, OM_CL ** 0.5))
        v = TVV * math.exp(random.gauss(0, OM_V ** 0.5))
        ke = cl / v
        # dosing record
        rows.append({"ID": sid, "TIME": 0, "DV": ".", "AMT": dose,
                     "EVID": 1, "MDV": 1, "CMT": 1, "DOSE": dose})
        for t in TIMES:
            if t == 0:
                continue
            # 1-cmt oral, F=1
            conc = (dose * KA) / (v * (KA - ke)) * (math.exp(-ke * t) - math.exp(-KA * t))
            conc *= (1 + random.gauss(0, PROP_ERR))
            conc = max(conc, 0.0)
            rows.append({"ID": sid, "TIME": t, "DV": round(conc, 4), "AMT": ".",
                         "EVID": 0, "MDV": 0, "CMT": 1, "DOSE": dose})

with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT", "DOSE"])
    w.writeheader()
    w.writerows(rows)

print(f"wrote {out} ({len(rows)} rows, {sid} subjects)")
