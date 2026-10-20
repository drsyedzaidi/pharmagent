#!/usr/bin/env python3
"""Rebuild the pharmaverse PK fixtures from the two pinned source CSVs. Deterministic; copies rows byte-for-byte.

    python3 build_subset.py <pc.csv from pharmaversesdtm@7ec53ea> <adppk.csv from pharmaverseadam@c7d3881> [outdir]

Subject rule (documented, not tuned): subjects present in BOTH files, ordered by USUBJID; from each COHORT (1, 2) the
first 3 with >= 14 quantifiable pc rows go into BOTH subsets. In addition, the pc subject with the FEWEST quantifiable
rows over the whole of pc (ties: first by USUBJID) goes into pc_subset.csv only, as the all-BLQ edge case (such
subjects exist only in pc; adppk has none). Nothing else is selected, no value is edited, no column is dropped.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

SDTM_COMMIT = "7ec53ea88dfe1c168f9efbef59233c4426813b89"        # pharmaverse/pharmaversesdtm (inst/extdata/pc.csv)
ADAM_COMMIT = "c7d3881173b01504286cf6784622420401ddaa41"        # pharmaverse/pharmaverseadam (inst/extdata/adppk.csv)
MIN_QUANT, PER_COHORT = 14, 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = list(csv.DictReader(lines))
    return lines[0], lines[1:], rows


def select(pc_rows: list[dict], ad_rows: list[dict]) -> tuple[list[str], str]:
    """(subjects in both subsets, the pc-only all-BLQ edge-case subject)."""
    quant: dict[str, int] = {}
    for r in pc_rows:
        quant.setdefault(r["USUBJID"], 0)
        if "BLQ" not in r["PCORRES"]:
            quant[r["USUBJID"]] += 1
    cohort = {r["USUBJID"]: r["COHORT"] for r in ad_rows}
    both = sorted(set(quant) & set(cohort))
    chosen: list[str] = []
    for c in sorted(set(cohort.values())):
        chosen += [s for s in both if cohort[s] == c and quant[s] >= MIN_QUANT][:PER_COHORT]
    edge = min(quant, key=lambda s: (quant[s], s))
    return sorted(chosen), edge


def write_subset(src: Path, dst: Path, subjects: set[str]) -> int:
    head, body, rows = read(src)
    keep = [line for line, row in zip(body, rows, strict=True) if row["USUBJID"] in subjects]
    dst.write_text(head + "".join(keep), encoding="utf-8")
    return len(keep)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    pc, ad = Path(argv[1]), Path(argv[2])
    out = Path(argv[3]) if len(argv) > 3 else Path(__file__).parent
    _, _, pc_rows = read(pc)
    _, _, ad_rows = read(ad)
    subjects, edge = select(pc_rows, ad_rows)
    n_pc = write_subset(pc, out / "pc_subset.csv", set(subjects) | {edge})
    n_ad = write_subset(ad, out / "adppk_subset.csv", set(subjects))
    manifest = {
        "sources": {
            "pc": {"repo": "pharmaverse/pharmaversesdtm", "commit": SDTM_COMMIT, "path": "inst/extdata/pc.csv",
                   "sha256": sha256(pc), "rows": len(pc_rows)},
            "adppk": {"repo": "pharmaverse/pharmaverseadam", "commit": ADAM_COMMIT, "path": "inst/extdata/adppk.csv",
                      "sha256": sha256(ad), "rows": len(ad_rows)},
        },
        "selection": {"rule": __doc__.split("Subject rule")[1].split("Nothing else")[0].strip(" (:)\n"),
                      "subjects_in_both": subjects, "pc_only_all_blq_subject": edge,
                      "min_quantifiable_pc_rows": MIN_QUANT, "per_cohort": PER_COHORT},
        "subset": {"pc_subset.csv": {"sha256": sha256(out / "pc_subset.csv"), "rows": n_pc},
                   "adppk_subset.csv": {"sha256": sha256(out / "adppk_subset.csv"), "rows": n_ad}},
        "modifications": "row subsetting by USUBJID only; every kept row is byte-identical to the source row",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(subjects)} subjects in both + 1 pc-only edge subject; pc {n_pc} rows, adppk {n_ad} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
