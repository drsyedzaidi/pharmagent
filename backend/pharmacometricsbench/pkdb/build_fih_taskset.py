"""Build the FIH-PK task set from PK-DB — harvest → pool → write JSONL.

One command turns real, cited PK-DB answer keys into 2-fold-graded ``fih_pk`` tasks:

    python -m pharmacometricsbench.pkdb.build_fih_taskset            # anonymous
    PKDB_API_TOKEN=... python -m pharmacometricsbench.pkdb.build_fih_taskset

Anonymous access exposes dosing/study data only, so the answer-key harvest is empty
and this writes **zero tasks** — by design, never a fabricated one. Supply a free
PK-DB token (register at pk-db.com; see the README) and the same command produces
the real cited task set. The output is a standard JSONL the runner can load with
``--tasks``.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

from ..spec import dump_tasks
from .fih_tasks import build_fih_pk_tasks
from .loader import SEED_DRUGS, PKDBClient


def build(drugs: list[str], token: str | None, *, verify_tls: bool = True,
          min_studies: int = 2):
    client = PKDBClient(token=token, open_only=True, verify_tls=verify_tls)
    harvests = []
    for drug in drugs:
        try:
            harvests.append(client.harvest_drug(drug))
        except Exception as exc:  # a single drug's network hiccup shouldn't abort
            print(f"  ! {drug}: harvest failed ({exc})")
    tasks = build_fih_pk_tasks(harvests, min_studies=min_studies)
    return client, harvests, tasks


def _report(client: PKDBClient, harvests, tasks) -> None:
    n_keys = sum(len(h.pk_parameters) for h in harvests)
    print(f"\nPK-DB FIH-task build  (authenticated={client.authenticated})")
    print("-" * 60)
    for h in harvests:
        params = Counter(p.measurement_type for p in h.pk_parameters)
        print(f"  {h.drug:14s} studies={len(h.studies):3d}  answer-keys={len(h.pk_parameters):3d}"
              f"  {dict(params) if params else ''}")
    print("-" * 60)
    print(f"  totals: answer-keys={n_keys}  ->  fih tasks={len(tasks)}")
    if tasks:
        by = Counter(t.targets[0].name for t in tasks)
        print(f"  tasks by parameter: {dict(by)}")
    else:
        note = ("\n  0 tasks written — no drug-level PK answer keys were harvested.")
        if not client.authenticated:
            note += ("\n  This is expected anonymously: the PK-DB outputs table is"
                     "\n  auth-gated. Register a free account at pk-db.com, then run"
                     "\n  with PKDB_API_TOKEN set to produce the real cited task set.")
        print(note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drugs", nargs="*", default=SEED_DRUGS,
                    help="drugs to harvest (default: the FIH seed set)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                    "..", "tasks", "fih_v0.jsonl"))
    ap.add_argument("--min-studies", type=int, default=2,
                    help="min distinct source studies to pool a parameter")
    ap.add_argument("--token", default=os.environ.get("PKDB_API_TOKEN"))
    ap.add_argument("--no-verify-tls", action="store_true")
    args = ap.parse_args()

    client, harvests, tasks = build(
        args.drugs, args.token, verify_tls=not args.no_verify_tls,
        min_studies=args.min_studies)
    _report(client, harvests, tasks)
    if tasks:
        out = os.path.abspath(args.out)
        dump_tasks(tasks, out)
        print(f"\n  wrote {len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
