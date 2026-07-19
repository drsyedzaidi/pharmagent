"""Run the parity harness under CPython (the shipping backend interpreter).

    backend/.venv/bin/python pyodide-spike/run_cpython.py [--quick]

Writes results_cpython.json next to this file. Must run with the backend venv so
numpy/scipy/pandas and the app package are importable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
sys.path.insert(0, str(_BACKEND))   # make `app.*` importable
sys.path.insert(0, str(_HERE))      # make `harness` importable

import harness  # noqa: E402  (path set above)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer fit iterations (faster smoke)")
    ap.add_argument("--csv", default=str(_BACKEND / "sample_data" / "theoph_pk.csv"))
    ap.add_argument("--out", default=str(_HERE / "results_cpython.json"))
    args = ap.parse_args()

    result = harness.run(args.csv, quick=args.quick)
    Path(args.out).write_text(json.dumps(result, indent=2))
    v = result["versions"]
    print(f"[cpython] wrote {args.out}")
    print(f"[cpython] numpy {v['numpy']}  scipy {v['scipy']}  pandas {v['pandas']}  ({result['mode']} mode)")
    print(f"[cpython] FOCE-I CL={result['focei']['theta']['CL']:.4f}  "
          f"SAEM CL={result['saem']['theta']['CL']:.4f}  "
          f"converged={result['focei']['converged']}/{result['saem']['converged']}")


if __name__ == "__main__":
    main()
