"""Reference values for the validation suite.

Each entry pins a canonical public dataset to (a) published popPK literature
consensus values with generous bands, and (b) an OPTIONAL slot for the user's
own NONMEM / Monolix / nlmixr2 run so exact tool concordance can be asserted.

Why bands and not a single number: the same dataset yields slightly different
estimates across estimation methods (FO / FOCE-I / SAEM / Laplace), data
subsets, and tools. The bands below are deliberately wide (~+/-20-30%) so the
test is a meaningful *regression + plausibility gate* anchored to the published
literature — not a brittle equality check. For a strict cross-tool equality
check, populate ``tool_reference`` (see THEOPHYLLINE) with your own run's
estimates; the suite then additionally asserts agreement within ``rel_tol``.
"""
from __future__ import annotations

# ── Theophylline ─────────────────────────────────────────────────────────────
# Boeckmann, Sheiner & Beal theophylline data (R ``datasets::Theoph``;
# Upton 1982): 12 subjects, single oral dose (~320 mg absolute), serum
# theophylline (mg/L) over 0-25 h. The canonical one-compartment, first-order
# absorption population PK model. Published estimates for the absolute-dose
# parameterization (nlme [Pinheiro & Bates 2000], nlmixr2 ``theo_sd`` FOCE-i,
# NONMEM ADVAN2 TRANS2) cluster tightly around CL/F~2.7 L/h, V/F~32 L,
# Ka~1.5 /h, t1/2~8 h, with between-subject CV ~25-40% and ~10-20% residual.
THEOPHYLLINE = {
    "dataset": "theoph_pk.csv",
    "model": "oral_1cmt",
    "description": ("Boeckmann/Sheiner/Beal Theophylline (R datasets::Theoph): "
                    "12 subjects, single oral dose, conc mg/L over 0-25 h."),
    "literature_source": ("nlme (Pinheiro & Bates 2000), nlmixr2 theo_sd FOCE-i, "
                          "NONMEM ADVAN2/TRANS2 — consensus 1-cmt first-order model"),
    # name -> (published value, (lo, hi) acceptance band)
    "literature": {
        "CL":         (2.7, (2.2, 3.3)),     # L/h
        "V":          (32.0, (27.0, 40.0)),   # L
        "KA":         (1.5, (0.8, 2.5)),      # 1/h
        "t_half":     (8.0, (6.0, 11.0)),     # h (NCA terminal)
        "iiv_cl_pct": (30.0, (10.0, 55.0)),   # between-subject %CV on CL
        "sigma_prop": (0.15, (0.05, 0.35)),   # proportional residual
    },
    # OPTIONAL exact concordance with YOUR estimator. To enable: set this to e.g.
    #   {"tool": "NONMEM 7.5 FOCE-I", "rel_tol": 0.15,
    #    "CL": 2.81, "V": 32.6, "KA": 1.49}
    # taken from your .lst / run record. The suite then asserts PharmAgent's
    # FOCE-I theta matches each provided parameter within rel_tol (relative).
    "tool_reference": None,
}

DATASETS = {"theophylline": THEOPHYLLINE}
