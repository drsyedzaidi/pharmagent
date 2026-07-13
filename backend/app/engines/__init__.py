"""Cross-engine pharmacometric orchestration (v0).

Run the same candidate models across multiple estimation engines (this app's
FOCE-I/SAEM today; nlmixr2/FeRx/Monolix via adapters), normalize every fit into
one ``EngineResult``, and pick a winner on an engine-agnostic footing —
predictions and VPC coverage — because native OFV/AIC/BIC are NOT comparable
across estimation algorithms.

See ``pharmagent/PHARMACOMETRICSBENCH.md`` sibling doc and the module docstrings.
"""
from .base import (
    NATIVE_VERSION,
    CandidateSpec,
    EngineAdapter,
    EngineResult,
    aic_bic,
    k_from_result,
)
from .ensemble import build_ensemble
from .mock import MockEngineAdapter
from .native import PharmAgentAdapter
from .nlmixr2 import Nlmixr2Adapter
from .runner import run_matrix, run_matrix_subjects
from .scoring import score_predictions, vpc_coverage
from .select import SELECTION_METRIC, select_winner

__all__ = [
    "EngineResult", "CandidateSpec", "EngineAdapter",
    "aic_bic", "k_from_result", "NATIVE_VERSION",
    "PharmAgentAdapter", "MockEngineAdapter", "Nlmixr2Adapter", "build_ensemble",
    "run_matrix", "run_matrix_subjects",
    "score_predictions", "vpc_coverage",
    "select_winner", "SELECTION_METRIC",
]
