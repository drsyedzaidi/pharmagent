"""Visualization tools (data_manager): exploratory Flexplot-style data plots.

A single read-only tool that turns a variable selection into fully precomputed
plot geometry (``app.compute.flexplot``). Owned by ``data_manager``, whose
charter is to "load, profile, validate, and visualize PK datasets". The tool
writes only the derived geometry (``flexplot_data``) to state — never raw rows.
"""
from __future__ import annotations

from typing import Any

from app.compute.flexplot import flexplot
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult


def generate_flexplot(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Build a flexplot from the loaded dataset and stash the geometry in state."""
    dsid = args.get("dataset_id") or state.dataset_id
    if not dsid or dsid not in ctx.dataset_store:
        raise ValueError("no dataset loaded — upload a CSV first")
    df = ctx.dataset_store[dsid]

    y = args.get("y")
    if not y or y not in df.columns:
        raise ValueError(f"outcome variable {y!r} is not in the dataset")
    x = args.get("x") or None
    if x and x not in df.columns:
        raise ValueError(f"predictor variable {x!r} is not in the dataset")

    payload = flexplot(
        df,
        y=y,
        x=x,
        color_by=args.get("color_by") or None,
        panel_by=args.get("panel_by") or None,
        fit=args.get("fit", "loess"),
        geom=args.get("geom", "points"),
        center=args.get("center", "median_iqr"),
        ghost=bool(args.get("ghost", False)),
        log_y=bool(args.get("log_y", False)),
        jitter=float(args.get("jitter", 0.2)),
        n_bins=int(args.get("n_bins", 10)),
        ci=float(args.get("ci", 0.95)),
    )

    detail = f" vs {x}" if x else ""
    return ToolResult(
        summary=(f"Flexplot ready: {payload['kind']} of {y}{detail} "
                 f"(n={payload['summary']['n']})."),
        action=f"generate_flexplot({dsid}: {y}{detail})",
        writes={"flexplot_data": payload},
        result={"kind": payload["kind"], "n": payload["summary"]["n"]},
    )


TOOLS = [
    Tool(
        "generate_flexplot",
        "Build an exploratory Flexplot-style plot (scatter / dot plot / histogram / "
        "density) of dataset variables, with an optional loess or linear fit, "
        "confidence band, colour grouping, and panels. Set 'y' to the outcome; "
        "'x' to an optional predictor.",
        "data_manager",
        {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "y": {"type": "string", "description": "outcome variable (required)"},
                "x": {"type": "string", "description": "optional predictor variable"},
                "color_by": {"type": "string", "description": "optional categorical colour grouping"},
                "panel_by": {"type": "string", "description": "optional facet variable"},
                "fit": {"type": "string", "enum": ["loess", "linear", "none"]},
                "geom": {"type": "string", "enum": ["points", "line", "smooth", "density"]},
                "center": {"type": "string", "enum": ["median_iqr", "mean_se", "mean_sd"]},
                "ghost": {"type": "boolean"},
                "log_y": {"type": "boolean"},
                "jitter": {"type": "number"},
                "n_bins": {"type": "integer"},
                "ci": {"type": "number"},
            },
            "required": ["y"],
        },
        generate_flexplot,
    ),
]
