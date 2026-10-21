"""Narrowly scoped Plot Digitizer tool wrapper (`digitize_plot`).

- Zero imports from `app.evidence` or any LLM client.
- Enforces `allowed_dir=ctx.data_dir` on any local image path.
- Never embeds raw image bytes or coordinate arrays into the LLM-facing summary.
"""
from __future__ import annotations

from typing import Any

from app.compute.digitize import digitize_plot, propose_plot_from_image
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


def _run_digitize_plot(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    image_path = args.get("image_path") or args.get("source_path")
    points = args.get("points")
    allowed_dir = ctx.data_dir if ctx and ctx.data_dir else None

    if image_path and not points:
        res = propose_plot_from_image(
            image_path,
            n_candidates=int(args.get("n_candidates") or 8),
            x_range=tuple(args.get("x_range") or (0.0, 24.0)),
            y_range=tuple(args.get("y_range") or (1.0, 100.0)),
            y_scale=str(args.get("y_scale") or "linear"),
            allowed_dir=allowed_dir,
        )
    else:
        if not points:
            raise ValueError("digitize_plot requires either 'points' with 'x_axis'/'y_axis' or an 'image_path'.")
        x_axis = args.get("x_axis") or args.get("x_calibration")
        y_axis = args.get("y_axis") or args.get("y_calibration")
        if not x_axis or not y_axis:
            raise ValueError("digitize_plot requires both 'x_axis' and 'y_axis' calibration dictionaries.")
        res = digitize_plot(
            x_axis=x_axis,
            y_axis=y_axis,
            points=points,
            pixel_sigma=float(args.get("pixel_sigma", 1.5)),
            anchor_sigma_px=float(args["anchor_sigma_px"]) if args.get("anchor_sigma_px") is not None else None,
            error_type=str(args.get("error_type", "se")),
            sample_n=int(args["sample_n"]) if args.get("sample_n") is not None else None,
            strict_bounds=bool(args.get("strict_bounds", False)),
            calibration_points_2d=args.get("calibration_points_2d"),
            source_path=str(image_path) if image_path else None,
            source_sha256=args.get("source_sha256"),
            allowed_dir=allowed_dir,
            figure_label=str(args.get("figure_label") or "Figure 1"),
            approve_all=bool(args.get("approve_all", False)),
        )

    summary = (
        f"Digitized {res['n_points']} point(s) across {len(res['series'])} series "
        f"({res['n_verified']}/{res['n_points']} verified, "
        f"{res['n_out_of_bounds']} out-of-bounds; X={res['x_axis']['scale']}, "
        f"Y={res['y_axis']['scale']}, export_sha256={res['export_sha256'][:12]})."
    )
    prior_widgets = [w for w in (state.widgets or []) if w.get("type") != "plot_digitizer"]
    next_widgets = [*prior_widgets, {"type": "plot_digitizer", "data": res}]

    return ToolResult(
        summary=summary,
        writes={"widgets": next_widgets},
        result=res,
        action=f"digitize_plot({res['figure_label']!r}, n={res['n_points']}, verified={res['n_verified']})",
    )


TOOLS: list[Tool] = [
    Tool(
        name="digitize_plot",
        description=(
            "Calibrate a plot's X and Y axes (linear or log10) from two reference pixel anchors per axis, "
            "convert marker and error-bar pixels into data coordinates with propagated calibration and "
            "pixel uncertainty, or propose initial contour points from a local image."
        ),
        agent="data_manager",
        input_schema={
            "type": "object",
            "properties": {
                "x_axis": {
                    "type": "object",
                    "description": "X axis calibration: {p1_px, p2_px, v1, v2, scale ('linear'|'log10'), label, unit}.",
                },
                "y_axis": {
                    "type": "object",
                    "description": "Y axis calibration: {p1_px, p2_px, v1, v2, scale ('linear'|'log10'), label, unit}.",
                },
                "points": {
                    "type": "array",
                    "description": "Points [{id, series, u_px, v_px, v_low_px, v_high_px, origin, verified}].",
                },
                "pixel_sigma": {"type": "number", "description": "Pixel picking 1-SD uncertainty (default 1.5 px)."},
                "anchor_sigma_px": {"type": "number", "description": "Calibration anchor 1-SD uncertainty in px."},
                "error_type": {"type": "string", "enum": ["se", "sd", "ci95"]},
                "sample_n": {"type": "integer"},
                "strict_bounds": {"type": "boolean"},
                "image_path": {"type": "string", "description": "Optional local image path inside data_dir."},
                "figure_label": {"type": "string"},
                "approve_all": {"type": "boolean"},
            },
        },
        run=_run_digitize_plot,
    ),
]


def register_digitize_tools(registry: ToolRegistry) -> None:
    for t in TOOLS:
        registry.register(t)
