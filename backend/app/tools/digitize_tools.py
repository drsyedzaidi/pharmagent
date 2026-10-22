"""Narrowly scoped Plot Digitizer tool wrappers (`digitize_plot` and `propose_plot_from_image`).

- Zero imports from `app.evidence` or any LLM client.
- Enforces mandatory `allowed_dir=ctx.data_dir` on any local image path (refuses missing/empty `data_dir`).
- Marks automatic image contour proposal (`propose_plot_from_image`) as `expensive=True, proposable=True`
  and routes it through bounded `JobManager` admission control (`submit_plot_proposal_job`).
- Never embeds raw image bytes, coordinate arrays, or source filenames into `ToolResult.summary` or `ToolResult.action`.
"""
from __future__ import annotations

from typing import Any

from app.compute.digitize import digitize_plot, propose_plot_from_image
from app.core.audit import AuditChain
from app.core.jobs import JobManager
from app.core.pharmstate import PharmState
from app.tools.base import ExpensiveToolError, Tool, ToolContext, ToolRegistry, ToolResult


def _build_digitizer_tool_result(
    state: PharmState,
    res: dict[str, Any],
    *,
    tool_name: str,
) -> ToolResult:
    ref_hash = (res.get("source_sha256") or res["export_sha256"])[:12]
    summary = (
        f"Digitized {res['n_points']} point(s) across {len(res['series'])} series "
        f"(status={res['status']}, {res['n_verified']}/{res['n_points']} verified, "
        f"bounds={res['bounds_status']}, oob={res['n_out_of_bounds']}; "
        f"X={res['x_axis']['scale']}, Y={res['y_axis']['scale']}, ref={ref_hash})."
    )
    prior_widgets = [w for w in (state.widgets or []) if w.get("type") != "plot_digitizer"]
    next_widgets = [*prior_widgets, {"type": "plot_digitizer", "data": res}]

    return ToolResult(
        summary=summary,
        writes={"widgets": next_widgets},
        result=res,
        action=(
            f"{tool_name}(ref={ref_hash}, status={res['status']}, "
            f"n={res['n_points']}, verified={res['n_verified']})"
        ),
    )


def _run_digitize_plot(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    image_path = args.get("image_path") or args.get("source_path")
    points = args.get("points")
    allowed_dir = (ctx.data_dir or "").strip() if ctx and ctx.data_dir else ""

    if image_path and not allowed_dir:
        raise ValueError("allowed_dir is required at the tool boundary when loading an image.")

    if image_path and not points:
        raise ExpensiveToolError(
            "Automatic image contour proposal is expensive — invoke 'propose_plot_from_image' "
            "through bounded job control (allow_expensive=True)."
        )

    if not points:
        raise ValueError("digitize_plot requires 'points' with 'x_axis' and 'y_axis'.")
    x_axis = args.get("x_axis") or args.get("x_calibration")
    y_axis = args.get("y_axis") or args.get("y_calibration")
    if not x_axis or not y_axis:
        raise ValueError("digitize_plot requires both 'x_axis' and 'y_axis' calibration dictionaries.")

    res = digitize_plot(
        x_axis=x_axis,
        y_axis=y_axis,
        plot_roi=args.get("plot_roi"),
        points=points,
        pixel_sigma=float(args.get("pixel_sigma", 1.5)),
        anchor_sigma_px=float(args["anchor_sigma_px"]) if args.get("anchor_sigma_px") is not None else None,
        error_type=str(args.get("error_type", "se")),
        sample_n=int(args["sample_n"]) if args.get("sample_n") is not None else None,
        strict_bounds=bool(args.get("strict_bounds", False)),
        oob_overrides=args.get("oob_overrides"),
        calibration_points_2d=args.get("calibration_points_2d"),
        source_path=str(image_path) if image_path else None,
        source_sha256=args.get("source_sha256"),
        allowed_dir=allowed_dir if image_path else None,
        figure_label=str(args.get("figure_label") or "Figure 1"),
        approve_all=bool(args.get("approve_all", False)),
    )
    return _build_digitizer_tool_result(state, res, tool_name="digitize_plot")


def _run_propose_plot_from_image(
    state: PharmState,
    ctx: ToolContext,
    args: dict[str, Any],
) -> ToolResult:
    image_path = args.get("image_path") or args.get("source_path")
    allowed_dir = (ctx.data_dir or "").strip() if ctx and ctx.data_dir else ""
    if not image_path:
        raise ValueError("propose_plot_from_image requires 'image_path'.")
    if not allowed_dir:
        raise ValueError("allowed_dir is required at the tool boundary when loading an image.")

    res = propose_plot_from_image(
        image_path,
        allowed_dir=allowed_dir,
        n_candidates=int(args.get("n_candidates") or 8),
        x_range=tuple(args.get("x_range") or (0.0, 24.0)),
        y_range=tuple(args.get("y_range") or (1.0, 100.0)),
        y_scale=str(args.get("y_scale") or "linear"),
        figure_label=str(args["figure_label"]) if args.get("figure_label") else None,
    )
    return _build_digitizer_tool_result(state, res, tool_name="propose_plot_from_image")


def submit_plot_proposal_job(
    job_manager: JobManager,
    registry: ToolRegistry,
    *,
    session_id: str,
    state: PharmState,
    ctx: ToolContext,
    args: dict[str, Any],
    audit: AuditChain,
    timestamp: str,
    actor: str = "",
) -> str:
    """Route automatic image proposal (`propose_plot_from_image`) through `JobManager`
    bounded admission control (`_MAX_INFLIGHT` and `_MAX_PER_SESSION`) with
    `allow_expensive=True`.
    """
    def _job_fn() -> dict[str, Any]:
        _, tool_res = registry.execute(
            "propose_plot_from_image",
            state=state,
            ctx=ctx,
            args=args,
            audit=audit,
            timestamp=timestamp,
            actor=actor,
            allow_expensive=True,
        )
        return tool_res.result

    return job_manager.submit(
        session_id=session_id,
        kind="propose_plot_from_image",
        fn=_job_fn,
    )


TOOLS: list[Tool] = [
    Tool(
        name="digitize_plot",
        description=(
            "Calibrate a plot's X and Y axes (linear or log10) from two reference pixel anchors per axis "
            "and convert verified marker and error-bar pixels into series-keyed data coordinates."
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
                "plot_roi": {
                    "type": "object",
                    "description": "Explicit plot-region bounds: {u_min_px, u_max_px, v_min_px, v_max_px}.",
                },
                "points": {
                    "type": "array",
                    "description": "Points [{id, series, u_px, v_px, v_low_px, v_high_px, origin, verified}].",
                },
                "pixel_sigma": {"type": "number", "description": "Pixel picking metadata (default 1.5 px)."},
                "anchor_sigma_px": {"type": "number", "description": "Calibration anchor metadata in px."},
                "error_type": {"type": "string", "enum": ["se", "sd", "ci95"]},
                "sample_n": {"type": "integer"},
                "strict_bounds": {"type": "boolean"},
                "oob_overrides": {"type": "object"},
                "image_path": {"type": "string", "description": "Optional direct-child image filename in data_dir."},
                "figure_label": {"type": "string"},
                "approve_all": {"type": "boolean"},
            },
            "required": ["x_axis", "y_axis", "points"],
        },
        run=_run_digitize_plot,
        expensive=False,
    ),
    Tool(
        name="propose_plot_from_image",
        description=(
            "Expensive bounded background job: detect candidate axis anchors and unverified contour points "
            "from a direct-child image inside data_dir using uint8 strip projections."
        ),
        agent="data_manager",
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Direct-child image filename in data_dir."},
                "n_candidates": {"type": "integer"},
                "x_range": {"type": "array", "items": {"type": "number"}},
                "y_range": {"type": "array", "items": {"type": "number"}},
                "y_scale": {"type": "string", "enum": ["linear", "log10"]},
                "figure_label": {"type": "string"},
            },
            "required": ["image_path"],
        },
        run=_run_propose_plot_from_image,
        expensive=True,
        proposable=True,
    ),
]


def register_digitize_tools(registry: ToolRegistry) -> None:
    for t in TOOLS:
        registry.register(t)
