"""Comprehensive unit and verification tests for the Plot Digitizer computational core
(`app.compute.digitize`) and narrow tool wrapper (`app.tools.digitize_tools`).

Covers all 10 required verification categories:
1. Pixel-to-data conversion on linear and logarithmic axes
2. Reversed axes (pixel-reversed, value-reversed, and doubly reversed)
3. Cropped/offset plotting regions
4. Known-point recovery against independent closed-form ground truth (`digitize_reference_fixture.json`)
5. Calibration uncertainty propagated into reported output
6. Clicks outside the calibrated region (flagged in default mode; rejected in `strict_bounds=True`)
7. Degenerate and collinear calibration points (1D and 2D)
8. Invalid images, resource limits (bytes, pixels, dimensions, formats), and directory confinement
9. Deterministic CSV export with calibration metadata and SHA-256 digest
10. Privacy & isolation invariant: no raw image bytes or extracted point arrays ever sent to an LLM
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from app.agents.base import Agent
from app.compute.digitize import (
    AxisCalibration,
    digitize_plot,
    export_digitized_csv,
    propose_plot_from_image,
    validate_and_load_image,
    validate_non_collinear_2d,
)
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolRegistry
from app.tools.digitize_tools import register_digitize_tools

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "digitize_reference_fixture.json"


# ── 1. Pixel-to-data conversion on linear and logarithmic axes ───────────────

def test_pixel_to_data_linear_and_log10_axes():
    lin = AxisCalibration(p1_px=100.0, p2_px=500.0, v1=0.0, v2=20.0, scale="linear")
    lin.validate("x_axis")
    for px, expected_v in [(100.0, 0.0), (200.0, 5.0), (300.0, 10.0), (450.0, 17.5), (500.0, 20.0)]:
        val = lin.pixel_to_value(px)
        assert math.isclose(val, expected_v, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(lin.value_to_pixel(val), px, rel_tol=1e-12, abs_tol=1e-12)

    log_ax = AxisCalibration(p1_px=100.0, p2_px=400.0, v1=0.1, v2=100.0, scale="log10")
    log_ax.validate("y_axis")
    for px, expected_v in [(100.0, 0.1), (200.0, 1.0), (300.0, 10.0), (400.0, 100.0)]:
        val = log_ax.pixel_to_value(px)
        assert math.isclose(val, expected_v, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(log_ax.value_to_pixel(val), px, rel_tol=1e-12, abs_tol=1e-12)


# ── 2. Reversed axes ─────────────────────────────────────────────────────────

def test_reversed_axes_pixel_and_value_directions():
    # (a) Standard screen Y-axis: pixel decreases upward (p1_px=400 -> v1=1, p2_px=100 -> v2=1000)
    rev_px_log = AxisCalibration(p1_px=400.0, p2_px=100.0, v1=1.0, v2=1000.0, scale="log10")
    assert math.isclose(rev_px_log.pixel_to_value(300.0), 10.0, rel_tol=1e-12)
    assert math.isclose(rev_px_log.pixel_to_value(200.0), 100.0, rel_tol=1e-12)
    assert math.isclose(rev_px_log.value_to_pixel(100.0), 200.0, rel_tol=1e-12)

    # (b) Reversed value axis: v1=100% inhibition at left/bottom (p1=100), v2=0% at right/top (p2=500)
    rev_val_lin = AxisCalibration(p1_px=100.0, p2_px=500.0, v1=100.0, v2=0.0, scale="linear")
    assert math.isclose(rev_val_lin.pixel_to_value(100.0), 100.0, abs_tol=1e-12)
    assert math.isclose(rev_val_lin.pixel_to_value(200.0), 75.0, abs_tol=1e-12)
    assert math.isclose(rev_val_lin.pixel_to_value(500.0), 0.0, abs_tol=1e-12)
    assert math.isclose(rev_val_lin.value_to_pixel(75.0), 200.0, abs_tol=1e-12)

    # (c) Doubly reversed axis: p1_px=600 -> v1=50, p2_px=200 -> v2=10
    both_rev = AxisCalibration(p1_px=600.0, p2_px=200.0, v1=50.0, v2=10.0, scale="linear")
    assert math.isclose(both_rev.pixel_to_value(500.0), 40.0, abs_tol=1e-12)
    assert math.isclose(both_rev.value_to_pixel(40.0), 500.0, abs_tol=1e-12)
    assert both_rev.in_pixel_bounds(400.0) is True
    assert both_rev.in_pixel_bounds(150.0) is False


# ── 3. Cropped/offset plotting regions ───────────────────────────────────────

def test_cropped_and_offset_plotting_regions():
    # Sub-region cropped out of a multi-panel figure:
    # X anchored at [237.5 px -> 2.0 h, 812.5 px -> 14.0 h] (575 px for 12 h)
    # Y anchored at [592.0 px -> 5.0 ng/mL, 142.0 px -> 500.0 ng/mL] on log10 (450 px for 2 decades)
    out = digitize_plot(
        x_axis={"p1_px": 237.5, "p2_px": 812.5, "v1": 2.0, "v2": 14.0, "scale": "linear", "label": "Time", "unit": "h"},
        y_axis={"p1_px": 592.0, "p2_px": 142.0, "v1": 5.0, "v2": 500.0, "scale": "log10", "label": "Conc", "unit": "ng/mL"},
        points=[
            {"id": "c1", "u_px": 237.5, "v_px": 592.0},          # (2.0 h, 5.0 ng/mL)
            {"id": "c2", "u_px": 525.0, "v_px": 367.0},          # midpoint -> (8.0 h, 50.0 ng/mL)
            {"id": "c3", "u_px": 812.5, "v_px": 142.0},          # (14.0 h, 500.0 ng/mL)
        ],
    )
    pts = out["points"]
    assert math.isclose(pts[0]["x"], 2.0, abs_tol=1e-6)
    assert math.isclose(pts[0]["y"], 5.0, abs_tol=1e-6)
    assert math.isclose(pts[1]["x"], 8.0, abs_tol=1e-6)
    assert math.isclose(pts[1]["y"], 50.0, abs_tol=1e-6)
    assert math.isclose(pts[2]["x"], 14.0, abs_tol=1e-6)
    assert math.isclose(pts[2]["y"], 500.0, abs_tol=1e-6)
    assert out["n_out_of_bounds"] == 0


# ── 4. Known-point recovery against independent ground truth fixture ─────────

def test_known_point_recovery_against_independent_reference_fixture():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    # Scenario A: Linear X + Reversed Screen Linear Y + Error Bars + Exact Trapezoidal AUClast
    case_lin = fixture["linear_cropped_reversed_y"]
    res_lin = digitize_plot(**case_lin["input"])
    exp_lin = case_lin["expected"]

    assert [p["x"] for p in res_lin["points"]] == exp_lin["x"]
    assert [p["y"] for p in res_lin["points"]] == exp_lin["y"]
    assert [p["se_bar"] for p in res_lin["points"]] == exp_lin["se_bar"]

    # Check closed-form Taylor uncertainty values at f=0 and f=0.5 (t=12h)
    assert math.isclose(res_lin["points"][0]["sigma_x"], exp_lin["sigma_x_first"], abs_tol=1e-6)
    assert math.isclose(res_lin["points"][5]["sigma_x"], exp_lin["sigma_x_mid_12h"], abs_tol=1e-6)
    assert math.isclose(res_lin["points"][0]["sigma_y_cal"], exp_lin["sigma_y_cal_first"], abs_tol=1e-6)

    pk = res_lin["derived_pk"]
    assert math.isclose(pk["cmax"], exp_lin["cmax"], abs_tol=1e-6)
    assert math.isclose(pk["tmax"], exp_lin["tmax"], abs_tol=1e-6)
    assert math.isclose(pk["auclast"], exp_lin["auclast"], abs_tol=1e-6)

    # Independent quadrature check for AUClast sigma using fixture weights
    w = exp_lin["trapezoidal_weights"]
    expected_auc_sigma = math.sqrt(
        sum((wi * pt["total_se_y"]) ** 2 for wi, pt in zip(w, res_lin["points"]))
    )
    assert math.isclose(pk["auclast_sigma"], expected_auc_sigma, abs_tol=1e-5)

    # Scenario B: Semi-log10 Y + Cropped Offset X
    case_log = fixture["semilog10_cropped_offset"]
    res_log = digitize_plot(**case_log["input"])
    exp_log = case_log["expected"]
    for pt, ex_x, ex_y, ex_sig in zip(
        res_log["points"], exp_log["x"], exp_log["y"], exp_log["sigma_y_cal"]
    ):
        assert math.isclose(pt["x"], ex_x, abs_tol=1e-6)
        assert math.isclose(pt["y"], ex_y, abs_tol=1e-5)
        assert math.isclose(pt["sigma_y_cal"], ex_sig, abs_tol=1e-5)


# ── 5. Calibration uncertainty propagation & error-bar modes ─────────────────

def test_calibration_uncertainty_propagation_and_error_bar_modes():
    x_ax = {"p1_px": 100.0, "p2_px": 500.0, "v1": 0.0, "v2": 10.0, "scale": "linear"}
    y_ax = {"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "linear"}

    # Compare midpoint (f=0.5 -> v_px=250) vs anchor endpoint (f=0.0 -> v_px=400) vs extrapolated (f=1.5 -> v_px=-50)
    out = digitize_plot(
        x_axis=x_ax,
        y_axis=y_ax,
        pixel_sigma=1.5,
        anchor_sigma_px=1.2,
        points=[
            {"id": "mid", "u_px": 300.0, "v_px": 250.0},
            {"id": "end", "u_px": 100.0, "v_px": 400.0},
            {"id": "ext", "u_px": 500.0, "v_px": 50.0},
        ],
    )
    sig_mid = out["points"][0]["sigma_y_cal"]
    sig_end = out["points"][1]["sigma_y_cal"]
    sig_ext = out["points"][2]["sigma_y_cal"]
    # Leverage (1-f)^2 + f^2 is 0.5 at mid, 1.0 at endpoint, and >1.0 outside calibrated span
    assert sig_mid < sig_end < sig_ext

    # Verify error_type='sd' (divides by sqrt(n)) and error_type='ci95' (divides by 1.959964)
    # Here v_px=250 -> y=50; v_low_px=280 -> y=40; v_high_px=220 -> y=60 (half-width = 10)
    pts_bar = [
        {"id": "p1", "u_px": 100.0, "v_px": 400.0},
        {"id": "p2", "u_px": 300.0, "v_px": 250.0, "v_low_px": 280.0, "v_high_px": 220.0},
    ]
    out_sd = digitize_plot(x_axis=x_ax, y_axis=y_ax, points=pts_bar, error_type="sd", sample_n=25)
    assert math.isclose(out_sd["points"][1]["se_bar"], 10.0 / 5.0, abs_tol=1e-6)
    assert math.isclose(
        out_sd["points"][1]["total_se_y"],
        math.hypot(out_sd["points"][1]["sigma_y_cal"], 2.0),
        abs_tol=1e-6,
    )

    out_ci = digitize_plot(x_axis=x_ax, y_axis=y_ax, points=pts_bar, error_type="ci95")
    assert math.isclose(out_ci["points"][1]["se_bar"], 10.0 / 1.959963984540054, abs_tol=1e-6)


# ── 6. Clicks outside the calibrated region ──────────────────────────────────

def test_clicks_outside_calibrated_region_flagged_and_strict_rejected():
    x_ax = {"p1_px": 100.0, "p2_px": 500.0, "v1": 0.0, "v2": 10.0, "scale": "linear"}
    y_ax = {"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "linear"}
    pts = [
        {"id": "inside", "u_px": 250.0, "v_px": 250.0},
        {"id": "outside_x", "u_px": 540.0, "v_px": 250.0},
        {"id": "outside_y", "u_px": 250.0, "v_px": 80.0},
    ]

    # Default mode (strict_bounds=False): flags out-of-bounds points and emits warnings
    res = digitize_plot(x_axis=x_ax, y_axis=y_ax, points=pts, strict_bounds=False)
    assert res["n_out_of_bounds"] == 2
    assert len(res["warnings"]) == 2
    assert res["points"][0]["out_of_bounds"] is False
    assert res["points"][1]["out_of_bounds_x"] is True and res["points"][1]["out_of_bounds_y"] is False
    assert res["points"][2]["out_of_bounds_x"] is False and res["points"][2]["out_of_bounds_y"] is True

    # Strict mode (strict_bounds=True): refuses out-of-bounds clicks with ValueError
    with pytest.raises(ValueError, match="outside the calibrated axis region"):
        digitize_plot(x_axis=x_ax, y_axis=y_ax, points=pts, strict_bounds=True)


# ── 7. Degenerate and collinear calibration points ───────────────────────────

def test_degenerate_and_collinear_calibration_points_rejected():
    valid_x = {"p1_px": 100.0, "p2_px": 500.0, "v1": 0.0, "v2": 10.0, "scale": "linear"}
    valid_y = {"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "linear"}
    pts = [{"id": "p1", "u_px": 200.0, "v_px": 200.0}]

    # Coincident or <1 px pixel anchors
    with pytest.raises(ValueError, match="degenerate"):
        digitize_plot(
            x_axis={"p1_px": 100.0, "p2_px": 100.4, "v1": 0.0, "v2": 10.0, "scale": "linear"},
            y_axis=valid_y,
            points=pts,
        )

    # Identical reference values
    with pytest.raises(ValueError, match="distinct"):
        digitize_plot(
            x_axis={"p1_px": 100.0, "p2_px": 500.0, "v1": 5.0, "v2": 5.0, "scale": "linear"},
            y_axis=valid_y,
            points=pts,
        )

    # Non-positive reference value on log10 axis
    with pytest.raises(ValueError, match="strictly positive"):
        digitize_plot(
            x_axis=valid_x,
            y_axis={"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "log10"},
            points=pts,
        )

    # NaN / Inf / bool anchor coordinates
    with pytest.raises(ValueError, match="finite"):
        digitize_plot(
            x_axis={"p1_px": float("nan"), "p2_px": 500.0, "v1": 0.0, "v2": 10.0},
            y_axis=valid_y,
            points=pts,
        )
    with pytest.raises(ValueError, match="finite"):
        digitize_plot(
            x_axis={"p1_px": True, "p2_px": 500.0, "v1": 0.0, "v2": 10.0},
            y_axis=valid_y,
            points=pts,
        )

    # 2D collinear X and Y axis vectors (both nearly horizontal)
    with pytest.raises(ValueError, match="Collinear X and Y axis calibration vectors"):
        digitize_plot(
            x_axis={**valid_x, "p1_uv": [100.0, 400.0], "p2_uv": [500.0, 400.0]},
            y_axis={**valid_y, "p1_uv": [100.0, 400.0], "p2_uv": [500.0, 402.0]},
            points=pts,
        )

    # 3 collinear 2D calibration anchor points on a diagonal line
    with pytest.raises(ValueError, match="collinear or degenerate"):
        digitize_plot(
            x_axis=valid_x,
            y_axis=valid_y,
            calibration_points_2d=[(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)],
            points=pts,
        )

    # Orthogonal 2D calibration anchors succeed
    x_ortho = AxisCalibration(p1_px=100.0, p2_px=500.0, v1=0.0, v2=10.0, p1_uv=(100.0, 400.0), p2_uv=(500.0, 400.0))
    y_ortho = AxisCalibration(p1_px=400.0, p2_px=100.0, v1=0.0, v2=100.0, p1_uv=(100.0, 400.0), p2_uv=(100.0, 100.0))
    validate_non_collinear_2d(
        x_ortho,
        y_ortho,
        extra_anchors_uv=[(100.0, 400.0), (500.0, 400.0), (100.0, 100.0)],
    )


# ── 8. Invalid images, resource limits, and directory confinement ────────────

def test_invalid_images_resource_limits_and_directory_confinement(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    # 1. Missing file
    with pytest.raises(ValueError, match="not found"):
        validate_and_load_image(allowed_dir / "missing.png")

    # 2. Directory passed instead of file
    with pytest.raises(ValueError, match="not a regular file"):
        validate_and_load_image(allowed_dir)

    # 3. Empty (0-byte) file
    empty_file = allowed_dir / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_and_load_image(empty_file)

    # 4. Corrupt / non-image bytes
    corrupt_file = allowed_dir / "corrupt.png"
    corrupt_file.write_bytes(b"not-an-image-payload-1234567890")
    with pytest.raises(ValueError, match="Invalid or corrupted image"):
        validate_and_load_image(corrupt_file)

    # 5. Image smaller than MIN_IMAGE_DIMENSION (32x32)
    tiny_file = allowed_dir / "tiny.png"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(tiny_file)
    with pytest.raises(ValueError, match="smaller than minimum"):
        validate_and_load_image(tiny_file)

    # 6. Unsupported image format (GIF)
    gif_file = allowed_dir / "anim.gif"
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(gif_file, format="GIF")
    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_and_load_image(gif_file)

    # 7. File byte limit exceeded
    valid_img = allowed_dir / "plot.png"
    im = Image.new("RGB", (320, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.line([(40, 20), (40, 200), (290, 200)], fill=(0, 0, 0), width=3)
    draw.line([(45, 195), (100, 60), (180, 120), (280, 185)], fill=(10, 10, 10), width=3)
    im.save(valid_img)
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        validate_and_load_image(valid_img, max_bytes=100)

    # 8. Pixel count limit exceeded (decompression-bomb guard)
    with pytest.raises(ValueError, match="pixel"):
        validate_and_load_image(valid_img, max_pixels=10_000)

    # 9. Directory confinement: reject image outside allowed_dir
    outside_img = outside_dir / "secret.png"
    im.save(outside_img)
    with pytest.raises(ValueError, match="outside the allowed data directory"):
        validate_and_load_image(outside_img, allowed_dir=allowed_dir)

    # 10. Valid local image proposal marks all points contour_proposed & verified=False
    prop = propose_plot_from_image(valid_img, n_candidates=6, allowed_dir=allowed_dir)
    assert prop["status"] == "ok"
    assert len(prop["source_sha256"]) == 64
    assert prop["image_dims"]["width"] == 320
    assert prop["image_dims"]["height"] == 240
    assert prop["n_points"] == 6
    assert prop["n_verified"] == 0
    assert prop["all_verified"] is False
    assert all(p["origin"] == "contour_proposed" and p["verified"] is False for p in prop["points"])


# ── 9. Deterministic export with calibration metadata ────────────────────────

def test_deterministic_export_with_calibration_metadata():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    inp = fixture["linear_cropped_reversed_y"]["input"]

    res1 = digitize_plot(**inp, figure_label="PK_Ref_Curve", source_sha256="a" * 64)
    res2 = digitize_plot(**inp, figure_label="PK_Ref_Curve", source_sha256="a" * 64)

    assert res1["export_sha256"] == res2["export_sha256"]
    assert res1["csv_export"] == res2["csv_export"]
    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)

    csv_lines = res1["csv_export"].splitlines()
    assert csv_lines[0] == "# format=pharmagent-digitizer-v1"
    assert "# figure_label=PK_Ref_Curve" in csv_lines
    assert f"# source_sha256={'a' * 64}" in csv_lines
    assert any(line.startswith("# x_axis={") and '"scale":"linear"' in line for line in csv_lines)
    assert any(line.startswith("# y_axis={") and '"p1_px":420.0' in line for line in csv_lines)
    assert "# pixel_sigma=1.5" in csv_lines
    assert "# anchor_sigma_px=1.0" in csv_lines
    assert "# error_type=se" in csv_lines

    data_rows = [line for line in csv_lines if not line.startswith("#")]
    assert data_rows[0].startswith("id,series,index,u_px,v_px,")
    assert len(data_rows) == 1 + len(inp["points"])

    # Standalone export_digitized_csv matches embedded export
    standalone = export_digitized_csv(res1)
    assert standalone == {"csv": res1["csv_export"], "export_sha256": res1["export_sha256"]}


# ── 10. Privacy & isolation: no raw image or extracted data sent to LLM ──────

class _SpyLLM:
    """Spy LLM that records every state_summary and argument passed to select_tool."""

    def __init__(self, first_call: dict[str, Any]):
        self._first_call = first_call
        self.calls: list[dict[str, Any]] = []

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        return "data_manager"

    def select_tool(
        self,
        agent: str,
        message: str,
        tools: list[Tool],
        state_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        self.calls.append({"agent": agent, "message": message, "state_summary": state_summary})
        if len(self.calls) == 1:
            return self._first_call
        return None


def test_no_raw_image_or_extracted_data_sent_to_llm(tmp_path: Path):
    # (a) Static AST import audit: neither digitize.py nor digitize_tools.py imports LLM, evidence, or HTTP libraries
    forbidden_prefixes = ("app.evidence", "app.core.llm", "anthropic", "openai", "requests", "httpx", "urllib")
    for mod_path in (
        Path(__file__).resolve().parents[1] / "app" / "compute" / "digitize.py",
        Path(__file__).resolve().parents[1] / "app" / "tools" / "digitize_tools.py",
    ):
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.append(node.module)
        for imp in imported_names:
            assert not imp.startswith(forbidden_prefixes), (
                f"{mod_path.name} imports forbidden module {imp!r}"
            )

    # (b) Runtime tool execution via Agent.run_turn with SpyLLM:
    # Verify no base64 image data is produced and no point coordinates or AUClast values leak to the LLM
    img_path = tmp_path / "curve.png"
    im = Image.new("RGB", (300, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.line([(30, 20), (30, 170), (270, 170)], fill=(0, 0, 0), width=3)
    draw.line([(35, 165), (90, 50), (260, 155)], fill=(0, 0, 0), width=3)
    im.save(img_path)

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    inp = fixture["linear_cropped_reversed_y"]["input"]

    reg = ToolRegistry()
    register_digitize_tools(reg)
    audit = AuditChain()
    ctx = ToolContext(data_dir=str(tmp_path))
    state = PharmState(session_id="s_privacy")

    spy = _SpyLLM(
        first_call={
            "name": "digitize_plot",
            "input": {
                **inp,
                "image_path": str(img_path),
                "figure_label": "Confidential_Subject_Curve",
            },
        }
    )
    agent = Agent(name="data_manager", system_prompt="Data manager test agent")
    turn_res = agent.run_turn(
        state=state,
        message="Digitize plot",
        llm=spy,
        registry=reg,
        ctx=ctx,
        audit=audit,
        clock=lambda: "2026-10-21T00:00:00Z",
    )

    # Tool executed and stored result in state.widgets
    assert len(turn_res.state.widgets) == 1
    w_data = turn_res.state.widgets[0]["data"]
    assert "image_data_url" not in w_data
    assert w_data["derived_pk"]["auclast"] == 603.0

    # Inspect second LLM call's state_summary (after tool execution)
    assert len(spy.calls) == 2
    second_summary_json = json.dumps(spy.calls[1]["state_summary"])
    # Neither the raw y values (45.0, 80.0, 60.0), nor AUClast (603.0), nor CSV export, nor image bytes appear in state_summary
    assert "image_data_url" not in second_summary_json
    assert "csv_export" not in second_summary_json
    assert "603.0" not in second_summary_json
    assert "u_px" not in second_summary_json
    assert audit.verify() is True
