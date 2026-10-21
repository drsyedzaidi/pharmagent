"""Comprehensive unit and security verification tests for the Plot Digitizer computational core
(`app.compute.digitize`) and narrow tool wrapper (`app.tools.digitize_tools`).

Covers:
1. Pixel-to-data conversion on linear and logarithmic axes
2. Reversed axes (pixel-reversed, value-reversed, and doubly reversed)
3. Cropped/offset plotting regions
4. Known-point and directly digitized error-bar bound recovery against hand-derived reference fixture
5. Calibration metadata preservation and removal of unvalidated symmetric CI / sigma_AUC claims
6. Clicks outside the calibrated region (flagged in default mode; rejected in `strict_bounds=True`)
7. Degenerate and collinear calibration points (1D and 2D)
8. Descriptor-based image loading (`dir_fd` + `O_NOFOLLOW` + `fstat` + bounded `os.read`), symlink rejection,
   nested-path rejection, missing `allowed_dir` refusal, swap/TOCTOU immunity, growing-file rejection,
   and no global `Image.MAX_IMAGE_PIXELS` mutation
9. Deterministic CSV export with CR/LF control-char rejection and spreadsheet formula prefix neutralization
10. Privacy & audit hygiene: no raw image bytes, coordinates, or source filenames in LLM context or audit actions
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
    MAX_IMAGE_PIXELS,
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

FORBIDDEN_UNCERTAINTY_KEYS = {
    "sigma_x",
    "sigma_y_cal",
    "cal_se_y",
    "se_bar",
    "bar_se_y",
    "total_se_y",
    "ci95_y",
    "cv_pct",
    "cmax_sigma",
    "auclast_sigma",
    "auclast_ci95",
}


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
    rev_px_log = AxisCalibration(p1_px=400.0, p2_px=100.0, v1=1.0, v2=1000.0, scale="log10")
    assert math.isclose(rev_px_log.pixel_to_value(300.0), 10.0, rel_tol=1e-12)
    assert math.isclose(rev_px_log.pixel_to_value(200.0), 100.0, rel_tol=1e-12)
    assert math.isclose(rev_px_log.value_to_pixel(100.0), 200.0, rel_tol=1e-12)

    rev_val_lin = AxisCalibration(p1_px=100.0, p2_px=500.0, v1=100.0, v2=0.0, scale="linear")
    assert math.isclose(rev_val_lin.pixel_to_value(100.0), 100.0, abs_tol=1e-12)
    assert math.isclose(rev_val_lin.pixel_to_value(200.0), 75.0, abs_tol=1e-12)
    assert math.isclose(rev_val_lin.pixel_to_value(500.0), 0.0, abs_tol=1e-12)
    assert math.isclose(rev_val_lin.value_to_pixel(75.0), 200.0, abs_tol=1e-12)

    both_rev = AxisCalibration(p1_px=600.0, p2_px=200.0, v1=50.0, v2=10.0, scale="linear")
    assert math.isclose(both_rev.pixel_to_value(500.0), 40.0, abs_tol=1e-12)
    assert math.isclose(both_rev.value_to_pixel(40.0), 500.0, abs_tol=1e-12)
    assert both_rev.in_pixel_bounds(400.0) is True
    assert both_rev.in_pixel_bounds(150.0) is False


# ── 3. Cropped/offset plotting regions ───────────────────────────────────────

def test_cropped_and_offset_plotting_regions():
    out = digitize_plot(
        x_axis={"p1_px": 237.5, "p2_px": 812.5, "v1": 2.0, "v2": 14.0, "scale": "linear", "label": "Time", "unit": "h"},
        y_axis={"p1_px": 592.0, "p2_px": 142.0, "v1": 5.0, "v2": 500.0, "scale": "log10", "label": "Conc", "unit": "ng/mL"},
        points=[
            {"id": "c1", "u_px": 237.5, "v_px": 592.0},
            {"id": "c2", "u_px": 525.0, "v_px": 367.0},
            {"id": "c3", "u_px": 812.5, "v_px": 142.0},
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


# ── 4 & 5. Hand-derived reference fixture recovery & reduced uncertainty claim ─

def test_known_point_recovery_and_no_unvalidated_uncertainty_claims():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    # Scenario A: Linear X + Reversed Screen Linear Y + Directly Digitized Error Bars
    case_lin = fixture["linear_cropped_reversed_y"]
    res_lin = digitize_plot(**case_lin["input"])
    exp_lin = case_lin["expected"]

    assert [p["x"] for p in res_lin["points"]] == exp_lin["x"]
    assert [p["y"] for p in res_lin["points"]] == exp_lin["y"]
    assert [p["y_low"] for p in res_lin["points"]] == exp_lin["y_low"]
    assert [p["y_high"] for p in res_lin["points"]] == exp_lin["y_high"]

    pk = res_lin["derived_pk"]
    assert math.isclose(pk["cmax"], exp_lin["cmax"], abs_tol=1e-6)
    assert math.isclose(pk["tmax"], exp_lin["tmax"], abs_tol=1e-6)
    assert math.isclose(pk["auclast"], exp_lin["auclast"], abs_tol=1e-6)

    # Verify unvalidated symmetric CI / sigma_AUC keys are absent from points and derived_pk
    for pt in res_lin["points"]:
        assert FORBIDDEN_UNCERTAINTY_KEYS.isdisjoint(pt.keys())
    assert FORBIDDEN_UNCERTAINTY_KEYS.isdisjoint(pk.keys())

    # Verify calibration metadata is preserved
    assert res_lin["pixel_sigma"] == 1.5
    assert res_lin["anchor_sigma_px"] == 1.0
    assert res_lin["error_type"] == "se"

    # Scenario B: Semi-log10 Y + Cropped Offset X (multiplicative 2-fold bounds [0.5, 2.0] and [5.0, 20.0])
    case_log = fixture["semilog10_cropped_offset"]
    res_log = digitize_plot(**case_log["input"])
    exp_log = case_log["expected"]
    for pt, ex_x, ex_y in zip(res_log["points"], exp_log["x"], exp_log["y"]):
        assert math.isclose(pt["x"], ex_x, abs_tol=1e-6)
        assert math.isclose(pt["y"], ex_y, abs_tol=1e-5)
    assert math.isclose(res_log["points"][0]["y_low"], exp_log["y_low_first_two"][0], abs_tol=1e-5)
    assert math.isclose(res_log["points"][0]["y_high"], exp_log["y_high_first_two"][0], abs_tol=1e-5)
    assert math.isclose(res_log["points"][1]["y_low"], exp_log["y_low_first_two"][1], abs_tol=1e-5)
    assert math.isclose(res_log["points"][1]["y_high"], exp_log["y_high_first_two"][1], abs_tol=1e-5)


# ── 6. Clicks outside the calibrated region ──────────────────────────────────

def test_clicks_outside_calibrated_region_flagged_and_strict_rejected():
    x_ax = {"p1_px": 100.0, "p2_px": 500.0, "v1": 0.0, "v2": 10.0, "scale": "linear"}
    y_ax = {"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "linear"}
    pts = [
        {"id": "inside", "u_px": 250.0, "v_px": 250.0},
        {"id": "outside_x", "u_px": 540.0, "v_px": 250.0},
        {"id": "outside_y", "u_px": 250.0, "v_px": 80.0},
    ]

    res = digitize_plot(x_axis=x_ax, y_axis=y_ax, points=pts, strict_bounds=False)
    assert res["n_out_of_bounds"] == 2
    assert len(res["warnings"]) == 2
    assert res["points"][0]["out_of_bounds"] is False
    assert res["points"][1]["out_of_bounds_x"] is True and res["points"][1]["out_of_bounds_y"] is False
    assert res["points"][2]["out_of_bounds_x"] is False and res["points"][2]["out_of_bounds_y"] is True

    with pytest.raises(ValueError, match="outside the calibrated axis region"):
        digitize_plot(x_axis=x_ax, y_axis=y_ax, points=pts, strict_bounds=True)


# ── 7. Degenerate and collinear calibration points ───────────────────────────

def test_degenerate_and_collinear_calibration_points_rejected():
    valid_x = {"p1_px": 100.0, "p2_px": 500.0, "v1": 0.0, "v2": 10.0, "scale": "linear"}
    valid_y = {"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "linear"}
    pts = [{"id": "p1", "u_px": 200.0, "v_px": 200.0}]

    with pytest.raises(ValueError, match="degenerate"):
        digitize_plot(
            x_axis={"p1_px": 100.0, "p2_px": 100.4, "v1": 0.0, "v2": 10.0, "scale": "linear"},
            y_axis=valid_y,
            points=pts,
        )

    with pytest.raises(ValueError, match="distinct"):
        digitize_plot(
            x_axis={"p1_px": 100.0, "p2_px": 500.0, "v1": 5.0, "v2": 5.0, "scale": "linear"},
            y_axis=valid_y,
            points=pts,
        )

    with pytest.raises(ValueError, match="strictly positive"):
        digitize_plot(
            x_axis=valid_x,
            y_axis={"p1_px": 400.0, "p2_px": 100.0, "v1": 0.0, "v2": 100.0, "scale": "log10"},
            points=pts,
        )

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

    with pytest.raises(ValueError, match="Collinear X and Y axis calibration vectors"):
        digitize_plot(
            x_axis={**valid_x, "p1_uv": [100.0, 400.0], "p2_uv": [500.0, 400.0]},
            y_axis={**valid_y, "p1_uv": [100.0, 400.0], "p2_uv": [500.0, 402.0]},
            points=pts,
        )

    with pytest.raises(ValueError, match="collinear or degenerate"):
        digitize_plot(
            x_axis=valid_x,
            y_axis=valid_y,
            calibration_points_2d=[(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)],
            points=pts,
        )

    x_ortho = AxisCalibration(p1_px=100.0, p2_px=500.0, v1=0.0, v2=10.0, p1_uv=(100.0, 400.0), p2_uv=(500.0, 400.0))
    y_ortho = AxisCalibration(p1_px=400.0, p2_px=100.0, v1=0.0, v2=100.0, p1_uv=(100.0, 400.0), p2_uv=(100.0, 100.0))
    validate_non_collinear_2d(
        x_ortho,
        y_ortho,
        extra_anchors_uv=[(100.0, 400.0), (500.0, 400.0), (100.0, 100.0)],
    )


# ── 7. Descriptor-based image confinement & symlink rejection ────────────────

def test_descriptor_image_loading_confinement_and_symlinks(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    valid_img = allowed_dir / "plot.png"
    im = Image.new("RGB", (320, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.line([(40, 20), (40, 200), (290, 200)], fill=(0, 0, 0), width=3)
    draw.line([(45, 195), (100, 60), (180, 120), (280, 185)], fill=(10, 10, 10), width=3)
    im.save(valid_img)

    # 1. Missing or empty allowed_dir is refused
    with pytest.raises(ValueError, match="allowed_dir is required"):
        validate_and_load_image(valid_img, allowed_dir=None)
    with pytest.raises(ValueError, match="allowed_dir is required"):
        validate_and_load_image(valid_img, allowed_dir="")
    with pytest.raises(ValueError, match="allowed_dir"):
        validate_and_load_image(valid_img, allowed_dir=tmp_path / "nonexistent_dir")

    # 2. Nested child directory inside allowed_dir is refused (only direct child allowed)
    nested_dir = allowed_dir / "sub"
    nested_dir.mkdir()
    nested_img = nested_dir / "nested.png"
    im.save(nested_img)
    with pytest.raises(ValueError, match="direct child"):
        validate_and_load_image(nested_img, allowed_dir=allowed_dir)
    with pytest.raises(ValueError, match="direct child"):
        validate_and_load_image("sub/nested.png", allowed_dir=allowed_dir)
    with pytest.raises(ValueError, match="direct child"):
        validate_and_load_image("../outside/secret.png", allowed_dir=allowed_dir)

    # 3. Symlink inside allowed_dir (pointing either outside or to a valid sibling) is refused via O_NOFOLLOW
    outside_img = outside_dir / "secret.png"
    im.save(outside_img)
    symlink_outside = allowed_dir / "link_outside.png"
    symlink_outside.symlink_to(outside_img)
    with pytest.raises(ValueError, match="Symlink|not permitted"):
        validate_and_load_image(symlink_outside, allowed_dir=allowed_dir)

    symlink_inside = allowed_dir / "link_inside.png"
    symlink_inside.symlink_to(valid_img)
    with pytest.raises(ValueError, match="Symlink|not permitted"):
        validate_and_load_image(symlink_inside, allowed_dir=allowed_dir)

    # 4. Symlink as allowed_dir itself is refused via O_NOFOLLOW
    symlink_dir = tmp_path / "symlink_allowed"
    symlink_dir.symlink_to(allowed_dir)
    with pytest.raises(ValueError, match="Symlink|allowed_dir"):
        validate_and_load_image("plot.png", allowed_dir=symlink_dir)


# ── 8. TOCTOU swap, growing file & fixed Pillow decompression limits ─────────

def test_descriptor_image_loading_toctou_growing_file_and_limits(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    valid_img = allowed_dir / "plot.png"
    im = Image.new("RGB", (320, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.line([(40, 20), (40, 200), (290, 200)], fill=(0, 0, 0), width=3)
    draw.line([(45, 195), (100, 60), (180, 120), (280, 185)], fill=(10, 10, 10), width=3)
    im.save(valid_img)
    outside_img = outside_dir / "secret.png"
    im.save(outside_img)

    # 1. Deterministic growing-file test: file grows past max_bytes after fstat(fd) before os.read(fd)
    grow_img = allowed_dir / "grow.png"
    im.save(grow_img)
    initial_size = grow_img.stat().st_size

    def _append_bytes_after_fstat() -> None:
        with open(grow_img, "ab") as f:
            f.write(b"X" * 2048)

    with pytest.raises(ValueError, match="grew beyond maximum limit|exceeds maximum limit"):
        validate_and_load_image(
            grow_img,
            allowed_dir=allowed_dir,
            max_bytes=initial_size + 512,
            _after_fstat_hook=_append_bytes_after_fstat,
        )

    # 2. Deterministic swap/TOCTOU immunity: replacing the directory entry with a symlink after fd is open
    # still reads the originally opened regular file inode
    swap_img = allowed_dir / "swap.png"
    im.save(swap_img)
    expected_sha = validate_and_load_image(swap_img, allowed_dir=allowed_dir)[1]

    def _swap_entry_with_symlink_after_fstat() -> None:
        swap_img.unlink()
        swap_img.symlink_to(outside_img)

    _, sha_after_swap, dims_after_swap, _ = validate_and_load_image(
        swap_img,
        allowed_dir=allowed_dir,
        _after_fstat_hook=_swap_entry_with_symlink_after_fstat,
    )
    assert sha_after_swap == expected_sha
    assert dims_after_swap["width"] == 320
    # And the next call now sees the symlink and refuses it!
    with pytest.raises(ValueError, match="Symlink|not permitted"):
        validate_and_load_image(swap_img, allowed_dir=allowed_dir)

    # 3. Missing, directory, empty, corrupt, tiny, unsupported format, and pixel ceiling checks
    with pytest.raises(ValueError, match="not found"):
        validate_and_load_image("missing.png", allowed_dir=allowed_dir)

    empty_file = allowed_dir / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_and_load_image(empty_file, allowed_dir=allowed_dir)

    corrupt_file = allowed_dir / "corrupt.png"
    corrupt_file.write_bytes(b"not-an-image-payload-1234567890")
    with pytest.raises(ValueError, match="Invalid or corrupted image"):
        validate_and_load_image(corrupt_file, allowed_dir=allowed_dir)

    tiny_file = allowed_dir / "tiny.png"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(tiny_file)
    with pytest.raises(ValueError, match="smaller than minimum"):
        validate_and_load_image(tiny_file, allowed_dir=allowed_dir)

    gif_file = allowed_dir / "anim.gif"
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(gif_file, format="GIF")
    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_and_load_image(gif_file, allowed_dir=allowed_dir)

    # 4. Verify Image.MAX_IMAGE_PIXELS is fixed and never mutated across calls
    assert Image.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS
    with pytest.raises(ValueError, match="pixel"):
        validate_and_load_image(valid_img, allowed_dir=allowed_dir, max_pixels=10_000)
    assert Image.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS

    # 5. Valid local image proposal marks all points contour_proposed & verified=False
    # and does NOT expose the filename in figure_label
    prop = propose_plot_from_image(valid_img, n_candidates=6, allowed_dir=allowed_dir)
    assert prop["status"] == "ok"
    assert len(prop["source_sha256"]) == 64
    assert prop["figure_label"] == f"source_{prop['source_sha256'][:12]}"
    assert "plot.png" not in prop["figure_label"]
    assert prop["n_points"] == 6
    assert prop["n_verified"] == 0
    assert prop["all_verified"] is False
    assert all(p["origin"] == "contour_proposed" and p["verified"] is False for p in prop["points"])


# ── 9. Deterministic export, CR/LF control-char rejection & formula guard ────

def test_deterministic_export_crlf_rejection_and_formula_neutralization():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    inp = fixture["linear_cropped_reversed_y"]["input"]

    res1 = digitize_plot(**inp, figure_label="PK_Ref_Curve", source_sha256="a" * 64)
    res2 = digitize_plot(**inp, figure_label="PK_Ref_Curve", source_sha256="a" * 64)

    assert res1["export_sha256"] == res2["export_sha256"]
    assert res1["csv_export"] == res2["csv_export"]
    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)

    standalone = export_digitized_csv(res1)
    assert standalone == {"csv": res1["csv_export"], "export_sha256": res1["export_sha256"]}

    # Reject CR/LF and control characters in metadata fields
    for bad_str in ("Curve\n# injected=true", "Curve\r\n2", "Bad\x00Label", "Bad\x1bLabel"):
        with pytest.raises(ValueError, match="control characters|CR/LF"):
            digitize_plot(**inp, figure_label=bad_str)
        with pytest.raises(ValueError, match="control characters|CR/LF"):
            digitize_plot(
                x_axis={**inp["x_axis"], "label": bad_str},
                y_axis=inp["y_axis"],
                points=inp["points"],
            )
        with pytest.raises(ValueError, match="control characters|CR/LF"):
            digitize_plot(
                x_axis=inp["x_axis"],
                y_axis=inp["y_axis"],
                points=[{**inp["points"][0], "id": bad_str}],
            )

    # Neutralize spreadsheet formula prefixes (=, +, -, @) in CSV cells while keeping structured JSON intact
    formula_pts = [
        {"id": "=cmd|' /C calc'!A0", "series": "+SUM(A1:A2)", "u_px": 150.0, "v_px": 420.0},
        {"id": "-2+3", "series": "@IMPORTXML()", "u_px": 630.0, "v_px": 120.0},
    ]
    res_f = digitize_plot(x_axis=inp["x_axis"], y_axis=inp["y_axis"], points=formula_pts)
    # Structured provenance record keeps exact strings
    assert res_f["points"][0]["id"] == "=cmd|' /C calc'!A0"
    assert res_f["points"][0]["series"] == "+SUM(A1:A2)"
    # CSV cells are prefixed with "'" so spreadsheet parsers treat them as literal text
    assert "'=cmd|' /C calc'!A0,'+SUM(A1:A2)" in res_f["csv_export"]
    assert "'-2+3,'@IMPORTXML()" in res_f["csv_export"]


# ── 10. Privacy & audit hygiene: no image, coordinates, or filename leaks ────

class _SpyLLM:
    """Spy LLM that records every state_summary, observations, and argument passed to select_tool."""

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


def test_no_raw_image_coordinates_or_filenames_sent_to_llm_or_audit(tmp_path: Path):
    # (a) Static AST import audit
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

    # (b) Tool boundary refuses missing/empty ctx.data_dir when image_path is supplied
    reg = ToolRegistry()
    register_digitize_tools(reg)
    audit = AuditChain()
    state = PharmState(session_id="s_privacy")

    with pytest.raises(ValueError, match="allowed_dir is required"):
        reg.execute(
            "digitize_plot",
            state=state,
            ctx=ToolContext(data_dir=""),
            args={"image_path": "any.png"},
            audit=audit,
            timestamp="2026-10-21T00:00:00Z",
        )

    # (c) Runtime tool execution via Agent.run_turn with sensitive filename & figure_label
    sensitive_filename = "secret_patient_study_42_curve.png"
    img_path = tmp_path / sensitive_filename
    im = Image.new("RGB", (300, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.line([(30, 20), (30, 170), (270, 170)], fill=(0, 0, 0), width=3)
    draw.line([(35, 165), (90, 50), (260, 155)], fill=(0, 0, 0), width=3)
    im.save(img_path)

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    inp = fixture["linear_cropped_reversed_y"]["input"]

    spy = _SpyLLM(
        first_call={
            "name": "digitize_plot",
            "input": {
                **inp,
                "image_path": sensitive_filename,
                "figure_label": "Confidential_Manuscript_Fig1",
            },
        }
    )
    agent = Agent(name="data_manager", system_prompt="Data manager test agent")
    turn_res = agent.run_turn(
        state=state,
        message="Digitize plot",
        llm=spy,
        registry=reg,
        ctx=ToolContext(data_dir=str(tmp_path)),
        audit=audit,
        clock=lambda: "2026-10-21T00:00:01Z",
    )

    assert len(turn_res.state.widgets) == 1
    w_data = turn_res.state.widgets[0]["data"]
    assert "image_data_url" not in w_data
    assert w_data["derived_pk"]["auclast"] == 603.0

    # Verify neither the second LLM call's state_summary/turn_observations nor the audit action
    # contains the filename, figure_label, coordinates, or AUClast value
    assert len(spy.calls) == 2
    second_summary_json = json.dumps(spy.calls[1]["state_summary"])
    assert sensitive_filename not in second_summary_json
    assert "Confidential_Manuscript_Fig1" not in second_summary_json
    assert "image_data_url" not in second_summary_json
    assert "csv_export" not in second_summary_json
    assert "603.0" not in second_summary_json
    assert "u_px" not in second_summary_json

    audit_list_json = json.dumps(audit.to_list())
    assert sensitive_filename not in audit_list_json
    assert "Confidential_Manuscript_Fig1" not in audit_list_json
    assert audit.verify() is True
