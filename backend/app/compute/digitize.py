"""Deterministic Plot Digitizer with 2-point axis calibration, editable points,
directly digitized error-bar bounds, point-estimate PK metrics, and local contour proposals.

Design invariants ("Models propose, tools compute, humans verify"):
- Pixel-to-data mapping (`linear` or `log10`), reversed/cropped axis handling,
  direct error-bar bound conversion, and linear-trapezoidal AUClast point-estimate
  integration are 100% deterministic NumPy/math calculations.
- Uncertainty propagation (joint Monte Carlo over shared calibration anchors,
  click jitter, time/concentration errors, and log-space error bars) is deferred
  to a separately validated slice; this module preserves calibration metadata and
  directly digitized bounds without emitting unvalidated symmetric CI or sigma_AUC claims.
- Local image loading (`validate_and_load_image`) requires a mandatory `allowed_dir`,
  accepts only direct regular-file children opened via `dir_fd` with `O_NOFOLLOW`,
  validates via `fstat`, reads at most `max_bytes + 1` from the descriptor, and
  uses a fixed application-wide `Image.MAX_IMAGE_PIXELS` ceiling.
- No raw image bytes (`image_data_url`), source filenames, or extracted coordinate
  tables are ever sent to an LLM or audit action strings.
"""
from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import math
import os
import stat
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

VALID_SCALES = ("linear", "log10")
VALID_ERROR_TYPES = ("se", "sd", "ci95")
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "TIFF", "BMP")

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB hard cap
MAX_IMAGE_PIXELS = 25_000_000       # 25 megapixels hard cap
MIN_IMAGE_DIMENSION = 32            # minimum width/height in pixels
MIN_AXIS_SPAN_PX = 1.0              # minimum 1 px separation between calibration anchors
MIN_AXIS_ANGLE_SIN = math.sin(math.radians(5.0))  # minimum 5 deg angle between 2D axes

# Fixed application-wide Pillow pixel ceiling (never mutated per request).
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _is_finite_number(val: Any) -> bool:
    """Return True if val is a finite int/float (excluding bool)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def _validate_metadata_text(value: str, field_name: str, *, max_len: int = 256) -> str:
    """Reject CR/LF and ASCII/C1 control characters in metadata strings."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if len(value) > max_len:
        raise ValueError(f"{field_name} exceeds maximum length ({max_len} chars).")
    for ch in value:
        code = ord(ch)
        if code < 32 or 127 <= code <= 159:
            raise ValueError(
                f"{field_name} must not contain CR/LF or control characters (found U+{code:04X})."
            )
    return value.strip()


def _neutralize_csv_cell(value: Any) -> Any:
    """Prefix textual CSV cells starting with spreadsheet formula triggers with `'`."""
    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip(" ")
    if value[0] in _FORMULA_PREFIXES or (stripped and stripped[0] in _FORMULA_PREFIXES):
        return f"'{value}"
    return value


@dataclass(frozen=True)
class AxisCalibration:
    """Two-point calibration for a single axis (X or Y).

    Supports standard, reversed (p2_px < p1_px or v2 < v1), and cropped/offset
    coordinate systems on linear or base-10 logarithmic scales.
    """

    p1_px: float
    p2_px: float
    v1: float
    v2: float
    scale: str = "linear"
    label: str = ""
    unit: str = ""
    p1_uv: tuple[float, float] | None = None
    p2_uv: tuple[float, float] | None = None

    def validate(self, axis_name: str = "axis") -> None:
        if self.scale not in VALID_SCALES:
            raise ValueError(f"{axis_name} scale must be one of {VALID_SCALES}, got {self.scale!r}")
        if not (_is_finite_number(self.p1_px) and _is_finite_number(self.p2_px)):
            raise ValueError(f"{axis_name} pixel anchors must be finite numbers.")
        if abs(self.p2_px - self.p1_px) < MIN_AXIS_SPAN_PX:
            raise ValueError(
                f"{axis_name} pixel anchors p1_px and p2_px are degenerate "
                f"(|p2_px - p1_px| = {abs(self.p2_px - self.p1_px):.4g} < {MIN_AXIS_SPAN_PX} px)."
            )
        if not (_is_finite_number(self.v1) and _is_finite_number(self.v2)):
            raise ValueError(f"{axis_name} reference values v1 and v2 must be finite numbers.")
        if abs(self.v2 - self.v1) <= 1e-12:
            raise ValueError(f"{axis_name} reference values v1 and v2 must be distinct.")
        if self.scale == "log10" and (self.v1 <= 0.0 or self.v2 <= 0.0):
            raise ValueError(f"{axis_name} log10 scale requires strictly positive v1 and v2.")

        _validate_metadata_text(self.label, f"{axis_name}.label")
        _validate_metadata_text(self.unit, f"{axis_name}.unit")

        if (self.p1_uv is None) ^ (self.p2_uv is None):
            raise ValueError(f"{axis_name} must provide both p1_uv and p2_uv when 2D anchors are used.")
        if self.p1_uv is not None and self.p2_uv is not None:
            if (
                len(self.p1_uv) != 2
                or len(self.p2_uv) != 2
                or not all(_is_finite_number(c) for c in (*self.p1_uv, *self.p2_uv))
            ):
                raise ValueError(f"{axis_name} 2D anchors p1_uv and p2_uv must be finite (u, v) pairs.")
            dist_2d = math.hypot(self.p2_uv[0] - self.p1_uv[0], self.p2_uv[1] - self.p1_uv[1])
            if dist_2d < MIN_AXIS_SPAN_PX:
                raise ValueError(f"{axis_name} 2D anchors p1_uv and p2_uv are coincident/degenerate.")

    def fractional_coord(self, px: float) -> float:
        """Signed fractional position along the calibrated axis (0 at p1_px, 1 at p2_px)."""
        if not _is_finite_number(px):
            raise ValueError(f"Pixel coordinate must be finite, got {px!r}")
        return float((float(px) - self.p1_px) / (self.p2_px - self.p1_px))

    def in_pixel_bounds(self, px: float, *, tol_px: float = 1e-6) -> bool:
        """Return True if px lies within [min(p1_px, p2_px), max(p1_px, p2_px)] up to tol_px."""
        lo = min(self.p1_px, self.p2_px) - max(0.0, float(tol_px))
        hi = max(self.p1_px, self.p2_px) + max(0.0, float(tol_px))
        return lo <= float(px) <= hi

    def pixel_to_value(self, px: float) -> float:
        """Deterministic mapping from pixel coordinate to calibrated axis value."""
        frac = self.fractional_coord(px)
        if self.scale == "log10":
            lv1 = math.log10(self.v1)
            lv2 = math.log10(self.v2)
            return float(10.0 ** (lv1 + frac * (lv2 - lv1)))
        return float(self.v1 + frac * (self.v2 - self.v1))

    def value_to_pixel(self, val: float) -> float:
        """Inverse mapping from calibrated data value to pixel coordinate."""
        if not _is_finite_number(val):
            raise ValueError(f"Data value must be finite, got {val!r}")
        fval = float(val)
        if self.scale == "log10":
            if fval <= 0.0:
                raise ValueError("Cannot map non-positive value onto a log10 axis.")
            lv1 = math.log10(self.v1)
            lv2 = math.log10(self.v2)
            frac = (math.log10(fval) - lv1) / (lv2 - lv1)
        else:
            frac = (fval - self.v1) / (self.v2 - self.v1)
        return float(self.p1_px + frac * (self.p2_px - self.p1_px))


def validate_non_collinear_2d(
    x_cal: AxisCalibration,
    y_cal: AxisCalibration,
    extra_anchors_uv: Sequence[tuple[float, float]] | None = None,
) -> None:
    """Reject collinear or degenerate 2D calibration anchor geometries."""
    if x_cal.p1_uv is not None and x_cal.p2_uv is not None and y_cal.p1_uv is not None and y_cal.p2_uv is not None:
        dx = (x_cal.p2_uv[0] - x_cal.p1_uv[0], x_cal.p2_uv[1] - x_cal.p1_uv[1])
        dy = (y_cal.p2_uv[0] - y_cal.p1_uv[0], y_cal.p2_uv[1] - y_cal.p1_uv[1])
        nx = math.hypot(*dx)
        ny = math.hypot(*dy)
        if nx < MIN_AXIS_SPAN_PX or ny < MIN_AXIS_SPAN_PX:
            raise ValueError("Degenerate 2D axis calibration anchors.")
        sin_angle = abs(dx[0] * dy[1] - dx[1] * dy[0]) / (nx * ny)
        if sin_angle < MIN_AXIS_ANGLE_SIN:
            raise ValueError(
                "Collinear X and Y axis calibration vectors "
                f"(|sin(theta)| = {sin_angle:.4f} < {MIN_AXIS_ANGLE_SIN:.4f})."
            )

    if extra_anchors_uv is not None:
        pts = list(extra_anchors_uv)
        if len(pts) < 3:
            raise ValueError("At least 3 2D anchor points are required to span a 2D plane.")
        for u, v in pts:
            if not (_is_finite_number(u) and _is_finite_number(v)):
                raise ValueError("2D calibration anchors must have finite coordinates.")
        u0, v0 = float(pts[0][0]), float(pts[0][1])
        max_cross = 0.0
        max_len = 0.0
        for u1, v1 in pts[1:]:
            du1, dv1 = float(u1) - u0, float(v1) - v0
            max_len = max(max_len, math.hypot(du1, dv1))
            for u2, v2 in pts[1:]:
                du2, dv2 = float(u2) - u0, float(v2) - v0
                max_cross = max(max_cross, abs(du1 * dv2 - dv1 * du2))
        if max_len < MIN_AXIS_SPAN_PX or max_cross < (MIN_AXIS_ANGLE_SIN * (max_len ** 2)):
            raise ValueError("Calibration anchor points are collinear or degenerate in 2D pixel space.")


def parse_axis_calibration(raw: dict[str, Any], axis_name: str) -> AxisCalibration:
    if not isinstance(raw, dict):
        raise ValueError(f"{axis_name} must be a dict with p1_px, p2_px, v1, v2.")
    for req_key in ("p1_px", "p2_px", "v1", "v2"):
        if req_key not in raw or not _is_finite_number(raw[req_key]):
            raise ValueError(f"Invalid {axis_name} calibration: '{req_key}' must be a finite number.")
    raw_label = str(raw.get("label", axis_name.upper()))
    raw_unit = str(raw.get("unit", ""))
    _validate_metadata_text(raw_label, f"{axis_name}.label")
    _validate_metadata_text(raw_unit, f"{axis_name}.unit")
    p1_uv = tuple(float(c) for c in raw["p1_uv"]) if raw.get("p1_uv") is not None else None
    p2_uv = tuple(float(c) for c in raw["p2_uv"]) if raw.get("p2_uv") is not None else None
    cal = AxisCalibration(
        p1_px=float(raw["p1_px"]),
        p2_px=float(raw["p2_px"]),
        v1=float(raw["v1"]),
        v2=float(raw["v2"]),
        scale=str(raw.get("scale", "linear")).lower().strip(),
        label=raw_label.strip(),
        unit=raw_unit.strip(),
        p1_uv=p1_uv,  # type: ignore[arg-type]
        p2_uv=p2_uv,  # type: ignore[arg-type]
    )
    cal.validate(axis_name)
    return cal


def _trapezoidal_auc(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    """Compute point-estimate Cmax, Tmax, and linear-trapezoidal AUClast on a
    digitized concentration-time series sorted by X."""
    pts = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(x) and math.isfinite(y) and x >= 0.0 and y >= 0.0
    ]
    if len(pts) < 2:
        return {"cmax": None, "tmax": None, "auclast": None}
    pts.sort(key=lambda t: t[0])
    t_arr = np.array([p[0] for p in pts], dtype=float)
    c_arr = np.array([p[1] for p in pts], dtype=float)

    idx_max = int(np.argmax(c_arr))
    cmax = float(c_arr[idx_max])
    tmax = float(t_arr[idx_max])
    dt = np.diff(t_arr)
    auclast = float(np.sum(0.5 * dt * (c_arr[:-1] + c_arr[1:])))

    return {
        "cmax": round(cmax, 6),
        "tmax": round(tmax, 6),
        "auclast": round(auclast, 6),
    }


def validate_and_load_image(
    image_path: str | Path,
    *,
    allowed_dir: str | Path | None,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
    min_dim: int = MIN_IMAGE_DIMENSION,
    _after_fstat_hook: Callable[[], None] | None = None,
) -> tuple[str, str, dict[str, Any], np.ndarray]:
    """Safely open a direct child image inside `allowed_dir` using `dir_fd` + `O_NOFOLLOW`,
    validate via `fstat`, read at most `max_bytes + 1` bytes from the descriptor, and decode
    from bounded in-memory bytes without mutating `Image.MAX_IMAGE_PIXELS`.

    Returns `(filename, sha256_hex, image_dims_meta, grayscale_float_array)`.
    """
    if allowed_dir is None or not str(allowed_dir).strip():
        raise ValueError("allowed_dir is required for image loading.")
    if not image_path or not str(image_path).strip():
        raise ValueError("Image path must be a non-empty path.")

    raw_str = str(image_path).strip()
    if "\x00" in raw_str:
        raise ValueError("Image path contains null byte.")

    dir_str = str(allowed_dir).strip()
    if "\x00" in dir_str:
        raise ValueError("allowed_dir contains null byte.")
    dir_path = Path(dir_str).expanduser()

    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        dir_fd = os.open(str(dir_path), dir_flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, getattr(errno, "EFTYPE", -1)):
            raise ValueError("Symlinks are not permitted for allowed_dir.") from exc
        raise ValueError(f"allowed_dir cannot be opened as a directory: {allowed_dir}") from exc

    try:
        dir_st = os.fstat(dir_fd)
        if not stat.S_ISDIR(dir_st.st_mode):
            raise ValueError("allowed_dir is not a directory.")

        cand = Path(raw_str)
        filename = cand.name
        if filename in ("", ".", "..") or "/" in filename or "\\" in filename:
            raise ValueError("Image path must be a direct child of allowed_dir.")

        if cand.is_absolute():
            try:
                parent_st = os.stat(str(cand.parent), follow_symlinks=False)
            except OSError as exc:
                raise ValueError("Image path must be a direct child of allowed_dir.") from exc
            if (parent_st.st_dev, parent_st.st_ino) != (dir_st.st_dev, dir_st.st_ino):
                raise ValueError("Image path must be a direct child of allowed_dir (outside or nested path refused).")
        else:
            if len(cand.parts) != 1 or cand.parts[0] != filename:
                raise ValueError(
                    "Image path must be a direct child filename inside allowed_dir (nested paths refused)."
                )

        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(filename, file_flags, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, getattr(errno, "EFTYPE", -1)):
                raise ValueError(f"Symlinks are not permitted for image files: {filename}") from exc
            if exc.errno == errno.ENOENT:
                raise ValueError(f"Image file not found: {filename}") from exc
            if exc.errno == errno.EISDIR:
                raise ValueError(f"Image path is not a regular file: {filename}") from exc
            raise ValueError(f"Unable to open image file {filename}: {exc}") from exc

        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Image path is not a regular file: {filename}")
            if st.st_size <= 0:
                raise ValueError("Image file is empty (0 bytes).")
            if st.st_size > max_bytes:
                raise ValueError(
                    f"Image file size ({st.st_size} bytes) exceeds maximum limit ({max_bytes} bytes)."
                )

            if _after_fstat_hook is not None:
                _after_fstat_hook()

            chunks: list[bytes] = []
            total = 0
            read_cap = max_bytes + 1
            while total < read_cap:
                chunk = os.read(fd, min(65536, read_cap - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)

            if total <= 0:
                raise ValueError("Image file is empty (0 bytes).")
            if total > max_bytes:
                raise ValueError(
                    f"Image file grew beyond maximum limit ({max_bytes} bytes) during read."
                )
            raw_bytes = b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)

    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

    effective_max_pixels = min(int(max_pixels), MAX_IMAGE_PIXELS)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw_bytes)) as im:
                fmt = (im.format or "").upper()
                if fmt not in ALLOWED_IMAGE_FORMATS:
                    raise ValueError(
                        f"Unsupported image format {fmt!r}; allowed formats are {ALLOWED_IMAGE_FORMATS}."
                    )
                w, h = int(im.width), int(im.height)
                if w * h > effective_max_pixels:
                    raise ValueError(
                        f"Image pixel count ({w * h}) exceeds maximum limit ({effective_max_pixels} pixels)."
                    )
                if w < min_dim or h < min_dim:
                    raise ValueError(
                        f"Image dimensions ({w}x{h}) are smaller than minimum ({min_dim}x{min_dim})."
                    )
                im.load()
                gray = np.asarray(im.convert("L"), dtype=float)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(
            f"Image pixel count exceeds maximum limit ({effective_max_pixels} pixels)."
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    dims = {"width": w, "height": h, "format": fmt, "bytes": len(raw_bytes)}
    return filename, sha256_hex, dims, gray


def export_digitized_csv(result: dict[str, Any]) -> dict[str, str]:
    """Deterministically serialize digitized points and calibration metadata to CSV.

    - Rejects CR/LF and control characters in metadata fields.
    - Neutralizes leading spreadsheet formula prefixes (`=`, `+`, `-`, `@`) in CSV cells.
    Returns `{"csv": csv_string, "export_sha256": sha256_hex}`.
    """
    fig_label = _validate_metadata_text(str(result.get("figure_label") or ""), "figure_label")
    src_sha = _validate_metadata_text(str(result.get("source_sha256") or ""), "source_sha256")
    err_type = _validate_metadata_text(str(result.get("error_type") or "se"), "error_type")

    x_axis_dict = dict(result.get("x_axis") or {})
    y_axis_dict = dict(result.get("y_axis") or {})
    if "label" in x_axis_dict:
        _validate_metadata_text(str(x_axis_dict["label"]), "x_axis.label")
    if "unit" in x_axis_dict:
        _validate_metadata_text(str(x_axis_dict["unit"]), "x_axis.unit")
    if "label" in y_axis_dict:
        _validate_metadata_text(str(y_axis_dict["label"]), "y_axis.label")
    if "unit" in y_axis_dict:
        _validate_metadata_text(str(y_axis_dict["unit"]), "y_axis.unit")

    x_meta = json.dumps(x_axis_dict, sort_keys=True, separators=(",", ":"))
    y_meta = json.dumps(y_axis_dict, sort_keys=True, separators=(",", ":"))
    buf = io.StringIO()
    buf.write("# format=pharmagent-digitizer-v1\n")
    buf.write(f"# figure_label={fig_label}\n")
    buf.write(f"# source_sha256={src_sha}\n")
    buf.write(f"# x_axis={x_meta}\n")
    buf.write(f"# y_axis={y_meta}\n")
    buf.write(f"# pixel_sigma={result.get('pixel_sigma')}\n")
    buf.write(f"# anchor_sigma_px={result.get('anchor_sigma_px')}\n")
    buf.write(f"# error_type={err_type}\n")

    fieldnames = [
        "id",
        "series",
        "index",
        "u_px",
        "v_px",
        "v_low_px",
        "v_high_px",
        "x",
        "y",
        "y_low",
        "y_high",
        "out_of_bounds",
        "origin",
        "verified",
    ]
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(fieldnames)

    pts = list(result.get("points") or [])
    pts_sorted = sorted(
        pts,
        key=lambda r: (str(r.get("series", "")), int(r.get("index", 0)), str(r.get("id", ""))),
    )
    for pt in pts_sorted:
        pt_id = _validate_metadata_text(str(pt.get("id", "")), "point.id")
        pt_series = _validate_metadata_text(str(pt.get("series", "")), "point.series")
        pt_origin = _validate_metadata_text(str(pt.get("origin", "manual")), "point.origin")
        writer.writerow([
            _neutralize_csv_cell(pt_id),
            _neutralize_csv_cell(pt_series),
            pt.get("index", ""),
            pt.get("u_px", ""),
            pt.get("v_px", ""),
            "" if pt.get("v_low_px") is None else pt.get("v_low_px"),
            "" if pt.get("v_high_px") is None else pt.get("v_high_px"),
            pt.get("x", ""),
            pt.get("y", ""),
            "" if pt.get("y_low") is None else pt.get("y_low"),
            "" if pt.get("y_high") is None else pt.get("y_high"),
            bool(pt.get("out_of_bounds", False)),
            _neutralize_csv_cell(pt_origin),
            bool(pt.get("verified", False)),
        ])

    csv_text = buf.getvalue()
    export_sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    return {"csv": csv_text, "export_sha256": export_sha}


def digitize_plot(
    *,
    x_axis: dict[str, Any] | None = None,
    y_axis: dict[str, Any] | None = None,
    x_calibration: dict[str, Any] | None = None,
    y_calibration: dict[str, Any] | None = None,
    points: list[dict[str, Any]],
    pixel_sigma: float = 1.5,
    pixel_jitter: float | None = None,
    anchor_sigma_px: float | None = None,
    error_type: str = "se",
    error_bar_type: str | None = None,
    sample_n: int | None = None,
    strict_bounds: bool = False,
    calibration_points_2d: Sequence[tuple[float, float]] | None = None,
    source_path: str | None = None,
    source_sha256: str | None = None,
    allowed_dir: str | Path | None = None,
    figure_label: str = "Figure 1",
    approve_all: bool = False,
) -> dict[str, Any]:
    """Deterministically calibrate pixel points and directly digitized error-bar bounds
    into data coordinates with bounds checking and canonical CSV export metadata.
    """
    if pixel_jitter is not None:
        pixel_sigma = float(pixel_jitter)
    if error_bar_type is not None:
        error_type = str(error_bar_type)

    if not _is_finite_number(pixel_sigma) or float(pixel_sigma) < 0.0:
        raise ValueError(f"pixel_sigma must be a finite non-negative number, got {pixel_sigma!r}")
    eff_anchor_sigma = (
        float(anchor_sigma_px)
        if anchor_sigma_px is not None
        else round(0.75 * float(pixel_sigma), 6)
    )
    if not _is_finite_number(eff_anchor_sigma) or eff_anchor_sigma < 0.0:
        raise ValueError(f"anchor_sigma_px must be a finite non-negative number, got {anchor_sigma_px!r}")

    clean_fig_label = _validate_metadata_text(str(figure_label or "Figure 1"), "figure_label")

    x_cal = parse_axis_calibration(x_axis if x_axis is not None else (x_calibration or {}), "x_axis")
    y_cal = parse_axis_calibration(y_axis if y_axis is not None else (y_calibration or {}), "y_axis")
    validate_non_collinear_2d(x_cal, y_cal, extra_anchors_uv=calibration_points_2d)

    if not isinstance(points, list) or not points:
        raise ValueError("digitize_plot requires a non-empty list of points.")
    et = (error_type or "se").lower().strip()
    if et not in VALID_ERROR_TYPES:
        raise ValueError(f"error_type must be one of {VALID_ERROR_TYPES}, got {error_type!r}")
    if sample_n is not None and int(sample_n) < 1:
        raise ValueError("sample_n must be >= 1.")

    computed_sha = _validate_metadata_text(str(source_sha256), "source_sha256") if source_sha256 else None
    image_dims: dict[str, Any] | None = None
    if source_path:
        _, computed_sha, image_dims, _ = validate_and_load_image(
            source_path, allowed_dir=allowed_dir
        )

    digitized: list[dict[str, Any]] = []
    by_series: dict[str, list[dict[str, Any]]] = {}
    warnings_list: list[str] = []
    n_out_of_bounds = 0

    for idx, pt in enumerate(points, 1):
        if not isinstance(pt, dict) or "u_px" not in pt or "v_px" not in pt:
            raise ValueError(f"Point #{idx} must include 'u_px' and 'v_px' pixel coordinates.")
        if not (_is_finite_number(pt["u_px"]) and _is_finite_number(pt["v_px"])):
            raise ValueError(f"Point #{idx} pixel coordinates 'u_px' and 'v_px' must be finite numbers.")
        u_px = float(pt["u_px"])
        v_px = float(pt["v_px"])
        series = _validate_metadata_text(str(pt.get("series") or "Series 1"), f"points[{idx}].series")
        pt_id = _validate_metadata_text(str(pt.get("id") or f"pt_{idx}"), f"points[{idx}].id")
        origin = _validate_metadata_text(str(pt.get("origin") or "manual"), f"points[{idx}].origin")
        verified = bool(pt.get("verified", origin == "manual")) or bool(approve_all)

        v_low_px: float | None = None
        v_high_px: float | None = None
        if pt.get("v_low_px") is not None:
            if not _is_finite_number(pt["v_low_px"]):
                raise ValueError(f"Point {pt_id!r} v_low_px must be a finite number.")
            v_low_px = float(pt["v_low_px"])
        if pt.get("v_high_px") is not None:
            if not _is_finite_number(pt["v_high_px"]):
                raise ValueError(f"Point {pt_id!r} v_high_px must be a finite number.")
            v_high_px = float(pt["v_high_px"])

        oob_x = not x_cal.in_pixel_bounds(u_px)
        oob_y = not y_cal.in_pixel_bounds(v_px)
        oob_bar = (
            (v_low_px is not None and not y_cal.in_pixel_bounds(v_low_px))
            or (v_high_px is not None and not y_cal.in_pixel_bounds(v_high_px))
        )
        oob = oob_x or oob_y or oob_bar
        if oob:
            if strict_bounds:
                raise ValueError(
                    f"Point {pt_id!r} at ({u_px}, {v_px}) lies outside the calibrated axis region."
                )
            n_out_of_bounds += 1
            warnings_list.append(
                f"Point {pt_id!r} ({u_px}, {v_px}) is outside the calibrated pixel bounds."
            )

        x_val = x_cal.pixel_to_value(u_px)
        y_val = y_cal.pixel_to_value(v_px)

        y_low_raw = y_cal.pixel_to_value(v_low_px) if v_low_px is not None else None
        y_high_raw = y_cal.pixel_to_value(v_high_px) if v_high_px is not None else None

        if y_low_raw is not None and y_high_raw is not None:
            y_low, y_high = min(y_low_raw, y_high_raw), max(y_low_raw, y_high_raw)
        else:
            y_low, y_high = y_low_raw, y_high_raw

        row = {
            "id": pt_id,
            "index": idx,
            "series": series,
            "u_px": round(u_px, 4),
            "v_px": round(v_px, 4),
            "v_low_px": round(v_low_px, 4) if v_low_px is not None else None,
            "v_high_px": round(v_high_px, 4) if v_high_px is not None else None,
            "x": round(x_val, 6),
            "y": round(y_val, 6),
            "y_low": round(y_low, 6) if y_low is not None else None,
            "y_high": round(y_high, 6) if y_high is not None else None,
            "out_of_bounds_x": oob_x,
            "out_of_bounds_y": oob_y,
            "out_of_bounds": oob,
            "origin": origin,
            "verified": verified,
        }
        digitized.append(row)
        by_series.setdefault(series, []).append(row)

    series_summaries: dict[str, dict[str, Any]] = {}
    for s_name, s_pts in by_series.items():
        s_pts.sort(key=lambda r: r["x"])
        pk = _trapezoidal_auc(
            [r["x"] for r in s_pts],
            [r["y"] for r in s_pts],
        )
        n_ver = sum(1 for r in s_pts if r["verified"])
        series_summaries[s_name] = {
            "n_points": len(s_pts),
            "n_verified": n_ver,
            "all_verified": n_ver == len(s_pts),
            "x_range": [s_pts[0]["x"], s_pts[-1]["x"]],
            "y_range": [min(r["y"] for r in s_pts), max(r["y"] for r in s_pts)],
            "pk_metrics": pk,
        }

    n_verified_total = sum(1 for r in digitized if r["verified"])
    x_axis_out = {
        "p1_px": x_cal.p1_px,
        "p2_px": x_cal.p2_px,
        "v1": x_cal.v1,
        "v2": x_cal.v2,
        "scale": x_cal.scale,
        "label": x_cal.label or "Time",
        "unit": x_cal.unit or "h",
    }
    y_axis_out = {
        "p1_px": y_cal.p1_px,
        "p2_px": y_cal.p2_px,
        "v1": y_cal.v1,
        "v2": y_cal.v2,
        "scale": y_cal.scale,
        "label": y_cal.label or "Concentration",
        "unit": y_cal.unit or "ng/mL",
    }

    out: dict[str, Any] = {
        "status": "ok",
        "figure_label": clean_fig_label,
        "source_sha256": computed_sha,
        "image_dims": image_dims,
        "pixel_sigma": float(pixel_sigma),
        "pixel_jitter": float(pixel_sigma),
        "anchor_sigma_px": eff_anchor_sigma,
        "error_type": et,
        "error_bar_type": et,
        "strict_bounds": bool(strict_bounds),
        "x_axis": x_axis_out,
        "y_axis": y_axis_out,
        "x_calibration": x_axis_out,
        "y_calibration": y_axis_out,
        "n_points": len(digitized),
        "n_verified": n_verified_total,
        "all_verified": n_verified_total == len(digitized),
        "n_out_of_bounds": n_out_of_bounds,
        "warnings": warnings_list,
        "points": digitized,
        "series": series_summaries,
        "derived_pk": next(iter(series_summaries.values()), {}).get("pk_metrics", {}),
    }
    export_bundle = export_digitized_csv(out)
    out["csv_export"] = export_bundle["csv"]
    out["export_sha256"] = export_bundle["export_sha256"]
    return out


def propose_plot_from_image(
    image_path: str | Path,
    *,
    allowed_dir: str | Path | None,
    n_candidates: int = 8,
    x_range: tuple[float, float] = (0.0, 24.0),
    y_range: tuple[float, float] = (1.0, 100.0),
    y_scale: str = "linear",
    figure_label: str | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> dict[str, Any]:
    """Detect plot axes and candidate curve markers from an image using local
    deterministic luminance/contour analysis (zero cloud/LLM calls).

    - Requires a mandatory `allowed_dir` and opens only direct regular-file children.
    - Uses an opaque hash label (`source_<sha[:12]>`) by default rather than exposing
      the filesystem filename.
    - Marks every proposed point `origin='contour_proposed'` and `verified=False`.
    """
    _, sha, dims, gray = validate_and_load_image(
        image_path,
        allowed_dir=allowed_dir,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
    )
    h, w = gray.shape

    row_dark = np.mean(gray < 140, axis=1)
    col_dark = np.mean(gray < 140, axis=0)

    x_left = int(np.argmax(col_dark[int(0.05 * w):int(0.35 * w)]) + int(0.05 * w))
    x_right = int(0.90 * w)
    y_bottom = int(np.argmax(row_dark[int(0.65 * h):int(0.95 * h)]) + int(0.65 * h))
    y_top = int(0.10 * h)

    if x_right - x_left < 20:
        x_left, x_right = int(0.12 * w), int(0.90 * w)
    if y_bottom - y_top < 20:
        y_top, y_bottom = int(0.10 * h), int(0.85 * h)

    n_candidates = max(3, min(int(n_candidates), 25))
    u_grid = np.linspace(x_left + 10, x_right - 10, n_candidates)
    proposed_pts: list[dict[str, Any]] = []
    for idx, u_center in enumerate(u_grid, 1):
        u0 = max(x_left + 2, int(u_center - 4))
        u1 = min(x_right - 2, int(u_center + 4))
        strip = gray[y_top + 2:y_bottom - 2, u0:u1 + 1]
        if strip.size == 0:
            continue
        min_row = int(np.argmin(np.min(strip, axis=1)))
        v_px = float(y_top + 2 + min_row)
        proposed_pts.append({
            "id": f"prop_{idx}",
            "series": "Proposed Series 1",
            "u_px": round(float(u_center), 1),
            "v_px": round(v_px, 1),
            "origin": "contour_proposed",
            "verified": False,
        })

    opaque_label = figure_label if figure_label else f"source_{sha[:12]}"
    out = digitize_plot(
        x_axis={
            "p1_px": float(x_left),
            "p2_px": float(x_right),
            "v1": float(x_range[0]),
            "v2": float(x_range[1]),
            "scale": "linear",
            "label": "Time",
            "unit": "h",
        },
        y_axis={
            "p1_px": float(y_bottom),
            "p2_px": float(y_top),
            "v1": float(y_range[0]),
            "v2": float(y_range[1]),
            "scale": y_scale,
            "label": "Concentration",
            "unit": "ng/mL",
        },
        points=proposed_pts,
        source_sha256=sha,
        figure_label=opaque_label,
        approve_all=False,
    )
    out["image_dims"] = dims
    return out
