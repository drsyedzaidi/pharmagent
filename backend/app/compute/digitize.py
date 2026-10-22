"""Deterministic Plot Digitizer with 2-point axis calibration, explicit plot-region (ROI)
bounds, verified-only per-series PK metrics, directly digitized error-bar bounds, and
bounded-memory `uint8` local contour proposals.

Design invariants ("Models propose, tools compute, humans verify"):
- Pixel-to-data mapping (`linear` or `log10`), reversed/cropped axis handling,
  direct error-bar bound conversion, and linear-trapezoidal AUClast point-estimate
  integration are 100% deterministic NumPy/math calculations.
- Calibration anchors (`x_axis`, `y_axis`) define the coordinate transformation; points
  outside the calibration-anchor span are labeled `extrapolated=True`, not automatically
  out-of-bounds.
- Explicit plot-region bounds (`plot_roi`) define whether a point is inside the graph
  (`bounds_status` is `"in_bounds"` or `"out_of_bounds"` when `plot_roi` is declared, and
  `"unknown"` when `plot_roi` is omitted). `strict_bounds=True` requires and enforces `plot_roi`.
- Derived PK metrics (`cmax`, `tmax`, `auclast`) are always keyed by series (`series` /
  `series_pk`) with no ambiguous top-level scalar summary, and are withheld with a named
  unresolved status (`unresolved_unverified_points`, `unresolved_out_of_bounds`, etc.)
  unless every point in the series is human-verified and either inside `plot_roi` or covered
  by a separately recorded override.
- Local image loading (`validate_and_load_image`) requires a mandatory `allowed_dir`,
  opens only direct regular-file children via `dir_fd` with `O_NOFOLLOW`, validates via
  `fstat`, reads at most `max_bytes + 1` bytes, never mutates `Image.MAX_IMAGE_PIXELS`,
  enforces a 4 MiP (`4,194,304` px) ceiling, and keeps grayscale image data in `uint8`
  (1 byte/pixel) with strip-wise contour projections.
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
import threading
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

# Measured per-call image memory budget:
# - Raw compressed file bytes: <= 10.0 MiB (MAX_IMAGE_BYTES)
# - Decoded Pillow RGBA/RGB buffer during load: <= 4.0 MiP * 4 B = 16.0 MiB
# - Retained grayscale uint8 array: <= 4.0 MiP * 1 B = 4.0 MiB
# - Strip-wise boolean projection scratch (256 rows x 2048 cols): <= 0.5 MiB
# Total peak per call <= 30.5 MiB (31,981,568 B); bounded to 2 concurrent proposals.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB hard cap
MAX_IMAGE_PIXELS = 4_194_304        # 4 MiP (e.g., 2048x2048) hard cap
MIN_IMAGE_DIMENSION = 32            # minimum width/height in pixels
PROJECTION_STRIP_ROWS = 256         # strip height for contour projection
IMAGE_MEMORY_BUDGET_BYTES = (
    MAX_IMAGE_BYTES
    + (MAX_IMAGE_PIXELS * 4)
    + (MAX_IMAGE_PIXELS * 1)
    + (PROJECTION_STRIP_ROWS * 2048)
)
MAX_CONCURRENT_IMAGE_PROPOSALS = 2
_IMAGE_PROPOSAL_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_IMAGE_PROPOSALS)

MIN_AXIS_SPAN_PX = 1.0              # minimum 1 px separation between calibration anchors
MIN_AXIS_ANGLE_SIN = math.sin(math.radians(5.0))  # minimum 5 deg angle between 2D axes

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

    def in_anchor_span(self, px: float, *, tol_px: float = 1e-6) -> bool:
        """Return True if px lies within the calibration-anchor span [min(p1_px, p2_px), max(p1_px, p2_px)]."""
        lo = min(self.p1_px, self.p2_px) - max(0.0, float(tol_px))
        hi = max(self.p1_px, self.p2_px) + max(0.0, float(tol_px))
        return lo <= float(px) <= hi

    def in_pixel_bounds(self, px: float, *, tol_px: float = 1e-6) -> bool:
        """Alias for `in_anchor_span`."""
        return self.in_anchor_span(px, tol_px=tol_px)

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


@dataclass(frozen=True)
class PlotROI:
    """Explicit plot-region bounding box in image pixel coordinates.

    Separated from calibration anchors (`AxisCalibration`) because calibration ticks
    are frequently interior ticks rather than the edges of the plotting frame.
    """

    u_min_px: float
    u_max_px: float
    v_min_px: float
    v_max_px: float

    def validate(self) -> None:
        for name, val in (
            ("u_min_px", self.u_min_px),
            ("u_max_px", self.u_max_px),
            ("v_min_px", self.v_min_px),
            ("v_max_px", self.v_max_px),
        ):
            if not _is_finite_number(val):
                raise ValueError(f"plot_roi.{name} must be a finite number.")
        if self.u_max_px - self.u_min_px < MIN_AXIS_SPAN_PX:
            raise ValueError(
                f"plot_roi horizontal span must be >= {MIN_AXIS_SPAN_PX} px "
                f"(got u_min_px={self.u_min_px}, u_max_px={self.u_max_px})."
            )
        if self.v_max_px - self.v_min_px < MIN_AXIS_SPAN_PX:
            raise ValueError(
                f"plot_roi vertical span must be >= {MIN_AXIS_SPAN_PX} px "
                f"(got v_min_px={self.v_min_px}, v_max_px={self.v_max_px})."
            )

    def in_u_bounds(self, u_px: float, *, tol_px: float = 1e-6) -> bool:
        tol = max(0.0, float(tol_px))
        return (self.u_min_px - tol) <= float(u_px) <= (self.u_max_px + tol)

    def in_v_bounds(self, v_px: float, *, tol_px: float = 1e-6) -> bool:
        tol = max(0.0, float(tol_px))
        return (self.v_min_px - tol) <= float(v_px) <= (self.v_max_px + tol)

    def to_dict(self) -> dict[str, float]:
        return {
            "u_min_px": float(self.u_min_px),
            "u_max_px": float(self.u_max_px),
            "v_min_px": float(self.v_min_px),
            "v_max_px": float(self.v_max_px),
        }


def parse_plot_roi(raw: dict[str, Any] | PlotROI | None) -> PlotROI | None:
    if raw is None:
        return None
    if isinstance(raw, PlotROI):
        raw.validate()
        return raw
    if not isinstance(raw, dict):
        raise ValueError("plot_roi must be a dict with u_min_px, u_max_px, v_min_px, v_max_px.")
    for req_key in ("u_min_px", "u_max_px", "v_min_px", "v_max_px"):
        if req_key not in raw or not _is_finite_number(raw[req_key]):
            raise ValueError(f"Invalid plot_roi: '{req_key}' must be a finite number.")
    roi = PlotROI(
        u_min_px=float(raw["u_min_px"]),
        u_max_px=float(raw["u_max_px"]),
        v_min_px=float(raw["v_min_px"]),
        v_max_px=float(raw["v_max_px"]),
    )
    roi.validate()
    return roi


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
    verified, in-bounds digitized concentration-time series sorted by X."""
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


def _compute_dark_projections_uint8(
    gray: np.ndarray,
    *,
    threshold: int = 140,
    strip_rows: int = PROJECTION_STRIP_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row and per-column counts of dark pixels (`gray < threshold`)
    on a 2D `uint8` grayscale image in row strips without allocating full-frame
    boolean or float64 arrays.
    """
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError(
            f"_compute_dark_projections_uint8 requires a 2D uint8 ndarray, got "
            f"dtype={getattr(gray, 'dtype', None)}, ndim={getattr(gray, 'ndim', None)}."
        )
    h, w = gray.shape
    step = max(1, int(strip_rows))
    thresh_u8 = np.uint8(max(0, min(255, int(threshold))))
    row_dark_counts = np.empty(h, dtype=np.int32)
    col_dark_counts = np.zeros(w, dtype=np.int32)

    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        strip_mask = gray[r0:r1, :] < thresh_u8
        row_dark_counts[r0:r1] = np.count_nonzero(strip_mask, axis=1)
        col_dark_counts += np.count_nonzero(strip_mask, axis=0)

    return row_dark_counts, col_dark_counts


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
    into a `uint8` grayscale array (`1 byte/pixel`) without mutating `Image.MAX_IMAGE_PIXELS`.

    Returns `(filename, sha256_hex, image_dims_meta, grayscale_uint8_array)`.
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
                with im.convert("L") as gray_im:
                    gray = np.array(gray_im, dtype=np.uint8, copy=True)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(
            f"Image pixel count exceeds maximum limit ({effective_max_pixels} pixels)."
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    dims = {
        "width": w,
        "height": h,
        "format": fmt,
        "bytes": len(raw_bytes),
        "dtype": str(gray.dtype),
    }
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
    roi_meta = json.dumps(result.get("plot_roi"), sort_keys=True, separators=(",", ":"))
    buf = io.StringIO()
    buf.write("# format=pharmagent-digitizer-v1\n")
    buf.write(f"# figure_label={fig_label}\n")
    buf.write(f"# source_sha256={src_sha}\n")
    buf.write(f"# x_axis={x_meta}\n")
    buf.write(f"# y_axis={y_meta}\n")
    buf.write(f"# plot_roi={roi_meta}\n")
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
        "extrapolated",
        "bounds_status",
        "out_of_bounds",
        "oob_override",
        "origin",
        "verified",
    ]
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(fieldnames)

    pts = list(result.get("points") or [])
    pts_sorted = sorted(
        pts,
        key=lambda r: (
            str(r.get("series", "")),
            float(r.get("x", 0.0)),
            str(r.get("id", "")),
            float(r.get("u_px", 0.0)),
            float(r.get("v_px", 0.0)),
        ),
    )
    for canonical_idx, pt in enumerate(pts_sorted, 1):
        pt_id = _validate_metadata_text(str(pt.get("id", "")), "point.id")
        pt_series = _validate_metadata_text(str(pt.get("series", "")), "point.series")
        pt_origin = _validate_metadata_text(str(pt.get("origin", "manual")), "point.origin")
        pt_bstatus = _validate_metadata_text(str(pt.get("bounds_status", "unknown")), "point.bounds_status")
        oob_val = pt.get("out_of_bounds")
        writer.writerow([
            _neutralize_csv_cell(pt_id),
            _neutralize_csv_cell(pt_series),
            canonical_idx,
            pt.get("u_px", ""),
            pt.get("v_px", ""),
            "" if pt.get("v_low_px") is None else pt.get("v_low_px"),
            "" if pt.get("v_high_px") is None else pt.get("v_high_px"),
            pt.get("x", ""),
            pt.get("y", ""),
            "" if pt.get("y_low") is None else pt.get("y_low"),
            "" if pt.get("y_high") is None else pt.get("y_high"),
            bool(pt.get("extrapolated", False)),
            _neutralize_csv_cell(pt_bstatus),
            "" if oob_val is None else bool(oob_val),
            bool(pt.get("oob_override", False)),
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
    plot_roi: dict[str, Any] | PlotROI | None = None,
    points: list[dict[str, Any]],
    pixel_sigma: float = 1.5,
    pixel_jitter: float | None = None,
    anchor_sigma_px: float | None = None,
    error_type: str = "se",
    error_bar_type: str | None = None,
    sample_n: int | None = None,
    strict_bounds: bool = False,
    oob_overrides: dict[str, str] | None = None,
    calibration_points_2d: Sequence[tuple[float, float]] | None = None,
    source_path: str | None = None,
    source_sha256: str | None = None,
    allowed_dir: str | Path | None = None,
    figure_label: str = "Figure 1",
    approve_all: bool = False,
) -> dict[str, Any]:
    """Deterministically calibrate pixel points and directly digitized error-bar bounds
    into data coordinates with explicit `plot_roi` bounds checking, verified-only
    series-keyed PK metrics, and canonical CSV export metadata.
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

    roi_obj = parse_plot_roi(plot_roi)
    if strict_bounds and roi_obj is None:
        raise ValueError("strict_bounds=True requires an explicit plot_roi.")

    clean_overrides: dict[str, str] = {}
    if oob_overrides is not None:
        if not isinstance(oob_overrides, dict):
            raise ValueError("oob_overrides must be a dict mapping point_id -> non-empty reason string.")
        for k, v in oob_overrides.items():
            clean_k = _validate_metadata_text(str(k), "oob_overrides.key")
            clean_v = _validate_metadata_text(str(v), f"oob_overrides[{clean_k!r}]")
            if not clean_v:
                raise ValueError(f"oob_overrides[{clean_k!r}] must provide a non-empty reason.")
            clean_overrides[clean_k] = clean_v

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

    by_series: dict[str, list[dict[str, Any]]] = {}
    warnings_list: list[str] = []

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
        verified = bool(pt.get("verified", False)) or bool(approve_all)

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

        extrap_x = not x_cal.in_anchor_span(u_px)
        extrap_y = (
            (not y_cal.in_anchor_span(v_px))
            or (v_low_px is not None and not y_cal.in_anchor_span(v_low_px))
            or (v_high_px is not None and not y_cal.in_anchor_span(v_high_px))
        )
        extrapolated = extrap_x or extrap_y

        if roi_obj is None:
            oob_x: bool | None = None
            oob_y: bool | None = None
            oob: bool | None = None
            pt_bounds_status = "unknown"
        else:
            oob_x = not roi_obj.in_u_bounds(u_px)
            oob_y = (
                (not roi_obj.in_v_bounds(v_px))
                or (v_low_px is not None and not roi_obj.in_v_bounds(v_low_px))
                or (v_high_px is not None and not roi_obj.in_v_bounds(v_high_px))
            )
            oob = oob_x or oob_y
            pt_bounds_status = "out_of_bounds" if oob else "in_bounds"
            if oob and strict_bounds:
                raise ValueError(
                    f"Point {pt_id!r} at ({u_px}, {v_px}) lies outside the declared plot_roi."
                )

        pt_override_reason = clean_overrides.get(pt_id)
        if pt_override_reason is None and pt.get("oob_override_reason") is not None:
            cand_reason = _validate_metadata_text(
                str(pt["oob_override_reason"]), f"points[{idx}].oob_override_reason"
            )
            if cand_reason:
                pt_override_reason = cand_reason
        if bool(pt.get("oob_override")) and not pt_override_reason:
            raise ValueError(
                f"Point {pt_id!r} specifies oob_override=True without a non-empty oob_override_reason."
            )
        has_oob_override = bool(pt_override_reason)

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
            "index": 0,  # Assigned canonically after sorting by (series, x, id)
            "series": series,
            "u_px": round(u_px, 4),
            "v_px": round(v_px, 4),
            "v_low_px": round(v_low_px, 4) if v_low_px is not None else None,
            "v_high_px": round(v_high_px, 4) if v_high_px is not None else None,
            "x": round(x_val, 6),
            "y": round(y_val, 6),
            "y_low": round(y_low, 6) if y_low is not None else None,
            "y_high": round(y_high, 6) if y_high is not None else None,
            "extrapolated_x": extrap_x,
            "extrapolated_y": extrap_y,
            "extrapolated": extrapolated,
            "bounds_status": pt_bounds_status,
            "out_of_bounds_x": oob_x,
            "out_of_bounds_y": oob_y,
            "out_of_bounds": oob,
            "oob_override": has_oob_override,
            "oob_override_reason": pt_override_reason,
            "origin": origin,
            "verified": verified,
        }
        by_series.setdefault(series, []).append(row)

    seen_point_ids = {
        r["id"] for s_pts in by_series.values() for r in s_pts
    }
    unknown_override_ids = sorted(set(clean_overrides.keys()) - seen_point_ids)
    if unknown_override_ids:
        raise ValueError(
            f"oob_overrides references unknown point IDs not present in points: {unknown_override_ids}"
        )

    digitized: list[dict[str, Any]] = []
    series_summaries: dict[str, dict[str, Any]] = {}
    series_pk: dict[str, dict[str, float | None]] = {}
    all_unverified_ids: list[str] = []
    all_blocking_oob_ids: list[str] = []
    all_blocking_ids: list[str] = []
    unresolved_reasons: list[str] = []

    for s_name in sorted(by_series.keys()):
        s_pts = by_series[s_name]
        s_pts.sort(key=lambda r: (r["x"], r["id"], r["u_px"], r["v_px"]))
        for r in s_pts:
            r["index"] = len(digitized) + 1
            digitized.append(r)
            if r["out_of_bounds"] is True:
                warnings_list.append(
                    f"Point {r['id']!r} ({r['u_px']}, {r['v_px']}) in series {s_name!r} "
                    "is outside the declared plot_roi."
                )

        unverified_ids = sorted(r["id"] for r in s_pts if not r["verified"])
        blocking_oob_ids = sorted(
            r["id"] for r in s_pts if r["out_of_bounds"] is True and not r["oob_override"]
        )
        negative_ids = sorted(
            r["id"] for r in s_pts if r["x"] < 0.0 or r["y"] < 0.0
        )
        blocking_ids = sorted(set(unverified_ids) | set(blocking_oob_ids) | set(negative_ids))
        recorded_overrides = [
            {"id": r["id"], "reason": str(r["oob_override_reason"])}
            for r in sorted(s_pts, key=lambda item: item["id"])
            if r["out_of_bounds"] is True and r["oob_override"]
        ]

        if unverified_ids and blocking_oob_ids:
            s_metrics_status = "unresolved_unverified_and_out_of_bounds"
            s_status = s_metrics_status
            s_reason: str | None = (
                f"Series {s_name!r} metrics withheld: unverified points {unverified_ids} "
                f"and out-of-bounds points without override {blocking_oob_ids}."
            )
            pk: dict[str, float | None] = {"cmax": None, "tmax": None, "auclast": None}
        elif unverified_ids:
            s_metrics_status = "unresolved_unverified_points"
            s_status = s_metrics_status
            s_reason = (
                f"Series {s_name!r} metrics withheld: unverified points {unverified_ids} "
                "must be human-verified before computing Cmax, Tmax, and AUClast."
            )
            pk = {"cmax": None, "tmax": None, "auclast": None}
        elif blocking_oob_ids:
            s_metrics_status = "unresolved_out_of_bounds"
            s_status = s_metrics_status
            s_reason = (
                f"Series {s_name!r} metrics withheld: out-of-bounds points {blocking_oob_ids} "
                "lie outside plot_roi without a recorded override."
            )
            pk = {"cmax": None, "tmax": None, "auclast": None}
        elif negative_ids:
            s_metrics_status = "unresolved_negative_coordinates"
            s_status = s_metrics_status
            s_reason = (
                f"Series {s_name!r} metrics withheld: points {negative_ids} have negative "
                "calibrated time or concentration values."
            )
            pk = {"cmax": None, "tmax": None, "auclast": None}
        else:
            pk = _trapezoidal_auc(
                [r["x"] for r in s_pts],
                [r["y"] for r in s_pts],
            )
            if pk["cmax"] is None:
                s_metrics_status = "unresolved_insufficient_points"
                s_status = s_metrics_status
                s_reason = (
                    f"Series {s_name!r} metrics withheld: at least 2 finite non-negative "
                    "verified points are required."
                )
            else:
                s_metrics_status = "resolved"
                s_status = "ok"
                s_reason = None

        all_unverified_ids.extend(unverified_ids)
        all_blocking_oob_ids.extend(blocking_oob_ids)
        all_blocking_ids.extend(blocking_ids)
        if s_reason:
            unresolved_reasons.append(s_reason)

        n_ver = sum(1 for r in s_pts if r["verified"])
        n_s_oob = sum(1 for r in s_pts if r["out_of_bounds"] is True)
        n_s_extrap = sum(1 for r in s_pts if r["extrapolated"] is True)
        s_bounds_status = (
            "unknown"
            if roi_obj is None
            else ("out_of_bounds" if n_s_oob > 0 else "in_bounds")
        )

        series_summaries[s_name] = {
            "series": s_name,
            "status": s_status,
            "metrics_status": s_metrics_status,
            "unresolved_reason": s_reason,
            "blocking_point_ids": blocking_ids,
            "unverified_point_ids": unverified_ids,
            "blocking_oob_point_ids": blocking_oob_ids,
            "recorded_oob_overrides": recorded_overrides,
            "n_points": len(s_pts),
            "n_verified": n_ver,
            "all_verified": n_ver == len(s_pts),
            "n_extrapolated": n_s_extrap,
            "n_out_of_bounds": n_s_oob,
            "bounds_status": s_bounds_status,
            "x_range": [s_pts[0]["x"], s_pts[-1]["x"]],
            "y_range": [min(r["y"] for r in s_pts), max(r["y"] for r in s_pts)],
            "pk_metrics": pk,
        }
        series_pk[s_name] = pk

    n_verified_total = sum(1 for r in digitized if r["verified"])
    n_extrapolated_total = sum(1 for r in digitized if r["extrapolated"] is True)
    n_out_of_bounds_total = sum(1 for r in digitized if r["out_of_bounds"] is True)
    overall_bounds_status = (
        "unknown"
        if roi_obj is None
        else ("out_of_bounds" if n_out_of_bounds_total > 0 else "in_bounds")
    )

    if all_unverified_ids and all_blocking_oob_ids:
        overall_metrics_status = "unresolved_unverified_and_out_of_bounds"
        overall_status = overall_metrics_status
    elif all_unverified_ids:
        overall_metrics_status = "unresolved_unverified_points"
        overall_status = overall_metrics_status
    elif all_blocking_oob_ids:
        overall_metrics_status = "unresolved_out_of_bounds"
        overall_status = overall_metrics_status
    elif any(s["metrics_status"] != "resolved" for s in series_summaries.values()):
        overall_metrics_status = "unresolved_insufficient_points"
        overall_status = overall_metrics_status
    else:
        overall_metrics_status = "resolved"
        overall_status = "ok"

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
        "status": overall_status,
        "metrics_status": overall_metrics_status,
        "unresolved_reason": " ".join(unresolved_reasons) if unresolved_reasons else None,
        "blocking_point_ids": sorted(set(all_blocking_ids)),
        "figure_label": clean_fig_label,
        "source_sha256": computed_sha,
        "image_dims": image_dims,
        "pixel_sigma": float(pixel_sigma),
        "pixel_jitter": float(pixel_sigma),
        "anchor_sigma_px": eff_anchor_sigma,
        "error_type": et,
        "error_bar_type": et,
        "strict_bounds": bool(strict_bounds),
        "plot_roi": roi_obj.to_dict() if roi_obj is not None else None,
        "bounds_status": overall_bounds_status,
        "x_axis": x_axis_out,
        "y_axis": y_axis_out,
        "x_calibration": x_axis_out,
        "y_calibration": y_axis_out,
        "n_points": len(digitized),
        "n_verified": n_verified_total,
        "all_verified": n_verified_total == len(digitized),
        "n_extrapolated": n_extrapolated_total,
        "n_out_of_bounds": n_out_of_bounds_total,
        "warnings": warnings_list,
        "points": digitized,
        "series": series_summaries,
        "series_pk": series_pk,
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
    deterministic `uint8` luminance/contour analysis (zero cloud/LLM calls).

    - Enforces bounded concurrency via `_IMAGE_PROPOSAL_SEMAPHORE`.
    - Requires a mandatory `allowed_dir` and opens only direct regular-file children.
    - Uses strip-wise `uint8` projections (`_compute_dark_projections_uint8`) without
      full-frame boolean or float64 arrays.
    - Marks every proposed point `origin='contour_proposed'` and `verified=False`,
      so `digitize_plot` withholds derived PK metrics (`status='unresolved_unverified_points'`)
      until a human verifies every point.
    """
    if not _IMAGE_PROPOSAL_SEMAPHORE.acquire(blocking=False):
        raise RuntimeError(
            f"Image proposal admission limit reached (max {MAX_CONCURRENT_IMAGE_PROPOSALS} concurrent proposals)."
        )
    try:
        _, sha, dims, gray = validate_and_load_image(
            image_path,
            allowed_dir=allowed_dir,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
        )
        h, w = gray.shape
        row_dark_counts, col_dark_counts = _compute_dark_projections_uint8(
            gray, threshold=140, strip_rows=PROJECTION_STRIP_ROWS
        )

        c0, c1 = int(0.05 * w), max(int(0.05 * w) + 1, int(0.35 * w))
        r0, r1 = int(0.65 * h), max(int(0.65 * h) + 1, int(0.95 * h))
        x_left = int(np.argmax(col_dark_counts[c0:c1]) + c0)
        x_right = int(0.90 * w)
        y_bottom = int(np.argmax(row_dark_counts[r0:r1]) + r0)
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
            plot_roi={
                "u_min_px": float(x_left),
                "u_max_px": float(x_right),
                "v_min_px": float(y_top),
                "v_max_px": float(y_bottom),
            },
            points=proposed_pts,
            source_sha256=sha,
            figure_label=opaque_label,
            approve_all=False,
        )
        out["image_dims"] = dims
        return out
    finally:
        _IMAGE_PROPOSAL_SEMAPHORE.release()
