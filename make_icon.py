"""Generate PharmAgent.app's icon with the Python stdlib only (no Pillow).

Design: a dark-luxury rounded-square (macOS squircle-ish) with a vertical
gradient, an oral PK concentration-time curve drawn as a glowing accent stroke
with a translucent area fill, and a few observation dots on the curve. Renders a
1024x1024 RGBA PNG via analytic anti-aliasing (signed-distance rounded rect +
soft-disc curve stamping), then `iconutil` turns the iconset into AppIcon.icns.
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

N = 1024  # canvas size


# ── tiny vector / color helpers ────────────────────────────────────────────────
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3 - 2 * t)


def over(dst: tuple[float, float, float, float],
         src: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Straight-alpha 'source over' compositing."""
    sr, sg, sb, sa = src
    dr, dg, db, da = dst
    oa = sa + da * (1 - sa)
    if oa <= 1e-9:
        return (0.0, 0.0, 0.0, 0.0)
    orr = (sr * sa + dr * da * (1 - sa)) / oa
    og = (sg * sa + dg * da * (1 - sa)) / oa
    ob = (sb * sa + db * da * (1 - sa)) / oa
    return (orr, og, ob, oa)


def sd_rounded_rect(px: float, py: float, cx: float, cy: float,
                    hx: float, hy: float, r: float) -> float:
    """Signed distance from (px,py) to a rounded rect centered at (cx,cy)."""
    qx = abs(px - cx) - (hx - r)
    qy = abs(py - cy) - (hy - r)
    ax, ay = max(qx, 0.0), max(qy, 0.0)
    return math.hypot(ax, ay) + min(max(qx, qy), 0.0) - r


# ── PK curve: one-compartment oral, c(t) ∝ ka/(ka-ke)·(e^-ke t − e^-ka t) ───────
KA, KE = 1.1, 0.28


def conc(t: float) -> float:
    return (KA / (KA - KE)) * (math.exp(-KE * t) - math.exp(-KA * t))


# ── build the curve polyline in pixel space ─────────────────────────────────────
MARGIN = 96.0                      # transparent border around the squircle
RECT_HALF = (N - 2 * MARGIN) / 2   # half-size of the icon square
CX = CY = N / 2
CORNER = RECT_HALF * 2 * 0.225     # macOS-ish corner radius

# plot box inside the squircle
PLOT_L, PLOT_R = MARGIN + 150, N - MARGIN - 110
PLOT_T, PLOT_B = MARGIN + 175, N - MARGIN - 215
T_MAX = 14.0
_cmax = max(conc(t) for t in [i * 0.02 for i in range(int(T_MAX / 0.02))])


def curve_xy(t: float) -> tuple[float, float]:
    x = lerp(PLOT_L, PLOT_R, t / T_MAX)
    y = lerp(PLOT_B, PLOT_T, conc(t) / _cmax)
    return x, y


_PTS = [curve_xy(i * (T_MAX / 600)) for i in range(601)]
_DOTS = [curve_xy(t) for t in (1.4, 3.1, 5.6, 9.2)]

# ── palette (dark luxury, teal accent) ──────────────────────────────────────────
BG_TOP = (0.043, 0.071, 0.125)     # #0B1220
BG_BOT = (0.075, 0.106, 0.180)     # #131B2E
ACCENT = (0.235, 0.839, 0.776)     # #3CD6C6 teal
ACCENT_HI = (0.62, 0.95, 0.91)     # lighter teal for dots/glow core
AXIS = (0.55, 0.62, 0.78)


def dist_to_polyline(px: float, py: float, pts: list[tuple[float, float]]) -> float:
    best = 1e18
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            d = math.hypot(px - ax, py - ay)
        else:
            t = clamp(((px - ax) * dx + (py - ay) * dy) / seg2)
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        if d < best:
            best = d
    return best


def render() -> bytearray:
    buf = bytearray(N * N * 4)
    # precompute curve bounding box to skip distance work far from it
    minx = min(p[0] for p in _PTS) - 60
    maxx = max(p[0] for p in _PTS) + 60
    miny = min(p[1] for p in _PTS) - 60
    maxy = max(p[1] for p in _PTS) + 60

    stroke = 13.0       # half-width of the curve stroke
    glow = 34.0         # glow radius

    for y in range(N):
        for x in range(N):
            px, py = x + 0.5, y + 0.5
            # 1. background squircle
            d = sd_rounded_rect(px, py, CX, CY, RECT_HALF, RECT_HALF, CORNER)
            cov = clamp(0.5 - d)        # 1 inside, 0 outside, AA on the 1px edge
            if cov <= 0.0:
                continue
            gt = smoothstep(0.0, 1.0, (py - MARGIN) / (N - 2 * MARGIN))
            col = (lerp(BG_TOP[0], BG_BOT[0], gt),
                   lerp(BG_TOP[1], BG_BOT[1], gt),
                   lerp(BG_TOP[2], BG_BOT[2], gt),
                   cov)

            # subtle baseline axis
            if PLOT_L - 4 <= px <= PLOT_R + 4:
                da = abs(py - PLOT_B)
                if da < 2.5:
                    a = (1 - da / 2.5) * 0.5 * cov
                    col = over(col, (AXIS[0], AXIS[1], AXIS[2], a))

            if minx <= px <= maxx and miny <= py <= maxy:
                dc = dist_to_polyline(px, py, _PTS)
                # area fill under the curve (between curve and baseline)
                if py >= PLOT_B - 0.0:
                    pass
                # glow
                if dc < glow:
                    ga = (1 - dc / glow) ** 2 * 0.33 * cov
                    col = over(col, (ACCENT[0], ACCENT[1], ACCENT[2], ga))
                # crisp stroke
                sa = clamp(stroke - dc + 0.5) * cov
                if sa > 0:
                    col = over(col, (ACCENT[0], ACCENT[1], ACCENT[2], sa))

                # observation dots
                for dx0, dy0 in _DOTS:
                    dd = math.hypot(px - dx0, py - dy0)
                    if dd < 17:
                        ring = clamp(11 - dd + 0.5) * cov
                        if ring > 0:
                            col = over(col, (ACCENT_HI[0], ACCENT_HI[1], ACCENT_HI[2], ring))

            i = (y * N + x) * 4
            buf[i] = int(clamp(col[0]) * 255 + 0.5)
            buf[i + 1] = int(clamp(col[1]) * 255 + 0.5)
            buf[i + 2] = int(clamp(col[2]) * 255 + 0.5)
            buf[i + 3] = int(clamp(col[3]) * 255 + 0.5)
    return buf


def write_png(buf: bytearray, path: Path) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = bytearray()
    for y in range(N):
        raw.append(0)  # no filter
        raw.extend(buf[(y * N) * 4:(y * N + N) * 4])
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", N, N, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


if __name__ == "__main__":
    out = Path(__file__).parent / "icon_1024.png"
    write_png(render(), out)
    print(f"wrote {out}")
