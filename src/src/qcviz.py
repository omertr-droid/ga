#!/usr/bin/env python
"""Shared QC-visualization helpers — labeled panels, scale bars, overlays, montages.

House style for every pipeline step's human-readable PNGs:
  - each sub-image gets a title bar; the whole panel gets an optional header strip,
  - a 1 mm scale bar (white) is drawn on spatial images (everything lives in a mm frame),
  - overlays carry a printed colour key, GA contours are drawn where a lesion is shown,
  - montage() lays out one tile per eye, caller sorts worst-first for triage.

Pure drawing on numpy uint8 RGB arrays via cv2 (no font files needed). Imported by
register_qc.py and the later measurement steps.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TITLE_H = 22
HEADER_H = 28


def ensure_rgb(img):
    """uint8 HxW or HxWx3 -> contiguous uint8 HxWx3."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = norm8(a)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    return np.ascontiguousarray(a)


def norm8(a):
    """Percentile-normalise an arbitrary array to uint8 (1-99%)."""
    a = np.asarray(a, float)
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    if hi <= lo:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1).__mul__(255).astype(np.uint8)


def _text(img, s, org, scale=0.45, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def add_scalebar(img, mm_per_px, mm=1.0, margin=10):
    """Draw a white mm scale bar at the bottom-left (in place)."""
    if not mm_per_px:
        return img
    h, w = img.shape[:2]
    length = int(round(mm / mm_per_px))
    length = max(4, min(length, w - 2 * margin))
    y = h - margin
    x0 = margin
    cv2.rectangle(img, (x0, y - 4), (x0 + length, y), WHITE, -1)
    cv2.rectangle(img, (x0 - 1, y - 5), (x0 + length + 1, y + 1), BLACK, 1)
    _text(img, f"{mm:g} mm", (x0, y - 8), 0.4)
    return img


def add_title(tile, title):
    """Stack a black title bar with white text above a tile."""
    w = tile.shape[1]
    bar = np.zeros((TITLE_H, w, 3), np.uint8)
    _text(bar, title[: max(1, int(w / 8))], (4, TITLE_H - 7), 0.42)
    return np.vstack([bar, tile])


def add_header(row, header):
    """Stack a dark header strip across the full panel width."""
    w = row.shape[1]
    bar = np.full((HEADER_H, w, 3), (40, 40, 40), np.uint8)
    _text(bar, header, (6, HEADER_H - 9), 0.5)
    return np.vstack([bar, row])


def _pad_h(tile, H):
    if tile.shape[0] == H:
        return tile
    pad = np.zeros((H - tile.shape[0], tile.shape[1], 3), np.uint8)
    return np.vstack([tile, pad])


def panel(images, titles, header=None, mm_per_px=None, scalebar_mm=1.0, sep=6, bar_on=None):
    """Lay images side-by-side, each titled; optional header strip + scale bars.

    bar_on: optional iterable of bools (one per image) to draw the scale bar only on
    selected tiles; default = all.
    """
    if bar_on is None:
        bar_on = [True] * len(images)
    tiles = []
    for img, title, draw_bar in zip(images, titles, bar_on):
        t = ensure_rgb(img).copy()
        if mm_per_px and draw_bar:
            add_scalebar(t, mm_per_px, scalebar_mm)
        tiles.append(add_title(t, title))
    H = max(t.shape[0] for t in tiles)
    tiles = [_pad_h(t, H) for t in tiles]
    gap = np.zeros((H, sep, 3), np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    if header:
        row = add_header(row, header)
    return row


def checkerboard(a, b, n=8):
    """Interleave two same-size images in an n x n checker pattern."""
    a, b = ensure_rgb(a), ensure_rgb(b)
    h, w = a.shape[:2]
    out = a.copy()
    sy, sx = h / n, w / n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                y0, y1 = int(i * sy), int((i + 1) * sy)
                x0, x1 = int(j * sx), int((j + 1) * sx)
                out[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    return out


def redgreen(a, b):
    """a -> red channel, b -> green channel (arrays are RGB). Overlap reads as yellow."""
    a = ensure_rgb(a)[:, :, 0]
    b = ensure_rgb(b)[:, :, 0]
    out = np.zeros((a.shape[0], a.shape[1], 3), np.uint8)
    out[:, :, 0] = a  # R
    out[:, :, 1] = b  # G
    return out


def draw_contour(img, mask, color=(255, 255, 0), thick=1):
    """Draw the boundary of a binary mask onto an RGB image (in place-ish, returns copy)."""
    out = ensure_rgb(img).copy()
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.any():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, thick, cv2.LINE_AA)
    return out


def label_tile(tile, text, scale=0.4):
    """Small caption bar under a montage tile."""
    w = tile.shape[1]
    bar = np.zeros((18, w, 3), np.uint8)
    _text(bar, text[: max(1, int(w / 7))], (3, 13), scale)
    return np.vstack([ensure_rgb(tile), bar])


def montage(tiles, cols=6, pad=4, bg=30):
    """Grid of equal-size RGB tiles (caller pre-labels + pre-sorts)."""
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    H = max(t.shape[0] for t in tiles)
    W = max(t.shape[1] for t in tiles)
    tiles = [ensure_rgb(t) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i : i + cols]
        cells = []
        for t in chunk:
            cell = np.full((H, W, 3), bg, np.uint8)
            cell[: t.shape[0], : t.shape[1]] = t
            cells.append(cell)
        while len(cells) < cols:
            cells.append(np.full((H, W, 3), bg, np.uint8))
        gap = np.full((H, pad, 3), bg, np.uint8)
        row = cells[0]
        for c in cells[1:]:
            row = np.hstack([row, gap, c])
        rows.append(row)
    vgap = np.full((pad, rows[0].shape[1], 3), bg, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, vgap, r])
    return out


def save_rgb(path, rgb):
    """Write an RGB uint8 array as PNG (handles cv2's BGR convention)."""
    cv2.imwrite(str(path), cv2.cvtColor(ensure_rgb(rgb), cv2.COLOR_RGB2BGR))
