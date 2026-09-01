"""Single source of spatial calibration for the reader.

Spectralis en-face geometry is ANGULAR; with no biometry entered HEYEX uses the 24 mm model eye
(0.2924 mm/deg), so we measure each volume's field directly from the per-B-scan angular endpoints.
Axial sampling is metric (from the E2E). All ported from the proven pipeline:
  - MM_PER_DEG, class_fov_mm  <- src/validate_spectralis.py
  - AXIAL_UM_PER_PX           <- src/bm.py
"""
from collections import defaultdict

import numpy as np

# Spectralis axial sampling (metric), from the E2E — identical to bm.AXIAL_UM_PER_PX.
AXIAL_UM_PER_PX = 3.8716699928045273
# 24 mm model eye; reproduces the HEYEX display when no biometry was entered (CLAUDE.md Calibration).
MM_PER_DEG = 0.2924


def axial_mm_per_px():
    return AXIAL_UM_PER_PX / 1000.0


def class_fov_mm(bscan_data):
    """Map each volume class (numImages, imgSizeX) -> (H_mm, V_mm), measured directly from the
    per-B-scan angular endpoints (posX span = horizontal field; centrePosY span = vertical).

    Ported verbatim from src/validate_spectralis.py:class_fov_mm. Coverage is judged by FIELD
    EXTENT, not B-scan count.
    """
    groups = defaultdict(list)
    for b in bscan_data:
        groups[(b.get("numImages"), b.get("imgSizeX"))].append(b)
    out = {}
    for key, grp in groups.items():
        hs = [b["posX2"] - b["posX1"] for b in grp
              if b.get("posX1") is not None and b.get("posX2") is not None]
        cys = [b["centrePosY"] for b in grp if b.get("centrePosY") is not None]
        if not hs:
            continue
        h = float(np.median(hs)) * MM_PER_DEG
        v = (float(max(cys) - min(cys)) * MM_PER_DEG) if cys else 0.0
        out[key] = (h, v)
    return out


def lateral_mm_per_px(fov_mm, n_ascans):
    """Horizontal (A-scan) mm/px for a volume of the given field and width."""
    if not n_ascans:
        return 0.0
    return float(fov_mm[0]) / float(n_ascans)


def slow_mm_per_px(fov_mm, n_bscans):
    """Slow-axis (B-scan-to-B-scan) mm spacing."""
    if not n_bscans:
        return 0.0
    return float(fov_mm[1]) / float(n_bscans)
