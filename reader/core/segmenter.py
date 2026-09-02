"""GA-segmentation seam (Phase 2). The viewer/API talk to this interface only, so a DL model drops
in later by implementing GaSegmenter — no transport or UI change (CLAUDE.md: keep atrophy behind an
interface so a DL segmenter can drop in later).
"""
from typing import Protocol, Tuple

import numpy as np

from . import projection as proj


class GaSegmenter(Protocol):
    def segment(self, enface: np.ndarray, fov_mm: tuple) -> Tuple[np.ndarray, float]:
        """en-face feature frame -> (binary mask, area_mm2)."""
        ...


class HeuristicSegmenter:
    """Placeholder threshold on the gated feature (NOT validated as an area estimate — CLAUDE.md
    found a global threshold gives R^2~0.02). Exists so the /ga endpoint returns something during
    development; replaced by the learned model once in-frame masks land."""
    def __init__(self, thresh=0.15):
        self.thresh = thresh

    def segment(self, enface, fov_mm):
        mask = np.asarray(enface, float) > self.thresh
        area_mm2 = float(mask.sum()) * (proj.ENFACE_MMPP ** 2)
        return mask.astype(np.uint8), area_mm2
