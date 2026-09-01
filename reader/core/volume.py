"""Domain models for an opened E2E and a loaded OCT volume."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VolumeRef:
    """Lightweight descriptor of one OCT volume inside an E2E (no pixel data)."""
    index: int                 # position in raw.read_oct_volume()
    eye: str                   # OD / OS / U
    n_bscans: int
    H: int
    W: int
    fov_mm: tuple              # (H_mm, V_mm) measured from angular geometry; (0,0) if unknown
    is_6x6: bool               # the ~6x6 square scan (512-wide, ~5.8mm) — the reader's default
    kind: str                  # human label e.g. "97line/512px (6x6)"


@dataclass
class RawE2E:
    """An opened E2E held in memory: the oct-converter object + its decoded metadata/volumes."""
    path: str
    eid: str
    reader: object             # oct_converter E2E instance
    vols: list                 # read_oct_volume() result
    cfov: dict                 # (numImages,imgSizeX) -> (H_mm,V_mm)
    md: dict                   # read_all_metadata()
    funduses: list = field(default_factory=list)  # read_fundus_image() result (lazy/optional)
    refs: list = field(default_factory=list)       # [VolumeRef]


@dataclass
class OctVolume:
    """A fully loaded volume ready to view + project. Surfaces are row indices per A-scan (n,W).

    *_display  = what the B-scan viewer draws (device contour with NaN gaps where missing, or the
                 self-seg line); may contain NaN -> sent as null.
    *_filled   = gap-filled + smoothed surfaces for the projection math (m2_bm.fill_bm).
    *_src      = "device" or "auto" (self-segmented), per layer.
    """
    volume_id: str
    eid: str
    index: int
    eye: str
    vol: np.ndarray                      # (n, H, W) float
    ilm_display: Optional[np.ndarray]    # (n, W) or None
    bm_display: Optional[np.ndarray]
    ilm: np.ndarray                      # (n, W) filled — always present
    bm: np.ndarray                       # (n, W) filled — always present
    ilm_src: str
    bm_src: str
    fov_mm: tuple
    # (n, W) bool, True = real in-field A-scan; False = saturated machine-fill ('white band') column.
    # None => treat every column valid (back-compat: consumers guard on `field_valid is None`).
    field_valid: Optional[np.ndarray] = None
    # Does the en-face need its rows flipped to reach fundus orientation? True for every eye whose posY
    # DECREASES with B-scan index (the norm). 003-016 and 003-130 raster the slow axis in reverse, so
    # their rows are already fundus-ordered and a blanket flip would turn the projection upside-down.
    # See e2e_source.enface_flip_for. Default True = the historical unconditional flip.
    enface_flip: bool = True

    @property
    def field_invalid(self):
        """(n, W) bool of machine-fill / out-of-field A-scans, or None if no mask was computed."""
        return None if self.field_valid is None else ~self.field_valid

    @property
    def n_bscans(self):
        return int(self.vol.shape[0])

    @property
    def H(self):
        return int(self.vol.shape[1])

    @property
    def W(self):
        return int(self.vol.shape[2])
