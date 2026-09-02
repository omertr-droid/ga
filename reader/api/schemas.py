"""Request/response models (the typed contract). Responses are mostly plain dicts; these cover the
request bodies + a couple of Phase-2 shapes."""
from typing import List, Optional

from pydantic import BaseModel


class OpenE2EIn(BaseModel):
    path: str


class CorrectionIn(BaseModel):           # Phase 2 — manual layer correction
    layer: str                            # "ilm" | "bm"
    scope: str = "bscan"                  # "bscan" = this B-scan (absolute row) | "all" = whole volume
    bscan: Optional[int] = None           # required for scope="bscan"
    # scope="bscan": edited row as a full per-A-scan `ys` (null = unset -> base) OR a rigid `shift` px.
    # scope="all":   `shift` px applied to the whole layer surface (every B-scan).
    ys: Optional[List[Optional[float]]] = None
    shift: Optional[float] = None


class BmValidateIn(BaseModel):           # BM tab — mark a B-scan's BM validated/approved
    bscan: int
    validated: bool = True
    by: Optional[str] = None


class RunCreateIn(BaseModel):            # Segment tab — start a new (empty) labeling run
    label: Optional[str] = None
    source: str = "manual"               # "manual" | "threshold" | "model"
    concept: Optional[str] = None        # the prompt text (model runs)
    run: Optional[str] = None            # explicit run id (e.g. "gold" or "gold:rpe"); else auto from source
    cls: Optional[str] = None            # annotation class (wedge | rpe); stamps meta class + default invert
    invert: Optional[bool] = None        # override the class's default invert (RPE->loss gap)


class SeedIn(BaseModel):                 # Segment tab — build the threshold-baseline run
    threshold: float = 0.30              # hypertransmission ratio threshold
    slab_lo: Optional[float] = None      # sub-BM slab (µm); None -> m3 default (10)
    slab_hi: Optional[float] = None      # None -> m3 default (340)
    run: Optional[str] = None            # target run id (default "threshold")


class EndpointIn(BaseModel):             # Segment tab — MedSAM3 Colab tunnel URL (Phase 2)
    url: str


class LabelBscanIn(BaseModel):           # Segment tab — MedSAM3 label one B-scan
    bscan: int
    concept: str
    threshold: float = 0.5               # mask binarisation threshold passed to the model
    run: Optional[str] = None            # target run; default derived from the concept


class LabelVolumeIn(BaseModel):          # Segment tab — MedSAM3 label the whole volume (background job)
    concept: str
    threshold: float = 0.5
    run: Optional[str] = None
    bscan_range: Optional[List[int]] = None   # [lo, hi] inclusive; default = all B-scans
    cls: Optional[str] = None            # annotation class to stamp on the run (e.g. "rpe")
    invert: Optional[bool] = None        # set the run's invert (RPE-present -> loss gap)


class StatusIn(BaseModel):               # Studio — set a B-scan's annotation state
    run: str
    bscan: int
    state: Optional[str] = None          # ga | ga_free | borderline | todo
    reviewed: Optional[bool] = None


class MarkFreeIn(BaseModel):             # Studio — mark B-scan(s) GA-free (explicit negatives)
    run: str
    bscan: Optional[int] = None
    bscan_range: Optional[List[int]] = None   # [lo, hi]; if all None -> all remaining 'todo'


class SamBscanIn(BaseModel):             # Studio — SAM2 box/point assist on one B-scan
    bscan: int
    box: Optional[List[float]] = None         # [x1, y1, x2, y2] in B-scan pixels
    points: Optional[List[List[float]]] = None   # [[x, y, label], ...]  (label 1 fg / 0 bg)
    run: Optional[str] = None


class SamPropagateIn(BaseModel):         # Studio — apply a SAM2 box across a B-scan range (background)
    bscan: int
    box: List[float]
    bscan_range: Optional[List[int]] = None   # [lo, hi]; default = a window around bscan
    run: Optional[str] = None


class SamEnfaceIn(BaseModel):            # Studio — SAM2-annotate the B-scans under an en-face box (drag
    x0: float                             # on the projection). Coords are normalized [0,1] of the en-face.
    y0: float
    x1: float
    y1: float
    run: Optional[str] = None


class PreviewIn(BaseModel):              # live (unsaved) edit preview -> projection PNG
    feature: str
    bscan: Optional[int] = None          # a pending per-B-scan edit (with layer + ys/shift)
    layer: Optional[str] = None
    ys: Optional[List[Optional[float]]] = None
    shift: Optional[float] = None        # pending per-B-scan shift (px)
    global_ilm: Optional[float] = None   # pending whole-volume ILM shift (px)
    global_bm: Optional[float] = None    # pending whole-volume BM shift (px)
    slab_lo: Optional[float] = None      # sub-BM slab window (µm) for raw/f_trans (the "green zone")
    slab_hi: Optional[float] = None
    lo: Optional[float] = None           # display window (defaults to the feature window)
    hi: Optional[float] = None
