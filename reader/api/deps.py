"""Dependency-injection singletons. The ONLY place that binds concrete implementations, so swapping
the layer-correction store or the GA segmenter (Phase 2) is a one-line change here — routes and the
frontend are untouched.
"""
import json
import os

from reader.core.layer_store import JsonSidecarLayerStore
from reader.core.mask_store import PngMaskStore
from reader.core.segmenter import HeuristicSegmenter
from .session import SessionStore

_DS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store")
_DATA_STORE = os.path.join(_DS, "corrections")      # reader/data_store/corrections
_SEG_STORE = os.path.join(_DS, "segmentations")     # reader/data_store/segmentations (Segment tab)
_ENDPOINT_FILE = os.path.join(_SEG_STORE, "_endpoint.json")   # MedSAM3 Colab tunnel URL (Phase 2)

_store = SessionStore()
_layer_store = JsonSidecarLayerStore(_DATA_STORE)   # Phase 2: persisted manual layer corrections
_mask_store = PngMaskStore(_SEG_STORE)              # Segment-tab GA masks (runs)
_segmenter = HeuristicSegmenter()                   # Phase 2: swap for the learned model
_store.attach_layer_store(_layer_store)             # cached projections fold in corrections


def get_store() -> SessionStore:
    return _store


def get_layer_store():
    return _layer_store


def get_segmenter():
    return _segmenter


def get_mask_store() -> PngMaskStore:
    return _mask_store


def get_seg_endpoint() -> str:
    """The MedSAM3 service URL (Colab tunnel), set by the user in the Segment tab (Phase 2)."""
    try:
        with open(_ENDPOINT_FILE) as f:
            return (json.load(f) or {}).get("url", "") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def set_seg_endpoint(url: str) -> None:
    os.makedirs(_SEG_STORE, exist_ok=True)
    tmp = _ENDPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"url": url or ""}, f)
    os.replace(tmp, _ENDPOINT_FILE)
