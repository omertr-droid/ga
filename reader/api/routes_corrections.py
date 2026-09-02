"""Phase-2 layer corrections. A correction is a full-width row (one y per A-scan) for ILM or BM on
one B-scan; it's persisted by the JsonSidecarLayerStore and folded into both the viewer overlay
(device_layers_json) and the projection (effective_surfaces). Saving/deleting drops the volume's
cached projections so the next request recomputes from the corrected surfaces."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from reader.core import layers
from .deps import get_store, get_layer_store
from .schemas import CorrectionIn, BmValidateIn

router = APIRouter()


def _ov(store, volume_id):
    try:
        return store.get_volume(volume_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Open the E2E first")


def resolve_row(base_row, ys, shift):
    """Turn an edit into the row to STORE (a Python list). A rigid `shift` returns the full base+shift
    row. An explicit `ys` is stored AS GIVEN with null/non-finite preserved as None: a column the human
    or DL did NOT provide stays UNSET — never silently back-filled from the `fill_bm` base. This keeps
    missing-BM detection honest (fixing the left gap can't auto-complete the right); the projection fills
    nulls itself in layers.effective_surfaces, so it never sees a NaN."""
    base = np.asarray(base_row, float)
    if shift is not None:
        return [float(v) for v in (base + float(shift))]
    if ys is not None:
        return [None if (y is None or not np.isfinite(y)) else float(y) for y in ys]
    return [float(v) for v in base]


def _refresh_missing(ov, layer_store):
    """Recompute the per-eye missing-BM map and refresh the Library's cached count (so the count never
    goes stale between visits to GET /bm_status). Returns the {bscan: runs} dict."""
    miss = layers.bm_missing_by_bscan(ov, layer_store)
    setter = getattr(layer_store, "set_missing_count", None)
    if setter:
        setter(ov.eid, ov.eye, len(miss))
    return miss


def validate_edit(ov, bscan, layer, ys=None):
    if layer not in ("ilm", "bm"):
        raise HTTPException(status_code=400, detail=f"unknown layer {layer!r}")
    if not (0 <= bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="bscan out of range")
    if ys is not None and len(ys) != ov.W:
        raise HTTPException(status_code=400, detail=f"expected {ov.W} values, got {len(ys)}")


@router.get("/volumes/{volume_id}/corrections")
def get_corrections(volume_id: str, bscan: int = Query(default=None),
                    store=Depends(get_store), layer_store=Depends(get_layer_store)):
    ov = _ov(store, volume_id)
    if bscan is None:
        return {"bscans": layer_store.corrected_indices(ov.eid, ov.eye)}
    return layer_store.get_corrected(ov.eid, ov.eye, bscan) or {}


@router.put("/volumes/{volume_id}/corrections")
def put_correction(volume_id: str, body: CorrectionIn,
                   store=Depends(get_store), layer_store=Depends(get_layer_store)):
    ov = _ov(store, volume_id)
    if body.layer not in ("ilm", "bm"):
        raise HTTPException(status_code=400, detail=f"unknown layer {body.layer!r}")
    if body.scope == "all":                                   # whole-volume rigid shift
        if body.shift is None:
            raise HTTPException(status_code=400, detail="scope='all' requires a shift")
        total = layer_store.add_global(ov.eid, ov.eye, body.layer, body.shift)
        store.invalidate_projection(volume_id)
        return {"ok": True, "scope": "all", "layer": body.layer, "global": total}
    if body.bscan is None:
        raise HTTPException(status_code=400, detail="scope='bscan' requires a bscan")
    validate_edit(ov, body.bscan, body.layer, body.ys)
    surf = layers.effective_surfaces(ov, layer_store)         # stack on any prior correction (shift base)
    base = (surf[0] if body.layer == "ilm" else surf[1])[body.bscan]
    ys = resolve_row(base, body.ys, body.shift)               # honest row: nulls preserved (no fill_bm)
    layer_store.put_corrected(ov.eid, ov.eye, body.bscan, body.layer, ys)
    store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "scope": "bscan", "bscan": body.bscan, "layer": body.layer, "ys": ys,
            "bm_missing": layers.bm_missing_runs(ov, layer_store, body.bscan)}


@router.post("/volumes/{volume_id}/corrections/resegment_bm")
def resegment_bm(volume_id: str, scope: str = Query(default="bscan"), bscan: int = Query(default=None),
                 store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Re-segment BM from the raw volume (fresh self-seg + GA-aware de-glitch) and store it as a BM
    correction, overriding whatever's there (device / cached self-seg / a prior edit). scope='bscan'
    (one B-scan, fast) or 'all' (whole volume, ~10s). Returns the new BM row(s)."""
    import bm as bmseg
    ov = _ov(store, volume_id)
    inv = ov.field_invalid                     # interpolate BM across saturated machine-fill columns
    if scope == "all":
        new = bmseg.resegment_bm_volume(ov.vol, invalid=inv)
        for i in range(ov.n_bscans):
            layer_store.put_corrected(ov.eid, ov.eye, i, "bm", new[i].tolist())
        store.invalidate_projection(volume_id)
        _refresh_missing(ov, layer_store)
        return {"ok": True, "scope": "all", "n": int(ov.n_bscans)}
    if bscan is None or not (0 <= bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="scope='bscan' requires a valid bscan")
    ys = bmseg.resegment_bm(ov.vol[bscan], invalid_row=(None if inv is None else inv[bscan])).tolist()
    layer_store.put_corrected(ov.eid, ov.eye, bscan, "bm", ys)
    store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "scope": "bscan", "bscan": bscan, "layer": "bm", "ys": ys,
            "bm_missing": layers.bm_missing_runs(ov, layer_store, bscan)}


@router.post("/volumes/{volume_id}/corrections/label_bm_dl")
def label_bm_dl(volume_id: str, scope: str = Query(default="bscan"), bscan: int = Query(default=None),
                mode: str = Query(default="overwrite"),
                store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Label BM with the DL model and store it as a BM correction — an editable starting label the user
    then fixes -> gold. scope='bscan' runs a 3-slice window (true 2.5D neighbours, fast); scope='all'
    runs the whole volume. mode='blanks' (scope='bscan' only) fills ONLY the missing columns with the DL
    prediction and KEEPS the existing device/human BM elsewhere — 'fill in the blanks'. Requires the DL
    model (env OCT_BM_DL + outputs/bm_dl/bm_unet.onnx)."""
    import bm_dl
    if not bm_dl.available():
        raise HTTPException(status_code=400, detail="DL BM model not available — set OCT_BM_DL and "
                            "place bm_unet.onnx in outputs/bm_dl/")
    ov = _ov(store, volume_id)
    if scope == "all":
        bm = bm_dl.segment_volume(ov.vol)
        for i in range(ov.n_bscans):
            layer_store.put_corrected(ov.eid, ov.eye, i, "bm", bm[i].tolist(), source="model")
        store.invalidate_projection(volume_id)
        _refresh_missing(ov, layer_store)
        return {"ok": True, "scope": "all", "n": int(ov.n_bscans)}
    if bscan is None or not (0 <= bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="scope='bscan' requires a valid bscan")
    lo, hi = max(0, bscan - 1), min(ov.n_bscans, bscan + 2)     # window -> true 2.5D neighbours for bscan
    bm = bm_dl.segment_volume(ov.vol[lo:hi])
    dl = bm[bscan - lo]
    if mode == "blanks":                                        # fill ONLY the missing cols; keep device/human BM
        existing = (layer_store.get_corrected(ov.eid, ov.eye, bscan) or {}).get("bm")
        ys = [None] * ov.W if existing is None else list(existing)
        for s, e in layers.bm_missing_runs(ov, layer_store, bscan):
            for x in range(s, e + 1):
                ys[x] = float(dl[x])
    else:
        ys = dl.tolist()                                        # overwrite the whole B-scan
    layer_store.put_corrected(ov.eid, ov.eye, bscan, "bm", ys, source="model")
    store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "scope": "bscan", "bscan": bscan, "layer": "bm", "ys": ys, "mode": mode,
            "bm_missing": layers.bm_missing_runs(ov, layer_store, bscan)}


def presegment_eye(ov, store, layer_store, volume_id=None):
    """Auto pre-segment BM with the DL model and CACHE it as corrections. Touches ONLY unsegmented +
    unvalidated B-scans: a B-scan is filled iff it is NOT validated, has NO existing BM correction, and
    still has missing (no-BM) columns. So it never overwrites device BM, a human edit, a prior model fill,
    or a validated B-scan, and is idempotent on re-open (already-filled B-scans skip -> no DL pass). The
    fill is tagged source='model' (the filmstrip's 'pre-segmented (model)' state). Soft no-op when the DL
    model is unavailable. Capability-only: does NOT need OCT_BM_DL. Shared by the endpoint (BM tab on open)
    and volume-open (server-side guarantee). Returns {ok, available, presegmented, n}."""
    import bm_dl
    if not bm_dl.available():
        return {"ok": True, "available": False, "presegmented": [], "n": 0}
    validated = set(layer_store.bm_validated(ov.eid, ov.eye))
    elig = [i for i in range(ov.n_bscans)
            if i not in validated
            and (layer_store.get_corrected(ov.eid, ov.eye, i) or {}).get("bm") is None
            and layers.bm_missing_runs(ov, layer_store, i)]
    if not elig:
        return {"ok": True, "available": True, "presegmented": [], "n": 0}
    bm = bm_dl.segment_volume(ov.vol)                            # one batched pass over the whole volume
    for i in elig:
        ys = [None] * ov.W                                      # leave non-missing cols unset -> device base kept
        for s, e in layers.bm_missing_runs(ov, layer_store, i):
            for x in range(s, e + 1):
                ys[x] = float(bm[i][x])
        layer_store.put_corrected(ov.eid, ov.eye, i, "bm", ys, source="model")
    if volume_id is not None:
        store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "available": True, "presegmented": elig, "n": len(elig)}


@router.post("/volumes/{volume_id}/corrections/presegment_bm")
def presegment_bm(volume_id: str, store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Auto pre-segment BM with the DL model and CACHE it as corrections — the BM tab fires this on open
    so the user never clicks Segment (also done server-side in volume-open as a guarantee). See
    presegment_eye for the eligibility rules."""
    ov = _ov(store, volume_id)
    return presegment_eye(ov, store, layer_store, volume_id)


@router.post("/volumes/{volume_id}/corrections/copy_prev")
def copy_prev(volume_id: str, bscan: int = Query(...), layer: str = Query(default="bm"),
              store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Copy the PREVIOUS B-scan's effective layer onto this one as its correction — a BM-labeling
    starting point the user then splines + validates. Returns the copied full-width row."""
    ov = _ov(store, volume_id)
    if layer not in ("ilm", "bm"):
        raise HTTPException(status_code=400, detail=f"unknown layer {layer!r}")
    if not (1 <= bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="need 1 <= bscan < n_bscans (B-scan 0 has no previous)")
    surf = layers.effective_surfaces(ov, layer_store)         # filled -> NaN-free
    ys = [float(v) for v in (surf[0] if layer == "ilm" else surf[1])[bscan - 1]]
    layer_store.put_corrected(ov.eid, ov.eye, bscan, layer, ys)
    store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "bscan": bscan, "from": bscan - 1, "layer": layer, "ys": ys}


@router.get("/volumes/{volume_id}/bm_status")
def get_bm_status(volume_id: str, store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Per-eye BM-validation progress for the filmstrip: which B-scans are validated / corrected, plus
    how many still have missing (device-gap) BM columns — cached so the Library can show it."""
    ov = _ov(store, volume_id)
    miss = layers.bm_missing_by_bscan(ov, layer_store)
    setter = getattr(layer_store, "set_missing_count", None)
    if setter:
        setter(ov.eid, ov.eye, len(miss))
    return {"n_bscans": int(ov.n_bscans),
            "validated": layer_store.bm_validated(ov.eid, ov.eye),
            "corrected": layer_store.corrected_indices(ov.eid, ov.eye),
            "missing_bscans": len(miss)}


@router.put("/volumes/{volume_id}/bm_status")
def put_bm_status(volume_id: str, body: BmValidateIn,
                  store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Mark (or unmark) a B-scan's BM validated. A B-scan can only be validated when 100% of its
    columns have a BM value (no device gap) — so green ⇒ a complete label. No projection invalidation."""
    ov = _ov(store, volume_id)
    if not (0 <= body.bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="bscan out of range")
    if body.validated:
        runs = layers.bm_missing_runs(ov, layer_store, body.bscan)
        if runs:
            spans = ", ".join(f"{s}–{e}" for s, e in runs)
            raise HTTPException(status_code=422, detail=(
                f"Cannot validate B-scan {body.bscan + 1}: columns {spans} have no BM. "
                "Fill them (Label with DL or edit), then validate."))
    layer_store.set_bm_validated(ov.eid, ov.eye, body.bscan, body.validated, body.by)
    return {"ok": True, "bscan": body.bscan, "validated": body.validated}


@router.delete("/volumes/{volume_id}/bm_status")
def clear_bm_status(volume_id: str, store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Clear EVERY BM-validation flag for this eye — part of the BM tab's full reset to device."""
    ov = _ov(store, volume_id)
    n = layer_store.clear_bm_validated_all(ov.eid, ov.eye)
    return {"ok": True, "cleared": int(n)}


@router.delete("/volumes/{volume_id}/corrections/all")
def delete_all_corrections(volume_id: str, layer: str = Query(default=None),
                           store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Reset a whole eye to the device layer: drop EVERY per-B-scan correction for `layer` (or all
    layers when omitted) plus its whole-volume shift — the BM tab's bulk 'reset to device'. Validation
    flags are left untouched (mirrors the per-B-scan Reset-to-device)."""
    ov = _ov(store, volume_id)
    n = layer_store.clear_corrected_all(ov.eid, ov.eye, layer)
    layer_store.clear_global(ov.eid, ov.eye, layer)
    store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "layer": layer, "cleared": int(n)}


@router.delete("/volumes/{volume_id}/corrections")
def delete_correction(volume_id: str, bscan: int = Query(default=None),
                      layer: str = Query(default=None), scope: str = Query(default="bscan"),
                      store=Depends(get_store), layer_store=Depends(get_layer_store)):
    ov = _ov(store, volume_id)
    if scope == "all":
        layer_store.clear_global(ov.eid, ov.eye, layer)
    else:
        layer_store.delete_corrected(ov.eid, ov.eye, bscan, layer)
    store.invalidate_projection(volume_id)
    _refresh_missing(ov, layer_store)
    return {"ok": True, "scope": scope, "bscan": bscan, "layer": layer}
