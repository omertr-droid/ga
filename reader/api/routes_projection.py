"""En-face projection: feature PNG at a display window, meta, and live band-tuned recompute."""
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from reader.core import layers, projection as proj, render
from .deps import get_store, get_layer_store
from .routes_corrections import resolve_row, validate_edit
from .schemas import PreviewIn

router = APIRouter()


def _vol(store, volume_id):
    try:
        return store.get_volume(volume_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Open the E2E first")


@router.get("/volumes/{volume_id}/projection.png")
def projection_png(volume_id: str, feature: str = Query(default=proj.DEFAULT_FEATURE),
                   lo: float = Query(default=None), hi: float = Query(default=None),
                   store=Depends(get_store)):
    if feature not in proj.FEATURES:
        raise HTTPException(status_code=400, detail=f"unknown feature {feature!r}")
    _vol(store, volume_id)                      # 404 if not open
    m = store.get_projection(volume_id, feature)
    window = (lo, hi) if lo is not None and hi is not None else proj.default_window(feature)
    return Response(render.projection_png(m, window), media_type="image/png")


@router.get("/volumes/{volume_id}/projection/meta")
def projection_meta(volume_id: str, feature: str = Query(default=proj.DEFAULT_FEATURE),
                    store=Depends(get_store)):
    ov = _vol(store, volume_id)
    return {
        "feature": feature, "features": list(proj.FEATURES),
        "default_window": list(proj.default_window(feature)),
        "windows": {k: list(v) for k, v in proj.FEATURES.items()},
        "mmpp": proj.ENFACE_MMPP, "bands": proj.bands_meta(), "fov_mm": list(ov.fov_mm),
    }


@router.get("/volumes/{volume_id}/projection/recompute.png")
def projection_recompute(volume_id: str,
                         slab_lo: float = Query(default=10.0), slab_hi: float = Query(default=340.0),
                         lo: float = Query(default=None), hi: float = Query(default=None),
                         store=Depends(get_store), layer_store=Depends(get_layer_store)):
    ov = _vol(store, volume_id)
    surf = layers.effective_surfaces(ov, layer_store)
    m = proj.recompute_transmit(ov, surf, slab_lo, slab_hi)
    window = (lo, hi) if lo is not None and hi is not None else proj.default_window("f_trans")
    return Response(render.projection_png(m, window), media_type="image/png")


@router.post("/volumes/{volume_id}/projection/preview.png")
def projection_preview(volume_id: str, body: PreviewIn,
                       store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Projection with LIVE (unsaved) adjustments — the real-time feedback in the B-scan workbench:
      - a whole-volume rigid layer shift (`global_ilm`/`global_bm`),
      - a sub-BM slab (`slab_lo`/`slab_hi`, the green zone) for raw/f_trans,
      - a per-B-scan edit (`bscan`+`layer`+`shift`|`ys`).
    A pure per-B-scan edit at the default slab patches only that B-scan's cached native row (fast); a
    global shift or slab change recomputes the whole en-face."""
    if body.feature not in proj.FEATURES:
        raise HTTPException(status_code=400, detail=f"unknown feature {body.feature!r}")
    ov = _vol(store, volume_id)
    surf = layers.effective_surfaces(ov, layer_store)            # saved global + per-B-scan corrections
    gi, gb = body.global_ilm or 0.0, body.global_bm or 0.0       # pending (unsaved) whole-volume shift
    if gi or gb:
        surf = (surf[0] + gi, surf[1] + gb)
    slab = (body.slab_lo, body.slab_hi) if body.slab_lo is not None and body.slab_hi is not None else None

    if body.bscan is not None and body.layer:                    # a pending per-B-scan edit
        validate_edit(ov, body.bscan, body.layer, body.ys)
        base = (surf[0] if body.layer == "ilm" else surf[1])[body.bscan]
        row = resolve_row(base, body.ys, body.shift)
        if slab is None and not (gi or gb) and body.feature in proj.SINGLE_NATIVE:
            nat = store.get_native(volume_id, body.feature)      # fast: patch the one row
            m = proj.preview_enface(ov, body.feature, surf, body.bscan, body.layer, row, nat=nat)
        else:
            ilm, bm = surf[0].copy(), surf[1].copy()
            (ilm if body.layer == "ilm" else bm)[body.bscan] = row
            m = proj.render_feature(ov, body.feature, (ilm, bm), slab)
    else:
        m = proj.render_feature(ov, body.feature, surf, slab)

    window = (body.lo, body.hi) if body.lo is not None and body.hi is not None \
        else proj.default_window(body.feature)
    return Response(render.projection_png(m, window), media_type="image/png")
