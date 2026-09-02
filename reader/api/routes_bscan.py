"""B-scan image, layer arrays (JSON, separate from the image), and the localizer with position line."""
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from reader.core import e2e_source, layers, render
from .deps import get_store, get_layer_store

router = APIRouter()


def _vol(store, volume_id):
    try:
        return store.get_volume(volume_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Open the E2E first")


@router.get("/volumes/{volume_id}/bscan/{idx}.png")
def bscan_png(volume_id: str, idx: int, store=Depends(get_store)):
    ov = _vol(store, volume_id)
    if not (0 <= idx < ov.n_bscans):
        raise HTTPException(status_code=404, detail="B-scan index out of range")
    return Response(render.bscan_png(ov, idx), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/volumes/{volume_id}/layers")
def layers_json(volume_id: str, store=Depends(get_store), layer_store=Depends(get_layer_store)):
    ov = _vol(store, volume_id)
    return layers.device_layers_json(ov, layer_store)   # merges saved corrections (per bscan)


@router.get("/volumes/{volume_id}/localizer.png")
def localizer_png(volume_id: str, bscan: int = Query(default=0), store=Depends(get_store)):
    ov = _vol(store, volume_id)
    raw = store.get_raw(ov.eid)
    loc = e2e_source.localizer_image(raw, ov.eye) if raw is not None else None
    if loc is None:
        raise HTTPException(status_code=404, detail="No localizer for this eye")
    return Response(render.localizer_png(loc, bscan, ov.n_bscans), media_type="image/png")
