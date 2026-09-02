"""Phase-2 GA segmentation overlay. 501 stub in the MVP; wiring = implement GaSegmenter and bind it
in deps.py (get_segmenter). Drawn by the frontend as a toggleable outline (qcviz.draw_contour)."""
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
_MSG = "GA segmentation overlay is a Phase-2 feature (not yet implemented)"


@router.get("/volumes/{volume_id}/ga.png")
def ga_png(volume_id: str, feature: str = Query(default="f_gated")):
    raise HTTPException(status_code=501, detail=_MSG)


@router.get("/volumes/{volume_id}/ga/meta")
def ga_meta(volume_id: str):
    raise HTTPException(status_code=501, detail=_MSG)
