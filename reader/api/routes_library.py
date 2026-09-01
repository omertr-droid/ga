"""Cohort Library for the BM-validation workflow: list the device-seeded scans with per-scan
validation progress (red/orange/green), plus a light per-eye status refresh. Listing is pure
CSV + JSON (no E2E decode), backed by results/bm_worklist.csv (precomputed by src/build_bm_worklist.py).
Also exposes the BM-dataset export (the 'Export BM dataset' button): packs every validated eye into
outputs/bm_dataset/bm_dataset.zip and streams it back for download.
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from reader.core import library
from .deps import get_layer_store

router = APIRouter()


@router.get("/library")
def library_list(layer_store=Depends(get_layer_store)):
    return library.list_scans(layer_store)


@router.get("/library/bm_status/{eid}/{eye}")
def library_bm_status(eid: str, eye: str, layer_store=Depends(get_layer_store)):
    """Validated B-scan indices for one eye — lets the Library refresh a single row's progress
    after a validation pass without re-listing the whole cohort."""
    return {"eid": eid, "eye": eye, "validated": layer_store.bm_validated(eid, eye)}


@router.post("/library/export_bm_dataset")
def export_bm_dataset(classical: bool = False):
    """Run the BM-dataset export over every validated eye -> outputs/bm_dataset/ (npz + manifest +
    splits + bm_dataset.zip) and stream the zip back for download. Saturated 'white band' columns are
    excluded (weight 0) by the export, so the zip is training-ready. `classical=false` (default) skips
    the slow graph-search eval baseline for a responsive button; pass classical=true for full parity
    with the CLI. (A `def` route, so FastAPI runs it in a threadpool — the long export won't block.)"""
    import export_bm_dataset as ebd
    try:
        res = ebd.run_export(want_classical=classical, want_zip=True, write=True)
    except Exception as e:                                              # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"export failed: {type(e).__name__}: {e}")
    if not res or not res.get("zip_path") or not os.path.exists(res["zip_path"]):
        raise HTTPException(status_code=400,
                            detail="no validated eyes to export — validate BM in the Library first")
    return FileResponse(
        res["zip_path"], filename="bm_dataset.zip", media_type="application/zip",
        headers={"X-Export-Eyes": str(res["n_eyes"]), "X-Export-Bscans": str(res["n_bscans"]),
                 "X-Export-Patients": str(res["n_patients"])})
