"""The clinic API router. Three groups of endpoints:

  * upload   — ``/upload/open`` (list 6x6 scans, no GA) then ``/upload/process`` (compute + record);
               ``/reopen`` re-opens a database row.
  * database — ``/db`` (patient list), ``/db/{patient_id}`` (a patient's visits), ``/db.xlsx`` /
               ``/db.csv`` (export), ``DELETE /db`` (drop a row; the E2E is never touched).
  * panels   — ``/scan/{vid}/…`` PNGs + JSON for the 3-panel viewer, served through one ViewSource.

There is no PLEX anywhere (no ``/plex`` route, empty plex meta fields) — the OCT-only invariant.
"""
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from clinic.core import db, fsbrowse, staging, xlsx
from .deps import get_store
from .schemas import OpenIn, ProcessIn, ReopenIn

router = APIRouter()


# --------------------------------------------------------------------------- helpers
def _src(store, vid):
    try:
        return store.source(vid)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="unknown or unloaded scan")


def _png(data):
    if data is None:
        raise HTTPException(status_code=404, detail="no image")
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})


def _e2e_error(e):
    """Map an E2E open/process failure to the right HTTP error (locked file -> 423, else 500)."""
    if db.is_lock_error(e):
        raise HTTPException(status_code=423,
                            detail="The E2E file is open or locked in another program; close it and try again.")
    raise HTTPException(status_code=500, detail=f"could not read E2E: {e}")


# --------------------------------------------------------------------------- health / config
@router.get("/health")
def health():
    return {"ok": True, "app": "ga-clinic"}


@router.get("/config")
def config():
    dl_available, dl_default = False, False
    try:
        import bm_dl                                          # clinic.core import already set up sys.path
        dl_available, dl_default = bool(bm_dl.discoverable()), bool(bm_dl.enabled())
    except Exception:
        pass
    return {"title": "GA Clinic", "dl_available": dl_available, "dl_default": dl_default}


# --------------------------------------------------------------------------- file browser (Browse…)
@router.get("/fs/list")
def fs_list(path: str = ""):
    """List the sub-folders and .E2E files of a server-side directory (defaults to the user's home),
    for the upload 'Browse…' picker. Localhost single-user only."""
    return fsbrowse.list_dir(path or None)


# --------------------------------------------------------------------------- upload (3 steps)
@router.post("/upload/stage")
async def upload_stage(request: Request, name: str = ""):
    """Step 0 (drag-and-drop only): stream a dropped .E2E to disk and hand back its server path.

    The body is raw bytes (``application/octet-stream``), not multipart — ``python-multipart`` is not
    installed, and streaming avoids buffering a ~300 MB file in memory. ``name`` is echoed back for
    display and NEVER touches the filesystem: the staged path is ``<sha256 of the content>.E2E``, so a
    malicious filename cannot escape the uploads directory, and re-dropping a file is idempotent.
    The client then runs the ordinary ``/upload/open`` -> ``/upload/process`` flow on the returned path.
    """
    try:
        res = await staging.stage_stream(request)
    except staging.StageError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)
    res["name"] = name
    return res


@router.post("/upload/open")
def upload_open(body: OpenIn, store=Depends(get_store)):
    """Step 1: decode the E2E once and return its 6x6-measurable scans (+ DL availability + identity)."""
    if not body.path or not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail="file not found on this machine")
    try:
        return store.list_scans(body.path)
    except Exception as e:                                     # noqa: BLE001
        _e2e_error(e)


@router.post("/upload/process")
def upload_process(body: ProcessIn, store=Depends(get_store)):
    """Step 2: process the chosen scan with the chosen BM source; record it; return vid + meta."""
    if not body.path or not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail="file not found on this machine")
    try:
        vid, meta, reg, warning = store.process(body.path, body.index, body.bm_choice)
    except ValueError as e:                                   # bad index / not a 6x6 volume
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:                                    # noqa: BLE001
        _e2e_error(e)
    return {"vid": vid, "meta": meta, "db": reg, "warning": warning}


@router.post("/upload/finish")
def upload_finish(body: OpenIn, store=Depends(get_store)):
    """Drop the large decoded E2E and DL session after all selected eyes have been processed."""
    return store.finish_batch(body.path or None)


@router.post("/reopen")
def reopen(body: ReopenIn, store=Depends(get_store)):
    """Re-open a recorded scan by its database record id (serve from cache, else re-process)."""
    try:
        vid, meta, reg, warning = store.reopen(body.record_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such record")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except db.DbLocked as e:
        raise HTTPException(status_code=423, detail=str(e))
    except Exception as e:                                    # noqa: BLE001
        _e2e_error(e)
    finally:
        # Reopen processes one recorded eye, so there is no second-eye batch to keep the decoder/model for.
        store.finish_batch()
    return {"vid": vid, "meta": meta, "db": reg, "warning": warning}


# --------------------------------------------------------------------------- patient database
@router.get("/db")
def db_patients():
    """The patient list for Home. Returns ``locked=True`` (not 500) if patients.csv is held open."""
    try:
        return {"patients": db.patients(), "locked": False, "csv_path": db.csv_path()}
    except db.DbLocked as e:
        return {"patients": [], "locked": True, "message": str(e), "csv_path": db.csv_path()}


@router.get("/db.xlsx")
def db_xlsx():
    """Download the patient database as an Excel workbook."""
    try:
        rows = db.all_records()
    except db.DbLocked as e:
        raise HTTPException(status_code=423, detail=str(e))
    if not rows:
        raise HTTPException(status_code=404, detail="the patient database is empty")
    try:
        data = xlsx.workbook_bytes(rows)
    except RuntimeError as e:                                 # openpyxl missing
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=data,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": "attachment; filename=ga_clinic.xlsx",
                             "Cache-Control": "no-store"})


@router.get("/db.csv")
def db_csv():
    """Download the raw patients.csv."""
    try:
        data = db.csv_bytes()
    except db.DbLocked as e:
        raise HTTPException(status_code=423, detail=str(e))
    if data is None:
        raise HTTPException(status_code=404, detail="no patient database yet")
    return Response(content=data, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=ga_clinic.csv",
                             "Cache-Control": "no-store"})


@router.get("/db/{patient_id}")
def db_patient(patient_id: str):
    """One patient's scans (date/eye/GA), sorted chronologically."""
    try:
        p = db.patient(patient_id)
    except db.DbLocked as e:
        raise HTTPException(status_code=423, detail=str(e))
    if p is None:
        raise HTTPException(status_code=404, detail="no such patient")
    return p


@router.delete("/db")
def db_delete(record_id: str, delete_staged: bool = False):
    """Remove one tracked scan; optionally remove an unreferenced clinic-staged E2E copy."""
    try:
        result = db.delete(record_id, delete_staged=delete_staged)
    except db.DbLocked as e:
        raise HTTPException(status_code=423, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="no such record")
    return result


# --------------------------------------------------------------------------- viewer panels (per vid)
@router.get("/scan/{vid}/meta")
def meta(vid: str, store=Depends(get_store)):
    return _src(store, vid).meta_public()


@router.get("/scan/{vid}/bscan/{idx}.png")
def bscan(vid: str, idx: int, store=Depends(get_store)):
    return _png(_src(store, vid).bscan_png(idx))


@router.get("/scan/{vid}/localizer.png")
def localizer(vid: str, store=Depends(get_store)):
    return _png(_src(store, vid).localizer_png())


@router.get("/scan/{vid}/projection.png")
def projection(vid: str, store=Depends(get_store)):
    return _png(_src(store, vid).projection_png())


@router.get("/scan/{vid}/ga_overlay.png")
def ga_overlay(vid: str, store=Depends(get_store)):
    return _png(_src(store, vid).ga_overlay_png())


@router.get("/scan/{vid}/loc_lines")
def loc_lines(vid: str, store=Depends(get_store)):
    return {"lines": _src(store, vid).loc_lines()}


@router.get("/scan/{vid}/bm")
def bm(vid: str, store=Depends(get_store)):
    s = _src(store, vid)
    m = s.meta_json()
    return {"bm": s.bm_rows(), "axial_um_per_px": m.get("axial_um_per_px"),
            "slab_um": m.get("slab_um"), "field_invalid": s.field_invalid_runs()}


@router.get("/scan/{vid}/ga_native")
def ga_native(vid: str, store=Depends(get_store)):
    return {"intervals": _src(store, vid).ga_intervals()}


@router.get("/scan/{vid}/dashboard")
def dashboard(vid: str, store=Depends(get_store)):
    return {"oac_area_mm2": _src(store, vid).meta_json().get("oac_area_mm2")}
