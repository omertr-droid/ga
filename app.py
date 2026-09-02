import os, tempfile, uuid, time, threading
from collections import OrderedDict
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Request
from fastapi.responses import Response

# Imports from your GA repo (repo root is on PYTHONPATH in the Dockerfile)
from reader.core import e2e_source, layers
from viewer.core import viewmodel, ga_native

API_KEY = os.environ.get("GA_API_KEY", "")   # shared secret; set on Render
JOBS = {}
JOB_LOCK = threading.Lock()
MAX_JOBS = 6
AXIAL_UM_PER_PX = float(viewmodel.AXIAL_UM_PER_PX)

app = FastAPI(title="GA Algorithm API")

def _check_key(x_api_key: str):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

def _put(job_id, vm, ov, meta):
    with JOB_LOCK:
        JOBS[job_id] = {"vm": vm, "ov": ov, "meta": meta, "ts": time.time()}
        while len(JOBS) > MAX_JOBS:
            JOBS.pop(next(iter(JOBS)))

def _get(job_id):
    with JOB_LOCK:
        return JOBS.get(job_id)

def _pick_volume(raw):
    """Pick the 6×6 / 97-line OCT volume = the one with the most B-scans."""
    best_idx, best_n = None, -1
    for i in range(32):
        try:
            ov = e2e_source.load_volume(raw, i)
        except Exception:
            continue
        nb = int(getattr(ov, "n_bscans", 0) or 0)
        if nb > best_n:
            best_n, best_idx = nb, i
    if best_idx is None:
        raise HTTPException(422, "No OCT volume found in this E2E")
    return best_idx, e2e_source.load_volume(raw, best_idx)

def _meta(ov):
    def g(*names, default=None):
        for n in names:
            v = getattr(ov, n, None)
            if v is not None:
                return v
        return default
    eye = g("laterality", "eye")
    if eye is not None:
        s = str(eye).strip().upper()
        eye = "OD" if s in ("R","OD","RIGHT","82") else "OS" if s in ("L","OS","LEFT","76") else eye
    return {
        "eye": eye,
        "subject": g("subject", "patient_id", "patient"),
        "acq_date": g("acq_date", "acquisition_date", "date", "study_date"),
    }

@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...), x_api_key: str = Header(default="")):
    _check_key(x_api_key)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    with tempfile.NamedTemporaryFile(suffix=".E2E", delete=False) as tf:
        tf.write(data); tmp = tf.name
    try:
        raw = e2e_source.open_e2e(tmp)
        idx, ov = _pick_volume(raw)
        ilm, bm = layers.effective_surfaces(ov, None)
        vm = viewmodel.compute(raw, ov, ilm, bm)
        meta = _meta(ov)
        job_id = uuid.uuid4().hex
        _put(job_id, vm, ov, meta)

        ilm_rows = np.asarray(vm["ilm"]).tolist()
        bm_rows = np.asarray(vm["bm"]).tolist()
        ga_intervals = [ga_native.intervals(row) for row in np.asarray(vm["ga_native"], bool)]
        base = str(request.base_url).rstrip("/")
        return {
            "eye": meta["eye"],
            "subject": meta["subject"],
            "acq_date": meta["acq_date"],
            "oac_area_mm2": float(vm["oac_area_mm2"]),
            "n_bscans": int(ov.n_bscans),
            "fov_mm": [float(x) for x in list(getattr(ov, "fov_mm", [6, 6]))[:2]],
            "axial_um_per_px": AXIAL_UM_PER_PX,
            "slab_um": [float(x) for x in np.asarray(vm["slab_um"]).tolist()],
            "projection_png_url": f"{base}/result/{job_id}/projection.png",
            "ga_overlay_png_url": f"{base}/result/{job_id}/ga_overlay.png",
            "bscan_png_url_template": f"{base}/result/{job_id}/bscan/{{i}}.png",
            "ilm_rows": ilm_rows,
            "bm_rows": bm_rows,
            "ga_intervals": ga_intervals,
            "enface_flip": bool(vm["enface_flip"]),
        }
    finally:
        try: os.unlink(tmp)
        except Exception: pass

@app.get("/result/{job_id}/projection.png")
def result_projection(job_id: str, x_api_key: str = Header(default="")):
    _check_key(x_api_key)
    job = _get(job_id)
    if not job: raise HTTPException(404, "Result expired or not found")
    return Response(content=job["vm"]["projection_png"], media_type="image/png")

@app.get("/result/{job_id}/ga_overlay.png")
def result_overlay(job_id: str, x_api_key: str = Header(default="")):
    _check_key(x_api_key)
    job = _get(job_id)
    if not job: raise HTTPException(404, "Result expired or not found")
    return Response(content=job["vm"]["ga_overlay_png"], media_type="image/png")

@app.get("/result/{job_id}/bscan/{i}.png")
def result_bscan(job_id: str, i: int, x_api_key: str = Header(default="")):
    _check_key(x_api_key)
    job = _get(job_id)
    if not job: raise HTTPException(404, "Result expired or not found")
    vol = np.asarray(job["vm"]["vol_u8"])
    i = max(0, min(vol.shape[0] - 1, int(i)))
    ok, buf = cv2.imencode(".png", vol[i])
    if not ok: raise HTTPException(500, "PNG encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")

@app.get("/health")
def health():
    return {"ok": True}
