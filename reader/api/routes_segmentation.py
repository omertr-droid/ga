"""Segment tab: labeling RUNS -> en-face GA footprint + area (mm2).

A *run* is one labeling attempt (a threshold/manual/prompt pass) holding per-B-scan binary masks
(PngMaskStore). Footprint + area are computed LIVE from a run's masks (core.footprint), so trying a
different prompt/threshold is just another run to create and compare. Phase-1 sources: a threshold
baseline (POST /seed) + manual brush (PUT /mask). Phase 2 adds MedSAM3 (the /segment/endpoint URL
points the reader at the Colab service).
"""
import json
import os
import re
import threading

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from reader.core import footprint as fp, oac_ga, render, seg_classes as sc, layers as core_layers
from reader.core.segmenter_client import Sam2Client, Sam3Client, SegmenterError
from paths import REPO_ROOT
from . import deps
from .deps import get_mask_store, get_store, get_layer_store
from .schemas import (EndpointIn, LabelBscanIn, LabelVolumeIn, MarkFreeIn, RunCreateIn, SamBscanIn,
                      SamEnfaceIn, SamPropagateIn, SeedIn, StatusIn)

router = APIRouter()
_KEY_RE = re.compile(r"NHAMD-003-\d+-V\d+", re.I)   # cohort subject-visit key embedded in the E2E path

_JOBS = {}                                          # volume_id -> {running, done, total, run, concept, error}
_JOBLOCK = threading.Lock()


def _ov(store, volume_id):
    try:
        return store.get_volume(volume_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Open the E2E first")


def _eff_bm(ov, layer_store):
    """Effective (corrected) BM surface (n, W): folds in BM-tab corrections so wedges/cues sit on the BM
    the user validated, not the stale device line. Falls back to ov.bm when no corrections exist."""
    return core_layers.effective_surfaces(ov, layer_store)[1]


def _png(arr) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _decode_mask(raw, H, W):
    """PNG bytes -> bool (H,W) mask: RGBA marks painted by alpha>0, else any channel >127.
    NEAREST-resized to (H,W) when the model returns a different size."""
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 3:
        m = a[..., 3] > 0 if a.shape[2] == 4 else a.max(axis=2) > 127
    else:
        m = a > 127
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return m


def _model_run_id(concept):
    """A stable run id per concept, so different prompts become separate, comparable runs."""
    slug = re.sub(r"[^a-z0-9]+", "-", (concept or "model").lower()).strip("-")[:24] or "model"
    return "model-" + slug


def _cohort_ref(store, ov):
    """Best-effort link from the opened E2E to its cohort eye (advRPE reference). Returns paths +
    the advRPE area, or None when the E2E doesn't map to a cohort subject-visit. Used only as a
    comparison VIEW — the advRPE (PLEX) frame is NOT pixel-registered to our projection."""
    raw = store.get_raw(ov.eid)
    m = _KEY_RE.search(getattr(raw, "path", "") or "")
    if not m:
        return None
    key = m.group(0).upper()
    edir = os.path.join(REPO_ROOT, "cohort", key, ov.eye)
    if not os.path.isdir(edir):
        return None
    area = None
    try:
        with open(os.path.join(REPO_ROOT, "cohort", key, "meta.json")) as f:
            area = json.load(f).get("eyes", {}).get(ov.eye, {}).get("advRPE_area_mm2")
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return {"key": key, "area": (float(area) if area not in (None, "") else None),
            "outline": os.path.join(edir, "advrpe_ga_outline.png"),
            "faf": os.path.join(edir, "spectralis_baf.png"),
            "ga_mask": os.path.join(edir, "ga_mask.png"),
            "subrpe": os.path.join(edir, "advrpe_subrpe_enface.png")}


_PLEX_REG = {}          # volume_id -> registered galabel (512 bool), cached (the lock is the slow part)


def _plex_registered(ov, ref):
    """Two-map GA-driven lock (src/register_projection method, run live on the reader's volume): register
    a sub-RPE shadowgram -> advRPE SubRPE, then warp the advRPE GA mask into the projection's 6×6 frame.
    Returns a 512×512 bool label (the to_6mm frame, center-equivalent to the reader's to_enface frame),
    or None. GA-driven => a visual/label aid, NOT an independent registration."""
    if ov.volume_id in _PLEX_REG:
        return _PLEX_REG[ov.volume_id]
    import m3_projections as mp
    import qcviz as qv
    import register_qc as rq
    adv = cv2.imread(ref["subrpe"], cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(ref["ga_mask"], cv2.IMREAD_GRAYSCALE)
    if adv is None or m is None:
        return None
    adv6 = cv2.resize(adv, (512, 512))
    mask6 = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
    shadow = mp.to_6mm(mp.destripe2d(mp.band(ov.vol, ov.bm, 10, 340, "mean")), list(ov.fov_mm),
                       getattr(ov, "enface_flip", True))     # same frame as to_enface, or PLEX misregisters
    shadow6 = cv2.resize(qv.norm8(shadow), (512, 512))
    try:
        reg = rq.register(shadow6, adv6)                       # shadowgram -> advRPE
    except Exception:
        return None
    flipf = np.fliplr if reg["flip"] else (lambda a: a)
    minv = cv2.invertAffineTransform(reg["M"])                 # advRPE GA -> our frame (inverse)
    gl = flipf(rq.warp((mask6 * 255).astype(np.uint8), minv, cv2.INTER_NEAREST)) > 127
    _PLEX_REG[ov.volume_id] = gl
    return gl


def _center_fit(src, out):
    """Center-place a (H,W) bool array into an out×out bool frame (crop if larger, pad if smaller).
    Both are at ENFACE_MMPP, so this scale-matches + center-aligns the advRPE 6×6 to our projection frame."""
    H, W = src.shape
    dst = np.zeros((out, out), bool)
    oy, ox = (out - H) // 2, (out - W) // 2
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    h, w = min(H - sy0, out - dy0), min(W - sx0, out - dx0)
    if h > 0 and w > 0:
        dst[dy0:dy0 + h, dx0:dx0 + w] = src[sy0:sy0 + h, sx0:sx0 + w]
    return dst


# ------------------------------------------------------------------ MedSAM3 endpoint (Phase 2)
@router.get("/segment/endpoint")
def get_endpoint():
    return {"url": deps.get_seg_endpoint()}


@router.put("/segment/endpoint")
def put_endpoint(body: EndpointIn):
    url = (body.url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url                          # tolerate a pasted URL missing its scheme
    deps.set_seg_endpoint(url)
    return {"url": deps.get_seg_endpoint()}


# ------------------------------------------------------------------ runs
def _run_invert(ms, ov, run):
    return bool(ms.list_runs(ov.eid, ov.eye)["runs"].get(run, {}).get("invert"))


def _ensure_class_run(ms, ov, base, cls, source=None, make_active=False):
    """Idempotently create the class-run `run_id(base, cls)` with its class/invert metadata, returning
    its id. The DEFAULT class reuses the base id (so a legacy 'gold' run is already the wedge class)."""
    rid = sc.run_id(base, cls)
    if not ms.run_exists(ov.eid, ov.eye, rid):
        c = sc.cfg(cls)
        ms.create_run(ov.eid, ov.eye, rid,
                      {"label": (base if cls == sc.DEFAULT_CLASS else f"{base}:{c['name']}"),
                       "source": source or c["seed"], "concept": c["concept"],
                       "class": cls, "invert": c["invert_default"]}, make_active=make_active)
    return rid


def _run_summary(ov, ms, run_id, meta, min_diam_um):
    idxs = ms.mask_indices(ov.eid, ov.eye, run_id)
    invert = bool(meta.get("invert"))
    area = fp.run_footprint(ov, ms, run_id, min_diam_um, invert)[1] if idxs else 0.0
    base, cls = sc.base_class(run_id, meta)
    out = {"id": run_id, "n_masks": len(idxs), "area_mm2": round(area, 4), "invert": invert,
           "base": base, "class": cls}
    out.update({k: meta.get(k) for k in ("label", "source", "concept", "threshold", "created")})
    return out


@router.get("/volumes/{volume_id}/segment/runs")
def list_runs(volume_id: str, min_diam_um: float = Query(default=250.0),
              store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    data = ms.list_runs(ov.eid, ov.eye)
    runs = [_run_summary(ov, ms, rid, meta, min_diam_um) for rid, meta in data["runs"].items()]
    runs.sort(key=lambda r: r.get("created") or "")
    active = data["active"]
    active_class = sc.base_class(active, data["runs"].get(active, {}))[1] if active else None
    return {"runs": runs, "active": active, "active_class": active_class, "min_diam_um": min_diam_um}


@router.post("/volumes/{volume_id}/segment/runs")
def create_run(volume_id: str, body: RunCreateIn,
               store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    rid = (body.run if (body.run and not ms.run_exists(ov.eid, ov.eye, body.run))
           else ms.new_run_id(ov.eid, ov.eye, prefix=(body.source or "manual")))
    m = {"label": body.label or rid, "source": body.source, "concept": body.concept}
    if body.cls:
        m["class"] = body.cls
        m["invert"] = body.invert if body.invert is not None else sc.cfg(body.cls)["invert_default"]
    elif body.invert is not None:
        m["invert"] = bool(body.invert)
    meta = ms.create_run(ov.eid, ov.eye, rid, m)
    return {"id": rid, **{k: meta.get(k) for k in ("label", "source", "concept", "class", "invert", "created")}}


@router.delete("/volumes/{volume_id}/segment/runs/{run}")
def delete_run(volume_id: str, run: str, store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    ms.delete_run(ov.eid, ov.eye, run)
    return {"ok": True, "active": ms.list_runs(ov.eid, ov.eye)["active"]}


@router.put("/volumes/{volume_id}/segment/runs/{run}/active")
def set_active(volume_id: str, run: str, store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    if not ms.run_exists(ov.eid, ov.eye, run):
        raise HTTPException(status_code=404, detail=f"no run {run!r}")
    ms.set_active(ov.eid, ov.eye, run)
    return {"ok": True, "active": run}


@router.put("/volumes/{volume_id}/segment/runs/{run}/invert")
def set_invert(volume_id: str, run: str, value: bool = Query(...),
               store=Depends(get_store), ms=Depends(get_mask_store)):
    """Toggle invert (RPE->loss): GA = interior gaps of the segmented band. Live; recomputes the area."""
    ov = _ov(store, volume_id)
    if not ms.run_exists(ov.eid, ov.eye, run):
        raise HTTPException(status_code=404, detail=f"no run {run!r}")
    ms.update_run(ov.eid, ov.eye, run, {"invert": bool(value)})
    return {"ok": True, "run": run, "invert": bool(value)}


# ------------------------------------------------------------------ threshold baseline (seed)
@router.post("/volumes/{volume_id}/segment/seed")
def seed(volume_id: str, body: SeedIn, store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    run = body.run or "threshold"
    lo = body.slab_lo if body.slab_lo is not None else fp.proj.RAW_SLAB[0]
    hi = body.slab_hi if body.slab_hi is not None else fp.proj.RAW_SLAB[1]
    meta = {"label": "threshold baseline", "source": "threshold", "threshold": body.threshold,
            "concept": f"hyper>{body.threshold:g}"}
    if ms.run_exists(ov.eid, ov.eye, run):
        ms.clear_masks(ov.eid, ov.eye, run)
        ms.update_run(ov.eid, ov.eye, run, meta)
    else:
        ms.create_run(ov.eid, ov.eye, run, meta)
    n = 0
    for i, mask in fp.threshold_masks(ov, body.threshold, lo, hi):
        ms.put_mask(ov.eid, ov.eye, run, i, mask)
        n += 1
    ms.set_active(ov.eid, ov.eye, run)
    return {"id": run, "n_masks": n}


@router.post("/volumes/{volume_id}/segment/seed_rpe")
def seed_rpe(volume_id: str, base: str = Query("gold"),
             present_thr: float = Query(fp.RPE_PRESENT_THR),
             store=Depends(get_store), ms=Depends(get_mask_store)):
    """Classical RPE-PRESENT seed (no Colab): peak-track the RPE surface (drusen-aware, follows it up over
    drusen) and paint the intact band; GA = its inverted interior gap. Faded/ambiguous columns make the
    B-scan 'borderline' so the genuinely-uncertain drusen->GA transition is reviewed, not guessed."""
    ov = _ov(store, volume_id)
    run = sc.run_id(base, "rpe")
    meta = {"label": "RPE (classical peak-track)", "source": "classical", "class": "rpe",
            "concept": "RPE band", "invert": True}
    if ms.run_exists(ov.eid, ov.eye, run):
        ms.clear_masks(ov.eid, ov.eye, run)
        ms.update_run(ov.eid, ov.eye, run, meta)
    else:
        ms.create_run(ov.eid, ov.eye, run, meta)
    counts = {"ga": 0, "ga_free": 0, "borderline": 0}
    n_painted = 0
    for i, mask, prom in fp.rpe_present_masks(ov, present_thr):
        if mask.any():
            ms.put_mask(ov.eid, ov.eye, run, i, mask)
            n_painted += 1
        st = fp.rpe_status_for(prom, present_thr)
        ms.set_status(ov.eid, ov.eye, run, i, state=st)
        counts[st] = counts.get(st, 0) + 1
    ms.set_active(ov.eid, ov.eye, run)
    return {"id": run, "n_masks": n_painted, "status_counts": counts}


# ------------------------------------------------------------------ masks (per run, per B-scan)
@router.get("/volumes/{volume_id}/segment/mask.png")
def get_mask(volume_id: str, run: str = Query(...), bscan: int = Query(...),
             cls: str = Query("wedge"), store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    m = ms.get_mask(ov.eid, ov.eye, run, bscan)
    if m is None:
        raise HTTPException(status_code=404, detail="no mask")
    bgra = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
    bgra[m] = sc.bgra(cls)                # class color (alpha marks painted; CSS dims for display)
    return Response(_png(bgra), media_type="image/png")


@router.put("/volumes/{volume_id}/segment/mask")
async def put_mask(volume_id: str, request: Request, run: str = Query(...), bscan: int = Query(...),
                   store=Depends(get_store), ms=Depends(get_mask_store)):
    """Store a B-scan mask. Body = a PNG (the brushed mask canvas): a 4-channel PNG marks painted
    pixels by alpha>0; otherwise any channel >127. Resized NEAREST to (H,W) if needed."""
    ov = _ov(store, volume_id)
    if not (0 <= bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="bscan out of range")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty body (expected a PNG)")
    mask = _decode_mask(raw, ov.H, ov.W)
    if mask is None:
        raise HTTPException(status_code=400, detail="could not decode PNG")
    if not ms.run_exists(ov.eid, ov.eye, run):
        ms.create_run(ov.eid, ov.eye, run, {"label": run, "source": "manual"})
    ms.put_mask(ov.eid, ov.eye, run, bscan, mask)
    if mask.any():
        ms.set_status(ov.eid, ov.eye, run, bscan, state="ga")    # painted ⇒ GA (overrides prior ga_free)
    else:
        ms.clear_status(ov.eid, ov.eye, run, bscan)              # erased empty ⇒ back to todo
    return {"ok": True, "run": run, "bscan": bscan, "painted": int(mask.sum())}


@router.delete("/volumes/{volume_id}/segment/mask")
def delete_mask(volume_id: str, run: str = Query(...), bscan: int = Query(...),
                store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    ms.delete_mask(ov.eid, ov.eye, run, bscan)
    ms.clear_status(ov.eid, ov.eye, run, bscan)                  # cleared ⇒ back to todo
    return {"ok": True}


# ------------------------------------------------------------------ footprint + area
@router.get("/volumes/{volume_id}/segment/area")
def area(volume_id: str, run: str = Query(...), min_diam_um: float = Query(default=250.0),
         store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    idxs = ms.mask_indices(ov.eid, ov.eye, run)
    inv = _run_invert(ms, ov, run)
    a = fp.run_footprint(ov, ms, run, min_diam_um, inv)[1] if idxs else 0.0
    return {"run": run, "area_mm2": round(a, 4), "n_masks": len(idxs), "invert": inv}


@router.get("/volumes/{volume_id}/segment/footprint.png")
def footprint_png(volume_id: str, run: str = Query(...), min_diam_um: float = Query(default=250.0),
                  store=Depends(get_store), ms=Depends(get_mask_store)):
    """The run's cRORA footprint as a translucent yellow RGBA overlay in the en-face frame (same size
    as projection.png, so the client draws it 1:1 over the projection)."""
    ov = _ov(store, volume_id)
    idxs = ms.mask_indices(ov.eid, ov.eye, run)
    if idxs:
        keep = fp.run_footprint(ov, ms, run, min_diam_um, _run_invert(ms, ov, run))[0]
    else:
        out = max(64, int(round(max(ov.fov_mm) / fp.MMPP)))
        keep = np.zeros((out, out), bool)
    bgra = np.zeros((keep.shape[0], keep.shape[1], 4), np.uint8)
    bgra[keep] = (0, 255, 255, 150)          # cv2 BGRA == yellow, ~59% alpha
    return Response(_png(bgra), media_type="image/png")


# --------------------------------------------------- combined cRORA area = AND over the base's classes
def _combined_flags(ov, ms, base):
    """AND the per-class column-flag maps of a base's class-runs (each with its own invert):
    wedge -> painted columns; rpe -> RPE-loss gaps. Returns (n,W) bool, or None if no class has masks."""
    flags = None
    for cls in sc.SHIP:
        rid = sc.run_id(base, cls)
        if not ms.run_exists(ov.eid, ov.eye, rid) or not ms.mask_indices(ov.eid, ov.eye, rid):
            continue
        f = fp.native_flags(ov, ms, rid, invert=_run_invert(ms, ov, rid)) > 0
        flags = f if flags is None else (flags & f)
    return flags


@router.get("/volumes/{volume_id}/segment/combined_area")
def combined_area(volume_id: str, base: str = Query("gold"), min_diam_um: float = Query(default=250.0),
                  store=Depends(get_store), ms=Depends(get_mask_store)):
    """cRORA GA area (mm²) = (hypertransmission wedge) ∧ (RPE-loss) over the base's class-runs."""
    ov = _ov(store, volume_id)
    flags = _combined_flags(ov, ms, base)
    if flags is None or not flags.any():
        return {"base": base, "area_mm2": 0.0, "available": bool(flags is not None)}
    a = fp.footprint_from_flags(flags.astype(np.float32), ov.fov_mm, min_diam_um)[1]
    return {"base": base, "area_mm2": round(float(a), 4), "available": True}


@router.get("/volumes/{volume_id}/segment/combined_footprint.png")
def combined_footprint_png(volume_id: str, base: str = Query("gold"),
                           min_diam_um: float = Query(default=250.0),
                           store=Depends(get_store), ms=Depends(get_mask_store)):
    """The combined cRORA GA footprint as a translucent GREEN overlay in the en-face frame."""
    ov = _ov(store, volume_id)
    flags = _combined_flags(ov, ms, base)
    out = max(64, int(round(max(ov.fov_mm) / fp.MMPP)))
    if flags is None or not flags.any():
        keep = np.zeros((out, out), bool)
    else:
        keep = fp.footprint_from_flags(flags.astype(np.float32), ov.fov_mm, min_diam_um)[0]
    bgra = np.zeros((keep.shape[0], keep.shape[1], 4), np.uint8)
    bgra[keep] = (0, 210, 0, 160)            # cv2 BGRA == green
    return Response(_png(bgra), media_type="image/png")


# ------------------------------------------------------------------ advRPE reference (best-effort)
@router.get("/volumes/{volume_id}/segment/reference")
def reference(volume_id: str, store=Depends(get_store)):
    ov = _ov(store, volume_id)
    ref = _cohort_ref(store, ov)
    if not ref:
        return {"found": False}
    return {"found": True, "key": ref["key"], "advRPE_area_mm2": ref["area"],
            "has_outline": os.path.exists(ref["outline"]), "has_faf": os.path.exists(ref["faf"]),
            "has_plex_mask": os.path.exists(ref["ga_mask"])}


@router.get("/volumes/{volume_id}/segment/reference/{kind}.png")
def reference_png(volume_id: str, kind: str, store=Depends(get_store)):
    ov = _ov(store, volume_id)
    ref = _cohort_ref(store, ov)
    path = ref.get({"outline": "outline", "faf": "faf"}.get(kind)) if ref else None
    if not (path and os.path.exists(path)):
        raise HTTPException(status_code=404, detail="no reference image")
    with open(path, "rb") as f:
        return Response(f.read(), media_type="image/png")


@router.get("/volumes/{volume_id}/segment/plex_mask.png")
def plex_mask_png(volume_id: str, registered: bool = Query(default=False), store=Depends(get_store)):
    """The advRPE (PLEX) GA as a cyan CONTOUR overlay in the projection frame.
    registered=False: center-aligned + scale-matched 6×6 (approximate — not a per-eye lock).
    registered=True : the two-map GA-driven lock (src/register_projection method, run live) — properly
    aligned, but GA-driven so it's a visual/label aid, not an independent ground truth (can mis-lock on
    faint-vessel eyes)."""
    ov = _ov(store, volume_id)
    ref = _cohort_ref(store, ov)
    if not (ref and os.path.exists(ref["ga_mask"])):
        raise HTTPException(status_code=404, detail="no PLEX GA mask for this eye")
    out = max(64, int(round(max(ov.fov_mm) / fp.MMPP)))
    if registered:
        gl = _plex_registered(ov, ref)
        if gl is None:
            raise HTTPException(status_code=502, detail="registration failed (no SubRPE / faint vessels)")
        fit = _center_fit(gl, out).astype(np.uint8)
    else:
        m = cv2.imread(ref["ga_mask"], cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="could not read PLEX GA mask")
        fit = _center_fit(m > 127, out).astype(np.uint8)
    bgra = np.zeros((out, out, 4), np.uint8)
    cnts, _ = cv2.findContours(fit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgra, cnts, -1, (255, 255, 0, 255), 2)   # cv2 BGRA == cyan outline
    return Response(_png(bgra), media_type="image/png")


# ------------------------------------------------------------------ MedSAM3 labeling (Phase 2, Colab)
def _require_endpoint():
    url = deps.get_seg_endpoint()
    if not url:
        raise HTTPException(status_code=503, detail="set the MedSAM3 endpoint (Colab URL) first")
    return url


@router.get("/segment/health")
def seg_health():
    url = deps.get_seg_endpoint()
    if not url:
        return {"reachable": False, "reason": "no endpoint set"}
    try:
        return {"reachable": True, "url": url, "info": Sam3Client(url).health()}
    except SegmenterError as e:
        return {"reachable": False, "url": url, "reason": str(e)}


@router.post("/volumes/{volume_id}/segment/label_bscan")
def label_bscan(volume_id: str, body: LabelBscanIn,
                store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    url = _require_endpoint()
    if not (0 <= body.bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="bscan out of range")
    run = body.run or _model_run_id(body.concept)
    if not ms.run_exists(ov.eid, ov.eye, run):
        ms.create_run(ov.eid, ov.eye, run,
                      {"label": body.concept or run, "source": "model", "concept": body.concept})
    try:
        mask_png = Sam3Client(url).segment(render.bscan_png(ov, body.bscan), body.concept, body.threshold)
    except SegmenterError as e:
        raise HTTPException(status_code=502, detail=str(e))
    mask = _decode_mask(mask_png, ov.H, ov.W)
    if mask is None:
        raise HTTPException(status_code=502, detail="model returned an undecodable mask")
    ms.put_mask(ov.eid, ov.eye, run, body.bscan, mask)
    ms.set_active(ov.eid, ov.eye, run)
    return {"ok": True, "run": run, "bscan": body.bscan, "painted": int(mask.sum())}


def _run_batch(volume_id, ov, ms, url, run, concept, threshold, idxs):
    client = Sam3Client(url)
    try:
        for i in idxs:
            with _JOBLOCK:
                if not _JOBS.get(volume_id, {}).get("running"):
                    break                               # cancelled / superseded
            try:
                mask = _decode_mask(client.segment(render.bscan_png(ov, i), concept, threshold), ov.H, ov.W)
                if mask is not None:
                    ms.put_mask(ov.eid, ov.eye, run, i, mask)
            except Exception as e:                      # ANY failure (bad URL, model error) -> stop + surface
                with _JOBLOCK:
                    _JOBS[volume_id]["error"] = str(e)
                break
            with _JOBLOCK:
                _JOBS[volume_id]["done"] += 1
    finally:
        with _JOBLOCK:
            if volume_id in _JOBS:                      # ALWAYS clear running, even on an unexpected exception
                _JOBS[volume_id]["running"] = False


@router.post("/volumes/{volume_id}/segment/label_volume")
def label_volume(volume_id: str, body: LabelVolumeIn,
                 store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    url = _require_endpoint()
    with _JOBLOCK:
        if _JOBS.get(volume_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="a labeling run is already in progress")
    rng = body.bscan_range or [0, ov.n_bscans - 1]
    idxs = list(range(max(0, rng[0]), min(ov.n_bscans, rng[1] + 1)))
    run = body.run or _model_run_id(body.concept)
    meta = {"label": body.concept or run, "source": "model", "concept": body.concept,
            "threshold": body.threshold}
    if body.cls:                                  # class-run (e.g. rpe): keep class + invert across re-run
        meta["class"] = body.cls
    if body.invert is not None:
        meta["invert"] = bool(body.invert)
    if ms.run_exists(ov.eid, ov.eye, run):
        ms.clear_masks(ov.eid, ov.eye, run)
        ms.update_run(ov.eid, ov.eye, run, meta)
    else:
        ms.create_run(ov.eid, ov.eye, run, meta)
    ms.set_active(ov.eid, ov.eye, run)
    with _JOBLOCK:
        _JOBS[volume_id] = {"running": True, "done": 0, "total": len(idxs), "run": run,
                            "concept": body.concept, "error": None}
    threading.Thread(target=_run_batch,
                     args=(volume_id, ov, ms, url, run, body.concept, body.threshold, idxs),
                     daemon=True).start()
    return {"ok": True, "run": run, "total": len(idxs)}


@router.get("/volumes/{volume_id}/segment/label_job")
def label_job(volume_id: str, store=Depends(get_store)):
    _ov(store, volume_id)
    with _JOBLOCK:
        j = _JOBS.get(volume_id)
        return dict(j) if j else {"running": False, "done": 0, "total": 0, "run": None}


# ------------------------------------------------------------------ annotation studio: status + guide
_PLEX_NAT = {}          # volume_id -> native (n,W) bool PLEX-GA guide (from the registered lock), cached


def _center_extract(a, fh, fw):
    """Inverse of _center_fit: pull the centered (fh,fw) field out of a square en-face frame."""
    H, W = a.shape
    out = np.zeros((fh, fw), a.dtype)
    oy, ox = (fh - H) // 2, (fw - W) // 2
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    h, w = min(H - sy0, fh - dy0), min(W - sx0, fw - dx0)
    if h > 0 and w > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = a[sy0:sy0 + h, sx0:sx0 + w]
    return out


def _plex_native(ov, ref):
    """Registered PLEX GA mapped back to native (n_bscans, W) bool — the per-B-scan column guide. Inverse
    of to_enface (un-resample + un-flip); approximate (a guide for the annotator, not a label)."""
    if ov.volume_id in _PLEX_NAT:
        return _PLEX_NAT[ov.volume_id]
    gl = _plex_registered(ov, ref)
    if gl is None:
        return None
    n, _, W = ov.vol.shape
    fh = max(1, int(round(ov.fov_mm[1] / fp.MMPP)))
    fw = max(1, int(round(ov.fov_mm[0] / fp.MMPP)))
    field = _center_extract(gl.astype(np.uint8), fh, fw)
    nat = cv2.resize(field, (W, n), interpolation=cv2.INTER_NEAREST)
    native = (nat[::-1] if getattr(ov, "enface_flip", True) else nat) > 0    # match to_enface's flip
    _PLEX_NAT[ov.volume_id] = native
    return native


def _intervals(row):
    """[start, end) column intervals where a boolean row is True."""
    out, x, n = [], 0, len(row)
    while x < n:
        if row[x]:
            s = x
            while x < n and row[x]:
                x += 1
            out.append([int(s), int(x)])
        else:
            x += 1
    return out


@router.get("/volumes/{volume_id}/segment/status")
def get_status(volume_id: str, run: str = Query(...), store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    d = ms.derived_status(ov.eid, ov.eye, run, ov.n_bscans)
    counts = {}
    for v in d.values():
        counts[v["state"]] = counts.get(v["state"], 0) + 1
    return {"run": run, "n_bscans": ov.n_bscans, "counts": counts,
            "reviewed": sum(1 for v in d.values() if v["reviewed"]),
            "states": {str(k): v["state"] for k, v in d.items()},
            "reviewed_bscans": [k for k, v in d.items() if v["reviewed"]]}


@router.put("/volumes/{volume_id}/segment/status")
def put_status(volume_id: str, body: StatusIn, store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    if not (0 <= body.bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="bscan out of range")
    if body.state == "todo":
        ms.clear_status(ov.eid, ov.eye, body.run, body.bscan)
    elif body.state is not None or body.reviewed is not None:
        if body.state == "ga_free":
            ms.delete_mask(ov.eid, ov.eye, body.run, body.bscan)     # an explicit negative has no mask
        ms.set_status(ov.eid, ov.eye, body.run, body.bscan, state=body.state, reviewed=body.reviewed)
    return {"ok": True, "run": body.run, "bscan": body.bscan}


@router.post("/volumes/{volume_id}/segment/mark_free")
def mark_free(volume_id: str, body: MarkFreeIn, store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    if body.bscan is not None:
        idxs = [body.bscan]
    elif body.bscan_range:
        lo, hi = body.bscan_range
        idxs = list(range(max(0, lo), min(ov.n_bscans, hi + 1)))
    else:                                                            # default: all remaining 'todo'
        d = ms.derived_status(ov.eid, ov.eye, body.run, ov.n_bscans)
        idxs = [i for i, v in d.items() if v["state"] == "todo"]
    for i in idxs:
        ms.delete_mask(ov.eid, ov.eye, body.run, i)
        ms.set_status(ov.eid, ov.eye, body.run, i, state="ga_free")
    return {"ok": True, "marked": len(idxs)}


@router.get("/volumes/{volume_id}/segment/plex_guide")
def plex_guide(volume_id: str, bscan: int = Query(...), store=Depends(get_store)):
    """GA column intervals the registered PLEX reference marks on this B-scan (annotator guide)."""
    ov = _ov(store, volume_id)
    ref = _cohort_ref(store, ov)
    if not (ref and os.path.exists(ref["ga_mask"])):
        return {"bscan": bscan, "cols": [], "available": False}
    nat = _plex_native(ov, ref)
    if nat is None or not (0 <= bscan < nat.shape[0]):
        return {"bscan": bscan, "cols": [], "available": False}
    return {"bscan": bscan, "cols": _intervals(nat[bscan]), "available": True}


# ------------------------------------------------------------------ SAM2 box/point assist (Phase C)
def _active_or_gold(ms, ov, run):
    return run or ms.list_runs(ov.eid, ov.eye).get("active") or "gold"


@router.post("/volumes/{volume_id}/segment/sam_bscan")
def sam_bscan(volume_id: str, body: SamBscanIn, store=Depends(get_store), ms=Depends(get_mask_store)):
    ov = _ov(store, volume_id)
    url = _require_endpoint()
    if not (0 <= body.bscan < ov.n_bscans):
        raise HTTPException(status_code=400, detail="bscan out of range")
    if not (body.box or body.points):
        raise HTTPException(status_code=400, detail="provide a box or points")
    run = _active_or_gold(ms, ov, body.run)
    if not ms.run_exists(ov.eid, ov.eye, run):
        ms.create_run(ov.eid, ov.eye, run, {"label": run, "source": "manual"})
    try:
        mask_png = Sam2Client(url).segment(render.bscan_png(ov, body.bscan), box=body.box, points=body.points)
    except SegmenterError as e:
        raise HTTPException(status_code=502, detail=str(e))
    mask = _decode_mask(mask_png, ov.H, ov.W)
    painted = int(mask.sum()) if mask is not None else 0
    if mask is not None and mask.any():
        ms.put_mask(ov.eid, ov.eye, run, body.bscan, mask)
        ms.set_status(ov.eid, ov.eye, run, body.bscan, state="ga")
    ms.set_active(ov.eid, ov.eye, run)
    return {"ok": True, "run": run, "bscan": body.bscan, "painted": painted}


def _sam_batch(volume_id, ov, ms, url, run, box, idxs):
    client = Sam2Client(url)
    for i in idxs:
        with _JOBLOCK:
            if not _JOBS.get(volume_id, {}).get("running"):
                break
        try:
            mask = _decode_mask(client.segment(render.bscan_png(ov, i), box=box), ov.H, ov.W)
            if mask is not None and mask.any():
                ms.put_mask(ov.eid, ov.eye, run, i, mask)
                ms.set_status(ov.eid, ov.eye, run, i, state="ga")
        except SegmenterError as e:
            with _JOBLOCK:
                _JOBS[volume_id]["error"] = str(e)
            break
        with _JOBLOCK:
            _JOBS[volume_id]["done"] += 1
    with _JOBLOCK:
        _JOBS[volume_id]["running"] = False


@router.post("/volumes/{volume_id}/segment/sam_propagate")
def sam_propagate(volume_id: str, body: SamPropagateIn, store=Depends(get_store), ms=Depends(get_mask_store)):
    """Apply one SAM2 box across a B-scan range (background) — re-segments each B-scan with the same box.
    Useful where the lesion is roughly aligned across adjacent scans; correct afterward. (Not video-memory
    tracking.) Progress via GET …/segment/label_job."""
    ov = _ov(store, volume_id)
    url = _require_endpoint()
    with _JOBLOCK:
        if _JOBS.get(volume_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="a job is already in progress")
    run = _active_or_gold(ms, ov, body.run)
    if not ms.run_exists(ov.eid, ov.eye, run):
        ms.create_run(ov.eid, ov.eye, run, {"label": run, "source": "manual"})
    rng = body.bscan_range or [body.bscan - 5, body.bscan + 5]
    idxs = list(range(max(0, rng[0]), min(ov.n_bscans, rng[1] + 1)))
    with _JOBLOCK:
        _JOBS[volume_id] = {"running": True, "done": 0, "total": len(idxs), "run": run,
                            "concept": "sam2 box", "error": None}
    threading.Thread(target=_sam_batch, args=(volume_id, ov, ms, url, run, body.box, idxs), daemon=True).start()
    return {"ok": True, "run": run, "total": len(idxs)}


def _enface_to_native(ov, fx, fy):
    """Inverse of projection.to_enface (resample centred about the image centre + slow-axis flip):
    a normalized en-face point (fx,fy)∈[0,1] -> (bscan, col) in the native volume."""
    fovx, fovy = float(ov.fov_mm[0]), float(ov.fov_mm[1])
    n, W = ov.n_bscans, ov.W
    out = max(64, int(round(max(fovx, fovy) / fp.MMPP)))
    sx, sy = (fovx / W) / fp.MMPP, (fovy / n) / fp.MMPP
    xd, yd = fx * (out - 1), fy * (out - 1)
    col = int(round((xd - (out - 1) / 2.0) / sx + (W - 1) / 2.0))
    ys = (yd - (out - 1) / 2.0) / sy + (n - 1) / 2.0          # row in the flipped (fundus) frame
    # undo the slow-axis flip ONLY if to_enface applied it (reverse-scanned rasters pass it through)
    bscan = int(round((n - 1) - ys)) if getattr(ov, "enface_flip", True) else int(round(ys))
    return max(0, min(n - 1, bscan)), max(0, min(W - 1, col))


def _sam_enface_batch(volume_id, ov, ms, url, run, c_lo, c_hi, idxs):
    import m3_projections as mp                                     # module-level mp isn't in scope here
    client = Sam2Client(url)
    lo_px, hi_px = mp.SLAB_UM[0] / mp.AX, mp.SLAB_UM[1] / mp.AX     # the sub-BM slab depth (the HT zone)
    for i in idxs:
        with _JOBLOCK:
            if not _JOBS.get(volume_id, {}).get("running"):
                break
        try:
            seg = np.asarray(ov.bm[i][c_lo:c_hi + 1], float)
            seg = seg[np.isfinite(seg)]
            if seg.size:
                bm_row = float(np.median(seg))
                box = [float(c_lo), bm_row + lo_px, float(c_hi), bm_row + hi_px]   # cols × sub-BM slab
                mask = _decode_mask(client.segment(render.bscan_png(ov, i), box=box), ov.H, ov.W)
                if mask is not None and mask.any():
                    ms.put_mask(ov.eid, ov.eye, run, i, mask)
                    ms.set_status(ov.eid, ov.eye, run, i, state="ga")
        except SegmenterError as e:
            with _JOBLOCK:
                _JOBS[volume_id]["error"] = str(e)
            break
        with _JOBLOCK:
            _JOBS[volume_id]["done"] += 1
    with _JOBLOCK:
        _JOBS[volume_id]["running"] = False


@router.post("/volumes/{volume_id}/segment/sam_enface")
def sam_enface(volume_id: str, body: SamEnfaceIn, store=Depends(get_store), ms=Depends(get_mask_store)):
    """Drag a box on the projection -> SAM2-annotate the B-scans it covers (the column span × the sub-BM
    slab on each). Lets you locate GA on the en-face (with the PLEX overlay) and label the stack at once."""
    ov = _ov(store, volume_id)
    url = _require_endpoint()
    with _JOBLOCK:
        if _JOBS.get(volume_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="a job is already in progress")
    run = _active_or_gold(ms, ov, body.run)
    if not ms.run_exists(ov.eid, ov.eye, run):
        ms.create_run(ov.eid, ov.eye, run, {"label": run, "source": "manual"})
    b0, c0 = _enface_to_native(ov, body.x0, body.y0)
    b1, c1 = _enface_to_native(ov, body.x1, body.y1)
    b_lo, b_hi = sorted((b0, b1))
    c_lo, c_hi = sorted((c0, c1))
    idxs = list(range(b_lo, b_hi + 1))
    with _JOBLOCK:
        _JOBS[volume_id] = {"running": True, "done": 0, "total": len(idxs), "run": run,
                            "concept": "sam2 en-face box", "error": None}
    threading.Thread(target=_sam_enface_batch, args=(volume_id, ov, ms, url, run, c_lo, c_hi, idxs),
                     daemon=True).start()
    return {"ok": True, "run": run, "total": len(idxs), "bscan_range": [b_lo, b_hi], "col_range": [c_lo, c_hi]}


# ------------------------------------------------------------------ PLEX-guided en-face GA editor
def _enface_size(ov):
    return max(64, int(round(max(ov.fov_mm) / fp.MMPP)))


def _rgba_ga(mask_bool):
    """A bool en-face mask -> RGBA PNG bytes (opaque yellow where GA, transparent elsewhere)."""
    g = np.asarray(mask_bool) > 0
    rgba = np.zeros((g.shape[0], g.shape[1], 4), np.uint8)
    rgba[g] = (0, 255, 255, 255)                                  # BGRA: yellow, opaque
    return cv2.imencode(".png", rgba)[1].tobytes()


def _enface_to_native_mask(ov, enf):
    """En-face GA mask (out×out bool) -> native (n, W) bool GA-column map. Inverse of
    projection.to_enface (centred resample about the image centre + slow-axis flip)."""
    n, W = ov.n_bscans, ov.W
    fh = max(1, int(round(ov.fov_mm[1] / fp.MMPP)))
    fw = max(1, int(round(ov.fov_mm[0] / fp.MMPP)))
    field = _center_extract(enf.astype(np.uint8), fh, fw)
    nat = cv2.resize(field, (W, n), interpolation=cv2.INTER_NEAREST)
    return (nat[::-1] if getattr(ov, "enface_flip", True) else nat) > 0    # match to_enface's flip


def _decode_enface(raw):
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr[..., 3] if arr.shape[2] == 4 else arr[..., 0]   # alpha if RGBA, else first channel
    return arr > 127


@router.get("/volumes/{volume_id}/segment/enface_plex.png")
def enface_plex_png(volume_id: str, registered: bool = Query(default=True), store=Depends(get_store)):
    """The advRPE/PLEX GA as a FILLED en-face mask in the projection's frame (the editor's seed).
    registered=True: the two-map GA-driven lock; False: center-fit (scale-matched, no lock) — the manual
    starting point the user nudges to fix registration. Matches plex_mask.png's base so the cyan contour
    the user aligns and the seeded footprint coincide."""
    ov = _ov(store, volume_id)
    ref = _cohort_ref(store, ov)
    s = _enface_size(ov)
    gl = None
    if ref and os.path.exists(ref["ga_mask"]):
        if registered:
            r = _plex_registered(ov, ref)
            gl = _center_fit(r, s) if r is not None else None
        else:
            m = cv2.imread(ref["ga_mask"], cv2.IMREAD_GRAYSCALE)
            gl = _center_fit(m > 127, s) if m is not None else None
    if gl is None:
        gl = np.zeros((s, s), bool)
    return Response(_rgba_ga(gl), media_type="image/png")


@router.get("/volumes/{volume_id}/segment/enface_mask.png")
def enface_mask_png(volume_id: str, run: str = Query(...), min_diam_um: float = Query(250.0),
                    store=Depends(get_store), ms=Depends(get_mask_store)):
    """The run's current GA footprint as a FILLED en-face mask (loaded into the editor on open)."""
    ov = _ov(store, volume_id)
    keep, _ = fp.run_footprint(ov, ms, run, float(min_diam_um))
    return Response(_rgba_ga(keep), media_type="image/png")


@router.post("/volumes/{volume_id}/segment/enface_commit")
async def enface_commit(volume_id: str, request: Request, run: str = Query(...),
                        min_diam_um: float = Query(250.0), cls: str = Query("wedge"),
                        store=Depends(get_store), ms=Depends(get_mask_store),
                        layer_store=Depends(get_layer_store)):
    """Receive the edited en-face GA mask -> paint the class's depth band on each flagged column's B-scan
    (wedge = sub-BM slab; rpe = the RPE band) and return the area. The en-face map is the source of truth;
    the B-scan masks are derived."""
    ov = _ov(store, volume_id)
    enf = _decode_enface(await request.body())
    if enf is None:
        raise HTTPException(status_code=400, detail="bad en-face mask PNG")
    if not ms.run_exists(ov.eid, ov.eye, run):
        ms.create_run(ov.eid, ov.eye, run, {"label": run, "source": "manual", "class": cls})
    native = _enface_to_native_mask(ov, enf)
    bm_full = _eff_bm(ov, layer_store)                            # corrected BM (BM-tab edits folded in)
    lo_px, hi_px = sc.band_px(cls)                                 # per-class depth band below/above BM
    for i in range(ov.n_bscans):
        cols = np.where(native[i])[0]
        bm_i = np.asarray(bm_full[i], float)
        m = np.zeros((ov.H, ov.W), bool)
        for c in cols:
            b = bm_i[c]
            if np.isfinite(b):
                m[max(0, int(b + lo_px)):min(ov.H, int(b + hi_px)), c] = True
        if m.any():
            ms.put_mask(ov.eid, ov.eye, run, i, m)
            ms.set_status(ov.eid, ov.eye, run, i, state="ga")
        else:
            ms.delete_mask(ov.eid, ov.eye, run, i)
            ms.clear_status(ov.eid, ov.eye, run, i)
    ms.set_active(ov.eid, ov.eye, run)
    _, area = fp.run_footprint(ov, ms, run, float(min_diam_um))
    return {"ok": True, "run": run, "area_mm2": round(float(area), 4)}


# ------------------------------------------------------------------ dataset export (this eye)
@router.post("/volumes/{volume_id}/segment/export")
def export_dataset(volume_id: str, run: str = Query(default=None),
                   store=Depends(get_store), ms=Depends(get_mask_store)):
    """Export the OPEN eye's labeled B-scans (ga + ga_free; todo/borderline skipped) into
    outputs/ga_bscan_dataset/, MERGING this eye into manifest.csv / the summary / splits.json (replacing
    any prior rows for it — never clobbering the other eyes). Reuses src/export_bscan_dataset (same B-scan
    render + multi-class labels as the CLI), handed the already-open volume + live mask store so it's fast
    and identical to what's on screen. Needs the E2E to map to a cohort subject-visit (the advRPE key)."""
    ov = _ov(store, volume_id)
    ref = _cohort_ref(store, ov)
    if not ref:
        raise HTTPException(status_code=400,
                            detail="this E2E doesn't map to a cohort subject-visit "
                                   "(need an NHAMD-003-…-V… key + cohort/<key>/<eye>/)")
    import export_bscan_dataset as ed
    try:
        s = ed.export_single(ref["key"], ov.eye, "" if ref["area"] is None else ref["area"],
                             run_arg=run, ov=ov, ms=ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export failed: {type(e).__name__}: {e}")
    if not s:
        raise HTTPException(status_code=400,
                            detail="no annotation run to export — seed/paint or mark GA-free first")
    return {"ok": True, "out_dir": ed.OUT, **s}


# ------------------------------------------------------------------ OAC GA auto-detector (this eye)
@router.get("/volumes/{volume_id}/segment/oac_ga.png")
def oac_ga_overlay(volume_id: str, frac: float = Query(default=0.50), order: int = Query(default=2),
                   store=Depends(get_store), layer_store=Depends(get_layer_store)):
    """Run the BM-anchored OAC GA detector (`reader/core/oac_ga`) on the OPEN eye, using the EFFECTIVE
    (corrected) BM, and return the RPE-loss en-face with the green cRORA footprint drawn. The OAC area (+
    the advRPE reference when the eye maps to the cohort) ride back in response headers, so the client gets
    the picture and the number in one request. Identical logic to the CLI `src/oac_area.py`.

    `order` = healthy-baseline polynomial order: 2 = QUADRATIC (radial falloff; best on focal eyes like 005,
    gold-validated Dice 0.93) or 1 = LINEAR (cleaner on large eccentric lesions like 008). Both kept for
    testing as more eyes get BM-corrected."""
    ov = _ov(store, volume_id)
    bm = _eff_bm(ov, layer_store)
    rpe6, mask, area = oac_ga.detect(ov, bm, frac=float(frac), trend_order=int(order))
    headers = {"X-OAC-Area-mm2": f"{area:.4f}", "X-OAC-Frac": f"{float(frac):.3f}",
               "X-OAC-Order": str(int(order))}
    ref = _cohort_ref(store, ov)
    if ref and ref.get("area") is not None:
        headers["X-AdvRPE-Area-mm2"] = f"{ref['area']:.4f}"
    return Response(oac_ga.overlay_png(rpe6, mask), media_type="image/png", headers=headers)
