"""Browse the filesystem and open an E2E by path."""
from fastapi import APIRouter, Depends, HTTPException, Query

from reader.core import filesystem, e2e_source, ids
from .deps import get_store
from .schemas import OpenE2EIn

router = APIRouter()


@router.get("/health")
def health(store=Depends(get_store)):
    return {"ok": True, "loaded": store.loaded_e2e()}


@router.get("/fs/list")
def fs_list(dir: str = Query(default=None)):
    return filesystem.list_dir(dir)


@router.post("/e2e/open")
def e2e_open(body: OpenE2EIn, store=Depends(get_store)):
    if not filesystem.is_e2e(body.path):
        raise HTTPException(status_code=400, detail="Not an .E2E file under the allowed root")
    raw = store.open_e2e(body.path)
    eyes = {}
    for r in raw.refs:
        eyes.setdefault(r.eye, []).append({
            "volume_id": ids.volume_id(raw.eid, r.index), "index": r.index, "kind": r.kind,
            "n_bscans": r.n_bscans, "W": r.W, "H": r.H, "fov_mm": list(r.fov_mm), "is_6x6": r.is_6x6,
        })
    default = {eye: ids.volume_id(raw.eid, e2e_source.default_volume_index(raw, eye))
               for eye in eyes}
    return {"e2e_id": raw.eid, "path": raw.path, "eyes": eyes, "default": default}


def _patient_of(raw):
    """The E2E's embedded patient identity, best-effort — the topbar fallback when a scan isn't opened
    from the cohort Library (which carries the subject/visit key instead). read_all_metadata()'s
    patient_data is a 1-element list of dicts: {first_name, surname, patient_id (e.g. '003-001'), ...}."""
    try:
        pd = (getattr(raw, "md", None) or {}).get("patient_data")
    except Exception:
        pd = None
    if isinstance(pd, (list, tuple)):
        pd = pd[0] if pd else None
    if not isinstance(pd, dict):
        return {}
    out = {}
    for k in ("patient_id", "first_name", "surname"):
        v = pd.get(k)
        if v is not None and str(v).strip():
            out[k] = str(v).strip()
    return out


@router.post("/volumes/{volume_id}/open")
def volume_open(volume_id: str, store=Depends(get_store)):
    try:
        ov = store.get_volume(volume_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Open the E2E first")
    from reader.core import calibration as cal
    return {
        "volume_id": ov.volume_id, "eye": ov.eye,
        "n_bscans": ov.n_bscans, "H": ov.H, "W": ov.W, "fov_mm": list(ov.fov_mm),
        "axial_mm_per_px": cal.axial_mm_per_px(),
        "lateral_mm_per_px": cal.lateral_mm_per_px(ov.fov_mm, ov.W),
        "layers": ["ilm", "bm"], "sources": {"ilm": ov.ilm_src, "bm": ov.bm_src},
        # False on a reverse-scanned raster (003-016/003-130): to_enface did NOT flip the rows, so the
        # projection click-probe must map rows straight through instead of un-flipping them.
        "enface_flip": bool(getattr(ov, "enface_flip", True)),
        "patient": _patient_of(store.get_raw(ov.eid)),
    }
