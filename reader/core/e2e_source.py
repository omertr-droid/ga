"""Open a Spectralis E2E by path and load a chosen OCT volume with its layer surfaces.

Reuses the proven pipeline readers (no copying):
  - oct_converter.E2E.read_oct_volume() / read_all_metadata() / read_fundus_image()  (safe reader;
    eyepy get_volume() crashes on sparse-layer files)
  - m2_bm._device_ilm / _device_bm / fill_bm        (device contour extraction + gap fill)
  - bm.segment_volume + a self-seg ILM via bm's graph helpers   (OCT-only fallback)

On the 6x6 (97-line) scan device contours are absent ~half the time, so self-segmentation is a
first-class path here, and each layer is tagged "device" or "auto".
"""
import os

import numpy as np
from oct_converter.readers import E2E

from . import m2_bm
import bm as bmseg

from . import calibration as cal
from . import fieldmask
from . import ids as idmod
from .volume import RawE2E, VolumeRef, OctVolume

# Per-volume surface memoization. The device/self-seg surfaces are deterministic from the volume, so the
# slow classical self-seg (~5-12s, the "segmenting layers" wait) is computed ONCE per (eid, index) and
# cached to disk -> later cold opens load arrays instead of re-segmenting. Pure memoization: a cache hit
# returns byte-identical surfaces, so behaviour is unchanged, only faster. Bump _SURF_VER to invalidate
# if the segmentation algorithm changes.
_SURF_CACHE = os.path.join(
    os.path.abspath(os.environ.get("OCT_CLINIC_DATA") or
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store")),
    "surfcache",
)
_SURF_VER = 1


def _surf_cache_path(eid, index):
    return os.path.join(_SURF_CACHE, f"{eid}_{int(index):02d}.npz")


def _load_surf_cache(eid, index, shape):
    """(ilm_display, bm_display, ilm_src, bm_src) from the memoized surfaces, or None on miss/mismatch."""
    p = _surf_cache_path(eid, index)
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            if int(z["ver"]) != _SURF_VER:
                return None
            ilm_d, bm_d = z["ilm_display"], z["bm_display"]
            if ilm_d.shape != tuple(shape) or bm_d.shape != tuple(shape):
                return None
            return ilm_d, bm_d, str(z["ilm_src"]), str(z["bm_src"])
    except (OSError, ValueError, KeyError):
        return None


def _save_surf_cache(eid, index, ilm_display, bm_display, ilm_src, bm_src):
    try:
        os.makedirs(_SURF_CACHE, exist_ok=True)
        p = _surf_cache_path(eid, index)
        tmp = p + ".tmp.npz"
        np.savez(tmp, ver=_SURF_VER,
                 ilm_display=np.asarray(ilm_display, np.float32),
                 bm_display=np.asarray(bm_display, np.float32),
                 ilm_src=str(ilm_src), bm_src=str(bm_src))
        os.replace(tmp, p)                              # atomic; safe across the offline batch + the server
    except OSError:
        pass


def eye_tag(lat):
    s = str(lat).strip().upper()
    if s in ("R", "OD", "RIGHT", "82"):
        return "OD"
    if s in ("L", "OS", "LEFT", "76"):
        return "OS"
    return "U"


def _parse_sid(id_str):
    try:
        return int(str(id_str).split("_")[-1])
    except Exception:
        return None


def _modality_map(path):
    """series_id -> en-face modality (NIR/BAF/...) via eyepy, best-effort (returns {} on failure)."""
    try:
        from eyepy.io.he import HeE2eReader
        out = {}
        with HeE2eReader(path) as r:
            for s in r.series:
                sid = getattr(s, "id", None)
                try:
                    mod = str(s.enface_modality())
                except Exception:
                    mod = "UNKNOWN"
                if sid is not None:
                    out[int(sid)] = mod
        return out
    except Exception:
        return {}


def _vol_shape(v):
    try:
        return tuple(int(x) for x in np.asarray(v.volume[0]).shape)
    except Exception:
        return None


def _is_6x6(n_bscans, W, fov):
    """The squarish ~6 mm field: 512-wide, ~90-130 B-scans, both axes within ~5-7 mm."""
    if W != 512 or not (80 <= n_bscans <= 130):
        return False
    if fov and (fov[0] < 4.5 or fov[1] < 4.5):
        return False
    return True


def _volume_refs(vols, cfov):
    refs = []
    for i, v in enumerate(vols):
        shp = _vol_shape(v)
        if shp is None:
            continue
        H, W = shp
        n = len(v.volume)
        fov = cfov.get((n, W)) or (0.0, 0.0)
        is6 = _is_6x6(n, W, fov)
        kind = f"{n}line/{W}px" + (" (6x6)" if is6 else
                                   " (30deg)" if W >= 768 and n > 1 else
                                   " (1-line)" if n <= 1 else "")
        refs.append(VolumeRef(index=i, eye=eye_tag(getattr(v, "laterality", None)),
                              n_bscans=n, H=H, W=W, fov_mm=(round(fov[0], 3), round(fov[1], 3)),
                              is_6x6=is6, kind=kind))
    return refs


def open_e2e(path):
    """Open an E2E once; decode volumes + metadata + FOV classes + volume refs. Fundus is lazy."""
    reader = E2E(path)
    vols = reader.read_oct_volume()
    # oct-converter's intensity LUT yields float64 B-scans.  Every downstream radiometry/GA path begins
    # by converting to float32, so retaining the whole multi-volume E2E in float64 only doubles memory.
    # Convert one volume at a time so the superseded list/arrays are released immediately.
    for v in vols:
        try:
            v.volume = np.asarray(v.volume, np.float32)
        except (TypeError, ValueError):
            pass
    md = reader.read_all_metadata()
    cfov = cal.class_fov_mm(md.get("bscan_data") or [])
    raw = RawE2E(path=path, eid=idmod.e2e_id(path), reader=reader, vols=vols, cfov=cfov, md=md)
    raw.refs = _volume_refs(vols, cfov)
    raw.modmap = _modality_map(path)   # attached dynamically (dataclass has no slots)
    return raw


def default_volume_index(raw, eye=None):
    """Pick the 6x6 scan (most B-scans), optionally for a given eye; else the largest volume."""
    refs = [r for r in raw.refs if eye is None or r.eye == eye]
    six = [r for r in refs if r.is_6x6]
    if six:
        return max(six, key=lambda r: r.n_bscans).index
    if refs:
        return max(refs, key=lambda r: r.n_bscans).index
    return 0


# --------------------------------------------------------------------------- layer surfaces
def _self_ilm_volume(vol):
    """Self-segment the ILM surface (n,W), OCT-only. Back-compat wrapper over the hardened combined
    pass (bm.segment_surfaces_volume), which anchors the RPE/BM complex first and derives a
    speckle-proof ILM from it (the legacy top-down ILM seed latched onto vitreous noise)."""
    return bmseg.segment_surfaces_volume(np.asarray(vol, float))[0]


def _safe_surface(val, shape=None):
    """Coerce a device contour to a clean (n,W) float array, or None if ragged/empty/mis-shaped.
    (m2_bm._device_* do a bare np.asarray that ValueErrors on ragged contours -- some E2Es have
    them, e.g. 003-004's 6x6 -- which would 500 the loader; this guards against that.)"""
    try:
        a = np.asarray(val, dtype=float)
    except (ValueError, TypeError):
        return None
    if a.ndim != 2:
        return None
    if shape is not None and a.shape != tuple(shape):
        return None
    return a


def _device_layers(v, shape=None):
    """(ilm, bm) device contours or None each, robust to ragged/odd contours. ILM = shallowest
    finite-coverage contour, BM = deepest. A single contour is treated as BM (the anchor; self-seg
    ILM). `shape` = expected (n, W) of the volume; mis-shaped contours are rejected."""
    c = getattr(v, "contours", None)
    if not isinstance(c, dict):
        return None, None
    valid = []
    for val in c.values():
        a = _safe_surface(val, shape)
        if a is None:
            continue
        fin = a[np.isfinite(a) & (a > 0)]
        if fin.size > 0.3 * a.size:
            valid.append((float(fin.mean()), a))
    if not valid:
        return None, None
    valid.sort(key=lambda t: t[0])          # shallow (small y) first
    if len(valid) == 1:
        return None, valid[0][1]            # one contour -> keep as BM, self-seg ILM
    return valid[0][1], valid[-1][1]


def bscan_records(raw, index):
    """This volume's per-B-scan position records in B-scan (aktImage == vol) order, or None."""
    v = raw.vols[index]
    n = len(v.volume)
    W = int(np.asarray(v.volume[0]).shape[1])
    bd = raw.md.get("bscan_data") or []
    recs = [b for b in bd
            if b.get("numImages") == n and b.get("imgSizeX") == W and b.get("posX1") is not None]
    if len(recs) < n:
        return None
    same = [r for r in raw.refs if r.n_bscans == n and r.W == W]
    ordinal = next((k for k, r in enumerate(same) if r.index == index), 0)
    nblocks = max(1, len(recs) // n)
    b0 = min(ordinal, nblocks - 1) * n
    blk = recs[b0:b0 + n]
    if len(blk) != n:
        blk = recs[:n]
    akt = [b.get("aktImage") for b in blk]
    if sorted(a for a in akt if a is not None) == list(range(n)):
        order = sorted(range(n), key=lambda k: blk[k]["aktImage"])
    else:                                              # fallback: slow-axis monotonic across a raster
        order = sorted(range(n), key=lambda k: blk[k].get("centrePosY", 0.0))
    return [blk[k] for k in order]


def enface_flip_for(raw, index):
    """Should the en-face rows be flipped to reach fundus orientation? True for the usual raster.

    An en-face built as `row = B-scan i` is fundus-ordered only if posY DECREASES with i (posY positive is
    inferior, so B-scan 0 is then the bottom of the field and the rows must be reversed). 003-016 and
    003-130 raster the slow axis in REVERSE (posY increases with i): their rows already run superior ->
    inferior, and the blanket flip would stand the projection on its head. Unknown metadata -> True (the
    historical behaviour), so nothing regresses when the records are missing."""
    recs = bscan_records(raw, index)
    if not recs or len(recs) < 2:
        return True
    y0 = (recs[0].get("posY1", 0.0) + recs[0].get("posY2", 0.0)) / 2.0
    y1 = (recs[-1].get("posY1", 0.0) + recs[-1].get("posY2", 0.0)) / 2.0
    return bool(y0 > y1)


def load_volume(raw, index):
    """Load volume `index` from an opened E2E into an OctVolume (device layers where present,
    self-seg where absent; both gap-filled for the projection)."""
    v = raw.vols[index]
    ref = raw.refs[index]
    # OCT radiometry, OAC, DL inference and all persisted viewer arrays are float32.  Avoid a second,
    # volume-sized float64 copy here; the classical BM fallback may still promote its own working data.
    vol = np.asarray(v.volume, np.float32)

    inv = fieldmask.invalid_mask(vol)               # (n,W) saturated machine-fill columns (off-field band)

    surf_shape = (vol.shape[0], vol.shape[2])
    cached = _load_surf_cache(raw.eid, index, surf_shape)   # skip the ~5-12s self-seg on a cold re-open
    if cached is not None:
        ilm_raw, bm_raw, ilm_src, bm_src = cached
    else:
        ilm_raw, bm_raw = _device_layers(v, surf_shape)
        ilm_src = "device" if ilm_raw is not None else "auto"
        bm_src = "device" if bm_raw is not None else "auto"
        if ilm_raw is None or bm_raw is None:
            self_ilm, self_bm = bmseg.segment_surfaces_volume(vol, invalid=inv)   # interp across the band
            if ilm_raw is None:
                ilm_raw = self_ilm
            if bm_raw is None:
                bm_raw = self_bm
        _save_surf_cache(raw.eid, index, ilm_raw, bm_raw, ilm_src, bm_src)

    return OctVolume(
        volume_id=idmod.volume_id(raw.eid, index), eid=raw.eid, index=index, eye=ref.eye,
        vol=vol,
        ilm_display=np.asarray(ilm_raw, float), bm_display=np.asarray(bm_raw, float),
        ilm=m2_bm.fill_bm(ilm_raw, invalid=inv), bm=m2_bm.fill_bm(bm_raw, invalid=inv),
        ilm_src=ilm_src, bm_src=bm_src,
        fov_mm=tuple(float(x) for x in ref.fov_mm),
        field_valid=~inv,
        enface_flip=enface_flip_for(raw, index),
    )


# --------------------------------------------------------------------------- localizer
def _funduses(raw):
    if not raw.funduses:
        try:
            raw.funduses = raw.reader.read_fundus_image() or []
        except Exception:
            raw.funduses = []
    return raw.funduses


def localizer_image(raw, eye):
    """The IR/NIR SLO localizer for an eye (the B-scan position reference), or None.
    Prefers an IR/NIR modality over BAF using the eyepy modality map."""
    funduses = _funduses(raw)
    cands = [f for f in funduses if eye_tag(getattr(f, "laterality", None)) == eye]
    if not cands:
        return None

    def mod_of(f):
        return str(raw.modmap.get(_parse_sid(getattr(f, "image_id", None)), "")).upper()

    ir = [f for f in cands if "IR" in mod_of(f) or "NIR" in mod_of(f)]
    pick = (ir or cands)[0]
    try:
        return np.asarray(pick.image)
    except Exception:
        return None
