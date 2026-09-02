#!/usr/bin/env python
"""M2 driver/QC: run BM self-segmentation on an eye, validate vs the device BM where present,
and write annotated B-scans (our BM yellow, device BM cyan, the 64-400 um sub-BM slab shaded).

Run: oct_env\\Scripts\\python.exe m2_bm.py [SUBJECT EYE ...]   (default: 003-008 OD, 003-009 OD,
003-001 OD -- a big-GA, a no-GA, and a multifocal eye)
"""
import json
import os
import sys

import cv2
import numpy as np
from oct_converter.readers import E2E

import bm as bmseg
import qcviz as qv

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                       # so `reader.core.fieldmask` resolves
from reader.core import fieldmask                   # noqa: E402

COH = os.path.join(ROOT, "cohort")
OUT = os.path.join(OUT_DIR, "m2_bm_out")
SLAB_LO_PX = 64.0 / bmseg.AXIAL_UM_PER_PX     # ~16.5 px below BM
SLAB_HI_PX = 400.0 / bmseg.AXIAL_UM_PER_PX    # ~103 px below BM


def eye_of(lat):
    return "OD" if str(lat).strip().upper() in ("R", "OD", "RIGHT") else "OS"


def _device_bm(v):
    """Deepest finite-coverage device contour on a volume = BM (contour1), else None."""
    contours = getattr(v, "contours", None)
    if not isinstance(contours, dict):
        return None
    best, best_mean = None, -1
    for val in contours.values():
        a = np.asarray(val, float)
        fin = a[np.isfinite(a) & (a > 0)]
        if fin.size > 0.3 * a.size and fin.mean() > best_mean:
            best, best_mean = a, fin.mean()
    return best


def _device_ilm(v):
    """Shallowest finite-coverage device contour = ILM (contour0), else None."""
    contours = getattr(v, "contours", None)
    if not isinstance(contours, dict):
        return None
    best, best_mean = None, 1e18
    for val in contours.values():
        a = np.asarray(val, float)
        fin = a[np.isfinite(a) & (a > 0)]
        if fin.size > 0.3 * a.size and fin.mean() < best_mean:
            best, best_mean = a, fin.mean()
    return best


def load_subject(subject):
    """Open the E2E once; return {eye: (volume[n,H,W], device_bm[n,W] or None)} for the 30deg volume."""
    with open(os.path.join(COH, subject, "meta.json")) as f:
        e2e_rel = json.load(f)["e2e_file"]
    vols = E2E(os.path.join(DATA_DIR, e2e_rel)).read_oct_volume()
    out = {}
    for eye in ("OD", "OS"):
        cands = [v for v in vols if eye_of(getattr(v, "laterality", None)) == eye
                 and np.asarray(v.volume).shape[-1] >= 768 and len(v.volume) > 5]
        if not cands:
            continue
        v = max(cands, key=lambda v: len(v.volume))
        out[eye] = (np.asarray(v.volume, float), _device_bm(v))
    return out


def load_subject_layers(subject):
    """Like load_subject but also returns device ILM: {eye: (vol, ilm or None, bm or None)}."""
    with open(os.path.join(COH, subject, "meta.json")) as f:
        e2e_rel = json.load(f)["e2e_file"]
    vols = E2E(os.path.join(DATA_DIR, e2e_rel)).read_oct_volume()
    out = {}
    for eye in ("OD", "OS"):
        cands = [v for v in vols if eye_of(getattr(v, "laterality", None)) == eye
                 and np.asarray(v.volume).shape[-1] >= 768 and len(v.volume) > 5]
        if not cands:
            continue
        v = max(cands, key=lambda v: len(v.volume))
        out[eye] = (np.asarray(v.volume, float), _device_ilm(v), _device_bm(v))
    return out


def qc_bscan(bscan, bm_row, dev_row, H):
    """Annotated, vertically-zoomed B-scan: BM (yellow), device BM (cyan), slab band (green)."""
    rgb = qv.ensure_rgb(qv.norm8(bscan))
    W = rgb.shape[1]
    xs = np.arange(W)
    # translucent slab band BM+lo .. BM+hi
    band = rgb.copy()
    for x in xs:
        y0 = int(round(bm_row[x] + SLAB_LO_PX))
        y1 = int(round(bm_row[x] + SLAB_HI_PX))
        if 0 <= y0 < H:
            band[y0:min(y1, H), x] = (0, 180, 0)
    rgb = cv2.addWeighted(rgb, 0.7, band, 0.3, 0)
    if dev_row is not None:
        for x in xs:
            y = dev_row[x]
            if np.isfinite(y) and 0 <= int(y) < H:
                rgb[max(0, int(y) - 1):int(y) + 1, x] = (0, 220, 255)   # device BM cyan
    for x in xs:
        y = int(round(bm_row[x]))
        if 0 <= y < H:
            rgb[max(0, y - 1):y + 1, x] = (255, 255, 0)                 # our BM yellow
    # zoom vertically to the BM neighbourhood
    lo = max(0, int(bm_row.min()) - 60)
    hi = min(H, int(bm_row.max() + SLAB_HI_PX) + 30)
    crop = rgb[lo:hi]
    return cv2.resize(crop, (W, crop.shape[0] * 3), interpolation=cv2.INTER_NEAREST)


def slab_enface(vol, bm):
    """Mean OCT intensity in the BM+64..BM+400 um sub-BM slab -> en-face hypertransmission map."""
    n, H, W = vol.shape
    lo = np.clip(np.round(bm + SLAB_LO_PX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bm + SLAB_HI_PX), 1, H).astype(int)
    out = np.zeros((n, W), np.float32)
    for i in range(n):
        for x in range(W):
            a, b = lo[i, x], hi[i, x]
            if b > a:
                out[i, x] = vol[i, a:b, x].mean()
    return out


def fill_bm(dev, invalid=None):
    """Interpolate missing columns of a device BM surface + light WITHIN-B-scan smoothing, so it's usable.

    `invalid` (n,W) bool: saturated machine-fill A-scans. They are dropped from the valid-anchor set so
    interpolation goes ACROSS them rather than treating a saturated column's (wrong) value as a true
    anchor -- otherwise the corruption spreads to neighbours."""
    from scipy.ndimage import gaussian_filter, median_filter
    out = np.asarray(dev, float).copy()
    for i in range(out.shape[0]):
        row, m = out[i], np.isfinite(out[i]) & (out[i] > 0)
        if invalid is not None:
            m = m & ~np.asarray(invalid[i], bool)
        if 5 < m.sum() < len(row):
            row[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), row[m])
        out[i] = row
    # Denoise WITHIN each B-scan only (fast axis). The previous 2D smoothing mixed the SLOW axis (median
    # size-3 + gaussian sigma-1 across B-scans), dragging the BM up to ~40px off the device line across gap
    # runs (002 OD b72-77) — a dive the reader never showed (it draws raw bm_display) but the OAC area AND
    # the BM-DL training labels inherited. size-1 / sigma-0 on the slow axis keeps each B-scan on its own
    # device contour (max ~0.5px off the device line vs 38.5px before).
    return gaussian_filter(median_filter(out, size=(1, 7)), sigma=(0.0, 1.0))


def enface_panel(subject, eye, vol, bm, dev_bm, invalid=None):
    """Compare the slab en-face from OUR BM vs from the DEVICE BM, against the advRPE reference."""
    sq = lambda a: cv2.resize(qv.norm8(a), (512, 512), interpolation=cv2.INTER_LINEAR)
    tiles = [sq(slab_enface(vol, bm))]
    titles = ["slab from OUR BM"]
    if dev_bm is not None and dev_bm.shape == bm.shape:
        tiles.append(sq(slab_enface(vol, fill_bm(dev_bm, invalid=invalid))))
        titles.append("slab from DEVICE BM")
    adv = cv2.imread(os.path.join(COH, subject, eye, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
    if adv is not None:
        tiles.append(adv); titles.append("advRPE SubRPE 6x6 (reference)")
    panel = qv.panel(tiles, titles, header=f"{subject} {eye}  sub-BM slab en-face: OUR BM vs "
                                           f"DEVICE BM vs reference (full field vs 6x6)")
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_enface.png"), panel)


def _err_stats(bm, dev_bm, W):
    """Central-6mm and full-width BM error (um) vs device BM; (central, full) text + central dict."""
    if dev_bm is None or dev_bm.shape != bm.shape:
        return "no device BM", {"has_device": 0}
    cen = np.zeros(W, bool)
    h = int(round(0.5 * (6.0 / 8.77) * W))
    cen[W // 2 - h: W // 2 + h] = True

    def stat(colmask):
        m = (np.isfinite(dev_bm) & (dev_bm > 0)) & colmask[None, :]
        d = np.abs(bm[m] - dev_bm[m]) * bmseg.AXIAL_UM_PER_PX
        return d.mean(), float(np.median(d)), float(np.percentile(d, 95))

    cm, cmd, cp = stat(cen)
    fm, fmd, fp = stat(np.ones(W, bool))
    txt = f"central6mm mean={cm:.1f} median={cmd:.1f} p95={cp:.1f}um (full median={fmd:.1f})"
    return txt, {"has_device": 1, "err_mean_um": round(cm, 1), "err_median_um": round(cmd, 1),
                 "err_p95_um": round(cp, 1), "err_full_median_um": round(fmd, 1)}


def process(subject, eye, vol, dev_bm):
    os.makedirs(OUT, exist_ok=True)
    n, H, W = vol.shape
    inv = fieldmask.invalid_mask(vol)               # saturated machine-fill columns (out-of-field band)
    bm = bmseg.segment_volume(vol, invalid=inv)
    err_txt, err = _err_stats(bm, dev_bm, W)
    print(f"[{subject} {eye}] n={n}  {err_txt}", flush=True)

    for bi in (n // 4, n // 2, 3 * n // 4):
        dev = dev_bm[bi] if dev_bm is not None else None
        tile = qc_bscan(vol[bi], bm[bi], dev, H)
        hdr = (f"{subject} {eye}  B-scan {bi}/{n}   yellow=our BM  cyan=device BM  "
               f"green=64-400um slab   {err_txt}")
        qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_bscan{bi:03d}.png"), qv.add_title(tile, hdr))
    enface_panel(subject, eye, vol, bm, dev_bm, invalid=inv)
    return {"subject": subject, "eye": eye, "n_bscans": n, **err}


def ok_eyes():
    import csv
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("qc_status") == "ok"]
    by_sub = {}
    for r in rows:
        by_sub.setdefault(r["subject"], []).append(r["eye"])
    return by_sub


def main():
    import csv
    args = sys.argv[1:]
    if args and args[0].lower() != "all":               # explicit SUBJECT EYE pairs
        pairs = {}
        for i in range(0, len(args) - 1, 2):
            s = args[i] if args[i].startswith("NHAMD") else "NHAMD-003-" + args[i]
            pairs.setdefault(s, []).append(args[i + 1])
        by_sub = pairs
    else:                                               # default / 'all' -> every qc_ok eye
        by_sub = ok_eyes()

    results = []
    for subject in sorted(by_sub):
        try:
            loaded = load_subject(subject)
        except Exception as ex:
            print(f"[{subject}] LOAD ERROR {type(ex).__name__}: {ex}", flush=True)
            continue
        for eye in by_sub[subject]:
            if eye not in loaded:
                print(f"[{subject} {eye}] no 30deg volume", flush=True)
                continue
            vol, dev_bm = loaded[eye]
            results.append(process(subject, eye, vol, dev_bm))

    if len(results) > 3:                                # write the cohort BM-error summary
        cols = ["subject", "eye", "n_bscans", "has_device", "err_mean_um", "err_median_um",
                "err_p95_um", "err_full_median_um"]
        with open(os.path.join(RESULTS_DIR, "m2_bm_errors.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow({c: r.get(c, "") for c in cols})
        dev = [r for r in results if r.get("has_device")]
        if dev:
            md = np.median([r["err_median_um"] for r in dev])
            print(f"\ncohort BM (n={len(dev)} w/ device): median of per-eye central errors = {md:.1f}um")
        print(f"-> m2_bm_errors.csv")
    print(f"wrote {len(results)} eyes x (3 B-scans + en-face) -> m2_bm_out/")


if __name__ == "__main__":
    main()
