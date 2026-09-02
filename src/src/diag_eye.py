#!/usr/bin/env python
"""Per-eye failure diagnosis for the OAC GA detector (reference config, DL BM) vs the advRPE reference.

For ONE eye it reproduces the cohort number (DL Bruch's-membrane + the reader/compare_plex default config)
and renders two panels so a human/agent can SEE why our area disagrees with advRPE -- with NO annotation:

  <subj>_<eye>_diag_enface.png : OAC RPE-loss (dark=GA) | our cRORA footprint (green) | advRPE SubRPE
                                 en-face | advRPE GA outline -- the EN-FACE seg comparison (over/under-call).
  <subj>_<eye>_diag_bscans.png : a strip of B-scans across the volume with the DL BM (red) + the OAC RPE
                                 sampling band (cyan) drawn, so a BM dive into the bright sub-RPE under GA
                                 (the classic false-negative) is visible.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\diag_eye.py NHAMD-003-015 V3 OS
Output -> outputs/oac_diag/.
"""
import argparse
import csv
import glob
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import bm_dl  # noqa: E402
import qcviz as qv  # noqa: E402
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402
from reader.core import projection as proj  # noqa: E402
from sweep_oac import REF_FP, REF_PREP  # noqa: E402  (the reference config, single source of truth)

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
COHORT = os.path.join(_REPO, "cohort")
OUT = os.path.join(OUT_DIR, "oac_diag")


def row_for(subject, visit, eye):
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("qc_status") or "").strip() != "ok":
                continue
            if subject.lower() in r["subject"].lower() and r["visit"] == visit and r["eye"].upper() == eye:
                return r
    raise SystemExit(f"{subject} {visit} {eye} not found qc_ok in {PAIRING}")


def find_advrpe(subject, visit, eye):
    """Return (subrpe_enface_path, ga_outline_path, ir_path) from cohort/, or (None,None,None)."""
    key = None
    for d in glob.glob(os.path.join(COHORT, "*")):
        b = os.path.basename(d)
        if subject.split("-")[-1] in b and visit in b:
            key = d
            break
    if key is None:
        return None, None, None
    sub = os.path.join(key, eye)
    if not os.path.isdir(sub):
        sub = key

    def pick(name):
        p = os.path.join(sub, name)
        return p if os.path.exists(p) else None
    return pick("advrpe_subrpe_enface.png"), pick("advrpe_ga_outline.png"), pick("spectralis_ir.png")


def g8(m):
    return qv.ensure_rgb(qv.norm8(np.nan_to_num(np.asarray(m, np.float32))))


def load_img(p, shape):
    if p is None or not os.path.exists(p):
        return np.zeros((*shape, 3), np.uint8)
    im = np.array(Image.open(p).convert("RGB"))
    return im


def bscan_strip(ov, bm, n=6):
    """Horizontal montage of n B-scans with the DL BM (red) + OAC RPE band BM-50..-8um (cyan) drawn."""
    H, W = ov.H, ov.W
    ax = proj.AX
    lo_off, hi_off = -50.0 / ax, -8.0 / ax    # rows above BM (band)
    idxs = np.linspace(0, ov.n_bscans - 1, n).round().astype(int)
    tiles, titles = [], []
    for i in idxs:
        rgb = g8(ov.vol[i]).copy()
        bmi = np.asarray(bm[i], float)
        cols = np.arange(W)
        ok = np.isfinite(bmi)
        rr = np.clip(np.round(bmi), 0, H - 1).astype(int)
        rgb[rr[ok], cols[ok]] = (255, 40, 40)                      # BM red
        for off in (lo_off, hi_off):
            br = np.clip(np.round(bmi + off), 0, H - 1).astype(int)
            rgb[br[ok], cols[ok]] = (0, 220, 220)                  # band edges cyan
        tiles.append(rgb)
        titles.append(f"b{i}")
    return qv.panel(tiles, titles, header="B-scans: DL BM (red) + OAC RPE band (cyan)",
                    mm_per_px=proj.ENFACE_MMPP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("visit")
    ap.add_argument("eye")
    a = ap.parse_args()
    eye = a.eye.upper()
    r = row_for(a.subject, a.visit, eye)
    subj = r["subject"]
    try:
        adv = float(r["advRPE_area_mm2"])
    except (TypeError, ValueError):
        adv = float("nan")
    e2e_path = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))

    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    bm = bm_dl.segment_volume(ov.vol)
    P = oac_ga.prep(ov, bm, **REF_PREP)
    mask, area = oac_ga.footprint(P, **REF_FP)
    rpe6 = P["rpe6"]
    print(f"{subj} {eye}: n={ov.n_bscans} bm_src={ov.bm_src}  OUR(DL,ref)={area:.3f}  advRPE={adv:.3f}  "
          f"diff={area - adv:+.3f} mm2", flush=True)

    os.makedirs(OUT, exist_ok=True)
    foot = qv.draw_contour(g8(rpe6), mask, color=(0, 255, 0), thick=2)
    foot[mask] = (0.45 * foot[mask] + 0.55 * np.array([0, 200, 0])).astype(np.uint8)
    sub_p, out_p, ir_p = find_advrpe(a.subject, a.visit, eye)
    shape = rpe6.shape
    enf = qv.panel(
        [g8(rpe6), foot, load_img(sub_p, shape), load_img(out_p, shape)],
        ["OAC RPE-loss (dark=GA)", f"our cRORA footprint {area:.2f} mm2",
         "advRPE SubRPE en-face", f"advRPE GA outline ({adv:.2f} mm2)"],
        header=f"{subj} {eye}  OUR(DL,ref)={area:.2f} vs advRPE={adv:.2f} mm2  (diff {area - adv:+.2f})  "
               f"bm_src={ov.bm_src}",
        mm_per_px=proj.ENFACE_MMPP)
    p1 = os.path.join(OUT, f"{subj}_{eye}_diag_enface.png")
    qv.save_rgb(p1, enf)
    print(f"  wrote {p1}", flush=True)
    try:
        p2 = os.path.join(OUT, f"{subj}_{eye}_diag_bscans.png")
        qv.save_rgb(p2, bscan_strip(ov, bm))
        print(f"  wrote {p2}", flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"  B-scan strip failed: {e}", flush=True)


if __name__ == "__main__":
    main()
