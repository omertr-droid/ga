#!/usr/bin/env python
"""BM / band-placement audit on B-scans: is the OAC RPE band actually over the RPE, and is the RPE present?

Renders B-scans (BM-centered vertical crop, upscaled) with every layer the detector uses drawn, so a
misplaced band / present-RPE can be SEEN:
  DL BM            red
  RPE peak         magenta dots  (mp.rpe_surface -- the ACTUAL RPE/EZ reflectivity peak; follows drusen up)
  OAC RPE band     cyan lines    (BM-50..-8 um -- mean OAC here = 'RPE present'; LOW = GA)
  hyper slab       orange lines  (BM+130..250 um -- the sub-BM hypertransmission slab)
GA-CALLED columns (our detector fires) get a red tick on the top edge -> see if RPE-present columns are
being called GA (a band-placement artifact) vs genuinely RPE-gone.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\band_audit.py NHAMD-003-016 V2 OD 31 34 37 40
  oct_env\\Scripts\\python.exe src\\band_audit.py            # default: 016 (firing), 005 (GA), 006 (control)
Output -> outputs/band_audit/<subject>_<eye>_bands.png
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OCT_BM_DL", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import bm_dl  # noqa: E402
import m3_projections as mp  # noqa: E402
import qcviz as qv  # noqa: E402
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402
from reader.core import projection as proj  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT = os.path.join(OUT_DIR, "band_audit")
AX = float(mp.AX)
OAC_BAND = mp.OAC_RPE_UM            # (-50, -8) um
SLAB = (130.0, 250.0)              # um below BM


def row_for(subject, visit, eye):
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("qc_status") or "").strip() == "ok" and subject in r["subject"] \
                    and r["visit"] == visit and r["eye"].upper() == eye:
                return r
    raise SystemExit(f"{subject} {visit} {eye} not qc_ok")


def ga_columns(ov, bm):
    """Per-(n,W) boolean: which A-scans the production detector flags as GA (en-face footprint resampled
    back to native columns via the viewer's ga_native helper)."""
    from viewer.core import ga_native
    _rpe6, mask, _area = oac_ga.detect(ov, bm)
    return ga_native.enface_to_native(mask.astype(np.float32), ov.fov_mm, ov.n_bscans, ov.W) > 0.5


def crop_draw(bscan, bm_i, rpe_i, ga_col):
    """BM-centered vertical crop of one B-scan with the layers/bands drawn, upscaled ~3x vertically."""
    H, W = bscan.shape
    top = int(max(0, np.nanmin(bm_i) - 100 / AX))
    bot = int(min(H, np.nanmax(bm_i) + 280 / AX))
    crop = bscan[top:bot]
    rgb = qv.ensure_rgb(qv.norm8(crop)).copy()
    cols = np.arange(W)

    def put(rows, color, dotted=False):
        rr = np.round(rows - top).astype(int)
        ok = (rr >= 0) & (rr < rgb.shape[0])
        if dotted:
            ok &= (cols % 3 == 0)
        rgb[rr[ok], cols[ok]] = color

    put(bm_i + SLAB[0] / AX, (255, 150, 0))                 # slab top  orange
    put(bm_i + SLAB[1] / AX, (255, 150, 0))                 # slab bot  orange
    put(bm_i + OAC_BAND[0] / AX, (0, 220, 220))             # OAC band top  cyan
    put(bm_i + OAC_BAND[1] / AX, (0, 220, 220))             # OAC band bot  cyan
    put(bm_i, (255, 40, 40))                                # BM  red
    if rpe_i is not None:
        put(rpe_i, (255, 0, 255), dotted=True)              # RPE peak  magenta dots
    # GA-called columns -> red tick on the top edge
    if ga_col is not None:
        gc = np.where(ga_col)[0]
        rgb[0:3, gc] = (255, 40, 40)
    out_h = int(rgb.shape[0] * 4)
    return cv2.resize(rgb, (int(W * 1.4), out_h), interpolation=cv2.INTER_NEAREST)


def render(subject, visit, eye, idxs):
    r = row_for(subject, visit, eye)
    subj = r["subject"]
    try:
        adv = float(r["advRPE_area_mm2"])
    except (TypeError, ValueError):
        adv = float("nan")
    raw = e2e_source.open_e2e(os.path.join(DATA_DIR, *r["e2e_file"].split("/")))
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    bm = bm_dl.segment_volume(ov.vol)
    rpe_row, _prom = mp.rpe_surface(ov.vol, bm)
    try:
        ga_nat = ga_columns(ov, bm)
    except Exception:                                       # noqa: BLE001
        ga_nat = None
    if not idxs:
        idxs = [int(round(x)) for x in np.linspace(ov.n_bscans * 0.35, ov.n_bscans * 0.6, 4)]
    tiles, titles = [], []
    for i in idxs:
        i = int(np.clip(i, 0, ov.n_bscans - 1))
        gc = ga_nat[i] if ga_nat is not None else None
        tiles.append(crop_draw(ov.vol[i], np.asarray(bm[i], float),
                               np.asarray(rpe_row[i], float) if rpe_row is not None else None, gc))
        nfire = int(gc.sum()) if gc is not None else -1
        titles.append(f"b{i}  GA-cols={nfire}")
    os.makedirs(OUT, exist_ok=True)
    panel = qv.panel(tiles, titles,
                     header=f"{subj} {eye}  band audit (PLEX {adv:.2f} mm²)  | red=BM  magenta=RPE-peak  "
                            f"cyan=OAC band(BM{OAC_BAND[0]:.0f}..{OAC_BAND[1]:.0f})  orange=slab(BM+{SLAB[0]:.0f}..{SLAB[1]:.0f})  "
                            f"red ticks=GA-called cols")
    out = os.path.join(OUT, f"{subj}_{eye}_bands.png")
    qv.save_rgb(out, panel)
    print(f"  wrote {out}", flush=True)


DEFAULT = [("NHAMD-003-016", "V2", "OD", [31, 34, 37, 40]),     # the FP, firing B-scans
           ("NHAMD-003-005", "V3", "OD", [44, 48, 52, 56]),     # true focal GA
           ("NHAMD-003-006", "V3", "OS", [44, 48, 52, 56])]     # clean control


def main():
    a = sys.argv[1:]
    if a:
        idxs = [int(x) for x in a[3:]] if len(a) > 3 else []
        jobs = [((a[0] if a[0].startswith("NHAMD") else "NHAMD-003-" + a[0]), a[1], a[2].upper(), idxs)]
    else:
        jobs = DEFAULT
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})", flush=True)
    for subj, visit, eye, idxs in jobs:
        try:
            render(subj, visit, eye, idxs)
        except Exception as e:                              # noqa: BLE001
            import traceback
            print(f"  FAILED {subj} {eye}: {e!r}"); traceback.print_exc()


if __name__ == "__main__":
    main()
