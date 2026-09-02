#!/usr/bin/env python
"""RPE-loss improvement experiments + annotated comparison grids (deep-dive follow-up).

Runs, on the two BM-trustworthy eyes (005 OD focal/gold, 008 OS large), the three changes from the
RPE-loss deep dive and writes grids so the change is visible at a glance:

  (b) PEAK-anchored OAC band   -- sample OAC around the per-A-scan RPE peak (mp.band_peak_oac), not a
                                  fixed BM offset, so a diving BM under GA can't read 'RPE present'.
  (c) NOISE-FLOOR OAC          -- subtract a per-B-scan vitreous floor in the estimator (oac_volume).
  (a) RPE-PRESENT recovery     -- does the EXISTING above-BM channel (proj_rpe_present_ilm / rpe_surface
                                  prominence, both blind to sub-BM hypertransmission) recover the BM-dive
                                  false-negative columns the fixed-band OAC misses?  (diagnostic)
  (c) RADIAL-TREND plot        -- is the macula->periphery falloff instrumental (flattens under the
                                  noise-floor correction) or anatomical (persists)?

Each variant's GA footprint is annotated with OUR area vs the PLEX (advRPE) reference area (and Dice vs
the in-frame gold where it exists). OCT-only is preserved: PLEX is a side reference tile, never blended.

Run (from repo root):
  oct_env\\Scripts\\python.exe src\\oac_experiments.py
  oct_env\\Scripts\\python.exe src\\oac_experiments.py NHAMD-003-005-V3 V3 OD   # one eye
"""
import csv
import os
import re
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import m3_projections as mp
import qcviz as qv
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR
from reader.core import e2e_source, layers, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj
from reader.core.layer_store import JsonSidecarLayerStore

CORR_DIR = os.path.join(_REPO, "reader", "data_store", "corrections")
COH = os.path.join(_REPO, "cohort")
OUT = os.path.join(OUT_DIR, "oac_experiments")
MMPP2 = oac_ga.MMPP2
FRAC = 0.50

# (subject key, visit, eye) -- the BM-validated eyes that can non-circularly judge a SIGNAL change.
EYES = [("NHAMD-003-005-V3", "V3", "OD"), ("NHAMD-003-008-V1", "V1", "OS")]

# label, prep kwargs (defaults reproduce the validated fixed-band/no-floor path)
VARIANTS = [
    ("legacy", {}),
    ("peak", {"rpe_band": "peak"}),
    ("floor", {"noise_floor": True}),
    ("peak+floor", {"rpe_band": "peak", "noise_floor": True}),
]


def resolve(subject, visit, eye):
    want = subject if re.search(r"-V\d+$", subject) else f"{subject}-{visit}"
    eye = eye.upper()
    with open(os.path.join(RESULTS_DIR, "bm_worklist.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["subject"] == want and r["eye"].upper() == eye:
                return r, os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    raise SystemExit(f"{want} {eye} not in bm_worklist.csv")


def load_gold(subject, eye, n_bscans, fov):
    """In-frame gold GA mask (en-face) from the exported per-B-scan labels, or None."""
    lab = os.path.join(OUT_DIR, "ga_bscan_dataset", "labels")
    rows, found, W = [], False, None
    for i in range(n_bscans):
        p = os.path.join(lab, f"{subject}_{eye}_b{i:04d}.png")
        if os.path.exists(p):
            found = True
            col = (np.array(Image.open(p)) == 1).any(axis=0)
            W = len(col)
            rows.append(col)
        else:
            rows.append(None)
    if not found:
        return None
    native = np.array([(r if r is not None else np.zeros(W, bool)) for r in rows], np.float32)
    return proj.to_enface(native, fov) > 0.5


def plex_tile(subject, eye, plex_area):
    """advRPE SubRPE en-face with its GA outline (cyan) -- the reference VIEW (not registered to ours)."""
    edir = os.path.join(COH, subject, eye)
    sub = cv2.imread(os.path.join(edir, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
    gam = cv2.imread(os.path.join(edir, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
    if sub is None:
        return np.zeros((256, 256, 3), np.uint8)
    tile = qv.ensure_rgb(sub)
    if gam is not None:
        tile = qv.draw_contour(tile, gam > 127, color=(0, 220, 255), thick=2)
    return tile


def foot_tile(rpe6, mask):
    """RPE-loss en-face (dark = GA) with the green cRORA footprint (fill + outline)."""
    rgb = qv.ensure_rgb(qv.norm8(np.nan_to_num(np.asarray(rpe6, np.float32)))).astype(np.float32)
    m = np.asarray(mask, bool)
    rgb[m] = 0.5 * rgb[m] + 0.5 * np.array([0, 200, 0], np.float32)
    return qv.draw_contour(rgb.astype(np.uint8), m, color=(0, 255, 0), thick=1)


def fpfn_tile(rpe6, mask, gold):
    base = (qv.norm8(np.nan_to_num(rpe6)).astype(np.float32) * 0.45).astype(np.uint8)
    rgb = qv.ensure_rgb(base)
    rgb[mask & gold] = (0, 230, 0)        # TP
    rgb[mask & ~gold] = (255, 40, 40)     # FP
    rgb[~mask & gold] = (40, 110, 255)    # FN
    return rgb


def dice(a, b):
    return 2 * float((a & b).sum()) / (float(a.sum()) + float(b.sum()) + 1e-9)


def run_eye(subject, visit, eye):
    row, e2e_path = resolve(subject, visit, eye)
    subject, eye = row["subject"], row["eye"].upper()
    plex_area = float(row["advRPE_area_mm2"])
    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    ilm, bm = layers.effective_surfaces(ov, JsonSidecarLayerStore(CORR_DIR))
    gold = load_gold(subject, eye, ov.n_bscans, ov.fov_mm)
    print(f"\n=== {subject} {eye}  n={ov.n_bscans} fov={tuple(round(f,2) for f in ov.fov_mm)} "
          f"PLEX={plex_area:.2f}mm2  gold={'yes' if gold is not None else 'no'} ===")

    # --- the four variants ---
    preps, results = {}, {}
    for label, kw in VARIANTS:
        P = oac_ga.prep(ov, bm, ilm=ilm, **kw)
        mask, area = oac_ga.footprint(P, FRAC)
        preps[label] = P
        results[label] = (mask, area)
        d = dice(mask, gold) if gold is not None else None
        print(f"  {label:10}  area = {area:6.3f} mm2  (PLEX {plex_area:.2f})"
              + (f"  Dice = {d:.3f}" if d is not None else ""))

    # --- footprint comparison grid (the headline: annotated area vs PLEX) ---
    tiles, titles = [], []
    for label, _ in VARIANTS:
        mask, area = results[label]
        d = dice(mask, gold) if gold is not None else None
        tiles.append(foot_tile(preps[label]["rpe6"], mask))
        titles.append(f"{label}  {area:.2f}|PLEX {plex_area:.2f}" + (f" D{d:.2f}" if d is not None else ""))
    tiles.append(plex_tile(subject, eye, plex_area))
    titles.append(f"PLEX advRPE {plex_area:.2f} mm2 (ref)")
    grid = qv.panel(tiles, titles, mm_per_px=proj.ENFACE_MMPP,
                    header=f"{subject} {eye}  RPE-loss variants @ frac={FRAC}  |  OCT area vs PLEX {plex_area:.2f} mm2",
                    bar_on=[True] * len(VARIANTS) + [False])
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_grid.png"), grid)

    # --- (a) RPE-PRESENT recovery diagnostic (existing above-BM channel vs fixed-band OAC) ---
    P0 = preps["legacy"]
    core, b_leg = P0["core"], results["legacy"][0]
    pres6 = proj.to_enface(mp.destripe2d(mp.proj_rpe_present_ilm(ov.vol, ilm, bm), signed=False), ov.fov_mm)
    _, prom = mp.rpe_surface(ov.vol, bm)
    prom6 = proj.to_enface(prom, ov.fov_mm)
    gone = mp.rpe_gone_gate(gaussian_filter(pres6, mp.GATE_SMOOTH_PX))           # 1 = RPE gone
    b_pres = fp.crora(((gone > 0.5) & core), 250.0)
    recov = b_pres & ~b_leg                                                      # GA the above-BM channel adds
    ov_rgb = qv.ensure_rgb(qv.norm8(np.nan_to_num(P0["rpe6"])))
    ov_rgb = qv.draw_contour(ov_rgb, b_leg, color=(0, 255, 0), thick=1)          # legacy footprint
    ov_rgb[recov] = (255, 0, 255)                                                # recovered (magenta)
    diag = qv.panel(
        [qv.norm8(np.nan_to_num(P0["rpe6"])), qv.norm8(np.nan_to_num(prom6)),
         qv.norm8(np.nan_to_num(gone)), ov_rgb],
        ["OAC RPE-loss (dark=GA)", "RPE-surf prominence (dark=gone)",
         "rpe_gone weight (bright=gone)",
         f"green=legacy {results['legacy'][1]:.2f}  magenta=recovered {float(recov.sum())*MMPP2:.2f}"],
        mm_per_px=proj.ENFACE_MMPP,
        header=f"{subject} {eye}  (a) does the above-BM RPE-present channel recover BM-dive columns?  "
               f"rpe_present GA={float(b_pres.sum())*MMPP2:.2f} mm2")
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_diag.png"), diag)

    # --- (c) radial trend: healthy-column RPE-loss vs radius, legacy vs noise-floor ---
    H, W = P0["rpe6"].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2.0) / W) ** 2 + ((yy - H / 2.0) / H) ** 2)
    detected = b_leg | results["peak"][0] | results["floor"][0]
    if gold is not None:
        detected = detected | gold
    healthy = core & ~detected
    edges = np.linspace(0, float(r[core].max()), 13)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, color in [("legacy", "tab:blue"), ("floor", "tab:orange")]:
        m = preps[label]["rpe6"]
        prof = [float(np.nanmean(m[healthy & (r >= lo) & (r < hi)]))
                if (healthy & (r >= lo) & (r < hi)).sum() > 20 else np.nan
                for lo, hi in zip(edges[:-1], edges[1:])]
        prof = np.array(prof)
        ax[0].plot(ctr, prof, "-o", color=color, label=label, ms=4)
        c0 = prof[np.isfinite(prof)][0] if np.isfinite(prof).any() else 1.0
        ax[1].plot(ctr, prof / (c0 + 1e-9), "-o", color=color, label=label, ms=4)
    ax[0].set(title="healthy RPE-loss vs radius (raw)", xlabel="normalised radius", ylabel="mean OAC")
    ax[1].set(title="normalised to centre (shape)", xlabel="normalised radius", ylabel="relative")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"{subject} {eye}  (c) is the macula->periphery falloff instrumental (flattens) "
                 f"or anatomical (persists)?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{subject}_{eye}_radial.png"), dpi=110)
    plt.close(fig)

    return grid


def main():
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    eyes = [(args[0], args[1], args[2])] if len(args) >= 3 else EYES
    grids = []
    for subject, visit, eye in eyes:
        grids.append(run_eye(subject, visit, eye))
    if len(grids) > 1:
        W = max(g.shape[1] for g in grids)
        padded = [np.pad(g, ((0, 0), (0, W - g.shape[1]), (0, 0))) for g in grids]
        montage = padded[0]
        for g in padded[1:]:
            montage = np.vstack([montage, np.full((8, W, 3), 40, np.uint8), g])
        qv.save_rgb(os.path.join(OUT, "_MONTAGE_footprints.png"), montage)
    print(f"\nwrote grids -> {OUT}")


if __name__ == "__main__":
    main()
