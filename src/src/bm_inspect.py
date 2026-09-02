#!/usr/bin/env python
"""Visual BM + DRUSEN audit on the problematic false-positive control(s) (016 OD; 009 OS for the record).

NO ILM is drawn (ILM is not used by the area pipeline and is not accurate). For each rendered B-scan:
  - DL BM (yellow)            = Bruch's membrane (the production anchor)
  - RPE peak row (red)        = where the actual RPE reflectivity peaks (OAC argmax in the band above BM).
                               DRUSEN -> red sits WELL ABOVE yellow (RPE present but lifted);
                               true GA -> red hugs yellow / vanishes (RPE gone).
  - sub-BM hyper slab (green) = the BM+130..250um window that fires the 2nd (hypertransmission) criterion
  - blue ticks (top)          = columns flagged as drusen (RPE->BM elevation > ELEV_UM um)
If the green slab + the false GA call sit under columns where red is high above yellow (blue ticks), the
detector is calling DRUSEN as GA -> the fix is an RPE-elevation exclusion, not the hyper floor.

Run (repo root):  oct_env\\Scripts\\python.exe src\\bm_inspect.py
Out -> outputs/bm_inspect/<slug>/b####.png (full-res, one B-scan per file).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OCT_BM_DL", "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bm_dl  # noqa: E402
import m3_projections as mp  # noqa: E402
import qcviz as qv  # noqa: E402
from paths import DATA_DIR, OUT_DIR  # noqa: E402
from reader.core import e2e_source  # noqa: E402

AX = float(mp.AX)                 # axial um/px
ELEV_UM = 30.0                    # column flagged "drusen" if RPE peak sits > this far above BM
OUT = os.path.join(OUT_DIR, "bm_inspect")

EYES = [
    ("NHAMD-003-016-V2", "NHAMD01N.E2E", "OD"),
    ("NHAMD-003-009-V2", "NHAMD02N.E2E", "OS"),
]


def draw_bscan(img, dlbm, peak_row, elev_um, path, title):
    H, W = img.shape
    x = np.arange(W)
    fig, ax = plt.subplots(figsize=(W / 70, H / 70), dpi=110)
    ax.imshow(qv.norm8(img), cmap="gray", aspect="auto", interpolation="nearest")
    s0, s1 = dlbm + 130.0 / AX, dlbm + 250.0 / AX
    ax.fill_between(x, s0, s1, color="lime", alpha=0.16, lw=0, label="sub-BM hyper slab")
    ax.plot(x, dlbm, color="yellow", lw=1.2, label="DL BM (Bruch's)")
    ax.plot(x, peak_row, color="red", lw=0.9, label="RPE peak (red>>yellow = drusen lift)")
    drus = elev_um > ELEV_UM
    if drus.any():                                          # blue ticks: columns read as drusen
        ax.plot(x[drus], np.full(drus.sum(), 4), "s", color="deepskyblue", ms=1.5,
                label=f"drusen cols (elev>{ELEV_UM:.0f}um)")
    ax.set_xlim(0, W - 1); ax.set_ylim(H - 1, 0)
    ax.set_title(title, fontsize=7)
    ax.legend(loc="upper right", fontsize=5, framealpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for sub, fn, eye in EYES:
        slug = f"{sub[-6:].replace('-', '')}_{eye}"
        d = os.path.join(OUT, slug); os.makedirs(d, exist_ok=True)
        p = os.path.join(DATA_DIR, sub + "-SPECTRALIS", fn)
        raw = e2e_source.open_e2e(p)
        ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
        dlbm = bm_dl.segment_volume(ov.vol)
        oac = mp.oac_volume(ov.vol)
        peak = mp.band_argmax_row(oac, dlbm, *mp.OAC_RPE_UM)          # RPE/OAC peak row above BM
        elev_um = np.clip((dlbm - peak) * AX, 0.0, None)             # how far RPE sits above BM
        from m3_slab import hyper_enface
        score = np.nanmean(hyper_enface(ov.vol, dlbm), axis=1)
        order = np.argsort(score)[::-1]
        pick = sorted(set(list(order[:6]) + [ov.n_bscans // 2]))
        med_elev = float(np.median(elev_um))
        print(f"{slug}: n={ov.n_bscans} brightest-slab={list(order[:6])} median_elev={med_elev:.1f}um "
              f"drusen_col_frac={float((elev_um > ELEV_UM).mean()):.2f}", flush=True)
        for bi in pick:
            title = (f"{sub} {eye}  B-scan {bi}/{ov.n_bscans}  slab={score[bi]:.1f}  "
                     f"drusen-cols={int((elev_um[bi] > ELEV_UM).sum())}/{ov.W}  "
                     f"(yellow=BM red=RPE-peak green=hyper-slab blue=drusen)")
            draw_bscan(ov.vol[bi], dlbm[bi], peak[bi], elev_um[bi],
                       os.path.join(d, f"b{bi:04d}.png"), title)
        print(f"  wrote {len(pick)} -> {d}", flush=True)


if __name__ == "__main__":
    main()
