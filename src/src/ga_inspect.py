#!/usr/bin/env python
"""Show, on the actual B-scans, WHERE the detector calls GA -- so a clinician can judge if it is GA.

For each eye, run the production detector (DL BM, radial2), take its en-face GA footprint, map it back to
native B-scan A-scans (viewer.core.ga_native.enface_to_native), and render every firing B-scan full-res
with the called-GA columns marked (red span + top bar) and Bruch's membrane (yellow). The reader looks at
the RPE under the red span: RPE GONE (+ bright transmission below) = real GA; RPE PRESENT = false call.

016 OD is the suspect (PLEX 0, we call ~1.3 mm2); 005 OD is the gold true-GA reference (same overlay).
Run (repo root):  oct_env\\Scripts\\python.exe src\\ga_inspect.py
Out -> outputs/ga_inspect/<slug>/b####.png
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
from reader.core import e2e_source, oac_ga  # noqa: E402
from viewer.core import ga_native  # noqa: E402

AX = float(mp.AX)
OUT = os.path.join(OUT_DIR, "ga_inspect")

EYES = [
    ("NHAMD-003-016-V2", "NHAMD01N.E2E", "OD", "SUSPECT (PLEX 0, we call ~1.3 mm2 - is it GA?)"),
    ("NHAMD-003-005-V3", "NHAMD01N.E2E", "OD", "TRUE-GA REFERENCE (gold, PLEX 1.08, Dice 0.94)"),
]


def runs(maskrow):
    out, x, W = [], 0, len(maskrow)
    while x < W:
        if maskrow[x]:
            s = x
            while x < W and maskrow[x]:
                x += 1
            out.append((s, x - 1))
        else:
            x += 1
    return out


def draw(img, bm, peak, ga_cols, path, title):
    H, W = img.shape
    x = np.arange(W)
    fig, ax = plt.subplots(figsize=(W / 60, H / 60), dpi=120)
    ax.imshow(qv.norm8(img), cmap="gray", aspect="auto", interpolation="nearest")
    for s, e in runs(ga_cols):                                   # called-GA A-scans = red span + top bar
        ax.axvspan(s - 0.5, e + 0.5, color="red", alpha=0.16, lw=0)
        ax.plot([s, e], [6, 6], color="red", lw=3, solid_capstyle="butt")
    ax.plot(x, bm, color="yellow", lw=1.1, label="Bruch's membrane")
    ax.plot(x, peak, color="orange", lw=0.8, label="RPE peak (orange present = RPE there)")
    ax.plot([], [], color="red", lw=3, label=f"detector says GA here ({int(ga_cols.sum())} A-scans)")
    ax.set_xlim(0, W - 1); ax.set_ylim(H - 1, 0)
    ax.set_title(title, fontsize=7)
    ax.legend(loc="upper right", fontsize=5.5, framealpha=0.65)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for sub, fn, eye, note in EYES:
        slug = f"{sub[-6:].replace('-', '')}_{eye}"
        d = os.path.join(OUT, slug); os.makedirs(d, exist_ok=True)
        p = os.path.join(DATA_DIR, sub + "-SPECTRALIS", fn)
        raw = e2e_source.open_e2e(p)
        ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
        bm = bm_dl.segment_volume(ov.vol)
        P = oac_ga.prep(ov, bm, baseline="radial2")
        mask, area = oac_ga.footprint(P, 0.50)                   # en-face GA footprint
        ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, ov.n_bscans, ov.W).astype(bool)
        oac = mp.oac_volume(ov.vol)
        peak = mp.band_argmax_row(oac, bm, *mp.OAC_RPE_UM)
        counts = ga_nat.sum(axis=1)
        firing = [int(b) for b in np.argsort(counts)[::-1] if counts[b] > 0][:8]
        firing = sorted(firing)
        print(f"{slug}: area={area:.3f} mm2  firing B-scans (top, by #GA A-scans)={firing}", flush=True)
        for bi in firing:
            title = (f"{sub} {eye}  B-scan {bi}/{ov.n_bscans}  |  {note}  |  GA A-scans={int(counts[bi])}/{ov.W}"
                     f"  total area={area:.2f} mm2  (yellow=BM, orange=RPE peak, RED=detector's GA call)")
            draw(ov.vol[bi], bm[bi], peak[bi], ga_nat[bi], os.path.join(d, f"b{bi:04d}.png"), title)
        print(f"  wrote {len(firing)} -> {d}", flush=True)


if __name__ == "__main__":
    main()
