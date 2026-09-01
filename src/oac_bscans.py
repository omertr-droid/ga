#!/usr/bin/env python
"""Per-B-scan agreement view for the OAC GA detector vs the exported gold label, to understand false
POSITIVES (OAC calls GA, gold doesn't) and false NEGATIVES (gold GA, OAC misses).

Renders FULL B-scans (generous crop, no vertical stretch -- recognizable, unlike a tight RPE strip)
with the corrected BM (yellow) and per-column bars: gold GA (cyan), FP = OAC&~gold (orange),
FN = gold&~OAC (red), OAC call (green). The OAC call is the NATIVE per-A-scan RPE-loss (mean OAC above
BM < frac*baseline); the en-face footprint additionally smooths + margin-erodes + cRORAs, so it removes
the isolated/edge FP this native view shows -- i.e. this OVER-shows FP relative to the final footprint.

Run: oct_env\\Scripts\\python.exe src\\oac_bscans.py [SUBJECT VISIT EYE] [fp|fn] [idx ...]
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import m3_projections as mp
import qcviz as qv
from reader.core import e2e_source, layers
from reader.core.layer_store import JsonSidecarLayerStore
from oac_area import CORR_DIR, OUT, load_gold_native, resolve

AX = mp.AX
RPE_BASE_PCT, RPE_FRAC, REDUCER = 95.0, 0.45, "mean"     # match oac_area's current detector


def render(cb, bmc, gold_cols, oac_cols, idx):
    """Full-B-scan tile: BM line (yellow) + column bars gold(cyan)/FP(orange)/FN(red)/OAC(green)."""
    Hc, W = cb.shape
    fp = oac_cols & ~gold_cols
    fn = gold_cols & ~oac_cols
    rgb = qv.ensure_rgb(qv.norm8(cb))
    for x in range(W):
        b = bmc[x]
        if np.isfinite(b):
            yb = int(round(b))
            if 0 <= yb < Hc:
                rgb[max(0, yb - 1):min(Hc, yb + 1), x] = (255, 255, 0)    # BM (yellow)
        if gold_cols[x]:
            rgb[2:9, x] = (0, 255, 255)          # gold GA (cyan)
        if fp[x]:
            rgb[9:16, x] = (255, 150, 0)         # FP = OAC & ~gold (orange)
        if fn[x]:
            rgb[16:23, x] = (255, 40, 40)        # FN = gold & ~OAC (red)
        if oac_cols[x]:
            rgb[Hc - 9:Hc - 2, x] = (0, 255, 0)  # OAC call (green)
    return qv.label_tile(rgb, f"b{idx:03d}  gold={int(gold_cols.sum())} oac={int(oac_cols.sum())} "
                              f"FP={int(fp.sum())} FN={int(fn.sum())}")


def main():
    a = sys.argv[1:]
    mode = "fn" if "fn" in a else "fp"
    a = [t for t in a if t not in ("fp", "fn")]
    idxs = [int(t) for t in a if t.isdigit()]
    sve = [t for t in a if not t.isdigit()]
    subject, visit, eye = (sve + ["NHAMD-003-005", "V3", "OD"][len(sve):])[:3]

    row, e2e = resolve(subject, visit, eye)
    eye = row["eye"].upper()
    raw = e2e_source.open_e2e(e2e)
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    lstore = JsonSidecarLayerStore(CORR_DIR)
    _, bm = layers.effective_surfaces(ov, lstore)
    corr = set(lstore.corrected_indices(ov.eid, eye))
    gold = load_gold_native(row["subject"], eye, ov.n_bscans)
    if gold is None:
        gold = np.zeros((ov.n_bscans, ov.W), np.float32)   # no gold (e.g. 008) -> show the OAC call only
    goldb = gold > 0.5
    rpe = mp.destripe2d(mp.band(mp.oac_volume(ov.vol), bm, *mp.OAC_RPE_UM, REDUCER), signed=False)
    oac_flag = rpe < RPE_FRAC * (np.nanpercentile(rpe, RPE_BASE_PCT) + 1e-6)

    fp = {i: int((oac_flag[i] & ~goldb[i]).sum()) for i in range(ov.n_bscans)}
    fn = {i: int((goldb[i] & ~oac_flag[i]).sum()) for i in range(ov.n_bscans)}
    score = fp if mode == "fp" else fn
    pick = sorted(idxs) if idxs else sorted(sorted(range(ov.n_bscans), key=lambda i: -score[i])[:6])

    print(f"{row['subject']} {eye}: top-{mode} B-scans  (bm gold oac FP FN):")
    for i in pick:
        print(f"  b{i:03d}: bm={'CORR' if i in corr else 'dev '} gold={int(goldb[i].sum()):3d} "
              f"oac={int(oac_flag[i].sum()):3d} FP={fp[i]:3d} FN={fn[i]:3d}")

    bmp = bm[pick]
    lo = max(0, int(np.nanmin(bmp)) - int(round(500 / AX)))    # generous crop: whole retina stays visible
    hi = min(ov.H, int(np.nanmax(bmp)) + int(round(280 / AX)))
    tiles = [render(ov.vol[i][lo:hi], bm[i] - lo, goldb[i], oac_flag[i], i) for i in pick]
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{row['subject']}_{eye}_bscans_{mode}.png")
    qv.save_rgb(out, qv.montage(tiles, cols=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
