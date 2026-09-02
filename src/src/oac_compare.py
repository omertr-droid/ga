#!/usr/bin/env python
"""Side-by-side compare of the QUADRATIC vs LINEAR healthy-baseline for the OAC GA detector, per eye, so
the operator can pick. For each eye renders [quadratic | linear] overlays (RPE-loss map + green cRORA
footprint fill), labelled with area (+ Dice vs the in-frame gold where it exists).

Run: oct_env\\Scripts\\python.exe src\\oac_compare.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import qcviz as qv
from reader.core import e2e_source, layers, oac_ga
from reader.core import projection as proj
from reader.core.layer_store import JsonSidecarLayerStore
from oac_area import CORR_DIR, OUT, load_gold_native, resolve


def overlay_rgb(rpe6, mask):
    rgb = qv.ensure_rgb(qv.norm8(np.nan_to_num(np.asarray(rpe6, np.float32)))).astype(np.float32)
    m = np.asarray(mask, bool)
    rgb[m] = 0.45 * rgb[m] + 0.55 * np.array([0, 200, 0], np.float32)
    return qv.draw_contour(rgb.astype(np.uint8), m, color=(0, 255, 0), thick=1)


def dice(mask, gold_mask):
    if gold_mask is None:
        return None
    return 2 * float((mask & gold_mask).sum()) / (float(mask.sum()) + float(gold_mask.sum()) + 1e-9)


def do_eye(subject, visit, eye):
    row, e2e = resolve(subject, visit, eye)
    eye = row["eye"].upper()
    adv = float(row["advRPE_area_mm2"])
    raw = e2e_source.open_e2e(e2e)
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    _, bm = layers.effective_surfaces(ov, JsonSidecarLayerStore(CORR_DIR))
    gold = load_gold_native(row["subject"], eye, ov.n_bscans)
    gold_mask = (proj.to_enface(gold, ov.fov_mm) > 0.5) if gold is not None else None
    gold_area = float(gold_mask.sum()) * proj.ENFACE_MMPP ** 2 if gold_mask is not None else None

    tiles, titles = [], []
    for od, name in [(2, "QUADRATIC (005-clean)"), (1, "LINEAR (008-clean)")]:
        P = oac_ga.prep(ov, bm, trend_order=od)
        mask, area = oac_ga.footprint(P, 0.5)
        d = dice(mask, gold_mask)
        tiles.append(overlay_rgb(P["rpe6"], mask))
        titles.append(f"{name}  {area:.2f} mm2" + (f"  Dice {d:.2f}" if d is not None else ""))
        print(f"  {row['subject']} {eye} {name}: area {area:.2f}" + (f" Dice {d:.3f}" if d is not None else ""))
    hdr = (f"{row['subject']} {eye}   advRPE {adv:.2f} mm2"
           + (f"   gold {gold_area:.2f} mm2" if gold_area is not None else "   (no gold)"))
    out = os.path.join(OUT, f"{row['subject']}_{eye}_compare.png")
    qv.save_rgb(out, qv.panel(tiles, titles, header=hdr, mm_per_px=proj.ENFACE_MMPP))
    print(f"  wrote {out}")
    return out


def main():
    """No args -> the two BM-corrected eyes (005 OD, 008 OS). `SUBJECT VISIT EYE` -> any eye (run it on a
    newly BM-corrected eye to compare the quadratic vs linear baseline before picking)."""
    os.makedirs(OUT, exist_ok=True)
    a = sys.argv[1:]
    eyes = [tuple(a[:3])] if len(a) >= 3 else [("NHAMD-003-005", "V3", "OD"), ("NHAMD-003-008", "V1", "OS")]
    for s, v, e in eyes:
        do_eye(s, v, e)


if __name__ == "__main__":
    main()
