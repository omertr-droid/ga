#!/usr/bin/env python
"""Legacy-vs-NOISE-FLOOR GA footprint grids for the BM-validated reader eyes beyond 005 OD / 008 OS.

For each reader-validated eye it computes the OAC GA footprint with the legacy estimator and with the
noise-floor estimator fix (oac_volume floor=, via oac_ga.prep(noise_floor=True)), annotated OUR area
vs the PLEX (advRPE) reference. Several of these are controls -> this doubles as a SPECIFICITY check
(the noise floor must keep their area ~0). Reuses the helpers from src/oac_experiments.py.

Run (from repo root):  oct_env\\Scripts\\python.exe src\\oac_floor_grid.py
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import qcviz as qv
from oac_experiments import CORR_DIR, FRAC, dice, foot_tile, load_gold, plex_tile, resolve
from paths import OUT_DIR
from reader.core import e2e_source, layers, oac_ga
from reader.core import projection as proj
from reader.core.layer_store import JsonSidecarLayerStore

OUT = os.path.join(OUT_DIR, "oac_experiments")

# BM-validated reader eyes beyond the two already shown (005 OD, 008 OS). 002 OD is PARTIAL (39/97
# validated; the rest device BM). Ordered GA-first then controls.
EYES = [
    ("NHAMD-003-015-V3", "V3", "OD"),
    ("NHAMD-003-006-V3", "V3", "OD"),
    ("NHAMD-003-006-V3", "V3", "OS"),
    ("NHAMD-003-012-V3", "V3", "OD"),
    ("NHAMD-003-002-V2", "V2", "OD"),
]


def run_eye(subject, visit, eye):
    row, e2e_path = resolve(subject, visit, eye)
    subject, eye = row["subject"], row["eye"].upper()
    plex_area = float(row["advRPE_area_mm2"])
    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    ilm, bm = layers.effective_surfaces(ov, JsonSidecarLayerStore(CORR_DIR))
    gold = load_gold(subject, eye, ov.n_bscans, ov.fov_mm)

    out = {}
    for label, kw in [("legacy", {}), ("floor", {"noise_floor": True})]:
        P = oac_ga.prep(ov, bm, ilm=ilm, **kw)
        mask, area = oac_ga.footprint(P, FRAC)
        out[label] = (P, mask, area, dice(mask, gold) if gold is not None else None)
    la, fa = out["legacy"][2], out["floor"][2]
    print(f"  {subject} {eye:3}  legacy {la:6.3f}  floor {fa:6.3f}  PLEX {plex_area:6.2f}"
          + (f"  Dice {out['legacy'][3]:.3f}->{out['floor'][3]:.3f}" if gold is not None else ""), flush=True)

    tiles, titles = [], []
    for label in ("legacy", "floor"):
        P, mask, area, d = out[label]
        tiles.append(foot_tile(P["rpe6"], mask))
        titles.append(f"{label} {area:.2f}|PLEX {plex_area:.2f}" + (f" D{d:.2f}" if d is not None else ""))
    tiles.append(plex_tile(subject, eye, plex_area))
    titles.append(f"PLEX advRPE {plex_area:.2f}")
    return qv.panel(tiles, titles, mm_per_px=proj.ENFACE_MMPP, bar_on=[True, True, False],
                    header=f"{subject} {eye}  legacy vs NOISE-FLOOR  |  PLEX {plex_area:.2f} mm2  "
                           f"(area delta {fa - la:+.2f})")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [run_eye(*e) for e in EYES]
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    montage = rows[0]
    for r in rows[1:]:
        montage = np.vstack([montage, np.full((8, W, 3), 40, np.uint8), r])
    path = os.path.join(OUT, "_MONTAGE_floor_readereyes.png")
    qv.save_rgb(path, montage)
    print("wrote", path)


if __name__ == "__main__":
    main()
