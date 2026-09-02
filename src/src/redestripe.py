#!/usr/bin/env python
"""FAST destripe iteration: read the cached PRE-destripe native maps (features/*.npz: f_trans_nat,
f_rpe_nat, fov), re-apply the destripe (m3_projections.destripe2d -- edit it or its params here), and
overwrite the 6 mm f_trans/f_rpe in place. No E2E reload, runs in seconds, so destripe variants iterate
instantly. After running, regenerate eye_panels.py / rpe_review.py to view.

Run: oct_env\\Scripts\\python.exe redestripe.py
"""
import glob
import os

import numpy as np

import m3_projections as mp

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
FEAT = os.path.join(OUT_DIR, "features")

# --- the destripe to apply (edit here, re-run, view) ---
SLOW_SIGMA = 2.0


def main():
    n = 0
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        d = dict(np.load(p, allow_pickle=True))
        if "f_trans_nat" not in d or "fov" not in d:
            print(f"  skip {os.path.basename(p)} (no native maps; re-run ga_features.py all)", flush=True)
            continue
        fov = [float(v) for v in d["fov"]]
        f_trans = mp.to_6mm(mp.destripe2d(d["f_trans_nat"], SLOW_SIGMA, signed=False), fov)
        f_rpe = mp.to_6mm(mp.destripe2d(d["f_rpe_nat"], SLOW_SIGMA, signed=True), fov)
        d["f_trans"] = np.nan_to_num(f_trans, nan=0.0).astype(np.float32)
        d["f_rpe"] = np.nan_to_num(f_rpe, nan=0.0).astype(np.float32)
        if "f_pres_nat" in d:                                  # gated = transmission x RPE-gone gate
            f_gated = mp.gated_feature(d["f_trans_nat"], d["f_pres_nat"], fov, SLOW_SIGMA)
            d["f_gated"] = np.nan_to_num(f_gated, nan=0.0).astype(np.float32)
        np.savez_compressed(p, **d)
        n += 1
        print(f"  redestriped {d['subject']} {d['eye']}", flush=True)
    print(f"\nredestriped {n} eyes (slow_sigma={SLOW_SIGMA}, med+gain trans / med rpe)")


if __name__ == "__main__":
    main()
