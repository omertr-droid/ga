#!/usr/bin/env python
"""FAST destripe bake-off on cached NATIVE maps (no E2E). Controls = flat truth, so any horizontal
stripe is pure artifact. Displays every variant at the SAME fixed window the panels use ([0.18,0.62])
so it matches what the user sees, scores residual banding, and renders so we can pick the winner.

Run: oct_env\\Scripts\\python.exe tune_destripe.py
"""
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import m3_projections as mp
import qcviz as qv

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
FEAT = os.path.join(OUT_DIR, "features")
DISP_LO, DISP_HI = 0.18, 0.62
CONTROLS = [("NHAMD-003-002-V2", "OD"), ("NHAMD-003-006-V3", "OS"), ("NHAMD-003-016-V2", "OD"),
            ("NHAMD-003-002-V2", "OS")]


def band_score(nat):
    rm = np.nanmean(nat, axis=1)
    return float(np.nanstd(rm - gaussian_filter1d(rm, 4)) / (np.nanstd(nat) + 1e-6))


def med_gain(n):
    return mp.destripe2d(n, 2.0, signed=False)


def slowmed(n, k):
    return median_filter(n, size=(k, 1), mode="nearest")


def fourier_notch(nat, kx_thr=0.03, ky_thr=0.10):
    """Remove horizontal stripes = spectral energy at low kx (constant along x) and high ky (fast along
    the slow axis). Zero that band, keep everything else (incl. low-ky large lesions)."""
    F = np.fft.rfft2(nat)
    ky = np.abs(np.fft.fftfreq(nat.shape[0]))[:, None]
    kx = np.fft.rfftfreq(nat.shape[1])[None, :]
    F[(kx < kx_thr) & (ky > ky_thr)] = 0
    return np.fft.irfft2(F, s=nat.shape).astype(np.float32)


VARIANTS = [
    ("raw", lambda n: n),
    ("med+gain", med_gain),
    ("slowmed3", lambda n: slowmed(n, 3)),
    ("med+gain+sm3", lambda n: slowmed(med_gain(n), 3)),
    ("fourier-notch", fourier_notch),
    ("notch+med", lambda n: med_gain(fourier_notch(n))),
]


def main():
    eyes = []
    for subject, eye in CONTROLS:
        p = os.path.join(FEAT, f"{subject}_{eye}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        if "f_trans_nat" in d:
            eyes.append((subject.replace("NHAMD-003-", "") + " " + eye, d["f_trans_nat"].astype(np.float32)))

    print("%-14s " % "variant" + "  ".join("%-10s" % tag for tag, _ in eyes))
    rows = []
    for tag, fn in VARIANTS:
        scores, tiles = [], []
        for etag, nat in eyes:
            out = fn(nat)
            scores.append(band_score(out))
            disp = qv.norm8(np.clip(out[::-1], DISP_LO, DISP_HI))         # fixed window = what panels show
            tiles.append(qv.label_tile(cv2.resize(qv.ensure_rgb(disp), (300, 300)), f"{etag} {tag} b={scores[-1]:.3f}"))
        print("%-14s " % tag + "  ".join("%-10.3f" % s for s in scores))
        h = tiles[0].shape[0]
        parts = []
        for i, t in enumerate(tiles):
            if i:
                parts.append(np.zeros((h, 6, 3), np.uint8))
            parts.append(t)
        rows.append(np.hstack(parts))
    grid = rows[0]
    for r in rows[1:]:
        grid = np.vstack([grid, np.full((6, grid.shape[1], 3), 90, np.uint8), r])
    qv.save_rgb(os.path.join(OUT_DIR, "tune_destripe.png"),
                qv.add_header(grid, "destripe bake-off (controls, native f_trans, FIXED window; lower b = flatter)"))
    print("\nwrote tune_destripe.png")


if __name__ == "__main__":
    main()
