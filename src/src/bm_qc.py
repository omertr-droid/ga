#!/usr/bin/env python
"""BM quality control — phone-friendly, big readable headers. Per device-BM eye: a header line
(eye, roughness, auto-flag) + the foveal and WORST B-scans with BM + slab drawn. Sorted worst-first,
split into batches of 6 eyes per image so the headlines are legible on a phone.

Run: oct_env\\Scripts\\python.exe bm_qc.py
"""
import csv
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import m2_bm
import m3_projections as mp
import qcviz as qv

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
TILE_W, TILE_H, PER_IMG = 380, 260, 6


def flag(score):
    if not np.isfinite(score):
        return "BAD (NaN BM)"
    return "GOOD" if score < 2.5 else ("SUSPECT" if score < 4 else "LIKELY BAD")


def bscan_tile(vol, bm, ilm, dev, i, cap):
    t = mp.bscan_bands(vol[i], bm[i], dev[i] if dev is not None else None,
                       ilm[i] if ilm is not None else None)
    t = cv2.resize(qv.ensure_rgb(t), (TILE_W, TILE_H))
    cv2.rectangle(t, (0, TILE_H - 20), (TILE_W, TILE_H), (0, 0, 0), -1)
    qv._text(t, cap, (6, TILE_H - 6), 0.5)
    return t


def card(vol, bm, ilm, dev, n, worst, header):
    row = np.hstack([bscan_tile(vol, bm, ilm, dev, n // 2, "foveal"),
                     np.zeros((TILE_H, 6, 3), np.uint8),
                     bscan_tile(vol, bm, ilm, dev, worst, f"worst B-scan {worst}/{n}")])
    hb = np.full((46, row.shape[1], 3), (45, 45, 45), np.uint8)
    qv._text(hb, header, (10, 32), 0.85, thick=2)
    return np.vstack([hb, row])


def main():
    by_sub = {}
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("qc_status") == "ok":
                by_sub.setdefault(r["subject"], []).append(r["eye"])

    cards = []
    for subject in sorted(by_sub):
        loaded = m2_bm.load_subject_layers(subject)
        for eye in by_sub[subject]:
            if eye not in loaded:
                continue
            vol, ilm, dev_bm = loaded[eye]
            if dev_bm is None:
                continue
            bm = m2_bm.fill_bm(dev_bm)
            ilmf = m2_bm.fill_bm(ilm) if ilm is not None else None
            n = len(bm)
            resid = bm - gaussian_filter(bm, sigma=(1.5, 5))
            score = float(resid.std())
            worst = int(np.argmax(np.nan_to_num(resid.std(axis=1), nan=1e9)))
            tag = subject.replace("NHAMD-003-", "")
            hdr = f"{tag} {eye}   roughness={score:.1f}px   [{flag(score)}]"
            cards.append((np.nan_to_num(score, nan=1e9), card(vol, bm, ilmf, dev_bm, n, worst, hdr)))
            print(f"  {tag} {eye} rough={score:.1f} [{flag(score)}]", flush=True)

    cards.sort(key=lambda c: -c[0])                  # worst (incl. NaN) first
    for k in range(0, len(cards), PER_IMG):
        batch = [c for _, c in cards[k:k + PER_IMG]]
        W = max(b.shape[1] for b in batch)
        batch = [np.pad(b, ((0, 0), (0, W - b.shape[1]), (0, 0))) for b in batch]
        grid = batch[0]
        for b in batch[1:]:
            grid = np.vstack([grid, np.full((8, W, 3), 90, np.uint8), b])
        qv.save_rgb(os.path.join(OUT_DIR, f"bm_qc_p{k // PER_IMG + 1}.png"), grid)
    print(f"\nwrote {(len(cards) + PER_IMG - 1) // PER_IMG} BM-QC images (bm_qc_p*.png)")


if __name__ == "__main__":
    main()
