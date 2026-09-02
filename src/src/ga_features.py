#!/usr/bin/env python
"""Compute + CACHE the two OCT GA features per eye, in the fovea-centred 6x6 mm frame:

  f_trans  transmission fraction  (hypertransmission; m3_projections.proj_transmit + destripe)
  f_rpe    RPE-loss cue           (Michelson contrast inner-retina vs RPE band; high = RPE lost)

E2E loading is the slow part, so we do it once and write features/<subj>_<eye>.npz (f_trans, f_rpe,
advRPE area, bm_source). validate_area.py then iterates the LOO-CV in seconds on these caches.

Run: oct_env\\Scripts\\python.exe ga_features.py [all | SUBJECT EYE ...]
"""
import csv
import json
import os
import sys

import numpy as np

import bm as bmseg
import m2_bm
import m3_projections as mp

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
FEAT = os.path.join(OUT_DIR, "features")
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")


def native_for(vol, ilm, dev_bm):
    """The PRE-destripe native en-faces (rows=B-scan, cols=A-scan). Caching these (the slow E2E part)
    lets redestripe.py re-apply any destripe/gate variant in seconds without reloading the E2E.
      t_nat = transmission fraction; r_nat = RPE-loss (signed, correlated w/ transmission);
      p_nat = RPE-present prominence (transmission-INDEPENDENT, above BM) -> the specificity gate."""
    bmrow, ilmrow = m2_bm.fill_bm(dev_bm), m2_bm.fill_bm(ilm)
    t_nat = mp.proj_transmit_ilm(vol, ilmrow, bmrow)
    r_nat = mp.proj_rpe_loss_ilm(vol, ilmrow, bmrow)
    p_nat = mp.proj_rpe_present_ilm(vol, ilmrow, bmrow)
    return t_nat.astype(np.float32), r_nat.astype(np.float32), p_nat.astype(np.float32)


def finish(t_nat, r_nat, p_nat, fov):
    """Native -> destriped 6 mm features. Edit the destripe/gate here (or in redestripe.py) and re-run.
    f_trans positive (median+gain); f_rpe signed (median); f_gated = transmission x RPE-gone gate (the
    deliverable: hypertransmission ONLY where the RPE is actually absent)."""
    f_trans = mp.to_6mm(mp.destripe2d(t_nat, signed=False), fov)
    f_rpe = mp.to_6mm(mp.destripe2d(r_nat, signed=True), fov)
    f_gated = mp.gated_feature(t_nat, p_nat, fov)
    return (np.nan_to_num(f_trans, nan=0.0), np.nan_to_num(f_rpe, nan=0.0), np.nan_to_num(f_gated, nan=0.0))


def ok_eyes():
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("qc_status") == "ok"]
    return [(r["subject"], r["eye"], float(r.get("advRPE_area_mm2") or 0.0)) for r in rows]


def fov_of(subject, eye):
    with open(os.path.join(COH, subject, "meta.json")) as f:
        fov = json.load(f)["eyes"][eye]["fov_mm"]
    return [float(fov[0]), float(fov[1])]


def main():
    os.makedirs(FEAT, exist_ok=True)
    args = sys.argv[1:]
    eyes = ok_eyes()
    if args and args[0].lower() != "all":
        want = {(args[i] if args[i].startswith("NHAMD") else "NHAMD-003-" + args[i], args[i + 1])
                for i in range(0, len(args) - 1, 2)}
        eyes = [e for e in eyes if (e[0], e[1]) in want]

    by_sub = {}
    for s, e, a in eyes:
        by_sub.setdefault(s, []).append((e, a))
    n = 0
    for subject in sorted(by_sub):
        try:
            loaded = m2_bm.load_subject_layers(subject)
        except Exception as ex:
            print(f"[{subject}] LOAD ERROR {type(ex).__name__}: {ex}", flush=True)
            continue
        for eye, area in by_sub[subject]:
            if eye not in loaded:
                print(f"[{subject} {eye}] no 30deg volume", flush=True)
                continue
            vol, ilm, dev_bm = loaded[eye]
            if ilm is None or dev_bm is None:                  # good-BM (device ILM+BM) eyes only
                print(f"  skip {subject} {eye} (no device ILM/BM)", flush=True)
                continue
            fov = fov_of(subject, eye)
            t_nat, r_nat, p_nat = native_for(vol, ilm, dev_bm)
            f_trans, f_rpe, f_gated = finish(t_nat, r_nat, p_nat, fov)
            np.savez_compressed(os.path.join(FEAT, f"{subject}_{eye}.npz"),
                                f_trans=f_trans.astype(np.float32), f_rpe=f_rpe.astype(np.float32),
                                f_gated=f_gated.astype(np.float32),
                                f_trans_nat=t_nat, f_rpe_nat=r_nat, f_pres_nat=p_nat,
                                fov=np.asarray(fov, np.float32),
                                area=np.float32(area), bm_source="device", subject=subject, eye=eye)
            n += 1
            print(f"  cached {subject} {eye}  area={area:.2f}", flush=True)
    print(f"\nwrote {n} feature caches -> features/")


if __name__ == "__main__":
    main()
