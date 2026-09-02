#!/usr/bin/env python
"""E1.1 -- multifocal / nascent-GA RECOVERY experiment, SPECIFICITY-GUARDED (workflow ga-error-experiments).

The spatial audit found our genuine misses are small MULTIFOCAL / nascent lesions (001 OD, 026) where each
focus falls under the cRORA 250um size gate and/or is erased by the complete-loss (min_depth) gate -- while
our over-calls are ~zero. The obvious levers to recover them (lower the size floor, relax complete-loss,
bridge proximate foci) are the SAME levers that would un-reject the PLEX over-call eyes (014/010/006 OD --
where min_depth CORRECTLY rejects incomplete loss). So the real question, answerable BEFORE any gold exists:

    is there a knob that recovers the TARGET multifocal eyes WITHOUT lighting up the GUARD eyes
    (PLEX-false-positive eyes + no-GA controls)?

This sweeps footprint variants and reports area (mm2) per eye, grouped target / guard / clean, so the
guard column is the pass/fail. A variant that moves a guard eye off ~0 is DISQUALIFIED no matter how much
target GA it recovers -- that would be 'improving sensitivity' by reproducing PLEX's mistakes.

Run:  oct_env\\Scripts\\python.exe src\\exp_multifocal_recovery.py   ->  results/exp_multifocal_recovery.csv
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OCT_BM_DL", "1")

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

import bm_dl
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR
from reader.core import e2e_source, layers, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj
from reader.core.layer_store import JsonSidecarLayerStore

LIB = os.path.join(_REPO, "viewer", "data_store", "library")
CORR = os.path.join(_REPO, "reader", "data_store", "corrections")
MMPP2 = oac_ga.MMPP2

# (subject, visit, eye, group). targets = real multifocal misses; guards MUST stay ~0 (PLEX-FP + controls);
# clean = large/focal eyes that must not drift.
EYES = [
    ("NHAMD-003-001", "V1", "OD", "target"), ("NHAMD-003-026", "V3", "OD", "target"),
    ("NHAMD-003-026", "V3", "OS", "target"), ("NHAMD-003-001", "V1", "OS", "target"),
    ("NHAMD-003-014", "V1", "OD", "guard-plexFP"), ("NHAMD-003-010", "V1", "OD", "guard-plexFP"),
    ("NHAMD-003-006", "V3", "OD", "guard-plexFP"),
    ("NHAMD-003-012", "V3", "OD", "guard-control"), ("NHAMD-003-012", "V3", "OS", "guard-control"),
    ("NHAMD-003-002", "V2", "OS", "guard-control"), ("NHAMD-003-016", "V2", "OD", "guard-control"),
    ("NHAMD-003-005", "V3", "OD", "clean"), ("NHAMD-003-008", "V1", "OS", "clean"),
    ("NHAMD-003-003", "V3", "OS", "clean"), ("NHAMD-003-004", "V1", "OD", "clean"),
]


def resolve(subject, visit, eye):
    want = f"{subject}-{visit}"
    with open(os.path.join(RESULTS_DIR, "bm_worklist.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["subject"] == want and r["eye"].upper() == eye.upper():
                return os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return None


def baked_bm(subject, visit, eye, ov):
    bp = os.path.join(LIB, f"{subject}-{visit}_{eye.upper()}", "bundle.npz")
    if os.path.exists(bp):
        with np.load(bp) as z:
            if "bm_dl" in z.files:
                b = np.asarray(z["bm_dl"], np.float32)
                if b.shape == (ov.n_bscans, ov.W):
                    return b
    if bm_dl.available():
        return bm_dl.segment_volume(ov.vol).astype(np.float32)
    return layers.effective_surfaces(ov, JsonSidecarLayerStore(CORR))[1]


def area(mask):
    return float(np.asarray(mask, bool).sum()) * MMPP2


def variant_area(P, name):
    """Return area (mm2) for a named footprint variant. Baseline uses the shipped default footprint()."""
    if name == "baseline":
        return oac_ga.footprint(P)[1]
    if name == "size180":
        return oac_ga.footprint(P, min_diam_um=180.0)[1]
    if name == "size125":
        return oac_ga.footprint(P, min_diam_um=125.0)[1]
    if name == "depth0.35":
        return oac_ga.footprint(P, min_depth=0.35)[1]
    if name == "depth_off":
        return oac_ga.footprint(P, min_depth=None)[1]
    if name == "bridge0.30":
        # bridge proximate sub-cRORA foci: close the post-hyper-fill mask by ~0.30mm BEFORE the size gate,
        # so a constellation counts as one cRORA lesion. Keep the default min_depth complete-loss gate.
        return _bridge(P, close_mm=0.30, min_depth=0.27)[1]
    if name == "bridge0.30+depth0.35":
        return _bridge(P, close_mm=0.30, min_depth=0.35)[1]
    raise ValueError(name)


def _bridge(P, close_mm, min_depth):
    from skimage import measure
    S = oac_ga.footprint_stages(P)
    filled = S["filled"]                                 # post require-hyper + hole-fill, pre size gate
    it = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
    bridged = binary_fill_holes(binary_closing(filled, iterations=it))
    sized = fp.crora(bridged, 250.0)
    # re-apply the complete-loss gate on the (now merged) components
    if min_depth is not None and sized.any():
        ratio = P["loss6"] / np.maximum(P["base"], 1e-6)
        lbl = measure.label(sized); keep = np.zeros_like(sized)
        for rp in measure.regionprops(lbl):
            comp = lbl == rp.label
            if float(ratio[comp].min()) < float(min_depth):
                keep[comp] = True
        sized = keep
    return sized, area(sized)


VARIANTS = ["baseline", "size180", "size125", "depth0.35", "depth_off", "bridge0.30", "bridge0.30+depth0.35"]


def main():
    rows = []
    for subject, visit, eye, group in EYES:
        e2e = resolve(subject, visit, eye)
        if not e2e or not os.path.exists(e2e):
            print(f"  skip {subject} {eye}: no E2E"); continue
        raw = e2e_source.open_e2e(e2e)
        ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye.upper()))
        bm = baked_bm(subject, visit, eye, ov)
        P = oac_ga.prep(ov, bm, baseline="radial2")
        rec = {"eye": f"{subject[6:]}-{visit} {eye}", "group": group}
        for v in VARIANTS:
            try:
                rec[v] = round(variant_area(P, v), 3)
            except Exception as ex:
                rec[v] = f"ERR:{ex}"
        rows.append(rec)
        print(f"  {rec['eye']:16} [{group:13}] " + "  ".join(f"{v}={rec[v]}" for v in VARIANTS))

    out = os.path.join(RESULTS_DIR, "exp_multifocal_recovery.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eye", "group"] + VARIANTS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")

    # verdict per variant: target recovery vs guard leakage
    def num(x):
        return x if isinstance(x, (int, float)) else 0.0
    print(f"\n{'variant':22} {'target_recover_mm2':>20} {'guard_MAX_mm2':>15} {'clean_maxdrift':>15}  verdict")
    base = {r["eye"]: num(r["baseline"]) for r in rows}
    for v in VARIANTS:
        trec = sum(num(r[v]) - base[r["eye"]] for r in rows if r["group"] == "target")
        gmax = max((num(r[v]) for r in rows if r["group"].startswith("guard")), default=0.0)
        cdrift = max((abs(num(r[v]) - base[r["eye"]]) for r in rows if r["group"] == "clean"), default=0.0)
        ok = "PASS" if (gmax < 0.25 and cdrift < 0.5) else "FAIL(guard/clean)"
        print(f"{v:22} {trec:20.3f} {gmax:15.3f} {cdrift:15.3f}  {ok}")


if __name__ == "__main__":
    main()
