"""STANDALONE experiment (workflow rpe-presence-veto): test an RPE-PRESENCE veto on the OAC GA detector.

DOES NOT edit reader/core/oac_ga.py or m3_projections.py. It reuses them read-only:
  - oac_ga.prep(..., baseline='radial2') + oac_ga.footprint(p, 0.5)  -> the en-face GA call (must-hold)
  - mp.rpe_surface(vol, bm) -> (row, prom); HIGH prom = a real RPE band is present (drusen-aware, BM-anchored,
    ILM-FREE). The veto drops GA-called A-scans where prom says the RPE is structurally present.

For each eye:
  ov = load_volume; bm = bm_dl.segment_volume(ov.vol)            # DL BM (the must-hold config)
  row, prom = mp.rpe_surface(ov.vol, bm)                          # (n,W) native
  P = oac_ga.prep(ov, bm, baseline='radial2'); mask,area = oac_ga.footprint(P,0.5)
  ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, n, W)      # GA-called columns, native
  veto: a GA column (bi,x) is kept ONLY IF prom[bi,x] <= T  (RPE structurally present -> drop)
        optional spatial denoise (median over a small (slow,fast) window) before thresholding,
        and an optional per-en-face-pixel MIN-RUN so a lone vetoed column can't punch a hole.
  Recompute the kept-GA area by mapping the surviving native columns back to the en-face mask and
  re-running the SAME cRORA (>=250um) step, so the veto sits AFTER the hyper criteria + BEFORE/with cRORA.

Sweeps T and prints kept area vs the must-hold targets + the FP. Dumps per-eye prom stats over GA-called
columns so the separability (016 FP vs real GA) is visible. Writes outputs/veto/veto_sweep.csv and per-eye
prom histograms.
"""
import os, sys, csv, argparse
os.environ.setdefault("OCT_BM_DL", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))         # src/ on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))      # repo root (reader/, viewer/)

import numpy as np
from scipy.ndimage import median_filter

import paths
import bm_dl
import m3_projections as mp
from reader.core import e2e_source, oac_ga, projection as proj, footprint as fp
from viewer.core import ga_native

OUT = os.path.join(paths.OUT_DIR, "veto")
os.makedirs(OUT, exist_ok=True)

# (subject, eye, role) -- the focus set. roles: FP (must drop), and must-hold GA eyes.
EYES = [
    ("NHAMD-003-016-V2", "OD", "FP"),          # the false positive to veto -> want area < 0.25
    ("NHAMD-003-005-V3", "OD", "GA-gold"),     # 1.0548, Dice 0.940 -- must hold
    ("NHAMD-003-005-V3", "OS", "GA-faint"),    # 0.526 faint -- erase-risk, must hold
    ("NHAMD-003-008-V1", "OD", "GA-large"),    # 12.853 -- must hold
    ("NHAMD-003-015-V3", "OD", "GA"),          # 1.498 -- must hold
    ("NHAMD-003-011-V3", "OD", "DRUSEN"),      # drusen eye -- specificity probe (does prom veto FP not GA?)
    ("NHAMD-003-011-V3", "OS", "DRUSEN"),
]

MMPP2 = oac_ga.MMPP2
T_SWEEP = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]


def e2e_path(subject, eye):
    import re
    rows = list(csv.DictReader(open(os.path.join(paths.RESULTS_DIR, "spectralis_ga_pairing.csv"))))
    for r in rows:
        if r["subject"] == subject and r["eye"] == eye and r.get("qc_status") == "ok":
            return os.path.join(paths.DATA_DIR, r["e2e_file"]), float(r["advRPE_area_mm2"])
    raise SystemExit(f"no ok pairing row for {subject} {eye}")


def native_to_enface_mask(nat, fov_mm, out_shape):
    """Inverse of ga_native.enface_to_native: native (n,W) bool -> en-face (out,out) bool, so we can
    re-run the SAME cRORA on the post-veto columns. Mirrors projection.to_enface geometry (centre pad +
    slow-axis flip) the same way ga_native does its forward map, just transposed."""
    import cv2
    n, W = nat.shape
    H = out_shape[0]
    fh = max(1, int(round(fov_mm[1] / ga_native.MMPP)))
    fw = max(1, int(round(fov_mm[0] / ga_native.MMPP)))
    field = cv2.resize(np.asarray(nat, bool)[::-1].astype(np.uint8), (fw, fh),
                       interpolation=cv2.INTER_NEAREST)
    # centre-pad the field back into the square en-face frame (inverse of center_extract)
    out = np.zeros((H, out_shape[1]), np.uint8)
    oy, ox = (H - fh) // 2, (out_shape[1] - fw) // 2
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    h, w = min(fh - sy0, H - dy0), min(fw - sx0, out_shape[1] - dx0)
    if h > 0 and w > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = field[sy0:sy0 + h, sx0:sx0 + w]
    return out > 0


def run_eye(subject, eye, role, denoise=True, min_run=0):
    path, plex = e2e_path(subject, eye)
    raw = e2e_source.open_e2e(path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    n, W = ov.n_bscans, ov.W
    bm = bm_dl.segment_volume(ov.vol)
    row, prom = mp.rpe_surface(ov.vol, bm)                 # (n,W), HIGH = RPE present
    P = oac_ga.prep(ov, bm, baseline="radial2")
    mask, area0 = oac_ga.footprint(P, 0.5)                 # the must-hold en-face GA call
    out_shape = mask.shape
    ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, n, W).astype(bool)

    # prom over the GA-CALLED columns: this is the separability we care about.
    promG = prom[ga_nat]
    promstats = {p: float(np.percentile(promG, p)) for p in (10, 25, 50, 75, 90)} if promG.size else {}

    # spatially denoise prom over the native frame BEFORE thresholding (speckle-proof): small median in
    # (slow, fast). 1px slow ~ a B-scan; 7px fast ~ 0.05mm. Cheap, robust, monotone-friendly.
    prom_use = median_filter(prom, size=(1, 7)) if denoise else prom

    rows = []
    for T in T_SWEEP:
        present = prom_use > T                              # RPE structurally present -> veto
        keep_nat = ga_nat & ~present                        # surviving GA columns
        if min_run > 0:
            # require a kept GA column to have >=min_run consecutive kept neighbours along fast axis,
            # so a single vetoed A-scan inside a solid GA run can't fragment it (re-close holes).
            keep_nat = _enforce_min_run(keep_nat, min_run)
        keep_enf = native_to_enface_mask(keep_nat, ov.fov_mm, out_shape) & mask
        kept = fp.crora(keep_enf, 250.0)
        area = float(kept.sum()) * MMPP2
        rows.append((T, area))
    return {"subject": subject, "eye": eye, "role": role, "plex": plex,
            "area0": area0, "promstats": promstats, "n": n, "W": W,
            "n_ga_cols": int(ga_nat.sum()), "sweep": rows, "prom": prom, "ga_nat": ga_nat}


def _enforce_min_run(keep, k):
    """Bridge short veto-holes: along the fast axis, close gaps < 2k+1 so a single vetoed A-scan inside a
    solid kept GA run can't fragment it. Conservative -- only re-fills small holes, never grows new area."""
    from scipy.ndimage import binary_closing
    return binary_closing(keep, structure=np.ones((1, 2 * k + 1), bool), iterations=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--min-run", type=int, default=0)
    a = ap.parse_args()
    results = []
    for subject, eye, role in EYES:
        try:
            r = run_eye(subject, eye, role, denoise=not a.no_denoise, min_run=a.min_run)
        except Exception as e:
            print(f"!! {subject} {eye}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue
        results.append(r)
        ps = r["promstats"]
        psf = " ".join(f"p{p}={ps.get(p, float('nan')):.2f}" for p in (10, 25, 50, 75, 90))
        print(f"\n=== {subject} {eye} [{role}] PLEX={r['plex']:.3f} base_area={r['area0']:.3f} "
              f"n_ga_cols={r['n_ga_cols']} ===")
        print(f"    prom over GA-called cols: {psf}")
        print(f"    {'T':>5} {'kept_area':>10}")
        for T, area in r["sweep"]:
            print(f"    {T:5.2f} {area:10.4f}")

    # CSV: one row per (eye, T)
    csvp = os.path.join(OUT, "veto_sweep.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "eye", "role", "plex", "base_area", "T", "kept_area",
                    "n_ga_cols", "promG_p10", "promG_p50", "promG_p90"])
        for r in results:
            ps = r["promstats"]
            for T, area in r["sweep"]:
                w.writerow([r["subject"], r["eye"], r["role"], f"{r['plex']:.4f}",
                            f"{r['area0']:.4f}", T, f"{area:.4f}", r["n_ga_cols"],
                            f"{ps.get(10, float('nan')):.3f}", f"{ps.get(50, float('nan')):.3f}",
                            f"{ps.get(90, float('nan')):.3f}"])
    print(f"\nwrote {csvp}")

    # compact verdict table at a few candidate thresholds
    print("\n=== VERDICT TABLE (kept area at candidate T) ===")
    cand = [1.4, 1.5, 1.6, 1.7, 1.9]
    hdr = "  ".join(f"T={t}" for t in cand)
    print(f"{'eye':<22}{'role':<10}{'PLEX':>7}{'base':>8}   {hdr}")
    for r in results:
        d = {T: a for T, a in r["sweep"]}
        cells = "  ".join(f"{d[t]:6.3f}" for t in cand)
        print(f"{r['subject']+' '+r['eye']:<22}{r['role']:<10}{r['plex']:>7.2f}{r['area0']:>8.3f}   {cells}")


if __name__ == "__main__":
    main()
