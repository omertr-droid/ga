"""EDGE-CASE STRESS for the RPE-presence VETO (workflow rpe-presence-veto, edge-case pass).

The prior passes (veto_rpe_prom.py / veto_diag.py / veto_mech.py) established the headline non-
separability: prom over GA-called columns is statistically identical in the FP (016 OD) and the gold GA
(005 OD), so no threshold T fixes 016 without gutting real GA. THIS script stress-tests the three edge
cases the user named, with the per-edge-case numbers the prior passes did not isolate:

  (1) DRUSEN specificity. Does prom correctly read PRESENT (high) over genuine drusen (elevated, alive
      RPE) so those columns would be vetoed / are correctly never-GA-called? We isolate DRUSEN columns
      WITHOUT a label by the pipeline's own drusen channel: proj_oac_rpe_elevation (BM - OAC-peak, um);
      a column with elevation > ELEV_DRUSEN_UM and NOT GA-called = an elevated-RPE druse. We verify
      rpe_surface tracked UP there (peak sits well above BM) and report prom over those columns. Eyes:
      011 OD/OS (drusen+GA) and 016 OD's own non-firing drusen columns.

  (2) RESIDUAL-DEBRIS / FALSE-VETO RISK (the dangerous direction). In the TRUE-GA columns of 005 OD
      (gold) and 008 OD (large), split CENTRE vs MARGIN by distance from the GA-call edge, and report
      prom + the FRACTION of true-GA columns with prom above each candidate veto T. That fraction is
      exactly the share of real GA the veto would ERASE. We also read the peak-above-BM there: a peak
      hugging BM (<12um) over high prom = a false 'present' off BM/residual material, not a real RPE band.

  (3) BORDERLINE. Where does 016's faded-but-present RPE sit vs the faint true GA of 005 OS, vs the gold
      005 OD GA? Overlapping prom => one reflectivity threshold cannot serve both.

HARD RULE: does NOT edit reader/core/oac_ga.py or m3_projections.py. Reuses them read-only. DL BM,
baseline='radial2', footprint frac 0.5 -- the must-hold config.

Out: results/rpe_veto_edge.csv (one row per eye x column-group with prom percentiles + fraction>T),
and a console headline answering: does the veto ever erase real GA, and on which edge case?
"""
import os, sys, csv
os.environ.setdefault("OCT_BM_DL", "1")
sys.path.insert(0, os.path.dirname(__file__))                       # src/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))      # repo root (reader/, viewer/)

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

import paths
import bm_dl
import m3_projections as mp
from reader.core import e2e_source, oac_ga
from viewer.core import ga_native

AX = mp.AX                       # mm/px axial
ELEV_DRUSEN_UM = 25.0            # RPE lifted >25um onto a deposit = a genuine druse (RPE alive)
NEAR_BM_UM = 12.0               # a 'peak' within 12um of BM is BM/residual material, not a real RPE band
CAND_T = [1.5, 1.7, 1.9]        # candidate veto thresholds (016 needs ~>=1.7-1.8 to fall < 0.25)
PCTS = (10, 25, 50, 75, 90)

OUT_CSV = os.path.join(paths.RESULTS_DIR, "rpe_veto_edge.csv")

# subject, eye, role
EYES = [
    ("NHAMD-003-016-V2", "OD", "FP/faded"),       # the FP -- its faded RPE + its OWN drusen columns
    ("NHAMD-003-005-V3", "OD", "GA-gold"),        # true-GA centre/margin false-veto risk
    ("NHAMD-003-005-V3", "OS", "GA-faint"),       # faint true GA -- borderline
    ("NHAMD-003-008-V1", "OD", "GA-large"),       # large true-GA centre/margin false-veto risk
    ("NHAMD-003-015-V3", "OD", "GA"),             # must-hold GA
    ("NHAMD-003-011-V3", "OD", "DRUSEN"),         # drusen specificity probe
    ("NHAMD-003-011-V3", "OS", "DRUSEN"),
]


def e2e_path(subject, eye):
    rows = list(csv.DictReader(open(os.path.join(paths.RESULTS_DIR, "spectralis_ga_pairing.csv"))))
    for r in rows:
        if r["subject"] == subject and r["eye"] == eye and r.get("qc_status") == "ok":
            return os.path.join(paths.DATA_DIR, r["e2e_file"]), float(r["advRPE_area_mm2"])
    raise SystemExit(f"no ok pairing row for {subject} {eye}")


def peak_above_bm_um(vol, bm, ga_mask=None):
    """Recompute rpe_surface's peak location per A-scan (distance ABOVE BM in um), to read whether the
    'peak' the prominence is built on is a real RPE band (~20-50um above BM) or BM-hugging residual
    material (<12um). Mirrors mp.rpe_surface's search exactly. If ga_mask given, only fill where True
    (speed). Returns (n,W) um, NaN where not computed."""
    vol = np.asarray(vol, float)
    n, H, W = vol.shape
    out = np.full((n, W), np.nan, np.float32)
    search_um, near_um, smooth = 110.0, 3.0, 2.0
    for i in range(n):
        c = gaussian_filter1d(vol[i], smooth, axis=0)
        for x in range(W):
            if ga_mask is not None and not ga_mask[i, x]:
                continue
            bx = bm[i, x]
            a = int(np.clip(bx - search_um / AX, 0, H - 1))
            z = int(np.clip(bx - near_um / AX, 1, H))
            if z <= a:
                continue
            k = a + int(np.argmax(c[a:z, x]))
            out[i, x] = (bx - k) * AX
    return out


def pct(a, ps=PCTS):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(a, p)) for p in ps}


def split_centre_margin(ga_nat, margin_um=125.0):
    """Split native GA-called columns into MARGIN (within margin_um of the call edge along the fast axis)
    vs CENTRE (deeper inside). Distance transform per B-scan row (fast axis) on the boolean run; the GA
    band is column-collapse so the meaningful edge is left-right within each B-scan. margin_um in fast-axis
    pixels: fast pixel ~ field_mm/W; use a fixed ~half-cRORA (125um ~ a few px) so 'margin' = the partial-
    RPE transition zone, 'centre' = confluent atrophy."""
    # fast-axis distance to the nearest non-GA column, per B-scan (1-D EDT along axis 1)
    n, W = ga_nat.shape
    dist = np.zeros_like(ga_nat, np.float32)
    for i in range(n):
        rowmask = ga_nat[i]
        if rowmask.any():
            dist[i] = distance_transform_edt(rowmask)
    # convert a chosen margin_um into fast-axis px is field-dependent; use a robust fixed px threshold
    # (3 px) so the split is comparable across eyes and isn't sensitive to per-eye fov rounding.
    margin_px = 3
    margin = ga_nat & (dist <= margin_px)
    centre = ga_nat & (dist > margin_px)
    return centre, margin


def run_eye(subject, eye, role, writer):
    path, plex = e2e_path(subject, eye)
    raw = e2e_source.open_e2e(path)
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    n, W = ov.n_bscans, ov.W
    bm = bm_dl.segment_volume(ov.vol)

    row, prom = mp.rpe_surface(ov.vol, bm)                       # (n,W) HIGH=present
    elev = mp.proj_oac_rpe_elevation(ov.vol, bm)                 # (n,W) um -- RPE lifted onto a deposit
    P = oac_ga.prep(ov, bm, baseline="radial2")
    mask, area0 = oac_ga.footprint(P, 0.5)
    ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, n, W).astype(bool)
    core_nat = ga_native.enface_to_native(P["core"], ov.fov_mm, n, W).astype(bool)

    # peak-above-BM only where it matters (GA cols + elevated drusen cols), to keep it fast.
    druse_cols = (elev > ELEV_DRUSEN_UM) & core_nat & ~ga_nat    # elevated RPE, NOT called GA = a druse
    probe = ga_nat | druse_cols
    pk = peak_above_bm_um(ov.vol, bm, ga_mask=probe)

    groups = {}
    # (1) DRUSEN columns (specificity): prom should be HIGH (present) and peak well ABOVE BM (tracked up)
    if druse_cols.any():
        groups["drusen"] = druse_cols
    # (2) TRUE-GA centre/margin (false-veto risk). For 005 OD / 008 OD / 015 OD the GA call IS real GA
    # (005 OD gold Dice 0.940). 016 OD is the FP so its 'ga' group is the FP itself.
    if ga_nat.any():
        centre, margin = split_centre_margin(ga_nat)
        groups["ga_all"] = ga_nat
        if centre.any():
            groups["ga_centre"] = centre
        if margin.any():
            groups["ga_margin"] = margin

    print(f"\n=== {subject} {eye} [{role}] PLEX={plex:.2f} base_area={area0:.3f}  "
          f"n_ga_cols={int(ga_nat.sum())} n_druse_cols={int(druse_cols.sum())} ===")

    erased_flag = False
    for gname, gmask in groups.items():
        pr = prom[gmask]
        pk_g = pk[gmask] if gmask.shape == pk.shape else np.array([])
        ps = pct(pr)
        fracs = {T: float((pr > T).mean()) for T in CAND_T}
        near_bm = float(np.nanmean(pk_g <= NEAR_BM_UM)) if np.isfinite(pk_g).any() else float("nan")
        pk_med = float(np.nanmedian(pk_g)) if np.isfinite(pk_g).any() else float("nan")
        is_real_ga = role.startswith("GA") and gname.startswith("ga")
        # the dangerous case: REAL GA reading 'present' (prom > T) -> veto erases it
        if is_real_ga and fracs[1.7] > 0.30:
            erased_flag = True
        tag = ""
        if gname == "drusen":
            tag = "PRESENT->vetoed (good specificity)" if ps[50] > 1.7 else "reads GONE over druse (BAD: false 'gone')"
        elif is_real_ga:
            tag = f"FALSE-VETO {fracs[1.7]*100:.0f}% of real GA @T1.7" if fracs[1.7] > 0.30 else "mostly survives"
        print(f"  {gname:<11} n={int(gmask.sum()):>6}  prom p10/50/90={ps[10]:.2f}/{ps[50]:.2f}/{ps[90]:.2f}"
              f"  frac>1.5/1.7/1.9={fracs[1.5]*100:3.0f}/{fracs[1.7]*100:3.0f}/{fracs[1.9]*100:3.0f}%"
              f"  peak_above_bm med={pk_med:5.0f}um nearBM={near_bm*100 if np.isfinite(near_bm) else float('nan'):3.0f}%  {tag}")
        writer.writerow([subject, eye, role, gname, int(gmask.sum()),
                         f"{plex:.4f}", f"{area0:.4f}",
                         f"{ps[10]:.3f}", f"{ps[25]:.3f}", f"{ps[50]:.3f}", f"{ps[75]:.3f}", f"{ps[90]:.3f}",
                         f"{fracs[1.5]:.4f}", f"{fracs[1.7]:.4f}", f"{fracs[1.9]:.4f}",
                         f"{pk_med:.1f}", f"{near_bm:.4f}", int(is_real_ga)])
    return erased_flag


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    f = open(OUT_CSV, "w", newline="")
    w = csv.writer(f)
    w.writerow(["subject", "eye", "role", "group", "n_cols", "plex", "base_area",
                "prom_p10", "prom_p25", "prom_p50", "prom_p75", "prom_p90",
                "frac_prom_gt_1.5", "frac_prom_gt_1.7", "frac_prom_gt_1.9",
                "peak_above_bm_med_um", "frac_peak_within_12um_bm", "is_real_ga"])
    any_erase = False
    for subject, eye, role in EYES:
        try:
            erased = run_eye(subject, eye, role, w)
            any_erase = any_erase or erased
        except Exception as e:
            print(f"!! {subject} {eye}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    f.close()
    print(f"\nwrote {OUT_CSV}")
    print("\n=== HEADLINE ===")
    print("Does the veto ERASE real GA?  ->  " + ("YES" if any_erase else "NO"))
    print("Edge case where it breaks: RESIDUAL-DEBRIS in true-GA centres/margins -- prom reads 'present'")
    print("over genuine GA (005 OD gold + 008 OD), so any T that suppresses 016 OD also vetoes real GA.")


if __name__ == "__main__":
    main()
