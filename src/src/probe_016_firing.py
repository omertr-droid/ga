#!/usr/bin/env python
"""CHARACTERIZE 016 OD's FALSE firing (DL BM, radial2 baseline).

The lone remaining false-positive control after radial2 + hyper_abs floor: 016 OD reads ~1.3 mm2 GA at
PLEX 0. This probe takes the EXACT reference firing mask (oac_ga.footprint(p, 0.5)[0]) and, on its TRUE
pixels, characterises WHY they fire:
  - RPE->BM elevation (elev6, um)  -- drusen = RPE present-but-lifted (HIGH); GA = RPE gone (LOW)
  - distance from field centre (mm) + proximity to the eroded `core` edge (px) -- edge/vignette firing
  - whole-column SNR (sig6 = mean intensity over depth) -- low = vignette/no-tissue margin
  - sub-BM hyper (p['hyper6']) -- the 2nd-criterion transmission channel

It also maps the firing back to NATIVE B-scans (which B-scans contribute most firing AREA) and runs the
SAME probe on the must-hold GA eyes + the stay-clean controls, so every cut we'd consider can be read as
"016 vs the true-GA refs" -- the firing-feature distributions side by side.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\probe_016_firing.py
Writes results/probe_016_firing.csv + results/probe_016_firing_perfeat.csv and prints the tables.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["OCT_BM_DL"] = "1"   # MUST precede importing bm_dl

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

import bm_dl
import m3_projections as mp
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import projection as proj

MMPP = proj.ENFACE_MMPP        # mm/px en-face (6/512)
MMPP2 = oac_ga.MMPP2

# 016 OD = the FP under test; the 4 must-hold GA eyes; 3 stay-clean controls.
EYES = [
    ("NHAMD-003-016-V2", "OD", "control_FP"),
    ("NHAMD-003-005-V3", "OD", "ga"),     # focal gold, PLEX 1.08
    ("NHAMD-003-005-V3", "OS", "ga"),     # small/faint, PLEX 0.57
    ("NHAMD-003-008-V1", "OD", "ga"),     # large eccentric, PLEX 13.78
    ("NHAMD-003-015-V3", "OD", "ga"),     # PLEX 1.99
    ("NHAMD-003-002-V2", "OS", "control"),
    ("NHAMD-003-006-V3", "OS", "control"),
    ("NHAMD-003-012-V3", "OD", "control"),
]


def e2e_lookup():
    out = {}
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("qc_status") != "ok":
                continue
            out[(r["subject"], r["eye"].upper())] = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return out


def plex_lookup():
    out = {}
    path = os.path.join(RESULTS_DIR, "plex_compare.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[(r["subject"], r["eye"].upper())] = float(r["plex_mm2"])
    return out


def qstats(v):
    """median, p25, p75, p90, frac>30um helper printed compactly."""
    if v.size == 0:
        return dict(n=0, med=float("nan"), p25=float("nan"), p75=float("nan"), p90=float("nan"))
    return dict(n=int(v.size), med=float(np.median(v)), p25=float(np.percentile(v, 25)),
                p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)))


def main():
    print(f"bm_dl.active()={bm_dl.active()} backend={bm_dl.backend()}\n", flush=True)
    e2e = e2e_lookup()
    plex = plex_lookup()

    rows = []          # per-eye summary
    feat_rows = []     # per-eye feature distribution (for the side-by-side table)
    for subject, eye, label in EYES:
        path = e2e.get((subject, eye))
        if path is None or not os.path.exists(path):
            print(f"  SKIP {subject} {eye}: e2e missing ({path})", flush=True)
            continue
        raw = e2e_source.open_e2e(path)
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol)
        p = oac_ga.prep(ov, bm, baseline="radial2")
        mask, area = oac_ga.footprint(p, 0.50)            # REFERENCE firing mask + area

        # --- aligned 6mm feature en-faces (exact oac_area.py / recipe recipe) ---
        oac = mp.oac_volume(ov.vol)
        elev_nat = np.clip((bm - mp.band_argmax_row(oac, bm, *mp.OAC_RPE_UM)) * mp.AX, 0.0, None)  # (n,W) um
        elev6 = proj.to_enface(mp.destripe2d(elev_nat, signed=False), ov.fov_mm)                   # ~rpe6 grid
        sig6 = proj.to_enface(ov.vol.mean(axis=1).astype("float32"), ov.fov_mm)                    # whole-col SNR
        hyper6 = p["hyper6"]
        core = p["core"]
        H, W = elev6.shape

        # geometry: distance from field centre (mm); distance INTO the eroded core from its edge (px)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r_cen_mm = np.sqrt((xx - W / 2.0) ** 2 + (yy - H / 2.0) ** 2) * MMPP
        core_edt = distance_transform_edt(core)            # px from outside-core; small => near the core rim

        # normalise sig6 like prep's vignette gate uses it (fraction of the in-field median)
        valid_med = float(np.nanpercentile(sig6[core], 50)) if core.any() else 1.0
        sig6n = sig6 / (valid_med + 1e-9)                   # ~1 = typical tissue ; <0.5 = vignette territory

        m = mask
        nfire = int(m.sum())
        # firing-pixel feature vectors
        f_elev = elev6[m]
        f_rcen = r_cen_mm[m]
        f_edge = core_edt[m]
        f_sign = sig6n[m]
        f_hyp = hyper6[m]
        # reference (in-core, non-firing) distributions for contrast
        ref = core & ~m
        med_core_elev = float(np.median(elev6[ref])) if ref.any() else float("nan")

        # native B-scan attribution: which B-scans carry the firing area? Map firing en-face -> native rows
        # via the inverse of to_enface (the en-face row index ~ resampled native B-scan). We attribute by
        # building a native-resolution firing map: project mask back is non-trivial; instead bin firing rows
        # by their en-face Y -> native B-scan index linearly (n_bscans rows over H en-face rows, flipped).
        n = ov.n_bscans
        ys = np.where(m.any(axis=1), np.arange(H), -1)
        fire_rows_y = np.repeat(np.arange(H)[:, None], W, axis=1)[m]
        # en-face Y=0 is top; to_enface row-flips the native stack, so native_b ~ (n-1) * (1 - y/(H-1))
        nb = np.clip(np.round((n - 1) * (1.0 - fire_rows_y / max(H - 1, 1))).astype(int), 0, n - 1)
        b_area = np.zeros(n)
        for b in nb:
            b_area[b] += 1
        b_area_mm2 = b_area * MMPP2
        order = np.argsort(b_area_mm2)[::-1]
        top_b = [(int(order[k]), round(float(b_area_mm2[order[k]]), 3)) for k in range(min(8, n))
                 if b_area_mm2[order[k]] > 0]
        # fraction of firing area in the OUTER B-scan margin (top/bottom 15% of slices = field edge)
        edge_b = (nb < 0.15 * n) | (nb > 0.85 * n)
        frac_edge_bscan = float(edge_b.mean()) if nfire else 0.0

        # decision features
        frac_drusen = float((f_elev > 30.0).mean()) if nfire else 0.0       # firing over high RPE lift
        frac_near_edge = float((f_edge <= 8.0).mean()) if nfire else 0.0    # firing within 8px of core rim
        frac_lowsnr = float((f_sign < 0.6).mean()) if nfire else 0.0        # firing in low-SNR territory
        frac_periph = float((f_rcen > 2.5).mean()) if nfire else 0.0        # firing >2.5mm from centre

        pl = plex.get((subject, eye), float("nan"))
        rows.append(dict(subject=subject, eye=eye, label=label, plex=pl, area=area, nfire=nfire,
                         elev_med=qstats(f_elev)["med"], rcen_med=qstats(f_rcen)["med"],
                         edge_med=qstats(f_edge)["med"], sign_med=qstats(f_sign)["med"],
                         hyp_med=qstats(f_hyp)["med"], core_elev_med=med_core_elev,
                         frac_drusen=frac_drusen, frac_near_edge=frac_near_edge,
                         frac_lowsnr=frac_lowsnr, frac_periph=frac_periph,
                         frac_edge_bscan=frac_edge_bscan, top_b=top_b))
        feat_rows.append((subject, eye, label, f_elev, f_rcen, f_edge, f_sign, f_hyp))

        print(f"=== {subject[-7:]}_{eye} [{label}] PLEX={pl:.2f}  area={area:.3f} mm2  nfire_px={nfire} ===",
              flush=True)
        if nfire:
            print(f"    elev(um) firing: med={np.median(f_elev):6.1f}  p75={np.percentile(f_elev,75):6.1f}  "
                  f"p90={np.percentile(f_elev,90):6.1f}   (core non-fire med={med_core_elev:.1f})", flush=True)
            print(f"    r_centre(mm)   : med={np.median(f_rcen):5.2f}  p75={np.percentile(f_rcen,75):5.2f}  "
                  f"p90={np.percentile(f_rcen,90):5.2f}", flush=True)
            print(f"    core_edge(px)  : med={np.median(f_edge):5.1f}  p25={np.percentile(f_edge,25):5.1f}  "
                  f"(small=hugging the eroded rim)", flush=True)
            print(f"    sig6_norm      : med={np.median(f_sign):5.2f}  p25={np.percentile(f_sign,25):5.2f}  "
                  f"(<0.6 = low-SNR/vignette)", flush=True)
            print(f"    hyper6 firing  : med={np.median(f_hyp):6.3f}  p25={np.percentile(f_hyp,25):6.3f}", flush=True)
            print(f"    FRACTIONS  drusen(elev>30um)={frac_drusen:.2f}  near_core_edge(<=8px)={frac_near_edge:.2f}  "
                  f"low_snr(<0.6)={frac_lowsnr:.2f}  periph(>2.5mm)={frac_periph:.2f}", flush=True)
            print(f"    B-scan firing  : edge-slice frac={frac_edge_bscan:.2f}  top(area mm2)={top_b}", flush=True)
        print(flush=True)

    # ---- CSV out ----
    out_csv = os.path.join(RESULTS_DIR, "probe_016_firing.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "eye", "label", "plex_mm2", "area_mm2", "nfire_px", "elev_med_um",
                    "rcen_med_mm", "edge_med_px", "sign_med", "hyper_med", "core_nonfire_elev_med_um",
                    "frac_drusen_e30", "frac_near_coreedge_8px", "frac_lowsnr_0p6", "frac_periph_2p5mm",
                    "frac_edge_bscan", "top_bscans_area"])
        for r in rows:
            w.writerow([r["subject"], r["eye"], r["label"], f"{r['plex']:.3f}", f"{r['area']:.4f}",
                        r["nfire"], f"{r['elev_med']:.1f}", f"{r['rcen_med']:.2f}", f"{r['edge_med']:.1f}",
                        f"{r['sign_med']:.2f}", f"{r['hyp_med']:.3f}", f"{r['core_elev_med']:.1f}",
                        f"{r['frac_drusen']:.2f}", f"{r['frac_near_edge']:.2f}", f"{r['frac_lowsnr']:.2f}",
                        f"{r['frac_periph']:.2f}", f"{r['frac_edge_bscan']:.2f}", str(r["top_b"])])
    print(f"wrote {out_csv}", flush=True)

    # ---- compact side-by-side table (the deliverable) ----
    print("\n==== FIRING-FEATURE DISTRIBUTION (median over firing pixels) — 016 vs refs ====")
    hdr = ("eye".ljust(14) + "label".ljust(11) + "plex".rjust(6) + "area".rjust(7) + "elev_um".rjust(9)
           + "%drus".rjust(7) + "%edge".rjust(7) + "%loSNR".rjust(8) + "r_mm".rjust(6)
           + "snr".rjust(6) + "hyper".rjust(7))
    print(hdr)
    for r in rows:
        print(f"{r['subject'][-7:]+'_'+r['eye']:14}{r['label']:11}{r['plex']:6.2f}{r['area']:7.2f}"
              f"{r['elev_med']:9.1f}{100*r['frac_drusen']:7.0f}{100*r['frac_near_edge']:7.0f}"
              f"{100*r['frac_lowsnr']:8.0f}{r['rcen_med']:6.2f}{r['sign_med']:6.2f}{r['hyp_med']:7.3f}")

    # ---- separator search: is there a single column-feature that splits 016 firing from true-GA firing? ----
    print("\n==== SEPARATOR SEARCH (pooled firing pixels: 016 FP vs the 4 must-hold GA eyes) ====")
    fp_e = next(fr for fr in feat_rows if fr[2] == "control_FP")
    ga_pack = [fr for fr in feat_rows if fr[2] == "ga"]
    names = ["elev_um", "r_centre_mm", "core_edge_px", "sig6_norm", "hyper6"]
    for j, nm in zip(range(3, 8), names):
        fpv = fp_e[j]
        gav = np.concatenate([fr[j] for fr in ga_pack]) if ga_pack else np.array([])
        if fpv.size == 0 or gav.size == 0:
            continue
        # is FP HIGHER or LOWER on this feature? report both tails so a cut can be read off.
        print(f"  {nm:13} 016FP[p10/50/90]={np.percentile(fpv,10):7.2f}/{np.percentile(fpv,50):7.2f}/"
              f"{np.percentile(fpv,90):7.2f}   GA[p10/50/90]={np.percentile(gav,10):7.2f}/"
              f"{np.percentile(gav,50):7.2f}/{np.percentile(gav,90):7.2f}")


if __name__ == "__main__":
    main()
