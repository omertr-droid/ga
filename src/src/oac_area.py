#!/usr/bin/env python
"""Single-eye TEST of the BM-anchored OAC GA pipeline.

Opens ONE cohort eye's native 6x6 volume (the same scan the reader uses), folds in the eye's
MANUALLY CORRECTED BM (reader/data_store/corrections/<eid>_<eye>/, via layers.effective_surfaces),
computes the three BM-anchored OAC channels (m3_projections), turns them into a GA footprint + mm^2
area (reader.core.footprint, the SAME column-collapse -> cRORA >=250um -> 6x6 area the runs use), and
writes a 4-panel figure to outputs/oac/.

The three channels (Vermeer mu = I/(2*sum_below), sampled relative to BM):
  OAC-max ABOVE BM   high = RPE present (strong attenuation), LOW = RPE gone = GA   (the primary signal)
  RPE->BM elevation  (BM row - OAC-peak row) um; large = drusen lift (RPE alive => GA-free); ~0 = flat
  sub-BM hyper       mean OAC just below BM; HIGH = light penetrates choroid = GA

A single eye cannot calibrate an ABSOLUTE OAC cutoff, so the GA call is a WITHIN-EYE rule: normalise the
RPE-OAC to the eye's own healthy-RPE high percentile, require above-median sub-BM hypertransmission, and
exclude drusen lift. The area is therefore a first look (a small threshold sweep is printed); the
calibrated number comes from the cohort LOO-CV in validate_area.py.

Run (from repo root):
  oct_env\\Scripts\\python.exe src\\oac_area.py NHAMD-003-005 V3 OD
  oct_env\\Scripts\\python.exe src\\oac_area.py NHAMD-003-005-V3 V3 OD --rpe-frac 0.45 --sub-pct 55
"""
import argparse
import csv
import os
import re
import sys

# --- imports: put repo root (for `reader...`) and src/ (for m3_projections/paths/qcviz) on the path ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter

import m3_projections as mp
import m3_slab
import qcviz as qv
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR
from reader.core import e2e_source, layers, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj
from reader.core.layer_store import JsonSidecarLayerStore

CORR_DIR = os.path.join(_REPO, "reader", "data_store", "corrections")
OUT = os.path.join(OUT_DIR, "oac")


def resolve(subject, visit, eye):
    """Find the eye in results/bm_worklist.csv -> (row dict, absolute E2E path). subject may be given
    with or without the trailing -V<visit>."""
    want = subject if re.search(r"-V\d+$", subject) else f"{subject}-{visit}"
    eye = eye.upper()
    path = os.path.join(RESULTS_DIR, "bm_worklist.csv")
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["subject"] == want and r["eye"].upper() == eye:
                return r, os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    raise SystemExit(f"{want} {eye} not found in {path}")


def load_gold_native(subject, eye, n_bscans):
    """Per-B-scan GA columns from the exported gold labels (label==1 = the annotated GA wedge), as a
    native (n_bscans, W) float map -- the in-frame ground truth for Dice. None if no labels are found."""
    lab = os.path.join(OUT_DIR, "ga_bscan_dataset", "labels")
    rows, found, W = [], False, None
    for i in range(n_bscans):
        p = os.path.join(lab, f"{subject}_{eye}_b{i:04d}.png")
        if os.path.exists(p):
            found = True
            col = (np.array(Image.open(p)) == 1).any(axis=0)
            W = len(col)
            rows.append(col)
        else:
            rows.append(None)
    if not found:
        return None
    return np.array([(r if r is not None else np.zeros(W, bool)) for r in rows], np.float32)


def main():
    ap = argparse.ArgumentParser(description="single-eye BM-anchored OAC GA area test")
    ap.add_argument("subject", help="e.g. NHAMD-003-005 or NHAMD-003-005-V3")
    ap.add_argument("visit", help="e.g. V3")
    ap.add_argument("eye", help="OD or OS")
    ap.add_argument("--rpe-frac", type=float, default=0.50,
                    help="GA where smoothed RPE-loss < frac * healthy-RPE baseline (default 0.50; "
                         "UNCALIBRATED -- the true iRORA->cRORA cut needs multiple eyes + a real gold)")
    ap.add_argument("--reducer", choices=("mean", "max"), default="mean",
                    help="reduce OAC over the RPE band per A-scan: 'mean' (robust to single-pixel speckle, "
                         "tracks partial RPE thinning; best on 005 OD) or 'max' (fragile to one bright "
                         "pixel; stricter complete-loss) (default mean)")
    ap.add_argument("--rpe-hi-pct", type=float, default=95.0,
                    help="percentile of RPE-OAC taken as the healthy-RPE baseline (default 95)")
    ap.add_argument("--smooth-px", type=float, default=2.0,
                    help="spatial smoothing of the en-face channels before the call (px; GA is >=250um "
                         "coherent, so this denoises the per-pixel decision) (default 2)")
    ap.add_argument("--sub-pct", type=float, default=0.0,
                    help="optional specificity gate: require sub-BM OAC above this within-eye percentile "
                         "(0=off; the sub-BM-OAC channel does NOT brighten on GA here, so off by default)")
    ap.add_argument("--elev-max", type=float, default=99999.0,
                    help="optional drusen exclusion: drop columns whose RPE->BM elevation exceeds this um "
                         "(large=off; the elevation map is stripe-noisy here, so off by default)")
    ap.add_argument("--min-diam-um", type=float, default=250.0, help="cRORA min diameter (default 250)")
    ap.add_argument("--margin-mm", type=float, default=0.30,
                    help="erode a ring this wide off the field edge before measuring (mm); the rim reads "
                         "low-OAC from weak signal / edge BM, not GA (default 0.30; 0=off)")
    ap.add_argument("--baseline", choices=("radial2", "trend", "global"), default="radial2",
                    help="healthy-RPE baseline: 'radial2' = foveal isotonic-radial + angular (PRODUCTION "
                         "DEFAULT; best on the linear/quad tradeoff), 'trend' = robust low-order polynomial "
                         "SURFACE, or 'global' = one p95 (flags the dim periphery as false GA) (default radial2)")
    ap.add_argument("--trend-order", type=int, default=2,
                    help="polynomial order of the 'trend' baseline surface (default 2 = quadratic)")
    ap.add_argument("--sig-frac", type=float, default=0.0,
                    help="EXPERIMENTAL signal-quality gate (default 0=OFF): drop pixels whose inner-retinal "
                         "signal is below this fraction of the field's typical. Cleans vignette corners BUT "
                         "also removes severe GA centres (inner retina thins there too) -- net harmful, off.")
    a = ap.parse_args()

    row, e2e_path = resolve(a.subject, a.visit, a.eye)
    subject, eye, adv = row["subject"], row["eye"].upper(), float(row["advRPE_area_mm2"])
    if not os.path.exists(e2e_path):
        raise SystemExit(f"E2E not found on disk: {e2e_path}")

    # --- open the same 6x6 volume the reader uses + fold in the corrected BM ---
    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    lstore = JsonSidecarLayerStore(CORR_DIR)
    ilm, bm = layers.effective_surfaces(ov, lstore)
    corrected = lstore.corrected_indices(ov.eid, ov.eye)
    print(f"{subject} {eye}: eid={ov.eid} vol_idx={idx} n={ov.n_bscans} H={ov.H} W={ov.W} "
          f"fov={tuple(round(f, 3) for f in ov.fov_mm)} bm_src={ov.bm_src} corrected_bscans={len(corrected)}")
    if ov.n_bscans != int(row["n_bscans"]):
        print(f"  WARNING: opened {ov.n_bscans} B-scans but worklist says {row['n_bscans']}")
    if not corrected:
        print(f"  WARNING: no BM corrections found under {os.path.join(CORR_DIR, ov.eid + '_' + ov.eye)} "
              f"-> using {ov.bm_src} BM (eid/path may differ from the reader's)")

    # --- detector core: RPE-loss en-face + measurement core + robust healthy baseline, from the SHARED
    # module reader/core/oac_ga (so this CLI and the reader's OAC GA button run identical code). ---
    P = oac_ga.prep(ov, bm, reducer=a.reducer, smooth_px=a.smooth_px, margin_mm=a.margin_mm,
                    baseline=a.baseline, trend_order=a.trend_order, rpe_hi_pct=a.rpe_hi_pct,
                    sig_frac=a.sig_frac)
    rpe6, loss6, core, base, g_base, rpe_ds = (P["rpe6"], P["loss6"], P["core"], P["base"],
                                               P["g_base"], P["rpe_nat"])
    mmpp2 = oac_ga.MMPP2

    # --- auxiliary channels (figure + the optional gates only; non-gating by default) ---
    oac = mp.oac_volume(ov.vol)
    elev = np.clip((bm.astype(np.float32) - mp.band_argmax_row(oac, bm, *mp.OAC_RPE_UM)) * mp.AX, 0.0, None)
    # sub-BM hypertransmission = 2nd cRORA criterion, as INTENSITY (m3_slab.hyper_enface) not OAC (OAC
    # normalizes illumination out, so the choroid does not brighten on GA in OAC).
    sub_ds = mp.destripe2d(m3_slab.hyper_enface(ov.vol, bm), signed=False)
    sub6 = proj.to_enface(sub_ds, ov.fov_mm)
    elev6 = proj.to_enface(elev, ov.fov_mm)
    sub_sm = gaussian_filter(sub6, a.smooth_px)
    elev_sm = gaussian_filter(elev6, a.smooth_px)

    def pct(m, p):
        return float(np.nanpercentile(m, p))

    sub_thr = pct(sub6, a.sub_pct)
    print("  channel percentiles (p1/50/90/95/99):")
    for nm, m in [("rpe_loss", rpe_ds), ("sub_bm", sub_ds), ("elev_um", elev)]:
        print(f"    {nm:8} " + " ".join(f"{pct(m, p):8.3f}" for p in (1, 50, 90, 95, 99)))

    def ga_mask(frac):
        if a.sub_pct <= 0 and a.elev_max >= 1e4:
            return oac_ga.footprint(P, frac, a.min_diam_um)[0]            # the shared (un-gated) path
        b = (loss6 < frac * base) & core                                 # + optional within-eye gates
        if a.sub_pct > 0:
            b = b & (sub_sm > sub_thr)
        if a.elev_max < 1e4:
            b = b & (elev_sm < a.elev_max)
        return fp.crora(binary_fill_holes(b), a.min_diam_um)

    # --- gold ground truth (the GA span you annotated + exported) for in-frame Dice ---
    gold_native = load_gold_native(subject, eye, ov.n_bscans)
    gold_mask = (proj.to_enface(gold_native, ov.fov_mm) > 0.5) if gold_native is not None else None
    gold_area = float(gold_mask.sum()) * mmpp2 if gold_mask is not None else None

    def dice_of(m):
        if gold_mask is None:
            return None
        return 2 * float((m & gold_mask).sum()) / (float(m.sum()) + float(gold_mask.sum()) + 1e-9)

    print("  RPE baseline: {} (global p{:g}={:.3f}{}) ; smooth {:g}px ; margin {:g}mm ; sub {} ; elev {}"
          .format(a.baseline, a.rpe_hi_pct, g_base,
                  f", trend median {float(np.median(base[core])):.3f}" if a.baseline == "trend" else "",
                  a.smooth_px, a.margin_mm,
                  f"p{a.sub_pct:g}" if a.sub_pct > 0 else "OFF",
                  f"<{a.elev_max:g}um" if a.elev_max < 1e4 else "OFF"))
    print(f"  reducer={a.reducer} ; sweep --rpe-frac (advRPE {adv:.3f}"
          + (f", gold {gold_area:.3f}" if gold_area is not None else "") + " mm2):")
    for frac in (0.40, 0.45, 0.50, 0.55, 0.60):
        mf = ga_mask(frac)
        d = dice_of(mf)
        star = "  <- selected" if abs(frac - a.rpe_frac) < 1e-9 else ""
        print(f"    rpe_frac={frac:.2f}  area = {float(mf.sum()) * mmpp2:6.3f} mm2"
              + (f"  Dice = {d:.3f}" if d is not None else "") + star)

    mask = ga_mask(a.rpe_frac)
    area = float(mask.sum()) * mmpp2
    dice = dice_of(mask)
    msg = f"  ==> OAC GA area = {area:.4f} mm2   advRPE = {adv:.4f} mm2   diff = {area - adv:+.4f} mm2"
    print(msg + (f"   Dice(OAC,gold) = {dice:.3f}" if dice is not None else ""))

    # --- figure: signal, OAC vs gold footprints, and the FP/FN error map ---
    os.makedirs(OUT, exist_ok=True)

    def g8(m):
        return qv.ensure_rgb(qv.norm8(np.nan_to_num(np.asarray(m, np.float32))))

    oac_tile = qv.draw_contour(g8(rpe6), mask, color=(0, 255, 0), thick=2)
    if gold_mask is not None:
        gold_tile = qv.draw_contour(g8(rpe6), gold_mask, color=(0, 255, 255), thick=2)
        fpfn = qv.ensure_rgb((qv.norm8(np.nan_to_num(rpe6)).astype(np.float32) * 0.45).astype(np.uint8))
        fpfn[mask & gold_mask] = (0, 230, 0)         # TP green
        fpfn[mask & ~gold_mask] = (255, 40, 40)      # FP red
        fpfn[~mask & gold_mask] = (40, 110, 255)     # FN blue
        tiles = [g8(rpe6), oac_tile, gold_tile, fpfn]
        titles = ["OAC RPE-loss (dark=GA)", f"OAC footprint {area:.2f} mm2",
                  f"gold label {gold_area:.2f} mm2", f"green=hit red=FP blue=FN  Dice {dice:.2f}"]
        htail = (f"OAC {area:.2f} vs advRPE {adv:.2f} vs gold {gold_area:.2f} mm2  Dice {dice:.2f}")
    else:
        tiles = [g8(rpe6), g8(sub6), oac_tile, g8(elev6)]
        titles = ["OAC RPE-loss (dark=GA)", "sub-BM hyper", f"OAC footprint {area:.2f} mm2", "RPE->BM elev"]
        htail = f"OAC {area:.2f} vs advRPE {adv:.2f} mm2"
    panel = qv.panel(
        tiles, titles,
        header=f"{subject} {eye}  BM-anchored OAC GA  |  {htail}  |  "
               f"reducer={a.reducer} frac={a.rpe_frac:g} margin={a.margin_mm:g}mm baseline={a.baseline} "
               f"corrected_bm={len(corrected)}/{ov.n_bscans}",
        mm_per_px=proj.ENFACE_MMPP)
    out_png = os.path.join(OUT, f"{subject}_{eye}_oac.png")
    qv.save_rgb(out_png, panel)
    print(f"  wrote {out_png}")
    return out_png


if __name__ == "__main__":
    main()
