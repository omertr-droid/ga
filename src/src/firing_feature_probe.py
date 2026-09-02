#!/usr/bin/env python
"""PHASE-1 CHARACTERIZATION — per-FIRING-COLUMN feature distributions for the FP control (016 OD)
vs the must-hold true-GA eyes (005 OD, 005 OS faint, 008 OD, 015 OD), plus stay-clean controls.

We rebuild the EXACT oac_ga.footprint firing boolean (radial2 baseline, DL BM), then for every FIRING
en-face pixel (the cRORA mask after holes/crora) we read four per-column features aligned to rpe6:
  - elevation (um)   = RPE->BM lift (drusen high / GA low)            elev6
  - edge-distance    = distance (mm) to the nearest in-field core edge (central vs field-margin)
  - whole-col SNR    = whole-A-scan mean intensity en-face            sig6  (BM-independent vignette cue)
  - sub-BM hyper      = the per-eye normalised hypertransmission       hyper6 (from prep)

The QUESTION phase-1 answers: in REAL GA (RPE GONE) elevation is LOW, the lesion is CENTRAL (high
edge-distance) with retained inner-retinal SNR; characterise that so phase-2 knows what a gate must
PRESERVE. We flag any real-GA firing that itself sits at HIGH elevation or LOW SNR / near the edge (a
gate tuned to kill 016 would erase it).

Run (repo root):  oct_env\\Scripts\\python.exe src\\firing_feature_probe.py
Writes results/firing_features.csv (per-eye firing-column percentile summary) and prints the tables.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["OCT_BM_DL"] = "1"   # MUST be set before importing bm_dl

import numpy as np
from scipy.ndimage import distance_transform_edt

import bm_dl
import m3_projections as mp
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import projection as proj

MMPP = oac_ga.MMPP

# (subject, eye, label). FP = the lone remaining false-positive control. must-hold = true GA we must
# preserve (005 OS is the FAINT erase-risk). stay-clean = controls that must remain ~0.
EYES = [
    ("NHAMD-003-016-V2", "OD", "FP"),        # PLEX 0, radial2 reads ~1.3 -> the target
    ("NHAMD-003-005-V3", "OD", "GA"),        # gold Dice 0.940, PLEX 1.08
    ("NHAMD-003-005-V3", "OS", "GA-faint"),  # small/faint, the erase-risk stress; PLEX 0.57
    ("NHAMD-003-008-V1", "OD", "GA-large"),  # large eccentric; PLEX 13.78
    ("NHAMD-003-015-V3", "OD", "GA"),        # PLEX 1.99
    ("NHAMD-003-002-V2", "OS", "control"),   # stay-clean
    ("NHAMD-003-006-V3", "OS", "control"),   # stay-clean
    ("NHAMD-003-012-V3", "OD", "control"),   # stay-clean
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


def pct(a, q):
    return float(np.percentile(a, q)) if a.size else float("nan")


def summarize(name, vals):
    """Return p10/p25/p50/p75/p90 of a 1-D array (NaN-safe)."""
    if vals.size == 0:
        return {f"{name}_p{q}": float("nan") for q in (10, 25, 50, 75, 90)}
    return {f"{name}_p{q}": pct(vals, q) for q in (10, 25, 50, 75, 90)}


def main():
    print(f"bm_dl.active()={bm_dl.active()} backend={bm_dl.backend()}", flush=True)
    e2e = e2e_lookup()
    plex = plex_lookup()

    rows = []
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

        # REFERENCE firing mask = the production gate (radial2, DL BM) -- the cRORA en-face GA mask.
        mask, area = oac_ga.footprint(p, 0.50)

        # ---- per-column feature en-faces aligned to rpe6 (per the workflow recipe) ----
        oac = mp.oac_volume(ov.vol)
        elev_nat = np.clip((bm - mp.band_argmax_row(oac, bm, *mp.OAC_RPE_UM)) * mp.AX, 0.0, None)  # (n,W) um
        elev6 = proj.to_enface(mp.destripe2d(elev_nat, signed=False), ov.fov_mm)        # RPE->BM elevation
        sig6 = proj.to_enface(ov.vol.mean(axis=1).astype("float32"), ov.fov_mm)         # whole-col SNR
        hyper6 = p["hyper6"]                                                            # sub-BM hyper (norm)
        core = p["core"]

        # edge-distance (mm): distance from each pixel to the nearest NON-core pixel (field-margin/vignette).
        # High = central/interior; 0 = on the field edge. Computed inside core only.
        edist6 = distance_transform_edt(core) * MMPP                                    # mm

        # rpe-loss DEPTH = how far below baseline the firing column sits (fraction): base/loss ratio proxy
        # (smaller loss6/base = darker = stronger RPE loss). Reported for context.
        with np.errstate(divide="ignore", invalid="ignore"):
            lossratio6 = np.where(p["base"] > 1e-6, p["loss6"] / p["base"], np.nan)

        # SNR normalised to the eye's in-field median so eyes are comparable (a relative inner-retina cue).
        sig_med = float(np.nanpercentile(sig6[core], 50)) + 1e-6
        signorm6 = sig6 / sig_med

        m = np.asarray(mask, bool)
        nfire = int(m.sum())
        pl = plex.get((subject, eye), float("nan"))
        print(f"\n==== {subject[-7:]}_{eye} [{label}] plex={pl:.3f} area={area:.3f} "
              f"firing_px={nfire} ====", flush=True)

        if nfire == 0:
            rows.append(dict(subject=subject, eye=eye, label=label, plex=pl, area=area, nfire=0))
            print("   (no firing -- nothing to characterise)", flush=True)
            continue

        elev_f = elev6[m]
        edist_f = edist6[m]
        signorm_f = signorm6[m]
        hyper_f = hyper6[m]
        lossr_f = lossratio6[m]
        lossr_f = lossr_f[np.isfinite(lossr_f)]

        # field half-extent (mm) to interpret edge-distance: max edist over core = field 'radius'.
        field_reach = float((distance_transform_edt(core) * MMPP).max()) if core.any() else float("nan")

        rec = dict(subject=subject, eye=eye, label=label, plex=pl, area=area, nfire=nfire,
                   field_reach_mm=field_reach)
        rec.update(summarize("elev", elev_f))
        rec.update(summarize("edist", edist_f))
        rec.update(summarize("signorm", signorm_f))
        rec.update(summarize("hyper", hyper_f))
        rec.update(summarize("lossr", lossr_f))

        # ---- the eye-independent gate-fraction probes: what fraction of THIS eye's firing would be
        #      ERASED by each candidate gate, at a few cut levels ----
        for X in (20.0, 25.0, 30.0, 40.0):
            rec[f"erase_elev_gt{int(X)}"] = float((elev_f > X).mean())          # drusen-elevation gate
        for D in (0.30, 0.50, 0.75, 1.00):
            rec[f"erase_edist_lt{D:.2f}"] = float((edist_f < D).mean())         # field-edge gate
        for Y in (0.50, 0.65, 0.80, 1.00):
            rec[f"erase_signorm_lt{Y:.2f}"] = float((signorm_f < Y).mean())     # low-SNR gate
        rows.append(rec)

        print(f"   elev  um : p10/25/50/75/90 = {pct(elev_f,10):5.1f} {pct(elev_f,25):5.1f} "
              f"{pct(elev_f,50):5.1f} {pct(elev_f,75):5.1f} {pct(elev_f,90):5.1f}", flush=True)
        print(f"   edist mm : p10/25/50/75/90 = {pct(edist_f,10):5.2f} {pct(edist_f,25):5.2f} "
              f"{pct(edist_f,50):5.2f} {pct(edist_f,75):5.2f} {pct(edist_f,90):5.2f}  "
              f"(field reach {field_reach:.2f}mm)", flush=True)
        print(f"   SNRnorm  : p10/25/50/75/90 = {pct(signorm_f,10):5.2f} {pct(signorm_f,25):5.2f} "
              f"{pct(signorm_f,50):5.2f} {pct(signorm_f,75):5.2f} {pct(signorm_f,90):5.2f}", flush=True)
        print(f"   hyper    : p10/25/50/75/90 = {pct(hyper_f,10):5.2f} {pct(hyper_f,25):5.2f} "
              f"{pct(hyper_f,50):5.2f} {pct(hyper_f,75):5.2f} {pct(hyper_f,90):5.2f}", flush=True)
        print(f"   loss/base: p10/25/50/75/90 = {pct(lossr_f,10):5.2f} {pct(lossr_f,25):5.2f} "
              f"{pct(lossr_f,50):5.2f} {pct(lossr_f,75):5.2f} {pct(lossr_f,90):5.2f}", flush=True)
        print(f"   erase-frac elev>30um = {(elev_f>30).mean():.2f} | edist<0.50mm = "
              f"{(edist_f<0.50).mean():.2f} | SNRnorm<0.65 = {(signorm_f<0.65).mean():.2f}", flush=True)

    # ---- write CSV ----
    out_csv = os.path.join(RESULTS_DIR, "firing_features.csv")
    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, ""))
                            for k in keys})
        print(f"\nwrote {out_csv}", flush=True)

    # ---- compact comparison table (median features per eye) ----
    print("\n==== FIRING-COLUMN MEDIANS (p50) PER EYE ====")
    hdr = (f"{'eye':18}{'label':10}{'plex':>6}{'area':>7}{'nfire':>7}"
           f"{'elev50':>8}{'edist50':>8}{'reach':>7}{'SNR50':>7}{'hyper50':>8}{'lossr50':>8}")
    print(hdr)
    for r in rows:
        if r.get("nfire", 0) == 0:
            print(f"{r['subject'][-7:]+'_'+r['eye']:18}{r['label']:10}{r['plex']:6.2f}"
                  f"{r['area']:7.2f}{0:7d}   (no firing)")
            continue
        print(f"{r['subject'][-7:]+'_'+r['eye']:18}{r['label']:10}{r['plex']:6.2f}{r['area']:7.2f}"
              f"{r['nfire']:7d}{r['elev_p50']:8.1f}{r['edist_p50']:8.2f}{r['field_reach_mm']:7.2f}"
              f"{r['signorm_p50']:7.2f}{r['hyper_p50']:8.2f}{r['lossr_p50']:8.2f}")

    # ---- ERASE-FRACTION table: what each candidate gate would remove per eye (the decisive contrast) ----
    print("\n==== ERASE-FRACTION per candidate gate (fraction of THIS eye's firing removed) ====")
    print(f"{'eye':18}{'label':10}"
          f"{'elev>20':>8}{'elev>30':>8}{'elev>40':>8}"
          f"{'edst<.3':>8}{'edst<.5':>8}{'edst<.75':>9}"
          f"{'SNR<.5':>8}{'SNR<.65':>8}{'SNR<.8':>8}")
    for r in rows:
        if r.get("nfire", 0) == 0:
            continue
        print(f"{r['subject'][-7:]+'_'+r['eye']:18}{r['label']:10}"
              f"{r['erase_elev_gt20']:8.2f}{r['erase_elev_gt30']:8.2f}{r['erase_elev_gt40']:8.2f}"
              f"{r['erase_edist_lt0.30']:8.2f}{r['erase_edist_lt0.50']:8.2f}{r['erase_edist_lt0.75']:9.2f}"
              f"{r['erase_signorm_lt0.50']:8.2f}{r['erase_signorm_lt0.65']:8.2f}"
              f"{r['erase_signorm_lt0.80']:8.2f}")


if __name__ == "__main__":
    main()
