#!/usr/bin/env python
"""LOO-CV-validated OCT-only GA area, vs the registration-free advRPE area. THE deliverable.

Light area-supervised models on the cached en-face features (ga_features.py), each thresholded ->
cRORA >=250 um -> area (mm2), with the single detection threshold calibrated by LEAVE-ONE-EYE-OUT
cross-validation (fit on the other eyes minimising area MAE incl. controls=0; predict the held-out eye):
  baseline    GA = f_trans > t                                   (transmission alone)
  gated lo/hi GA = f_gated(lo,hi) > t                            (transmission x RPE-gone gate; the fix)
  2-feature   GA = (f_trans > t1) AND (f_rpe > t2)               (older combiner, for reference)
The gated rows also SWEEP the gate (lo,hi) = tuning. Reported on the good-BM eyes (primary, cleanest)
and all device-BM eyes (secondary, more n). Cache-only + fast. Outputs metrics txt, per-eye CSV,
scatter + Bland-Altman for the winner.

Run: oct_env\\Scripts\\python.exe validate_area.py
"""
import csv
import glob
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import m3_projections as mp  # noqa: E402

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
FEAT = os.path.join(OUT_DIR, "features")
MMPP = 6.0 / 512.0
MIN_DIAM_PX = 0.250 / MMPP
PRESENT = 0.10                      # mm2 above which we call "GA present" (for sens/spec)


def crora_mask(binimg):
    binimg = morphology.remove_small_holes(binimg, area_threshold=int(MIN_DIAM_PX ** 2))
    lbl = measure.label(binimg)
    keep = np.zeros_like(binimg, bool)
    for r in measure.regionprops(lbl):
        if r.axis_major_length >= MIN_DIAM_PX:
            keep[lbl == r.label] = True
    return keep


def area_of(mask):
    return float(mask.sum()) * MMPP ** 2


def ccc(p, r):
    p, r = np.asarray(p), np.asarray(r)
    return float(2 * np.cov(p, r)[0, 1] / (p.var() + r.var() + (p.mean() - r.mean()) ** 2 + 1e-12))


def gated_of(d, lo, hi):
    """Recompute f_gated from cached natives at gate thresholds (lo,hi) -> the gate sweep / tuning."""
    f_trans = mp.to_6mm(mp.destripe2d(d["t_nat"], signed=False), d["fov"])
    pres6 = mp.to_6mm(mp.destripe2d(d["p_nat"], signed=False), d["fov"])
    gate = mp.rpe_gone_gate(gaussian_filter(pres6, mp.GATE_SMOOTH_PX), lo, hi)
    return np.nan_to_num(np.clip(f_trans, 0, None) * gate, nan=0.0)


def load():
    good = set()
    with open(os.path.join(RESULTS_DIR, "bm_good.csv")) as f:
        for r in csv.DictReader(f):
            good.add((r["subject"], r["eye"]))
    data = []
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        if "f_gated" not in d:
            continue
        sub, eye = str(d["subject"]), str(d["eye"])
        data.append(dict(subject=sub, eye=eye, f_trans=d["f_trans"], f_rpe=d["f_rpe"],
                         t_nat=d["f_trans_nat"], p_nat=d["f_pres_nat"], fov=[float(v) for v in d["fov"]],
                         area=float(d["area"]), bm=str(d["bm_source"]), good=(sub, eye) in good))
    return data


def loo_predict(pred_grid, ref):
    """pred_grid: (N, *grid) precomputed areas. Returns held-out predictions (fit thr on the rest)."""
    N = len(ref)
    out = np.zeros(N)
    for e in range(N):
        tr = np.arange(N) != e
        err = np.abs(pred_grid[tr] - ref[tr].reshape((-1,) + (1,) * (pred_grid.ndim - 1))).mean(0)
        out[e] = pred_grid[e].ravel()[int(np.argmin(err))]
    return out


def grid1(feats, ref, lo_pct=55, hi_pct=99.7, n=24):
    pool = np.concatenate([f.ravel() for f in feats])
    T = np.unique(np.percentile(pool, np.linspace(lo_pct, hi_pct, n)))
    pg = np.array([[area_of(crora_mask(f > t)) for t in T] for f in feats])
    return loo_predict(pg, ref)


def grid2(fts, frs, ref):
    poolt = np.concatenate([f.ravel() for f in fts]); poolr = np.concatenate([f.ravel() for f in frs])
    T1 = np.unique(np.percentile(poolt, np.linspace(55, 99.5, 18)))
    T2 = np.unique(np.percentile(poolr, np.linspace(40, 99, 14)))
    pg = np.zeros((len(ref), len(T1), len(T2)))
    for i, (ft, fr) in enumerate(zip(fts, frs)):
        for a, t1 in enumerate(T1):
            base = ft > t1
            for b, t2 in enumerate(T2):
                pg[i, a, b] = area_of(crora_mask(base & (fr > t2)))
    return loo_predict(pg, ref)


def report(pred, ref, label):
    diff = pred - ref
    rr = np.corrcoef(pred, ref)[0, 1]
    ctrl = ref < 0.05
    pres_r, pres_p = ref > PRESENT, pred > PRESENT
    tp = int((pres_r & pres_p).sum()); tn = int((~pres_r & ~pres_p).sum())
    fp = int((~pres_r & pres_p).sum()); fn = int((pres_r & ~pres_p).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return dict(label=label, R2=rr ** 2, CCC=ccc(pred, ref), MAE=np.abs(diff).mean(),
                RMSE=np.sqrt((diff ** 2).mean()), bias=diff.mean(), loA=1.96 * diff.std(ddof=1),
                ctrlFP=float(pred[ctrl].mean()) if ctrl.any() else float("nan"),
                sens=sens, spec=spec)


def main():
    data = load()
    if not data:
        print("no feature caches with f_gated; run ga_features.py all first")
        return
    ref = np.array([d["area"] for d in data])
    good = np.array([d["good"] for d in data])
    dev = np.array([d["bm"] == "device" for d in data])
    gi = np.where(good)[0]
    print(f"loaded {len(data)} eyes; good-BM={good.sum()} (controls={int((ref[gi]<0.05).sum())}), "
          f"device-BM={dev.sum()}\n", flush=True)

    def sub(idx, fn):
        return idx, fn([data[i] for i in idx], ref[idx])

    rows, preds = [], {}
    # --- primary: good-BM eyes ---
    feats_t = [d["f_trans"] for d in data]
    feats_r = [d["f_rpe"] for d in data]
    GATES = [(1.0, 1.9), (0.8, 1.5), (1.2, 2.2)]
    print("=== GOOD-BM eyes (primary) ===", flush=True)
    p = grid1([feats_t[i] for i in gi], ref[gi]); preds["baseline"] = (gi, p)
    rows.append(report(p, ref[gi], "good: baseline trans"))
    for lo, hi in GATES:
        fg = [gated_of(data[i], lo, hi) for i in gi]
        p = grid1(fg, ref[gi], lo_pct=70)
        preds[f"gated {lo}/{hi}"] = (gi, p)
        rows.append(report(p, ref[gi], f"good: gated {lo}/{hi}"))
        print(f"  done gated {lo}/{hi}", flush=True)
    p = grid2([feats_t[i] for i in gi], [feats_r[i] for i in gi], ref[gi])
    preds["2feature"] = (gi, p)
    rows.append(report(p, ref[gi], "good: 2feature trans&rpe"))

    # --- secondary: all device-BM (more n) for baseline + default gate ---
    di = np.where(dev)[0]
    pb = grid1([feats_t[i] for i in di], ref[di]); rows.append(report(pb, ref[di], "dev-all: baseline trans"))
    fgd = [gated_of(data[i], 1.0, 1.9) for i in di]
    pg = grid1(fgd, ref[di], lo_pct=70); rows.append(report(pg, ref[di], "dev-all: gated 1.0/1.9"))

    hdr = f"{'model':26} {'R2':>5} {'CCC':>5} {'MAE':>6} {'RMSE':>6} {'bias':>6} {'LoA':>6} {'ctrlFP':>7} {'sens':>5} {'spec':>5}"
    print("\n" + hdr)
    lines = [hdr]
    for m in rows:
        s = (f"{m['label']:26} {m['R2']:5.2f} {m['CCC']:5.2f} {m['MAE']:6.3f} {m['RMSE']:6.3f} "
             f"{m['bias']:6.2f} {m['loA']:6.2f} {m['ctrlFP']:7.3f} {m['sens']:5.2f} {m['spec']:5.2f}")
        print(s); lines.append(s)
    with open(os.path.join(RESULTS_DIR, "area_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # winner among good-BM gated/baseline by MAE
    cands = {k: v for k, v in preds.items() if k != "2feature"}
    wname = min(cands, key=lambda k: report(cands[k][1], ref[cands[k][0]], k)["MAE"])
    widx, wpred = preds[wname]
    wref = ref[widx]
    print(f"\nwinner (good-BM, by MAE): {wname}")

    with open(os.path.join(RESULTS_DIR, "area_validation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "eye", "advRPE_mm2", f"pred_{wname}_mm2", "abs_err"])
        for i, pr in zip(widx, wpred):
            w.writerow([data[i]["subject"], data[i]["eye"], f"{ref[i]:.3f}", f"{pr:.3f}", f"{abs(pr-ref[i]):.3f}"])

    mx = max(wref.max(), wpred.max()) * 1.1
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(wref, wpred, c="tab:blue", s=44, edgecolor="k", lw=0.4)
    plt.plot([0, mx], [0, mx], "k--", lw=1)
    m = report(wpred, wref, wname)
    plt.xlabel("advRPE GA area (mm$^2$)"); plt.ylabel("OCT-only predicted area (mm$^2$)")
    plt.title(f"LOO-CV area, good-BM, {wname}\nR$^2$={m['R2']:.2f}  CCC={m['CCC']:.2f}  MAE={m['MAE']:.2f}  ctrlFP={m['ctrlFP']:.2f}")
    plt.xlim(0, mx); plt.ylim(0, mx); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "area_scatter.png"), dpi=130); plt.close()

    diff = wpred - wref; mean = (wpred + wref) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    plt.figure(figsize=(5.6, 4.4))
    plt.scatter(mean, diff, c="tab:blue", s=44, edgecolor="k", lw=0.4)
    for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        plt.axhline(y, color="r", ls=ls, lw=1)
    plt.xlabel("mean of OCT & advRPE (mm$^2$)"); plt.ylabel("OCT - advRPE (mm$^2$)")
    plt.title(f"Bland-Altman (good-BM, {wname}): bias={bias:+.2f}, LoA=[{bias-1.96*sd:+.2f},{bias+1.96*sd:+.2f}]")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "bland_altman.png"), dpi=130); plt.close()
    print("wrote area_metrics.txt, area_validation.csv, area_scatter.png, bland_altman.png")


if __name__ == "__main__":
    main()
