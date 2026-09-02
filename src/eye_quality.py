#!/usr/bin/env python
"""Per-eye QUALITY / striping metrics for the OAC GA detector — a DIAGNOSTIC experiment (no production change).

The failure diagnosis (docs/GA_METHOD_ASSESSMENT.md §8) found the dominant scatter driver is low-SNR /
slow-axis STRIPING of the OAC RPE-loss en-face, which both manufactures speckle footprint (over-call) and
fragments real lesions below the cRORA gate (under-call). This script measures, per qc_ok eye (DL BM +
reference config), candidate quality metrics and checks whether they SEPARATE the known striping-failure eyes
from the clean agreers. If they do, a per-eye low-confidence FLAG is justified (then a trivial wire-in).

Metrics on the reference run (rpe6 = OAC RPE-loss en-face, core = measurement field, mask = cRORA footprint):
  hf_noise    std(rpe6 - gaussian(rpe6,4)) over core / median(rpe6[core])   -- general speckle/stripe energy
  stripe_row  median |adjacent-row-mean diff| over core / median           -- SLOW-AXIS (B-scan) striping
  stripe_col  median |adjacent-col-mean diff| over core / median           -- fast-axis striping
  n_comp      connected components in the final footprint                  -- speckle => many; lesion => few
  largest_frac  largest component area / total footprint area              -- 1 => one coherent lesion
  fill_frac   footprint area / core area                                   -- high on a control => suspicious

Run (repo root):
  oct_env\\Scripts\\python.exe src\\eye_quality.py            # all qc_ok eyes
  oct_env\\Scripts\\python.exe src\\eye_quality.py --only 005 # smoke test
Output -> results/eye_quality.csv + console separation report.
"""
import argparse
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage import measure  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402
from sweep_oac import REF_FP, REF_PREP  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "eye_quality.csv")

# Ground-truth labels from the §8 image diagnosis (subject-substr, eye) for the separation check.
BAD = {("015", "OS"), ("014", "OD"), ("001", "OD"), ("010", "OD"), ("016", "OD")}   # striping failures
GOOD = {("004", "OD"), ("004", "OS"), ("005", "OD"), ("005", "OS"),
        ("011", "OD"), ("011", "OS")}                                                # clean agreers


def label_of(subject, eye):
    pat = subject.split("-")[2] if len(subject.split("-")) > 2 else subject
    for s, e in BAD:
        if s in subject and e == eye:
            return "BAD"
    for s, e in GOOD:
        if s in subject and e == eye:
            return "GOOD"
    return ""


def metrics(P, mask):
    rpe6, core = P["rpe6"], P["core"]
    med = float(np.median(rpe6[core])) + 1e-9
    hf = rpe6 - gaussian_filter(rpe6, 4.0)
    hf_noise = float(np.std(hf[core])) / med

    # directional striping: mean of a line over the field, then high-freq adjacent differences
    rowmean = np.array([rpe6[r, core[r]].mean() if core[r].any() else np.nan for r in range(rpe6.shape[0])])
    colmean = np.array([rpe6[core[:, c], c].mean() if core[:, c].any() else np.nan
                        for c in range(rpe6.shape[1])])
    stripe_row = float(np.nanmedian(np.abs(np.diff(rowmean)))) / med
    stripe_col = float(np.nanmedian(np.abs(np.diff(colmean)))) / med

    lbl = measure.label(mask)
    areas = [r.area for r in measure.regionprops(lbl)]
    n_comp = len(areas)
    tot = float(mask.sum())
    largest_frac = (max(areas) / tot) if areas else float("nan")
    fill_frac = tot / (float(core.sum()) + 1e-9)
    return dict(hf_noise=hf_noise, stripe_row=stripe_row, stripe_col=stripe_col,
                n_comp=n_comp, largest_frac=largest_frac, fill_frac=fill_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    print(f"DL BM: {bm_dl.model_path()} backend={bm_dl.backend()}", flush=True)

    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]
    if args.only:
        rows = [r for r in rows if args.only.lower() in r["subject"].lower()]

    out, raw_cache = [], {}
    for r in rows:
        subj, visit, eye = r["subject"], r["visit"], r["eye"].upper()
        e2e_path = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e_path):
            print(f"  SKIP {subj} {eye}: E2E missing", flush=True)
            continue
        try:
            adv = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            adv = float("nan")
        if e2e_path not in raw_cache:
            raw_cache.clear()
            raw_cache[e2e_path] = e2e_source.open_e2e(e2e_path)
        raw = raw_cache[e2e_path]
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol)
        P = oac_ga.prep(ov, bm, **REF_PREP)
        mask, area = oac_ga.footprint(P, **REF_FP)
        m = metrics(P, mask)
        rec = dict(subject=subj, visit=visit, eye=eye, label=label_of(subj, eye),
                   plex=round(adv, 3), area=round(area, 3), abs_err=round(abs(area - adv), 3),
                   bm_src=ov.bm_src, **{k: round(v, 4) for k, v in m.items()})
        out.append(rec)
        print(f"  {subj[-7:]} {eye:2} {rec['label']:4} plex={adv:5.2f} ours={area:5.2f} "
              f"err={rec['abs_err']:4.2f} | hf={m['hf_noise']:.3f} strow={m['stripe_row']:.3f} "
              f"ncomp={m['n_comp']:3d} lfrac={m['largest_frac'] if not np.isnan(m['largest_frac']) else 0:.2f} "
              f"fill={m['fill_frac']:.3f}", flush=True)

    cols = ["subject", "visit", "eye", "label", "plex", "area", "abs_err", "bm_src",
            "hf_noise", "stripe_row", "stripe_col", "n_comp", "largest_frac", "fill_frac"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT_CSV} ({len(out)} eyes)", flush=True)

    # ---------- separation report ----------
    def vals(group, key):
        return [o[key] for o in out if o["label"] == group and o[key] is not None
                and not (isinstance(o[key], float) and np.isnan(o[key]))]

    bad_n = sum(1 for o in out if o["label"] == "BAD")
    good_n = sum(1 for o in out if o["label"] == "GOOD")
    print(f"\nSeparation check  (BAD={bad_n} striping eyes, GOOD={good_n} clean eyes)")
    print(f"{'metric':12} {'GOOD range':>22} {'BAD range':>22}  separates?")
    HIGHER_WORSE = {"hf_noise": 1, "stripe_row": 1, "stripe_col": 1, "n_comp": 1, "fill_frac": 1,
                    "largest_frac": -1}
    for k, dirn in HIGHER_WORSE.items():
        g, b = vals("GOOD", k), vals("BAD", k)
        if not g or not b:
            continue
        if dirn == 1:
            sep = max(g) < min(b)
            gr, br = f"[{min(g):.3f},{max(g):.3f}]", f"[{min(b):.3f},{max(b):.3f}]"
            gap = f"gap {min(b) - max(g):+.3f}" if sep else f"OVERLAP (good max {max(g):.3f} >= bad min {min(b):.3f})"
        else:
            sep = min(g) > max(b)
            gr, br = f"[{min(g):.3f},{max(g):.3f}]", f"[{min(b):.3f},{max(b):.3f}]"
            gap = f"gap {min(g) - max(b):+.3f}" if sep else f"OVERLAP (good min {min(g):.3f} <= bad max {max(b):.3f})"
        print(f"{k:12} {gr:>22} {br:>22}  {'YES ' + gap if sep else 'no  ' + gap}")

    # correlation of each metric with abs_err across all eyes (exclude the huge-lesion 008 as it is its own bucket)
    use = [o for o in out if "008" not in o["subject"]]
    err = np.array([o["abs_err"] for o in use], float)
    print("\nPearson r of each metric vs |area error| (excl. 008, n=%d):" % len(use))
    for k in ("hf_noise", "stripe_row", "stripe_col", "n_comp", "largest_frac", "fill_frac"):
        v = np.array([o[k] if o[k] is not None else np.nan for o in use], float)
        ok = np.isfinite(v) & np.isfinite(err)
        if ok.sum() > 2:
            print(f"  {k:12} r = {np.corrcoef(v[ok], err[ok])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
