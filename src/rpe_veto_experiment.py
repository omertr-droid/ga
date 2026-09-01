#!/usr/bin/env python
"""SEPARATION TEST: does an RPE-PRESENCE veto (mp.rpe_surface prominence `prom`) fix the 016 OD false
positive WITHOUT erasing real GA?

STANDALONE -- does NOT edit reader/core/oac_ga.py or src/m3_projections.py (HARD RULE). For each eye:
  1. open on DL BM (OCT_BM_DL=1) -> bm = bm_dl.segment_volume(ov.vol),
  2. existing en-face GA call: oac_ga.footprint(oac_ga.prep(ov, bm, baseline='radial2'), 0.5),
  3. map the en-face GA mask -> native (n,W) columns (viewer.core.ga_native.enface_to_native),
  4. prom[n,W] = mp.rpe_surface(ov.vol, bm)[1]  (HIGH = RPE structurally present),
  5. report the prom DISTRIBUTION (p10/50/90) over the GA-CALLED columns,
  6. SWEEP a veto threshold T (drop GA columns with prom > T), recompute kept cRORA area per eye,
  7. report: the T that drives 016 OD < 0.25 mm2 and the WORST must-hold GA %-change there.

A round-trip control (native -> en-face -> cRORA with NO veto) is reported per eye so the kept-area
comparison is fair (any resampling loss is applied identically to FP and real GA).

Writes results/rpe_veto.csv.
Run:  oct_env\\Scripts\\python.exe src\\rpe_veto_experiment.py
"""
import csv
import os
import sys

os.environ["OCT_BM_DL"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image

import bm_dl
import m3_projections as mp
from paths import DATA_DIR, RESULTS_DIR, OUT_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from viewer.core import ga_native

# (key, eye, role). 016 OD = the FP to veto; the rest are MUST-HOLD GA eyes (+ a drusen probe pair).
EYES = [
    ("NHAMD-003-016-V2", "OD", "FP_control"),
    ("NHAMD-003-005-V3", "OD", "GA_gold_focal"),   # GOLD, Dice 0.940 -- erase-test
    ("NHAMD-003-005-V3", "OS", "GA_faint"),        # faint, erase-risk
    ("NHAMD-003-008-V1", "OD", "GA_large"),        # large confluent
    ("NHAMD-003-015-V3", "OD", "GA_focal"),
    ("NHAMD-003-011-V3", "OD", "GA+drusen"),       # drusen edge-case probe
    ("NHAMD-003-011-V3", "OS", "GA+drusen"),
]

# threshold sweep (prom: HIGH = RPE present -> veto)
VETO_T = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]
FP_KEY, FP_EYE = "NHAMD-003-016-V2", "OD"
TARGET = 0.25   # mm^2: 016 OD must fall under this


def resolve(key, eye):
    """E2E path + metadata row for an eye, from the BM worklist else the master pairing index."""
    eye = eye.upper()
    for fname, ok in (("bm_worklist.csv", None), ("spectralis_ga_pairing.csv", "ok")):
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["subject"] == key and r["eye"].upper() == eye and (ok is None or r.get("qc_status") == ok):
                    return r, os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return None, None


def load_gold_native(subject, eye, n):
    """In-frame gold GA columns (n,W) bool from outputs/ga_bscan_dataset/labels, or None."""
    lab = os.path.join(OUT_DIR, "ga_bscan_dataset", "labels")
    rows, found, W = [], False, None
    for i in range(n):
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
    return np.array([(r if r is not None else np.zeros(W, bool)) for r in rows], bool)


def kept_area(native_cols, ov):
    """native (n,W) bool GA columns -> en-face cRORA area mm^2 via the SAME pipeline primitive."""
    _, area = fp.footprint_from_flags(native_cols, ov.fov_mm)
    return area


def run_eye(key, eye, role):
    row, e2e = resolve(key, eye)
    if e2e is None or not os.path.exists(e2e):
        print(f"\n=== {key} {eye} ({role}): NOT FOUND / no E2E on disk -- skipped")
        return None
    raw = e2e_source.open_e2e(e2e)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    bm = bm_dl.segment_volume(ov.vol)
    adv = float(row.get("advRPE_area_mm2") or "nan")

    # --- existing en-face GA call (validated radial2 default) ---
    p = oac_ga.prep(ov, bm, baseline="radial2")
    mask, area0 = oac_ga.footprint(p, 0.5)
    ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, ov.n_bscans, ov.W).astype(bool)

    # --- prom (HIGH = RPE present) ---
    _, prom = mp.rpe_surface(ov.vol, bm)
    prom = np.asarray(prom, np.float32)

    n_called = int(ga_nat.sum())
    # round-trip control: native GA cols -> en-face cRORA, NO veto. (fairness baseline)
    area_rt = kept_area(ga_nat, ov) if n_called else 0.0
    rt_frac = area_rt / (area0 + 1e-9)

    print(f"\n=== {key} {eye} ({role}) ===")
    print(f"  PLEX(advRPE)={adv:.3f}  area0={area0:.4f}  GA-called A-scans={n_called}  "
          f"round-trip area(no veto)={area_rt:.4f} ({100 * rt_frac:.1f}% of area0)")
    if n_called == 0:
        print("  (no GA called -- veto is a no-op)")
        return dict(key=key, eye=eye, role=role, adv=adv, area0=area0, n_called=0,
                    area_rt=0.0, p10=np.nan, p50=np.nan, p90=np.nan, veto={t: 0.0 for t in VETO_T})

    promin = prom[ga_nat]
    p10, p50, p90 = np.percentile(promin, [10, 50, 90])
    frac_present = float((promin > 1.5).mean())
    print(f"  prom INSIDE GA call  p10/p50/p90 = {p10:.2f} / {p50:.2f} / {p90:.2f}   "
          f"(>1.5 'present': {100 * frac_present:.0f}% of called cols)")

    # gold split (decisive): does prom separate TRUE-GA cols from over-call cols?
    gold = load_gold_native(key, eye, ov.n_bscans)
    if gold is not None:
        tp = ga_nat & gold
        fpc = ga_nat & ~gold
        if tp.any():
            tq = np.percentile(prom[tp], [10, 50, 90])
            print(f"  [gold] prom on TRUE-GA cols   p10/50/90 = {tq[0]:.2f} / {tq[1]:.2f} / {tq[2]:.2f}  (n={int(tp.sum())})")
        if fpc.any():
            fq = np.percentile(prom[fpc], [10, 50, 90])
            print(f"  [gold] prom on OVER-CALL cols p10/50/90 = {fq[0]:.2f} / {fq[1]:.2f} / {fq[2]:.2f}  (n={int(fpc.sum())})")

    # --- sweep the veto threshold ---
    veto = {}
    print("    T     kept_cols   area_kept   d_area    frac_kept")
    for T in VETO_T:
        keep = ga_nat & ~(prom > T)
        a = kept_area(keep, ov)
        veto[T] = a
        print(f"   {T:4.1f}   {int(keep.sum()):8d}   {a:8.4f}  {a - area0:+8.4f}   {a / (area0 + 1e-9):6.2f}")

    return dict(key=key, eye=eye, role=role, adv=adv, area0=area0, n_called=n_called,
                area_rt=area_rt, p10=float(p10), p50=float(p50), p90=float(p90), veto=veto)


def main():
    print("RPE-presence VETO separation test (DL BM, baseline=radial2, footprint frac=0.5).")
    print(f"bm_dl.active()={bm_dl.active()}   prom = mp.rpe_surface[1]; HIGH = RPE present.")
    rows = [r for r in (run_eye(*e) for e in EYES) if r]

    # ----- separation summary -----
    print("\n\n========== SEPARATION: prom (p10/p50/p90) INSIDE the GA call ==========")
    print("eye/role".ljust(28) + "PLEX".rjust(7) + "area0".rjust(8) +
          "p10".rjust(7) + "p50".rjust(7) + "p90".rjust(7))
    for r in rows:
        print(f"{r['key'][-7:]}-{r['eye']} {r['role']}".ljust(28) +
              f"{r['adv']:7.2f}{r['area0']:8.3f}" +
              (f"{r['p10']:7.2f}{r['p50']:7.2f}{r['p90']:7.2f}" if r['n_called'] else "    n/a    n/a    n/a"))

    fp_row = next((r for r in rows if r["key"] == FP_KEY and r["eye"] == FP_EYE), None)

    # ----- find the T that drives the FP under TARGET; report worst must-hold there -----
    print("\n========== VETO sweep: vetoed area by global threshold T ==========")
    hdr = "eye/role".ljust(28) + "area0".rjust(8) + "".join(f"T={t}".rjust(8) for t in VETO_T)
    print(hdr)
    for r in rows:
        line = f"{r['key'][-7:]}-{r['eye']} {r['role']}".ljust(28) + f"{r['area0']:8.3f}"
        line += "".join(f"{r['veto'][t]:8.3f}" for t in VETO_T)
        print(line)

    print(f"\n========== Can a single T fix {FP_KEY[-7:]} {FP_EYE} (< {TARGET} mm2) WITHOUT gutting real GA? ==========")
    musthold = [r for r in rows if r["role"].startswith("GA") and "drusen" not in r["role"]]
    chosen_T = None
    if fp_row:
        # The LARGEST T that still pulls the FP under TARGET = the most GENEROUS veto (best chance for
        # real GA to survive). If even that gentle T guts the must-hold set, no T can save the veto.
        qualifying = [T for T in VETO_T if fp_row["veto"][T] < TARGET]
        if qualifying:
            chosen_T = max(qualifying)
        else:
            print(f"  No swept T drives the FP below {TARGET} mm2 (min vetoed = {min(fp_row['veto'].values()):.3f} at T={min(VETO_T)}).")
            chosen_T = min(VETO_T)
        fp_at = fp_row["veto"][chosen_T]
        print(f"  Smallest T with FP < {TARGET}:  T = {chosen_T}  ->  FP 016 OD = {fp_at:.4f} mm2 "
              f"(from {fp_row['area0']:.4f}, {100 * fp_at / (fp_row['area0'] + 1e-9):.0f}% kept)")
        print(f"  Must-hold GA eyes AT THAT T:")
        worst_name, worst_pct = None, 1.0
        for r in musthold:
            kept = r["veto"][chosen_T]
            frac = kept / (r["area0"] + 1e-9)
            print(f"    {r['key'][-7:]}-{r['eye']:2s} {r['role']:14s}  {r['area0']:7.3f} -> {kept:7.3f}  "
                  f"({100 * frac:5.0f}% kept, {100 * (frac - 1):+.0f}%)")
            if frac < worst_pct:
                worst_pct, worst_name = frac, f"{r['key'][-7:]}-{r['eye']}"
        print(f"\n  WORST must-hold at T={chosen_T}: {worst_name} retains {100 * worst_pct:.0f}% "
              f"({100 * (worst_pct - 1):+.0f}%).")

        # does the FP sit HIGHER (more 'present') than real GA, as the veto needs?
        gold = next((r for r in musthold if r["key"].endswith("005-V3") and r["eye"] == "OD"), None)
        print(f"\n  Does the FP's prom sit HIGHER than true GA (what a veto needs)?")
        print(f"    FP 016 OD     prom p10/50/90 = {fp_row['p10']:.2f} / {fp_row['p50']:.2f} / {fp_row['p90']:.2f}")
        if gold:
            print(f"    GOLD 005 OD   prom p10/50/90 = {gold['p10']:.2f} / {gold['p50']:.2f} / {gold['p90']:.2f}")
            sep = fp_row["p50"] - gold["p50"]
            print(f"    median gap (FP - gold) = {sep:+.2f}  ->  "
                  + ("FP barely higher; not separable" if abs(sep) < 0.3 else
                     ("FP higher (some signal)" if sep > 0 else "FP NOT higher -- veto impossible")))

    # ----- write results/rpe_veto.csv -----
    out = os.path.join(RESULTS_DIR, "rpe_veto.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "eye", "role", "plex_advRPE", "area0", "n_called",
                    "roundtrip_area_noveto", "prom_p10", "prom_p50", "prom_p90",
                    "chosen_T_fixes016", "area_at_chosenT", "frac_kept_at_chosenT"]
                   + [f"veto_T{t}" for t in VETO_T])
        for r in rows:
            cT = chosen_T if chosen_T is not None else ""
            aT = r["veto"].get(chosen_T, 0.0) if chosen_T is not None else ""
            fT = (aT / (r["area0"] + 1e-9)) if (chosen_T is not None and r["n_called"]) else ""
            w.writerow([r["key"], r["eye"], r["role"], f"{r['adv']:.4f}", f"{r['area0']:.4f}",
                        r["n_called"], f"{r['area_rt']:.4f}",
                        f"{r['p10']:.3f}" if r["n_called"] else "",
                        f"{r['p50']:.3f}" if r["n_called"] else "",
                        f"{r['p90']:.3f}" if r["n_called"] else "",
                        cT, (f"{aT:.4f}" if aT != "" else ""), (f"{fT:.4f}" if fT != "" else "")]
                       + [f"{r['veto'].get(t, 0.0):.4f}" for t in VETO_T])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
