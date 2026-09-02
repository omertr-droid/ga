#!/usr/bin/env python
"""Score whether an OCT-INTRINSIC per-eye descriptor (no PLEX) can pick linear-vs-quad per eye.
Reads results/baseline_order_descriptors.csv (quad-mask shape descriptors + dl_quad/dl_lin/plex)."""
import csv
import numpy as np

rows = list(csv.DictReader(open("results/baseline_order_descriptors.csv")))


def F(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


eye = [r["eye"] for r in rows]
plex = np.array([F(r["plex"]) for r in rows])
q = np.array([F(r["dl_quad"]) for r in rows])
l = np.array([F(r["dl_lin"]) for r in rows])
eq, el = np.abs(q - plex), np.abs(l - plex)

D = {k: np.array([F(r[k]) for r in rows]) for k in
     ("area_frac", "ecc", "solidity", "frac_largest", "centroid_off", "n_comp")}

print(f"n={len(rows)}  always-QUAD MAE={eq.mean():.3f}  always-LIN MAE={el.mean():.3f}  "
      f"ORACLE MAE={np.minimum(eq, el).mean():.3f}\n")

# Which eyes does linear genuinely help (el < eq by > 0.1)?
lin_helps = (el < eq - 0.1)
quad_helps = (eq < el - 0.1)
print(f"eyes where LINEAR clearly better (>0.1): {int(lin_helps.sum())}   "
      f"QUAD clearly better: {int(quad_helps.sum())}   ~tie: {int((~lin_helps & ~quad_helps).sum())}\n")

print("eyes where LINEAR is the better order — their intrinsic descriptors:")
print(f"{'eye':12s} {'PLEX':>5s} {'q':>5s} {'l':>5s} | {'afrac':>5s} {'ecc':>4s} {'sol':>4s} {'fbig':>4s} {'off':>4s} {'nc':>3s}")
for i in np.where(lin_helps)[0]:
    print(f"{eye[i]:12s} {plex[i]:5.2f} {q[i]:5.2f} {l[i]:5.2f} | "
          f"{D['area_frac'][i]:5.3f} {D['ecc'][i]:4.2f} {D['solidity'][i]:4.2f} "
          f"{D['frac_largest'][i]:4.2f} {D['centroid_off'][i]:4.2f} {int(D['n_comp'][i]):3d}")
print("\neyes where QUAD is the better order:")
for i in np.where(quad_helps)[0]:
    print(f"{eye[i]:12s} {plex[i]:5.2f} {q[i]:5.2f} {l[i]:5.2f} | "
          f"{D['area_frac'][i]:5.3f} {D['ecc'][i]:4.2f} {D['solidity'][i]:4.2f} "
          f"{D['frac_largest'][i]:4.2f} {D['centroid_off'][i]:4.2f} {int(D['n_comp'][i]):3d}")


def best_threshold(desc, hi_uses_linear=True):
    """Sweep a threshold on `desc`: above -> linear (or below if not hi_uses_linear). Return best MAE."""
    best = (None, 1e9, None)
    vals = np.unique(desc[np.isfinite(desc)])
    for T in vals:
        pick_lin = (desc >= T) if hi_uses_linear else (desc <= T)
        err = np.where(pick_lin, el, eq)
        mae = err.mean()
        if mae < best[1]:
            best = (T, mae, int(pick_lin.sum()))
    return best


print("\n=== single-descriptor adaptive rule (above T -> LINEAR), best MAE ===")
print(f"{'descriptor':14s} {'bestT':>7s} {'MAE':>6s} {'n->lin':>6s}")
# detected quad AREA in mm^2 too (the strongest size signal)
alld = dict(D)
alld["quad_area_mm2"] = q
alld["max_area_mm2"] = np.maximum(q, l)
for k, v in alld.items():
    for direction in (True, False):
        T, mae, nlin = best_threshold(v, direction)
        tag = ">=" if direction else "<="
        print(f"{k:14s} {tag}{T:6.3f} {mae:6.3f} {nlin:6d}")

# Correlations of each descriptor with the SIGNED order benefit (el-eq): negative = linear helps
benefit = el - eq   # >0 => linear worse; <0 => linear better
print("\n=== corr(descriptor, el-eq)  [el-eq<0 means linear helps] ===")
for k, v in alld.items():
    m = np.isfinite(v) & np.isfinite(benefit)
    r = np.corrcoef(v[m], benefit[m])[0, 1]
    print(f"  {k:14s} r={r:+.3f}")
