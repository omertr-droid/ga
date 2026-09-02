#!/usr/bin/env python
"""Scatter + Bland-Altman agreement figure for the OLD vs NEW production default, from default_compare.csv.

Reads results/default_compare.csv (written by src/compare_default.py — no E2E/recompute needed) and renders
a 2x2 panel: PLEX-vs-ours scatter (OLD | NEW) over Bland-Altman (OLD | NEW), so the radial2+floor upgrade is
visible. Output -> outputs/default_compare_agreement.png.

Run (repo root):  oct_env\\Scripts\\python.exe src\\plot_default.py
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paths import OUT_DIR, RESULTS_DIR  # noqa: E402
from summarize_plex import ccc  # noqa: E402

CSV_IN = os.path.join(RESULTS_DIR, "default_compare.csv")
FIG = os.path.join(OUT_DIR, "default_compare_agreement.png")
CONTROL_THR = 0.05


def main():
    with open(CSV_IN, newline="") as f:
        rows = list(csv.DictReader(f))
    plex = np.array([float(r["plex"]) for r in rows])
    old = np.array([float(r["old"]) for r in rows])
    new = np.array([float(r["new"]) for r in rows])
    ctrl = plex < CONTROL_THR
    n = len(plex)

    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    hi = max(plex.max(), old.max(), new.max()) * 1.08

    for j, (ours, name) in enumerate([(old, "OLD (quad, no floor)"), (new, "NEW (radial2 + floor)")]):
        d = ours - plex
        r = np.corrcoef(plex, ours)[0, 1]
        bias, sd = d.mean(), d.std(ddof=1)
        lo, hiL = bias - 1.96 * sd, bias + 1.96 * sd

        a = ax[0, j]
        a.plot([0, hi], [0, hi], "k--", lw=1, label="perfect agreement")
        a.scatter(plex[~ctrl], ours[~ctrl], c="#1a6fb0", s=55, edgecolor="k", lw=0.5, label="GA eyes", zorder=3)
        a.scatter(plex[ctrl], ours[ctrl], c="#e07b00", s=55, edgecolor="k", lw=0.5, label="controls", zorder=3)
        a.set_xlabel("PLEX advRPE GA area (mm²)")
        a.set_ylabel("Our OCT-only GA area (mm²)")
        a.set_title(f"{name}\nr={r:.3f}  CCC={ccc(plex, ours):.3f}  bias={bias:+.2f}  MAE={np.abs(d).mean():.2f} mm²")
        a.set_xlim(-0.5, hi); a.set_ylim(-0.5, hi)
        a.legend(loc="upper left", fontsize=8); a.grid(alpha=0.25)

        b = ax[1, j]
        mean_ax = (plex + ours) / 2
        b.axhline(bias, color="b", lw=1.2, label=f"bias {bias:+.2f}")
        b.axhline(lo, color="r", ls="--", lw=1, label=f"95% LoA [{lo:+.2f}, {hiL:+.2f}]")
        b.axhline(hiL, color="r", ls="--", lw=1)
        b.axhline(0, color="k", lw=0.6, alpha=0.5)
        b.scatter(mean_ax[~ctrl], d[~ctrl], c="#1a6fb0", s=50, edgecolor="k", lw=0.5, zorder=3)
        b.scatter(mean_ax[ctrl], d[ctrl], c="#e07b00", s=50, edgecolor="k", lw=0.5, zorder=3)
        b.set_xlabel("Mean of the two methods (mm²)")
        b.set_ylabel("Difference  ours − PLEX (mm²)")
        b.set_title(f"{name} — Bland–Altman")
        b.legend(loc="upper right", fontsize=8); b.grid(alpha=0.25)

    fig.suptitle(f"OCT-only GA area vs PLEX advRPE — production default before/after ({n} qc_ok eyes, DL BM)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
