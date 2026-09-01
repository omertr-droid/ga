#!/usr/bin/env python
"""Turn results/plex_compare.csv into a doctor-readable agreement report + figures.

Reads the per-eye OCT-vs-PLEX areas produced by src/compare_plex.py and reports, for each scenario:
  Scenario A (HYBRID) = validated BM where validated, DL BM otherwise   (columns A_quad / A_lin)
  Scenario B (ALL-DL) = DL BM for every eye                              (columns B_quad / B_lin)
descriptive agreement statistics (bias, MAE, correlation, Lin's CCC, Bland-Altman limits, % within
tolerance), control specificity, the per-eye disagreement table, and scatter + Bland-Altman figures.

Run (repo root, after compare_plex.py):
  oct_env\\Scripts\\python.exe src\\summarize_plex.py
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
from scipy import stats as st  # noqa: E402

from paths import OUT_DIR, RESULTS_DIR  # noqa: E402

CSV_IN = os.path.join(RESULTS_DIR, "plex_compare.csv")
MD_OUT = os.path.join(RESULTS_DIR, "plex_compare_summary.md")
FIG_OUT = os.path.join(OUT_DIR, "plex_agreement.png")

CONTROL_THR = 0.05      # PLEX < this mm^2 = a no-GA control eye
CALL_THR = 0.25         # we "call no significant GA" below this mm^2 (cRORA ~250um floor)


def ccc(x, y):
    """Lin's concordance correlation coefficient (agreement incl. bias, not just correlation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(), y.var()
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    denom = vx + vy + (x.mean() - y.mean()) ** 2
    return 2 * cov / denom if denom > 0 else float("nan")


def stat_block(plex, ours):
    """All agreement metrics for paired arrays (plex reference, our measurement)."""
    plex, ours = np.asarray(plex, float), np.asarray(ours, float)
    d = ours - plex
    n = len(plex)
    out = {
        "n": n,
        "mean_plex": plex.mean(), "mean_ours": ours.mean(),
        "bias": d.mean(), "bias_sd": d.std(ddof=1) if n > 1 else float("nan"),
        "mae": np.abs(d).mean(), "median_ae": np.median(np.abs(d)),
        "rmse": np.sqrt((d ** 2).mean()),
        "loa_lo": d.mean() - 1.96 * (d.std(ddof=1) if n > 1 else 0),
        "loa_hi": d.mean() + 1.96 * (d.std(ddof=1) if n > 1 else 0),
        "within_0p5": float(np.mean(np.abs(d) <= 0.5) * 100),
        "within_1p0": float(np.mean(np.abs(d) <= 1.0) * 100),
        "pearson_r": st.pearsonr(plex, ours)[0] if n > 1 else float("nan"),
        "spearman_r": st.spearmanr(plex, ours)[0] if n > 1 else float("nan"),
        "ccc": ccc(plex, ours),
    }
    out["r2"] = out["pearson_r"] ** 2
    return out


def rel_errors(plex, ours, thr=0.5):
    """Mean/median absolute PERCENT error on eyes with PLEX >= thr (where a % is meaningful)."""
    plex, ours = np.asarray(plex, float), np.asarray(ours, float)
    m = plex >= thr
    if m.sum() == 0:
        return float("nan"), float("nan"), 0
    pe = np.abs(ours[m] - plex[m]) / plex[m] * 100
    return float(pe.mean()), float(np.median(pe)), int(m.sum())


def fmt(v, p=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"


def main():
    with open(CSV_IN, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("plex_mm2", "eff_quad", "eff_lin", "dl_quad", "dl_lin",
                  "A_quad", "A_lin", "B_quad", "B_lin"):
            r[k] = float(r[k])
        r["is_validated"] = int(r["is_validated"])
        r["sat_band"] = int(r["sat_band"])

    plex = np.array([r["plex_mm2"] for r in rows])
    is_ctrl = plex < CONTROL_THR
    ga = ~is_ctrl
    n_val = sum(r["is_validated"] for r in rows)

    # Three BM regimes × two baselines. "Reader-as-run" = exactly what the reader app shows today
    # (validated BM where validated, else the classical self-seg BM) = the eff_* columns.
    scenarios = {
        "eff_lin": "Reader-as-run (validated BM else classical self-seg) · linear",
        "A_lin": "Hybrid (validated BM else DL) · linear",
        "B_lin": "All-DL (DL BM everywhere) · linear",
        "eff_quad": "Reader-as-run · quadratic",
        "A_quad": "Hybrid · quadratic",
        "B_quad": "All-DL · quadratic",
    }
    blocks = {}
    for col in scenarios:
        ours = np.array([r[col] for r in rows])
        blocks[col] = {
            "all": stat_block(plex, ours),
            "ga": stat_block(plex[ga], ours[ga]),
            "rel": rel_errors(plex, ours),
            "spec_n_ctrl": int(is_ctrl.sum()),
            "spec_correct": int(np.sum(ours[is_ctrl] < CALL_THR)),
            "ours": ours,
        }

    # ---------------- markdown report ----------------
    L = []
    L.append("# OCT-only GA area vs PLEX (advRPE) reference — cohort agreement\n")
    L.append(f"- Eyes analysed: **{len(rows)}** (qc_status == ok). GA-present (PLEX ≥ "
             f"{CONTROL_THR} mm²): **{int(ga.sum())}**; no-GA controls: **{int(is_ctrl.sum())}**.")
    L.append(f"- Bruch's-membrane source — hybrid scenario: **{n_val} eyes hand-validated**, "
             f"**{len(rows) - n_val} eyes via the DL BM model**.")
    L.append(f"- PLEX reference = Zeiss advRPE DL GA segmenter on the PLEX 6×6 cube (silver standard). "
             f"Our number = OAC RPE-loss detector (reader.core.oac_ga), OCT only.")
    L.append(f"- GA-area range in the cohort: {plex.min():.2f} → {plex.max():.2f} mm² "
             f"(median {np.median(plex):.2f}).\n")

    def metrics_table(title, key, subset="all"):
        L.append(f"\n### {title}\n")
        b = blocks[key][subset]
        rel = blocks[key]["rel"]
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Eyes (n) | {b['n']} |")
        L.append(f"| Mean PLEX area | {fmt(b['mean_plex'])} mm² |")
        L.append(f"| Mean our area | {fmt(b['mean_ours'])} mm² |")
        L.append(f"| Mean difference (bias, ours − PLEX) | {fmt(b['bias'])} mm² |")
        L.append(f"| Mean ABSOLUTE difference (MAE) | {fmt(b['mae'])} mm² |")
        L.append(f"| Median absolute difference | {fmt(b['median_ae'])} mm² |")
        L.append(f"| RMSE | {fmt(b['rmse'])} mm² |")
        L.append(f"| Correlation (Pearson r) | {fmt(b['pearson_r'],3)} |")
        L.append(f"| R² (variance explained) | {fmt(b['r2'],3)} |")
        L.append(f"| Spearman rank r | {fmt(b['spearman_r'],3)} |")
        L.append(f"| Lin's concordance (CCC) | {fmt(b['ccc'],3)} |")
        L.append(f"| Bland–Altman 95% limits of agreement | {fmt(b['loa_lo'])} to {fmt(b['loa_hi'])} mm² |")
        L.append(f"| Eyes within ±0.5 mm² of PLEX | {fmt(b['within_0p5'],0)}% |")
        L.append(f"| Eyes within ±1.0 mm² of PLEX | {fmt(b['within_1p0'],0)}% |")
        L.append(f"| Mean abs. % error (PLEX ≥ 0.5 mm²) | {fmt(rel[0],0)}% |")
        L.append(f"| Median abs. % error (PLEX ≥ 0.5 mm²) | {fmt(rel[1],0)}% |")

    L.append("\n## Headline — LINEAR baseline (the setting the reader app runs)\n")
    L.append("Three BM regimes. **Reader-as-run** = exactly what the reader shows today (validated BM "
             "where validated, classical self-seg BM otherwise). **Hybrid** swaps the DL BM model in for "
             "the non-validated eyes; **All-DL** uses the DL model everywhere. (Default frac=0.50.)\n")
    metrics_table("Reader-as-run (validated BM else self-seg) — ALL eyes", "eff_lin", "all")
    metrics_table("Hybrid (validated BM else DL) — ALL eyes", "A_lin", "all")
    metrics_table("All-DL (DL BM everywhere) — ALL eyes", "B_lin", "all")
    metrics_table("Reader-as-run — GA-present eyes only", "eff_lin", "ga")

    # control specificity
    L.append("\n## Control specificity (no-GA eyes, PLEX < 0.05 mm²)\n")
    L.append(f"A correct control = our area < {CALL_THR} mm² (below the cRORA size floor).\n")
    L.append("| Regime (linear) | Controls correctly read ~0 |")
    L.append("|---|---|")
    for key, lab in [("eff_lin", "Reader-as-run"), ("A_lin", "Hybrid"), ("B_lin", "All-DL")]:
        L.append(f"| {lab} | {blocks[key]['spec_correct']}/{blocks[key]['spec_n_ctrl']} |")
    L.append("\nPer control eye (our area, mm², linear):\n")
    ctrl_rows = [r for r, c in zip(rows, is_ctrl) if c]
    L.append("| Eye | PLEX | Reader-as-run | Hybrid | All-DL |")
    L.append("|---|---|---|---|---|")
    for r in sorted(ctrl_rows, key=lambda r: r["subject"]):
        L.append(f"| {r['subject'][-7:]} {r['eye']} | {fmt(r['plex_mm2'])} | "
                 f"{fmt(r['eff_lin'])} | {fmt(r['A_lin'])} | {fmt(r['B_lin'])} |")

    # per-eye table (linear), sorted by PLEX desc, all three regimes
    L.append("\n## Per-eye comparison (LINEAR), sorted by GA size\n")
    L.append("BM(read) = the Bruch's-membrane the reader uses (validated = hand-checked, else self-seg). "
             "%diff is vs the Reader-as-run column.\n")
    L.append("| Eye | BM(read) | PLEX | Reader-as-run | Hybrid | All-DL | Δ read (mm²) | Δ read (%) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["plex_mm2"]):
        bm = "validated" if r["is_validated"] else "self-seg"
        diff = r["eff_lin"] - r["plex_mm2"]
        pct = (diff / r["plex_mm2"] * 100) if r["plex_mm2"] >= CONTROL_THR else None
        flag = " ⚠️clip" if r["sat_band"] and r["subject"].endswith("008-V1") and r["eye"] == "OD" else ""
        L.append(f"| {r['subject'][-7:]} {r['eye']}{flag} | {bm} | {fmt(r['plex_mm2'])} | "
                 f"{fmt(r['eff_lin'])} | {fmt(r['A_lin'])} | {fmt(r['B_lin'])} | "
                 f"{diff:+.2f} | {('—' if pct is None else f'{pct:+.0f}%')} |")

    # baseline sensitivity (quad vs lin) for all three regimes
    L.append("\n## Baseline sensitivity (linear vs quadratic healthy-RPE trend)\n")
    L.append("The reader UI runs **linear**; **quadratic** is the code default. Both validated against the "
             "live reader to 4 decimals. The number is also frac-sensitive (default 0.50). Cohort agreement:\n")
    L.append("| Regime · baseline | Bias (mm²) | MAE (mm²) | R² | CCC | Within ±1 mm² |")
    L.append("|---|---|---|---|---|---|")
    for key, lab in [("eff_lin", "Reader-as-run · linear"), ("eff_quad", "Reader-as-run · quadratic"),
                     ("A_lin", "Hybrid · linear"), ("A_quad", "Hybrid · quadratic"),
                     ("B_lin", "All-DL · linear"), ("B_quad", "All-DL · quadratic")]:
        b = blocks[key]["all"]
        L.append(f"| {lab} | {fmt(b['bias'])} | {fmt(b['mae'])} | {fmt(b['r2'],3)} | "
                 f"{fmt(b['ccc'],3)} | {fmt(b['within_1p0'],0)}% |")

    L.append(f"\n\n*Figures: {os.path.relpath(FIG_OUT, _REPO)} (scatter + Bland–Altman).*")
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {MD_OUT}")

    # ---------------- figures: scatter + Bland-Altman, reader-as-run & all-DL (LINEAR) ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    val = np.array([r["is_validated"] for r in rows], bool)
    for j, (key, title) in enumerate([("eff_lin", "Reader-as-run (validated/self-seg)"),
                                      ("B_lin", "All-DL")]):
        ours = blocks[key]["ours"]
        b = blocks[key]["all"]
        ax = axes[0, j]
        hi = max(plex.max(), ours.max()) * 1.08
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="perfect agreement")
        ax.scatter(plex[val], ours[val], c="#1a8a3a", s=60, edgecolor="k", lw=0.5,
                   label="validated BM", zorder=3)
        ax.scatter(plex[~val], ours[~val], c="#e07b00", s=60, edgecolor="k", lw=0.5,
                   label="non-validated BM", zorder=3)
        ax.set_xlabel("PLEX advRPE GA area (mm²)")
        ax.set_ylabel("Our OCT-only GA area (mm²)")
        ax.set_title(f"{title}\nr={b['pearson_r']:.3f}  CCC={b['ccc']:.3f}  "
                     f"bias={b['bias']:+.2f} mm²")
        ax.set_xlim(-0.4, hi)
        ax.set_ylim(-0.4, hi)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)

        ax = axes[1, j]
        mean_ax = (plex + ours) / 2
        diff = ours - plex
        ax.axhline(b["bias"], color="b", lw=1.2, label=f"bias {b['bias']:+.2f}")
        ax.axhline(b["loa_lo"], color="r", ls="--", lw=1, label=f"95% LoA [{b['loa_lo']:.2f}, {b['loa_hi']:.2f}]")
        ax.axhline(b["loa_hi"], color="r", ls="--", lw=1)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.scatter(mean_ax[val], diff[val], c="#1a8a3a", s=55, edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(mean_ax[~val], diff[~val], c="#e07b00", s=55, edgecolor="k", lw=0.5, zorder=3)
        ax.set_xlabel("Mean of the two methods (mm²)")
        ax.set_ylabel("Difference  ours − PLEX (mm²)")
        ax.set_title(f"{title} — Bland–Altman")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("OCT-only GA area vs PLEX advRPE reference (LINEAR baseline)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_OUT, dpi=130)
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
