#!/usr/bin/env python
"""Autonomous hyperparameter sweep of the OAC GA detector vs the PLEX (advRPE) reference.

Generalises src/compare_plex.py. For every qc_ok cohort eye it opens the native 6x6 (97-line) volume
ONCE and computes the DL Bruch's-membrane surface ONCE (the all-DL scenario -- NO annotation, NO
training, pure inference + measurement, safe to run unattended). It then evaluates the OAC GA area over a
list of NAMED detector configs (each a one- or few-knob override of the reference = the SAME defaults the
reader/compare_plex run) and scores each config's COHORT AGREEMENT with the advRPE silver reference
(bias, MAE, RMSE, Pearson r, Lin's CCC, Bland-Altman 95% LoA, % within +-1 mm2) plus CONTROL SPECIFICITY
(no-GA eyes reading below the call floor). Dice vs the in-frame exported gold is tracked on 005 OD (the
one gold eye) as a GUARDRAIL so a config that improves cohort agreement cannot silently wreck the
validated eye.

prep() (the expensive OAC volume + robust baseline) is cached per (eye, prep-signature); the cheap
footprint() threshold is swept on top. The reference config reproduces compare_plex's headline
(quadratic, sig-gate on) so deltas are interpretable.

IMPORTANT: scoring is against advRPE = an automatic, cross-device SILVER reference that shares our
hypertransmission confound. A config that AGREES better with advRPE is a hypothesis, not proven truth;
the 005 Dice guardrail + a human-graded gold set are what finally adjudicate. The 250 um cRORA floor and
the dual RPE-loss+hypertransmission requirement are clinical DEFINITIONS, not knobs -- min_diam is swept
ONLY to report definition sensitivity, never to tune the production number.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\sweep_oac.py                # full grid, all qc_ok eyes
  oct_env\\Scripts\\python.exe src\\sweep_oac.py --only 005     # quick smoke test (one subject)
  oct_env\\Scripts\\python.exe src\\sweep_oac.py --quick        # reference + a few configs only
Output -> results/oac_sweep.csv (one row/config) + results/oac_sweep_summary.md (ranked).
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
from PIL import Image  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402
from reader.core import projection as proj  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "oac_sweep.csv")
OUT_MD = os.path.join(RESULTS_DIR, "oac_sweep_summary.md")
GOLD_DIR = os.path.join(OUT_DIR, "ga_bscan_dataset", "labels")

CONTROL_THR = 0.05      # PLEX < this mm^2 = a no-GA control eye
CALL_THR = 0.25         # we "call no significant GA" below this mm^2 (cRORA ~250 um floor)

# --- reference config = compare_plex / reader default (quadratic, vignette gate ON) ----------------
REF_PREP = dict(reducer="mean", smooth_px=2.0, margin_mm=0.30, baseline="trend",
                trend_order=2, rpe_hi_pct=95.0, sig_frac=0.5, base_cap=1.15, radial=False)
REF_FP = dict(frac=0.50, min_diam_um=250.0, hyper_fill=True, close_mm=0.15,
              hyper_frac=0.7, hyper_keep=0.4, fill_all_holes=True)


def merge(base, **over):
    d = dict(base)
    d.update(over)
    return d


def build_configs():
    """List of {name, knob, prep, fp}. `knob` = a short human label of what changed vs reference."""
    C = []

    def add(name, knob, prep=None, fp=None):
        C.append({"name": name, "knob": knob,
                  "prep": merge(REF_PREP, **(prep or {})), "fp": merge(REF_FP, **(fp or {}))})

    add("reference", "ref (quad, gate on)")
    # --- frac: the single biggest, explicitly-uncalibrated lever ---
    for f in (0.40, 0.45, 0.55, 0.60):
        add(f"frac_{f}", f"frac={f}", fp=dict(frac=f))
    # --- baseline order + the radial branch (currently dead) + base_cap (the corner-FP band-aid) ---
    add("linear", "trend_order=1", prep=dict(trend_order=1))
    add("radial", "radial=True", prep=dict(radial=True))
    add("radial_linear", "radial+order1", prep=dict(radial=True, trend_order=1))
    for bc in (1.05, 1.10, 1.25, 1.40):
        add(f"base_cap_{bc}", f"base_cap={bc}", prep=dict(base_cap=bc))
    add("radial_caploose", "radial+cap1.40", prep=dict(radial=True, base_cap=1.40))
    add("linear_caploose", "order1+cap1.25", prep=dict(trend_order=1, base_cap=1.25))
    # --- OAC reduction statistic + healthy-RPE percentile ---
    add("reducer_max", "reducer=max", prep=dict(reducer="max"))
    for p in (90.0, 97.5):
        add(f"rpe_hi_{p}", f"rpe_hi_pct={p}", prep=dict(rpe_hi_pct=p))
    # --- the documented divergent default: vignette signal gate off (= the CLI default) ---
    add("siggate_off", "sig_frac=0", prep=dict(sig_frac=0.0))
    # --- field-rim erosion width ---
    for m in (0.20, 0.50):
        add(f"margin_{m}", f"margin_mm={m}", prep=dict(margin_mm=m))
    # --- hypertransmission combiner knobs (specificity / large-lesion centre recovery) ---
    for hk in (0.2, 0.3, 0.5, 0.6):
        add(f"hyper_keep_{hk}", f"hyper_keep={hk}", fp=dict(hyper_keep=hk))
    for hf in (0.6, 0.85):
        add(f"hyper_frac_{hf}", f"hyper_frac={hf}", fp=dict(hyper_frac=hf))
    # --- the foveal-sparing / ring-lesion fill fix (oac_ga.py:141) ---
    add("fillfix", "fill_all_holes=False", fp=dict(fill_all_holes=False))
    add("linear_fillfix", "order1+fillfix", prep=dict(trend_order=1), fp=dict(fill_all_holes=False))
    # --- min_diam: DEFINITION SENSITIVITY ONLY (250 um is the cRORA standard; do not tune to fit) ---
    for d in (125.0, 200.0, 300.0):
        add(f"mindiam_{int(d)}", f"min_diam={int(d)}um (defn)", fp=dict(min_diam_um=d))
    return C


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    denom = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return 2 * cov / denom if denom > 0 else float("nan")


def metrics(plex, ours, is_ctrl):
    plex, ours = np.asarray(plex, float), np.asarray(ours, float)
    d = ours - plex
    n = len(plex)
    sd = d.std(ddof=1) if n > 1 else 0.0
    nctrl = int(is_ctrl.sum())
    spec = int(np.sum(ours[is_ctrl] < CALL_THR)) if nctrl else 0
    return {
        "n": n, "mean_plex": plex.mean(), "mean_ours": ours.mean(),
        "bias": d.mean(), "mae": np.abs(d).mean(), "rmse": float(np.sqrt((d ** 2).mean())),
        "r": float(np.corrcoef(plex, ours)[0, 1]) if n > 1 else float("nan"),
        "ccc": ccc(plex, ours),
        "loa_lo": d.mean() - 1.96 * sd, "loa_hi": d.mean() + 1.96 * sd,
        "within1": float(np.mean(np.abs(d) <= 1.0) * 100),
        "within0p5": float(np.mean(np.abs(d) <= 0.5) * 100),
        "n_ctrl": nctrl, "spec_correct": spec,
        "spec_frac": (spec / nctrl) if nctrl else float("nan"),
    }


def load_gold_native(subj, visit, eye, n_bscans):
    """In-frame exported-gold GA columns -> native (n,W) float, or None. Tries a couple subject forms."""
    cands = [subj, f"{subj}-{visit}"] if not subj.endswith(visit) else [subj]
    for s in cands:
        rows, found, W = [], False, None
        for i in range(n_bscans):
            p = os.path.join(GOLD_DIR, f"{s}_{eye}_b{i:04d}.png")
            if os.path.exists(p):
                found = True
                col = (np.array(Image.open(p)) == 1).any(axis=0)
                W = len(col)
                rows.append(col)
            else:
                rows.append(None)
        if found:
            return np.array([(r if r is not None else np.zeros(W, bool)) for r in rows], np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on subject (quick test)")
    ap.add_argument("--quick", action="store_true", help="reference + a handful of configs only")
    args = ap.parse_args()

    configs = build_configs()
    if args.quick:
        keep = {"reference", "linear", "radial", "fillfix", "frac_0.45", "frac_0.55", "siggate_off"}
        configs = [c for c in configs if c["name"] in keep]
    print(f"DL BM model: {bm_dl.model_path()}  backend={bm_dl.backend()}", flush=True)
    print(f"{len(configs)} configs", flush=True)

    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]
    if args.only:
        rows = [r for r in rows if args.only.lower() in r["subject"].lower()]
    print(f"{len(rows)} qc_ok eyes", flush=True)

    # per-config accumulators
    areas = {c["name"]: [] for c in configs}      # parallel to `eyes_meta`
    eyes_meta = []                                # (subj, eye, plex)
    gold_dice = {c["name"]: None for c in configs}

    raw_cache = {}
    for r in rows:
        subj, visit, eye = r["subject"], r["visit"], r["eye"].upper()
        e2e_path = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e_path):
            print(f"  SKIP {subj} {eye}: E2E missing", flush=True)
            continue
        try:
            plex = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            plex = float("nan")
        if e2e_path not in raw_cache:
            raw_cache.clear()
            raw_cache[e2e_path] = e2e_source.open_e2e(e2e_path)
        raw = raw_cache[e2e_path]
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol)                    # DL BM for every eye (no annotation)

        is_gold = ("005" in subj) and eye == "OD"
        gold_mask = None
        if is_gold:
            gn = load_gold_native(subj, visit, eye, ov.n_bscans)
            if gn is not None:
                gold_mask = proj.to_enface(gn, ov.fov_mm) > 0.5

        eyes_meta.append((subj, eye, plex))
        prep_cache = {}
        for c in configs:
            sig = tuple(sorted(c["prep"].items()))
            if sig not in prep_cache:
                prep_cache[sig] = oac_ga.prep(ov, bm, **c["prep"])
            P = prep_cache[sig]
            mask, area = oac_ga.footprint(P, **c["fp"])
            areas[c["name"]].append(area)
            if gold_mask is not None:
                inter = float((mask & gold_mask).sum())
                gold_dice[c["name"]] = 2 * inter / (float(mask.sum()) + float(gold_mask.sum()) + 1e-9)
        print(f"  {subj[-7:]} {eye:2}  PLEX={plex:6.2f}  ref_area="
              f"{areas['reference'][-1]:6.2f}  bm_src={ov.bm_src}", flush=True)

    plex = np.array([m[2] for m in eyes_meta], float)
    is_ctrl = plex < CONTROL_THR

    # --- score every config ---
    res = []
    for c in configs:
        ours = np.array(areas[c["name"]], float)
        mt = metrics(plex, ours, is_ctrl)
        mt.update(name=c["name"], knob=c["knob"], dice005=gold_dice[c["name"]])
        res.append(mt)
    ref = next(x for x in res if x["name"] == "reference")

    # --- CSV ---
    cols = ["name", "knob", "n", "mean_plex", "mean_ours", "bias", "mae", "rmse", "r", "ccc",
            "loa_lo", "loa_hi", "within1", "within0p5", "n_ctrl", "spec_correct", "spec_frac", "dice005"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in res:
            w.writerow({k: (round(x[k], 4) if isinstance(x.get(k), float) else x.get(k)) for k in cols})
    print(f"\nwrote {OUT_CSV}  ({len(res)} configs)", flush=True)

    # --- ranked summary.md ---
    def f(v, p=2):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"

    ref_dice = ref["dice005"]
    def safe(x):
        ok = (x["mae"] <= ref["mae"] + 1e-9) and (x["spec_frac"] >= ref["spec_frac"] - 1e-9)
        if ref_dice is not None and x["dice005"] is not None:
            ok = ok and (x["dice005"] >= ref_dice - 0.03)
        is_defn = "(defn)" in x["knob"]
        return ok and not is_defn and x["name"] != "reference"

    by_mae = sorted(res, key=lambda x: x["mae"])
    L = []
    L.append("# OAC GA detector — autonomous hyperparameter sweep vs PLEX (advRPE)\n")
    L.append(f"- Eyes: **{len(eyes_meta)}** qc_ok (GA-present {int((~is_ctrl).sum())}, "
             f"controls {int(is_ctrl.sum())}). BM = **DL model, every eye** (no annotation).")
    L.append(f"- Reference = reader/compare_plex default (quadratic baseline, vignette gate on). "
             f"Scoring vs advRPE = SILVER (automatic, cross-device) — agreement gains are hypotheses, "
             f"the 005 Dice column is the guardrail.\n")
    L.append(f"**Reference:** bias {f(ref['bias'])}  MAE {f(ref['mae'])}  r {f(ref['r'],3)}  "
             f"CCC {f(ref['ccc'],3)}  within±1 {f(ref['within1'],0)}%  "
             f"specificity {ref['spec_correct']}/{ref['n_ctrl']}  005 Dice {f(ref_dice,3)}\n")
    L.append("## Configs ranked by MAE (Δ vs reference)\n")
    L.append("| Config | Knob | bias | MAE | ΔMAE | r | CCC | ±1mm² | spec | 005 Dice | safe-gain |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for x in by_mae:
        dmae = x["mae"] - ref["mae"]
        flag = "✅" if safe(x) else ("·" if x["name"] != "reference" else "ref")
        L.append(f"| {x['name']} | {x['knob']} | {f(x['bias'])} | {f(x['mae'])} | {dmae:+.2f} | "
                 f"{f(x['r'],3)} | {f(x['ccc'],3)} | {f(x['within1'],0)}% | "
                 f"{x['spec_correct']}/{x['n_ctrl']} | {f(x['dice005'],3)} | {flag} |")
    gains = [x for x in res if safe(x)]
    L.append("\n## Safe improvements (lower MAE, no worse specificity, 005 Dice held)\n")
    if gains:
        for x in sorted(gains, key=lambda x: x["mae"]):
            L.append(f"- **{x['name']}** ({x['knob']}): MAE {f(ref['mae'])}→{f(x['mae'])} "
                     f"({x['mae']-ref['mae']:+.2f}), spec {ref['spec_correct']}→{x['spec_correct']}"
                     f"/{x['n_ctrl']}, 005 Dice {f(x['dice005'],3)}.")
    else:
        L.append("- None beat the reference on all guardrails — the default is on the Pareto front "
                 "for this silver reference; real gains likely need a learned baseline or gold labels.")
    L.append("\n## Caveats\n")
    L.append("- advRPE is an automatic, cross-device SILVER reference sharing our hypertransmission "
             "confound; lower MAE-vs-advRPE ≠ proven accuracy. min_diam rows are DEFINITION sensitivity "
             "(250 µm is the cRORA standard), excluded from 'safe-gain'. Dice exists only for 005 OD.")
    with open(OUT_MD, "w", encoding="utf-8") as fmd:
        fmd.write("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
