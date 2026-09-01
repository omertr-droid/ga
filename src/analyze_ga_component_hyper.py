#!/usr/bin/env python
"""Post-hoc component-hypertransmission experiment, anchored by spatial OCT review.

This does not rerun inference and does not change production.  It asks whether a robust component-level
hypertransmission statistic can remove the weak-transmission components seen on the false-reference /
drusenoid audit cases while preserving the fixed 005 OD gold lesion and the visually unequivocal large
GA anchors.  Area metrics are reported only as secondary diagnostics; threshold feasibility is defined
by the predeclared anatomical anchors below, never by minimizing cohort MAE.
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MATRIX = os.path.join(RESULTS, "ga_experiment_matrix.csv")
COMPONENTS = os.path.join(RESULTS, "ga_experiment_components.csv")
SWEEP_OUT = os.path.join(RESULTS, "ga_component_hyper_sweep.csv")
FINDINGS_OUT = os.path.join(RESULTS, "ga_component_hyper_findings.md")

THRESHOLDS = tuple(round(0.40 + 0.01 * i, 2) for i in range(71))
PROVISIONAL = 0.60
JOINT_LOSS_P50 = 0.32
JOINT_HYPER_P50 = 0.70

# These are role-labelled from the lesion/B-scan audit, not selected by area residual.  Only 005 OD is
# in-frame gold.  The large lesions are preservation sentinels; they are not additional gold masks.
POSITIVE_GOLD = "NHAMD-003-005-V3_OD"
CONFIRMED_NO_GA = ("NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD")
OBVIOUS_GA_PRESERVE = (
    "NHAMD-003-003-V3_OS", "NHAMD-003-008-V1_OS", "NHAMD-003-015-V3_OD",
)
DISPLAY = {
    "NHAMD-003-001-V1_OD", "NHAMD-003-001-V1_OS", "NHAMD-003-003-V3_OS",
    "NHAMD-003-005-V3_OD", "NHAMD-003-006-V3_OD", "NHAMD-003-008-V1_OS",
    "NHAMD-003-010-V1_OD", "NHAMD-003-011-V3_OD", "NHAMD-003-014-V1_OD",
    "NHAMD-003-015-V3_OD", "NHAMD-003-026-V3_OD", "NHAMD-003-026-V3_OS",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def b(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")


def latest_complete_current(matrix):
    by_hash = defaultdict(list)
    for r in matrix:
        if r.get("protocol") == "current97_native" and r.get("intensity") == "display":
            by_hash[r["config_hash"]].append(r)
    complete = [(h, rows) for h, rows in by_hash.items() if len({r["slug"] for r in rows}) == 25]
    if not complete:
        raise RuntimeError("no complete 25-eye current/display arm")
    h, rows = complete[-1]
    ids = {r["config_id"] for r in rows}
    if len(ids) != 1:
        raise RuntimeError(f"ambiguous current config ids for {h}: {ids}")
    return h, next(iter(ids)), rows


def metric(pred, target, ref_key):
    diffs = []
    for slug, row in target.items():
        ref = f(row.get(ref_key))
        if ref is not None:
            diffs.append(pred[slug] - ref)
    n = len(diffs)
    return {
        "n": n,
        "bias": sum(diffs) / n,
        "mae": sum(abs(x) for x in diffs) / n,
        "rmse": math.sqrt(sum(x * x for x in diffs) / n),
    }


def detection(pred, target, class_key, floor=0.25):
    tp = fn = tn = fp = 0
    for slug, row in target.items():
        cls = row.get(class_key)
        call = pred[slug] >= floor
        if cls == "positive":
            tp += int(call); fn += int(not call)
        elif cls == "negative":
            fp += int(call); tn += int(not call)
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def fmt(x, n=3):
    return "—" if x is None else f"{float(x):.{n}f}"


def main():
    matrix = read_csv(MATRIX)
    comps = read_csv(COMPONENTS)
    config_hash, config_id, eyes = latest_complete_current(matrix)
    target = {r["slug"]: r for r in eyes}
    comp_by = defaultdict(list)
    for c in comps:
        if c.get("config_hash") == config_hash and c.get("config_id") == config_id:
            comp_by[c["slug"]].append(c)

    baseline = {slug: f(row["ours_area_mm2"]) or 0.0 for slug, row in target.items()}
    reconstructed = {
        slug: sum((f(c["measured_area_mm2"]) or 0.0) for c in comp_by[slug]
                  if b(c.get("kept_by_depth")) and b(c.get("in_measurement")))
        for slug in target
    }
    delta = max(abs(baseline[s] - reconstructed[s]) for s in target)
    if delta > 1e-9:
        raise AssertionError(f"current final components do not reconstruct the mask: {delta}")

    sweep = []
    predictions = {}
    for threshold in THRESHOLDS:
        pred = {
            slug: sum(
                (f(c["measured_area_mm2"]) or 0.0)
                for c in comp_by[slug]
                if b(c.get("kept_by_depth")) and b(c.get("in_measurement"))
                and f(c.get("hyper_p50")) is not None and f(c["hyper_p50"]) >= threshold
            )
            for slug in target
        }
        predictions[threshold] = pred
        raw = metric(pred, target, "plex_raw_mm2")
        corrected = metric(pred, target, "plex_corrected_mm2")
        det_raw = detection(pred, target, "ref_class_raw")
        det_corrected = detection(pred, target, "ref_class_corrected")
        gold_retain = pred[POSITIVE_GOLD] / max(baseline[POSITIVE_GOLD], 1e-12)
        preserve = min(pred[s] / max(baseline[s], 1e-12) for s in OBVIOUS_GA_PRESERVE)
        negatives_max = max(pred[s] for s in CONFIRMED_NO_GA)
        row = {
            "config_hash": config_hash,
            "config_id": config_id,
            "hyper_p50_threshold": threshold,
            "anchor_feasible": gold_retain >= 0.95 and preserve >= 0.95 and negatives_max < 0.05,
            "gold_005_retained_fraction": gold_retain,
            "large_GA_min_retained_fraction": preserve,
            "confirmed_no_GA_max_mm2": negatives_max,
            "raw_bias": raw["bias"], "raw_mae": raw["mae"], "raw_rmse": raw["rmse"],
            "corrected_bias": corrected["bias"], "corrected_mae": corrected["mae"],
            "corrected_rmse": corrected["rmse"],
            "raw_sensitivity_0p25": det_raw["sensitivity"],
            "raw_specificity_0p25": det_raw["specificity"],
            "corrected_sensitivity_0p25": det_corrected["sensitivity"],
            "corrected_specificity_0p25": det_corrected["specificity"],
        }
        for slug in sorted(DISPLAY):
            row[f"area_{slug}"] = pred[slug]
        sweep.append(row)

    fields = list(sweep[0])
    tmp = SWEEP_OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(fobj, fieldnames=fields)
        w.writeheader(); w.writerows(sweep)
    os.replace(tmp, SWEEP_OUT)

    chosen = predictions[PROVISIONAL]
    joint = {
        slug: sum(
            (f(c["measured_area_mm2"]) or 0.0)
            for c in comp_by[slug]
            if b(c.get("kept_by_depth")) and b(c.get("in_measurement"))
            and ((f(c.get("loss_base_p50")) is not None and f(c["loss_base_p50"]) <= JOINT_LOSS_P50)
                 or (f(c.get("hyper_p50")) is not None and f(c["hyper_p50"]) >= JOINT_HYPER_P50))
        )
        for slug in target
    }
    base_raw = metric(baseline, target, "plex_raw_mm2")
    rule_raw = metric(chosen, target, "plex_raw_mm2")
    base_corr = metric(baseline, target, "plex_corrected_mm2")
    rule_corr = metric(chosen, target, "plex_corrected_mm2")
    changed = [s for s in target if abs(chosen[s] - baseline[s]) >= 0.01]
    changed.sort(key=lambda s: abs(chosen[s] - baseline[s]), reverse=True)
    feasible = [r for r in sweep if r["anchor_feasible"]]

    lines = [
        "# Spatially anchored component-hypertransmission experiment", "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from config `{config_hash}`.", "",
        "## Decision", "",
        "Keep production unchanged. **Reject a component-median hypertransmission threshold as a direct "
        f"hard gate.** The round probe `hyper_p50 >= {PROVISIONAL:.2f}` looked safe on the small numeric "
        "anchor set, but the purple location overlays show that it removes PLEX-overlapping, low-OAC, "
        "hypertransmitting components in 003 OD and 011 OS that cannot safely be called non-GA. Keep the "
        "component hyper distribution as a feature for a future anatomy-aware combiner, not as a cutoff.", "",
        "This experiment never chooses a threshold by minimizing PLEX area error. Initial feasibility was defined "
        "only by preserving the 005 OD in-frame gold lesion, preserving three visually unequivocal GA "
        "sentinels, and clearing the clinically adjudicated no-GA eyes 006 OD and 010 OD.", "",
        "## Anchor result", "",
        "| Anchor | Current area | At 0.60 | Retained / interpretation |",
        "|---|---:|---:|---|",
        f"| 005 OD — in-frame gold | {baseline[POSITIVE_GOLD]:.3f} | {chosen[POSITIVE_GOLD]:.3f} | "
        f"{chosen[POSITIVE_GOLD]/baseline[POSITIVE_GOLD]:.0%} retained |",
    ]
    for slug in CONFIRMED_NO_GA:
        lines.append(f"| {slug.replace('NHAMD-003-', '')} — adjudicated no GA | {baseline[slug]:.3f} | "
                     f"{chosen[slug]:.3f} | should be zero |")
    for slug in OBVIOUS_GA_PRESERVE:
        lines.append(f"| {slug.replace('NHAMD-003-', '')} — visible GA sentinel | {baseline[slug]:.3f} | "
                     f"{chosen[slug]:.3f} | {chosen[slug]/baseline[slug]:.0%} retained |")
    lines += [
        "", f"The **numerically** feasible threshold band on the fixed 0.01 grid is "
        f"{fmt(feasible[0]['hyper_p50_threshold'], 2) if feasible else 'none'}–"
        f"{fmt(feasible[-1]['hyper_p50_threshold'], 2) if feasible else 'none'}. The spatial counterexamples "
        "invalidate that apparent feasibility; this is the concrete demonstration that area/anchor numbers alone "
        "were insufficient.", "",
        "## Components changed at the provisional 0.60 gate", "",
        "| Eye | Current | Candidate | Change | Spatial reading |",
        "|---|---:|---:|---:|---|",
    ]
    notes = {
        "NHAMD-003-006-V3_OD": "removes our small edge/peripapillary shared-confound call",
        "NHAMD-003-001-V1_OS": "removes the ours-only drusenoid/PED-looking component",
        "NHAMD-003-026-V3_OS": "removes spatially disjoint ours-only drusenoid/PED-looking components",
        "NHAMD-003-011-V3_OD": "also removes several uncertain foci — requires human B-scan labels",
        "NHAMD-003-011-V3_OS": "purple spans include PLEX-overlapping likely-real foci and a drusenoid false-call candidate",
        "NHAMD-003-003-V3_OD": "removes a 0.43 mm2 purple PLEX-overlapping low-OAC/transmitting component — unsafe",
        "NHAMD-003-003-V3_OS": "removes only a tiny weak-transmission satellite",
    }
    for slug in changed:
        lines.append(f"| {slug.replace('NHAMD-003-', '')} | {baseline[slug]:.3f} | {chosen[slug]:.3f} | "
                     f"{chosen[slug]-baseline[slug]:+.3f} | {notes.get(slug, 'component-level change; inspect panel')} |")
    lines += [
        "", "## A simple two-channel hand rule also fails", "",
        f"The next round hypothesis retained a component when `loss_base_p50 <= {JOINT_LOSS_P50:.2f}` **or** "
        f"`hyper_p50 >= {JOINT_HYPER_P50:.2f}`. It rescues the weak-hyper 003/011 components, but deletes "
        f"005 OS completely ({baseline['NHAMD-003-005-V3_OS']:.3f} → "
        f"{joint['NHAMD-003-005-V3_OS']:.3f} mm2). The 005 OS panel shows a compact PLEX-overlapping lesion "
        "with OAC loss and transmission, so this global union rule is also unsafe. Moving the constants just "
        "across 005 OS would be case-specific overfitting, not validation.", "",
        "## Area metrics are secondary", "",
        "| Reference accounting | Current MAE | Candidate MAE | Current bias | Candidate bias |",
        "|---|---:|---:|---:|---:|",
        f"| Raw PLEX | {base_raw['mae']:.3f} | {rule_raw['mae']:.3f} | {base_raw['bias']:+.3f} | "
        f"{rule_raw['bias']:+.3f} |",
        f"| Adjudicated PLEX | {base_corr['mae']:.3f} | {rule_corr['mae']:.3f} | {base_corr['bias']:+.3f} | "
        f"{rule_corr['bias']:+.3f} |",
        "", "A worse PLEX MAE would not invalidate the rule where PLEX labels the wrong anatomy, and a better "
        "MAE would not validate it. The decision must come from the component/B-scan labels in "
        "`outputs/ga_lesion_audit/`.", "",
        "## Required next test", "",
        "1. Human-label the disputed spans in 001 OS, 003 OD, 005 OS, 011 OD/OS, 014 OD and 026 OS as complete RPE loss, "
        "incomplete/transition, drusen/PED, or non-retinal/peripapillary.",
        "2. Use component hyper quantiles, OAC-loss quantiles, RPE-to-BM elevation, retinal thickness/BM slope, "
        "edge/peripapillary context and cross-channel co-localization as inputs to a small observable combiner. "
        "Do not turn any one of them into another hand gate.",
        "3. Evaluate patient-split with lesion-level precision/recall plus ours-only and label-only area separately; "
        "keep 005 OD as the fixed gold sentinel. Do not optimize on random components from the same eye.",
        "", "## Artifacts", "",
        "- `results/ga_component_hyper_sweep.csv` — full fixed threshold grid",
        "- `results/ga_component_hyper_findings.md` — this memo",
        "- `results/ga_spatial_decomp.csv` — non-cancelling spatial areas",
        "- `results/ga_lesion_components.csv` — component anatomy/features",
    ]
    tmp = FINDINGS_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines) + "\n")
    os.replace(tmp, FINDINGS_OUT)
    print(f"wrote {SWEEP_OUT}")
    print(f"wrote {FINDINGS_OUT}")


if __name__ == "__main__":
    main()
