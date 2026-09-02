#!/usr/bin/env python
"""Post-hoc, target-aware analysis of ``ga_experiment_matrix.py`` artifacts.

No OCT inference is rerun.  This script evaluates the saved stage waterfall and simulates only the
last component-depth veto from pre-depth component statistics.  The fixed grid is diagnostic, not a
new tuned default.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MATRIX = os.path.join(RESULTS, "ga_experiment_matrix.csv")
STAGES = os.path.join(RESULTS, "ga_experiment_stages.csv")
COMPONENTS = os.path.join(RESULTS, "ga_experiment_components.csv")
SWEEP_OUT = os.path.join(RESULTS, "ga_depth_gate_sweep.csv")
STAGE_OUT = os.path.join(RESULTS, "ga_experiment_stage_metrics.csv")
FINDINGS_OUT = os.path.join(RESULTS, "ga_experiment_findings.md")

STATS = ("loss_base_min", "loss_base_p01", "loss_base_p05")
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.00, 0.501, 0.01))
GUARDS = (
    "NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD", "NHAMD-003-014-V1_OD",
    "NHAMD-003-005-V3_OD", "NHAMD-003-008-V1_OS", "NHAMD-003-026-V3_OD",
)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def b(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")


def latest_complete_hash(matrix):
    counts = defaultdict(set)
    for r in matrix:
        if r.get("protocol") == "current97_native" and r.get("intensity") == "display":
            counts[r["config_hash"]].add(r["slug"])
    complete = [h for h, slugs in counts.items() if len(slugs) == 25]
    if not complete:
        raise RuntimeError("no config hash has a complete 25-eye parity arm")
    return complete[-1]


def metrics(pred, targets, ref_mode="corrected"):
    ref_key = "plex_raw_mm2" if ref_mode == "raw" else "plex_corrected_mm2"
    vals = [(pred[s], f(r[ref_key])) for s, r in targets.items() if s in pred and f(r[ref_key]) is not None]
    d = np.asarray([o - ref for o, ref in vals], float)
    classes = "ref_class_raw" if ref_mode == "raw" else "ref_class_corrected"
    tp = fn = tn = fp = 0
    for slug, area in pred.items():
        cls = targets[slug][classes]
        if cls == "positive":
            tp += int(area > 0); fn += int(area <= 0)
        elif cls == "negative":
            fp += int(area > 0); tn += int(area <= 0)
    return {
        "n": len(vals), "bias": float(d.mean()), "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def config_groups(matrix, config_hash):
    groups = defaultdict(list)
    for r in matrix:
        if r.get("config_hash") == config_hash:
            groups[r["config_id"]].append(r)
    return {cid: rows for cid, rows in groups.items()
            if len(rows) == 25 and rows[0].get("intensity") == "display"}


def depth_sweep(matrix, components, config_hash):
    groups = config_groups(matrix, config_hash)
    comp_by = defaultdict(list)
    for c in components:
        if c.get("config_hash") == config_hash:
            comp_by[(c["config_id"], c["slug"])].append(c)
    rows = []
    for cid, eyes in sorted(groups.items()):
        targets = {r["slug"]: r for r in eyes}
        for stat in STATS:
            for threshold in THRESHOLDS:
                pred = {}
                for slug in targets:
                    pred[slug] = sum(
                        f(c.get("measured_area_mm2")) or 0.0
                        for c in comp_by.get((cid, slug), [])
                        if (f(c.get(stat)) is not None and f(c.get(stat)) < threshold and
                            b(c.get("in_measurement")))
                    )
                raw, corrected = metrics(pred, targets, "raw"), metrics(pred, targets, "corrected")
                negatives = [s for s, r in targets.items() if r["ref_class_corrected"] == "negative"]
                out = {
                    "config_hash": config_hash, "config_id": cid,
                    "protocol": eyes[0]["protocol"], "intensity": eyes[0]["intensity"],
                    "depth_statistic": stat, "threshold": threshold,
                    "raw_bias": raw["bias"], "raw_mae": raw["mae"],
                    "corrected_bias": corrected["bias"], "corrected_mae": corrected["mae"],
                    "tp": corrected["tp"], "fn": corrected["fn"],
                    "tn": corrected["tn"], "fp": corrected["fp"],
                    "sensitivity": corrected["sensitivity"], "specificity": corrected["specificity"],
                    "max_confirmed_negative_mm2": max((pred[s] for s in negatives), default=0.0),
                }
                out.update({f"area_{s}": pred.get(s) for s in GUARDS})
                rows.append(out)

        # The default min-depth row must reconstruct the saved final mask area exactly, eye by eye.
        default = next(r for r in rows if r["config_id"] == cid and
                       r["depth_statistic"] == "loss_base_min" and r["threshold"] == 0.27)
        pred_default = {}
        for slug in targets:
            pred_default[slug] = sum(
                f(c.get("measured_area_mm2")) or 0.0 for c in comp_by.get((cid, slug), [])
                if f(c.get("loss_base_min")) is not None and f(c["loss_base_min"]) < 0.27 and
                b(c.get("in_measurement")))
        max_delta = max(abs(pred_default[s] - f(targets[s]["ours_area_mm2"])) for s in targets)
        if max_delta > 1e-9:
            raise AssertionError(f"component sweep does not reconstruct final for {cid}: {max_delta}")
        default["default_reconstruction_max_delta_mm2"] = max_delta
    write_csv(SWEEP_OUT, rows)
    return rows


def stage_metrics(matrix, stages, config_hash):
    groups = config_groups(matrix, config_hash)
    stage_by = defaultdict(dict)
    for s in stages:
        if s.get("config_hash") == config_hash:
            stage_by[(s["config_id"], s["slug"])][s["stage"]] = s
    out = []
    for cid, eyes in sorted(groups.items()):
        targets = {r["slug"]: r for r in eyes}
        names = sorted(set.intersection(*[set(stage_by[(cid, slug)]) for slug in targets]))
        for name in names:
            pred = {slug: f(stage_by[(cid, slug)][name]["area_mm2"]) for slug in targets}
            raw, corrected = metrics(pred, targets, "raw"), metrics(pred, targets, "corrected")
            out.append({
                "config_hash": config_hash, "config_id": cid, "protocol": eyes[0]["protocol"],
                "stage": name, "raw_bias": raw["bias"], "raw_mae": raw["mae"],
                "corrected_bias": corrected["bias"], "corrected_mae": corrected["mae"],
                "tp": corrected["tp"], "fn": corrected["fn"],
                "tn": corrected["tn"], "fp": corrected["fp"],
                "sensitivity": corrected["sensitivity"], "specificity": corrected["specificity"],
            })
    write_csv(STAGE_OUT, out)
    return out


def fmt(x, n=3):
    return "—" if x is None else f"{float(x):.{n}f}"


def findings(matrix, stages, components, sweep, stage_rows, config_hash):
    groups = config_groups(matrix, config_hash)
    current_id = next(cid for cid in groups if cid.startswith("current97_native__display__"))
    wide_id = next(cid for cid in groups if cid.startswith("wide30_scancenter6__display__"))
    matrix_by = {(r["config_id"], r["slug"]): r for r in matrix if r.get("config_hash") == config_hash}
    stage_by = {(r["config_id"], r["slug"], r["stage"]): r
                for r in stages if r.get("config_hash") == config_hash}

    def stage_area(cid, slug, stage):
        return f(stage_by[(cid, slug, stage)]["area_mm2"])

    # Can any single depth statistic rescue 014 while every confirmed negative stays below 0.05 mm²?
    feasibility = []
    for cid in (current_id, wide_id):
        for stat in STATS:
            candidates = [r for r in sweep if r["config_id"] == cid and r["depth_statistic"] == stat and
                          f(r["max_confirmed_negative_mm2"]) < 0.05]
            best = max(candidates,
                       key=lambda r: (f(r["area_NHAMD-003-014-V1_OD"]), f(r["threshold"])),
                       default=None)
            feasibility.append((cid, stat, best))

    # Dominant components: kept FP component in each adjudicated negative; largest rejected 014 component.
    comp = [r for r in components if r.get("config_hash") == config_hash and r["config_id"] == wide_id]
    dominant = {}
    for slug in ("NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD"):
        cands = [r for r in comp if r["slug"] == slug and b(r["kept_by_depth"]) and f(r["measured_area_mm2"]) > 0]
        dominant[slug] = max(cands, key=lambda r: f(r["measured_area_mm2"]))
    cands = [r for r in comp if r["slug"] == "NHAMD-003-014-V1_OD" and not b(r["kept_by_depth"])
             and f(r["measured_area_mm2"]) > 0]
    dominant["NHAMD-003-014-V1_OD"] = max(cands, key=lambda r: f(r["measured_area_mm2"]))

    cur_metrics = metrics({r["slug"]: f(r["ours_area_mm2"]) for r in groups[current_id]},
                          {r["slug"]: r for r in groups[current_id]}, "corrected")
    wide_metrics = metrics({r["slug"]: f(r["ours_area_mm2"]) for r in groups[wide_id]},
                           {r["slug"]: r for r in groups[wide_id]}, "corrected")
    lines = [
        "# GA experiment findings and next experiments", "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from config `{config_hash}`. This file is "
        "target-aware evaluation; OCT inference remains isolated in the matrix artifacts.", "",
        "## Decision", "",
        "Keep the current 97-line/display-radiometry detector as the production default. Reject both direct "
        "changes tested: inverse-linear transport failed the 005 OD gold sentinel (4.207 vs 1.057 mm²), and "
        "direct wide-scan inference lost specificity on both adjudicated PLEX-false-positive eyes.", "",
        f"Current adjusted MAE is {cur_metrics['mae']:.3f} mm²; wide adjusted MAE is "
        f"{wide_metrics['mae']:.3f} mm². Wide component specificity falls from "
        f"{cur_metrics['specificity']:.2f} to {wide_metrics['specificity']:.2f}.", "",
        "The raw-reference result is the primary one; adjudicated metrics are an optimistic sensitivity analysis.", "",
        "## Where the complete-loss veto acts", "",
        "| Eye / arm | cRORA-sized area | Final area | Area removed by depth veto |",
        "|---|---:|---:|---:|",
    ]
    for slug in ("NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD", "NHAMD-003-014-V1_OD"):
        for cid, label in ((current_id, "current"), (wide_id, "wide")):
            sized = stage_area(cid, slug, "sized"); final = stage_area(cid, slug, "final")
            lines.append(f"| {slug} · {label} | {sized:.3f} | {final:.3f} | {sized-final:.3f} |")
    lines += [
        "", "Wide context exposes a near-reference-sized 014 OD candidate (2.643 mm² vs PLEX 2.581), but "
        "the depth veto removes 2.563 mm². Yet the wide false-positive components in 006/010 are even deeper, "
        "so relaxing the same depth threshold cannot distinguish them.", "",
        "## Dominant wide components", "",
        "| Eye | Current decision | Component area | loss/base min | p01 | p05 | measurement fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for slug in ("NHAMD-003-006-V3_OD", "NHAMD-003-010-V1_OD", "NHAMD-003-014-V1_OD"):
        r = dominant[slug]
        lines.append(f"| {slug} | {'kept FP' if b(r['kept_by_depth']) else 'rejected target'} | "
                     f"{fmt(r['measured_area_mm2'])} | {fmt(r['loss_base_min'])} | "
                     f"{fmt(r['loss_base_p01'])} | {fmt(r['loss_base_p05'])} | "
                     f"{fmt(r['measurement_fraction'], 2)} |")
    lines += ["", "## Fixed depth-grid feasibility", "",
              "Maximum 014 OD area obtainable while every confirmed negative remains <0.05 mm²:", "",
              "| Arm | Component statistic | Best threshold | 014 OD rescued area |", "|---|---|---:|---:|"]
    for cid, stat, best in feasibility:
        label = "current" if cid == current_id else "wide"
        lines.append(f"| {label} | `{stat}` | {fmt(best['threshold'], 2) if best else '—'} | "
                     f"{fmt(best['area_NHAMD-003-014-V1_OD']) if best else '—'} |")
    lines += [
        "", "This is a descriptive fixed-grid result, not threshold selection. If rescued area remains zero, "
        "no global min/p01/p05 depth rule can solve the 014-vs-negative conflict.", "",
        "## Next experiments, in order", "",
        "1. **Component combiner, patient-split.** Keep current OAC/hyper/cRORA candidates, but replace the "
        "single-minimum depth veto with a small OCT-only component model using the full depth distribution, "
        "hypertransmission distribution, RPE→BM elevation, crop containment, field-edge distance, and "
        "current-vs-wide support consistency. Nested leave-one-patient-out only; never random-component CV.",
        "2. **014 OD B-scan adjudication.** Label the candidate spans in-frame and inspect why complete-loss "
        "depth is shallow. PLEX area alone cannot tell whether the whole 2.58 mm² candidate is true cRORA.",
        "3. **Use 30° as auxiliary context only.** It may propose/describe components, but the production "
        "measurement remains the current OCT volume and central endpoint. Never substitute its mask directly.",
        "4. **Add targeted human gold.** Prioritize 006 OD, 010 OD, 014 OD, 026 OD, plus 005 OD as the fixed "
        "anchor. Those cases span the exact specificity/recall conflict the automatic PLEX reference hides.",
        "", "## Generated artifacts", "",
        f"- `{os.path.relpath(SWEEP_OUT, ROOT)}` — fixed min/p01/p05 threshold grid",
        f"- `{os.path.relpath(STAGE_OUT, ROOT)}` — cohort metrics at every detector stage",
        f"- `{os.path.relpath(FINDINGS_OUT, ROOT)}` — this decision memo",
    ]
    tmp = FINDINGS_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, FINDINGS_OUT)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-hash")
    a = p.parse_args(argv)
    matrix, stages, components = read_csv(MATRIX), read_csv(STAGES), read_csv(COMPONENTS)
    config_hash = a.config_hash or latest_complete_hash(matrix)
    sweep = depth_sweep(matrix, components, config_hash)
    stage_rows = stage_metrics(matrix, stages, config_hash)
    findings(matrix, stages, components, sweep, stage_rows, config_hash)
    print(f"depth sweep: {SWEEP_OUT}")
    print(f"stage metrics: {STAGE_OUT}")
    print(f"findings: {FINDINGS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
