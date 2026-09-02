#!/usr/bin/env python
"""Build an OCT-first blinded worklist for the anatomy-aware GA component combiner.

Input is ``results/ga_lesion_components.csv`` from ``ga_lesion_audit.py --set --no-render``.  Only
``sized_candidate`` rows are learning units: every component has already met the 250 um cRORA size
criterion, but may have been accepted or rejected by the current complete-loss veto.

Two files are deliberately produced:

* ``ga_component_label_worklist.csv`` — primary OCT grading, with PLEX/current decisions hidden.
* ``ga_component_label_context.csv`` — algorithm/reference reconciliation, opened only after the
  primary morphology label has been locked.
"""
from __future__ import annotations

import csv
import math
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SOURCE = os.path.join(RESULTS, "ga_lesion_components.csv")
BLIND_OUT = os.path.join(RESULTS, "ga_component_label_worklist.csv")
CONTEXT_OUT = os.path.join(RESULTS, "ga_component_label_context.csv")


P0 = {
    "NHAMD-003-005-V3_OD": "fixed in-frame gold anchor",
    "NHAMD-003-006-V3_OD": "adjudicated no-GA / shared peripapillary confound",
    "NHAMD-003-010-V1_OD": "adjudicated no-GA / PLEX registration-review case",
    "NHAMD-003-014-V1_OD": "large depth-veto / likely PLEX-overcall conflict",
    "NHAMD-003-026-V3_OS": "zero-overlap spatial cancellation case",
    "NHAMD-003-003-V3_OD": "weak-hyper hard-gate counterexample",
    "NHAMD-003-011-V3_OS": "mixed weak-hyper true/false component counterexample",
    "NHAMD-003-005-V3_OS": "two-channel scalar-rule counterexample",
}
P1 = {
    "NHAMD-003-001-V1_OD": "multifocal drusen/calcific-vs-GA disagreement",
    "NHAMD-003-001-V1_OS": "known partial PLEX overcall plus ours-only component",
    "NHAMD-003-011-V3_OD": "large two-sided multifocal disagreement",
    "NHAMD-003-015-V3_OD": "likely true-GA extent disagreement",
    "NHAMD-003-026-V3_OD": "same-lesion extent disagreement",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def num(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def slug(row):
    return f"{row['subject']}_{row['eye']}"


def priority(row):
    s = slug(row)
    if s in P0:
        return 0, P0[s]
    if s in P1:
        return 1, P1[s]
    return 2, "large-GA preservation/boundary control"


def disposition(row):
    final = num(row.get("final_fraction"))
    rejected = num(row.get("depth_rejected_fraction"))
    if final >= 0.80:
        return "current_accepted"
    if rejected >= 0.80:
        return "current_depth_rejected"
    if final > 0 or rejected > 0:
        return "current_mixed_fragment"
    return "current_not_final"


def plex_relation(row):
    overlap = num(row.get("plex_fraction"))
    if overlap >= 0.80:
        return "mostly_inside_registered_PLEX"
    if overlap >= 0.10:
        return "partial_registered_PLEX_overlap"
    return "outside_registered_PLEX"


def main():
    rows = [r for r in read_csv(SOURCE) if r.get("source") == "sized_candidate"]
    rows.sort(key=lambda r: (priority(r)[0], r["subject"], r["eye"], -num(r.get("area_mm2"))))

    blind, context = [], []
    for order, r in enumerate(rows, start=1):
        p, reason = priority(r)
        cid = f"{r['subject']}_{r['eye']}_S{int(r['component']):02d}"
        blind.append({
            "review_order": order,
            "candidate_id": cid,
            "priority": f"P{p}",
            "subject": r["subject"],
            "eye": r["eye"],
            "candidate_component": r["component"],
            "bscan_min": r["native_bscan_min"],
            "bscan_max": r["native_bscan_max"],
            "x_min": r["native_x_min"],
            "x_max": r["native_x_max"],
            "representative_bscan": r["representative_bscan"],
            "representative_x0": r["representative_x0"],
            "representative_x1": r["representative_x1"],
            "review_status": "todo",
            "human_ga_label": "",
            "human_phenotype": "",
            "human_confidence": "",
            "mixed_requires_span_refinement": "",
            "reviewer": "",
            "reviewed_at": "",
            "grader_notes": "",
        })
        context.append({
            "candidate_id": cid,
            "priority_reason": reason,
            "area_mm2": r["area_mm2"],
            "current_disposition": disposition(r),
            "registered_PLEX_relation": plex_relation(r),
            "adjudication": r.get("adjudication", ""),
            "registration_flag": r.get("registration_flag", ""),
            "candidate_fraction": r.get("candidate_fraction", ""),
            "hyper_kept_fraction": r.get("hyper_kept_fraction", ""),
            "depth_rejected_fraction": r.get("depth_rejected_fraction", ""),
            "loss_base_min": r.get("loss_base_min", ""),
            "loss_base_p05": r.get("loss_base_p05", ""),
            "loss_base_median": r.get("loss_base_median", ""),
            "hyper_median": r.get("hyper_median", ""),
            "core_edge_distance_p10_mm": r.get("core_edge_distance_p10_mm", ""),
            "rpe_absent_fraction": r.get("rpe_absent_fraction", ""),
            "rpe_faded_fraction": r.get("rpe_faded_fraction", ""),
            "rpe_intact_fraction": r.get("rpe_intact_fraction", ""),
            "rpe_to_bm_median_um": r.get("rpe_to_bm_median_um", ""),
            "retina_thickness_median_um": r.get("retina_thickness_median_um", ""),
            "bm_slope_p95_px_per_ascan": r.get("bm_slope_p95_px_per_ascan", ""),
            "reconciliation_panel": f"outputs/ga_lesion_audit/{r['subject']}_{r['eye']}_lesion_audit.png",
            "reconciliation_result": "",
            "reconciliation_notes": "",
        })

    blind_fields = list(blind[0])
    context_fields = list(context[0])
    write_csv(BLIND_OUT, blind_fields, blind)
    write_csv(CONTEXT_OUT, context_fields, context)
    print(f"wrote {BLIND_OUT} ({len(blind)} OCT candidates)")
    print(f"wrote {CONTEXT_OUT}")


if __name__ == "__main__":
    main()
