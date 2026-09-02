#!/usr/bin/env python
"""Precompute the GA GOLD-GRADING worklist -- the prioritized list of eyes for hand-grading GA spans on
B-scans (reader Segment tab, per docs/LABELING_PROTOCOL.md), so the OCT detector can finally be measured
against GOLD instead of the over-calling PLEX/advRPE SILVER reference.

Why (workflow ga-error-experiments, 2026-07-11): a spatial audit (src/ga_error_audit.py) of the 25-eye
library found the OCT-vs-PLEX disagreement is ~one-directional (pooled ours-only 2.2 vs PLEX-only 20.4 mm2
-> we rarely over-call) and that an independent 3-lens adjudication judged 11/12 disagreement eyes a PLEX
(partial) FALSE POSITIVE -- i.e. most of our apparent 'misses' are the reference over-calling drusen /
incomplete loss, NOT real GA we are blind to. So the top lever is FIXING THE RULER: a small gold set on
exactly these eyes. Netted area also HIDES spatial error (026 OS: ours 0.21 vs PLEX 0.55 but Dice 0.00 --
different foci), so gold must be graded + scored SPATIALLY.

Merges:
  results/spectralis_ga_pairing.csv   (qc_status==ok -> the 25 gradeable eyes, E2E path, advRPE area)
  results/ga_spatial_decomp.csv       (overlap[TP]/ours-only[FP]/PLEX-only[FN]/dice -> spatial disagreement)
  results/ga_adjudication_auto.csv    (PROVISIONAL 3-lens auto verdict -- a starting hypothesis, NOT gold)

-> results/ga_grading_worklist.csv, ranked by decision value (how much grading this eye moves/clarifies the
   validation), with a per-eye CATEGORY, the DECISION the grader must make, and the audit-panel path.

Run from repo root:  oct_env\\Scripts\\python.exe src\\build_ga_grading_worklist.py
"""
import csv
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
import paths  # noqa: E402

PAIRING = os.path.join(paths.RESULTS_DIR, "spectralis_ga_pairing.csv")
SPATIAL = os.path.join(paths.RESULTS_DIR, "ga_spatial_decomp.csv")
AUTOADJ = os.path.join(paths.RESULTS_DIR, "ga_adjudication_auto.csv")
AUDIT = os.path.join(paths.OUT_DIR, "ga_audit")
OUT = os.path.join(paths.RESULTS_DIR, "ga_grading_worklist.csv")

COLS = ["priority", "subject", "visit", "eye", "category", "decision", "plex_mm2", "ours_mm2",
        "overlap", "ours_only", "plex_only", "dice", "disagreement", "auto_verdict", "auto_conf",
        "score", "audit_panel", "e2e_file", "n_bscans"]


def norm_subject(s):
    """ga_spatial_decomp keys carry a doubled -V<n>-V<n> suffix (a driver quirk); collapse to one."""
    return re.sub(r"(-V\d+)(-V\d+)$", r"\1", s or "")


def load_map(path, keyfn):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {keyfn(r): r for r in csv.DictReader(f) if not (r.get("subject", "") or "").startswith("#")}


def fnum(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return default


def categorize(plex, ours, overlap, ours_only, plex_only, dice, verdict):
    """Assign a working error-mode category (grading confirms/overturns it)."""
    disagree = ours_only + plex_only
    if plex < 0.25 and ours < 0.25:
        return "control", "Confirm NO GA (specificity). Mark 'Rest GA-free'; flag any true focal loss."
    if verdict == "plex_false_positive":
        return "PLEX_overcall_suspect", ("Reference likely WRONG. Decide: is the RPE truly present (drusen/"
                                         "incomplete) across the PLEX region -> correct advRPE to 0/partial.")
    if plex_only > 0.35 and ours < 0.25:
        return "PLEX_overcall_or_total_miss", ("We call ~nothing, PLEX calls GA. Decide per PLEX region: real "
                                               "complete loss (our FN) vs drusen/incomplete (PLEX over-call).")
    if overlap < 0.1 and ours_only > 0.1 and plex_only > 0.1:
        return "spatial_cancellation", ("Area ~agrees but Dice~0: we and PLEX call DIFFERENT foci. Grade each "
                                        "focus; do NOT trust the netted area here.")
    if ours_only > 0.3 and plex_only > 0.3:
        return "both_direction_error", ("Mixed: grade BOTH our-only and PLEX-only foci; some of each may be "
                                        "real GA the other missed.")
    if plex >= 0.25 and dice >= 0.75:
        return "good_agreement_gold", "Strong overlap; grade to pin the Dice ceiling + margin truth."
    return "review", "Grade the RPE band across all flagged foci."


def main():
    pairing = [r for r in csv.DictReader(open(PAIRING, newline="")) if (r.get("qc_status") or "").strip() == "ok"]
    spatial = load_map(SPATIAL, lambda r: (norm_subject(r["subject"]), r["eye"].upper()))
    autoadj = load_map(AUTOADJ, lambda r: (r["subject"], r["eye"].upper()))

    rows = []
    for p in pairing:
        # pairing 'subject' already carries the visit (e.g. NHAMD-003-014-V1); only append if it doesn't.
        subj = p["subject"] if re.search(r"-V\d+$", p["subject"]) else f"{p['subject']}-{p['visit']}"
        eye = p["eye"].upper()
        sp = spatial.get((subj, eye), {})
        aj = autoadj.get((subj, eye), {})
        plex = fnum(p, "advRPE_area_mm2")
        ours = fnum(sp, "ours_mm2")
        # tolerate both spatial-CSV schemas: new (overlap/ours_only/plex_only) and old (TP/FP/FN).
        overlap = fnum(sp, "overlap", fnum(sp, "TP"))
        ours_only = fnum(sp, "ours_only", fnum(sp, "FP"))
        plex_only = fnum(sp, "plex_only", fnum(sp, "FN"))
        dice = fnum(sp, "dice")
        disagree = ours_only + plex_only
        verdict = aj.get("auto_verdict", "")
        conf = aj.get("confidence", "")
        cat, decision = categorize(plex, ours, overlap, ours_only, plex_only, dice, verdict)

        # decision-value score: disagreement magnitude, up-weighted when (a) not yet hand-adjudicated and
        # the auto verdict is a PLEX over-call worth confirming, (b) spatial cancellation, (c) a boundary
        # small lesion; controls get a modest floor (needed for specificity but cheap to grade).
        score = disagree
        if cat == "PLEX_overcall_suspect" or cat == "PLEX_overcall_or_total_miss":
            score *= 1.8                         # confirming a reference error is the biggest metric mover
        if cat == "spatial_cancellation":
            score *= 1.6
        if cat == "both_direction_error":
            score *= 1.3
        if 0.25 <= plex <= 1.0:                  # near the iRORA->cRORA boundary = highest clinical ambiguity
            score += 0.4
        if cat == "control":
            score = max(score, 0.15)
        panel = os.path.join(AUDIT, f"{subj}_{eye}_audit.png")
        rows.append({
            "subject": subj, "visit": p["visit"], "eye": eye, "category": cat, "decision": decision,
            "plex_mm2": round(plex, 3), "ours_mm2": round(ours, 3), "overlap": round(overlap, 3),
            "ours_only": round(ours_only, 3), "plex_only": round(plex_only, 3), "dice": round(dice, 3),
            "disagreement": round(disagree, 3), "auto_verdict": verdict, "auto_conf": conf,
            "score": round(score, 3), "audit_panel": panel if os.path.exists(panel) else "",
            "e2e_file": p["e2e_file"], "n_bscans": p.get("n_bscans", ""),
        })

    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["priority"] = i
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})

    print(f"wrote {OUT}  ({len(rows)} eyes)\n")
    from collections import Counter
    cc = Counter(r["category"] for r in rows)
    print("categories:", dict(cc))
    print(f"\n{'#':>2} {'eye':13} {'category':30} {'plex':>5} {'ours':>5} {'dice':>5} {'score':>5}")
    for r in rows:
        print(f"{r['priority']:>2} {r['subject'][6:]+' '+r['eye']:13} {r['category']:30} "
              f"{r['plex_mm2']:5.2f} {r['ours_mm2']:5.2f} {r['dice']:5.2f} {r['score']:5.2f}")


if __name__ == "__main__":
    main()
