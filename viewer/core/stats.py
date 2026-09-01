"""Cohort agreement statistics: our OCT-only GA area vs the PLEX (advRPE) reference.

The numbers come from the baked library index (`bundle.read_index()` / `bundle.library()`) -- i.e. from
exactly the eyes and areas the doctor viewer shows on its Library cards -- so a figure quoted in the PDF
report can never disagree with what a reader sees on screen. Nothing here reads an E2E, a model, or
results/*.csv.

Consumed by `src/ga_report.py`, which renders the shareable PDF. Kept dependency-light (numpy only, no
scipy) so it stays importable from inside the offline viewer packages, which vendor numpy + opencv +
fastapi + uvicorn and nothing else.

What is deliberately NOT reported: Pearson r / R^2. Across this cohort they are ~0.98/0.97, but that is
an artifact of the area range -- five eyes spanning 5-15 mm^2 define the line, and r falls to ~0.72 once
they are dropped. Reporting it would overstate the result. Agreement (bias, MAE, limits of agreement)
and the detection 2x2 are the honest summary, and the zoomed scatter shows where the eyes actually live.
"""
import csv
import math
import os

import numpy as np

CONTROL_THR = 0.05   # PLEX < this mm^2 = a no-GA control eye        (same constant as src/summarize_plex.py)
CALL_THR = 0.25      # below this mm^2 we call "no significant GA"   (the cRORA ~250 um floor)

# Verdicts that say the REFERENCE is WHOLLY wrong, so the eye cannot fairly be scored against it at all.
REF_ERROR = ("plex_false_positive", "plex_false_negative")

# Verdicts where the reference is only PARTLY wrong: it over- (or under-) calls, but real GA is present
# that we also got wrong. The true area is unknown, so neither number can be scored -- and crucially the
# eye is NOT excluded. Dropping it would credit us for a lesion we did in fact miss; the Delta against
# the reference merely overstates by an unknown amount. It stays in every statistic, flagged.
REF_PARTIAL = ("plex_partial_false_positive", "plex_partial_false_negative")


def load_adjudication(path):
    """Read results/plex_adjudication.csv -> {(subject, eye): {"verdict", "note"}}. Missing file = {}.

    Hand-entered clinical judgements about the reference. Comment lines start with '#'.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    out = {}
    for r in csv.DictReader(lines):
        subj, eye = (r.get("subject") or "").strip(), (r.get("eye") or "").strip()
        verdict = (r.get("verdict") or "").strip()
        if subj and eye and verdict:
            out[(subj, eye)] = {"verdict": verdict, "note": (r.get("note") or "").strip()}
    return out


def agrees(ours, plex):
    """The Library card's own agreement rule -- within 0.5 mm^2 OR within 25% of the reference.

    Kept identical to the `agree` test in viewer/web/js/library_view.js so a card that reads
    "agrees with PLEX" is counted as agreeing here.
    """
    return abs(ours - plex) <= max(0.5, 0.25 * plex)


def _classify(ours, plex):
    if plex < CONTROL_THR:
        return "control"
    if agrees(ours, plex):
        return "agree"
    return "under" if ours < plex else "over"


def _loa(d):
    """Bland-Altman bias and 95% limits of agreement."""
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def _finite(v):
    """JSON has no NaN/Infinity -- hand the frontend a null instead."""
    return None if v is None or not math.isfinite(v) else float(v)


def cohort_stats(eyes, adjudication=None):
    """Agreement summary + the per-eye table, from the library index rows.

    `eyes` is the list `bundle.library()` returns. Rows without a PLEX reference are skipped (they
    carry nothing to agree with); rows fall back to the base BM area when no DL twin was baked, which
    mirrors how the cards choose `Our GA (DL)` vs `Our GA`.

    `adjudication` is the mapping from `load_adjudication()`. Eyes whose REFERENCE was judged wrong are
    still measured and still plotted; they feed a second, separately-labelled metrics block in which a
    false-positive reference is CORRECTED to 0 rather than deleted. Nothing is silently dropped.
    """
    adjudication = adjudication or {}
    rows = []
    for e in eyes:
        plex = e.get("plex_area_mm2")
        ours = e.get("oac_area_dl_mm2")
        if ours is None:
            ours = e.get("oac_area_mm2")
        if plex is None or ours is None:
            continue
        ours, plex = float(ours), float(plex)
        adj = adjudication.get((e.get("subject"), e.get("eye")), {})
        rows.append({
            "slug": e.get("slug"), "subject": e.get("subject"), "eye": e.get("eye"),
            "patient_id": e.get("patient_id"), "visit": e.get("visit"),
            "ours_mm2": round(ours, 3), "plex_mm2": round(plex, 3),
            "delta_mm2": round(ours - plex, 3),
            "bm": "dl" if e.get("oac_area_dl_mm2") is not None else "base",
            "status": _classify(ours, plex),
            # A GA-positive eye we call as "no significant GA" -- a miss, and the cohort's real weakness.
            "missed": bool(plex >= CALL_THR and ours < CALL_THR),
            "adjudication": adj.get("verdict"), "adjudication_note": adj.get("note"),
        })
    rows.sort(key=lambda r: r["plex_mm2"])

    if not rows:
        return {"n_eyes": 0, "n_patients": 0, "metrics": {}, "eyes": []}

    metrics = _agreement(rows)
    metrics["control_thr"] = CONTROL_THR
    metrics["call_thr"] = CALL_THR

    # Second, clearly-separate scoring on the eyes whose reference was not judged wrong. Never replaces
    # the primary numbers -- the reader must be able to see both, and see how many eyes were removed.
    ref_err = [r for r in rows if r["adjudication"] in REF_ERROR]
    partial = [r for r in rows if r["adjudication"] in REF_PARTIAL]
    metrics["n_ref_error"] = len(ref_err)
    metrics["ref_error_slugs"] = [r["slug"] for r in ref_err]
    metrics["n_ref_partial"] = len(partial)
    metrics["ref_partial_slugs"] = [r["slug"] for r in partial]

    if ref_err:
        # A false-positive reference is CORRECTED to 0 (the verdict is "no atrophy on the OCT"), not
        # deleted. Deleting also removes the eye from the negatives, throwing away a true negative we have
        # earned: correcting gives the same sensitivity but a specificity over 9 negatives instead of 7,
        # and keeps n at 25. A false-NEGATIVE reference has no known true area, so that eye must come out.
        kept = []
        for r in rows:
            if r["adjudication"] == "plex_false_negative":
                continue
            if r["adjudication"] == "plex_false_positive":
                r = dict(r, plex_mm2=0.0, delta_mm2=round(r["ours_mm2"], 3),
                         status=_classify(r["ours_mm2"], 0.0), missed=False)
            kept.append(r)
        metrics["adjudicated"] = _agreement(kept) if kept else None

    return {"n_eyes": len(rows), "n_patients": metrics["n_patients"], "metrics": metrics, "eyes": rows}


def _agreement(rows):
    """Every agreement + detection metric for one set of eyes. Called twice: all eyes, and again with
    the reference-error eyes removed."""
    ours = np.array([r["ours_mm2"] for r in rows], float)
    plex = np.array([r["plex_mm2"] for r in rows], float)
    d = ours - plex
    n = len(rows)

    bias, bias_sd, loa_lo, loa_hi = _loa(d)

    ctrl = plex < CONTROL_THR
    pos = plex >= CALL_THR                       # GA-positive by the reference
    call = ours >= CALL_THR                      # we call significant GA
    tp = int((pos & call).sum())
    fn = int((pos & ~call).sum())
    tn = int((~pos & ~call).sum())
    fp = int((~pos & call).sum())

    # Where the detector fires, how close is it? (Excludes the misses, which the tiles report separately.)
    det = pos & call
    det_d = d[det]

    n_agree = int(sum(1 for r in rows if r["status"] in ("agree", "control")))

    return {
        "n_eyes": n,
        "n_patients": len({r["patient_id"] for r in rows}),
        "plex_min": _finite(plex.min()), "plex_max": _finite(plex.max()),
        "plex_median": _finite(float(np.median(plex))),

        "bias": _finite(bias), "bias_sd": _finite(bias_sd),
        "loa_lo": _finite(loa_lo), "loa_hi": _finite(loa_hi),
        "mae": _finite(float(np.abs(d).mean())),
        "median_ae": _finite(float(np.median(np.abs(d)))),
        "rmse": _finite(float(np.sqrt((d ** 2).mean()))),
        "within_0p5": int((np.abs(d) <= 0.5).sum()),
        "within_1p0": int((np.abs(d) <= 1.0).sum()),

        "n_agree": n_agree,

        "n_controls": int(ctrl.sum()),
        "control_max": _finite(float(ours[ctrl].max())) if ctrl.any() else None,
        "controls_clean": bool((ours[ctrl] < CONTROL_THR).all()) if ctrl.any() else True,

        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": _finite(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": _finite(tn / (tn + fp)) if (tn + fp) else None,

        "det_n": int(det.sum()),
        "det_bias": _finite(float(det_d.mean())) if det.any() else None,
        "det_mae": _finite(float(np.abs(det_d).mean())) if det.any() else None,
    }
