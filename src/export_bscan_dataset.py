#!/usr/bin/env python
"""Export the annotation-studio masks into a training-ready B-scan GA dataset.

Reads the reader's live annotations (reader/data_store/segmentations/<eid>_<eye>/<run>/ masks +
status.json) for every `qc_status==ok` eye in the master index, and writes image/label pairs for the
B-scans the grader explicitly labelled — `ga` (painted) and `ga_free` (explicit negative). `todo`
(never looked at) and `borderline` (ambiguous) are EXCLUDED, so the set is clean: real positives +
real negatives, no false negatives.

Key properties:
  * Loads the SAME 6x6 volume the reader annotated and renders the SAME norm8 B-scan (train/infer parity).
  * Exports the active run's annotation GROUP (its wedge + rpe class-runs; override the base with --run).
  * MULTI-CLASS label PNG, values 0=background, 1=hypertransmission wedge, 2=RPE-present (NOT 0/255).
    A B-scan is included if EITHER class is `ga`/`ga_free` (union); RPE wins where the two overlap.
  * Patient-level GroupKFold in splits.json (no multi-visit leakage).
  * 2.5D-ready: neighbours are <stem>_b<idx±k>; n_bscans is in the manifest.

Out: outputs/ga_bscan_dataset/{images,labels,manifest.csv,splits.json} (regenerable, gitignored)
     results/ga_bscan_dataset_summary.csv  (committed per-eye progress index)
Run: oct_env\\Scripts\\python.exe src\\export_bscan_dataset.py [all | SUBJECT EYE ...] [--run gold]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

from paths import DATA_DIR, OUT_DIR, REPO_ROOT, RESULTS_DIR

sys.path.insert(0, REPO_ROOT)                          # so `import reader` resolves from a src/ script
from reader.core import e2e_source, ids, render        # noqa: E402
from reader.core import footprint as fp                 # noqa: E402
from reader.core import seg_classes as sc               # noqa: E402
from reader.core.mask_store import PngMaskStore         # noqa: E402

OUT = os.path.join(OUT_DIR, "ga_bscan_dataset")
IMG, LBL = os.path.join(OUT, "images"), os.path.join(OUT, "labels")
SEG_STORE = os.path.join(REPO_ROOT, "reader", "data_store", "segmentations")
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
SUMMARY = os.path.join(RESULTS_DIR, "ga_bscan_dataset_summary.csv")

MANIFEST_COLS = ["subject", "eye", "visit", "patient", "bscan", "state", "n_ga_px", "n_wedge_px",
                 "n_rpe_px", "reviewed", "by", "n_bscans", "H", "W", "fov_x_mm", "fov_y_mm",
                 "advRPE_area_mm2", "wedge_run", "rpe_run", "image", "label"]
SUMMARY_COLS = ["subject", "eye", "patient", "run", "n_ga", "n_ga_free", "n_borderline", "n_todo",
                "pct_done", "footprint_area_mm2", "combined_area_mm2", "advRPE_area_mm2"]


def _read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _find_6x6(raw, eye):
    refs = [r for r in raw.refs if r.eye == eye and getattr(r, "is_6x6", False)]
    return refs[0] if refs else None


def _patient(subject):
    p = subject.split("-")
    return p[2] if len(p) > 2 else subject


def _visit(subject):
    for p in subject.split("-"):
        if len(p) >= 2 and p[0] in "Vv" and p[1:].isdigit():
            return p.upper()
    return ""


def _combined_area(eid, eye, ov, ms, data, class_runs):
    """cRORA area (mm²) = AND of the class-runs' column flags (each with its stored invert)."""
    flags = None
    for rid in class_runs:
        if rid not in data["runs"] or not ms.mask_indices(eid, eye, rid):
            continue
        f = fp.native_flags(ov, ms, rid, invert=bool(data["runs"][rid].get("invert"))) > 0
        flags = f if flags is None else (flags & f)
    if flags is None or not flags.any():
        return 0.0
    return float(fp.footprint_from_flags(flags.astype(np.float32), ov.fov_mm, 250.0)[1])


def export_eye(subject, eye, e2e_file, advrpe_area, run_arg, manifest, ov=None, ms=None):
    ms = ms or PngMaskStore(SEG_STORE)
    if ov is not None:                                 # reader passes the already-open 6x6 volume + store
        eid = ov.eid
    else:
        path = e2e_file if os.path.isabs(e2e_file) else os.path.join(DATA_DIR, e2e_file)
        if not os.path.exists(path):
            print(f"  {subject} {eye}: skip (E2E not found)"); return None
        eid = ids.e2e_id(path)
    data = ms.list_runs(eid, eye)
    run = run_arg or data.get("active")
    if not run or run not in data["runs"]:
        print(f"  {subject} {eye}: skip (no annotation run; active={data.get('active')})"); return None

    base = sc.base_class(run, data["runs"].get(run))[0]    # the annotation group; export its class-runs
    wedge_run, rpe_run = sc.run_id(base, "wedge"), sc.run_id(base, "rpe")
    has_wedge, has_rpe = wedge_run in data["runs"], rpe_run in data["runs"]
    if not has_wedge and not has_rpe:                      # fall back: treat the named run as the wedge class
        wedge_run, has_wedge = run, True

    if ov is None:
        raw = e2e_source.open_e2e(path)                # load the SAME 6x6 volume the reader annotated
        ref = _find_6x6(raw, eye)
        if ref is None:
            print(f"  {subject} {eye}: skip (no 6x6 volume)"); return None
        ov = e2e_source.load_volume(raw, ref.index)
    n = ov.n_bscans
    dw = ms.derived_status(eid, eye, wedge_run, n) if has_wedge else {}
    dr = ms.derived_status(eid, eye, rpe_run, n) if has_rpe else {}
    pat, vis = _patient(subject), _visit(subject)
    counts = {"ga": 0, "ga_free": 0, "borderline": 0, "todo": 0}

    for i in range(n):
        ws = (dw.get(i) or {}).get("state", "todo")
        rs = (dr.get(i) or {}).get("state", "todo")
        cs = ("ga" if "ga" in (ws, rs) else "ga_free" if "ga_free" in (ws, rs)
              else "borderline" if "borderline" in (ws, rs) else "todo")   # union over classes
        counts[cs] = counts.get(cs, 0) + 1
        if cs not in ("ga", "ga_free"):
            continue                                   # exclude todo (unseen) + borderline (ambiguous)
        stem = f"{subject}_{eye}_b{i:04d}"
        with open(os.path.join(IMG, stem + ".png"), "wb") as f:
            f.write(render.bscan_png(ov, i))           # same render the model sees at inference
        wm = ms.get_mask(eid, eye, wedge_run, i) if (has_wedge and ws == "ga") else None
        rm = ms.get_mask(eid, eye, rpe_run, i) if (has_rpe and rs == "ga") else None
        lab = np.zeros((ov.H, ov.W), np.uint8)         # MULTI-CLASS: 0=bg, 1=hypertransmission, 2=RPE-present
        if wm is not None:
            lab[wm] = 1
        if rm is not None:
            lab[rm] = 2                                # RPE wins on overlap (the finer structure)
        cv2.imwrite(os.path.join(LBL, stem + ".png"), lab)
        manifest.append({
            "subject": subject, "eye": eye, "visit": vis, "patient": pat, "bscan": i, "state": cs,
            "n_ga_px": int(wm.sum()) if wm is not None else 0,
            "n_wedge_px": int(wm.sum()) if wm is not None else 0,
            "n_rpe_px": int(rm.sum()) if rm is not None else 0,
            "reviewed": int(bool((dw.get(i) or {}).get("reviewed") or (dr.get(i) or {}).get("reviewed"))),
            "by": (dw.get(i) or {}).get("by") or (dr.get(i) or {}).get("by") or "",
            "n_bscans": n, "H": ov.H, "W": ov.W,
            "fov_x_mm": round(float(ov.fov_mm[0]), 4), "fov_y_mm": round(float(ov.fov_mm[1]), 4),
            "advRPE_area_mm2": advrpe_area, "wedge_run": wedge_run if has_wedge else "",
            "rpe_run": rpe_run if has_rpe else "",
            "image": f"images/{stem}.png", "label": f"labels/{stem}.png"})

    w_area = (fp.run_footprint(ov, ms, wedge_run, 250.0)[1]
              if has_wedge and ms.mask_indices(eid, eye, wedge_run) else 0.0)
    c_area = _combined_area(eid, eye, ov, ms, data, [wedge_run, rpe_run])
    print(f"  {subject} {eye}: base={base} wedge={wedge_run} rpe={rpe_run if has_rpe else '-'}  "
          f"ga={counts['ga']} ga_free={counts['ga_free']} borderline={counts['borderline']} todo={counts['todo']}  "
          f"wedge={w_area:.2f} combined={c_area:.2f} (advRPE={advrpe_area})")
    return {"subject": subject, "eye": eye, "patient": pat, "run": base, "n_ga": counts["ga"],
            "n_ga_free": counts["ga_free"], "n_borderline": counts["borderline"], "n_todo": counts["todo"],
            "pct_done": round(100 * (n - counts["todo"]) / max(1, n), 1),
            "footprint_area_mm2": round(w_area, 3), "combined_area_mm2": round(c_area, 3),
            "advRPE_area_mm2": advrpe_area}


def export_single(subject, eye, advrpe_area, run_arg=None, ov=None, ms=None):
    """Export ONE eye and MERGE it into the dataset, replacing any prior rows for this subject+eye in
    manifest.csv / the summary / splits.json (the CLI's whole-cohort overwrite would clobber the others).
    Returns the per-eye summary dict (n_ga, areas, …) or None if there is no annotation run to export.
    The reader's Export button calls this with the already-open volume (ov) + live mask store (ms), so it
    is fast and the image/label/area outputs are byte-identical to the CLI's."""
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(LBL, exist_ok=True)
    new_rows = []
    s = export_eye(subject, eye, None, advrpe_area, run_arg, new_rows, ov=ov, ms=ms)
    if not s:
        return None
    mpath = os.path.join(OUT, "manifest.csv")
    kept = [r for r in _read_rows(mpath) if not (r.get("subject") == subject and r.get("eye") == eye)]
    _write_rows(mpath, MANIFEST_COLS, kept + new_rows)
    summary = [r for r in _read_rows(SUMMARY) if not (r.get("subject") == subject and r.get("eye") == eye)]
    summary.append(s)
    _write_rows(SUMMARY, SUMMARY_COLS, summary)
    write_splits(summary)
    return s


def write_splits(summary):
    pats = sorted({r["patient"] for r in summary})
    k = max(1, min(5, len(pats)))
    fold_of = {p: i % k for i, p in enumerate(pats)}   # patient-grouped round-robin (no leakage)
    folds = {str(f): [] for f in range(k)}
    eye_fold = {}
    for r in summary:
        key = f"{r['subject']}_{r['eye']}"; f = fold_of[r["patient"]]
        folds[str(f)].append(key); eye_fold[key] = f
    with open(os.path.join(OUT, "splits.json"), "w") as fh:
        json.dump({"k": k, "by": "patient", "folds": folds, "eye_fold": eye_fold}, fh, indent=1)


def main():
    args = sys.argv[1:]
    run_arg = None
    if "--run" in args:
        i = args.index("--run"); run_arg = args[i + 1]; del args[i:i + 2]
    os.makedirs(IMG, exist_ok=True); os.makedirs(LBL, exist_ok=True)

    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("qc_status") == "ok"]
    if args and args[0].lower() != "all":
        pairs = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]   # e.g. ("004","OS") or full key
        rows = [r for r in rows if any(s in r["subject"] and e == r["eye"] for s, e in pairs)]

    manifest, summary = [], []
    for r in rows:
        try:
            s = export_eye(r["subject"], r["eye"], r["e2e_file"], r.get("advRPE_area_mm2", ""), run_arg, manifest)
            if s:
                summary.append(s)
        except Exception as ex:
            print(f"  {r['subject']} {r['eye']}: ERROR {type(ex).__name__}: {ex}")

    if manifest:
        _write_rows(os.path.join(OUT, "manifest.csv"), MANIFEST_COLS, manifest)
        write_splits(summary)

    _write_rows(SUMMARY, SUMMARY_COLS, summary)
    print(f"\nDONE {len(summary)} eyes, {len(manifest)} B-scans -> {OUT}/  "
          f"(images/ labels/ manifest.csv splits.json) + {SUMMARY}")


if __name__ == "__main__":
    main()
