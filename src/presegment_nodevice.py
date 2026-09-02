#!/usr/bin/env python
"""Offline pre-segmentation of every NO-DEVICE cohort eye for the reader's BM-validation Library.

The reader has no device BM on these eyes, so opening one used to (a) run the slow classical self-seg
(~5-12s) and (b) run the DL model (~25s on CPU) live -> a long "segmenting layers" wait. This script does
both ONCE, offline, and persists the results to disk so the reader opens them instantly + already
segmented:
  - `load_volume` warms the per-volume **surface cache** (reader/data_store/surfcache/) -> no self-seg on
    later cold opens;
  - `presegment_eye` writes the **DL BM as cached corrections** (reader/data_store/corrections/),
    tagged bm_src="model" -> the eye shows up pre-segmented (purple), no Segment click.
Idempotent: an eye whose BM corrections already cover it is skipped (no DL pass). Run from the repo root:
    oct_env\\Scripts\\python.exe src\\presegment_nodevice.py [--all] [--only <subjsubstr>]
`--all` also warms device eyes' surface cache (their opens are already fast, so off by default).
"""
import argparse
import csv
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import paths  # noqa: E402
from reader.core import e2e_source, ids  # noqa: E402
from reader.core.layer_store import JsonSidecarLayerStore  # noqa: E402
from reader.api.routes_corrections import presegment_eye  # noqa: E402
from reader.api.deps import _DATA_STORE  # the corrections dir the server reads  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also warm device eyes' surface cache")
    ap.add_argument("--only", default=None, help="only subjects whose key contains this substring")
    args = ap.parse_args()

    ls = JsonSidecarLayerStore(_DATA_STORE)
    wl = os.path.join(paths.RESULTS_DIR, "bm_worklist.csv")
    rows = list(csv.DictReader(open(wl, newline="")))
    todo = []
    for r in rows:
        has_dev = str(r.get("has_device_bm")).strip().lower() in ("true", "1", "yes")
        if has_dev and not args.all:
            continue
        if args.only and args.only not in f"{r.get('subject')} {r.get('eye')}":
            continue
        todo.append(r)
    print(f"{len(todo)} eye(s) to precompute (no-device{' + device' if args.all else ''})\n")

    for r in todo:
        subj, eye = r.get("subject", ""), (r.get("eye") or "").strip().upper()
        path = os.path.join(paths.DATA_DIR, *r["e2e_file"].split("/"))
        tag = f"{subj} {eye}"
        if not os.path.exists(path):
            print(f"  SKIP {tag}: E2E missing ({path})")
            continue
        t0 = time.perf_counter()
        try:
            raw = e2e_source.open_e2e(path)
            idx = e2e_source.default_volume_index(raw, eye)
            ov = e2e_source.load_volume(raw, idx)                  # warms the surface cache
            res = presegment_eye(ov, None, ls, volume_id=None)     # writes DL BM corrections (idempotent)
        except Exception as e:                                     # noqa: BLE001
            print(f"  FAIL {tag}: {type(e).__name__}: {e}")
            continue
        dt = time.perf_counter() - t0
        note = (f"available={res.get('available')}" if not res.get("available")
                else f"n={res.get('n')} (already segmented)" if res.get("n") == 0
                else f"segmented n={res.get('n')}")
        print(f"  OK   {tag:22s} bm_src={ov.bm_src:6s} {note:28s} {dt:6.1f}s")

    print("\nDone. Surface cache -> reader/data_store/surfcache/ ; DL corrections -> reader/data_store/corrections/")


if __name__ == "__main__":
    main()
