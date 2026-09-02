"""Refresh the Library's cached 'missing-BM B-scan' count for every cohort 6x6 eye, using the REAL
correction store — so the Library shows the ACCURATE count (device gaps minus the saturated band minus
saved corrections) at a glance, without opening each eye. This is heavier than build_bm_worklist (it
loads pixels + folds in corrections), so run it on demand: after a batch of edits, or to reconcile the
counts. Writes reader/data_store/corrections/<eid>_<eye>/missing.json (read by core.library).

Run from repo root:  oct_env\\Scripts\\python.exe src\\refresh_bm_missing.py
"""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import paths                                              # noqa: E402
from reader.core import e2e_source, layers, ids           # noqa: E402
from reader.core.layer_store import JsonSidecarLayerStore  # noqa: E402

WORKLIST = os.path.join(paths.RESULTS_DIR, "bm_worklist.csv")
STORE = JsonSidecarLayerStore(os.path.join(REPO, "reader", "data_store", "corrections"))


def main():
    rows = list(csv.DictReader(open(WORKLIST, newline="")))
    opened = {}
    for r in rows:
        eye = r["eye"].strip().upper()
        abspath = os.path.join(paths.DATA_DIR, *r["e2e_file"].strip().split("/"))
        eid = ids.e2e_id(abspath)
        try:
            raw = opened.get(abspath) or e2e_source.open_e2e(abspath)
            opened[abspath] = raw
            ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
        except Exception as e:                             # noqa: BLE001
            print(f"  !! {r['subject']} {eye}: {e}")
            continue
        n = len(layers.bm_missing_by_bscan(ov, STORE))     # B-scans with >=1 missing BM column (real)
        STORE.set_missing_count(eid, eye, n)
        print(f"  {r['subject'].replace('NHAMD-003-',''):>8} {eye}  missing B-scans = {n:>3} / {ov.n_bscans}")
    print("done — Library now shows accurate per-eye missing counts (reload the Library tab).")


if __name__ == "__main__":
    main()
