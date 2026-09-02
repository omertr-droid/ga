#!/usr/bin/env python
"""Sanity check for the B-scan indexing doubt: render PLAIN full B-scans exactly as the reader serves
them (qv.norm8(ov.vol[idx]), the content of render.bscan_png), and print the E2E's volume list + the
volume index the CLI opened -- so we can confirm the CLI shows the SAME scan/slice as the reader.

Run: oct_env\\Scripts\\python.exe src\\oac_fullbscan.py [SUBJECT VISIT EYE] [idx ...]
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qcviz as qv
from reader.core import e2e_source
from oac_area import OUT, resolve


def main():
    a = sys.argv[1:]
    sve = [t for t in a if not t.isdigit()]
    idxs = [int(t) for t in a if t.isdigit()] or [34, 35, 36, 37, 38]
    subject, visit, eye = (sve + ["NHAMD-003-005", "V3", "OD"][len(sve):])[:3]
    row, e2e = resolve(subject, visit, eye)
    eye = row["eye"].upper()
    raw = e2e_source.open_e2e(e2e)
    print(f"E2E = {e2e}")
    print("volume list (index: eye n_bscans HxW fov_mm is_6x6 kind):")
    for r in raw.refs:
        print(f"  [{r.index}] {r.eye}  n={r.n_bscans}  {r.H}x{r.W}  fov={r.fov_mm}  6x6={r.is_6x6}  {r.kind}")
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    print(f"\nCLI opened vol_idx={idx}  eid={ov.eid}  eye={ov.eye}  n={ov.n_bscans}  H={ov.H}  W={ov.W}")
    tiles = [qv.label_tile(qv.ensure_rgb(qv.norm8(ov.vol[i])), f"{subject} {eye} b{i:03d}  (vol_idx={idx})")
             for i in idxs]
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{row['subject']}_{eye}_fullbscans.png")
    qv.save_rgb(out, qv.montage(tiles, cols=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
