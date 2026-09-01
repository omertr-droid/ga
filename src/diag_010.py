#!/usr/bin/env python
"""Reproduce the reader's EXACT OAC GA computation for one eye, to find why compare_plex.py disagreed.

The reader route (reader/api/routes_segmentation.py:939) does:
    bm = effective_surfaces(ov, layer_store)[1]
    area = oac_ga.detect(ov, bm, frac=0.50, trend_order=2)[2]
We replicate that and also report the classical-vs-DL base BM, so the divergence is visible.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bm as bmmod  # noqa: E402
import bm_dl  # noqa: E402
from paths import DATA_DIR  # noqa: E402
from reader.core import e2e_source, layers as core_layers, oac_ga  # noqa: E402
from reader.core.layer_store import JsonSidecarLayerStore  # noqa: E402

CORR = os.path.join(_REPO, "reader", "data_store", "corrections")
E2E = os.path.join(DATA_DIR, "NHAMD-003-010-V1-SPECTRALIS", "NHAMD01N.E2E")
EYE = "OD"


def area(ov, bm, order):
    return oac_ga.detect(ov, bm, trend_order=order)[2]


def main():
    print(f"OCT_BM_DL env = {os.environ.get('OCT_BM_DL')!r}  bm_dl.active()={bm_dl.active()}")
    raw = e2e_source.open_e2e(E2E)
    idx = e2e_source.default_volume_index(raw, EYE)
    ov = e2e_source.load_volume(raw, idx)
    store = JsonSidecarLayerStore(CORR)
    corrected = store.corrected_indices(ov.eid, ov.eye)
    print(f"eid={ov.eid} eye={ov.eye} bm_src={ov.bm_src} n={ov.n_bscans} "
          f"fov={tuple(round(f,3) for f in ov.fov_mm)} corrected_bscans={len(corrected)}")

    # the reader's exact BM + numbers
    ilm, bm_eff = core_layers.effective_surfaces(ov, store)
    print(f"\n[reader path] effective_surfaces BM  ->  order2={area(ov, bm_eff, 2):.4f}  "
          f"order1={area(ov, bm_eff, 1):.4f} mm^2")

    # what is ov.bm (the base the reader's effective_surfaces started from)?
    print(f"ov.bm vs bm_eff identical? {np.allclose(np.nan_to_num(ov.bm), np.nan_to_num(bm_eff))}")

    # classical self-seg explicitly (DL OFF), and DL explicitly
    bm_classical = bmmod.segment_volume(ov.vol) if hasattr(bmmod, "segment_volume") else None
    if bm_classical is not None:
        print(f"[classical bm.segment_volume] order2={area(ov, bm_classical, 2):.4f}  "
              f"order1={area(ov, bm_classical, 1):.4f} mm^2  "
              f"(== ov.bm? {np.allclose(bm_classical, np.nan_to_num(ov.bm), atol=0.5)})")
    try:
        bm_dl_surf = bm_dl.segment_volume(ov.vol)
        print(f"[DL bm_dl.segment_volume]    order2={area(ov, bm_dl_surf, 2):.4f}  "
              f"order1={area(ov, bm_dl_surf, 1):.4f} mm^2")
        # mean abs row difference classical vs DL
        if bm_classical is not None:
            print(f"  mean|classical-DL| BM rows = {np.nanmean(np.abs(bm_classical-bm_dl_surf)):.2f} px")
    except Exception as e:
        print(f"[DL] unavailable: {e!r}")


if __name__ == "__main__":
    main()
