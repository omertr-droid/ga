#!/usr/bin/env python
"""diag_eye with the classical BM self-seg monkeypatched to DL BM (avoids an OOM in the graph search
when the eye has no device BM). Renders the same two panels as src/diag_eye.py."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OCT_BM_DL", "1")

import numpy as np  # noqa: E402
import bm_dl  # noqa: E402
import bm as bmseg  # noqa: E402


# Patch the expensive/ OOM-prone classical self-seg used inside load_volume with DL BM.
def _dl_surfaces_volume(vol, invalid=None):
    bm = bm_dl.segment_volume(np.asarray(vol, float))
    ilm = np.full_like(bm, np.nan)            # ILM not needed for OAC; leave NaN -> filled flat
    return ilm, bm


bmseg.segment_surfaces_volume = _dl_surfaces_volume

# e2e_source binds bm as `bmseg` at module load -> patch THAT reference too.
from reader.core import e2e_source  # noqa: E402
e2e_source.bmseg.segment_surfaces_volume = _dl_surfaces_volume

# now run the normal diag_eye main with the patch in place
import diag_eye  # noqa: E402
diag_eye.main()
