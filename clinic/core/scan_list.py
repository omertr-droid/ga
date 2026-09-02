"""List the GA-measurable scans inside an opened E2E, and cheaply detect device Bruch's-membrane.

The upload chooser shows ONLY 6x6-measurable volumes (one per eye) — the only fields wide enough for
a 6x6 mm GA measurement. We never process GA here: this is pure metadata, so the chooser renders
instantly after the one-time E2E decode.

Device-BM presence drives the next step (the BM-segmentation prompt). It is detectable in microseconds
WITHOUT a full ``load_volume`` (which runs the ~5-12s classical self-seg on a cold open): the device
contours are already decoded on the volume object as ``v.contours``. We call the SAME extractor
``load_volume`` uses (``e2e_source._device_layers``), so ``has_device_bm`` equals the ``bm_src ==
"device"`` the later load will report — no guessing, no extra work.
"""
import numpy as np

from reader.core import e2e_source


def has_device_bm(raw, index) -> bool:
    """True if volume ``index`` ships a device Bruch's-membrane contour (the deep finite-coverage
    contour ``load_volume`` would tag ``bm_src="device"``). Reads already-decoded ``v.contours`` only."""
    try:
        v = raw.vols[index]
        n = len(v.volume)
        W = int(np.asarray(v.volume[0]).shape[1])
    except Exception:
        return False
    _ilm, bm = e2e_source._device_layers(v, (n, W))
    return bm is not None


def bm_prompt(has_device: bool, dl_available: bool) -> str:
    """Which BM prompt the frontend should show for a scan:
      * ``"choice"``    — device BM present: offer DL vs Device.
      * ``"dl_only"``   — no device BM but a DL model is present: "This will be DL BM segmented."
      * ``"auto_only"`` — no device BM and no DL model: classical auto self-segmentation.
    """
    if has_device:
        return "choice"
    return "dl_only" if dl_available else "auto_only"


def measurable_scans(raw, dl_available: bool) -> list:
    """The 6x6-measurable volumes in ``raw`` (``VolumeRef.is_6x6``), each described for the chooser:
    ``index, eye, kind, n_bscans, fov_mm, has_device_bm, bm_prompt``. Usually one per eye."""
    scans = []
    for r in raw.refs:
        if not r.is_6x6:
            continue
        dev = has_device_bm(raw, r.index)
        scans.append({
            "index": r.index,
            "eye": r.eye,
            "kind": r.kind,
            "n_bscans": r.n_bscans,
            "fov_mm": [round(float(x), 3) for x in r.fov_mm],
            "has_device_bm": dev,
            "bm_prompt": bm_prompt(dev, dl_available),
        })
    return scans
