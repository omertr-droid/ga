"""Process one chosen scan into a serveable view — the clinic's single GA-computation choke point.

``process()`` reproduces the proven viewer upload path (``viewer/api/deps.py:ViewerStore.upload``) in
the exact same call order — ``load_volume`` -> ``effective_surfaces(ov, None)`` -> optional
``bm_dl.segment_volume`` swap -> ``viewmodel.compute(raw, ov, ilm, bm)`` with the same defaults
(``baseline="radial2"``, ``frac=0.50``) — so a clinic scan reports the SAME ``oac_area_mm2`` as the
viewer/CLI. The only difference is that the clinic processes the volume the user explicitly chose
(``index`` + ``bm_choice``) rather than auto-picking.

The result is wrapped in ``viewer.core.bundle.LiveSource`` — the same in-memory ViewSource the viewer
serves uploads from — so all the panel endpoints reuse one code path. No PLEX: the meta's plex fields
are empty, so ``meta_public()`` reports ``has_plex=false`` with no special casing.
"""
from reader.core import e2e_source, layers as core_layers
from viewer.core import bundle, viewmodel

from . import identity


def open_e2e(path):
    """Decode an E2E once (volumes + metadata + refs). The expensive step; cache the result upstream."""
    raw = e2e_source.open_e2e(path)
    # The clinic chooser can only process native 6x6-measurable volumes.  A Spectralis E2E also carries
    # large 30-degree and single-line scans; retaining those decoded pixels after refs are built adds
    # hundreds of MB but can never affect a clinic result.  Keep their metadata/contours, drop pixels.
    keep = {r.index for r in raw.refs if r.is_6x6}
    for i, v in enumerate(raw.vols):
        if i not in keep:
            try:
                v.volume = None
            except Exception:
                pass
    return raw


def list_scans(raw):
    """6x6-measurable scans for the chooser (+ DL availability + patient identity). No GA processing."""
    import bm_dl                                              # reader.core put src/ on sys.path
    from . import scan_list
    dl_available = False
    try:
        dl_available = bool(bm_dl.discoverable())
    except Exception:
        dl_available = False
    pd = identity.patient_data(raw)
    return {
        "path": raw.path,
        "eid": raw.eid,
        "patient": {
            "patient_id": pd.get("patient_id", "") or "",
            "patient_name": identity.patient_name(raw),
        },
        "dl_available": dl_available,
        "dl_default": _dl_default(),
        "scans": scan_list.measurable_scans(raw, dl_available),
    }


def _dl_default() -> bool:
    import bm_dl
    try:
        return bool(bm_dl.enabled())
    except Exception:
        return False


def _resolve_bm(ov, ilm, bm, bm_choice):
    """Apply the user's BM choice. Returns ``(bm, bm_source, warning)``.
      * ``"dl"``     — run the DL Bruch's-membrane model; fall back to ``ov.bm`` (+ warning) on any failure.
      * ``"device"`` / ``"auto"`` — keep ``ov.bm`` (device contour or classical self-seg per ``ov.bm_src``).
    """
    if bm_choice == "dl":
        import bm_dl                                          # reader.core put src/ on sys.path
        if not bm_dl.discoverable():
            return bm, ov.bm_src, ("DL Bruch's-membrane model is unavailable; "
                                   f"used the {ov.bm_src} segmentation instead.")
        try:
            # Two B-scans per ONNX batch keeps peak activation memory modest on ordinary clinic PCs.
            # Batch size does not change per-image inference or the resulting BM surface.
            dl_bm = bm_dl.segment_volume(ov.vol, bs=2).astype(bm.dtype)
            return dl_bm, "dl", None
        except Exception as e:                                # noqa: BLE001 (never fail a scan over BM)
            return bm, ov.bm_src, (f"DL Bruch's-membrane failed ({e}); "
                                   f"used the {ov.bm_src} segmentation instead.")
    # "device" / "auto": ov.bm already holds the device contour or the classical self-seg.
    return bm, ov.bm_src, None


def process(raw, index, bm_choice):
    """Load volume ``index``, apply ``bm_choice``, compute GA, and return ``(vid, LiveSource, warning)``.

    Validates ``index`` is a 6x6 volume. ``bm_choice`` in {``device``, ``dl``, ``auto``}; an unknown
    value is treated as ``auto`` (keep the loaded BM)."""
    if not (0 <= int(index) < len(raw.refs)):
        raise ValueError(f"scan index {index} is out of range for this E2E")
    ref = raw.refs[int(index)]
    if not ref.is_6x6:
        raise ValueError("the chosen scan is not a 6x6-measurable volume")

    ov = e2e_source.load_volume(raw, int(index))
    ilm, bm = core_layers.effective_surfaces(ov, None)        # uploads: no corrections store
    bm, bm_source, warning = _resolve_bm(ov, ilm, bm, bm_choice)
    vm = viewmodel.compute(raw, ov, ilm, bm)                  # THE GA number (baseline="radial2")

    vid = f"live:{ov.volume_id}" + ("|dl" if bm_source == "dl" else "")
    meta = {
        "schema": bundle.SCHEMA, "slug": vid,
        "subject": identity.patient_name(raw) or identity.patient_id(raw, int(index)),
        "visit": None, "eye": ov.eye,
        "patient_id": identity.patient_id(raw, int(index)),
        "patient_name": identity.patient_name(raw),
        "acq_date": identity.acq_date(raw, int(index)),
        "n_bscans": ov.n_bscans, "H": ov.H, "W": ov.W,
        "fov_mm": [round(float(x), 3) for x in ov.fov_mm],
        "axial_um_per_px": viewmodel.AXIAL_UM_PER_PX, "enface_mmpp": bundle.ga_native.MMPP,
        "enface_out": int(vm["out"]), "feature": viewmodel.FEATURE,
        "slab_um": [float(x) for x in viewmodel.SLAB_UM],
        "oac_area_mm2": round(float(vm["oac_area_mm2"]), 4),
        # OCT-only: no PLEX anywhere. These empty fields keep meta_public()'s has_plex == False.
        "plex_area_mm2": None, "plex_source": None, "plex_polygons": [],
        "localizer_sid": vm["localizer_sid"], "is_control": False, "uploaded": True,
        "bm_source": bm_source, "volume_index": int(index), "bm_choice": bm_choice,
        # False on a reverse-scanned raster: the en-face rows were not flipped, so the viewer's
        # click-to-probe must map projection row -> B-scan directly instead of un-flipping.
        "enface_flip": bool(vm.get("enface_flip", True)),
    }
    return vid, bundle.LiveSource(vm, meta), warning
