"""Bake-time PLEX (advRPE) GA registration -> projection-frame OUTLINE polygons (the blue line).

Ports reader/api/routes_segmentation._plex_registered (the two-map GA-driven lock) WITHOUT the web
deps, so it runs in the offline baker. Registers a sub-RPE shadowgram from our volume to the advRPE
SubRPE en-face, then warps the advRPE GA mask into our projection frame. This is GA-driven => a
reference/label overlay, not an independent registration. The blue OUTLINE shows where PLEX called GA;
the PLEX AREA number is independent (read straight from the advRPE CSV).

OCT-only constraint: this is used ONLY to draw a reference overlay + report PLEX's own area beside ours.
It never feeds our computed GA number.
"""
import cv2
import numpy as np

from . import ga_native


def registered_label(vol, bm, fov_mm, subrpe_path, ga_mask_path, enface_flip=True):
    """512x512 bool advRPE-GA label in the to_6mm frame (centre-equivalent to to_enface), or None.

    `enface_flip` must match ov.enface_flip, or the shadowgram we register against is upside-down
    relative to the projection the label will be drawn on (reverse-scanned 003-016 / 003-130)."""
    import m3_projections as mp
    import qcviz as qv
    import register_qc as rq
    adv = cv2.imread(subrpe_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(ga_mask_path, cv2.IMREAD_GRAYSCALE)
    if adv is None or m is None:
        return None
    adv6 = cv2.resize(adv, (512, 512))
    mask6 = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
    shadow = mp.to_6mm(mp.destripe2d(mp.band(vol, bm, 10, 340, "mean")), list(fov_mm), enface_flip)
    shadow6 = cv2.resize(qv.norm8(shadow), (512, 512))
    try:
        reg = rq.register(shadow6, adv6)
    except Exception:
        return None
    flipf = np.fliplr if reg.get("flip") else (lambda a: a)
    minv = cv2.invertAffineTransform(reg["M"])               # advRPE GA -> our frame (inverse)
    return flipf(rq.warp((mask6 * 255).astype(np.uint8), minv, cv2.INTER_NEAREST)) > 127


def outline_polygons(label512, out, epsilon=1.5, min_pts=6):
    """Center-fit the 512 label into the `out` en-face frame, then external contours as polygons
    [[ [x,y], ... ], ...] in en-face pixels (for a crisp client-drawn blue line). [] if empty/None."""
    if label512 is None:
        return []
    fit = ga_native.center_extract(label512.astype(np.uint8), out, out)
    cnts, _ = cv2.findContours(fit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        ap = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(ap) >= min_pts or cv2.contourArea(c) > 25:
            polys.append([[int(x), int(y)] for x, y in ap])
    return polys
