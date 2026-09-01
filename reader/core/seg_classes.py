"""Per-class annotation registry — the single source of truth for the Segment tab's GA classes.

A B-scan annotation is split into CLASSES (now: the hypertransmission **wedge** and the **RPE** band).
Each class is an ordinary binary-mask RUN (reader.core.mask_store); a class's run id is derived from a
*base* id so several classes form one "annotation" group:

    base 'gold'  ->  wedge run 'gold'  +  rpe run 'gold:rpe'

The DEFAULT class (`wedge`) uses the base id verbatim, so existing single-class 'gold' runs ARE the wedge
class with **zero migration**. cRORA GA = (wedge present) AND (RPE-loss), where RPE-loss is the inverted
interior gap in the painted RPE-present band (footprint.native_flags(invert=True)).

Adding a class later (drusen/PED, ...) is config-only: add an entry below + to SHIP. Bands come from the
pipeline (m3_projections) so the studio and the area math agree on geometry.
"""
import m3_projections as mp

# area_role: "pos" -> painted columns ARE the lesion (wedge);
#            "gap" -> painted band is INTACT tissue, the lesion is its inverted interior gap (rpe).
CLASSES = {
    "wedge": {
        "key": "wedge", "name": "Hypertransmission wedge",
        "band_um": tuple(mp.SLAB_UM),          # (10, 340): sub-BM transmitted-light slab (below BM)
        "color_rgb": (255, 255, 0),            # yellow
        "seed": "threshold", "concept": "hypertransmission wedge",
        "area_role": "pos", "invert_default": False,
    },
    "rpe": {
        "key": "rpe", "name": "RPE band (present)",
        "band_um": tuple(mp.RPEBAND_UM),       # (-45, -5): outer hyperreflective complex (EZ/IZ/RPE), above BM
        "color_rgb": (255, 0, 200),            # magenta
        "seed": "medsam3", "concept": "RPE band",
        "area_role": "gap", "invert_default": True,
    },
}
SHIP = ["wedge", "rpe"]          # classes shipped now (also the selector order)
DEFAULT_CLASS = "wedge"


def cfg(cls: str) -> dict:
    return CLASSES[cls if cls in CLASSES else DEFAULT_CLASS]


def run_id(base: str, cls: str) -> str:
    """Class run id. The DEFAULT class uses the base verbatim (legacy 'gold' == wedge, no migration)."""
    return base if cls == DEFAULT_CLASS else f"{base}:{cls}"


def base_class(rid: str, meta: dict = None):
    """(run id, optional meta) -> (base, class). 'gold'->('gold','wedge'); 'gold:rpe'->('gold','rpe')."""
    if meta and meta.get("class") in CLASSES:
        return (rid.split(":", 1)[0], meta["class"])
    if ":" in rid:
        b, c = rid.split(":", 1)
        return (b, c if c in CLASSES else DEFAULT_CLASS)
    return (rid, DEFAULT_CLASS)


def band_px(cls: str):
    """Class depth band as (lo_px, hi_px) relative to BM, for rasterising the en-face footprint onto
    B-scans (mp.AX = axial µm/px)."""
    lo, hi = cfg(cls)["band_um"]
    return lo / mp.AX, hi / mp.AX


def bgra(cls: str, alpha: int = 255):
    """cv2 BGRA tuple for a class overlay color (color_rgb is R,G,B)."""
    r, g, b = cfg(cls)["color_rgb"]
    return (int(b), int(g), int(r), int(alpha))
