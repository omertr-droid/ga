#!/usr/bin/env python
"""Optional DL Bruch's-membrane segmenter — a gated drop-in for bm.segment_bm / bm.segment_volume.

OFF unless opted in. `active()` is True only when a model + an inference backend are present
(`available()`) AND the opt-in env flag `OCT_BM_DL` is set. `bm.py` delegates to this when active(); the
doctor viewer can also call `segment_volume()` directly when its "DL BM" option is on. ANY failure
(missing model, bad backend, wrong shape) is swallowed by the caller, which falls back to the classical
`bm.py` segmenter — so this is always safe to import.

Backends (first that can load the found model): ONNX via onnxruntime (light — ships in the offline
viewer, no torch) then torch (.pt, dev box only). Model search order:
    $OCT_BM_DL_MODEL  ->  outputs/bm_dl/bm_unet.onnx  ->  outputs/bm_dl/bm_unet_all.pt
    ->  segmenter_service/bm_unet.onnx  ->  segmenter_service/bm_unet_all.pt

Preprocessing matches the training notebook EXACTLY (train/infer parity): per-B-scan norm8 (percentile
1-99 -> uint8) /255 -> 2.5D channels [i-1, i, i+1] -> bottom reflect-pad to a /32 height -> U-Net ->
soft-argmax over the NATIVE (unpadded) rows -> BM row per A-scan. The model is BM-only (ILM unchanged).
"""
import json
import importlib.util
import os

import numpy as np

try:
    from paths import OUT_DIR, REPO_ROOT
except Exception:                                          # importable even without src on the path
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT_DIR = os.path.join(REPO_ROOT, "outputs")

_ENV_FLAG = "OCT_BM_DL"          # opt-in: "1"/"true"/"yes"/"on"
_ENV_MODEL = "OCT_BM_DL_MODEL"   # explicit model path override
_ENV_SMOOTH = "OCT_BM_DL_SMOOTH_SIGMA"   # WITHIN-B-scan column-smoothing sigma (px); 0 disables
_ENV_INPUT = "OCT_BM_DL_INPUT"   # "2.5d" (3 adjacent B-scans, default) | "2d" (single slice replicated)
_DEFAULT_SMOOTH = 1.5            # removes the per-column soft-argmax jitter (slope-flip 4.4%->0.9%) at no
#                                  accuracy cost (validated, 11 eyes); NEVER smooth across B-scans (they are
#                                  axially misaligned ~4px -> cross-slice smoothing wrecks accuracy).

_SEARCH = [
    os.path.join(OUT_DIR, "bm_dl", "bm_unet.onnx"),
    os.path.join(OUT_DIR, "bm_dl", "bm_unet_all.pt"),
    os.path.join(REPO_ROOT, "segmenter_service", "bm_unet.onnx"),
    os.path.join(REPO_ROOT, "segmenter_service", "bm_unet_all.pt"),
]

_session = None          # cached backend handle: an ort.InferenceSession or ("torch", model, torch)
_backend = None          # "onnx" | "torch" | None
_loaded_path = None      # path of the loaded model
_failed_path = None      # path we already tried and failed (don't re-import every call)


# --------------------------------------------------------------------------- discovery / gating
def model_path():
    """First existing model file ($OCT_BM_DL_MODEL or the search list), or None."""
    p = os.environ.get(_ENV_MODEL, "").strip()
    if p and os.path.exists(p):
        return p
    for c in _SEARCH:
        if os.path.exists(c):
            return c
    return None


def enabled():
    """The opt-in env flag (independent of whether a model exists)."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def discoverable():
    """Cheap capability check that does NOT allocate an inference session or load model weights.

    The clinic uses this for its landing page and scan chooser.  Actual inference still goes through
    ``_load``/``segment_volume`` and retains the existing safe fallback on a broken backend/model.
    """
    path = model_path()
    if path is None:
        return False
    module = "onnxruntime" if path.endswith(".onnx") else "torch"
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _load():
    """Lazy-load the model into a backend handle. Returns True if a usable backend is ready."""
    global _session, _backend, _loaded_path, _failed_path
    path = model_path()
    if path is None:
        return False
    if _session is not None and _loaded_path == path:
        return True
    if _failed_path == path:
        return False
    if path.endswith(".onnx"):
        try:
            import onnxruntime as ort
            _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            _backend, _loaded_path = "onnx", path
            return True
        except Exception:
            _session = None; _backend = None; _failed_path = path
            return False
    try:                                                   # torch .pt (dev box only)
        import torch
        import segmentation_models_pytorch as smp
        ckpt = torch.load(path, map_location="cpu")
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
        model.eval()
        _session = ("torch", model, torch)
        _backend, _loaded_path = "torch", path
        return True
    except Exception:
        _session = None; _backend = None; _failed_path = path
        return False


def available():
    """True when a model + a working backend are present (capability — independent of the opt-in flag)."""
    return _load()


def release():
    """Release loaded model/session memory after a clinic upload batch.

    Development callers can ignore this and keep the historical process-wide cache.  The next call to
    ``segment_volume`` simply reloads the same model.
    """
    global _session, _backend, _loaded_path
    had_session = _session is not None
    _session = None
    _backend = None
    _loaded_path = None
    return had_session


def active():
    """True when DL BM should actually be used: opted in AND available."""
    return enabled() and available()


def backend():
    """The loaded backend name ("onnx"/"torch") or None."""
    _load()
    return _backend if _loaded_path == model_path() else None


def info():
    """Small dict for the viewer's /api/config: {available, enabled, backend, model}."""
    return {"available": available(), "enabled": enabled(), "backend": backend(),
            "model": os.path.basename(model_path()) if model_path() else None}


# --------------------------------------------------------------------------- inference
def _norm8(b):
    """Percentile 1-99 contrast stretch to uint8 — byte-identical to qcviz.norm8 / render.bscan_png."""
    a = np.asarray(b, np.float32)
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    if hi <= lo:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255.0).astype(np.uint8)


def _pad_h(stack, pad_h):
    """Bottom reflect-pad (C, H, W) -> (C, pad_h, W) so BM rows stay valid (matches the notebook)."""
    h = stack.shape[1]
    return stack if h >= pad_h else np.pad(stack, ((0, 0), (0, pad_h - h), (0, 0)), mode="reflect")


def _infer(batch):
    """batch (n,3,PAD_H,W) float32 -> logits (n,1,PAD_H,W) float32."""
    if _backend == "onnx":
        name = _session.get_inputs()[0].name
        return np.asarray(_session.run(None, {name: batch.astype(np.float32)})[0])
    _, model, torch = _session
    with torch.no_grad():
        return model(torch.from_numpy(batch.astype(np.float32))).cpu().numpy()


def _soft_argmax(logits, h_native):
    """logits (n,1,PAD_H,W) -> expected BM row per column (n,W), over the native (unpadded) rows."""
    z = logits[:, 0, :h_native, :].astype(np.float32)
    z = z - z.max(axis=1, keepdims=True)                   # numerical stability
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True) + 1e-9
    rows = np.arange(h_native, dtype=np.float32)[None, :, None]
    return (p * rows).sum(axis=1)


def _smooth_cols(surf, sigma):
    """Light 1-D Gaussian smoothing of the BM surface ALONG COLUMNS (within each B-scan), pure numpy
    (reflect-padded) so the offline viewer needs no scipy. Removes the high-frequency, sub-pixel
    per-column soft-argmax jitter while preserving the surface (validated: slope-flip 4.4%->0.9%, BM
    error unchanged). Operates per row independently -> NEVER mixes across B-scans (those are axially
    misaligned, so cross-slice smoothing would corrupt the depth)."""
    if sigma is None or sigma <= 0:
        return surf
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    W = surf.shape[-1]
    p = np.pad(np.asarray(surf, np.float32), ((0, 0), (r, r)), mode="reflect")
    out = np.zeros_like(surf, dtype=np.float32)
    for j, kj in enumerate(k):
        out += kj * p[:, j:j + W]
    return out


def _smooth_sigma(override=None):
    if override is not None:
        return float(override)
    try:
        return float(os.environ.get(_ENV_SMOOTH, _DEFAULT_SMOOTH))
    except (TypeError, ValueError):
        return _DEFAULT_SMOOTH


def _input_mode(override=None):
    """Channel-build mode: explicit override > the model's own `<model>.meta.json` sidecar > env > 2.5d.
    The sidecar (written next to the model) makes a 2D-trained model SELF-DESCRIBING, so inference feeds
    single-slice without relying on an env var — train/infer parity travels with the model file."""
    if override is not None:
        return override
    p = model_path()
    if p:
        try:
            with open(p + ".meta.json") as f:
                m = (json.load(f).get("input_mode") or "").strip().lower()
            if m in ("2d", "2.5d"):
                return m
        except (OSError, ValueError):
            pass
    return "2d" if os.environ.get(_ENV_INPUT, "2.5d").strip().lower() == "2d" else "2.5d"


def segment_volume(volume, bs=8, smooth_sigma=None, mode=None):
    """BM bm[n_bscans, W] float32 for a whole volume via the DL model, with a light within-B-scan column
    smoother (sigma px; default 1.5, env OCT_BM_DL_SMOOTH_SIGMA, 0 disables). `mode` selects the channel
    build: "2.5d" = 3 adjacent B-scans [i-1,i,i+1] (default; env OCT_BM_DL_INPUT) or "2d" = the single
    slice replicated to 3 channels (for a model trained 2D — train/infer parity). Raises RuntimeError if
    no model/backend is available (callers fall back to classical)."""
    if not _load():
        raise RuntimeError("bm_dl: no model/backend available")
    md = _input_mode(mode)
    vol = np.asarray(volume, float)
    n, H, W = vol.shape
    pad_h = ((H + 31) // 32) * 32
    norm = np.stack([_norm8(vol[i]).astype(np.float32) / 255.0 for i in range(n)], 0)   # (n,H,W)
    out = np.zeros((n, W), np.float32)
    for s in range(0, n, bs):
        batch = []
        for i in range(s, min(s + bs, n)):
            if md == "2d":
                st = np.stack([norm[i], norm[i], norm[i]], 0)                            # single slice x3
            else:
                st = np.stack([norm[max(i - 1, 0)], norm[i], norm[min(i + 1, n - 1)]], 0)  # (3,H,W) 2.5D
            batch.append(_pad_h(st, pad_h))
        logits = _infer(np.stack(batch, 0))
        out[s:s + len(batch)] = _soft_argmax(logits, H)
    return _smooth_cols(out, _smooth_sigma(smooth_sigma))


def segment_bm(bscan):
    """BM y[W] float32 for ONE B-scan (the slice fills all three 2.5D channels, as at training).
    Raises RuntimeError if no model/backend is available."""
    return segment_volume(np.asarray(bscan, float)[None])[0]
