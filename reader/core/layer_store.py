"""JsonSidecarLayerStore — persisted manual layer corrections (Phase 2).

Implements the core.layers.LayerStore Protocol. Each corrected B-scan is one JSON sidecar
    reader/data_store/corrections/<eid>_<eye>/bscan_<idx>.json   ->  {"ilm": [y|null, ...], "bm": [...]}
holding a full-width row per layer (one value per A-scan; null = unset -> the route null-fills from the
auto/device surface before storing, so stored rows are projection-clean). A lazy in-RAM cache keyed by
(eid, eye) avoids re-reading disk on every B-scan/projection request; writes update both disk and cache.

Bound in api/deps.py (the one-line swap from NullLayerStore). The projection reads corrected surfaces
through layers.effective_surfaces, and the viewer shows them through layers.device_layers_json — neither
needs to change when this store is enabled.
"""
import datetime
import json
import os
import threading
from typing import Optional


def _safe(part: str) -> str:
    """Filesystem-safe path component (eid is a hex hash; eye is OD/OS/U; be defensive anyway)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(part))


class JsonSidecarLayerStore:
    def __init__(self, root: str):
        self.root = root
        self._lock = threading.Lock()
        self._cache = {}            # (eid, eye) -> {bscan:int -> {"ilm": [...]|None, "bm": [...]|None}}
        self._gcache = {}           # (eid, eye) -> {"ilm": shift_px, "bm": shift_px} (whole-volume)
        self._status = {}           # (eid, eye) -> {bscan:int -> {"validated":bool,"by":..,"ts":..}}
        self._missing = {}          # (eid, eye) -> int: # B-scans with missing BM cols (lazy Library cache)

    # ------------------------------------------------------------------ paths
    def _dir(self, eid: str, eye: str) -> str:
        return os.path.join(self.root, f"{_safe(eid)}_{_safe(eye)}")

    def _file(self, eid: str, eye: str, bscan: int) -> str:
        return os.path.join(self._dir(eid, eye), f"bscan_{int(bscan):04d}.json")

    def _gfile(self, eid: str, eye: str) -> str:
        return os.path.join(self._dir(eid, eye), "global.json")

    def _status_file(self, eid: str, eye: str) -> str:
        return os.path.join(self._dir(eid, eye), "bm_status.json")

    def _missing_file(self, eid: str, eye: str) -> str:
        return os.path.join(self._dir(eid, eye), "missing.json")

    # ------------------------------------------------------------------ cache (lazy load from disk)
    def _load(self, eid: str, eye: str) -> dict:
        key = (eid, eye)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = {}
        d = self._dir(eid, eye)
        if os.path.isdir(d):
            for name in os.listdir(d):
                if not (name.startswith("bscan_") and name.endswith(".json")):
                    continue
                try:
                    bi = int(name[len("bscan_"):-len(".json")])
                    with open(os.path.join(d, name)) as f:
                        out[bi] = json.load(f)
                except (ValueError, OSError, json.JSONDecodeError):
                    continue
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ LayerStore Protocol
    def get_corrected(self, eid: str, eye: str, bscan: int) -> Optional[dict]:
        with self._lock:
            return self._load(eid, eye).get(int(bscan))

    def put_corrected(self, eid: str, eye: str, bscan: int, layer_key: str, ys: list,
                      source: str = "user") -> None:
        """Store a layer correction. `source` tags WHO produced it: "user" (a human edit / classical
        re-seg — the default) or "model" (a DL pre-segmentation / Label-with-DL). Persisted as the
        companion key f"{layer_key}_src" so the filmstrip can distinguish a model pre-seg awaiting review
        from a human edit. A later human edit re-writes this with the default -> the tag flips to "user"."""
        if layer_key not in ("ilm", "bm"):
            raise ValueError(f"unknown layer {layer_key!r}")
        with self._lock:
            store = self._load(eid, eye)
            rec = dict(store.get(int(bscan)) or {})
            rec[layer_key] = list(ys)
            rec[f"{layer_key}_src"] = source
            store[int(bscan)] = rec
            d = self._dir(eid, eye)
            os.makedirs(d, exist_ok=True)
            tmp = self._file(eid, eye, bscan) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, self._file(eid, eye, bscan))   # atomic write

    # ------------------------------------------------------------------ extras (delete / list)
    def delete_corrected(self, eid: str, eye: str, bscan: int, layer_key: Optional[str] = None) -> None:
        """Drop one layer's correction for a B-scan (layer_key given) or the whole B-scan (None)."""
        with self._lock:
            store = self._load(eid, eye)
            rec = store.get(int(bscan))
            if rec is None:
                return
            if layer_key is None or layer_key not in rec:
                rec = None
            else:
                drop = {layer_key, f"{layer_key}_src"}         # drop the source tag with its layer
                rec = {k: v for k, v in rec.items() if k not in drop}
                if not any(k in rec for k in ("ilm", "bm")):   # only stray *_src tags left -> empty
                    rec = None
            path = self._file(eid, eye, bscan)
            if rec is None:
                store.pop(int(bscan), None)
                if os.path.exists(path):
                    os.remove(path)
            else:
                store[int(bscan)] = rec
                with open(path, "w") as f:
                    json.dump(rec, f)

    def clear_corrected_all(self, eid: str, eye: str, layer_key: Optional[str] = None) -> int:
        """Bulk 'reset to device': drop `layer_key`'s correction (or the whole record when None) from
        EVERY corrected B-scan for this eye. Returns the number of B-scans affected."""
        with self._lock:
            store = self._load(eid, eye)
            affected = 0
            for bi in list(store.keys()):
                rec = store.get(bi) or {}
                if layer_key is not None and layer_key not in rec:
                    continue
                affected += 1
                if layer_key is None:
                    rec = None
                else:
                    drop = {layer_key, f"{layer_key}_src"}     # drop the source tag with its layer
                    rec = {k: v for k, v in rec.items() if k not in drop}
                    if not any(k in rec for k in ("ilm", "bm")):
                        rec = None
                path = self._file(eid, eye, bi)
                if rec is None:
                    store.pop(bi, None)
                    if os.path.exists(path):
                        os.remove(path)
                else:
                    store[bi] = rec
                    with open(path, "w") as f:
                        json.dump(rec, f)
            return affected

    def corrected_indices(self, eid: str, eye: str) -> list:
        with self._lock:
            return sorted(self._load(eid, eye).keys())

    # ------------------------------------------------------------------ whole-volume (global) shift
    def _load_global(self, eid: str, eye: str) -> dict:
        key = (eid, eye)
        g = self._gcache.get(key)
        if g is None:
            g = {}
            f = self._gfile(eid, eye)
            if os.path.exists(f):
                try:
                    with open(f) as fh:
                        g = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    g = {}
            self._gcache[key] = g
        return g

    def get_global(self, eid: str, eye: str) -> dict:
        """{'ilm': px, 'bm': px} rigid offset applied to the WHOLE layer surface (all B-scans)."""
        with self._lock:
            return dict(self._load_global(eid, eye))

    def add_global(self, eid: str, eye: str, layer_key: str, delta: float) -> float:
        """Accumulate a rigid whole-volume shift for a layer; returns the new total."""
        if layer_key not in ("ilm", "bm"):
            raise ValueError(f"unknown layer {layer_key!r}")
        with self._lock:
            g = self._load_global(eid, eye)
            g[layer_key] = float(g.get(layer_key, 0.0)) + float(delta)
            os.makedirs(self._dir(eid, eye), exist_ok=True)
            tmp = self._gfile(eid, eye) + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(g, fh)
            os.replace(tmp, self._gfile(eid, eye))
            return g[layer_key]

    def clear_global(self, eid: str, eye: str, layer_key=None) -> None:
        with self._lock:
            g = self._load_global(eid, eye)
            if layer_key is None:
                g.clear()
            else:
                g.pop(layer_key, None)
            f = self._gfile(eid, eye)
            if g:
                with open(f, "w") as fh:
                    json.dump(g, fh)
            elif os.path.exists(f):
                os.remove(f)

    # ------------------------------------------------------------------ BM validation status
    # Per-B-scan "this BM is validated/approved" flag for the BM-labeling workflow. One sidecar/eye
    #     reader/data_store/corrections/<eid>_<eye>/bm_status.json  ->  {"<bscan>": {"validated","by","ts"}}
    # kept here so the validated flag and the BM corrections share one per-(eid, eye) folder.
    def _load_status(self, eid: str, eye: str) -> dict:
        key = (eid, eye)
        cached = self._status.get(key)
        if cached is not None:
            return cached
        out = {}
        f = self._status_file(eid, eye)
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    out = {int(k): v for k, v in json.load(fh).items()}
            except (ValueError, OSError, json.JSONDecodeError):
                out = {}
        self._status[key] = out
        return out

    def _save_status(self, eid: str, eye: str) -> None:
        s = self._status.get((eid, eye), {})
        f = self._status_file(eid, eye)
        if s:
            os.makedirs(self._dir(eid, eye), exist_ok=True)
            tmp = f + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({str(k): v for k, v in s.items()}, fh)
            os.replace(tmp, f)                              # atomic write
        elif os.path.exists(f):
            os.remove(f)

    def set_bm_validated(self, eid: str, eye: str, bscan: int, validated: bool = True,
                         by: Optional[str] = None) -> None:
        with self._lock:
            s = self._load_status(eid, eye)
            if validated:
                rec = dict(s.get(int(bscan)) or {})
                rec["validated"] = True
                if by is not None:
                    rec["by"] = by
                rec["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
                s[int(bscan)] = rec
            else:
                s.pop(int(bscan), None)
            self._save_status(eid, eye)

    def bm_validated(self, eid: str, eye: str) -> list:
        """Sorted list of B-scan indices marked validated for this eye."""
        with self._lock:
            s = self._load_status(eid, eye)
            return sorted(k for k, v in s.items() if v.get("validated"))

    def clear_bm_validated_all(self, eid: str, eye: str) -> int:
        """Clear EVERY BM-validation flag for this eye (the bulk 'reset to device' wipes validations
        too, so the whole eye starts from the device segmentation again). Returns how many were set."""
        with self._lock:
            s = self._load_status(eid, eye)
            n = sum(1 for v in s.values() if v.get("validated"))
            s.clear()
            self._save_status(eid, eye)        # empty -> removes the sidecar
            return n

    # ------------------------------------------------------------------ missing-BM count (Library cache)
    # How many B-scans still have device-gap BM columns. Computed when the eye's BM tab is opened
    # (routes get_bm_status) and cached to missing.json so the Library lists it without decoding the E2E.
    def set_missing_count(self, eid: str, eye: str, n: int) -> None:
        with self._lock:
            key = (eid, eye)
            n = int(n)
            if self._missing.get(key) == n:
                return                                       # unchanged -> skip the disk write
            self._missing[key] = n
            os.makedirs(self._dir(eid, eye), exist_ok=True)
            tmp = self._missing_file(eid, eye) + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"n": n, "ts": datetime.datetime.now().isoformat(timespec="seconds")}, f)
            os.replace(tmp, self._missing_file(eid, eye))    # atomic write

    def get_missing_count(self, eid: str, eye: str):
        """Cached # of B-scans with missing BM columns, or None if never computed (eye not yet opened)."""
        with self._lock:
            key = (eid, eye)
            if key in self._missing:
                return self._missing[key]
            f = self._missing_file(eid, eye)
            if os.path.exists(f):
                try:
                    with open(f) as fh:
                        n = int(json.load(fh).get("n"))
                    self._missing[key] = n
                    return n
                except (ValueError, OSError, json.JSONDecodeError, TypeError):
                    return None
            return None
