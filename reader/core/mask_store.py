"""PngMaskStore — persisted GA segmentation masks, the Segment-tab analogue of JsonSidecarLayerStore.

A *run* is one labeling attempt (a prompt/threshold/manual pass). Each run holds one binary B-scan
mask per labeled B-scan; collapsing those to per-A-scan column flags and projecting to the en-face frame
gives the run's GA footprint + area (see core.footprint). Layout on disk:

    reader/data_store/segmentations/<eid>_<eye>/runs.json              # {runs:{id:meta}, active:id}
    reader/data_store/segmentations/<eid>_<eye>/<run_id>/bscan_<idx>.png   # binary mask (0/255, HxW)

Runs are cheap to create/compare/delete (try a prompt, keep the good one). Masks are 2D (depth x A-scan)
so the user can SEE and brush them on the B-scan; the footprint math only reads their per-column extent.
Atomic writes, a threading.Lock, and a lazy per-(eid,eye) cache mirror the layer store. Bound in
api/deps.py (get_mask_store).
"""
import datetime
import json
import os
import threading
from typing import Optional

import cv2
import numpy as np


def _safe(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(part))


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class PngMaskStore:
    def __init__(self, root: str):
        self.root = root
        self._lock = threading.Lock()
        self._runs = {}     # (eid, eye) -> {"runs": {id: meta}, "active": id|None}
        self._masks = {}    # (eid, eye, run) -> {bscan:int -> bool ndarray (H,W)}
        self._status = {}   # (eid, eye, run) -> {bscan:int -> {state, by, reviewed, ts}}

    # ------------------------------------------------------------------ paths
    def _dir(self, eid, eye) -> str:
        return os.path.join(self.root, f"{_safe(eid)}_{_safe(eye)}")

    def _run_dir(self, eid, eye, run) -> str:
        return os.path.join(self._dir(eid, eye), _safe(run))

    def _mask_file(self, eid, eye, run, bscan) -> str:
        return os.path.join(self._run_dir(eid, eye, run), f"bscan_{int(bscan):04d}.png")

    def _runs_file(self, eid, eye) -> str:
        return os.path.join(self._dir(eid, eye), "runs.json")

    # ------------------------------------------------------------------ runs.json (lazy)
    def _load_runs(self, eid, eye) -> dict:
        key = (eid, eye)
        r = self._runs.get(key)
        if r is not None:
            return r
        r = {"runs": {}, "active": None}
        f = self._runs_file(eid, eye)
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    disk = json.load(fh)
                if isinstance(disk, dict) and isinstance(disk.get("runs"), dict):
                    r = {"runs": disk["runs"], "active": disk.get("active")}
            except (OSError, json.JSONDecodeError):
                pass
        self._runs[key] = r
        return r

    def _save_runs(self, eid, eye) -> None:
        r = self._runs[(eid, eye)]
        os.makedirs(self._dir(eid, eye), exist_ok=True)
        tmp = self._runs_file(eid, eye) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(r, fh)
        os.replace(tmp, self._runs_file(eid, eye))

    def list_runs(self, eid, eye) -> dict:
        """{"runs": {id: meta}, "active": id|None} — a copy safe to serialize."""
        with self._lock:
            r = self._load_runs(eid, eye)
            return {"runs": {k: dict(v) for k, v in r["runs"].items()}, "active": r["active"]}

    def create_run(self, eid, eye, run_id, meta: dict, make_active=True) -> dict:
        with self._lock:
            r = self._load_runs(eid, eye)
            m = {"label": run_id, "source": "manual", "concept": None, "threshold": None}
            m.update(meta or {})
            m["created"] = m.get("created") or _now()
            r["runs"][run_id] = m
            if make_active or r["active"] is None:
                r["active"] = run_id
            self._save_runs(eid, eye)
            return dict(m)

    def update_run(self, eid, eye, run_id, patch: dict) -> None:
        with self._lock:
            r = self._load_runs(eid, eye)
            if run_id in r["runs"]:
                r["runs"][run_id].update(patch or {})
                self._save_runs(eid, eye)

    def set_active(self, eid, eye, run_id) -> None:
        with self._lock:
            r = self._load_runs(eid, eye)
            if run_id is None or run_id in r["runs"]:
                r["active"] = run_id
                self._save_runs(eid, eye)

    def delete_run(self, eid, eye, run_id) -> None:
        import shutil
        with self._lock:
            r = self._load_runs(eid, eye)
            r["runs"].pop(run_id, None)
            if r["active"] == run_id:
                r["active"] = next(iter(r["runs"]), None)
            self._save_runs(eid, eye)
            self._masks.pop((eid, eye, run_id), None)
            self._status.pop((eid, eye, run_id), None)
            d = self._run_dir(eid, eye, run_id)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    def run_exists(self, eid, eye, run_id) -> bool:
        with self._lock:
            return run_id in self._load_runs(eid, eye)["runs"]

    def new_run_id(self, eid, eye, prefix="run") -> str:
        with self._lock:
            runs = self._load_runs(eid, eye)["runs"]
        i = 1
        while f"{prefix}-{i}" in runs:
            i += 1
        return f"{prefix}-{i}"

    # ------------------------------------------------------------------ masks (lazy per run)
    def _load_run_masks(self, eid, eye, run) -> dict:
        key = (eid, eye, run)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        out = {}
        d = self._run_dir(eid, eye, run)
        if os.path.isdir(d):
            for name in os.listdir(d):
                if not (name.startswith("bscan_") and name.endswith(".png")):
                    continue
                try:
                    bi = int(name[len("bscan_"):-len(".png")])
                    a = cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE)
                    if a is not None:
                        out[bi] = a > 127
                except (ValueError, OSError):
                    continue
        self._masks[key] = out
        return out

    def get_mask(self, eid, eye, run, bscan) -> Optional[np.ndarray]:
        """Binary mask (H,W bool) for one B-scan of a run, or None."""
        with self._lock:
            return self._load_run_masks(eid, eye, run).get(int(bscan))

    def mask_indices(self, eid, eye, run) -> list:
        with self._lock:
            return sorted(self._load_run_masks(eid, eye, run).keys())

    def put_mask(self, eid, eye, run, bscan, mask) -> None:
        """Persist a binary mask (any truthy array HxW) as a 0/255 PNG, atomically; update cache.
        An all-empty mask deletes the B-scan's mask (keeps the store sparse)."""
        m = np.asarray(mask)
        m = (m > 0) if m.dtype != bool else m
        if not m.any():
            self.delete_mask(eid, eye, run, bscan)
            return
        with self._lock:
            self._load_run_masks(eid, eye, run)[int(bscan)] = m
            d = self._run_dir(eid, eye, run)
            os.makedirs(d, exist_ok=True)
            ok, buf = cv2.imencode(".png", (m.astype(np.uint8) * 255))
            if not ok:
                raise RuntimeError("mask PNG encode failed")
            tmp = self._mask_file(eid, eye, run, bscan) + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(buf.tobytes())
            os.replace(tmp, self._mask_file(eid, eye, run, bscan))

    def delete_mask(self, eid, eye, run, bscan) -> None:
        with self._lock:
            self._load_run_masks(eid, eye, run).pop(int(bscan), None)
            f = self._mask_file(eid, eye, run, bscan)
            if os.path.exists(f):
                os.remove(f)

    def clear_masks(self, eid, eye, run) -> None:
        """Drop every mask in a run (keep the run + its metadata)."""
        import shutil
        with self._lock:
            self._masks[(eid, eye, run)] = {}
            self._status[(eid, eye, run)] = {}
            d = self._run_dir(eid, eye, run)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ per-B-scan status (the studio)
    # Sidecar <eid>_<eye>/<run>/status.json holds EXPLICIT states only (ga_free / borderline / reviewed,
    # and 'ga' when painted so it overrides a prior ga_free). A B-scan with a mask but no explicit entry
    # derives to 'ga'; with neither, to 'todo'. So 'todo' (unseen) is never confused with 'ga_free'.
    def _status_file(self, eid, eye, run) -> str:
        return os.path.join(self._run_dir(eid, eye, run), "status.json")

    def _load_status(self, eid, eye, run) -> dict:
        key = (eid, eye, run)
        cached = self._status.get(key)
        if cached is not None:
            return cached
        out = {}
        f = self._status_file(eid, eye, run)
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    out = {int(k): v for k, v in json.load(fh).items()}
            except (OSError, json.JSONDecodeError, ValueError, AttributeError):
                out = {}
        self._status[key] = out
        return out

    def _save_status(self, eid, eye, run) -> None:
        os.makedirs(self._run_dir(eid, eye, run), exist_ok=True)
        tmp = self._status_file(eid, eye, run) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({str(k): v for k, v in self._status[(eid, eye, run)].items()}, fh)
        os.replace(tmp, self._status_file(eid, eye, run))

    def set_status(self, eid, eye, run, bscan, state=None, by=None, reviewed=None) -> None:
        with self._lock:
            s = self._load_status(eid, eye, run)
            rec = dict(s.get(int(bscan)) or {})
            if state is not None:
                rec["state"] = state
            if by is not None:
                rec["by"] = by
            if reviewed is not None:
                rec["reviewed"] = bool(reviewed)
            rec["ts"] = _now()
            s[int(bscan)] = rec
            self._save_status(eid, eye, run)

    def clear_status(self, eid, eye, run, bscan) -> None:
        with self._lock:
            s = self._load_status(eid, eye, run)
            if int(bscan) in s:
                s.pop(int(bscan))
                self._save_status(eid, eye, run)

    def derived_status(self, eid, eye, run, n_bscans) -> dict:
        """Per-B-scan state: explicit state wins, else a mask ⇒ 'ga', else 'todo'.
        Returns {bscan:int -> {state, reviewed, by}} for bscan in [0, n_bscans)."""
        with self._lock:
            explicit = dict(self._load_status(eid, eye, run))
            masks = set(self._load_run_masks(eid, eye, run).keys())
        out = {}
        for i in range(int(n_bscans)):
            rec = explicit.get(i) or {}
            state = rec.get("state") or ("ga" if i in masks else "todo")
            out[i] = {"state": state, "reviewed": bool(rec.get("reviewed")), "by": rec.get("by")}
        return out
