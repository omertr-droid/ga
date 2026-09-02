"""In-memory session for the clinic: cache opened E2Es and processed scans, resolve a ``vid`` to a
serveable ViewSource, and persist every processed scan to the patient database.

Two LRU caches:
  * ``_raw``  — ``eid -> RawE2E``. Opening an E2E runs the costly ``read_oct_volume`` + metadata decode
    once; the chooser, ``process`` and any re-open then reuse it. Small cap (a clinician works one or
    two files at a time).
  * ``_live`` — ``vid -> LiveSource``. A processed (volume, BM-choice) pair, held so the panel endpoints
    serve it without recomputing. A scan processed both device and DL caches two distinct vids
    (``…`` and ``…|dl``).

The store is the only stateful object; the API layer holds a single module-level instance.
"""
import threading
from collections import OrderedDict

from reader.core import ids

from . import db, pipeline

# A clinician processes one E2E at a time (usually OD + OS).  RawE2E owns every decoded volume in the
# file and is by far the largest object, while a LiveSource is a compact uint8 viewer result.  Bound the
# steady state accordingly; ``finish_batch`` drops even the one raw object after both eyes are done.
_MAX_RAW = 1
_MAX_LIVE = 2


class ClinicStore:
    def __init__(self, max_raw=_MAX_RAW, max_live=_MAX_LIVE):
        self._raw = OrderedDict()      # eid -> RawE2E
        self._live = OrderedDict()     # vid -> LiveSource
        self._max_raw = max_raw
        self._max_live = max_live
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ E2E open (cached)
    def open(self, path):
        """Open (or reuse) an E2E by path; cache the decoded RawE2E keyed by its stable id."""
        eid = ids.e2e_id(path)
        with self._lock:
            raw = self._raw.get(eid)
            if raw is not None:
                self._raw.move_to_end(eid)
        if raw is None:
            raw = pipeline.open_e2e(path)                     # costly decode, outside the lock
            with self._lock:
                self._raw[eid] = raw
                self._raw.move_to_end(eid)
                while len(self._raw) > self._max_raw:
                    self._raw.popitem(last=False)
        return raw

    # ------------------------------------------------------------------ chooser (step 1)
    def list_scans(self, path):
        """The upload chooser payload for an E2E: 6x6 scans + DL availability + identity (no GA)."""
        return pipeline.list_scans(self.open(path))

    # ------------------------------------------------------------------ process (step 2)
    def process(self, path, index, bm_choice):
        """Process a chosen scan, cache it under its vid, and record it to the patient database.
        Returns ``(vid, meta_public, db_result, warning)`` where db_result = {saved, warning, record_id}."""
        raw = self.open(path)
        vid, src, warning = pipeline.process(raw, index, bm_choice)
        with self._lock:
            self._live[vid] = src
            self._live.move_to_end(vid)
            while len(self._live) > self._max_live:
                self._live.popitem(last=False)
        reg = db.try_record(src.meta_json(), path, vid, index, bm_choice)
        return vid, src.meta_public(), reg, warning

    def finish_batch(self, path=None):
        """Release decoded E2E + DL-session memory after an upload/reopen batch.

        Processed ``LiveSource`` eyes stay resident for the viewer.  If ``path`` is omitted every raw
        decoder is released (used by the one-eye database reopen path).  Returns small diagnostics for
        smoke tests and the API; release failures never invalidate a successfully processed scan.
        """
        with self._lock:
            if path:
                raw_released = int(self._raw.pop(ids.e2e_id(path), None) is not None)
            else:
                raw_released = len(self._raw)
                self._raw.clear()
            live_retained = len(self._live)
        model_released = False
        try:
            import bm_dl
            model_released = bool(bm_dl.release())
        except Exception:
            pass
        # CPython frees arrays by reference counting; collection also clears any short-lived cycles from
        # E2E metadata/reader objects.  Import locally so normal module import remains cheap.
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        return {"ok": True, "raw_released": raw_released, "model_released": model_released,
                "live_retained": live_retained}

    # ------------------------------------------------------------------ re-open from the database
    def reopen(self, record_id):
        """Re-open a scan recorded in the database. Serve it from cache if still resident, else
        re-process from the stored ``e2e_path`` / ``volume_index`` / ``bm_choice`` (deterministic — the
        recomputed area matches the stored one). Returns ``(vid, meta_public, db_result, warning)``.
        Raises KeyError (no such record) or FileNotFoundError (the E2E is not on this machine)."""
        row = db.get(record_id)
        if row is None:
            raise KeyError(record_id)
        vid = row.get("vid") or ""
        with self._lock:
            src = self._live.get(vid)
            if src is not None:
                self._live.move_to_end(vid)
        if src is not None:
            return vid, src.meta_public(), {"saved": True, "warning": None, "record_id": record_id}, None

        path = row.get("e2e_path") or ""
        if not path:
            raise FileNotFoundError("this record has no stored E2E path, so it cannot be re-opened")
        import os
        if not os.path.exists(path):
            raise FileNotFoundError("the E2E for this scan is not on this computer")
        try:
            index = int(row.get("volume_index"))
        except (TypeError, ValueError):
            raise FileNotFoundError("this record has no stored scan index, so it cannot be re-opened")
        return self.process(path, index, row.get("bm_choice") or "auto")

    # ------------------------------------------------------------------ panel serving (step 3)
    def has(self, vid) -> bool:
        with self._lock:
            return vid in self._live

    def source(self, vid):
        """Resolve a ``live:`` vid to its LiveSource; raise KeyError if evicted/unknown (-> 404)."""
        if not vid or not vid.startswith("live:"):
            raise KeyError(vid)
        with self._lock:
            src = self._live.get(vid)
        if src is None:
            raise KeyError(vid)
        return src
