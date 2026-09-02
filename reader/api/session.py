"""In-memory session: decode each E2E once, hold the loaded volumes + projections, serve slices fast.

E2E decode + self-segmentation are slow (seconds); B-scan/layer/projection requests then slice cached
arrays. Bounded by small LRU caps so RAM stays sane. Locks serialize the first (slow) load of a given
E2E/volume so two concurrent requests don't decode twice. Handlers are sync `def`, so FastAPI runs them
in a threadpool — these blocking loads never stall the event loop.
"""
import threading
from collections import OrderedDict

from reader.core import e2e_source, ids, layers, projection


class SessionStore:
    def __init__(self, max_e2e=4, max_vol=6, max_proj=12):
        self._raw = OrderedDict()    # eid -> RawE2E
        self._vol = OrderedDict()    # volume_id -> OctVolume
        self._proj = OrderedDict()   # (volume_id, feature) -> en-face float frame
        self._nat = OrderedDict()    # (volume_id, feature) -> pre-finish native (n,W) for fast preview
        self._glock = threading.Lock()
        self._klocks = {}            # key -> lock (per eid / per volume_id)
        self._layer_store = None     # set by deps.attach_layer_store (Phase-2 corrections)

    def attach_layer_store(self, store):
        """Bound once at startup (deps.py) so cached projections fold in manual corrections."""
        self._layer_store = store

    # ------------------------------------------------------------------ lock helper
    def _key_lock(self, key):
        with self._glock:
            lk = self._klocks.get(key)
            if lk is None:
                lk = self._klocks[key] = threading.Lock()
            return lk

    @staticmethod
    def _lru_put(od, key, val, cap):
        od[key] = val
        od.move_to_end(key)
        while len(od) > cap:
            od.popitem(last=False)

    # ------------------------------------------------------------------ E2E
    def open_e2e(self, path):
        eid = ids.e2e_id(path)
        with self._key_lock(("e2e", eid)):
            with self._glock:
                raw = self._raw.get(eid)
            if raw is None:
                raw = e2e_source.open_e2e(path)
                with self._glock:
                    self._lru_put(self._raw, eid, raw, 4)
            else:
                with self._glock:
                    self._raw.move_to_end(eid)
            return raw

    def get_raw(self, eid):
        with self._glock:
            raw = self._raw.get(eid)
            if raw is not None:
                self._raw.move_to_end(eid)
            return raw

    # ------------------------------------------------------------------ volume
    def get_volume(self, volume_id):
        eid, idx = ids.parse_volume_id(volume_id)
        with self._key_lock(("vol", volume_id)):
            with self._glock:
                ov = self._vol.get(volume_id)
            if ov is not None:
                with self._glock:
                    self._vol.move_to_end(volume_id)
                return ov
            raw = self.get_raw(eid)
            if raw is None:
                raise KeyError(f"E2E {eid} not open")
            ov = e2e_source.load_volume(raw, idx)
            with self._glock:
                self._lru_put(self._vol, volume_id, ov, 6)
            return ov

    # ------------------------------------------------------------------ native map (cached, patchable)
    def get_native(self, volume_id, feature):
        """Pre-finish native (n,W) map for a SINGLE_NATIVE feature, cached so a live edit preview can
        patch just the edited B-scan's row instead of re-summing the whole volume."""
        key = (volume_id, feature)
        with self._key_lock(("nat", key)):
            with self._glock:
                nat = self._nat.get(key)
            if nat is not None:
                with self._glock:
                    self._nat.move_to_end(key)
                return nat
            ov = self.get_volume(volume_id)
            surf = layers.effective_surfaces(ov, self._layer_store)
            nat = projection.native_full(ov, feature, surf)
            with self._glock:
                self._lru_put(self._nat, key, nat, 8)
            return nat

    # ------------------------------------------------------------------ projection (cached float frame)
    def get_projection(self, volume_id, feature):
        key = (volume_id, feature)
        with self._key_lock(("proj", key)):
            with self._glock:
                m = self._proj.get(key)
            if m is not None:
                with self._glock:
                    self._proj.move_to_end(key)
                return m
            ov = self.get_volume(volume_id)
            if feature in projection.SINGLE_NATIVE:
                m = projection.finish(ov, feature, self.get_native(volume_id, feature))
            else:
                surf = layers.effective_surfaces(ov, self._layer_store)   # folds in corrections
                m = projection.enface(ov, feature, surf)
            with self._glock:
                self._lru_put(self._proj, key, m, 12)
            return m

    def invalidate_projection(self, volume_id):
        """Drop all cached projection + native frames for a volume (call after a correction lands so
        the next request recomputes from the corrected surfaces)."""
        with self._glock:
            for od in (self._proj, self._nat):
                for key in [k for k in od if k[0] == volume_id]:
                    del od[key]

    def loaded_e2e(self):
        with self._glock:
            return list(self._raw.keys())
