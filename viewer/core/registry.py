"""Persistent registry of scans loaded into the doctor viewer, so a patient can be tracked for GA
over time.

The viewer's upload path processes a new E2E into an in-memory LRU (max 4) that vanishes on restart —
nothing is recorded, so a patient's GA cannot be followed across scans. This module persists one CSV
row per loaded scan, keyed by (patient, eye, acquisition), so re-loading the same scan UPDATES its row
rather than duplicating it. The Patients tab reads it back, groups by patient, and shows GA over time.

Robust to the classic Windows gotcha the user called out: the doctor has registry.csv open in Excel
(or OneDrive is mid-sync), which holds the file, so writing raises PermissionError. We catch that and
raise RegistryLocked with a plain-language message; callers surface it as a soft warning and never lose
the loaded scan.
"""
import csv
import hashlib
import os
import tempfile
import threading
from datetime import datetime

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store")
_CSV = os.path.join(_DIR, "registry.csv")
_LOCK = threading.RLock()

# CSV columns (stable order). Re-reading is tolerant of older/missing columns (DictReader + defaults).
FIELDS = ["record_id", "logged_at", "source", "patient_id", "eye", "visit", "acq_date",
          "n_bscans", "fov_w_mm", "fov_h_mm", "ga_area_mm2", "bm_source", "status", "note",
          "e2e_path", "vid"]


class RegistryError(Exception):
    """Base class for registry problems."""


class RegistryLocked(RegistryError):
    """registry.csv could not be read/written because another program holds it (Excel / OneDrive)."""

    def __init__(self, path=_CSV):
        super().__init__("The tracking log (registry.csv) is open in another program (e.g. Excel) or is "
                         "being synced. Close it and try again.")
        self.path = path


def csv_path():
    return _CSV


# --------------------------------------------------------------------------- helpers
def is_lock_error(exc):
    """True if `exc` is the OS telling us the file is held by another process (cross-Windows-version)."""
    if isinstance(exc, PermissionError):
        return True
    msg = str(exc).lower()
    return ("being used by another process" in msg or "permission denied" in msg
            or "sharing violation" in msg or "access is denied" in msg)


def record_id_for(patient_id, eye, acq_date, vid=""):
    """Stable, URL-safe id for a loaded scan — so re-loading the same scan upserts, never dupes.
    `vid` (the per-volume id) disambiguates scans whose patient/eye/date metadata is empty: without it,
    two different date-less E2Es would hash identically and the second would silently overwrite the
    first, losing a tracked scan. vid is deterministic per (file, volume), so re-loading still upserts."""
    key = (f"{(patient_id or '').strip()}|{(eye or '').strip()}|{(acq_date or '').strip()}"
           f"|{(vid or '').strip()}").lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _parse_dt(s):
    """Best-effort parse of a Spectralis acq_date ('03-Apr-2022 09:14:02') into a datetime, else None."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _enrich(row):
    """Add a sortable ISO timestamp (acq_iso) the browser can order by; acq_date strings it cannot."""
    dt = _parse_dt(row.get("acq_date"))
    row["acq_iso"] = dt.isoformat() if dt else None
    return row


# --------------------------------------------------------------------------- read
def _read_raw():
    # Open directly (no exists() pre-check): a missing file -> FileNotFoundError -> []. Pre-checking
    # then opening is a TOCTOU race — the file can be deleted (OneDrive, manual) in between, which would
    # else surface as an unhandled 500 instead of "no log yet".
    try:
        with open(_CSV, newline="", encoding="utf-8") as f:
            return [{k: (r.get(k, "") or "") for k in FIELDS} for r in csv.DictReader(f)]
    except FileNotFoundError:
        return []
    except OSError as e:
        if is_lock_error(e):
            raise RegistryLocked() from e
        raise


def all_records():
    """Every recorded scan (enriched with acq_iso). Raises RegistryLocked if the file is held."""
    with _LOCK:
        return [_enrich(r) for r in _read_raw()]


def csv_bytes():
    """Raw file bytes for download, or None if no log exists yet. Raises RegistryLocked if held.
    Holds _LOCK (like all_records) so a download can't observe a half-superseded snapshot mid-write."""
    with _LOCK:
        try:
            with open(_CSV, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            if is_lock_error(e):
                raise RegistryLocked() from e
            raise


# --------------------------------------------------------------------------- write (atomic)
def _write_all(rows):
    """Atomically replace registry.csv (write a temp sibling, then os.replace). os.replace over a file
    held by Excel/OneDrive raises PermissionError -> RegistryLocked, leaving the existing file intact."""
    try:                                                       # dir create/temp open: map a held/locked
        os.makedirs(_DIR, exist_ok=True)                       # data_store to RegistryLocked too, so the
        fd, tmp = tempfile.mkstemp(dir=_DIR, prefix=".registry-", suffix=".tmp")  # message stays accurate
    except OSError as e:
        if is_lock_error(e):
            raise RegistryLocked() from e
        raise
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        os.replace(tmp, _CSV)                                   # atomic on the same volume
    except OSError as e:
        _safe_unlink(tmp)
        if is_lock_error(e):
            raise RegistryLocked() from e
        raise
    except Exception:
        _safe_unlink(tmp)
        raise


def _safe_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def record(entry):
    """Upsert one scan by record_id (preserves position; refreshes logged_at). Returns the stored row.
    Raises RegistryLocked if the file is held."""
    rid = entry.get("record_id") or record_id_for(entry.get("patient_id"), entry.get("eye"),
                                                   entry.get("acq_date"), entry.get("vid"))
    row = {k: ("" if entry.get(k) is None else entry.get(k, "")) for k in FIELDS}
    row["record_id"] = rid
    row["logged_at"] = datetime.now().isoformat(timespec="seconds")
    with _LOCK:
        rows = _read_raw()
        for i, r in enumerate(rows):
            if r.get("record_id") == rid:
                rows[i] = row
                break
        else:
            rows.append(row)
        _write_all(rows)
    return _enrich(dict(row))


def delete(record_id):
    """Remove one record (the E2E file is untouched). Returns True if a row was removed."""
    with _LOCK:
        rows = _read_raw()
        kept = [r for r in rows if r.get("record_id") != record_id]
        if len(kept) == len(rows):
            return False
        _write_all(kept)
        return True


# --------------------------------------------------------------------------- viewer glue
def build_entry(meta, e2e_path, vid, source="upload", status="ok", note=""):
    """Map an upload/viewmodel `meta` dict (+ its E2E path & vid) onto a registry row."""
    fov = meta.get("fov_mm") or [None, None]
    pid, eye, acq = meta.get("patient_id"), meta.get("eye"), meta.get("acq_date")
    return {
        "record_id": record_id_for(pid, eye, acq, vid),
        "source": source, "patient_id": pid or "", "eye": eye or "", "visit": meta.get("visit") or "",
        "acq_date": acq or "", "n_bscans": meta.get("n_bscans"),
        "fov_w_mm": (round(float(fov[0]), 3) if fov and fov[0] is not None else ""),
        "fov_h_mm": (round(float(fov[1]), 3) if fov and len(fov) > 1 and fov[1] is not None else ""),
        "ga_area_mm2": meta.get("oac_area_mm2"), "bm_source": meta.get("bm_source") or "",
        "status": status, "note": note, "e2e_path": e2e_path or "", "vid": vid or "",
    }


def try_record_upload(meta, e2e_path, vid):
    """Record a freshly-loaded upload, swallowing lock/IO errors into a soft result so the scan still
    opens. Returns {saved, warning, record_id}."""
    try:
        row = record(build_entry(meta, e2e_path, vid))
        return {"saved": True, "warning": None, "record_id": row["record_id"]}
    except RegistryLocked as e:
        return {"saved": False, "warning": str(e), "record_id": None}
    except Exception as e:                                      # noqa: BLE001  (never break an upload)
        return {"saved": False, "warning": f"Could not update the tracking log: {e}", "record_id": None}
