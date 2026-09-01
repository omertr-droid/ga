"""The clinic's standalone patient database — one CSV row per scan/visit, upserted.

Adapted from ``viewer/core/registry.py`` (same atomic-write + Windows file-lock handling, credited
here), with three additions the clinic needs:
  * ``patient_name``  — so the database lists patients by name, not just id.
  * ``volume_index``  — the chosen volume's index in the E2E, so a row can be re-opened deterministically.
  * ``bm_choice``     — the user's BM choice (device/dl/auto), so a re-open reproduces the same number.

It is separate from the viewer's registry. Source runs default to ``clinic/data_store``; portable
packages set ``OCT_CLINIC_DATA`` to top-level ``user_data`` so app code and patient state stay separate.

Robust to the classic Windows gotcha: the file is open in Excel / mid-OneDrive-sync, so writing raises
``PermissionError``. We catch that and raise ``DbLocked``; callers surface it as a soft warning on
write (the scan still opens) and HTTP 423 on read/delete/export — the loaded scan is never lost.
"""
import csv
import hashlib
import os
import tempfile
import threading
from datetime import datetime

# A packaged prototype keeps its mutable state beside the launcher in ``user_data/``.  Source/dev runs
# retain the historical in-tree default.  Keeping this behind one env var makes the Windows and macOS
# folders portable (copy the folder = copy the database) without ever writing into packaged app code.
_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store")
_DIR = os.path.abspath(os.environ.get("OCT_CLINIC_DATA") or _DEFAULT_DIR)
_CSV = os.path.join(_DIR, "patients.csv")
_LOCK = threading.RLock()

# CSV columns (stable order). Re-reading tolerates older/missing columns (DictReader + defaults).
FIELDS = ["record_id", "logged_at", "source", "patient_id", "patient_name", "eye", "visit",
          "acq_date", "n_bscans", "fov_w_mm", "fov_h_mm", "ga_area_mm2", "bm_source",
          "volume_index", "bm_choice", "e2e_path", "vid", "status", "note"]


class DbError(Exception):
    """Base class for patient-database problems."""


class DbLocked(DbError):
    """patients.csv could not be read/written because another program holds it (Excel / OneDrive)."""

    def __init__(self, path=_CSV):
        super().__init__("The patient database (patients.csv) is open in another program (e.g. Excel) "
                         "or is being synced. Close it and try again.")
        self.path = path


def csv_path():
    return _CSV


# --------------------------------------------------------------------------- helpers
def is_lock_error(exc) -> bool:
    """True if ``exc`` is the OS telling us the file is held by another process (cross-Windows-version)."""
    if isinstance(exc, PermissionError):
        return True
    msg = str(exc).lower()
    return ("being used by another process" in msg or "permission denied" in msg
            or "sharing violation" in msg or "access is denied" in msg)


def record_id_for(patient_id, eye, acq_date, vid="") -> str:
    """Stable id for a scan so re-loading upserts, never dupes. ``vid`` (the per-volume id) disambiguates
    scans whose patient/eye/date metadata is empty: without it, two date-less E2Es would hash identically
    and the second would overwrite the first. vid is deterministic per (file, volume), so reloads upsert."""
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
    """Add a sortable ISO timestamp (acq_iso) the browser/Excel export can order by."""
    dt = _parse_dt(row.get("acq_date"))
    row["acq_iso"] = dt.isoformat() if dt else None
    row["staged_copy"] = _is_staged_path(row.get("e2e_path"))
    return row


def _is_staged_path(path) -> bool:
    """True only for a file inside this clinic data root's content-addressed uploads directory."""
    if not path:
        return False
    uploads = os.path.realpath(os.path.join(_DIR, "uploads"))
    candidate = os.path.realpath(os.path.abspath(str(path)))
    try:
        return os.path.commonpath([uploads, candidate]) == uploads
    except (OSError, ValueError):                            # different Windows drives / invalid path
        return False


# --------------------------------------------------------------------------- read
def _read_raw():
    # Open directly (no exists() pre-check): a missing file -> FileNotFoundError -> []. Pre-checking then
    # opening is a TOCTOU race — the file can be deleted (OneDrive, manual) in between, surfacing as an
    # unhandled 500 instead of "no database yet".
    try:
        with open(_CSV, newline="", encoding="utf-8") as f:
            return [{k: (r.get(k, "") or "") for k in FIELDS} for r in csv.DictReader(f)]
    except FileNotFoundError:
        return []
    except OSError as e:
        if is_lock_error(e):
            raise DbLocked() from e
        raise


def all_records():
    """Every recorded scan (enriched with acq_iso). Raises DbLocked if the file is held."""
    with _LOCK:
        return [_enrich(r) for r in _read_raw()]


def get(record_id):
    """One record by id (enriched), or None. Raises DbLocked if the file is held."""
    for r in all_records():
        if r.get("record_id") == record_id:
            return r
    return None


def csv_bytes():
    """Raw file bytes for download, or None if no database exists yet. Raises DbLocked if held.
    Holds _LOCK (like all_records) so a download can't observe a half-superseded snapshot mid-write."""
    with _LOCK:
        try:
            with open(_CSV, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            if is_lock_error(e):
                raise DbLocked() from e
            raise


# --------------------------------------------------------------------------- write (atomic)
def _safe_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _write_all(rows):
    """Atomically replace patients.csv (write a temp sibling, then os.replace). os.replace over a file
    held by Excel/OneDrive raises PermissionError -> DbLocked, leaving the existing file intact."""
    try:                                                       # dir create / temp open: map a held data_store
        os.makedirs(_DIR, exist_ok=True)                       # to DbLocked too, so the message stays accurate
        fd, tmp = tempfile.mkstemp(dir=_DIR, prefix=".patients-", suffix=".tmp")
    except OSError as e:
        if is_lock_error(e):
            raise DbLocked() from e
        raise
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        os.replace(tmp, _CSV)                                  # atomic on the same volume
    except OSError as e:
        _safe_unlink(tmp)
        if is_lock_error(e):
            raise DbLocked() from e
        raise
    except Exception:
        _safe_unlink(tmp)
        raise


def record(entry):
    """Upsert one scan by record_id (preserves row position; refreshes logged_at). Returns the stored
    row (enriched). Raises DbLocked if the file is held."""
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


def delete(record_id, delete_staged=False) -> dict | None:
    """Remove one record and optionally its now-unreferenced staged E2E copy.

    A browsed/original E2E is never deleted.  A staged copy is deleted only when the caller explicitly
    asks and no remaining database row references that same path (OD and OS commonly share one file).
    Returns a result dict, or ``None`` when no row matched.
    """
    with _LOCK:
        rows = _read_raw()
        removed = next((r for r in rows if r.get("record_id") == record_id), None)
        if removed is None:
            return None
        kept = [r for r in rows if r.get("record_id") != record_id]
        _write_all(kept)

        path = removed.get("e2e_path") or ""
        staged_orphan = bool(_is_staged_path(path) and not any(r.get("e2e_path") == path for r in kept))
        staged_deleted = False
        staged_warning = None
        if delete_staged and staged_orphan:
            try:
                os.unlink(path)
                staged_deleted = True
            except FileNotFoundError:
                staged_deleted = True                         # already gone = desired final state
            except OSError as e:
                staged_warning = f"The database row was removed, but the stored E2E copy could not be deleted: {e}"
        return {
            "ok": True,
            "staged_orphan": staged_orphan,
            "staged_deleted": staged_deleted,
            "warning": staged_warning,
        }


# --------------------------------------------------------------------------- pipeline glue
def build_entry(meta, e2e_path, vid, volume_index, bm_choice, source="upload", status="ok", note=""):
    """Map a processed-scan ``meta`` dict (+ its E2E path, vid, chosen volume index and BM choice) onto a
    database row. ``meta`` is the clinic meta built by ``pipeline.process`` (carries patient_name)."""
    fov = meta.get("fov_mm") or [None, None]
    pid, eye, acq = meta.get("patient_id"), meta.get("eye"), meta.get("acq_date")
    # Key on the BASE volume id (strip any "|dl" BM-variant suffix) so re-processing the SAME physical
    # scan with a different Bruch's-membrane choice UPSERTS one visit (last choice wins) rather than
    # adding a duplicate row at the same date/eye. The full vid (with suffix) is still stored + reused.
    key_vid = (vid or "").split("|", 1)[0]
    return {
        "record_id": record_id_for(pid, eye, acq, key_vid),
        "source": source, "patient_id": pid or "", "patient_name": meta.get("patient_name") or "",
        "eye": eye or "", "visit": meta.get("visit") or "", "acq_date": acq or "",
        "n_bscans": meta.get("n_bscans"),
        "fov_w_mm": (round(float(fov[0]), 3) if fov and fov[0] is not None else ""),
        "fov_h_mm": (round(float(fov[1]), 3) if fov and len(fov) > 1 and fov[1] is not None else ""),
        "ga_area_mm2": meta.get("oac_area_mm2"), "bm_source": meta.get("bm_source") or "",
        "volume_index": ("" if volume_index is None else int(volume_index)),
        "bm_choice": bm_choice or "", "e2e_path": e2e_path or "", "vid": vid or "",
        "status": status, "note": note,
    }


def try_record(meta, e2e_path, vid, volume_index, bm_choice):
    """Record a freshly-processed scan, swallowing lock/IO errors into a soft result so the scan still
    opens. Returns ``{saved, warning, record_id}``."""
    try:
        row = record(build_entry(meta, e2e_path, vid, volume_index, bm_choice))
        return {"saved": True, "warning": None, "record_id": row["record_id"]}
    except DbLocked as e:
        return {"saved": False, "warning": str(e), "record_id": None}
    except Exception as e:                                      # noqa: BLE001 (never break a scan)
        return {"saved": False, "warning": f"Could not update the patient database: {e}", "record_id": None}


# --------------------------------------------------------------------------- queries (Home + detail)
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def patients():
    """One entry per patient for the Home list: ``{patient_id, patient_name, n_visits, eyes,
    latest_date, latest_iso}`` — sorted by name then id. Raises DbLocked if the file is held."""
    by_pid = {}
    for r in all_records():
        pid = r.get("patient_id") or "unknown"
        by_pid.setdefault(pid, []).append(r)
    out = []
    for pid, rows in by_pid.items():
        name = next((r.get("patient_name") for r in rows if r.get("patient_name")), "") or ""
        eyes = sorted({(r.get("eye") or "").strip() for r in rows if (r.get("eye") or "").strip()})
        latest = max(rows, key=lambda r: (r.get("acq_iso") or "", r.get("logged_at") or ""))
        out.append({
            "patient_id": pid, "patient_name": name, "n_visits": len(rows), "eyes": eyes,
            "latest_date": (latest.get("acq_date") or "").split(" ")[0],
            "latest_iso": latest.get("acq_iso"),
        })
    out.sort(key=lambda p: ((p["patient_name"] or "~").lower(), p["patient_id"]))
    return out


def patient(patient_id):
    """A patient's visits for the detail screen: ``{patient_id, patient_name, visits:[row,…]}`` with
    rows sorted by acquisition date then eye. Returns None if the patient has no records."""
    rows = [r for r in all_records() if (r.get("patient_id") or "unknown") == patient_id]
    if not rows:
        return None
    name = next((r.get("patient_name") for r in rows if r.get("patient_name")), "") or ""
    rows.sort(key=lambda r: (r.get("acq_iso") or r.get("record_id") or "", r.get("eye") or ""))
    for r in rows:
        r["ga_area_mm2_num"] = _num(r.get("ga_area_mm2"))
    return {"patient_id": patient_id, "patient_name": name, "visits": rows}
