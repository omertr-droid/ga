#!/usr/bin/env python
r"""Packaged GA Clinic real-E2E smoke test with lightweight RAM diagnostics.

Example (from a built Windows folder)::

    python\python.exe app\src\smoke_clinic.py scan.E2E --index 1 --expected 0.1

The test verifies that listing scans does not eagerly load ONNX, processes one chosen eye through the
real clinic choke point, keeps only compact live results, and releases the raw decoder/model at batch end.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import sys
import time


def _windows_memory():
    if sys.platform != "win32":
        return {}

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    ok = get_memory(get_process(), ctypes.byref(c), c.cb)
    if not ok:
        return {}
    mb = 1024.0 * 1024.0
    return {"rss_mb": round(c.WorkingSetSize / mb, 1), "peak_rss_mb": round(c.PeakWorkingSetSize / mb, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e2e")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--expected", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=0.001)
    args = ap.parse_args()

    # Import after the launcher/test harness has set OCT_CLINIC_DATA.
    from clinic.core.store import ClinicStore
    import bm_dl                                                # clinic.core puts packaged app/src on sys.path

    out = {"start": _windows_memory(), "e2e": os.path.abspath(args.e2e)}
    store = ClinicStore()
    t0 = time.perf_counter()
    listing = store.list_scans(args.e2e)
    out["open_seconds"] = round(time.perf_counter() - t0, 2)
    out["after_open"] = _windows_memory()
    out["onnx_loaded_after_listing"] = bm_dl._session is not None
    scans = listing.get("scans") or []
    if not scans:
        raise SystemExit("no measurable scan")
    chosen = next((s for s in scans if s["index"] == args.index), scans[0])

    t0 = time.perf_counter()
    vid, meta, db_result, warning = store.process(
        args.e2e, chosen["index"], "dl" if listing.get("dl_available") else "auto")
    out["process_seconds"] = round(time.perf_counter() - t0, 2)
    out["after_process"] = _windows_memory()
    out["index"] = chosen["index"]
    out["eye"] = chosen.get("eye")
    out["vid"] = vid
    out["area_mm2"] = meta.get("oac_area_mm2")
    out["bm_source"] = meta.get("bm_source")
    out["warning"] = warning
    out["db_saved"] = bool(db_result.get("saved"))
    out["raw_cached_before_finish"] = len(store._raw)
    out["live_cached_before_finish"] = len(store._live)
    out["onnx_loaded_before_finish"] = bm_dl._session is not None

    out["finish"] = store.finish_batch(args.e2e)
    out["after_finish"] = _windows_memory()
    out["raw_cached_after_finish"] = len(store._raw)
    out["live_cached_after_finish"] = len(store._live)
    out["onnx_loaded_after_finish"] = bm_dl._session is not None

    if out["onnx_loaded_after_listing"]:
        raise AssertionError("listing/config eagerly loaded ONNX")
    if out["raw_cached_after_finish"] != 0 or out["live_cached_after_finish"] != 1:
        raise AssertionError("unexpected cache lifecycle")
    if out["onnx_loaded_after_finish"]:
        raise AssertionError("ONNX session was retained after finish")
    if args.expected is not None and abs(float(out["area_mm2"]) - args.expected) > args.tolerance:
        raise AssertionError(f"area {out['area_mm2']} differs from expected {args.expected}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
