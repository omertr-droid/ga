"""Stable, opaque ids for an opened E2E and a volume within it.

An E2E is opened by absolute path; we key the in-memory session by a short stable hash of that path
(so the same file reopened yields the same id) and a volume by (e2e_id, volume_index).
"""
import hashlib
import os


def e2e_id(path):
    """Short stable id for an E2E file path (case-insensitive on Windows)."""
    norm = os.path.normcase(os.path.abspath(path))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def volume_id(eid, index):
    return f"{eid}.{int(index)}"


def parse_volume_id(vid):
    eid, idx = vid.rsplit(".", 1)
    return eid, int(idx)
