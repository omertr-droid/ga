"""viewer — a standalone, read-only, doctor-facing OCT GA viewer.

A simple "1 menu, 3 screens" sibling of `reader/`. It reuses `reader.core.*` (the GA/projection
math has a single source of truth) and serves pre-baked per-eye bundles so library scans open with
no E2E decode. See the plan and `viewer/READER_DOCTOR.md` (if present) for the design.
"""
