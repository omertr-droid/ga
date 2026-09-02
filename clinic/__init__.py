"""GA Clinic — a lean, standalone OCT Geographic-Atrophy reader.

A third app (sibling of ``reader/`` and ``viewer/``) built around a patient database rather than a
card grid, with PLEX removed entirely. It reuses the project's GA pipeline as the single source of
truth: every area comes from ``viewer.core.viewmodel.compute(..., baseline="radial2")`` — the exact
call the proven viewer upload path uses — so a clinic scan reports the same number as the viewer/CLI.

Run (dev, from the repo root)::

    oct_env\\Scripts\\python.exe -m uvicorn clinic.api.app:app --host 127.0.0.1 --port 8021

See ``clinic/CLINIC.md`` for the full architecture + operator documentation.
"""
