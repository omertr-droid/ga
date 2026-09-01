"""Persist a browser-dropped .E2E to a real file, so the path-based upload flow can consume it.

A drag-and-drop hands the browser the file's BYTES and its display name — never its absolute path
(``File.path`` is an Electron extension, not a web API). But every step downstream of here is
path-based: ``/upload/open`` and ``/upload/process`` gate on ``os.path.exists``, and ``db`` stores an
``e2e_path`` so ``/reopen`` can re-process a row later. So the bytes must land on disk first, and the
landing must PERSIST: for a dropped file this copy is the only server-side original that exists.

Two design points, both load-bearing:

* **Content-addressed name.** ``reader.core.ids.e2e_id`` derives the eid from the *path*
  (``sha1(normcase(abspath(path)))``), and ``vid`` -> ``record_id`` derive from the eid. Staging under a
  random temp name would therefore mint a fresh identity on every drop and append a DUPLICATE patient
  row for the same physical scan. Naming the file by its own sha256 makes the path a pure function of
  the content, so re-dropping the same file resolves to the same eid and ``db.record`` upserts.
* **Raw bytes, not multipart.** FastAPI's ``UploadFile``/``File()``/``Form()`` require
  ``python-multipart``, which is not installed (Starlette asserts on it at request time). Consuming
  ``request.stream()`` needs nothing extra and never buffers the (~300 MB) file in memory.

The canonical ``<sha256>.E2E`` only appears after the whole stream is written and validated
(temp sibling -> ``os.replace``, the same atomic idiom as ``db._write_all``), so a cancelled or crashed
upload can never leave a truncated file where something would later try to open one.
"""
import hashlib
import os
import tempfile

import anyio

from . import db

UPLOADS = os.path.join(db._DIR, "uploads")

# Guard against a runaway/hostile body: neither Starlette nor uvicorn caps the request size. The
# largest E2E observed in this project is ~305 MB; 600 MB leaves generous headroom.
MAX_STAGE_BYTES = 600 * 1024 * 1024

# Every Heidelberg E2E begins with this 4-byte ASCII header (verified against the cohort files).
E2E_MAGIC = b"CMDb"


class StageError(Exception):
    """A staging failure that maps cleanly onto an HTTP status."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _safe_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def staged_dir() -> str:
    return UPLOADS


async def stage_stream(request) -> dict:
    """Stream ``request``'s raw body to ``uploads/<sha256>.E2E``; return ``{path, size, dup}``.

    Rejects an empty body (400), an over-large one (413), and anything whose first four bytes are not
    the E2E magic (415) — the last as soon as those bytes arrive, so a wrong file is refused without
    writing hundreds of megabytes. A held/syncing uploads directory surfaces as 423, matching the
    lock handling everywhere else in the app.
    """
    try:
        os.makedirs(UPLOADS, exist_ok=True)
    except OSError as e:
        raise _os_error(e)

    try:
        fd, tmp = tempfile.mkstemp(dir=UPLOADS, prefix=".stage-", suffix=".tmp")
    except OSError as e:
        raise _os_error(e)

    h = hashlib.sha256()
    head = b""
    total = 0
    try:
        # wrap_file offloads each write to a worker thread: this handler must be async to consume
        # request.stream(), and a synchronous ~300 MB write loop would stall the event loop.
        async with anyio.wrap_file(os.fdopen(fd, "wb")) as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                if len(head) < 4:                              # fail fast on a non-E2E
                    head += bytes(chunk[: 4 - len(head)])
                    if len(head) == 4 and head != E2E_MAGIC:
                        raise StageError(415, "That file is not a Spectralis .E2E (bad header).")
                total += len(chunk)
                if total > MAX_STAGE_BYTES:
                    raise StageError(413, "That file is larger than the "
                                          f"{MAX_STAGE_BYTES // (1024 * 1024)} MB upload limit.")
                h.update(chunk)
                await f.write(chunk)

        if total == 0:
            raise StageError(400, "That file is empty.")
        if head != E2E_MAGIC:                                  # shorter than 4 bytes
            raise StageError(415, "That file is not a Spectralis .E2E (bad header).")

        target = os.path.join(UPLOADS, h.hexdigest() + ".E2E")
        if os.path.exists(target):                             # identical content already staged
            _safe_unlink(tmp)
            return {"path": target, "size": total, "dup": True}
        os.replace(tmp, target)                                # atomic: same directory, same volume
        return {"path": target, "size": total, "dup": False}

    except StageError:
        _safe_unlink(tmp)
        raise
    except OSError as e:
        _safe_unlink(tmp)
        raise _os_error(e)
    except BaseException:                                      # ClientDisconnect on abort, etc.
        _safe_unlink(tmp)
        raise


def _os_error(e):
    if db.is_lock_error(e):
        return StageError(423, "The uploads folder is locked or being synced; try again in a moment.")
    return StageError(500, f"Could not save the dropped file: {e}")
