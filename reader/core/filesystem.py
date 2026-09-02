"""Server-side directory browsing for the 'open E2E by path' flow.

The app is local and bound to 127.0.0.1, but we still confine browsing to a configurable root
(default: the repo's data/ dir) so a stray request can't walk the whole disk. The user can also
paste an absolute path directly (validated to live under the root, or the root can be widened).
"""
import os

from . import REPO_ROOT

# Default browse root = the repo data dir; override via reader.api config if needed.
DEFAULT_ROOT = os.path.join(REPO_ROOT, "data")


def _under(root, path):
    root = os.path.normcase(os.path.abspath(root))
    path = os.path.normcase(os.path.abspath(path))
    return path == root or path.startswith(root + os.sep)


def list_dir(dirpath, root=DEFAULT_ROOT):
    """List subdirectories and .E2E files under dirpath (confined to root)."""
    target = os.path.abspath(dirpath or root)
    if not _under(root, target):
        target = os.path.abspath(root)
    if not os.path.isdir(target):
        target = os.path.abspath(root)

    entries = []
    try:
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            try:
                if os.path.isdir(full):
                    entries.append({"name": name, "type": "dir", "path": full})
                elif name.lower().endswith(".e2e"):
                    entries.append({"name": name, "type": "e2e", "path": full,
                                    "size_mb": round(os.path.getsize(full) / 1e6, 1)})
            except OSError:
                continue
    except OSError:
        pass

    parent = os.path.dirname(target)
    if not _under(root, parent):
        parent = None
    return {"dir": target, "parent": parent, "root": os.path.abspath(root), "entries": entries}


def is_e2e(path, root=DEFAULT_ROOT):
    p = os.path.abspath(path or "")
    return bool(p) and _under(root, p) and os.path.isfile(p) and p.lower().endswith(".e2e")
