"""A minimal server-side file browser for the upload 'Browse…' button.

The clinic runs on the same machine as the browser, but a web page cannot read a file's absolute path
from the OS open-dialog (browsers hide it), and the offline package's embeddable Python has no tkinter
for a native dialog. So the reliable, package-safe way to let the user pick an .E2E is to list the
server's own filesystem here and navigate it in the UI — which yields the real absolute path the
pipeline needs, and works identically in dev and in the offline package.

This intentionally exposes directory listings to the local UI. That is acceptable for a single-user app
bound to 127.0.0.1: the user could already paste any absolute path. Only directories and ``.E2E`` files
are surfaced; permission errors are skipped, never raised.
"""
import os
import string


def _drives():
    """Existing filesystem roots, per OS: Windows drive letters; ``/`` (+ ``/Volumes`` for external
    disks) on macOS/Linux. Cross-platform so the same browser UI works on every shipping target."""
    if os.name == "nt":
        out = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
        return out or ["C:\\"]
    roots = ["/"]
    if os.path.isdir("/Volumes"):                       # macOS mounts external/network disks here
        roots.append("/Volumes")
    return roots


def list_dir(path=None) -> dict:
    """List the sub-folders and ``.E2E`` files of ``path`` (defaulting to the user's home).

    Returns ``{path, parent, dirs:[{name,path}], files:[{name,path,size}], roots:[...], home, not_found}``.
    If ``path`` is a file, its containing folder is listed (so pasting a file path opens its folder); an
    invalid/unreadable path falls back to home with ``not_found=True`` so the UI can say so.
    """
    home = os.path.expanduser("~")
    not_found = False
    if path:
        ap = os.path.abspath(os.path.expanduser(path.strip()))
        if os.path.isfile(ap):
            ap = os.path.dirname(ap)
        if os.path.isdir(ap):
            path = ap
        else:
            not_found = True
            path = home
    else:
        path = home

    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.name.startswith("$"):          # skip $Recycle.Bin and similar system entries
                        continue
                    if e.is_dir():
                        dirs.append({"name": e.name, "path": e.path})
                    elif e.is_file() and e.name.lower().endswith(".e2e"):
                        files.append({"name": e.name, "path": e.path, "size": e.stat().st_size})
                except OSError:
                    continue                            # unreadable entry — skip it
    except OSError:
        pass                                            # unreadable directory — return an empty (valid) listing

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())
    parent = os.path.dirname(path)
    if parent == path:                                  # at a filesystem root
        parent = None
    return {"path": path, "parent": parent, "dirs": dirs, "files": files,
            "roots": _drives(), "home": home, "not_found": not_found}
