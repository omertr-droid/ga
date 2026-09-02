"""Build a self-contained, OFFLINE, double-clickable macOS (Apple Silicon) folder for the doctor viewer.

Companion to package_app.py (Windows). It is CALLED from package_app.main() so a normal build produces
BOTH packages; run it on the Windows dev machine (oct_env, Python 3.11). It assembles, under dist/
(or $OCT_VIEWER_DIST):

  oct_ga_viewer_mac/
    runtime/python.tar.gz   <- standalone arm64 CPython 3.11 (python-build-standalone), kept UNEXTRACTED
                               so Windows never touches its unix symlinks; the Mac unpacks it at first run
    wheels/                 <- vendored arm64 cp311 wheels (the offline library-serving closure)
    app/                    <- viewer/ + reader/(core) + src/  + the baked library (reuses copy_app)
    requirements.txt        <- pinned top-level runtime set
    run.command             <- LF + exec bit: unpacks python, offline-installs the wheels (first run), launches
    READ_ME_FIRST.txt
  oct_ga_viewer_mac.tar.gz  <- the shippable archive (tar preserves run.command's executable bit)

The Mac owner: double-click the tar.gz to unpack, clear quarantine once
(`xattr -dr com.apple.quarantine <folder>`), then double-click run.command. Fully offline.

LIBRARY-ONLY: shows the eyes that ship with it; new-E2E upload is not supported, so the heavy scientific
stack (scipy/scikit-image/matplotlib/h5py/imagecodecs/eyepy/oct_converter) is NOT vendored — only
numpy + opencv + the web server. arm64 by default; pass --arch x86_64-apple-darwin for an Intel Mac.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

from package_app import DIST, copy_app, log

MAC_OUT = os.path.join(DIST, "oct_ga_viewer_mac")
WORK = os.path.join(DIST, "_pbs_mac")          # cache for the downloaded standalone-python tarball
PYVER = "3.11"
PORT = 8011

# Library-serving runtime closure (pinned to oct_env). pip resolves the transitive deps
# (pydantic, pydantic-core, starlette, anyio, sniffio, click, h11, annotated-types, idna, …).
REQS = [
    "numpy==2.4.6",
    "opencv-python-headless==4.13.0.92",   # headless = no GUI dylibs; `import cv2` is identical
    "fastapi==0.136.3",
    "uvicorn==0.49.0",
]

# python-build-standalone: resolve the latest release's install_only asset for the arch. The hardcoded
# fallback is used only if the GitHub API is unreachable; override either with --python <url-or-tarball>.
PBS_API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
PBS_FALLBACK = {
    "aarch64-apple-darwin":
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20250818/cpython-3.11.13+20250818-aarch64-apple-darwin-install_only.tar.gz",
    "x86_64-apple-darwin":
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20250818/cpython-3.11.13+20250818-x86_64-apple-darwin-install_only.tar.gz",
}


def _platform_tags(arch):
    """macOS wheel platform tags to accept (a range absorbs per-package minimum-OS differences)."""
    suffix = "arm64" if "aarch64" in arch else "x86_64"
    tags = ["macosx_10_9_x86_64"] if suffix == "x86_64" else []
    floor = 11 if suffix == "arm64" else 10
    for major in range(floor, 16):
        tags.append(f"macosx_{major}_0_{suffix}")
    return tags


def resolve_pbs_url(arch):
    """Pick the latest python-build-standalone 3.11 install_only tarball for arch (else the fallback)."""
    try:
        req = urllib.request.Request(PBS_API, headers={"User-Agent": "oct_ga-packager"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rel = json.load(r)
        cands = []
        for a in rel.get("assets", []):
            n = a.get("name", "")
            if n.startswith(f"cpython-{PYVER}.") and arch in n and n.endswith("-install_only.tar.gz"):
                cands.append((n, a["browser_download_url"]))
        if cands:
            cands.sort()                       # filename sort ⇒ highest 3.11.x last
            log(f"  resolved standalone python: {cands[-1][0]}")
            return cands[-1][1]
        log("  (no matching install_only asset in latest release; using fallback URL)")
    except Exception as e:                     # noqa: BLE001
        log(f"  (GitHub API lookup failed: {e}; using fallback URL)")
    return PBS_FALLBACK[arch]


def ensure_python_tarball(python_arg, arch):
    """Return a local path to the standalone-python install_only .tar.gz (download + cache as needed)."""
    if python_arg and os.path.isfile(python_arg):
        return python_arg
    os.makedirs(WORK, exist_ok=True)
    dst = os.path.join(WORK, f"python-{arch}.tar.gz")
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        log(f"  reusing cached {dst}")
        return dst
    url = python_arg if (python_arg and python_arg.startswith("http")) else resolve_pbs_url(arch)
    log(f"  downloading {url} …")
    urllib.request.urlretrieve(url, dst)
    return dst


def vendor_wheels(wheels_dir, arch, skip=False):
    """pip download the arm64 cp311 wheel closure (offline target) into wheels_dir."""
    if skip:
        log("(--no-site-packages) skipping wheel vendoring")
        return
    if os.path.isdir(wheels_dir):
        shutil.rmtree(wheels_dir)
    os.makedirs(wheels_dir)
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
           "--python-version", PYVER, "--implementation", "cp", "--abi", "cp311", "-d", wheels_dir]
    for tag in _platform_tags(arch):
        cmd += ["--platform", tag]
    cmd += REQS
    log("vendoring macOS wheels (pip download) …")
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("pip download failed — the offline wheel closure is incomplete; aborting.")
    whls = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")]
    sz = sum(os.path.getsize(os.path.join(wheels_dir, f)) for f in whls)
    log(f"  vendored {len(whls)} wheels ({sz / 1e6:.0f} MB)")


RUN_COMMAND = f"""#!/bin/bash
# GA Viewer - offline macOS launcher (double-click). Keep this window open while using the viewer.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OCT_VIEWER_LIBRARY_ONLY=1
PY="$DIR/runtime/python/bin/python3"
LIBS="$DIR/libs"

if [ ! -x "$PY" ]; then
  echo "GA Viewer - unpacking Python (first run only)..."
  tar -xzf "$DIR/runtime/python.tar.gz" -C "$DIR/runtime" || {{ echo "Unpack failed."; read -r _; exit 1; }}
fi
if [ ! -d "$LIBS" ]; then
  echo "GA Viewer - first-time setup (about a minute, no internet needed)..."
  "$PY" -m pip install --no-index --find-links "$DIR/wheels" --target "$LIBS" -r "$DIR/requirements.txt" \\
    || {{ echo "Setup failed (see the messages above)."; read -r _; exit 1; }}
fi
export PYTHONPATH="$LIBS:$DIR/app"
echo "GA Viewer starting - your browser opens when ready."
( for i in $(seq 1 150); do
    curl -s -o /dev/null "http://127.0.0.1:{PORT}/api/health" && {{ open "http://127.0.0.1:{PORT}/"; break; }}
    sleep 0.8
  done ) &
exec "$PY" -m uvicorn viewer.api.app:app --host 127.0.0.1 --port {PORT}
"""

READ_ME = f"""GA Viewer for macOS (Apple Silicon) — offline
=============================================

1. Double-click  oct_ga_viewer_mac.tar.gz  to unpack it (creates the  oct_ga_viewer_mac  folder).

2. Clear the download quarantine ONCE so macOS will run the bundled files:
     - Open Terminal (Spotlight -> "Terminal").
     - Type:   xattr -dr com.apple.quarantine     (note the trailing space)
     - Drag the unpacked  oct_ga_viewer_mac  folder onto the Terminal window, then press Return.

3. Double-click  run.command  inside the folder.
     - First launch only: it unpacks Python and installs its libraries (about a minute, NO internet
       needed), then your browser opens at  http://127.0.0.1:{PORT}/ .
     - If macOS still blocks it: right-click run.command -> Open -> Open (once).

4. Click a scan in the Library to open the 3-panel viewer. To stop, close the Terminal window.

Everything runs locally and offline; nothing is installed system-wide. This build is LIBRARY-ONLY:
it shows the scans that ship with it; loading your own .E2E files is not available in this version.
"""


def _tar_filter(ti):
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    if ti.isdir():
        ti.mode = 0o755
    elif ti.name.replace("\\", "/").endswith("/run.command"):
        ti.mode = 0o755                        # the one executable in the outer archive
    else:
        ti.mode = 0o644
    return ti


def make_tar():
    tar_path = MAC_OUT + ".tar.gz"
    log("creating oct_ga_viewer_mac.tar.gz …")
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(MAC_OUT, arcname="oct_ga_viewer_mac", filter=_tar_filter)
    return tar_path


def _write_launcher_and_docs():
    with open(os.path.join(MAC_OUT, "run.command"), "w", newline="\n", encoding="utf-8") as f:
        f.write(RUN_COMMAND)
    try:
        os.chmod(os.path.join(MAC_OUT, "run.command"), 0o755)   # ignored on Windows; the tar sets it
    except OSError:
        pass
    with open(os.path.join(MAC_OUT, "requirements.txt"), "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(REQS) + "\n")
    with open(os.path.join(MAC_OUT, "READ_ME_FIRST.txt"), "w", newline="\n", encoding="utf-8") as f:
        f.write(READ_ME)


def build_mac(args):
    refresh = getattr(args, "refresh_app", False)
    skip_wheels = getattr(args, "no_site_packages", False)
    python_arg = getattr(args, "python", None)
    arch = getattr(args, "arch", "aarch64-apple-darwin")

    runtime = os.path.join(MAC_OUT, "runtime")
    if refresh:
        if not os.path.isdir(MAC_OUT):
            raise SystemExit("--refresh-app needs an existing mac build; run a full build first")
        log("mac refresh-app: refreshing app code + launcher only …")
        copy_app(os.path.join(MAC_OUT, "app"))
        _write_launcher_and_docs()
    else:
        if os.path.isdir(MAC_OUT):
            shutil.rmtree(MAC_OUT)
        os.makedirs(runtime)
        log(f"mac: resolving standalone python ({arch}) …")
        tarball = ensure_python_tarball(python_arg, arch)
        shutil.copyfile(tarball, os.path.join(runtime, "python.tar.gz"))
        vendor_wheels(os.path.join(MAC_OUT, "wheels"), arch, skip=skip_wheels)
        log("mac: copying app code + baked library …")
        copy_app(os.path.join(MAC_OUT, "app"))
        _write_launcher_and_docs()

    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(MAC_OUT) for f in fs)
    log(f"built {MAC_OUT}  ({sz / 1e6:.0f} MB)")
    if not skip_wheels:
        log("created " + make_tar())
    log("mac: done. Ship oct_ga_viewer_mac.tar.gz; the Mac owner follows READ_ME_FIRST.txt.")
