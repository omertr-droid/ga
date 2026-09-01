"""Build a self-contained, OFFLINE, double-clickable macOS folder for the GA Clinic.

Companion to ``package_clinic.py`` (Windows). Run it ON THE WINDOWS DEV MACHINE (oct_env, Python 3.11);
it cross-vendors macOS wheels with ``pip download --platform macosx_…`` and ships a standalone CPython
(python-build-standalone) that the Mac unpacks + offline-installs on first run. Mirrors the proven
``package_app_mac.py`` machinery, but FULL-STACK (the clinic needs the live-upload + DL path, so the whole
scientific stack is vendored, not just numpy+opencv) and launches ``clinic.api.app``.

Assembles, under dist/ (or $OCT_CLINIC_DIST):

  oct_ga_clinic_mac/
    runtime/python.tar.gz   <- standalone arm64/x86_64 CPython 3.11, kept UNEXTRACTED (Windows never
                               touches its unix symlinks); the Mac unpacks it at first run
    wheels/                 <- explicit cp311 clinic wheel set (requirements-clinic.txt, no optional extras)
    app/                    <- clinic/ + viewer/core + reader/core + src/ + outputs/bm_dl model (reuses copy_app)
    requirements.txt        <- the pinned runtime set actually vendored
    run.command             <- LF + exec bit: unpacks python, offline-installs the wheels (first run), launches
    READ_ME_FIRST.txt
  oct_ga_clinic_mac.tar.gz  <- the shippable archive (tar preserves run.command's executable bit)

The Mac owner: double-click the tar.gz, clear quarantine once
(`xattr -dr com.apple.quarantine <folder>`), then double-click run.command. Fully offline.

Usage (from repo root, on the Windows dev box):
  oct_env\\Scripts\\python.exe src\\package_clinic_mac.py                 # arm64 (Apple Silicon, default)
  oct_env\\Scripts\\python.exe src\\package_clinic_mac.py --arch x86_64-apple-darwin   # Intel Mac
  oct_env\\Scripts\\python.exe src\\package_clinic_mac.py --no-site-packages           # structural dry-run
  oct_env\\Scripts\\python.exe src\\package_clinic_mac.py --refresh-app                # refresh app+launcher only

NOTE: the wheel-vendoring step runs on the dev box and fails LOUDLY if any pinned package has no macOS
cp311 wheel — adjust that pin and re-run. Verify the finished bundle on an actual Mac before shipping.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

from package_clinic import DIST, PORT, copy_app, log

MAC_OUT = os.path.join(DIST, "oct_ga_clinic_mac")
WORK = os.path.join(DIST, "_pbs_clinic_mac")       # cache for the downloaded standalone-python tarball
PYVER = "3.11"

PBS_API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
PBS_FALLBACK = {
    "aarch64-apple-darwin":
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20250818/cpython-3.11.13+20250818-aarch64-apple-darwin-install_only.tar.gz",
    "x86_64-apple-darwin":
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20250818/cpython-3.11.13+20250818-x86_64-apple-darwin-install_only.tar.gz",
}


def _norm(name):
    return name.split("==")[0].strip().lower().replace("_", "-")


def mac_reqs():
    """Explicit closed macOS wheel set for the clinic (downloaded/installed with ``--no-deps``)."""
    reqs = []
    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "requirements-clinic.txt")
    with open(req_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


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
    import json
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
            cands.sort()
            log(f"  resolved standalone python: {cands[-1][0]}")
            return cands[-1][1]
    except Exception as e:                                 # noqa: BLE001
        log(f"  (GitHub API lookup failed: {e}; using fallback URL)")
    return PBS_FALLBACK[arch]


def ensure_python_tarball(python_arg, arch):
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
    """pip download the cp311 macOS wheel closure (offline target) into wheels_dir."""
    if skip:
        log("(--no-site-packages) skipping wheel vendoring")
        return
    if os.path.isdir(wheels_dir):
        shutil.rmtree(wheels_dir)
    os.makedirs(wheels_dir)
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
           "--python-version", PYVER, "--implementation", "cp", "--abi", "cp311", "-d", wheels_dir]
    for tag in _platform_tags(arch):
        cmd += ["--platform", tag]
    cmd += mac_reqs()
    log("vendoring macOS wheels (pip download — the full scientific stack, this can take a while) …")
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("pip download failed — a pinned clinic package has no macOS cp311 wheel; "
                         "adjust requirements-clinic.txt and re-run.")
    whls = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")]
    sz = sum(os.path.getsize(os.path.join(wheels_dir, f)) for f in whls)
    log(f"  vendored {len(whls)} wheels ({sz / 1e6:.0f} MB)")


RUN_COMMAND = f"""#!/bin/bash
# GA Clinic - offline macOS launcher (double-click). Keep this window open while using the app.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OCT_BM_DL=1
export OCT_CLINIC_DATA="$DIR/user_data"
mkdir -p "$OCT_CLINIC_DATA"
PY="$DIR/runtime/python/bin/python3"
LIBS="$DIR/libs"

# A second double-click reopens the existing localhost instance.
if curl -fsS "http://127.0.0.1:{PORT}/api/health" 2>/dev/null | grep -q '"app":"ga-clinic"'; then
  open "http://127.0.0.1:{PORT}/"
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "GA Clinic - unpacking Python (first run only)..."
  tar -xzf "$DIR/runtime/python.tar.gz" -C "$DIR/runtime" || {{ echo "Unpack failed."; read -r _; exit 1; }}
fi
if [ ! -d "$LIBS" ]; then
  echo "GA Clinic - first-time setup (a couple of minutes, no internet needed)..."
  TMP_LIBS="$DIR/.libs-install"
  rm -rf "$TMP_LIBS"
  "$PY" -m pip install --no-index --no-deps --find-links "$DIR/wheels" --target "$TMP_LIBS" -r "$DIR/requirements.txt" \\
    || {{ rm -rf "$TMP_LIBS"; echo "Setup failed (see the messages above)."; read -r _; exit 1; }}
  mv "$TMP_LIBS" "$LIBS"
fi
export PYTHONPATH="$LIBS:$DIR/app"
echo "GA Clinic starting - your browser opens when ready."
( for i in $(seq 1 150); do
    curl -s -o /dev/null "http://127.0.0.1:{PORT}/api/health" && {{ open "http://127.0.0.1:{PORT}/"; break; }}
    sleep 0.8
  done ) &
exec "$PY" -m uvicorn clinic.api.app:app --host 127.0.0.1 --port {PORT}
"""

READ_ME = f"""GA Clinic for macOS — offline
=============================

1. Double-click  oct_ga_clinic_mac.tar.gz  to unpack it (creates the  oct_ga_clinic_mac  folder).

2. Clear the download quarantine ONCE so macOS will run the bundled files:
     - Open Terminal (Spotlight -> "Terminal").
     - Type:   xattr -dr com.apple.quarantine     (note the trailing space)
     - Drag the unpacked  oct_ga_clinic_mac  folder onto the Terminal window, then press Return.

3. Double-click  run.command  inside the folder.
     - First launch only: it unpacks Python and installs its libraries (a couple of minutes, NO internet
       needed), then your browser opens at  http://127.0.0.1:{PORT}/ .
     - Double-clicking again while it is running simply reopens the browser page.
     - If macOS still blocks it: right-click run.command -> Open -> Open (once).

4. Database = the patients tracked on this Mac. Upload E2E = Browse for a .E2E file (or paste its path),
   choose a scan, then the Bruch's-membrane source. Export Excel saves the database as .xlsx.
   Patient state, managed drag/drop copies and caches live in user_data/. To stop, close the Terminal window.

Everything runs locally and offline; nothing is installed system-wide.
"""


def _tar_filter(ti):
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    if ti.isdir():
        ti.mode = 0o755
    elif ti.name.replace("\\", "/").endswith("/run.command"):
        ti.mode = 0o755
    else:
        ti.mode = 0o644
    return ti


def make_tar():
    tar_path = MAC_OUT + ".tar.gz"
    log("creating oct_ga_clinic_mac.tar.gz …")
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(MAC_OUT, arcname="oct_ga_clinic_mac", filter=_tar_filter)
    return tar_path


def _write_launcher_and_docs():
    with open(os.path.join(MAC_OUT, "run.command"), "w", newline="\n", encoding="utf-8") as f:
        f.write(RUN_COMMAND)
    try:
        os.chmod(os.path.join(MAC_OUT, "run.command"), 0o755)   # ignored on Windows; the tar sets it
    except OSError:
        pass
    with open(os.path.join(MAC_OUT, "requirements.txt"), "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(mac_reqs()) + "\n")
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
        os.makedirs(os.path.join(MAC_OUT, "user_data"), exist_ok=True)
        _write_launcher_and_docs()
    else:
        if os.path.isdir(MAC_OUT):
            shutil.rmtree(MAC_OUT)
        os.makedirs(runtime)
        log(f"mac: resolving standalone python ({arch}) …")
        tarball = ensure_python_tarball(python_arg, arch)
        shutil.copyfile(tarball, os.path.join(runtime, "python.tar.gz"))
        vendor_wheels(os.path.join(MAC_OUT, "wheels"), arch, skip=skip_wheels)
        log("mac: copying app code + DL model …")
        copy_app(os.path.join(MAC_OUT, "app"))
        os.makedirs(os.path.join(MAC_OUT, "user_data"), exist_ok=True)
        _write_launcher_and_docs()

    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(MAC_OUT) for f in fs)
    log(f"built {MAC_OUT}  ({sz / 1e6:.0f} MB)")
    if not skip_wheels:
        log("created " + make_tar())
    log("mac: done. Ship oct_ga_clinic_mac.tar.gz; the Mac owner follows READ_ME_FIRST.txt. "
        "Verify on an actual Mac before release.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the offline GA Clinic package for macOS.")
    ap.add_argument("--python", default=None, help="path/URL to a python-build-standalone install_only .tar.gz")
    ap.add_argument("--arch", default="aarch64-apple-darwin",
                    help="aarch64-apple-darwin (Apple Silicon, default) or x86_64-apple-darwin (Intel)")
    ap.add_argument("--no-site-packages", action="store_true", help="skip wheel vendoring (structural dry-run)")
    ap.add_argument("--refresh-app", action="store_true", help="reuse the runtime; refresh app + launcher only")
    build_mac(ap.parse_args())


if __name__ == "__main__":
    main()
