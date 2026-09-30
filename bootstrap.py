#!/usr/bin/env python3
"""
fatmap bootstrap - sets up everything needed to build, on any platform.

  1. installs Meson + Ninja into a private venv (.tools/), no system changes
  2. fetches third party code into external/ (glm, SDL3 wrap, optional Tracy)
  3. configures and builds with Meson + Ninja

Usage:
  python bootstrap.py                 release build in build/
  python bootstrap.py --debug         debug build in build-debug/
  python bootstrap.py --profile       debugoptimized build in build-prof/ (symbols
                                      for VTune / Superluminal / perf / Instruments)
  python bootstrap.py --msvc          force the Visual Studio toolchain (Windows)
  python bootstrap.py --tracy         also fetch Tracy and enable it
  python bootstrap.py --no-sandbox    skip SDL3 + sandbox
  python bootstrap.py --test          run the test suite after building
  python bootstrap.py --fetch-only    only install tools + fetch dependencies
"""
import argparse
import os
import shutil
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(ROOT, ".tools")
EXTERNAL = os.path.join(ROOT, "external")  # Meson subproject_dir (see *.wrap)

IS_WIN = os.name == "nt"


def log(msg):
    print("[bootstrap] " + msg, flush=True)


def run(cmd, env=None, cwd=None):
    log("$ " + " ".join(cmd))
    subprocess.check_call(cmd, env=env, cwd=cwd or ROOT)


def tools_bin():
    return os.path.join(TOOLS, "Scripts" if IS_WIN else "bin")


def tool(name):
    exe = name + (".exe" if IS_WIN else "")
    return os.path.join(tools_bin(), exe)


def ensure_tools(use_system):
    if use_system:
        for t in ("meson", "ninja"):
            if not shutil.which(t):
                sys.exit("error: --system-tools given but '%s' is not on PATH" % t)
        return "meson"
    if not os.path.exists(tool("meson")) or not os.path.exists(tool("ninja")):
        log("creating tool venv in .tools/")
        venv.EnvBuilder(with_pip=True, clear=False).create(TOOLS)
        py = os.path.join(tools_bin(), "python" + (".exe" if IS_WIN else ""))
        run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", "meson>=1.4", "ninja"])
    return tool("meson")


def tool_env():
    env = dict(os.environ)
    env["PATH"] = tools_bin() + os.pathsep + env.get("PATH", "")
    return env


def fetch_deps(meson, env, want_tracy, want_sdl):
    """All third party code lives in external/ as Meson wraps (git clones or
    source archives). Meson would also fetch them lazily during setup; doing
    it up front makes offline rebuilds possible."""
    names = ["glm"]
    if want_sdl:
        names.append("sdl3")
    if want_tracy:
        names.append("tracy")
    run([meson, "subprojects", "download"] + names, env=env)


def main():
    ap = argparse.ArgumentParser(description="fatmap bootstrap")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--profile", action="store_true", help="debugoptimized build with symbols")
    ap.add_argument("--builddir", default=None)
    ap.add_argument("--msvc", action="store_true", help="use the Visual Studio toolchain (--vsenv)")
    ap.add_argument("--tracy", action="store_true")
    ap.add_argument("--no-sandbox", action="store_true")
    ap.add_argument("--system-tools", action="store_true", help="use meson/ninja from PATH")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--clean", action="store_true", help="wipe the build dir first")
    a = ap.parse_args()

    if sys.version_info < (3, 8):
        sys.exit("error: Python 3.8+ required")

    meson = ensure_tools(a.system_tools)
    env = tool_env()
    fetch_deps(meson, env, a.tracy, not a.no_sandbox)
    if a.fetch_only:
        return

    if a.debug:
        buildtype, default_dir = "debug", "build-debug"
    elif a.profile:
        buildtype, default_dir = "debugoptimized", "build-prof"
    else:
        buildtype, default_dir = "release", "build"
    builddir = a.builddir or default_dir
    if a.msvc and not a.builddir:
        builddir += "-msvc"
    bpath = os.path.join(ROOT, builddir)
    if a.clean and os.path.exists(bpath):
        shutil.rmtree(bpath)

    setup = [meson, "setup", builddir, "--buildtype=" + buildtype,
             "-Dsandbox=" + ("disabled" if a.no_sandbox else "enabled"),
             "-Dtracy=" + ("true" if a.tracy else "false")]
    if a.msvc:
        setup.append("--vsenv")
    if os.path.exists(os.path.join(bpath, "build.ninja")):
        setup.append("--reconfigure")
    run(setup, env=env)
    run([meson, "compile", "-C", builddir], env=env)
    if a.test:
        run([meson, "test", "-C", builddir, "--print-errorlogs"], env=env)

    exe = ".exe" if IS_WIN else ""
    log("done.")
    log("  sandbox : %s" % os.path.join(builddir, "fatmap_sandbox" + exe))
    log("  tests   : %s" % os.path.join(builddir, "fm_test" + exe))
    log("  bench   : %s" % os.path.join(builddir, "fm_bench" + exe))
    log("  rebuild : %s compile -C %s" % (os.path.relpath(meson, ROOT), builddir))


if __name__ == "__main__":
    main()
