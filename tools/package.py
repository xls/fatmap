#!/usr/bin/env python3
"""
Build a fatmap binary release package for the current platform.

  python tools/package.py                 native compiler (gcc / clang)
  python tools/package.py --msvc          Visual Studio toolchain (Windows)
  python tools/package.py --verify        also build + run examples/consumer
                                          against the package with CMake and Meson

Steps: Meson release build (static library, sandbox off), run the tests,
`meson install` into a staging prefix, add docs + the consumer example, and
archive it as dist/fatmap-<version>-<platform>.(zip|tar.gz) + .sha256.

Package layout (a normal install prefix, relocatable):
  include/fatmap/*.h           C API
  include/fatmap/fatmap.hpp    C++17 header wrapper
  lib/libfatmap.a | fatmap.lib static library
  lib/pkgconfig/fatmap.pc      pkg-config (also fatmapxx.pc)
  lib/cmake/fatmap/            CMake package: find_package(fatmap)
  bin/fm-spirvc                SPIR-V -> C shader compiler (with -Dspirv)
  examples/consumer/           CMake + Meson example project
  README.md, PACKAGING.md
"""
import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import bootstrap  # noqa: E402  (tool venv helpers)

IS_WIN = os.name == "nt"


def log(msg):
    print("[package] " + msg, flush=True)


def run(cmd, env=None, cwd=None):
    log("$ " + " ".join(cmd))
    subprocess.check_call(cmd, env=env, cwd=cwd or ROOT)


def project_version():
    s = open(os.path.join(ROOT, "meson.build")).read()
    return re.search(r"version\s*:\s*'([^']+)'", s).group(1)


def platform_id(msvc):
    sysname = platform.system().lower()
    osname = {"windows": "windows", "linux": "linux", "darwin": "macos"}.get(sysname, sysname)
    m = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64", "i686": "x86", "x86": "x86"}.get(m, m)
    pid = "%s-%s" % (osname, arch)
    if osname == "windows":
        pid += "-msvc" if msvc else "-mingw"
    return pid


def archive(stage, out_base, use_zip):
    name = os.path.basename(stage)
    parent = os.path.dirname(stage)
    files = []
    for dp, dn, fn in os.walk(stage):
        dn.sort()
        for f in sorted(fn):
            full = os.path.join(dp, f)
            files.append((full, os.path.relpath(full, parent).replace(os.sep, "/")))
    if use_zip:
        path = out_base + ".zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for full, rel in files:
                z.write(full, rel)
    else:
        path = out_base + ".tar.gz"
        with tarfile.open(path, "w:gz", compresslevel=9) as t:
            t.add(stage, arcname=name)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w", newline="\n") as f:
        f.write("%s  %s\n" % (h, os.path.basename(path)))
    return path, h


def verify(stage, work, msvc, env):
    """Build examples/consumer from the package with CMake and with Meson, run both."""
    src = os.path.join(stage, "examples", "consumer")
    exe = ".exe" if IS_WIN else ""
    if shutil.which("cmake"):
        bd = os.path.join(work, "cmake")
        shutil.rmtree(bd, ignore_errors=True)
        cfg = ["cmake", "-S", src, "-B", bd, "-DCMAKE_PREFIX_PATH=" + stage, "-DCMAKE_BUILD_TYPE=Release"]
        if not msvc:
            cfg += ["-G", "Ninja"]
            if IS_WIN:
                cfg += ["-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++"]
        run(cfg, env=env)
        run(["cmake", "--build", bd, "--config", "Release"], env=env)
        for t in ("consumer_c", "consumer_cpp"):
            p = os.path.join(bd, "Release", t + exe) if msvc else os.path.join(bd, t + exe)
            run([p], env=env, cwd=bd)
    else:
        log("cmake not found: skipping the CMake consumer check")
    meson = bootstrap.tool("meson") if os.path.exists(bootstrap.tool("meson")) else "meson"
    bd = os.path.join(work, "meson")
    shutil.rmtree(bd, ignore_errors=True)
    cfg = [meson, "setup", bd, src, "--buildtype=release",
           "--pkg-config-path=" + os.path.join(stage, "lib", "pkgconfig"), "--cmake-prefix-path=" + stage]
    if msvc:
        cfg.append("--vsenv")
    run(cfg, env=env)
    run([meson, "compile", "-C", bd], env=env)
    for t in ("consumer_c", "consumer_cpp"):
        run([os.path.join(bd, t + exe)], env=env, cwd=bd)


def main():
    ap = argparse.ArgumentParser(description="fatmap release packaging")
    ap.add_argument("--msvc", action="store_true", help="Visual Studio toolchain (--vsenv)")
    ap.add_argument("--platform", help="override the platform id in the archive name")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--verify", action="store_true", help="build + run the consumer example from the package")
    ap.add_argument("--system-tools", action="store_true", help="use meson/ninja from PATH")
    ap.add_argument("--threads", default="auto", choices=["auto", "enabled", "disabled"])
    ap.add_argument("--setup-arg", action="append", default=[], help="extra `meson setup` argument")
    a = ap.parse_args()

    meson = bootstrap.ensure_tools(a.system_tools)
    env = bootstrap.tool_env()
    version = project_version()
    pid = a.platform or platform_id(a.msvc)
    if a.threads == "disabled":
        pid += "-nothreads"
    name = "fatmap-%s-%s" % (version, pid)
    builddir = os.path.join(ROOT, "build-pkg-" + pid)
    work = os.path.join(a.out, "work")
    stage = os.path.join(work, name)

    shutil.rmtree(builddir, ignore_errors=True)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(a.out, exist_ok=True)

    setup = [meson, "setup", builddir, "--buildtype=release", "-Ddefault_library=static",
             "-Dsandbox=disabled", "-Dtests=true", "-Dprofile=true", "-Dthreads=" + a.threads,
             "--libdir=lib", "--includedir=include", "--prefix=" + stage,
             "-Dpkgconfig.relocatable=true"] + a.setup_arg
    if a.msvc:
        setup.append("--vsenv")
    run(setup, env=env)
    run([meson, "compile", "-C", builddir], env=env)
    if not a.no_test:
        run([meson, "test", "-C", builddir, "--print-errorlogs"], env=env)
    run([meson, "install", "-C", builddir, "--no-rebuild", "--skip-subprojects"], env=env)

    shutil.copy(os.path.join(ROOT, "README.md"), stage)
    shutil.copy(os.path.join(ROOT, "docs", "PACKAGING.md"), stage)
    shutil.copytree(os.path.join(ROOT, "examples", "consumer"), os.path.join(stage, "examples", "consumer"))

    if a.verify:
        verify(stage, os.path.join(work, "verify-" + pid), a.msvc, env)

    path, h = archive(stage, os.path.join(a.out, name), IS_WIN)
    log("wrote %s (%d KiB, sha256 %s)" % (os.path.relpath(path, ROOT), os.path.getsize(path) // 1024, h))


if __name__ == "__main__":
    main()
