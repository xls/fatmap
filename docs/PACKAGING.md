# Using fatmap in your project

fatmap is a C11 static library with a header only C++17 wrapper. There are
three ways to use it, none of which need CMake to *build* fatmap:

1. download a prebuilt release package (CMake, pkg-config, or plain flags)
2. build it from source as a Meson subproject
3. copy the sources into your tree (it is plain C with no dependencies)

## 1. Release packages

Every [GitHub release](https://github.com/xls/fatmap/releases) has one
archive per platform:

| archive                                   | toolchain                         |
|-------------------------------------------|-----------------------------------|
| `fatmap-<ver>-windows-x64-msvc.zip`       | Visual Studio 2019+ (`fatmap.lib`, /MD) |
| `fatmap-<ver>-windows-x64-mingw.zip`      | MinGW-w64 gcc / clang (`libfatmap.a`) |
| `fatmap-<ver>-windows-x86-msvc.zip`       | Visual Studio 2019+, 32 bit (`fatmap.lib`, /MD) |
| `fatmap-<ver>-linux-x64.tar.gz`           | gcc / clang, glibc 2.35+          |
| `fatmap-<ver>-linux-arm64.tar.gz`         | gcc / clang, glibc 2.35+          |
| `fatmap-<ver>-macos-arm64.tar.gz`         | Apple clang, macOS 11+            |
| `fatmap-<ver>-macos-x64.tar.gz`           | Apple clang, macOS 11+            |

Each one is an ordinary install prefix:

```
fatmap-<ver>-<platform>/
  include/fatmap/        fatmap.h (C API) + fatmap.hpp (C++ wrapper)
  lib/                   libfatmap.a or fatmap.lib
  lib/pkgconfig/         fatmap.pc, fatmapxx.pc
  lib/cmake/fatmap/      fatmapConfig.cmake, fatmapConfigVersion.cmake
  bin/fm-spirvc          SPIR-V -> C shader compiler (fm-spirvc --help)
  examples/consumer/     a C and a C++ program with CMake + Meson files
```

The libraries are release builds with every SIMD backend compiled in
(SSE2 + AVX2 on x64, NEON on arm64, picked at runtime), the built-in thread
pool and the profiler counters enabled. They contain no LTO bitcode, so any
compiler version of the same family can link them. The package is
relocatable: unpack it anywhere.

### CMake

```cmake
find_package(fatmap 0.9 REQUIRED)
target_link_libraries(app PRIVATE fatmap::fatmap)     # C
target_link_libraries(app PRIVATE fatmap::fatmapxx)   # C++ wrapper (implies fatmap::fatmap)
```

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/fatmap-<ver>-<platform>
```

The imported targets carry the include path and the system libraries
(`m` and `Threads::Threads` on Linux / macOS, nothing on Windows).

### Meson

```meson
fatmap = dependency('fatmap')    # finds fatmap.pc
```

```
meson setup build --pkg-config-path /path/to/fatmap-<ver>-<platform>/lib/pkgconfig
```

Without pkg-config installed (common on Windows), Meson can read the CMake
package instead:
`dependency('fatmap', method : 'cmake', modules : ['fatmap::fatmap'])` with
`--cmake-prefix-path /path/to/fatmap-<ver>-<platform>`.

### pkg-config / Make / anything else

```
cc app.c $(pkg-config --cflags --libs fatmap)
```

or by hand:

```
cc  app.c -I<pkg>/include <pkg>/lib/libfatmap.a -lm -pthread     # Linux / macOS
gcc app.c -I<pkg>/include <pkg>/lib/libfatmap.a                  # MinGW
cl  app.c /I<pkg>\include <pkg>\lib\fatmap.lib                   # MSVC
```

No preprocessor defines are needed. (`FM_SHARED` is only for a DLL build.)

### Notes

- MSVC: the library uses the release DLL runtime (`/MD`). Debug
  configurations (`/MDd`) link fine for a C library but may print
  `LNK4098`; add `/NODEFAULTLIB:msvcrt` there if it bothers you.
- Linking into a shared library on Linux is fine: the archive is built with
  `-fPIC`.
- The C++ wrapper is header only, so C++ users only need a C++17 compiler.

## 2. Meson subproject (build from source)

Put this in `subprojects/fatmap.wrap`:

```ini
[wrap-git]
url = https://github.com/xls/fatmap.git
revision = v0.9.1
depth = 1

[provide]
fatmap = fatmap_dep
fatmapxx = fatmapxx_dep
```

Then `dependency('fatmap')` (or `'fatmapxx'`) builds fatmap as part of your
project. As a subproject fatmap skips its tests, the SDL sandbox and all
install rules. Options pass through as usual, for example
`default_options : ['fatmap:threads=disabled']` for a platform without a
thread library.

## 3. Building a package yourself

```
python tools/package.py [--msvc] [--verify] [--threads disabled]
```

builds, tests and installs into `dist/work/fatmap-<ver>-<platform>/` and
writes `dist/fatmap-<ver>-<platform>.zip|tar.gz` plus a `.sha256` file.
`--verify` also builds and runs `examples/consumer` against the package
with CMake and Meson. CI (`.github/workflows/release.yml`) runs this on
every platform and attaches the archives to the release when a `v*` tag is
pushed.
