# fatmap

A fast software renderer in pure C11: an HTML-canvas style 2D API today, a
glm-style 3D pipeline next. Named after Mats Byggmastar's classic *fatmap*
texture mapping articles, whose ideas (constant gradients, sub-pixel
correct edge setup, fixed point stepping) still live in the core.

The library is layered so the low level pieces can be used on their own, for
example as the software backend of an OpenGL / Direct3D style API:

```
 fatmap++ (C++17 header wrapper)       bindings (rust, python, ...)
        |                                      |
 fm2d  - HTML canvas API: paths, strokes, dashes, gradients, patterns,
         clipping, 27 composite ops, drawImage, Path2D, hit testing
        |
 fm_exec  - command lists + executors (multithreaded, optional)
 fm_pipe  - op based pixel pipeline: fetch -> coverage -> blend -> store
            samplers (nearest / bilinear, repeat / clamp / mirror / border)
 fm_raster- coverage rasterizer: analytic AA or GL/D3D point sampling,
            nonzero / even-odd, row-range rendering
 fm_core  - surfaces, colors, blend kernels, CPU detection, SIMD dispatch
 fm_math  - glm compatible vec/mat/quat (header only)
 fm_profile - built-in zone profiler (+ optional Tracy)
```

## Building

Meson + Ninja, no system dependencies. `bootstrap` installs Meson and Ninja
into a private venv (`.tools/`), fetches third party code into `external/`
(SDL3 for the sandbox, glm for the C++ tests, optionally Tracy) and builds.

```sh
./bootstrap.sh              # Linux / macOS / MSYS
bootstrap.cmd               # Windows (uses whatever compiler is on PATH)
bootstrap.cmd --msvc        # Windows, force Visual Studio (build-msvc/)
python bootstrap.py --help  # --debug, --profile, --tracy, --no-sandbox, --test ...
```

After bootstrapping, rebuild with `.tools/Scripts/meson compile -C build`
(or `.tools/bin/meson` on Linux/macOS).

Meson options (`-Doption=value`):

| option    | default | meaning |
|-----------|---------|---------|
| `threads` | auto    | thread pool executor; `disabled` needs no thread library |
| `profile` | true    | compile built-in profiler zones |
| `tracy`   | false   | forward zones to Tracy (`bootstrap --tracy`) |
| `sandbox` | auto    | SDL3 sandbox app |
| `tests`   | true    | tests + benchmark |

Tested: Windows x64 (GCC 15 / MinGW, MSVC 19.5x), Linux x64 and Linux
AArch64 (GCC, Alpine containers; AArch64 under QEMU).

## Quick start (C)

```c
#include <fatmap/fatmap.h>

fm_surface* fb  = fm_surface_create(1280, 720, FM_FORMAT_ARGB32);
fm2d_ctx*   ctx = fm2d_create(fb);

fm2d_clear(ctx, FM_RGB(20, 20, 30));
fm2d_paint* g = fm2d_paint_linear(0, 0, 400, 0);
fm2d_paint_add_stop(g, 0, FM_RGB(255, 0, 0));
fm2d_paint_add_stop(g, 1, FM_RGB(0, 0, 255));
fm2d_set_fill_paint(ctx, g);
fm2d_begin_path(ctx);
fm2d_arc(ctx, 200, 200, 150, 0, 6.2831853f, 0);
fm2d_fill(ctx, FM_FILL_NONZERO);
fm2d_paint_release(g);

fm_surface_write_png(fb, "out.png");
```

## Quick start (C++)

```cpp
#include <fatmap/fatmap.hpp>

fm::Surface  fb(1280, 720);
fm::Canvas2D ctx(fb);
auto grad = fm::Canvas2D::createLinearGradient(0, 0, 400, 0);
grad.addColorStop(0, "red").addColorStop(1, "#00f");
ctx.fillStyle(grad);
ctx.beginPath();
ctx.arc(200, 200, 150, 0, 2 * fm::pi);
ctx.fill();
```

## Multithreading

Contexts, rasterizers, pipelines and command lists are independent objects:
use one per thread freely. For parallel rendering of one frame, switch a
context to deferred mode:

```c
fm_executor* ex = fm_executor_create(0);   /* 0 = one worker per CPU */
fm2d_set_deferred(ctx, 1);
fm2d_set_executor(ctx, ex);
... draw ...
fm2d_flush(ctx);                           /* executes the command list */
```

Draws are recorded into an `fm_cmdlist`, then executed in two parallel
phases: geometry (flatten / stroke / edge build, per command) and raster
(per 32-row strip, all commands in order, no locks). The result is
bit-identical to immediate mode for any thread count. With
`-Dthreads=disabled` the same code runs serially; platforms with their own
task system can plug in an `fm_executor` with a custom `parallel_for`.

Rules in deferred mode (as with GPU APIs): pixels are valid after
`fm2d_flush`, and images drawn must not change until then. Gradients and
patterns are snapshotted automatically. Drawing the target onto itself
flushes first.

## SIMD

Every kernel exists for scalar, SSE2, AVX2 (x86, runtime selected) and NEON
(AArch64). The op formulas live in one template (`src/core/fm_kernels_tmpl.h`)
instantiated per backend, and every backend is tested to be bit-identical to
scalar. Force a level with `fm_simd_set()` or the `FM_SIMD` environment
variable (`scalar`, `sse2`, `avx2`, `neon`).

## Profiling

* `fm_profile.h`: zones per draw call (`2d.fill`, `2d.stroke`, `cmd.raster`,
  ...), `fm_prof_report()` prints a table. Press `P` in the sandbox.
* `fm_bench`: every workload at every SIMD level, plus command list serial
  and multithreaded columns. `--csv`, `--prof`, `--threads`, `--strip`.
* `bootstrap --profile` builds `build-prof/` (optimized + symbols) for VTune,
  Superluminal, perf or Instruments; `bootstrap --tracy` enables Tracy.

See `docs/PERF.md` for current numbers.

## Sandbox

`fatmap_sandbox` (SDL3, 1280x720): keys `1`-`6` scenes, `S` SIMD level, `T`
threads, `A` anti-aliasing, `B` bilinear, `Up`/`Down` object count, `P`
profiler, `V` vsync. `--shots <dir>` renders every scene single and
multithreaded, prints timings and saves PNGs (handy for CI).

## Tests

`meson test -C build` runs `fm_test` (SIMD equivalence for all 27 blend ops,
coverage accuracy, fill rules, AA modes, strokes, dashes, blend math,
gradients, clipping, images, hit testing, CSS colors, deferred / threaded
equivalence) and `fm_cpp_test` (C++ wrapper, fm_math vs glm).

## Status

Done: core, SIMD kernels, rasterizer, pipeline, 2D canvas API, command
lists + threading, sandbox, bench, C++ 2D wrapper.

Next:
* 3D: glm style pipeline, clipping, fatmap style triangle setup with
  perspective correct texturing, depth buffer, tile binning
* canvas: text, shadows, filters, unbounded composite ops (`copy`,
  `source-in`, `source-out`, `destination-in`, `destination-atop` clear
  outside the shape in browsers; fatmap currently only affects the shape)
* performance: SIMD radial/conic gradients and bilinear, span splitting for
  clamped sampling, stroker allocation, AVX-512 tier
