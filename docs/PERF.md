# fatmap performance log

Measured with `fm_bench --time 0.3` (1280x720 ARGB32 target). Numbers are
ms per iteration; lower is better. Re-run after every performance relevant
change and append a new section (keep the old ones for comparison).

## 2026-09-30 - baseline after command lists + span split sampling

Machine: AMD Ryzen 9 9950X3D (16 cores / 32 threads), Windows 11, GCC 15.2
(MinGW), `-O3` release build, AVX2 selected at runtime.

Columns: scalar / sse2 / avx2 = immediate mode at that SIMD level;
cmdlist = deferred, serial, avx2; mt32 = deferred, 32 workers, avx2.

| workload | scalar | sse2 | avx2 | cmdlist | mt32 |
|---|---:|---:|---:|---:|---:|
| clear | 0.091 | 0.031 | 0.027 | 0.027 | 0.163 |
| rect_opaque (200x 100x100) | 0.274 | 0.154 | 0.146 | 0.150 | 0.260 |
| rect_alpha | 5.407 | 0.373 | 0.366 | 0.407 | 0.396 |
| rect_aa (sub-pixel) | 10.003 | 2.143 | 1.386 | 1.443 | 0.522 |
| circles_small_aa (1000) | 6.255 | 3.411 | 3.955 | 3.374 | 0.973 |
| circles_small_aliased | 3.306 | 1.839 | 1.815 | 2.053 | 0.840 |
| circles_large_aa (20) | 6.217 | 1.042 | 0.906 | 0.948 | 0.278 |
| lines_2000 | 13.005 | 8.871 | 8.964 | 10.094 | 1.500 |
| polyline_round | 6.460 | 4.728 | 4.735 | 4.265 | 1.753 |
| linear_gradient | 3.861 | 0.578 | 0.455 | 0.608 | 0.188 |
| radial_gradient | 3.234 | 3.150 | 2.942 | 2.930 | 0.366 |
| pattern_bilinear (rotated, repeat) | 11.695 | 9.766 | 9.653 | 9.636 | 0.878 |
| pattern_nearest | 3.760 | 2.501 | 2.334 | 2.355 | 0.305 |
| sprites_1to1 (1000x 64x64) | 6.793 | 1.245 | 0.958 | 1.047 | 0.558 |
| sprites_scaled (300, bilinear) | 19.808 | 8.690 | 7.696 | 7.721 | 1.161 |
| blend_multiply | 3.740 | 0.461 | 0.222 | 0.233 | 0.127 |
| blend_overlay | 9.210 | 13.203 | 12.149 | 9.451 | 0.876 |
| clip_circle | 1.184 | 0.190 | 0.124 | 0.130 | 0.120 |

Change log vs the first baseline (same day, before command lists):
* sprites_scaled avx2 16.0 -> 7.7 ms: clamped bilinear spans are split into
  edge / interior parts (exact in-bounds interval from the constant
  gradients) instead of sending the whole span down the wrapping path.

## 2026-09-30 - 3D pipeline (fm3d) first numbers

Same machine and build. `cmdlist` = deferred tiles, serial; `mt32` = 32
workers. The machine was noisier during these runs (Docker VM resident), so
treat single numbers as +-20 %.

| workload | scalar | sse2 | avx2 | cmdlist | mt32 |
|---|---:|---:|---:|---:|---:|
| 3d_cubes_2000 (24k tris, bilinear) | 22.848 | 17.107 | 17.326 | 20.609 | 2.786 |
| 3d_floor_bilinear (8k tris) | 12.299 | 8.602 | 8.251 | 8.954 | 1.184 |
| 3d_floor_trilinear | 23.219 | 16.873 | 16.585 | 17.189 | 1.598 |
| 3d_alpha_quads_30 (18M px) | 75.457 | 31.782 | 23.325 | 28.691 | 3.098 |
| 3d_small_tris_10k | 4.588 | 4.361 | 4.289 | 4.778 | 1.180 |

Changes during bring-up:
* fragment sampling grouped per mip level per batch + SIMD point bilinear
  kernel with in-bounds compaction: floor bilinear 16.8 -> 11.5 ms
* inline exact floor instead of libm floorf in wrapping / fixed conversion
  / LOD: floor bilinear 11.5 -> 8.3 ms, trilinear 26 -> 16.6 ms

The single threaded fragment path is still largely scalar (about 15 ns per
textured pixel): the next step is a SIMD fragment stage (vectorized plane
evaluation and perspective divide, gather based sampling) compiled per ISA.

## Observations and next targets

* Per-draw overhead dominates small shapes (circles_small ~3.4 us per
  circle, lines ~4.4 us per line): flattening with sin/cos, the stroker's
  malloc per call, qsort per draw. Targets: cached unit-circle arcs,
  scratch reuse in the stroker, insertion sort for tiny edge lists.
* Scalar fetch stages are the SIMD gaps: radial / conic gradients, the
  complex blend modes (overlay ~9 ms full screen), bilinear with wrapping.
* Thread dispatch costs ~0.1 ms per flush, which hurts trivially small
  frames (clear, rect_opaque): small command lists could run serially
  automatically.
* The single-threaded command list is within noise of immediate mode;
  every strip scans every command, so a per-strip bucket index would help
  scenes with thousands of commands.
* AVX2 is sometimes no faster than SSE2: those workloads are bound by
  geometry or scalar fetch, not by blending.
* Not yet exploited: AVX-512 (Zen 5 has full width units).
