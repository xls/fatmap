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
(Done: fragment ops, then plane evaluation, see below.)

## 2026-09-30 - SIMD rasterizer interpolation

New kernels in the dispatch table: `plane` (z, with optional clamp),
`plane_recip` (perspective 1/w, exact IEEE divide so every backend stays
bit-identical; no rcp estimate, whose bits differ between CPU vendors) and
`plane_mul` (varying * w), 8 wide on AVX2, 4 wide on SSE2 / NEON; plus
`acc_add` for long interior runs of 2D edges. 3D batches narrower than 8
columns keep the same expression inline (an indirect call costs more there).
The 3D coverage mask was already an exact per row span; it is now filled
with memset instead of a per pixel compare. `fm_kernel_test` checks every
table entry against scalar, including NaN / inf / -0 / denormal inputs.

Interleaved A/B against the previous commit, best of 9 rounds (ms):

| workload | sse2 before | sse2 after | avx2 before | avx2 after |
|---|---:|---:|---:|---:|
| 3d_alpha_quads_30 | 30.53 | 23.49 (-23 %) | 25.55 | 17.13 (-33 %) |
| 3d_cubes_occluded | 8.23 | 7.23 (-12 %) | 7.93 | 6.99 (-12 %) |
| 3d_cubes_2000 | 17.36 | 16.84 | 17.96 | 17.44 |
| 3d_small_tris_10k | 6.28 | 5.87 | 5.98 | 6.08 |

Other rows moved within this machine's +-9 % noise. 2D workloads are
unchanged: their edges rarely have runs long enough for `acc_add`.

## 2026-09-30 - cache experiments

No hardware counters on this machine, so each idea was a build variant
measured by interleaved A/B (best of 7 to 11 rounds).

* 3D tile size (`fm_bench --tile N`): smaller tiles are slower even with
  MSAA, where a 64 x 64 tile (256 KB at 4x) no longer fits L1: 32 x 32 costs
  +18 % (msaa4) to +84 % (occluded) single threaded. Larger tiles are 5 to
  13 % faster single threaded but lose load balance at 32 threads. Kept
  64; the cost is per tile x triangle overhead, not cache capacity.
* 2D band height 8 / 16 / 32 rows (`-DFM_RASTER_BAND=`): no difference
  single threaded (+-2 %), even though a full width band is 80 KB. The
  accumulator is walked sequentially and L2 keeps up. Kept 16.
* Tile prefetch (FM3D_TILE_PREFETCH, now on): each tile row lies on its
  own page, where hardware prefetchers stop, so phase C requests all color
  and depth lines of a tile up front. alpha_quads -8 %, small_tris -5 %,
  the rest unchanged, never slower.
* Texture locality (new workloads `3d_tex_{small,large}_rot{0,90}`, 1
  texel per pixel, 256^2 vs 4096^2): the large texture costs +30..40 %
  single threaded, rot90 another ~+20 % because each pixel steps one
  texture row (16 KB) and nothing streams. At 32 threads all four are
  equal (~0.57 ms): enough misses in flight hide the latency.
* Texture prefetch in the sampler (2 lines per bilinear pixel before the
  gather): -11 % on large rot90 tiled, but +4..7 % on every cached case.
  Dropped. Page sized texture tiles (32 x 32 texels = 4 KB) would fix the
  rot90 case properly; not worth the sampler rework until a real scene
  shows it.

## 2026-09-30 - phase C overhead: per tile work lists, SIMD hi-Z scan

Found with `perf` sampling (Docker Linux VM, software clock, tiled single
thread, `fm_bench --column cmdlist`). Baseline profile of 3d_cubes_occluded:
fm3d_phase_tile itself 36 % of the time, of which the loop over every
command of the frame (lines checking `d->st->rect` and each chunk's bin box)
was ~45 % and the hi-Z bounds scan ~20 %. With 2000 draws that loop ran
2000 x 240 = 480k times per frame; it is also why smaller tiles were slower.

* Per tile work lists: built after binning (count, prefix sum, fill), one
  entry per clear or per triangle chunk touching the tile, in submission
  order. A tile now only visits its own work.
* Hi-Z scan: D32F rows through a new `minmax_f32` kernel (depths are
  clamped to [0, 1], so float order is key order), D16 / D24S8 with the
  format switch hoisted out of the pixel loop.
* Tried and reverted: incremental (division free) edge stepping for the
  row spans. The profile showed fm3d_row_span_at at 6.9 %, and the stepped
  version measured within noise (its carry branch mispredicts; branchless
  plus precomputed steps did not change that).

Single thread tiled (`cmdlist`), best of 21 interleaved runs, ms:

| workload | before | lists | lists + hi-Z |
|---|---:|---:|---:|
| 3d_cubes_2000 | 23.01 | 19.72 (-14 %) | 19.59 (-15 %) |
| 3d_cubes_occluded | 6.98 | 4.72 (-32 %) | 4.33 (-38 %) |

32 threads: 3d_cubes_2000 3.84 -> 3.10 (-19 %), 3d_cubes_occluded -5 to
-7 %. Workloads with few draws (floor, quads, textures) are unchanged
within noise. The list build runs serially: ~0.08 ms per frame at 2000
draws. Measurements this session were noisy (a background process kept
~2 cores busy, total load ~26 %): single runs of identical binaries varied
by up to +-15 %, so only effects repeated across runs are reported.

## 2026-09-30 - fragment path fast paths (profile driven)

Measurement fix first: the dev machine is a Ryzen 9 9950X3D whose two
CCDs differ (96 MB V-Cache vs 32 MB, different clocks). Unpinned single
thread runs were bimodal (+-15..20 %). Pinned to CCD0 (`start /affinity
0xFFF0`) repeat runs agree within ~1 %. Multithreaded runs cannot be
pinned that way, and an A/A test showed the same binary 20 % faster from a
different directory, so mt numbers below are not used for claims.

Linux / perf profile (immediate, AVX2) of textured 3D before these
changes: memcpy 10..14 %, fm__sample_fixed 6..19 %, and for large flat
quads fm3d_raster_tri + fs_fixed 56 % against 30 % in the blend kernel.

* Tail copies: the SIMD kernels staged span tails through variable size
  memcpy (3 to 4 library calls per kernel call). Out of line fixed copies
  now (inlining them slowed long span loops by 4 to 9 %).
* Fully covered batches: flagged from the exact row spans, re-checked
  after early depth / stencil; skip mask scans, active quad lists and
  trimming, blend with the unmasked kernels.
* Power of two repeat textures: `bilinear_pts_wrap` wraps every tap by
  masking, no in-bounds / wrap split.

Pinned single thread, best of 9, before = 42993ed (ms):

| workload | avx2 before | avx2 now | cmdlist before | cmdlist now |
|---|---:|---:|---:|---:|
| 3d_alpha_quads_30 | 13.15 | 8.24 (-37 %) | 16.32 | 11.73 (-28 %) |
| 3d_tex_small_rot0 | 4.24 | 3.45 (-19 %) | 4.53 | 3.76 (-17 %) |
| 3d_tex_large_rot90 | 5.51 | 4.75 (-14 %) | 5.40 | 4.61 (-15 %) |
| 3d_floor_trilinear | 13.39 | 12.04 (-10 %) | 14.59 | 13.43 (-8 %) |
| 3d_floor_bilinear | 7.79 | 7.41 (-5 %) | 8.98 | 8.57 (-5 %) |
| 3d_small_tris_10k | 4.54 | 4.33 (-5 %) | 5.19 | 4.86 (-6 %) |
| 3d_cubes_2000 | 17.19 | 17.17 | 18.83 | 18.69 |

2D workloads unchanged (+-2 %).

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
* Cube scenes (many small triangles) did not move with the fragment fast
  paths: they are bound by per triangle setup and the vertex stage.
  Next: SIMD triangle setup / small triangle path, finer hi-Z with early
  accept.
