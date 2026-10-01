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

## 2026-10-01 - batch masks as words, multithreaded scaling

Single thread (pinned, best of 7), batch masks built with two 16 byte
compares, scanned as 64 bit words (21f2145): 3d_cubes_2000 -7 %,
occluded -6 %, bilinear floor -8 %, small_tris -8 %, alpha_quads -6 %.

Overdraw (new fm3d_stats fragment counters): shaded fragments are ~1 per
covered pixel in every 3D workload, so finer hi-Z / hidden surface
removal would only save rasterization of already rejected fragments.

Multithreaded scaling was poor (tile phase flat beyond 8 threads). Core
placement test on the 9950X3D: 16 threads on 16 cores over both CCDs were
slower than 8 on one CCD, because dynamic hand out moved tiles between
CCDs every frame.

* Tile affinity + stealing (ea93738, a85c7cc): 16 threads over both CCDs
  2.00 -> 1.20 ms tile phase; 32 threads: textured quads -28..-33 %,
  alpha_quads -28 %, bilinear floor -12 %.
* Pool spin 20000 -> 2000 pauses (878060e): 3D frames -10..-21 %, small 2D
  frames up to -59 % (clear, rect, blend_multiply, linear_gradient). The
  old "~0.1 ms dispatch cost per flush" note below was mostly this.

Tried and reverted (measured, not kept):

* Reciprocal + exact correction instead of 64 bit division for row spans:
  +4..9 % on Zen 5 (fast divider). May win on older Intel cores.
* Fused texcoord + gather kernel from float u, v (8 wide AVX2): +9..14 % on
  cubes / floors, small gains on full screen quads.
* Chunked task claiming in the pool: within +-2 %.
* Affinity for 2D strips: no gain, +13..17 % on ~0.1 ms frames (23 strips on
  32 workers).

Remaining for many-draw scenes (3d_cubes_2000 at 32 threads, ~2.5 ms):
recording copies ~1.9 KB of state + vertices per draw on the calling
thread (0.45 ms, cross-CCD cache traffic), vertex phase and setup scale
poorly for 2000 tiny draws. Vertex buffer objects (upload once, draw by
reference) would remove most of the copying.

## 2026-10-01 - SPIR-V: lowered control flow, C backend, vectorizable math

Pinned to CCD0, best of 3. Seascape: tests/spirv/seascape.frag at
480x270; 3d_shader_*: t_control.frag full screen (1280x720). The pinning
mask also limits the "mt32" column to the 12 logical CPUs of the mask;
unpinned, 32 threads: 3d_seascape_spirv 47 ms, 3d_seascape_aot 38 ms.

| ms (1 thread / 12 threads)  | before        | after       |
|-----------------------------|---------------|-------------|
| 3d_shader_c (C callback)    | 117.0 / 14.3  | 57.6 / 6.7  |
| 3d_shader_spirv (interp.)   | 172.7 / 18.8  | 39.9 / 5.0  |
| 3d_shader_aot (SPIR-V -> C) | 174.1 / 17.6  | 30.3 / 4.0  |
| 3d_seascape_spirv           | 3967 / 433    | 651 / 86    |
| 3d_seascape_aot             | 3874 / 404    | 568 / 74    |

* Lowering the structured control flow once (sv_lower: mask ops, IF /
  LOOP, block bodies) instead of walking the CFG per batch: neutral
  (+1 % / -5 %), but it is what the C backend prints.
* The first C backend was only 2-8 % faster than the interpreter: with the
  C library's math stubbed out, a Seascape frame took 93 ms instead of
  3992 ms. libm's sinf / cosf / powf are scalar calls (and differ between
  platforms), so no loop calling them vectorizes.
* fm_vmath.h (straight line sin / cos / exp / log / pow, double core,
  within 1 ulp; exact floor / round / fmin / fmax): AVX2 sin 1.2 ns vs
  libm 23 ns per value, pow 3.5 vs 36 ns. Interpreter 4-6x faster.
* Generated loops still scalar: GCC did not inline fm_powf / fm__sincos
  in the large shader functions (always_inline now), and floorf / fminf /
  fmaxf are libm calls at the SSE2 baseline (they need SSE4.1 or have NaN
  rules no min / max instruction has). With both fixed every hot loop of
  the generated Seascape vectorizes: 1979 -> 568 ms (SSE2), 279 ms -mavx2.
* An embeddable C compiler as a runtime "JIT" would not pay: the same
  generated Seascape compiled -O0 (TCC class) takes 11.0 s, scalar -O1 /
  -O2 (MIR class) 2.6 / 2.1 s, against 0.65 s for the interpreter, whose
  per operation loops are vectorized. Runtime speed needs SIMD codegen.
* Reference: Mesa 25.0.7 llvmpipe (LLVM 19, 256 bit) in a linux/amd64
  container on the same machine renders the same Seascape frame (mean
  rgb difference 0.7) in 59 ms on 1 thread, 4.1 ms on 32. Breakdown of
  the 5x on the compiled path: -mavx2 2.0x (568 -> 279 ms), float
  precision math 1.6x (279 -> 172 ms, image still matches llvmpipe), ray
  march loops at 34 of 64 active lanes, the rest presumably values going
  through memory between blocks.

## 2026-10-01 - SPIR-V per ISA (AVX2, AVX-512), fast math

New level FM_SIMD_AVX512 (F / DQ / BW / VL; the 2D / 3D kernels reuse the
AVX2 table for now). The SPIR-V executor is a template (fm3d_spirv_exec.h)
compiled per ISA; the generated C has target("avx2") / target("avx512...")
copies of each stage (GCC / Clang) and dispatches on fm_simd_current().
Seascape 480x270, ms, one thread pinned to CCD0:

| backend               | SSE2 | AVX2 | AVX-512 | 32 threads (unpinned) |
|-----------------------|------|------|---------|-----------------------|
| interpreter           | 655  | 368  | 258     | 21.4                  |
| interpreter, fast     |      | 259  | 215     | 15.0                  |
| compiled to C         | 568  | 285  | 168     | 11.4                  |
| compiled to C, fast   |      | 205  | 130     | 8.6                   |
| llvmpipe (reference)  |      | 59   |         | 4.1                   |

3d_shader (t_control.frag, 1280x720): interpreter 40.7 / 27.1 / 22.6 ms,
compiled 31.7 / 20.5 / 15.5 ms (SSE2 / AVX2 / AVX-512); the hand written C
callback (scalar per pixel loop) 58 ms.

* fm_fast_* (float only): 1.36-1.46x over the double core math, images
  within rounding of llvmpipe's. The first version returned inf for sin of
  |x| > 6.6e6 (unreduced argument through the polynomial; the precise one
  above 1.7e9): sky pixels evaluate the sea at ~1000 units and blend it
  with weight 0, inf * 0 = NaN blacked a batch. Remainders are clamped now.
* Divergence: in the ray march loops 53.5 % of the 64 lanes are active,
  73.9 % of quad aligned 16 lane chunks. Chunk granular control flow is
  worth up to ~1.3x in those loops.

## 2026-10-01 - SPIR-V: quad groups, divergence analysis

Fragment lanes of both shader backends are in quad group order now (16
lanes = columns 8k..8k+7 of both rows: four whole 2x2 quads). The
generated C runs the program per 16 lane group when a loop's exits can
differ per lane (each group's loops end when its pixels are done), else on
the whole batch. Divergence analysis (optimistic fixed point: a phi is
lane uniform while its incoming values are and every branch choosing
between them is) decides that, and makes uniform loop counters scalars.
fm3d_sample_quads samples quads in that order. ms, one thread on CCD0:

| workload (generated C)    | AVX2 before | after | AVX-512 before | after |
|---------------------------|-------------|-------|----------------|-------|
| 3d_shader_aot             | 20.9        | 19.3  | 16.8           | 14.7  |
| 3d_seascape_aot           | 301         | 272   | 181            | 173   |
| 3d_seascape_aot_fast      | 211         | 189   | 134            | 130   |

* Groups for every program first: Seascape -4..-11 %, but the short
  control shader +23..+46 % (4x the per group fixed cost: masks, uniform
  loads, variable setup, four sampler calls of 4 quads; the sampler itself
  only +10 %). Hence groups only for divergent loops.
* Batch order <-> group order copies first as per lane gathers: the
  interpreter +20 % on the control shader; as runs of 8 contiguous pixels
  (memcpy) it is back to +1 %.

## 2026-10-01 - a game like shader: Doom 3 BFG light interactions vs llvmpipe

Seascape is one long ray marcher; games run short shaders that sample a
lot. 3d_bfg_* (tests/spirv/bfg_interaction.*, tests/bfg_scene.h): BFG's
interaction programs, 1280x720, a 64 x 36 grid, 4 additive light passes,
5 trilinear textures per pixel (3.7 M fragments, 18 M samples). fatgl's
tools/bench renders the same scene through GL on fatgl and on Mesa 25.0.7
llvmpipe (Docker, same machine); the images match (mean rgb within 1.3).

| ms, 1 thread pinned (32 threads)          | full        |
|-------------------------------------------|-------------|
| Mesa llvmpipe                             | 133 (11.0)  |
| fatgl (glslang SPIR-V, interpreter, fast) | 854 (56.6)  |
| fatgl, the same SPIR-V after spirv-opt -O | 610 (43.3)  |
| 3d_bfg_spirv_O0 (interpreter)             | 905 (61)    |
| 3d_bfg_spirv (glslc -O, interpreter)      | 690 (51)    |
| 3d_bfg_aot_fast (compiled to C)           | 478 (29.5)  |

Split by fragment shader variant (fatgl / llvmpipe, 1 thread):

| variant                           | fatgl  | llvmpipe | gap  |
|-----------------------------------|--------|----------|------|
| constant output (pipeline floor)  | 113    | 9.5      | 12x  |
| + 5 texture fetches               | +281   | +110     | 2.5x |
| + the lighting math               | +553   | +16      | 35x  |

* The math is the interpreter: ~60 flops per pixel at ~150 ns. Inlining
  glslang's helper calls by hand saved 130 ms of it (parameter / return
  copies the inliner leaves).
* The floor (profiled, sampling): ~40 % per batch interpreter overhead
  (sv_run_fs / sv_body / sv_setup) for a two instruction shader, ~15 %
  interpolating all 8 vec4 varyings when one is read, ~15 % triangle
  rasterization, ~10 % conversion + blending.
* Texturing: ~15 ns per trilinear sample against ~6 for llvmpipe.
* AVX-512 is no faster than AVX2 for the compiled shader: sampling and the
  pipeline stages around the shader run the AVX2 kernels.
* Plan to parity (docs/BACKLOG.md): SPIR-V optimization for unoptimized
  input (1.3x measured), the JIT (35x gap on math), the fragment
  pipeline floor (used varyings only, per batch setup), SIMD trilinear.

## 2026-10-02 - the SPIR-V JIT (machine code, AVX2 / AVX-512)

src/3d/fm3d_jit*.c: the lowered program as a 16 lane register program
(components split, copies renamed, stored values forwarded, dead code
removed), then x86 machine code with linear scan allocation. Bit for
bit the interpreter's images (test_jit, AVX2 and AVX-512, x86-64 and
x86-32).

| ms, 1 thread pinned (fm_bench)  | interpreter avx2 / avx512 | JIT avx2 / avx512 |
|---------------------------------|---------------------------|-------------------|
| 3d_bfg_spirv (glslc -O)         | 711 / 667                 | 396 / 418         |
| 3d_bfg_spirv_O0                 | 1168 / 1176               | 436 / 413         |
| 3d_seascape_fast                | 266 / 209                 | 140 / 147         |

fatgl tools/bench (BFG interactions through GL), 1 thread / 32 threads:

| build               | interpreter  | JIT        | Mesa llvmpipe |
|---------------------|--------------|------------|---------------|
| x86-64 (MinGW)      | 871 / 60.8   | 368 / 25.2 | 133 / 11.0    |
| x86-32 (MSVC)       | 3040 / 175   | 536 / 37.1 |               |

* The -O0 SPIR-V (what fatgl hands over) now runs as fast as glslc -O's:
  the JIT's own cleanups do what spirv-opt did.
* Seascape reaches the C backend's speed (130 ms).
* BFG's ~400 ms left are the pipeline floor and texturing measured before
  (113 + 281 ms): the next targets, with the varyings only read
  interpolated, the per batch fixed costs and SIMD trilinear.
* AVX-512 is not faster than AVX2 yet: compares and selects go through k
  registers and back to vectors, and the sampling helpers dominate.

## 2026-10-02 - toward llvmpipe: SIMD sampling, the fragment floor, JIT tuning

fatgl tools/bench (BFG light interactions, 1280x720), x64, best of 30,
single thread pinned to CCD0:

| step                                              | 1 thread | 32 threads |
|---------------------------------------------------|----------|------------|
| JIT (start of the day)                            | 368      | 25.2       |
| AVX2 quad sampler, packed mip levels              | 366      | 27.9       |
| vector LOD, paired texel loads, 2 channel lerp    | 247      | 15.9       |
| AVX-512 quad sampler                              | 215      | 13.5       |
| JIT interpolates the varyings itself              | 187      | 11.3       |
| one TLS lookup per batch (MinGW emulated TLS)     | ~178     | 10.7       |
| GL_ONE/GL_ONE blend, masked tails, packed colors  | ~170     | 10.6       |
| k1 compare / select fusion, fewer xmm saves        | 165      | -          |
| direct sampler calls, texture windows, textureProj | 158-161 | 10.0       |
| Mesa llvmpipe 25.0 (LLVM 19, 256 bit)             | 133      | 11.0       |

With FM_SIMD=avx2 (no AVX-512): 212 ms single thread.

Split by fragment shader variant (1 thread, ms):

| variant                          | fatgl | llvmpipe |
|----------------------------------|-------|----------|
| constant output (pipeline floor) | 21.5  | 9.5      |
| + 5 texture fetches              | +109  | +110     |
| + the lighting math              | +31   | +16      |

* Texturing is at llvmpipe's level now: trilinear at ~21 cycles per
  sample with AVX-512 (16 lanes per register, texel windows), ~31 with
  AVX2. Hardware gathers are microcoded on Zen (~3 cycles per element):
  scalar loads, 64 bit pairs and windows instead.
* Seascape fast, AVX-512 1 thread: 134.5 -> 115.3 ms (k register round
  trips gone), faster than AVX2 (122.6) and the C backend (131).
* Precise math in the JIT runs the interpreter's vectorized sv_math (it
  was a scalar loop per component): Seascape precise 2090 -> 263 ms.
* Left: the floor's per batch costs (raster + shade bookkeeping, glue),
  values live across sampler calls (spilled), the zero initialized
  variables of a shader wide if (spilled), the AVX2 sampler.
* The rest of fm_bench is unchanged (3d_small_tris_10k, alpha quads,
  floors within 1 %).

## Observations and next targets

* Per-draw overhead dominates small shapes (circles_small ~3.4 us per
  circle, lines ~4.4 us per line): flattening with sin/cos, the stroker's
  malloc per call, qsort per draw. Targets: cached unit-circle arcs,
  scratch reuse in the stroker, insertion sort for tiny edge lists.
* Scalar fetch stages are the SIMD gaps: radial / conic gradients, the
  complex blend modes (overlay ~9 ms full screen), bilinear with wrapping.
* Thread dispatch cost per flush: mostly fixed by the shorter pool spin
  (clear 0.29 -> 0.12 ms at 32 threads).
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
