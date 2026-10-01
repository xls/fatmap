# Backlog

Deferred plans, so they do not get lost. Measurements behind the
performance items are in docs/PERF.md.

## 3D: parity with Mesa llvmpipe (in progress)

The bar: llvmpipe renders Doom 3 BFG's light interactions (fm_bench
3d_bfg_*, fatgl's tools/bench) in 133 ms on one thread, 11 ms on 32; fatgl
takes 854 / 56.6 ms. The gap by part: pipeline floor 12x, texturing 2.5x,
shader math 35x (interpreter). In order:

1. **SPIR-V optimization of unoptimized input.** glslang (fatgl) emits
   loads / stores of function variables, copies, inlined call
   parameters; spirv-opt -O on the same module: 854 -> 610 ms. Either
   SPIRV-Tools in fatgl (glslang's optimizer hook, C++ like glslang) or a
   pass in fatmap's loader (store to load forwarding of function
   variables, copy propagation, dead code) so every user benefits.
2. **SIMD JIT on the lowered program** (x86-64 and x86-32: games run 32
   bit; SSE2 / AVX2 / AVX-512; values of a segment in registers). The C
   backend shows the floor of what it buys: 690 -> 478 ms here, 215 ->
   130 ms on Seascape; keeping values in registers across blocks is
   where llvmpipe still wins. Must stay bit-identical to the interpreter.
3. **Fragment pipeline floor** (113 vs 9.5 ms for a constant shader):
   interpolate only the varyings the fragment program reads, cut the per
   batch setup of the shader stage, SIMD triangle setup / small triangle
   path, fused straight conversion + blend.
4. **Texture sampling** (~15 vs ~6 ns per trilinear sample): gather based
   SIMD trilinear with the LOD per quad, AVX-512 versions of the sampling
   and pipeline kernels (they reuse AVX2 today).

Re-measure against llvmpipe (fatgl tools/bench, Docker) after each step.

## 2D (HTML canvas API): deferred

Text is the largest gap against the canvas goal.

1. **Text.**
   * Fonts: TrueType / OpenType outlines through stb_truetype vendored
     in external/ (no system font APIs in the core); fm2d_font from
     memory or a file, `font` string parsing (size, family, weight,
     style) mapped onto loaded faces.
   * Glyphs as fm2d paths (quadratic / cubic outlines), so text gets the
     same anti-aliasing, transforms, paints and clipping as shapes;
     a glyph cache of flattened outlines per face / size / transform
     class (axis aligned scale reuses cached edges).
   * fillText / strokeText / measureText (TextMetrics: width, actual
     bounding box, font ascent / descent), textAlign, textBaseline,
     direction, letterSpacing / wordSpacing, maxWidth (horizontal
     scale).
   * Kerning from the font's kern / GPOS pair tables; no complex script
     shaping at first (a HarfBuzz hook later if needed).
   * Bench cases (glyph throughput, cached vs uncached) and profiler
     zones, like every feature.
2. **Shadows**: shadowColor, shadowBlur, shadowOffsetX / Y; render the
   shape's coverage to a scratch mask, SIMD box blur passes (three box
   passes approximate the Gaussian browsers use), composite under the
   shape.
3. **Filters** (`ctx.filter`): blur, brightness, contrast, grayscale,
   hue-rotate, invert, opacity, saturate, sepia, drop-shadow; draw into
   a layer, filter, composite. Color matrix filters as one SIMD kernel.
4. **Unbounded composite operations**: copy, source-in, source-out,
   destination-in, destination-atop clear the destination outside the
   drawn shape in browsers; fatmap only touches the shape. Needs a
   coverage pass over the whole clip region for those operators.
5. **Performance**: SIMD radial / conic gradients and the complex blend
   modes (overlay ~9 ms full screen, scalar), bilinear with wrapping;
   per draw overhead of small shapes (cached unit circle arcs, stroker
   scratch reuse instead of malloc per call, insertion sort for small
   edge lists); a per strip bucket index for command lists with
   thousands of commands.

## fatgl (games)

* Doom 3 classic: ARB_texture_env_combine / dot3 (generated ARB fragment
  programs), cube maps in fixed function texturing, two sided stencil
  (EXT_stencil_two_side / ATI_separate_stencil).
* Application supplied mipmap levels (fatmap builds them from level 0),
  border colors other than transparent black, shadow samplers, multiple
  render targets, float / sRGB render targets, MSAA, cube map arrays,
  GL_PACK_SWAP_BYTES.
* Fewer flushes per frame (buffer re-uploads), per draw overhead.
* Other operating systems.
