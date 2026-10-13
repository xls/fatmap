# Backlog

Deferred plans, so they do not get lost. Measurements behind the
performance items are in docs/PERF.md.

## 3D: ahead of Mesa llvmpipe, what is left

Whole Doom 3 BFG frames (fatgl tools/bfgscene, 1280x720) run 2 - 3x faster
than llvmpipe since 0.10.0 (docs/PERF.md, 2026-10-13). Open items:

1. **Anisotropic filtering.** llvmpipe does BFG's 8x, fatmap ignores it:
   the same image needs it, and it costs texture time (hell is ~50 %
   sampling already).
2. **Sampler setup.** ~35 % of fm3d_sample16_avx512 is LOD, levels and
   coordinates before any texel is read; samples of the same coordinates
   in one shader (BFG's bump / diffuse / specular) could share it (JIT CSE).
3. **The front end at 32 threads.** alphalabs: 2.7 ms recording (main
   thread), then vertex 1.0 + setup 1.6 ms before the 3.7 ms of tiles.
   Tried 2026-10-13: running vertex + setup of recorded draws on the
   pool's threads while recording (batches of 16 - 256 draws, 2 - 31
   threads, waking only those): the flush lost ~1.6 ms but recording took
   0.4 - 2.3 ms longer with the workers busy, no net win. Also no gain:
   vertex + setup as one job (one barrier less), the AVX-512 texel window
   for two texel rows per pixel row (hits 40 % of the misses, the rest
   span more rows).

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
