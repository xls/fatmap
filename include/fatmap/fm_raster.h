/*
 * fatmap - core coverage rasterizer.
 *
 * Feed it line segments (any orientation, device coordinates, float), then
 * render with a fill rule and an anti-aliasing mode. Output is delivered as
 * horizontal spans to a callback, which is normally an fm_pipeline but can
 * be anything (a GL/DX style backend, a mask writer, a hit tester...).
 *
 * AA modes:
 *   FM_AA_ANALYTIC  exact area coverage (HTML canvas quality). Interior runs
 *                   are delivered with cov == NULL so sinks can take
 *                   full-coverage fast paths.
 *   FM_AA_NONE      point sampling at pixel centers, top-left rule
 *                   (OpenGL / Direct3D rasterization rules). Always cov == NULL.
 *
 * Sub-paths are implicitly closed by the caller: the rasterizer only sums
 * signed edges, so every contour you add must be closed.
 */
#ifndef FATMAP_FM_RASTER_H
#define FATMAP_FM_RASTER_H

#include "fm_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum fm_fill_rule { FM_FILL_NONZERO = 0, FM_FILL_EVENODD = 1 } fm_fill_rule;
typedef enum fm_aa_mode { FM_AA_NONE = 0, FM_AA_ANALYTIC = 1 } fm_aa_mode;

/* cov: n coverage bytes (0..255), or NULL for full coverage. The sink may
 * modify cov in place. */
typedef void (*fm_span_fn)(void* user, int y, int x, int n, uint8_t* cov);

typedef struct fm_rasterizer fm_rasterizer;

FM_API fm_rasterizer* fm_rasterizer_create(void);
FM_API void           fm_rasterizer_destroy(fm_rasterizer* r);
/* Clears edges and sets the integer clip rectangle [x0,x1) x [y0,y1). */
FM_API void fm_rasterizer_reset(fm_rasterizer* r, int x0, int y0, int x1, int y1);
FM_API void fm_rasterizer_add_line(fm_rasterizer* r, float x0, float y0, float x1, float y1);
/* Closed polygon from count (x,y) pairs. */
FM_API void fm_rasterizer_add_polygon(fm_rasterizer* r, const float* xy, int count);
FM_API int  fm_rasterizer_edge_count(const fm_rasterizer* r);
/* Integer pixel bounds of the added geometry clipped to the clip rect.
 * Returns 0 if empty. out = {x0, y0, x1, y1}. */
FM_API int  fm_rasterizer_bounds(const fm_rasterizer* r, int out[4]);
FM_API void fm_rasterizer_render(fm_rasterizer* r, fm_fill_rule rule, fm_aa_mode aa, fm_span_fn fn, void* user);
/* Render only rows [y0, y1). Output is identical to a full render, so a
 * frame can be split into row bands (e.g. one per thread). Edges are kept
 * until the next reset, so a rasterizer can be rendered repeatedly. */
FM_API void fm_rasterizer_render_rows(fm_rasterizer* r, int y0, int y1, fm_fill_rule rule, fm_aa_mode aa,
                                      fm_span_fn fn, void* user);

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_RASTER_H */
