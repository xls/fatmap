/* fatmap - rasterizer internals shared with the command list executor */
#ifndef FATMAP_FM_RASTER_INTERNAL_H
#define FATMAP_FM_RASTER_INTERNAL_H

#include <fatmap/fm_raster.h>

typedef struct fm_redge {
    float x0, y0, x1, y1; /* y0 < y1, clipped */
    float dxdy;
    float dir; /* +1 / -1 */
} fm_redge;

/* Sorts the edges (once) and computes the pixel bounds. Returns 0 if empty. */
int             fm__raster_prepare(fm_rasterizer* r, int bb[4]);
const fm_redge* fm__raster_edges(const fm_rasterizer* r, int* n);
/* Render a sorted edge list (from any rasterizer) limited to rows [y0, y1),
 * using `scratch` for accumulation buffers. Output does not depend on the
 * row split: bands are aligned to absolute multiples of FM_RASTER_BAND. */
void fm__raster_render(fm_rasterizer* scratch, const fm_redge* e, int n, const int bb[4], int y0, int y1,
                       int evenodd, int aa, fm_span_fn fn, void* user);

#define FM_RASTER_BAND 16

#endif
