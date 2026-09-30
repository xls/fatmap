/*
 * fatmap - op based pixel pipeline.
 *
 * A pipeline is a chain of ops that turns coverage spans into pixels:
 *
 *   fetch (solid | linear | radial | conic | texture+sampler | custom)
 *     -> coverage (span coverage x clip mask x global alpha)
 *     -> blend op (any fm_blend_op)
 *     -> store (ARGB32 target)
 *
 * When the configuration changes the pipeline resolves the op chain once
 * into a fused, SIMD dispatched fast path (e.g. opaque solid + full coverage
 * becomes a plain fill, integer translated textures become direct blits).
 * fm_pipeline_span() has the fm_span_fn signature so a pipeline plugs
 * straight into fm_rasterizer_render() or into any custom rasterizer.
 *
 * Samplers (filter + wrap ops) are shared with the 3D renderer.
 */
#ifndef FATMAP_FM_PIPE_H
#define FATMAP_FM_PIPE_H

#include "fm_core.h"
#include "fm_raster.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 2D affine transform, canvas convention:
 *   x' = a*x + c*y + e
 *   y' = b*x + d*y + f  */
typedef struct fm_affine {
    float a, b, c, d, e, f;
} fm_affine;

FM_API fm_affine fm_affine_identity(void);
FM_API fm_affine fm_affine_mul(const fm_affine* m, const fm_affine* n); /* m * n (n applied first) */
FM_API int       fm_affine_invert(const fm_affine* m, fm_affine* out);  /* 0 if singular */

/* ---- samplers ---------------------------------------------------------- */

typedef enum fm_filter { FM_FILTER_NEAREST = 0, FM_FILTER_BILINEAR = 1 } fm_filter;

typedef enum fm_wrap {
    FM_WRAP_REPEAT = 0,
    FM_WRAP_CLAMP  = 1, /* clamp to edge */
    FM_WRAP_MIRROR = 2,
    FM_WRAP_BORDER = 3  /* transparent outside */
} fm_wrap;

typedef struct fm_sampler {
    fm_filter filter;
    fm_wrap   wrap_u;
    fm_wrap   wrap_v;
} fm_sampler;

/* Fetch n texels along an affine span, fatmap style: (u, v) in texel units
 * for the first pixel center, constant (du, dv) per pixel. Output premultiplied.
 * Texture dimensions must be <= 32767. */
FM_API void fm_sample_span(const fm_surface* tex, const fm_sampler* s, float u, float v, float du, float dv,
                           int n, uint32_t* out);

/* ---- gradients --------------------------------------------------------- */

typedef enum fm_extend { FM_EXTEND_PAD = 0, FM_EXTEND_REPEAT = 1, FM_EXTEND_REFLECT = 2 } fm_extend;

typedef struct fm_gradient fm_gradient;

#define FM_GRADIENT_LUT_SIZE 1024

/* The color LUT is rebuilt eagerly on every change, so a gradient is
 * read-only while rendering and may be shared between threads. */
FM_API fm_gradient* fm_gradient_create(void);
FM_API void         fm_gradient_destroy(fm_gradient* g);
FM_API void         fm_gradient_add_stop(fm_gradient* g, float offset, fm_color c);
FM_API void         fm_gradient_clear_stops(fm_gradient* g);
FM_API void         fm_gradient_set_extend(fm_gradient* g, fm_extend e);
FM_API int          fm_gradient_stop_count(const fm_gradient* g);
FM_API fm_extend    fm_gradient_get_extend(const fm_gradient* g);
/* FM_GRADIENT_LUT_SIZE premultiplied colors, valid until the next change. */
FM_API const uint32_t* fm_gradient_lut(const fm_gradient* g);
/* Increments on every change (for snapshot caches). */
FM_API unsigned     fm_gradient_version(const fm_gradient* g);

/* ---- pipeline ---------------------------------------------------------- */

/* Custom fetch op: produce n premultiplied pixels for device row y, x..x+n-1. */
typedef void (*fm_shade_fn)(void* user, int x, int y, int n, uint32_t* out);

typedef enum fm_source_type {
    FM_SOURCE_NONE = 0, /* transparent */
    FM_SOURCE_SOLID,
    FM_SOURCE_LINEAR,
    FM_SOURCE_RADIAL,
    FM_SOURCE_CONIC,
    FM_SOURCE_TEXTURE,
    FM_SOURCE_CUSTOM
} fm_source_type;

/* Complete, copyable description of a draw: what command lists snapshot.
 * All pointers must stay valid until the draw has executed. */
typedef struct fm_draw_state {
    fm_blend_op       op;
    float             alpha;
    const fm_surface* clip_mask; /* A8, target sized, or NULL */
    fm_source_type    source;
    fm_color          color;     /* FM_SOURCE_SOLID */
    const uint32_t*   lut;       /* gradients: FM_GRADIENT_LUT_SIZE entries */
    fm_extend         extend;
    float             params[6]; /* linear x0 y0 x1 y1 | radial x0 y0 r0 x1 y1 r1 | conic angle cx cy */
    fm_affine         xf;        /* source space -> device */
    const fm_surface* texture;
    fm_sampler        sampler;
    fm_shade_fn       custom;
    void*             custom_user;
} fm_draw_state;

FM_API void fm_draw_state_init(fm_draw_state* s); /* opaque black, source-over, alpha 1 */

typedef struct fm_pipeline fm_pipeline;

FM_API fm_pipeline* fm_pipeline_create(void);
FM_API void         fm_pipeline_destroy(fm_pipeline* p);

FM_API void fm_pipeline_set_target(fm_pipeline* p, fm_surface* dst);
FM_API void fm_pipeline_set_blend(fm_pipeline* p, fm_blend_op op);
FM_API void fm_pipeline_set_alpha(fm_pipeline* p, float alpha);
/* A8 surface with the same size as the target, or NULL. */
FM_API void fm_pipeline_set_clip_mask(fm_pipeline* p, const fm_surface* mask);

/* Fetch ops. xf maps source space to device space (NULL = identity). The
 * pipeline keeps pointers to g / tex, they must outlive the draws. */
FM_API void fm_pipeline_set_solid(fm_pipeline* p, fm_color c);
FM_API void fm_pipeline_set_linear(fm_pipeline* p, const fm_gradient* g, float x0, float y0, float x1, float y1,
                                   const fm_affine* xf);
FM_API void fm_pipeline_set_radial(fm_pipeline* p, const fm_gradient* g, float x0, float y0, float r0, float x1,
                                   float y1, float r1, const fm_affine* xf);
FM_API void fm_pipeline_set_conic(fm_pipeline* p, const fm_gradient* g, float start_angle, float cx, float cy,
                                  const fm_affine* xf);
FM_API void fm_pipeline_set_texture(fm_pipeline* p, const fm_surface* tex, const fm_sampler* s,
                                    const fm_affine* xf);
FM_API void fm_pipeline_set_custom(fm_pipeline* p, fm_shade_fn fn, void* user);
/* Transparent source: draws nothing (useful for degenerate gradients). */
FM_API void fm_pipeline_set_none(fm_pipeline* p);

FM_API void                 fm_pipeline_set_state(fm_pipeline* p, const fm_draw_state* s);
FM_API const fm_draw_state* fm_pipeline_get_state(const fm_pipeline* p);

/* fm_span_fn compatible: pass the pipeline as user pointer. */
FM_API void fm_pipeline_span(void* pipeline, int y, int x, int n, uint8_t* cov);
/* Convenience: full coverage integer rectangle [x0,x1) x [y0,y1), clipped to target. */
FM_API void fm_pipeline_fill_rect(fm_pipeline* p, int x0, int y0, int x1, int y1);

/* ---- coverage mask sink ------------------------------------------------ */
/* fm_span_fn writing coverage into an A8 surface: out = cov * in / 255
 * (in may be NULL). Used for clip masks; spans must lie inside out. */
typedef struct fm_mask_sink {
    fm_surface*       out;
    const fm_surface* in;
} fm_mask_sink;
FM_API void fm_mask_span(void* sink, int y, int x, int n, uint8_t* cov);

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_PIPE_H */
