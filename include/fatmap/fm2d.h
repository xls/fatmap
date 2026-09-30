/*
 * fatmap - 2D vector API modelled on the HTML canvas 2D context.
 *
 * Built on the core rasterizer (fm_raster.h) and pixel pipeline (fm_pipe.h).
 * Coordinates follow canvas semantics: path points are transformed by the
 * current transform when they are added; line width, dashes and paints use
 * the transform at fill/stroke time.
 */
#ifndef FATMAP_FM2D_H
#define FATMAP_FM2D_H

#include "fm_core.h"
#include "fm_raster.h"
#include "fm_pipe.h"
#include "fm_exec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fm2d_ctx   fm2d_ctx;
typedef struct fm2d_path  fm2d_path;  /* Path2D */
typedef struct fm2d_paint fm2d_paint; /* CanvasGradient / CanvasPattern */

typedef enum fm2d_line_cap { FM2D_CAP_BUTT = 0, FM2D_CAP_ROUND, FM2D_CAP_SQUARE } fm2d_line_cap;
typedef enum fm2d_line_join { FM2D_JOIN_MITER = 0, FM2D_JOIN_ROUND, FM2D_JOIN_BEVEL } fm2d_line_join;
typedef enum fm2d_repetition {
    FM2D_REPEAT = 0,
    FM2D_REPEAT_X,
    FM2D_REPEAT_Y,
    FM2D_NO_REPEAT
} fm2d_repetition;

/* ---- context ------------------------------------------------------------ */

FM_API fm2d_ctx*   fm2d_create(fm_surface* target);
FM_API void        fm2d_destroy(fm2d_ctx* ctx);
FM_API void        fm2d_set_target(fm2d_ctx* ctx, fm_surface* target); /* also resets clip */
FM_API fm_surface* fm2d_get_target(fm2d_ctx* ctx);

/* Rasterizer anti-aliasing for this context (default FM_AA_ANALYTIC). */
FM_API void       fm2d_set_antialias(fm2d_ctx* ctx, fm_aa_mode aa);
FM_API fm_aa_mode fm2d_get_antialias(fm2d_ctx* ctx);
/* Deferred rendering: draws are recorded into a command list and executed
 * by fm2d_flush() (on the executor if set, else serially). Pixels in the
 * target are only valid after a flush. Images drawn in deferred mode must
 * not change until the flush (like GPU APIs). Output is identical to
 * immediate mode for any thread count. */
FM_API void fm2d_set_deferred(fm2d_ctx* ctx, int on); /* turning it off flushes */
FM_API int  fm2d_get_deferred(fm2d_ctx* ctx);
FM_API void fm2d_set_executor(fm2d_ctx* ctx, fm_executor* ex); /* NULL = serial */
FM_API void fm2d_set_strip_height(fm2d_ctx* ctx, int rows);   /* default 32 */
FM_API void fm2d_flush(fm2d_ctx* ctx);
/* Fill the whole target with c (ignores transform, clip, alpha and op). */
FM_API void fm2d_clear(fm2d_ctx* ctx, fm_color c);

/* Curve flattening tolerance in device pixels (default 0.1). */
FM_API void       fm2d_set_tolerance(fm2d_ctx* ctx, float tol);

/* ---- state -------------------------------------------------------------- */

FM_API void fm2d_save(fm2d_ctx* ctx);
FM_API void fm2d_restore(fm2d_ctx* ctx);
FM_API void fm2d_reset(fm2d_ctx* ctx); /* ctx.reset(): state, path, clip (not pixels) */

/* ---- transforms --------------------------------------------------------- */

FM_API void      fm2d_translate(fm2d_ctx* ctx, float x, float y);
FM_API void      fm2d_scale(fm2d_ctx* ctx, float x, float y);
FM_API void      fm2d_rotate(fm2d_ctx* ctx, float radians);
FM_API void      fm2d_transform(fm2d_ctx* ctx, float a, float b, float c, float d, float e, float f);
FM_API void      fm2d_set_transform(fm2d_ctx* ctx, float a, float b, float c, float d, float e, float f);
FM_API void      fm2d_reset_transform(fm2d_ctx* ctx);
FM_API fm_affine fm2d_get_transform(fm2d_ctx* ctx);

/* ---- compositing -------------------------------------------------------- */

FM_API void        fm2d_set_global_alpha(fm2d_ctx* ctx, float alpha);
FM_API float       fm2d_get_global_alpha(fm2d_ctx* ctx);
FM_API void        fm2d_set_composite_op(fm2d_ctx* ctx, fm_blend_op op);
FM_API fm_blend_op fm2d_get_composite_op(fm2d_ctx* ctx);
FM_API void        fm2d_set_image_smoothing(fm2d_ctx* ctx, int enabled);

/* ---- styles ------------------------------------------------------------- */

FM_API void fm2d_set_fill_color(fm2d_ctx* ctx, fm_color c);
FM_API void fm2d_set_stroke_color(fm2d_ctx* ctx, fm_color c);
FM_API void fm2d_set_fill_paint(fm2d_ctx* ctx, fm2d_paint* paint);   /* retains */
FM_API void fm2d_set_stroke_paint(fm2d_ctx* ctx, fm2d_paint* paint); /* retains */
FM_API void fm2d_set_line_width(fm2d_ctx* ctx, float w);
FM_API void fm2d_set_line_cap(fm2d_ctx* ctx, fm2d_line_cap cap);
FM_API void fm2d_set_line_join(fm2d_ctx* ctx, fm2d_line_join join);
FM_API void fm2d_set_miter_limit(fm2d_ctx* ctx, float limit);
FM_API void fm2d_set_line_dash(fm2d_ctx* ctx, const float* segments, int count);
FM_API void fm2d_set_line_dash_offset(fm2d_ctx* ctx, float offset);

/* ---- paints (gradients and patterns) ------------------------------------ */

FM_API fm2d_paint* fm2d_paint_linear(float x0, float y0, float x1, float y1);
FM_API fm2d_paint* fm2d_paint_radial(float x0, float y0, float r0, float x1, float y1, float r1);
FM_API fm2d_paint* fm2d_paint_conic(float start_angle, float cx, float cy);
/* Copies the image (like createPattern snapshots it). */
FM_API fm2d_paint* fm2d_paint_pattern(const fm_surface* image, fm2d_repetition rep);
FM_API void        fm2d_paint_add_stop(fm2d_paint* p, float offset, fm_color c);
FM_API void        fm2d_paint_set_extend(fm2d_paint* p, fm_extend e); /* extension: gradient spread */
FM_API void        fm2d_paint_set_transform(fm2d_paint* p, const fm_affine* m);
FM_API fm2d_paint* fm2d_paint_retain(fm2d_paint* p);
FM_API void        fm2d_paint_release(fm2d_paint* p);

/* ---- current path ------------------------------------------------------- */

FM_API void fm2d_begin_path(fm2d_ctx* ctx);
FM_API void fm2d_close_path(fm2d_ctx* ctx);
FM_API void fm2d_move_to(fm2d_ctx* ctx, float x, float y);
FM_API void fm2d_line_to(fm2d_ctx* ctx, float x, float y);
FM_API void fm2d_quad_to(fm2d_ctx* ctx, float cpx, float cpy, float x, float y);
FM_API void fm2d_bezier_to(fm2d_ctx* ctx, float c1x, float c1y, float c2x, float c2y, float x, float y);
FM_API void fm2d_arc(fm2d_ctx* ctx, float x, float y, float r, float a0, float a1, int ccw);
FM_API void fm2d_arc_to(fm2d_ctx* ctx, float x1, float y1, float x2, float y2, float r);
FM_API void fm2d_ellipse(fm2d_ctx* ctx, float x, float y, float rx, float ry, float rotation, float a0, float a1,
                         int ccw);
FM_API void fm2d_rect(fm2d_ctx* ctx, float x, float y, float w, float h);
/* radii: 1..4 corner radii (canvas order), NULL = 0 */
FM_API void fm2d_round_rect(fm2d_ctx* ctx, float x, float y, float w, float h, const float* radii, int count);

/* ---- drawing ------------------------------------------------------------ */

FM_API void fm2d_fill(fm2d_ctx* ctx, fm_fill_rule rule);
FM_API void fm2d_stroke(fm2d_ctx* ctx);
FM_API void fm2d_clip(fm2d_ctx* ctx, fm_fill_rule rule);
FM_API void fm2d_fill_path(fm2d_ctx* ctx, const fm2d_path* path, fm_fill_rule rule);
FM_API void fm2d_stroke_path(fm2d_ctx* ctx, const fm2d_path* path);
FM_API void fm2d_clip_path(fm2d_ctx* ctx, const fm2d_path* path, fm_fill_rule rule);
FM_API void fm2d_fill_rect(fm2d_ctx* ctx, float x, float y, float w, float h);
FM_API void fm2d_stroke_rect(fm2d_ctx* ctx, float x, float y, float w, float h);
FM_API void fm2d_clear_rect(fm2d_ctx* ctx, float x, float y, float w, float h);
FM_API int  fm2d_is_point_in_path(fm2d_ctx* ctx, float x, float y, fm_fill_rule rule);
FM_API int  fm2d_is_point_in_stroke(fm2d_ctx* ctx, float x, float y);

FM_API void fm2d_draw_image(fm2d_ctx* ctx, const fm_surface* img, float dx, float dy);
FM_API void fm2d_draw_image_scaled(fm2d_ctx* ctx, const fm_surface* img, float dx, float dy, float dw, float dh);
FM_API void fm2d_draw_image_sub(fm2d_ctx* ctx, const fm_surface* img, float sx, float sy, float sw, float sh,
                                float dx, float dy, float dw, float dh);

/* ---- Path2D --------------------------------------------------------------- */

FM_API fm2d_path* fm2d_path_create(void);
FM_API fm2d_path* fm2d_path_clone(const fm2d_path* p);
FM_API void       fm2d_path_destroy(fm2d_path* p);
FM_API void       fm2d_path_add_path(fm2d_path* p, const fm2d_path* other, const fm_affine* m);
FM_API void       fm2d_path_close(fm2d_path* p);
FM_API void       fm2d_path_move_to(fm2d_path* p, float x, float y);
FM_API void       fm2d_path_line_to(fm2d_path* p, float x, float y);
FM_API void       fm2d_path_quad_to(fm2d_path* p, float cpx, float cpy, float x, float y);
FM_API void       fm2d_path_bezier_to(fm2d_path* p, float c1x, float c1y, float c2x, float c2y, float x, float y);
FM_API void       fm2d_path_arc(fm2d_path* p, float x, float y, float r, float a0, float a1, int ccw);
FM_API void       fm2d_path_arc_to(fm2d_path* p, float x1, float y1, float x2, float y2, float r);
FM_API void       fm2d_path_ellipse(fm2d_path* p, float x, float y, float rx, float ry, float rotation, float a0,
                                    float a1, int ccw);
FM_API void       fm2d_path_rect(fm2d_path* p, float x, float y, float w, float h);
FM_API void       fm2d_path_round_rect(fm2d_path* p, float x, float y, float w, float h, const float* radii,
                                       int count);

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM2D_H */
