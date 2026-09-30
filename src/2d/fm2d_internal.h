/* fatmap - 2D internals */
#ifndef FATMAP_FM2D_INTERNAL_H
#define FATMAP_FM2D_INTERNAL_H

#include <fatmap/fm2d.h>
#include "../core/fm_internal.h"
#include <math.h>

#define FM2D_PI 3.14159265358979323846f

typedef struct fm_pt {
    float x, y;
} fm_pt;

enum { FMV_MOVE = 0, FMV_LINE, FMV_CUBIC, FMV_CLOSE };

struct fm2d_path {
    uint8_t* verbs;
    int      nv, cv;
    fm_pt*   pts;
    int      np, cp;
    int      has_sub;   /* current point valid */
    int      need_move; /* after close: next segment starts with MOVE(start) */
    fm_pt    start;
};

void fmp_init(fm2d_path* p);
void fmp_free(fm2d_path* p);
void fmp_clear(fm2d_path* p);
int  fmp_copy(fm2d_path* dst, const fm2d_path* src);
void fmp_move_to(fm2d_path* p, fm_pt a);
void fmp_line_to(fm2d_path* p, fm_pt a);
void fmp_cubic_to(fm2d_path* p, fm_pt c1, fm_pt c2, fm_pt a);
void fmp_close(fm2d_path* p);
int  fmp_current(const fm2d_path* p, fm_pt* out);

/* canvas path builders; m maps user -> path space (NULL = identity) */
void fmb_move_to(fm2d_path* p, const fm_affine* m, float x, float y);
void fmb_line_to(fm2d_path* p, const fm_affine* m, float x, float y);
void fmb_quad_to(fm2d_path* p, const fm_affine* m, float cx, float cy, float x, float y);
void fmb_bezier_to(fm2d_path* p, const fm_affine* m, float c1x, float c1y, float c2x, float c2y, float x, float y);
void fmb_ellipse(fm2d_path* p, const fm_affine* m, float cx, float cy, float rx, float ry, float rot, float a0,
                 float a1, int ccw);
void fmb_arc_to(fm2d_path* p, const fm_affine* m, float x1, float y1, float x2, float y2, float r);
void fmb_rect(fm2d_path* p, const fm_affine* m, float x, float y, float w, float h);
void fmb_round_rect(fm2d_path* p, const fm_affine* m, float x, float y, float w, float h, const float* radii,
                    int count);

FM_INLINE fm_pt fm_xform(const fm_affine* m, float x, float y)
{
    fm_pt r;
    if (!m) {
        r.x = x;
        r.y = y;
    } else {
        r.x = m->a * x + m->c * y + m->e;
        r.y = m->b * x + m->d * y + m->f;
    }
    return r;
}

/* ---- flattened contours ---- */

typedef struct fm_contour {
    int   start, count, closed;
    int   has_tangents; /* t0/t1: exact unit tangents at the contour ends (curves) */
    fm_pt t0, t1;
} fm_contour;

typedef struct fm_flat {
    fm_pt*      pts;
    int         np, cp;
    fm_contour* cs;
    int         nc, cc;
} fm_flat;

void fmf_free(fm_flat* f);
void fmf_clear(fm_flat* f);
int  fmf_begin(fm_flat* f, int closed);
void fmf_add(fm_flat* f, fm_pt p);
void fmf_end(fm_flat* f); /* drops a contour with < 1 point */
/* flatten p (transformed by xf) with tolerance tol (in output units) */
void fmf_flatten(fm_flat* out, const fm2d_path* p, const fm_affine* xf, float tol);

/* ---- stroker ---- */

typedef struct fm_stroke_style {
    float        width;
    int          cap;
    int          join;
    float        miter_limit;
    const float* dash;
    int          ndash;
    float        dash_offset;
} fm_stroke_style;

/* in: flattened contours (user space). out: closed polygons, fill nonzero. */
void fms_stroke(const fm_flat* in, const fm_stroke_style* st, float tol, fm_flat* out);

/* winding number of point against closed polygons */
int fmf_winding(const fm_flat* f, float x, float y);

#endif
