/*
 * fatmap - HTML canvas style 2D context on top of fm_raster + fm_pipe.
 *
 * Every draw is described by a geometry builder (fm2d_geom) and a draw
 * state (fm_draw_state). In immediate mode (default) it is rasterized right
 * away; in deferred mode it is recorded into an fm_cmdlist and executed by
 * fm2d_flush(), optionally in parallel on an fm_executor. Both paths share
 * the same builder and state code, so they produce identical pixels.
 */
#include "fm2d_internal.h"
#include <fatmap/fm_exec.h>

enum { FMP_LINEAR = 1, FMP_RADIAL, FMP_CONIC, FMP_PATTERN };

struct fm2d_paint {
    int             refs;
    int             type;
    fm_gradient*    grad;
    float           p[6];
    fm_surface*     image;
    fm2d_repetition rep;
    fm_affine       xf;
};

typedef struct fm2d_clipstate fm2d_clipstate;
struct fm2d_clipstate {
    int             refs;
    fm_surface*     mask;      /* owned A8 mask (target sized) or NULL */
    fm2d_clipstate* mask_from; /* rect clip nested in a mask clip shares its mask */
    int             x0, y0, x1, y1;
};

typedef struct fm2d_state {
    fm_affine       ctm;
    fm_color        fill_color, stroke_color;
    fm2d_paint*     fill_paint;
    fm2d_paint*     stroke_paint;
    float           line_width, miter_limit;
    int             cap, join;
    float*          dash;
    int             ndash;
    float           dash_offset;
    float           alpha;
    fm_blend_op     op;
    int             smoothing;
    fm2d_clipstate* clip;
} fm2d_state;

typedef struct fm2d_lut_cache {
    const fm_gradient* g;
    unsigned           version;
    const uint32_t*    copy;
} fm2d_lut_cache;

#define FM2D_LUT_CACHE 8

struct fm2d_ctx {
    fm_surface*    target;
    fm2d_state     st;
    fm2d_state*    stack;
    int            nstack, cstack;
    fm2d_path      path; /* device space */
    fm2d_path      tmp;
    fm_rasterizer* rast;
    fm_pipeline*   pipe;
    fm_scratch     scratch; /* immediate mode geometry scratch */
    fm_aa_mode     aa;
    float          tol;
    /* deferred mode */
    int            deferred;
    fm_cmdlist*    list;
    fm_executor*   exec;
    fm2d_lut_cache lut_cache[FM2D_LUT_CACHE];
    int            nlut;
};

/* ---- geometry builder (runs immediately or in a worker) --------------------------- */

typedef struct fm2d_geom {
    const fm2d_path* path;
    int              stroke;
    int              has_xf;
    fm_affine        xf;  /* fill: path -> device, stroke: path -> user */
    fm_affine        ctm; /* stroke: user -> device */
    float            tol; /* fill: device units, stroke: user units */
    fm_stroke_style  ss;
} fm2d_geom;

typedef struct fm2d_scratch {
    fm_flat flat, outline;
} fm2d_scratch;

static void fm2d_scratch_free(void* p)
{
    fm2d_scratch* s = (fm2d_scratch*)p;
    fmf_free(&s->flat);
    fmf_free(&s->outline);
    free(s);
}

static void fm_add_flat(fm_rasterizer* r, const fm_flat* f, const fm_affine* xf)
{
    for (int k = 0; k < f->nc; k++) {
        const fm_pt* p = f->pts + f->cs[k].start;
        int          n = f->cs[k].count;
        if (n < 2) continue;
        fm_pt prev = fm_xform(xf, p[n - 1].x, p[n - 1].y);
        for (int i = 0; i < n; i++) {
            fm_pt q = fm_xform(xf, p[i].x, p[i].y);
            fm_rasterizer_add_line(r, prev.x, prev.y, q.x, q.y);
            prev = q;
        }
    }
}

static void fm2d_build_geom(void* data, fm_rasterizer* out, fm_scratch* sc)
{
    const fm2d_geom* g = (const fm2d_geom*)data;
    fm2d_scratch*    s = (fm2d_scratch*)sc->ptr;
    if (!s) {
        s = (fm2d_scratch*)calloc(1, sizeof(fm2d_scratch));
        if (!s) return;
        sc->ptr     = s;
        sc->free_fn = fm2d_scratch_free;
    }
    fmf_flatten(&s->flat, g->path, g->has_xf ? &g->xf : NULL, g->tol);
    if (!g->stroke) {
        fm_add_flat(out, &s->flat, NULL);
    } else {
        fms_stroke(&s->flat, &g->ss, g->tol, &s->outline);
        fm_add_flat(out, &s->outline, &g->ctm);
    }
}

/* ---- ref counted helpers ------------------------------------------------------------ */

static void fm_clip_release(fm2d_clipstate* c)
{
    if (c && --c->refs == 0) {
        fm_surface_destroy(c->mask);
        fm_clip_release(c->mask_from);
        free(c);
    }
}

static fm2d_clipstate* fm_clip_retain(fm2d_clipstate* c)
{
    if (c) c->refs++;
    return c;
}

static const fm_surface* fm_clip_mask(const fm2d_clipstate* c)
{
    if (!c) return NULL;
    return c->mask ? c->mask : (c->mask_from ? c->mask_from->mask : NULL);
}

fm2d_paint* fm2d_paint_retain(fm2d_paint* p)
{
    if (p) p->refs++;
    return p;
}

void fm2d_paint_release(fm2d_paint* p)
{
    if (p && --p->refs == 0) {
        fm_gradient_destroy(p->grad);
        fm_surface_destroy(p->image);
        free(p);
    }
}

static void fm_release_paint_cb(void* p) { fm2d_paint_release((fm2d_paint*)p); }
static void fm_release_clip_cb(void* p) { fm_clip_release((fm2d_clipstate*)p); }
static void fm_release_surface_cb(void* p) { fm_surface_destroy((fm_surface*)p); }

static void fm_state_defaults(fm2d_state* s)
{
    memset(s, 0, sizeof(*s));
    s->ctm          = fm_affine_identity();
    s->fill_color   = 0xff000000u;
    s->stroke_color = 0xff000000u;
    s->line_width   = 1.0f;
    s->miter_limit  = 10.0f;
    s->cap          = FM2D_CAP_BUTT;
    s->join         = FM2D_JOIN_MITER;
    s->alpha        = 1.0f;
    s->op           = FM_OP_SRC_OVER;
    s->smoothing    = 1;
}

static void fm_state_free(fm2d_state* s)
{
    fm2d_paint_release(s->fill_paint);
    fm2d_paint_release(s->stroke_paint);
    fm_clip_release(s->clip);
    free(s->dash);
    memset(s, 0, sizeof(*s));
}

static void fm_state_copy(fm2d_state* d, const fm2d_state* s)
{
    *d = *s;
    fm2d_paint_retain(d->fill_paint);
    fm2d_paint_retain(d->stroke_paint);
    fm_clip_retain(d->clip);
    if (s->ndash > 0) {
        d->dash = (float*)malloc((size_t)s->ndash * sizeof(float));
        if (d->dash)
            memcpy(d->dash, s->dash, (size_t)s->ndash * sizeof(float));
        else
            d->ndash = 0;
    } else {
        d->dash = NULL;
    }
}

/* ---- context ---------------------------------------------------------------------------- */

fm2d_ctx* fm2d_create(fm_surface* target)
{
    fm__init();
    fm2d_ctx* c = (fm2d_ctx*)calloc(1, sizeof(fm2d_ctx));
    if (!c) return NULL;
    c->rast = fm_rasterizer_create();
    c->pipe = fm_pipeline_create();
    c->list = fm_cmdlist_create();
    if (!c->rast || !c->pipe || !c->list) {
        fm_rasterizer_destroy(c->rast);
        fm_pipeline_destroy(c->pipe);
        fm_cmdlist_destroy(c->list);
        free(c);
        return NULL;
    }
    fmp_init(&c->path);
    fmp_init(&c->tmp);
    fm_state_defaults(&c->st);
    c->aa     = FM_AA_ANALYTIC;
    c->tol    = 0.1f;
    c->target = target;
    fm_cmdlist_reset(c->list, target);
    return c;
}

void fm2d_destroy(fm2d_ctx* c)
{
    if (!c) return;
    fm2d_flush(c);
    fm_state_free(&c->st);
    for (int i = 0; i < c->nstack; i++) fm_state_free(&c->stack[i]);
    free(c->stack);
    fmp_free(&c->path);
    fmp_free(&c->tmp);
    if (c->scratch.ptr && c->scratch.free_fn) c->scratch.free_fn(c->scratch.ptr);
    fm_cmdlist_destroy(c->list);
    fm_rasterizer_destroy(c->rast);
    fm_pipeline_destroy(c->pipe);
    free(c);
}

void fm2d_flush(fm2d_ctx* c)
{
    if (!c || fm_cmdlist_count(c->list) == 0) return;
    FM_PROF_BEGIN(z, "2d.flush");
    fm_cmdlist_execute(c->list, c->exec);
    fm_cmdlist_reset(c->list, c->target);
    c->nlut = 0;
    FM_PROF_END(z);
}

void fm2d_set_deferred(fm2d_ctx* c, int on)
{
    if (!on) fm2d_flush(c);
    c->deferred = on != 0;
}
int  fm2d_get_deferred(fm2d_ctx* c) { return c->deferred; }
void fm2d_set_executor(fm2d_ctx* c, fm_executor* ex) { c->exec = ex; }
void fm2d_set_strip_height(fm2d_ctx* c, int rows) { fm_cmdlist_set_strip_height(c->list, rows); }

void fm2d_set_target(fm2d_ctx* c, fm_surface* target)
{
    fm2d_flush(c);
    c->target = target;
    fm_cmdlist_reset(c->list, target);
    fm_clip_release(c->st.clip);
    c->st.clip = NULL;
    for (int i = 0; i < c->nstack; i++) {
        fm_clip_release(c->stack[i].clip);
        c->stack[i].clip = NULL;
    }
}

fm_surface* fm2d_get_target(fm2d_ctx* c) { return c->target; }
void        fm2d_set_antialias(fm2d_ctx* c, fm_aa_mode aa) { c->aa = aa; }
fm_aa_mode  fm2d_get_antialias(fm2d_ctx* c) { return c->aa; }
void        fm2d_set_tolerance(fm2d_ctx* c, float tol)
{
    if (tol > 0.001f && isfinite(tol)) c->tol = tol;
}

void fm2d_save(fm2d_ctx* c)
{
    if (c->nstack == c->cstack) {
        int         nc = c->cstack ? c->cstack * 2 : 8;
        fm2d_state* ns = (fm2d_state*)realloc(c->stack, (size_t)nc * sizeof(fm2d_state));
        if (!ns) return;
        c->stack  = ns;
        c->cstack = nc;
    }
    fm_state_copy(&c->stack[c->nstack++], &c->st);
}

void fm2d_restore(fm2d_ctx* c)
{
    if (c->nstack == 0) return;
    fm_state_free(&c->st);
    c->st = c->stack[--c->nstack];
}

void fm2d_reset(fm2d_ctx* c)
{
    while (c->nstack) fm2d_restore(c);
    fm_state_free(&c->st);
    fm_state_defaults(&c->st);
    fmp_clear(&c->path);
}

/* ---- transforms ----------------------------------------------------------------------------- */

static void fm_ctm_mul(fm2d_ctx* c, fm_affine m) { c->st.ctm = fm_affine_mul(&c->st.ctm, &m); }

void fm2d_translate(fm2d_ctx* c, float x, float y)
{
    fm_affine m = { 1, 0, 0, 1, x, y };
    if (isfinite(x) && isfinite(y)) fm_ctm_mul(c, m);
}
void fm2d_scale(fm2d_ctx* c, float x, float y)
{
    fm_affine m = { x, 0, 0, y, 0, 0 };
    if (isfinite(x) && isfinite(y)) fm_ctm_mul(c, m);
}
void fm2d_rotate(fm2d_ctx* c, float a)
{
    if (!isfinite(a)) return;
    float     cs = cosf(a), sn = sinf(a);
    fm_affine m  = { cs, sn, -sn, cs, 0, 0 };
    fm_ctm_mul(c, m);
}
void fm2d_transform(fm2d_ctx* c, float a, float b, float cc, float d, float e, float f)
{
    fm_affine m = { a, b, cc, d, e, f };
    if (isfinite(a + b + cc + d + e + f)) fm_ctm_mul(c, m);
}
void fm2d_set_transform(fm2d_ctx* c, float a, float b, float cc, float d, float e, float f)
{
    fm_affine m = { a, b, cc, d, e, f };
    if (isfinite(a + b + cc + d + e + f)) c->st.ctm = m;
}
void      fm2d_reset_transform(fm2d_ctx* c) { c->st.ctm = fm_affine_identity(); }
fm_affine fm2d_get_transform(fm2d_ctx* c) { return c->st.ctm; }

/* ---- styles ------------------------------------------------------------------------------------ */

void fm2d_set_global_alpha(fm2d_ctx* c, float a)
{
    if (a >= 0.0f && a <= 1.0f) c->st.alpha = a;
}
float fm2d_get_global_alpha(fm2d_ctx* c) { return c->st.alpha; }
void  fm2d_set_composite_op(fm2d_ctx* c, fm_blend_op op)
{
    if (op >= 0 && op < FM_OP_COUNT) c->st.op = op;
}
fm_blend_op fm2d_get_composite_op(fm2d_ctx* c) { return c->st.op; }
void        fm2d_set_image_smoothing(fm2d_ctx* c, int on) { c->st.smoothing = on != 0; }

void fm2d_set_fill_color(fm2d_ctx* c, fm_color col)
{
    fm2d_paint_release(c->st.fill_paint);
    c->st.fill_paint = NULL;
    c->st.fill_color = col;
}
void fm2d_set_stroke_color(fm2d_ctx* c, fm_color col)
{
    fm2d_paint_release(c->st.stroke_paint);
    c->st.stroke_paint = NULL;
    c->st.stroke_color = col;
}
void fm2d_set_fill_paint(fm2d_ctx* c, fm2d_paint* p)
{
    fm2d_paint_retain(p);
    fm2d_paint_release(c->st.fill_paint);
    c->st.fill_paint = p;
}
void fm2d_set_stroke_paint(fm2d_ctx* c, fm2d_paint* p)
{
    fm2d_paint_retain(p);
    fm2d_paint_release(c->st.stroke_paint);
    c->st.stroke_paint = p;
}
void fm2d_set_line_width(fm2d_ctx* c, float w)
{
    if (w > 0.0f && isfinite(w)) c->st.line_width = w;
}
void fm2d_set_line_cap(fm2d_ctx* c, fm2d_line_cap cap) { c->st.cap = cap; }
void fm2d_set_line_join(fm2d_ctx* c, fm2d_line_join j) { c->st.join = j; }
void fm2d_set_miter_limit(fm2d_ctx* c, float l)
{
    if (l > 0.0f && isfinite(l)) c->st.miter_limit = l;
}
void fm2d_set_line_dash(fm2d_ctx* c, const float* seg, int count)
{
    if (count < 0 || (count > 0 && !seg)) return;
    for (int i = 0; i < count; i++)
        if (!(seg[i] >= 0.0f) || !isfinite(seg[i])) return;
    free(c->st.dash);
    c->st.dash  = NULL;
    c->st.ndash = 0;
    if (count == 0) return;
    int n      = (count & 1) ? count * 2 : count;
    c->st.dash = (float*)malloc((size_t)n * sizeof(float));
    if (!c->st.dash) return;
    for (int i = 0; i < n; i++) c->st.dash[i] = seg[i % count];
    c->st.ndash = n;
}
void fm2d_set_line_dash_offset(fm2d_ctx* c, float o)
{
    if (isfinite(o)) c->st.dash_offset = o;
}

/* ---- paints ------------------------------------------------------------------------------------ */

static fm2d_paint* fm_paint_new(int type)
{
    fm__init();
    fm2d_paint* p = (fm2d_paint*)calloc(1, sizeof(fm2d_paint));
    if (!p) return NULL;
    p->refs = 1;
    p->type = type;
    p->xf   = fm_affine_identity();
    if (type != FMP_PATTERN) {
        p->grad = fm_gradient_create();
        if (!p->grad) {
            free(p);
            return NULL;
        }
    }
    return p;
}

fm2d_paint* fm2d_paint_linear(float x0, float y0, float x1, float y1)
{
    fm2d_paint* p = fm_paint_new(FMP_LINEAR);
    if (p) {
        p->p[0] = x0;
        p->p[1] = y0;
        p->p[2] = x1;
        p->p[3] = y1;
    }
    return p;
}

fm2d_paint* fm2d_paint_radial(float x0, float y0, float r0, float x1, float y1, float r1)
{
    if (r0 < 0.0f || r1 < 0.0f) return NULL;
    fm2d_paint* p = fm_paint_new(FMP_RADIAL);
    if (p) {
        p->p[0] = x0;
        p->p[1] = y0;
        p->p[2] = r0;
        p->p[3] = x1;
        p->p[4] = y1;
        p->p[5] = r1;
    }
    return p;
}

fm2d_paint* fm2d_paint_conic(float start_angle, float cx, float cy)
{
    fm2d_paint* p = fm_paint_new(FMP_CONIC);
    if (p) {
        p->p[0] = start_angle;
        p->p[1] = cx;
        p->p[2] = cy;
    }
    return p;
}

fm2d_paint* fm2d_paint_pattern(const fm_surface* image, fm2d_repetition rep)
{
    if (!image || image->format != FM_FORMAT_ARGB32) return NULL;
    fm2d_paint* p = fm_paint_new(FMP_PATTERN);
    if (!p) return NULL;
    p->image = fm_surface_clone(image);
    p->rep   = rep;
    if (!p->image) {
        free(p);
        return NULL;
    }
    return p;
}

void fm2d_paint_add_stop(fm2d_paint* p, float offset, fm_color c)
{
    if (p && p->grad) fm_gradient_add_stop(p->grad, offset, c);
}
void fm2d_paint_set_extend(fm2d_paint* p, fm_extend e)
{
    if (p && p->grad) fm_gradient_set_extend(p->grad, e);
}
void fm2d_paint_set_transform(fm2d_paint* p, const fm_affine* m)
{
    if (p) p->xf = m ? *m : fm_affine_identity();
}

/* ---- draw state + submission ------------------------------------------------------------ */

static int fm_clip_bounds(fm2d_ctx* c, int b[4])
{
    if (!c->target) return 0;
    b[0] = 0;
    b[1] = 0;
    b[2] = c->target->width;
    b[3] = c->target->height;
    if (c->st.clip) {
        b[0] = FM_MAX(b[0], c->st.clip->x0);
        b[1] = FM_MAX(b[1], c->st.clip->y0);
        b[2] = FM_MIN(b[2], c->st.clip->x1);
        b[3] = FM_MIN(b[3], c->st.clip->y1);
    }
    return b[2] > b[0] && b[3] > b[1];
}

static void fm_make_state(fm2d_ctx* c, fm2d_paint* pt, fm_color color, fm_blend_op op, float alpha,
                          fm_draw_state* s)
{
    fm_draw_state_init(s);
    s->op        = op;
    s->alpha     = alpha;
    s->clip_mask = fm_clip_mask(c->st.clip);
    if (!pt) {
        s->source = FM_SOURCE_SOLID;
        s->color  = color;
        return;
    }
    s->xf = fm_affine_mul(&c->st.ctm, &pt->xf);
    memcpy(s->params, pt->p, sizeof(s->params));
    switch (pt->type) {
    case FMP_LINEAR: s->source = FM_SOURCE_LINEAR; break;
    case FMP_RADIAL: s->source = FM_SOURCE_RADIAL; break;
    case FMP_CONIC: s->source = FM_SOURCE_CONIC; break;
    case FMP_PATTERN:
        s->source         = FM_SOURCE_TEXTURE;
        s->texture        = pt->image;
        s->sampler.filter = c->st.smoothing ? FM_FILTER_BILINEAR : FM_FILTER_NEAREST;
        s->sampler.wrap_u = (pt->rep == FM2D_REPEAT || pt->rep == FM2D_REPEAT_X) ? FM_WRAP_REPEAT : FM_WRAP_BORDER;
        s->sampler.wrap_v = (pt->rep == FM2D_REPEAT || pt->rep == FM2D_REPEAT_Y) ? FM_WRAP_REPEAT : FM_WRAP_BORDER;
        return;
    default: s->source = FM_SOURCE_NONE; return;
    }
    s->lut    = fm_gradient_lut(pt->grad);
    s->extend = fm_gradient_get_extend(pt->grad);
}

/* Deferred: make everything the state points at survive until the flush.
 * Gradient LUTs are snapshotted (so later add_stop calls do not leak into
 * recorded draws), patterns and clip masks are retained. */
static void fm_pin_state(fm2d_ctx* c, fm2d_paint* pt, fm_draw_state* s)
{
    if (c->st.clip && s->clip_mask) fm_cmdlist_defer_release(c->list, fm_release_clip_cb, fm_clip_retain(c->st.clip));
    if (!pt) return;
    if (pt->type == FMP_PATTERN) {
        fm_cmdlist_defer_release(c->list, fm_release_paint_cb, fm2d_paint_retain(pt));
        return;
    }
    unsigned ver = fm_gradient_version(pt->grad);
    for (int i = 0; i < c->nlut; i++) {
        if (c->lut_cache[i].g == pt->grad && c->lut_cache[i].version == ver) {
            s->lut = c->lut_cache[i].copy;
            return;
        }
    }
    uint32_t* copy = (uint32_t*)fm_cmdlist_alloc(c->list, FM_GRADIENT_LUT_SIZE * sizeof(uint32_t));
    if (!copy) return;
    memcpy(copy, s->lut, FM_GRADIENT_LUT_SIZE * sizeof(uint32_t));
    s->lut           = copy;
    int slot         = c->nlut < FM2D_LUT_CACHE ? c->nlut++ : (int)(ver % FM2D_LUT_CACHE);
    c->lut_cache[slot].g       = pt->grad;
    c->lut_cache[slot].version = ver;
    c->lut_cache[slot].copy    = copy;
}

/* copy a path (and dashes) into the command list arena */
static fm2d_geom* fm_geom_record(fm2d_ctx* c, const fm2d_geom* g)
{
    fm2d_geom* d  = (fm2d_geom*)fm_cmdlist_alloc(c->list, sizeof(fm2d_geom));
    fm2d_path* pc = (fm2d_path*)fm_cmdlist_alloc(c->list, sizeof(fm2d_path));
    if (!d || !pc) return NULL;
    *d = *g;
    memset(pc, 0, sizeof(*pc));
    const fm2d_path* s = g->path;
    pc->verbs          = (uint8_t*)fm_cmdlist_alloc(c->list, (size_t)FM_MAX(s->nv, 1));
    pc->pts            = (fm_pt*)fm_cmdlist_alloc(c->list, (size_t)FM_MAX(s->np, 1) * sizeof(fm_pt));
    if (!pc->verbs || !pc->pts) return NULL;
    memcpy(pc->verbs, s->verbs, (size_t)s->nv);
    memcpy(pc->pts, s->pts, (size_t)s->np * sizeof(fm_pt));
    pc->nv = pc->cv = s->nv;
    pc->np = pc->cp = s->np;
    d->path         = pc;
    if (g->stroke && g->ss.ndash > 0) {
        float* dash = (float*)fm_cmdlist_alloc(c->list, (size_t)g->ss.ndash * sizeof(float));
        if (!dash) return NULL;
        memcpy(dash, g->ss.dash, (size_t)g->ss.ndash * sizeof(float));
        d->ss.dash = dash;
    }
    return d;
}

static void fm_submit_geom(fm2d_ctx* c, const fm2d_geom* g, fm_fill_rule rule, fm2d_paint* pt, fm_color color,
                           fm_blend_op op, float alpha)
{
    int cb[4];
    if (!fm_clip_bounds(c, cb)) return;
    fm_draw_state st;
    fm_make_state(c, pt, color, op, alpha, &st);
    if (c->deferred) {
        FM_PROF_BEGIN(z, "2d.record");
        fm2d_geom* d = fm_geom_record(c, g);
        if (d) {
            fm_pin_state(c, pt, &st);
            fm_cmdlist_fill(c->list, fm2d_build_geom, d, rule, c->aa, cb, &st);
        }
        FM_PROF_END(z);
        return;
    }
    FM_PROF_BEGIN(zg, "2d.geometry");
    fm_rasterizer_reset(c->rast, cb[0], cb[1], cb[2], cb[3]);
    fm2d_build_geom((void*)g, c->rast, &c->scratch);
    FM_PROF_END(zg);
    fm_pipeline_set_target(c->pipe, c->target);
    fm_pipeline_set_state(c->pipe, &st);
    fm_rasterizer_render(c->rast, rule, c->aa, fm_pipeline_span, c->pipe);
}

static void fm_submit_rect(fm2d_ctx* c, const int r[4], const fm_draw_state* st0, fm2d_paint* pt)
{
    if (c->deferred) {
        fm_draw_state st = *st0;
        fm_pin_state(c, pt, &st);
        fm_cmdlist_fill_rect(c->list, r[0], r[1], r[2], r[3], &st);
        return;
    }
    FM_PROF_BEGIN(z, "2d.rect_fast");
    fm_pipeline_set_target(c->pipe, c->target);
    fm_pipeline_set_state(c->pipe, st0);
    fm_pipeline_fill_rect(c->pipe, r[0], r[1], r[2], r[3]);
    FM_PROF_ITEMS(z, (uint64_t)(r[2] - r[0]) * (uint64_t)(r[3] - r[1]));
    FM_PROF_END(z);
}

static float fm_ctm_scale(const fm_affine* m)
{
    float s = sqrtf(FM_MAX(m->a * m->a + m->b * m->b, m->c * m->c + m->d * m->d));
    return s > 1e-6f ? s : 1e-6f;
}

static void fm_geom_fill(fm2d_geom* g, const fm2d_path* path, const fm_affine* to_device, float tol)
{
    memset(g, 0, sizeof(*g));
    g->path   = path;
    g->has_xf = to_device != NULL;
    if (to_device) g->xf = *to_device;
    g->tol = tol;
}

/* to_user: path space -> user space (NULL = path already in user space) */
static void fm_geom_stroke(fm2d_ctx* c, fm2d_geom* g, const fm2d_path* path, const fm_affine* to_user)
{
    memset(g, 0, sizeof(*g));
    g->path           = path;
    g->stroke         = 1;
    g->has_xf         = to_user != NULL;
    if (to_user) g->xf = *to_user;
    g->ctm            = c->st.ctm;
    g->tol            = c->tol / fm_ctm_scale(&c->st.ctm);
    g->ss.width       = c->st.line_width;
    g->ss.cap         = c->st.cap;
    g->ss.join        = c->st.join;
    g->ss.miter_limit = c->st.miter_limit;
    g->ss.dash        = c->st.dash;
    g->ss.ndash       = c->st.ndash;
    g->ss.dash_offset = c->st.dash_offset;
}

static void fm_fill_impl(fm2d_ctx* c, const fm2d_path* path, const fm_affine* to_device, fm_fill_rule rule)
{
    FM_PROF_BEGIN(z, "2d.fill");
    fm2d_geom g;
    fm_geom_fill(&g, path, to_device, c->tol);
    fm_submit_geom(c, &g, rule, c->st.fill_paint, c->st.fill_color, c->st.op, c->st.alpha);
    FM_PROF_END(z);
}

static void fm_stroke_impl(fm2d_ctx* c, const fm2d_path* path, int path_is_device)
{
    FM_PROF_BEGIN(z, "2d.stroke");
    fm_affine inv;
    if (!path_is_device || fm_affine_invert(&c->st.ctm, &inv)) {
        fm2d_geom g;
        fm_geom_stroke(c, &g, path, path_is_device ? &inv : NULL);
        fm_submit_geom(c, &g, FM_FILL_NONZERO, c->st.stroke_paint, c->st.stroke_color, c->st.op, c->st.alpha);
    }
    FM_PROF_END(z);
}

/* pixel aligned axis rectangle? */
static int fm_flat_is_int_rect(const fm_flat* f, int r[4])
{
    if (f->nc != 1) return 0;
    int          n = f->cs[0].count;
    const fm_pt* p = f->pts + f->cs[0].start;
    if (n == 5 && p[4].x == p[0].x && p[4].y == p[0].y) n = 4;
    if (n != 4) return 0;
    float x0 = p[0].x, x1 = p[0].x, y0 = p[0].y, y1 = p[0].y;
    for (int i = 0; i < 4; i++) {
        if (p[i].x != floorf(p[i].x) || p[i].y != floorf(p[i].y)) return 0;
        x0 = FM_MIN(x0, p[i].x);
        x1 = FM_MAX(x1, p[i].x);
        y0 = FM_MIN(y0, p[i].y);
        y1 = FM_MAX(y1, p[i].y);
    }
    for (int i = 0; i < 4; i++) {
        fm_pt a = p[i], b = p[(i + 1) & 3];
        if (a.x != b.x && a.y != b.y) return 0;
        if ((a.x != x0 && a.x != x1) || (a.y != y0 && a.y != y1)) return 0;
    }
    r[0] = (int)x0;
    r[1] = (int)y0;
    r[2] = (int)x1;
    r[3] = (int)y1;
    return 1;
}

static void fm_clip_impl(fm2d_ctx* c, const fm2d_path* path, const fm_affine* to_device, fm_fill_rule rule)
{
    if (!c->target) return;
    FM_PROF_BEGIN(z, "2d.clip");
    int cb[4];
    int has = fm_clip_bounds(c, cb);
    fm2d_clipstate* nc = (fm2d_clipstate*)calloc(1, sizeof(fm2d_clipstate));
    if (!nc) {
        FM_PROF_END(z);
        return;
    }
    nc->refs = 1;
    if (has) {
        /* cheap path first: a pixel aligned rectangle needs no mask */
        fm2d_scratch* s = (fm2d_scratch*)c->scratch.ptr;
        if (!s) {
            s = (fm2d_scratch*)calloc(1, sizeof(fm2d_scratch));
            if (s) {
                c->scratch.ptr     = s;
                c->scratch.free_fn = fm2d_scratch_free;
            }
        }
        int rr[4];
        if (s) fmf_flatten(&s->flat, path, to_device, c->tol);
        if (s && fm_flat_is_int_rect(&s->flat, rr)) {
            nc->x0 = FM_MAX(cb[0], rr[0]);
            nc->y0 = FM_MAX(cb[1], rr[1]);
            nc->x1 = FM_MIN(cb[2], rr[2]);
            nc->y1 = FM_MIN(cb[3], rr[3]);
            if (fm_clip_mask(c->st.clip)) nc->mask_from = fm_clip_retain(c->st.clip->mask ? c->st.clip : c->st.clip->mask_from);
        } else {
            fm2d_geom g;
            fm_geom_fill(&g, path, to_device, c->tol);
            fm_rasterizer_reset(c->rast, cb[0], cb[1], cb[2], cb[3]);
            fm2d_build_geom(&g, c->rast, &c->scratch);
            int rb[4];
            if (fm_rasterizer_bounds(c->rast, rb)) {
                nc->mask = fm_surface_create(c->target->width, c->target->height, FM_FORMAT_A8);
                if (nc->mask) {
                    nc->x0 = rb[0];
                    nc->y0 = rb[1];
                    nc->x1 = rb[2];
                    nc->y1 = rb[3];
                    const fm_surface* old = fm_clip_mask(c->st.clip);
                    if (c->deferred) {
                        fm2d_geom* d = fm_geom_record(c, &g);
                        if (d) {
                            fm_cmdlist_mask(c->list, fm2d_build_geom, d, rule, c->aa, cb, nc->mask, old);
                            fm_cmdlist_defer_release(c->list, fm_release_clip_cb, fm_clip_retain(nc));
                            if (c->st.clip) fm_cmdlist_defer_release(c->list, fm_release_clip_cb, fm_clip_retain(c->st.clip));
                        }
                    } else {
                        fm_mask_sink sink = { nc->mask, old };
                        fm_rasterizer_render(c->rast, rule, c->aa, fm_mask_span, &sink);
                    }
                }
            }
        }
    }
    fm_clip_release(c->st.clip);
    c->st.clip = nc;
    FM_PROF_END(z);
}

/* ---- current path API ----------------------------------------------------------------------- */

void fm2d_begin_path(fm2d_ctx* c) { fmp_clear(&c->path); }
void fm2d_close_path(fm2d_ctx* c) { fmp_close(&c->path); }
void fm2d_move_to(fm2d_ctx* c, float x, float y) { fmb_move_to(&c->path, &c->st.ctm, x, y); }
void fm2d_line_to(fm2d_ctx* c, float x, float y) { fmb_line_to(&c->path, &c->st.ctm, x, y); }
void fm2d_quad_to(fm2d_ctx* c, float cx, float cy, float x, float y) { fmb_quad_to(&c->path, &c->st.ctm, cx, cy, x, y); }
void fm2d_bezier_to(fm2d_ctx* c, float c1x, float c1y, float c2x, float c2y, float x, float y)
{
    fmb_bezier_to(&c->path, &c->st.ctm, c1x, c1y, c2x, c2y, x, y);
}
void fm2d_arc(fm2d_ctx* c, float x, float y, float r, float a0, float a1, int ccw)
{
    fmb_ellipse(&c->path, &c->st.ctm, x, y, r, r, 0.0f, a0, a1, ccw);
}
void fm2d_arc_to(fm2d_ctx* c, float x1, float y1, float x2, float y2, float r)
{
    fmb_arc_to(&c->path, &c->st.ctm, x1, y1, x2, y2, r);
}
void fm2d_ellipse(fm2d_ctx* c, float x, float y, float rx, float ry, float rot, float a0, float a1, int ccw)
{
    fmb_ellipse(&c->path, &c->st.ctm, x, y, rx, ry, rot, a0, a1, ccw);
}
void fm2d_rect(fm2d_ctx* c, float x, float y, float w, float h) { fmb_rect(&c->path, &c->st.ctm, x, y, w, h); }
void fm2d_round_rect(fm2d_ctx* c, float x, float y, float w, float h, const float* radii, int count)
{
    fmb_round_rect(&c->path, &c->st.ctm, x, y, w, h, radii, count);
}

void fm2d_fill(fm2d_ctx* c, fm_fill_rule rule) { fm_fill_impl(c, &c->path, NULL, rule); }
void fm2d_stroke(fm2d_ctx* c) { fm_stroke_impl(c, &c->path, 1); }
void fm2d_clip(fm2d_ctx* c, fm_fill_rule rule) { fm_clip_impl(c, &c->path, NULL, rule); }
void fm2d_fill_path(fm2d_ctx* c, const fm2d_path* p, fm_fill_rule rule)
{
    if (p) fm_fill_impl(c, p, &c->st.ctm, rule);
}
void fm2d_stroke_path(fm2d_ctx* c, const fm2d_path* p)
{
    if (p) fm_stroke_impl(c, p, 0);
}
void fm2d_clip_path(fm2d_ctx* c, const fm2d_path* p, fm_fill_rule rule)
{
    if (p) fm_clip_impl(c, p, &c->st.ctm, rule);
}

/* integer device rect for an axis aligned user rect, if exact */
static int fm_int_device_rect(const fm_affine* m, float x, float y, float w, float h, int r[4])
{
    if (m->b != 0.0f || m->c != 0.0f) return 0;
    float x0 = m->a * x + m->e, x1 = m->a * (x + w) + m->e;
    float y0 = m->d * y + m->f, y1 = m->d * (y + h) + m->f;
    if (x0 > x1) {
        float t = x0;
        x0      = x1;
        x1      = t;
    }
    if (y0 > y1) {
        float t = y0;
        y0      = y1;
        y1      = t;
    }
    if (x0 != floorf(x0) || x1 != floorf(x1) || y0 != floorf(y0) || y1 != floorf(y1)) return 0;
    if (fabsf(x0) > 1e8f || fabsf(x1) > 1e8f || fabsf(y0) > 1e8f || fabsf(y1) > 1e8f) return 0;
    r[0] = (int)x0;
    r[1] = (int)y0;
    r[2] = (int)x1;
    r[3] = (int)y1;
    return 1;
}

/* draw a user space rect with a prepared state (fast path for pixel aligned) */
static void fm_rect_draw(fm2d_ctx* c, float x, float y, float w, float h, const fm_draw_state* st, fm2d_paint* pt)
{
    int cb[4], r[4];
    if (!fm_clip_bounds(c, cb)) return;
    if (fm_int_device_rect(&c->st.ctm, x, y, w, h, r)) {
        r[0] = FM_MAX(r[0], cb[0]);
        r[1] = FM_MAX(r[1], cb[1]);
        r[2] = FM_MIN(r[2], cb[2]);
        r[3] = FM_MIN(r[3], cb[3]);
        if (r[2] > r[0] && r[3] > r[1]) fm_submit_rect(c, r, st, pt);
        return;
    }
    fmp_clear(&c->tmp);
    fmb_rect(&c->tmp, &c->st.ctm, x, y, w, h);
    fm2d_geom g;
    fm_geom_fill(&g, &c->tmp, NULL, c->tol);
    if (c->deferred) {
        fm2d_geom* d = fm_geom_record(c, &g);
        if (!d) return;
        fm_draw_state s2 = *st;
        fm_pin_state(c, pt, &s2);
        fm_cmdlist_fill(c->list, fm2d_build_geom, d, FM_FILL_NONZERO, c->aa, cb, &s2);
        return;
    }
    FM_PROF_BEGIN(z, "2d.rect");
    fm_rasterizer_reset(c->rast, cb[0], cb[1], cb[2], cb[3]);
    fm2d_build_geom(&g, c->rast, &c->scratch);
    fm_pipeline_set_target(c->pipe, c->target);
    fm_pipeline_set_state(c->pipe, st);
    fm_rasterizer_render(c->rast, FM_FILL_NONZERO, c->aa, fm_pipeline_span, c->pipe);
    FM_PROF_END(z);
}

void fm2d_fill_rect(fm2d_ctx* c, float x, float y, float w, float h)
{
    if (!isfinite(x + y + w + h) || w == 0.0f || h == 0.0f) return;
    fm_draw_state st;
    fm_make_state(c, c->st.fill_paint, c->st.fill_color, c->st.op, c->st.alpha, &st);
    fm_rect_draw(c, x, y, w, h, &st, c->st.fill_paint);
}

void fm2d_clear_rect(fm2d_ctx* c, float x, float y, float w, float h)
{
    if (!isfinite(x + y + w + h) || w == 0.0f || h == 0.0f) return;
    fm_draw_state st;
    fm_make_state(c, NULL, 0xff000000u, FM_OP_CLEAR, 1.0f, &st);
    fm_rect_draw(c, x, y, w, h, &st, NULL);
}

void fm2d_clear(fm2d_ctx* c, fm_color color)
{
    if (!c->target) return;
    fm_draw_state st;
    fm_draw_state_init(&st);
    st.op    = FM_OP_COPY;
    st.color = color;
    int r[4] = { 0, 0, c->target->width, c->target->height };
    fm_submit_rect(c, r, &st, NULL);
}

void fm2d_stroke_rect(fm2d_ctx* c, float x, float y, float w, float h)
{
    if (!isfinite(x + y + w + h)) return;
    fmp_clear(&c->tmp);
    if (w == 0.0f || h == 0.0f) {
        fmb_move_to(&c->tmp, NULL, x, y);
        fmb_line_to(&c->tmp, NULL, x + w, y + h);
    } else {
        fmb_rect(&c->tmp, NULL, x, y, w, h);
    }
    fm_stroke_impl(c, &c->tmp, 0);
}

int fm2d_is_point_in_path(fm2d_ctx* c, float x, float y, fm_fill_rule rule)
{
    fm_flat f;
    memset(&f, 0, sizeof(f));
    fmf_flatten(&f, &c->path, NULL, c->tol);
    int w = fmf_winding(&f, x, y);
    fmf_free(&f);
    return rule == FM_FILL_EVENODD ? (w & 1) != 0 : w != 0;
}

int fm2d_is_point_in_stroke(fm2d_ctx* c, float x, float y)
{
    fm_affine inv;
    if (!fm_affine_invert(&c->st.ctm, &inv)) return 0;
    fm2d_geom g;
    fm_geom_stroke(c, &g, &c->path, &inv);
    fm_flat f, o;
    memset(&f, 0, sizeof(f));
    memset(&o, 0, sizeof(o));
    fmf_flatten(&f, &c->path, &inv, g.tol);
    fms_stroke(&f, &g.ss, g.tol, &o);
    fm_pt u = fm_xform(&inv, x, y);
    int   w = fmf_winding(&o, u.x, u.y);
    fmf_free(&f);
    fmf_free(&o);
    return w != 0;
}

/* ---- images ---------------------------------------------------------------------------------- */

static int fm_aliases_target(fm2d_ctx* c, const fm_surface* img)
{
    const fm_surface* t = c->target;
    if (!t || !img) return 0;
    const uint8_t* a  = (const uint8_t*)t->data;
    const uint8_t* ae = a + (size_t)t->stride * (size_t)t->height;
    const uint8_t* b  = (const uint8_t*)img->data;
    return b >= a && b < ae;
}

void fm2d_draw_image_sub(fm2d_ctx* c, const fm_surface* img, float sx, float sy, float sw, float sh, float dx,
                         float dy, float dw, float dh)
{
    if (!img || img->format != FM_FORMAT_ARGB32 || !isfinite(sx + sy + sw + sh + dx + dy + dw + dh)) return;
    if (sw < 0) { sx += sw; sw = -sw; }
    if (sh < 0) { sy += sh; sh = -sh; }
    if (dw < 0) { dx += dw; dw = -dw; }
    if (dh < 0) { dy += dh; dh = -dh; }
    if (sw == 0 || sh == 0 || dw == 0 || dh == 0) return;
    /* clip the source rect to the image, shrink the destination in proportion */
    float kx = dw / sw, ky = dh / sh;
    if (sx < 0) { dx -= sx * kx; dw += sx * kx; sw += sx; sx = 0; }
    if (sy < 0) { dy -= sy * ky; dh += sy * ky; sh += sy; sy = 0; }
    if (sx + sw > (float)img->width) { float o = sx + sw - (float)img->width; sw -= o; dw -= o * kx; }
    if (sy + sh > (float)img->height) { float o = sy + sh - (float)img->height; sh -= o; dh -= o * ky; }
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    int cb[4];
    if (!fm_clip_bounds(c, cb)) return;
    /* reading the target while draws into it are pending: resolve them first */
    if (c->deferred && fm_aliases_target(c, img)) fm2d_flush(c);

    FM_PROF_BEGIN(z, "2d.draw_image");
    int         ix0 = (int)floorf(sx), iy0 = (int)floorf(sy);
    int         ix1 = (int)ceilf(sx + sw), iy1 = (int)ceilf(sy + sh);
    fm_surface* view = fm_surface_sub(img, ix0, iy0, ix1 - ix0, iy1 - iy0);
    if (!view) {
        FM_PROF_END(z);
        return;
    }
    /* view -> device: ctm * T(dx,dy) * S(kx,ky) * T(ix0 - sx, iy0 - sy) */
    fm_affine     a = { kx, 0, 0, ky, dx + (ix0 - sx) * kx, dy + (iy0 - sy) * ky };
    fm_draw_state st;
    fm_make_state(c, NULL, 0, c->st.op, c->st.alpha, &st);
    st.source         = FM_SOURCE_TEXTURE;
    st.texture        = view;
    st.xf             = fm_affine_mul(&c->st.ctm, &a);
    st.sampler.filter = c->st.smoothing ? FM_FILTER_BILINEAR : FM_FILTER_NEAREST;
    st.sampler.wrap_u = st.sampler.wrap_v = FM_WRAP_CLAMP;
    fm_rect_draw(c, dx, dy, dw, dh, &st, NULL);
    if (c->deferred)
        fm_cmdlist_defer_release(c->list, fm_release_surface_cb, view);
    else
        fm_surface_destroy(view);
    FM_PROF_END(z);
}

void fm2d_draw_image(fm2d_ctx* c, const fm_surface* img, float dx, float dy)
{
    if (!img) return;
    fm2d_draw_image_sub(c, img, 0, 0, (float)img->width, (float)img->height, dx, dy, (float)img->width,
                        (float)img->height);
}

void fm2d_draw_image_scaled(fm2d_ctx* c, const fm_surface* img, float dx, float dy, float dw, float dh)
{
    if (!img) return;
    fm2d_draw_image_sub(c, img, 0, 0, (float)img->width, (float)img->height, dx, dy, dw, dh);
}

/* ---- Path2D ------------------------------------------------------------------------------------ */

fm2d_path* fm2d_path_create(void)
{
    fm2d_path* p = (fm2d_path*)calloc(1, sizeof(fm2d_path));
    return p;
}

fm2d_path* fm2d_path_clone(const fm2d_path* src)
{
    fm2d_path* p = fm2d_path_create();
    if (p && src && !fmp_copy(p, src)) {
        fm2d_path_destroy(p);
        return NULL;
    }
    return p;
}

void fm2d_path_destroy(fm2d_path* p)
{
    if (!p) return;
    fmp_free(p);
    free(p);
}

void fm2d_path_add_path(fm2d_path* p, const fm2d_path* o, const fm_affine* m)
{
    if (!p || !o) return;
    int pi = 0;
    for (int v = 0; v < o->nv; v++) {
        switch (o->verbs[v]) {
        case FMV_MOVE: fmp_move_to(p, fm_xform(m, o->pts[pi].x, o->pts[pi].y)); pi++; break;
        case FMV_LINE: fmp_line_to(p, fm_xform(m, o->pts[pi].x, o->pts[pi].y)); pi++; break;
        case FMV_CUBIC:
            fmp_cubic_to(p, fm_xform(m, o->pts[pi].x, o->pts[pi].y), fm_xform(m, o->pts[pi + 1].x, o->pts[pi + 1].y),
                         fm_xform(m, o->pts[pi + 2].x, o->pts[pi + 2].y));
            pi += 3;
            break;
        case FMV_CLOSE: fmp_close(p); break;
        }
    }
}

void fm2d_path_close(fm2d_path* p) { fmp_close(p); }
void fm2d_path_move_to(fm2d_path* p, float x, float y) { fmb_move_to(p, NULL, x, y); }
void fm2d_path_line_to(fm2d_path* p, float x, float y) { fmb_line_to(p, NULL, x, y); }
void fm2d_path_quad_to(fm2d_path* p, float cx, float cy, float x, float y) { fmb_quad_to(p, NULL, cx, cy, x, y); }
void fm2d_path_bezier_to(fm2d_path* p, float c1x, float c1y, float c2x, float c2y, float x, float y)
{
    fmb_bezier_to(p, NULL, c1x, c1y, c2x, c2y, x, y);
}
void fm2d_path_arc(fm2d_path* p, float x, float y, float r, float a0, float a1, int ccw)
{
    fmb_ellipse(p, NULL, x, y, r, r, 0.0f, a0, a1, ccw);
}
void fm2d_path_arc_to(fm2d_path* p, float x1, float y1, float x2, float y2, float r)
{
    fmb_arc_to(p, NULL, x1, y1, x2, y2, r);
}
void fm2d_path_ellipse(fm2d_path* p, float x, float y, float rx, float ry, float rot, float a0, float a1, int ccw)
{
    fmb_ellipse(p, NULL, x, y, rx, ry, rot, a0, a1, ccw);
}
void fm2d_path_rect(fm2d_path* p, float x, float y, float w, float h) { fmb_rect(p, NULL, x, y, w, h); }
void fm2d_path_round_rect(fm2d_path* p, float x, float y, float w, float h, const float* radii, int count)
{
    fmb_round_rect(p, NULL, x, y, w, h, radii, count);
}
