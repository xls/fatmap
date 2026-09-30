/*
 * fatmap - op based pixel pipeline: fetch -> coverage -> blend -> store.
 *
 * fm_pipe_prepare() resolves the op chain once per configuration into one
 * fused "run" function; the per-span path is then branch-light and SIMD
 * dispatched through fm_k. Spans are processed in L1 sized chunks.
 */
#include <fatmap/fm_pipe.h>
#include "fm_kernels_common.h"
#include <math.h>

#define FM_PIPE_CHUNK 256

/* ---- affine ------------------------------------------------------------------ */

fm_affine fm_affine_identity(void)
{
    fm_affine m = { 1, 0, 0, 1, 0, 0 };
    return m;
}

fm_affine fm_affine_mul(const fm_affine* m, const fm_affine* n)
{
    fm_affine r;
    r.a = m->a * n->a + m->c * n->b;
    r.b = m->b * n->a + m->d * n->b;
    r.c = m->a * n->c + m->c * n->d;
    r.d = m->b * n->c + m->d * n->d;
    r.e = m->a * n->e + m->c * n->f + m->e;
    r.f = m->b * n->e + m->d * n->f + m->f;
    return r;
}

int fm_affine_invert(const fm_affine* m, fm_affine* out)
{
    double det = (double)m->a * m->d - (double)m->b * m->c;
    if (det == 0.0 || !isfinite(det)) return 0;
    double    id = 1.0 / det;
    fm_affine r;
    r.a  = (float)(m->d * id);
    r.b  = (float)(-m->b * id);
    r.c  = (float)(-m->c * id);
    r.d  = (float)(m->a * id);
    r.e  = (float)(((double)m->c * m->f - (double)m->d * m->e) * id);
    r.f  = (float)(((double)m->b * m->e - (double)m->a * m->f) * id);
    *out = r;
    return 1;
}

/* ---- samplers ------------------------------------------------------------------ */

FM_INLINE int fm_wrap_coord(int i, int n, fm_wrap w)
{
    switch (w) {
    case FM_WRAP_REPEAT:
        i %= n;
        return i < 0 ? i + n : i;
    case FM_WRAP_CLAMP: return i < 0 ? 0 : (i >= n ? n - 1 : i);
    case FM_WRAP_MIRROR: {
        int p = 2 * n;
        i %= p;
        if (i < 0) i += p;
        return i < n ? i : p - 1 - i;
    }
    default: return (i < 0 || i >= n) ? -1 : i;
    }
}

FM_INLINE int32_t fm_fx_floor(int64_t v) { return (int32_t)(v >> 16); }

FM_INLINE int64_t fm_to_fixed(float f)
{
    double d = (double)f * 65536.0;
    if (d > 1e15) d = 1e15;
    if (d < -1e15) d = -1e15;
    return (int64_t)floor(d + 0.5);
}

/* keep integer coordinates in a range where % is safe */
FM_INLINE int fm_icoord(int64_t v)
{
    int64_t i = v >> 16;
    if (i > (1 << 30)) i = 1 << 30;
    if (i < -(1 << 30)) i = -(1 << 30);
    return (int)i;
}

FM_INLINE int64_t fm_floor_div(int64_t a, int64_t b) /* b > 0 */
{
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

/* indices i in [0, n) with lo <= U + i * D < hi, as [*a, *b) */
static void fm_axis_range(int64_t U, int64_t D, int64_t lo, int64_t hi, int n, int* a, int* b)
{
    int64_t ia, ib;
    if (D == 0) {
        ia = 0;
        ib = (U >= lo && U < hi) ? n : 0;
    } else if (D > 0) {
        ia = fm_floor_div(lo - U + D - 1, D); /* ceil */
        ib = fm_floor_div(hi - U + D - 1, D);
    } else {
        int64_t d = -D;
        ia = fm_floor_div(U - hi, d) + 1;
        ib = fm_floor_div(U - lo, d) + 1;
    }
    ia = FM_CLAMP(ia, 0, (int64_t)n);
    ib = FM_CLAMP(ib, 0, (int64_t)n);
    *a = (int)ia;
    *b = (int)FM_MAX(ia, ib);
}

/* general path: per pixel wrapping / border handling */
static void fm_sample_nearest_wrap(const fm_surface* tex, const fm_sampler* s, int64_t U, int64_t V, int64_t DU,
                                   int64_t DV, int n, uint32_t* out)
{
    const int       w = tex->width, h = tex->height, stride = tex->stride >> 2;
    const uint32_t* base = (const uint32_t*)tex->data;
    for (int i = 0; i < n; i++) {
        int x  = fm_wrap_coord(fm_icoord(U), w, s->wrap_u);
        int y  = fm_wrap_coord(fm_icoord(V), h, s->wrap_v);
        out[i] = (x < 0 || y < 0) ? 0u : base[(ptrdiff_t)y * stride + x];
        U += DU;
        V += DV;
    }
}

static void fm_sample_bilinear_wrap(const fm_surface* tex, const fm_sampler* s, int64_t U, int64_t V, int64_t DU,
                                    int64_t DV, int n, uint32_t* out)
{
    const int       w = tex->width, h = tex->height, stride = tex->stride >> 2;
    const uint32_t* base = (const uint32_t*)tex->data;
    for (int i = 0; i < n; i++) {
        int             ix = fm_icoord(U), iy = fm_icoord(V);
        uint32_t        fx = (uint32_t)(U >> 8) & 255u, fy = (uint32_t)(V >> 8) & 255u;
        int             x0 = fm_wrap_coord(ix, w, s->wrap_u), x1 = fm_wrap_coord(ix + 1, w, s->wrap_u);
        int             y0 = fm_wrap_coord(iy, h, s->wrap_v), y1 = fm_wrap_coord(iy + 1, h, s->wrap_v);
        const uint32_t* r0 = y0 < 0 ? NULL : base + (ptrdiff_t)y0 * stride;
        const uint32_t* r1 = y1 < 0 ? NULL : base + (ptrdiff_t)y1 * stride;
        uint32_t        p00 = (r0 && x0 >= 0) ? r0[x0] : 0u, p01 = (r0 && x1 >= 0) ? r0[x1] : 0u;
        uint32_t        p10 = (r1 && x0 >= 0) ? r1[x0] : 0u, p11 = (r1 && x1 >= 0) ? r1[x1] : 0u;
        out[i] = fm_bilerp(p00, p01, p10, p11, fx, fy);
        U += DU;
        V += DV;
    }
}

/* Spans are split into [0,a) wrap, [a,b) fast in-bounds, [b,n) wrap: u and v
 * are linear in i, so the in-bounds part is one interval (fatmap style
 * constant gradients). Both paths use the same math, so the split is exact. */
void fm_sample_span(const fm_surface* tex, const fm_sampler* s, float u, float v, float du, float dv, int n,
                    uint32_t* out)
{
    if (n <= 0) return;
    const int       w = tex->width, h = tex->height;
    const int       stride = tex->stride >> 2;
    const uint32_t* base   = (const uint32_t*)tex->data;
    int64_t         U = fm_to_fixed(u), V = fm_to_fixed(v);
    int64_t         DU = fm_to_fixed(du), DV = fm_to_fixed(dv);
    int             bilinear = s->filter != FM_FILTER_NEAREST;
    if (bilinear) {
        U -= 32768; /* taps at (u - 0.5, v - 0.5) */
        V -= 32768;
    }
    /* in-bounds interval: nearest needs 0 <= x < w, bilinear 0 <= x0 && x0 + 1 < w */
    int64_t ulim = (int64_t)(bilinear ? w - 1 : w) << 16, vlim = (int64_t)(bilinear ? h - 1 : h) << 16;
    int     ua, ub, va, vb;
    fm_axis_range(U, DU, 0, ulim, n, &ua, &ub);
    fm_axis_range(V, DV, 0, vlim, n, &va, &vb);
    int a = FM_MAX(ua, va), b = FM_MIN(ub, vb);
    if (b <= a) a = b = n;

    if (a > 0) {
        if (bilinear)
            fm_sample_bilinear_wrap(tex, s, U, V, DU, DV, a, out);
        else
            fm_sample_nearest_wrap(tex, s, U, V, DU, DV, a, out);
    }
    if (b > a) {
        int64_t Ua = U + DU * a, Va = V + DV * a;
        int     m  = b - a;
        if (bilinear) {
            fm_k->bilinear(base, stride, (int32_t)Ua, (int32_t)Va, (int32_t)DU, (int32_t)DV, m, out + a);
        } else if (DV == 0 && DU == 65536) {
            memcpy(out + a, base + (ptrdiff_t)fm_fx_floor(Va) * stride + fm_fx_floor(Ua), (size_t)m * 4);
        } else {
            int32_t iu = (int32_t)Ua, iv = (int32_t)Va, idu = (int32_t)DU, idv = (int32_t)DV;
            for (int i = 0; i < m; i++) {
                out[a + i] = base[(ptrdiff_t)(iv >> 16) * stride + (iu >> 16)];
                iu += idu;
                iv += idv;
            }
        }
    }
    if (b < n) {
        int64_t Ub = U + DU * b, Vb = V + DV * b;
        if (bilinear)
            fm_sample_bilinear_wrap(tex, s, Ub, Vb, DU, DV, n - b, out + b);
        else
            fm_sample_nearest_wrap(tex, s, Ub, Vb, DU, DV, n - b, out + b);
    }
}

static void fm_bilinear_wrap1(const uint32_t* base, int w, int h, int stride, const fm_sampler* s, int32_t U,
                              int32_t V, uint32_t* out);

static int32_t fm_pt_fixed(float f)
{
    f = FM_CLAMP(f, -32767.0f, 32767.0f);
    return (int32_t)fm_floorf(f * 65536.0f + 0.5f);
}

/* fetch at 16.16 fixed texel coordinates (bilinear: already minus half a
 * texel). In-bounds bilinear taps are compacted and fetched by the SIMD
 * kernel; wrapping taps take the per pixel path (same math). */
void fm__sample_fixed(const fm_surface* tex, const fm_sampler* s, const int32_t* U, const int32_t* V, int n,
                      uint32_t* out)
{
    const int       w = tex->width, h = tex->height, stride = tex->stride >> 2;
    const uint32_t* base = (const uint32_t*)tex->data;
    if (s->filter == FM_FILTER_NEAREST) {
        for (int i = 0; i < n; i++) {
            int x = U[i] >> 16, y = V[i] >> 16;
            if ((unsigned)x >= (unsigned)w) x = fm_wrap_coord(x, w, s->wrap_u);
            if ((unsigned)y >= (unsigned)h) y = fm_wrap_coord(y, h, s->wrap_v);
            out[i] = (x < 0 || y < 0) ? 0u : base[(ptrdiff_t)y * stride + x];
        }
        return;
    }
    enum { CH = 64 };
    int32_t  cu[CH], cv[CH];
    int      ci[CH];
    uint32_t co[CH];
    for (int i0 = 0; i0 < n; i0 += CH) {
        int m = FM_MIN(CH, n - i0), k = 0;
        for (int i = i0; i < i0 + m; i++) {
            if ((unsigned)(U[i] >> 16) < (unsigned)(w - 1) && (unsigned)(V[i] >> 16) < (unsigned)(h - 1)) {
                cu[k] = U[i];
                cv[k] = V[i];
                ci[k] = i;
                k++;
            } else {
                ci[CH - 1 - (i - i0 - k)] = i; /* non compacted, stored from the back */
            }
        }
        if (k == m) { /* all in bounds: no scatter needed */
            fm_k->bilinear_pts(base, stride, U + i0, V + i0, m, out + i0);
            continue;
        }
        if (k) {
            fm_k->bilinear_pts(base, stride, cu, cv, k, co);
            for (int j = 0; j < k; j++) out[ci[j]] = co[j];
        }
        for (int j = k; j < m; j++) {
            int i = ci[CH - 1 - (j - k)];
            fm_bilinear_wrap1(base, w, h, stride, s, U[i], V[i], &out[i]);
        }
    }
}

void fm_sample_points(const fm_surface* tex, const fm_sampler* s, const float* u, const float* v, int n,
                      uint32_t* out)
{
    int32_t U[64], V[64];
    int     bi = s->filter != FM_FILTER_NEAREST ? 32768 : 0;
    for (int i0 = 0; i0 < n; i0 += 64) {
        int m = FM_MIN(64, n - i0);
        for (int i = 0; i < m; i++) {
            U[i] = fm_pt_fixed(u[i0 + i]) - bi;
            V[i] = fm_pt_fixed(v[i0 + i]) - bi;
        }
        fm__sample_fixed(tex, s, U, V, m, out + i0);
    }
}

static void fm_bilinear_wrap1(const uint32_t* base, int w, int h, int stride, const fm_sampler* s, int32_t U,
                              int32_t V, uint32_t* out)
{
    {
        int      ix = U >> 16, iy = V >> 16;
        uint32_t fx = (uint32_t)(U >> 8) & 255u, fy = (uint32_t)(V >> 8) & 255u;
        int             x0 = fm_wrap_coord(ix, w, s->wrap_u), x1 = fm_wrap_coord(ix + 1, w, s->wrap_u);
        int             y0 = fm_wrap_coord(iy, h, s->wrap_v), y1 = fm_wrap_coord(iy + 1, h, s->wrap_v);
        const uint32_t* r0 = y0 < 0 ? NULL : base + (ptrdiff_t)y0 * stride;
        const uint32_t* r1 = y1 < 0 ? NULL : base + (ptrdiff_t)y1 * stride;
        uint32_t        p00 = (r0 && x0 >= 0) ? r0[x0] : 0u, p01 = (r0 && x1 >= 0) ? r0[x1] : 0u;
        uint32_t        p10 = (r1 && x0 >= 0) ? r1[x0] : 0u, p11 = (r1 && x1 >= 0) ? r1[x1] : 0u;
        *out = fm_bilerp(p00, p01, p10, p11, fx, fy);
    }
}

/* ---- gradients ------------------------------------------------------------------ */

#if FM_GRAD_LUT_SIZE != FM_GRADIENT_LUT_SIZE
#  error "gradient LUT size mismatch"
#endif

typedef struct fm_stop {
    float    off;
    fm_color c;
} fm_stop;

struct fm_gradient {
    fm_stop*  stops;
    int       n, cap;
    fm_extend extend;
    unsigned  version;
    uint32_t  lut[FM_GRAD_LUT_SIZE];
};

static void fm_premul_f(fm_color c, float* o)
{
    float a = (float)(c >> 24) / 255.0f;
    o[0]    = (float)((c >> 16) & 255) / 255.0f * a;
    o[1]    = (float)((c >> 8) & 255) / 255.0f * a;
    o[2]    = (float)(c & 255) / 255.0f * a;
    o[3]    = a;
}

/* eager rebuild: gradients are immutable while rendering */
static void fm_gradient_build(fm_gradient* g)
{
    g->version++;
    if (g->n == 0) {
        memset(g->lut, 0, sizeof(g->lut));
        return;
    }
    int seg = 0;
    for (int i = 0; i < FM_GRAD_LUT_SIZE; i++) {
        float t = (float)i / (float)(FM_GRAD_LUT_SIZE - 1);
        float c[4];
        if (t <= g->stops[0].off) {
            fm_premul_f(g->stops[0].c, c);
        } else if (t >= g->stops[g->n - 1].off) {
            fm_premul_f(g->stops[g->n - 1].c, c);
        } else {
            /* last segment whose start <= t (handles hard stops) */
            while (seg + 1 < g->n - 1 && g->stops[seg + 1].off <= t) seg++;
            const fm_stop* s0 = &g->stops[seg];
            const fm_stop* s1 = &g->stops[seg + 1];
            float          a[4], b[4];
            fm_premul_f(s0->c, a);
            fm_premul_f(s1->c, b);
            float span = s1->off - s0->off;
            float f    = span > 0.0f ? (t - s0->off) / span : 1.0f;
            for (int k = 0; k < 4; k++) c[k] = a[k] + (b[k] - a[k]) * f;
        }
        uint32_t A = (uint32_t)(FM_CLAMP(c[3], 0.0f, 1.0f) * 255.0f + 0.5f);
        uint32_t R = FM_MIN(A, (uint32_t)(FM_CLAMP(c[0], 0.0f, 1.0f) * 255.0f + 0.5f));
        uint32_t G = FM_MIN(A, (uint32_t)(FM_CLAMP(c[1], 0.0f, 1.0f) * 255.0f + 0.5f));
        uint32_t B = FM_MIN(A, (uint32_t)(FM_CLAMP(c[2], 0.0f, 1.0f) * 255.0f + 0.5f));
        g->lut[i]  = (A << 24) | (R << 16) | (G << 8) | B;
    }
}

fm_gradient* fm_gradient_create(void)
{
    fm__init();
    fm_gradient* g = (fm_gradient*)calloc(1, sizeof(fm_gradient));
    if (g) fm_gradient_build(g);
    return g;
}

void fm_gradient_destroy(fm_gradient* g)
{
    if (!g) return;
    free(g->stops);
    free(g);
}

void fm_gradient_add_stop(fm_gradient* g, float offset, fm_color c)
{
    if (!g || !isfinite(offset)) return;
    offset = FM_CLAMP(offset, 0.0f, 1.0f);
    if (g->n == g->cap) {
        int      nc = g->cap ? g->cap * 2 : 8;
        fm_stop* ns = (fm_stop*)realloc(g->stops, (size_t)nc * sizeof(fm_stop));
        if (!ns) return;
        g->stops = ns;
        g->cap   = nc;
    }
    /* stable insert: after all stops with offset <= new offset */
    int i = g->n;
    while (i > 0 && g->stops[i - 1].off > offset) {
        g->stops[i] = g->stops[i - 1];
        i--;
    }
    g->stops[i].off = offset;
    g->stops[i].c   = c;
    g->n++;
    fm_gradient_build(g);
}

void fm_gradient_clear_stops(fm_gradient* g)
{
    if (!g) return;
    g->n = 0;
    fm_gradient_build(g);
}

void fm_gradient_set_extend(fm_gradient* g, fm_extend e)
{
    if (!g) return;
    g->extend = e;
    g->version++;
}

int             fm_gradient_stop_count(const fm_gradient* g) { return g ? g->n : 0; }
fm_extend       fm_gradient_get_extend(const fm_gradient* g) { return g ? g->extend : FM_EXTEND_PAD; }
const uint32_t* fm_gradient_lut(const fm_gradient* g) { return g ? g->lut : NULL; }
unsigned        fm_gradient_version(const fm_gradient* g) { return g ? g->version : 0; }

/* ---- pipeline -------------------------------------------------------------------- */

void fm_draw_state_init(fm_draw_state* s)
{
    memset(s, 0, sizeof(*s));
    s->op             = FM_OP_SRC_OVER;
    s->alpha          = 1.0f;
    s->source         = FM_SOURCE_SOLID;
    s->color          = 0xff000000u;
    s->xf             = fm_affine_identity();
    s->sampler.filter = FM_FILTER_BILINEAR;
    s->sampler.wrap_u = FM_WRAP_CLAMP;
    s->sampler.wrap_v = FM_WRAP_CLAMP;
}

typedef void (*fm_run_fn)(struct fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m);
typedef void (*fm_fetch_fn)(struct fm_pipeline* p, int x, int y, int n, uint32_t* out);

struct fm_pipeline {
    fm_surface*   dst;
    fm_draw_state st;
    int           dirty;

    /* resolved op chain */
    fm_run_fn   run;
    fm_fetch_fn fetch;
    uint32_t    alpha8;
    int         cov_alpha;
    uint32_t    solid;
    fm_affine   inv;
    float       lin_gx, lin_gy, lin_g0;
    int         tex_dx, tex_dy;

    uint32_t scratch[FM_PIPE_CHUNK];
    uint8_t  covbuf[FM_PIPE_CHUNK];
};

fm_pipeline* fm_pipeline_create(void)
{
    fm__init();
    fm_pipeline* p = (fm_pipeline*)calloc(1, sizeof(fm_pipeline));
    if (!p) return NULL;
    fm_draw_state_init(&p->st);
    p->dirty = 1;
    return p;
}

void fm_pipeline_destroy(fm_pipeline* p) { free(p); }

void fm_pipeline_set_state(fm_pipeline* p, const fm_draw_state* s)
{
    p->st    = *s;
    p->dirty = 1;
}

const fm_draw_state* fm_pipeline_get_state(const fm_pipeline* p) { return &p->st; }

void fm_pipeline_set_target(fm_pipeline* p, fm_surface* dst)
{
    p->dst   = dst;
    p->dirty = 1;
}
void fm_pipeline_set_blend(fm_pipeline* p, fm_blend_op op)
{
    p->st.op = (op >= 0 && op < FM_OP_COUNT) ? op : FM_OP_SRC_OVER;
    p->dirty = 1;
}
void fm_pipeline_set_alpha(fm_pipeline* p, float a)
{
    p->st.alpha = a;
    p->dirty    = 1;
}
void fm_pipeline_set_clip_mask(fm_pipeline* p, const fm_surface* mask)
{
    p->st.clip_mask = mask;
    p->dirty        = 1;
}
static void fm_set_xf(fm_pipeline* p, const fm_affine* xf) { p->st.xf = xf ? *xf : fm_affine_identity(); }

void fm_pipeline_set_solid(fm_pipeline* p, fm_color c)
{
    p->st.source = FM_SOURCE_SOLID;
    p->st.color  = c;
    p->dirty     = 1;
}
void fm_pipeline_set_none(fm_pipeline* p)
{
    p->st.source = FM_SOURCE_NONE;
    p->dirty     = 1;
}
static void fm_set_grad(fm_pipeline* p, fm_source_type t, const fm_gradient* g, const fm_affine* xf)
{
    p->st.source = t;
    p->st.lut    = fm_gradient_lut(g);
    p->st.extend = fm_gradient_get_extend(g);
    fm_set_xf(p, xf);
    p->dirty = 1;
}
void fm_pipeline_set_linear(fm_pipeline* p, const fm_gradient* g, float x0, float y0, float x1, float y1,
                            const fm_affine* xf)
{
    fm_set_grad(p, FM_SOURCE_LINEAR, g, xf);
    p->st.params[0] = x0;
    p->st.params[1] = y0;
    p->st.params[2] = x1;
    p->st.params[3] = y1;
}
void fm_pipeline_set_radial(fm_pipeline* p, const fm_gradient* g, float x0, float y0, float r0, float x1, float y1,
                            float r1, const fm_affine* xf)
{
    fm_set_grad(p, FM_SOURCE_RADIAL, g, xf);
    p->st.params[0] = x0;
    p->st.params[1] = y0;
    p->st.params[2] = r0;
    p->st.params[3] = x1;
    p->st.params[4] = y1;
    p->st.params[5] = r1;
}
void fm_pipeline_set_conic(fm_pipeline* p, const fm_gradient* g, float start_angle, float cx, float cy,
                           const fm_affine* xf)
{
    fm_set_grad(p, FM_SOURCE_CONIC, g, xf);
    p->st.params[0] = start_angle;
    p->st.params[1] = cx;
    p->st.params[2] = cy;
}
void fm_pipeline_set_texture(fm_pipeline* p, const fm_surface* tex, const fm_sampler* s, const fm_affine* xf)
{
    p->st.source  = FM_SOURCE_TEXTURE;
    p->st.texture = tex;
    if (s) {
        p->st.sampler = *s;
    } else {
        p->st.sampler.filter = FM_FILTER_BILINEAR;
        p->st.sampler.wrap_u = p->st.sampler.wrap_v = FM_WRAP_CLAMP;
    }
    fm_set_xf(p, xf);
    p->dirty = 1;
}
void fm_pipeline_set_custom(fm_pipeline* p, fm_shade_fn fn, void* user)
{
    p->st.source      = FM_SOURCE_CUSTOM;
    p->st.custom      = fn;
    p->st.custom_user = user;
    p->dirty          = 1;
}

/* ---- fetch ops ---- */

static void fm_fetch_linear(fm_pipeline* p, int x, int y, int n, uint32_t* out)
{
    float t0 = p->lin_gx * ((float)x + 0.5f) + p->lin_gy * ((float)y + 0.5f) + p->lin_g0;
    fm_k->linear_grad(p->st.lut, t0, p->lin_gx, n, (int)p->st.extend, out);
}

static void fm_fetch_radial(fm_pipeline* p, int x, int y, int n, uint32_t* out)
{
    const fm_affine* m   = &p->inv;
    const float*     gp  = p->st.params;
    const uint32_t*  lut = p->st.lut;
    int              ext = (int)p->st.extend;
    float            X = (float)x + 0.5f, Y = (float)y + 0.5f;
    float            u0 = m->a * X + m->c * Y + m->e, v0 = m->b * X + m->d * Y + m->f;
    float            x0 = gp[0], y0 = gp[1], r0 = gp[2];
    float            cdx = gp[3] - x0, cdy = gp[4] - y0, dr = gp[5] - r0;
    float            a   = cdx * cdx + cdy * cdy - dr * dr;
    int              lin = fabsf(a) < 1e-9f;
    float            ia  = lin ? 0.0f : 1.0f / a;
    for (int i = 0; i < n; i++) {
        float pdx = u0 + (float)i * m->a - x0, pdy = v0 + (float)i * m->b - y0;
        float b = pdx * cdx + pdy * cdy + r0 * dr;
        float c = pdx * pdx + pdy * pdy - r0 * r0;
        float t;
        if (lin) {
            if (b == 0.0f) {
                out[i] = 0;
                continue;
            }
            t = c / (2.0f * b);
            if (r0 + t * dr < 0.0f) {
                out[i] = 0;
                continue;
            }
        } else {
            float disc = b * b - a * c;
            if (disc < 0.0f) {
                out[i] = 0;
                continue;
            }
            float sq = sqrtf(disc);
            float t1 = (b + sq) * ia, t2 = (b - sq) * ia;
            float tmax = FM_MAX(t1, t2), tmin = FM_MIN(t1, t2);
            if (r0 + tmax * dr >= 0.0f)
                t = tmax;
            else if (r0 + tmin * dr >= 0.0f)
                t = tmin;
            else {
                out[i] = 0;
                continue;
            }
        }
        out[i] = lut[fm_grad_index(t, ext)];
    }
}

static void fm_fetch_conic(fm_pipeline* p, int x, int y, int n, uint32_t* out)
{
    const fm_affine* m      = &p->inv;
    float            X      = (float)x + 0.5f, Y = (float)y + 0.5f;
    float            u0     = m->a * X + m->c * Y + m->e - p->st.params[1];
    float            v0     = m->b * X + m->d * Y + m->f - p->st.params[2];
    const float      inv2pi = 0.15915494309189535f;
    for (int i = 0; i < n; i++) {
        float ang = atan2f(v0 + (float)i * m->b, u0 + (float)i * m->a) - p->st.params[0];
        float t   = ang * inv2pi;
        t         = t - floorf(t);
        out[i]    = p->st.lut[fm_grad_index(t, 0)];
    }
}

static void fm_fetch_texture(fm_pipeline* p, int x, int y, int n, uint32_t* out)
{
    const fm_affine* m = &p->inv;
    float            X = (float)x + 0.5f, Y = (float)y + 0.5f;
    fm_sample_span(p->st.texture, &p->st.sampler, m->a * X + m->c * Y + m->e, m->b * X + m->d * Y + m->f, m->a,
                   m->b, n, out);
}

static void fm_fetch_custom(fm_pipeline* p, int x, int y, int n, uint32_t* out)
{
    p->st.custom(p->st.custom_user, x, y, n, out);
}

/* ---- run ops (fused fetch + blend) ---- */

static void fm_run_fill(fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m)
{
    (void)x;
    (void)y;
    if (m)
        fm_k->solid_over_mask(d, p->solid, m, n);
    else
        fm_k->fill(d, p->solid, n);
}

static void fm_run_solid_over(fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m)
{
    (void)x;
    (void)y;
    if (m)
        fm_k->solid_over_mask(d, p->solid, m, n);
    else
        fm_k->solid_over(d, p->solid, n);
}

static void fm_run_solid_op(fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m)
{
    (void)x;
    (void)y;
    fm_k->span_op(d, p->scratch, m, n, p->st.op);
}

static void fm_run_shade_over(fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m)
{
    p->fetch(p, x, y, n, p->scratch);
    if (m)
        fm_k->span_over_mask(d, p->scratch, m, n);
    else
        fm_k->span_over(d, p->scratch, n);
}

static void fm_run_shade_op(fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m)
{
    p->fetch(p, x, y, n, p->scratch);
    fm_k->span_op(d, p->scratch, m, n, p->st.op);
}

/* integer translated texture: blend straight from the texture rows */
static void fm_run_tex_direct(fm_pipeline* p, uint32_t* d, int x, int y, int n, const uint8_t* m)
{
    const fm_surface* tex = p->st.texture;
    int               tx = x + p->tex_dx, ty = y + p->tex_dy;
    if (ty < 0 || ty >= tex->height || tx < 0 || tx + n > tex->width) {
        if (p->st.op == FM_OP_SRC_OVER)
            fm_run_shade_over(p, d, x, y, n, m);
        else
            fm_run_shade_op(p, d, x, y, n, m);
        return;
    }
    const uint32_t* s = fm_surface_row32(tex, ty) + tx;
    if (p->st.op == FM_OP_SRC_OVER) {
        if (m)
            fm_k->span_over_mask(d, s, m, n);
        else
            fm_k->span_over(d, s, n);
    } else if (p->st.op == FM_OP_COPY && !m) {
        memcpy(d, s, (size_t)n * 4);
    } else {
        fm_k->span_op(d, s, m, n, p->st.op);
    }
}

static void fm_pipe_prepare(fm_pipeline* p)
{
    fm_draw_state* st = &p->st;
    p->dirty          = 0;
    p->run            = NULL;
    p->cov_alpha      = 0;
    if (!p->dst || p->dst->format != FM_FORMAT_ARGB32) return;
    float a = st->alpha;
    if (!(a == a)) a = 1.0f;
    p->alpha8 = (uint32_t)(FM_CLAMP(a, 0.0f, 1.0f) * 255.0f + 0.5f);
    if (p->alpha8 == 0) return;
    if (st->op < 0 || st->op >= FM_OP_COUNT) st->op = FM_OP_SRC_OVER;
    if (st->clip_mask && st->clip_mask->format != FM_FORMAT_A8) st->clip_mask = NULL;

    fm_source_type src    = st->source;
    int            direct = 0;
    if ((src == FM_SOURCE_LINEAR || src == FM_SOURCE_RADIAL || src == FM_SOURCE_CONIC) && !st->lut)
        src = FM_SOURCE_NONE;
    if (src == FM_SOURCE_TEXTURE && (!st->texture || st->texture->format != FM_FORMAT_ARGB32)) src = FM_SOURCE_NONE;
    if (src == FM_SOURCE_CUSTOM && !st->custom) src = FM_SOURCE_NONE;
    if (src != FM_SOURCE_SOLID && src != FM_SOURCE_NONE && src != FM_SOURCE_CUSTOM) {
        if (!fm_affine_invert(&st->xf, &p->inv)) src = FM_SOURCE_NONE;
    }
    if (src == FM_SOURCE_LINEAR) {
        const float* gp = st->params;
        float        dx = gp[2] - gp[0], dy = gp[3] - gp[1];
        float        L  = dx * dx + dy * dy;
        if (L == 0.0f) {
            src = FM_SOURCE_NONE;
        } else {
            const fm_affine* m = &p->inv;
            p->lin_gx          = (m->a * dx + m->b * dy) / L;
            p->lin_gy          = (m->c * dx + m->d * dy) / L;
            p->lin_g0          = ((m->e - gp[0]) * dx + (m->f - gp[1]) * dy) / L;
            p->fetch           = fm_fetch_linear;
        }
    } else if (src == FM_SOURCE_RADIAL) {
        const float* gp = st->params;
        if (gp[0] == gp[3] && gp[1] == gp[4] && gp[2] == gp[5])
            src = FM_SOURCE_NONE;
        else
            p->fetch = fm_fetch_radial;
    } else if (src == FM_SOURCE_CONIC) {
        p->fetch = fm_fetch_conic;
    } else if (src == FM_SOURCE_TEXTURE) {
        p->fetch           = fm_fetch_texture;
        const fm_affine* m = &p->inv;
        if (m->a == 1.0f && m->b == 0.0f && m->c == 0.0f && m->d == 1.0f && m->e == floorf(m->e) &&
            m->f == floorf(m->f) && fabsf(m->e) < 1e8f && fabsf(m->f) < 1e8f) {
            direct    = 1;
            p->tex_dx = (int)m->e;
            p->tex_dy = (int)m->f;
        }
    } else if (src == FM_SOURCE_CUSTOM) {
        p->fetch = fm_fetch_custom;
    }

    if (src == FM_SOURCE_SOLID || src == FM_SOURCE_NONE) {
        uint32_t s = src == FM_SOURCE_SOLID ? fm_px_scale(fm_premul_inline(st->color), p->alpha8) : 0u;
        p->solid   = s;
        if (st->op == FM_OP_SRC_OVER) {
            if (s == 0) return;
            p->run = (s >> 24) == 255 ? fm_run_fill : fm_run_solid_over;
        } else if (st->op == FM_OP_COPY && (s >> 24) == 255) {
            p->run = fm_run_fill;
        } else {
            fm_k->fill(p->scratch, s, FM_PIPE_CHUNK);
            p->run = fm_run_solid_op;
        }
        return;
    }
    p->cov_alpha = p->alpha8 < 255;
    if (direct)
        p->run = fm_run_tex_direct;
    else
        p->run = st->op == FM_OP_SRC_OVER ? fm_run_shade_over : fm_run_shade_op;
}

void fm_pipeline_span(void* pipeline, int y, int x, int n, uint8_t* cov)
{
    fm_pipeline* p = (fm_pipeline*)pipeline;
    if (p->dirty) fm_pipe_prepare(p);
    if (!p->run) return;
    fm_surface* dst = p->dst;
    if (y < 0 || y >= dst->height) return;
    if (x < 0) {
        n += x;
        if (cov) cov -= x;
        x = 0;
    }
    if (x + n > dst->width) n = dst->width - x;
    if (n <= 0) return;

    uint32_t*      d  = fm_surface_row32(dst, y) + x;
    const uint8_t* cr = p->st.clip_mask ? fm_surface_row8(p->st.clip_mask, y) + x : NULL;
    for (int i = 0; i < n; i += FM_PIPE_CHUNK) {
        int      c = FM_MIN(FM_PIPE_CHUNK, n - i);
        uint8_t* m = cov ? cov + i : NULL;
        if (cr) {
            if (m)
                fm_k->mask_mul(m, cr + i, c);
            else {
                memcpy(p->covbuf, cr + i, (size_t)c);
                m = p->covbuf;
            }
        }
        if (p->cov_alpha) {
            if (m)
                fm_k->mask_scale(m, p->alpha8, c);
            else {
                memset(p->covbuf, (int)p->alpha8, (size_t)c);
                m = p->covbuf;
            }
        }
        p->run(p, d + i, x + i, y, c, m);
    }
}

void fm_pipeline_fill_rect(fm_pipeline* p, int x0, int y0, int x1, int y1)
{
    if (!p->dst) return;
    x0 = FM_MAX(x0, 0);
    y0 = FM_MAX(y0, 0);
    x1 = FM_MIN(x1, p->dst->width);
    y1 = FM_MIN(y1, p->dst->height);
    for (int y = y0; y < y1; y++) fm_pipeline_span(p, y, x0, x1 - x0, NULL);
}

/* ---- mask sink ---- */

void fm_mask_span(void* sink, int y, int x, int n, uint8_t* cov)
{
    fm_mask_sink* s = (fm_mask_sink*)sink;
    uint8_t*      d = fm_surface_row8(s->out, y) + x;
    if (cov)
        memcpy(d, cov, (size_t)n);
    else
        memset(d, 255, (size_t)n);
    if (s->in) fm_k->mask_mul(d, fm_surface_row8(s->in, y) + x, n);
}
