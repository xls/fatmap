/*
 * fatmap - 3D pipeline stages: vertex stage, homogeneous clipping, triangle
 * setup, quad rasterization, fixed function fragment stage, output merger.
 *
 * Setup follows fatmap: every interpolant is a plane with constant
 * gradients over the triangle,
 *   d/dx = (d1 * dy2 - d2 * dy1) / area,  d/dy = (d2 * dx1 - d1 * dx2) / area
 * (d1, d2 = attribute deltas to vertices 1 and 2). Values are evaluated per
 * pixel from the plane (never accumulated), so results do not depend on how
 * the screen is split into tiles or batches.
 */
#include "fm3d_internal.h"

/* ---- vertex stage ------------------------------------------------------------ */

void fm3d_vs_fixed(const fm3d_dstate* st, const fm3d_vertex* in, int n, fm3d_vout* out)
{
    const fm_mat4* m = &st->mvp;
    for (int i = 0; i < n; i++) {
        const fm3d_vertex* v = &in[i];
        fm3d_vout*         o = &out[i];
        o->pos[0] = m->c[0].x * v->x + m->c[1].x * v->y + m->c[2].x * v->z + m->c[3].x;
        o->pos[1] = m->c[0].y * v->x + m->c[1].y * v->y + m->c[2].y * v->z + m->c[3].y;
        o->pos[2] = m->c[0].z * v->x + m->c[1].z * v->y + m->c[2].z * v->z + m->c[3].z;
        o->pos[3] = m->c[0].w * v->x + m->c[1].w * v->y + m->c[2].w * v->z + m->c[3].w;
        o->var[FM3D_VAR_U] = v->u;
        o->var[FM3D_VAR_V] = v->v;
        o->var[FM3D_VAR_R] = (float)((v->color >> 16) & 255) * (1.0f / 255.0f);
        o->var[FM3D_VAR_G] = (float)((v->color >> 8) & 255) * (1.0f / 255.0f);
        o->var[FM3D_VAR_B] = (float)(v->color & 255) * (1.0f / 255.0f);
        o->var[FM3D_VAR_A] = (float)(v->color >> 24) * (1.0f / 255.0f);
    }
}

/* ---- clipping ------------------------------------------------------------------ */

enum { FM3D_NPLANES = 7 };

static float fm3d_plane_dist(const fm3d_dstate* st, int p, const float* v)
{
    switch (p) {
    case 0: return v[3] - 1e-6f;                                                  /* w > 0 */
    case 1: return st->clip_depth == FM3D_DEPTH_ZERO_ONE ? v[2] : v[2] + v[3];    /* near */
    case 2: return v[3] - v[2];                                                    /* far */
    case 3: return FM3D_GUARD * v[3] - v[0];                                       /* guard band */
    case 4: return FM3D_GUARD * v[3] + v[0];
    case 5: return FM3D_GUARD * v[3] - v[1];
    default: return FM3D_GUARD * v[3] + v[1];
    }
}

static unsigned fm3d_outcode(const fm3d_dstate* st, const fm3d_vout* v)
{
    unsigned c = 0;
    for (int p = 0; p < FM3D_NPLANES; p++) {
        if (st->depth_clamp && (p == 1 || p == 2)) continue; /* depth clamp: no near / far clipping */
        if (fm3d_plane_dist(st, p, v->pos) < 0.0f) c |= 1u << p;
    }
    return c;
}

static void fm3d_lerp_vout(const fm3d_vout* a, const fm3d_vout* b, float t, int nvar, fm3d_vout* o)
{
    for (int k = 0; k < 4; k++) o->pos[k] = a->pos[k] + (b->pos[k] - a->pos[k]) * t;
    for (int k = 0; k < nvar; k++) o->var[k] = a->var[k] + (b->var[k] - a->var[k]) * t;
}

/* ---- setup --------------------------------------------------------------------- */

typedef struct fm3d_sv { /* projected vertex */
    int32_t X, Y;        /* 28.4 */
    float   z, invw;
    float   var[FM3D_MAX_VARYINGS];
} fm3d_sv;

static int64_t fm3d_floor_div(int64_t a, int64_t b) /* b > 0 */
{
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

static void fm3d_project(const fm3d_dstate* st, const fm3d_vout* v, fm3d_sv* o)
{
    float iw = 1.0f / v->pos[3];
    float nx = v->pos[0] * iw, ny = v->pos[1] * iw, nz = v->pos[2] * iw;
    float sx = (float)st->vp[0] + (nx + 1.0f) * 0.5f * (float)st->vp[2];
    float sy = (float)st->vp[1] + (1.0f - ny) * 0.5f * (float)st->vp[3];
    o->X     = (int32_t)fm_floorf(sx * 16.0f + 0.5f);
    o->Y     = (int32_t)fm_floorf(sy * 16.0f + 0.5f);
    float z  = st->clip_depth == FM3D_DEPTH_ZERO_ONE ? nz : nz * 0.5f + 0.5f;
    if (!st->depth_clamp) z = FM_CLAMP(z, 0.0f, 1.0f); /* depth clamp: clamped per pixel */
    o->z     = st->depth_near + (st->depth_far - st->depth_near) * z;
    o->invw  = iw;
    if (st->perspective)
        for (int k = 0; k < st->nvar; k++) o->var[k] = v->var[k] * iw;
    else
        for (int k = 0; k < st->nvar; k++) o->var[k] = v->var[k];
}

static void fm3d_plane(float a0, float a1, float a2, float dx1, float dy1, float dx2, float dy2, float inv, float* p)
{
    float d1 = a1 - a0, d2 = a2 - a0;
    p[0]     = a0;
    p[1]     = (d1 * dy2 - d2 * dy1) * inv;
    p[2]     = (d2 * dx1 - d1 * dx2) * inv;
}

static uint32_t fm3d_premul_f(float r, float g, float b, float a)
{
    a          = FM_CLAMP(a, 0.0f, 1.0f);
    uint32_t A = (uint32_t)(a * 255.0f + 0.5f);
    uint32_t R = (uint32_t)(FM_CLAMP(r, 0.0f, 1.0f) * a * 255.0f + 0.5f);
    uint32_t G = (uint32_t)(FM_CLAMP(g, 0.0f, 1.0f) * a * 255.0f + 0.5f);
    uint32_t B = (uint32_t)(FM_CLAMP(b, 0.0f, 1.0f) * a * 255.0f + 0.5f);
    return (A << 24) | (FM_MIN(R, A) << 16) | (FM_MIN(G, A) << 8) | FM_MIN(B, A);
}

static void fm3d_setup(const fm3d_dstate* st, const fm3d_sv* v0, const fm3d_sv* v1, const fm3d_sv* v2,
                       fm3d_sink* sink)
{
    int64_t area = (int64_t)(v1->X - v0->X) * (v2->Y - v0->Y) - (int64_t)(v2->X - v0->X) * (v1->Y - v0->Y);
    if (area == 0) {
        sink->stats.triangles_culled++;
        return;
    }
    /* area > 0 (y-down screen formula) means visually clockwise, i.e.
     * clockwise in NDC as well: counter clockwise fronts have area < 0 */
    int front = st->front == FM3D_FRONT_CCW ? area < 0 : area > 0;
    if (st->cull == FM3D_CULL_FRONT_AND_BACK || (st->cull == FM3D_CULL_BACK && !front) ||
        (st->cull == FM3D_CULL_FRONT && front)) {
        sink->stats.triangles_culled++;
        return;
    }
    const fm3d_sv* p[3] = { v0, v1, v2 };
    if (area < 0) { /* make edges positive inside */
        p[1] = v2;
        p[2] = v1;
    }

    int32_t minX = FM_MIN(p[0]->X, FM_MIN(p[1]->X, p[2]->X)), maxX = FM_MAX(p[0]->X, FM_MAX(p[1]->X, p[2]->X));
    int32_t minY = FM_MIN(p[0]->Y, FM_MIN(p[1]->Y, p[2]->Y)), maxY = FM_MAX(p[0]->Y, FM_MAX(p[1]->Y, p[2]->Y));
    int     minx = (int)fm3d_floor_div((int64_t)minX - 8, 16), maxx = (int)fm3d_floor_div((int64_t)maxX - 8, 16);
    int     miny = (int)fm3d_floor_div((int64_t)minY - 8, 16), maxy = (int)fm3d_floor_div((int64_t)maxY - 8, 16);
    minx = FM_MAX(minx, st->rect[0]);
    miny = FM_MAX(miny, st->rect[1]);
    maxx = FM_MIN(maxx, st->rect[2] - 1);
    maxy = FM_MIN(maxy, st->rect[3] - 1);
    if (maxx < minx || maxy < miny) return; /* off screen */

    int       nvar = st->nvar;
    size_t    size = sizeof(fm3d_tri) + sizeof(float) * 3 * (size_t)nvar;
    fm3d_tri* t    = sink->arena ? (fm3d_tri*)fm_arena_alloc(sink->arena, size) : (fm3d_tri*)sink->tmp;
    if (!t) return;
    t->st   = st;
    t->minx = minx;
    t->miny = miny;
    t->maxx = maxx;
    t->maxy = maxy;
    for (int e = 0; e < 3; e++) {
        const fm3d_sv* a = p[e];
        const fm3d_sv* b = p[(e + 1) % 3];
        int64_t        A = (int64_t)a->Y - b->Y, B = (int64_t)b->X - a->X;
        int64_t        C = -(A * a->X + B * a->Y);
        int            top_left = A > 0 || (A == 0 && B > 0);
        t->A[e]  = A;
        t->B[e]  = B;
        t->K0[e] = 8 * A + C - (top_left ? 0 : 1);
    }

    float x0 = (float)p[0]->X * (1.0f / 16.0f), y0 = (float)p[0]->Y * (1.0f / 16.0f);
    float dx1 = (float)p[1]->X * (1.0f / 16.0f) - x0, dy1 = (float)p[1]->Y * (1.0f / 16.0f) - y0;
    float dx2 = (float)p[2]->X * (1.0f / 16.0f) - x0, dy2 = (float)p[2]->Y * (1.0f / 16.0f) - y0;
    float inv = 1.0f / (dx1 * dy2 - dx2 * dy1);
    t->x0f    = x0;
    t->y0f    = y0;
    fm3d_plane(p[0]->z, p[1]->z, p[2]->z, dx1, dy1, dx2, dy2, inv, t->z);
    t->flags = front ? 0 : FM3D_TRI_BACK;
    if (st->depth_clamp) t->flags |= FM3D_TRI_ZCLAMP;
    if (st->depth_bias_factor != 0.0f || st->depth_bias_units != 0.0f) {
        /* glPolygonOffset: constant per triangle, units in the smallest
         * resolvable step of the depth format; final depth clamped */
        fm_format df    = st->depth ? st->depth->format : FM_FORMAT_D32F;
        float     r     = df == FM_FORMAT_D16 ? 1.0f / 65535.0f
                        : (df == FM_FORMAT_D24S8 ? 1.0f / 16777215.0f : 1.0f / 16777216.0f);
        float     slope = FM_MAX(fabsf(t->z[1]), fabsf(t->z[2]));
        t->z[0] += st->depth_bias_factor * slope + st->depth_bias_units * r;
        t->flags |= FM3D_TRI_ZCLAMP;
    }
    fm3d_plane(p[0]->invw, p[1]->invw, p[2]->invw, dx1, dy1, dx2, dy2, inv, t->w);
    t->nvar = nvar;
    for (int k = 0; k < nvar; k++)
        fm3d_plane(p[0]->var[k], p[1]->var[k], p[2]->var[k], dx1, dy1, dx2, dy2, inv, t->var + 3 * k);

    /* flat vertex color (common for textured meshes): skip interpolation */
    if (st->fs == fm3d_fs_fixed) {
        int flat = 1;
        for (int k = FM3D_VAR_R; k <= FM3D_VAR_A; k++) {
            /* compare the unprojected colors */
            float c0 = st->perspective ? p[0]->var[k] / p[0]->invw : p[0]->var[k];
            float c1 = st->perspective ? p[1]->var[k] / p[1]->invw : p[1]->var[k];
            float c2 = st->perspective ? p[2]->var[k] / p[2]->invw : p[2]->var[k];
            if (fabsf(c1 - c0) > 1e-4f || fabsf(c2 - c0) > 1e-4f) flat = 0;
        }
        if (flat) {
            t->flags |= FM3D_TRI_FLAT;
            float iw = st->perspective ? 1.0f / p[0]->invw : 1.0f;
            t->flat  = fm3d_premul_f(p[0]->var[FM3D_VAR_R] * iw, p[0]->var[FM3D_VAR_G] * iw,
                                     p[0]->var[FM3D_VAR_B] * iw, p[0]->var[FM3D_VAR_A] * iw);
        }
    }
    sink->stats.triangles_drawn++;
    sink->emit(sink, t);
}

void fm3d_process_tri(const fm3d_dstate* st, const fm3d_vout* a, const fm3d_vout* b, const fm3d_vout* c,
                      fm3d_sink* sink)
{
    sink->stats.triangles_in++;
    unsigned ca = fm3d_outcode(st, a), cb = fm3d_outcode(st, b), cc = fm3d_outcode(st, c);
    if (ca & cb & cc) {
        sink->stats.triangles_culled++;
        return; /* fully outside one plane */
    }
    fm3d_sv sv[FM3D_MAX_CLIP];
    if (!(ca | cb | cc)) {
        fm3d_project(st, a, &sv[0]);
        fm3d_project(st, b, &sv[1]);
        fm3d_project(st, c, &sv[2]);
        fm3d_setup(st, &sv[0], &sv[1], &sv[2], sink);
        return;
    }
    /* Sutherland-Hodgman against the planes that are crossed */
    sink->stats.triangles_clipped++;
    fm3d_vout bufA[FM3D_MAX_CLIP], bufB[FM3D_MAX_CLIP];
    fm3d_vout* in  = bufA;
    fm3d_vout* out = bufB;
    int        n   = 3;
    in[0]          = *a;
    in[1]          = *b;
    in[2]          = *c;
    unsigned crossed = ca | cb | cc;
    for (int pl = 0; pl < FM3D_NPLANES && n >= 3; pl++) {
        if (!(crossed & (1u << pl))) continue;
        int m = 0;
        for (int i = 0; i < n; i++) {
            const fm3d_vout* p = &in[i];
            const fm3d_vout* q = &in[(i + 1) % n];
            float            dp = fm3d_plane_dist(st, pl, p->pos), dq = fm3d_plane_dist(st, pl, q->pos);
            if (dp >= 0.0f && m < FM3D_MAX_CLIP) out[m++] = *p;
            if ((dp >= 0.0f) != (dq >= 0.0f) && m < FM3D_MAX_CLIP)
                fm3d_lerp_vout(p, q, dp / (dp - dq), st->nvar, &out[m++]);
        }
        fm3d_vout* t = in;
        in           = out;
        out          = t;
        n            = m;
    }
    if (n < 3) return;
    for (int i = 0; i < n; i++) fm3d_project(st, &in[i], &sv[i]);
    for (int i = 1; i + 1 < n; i++) fm3d_setup(st, &sv[0], &sv[i], &sv[i + 1], sink);
}

/* ---- rasterization -------------------------------------------------------------- */

/* inclusive span [lo, hi] of pixel centers inside the triangle on row py */
static int fm3d_row_span(const fm3d_tri* t, int py, int* lo, int* hi)
{
    int64_t yc = 16 * (int64_t)py + 8;
    int64_t l = INT32_MIN, h = INT32_MAX;
    for (int e = 0; e < 3; e++) {
        int64_t A = t->A[e], K = t->K0[e] + t->B[e] * yc;
        if (A > 0) {
            int64_t v = -fm3d_floor_div(K, 16 * A); /* ceil(-K / 16A) */
            if (v > l) l = v;
        } else if (A < 0) {
            int64_t v = fm3d_floor_div(K, -16 * A);
            if (v < h) h = v;
        } else if (K < 0) {
            return 0;
        }
    }
    *lo = (int)FM_MAX(l, (int64_t)t->minx);
    *hi = (int)FM_MIN(h, (int64_t)t->maxx);
    return *lo <= *hi;
}

static int fm3d_depth_pass(fm3d_compare f, float z, float d)
{
    switch (f) {
    case FM3D_NEVER: return 0;
    case FM3D_LESS: return z < d;
    case FM3D_EQUAL: return z == d;
    case FM3D_LEQUAL: return z <= d;
    case FM3D_GREATER: return z > d;
    case FM3D_NOTEQUAL: return z != d;
    case FM3D_GEQUAL: return z >= d;
    default: return 1;
    }
}

static int fm3d_row_valid(const fm3d_batch* b, int r, const fm_surface* s)
{
    int py = b->y + r;
    return py >= 0 && py < s->height;
}

static uint8_t fm3d_stencil_apply(fm3d_stencil_op op, uint8_t s, uint8_t ref)
{
    switch (op) {
    case FM3D_STENCIL_ZERO: return 0;
    case FM3D_STENCIL_REPLACE: return ref;
    case FM3D_STENCIL_INCR: return (uint8_t)(s == 255 ? 255 : s + 1);
    case FM3D_STENCIL_DECR: return (uint8_t)(s == 0 ? 0 : s - 1);
    case FM3D_STENCIL_INVERT: return (uint8_t)~s;
    case FM3D_STENCIL_INCR_WRAP: return (uint8_t)(s + 1);
    case FM3D_STENCIL_DECR_WRAP: return (uint8_t)(s - 1);
    default: return s;
    }
}

static int fm3d_cmp_u(fm3d_compare f, uint32_t a, uint32_t b)
{
    switch (f) {
    case FM3D_NEVER: return 0;
    case FM3D_LESS: return a < b;
    case FM3D_EQUAL: return a == b;
    case FM3D_LEQUAL: return a <= b;
    case FM3D_GREATER: return a > b;
    case FM3D_NOTEQUAL: return a != b;
    case FM3D_GEQUAL: return a >= b;
    default: return 1;
    }
}

/* stencil test -> depth test -> stencil ops -> depth write (GL order).
 * Depth is compared as keys in the stored precision of the depth format. */
static void fm3d_zs_stage(const fm3d_dstate* st, const fm3d_tri* t, fm3d_batch* b, int ztest, int zwrite)
{
    fm_surface* D       = st->depth;
    fm_surface* S       = st->stencil_buf;
    int         stencil = st->stencil_on && S;
    if (!D && !stencil) return;
    fm_format                       fmt    = D ? D->format : FM_FORMAT_D32F;
    if (!stencil && fmt == FM_FORMAT_D32F) {
        /* SIMD fast path: D32F without stencil */
        float lo = 2.0f, hi = -1.0f;
        int   nw = 0;
        for (int r = 0; r < 2; r++) {
            if (!fm3d_row_valid(b, r, st->color)) continue;
            fm_k->depth_f32(b->z + r * FM3D_QCOLS, fm_surface_rowf(D, b->y + r) + b->x, b->mask + r * FM3D_QCOLS,
                            b->cols, ztest ? (int)st->depth_func : (int)FM3D_ALWAYS, zwrite, &lo, &hi, &nw);
        }
        if (b->hiz && nw) {
            fm3d_hiz* h = b->hiz;
            h->kmin     = FM_MIN(h->kmin, fm3d_fkey(lo));
            h->kmax     = FM_MAX(h->kmax, fm3d_fkey(hi));
            h->written += nw;
        }
        return;
    }
    int                             packed = stencil && S->format == FM_FORMAT_D24S8; /* stencil in the depth word */
    const struct fm3d_stencil_face* sf     = &st->stencil[(t->flags & FM3D_TRI_BACK) ? 1 : 0];
    uint8_t                         ref = sf->ref, rm = sf->read_mask, wm = sf->write_mask;
    uint32_t                        rr   = (uint32_t)(ref & rm);
    uint32_t                        wmin = 0xffffffffu, wmax = 0;
    int                             nw   = 0;
    for (int r = 0; r < 2; r++) {
        if (!fm3d_row_valid(b, r, st->color)) continue;
        uint8_t* drow = D ? fm_surface_row8(D, b->y + r) : NULL;
        uint8_t* srow = (stencil && !packed) ? fm_surface_row8(S, b->y + r) + b->x : NULL;
        uint8_t* m    = b->mask + r * FM3D_QCOLS;
        float*   z    = b->z + r * FM3D_QCOLS;
        for (int c = 0; c < b->cols; c++) {
            if (!m[c]) continue;
            int      px = b->x + c;
            uint32_t word = 0, dkey = 0;
            float    zc = fm_clamp01(z[c]);
            if (drow) {
                switch (fmt) {
                case FM_FORMAT_D16: dkey = ((uint16_t*)drow)[px]; break;
                case FM_FORMAT_D24S8:
                    word = ((uint32_t*)drow)[px];
                    dkey = word & 0xffffffu;
                    break;
                default: dkey = fm3d_fkey(((float*)drow)[px]); break;
                }
            }
            uint32_t zk      = drow ? fm3d_zkey(fmt, zc) : 0;
            int      zp      = !ztest || !drow || fm3d_cmp_u(st->depth_func, zk, dkey);
            int      dirty   = 0; /* packed word changed */
            if (stencil) {
                uint8_t sv = srow ? srow[c] : (uint8_t)(word >> 24);
                int     sp = fm3d_cmp_u(sf->func, rr, (uint32_t)(sv & rm));
                uint8_t ns = fm3d_stencil_apply(!sp ? sf->sfail : (zp ? sf->dppass : sf->dpfail), sv, ref);
                ns         = (uint8_t)((ns & wm) | (sv & ~wm));
                if (srow)
                    srow[c] = ns;
                else if (ns != sv) {
                    word  = (word & 0xffffffu) | ((uint32_t)ns << 24);
                    dirty = 1;
                }
                if (!sp) zp = 0;
            }
            if (!zp) {
                m[c] = 0;
            } else if (zwrite && drow) {
                switch (fmt) {
                case FM_FORMAT_D16: ((uint16_t*)drow)[px] = (uint16_t)zk; break;
                case FM_FORMAT_D24S8:
                    word  = (word & 0xff000000u) | zk;
                    dirty = 1;
                    break;
                default: ((float*)drow)[px] = zc; break;
                }
                wmin = FM_MIN(wmin, zk);
                wmax = FM_MAX(wmax, zk);
                nw++;
            }
            if (dirty) ((uint32_t*)drow)[px] = word;
        }
    }
    if (b->hiz && nw) {
        fm3d_hiz* h = b->hiz;
        h->kmin     = FM_MIN(h->kmin, wmin);
        h->kmax     = FM_MAX(h->kmax, wmax);
        h->written += nw;
    }
}

static void fm3d_shade_batch(const fm3d_tri* t, fm3d_batch* b)
{
    const fm3d_dstate* st   = t->st;
    int                cols = b->cols;
    float              dxv[FM3D_QCOLS], dyr[2];
    for (int c = 0; c < cols; c++) dxv[c] = ((float)(b->x + c) + 0.5f) - t->x0f;
    dyr[0] = ((float)b->y + 0.5f) - t->y0f;
    dyr[1] = ((float)b->y + 1.5f) - t->y0f;

    /* depth plane + early stencil / depth (fragment stage cannot discard) */
    int late    = st->alpha_func != FM3D_ALWAYS;
    int dtest   = st->depth && st->depth_func != FM3D_ALWAYS;
    int dwrite  = st->depth && st->depth_write;
    /* the depth / stencil stage only runs when it can reject or write */
    int zs      = dtest || dwrite || (st->stencil_on && st->stencil_buf);
    if (zs && st->depth) {
        for (int r = 0; r < 2; r++) {
            float  rb = t->z[0] + t->z[2] * dyr[r];
            float* z  = b->z + r * FM3D_QCOLS;
            for (int c = 0; c < cols; c++) z[c] = rb + t->z[1] * dxv[c];
            if (t->flags & FM3D_TRI_ZCLAMP)
                for (int c = 0; c < cols; c++) z[c] = FM_CLAMP(z[c], 0.0f, 1.0f);
        }
    }
    if (zs && !late) fm3d_zs_stage(st, t, b, dtest, dwrite);
    int any = 0;
    for (int i = 0; i < FM3D_QN; i++) any |= b->mask[i];
    if (!any) return;
    /* depth / stencil only pass: no shading needed */
    if (!st->color_write && !late) return;

    /* w and varyings (perspective correct); nothing to do without varyings */
    for (int r = 0; r < 2 && b->need; r++) {
        float* w = b->w + r * FM3D_QCOLS;
        if (st->perspective) {
            float rb = t->w[0] + t->w[2] * dyr[r];
            for (int c = 0; c < cols; c++) w[c] = 1.0f / (rb + t->w[1] * dxv[c]);
        } else {
            for (int c = 0; c < cols; c++) w[c] = 1.0f;
        }
    }
    for (int k = 0; k < t->nvar; k++) {
        if (!(b->need & (1u << k))) continue;
        const float* pl = t->var + 3 * k;
        for (int r = 0; r < 2; r++) {
            float        rb = pl[0] + pl[2] * dyr[r];
            float*       v  = b->var[k] + r * FM3D_QCOLS;
            const float* w  = b->w + r * FM3D_QCOLS;
            for (int c = 0; c < cols; c++) v[c] = (rb + pl[1] * dxv[c]) * w[c];
        }
    }

    b->uniform = 0;
    st->fs(st, b);

    if (zs && late) fm3d_zs_stage(st, t, b, dtest, dwrite);
    if (!st->color_write) return;

    /* output merger */
    for (int r = 0; r < 2; r++) {
        if (!fm3d_row_valid(b, r, st->color)) continue;
        uint8_t* m = b->mask + r * FM3D_QCOLS;
        int      c0 = 0, c1 = cols;
        while (c0 < c1 && !m[c0]) c0++;
        while (c1 > c0 && !m[c1 - 1]) c1--;
        if (c0 >= c1) continue;
        if (st->opacity8 < 255) fm_k->mask_scale(m + c0, st->opacity8, c1 - c0);
        uint32_t* d = fm_surface_row32(st->color, b->y + r) + b->x + c0;
        if (b->uniform)
            fm_blend_solid(d, b->color[0], m + c0, c1 - c0, st->op);
        else
            fm_blend_span(d, b->color + r * FM3D_QCOLS + c0, m + c0, c1 - c0, st->op);
    }
}

void fm3d_raster_tri(const fm3d_tri* t, const int rc[4], fm3d_batch* b)
{
    const fm3d_dstate* st = t->st;
    int                x0 = FM_MAX(rc[0], t->minx), x1 = FM_MIN(rc[2] - 1, t->maxx);
    int                y0 = FM_MAX(rc[1], t->miny), y1 = FM_MIN(rc[3] - 1, t->maxy);
    if (x1 < x0 || y1 < y0) return;
    b->tri  = t;
    b->need = 0;
    if (st->fs == fm3d_fs_fixed) {
        if (st->tex) b->need |= (1u << FM3D_VAR_U) | (1u << FM3D_VAR_V);
        if (!(t->flags & FM3D_TRI_FLAT))
            b->need |= (1u << FM3D_VAR_R) | (1u << FM3D_VAR_G) | (1u << FM3D_VAR_B) | (1u << FM3D_VAR_A);
    } else {
        b->need = (1u << t->nvar) - 1;
    }
    for (int y = y0 & ~1; y <= y1; y += 2) {
        int lo[2], hi[2], ok[2];
        for (int r = 0; r < 2; r++) {
            int py = y + r;
            ok[r]  = py >= y0 && py <= y1 && fm3d_row_span(t, py, &lo[r], &hi[r]);
            if (ok[r]) {
                lo[r] = FM_MAX(lo[r], x0);
                hi[r] = FM_MIN(hi[r], x1);
                ok[r] = lo[r] <= hi[r];
            }
        }
        if (!ok[0] && !ok[1]) continue;
        int xa = ok[0] ? lo[0] : lo[1], xb = ok[0] ? hi[0] : hi[1];
        if (ok[0] && ok[1]) {
            xa = FM_MIN(lo[0], lo[1]);
            xb = FM_MAX(hi[0], hi[1]);
        }
        xa &= ~1;
        xb = (xb + 2) & ~1; /* exclusive, even */
        for (int bx = xa; bx < xb; bx += FM3D_QCOLS) {
            int cols = FM_MIN(FM3D_QCOLS, xb - bx);
            b->x     = bx;
            b->y     = y;
            b->cols  = cols;
            for (int r = 0; r < 2; r++) {
                uint8_t* m = b->mask + r * FM3D_QCOLS;
                for (int c = 0; c < FM3D_QCOLS; c++) {
                    int px = bx + c;
                    m[c]   = (c < cols && ok[r] && px >= lo[r] && px <= hi[r]) ? 255 : 0;
                }
            }
            fm3d_shade_batch(t, b);
        }
    }
}

/* ---- fixed function fragment stage (texenv) ------------------------------------- */

static float fm3d_log2_fast(float x)
{
    union {
        float    f;
        uint32_t i;
    } u;
    u.f     = x;
    float e = (float)(int)((u.i >> 23) & 255) - 127.0f;
    u.i     = (u.i & 0x007fffffu) | 0x3f800000u; /* mantissa in [1, 2) */
    float m = u.f - 1.0f;
    return e + m * (1.3465f - 0.3465f * m);
}

static int fm3d_alpha_pass(fm3d_compare f, uint32_t a, uint32_t ref)
{
    return fm3d_depth_pass(f, (float)a, (float)ref);
}

/* texture pass for one mip level over the quads in grp (texel coordinates
 * through the SIMD texcoord op, fetch through the sampling kernels) */
static void fm3d_sample_level(const fm3d_dstate* st, const fm3d_batch* b, const fm_sampler* s2, int lvl,
                              const int* grp, int ng, int all, uint32_t* out)
{
    const fm_surface* L  = st->tex->level[lvl];
    int               bi = s2->filter != FM_FILTER_NEAREST;
    int32_t           U[FM3D_QN], V[FM3D_QN];
    if (all) { /* every quad at this level: whole rows, no gather / scatter */
        for (int r = 0; r < 2; r++) {
            int o = r * FM3D_QCOLS;
            fm_k->texcoord(b->var[FM3D_VAR_U] + o, b->cols, (int)s2->wrap_u, (float)L->width, bi, U + o);
            fm_k->texcoord(b->var[FM3D_VAR_V] + o, b->cols, (int)s2->wrap_v, (float)L->height, bi, V + o);
            fm__sample_fixed(L, s2, U + o, V + o, b->cols, out + o);
        }
        return;
    }
    float    us[FM3D_QN], vs[FM3D_QN];
    uint32_t tx[FM3D_QN];
    int      ix[FM3D_QN], n = 0;
    for (int k = 0; k < ng; k++) {
        int q = grp[k];
        int idx[4] = { 2 * q, 2 * q + 1, FM3D_QCOLS + 2 * q, FM3D_QCOLS + 2 * q + 1 };
        for (int j = 0; j < 4; j++) {
            us[n] = b->var[FM3D_VAR_U][idx[j]];
            vs[n] = b->var[FM3D_VAR_V][idx[j]];
            ix[n] = idx[j];
            n++;
        }
    }
    fm_k->texcoord(us, n, (int)s2->wrap_u, (float)L->width, bi, U);
    fm_k->texcoord(vs, n, (int)s2->wrap_v, (float)L->height, bi, V);
    fm__sample_fixed(L, s2, U, V, n, tx);
    for (int j = 0; j < n; j++) out[ix[j]] = tx[j];
}

void fm3d_fs_fixed(const fm3d_dstate* st, fm3d_batch* b)
{
    const fm3d_tri* t    = b->tri;
    int             cols = b->cols;
    int             nq   = cols / 2;
    int             flat = (t->flags & FM3D_TRI_FLAT) != 0;
    int             act[FM3D_QCOLS / 2], na = 0;
    for (int q = 0; q < nq; q++) {
        int i = 2 * q;
        if (b->mask[i] | b->mask[i + 1] | b->mask[FM3D_QCOLS + i] | b->mask[FM3D_QCOLS + i + 1]) act[na++] = q;
    }
    if (!na) return;

    /* flat color, no texture: one color for the batch (solid blend path) */
    if (flat && !st->tex) {
        b->color[0] = t->flat;
        b->uniform  = 1;
        if (st->alpha_func != FM3D_ALWAYS && !fm3d_alpha_pass(st->alpha_func, t->flat >> 24, st->alpha_ref8))
            memset(b->mask, 0, sizeof(b->mask));
        return;
    }

    /* vertex color (SIMD premultiply), whole rows */
    uint32_t vc[FM3D_QN];
    for (int r = 0; r < 2; r++) {
        int o = r * FM3D_QCOLS;
        if (flat) {
            for (int c = 0; c < cols; c++) vc[o + c] = t->flat;
        } else {
            fm_k->premul_f(b->var[FM3D_VAR_R] + o, b->var[FM3D_VAR_G] + o, b->var[FM3D_VAR_B] + o,
                           b->var[FM3D_VAR_A] + o, cols, vc + o);
        }
    }

    if (!st->tex) {
        memcpy(b->color, vc, sizeof(vc));
    } else {
        /* per quad LOD from the quad derivatives (GPU semantics) */
        const fm3d_texture* tex  = st->tex;
        fm3d_filter         f    = st->sampler.filter;
        int                 mip  = f >= FM3D_FILTER_NEAREST_MIPMAP && tex->levels > 1;
        float               maxl = (float)(tex->levels - 1);
        int                 la[FM3D_QCOLS / 2], lb[FM3D_QCOLS / 2];
        uint32_t            fw[FM3D_QCOLS / 2];
        const float*        U = b->var[FM3D_VAR_U];
        const float*        V = b->var[FM3D_VAR_V];
        float               W0 = (float)tex->level[0]->width, H0 = (float)tex->level[0]->height;
        fm_sampler          s2;
        s2.wrap_u = st->sampler.wrap_u;
        s2.wrap_v = st->sampler.wrap_v;
        s2.filter = (f == FM3D_FILTER_NEAREST || f == FM3D_FILTER_NEAREST_MIPMAP) ? FM_FILTER_NEAREST : FM_FILTER_BILINEAR;
        int any_fw = 0;
        for (int k = 0; k < na; k++) {
            int q = act[k];
            la[k] = 0;
            lb[k] = 0;
            fw[k] = 0;
            if (!mip) continue;
            int   i0 = 2 * q, i1 = i0 + 1, i2 = FM3D_QCOLS + i0;
            float dudx = (U[i1] - U[i0]) * W0, dvdx = (V[i1] - V[i0]) * H0;
            float dudy = (U[i2] - U[i0]) * W0, dvdy = (V[i2] - V[i0]) * H0;
            float rho2 = FM_MAX(dudx * dudx + dvdx * dvdx, dudy * dudy + dvdy * dvdy);
            float lod  = (rho2 > 0.0f ? 0.5f * fm3d_log2_fast(rho2) : -100.0f) + st->sampler.lod_bias;
            if (f == FM3D_FILTER_TRILINEAR) {
                if (lod > 0.0f) {
                    float l = FM_MIN(lod, maxl);
                    la[k]   = (int)fm_floorf(l);
                    lb[k]   = FM_MIN(la[k] + 1, tex->levels - 1);
                    fw[k]   = (uint32_t)((l - (float)la[k]) * 256.0f);
                    any_fw |= fw[k] != 0;
                }
            } else {
                la[k] = (int)fm_floorf(FM_CLAMP(lod + 0.5f, 0.0f, maxl));
            }
        }
        /* sample: one pass per distinct level (usually one or two) */
        uint32_t ta[FM3D_QN], tb[FM3D_QN];
        for (int pass = 0; pass < (any_fw ? 2 : 1); pass++) {
            int done[FM3D_QCOLS / 2], nd = 0;
            for (int k = 0; k < na; k++) {
                done[k] = pass == 1 && fw[k] == 0; /* second level only where blended */
                nd += done[k];
            }
            int first = 1;
            while (nd < na) {
                int lvl = -1, grp[FM3D_QCOLS / 2], ng = 0;
                for (int k = 0; k < na; k++) {
                    if (done[k]) continue;
                    int l = pass ? lb[k] : la[k];
                    if (lvl < 0) lvl = l;
                    if (l == lvl) {
                        grp[ng++] = act[k];
                        done[k]   = 1;
                        nd++;
                    }
                }
                /* whole rows when a single level covers every active quad */
                fm3d_sample_level(st, b, &s2, lvl, grp, ng, first && nd == na && ng == na, pass ? tb : ta);
                first = 0;
            }
        }
        uint32_t* tex_px = ta;
        if (any_fw) {
            uint8_t f8[FM3D_QN];
            memset(f8, 0, sizeof(f8));
            for (int k = 0; k < na; k++) {
                int q = act[k];
                if (!fw[k]) { /* no blend: keep level a */
                    tb[2 * q] = ta[2 * q], tb[2 * q + 1] = ta[2 * q + 1];
                    tb[FM3D_QCOLS + 2 * q] = ta[FM3D_QCOLS + 2 * q], tb[FM3D_QCOLS + 2 * q + 1] = ta[FM3D_QCOLS + 2 * q + 1];
                }
                f8[2 * q] = f8[2 * q + 1] = f8[FM3D_QCOLS + 2 * q] = f8[FM3D_QCOLS + 2 * q + 1] = (uint8_t)fw[k];
            }
            for (int r = 0; r < 2; r++) {
                int o = r * FM3D_QCOLS;
                fm_k->lerp8(ta + o, tb + o, f8 + o, cols, ta + o);
            }
        }
        for (int r = 0; r < 2; r++) {
            int o = r * FM3D_QCOLS;
            fm_k->combine((int)st->texenv, tex_px + o, vc + o, cols, b->color + o);
        }
    }
    if (st->alpha_func != FM3D_ALWAYS) {
        for (int k = 0; k < na; k++) {
            int q = act[k];
            int idx[4] = { 2 * q, 2 * q + 1, FM3D_QCOLS + 2 * q, FM3D_QCOLS + 2 * q + 1 };
            for (int j = 0; j < 4; j++)
                if (!fm3d_alpha_pass(st->alpha_func, b->color[idx[j]] >> 24, st->alpha_ref8)) b->mask[idx[j]] = 0;
        }
    }
}
