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
#if FM_ARCH_X86
#  include <emmintrin.h> /* SSE2: baseline on x86-64 */
#elif FM_ARCH_ARM64
#  include <arm_neon.h>
#endif

/* D3D11 standard sample patterns (1/16 pixel units from the pixel center) */
const int8_t fm3d_samples4[4][2] = { { -2, -6 }, { 6, -2 }, { -6, 2 }, { 2, 6 } };
const int8_t fm3d_samples8[8][2] = { { 1, -3 }, { -1, 3 }, { 5, 1 }, { -3, -5 }, { -5, 5 }, { -7, -1 }, { 3, 7 }, { 7, -7 } };

static const int8_t (*fm3d_pattern(int S))[2] { return S == 8 ? fm3d_samples8 : fm3d_samples4; }

/* ---- vertex stage ------------------------------------------------------------ */

#if FM_FEATURE_TNL
/* lit vertex colors for one block (<= 256): eye space positions / normals
 * into SoA arrays, then the SIMD lighting kernel. sp: skinned positions
 * (3 floats per vertex) or NULL. */
static void fm3d_vs_light(const fm3d_dstate* st, const fm3d_vertex* in, int n, const float* sp, float* lit[4])
{
    float          buf[10][256];
    const fm_mat4* mv = &st->mv;
    const float*   N  = st->nrm;
    float          sn[3 * 256];
    if (sp) { /* skinned normals: M * (p + n) - M * p = M * n for the blended matrix */
        float pn[3 * 256];
        for (int i = 0; i < n; i++) {
            pn[3 * i]     = in[i].x + in[i].nx;
            pn[3 * i + 1] = in[i].y + in[i].ny;
            pn[3 * i + 2] = in[i].z + in[i].nz;
        }
        fm_k->skin4(st->bones, st->nbones, st->skin + (in - st->skin_vbase), (int)sizeof(fm3d_skin_vertex), pn, 3, n, sn);
        for (int i = 0; i < 3 * n; i++) sn[i] -= sp[i];
    }
    for (int i = 0; i < n; i++) {
        const fm3d_vertex* v = &in[i];
        float              x = sp ? sp[3 * i] : v->x, y = sp ? sp[3 * i + 1] : v->y, z = sp ? sp[3 * i + 2] : v->z;
        float              nx = sp ? sn[3 * i] : v->nx, ny = sp ? sn[3 * i + 1] : v->ny, nz = sp ? sn[3 * i + 2] : v->nz;
        buf[0][i] = mv->c[0].x * x + mv->c[1].x * y + mv->c[2].x * z + mv->c[3].x;
        buf[1][i] = mv->c[0].y * x + mv->c[1].y * y + mv->c[2].y * z + mv->c[3].y;
        buf[2][i] = mv->c[0].z * x + mv->c[1].z * y + mv->c[2].z * z + mv->c[3].z;
        buf[3][i] = N[0] * nx + N[3] * ny + N[6] * nz;
        buf[4][i] = N[1] * nx + N[4] * ny + N[7] * nz;
        buf[5][i] = N[2] * nx + N[5] * ny + N[8] * nz;
        buf[6][i] = (float)((v->color >> 16) & 255) * (1.0f / 255.0f);
        buf[7][i] = (float)((v->color >> 8) & 255) * (1.0f / 255.0f);
        buf[8][i] = (float)(v->color & 255) * (1.0f / 255.0f);
        buf[9][i] = (float)(v->color >> 24) * (1.0f / 255.0f);
    }
    const float* ip[10];
    for (int k = 0; k < 10; k++) ip[k] = buf[k];
    fm_k->light(st->lp, ip, n, lit);
}
#endif

void fm3d_vs_fixed(const fm3d_dstate* st, const void* vin, int n, fm3d_vout* out)
{
    const fm3d_vertex* in = (const fm3d_vertex*)vin; /* the fixed stage reads fm3d_vertex */
#if FM_FEATURE_TNL
    float lr[256], lg[256], lb[256], la[256];
    float* lit[4] = { lr, lg, lb, la };
#endif
    const fm_mat4* m = &st->mvp;
    float          sp[3 * 256];
    for (int i = 0; i < n; i++) {
        const fm3d_vertex* v = &in[i];
        fm3d_vout*         o = &out[i];
        float              x = v->x, y = v->y, z = v->z;
        if (st->skin) { /* vertex blending, 256 vertices at a time through the SIMD kernel */
            if ((i & 255) == 0) {
                int blk = FM_MIN(256, n - i);
                fm_k->skin4(st->bones, st->nbones, st->skin + (in + i - st->skin_vbase), (int)sizeof(fm3d_skin_vertex),
                            &v->x, (int)(sizeof(fm3d_vertex) / sizeof(float)), blk, sp);
            }
            x = sp[3 * (i & 255)];
            y = sp[3 * (i & 255) + 1];
            z = sp[3 * (i & 255) + 2];
        }
#if FM_FEATURE_TNL
        if (st->lp && (i & 255) == 0) fm3d_vs_light(st, in + i, FM_MIN(256, n - i), st->skin ? sp : NULL, lit);
#endif
        o->pos[0] = m->c[0].x * x + m->c[1].x * y + m->c[2].x * z + m->c[3].x;
        o->pos[1] = m->c[0].y * x + m->c[1].y * y + m->c[2].y * z + m->c[3].y;
        o->pos[2] = m->c[0].z * x + m->c[1].z * y + m->c[2].z * z + m->c[3].z;
        o->pos[3] = m->c[0].w * x + m->c[1].w * y + m->c[2].w * z + m->c[3].w;
        o->var[FM3D_VAR_U] = v->u;
        o->var[FM3D_VAR_V] = v->v;
        o->var[FM3D_VAR_R] = (float)((v->color >> 16) & 255) * (1.0f / 255.0f);
        o->var[FM3D_VAR_G] = (float)((v->color >> 8) & 255) * (1.0f / 255.0f);
        o->var[FM3D_VAR_B] = (float)(v->color & 255) * (1.0f / 255.0f);
        o->var[FM3D_VAR_A] = (float)(v->color >> 24) * (1.0f / 255.0f);
#if FM_FEATURE_TNL
        if (st->lp) {
            o->var[FM3D_VAR_R] = lr[i & 255];
            o->var[FM3D_VAR_G] = lg[i & 255];
            o->var[FM3D_VAR_B] = lb[i & 255];
            o->var[FM3D_VAR_A] = la[i & 255];
        }
#endif
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
    if (st->msaa > 1) { /* samples reach up to 7/16 px beyond the centers */
        minx--;
        miny--;
        maxx++;
        maxy++;
    }
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

/* inclusive span [lo, hi] of pixels whose sample at (ox, oy) (1/16 px from
 * the center) is inside the triangle on row py */
static int fm3d_row_span_at(const fm3d_tri* t, int py, int ox, int oy, int* lo, int* hi)
{
    int64_t yc = 16 * (int64_t)py + 8 + oy;
    int64_t l = INT32_MIN, h = INT32_MAX;
    for (int e = 0; e < 3; e++) {
        int64_t A = t->A[e], K = t->K0[e] + ox * A + t->B[e] * yc;
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

static int fm3d_row_span(const fm3d_tri* t, int py, int* lo, int* hi) { return fm3d_row_span_at(t, py, 0, 0, lo, hi); }

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

/* MSAA depth / stencil per sample; pixels keep coverage if any sample passes */
static void fm3d_zs_ms(const fm3d_dstate* st, const fm3d_tri* t, fm3d_batch* b, int ztest, int zwrite)
{
    int                 S       = st->msaa;
    const int8_t(*pat)[2]       = fm3d_pattern(S);
    int                 stencil = st->stencil_on && st->stencil_buf;
    int                 depth   = st->depth != NULL;
    float               dzs[FM3D_MAX_SAMPLES];
    for (int k = 0; k < S; k++) dzs[k] = t->z[1] * ((float)pat[k][0] * 0.0625f) + t->z[2] * ((float)pat[k][1] * 0.0625f);
    const struct fm3d_stencil_face* sf = &st->stencil[(t->flags & FM3D_TRI_BACK) ? 1 : 0];
    for (int r = 0; r < 2; r++) {
        if (!fm3d_row_valid(b, r, st->color)) continue;
        size_t   base = ((size_t)(b->y + r) * (size_t)st->ms_w + (size_t)b->x) * (size_t)S;
        float*   zb   = st->ms_depth + base;
        uint8_t* sb   = st->ms_stencil + base;
        uint8_t* sm   = b->smask + r * FM3D_QCOLS;
        float*   z    = b->z + r * FM3D_QCOLS;
        if (!stencil) {
            if (depth) fm_k->depth_ms(z, dzs, S, zb, sm, b->cols, ztest ? (int)st->depth_func : (int)FM3D_ALWAYS, zwrite);
        } else {
            uint8_t ref = sf->ref, rm = sf->read_mask, wm = sf->write_mask;
            for (int c = 0; c < b->cols; c++) {
                uint8_t bits = sm[c];
                if (!bits) continue;
                for (int k = 0; k < S; k++) {
                    if (!(bits & (1u << k))) continue;
                    size_t  i  = (size_t)c * (size_t)S + (size_t)k;
                    float   zs = fm_clamp01(z[c] + dzs[k]);
                    int     zp = !ztest || !depth || fm3d_depth_pass(st->depth_func, zs, zb[i]);
                    uint8_t sv = sb[i];
                    int     sp = fm3d_cmp_u(sf->func, (uint32_t)(ref & rm), (uint32_t)(sv & rm));
                    uint8_t ns = fm3d_stencil_apply(!sp ? sf->sfail : (zp ? sf->dppass : sf->dpfail), sv, ref);
                    sb[i]      = (uint8_t)((ns & wm) | (sv & ~wm));
                    if (!sp || !zp)
                        bits = (uint8_t)(bits & ~(1u << k));
                    else if (zwrite && depth)
                        zb[i] = zs;
                }
                sm[c] = bits;
            }
        }
        uint8_t* m = b->mask + r * FM3D_QCOLS;
        for (int c = 0; c < b->cols; c++)
            if (!sm[c]) m[c] = 0;
    }
}

static void fm3d_merge_ms(const fm3d_dstate* st, fm3d_batch* b)
{
    int S = st->msaa;
    for (int r = 0; r < 2; r++) {
        if (!fm3d_row_valid(b, r, st->color)) continue;
        uint8_t* m  = b->mask + r * FM3D_QCOLS;
        uint8_t* sm = b->smask + r * FM3D_QCOLS;
        int      c0 = 0, c1 = b->cols;
        while (c0 < c1 && !m[c0]) c0++;
        while (c1 > c0 && !m[c1 - 1]) c1--;
        if (c0 >= c1) continue;
        uint32_t src[FM3D_QCOLS * FM3D_MAX_SAMPLES];
        uint8_t  cov[FM3D_QCOLS * FM3D_MAX_SAMPLES];
        int      n = (c1 - c0) * S;
        for (int c = c0; c < c1; c++) {
            uint32_t col  = b->uniform ? b->color[0] : b->color[r * FM3D_QCOLS + c];
            uint8_t  bits = m[c] ? sm[c] : 0;
            for (int k = 0; k < S; k++) {
                src[(c - c0) * S + k] = col;
                cov[(c - c0) * S + k] = (bits >> k) & 1 ? 255 : 0;
            }
        }
        if (st->opacity8 < 255) fm_k->mask_scale(cov, st->opacity8, n);
        uint32_t* d = st->ms_color + ((size_t)(b->y + r) * (size_t)st->ms_w + (size_t)(b->x + c0)) * (size_t)S;
        if (b->uniform)
            fm_blend_solid(d, b->color[0], cov, n, st->op);
        else
            fm_blend_span(d, src, cov, n, st->op);
    }
}

/* Plane interpolation for one batch row. Wide rows use the SIMD kernels;
 * narrow ones (small triangles) inline the same expression, which is
 * cheaper than an indirect call and gives the same bits (see fm_plane1). */
#define FM3D_KERNEL_MIN_COLS 8

FM_INLINE void fm3d_interp(float a, float bx, const float* dx, int n, int clamp01, float* out)
{
    if (n >= FM3D_KERNEL_MIN_COLS) {
        fm_k->plane(a, bx, dx, n, clamp01, out);
        return;
    }
    for (int c = 0; c < n; c++) {
        float v = a + bx * dx[c];
        out[c]  = clamp01 ? fm_clamp01(v) : v;
    }
}

FM_INLINE void fm3d_interp_recip(float a, float bx, const float* dx, int n, float* out)
{
    if (n >= FM3D_KERNEL_MIN_COLS) {
        fm_k->plane_recip(a, bx, dx, n, out);
        return;
    }
    for (int c = 0; c < n; c++) out[c] = 1.0f / (a + bx * dx[c]);
}

FM_INLINE void fm3d_interp_mul(float a, float bx, const float* dx, const float* w, int n, float* out)
{
    if (n >= FM3D_KERNEL_MIN_COLS) {
        fm_k->plane_mul(a, bx, dx, w, n, out);
        return;
    }
    for (int c = 0; c < n; c++) out[c] = (a + bx * dx[c]) * w[c];
}

/* ---- batch mask rows as 64 bit words (FM3D_QCOLS = 32 bytes = 4 words; all
 * targets are little endian: byte i of a word is bits 8i .. 8i + 7) ---- */

#if FM3D_QCOLS != 32
#  error "mask word helpers assume 32 column batches"
#endif

FM_INLINE uint64_t fm3d_ld64(const uint8_t* p)
{
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}
FM_INLINE void fm3d_st64(uint8_t* p, uint64_t w) { memcpy(p, &w, 8); }

FM_INLINE int fm3d_ctz64(uint64_t x) /* x != 0 */
{
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanForward64(&i, x);
    return (int)i;
#else
    return __builtin_ctzll(x);
#endif
}
FM_INLINE int fm3d_msb64(uint64_t x) /* x != 0: index of the highest set bit */
{
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanReverse64(&i, x);
    return (int)i;
#else
    return 63 - __builtin_clzll(x);
#endif
}

/* row m = 255 for columns [c0, c1), 0 elsewhere (0 <= c0, c1 <= 32): two
 * byte compares of a column index vector (SSE2 / NEON are baseline) */
FM_INLINE void fm3d_mask_fill(uint8_t* m, int c0, int c1)
{
#if FM_ARCH_X86
    const __m128i i0 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i i1 = _mm_add_epi8(i0, _mm_set1_epi8(16));
    const __m128i lo = _mm_set1_epi8((char)(c0 - 1)), hi = _mm_set1_epi8((char)c1);
    _mm_storeu_si128((__m128i*)m, _mm_and_si128(_mm_cmpgt_epi8(i0, lo), _mm_cmplt_epi8(i0, hi)));
    _mm_storeu_si128((__m128i*)(m + 16), _mm_and_si128(_mm_cmpgt_epi8(i1, lo), _mm_cmplt_epi8(i1, hi)));
#elif FM_ARCH_ARM64
    static const int8_t idx[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const int8x16_t     i0 = vld1q_s8(idx), i1 = vaddq_s8(i0, vdupq_n_s8(16));
    const int8x16_t     lo = vdupq_n_s8((int8_t)c0), hi = vdupq_n_s8((int8_t)c1);
    vst1q_u8(m, vandq_u8(vcgeq_s8(i0, lo), vcltq_s8(i0, hi)));
    vst1q_u8(m + 16, vandq_u8(vcgeq_s8(i1, lo), vcltq_s8(i1, hi)));
#else
    for (int c = 0; c < FM3D_QCOLS; c++) m[c] = (c >= c0 && c < c1) ? 255 : 0;
#endif
}

/* any nonzero byte in the batch (both rows) */
FM_INLINE int fm3d_mask_any(const uint8_t* m)
{
    uint64_t a = 0;
    for (int k = 0; k < 8; k++) a |= fm3d_ld64(m + 8 * k);
    return a != 0;
}

/* first / one past last nonzero byte of a row; 0 if the row is empty */
FM_INLINE int fm3d_mask_trim(const uint8_t* m, int* c0, int* c1)
{
    int k = 0;
    while (k < 4 && !fm3d_ld64(m + 8 * k)) k++;
    if (k == 4) return 0;
    *c0 = 8 * k + (fm3d_ctz64(fm3d_ld64(m + 8 * k)) >> 3);
    int j = 3;
    while (!fm3d_ld64(m + 8 * j)) j--;
    *c1 = 8 * j + (fm3d_msb64(fm3d_ld64(m + 8 * j)) >> 3) + 1;
    return 1;
}

/* number of 255 bytes in the batch mask (bytes are 0 or 255) */
FM_INLINE uint64_t fm3d_mask_count(const uint8_t* m)
{
    uint64_t n = 0;
    for (int k = 0; k < 8; k++) {
        uint64_t w = fm3d_ld64(m + 8 * k) & 0x0101010101010101ull; /* one bit per byte */
        n += (w * 0x0101010101010101ull) >> 56;                   /* horizontal byte sum */
    }
    return n;
}

/* all mask bytes of both batch rows 255? (8 bytes at a time) */
FM_INLINE int fm3d_mask_full(const fm3d_batch* b)
{
    int cols = b->cols;
    for (int r = 0; r < 2; r++) {
        const uint8_t* m = b->mask + r * FM3D_QCOLS;
        int            c = 0;
        for (; c + 8 <= cols; c += 8) {
            uint64_t w;
            memcpy(&w, m + c, 8);
            if (w != ~(uint64_t)0) return 0;
        }
        for (; c < cols; c++)
            if (m[c] != 255) return 0;
    }
    return 1;
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
#if FM_FEATURE_SHADERS
    late |= st->fs_discards; /* the shader may discard: depth / stencil after it */
#endif
    int dtest   = st->depth && st->depth_func != FM3D_ALWAYS;
    int dwrite  = st->depth && st->depth_write;
    /* the depth / stencil stage only runs when it can reject or write */
    int zs      = dtest || dwrite || (st->stencil_on && st->stencil_buf);
    if (zs && st->depth) {
        int zclamp = (t->flags & FM3D_TRI_ZCLAMP) != 0;
        for (int r = 0; r < 2; r++)
            fm3d_interp(t->z[0] + t->z[2] * dyr[r], t->z[1], dxv, cols, zclamp, b->z + r * FM3D_QCOLS);
    }
    int msaa = st->msaa > 1;
    if (!msaa) b->frag_in += b->full ? (uint64_t)(2 * cols) : fm3d_mask_count(b->mask);
    if (zs && !late) {
        if (msaa)
            fm3d_zs_ms(st, t, b, dtest, dwrite);
        else
            fm3d_zs_stage(st, t, b, dtest, dwrite);
    }
    if (b->full && zs && !late) b->full = fm3d_mask_full(b); /* depth / stencil may have rejected pixels */
    if (!b->full && !fm3d_mask_any(b->mask)) return;
    /* depth / stencil only pass: no shading needed */
    if (!st->color_write && !late) return;

    /* w and varyings (perspective correct); nothing to do without varyings */
    for (int r = 0; r < 2 && b->need; r++) {
        float* w = b->w + r * FM3D_QCOLS;
        if (st->perspective)
            fm3d_interp_recip(t->w[0] + t->w[2] * dyr[r], t->w[1], dxv, cols, w);
        else
            for (int c = 0; c < cols; c++) w[c] = 1.0f;
    }
    for (int k = 0; k < t->nvar; k++) {
        if (!(b->need & (1u << k))) continue;
        const float* pl = t->var + 3 * k;
        for (int r = 0; r < 2; r++)
            fm3d_interp_mul(pl[0] + pl[2] * dyr[r], pl[1], dxv, b->w + r * FM3D_QCOLS, cols,
                            b->var[k] + r * FM3D_QCOLS);
    }

    if (!msaa) b->frag_shaded += b->full ? (uint64_t)(2 * cols) : fm3d_mask_count(b->mask);
    b->uniform = 0;
    if (late) b->full = 0; /* alpha test clears mask bytes */
    st->fs(st, b);

    if (zs && late) {
        if (msaa) {
            for (int i = 0; i < FM3D_QN; i++)
                if (!b->mask[i]) b->smask[i] = 0; /* alpha test killed the pixel */
            fm3d_zs_ms(st, t, b, dtest, dwrite);
        } else {
            fm3d_zs_stage(st, t, b, dtest, dwrite);
        }
    }
    if (!st->color_write) return;
    if (msaa) {
        fm3d_merge_ms(st, b);
        return;
    }

    /* output merger */
    if (b->full && st->opacity8 == 255) {
        /* fully covered: unmasked blend kernels, no trimming */
        for (int r = 0; r < 2; r++) {
            uint32_t* d = fm_surface_row32(st->color, b->y + r) + b->x;
            if (b->uniform)
                fm_blend_solid(d, b->color[0], NULL, cols, st->op);
            else
                fm_blend_span(d, b->color + r * FM3D_QCOLS, NULL, cols, st->op);
        }
        return;
    }
    for (int r = 0; r < 2; r++) {
        if (!fm3d_row_valid(b, r, st->color)) continue;
        uint8_t* m = b->mask + r * FM3D_QCOLS;
        int      c0, c1; /* bytes past cols are always 0 */
        if (!fm3d_mask_trim(m, &c0, &c1)) continue;
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
    int                 S   = st->msaa > 1 ? st->msaa : 1;
    const int8_t(*pat)[2]   = fm3d_pattern(S);
    int slo[2][FM3D_MAX_SAMPLES], shi[2][FM3D_MAX_SAMPLES], sok[2][FM3D_MAX_SAMPLES];
    for (int y = y0 & ~1; y <= y1; y += 2) {
        int lo[2], hi[2], ok[2];
        for (int r = 0; r < 2; r++) {
            int py = y + r;
            if (S == 1) {
                ok[r] = py >= y0 && py <= y1 && fm3d_row_span(t, py, &lo[r], &hi[r]);
                if (ok[r]) {
                    lo[r] = FM_MAX(lo[r], x0);
                    hi[r] = FM_MIN(hi[r], x1);
                    ok[r] = lo[r] <= hi[r];
                }
                continue;
            }
            /* MSAA: one span per sample position, pixel span = union */
            ok[r] = 0;
            lo[r] = INT32_MAX;
            hi[r] = INT32_MIN;
            for (int k = 0; k < S; k++) {
                sok[r][k] = py >= y0 && py <= y1 && fm3d_row_span_at(t, py, pat[k][0], pat[k][1], &slo[r][k], &shi[r][k]);
                if (sok[r][k]) {
                    slo[r][k] = FM_MAX(slo[r][k], x0);
                    shi[r][k] = FM_MIN(shi[r][k], x1);
                    sok[r][k] = slo[r][k] <= shi[r][k];
                }
                if (sok[r][k]) {
                    ok[r] = 1;
                    lo[r] = FM_MIN(lo[r], slo[r][k]);
                    hi[r] = FM_MAX(hi[r], shi[r][k]);
                }
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
            b->full  = S == 1 && ok[0] && ok[1] && lo[0] <= bx && lo[1] <= bx && hi[0] >= bx + cols - 1 &&
                      hi[1] >= bx + cols - 1;
            for (int r = 0; r < 2; r++) {
                uint8_t* m = b->mask + r * FM3D_QCOLS;
                if (S == 1) {
                    /* the row span is exact: expand [lo, hi] into the byte mask */
                    if (ok[r])
                        fm3d_mask_fill(m, FM_MAX(lo[r] - bx, 0), FM_MIN(hi[r] - bx + 1, cols));
                    else
                        fm3d_mask_fill(m, 0, 0);
                    continue;
                }
                uint8_t* sm = b->smask + r * FM3D_QCOLS;
                for (int c = 0; c < FM3D_QCOLS; c++) {
                    int     px   = bx + c;
                    uint8_t bits = 0;
                    if (c < cols && ok[r])
                        for (int k = 0; k < S; k++)
                            bits |= (uint8_t)((sok[r][k] && px >= slo[r][k] && px <= shi[r][k]) << k);
                    sm[c] = bits;
                    m[c]  = bits ? 255 : 0;
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

#if FM_FEATURE_SHADERS
/* program vertex stage: the shader writes straight into the vout records */
void fm3d_vs_program(const fm3d_dstate* st, const void* in, int n, fm3d_vout* out)
{
    fm3d_vs_io io;
    io.vertices   = in;
    io.stride     = st->vstride;
    io.count      = n;
    io.uniforms   = st->uniforms;
    io.pos        = out->pos;
    io.varyings   = out->var;
    io.out_stride = (int)(sizeof(fm3d_vout) / sizeof(float));
    io.user       = st->user;
    io.uniform_size = st->uniform_size;
    const fm3d_texture* units[FM3D_MAX_TEXTURE_UNITS];
    fm3d_sampler        us[FM3D_MAX_TEXTURE_UNITS];
    for (int u = 0; u < FM3D_MAX_TEXTURE_UNITS; u++) units[u] = u ? st->units[u] : st->tex, us[u] = u ? st->usamp[u] : st->sampler;
    io.textures = units;
    io.samplers = us;
    st->user_vs(&io);
}

/* program fragment stage: float RGBA from the shader -> premultiplied
 * ARGB (SIMD premul_f), alpha test as in the fixed stage */
void fm3d_fs_program(const fm3d_dstate* st, fm3d_batch* b)
{
    float       rgba[4][FM3D_QN];
    const float* vp[FM3D_MAX_VARYINGS];
    for (int k = 0; k < st->nvar; k++) vp[k] = b->var[k];
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < FM3D_QN; i++) rgba[k][i] = 0.0f;
    fm3d_fs_io io;
    io.x        = b->x;
    io.y        = b->y;
    io.cols     = b->cols;
    io.varyings = vp;
    io.z        = b->z;
    io.mask     = b->mask;
    for (int k = 0; k < 4; k++) io.out[k] = rgba[k];
    io.uniforms = st->uniforms;
    io.texture  = st->tex;
    io.sampler  = &st->sampler;
    io.user     = st->user;
    io.uniform_size = st->uniform_size;
    const fm3d_texture* units[FM3D_MAX_TEXTURE_UNITS];
    fm3d_sampler        us[FM3D_MAX_TEXTURE_UNITS];
    for (int u = 0; u < FM3D_MAX_TEXTURE_UNITS; u++) units[u] = u ? st->units[u] : st->tex, us[u] = u ? st->usamp[u] : st->sampler;
    io.textures = units;
    io.samplers = us;
    st->user_fs(&io);
    for (int r = 0; r < 2; r++) {
        int o = r * FM3D_QCOLS;
        fm_k->premul_f(rgba[0] + o, rgba[1] + o, rgba[2] + o, rgba[3] + o, b->cols, b->color + o);
    }
    if (st->alpha_func != FM3D_ALWAYS)
        for (int i = 0; i < FM3D_QN; i++)
            if (b->mask[i] && !fm3d_alpha_pass(st->alpha_func, b->color[i] >> 24, st->alpha_ref8)) b->mask[i] = 0;
}
#endif

#if FM_FEATURE_SHADERS
void fm3d_sample_batch(const fm3d_fs_io* io, const fm3d_texture* tex, const fm3d_sampler* s, const float* U,
                       const float* V, float* r, float* g, float* bo, float* a)
{
    if (!io || !tex || !s) return;
    fm3d_filter f    = s->filter;
    int         mip  = f >= FM3D_FILTER_NEAREST_MIPMAP && tex->levels > 1;
    float       maxl = (float)(tex->levels - 1);
    float       W0 = (float)tex->level[0]->width, H0 = (float)tex->level[0]->height;
    fm_sampler  s2;
    s2.wrap_u = s->wrap_u;
    s2.wrap_v = s->wrap_v;
    s2.filter = (f == FM3D_FILTER_NEAREST || f == FM3D_FILTER_NEAREST_MIPMAP) ? FM_FILTER_NEAREST : FM_FILTER_BILINEAR;
    int      nq = io->cols / 2, la[FM3D_QCOLS / 2], lb[FM3D_QCOLS / 2];
    float    fw[FM3D_QCOLS / 2];
    uint32_t pa[FM3D_QN], pb[FM3D_QN];
    /* per quad level of detail (the fixed pipeline's formula) */
    for (int q = 0; q < nq; q++) {
        la[q] = lb[q] = 0;
        fw[q]         = 0.0f;
        if (!mip) continue;
        int   i0 = 2 * q, i1 = i0 + 1, i2 = FM3D_QCOLS + i0;
        float dudx = (U[i1] - U[i0]) * W0, dvdx = (V[i1] - V[i0]) * H0;
        float dudy = (U[i2] - U[i0]) * W0, dvdy = (V[i2] - V[i0]) * H0;
        float rho2 = FM_MAX(dudx * dudx + dvdx * dvdx, dudy * dudy + dvdy * dvdy);
        float lod  = (rho2 > 0.0f ? 0.5f * fm3d_log2_fast(rho2) : -100.0f) + s->lod_bias;
        if (f == FM3D_FILTER_TRILINEAR) {
            if (lod > 0.0f) {
                float l = FM_MIN(lod, maxl);
                la[q]   = (int)fm_floorf(l);
                lb[q]   = FM_MIN(la[q] + 1, tex->levels - 1);
                fw[q]   = l - (float)la[q];
            }
        } else {
            la[q] = lb[q] = (int)fm_floorf(FM_CLAMP(lod + 0.5f, 0.0f, maxl));
        }
    }
    /* one sampling pass per distinct level (texel space points) */
    for (int pass = 0; pass < 2; pass++) {
        int done[FM3D_QCOLS / 2];
        for (int q = 0; q < nq; q++) done[q] = pass == 1 && fw[q] == 0.0f;
        for (;;) {
            int lvl = -1;
            for (int q = 0; q < nq && lvl < 0; q++)
                if (!done[q]) lvl = pass ? lb[q] : la[q];
            if (lvl < 0) break;
            const fm_surface* L = tex->level[lvl];
            float             us[FM3D_QN], vs[FM3D_QN];
            int               ix[FM3D_QN], n = 0;
            for (int q = 0; q < nq; q++) {
                if (done[q] || (pass ? lb[q] : la[q]) != lvl) continue;
                done[q]    = 1;
                int idx[4] = { 2 * q, 2 * q + 1, FM3D_QCOLS + 2 * q, FM3D_QCOLS + 2 * q + 1 };
                for (int j = 0; j < 4; j++) {
                    us[n]   = U[idx[j]] * (float)L->width;
                    vs[n]   = V[idx[j]] * (float)L->height;
                    ix[n++] = idx[j];
                }
            }
            uint32_t px[FM3D_QN];
            fm_sample_points(L, &s2, us, vs, n, px);
            for (int j = 0; j < n; j++) (pass ? pb : pa)[ix[j]] = px[j];
        }
    }
    for (int q = 0; q < nq; q++) {
        int idx[4] = { 2 * q, 2 * q + 1, FM3D_QCOLS + 2 * q, FM3D_QCOLS + 2 * q + 1 };
        for (int j = 0; j < 4; j++) {
            int      i = idx[j];
            uint32_t p = pa[i];
            float    w = fw[q];
            float    c[4];
            for (int k = 0; k < 4; k++) { /* premultiplied, blended between levels */
                int   sh = k == 3 ? 24 : 16 - 8 * k;
                float x0 = (float)((p >> sh) & 255);
                c[k]     = w > 0.0f ? x0 + ((float)((pb[i] >> sh) & 255) - x0) * w : x0;
            }
            float al = c[3] * (1.0f / 255.0f), ia = al > 0.0f ? 1.0f / (al * 255.0f) : 0.0f;
            r[i] = c[0] * ia, g[i] = c[1] * ia, bo[i] = c[2] * ia, a[i] = al;
        }
    }
}
#endif

void fm3d_fs_fixed(const fm3d_dstate* st, fm3d_batch* b)
{
    const fm3d_tri* t    = b->tri;
    int             cols = b->cols;
    int             nq   = cols / 2;
    int             flat = (t->flags & FM3D_TRI_FLAT) != 0;
    int             act[FM3D_QCOLS / 2], na = 0;
    if (b->full) {
        for (int q = 0; q < nq; q++) act[q] = q;
        na = nq;
    } else {
        for (int q = 0; q < nq; q++) {
            int i = 2 * q;
            if (b->mask[i] | b->mask[i + 1] | b->mask[FM3D_QCOLS + i] | b->mask[FM3D_QCOLS + i + 1]) act[na++] = q;
        }
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
