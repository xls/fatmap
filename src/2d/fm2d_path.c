/* fatmap - 2D paths: storage, canvas builders (arcs, arcTo, roundRect), flattening */
#include "fm2d_internal.h"

/* ---- storage ------------------------------------------------------------------ */

void fmp_init(fm2d_path* p) { memset(p, 0, sizeof(*p)); }

void fmp_free(fm2d_path* p)
{
    free(p->verbs);
    free(p->pts);
    fmp_init(p);
}

void fmp_clear(fm2d_path* p)
{
    p->nv        = 0;
    p->np        = 0;
    p->has_sub   = 0;
    p->need_move = 0;
}

static int fmp_grow(fm2d_path* p, int nverbs, int npts)
{
    if (p->nv + nverbs > p->cv) {
        int      nc = FM_MAX(p->cv * 2, p->nv + nverbs + 16);
        uint8_t* nv = (uint8_t*)realloc(p->verbs, (size_t)nc);
        if (!nv) return 0;
        p->verbs = nv;
        p->cv    = nc;
    }
    if (p->np + npts > p->cp) {
        int    nc = FM_MAX(p->cp * 2, p->np + npts + 32);
        fm_pt* np = (fm_pt*)realloc(p->pts, (size_t)nc * sizeof(fm_pt));
        if (!np) return 0;
        p->pts = np;
        p->cp  = nc;
    }
    return 1;
}

int fmp_copy(fm2d_path* dst, const fm2d_path* src)
{
    fmp_clear(dst);
    if (!fmp_grow(dst, src->nv, src->np)) return 0;
    memcpy(dst->verbs, src->verbs, (size_t)src->nv);
    memcpy(dst->pts, src->pts, (size_t)src->np * sizeof(fm_pt));
    dst->nv        = src->nv;
    dst->np        = src->np;
    dst->has_sub   = src->has_sub;
    dst->need_move = src->need_move;
    dst->start     = src->start;
    return 1;
}

void fmp_move_to(fm2d_path* p, fm_pt a)
{
    if (p->nv > 0 && p->verbs[p->nv - 1] == FMV_MOVE) {
        p->pts[p->np - 1] = a; /* collapse consecutive moves */
    } else {
        if (!fmp_grow(p, 1, 1)) return;
        p->verbs[p->nv++] = FMV_MOVE;
        p->pts[p->np++]   = a;
    }
    p->start     = a;
    p->has_sub   = 1;
    p->need_move = 0;
}

static void fmp_ensure(fm2d_path* p, fm_pt a)
{
    if (!p->has_sub)
        fmp_move_to(p, a);
    else if (p->need_move)
        fmp_move_to(p, p->start);
}

void fmp_line_to(fm2d_path* p, fm_pt a)
{
    if (!p->has_sub) {
        fmp_move_to(p, a);
        return;
    }
    fmp_ensure(p, a);
    if (!fmp_grow(p, 1, 1)) return;
    p->verbs[p->nv++] = FMV_LINE;
    p->pts[p->np++]   = a;
}

void fmp_cubic_to(fm2d_path* p, fm_pt c1, fm_pt c2, fm_pt a)
{
    fmp_ensure(p, c1);
    if (!fmp_grow(p, 1, 3)) return;
    p->verbs[p->nv++] = FMV_CUBIC;
    p->pts[p->np++]   = c1;
    p->pts[p->np++]   = c2;
    p->pts[p->np++]   = a;
}

void fmp_close(fm2d_path* p)
{
    if (!p->has_sub || p->need_move) return;
    if (!fmp_grow(p, 1, 0)) return;
    p->verbs[p->nv++] = FMV_CLOSE;
    p->need_move      = 1;
}

int fmp_current(const fm2d_path* p, fm_pt* out)
{
    if (!p->has_sub) return 0;
    *out = p->need_move ? p->start : p->pts[p->np - 1];
    return 1;
}

/* ---- builders -------------------------------------------------------------------- */

void fmb_move_to(fm2d_path* p, const fm_affine* m, float x, float y)
{
    if (!isfinite(x) || !isfinite(y)) return;
    fmp_move_to(p, fm_xform(m, x, y));
}

void fmb_line_to(fm2d_path* p, const fm_affine* m, float x, float y)
{
    if (!isfinite(x) || !isfinite(y)) return;
    fmp_line_to(p, fm_xform(m, x, y));
}

void fmb_bezier_to(fm2d_path* p, const fm_affine* m, float c1x, float c1y, float c2x, float c2y, float x, float y)
{
    if (!isfinite(c1x + c1y + c2x + c2y + x + y)) return;
    fmp_cubic_to(p, fm_xform(m, c1x, c1y), fm_xform(m, c2x, c2y), fm_xform(m, x, y));
}

static int fm_user_current(const fm2d_path* p, const fm_affine* m, float* x, float* y)
{
    fm_pt c;
    if (!fmp_current(p, &c)) return 0;
    if (m) {
        fm_affine inv;
        if (!fm_affine_invert(m, &inv)) return 0;
        c = fm_xform(&inv, c.x, c.y);
    }
    *x = c.x;
    *y = c.y;
    return 1;
}

void fmb_quad_to(fm2d_path* p, const fm_affine* m, float cx, float cy, float x, float y)
{
    if (!isfinite(cx + cy + x + y)) return;
    float x0, y0;
    if (!fm_user_current(p, m, &x0, &y0)) {
        fmb_move_to(p, m, cx, cy);
        x0 = cx;
        y0 = cy;
    }
    /* exact quad -> cubic elevation */
    float c1x = x0 + (cx - x0) * (2.0f / 3.0f), c1y = y0 + (cy - y0) * (2.0f / 3.0f);
    float c2x = x + (cx - x) * (2.0f / 3.0f), c2y = y + (cy - y) * (2.0f / 3.0f);
    fmp_cubic_to(p, fm_xform(m, c1x, c1y), fm_xform(m, c2x, c2y), fm_xform(m, x, y));
}

void fmb_ellipse(fm2d_path* p, const fm_affine* m, float cx, float cy, float rx, float ry, float rot, float a0,
                 float a1, int ccw)
{
    if (!isfinite(cx + cy + rx + ry + rot + a0 + a1) || rx < 0.0f || ry < 0.0f) return;
    const float tau = 2.0f * FM2D_PI;
    float       sweep;
    if (!ccw && a1 - a0 >= tau)
        sweep = tau;
    else if (ccw && a0 - a1 >= tau)
        sweep = -tau;
    else {
        sweep = fmodf(a1 - a0, tau);
        if (!ccw && sweep < 0.0f) sweep += tau;
        if (ccw && sweep > 0.0f) sweep -= tau;
    }
    float cr = cosf(rot), sr = sinf(rot);
#define FM_EPT(u, v) fm_xform(m, cx + rx * (u) * cr - ry * (v) * sr, cy + rx * (u) * sr + ry * (v) * cr)
    fm_pt start = FM_EPT(cosf(a0), sinf(a0));
    if (p->has_sub)
        fmp_line_to(p, start);
    else
        fmp_move_to(p, start);
    if (sweep == 0.0f) return;
    int   segs = (int)ceilf(fabsf(sweep) / (FM2D_PI * 0.5f) - 1e-4f);
    float th   = sweep / (float)FM_MAX(segs, 1);
    float k    = 4.0f / 3.0f * tanf(th * 0.25f);
    float a    = a0;
    for (int i = 0; i < FM_MAX(segs, 1); i++) {
        float c0 = cosf(a), s0 = sinf(a);
        float b  = (i == segs - 1) ? a0 + sweep : a + th;
        float c1 = cosf(b), s1 = sinf(b);
        fmp_cubic_to(p, FM_EPT(c0 - k * s0, s0 + k * c0), FM_EPT(c1 + k * s1, s1 - k * c1), FM_EPT(c1, s1));
        a = b;
    }
#undef FM_EPT
}

void fmb_arc_to(fm2d_path* p, const fm_affine* m, float x1, float y1, float x2, float y2, float r)
{
    if (!isfinite(x1 + y1 + x2 + y2 + r) || r < 0.0f) return;
    float x0, y0;
    if (!fm_user_current(p, m, &x0, &y0)) {
        fmb_move_to(p, m, x1, y1);
        return;
    }
    float ux = x0 - x1, uy = y0 - y1, vx = x2 - x1, vy = y2 - y1;
    float lu = sqrtf(ux * ux + uy * uy), lv = sqrtf(vx * vx + vy * vy);
    float cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
    if (r == 0.0f || lu < 1e-6f || lv < 1e-6f || fabsf(cross) < 1e-6f * lu * lv) {
        fmb_line_to(p, m, x1, y1);
        return;
    }
    ux /= lu;
    uy /= lu;
    vx /= lv;
    vy /= lv;
    float cosang = FM_CLAMP(ux * vx + uy * vy, -1.0f, 1.0f);
    float ang    = acosf(cosang);
    float tlen   = r / tanf(ang * 0.5f);
    float t1x = x1 + ux * tlen, t1y = y1 + uy * tlen;
    float t2x = x1 + vx * tlen, t2y = y1 + vy * tlen;
    float bx = ux + vx, by = uy + vy, bl = sqrtf(bx * bx + by * by);
    float d  = r / sinf(ang * 0.5f);
    float ccx = x1 + bx / bl * d, ccy = y1 + by / bl * d;
    fmb_line_to(p, m, t1x, t1y);
    fmb_ellipse(p, m, ccx, ccy, r, r, 0.0f, atan2f(t1y - ccy, t1x - ccx), atan2f(t2y - ccy, t2x - ccx), cross < 0.0f);
}

void fmb_rect(fm2d_path* p, const fm_affine* m, float x, float y, float w, float h)
{
    if (!isfinite(x + y + w + h)) return;
    fmb_move_to(p, m, x, y);
    fmb_line_to(p, m, x + w, y);
    fmb_line_to(p, m, x + w, y + h);
    fmb_line_to(p, m, x, y + h);
    fmp_close(p);
    fmb_move_to(p, m, x, y);
}

void fmb_round_rect(fm2d_path* p, const fm_affine* m, float x, float y, float w, float h, const float* radii,
                    int count)
{
    if (!isfinite(x + y + w + h)) return;
    float r[4] = { 0, 0, 0, 0 }; /* tl tr br bl */
    if (radii && count > 0) {
        for (int i = 0; i < count && i < 4; i++)
            if (!(radii[i] >= 0.0f) || !isfinite(radii[i])) return;
        switch (count) {
        case 1: r[0] = r[1] = r[2] = r[3] = radii[0]; break;
        case 2:
            r[0] = r[2] = radii[0];
            r[1] = r[3] = radii[1];
            break;
        case 3:
            r[0] = radii[0];
            r[1] = r[3] = radii[1];
            r[2] = radii[2];
            break;
        default:
            r[0] = radii[0];
            r[1] = radii[1];
            r[2] = radii[2];
            r[3] = radii[3];
            break;
        }
    }
    if (w < 0.0f) {
        x += w;
        w = -w;
        float t;
        t = r[0]; r[0] = r[1]; r[1] = t;
        t = r[3]; r[3] = r[2]; r[2] = t;
    }
    if (h < 0.0f) {
        y += h;
        h = -h;
        float t;
        t = r[0]; r[0] = r[3]; r[3] = t;
        t = r[1]; r[1] = r[2]; r[2] = t;
    }
    float sc = 1.0f;
    if (r[0] + r[1] > 0.0f) sc = FM_MIN(sc, w / (r[0] + r[1]));
    if (r[1] + r[2] > 0.0f) sc = FM_MIN(sc, h / (r[1] + r[2]));
    if (r[2] + r[3] > 0.0f) sc = FM_MIN(sc, w / (r[2] + r[3]));
    if (r[3] + r[0] > 0.0f) sc = FM_MIN(sc, h / (r[3] + r[0]));
    for (int i = 0; i < 4; i++) r[i] *= sc;
    const float hp = FM2D_PI * 0.5f;
    fmb_move_to(p, m, x + r[0], y);
    fmb_line_to(p, m, x + w - r[1], y);
    if (r[1] > 0) fmb_ellipse(p, m, x + w - r[1], y + r[1], r[1], r[1], 0, -hp, 0, 0);
    fmb_line_to(p, m, x + w, y + h - r[2]);
    if (r[2] > 0) fmb_ellipse(p, m, x + w - r[2], y + h - r[2], r[2], r[2], 0, 0, hp, 0);
    fmb_line_to(p, m, x + r[3], y + h);
    if (r[3] > 0) fmb_ellipse(p, m, x + r[3], y + h - r[3], r[3], r[3], 0, hp, FM2D_PI, 0);
    fmb_line_to(p, m, x, y + r[0]);
    if (r[0] > 0) fmb_ellipse(p, m, x + r[0], y + r[0], r[0], r[0], 0, FM2D_PI, FM2D_PI + hp, 0);
    fmp_close(p);
    fmb_move_to(p, m, x, y);
}

/* ---- flattening -------------------------------------------------------------------- */

void fmf_free(fm_flat* f)
{
    free(f->pts);
    free(f->cs);
    memset(f, 0, sizeof(*f));
}

void fmf_clear(fm_flat* f)
{
    f->np = 0;
    f->nc = 0;
}

int fmf_begin(fm_flat* f, int closed)
{
    if (f->nc == f->cc) {
        int         nc = f->cc ? f->cc * 2 : 16;
        fm_contour* ncs = (fm_contour*)realloc(f->cs, (size_t)nc * sizeof(fm_contour));
        if (!ncs) return 0;
        f->cs = ncs;
        f->cc = nc;
    }
    fm_contour* c   = &f->cs[f->nc++];
    c->start        = f->np;
    c->count        = 0;
    c->closed       = closed;
    c->has_tangents = 0;
    c->t0.x = c->t0.y = c->t1.x = c->t1.y = 0.0f;
    return 1;
}

void fmf_add(fm_flat* f, fm_pt p)
{
    if (f->np == f->cp) {
        int    nc = f->cp ? f->cp * 2 : 256;
        fm_pt* np = (fm_pt*)realloc(f->pts, (size_t)nc * sizeof(fm_pt));
        if (!np) return;
        f->pts = np;
        f->cp  = nc;
    }
    f->pts[f->np++] = p;
    f->cs[f->nc - 1].count++;
}

void fmf_end(fm_flat* f)
{
    if (f->nc > 0 && f->cs[f->nc - 1].count < 1) {
        f->np = f->cs[f->nc - 1].start;
        f->nc--;
    }
}

static void fmf_cubic(fm_flat* f, fm_pt p0, fm_pt p1, fm_pt p2, fm_pt p3, float tol)
{
    float ddx = fabsf(p0.x - 2 * p1.x + p2.x), ddy = fabsf(p0.y - 2 * p1.y + p2.y);
    float eex = fabsf(p1.x - 2 * p2.x + p3.x), eey = fabsf(p1.y - 2 * p2.y + p3.y);
    float dd  = sqrtf(FM_MAX(ddx * ddx + ddy * ddy, eex * eex + eey * eey));
    int   n   = (int)ceilf(sqrtf(0.75f * dd / tol));
    n         = FM_CLAMP(n, 1, 1000);
    float dt  = 1.0f / (float)n;
    for (int i = 1; i < n; i++) {
        float t = (float)i * dt, u = 1.0f - t;
        float a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
        fm_pt q = { a * p0.x + b * p1.x + c * p2.x + d * p3.x, a * p0.y + b * p1.y + c * p2.y + d * p3.y };
        fmf_add(f, q);
    }
    fmf_add(f, p3);
}

/* record segment tangents: first one becomes the contour start tangent,
 * the latest one the end tangent */
static void fmf_tangent(fm_flat* f, fm_pt a, fm_pt b)
{
    float dx = b.x - a.x, dy = b.y - a.y, l = sqrtf(dx * dx + dy * dy);
    if (!(l > 1e-9f) || f->nc == 0) return;
    fm_contour* c = &f->cs[f->nc - 1];
    fm_pt       t = { dx / l, dy / l };
    if (!c->has_tangents) {
        c->t0           = t;
        c->has_tangents = 1;
    }
    c->t1 = t;
}

static void fmf_seg_tangents(fm_flat* f, fm_pt p0, fm_pt c1, fm_pt c2, fm_pt p3)
{
    /* start: first control point distinct from p0; end: last distinct from p3 */
    fm_pt s = c1, e = c2;
    if (s.x == p0.x && s.y == p0.y) s = (c2.x != p0.x || c2.y != p0.y) ? c2 : p3;
    if (e.x == p3.x && e.y == p3.y) e = (c1.x != p3.x || c1.y != p3.y) ? c1 : p0;
    fmf_tangent(f, p0, s);
    fmf_tangent(f, e, p3);
}

void fmf_flatten(fm_flat* out, const fm2d_path* p, const fm_affine* xf, float tol)
{
    fmf_clear(out);
    if (tol <= 0.0f) tol = 0.2f;
    int   pi = 0, open = 0;
    fm_pt last = { 0, 0 };
    for (int v = 0; v < p->nv; v++) {
        switch (p->verbs[v]) {
        case FMV_MOVE:
            if (open) fmf_end(out);
            last = fm_xform(xf, p->pts[pi].x, p->pts[pi].y);
            pi++;
            if (!fmf_begin(out, 0)) return;
            fmf_add(out, last);
            open = 1;
            break;
        case FMV_LINE:
        {
            fm_pt q = fm_xform(xf, p->pts[pi].x, p->pts[pi].y);
            pi++;
            if (open) {
                fmf_tangent(out, last, q);
                fmf_add(out, q);
            }
            last = q;
            break;
        }
        case FMV_CUBIC: {
            fm_pt c1 = fm_xform(xf, p->pts[pi].x, p->pts[pi].y);
            fm_pt c2 = fm_xform(xf, p->pts[pi + 1].x, p->pts[pi + 1].y);
            fm_pt e  = fm_xform(xf, p->pts[pi + 2].x, p->pts[pi + 2].y);
            pi += 3;
            if (open) {
                fmf_seg_tangents(out, last, c1, c2, e);
                fmf_cubic(out, last, c1, c2, e, tol);
            }
            last = e;
            break;
        }
        case FMV_CLOSE:
            if (open) {
                out->cs[out->nc - 1].closed = 1;
                fmf_end(out);
                open = 0;
            }
            break;
        }
    }
    if (open) fmf_end(out);
}

int fmf_winding(const fm_flat* f, float x, float y)
{
    int w = 0;
    for (int c = 0; c < f->nc; c++) {
        const fm_pt* p = f->pts + f->cs[c].start;
        int          n = f->cs[c].count;
        if (n < 2) continue;
        for (int i = 0; i < n; i++) {
            fm_pt a = p[i], b = p[(i + 1) % n];
            if (a.y <= y) {
                if (b.y > y && (b.x - a.x) * (y - a.y) - (x - a.x) * (b.y - a.y) > 0) w++;
            } else {
                if (b.y <= y && (b.x - a.x) * (y - a.y) - (x - a.x) * (b.y - a.y) < 0) w--;
            }
        }
    }
    return w;
}
