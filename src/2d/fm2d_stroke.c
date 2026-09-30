/*
 * fatmap - stroker. Converts flattened polylines into closed outline
 * polygons that are filled with the nonzero rule.
 *
 * Open polyline:   left side forward + end cap + left side of the reversed
 *                  polyline + start cap = one closed contour.
 * Closed polyline: left side forward and left side reversed as two contours
 *                  (opposite winding -> ring).
 * Inner joins route through the pivot point (Skia style) so the overlapping
 * inner corner stays positively wound instead of needing exact clipping.
 */
#include "fm2d_internal.h"

typedef struct fm_stroker {
    fm_flat*               out;
    const fm_stroke_style* st;
    float                  hw;
    float                  tol;
    float                  round_step; /* radians per round segment */
} fm_stroker;

FM_INLINE fm_pt fm_p(float x, float y)
{
    fm_pt r = { x, y };
    return r;
}

static void fm_arc_pts(fm_stroker* s, fm_pt c, float a0, float sweep)
{
    int n = (int)ceilf(fabsf(sweep) / s->round_step);
    if (n < 1) return;
    float da = sweep / (float)n;
    for (int i = 1; i < n; i++) {
        float a = a0 + da * (float)i;
        fmf_add(s->out, fm_p(c.x + cosf(a) * s->hw, c.y + sinf(a) * s->hw));
    }
}

/* join at p between direction d0 and d1 on the left (n = (-d.y, d.x)) side */
static void fm_join(fm_stroker* s, fm_pt p, fm_pt d0, fm_pt d1, int smooth)
{
    float hw = s->hw;
    fm_pt n0 = fm_p(-d0.y, d0.x), n1 = fm_p(-d1.y, d1.x);
    float cross = d0.x * d1.y - d0.y * d1.x;
    float dot   = d0.x * d1.x + d0.y * d1.y;
    fm_pt a     = fm_p(p.x + n0.x * hw, p.y + n0.y * hw);
    fm_pt b     = fm_p(p.x + n1.x * hw, p.y + n1.y * hw);
    if (fabsf(cross) < 1e-6f && dot > 0.0f) {
        fmf_add(s->out, a);
        return;
    }
    if (cross > 0.0f) { /* inner side */
        fmf_add(s->out, a);
        fmf_add(s->out, p);
        fmf_add(s->out, b);
        return;
    }
    fmf_add(s->out, a);
    int join = smooth ? FM2D_JOIN_ROUND : s->st->join;
    if (join == FM2D_JOIN_ROUND) {
        float sweep = (fabsf(cross) < 1e-6f) ? -FM2D_PI : atan2f(n0.x * n1.y - n0.y * n1.x, n0.x * n1.x + n0.y * n1.y);
        fm_arc_pts(s, p, atan2f(n0.y, n0.x), sweep);
    } else if (join == FM2D_JOIN_MITER) {
        float ch = sqrtf(FM_MAX(0.0f, (1.0f + dot) * 0.5f)); /* cos(turn/2) */
        if (ch > 1e-6f && 1.0f / ch <= s->st->miter_limit) {
            float mx = n0.x + n1.x, my = n0.y + n1.y, ml = sqrtf(mx * mx + my * my);
            if (ml > 1e-9f) {
                float k = hw / ch / ml;
                fmf_add(s->out, fm_p(p.x + mx * k, p.y + my * k));
            }
        }
    }
    fmf_add(s->out, b);
}

static void fm_cap(fm_stroker* s, fm_pt p, fm_pt d)
{
    float hw = s->hw;
    fm_pt n  = fm_p(-d.y, d.x);
    if (s->st->cap == FM2D_CAP_SQUARE) {
        fmf_add(s->out, fm_p(p.x + n.x * hw + d.x * hw, p.y + n.y * hw + d.y * hw));
        fmf_add(s->out, fm_p(p.x - n.x * hw + d.x * hw, p.y - n.y * hw + d.y * hw));
    } else if (s->st->cap == FM2D_CAP_ROUND) {
        fm_arc_pts(s, p, atan2f(n.y, n.x), -FM2D_PI);
    }
}

FM_INLINE fm_pt fm_dir(fm_pt a, fm_pt b)
{
    float dx = b.x - a.x, dy = b.y - a.y, l = sqrtf(dx * dx + dy * dy);
    return l > 0.0f ? fm_p(dx / l, dy / l) : fm_p(1.0f, 0.0f);
}

/* one side of an open polyline p[0..n-1] (rev: walk the array backwards).
 * ts / te: optional exact unit tangents at the walk start / end. */
static void fm_side_open(fm_stroker* s, const fm_pt* p, int n, int rev, const fm_pt* ts, const fm_pt* te)
{
#define FM_AT(i) (rev ? p[n - 1 - (i)] : p[(i)])
    fm_pt d0 = fm_dir(FM_AT(0), FM_AT(1));
    fm_pt ds = ts ? *ts : d0;
    fmf_add(s->out, fm_p(FM_AT(0).x - ds.y * s->hw, FM_AT(0).y + ds.x * s->hw));
    for (int i = 1; i < n - 1; i++) {
        fm_pt d1 = fm_dir(FM_AT(i), FM_AT(i + 1));
        fm_join(s, FM_AT(i), d0, d1, 0);
        d0 = d1;
    }
    fm_pt de = te ? *te : d0;
    fmf_add(s->out, fm_p(FM_AT(n - 1).x - de.y * s->hw, FM_AT(n - 1).y + de.x * s->hw));
    fm_cap(s, FM_AT(n - 1), de);
#undef FM_AT
}

static void fm_side_closed(fm_stroker* s, const fm_pt* p, int n, int rev)
{
#define FM_AT(i) (rev ? p[(n - 1 - (i) + n) % n] : p[(i) % n])
    fm_pt d0 = fm_dir(FM_AT(n - 1), FM_AT(0));
    for (int i = 0; i < n; i++) {
        fm_pt d1 = fm_dir(FM_AT(i), FM_AT(i + 1));
        fm_join(s, FM_AT(i), d0, d1, 0);
        d0 = d1;
    }
#undef FM_AT
}

static void fm_dot(fm_stroker* s, fm_pt p)
{
    float hw = s->hw;
    if (s->st->cap == FM2D_CAP_ROUND) {
        if (!fmf_begin(s->out, 1)) return;
        fmf_add(s->out, fm_p(p.x + hw, p.y));
        fm_arc_pts(s, p, 0.0f, 2.0f * FM2D_PI);
        fmf_end(s->out);
    } else if (s->st->cap == FM2D_CAP_SQUARE) {
        if (!fmf_begin(s->out, 1)) return;
        fmf_add(s->out, fm_p(p.x - hw, p.y - hw));
        fmf_add(s->out, fm_p(p.x + hw, p.y - hw));
        fmf_add(s->out, fm_p(p.x + hw, p.y + hw));
        fmf_add(s->out, fm_p(p.x - hw, p.y + hw));
        fmf_end(s->out);
    }
}

static void fm_stroke_poly(fm_stroker* s, const fm_pt* p, int n, int closed, const fm_pt* t0, const fm_pt* t1)
{
    if (n == 1) {
        fm_dot(s, p[0]);
        return;
    }
    if (closed && n >= 3) {
        if (!fmf_begin(s->out, 1)) return;
        fm_side_closed(s, p, n, 0);
        fmf_end(s->out);
        if (!fmf_begin(s->out, 1)) return;
        fm_side_closed(s, p, n, 1);
        fmf_end(s->out);
        return;
    }
    if (!fmf_begin(s->out, 1)) return;
    fm_pt r0, r1;
    if (t0 && t1) {
        r0 = fm_p(-t1->x, -t1->y);
        r1 = fm_p(-t0->x, -t0->y);
    }
    fm_side_open(s, p, n, 0, t0, t1);
    fm_side_open(s, p, n, 1, (t0 && t1) ? &r0 : NULL, (t0 && t1) ? &r1 : NULL);
    fmf_end(s->out);
}

/* dedupe consecutive points into buf; returns count */
static int fm_dedupe(const fm_pt* p, int n, int closed, fm_pt* buf)
{
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m > 0) {
            float dx = p[i].x - buf[m - 1].x, dy = p[i].y - buf[m - 1].y;
            if (dx * dx + dy * dy < 1e-12f) continue;
        }
        buf[m++] = p[i];
    }
    if (closed && m > 1) {
        float dx = buf[m - 1].x - buf[0].x, dy = buf[m - 1].y - buf[0].y;
        if (dx * dx + dy * dy < 1e-12f) m--;
    }
    return m;
}

typedef struct fm_dasher {
    const float* d;
    int          nd;
    int          idx;
    float        rem;
    int          on;
} fm_dasher;

/* piece capacity: 2n + 8 points, first-piece capacity: n + 4 points */
static void fm_dash_poly(fm_stroker* s, const fm_pt* p, int n, int closed, fm_pt* piece, fm_pt* first, fm_dasher ds)
{
    const int pcap = 2 * n + 8, fcap = n + 4;
    int       segs = closed ? n : n - 1;
    int       np = 0, first_len = -1;
    int       started_on = ds.on;
    if (ds.on) piece[np++] = p[0];
    for (int i = 0; i < segs; i++) {
        fm_pt a = p[i], b = p[(i + 1) % n];
        float dx = b.x - a.x, dy = b.y - a.y, L = sqrtf(dx * dx + dy * dy);
        float t = 0.0f;
        int   guard = 0;
        while (L - t > 1e-9f && guard++ < 1000000) {
            float step = FM_MIN(ds.rem, L - t);
            t += step;
            ds.rem -= step;
            if (ds.rem > 1e-9f) break; /* dash continues past b */
            fm_pt q = fm_p(a.x + dx * (t / L), a.y + dy * (t / L));
            if (ds.on) {
                if (np < pcap) piece[np++] = q;
                if (started_on && closed && first_len < 0) {
                    first_len = FM_MIN(np, fcap);
                    memcpy(first, piece, (size_t)first_len * sizeof(fm_pt));
                } else {
                    fm_stroke_poly(s, piece, np, 0, NULL, NULL);
                }
                np = 0;
            }
            ds.idx = (ds.idx + 1) % ds.nd;
            ds.rem = ds.d[ds.idx];
            ds.on  = !ds.on;
            if (ds.on) piece[np++] = q;
        }
        if (ds.on && np < pcap && (piece[np - 1].x != b.x || piece[np - 1].y != b.y)) piece[np++] = b;
    }
    if (closed && first_len >= 0 && ds.on && np > 0) {
        /* the contour ends inside a dash that also started it: join them */
        for (int k = 1; k < first_len && np < pcap; k++) piece[np++] = first[k];
        fm_stroke_poly(s, piece, np, 0, NULL, NULL);
        return;
    }
    if (first_len >= 0) fm_stroke_poly(s, first, first_len, 0, NULL, NULL);
    if (ds.on && np > 0) fm_stroke_poly(s, piece, np, 0, NULL, NULL);
}

void fms_stroke(const fm_flat* in, const fm_stroke_style* st, float tol, fm_flat* out)
{
    fmf_clear(out);
    if (!(st->width > 0.0f)) return;
    fm_stroker s;
    s.out = out;
    s.st  = st;
    s.hw  = st->width * 0.5f;
    s.tol = tol;
    float r = s.hw > tol ? 1.0f - tol / s.hw : 0.0f;
    s.round_step = r > 0.0f ? 2.0f * acosf(r) : FM2D_PI * 0.5f;
    s.round_step = FM_CLAMP(s.round_step, 0.02f, FM2D_PI * 0.5f);

    /* dash pattern */
    float total = 0.0f;
    for (int i = 0; i < st->ndash; i++) total += st->dash[i];
    int dashed = st->ndash > 0 && total > 0.0f;
    fm_dasher ds0;
    memset(&ds0, 0, sizeof(ds0));
    if (dashed) {
        ds0.d     = st->dash;
        ds0.nd    = st->ndash;
        float ph  = fmodf(st->dash_offset, total);
        if (ph < 0.0f) ph += total;
        ds0.idx = 0;
        ds0.rem = st->dash[0];
        ds0.on  = 1;
        int guard = 0;
        while (ph > 0.0f && guard++ < 100000) {
            if (ph >= ds0.rem) {
                ph -= ds0.rem;
                ds0.idx = (ds0.idx + 1) % ds0.nd;
                ds0.rem = st->dash[ds0.idx];
                ds0.on  = !ds0.on;
            } else {
                ds0.rem -= ph;
                ph = 0.0f;
            }
        }
    }

    int maxn = 0;
    for (int c = 0; c < in->nc; c++) maxn = FM_MAX(maxn, in->cs[c].count);
    /* buf: n, piece: 2n + 8, first: n + 4 */
    fm_pt* buf = (fm_pt*)malloc(((size_t)maxn * 4 + 16) * sizeof(fm_pt));
    if (!buf) return;
    fm_pt* piece = buf + maxn + 2;
    fm_pt* first = piece + 2 * maxn + 8;

    for (int c = 0; c < in->nc; c++) {
        const fm_contour* ct = &in->cs[c];
        if (ct->count < 2) continue; /* lone moveTo is not stroked */
        int n = fm_dedupe(in->pts + ct->start, ct->count, ct->closed, buf);
        if (n < 1) continue;
        if (dashed && n >= 2)
            fm_dash_poly(&s, buf, n, ct->closed, piece, first, ds0);
        else
            fm_stroke_poly(&s, buf, n, ct->closed, ct->has_tangents ? &ct->t0 : NULL,
                           ct->has_tangents ? &ct->t1 : NULL);
    }
    free(buf);
}
