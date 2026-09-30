/*
 * fatmap - core coverage rasterizer.
 *
 * FM_AA_ANALYTIC: signed area accumulation (font-rs / libart family). Each
 * edge deposits exact area deltas into a float accumulation band; a prefix
 * sum per row yields coverage. A per-row "touched" bitmap (one bit per 32
 * cells) lets untouched runs skip the prefix sum entirely: their coverage is
 * the running sum, so interiors become full-coverage spans (cov == NULL) and
 * empty gaps are skipped.
 *
 * FM_AA_NONE: scanline point sampling at pixel centers with an active edge
 * list, top-left fill convention (pixel centers on left/top edges are in).
 */
#include "fm_raster_internal.h"
#include "fm_kernels_common.h"
#include <limits.h>
#include <math.h>

#define FM_BAND  FM_RASTER_BAND
#define FM_CHUNK 32 /* cells per touched bit */

struct fm_rasterizer {
    fm_redge* e;
    int       n, cap;
    int       cx0, cy0, cx1, cy1;
    float     bx0, by0, bx1, by1;
    int       has_right;
    int       sorted;

    float*    acc;
    size_t    acc_cap;
    uint64_t* touched;
    size_t    touched_cap;
    uint8_t*  mask;
    size_t    mask_cap;
    int*      active;
    size_t    active_cap;
    float*    xs;
    size_t    xs_cap;
    float*    ws;
    size_t    ws_cap;
};

fm_rasterizer* fm_rasterizer_create(void)
{
    fm__init();
    fm_rasterizer* r = (fm_rasterizer*)calloc(1, sizeof(fm_rasterizer));
    if (r) fm_rasterizer_reset(r, 0, 0, 1 << 15, 1 << 15);
    return r;
}

void fm_rasterizer_destroy(fm_rasterizer* r)
{
    if (!r) return;
    free(r->e);
    fm_aligned_free(r->acc);
    free(r->touched);
    fm_aligned_free(r->mask);
    free(r->active);
    free(r->xs);
    free(r->ws);
    free(r);
}

void fm_rasterizer_reset(fm_rasterizer* r, int x0, int y0, int x1, int y1)
{
    r->n         = 0;
    r->cx0       = x0;
    r->cy0       = y0;
    r->cx1       = FM_MAX(x0, x1);
    r->cy1       = FM_MAX(y0, y1);
    r->bx0       = 1e30f;
    r->by0       = 1e30f;
    r->bx1       = -1e30f;
    r->by1       = -1e30f;
    r->has_right = 0;
    r->sorted    = 1;
}

int fm_rasterizer_edge_count(const fm_rasterizer* r) { return r->n; }

static void fm_push_edge(fm_rasterizer* r, float x0, float y0, float x1, float y1, float dir)
{
    if (!(y1 > y0)) return;
    if (r->n == r->cap) {
        int       nc = r->cap ? r->cap * 2 : 256;
        fm_redge* ne = (fm_redge*)realloc(r->e, (size_t)nc * sizeof(fm_redge));
        if (!ne) return;
        r->e   = ne;
        r->cap = nc;
    }
    fm_redge* e = &r->e[r->n++];
    r->sorted   = 0;
    e->x0       = x0;
    e->y0       = y0;
    e->x1       = x1;
    e->y1       = y1;
    e->dxdy     = (x1 - x0) / (y1 - y0);
    e->dir      = dir;
    r->bx0      = FM_MIN(r->bx0, FM_MIN(x0, x1));
    r->bx1      = FM_MAX(r->bx1, FM_MAX(x0, x1));
    r->by0      = FM_MIN(r->by0, y0);
    r->by1      = FM_MAX(r->by1, y1);
}

/* x clipping: parts left of the clip collapse onto x = cx0 (they still carry
 * winding), parts right of the clip are dropped (they cannot affect pixels
 * to their left). */
static void fm_add_clip_x(fm_rasterizer* r, float x0, float y0, float x1, float y1, float dir, int depth)
{
    float fx0 = (float)r->cx0, fx1 = (float)r->cx1;
    if (x0 >= fx1 && x1 >= fx1) {
        r->has_right = 1;
        return;
    }
    if (x0 <= fx0 && x1 <= fx0) {
        fm_push_edge(r, fx0, y0, fx0, y1, dir);
        return;
    }
    if (depth < 4) {
        if ((x0 < fx0) != (x1 < fx0)) {
            float ym = y0 + (fx0 - x0) * (y1 - y0) / (x1 - x0);
            ym       = FM_CLAMP(ym, y0, y1);
            fm_add_clip_x(r, x0, y0, fx0, ym, dir, depth + 1);
            fm_add_clip_x(r, fx0, ym, x1, y1, dir, depth + 1);
            return;
        }
        if ((x0 > fx1) != (x1 > fx1)) {
            float ym = y0 + (fx1 - x0) * (y1 - y0) / (x1 - x0);
            ym       = FM_CLAMP(ym, y0, y1);
            fm_add_clip_x(r, x0, y0, fx1, ym, dir, depth + 1);
            fm_add_clip_x(r, fx1, ym, x1, y1, dir, depth + 1);
            return;
        }
    }
    fm_push_edge(r, FM_CLAMP(x0, fx0, fx1), y0, FM_CLAMP(x1, fx0, fx1), y1, dir);
}

void fm_rasterizer_add_line(fm_rasterizer* r, float x0, float y0, float x1, float y1)
{
    if (!(isfinite(x0) && isfinite(y0) && isfinite(x1) && isfinite(y1))) return;
    if (y0 == y1) return;
    float dir = 1.0f;
    if (y0 > y1) {
        float t;
        t   = x0; x0 = x1; x1 = t;
        t   = y0; y0 = y1; y1 = t;
        dir = -1.0f;
    }
    float fy0 = (float)r->cy0, fy1 = (float)r->cy1;
    if (y1 <= fy0 || y0 >= fy1) return;
    if (y0 < fy0) {
        x0 = x0 + (x1 - x0) * (fy0 - y0) / (y1 - y0);
        y0 = fy0;
    }
    if (y1 > fy1) {
        x1 = x0 + (x1 - x0) * (fy1 - y0) / (y1 - y0);
        y1 = fy1;
    }
    fm_add_clip_x(r, x0, y0, x1, y1, dir, 0);
}

void fm_rasterizer_add_polygon(fm_rasterizer* r, const float* xy, int count)
{
    if (count < 2) return;
    for (int i = 0; i < count; i++) {
        int j = (i + 1 == count) ? 0 : i + 1;
        fm_rasterizer_add_line(r, xy[i * 2], xy[i * 2 + 1], xy[j * 2], xy[j * 2 + 1]);
    }
}

int fm_rasterizer_bounds(const fm_rasterizer* r, int out[4])
{
    if (r->n == 0) return 0;
    int x0 = FM_MAX(r->cx0, (int)floorf(r->bx0));
    int x1 = r->has_right ? r->cx1 : FM_MIN(r->cx1, (int)ceilf(r->bx1));
    int y0 = FM_MAX(r->cy0, (int)floorf(r->by0));
    int y1 = FM_MIN(r->cy1, (int)ceilf(r->by1));
    if (x1 <= x0 || y1 <= y0) return 0;
    out[0] = x0;
    out[1] = y0;
    out[2] = x1;
    out[3] = y1;
    return 1;
}

static int fm_edge_cmp(const void* a, const void* b)
{
    float ya = ((const fm_redge*)a)->y0, yb = ((const fm_redge*)b)->y0;
    return (ya > yb) - (ya < yb);
}

static int fm_reserve(void** p, size_t* cap, size_t need, size_t elem, int aligned, int zero)
{
    if (*cap >= need) return 1;
    size_t nc = need + need / 2;
    void*  np;
    if (aligned) {
        fm_aligned_free(*p);
        np = fm_aligned_alloc(nc * elem, 64);
    } else {
        free(*p);
        np = malloc(nc * elem);
    }
    *p = np;
    if (!np) {
        *cap = 0;
        return 0;
    }
    if (zero) memset(np, 0, nc * elem);
    *cap = nc;
    return 1;
}

FM_INLINE void fm_mark(uint64_t* tb, int lo, int hi, int W)
{
    if (lo >= W) return;
    if (hi >= W) hi = W - 1;
    int c0 = lo / FM_CHUNK, c1 = hi / FM_CHUNK;
    for (int c = c0; c <= c1; c++) tb[c >> 6] |= (uint64_t)1 << (c & 63);
}

/* band_y / band_end: the aligned band (drives the x stepping so results do
 * not depend on how rows are split); rows outside [row_lo, row_hi) are
 * stepped over but not written. */
static void fm_draw_edge(float* acc, int stride, uint64_t* touched, int wpr, float ox, int band_y, int band_end,
                         int row_lo, int row_hi, const fm_redge* e, int W)
{
    float ys = e->y0 > (float)band_y ? e->y0 : (float)band_y;
    float ye = e->y1 < (float)band_end ? e->y1 : (float)band_end;
    if (ys >= ye) return;
    float dxdy = e->dxdy;
    float x    = e->x0 - ox + (ys - e->y0) * dxdy;
    int   yi   = (int)floorf(ys);
    int   yend = (int)ceilf(ye);
    float fW   = (float)W;
    for (; yi < yend; yi++) {
        float y_top = FM_MAX((float)yi, ys), y_bot = FM_MIN((float)(yi + 1), ye);
        float dy    = y_bot - y_top;
        float xnext = x + dxdy * dy;
        if (yi < row_lo || yi >= row_hi) {
            x = xnext;
            continue;
        }
        float d = dy * e->dir;
        float xa    = FM_MIN(x, xnext), xb = FM_MAX(x, xnext);
        xa          = FM_CLAMP(xa, 0.0f, fW);
        xb          = FM_CLAMP(xb, 0.0f, fW);
        float*    a  = acc + (size_t)(yi - band_y) * (size_t)stride;
        uint64_t* tb = touched + (size_t)(yi - band_y) * (size_t)wpr;
        float     x0floor = floorf(xa);
        int       x0i     = (int)x0floor;
        float     x1ceil  = ceilf(xb);
        int       x1i     = (int)x1ceil;
        if (x1i <= x0i + 1) {
            float xmf = 0.5f * (xa + xb) - x0floor;
            a[x0i] += d - d * xmf;
            a[x0i + 1] += d * xmf;
            fm_mark(tb, x0i, x0i + 1, W);
        } else {
            float s   = 1.0f / (xb - xa);
            float x0f = xa - x0floor;
            float a0  = 0.5f * s * (1.0f - x0f) * (1.0f - x0f);
            float x1f = xb - x1ceil + 1.0f;
            float am  = 0.5f * s * x1f * x1f;
            a[x0i] += d * a0;
            if (x1i == x0i + 2) {
                a[x0i + 1] += d * (1.0f - a0 - am);
            } else {
                float a1 = s * (1.5f - x0f);
                a[x0i + 1] += d * (a1 - a0);
                float ds = d * s;
                int   run = x1i - 1 - (x0i + 2);
                if (run >= 8)
                    fm_k->acc_add(a + x0i + 2, ds, run);
                else
                    for (int xi = x0i + 2; xi < x1i - 1; xi++) a[xi] += ds;
                float a2 = a1 + (float)(x1i - x0i - 3) * s;
                a[x1i - 1] += d * (1.0f - a2 - am);
            }
            a[x1i] += d * am;
            fm_mark(tb, x0i, x1i, W);
        }
        x = xnext;
    }
}

FM_INLINE void fm_emit_trim(fm_span_fn fn, void* user, int y, int x0, uint8_t* m, int a, int b, uint64_t* px)
{
    while (a < b && m[a] == 0) a++;
    while (b > a && m[b - 1] == 0) b--;
    if (a < b) {
        fn(user, y, x0 + a, b - a, m + a);
        *px += (uint64_t)(b - a);
    }
}

static uint64_t fm_render_analytic(fm_rasterizer* r, const fm_redge* edges, int ne, const int bb[4], int y0, int y1,
                                   int evenodd, fm_span_fn fn, void* user)
{
    const int ix0 = bb[0], ix1 = bb[2];
    const int iy0 = FM_MAX(bb[1], y0), iy1 = FM_MIN(bb[3], y1);
    if (iy1 <= iy0) return 0;
    const int W       = ix1 - ix0;
    const int stride  = (W + 2 + 15) & ~15;
    const int nchunks = (W + FM_CHUNK - 1) / FM_CHUNK;
    const int wpr     = (nchunks + 63) / 64;
    uint64_t  px      = 0;

    if (!fm_reserve((void**)&r->acc, &r->acc_cap, (size_t)stride * FM_BAND, sizeof(float), 1, 1)) return 0;
    if (!fm_reserve((void**)&r->touched, &r->touched_cap, (size_t)wpr * FM_BAND, sizeof(uint64_t), 0, 1)) return 0;
    if (!fm_reserve((void**)&r->mask, &r->mask_cap, (size_t)stride, 1, 1, 0)) return 0;
    if (!fm_reserve((void**)&r->active, &r->active_cap, (size_t)ne, sizeof(int), 0, 0)) return 0;

    float*    acc     = r->acc;
    uint64_t* touched = r->touched;
    uint8_t*  mask    = r->mask;
    int*      active  = r->active;
    int       na = 0, next = 0;
    float     ox = (float)ix0;

    /* bands aligned to absolute multiples of FM_BAND */
    int by0 = iy0 - (((iy0 % FM_BAND) + FM_BAND) % FM_BAND);
    for (int by = by0; by < iy1; by += FM_BAND) {
        int bend = by + FM_BAND;
        int rlo = FM_MAX(by, iy0), rhi = FM_MIN(bend, iy1);
        while (next < ne && edges[next].y0 < (float)bend) active[na++] = next++;
        int keep = 0;
        for (int k = 0; k < na; k++) {
            const fm_redge* e = &edges[active[k]];
            fm_draw_edge(acc, stride, touched, wpr, ox, by, bend, rlo, rhi, e, W);
            if (e->y1 > (float)bend) active[keep++] = active[k];
        }
        na = keep;

        for (int y = rlo; y < rhi; y++) {
            float*    row   = acc + (size_t)(y - by) * (size_t)stride;
            uint64_t* tb    = touched + (size_t)(y - by) * (size_t)wpr;
            float     carry = 0.0f;
            int       c     = 0;
            while (c < nchunks) {
                int t  = (int)((tb[c >> 6] >> (c & 63)) & 1);
                int c1 = c + 1;
                while (c1 < nchunks && (int)((tb[c1 >> 6] >> (c1 & 63)) & 1) == t) c1++;
                int xa = c * FM_CHUNK, xb = FM_MIN(c1 * FM_CHUNK, W);
                if (t) {
                    carry = fm_k->accumulate(row + xa, mask + xa, xb - xa, carry, evenodd);
                    fm_emit_trim(fn, user, y, ix0, mask, xa, xb, &px);
                } else {
                    uint8_t cv = fm_cov_u8(carry, evenodd);
                    if (cv == 255) {
                        fn(user, y, ix0 + xa, xb - xa, NULL);
                        px += (uint64_t)(xb - xa);
                    } else if (cv) {
                        memset(mask + xa, cv, (size_t)(xb - xa));
                        fn(user, y, ix0 + xa, xb - xa, mask + xa);
                        px += (uint64_t)(xb - xa);
                    }
                }
                c = c1;
            }
            row[W]     = 0.0f;
            row[W + 1] = 0.0f;
            memset(tb, 0, (size_t)wpr * sizeof(uint64_t));
        }
    }
    return px;
}

static uint64_t fm_render_aliased(fm_rasterizer* r, const fm_redge* edges, int ne, const int bb[4], int y0, int y1,
                                  int evenodd, fm_span_fn fn, void* user)
{
    const int iy0 = FM_MAX(bb[1], y0), iy1 = FM_MIN(bb[3], y1);
    const int cx0 = r->cx0, cx1 = r->cx1;
    uint64_t  px  = 0;
    if (!fm_reserve((void**)&r->active, &r->active_cap, (size_t)ne, sizeof(int), 0, 0)) return 0;
    if (!fm_reserve((void**)&r->xs, &r->xs_cap, (size_t)ne, sizeof(float), 0, 0)) return 0;
    if (!fm_reserve((void**)&r->ws, &r->ws_cap, (size_t)ne, sizeof(float), 0, 0)) return 0;

    int*   active = r->active;
    float* xs     = r->xs;
    float* ws     = r->ws;
    int    na = 0, next = 0;

    for (int y = iy0; y < iy1; y++) {
        float yc = (float)y + 0.5f;
        while (next < ne && edges[next].y0 <= yc) active[na++] = next++;
        int nx = 0, keep = 0;
        for (int k = 0; k < na; k++) {
            const fm_redge* e = &edges[active[k]];
            if (e->y1 <= yc) continue;
            active[keep++] = active[k];
            if (e->y0 > yc) continue;
            float x = e->x0 + (yc - e->y0) * e->dxdy;
            /* insertion sort by x */
            int j = nx++;
            while (j > 0 && xs[j - 1] > x) {
                xs[j] = xs[j - 1];
                ws[j] = ws[j - 1];
                j--;
            }
            xs[j] = x;
            ws[j] = e->dir;
        }
        na = keep;
        int   w = 0, inside = 0;
        float xa = 0.0f;
        for (int k = 0; k < nx; k++) {
            w += (int)ws[k];
            int now = evenodd ? (w & 1) : (w != 0);
            if (now && !inside) {
                xa = xs[k];
            } else if (!now && inside) {
                int p0 = FM_MAX(cx0, (int)ceilf(xa - 0.5f));
                int p1 = FM_MIN(cx1, (int)ceilf(xs[k] - 0.5f));
                if (p1 > p0) {
                    fn(user, y, p0, p1 - p0, NULL);
                    px += (uint64_t)(p1 - p0);
                }
            }
            inside = now;
        }
        if (inside) {
            int p0 = FM_MAX(cx0, (int)ceilf(xa - 0.5f));
            if (cx1 > p0) {
                fn(user, y, p0, cx1 - p0, NULL);
                px += (uint64_t)(cx1 - p0);
            }
        }
    }
    return px;
}

int fm__raster_prepare(fm_rasterizer* r, int bb[4])
{
    if (!fm_rasterizer_bounds(r, bb)) return 0;
    if (!r->sorted) {
        qsort(r->e, (size_t)r->n, sizeof(fm_redge), fm_edge_cmp);
        r->sorted = 1;
    }
    return 1;
}

const fm_redge* fm__raster_edges(const fm_rasterizer* r, int* n)
{
    *n = r->n;
    return r->e;
}

void fm__raster_render(fm_rasterizer* scratch, const fm_redge* e, int n, const int bb[4], int y0, int y1,
                       int evenodd, int aa, fm_span_fn fn, void* user)
{
    if (aa == FM_AA_NONE)
        fm_render_aliased(scratch, e, n, bb, y0, y1, evenodd, fn, user);
    else
        fm_render_analytic(scratch, e, n, bb, y0, y1, evenodd, fn, user);
}

void fm_rasterizer_render_rows(fm_rasterizer* r, int y0, int y1, fm_fill_rule rule, fm_aa_mode aa, fm_span_fn fn,
                               void* user)
{
    int bb[4];
    if (!fn || !fm__raster_prepare(r, bb)) return;
    int evenodd = rule == FM_FILL_EVENODD;
    if (aa == FM_AA_NONE) {
        FM_PROF_BEGIN(z_a, "raster.aliased");
        uint64_t px = fm_render_aliased(r, r->e, r->n, bb, y0, y1, evenodd, fn, user);
        FM_PROF_ITEMS(z_a, px);
        FM_PROF_END(z_a);
        (void)px;
    } else {
        FM_PROF_BEGIN(z_aa, "raster.analytic");
        uint64_t px = fm_render_analytic(r, r->e, r->n, bb, y0, y1, evenodd, fn, user);
        FM_PROF_ITEMS(z_aa, px);
        FM_PROF_END(z_aa);
        (void)px;
    }
}

void fm_rasterizer_render(fm_rasterizer* r, fm_fill_rule rule, fm_aa_mode aa, fm_span_fn fn, void* user)
{
    fm_rasterizer_render_rows(r, INT_MIN, INT_MAX, rule, aa, fn, user);
}
