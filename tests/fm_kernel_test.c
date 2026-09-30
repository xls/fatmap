/*
 * fatmap kernel tests: every SIMD kernel table entry against the scalar
 * reference, called directly (internal header; needs the static library).
 * Covers lengths 0..70 (all vector widths + tails) and special values.
 */
#include "fm_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail, g_pass;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (cond) {                                     \
            g_pass++;                                   \
        } else {                                        \
            g_fail++;                                   \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
        }                                               \
    } while (0)

static uint32_t g_rng = 987654321u;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float rndf(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xffffff) / 16777216.0f; }

/* a float with a bias toward the values that break sloppy SIMD code */
static float special(void)
{
    switch (rnd() % 16) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return 1.0f;
    case 3: return NAN;
    case 4: return INFINITY;
    case 5: return -INFINITY;
    case 6: return 1e-40f; /* denormal */
    default: return rndf(-3.0f, 3.0f);
    }
}

/* bit equality, any NaN equals any NaN (payloads are not meaningful) */
static int same(const float* a, const float* b, int n)
{
    for (int i = 0; i < n; i++) {
        if (isnan(a[i]) && isnan(b[i])) continue;
        if (memcmp(&a[i], &b[i], 4) != 0) return 0;
    }
    return 1;
}

#define N 72

#if FM_FEATURE_TNL
/* random light setup + vertices for the lighting kernel */
static void rnd_lights(fm_light_params* p)
{
    memset(p, 0, sizeof(*p));
    p->nlights = 1 + (int)(rnd() % FM_MAX_LIGHTS);
    for (int l = 0; l < p->nlights; l++) {
        struct fm_light_k* L = &p->l[l];
        L->type              = (int)(rnd() % 3);
        float dx = rndf(-1, 1), dy = rndf(-1, 1), dz = rndf(-1, 1);
        float n  = sqrtf(dx * dx + dy * dy + dz * dz) + 1e-3f;
        L->dir[0] = dx / n, L->dir[1] = dy / n, L->dir[2] = dz / n;
        for (int k = 0; k < 3; k++) {
            L->pos[k] = rndf(-5, 5);
            L->amb[k] = rndf(0, 0.3f), L->dif[k] = rndf(0, 1), L->spe[k] = rndf(0, 1);
        }
        L->katt[0]  = rndf(0.2f, 1.5f), L->katt[1] = rndf(0, 0.3f), L->katt[2] = rndf(0, 0.1f);
        L->spot_cos = rndf(0.3f, 0.95f);
        L->spot_exp = (rnd() & 1) ? rndf(0, 40) : 0.0f;
    }
    for (int k = 0; k < 3; k++) {
        p->mat_amb[k] = rndf(0, 1), p->mat_dif[k] = rndf(0, 1), p->mat_spe[k] = rndf(0, 1);
        p->mat_emi[k] = rndf(0, 0.2f), p->gamb[k] = rndf(0, 0.3f);
    }
    p->mat_dif[3]     = rndf(0, 1);
    p->shininess      = rndf(1, 120);
    p->color_material = (int)(rnd() & 1);
}

/* the same model in double precision with libm (independent reference) */
static void light_ref(const fm_light_params* p, const float* in[10], int i, double out[4])
{
    double P[3] = { in[0][i], in[1][i], in[2][i] }, Nn[3] = { in[3][i], in[4][i], in[5][i] }, V[3];
    double nl = sqrt(Nn[0] * Nn[0] + Nn[1] * Nn[1] + Nn[2] * Nn[2]);
    for (int k = 0; k < 3; k++) Nn[k] /= nl, V[k] = -P[k];
    double vl = sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
    for (int k = 0; k < 3; k++) V[k] /= vl;
    double a[3] = { 0, 0, 0 }, d[3] = { 0, 0, 0 }, sp[3] = { 0, 0, 0 };
    for (int l = 0; l < p->nlights; l++) {
        const struct fm_light_k* L = &p->l[l];
        double                   Ld[3], att = 1;
        if (L->type == 0) {
            for (int k = 0; k < 3; k++) Ld[k] = L->dir[k];
        } else {
            double D[3] = { L->pos[0] - P[0], L->pos[1] - P[1], L->pos[2] - P[2] };
            double d2 = D[0] * D[0] + D[1] * D[1] + D[2] * D[2], dl = sqrt(d2);
            for (int k = 0; k < 3; k++) Ld[k] = D[k] / dl;
            att = 1.0 / (L->katt[0] + L->katt[1] * dl + L->katt[2] * d2);
            if (L->type == 2) {
                double sd = -(Ld[0] * L->dir[0] + Ld[1] * L->dir[1] + Ld[2] * L->dir[2]);
                att *= sd >= L->spot_cos ? (L->spot_exp > 0 ? pow(sd > 0 ? sd : 0, L->spot_exp) : 1.0) : 0.0;
            }
        }
        double ndl = Nn[0] * Ld[0] + Nn[1] * Ld[1] + Nn[2] * Ld[2];
        ndl        = ndl > 0 ? ndl : 0;
        double H[3] = { Ld[0] + V[0], Ld[1] + V[1], Ld[2] + V[2] };
        double hl   = sqrt(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
        double ndh  = (Nn[0] * H[0] + Nn[1] * H[1] + Nn[2] * H[2]) / hl;
        double sv   = ndl > 0 && ndh > 0 ? pow(ndh, p->shininess) : 0;
        for (int k = 0; k < 3; k++) a[k] += att * L->amb[k], d[k] += att * ndl * L->dif[k], sp[k] += att * sv * L->spe[k];
    }
    for (int k = 0; k < 3; k++) {
        double md = p->color_material ? in[6 + k][i] : p->mat_dif[k], ma = p->color_material ? md : p->mat_amb[k];
        double c  = p->mat_emi[k] + p->gamb[k] * ma + a[k] * ma + d[k] * md + sp[k] * p->mat_spe[k];
        out[k]    = c < 0 ? 0 : (c > 1 ? 1 : c);
    }
    double al = p->color_material ? in[9][i] : p->mat_dif[3];
    out[3]    = al < 0 ? 0 : (al > 1 ? 1 : al);
}

static void light_input(float buf[10][N], const float* in[10], int n)
{
    for (int i = 0; i < n; i++) {
        buf[0][i] = rndf(-3, 3), buf[1][i] = rndf(-3, 3), buf[2][i] = rndf(-8, -1); /* in front of the eye */
        buf[3][i] = rndf(-1, 1), buf[4][i] = rndf(-1, 1), buf[5][i] = rndf(-1, 1) + 0.01f;
        for (int k = 6; k < 10; k++) buf[k][i] = rndf(0, 1);
    }
    for (int k = 0; k < 10; k++) in[k] = buf[k];
}
#endif

static void test_table(const fm_kernels* k)
{
    const fm_kernels* s   = &fm_kernels_scalar;
    const char*       nm  = fm_simd_name(k->level);
    int               bad = 0;
    float dx[N], w[N], r0[N], r1[N], acc0[N], acc1[N];
    for (int iter = 0; iter < 4000; iter++) {
        int   n    = (int)(rnd() % 71);
        int   spec = (iter & 3) == 0; /* every 4th run uses special values */
        float a    = spec ? special() : rndf(-2.0f, 2.0f);
        float b    = spec ? special() : rndf(-0.1f, 0.1f);
        for (int i = 0; i < N; i++) {
            dx[i] = spec && (rnd() & 7) == 0 ? special() : ((float)i + 0.5f) - rndf(0.0f, 1.0f);
            w[i]  = spec && (rnd() & 7) == 0 ? special() : rndf(0.1f, 4.0f);
        }
        for (int clamp = 0; clamp < 2; clamp++) {
            memset(r0, 0x5a, sizeof(r0));
            memset(r1, 0x5a, sizeof(r1));
            s->plane(a, b, dx, n, clamp, r0);
            k->plane(a, b, dx, n, clamp, r1);
            if (!same(r0, r1, N) && bad++ < 5) printf("  %s plane n=%d clamp=%d a=%g b=%g\n", nm, n, clamp, a, b);
        }
        memset(r0, 0x5a, sizeof(r0));
        memset(r1, 0x5a, sizeof(r1));
        s->plane_recip(a, b, dx, n, r0);
        k->plane_recip(a, b, dx, n, r1);
        if (!same(r0, r1, N) && bad++ < 5) printf("  %s plane_recip n=%d a=%g b=%g\n", nm, n, a, b);

        memset(r0, 0x5a, sizeof(r0));
        memset(r1, 0x5a, sizeof(r1));
        s->plane_mul(a, b, dx, w, n, r0);
        k->plane_mul(a, b, dx, w, n, r1);
        if (!same(r0, r1, N) && bad++ < 5) printf("  %s plane_mul n=%d a=%g b=%g\n", nm, n, a, b);

        for (int i = 0; i < N; i++) acc0[i] = acc1[i] = spec ? special() : rndf(-1.0f, 1.0f);
        s->acc_add(acc0, a, n);
        k->acc_add(acc1, a, n);
        if (!same(acc0, acc1, N) && bad++ < 5) printf("  %s acc_add n=%d v=%g\n", nm, n, a);
    }
    CHECK(bad == 0, "%s rasterizer kernels differ from scalar in %d cases", nm, bad);

    /* minmax_f32: depth rows (non negative, may hold -0) */
    int badm = 0;
    for (int iter = 0; iter < 4000; iter++) {
        int n = 1 + (int)(rnd() % 70);
        for (int i = 0; i < N; i++) {
            switch (rnd() % 8) {
            case 0: dx[i] = 0.0f; break;
            case 1: dx[i] = -0.0f; break;
            case 2: dx[i] = 1.0f; break;
            case 3: dx[i] = 1e-40f; break;
            default: dx[i] = rndf(0.0f, 1.0f); break;
            }
        }
        float a0, b0, a1, b1;
        s->minmax_f32(dx, n, &a0, &b0);
        k->minmax_f32(dx, n, &a1, &b1);
        /* -0 and +0 are interchangeable here */
        if ((a0 != a1 || b0 != b1) && badm++ < 5) printf("  %s minmax_f32 n=%d\n", nm, n);
    }
    CHECK(badm == 0, "%s minmax_f32 differs from scalar in %d cases", nm, badm);

    /* bilinear_pts_wrap: random texels, coordinates far outside the texture */
    int             badw = 0;
    static uint32_t tex[64 * 32];
    for (int i = 0; i < 64 * 32; i++) tex[i] = rnd();
    for (int iter = 0; iter < 2000; iter++) {
        int     n = (int)(rnd() % 70), tw = 1 << (1 + rnd() % 6), th = 1 << (rnd() % 6);
        int32_t U[N], V[N];
        uint32_t o0[N], o1[N];
        for (int i = 0; i < N; i++) {
            U[i] = (int32_t)(rnd() % (1u << 26)) - (1 << 25);
            V[i] = (int32_t)(rnd() % (1u << 26)) - (1 << 25);
        }
        s->bilinear_pts_wrap(tex, 64, U, V, n, tw - 1, th - 1, o0);
        k->bilinear_pts_wrap(tex, 64, U, V, n, tw - 1, th - 1, o1);
        if (memcmp(o0, o1, (size_t)n * 4) != 0 && badw++ < 5) printf("  %s bilinear_pts_wrap n=%d\n", nm, n);
    }
    CHECK(badw == 0, "%s bilinear_pts_wrap differs from scalar in %d cases", nm, badw);

#if FM_FEATURE_TNL
    /* lighting kernel: bit identical to the scalar reference */
    int badl = 0;
    for (int iter = 0; iter < 800; iter++) {
        fm_light_params lp;
        rnd_lights(&lp);
        int          n = 1 + (int)(rnd() % 70);
        static float lbuf[10][N], o0[4][N], o1[4][N];
        const float* lin[10];
        light_input(lbuf, lin, n);
        float* out0[4] = { o0[0], o0[1], o0[2], o0[3] };
        float* out1[4] = { o1[0], o1[1], o1[2], o1[3] };
        s->light(&lp, lin, n, out0);
        k->light(&lp, lin, n, out1);
        for (int c = 0; c < 4; c++)
            if (!same(o0[c], o1[c], n) && badl++ < 5) printf("  %s light n=%d channel %d\n", nm, n, c);
    }
    CHECK(badl == 0, "%s light differs from scalar in %d cases", nm, badl);
#endif
}

int main(void)
{
    fm_init();
    int tested = 0;
#ifdef FM_HAVE_SSE2
    if (fm_simd_supported(FM_SIMD_SSE2)) test_table(&fm_kernels_sse2), tested++;
#endif
#ifdef FM_HAVE_AVX2
    if (fm_simd_supported(FM_SIMD_AVX2)) test_table(&fm_kernels_avx2), tested++;
#endif
#ifdef FM_HAVE_NEON
    if (fm_simd_supported(FM_SIMD_NEON)) test_table(&fm_kernels_neon), tested++;
#endif
    /* the reference must match its own semantics too */
    {
        float dx[3] = { 0.5f, 1.5f, 2.5f }, out[3];
        fm_kernels_scalar.plane(-0.5f, 1.0f, dx, 3, 1, out);
        CHECK(out[0] == 0.0f && out[1] == 1.0f && out[2] == 1.0f, "plane clamp01");
        fm_kernels_scalar.plane_recip(0.0f, 2.0f, dx, 3, out);
        CHECK(out[0] == 1.0f && out[1] == 1.0f / 3.0f && out[2] == 0.2f, "plane_recip");
    }
    /* scalar bilinear_pts_wrap against an independent reference: modulo
     * wrapping as in the generic sampler path, its own lerp */
    {
        static uint32_t tex[16 * 8];
        for (int i = 0; i < 16 * 8; i++) tex[i] = rnd();
        int bad = 0;
        for (int it = 0; it < 5000; it++) {
            int32_t  u = (int32_t)(rnd() % (1u << 24)) - (1 << 23), v = (int32_t)(rnd() % (1u << 24)) - (1 << 23);
            uint32_t got;
            fm_kernels_scalar.bilinear_pts_wrap(tex, 16, &u, &v, 1, 15, 7, &got);
            int x0 = (u >> 16) % 16, y0 = (v >> 16) % 8;
            x0 += x0 < 0 ? 16 : 0;
            y0 += y0 < 0 ? 8 : 0;
            int      x1 = (x0 + 1) % 16, y1 = (y0 + 1) % 8;
            uint32_t fx = (uint32_t)(u >> 8) & 255u, fy = (uint32_t)(v >> 8) & 255u, ref = 0;
            for (int sh = 0; sh < 32; sh += 8) {
                uint32_t a = (tex[y0 * 16 + x0] >> sh) & 255, b = (tex[y0 * 16 + x1] >> sh) & 255;
                uint32_t c = (tex[y1 * 16 + x0] >> sh) & 255, d = (tex[y1 * 16 + x1] >> sh) & 255;
                uint32_t t = (a * (256 - fx) + b * fx) >> 8, bo = (c * (256 - fx) + d * fx) >> 8;
                ref |= ((t * (256 - fy) + bo * fy) >> 8) << sh;
            }
            bad += got != ref;
        }
        CHECK(bad == 0, "bilinear_pts_wrap matches the modulo reference (%d mismatches)", bad);
    }
#if FM_FEATURE_TNL
    /* the scalar reference against double precision libm math */
    {
        double maxe = 0;
        for (int iter = 0; iter < 400; iter++) {
            fm_light_params lp;
            rnd_lights(&lp);
            static float lbuf[10][N], o[4][N];
            const float* lin[10];
            light_input(lbuf, lin, N);
            float* out[4] = { o[0], o[1], o[2], o[3] };
            fm_kernels_scalar.light(&lp, lin, N, out);
            for (int i = 0; i < N; i++) {
                double rf[4];
                light_ref(&lp, lin, i, rf);
                for (int c = 0; c < 4; c++) {
                    double e = fabs(rf[c] - (double)o[c][i]);
                    maxe     = e > maxe ? e : maxe;
                }
            }
        }
        CHECK(maxe < 2e-3, "lighting matches a double precision reference (max error %.2e)", maxe);
        printf("  lighting max error vs double reference: %.2e\n", maxe);
    }
#endif
    printf("fm_kernel_test: %d backends, %d passed, %d failed\n", tested, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
