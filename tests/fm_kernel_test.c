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
    printf("fm_kernel_test: %d backends, %d passed, %d failed\n", tested, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
