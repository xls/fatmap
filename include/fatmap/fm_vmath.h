/*
 * fatmap - vectorizable, deterministic float math (header only).
 *
 * sin / cos / tan / exp / exp2 / log / log2 / pow, floor / ceil / trunc /
 * round / rint and fmin / fmax for float, written as straight line code
 * (selects, no branches, no calls, no tables) so a compiler vectorizes the
 * loops that call them. They are what fatmap's
 * shader backends use (the SPIR-V interpreter and the C it generates), and
 * C shaders can use them too.
 *
 * Results only depend on IEEE add / sub / mul / div and conversions, so
 * they are the same bits on every platform and whether a loop is
 * vectorized or not, unlike the C library's functions (which differ
 * between MSVC, glibc, macOS, ...). Compile without -ffast-math (it would
 * remove the rounding steps) and without floating point contraction.
 *
 * Rounding and min / max are exact (fminf / fmaxf return x for equal
 * values, the other operand for one NaN).
 * Accuracy (the core math runs in double): within 1 ulp of the correctly
 * rounded result for sin / cos / tan with |x| < 2^20, exp / exp2 / log /
 * log2 / pow over the whole float range; C99 results for the special cases
 * (NaN, infinities, zeros, pow of negative numbers). Sines of larger
 * arguments lose precision gradually (like GPUs, unlike the C library).
 *
 * fm_fast_*: the same functions in float only, about twice as fast, at
 * the precision GPUs give (and well inside what Vulkan requires): a few
 * ulp for exp2 / log2 / exp / log, absolute error below 1e-6 for sin /
 * cos with |x| < 8192 (Vulkan: 2^-11 in [-pi, pi]); pow is
 * exp2(y * log2(x)), so its relative error grows with |y * log2(x)|.
 * Also deterministic.
 *
 * Polynomials: sin / cos minimax from FreeBSD msun (k_sinf.c / k_cosf.c,
 * BSD license) and Cephes (sinf.c, MIT style license); log2 / exp2 are
 * the atanh and Taylor series.
 */
#ifndef FATMAP_FM_VMATH_H
#define FATMAP_FM_VMATH_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* always inlined: a loop only vectorizes when the calls are gone */
#if defined(_MSC_VER)
#  define FM_VMATH_FN static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define FM_VMATH_FN static inline __attribute__((always_inline))
#else
#  define FM_VMATH_FN static inline
#endif

FM_VMATH_FN uint32_t fm__fbits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
FM_VMATH_FN float fm__bitsf(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}
FM_VMATH_FN uint64_t fm__dbits(double d)
{
    uint64_t u;
    memcpy(&u, &d, 8);
    return u;
}
FM_VMATH_FN double fm__bitsd(uint64_t u)
{
    double d;
    memcpy(&d, &u, 8);
    return d;
}

/* branch free selects: c ? a : b (c is 0 / 1) */
FM_VMATH_FN float fm__self(int c, float a, float b)
{
    uint32_t m = 0u - (uint32_t)c;
    return fm__bitsf((fm__fbits(a) & m) | (fm__fbits(b) & ~m));
}
FM_VMATH_FN double fm__seld(int c, double a, double b)
{
    uint64_t m = 0ull - (uint64_t)c;
    return fm__bitsd((fm__dbits(a) & m) | (fm__dbits(b) & ~m));
}

/* round to nearest even, |v| < 2^51 (adding 1.5 * 2^52 rounds) */
FM_VMATH_FN double fm__rint(double v) { return (v + 6755399441055744.0) - 6755399441055744.0; }

/* lo when v < lo or NaN, hi when v > hi */
FM_VMATH_FN double fm__clampd(double v, double lo, double hi)
{
    v = fm__seld(v > lo, v, lo);
    return fm__seld(v < hi, v, hi);
}

/* ---- rounding, min / max: exact (the C library's results), but straight
 * line code; floorf & co. need SSE4.1 to vectorize, fminf / fmaxf have NaN
 * rules no min / max instruction has ---- */

FM_VMATH_FN float fm__abs(float x) { return fm__bitsf(fm__fbits(x) & 0x7fffffffu); }
FM_VMATH_FN float fm__withsign(float r, float x) { return fm__bitsf(fm__fbits(r) | (fm__fbits(x) & 0x80000000u)); }

/* toward zero; |x| >= 2^23, inf, NaN are already integral */
FM_VMATH_FN float fm_truncf(float x)
{
    float t = fm__withsign((float)(int32_t)fm__self(fm__abs(x) < 8388608.0f, x, 0.0f), x);
    return fm__self(fm__abs(x) < 8388608.0f, t, x);
}
FM_VMATH_FN float fm_floorf(float x)
{
    float t = fm_truncf(x);
    return fm__self(t > x, t - 1.0f, t);
}
FM_VMATH_FN float fm_ceilf(float x)
{
    float t = fm_truncf(x);
    return fm__self(t < x, t + 1.0f, t);
}
/* halfway cases away from zero (roundf) */
FM_VMATH_FN float fm_roundf(float x)
{
    float t = fm_truncf(x);
    return fm__self(fm__abs(x - t) >= 0.5f, t + fm__withsign(1.0f, x), t);
}
/* halfway cases to even (rintf in the default rounding mode) */
FM_VMATH_FN float fm_rintf(float x)
{
    float m = fm__withsign(8388608.0f, x);
    float r = fm__withsign((x + m) - m, x);
    return fm__self(fm__abs(x) < 8388608.0f, r, x);
}
/* the other operand when one is NaN (fminf / fmaxf); x for equal values */
FM_VMATH_FN float fm_fminf(float x, float y) { return fm__self((y < x) | (x != x), y, x); }
FM_VMATH_FN float fm_fmaxf(float x, float y) { return fm__self((y > x) | (x != x), y, x); }

/* log2(x) for finite x > 0 (denormals too) */
FM_VMATH_FN double fm__log2_core(float x)
{
    int      den = fm__fbits(x) < 0x00800000u;
    uint32_t u   = fm__fbits(fm__self(den, x * 8388608.0f, x)); /* denormals * 2^23: exact */
    int      e   = (int)(u >> 23) - 127 - 23 * den;
    uint32_t mu  = (u & 0x007fffffu) | 0x3f800000u; /* mantissa in [1, 2) */
    int      hi  = mu > 0x3fb504f3u;                /* > sqrt(2): use m / 2 in [0.707, 1) */
    mu -= (uint32_t)hi << 23;
    double m = (double)fm__bitsf(mu), s = (m - 1.0) / (m + 1.0), s2 = s * s;
    double p = 0.3205988979753252 + s2 * 0.2623081892525388;
    p        = 0.5770780163555853 + s2 * (0.41219858311113244 + s2 * p);
    p        = s * (2.8853900817779268 + s2 * (0.9617966939259757 + s2 * p));
    return (double)(e + hi) + p;
}

/* 2^y for |y| <= 1000 */
FM_VMATH_FN double fm__exp2_core(double y)
{
    double n = fm__rint(y), f = y - n; /* f in [-0.5, 0.5] */
    double p = 0.0013333558146428441 + f * (0.00015403530393381606 + f * (1.5252733804059838e-05 + f * 1.3215486790144305e-06));
    p        = 1.0 + f * (0.6931471805599453 + f * (0.2402265069591007 + f * (0.055504108664821576 + f * (0.009618129107628477 + f * p))));
    /* 2^n: n + 1023 in the exponent field (low bits of n + 1023 + 2^52) */
    double s = (n + 1023.0) + 4503599627370496.0;
    return p * fm__bitsd(fm__dbits(s) << 52);
}

FM_VMATH_FN float fm_exp2f(float x)
{
    float r = (float)fm__exp2_core(fm__clampd((double)x, -200.0, 200.0));
    return fm__self(x != x, x, r);
}

FM_VMATH_FN float fm_expf(float x)
{
    float r = (float)fm__exp2_core(fm__clampd((double)x * 1.4426950408889634, -200.0, 200.0));
    return fm__self(x != x, x, r);
}

/* the special cases of log / log2: 0 -> -inf, < 0 -> NaN, inf -> inf, NaN */
FM_VMATH_FN float fm__log_special(float x, float r)
{
    r = fm__self(x == fm__bitsf(0x7f800000u), x, r);
    r = fm__self(x == 0.0f, fm__bitsf(0xff800000u), r);
    r = fm__self(x < 0.0f, fm__bitsf(0x7fc00000u), r);
    return fm__self(x != x, x, r);
}

FM_VMATH_FN float fm_log2f(float x)
{
    float xs = fm__self((x > 0.0f) & (x < fm__bitsf(0x7f800000u)), x, 1.0f);
    return fm__log_special(x, (float)fm__log2_core(xs));
}

FM_VMATH_FN float fm_logf(float x)
{
    float xs = fm__self((x > 0.0f) & (x < fm__bitsf(0x7f800000u)), x, 1.0f);
    return fm__log_special(x, (float)(fm__log2_core(xs) * 0.6931471805599453));
}

/* x^y with C99 powf's special cases */
FM_VMATH_FN float fm_powf(float x, float y)
{
    const float inf = fm__bitsf(0x7f800000u), nan = fm__bitsf(0x7fc00000u);
    float       ax  = fm__bitsf(fm__fbits(x) & 0x7fffffffu), ay = fm__bitsf(fm__fbits(y) & 0x7fffffffu);
    float       axs = fm__self((ax > 0.0f) & (ax < inf), ax, 1.0f);
    double      t   = fm__clampd((double)y * fm__log2_core(axs), -1000.0, 1000.0);
    float       r   = (float)fm__exp2_core(t); /* |x|^y for finite nonzero x */
    /* y an integer (all floats >= 2^24 are), odd */
    double yr   = fm__rint(fm__clampd((double)y, -16777216.0, 16777216.0));
    int    yint = (ay >= 16777216.0f) | (yr == (double)y);
    int    odd  = (ay < 16777216.0f) & (yr == (double)y) & ((int)yr & 1);
    float  big  = fm__self(y > 0.0f, inf, 0.0f); /* |x| > 1 (or inf) to the y */
    r           = fm__self(ax == inf, big, r);
    r           = fm__self(ax == 0.0f, fm__self(y > 0.0f, 0.0f, fm__self(y < 0.0f, inf, r)), r);
    r           = fm__self(ay == inf, fm__self(ax == 1.0f, 1.0f, fm__self(ax > 1.0f, big, fm__self(y > 0.0f, 0.0f, inf))), r);
    /* negative x (and -0): the sign of odd integer powers; NaN for finite x < 0 to a fractional power */
    float neg = fm__self(yint, fm__bitsf(fm__fbits(r) ^ ((uint32_t)odd << 31)), fm__self((ax == 0.0f) | (ax == inf), r, nan));
    r         = fm__self((int)(fm__fbits(x) >> 31), neg, r);
    r         = fm__self((x != x) | (y != y), nan, r);
    return fm__self((y == 0.0f) | (x == 1.0f), 1.0f, r);
}

/* sin (q = 0) or cos (q = 1) of x, r = x - n pi/2 in double (two part
 * pi/2, fdlibm's medium range reduction) */
FM_VMATH_FN double fm__reduce(float x, int* n)
{
    double xd = (double)x;
    double fn = fm__rint(fm__clampd(xd * 0.6366197723675814, -1073741824.0, 1073741824.0));
    *n        = (int)fn;
    double r  = (xd - fn * 1.57079632673412561417) - fn * 6.07710050650619224932e-11;
    return fm__clampd(r, -1.0, 1.0); /* |r| <= pi / 4 when reduced; huge |x|: bounded garbage, not inf */
}
FM_VMATH_FN double fm__sinp(double r, double z)
{
    return r + r * z * (-0.166666666416265235595 + z * (0.0083333293858894631756 + z * (-0.000198393348360966317347 + z * 0.0000027183114939898219064)));
}
FM_VMATH_FN double fm__cosp(double z)
{
    return 1.0 + z * (-0.499999997251031003120 + z * (0.0416666233237390631894 + z * (-0.00138867637746099294692 + z * 0.0000243904487962774090654)));
}

FM_VMATH_FN float fm__sincos(float x, int q)
{
    int    n;
    double r = fm__reduce(x, &n), z = r * r;
    int    k = n + q;
    double v = fm__seld(k & 1, fm__cosp(z), fm__sinp(r, z));
    v        = fm__bitsd(fm__dbits(v) ^ ((uint64_t)((k >> 1) & 1) << 63)); /* quadrants 2, 3: negated */
    return fm__self(x - x == 0.0f, (float)v, x - x);                        /* inf / NaN: NaN */
}

FM_VMATH_FN float fm_sinf(float x) { return fm__sincos(x, 0); }
FM_VMATH_FN float fm_cosf(float x) { return fm__sincos(x, 1); }

FM_VMATH_FN float fm_tanf(float x)
{
    int    n;
    double r = fm__reduce(x, &n), z = r * r, sp = fm__sinp(r, z), cp = fm__cosp(z);
    double v = fm__seld(n & 1, -cp / sp, sp / cp);
    return fm__self(x - x == 0.0f, (float)v, x - x);
}

/* ---- fast: float only ---- */

/* log2(x), finite x > 0 */
FM_VMATH_FN float fm__fast_log2_core(float x)
{
    int      den = fm__fbits(x) < 0x00800000u;
    uint32_t u   = fm__fbits(fm__self(den, x * 8388608.0f, x));
    int      e   = (int)(u >> 23) - 127 - 23 * den;
    uint32_t mu  = (u & 0x007fffffu) | 0x3f800000u;
    int      hi  = mu > 0x3fb504f3u;
    mu -= (uint32_t)hi << 23;
    float m = fm__bitsf(mu), s = (m - 1.0f) / (m + 1.0f), s2 = s * s;
    float p = s * (2.88539008f + s2 * (0.961796694f + s2 * (0.577078016f + s2 * (0.412198583f + s2 * 0.320598898f))));
    return (float)(e + hi) + p;
}

/* 2^y; y clamped to [-151, 128]: 2^n as two normal factors 2^(n/2) 2^(n - n/2),
 * so results underflow to correctly rounded denormals / 0 and overflow to inf */
FM_VMATH_FN float fm__fast_exp2_core(float y)
{
    y       = fm__self(y > -151.0f, y, -151.0f);
    y       = fm__self(y < 128.0f, y, 128.0f);
    float n = (y + 12582912.0f) - 12582912.0f, f = y - n; /* f in [-0.5, 0.5] */
    float p = 0.00961812911f + f * (0.00133335581f + f * 0.000154035304f);
    p       = 1.0f + f * (0.693147181f + f * (0.240226507f + f * (0.0555041087f + f * p)));
    int   k = (int)n, k1 = k / 2, k2 = k - k1;
    return p * fm__bitsf((uint32_t)(k1 + 127) << 23) * fm__bitsf((uint32_t)(k2 + 127) << 23);
}

FM_VMATH_FN float fm_fast_exp2f(float x) { return fm__self(x != x, x, fm__fast_exp2_core(x)); }
FM_VMATH_FN float fm_fast_expf(float x) { return fm__self(x != x, x, fm__fast_exp2_core(x * 1.44269504f)); }

FM_VMATH_FN float fm_fast_log2f(float x)
{
    float xs = fm__self((x > 0.0f) & (x < fm__bitsf(0x7f800000u)), x, 1.0f);
    return fm__log_special(x, fm__fast_log2_core(xs));
}

FM_VMATH_FN float fm_fast_logf(float x)
{
    float xs = fm__self((x > 0.0f) & (x < fm__bitsf(0x7f800000u)), x, 1.0f);
    return fm__log_special(x, fm__fast_log2_core(xs) * 0.693147181f);
}

/* x^y = exp2(y log2 |x|) with fm_powf's special cases */
FM_VMATH_FN float fm_fast_powf(float x, float y)
{
    const float inf = fm__bitsf(0x7f800000u), nan = fm__bitsf(0x7fc00000u);
    float       ax  = fm__bitsf(fm__fbits(x) & 0x7fffffffu), ay = fm__bitsf(fm__fbits(y) & 0x7fffffffu);
    float       axs = fm__self((ax > 0.0f) & (ax < inf), ax, 1.0f);
    float       t   = y * fm__fast_log2_core(axs);
    float       r   = fm__fast_exp2_core(fm__self(t != t, 0.0f, t));
    float       yr  = (fm__self(ay < 8388608.0f, y, 0.0f) + 12582912.0f) - 12582912.0f; /* rint for |y| < 2^23 */
    int         yint = (ay >= 8388608.0f) | (yr == y);
    int         odd  = (ay < 8388608.0f) & (yr == y) & ((int)yr & 1);
    float       big  = fm__self(y > 0.0f, inf, 0.0f);
    r                = fm__self(ax == inf, big, r);
    r                = fm__self(ax == 0.0f, fm__self(y > 0.0f, 0.0f, fm__self(y < 0.0f, inf, r)), r);
    r                = fm__self(ay == inf, fm__self(ax == 1.0f, 1.0f, fm__self(ax > 1.0f, big, fm__self(y > 0.0f, 0.0f, inf))), r);
    float neg        = fm__self(yint, fm__bitsf(fm__fbits(r) ^ ((uint32_t)odd << 31)), fm__self((ax == 0.0f) | (ax == inf), r, nan));
    r                = fm__self((int)(fm__fbits(x) >> 31), neg, r);
    r                = fm__self((x != x) | (y != y), nan, r);
    return fm__self((y == 0.0f) | (x == 1.0f), 1.0f, r);
}

/* x - n pi/2 with pi/2 in three float parts (Cephes) */
FM_VMATH_FN float fm__fast_reduce(float x, int* n)
{
    float c  = x * 0.636619772f;
    float fn = (fm__self(fm__abs(c) < 4194304.0f, c, 0.0f) + 12582912.0f) - 12582912.0f;
    *n       = (int)fn;
    float r  = ((x - fn * 1.5703125f) - fn * 4.83751297e-4f) - fn * 7.54978995e-8f;
    r        = fm__self(r > -1.0f, r, -1.0f); /* |x| > 6.6e6 is not reduced: bounded garbage, not inf */
    return fm__self(r < 1.0f, r, 1.0f);
}
FM_VMATH_FN float fm__fast_sinp(float r, float z) { return r + r * z * (-0.166666546f + z * (0.00833216087f + z * -0.000195152959f)); }
FM_VMATH_FN float fm__fast_cosp(float z)
{
    return 1.0f + z * (-0.5f + z * (0.0416666642f + z * (-0.00138873170f + z * 0.0000244331571f)));
}

FM_VMATH_FN float fm__fast_sincos(float x, int q)
{
    int   n;
    float r = fm__fast_reduce(x, &n), z = r * r;
    int   k = n + q;
    float v = fm__self(k & 1, fm__fast_cosp(z), fm__fast_sinp(r, z));
    v       = fm__bitsf(fm__fbits(v) ^ ((uint32_t)((k >> 1) & 1) << 31));
    return fm__self(x - x == 0.0f, v, x - x);
}

FM_VMATH_FN float fm_fast_sinf(float x) { return fm__fast_sincos(x, 0); }
FM_VMATH_FN float fm_fast_cosf(float x) { return fm__fast_sincos(x, 1); }

FM_VMATH_FN float fm_fast_tanf(float x)
{
    int   n;
    float r = fm__fast_reduce(x, &n), z = r * r, sp = fm__fast_sinp(r, z), cp = fm__fast_cosp(z);
    float v = fm__self(n & 1, -cp / sp, sp / cp);
    return fm__self(x - x == 0.0f, v, x - x);
}

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_VMATH_H */
