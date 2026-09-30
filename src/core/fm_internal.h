/* fatmap - internal core helpers (not installed) */
#ifndef FATMAP_FM_INTERNAL_H
#define FATMAP_FM_INTERNAL_H

#include <fatmap/fm_core.h>
#include <fatmap/fm_profile.h>
#include <stdlib.h>
#include <string.h>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#  define FM_ARCH_X86 1
#elif defined(_M_ARM64) || defined(__aarch64__)
#  define FM_ARCH_ARM64 1
#endif

#if defined(_MSC_VER)
#  define FM_INLINE    static __forceinline
#  define FM_NOINLINE  __declspec(noinline)
#  define FM_LIKELY(x) (x)
#  define FM_UNLIKELY(x) (x)
#else
#  define FM_INLINE    static inline __attribute__((always_inline))
#  define FM_NOINLINE  __attribute__((noinline))
#  define FM_LIKELY(x) __builtin_expect(!!(x), 1)
#  define FM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#define FM_MIN(a, b) ((a) < (b) ? (a) : (b))
#define FM_MAX(a, b) ((a) > (b) ? (a) : (b))
#define FM_CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

/* Exact x / 255 with rounding for x in [0, 65535 - 255]. */
FM_INLINE uint32_t fm_div255(uint32_t x)
{
    x += 128;
    return (x + (x >> 8)) >> 8;
}

FM_INLINE uint32_t fm_premul_inline(fm_color c)
{
    uint32_t a = c >> 24;
    if (a == 255) return c;
    if (a == 0) return 0;
    uint32_t r = fm_div255(((c >> 16) & 255) * a);
    uint32_t g = fm_div255(((c >> 8) & 255) * a);
    uint32_t b = fm_div255((c & 255) * a);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

/* Scale all 4 channels of a premultiplied pixel by k/255. */
FM_INLINE uint32_t fm_px_scale(uint32_t p, uint32_t k)
{
    uint32_t a = fm_div255((p >> 24) * k);
    uint32_t r = fm_div255(((p >> 16) & 255) * k);
    uint32_t g = fm_div255(((p >> 8) & 255) * k);
    uint32_t b = fm_div255((p & 255) * k);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

void* fm_aligned_alloc(size_t size, size_t align);
void  fm_aligned_free(void* p);
void  fm__init(void); /* idempotent: CPU detection + kernel selection */

/* ------------------------------------------------------------------------
 * Kernel table. Every backend must produce bit-identical output; the
 * op formulas live in fm_kernels_tmpl.h and are shared by all backends.
 * ---------------------------------------------------------------------- */

#define FM_GRAD_LUT_BITS 10
#define FM_GRAD_LUT_SIZE (1 << FM_GRAD_LUT_BITS)

typedef struct fm_kernels {
    fm_simd_level level;
    void (*fill)(uint32_t* d, uint32_t v, int n);
    void (*solid_over)(uint32_t* d, uint32_t s, int n);
    void (*solid_over_mask)(uint32_t* d, uint32_t s, const uint8_t* m, int n);
    void (*span_over)(uint32_t* d, const uint32_t* s, int n);
    void (*span_over_mask)(uint32_t* d, const uint32_t* s, const uint8_t* m, int n);
    /* any op except SRC_OVER; m may be NULL */
    void (*span_op)(uint32_t* d, const uint32_t* s, const uint8_t* m, int n, int op);
    void (*mask_mul)(uint8_t* m, const uint8_t* c, int n);
    void (*mask_scale)(uint8_t* m, uint32_t k, int n);
    /* prefix-sum coverage accumulation; clears acc; returns running sum */
    float (*accumulate)(float* acc, uint8_t* mask, int n, float carry, int evenodd);
    /* bilinear fetch, all taps in bounds. u,v 16.16 of the top-left tap. */
    void (*bilinear)(const uint32_t* tex, int stride_px, int32_t u, int32_t v, int32_t du, int32_t dv, int n,
                     uint32_t* out);
    /* linear gradient: t = t0 + i*dt, extend 0 pad / 1 repeat / 2 reflect */
    void (*linear_grad)(const uint32_t* lut, float t0, float dt, int n, int extend, uint32_t* out);
} fm_kernels;

extern const fm_kernels* fm_k;
extern const fm_kernels  fm_kernels_scalar;
#ifdef FM_HAVE_SSE2
extern const fm_kernels fm_kernels_sse2;
#endif
#ifdef FM_HAVE_AVX2
extern const fm_kernels fm_kernels_avx2;
#endif
#ifdef FM_HAVE_NEON
extern const fm_kernels fm_kernels_neon;
#endif

/* Scalar float implementation of the non-trivial blend modes (overlay,
 * dodge/burn, hard/soft light, hue/saturation/color/luminosity). Shared by
 * all backends so results stay identical. m may be NULL. */
void fm_span_op_complex(uint32_t* d, const uint32_t* s, const uint8_t* m, int n, int op);
int  fm_op_is_complex(int op);

#endif /* FATMAP_FM_INTERNAL_H */
