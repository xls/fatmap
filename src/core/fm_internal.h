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

/* cache prefetch hints (no-ops where unsupported); _W = about to write */
#if defined(_MSC_VER) && FM_ARCH_X86
#  include <intrin.h>
#  define FM_PREFETCH(p)   _mm_prefetch((const char*)(p), _MM_HINT_T0)
#  define FM_PREFETCH_W(p) _m_prefetchw((const void*)(p))
#elif defined(__GNUC__) || defined(__clang__)
#  define FM_PREFETCH(p)   __builtin_prefetch((p), 0, 3)
#  define FM_PREFETCH_W(p) __builtin_prefetch((p), 1, 3)
#else
#  define FM_PREFETCH(p)   ((void)(p))
#  define FM_PREFETCH_W(p) ((void)(p))
#endif

#define FM_MIN(a, b) ((a) < (b) ? (a) : (b))
#define FM_MAX(a, b) ((a) > (b) ? (a) : (b))
#define FM_CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

/* clamp to [0, 1] with SIMD min/max semantics (NaN and -0 -> 0) */
static inline float fm_clamp01(float x) { return x > 0.0f ? (x < 1.0f ? x : 1.0f) : 0.0f; }

/* floorf() without the libm call; exact for |x| < 2^31 */
static inline float fm_floorf(float x)
{
    float f = (float)(int)x;
    return f > x ? f - 1.0f : f;
}

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
    /* bilinear fetch at independent points, all taps in bounds; U, V are
     * 16.16 top-left tap coordinates (sample position - 0.5) */
    void (*bilinear_pts)(const uint32_t* tex, int stride_px, const int32_t* U, const int32_t* V, int n,
                         uint32_t* out);
    /* ---- 3D fragment ops (bit-identical across backends) ---- */
    /* D32F depth test / write for active (m != 0) pixels: clamp z to [0, 1],
     * compare with func (fm3d_compare values), clear m on failure, write on
     * pass; extends [*wmin, *wmax] and *nw with the written values */
    void (*depth_f32)(const float* z, float* zb, uint8_t* m, int n, int func, int write, float* wmin, float* wmax,
                      int* nw);
    /* vertex blending (skinning): per vertex M = sum_k w_k * bones[j_k]
     * (column major mat4s), out = M * (x, y, z, 1).xyz. skin records hold
     * uint16 joint[4] at offset 0 and float weight[4] at offset 8. */
    void (*skin4)(const float* bones, int nbones, const void* skin, int skin_stride, const float* pos, int pos_stride,
                  int n, float* out3);
    /* MSAA depth: per pixel S samples (4 or 8) at zc[i] + dzs[s], clamped;
     * sample bits in smask are cleared on failure, depth written on pass */
    void (*depth_ms)(const float* zc, const float* dzs, int S, float* zb, uint8_t* smask, int n, int func, int write);
    /* MSAA resolve: average S consecutive samples per pixel (rounded) */
    void (*resolve)(const uint32_t* samples, int S, int n, uint32_t* out);
    /* texture coordinate op: wrap (fm_wrap) normalized u, scale by size,
     * convert to 16.16 fixed; bilinear subtracts half a texel */
    void (*texcoord)(const float* u, int n, int wrap, float size, int bilinear, int32_t* out);
    /* straight float color (0..1) -> premultiplied ARGB32 */
    void (*premul_f)(const float* r, const float* g, const float* b, const float* a, int n, uint32_t* out);
    /* texenv combine (fm3d_texenv values) of texel t with color c */
    void (*combine)(int env, const uint32_t* t, const uint32_t* c, int n, uint32_t* out);
    /* per pixel lerp with 8 bit weights: (a * (256 - f) + b * f) >> 8 */
    void (*lerp8)(const uint32_t* a, const uint32_t* b, const uint8_t* f, int n, uint32_t* out);
    /* linear gradient: t = t0 + i*dt, extend 0 pad / 1 repeat / 2 reflect */
    void (*linear_grad)(const uint32_t* lut, float t0, float dt, int n, int extend, uint32_t* out);
    /* ---- rasterizer ---- */
    /* 2D coverage: acc[i] += v (long interior runs of an edge) */
    void (*acc_add)(float* acc, float v, int n);
    /* plane equation along a row: out = a + b * dx[i] (mul, then add; no
     * FMA), clamped to [0, 1] with fm_clamp01 semantics when clamp01 */
    void (*plane)(float a, float b, const float* dx, int n, int clamp01, float* out);
    /* perspective weight: out = 1 / (a + b * dx[i]) (IEEE division) */
    void (*plane_recip)(float a, float b, const float* dx, int n, float* out);
    /* perspective correct varying: out = (a + b * dx[i]) * w[i] */
    void (*plane_mul)(float a, float b, const float* dx, const float* w, int n, float* out);
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
struct fm_sampler;
void fm__sample_fixed(const fm_surface* tex, const struct fm_sampler* s, const int32_t* U, const int32_t* V, int n,
                      uint32_t* out);

void fm_span_op_complex(uint32_t* d, const uint32_t* s, const uint8_t* m, int n, int op);
int  fm_op_is_complex(int op);

#endif /* FATMAP_FM_INTERNAL_H */
