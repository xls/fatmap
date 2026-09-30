/* fatmap - SSE2 helpers shared by the SSE2 and AVX2 backends.
 * Include after defining FMK(). */
#ifndef FATMAP_FM_KERNELS_X86_H
#define FATMAP_FM_KERNELS_X86_H

#include "fm_kernels_common.h"
#include <emmintrin.h>

/* coverage of 4 accumulator sums (matches fm_cov_u8) */
FM_INLINE __m128i fmx_cov4(__m128 x, int evenodd)
{
    const __m128 absm = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    __m128       c    = _mm_and_ps(x, absm);
    if (evenodd) {
        __m128 two = _mm_set1_ps(2.0f);
        __m128 t   = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(c, _mm_set1_ps(0.5f))));
        c          = _mm_sub_ps(c, _mm_mul_ps(t, two));
        c          = _mm_min_ps(c, _mm_sub_ps(two, c));
    } else {
        c = _mm_min_ps(c, _mm_set1_ps(1.0f));
    }
    return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(c, _mm_set1_ps(255.0f)), _mm_set1_ps(0.5f)));
}

static float FMK(accumulate)(float* acc, uint8_t* mask, int n, float carry, int evenodd)
{
    __m128       cv   = _mm_set1_ps(carry);
    const __m128 zero = _mm_setzero_ps();
    int          i    = 0;
    for (; i + 8 <= n; i += 8) {
        __m128 x = _mm_loadu_ps(acc + i);
        __m128 y = _mm_loadu_ps(acc + i + 4);
        x        = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        y        = _mm_add_ps(y, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(y), 4)));
        x        = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
        y        = _mm_add_ps(y, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(y), 8)));
        x        = _mm_add_ps(x, cv);
        cv       = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
        y        = _mm_add_ps(y, cv);
        cv       = _mm_shuffle_ps(y, y, _MM_SHUFFLE(3, 3, 3, 3));
        _mm_storeu_ps(acc + i, zero);
        _mm_storeu_ps(acc + i + 4, zero);
        __m128i a = fmx_cov4(x, evenodd), b = fmx_cov4(y, evenodd);
        __m128i p = _mm_packus_epi16(_mm_packs_epi32(a, b), _mm_setzero_si128());
        _mm_storel_epi64((__m128i*)(mask + i), p);
    }
    for (; i + 4 <= n; i += 4) {
        __m128 x = _mm_loadu_ps(acc + i);
        x        = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        x        = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
        x        = _mm_add_ps(x, cv);
        cv       = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
        _mm_storeu_ps(acc + i, zero);
        __m128i  a = fmx_cov4(x, evenodd);
        __m128i  p = _mm_packus_epi16(_mm_packs_epi32(a, a), a);
        uint32_t w = (uint32_t)_mm_cvtsi128_si32(p);
        memcpy(mask + i, &w, 4);
    }
    carry = _mm_cvtss_f32(cv);
    for (; i < n; i++) {
        carry   = carry + acc[i];
        acc[i]  = 0.0f;
        mask[i] = fm_cov_u8(carry, evenodd);
    }
    return carry;
}

static void FMK(bilinear)(const uint32_t* tex, int stride, int32_t u, int32_t v, int32_t du, int32_t dv, int n,
                          uint32_t* out)
{
    const __m128i zero = _mm_setzero_si128();
    for (int i = 0; i < n; i++) {
        const uint32_t* r0 = tex + (ptrdiff_t)(v >> 16) * stride + (u >> 16);
        short           fx = (short)((u >> 8) & 255), fy = (short)((v >> 8) & 255);
        short           ix = (short)(256 - fx), iy = (short)(256 - fy);
        __m128i         wx = _mm_set_epi16(fx, fx, fx, fx, ix, ix, ix, ix);
        __m128i         wy = _mm_set_epi16(fy, fy, fy, fy, iy, iy, iy, iy);
        __m128i         t  = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)r0), zero);
        __m128i         b  = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(r0 + stride)), zero);
        t                  = _mm_mullo_epi16(t, wx);
        b                  = _mm_mullo_epi16(b, wx);
        t                  = _mm_srli_epi16(_mm_add_epi16(t, _mm_srli_si128(t, 8)), 8);
        b                  = _mm_srli_epi16(_mm_add_epi16(b, _mm_srli_si128(b, 8)), 8);
        __m128i tb         = _mm_mullo_epi16(_mm_unpacklo_epi64(t, b), wy);
        tb                 = _mm_srli_epi16(_mm_add_epi16(tb, _mm_srli_si128(tb, 8)), 8);
        out[i]             = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(tb, tb));
        u += du;
        v += dv;
    }
}

/* one bilinear texel, two stage 8-bit lerp (matches fm_bilerp) */
FM_INLINE uint32_t fmx_bilerp1(const uint32_t* r0, int stride, int32_t u, int32_t v)
{
    const __m128i zero = _mm_setzero_si128();
    short         fx = (short)((u >> 8) & 255), fy = (short)((v >> 8) & 255);
    short         ix = (short)(256 - fx), iy = (short)(256 - fy);
    __m128i       wx = _mm_set_epi16(fx, fx, fx, fx, ix, ix, ix, ix);
    __m128i       wy = _mm_set_epi16(fy, fy, fy, fy, iy, iy, iy, iy);
    __m128i       t  = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)r0), zero);
    __m128i       b  = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(r0 + stride)), zero);
    t                = _mm_mullo_epi16(t, wx);
    b                = _mm_mullo_epi16(b, wx);
    t                = _mm_srli_epi16(_mm_add_epi16(t, _mm_srli_si128(t, 8)), 8);
    b                = _mm_srli_epi16(_mm_add_epi16(b, _mm_srli_si128(b, 8)), 8);
    __m128i tb       = _mm_mullo_epi16(_mm_unpacklo_epi64(t, b), wy);
    tb               = _mm_srli_epi16(_mm_add_epi16(tb, _mm_srli_si128(tb, 8)), 8);
    return (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(tb, tb));
}

static void FMK(bilinear_pts)(const uint32_t* tex, int stride, const int32_t* U, const int32_t* V, int n,
                              uint32_t* out)
{
    for (int i = 0; i < n; i++)
        out[i] = fmx_bilerp1(tex + (ptrdiff_t)(V[i] >> 16) * stride + (U[i] >> 16), stride, U[i], V[i]);
}

FM_INLINE __m128 fmx_floor(__m128 t)
{
    __m128 f = _mm_cvtepi32_ps(_mm_cvttps_epi32(t));
    return _mm_sub_ps(f, _mm_and_ps(_mm_cmpgt_ps(f, t), _mm_set1_ps(1.0f)));
}

static void FMK(linear_grad)(const uint32_t* lut, float t0, float dt, int n, int extend, uint32_t* out)
{
    const __m128 vt0  = _mm_set1_ps(t0), vdt = _mm_set1_ps(dt);
    const __m128 one  = _mm_set1_ps(1.0f), zero = _mm_setzero_ps(), two = _mm_set1_ps(2.0f);
    const __m128 lutm = _mm_set1_ps((float)(FM_GRAD_LUT_SIZE - 1)), half = _mm_set1_ps(0.5f);
    __m128i      vi   = _mm_set_epi32(3, 2, 1, 0);
    const __m128i four = _mm_set1_epi32(4);
    int          i    = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 t = _mm_add_ps(vt0, _mm_mul_ps(_mm_cvtepi32_ps(vi), vdt));
        vi       = _mm_add_epi32(vi, four);
        if (extend == 0) {
            t = _mm_min_ps(_mm_max_ps(t, zero), one);
        } else {
            t = _mm_min_ps(_mm_max_ps(t, _mm_set1_ps(-1e6f)), _mm_set1_ps(1e6f));
            if (extend == 1) {
                t = _mm_sub_ps(t, fmx_floor(t));
            } else {
                t = _mm_mul_ps(t, half);
                t = _mm_sub_ps(t, fmx_floor(t));
                t = _mm_mul_ps(t, two);
                t = _mm_min_ps(t, _mm_sub_ps(two, t));
            }
        }
        __m128i idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(t, lutm), half));
        int32_t ix[4];
        _mm_storeu_si128((__m128i*)ix, idx);
        out[i]     = lut[ix[0]];
        out[i + 1] = lut[ix[1]];
        out[i + 2] = lut[ix[2]];
        out[i + 3] = lut[ix[3]];
    }
    for (; i < n; i++) out[i] = lut[fm_grad_index(t0 + (float)i * dt, extend)];
}

#endif
