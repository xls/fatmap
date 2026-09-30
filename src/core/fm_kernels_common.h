/* fatmap - scalar helpers shared by kernel backends (tails must match SIMD) */
#ifndef FATMAP_FM_KERNELS_COMMON_H
#define FATMAP_FM_KERNELS_COMMON_H

#include "fm_internal.h"
#include <math.h>

/* coverage accumulator value -> 0..255 (matches the SIMD lane math) */
FM_INLINE uint8_t fm_cov_u8(float s, int evenodd)
{
    float c = fabsf(s);
    if (evenodd) {
        float h = c * 0.5f;
        float t = (float)(int)h;
        c       = c - t * 2.0f;
        c       = fminf(c, 2.0f - c);
    } else {
        c = fminf(c, 1.0f);
    }
    return (uint8_t)(int)(c * 255.0f + 0.5f);
}

/* gradient parameter -> LUT index (matches the SIMD lane math) */
FM_INLINE int fm_grad_index(float t, int extend)
{
    if (extend == 0) {
        t = fmaxf(t, 0.0f);
        t = fminf(t, 1.0f);
    } else {
        t = fmaxf(t, -1e6f);
        t = fminf(t, 1e6f);
        if (extend == 1) {
            t = t - floorf(t);
        } else {
            t = t * 0.5f;
            t = t - floorf(t);
            t = t * 2.0f;
            t = fminf(t, 2.0f - t);
        }
    }
    return (int)(t * (float)(FM_GRAD_LUT_SIZE - 1) + 0.5f);
}

/* ---- 3D fragment op reference semantics (SIMD versions must match) ---- */

FM_INLINE int fm_fcmp(int func, float a, float b) /* fm3d_compare order */
{
    switch (func) {
    case 0: return 0;
    case 1: return a < b;
    case 2: return a == b;
    case 3: return a <= b;
    case 4: return a > b;
    case 5: return a != b;
    case 6: return a >= b;
    default: return 1;
    }
}

FM_INLINE float fm_wrap_norm_f(float u, int w) /* fm_wrap: 0 repeat, 1 clamp, 2 mirror, 3 border */
{
    u = u > -1e6f ? (u < 1e6f ? u : 1e6f) : -1e6f;
    if (w == 0) return u - fm_floorf(u);
    if (w == 2) return u - 2.0f * fm_floorf(u * 0.5f);
    return u > -1.0f ? (u < 2.0f ? u : 2.0f) : -1.0f;
}

FM_INLINE int32_t fm_texcoord1(float u, int wrap, float size, int bilinear)
{
    float f = fm_wrap_norm_f(u, wrap) * size;
    f       = f > -32767.0f ? (f < 32767.0f ? f : 32767.0f) : -32767.0f;
    int32_t q = (int32_t)fm_floorf(f * 65536.0f + 0.5f);
    return bilinear ? q - 32768 : q;
}

FM_INLINE uint32_t fm_premul_f1(float r, float g, float b, float a)
{
    a          = fm_clamp01(a);
    uint32_t A = (uint32_t)(a * 255.0f + 0.5f);
    uint32_t R = (uint32_t)(fm_clamp01(r) * a * 255.0f + 0.5f);
    uint32_t G = (uint32_t)(fm_clamp01(g) * a * 255.0f + 0.5f);
    uint32_t B = (uint32_t)(fm_clamp01(b) * a * 255.0f + 0.5f);
    R = R < A ? R : A;
    G = G < A ? G : A;
    B = B < A ? B : A;
    return (A << 24) | (R << 16) | (G << 8) | B;
}

/* one bilinear texel, 8-bit weights, two stage lerp (matches SIMD) */
FM_INLINE uint32_t fm_bilerp(uint32_t p00, uint32_t p01, uint32_t p10, uint32_t p11, uint32_t fx, uint32_t fy)
{
    uint32_t r = 0;
    for (int sh = 0; sh < 32; sh += 8) {
        uint32_t t = ((((p00 >> sh) & 255) * (256 - fx) + ((p01 >> sh) & 255) * fx) >> 8);
        uint32_t b = ((((p10 >> sh) & 255) * (256 - fx) + ((p11 >> sh) & 255) * fx) >> 8);
        r |= ((t * (256 - fy) + b * fy) >> 8) << sh;
    }
    return r;
}

#endif
