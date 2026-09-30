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
