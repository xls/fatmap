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
    if (w == 0) return u - fm_ffloor(u);
    if (w == 2) return u - 2.0f * fm_ffloor(u * 0.5f);
    return u > -1.0f ? (u < 2.0f ? u : 2.0f) : -1.0f;
}

/* rasterizer interpolation (reference semantics for every backend) */
FM_INLINE float fm_plane1(float a, float b, float dx) { return a + b * dx; }

FM_INLINE int32_t fm_texcoord1(float u, int wrap, float size, int bilinear)
{
    float f = fm_wrap_norm_f(u, wrap) * size;
    f       = f > -32767.0f ? (f < 32767.0f ? f : 32767.0f) : -32767.0f;
    int32_t q = (int32_t)fm_ffloor(f * 65536.0f + 0.5f);
    return bilinear ? q - 32768 : q;
}

FM_INLINE uint32_t fm_straight_f1(float r, float g, float b, float a)
{
    uint32_t A = (uint32_t)(fm_clamp01(a) * 255.0f + 0.5f);
    uint32_t R = (uint32_t)(fm_clamp01(r) * 255.0f + 0.5f);
    uint32_t G = (uint32_t)(fm_clamp01(g) * 255.0f + 0.5f);
    uint32_t B = (uint32_t)(fm_clamp01(b) * 255.0f + 0.5f);
    return (A << 24) | (R << 16) | (G << 8) | B;
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

/* reference vertex blend (the SIMD versions use the same per lane order) */
FM_INLINE void fm_skin1(const float* bones, int nbones, const uint8_t* rec, const float* p, float* o)
{
    uint16_t j[4];
    float    w[4], m[16];
    memcpy(j, rec, sizeof(j));
    memcpy(w, rec + 8, sizeof(w));
    for (int k = 0; k < 4; k++) {
        const float* b = bones + 16 * (j[k] < nbones ? j[k] : 0);
        for (int e = 0; e < 16; e++) m[e] = k ? m[e] + w[k] * b[e] : w[k] * b[e];
    }
    for (int r = 0; r < 3; r++) o[r] = ((m[r] * p[0] + m[4 + r] * p[1]) + m[8 + r] * p[2]) + m[12 + r];
}

FM_INLINE void fm_depth_ms1(const float* zc, const float* dzs, int S, float* zb, uint8_t* smask, int i, int func,
                            int write)
{
    uint8_t bits = smask[i];
    if (!bits) return;
    float* d = zb + (size_t)i * (size_t)S;
    for (int s = 0; s < S; s++) {
        if (!(bits & (1u << s))) continue;
        float z = fm_clamp01(zc[i] + dzs[s]);
        if (!fm_fcmp(func, z, d[s]))
            bits = (uint8_t)(bits & ~(1u << s));
        else if (write)
            d[s] = z;
    }
    smask[i] = bits;
}

FM_INLINE uint32_t fm_resolve1(const uint32_t* s, int S)
{
    int      sh = S == 8 ? 3 : (S == 4 ? 2 : (S == 2 ? 1 : 0));
    uint32_t o  = 0;
    for (int c = 0; c < 32; c += 8) {
        uint32_t sum = 0;
        for (int k = 0; k < S; k++) sum += (s[k] >> c) & 255;
        o |= ((sum + (uint32_t)(S >> 1)) >> sh) << c;
    }
    return o;
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

/* one bilinear sample of a power of two repeat texture (reference) */
FM_INLINE uint32_t fm_bilerp_wrap1(const uint32_t* tex, int stride, int32_t u, int32_t v, int wm, int hm)
{
    int             x0 = (u >> 16) & wm, x1 = (x0 + 1) & wm;
    int             y0 = (v >> 16) & hm, y1 = (y0 + 1) & hm;
    const uint32_t* r0 = tex + (ptrdiff_t)y0 * stride;
    const uint32_t* r1 = tex + (ptrdiff_t)y1 * stride;
    return fm_bilerp(r0[x0], r0[x1], r1[x0], r1[x1], (uint32_t)(u >> 8) & 255u, (uint32_t)(v >> 8) & 255u);
}

#endif
