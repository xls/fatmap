/* fatmap - scalar reference backend (2 pixel "vectors" of u16 lanes) */
#include "fm_kernels_common.h"

#define FMK(name)  name##_scalar
#define FMK_TABLE  fm_kernels_scalar
#define FMK_LEVEL  FM_SIMD_SCALAR
#define FMK_PX     2

typedef struct { uint32_t p[2]; } vpx;
typedef struct { uint16_t c[4]; } vw;

FM_INLINE vpx vpx_load(const void* p) { vpx r; memcpy(r.p, p, 8); return r; }
FM_INLINE void vpx_store(void* p, vpx v) { memcpy(p, v.p, 8); }
FM_INLINE vpx vpx_set1(uint32_t v) { vpx r; r.p[0] = v; r.p[1] = v; return r; }

FM_INLINE vw fmk_unpack1(uint32_t p)
{
    vw r;
    r.c[0] = (uint16_t)(p & 255);
    r.c[1] = (uint16_t)((p >> 8) & 255);
    r.c[2] = (uint16_t)((p >> 16) & 255);
    r.c[3] = (uint16_t)(p >> 24);
    return r;
}
/* emulate SSE2 packus_epi16 (signed saturation to u8) */
FM_INLINE uint32_t fmk_sat(uint16_t v) { return v > 32767 ? 0u : (v > 255 ? 255u : v); }
FM_INLINE uint32_t fmk_pack1(vw w)
{
    return fmk_sat(w.c[0]) | (fmk_sat(w.c[1]) << 8) | (fmk_sat(w.c[2]) << 16) | (fmk_sat(w.c[3]) << 24);
}
FM_INLINE vw  vw_lo(vpx v) { return fmk_unpack1(v.p[0]); }
FM_INLINE vw  vw_hi(vpx v) { return fmk_unpack1(v.p[1]); }
FM_INLINE vpx vw_pack(vw lo, vw hi) { vpx r; r.p[0] = fmk_pack1(lo); r.p[1] = fmk_pack1(hi); return r; }
FM_INLINE vpx vpx_mask_load(const uint8_t* m)
{
    vpx r;
    r.p[0] = m[0] * 0x01010101u;
    r.p[1] = m[1] * 0x01010101u;
    return r;
}

#define FMK_LANES(expr)                                     \
    vw r;                                                   \
    for (int k = 0; k < 4; k++) r.c[k] = (uint16_t)(expr); \
    return r

FM_INLINE vw vw_set1(uint16_t v) { FMK_LANES(v); }
FM_INLINE vw vw_add(vw a, vw b) { FMK_LANES(a.c[k] + b.c[k]); }
FM_INLINE vw vw_sub(vw a, vw b) { FMK_LANES(a.c[k] - b.c[k]); }
FM_INLINE vw vw_mul(vw a, vw b) { FMK_LANES((uint32_t)a.c[k] * b.c[k]); }
FM_INLINE vw vw_min(vw a, vw b) { FMK_LANES(a.c[k] < b.c[k] ? a.c[k] : b.c[k]); }
FM_INLINE vw vw_max(vw a, vw b) { FMK_LANES(a.c[k] > b.c[k] ? a.c[k] : b.c[k]); }
FM_INLINE vw vw_div255(vw a)
{
    vw r;
    for (int k = 0; k < 4; k++) {
        uint16_t t = (uint16_t)(a.c[k] + 128);
        r.c[k]     = (uint16_t)((uint16_t)(t + (t >> 8)) >> 8);
    }
    return r;
}
FM_INLINE vw vw_alpha(vw a) { FMK_LANES(a.c[3]); }
FM_INLINE vw vw_alpha_merge(vw c, vw a) { c.c[3] = a.c[3]; return c; }

FM_INLINE int fmk_mask_zero(const uint8_t* m) { return (m[0] | m[1]) == 0; }
FM_INLINE int fmk_mask_full(const uint8_t* m) { return (m[0] & m[1]) == 255; }
FM_INLINE int vpx_all_opaque(vpx v) { return (v.p[0] >> 24) == 255 && (v.p[1] >> 24) == 255; }
FM_INLINE int vpx_all_zero(vpx v) { return (v.p[0] | v.p[1]) == 0; }

/* ---- accumulate: emulates the 4-wide SIMD prefix sum order exactly ---- */
static float accumulate_scalar(float* acc, uint8_t* mask, int n, float carry, int evenodd)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float a0 = acc[i], a1 = acc[i + 1], a2 = acc[i + 2], a3 = acc[i + 3];
        float b0 = a0, b1 = a1 + a0, b2 = a2 + a1, b3 = a3 + a2;
        float c0 = b0, c1 = b1, c2 = b2 + b0, c3 = b3 + b1;
        float s0 = c0 + carry, s1 = c1 + carry, s2 = c2 + carry, s3 = c3 + carry;
        carry    = s3;
        acc[i] = acc[i + 1] = acc[i + 2] = acc[i + 3] = 0.0f;
        mask[i]     = fm_cov_u8(s0, evenodd);
        mask[i + 1] = fm_cov_u8(s1, evenodd);
        mask[i + 2] = fm_cov_u8(s2, evenodd);
        mask[i + 3] = fm_cov_u8(s3, evenodd);
    }
    for (; i < n; i++) {
        carry  = carry + acc[i];
        acc[i] = 0.0f;
        mask[i] = fm_cov_u8(carry, evenodd);
    }
    return carry;
}

static void bilinear_scalar(const uint32_t* tex, int stride, int32_t u, int32_t v, int32_t du, int32_t dv, int n,
                            uint32_t* out)
{
    for (int i = 0; i < n; i++) {
        const uint32_t* r0 = tex + (ptrdiff_t)(v >> 16) * stride + (u >> 16);
        out[i] = fm_bilerp(r0[0], r0[1], r0[stride], r0[stride + 1], (uint32_t)(u >> 8) & 255u,
                           (uint32_t)(v >> 8) & 255u);
        u += du;
        v += dv;
    }
}

static void linear_grad_scalar(const uint32_t* lut, float t0, float dt, int n, int extend, uint32_t* out)
{
    for (int i = 0; i < n; i++) out[i] = lut[fm_grad_index(t0 + (float)i * dt, extend)];
}

#include "fm_kernels_tmpl.h"
