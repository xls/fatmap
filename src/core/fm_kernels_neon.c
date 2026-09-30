/* fatmap - NEON backend (AArch64, 4 pixels per block) */
#include "fm_internal.h"

#if FM_ARCH_ARM64

#include "fm_kernels_common.h"
#include <arm_neon.h>

#define FMK(name) name##_neon
#define FMK_TABLE fm_kernels_neon
#define FMK_LEVEL FM_SIMD_NEON
#define FMK_PX    4

typedef uint8x16_t vpx;
typedef uint16x8_t vw;

FM_INLINE vpx  vpx_load(const void* p) { return vld1q_u8((const uint8_t*)p); }
FM_INLINE void vpx_store(void* p, vpx v) { vst1q_u8((uint8_t*)p, v); }
FM_INLINE vpx  vpx_set1(uint32_t v) { return vreinterpretq_u8_u32(vdupq_n_u32(v)); }
FM_INLINE vw   vw_lo(vpx p) { return vmovl_u8(vget_low_u8(p)); }
FM_INLINE vw   vw_hi(vpx p) { return vmovl_u8(vget_high_u8(p)); }
FM_INLINE vpx  vw_pack(vw lo, vw hi) { return vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)); }
FM_INLINE vpx  vpx_mask_load(const uint8_t* m)
{
    static const uint8_t idx[16] = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
    uint32_t             w;
    memcpy(&w, m, 4);
    uint8x16_t x = vreinterpretq_u8_u32(vdupq_n_u32(w));
    return vqtbl1q_u8(x, vld1q_u8(idx));
}
FM_INLINE vw vw_set1(uint16_t v) { return vdupq_n_u16(v); }
FM_INLINE vw vw_add(vw a, vw b) { return vaddq_u16(a, b); }
FM_INLINE vw vw_sub(vw a, vw b) { return vsubq_u16(a, b); }
FM_INLINE vw vw_mul(vw a, vw b) { return vmulq_u16(a, b); }
FM_INLINE vw vw_min(vw a, vw b) { return vminq_u16(a, b); }
FM_INLINE vw vw_max(vw a, vw b) { return vmaxq_u16(a, b); }
FM_INLINE vw vw_div255(vw a)
{
    uint16x8_t t = vaddq_u16(a, vdupq_n_u16(128));
    return vshrq_n_u16(vsraq_n_u16(t, t, 8), 8);
}
FM_INLINE vw vw_alpha(vw a)
{
    static const uint8_t idx[16] = { 6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15 };
    return vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(a), vld1q_u8(idx)));
}
FM_INLINE vw vw_alpha_merge(vw c, vw a)
{
    static const uint16_t am[8] = { 0, 0, 0, 0xffff, 0, 0, 0, 0xffff };
    return vbslq_u16(vld1q_u16(am), a, c);
}
FM_INLINE int fmk_mask_zero(const uint8_t* m)
{
    uint32_t w;
    memcpy(&w, m, 4);
    return w == 0;
}
FM_INLINE int fmk_mask_full(const uint8_t* m)
{
    uint32_t w;
    memcpy(&w, m, 4);
    return w == 0xffffffffu;
}
FM_INLINE int vpx_all_opaque(vpx v)
{
    uint32x4_t a = vshrq_n_u32(vreinterpretq_u32_u8(v), 24);
    return vminvq_u32(a) == 255;
}
FM_INLINE int vpx_all_zero(vpx v) { return vmaxvq_u32(vreinterpretq_u32_u8(v)) == 0; }

FM_INLINE uint32x4_t fmn_cov4(float32x4_t x, int evenodd)
{
    float32x4_t c = vabsq_f32(x);
    if (evenodd) {
        float32x4_t two = vdupq_n_f32(2.0f);
        float32x4_t t   = vcvtq_f32_s32(vcvtq_s32_f32(vmulq_f32(c, vdupq_n_f32(0.5f))));
        c               = vsubq_f32(c, vmulq_f32(t, two));
        c               = vminq_f32(c, vsubq_f32(two, c));
    } else {
        c = vminq_f32(c, vdupq_n_f32(1.0f));
    }
    return vreinterpretq_u32_s32(vcvtq_s32_f32(vaddq_f32(vmulq_f32(c, vdupq_n_f32(255.0f)), vdupq_n_f32(0.5f))));
}

static float FMK(accumulate)(float* acc, uint8_t* mask, int n, float carry, int evenodd)
{
    float32x4_t       cv   = vdupq_n_f32(carry);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int               i    = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(acc + i);
        x             = vaddq_f32(x, vextq_f32(zero, x, 3));
        x             = vaddq_f32(x, vextq_f32(zero, x, 2));
        x             = vaddq_f32(x, cv);
        cv            = vdupq_laneq_f32(x, 3);
        vst1q_f32(acc + i, zero);
        uint16x4_t h = vmovn_u32(fmn_cov4(x, evenodd));
        uint8x8_t  b = vmovn_u16(vcombine_u16(h, h));
        uint32_t   w = vget_lane_u32(vreinterpret_u32_u8(b), 0);
        memcpy(mask + i, &w, 4);
    }
    carry = vgetq_lane_f32(cv, 0);
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
    for (int i = 0; i < n; i++) {
        const uint32_t* r0 = tex + (ptrdiff_t)(v >> 16) * stride + (u >> 16);
        uint16_t        fx = (uint16_t)((u >> 8) & 255), fy = (uint16_t)((v >> 8) & 255);
        uint16x4_t      ix = vdup_n_u16((uint16_t)(256 - fx)), vx = vdup_n_u16(fx);
        uint16x4_t      iy = vdup_n_u16((uint16_t)(256 - fy)), vy = vdup_n_u16(fy);
        uint16x8_t      t  = vmovl_u8(vld1_u8((const uint8_t*)r0));
        uint16x8_t      b  = vmovl_u8(vld1_u8((const uint8_t*)(r0 + stride)));
        uint16x4_t      th = vshr_n_u16(vadd_u16(vmul_u16(vget_low_u16(t), ix), vmul_u16(vget_high_u16(t), vx)), 8);
        uint16x4_t      bh = vshr_n_u16(vadd_u16(vmul_u16(vget_low_u16(b), ix), vmul_u16(vget_high_u16(b), vx)), 8);
        uint16x4_t      r  = vshr_n_u16(vadd_u16(vmul_u16(th, iy), vmul_u16(bh, vy)), 8);
        uint8x8_t       p  = vmovn_u16(vcombine_u16(r, r));
        out[i]             = vget_lane_u32(vreinterpret_u32_u8(p), 0);
        u += du;
        v += dv;
    }
}

static void FMK(linear_grad)(const uint32_t* lut, float t0, float dt, int n, int extend, uint32_t* out)
{
    static const int32_t base[4] = { 0, 1, 2, 3 };
    const float32x4_t    vt0 = vdupq_n_f32(t0), vdt = vdupq_n_f32(dt);
    const float32x4_t    one = vdupq_n_f32(1.0f), zero = vdupq_n_f32(0.0f), two = vdupq_n_f32(2.0f);
    const float32x4_t    lutm = vdupq_n_f32((float)(FM_GRAD_LUT_SIZE - 1)), half = vdupq_n_f32(0.5f);
    int32x4_t            vi = vld1q_s32(base);
    int                  i  = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t t = vaddq_f32(vt0, vmulq_f32(vcvtq_f32_s32(vi), vdt));
        vi            = vaddq_s32(vi, vdupq_n_s32(4));
        if (extend == 0) {
            t = vminq_f32(vmaxq_f32(t, zero), one);
        } else {
            t = vminq_f32(vmaxq_f32(t, vdupq_n_f32(-1e6f)), vdupq_n_f32(1e6f));
            if (extend == 1) {
                t = vsubq_f32(t, vrndmq_f32(t));
            } else {
                t = vmulq_f32(t, half);
                t = vsubq_f32(t, vrndmq_f32(t));
                t = vmulq_f32(t, two);
                t = vminq_f32(t, vsubq_f32(two, t));
            }
        }
        int32_t ix[4];
        vst1q_s32(ix, vcvtq_s32_f32(vaddq_f32(vmulq_f32(t, lutm), half)));
        out[i]     = lut[ix[0]];
        out[i + 1] = lut[ix[1]];
        out[i + 2] = lut[ix[2]];
        out[i + 3] = lut[ix[3]];
    }
    for (; i < n; i++) out[i] = lut[fm_grad_index(t0 + (float)i * dt, extend)];
}

#include "fm_kernels_tmpl.h"

#endif
