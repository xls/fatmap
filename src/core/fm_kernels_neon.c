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
FM_INLINE vw vw_shr8(vw a) { return vshrq_n_u16(a, 8); }
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

static void FMK(bilinear_pts)(const uint32_t* tex, int stride, const int32_t* U, const int32_t* V, int n,
                              uint32_t* out)
{
    for (int i = 0; i < n; i++) {
        const uint32_t* r0 = tex + (ptrdiff_t)(V[i] >> 16) * stride + (U[i] >> 16);
        uint16_t        fx = (uint16_t)((U[i] >> 8) & 255), fy = (uint16_t)((V[i] >> 8) & 255);
        uint16x4_t      ix = vdup_n_u16((uint16_t)(256 - fx)), vx = vdup_n_u16(fx);
        uint16x4_t      iy = vdup_n_u16((uint16_t)(256 - fy)), vy = vdup_n_u16(fy);
        uint16x8_t      t  = vmovl_u8(vld1_u8((const uint8_t*)r0));
        uint16x8_t      b  = vmovl_u8(vld1_u8((const uint8_t*)(r0 + stride)));
        uint16x4_t      th = vshr_n_u16(vadd_u16(vmul_u16(vget_low_u16(t), ix), vmul_u16(vget_high_u16(t), vx)), 8);
        uint16x4_t      bh = vshr_n_u16(vadd_u16(vmul_u16(vget_low_u16(b), ix), vmul_u16(vget_high_u16(b), vx)), 8);
        uint16x4_t      r  = vshr_n_u16(vadd_u16(vmul_u16(th, iy), vmul_u16(bh, vy)), 8);
        uint8x8_t       p  = vmovn_u16(vcombine_u16(r, r));
        out[i]             = vget_lane_u32(vreinterpret_u32_u8(p), 0);
    }
}

FM_INLINE uint32x4_t fmn_fcmp(int func, float32x4_t a, float32x4_t b)
{
    switch (func) {
    case 0: return vdupq_n_u32(0);
    case 1: return vcltq_f32(a, b);
    case 2: return vceqq_f32(a, b);
    case 3: return vcleq_f32(a, b);
    case 4: return vcgtq_f32(a, b);
    case 5: return vmvnq_u32(vceqq_f32(a, b));
    case 6: return vcgeq_f32(a, b);
    default: return vdupq_n_u32(0xffffffffu);
    }
}

/* NEON max/min return NaN for NaN inputs; use compare + select to match
 * the scalar "x > lo ? x : lo" semantics exactly */
FM_INLINE float32x4_t fmn_clamp(float32x4_t x, float lo, float hi)
{
    float32x4_t l = vdupq_n_f32(lo), h = vdupq_n_f32(hi);
    x             = vbslq_f32(vcgtq_f32(x, l), x, l);
    return vbslq_f32(vcltq_f32(x, h), x, h);
}

FM_INLINE float32x4_t fmn_floor(float32x4_t t)
{
    float32x4_t f = vcvtq_f32_s32(vcvtq_s32_f32(t));
    return vsubq_f32(f, vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(f, t), vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
}

static void FMK(depth_f32)(const float* z, float* zb, uint8_t* m, int n, int func, int write, float* wmin,
                           float* wmax, int* nw)
{
    float l = *wmin, h = *wmax;
    int   cnt = *nw, i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t mw;
        memcpy(&mw, m + i, 4);
        if (!mw) continue;
        float32x4_t zc   = fmn_clamp(vld1q_f32(z + i), 0.0f, 1.0f);
        float32x4_t old  = vld1q_f32(zb + i);
        uint32x4_t  pass = fmn_fcmp(func, zc, old);
        uint32_t    pl[4];
        float       zl[4];
        vst1q_u32(pl, pass);
        vst1q_f32(zl, zc);
        for (int k = 0; k < 4; k++) {
            if (!m[i + k]) continue;
            if (!pl[k]) {
                m[i + k] = 0;
                continue;
            }
            if (write) {
                zb[i + k] = zl[k];
                l         = zl[k] < l ? zl[k] : l;
                h         = zl[k] > h ? zl[k] : h;
                cnt++;
            }
        }
    }
    for (; i < n; i++) {
        if (!m[i]) continue;
        float zc = fm_clamp01(z[i]);
        if (!fm_fcmp(func, zc, zb[i])) {
            m[i] = 0;
            continue;
        }
        if (write) {
            zb[i] = zc;
            l     = zc < l ? zc : l;
            h     = zc > h ? zc : h;
            cnt++;
        }
    }
    *wmin = l;
    *wmax = h;
    *nw   = cnt;
}

static void FMK(skin4)(const float* bones, int nbones, const void* skin, int skin_stride, const float* pos, int pos_stride,
                       int n, float* out3)
{
    for (int i = 0; i < n; i++) {
        const uint8_t* rec = (const uint8_t*)skin + (size_t)i * (size_t)skin_stride;
        const float*   p   = pos + (size_t)i * (size_t)pos_stride;
        uint16_t       j[4];
        float          w[4];
        memcpy(j, rec, sizeof(j));
        memcpy(w, rec + 8, sizeof(w));
        float32x4_t c[4];
        for (int k = 0; k < 4; k++) {
            const float* b = bones + 16 * (j[k] < nbones ? j[k] : 0);
            for (int col = 0; col < 4; col++) {
                float32x4_t t = vmulq_f32(vdupq_n_f32(w[k]), vld1q_f32(b + 4 * col));
                c[col]        = k ? vaddq_f32(c[col], t) : t;
            }
        }
        float32x4_t o = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_f32(c[0], vdupq_n_f32(p[0])), vmulq_f32(c[1], vdupq_n_f32(p[1]))),
                                            vmulq_f32(c[2], vdupq_n_f32(p[2]))),
                                  c[3]);
        float r[4];
        vst1q_f32(r, o);
        out3[3 * i]     = r[0];
        out3[3 * i + 1] = r[1];
        out3[3 * i + 2] = r[2];
    }
}

static void FMK(depth_ms)(const float* zc, const float* dzs, int S, float* zb, uint8_t* smask, int n, int func,
                          int write)
{
    if (S != 4 && S != 8) {
        for (int i = 0; i < n; i++) fm_depth_ms1(zc, dzs, S, zb, smask, i, func, write);
        return;
    }
    static const uint32_t bitl[4] = { 1, 2, 4, 8 };
    const uint32x4_t      bitv    = vld1q_u32(bitl);
    for (int i = 0; i < n; i++) {
        uint8_t bits = smask[i];
        if (!bits) continue;
        float* d = zb + (size_t)i * (size_t)S;
        for (int h = 0; h < S / 4; h++) {
            uint32_t hb = (bits >> (4 * h)) & 15u;
            if (!hb) continue;
            float32x4_t zs   = fmn_clamp(vaddq_f32(vdupq_n_f32(zc[i]), vld1q_f32(dzs + 4 * h)), 0.0f, 1.0f);
            float32x4_t old  = vld1q_f32(d + 4 * h);
            uint32x4_t  act  = vceqq_u32(vandq_u32(vdupq_n_u32(hb), bitv), bitv);
            uint32x4_t  pass = vandq_u32(fmn_fcmp(func, zs, old), act);
            uint32_t    pl[4];
            vst1q_u32(pl, pass);
            uint32_t pm = (pl[0] & 1u) | (pl[1] & 2u) | (pl[2] & 4u) | (pl[3] & 8u);
            if (write && pm) vst1q_f32(d + 4 * h, vbslq_f32(pass, zs, old));
            bits = (uint8_t)((bits & ~(15u << (4 * h))) | (pm << (4 * h)));
        }
        smask[i] = bits;
    }
}

static void FMK(resolve)(const uint32_t* s, int S, int n, uint32_t* out)
{
    if (S != 4 && S != 8) {
        for (int i = 0; i < n; i++) out[i] = fm_resolve1(s + (size_t)i * (size_t)S, S);
        return;
    }
    for (int i = 0; i < n; i++) {
        const uint32_t* p   = s + (size_t)i * (size_t)S;
        uint8x16_t      x   = vld1q_u8((const uint8_t*)p);
        uint16x8_t      sum = vaddl_u8(vget_low_u8(x), vget_high_u8(x));
        if (S == 8) {
            uint8x16_t y = vld1q_u8((const uint8_t*)(p + 4));
            sum          = vaddq_u16(sum, vaddl_u8(vget_low_u8(y), vget_high_u8(y)));
        }
        uint16x4_t t = vadd_u16(vget_low_u16(sum), vget_high_u16(sum));
        t            = vadd_u16(t, vdup_n_u16((uint16_t)(S >> 1)));
        t            = S == 8 ? vshr_n_u16(t, 3) : vshr_n_u16(t, 2);
        uint8x8_t b  = vmovn_u16(vcombine_u16(t, t));
        out[i]       = vget_lane_u32(vreinterpret_u32_u8(b), 0);
    }
}

static void FMK(texcoord)(const float* u, int n, int wrap, float size, int bilinear, int32_t* out)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = fmn_clamp(vld1q_f32(u + i), -1e6f, 1e6f);
        if (wrap == 0)
            x = vsubq_f32(x, fmn_floor(x));
        else if (wrap == 2)
            x = vsubq_f32(x, vmulq_f32(vdupq_n_f32(2.0f), fmn_floor(vmulq_f32(x, vdupq_n_f32(0.5f)))));
        else
            x = fmn_clamp(x, -1.0f, 2.0f);
        float32x4_t f = fmn_clamp(vmulq_f32(x, vdupq_n_f32(size)), -32767.0f, 32767.0f);
        int32x4_t   q = vcvtq_s32_f32(fmn_floor(vaddq_f32(vmulq_f32(f, vdupq_n_f32(65536.0f)), vdupq_n_f32(0.5f))));
        if (bilinear) q = vsubq_s32(q, vdupq_n_s32(32768));
        vst1q_s32(out + i, q);
    }
    for (; i < n; i++) out[i] = fm_texcoord1(u[i], wrap, size, bilinear);
}

static void FMK(premul_f)(const float* r, const float* g, const float* b, const float* a, int n, uint32_t* out)
{
    int i = 0;
    const float32x4_t k = vdupq_n_f32(255.0f), hf = vdupq_n_f32(0.5f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t av = fmn_clamp(vld1q_f32(a + i), 0.0f, 1.0f);
        uint32x4_t  A  = vreinterpretq_u32_s32(vcvtq_s32_f32(vaddq_f32(vmulq_f32(av, k), hf)));
        uint32x4_t  R  = vreinterpretq_u32_s32(vcvtq_s32_f32(vaddq_f32(vmulq_f32(vmulq_f32(fmn_clamp(vld1q_f32(r + i), 0, 1), av), k), hf)));
        uint32x4_t  G  = vreinterpretq_u32_s32(vcvtq_s32_f32(vaddq_f32(vmulq_f32(vmulq_f32(fmn_clamp(vld1q_f32(g + i), 0, 1), av), k), hf)));
        uint32x4_t  B  = vreinterpretq_u32_s32(vcvtq_s32_f32(vaddq_f32(vmulq_f32(vmulq_f32(fmn_clamp(vld1q_f32(b + i), 0, 1), av), k), hf)));
        R              = vminq_u32(R, A);
        G              = vminq_u32(G, A);
        B              = vminq_u32(B, A);
        uint32x4_t o   = vorrq_u32(vorrq_u32(vshlq_n_u32(A, 24), vshlq_n_u32(R, 16)), vorrq_u32(vshlq_n_u32(G, 8), B));
        vst1q_u32(out + i, o);
    }
    for (; i < n; i++) out[i] = fm_premul_f1(r[i], g[i], b[i], a[i]);
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

/* ---- rasterizer interpolation ---- */

#define FMN_PLANE4(A, B, DX) vaddq_f32(A, vmulq_f32(B, vld1q_f32(DX)))

static void FMK(acc_add)(float* acc, float v, int n)
{
    int               i  = 0;
    const float32x4_t v4 = vdupq_n_f32(v);
    for (; i + 4 <= n; i += 4) vst1q_f32(acc + i, vaddq_f32(vld1q_f32(acc + i), v4));
    for (; i < n; i++) acc[i] += v;
}

static void FMK(plane)(float a, float b, const float* dx, int n, int clamp01, float* out)
{
    int               i = 0;
    const float32x4_t A = vdupq_n_f32(a), B = vdupq_n_f32(b);
    if (clamp01)
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, fmn_clamp(FMN_PLANE4(A, B, dx + i), 0.0f, 1.0f));
    else
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, FMN_PLANE4(A, B, dx + i));
    for (; i < n; i++) {
        float v = fm_plane1(a, b, dx[i]);
        out[i]  = clamp01 ? fm_clamp01(v) : v;
    }
}

static void FMK(plane_recip)(float a, float b, const float* dx, int n, float* out)
{
    int               i = 0;
    const float32x4_t A = vdupq_n_f32(a), B = vdupq_n_f32(b), o = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vdivq_f32(o, FMN_PLANE4(A, B, dx + i)));
    for (; i < n; i++) out[i] = 1.0f / fm_plane1(a, b, dx[i]);
}

static void FMK(plane_mul)(float a, float b, const float* dx, const float* w, int n, float* out)
{
    int               i = 0;
    const float32x4_t A = vdupq_n_f32(a), B = vdupq_n_f32(b);
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vmulq_f32(FMN_PLANE4(A, B, dx + i), vld1q_f32(w + i)));
    for (; i < n; i++) out[i] = fm_plane1(a, b, dx[i]) * w[i];
}

static void FMK(minmax_f32)(const float* p, int n, float* mn, float* mx)
{
    int   i = 0;
    float a = p[0], b = p[0];
    if (n >= 4) {
        float32x4_t lo = vld1q_f32(p), hi = lo;
        for (i = 4; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(p + i);
            lo            = vminq_f32(lo, v);
            hi            = vmaxq_f32(hi, v);
        }
        a = vminvq_f32(lo);
        b = vmaxvq_f32(hi);
    }
    for (; i < n; i++) {
        a = p[i] < a ? p[i] : a;
        b = p[i] > b ? p[i] : b;
    }
    *mn = a;
    *mx = b;
}

#include "fm_kernels_tmpl.h"

#endif
