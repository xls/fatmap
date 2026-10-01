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
FM_INLINE vw vw_shr8(vw a) { FMK_LANES(a.c[k] >> 8); }
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

static void bilinear_pts_scalar(const uint32_t* tex, int stride, const int32_t* U, const int32_t* V, int n,
                                uint32_t* out)
{
    for (int i = 0; i < n; i++) {
        const uint32_t* r0 = tex + (ptrdiff_t)(V[i] >> 16) * stride + (U[i] >> 16);
        out[i] = fm_bilerp(r0[0], r0[1], r0[stride], r0[stride + 1], (uint32_t)(U[i] >> 8) & 255u,
                           (uint32_t)(V[i] >> 8) & 255u);
    }
}

static void depth_f32_scalar(const float* z, float* zb, uint8_t* m, int n, int func, int write, float* wmin,
                             float* wmax, int* nw)
{
    float lo = *wmin, hi = *wmax;
    int   cnt = *nw;
    for (int i = 0; i < n; i++) {
        if (!m[i]) continue;
        float zc = fm_clamp01(z[i]);
        if (!fm_fcmp(func, zc, zb[i])) {
            m[i] = 0;
            continue;
        }
        if (write) {
            zb[i] = zc;
            lo    = zc < lo ? zc : lo;
            hi    = zc > hi ? zc : hi;
            cnt++;
        }
    }
    *wmin = lo;
    *wmax = hi;
    *nw   = cnt;
}

static void skin4_scalar(const float* bones, int nbones, const void* skin, int skin_stride, const float* pos,
                         int pos_stride, int n, float* out3)
{
    for (int i = 0; i < n; i++)
        fm_skin1(bones, nbones, (const uint8_t*)skin + (size_t)i * (size_t)skin_stride, pos + (size_t)i * (size_t)pos_stride,
                 out3 + 3 * i);
}

static void depth_ms_scalar(const float* zc, const float* dzs, int S, float* zb, uint8_t* smask, int n, int func,
                            int write)
{
    for (int i = 0; i < n; i++) fm_depth_ms1(zc, dzs, S, zb, smask, i, func, write);
}

static void resolve_scalar(const uint32_t* s, int S, int n, uint32_t* out)
{
    for (int i = 0; i < n; i++) out[i] = fm_resolve1(s + (size_t)i * (size_t)S, S);
}

static void texcoord_scalar(const float* u, int n, int wrap, float size, int bilinear, int32_t* out)
{
    for (int i = 0; i < n; i++) out[i] = fm_texcoord1(u[i], wrap, size, bilinear);
}

static void premul_f_scalar(const float* r, const float* g, const float* b, const float* a, int n, uint32_t* out)
{
    for (int i = 0; i < n; i++) out[i] = fm_premul_f1(r[i], g[i], b[i], a[i]);
}

static void linear_grad_scalar(const uint32_t* lut, float t0, float dt, int n, int extend, uint32_t* out)
{
    for (int i = 0; i < n; i++) out[i] = lut[fm_grad_index(t0 + (float)i * dt, extend)];
}

static void acc_add_scalar(float* acc, float v, int n)
{
    for (int i = 0; i < n; i++) acc[i] += v;
}

static void plane_scalar(float a, float b, const float* dx, int n, int clamp01, float* out)
{
    for (int i = 0; i < n; i++) out[i] = fm_plane1(a, b, dx[i]);
    if (clamp01)
        for (int i = 0; i < n; i++) out[i] = fm_clamp01(out[i]);
}

static void plane_recip_scalar(float a, float b, const float* dx, int n, float* out)
{
    for (int i = 0; i < n; i++) out[i] = 1.0f / fm_plane1(a, b, dx[i]);
}

static void plane_mul_scalar(float a, float b, const float* dx, const float* w, int n, float* out)
{
    for (int i = 0; i < n; i++) out[i] = fm_plane1(a, b, dx[i]) * w[i];
}

static void minmax_f32_scalar(const float* p, int n, float* mn, float* mx)
{
    float a = p[0], b = p[0];
    for (int i = 1; i < n; i++) {
        a = p[i] < a ? p[i] : a;
        b = p[i] > b ? p[i] : b;
    }
    *mn = a;
    *mx = b;
}

static void bilinear_pts_wrap_scalar(const uint32_t* tex, int stride, const int32_t* U, const int32_t* V, int n,
                                     int wm, int hm, uint32_t* out)
{
    for (int i = 0; i < n; i++) out[i] = fm_bilerp_wrap1(tex, stride, U[i], V[i], wm, hm);
}

/* float template, 1 lane: the reference every SIMD backend must match */
#define FMF_W 1
typedef float vf;
typedef int   vm;
FM_INLINE vf vf_set(float f) { return f; }
FM_INLINE vf vf_ld(const float* p) { return *p; }
FM_INLINE void vf_st(float* p, vf v) { *p = v; }
FM_INLINE vf vf_add(vf a, vf b) { return a + b; }
FM_INLINE vf vf_sub(vf a, vf b) { return a - b; }
FM_INLINE vf vf_mul(vf a, vf b) { return a * b; }
FM_INLINE vf vf_div(vf a, vf b) { return a / b; }
FM_INLINE vf vf_sqrt(vf a) { return sqrtf(a); }
FM_INLINE vm vf_gt(vf a, vf b) { return a > b; }
FM_INLINE vm vf_ge(vf a, vf b) { return a >= b; }
FM_INLINE vf vf_sel(vm m, vf a, vf b) { return m ? a : b; }
FM_INLINE vf vf_floor(vf a) { return fm_ffloor(a); }
FM_INLINE uint32_t fmf_bits(vf a)
{
    uint32_t u;
    memcpy(&u, &a, 4);
    return u;
}
FM_INLINE vf fmf_from_bits(uint32_t u)
{
    vf a;
    memcpy(&a, &u, 4);
    return a;
}
FM_INLINE vf vf_exp_of(vf a) { return (float)((int)((fmf_bits(a) >> 23) & 255u) - 127); }
FM_INLINE vf vf_mant_of(vf a) { return fmf_from_bits((fmf_bits(a) & 0x7fffffu) | 0x3f800000u); }
FM_INLINE vf vf_pow2i(vf i) { return fmf_from_bits((uint32_t)((int)i + 127) << 23); }
#include "fm_kernels_ftmpl.h"

#include "fm_kernels_tmpl.h"
