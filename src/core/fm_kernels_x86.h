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

#if defined(__AVX2__)
#include <immintrin.h>
/* one channel of 8 pixels: two stage 8 bit lerp in 32-bit lanes */
FM_INLINE __m256i fmx8_chan(__m256i p00, __m256i p01, __m256i p10, __m256i p11, __m256i fx, __m256i ix, __m256i fy,
                            __m256i iy, int sh)
{
    const __m256i m = _mm256_set1_epi32(255);
    __m256i       a = _mm256_and_si256(_mm256_srli_epi32(p00, sh), m), b = _mm256_and_si256(_mm256_srli_epi32(p01, sh), m);
    __m256i       c = _mm256_and_si256(_mm256_srli_epi32(p10, sh), m), d = _mm256_and_si256(_mm256_srli_epi32(p11, sh), m);
    /* products < 65536: 16 bit multiplies on 32 bit lanes are exact */
    __m256i t = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi16(a, ix), _mm256_mullo_epi16(b, fx)), 8);
    __m256i u = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi16(c, ix), _mm256_mullo_epi16(d, fx)), 8);
    return _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi16(t, iy), _mm256_mullo_epi16(u, fy)), 8);
}
#endif

static void FMK(bilinear_pts)(const uint32_t* tex, int stride, const int32_t* U, const int32_t* V, int n,
                              uint32_t* out)
{
    int i = 0;
#if defined(__AVX2__)
    const __m256i k255 = _mm256_set1_epi32(255), k256 = _mm256_set1_epi32(256), vs = _mm256_set1_epi32(stride);
    const int*    base = (const int*)tex;
    for (; i + 8 <= n; i += 8) {
        __m256i u   = _mm256_loadu_si256((const __m256i*)(U + i)), v = _mm256_loadu_si256((const __m256i*)(V + i));
        __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srai_epi32(v, 16), vs), _mm256_srai_epi32(u, 16));
        __m256i p00 = _mm256_i32gather_epi32(base, idx, 4);
        __m256i p01 = _mm256_i32gather_epi32(base + 1, idx, 4);
        __m256i p10 = _mm256_i32gather_epi32(base + stride, idx, 4);
        __m256i p11 = _mm256_i32gather_epi32(base + stride + 1, idx, 4);
        __m256i fx  = _mm256_and_si256(_mm256_srli_epi32(u, 8), k255), fy = _mm256_and_si256(_mm256_srli_epi32(v, 8), k255);
        __m256i ix  = _mm256_sub_epi32(k256, fx), iy = _mm256_sub_epi32(k256, fy);
        __m256i o   = fmx8_chan(p00, p01, p10, p11, fx, ix, fy, iy, 0);
        o           = _mm256_or_si256(o, _mm256_slli_epi32(fmx8_chan(p00, p01, p10, p11, fx, ix, fy, iy, 8), 8));
        o           = _mm256_or_si256(o, _mm256_slli_epi32(fmx8_chan(p00, p01, p10, p11, fx, ix, fy, iy, 16), 16));
        o           = _mm256_or_si256(o, _mm256_slli_epi32(fmx8_chan(p00, p01, p10, p11, fx, ix, fy, iy, 24), 24));
        _mm256_storeu_si256((__m256i*)(out + i), o);
    }
#endif
    for (; i < n; i++)
        out[i] = fmx_bilerp1(tex + (ptrdiff_t)(V[i] >> 16) * stride + (U[i] >> 16), stride, U[i], V[i]);
}

FM_INLINE __m128 fmx_floor(__m128 t)
{
    __m128 f = _mm_cvtepi32_ps(_mm_cvttps_epi32(t));
    return _mm_sub_ps(f, _mm_and_ps(_mm_cmpgt_ps(f, t), _mm_set1_ps(1.0f)));
}

/* ---- 3D fragment ops ---- */

FM_INLINE __m128 fmx_fcmp(int func, __m128 a, __m128 b)
{
    switch (func) {
    case 0: return _mm_setzero_ps();
    case 1: return _mm_cmplt_ps(a, b);
    case 2: return _mm_cmpeq_ps(a, b);
    case 3: return _mm_cmple_ps(a, b);
    case 4: return _mm_cmpgt_ps(a, b);
    case 5: return _mm_cmpneq_ps(a, b);
    case 6: return _mm_cmpge_ps(a, b);
    default: return _mm_castsi128_ps(_mm_set1_epi32(-1));
    }
}

FM_INLINE __m128 fmx_clamp01(__m128 x) { return _mm_min_ps(_mm_max_ps(x, _mm_setzero_ps()), _mm_set1_ps(1.0f)); }

static void FMK(depth_f32)(const float* z, float* zb, uint8_t* m, int n, int func, int write, float* wmin,
                           float* wmax, int* nw)
{
    __m128 lo = _mm_set1_ps(*wmin), hi = _mm_set1_ps(*wmax);
    int    cnt = *nw, i = 0;
    const __m128 pinf = _mm_set1_ps(1e30f), ninf = _mm_set1_ps(-1e30f);
    for (; i + 4 <= n; i += 4) {
        uint32_t mw;
        memcpy(&mw, m + i, 4);
        if (!mw) continue;
        __m128i act = _mm_cmpeq_epi32(_mm_setzero_si128(),
                                      _mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128((int)mw), _mm_setzero_si128()),
                                                         _mm_setzero_si128()));
        act          = _mm_xor_si128(act, _mm_set1_epi32(-1)); /* lanes with m != 0 */
        __m128 zc    = fmx_clamp01(_mm_loadu_ps(z + i));
        __m128 old   = _mm_loadu_ps(zb + i);
        __m128 pass  = _mm_and_ps(fmx_fcmp(func, zc, old), _mm_castsi128_ps(act));
        int    pm    = _mm_movemask_ps(pass);
        /* failed active lanes clear their coverage byte */
        for (int k = 0; k < 4; k++)
            if (!(pm & (1 << k))) m[i + k] = 0;
        if (write && pm) {
            _mm_storeu_ps(zb + i, _mm_or_ps(_mm_and_ps(pass, zc), _mm_andnot_ps(pass, old)));
            lo = _mm_min_ps(lo, _mm_or_ps(_mm_and_ps(pass, zc), _mm_andnot_ps(pass, pinf)));
            hi = _mm_max_ps(hi, _mm_or_ps(_mm_and_ps(pass, zc), _mm_andnot_ps(pass, ninf)));
            cnt += (pm & 1) + ((pm >> 1) & 1) + ((pm >> 2) & 1) + ((pm >> 3) & 1);
        }
    }
    float l4[4], h4[4];
    _mm_storeu_ps(l4, lo);
    _mm_storeu_ps(h4, hi);
    float l = FM_MIN(FM_MIN(l4[0], l4[1]), FM_MIN(l4[2], l4[3]));
    float h = FM_MAX(FM_MAX(h4[0], h4[1]), FM_MAX(h4[2], h4[3]));
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

/* vertex blending: the blended matrix is built column by column in SSE
 * lanes (same order as fm_skin1), then applied to the position */
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
        __m128 c[4];
        for (int k = 0; k < 4; k++) {
            const float* b  = bones + 16 * (j[k] < nbones ? j[k] : 0);
            __m128       wk = _mm_set1_ps(w[k]);
            for (int col = 0; col < 4; col++) {
                __m128 t = _mm_mul_ps(wk, _mm_loadu_ps(b + 4 * col));
                c[col]   = k ? _mm_add_ps(c[col], t) : t;
            }
        }
        __m128 o = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(c[0], _mm_set1_ps(p[0])), _mm_mul_ps(c[1], _mm_set1_ps(p[1]))),
                                         _mm_mul_ps(c[2], _mm_set1_ps(p[2]))),
                              c[3]);
        float r[4];
        _mm_storeu_ps(r, o);
        out3[3 * i]     = r[0];
        out3[3 * i + 1] = r[1];
        out3[3 * i + 2] = r[2];
    }
}

/* 4 samples of one pixel in one vector (8x: two vectors) */
static void FMK(depth_ms)(const float* zc, const float* dzs, int S, float* zb, uint8_t* smask, int n, int func,
                          int write)
{
    if (S != 4 && S != 8) {
        for (int i = 0; i < n; i++) fm_depth_ms1(zc, dzs, S, zb, smask, i, func, write);
        return;
    }
    const __m128  d0 = _mm_loadu_ps(dzs), d1 = S == 8 ? _mm_loadu_ps(dzs + 4) : _mm_setzero_ps();
    const __m128i bitv = _mm_set_epi32(8, 4, 2, 1);
    for (int i = 0; i < n; i++) {
        uint8_t bits = smask[i];
        if (!bits) continue;
        float* d = zb + (size_t)i * (size_t)S;
        __m128 z = _mm_set1_ps(zc[i]);
        for (int h = 0; h < S / 4; h++) {
            uint32_t hb = (bits >> (4 * h)) & 15u;
            if (!hb) continue;
            __m128  zs   = fmx_clamp01(_mm_add_ps(z, h ? d1 : d0));
            __m128  old  = _mm_loadu_ps(d + 4 * h);
            __m128i act  = _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32((int)hb), bitv), bitv);
            __m128  pass = _mm_and_ps(fmx_fcmp(func, zs, old), _mm_castsi128_ps(act));
            int     pm   = _mm_movemask_ps(pass);
            if (write && pm) _mm_storeu_ps(d + 4 * h, _mm_or_ps(_mm_and_ps(pass, zs), _mm_andnot_ps(pass, old)));
            bits = (uint8_t)((bits & ~(15u << (4 * h))) | ((uint32_t)pm << (4 * h)));
        }
        smask[i] = bits;
    }
}

static void FMK(resolve)(const uint32_t* s, int S, int n, uint32_t* out)
{
    const __m128i zero = _mm_setzero_si128();
    if (S != 4 && S != 8) {
        for (int i = 0; i < n; i++) out[i] = fm_resolve1(s + (size_t)i * (size_t)S, S);
        return;
    }
    const __m128i rnd = _mm_set1_epi16((short)(S >> 1));
    const int     sh  = S == 8 ? 3 : 2;
    for (int i = 0; i < n; i++) {
        const uint32_t* p   = s + (size_t)i * (size_t)S;
        __m128i         x   = _mm_loadu_si128((const __m128i*)p);
        __m128i         sum = _mm_add_epi16(_mm_unpacklo_epi8(x, zero), _mm_unpackhi_epi8(x, zero));
        if (S == 8) {
            __m128i y = _mm_loadu_si128((const __m128i*)(p + 4));
            sum       = _mm_add_epi16(sum, _mm_add_epi16(_mm_unpacklo_epi8(y, zero), _mm_unpackhi_epi8(y, zero)));
        }
        sum = _mm_add_epi16(sum, _mm_srli_si128(sum, 8));
        sum = _mm_srli_epi16(_mm_add_epi16(sum, rnd), sh);
        out[i] = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(sum, sum));
    }
}

static void FMK(texcoord)(const float* u, int n, int wrap, float size, int bilinear, int32_t* out)
{
    const __m128  sz = _mm_set1_ps(size), two = _mm_set1_ps(2.0f), half = _mm_set1_ps(0.5f);
    const __m128i bias = _mm_set1_epi32(bilinear ? 32768 : 0);
    int           i    = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 x = _mm_min_ps(_mm_max_ps(_mm_loadu_ps(u + i), _mm_set1_ps(-1e6f)), _mm_set1_ps(1e6f));
        if (wrap == 0)
            x = _mm_sub_ps(x, fmx_floor(x));
        else if (wrap == 2)
            x = _mm_sub_ps(x, _mm_mul_ps(two, fmx_floor(_mm_mul_ps(x, half))));
        else
            x = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-1.0f)), _mm_set1_ps(2.0f));
        __m128 f = _mm_mul_ps(x, sz);
        f        = _mm_min_ps(_mm_max_ps(f, _mm_set1_ps(-32767.0f)), _mm_set1_ps(32767.0f));
        __m128i q = _mm_cvttps_epi32(fmx_floor(_mm_add_ps(_mm_mul_ps(f, _mm_set1_ps(65536.0f)), half)));
        _mm_storeu_si128((__m128i*)(out + i), _mm_sub_epi32(q, bias));
    }
    for (; i < n; i++) out[i] = fm_texcoord1(u[i], wrap, size, bilinear);
}

FM_INLINE __m128i fmx_min_epi32(__m128i a, __m128i b)
{
    __m128i gt = _mm_cmpgt_epi32(a, b);
    return _mm_or_si128(_mm_and_si128(gt, b), _mm_andnot_si128(gt, a));
}

static void FMK(premul_f)(const float* r, const float* g, const float* b, const float* a, int n, uint32_t* out)
{
    const __m128 k255 = _mm_set1_ps(255.0f), half = _mm_set1_ps(0.5f);
    int          i    = 0;
    for (; i + 4 <= n; i += 4) {
        __m128  av = fmx_clamp01(_mm_loadu_ps(a + i));
        __m128i A  = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(av, k255), half));
        __m128i R  = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(fmx_clamp01(_mm_loadu_ps(r + i)), av), k255), half));
        __m128i G  = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(fmx_clamp01(_mm_loadu_ps(g + i)), av), k255), half));
        __m128i B  = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(fmx_clamp01(_mm_loadu_ps(b + i)), av), k255), half));
        R          = fmx_min_epi32(R, A);
        G          = fmx_min_epi32(G, A);
        B          = fmx_min_epi32(B, A);
        __m128i o  = _mm_or_si128(_mm_or_si128(_mm_slli_epi32(A, 24), _mm_slli_epi32(R, 16)),
                                  _mm_or_si128(_mm_slli_epi32(G, 8), B));
        _mm_storeu_si128((__m128i*)(out + i), o);
    }
    for (; i < n; i++) out[i] = fm_premul_f1(r[i], g[i], b[i], a[i]);
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

/* ---- rasterizer interpolation (SSE2 4 wide; the AVX2 build adds 8 wide) ---- */

#if defined(__AVX2__)
#  define FMX_PLANE8(A, B, DX) _mm256_add_ps(A, _mm256_mul_ps(B, _mm256_loadu_ps(DX)))
#endif
#define FMX_PLANE4(A, B, DX) _mm_add_ps(A, _mm_mul_ps(B, _mm_loadu_ps(DX)))

static void FMK(acc_add)(float* acc, float v, int n)
{
    int i = 0;
#if defined(__AVX2__)
    const __m256 v8 = _mm256_set1_ps(v);
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(acc + i, _mm256_add_ps(_mm256_loadu_ps(acc + i), v8));
#endif
    const __m128 v4 = _mm_set1_ps(v);
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(acc + i, _mm_add_ps(_mm_loadu_ps(acc + i), v4));
    for (; i < n; i++) acc[i] += v;
}

static void FMK(plane)(float a, float b, const float* dx, int n, int clamp01, float* out)
{
    int i = 0;
#if defined(__AVX2__)
    {
        const __m256 A = _mm256_set1_ps(a), B = _mm256_set1_ps(b);
        const __m256 z = _mm256_setzero_ps(), o = _mm256_set1_ps(1.0f);
        if (clamp01)
            for (; i + 8 <= n; i += 8)
                _mm256_storeu_ps(out + i, _mm256_min_ps(_mm256_max_ps(FMX_PLANE8(A, B, dx + i), z), o));
        else
            for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, FMX_PLANE8(A, B, dx + i));
    }
#endif
    const __m128 A = _mm_set1_ps(a), B = _mm_set1_ps(b);
    if (clamp01)
        for (; i + 4 <= n; i += 4) _mm_storeu_ps(out + i, fmx_clamp01(FMX_PLANE4(A, B, dx + i)));
    else
        for (; i + 4 <= n; i += 4) _mm_storeu_ps(out + i, FMX_PLANE4(A, B, dx + i));
    for (; i < n; i++) {
        float v = fm_plane1(a, b, dx[i]);
        out[i]  = clamp01 ? fm_clamp01(v) : v;
    }
}

static void FMK(plane_recip)(float a, float b, const float* dx, int n, float* out)
{
    int i = 0;
#if defined(__AVX2__)
    {
        const __m256 A = _mm256_set1_ps(a), B = _mm256_set1_ps(b), o = _mm256_set1_ps(1.0f);
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, _mm256_div_ps(o, FMX_PLANE8(A, B, dx + i)));
    }
#endif
    const __m128 A = _mm_set1_ps(a), B = _mm_set1_ps(b), o = _mm_set1_ps(1.0f);
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(out + i, _mm_div_ps(o, FMX_PLANE4(A, B, dx + i)));
    for (; i < n; i++) out[i] = 1.0f / fm_plane1(a, b, dx[i]);
}

static void FMK(plane_mul)(float a, float b, const float* dx, const float* w, int n, float* out)
{
    int i = 0;
#if defined(__AVX2__)
    {
        const __m256 A = _mm256_set1_ps(a), B = _mm256_set1_ps(b);
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(out + i, _mm256_mul_ps(FMX_PLANE8(A, B, dx + i), _mm256_loadu_ps(w + i)));
    }
#endif
    const __m128 A = _mm_set1_ps(a), B = _mm_set1_ps(b);
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(out + i, _mm_mul_ps(FMX_PLANE4(A, B, dx + i), _mm_loadu_ps(w + i)));
    for (; i < n; i++) out[i] = fm_plane1(a, b, dx[i]) * w[i];
}

#endif
