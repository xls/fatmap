/*
 * fatmap - texture sampling of one 16 lane quad group (lanes 0..7: row 0,
 * 8..15: row 1, quad q = columns 2q, 2q + 1) with AVX2: the level of detail
 * per quad, both trilinear levels, every wrap mode, texel gathers from the
 * packed level block, the bilinear lerp and the conversion to floats in one
 * pass. The same integer and float operations, in the same order, as
 * fm3d__sample_quads + fm_sample_points + fm__sample_fixed, so every
 * backend keeps rendering the same bits.
 */
#include "fm3d_internal.h"

#if FM_FEATURE_SHADERS && defined(__AVX2__)

#include <immintrin.h>

/* fm_wrap_coord on 8 lanes (n: per lane sizes); border lanes leave *valid */
FM_INLINE __m256i fs_wrap(__m256i x, __m256i n, int pow2, fm_wrap w, __m256i* valid)
{
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi32(1);
    switch (w) {
    case FM_WRAP_REPEAT: {
        if (pow2) return _mm256_and_si256(x, _mm256_sub_epi32(n, one));
        /* x - n floor(x / n): exact for |x| <= 2^16 */
        __m256  q = _mm256_floor_ps(_mm256_div_ps(_mm256_cvtepi32_ps(x), _mm256_cvtepi32_ps(n)));
        return _mm256_sub_epi32(x, _mm256_mullo_epi32(_mm256_cvttps_epi32(q), n));
    }
    case FM_WRAP_CLAMP: return _mm256_min_epi32(_mm256_max_epi32(x, zero), _mm256_sub_epi32(n, one));
    case FM_WRAP_MIRROR: {
        __m256i p = _mm256_add_epi32(n, n);
        __m256  q = _mm256_floor_ps(_mm256_div_ps(_mm256_cvtepi32_ps(x), _mm256_cvtepi32_ps(p)));
        __m256i r = _mm256_sub_epi32(x, _mm256_mullo_epi32(_mm256_cvttps_epi32(q), p));
        __m256i m = _mm256_sub_epi32(_mm256_sub_epi32(p, one), r);
        return _mm256_blendv_epi8(m, r, _mm256_cmpgt_epi32(n, r)); /* r < n ? r : p - 1 - r */
    }
    default: { /* border: outside reads 0 */
        __m256i in = _mm256_andnot_si256(_mm256_cmpgt_epi32(zero, x), _mm256_cmpgt_epi32(n, x));
        *valid     = _mm256_and_si256(*valid, in);
        return _mm256_and_si256(x, in);
    }
    }
}

/* FM_CLAMP(f, -32767, 32767): NaN stays NaN, like the C */
FM_INLINE __m256 fs_clamp(__m256 f)
{
    const __m256 lo = _mm256_set1_ps(-32767.0f), hi = _mm256_set1_ps(32767.0f);
    f = _mm256_blendv_ps(f, lo, _mm256_cmp_ps(f, lo, _CMP_LT_OQ));
    return _mm256_blendv_ps(f, hi, _mm256_cmp_ps(f, hi, _CMP_GT_OQ));
}

/* (int32_t)fm_ffloor(f * 65536 + 0.5f) */
FM_INLINE __m256i fs_fixed(__m256 f)
{
    __m256 t = _mm256_add_ps(_mm256_mul_ps(fs_clamp(f), _mm256_set1_ps(65536.0f)), _mm256_set1_ps(0.5f));
    return _mm256_cvttps_epi32(_mm256_floor_ps(t));
}

FM_INLINE __m256i fs_chan(__m256i p00, __m256i p01, __m256i p10, __m256i p11, __m256i fx, __m256i ix, __m256i fy, __m256i iy, int sh)
{
    const __m256i m = _mm256_set1_epi32(255);
    __m256i       a = _mm256_and_si256(_mm256_srli_epi32(p00, sh), m), b = _mm256_and_si256(_mm256_srli_epi32(p01, sh), m);
    __m256i       c = _mm256_and_si256(_mm256_srli_epi32(p10, sh), m), d = _mm256_and_si256(_mm256_srli_epi32(p11, sh), m);
    __m256i       t = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi16(a, ix), _mm256_mullo_epi16(b, fx)), 8);
    __m256i       u = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi16(c, ix), _mm256_mullo_epi16(d, fx)), 8);
    return _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi16(t, iy), _mm256_mullo_epi16(u, fy)), 8);
}

FM_INLINE __m256i fs_gather(const int* base, __m256i idx, __m256i valid)
{
    return _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), base, idx, valid, 4);
}

/* 8 lanes (one row) at per lane levels lv (lanes 2q, 2q + 1: quad q) */
FM_INLINE __m256i fs_level8(const fm3d_texture* t, const fm3d_sampler* s, int bilinear, const float* U, const float* V, const int* lv)
{
    int32_t wv[8], hv[8], ov[8];
    float   wf[8], hf[8];
    for (int l = 0; l < 8; l++) {
        const fm_surface* L = t->level[lv[l >> 1]];
        wv[l] = L->width, hv[l] = L->height, ov[l] = t->poff[lv[l >> 1]];
        wf[l] = (float)L->width, hf[l] = (float)L->height;
    }
    const fm_surface* L0   = t->level[0];
    int               pow2 = !(L0->width & (L0->width - 1)) && !(L0->height & (L0->height - 1));
    __m256i           W = _mm256_loadu_si256((const __m256i*)wv), H = _mm256_loadu_si256((const __m256i*)hv);
    __m256i           O = _mm256_loadu_si256((const __m256i*)ov);
    __m256            us = _mm256_mul_ps(_mm256_loadu_ps(U), _mm256_loadu_ps(wf));
    __m256            vs = _mm256_mul_ps(_mm256_loadu_ps(V), _mm256_loadu_ps(hf));
    __m256i           X = fs_fixed(us), Y = fs_fixed(vs);
    const int*        base = (const int*)t->pack;
    __m256i           valid = _mm256_set1_epi32(-1);
    if (!bilinear) {
        __m256i x = fs_wrap(_mm256_srai_epi32(X, 16), W, pow2, s->wrap_u, &valid);
        __m256i y = fs_wrap(_mm256_srai_epi32(Y, 16), H, pow2, s->wrap_v, &valid);
        return fs_gather(base, _mm256_add_epi32(O, _mm256_add_epi32(_mm256_mullo_epi32(y, W), x)), valid);
    }
    const __m256i k32768 = _mm256_set1_epi32(32768), one = _mm256_set1_epi32(1), k255 = _mm256_set1_epi32(255), k256 = _mm256_set1_epi32(256);
    X               = _mm256_sub_epi32(X, k32768);
    Y               = _mm256_sub_epi32(Y, k32768);
    __m256i ix      = _mm256_srai_epi32(X, 16), iy = _mm256_srai_epi32(Y, 16);
    __m256i v00 = valid, v01 = valid, v10 = valid, v11 = valid, vx0 = valid, vx1 = valid, vy0 = valid, vy1 = valid;
    __m256i x0 = fs_wrap(ix, W, pow2, s->wrap_u, &vx0), x1 = fs_wrap(_mm256_add_epi32(ix, one), W, pow2, s->wrap_u, &vx1);
    __m256i y0 = fs_wrap(iy, H, pow2, s->wrap_v, &vy0), y1 = fs_wrap(_mm256_add_epi32(iy, one), H, pow2, s->wrap_v, &vy1);
    v00 = _mm256_and_si256(vx0, vy0), v01 = _mm256_and_si256(vx1, vy0), v10 = _mm256_and_si256(vx0, vy1), v11 = _mm256_and_si256(vx1, vy1);
    __m256i r0 = _mm256_add_epi32(O, _mm256_mullo_epi32(y0, W)), r1 = _mm256_add_epi32(O, _mm256_mullo_epi32(y1, W));
    __m256i p00 = fs_gather(base, _mm256_add_epi32(r0, x0), v00), p01 = fs_gather(base, _mm256_add_epi32(r0, x1), v01);
    __m256i p10 = fs_gather(base, _mm256_add_epi32(r1, x0), v10), p11 = fs_gather(base, _mm256_add_epi32(r1, x1), v11);
    __m256i fx = _mm256_and_si256(_mm256_srli_epi32(X, 8), k255), fy = _mm256_and_si256(_mm256_srli_epi32(Y, 8), k255);
    __m256i ixw = _mm256_sub_epi32(k256, fx), iyw = _mm256_sub_epi32(k256, fy);
    __m256i o   = fs_chan(p00, p01, p10, p11, fx, ixw, fy, iyw, 0);
    o           = _mm256_or_si256(o, _mm256_slli_epi32(fs_chan(p00, p01, p10, p11, fx, ixw, fy, iyw, 8), 8));
    o           = _mm256_or_si256(o, _mm256_slli_epi32(fs_chan(p00, p01, p10, p11, fx, ixw, fy, iyw, 16), 16));
    o           = _mm256_or_si256(o, _mm256_slli_epi32(fs_chan(p00, p01, p10, p11, fx, ixw, fy, iyw, 24), 24));
    return o;
}

FM_INLINE __m256 fs_byte(__m256i p, int sh) { return _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, sh), _mm256_set1_epi32(255))); }

int fm3d_sample16_avx2(const fm3d_texture* t, const fm3d_sampler* s, const float* U, const float* V, float* r, float* g, float* b,
                       float* a)
{
    if (!t->pack || (unsigned)s->wrap_u > 3u || (unsigned)s->wrap_v > 3u) return 0;
    fm3d_filter f    = s->filter;
    int         mip  = f >= FM3D_FILTER_NEAREST_MIPMAP && t->levels > 1;
    int         bil  = !(f == FM3D_FILTER_NEAREST || f == FM3D_FILTER_NEAREST_MIPMAP);
    float       maxl = (float)(t->levels - 1);
    float       W0 = (float)t->level[0]->width, H0 = (float)t->level[0]->height;
    int         la[4], lb[4], any_b = 0;
    float       fw[4];
    for (int q = 0; q < 4; q++) { /* per quad level of detail (fm3d__sample_quads' formula) */
        la[q] = lb[q] = 0;
        fw[q]         = 0.0f;
        if (!mip) continue;
        int   i0 = 2 * q, i1 = i0 + 1, i2 = 8 + i0;
        float dudx = (U[i1] - U[i0]) * W0, dvdx = (V[i1] - V[i0]) * H0;
        float dudy = (U[i2] - U[i0]) * W0, dvdy = (V[i2] - V[i0]) * H0;
        float rho2 = FM_MAX(dudx * dudx + dvdx * dvdx, dudy * dudy + dvdy * dvdy);
        float lod  = (rho2 > 0.0f ? 0.5f * fm3d_log2_fast(rho2) : -100.0f) + s->lod_bias;
        if (f == FM3D_FILTER_TRILINEAR) {
            if (lod > 0.0f) {
                float l = FM_MIN(lod, maxl);
                la[q]   = (int)fm_ffloor(l);
                lb[q]   = FM_MIN(la[q] + 1, t->levels - 1);
                fw[q]   = l - (float)la[q];
            }
        } else {
            la[q] = lb[q] = (int)fm_ffloor(FM_CLAMP(lod + 0.5f, 0.0f, maxl));
        }
        any_b |= fw[q] != 0.0f;
    }
    const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f), k255 = _mm256_set1_ps(255.0f), zero = _mm256_setzero_ps();
    __m256       w = _mm256_setr_ps(fw[0], fw[0], fw[1], fw[1], fw[2], fw[2], fw[3], fw[3]);
    __m256       wpos = _mm256_cmp_ps(w, zero, _CMP_GT_OQ);
    for (int row = 0; row < 2; row++) {
        const float* u  = U + 8 * row;
        const float* v  = V + 8 * row;
        __m256i      pa = fs_level8(t, s, bil, u, v, la);
        __m256       c[4];
        for (int k = 0; k < 4; k++) c[k] = fs_byte(pa, k == 3 ? 24 : 16 - 8 * k);
        if (any_b) { /* premultiplied (or straight) channels blended between the levels */
            __m256i pb = fs_level8(t, s, bil, u, v, lb);
            for (int k = 0; k < 4; k++) {
                __m256 x1 = fs_byte(pb, k == 3 ? 24 : 16 - 8 * k);
                c[k]      = _mm256_blendv_ps(c[k], _mm256_add_ps(c[k], _mm256_mul_ps(_mm256_sub_ps(x1, c[k]), w)), wpos);
            }
        }
        __m256 al = _mm256_mul_ps(c[3], inv255), ia;
        if (t->straight) ia = inv255;
        else ia = _mm256_blendv_ps(zero, _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(al, k255)), _mm256_cmp_ps(al, zero, _CMP_GT_OQ));
        _mm256_storeu_ps(r + 8 * row, _mm256_mul_ps(c[0], ia));
        _mm256_storeu_ps(g + 8 * row, _mm256_mul_ps(c[1], ia));
        _mm256_storeu_ps(b + 8 * row, _mm256_mul_ps(c[2], ia));
        _mm256_storeu_ps(a + 8 * row, al);
    }
    return 1;
}

#endif
