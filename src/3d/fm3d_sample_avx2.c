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

/* the bilinear lerp of two channels at once (16 bit lanes: 0x00ff00ff
 * masked texels, weights in both halves); a * (256 - f) + b * f <= 65280, so
 * every step is the 32 bit lane code's exact integer result */
FM_INLINE __m256i fs_lerp2(__m256i p00, __m256i p01, __m256i p10, __m256i p11, __m256i fx, __m256i ix, __m256i fy, __m256i iy)
{
    __m256i t = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(p00, ix), _mm256_mullo_epi16(p01, fx)), 8);
    __m256i u = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(p10, ix), _mm256_mullo_epi16(p11, fx)), 8);
    return _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(t, iy), _mm256_mullo_epi16(u, fy)), 8);
}

FM_INLINE __m256i fs_gather(const int* base, __m256i idx, __m256i valid)
{
    /* scalar loads: hardware gathers are microcoded on AMD (about 3 cycles per element) */
    int32_t i[8];
    _mm256_storeu_si256((__m256i*)i, _mm256_and_si256(idx, valid));
    __m128i lo = _mm_cvtsi32_si128(base[i[0]]), hi = _mm_cvtsi32_si128(base[i[4]]);
    lo = _mm_insert_epi32(lo, base[i[1]], 1), hi = _mm_insert_epi32(hi, base[i[5]], 1);
    lo = _mm_insert_epi32(lo, base[i[2]], 2), hi = _mm_insert_epi32(hi, base[i[6]], 2);
    lo = _mm_insert_epi32(lo, base[i[3]], 3), hi = _mm_insert_epi32(hi, base[i[7]], 3);
    return _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), valid);
}

/* texel pairs (i, i + 1) of 8 lanes: *lo the texels at i, *hi at i + 1 */
FM_INLINE void fs_pairs(const int* base, __m256i idx, __m256i* lo, __m256i* hi)
{
    int32_t i[8];
    _mm256_storeu_si256((__m256i*)i, idx);
#define FS_P2(a, b) _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(base + i[a]))), (const __m64*)(base + i[b])))
    __m256i A = _mm256_inserti128_si256(_mm256_castsi128_si256(FS_P2(0, 1)), FS_P2(4, 5), 1);
    __m256i B = _mm256_inserti128_si256(_mm256_castsi128_si256(FS_P2(2, 3)), FS_P2(6, 7), 1);
#undef FS_P2
    *lo = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(A), _mm256_castsi256_ps(B), _MM_SHUFFLE(2, 0, 2, 0)));
    *hi = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(A), _mm256_castsi256_ps(B), _MM_SHUFFLE(3, 1, 3, 1)));
}

/* the per lane level sizes / offsets of one level choice per quad (lanes 2q, 2q + 1) */
typedef struct fs_lv {
    __m256i W, H, O;
    __m256  wf, hf;
} fs_lv;

FM_INLINE __m256i fs_dup(const int32_t* tab, const int32_t* q)
{
    __m128i v = _mm_setr_epi32(tab[q[0]], tab[q[1]], tab[q[2]], tab[q[3]]);
    return _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(v), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
}

FM_INLINE fs_lv fs_levels(const fm3d_texture* t, __m128i lv)
{
    int32_t q[4];
    _mm_storeu_si128((__m128i*)q, lv);
    fs_lv L;
    L.W  = fs_dup(t->pw, q);
    L.H  = fs_dup(t->ph, q);
    L.O  = fs_dup(t->poff, q);
    L.wf = _mm256_cvtepi32_ps(L.W), L.hf = _mm256_cvtepi32_ps(L.H);
    return L;
}

/* 8 lanes (one row) at the levels in L */
FM_INLINE __m256i fs_level8(const fm3d_texture* t, const fm3d_sampler* s, int bilinear, int pow2, const float* U, const float* V, const fs_lv* L)
{
    const __m256i W = L->W, H = L->H, O = L->O;
    __m256i       X = fs_fixed(_mm256_mul_ps(_mm256_loadu_ps(U), L->wf)), Y = fs_fixed(_mm256_mul_ps(_mm256_loadu_ps(V), L->hf));
    const int*    base = (const int*)t->pack;
    __m256i       valid = _mm256_set1_epi32(-1);
    if (!bilinear) {
        __m256i x = fs_wrap(_mm256_srai_epi32(X, 16), W, pow2, s->wrap_u, &valid);
        __m256i y = fs_wrap(_mm256_srai_epi32(Y, 16), H, pow2, s->wrap_v, &valid);
        return fs_gather(base, _mm256_add_epi32(O, _mm256_add_epi32(_mm256_mullo_epi32(y, W), x)), valid);
    }
    const __m256i k32768 = _mm256_set1_epi32(32768), one = _mm256_set1_epi32(1), k255 = _mm256_set1_epi32(255), k256 = _mm256_set1_epi32(256);
    X                    = _mm256_sub_epi32(X, k32768);
    Y                    = _mm256_sub_epi32(Y, k32768);
    __m256i ix = _mm256_srai_epi32(X, 16), iy = _mm256_srai_epi32(Y, 16);
    __m256i vx0 = valid, vx1 = valid, vy0 = valid, vy1 = valid;
    __m256i x0 = fs_wrap(ix, W, pow2, s->wrap_u, &vx0), x1 = fs_wrap(_mm256_add_epi32(ix, one), W, pow2, s->wrap_u, &vx1);
    __m256i y0 = fs_wrap(iy, H, pow2, s->wrap_v, &vy0), y1 = fs_wrap(_mm256_add_epi32(iy, one), H, pow2, s->wrap_v, &vy1);
    __m256i v00 = _mm256_and_si256(vx0, vy0), v01 = _mm256_and_si256(vx1, vy0), v10 = _mm256_and_si256(vx0, vy1), v11 = _mm256_and_si256(vx1, vy1);
    __m256i r0 = _mm256_add_epi32(O, _mm256_mullo_epi32(y0, W)), r1 = _mm256_add_epi32(O, _mm256_mullo_epi32(y1, W));
    __m256i p00, p01, p10, p11;
    /* 64 bit pair loads at xp = clamp(x0, 0, W - 2): x0 is xp or xp + 1, so is x1 but at a
     * repeat seam (x1 = 0: those lanes load it on their own). Borders: an outside x takes
     * its neighbor's, the texel is masked to 0. Every address stays inside the level; 1
     * texel wide levels take the gathers. */
    __m256i xa = _mm256_blendv_epi8(x1, x0, vx0), xb = _mm256_blendv_epi8(x0, x1, vx1);
    __m256i xp = _mm256_max_epi32(_mm256_min_epi32(xa, _mm256_sub_epi32(W, _mm256_set1_epi32(2))), _mm256_setzero_si256());
    __m256i d0 = _mm256_sub_epi32(xa, xp), d1 = _mm256_sub_epi32(xb, xp);
    __m256i seam = _mm256_cmpeq_epi32(_mm256_andnot_si256(one, d1), _mm256_setzero_si256()); /* d1 in {0, 1}: no seam */
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(_mm256_set1_epi32(2), W)) == 0) {
        __m256i lo0, hi0, lo1, hi1, s0 = _mm256_cmpeq_epi32(d0, one), s1 = _mm256_cmpeq_epi32(d1, one);
        fs_pairs(base, _mm256_add_epi32(r0, xp), &lo0, &hi0);
        fs_pairs(base, _mm256_add_epi32(r1, xp), &lo1, &hi1);
        p00 = _mm256_and_si256(_mm256_blendv_epi8(lo0, hi0, s0), v00), p01 = _mm256_blendv_epi8(lo0, hi0, s1);
        p10 = _mm256_and_si256(_mm256_blendv_epi8(lo1, hi1, s0), v10), p11 = _mm256_blendv_epi8(lo1, hi1, s1);
        int sm = ~_mm256_movemask_ps(_mm256_castsi256_ps(seam)) & 255;
        if (sm) {
            int32_t a0[8], a1[8], i0[8], i1[8];
            _mm256_storeu_si256((__m256i*)a0, p01), _mm256_storeu_si256((__m256i*)a1, p11);
            _mm256_storeu_si256((__m256i*)i0, _mm256_add_epi32(r0, xb)), _mm256_storeu_si256((__m256i*)i1, _mm256_add_epi32(r1, xb));
            for (int l = 0; l < 8; l++)
                if (sm >> l & 1) a0[l] = base[i0[l]], a1[l] = base[i1[l]];
            p01 = _mm256_loadu_si256((const __m256i*)a0), p11 = _mm256_loadu_si256((const __m256i*)a1);
        }
        p01 = _mm256_and_si256(p01, v01), p11 = _mm256_and_si256(p11, v11);
    } else {
        p00 = fs_gather(base, _mm256_add_epi32(r0, x0), v00), p01 = fs_gather(base, _mm256_add_epi32(r0, x1), v01);
        p10 = fs_gather(base, _mm256_add_epi32(r1, x0), v10), p11 = fs_gather(base, _mm256_add_epi32(r1, x1), v11);
    }
    __m256i fx = _mm256_and_si256(_mm256_srli_epi32(X, 8), k255), fy = _mm256_and_si256(_mm256_srli_epi32(Y, 8), k255);
    __m256i ixw = _mm256_sub_epi32(k256, fx), iyw = _mm256_sub_epi32(k256, fy);
    fx = _mm256_or_si256(fx, _mm256_slli_epi32(fx, 16)), fy = _mm256_or_si256(fy, _mm256_slli_epi32(fy, 16));
    ixw = _mm256_or_si256(ixw, _mm256_slli_epi32(ixw, 16)), iyw = _mm256_or_si256(iyw, _mm256_slli_epi32(iyw, 16));
    const __m256i m = _mm256_set1_epi32(0x00ff00ff);
    __m256i       rb = fs_lerp2(_mm256_and_si256(p00, m), _mm256_and_si256(p01, m), _mm256_and_si256(p10, m), _mm256_and_si256(p11, m), fx, ixw, fy, iyw);
    __m256i       ga = fs_lerp2(_mm256_and_si256(_mm256_srli_epi32(p00, 8), m), _mm256_and_si256(_mm256_srli_epi32(p01, 8), m),
                                _mm256_and_si256(_mm256_srli_epi32(p10, 8), m), _mm256_and_si256(_mm256_srli_epi32(p11, 8), m), fx, ixw, fy, iyw);
    return _mm256_or_si256(rb, _mm256_slli_epi32(ga, 8));
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
    /* per quad level of detail, the 4 quads at once (fm3d__sample_quads' formula, its
     * operation order: FM_MAX / FM_MIN are maxps / minps with the same operands) */
    __m128i la = _mm_setzero_si128(), lb = la;
    __m128  fw = _mm_setzero_ps();
    int     any_b = 0;
    if (mip) {
        const __m128 w0 = _mm_set1_ps(W0), h0 = _mm_set1_ps(H0), zero = _mm_setzero_ps(), mx = _mm_set1_ps(maxl);
        __m128       ua = _mm_loadu_ps(U), ub = _mm_loadu_ps(U + 4), uc = _mm_loadu_ps(U + 8), ud = _mm_loadu_ps(U + 12);
        __m128       va = _mm_loadu_ps(V), vb = _mm_loadu_ps(V + 4), vc = _mm_loadu_ps(V + 8), vd = _mm_loadu_ps(V + 12);
        __m128       u0 = _mm_shuffle_ps(ua, ub, _MM_SHUFFLE(2, 0, 2, 0)), u1 = _mm_shuffle_ps(ua, ub, _MM_SHUFFLE(3, 1, 3, 1));
        __m128       v0 = _mm_shuffle_ps(va, vb, _MM_SHUFFLE(2, 0, 2, 0)), v1 = _mm_shuffle_ps(va, vb, _MM_SHUFFLE(3, 1, 3, 1));
        __m128       u2 = _mm_shuffle_ps(uc, ud, _MM_SHUFFLE(2, 0, 2, 0)), v2 = _mm_shuffle_ps(vc, vd, _MM_SHUFFLE(2, 0, 2, 0));
        __m128       dudx = _mm_mul_ps(_mm_sub_ps(u1, u0), w0), dvdx = _mm_mul_ps(_mm_sub_ps(v1, v0), h0);
        __m128       dudy = _mm_mul_ps(_mm_sub_ps(u2, u0), w0), dvdy = _mm_mul_ps(_mm_sub_ps(v2, v0), h0);
        __m128 rho2 = _mm_max_ps(_mm_add_ps(_mm_mul_ps(dudx, dudx), _mm_mul_ps(dvdx, dvdx)), _mm_add_ps(_mm_mul_ps(dudy, dudy), _mm_mul_ps(dvdy, dvdy)));
        /* fm3d_log2_fast */
        __m128i ri = _mm_castps_si128(rho2);
        __m128  e  = _mm_sub_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(ri, 23), _mm_set1_epi32(255))), _mm_set1_ps(127.0f));
        __m128  m  = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(ri, _mm_set1_epi32(0x007fffff)), _mm_set1_epi32(0x3f800000))), _mm_set1_ps(1.0f));
        __m128  lg = _mm_add_ps(e, _mm_mul_ps(m, _mm_sub_ps(_mm_set1_ps(1.3465f), _mm_mul_ps(_mm_set1_ps(0.3465f), m))));
        __m128  lod = _mm_add_ps(_mm_blendv_ps(_mm_set1_ps(-100.0f), _mm_mul_ps(_mm_set1_ps(0.5f), lg), _mm_cmpgt_ps(rho2, zero)), _mm_set1_ps(s->lod_bias));
        if (f == FM3D_FILTER_TRILINEAR) {
            __m128  pos = _mm_cmpgt_ps(lod, zero); /* NaN: level 0 */
            __m128  l   = _mm_min_ps(lod, mx);
            __m128i li  = _mm_cvttps_epi32(l); /* l > 0: the floor */
            la = _mm_and_si128(li, _mm_castps_si128(pos));
            lb = _mm_and_si128(_mm_min_epi32(_mm_add_epi32(li, _mm_set1_epi32(1)), _mm_set1_epi32(t->levels - 1)), _mm_castps_si128(pos));
            fw = _mm_and_ps(_mm_sub_ps(l, _mm_cvtepi32_ps(li)), pos);
            any_b = _mm_movemask_ps(_mm_cmpneq_ps(fw, zero)) != 0;
        } else {
            /* FM_CLAMP(lod + 0.5, 0, maxl) (NaN: level 0) */
            la = lb = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(_mm_add_ps(lod, _mm_set1_ps(0.5f)), zero), mx));
        }
    }
    const fm_surface* L0   = t->level[0];
    int               pow2 = !(L0->width & (L0->width - 1)) && !(L0->height & (L0->height - 1));
    fs_lv             LA = fs_levels(t, la), LB;
    if (any_b) LB = fs_levels(t, lb);
    const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f), k255 = _mm256_set1_ps(255.0f), zero = _mm256_setzero_ps();
    __m256       w = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(fw), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
    __m256       wpos = _mm256_cmp_ps(w, zero, _CMP_GT_OQ);
    for (int row = 0; row < 2; row++) {
        const float* u  = U + 8 * row;
        const float* v  = V + 8 * row;
        __m256i      pa = fs_level8(t, s, bil, pow2, u, v, &LA);
        __m256       c[4];
        for (int k = 0; k < 4; k++) c[k] = fs_byte(pa, k == 3 ? 24 : 16 - 8 * k);
        if (any_b) { /* premultiplied (or straight) channels blended between the levels */
            __m256i pb = fs_level8(t, s, bil, pow2, u, v, &LB);
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
