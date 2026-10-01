/*
 * fatmap - texture sampling of one 16 lane quad group with AVX-512: both
 * rows in one register (lanes 0..7: row 0, 8..15: row 1). The AVX2
 * sampler's algorithm (fm3d_sample_avx2.c) at twice the width: the same
 * integer and float operations, so every backend renders the same bits.
 */
#include "fm3d_internal.h"

#if FM_FEATURE_SHADERS && defined(__AVX512F__) && defined(__AVX512BW__)

#include <immintrin.h>
#include "fm3d_sample_simd.h"

#define FZ_FLOOR (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)

/* fm_wrap_coord on 16 lanes (n: per lane sizes); border lanes leave *valid */
FM_INLINE __m512i fz_wrap(__m512i x, __m512i n, int pow2, fm_wrap w, __mmask16* valid)
{
    const __m512i zero = _mm512_setzero_si512(), one = _mm512_set1_epi32(1);
    switch (w) {
    case FM_WRAP_REPEAT: {
        if (pow2) return _mm512_and_si512(x, _mm512_sub_epi32(n, one));
        __m512 q = _mm512_roundscale_ps(_mm512_div_ps(_mm512_cvtepi32_ps(x), _mm512_cvtepi32_ps(n)), FZ_FLOOR);
        return _mm512_sub_epi32(x, _mm512_mullo_epi32(_mm512_cvttps_epi32(q), n));
    }
    case FM_WRAP_CLAMP: return _mm512_min_epi32(_mm512_max_epi32(x, zero), _mm512_sub_epi32(n, one));
    case FM_WRAP_MIRROR: {
        __m512i p = _mm512_add_epi32(n, n);
        __m512  q = _mm512_roundscale_ps(_mm512_div_ps(_mm512_cvtepi32_ps(x), _mm512_cvtepi32_ps(p)), FZ_FLOOR);
        __m512i r = _mm512_sub_epi32(x, _mm512_mullo_epi32(_mm512_cvttps_epi32(q), p));
        __m512i m = _mm512_sub_epi32(_mm512_sub_epi32(p, one), r);
        return _mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(n, r), m, r); /* r < n ? r : p - 1 - r */
    }
    default: { /* border: outside reads 0 */
        __mmask16 in = _mm512_cmpge_epi32_mask(x, zero) & _mm512_cmpgt_epi32_mask(n, x);
        *valid &= in;
        return _mm512_maskz_mov_epi32(in, x);
    }
    }
}

/* (int32_t)fm_ffloor(FM_CLAMP(f, -32767, 32767) * 65536 + 0.5f): NaN stays NaN in the clamp, like the C */
FM_INLINE __m512i fz_fixed(__m512 f)
{
    const __m512 lo = _mm512_set1_ps(-32767.0f), hi = _mm512_set1_ps(32767.0f);
    f        = _mm512_mask_mov_ps(f, _mm512_cmp_ps_mask(f, lo, _CMP_LT_OQ), lo);
    f        = _mm512_mask_mov_ps(f, _mm512_cmp_ps_mask(f, hi, _CMP_GT_OQ), hi);
    __m512 t = _mm512_add_ps(_mm512_mul_ps(f, _mm512_set1_ps(65536.0f)), _mm512_set1_ps(0.5f));
    return _mm512_cvttps_epi32(_mm512_roundscale_ps(t, FZ_FLOOR));
}

/* two channels at once in 16 bit lanes (see fs_lerp2) */
FM_INLINE __m512i fz_lerp2(__m512i p00, __m512i p01, __m512i p10, __m512i p11, __m512i fx, __m512i ix, __m512i fy, __m512i iy)
{
    __m512i t = _mm512_srli_epi16(_mm512_add_epi16(_mm512_mullo_epi16(p00, ix), _mm512_mullo_epi16(p01, fx)), 8);
    __m512i u = _mm512_srli_epi16(_mm512_add_epi16(_mm512_mullo_epi16(p10, ix), _mm512_mullo_epi16(p11, fx)), 8);
    return _mm512_srli_epi16(_mm512_add_epi16(_mm512_mullo_epi16(t, iy), _mm512_mullo_epi16(u, fy)), 8);
}

/* base[idx] on the lanes in valid, 0 elsewhere (scalar loads: hardware gathers are microcoded on AMD) */
FM_INLINE __m512i fz_gather(const int* base, __m512i idx, __mmask16 valid)
{
    int32_t i[16];
    _mm512_storeu_si512((void*)i, _mm512_maskz_mov_epi32(valid, idx));
    __m128i q[4];
    for (int k = 0; k < 4; k++) {
        q[k] = _mm_cvtsi32_si128(base[i[4 * k]]);
        q[k] = _mm_insert_epi32(q[k], base[i[4 * k + 1]], 1);
        q[k] = _mm_insert_epi32(q[k], base[i[4 * k + 2]], 2);
        q[k] = _mm_insert_epi32(q[k], base[i[4 * k + 3]], 3);
    }
    __m256i lo = _mm256_inserti128_si256(_mm256_castsi128_si256(q[0]), q[1], 1);
    __m256i hi = _mm256_inserti128_si256(_mm256_castsi128_si256(q[2]), q[3], 1);
    return _mm512_maskz_mov_epi32(valid, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
}

/* texel pairs (i, i + 1) of 16 lanes: *lo the texels at i, *hi at i + 1 */
FM_INLINE void fz_pairs(const int* base, __m512i idx, __m512i* lo, __m512i* hi)
{
    int32_t ib[16];
    _mm512_storeu_si512((void*)ib, idx);
    const volatile int32_t* i = ib; /* reloaded with loads (3+ per cycle), not 16 vector -> GPR extracts (1 per cycle) */
#define FZ_P2(a, b) _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(base + i[a]))), (const __m64*)(base + i[b])))
    __m256i a0 = _mm256_inserti128_si256(_mm256_castsi128_si256(FZ_P2(0, 1)), FZ_P2(4, 5), 1);
    __m256i a1 = _mm256_inserti128_si256(_mm256_castsi128_si256(FZ_P2(8, 9)), FZ_P2(12, 13), 1);
    __m256i b0 = _mm256_inserti128_si256(_mm256_castsi128_si256(FZ_P2(2, 3)), FZ_P2(6, 7), 1);
    __m256i b1 = _mm256_inserti128_si256(_mm256_castsi128_si256(FZ_P2(10, 11)), FZ_P2(14, 15), 1);
#undef FZ_P2
    __m512 A = _mm512_castsi512_ps(_mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1));
    __m512 B = _mm512_castsi512_ps(_mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1));
    *lo      = _mm512_castps_si512(_mm512_shuffle_ps(A, B, _MM_SHUFFLE(2, 0, 2, 0)));
    *hi      = _mm512_castps_si512(_mm512_shuffle_ps(A, B, _MM_SHUFFLE(3, 1, 3, 1)));
}

/* the texel pairs of both pixel rows (8 lane halves) from one 16 texel window per texel
 * row: every lane of a half on the same two texel rows of one level, its pair columns xp
 * spanning at most 15 texels (the common case, a texture near 1 texel per pixel). Four
 * masked loads (never past the row) and four permutes instead of 64 pair loads; 0: not
 * applicable */
FM_INLINE int fz_window(const int* base, __m512i r0, __m512i r1, __m512i xp, __m512i W, __m512i* lo0, __m512i* hi0, __m512i* lo1,
                        __m512i* hi1)
{
    const __m512i bh = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8);
    __m512i       r0b = _mm512_permutexvar_epi32(bh, r0), r1b = _mm512_permutexvar_epi32(bh, r1), Wb = _mm512_permutexvar_epi32(bh, W);
    if ((__mmask16)(_mm512_cmpeq_epi32_mask(r0, r0b) & _mm512_cmpeq_epi32_mask(r1, r1b) & _mm512_cmpeq_epi32_mask(W, Wb)) != 0xFFFF)
        return 0;
    const __m512i p1 = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i p2 = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    const __m512i p4 = _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
    __m512i       mn = _mm512_min_epi32(xp, _mm512_permutexvar_epi32(p1, xp)), mx = _mm512_max_epi32(xp, _mm512_permutexvar_epi32(p1, xp));
    mn = _mm512_min_epi32(mn, _mm512_permutexvar_epi32(p2, mn)), mx = _mm512_max_epi32(mx, _mm512_permutexvar_epi32(p2, mx));
    mn = _mm512_min_epi32(mn, _mm512_permutexvar_epi32(p4, mn)), mx = _mm512_max_epi32(mx, _mm512_permutexvar_epi32(p4, mx));
    if (_mm512_cmpgt_epi32_mask(_mm512_sub_epi32(mx, mn), _mm512_set1_epi32(14))) return 0;
    __m128i lo = _mm512_castsi512_si128(mn), hi = _mm512_extracti32x4_epi32(mn, 2);
    int     m0 = _mm_cvtsi128_si32(lo), m1 = _mm_cvtsi128_si32(hi);
    int     w0 = _mm_cvtsi128_si32(_mm512_castsi512_si128(W)), w1 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(W, 2));
    int     a0 = _mm_cvtsi128_si32(_mm512_castsi512_si128(r0)), a1 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(r0, 2));
    int     b0 = _mm_cvtsi128_si32(_mm512_castsi512_si128(r1)), b1 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(r1, 2));
    int     n0 = w0 - m0 < 16 ? w0 - m0 : 16, n1 = w1 - m1 < 16 ? w1 - m1 : 16; /* the row's texels from the window start */
    __mmask16 k0 = (__mmask16)((1u << n0) - 1u), k1 = (__mmask16)((1u << n1) - 1u);
    __m512i A0 = _mm512_maskz_loadu_epi32(k0, base + a0 + m0), B0 = _mm512_maskz_loadu_epi32(k1, base + a1 + m1);
    __m512i A1 = _mm512_maskz_loadu_epi32(k0, base + b0 + m0), B1 = _mm512_maskz_loadu_epi32(k1, base + b1 + m1);
    __m512i i  = _mm512_add_epi32(_mm512_sub_epi32(xp, mn), _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16));
    __m512i i1 = _mm512_add_epi32(i, _mm512_set1_epi32(1));
    *lo0 = _mm512_permutex2var_epi32(A0, i, B0), *hi0 = _mm512_permutex2var_epi32(A0, i1, B0);
    *lo1 = _mm512_permutex2var_epi32(A1, i, B1), *hi1 = _mm512_permutex2var_epi32(A1, i1, B1);
    return 1;
}

/* the per lane level sizes / offsets of one level choice per quad (lanes 2q, 2q + 1 and 8 + 2q, 9 + 2q) */
typedef struct fz_lv {
    __m512i W, H, O;
    __m512  wf, hf;
} fz_lv;

FM_INLINE __m512i fz_dup(const int32_t* tab, const int32_t* q)
{
    __m128i v = _mm_setr_epi32(tab[q[0]], tab[q[1]], tab[q[2]], tab[q[3]]);
    return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3), _mm512_castsi128_si512(v));
}

FM_INLINE fz_lv fz_levels(const fm3d_texture* t, __m128i lv)
{
    int32_t q[4];
    _mm_storeu_si128((__m128i*)q, lv);
    fz_lv L;
    L.W  = fz_dup(t->pw, q);
    L.H  = fz_dup(t->ph, q);
    L.O  = fz_dup(t->poff, q);
    L.wf = _mm512_cvtepi32_ps(L.W), L.hf = _mm512_cvtepi32_ps(L.H);
    return L;
}

/* 16 lanes at the levels in L */
FM_INLINE __m512i fz_level16(const fm3d_texture* t, const fm3d_sampler* s, int bilinear, int pow2, const float* U, const float* V, const fz_lv* L)
{
    const __m512i W = L->W, H = L->H, O = L->O;
    __m512i       X = fz_fixed(_mm512_mul_ps(_mm512_loadu_ps(U), L->wf)), Y = fz_fixed(_mm512_mul_ps(_mm512_loadu_ps(V), L->hf));
    const int*    base  = (const int*)t->pack;
    __mmask16     valid = 0xFFFF;
    if (!bilinear) {
        __m512i x = fz_wrap(_mm512_srai_epi32(X, 16), W, pow2, s->wrap_u, &valid);
        __m512i y = fz_wrap(_mm512_srai_epi32(Y, 16), H, pow2, s->wrap_v, &valid);
        return fz_gather(base, _mm512_add_epi32(O, _mm512_add_epi32(_mm512_mullo_epi32(y, W), x)), valid);
    }
    const __m512i k32768 = _mm512_set1_epi32(32768), one = _mm512_set1_epi32(1), k255 = _mm512_set1_epi32(255), k256 = _mm512_set1_epi32(256);
    X                    = _mm512_sub_epi32(X, k32768);
    Y                    = _mm512_sub_epi32(Y, k32768);
    __m512i   ix = _mm512_srai_epi32(X, 16), iy = _mm512_srai_epi32(Y, 16);
    __mmask16 vx0 = valid, vx1 = valid, vy0 = valid, vy1 = valid;
    __m512i   x0 = fz_wrap(ix, W, pow2, s->wrap_u, &vx0), x1 = fz_wrap(_mm512_add_epi32(ix, one), W, pow2, s->wrap_u, &vx1);
    __m512i   y0 = fz_wrap(iy, H, pow2, s->wrap_v, &vy0), y1 = fz_wrap(_mm512_add_epi32(iy, one), H, pow2, s->wrap_v, &vy1);
    __mmask16 v00 = vx0 & vy0, v01 = vx1 & vy0, v10 = vx0 & vy1, v11 = vx1 & vy1;
    __m512i   r0 = _mm512_add_epi32(O, _mm512_mullo_epi32(y0, W)), r1 = _mm512_add_epi32(O, _mm512_mullo_epi32(y1, W));
    __m512i   p00, p01, p10, p11;
    if (!_mm512_cmpgt_epi32_mask(_mm512_set1_epi32(2), W)) { /* pair loads (see fs_level8) */
        __m512i   xa = _mm512_mask_blend_epi32(vx0, x1, x0), xb = _mm512_mask_blend_epi32(vx1, x0, x1);
        __m512i   xp = _mm512_max_epi32(_mm512_min_epi32(xa, _mm512_sub_epi32(W, _mm512_set1_epi32(2))), _mm512_setzero_si512());
        __m512i   d0 = _mm512_sub_epi32(xa, xp), d1 = _mm512_sub_epi32(xb, xp);
        __mmask16 s0 = _mm512_cmpeq_epi32_mask(d0, one), s1 = _mm512_cmpeq_epi32_mask(d1, one);
        __mmask16 seam = ~_mm512_cmpeq_epi32_mask(_mm512_andnot_si512(one, d1), _mm512_setzero_si512()); /* d1 not in {0, 1} */
        __m512i   lo0, hi0, lo1, hi1;
        if (!fz_window(base, r0, r1, xp, W, &lo0, &hi0, &lo1, &hi1)) {
            fz_pairs(base, _mm512_add_epi32(r0, xp), &lo0, &hi0);
            fz_pairs(base, _mm512_add_epi32(r1, xp), &lo1, &hi1);
        }
        p00 = _mm512_maskz_mov_epi32(v00, _mm512_mask_blend_epi32(s0, lo0, hi0)), p01 = _mm512_mask_blend_epi32(s1, lo0, hi0);
        p10 = _mm512_maskz_mov_epi32(v10, _mm512_mask_blend_epi32(s0, lo1, hi1)), p11 = _mm512_mask_blend_epi32(s1, lo1, hi1);
        if (seam) {
            int32_t a0[16], a1[16], i0[16], i1[16];
            _mm512_storeu_si512((void*)a0, p01), _mm512_storeu_si512((void*)a1, p11);
            _mm512_storeu_si512((void*)i0, _mm512_add_epi32(r0, xb)), _mm512_storeu_si512((void*)i1, _mm512_add_epi32(r1, xb));
            for (int l = 0; l < 16; l++)
                if (seam >> l & 1) a0[l] = base[i0[l]], a1[l] = base[i1[l]];
            p01 = _mm512_loadu_si512((const void*)a0), p11 = _mm512_loadu_si512((const void*)a1);
        }
        p01 = _mm512_maskz_mov_epi32(v01, p01), p11 = _mm512_maskz_mov_epi32(v11, p11);
    } else {
        p00 = fz_gather(base, _mm512_add_epi32(r0, x0), v00), p01 = fz_gather(base, _mm512_add_epi32(r0, x1), v01);
        p10 = fz_gather(base, _mm512_add_epi32(r1, x0), v10), p11 = fz_gather(base, _mm512_add_epi32(r1, x1), v11);
    }
    __m512i fx = _mm512_and_si512(_mm512_srli_epi32(X, 8), k255), fy = _mm512_and_si512(_mm512_srli_epi32(Y, 8), k255);
    __m512i ixw = _mm512_sub_epi32(k256, fx), iyw = _mm512_sub_epi32(k256, fy);
    fx  = _mm512_or_si512(fx, _mm512_slli_epi32(fx, 16)), fy = _mm512_or_si512(fy, _mm512_slli_epi32(fy, 16));
    ixw = _mm512_or_si512(ixw, _mm512_slli_epi32(ixw, 16)), iyw = _mm512_or_si512(iyw, _mm512_slli_epi32(iyw, 16));
    const __m512i m  = _mm512_set1_epi32(0x00ff00ff);
    __m512i       rb = fz_lerp2(_mm512_and_si512(p00, m), _mm512_and_si512(p01, m), _mm512_and_si512(p10, m), _mm512_and_si512(p11, m), fx, ixw, fy, iyw);
    __m512i       ga = fz_lerp2(_mm512_and_si512(_mm512_srli_epi32(p00, 8), m), _mm512_and_si512(_mm512_srli_epi32(p01, 8), m),
                                _mm512_and_si512(_mm512_srli_epi32(p10, 8), m), _mm512_and_si512(_mm512_srli_epi32(p11, 8), m), fx, ixw, fy, iyw);
    return _mm512_or_si512(rb, _mm512_slli_epi32(ga, 8));
}

FM_INLINE __m512 fz_byte(__m512i p, int sh) { return _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(p, sh), _mm512_set1_epi32(255))); }

int fm3d_sample16_avx512(const fm3d_texture* t, const fm3d_sampler* s, const float* U, const float* V, float* r, float* g, float* b,
                         float* a)
{
    if (!t->pack || (unsigned)s->wrap_u > 3u || (unsigned)s->wrap_v > 3u) return 0;
    fm3d_filter f   = s->filter;
    int         bil = !(f == FM3D_FILTER_NEAREST || f == FM3D_FILTER_NEAREST_MIPMAP);
    __m128i     la, lb;
    __m128      fw;
    int         any_b = fs_lod4(t, s, U, V, &la, &lb, &fw);
    const fm_surface* L0   = t->level[0];
    int               pow2 = !(L0->width & (L0->width - 1)) && !(L0->height & (L0->height - 1));
    fz_lv             LA   = fz_levels(t, la);
    __m512i           pa   = fz_level16(t, s, bil, pow2, U, V, &LA);
    __m512            c[4];
    for (int k = 0; k < 4; k++) c[k] = fz_byte(pa, k == 3 ? 24 : 16 - 8 * k);
    if (any_b) { /* premultiplied (or straight) channels blended between the levels */
        const __m512 zero = _mm512_setzero_ps();
        fz_lv        LB   = fz_levels(t, lb);
        __m512i      pb   = fz_level16(t, s, bil, pow2, U, V, &LB);
        __m512       w    = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3), _mm512_castps128_ps512(fw));
        __mmask16    wpos = _mm512_cmp_ps_mask(w, zero, _CMP_GT_OQ);
        for (int k = 0; k < 4; k++) {
            __m512 x1 = fz_byte(pb, k == 3 ? 24 : 16 - 8 * k);
            c[k]      = _mm512_mask_mov_ps(c[k], wpos, _mm512_add_ps(c[k], _mm512_mul_ps(_mm512_sub_ps(x1, c[k]), w)));
        }
    }
    const __m512 inv255 = _mm512_set1_ps(1.0f / 255.0f);
    __m512       al = _mm512_mul_ps(c[3], inv255), ia;
    if (t->straight) ia = inv255;
    else
        ia = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(al, _mm512_setzero_ps(), _CMP_GT_OQ),
                                 _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_mul_ps(al, _mm512_set1_ps(255.0f))));
    _mm512_storeu_ps(r, _mm512_mul_ps(c[0], ia));
    _mm512_storeu_ps(g, _mm512_mul_ps(c[1], ia));
    _mm512_storeu_ps(b, _mm512_mul_ps(c[2], ia));
    _mm512_storeu_ps(a, al);
    return 1;
}

#endif
