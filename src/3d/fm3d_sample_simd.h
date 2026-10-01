/*
 * fatmap - shared parts of the SIMD quad samplers (fm3d_sample_avx2.c,
 * fm3d_sample_avx512.c): the level of detail of the 4 quads of a 16 lane
 * group, with fm3d__sample_quads' formula in its operation order (FM_MAX /
 * FM_MIN are maxps / minps with the same operands), so every backend
 * selects the same levels and weights. Include after <immintrin.h>.
 */
#ifndef FM3D_SAMPLE_SIMD_H
#define FM3D_SAMPLE_SIMD_H

/* *la, *lb: the levels per quad, *fw: the weight of *lb; returns whether any quad blends two levels */
FM_INLINE int fs_lod4(const fm3d_texture* t, const fm3d_sampler* s, const float* U, const float* V, __m128i* pla, __m128i* plb, __m128* pfw)
{
    fm3d_filter f    = s->filter;
    int         mip  = f >= FM3D_FILTER_NEAREST_MIPMAP && t->levels > 1;
    float       maxl = (float)(t->levels - 1);
    float       W0 = (float)t->level[0]->width, H0 = (float)t->level[0]->height;
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
    *pla = la, *plb = lb, *pfw = fw;
    return any_b;
}

#endif
