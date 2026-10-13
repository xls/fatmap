/*
 * fatmap - AVX-512 kernels: the AVX2 table with the entries below replaced
 * (fm__kernels_avx512, from fm_cpu.c). Same results as the scalar reference
 * (fm_kernels_common.h), like every backend.
 */
#include "fm_internal.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)

#include <immintrin.h>
#include "fm_kernels_common.h"

/* fm_ucmp on 16 lanes (unsigned) */
FM_INLINE __mmask16 fk_cmp(int func, __m512i a, __m512i b)
{
    switch (func) {
    case 0: return 0;
    case 1: return _mm512_cmplt_epu32_mask(a, b);
    case 2: return _mm512_cmpeq_epu32_mask(a, b);
    case 3: return _mm512_cmple_epu32_mask(a, b);
    case 4: return _mm512_cmpgt_epu32_mask(a, b);
    case 5: return _mm512_cmpneq_epu32_mask(a, b);
    case 6: return _mm512_cmpge_epu32_mask(a, b);
    default: return 0xFFFF;
    }
}

/* fm_stencil_op1 on 16 lanes (s, ref in 0..255) */
FM_INLINE __m512i fk_sop(int op, __m512i s, __m512i ref)
{
    const __m512i ff = _mm512_set1_epi32(255), one = _mm512_set1_epi32(1);
    switch (op) {
    case 1: return _mm512_setzero_si512();
    case 2: return ref;
    case 3: return _mm512_min_epu32(_mm512_add_epi32(s, one), ff);
    case 4: return _mm512_sub_epi32(s, _mm512_min_epu32(s, one));
    case 5: return _mm512_xor_si512(s, ff);
    case 6: return _mm512_and_si512(_mm512_add_epi32(s, one), ff);
    case 7: return _mm512_and_si512(_mm512_sub_epi32(s, one), ff);
    default: return s;
    }
}

/* zs_d24s8_rows: 16 span pixels per step, the plane's z in registers (the template's
 * expression: ((float)x + 0.5f - x0) * b + a, no contraction), the key in double like
 * fm_zkey24; only pixels whose word changes are stored */
static void fk_zs_d24s8_rows(const fm_zs24_rows* r, const fm_zs24* p, uint32_t* wmin, uint32_t* wmax, int* nw)
{
    const __m512i lo24 = _mm512_set1_epi32(0xffffff), ff = _mm512_set1_epi32(255);
    const __m512i ref = _mm512_set1_epi32(p->ref), rm = _mm512_set1_epi32(p->rmask), wm = _mm512_set1_epi32(p->wmask);
    const __m512i rr  = _mm512_set1_epi32(p->ref & p->rmask);
    const __m512i blo = _mm512_set1_epi32((int)p->blo), bhi = _mm512_set1_epi32((int)p->bhi);
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512d k24 = _mm512_set1_pd(16777215.0), half = _mm512_set1_pd(0.5);
    const __m512  zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.0f), h = _mm512_set1_ps(0.5f);
    __m512i       lo = _mm512_set1_epi32(-1), hi = _mm512_setzero_si512(), cv = _mm512_setzero_si512();
    int           zkeys = p->zfunc != 7 || p->zwrite; /* z unused without the test and writes */
    for (int k = 0; k < 2; k++) {
        int x = r->lo[k], n = r->hi[k] - x + 1;
        if (n <= 0) continue;
        if (n > 64) n = 64;
        const __m512 a = _mm512_set1_ps(r->a[k]), b = _mm512_set1_ps(r->bz), x0 = _mm512_set1_ps(r->x0f);
        uint32_t*    row = r->row[k] + x;
        for (int i = 0; i < n; i += 16) {
            __mmask16 act = n - i >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << (n - i)) - 1u);
            __m512i   zk  = _mm512_setzero_si512();
            if (zkeys) {
                __m512 xf = _mm512_add_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(x + i), iota)), h);
                __m512 z  = _mm512_add_ps(a, _mm512_mul_ps(b, _mm512_sub_ps(xf, x0)));
                z         = _mm512_min_ps(_mm512_max_ps(z, zero), one); /* fm_clamp01: NaN -> 0 */
                __m256i k0 = _mm512_cvttpd_epi32(_mm512_add_pd(_mm512_mul_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(z)), k24), half));
                __m256i k1 = _mm512_cvttpd_epi32(
                    _mm512_add_pd(_mm512_mul_pd(_mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(z), 1))), k24), half));
                zk = _mm512_inserti64x4(_mm512_castsi256_si512(k0), k1, 1);
            }
            __m512i   w  = _mm512_maskz_loadu_epi32(act, row + i);
            __m512i   d  = _mm512_and_si512(w, lo24);
            __mmask16 zp = fk_cmp(p->zfunc, zk, d);
            if (p->bounds) /* outside the bounds: discarded before the stencil test */
                act &= (__mmask16)~(_mm512_cmplt_epu32_mask(d, blo) | _mm512_cmpgt_epu32_mask(d, bhi));
            __m512i nwv = w;
            if (p->stencil) {
                __m512i   s  = _mm512_srli_epi32(w, 24);
                __mmask16 sp = fk_cmp(p->sfunc, rr, _mm512_and_si512(s, rm));
                __m512i   ns = _mm512_mask_blend_epi32(
                    sp, fk_sop(p->sfail, s, ref), _mm512_mask_blend_epi32(zp, fk_sop(p->dpfail, s, ref), fk_sop(p->dppass, s, ref)));
                ns  = _mm512_ternarylogic_epi32(wm, ns, s, 0xCA); /* wm ? ns : s */
                nwv = _mm512_or_si512(d, _mm512_slli_epi32(_mm512_and_si512(ns, ff), 24));
                zp &= sp;
            }
            __mmask16 pass = zp & act;
            if (p->zwrite && pass) {
                nwv = _mm512_mask_mov_epi32(nwv, pass, _mm512_or_si512(_mm512_andnot_si512(lo24, nwv), zk));
                lo  = _mm512_mask_min_epu32(lo, pass, lo, zk);
                hi  = _mm512_mask_max_epu32(hi, pass, hi, zk);
                cv  = _mm512_mask_sub_epi32(cv, pass, cv, _mm512_set1_epi32(-1)); /* written pixels per lane */
            }
            _mm512_mask_storeu_epi32(row + i, act & _mm512_cmpneq_epu32_mask(nwv, w), nwv);
        }
    }
    int cnt = _mm512_reduce_add_epi32(cv);
    if (cnt) {
        uint32_t l = (uint32_t)_mm512_reduce_min_epu32(lo), u = (uint32_t)_mm512_reduce_max_epu32(hi);
        if (l < *wmin) *wmin = l;
        if (u > *wmax) *wmax = u;
        *nw += cnt;
    }
}

void fm__kernels_avx512(fm_kernels* k) { k->zs_d24s8_rows = fk_zs_d24s8_rows; }

#endif
