/* fatmap - AVX2 backend (8 pixels per block). Compiled with -mavx2 / /arch:AVX2,
 * only selected at runtime when the CPU and OS support AVX2. */
#include "fm_internal.h"

#if FM_ARCH_X86

#define FMK(name) name##_avx2
#define FMK_TABLE fm_kernels_avx2
#define FMK_LEVEL FM_SIMD_AVX2
#define FMK_PX    8
#define FMK_ZS24_WIDE /* zs_d24s8 below */

#include "fm_kernels_x86.h"
#include <immintrin.h>

typedef __m256i vpx;
typedef __m256i vw;

FM_INLINE vpx  vpx_load(const void* p) { return _mm256_loadu_si256((const __m256i*)p); }
FM_INLINE void vpx_store(void* p, vpx v) { _mm256_storeu_si256((__m256i*)p, v); }
FM_INLINE vpx  vpx_set1(uint32_t v) { return _mm256_set1_epi32((int)v); }
FM_INLINE vpx  vpx_adds8(vpx a, vpx b) { return _mm256_adds_epu8(a, b); } /* per byte min(a + b, 255) */
/* unpack / pack work within 128-bit lanes; mask_load builds the same order */
FM_INLINE vw  vw_lo(vpx p) { return _mm256_unpacklo_epi8(p, _mm256_setzero_si256()); }
FM_INLINE vw  vw_hi(vpx p) { return _mm256_unpackhi_epi8(p, _mm256_setzero_si256()); }
FM_INLINE vpx vw_pack(vw lo, vw hi) { return _mm256_packus_epi16(lo, hi); }
FM_INLINE vpx vpx_mask_load(const uint8_t* m)
{
    __m128i x  = _mm_loadl_epi64((const __m128i*)m);
    x          = _mm_unpacklo_epi8(x, x);
    __m128i lo = _mm_unpacklo_epi16(x, x);
    __m128i hi = _mm_unpackhi_epi16(x, x);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}
/* partial blocks (r < 8 pixels) without copies: masked loads / stores touch only the r
 * pixels (no reads past the span, no writes to pixels other threads own) */
#define FMK_HAVE_PARTIAL 1
FM_INLINE __m256i vpx_lanes(int r) { return _mm256_cmpgt_epi32(_mm256_set1_epi32(r), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)); }
FM_INLINE vpx     vpx_load_n(const uint32_t* p, int r) { return _mm256_maskload_epi32((const int*)p, vpx_lanes(r)); }
FM_INLINE void    vpx_store_n(uint32_t* p, vpx v, int r) { _mm256_maskstore_epi32((int*)p, vpx_lanes(r), v); }
FM_INLINE vpx     vpx_mask_load_n(const uint8_t* m, int r)
{
    uint64_t b = 0;
    for (int k = 0; k < r; k++) b |= (uint64_t)m[k] << (8 * k);
    return vpx_mask_load((const uint8_t*)&b);
}
FM_INLINE vw vw_set1(uint16_t v) { return _mm256_set1_epi16((short)v); }
FM_INLINE vw vw_add(vw a, vw b) { return _mm256_add_epi16(a, b); }
FM_INLINE vw vw_sub(vw a, vw b) { return _mm256_sub_epi16(a, b); }
FM_INLINE vw vw_mul(vw a, vw b) { return _mm256_mullo_epi16(a, b); }
FM_INLINE vw vw_min(vw a, vw b) { return _mm256_min_epu16(a, b); }
FM_INLINE vw vw_max(vw a, vw b) { return _mm256_max_epu16(a, b); }
FM_INLINE vw vw_div255(vw a)
{
    return _mm256_mulhi_epu16(_mm256_add_epi16(a, _mm256_set1_epi16(128)), _mm256_set1_epi16(257));
}
FM_INLINE vw vw_shr8(vw a) { return _mm256_srli_epi16(a, 8); }
FM_INLINE vw vw_alpha(vw a)
{
    return _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(a, _MM_SHUFFLE(3, 3, 3, 3)), _MM_SHUFFLE(3, 3, 3, 3));
}
FM_INLINE vw  vw_alpha_merge(vw c, vw a) { return _mm256_blend_epi16(c, a, 0x88); }
FM_INLINE int fmk_mask_zero(const uint8_t* m)
{
    uint64_t w;
    memcpy(&w, m, 8);
    return w == 0;
}
FM_INLINE int fmk_mask_full(const uint8_t* m)
{
    uint64_t w;
    memcpy(&w, m, 8);
    return w == ~(uint64_t)0;
}
FM_INLINE int vpx_all_opaque(vpx v)
{
    __m256i a = _mm256_srli_epi32(v, 24);
    return _mm256_movemask_epi8(_mm256_cmpeq_epi32(a, _mm256_set1_epi32(255))) == -1;
}
FM_INLINE int vpx_all_zero(vpx v) { return _mm256_testz_si256(v, v); }

/* ---- zs_d24s8, 8 pixels per step ---- */
FM_INLINE __m256i fmy_icmp(int func, __m256i a, __m256i b) /* values below 2^31 */
{
    const __m256i ones = _mm256_set1_epi32(-1);
    switch (func) {
    case 0: return _mm256_setzero_si256();
    case 1: return _mm256_cmpgt_epi32(b, a);
    case 2: return _mm256_cmpeq_epi32(a, b);
    case 3: return _mm256_xor_si256(_mm256_cmpgt_epi32(a, b), ones);
    case 4: return _mm256_cmpgt_epi32(a, b);
    case 5: return _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), ones);
    case 6: return _mm256_xor_si256(_mm256_cmpgt_epi32(b, a), ones);
    default: return ones;
    }
}

FM_INLINE __m256i fmy_stencil_op(int op, __m256i s, __m256i ref)
{
    const __m256i ff = _mm256_set1_epi32(255), one = _mm256_set1_epi32(1);
    switch (op) {
    case 1: return _mm256_setzero_si256();
    case 2: return ref;
    case 3: return _mm256_min_epu32(_mm256_add_epi32(s, one), ff);
    case 4: return _mm256_sub_epi32(s, _mm256_min_epu32(s, one));
    case 5: return _mm256_xor_si256(s, ff);
    case 6: return _mm256_and_si256(_mm256_add_epi32(s, one), ff);
    case 7: return _mm256_and_si256(_mm256_sub_epi32(s, one), ff);
    default: return s;
    }
}

static void FMK(zs_d24s8)(const float* z, uint32_t* zb, uint8_t* m, int n, const fm_zs24* p, uint32_t* wmin,
                          uint32_t* wmax, int* nw)
{
    const __m256i lo24 = _mm256_set1_epi32(0xffffff), ff = _mm256_set1_epi32(255);
    const __m256i ref = _mm256_set1_epi32(p->ref), rm = _mm256_set1_epi32(p->rmask), wm = _mm256_set1_epi32(p->wmask);
    const __m256i rr  = _mm256_set1_epi32(p->ref & p->rmask);
    const __m256i blo = _mm256_set1_epi32((int)p->blo), bhi = _mm256_set1_epi32((int)p->bhi);
    const __m256d k24 = _mm256_set1_pd(16777215.0), half = _mm256_set1_pd(0.5);
    const __m256  zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    __m256i       lo = _mm256_set1_epi32(-1), hi = _mm256_setzero_si256();
    int           cnt = 0, i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t mw;
        memcpy(&mw, m + i, 8);
        if (!mw) continue;
        __m256i act = _mm256_cmpgt_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(m + i))), _mm256_setzero_si256());
        __m256  zc  = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(z + i), zero), one); /* fmx_clamp01: NaN -> 0 */
        /* the key in double precision like fm_zkey24 (exact: 24 x 24 bit product) */
        __m128i k0  = _mm256_cvttpd_epi32(_mm256_add_pd(_mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(zc)), k24), half));
        __m128i k1  = _mm256_cvttpd_epi32(_mm256_add_pd(_mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(zc, 1)), k24), half));
        __m256i zk  = _mm256_inserti128_si256(_mm256_castsi128_si256(k0), k1, 1);
        __m256i w   = _mm256_loadu_si256((const __m256i*)(zb + i));
        __m256i zp  = fmy_icmp(p->zfunc, zk, _mm256_and_si256(w, lo24));
        int     am  = _mm256_movemask_ps(_mm256_castsi256_ps(act)); /* the covered pixels */
        if (p->bounds) { /* outside the bounds: discarded before the stencil test */
            __m256i d = _mm256_and_si256(w, lo24);
            act       = _mm256_andnot_si256(_mm256_or_si256(_mm256_cmpgt_epi32(blo, d), _mm256_cmpgt_epi32(d, bhi)), act);
        }
        __m256i nwv = w;
        if (p->stencil) {
            __m256i s  = _mm256_srli_epi32(w, 24);
            __m256i sp = fmy_icmp(p->sfunc, rr, _mm256_and_si256(s, rm));
            __m256i ns = _mm256_blendv_epi8(fmy_stencil_op(p->sfail, s, ref),
                                            _mm256_blendv_epi8(fmy_stencil_op(p->dpfail, s, ref), fmy_stencil_op(p->dppass, s, ref), zp), sp);
            ns         = _mm256_or_si256(_mm256_and_si256(ns, wm), _mm256_andnot_si256(wm, s));
            nwv        = _mm256_or_si256(_mm256_and_si256(w, lo24), _mm256_slli_epi32(_mm256_and_si256(ns, ff), 24));
            zp         = _mm256_and_si256(zp, sp);
        }
        __m256i pass = _mm256_and_si256(zp, act);
        int     pm   = _mm256_movemask_ps(_mm256_castsi256_ps(pass));
        if (pm != am) { /* failed covered pixels clear their bytes */
            __m256i  b8   = _mm256_packs_epi16(_mm256_packs_epi32(pass, pass), _mm256_setzero_si256());
            uint64_t keep = (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(b8)) |
                            ((uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(b8, 1)) << 32);
            mw &= keep;
            memcpy(m + i, &mw, 8);
        }
        if (p->zwrite && pm) {
            nwv = _mm256_blendv_epi8(nwv, _mm256_or_si256(_mm256_andnot_si256(lo24, nwv), zk), pass);
            lo  = _mm256_min_epu32(lo, _mm256_or_si256(zk, _mm256_xor_si256(pass, _mm256_set1_epi32(-1))));
            hi  = _mm256_max_epu32(hi, _mm256_and_si256(zk, pass));
            unsigned c = (unsigned)pm - (((unsigned)pm >> 1) & 0x55u);
            c          = (c & 0x33u) + ((c >> 2) & 0x33u);
            cnt += (int)((c + (c >> 4)) & 0x0fu);
        }
        _mm256_storeu_si256((__m256i*)(zb + i), _mm256_blendv_epi8(w, nwv, act));
    }
    if (cnt) {
        uint32_t l8[8], h8[8];
        _mm256_storeu_si256((__m256i*)l8, lo);
        _mm256_storeu_si256((__m256i*)h8, hi);
        for (int k = 0; k < 8; k++) {
            if (l8[k] < *wmin) *wmin = l8[k];
            if (h8[k] > *wmax) *wmax = h8[k];
        }
        *nw += cnt;
    }
    if (n - i >= 4) {
        FMK(zs_d24s8_x4)(z + i, zb + i, m + i, n - i, p, wmin, wmax, nw);
        return;
    }
    for (; i < n; i++) fm_zs24_1(z, zb, m, i, p, wmin, wmax, nw);
}

#include "fm_kernels_tmpl.h"

#endif
