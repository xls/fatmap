/* fatmap - SSE2 backend (x86 baseline, 4 pixels per block) */
#include "fm_internal.h"

#if FM_ARCH_X86

#define FMK(name) name##_sse2
#define FMK_TABLE fm_kernels_sse2
#define FMK_LEVEL FM_SIMD_SSE2
#define FMK_PX    4

#include "fm_kernels_x86.h"

typedef __m128i vpx;
typedef __m128i vw;

FM_INLINE vpx  vpx_load(const void* p) { return _mm_loadu_si128((const __m128i*)p); }
FM_INLINE void vpx_store(void* p, vpx v) { _mm_storeu_si128((__m128i*)p, v); }
FM_INLINE vpx  vpx_set1(uint32_t v) { return _mm_set1_epi32((int)v); }
FM_INLINE vw   vw_lo(vpx p) { return _mm_unpacklo_epi8(p, _mm_setzero_si128()); }
FM_INLINE vw   vw_hi(vpx p) { return _mm_unpackhi_epi8(p, _mm_setzero_si128()); }
FM_INLINE vpx  vw_pack(vw lo, vw hi) { return _mm_packus_epi16(lo, hi); }
FM_INLINE vpx  vpx_mask_load(const uint8_t* m)
{
    uint32_t w;
    memcpy(&w, m, 4);
    __m128i x = _mm_cvtsi32_si128((int)w);
    x         = _mm_unpacklo_epi8(x, x);
    return _mm_unpacklo_epi16(x, x);
}
FM_INLINE vw vw_set1(uint16_t v) { return _mm_set1_epi16((short)v); }
FM_INLINE vw vw_add(vw a, vw b) { return _mm_add_epi16(a, b); }
FM_INLINE vw vw_sub(vw a, vw b) { return _mm_sub_epi16(a, b); }
FM_INLINE vw vw_mul(vw a, vw b) { return _mm_mullo_epi16(a, b); }
FM_INLINE vw vw_min(vw a, vw b) { return _mm_sub_epi16(a, _mm_subs_epu16(a, b)); }
FM_INLINE vw vw_max(vw a, vw b) { return _mm_add_epi16(_mm_subs_epu16(a, b), b); }
FM_INLINE vw vw_div255(vw a)
{
    return _mm_mulhi_epu16(_mm_add_epi16(a, _mm_set1_epi16(128)), _mm_set1_epi16(257));
}
FM_INLINE vw vw_alpha(vw a)
{
    return _mm_shufflehi_epi16(_mm_shufflelo_epi16(a, _MM_SHUFFLE(3, 3, 3, 3)), _MM_SHUFFLE(3, 3, 3, 3));
}
FM_INLINE vw vw_alpha_merge(vw c, vw a)
{
    const __m128i am = _mm_set_epi16(-1, 0, 0, 0, -1, 0, 0, 0);
    return _mm_or_si128(_mm_and_si128(am, a), _mm_andnot_si128(am, c));
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
    __m128i a = _mm_srli_epi32(v, 24);
    return _mm_movemask_epi8(_mm_cmpeq_epi32(a, _mm_set1_epi32(255))) == 0xffff;
}
FM_INLINE int vpx_all_zero(vpx v) { return _mm_movemask_epi8(_mm_cmpeq_epi32(v, _mm_setzero_si128())) == 0xffff; }

#include "fm_kernels_tmpl.h"

#endif
