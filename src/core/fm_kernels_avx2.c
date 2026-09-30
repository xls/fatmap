/* fatmap - AVX2 backend (8 pixels per block). Compiled with -mavx2 / /arch:AVX2,
 * only selected at runtime when the CPU and OS support AVX2. */
#include "fm_internal.h"

#if FM_ARCH_X86

#define FMK(name) name##_avx2
#define FMK_TABLE fm_kernels_avx2
#define FMK_LEVEL FM_SIMD_AVX2
#define FMK_PX    8

#include "fm_kernels_x86.h"
#include <immintrin.h>

typedef __m256i vpx;
typedef __m256i vw;

FM_INLINE vpx  vpx_load(const void* p) { return _mm256_loadu_si256((const __m256i*)p); }
FM_INLINE void vpx_store(void* p, vpx v) { _mm256_storeu_si256((__m256i*)p, v); }
FM_INLINE vpx  vpx_set1(uint32_t v) { return _mm256_set1_epi32((int)v); }
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

#include "fm_kernels_tmpl.h"

#endif
