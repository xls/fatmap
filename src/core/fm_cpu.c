/* fatmap - CPU detection, SIMD dispatch, public span entry points */
#include "fm_internal.h"
#include "fm_atomic.h"

#if FM_ARCH_X86
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

const fm_kernels* fm_k = &fm_kernels_scalar;

static unsigned      g_features;
static fm_simd_level g_best = FM_SIMD_SCALAR;
static volatile long g_init_state; /* 0 = no, 1 = running, 2 = done */

#if FM_ARCH_X86
static void fm_cpuid(int leaf, int sub, int r[4])
{
#  if defined(_MSC_VER)
    __cpuidex(r, leaf, sub);
#  else
    unsigned a, b, c, d;
    __cpuid_count((unsigned)leaf, (unsigned)sub, a, b, c, d);
    r[0] = (int)a;
    r[1] = (int)b;
    r[2] = (int)c;
    r[3] = (int)d;
#  endif
}

static uint64_t fm_xgetbv0(void)
{
#  if defined(_MSC_VER)
    return _xgetbv(0);
#  else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#  endif
}
#endif

static unsigned fm_detect(void)
{
    unsigned f = 0;
#if FM_ARCH_X86
    int r[4];
    fm_cpuid(0, 0, r);
    int max_leaf = r[0];
    fm_cpuid(1, 0, r);
    if (r[3] & (1 << 26)) f |= FM_CPU_SSE2;
    if (r[2] & (1 << 19)) f |= FM_CPU_SSE41;
    int osxsave = (r[2] & (1 << 27)) != 0;
    int avx     = (r[2] & (1 << 28)) != 0;
    int fma     = (r[2] & (1 << 12)) != 0;
    if (osxsave && avx && (fm_xgetbv0() & 6) == 6) {
        if (fma) f |= FM_CPU_FMA;
        if (max_leaf >= 7) {
            fm_cpuid(7, 0, r);
            if (r[1] & (1 << 5)) f |= FM_CPU_AVX2;
            /* AVX-512 F (16), DQ (17), BW (30), VL (31) and the opmask / zmm state */
            unsigned need = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
            if (((unsigned)r[1] & need) == need && (fm_xgetbv0() & 0xe6) == 0xe6) f |= FM_CPU_AVX512;
        }
    }
#elif FM_ARCH_ARM64
    f |= FM_CPU_NEON; /* mandatory on AArch64 */
#endif
    return f;
}

#ifdef FM_HAVE_AVX2
static fm_kernels g_kernels_avx512; /* the AVX2 kernels, reporting the AVX-512 level */
#endif

static const fm_kernels* fm_table_for(fm_simd_level level)
{
    switch (level) {
#ifdef FM_HAVE_AVX2
    case FM_SIMD_AVX512: return (g_features & FM_CPU_AVX512) ? &g_kernels_avx512 : NULL;
#endif
#ifdef FM_HAVE_SSE2
    case FM_SIMD_SSE2: return (g_features & FM_CPU_SSE2) ? &fm_kernels_sse2 : NULL;
#endif
#ifdef FM_HAVE_AVX2
    case FM_SIMD_AVX2: return (g_features & FM_CPU_AVX2) ? &fm_kernels_avx2 : NULL;
#endif
#ifdef FM_HAVE_NEON
    case FM_SIMD_NEON: return (g_features & FM_CPU_NEON) ? &fm_kernels_neon : NULL;
#endif
    case FM_SIMD_SCALAR: return &fm_kernels_scalar;
    default: return NULL;
    }
}

static void fm_init_once(void)
{
    g_features = fm_detect();
    g_best     = FM_SIMD_SCALAR;
#ifdef FM_HAVE_AVX2
    g_kernels_avx512       = fm_kernels_avx2;
    g_kernels_avx512.level = FM_SIMD_AVX512;
#endif
    static const fm_simd_level order[] = { FM_SIMD_AVX512, FM_SIMD_AVX2, FM_SIMD_NEON, FM_SIMD_SSE2 };
    for (int i = 0; i < 4; i++) {
        if (fm_table_for(order[i])) {
            g_best = order[i];
            break;
        }
    }
    const char* env = getenv("FM_SIMD");
    fm_simd_level lvl = g_best;
    if (env) {
        for (int l = FM_SIMD_SCALAR; l < FM_SIMD_LEVELS; l++)
            if (strcmp(env, fm_simd_name((fm_simd_level)l)) == 0 && fm_table_for((fm_simd_level)l))
                lvl = (fm_simd_level)l;
    }
    fm_k = fm_table_for(lvl);
}

void fm__init(void)
{
    if (fm_atomic_load(&g_init_state) == 2) return;
    if (fm_atomic_cas(&g_init_state, 0, 1)) {
        fm_init_once();
        fm_atomic_store(&g_init_state, 2);
        return;
    }
    while (fm_atomic_load(&g_init_state) != 2) fm_cpu_relax();
}

void fm_init(void) { fm__init(); }

unsigned fm_cpu_features(void)
{
    fm__init();
    return g_features;
}

fm_simd_level fm_simd_best(void)
{
    fm__init();
    return g_best;
}

fm_simd_level fm_simd_current(void)
{
    fm__init();
    return fm_k->level;
}

int fm_simd_supported(fm_simd_level level)
{
    fm__init();
    return fm_table_for(level) != NULL;
}

int fm_simd_set(fm_simd_level level)
{
    fm__init();
    const fm_kernels* t = fm_table_for(level);
    if (!t) return 0;
    fm_k = t;
    return 1;
}

const char* fm_simd_name(fm_simd_level level)
{
    switch (level) {
    case FM_SIMD_SCALAR: return "scalar";
    case FM_SIMD_SSE2: return "sse2";
    case FM_SIMD_AVX2: return "avx2";
    case FM_SIMD_NEON: return "neon";
    case FM_SIMD_AVX512: return "avx512";
    default: return "unknown";
    }
}

const char* fm_version_string(void) { return "0.8.0"; }

/* ---- memory ---------------------------------------------------------------- */

void* fm_aligned_alloc(size_t size, size_t align)
{
    /* portable: over-allocate and stash the original pointer */
    uint8_t* raw = (uint8_t*)malloc(size + align + sizeof(void*));
    if (!raw) return NULL;
    uintptr_t p = ((uintptr_t)(raw + sizeof(void*)) + (align - 1)) & ~(uintptr_t)(align - 1);
    ((void**)p)[-1] = raw;
    return (void*)p;
}

void fm_aligned_free(void* p)
{
    if (p) free(((void**)p)[-1]);
}

/* ---- public span entry points --------------------------------------------- */

void fm_fill_span(uint32_t* dst, uint32_t value, int n)
{
    fm__init();
    if (n > 0) fm_k->fill(dst, value, n);
}

void fm_blend_span(uint32_t* dst, const uint32_t* src, const uint8_t* cov, int n, fm_blend_op op)
{
    fm__init();
    if (n <= 0) return;
    if (op == FM_OP_SRC_OVER) {
        if (cov)
            fm_k->span_over_mask(dst, src, cov, n);
        else
            fm_k->span_over(dst, src, n);
    } else {
        fm_k->span_op(dst, src, cov, n, op);
    }
}

void fm_blend_solid(uint32_t* dst, uint32_t src, const uint8_t* cov, int n, fm_blend_op op)
{
    fm__init();
    if (n <= 0) return;
    if (op == FM_OP_SRC_OVER) {
        if (cov)
            fm_k->solid_over_mask(dst, src, cov, n);
        else
            fm_k->solid_over(dst, src, n);
        return;
    }
    if (op == FM_OP_COPY && !cov) {
        fm_k->fill(dst, src, n);
        return;
    }
    uint32_t tmp[256];
    int      filled = 0;
    for (int i = 0; i < n; i += 256) {
        int c = FM_MIN(256, n - i);
        if (c > filled) {
            fm_k->fill(tmp + filled, src, c - filled);
            filled = c;
        }
        fm_k->span_op(dst + i, tmp, cov ? cov + i : NULL, c, op);
    }
}
