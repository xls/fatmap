/*
 * fatmap - SPIR-V batch executor (FM_FEATURE_SPIRV), a template: included
 * once per instruction set (fm3d_spirv.c: the baseline; fm3d_spirv_avx2.c,
 * fm3d_spirv_avx512.c compiled with their ISA flags), so the per operation
 * lane loops vectorize at the full width. SV_FN(name) gives the ISA's names
 * for the entry points (sv_run_vs / sv_run_fs); everything else is static.
 * The same C code, so every ISA renders the same bits.
 */
#include "fm3d_spirv_internal.h"
#include "fatmap/fm_vmath.h"

#if FM_FEATURE_SPIRV

#include <math.h>
#if FM_ARCH_X86
#  include <emmintrin.h>
#elif FM_ARCH_ARM64
#  include <arm_neon.h>
#endif

/* ---- execution ------------------------------------------------------------------ */

typedef struct sv_exec {
    const sv_stage*            s;
    uint32_t*                  x;   /* scratch lane blocks */
    uint32_t*                  tmp; /* phi staging */
    const uint8_t*             ubo;
    size_t                     ubo_n; /* the block being read (sv_load picks it by binding) */
    const uint8_t*             blk[FM3D_MAX_UNIFORM_BLOCKS];
    size_t                     blkn[FM3D_MAX_UNIFORM_BLOCKS];
    const fm3d_fs_io*          fio; /* fragment stage */
    const fm3d_texture* const* tex;
    const fm3d_sampler*        samp;
    uint64_t*                  M; /* mask slots of the lowered program */
} sv_exec;

/* lane mask of the nonzero entries of a 64 lane block */
FM_INLINE uint64_t sv_bits(const uint32_t* c)
{
    uint64_t m = 0;
#if FM_ARCH_X86
    const __m128i z = _mm_setzero_si128();
    for (int i = 0; i < SV_L; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(c + i));
        int     b = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v, z))) ^ 15; /* nonzero */
        m |= (uint64_t)b << i;
    }
#elif FM_ARCH_ARM64
    static const int32_t sh[4] = { 0, 1, 2, 3 };
    const int32x4_t      vs    = vld1q_s32(sh);
    for (int i = 0; i < SV_L; i += 4) {
        uint32x4_t nz = vtstq_u32(vld1q_u32(c + i), vld1q_u32(c + i)); /* all ones where nonzero */
        uint32x4_t b  = vshlq_u32(vandq_u32(nz, vdupq_n_u32(1)), vs);
        m |= (uint64_t)vaddvq_u32(b) << i;
    }
#else
    for (int l = 0; l < SV_L; l++) m |= (uint64_t)(c[l] != 0) << l;
#endif
    return m;
}

/* per lane all ones / zero words of a lane mask, for vectorizable blends */
FM_INLINE void sv_expand(uint64_t m, uint32_t* lm)
{
    static const uint32_t nib[16][4] = { { 0, 0, 0, 0 }, { ~0u, 0, 0, 0 }, { 0, ~0u, 0, 0 }, { ~0u, ~0u, 0, 0 },
                                         { 0, 0, ~0u, 0 }, { ~0u, 0, ~0u, 0 }, { 0, ~0u, ~0u, 0 }, { ~0u, ~0u, ~0u, 0 },
                                         { 0, 0, 0, ~0u }, { ~0u, 0, 0, ~0u }, { 0, ~0u, 0, ~0u }, { ~0u, ~0u, 0, ~0u },
                                         { 0, 0, ~0u, ~0u }, { ~0u, 0, ~0u, ~0u }, { 0, ~0u, ~0u, ~0u }, { ~0u, ~0u, ~0u, ~0u } };
    for (int i = 0; i < SV_L; i += 4) memcpy(lm + i, nib[(m >> i) & 15], 16);
}

FM_INLINE void sv_blend(uint32_t* dst, const uint32_t* src, const uint32_t* lm, int nblocks)
{
    for (int b = 0; b < nblocks; b++) {
        uint32_t*       d = dst + (size_t)b * SV_L;
        const uint32_t* s = src + (size_t)b * SV_L;
        for (int l = 0; l < SV_L; l++) d[l] = (s[l] & lm[l]) | (d[l] & ~lm[l]);
    }
}

/* copy n lane blocks (constant size memcpy per block: inlined as vector moves) */
FM_INLINE void sv_copy(uint32_t* dst, const uint32_t* src, int n)
{
    for (int b = 0; b < n; b++) memcpy(dst + (size_t)b * SV_L, src + (size_t)b * SV_L, SV_L * sizeof(uint32_t));
}

FM_INLINE uint32_t* R(const sv_exec* E, int id)
{
    const sv_id* d = &E->s->ids[id];
    return d->reg < 0 ? E->s->cdata + (size_t)(-d->reg - 1) * SV_L : E->x + (size_t)d->reg * SV_L;
}
#define RF(id) ((float*)R(E, (int)(id)))
#define RI(id) ((int32_t*)R(E, (int)(id)))
#define RU(id) ((uint32_t*)R(E, (int)(id)))
#define NC(id) (E->s->ids[(int)(id)].comps)
#define FOR_L for (int l = 0; l < SV_L; l++)

static uint32_t sv_u32(const sv_exec* E, long off)
{
    uint32_t v = 0;
    if (off >= 0 && (size_t)off + 4 <= E->ubo_n) memcpy(&v, E->ubo + off, 4);
    return v;
}

/* read a value of type t from the uniform block at byte offset (base +
 * per lane off[l] when off) */
static void sv_uload(const sv_exec* E, int t, long base, const int* off, int mstride, uint32_t* dst, int* c)
{
    const sv_id* T = &E->s->ids[t];
    switch (T->kind) {
    case T_BOOL: case T_INT: case T_UINT: case T_FLOAT: {
        uint32_t* d = dst + (size_t)(*c)++ * SV_L;
        if (!off) {
            sv_bcast(d, sv_u32(E, base));
        } else {
            FOR_L d[l] = sv_u32(E, base + off[l]);
        }
        if (T->kind == T_BOOL) FOR_L d[l] = d[l] != 0;
        break;
    }
    case T_VEC:
        for (int i = 0; i < T->count; i++) sv_uload(E, T->elem, base + 4 * i, off, 0, dst, c);
        break;
    case T_MAT:
        for (int i = 0; i < T->count; i++) sv_uload(E, T->elem, base + (long)i * mstride, off, 0, dst, c);
        break;
    case T_ARR:
        for (int i = 0; i < T->count; i++) sv_uload(E, T->elem, base + (long)i * T->astride, off, mstride, dst, c);
        break;
    case T_STRUCT:
        for (int m = 0; m < T->nmem; m++) sv_uload(E, T->mem[m], base + T->mboff[m], off, T->mstride[m], dst, c);
        break;
    default: break;
    }
}

/* per lane dynamic offsets of a pointer (NULL when static); *uni is set
 * when every lane has the same offset (loop counters, uniform indices) */
static const int* sv_dyn(const sv_exec* E, const sv_id* P, int* buf, int* uni)
{
    *uni = 0;
    if (!P->ndyn) return NULL;
    FOR_L buf[l] = 0;
    for (int k = 0; k < P->ndyn; k++) {
        const int32_t* ix = RI(P->dyn[k]);
        FOR_L buf[l] += ix[l] * P->dstride[k];
    }
    int same = 1;
    FOR_L same &= buf[l] == buf[0];
    *uni = same;
    return buf;
}

static void sv_load(sv_exec* E, int ptr, uint32_t* dst, int comps)
{
    const sv_id* P = &E->s->ids[ptr];
    const sv_id* V = &E->s->ids[P->pvar];
    int          ob[SV_L], uni;
    const int*   off = sv_dyn(E, P, ob, &uni);
    if (sv_is_ptr_storage_uniform(V->storage)) {
        int c = 0;
        int b    = V->binding >= 0 && V->binding < FM3D_MAX_UNIFORM_BLOCKS ? V->binding : 0; /* push constants: 0 */
        E->ubo   = E->blk[b];
        E->ubo_n = E->blkn[b];
        sv_uload(E, P->type, P->poff + (uni ? off[0] : 0), uni ? NULL : off, P->pmstride, dst, &c);
        return;
    }
    const uint32_t* src = E->x + (size_t)V->reg * SV_L;
    if (off && uni) { /* one offset for every lane: a block copy */
        int o = P->poff + off[0], maxo = V->comps - comps;
        o     = o < 0 ? 0 : (o > maxo ? maxo : o);
        sv_copy(dst, src + (size_t)o * SV_L, comps);
        return;
    }
    if (!off) {
        sv_copy(dst, src + (size_t)P->poff * SV_L, comps);
        return;
    }
    int maxo = V->comps - comps;
    for (int c = 0; c < comps; c++) FOR_L {
            int o = P->poff + off[l];
            o     = o < 0 ? 0 : (o > maxo ? maxo : o);
            dst[(size_t)c * SV_L + l] = src[(size_t)(o + c) * SV_L + l];
        }
}

static void sv_store(sv_exec* E, int ptr, const uint32_t* src, int comps, uint64_t mask)
{
    const sv_id* P = &E->s->ids[ptr];
    const sv_id* V = &E->s->ids[P->pvar];
    if (V->reg < 0 || sv_is_ptr_storage_uniform(V->storage)) return;
    uint32_t* dst = E->x + (size_t)V->reg * SV_L;
    if (!P->ndyn) { /* static offset: block copy or a lane blend */
        uint32_t* d = dst + (size_t)P->poff * SV_L;
        if (mask == ~0ull) {
            sv_copy(d, src, comps);
        } else {
            uint32_t lm[SV_L];
            sv_expand(mask, lm);
            sv_blend(d, src, lm, comps);
        }
        return;
    }
    int        ob[SV_L], uni;
    const int* off  = sv_dyn(E, P, ob, &uni);
    int        maxo = V->comps - comps;
    if (uni) { /* one offset for every lane */
        int o = P->poff + off[0];
        o     = o < 0 ? 0 : (o > maxo ? maxo : o);
        uint32_t lm[SV_L];
        sv_expand(mask, lm);
        sv_blend(dst + (size_t)o * SV_L, src, lm, comps);
        return;
    }
    for (int c = 0; c < comps; c++) FOR_L {
            if (!((mask >> l) & 1)) continue;
            int o = P->poff + (off ? off[l] : 0);
            o     = o < 0 ? 0 : (o > maxo ? maxo : o);
            dst[(size_t)(o + c) * SV_L + l] = src[(size_t)c * SV_L + l];
        }
}

static float sv_fclamp(float x, float lo, float hi) { return fm_fminf(fm_fmaxf(x, lo), hi); }

/* ---- texture sampling ---- */


static void sv_sample(sv_exec* E, const uint32_t* in, int explicit_lod, int proj)
{
    const sv_id* img = &E->s->ids[in[3]];
    float*       out = RF(in[2]);
    const float* cu  = RF(in[4]);
    const float* cv  = cu + SV_L;
    float        pu[SV_L], pv[SV_L];
    if (proj) { /* textureProj: divided by the last coordinate */
        const float* q = cu + (size_t)(E->s->ids[in[4]].comps - 1) * SV_L;
        FOR_L pu[l] = cu[l] / q[l], pv[l] = cv[l] / q[l];
        cu = pu, cv = pv;
    }
    const fm3d_texture* t = img->cls == C_IMG && img->unit >= 0 ? E->tex[img->unit] : NULL;
    if (!t) {
        for (int k = 0; k < 4 * SV_L; k++) out[k] = k >= 3 * SV_L ? 1.0f : 0.0f;
        return;
    }
    const fm3d_sampler* s = &E->samp[img->unit];
    int                 wc = (int)WC(in);
    int                 nc = E->s->ids[in[4]].comps;
    if (img->dim != 1 || img->arrayed) { /* 1D, 3D, cube, rectangle, arrays */
        static const float half[SV_L] = { 0.5f };
        float hv[SV_L], ru[SV_L], rv[SV_L];
        const float *c0 = cu, *c1 = nc > 1 ? cv : NULL, *c2 = nc > 2 ? cu + 2 * SV_L : NULL;
        (void)half;
        if (img->dim == 0) { /* 1D: one row; 1D arrays: (s, layer) -> (s, 0.5, layer) */
            FOR_L hv[l] = 0.5f;
            c2 = img->arrayed ? c1 : NULL;
            c1 = hv;
        } else if (img->dim == 4) { /* rectangle: texel coordinates */
            float w = (float)t->level[0]->width, h = (float)t->level[0]->height;
            FOR_L ru[l] = cu[l] / w, rv[l] = cv[l] / h;
            c0 = ru, c1 = rv, c2 = NULL;
        }
        const float* lod = explicit_lod && wc >= 7 && (in[5] & 2u) ? RF(in[6]) : NULL;
        fm3d_sample_tex(t, s, c0, c1 ? c1 : hv, c2, SV_L, !explicit_lod && E->fio ? E->fio->cols / 2 : 0, lod, out, out + SV_L,
                        out + 2 * SV_L, out + 3 * SV_L);
        return;
    }
    if (explicit_lod) {
        const float* lod = NULL;
        if (wc >= 7 && (in[5] & 2u)) lod = RF(in[6]); /* ImageOperands Lod */
        fm3d_sample_lod(t, s, cu, cv, lod, SV_L, out, out + SV_L, out + 2 * SV_L, out + 3 * SV_L);
        return;
    }
    if (E->fio) {
        fm3d_sample_quads(t, s, cu, cv, E->fio->cols / 2, out, out + SV_L, out + 2 * SV_L, out + 3 * SV_L);
        return;
    }
    fm3d_sample_lod(t, s, cu, cv, NULL, SV_L, out, out + SV_L, out + 2 * SV_L, out + 3 * SV_L); /* vertex stage: base level */
}

/* ---- GLSL.std.450 ---- */

/* the transcendental GLSL.std.450 functions over N floats (the interpreter and
 * the JIT's math helper: the same compiled loops, so the same bits) */
void SV_FN(sv_math)(int fn, int fast, int N, const float* a, const float* b, float* f)
{
    int k;
    switch (fn) {
    case 13:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_sinf(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_sinf(a[k]);
        break;
    case 14:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_cosf(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_cosf(a[k]);
        break;
    case 15:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_tanf(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_tanf(a[k]);
        break;
    case 16: for (k = 0; k < N; k++) f[k] = asinf(a[k]); break;
    case 17: for (k = 0; k < N; k++) f[k] = acosf(a[k]); break;
    case 18: for (k = 0; k < N; k++) f[k] = atanf(a[k]); break;
    case 19: for (k = 0; k < N; k++) f[k] = sinhf(a[k]); break;
    case 20: for (k = 0; k < N; k++) f[k] = coshf(a[k]); break;
    case 21: for (k = 0; k < N; k++) f[k] = tanhf(a[k]); break;
    case 25: for (k = 0; k < N; k++) f[k] = atan2f(a[k], b[k]); break;
    case 26:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_powf(a[k], b[k]);
        else for (k = 0; k < N; k++) f[k] = fm_powf(a[k], b[k]);
        break;
    case 27:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_expf(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_expf(a[k]);
        break;
    case 28:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_logf(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_logf(a[k]);
        break;
    case 29:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_exp2f(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_exp2f(a[k]);
        break;
    case 30:
        if (fast) for (k = 0; k < N; k++) f[k] = fm_fast_log2f(a[k]);
        else for (k = 0; k < N; k++) f[k] = fm_log2f(a[k]);
        break;
    default: break;
    }
}

static void sv_ext(sv_exec* E, const uint32_t* in)
{
    int       n = NC(in[2]);
    uint32_t* r = R(E, (int)in[2]);
    float*    f = (float*)r;
    int32_t*  s = (int32_t*)r;
    int       fn = (int)in[4];
    const float* a = WC(in) > 5 ? RF(in[5]) : NULL;
    const float* b = WC(in) > 6 ? RF(in[6]) : NULL;
    const float* c = WC(in) > 7 ? RF(in[7]) : NULL;
    const int32_t* ia = (const int32_t*)a;
    const int32_t* ib = (const int32_t*)b;
    const int32_t* ic = (const int32_t*)c;
    const uint32_t* ua = (const uint32_t*)a;
    const uint32_t* ub = (const uint32_t*)b;
    const uint32_t* uc = (const uint32_t*)c;
    int k, N = n * SV_L, fast = E->s->fast;
    switch (fn) {
    case 1: for (k = 0; k < N; k++) f[k] = fm_roundf(a[k]); break;
    case 2: for (k = 0; k < N; k++) f[k] = fm_rintf(a[k]); break;
    case 3: for (k = 0; k < N; k++) f[k] = fm_truncf(a[k]); break;
    case 4: for (k = 0; k < N; k++) f[k] = fabsf(a[k]); break;
    case 5: for (k = 0; k < N; k++) s[k] = ia[k] < 0 ? (int32_t)(0u - (uint32_t)ia[k]) : ia[k]; break;
    case 6: for (k = 0; k < N; k++) f[k] = a[k] > 0 ? 1.0f : (a[k] < 0 ? -1.0f : 0.0f); break;
    case 7: for (k = 0; k < N; k++) s[k] = ia[k] > 0 ? 1 : (ia[k] < 0 ? -1 : 0); break;
    case 8: for (k = 0; k < N; k++) f[k] = fm_floorf(a[k]); break;
    case 9: for (k = 0; k < N; k++) f[k] = fm_ceilf(a[k]); break;
    case 10: for (k = 0; k < N; k++) f[k] = a[k] - fm_floorf(a[k]); break;
    case 11: for (k = 0; k < N; k++) f[k] = a[k] * 0.017453292519943295f; break;
    case 12: for (k = 0; k < N; k++) f[k] = a[k] * 57.29577951308232f; break;
    case 13: case 14: case 15: case 16: case 17: case 18: case 19: case 20: case 21: case 25: case 26: case 27: case 28: case 29:
    case 30: SV_FN(sv_math)(fn, fast, N, a, b, f); break;
    case 31: for (k = 0; k < N; k++) f[k] = sqrtf(a[k]); break;
    case 32: for (k = 0; k < N; k++) f[k] = 1.0f / sqrtf(a[k]); break;
    case 37: case 79: for (k = 0; k < N; k++) f[k] = fm_fminf(a[k], b[k]); break;
    case 38: for (k = 0; k < N; k++) r[k] = ua[k] < ub[k] ? ua[k] : ub[k]; break;
    case 39: for (k = 0; k < N; k++) s[k] = ia[k] < ib[k] ? ia[k] : ib[k]; break;
    case 40: case 80: for (k = 0; k < N; k++) f[k] = fm_fmaxf(a[k], b[k]); break;
    case 41: for (k = 0; k < N; k++) r[k] = ua[k] > ub[k] ? ua[k] : ub[k]; break;
    case 42: for (k = 0; k < N; k++) s[k] = ia[k] > ib[k] ? ia[k] : ib[k]; break;
    case 43: case 81: for (k = 0; k < N; k++) f[k] = sv_fclamp(a[k], b[k], c[k]); break;
    case 44: for (k = 0; k < N; k++) { uint32_t v = ua[k] > ub[k] ? ua[k] : ub[k]; r[k] = v < uc[k] ? v : uc[k]; } break;
    case 45: for (k = 0; k < N; k++) { int32_t v = ia[k] > ib[k] ? ia[k] : ib[k]; s[k] = v < ic[k] ? v : ic[k]; } break;
    case 46: for (k = 0; k < N; k++) f[k] = a[k] * (1.0f - c[k]) + b[k] * c[k]; break;
    case 48: for (k = 0; k < N; k++) f[k] = b[k] < a[k] ? 0.0f : 1.0f; break;
    case 49:
        for (k = 0; k < N; k++) {
            float t = sv_fclamp((c[k] - a[k]) / (b[k] - a[k]), 0.0f, 1.0f);
            f[k]    = t * t * (3.0f - 2.0f * t);
        }
        break;
    case 50: for (k = 0; k < N; k++) f[k] = a[k] * b[k] + c[k]; break;
    case 66: case 67: { /* length, distance */
        int m = NC(in[5]);
        FOR_L
        {
            float d = 0;
            for (int j = 0; j < m; j++) {
                float v = fn == 66 ? a[j * SV_L + l] : a[j * SV_L + l] - b[j * SV_L + l];
                d += v * v;
            }
            f[l] = sqrtf(d);
        }
        break;
    }
    case 68: /* cross */
        FOR_L
        {
            float ax = a[l], ay = a[SV_L + l], az = a[2 * SV_L + l], bx = b[l], by = b[SV_L + l], bz = b[2 * SV_L + l];
            f[l] = ay * bz - az * by, f[SV_L + l] = az * bx - ax * bz, f[2 * SV_L + l] = ax * by - ay * bx;
        }
        break;
    case 69: /* normalize */
        FOR_L
        {
            float d = 0;
            for (int j = 0; j < n; j++) d += a[j * SV_L + l] * a[j * SV_L + l];
            float inv = 1.0f / sqrtf(d);
            for (int j = 0; j < n; j++) f[j * SV_L + l] = a[j * SV_L + l] * inv;
        }
        break;
    case 70: /* faceforward(N, I, Nref) */
        FOR_L
        {
            float d = 0;
            for (int j = 0; j < n; j++) d += c[j * SV_L + l] * b[j * SV_L + l];
            for (int j = 0; j < n; j++) f[j * SV_L + l] = d < 0 ? a[j * SV_L + l] : -a[j * SV_L + l];
        }
        break;
    case 71: /* reflect(I, N) */
        FOR_L
        {
            float d = 0;
            for (int j = 0; j < n; j++) d += b[j * SV_L + l] * a[j * SV_L + l];
            for (int j = 0; j < n; j++) f[j * SV_L + l] = a[j * SV_L + l] - 2.0f * d * b[j * SV_L + l];
        }
        break;
    case 72: /* refract(I, N, eta) */
        FOR_L
        {
            float d = 0, eta = c[l];
            for (int j = 0; j < n; j++) d += b[j * SV_L + l] * a[j * SV_L + l];
            float kk = 1.0f - eta * eta * (1.0f - d * d);
            for (int j = 0; j < n; j++)
                f[j * SV_L + l] = kk < 0 ? 0.0f : eta * a[j * SV_L + l] - (eta * d + sqrtf(kk)) * b[j * SV_L + l];
        }
        break;
    default: break;
    }
}

/* ---- one block's instructions (phis first) ---- */

static void sv_body(sv_exec* E, const sv_block* B, uint64_t mask)
{
    const sv_stage* S   = E->s;
    uint64_t*       M   = E->M;
    const int*      pm  = B->phim;
    int             nst = 0;
    /* phis: stage every result, then commit (they read the old values) */
    for (int ii = 0; ii < B->nphi; ii++) {
        const uint32_t* in    = S->w + B->insts[ii].at;
        int             comps = B->insts[ii].n;
        uint32_t*       t     = E->tmp + (size_t)nst * SV_L;
        sv_copy(t, R(E, (int)in[2]), comps);
        /* per incoming edge: the lanes that came from its block (usually all
         * of them, e.g. every lane taking a loop back edge) */
        for (int k = 3; k + 1 < (int)WC(in); k += 2) {
            uint64_t m = M[*pm++] & mask;
            if (!m) continue;
            const uint32_t* src = R(E, (int)in[k]);
            if (m == ~0ull) {
                sv_copy(t, src, comps);
                continue;
            }
            uint32_t lm[SV_L];
            sv_expand(m, lm);
            sv_blend(t, src, lm, comps);
        }
        nst += comps;
    }
    for (int i = 0; i < B->nclr; i++) M[B->clr[i]] = 0; /* edges consumed: the block runs now */
    nst = 0;
    for (int ii = 0; ii < B->ninst; ii++) {
        const uint32_t* in = S->w + B->insts[ii].at;
        int             op = B->insts[ii].op;
        if (op == OpPhi) {
            int comps = B->insts[ii].n;
            sv_copy(R(E, (int)in[2]), E->tmp + (size_t)nst * SV_L, comps);
            nst += comps;
            continue;
        }
        int n = B->insts[ii].n;
        int N = n * SV_L, k;
        switch (op) {
        case OpVariable: case OpSelectionMerge: case OpLoopMerge: case OpUndef: break;
        case OpLoad:
            if (S->ids[in[2]].cls == C_IMG) break;
            sv_load(E, (int)in[3], R(E, (int)in[2]), n);
            break;
        case OpStore: sv_store(E, (int)in[1], R(E, (int)in[2]), NC(in[2]), mask); break;
        case OpAccessChain: case OpInBoundsAccessChain: break; /* resolved statically */
        case OpCopyObject: case OpUConvert: case OpSConvert: case OpFConvert: case OpBitcast:
            sv_copy(R(E, (int)in[2]), R(E, (int)in[3]), n);
            break;
        case OpCompositeConstruct: {
            uint32_t* r = R(E, (int)in[2]);
            int       c = 0;
            for (int j = 3; j < (int)WC(in) && c < n; j++) {
                int m = NC(in[j]);
                sv_copy(r + (size_t)c * SV_L, R(E, (int)in[j]), m);
                c += m;
            }
            break;
        }
        case OpCompositeExtract: case OpCompositeInsert: {
            int base = op == OpCompositeExtract ? (int)in[3] : (int)in[4];
            int t = S->ids[base].type, off = 0, ct = t;
            for (int j = op == OpCompositeExtract ? 4 : 5; j < (int)WC(in); j++) {
                int o = sv_child(S, ct, (int)in[j], &ct);
                if (o < 0) break;
                off += o;
            }
            if (op == OpCompositeExtract) {
                sv_copy(R(E, (int)in[2]), R(E, base) + (size_t)off * SV_L, n);
            } else {
                sv_copy(R(E, (int)in[2]), R(E, base), n);
                sv_copy(R(E, (int)in[2]) + (size_t)off * SV_L, R(E, (int)in[3]), NC(in[3]));
            }
            break;
        }
        case OpVectorShuffle: {
            uint32_t* r  = R(E, (int)in[2]);
            int       n1 = NC(in[3]);
            for (int j = 0; j < n; j++) {
                uint32_t sel = in[5 + j];
                if (sel == 0xffffffffu) memset(r + (size_t)j * SV_L, 0, SV_L * 4);
                else sv_copy(r + (size_t)j * SV_L, (sel < (uint32_t)n1 ? R(E, (int)in[3]) + (size_t)sel * SV_L : R(E, (int)in[4]) + (size_t)(sel - (uint32_t)n1) * SV_L), 1);
            }
            break;
        }
        case OpVectorExtractDynamic: {
            uint32_t* r = R(E, (int)in[2]);
            const uint32_t* v = R(E, (int)in[3]);
            const int32_t*  ix = RI(in[4]);
            int             m = NC(in[3]);
            FOR_L { int i = ix[l] < 0 ? 0 : (ix[l] >= m ? m - 1 : ix[l]); r[l] = v[(size_t)i * SV_L + l]; }
            break;
        }
        case OpVectorInsertDynamic: {
            uint32_t* r = R(E, (int)in[2]);
            sv_copy(r, R(E, (int)in[3]), n);
            const uint32_t* c = R(E, (int)in[4]);
            const int32_t*  ix = RI(in[5]);
            FOR_L { int i = ix[l] < 0 ? 0 : (ix[l] >= n ? n - 1 : ix[l]); r[(size_t)i * SV_L + l] = c[l]; }
            break;
        }
        case OpTranspose: {
            const sv_id* T = &S->ids[S->ids[in[3]].type];
            int          cols = T->count, rows = S->ids[T->elem].comps;
            uint32_t*    r = R(E, (int)in[2]);
            const uint32_t* a = R(E, (int)in[3]);
            for (int cc = 0; cc < cols; cc++)
                for (int rr = 0; rr < rows; rr++) memcpy(r + (size_t)(rr * cols + cc) * SV_L, a + (size_t)(cc * rows + rr) * SV_L, SV_L * 4);
            break;
        }
        case OpImageSampleImplicitLod: sv_sample(E, in, 0, 0); break;
        case OpImageSampleExplicitLod: sv_sample(E, in, 1, 0); break;
        case OpImageSampleProjImplicitLod: sv_sample(E, in, 0, 1); break;
        case OpImageSampleProjExplicitLod: sv_sample(E, in, 1, 1); break;
        case OpConvertFToS: { const float* a = RF(in[3]); int32_t* r = RI(in[2]); for (k = 0; k < N; k++) { float v = a[k]; r[k] = v >= 2147483520.0f ? INT32_MAX : (v <= -2147483648.0f ? INT32_MIN : (v == v ? (int32_t)v : 0)); } break; }
        case OpConvertFToU: { const float* a = RF(in[3]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) { float v = a[k]; r[k] = v >= 4294967040.0f ? UINT32_MAX : (v > 0 ? (uint32_t)v : 0u); } break; }
        case OpConvertSToF: { const int32_t* a = RI(in[3]); float* r = RF(in[2]); for (k = 0; k < N; k++) r[k] = (float)a[k]; break; }
        case OpConvertUToF: { const uint32_t* a = RU(in[3]); float* r = RF(in[2]); for (k = 0; k < N; k++) r[k] = (float)a[k]; break; }
        case OpFNegate: { const float* a = RF(in[3]); float* r = RF(in[2]); for (k = 0; k < N; k++) r[k] = -a[k]; break; }
        case OpSNegate: { const uint32_t* a = RU(in[3]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) r[k] = 0u - a[k]; break; }
#define FBIN(OPC, EXPR) case OPC: { const float* a = RF(in[3]); const float* b = RF(in[4]); float* r = RF(in[2]); for (k = 0; k < N; k++) { float x = a[k], y = b[k]; r[k] = (EXPR); } break; }
#define UBIN(OPC, EXPR) case OPC: { const uint32_t* a = RU(in[3]); const uint32_t* b = RU(in[4]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) { uint32_t x = a[k], y = b[k]; r[k] = (EXPR); } break; }
#define SBIN(OPC, EXPR) case OPC: { const int32_t* a = RI(in[3]); const int32_t* b = RI(in[4]); int32_t* r = RI(in[2]); for (k = 0; k < N; k++) { int32_t x = a[k], y = b[k]; r[k] = (EXPR); } break; }
#define FCMP(OPC, EXPR) case OPC: { const float* a = RF(in[3]); const float* b = RF(in[4]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) { float x = a[k], y = b[k]; r[k] = (EXPR) ? 1u : 0u; } break; }
#define UCMP(OPC, EXPR) case OPC: { const uint32_t* a = RU(in[3]); const uint32_t* b = RU(in[4]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) { uint32_t x = a[k], y = b[k]; r[k] = (EXPR) ? 1u : 0u; } break; }
#define SCMP(OPC, EXPR) case OPC: { const int32_t* a = RI(in[3]); const int32_t* b = RI(in[4]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) { int32_t x = a[k], y = b[k]; r[k] = (EXPR) ? 1u : 0u; } break; }
        FBIN(OpFAdd, x + y)
        FBIN(OpFSub, x - y)
        FBIN(OpFMul, x * y)
        FBIN(OpFDiv, x / y)
        FBIN(OpFRem, fmodf(x, y))
        FBIN(OpFMod, x - y * fm_floorf(x / y))
        UBIN(OpIAdd, x + y)
        UBIN(OpISub, x - y)
        UBIN(OpIMul, x * y)
        UBIN(OpUDiv, y ? x / y : 0u)
        UBIN(OpUMod, y ? x % y : 0u)
        SBIN(OpSDiv, (y == 0 || (x == INT32_MIN && y == -1)) ? 0 : x / y)
        SBIN(OpSRem, (y == 0 || y == -1) ? 0 : x % y)
        SBIN(OpSMod, (y == 0 || y == -1) ? 0 : ((x % y) != 0 && ((x % y) < 0) != (y < 0) ? x % y + y : x % y))
        UBIN(OpShiftRightLogical, x >> (y & 31))
        SBIN(OpShiftRightArithmetic, (int32_t)(x >> (y & 31)))
        UBIN(OpShiftLeftLogical, x << (y & 31))
        UBIN(OpBitwiseOr, x | y)
        UBIN(OpBitwiseXor, x ^ y)
        UBIN(OpBitwiseAnd, x & y)
        UBIN(OpLogicalOr, (x | y) != 0)
        UBIN(OpLogicalAnd, (x != 0) & (y != 0))
        UBIN(OpLogicalEqual, (x != 0) == (y != 0))
        UBIN(OpLogicalNotEqual, (x != 0) != (y != 0))
        FCMP(OpFOrdEqual, x == y)
        FCMP(OpFUnordEqual, !(x != y) || x != x || y != y)
        FCMP(OpFOrdNotEqual, x != y && x == x && y == y)
        FCMP(OpFUnordNotEqual, x != y)
        FCMP(OpFOrdLessThan, x < y)
        FCMP(OpFUnordLessThan, !(x >= y))
        FCMP(OpFOrdGreaterThan, x > y)
        FCMP(OpFUnordGreaterThan, !(x <= y))
        FCMP(OpFOrdLessThanEqual, x <= y)
        FCMP(OpFUnordLessThanEqual, !(x > y))
        FCMP(OpFOrdGreaterThanEqual, x >= y)
        FCMP(OpFUnordGreaterThanEqual, !(x < y))
        UCMP(OpIEqual, x == y)
        UCMP(OpINotEqual, x != y)
        UCMP(OpUGreaterThan, x > y)
        UCMP(OpUGreaterThanEqual, x >= y)
        UCMP(OpULessThan, x < y)
        UCMP(OpULessThanEqual, x <= y)
        SCMP(OpSGreaterThan, x > y)
        SCMP(OpSGreaterThanEqual, x >= y)
        SCMP(OpSLessThan, x < y)
        SCMP(OpSLessThanEqual, x <= y)
        case OpNot: { const uint32_t* a = RU(in[3]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) r[k] = ~a[k]; break; }
        case OpLogicalNot: { const uint32_t* a = RU(in[3]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) r[k] = a[k] == 0; break; }
        case OpIsNan: { const float* a = RF(in[3]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) r[k] = a[k] != a[k]; break; }
        case OpIsInf: { const float* a = RF(in[3]); uint32_t* r = RU(in[2]); for (k = 0; k < N; k++) r[k] = isinf(a[k]) ? 1u : 0u; break; }
        case OpAny: case OpAll: {
            const uint32_t* a = RU(in[3]);
            uint32_t*       r = RU(in[2]);
            int             m = NC(in[3]);
            FOR_L
            {
                uint32_t v = op == OpAll;
                for (int j = 0; j < m; j++) v = op == OpAll ? (v && a[j * SV_L + l]) : (v || a[j * SV_L + l]);
                r[l] = v != 0;
            }
            break;
        }
        case OpSelect: {
            const uint32_t* cnd = RU(in[3]);
            const uint32_t* a = RU(in[4]);
            const uint32_t* b = RU(in[5]);
            uint32_t*       r = RU(in[2]);
            int             cc = NC(in[3]);
            for (int j = 0; j < n; j++) {
                const uint32_t* cj = cnd + (size_t)(cc == 1 ? 0 : j) * SV_L;
                FOR_L r[j * SV_L + l] = cj[l] ? a[j * SV_L + l] : b[j * SV_L + l];
            }
            break;
        }
        case OpVectorTimesScalar: case OpMatrixTimesScalar: {
            const float* a = RF(in[3]);
            const float* b = RF(in[4]);
            float*       r = RF(in[2]);
            for (int j = 0; j < n; j++) FOR_L r[j * SV_L + l] = a[j * SV_L + l] * b[l];
            break;
        }
        case OpDot: {
            const float* a = RF(in[3]);
            const float* b = RF(in[4]);
            float*       r = RF(in[2]);
            int          m = NC(in[3]);
            FOR_L
            {
                float d = a[l] * b[l];
                for (int j = 1; j < m; j++) d += a[j * SV_L + l] * b[j * SV_L + l];
                r[l] = d;
            }
            break;
        }
        case OpMatrixTimesVector: { /* r[row] = sum_c M[c][row] * v[c] */
            const sv_id* T = &S->ids[S->ids[in[3]].type];
            int          cols = T->count, rows = S->ids[T->elem].comps;
            const float* m = RF(in[3]);
            const float* v = RF(in[4]);
            float*       r = RF(in[2]);
            for (int rr = 0; rr < rows; rr++) FOR_L
                {
                    float d = m[(size_t)rr * SV_L + l] * v[l];
                    for (int cc = 1; cc < cols; cc++) d += m[(size_t)(cc * rows + rr) * SV_L + l] * v[(size_t)cc * SV_L + l];
                    r[(size_t)rr * SV_L + l] = d;
                }
            break;
        }
        case OpVectorTimesMatrix: { /* r[c] = sum_row v[row] * M[c][row] */
            const sv_id* T = &S->ids[S->ids[in[4]].type];
            int          cols = T->count, rows = S->ids[T->elem].comps;
            const float* v = RF(in[3]);
            const float* m = RF(in[4]);
            float*       r = RF(in[2]);
            for (int cc = 0; cc < cols; cc++) FOR_L
                {
                    float d = v[l] * m[(size_t)(cc * rows) * SV_L + l];
                    for (int rr = 1; rr < rows; rr++) d += v[(size_t)rr * SV_L + l] * m[(size_t)(cc * rows + rr) * SV_L + l];
                    r[(size_t)cc * SV_L + l] = d;
                }
            break;
        }
        case OpMatrixTimesMatrix: {
            const sv_id* TA = &S->ids[S->ids[in[3]].type];
            const sv_id* TB = &S->ids[S->ids[in[4]].type];
            int          rows = S->ids[TA->elem].comps, inner = TA->count, cols = TB->count;
            const float* A = RF(in[3]);
            const float* B2 = RF(in[4]);
            float*       r = RF(in[2]);
            for (int cc = 0; cc < cols; cc++)
                for (int rr = 0; rr < rows; rr++) FOR_L
                    {
                        float d = A[(size_t)rr * SV_L + l] * B2[(size_t)(cc * inner) * SV_L + l];
                        for (int j = 1; j < inner; j++) d += A[(size_t)(j * rows + rr) * SV_L + l] * B2[(size_t)(cc * inner + j) * SV_L + l];
                        r[(size_t)(cc * rows + rr) * SV_L + l] = d;
                    }
            break;
        }
        case OpOuterProduct: {
            int          rows = NC(in[3]), cols = NC(in[4]);
            const float* a = RF(in[3]);
            const float* b = RF(in[4]);
            float*       r = RF(in[2]);
            for (int cc = 0; cc < cols; cc++)
                for (int rr = 0; rr < rows; rr++) FOR_L r[(size_t)(cc * rows + rr) * SV_L + l] = a[(size_t)rr * SV_L + l] * b[(size_t)cc * SV_L + l];
            break;
        }
        case OpDPdx: case OpDPdy: case OpFwidth: case OpDPdxFine: case OpDPdyFine: case OpFwidthFine:
        case OpDPdxCoarse: case OpDPdyCoarse: case OpFwidthCoarse: {
            const float* a = RF(in[3]);
            float*       r = RF(in[2]);
            int          dx = op == OpDPdx || op == OpDPdxFine || op == OpDPdxCoarse;
            int          dy = op == OpDPdy || op == OpDPdyFine || op == OpDPdyCoarse;
            for (int j = 0; j < n; j++) {
                const float* v = a + (size_t)j * SV_L;
                float*       o = r + (size_t)j * SV_L;
                if (!E->fio) { /* vertex stage: no neighbours */
                    FOR_L o[l] = 0.0f;
                    continue;
                }
                FOR_L
                {
                    float ddx = v[l | 1] - v[l & ~1], ddy = v[l | 8] - v[l & ~8]; /* quad group order */
                    o[l]      = dx ? ddx : (dy ? ddy : fabsf(ddx) + fabsf(ddy));
                }
            }
            break;
        }
        case OpExtInst: sv_ext(E, in); break;
        default: break;
        }
    }
}

/* ---- the lowered program ---- */

static void sv_ir_run(sv_exec* E)
{
    const sv_stage* S = E->s;
    const sv_ir*    I = S->ir;
    uint64_t*       M = E->M;
    for (int pc = 0; pc < S->nir;) {
        const sv_ir* o = &I[pc];
        switch (o->op) {
        case IR_ZERO: M[o->a] = 0; break;
        case IR_COPY: M[o->a] = M[o->b]; break;
        case IR_OR: M[o->a] = M[o->b] | M[o->c]; break;
        case IR_ANDN: M[o->a] = M[o->b] & ~M[o->c]; break;
        case IR_COND: M[o->a] = M[o->c] ? sv_bits(RU(o->b)) & M[o->c] : 0; break;
        case IR_EQ: {
            const uint32_t* v = RU(o->b);
            uint64_t        m = 0;
            FOR_L m |= (uint64_t)(v[l] == o->lit) << l;
            M[o->a] = m & M[o->c];
            break;
        }
        case IR_BODY: sv_body(E, &S->blocks[o->a], M[o->b]); break;
        case IR_IF:
            if (!M[o->a]) {
                pc = o->jump + 1;
                continue;
            }
            break;
        case IR_LOOP: M[o->a] = 0; break;
        case IR_BREAKZ:
            if (!M[o->a] || M[o->b]++ >= SV_MAXITER) {
                pc = o->jump;
                continue;
            }
            break;
        case IR_BREAK: case IR_ENDLOOP: pc = o->jump; continue;
        default: break; /* IR_ENDIF */
        }
        pc++;
    }
}

static int sv_setup(sv_exec* E, const sv_stage* s)
{
    memset(E, 0, sizeof(*E));
    E->s = s;
    E->x = sv_scratch((size_t)s->nscratch + (size_t)s->nphi + 1);
    if (!E->x) return 0;
    E->tmp = E->x + (size_t)s->nscratch * SV_L;
    E->M = sv_mask_scratch(s->nmask);
    if (!E->M) return 0;
    memset(E->M, 0, (size_t)s->nmask * sizeof(uint64_t));
    for (int i = 0; i < s->nvars; i++) { /* zero or initialized variables */
        const sv_id* v = &s->ids[s->vars[i]];
        uint32_t*    d = E->x + (size_t)v->reg * SV_L;
        if (v->init && s->ids[v->init].cls == C_CONST)
            memcpy(d, sv_cblock((sv_stage*)s, (sv_id*)&s->ids[v->init]), (size_t)v->comps * SV_L * 4);
        else
            memset(d, 0, (size_t)v->comps * SV_L * 4);
    }
    return 1;
}

/* ---- stage entry points (fm3d_program callbacks) ---- */

void SV_FN(sv_run_vs)(const fm3d_vs_io* io)
{
    const fm3d_spirv* P = (const fm3d_spirv*)io->user;
    const sv_stage*   s = P->vs;
    for (int base = 0; base < io->count; base += SV_L) {
        int      n = io->count - base < SV_L ? io->count - base : SV_L;
        uint64_t mask = n == SV_L ? ~0ull : ((1ull << n) - 1);
        sv_exec  E;
        if (!sv_setup(&E, s)) return;
        E.ubo   = (const uint8_t*)io->uniforms;
        E.ubo_n = io->uniforms ? io->uniform_size : 0;
    for (int b = 0; b < FM3D_MAX_UNIFORM_BLOCKS; b++) {
        E.blk[b]  = io->blocks ? (const uint8_t*)io->blocks[b] : (b ? NULL : (const uint8_t*)io->uniforms);
        E.blkn[b] = io->blocks ? io->block_sizes[b] : (b ? 0 : E.ubo_n);
    }
        for (int b = 0; b < FM3D_MAX_UNIFORM_BLOCKS; b++) {
            E.blk[b]  = io->blocks ? (const uint8_t*)io->blocks[b] : (b ? NULL : (const uint8_t*)io->uniforms);
            E.blkn[b] = io->blocks ? io->block_sizes[b] : (b ? 0 : E.ubo_n);
        }
        E.tex   = io->textures;
        E.samp  = io->samplers;
        for (int i = 0; i < s->nin; i++) { /* vertex attributes */
            const sv_io* vi = &s->in[i];
            if (vi->builtin == BI_VertexIndex || vi->builtin == BI_InstanceIndex) {
                int32_t* d = (int32_t*)(E.x + (size_t)s->ids[vi->var].reg * SV_L);
                FOR_L d[l] = vi->builtin == BI_VertexIndex ? io->first_vertex + base + l : io->instance;
                continue;
            }
            const fm3d_vertex_attrib* a = NULL;
            for (int k = 0; k < P->nattr; k++)
                if (P->attr[k].location == vi->loc) a = &P->attr[k];
            float* d = (float*)(E.x + (size_t)s->ids[vi->var].reg * SV_L);
            for (int c = 0; c < vi->comps; c++)
                for (int l = 0; l < n; l++) {
                    float v = c == 3 ? 1.0f : 0.0f;
                    if (a && c < a->components) {
                        const char* vp = (const char*)io->vertices + (size_t)(base + l) * (size_t)io->stride + a->offset;
                        memcpy(&v, vp + 4 * c, 4);
                    }
                    d[(size_t)c * SV_L + l] = v;
                }
        }
        E.M[SV_M_ENTRY] = mask;
        sv_ir_run(&E);
        const float* pos = (const float*)(E.x + (size_t)(s->ids[s->pos_var].reg + s->pos_off) * SV_L);
        for (int l = 0; l < n; l++) {
            float* o = io->pos + (size_t)(base + l) * (size_t)io->out_stride;
            for (int c = 0; c < 4; c++) o[c] = pos[(size_t)c * SV_L + l];
        }
        if (s->ps_var >= 0 && P->ps_slot >= 0) {
            const float* ps = (const float*)(E.x + (size_t)(s->ids[s->ps_var].reg + s->ps_off) * SV_L);
            for (int l = 0; l < n; l++) io->varyings[(size_t)(base + l) * (size_t)io->out_stride + (size_t)P->ps_slot] = ps[l];
        }
        for (int i = 0; i < s->nout; i++) {
            const sv_io* vo = &s->out[i];
            if (vo->builtin >= 0) continue;
            const float* src = (const float*)(E.x + (size_t)s->ids[vo->var].reg * SV_L);
            int          slot = P->vslot[i];
            if (slot < 0) continue;
            for (int l = 0; l < n; l++) {
                float* q = io->varyings + (size_t)(base + l) * (size_t)io->out_stride;
                for (int c = 0; c < vo->comps && slot + c < FM3D_MAX_SHADER_VARYINGS; c++) q[slot + c] = src[(size_t)c * SV_L + l];
            }
        }
    }
}

void SV_FN(sv_run_fs)(const fm3d_fs_io* io)
{
    const fm3d_spirv* P = (const fm3d_spirv*)io->user;
    const sv_stage*   s = P->fs;
    sv_exec           E;
    if (!sv_setup(&E, s)) return;
    E.ubo   = (const uint8_t*)io->uniforms;
    E.ubo_n = io->uniforms ? io->uniform_size : 0;
    for (int b = 0; b < FM3D_MAX_UNIFORM_BLOCKS; b++) {
        E.blk[b]  = io->blocks ? (const uint8_t*)io->blocks[b] : (b ? NULL : (const uint8_t*)io->uniforms);
        E.blkn[b] = io->blocks ? io->block_sizes[b] : (b ? 0 : E.ubo_n);
    }
    E.fio   = io;
    E.tex   = io->textures;
    E.samp  = io->samplers;
    uint64_t lanes = 0; /* every pixel of the batch's quads (helpers included), quad group order */
    FOR_L if ((sv_frag_pixel(l) & 31) < io->cols) lanes |= 1ull << l;
    for (int i = 0; i < s->nin; i++) {
        const sv_io* fi = &s->in[i];
        float*       d = (float*)(E.x + (size_t)s->ids[fi->var].reg * SV_L);
        if (fi->builtin == BI_FragCoord) {
            FOR_L
            {
                int px          = sv_frag_pixel(l);
                d[l]            = (float)(io->x + (px & 31)) + 0.5f;
                d[SV_L + l]     = (float)(io->y + (px >> 5)) + 0.5f;
                d[2 * SV_L + l] = io->z ? io->z[px] : 0.0f;
                d[3 * SV_L + l] = 1.0f;
            }
            continue;
        }
        if (fi->builtin == BI_FrontFacing) {
            uint32_t ff = io->back_facing ? 0u : 1u;
            FOR_L((uint32_t*)d)[l] = ff;
            continue;
        }
        int slot = P->fslot[i];
        for (int c = 0; c < fi->comps; c++) {
            if (slot < 0 || slot + c >= FM3D_MAX_SHADER_VARYINGS) {
                FOR_L d[(size_t)c * SV_L + l] = 0.0f;
                continue;
            }
            sv_to_groups(d + (size_t)c * SV_L, io->varyings[slot + c]);
        }
    }
    E.M[SV_M_ENTRY] = lanes;
    sv_ir_run(&E);
    for (int i = 0; i < s->nout; i++) {
        const sv_io* fo = &s->out[i];
        if (fo->builtin == BI_FragDepth && io->depth_out) { /* gl_FragDepth */
            sv_from_groups(io->depth_out, (const float*)(E.x + (size_t)s->ids[fo->var].reg * SV_L));
            continue;
        }
        if (fo->loc != 0) continue;
        const float* src = (const float*)(E.x + (size_t)s->ids[fo->var].reg * SV_L);
        for (int c = 0; c < 4; c++) {
            if (c < fo->comps) sv_from_groups(io->out[c], src + (size_t)c * SV_L);
            else FOR_L io->out[c][l] = c == 3 ? 1.0f : 0.0f;
        }
    }
    FOR_L if ((E.M[SV_M_KILLED] >> l) & 1) io->mask[sv_frag_pixel(l)] = 0;
}

#endif /* FM_FEATURE_SPIRV */
