/*
 * fatmap - SPIR-V JIT: the stage entry points (inputs / outputs per 16
 * lane group, the frame), the C helpers the code calls, the reference
 * executor and executable memory.
 */
#if !defined(_WIN32) && !defined(_DEFAULT_SOURCE)
#  define _DEFAULT_SOURCE 1 /* MAP_ANONYMOUS under -std=c11 */
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE 1
#endif
#include "fm3d_jit.h"

#if FM_FEATURE_SPIRV && FM_FEATURE_JIT

#include "fatmap/fm_vmath.h"
#include <math.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif


/* ---- executable memory ---- */

void* fmj_code_alloc(size_t bytes)
{
#if defined(_WIN32)
    return VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
#endif
}

int fmj_code_seal(void* code, size_t bytes)
{
#if defined(_WIN32)
    DWORD old;
    if (!VirtualProtect(code, bytes, PAGE_EXECUTE_READ, &old)) return 0;
    FlushInstructionCache(GetCurrentProcess(), code, bytes);
    return 1;
#else
    return mprotect(code, bytes, PROT_READ | PROT_EXEC) == 0;
#endif
}

void fmj_code_free(void* code, size_t bytes)
{
    if (!code) return;
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(code, 0, MEM_RELEASE);
#else
    munmap(code, bytes);
#endif
}

/* ---- per thread scratch: the frame, padded uniform blocks, reference registers ---- */

typedef struct fmj_tls {
    uint8_t*  fr;
    size_t    frcap;
    uint8_t*  pad[FM3D_MAX_UNIFORM_BLOCKS];
    size_t    padcap[FM3D_MAX_UNIFORM_BLOCKS];
    uint32_t* regs;
    size_t    regcap;
} fmj_tls;

static fmj_tls* fmj_tls_get(void) { return (fmj_tls*)fm__tls(FM__TLS_JIT, sizeof(fmj_tls)); }

static uint8_t* fmj_frame_get(fmj_tls* T, size_t bytes)
{
    if (!T) return NULL;
    if (bytes + 64 > T->frcap) {
        free(T->fr);
        T->fr    = (uint8_t*)malloc(bytes + 64);
        T->frcap = T->fr ? bytes + 64 : 0;
    }
    if (!T->fr) return NULL;
    return (uint8_t*)(((uintptr_t)T->fr + 63) & ~(uintptr_t)63);
}

/* uniform blocks: every static read inside the pointer (a zero padded copy when the block is shorter) */
static void fmj_ubos(fmj_tls* T, fmj_frame* f, const fmj_prog* p, const void* uniforms, size_t uniform_size, const void* const* blocks,
                     const size_t* block_sizes)
{
    for (uint32_t m = p->ublocks; m; m &= m - 1) { /* the blocks the program reads */
        int b = 0;
        while (!(m >> b & 1)) b++;
        const uint8_t* ptr = blocks ? (const uint8_t*)blocks[b] : (b ? NULL : (const uint8_t*)uniforms);
        size_t         n   = blocks ? block_sizes[b] : (b || !uniforms ? 0 : uniform_size);
        if (!ptr) n = 0;
        f->ubo_n[b] = n;
        size_t need = (size_t)p->ubo_need[b];
        if (need > n) {
            if (need > T->padcap[b]) {
                free(T->pad[b]);
                T->pad[b]    = (uint8_t*)malloc(need);
                T->padcap[b] = T->pad[b] ? need : 0;
            }
            if (T->pad[b]) {
                memset(T->pad[b], 0, need);
                if (n) memcpy(T->pad[b], ptr, n);
                ptr = T->pad[b];
            }
        }
        f->ubo[b] = ptr;
    }
}

static void fmj_exec(const fmj_prog* p, uint8_t* F)
{
    if (p->fn) p->fn(F);
    else fmj_run_ref(p, F);
}

/* ---- stage entry points ---- */

void fmj_run_fs(const fm3d_fs_io* io)
{
    const fm3d_spirv* P = (const fm3d_spirv*)io->user;
    const fmj_prog*   p = P->jfs;
    const sv_stage*   s = P->fs;
    fmj_tls*          T = fmj_tls_get();
    uint8_t*          F = fmj_frame_get(T, (size_t)p->frame_size);
    if (!F) return;
    fmj_frame* f = (fmj_frame*)F;
    f->io = io, f->prog = p, f->fs = 1;
    fmj_ubos(T, f, p, io->uniforms, io->uniform_size, io->blocks, io->block_sizes);
    f->ubo[FMJ_PLANES] = (const uint8_t*)io->planes;
    uint32_t* M = (uint32_t*)(F + p->off_m);
    for (int g = 0; g < 4 && 8 * g < io->cols; g++) {
        if (io->mask) { /* no pixel of the group covered: nothing to shade (quads never span groups) */
            uint64_t m0, m1;
            memcpy(&m0, io->mask + 8 * g, 8), memcpy(&m1, io->mask + 32 + 8 * g, 8);
            if (!(m0 | m1)) continue;
        }
        f->group = g;
        f->nq    = io->cols - 8 * g >= 8 ? 4 : (io->cols - 8 * g) / 2;
        memset(M, 0, (size_t)p->nmask * 4);
        int      nc = io->cols - 8 * g; /* the group's columns (lanes l & 7 of both rows) */
        uint32_t e  = nc >= 8 ? 0xFFu : (1u << nc) - 1u;
        M[SV_M_ENTRY] = e | (e << 8);
        if (p->interp) { /* in words 0..2: dx, dy, w of the group's lanes (row 0, then row 1) */
            float* d = (float*)(F + p->off_in);
            memcpy(d, io->dx + 8 * g, 32), memcpy(d + 8, io->dx + 8 * g, 32);
            for (int l = 0; l < 8; l++) d[16 + l] = io->dy[0], d[24 + l] = io->dy[1];
            memcpy(d + 32, io->w + 8 * g, 32), memcpy(d + 40, io->w + 32 + 8 * g, 32);
        }
        for (int i = 0; i < s->nin; i++) {
            const sv_io* fi = &s->in[i];
            int          w  = p->vw[fi->var];
            if (w < 0) continue;
            float* d = (float*)(F + p->off_in + 64 * w);
            if (fi->builtin == BI_FragCoord) {
                for (int l = 0; l < FMJ_V; l++) {
                    int px    = sv_frag_pixel(16 * g + l);
                    d[l]      = (float)(io->x + (px & 31)) + 0.5f;
                    d[16 + l] = (float)(io->y + (px >> 5)) + 0.5f;
                    d[32 + l] = io->z ? io->z[px] : 0.0f;
                    d[48 + l] = 1.0f;
                }
                continue;
            }
            if (fi->builtin == BI_FrontFacing) {
                uint32_t ff = io->back_facing ? 0u : 0xffffffffu;
                for (int l = 0; l < FMJ_V; l++) ((uint32_t*)d)[l] = ff;
                continue;
            }
            int slot = P->fslot[i];
            for (int c = 0; c < fi->comps; c++) {
                float* dc = d + 16 * c;
                if (slot < 0 || slot + c >= FM3D_MAX_SHADER_VARYINGS) {
                    memset(dc, 0, 64);
                    continue;
                }
                memcpy(dc, io->varyings[slot + c] + 8 * g, 32); /* row 0 of the group, then row 1 */
                memcpy(dc + 8, io->varyings[slot + c] + 32 + 8 * g, 32);
            }
        }
        fmj_exec(p, F);
        for (int i = 0; i < s->nout; i++) {
            const sv_io* fo = &s->out[i];
            int          w  = p->vw[fo->var];
            if (w < 0) continue;
            const float* src = (const float*)(F + p->off_out + 64 * w);
            if (fo->builtin == BI_FragDepth) {
                if (io->depth_out) {
                    memcpy(io->depth_out + 8 * g, src, 32);
                    memcpy(io->depth_out + 32 + 8 * g, src + 8, 32);
                }
                continue;
            }
            if (fo->loc != 0) continue;
            if (io->color && p->packw >= 0) { /* the packed straight colors */
                const uint32_t* pk = (const uint32_t*)(F + p->off_out + 64 * p->packw);
                memcpy(io->color + 8 * g, pk, 32), memcpy(io->color + 32 + 8 * g, pk + 8, 32);
                continue;
            }
            for (int c = 0; c < 4; c++) {
                float* o0 = io->out[c] + 8 * g;
                float* o1 = io->out[c] + 32 + 8 * g;
                if (c < fo->comps) {
                    memcpy(o0, src + 16 * c, 32);
                    memcpy(o1, src + 16 * c + 8, 32);
                } else {
                    float v = c == 3 ? 1.0f : 0.0f;
                    for (int k = 0; k < 8; k++) o0[k] = o1[k] = v;
                }
            }
        }
        uint32_t k = M[SV_M_KILLED];
        for (int l = 0; k && l < FMJ_V; l++)
            if ((k >> l) & 1) io->mask[sv_frag_pixel(16 * g + l)] = 0;
    }
}

void fmj_run_vs(const fm3d_vs_io* io)
{
    const fm3d_spirv* P = (const fm3d_spirv*)io->user;
    const fmj_prog*   p = P->jvs;
    const sv_stage*   s = P->vs;
    fmj_tls*          T = fmj_tls_get();
    uint8_t*          F = fmj_frame_get(T, (size_t)p->frame_size);
    if (!F) return;
    fmj_frame* f = (fmj_frame*)F;
    f->io = io, f->prog = p, f->fs = 0, f->nq = 0;
    fmj_ubos(T, f, p, io->uniforms, io->uniform_size, io->blocks, io->block_sizes);
    uint32_t* M = (uint32_t*)(F + p->off_m);
    for (int base = 0; base < io->count; base += FMJ_V) {
        int n    = io->count - base < FMJ_V ? io->count - base : FMJ_V;
        f->group = base;
        memset(M, 0, (size_t)p->nmask * 4);
        M[SV_M_ENTRY] = n == FMJ_V ? 0xffffu : ((1u << n) - 1);
        for (int i = 0; i < s->nin; i++) {
            const sv_io* vi = &s->in[i];
            int          w  = p->vw[vi->var];
            if (w < 0) continue;
            float* d = (float*)(F + p->off_in + 64 * w);
            if (vi->builtin == BI_VertexIndex || vi->builtin == BI_InstanceIndex) {
                for (int l = 0; l < FMJ_V; l++) ((int32_t*)d)[l] = vi->builtin == BI_VertexIndex ? io->first_vertex + base + l : io->instance;
                continue;
            }
            const fm3d_vertex_attrib* a = NULL;
            for (int k = 0; k < P->nattr; k++)
                if (P->attr[k].location == vi->loc) a = &P->attr[k];
            for (int c = 0; c < vi->comps; c++)
                for (int l = 0; l < FMJ_V; l++) {
                    float v = c == 3 ? 1.0f : 0.0f;
                    if (l >= n) v = 0.0f;
                    else if (a && c < a->components) {
                        const char* vp = (const char*)io->vertices + (size_t)(base + l) * (size_t)io->stride + a->offset;
                        memcpy(&v, vp + 4 * c, 4);
                    }
                    d[16 * c + l] = v;
                }
        }
        fmj_exec(p, F);
        const float* pos = (const float*)(F + p->off_out + 64 * (p->vw[s->pos_var] + s->pos_off));
        for (int l = 0; l < n; l++) {
            float* o = io->pos + (size_t)(base + l) * (size_t)io->out_stride;
            for (int c = 0; c < 4; c++) o[c] = pos[16 * c + l];
        }
        if (s->ps_var >= 0 && P->ps_slot >= 0) {
            const float* ps = (const float*)(F + p->off_out + 64 * (p->vw[s->ps_var] + s->ps_off));
            for (int l = 0; l < n; l++) io->varyings[(size_t)(base + l) * (size_t)io->out_stride + (size_t)P->ps_slot] = ps[l];
        }
        for (int i = 0; i < s->nout; i++) {
            const sv_io* vo = &s->out[i];
            int          slot = P->vslot[i], w = p->vw[vo->var];
            if (vo->builtin >= 0 || slot < 0 || w < 0) continue;
            const float* src = (const float*)(F + p->off_out + 64 * w);
            for (int l = 0; l < n; l++) {
                float* q = io->varyings + (size_t)(base + l) * (size_t)io->out_stride;
                for (int c = 0; c < vo->comps && slot + c < FM3D_MAX_SHADER_VARYINGS; c++) q[slot + c] = src[16 * c + l];
            }
        }
    }
}

/* ---- helpers ---- */

static void fmj_h_sample(void* frame, const fmj_call* c)
{
    fmj_frame*                 f = (fmj_frame*)frame;
    uint8_t*                   F = (uint8_t*)frame;
    const float*               a = (const float*)(F + c->arg);
    float*                     out = (float*)(F + c->res);
    const fm3d_texture* const* tex = f->fs ? ((const fm3d_fs_io*)f->io)->textures : ((const fm3d_vs_io*)f->io)->textures;
    const fm3d_sampler*        smp = f->fs ? ((const fm3d_fs_io*)f->io)->samplers : ((const fm3d_vs_io*)f->io)->samplers;
    int                        op = c->fn, nc = c->ncoord;
    int                        proj = op == OpImageSampleProjImplicitLod || op == OpImageSampleProjExplicitLod;
    int                        expl = op == OpImageSampleExplicitLod || op == OpImageSampleProjExplicitLod;
    const float*               cu = a;
    const float*               cv = a + 16;
    float                      pu[16], pv[16];
    if (proj) {
        const float* q = a + 16 * (nc - 1);
        for (int l = 0; l < 16; l++) pu[l] = cu[l] / q[l], pv[l] = cv[l] / q[l];
        cu = pu, cv = pv;
    }
    const fm3d_texture* t = c->unit >= 0 && tex ? tex[c->unit] : NULL;
    if (!t) {
        for (int k = 0; k < 64; k++) out[k] = k >= 48 ? 1.0f : 0.0f;
        return;
    }
    const fm3d_sampler* s   = &smp[c->unit];
    const float*        lod = c->lod ? a + 16 * nc : NULL;
    if (c->dim != 1 || c->arrayed) { /* 1D, 3D, cube, rectangle, arrays */
        float        hv[16], ru[16], rv[16];
        const float *c0 = cu, *c1 = nc > 1 ? cv : NULL, *c2 = nc > 2 ? a + 32 : NULL;
        for (int l = 0; l < 16; l++) hv[l] = 0.5f;
        if (c->dim == 0) {
            c2 = c->arrayed ? c1 : NULL;
            c1 = hv;
        } else if (c->dim == 4) {
            float w = (float)t->level[0]->width, h = (float)t->level[0]->height;
            for (int l = 0; l < 16; l++) ru[l] = cu[l] / w, rv[l] = cv[l] / h;
            c0 = ru, c1 = rv, c2 = NULL;
        }
        fm3d_sample_tex(t, s, c0, c1 ? c1 : hv, c2, 16, !expl && f->fs ? f->nq : 0, lod, out, out + 16, out + 32, out + 48);
        return;
    }
    if (expl) {
        fm3d_sample_lod(t, s, cu, cv, lod, 16, out, out + 16, out + 32, out + 48);
        return;
    }
    if (f->fs) {
        fm3d_sample_quads(t, s, cu, cv, f->nq, out, out + 16, out + 32, out + 48);
        return;
    }
    fm3d_sample_lod(t, s, cu, cv, NULL, 16, out, out + 16, out + 32, out + 48);
}

/* fragment stage 2D implicit LOD: the quad group sampler directly (it declines textures
 * it does not cover: the general helper), the same bits as fm3d_sample_quads */
static void fmj_h_sample2(void* frame, const fmj_call* c)
{
    const fmj_frame*  f  = (const fmj_frame*)frame;
    const fm3d_fs_io* io = (const fm3d_fs_io*)f->io;
    uint8_t*          F  = (uint8_t*)frame;
    const float*      a  = (const float*)(F + c->arg);
    float*            o  = (float*)(F + c->res);
    const fm3d_texture* t = c->unit >= 0 && io->textures ? io->textures[c->unit] : NULL;
    if (!t) {
        fmj_h_sample(frame, c);
        return;
    }
    const float *cu = a, *cv = a + 16;
    float        pu[16], pv[16];
    if (c->fn == OpImageSampleProjImplicitLod) { /* textureProj: u / q, v / q, as the general helper */
        const float* q = a + 16 * (c->ncoord - 1);
        for (int l = 0; l < 16; l++) pu[l] = cu[l] / q[l], pv[l] = cv[l] / q[l];
        cu = pu, cv = pv;
    }
    if (c->s16(t, &io->samplers[c->unit], cu, cv, o, o + 16, o + 32, o + 48)) return;
    fmj_h_sample(frame, c);
}

static void fmj_h_math(void* frame, const fmj_call* c)
{
    uint8_t*     F = (uint8_t*)frame;
    const float* a = (const float*)(F + c->arg);
    const float* b = c->nargs > c->n ? a + FMJ_V * c->n : a; /* arrays: n components of x, then of y */
    float*       r = (float*)(F + c->res);
    int          N = FMJ_V * c->n, fast = c->s->fast;
    switch (fm_simd_current()) {
#ifdef FM_HAVE_AVX512_SPIRV
    case FM_SIMD_AVX512: sv_math_avx512(c->fn, fast, N, a, b, r); return;
#endif
#ifdef FM_HAVE_AVX2
    case FM_SIMD_AVX2: sv_math_avx2(c->fn, fast, N, a, b, r); return;
#endif
    default: sv_math_base(c->fn, fast, N, a, b, r); return;
    }
}

static void fmj_h_slow(void* frame, const fmj_call* c)
{
    uint8_t*        F  = (uint8_t*)frame;
    const uint32_t* ua = (const uint32_t*)(F + c->arg);
    const uint32_t* ub = ua + 16;
    uint32_t*       r  = (uint32_t*)(F + c->res);
    for (int k = 0; k < 16; k++) {
        uint32_t x = ua[k], y = ub[k];
        int32_t  sx = (int32_t)x, sy = (int32_t)y;
        float    fx, fy, fr;
        switch (c->fn) {
        case OpUDiv: r[k] = y ? x / y : 0u; break;
        case OpUMod: r[k] = y ? x % y : 0u; break;
        case OpSDiv: r[k] = (uint32_t)((sy == 0 || (sx == INT32_MIN && sy == -1)) ? 0 : sx / sy); break;
        case OpSRem: r[k] = (uint32_t)((sy == 0 || sy == -1) ? 0 : sx % sy); break;
        case OpSMod:
            r[k] = (uint32_t)((sy == 0 || sy == -1) ? 0 : ((sx % sy) != 0 && ((sx % sy) < 0) != (sy < 0) ? sx % sy + sy : sx % sy));
            break;
        case OpFRem:
            memcpy(&fx, &x, 4), memcpy(&fy, &y, 4);
            fr = fmodf(fx, fy);
            memcpy(&r[k], &fr, 4);
            break;
        default: r[k] = 0; break;
        }
    }
}

/* a uniform block value at per lane byte offsets (as sv_uload; bools all ones) */
static void fmj_uload(const fmj_frame* f, const sv_stage* s, int blk, int t, long base, const int32_t* off, int mstride, uint32_t* dst, int* c)
{
    const sv_id* T = &s->ids[t];
    switch (T->kind) {
    case T_BOOL: case T_INT: case T_UINT: case T_FLOAT: {
        uint32_t*      d = dst + 16 * (*c)++;
        const uint8_t* b = f->ubo[blk];
        size_t         n = f->ubo_n[blk];
        for (int l = 0; l < 16; l++) {
            long     o = base + off[l];
            uint32_t v = 0;
            if (b && o >= 0 && (size_t)o + 4 <= n) memcpy(&v, b + o, 4);
            if (T->kind == T_BOOL) v = v ? 0xffffffffu : 0u;
            d[l] = v;
        }
        break;
    }
    case T_VEC:
        for (int i = 0; i < T->count; i++) fmj_uload(f, s, blk, T->elem, base + 4 * i, off, 0, dst, c);
        break;
    case T_MAT:
        for (int i = 0; i < T->count; i++) fmj_uload(f, s, blk, T->elem, base + (long)i * mstride, off, 0, dst, c);
        break;
    case T_ARR:
        for (int i = 0; i < T->count; i++) fmj_uload(f, s, blk, T->elem, base + (long)i * T->astride, off, mstride, dst, c);
        break;
    case T_STRUCT:
        for (int m = 0; m < T->nmem; m++) fmj_uload(f, s, blk, T->mem[m], base + T->mboff[m], off, T->mstride[m], dst, c);
        break;
    default: break;
    }
}

static void fmj_h_uload(void* frame, const fmj_call* c)
{
    uint8_t* F = (uint8_t*)frame;
    int      k = 0;
    fmj_uload((const fmj_frame*)frame, c->s, c->binding, c->type, c->poff, (const int32_t*)(F + c->arg), c->mstride, (uint32_t*)(F + c->res), &k);
}

const fmj_helper fmj_helpers[FMJ_H_COUNT] = { fmj_h_sample, fmj_h_math, fmj_h_slow, fmj_h_uload, fmj_h_sample2 };

/* ---- reference executor ---- */

static int fmj_fcmp(float a, float b, uint32_t pr)
{
    int un = a != a || b != b;
    switch (pr) {
    case 0x00: return !un && a == b;
    case 0x11: return !un && a < b;
    case 0x12: return !un && a <= b;
    case 0x03: return un;
    case 0x04: return un || a != b;
    case 0x15: return !(a < b);
    case 0x16: return !(a <= b);
    case 0x07: return !un;
    case 0x08: return un || a == b;
    case 0x19: return !(a >= b);
    case 0x1a: return !(a > b);
    case 0x0c: return !un && a != b;
    case 0x1d: return !un && a >= b;
    case 0x1e: return !un && a > b;
    default: return 0;
    }
}

static int32_t fmj_cvtfi(float f)
{
    if (f != f || f >= 2147483648.0f || f < -2147483648.0f) return INT32_MIN;
    return (int32_t)f;
}

void fmj_run_ref(const fmj_prog* p, void* frame)
{
    uint8_t* F = (uint8_t*)frame;
    size_t   need = (size_t)p->nv * 16;
    fmj_tls* T = fmj_tls_get();
    if (!T) return;
    if (need > T->regcap) {
        free(T->regs);
        T->regs   = (uint32_t*)malloc(need * 4);
        T->regcap = T->regs ? need : 0;
    }
    uint32_t* V = T->regs;
    if (!V) return;
    for (int v = 0; v < p->nv; v++)
        if (p->vk[v] == FMJ_K_CONST)
            for (int l = 0; l < 16; l++) V[16 * v + l] = p->vc[v];
    uint32_t*        M  = (uint32_t*)(F + p->off_m);
    const fmj_frame* fr = (const fmj_frame*)frame;
#define RU(v) (V + 16 * (size_t)(v))
#define RF(v) ((float*)(V + 16 * (size_t)(v)))
#define RI(v) ((int32_t*)(V + 16 * (size_t)(v)))
    for (int pc = 0; pc < p->nops;) {
        const fmj_op* o = &p->ops[pc];
        uint32_t*     d = o->d >= 0 ? RU(o->d) : NULL;
        float*        df = (float*)d;
        int32_t*      di = (int32_t*)d;
        uint32_t      t[16];
        switch (o->op) {
        case J_MOV: memcpy(t, RU(o->a), 64), memcpy(d, t, 64); break;
        case J_FADD: for (int l = 0; l < 16; l++) ((float*)t)[l] = RF(o->a)[l] + RF(o->b)[l]; memcpy(d, t, 64); break;
        case J_FSUB: for (int l = 0; l < 16; l++) ((float*)t)[l] = RF(o->a)[l] - RF(o->b)[l]; memcpy(d, t, 64); break;
        case J_FMUL: for (int l = 0; l < 16; l++) ((float*)t)[l] = RF(o->a)[l] * RF(o->b)[l]; memcpy(d, t, 64); break;
        case J_FDIV: for (int l = 0; l < 16; l++) ((float*)t)[l] = RF(o->a)[l] / RF(o->b)[l]; memcpy(d, t, 64); break;
        case J_FSQRT: for (int l = 0; l < 16; l++) df[l] = sqrtf(RF(o->a)[l]); break;
        case J_FCMP: for (int l = 0; l < 16; l++) t[l] = fmj_fcmp(RF(o->a)[l], RF(o->b)[l], o->imm) ? 0xffffffffu : 0u; memcpy(d, t, 64); break;
        case J_IADD: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] + RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_ISUB: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] - RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_IMUL: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] * RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_AND: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] & RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_OR: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] | RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_XOR: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] ^ RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_ANDN: for (int l = 0; l < 16; l++) t[l] = ~RU(o->a)[l] & RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_SHL: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] << (RU(o->b)[l] & 31); memcpy(d, t, 64); break;
        case J_SHR: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] >> (RU(o->b)[l] & 31); memcpy(d, t, 64); break;
        case J_SAR: for (int l = 0; l < 16; l++) t[l] = (uint32_t)(RI(o->a)[l] >> (RU(o->b)[l] & 31)); memcpy(d, t, 64); break;
        case J_SHLI: for (int l = 0; l < 16; l++) d[l] = RU(o->a)[l] << o->imm; break;
        case J_SHRI: for (int l = 0; l < 16; l++) d[l] = RU(o->a)[l] >> o->imm; break;
        case J_SARI: for (int l = 0; l < 16; l++) di[l] = RI(o->a)[l] >> o->imm; break;
        case J_ICMPEQ: for (int l = 0; l < 16; l++) t[l] = RU(o->a)[l] == RU(o->b)[l] ? 0xffffffffu : 0u; memcpy(d, t, 64); break;
        case J_ICMPGT: for (int l = 0; l < 16; l++) t[l] = RI(o->a)[l] > RI(o->b)[l] ? 0xffffffffu : 0u; memcpy(d, t, 64); break;
        case J_CVTIF: for (int l = 0; l < 16; l++) df[l] = (float)RI(o->a)[l]; break;
        case J_CVTUF: for (int l = 0; l < 16; l++) df[l] = (float)RU(o->a)[l]; break;
        case J_CVTFI: for (int l = 0; l < 16; l++) di[l] = fmj_cvtfi(RF(o->a)[l]); break;
        case J_SEL: for (int l = 0; l < 16; l++) t[l] = RU(o->c)[l] ? RU(o->a)[l] : RU(o->b)[l]; memcpy(d, t, 64); break;
        case J_DDX: for (int l = 0; l < 16; l++) ((float*)t)[l] = RF(o->a)[l | 1] - RF(o->a)[l & ~1]; memcpy(d, t, 64); break;
        case J_DDY: for (int l = 0; l < 16; l++) ((float*)t)[l] = RF(o->a)[l | 8] - RF(o->a)[l & ~8]; memcpy(d, t, 64); break;
        case J_LDF: memcpy(d, F + o->imm, 64); break;
        case J_STF: memcpy(F + o->imm, RU(o->a), 64); break;
        case J_LDU: {
            uint32_t v;
            memcpy(&v, fr->ubo[o->b] + o->imm, 4);
            for (int l = 0; l < 16; l++) d[l] = v;
            break;
        }
        case J_MASKV: for (int l = 0; l < 16; l++) d[l] = (M[o->imm] >> l) & 1 ? 0xffffffffu : 0u; break;
        case J_MZERO: M[o->a] = 0; break;
        case J_MCOPY: M[o->a] = M[o->b]; break;
        case J_MOR: M[o->a] = M[o->b] | M[o->c]; break;
        case J_MAND: M[o->a] = M[o->b] & M[o->c]; break;
        case J_MANDN: M[o->a] = M[o->b] & ~M[o->c]; break;
        case J_MCOND: {
            uint32_t m = 0;
            for (int l = 0; l < 16; l++) m |= (RU(o->b)[l] != 0 ? 1u : 0u) << l;
            M[o->a] = m & M[o->c];
            break;
        }
        case J_IF:
            if (!M[o->a]) {
                pc = o->c + 1;
                continue;
            }
            break;
        case J_LOOP: M[o->a] = 0; break;
        case J_BREAKZ:
            if (!M[o->a] || M[o->b]++ >= SV_MAXITER) {
                pc = o->c + 1;
                continue;
            }
            break;
        case J_BREAK: pc = o->c + 1; continue;
        case J_ENDLOOP: pc = o->c + 1; continue;
        case J_CALL: fmj_helpers[o->imm](frame, &p->calls[o->a]); break;
        default: break;
        }
        pc++;
    }
#undef RU
#undef RF
#undef RI
}

#endif
