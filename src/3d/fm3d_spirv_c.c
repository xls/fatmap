/*
 * fatmap - SPIR-V to C (ahead of time compilation of fm3d_spirv programs).
 *
 * Prints the lowered program (fm3d_spirv_internal.h) as C: the stages
 * become fm3d_program callbacks over the same 64 lane batches the
 * interpreter runs, with the control flow as plain C over 64 bit lane
 * masks and every segment of a block fused into one loop over the lanes
 * (values that never leave their segment live in registers; lane uniform
 * values are scalars computed once). Each operation is the interpreter's
 * formula, so with the same floating point settings (no contraction) the
 * generated program renders bit identical images.
 *
 * The output only needs <fatmap/fatmap.h> built with -Dshaders (not the
 * SPIR-V backend). sin / cos / tan / exp / log / pow are fm_vmath.h's
 * (deterministic, vectorizable); the other GLSL functions call libm.
 */
#include "fm3d_spirv_internal.h"

#if FM_FEATURE_SPIRV

#include <math.h>
#include <stdarg.h>
#include <stdio.h>

enum { K_F = 0, K_U, K_I, K_W }; /* float, uint32 (bool), int32, mixed (union) */

typedef struct cg_buf {
    char*  p;
    size_t n, cap;
    int    fail;
} cg_buf;

typedef struct cg {
    cg_buf            o;
    const fm3d_spirv* P;
    const sv_stage*   s;
    int               fs;   /* fragment stage */
    int               ind;  /* indentation */
    int               seg;  /* segment whose locals are in scope (-1: none) */
    int               tmp;  /* temporaries */
    int               ni;
    char              nb[96][192]; /* rotating expression buffers */
    char*             err;
    size_t            errn;
    int               failed;
} cg;

static void cg_vput(cg_buf* b, const char* fmt, va_list ap)
{
    if (b->fail) return;
    va_list a2;
    va_copy(a2, ap);
    int n = vsnprintf(NULL, 0, fmt, a2);
    va_end(a2);
    if (n < 0) {
        b->fail = 1;
        return;
    }
    if (b->n + (size_t)n + 1 > b->cap) {
        size_t c  = b->cap ? b->cap : 65536;
        while (c < b->n + (size_t)n + 1) c *= 2;
        char* np = (char*)realloc(b->p, c);
        if (!np) {
            b->fail = 1;
            return;
        }
        b->p   = np;
        b->cap = c;
    }
    vsnprintf(b->p + b->n, (size_t)n + 1, fmt, ap);
    b->n += (size_t)n;
}

static void cg_put(cg* g, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    cg_vput(&g->o, fmt, ap);
    va_end(ap);
}

/* one indented line */
static void cg_line(cg* g, const char* fmt, ...)
{
    for (int i = 0; i < g->ind; i++) cg_put(g, "    ");
    va_list ap;
    va_start(ap, fmt);
    cg_vput(&g->o, fmt, ap);
    va_end(ap);
    cg_put(g, "\n");
}

static void cg_fail(cg* g, const char* fmt, ...)
{
    if (g->failed) return;
    g->failed = 1;
    if (g->err && g->errn) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(g->err, g->errn, fmt, ap);
        va_end(ap);
    }
}

static char* cg_str(cg* g, const char* fmt, ...)
{
    char*   b = g->nb[g->ni++ % 96];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof(g->nb[0]), fmt, ap);
    va_end(ap);
    if (n < 0 || n >= (int)sizeof(g->nb[0])) cg_fail(g, "internal: expression too long");
    return b;
}

/* ---- types ---- */

/* kind of flattened component c of type t */
static int cg_ck(const sv_stage* s, int t, int c)
{
    for (int guard = 0; guard < 64; guard++) {
        const sv_id* T = &s->ids[t];
        switch (T->kind) {
        case T_FLOAT: return K_F;
        case T_INT: return K_I;
        case T_BOOL: case T_UINT: return K_U;
        case T_VEC: case T_MAT: case T_ARR: {
            int ec = s->ids[T->elem].comps;
            if (ec <= 0) return K_U;
            c %= ec;
            t = T->elem;
            break;
        }
        case T_STRUCT: {
            int m = T->nmem - 1;
            while (m > 0 && T->moff[m] > c) m--;
            if (m < 0) return K_U;
            c -= T->moff[m];
            t = T->mem[m];
            break;
        }
        default: return K_U;
        }
    }
    return K_U;
}

static int cg_vk(const cg* g, int id, int c) { return cg_ck(g->s, g->s->ids[id].type, c); }

/* kind of an array holding every component of type t */
static int cg_ak(const sv_stage* s, int t)
{
    int n = s->ids[t].comps, k = n > 0 ? cg_ck(s, t, 0) : K_U;
    for (int c = 1; c < n; c++)
        if (cg_ck(s, t, c) != k) return K_W;
    return k;
}

static const char* cg_tn(int k)
{
    static const char* n[4] = { "float", "uint32_t", "int32_t", "spv_w" };
    return n[k];
}

static const char* cg_mem(int k)
{
    static const char* n[3] = { ".f", ".u", ".i" };
    return n[k];
}

/* bit preserving reinterpretation */
static const char* cg_conv(cg* g, int from, int to, const char* e)
{
    if (from == to) return e;
    if (from == K_F) return to == K_U ? cg_str(g, "spv_uf(%s)", e) : cg_str(g, "(int32_t)spv_uf(%s)", e);
    if (to == K_F) return from == K_U ? cg_str(g, "spv_fu(%s)", e) : cg_str(g, "spv_fu((uint32_t)(%s))", e);
    return to == K_U ? cg_str(g, "(uint32_t)(%s)", e) : cg_str(g, "(int32_t)(%s)", e);
}

/* element of an array of kind ak, read as `want` */
static const char* cg_at(cg* g, const char* arr, int ak, const char* idx, int want)
{
    if (ak == K_W) return cg_str(g, "%s[%s]%s", arr, idx, cg_mem(want));
    return cg_conv(g, ak, want, cg_str(g, "%s[%s]", arr, idx));
}

/* the same as an lvalue of the component's own kind */
static const char* cg_lv(cg* g, const char* arr, int ak, const char* idx, int k)
{
    if (ak == K_W) return cg_str(g, "%s[%s]%s", arr, idx, cg_mem(k));
    return cg_str(g, "%s[%s]", arr, idx);
}

static const char* cg_lit(cg* g, uint32_t bits, int want)
{
    if (want == K_U) return bits < 1000000u ? cg_str(g, "%uu", bits) : cg_str(g, "0x%08xu", bits);
    if (want == K_I) {
        int32_t v = (int32_t)bits;
        if (v > -1000000 && v < 1000000) return v < 0 ? cg_str(g, "(%d)", v) : cg_str(g, "%d", v);
        return cg_str(g, "(int32_t)0x%08xu", bits);
    }
    float f;
    memcpy(&f, &bits, 4);
    if (isfinite(f)) { /* shortest exact decimal of the float, else its bits */
        char t[64];
        for (int prec = 6; prec <= 9; prec++) {
            snprintf(t, sizeof(t), "%.*g", prec, (double)f);
            float    r  = strtof(t, NULL);
            uint32_t rb;
            memcpy(&rb, &r, 4);
            if (rb != bits) continue;
            int frac = strchr(t, '.') || strchr(t, 'e') || strchr(t, 'n') || strchr(t, 'i');
            return cg_str(g, "%s%s%sf%s", t[0] == '-' ? "(" : "", t, frac ? "" : ".0", t[0] == '-' ? ")" : "");
        }
    }
    return cg_str(g, "spv_fu(0x%08xu)", bits);
}

/* ---- value references ---- */

/* component c of value id as `want`; inside a segment loop its locals are
 * in scope (g->seg), otherwise per lane values are array reads at lane l */
static const char* cg_ref(cg* g, int id, int c, int want)
{
    const sv_stage* s = g->s;
    const sv_id*    d = &s->ids[id];
    int             k = cg_vk(g, id, c);
    switch (s->vcls[id]) {
    case VC_CONST: return cg_lit(g, sv_cblock(s, d)[(size_t)c * SV_L], want);
    case VC_UNIFORM: return cg_conv(g, k, want, cg_str(g, "u%d_%d", id, c));
    case VC_LOCAL:
        if (s->vseg[id] != g->seg) cg_fail(g, "internal: local %d read outside its segment", id);
        return cg_conv(g, k, want, cg_str(g, "v%d_%d", id, c));
    case VC_MAT:
        if (s->vseg[id] == g->seg) return cg_conv(g, k, want, cg_str(g, "v%d_%d", id, c));
        return cg_at(g, cg_str(g, "r%d", id), cg_ak(s, d->type), cg_str(g, "%d + l", c * SV_L), want);
    default: cg_fail(g, "internal: value %d has no class", id); return "0";
    }
}

/* a temporary holding an expression (operands used more than once) */
static const char* cg_tmp(cg* g, int k, const char* e)
{
    int t = g->tmp++;
    cg_line(g, "const %s t%d = %s;", cg_tn(k), t, e);
    return cg_str(g, "t%d", t);
}

/* define component c of value id from an expression of kind ek */
static void cg_def(cg* g, int id, int c, int ek, const char* e)
{
    const sv_stage* s = g->s;
    int             k = cg_vk(g, id, c);
    e                 = cg_conv(g, ek, k, e);
    if (s->vcls[id] == VC_UNIFORM) {
        cg_line(g, "u%d_%d = %s;", id, c, e);
        return;
    }
    cg_line(g, "const %s v%d_%d = %s;", cg_tn(k), id, c, e);
    if (s->vcls[id] == VC_MAT)
        cg_line(g, "%s = v%d_%d;", cg_lv(g, cg_str(g, "r%d", id), cg_ak(s, s->ids[id].type), cg_str(g, "%d + l", c * SV_L), k), id, c);
}

/* substitute @x / @y / @z in a template */
static const char* cg_subst(cg* g, const char* tpl, const char* x, const char* y, const char* z)
{
    char* b = g->nb[g->ni++ % 96];
    size_t n = 0, cap = sizeof(g->nb[0]);
    for (const char* p = tpl; *p && n + 1 < cap; p++) {
        const char* r = NULL;
        if (p[0] == '@' && (p[1] == 'x' || p[1] == 'y' || p[1] == 'z')) r = p[1] == 'x' ? x : (p[1] == 'y' ? y : z);
        if (r) {
            size_t m = strlen(r);
            if (n + m + 1 >= cap) break;
            memcpy(b + n, r, m);
            n += m;
            p++;
        } else {
            b[n++] = *p;
        }
    }
    b[n] = 0;
    if (n + 2 >= cap) cg_fail(g, "internal: expression too long");
    return b;
}

/* ---- uniform block loads (std140 offsets, as sv_uload) ---- */

static void cg_uload(cg* g, int id, int t, long base, int mstride, const char* dyn, int* c)
{
    const sv_stage* s = g->s;
    const sv_id*    T = &s->ids[t];
    switch (T->kind) {
    case T_BOOL: case T_INT: case T_UINT: case T_FLOAT: {
        const char* e = dyn ? cg_str(g, "spv_u32(ubo, ubo_n, %ldL + (long)(%s))", base, dyn) : cg_str(g, "spv_u32(ubo, ubo_n, %ldL)", base);
        if (T->kind == T_BOOL) e = cg_str(g, "(uint32_t)(%s != 0u)", e);
        cg_def(g, id, (*c)++, K_U, e);
        break;
    }
    case T_VEC:
        for (int i = 0; i < T->count; i++) cg_uload(g, id, T->elem, base + 4 * i, 0, dyn, c);
        break;
    case T_MAT:
        for (int i = 0; i < T->count; i++) cg_uload(g, id, T->elem, base + (long)i * mstride, 0, dyn, c);
        break;
    case T_ARR:
        for (int i = 0; i < T->count; i++) cg_uload(g, id, T->elem, base + (long)i * T->astride, mstride, dyn, c);
        break;
    case T_STRUCT:
        for (int m = 0; m < T->nmem; m++) cg_uload(g, id, T->mem[m], base + T->mboff[m], T->mstride[m], dyn, c);
        break;
    default: break;
    }
}

/* the dynamic offset of a pointer (sum of index * stride), NULL if static */
static const char* cg_dyn(cg* g, const sv_id* P)
{
    if (!P->ndyn) return NULL;
    const char* e = cg_str(g, "%s * %d", cg_ref(g, P->dyn[0], 0, K_I), P->dstride[0]);
    for (int k = 1; k < P->ndyn; k++) e = cg_str(g, "%s + %s * %d", e, cg_ref(g, P->dyn[k], 0, K_I), P->dstride[k]);
    return cg_tmp(g, K_I, e);
}

/* a clamped component offset into variable V for `comps` components */
static const char* cg_varoff(cg* g, const sv_id* P, const sv_id* V, int comps)
{
    const char* d = cg_dyn(g, P);
    int         maxo = V->comps - comps;
    const char* o = cg_tmp(g, K_I, cg_str(g, "%d + %s", P->poff, d));
    return cg_tmp(g, K_I, cg_str(g, "%s < 0 ? 0 : (%s > %d ? %d : %s)", o, o, maxo, maxo, o));
}

/* ---- GLSL.std.450 ---- */

static void cg_ext(cg* g, const uint32_t* in)
{
    const sv_stage* s  = g->s;
    int             id = (int)in[2], n = s->ids[id].comps, fn = (int)in[4], wc = (int)WC(in);
    int             A = wc > 5 ? (int)in[5] : 0, B = wc > 6 ? (int)in[6] : 0, C = wc > 7 ? (int)in[7] : 0;
    const char*     tpl = NULL;
    int             k = K_F; /* operand / result kind of the per component templates */
    switch (fn) {
    case 1: tpl = "fm_roundf(@x)"; break;
    case 2: tpl = "fm_rintf(@x)"; break;
    case 3: tpl = "fm_truncf(@x)"; break;
    case 4: tpl = "fabsf(@x)"; break;
    case 5: tpl = "@x < 0 ? (int32_t)(0u - (uint32_t)@x) : @x", k = K_I; break;
    case 6: tpl = "@x > 0 ? 1.0f : (@x < 0 ? -1.0f : 0.0f)"; break;
    case 7: tpl = "@x > 0 ? 1 : (@x < 0 ? -1 : 0)", k = K_I; break;
    case 8: tpl = "fm_floorf(@x)"; break;
    case 9: tpl = "fm_ceilf(@x)"; break;
    case 10: tpl = "@x - fm_floorf(@x)"; break;
    case 11: tpl = "@x * 0.017453292519943295f"; break;
    case 12: tpl = "@x * 57.29577951308232f"; break;
    case 13: tpl = "fm_sinf(@x)"; break;
    case 14: tpl = "fm_cosf(@x)"; break;
    case 15: tpl = "fm_tanf(@x)"; break;
    case 16: tpl = "asinf(@x)"; break;
    case 17: tpl = "acosf(@x)"; break;
    case 18: tpl = "atanf(@x)"; break;
    case 19: tpl = "sinhf(@x)"; break;
    case 20: tpl = "coshf(@x)"; break;
    case 21: tpl = "tanhf(@x)"; break;
    case 25: tpl = "atan2f(@x, @y)"; break;
    case 26: tpl = "fm_powf(@x, @y)"; break;
    case 27: tpl = "fm_expf(@x)"; break;
    case 28: tpl = "fm_logf(@x)"; break;
    case 29: tpl = "fm_exp2f(@x)"; break;
    case 30: tpl = "fm_log2f(@x)"; break;
    case 31: tpl = "sqrtf(@x)"; break;
    case 32: tpl = "1.0f / sqrtf(@x)"; break;
    case 37: case 79: tpl = "fm_fminf(@x, @y)"; break;
    case 38: tpl = "@x < @y ? @x : @y", k = K_U; break;
    case 39: tpl = "@x < @y ? @x : @y", k = K_I; break;
    case 40: case 80: tpl = "fm_fmaxf(@x, @y)"; break;
    case 41: tpl = "@x > @y ? @x : @y", k = K_U; break;
    case 42: tpl = "@x > @y ? @x : @y", k = K_I; break;
    case 43: case 81: tpl = "fm_fminf(fm_fmaxf(@x, @y), @z)"; break;
    case 44: tpl = "(@x > @y ? @x : @y) < @z ? (@x > @y ? @x : @y) : @z", k = K_U; break;
    case 45: tpl = "(@x > @y ? @x : @y) < @z ? (@x > @y ? @x : @y) : @z", k = K_I; break;
    case 46: tpl = "@x * (1.0f - @z) + @y * @z"; break;
    case 48: tpl = "@y < @x ? 0.0f : 1.0f"; break;
    case 50: tpl = "@x * @y + @z"; break;
    default: break;
    }
    if (tpl) {
        for (int c = 0; c < n; c++) {
            const char* x = A ? cg_tmp(g, k, cg_ref(g, A, c, k)) : "0";
            const char* y = B ? cg_tmp(g, k, cg_ref(g, B, c, k)) : "0";
            const char* z = C ? cg_tmp(g, k, cg_ref(g, C, c, k)) : "0";
            cg_def(g, id, c, k, cg_subst(g, tpl, x, y, z));
        }
        return;
    }
    switch (fn) {
    case 49: /* smoothstep */
        for (int c = 0; c < n; c++) {
            const char* x = cg_tmp(g, K_F, cg_ref(g, A, c, K_F));
            const char* y = cg_tmp(g, K_F, cg_ref(g, B, c, K_F));
            const char* z = cg_tmp(g, K_F, cg_ref(g, C, c, K_F));
            const char* t = cg_tmp(g, K_F, cg_str(g, "fm_fminf(fm_fmaxf((%s - %s) / (%s - %s), 0.0f), 1.0f)", z, x, y, x));
            cg_def(g, id, c, K_F, cg_str(g, "%s * %s * (3.0f - 2.0f * %s)", t, t, t));
        }
        break;
    case 66: case 67: { /* length, distance */
        int         m = s->ids[A].comps;
        const char* d = "0.0f";
        for (int j = 0; j < m; j++) {
            const char* v = fn == 66 ? cg_tmp(g, K_F, cg_ref(g, A, j, K_F))
                                     : cg_tmp(g, K_F, cg_str(g, "%s - %s", cg_ref(g, A, j, K_F), cg_ref(g, B, j, K_F)));
            d = cg_tmp(g, K_F, cg_str(g, "%s + %s * %s", d, v, v));
        }
        cg_def(g, id, 0, K_F, cg_str(g, "sqrtf(%s)", d));
        break;
    }
    case 68: { /* cross */
        const char* a[3];
        const char* b[3];
        for (int j = 0; j < 3; j++) a[j] = cg_tmp(g, K_F, cg_ref(g, A, j, K_F)), b[j] = cg_tmp(g, K_F, cg_ref(g, B, j, K_F));
        cg_def(g, id, 0, K_F, cg_str(g, "%s * %s - %s * %s", a[1], b[2], a[2], b[1]));
        cg_def(g, id, 1, K_F, cg_str(g, "%s * %s - %s * %s", a[2], b[0], a[0], b[2]));
        cg_def(g, id, 2, K_F, cg_str(g, "%s * %s - %s * %s", a[0], b[1], a[1], b[0]));
        break;
    }
    case 69: { /* normalize */
        const char* d = "0.0f";
        for (int j = 0; j < n; j++) {
            const char* v = cg_ref(g, A, j, K_F);
            d             = cg_tmp(g, K_F, cg_str(g, "%s + %s * %s", d, v, v));
        }
        const char* inv = cg_tmp(g, K_F, cg_str(g, "1.0f / sqrtf(%s)", d));
        for (int j = 0; j < n; j++) cg_def(g, id, j, K_F, cg_str(g, "%s * %s", cg_ref(g, A, j, K_F), inv));
        break;
    }
    case 70: case 71: case 72: { /* faceforward(N, I, Nref), reflect(I, N), refract(I, N, eta) */
        const char* d = "0.0f";
        for (int j = 0; j < n; j++)
            d = cg_tmp(g, K_F, fn == 70 ? cg_str(g, "%s + %s * %s", d, cg_ref(g, C, j, K_F), cg_ref(g, B, j, K_F))
                                        : cg_str(g, "%s + %s * %s", d, cg_ref(g, B, j, K_F), cg_ref(g, A, j, K_F)));
        if (fn == 70) {
            for (int j = 0; j < n; j++) {
                const char* a = cg_ref(g, A, j, K_F);
                cg_def(g, id, j, K_F, cg_str(g, "%s < 0 ? %s : -%s", d, a, a));
            }
        } else if (fn == 71) {
            for (int j = 0; j < n; j++)
                cg_def(g, id, j, K_F, cg_str(g, "%s - 2.0f * %s * %s", cg_ref(g, A, j, K_F), d, cg_ref(g, B, j, K_F)));
        } else {
            const char* eta = cg_tmp(g, K_F, cg_ref(g, C, 0, K_F));
            const char* kk  = cg_tmp(g, K_F, cg_str(g, "1.0f - %s * %s * (1.0f - %s * %s)", eta, eta, d, d));
            for (int j = 0; j < n; j++)
                cg_def(g, id, j, K_F,
                       cg_str(g, "%s < 0 ? 0.0f : %s * %s - (%s * %s + sqrtf(%s)) * %s", kk, eta, cg_ref(g, A, j, K_F), eta, d,
                              kk, cg_ref(g, B, j, K_F)));
        }
        break;
    }
    default: cg_fail(g, "GLSL.std.450 function %d is not supported by the C backend", fn); break;
    }
}

/* ---- one per lane (or lane uniform) instruction ---- */

typedef struct cg_bin {
    int         op, k, cmp; /* operand kind, result is a 0 / 1 compare */
    const char* tpl;
} cg_bin;

static const cg_bin cg_bins[] = {
    { OpFAdd, K_F, 0, "@x + @y" }, { OpFSub, K_F, 0, "@x - @y" }, { OpFMul, K_F, 0, "@x * @y" },
    { OpFDiv, K_F, 0, "@x / @y" }, { OpFRem, K_F, 0, "fmodf(@x, @y)" }, { OpFMod, K_F, 0, "@x - @y * fm_floorf(@x / @y)" },
    { OpIAdd, K_U, 0, "@x + @y" }, { OpISub, K_U, 0, "@x - @y" }, { OpIMul, K_U, 0, "@x * @y" },
    { OpUDiv, K_U, 0, "@y ? @x / @y : 0u" }, { OpUMod, K_U, 0, "@y ? @x % @y : 0u" },
    { OpSDiv, K_I, 0, "(@y == 0 || (@x == INT32_MIN && @y == -1)) ? 0 : @x / @y" },
    { OpSRem, K_I, 0, "(@y == 0 || @y == -1) ? 0 : @x % @y" },
    { OpSMod, K_I, 0, "(@y == 0 || @y == -1) ? 0 : ((@x % @y) != 0 && ((@x % @y) < 0) != (@y < 0) ? @x % @y + @y : @x % @y)" },
    { OpShiftRightLogical, K_U, 0, "@x >> (@y & 31)" }, { OpShiftRightArithmetic, K_I, 0, "(int32_t)(@x >> (@y & 31))" },
    { OpShiftLeftLogical, K_U, 0, "@x << (@y & 31)" }, { OpBitwiseOr, K_U, 0, "@x | @y" },
    { OpBitwiseXor, K_U, 0, "@x ^ @y" }, { OpBitwiseAnd, K_U, 0, "@x & @y" },
    { OpLogicalOr, K_U, 1, "(@x | @y) != 0" }, { OpLogicalAnd, K_U, 1, "(@x != 0) & (@y != 0)" },
    { OpLogicalEqual, K_U, 1, "(@x != 0) == (@y != 0)" }, { OpLogicalNotEqual, K_U, 1, "(@x != 0) != (@y != 0)" },
    { OpFOrdEqual, K_F, 1, "@x == @y" }, { OpFUnordEqual, K_F, 1, "!(@x != @y) || @x != @x || @y != @y" },
    { OpFOrdNotEqual, K_F, 1, "@x != @y && @x == @x && @y == @y" }, { OpFUnordNotEqual, K_F, 1, "@x != @y" },
    { OpFOrdLessThan, K_F, 1, "@x < @y" }, { OpFUnordLessThan, K_F, 1, "!(@x >= @y)" },
    { OpFOrdGreaterThan, K_F, 1, "@x > @y" }, { OpFUnordGreaterThan, K_F, 1, "!(@x <= @y)" },
    { OpFOrdLessThanEqual, K_F, 1, "@x <= @y" }, { OpFUnordLessThanEqual, K_F, 1, "!(@x > @y)" },
    { OpFOrdGreaterThanEqual, K_F, 1, "@x >= @y" }, { OpFUnordGreaterThanEqual, K_F, 1, "!(@x < @y)" },
    { OpIEqual, K_U, 1, "@x == @y" }, { OpINotEqual, K_U, 1, "@x != @y" }, { OpUGreaterThan, K_U, 1, "@x > @y" },
    { OpUGreaterThanEqual, K_U, 1, "@x >= @y" }, { OpULessThan, K_U, 1, "@x < @y" }, { OpULessThanEqual, K_U, 1, "@x <= @y" },
    { OpSGreaterThan, K_I, 1, "@x > @y" }, { OpSGreaterThanEqual, K_I, 1, "@x >= @y" }, { OpSLessThan, K_I, 1, "@x < @y" },
    { OpSLessThanEqual, K_I, 1, "@x <= @y" },
};

static void cg_inst(cg* g, const uint32_t* in, int op, int n)
{
    const sv_stage* s = g->s;
    int             id = (int)in[2], wc = (int)WC(in);
    for (size_t b = 0; b < sizeof(cg_bins) / sizeof(cg_bins[0]); b++) {
        if (cg_bins[b].op != op) continue;
        int k = cg_bins[b].k;
        for (int c = 0; c < n; c++) {
            const char* x = cg_tmp(g, k, cg_ref(g, (int)in[3], c, k));
            const char* y = cg_tmp(g, k, cg_ref(g, (int)in[4], c, k));
            const char* e = cg_subst(g, cg_bins[b].tpl, x, y, NULL);
            if (cg_bins[b].cmp) cg_def(g, id, c, K_U, cg_str(g, "(%s) ? 1u : 0u", e));
            else cg_def(g, id, c, k, e);
        }
        return;
    }
    switch (op) {
    case OpLoad: {
        const sv_id* P = &s->ids[in[3]];
        const sv_id* V = &s->ids[P->pvar];
        if (V->storage == SC_Uniform || V->storage == SC_PushConstant) {
            int c = 0;
            cg_uload(g, id, P->type, P->poff, P->pmstride, cg_dyn(g, P), &c);
            break;
        }
        int         ak  = cg_ak(s, V->type);
        const char* arr = cg_str(g, "x%d", P->pvar);
        if (!P->ndyn) {
            for (int c = 0; c < n; c++) {
                int k = cg_vk(g, id, c);
                cg_def(g, id, c, k, cg_at(g, arr, ak, cg_str(g, "%d + l", (P->poff + c) * SV_L), k));
            }
        } else {
            const char* o = cg_varoff(g, P, V, n);
            for (int c = 0; c < n; c++) {
                int k = cg_vk(g, id, c);
                cg_def(g, id, c, k, cg_at(g, arr, ak, cg_str(g, "(%s + %d) * 64 + l", o, c), k));
            }
        }
        break;
    }
    case OpCopyObject: case OpUConvert: case OpSConvert: case OpFConvert: case OpBitcast:
        for (int c = 0; c < n; c++) {
            int k = cg_vk(g, (int)in[3], c);
            cg_def(g, id, c, k, cg_ref(g, (int)in[3], c, k));
        }
        break;
    case OpCompositeConstruct: {
        int c = 0;
        for (int j = 3; j < wc && c < n; j++) {
            int m = s->ids[in[j]].comps;
            for (int i = 0; i < m && c < n; i++, c++) {
                int k = cg_vk(g, (int)in[j], i);
                cg_def(g, id, c, k, cg_ref(g, (int)in[j], i, k));
            }
        }
        break;
    }
    case OpCompositeExtract: case OpCompositeInsert: {
        int base = op == OpCompositeExtract ? (int)in[3] : (int)in[4];
        int t = s->ids[base].type, off = 0, ct = t;
        for (int j = op == OpCompositeExtract ? 4 : 5; j < wc; j++) {
            int o = sv_child(s, ct, (int)in[j], &ct);
            if (o < 0) break;
            off += o;
        }
        if (op == OpCompositeExtract) {
            for (int c = 0; c < n; c++) {
                int k = cg_vk(g, base, off + c);
                cg_def(g, id, c, k, cg_ref(g, base, off + c, k));
            }
        } else {
            int m = s->ids[in[3]].comps;
            for (int c = 0; c < n; c++) {
                int src = c >= off && c < off + m ? (int)in[3] : base, sc = src == base ? c : c - off;
                int k   = cg_vk(g, src, sc);
                cg_def(g, id, c, k, cg_ref(g, src, sc, k));
            }
        }
        break;
    }
    case OpVectorShuffle: {
        int n1 = s->ids[in[3]].comps;
        for (int j = 0; j < n; j++) {
            uint32_t sel = in[5 + j];
            if (sel == 0xffffffffu) {
                cg_def(g, id, j, K_U, "0u");
                continue;
            }
            int src = sel < (uint32_t)n1 ? (int)in[3] : (int)in[4], sc = sel < (uint32_t)n1 ? (int)sel : (int)sel - n1;
            int k   = cg_vk(g, src, sc);
            cg_def(g, id, j, k, cg_ref(g, src, sc, k));
        }
        break;
    }
    case OpVectorExtractDynamic: {
        int         m  = s->ids[in[3]].comps, k = cg_vk(g, (int)in[3], 0);
        const char* ix = cg_tmp(g, K_I, cg_ref(g, (int)in[4], 0, K_I));
        const char* i  = cg_tmp(g, K_I, cg_str(g, "%s < 0 ? 0 : (%s >= %d ? %d : %s)", ix, ix, m, m - 1, ix));
        const char* e  = cg_ref(g, (int)in[3], m - 1, k);
        for (int j = m - 2; j >= 0; j--) e = cg_str(g, "%s == %d ? %s : (%s)", i, j, cg_ref(g, (int)in[3], j, k), e);
        cg_def(g, id, 0, k, e);
        break;
    }
    case OpVectorInsertDynamic: {
        const char* ix = cg_tmp(g, K_I, cg_ref(g, (int)in[5], 0, K_I));
        const char* i  = cg_tmp(g, K_I, cg_str(g, "%s < 0 ? 0 : (%s >= %d ? %d : %s)", ix, ix, n, n - 1, ix));
        for (int j = 0; j < n; j++) {
            int k = cg_vk(g, id, j);
            cg_def(g, id, j, k, cg_str(g, "%s == %d ? %s : %s", i, j, cg_ref(g, (int)in[4], 0, k), cg_ref(g, (int)in[3], j, k)));
        }
        break;
    }
    case OpTranspose: {
        const sv_id* T    = &s->ids[s->ids[in[3]].type];
        int          cols = T->count, rows = s->ids[T->elem].comps;
        for (int cc = 0; cc < cols; cc++)
            for (int rr = 0; rr < rows; rr++) cg_def(g, id, rr * cols + cc, K_F, cg_ref(g, (int)in[3], cc * rows + rr, K_F));
        break;
    }
    case OpConvertFToS:
        for (int c = 0; c < n; c++) {
            const char* v = cg_tmp(g, K_F, cg_ref(g, (int)in[3], c, K_F));
            cg_def(g, id, c, K_I, cg_str(g, "%s >= 2147483520.0f ? INT32_MAX : (%s <= -2147483648.0f ? INT32_MIN : (%s == %s ? (int32_t)%s : 0))", v, v, v, v, v));
        }
        break;
    case OpConvertFToU:
        for (int c = 0; c < n; c++) {
            const char* v = cg_tmp(g, K_F, cg_ref(g, (int)in[3], c, K_F));
            cg_def(g, id, c, K_U, cg_str(g, "%s >= 4294967040.0f ? UINT32_MAX : (%s > 0 ? (uint32_t)%s : 0u)", v, v, v));
        }
        break;
    case OpConvertSToF: for (int c = 0; c < n; c++) cg_def(g, id, c, K_F, cg_str(g, "(float)%s", cg_ref(g, (int)in[3], c, K_I))); break;
    case OpConvertUToF: for (int c = 0; c < n; c++) cg_def(g, id, c, K_F, cg_str(g, "(float)%s", cg_ref(g, (int)in[3], c, K_U))); break;
    case OpFNegate: for (int c = 0; c < n; c++) cg_def(g, id, c, K_F, cg_str(g, "-%s", cg_ref(g, (int)in[3], c, K_F))); break;
    case OpSNegate: for (int c = 0; c < n; c++) cg_def(g, id, c, K_U, cg_str(g, "0u - %s", cg_ref(g, (int)in[3], c, K_U))); break;
    case OpNot: for (int c = 0; c < n; c++) cg_def(g, id, c, K_U, cg_str(g, "~%s", cg_ref(g, (int)in[3], c, K_U))); break;
    case OpLogicalNot:
        for (int c = 0; c < n; c++) cg_def(g, id, c, K_U, cg_str(g, "(uint32_t)(%s == 0u)", cg_ref(g, (int)in[3], c, K_U)));
        break;
    case OpIsNan:
        for (int c = 0; c < n; c++) {
            const char* v = cg_tmp(g, K_F, cg_ref(g, (int)in[3], c, K_F));
            cg_def(g, id, c, K_U, cg_str(g, "(uint32_t)(%s != %s)", v, v));
        }
        break;
    case OpIsInf:
        for (int c = 0; c < n; c++) cg_def(g, id, c, K_U, cg_str(g, "isinf(%s) ? 1u : 0u", cg_ref(g, (int)in[3], c, K_F)));
        break;
    case OpAny: case OpAll: {
        int         m = s->ids[in[3]].comps;
        const char* e = cg_str(g, "%s != 0u", cg_ref(g, (int)in[3], 0, K_U));
        for (int j = 1; j < m; j++) e = cg_str(g, "%s %s %s != 0u", e, op == OpAll ? "&&" : "||", cg_ref(g, (int)in[3], j, K_U));
        cg_def(g, id, 0, K_U, cg_str(g, "(%s) ? 1u : 0u", e));
        break;
    }
    case OpSelect: {
        int cc = s->ids[in[3]].comps;
        for (int j = 0; j < n; j++) {
            int k = cg_vk(g, id, j);
            cg_def(g, id, j, k,
                   cg_str(g, "%s ? %s : %s", cg_ref(g, (int)in[3], cc == 1 ? 0 : j, K_U), cg_ref(g, (int)in[4], j, k), cg_ref(g, (int)in[5], j, k)));
        }
        break;
    }
    case OpVectorTimesScalar: case OpMatrixTimesScalar: {
        const char* b = cg_ref(g, (int)in[4], 0, K_F);
        for (int j = 0; j < n; j++) cg_def(g, id, j, K_F, cg_str(g, "%s * %s", cg_ref(g, (int)in[3], j, K_F), b));
        break;
    }
    case OpDot: {
        int         m = s->ids[in[3]].comps;
        const char* d = cg_str(g, "%s * %s", cg_ref(g, (int)in[3], 0, K_F), cg_ref(g, (int)in[4], 0, K_F));
        for (int j = 1; j < m; j++) d = cg_tmp(g, K_F, cg_str(g, "%s + %s * %s", d, cg_ref(g, (int)in[3], j, K_F), cg_ref(g, (int)in[4], j, K_F)));
        cg_def(g, id, 0, K_F, d);
        break;
    }
    case OpMatrixTimesVector: { /* r[row] = sum_c M[c][row] * v[c] */
        const sv_id* T    = &s->ids[s->ids[in[3]].type];
        int          cols = T->count, rows = s->ids[T->elem].comps;
        for (int rr = 0; rr < rows; rr++) {
            const char* d = cg_str(g, "%s * %s", cg_ref(g, (int)in[3], rr, K_F), cg_ref(g, (int)in[4], 0, K_F));
            for (int cc = 1; cc < cols; cc++)
                d = cg_tmp(g, K_F, cg_str(g, "%s + %s * %s", d, cg_ref(g, (int)in[3], cc * rows + rr, K_F), cg_ref(g, (int)in[4], cc, K_F)));
            cg_def(g, id, rr, K_F, d);
        }
        break;
    }
    case OpVectorTimesMatrix: { /* r[c] = sum_row v[row] * M[c][row] */
        const sv_id* T    = &s->ids[s->ids[in[4]].type];
        int          cols = T->count, rows = s->ids[T->elem].comps;
        for (int cc = 0; cc < cols; cc++) {
            const char* d = cg_str(g, "%s * %s", cg_ref(g, (int)in[3], 0, K_F), cg_ref(g, (int)in[4], cc * rows, K_F));
            for (int rr = 1; rr < rows; rr++)
                d = cg_tmp(g, K_F, cg_str(g, "%s + %s * %s", d, cg_ref(g, (int)in[3], rr, K_F), cg_ref(g, (int)in[4], cc * rows + rr, K_F)));
            cg_def(g, id, cc, K_F, d);
        }
        break;
    }
    case OpMatrixTimesMatrix: {
        const sv_id* TA   = &s->ids[s->ids[in[3]].type];
        const sv_id* TB   = &s->ids[s->ids[in[4]].type];
        int          rows = s->ids[TA->elem].comps, inner = TA->count, cols = TB->count;
        for (int cc = 0; cc < cols; cc++)
            for (int rr = 0; rr < rows; rr++) {
                const char* d = cg_str(g, "%s * %s", cg_ref(g, (int)in[3], rr, K_F), cg_ref(g, (int)in[4], cc * inner, K_F));
                for (int j = 1; j < inner; j++)
                    d = cg_tmp(g, K_F, cg_str(g, "%s + %s * %s", d, cg_ref(g, (int)in[3], j * rows + rr, K_F), cg_ref(g, (int)in[4], cc * inner + j, K_F)));
                cg_def(g, id, cc * rows + rr, K_F, d);
            }
        break;
    }
    case OpOuterProduct: {
        int rows = s->ids[in[3]].comps, cols = s->ids[in[4]].comps;
        for (int cc = 0; cc < cols; cc++)
            for (int rr = 0; rr < rows; rr++)
                cg_def(g, id, cc * rows + rr, K_F, cg_str(g, "%s * %s", cg_ref(g, (int)in[3], rr, K_F), cg_ref(g, (int)in[4], cc, K_F)));
        break;
    }
    case OpExtInst: cg_ext(g, in); break;
    default: cg_fail(g, "opcode %d is not supported by the C backend", op); break;
    }
}

/* stores under the block's lane mask (lm) */
static void cg_store(cg* g, const uint32_t* in)
{
    const sv_stage* s = g->s;
    const sv_id*    P = &s->ids[in[1]];
    const sv_id*    V = &s->ids[P->pvar];
    if (V->storage != SC_Function && V->storage != SC_Private && V->storage != SC_Input && V->storage != SC_Output) return;
    int         comps = s->ids[in[2]].comps, ak = cg_ak(s, V->type);
    const char* arr   = cg_str(g, "x%d", P->pvar);
    if (!P->ndyn) {
        for (int c = 0; c < comps; c++) {
            int         k  = cg_ck(s, V->type, P->poff + c);
            const char* ix = cg_str(g, "%d + l", (P->poff + c) * SV_L);
            const char* lv = cg_lv(g, arr, ak, ix, k);
            cg_line(g, "%s = lm[l] ? %s : %s;", lv, cg_ref(g, (int)in[2], c, k), lv);
        }
        return;
    }
    const char* o = cg_varoff(g, P, V, comps);
    for (int c = 0; c < comps; c++) {
        int k = cg_ck(s, V->type, P->poff + c); /* dynamic: same component kinds along the array */
        cg_line(g, "if (lm[l]) %s = %s;", cg_lv(g, arr, ak, cg_str(g, "(%s + %d) * 64 + l", o, c), k), cg_ref(g, (int)in[2], c, k));
    }
}

/* a float array with the components of value id (lane l = c * 64 + l):
 * the value's array when it is a float array, else a filled temporary */
static const char* cg_farr(cg* g, int id, int comps)
{
    const sv_stage* s = g->s;
    if (s->vcls[id] == VC_MAT && cg_ak(s, s->ids[id].type) == K_F) return cg_str(g, "r%d", id);
    int t = g->tmp++;
    cg_line(g, "float t%d[%d];", t, comps * SV_L);
    cg_line(g, "for (int l = 0; l < 64; l++) {");
    g->ind++;
    for (int c = 0; c < comps; c++) cg_line(g, "t%d[%d + l] = %s;", t, c * SV_L, cg_ref(g, id, c, K_F));
    g->ind--;
    cg_line(g, "}");
    return cg_str(g, "t%d", t);
}

/* derivatives and texture samples: whole arrays */
static void cg_cross(cg* g, const uint32_t* in, int op, int n)
{
    const sv_stage* s  = g->s;
    int             id = (int)in[2];
    char            res[32];
    snprintf(res, sizeof(res), "r%d", id);
    if (cg_ak(s, s->ids[id].type) != K_F) {
        cg_fail(g, "internal: cross lane result %d is not float", id);
        return;
    }
    if (op == OpImageSampleImplicitLod || op == OpImageSampleExplicitLod) {
        int         unit = s->ids[in[3]].unit, wc = (int)WC(in);
        const char* uv   = cg_farr(g, (int)in[4], 2);
        char        u[64], v[64];
        snprintf(u, sizeof(u), "%s", uv);
        snprintf(v, sizeof(v), "%s + 64", uv);
        const char* lod = NULL;
        if (op == OpImageSampleExplicitLod && wc >= 7 && (in[5] & 2u)) lod = cg_farr(g, (int)in[6], 1);
        if (op == OpImageSampleImplicitLod && g->fs) cg_line(g, "spv_tex_batch(io, %d, %s, %s, %s);", unit, u, v, res);
        else cg_line(g, "spv_tex_lod(io->textures[%d], &io->samplers[%d], %s, %s, %s, %s);", unit, unit, u, v, lod ? lod : "NULL", res);
        return;
    }
    int dx = op == OpDPdx || op == OpDPdxFine || op == OpDPdxCoarse;
    int dy = op == OpDPdy || op == OpDPdyFine || op == OpDPdyCoarse;
    if (!g->fs) { /* vertex stage: no neighbours */
        cg_line(g, "memset(%s, 0, sizeof(%s));", res, res);
        return;
    }
    const char* a = cg_farr(g, (int)in[3], n);
    char        src[64];
    snprintf(src, sizeof(src), "%s", a);
    for (int j = 0; j < n; j++) {
        cg_line(g, "for (int l = 0; l < 64; l++) {");
        g->ind++;
        cg_line(g, "const float ddx = %s[%d + (l | 1)] - %s[%d + (l & ~1)], ddy = %s[%d + (l & 31) + 32] - %s[%d + (l & 31)];", src,
                j * SV_L, src, j * SV_L, src, j * SV_L, src, j * SV_L);
        cg_line(g, "%s[%d + l] = %s;", res, j * SV_L, dx ? "ddx" : (dy ? "ddy" : "fabsf(ddx) + fabsf(ddy)"));
        g->ind--;
        cg_line(g, "}");
    }
}

static int cg_has(const sv_block* B, int op)
{
    for (int ii = 0; ii < B->ninst; ii++)
        if (B->insts[ii].op == op) return 1;
    return 0;
}

static void cg_decl_mat(cg* g, int id)
{
    const sv_stage* s  = g->s;
    int             ak = cg_ak(s, s->ids[id].type);
    cg_line(g, "%s r%d[%d];", cg_tn(ak), id, s->ids[id].comps * SV_L);
}

/* one execution of a block under mask slot `m` */
static void cg_body(cg* g, int bi, int m)
{
    const sv_stage* s = g->s;
    const sv_block* B = &s->blocks[bi];
    cg_line(g, "{ /* block %%%d */", B->label);
    g->ind++;
    for (uint32_t v = 0; v < s->bound; v++) /* arrays only used inside this block */
        if (s->vcls[v] == VC_MAT && s->vscope[v] == VS_BLOCK && s->vblock[v] == bi) cg_decl_mat(g, (int)v);
    if (cg_has(B, OpStore)) {
        cg_line(g, "uint32_t lm[64];");
        cg_line(g, "spv_expand(m%d, lm);", m);
    }
    if (B->nphi) { /* phis: the lanes of each incoming edge; staged, then committed */
        const int* pm = B->phim;
        int        done[256], nd = 0;
        for (int ii = 0; ii < B->nphi; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            for (int k = 3; k + 1 < (int)WC(in); k += 2, pm++) {
                int seen = *pm == SV_M_ZERO;
                for (int j = 0; j < nd; j++) seen |= done[j] == *pm;
                if (seen) continue;
                if (nd < 256) done[nd++] = *pm;
                cg_line(g, "uint32_t le%d[64];", *pm);
                cg_line(g, "spv_expand(m%d & m%d, le%d);", *pm, m, *pm);
            }
        }
        g->seg = -2; /* every per lane value from its array */
        cg_line(g, "for (int l = 0; l < 64; l++) {");
        g->ind++;
        pm = B->phim;
        for (int ii = 0; ii < B->nphi; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             id = (int)in[2], n = s->ids[id].comps;
            const int*      p0 = pm;
            for (int c = 0; c < n; c++) {
                int k = cg_vk(g, id, c);
                cg_line(g, "%s q%d_%d = %s;", cg_tn(k), id, c, cg_ref(g, id, c, k));
                pm = p0;
                for (int j = 3; j + 1 < (int)WC(in); j += 2, pm++)
                    if (*pm != SV_M_ZERO) cg_line(g, "q%d_%d = le%d[l] ? %s : q%d_%d;", id, c, *pm, cg_ref(g, (int)in[j], c, k), id, c);
            }
            if (!n) pm = p0 + ((int)WC(in) - 3) / 2;
        }
        for (int ii = 0; ii < B->nphi; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             id = (int)in[2];
            for (int c = 0; c < s->ids[id].comps; c++) {
                int k = cg_vk(g, id, c);
                cg_line(g, "%s = q%d_%d;", cg_lv(g, cg_str(g, "r%d", id), cg_ak(s, s->ids[id].type), cg_str(g, "%d + l", c * SV_L), k), id, c);
            }
        }
        g->ind--;
        cg_line(g, "}");
    }
    for (int i = 0; i < B->nclr; i++) cg_line(g, "m%d = 0;", B->clr[i]); /* edges consumed */
    /* segments: lane uniform work first, then one loop over the lanes */
    for (int ii = B->nphi; ii < B->ninst && !g->failed;) {
        int seg = B->seg[ii], end = ii;
        while (end < B->ninst && B->seg[end] == seg) end++;
        g->seg = seg;
        if (sv_is_cross_lane(B->insts[ii].op)) {
            cg_cross(g, s->w + B->insts[ii].at, B->insts[ii].op, B->insts[ii].n);
            ii = end;
            continue;
        }
        int lanes = 0;
        for (int j = ii; j < end; j++) {
            const uint32_t* in = s->w + B->insts[j].at;
            if (B->insts[j].op != OpStore && s->vcls[in[2]] == VC_UNIFORM) cg_inst(g, in, B->insts[j].op, B->insts[j].n);
            else lanes = 1;
        }
        if (lanes) {
            cg_line(g, "for (int l = 0; l < 64; l++) {");
            g->ind++;
            for (int j = ii; j < end; j++) {
                const uint32_t* in = s->w + B->insts[j].at;
                if (B->insts[j].op == OpStore) cg_store(g, in);
                else if (s->vcls[in[2]] != VC_UNIFORM) cg_inst(g, in, B->insts[j].op, B->insts[j].n);
            }
            g->ind--;
            cg_line(g, "}");
        }
        ii = end;
    }
    g->seg = -1;
    g->ind--;
    cg_line(g, "}");
}

/* the lowered program */
static void cg_program(cg* g)
{
    const sv_stage* s = g->s;
    for (int i = 0; i < s->nir && !g->failed; i++) {
        const sv_ir* o = &s->ir[i];
        switch (o->op) {
        case IR_ZERO: cg_line(g, "m%d = 0;", o->a); break;
        case IR_COPY: cg_line(g, "m%d = m%d;", o->a, o->b); break;
        case IR_OR: cg_line(g, "m%d = m%d | m%d;", o->a, o->b, o->c); break;
        case IR_ANDN: cg_line(g, "m%d = m%d & ~m%d;", o->a, o->b, o->c); break;
        case IR_COND: case IR_EQ: {
            int         v  = o->b, ak = cg_ak(s, s->ids[v].type);
            const char* eq = o->op == IR_EQ ? cg_str(g, "== %uu", o->lit) : "!= 0u";
            if (s->vcls[v] == VC_CONST || s->vcls[v] == VC_UNIFORM) {
                cg_line(g, "m%d = (%s %s) ? m%d : 0;", o->a, cg_ref(g, v, 0, K_U), eq, o->c);
            } else if (o->op == IR_COND && ak == K_U) {
                cg_line(g, "m%d = m%d ? spv_bits(r%d) & m%d : 0;", o->a, o->c, v, o->c);
            } else {
                cg_line(g, "{");
                g->ind++;
                cg_line(g, "uint64_t e = 0;");
                cg_line(g, "for (int l = 0; l < 64; l++) e |= (uint64_t)(%s %s) << l;", cg_ref(g, v, 0, K_U), eq);
                cg_line(g, "m%d = e & m%d;", o->a, o->c);
                g->ind--;
                cg_line(g, "}");
            }
            break;
        }
        case IR_BODY: cg_body(g, o->a, o->b); break;
        case IR_IF: cg_line(g, "if (m%d) {", o->a), g->ind++; break;
        case IR_ENDIF: case IR_ENDLOOP: g->ind--, cg_line(g, "}"); break;
        case IR_LOOP: cg_line(g, "for (m%d = 0;;) {", o->a), g->ind++; break;
        case IR_BREAKZ: cg_line(g, "if (!m%d || m%d++ >= %uu) break;", o->a, o->b, SV_MAXITER); break;
        case IR_BREAK: cg_line(g, "break;"); break;
        default: break;
        }
    }
}

/* declarations, variables and the lowered program of one batch */
static void cg_batch_head(cg* g, size_t* stack)
{
    const sv_stage* s = g->s;
    size_t          bytes = 0;
    for (int i = 0; i < s->nmask; i += 16) {
        for (int j = 0; j < g->ind; j++) cg_put(g, "    ");
        cg_put(g, "uint64_t");
        for (int j = i; j < i + 16 && j < s->nmask; j++) cg_put(g, "%s m%d = 0", j > i ? "," : "", j);
        cg_put(g, ";\n");
    }
    for (uint32_t v = 0; v < s->bound; v++) {
        if (s->vcls[v] == VC_UNIFORM)
            for (int c = 0; c < s->ids[v].comps; c++) cg_line(g, "%s u%u_%d = 0;", cg_tn(cg_vk(g, (int)v, c)), v, c);
        if (s->vcls[v] == VC_MAT && s->vscope[v] == VS_FUNC) {
            cg_decl_mat(g, (int)v);
            bytes += (size_t)s->ids[v].comps * SV_L * 4;
            if (s->ids[s->bix[0] >= 0 ? 0 : 0].cls == C_NONE) {}
        }
    }
    for (int bi = 0; bi < s->nblocks; bi++) /* phis keep their old value in lanes without an edge */
        for (int ii = 0; ii < s->blocks[bi].nphi; ii++) {
            int id = (int)s->w[s->blocks[bi].insts[ii].at + 2];
            cg_line(g, "memset(r%d, 0, sizeof(r%d));", id, id);
        }
    for (int i = 0; i < s->nvars; i++) { /* variables: zero or their initializer */
        const sv_id* v  = &s->ids[s->vars[i]];
        int          ak = cg_ak(s, v->type);
        cg_line(g, "%s x%d[%d];", cg_tn(ak), s->vars[i], v->comps * SV_L);
        bytes += (size_t)v->comps * SV_L * 4;
        if (v->init && s->ids[v->init].cls == C_CONST) {
            for (int c = 0; c < v->comps; c++) {
                int k = cg_ck(s, v->type, c);
                cg_line(g, "for (int l = 0; l < 64; l++) %s = %s;", cg_lv(g, cg_str(g, "x%d", s->vars[i]), ak, cg_str(g, "%d + l", c * SV_L), k),
                        cg_lit(g, sv_cblock(s, &s->ids[v->init])[(size_t)c * SV_L], k));
            }
        } else {
            cg_line(g, "memset(x%d, 0, sizeof(x%d));", s->vars[i], s->vars[i]);
        }
    }
    *stack = bytes;
}

static void cg_stage_fs(cg* g, const char* name)
{
    const sv_stage* s = g->s;
    size_t          stack = 0;
    cg_put(g, "static void %s_fs(const fm3d_fs_io* io)\n{\n", name);
    g->ind = 1;
    cg_line(g, "const unsigned char* ubo   = (const unsigned char*)io->uniforms;");
    cg_line(g, "const size_t         ubo_n = io->uniforms ? io->uniform_size : 0;");
    cg_batch_head(g, &stack);
    for (int i = 0; i < s->nin; i++) { /* inputs */
        const sv_io* fi = &s->in[i];
        const sv_id* v  = &s->ids[fi->var];
        int          ak = cg_ak(s, v->type);
        const char*  x  = cg_str(g, "x%d", fi->var);
        if (fi->builtin == BI_FragCoord) {
            cg_line(g, "for (int l = 0; l < 64; l++) {");
            cg_line(g, "    %s = (float)(io->x + (l & 31)) + 0.5f;", cg_lv(g, x, ak, "l", K_F));
            cg_line(g, "    %s = (float)(io->y + (l >> 5)) + 0.5f;", cg_lv(g, x, ak, "64 + l", K_F));
            cg_line(g, "    %s = io->z ? io->z[l] : 0.0f;", cg_lv(g, x, ak, "128 + l", K_F));
            cg_line(g, "    %s = 1.0f;", cg_lv(g, x, ak, "192 + l", K_F));
            cg_line(g, "}");
            continue;
        }
        if (fi->builtin == BI_FrontFacing) {
            cg_line(g, "for (int l = 0; l < 64; l++) %s = 1u;", cg_lv(g, x, ak, "l", K_U));
            continue;
        }
        int slot = g->P->fslot[i];
        for (int c = 0; c < fi->comps; c++) {
            int k = cg_ck(s, v->type, c);
            if (slot < 0 || slot + c >= FM3D_MAX_SHADER_VARYINGS) continue; /* stays 0 */
            cg_line(g, "for (int l = 0; l < 64; l++) %s = %s;", cg_lv(g, x, ak, cg_str(g, "%d + l", c * SV_L), k),
                    cg_conv(g, K_F, k, cg_str(g, "io->varyings[%d][l]", slot + c)));
        }
    }
    cg_line(g, "for (int l = 0; l < 64; l++) /* every pixel of the batch's quads (helpers included) */");
    cg_line(g, "    if ((l & 31) < io->cols) m%d |= 1ull << l;", SV_M_ENTRY);
    cg_program(g);
    for (int i = 0; i < s->nout; i++) {
        const sv_io* fo = &s->out[i];
        if (fo->loc != 0) continue;
        int ak = cg_ak(s, s->ids[fo->var].type);
        for (int c = 0; c < 4; c++) {
            if (c < fo->comps)
                cg_line(g, "for (int l = 0; l < 64; l++) io->out[%d][l] = %s;", c,
                        cg_at(g, cg_str(g, "x%d", fo->var), ak, cg_str(g, "%d + l", c * SV_L), K_F));
            else
                cg_line(g, "for (int l = 0; l < 64; l++) io->out[%d][l] = %s;", c, c == 3 ? "1.0f" : "0.0f");
        }
    }
    if (s->kills) {}
    cg_line(g, "for (int l = 0; l < 64; l++)");
    cg_line(g, "    if ((m%d >> l) & 1) io->mask[l] = 0;", SV_M_KILLED);
    g->ind = 0;
    cg_put(g, "}\n/* %s_fs: %zu KB of arrays on the stack */\n\n", name, (stack + 1023) / 1024);
}

static void cg_stage_vs(cg* g, const char* name)
{
    const sv_stage*   s = g->s;
    const fm3d_spirv* P = g->P;
    size_t            stack = 0;
    cg_put(g, "static void %s_vs(const fm3d_vs_io* io)\n{\n", name);
    g->ind = 1;
    cg_line(g, "const unsigned char* ubo   = (const unsigned char*)io->uniforms;");
    cg_line(g, "const size_t         ubo_n = io->uniforms ? io->uniform_size : 0;");
    cg_line(g, "for (int base = 0; base < io->count; base += 64) {");
    g->ind = 2;
    cg_line(g, "const int n = io->count - base < 64 ? io->count - base : 64;");
    cg_batch_head(g, &stack);
    for (int i = 0; i < s->nin; i++) { /* vertex attributes */
        const sv_io*              vi = &s->in[i];
        const sv_id*              v  = &s->ids[vi->var];
        const fm3d_vertex_attrib* a  = NULL;
        for (int k = 0; k < P->nattr; k++)
            if (P->attr[k].location == vi->loc) a = &P->attr[k];
        int ak = cg_ak(s, v->type);
        cg_line(g, "for (int l = 0; l < n; l++) {");
        g->ind++;
        cg_line(g, "const char* vp = (const char*)io->vertices + (size_t)(base + l) * (size_t)io->stride;");
        for (int c = 0; c < vi->comps; c++) {
            int k = cg_ck(s, v->type, c);
            if (a && c < a->components) {
                cg_line(g, "{");
                cg_line(g, "    float f;");
                cg_line(g, "    memcpy(&f, vp + %d, 4);", a->offset + 4 * c);
                cg_line(g, "    %s = %s;", cg_lv(g, cg_str(g, "x%d", vi->var), ak, cg_str(g, "%d + l", c * SV_L), k), cg_conv(g, K_F, k, "f"));
                cg_line(g, "}");
            } else {
                cg_line(g, "%s = %s;", cg_lv(g, cg_str(g, "x%d", vi->var), ak, cg_str(g, "%d + l", c * SV_L), k),
                        cg_conv(g, K_F, k, c == 3 ? "1.0f" : "0.0f"));
            }
        }
        g->ind--;
        cg_line(g, "}");
    }
    cg_line(g, "m%d = n == 64 ? ~0ull : ((1ull << n) - 1);", SV_M_ENTRY);
    cg_program(g);
    { /* outputs */
        int ak = cg_ak(s, s->ids[s->pos_var].type);
        cg_line(g, "for (int l = 0; l < n; l++) {");
        g->ind++;
        cg_line(g, "float* o = io->pos + (size_t)(base + l) * (size_t)io->out_stride;");
        cg_line(g, "float* q = io->varyings + (size_t)(base + l) * (size_t)io->out_stride;");
        cg_line(g, "(void)q;");
        for (int c = 0; c < 4; c++)
            cg_line(g, "o[%d] = %s;", c, cg_at(g, cg_str(g, "x%d", s->pos_var), ak, cg_str(g, "%d + l", (s->pos_off + c) * SV_L), K_F));
        for (int i = 0; i < s->nout; i++) {
            const sv_io* vo = &s->out[i];
            int          slot = P->vslot[i];
            if (vo->builtin >= 0 || slot < 0) continue;
            int vak = cg_ak(s, s->ids[vo->var].type);
            for (int c = 0; c < vo->comps && slot + c < FM3D_MAX_SHADER_VARYINGS; c++)
                cg_line(g, "q[%d] = %s;", slot + c, cg_at(g, cg_str(g, "x%d", vo->var), vak, cg_str(g, "%d + l", c * SV_L), K_F));
        }
        g->ind--;
        cg_line(g, "}");
    }
    g->ind = 1;
    cg_line(g, "}");
    g->ind = 0;
    cg_put(g, "}\n/* %s_vs: %zu KB of arrays on the stack */\n\n", name, (stack + 1023) / 1024);
}

static const char cg_helpers[] =
    "#ifndef FM_SPIRV_C_HELPERS\n"
    "#define FM_SPIRV_C_HELPERS\n"
    "#if defined(__GNUC__)\n"
    "#  pragma GCC diagnostic ignored \"-Wunused-variable\"\n"
    "#  pragma GCC diagnostic ignored \"-Wunused-but-set-variable\"\n"
    "#  pragma GCC diagnostic ignored \"-Wunused-function\"\n"
    "#  pragma GCC diagnostic ignored \"-Wunused-parameter\"\n"
    "#elif defined(_MSC_VER)\n"
    "#  pragma warning(disable : 4100 4189 4101 4244 4702)\n"
    "#endif\n"
    "#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)\n"
    "#  include <emmintrin.h>\n"
    "#  define SPV_SSE2 1\n"
    "#elif defined(__aarch64__) || defined(_M_ARM64)\n"
    "#  include <arm_neon.h>\n"
    "#  define SPV_NEON 1\n"
    "#endif\n"
    "typedef union spv_w {\n"
    "    float    f;\n"
    "    uint32_t u;\n"
    "    int32_t  i;\n"
    "} spv_w;\n"
    "static float    spv_fu(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }\n"
    "static uint32_t spv_uf(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }\n"
    "/* a 32 bit word of the uniform block (0 outside it) */\n"
    "static uint32_t spv_u32(const unsigned char* b, size_t n, long off)\n"
    "{\n"
    "    uint32_t v = 0;\n"
    "    if (off >= 0 && (size_t)off + 4 <= n) memcpy(&v, b + off, 4);\n"
    "    return v;\n"
    "}\n"
    "/* all ones / zero words of a lane mask */\n"
    "static void spv_expand(uint64_t m, uint32_t* lm)\n"
    "{\n"
    "    static const uint32_t nib[16][4] = { { 0, 0, 0, 0 }, { ~0u, 0, 0, 0 }, { 0, ~0u, 0, 0 }, { ~0u, ~0u, 0, 0 },\n"
    "        { 0, 0, ~0u, 0 }, { ~0u, 0, ~0u, 0 }, { 0, ~0u, ~0u, 0 }, { ~0u, ~0u, ~0u, 0 }, { 0, 0, 0, ~0u },\n"
    "        { ~0u, 0, 0, ~0u }, { 0, ~0u, 0, ~0u }, { ~0u, ~0u, 0, ~0u }, { 0, 0, ~0u, ~0u }, { ~0u, 0, ~0u, ~0u },\n"
    "        { 0, ~0u, ~0u, ~0u }, { ~0u, ~0u, ~0u, ~0u } };\n"
    "    for (int i = 0; i < 64; i += 4) memcpy(lm + i, nib[(m >> i) & 15], 16);\n"
    "}\n"
    "/* lanes with a nonzero word */\n"
    "static uint64_t spv_bits(const uint32_t* c)\n"
    "{\n"
    "    uint64_t m = 0;\n"
    "#if SPV_SSE2\n"
    "    const __m128i z = _mm_setzero_si128();\n"
    "    for (int i = 0; i < 64; i += 4) {\n"
    "        __m128i v = _mm_loadu_si128((const __m128i*)(c + i));\n"
    "        m |= (uint64_t)(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v, z))) ^ 15) << i;\n"
    "    }\n"
    "#elif SPV_NEON\n"
    "    static const int32_t sh[4] = { 0, 1, 2, 3 };\n"
    "    const int32x4_t      vs    = vld1q_s32(sh);\n"
    "    for (int i = 0; i < 64; i += 4) {\n"
    "        uint32x4_t nz = vtstq_u32(vld1q_u32(c + i), vld1q_u32(c + i));\n"
    "        m |= (uint64_t)vaddvq_u32(vshlq_u32(vandq_u32(nz, vdupq_n_u32(1)), vs)) << i;\n"
    "    }\n"
    "#else\n"
    "    for (int l = 0; l < 64; l++) m |= (uint64_t)(c[l] != 0) << l;\n"
    "#endif\n"
    "    return m;\n"
    "}\n"
    "/* texture(): mipmapped batch sampling, (0, 0, 0, 1) without a texture */\n"
    "static void spv_tex_batch(const fm3d_fs_io* io, int unit, const float* u, const float* v, float* out)\n"
    "{\n"
    "    const fm3d_texture* t = io->textures[unit];\n"
    "    if (!t) {\n"
    "        for (int k = 0; k < 256; k++) out[k] = k >= 192 ? 1.0f : 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    fm3d_sample_batch(io, t, &io->samplers[unit], u, v, out, out + 64, out + 128, out + 192);\n"
    "}\n"
    "/* textureLod() (and texture() in a vertex shader: level 0) */\n"
    "static void spv_tex_lod(const fm3d_texture* t, const fm3d_sampler* s, const float* u, const float* v, const float* lod,\n"
    "                        float* out)\n"
    "{\n"
    "    if (!t) {\n"
    "        for (int k = 0; k < 256; k++) out[k] = k >= 192 ? 1.0f : 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    fm3d_sample_lod(t, s, u, v, lod, 64, out, out + 64, out + 128, out + 192);\n"
    "}\n"
    "#endif\n\n";

static int cg_ident(const char* n)
{
    if (!n || !*n || (!(n[0] == '_' || (n[0] >= 'a' && n[0] <= 'z') || (n[0] >= 'A' && n[0] <= 'Z')))) return 0;
    for (const char* p = n; *p; p++)
        if (!(*p == '_' || (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9'))) return 0;
    return strlen(n) < 64;
}

char* fm3d_spirv_to_c(const fm3d_spirv* P, const char* name, char* err, size_t errn)
{
    if (err && errn) err[0] = 0;
    if (!P) return NULL;
    if (!cg_ident(name)) {
        if (err && errn) snprintf(err, errn, "the program name must be a C identifier");
        return NULL;
    }
    cg* g = (cg*)calloc(1, sizeof(cg));
    if (!g) return NULL;
    g->P    = P;
    g->err  = err;
    g->errn = errn;
    g->seg  = -1;
    cg_put(g, "/* generated by fatmap (fm3d_spirv_to_c): do not edit.\n"
              " * Needs fatmap with -Dshaders. Compile optimized with vectorization\n"
              " * (-O3, MSVC /O2) and without floating point contraction or fast math\n"
              " * (-ffp-contract=off, MSVC /fp:precise) to render exactly what fatmap's\n"
              " * SPIR-V interpreter renders. */\n"
              "#include <fatmap/fatmap.h>\n#include <math.h>\n#include <stdint.h>\n#include <string.h>\n"
              "#if !FM_FEATURE_SHADERS\n#  error \"fatmap was built without programmable stages (-Dshaders)\"\n#endif\n\n");
    cg_put(g, "%s", cg_helpers);
    if (P->vs) {
        g->s  = P->vs;
        g->fs = 0;
        cg_stage_vs(g, name);
    }
    if (P->fs && !g->failed) {
        g->s  = P->fs;
        g->fs = 1;
        cg_stage_fs(g, name);
    }
    int discards = 0;
    if (P->fs)
        for (int bi = 0; bi < P->fs->nblocks; bi++) {
            int op = (int)OP(P->fs->w + P->fs->blocks[bi].term);
            discards |= op == OpKill || op == OpTerminateInvocation;
        }
    cg_put(g, "fm3d_program %s_program(void)\n{\n", name);
    cg_put(g, "    fm3d_program p = { %s, %s, %d, %d, NULL };\n", P->vs ? cg_str(g, "%s_vs", name) : "NULL",
           P->fs ? cg_str(g, "%s_fs", name) : "NULL", P->nvar, discards);
    cg_put(g, "    return p;\n}\n");
    char* out = NULL;
    if (g->o.fail) cg_fail(g, "out of memory");
    if (!g->failed) out = g->o.p, g->o.p = NULL;
    free(g->o.p);
    free(g);
    return out;
}

void fm3d_spirv_free_c(char* src) { free(src); }

#endif /* FM_FEATURE_SPIRV */
