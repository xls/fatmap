/*
 * fatmap - SPIR-V JIT: the lowered program as a 16 lane register machine
 * program (fm3d_jit.h), with the cleanups glslang's unoptimized output
 * needs (copy propagation, stored values forwarded to loads, dead code).
 */
#include "fm3d_jit.h"

#if FM_FEATURE_SPIRV && FM_FEATURE_JIT

#include <math.h>
#include <stdarg.h>
#include <stdio.h>

/* x86 vcmpps predicates (quiet) */
enum { P_EQ = 0x00, P_LT = 0x11, P_LE = 0x12, P_UNORD = 0x03, P_NEQ_U = 0x04, P_NLT_U = 0x15, P_NLE_U = 0x16, P_ORD = 0x07,
       P_EQ_U = 0x08, P_NGE_U = 0x19, P_NGT_U = 0x1a, P_NEQ = 0x0c, P_GE = 0x1d, P_GT = 0x1e };

enum { K_F = 0, K_U, K_I, K_B }; /* component kinds */

typedef struct jb {
    fmj_prog*         p;
    const sv_stage*   s;
    const fm3d_spirv* P;
    int*              val;  /* value id -> first vreg (-1) */
    int*              var;  /* variable id -> first vreg (-1) */
    int*              fwd;  /* variable vreg -> vreg holding its current value inside a body (-1) */
    int               fwd_n;
    uint8_t*          mu;   /* mask slots that are 0 or the entry mask */
    int               tmpm; /* a scratch mask slot */
    uint32_t*         hk;   /* constant cache: bits -> vreg */
    int*              hv;
    int               hcap;
    int               failed;
    char*             err;
    size_t            errn;
    int*              lazy; /* fs input variable -> its slot + 2 while its interpolation waits for the first load (0) */
} jb;

static void jb_fail(jb* J, const char* fmt, ...)
{
    if (J->failed) return;
    J->failed = 1;
    if (J->err && J->errn) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(J->err, J->errn, fmt, ap);
        va_end(ap);
    }
}

/* ---- vregs and ops ---- */

static int jv(jb* J)
{
    fmj_prog* p = J->p;
    if (p->nv == p->nvcap) {
        int       n  = p->nvcap ? p->nvcap * 2 : 1024;
        uint8_t*  nk = (uint8_t*)realloc(p->vk, (size_t)n);
        uint32_t* nc = nk ? (uint32_t*)realloc(p->vc, (size_t)n * 4) : NULL;
        if (nk) p->vk = nk;
        if (!nk || !nc) {
            jb_fail(J, "out of memory");
            return 0;
        }
        p->vc    = nc;
        p->nvcap = n;
    }
    p->vk[p->nv] = FMJ_K_VAL;
    p->vc[p->nv] = 0;
    return p->nv++;
}

/* a constant vreg (one per value) */
static int jc(jb* J, uint32_t bits)
{
    if (J->hcap == 0) {
        J->hcap = 1024;
        J->hk   = (uint32_t*)malloc((size_t)J->hcap * 4);
        J->hv   = (int*)malloc((size_t)J->hcap * sizeof(int));
        if (!J->hk || !J->hv) {
            jb_fail(J, "out of memory");
            return 0;
        }
        for (int i = 0; i < J->hcap; i++) J->hv[i] = -1;
    }
    uint32_t h = (bits * 2654435761u) & (uint32_t)(J->hcap - 1);
    for (int n = 0; n < J->hcap; n++, h = (h + 1) & (uint32_t)(J->hcap - 1)) {
        if (J->hv[h] < 0) {
            int v = jv(J);
            if (J->failed) return 0;
            J->p->vk[v] = FMJ_K_CONST;
            J->p->vc[v] = bits;
            J->hk[h] = bits, J->hv[h] = v;
            return v;
        }
        if (J->hk[h] == bits) return J->hv[h];
    }
    jb_fail(J, "too many constants");
    return 0;
}

static int jcf(jb* J, float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    return jc(J, b);
}

static void jset(jb* J, int op, int d, int a, int b, int c, uint32_t imm)
{
    fmj_prog* p = J->p;
    if (J->failed) return;
    if (p->nops == p->cap) {
        int     n  = p->cap ? p->cap * 2 : 1024;
        fmj_op* no = (fmj_op*)realloc(p->ops, (size_t)n * sizeof(fmj_op));
        if (!no) {
            jb_fail(J, "out of memory");
            return;
        }
        p->ops = no, p->cap = n;
    }
    fmj_op* o = &p->ops[p->nops++];
    o->op = (uint16_t)op, o->flags = 0, o->d = d, o->a = a, o->b = b, o->c = c, o->imm = imm;
}

static int jop(jb* J, int op, int a, int b, int c, uint32_t imm)
{
    int d = jv(J);
    jset(J, op, d, a, b, c, imm);
    return d;
}

#define FADD(a, b) jop(J, J_FADD, a, b, -1, 0)
#define FSUB(a, b) jop(J, J_FSUB, a, b, -1, 0)
#define FMUL(a, b) jop(J, J_FMUL, a, b, -1, 0)
#define FDIV(a, b) jop(J, J_FDIV, a, b, -1, 0)
#define FCMP(a, b, pr) jop(J, J_FCMP, a, b, -1, pr)
#define SEL(c, a, b) jop(J, J_SEL, a, b, c, 0)
#define AND(a, b) jop(J, J_AND, a, b, -1, 0)
#define OR(a, b) jop(J, J_OR, a, b, -1, 0)
#define XOR(a, b) jop(J, J_XOR, a, b, -1, 0)
#define IADD(a, b) jop(J, J_IADD, a, b, -1, 0)
#define ISUB(a, b) jop(J, J_ISUB, a, b, -1, 0)
#define SHLI(a, n) jop(J, J_SHLI, a, -1, -1, n)
#define SHRI(a, n) jop(J, J_SHRI, a, -1, -1, n)
#define SARI(a, n) jop(J, J_SARI, a, -1, -1, n)
#define ICMPGT(a, b) jop(J, J_ICMPGT, a, b, -1, 0)
#define ICMPEQ(a, b) jop(J, J_ICMPEQ, a, b, -1, 0)
#define CVTIF(a) jop(J, J_CVTIF, a, -1, -1, 0)
#define CVTFI(a) jop(J, J_CVTFI, a, -1, -1, 0)
#define KF(f) jcf(J, f)
#define KU(u) jc(J, u)
#define ALLONES jc(J, 0xffffffffu)
#define FNOTNAN(x) FCMP(x, x, P_ORD)
#define FISNAN(x) FCMP(x, x, P_UNORD)

/* ---- types ---- */

static int jk_of(const sv_stage* s, int t, int c)
{
    for (int guard = 0; guard < 64; guard++) {
        const sv_id* T = &s->ids[t];
        switch (T->kind) {
        case T_FLOAT: return K_F;
        case T_INT: return K_I;
        case T_UINT: return K_U;
        case T_BOOL: return K_B;
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

/* ---- values ---- */

static int jval(jb* J, int id)
{
    const sv_id* d = &J->s->ids[id];
    if (J->val[id] < 0) {
        int n = d->comps > 0 ? d->comps : 1;
        J->val[id] = J->p->nv;
        for (int c = 0; c < n; c++) jv(J);
    }
    return J->val[id];
}

/* component c of value id (constants: the pool; bools all ones) */
static int jref(jb* J, int id, int c)
{
    const sv_stage* s = J->s;
    const sv_id*    d = &s->ids[id];
    if (d->cls == C_CONST) {
        uint32_t bits = sv_cblock(s, d)[(size_t)c * SV_L];
        if (jk_of(s, d->type, c) == K_B) bits = bits ? 0xffffffffu : 0u;
        return jc(J, bits);
    }
    if (d->cls != C_VALUE) {
        jb_fail(J, "internal: value %d is not a value", id);
        return 0;
    }
    return jval(J, id) + c;
}

static int jdef(jb* J, int id, int c) { return jval(J, id) + c; }

static void jmov(jb* J, int d, int a)
{
    if (d != a) jset(J, J_MOV, d, a, -1, -1, 0);
}

static int jvar(jb* J, int id)
{
    if (J->var[id] < 0) {
        int n      = J->s->ids[id].comps > 0 ? J->s->ids[id].comps : 1;
        J->var[id] = J->p->nv;
        for (int c = 0; c < n; c++) jv(J);
    }
    return J->var[id];
}

/* ---- the GLSL.std.450 building blocks (fm_vmath.h, the same operations in the same order) ---- */

static int jabs(jb* J, int x) { return AND(x, KU(0x7fffffffu)); }
static int jwithsign(jb* J, int r, int x) { return OR(r, AND(x, KU(0x80000000u))); }

static int jtrunc(jb* J, int x)
{
    int small = FCMP(jabs(J, x), KF(8388608.0f), P_LT);
    int t     = jwithsign(J, CVTIF(CVTFI(SEL(small, x, KF(0.0f)))), x);
    return SEL(small, t, x);
}
static int jfloor(jb* J, int x)
{
    int t = jtrunc(J, x);
    return SEL(FCMP(t, x, P_GT), FSUB(t, KF(1.0f)), t);
}
static int jceil(jb* J, int x)
{
    int t = jtrunc(J, x);
    return SEL(FCMP(t, x, P_LT), FADD(t, KF(1.0f)), t);
}
static int jround(jb* J, int x)
{
    int t = jtrunc(J, x);
    return SEL(FCMP(jabs(J, FSUB(x, t)), KF(0.5f), P_GE), FADD(t, jwithsign(J, KF(1.0f), x)), t);
}
static int jrint(jb* J, int x)
{
    int m = jwithsign(J, KF(8388608.0f), x);
    int r = jwithsign(J, FSUB(FADD(x, m), m), x);
    return SEL(FCMP(jabs(J, x), KF(8388608.0f), P_LT), r, x);
}
static int jfmin(jb* J, int x, int y) { return SEL(OR(FCMP(y, x, P_LT), FISNAN(x)), y, x); }
static int jfmax(jb* J, int x, int y) { return SEL(OR(FCMP(y, x, P_GT), FISNAN(x)), y, x); }
/* unsigned a < b: compares with the sign bits flipped */
static int jult(jb* J, int a, int b)
{
    int k = KU(0x80000000u);
    return ICMPGT(XOR(b, k), XOR(a, k));
}
static int jodd(jb* J, int k) { return SARI(SHLI(k, 31), 31); } /* k & 1 as a lane mask */

static int jfast_log2_core(jb* J, int x)
{
    int den = ICMPGT(KU(0x00800000u), x); /* x > 0 finite: its bits are positive */
    int u   = SEL(den, FMUL(x, KF(8388608.0f)), x);
    int e   = ISUB(ISUB(SHRI(u, 23), KU(127)), AND(den, KU(23)));
    int mu  = OR(AND(u, KU(0x007fffffu)), KU(0x3f800000u));
    int hi  = ICMPGT(mu, KU(0x3fb504f3u));
    mu      = ISUB(mu, AND(hi, KU(1u << 23)));
    int s   = FDIV(FSUB(mu, KF(1.0f)), FADD(mu, KF(1.0f))), s2 = FMUL(s, s);
    int p   = FADD(KF(0.412198583f), FMUL(s2, KF(0.320598898f)));
    p       = FADD(KF(0.577078016f), FMUL(s2, p));
    p       = FADD(KF(0.961796694f), FMUL(s2, p));
    p       = FMUL(s, FADD(KF(2.88539008f), FMUL(s2, p)));
    return FADD(CVTIF(ISUB(e, hi)), p); /* (float)(e + hi): hi is -1 / 0 */
}

static int jfast_exp2_core(jb* J, int y)
{
    y     = SEL(FCMP(y, KF(-151.0f), P_GT), y, KF(-151.0f));
    y     = SEL(FCMP(y, KF(128.0f), P_LT), y, KF(128.0f));
    int n = FSUB(FADD(y, KF(12582912.0f)), KF(12582912.0f)), f = FSUB(y, n);
    int p = FADD(KF(0.00961812911f), FMUL(f, FADD(KF(0.00133335581f), FMUL(f, KF(0.000154035304f)))));
    p     = FADD(KF(0.0555041087f), FMUL(f, p));
    p     = FADD(KF(0.240226507f), FMUL(f, p));
    p     = FADD(KF(0.693147181f), FMUL(f, p));
    p     = FADD(KF(1.0f), FMUL(f, p));
    int k = CVTFI(n), k1 = SARI(IADD(k, SHRI(k, 31)), 1), k2 = ISUB(k, k1); /* k / 2 toward zero */
    return FMUL(FMUL(p, SHLI(IADD(k1, KU(127)), 23)), SHLI(IADD(k2, KU(127)), 23));
}

static int jlog_special(jb* J, int x, int r)
{
    r = SEL(FCMP(x, KU(0x7f800000u), P_EQ), x, r);
    r = SEL(FCMP(x, KF(0.0f), P_EQ), KU(0xff800000u), r);
    r = SEL(FCMP(x, KF(0.0f), P_LT), KU(0x7fc00000u), r);
    return SEL(FISNAN(x), x, r);
}

static int jfast_log2(jb* J, int x, int ln)
{
    int xs = SEL(AND(FCMP(x, KF(0.0f), P_GT), FCMP(x, KU(0x7f800000u), P_LT)), x, KF(1.0f));
    int r  = jfast_log2_core(J, xs);
    if (ln) r = FMUL(r, KF(0.693147181f));
    return jlog_special(J, x, r);
}

static int jfast_pow(jb* J, int x, int y)
{
    int inf = KU(0x7f800000u), nan = KU(0x7fc00000u), zero = KF(0.0f), one = KF(1.0f);
    int ax = jabs(J, x), ay = jabs(J, y);
    int axs = SEL(AND(FCMP(ax, zero, P_GT), FCMP(ax, inf, P_LT)), ax, one);
    int t   = FMUL(y, jfast_log2_core(J, axs));
    int r   = jfast_exp2_core(J, SEL(FISNAN(t), zero, t));
    int lt23 = FCMP(ay, KF(8388608.0f), P_LT);
    int yr   = FSUB(FADD(SEL(lt23, y, zero), KF(12582912.0f)), KF(12582912.0f));
    int yeq  = FCMP(yr, y, P_EQ);
    int yint = OR(FCMP(ay, KF(8388608.0f), P_GE), yeq);
    int odd  = AND(AND(lt23, yeq), jodd(J, CVTFI(yr)));
    int big  = SEL(FCMP(y, zero, P_GT), inf, zero);
    r        = SEL(FCMP(ax, inf, P_EQ), big, r);
    r        = SEL(FCMP(ax, zero, P_EQ), SEL(FCMP(y, zero, P_GT), zero, SEL(FCMP(y, zero, P_LT), inf, r)), r);
    r = SEL(FCMP(ay, inf, P_EQ), SEL(FCMP(ax, one, P_EQ), one, SEL(FCMP(ax, one, P_GT), big, SEL(FCMP(y, zero, P_GT), zero, inf))), r);
    int neg = SEL(yint, XOR(r, AND(odd, KU(0x80000000u))), SEL(OR(FCMP(ax, zero, P_EQ), FCMP(ax, inf, P_EQ)), r, nan));
    r       = SEL(SARI(x, 31), neg, r);
    r       = SEL(OR(FISNAN(x), FISNAN(y)), nan, r);
    return SEL(OR(FCMP(y, zero, P_EQ), FCMP(x, one, P_EQ)), one, r);
}

static int jfast_reduce(jb* J, int x, int* n)
{
    int c  = FMUL(x, KF(0.636619772f));
    int fn = FSUB(FADD(SEL(FCMP(jabs(J, c), KF(4194304.0f), P_LT), c, KF(0.0f)), KF(12582912.0f)), KF(12582912.0f));
    *n     = CVTFI(fn);
    int r  = FSUB(FSUB(FSUB(x, FMUL(fn, KF(1.5703125f))), FMUL(fn, KF(4.83751297e-4f))), FMUL(fn, KF(7.54978995e-8f)));
    r      = SEL(FCMP(r, KF(-1.0f), P_GT), r, KF(-1.0f));
    return SEL(FCMP(r, KF(1.0f), P_LT), r, KF(1.0f));
}
static int jfast_sinp(jb* J, int r, int z)
{
    int q = FADD(KF(0.00833216087f), FMUL(z, KF(-0.000195152959f)));
    q     = FADD(KF(-0.166666546f), FMUL(z, q));
    return FADD(r, FMUL(FMUL(r, z), q));
}
static int jfast_cosp(jb* J, int z)
{
    int q = FADD(KF(-0.00138873170f), FMUL(z, KF(0.0000244331571f)));
    q     = FADD(KF(0.0416666642f), FMUL(z, q));
    q     = FADD(KF(-0.5f), FMUL(z, q));
    return FADD(KF(1.0f), FMUL(z, q));
}
static int jfinite_or_nan(jb* J, int x, int v)
{
    int xx = FSUB(x, x);
    return SEL(FCMP(xx, KF(0.0f), P_EQ), v, xx);
}
static int jfast_sincos(jb* J, int x, int q)
{
    int n, r = jfast_reduce(J, x, &n), z = FMUL(r, r);
    int k = q ? IADD(n, KU(1)) : n;
    int v = SEL(jodd(J, k), jfast_cosp(J, z), jfast_sinp(J, r, z));
    v     = XOR(v, AND(SHLI(k, 30), KU(0x80000000u)));
    return jfinite_or_nan(J, x, v);
}
static int jfast_tan(jb* J, int x)
{
    int n, r = jfast_reduce(J, x, &n), z = FMUL(r, r), sp = jfast_sinp(J, r, z), cp = jfast_cosp(J, z);
    int v = SEL(jodd(J, n), FDIV(XOR(cp, KU(0x80000000u)), sp), FDIV(sp, cp));
    return jfinite_or_nan(J, x, v);
}

/* ---- helper calls: arguments / results in the frame's argument arrays ---- */

static fmj_call* jcall_new(jb* J, int kind, int fn, int n, int nargs)
{
    fmj_prog* p = J->p;
    if (p->ncalls == p->callcap) {
        int       nc = p->callcap ? p->callcap * 2 : 16;
        fmj_call* q  = (fmj_call*)realloc(p->calls, (size_t)nc * sizeof(fmj_call));
        if (!q) {
            jb_fail(J, "out of memory");
            return NULL;
        }
        p->calls = q, p->callcap = nc;
    }
    fmj_call* c = &p->calls[p->ncalls++];
    memset(c, 0, sizeof(*c));
    c->kind = kind, c->fn = fn, c->n = n, c->nargs = nargs, c->s = J->s;
    if (nargs + n > p->nargs) p->nargs = nargs + n;
    c->arg = 0;     /* argument array k: frame + off_args + 64 * k (resolved with the layout) */
    c->res = nargs; /* result arrays follow the arguments */
    return c;
}

#define ARG_OFF(k) ((int)(0x40000000 + 64 * (k))) /* placeholder offsets, relocated by jb_layout */

static void jcall_emit(jb* J, fmj_call* c, const int* args, int nargs, int* res, int nres)
{
    if (!c) {
        for (int k = 0; k < nres; k++) res[k] = 0;
        return;
    }
    for (int k = 0; k < nargs; k++) jset(J, J_STF, -1, args[k], -1, -1, (uint32_t)ARG_OFF(c->arg + k));
    jset(J, J_CALL, -1, (int)(c - J->p->calls), -1, -1, (uint32_t)c->kind);
    for (int k = 0; k < nres; k++) res[k] = jop(J, J_LDF, -1, -1, -1, (uint32_t)ARG_OFF(c->res + k));
}

/* ---- GLSL.std.450 ---- */

static void jb_ext(jb* J, const uint32_t* in)
{
    const sv_stage* s  = J->s;
    int             id = (int)in[2], n = s->ids[id].comps, fn = (int)in[4], wc = (int)WC(in);
    int             A = wc > 5 ? (int)in[5] : 0, B = wc > 6 ? (int)in[6] : 0, C = wc > 7 ? (int)in[7] : 0;
    int             fast = s->fast;
    /* the C library / precise fm_* math: one helper call for all components
     * (the interpreter's sv_math for the ISA, vectorized) */
    int hm = (fn >= 16 && fn <= 21) || fn == 25 || (!fast && ((fn >= 13 && fn <= 15) || (fn >= 26 && fn <= 30)));
    if (hm && n <= 16) {
        int       na = B ? 2 : 1, args[32], res[16];
        fmj_call* call = jcall_new(J, FMJ_H_MATH, fn, n, na * n);
        for (int c = 0; c < n; c++) {
            args[c] = jref(J, A, c);
            if (B) args[n + c] = jref(J, B, c);
        }
        jcall_emit(J, call, args, na * n, res, n);
        for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), res[c]);
        return;
    }
    /* per component functions */
    for (int c = 0; c < n && fn != 66 && fn != 67 && fn != 68 && fn != 69 && fn != 70 && fn != 71 && fn != 72; c++) {
        int x = A ? jref(J, A, c) : 0, y = B ? jref(J, B, c) : 0, z = C ? jref(J, C, c) : 0, r = -1;
        switch (fn) {
        case 1: r = jround(J, x); break;
        case 2: r = jrint(J, x); break;
        case 3: r = jtrunc(J, x); break;
        case 4: r = jabs(J, x); break;
        case 5: r = SEL(ICMPGT(KU(0), x), ISUB(KU(0), x), x); break;
        case 6: r = SEL(FCMP(x, KF(0.0f), P_GT), KF(1.0f), SEL(FCMP(x, KF(0.0f), P_LT), KF(-1.0f), KF(0.0f))); break;
        case 7: r = SEL(ICMPGT(x, KU(0)), KU(1), SEL(ICMPGT(KU(0), x), KU(0xffffffffu), KU(0))); break;
        case 8: r = jfloor(J, x); break;
        case 9: r = jceil(J, x); break;
        case 10: r = FSUB(x, jfloor(J, x)); break;
        case 11: r = FMUL(x, KF(0.017453292519943295f)); break;
        case 12: r = FMUL(x, KF(57.29577951308232f)); break;
        case 13: case 14: if (fast) r = jfast_sincos(J, x, fn == 14); break;
        case 15: if (fast) r = jfast_tan(J, x); break;
        case 26: if (fast) r = jfast_pow(J, x, y); break;
        case 27: if (fast) r = SEL(FISNAN(x), x, jfast_exp2_core(J, FMUL(x, KF(1.44269504f)))); break;
        case 28: if (fast) r = jfast_log2(J, x, 1); break;
        case 29: if (fast) r = SEL(FISNAN(x), x, jfast_exp2_core(J, x)); break;
        case 30: if (fast) r = jfast_log2(J, x, 0); break;
        case 31: r = jop(J, J_FSQRT, x, -1, -1, 0); break;
        case 32: r = FDIV(KF(1.0f), jop(J, J_FSQRT, x, -1, -1, 0)); break;
        case 37: case 79: r = jfmin(J, x, y); break;
        case 38: r = SEL(jult(J, x, y), x, y); break;
        case 39: r = SEL(ICMPGT(y, x), x, y); break;
        case 40: case 80: r = jfmax(J, x, y); break;
        case 41: r = SEL(jult(J, y, x), x, y); break;
        case 42: r = SEL(ICMPGT(x, y), x, y); break;
        case 43: case 81: r = jfmin(J, jfmax(J, x, y), z); break;
        case 44: {
            int v = SEL(jult(J, y, x), x, y);
            r     = SEL(jult(J, v, z), v, z);
            break;
        }
        case 45: {
            int v = SEL(ICMPGT(x, y), x, y);
            r     = SEL(ICMPGT(z, v), v, z);
            break;
        }
        case 46: r = FADD(FMUL(x, FSUB(KF(1.0f), z)), FMUL(y, z)); break;
        case 48: r = SEL(FCMP(y, x, P_LT), KF(0.0f), KF(1.0f)); break;
        case 49: {
            int t = jfmin(J, jfmax(J, FDIV(FSUB(z, x), FSUB(y, x)), KF(0.0f)), KF(1.0f));
            r     = FMUL(FMUL(t, t), FSUB(KF(3.0f), FMUL(KF(2.0f), t)));
            break;
        }
        case 50: r = FADD(FMUL(x, y), z); break;
        default: break;
        }
        if (r < 0) { /* the C library / precise fm_* math: a helper per component */
            int       na = C ? 3 : (B ? 2 : 1), args[3] = { x, y, z }, res;
            fmj_call* call = jcall_new(J, FMJ_H_MATH, fn, 1, na);
            jcall_emit(J, call, args, na, &res, 1);
            r = res;
        }
        jmov(J, jdef(J, id, c), r);
    }
    switch (fn) {
    case 66: case 67: { /* length, distance */
        int m = s->ids[A].comps, d = KF(0.0f);
        for (int j = 0; j < m; j++) {
            int v = fn == 66 ? jref(J, A, j) : FSUB(jref(J, A, j), jref(J, B, j));
            d     = FADD(d, FMUL(v, v));
        }
        jmov(J, jdef(J, id, 0), jop(J, J_FSQRT, d, -1, -1, 0));
        break;
    }
    case 68: { /* cross */
        int a[3], b[3];
        for (int j = 0; j < 3; j++) a[j] = jref(J, A, j), b[j] = jref(J, B, j);
        int r0 = FSUB(FMUL(a[1], b[2]), FMUL(a[2], b[1]));
        int r1 = FSUB(FMUL(a[2], b[0]), FMUL(a[0], b[2]));
        int r2 = FSUB(FMUL(a[0], b[1]), FMUL(a[1], b[0]));
        jmov(J, jdef(J, id, 0), r0), jmov(J, jdef(J, id, 1), r1), jmov(J, jdef(J, id, 2), r2);
        break;
    }
    case 69: { /* normalize */
        int d = KF(0.0f);
        for (int j = 0; j < n; j++) d = FADD(d, FMUL(jref(J, A, j), jref(J, A, j)));
        int inv = FDIV(KF(1.0f), jop(J, J_FSQRT, d, -1, -1, 0));
        int r[64];
        for (int j = 0; j < n && j < 64; j++) r[j] = FMUL(jref(J, A, j), inv);
        for (int j = 0; j < n && j < 64; j++) jmov(J, jdef(J, id, j), r[j]);
        break;
    }
    case 70: case 71: case 72: { /* faceforward(N, I, Nref), reflect(I, N), refract(I, N, eta) */
        int d = KF(0.0f), r[64];
        for (int j = 0; j < n; j++) d = fn == 70 ? FADD(d, FMUL(jref(J, C, j), jref(J, B, j))) : FADD(d, FMUL(jref(J, B, j), jref(J, A, j)));
        if (fn == 70) {
            int neg = FCMP(d, KF(0.0f), P_LT);
            for (int j = 0; j < n && j < 64; j++) r[j] = SEL(neg, jref(J, A, j), XOR(jref(J, A, j), KU(0x80000000u)));
        } else if (fn == 71) {
            for (int j = 0; j < n && j < 64; j++) r[j] = FSUB(jref(J, A, j), FMUL(FMUL(KF(2.0f), d), jref(J, B, j)));
        } else {
            int eta = jref(J, C, 0);
            int kk  = FSUB(KF(1.0f), FMUL(FMUL(eta, eta), FSUB(KF(1.0f), FMUL(d, d))));
            int neg = FCMP(kk, KF(0.0f), P_LT), sq = jop(J, J_FSQRT, kk, -1, -1, 0);
            for (int j = 0; j < n && j < 64; j++)
                r[j] = SEL(neg, KF(0.0f), FSUB(FMUL(eta, jref(J, A, j)), FMUL(FADD(FMUL(eta, d), sq), jref(J, B, j))));
        }
        for (int j = 0; j < n && j < 64; j++) jmov(J, jdef(J, id, j), r[j]);
        break;
    }
    default: break;
    }
}

/* ---- uniform block loads (std140 offsets, as sv_uload) ---- */

static void jb_uload(jb* J, int id, int t, long base, int mstride, int blk, int* c)
{
    const sv_stage* s = J->s;
    const sv_id*    T = &s->ids[t];
    switch (T->kind) {
    case T_BOOL: case T_INT: case T_UINT: case T_FLOAT: {
        int v;
        if (base < 0 || base > (1 << 26)) {
            v = KU(0); /* outside any block: 0 */
        } else {
            v = jop(J, J_LDU, -1, blk, -1, (uint32_t)base);
            J->p->ublocks |= 1u << blk;
            if (base + 4 > J->p->ubo_need[blk]) J->p->ubo_need[blk] = (int)base + 4;
        }
        if (T->kind == T_BOOL) v = XOR(ICMPEQ(v, KU(0)), ALLONES);
        jmov(J, jdef(J, id, (*c)++), v);
        break;
    }
    case T_VEC:
        for (int i = 0; i < T->count; i++) jb_uload(J, id, T->elem, base + 4 * i, 0, blk, c);
        break;
    case T_MAT:
        for (int i = 0; i < T->count; i++) jb_uload(J, id, T->elem, base + (long)i * mstride, 0, blk, c);
        break;
    case T_ARR:
        for (int i = 0; i < T->count; i++) jb_uload(J, id, T->elem, base + (long)i * T->astride, mstride, blk, c);
        break;
    case T_STRUCT:
        for (int m = 0; m < T->nmem; m++) jb_uload(J, id, T->mem[m], base + T->mboff[m], T->mstride[m], blk, c);
        break;
    default: break;
    }
}

/* the dynamic part of a pointer's offset (sum of index * stride), -1 if static */
static int jb_dyn(jb* J, const sv_id* P)
{
    if (!P->ndyn) return -1;
    int o = -1;
    for (int k = 0; k < P->ndyn; k++) {
        int ix = jref(J, P->dyn[k], 0), t = jop(J, J_IMUL, ix, KU((uint32_t)P->dstride[k]), -1, 0);
        o      = o < 0 ? t : IADD(o, t);
    }
    return o;
}

/* ---- loads / stores of variables ---- */

static int jmask_uniform(jb* J, int m) { return J->mu[m]; }

static void jb_interp(jb* J, int b, int slot, int comps);

static void jb_load(jb* J, const uint32_t* in, int n)
{
    const sv_stage* s  = J->s;
    int             id = (int)in[2];
    const sv_id*    P  = &s->ids[in[3]];
    const sv_id*    V  = &s->ids[P->pvar];
    if (J->lazy && J->lazy[P->pvar]) { /* the first load of a varying: interpolated here */
        int slot = J->lazy[P->pvar] - 2;
        J->lazy[P->pvar] = 0;
        jb_interp(J, jvar(J, P->pvar), slot, V->comps);
    }
    if (sv_is_ptr_storage_uniform(V->storage)) {
        int blk = V->binding >= 0 && V->binding < FM3D_MAX_UNIFORM_BLOCKS ? V->binding : 0, c = 0;
        if (!P->ndyn) {
            jb_uload(J, id, P->type, P->poff, P->pmstride, blk, &c);
            return;
        }
        int       off  = jb_dyn(J, P);
        fmj_call* call = jcall_new(J, FMJ_H_ULOAD, 0, n, 1);
        if (!call || n > 64) {
            jb_fail(J, "uniform value too large");
            return;
        }
        call->binding = blk, call->type = P->type, call->mstride = P->pmstride, call->poff = P->poff;
        J->p->ublocks |= 1u << blk;
        int res[64];
        jcall_emit(J, call, &off, 1, res, n);
        for (int k = 0; k < n; k++) jmov(J, jdef(J, id, k), res[k]);
        return;
    }
    int base = jvar(J, P->pvar);
    if (!P->ndyn) {
        for (int c = 0; c < n; c++) {
            int vr = base + P->poff + c, f = vr < J->fwd_n ? J->fwd[vr] : -1;
            jmov(J, jdef(J, id, c), f >= 0 ? f : vr);
        }
        return;
    }
    /* dynamic index into a variable: a select chain over the clamped offsets */
    int maxo = V->comps - n;
    if (maxo > 256) {
        jb_fail(J, "dynamically indexed variable too large");
        return;
    }
    int o = IADD(jb_dyn(J, P), KU((uint32_t)P->poff));
    for (int c = 0; c < n; c++) {
        int r = base + c; /* offset 0 (also below 0) */
        for (int k = 1; k <= maxo; k++) {
            int ge = ICMPGT(o, KU((uint32_t)(k - 1)));
            r      = SEL(ge, base + k + c, r);
        }
        jmov(J, jdef(J, id, c), r);
    }
}

static void jb_store(jb* J, const uint32_t* in, int m)
{
    const sv_stage* s = J->s;
    const sv_id*    P = &s->ids[in[1]];
    const sv_id*    V = &s->ids[P->pvar];
    if (V->storage != SC_Function && V->storage != SC_Private && V->storage != SC_Input && V->storage != SC_Output) return;
    int comps = s->ids[in[2]].comps, base = jvar(J, P->pvar), full = jmask_uniform(J, m), mv = -1;
    if (!full) mv = jop(J, J_MASKV, -1, -1, -1, (uint32_t)m);
    if (!P->ndyn) {
        for (int c = 0; c < comps; c++) {
            int vr = base + P->poff + c, src = jref(J, (int)in[2], c);
            if (full) {
                jmov(J, vr, src);
                if (vr < J->fwd_n) J->fwd[vr] = src != vr ? src : -1;
            } else {
                jset(J, J_SEL, vr, src, vr, mv, 0);
                if (vr < J->fwd_n) J->fwd[vr] = -1;
            }
        }
        return;
    }
    int maxo = V->comps - comps;
    if (maxo > 256) {
        jb_fail(J, "dynamically indexed variable too large");
        return;
    }
    int o = IADD(jb_dyn(J, P), KU((uint32_t)P->poff));
    for (int k = 0; k <= maxo; k++) {
        /* the lanes whose clamped offset is k */
        int hit = maxo == 0 ? ALLONES
                            : (k == 0 ? ICMPGT(KU(1), o) : (k == maxo ? ICMPGT(o, KU((uint32_t)(k - 1))) : ICMPEQ(o, KU((uint32_t)k))));
        if (!full) hit = AND(hit, mv);
        for (int c = 0; c < comps; c++) {
            int vr = base + k + c;
            jset(J, J_SEL, vr, jref(J, (int)in[2], c), vr, hit, 0);
            if (vr < J->fwd_n) J->fwd[vr] = -1;
        }
    }
}

/* ---- one block execution ---- */

static int jb_bin(jb* J, int op, int x, int y, int* cmp)
{
    *cmp = 0;
    switch (op) {
    case OpFAdd: return FADD(x, y);
    case OpFSub: return FSUB(x, y);
    case OpFMul: return FMUL(x, y);
    case OpFDiv: return FDIV(x, y);
    case OpFMod: return FSUB(x, FMUL(y, jfloor(J, FDIV(x, y))));
    case OpIAdd: return IADD(x, y);
    case OpISub: return ISUB(x, y);
    case OpIMul: return jop(J, J_IMUL, x, y, -1, 0);
    case OpShiftRightLogical: return jop(J, J_SHR, x, AND(y, KU(31)), -1, 0);
    case OpShiftRightArithmetic: return jop(J, J_SAR, x, AND(y, KU(31)), -1, 0);
    case OpShiftLeftLogical: return jop(J, J_SHL, x, AND(y, KU(31)), -1, 0);
    case OpBitwiseOr: return OR(x, y);
    case OpBitwiseXor: return XOR(x, y);
    case OpBitwiseAnd: return AND(x, y);
    case OpLogicalOr: return OR(x, y);
    case OpLogicalAnd: return AND(x, y);
    case OpLogicalEqual: return XOR(XOR(x, y), ALLONES);
    case OpLogicalNotEqual: return XOR(x, y);
    case OpFOrdEqual: return FCMP(x, y, P_EQ);
    case OpFUnordEqual: return FCMP(x, y, P_EQ_U);
    case OpFOrdNotEqual: return FCMP(x, y, P_NEQ);
    case OpFUnordNotEqual: return FCMP(x, y, P_NEQ_U);
    case OpFOrdLessThan: return FCMP(x, y, P_LT);
    case OpFUnordLessThan: return FCMP(x, y, P_NGE_U);
    case OpFOrdGreaterThan: return FCMP(x, y, P_GT);
    case OpFUnordGreaterThan: return FCMP(x, y, P_NLE_U);
    case OpFOrdLessThanEqual: return FCMP(x, y, P_LE);
    case OpFUnordLessThanEqual: return FCMP(x, y, P_NGT_U);
    case OpFOrdGreaterThanEqual: return FCMP(x, y, P_GE);
    case OpFUnordGreaterThanEqual: return FCMP(x, y, P_NLT_U);
    case OpIEqual: return ICMPEQ(x, y);
    case OpINotEqual: return XOR(ICMPEQ(x, y), ALLONES);
    case OpUGreaterThan: return jult(J, y, x);
    case OpUGreaterThanEqual: return XOR(jult(J, x, y), ALLONES);
    case OpULessThan: return jult(J, x, y);
    case OpULessThanEqual: return XOR(jult(J, y, x), ALLONES);
    case OpSGreaterThan: return ICMPGT(x, y);
    case OpSGreaterThanEqual: return XOR(ICMPGT(y, x), ALLONES);
    case OpSLessThan: return ICMPGT(y, x);
    case OpSLessThanEqual: return XOR(ICMPGT(x, y), ALLONES);
    default: *cmp = -1; return -1;
    }
}

static void jb_sample(jb* J, const uint32_t* in, int op)
{
    const sv_stage* s     = J->s;
    int             id    = (int)in[2], wc = (int)WC(in);
    const sv_id*    img   = &s->ids[in[3]];
    int             proj  = op == OpImageSampleProjImplicitLod || op == OpImageSampleProjExplicitLod;
    int             expl  = op == OpImageSampleExplicitLod || op == OpImageSampleProjExplicitLod;
    int             nc    = s->ids[in[4]].comps, lod = expl && wc >= 7 && (in[5] & 2u);
    int             args[8], na = 0;
    if (nc > 4) nc = 4;
    for (int c = 0; c < nc; c++) args[na++] = jref(J, (int)in[4], c);
    if (lod) args[na++] = jref(J, (int)in[6], 0);
    /* fragment stage 2D implicit LOD: straight to the ISA's quad sampler (SAMPLE2) */
    int (*s16)(const fm3d_texture*, const fm3d_sampler*, const float*, const float*, float*, float*, float*, float*) = NULL;
    if (J->p->fs && (op == OpImageSampleImplicitLod || (op == OpImageSampleProjImplicitLod && nc >= 3)) && img->dim == 1 && !img->arrayed &&
        nc >= 2) {
        fm_simd_level lv = fm_simd_current();
#if defined(FM_HAVE_AVX512_SPIRV)
        if (lv == FM_SIMD_AVX512) s16 = fm3d_sample16_avx512;
#endif
#if defined(FM_HAVE_AVX2)
        if (!s16 && (lv == FM_SIMD_AVX2 || lv == FM_SIMD_AVX512)) s16 = fm3d_sample16_avx2;
#endif
        (void)lv;
    }
    fmj_call* call = jcall_new(J, s16 ? FMJ_H_SAMPLE2 : FMJ_H_SAMPLE, op, 4, na);
    if (!call) return;
    call->unit = img->cls == C_IMG ? img->unit : -1, call->dim = img->dim, call->arrayed = img->arrayed, call->ncoord = nc, call->lod = lod;
    call->s16  = s16;
    (void)proj;
    int res[4];
    jcall_emit(J, call, args, na, res, 4);
    for (int c = 0; c < 4 && c < s->ids[id].comps; c++) jmov(J, jdef(J, id, c), res[c]);
}

static void jb_inst(jb* J, const uint32_t* in, int op, int n, int m)
{
    const sv_stage* s  = J->s;
    int             id = op == OpStore ? 0 : (int)in[2], wc = (int)WC(in);
    switch (op) {
    case OpLoad:
        if (s->ids[in[2]].cls == C_IMG) return;
        jb_load(J, in, n);
        return;
    case OpStore: jb_store(J, in, m); return;
    case OpCopyObject: case OpUConvert: case OpSConvert: case OpFConvert: case OpBitcast:
        for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), jref(J, (int)in[3], c));
        return;
    case OpCompositeConstruct: {
        int c = 0, r[256];
        for (int j = 3; j < wc && c < n && c < 256; j++)
            for (int i = 0; i < s->ids[in[j]].comps && c < n && c < 256; i++) r[c++] = jref(J, (int)in[j], i);
        for (int k = 0; k < c; k++) jmov(J, jdef(J, id, k), r[k]);
        return;
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
            for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), jref(J, base, off + c));
        } else {
            int mm = s->ids[in[3]].comps, r[256];
            for (int c = 0; c < n && c < 256; c++) r[c] = c >= off && c < off + mm ? jref(J, (int)in[3], c - off) : jref(J, base, c);
            for (int c = 0; c < n && c < 256; c++) jmov(J, jdef(J, id, c), r[c]);
        }
        return;
    }
    case OpVectorShuffle: {
        int n1 = s->ids[in[3]].comps, r[64];
        for (int j = 0; j < n && j < 64; j++) {
            uint32_t sel = in[5 + j];
            r[j] = sel == 0xffffffffu ? KU(0) : (sel < (uint32_t)n1 ? jref(J, (int)in[3], (int)sel) : jref(J, (int)in[4], (int)sel - n1));
        }
        for (int j = 0; j < n && j < 64; j++) jmov(J, jdef(J, id, j), r[j]);
        return;
    }
    case OpVectorExtractDynamic: {
        int mm = s->ids[in[3]].comps, ix = jref(J, (int)in[4], 0), r = jref(J, (int)in[3], 0);
        for (int k = 1; k < mm; k++) r = SEL(ICMPGT(ix, KU((uint32_t)(k - 1))), jref(J, (int)in[3], k), r);
        jmov(J, jdef(J, id, 0), r);
        return;
    }
    case OpVectorInsertDynamic: {
        int ix = jref(J, (int)in[5], 0), v = jref(J, (int)in[4], 0), r[64];
        for (int k = 0; k < n && k < 64; k++) {
            int hit = n == 1 ? ALLONES : (k == 0 ? ICMPGT(KU(1), ix) : (k == n - 1 ? ICMPGT(ix, KU((uint32_t)(k - 1))) : ICMPEQ(ix, KU((uint32_t)k))));
            r[k]    = SEL(hit, v, jref(J, (int)in[3], k));
        }
        for (int k = 0; k < n && k < 64; k++) jmov(J, jdef(J, id, k), r[k]);
        return;
    }
    case OpTranspose: {
        const sv_id* T = &s->ids[s->ids[in[3]].type];
        int          cols = T->count, rows = s->ids[T->elem].comps, r[256];
        for (int cc = 0; cc < cols; cc++)
            for (int rr = 0; rr < rows; rr++)
                if (rr * cols + cc < 256) r[rr * cols + cc] = jref(J, (int)in[3], cc * rows + rr);
        for (int k = 0; k < n && k < 256; k++) jmov(J, jdef(J, id, k), r[k]);
        return;
    }
    case OpConvertFToS:
        for (int c = 0; c < n; c++) {
            int v = jref(J, (int)in[3], c);
            int r = SEL(FNOTNAN(v), CVTFI(v), KU(0));
            r     = SEL(FCMP(v, KF(-2147483648.0f), P_LE), KU(0x80000000u), r);
            r     = SEL(FCMP(v, KF(2147483520.0f), P_GE), KU(0x7fffffffu), r);
            jmov(J, jdef(J, id, c), r);
        }
        return;
    case OpConvertFToU:
        for (int c = 0; c < n; c++) {
            int v  = jref(J, (int)in[3], c);
            int hi = FCMP(v, KF(2147483648.0f), P_GE);
            int t  = SEL(hi, IADD(CVTFI(FSUB(v, KF(2147483648.0f))), KU(0x80000000u)), CVTFI(v));
            int r  = SEL(FCMP(v, KF(0.0f), P_GT), t, KU(0));
            r      = SEL(FCMP(v, KF(4294967040.0f), P_GE), KU(0xffffffffu), r);
            jmov(J, jdef(J, id, c), r);
        }
        return;
    case OpConvertSToF: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), CVTIF(jref(J, (int)in[3], c))); return;
    case OpConvertUToF: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), jop(J, J_CVTUF, jref(J, (int)in[3], c), -1, -1, 0)); return;
    case OpFNegate: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), XOR(jref(J, (int)in[3], c), KU(0x80000000u))); return;
    case OpSNegate: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), ISUB(KU(0), jref(J, (int)in[3], c))); return;
    case OpNot: case OpLogicalNot: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), XOR(jref(J, (int)in[3], c), ALLONES)); return;
    case OpIsNan: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), FISNAN(jref(J, (int)in[3], c))); return;
    case OpIsInf: for (int c = 0; c < n; c++) jmov(J, jdef(J, id, c), FCMP(jabs(J, jref(J, (int)in[3], c)), KU(0x7f800000u), P_EQ)); return;
    case OpAny: case OpAll: {
        int mm = s->ids[in[3]].comps, r = jref(J, (int)in[3], 0);
        for (int j = 1; j < mm; j++) r = op == OpAll ? AND(r, jref(J, (int)in[3], j)) : OR(r, jref(J, (int)in[3], j));
        jmov(J, jdef(J, id, 0), r);
        return;
    }
    case OpSelect: {
        int cc = s->ids[in[3]].comps, r[256];
        for (int j = 0; j < n && j < 256; j++) r[j] = SEL(jref(J, (int)in[3], cc == 1 ? 0 : j), jref(J, (int)in[4], j), jref(J, (int)in[5], j));
        for (int j = 0; j < n && j < 256; j++) jmov(J, jdef(J, id, j), r[j]);
        return;
    }
    case OpVectorTimesScalar: case OpMatrixTimesScalar: {
        int b = jref(J, (int)in[4], 0);
        for (int j = 0; j < n; j++) jmov(J, jdef(J, id, j), FMUL(jref(J, (int)in[3], j), b));
        return;
    }
    case OpDot: {
        int mm = s->ids[in[3]].comps, d = FMUL(jref(J, (int)in[3], 0), jref(J, (int)in[4], 0));
        for (int j = 1; j < mm; j++) d = FADD(d, FMUL(jref(J, (int)in[3], j), jref(J, (int)in[4], j)));
        jmov(J, jdef(J, id, 0), d);
        return;
    }
    case OpMatrixTimesVector: {
        const sv_id* T = &s->ids[s->ids[in[3]].type];
        int          cols = T->count, rows = s->ids[T->elem].comps, r[64];
        for (int rr = 0; rr < rows && rr < 64; rr++) {
            int d = FMUL(jref(J, (int)in[3], rr), jref(J, (int)in[4], 0));
            for (int cc = 1; cc < cols; cc++) d = FADD(d, FMUL(jref(J, (int)in[3], cc * rows + rr), jref(J, (int)in[4], cc)));
            r[rr] = d;
        }
        for (int rr = 0; rr < rows && rr < 64; rr++) jmov(J, jdef(J, id, rr), r[rr]);
        return;
    }
    case OpVectorTimesMatrix: {
        const sv_id* T = &s->ids[s->ids[in[4]].type];
        int          cols = T->count, rows = s->ids[T->elem].comps, r[64];
        for (int cc = 0; cc < cols && cc < 64; cc++) {
            int d = FMUL(jref(J, (int)in[3], 0), jref(J, (int)in[4], cc * rows));
            for (int rr = 1; rr < rows; rr++) d = FADD(d, FMUL(jref(J, (int)in[3], rr), jref(J, (int)in[4], cc * rows + rr)));
            r[cc] = d;
        }
        for (int cc = 0; cc < cols && cc < 64; cc++) jmov(J, jdef(J, id, cc), r[cc]);
        return;
    }
    case OpMatrixTimesMatrix: {
        const sv_id* TA = &s->ids[s->ids[in[3]].type];
        const sv_id* TB = &s->ids[s->ids[in[4]].type];
        int          rows = s->ids[TA->elem].comps, inner = TA->count, cols = TB->count, r[256];
        for (int cc = 0; cc < cols; cc++)
            for (int rr = 0; rr < rows; rr++) {
                int d = FMUL(jref(J, (int)in[3], rr), jref(J, (int)in[4], cc * inner));
                for (int j = 1; j < inner; j++) d = FADD(d, FMUL(jref(J, (int)in[3], j * rows + rr), jref(J, (int)in[4], cc * inner + j)));
                if (cc * rows + rr < 256) r[cc * rows + rr] = d;
            }
        for (int k = 0; k < n && k < 256; k++) jmov(J, jdef(J, id, k), r[k]);
        return;
    }
    case OpOuterProduct: {
        int rows = s->ids[in[3]].comps, cols = s->ids[in[4]].comps;
        for (int cc = 0; cc < cols; cc++)
            for (int rr = 0; rr < rows; rr++) jmov(J, jdef(J, id, cc * rows + rr), FMUL(jref(J, (int)in[3], rr), jref(J, (int)in[4], cc)));
        return;
    }
    case OpImageSampleImplicitLod: case OpImageSampleExplicitLod: case OpImageSampleProjImplicitLod: case OpImageSampleProjExplicitLod:
        jb_sample(J, in, op);
        return;
    case OpDPdx: case OpDPdy: case OpFwidth: case OpDPdxFine: case OpDPdyFine: case OpFwidthFine: case OpDPdxCoarse:
    case OpDPdyCoarse: case OpFwidthCoarse: {
        int dx = op == OpDPdx || op == OpDPdxFine || op == OpDPdxCoarse;
        int dy = op == OpDPdy || op == OpDPdyFine || op == OpDPdyCoarse;
        for (int c = 0; c < n; c++) {
            int v = jref(J, (int)in[3], c), r;
            if (!J->p->fs) r = KF(0.0f);
            else if (dx) r = jop(J, J_DDX, v, -1, -1, 0);
            else if (dy) r = jop(J, J_DDY, v, -1, -1, 0);
            else r = FADD(jabs(J, jop(J, J_DDX, v, -1, -1, 0)), jabs(J, jop(J, J_DDY, v, -1, -1, 0)));
            jmov(J, jdef(J, id, c), r);
        }
        return;
    }
    case OpExtInst: jb_ext(J, in); return;
    case OpUDiv: case OpUMod: case OpSDiv: case OpSRem: case OpSMod: case OpFRem: {
        for (int c = 0; c < n; c++) {
            int       args[2] = { jref(J, (int)in[3], c), jref(J, (int)in[4], c) }, r;
            fmj_call* call    = jcall_new(J, FMJ_H_SLOW, op, 1, 2);
            jcall_emit(J, call, args, 2, &r, 1);
            jmov(J, jdef(J, id, c), r);
        }
        return;
    }
    case OpVariable: case OpSelectionMerge: case OpLoopMerge: case OpUndef: case OpAccessChain: case OpInBoundsAccessChain: return;
    default: break;
    }
    /* binary per component ops */
    {
        int cmp, r[256];
        for (int c = 0; c < n && c < 256; c++) {
            int x = jref(J, (int)in[3], c), y = jref(J, (int)in[4], c);
            r[c]  = jb_bin(J, op, x, y, &cmp);
            if (cmp < 0) {
                jb_fail(J, "opcode %d is not supported by the JIT", op);
                return;
            }
        }
        for (int c = 0; c < n && c < 256; c++) jmov(J, jdef(J, id, c), r[c]);
    }
}

static void jb_body(jb* J, int bi, int m)
{
    const sv_stage* s = J->s;
    const sv_block* B = &s->blocks[bi];
    for (int i = 0; i < J->fwd_n; i++) J->fwd[i] = -1; /* forwarded stores: within a body */
    if (B->nphi) { /* phis: staged (they read the old values), then committed */
        const int* pm = B->phim;
        int        stage[256], ns = 0;
        for (int ii = 0; ii < B->nphi; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             id = (int)in[2], n = s->ids[id].comps, t0 = ns;
            for (int c = 0; c < n && ns < 256; c++) {
                stage[ns] = jv(J);
                jmov(J, stage[ns], jref(J, id, c));
                ns++;
            }
            for (int k = 3; k + 1 < (int)WC(in); k += 2, pm++) {
                if (*pm == SV_M_ZERO) continue;
                jset(J, J_MAND, -1, J->tmpm, *pm, m, 0);
                int mv = jop(J, J_MASKV, -1, -1, -1, (uint32_t)J->tmpm);
                for (int c = 0; c < n && t0 + c < 256; c++) jset(J, J_SEL, stage[t0 + c], jref(J, (int)in[k], c), stage[t0 + c], mv, 0);
            }
        }
        ns = 0;
        for (int ii = 0; ii < B->nphi; ii++) {
            int id = (int)s->w[B->insts[ii].at + 2];
            for (int c = 0; c < s->ids[id].comps && ns < 256; c++) jmov(J, jdef(J, id, c), stage[ns++]);
        }
        if (ns >= 256) jb_fail(J, "too many phi components");
    }
    for (int i = 0; i < B->nclr; i++) jset(J, J_MZERO, -1, B->clr[i], -1, -1, 0); /* edges consumed */
    for (int ii = B->nphi; ii < B->ninst && !J->failed; ii++) jb_inst(J, s->w + B->insts[ii].at, B->insts[ii].op, B->insts[ii].n, m);
}

/* mask slots that only hold 0 or the entry mask (stores under them need no blend):
 * the lowered program's mask ops to a fixed point */
static void jb_mask_uniform(jb* J)
{
    const sv_stage* s = J->s;
    for (int i = 0; i < s->nmask; i++) J->mu[i] = 1;
    for (int round = 0; round < 64; round++) {
        int changed = 0;
        for (int i = 0; i < s->nir; i++) {
            const sv_ir* o = &s->ir[i];
            int          u = -1;
            switch (o->op) {
            case IR_COPY: u = J->mu[o->b]; break;
            case IR_OR: case IR_ANDN: u = J->mu[o->b] && J->mu[o->c]; break;
            case IR_COND: case IR_EQ: u = J->mu[o->c] && (s->vcls[o->b] == VC_UNIFORM || s->vcls[o->b] == VC_CONST); break;
            default: break;
            }
            if (u == 0 && J->mu[o->a]) J->mu[o->a] = 0, changed = 1;
        }
        if (!changed) break;
    }
}

/* ---- cleanups ---- */

/* copies of single definition values defined before them: uses read the source */
static void jb_copyprop(fmj_prog* p)
{
    int* ndef = (int*)calloc((size_t)p->nv, sizeof(int));
    int* dpos = (int*)malloc((size_t)p->nv * sizeof(int));
    int* first_use = (int*)malloc((size_t)p->nv * sizeof(int));
    int* rep = (int*)malloc((size_t)p->nv * sizeof(int));
    if (!ndef || !dpos || !first_use || !rep) goto done;
    for (int v = 0; v < p->nv; v++) dpos[v] = -1, first_use[v] = 0x7fffffff, rep[v] = v;
    for (int i = 0; i < p->nops; i++) {
        fmj_op* o = &p->ops[i];
        int     uses[3] = { o->a, o->b, o->c }, nu = 0;
        switch (o->op) {
        case J_MOV: case J_FSQRT: case J_SHLI: case J_SHRI: case J_SARI: case J_CVTIF: case J_CVTUF: case J_CVTFI: case J_DDX: case J_DDY:
        case J_STF: nu = 1; break;
        case J_SEL: nu = 3; break;
        case J_FADD: case J_FSUB: case J_FMUL: case J_FDIV: case J_FCMP: case J_IADD: case J_ISUB: case J_IMUL: case J_AND: case J_OR:
        case J_XOR: case J_ANDN: case J_SHL: case J_SHR: case J_SAR: case J_ICMPEQ: case J_ICMPGT: nu = 2; break;
        case J_MCOND: uses[0] = o->b, nu = 1; break;
        default: break;
        }
        for (int k = 0; k < nu; k++)
            if (uses[k] >= 0 && i < first_use[uses[k]]) first_use[uses[k]] = i;
        if (o->d >= 0) ndef[o->d]++, dpos[o->d] = i;
    }
    /* a MOV d = a can be dropped when d has one definition (this one) that
     * comes before every use of d, and a has at most one, before the MOV */
    for (int i = 0; i < p->nops; i++) {
        fmj_op* o = &p->ops[i];
        if (o->op != J_MOV) continue;
        int d = o->d, a = o->a;
        if (ndef[d] != 1 || first_use[d] < i) continue;
        if (p->vk[a] != FMJ_K_CONST && (ndef[a] > 1 || (ndef[a] == 1 && dpos[a] > i))) continue;
        rep[d] = a;
        o->op  = J_NOP, o->d = -1;
    }
    for (int i = 0; i < p->nops; i++) { /* rewrite uses through the replacement chains */
        fmj_op* o = &p->ops[i];
        int*    f[3] = { &o->a, &o->b, &o->c };
        int     nu = 0;
        switch (o->op) {
        case J_MOV: case J_FSQRT: case J_SHLI: case J_SHRI: case J_SARI: case J_CVTIF: case J_CVTUF: case J_CVTFI: case J_DDX: case J_DDY:
        case J_STF: nu = 1; break;
        case J_SEL: nu = 3; break;
        case J_FADD: case J_FSUB: case J_FMUL: case J_FDIV: case J_FCMP: case J_IADD: case J_ISUB: case J_IMUL: case J_AND: case J_OR:
        case J_XOR: case J_ANDN: case J_SHL: case J_SHR: case J_SAR: case J_ICMPEQ: case J_ICMPGT: nu = 2; break;
        case J_MCOND: f[0] = &o->b, nu = 1; break;
        default: break;
        }
        for (int k = 0; k < nu; k++) {
            int v = *f[k];
            for (int guard = 0; v >= 0 && rep[v] != v && guard < 1000; guard++) v = rep[v];
            *f[k] = v;
        }
    }
done:
    free(ndef), free(dpos), free(first_use), free(rep);
}

static int jb_uses(const fmj_op* o, int* u)
{
    switch (o->op) {
    case J_MOV: case J_FSQRT: case J_SHLI: case J_SHRI: case J_SARI: case J_CVTIF: case J_CVTUF: case J_CVTFI: case J_DDX: case J_DDY:
    case J_STF: u[0] = o->a; return 1;
    case J_SEL: u[0] = o->a, u[1] = o->b, u[2] = o->c; return 3;
    case J_FADD: case J_FSUB: case J_FMUL: case J_FDIV: case J_FCMP: case J_IADD: case J_ISUB: case J_IMUL: case J_AND: case J_OR:
    case J_XOR: case J_ANDN: case J_SHL: case J_SHR: case J_SAR: case J_ICMPEQ: case J_ICMPGT: u[0] = o->a, u[1] = o->b; return 2;
    case J_MCOND: u[0] = o->b; return 1;
    default: return 0;
    }
}

/* definitions nothing reads (global use counts, to a fixed point) */
static void jb_dce(fmj_prog* p)
{
    int* nuse = (int*)calloc((size_t)p->nv, sizeof(int));
    if (!nuse) return;
    for (int i = 0; i < p->nops; i++) {
        int u[3], n = jb_uses(&p->ops[i], u);
        for (int k = 0; k < n; k++)
            if (u[k] >= 0) nuse[u[k]]++;
    }
    for (int changed = 1; changed;) {
        changed = 0;
        for (int i = p->nops - 1; i >= 0; i--) {
            fmj_op* o = &p->ops[i];
            if (o->d < 0 || o->op == J_NOP || o->op == J_CALL || nuse[o->d]) continue;
            if (o->op == J_SEL && (o->b == o->d)) { /* d = c ? a : d also reads d: dead only without other uses */
            }
            int u[3], n = jb_uses(o, u);
            for (int k = 0; k < n; k++)
                if (u[k] >= 0) nuse[u[k]]--;
            o->op = J_NOP, o->d = -1;
            changed = 1;
        }
    }
    /* compact */
    int k = 0;
    for (int i = 0; i < p->nops; i++)
        if (p->ops[i].op != J_NOP) p->ops[k++] = p->ops[i];
    p->nops = k;
    free(nuse);
}

/* definitions overwritten before any read: a definition outside loops is dead when the
 * next touch of its vreg is a definition that always runs (outside every if and loop)
 * with no read between them in program order (the zero initializations of function
 * variables and phi slots, mostly: they otherwise live, and spill, over the program) */
static void jb_dead_defs(fmj_prog* p)
{
    int* last = (int*)malloc((size_t)p->nv * sizeof(int));
    if (!last) return;
    for (int v = 0; v < p->nv; v++) last[v] = -1;
    int dif = 0, dloop = 0;
    for (int i = 0; i < p->nops; i++) {
        fmj_op* o = &p->ops[i];
        switch (o->op) {
        case J_IF: dif++; break;
        case J_ENDIF: dif--; break;
        case J_LOOP: dloop++; break;
        case J_ENDLOOP: dloop--; break;
        default: break;
        }
        int u[3], n = jb_uses(o, u);
        for (int k = 0; k < n; k++)
            if (u[k] >= 0) last[u[k]] = -1;
        if (o->d < 0 || o->op == J_CALL || o->op == J_NOP) continue;
        if (last[o->d] >= 0 && dif == 0 && dloop == 0) p->ops[last[o->d]].op = J_NOP, p->ops[last[o->d]].d = -1;
        last[o->d] = dloop == 0 ? i : -1;
    }
    free(last);
}

/* jumps of the control flow markers: c = index of the matching end / start */
static int jb_link(fmj_prog* p, char* err, size_t errn)
{
    int stack[512], sp = 0;
    for (int i = 0; i < p->nops; i++) {
        fmj_op* o = &p->ops[i];
        if (o->op == J_IF || o->op == J_LOOP) {
            if (sp == 512) {
                if (err && errn) snprintf(err, errn, "control flow nested too deeply");
                return 0;
            }
            stack[sp++] = i;
        } else if (o->op == J_ENDIF || o->op == J_ENDLOOP) {
            if (!sp) {
                if (err && errn) snprintf(err, errn, "internal: unbalanced control flow");
                return 0;
            }
            int s = stack[--sp];
            p->ops[s].c = i;
            o->c        = s;
        }
    }
    for (int i = 0; i < p->nops; i++) { /* breaks: the enclosing loop's end */
        fmj_op* o = &p->ops[i];
        if (o->op != J_BREAKZ && o->op != J_BREAK) continue;
        int depth = 0, j = i + 1;
        for (; j < p->nops; j++) {
            if (p->ops[j].op == J_LOOP) depth++;
            else if (p->ops[j].op == J_ENDLOOP && depth-- == 0) break;
        }
        o->c = j;
    }
    return 1;
}

/* frame areas: masks, call arguments, inputs, outputs (spills: the code generator) */
static void jb_layout(fmj_prog* p)
{
    int off = (int)((sizeof(fmj_frame) + 63) & ~(size_t)63);
    p->off_m = off;
    off += ((p->nmask * 4 + 63) & ~63);
    p->off_args = off;
    off += p->nargs * 64;
    p->off_in = off;
    off += p->nin * 64;
    p->off_out = off;
    off += p->nout * 64;
    p->off_spill  = off;
    p->frame_size = off;
    for (int i = 0; i < p->nops; i++) {
        fmj_op* o = &p->ops[i];
        if ((o->op == J_LDF || o->op == J_STF) && o->imm >= 0x40000000u) o->imm = (uint32_t)p->off_args + (o->imm - 0x40000000u);
    }
    for (int i = 0; i < p->ncalls; i++) {
        p->calls[i].arg = p->off_args + 64 * p->calls[i].arg;
        p->calls[i].res = p->off_args + 64 * p->calls[i].res;
    }
}

/* ---- the program ---- */

/* fs input varying (fm3d_fs_io planes): component c of slot is
 * (p0 + p2 * dy + p1 * dx) * w, the raster's interpolation in its order */
static void jb_interp(jb* J, int b, int slot, int comps)
{
    const fm3d_spirv* P = J->P;
    int               DX = jop(J, J_LDF, -1, -1, -1, 0x20000000u), DY = jop(J, J_LDF, -1, -1, -1, 0x20000000u + 64), W = jop(J, J_LDF, -1, -1, -1, 0x20000000u + 128);
    for (int c = 0; c < comps; c++) {
        int k = slot + c;
        if (slot < 0 || k >= P->nvar || k >= FM3D_MAX_SHADER_VARYINGS) {
            jmov(J, b + c, KF(0.0f));
            continue;
        }
        int p0 = jop(J, J_LDU, -1, FMJ_PLANES, -1, (uint32_t)(12 * k)), p1 = jop(J, J_LDU, -1, FMJ_PLANES, -1, (uint32_t)(12 * k + 4));
        int p2 = jop(J, J_LDU, -1, FMJ_PLANES, -1, (uint32_t)(12 * k + 8));
        jmov(J, b + c, FMUL(FADD(FADD(p0, FMUL(p2, DY)), FMUL(p1, DX)), W));
    }
    J->p->interp = 1;
}

/* per variable: the control flow depth of its first load in program order (-1: never
 * loaded); a varying first read at depth 0 can be interpolated there (every lane group
 * runs it), not at the entry with a live range over the whole program */
static void jb_first_loads(const sv_stage* s, int* depth)
{
    for (uint32_t i = 0; i < s->bound; i++) depth[i] = -1;
    int d = 0;
    for (int i = 0; i < s->nir; i++) {
        const sv_ir* o = &s->ir[i];
        if (o->op == IR_IF || o->op == IR_LOOP) d++;
        if (o->op == IR_ENDIF || o->op == IR_ENDLOOP) d--;
        if (o->op != IR_BODY) continue;
        const sv_block* B = &s->blocks[o->a];
        for (int ii = B->nphi; ii < B->ninst; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            if ((in[0] & 0xffffu) != OpLoad) continue;
            int v = s->ids[in[3]].pvar;
            if (v >= 0 && (uint32_t)v < s->bound && depth[v] < 0) depth[v] = d;
        }
    }
}

fmj_prog* fmj_build(const sv_stage* s, const fm3d_spirv* P, int fs, char* err, size_t errn)
{
    if (err && errn) err[0] = 0;
    fmj_prog* p = (fmj_prog*)calloc(1, sizeof(fmj_prog));
    int*      first = NULL; /* fs: depth of each variable's first load */
    jb        J;
    memset(&J, 0, sizeof(J));
    if (!p) return NULL;
    J.p = p, J.s = s, J.P = P, J.err = err, J.errn = errn;
    p->fs = fs, p->s = s, p->nmask = s->nmask + 1;
    J.tmpm = s->nmask;
    J.val = (int*)malloc(s->bound * sizeof(int));
    J.var = (int*)malloc(s->bound * sizeof(int));
    J.mu  = (uint8_t*)malloc((size_t)p->nmask);
    p->vw = (int*)malloc(s->bound * sizeof(int));
    if (!J.val || !J.var || !J.mu || !p->vw) {
        jb_fail(&J, "out of memory");
        goto out;
    }
    for (uint32_t i = 0; i < s->bound; i++) J.val[i] = J.var[i] = p->vw[i] = -1;
    jb_mask_uniform(&J);
    J.mu[J.tmpm] = 0;
    /* variables: inputs from the frame (fs varyings: from the planes), the others zero /
     * initialized; outputs get frame words. fs in words 0..2: dx, dy, w of the group */
    if (fs) {
        p->nin = 3;
        J.lazy = (int*)calloc(s->bound, sizeof(int));
        if (!J.lazy) jb_fail(&J, "out of memory");
    }
    first = fs ? (int*)malloc(s->bound * sizeof(int)) : NULL;
    if (fs && first) jb_first_loads(s, first);
    for (int i = 0; i < s->nvars && !J.failed; i++) {
        int          vid = s->vars[i];
        const sv_id* v   = &s->ids[vid];
        int          b   = jvar(&J, vid);
        if (v->storage == SC_Input && fs) {
            int ii = -1, n = 0;
            for (int k = 0; k < s->nin; k++)
                if (s->in[k].var == vid) ii = k, n++;
            if (n == 1 && s->in[ii].builtin != BI_FragCoord && s->in[ii].builtin != BI_FrontFacing && s->in[ii].off == 0) {
                int d = first ? first[vid] : 1;
                p->interp = 1;
                if (d == 0 && J.lazy) J.lazy[vid] = P->fslot[ii] + 2; /* at its first load */
                else if (d != -1) jb_interp(&J, b, P->fslot[ii], v->comps); /* inside control flow first: at the entry */
                continue;
            }
        }
        if (v->storage == SC_Input) {
            p->vw[vid] = p->nin;
            for (int c = 0; c < v->comps; c++) jset(&J, J_LDF, b + c, -1, -1, -1, (uint32_t)(0x20000000 + 64 * (p->nin + c)));
            p->nin += v->comps;
            continue;
        }
        if (v->storage == SC_Output) p->vw[vid] = p->nout, p->nout += v->comps;
        for (int c = 0; c < v->comps; c++) {
            uint32_t bits = 0;
            if (v->init && s->ids[v->init].cls == C_CONST) {
                bits = sv_cblock(s, &s->ids[v->init])[(size_t)c * SV_L];
                if (jk_of(s, v->type, c) == K_B) bits = bits ? 0xffffffffu : 0u;
            }
            jmov(&J, b + c, jc(&J, bits));
        }
    }
    /* phis keep their old value in lanes without an edge: start at 0 */
    for (int bi = 0; bi < s->nblocks; bi++)
        for (int ii = 0; ii < s->blocks[bi].nphi; ii++) {
            int id = (int)s->w[s->blocks[bi].insts[ii].at + 2];
            for (int c = 0; c < s->ids[id].comps; c++) jmov(&J, jdef(&J, id, c), jc(&J, 0));
        }
    /* forwarded stores: one entry per variable vreg (they come first) */
    J.fwd_n = p->nv;
    J.fwd   = (int*)malloc(((size_t)p->nv + 1) * sizeof(int));
    if (!J.fwd) jb_fail(&J, "out of memory");
    /* the lowered program */
    for (int i = 0; i < s->nir && !J.failed; i++) {
        const sv_ir* o = &s->ir[i];
        switch (o->op) {
        case IR_ZERO: jset(&J, J_MZERO, -1, o->a, -1, -1, 0); break;
        case IR_COPY: jset(&J, J_MCOPY, -1, o->a, o->b, -1, 0); break;
        case IR_OR: jset(&J, J_MOR, -1, o->a, o->b, o->c, 0); break;
        case IR_ANDN: jset(&J, J_MANDN, -1, o->a, o->b, o->c, 0); break;
        case IR_COND: {
            int v = jref(&J, o->b, 0);
            jset(&J, J_MCOND, -1, o->a, v, o->c, 0);
            break;
        }
        case IR_EQ: {
            int v = jop(&J, J_ICMPEQ, jref(&J, o->b, 0), jc(&J, o->lit), -1, 0);
            jset(&J, J_MCOND, -1, o->a, v, o->c, 0);
            break;
        }
        case IR_BODY: jb_body(&J, o->a, o->b); break;
        case IR_IF: jset(&J, J_IF, -1, o->a, -1, -1, 0); break;
        case IR_ENDIF: jset(&J, J_ENDIF, -1, -1, -1, -1, 0); break;
        case IR_LOOP: jset(&J, J_LOOP, -1, o->a, -1, -1, 0); break;
        case IR_BREAKZ: jset(&J, J_BREAKZ, -1, o->a, o->b, -1, 0); break;
        case IR_BREAK: jset(&J, J_BREAK, -1, -1, -1, -1, 0); break;
        case IR_ENDLOOP: jset(&J, J_ENDLOOP, -1, -1, -1, -1, 0); break;
        default: break;
        }
    }
    /* outputs to the frame */
    for (int i = 0; i < s->nvars && !J.failed; i++) {
        int          vid = s->vars[i];
        const sv_id* v   = &s->ids[vid];
        if (v->storage != SC_Output) continue;
        for (int c = 0; c < v->comps; c++) jset(&J, J_STF, -1, J.var[vid] + c, -1, -1, (uint32_t)(0x30000000 + 64 * (p->vw[vid] + c)));
    }
    /* fs color 0 also as straight 8 bit ARGB (straight_f: (int)(min(max(c, 0), 1) * 255 + 0.5),
     * maxps / minps operand order; missing channels: 0, alpha 1) */
    p->packw = -1;
    for (int i = 0; fs && i < s->nout && !J.failed; i++) {
        const sv_io* fo = &s->out[i];
        if (fo->loc != 0 || fo->builtin >= 0 || fo->off != 0 || p->vw[fo->var] < 0) continue;
        jb* JP = &J;
        int ch[4];
        for (int c = 0; c < 4; c++) {
            if (c >= s->ids[fo->var].comps) {
                ch[c] = jc(JP, c == 3 ? 255u : 0u);
                continue;
            }
            int x = J.var[fo->var] + c;
#define J JP
            int mx = SEL(FCMP(x, KF(0.0f), P_GT), x, KF(0.0f));
            int mn = SEL(FCMP(mx, KF(1.0f), P_LT), mx, KF(1.0f));
            ch[c]  = CVTFI(FADD(FMUL(mn, KF(255.0f)), KF(0.5f)));
        }
        int pk = OR(OR(SHLI(ch[3], 24), SHLI(ch[0], 16)), OR(SHLI(ch[1], 8), ch[2]));
#undef J
        p->packw = p->nout++;
        jset(&J, J_STF, -1, pk, -1, -1, (uint32_t)(0x30000000 + 64 * p->packw));
        break;
    }
    if (J.failed) goto out;
    jb_copyprop(p);
    jb_dead_defs(p);
    jb_dce(p);
    if (!jb_link(p, err, errn)) {
        J.failed = 1;
        goto out;
    }
    jb_layout(p);
    { /* FMJ_IRDUMP=<file>: append the ops (debugging) */
        const char* dn = getenv("FMJ_IRDUMP");
        FILE*       df = dn ? fopen(dn, "a") : NULL;
        if (df) {
            fprintf(df, "== %s, %d ops, %d vregs\n", fs ? "fs" : "vs", p->nops, p->nv);
            for (int i = 0; i < p->nops; i++) {
                const fmj_op* o = &p->ops[i];
                fprintf(df, "%5d op %2d d %4d a %4d b %4d c %4d imm %08x%s\n", i, o->op, o->d, o->a, o->b, o->c, o->imm,
                        o->d >= 0 && p->vk[o->d] == FMJ_K_CONST ? " const" : "");
            }
            fclose(df);
        }
    }
    for (int i = 0; i < p->nops; i++) { /* input / output word offsets */
        fmj_op* o = &p->ops[i];
        if (o->op == J_LDF && o->imm >= 0x20000000u && o->imm < 0x30000000u) o->imm = (uint32_t)p->off_in + (o->imm - 0x20000000u);
        if (o->op == J_STF && o->imm >= 0x30000000u && o->imm < 0x40000000u) o->imm = (uint32_t)p->off_out + (o->imm - 0x30000000u);
    }
out:
    free(J.val), free(J.var), free(J.mu), free(J.fwd), free(J.hk), free(J.hv), free(J.lazy), free(first);
    if (J.failed) {
        fmj_free(p);
        return NULL;
    }
    return p;
}

void fmj_free(fmj_prog* p)
{
    if (!p) return;
    free(p->ops), free(p->vk), free(p->vc), free(p->calls), free(p->vw);
    if (p->code) fmj_code_free(p->code, p->code_cap);
    free(p);
}

#endif
