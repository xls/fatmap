/*
 * fatmap - SPIR-V JIT: x86 machine code for the register program
 * (fm3d_jit.h), x86-64 and x86-32, AVX-512 (one zmm per 16 lane value) or
 * AVX2 (a pair of ymm).
 *
 * Linear scan register allocation over the program in order: a value's
 * range runs from its first to its last mention, widened to whole loops it
 * is live around; a value whose range crosses a helper call lives in its
 * frame slot (calls clobber every vector register). Operands that are not in
 * registers (spilled values, constants from the pool in front of the code)
 * are memory operands where the instruction takes one, else go through the
 * two temporaries. The frame pointer stays in rbx / ebx.
 *
 * Every instruction is the IEEE operation the reference executor performs,
 * so the code renders the same bits.
 */
#include "fm3d_jit.h"

#if FM_FEATURE_SPIRV && FM_FEATURE_JIT

#include <stddef.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#  define FMJ_X64 1
#elif defined(__i386__) || defined(_M_IX86)
#  define FMJ_X64 0
#else
#  define FMJ_NO_X86 1
#endif

#if !defined(FMJ_NO_X86)

#if defined(_WIN32) && FMJ_X64
#  define FMJ_WIN64 1
#else
#  define FMJ_WIN64 0
#endif

enum { XR = 1, XF, XP, XA }; /* register, [frame + disp], pool (absolute / rip relative), [rax + disp] */

typedef struct xo {
    int       kind;
    int       r;
    int32_t   disp;
    uintptr_t addr;
} xo;

typedef struct xenc {
    uint8_t map, pp, op, w;
} xenc;

typedef struct xfix { /* a rel32 jump to an op index */
    size_t at;
    int    target;
} xfix;

typedef struct xc {
    fmj_prog* p;
    uint8_t*  buf;
    size_t    cap, n;
    uintptr_t base;    /* address of buf */
    int       e512;    /* AVX-512 (EVEX, zmm) else AVX2 (VEX, ymm pairs) */
    int       parts;   /* 1 or 2 */
    int       t0, t1;  /* temporaries (physical registers) */
    int*      loc;     /* vreg -> physical register (part 0) or -1 */
    int*      slot;    /* vreg -> spill slot or -1 */
    int*      pidx;    /* constant vreg -> pool entry */
    int       npool;
    int       pool_bits; /* pool entry with lane bits 1, 2, 4, ... */
    int       nslot;
    int       save_off;  /* Win64: xmm6-15 saved in the frame */
    int       spill0;    /* frame offset of spill slot 0 */
    size_t*   opat;      /* code offset of each op */
    xfix*     fix;
    int       nfix, fixcap;
    int       rip;       /* pending rip relative fixup: position of the disp32 (-1 none) */
    uintptr_t ripaddr;
    int       failed;
    uint32_t  used;  /* vector registers the code names (conservative: opcode extension fields count) */
    uint32_t  save;  /* Win64: the callee saved xmm6-15 to keep */
    int       norec; /* prologue / epilogue: not recorded */
    int*      nuse;  /* vreg -> uses */
    int       kvec;  /* AVX-512: the compare result left in k1 only (its select follows), -1 none */
} xc;

/* ---- bytes ---- */

static void xb(xc* X, uint8_t b)
{
    if (X->n < X->cap) X->buf[X->n] = b;
    else X->failed = 1;
    X->n++;
}
static void xd(xc* X, uint32_t v)
{
    for (int i = 0; i < 4; i++) xb(X, (uint8_t)(v >> (8 * i)));
}
static void xq(xc* X, uint64_t v)
{
    xd(X, (uint32_t)v);
    xd(X, (uint32_t)(v >> 32));
}

/* the end of an instruction: resolve a rip relative operand */
static void xend(xc* X)
{
    if (X->rip >= 0) {
        int32_t rel = (int32_t)(X->ripaddr - (X->base + X->n));
        if ((size_t)X->rip + 4 <= X->cap) memcpy(X->buf + X->rip, &rel, 4);
        X->rip = -1;
    }
}

static void xmodrm(xc* X, int reg, xo rm)
{
    switch (rm.kind) {
    case XR: xb(X, (uint8_t)(0xC0 | ((reg & 7) << 3) | (rm.r & 7))); break;
    case XF: xb(X, (uint8_t)(0x80 | ((reg & 7) << 3) | 3)), xd(X, (uint32_t)rm.disp); break;
    case XA: xb(X, (uint8_t)(0x80 | ((reg & 7) << 3) | 0)), xd(X, (uint32_t)rm.disp); break;
    case XP:
        xb(X, (uint8_t)(0x00 | ((reg & 7) << 3) | 5));
#if FMJ_X64
        X->rip     = (int)X->n;
        X->ripaddr = rm.addr;
        xd(X, 0);
#else
        xd(X, (uint32_t)rm.addr);
#endif
        break;
    default: X->failed = 1; break;
    }
}

/* a vector instruction: reg = ModRM.reg, vv = VEX / EVEX vvvv (-1 none), rm;
 * len: 0 128, 1 256 (VEX), 2 512 (EVEX); aaa: EVEX opmask */
static void xvl(xc* X, xenc e, int len, int reg, int vv, xo rm, int imm, int is4, int aaa)
{
    int v = vv < 0 ? 0 : vv;
    int rmr = rm.kind == XR ? rm.r : 0;
    if (!X->norec) {
        X->used |= 1u << (reg & 31);
        if (vv >= 0) X->used |= 1u << (vv & 31);
        if (rm.kind == XR) X->used |= 1u << (rmr & 31);
        if (is4 >= 0) X->used |= 1u << (is4 & 15);
    }
    if (len == 2) {
        uint8_t p0 = (uint8_t)(((reg & 8) ? 0 : 0x80) | ((rm.kind == XR && (rmr & 16)) ? 0 : 0x40) | ((rmr & 8) ? 0 : 0x20) |
                               ((reg & 16) ? 0 : 0x10) | e.map);
        uint8_t p1 = (uint8_t)((e.w << 7) | ((~v & 15) << 3) | 4 | e.pp);
        uint8_t p2 = (uint8_t)((2 << 5) | ((v & 16) ? 0 : 8) | (aaa & 7));
        xb(X, 0x62), xb(X, p0), xb(X, p1), xb(X, p2);
    } else {
        uint8_t b1 = (uint8_t)(((reg & 8) ? 0 : 0x80) | 0x40 | ((rmr & 8) ? 0 : 0x20) | e.map);
        uint8_t b2 = (uint8_t)((e.w << 7) | ((~v & 15) << 3) | (len << 2) | e.pp);
        xb(X, 0xC4), xb(X, b1), xb(X, b2);
    }
    xb(X, e.op);
    xmodrm(X, reg, rm);
    if (is4 >= 0) xb(X, (uint8_t)(is4 << 4));
    else if (imm >= 0) xb(X, (uint8_t)imm);
    xend(X);
}

static void xv(xc* X, xenc e, int reg, int vv, xo rm, int imm) { xvl(X, e, X->e512 ? 2 : 1, reg, vv, rm, imm, -1, 0); }

static xo xreg(int r)
{
    xo o = { XR, r, 0, 0 };
    return o;
}
static xo xfr(int32_t disp)
{
    xo o = { XF, 0, disp, 0 };
    return o;
}

/* the encodings (map: 1 0F, 2 0F38, 3 0F3A; pp: 0 -, 1 66, 2 F3, 3 F2) */
static const xenc E_MOVUPS_L = { 1, 0, 0x10, 0 }, E_MOVUPS_S = { 1, 0, 0x11, 0 }, E_MOVAPS = { 1, 0, 0x28, 0 };
static const xenc E_ADDPS = { 1, 0, 0x58, 0 }, E_SUBPS = { 1, 0, 0x5C, 0 }, E_MULPS = { 1, 0, 0x59, 0 }, E_DIVPS = { 1, 0, 0x5E, 0 };
static const xenc E_SQRTPS = { 1, 0, 0x51, 0 }, E_CMPPS = { 1, 0, 0xC2, 0 };
static const xenc E_PAND = { 1, 1, 0xDB, 0 }, E_PANDN = { 1, 1, 0xDF, 0 }, E_POR = { 1, 1, 0xEB, 0 }, E_PXOR = { 1, 1, 0xEF, 0 };
static const xenc E_PADDD = { 1, 1, 0xFE, 0 }, E_PSUBD = { 1, 1, 0xFA, 0 }, E_PMULLD = { 2, 1, 0x40, 0 };
static const xenc E_PSLLVD = { 2, 1, 0x47, 0 }, E_PSRLVD = { 2, 1, 0x45, 0 }, E_PSRAVD = { 2, 1, 0x46, 0 };
static const xenc E_PSHIFTI = { 1, 1, 0x72, 0 }; /* /6 sll, /2 srl, /4 sra */
static const xenc E_PCMPEQD = { 1, 1, 0x76, 0 }, E_PCMPGTD = { 1, 1, 0x66, 0 };
static const xenc E_CVTDQ2PS = { 1, 0, 0x5B, 0 }, E_CVTTPS2DQ = { 1, 2, 0x5B, 0 }, E_CVTUDQ2PS = { 1, 3, 0x7A, 0 };
static const xenc E_BLENDVPS = { 3, 1, 0x4A, 0 }, E_PBLENDMD = { 2, 1, 0x64, 0 };
static const xenc E_PERMILPS = { 3, 1, 0x04, 0 }, E_SHUFF32X4 = { 3, 1, 0x23, 0 };
static const xenc E_BROADCASTSS = { 2, 1, 0x18, 0 }, E_PBROADCASTD = { 2, 1, 0x58, 0 };
static const xenc E_PMOVD2M = { 2, 2, 0x39, 0 }, E_PMOVM2D = { 2, 2, 0x38, 0 };
static const xenc E_MOVMSKPS = { 1, 0, 0x50, 0 }, E_KMOVW_LD = { 1, 0, 0x90, 0 }, E_KMOVW_TOGPR = { 1, 0, 0x93, 0 };

/* ---- operands ---- */

static xo xop(xc* X, int v, int part)
{
    xo o = { 0, 0, 0, 0 };
    if (X->p->vk[v] == FMJ_K_CONST) {
        o.kind = XP;
        o.addr = X->base + (size_t)64 * (size_t)X->pidx[v] + (size_t)(32 * part);
        return o;
    }
    if (X->loc[v] >= 0) return xreg(X->loc[v] + (X->parts == 2 ? part : 0));
    if (X->slot[v] < 0) X->slot[v] = X->nslot++; /* never allocated (a value only read): a slot */
    return xfr(X->spill0 + 64 * X->slot[v] + 32 * part);
}

static void xload(xc* X, int r, xo src)
{
    if (src.kind == XR) {
        if (src.r != r) xv(X, E_MOVAPS, r, -1, src, -1);
    } else {
        xv(X, E_MOVUPS_L, r, -1, src, -1);
    }
}
static void xstore(xc* X, xo dst, int r)
{
    if (dst.kind == XR) {
        if (dst.r != r) xv(X, E_MOVAPS, dst.r, -1, xreg(r), -1);
    } else {
        xv(X, E_MOVUPS_S, r, -1, dst, -1);
    }
}
/* the operand as a register (a temporary when it is in memory) */
static int xin(xc* X, xo o, int tmp)
{
    if (o.kind == XR) return o.r;
    xload(X, tmp, o);
    return tmp;
}

/* d = op(a, b): a in vvvv (a register), b the ModRM operand */
static void xbin(xc* X, xenc e, int d, int a, int b, int commut, int imm)
{
    for (int p = 0; p < X->parts; p++) {
        xo A = xop(X, a, p), B = xop(X, b, p), D = xop(X, d, p);
        if (A.kind != XR && commut && B.kind == XR) {
            xo t = A;
            A = B, B = t;
        }
        int ar = xin(X, A, X->t1);
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, e, dr, ar, B, imm);
        if (D.kind != XR) xstore(X, D, dr);
    }
}

/* AVX-512 compares into k1, then all ones / zero lanes */
static void xcmpk(xc* X, xenc e, int d, int a, int b, int imm, const fmj_op* next)
{
    xo  A = xop(X, a, 0), B = xop(X, b, 0), D = xop(X, d, 0);
    int ar = xin(X, A, X->t1);
    xv(X, e, 1, ar, B, imm); /* reg field: k1 */
    if (next && next->op == J_SEL && next->c == d && next->a != d && next->b != d && X->nuse[d] == 1) {
        X->kvec = d; /* the select right after is the only use: it reads k1 */
        return;
    }
    int dr = D.kind == XR ? D.r : X->t0;
    xv(X, E_PMOVM2D, dr, -1, xreg(1), -1);
    if (D.kind != XR) xstore(X, D, dr);
}

static void xun(xc* X, xenc e, int d, int a)
{
    for (int p = 0; p < X->parts; p++) {
        xo  A = xop(X, a, p), D = xop(X, d, p);
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, e, dr, -1, A, -1);
        if (D.kind != XR) xstore(X, D, dr);
    }
}

static void xshifti(xc* X, int digit, int d, int a, int n)
{
    for (int p = 0; p < X->parts; p++) {
        xo  A = xop(X, a, p), D = xop(X, d, p);
        int ar = xin(X, A, X->t1);
        int dr = D.kind == XR ? D.r : X->t0;
        uint32_t u = X->used;
        xv(X, E_PSHIFTI, digit, dr, xreg(ar), n);
        X->used = u | (1u << dr) | (1u << ar); /* the reg field is the opcode extension, not a register */
        if (D.kind != XR) xstore(X, D, dr);
    }
}

static void xmov(xc* X, int d, int a)
{
    for (int p = 0; p < X->parts; p++) {
        xo A = xop(X, a, p), D = xop(X, d, p);
        if (A.kind == XR && D.kind == XR && A.r == D.r) continue;
        if (A.kind == XF && D.kind == XF && A.disp == D.disp) continue;
        if (D.kind == XR) xload(X, D.r, A);
        else xstore(X, D, xin(X, A, X->t0));
    }
}

/* ---- general purpose ---- */

static int32_t xm(xc* X, int slot) { return X->p->off_m + 4 * slot; }

/* op r32, [rbx + disp] (op: 8B mov, 0B or, 23 and) */
static void xg_rm(xc* X, uint8_t op, int r, int32_t disp)
{
    xb(X, op);
    xmodrm(X, r, xfr(disp));
}
static void xg_store(xc* X, int r, int32_t disp)
{
    xb(X, 0x89);
    xmodrm(X, r, xfr(disp));
}
static void xg_movimm(xc* X, int32_t disp, uint32_t imm)
{
    xb(X, 0xC7);
    xmodrm(X, 0, xfr(disp));
    xd(X, imm);
}

static void xjump_rel(xc* X, int target)
{
    if (X->nfix == X->fixcap) {
        int   n  = X->fixcap ? X->fixcap * 2 : 64;
        xfix* nf = (xfix*)realloc(X->fix, (size_t)n * sizeof(xfix));
        if (!nf) {
            X->failed = 1;
            return;
        }
        X->fix = nf, X->fixcap = n;
    }
    X->fix[X->nfix].at     = X->n;
    X->fix[X->nfix].target = target;
    X->nfix++;
    xd(X, 0);
}
static void xjz(xc* X, int target) { xb(X, 0x0F), xb(X, 0x84), xjump_rel(X, target); }
static void xjae(xc* X, int target) { xb(X, 0x0F), xb(X, 0x83), xjump_rel(X, target); }
static void xjmp(xc* X, int target) { xb(X, 0xE9), xjump_rel(X, target); }

/* lanes of vreg v whose sign bit is set -> eax */
static void xbits(xc* X, int v)
{
    if (X->e512) {
        int r = xin(X, xop(X, v, 0), X->t0);
        xv(X, E_PMOVD2M, 1, -1, xreg(r), -1);           /* k1 */
        xvl(X, E_KMOVW_TOGPR, 0, 0, -1, xreg(1), -1, -1, 0); /* kmovw eax, k1 */
        return;
    }
    int r0 = xin(X, xop(X, v, 0), X->t0);
    xvl(X, E_MOVMSKPS, 1, 0, -1, xreg(r0), -1, -1, 0); /* eax */
    int r1 = xin(X, xop(X, v, 1), X->t0);
    xvl(X, E_MOVMSKPS, 1, 1, -1, xreg(r1), -1, -1, 0); /* ecx */
    xb(X, 0xC1), xb(X, 0xE1), xb(X, 8);                /* shl ecx, 8 */
    xb(X, 0x09), xb(X, 0xC8);                          /* or eax, ecx */
}

/* ---- ops ---- */

static void xsel(xc* X, int d, int a, int b, int c)
{
    if (X->e512) {
        xo  C = xop(X, c, 0), A = xop(X, a, 0), B = xop(X, b, 0), D = xop(X, d, 0);
        if (X->kvec != c) {
            int cr = xin(X, C, X->t0);
            xv(X, E_PMOVD2M, 1, -1, xreg(cr), -1); /* k1 = c */
        }
        X->kvec = -1;
        int br = xin(X, B, X->t1);
        int dr = D.kind == XR ? D.r : X->t0;
        xvl(X, E_PBLENDMD, 2, dr, br, A, -1, -1, 1); /* dr{k1} = k1 ? a : b */
        if (D.kind != XR) xstore(X, D, dr);
        return;
    }
    for (int p = 0; p < 2; p++) { /* vblendvps reads every source before it writes */
        xo  C = xop(X, c, p), A = xop(X, a, p), B = xop(X, b, p), D = xop(X, d, p);
        int br = xin(X, B, X->t1);
        int cr = xin(X, C, X->t0);
        int dr = D.kind == XR ? D.r : X->t0;
        xvl(X, E_BLENDVPS, 1, dr, br, A, -1, cr, 0); /* dr = c ? a : b */
        if (D.kind != XR) xstore(X, D, dr);
    }
}

static void xddx(xc* X, int d, int a)
{
    for (int p = 0; p < X->parts; p++) {
        xo  A = xop(X, a, p), D = xop(X, d, p);
        xv(X, E_PERMILPS, X->t0, -1, A, 0xF5); /* v[l | 1] */
        xv(X, E_PERMILPS, X->t1, -1, A, 0xA0); /* v[l & ~1] */
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, E_SUBPS, dr, X->t0, xreg(X->t1), -1);
        if (D.kind != XR) xstore(X, D, dr);
    }
}

static void xddy(xc* X, int d, int a)
{
    if (X->e512) {
        int ar = xin(X, xop(X, a, 0), X->t1);
        xv(X, E_SHUFF32X4, X->t0, ar, xreg(ar), 0xEE); /* rows: v[l | 8] */
        xv(X, E_SHUFF32X4, X->t1, ar, xreg(ar), 0x44); /* v[l & ~8] */
        xo  D  = xop(X, d, 0);
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, E_SUBPS, dr, X->t0, xreg(X->t1), -1);
        if (D.kind != XR) xstore(X, D, dr);
        return;
    }
    int hr = xin(X, xop(X, a, 1), X->t1); /* row 1 - row 0, in both halves */
    xv(X, E_SUBPS, X->t0, hr, xop(X, a, 0), -1);
    for (int p = 0; p < 2; p++) xstore(X, xop(X, d, p), X->t0);
}

static void xcvtuf(xc* X, int d, int a)
{
    if (X->e512) {
        xun(X, E_CVTUDQ2PS, d, a);
        return;
    }
    int k65536 = X->p->nv, k0xffff = X->p->nv + 1; /* pool entries */
    for (int p = 0; p < 2; p++) { /* (float)(u >> 16) * 65536 + (float)(u & 0xffff): both exact, one rounding */
        xo  A  = xop(X, a, p), D = xop(X, d, p);
        int ar = xin(X, A, X->t1);
        xv(X, E_PSHIFTI, 2, X->t0, xreg(ar), 16);
        xv(X, E_CVTDQ2PS, X->t0, -1, xreg(X->t0), -1);
        xv(X, E_MULPS, X->t0, X->t0, xop(X, k65536, p), -1);
        xv(X, E_PAND, X->t1, ar, xop(X, k0xffff, p), -1);
        xv(X, E_CVTDQ2PS, X->t1, -1, xreg(X->t1), -1);
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, E_ADDPS, dr, X->t0, xreg(X->t1), -1);
        if (D.kind != XR) xstore(X, D, dr);
    }
}

static void xmaskv(xc* X, int d, int slot)
{
    if (X->e512) {
        xvl(X, E_KMOVW_LD, 0, 1, -1, xfr(xm(X, slot)), -1, -1, 0); /* kmovw k1, [M + slot] */
        xo  D  = xop(X, d, 0);
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, E_PMOVM2D, dr, -1, xreg(1), -1);
        if (D.kind != XR) xstore(X, D, dr);
        return;
    }
    for (int p = 0; p < 2; p++) {
        xo bits = { XP, 0, 0, X->base + (size_t)64 * (size_t)X->pool_bits + (size_t)(32 * p) };
        xv(X, E_PBROADCASTD, X->t0, -1, xfr(xm(X, slot)), -1);
        xv(X, E_PAND, X->t0, X->t0, bits, -1);
        xv(X, E_PCMPEQD, X->t0, X->t0, bits, -1);
        xstore(X, xop(X, d, p), X->t0);
    }
}

static void xldu(xc* X, int d, int blk, uint32_t off)
{
#if FMJ_X64
    xb(X, 0x48);
#endif
    xg_rm(X, 0x8B, 0, (int32_t)(offsetof(fmj_frame, ubo) + sizeof(void*) * (size_t)blk)); /* rax = ubo[blk] */
    xo src = { XA, 0, (int32_t)off, 0 };
    for (int p = 0; p < X->parts; p++) {
        xo  D  = xop(X, d, p);
        int dr = D.kind == XR ? D.r : X->t0;
        xv(X, E_BROADCASTSS, dr, -1, src, -1);
        if (D.kind != XR) xstore(X, D, dr);
    }
}

static void xcall(xc* X, const fmj_op* o)
{
    uintptr_t fn   = (uintptr_t)fmj_helpers[o->imm];
    uintptr_t desc = (uintptr_t)&X->p->calls[o->a];
    xb(X, 0xC5), xb(X, 0xF8), xb(X, 0x77); /* vzeroupper */
#if FMJ_X64
#  if FMJ_WIN64
    xb(X, 0x48), xb(X, 0x89), xb(X, 0xD9); /* mov rcx, rbx */
    xb(X, 0x48), xb(X, 0xBA), xq(X, desc); /* mov rdx, desc */
#  else
    xb(X, 0x48), xb(X, 0x89), xb(X, 0xDF); /* mov rdi, rbx */
    xb(X, 0x48), xb(X, 0xBE), xq(X, desc); /* mov rsi, desc */
#  endif
    xb(X, 0x48), xb(X, 0xB8), xq(X, fn); /* mov rax, fn */
    xb(X, 0xFF), xb(X, 0xD0);            /* call rax */
#else
    xb(X, 0xC7), xb(X, 0x44), xb(X, 0x24), xb(X, 0x04), xd(X, (uint32_t)desc); /* mov [esp + 4], desc */
    xb(X, 0x89), xb(X, 0x1C), xb(X, 0x24);                                      /* mov [esp], ebx */
    xb(X, 0xB8), xd(X, (uint32_t)fn);                                            /* mov eax, fn */
    xb(X, 0xFF), xb(X, 0xD0);                                                    /* call eax */
#endif
}

static void xprologue(xc* X)
{
#if FMJ_X64
    xb(X, 0x53); /* push rbx */
#  if FMJ_WIN64
    xb(X, 0x48), xb(X, 0x83), xb(X, 0xEC), xb(X, 0x20); /* sub rsp, 32 (shadow space; rsp stays 16 aligned) */
    xb(X, 0x48), xb(X, 0x89), xb(X, 0xCB);              /* mov rbx, rcx */
    X->norec = 1;
    for (int i = 6; i < 16; i++)
        if (X->save >> i & 1) xvl(X, E_MOVUPS_S, 0, i, -1, xfr(X->save_off + 16 * (i - 6)), -1, -1, 0);
    X->norec = 0;
#  else
    xb(X, 0x48), xb(X, 0x89), xb(X, 0xFB); /* mov rbx, rdi */
#  endif
#else
    xb(X, 0x55);                                         /* push ebp */
    xb(X, 0x89), xb(X, 0xE5);                            /* mov ebp, esp */
    xb(X, 0x53);                                         /* push ebx */
    xb(X, 0x8B), xb(X, 0x5D), xb(X, 0x08);               /* mov ebx, [ebp + 8] */
    xb(X, 0x83), xb(X, 0xE4), xb(X, 0xF0);               /* and esp, -16 */
    xb(X, 0x83), xb(X, 0xEC), xb(X, 0x10);               /* sub esp, 16 (helper arguments) */
#endif
}

static void xepilogue(xc* X)
{
#if FMJ_X64 && FMJ_WIN64
    X->norec = 1;
    for (int i = 6; i < 16; i++)
        if (X->save >> i & 1) xvl(X, E_MOVUPS_L, 0, i, -1, xfr(X->save_off + 16 * (i - 6)), -1, -1, 0);
    X->norec = 0;
#endif
    xb(X, 0xC5), xb(X, 0xF8), xb(X, 0x77); /* vzeroupper */
#if FMJ_X64
#  if FMJ_WIN64
    xb(X, 0x48), xb(X, 0x83), xb(X, 0xC4), xb(X, 0x20); /* add rsp, 32 */
#  endif
    xb(X, 0x5B); /* pop rbx */
#else
    xb(X, 0x8B), xb(X, 0x5D), xb(X, 0xFC); /* mov ebx, [ebp - 4] */
    xb(X, 0x89), xb(X, 0xEC);              /* mov esp, ebp */
    xb(X, 0x5D);                           /* pop ebp */
#endif
    xb(X, 0xC3);
}

/* ---- register allocation ---- */

static int xuses(const fmj_op* o, int* u)
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

typedef struct xiv {
    int v, s, e;
} xiv;

static int xiv_cmp(const void* a, const void* b) { return ((const xiv*)a)->s - ((const xiv*)b)->s; }

static int xalloc(xc* X, int nreg, const int* regs)
{
    fmj_prog* p  = X->p;
    int       nv = p->nv, n = p->nops;
    int*      s  = (int*)malloc((size_t)nv * sizeof(int));
    int*      e  = (int*)malloc((size_t)nv * sizeof(int));
    int*      fd = (int*)malloc((size_t)nv * sizeof(int)); /* first def / use inside the loop being checked */
    int*      fu = (int*)malloc((size_t)nv * sizeof(int));
    int*      spill = (int*)calloc((size_t)nv, sizeof(int));
    xiv*      iv = (xiv*)malloc((size_t)nv * sizeof(xiv));
    int       ok = s && e && fd && fu && spill && iv;
    if (!ok) goto done;
    for (int v = 0; v < nv; v++) s[v] = 0x7fffffff, e[v] = -1, X->loc[v] = -1, X->slot[v] = -1;
    for (int i = 0; i < n; i++) {
        const fmj_op* o = &p->ops[i];
        int           u[3], nu = xuses(o, u);
        for (int k = 0; k < nu; k++)
            if (u[k] >= 0) s[u[k]] = s[u[k]] < i ? s[u[k]] : i, e[u[k]] = e[u[k]] > i ? e[u[k]] : i;
        if (o->d >= 0) s[o->d] = s[o->d] < i ? s[o->d] : i, e[o->d] = e[o->d] > i ? e[o->d] : i;
    }
    /* loops: live around the back edge -> the whole loop */
    for (int round = 0; round < 32; round++) {
        int changed = 0;
        for (int L = 0; L < n; L++) {
            if (p->ops[L].op != J_LOOP) continue;
            int E = p->ops[L].c;
            for (int v = 0; v < nv; v++) fd[v] = fu[v] = 0x7fffffff;
            for (int i = L; i <= E; i++) {
                const fmj_op* o = &p->ops[i];
                int           u[3], nu = xuses(o, u);
                for (int k = 0; k < nu; k++)
                    if (u[k] >= 0 && fu[u[k]] > i) fu[u[k]] = i;
                if (o->d >= 0 && fd[o->d] > i) fd[o->d] = i;
            }
            for (int v = 0; v < nv; v++) {
                if (e[v] < L || s[v] > E || (s[v] <= L && e[v] >= E)) continue; /* outside, or already over the loop */
                int carried = fu[v] != 0x7fffffff && fu[v] <= fd[v];          /* read before (re)defined in an iteration */
                if (s[v] < L || e[v] > E || carried) {
                    s[v] = s[v] < L ? s[v] : L, e[v] = e[v] > E ? e[v] : E;
                    changed = 1;
                }
            }
        }
        if (!changed) break;
    }
    /* ranges over a helper call: in the frame */
    for (int i = 0; i < n; i++) {
        if (p->ops[i].op != J_CALL) continue;
        for (int v = 0; v < nv; v++)
            if (s[v] < i && e[v] > i) spill[v] = 1;
    }
    int ni = 0;
    for (int v = 0; v < nv; v++) {
        if (p->vk[v] == FMJ_K_CONST || e[v] < 0) continue;
        if (spill[v]) {
            X->slot[v] = X->nslot++;
            continue;
        }
        iv[ni].v = v, iv[ni].s = s[v], iv[ni].e = e[v];
        ni++;
    }
    qsort(iv, (size_t)ni, sizeof(xiv), xiv_cmp);
    {
        int  act[64], nact = 0; /* active intervals (indices into iv), by end */
        int  freer[64], nfree = 0;
        for (int k = 0; k < nreg; k++) freer[nfree++] = regs[nreg - 1 - k];
        for (int k = 0; k < ni; k++) {
            int st = iv[k].s;
            for (int j = 0; j < nact;) { /* expire: ranges that ended (a source's last use can hand its register to the result) */
                if (iv[act[j]].e <= st && !(iv[act[j]].e == st && iv[act[j]].s == st)) {
                    freer[nfree++] = X->loc[iv[act[j]].v];
                    act[j]         = act[--nact];
                } else {
                    j++;
                }
            }
            if (nfree) {
                X->loc[iv[k].v] = freer[--nfree];
                act[nact++]     = k;
                continue;
            }
            int far = -1; /* spill the furthest ending of the active and this one */
            for (int j = 0; j < nact; j++)
                if (far < 0 || iv[act[j]].e > iv[act[far]].e) far = j;
            if (far >= 0 && iv[act[far]].e > iv[k].e) {
                int sv          = iv[act[far]].v;
                X->loc[iv[k].v] = X->loc[sv];
                X->loc[sv]      = -1;
                X->slot[sv]     = X->nslot++;
                act[far]        = k;
            } else {
                X->slot[iv[k].v] = X->nslot++;
            }
        }
    }
done:
    free(s), free(e), free(fd), free(fu), free(spill), free(iv);
    return ok;
}

/* ---- the program ---- */

static int xprogram(xc* X)
{
    fmj_prog* p = X->p;
    xprologue(X);
    for (int i = 0; i < p->nops && !X->failed; i++) {
        const fmj_op* o = &p->ops[i];
        X->opat[i]      = X->n;
        switch (o->op) {
        case J_NOP: case J_CONST: break;
        case J_MOV: xmov(X, o->d, o->a); break;
        case J_FADD: xbin(X, E_ADDPS, o->d, o->a, o->b, 1, -1); break;
        case J_FSUB: xbin(X, E_SUBPS, o->d, o->a, o->b, 0, -1); break;
        case J_FMUL: xbin(X, E_MULPS, o->d, o->a, o->b, 1, -1); break;
        case J_FDIV: xbin(X, E_DIVPS, o->d, o->a, o->b, 0, -1); break;
        case J_FSQRT: xun(X, E_SQRTPS, o->d, o->a); break;
        case J_FCMP:
            if (X->e512) xcmpk(X, E_CMPPS, o->d, o->a, o->b, (int)o->imm, i + 1 < p->nops ? o + 1 : NULL);
            else xbin(X, E_CMPPS, o->d, o->a, o->b, 0, (int)o->imm);
            break;
        case J_IADD: xbin(X, E_PADDD, o->d, o->a, o->b, 1, -1); break;
        case J_ISUB: xbin(X, E_PSUBD, o->d, o->a, o->b, 0, -1); break;
        case J_IMUL: xbin(X, E_PMULLD, o->d, o->a, o->b, 1, -1); break;
        case J_AND: xbin(X, E_PAND, o->d, o->a, o->b, 1, -1); break;
        case J_OR: xbin(X, E_POR, o->d, o->a, o->b, 1, -1); break;
        case J_XOR: xbin(X, E_PXOR, o->d, o->a, o->b, 1, -1); break;
        case J_ANDN: xbin(X, E_PANDN, o->d, o->a, o->b, 0, -1); break;
        case J_SHL: xbin(X, E_PSLLVD, o->d, o->a, o->b, 0, -1); break;
        case J_SHR: xbin(X, E_PSRLVD, o->d, o->a, o->b, 0, -1); break;
        case J_SAR: xbin(X, E_PSRAVD, o->d, o->a, o->b, 0, -1); break;
        case J_SHLI: xshifti(X, 6, o->d, o->a, (int)o->imm); break;
        case J_SHRI: xshifti(X, 2, o->d, o->a, (int)o->imm); break;
        case J_SARI: xshifti(X, 4, o->d, o->a, (int)o->imm); break;
        case J_ICMPEQ:
            if (X->e512) xcmpk(X, E_PCMPEQD, o->d, o->a, o->b, -1, i + 1 < p->nops ? o + 1 : NULL);
            else xbin(X, E_PCMPEQD, o->d, o->a, o->b, 1, -1);
            break;
        case J_ICMPGT:
            if (X->e512) xcmpk(X, E_PCMPGTD, o->d, o->a, o->b, -1, i + 1 < p->nops ? o + 1 : NULL);
            else xbin(X, E_PCMPGTD, o->d, o->a, o->b, 0, -1);
            break;
        case J_CVTIF: xun(X, E_CVTDQ2PS, o->d, o->a); break;
        case J_CVTFI: xun(X, E_CVTTPS2DQ, o->d, o->a); break;
        case J_CVTUF: xcvtuf(X, o->d, o->a); break;
        case J_SEL: xsel(X, o->d, o->a, o->b, o->c); break;
        case J_DDX: xddx(X, o->d, o->a); break;
        case J_DDY: xddy(X, o->d, o->a); break;
        case J_LDF:
            for (int pp = 0; pp < X->parts; pp++) {
                xo D = xop(X, o->d, pp);
                if (D.kind == XR) xload(X, D.r, xfr((int32_t)o->imm + 32 * pp));
                else xload(X, X->t0, xfr((int32_t)o->imm + 32 * pp)), xstore(X, D, X->t0);
            }
            break;
        case J_STF:
            for (int pp = 0; pp < X->parts; pp++) xstore(X, xfr((int32_t)o->imm + 32 * pp), xin(X, xop(X, o->a, pp), X->t0));
            break;
        case J_LDU: xldu(X, o->d, o->b, o->imm); break;
        case J_MASKV: xmaskv(X, o->d, (int)o->imm); break;
        case J_MZERO: xg_movimm(X, xm(X, o->a), 0); break;
        case J_MCOPY: xg_rm(X, 0x8B, 0, xm(X, o->b)), xg_store(X, 0, xm(X, o->a)); break;
        case J_MOR: xg_rm(X, 0x8B, 0, xm(X, o->b)), xg_rm(X, 0x0B, 0, xm(X, o->c)), xg_store(X, 0, xm(X, o->a)); break;
        case J_MAND: xg_rm(X, 0x8B, 0, xm(X, o->b)), xg_rm(X, 0x23, 0, xm(X, o->c)), xg_store(X, 0, xm(X, o->a)); break;
        case J_MANDN:
            xg_rm(X, 0x8B, 0, xm(X, o->c));
            xb(X, 0xF7), xb(X, 0xD0); /* not eax */
            xg_rm(X, 0x23, 0, xm(X, o->b));
            xg_store(X, 0, xm(X, o->a));
            break;
        case J_MCOND:
            xbits(X, o->b);
            xg_rm(X, 0x23, 0, xm(X, o->c));
            xg_store(X, 0, xm(X, o->a));
            break;
        case J_IF:
            xg_rm(X, 0x8B, 0, xm(X, o->a));
            xb(X, 0x85), xb(X, 0xC0); /* test eax, eax */
            xjz(X, o->c + 1);
            break;
        case J_ENDIF: break;
        case J_LOOP: xg_movimm(X, xm(X, o->a), 0); break;
        case J_BREAKZ:
            xg_rm(X, 0x8B, 0, xm(X, o->a));
            xb(X, 0x85), xb(X, 0xC0);
            xjz(X, o->c + 1);
            xg_rm(X, 0x8B, 1, xm(X, o->b));                 /* ecx = counter */
            xb(X, 0x8D), xb(X, 0x51), xb(X, 0x01);          /* lea edx, [rcx + 1] */
            xg_store(X, 2, xm(X, o->b));
            xb(X, 0x81), xb(X, 0xF9), xd(X, SV_MAXITER);    /* cmp ecx, MAXITER */
            xjae(X, o->c + 1);
            break;
        case J_BREAK: xjmp(X, o->c + 1); break;
        case J_ENDLOOP: xjmp(X, o->c + 1); break;
        case J_CALL: xcall(X, o); break;
        default: X->failed = 1; break;
        }
    }
    X->opat[p->nops] = X->n;
    xepilogue(X);
    for (int k = 0; k < X->nfix; k++) { /* jumps: to the code of op `target` */
        int32_t rel = (int32_t)(X->opat[X->fix[k].target] - (X->fix[k].at + 4));
        if (X->fix[k].at + 4 <= X->cap) memcpy(X->buf + X->fix[k].at, &rel, 4);
    }
    return !X->failed;
}

int fmj_compile_x86(fmj_prog* p, char* err, size_t errn)
{
    fm_simd_level lvl = fm_simd_current();
    if (lvl != FM_SIMD_AVX2 && lvl != FM_SIMD_AVX512) {
        if (err && errn) snprintf(err, errn, "no machine code for this CPU (needs AVX2)");
        return 0;
    }
    xc X;
    memset(&X, 0, sizeof(X));
    X.p     = p;
    X.e512  = lvl == FM_SIMD_AVX512;
    X.parts = X.e512 ? 1 : 2;
    X.rip   = -1;
    int regs[32], nreg = 0;
#if FMJ_X64
    if (X.e512) { /* the volatile registers first: Win64 saves xmm6-15 only when they are used */
        for (int r = 0; r < 6; r++) regs[nreg++] = r;
        for (int r = 16; r < 30; r++) regs[nreg++] = r;
        for (int r = 6; r < 16; r++) regs[nreg++] = r;
        X.t0 = 30, X.t1 = 31;
    } else {
        for (int r = 0; r < 14; r += 2) regs[nreg++] = r;
        X.t0 = 14, X.t1 = 15;
    }
#else
    if (X.e512) {
        for (int r = 0; r < 6; r++) regs[nreg++] = r;
    } else {
        for (int r = 0; r < 6; r += 2) regs[nreg++] = r;
    }
    X.t0 = 6, X.t1 = 7;
#endif
    /* constants: the pool (64 bytes per value), plus the lane bit table and the AVX2 unsigned convert's two */
    int nvx  = p->nv + 2;
    X.loc    = (int*)malloc((size_t)nvx * sizeof(int));
    X.slot   = (int*)malloc((size_t)nvx * sizeof(int));
    X.pidx   = (int*)malloc((size_t)nvx * sizeof(int));
    X.opat   = (size_t*)malloc(((size_t)p->nops + 1) * sizeof(size_t));
    if (!X.loc || !X.slot || !X.pidx || !X.opat) goto fail;
    for (int v = 0; v < nvx; v++) X.pidx[v] = -1;
    for (int v = 0; v < p->nv; v++)
        if (p->vk[v] == FMJ_K_CONST) X.pidx[v] = X.npool++;
    X.pool_bits = X.npool++;
    if (!xalloc(&X, nreg, regs)) goto fail;
    X.nuse = (int*)calloc((size_t)nvx, sizeof(int));
    if (!X.nuse) goto fail;
    for (int i = 0; i < p->nops; i++) {
        int u[3], nu = xuses(&p->ops[i], u);
        for (int k = 0; k < nu; k++)
            if (u[k] >= 0) X.nuse[u[k]]++;
    }
    X.kvec = -1;
    /* the frame: the Win64 register save area, then the spill slots (sized after the code) */
    X.save_off = p->off_spill;
    X.spill0   = p->off_spill + 192;
    {
        size_t pool = (size_t)64 * (size_t)(X.npool + 2);
        X.cap       = pool + (size_t)p->nops * (size_t)(X.e512 ? 48 : 96) + 4096;
        X.buf       = (uint8_t*)fmj_code_alloc(X.cap);
        if (!X.buf) goto fail;
        X.base = (uintptr_t)X.buf;
        for (int v = 0; v < p->nv; v++)
            if (X.pidx[v] >= 0)
                for (int l = 0; l < 16; l++) memcpy(X.buf + 64 * (size_t)X.pidx[v] + 4 * (size_t)l, &p->vc[v], 4);
        for (int l = 0; l < 16; l++) {
            uint32_t b = 1u << l;
            memcpy(X.buf + 64 * (size_t)X.pool_bits + 4 * (size_t)l, &b, 4);
        }
        /* the AVX2 unsigned convert's constants as vregs nv (65536.0f) and nv + 1 (0xffff) */
        X.pidx[p->nv]     = X.npool++;
        X.pidx[p->nv + 1] = X.npool++;
        for (int l = 0; l < 16; l++) {
            float    f = 65536.0f;
            uint32_t m = 0xffffu;
            memcpy(X.buf + 64 * (size_t)X.pidx[p->nv] + 4 * (size_t)l, &f, 4);
            memcpy(X.buf + 64 * (size_t)X.pidx[p->nv + 1] + 4 * (size_t)l, &m, 4);
        }
        X.n = (size_t)64 * (size_t)X.npool;
    }
    size_t entry = X.n;
    {
        /* vk of the two extra pool vregs reads as constants */
        uint8_t*  vk = (uint8_t*)realloc(p->vk, (size_t)p->nv + 2);
        if (!vk) goto fail;
        p->vk          = vk;
        p->vk[p->nv]   = FMJ_K_CONST;
        p->vk[p->nv + 1] = FMJ_K_CONST;
    }
    X.save = 0xFFC0u; /* first pass: every callee saved register, recording the ones the code names */
    X.used = 0;
    if (!xprogram(&X) || X.n > X.cap) {
        if (err && errn) snprintf(err, errn, "machine code generation failed");
        goto fail;
    }
    if ((X.used & 0xFFC0u) != X.save) { /* again with only those saved (the same code otherwise) */
        X.save = X.used & 0xFFC0u;
        X.n    = entry;
        X.nfix = 0;
        X.rip  = -1;
        X.kvec = -1;
        if (!xprogram(&X) || X.n > X.cap) {
            if (err && errn) snprintf(err, errn, "machine code generation failed");
            goto fail;
        }
    }
    if (!fmj_code_seal(X.buf, X.cap)) {
        if (err && errn) snprintf(err, errn, "cannot make the code executable");
        goto fail;
    }
    {   /* FMJ_DUMP=<file>: append the code (objdump -D -b binary -m i386:x86-64) */
        const char* dump = getenv("FMJ_DUMP");
        FILE*       df   = dump ? fopen(dump, "ab") : NULL;
        if (df) {
            fwrite(X.buf + entry, 1, X.n - entry, df);
            fclose(df);
        }
    }
    p->nspill     = X.nslot;
    p->frame_size = X.spill0 + 64 * X.nslot + 64;
    p->code      = X.buf;
    p->code_cap  = X.cap;
    p->code_size = X.n;
    p->fn        = (void (*)(void*))(void*)(X.buf + entry);
    free(X.loc), free(X.slot), free(X.pidx), free(X.opat), free(X.fix), free(X.nuse);
    return 1;
fail:
    if (X.buf) fmj_code_free(X.buf, X.cap);
    free(X.loc), free(X.slot), free(X.pidx), free(X.opat), free(X.fix), free(X.nuse);
    if (err && errn && !err[0]) snprintf(err, errn, "out of memory");
    return 0;
}

#else /* not x86 */

int fmj_compile_x86(fmj_prog* p, char* err, size_t errn)
{
    (void)p;
    if (err && errn) snprintf(err, errn, "no machine code for this CPU");
    return 0;
}

#endif
#endif
