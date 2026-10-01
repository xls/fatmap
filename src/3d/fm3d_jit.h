/*
 * fatmap - SPIR-V JIT internals (FM_FEATURE_JIT).
 *
 * The lowered SPIR-V program (fm3d_spirv_internal.h) is translated to a
 * register machine program over 16 lane vectors (fmj_prog), which runs one
 * 16 lane group at a time: a fragment quad group (4 whole 2x2 quads) or 16
 * vertices. Every SPIR-V vector / matrix / struct is split into its 32 bit
 * components, so copies, constructs, extracts and shuffles are renames,
 * unused components disappear and function variables become registers.
 *
 *   sv_stage -> fmj_build (fm3d_jit_ir.c) -> fmj_prog
 *            -> fmj_run_ref (portable reference executor, tests)
 *            -> fmj_compile_x86 (fm3d_jit_x86.c: machine code)
 *
 * Booleans are all ones / zero words (compares produce them, selects take
 * them). Math is the interpreter's formulas in the same order, the fast
 * GLSL.std.450 functions expanded from fm_vmath.h's straight line code, so
 * every result has the interpreter's bits. Texture sampling, precise math
 * and rare operations call C helpers through the frame.
 *
 * Register machine values ("vregs") are not SSA: a value defined in a
 * duplicated block, a variable or a phi has several definitions; uses read
 * the latest. The control flow keeps the lowered program's shape (masks in
 * frame slots, IF / LOOP markers).
 */
#ifndef FM3D_JIT_H
#define FM3D_JIT_H
#include "fm3d_spirv_internal.h"

#ifndef FM_FEATURE_JIT
#  define FM_FEATURE_JIT 0
#endif

#if FM_FEATURE_SPIRV && FM_FEATURE_JIT

#define FMJ_V 16 /* lanes of a group */
#define FMJ_PLANES FM3D_MAX_UNIFORM_BLOCKS /* the pseudo uniform block of the varying planes */

enum {
    J_NOP = 0,
    /* d = imm in every lane (vreg of class FMJ_K_CONST: the constant pool) */
    J_CONST,
    J_MOV, /* d = a */
    /* float */
    J_FADD, J_FSUB, J_FMUL, J_FDIV, J_FSQRT,
    J_FCMP, /* d = a <imm: x86 vcmpps predicate> b ? ~0 : 0 */
    /* integer / bits */
    J_IADD, J_ISUB, J_IMUL, J_AND, J_OR, J_XOR,
    J_ANDN,  /* d = ~a & b */
    J_SHL, J_SHR, J_SAR,    /* by b (each lane's count, already & 31) */
    J_SHLI, J_SHRI, J_SARI, /* by imm */
    J_ICMPEQ, J_ICMPGT,     /* signed: d = a > b ? ~0 : 0 */
    J_CVTIF,  /* (float)(int32_t)a */
    J_CVTUF,  /* (float)(uint32_t)a */
    J_CVTFI,  /* (int32_t)a truncated (x86: 0x80000000 out of range) */
    J_SEL,    /* d = c ? a : b (c: all ones / zero per lane) */
    /* lanes: quad group order (x neighbour lane ^ 1, y neighbour lane ^ 8) */
    J_DDX, J_DDY,
    /* frame / uniform memory */
    J_LDF,   /* d = 16 words at frame offset imm */
    J_STF,   /* 16 words of a at frame offset imm */
    J_LDU,   /* d = the word at byte imm of uniform block b (padded) in every lane */
    /* lane masks: 16 bit words in frame mask slots */
    J_MASKV, /* d = per lane all ones / zero of mask slot imm */
    J_MZERO, /* M[a] = 0 */
    J_MCOPY, /* M[a] = M[b] */
    J_MOR,   /* M[a] = M[b] | M[c] */
    J_MAND,  /* M[a] = M[b] & M[c] */
    J_MANDN, /* M[a] = M[b] & ~M[c] */
    J_MCOND, /* M[a] = lanes where vreg b is nonzero (sign bit of an all ones bool), & M[c] */
    /* control flow (the lowered program's) */
    J_IF,      /* skip to the matching J_ENDIF when M[a] == 0 */
    J_ENDIF,
    J_LOOP,    /* M[a] = 0 (iteration counter) */
    J_BREAKZ,  /* leave the loop when M[a] == 0 or ++M[b] > SV_MAXITER */
    J_BREAK,
    J_ENDLOOP,
    /* a C helper: fmj_helpers[imm](frame, &prog->calls[a]); arguments and results in frame slots */
    J_CALL,
    J_COUNT
};

/* vreg classes */
enum { FMJ_K_VAL = 0, FMJ_K_CONST };

typedef struct fmj_op {
    uint16_t op;
    uint16_t flags;
    int      d, a, b, c;
    uint32_t imm;
} fmj_op;

/* helpers */
enum { FMJ_H_SAMPLE = 0, FMJ_H_MATH, FMJ_H_SLOW, FMJ_H_ULOAD, FMJ_H_COUNT };

typedef struct fmj_call {
    int kind;    /* FMJ_H_* */
    int fn;      /* MATH: GLSL.std.450 number; SLOW: SPIR-V opcode; SAMPLE: SPIR-V opcode */
    int n;       /* components of the result (per lane) */
    int unit, dim, arrayed, ncoord, lod; /* SAMPLE: texture unit, image dim, coordinate comps, a Lod operand */
    int binding, type, mstride, poff;    /* ULOAD: block, value type, matrix stride, constant byte offset */
    int arg;     /* frame offset of the first argument array (16 words each, consecutive) */
    int nargs;
    int res;     /* frame offset of the result arrays */
    const sv_stage* s;
} fmj_call;

typedef struct fmj_prog {
    fmj_op*   ops;
    int       nops, cap;
    int       nv;         /* vregs */
    uint8_t*  vk;         /* vreg class */
    uint32_t* vc;         /* FMJ_K_CONST: the bits */
    int       nvcap;
    int       nmask;
    fmj_call* calls;
    int       ncalls, callcap;
    int       nargs;      /* argument / result arrays of the largest call */
    /* frame layout (bytes from the frame base; 64 byte aligned areas) */
    int       off_m, off_args, off_in, off_out, off_spill, frame_size;
    int       nin, nout;  /* input / output words per lane */
    int       nspill;     /* set by the code generator */
    int*      vw;         /* variable id -> its first input / output word (-1) */
    int       ubo_need[FM3D_MAX_UNIFORM_BLOCKS + 1]; /* bytes read at static offsets (the glue pads the blocks) */
    int       interp; /* fs: the varyings come from the planes (in words 0..2: dx, dy, w) */
    int       fs;
    const sv_stage* s;
    /* machine code */
    void*     code;
    size_t    code_size, code_cap;
    void (*fn)(void* frame);
} fmj_prog;

/* the frame a group runs in (allocated per thread by the glue, 64 byte aligned) */
typedef struct fmj_frame {
    const void*       io;   /* fm3d_fs_io / fm3d_vs_io */
    const fmj_prog*   prog;
    const uint8_t*    ubo[FM3D_MAX_UNIFORM_BLOCKS + 1]; /* [FMJ_PLANES]: the triangle's varying planes */
    size_t            ubo_n[FM3D_MAX_UNIFORM_BLOCKS];
    int               group; /* fragment: quad group 0..3; vertex: first vertex of the group */
    int               nq;    /* fragment: quads of the batch in this group (texture LOD) */
    int               fs;
} fmj_frame;

/* translation; NULL with err set when the program needs the interpreter */
fmj_prog* fmj_build(const sv_stage* s, const fm3d_spirv* P, int fs, char* err, size_t errn);
void      fmj_free(fmj_prog* p);
/* the program on one group's frame, portably (testing the translation) */
void      fmj_run_ref(const fmj_prog* p, void* frame);
/* machine code; 0 when the CPU / program is not supported (the reference runs) */
int       fmj_compile_x86(fmj_prog* p, char* err, size_t errn);

/* executable memory (fm3d_jit.c) */
void* fmj_code_alloc(size_t bytes);
int   fmj_code_seal(void* code, size_t bytes); /* writable -> executable */
void  fmj_code_free(void* code, size_t bytes);

/* helpers called by the code */
typedef void (*fmj_helper)(void* frame, const fmj_call* call);
extern const fmj_helper fmj_helpers[FMJ_H_COUNT];

/* the stage entry points (fm3d_program callbacks, io->user = fm3d_spirv) */
void fmj_run_vs(const fm3d_vs_io* io);
void fmj_run_fs(const fm3d_fs_io* io);

#endif
#endif
