/*
 * fatmap - SPIR-V backend internals (FM_FEATURE_SPIRV): the analyzed module
 * and its lowered form, shared by the backends.
 *
 *   SPIR-V words -> sv_parse / sv_analyze (types, registers, resolved
 *   pointers, pre-decoded blocks) -> sv_lower (the structured control flow
 *   walked statically: lane mask ops, IF / LOOP markers and block bodies)
 *
 * The lowered program (sv_ir) is backend neutral: the batch interpreter
 * runs it (fm3d_spirv.c) and the C emitter prints it (fm3d_spirv_c.c); a
 * runtime code generator would consume the same form. Block bodies stay
 * SPIR-V instructions with their operands resolved (sv_id.reg etc.).
 */
#ifndef FM3D_SPIRV_INTERNAL_H
#define FM3D_SPIRV_INTERNAL_H
#include "fm3d_internal.h"

#if FM_FEATURE_SPIRV

#define SV_L 64 /* lanes */

/* ---- SPIR-V constants used ---- */
enum {
    OpNop = 0, OpUndef = 1, OpName = 5, OpMemberName = 6, OpString = 7, OpLine = 8, OpExtension = 10,
    OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
    OpCapability = 17, OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypeMatrix = 24, OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27, OpTypeArray = 28,
    OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32, OpTypeFunction = 33, OpConstantTrue = 41,
    OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44, OpConstantNull = 46, OpSpecConstantTrue = 48,
    OpSpecConstantFalse = 49, OpSpecConstant = 50, OpSpecConstantComposite = 51, OpFunction = 54,
    OpFunctionParameter = 55, OpFunctionEnd = 56, OpFunctionCall = 57, OpVariable = 59, OpLoad = 61, OpStore = 62,
    OpAccessChain = 65, OpInBoundsAccessChain = 66, OpDecorate = 71, OpMemberDecorate = 72,
    OpVectorExtractDynamic = 77, OpVectorInsertDynamic = 78, OpVectorShuffle = 79, OpCompositeConstruct = 80,
    OpCompositeExtract = 81, OpCompositeInsert = 82, OpCopyObject = 83, OpTranspose = 84, OpSampledImage = 86,
    OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88, OpImage = 100, OpConvertFToU = 109,
    OpConvertFToS = 110, OpConvertSToF = 111, OpConvertUToF = 112, OpUConvert = 113, OpSConvert = 114,
    OpFConvert = 115, OpBitcast = 124, OpSNegate = 126, OpFNegate = 127, OpIAdd = 128, OpFAdd = 129, OpISub = 130,
    OpFSub = 131, OpIMul = 132, OpFMul = 133, OpUDiv = 134, OpSDiv = 135, OpFDiv = 136, OpUMod = 137, OpSRem = 138,
    OpSMod = 139, OpFRem = 140, OpFMod = 141, OpVectorTimesScalar = 142, OpMatrixTimesScalar = 143,
    OpVectorTimesMatrix = 144, OpMatrixTimesVector = 145, OpMatrixTimesMatrix = 146, OpOuterProduct = 147,
    OpDot = 148, OpAny = 154, OpAll = 155, OpIsNan = 156, OpIsInf = 157, OpLogicalEqual = 164,
    OpLogicalNotEqual = 165, OpLogicalOr = 166, OpLogicalAnd = 167, OpLogicalNot = 168, OpSelect = 169,
    OpIEqual = 170, OpINotEqual = 171, OpUGreaterThan = 172, OpSGreaterThan = 173, OpUGreaterThanEqual = 174,
    OpSGreaterThanEqual = 175, OpULessThan = 176, OpSLessThan = 177, OpULessThanEqual = 178, OpSLessThanEqual = 179,
    OpFOrdEqual = 180, OpFUnordEqual = 181, OpFOrdNotEqual = 182, OpFUnordNotEqual = 183, OpFOrdLessThan = 184,
    OpFUnordLessThan = 185, OpFOrdGreaterThan = 186, OpFUnordGreaterThan = 187, OpFOrdLessThanEqual = 188,
    OpFUnordLessThanEqual = 189, OpFOrdGreaterThanEqual = 190, OpFUnordGreaterThanEqual = 191,
    OpShiftRightLogical = 194, OpShiftRightArithmetic = 195, OpShiftLeftLogical = 196, OpBitwiseOr = 197,
    OpBitwiseXor = 198, OpBitwiseAnd = 199, OpNot = 200, OpDPdx = 207, OpDPdy = 208, OpFwidth = 209,
    OpDPdxFine = 210, OpDPdyFine = 211, OpFwidthFine = 212, OpDPdxCoarse = 213, OpDPdyCoarse = 214,
    OpFwidthCoarse = 215, OpPhi = 245, OpLoopMerge = 246, OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249,
    OpBranchConditional = 250, OpSwitch = 251, OpKill = 252, OpReturn = 253, OpReturnValue = 254,
    OpUnreachable = 255, OpNoLine = 317, OpModuleProcessed = 330, OpExecutionModeId = 331,
    OpTerminateInvocation = 4416, OpDecorateString = 5632, OpMemberDecorateString = 5633
};
enum { SC_UniformConstant = 0, SC_Input = 1, SC_Uniform = 2, SC_Output = 3, SC_Private = 6, SC_Function = 7, SC_PushConstant = 9 };
enum { DEC_ArrayStride = 6, DEC_MatrixStride = 7, DEC_BuiltIn = 11, DEC_Location = 30, DEC_Binding = 33, DEC_Offset = 35 };
enum { BI_Position = 0, BI_PointSize = 1, BI_ClipDistance = 3, BI_CullDistance = 4, BI_FragCoord = 15, BI_PointCoord = 16, BI_FrontFacing = 17,
       BI_VertexIndex = 42, BI_InstanceIndex = 43 };

/* ---- program representation ---- */
enum { T_NONE = 0, T_VOID, T_BOOL, T_INT, T_UINT, T_FLOAT, T_VEC, T_MAT, T_ARR, T_STRUCT, T_PTR, T_FUNC, T_IMAGE, T_SIMAGE, T_SAMPLER };
enum { C_NONE = 0, C_TYPE, C_CONST, C_VAR, C_VALUE, C_PTR, C_LABEL, C_EXT, C_IMG };

#define SV_MAXDYN 4
typedef struct sv_id {
    uint8_t cls;
    uint8_t kind;    /* types: T_* */
    int     type;    /* result / pointee type */
    int     comps;   /* types: flattened 32 bit components */
    int     elem;    /* vec / mat / arr element type, ptr pointee */
    int     count;   /* vec / mat / arr length */
    int     nmem;    /* struct */
    int*    mem;     /* member types */
    int*    moff;    /* member component offsets */
    int*    mboff;   /* member byte offsets (Offset) */
    int*    mstride; /* member matrix strides */
    int*    mbuiltin;
    int     astride; /* ArrayStride */
    int     storage; /* ptr types, vars */
    int     reg;     /* values / vars: lane block offset (>= 0 scratch, < 0 constant: -(off + 1)) */
    int     loc, builtin, binding;
    /* pointers: variable + constant offset + dynamic indices (component or
     * byte units, see storage) */
    int pvar, poff, pmstride, ndyn;
    int dyn[SV_MAXDYN], dstride[SV_MAXDYN];
    int unit; /* C_IMG: texture unit */
    int init; /* variables: initializer constant (0 = none) */
} sv_id;

/* a pre-decoded instruction: word offset, opcode, result components */
typedef struct sv_inst {
    uint32_t at;
    uint16_t op, n;
} sv_inst;

/* ---- lowered control flow ----
 * Lane masks live in numbered 64 bit slots; slot 0 is always zero, 1 / 2
 * collect killed / returned lanes, 3 is the entry mask. Edge slots hold the
 * lanes that branched from a block into another since that block last ran
 * (its phis pick by them; the block's BODY clears them after the phis). */
enum {
    IR_ZERO,    /* M[a] = 0 */
    IR_COPY,    /* M[a] = M[b] */
    IR_OR,      /* M[a] = M[b] | M[c] */
    IR_ANDN,    /* M[a] = M[b] & ~M[c] */
    IR_COND,    /* M[a] = lanes where value b (bool) is true, & M[c] */
    IR_EQ,      /* M[a] = lanes where value b (32 bit) == lit, & M[c] */
    IR_BODY,    /* run block a under M[b]: phis, clear its edges, instructions (stores masked) */
    IR_IF,      /* skip to the matching IR_ENDIF (jump) when M[a] == 0 */
    IR_ENDIF,
    IR_LOOP,    /* M[a] = 0 (iteration counter) */
    IR_BREAKZ,  /* leave the loop (jump: past IR_ENDLOOP) when M[a] == 0 or ++M[b] > SV_MAXITER */
    IR_BREAK,   /* leave the loop */
    IR_ENDLOOP  /* back to the instruction after IR_LOOP (jump) */
};
#define SV_M_ZERO   0
#define SV_M_KILLED 1
#define SV_M_DONE   2
#define SV_M_ENTRY  3
#define SV_MAXITER  (1u << 20)

typedef struct sv_ir {
    uint8_t  op;
    int      a, b, c;
    uint32_t lit;
    int      jump;
} sv_ir;

/* ---- value classes (sv_classify) ----
 * How a backend keeps each SSA value. Instructions of a block are split
 * into segments at cross lane instructions (derivatives, texture samples):
 * inside a segment every instruction only touches its own lane, so a
 * backend may fuse a segment into one loop over the lanes. */
enum {
    VC_NONE = 0,
    VC_CONST,   /* compile time constant (sv_cblock) */
    VC_UNIFORM, /* the same for every lane: computed once per execution as scalars */
    VC_LOCAL,   /* per lane, used only inside its defining segment */
    VC_MAT      /* per lane, kept as 64 lane arrays */
};
enum { VS_BLOCK = 0, VS_FUNC }; /* VC_MAT scope: only used inside the defining block, or anywhere */

typedef struct sv_edgevar {
    int from, to; /* label id, block index */
    int slot;
} sv_edgevar;

typedef struct sv_block {
    int      label;
    size_t   start, term; /* word offsets: first instruction, terminator */
    int      merge, cont; /* -1 if none */
    int      loop;
    sv_inst* insts;       /* executable instructions (phis first) */
    int      ninst, nphi;
    int*     phim;        /* per phi incoming (value, parent) pair: edge slot (0 = never taken) */
    int*     clr;         /* edge slots into this block */
    int      nclr;
    int*     seg;         /* per instruction: segment (sv_classify) */
} sv_block;

typedef struct sv_io { /* an interface variable */
    int var, loc, comps, builtin, off; /* off: component offset inside var (member builtins) */
} sv_io;

typedef struct sv_stage {
    int         model; /* 0 vertex, 4 fragment */
    uint32_t*   w;
    size_t      nw;
    sv_id*      ids;
    uint32_t    bound;
    int         glsl; /* GLSL.std.450 import id */
    int         entry;
    sv_block*   blocks;
    int         nblocks;
    int*        bix;  /* label id -> block index */
    uint32_t*   cdata;
    int         ncdata;  /* lane blocks */
    int         nscratch;
    int         nphi;    /* comps of the largest phi set of a block (temp) */
    sv_io       in[32], out[32];
    int         nin, nout;
    int         pos_var, pos_off; /* vertex: gl_Position */
    int         ps_var, ps_off;   /* vertex: gl_PointSize (-1: not written) */
    int         kills;
    int         nvars;
    int*        vars; /* function / private / input / output variables to set up per batch */
    int         varcap;
    /* lowered program */
    sv_ir*      ir;
    int         nir, irmax;
    int         nmask;    /* mask slots */
    sv_edgevar* ev;
    int         nev, evmax;
    int         lower_err;
    int         fast;      /* fm3d_spirv_set_fast_math: fm_fast_* instead of fm_* */
    /* value classes */
    uint8_t*    vcls;
    uint8_t*    vscope;
    int*        vseg;   /* defining segment */
    int*        vblock; /* defining block */
    int         nseg;
    int         divergent; /* a loop whose exits can differ per lane */
} sv_stage;

struct fm3d_spirv {
    sv_stage* vs;
    sv_stage* fs;
    int       nvar;           /* varying slots */
    int       vslot[32];      /* vs output i -> slot */
    int       fslot[32];      /* fs input i -> slot */
    int       ps_slot, pc_slot; /* gl_PointSize / gl_PointCoord varyings (-1: none) */
    fm3d_vertex_attrib attr[16];
    int       nattr;
};

static inline void sv_bcast(uint32_t* dst, uint32_t v)
{
    for (int l = 0; l < SV_L; l++) dst[l] = v;
}

/* Fragment lanes of the shader backends are in quad group order: 16 lane
 * group k is columns 8k..8k+7 of both rows of the batch (lane = 16k + row
 * * 8 + column % 8), so every 16 lanes hold four whole 2x2 quads (x
 * neighbour: lane ^ 1, y neighbour: lane ^ 8). Batch pixel of a lane: */
static inline int sv_frag_pixel(int l) { return ((l >> 3) & 1) * 32 + ((l >> 4) << 3) + (l & 7); }

/* batch order <-> quad group order: 8 runs of 8 contiguous pixels */
static inline void sv_to_groups(float* d, const float* src)
{
    for (int g = 0; g < 4; g++) {
        memcpy(d + 16 * g, src + 8 * g, 8 * sizeof(float));
        memcpy(d + 16 * g + 8, src + 32 + 8 * g, 8 * sizeof(float));
    }
}
static inline void sv_from_groups(float* d, const float* src)
{
    for (int g = 0; g < 4; g++) {
        memcpy(d + 8 * g, src + 16 * g, 8 * sizeof(float));
        memcpy(d + 32 + 8 * g, src + 16 * g + 8, 8 * sizeof(float));
    }
}

static inline int sv_is_ptr_storage_uniform(int sc) { return sc == SC_Uniform || sc == SC_PushConstant; }

/* the module with the entry points' function calls inlined (malloc'd), NULL
 * when there are none or on error (err set) (fm3d_spirv_inline.c) */
uint32_t* sv_inline_calls(const uint32_t* w, size_t nw, size_t* out_words, char* err, size_t errn);

/* per thread scratch (fm3d_spirv.c) */
uint32_t* sv_scratch(size_t blocks);
uint64_t* sv_mask_scratch(int n);

/* the batch executor per ISA (fm3d_spirv_exec.h) */
void sv_run_vs_base(const fm3d_vs_io* io);
void sv_run_fs_base(const fm3d_fs_io* io);
void sv_run_vs_avx2(const fm3d_vs_io* io);
void sv_run_fs_avx2(const fm3d_fs_io* io);
void sv_run_vs_avx512(const fm3d_vs_io* io);
void sv_run_fs_avx512(const fm3d_fs_io* io);

/* value ids an instruction reads (pointer operands: their dynamic indices) */
int sv_operands(const sv_stage* s, const uint32_t* in, int op, int* ids, int max);
int sv_is_cross_lane(int op);

#define WC(p) ((p)[0] >> 16)
#define OP(p) ((p)[0] & 0xffffu)

static inline uint32_t* sv_cblock(const sv_stage* s, const sv_id* d) { return s->cdata + (size_t)(-d->reg - 1) * SV_L; }

/* component offset of child `idx` of type t (vectors, matrices, arrays, structs) */
static inline int sv_child(const sv_stage* s, int t, int idx, int* ct)
{
    const sv_id* T = &s->ids[t];
    switch (T->kind) {
    case T_VEC: case T_MAT: case T_ARR:
        *ct = T->elem;
        return idx * s->ids[T->elem].comps;
    case T_STRUCT:
        if (idx < 0 || idx >= T->nmem) return -1;
        *ct = T->mem[idx];
        return T->moff[idx];
    default: return -1;
    }
}

#endif /* FM_FEATURE_SPIRV */
#endif
