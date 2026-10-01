/*
 * fatmap - SPIR-V shader backend (FM_FEATURE_SPIRV).
 *
 * A "wide" interpreter: every instruction runs over a batch of 64 lanes
 * (64 fragments of a 2 x 32 batch, or up to 64 vertices), so the dispatch
 * cost is shared and the per op loops are plain 64 iteration array loops.
 * Values are stored component major (component c of lane l at c * 64 + l).
 * Structured control flow runs with 64 bit lane masks: selections run both
 * sides under their masks, loops iterate while any lane is active, OpPhi
 * picks per lane by the block the lane came from. The control flow is
 * lowered once, at creation (sv_lower, see fm3d_spirv_internal.h), so a
 * batch only runs the lowered program. Fragment helper lanes (uncovered
 * pixels of a quad) execute too, so derivatives are exact.
 *
 * Supported: GLSL.std.450 vertex / fragment shaders as glslc -O emits them
 * (functions inlined): scalars, vectors, matrices, arrays, structs, one
 * uniform block (std140 offsets from the decorations) or push constants,
 * sampler2D (implicit / explicit LOD), inputs / outputs by location,
 * gl_Position, gl_FragCoord, discard, derivatives and the common
 * GLSL.std.450 functions. Anything else is rejected when the program is
 * created, with a message.
 */
#include "fm3d_spirv_internal.h"
#include "fatmap/fm_vmath.h"

#if FM_FEATURE_SPIRV

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#if FM_ARCH_X86
#  include <emmintrin.h>
#elif FM_ARCH_ARM64
#  include <arm_neon.h>
#endif

#if defined(FM_NO_THREADS)
#  define SV_TLS
#elif defined(_MSC_VER)
#  define SV_TLS __declspec(thread)
#else
#  define SV_TLS _Thread_local
#endif

static int sv_err(char* err, size_t n, const char* fmt, ...)
{
    if (err && n) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, n, fmt, ap);
        va_end(ap);
    }
    return 0;
}


static void sv_stage_free(sv_stage* s)
{
    if (!s) return;
    if (s->ids)
        for (uint32_t i = 0; i < s->bound; i++) {
            free(s->ids[i].mem);
            free(s->ids[i].moff);
            free(s->ids[i].mboff);
            free(s->ids[i].mstride);
            free(s->ids[i].mbuiltin);
        }
    free(s->ids);
    for (int b = 0; s->blocks && b < s->nblocks; b++) {
        free(s->blocks[b].insts);
        free(s->blocks[b].phim);
        free(s->blocks[b].clr);
        free(s->blocks[b].seg);
    }
    free(s->vcls);
    free(s->vscope);
    free(s->vseg);
    free(s->vblock);
    free(s->ir);
    free(s->ev);
    free(s->blocks);
    free(s->bix);
    free(s->cdata);
    free(s->w);
    free(s);
}

/* ---- parsing ---------------------------------------------------------------- */

static int sv_is_ignored(int op)
{
    switch (op) {
    case OpNop: case OpName: case OpMemberName: case OpString: case OpLine: case OpExtension: case OpMemoryModel:
    case OpExecutionMode: case OpCapability: case OpNoLine: case OpModuleProcessed: case OpExecutionModeId:
    case OpDecorateString: case OpMemberDecorateString: case 3: case 4: case 2:
        return 1;
    default: return 0;
    }
}


/* flatten a constant into 32 bit components (scalar values) */
static int sv_const_comps(sv_stage* s, int id, uint32_t* out, int max)
{
    sv_id* d = &s->ids[id];
    if (d->cls != C_CONST || d->comps > max) return -1;
    uint32_t* src = sv_cblock(s, d);
    for (int c = 0; c < d->comps; c++) out[c] = src[(size_t)c * SV_L];
    return d->comps;
}

static int sv_const_int(sv_stage* s, int id, int* v)
{
    uint32_t c[1];
    if (id <= 0 || (uint32_t)id >= s->bound || s->ids[id].cls != C_CONST || s->ids[id].comps != 1) return 0;
    sv_const_comps(s, id, c, 1);
    *v = (int)c[0];
    return 1;
}

static int sv_alloc_const(sv_stage* s, sv_id* d, int comps)
{
    uint32_t* n = (uint32_t*)realloc(s->cdata, (size_t)(s->ncdata + comps) * SV_L * sizeof(uint32_t));
    if (!n) return 0;
    s->cdata = n;
    d->reg   = -(s->ncdata + 1);
    memset(s->cdata + (size_t)s->ncdata * SV_L, 0, (size_t)comps * SV_L * sizeof(uint32_t));
    s->ncdata += comps;
    return 1;
}

static void sv_bcast(uint32_t* dst, uint32_t v)
{
    for (int l = 0; l < SV_L; l++) dst[l] = v;
}

static sv_stage* sv_parse(const uint32_t* words, size_t nw, int want_model, char* err, size_t errn)
{
    if (!words || nw < 5 || words[0] != 0x07230203u) {
        sv_err(err, errn, "not a SPIR-V module (magic)");
        return NULL;
    }
    sv_stage* s = (sv_stage*)calloc(1, sizeof(sv_stage));
    if (!s) return NULL;
    s->bound = words[3];
    s->nw    = nw;
    s->w     = (uint32_t*)malloc(nw * sizeof(uint32_t));
    s->ids   = (sv_id*)calloc(s->bound, sizeof(sv_id));
    s->bix   = (int*)malloc(s->bound * sizeof(int));
    s->pos_var = -1;
    s->entry   = -1;
    if (!s->w || !s->ids || !s->bix || s->bound > (1u << 20)) goto fail;
    memcpy(s->w, words, nw * sizeof(uint32_t));
    for (uint32_t i = 0; i < s->bound; i++) {
        s->bix[i]          = -1;
        s->ids[i].loc      = -1;
        s->ids[i].builtin  = -1;
        s->ids[i].binding  = -1;
    }

    /* pass 1: decorations and the entry point */
    for (size_t p = 5; p < nw;) {
        const uint32_t* in = s->w + p;
        int             wc = (int)WC(in), op = (int)OP(in);
        if (wc == 0 || p + (size_t)wc > nw) {
            sv_err(err, errn, "truncated instruction at word %zu", p);
            goto fail;
        }
        if (op == OpEntryPoint) {
            if ((int)in[1] == want_model && s->entry < 0) s->entry = (int)in[2];
        } else if (op == OpDecorate && wc >= 3 && in[1] < s->bound) {
            sv_id* d = &s->ids[in[1]];
            if (in[2] == DEC_Location && wc >= 4) d->loc = (int)in[3];
            if (in[2] == DEC_BuiltIn && wc >= 4) d->builtin = (int)in[3];
            if (in[2] == DEC_Binding && wc >= 4) d->binding = (int)in[3];
            if (in[2] == DEC_ArrayStride && wc >= 4) d->astride = (int)in[3];
        } else if (op == OpExtInstImport) {
            const char* name = (const char*)(in + 2);
            if (!strncmp(name, "GLSL.std.450", (size_t)(wc - 2) * 4)) s->glsl = (int)in[1];
        }
        p += (size_t)wc;
    }
    if (s->entry < 0) {
        sv_err(err, errn, "no %s entry point", want_model == 0 ? "Vertex" : "Fragment");
        goto fail;
    }
    s->model = want_model;

    /* pass 2: types, constants, globals, the entry function's blocks */
    int in_entry = 0, in_other = 0, cur_block = -1;
    for (size_t p = 5; p < nw;) {
        const uint32_t* in = s->w + p;
        int             wc = (int)WC(in), op = (int)OP(in);
        uint32_t        rid = 0;
        if (in_other && op != OpFunctionEnd) { /* other functions: never called (no OpFunctionCall allowed) */
            p += (size_t)wc;
            continue;
        }
        switch (op) {
        case OpTypeVoid: case OpTypeBool: case OpTypeInt: case OpTypeFloat: case OpTypeVector: case OpTypeMatrix:
        case OpTypeArray: case OpTypeStruct: case OpTypePointer: case OpTypeFunction: case OpTypeImage:
        case OpTypeSampledImage: case OpTypeSampler: case OpTypeRuntimeArray: {
            rid = in[1];
            if (rid >= s->bound) goto bad_id;
            sv_id* T = &s->ids[rid];
            T->cls   = C_TYPE;
            T->comps = 1;
            switch (op) {
            case OpTypeVoid: T->kind = T_VOID, T->comps = 0; break;
            case OpTypeBool: T->kind = T_BOOL; break;
            case OpTypeInt:
                if (in[2] != 32) {
                    sv_err(err, errn, "only 32 bit integers");
                    goto fail;
                }
                T->kind = in[3] ? T_INT : T_UINT;
                break;
            case OpTypeFloat:
                if (in[2] != 32) {
                    sv_err(err, errn, "only 32 bit floats");
                    goto fail;
                }
                T->kind = T_FLOAT;
                break;
            case OpTypeVector: case OpTypeMatrix:
                T->kind  = op == OpTypeVector ? T_VEC : T_MAT;
                T->elem  = (int)in[2];
                T->count = (int)in[3];
                T->comps = T->count * s->ids[T->elem].comps;
                break;
            case OpTypeArray: {
                int n = 0;
                if (!sv_const_int(s, (int)in[3], &n) || n <= 0) goto unsupported;
                T->kind  = T_ARR;
                T->elem  = (int)in[2];
                T->count = n;
                T->comps = n * s->ids[T->elem].comps;
                break;
            }
            case OpTypeStruct: {
                T->kind  = T_STRUCT;
                T->nmem  = wc - 2;
                T->mem   = (int*)calloc((size_t)T->nmem + 1, sizeof(int));
                T->moff  = (int*)calloc((size_t)T->nmem + 1, sizeof(int));
                if (!T->mem || !T->moff) goto fail;
                if (!T->mboff) T->mboff = (int*)calloc((size_t)T->nmem + 1, sizeof(int));
                if (!T->mstride) T->mstride = (int*)calloc((size_t)T->nmem + 1, sizeof(int));
                if (!T->mbuiltin) {
                    T->mbuiltin = (int*)malloc(((size_t)T->nmem + 1) * sizeof(int));
                    for (int m = 0; T->mbuiltin && m <= T->nmem; m++) T->mbuiltin[m] = -1;
                }
                if (!T->mboff || !T->mstride || !T->mbuiltin) goto fail;
                T->comps = 0;
                for (int m = 0; m < T->nmem; m++) {
                    T->mem[m]  = (int)in[2 + m];
                    T->moff[m] = T->comps;
                    T->comps += s->ids[T->mem[m]].comps;
                }
                break;
            }
            case OpTypePointer: T->kind = T_PTR, T->storage = (int)in[2], T->elem = (int)in[3], T->comps = 0; break;
            case OpTypeFunction: T->kind = T_FUNC, T->comps = 0; break;
            case OpTypeImage: T->kind = T_IMAGE; break;
            case OpTypeSampledImage: T->kind = T_SIMAGE; break;
            case OpTypeSampler: T->kind = T_SAMPLER; break;
            default: goto unsupported;
            }
            break;
        }
        case OpMemberDecorate: /* after pass 1 order: struct types are declared later, so stash in the id */
            break;
        case OpConstantTrue: case OpConstantFalse: case OpConstant: case OpConstantComposite: case OpConstantNull:
        case OpSpecConstantTrue: case OpSpecConstantFalse: case OpSpecConstant: case OpSpecConstantComposite:
        case OpUndef: {
            rid = in[2];
            if (rid >= s->bound || in[1] >= s->bound) goto bad_id;
            sv_id* d = &s->ids[rid];
            int    comps = s->ids[in[1]].comps;
            d->cls   = C_CONST;
            d->type  = (int)in[1];
            d->comps = comps;
            if (!sv_alloc_const(s, d, comps)) goto fail;
            uint32_t* dst = sv_cblock(s, d);
            if (op == OpConstantTrue || op == OpSpecConstantTrue) sv_bcast(dst, 1);
            else if (op == OpConstant || op == OpSpecConstant) sv_bcast(dst, in[3]);
            else if (op == OpConstantComposite || op == OpSpecConstantComposite) {
                int c = 0;
                for (int k = 3; k < wc; k++) {
                    sv_id* e = &s->ids[in[k]];
                    if (e->cls != C_CONST) goto unsupported;
                    uint32_t* es = sv_cblock(s, e);
                    dst          = sv_cblock(s, d); /* cdata may have moved */
                    for (int j = 0; j < e->comps && c < comps; j++, c++) memcpy(dst + (size_t)c * SV_L, es + (size_t)j * SV_L, SV_L * 4);
                }
            }
            break;
        }
        case OpVariable: {
            rid = in[2];
            if (rid >= s->bound) goto bad_id;
            sv_id* d = &s->ids[rid];
            d->cls     = C_VAR;
            d->type    = s->ids[in[1]].elem; /* pointee */
            d->storage = (int)in[3];
            d->comps   = s->ids[d->type].comps;
            d->pvar    = (int)rid;
            d->init    = wc >= 5 ? (int)in[4] : 0;
            if (d->storage == SC_Function || d->storage == SC_Private || d->storage == SC_Input || d->storage == SC_Output) {
                d->reg = s->nscratch;
                s->nscratch += d->comps;
                if (s->nvars == 256) goto unsupported;
                s->vars[s->nvars++] = (int)rid;
            }
            if (!in_entry && d->storage == SC_Function) goto unsupported;
            break;
        }
        case OpExtInstImport: rid = in[1]; if (rid < s->bound) s->ids[rid].cls = C_EXT; break;
        case OpFunction:
            if ((int)in[2] == s->entry) in_entry = 1;
            else in_other = 1;
            break;
        case OpFunctionEnd: in_entry = in_other = 0; break;
        case OpFunctionCall:
            sv_err(err, errn, "function calls are not supported: compile with glslc -O (inlines everything)");
            goto fail;
        case OpLabel:
            if (in_entry) {
                sv_block* nb = (sv_block*)realloc(s->blocks, ((size_t)s->nblocks + 1) * sizeof(sv_block));
                if (!nb) goto fail;
                s->blocks = nb;
                cur_block = s->nblocks++;
                sv_block* b = &s->blocks[cur_block];
                memset(b, 0, sizeof(*b));
                b->label = (int)in[1];
                b->start = p + (size_t)wc;
                b->merge = b->cont = -1;
                s->bix[in[1]]     = cur_block;
                s->ids[in[1]].cls = C_LABEL;
            }
            break;
        default:
            if (in_entry && cur_block >= 0) {
                sv_block* b = &s->blocks[cur_block];
                if (op == OpSelectionMerge) b->merge = (int)in[1];
                if (op == OpLoopMerge) b->merge = (int)in[1], b->cont = (int)in[2], b->loop = 1;
                if (op == OpBranch || op == OpBranchConditional || op == OpSwitch || op == OpKill || op == OpReturn ||
                    op == OpReturnValue || op == OpUnreachable || op == OpTerminateInvocation)
                    b->term = p;

            }
            break;
        }
        (void)in_other;
        p += (size_t)wc;
    }
    /* member decorations (the struct types exist now) */
    for (size_t p = 5; p < nw;) {
        const uint32_t* in = s->w + p;
        int             wc = (int)WC(in), op = (int)OP(in);
        if (op == OpMemberDecorate && wc >= 4 && in[1] < s->bound) {
            sv_id* T = &s->ids[in[1]];
            int    m = (int)in[2];
            if (T->kind == T_STRUCT && m < T->nmem) {
                if (in[3] == DEC_Offset && wc >= 5) T->mboff[m] = (int)in[4];
                if (in[3] == DEC_MatrixStride && wc >= 5) T->mstride[m] = (int)in[4];
                if (in[3] == DEC_BuiltIn && wc >= 5) T->mbuiltin[m] = (int)in[4];
            }
        }
        p += (size_t)wc;
    }
    if (s->nblocks == 0) {
        sv_err(err, errn, "entry point has no body");
        goto fail;
    }
    return s;
bad_id:
    sv_err(err, errn, "id out of bounds");
    goto fail;
unsupported:
    sv_err(err, errn, "unsupported construct (arrays need constant lengths, one function)");
fail:
    if (err && errn && !err[0]) sv_err(err, errn, "out of memory");
    sv_stage_free(s);
    return NULL;
}

/* ---- analysis: result registers, pointers, the interface ---------------------- */

static int sv_is_ptr_storage_uniform(int sc) { return sc == SC_Uniform || sc == SC_PushConstant; }

static const char* sv_op_ok(sv_stage* s, const uint32_t* in, int op)
{
    switch (op) {
    case OpLoad: case OpStore: case OpAccessChain: case OpInBoundsAccessChain: case OpVectorExtractDynamic:
    case OpVectorInsertDynamic: case OpVectorShuffle: case OpCompositeConstruct: case OpCompositeExtract:
    case OpCompositeInsert: case OpCopyObject: case OpTranspose: case OpSampledImage: case OpImageSampleImplicitLod:
    case OpImageSampleExplicitLod: case OpConvertFToU: case OpConvertFToS: case OpConvertSToF: case OpConvertUToF:
    case OpUConvert: case OpSConvert: case OpFConvert: case OpBitcast: case OpSNegate: case OpFNegate: case OpIAdd:
    case OpFAdd: case OpISub: case OpFSub: case OpIMul: case OpFMul: case OpUDiv: case OpSDiv: case OpFDiv:
    case OpUMod: case OpSRem: case OpSMod: case OpFRem: case OpFMod: case OpVectorTimesScalar:
    case OpMatrixTimesScalar: case OpVectorTimesMatrix: case OpMatrixTimesVector: case OpMatrixTimesMatrix:
    case OpOuterProduct: case OpDot: case OpAny: case OpAll: case OpIsNan: case OpIsInf: case OpLogicalEqual:
    case OpLogicalNotEqual: case OpLogicalOr: case OpLogicalAnd: case OpLogicalNot: case OpSelect: case OpIEqual:
    case OpINotEqual: case OpUGreaterThan: case OpSGreaterThan: case OpUGreaterThanEqual: case OpSGreaterThanEqual:
    case OpULessThan: case OpSLessThan: case OpULessThanEqual: case OpSLessThanEqual: case OpFOrdEqual:
    case OpFUnordEqual: case OpFOrdNotEqual: case OpFUnordNotEqual: case OpFOrdLessThan: case OpFUnordLessThan:
    case OpFOrdGreaterThan: case OpFUnordGreaterThan: case OpFOrdLessThanEqual: case OpFUnordLessThanEqual:
    case OpFOrdGreaterThanEqual: case OpFUnordGreaterThanEqual: case OpShiftRightLogical:
    case OpShiftRightArithmetic: case OpShiftLeftLogical: case OpBitwiseOr: case OpBitwiseXor: case OpBitwiseAnd:
    case OpNot: case OpDPdx: case OpDPdy: case OpFwidth: case OpDPdxFine: case OpDPdyFine: case OpFwidthFine:
    case OpDPdxCoarse: case OpDPdyCoarse: case OpFwidthCoarse: case OpPhi: case OpLoopMerge: case OpSelectionMerge:
    case OpBranch: case OpBranchConditional: case OpSwitch: case OpKill: case OpReturn: case OpUnreachable:
    case OpTerminateInvocation: case OpVariable: case OpUndef:
        return NULL;
    case OpExtInst:
        if ((int)in[3] != s->glsl) return "extended instruction set other than GLSL.std.450";
        switch (in[4]) {
        case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9: case 10: case 11: case 12: case 13:
        case 14: case 15: case 16: case 17: case 18: case 19: case 20: case 21: case 25: case 26: case 27: case 28:
        case 29: case 30: case 31: case 32: case 37: case 38: case 39: case 40: case 41: case 42: case 43: case 44:
        case 45: case 46: case 48: case 49: case 50: case 66: case 67: case 68: case 69: case 70: case 71: case 72:
        case 79: case 80: case 81:
            return NULL;
        default: return "unsupported GLSL.std.450 function";
        }
    default: return "unsupported instruction";
    }
}

/* resolve an access chain: base (var or chain) + indices */
static int sv_chain(sv_stage* s, sv_id* d, int base, const uint32_t* idx, int nidx, char* err, size_t errn)
{
    sv_id* b = &s->ids[base];
    if (b->cls != C_VAR && b->cls != C_PTR) return sv_err(err, errn, "access chain on a non pointer");
    const sv_id* v   = &s->ids[b->pvar];
    int          uni = sv_is_ptr_storage_uniform(v->storage);
    *d               = *b;
    d->cls           = C_PTR;
    d->mem = d->moff = d->mboff = d->mstride = d->mbuiltin = NULL;
    int t            = b->cls == C_VAR ? b->type : b->type;
    for (int k = 0; k < nidx; k++) {
        const sv_id* T = &s->ids[t];
        int          ci = 0, is_const = sv_const_int(s, (int)idx[k], &ci);
        int          ct = 0, stride = 0;
        if (T->kind == T_STRUCT) {
            if (!is_const || ci < 0 || ci >= T->nmem) return sv_err(err, errn, "struct index must be constant");
            ct = T->mem[ci];
            if (uni) {
                d->poff += T->mboff[ci];
                d->pmstride = T->mstride[ci];
            } else {
                d->poff += T->moff[ci];
            }
            if (!uni && T->mbuiltin[ci] >= 0) d->builtin = T->mbuiltin[ci];
            t = ct;
            continue;
        }
        if (T->kind != T_VEC && T->kind != T_MAT && T->kind != T_ARR) return sv_err(err, errn, "bad access chain");
        ct = T->elem;
        if (uni)
            stride = T->kind == T_ARR ? T->astride : (T->kind == T_MAT ? d->pmstride : 4);
        else
            stride = s->ids[ct].comps;
        if (uni && T->kind == T_ARR && !stride) return sv_err(err, errn, "uniform array without ArrayStride");
        if (uni && T->kind == T_MAT && !stride) return sv_err(err, errn, "uniform matrix without MatrixStride");
        if (is_const) {
            d->poff += ci * stride;
        } else {
            if (d->ndyn == SV_MAXDYN) return sv_err(err, errn, "too many dynamic indices");
            d->dyn[d->ndyn]     = (int)idx[k];
            d->dstride[d->ndyn] = stride;
            d->ndyn++;
        }
        t = ct;
    }
    d->type = t;
    return 1;
}

static int sv_analyze(sv_stage* s, char* err, size_t errn)
{
    /* interface variables */
    for (uint32_t i = 0; i < s->bound; i++) {
        sv_id* v = &s->ids[i];
        if (v->cls != C_VAR) continue;
        if (v->storage == SC_Input || v->storage == SC_Output) {
            sv_io io;
            memset(&io, 0, sizeof(io));
            io.var     = (int)i;
            io.loc     = v->loc;
            io.comps   = v->comps;
            io.builtin = v->builtin;
            const sv_id* T = &s->ids[v->type];
            if (T->kind == T_STRUCT) { /* gl_PerVertex style block: find Position */
                for (int m = 0; m < T->nmem; m++)
                    if (T->mbuiltin[m] == BI_Position && v->storage == SC_Output) s->pos_var = (int)i, s->pos_off = T->moff[m];
                continue;
            }
            if (v->builtin == BI_Position && v->storage == SC_Output) {
                s->pos_var = (int)i, s->pos_off = 0;
                continue;
            }
            if (v->builtin >= 0 && v->builtin != BI_FragCoord && v->builtin != BI_FrontFacing && v->builtin != BI_PointSize &&
                v->builtin != BI_ClipDistance && v->builtin != BI_CullDistance)
                return sv_err(err, errn, "unsupported builtin %d", v->builtin);
            if (v->builtin < 0 && v->loc < 0) return sv_err(err, errn, "interface variable without Location");
            if (v->storage == SC_Input) {
                if (s->nin == 32) return sv_err(err, errn, "too many inputs");
                s->in[s->nin++] = io;
            } else {
                if (s->nout == 32) return sv_err(err, errn, "too many outputs");
                s->out[s->nout++] = io;
            }
        } else if (v->storage == SC_UniformConstant) {
            if (s->ids[v->type].kind != T_SIMAGE) return sv_err(err, errn, "only sampler2D uniforms");
            if (v->binding < 0 || v->binding >= FM3D_MAX_TEXTURE_UNITS) return sv_err(err, errn, "sampler binding out of range");
        }
    }
    if (s->model == 0 && s->pos_var < 0) return sv_err(err, errn, "vertex shader does not write gl_Position");

    /* function body: validate, give results registers, resolve pointers */
    int maxphi = 0;
    for (int bi = 0; bi < s->nblocks; bi++) {
        sv_block* B = &s->blocks[bi];
        if (!B->term) return sv_err(err, errn, "block without terminator");
        int phicomps = 0;
        for (size_t p = B->start; p <= B->term;) {
            const uint32_t* in = s->w + p;
            int             wc = (int)WC(in), op = (int)OP(in);
            if (!sv_is_ignored(op)) {
                const char* why = sv_op_ok(s, in, op);
                if (why) return sv_err(err, errn, "%s (opcode %d)", why, op);
            }
            switch (op) {
            case OpStore: case OpSelectionMerge: case OpLoopMerge: case OpBranch: case OpBranchConditional: case OpSwitch:
            case OpKill: case OpReturn: case OpUnreachable: case OpTerminateInvocation: case OpVariable: case OpUndef:
                break;
            case OpAccessChain: case OpInBoundsAccessChain:
                if (!sv_chain(s, &s->ids[in[2]], (int)in[3], in + 4, wc - 4, err, errn)) return 0;
                break;
            case OpSampledImage:
                return sv_err(err, errn, "separate images / samplers are not supported (use sampler2D)");
            default:
                if (sv_is_ignored(op)) break;
                if (wc >= 3 && in[2] < s->bound && in[1] < s->bound) {
                    sv_id* d = &s->ids[in[2]];
                    int    rt = (int)in[1];
                    if (op == OpLoad && s->ids[rt].kind == T_SIMAGE) { /* sampler2D load: a unit number */
                        const sv_id* pv = &s->ids[in[3]];
                        d->cls  = C_IMG;
                        d->unit = pv->cls == C_VAR ? pv->binding : -1;
                        if (d->unit < 0) return sv_err(err, errn, "sampler without binding");
                        break;
                    }
                    d->cls   = C_VALUE;
                    d->type  = rt;
                    d->comps = s->ids[rt].comps;
                    d->reg   = s->nscratch;
                    s->nscratch += d->comps;
                    if (op == OpPhi) phicomps += d->comps;
                }
                break;
            }
            p += (size_t)wc;
        }
        maxphi = phicomps > maxphi ? phicomps : maxphi;
    }
    s->nphi = maxphi;
    /* pre-decode every block: executable instructions only, phis first */
    for (int bi = 0; bi < s->nblocks; bi++) {
        sv_block* B = &s->blocks[bi];
        int       n = 0;
        for (size_t p = B->start; p < B->term; p += WC(s->w + p)) n++;
        B->insts = (sv_inst*)malloc((size_t)(n + 1) * sizeof(sv_inst));
        if (!B->insts) return sv_err(err, errn, "out of memory");
        for (size_t p = B->start; p < B->term; p += WC(s->w + p)) {
            const uint32_t* in = s->w + p;
            int             op = (int)OP(in);
            if (sv_is_ignored(op) || op == OpSelectionMerge || op == OpLoopMerge || op == OpVariable || op == OpUndef ||
                op == OpAccessChain || op == OpInBoundsAccessChain)
                continue;
            if (op == OpLoad && s->ids[in[2]].cls == C_IMG) continue;
            sv_inst* it = &B->insts[B->ninst++];
            it->at      = (uint32_t)p;
            it->op      = (uint16_t)op;
            it->n       = (uint16_t)(op == OpStore ? 0 : s->ids[in[2]].comps);
            if (op == OpPhi) B->nphi++;
        }
    }
    return 1;
}

/* ---- lowering: the structured control flow walked statically ------------------
 * Mirrors what a mask based executor does at run time (selections run both
 * sides under their masks, loops iterate while lanes are active, branches
 * to an enclosing construct's merge / continue target are collected there)
 * but emits sv_ir instead, so no backend walks the CFG per batch. Blocks
 * reached on several paths without a merge (tail duplication) are emitted
 * once per path. */

typedef struct sv_lc { /* an enclosing breakable construct */
    int                merge, cont; /* label ids (-1: none) */
    int                brk, cm;     /* mask slots collecting branches to them */
    struct sv_lc*      outer;
} sv_lc;

#define SV_MAXIR (1 << 20)

static int lw_emit(sv_stage* s, int op, int a, int b, int c, uint32_t lit)
{
    if (s->lower_err) return 0;
    if (s->nir == s->irmax) {
        int    n  = s->irmax ? s->irmax * 2 : 256;
        sv_ir* ni = n <= SV_MAXIR ? (sv_ir*)realloc(s->ir, (size_t)n * sizeof(sv_ir)) : NULL;
        if (!ni) {
            s->lower_err = 1;
            return 0;
        }
        s->ir    = ni;
        s->irmax = n;
    }
    sv_ir* o = &s->ir[s->nir];
    o->op = (uint8_t)op, o->a = a, o->b = b, o->c = c, o->lit = lit, o->jump = -1;
    return s->nir++;
}

static int lw_mask(sv_stage* s) { return s->nmask++; }

static int lw_new(sv_stage* s, int op, int b, int c, uint32_t lit)
{
    int d = lw_mask(s);
    lw_emit(s, op, d, b, c, lit);
    return d;
}

static int lw_edge(sv_stage* s, int from, int bi)
{
    for (int i = 0; i < s->nev; i++)
        if (s->ev[i].from == from && s->ev[i].to == bi) return s->ev[i].slot;
    if (s->nev == s->evmax) {
        int         n  = s->evmax ? s->evmax * 2 : 32;
        sv_edgevar* ne = (sv_edgevar*)realloc(s->ev, (size_t)n * sizeof(sv_edgevar));
        if (!ne) {
            s->lower_err = 1;
            return SV_M_ZERO;
        }
        s->ev    = ne;
        s->evmax = n;
    }
    sv_edgevar* e = &s->ev[s->nev++];
    e->from = from, e->to = bi, e->slot = lw_mask(s);
    return e->slot;
}

static void lw_edge_add(sv_stage* s, int target, int from, int m)
{
    if (m == SV_M_ZERO || target <= 0 || (uint32_t)target >= s->bound || s->bix[target] < 0) return;
    int e = lw_edge(s, from, s->bix[target]);
    lw_emit(s, IR_OR, e, e, m, 0);
}

static int lw_divert(sv_stage* s, int target, int m, const sv_lc* L)
{
    for (; L; L = L->outer) {
        if (target == L->merge) {
            if (m != SV_M_ZERO) lw_emit(s, IR_OR, L->brk, L->brk, m, 0);
            return 1;
        }
        if (target == L->cont) {
            if (m != SV_M_ZERO) lw_emit(s, IR_OR, L->cm, L->cm, m, 0);
            return 1;
        }
    }
    return 0;
}

static int lw_region(sv_stage* s, int cur, int stop, int mask, sv_lc* L, int depth);

static int lw_goto(sv_stage* s, int target, int stop, int m, sv_lc* L, int depth)
{
    if (m == SV_M_ZERO) return SV_M_ZERO;
    if (target == stop) return m;
    if (lw_divert(s, target, m, L)) return SV_M_ZERO;
    return lw_region(s, target, stop, m, L, depth + 1);
}

/* a loop construct from its header; returns the slot of the lanes leaving
 * it through the merge block */
static int lw_loop(sv_stage* s, const sv_block* H, int mask, sv_lc* outer, int depth)
{
    int   active = lw_new(s, IR_COPY, mask, 0, 0);
    int   brk    = lw_new(s, IR_ZERO, 0, 0, 0);
    int   cm     = lw_mask(s);
    int   cnt    = lw_mask(s);
    sv_lc L      = { H->merge, H->cont, brk, cm, outer };
    lw_emit(s, IR_LOOP, cnt, 0, 0, 0);
    lw_emit(s, IR_BREAKZ, active, cnt, 0, 0);
    lw_emit(s, IR_ZERO, cm, 0, 0, 0);
    lw_emit(s, IR_BODY, (int)(H - s->blocks), active, 0, 0);
    const uint32_t* t = s->w + H->term;
    int             r = -1;
    if (OP(t) == OpBranch) {
        lw_edge_add(s, (int)t[1], H->label, active);
        r = lw_goto(s, (int)t[1], L.cont, active, &L, depth);
    } else if (OP(t) == OpBranchConditional) {
        int mt = lw_new(s, IR_COND, (int)t[1], active, 0);
        int mf = lw_new(s, IR_ANDN, active, mt, 0);
        lw_edge_add(s, (int)t[2], H->label, mt);
        lw_edge_add(s, (int)t[3], H->label, mf);
        int a = lw_goto(s, (int)t[2], L.cont, mt, &L, depth);
        int b = lw_goto(s, (int)t[3], L.cont, mf, &L, depth);
        r     = lw_new(s, IR_OR, a, b, 0);
    } else {
        if (OP(t) == OpKill || OP(t) == OpTerminateInvocation) lw_emit(s, IR_OR, SV_M_KILLED, SV_M_KILLED, active, 0);
        else if (OP(t) == OpReturn) lw_emit(s, IR_OR, SV_M_DONE, SV_M_DONE, active, 0);
        lw_emit(s, IR_BREAK, 0, 0, 0, 0);
    }
    if (r >= 0) {
        int cl = lw_new(s, IR_OR, r, cm, 0);
        if (L.cont == H->label) {
            lw_emit(s, IR_COPY, active, cl, 0, 0);
        } else { /* the continue construct runs back to the header */
            int   cbrk = lw_new(s, IR_ZERO, 0, 0, 0);
            sv_lc Lc   = { L.merge, -1, cbrk, SV_M_ZERO, outer };
            int   res  = lw_region(s, L.cont, H->label, cl, &Lc, depth + 1);
            lw_emit(s, IR_COPY, active, res, 0, 0);
            lw_emit(s, IR_OR, brk, brk, cbrk, 0);
        }
    }
    lw_emit(s, IR_ENDLOOP, 0, 0, 0, 0);
    return brk;
}

/* blocks from `cur` until `stop` under mask; returns the slot of the lanes
 * arriving at stop. Work after a mask can shrink is guarded by IR_IF. */
static int lw_region(sv_stage* s, int cur, int stop, int mask, sv_lc* L, int depth)
{
    if (depth > 256) { /* malformed nesting */
        s->lower_err = 2;
        return SV_M_ZERO;
    }
    int res = lw_new(s, IR_ZERO, 0, 0, 0);
    int nif = 0;
    lw_emit(s, IR_IF, mask, 0, 0, 0), nif++;
    while (!s->lower_err) {
        if (cur == stop) {
            lw_emit(s, IR_COPY, res, mask, 0, 0);
            break;
        }
        int bi = cur > 0 && (uint32_t)cur < s->bound ? s->bix[cur] : -1;
        if (bi < 0) {
            s->lower_err = 2;
            break;
        }
        const sv_block* B = &s->blocks[bi];
        if (B->loop) {
            mask = lw_loop(s, B, mask, L, depth);
            cur  = B->merge;
            if (cur != stop && lw_divert(s, cur, mask, L)) break;
            lw_emit(s, IR_IF, mask, 0, 0, 0), nif++;
            continue;
        }
        lw_emit(s, IR_BODY, bi, mask, 0, 0);
        const uint32_t* t = s->w + B->term;
        int             op = (int)OP(t), done = 0;
        switch (op) {
        case OpBranch: {
            int to = (int)t[1];
            lw_edge_add(s, to, cur, mask);
            if (to != stop && lw_divert(s, to, mask, L)) done = 1;
            cur = to;
            break;
        }
        case OpBranchConditional: {
            int mt = lw_new(s, IR_COND, (int)t[1], mask, 0);
            int mf = lw_new(s, IR_ANDN, mask, mt, 0);
            lw_edge_add(s, (int)t[2], cur, mt);
            lw_edge_add(s, (int)t[3], cur, mf);
            if (B->merge >= 0) {
                int a = lw_goto(s, (int)t[2], B->merge, mt, L, depth);
                int b = lw_goto(s, (int)t[3], B->merge, mf, L, depth);
                mask  = lw_new(s, IR_OR, a, b, 0);
                cur   = B->merge;
                lw_emit(s, IR_IF, mask, 0, 0, 0), nif++;
            } else {
                int a = lw_goto(s, (int)t[2], stop, mt, L, depth);
                int b = lw_goto(s, (int)t[3], stop, mf, L, depth);
                lw_emit(s, IR_OR, res, a, b, 0);
                done = 1;
            }
            break;
        }
        case OpSwitch: { /* lanes grouped by target; cases run to the merge (fall through works) */
            if (B->merge < 0) {
                s->lower_err = 2;
                done         = 1;
                break;
            }
            int   wc  = (int)WC(t);
            int   sbrk = lw_new(s, IR_ZERO, 0, 0, 0);
            sv_lc Sx  = { B->merge, -1, sbrk, SV_M_ZERO, L };
            int   surv;
            if (s->ids[t[1]].cls == C_CONST) { /* e.g. the switch (0) construct of inlined early returns */
                uint32_t sel = sv_cblock(s, &s->ids[t[1]])[0];
                int      T   = (int)t[2];
                for (int k = 3; k + 1 < wc; k += 2)
                    if (sel == t[k]) {
                        T = (int)t[k + 1];
                        break;
                    }
                lw_edge_add(s, T, cur, mask);
                surv = lw_goto(s, T, B->merge, mask, &Sx, depth);
            } else {
                /* distinct targets in case order, then the default; the first
                 * case with a literal wins */
                int tg[64], tm[64], nt = 0;
                int any = lw_new(s, IR_ZERO, 0, 0, 0);
                for (int k = 3; k + 1 < wc && !s->lower_err; k += 2) {
                    int dup = 0;
                    for (int j = 3; j < k; j += 2) dup |= t[j] == t[k];
                    if (dup) continue;
                    int T = (int)t[k + 1], ti = -1;
                    for (int j = 0; j < nt; j++)
                        if (tg[j] == T) ti = j;
                    if (ti < 0) {
                        if (nt == 64) {
                            s->lower_err = 2;
                            break;
                        }
                        ti = nt++, tg[ti] = T, tm[ti] = lw_new(s, IR_ZERO, 0, 0, 0);
                    }
                    int e = lw_new(s, IR_EQ, (int)t[1], mask, t[k]);
                    lw_emit(s, IR_OR, tm[ti], tm[ti], e, 0);
                    lw_emit(s, IR_OR, any, any, e, 0);
                }
                int md = lw_new(s, IR_ANDN, mask, any, 0), ti = -1;
                for (int j = 0; j < nt; j++)
                    if (tg[j] == (int)t[2]) ti = j;
                if (ti < 0 && nt < 64) ti = nt++, tg[ti] = (int)t[2], tm[ti] = lw_new(s, IR_ZERO, 0, 0, 0);
                if (ti >= 0) lw_emit(s, IR_OR, tm[ti], tm[ti], md, 0);
                surv = lw_new(s, IR_ZERO, 0, 0, 0);
                for (int j = 0; j < nt; j++) {
                    lw_edge_add(s, tg[j], cur, tm[j]);
                    int r = lw_goto(s, tg[j], B->merge, tm[j], &Sx, depth);
                    lw_emit(s, IR_OR, surv, surv, r, 0);
                }
            }
            mask = lw_new(s, IR_OR, surv, sbrk, 0);
            cur  = B->merge;
            lw_emit(s, IR_IF, mask, 0, 0, 0), nif++;
            break;
        }
        case OpKill: case OpTerminateInvocation:
            lw_emit(s, IR_OR, SV_M_KILLED, SV_M_KILLED, mask, 0);
            done = 1;
            break;
        case OpReturn:
            lw_emit(s, IR_OR, SV_M_DONE, SV_M_DONE, mask, 0);
            done = 1;
            break;
        default: done = 1; break; /* OpUnreachable */
        }
        if (done) break;
    }
    while (nif--) lw_emit(s, IR_ENDIF, 0, 0, 0, 0);
    return res;
}

/* jumps, the phi edge table and per block edge lists */
static int lw_finish(sv_stage* s, char* err, size_t errn)
{
    int stack[512], sp = 0;
    for (int i = 0; i < s->nir; i++) {
        sv_ir* o = &s->ir[i];
        switch (o->op) {
        case IR_IF:
        case IR_LOOP:
            if (sp == 512) return sv_err(err, errn, "control flow nested too deeply");
            stack[sp++] = i;
            break;
        case IR_ENDIF: case IR_ENDLOOP: {
            if (!sp) return sv_err(err, errn, "internal: unbalanced lowering");
            int open = stack[--sp];
            s->ir[open].jump = i;
            if (o->op == IR_ENDLOOP) o->jump = open + 1;
            break;
        }
        default: break;
        }
    }
    for (int i = 0; i < s->nir; i++) { /* breaks: past the enclosing loop's end */
        sv_ir* o = &s->ir[i];
        if (o->op != IR_BREAKZ && o->op != IR_BREAK) continue;
        int depth = 0, j = i + 1;
        for (; j < s->nir; j++) {
            if (s->ir[j].op == IR_LOOP) depth++;
            else if (s->ir[j].op == IR_ENDLOOP && depth-- == 0) break;
        }
        o->jump = j + 1;
    }
    for (int bi = 0; bi < s->nblocks; bi++) {
        sv_block* B = &s->blocks[bi];
        int       np = 0;
        for (int ii = 0; ii < B->nphi; ii++) np += ((int)WC(s->w + B->insts[ii].at) - 3) / 2;
        B->phim = (int*)malloc((size_t)(np + 1) * sizeof(int));
        B->clr  = (int*)malloc((size_t)(s->nev + 1) * sizeof(int));
        if (!B->phim || !B->clr) return sv_err(err, errn, "out of memory");
        np = 0;
        for (int ii = 0; ii < B->nphi; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            for (int k = 3; k + 1 < (int)WC(in); k += 2) {
                int slot = SV_M_ZERO;
                for (int e = 0; e < s->nev; e++)
                    if (s->ev[e].to == bi && s->ev[e].from == (int)in[k + 1]) slot = s->ev[e].slot;
                B->phim[np++] = slot;
            }
        }
        B->nclr = 0;
        for (int e = 0; e < s->nev; e++)
            if (s->ev[e].to == bi) B->clr[B->nclr++] = s->ev[e].slot;
    }
    return 1;
}

static int sv_lower(sv_stage* s, char* err, size_t errn)
{
    s->nmask = SV_M_ENTRY + 1;
    lw_region(s, s->blocks[0].label, -1, SV_M_ENTRY, NULL, 0);
    if (s->lower_err == 1) return sv_err(err, errn, "out of memory (or the control flow is too large to lower)");
    if (s->lower_err) return sv_err(err, errn, "unsupported control flow structure");
    return lw_finish(s, err, errn);
}

/* ---- value classes ------------------------------------------------------------ */

int sv_is_cross_lane(int op)
{
    switch (op) {
    case OpDPdx: case OpDPdy: case OpFwidth: case OpDPdxFine: case OpDPdyFine: case OpFwidthFine: case OpDPdxCoarse:
    case OpDPdyCoarse: case OpFwidthCoarse: case OpImageSampleImplicitLod: case OpImageSampleExplicitLod:
        return 1;
    default: return 0;
    }
}

/* per lane and side effect free: lane uniform when every operand is */
static int sv_is_pure(int op)
{
    switch (op) {
    case OpLoad: case OpStore: case OpPhi: case OpVariable: case OpUndef: case OpAccessChain: case OpInBoundsAccessChain:
    case OpSampledImage:
        return 0;
    default: return !sv_is_cross_lane(op);
    }
}

static int sv_add_ptr_uses(const sv_stage* s, int ptr, int* ids, int n, int max)
{
    const sv_id* P = &s->ids[ptr];
    if (P->cls == C_PTR)
        for (int k = 0; k < P->ndyn && n < max; k++) ids[n++] = P->dyn[k];
    return n;
}

int sv_operands(const sv_stage* s, const uint32_t* in, int op, int* ids, int max)
{
    int n = 0, wc = (int)WC(in), first = 3, last = wc;
    switch (op) {
    case OpLoad: return sv_add_ptr_uses(s, (int)in[3], ids, 0, max);
    case OpStore:
        n = sv_add_ptr_uses(s, (int)in[1], ids, 0, max);
        if (n < max) ids[n++] = (int)in[2];
        return n;
    case OpPhi:
        for (int k = 3; k + 1 < wc && n < max; k += 2) ids[n++] = (int)in[k];
        return n;
    case OpCompositeExtract: last = 4; break;
    case OpCompositeInsert: case OpVectorShuffle: last = 5; break;
    case OpExtInst: first = 5; break;
    case OpImageSampleImplicitLod: case OpImageSampleExplicitLod: /* sampler (unit), coordinate, operands mask, ids */
        if (n < max) ids[n++] = (int)in[4];
        for (int k = 6; k < wc && n < max; k++) ids[n++] = (int)in[k];
        return n;
    default: break;
    }
    for (int k = first; k < last && n < max; k++) {
        uint32_t v = in[k];
        if (v < s->bound && (s->ids[v].cls == C_VALUE || s->ids[v].cls == C_CONST)) ids[n++] = (int)v;
    }
    return n;
}

static void sv_mat(sv_stage* s, int v, int bi, int func)
{
    if (s->vcls[v] != VC_LOCAL && s->vcls[v] != VC_MAT) return;
    s->vcls[v] = VC_MAT;
    if (func || s->vblock[v] != bi) s->vscope[v] = VS_FUNC;
}

/* constants / lane uniform values / segment locals / arrays, and the
 * segments; needs the lowered program (conditions read their values) */
static int sv_classify(sv_stage* s, char* err, size_t errn)
{
    s->vcls   = (uint8_t*)calloc(s->bound, 1);
    s->vscope = (uint8_t*)calloc(s->bound, 1);
    s->vseg   = (int*)malloc(s->bound * sizeof(int));
    s->vblock = (int*)malloc(s->bound * sizeof(int));
    if (!s->vcls || !s->vscope || !s->vseg || !s->vblock) return sv_err(err, errn, "out of memory");
    for (uint32_t i = 0; i < s->bound; i++) {
        s->vseg[i] = s->vblock[i] = -1;
        if (s->ids[i].cls == C_CONST) s->vcls[i] = VC_CONST;
    }
    int ops[64], nseg = 0;
    /* definitions (blocks are in dominance order, so operands are classified first) */
    for (int bi = 0; bi < s->nblocks; bi++) {
        sv_block* B = &s->blocks[bi];
        B->seg      = (int*)malloc(((size_t)B->ninst + 1) * sizeof(int));
        if (!B->seg) return sv_err(err, errn, "out of memory");
        int phiseg = nseg++, cur = nseg++;
        for (int ii = 0; ii < B->ninst; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             op = B->insts[ii].op;
            if (op == OpPhi) {
                B->seg[ii]         = phiseg;
                s->vcls[in[2]]     = VC_MAT;
                s->vscope[in[2]]   = VS_FUNC;
                s->vseg[in[2]]     = phiseg;
                s->vblock[in[2]]   = bi;
                continue;
            }
            int cross = sv_is_cross_lane(op);
            if (cross) B->seg[ii] = nseg++, cur = nseg++;
            else B->seg[ii] = cur;
            if (op == OpStore) continue;
            int v = (int)in[2], uni = 0;
            if (sv_is_pure(op)) {
                int n = sv_operands(s, in, op, ops, 64);
                uni   = n < 64;
                for (int k = 0; k < n; k++) uni &= s->vcls[ops[k]] == VC_CONST || s->vcls[ops[k]] == VC_UNIFORM;
            } else if (op == OpLoad) {
                const sv_id* P = &s->ids[in[3]];
                int          n = sv_operands(s, in, op, ops, 64);
                uni            = sv_is_ptr_storage_uniform(s->ids[P->pvar].storage);
                for (int k = 0; k < n; k++) uni &= s->vcls[ops[k]] == VC_CONST || s->vcls[ops[k]] == VC_UNIFORM;
            }
            s->vcls[v]   = cross ? VC_MAT : (uni ? VC_UNIFORM : VC_LOCAL);
            s->vscope[v] = VS_BLOCK;
            s->vseg[v]   = B->seg[ii];
            s->vblock[v] = bi;
        }
    }
    s->nseg = nseg;
    /* uses: anything read outside its segment, by a cross lane instruction or
     * a phi is kept as an array */
    for (int bi = 0; bi < s->nblocks; bi++) {
        const sv_block* B = &s->blocks[bi];
        for (int ii = 0; ii < B->ninst; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             op = B->insts[ii].op;
            int             n  = sv_operands(s, in, op, ops, 64);
            for (int k = 0; k < n; k++) {
                int v = ops[k];
                if (op == OpPhi) sv_mat(s, v, bi, 1);
                else if (sv_is_cross_lane(op) || s->vseg[v] != B->seg[ii]) sv_mat(s, v, bi, 0);
            }
        }
    }
    for (int i = 0; i < s->nir; i++) /* branch conditions, switch selectors */
        if (s->ir[i].op == IR_COND || s->ir[i].op == IR_EQ) sv_mat(s, s->ir[i].b, -1, 1);
    return 1;
}

/* ---- execution ------------------------------------------------------------------ */

typedef struct sv_exec {
    const sv_stage*            s;
    uint32_t*                  x;   /* scratch lane blocks */
    uint32_t*                  tmp; /* phi staging */
    const uint8_t*             ubo;
    size_t                     ubo_n;
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


static void sv_sample(sv_exec* E, const uint32_t* in, int explicit_lod)
{
    const sv_id* img = &E->s->ids[in[3]];
    float*       out = RF(in[2]);
    const float* cu  = RF(in[4]);
    const float* cv  = cu + SV_L;
    const fm3d_texture* t = img->cls == C_IMG && img->unit >= 0 ? E->tex[img->unit] : NULL;
    if (!t) {
        for (int k = 0; k < 4 * SV_L; k++) out[k] = k >= 3 * SV_L ? 1.0f : 0.0f;
        return;
    }
    const fm3d_sampler* s = &E->samp[img->unit];
    int                 wc = (int)WC(in);
    if (explicit_lod) {
        const float* lod = NULL;
        if (wc >= 7 && (in[5] & 2u)) lod = RF(in[6]); /* ImageOperands Lod */
        fm3d_sample_lod(t, s, cu, cv, lod, SV_L, out, out + SV_L, out + 2 * SV_L, out + 3 * SV_L);
        return;
    }
    if (E->fio) {
        fm3d_sample_batch(E->fio, t, s, cu, cv, out, out + SV_L, out + 2 * SV_L, out + 3 * SV_L);
        return;
    }
    fm3d_sample_lod(t, s, cu, cv, NULL, SV_L, out, out + SV_L, out + 2 * SV_L, out + 3 * SV_L); /* vertex stage: base level */
}

/* ---- GLSL.std.450 ---- */

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
    int k, N = n * SV_L;
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
    case 13: for (k = 0; k < N; k++) f[k] = fm_sinf(a[k]); break;
    case 14: for (k = 0; k < N; k++) f[k] = fm_cosf(a[k]); break;
    case 15: for (k = 0; k < N; k++) f[k] = fm_tanf(a[k]); break;
    case 16: for (k = 0; k < N; k++) f[k] = asinf(a[k]); break;
    case 17: for (k = 0; k < N; k++) f[k] = acosf(a[k]); break;
    case 18: for (k = 0; k < N; k++) f[k] = atanf(a[k]); break;
    case 19: for (k = 0; k < N; k++) f[k] = sinhf(a[k]); break;
    case 20: for (k = 0; k < N; k++) f[k] = coshf(a[k]); break;
    case 21: for (k = 0; k < N; k++) f[k] = tanhf(a[k]); break;
    case 25: for (k = 0; k < N; k++) f[k] = atan2f(a[k], b[k]); break;
    case 26: for (k = 0; k < N; k++) f[k] = fm_powf(a[k], b[k]); break;
    case 27: for (k = 0; k < N; k++) f[k] = fm_expf(a[k]); break;
    case 28: for (k = 0; k < N; k++) f[k] = fm_logf(a[k]); break;
    case 29: for (k = 0; k < N; k++) f[k] = fm_exp2f(a[k]); break;
    case 30: for (k = 0; k < N; k++) f[k] = fm_log2f(a[k]); break;
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
        case OpImageSampleImplicitLod: sv_sample(E, in, 0); break;
        case OpImageSampleExplicitLod: sv_sample(E, in, 1); break;
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
                    float ddx = v[l | 1] - v[l & ~1], ddy = v[(l & 31) + 32] - v[l & 31];
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

/* ---- per thread scratch ---- */

static SV_TLS uint32_t* sv_scr;
static SV_TLS size_t    sv_scap;

static SV_TLS uint64_t* sv_masks;
static SV_TLS int       sv_mcap;

static uint64_t* sv_mask_scratch(int n)
{
    if (n > sv_mcap) {
        free(sv_masks);
        sv_masks = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
        sv_mcap  = sv_masks ? n : 0;
    }
    return sv_mcap >= n ? sv_masks : NULL;
}

static uint32_t* sv_scratch(size_t blocks)
{
    size_t need = blocks * SV_L;
    if (need > sv_scap) {
        free(sv_scr);
        sv_scr  = (uint32_t*)malloc(need * sizeof(uint32_t));
        sv_scap = sv_scr ? need : 0;
    }
    return sv_scr;
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

static void sv_run_vs(const fm3d_vs_io* io)
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
        E.tex   = io->textures;
        E.samp  = io->samplers;
        for (int i = 0; i < s->nin; i++) { /* vertex attributes */
            const sv_io* vi = &s->in[i];
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

static void sv_run_fs(const fm3d_fs_io* io)
{
    const fm3d_spirv* P = (const fm3d_spirv*)io->user;
    const sv_stage*   s = P->fs;
    sv_exec           E;
    if (!sv_setup(&E, s)) return;
    E.ubo   = (const uint8_t*)io->uniforms;
    E.ubo_n = io->uniforms ? io->uniform_size : 0;
    E.fio   = io;
    E.tex   = io->textures;
    E.samp  = io->samplers;
    uint64_t lanes = 0; /* every pixel of the batch's quads (helpers included) */
    FOR_L if ((l & 31) < io->cols) lanes |= 1ull << l;
    for (int i = 0; i < s->nin; i++) {
        const sv_io* fi = &s->in[i];
        float*       d = (float*)(E.x + (size_t)s->ids[fi->var].reg * SV_L);
        if (fi->builtin == BI_FragCoord) {
            FOR_L
            {
                d[l]            = (float)(io->x + (l & 31)) + 0.5f;
                d[SV_L + l]     = (float)(io->y + (l >> 5)) + 0.5f;
                d[2 * SV_L + l] = io->z ? io->z[l] : 0.0f;
                d[3 * SV_L + l] = 1.0f;
            }
            continue;
        }
        if (fi->builtin == BI_FrontFacing) {
            FOR_L((uint32_t*)d)[l] = 1u;
            continue;
        }
        int slot = P->fslot[i];
        for (int c = 0; c < fi->comps; c++) {
            if (slot < 0 || slot + c >= FM3D_MAX_SHADER_VARYINGS) {
                FOR_L d[(size_t)c * SV_L + l] = 0.0f;
                continue;
            }
            memcpy(d + (size_t)c * SV_L, io->varyings[slot + c], SV_L * sizeof(float));
        }
    }
    E.M[SV_M_ENTRY] = lanes;
    sv_ir_run(&E);
    for (int i = 0; i < s->nout; i++) {
        const sv_io* fo = &s->out[i];
        if (fo->loc != 0) continue;
        const float* src = (const float*)(E.x + (size_t)s->ids[fo->var].reg * SV_L);
        for (int c = 0; c < 4; c++) {
            if (c < fo->comps) memcpy(io->out[c], src + (size_t)c * SV_L, SV_L * sizeof(float));
            else FOR_L io->out[c][l] = c == 3 ? 1.0f : 0.0f;
        }
    }
    FOR_L if ((E.M[SV_M_KILLED] >> l) & 1) io->mask[l] = 0;
}

/* ---- program creation / linking ---------------------------------------------- */

static int sv_has_kill(const sv_stage* s)
{
    for (int bi = 0; bi < s->nblocks; bi++) {
        int op = (int)OP(s->w + s->blocks[bi].term);
        if (op == OpKill || op == OpTerminateInvocation) return 1;
    }
    return 0;
}

fm3d_spirv* fm3d_spirv_create(const uint32_t* vs, size_t vs_words, const uint32_t* fs, size_t fs_words,
                              const fm3d_vertex_attrib* attribs, int nattribs, char* err, size_t errn)
{
    if (err && errn) err[0] = 0;
    if (!vs && !fs) return (fm3d_spirv*)(uintptr_t)sv_err(err, errn, "no shader stages");
    fm3d_spirv* P = (fm3d_spirv*)calloc(1, sizeof(fm3d_spirv));
    if (!P) return NULL;
    char e2[256];
    if (vs) {
        e2[0] = 0;
        P->vs = sv_parse(vs, vs_words, 0, e2, sizeof(e2));
        if (!P->vs || !sv_analyze(P->vs, e2, sizeof(e2)) || !sv_lower(P->vs, e2, sizeof(e2)) ||
            !sv_classify(P->vs, e2, sizeof(e2))) {
            sv_err(err, errn, "vertex shader: %s", e2);
            goto fail;
        }
    }
    if (fs) {
        e2[0] = 0;
        P->fs = sv_parse(fs, fs_words, 4, e2, sizeof(e2));
        if (!P->fs || !sv_analyze(P->fs, e2, sizeof(e2)) || !sv_lower(P->fs, e2, sizeof(e2)) ||
            !sv_classify(P->fs, e2, sizeof(e2))) {
            sv_err(err, errn, "fragment shader: %s", e2);
            goto fail;
        }
    }
    if (nattribs > 16) nattribs = 16;
    for (int i = 0; i < nattribs; i++) P->attr[i] = attribs[i];
    P->nattr = nattribs;
    /* varying slots: vertex outputs by location, fragment inputs matched */
    for (int i = 0; i < 32; i++) P->vslot[i] = P->fslot[i] = -1;
    if (P->vs) {
        for (int i = 0; i < P->vs->nin; i++) {
            int found = 0;
            for (int k = 0; k < nattribs; k++) found |= attribs[k].location == P->vs->in[i].loc;
            if (!found) {
                sv_err(err, errn, "vertex input location %d has no attribute", P->vs->in[i].loc);
                goto fail;
            }
        }
        int slot = 0;
        for (int loc = 0; loc < 32; loc++)
            for (int i = 0; i < P->vs->nout; i++) {
                const sv_io* o = &P->vs->out[i];
                if (o->builtin >= 0 || o->loc != loc) continue;
                if (!P->fs) { /* fixed fragment stage: location 0 = uv, 1 = rgba */
                    if (loc > 1) continue;
                    P->vslot[i] = loc == 0 ? 0 : 2;
                    continue;
                }
                P->vslot[i] = slot;
                slot += o->comps;
            }
        P->nvar = P->fs ? slot : 6;
        if (P->nvar > FM3D_MAX_SHADER_VARYINGS) {
            sv_err(err, errn, "more than %d varying components", FM3D_MAX_SHADER_VARYINGS);
            goto fail;
        }
    } else {
        P->nvar = 6; /* fixed vertex stage */
    }
    if (P->fs)
        for (int i = 0; i < P->fs->nin; i++) {
            const sv_io* fi = &P->fs->in[i];
            if (fi->builtin >= 0) continue;
            if (!P->vs) {
                if (fi->loc == 0) P->fslot[i] = 0;
                else if (fi->loc == 1) P->fslot[i] = 2;
                else {
                    sv_err(err, errn, "fragment input location %d: the fixed vertex stage has 0 (uv) and 1 (color)", fi->loc);
                    goto fail;
                }
                continue;
            }
            for (int k = 0; k < P->vs->nout; k++)
                if (P->vs->out[k].builtin < 0 && P->vs->out[k].loc == fi->loc) {
                    if (P->vs->out[k].comps < fi->comps) {
                        sv_err(err, errn, "location %d: vertex output smaller than fragment input", fi->loc);
                        goto fail;
                    }
                    P->fslot[i] = P->vslot[k];
                }
            if (P->fslot[i] < 0) {
                sv_err(err, errn, "fragment input location %d is not written by the vertex shader", fi->loc);
                goto fail;
            }
        }
    return P;
fail:
    fm3d_spirv_destroy(P);
    return NULL;
}

void fm3d_spirv_destroy(fm3d_spirv* P)
{
    if (!P) return;
    sv_stage_free(P->vs);
    sv_stage_free(P->fs);
    free(P);
}

fm3d_program fm3d_spirv_program(const fm3d_spirv* P)
{
    fm3d_program p;
    memset(&p, 0, sizeof(p));
    if (!P) return p;
    p.vs        = P->vs ? sv_run_vs : NULL;
    p.fs        = P->fs ? sv_run_fs : NULL;
    p.nvaryings = P->nvar;
    p.discards  = P->fs ? sv_has_kill(P->fs) : 0;
    p.user      = (void*)P;
    return p;
}

#endif /* FM_FEATURE_SPIRV */
