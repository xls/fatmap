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
 * Supported: GLSL.std.450 vertex / fragment shaders (function calls are
 * inlined first, fm3d_spirv_inline.c): scalars, vectors, matrices, arrays, structs, one
 * uniform blocks by binding (std140 offsets from the decorations; push
 * constants read block 0),
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
    free(s->vars);
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
    s->ps_var  = -1;
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
            case OpTypeImage: T->kind = T_IMAGE, T->dim = (uint8_t)in[3], T->arrayed = (uint8_t)in[5]; break;
            case OpTypeSampledImage:
                T->kind = T_SIMAGE;
                if (in[2] < s->bound) T->dim = s->ids[in[2]].dim, T->arrayed = s->ids[in[2]].arrayed;
                break;
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
                if (s->nvars == s->varcap) {
                    int  nc = s->varcap ? s->varcap * 2 : 64;
                    int* nv = (int*)realloc(s->vars, (size_t)nc * sizeof(int));
                    if (!nv) goto fail;
                    s->vars = nv, s->varcap = nc;
                }
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
            sv_err(err, errn, "internal: a function call survived inlining");
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


static const char* sv_op_ok(sv_stage* s, const uint32_t* in, int op)
{
    switch (op) {
    case OpLoad: case OpStore: case OpAccessChain: case OpInBoundsAccessChain: case OpVectorExtractDynamic:
    case OpVectorInsertDynamic: case OpVectorShuffle: case OpCompositeConstruct: case OpCompositeExtract:
    case OpCompositeInsert: case OpCopyObject: case OpTranspose: case OpSampledImage: case OpImageSampleImplicitLod:
    case OpImageSampleExplicitLod: case OpImageSampleProjImplicitLod: case OpImageSampleProjExplicitLod: case OpConvertFToU: case OpConvertFToS: case OpConvertSToF: case OpConvertUToF:
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
                for (int m = 0; m < T->nmem; m++) {
                    if (T->mbuiltin[m] == BI_Position && v->storage == SC_Output) s->pos_var = (int)i, s->pos_off = T->moff[m];
                    if (T->mbuiltin[m] == BI_PointSize && v->storage == SC_Output) s->ps_var = (int)i, s->ps_off = T->moff[m];
                }
                continue;
            }
            if (v->builtin == BI_Position && v->storage == SC_Output) {
                s->pos_var = (int)i, s->pos_off = 0;
                continue;
            }
            if (v->builtin == BI_PointSize && v->storage == SC_Output) {
                s->ps_var = (int)i, s->ps_off = 0;
                continue;
            }
            int vsid = s->model == 0 && (v->builtin == BI_VertexIndex || v->builtin == BI_InstanceIndex);
            if (v->builtin >= 0 && !vsid && v->builtin != BI_FragCoord && v->builtin != BI_FrontFacing && v->builtin != BI_PointSize &&
                v->builtin != BI_PointCoord && !(v->builtin == BI_FragDepth && s->model == 4) &&
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
        } else if (v->storage == SC_Uniform && v->binding >= FM3D_MAX_UNIFORM_BLOCKS) {
            return sv_err(err, errn, "uniform block binding %d out of range (0..%d)", v->binding, FM3D_MAX_UNIFORM_BLOCKS - 1);
        } else if (v->storage == SC_UniformConstant) {
            if (s->ids[v->type].kind != T_SIMAGE) return sv_err(err, errn, "only sampler2D uniforms");
            if (v->binding < 0 || v->binding >= FM3D_MAX_TEXTURE_UNITS) return sv_err(err, errn, "sampler binding out of range");
        }
    }
    if (s->model == 0 && s->pos_var < 0) return sv_err(err, errn, "vertex shader does not write gl_Position");

    /* function body: validate, give results registers, resolve pointers */
    int maxphi = 0, ps_written = 0;
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
            if (op == OpStore && s->ps_var >= 0 && in[1] < s->bound) { /* glslang declares gl_PointSize in every gl_PerVertex */
                const sv_id* ptr = &s->ids[in[1]];
                if ((int)in[1] == s->ps_var || (ptr->pvar == s->ps_var && ptr->poff == s->ps_off && ptr->ndyn == 0)) ps_written = 1;
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
                        d->dim = s->ids[rt].dim, d->arrayed = s->ids[rt].arrayed;
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
    if (!ps_written) s->ps_var = -1;
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
    case OpImageSampleProjImplicitLod: case OpImageSampleProjExplicitLod:
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
    case OpImageSampleProjImplicitLod: case OpImageSampleProjExplicitLod:
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

/* the lowered program around block `bi`'s executions: is every branch in
 * [lo, hi) of the IR uniform? */
static int sv_branches_uniform(const sv_stage* s, const uint8_t* uni, int lo, int hi)
{
    for (int i = lo; i < hi; i++)
        if ((s->ir[i].op == IR_COND || s->ir[i].op == IR_EQ) && !uni[s->ir[i].b]) return 0;
    return 1;
}

/* can phi block bi give every lane the same value? Loop headers / loop
 * merges: every branch of the loop (its exits) is uniform; selection
 * merges: every branch from the header to the merge is uniform. Checked at
 * every place the lowering emitted the block. */
static int sv_join_uniform(const sv_stage* s, const uint8_t* uni, int bi)
{
    const sv_block* B = &s->blocks[bi];
    int             hdr = -1, found = 0;
    if (!B->loop)
        for (int h = 0; h < s->nblocks && hdr < 0; h++)
            if (s->blocks[h].merge == B->label) hdr = h;
    for (int j = 0; j < s->nir; j++) {
        if (s->ir[j].op != IR_BODY || s->ir[j].a != bi) continue;
        found = 1;
        if (B->loop || (hdr >= 0 && s->blocks[hdr].loop)) { /* the loop around (header) / before (merge) */
            int depth = 0, lo = -1;
            if (B->loop) {
                for (int i = j; i >= 0 && lo < 0; i--) {
                    if (s->ir[i].op == IR_ENDLOOP) depth++;
                    else if (s->ir[i].op == IR_LOOP && depth-- == 0) lo = i;
                }
            } else { /* the merge runs after the loop whose header is hdr */
                for (int i = j; i >= 0 && lo < 0; i--)
                    if (s->ir[i].op == IR_BODY && s->ir[i].a == hdr)
                        for (int k = i; k >= 0 && lo < 0; k--)
                            if (s->ir[k].op == IR_LOOP) lo = k;
            }
            if (lo < 0 || !sv_branches_uniform(s, uni, lo, s->ir[lo].jump)) return 0;
        } else if (hdr >= 0) { /* selection merge: from the header's last run before it */
            int i = j;
            while (i >= 0 && !(s->ir[i].op == IR_BODY && s->ir[i].a == hdr)) i--;
            if (i < 0 || !sv_branches_uniform(s, uni, i, j)) return 0;
        } else {
            return 0;
        }
    }
    return found;
}

/* constants / lane uniform values / segment locals / arrays, and the
 * segments; needs the lowered program (conditions read their values).
 * Uniformity is a fixed point from the optimistic start (every phi
 * uniform): a phi stays uniform while its incoming values are and the
 * branches that choose between them are (divergence analysis). */
static int sv_classify(sv_stage* s, char* err, size_t errn)
{
    s->vcls   = (uint8_t*)calloc(s->bound, 1);
    s->vscope = (uint8_t*)calloc(s->bound, 1);
    s->vseg   = (int*)malloc(s->bound * sizeof(int));
    s->vblock = (int*)malloc(s->bound * sizeof(int));
    uint8_t* uni  = (uint8_t*)calloc(s->bound, 1);
    uint8_t* phiu = (uint8_t*)calloc(s->bound, 1);
    if (!s->vcls || !s->vscope || !s->vseg || !s->vblock || !uni || !phiu) {
        free(uni);
        free(phiu);
        return sv_err(err, errn, "out of memory");
    }
    for (uint32_t i = 0; i < s->bound; i++) {
        s->vseg[i] = s->vblock[i] = -1;
        if (s->ids[i].cls == C_CONST) s->vcls[i] = VC_CONST, uni[i] = 1;
    }
    int ops[64], nseg = 0;
    /* segments, defining blocks */
    for (int bi = 0; bi < s->nblocks; bi++) {
        sv_block* B = &s->blocks[bi];
        B->seg      = (int*)malloc(((size_t)B->ninst + 1) * sizeof(int));
        if (!B->seg) {
            free(uni);
            free(phiu);
            return sv_err(err, errn, "out of memory");
        }
        int phiseg = nseg++, cur = nseg++;
        for (int ii = 0; ii < B->ninst; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             op = B->insts[ii].op;
            if (op == OpPhi) {
                B->seg[ii] = phiseg;
                phiu[in[2]] = 1;
            } else if (sv_is_cross_lane(op)) {
                B->seg[ii] = nseg++, cur = nseg++;
            } else {
                B->seg[ii] = cur;
            }
            if (op == OpStore) continue;
            s->vseg[in[2]]   = B->seg[ii];
            s->vblock[in[2]] = bi;
        }
    }
    s->nseg = nseg;
    /* uniformity: blocks are in dominance order, so operands come first */
    for (int round = 0; round < 64; round++) {
        for (int bi = 0; bi < s->nblocks; bi++) {
            const sv_block* B = &s->blocks[bi];
            for (int ii = 0; ii < B->ninst; ii++) {
                const uint32_t* in = s->w + B->insts[ii].at;
                int             op = B->insts[ii].op, u = 0;
                if (op == OpStore) continue;
                if (op == OpPhi) {
                    u = phiu[in[2]];
                } else if (sv_is_pure(op) || op == OpLoad) {
                    int n = sv_operands(s, in, op, ops, 64);
                    u     = n < 64 && (op != OpLoad || sv_is_ptr_storage_uniform(s->ids[s->ids[in[3]].pvar].storage));
                    for (int k = 0; k < n; k++) u &= uni[ops[k]];
                }
                uni[in[2]] = (uint8_t)u;
            }
        }
        int changed = 0; /* demote phis whose values or choice can differ per lane */
        for (int bi = 0; bi < s->nblocks; bi++) {
            const sv_block* B = &s->blocks[bi];
            int             ju = -1;
            for (int ii = 0; ii < B->nphi; ii++) {
                const uint32_t* in = s->w + B->insts[ii].at;
                if (!phiu[in[2]]) continue;
                int ok = 1;
                for (int k = 3; k + 1 < (int)WC(in); k += 2) ok &= uni[in[k]];
                if (ok) {
                    if (ju < 0) ju = sv_join_uniform(s, uni, bi);
                    ok = ju;
                }
                if (!ok) phiu[in[2]] = 0, changed = 1;
            }
        }
        if (!changed) break;
    }
    /* any loop whose exits can differ per lane (a backend may run smaller lane groups then) */
    s->divergent = 0;
    for (int i = 0; i < s->nir; i++)
        if (s->ir[i].op == IR_LOOP && !sv_branches_uniform(s, uni, i, s->ir[i].jump)) s->divergent = 1;
    /* classes */
    for (int bi = 0; bi < s->nblocks; bi++) {
        const sv_block* B = &s->blocks[bi];
        for (int ii = 0; ii < B->ninst; ii++) {
            const uint32_t* in = s->w + B->insts[ii].at;
            int             op = B->insts[ii].op, v;
            if (op == OpStore) continue;
            v = (int)in[2];
            if (uni[v]) s->vcls[v] = VC_UNIFORM;
            else if (op == OpPhi) s->vcls[v] = VC_MAT, s->vscope[v] = VS_FUNC;
            else s->vcls[v] = sv_is_cross_lane(op) ? VC_MAT : VC_LOCAL, s->vscope[v] = VS_BLOCK;
        }
    }
    free(uni);
    free(phiu);
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

/* ---- per thread scratch (shared by the executors of every ISA) ---- */

static SV_TLS uint32_t* sv_scr;
static SV_TLS size_t    sv_scap;

static SV_TLS uint64_t* sv_masks;
static SV_TLS int       sv_mcap;

uint64_t* sv_mask_scratch(int n)
{
    if (n > sv_mcap) {
        free(sv_masks);
        sv_masks = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
        sv_mcap  = sv_masks ? n : 0;
    }
    return sv_mcap >= n ? sv_masks : NULL;
}

uint32_t* sv_scratch(size_t blocks)
{
    size_t need = blocks * SV_L;
    if (need > sv_scap) {
        free(sv_scr);
        sv_scr  = (uint32_t*)malloc(need * sizeof(uint32_t));
        sv_scap = sv_scr ? need : 0;
    }
    return sv_scr;
}

/* ---- the executors: baseline here, AVX2 / AVX-512 in their own units ---- */

#define SV_FN(name) name##_base
#include "fm3d_spirv_exec.h"
#undef SV_FN

static void sv_run_vs(const fm3d_vs_io* io)
{
    switch (fm_simd_current()) {
#ifdef FM_HAVE_AVX512_SPIRV
    case FM_SIMD_AVX512: sv_run_vs_avx512(io); return;
#endif
#ifdef FM_HAVE_AVX2
    case FM_SIMD_AVX2: sv_run_vs_avx2(io); return;
#endif
    default: sv_run_vs_base(io); return;
    }
}

static void sv_run_fs(const fm3d_fs_io* io)
{
    switch (fm_simd_current()) {
#ifdef FM_HAVE_AVX512_SPIRV
    case FM_SIMD_AVX512: sv_run_fs_avx512(io); return;
#endif
#ifdef FM_HAVE_AVX2
    case FM_SIMD_AVX2: sv_run_fs_avx2(io); return;
#endif
    default: sv_run_fs_base(io); return;
    }
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
        e2[0]          = 0;
        size_t    nw   = 0;
        uint32_t* inl  = sv_inline_calls(vs, vs_words, &nw, e2, sizeof(e2)); /* function calls */
        if (!inl && e2[0]) {
            sv_err(err, errn, "vertex shader: %s", e2);
            goto fail;
        }
        P->vs = sv_parse(inl ? inl : vs, inl ? nw : vs_words, 0, e2, sizeof(e2));
        free(inl);
        if (!P->vs || !sv_analyze(P->vs, e2, sizeof(e2)) || !sv_lower(P->vs, e2, sizeof(e2)) ||
            !sv_classify(P->vs, e2, sizeof(e2))) {
            sv_err(err, errn, "vertex shader: %s", e2);
            goto fail;
        }
    }
    if (fs) {
        e2[0]          = 0;
        size_t    nw   = 0;
        uint32_t* inl  = sv_inline_calls(fs, fs_words, &nw, e2, sizeof(e2));
        if (!inl && e2[0]) {
            sv_err(err, errn, "fragment shader: %s", e2);
            goto fail;
        }
        P->fs = sv_parse(inl ? inl : fs, inl ? nw : fs_words, 4, e2, sizeof(e2));
        free(inl);
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
            if (P->vs->in[i].builtin >= 0) continue; /* gl_VertexIndex, gl_InstanceIndex */
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
        P->ps_slot = P->pc_slot = -1;
        if (P->fs) { /* point size and point coordinate travel as varyings */
            if (P->vs->ps_var >= 0) P->ps_slot = slot++;
            for (int i = 0; i < P->fs->nin; i++)
                if (P->fs->in[i].builtin == BI_PointCoord && P->pc_slot < 0) P->pc_slot = slot, slot += 2;
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
            if (fi->builtin == BI_PointCoord) P->fslot[i] = P->pc_slot;
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

void fm3d_spirv_set_fast_math(fm3d_spirv* P, int on)
{
    if (!P) return;
    if (P->vs) P->vs->fast = on != 0;
    if (P->fs) P->fs->fast = on != 0;
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
    for (int i = 0; P->fs && i < P->fs->nout; i++) p.writes_depth |= P->fs->out[i].builtin == BI_FragDepth;
    p.user      = (void*)P;
    p.point_size_var  = P->ps_slot + 1;
    p.point_coord_var = P->pc_slot + 1;
    return p;
}

#endif /* FM_FEATURE_SPIRV */
