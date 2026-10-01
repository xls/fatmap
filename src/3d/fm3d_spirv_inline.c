/*
 * fatmap - SPIR-V function call inlining (FM_FEATURE_SPIRV).
 *
 * A module to module rewrite done before parsing when an entry point calls
 * functions (glslang without the SPIRV-Tools optimizer, glslc -O0): every
 * OpFunctionCall of the entry points is replaced by a copy of the callee
 * (repeatedly, so calls inside callees are inlined too), and functions
 * that are not entry points are dropped.
 *
 *  - The calling block is split: the head keeps its label (so branches into
 *    it and its phis stay right), the tail gets a new one (phis elsewhere
 *    naming the old block as their parent are renamed to it).
 *  - A callee with one return, at the end of its last block, is spliced in:
 *    the return branches to the tail, its value becomes the call's result
 *    (OpCopyObject).
 *  - Otherwise the callee body is wrapped in a loop that runs once (the
 *    tail is its merge block); every return stores the value in a function
 *    variable and branches to the merge, from any nesting depth, as
 *    spirv-opt does for early returns.
 */
#include "fm3d_spirv_internal.h"

#if FM_FEATURE_SPIRV

#include <stdarg.h>
#include <stdio.h>

typedef struct iv_inst { /* one instruction, its own copy of the words */
    uint32_t* w;
    int       n;
} iv_inst;

typedef struct iv_block {
    iv_inst* in; /* in[0] is the OpLabel */
    int      n, cap;
} iv_block;

typedef struct iv_func {
    uint32_t  id, type_id;
    iv_inst   head; /* OpFunction */
    iv_inst*  params;
    int       nparams;
    iv_block* b;
    int       nb, bcap;
    int       is_entry;
} iv_func;

typedef struct iv_mod {
    uint32_t  bound;
    iv_inst*  pre; /* everything before the first function (+ types we add) */
    int       npre, precap;
    iv_func*  f;
    int       nf;
    int       fail;
} iv_mod;

static int iv_err(char* err, size_t errn, const char* fmt, ...)
{
    if (err && errn) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errn, fmt, ap);
        va_end(ap);
    }
    return 0;
}

static iv_inst iv_copy(const uint32_t* w, int n)
{
    iv_inst i;
    i.w = (uint32_t*)malloc((size_t)n * sizeof(uint32_t));
    i.n = i.w ? n : 0;
    if (i.w) memcpy(i.w, w, (size_t)n * sizeof(uint32_t));
    return i;
}

static int iv_push(iv_inst** a, int* n, int* cap, iv_inst x)
{
    if (!x.w) return 0;
    if (*n == *cap) {
        int      c  = *cap ? *cap * 2 : 16;
        iv_inst* na = (iv_inst*)realloc(*a, (size_t)c * sizeof(iv_inst));
        if (!na) return 0;
        *a = na, *cap = c;
    }
    (*a)[(*n)++] = x;
    return 1;
}

static int iv_block_push(iv_block* b, iv_inst x) { return iv_push(&b->in, &b->n, &b->cap, x); }

static iv_block* iv_new_block(iv_func* f, int at) /* inserted at index at */
{
    if (f->nb == f->bcap) {
        int       c  = f->bcap ? f->bcap * 2 : 16;
        iv_block* nb = (iv_block*)realloc(f->b, (size_t)c * sizeof(iv_block));
        if (!nb) return NULL;
        f->b = nb, f->bcap = c;
    }
    memmove(f->b + at + 1, f->b + at, (size_t)(f->nb - at) * sizeof(iv_block));
    f->nb++;
    memset(&f->b[at], 0, sizeof(iv_block));
    return &f->b[at];
}

static iv_inst iv_make(int op, int n, ...)
{
    uint32_t w[16];
    va_list  ap;
    va_start(ap, n);
    w[0] = ((uint32_t)(n + 1) << 16) | (uint32_t)op;
    for (int i = 0; i < n && i < 15; i++) w[1 + i] = va_arg(ap, uint32_t);
    va_end(ap);
    return iv_copy(w, n + 1);
}

/* which words of a function body instruction are ids (the rest literals) */
static int iv_is_id(const uint32_t* w, int k)
{
    int op = (int)(w[0] & 0xffff), n = (int)(w[0] >> 16);
    if (k <= 0 || k >= n) return 0;
    switch (op) {
    case OpCompositeExtract: return k <= 3;
    case OpCompositeInsert: case OpVectorShuffle: return k <= 4;
    case OpExtInst: return k != 4;
    case OpSwitch: return k <= 2 || ((k - 3) & 1); /* selector, default, then (literal, label) pairs */
    case OpLoopMerge: return k <= 2;
    case OpSelectionMerge: return k == 1;
    case OpLoad: return k <= 3;
    case OpStore: return k <= 2;
    case OpLine: return k == 1;
    case OpImageSampleImplicitLod: case OpImageSampleExplicitLod: case OpImageSampleProjImplicitLod:
    case OpImageSampleProjExplicitLod: return k != 5;
    case OpBranchConditional: return k <= 3;
    case OpVariable: return k != 3;
    case OpDecorate: case OpMemberDecorate: case OpName: case OpMemberName: return 0;
    default: return 1;
    }
}

static void iv_free_inst(iv_inst* i)
{
    free(i->w);
    i->w = NULL, i->n = 0;
}

static void iv_free(iv_mod* m)
{
    for (int i = 0; i < m->npre; i++) iv_free_inst(&m->pre[i]);
    free(m->pre);
    for (int f = 0; f < m->nf; f++) {
        iv_func* F = &m->f[f];
        iv_free_inst(&F->head);
        for (int i = 0; i < F->nparams; i++) iv_free_inst(&F->params[i]);
        free(F->params);
        for (int b = 0; b < F->nb; b++) {
            for (int i = 0; i < F->b[b].n; i++) iv_free_inst(&F->b[b].in[i]);
            free(F->b[b].in);
        }
        free(F->b);
    }
    free(m->f);
}

static int iv_parse(iv_mod* m, const uint32_t* w, size_t nw, char* err, size_t errn)
{
    memset(m, 0, sizeof(*m));
    m->bound = w[3];
    uint32_t entries[16];
    int      ne = 0;
    iv_func* F  = NULL;
    int      fcap = 0, pcap = 0;
    for (size_t p = 5; p < nw;) {
        int n = (int)(w[p] >> 16), op = (int)(w[p] & 0xffff);
        if (n == 0 || p + (size_t)n > nw) return iv_err(err, errn, "truncated instruction");
        if (op == OpEntryPoint && n >= 3 && ne < 16) entries[ne++] = w[p + 2];
        if (op == OpFunction) {
            if (m->nf == fcap) {
                fcap      = fcap ? fcap * 2 : 8;
                iv_func* nf = (iv_func*)realloc(m->f, (size_t)fcap * sizeof(iv_func));
                if (!nf) return iv_err(err, errn, "out of memory");
                m->f = nf;
            }
            F = &m->f[m->nf++];
            memset(F, 0, sizeof(*F));
            F->id = w[p + 2], F->type_id = w[p + 1], F->head = iv_copy(w + p, n);
            pcap = 0;
            for (int e = 0; e < ne; e++) F->is_entry |= entries[e] == F->id;
        } else if (!F) {
            if (!iv_push(&m->pre, &m->npre, &m->precap, iv_copy(w + p, n))) return iv_err(err, errn, "out of memory");
        } else if (op == OpFunctionParameter) {
            if (!iv_push(&F->params, &F->nparams, &pcap, iv_copy(w + p, n))) return iv_err(err, errn, "out of memory");
        } else if (op == OpLabel) {
            iv_block* B = iv_new_block(F, F->nb);
            if (!B || !iv_block_push(B, iv_copy(w + p, n))) return iv_err(err, errn, "out of memory");
        } else if (op == OpFunctionEnd) {
            F = NULL;
        } else if (F->nb) {
            if (!iv_block_push(&F->b[F->nb - 1], iv_copy(w + p, n))) return iv_err(err, errn, "out of memory");
        }
        p += (size_t)n;
    }
    return 1;
}

static iv_func* iv_find(iv_mod* m, uint32_t id)
{
    for (int i = 0; i < m->nf; i++)
        if (m->f[i].id == id) return &m->f[i];
    return NULL;
}

/* a pointer type to t in Function storage (added to the globals if missing) */
static uint32_t iv_func_ptr_type(iv_mod* m, uint32_t t)
{
    for (int i = 0; i < m->npre; i++) {
        const uint32_t* w = m->pre[i].w;
        if ((w[0] & 0xffff) == OpTypePointer && w[2] == SC_Function && w[3] == t) return w[1];
    }
    uint32_t id = m->bound++;
    if (!iv_push(&m->pre, &m->npre, &m->precap, iv_make(OpTypePointer, 3, id, (uint32_t)SC_Function, t))) m->fail = 1;
    return id;
}

/* local id map of a callee copy: params -> arguments, results -> fresh ids */
typedef struct iv_map {
    uint32_t* from;
    uint32_t* to;
    int       n, cap;
} iv_map;

static void iv_map_add(iv_map* m, uint32_t a, uint32_t b)
{
    if (m->n == m->cap) {
        int c = m->cap ? m->cap * 2 : 64;
        m->from = (uint32_t*)realloc(m->from, (size_t)c * sizeof(uint32_t));
        m->to   = (uint32_t*)realloc(m->to, (size_t)c * sizeof(uint32_t));
        m->cap  = c;
    }
    if (m->from && m->to) m->from[m->n] = a, m->to[m->n++] = b;
}

static uint32_t iv_map_get(const iv_map* m, uint32_t a)
{
    for (int i = 0; i < m->n; i++)
        if (m->from[i] == a) return m->to[i];
    return a;
}

/* does instruction w define a result id (its position, else 0) */
static int iv_result_pos(const uint32_t* w)
{
    int op = (int)(w[0] & 0xffff), n = (int)(w[0] >> 16);
    switch (op) {
    case OpLabel: return 1;
    case OpStore: case OpBranch: case OpBranchConditional: case OpSwitch: case OpReturn: case OpReturnValue: case OpKill:
    case OpUnreachable: case OpLoopMerge: case OpSelectionMerge: case OpLine: case OpNoLine: case OpTerminateInvocation:
        return 0;
    default: return n >= 3 ? 2 : 0;
    }
}

static int iv_inline_one(iv_mod* m, iv_func* F, int bi, int ci, char* err, size_t errn)
{
    iv_block*       B    = &F->b[bi];
    const uint32_t* call = B->in[ci].w;
    uint32_t        rtype = call[1], rid = call[2];
    iv_func*        C    = iv_find(m, call[3]);
    if (!C || !C->nb) return iv_err(err, errn, "call of an unknown function");
    if ((int)(call[0] >> 16) - 4 != C->nparams) return iv_err(err, errn, "call with the wrong number of arguments");
    for (int i = ci + 1; i < B->n; i++)
        if ((B->in[i].w[0] & 0xffff) == OpLoopMerge) return iv_err(err, errn, "function call in a loop header block");
    uint32_t void_type = 0;
    for (int i = 0; i < m->npre; i++)
        if ((m->pre[i].w[0] & 0xffff) == OpTypeVoid) void_type = m->pre[i].w[1];
    int has_value = rtype != void_type;

    /* returns: one, at the end of the last block -> splice; else wrap */
    int nret = 0, last_ret = 0;
    for (int b = 0; b < C->nb; b++) {
        const iv_block* cb = &C->b[b];
        int             op = (int)(cb->in[cb->n - 1].w[0] & 0xffff);
        if (op == OpReturn || op == OpReturnValue) nret++, last_ret = b == C->nb - 1;
    }
    int wrap = !(nret == 1 && last_ret);

    iv_map map;
    memset(&map, 0, sizeof(map));
    for (int i = 0; i < C->nparams; i++) iv_map_add(&map, C->params[i].w[2], call[4 + i]);
    for (int b = 0; b < C->nb; b++)
        for (int i = 0; i < C->b[b].n; i++) {
            int rp = iv_result_pos(C->b[b].in[i].w);
            if (rp) iv_map_add(&map, C->b[b].in[i].w[rp], m->bound++);
        }
    uint32_t orig = B->in[0].w[1], tail_l = m->bound++, hdr_l = wrap ? m->bound++ : 0, cont_l = wrap ? m->bound++ : 0;
    uint32_t retvar = 0;
    if (wrap && has_value) {
        retvar          = m->bound++;
        uint32_t ptrt   = iv_func_ptr_type(m, rtype);
        iv_block* entry = &F->b[0]; /* the variable at the start of the function */
        iv_inst   v     = iv_make(OpVariable, 3, ptrt, retvar, (uint32_t)SC_Function);
        if (!iv_block_push(entry, v)) m->fail = 1;
        else memmove(entry->in + 2, entry->in + 1, (size_t)(entry->n - 2) * sizeof(iv_inst)), entry->in[1] = v;
        B = &F->b[bi]; /* entry may be B: find the call again */
        for (ci = 0; ci < B->n; ci++)
            if ((B->in[ci].w[0] & 0xffff) == OpFunctionCall && B->in[ci].w[2] == rid) break;
        if (ci == B->n) return iv_err(err, errn, "internal: lost the call");
    }

    /* phis naming the calling block as their parent now come from the tail */
    for (int b = 0; b < F->nb; b++)
        for (int i = 0; i < F->b[b].n; i++) {
            uint32_t* w = F->b[b].in[i].w;
            if ((w[0] & 0xffff) != OpPhi) continue;
            for (int k = 4; k < (int)(w[0] >> 16); k += 2)
                if (w[k] == orig) w[k] = tail_l;
        }

    /* the new blocks: head (B, cut at the call), [header], callee, [continue], tail */
    int       ncb   = C->nb + (wrap ? 2 : 0) + 1;
    iv_block* first = NULL;
    for (int k = 0; k < ncb; k++) first = iv_new_block(F, bi + 1);
    if (!first) return iv_err(err, errn, "out of memory");
    B             = &F->b[bi];
    int       at  = bi + 1;
    iv_block* tail = &F->b[bi + ncb];
    iv_block_push(tail, iv_make(OpLabel, 1, tail_l));
    if (has_value && !wrap) {
        /* the value of the single return (mapped) */
        const iv_block* lb = &C->b[C->nb - 1];
        iv_block_push(tail, iv_make(OpCopyObject, 3, rtype, rid, iv_map_get(&map, lb->in[lb->n - 1].w[1])));
    } else if (has_value) {
        iv_block_push(tail, iv_make(OpLoad, 3, rtype, rid, retvar));
    }
    for (int i = ci + 1; i < B->n; i++) iv_block_push(tail, B->in[i]); /* moved, not copied */
    iv_free_inst(&B->in[ci]);
    B->n = ci;
    uint32_t entry_l = iv_map_get(&map, C->b[0].in[0].w[1]);
    iv_block_push(B, iv_make(OpBranch, 1, wrap ? hdr_l : entry_l));
    if (wrap) {
        iv_block* h = &F->b[at++];
        iv_block_push(h, iv_make(OpLabel, 1, hdr_l));
        iv_block_push(h, iv_make(OpLoopMerge, 3, tail_l, cont_l, 0u));
        iv_block_push(h, iv_make(OpBranch, 1, entry_l));
    }
    for (int b = 0; b < C->nb; b++) {
        iv_block* nb = &F->b[at++];
        for (int i = 0; i < C->b[b].n; i++) {
            const uint32_t* w  = C->b[b].in[i].w;
            int             op = (int)(w[0] & 0xffff), n = (int)(w[0] >> 16);
            if (op == OpReturn || op == OpReturnValue) {
                if (wrap && op == OpReturnValue && has_value) iv_block_push(nb, iv_make(OpStore, 2, retvar, iv_map_get(&map, w[1])));
                iv_block_push(nb, iv_make(OpBranch, 1, tail_l));
                continue;
            }
            iv_inst c = iv_copy(w, n);
            for (int k = 1; c.w && k < n; k++)
                if (iv_is_id(w, k)) c.w[k] = iv_map_get(&map, w[k]);
            iv_block_push(nb, c);
        }
    }
    if (wrap) {
        iv_block* cb = &F->b[at++];
        iv_block_push(cb, iv_make(OpLabel, 1, cont_l));
        iv_block_push(cb, iv_make(OpBranch, 1, hdr_l));
    }
    free(map.from);
    free(map.to);
    return !m->fail;
}

static int iv_has_calls(const uint32_t* w, size_t nw)
{
    for (size_t p = 5; p < nw;) {
        int n = (int)(w[p] >> 16);
        if (!n) return 0;
        if ((w[p] & 0xffff) == OpFunctionCall) return 1;
        p += (size_t)n;
    }
    return 0;
}

/* the module with every call of its entry points inlined; NULL with no
 * calls (use the input) or on error (err set) */
uint32_t* sv_inline_calls(const uint32_t* w, size_t nw, size_t* out_words, char* err, size_t errn)
{
    if (err && errn) err[0] = 0;
    if (!w || nw < 5 || !iv_has_calls(w, nw)) return NULL;
    iv_mod m;
    if (!iv_parse(&m, w, nw, err, errn)) {
        iv_free(&m);
        return NULL;
    }
    int ok = 1, count = 0;
    for (int f = 0; f < m.nf && ok; f++) {
        if (!m.f[f].is_entry) continue;
        for (int b = 0; b < m.f[f].nb && ok;) {
            int ci = -1;
            for (int i = 0; i < m.f[f].b[b].n && ci < 0; i++)
                if ((m.f[f].b[b].in[i].w[0] & 0xffff) == OpFunctionCall) ci = i;
            if (ci < 0) {
                b++;
                continue;
            }
            if (++count > 100000) ok = iv_err(err, errn, "too many calls to inline (recursion?)");
            else ok = iv_inline_one(&m, &m.f[f], b, ci, err, errn);
        }
    }
    uint32_t* out = NULL;
    size_t    n   = 5;
    if (ok) { /* the entry points only */
        for (int i = 0; i < m.npre; i++) n += (size_t)m.pre[i].n;
        for (int f = 0; f < m.nf; f++) {
            if (!m.f[f].is_entry) continue;
            n += (size_t)m.f[f].head.n + 1;
            for (int b = 0; b < m.f[f].nb; b++)
                for (int i = 0; i < m.f[f].b[b].n; i++) n += (size_t)m.f[f].b[b].in[i].n;
        }
        out = (uint32_t*)malloc(n * sizeof(uint32_t));
    }
    if (out) {
        size_t o = 0;
        memcpy(out, w, 5 * sizeof(uint32_t));
        out[3] = m.bound;
        o      = 5;
#define IV_PUT(I) (memcpy(out + o, (I).w, (size_t)(I).n * sizeof(uint32_t)), o += (size_t)(I).n)
        for (int i = 0; i < m.npre; i++) IV_PUT(m.pre[i]);
        for (int f = 0; f < m.nf; f++) {
            if (!m.f[f].is_entry) continue;
            IV_PUT(m.f[f].head);
            for (int b = 0; b < m.f[f].nb; b++)
                for (int i = 0; i < m.f[f].b[b].n; i++) IV_PUT(m.f[f].b[b].in[i]);
            out[o++] = (1u << 16) | (uint32_t)OpFunctionEnd;
        }
#undef IV_PUT
        *out_words = o;
    } else if (ok) {
        iv_err(err, errn, "out of memory");
    }
    iv_free(&m);
    return out;
}

#endif /* FM_FEATURE_SPIRV */
