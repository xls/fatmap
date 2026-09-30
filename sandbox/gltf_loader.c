/* sandbox helper: minimal glTF 2.0 binary (.glb) loader (no dependencies) */
#include "gltf_loader.h"
#include "image_load.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny JSON DOM ------------------------------------------------------------ */

enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ };

typedef struct jv {
    int         type;
    double      num;
    const char* str; /* strings and object keys point into the source (not terminated) */
    int         slen;
    const char* key;
    int         klen;
    struct jv*  kid;
    int         nkid;
} jv;

typedef struct jparser {
    const char* p;
    const char* end;
} jparser;

static void jws(jparser* j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t')) j->p++;
}

static int jparse(jparser* j, jv* out);

static int jstring(jparser* j, const char** s, int* len)
{
    if (j->p >= j->end || *j->p != '"') return 0;
    const char* b = ++j->p;
    while (j->p < j->end && *j->p != '"') j->p += (*j->p == '\\') ? 2 : 1;
    if (j->p >= j->end) return 0;
    *s   = b;
    *len = (int)(j->p - b);
    j->p++;
    return 1;
}

static int jlist(jparser* j, jv* out, char close, int obj)
{
    int cap = 0;
    out->kid  = NULL;
    out->nkid = 0;
    j->p++;
    jws(j);
    if (j->p < j->end && *j->p == close) {
        j->p++;
        return 1;
    }
    for (;;) {
        if (out->nkid == cap) {
            cap      = cap ? cap * 2 : 8;
            out->kid = (jv*)realloc(out->kid, (size_t)cap * sizeof(jv));
        }
        jv* k = &out->kid[out->nkid];
        memset(k, 0, sizeof(*k));
        jws(j);
        if (obj) {
            if (!jstring(j, &k->key, &k->klen)) return 0;
            jws(j);
            if (j->p >= j->end || *j->p != ':') return 0;
            j->p++;
        }
        const char* key  = k->key;
        int         klen = k->klen;
        if (!jparse(j, k)) return 0;
        k->key  = key;
        k->klen = klen;
        out->nkid++;
        jws(j);
        if (j->p < j->end && *j->p == ',') {
            j->p++;
            continue;
        }
        if (j->p < j->end && *j->p == close) {
            j->p++;
            return 1;
        }
        return 0;
    }
}

static int jparse(jparser* j, jv* out)
{
    jws(j);
    if (j->p >= j->end) return 0;
    char c = *j->p;
    if (c == '{') {
        out->type = JOBJ;
        return jlist(j, out, '}', 1);
    }
    if (c == '[') {
        out->type = JARR;
        return jlist(j, out, ']', 0);
    }
    if (c == '"') {
        out->type = JSTR;
        return jstring(j, &out->str, &out->slen);
    }
    if (!strncmp(j->p, "true", 4) || !strncmp(j->p, "false", 5)) {
        out->type = JBOOL;
        out->num  = c == 't';
        j->p += c == 't' ? 4 : 5;
        return 1;
    }
    if (!strncmp(j->p, "null", 4)) {
        out->type = JNULL;
        j->p += 4;
        return 1;
    }
    char* e;
    out->type = JNUM;
    out->num  = strtod(j->p, &e);
    if (e == j->p) return 0;
    j->p = e;
    return 1;
}

static void jfree(jv* v)
{
    for (int i = 0; i < v->nkid; i++) jfree(&v->kid[i]);
    free(v->kid);
}

static const jv* jget(const jv* o, const char* key)
{
    if (!o || o->type != JOBJ) return NULL;
    int n = (int)strlen(key);
    for (int i = 0; i < o->nkid; i++)
        if (o->kid[i].klen == n && !memcmp(o->kid[i].key, key, (size_t)n)) return &o->kid[i];
    return NULL;
}

static const jv* jat(const jv* a, int i) { return (a && a->type == JARR && i >= 0 && i < a->nkid) ? &a->kid[i] : NULL; }
static double    jnum(const jv* v, double def) { return (v && (v->type == JNUM || v->type == JBOOL)) ? v->num : def; }
static int       jint(const jv* o, const char* key, int def) { return (int)jnum(jget(o, key), def); }

/* ---- accessors ---------------------------------------------------------------- */

typedef struct acc {
    const uint8_t* data;
    int            count, comps, ctype, stride, normalized;
} acc;

static int comps_of(const jv* t)
{
    if (!t || t->type != JSTR) return 0;
    static const struct { const char* s; int n; } tab[] = { { "SCALAR", 1 }, { "VEC2", 2 }, { "VEC3", 3 }, { "VEC4", 4 }, { "MAT4", 16 } };
    for (int i = 0; i < 5; i++)
        if ((int)strlen(tab[i].s) == t->slen && !memcmp(t->str, tab[i].s, (size_t)t->slen)) return tab[i].n;
    return 0;
}

static int csize(int ct) { return ct == 5126 || ct == 5125 ? 4 : (ct == 5122 || ct == 5123 ? 2 : 1); }

static int get_acc(const jv* root, const uint8_t* bin, size_t binlen, int index, acc* a)
{
    const jv* ac = jat(jget(root, "accessors"), index);
    if (!ac) return 0;
    const jv* bv = jat(jget(root, "bufferViews"), jint(ac, "bufferView", -1));
    if (!bv) return 0;
    a->count      = jint(ac, "count", 0);
    a->comps      = comps_of(jget(ac, "type"));
    a->ctype      = jint(ac, "componentType", 5126);
    a->normalized = (int)jnum(jget(ac, "normalized"), 0);
    size_t off    = (size_t)jint(bv, "byteOffset", 0) + (size_t)jint(ac, "byteOffset", 0);
    a->stride     = jint(bv, "byteStride", 0);
    if (!a->stride) a->stride = a->comps * csize(a->ctype);
    if (!a->comps || off + (size_t)(a->count ? a->count - 1 : 0) * (size_t)a->stride > binlen) return 0;
    a->data = bin + off;
    return 1;
}

static float acc_f(const acc* a, int i, int c)
{
    const uint8_t* p = a->data + (size_t)i * (size_t)a->stride + (size_t)c * (size_t)csize(a->ctype);
    switch (a->ctype) {
    case 5126: {
        float f;
        memcpy(&f, p, 4);
        return f;
    }
    case 5121: return a->normalized ? (float)p[0] / 255.0f : (float)p[0];
    case 5123: {
        uint16_t v;
        memcpy(&v, p, 2);
        return a->normalized ? (float)v / 65535.0f : (float)v;
    }
    case 5120: return a->normalized ? fmaxf((float)(int8_t)p[0] / 127.0f, -1.0f) : (float)(int8_t)p[0];
    case 5122: {
        int16_t v;
        memcpy(&v, p, 2);
        return a->normalized ? fmaxf((float)v / 32767.0f, -1.0f) : (float)v;
    }
    default: {
        uint32_t v;
        memcpy(&v, p, 4);
        return (float)v;
    }
    }
}

static uint32_t acc_u(const acc* a, int i, int c)
{
    const uint8_t* p = a->data + (size_t)i * (size_t)a->stride + (size_t)c * (size_t)csize(a->ctype);
    switch (a->ctype) {
    case 5121: return p[0];
    case 5123: {
        uint16_t v;
        memcpy(&v, p, 2);
        return v;
    }
    case 5125: {
        uint32_t v;
        memcpy(&v, p, 4);
        return v;
    }
    default: return (uint32_t)acc_f(a, i, c);
    }
}

/* ---- loader ------------------------------------------------------------------- */

/* decode image `index` (embedded in the binary chunk) */
static fm_surface* load_image(const jv* root, const uint8_t* bin, size_t binlen, int index)
{
    const jv* im = jat(jget(root, "images"), index);
    const jv* bv = jat(jget(root, "bufferViews"), jint(im, "bufferView", -1));
    if (!im || !bv) return NULL;
    size_t off = (size_t)jint(bv, "byteOffset", 0), len = (size_t)jint(bv, "byteLength", 0);
    if (off + len > binlen) return NULL;
    return img_load_mem(bin + off, len);
}

/* image index behind a material texture reference ({"index": texture}) */
static int tex_image(const jv* root, const jv* ref)
{
    const jv* tex = jat(jget(root, "textures"), (int)jnum(jget(ref, "index"), -1));
    return tex ? jint(tex, "source", -1) : -1;
}

int gltf_load(const char* path, gltf_model* m)
{
    memset(m, 0, sizeof(*m));
    m->image     = -1;
    m->mesh_node = -1;
    FILE* f  = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* d = (uint8_t*)malloc((size_t)sz);
    if (!d || fread(d, 1, (size_t)sz, f) != (size_t)sz || sz < 28 || memcmp(d, "glTF", 4)) {
        fclose(f);
        free(d);
        return 0;
    }
    fclose(f);
    uint32_t jlen, blen = 0;
    memcpy(&jlen, d + 12, 4);
    const uint8_t* bin = NULL;
    if (20 + (size_t)jlen + 8 <= (size_t)sz) {
        memcpy(&blen, d + 20 + jlen, 4);
        bin = d + 28 + jlen;
        if (28 + (size_t)jlen + blen > (size_t)sz) blen = 0;
    }
    jparser jp = { (const char*)d + 20, (const char*)d + 20 + jlen };
    jv      root;
    memset(&root, 0, sizeof(root));
    if (!jparse(&jp, &root) || !bin) {
        jfree(&root);
        free(d);
        return 0;
    }

    /* nodes */
    const jv* nodes = jget(&root, "nodes");
    m->nnodes       = nodes ? nodes->nkid : 0;
    m->nodes        = (gltf_node*)calloc((size_t)(m->nnodes ? m->nnodes : 1), sizeof(gltf_node));
    for (int i = 0; i < m->nnodes; i++) {
        gltf_node* n = &m->nodes[i];
        n->parent    = -1;
        n->r[3]      = 1;
        n->s[0] = n->s[1] = n->s[2] = 1;
    }
    for (int i = 0; i < m->nnodes; i++) {
        const jv*  nj = &nodes->kid[i];
        gltf_node* n  = &m->nodes[i];
        const jv*  t = jget(nj, "translation"), *r = jget(nj, "rotation"), *s = jget(nj, "scale"), *mt = jget(nj, "matrix");
        for (int k = 0; k < 3; k++) {
            n->t[k] = (float)jnum(jat(t, k), 0);
            n->s[k] = (float)jnum(jat(s, k), 1);
        }
        for (int k = 0; k < 4; k++) n->r[k] = (float)jnum(jat(r, k), k == 3 ? 1 : 0);
        if (mt) {
            n->has_matrix = 1;
            for (int k = 0; k < 16; k++) n->matrix[k] = (float)jnum(jat(mt, k), (k % 5) == 0);
        }
        if (m->mesh_node < 0 && jint(nj, "mesh", -1) == 0) m->mesh_node = i;
        const jv* ch = jget(nj, "children");
        for (int k = 0; ch && k < ch->nkid; k++) {
            int c = (int)jnum(&ch->kid[k], -1);
            if (c >= 0 && c < m->nnodes) m->nodes[c].parent = i;
        }
    }

    /* first mesh, all triangle primitives merged */
    const jv* mesh = jat(jget(&root, "meshes"), 0);
    const jv* prims = jget(mesh, "primitives");
    int       capv = 0, capi = 0, skinned = 0;
    for (int pi = 0; prims && pi < prims->nkid; pi++) {
        const jv* pr = &prims->kid[pi];
        if (jint(pr, "mode", 4) != 4) continue;
        const jv* at = jget(pr, "attributes");
        acc       pos, uv, nrm, jo, we, ix;
        if (!get_acc(&root, bin, blen, jint(at, "POSITION", -1), &pos)) continue;
        int has_uv = get_acc(&root, bin, blen, jint(at, "TEXCOORD_0", -1), &uv);
        int has_n  = get_acc(&root, bin, blen, jint(at, "NORMAL", -1), &nrm);
        int has_sk = get_acc(&root, bin, blen, jint(at, "JOINTS_0", -1), &jo) &&
                     get_acc(&root, bin, blen, jint(at, "WEIGHTS_0", -1), &we);
        skinned |= has_sk;
        int base = m->nv;
        if (m->nv + pos.count > capv) {
            capv    = (m->nv + pos.count) * 2;
            m->v    = (fm3d_vertex*)realloc(m->v, (size_t)capv * sizeof(fm3d_vertex));
            m->skin = (fm3d_skin_vertex*)realloc(m->skin, (size_t)capv * sizeof(fm3d_skin_vertex));
        }
        for (int i = 0; i < pos.count; i++) {
            fm3d_vertex* v = &m->v[m->nv + i];
            memset(v, 0, sizeof(*v));
            v->x = acc_f(&pos, i, 0), v->y = acc_f(&pos, i, 1), v->z = acc_f(&pos, i, 2);
            if (has_uv) v->u = acc_f(&uv, i, 0), v->v = acc_f(&uv, i, 1); /* glTF uv origin is top-left */
            if (has_n) v->nx = acc_f(&nrm, i, 0), v->ny = acc_f(&nrm, i, 1), v->nz = acc_f(&nrm, i, 2);
            v->color              = 0xffffffffu;
            fm3d_skin_vertex* s   = &m->skin[m->nv + i];
            float             sum = 0;
            for (int k = 0; k < 4; k++) {
                s->joint[k]  = has_sk ? (uint16_t)acc_u(&jo, i, k) : 0;
                s->weight[k] = has_sk ? acc_f(&we, i, k) : (k == 0 ? 1.0f : 0.0f);
                sum += s->weight[k];
            }
            if (sum > 0)
                for (int k = 0; k < 4; k++) s->weight[k] /= sum;
        }
        m->nv += pos.count;
        int has_ix = get_acc(&root, bin, blen, jint(pr, "indices", -1), &ix);
        int nidx   = has_ix ? ix.count : pos.count;
        if (m->ni + nidx > capi) {
            capi   = (m->ni + nidx) * 2;
            m->idx = (uint32_t*)realloc(m->idx, (size_t)capi * sizeof(uint32_t));
        }
        for (int i = 0; i < nidx; i++) m->idx[m->ni + i] = (uint32_t)base + (has_ix ? acc_u(&ix, i, 0) : (uint32_t)i);
        if (!has_n) { /* glTF: no normals -> face normals (flat where vertices are not shared) */
            for (int i = 0; i + 2 < nidx; i += 3) {
                fm3d_vertex* a = &m->v[m->idx[m->ni + i]];
                fm3d_vertex* b = &m->v[m->idx[m->ni + i + 1]];
                fm3d_vertex* c = &m->v[m->idx[m->ni + i + 2]];
                fm_vec3      n = fm_v3_cross(fm_v3(b->x - a->x, b->y - a->y, b->z - a->z), fm_v3(c->x - a->x, c->y - a->y, c->z - a->z));
                fm3d_vertex* t[3] = { a, b, c };
                for (int k = 0; k < 3; k++) t[k]->nx += n.x, t[k]->ny += n.y, t[k]->nz += n.z; /* area weighted */
            }
            for (int i = base; i < base + pos.count; i++) {
                fm_vec3 n = fm_v3_normalize(fm_v3(m->v[i].nx, m->v[i].ny, m->v[i].nz));
                m->v[i].nx = n.x, m->v[i].ny = n.y, m->v[i].nz = n.z;
            }
        }
        m->ni += nidx;
        /* base color texture -> image index */
        const jv* mat = jat(jget(&root, "materials"), jint(pr, "material", -1));
        const jv* bct = jget(jget(jget(mat, "pbrMetallicRoughness"), "baseColorTexture"), "index");
        const jv* tex = jat(jget(&root, "textures"), (int)jnum(bct, -1));
        if (tex && m->image < 0) m->image = jint(tex, "source", -1);
    }
    if (!skinned) {
        free(m->skin);
        m->skin = NULL;
    }
    /* material images of the first primitive */
    {
        const jv* pr  = jat(prims, 0);
        const jv* mat = jat(jget(&root, "materials"), jint(pr, "material", -1));
        if (m->image >= 0) m->base_color = load_image(&root, bin, blen, m->image);
        int ei = tex_image(&root, jget(mat, "emissiveTexture"));
        int oi = tex_image(&root, jget(mat, "occlusionTexture"));
        if (ei >= 0) m->emissive = load_image(&root, bin, blen, ei);
        if (oi >= 0) m->occlusion = load_image(&root, bin, blen, oi);
    }

    /* skin */
    const jv* skin = jat(jget(&root, "skins"), 0);
    const jv* js   = jget(skin, "joints");
    if (skin && js) {
        m->njoints  = js->nkid;
        m->joints   = (int*)calloc((size_t)m->njoints, sizeof(int));
        m->inv_bind = (float*)calloc((size_t)m->njoints * 16, sizeof(float));
        acc ib;
        int has_ib = get_acc(&root, bin, blen, jint(skin, "inverseBindMatrices", -1), &ib);
        for (int k = 0; k < m->njoints; k++) {
            m->joints[k] = (int)jnum(&js->kid[k], 0);
            for (int e = 0; e < 16; e++) m->inv_bind[16 * k + e] = has_ib ? acc_f(&ib, k, e) : (float)((e % 5) == 0);
        }
    }

    /* animations: key data copied out (the file buffer is freed) */
    const jv* anims = jget(&root, "animations");
    m->nanims       = anims ? anims->nkid : 0;
    m->anims        = (gltf_anim*)calloc((size_t)(m->nanims ? m->nanims : 1), sizeof(gltf_anim));
    size_t total    = 0;
    for (int pass = 0; pass < 2; pass++) {
        size_t used = 0;
        if (pass == 1) m->anim_data = (float*)malloc((total ? total : 1) * sizeof(float));
        for (int a = 0; a < m->nanims; a++) {
            const jv*  an = &anims->kid[a];
            const jv*  chs = jget(an, "channels"), *sms = jget(an, "samplers");
            gltf_anim* ga = &m->anims[a];
            if (pass == 1) {
                const jv* nm = jget(an, "name");
                if (nm && nm->type == JSTR)
                    snprintf(ga->name, sizeof(ga->name), "%.*s", nm->slen < 63 ? nm->slen : 63, nm->str);
                else
                    snprintf(ga->name, sizeof(ga->name), "anim%d", a);
                ga->ch  = (gltf_channel*)calloc((size_t)(chs ? chs->nkid : 1), sizeof(gltf_channel));
                ga->nch = 0;
            }
            for (int c = 0; chs && c < chs->nkid; c++) {
                const jv* ch  = &chs->kid[c];
                const jv* tg  = jget(ch, "target");
                const jv* pth = jget(tg, "path");
                const jv* sm  = jat(sms, jint(ch, "sampler", -1));
                int       path = -1;
                if (pth && pth->type == JSTR) {
                    if (pth->slen == 11 && !memcmp(pth->str, "translation", 11)) path = 0;
                    if (pth->slen == 8 && !memcmp(pth->str, "rotation", 8)) path = 1;
                    if (pth->slen == 5 && !memcmp(pth->str, "scale", 5)) path = 2;
                }
                acc in, out;
                if (path < 0 || !sm || !get_acc(&root, bin, blen, jint(sm, "input", -1), &in) ||
                    !get_acc(&root, bin, blen, jint(sm, "output", -1), &out))
                    continue;
                int w = path == 1 ? 4 : 3;
                if (out.count < in.count) continue;
                if (pass == 0) {
                    total += (size_t)in.count * (size_t)(1 + w);
                    continue;
                }
                gltf_channel* g = &ga->ch[ga->nch++];
                g->node         = jint(tg, "node", -1);
                g->path         = path;
                const jv* ip    = jget(sm, "interpolation");
                g->step         = ip && ip->type == JSTR && ip->slen == 4 && !memcmp(ip->str, "STEP", 4);
                g->nkeys        = in.count;
                float* tt       = m->anim_data + used;
                float* vv       = tt + in.count;
                for (int k = 0; k < in.count; k++) {
                    tt[k] = acc_f(&in, k, 0);
                    for (int e = 0; e < w; e++) vv[k * w + e] = acc_f(&out, k, e);
                }
                used += (size_t)in.count * (size_t)(1 + w);
                g->times  = tt;
                g->values = vv;
                if (in.count && tt[in.count - 1] > ga->duration) ga->duration = tt[in.count - 1];
            }
        }
    }
    jfree(&root);
    free(d);
    return m->nv > 0 && m->ni >= 3;
}

void gltf_free(gltf_model* m)
{
    fm_surface_destroy(m->base_color);
    fm_surface_destroy(m->emissive);
    fm_surface_destroy(m->occlusion);
    free(m->v);
    free(m->skin);
    free(m->idx);
    free(m->nodes);
    free(m->joints);
    free(m->inv_bind);
    for (int a = 0; a < m->nanims; a++) free(m->anims[a].ch);
    free(m->anims);
    free(m->anim_data);
    memset(m, 0, sizeof(*m));
}

/* ---- posing ------------------------------------------------------------------- */

static void sample(const gltf_channel* c, float t, float* out)
{
    int w = c->path == 1 ? 4 : 3;
    if (t <= c->times[0] || c->nkeys == 1) {
        memcpy(out, c->values, (size_t)w * sizeof(float));
        return;
    }
    if (t >= c->times[c->nkeys - 1]) {
        memcpy(out, c->values + (size_t)(c->nkeys - 1) * w, (size_t)w * sizeof(float));
        return;
    }
    int lo = 0, hi = c->nkeys - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (c->times[mid] <= t)
            lo = mid;
        else
            hi = mid;
    }
    const float* a = c->values + (size_t)lo * w;
    const float* b = c->values + (size_t)hi * w;
    float        f = c->step ? 0.0f : (t - c->times[lo]) / (c->times[hi] - c->times[lo]);
    if (c->path == 1) {
        fm_quat qa = { a[0], a[1], a[2], a[3] }, qb = { b[0], b[1], b[2], b[3] };
        fm_quat q  = fm_quat_slerp(qa, qb, f);
        out[0] = q.x, out[1] = q.y, out[2] = q.z, out[3] = q.w;
    } else {
        for (int e = 0; e < 3; e++) out[e] = a[e] + (b[e] - a[e]) * f;
    }
}

static void node_worlds(const gltf_model* m, int anim, float t, fm_mat4* world_out);

fm_mat4 gltf_mesh_matrix(const gltf_model* m)
{
    fm_mat4 r = fm_mat4_identity();
    if (m->mesh_node < 0 || m->mesh_node >= m->nnodes) return r;
    fm_mat4* w = (fm_mat4*)malloc((size_t)m->nnodes * sizeof(fm_mat4));
    if (!w) return r;
    node_worlds(m, -1, 0, w);
    r = w[m->mesh_node];
    free(w);
    return r;
}

void gltf_pose(const gltf_model* m, int anim, float t, fm_mat4* bones)
{
    int      n     = m->nnodes;
    fm_mat4* world = (fm_mat4*)malloc((size_t)(n ? n : 1) * sizeof(fm_mat4));
    if (!world) return;
    node_worlds(m, anim, t, world);
    for (int k = 0; k < m->njoints; k++) {
        int     j = m->joints[k];
        fm_mat4 ib;
        memcpy(&ib, m->inv_bind + 16 * k, sizeof(ib));
        bones[k] = (j >= 0 && j < n) ? fm_mat4_mul(world[j], ib) : fm_mat4_identity();
    }
    free(world);
}

/* world transforms of every node for animation `anim` at time t */
static void node_worlds(const gltf_model* m, int anim, float t, fm_mat4* world_out)
{
    int      n = m->nnodes;
    fm_mat4* local = (fm_mat4*)malloc((size_t)(n ? n : 1) * sizeof(fm_mat4) * 2);
    fm_mat4* world = local + n;
    int*     done  = (int*)calloc((size_t)(n ? n : 1), sizeof(int));
    float*   trs   = (float*)malloc((size_t)(n ? n : 1) * 10 * sizeof(float));
    for (int i = 0; i < n; i++) {
        memcpy(trs + 10 * i, m->nodes[i].t, 3 * sizeof(float));
        memcpy(trs + 10 * i + 3, m->nodes[i].r, 4 * sizeof(float));
        memcpy(trs + 10 * i + 7, m->nodes[i].s, 3 * sizeof(float));
    }
    if (anim >= 0 && anim < m->nanims) {
        const gltf_anim* a  = &m->anims[anim];
        float            tt = a->duration > 0 ? fmodf(t, a->duration) : 0.0f;
        if (tt < 0) tt += a->duration;
        for (int c = 0; c < a->nch; c++) {
            const gltf_channel* ch = &a->ch[c];
            if (ch->node < 0 || ch->node >= n) continue;
            sample(ch, tt, trs + 10 * ch->node + (ch->path == 0 ? 0 : (ch->path == 1 ? 3 : 7)));
        }
    }
    for (int i = 0; i < n; i++) {
        if (m->nodes[i].has_matrix) {
            memcpy(&local[i], m->nodes[i].matrix, sizeof(fm_mat4));
            continue;
        }
        const float* p = trs + 10 * i;
        fm_quat      q = { p[3], p[4], p[5], p[6] };
        fm_mat4      r = fm_mat4_from_quat(q);
        local[i]       = fm_scale(fm_mat4_mul(fm_translate(fm_mat4_identity(), fm_v3(p[0], p[1], p[2])), r),
                                  fm_v3(p[7], p[8], p[9]));
    }
    /* world transforms in parent order (iterate until all resolved) */
    for (int pass = 0; pass < n; pass++) {
        int progress = 0;
        for (int i = 0; i < n; i++) {
            if (done[i]) continue;
            int pa = m->nodes[i].parent;
            if (pa < 0) {
                world[i] = local[i];
            } else if (done[pa]) {
                world[i] = fm_mat4_mul(world[pa], local[i]);
            } else {
                continue;
            }
            done[i]  = 1;
            progress = 1;
        }
        if (!progress) break;
    }
    memcpy(world_out, world, (size_t)n * sizeof(fm_mat4));
    free(local);
    free(done);
    free(trs);
}
