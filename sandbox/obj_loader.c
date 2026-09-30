/* sandbox helper: Wavefront OBJ + MTL loader */
#include "obj_loader.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct fvec {
    float* d;
    int    n, cap, dim;
} fvec;

static void fvec_push(fvec* v, const float* x)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 1024;
        v->d   = (float*)realloc(v->d, (size_t)v->cap * (size_t)v->dim * sizeof(float));
    }
    memcpy(v->d + (size_t)v->n * v->dim, x, (size_t)v->dim * sizeof(float));
    v->n++;
}

typedef struct mtl {
    char  name[64];
    char  map_kd[256];
    float kd[3];
} mtl;

/* returns the next line (NUL terminated in place, CR stripped) or NULL */
static char* next_line(char** cur)
{
    char* s = *cur;
    if (!s || !*s) return NULL;
    char* e = strchr(s, '\n');
    if (e) {
        *e   = 0;
        *cur = e + 1;
    } else {
        *cur = s + strlen(s);
    }
    size_t n = strlen(s);
    if (n && s[n - 1] == '\r') s[n - 1] = 0;
    return s;
}

static char* read_file(const char* path, size_t* len)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (buf) buf[n] = 0;
    if (len) *len = (size_t)n;
    return buf;
}

static void base_name(const char* s, char* out, size_t cap)
{
    const char* b = s;
    for (const char* p = s; *p; p++)
        if (*p == '/' || *p == '\\') b = p + 1;
    size_t n = strlen(b);
    while (n && isspace((unsigned char)b[n - 1])) n--;
    if (n >= cap) n = cap - 1;
    memcpy(out, b, n);
    out[n] = 0;
}

static int load_mtl(const char* path, mtl** out)
{
    char* src = read_file(path, NULL);
    if (!src) return 0;
    int   n = 0, cap = 0;
    mtl*  m = NULL;
    char* cursor = src;
    for (char* line; (line = next_line(&cursor)) != NULL;) {
        while (isspace((unsigned char)*line)) line++;
        if (!strncmp(line, "newmtl", 6)) {
            if (n == cap) {
                cap = cap ? cap * 2 : 8;
                m   = (mtl*)realloc(m, (size_t)cap * sizeof(mtl));
            }
            memset(&m[n], 0, sizeof(mtl));
            m[n].kd[0] = m[n].kd[1] = m[n].kd[2] = 1.0f;
            base_name(line + 7, m[n].name, sizeof(m[n].name));
            n++;
        } else if (n && !strncmp(line, "Kd", 2)) {
            sscanf(line + 2, "%f %f %f", &m[n - 1].kd[0], &m[n - 1].kd[1], &m[n - 1].kd[2]);
        } else if (n && !strncmp(line, "map_Kd", 6)) {
            base_name(line + 7, m[n - 1].map_kd, sizeof(m[n - 1].map_kd));
        }
    }
    free(src);
    *out = m;
    return n;
}

/* vertex dedupe: open addressing on (v, vt, vn) */
typedef struct vkey {
    int      v, t, n;
    uint32_t idx;
} vkey;

typedef struct part_build {
    obj_part p;
    int      cap_v, cap_i;
    vkey*    hash;
    int      hcap;
} part_build;

static uint32_t part_vertex(part_build* b, int vi, int ti, int ni, const fvec* V, const fvec* T, const fvec* N)
{
    if (b->hcap == 0 || b->p.nv * 2 >= b->hcap) {
        int   nc = b->hcap ? b->hcap * 2 : 4096;
        vkey* nh = (vkey*)malloc((size_t)nc * sizeof(vkey));
        for (int i = 0; i < nc; i++) nh[i].v = -1;
        for (int i = 0; i < b->hcap; i++) {
            if (b->hash[i].v < 0) continue;
            uint32_t h = ((uint32_t)b->hash[i].v * 73856093u) ^ ((uint32_t)b->hash[i].t * 19349663u) ^
                         ((uint32_t)b->hash[i].n * 83492791u);
            for (uint32_t k = h & (uint32_t)(nc - 1);; k = (k + 1) & (uint32_t)(nc - 1))
                if (nh[k].v < 0) {
                    nh[k] = b->hash[i];
                    break;
                }
        }
        free(b->hash);
        b->hash = nh;
        b->hcap = nc;
    }
    uint32_t h = ((uint32_t)vi * 73856093u) ^ ((uint32_t)ti * 19349663u) ^ ((uint32_t)ni * 83492791u);
    for (uint32_t k = h & (uint32_t)(b->hcap - 1);; k = (k + 1) & (uint32_t)(b->hcap - 1)) {
        vkey* e = &b->hash[k];
        if (e->v < 0) {
            if (b->p.nv == b->cap_v) {
                b->cap_v = b->cap_v ? b->cap_v * 2 : 1024;
                b->p.v   = (fm3d_vertex*)realloc(b->p.v, (size_t)b->cap_v * sizeof(fm3d_vertex));
            }
            fm3d_vertex* o = &b->p.v[b->p.nv];
            memset(o, 0, sizeof(*o));
            const float* p = V->d + (size_t)vi * 3;
            o->x = p[0], o->y = p[1], o->z = p[2];
            if (ti >= 0 && ti < T->n) {
                o->u = T->d[(size_t)ti * 3];
                o->v = 1.0f - T->d[(size_t)ti * 3 + 1]; /* OBJ v is bottom-up */
            }
            if (ni >= 0 && ni < N->n) {
                const float* q = N->d + (size_t)ni * 3;
                o->nx = q[0], o->ny = q[1], o->nz = q[2];
            }
            o->color = 0xffffffffu;
            e->v = vi, e->t = ti, e->n = ni, e->idx = (uint32_t)b->p.nv++;
            return e->idx;
        }
        if (e->v == vi && e->t == ti && e->n == ni) return e->idx;
    }
}

static void part_tri(part_build* b, uint32_t a, uint32_t c, uint32_t d)
{
    if (b->p.ni + 3 > b->cap_i) {
        b->cap_i = b->cap_i ? b->cap_i * 2 : 3072;
        b->p.idx = (uint32_t*)realloc(b->p.idx, (size_t)b->cap_i * sizeof(uint32_t));
    }
    b->p.idx[b->p.ni++] = a;
    b->p.idx[b->p.ni++] = c;
    b->p.idx[b->p.ni++] = d;
}

static int resolve(int i, int n) { return i > 0 ? i - 1 : (i < 0 ? n + i : -1); }

int obj_load(const char* path, obj_model* out)
{
    memset(out, 0, sizeof(*out));
    char* src = read_file(path, NULL);
    if (!src) return 0;
    fvec V = { 0, 0, 0, 3 }, T = { 0, 0, 0, 3 }, N = { 0, 0, 0, 3 };
    mtl* mats  = NULL;
    int  nmats = 0;
    char dir[512];
    base_name(path, dir, sizeof(dir));
    size_t dl = strlen(path) - strlen(dir);
    memcpy(dir, path, dl);
    dir[dl] = 0;

    part_build* parts = NULL;
    int         np = 0, cur = -1;
    char        group[64] = "default", material[64] = "";
    char*       cursor = src;
    for (char* line; (line = next_line(&cursor)) != NULL;) {
        while (isspace((unsigned char)*line)) line++;
        if (line[0] == 'v' && line[1] == ' ') {
            float x[3] = { 0, 0, 0 };
            sscanf(line + 2, "%f %f %f", &x[0], &x[1], &x[2]);
            fvec_push(&V, x);
        } else if (line[0] == 'v' && line[1] == 't') {
            float x[3] = { 0, 0, 0 };
            sscanf(line + 3, "%f %f", &x[0], &x[1]);
            fvec_push(&T, x);
        } else if (line[0] == 'v' && line[1] == 'n') {
            float x[3] = { 0, 0, 1 };
            sscanf(line + 3, "%f %f %f", &x[0], &x[1], &x[2]);
            fvec_push(&N, x);
        } else if (!strncmp(line, "mtllib", 6)) {
            char name[256], full[800];
            base_name(line + 7, name, sizeof(name));
            snprintf(full, sizeof(full), "%s%s", dir, name);
            free(mats);
            nmats = load_mtl(full, &mats);
        } else if (line[0] == 'g' && line[1] == ' ') {
            base_name(line + 2, group, sizeof(group));
            cur = -1;
        } else if (!strncmp(line, "usemtl", 6)) {
            base_name(line + 7, material, sizeof(material));
            cur = -1;
        } else if (line[0] == 'f' && line[1] == ' ') {
            if (cur < 0) {
                for (int i = 0; i < np; i++)
                    if (!strcmp(parts[i].p.group, group) && !strcmp(parts[i].p.material, material)) cur = i;
                if (cur < 0) {
                    parts = (part_build*)realloc(parts, (size_t)(np + 1) * sizeof(part_build));
                    memset(&parts[np], 0, sizeof(part_build));
                    snprintf(parts[np].p.group, sizeof(parts[np].p.group), "%s", group);
                    snprintf(parts[np].p.material, sizeof(parts[np].p.material), "%s", material);
                    parts[np].p.kd[0] = parts[np].p.kd[1] = parts[np].p.kd[2] = 1.0f;
                    for (int m = 0; m < nmats; m++)
                        if (!strcmp(mats[m].name, material)) {
                            memcpy(parts[np].p.kd, mats[m].kd, sizeof(mats[m].kd));
                            snprintf(parts[np].p.map_kd, sizeof(parts[np].p.map_kd), "%s", mats[m].map_kd);
                        }
                    cur = np++;
                }
            }
            uint32_t poly[64];
            int      nv = 0;
            char*    p  = line + 2;
            while (*p && nv < 64) {
                while (isspace((unsigned char)*p)) p++;
                if (!*p) break;
                int vi = (int)strtol(p, &p, 10), ti = 0, ni = 0;
                if (*p == '/') {
                    p++;
                    if (*p != '/') ti = (int)strtol(p, &p, 10);
                    if (*p == '/') {
                        p++;
                        ni = (int)strtol(p, &p, 10);
                    }
                }
                while (*p && !isspace((unsigned char)*p)) p++;
                vi = resolve(vi, V.n);
                if (vi < 0 || vi >= V.n) continue;
                poly[nv++] = part_vertex(&parts[cur], vi, resolve(ti, T.n), resolve(ni, N.n), &V, &T, &N);
            }
            for (int k = 1; k + 1 < nv; k++) part_tri(&parts[cur], poly[0], poly[k], poly[k + 1]);
        }
    }
    free(src);
    free(mats);

    out->parts  = (obj_part*)calloc((size_t)(np ? np : 1), sizeof(obj_part));
    out->nparts = np;
    for (int k = 0; k < 3; k++) {
        out->bmin[k] = 1e30f;
        out->bmax[k] = -1e30f;
    }
    for (int i = 0; i < np; i++) {
        out->parts[i] = parts[i].p;
        free(parts[i].hash);
        for (int v = 0; v < parts[i].p.nv; v++) {
            const float* q = &parts[i].p.v[v].x;
            for (int k = 0; k < 3; k++) {
                if (q[k] < out->bmin[k]) out->bmin[k] = q[k];
                if (q[k] > out->bmax[k]) out->bmax[k] = q[k];
            }
        }
    }
    free(parts);
    free(V.d);
    free(T.d);
    free(N.d);
    return np > 0;
}

void obj_free(obj_model* m)
{
    for (int i = 0; i < m->nparts; i++) {
        free(m->parts[i].v);
        free(m->parts[i].idx);
    }
    free(m->parts);
    memset(m, 0, sizeof(*m));
}
