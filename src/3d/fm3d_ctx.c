/*
 * fatmap - 3D context: state, immediate rendering and deferred tiled
 * rendering.
 *
 * Deferred flush, all phases parallel on the executor:
 *   A  vertex stage over blocks of vertices of every draw
 *   B  primitive assembly, clipping, culling, setup over chunks of
 *      triangles; each chunk bins its triangles into screen tiles
 *   C  per tile: replay commands in order (clears, then every chunk of every
 *      draw with its tile bin), so painter's order is exact and tiles never
 *      share pixels (no locks, color + depth of a 64x64 tile stay in cache)
 */
#include "fm3d_internal.h"
#if !FM_FEATURE_VBO
typedef struct fm3d_buffer fm3d_buffer; /* never instantiated: buf is always NULL */
#endif

#define FM3D_VBLOCK   2048 /* vertices per phase A task */
#define FM3D_TCHUNK   512  /* triangles per phase B task */
#define FM3D_TILE_DEF 64

enum { FM3D_CMD_CLEAR_COLOR, FM3D_CMD_CLEAR_DEPTH, FM3D_CMD_CLEAR_STENCIL, FM3D_CMD_DRAW };

typedef struct fm3d_cmd {
    int      type;
    int      rect[4];
    uint32_t color;
    float    depth;
    int      draw;
} fm3d_cmd;

typedef struct fm3d_drawrec {
    fm3d_dstate*       st;
    const void*        v; /* st->vstride bytes per vertex */
    int                nv;
    const uint32_t*    idx; /* NULL: non indexed */
    int                ntri;
    fm3d_vout*         vout;
    int                chunk0, nchunks;
} fm3d_drawrec;

typedef struct fm3d_chunk {
    int        draw, tri0, ntri;
    fm3d_tri** tris;
    int        ntris;
    int        btx, bty, btw, bth; /* tile bounding box of the chunk's triangles */
    uint32_t*  bin_start;          /* btw * bth + 1 (local tile index) */
    uint32_t*  bin;
} fm3d_chunk;

typedef struct fm3d_vtask {
    int draw, v0, n;
} fm3d_vtask;

typedef struct fm3d_worker {
    fm_arena   arena;
    fm3d_batch batch;
    fm3d_stats stats;
} fm3d_worker;

struct fm3d_ctx {
    fm_surface*   color;
    fm_surface*   depth;
    fm_surface*   stencil;
    /* MSAA buffers (allocated for the current target size / sample count) */
    int           msaa;
    uint32_t*     ms_color;
    float*        ms_depth;
    uint8_t*      ms_stencil;
    int           ms_w, ms_h, ms_s;
    fm3d_dstate   st; /* current state (rect / mvp resolved per draw) */
#if FM_FEATURE_SHADERS
    void*         uni[FM3D_MAX_UNIFORM_BLOCKS]; /* uniform blocks by binding (owned copies) */
    size_t        uni_size[FM3D_MAX_UNIFORM_BLOCKS], uni_cap[FM3D_MAX_UNIFORM_BLOCKS];
    void*         uni_rec[FM3D_MAX_UNIFORM_BLOCKS]; /* last snapshots recorded in deferred mode */
    size_t        uni_rec_size[FM3D_MAX_UNIFORM_BLOCKS];
#endif
#if FM_FEATURE_TNL
    int             lighting, color_material;
    int             light_on[FM3D_MAX_LIGHTS];
    fm3d_light      lights[FM3D_MAX_LIGHTS]; /* world space */
    fm3d_material   material;
    fm_vec3         ambient_light;
    fm_light_params lp;      /* eye space, resolved per draw */
    fm_light_params* lp_rec; /* last copy recorded in deferred mode (shared by equal draws) */
#endif
    int           scissor_on;
    int           scissor[4];
    int           vp_set;
    /* immediate */
    fm3d_vout*    vbuf;
    int           vbuf_cap;
    fm3d_batch    batch;
    fm3d_stats    stats;
    /* deferred */
    int           deferred;
    fm_executor*  exec;
    int           tile;
    fm_arena      rec;   /* recorded data */
    fm_arena      frame; /* flush scratch */
    fm3d_cmd*     cmds;
    int           ncmd, ccmd;
    fm3d_drawrec* draws;
    int           ndraw, cdraw;
    fm3d_texture** held; /* textures retained by recorded draws */
    int           nheld, cheld;
#if FM_FEATURE_VBO
    fm3d_buffer** hbuf; /* vertex buffers referenced by recorded draws */
    int           nhbuf, chbuf;
#endif
    fm3d_worker*  workers;
    int           nworkers;
    /* flush data */
    fm3d_chunk*   chunks;
    int           nchunks;
    fm3d_vtask*   vtasks;
    int           nvtasks;
    int           tiles_x, tiles_y;
    fm3d_hiz*     hiz; /* per tile depth bounds (flush scratch) */
    /* per tile work list in submission order: chunk index, or a clear
     * command index with FM3D_TL_CLEAR set (flush scratch) */
    uint32_t*     tl_start; /* tiles + 1 */
    uint32_t*     tl;
};

/* ---- state ------------------------------------------------------------------------ */

fm3d_ctx* fm3d_create(void)
{
    fm__init();
    fm3d_ctx* c = (fm3d_ctx*)calloc(1, sizeof(fm3d_ctx));
    if (!c) return NULL;
    fm3d_dstate* s = &c->st;
    s->model = s->view = s->proj = s->mvp = fm_mat4_identity();
    s->clip_depth                         = FM3D_DEPTH_NEG_ONE_ONE;
    s->line_width = s->point_size = 1.0f;
    s->psize_var = s->pcoord_var = -1;
    s->cull                               = FM3D_CULL_NONE;
    s->front                              = FM3D_FRONT_CCW;
    s->perspective                        = 1;
    s->depth_func                         = FM3D_LESS;
    s->depth_write                        = 1;
    s->depth_near                         = 0.0f;
    s->depth_far                          = 1.0f;
    s->color_write                        = 1;
    for (int f = 0; f < 2; f++) {
        s->stencil[f].func       = FM3D_ALWAYS;
        s->stencil[f].read_mask  = 0xff;
        s->stencil[f].write_mask = 0xff;
    }
    s->sampler.filter                     = FM3D_FILTER_BILINEAR;
    s->sampler.wrap_u = s->sampler.wrap_v = FM_WRAP_REPEAT;
    s->texenv                             = FM3D_TEXENV_MODULATE;
    s->alpha_func                         = FM3D_ALWAYS;
    s->op                                 = FM_OP_SRC_OVER;
    s->opacity8                           = 255;
    s->nvar                               = FM3D_FIXED_NVAR;
    s->vs                                 = fm3d_vs_fixed;
    s->vstride                            = (int)sizeof(fm3d_vertex);
#if FM_FEATURE_TNL
    c->material      = fm3d_material_default();
    c->ambient_light = fm_v3(0.2f, 0.2f, 0.2f);
#endif
    s->fs                                 = fm3d_fs_fixed;
    c->tile                               = FM3D_TILE_DEF;
    return c;
}

#if FM_FEATURE_VBO
struct fm3d_buffer {
    int          refs;
    fm3d_vertex* v;
    int          nv;
    uint32_t*    idx;
    int          ni;
};

fm3d_buffer* fm3d_buffer_create(const fm3d_vertex* v, int nv, const uint32_t* idx, int ni)
{
    if (!v || nv <= 0 || (idx && ni <= 0)) return NULL;
    for (int i = 0; idx && i < ni; i++)
        if (idx[i] >= (uint32_t)nv) return NULL; /* validated once here, never per draw */
    fm3d_buffer* b = (fm3d_buffer*)calloc(1, sizeof(fm3d_buffer));
    if (!b) return NULL;
    b->v   = (fm3d_vertex*)malloc((size_t)nv * sizeof(fm3d_vertex));
    b->idx = idx ? (uint32_t*)malloc((size_t)ni * sizeof(uint32_t)) : NULL;
    if (!b->v || (idx && !b->idx)) {
        free(b->v);
        free(b->idx);
        free(b);
        return NULL;
    }
    memcpy(b->v, v, (size_t)nv * sizeof(fm3d_vertex));
    if (idx) memcpy(b->idx, idx, (size_t)ni * sizeof(uint32_t));
    b->refs = 1;
    b->nv   = nv;
    b->ni   = idx ? ni : 0;
    return b;
}

fm3d_buffer* fm3d_buffer_retain(fm3d_buffer* b)
{
    if (b) b->refs++;
    return b;
}

void fm3d_buffer_release(fm3d_buffer* b)
{
    if (!b || --b->refs > 0) return;
    free(b->v);
    free(b->idx);
    free(b);
}

int fm3d_buffer_vertex_count(const fm3d_buffer* b) { return b ? b->nv : 0; }
int fm3d_buffer_index_count(const fm3d_buffer* b) { return b ? b->ni : 0; }
#endif

static void fm3d_release_held(fm3d_ctx* c)
{
    for (int i = 0; i < c->nheld; i++) fm3d_texture_release(c->held[i]);
    c->nheld = 0;
#if FM_FEATURE_VBO
    for (int i = 0; i < c->nhbuf; i++) fm3d_buffer_release(c->hbuf[i]);
    c->nhbuf = 0;
#endif
}

void fm3d_destroy(fm3d_ctx* c)
{
    if (!c) return;
    fm3d_flush(c);
    fm3d_texture_release(c->st.tex);
    fm3d_texture_release(c->st.tex1);
#if FM_FEATURE_SHADERS
    for (int u = 1; u < FM3D_MAX_TEXTURE_UNITS; u++) fm3d_texture_release(c->st.units[u]);
#endif
    fm3d_release_held(c);
#if FM_FEATURE_VBO
    free(c->hbuf);
#endif
#if FM_FEATURE_SHADERS
    for (int b = 0; b < FM3D_MAX_UNIFORM_BLOCKS; b++) free(c->uni[b]);
#endif
    for (int i = 0; i < c->nworkers; i++) fm_arena_free(&c->workers[i].arena);
    free(c->workers);
    fm_arena_free(&c->rec);
    fm_arena_free(&c->frame);
    free(c->cmds);
    free(c->draws);
    free(c->held);
    free(c->vbuf);
    free(c->ms_color);
    free(c->ms_depth);
    free(c->ms_stencil);
    free(c);
}

/* (re)allocate the per sample buffers for the current target */
static int fm3d_ms_ensure(fm3d_ctx* c)
{
    if (c->msaa <= 1 || !c->color) return 0;
    if (c->ms_color && c->ms_w == c->color->width && c->ms_h == c->color->height && c->ms_s == c->msaa) return 1;
    free(c->ms_color);
    free(c->ms_depth);
    free(c->ms_stencil);
    size_t n      = (size_t)c->color->width * (size_t)c->color->height * (size_t)c->msaa;
    c->ms_color   = (uint32_t*)calloc(n, sizeof(uint32_t));
    c->ms_depth   = (float*)malloc(n * sizeof(float));
    c->ms_stencil = (uint8_t*)calloc(n, 1);
    if (!c->ms_color || !c->ms_depth || !c->ms_stencil) {
        free(c->ms_color);
        free(c->ms_depth);
        free(c->ms_stencil);
        c->ms_color = NULL;
        c->ms_depth = NULL;
        c->ms_stencil = NULL;
        return 0;
    }
    for (size_t i = 0; i < n; i++) c->ms_depth[i] = 1.0f;
    c->ms_w = c->color->width;
    c->ms_h = c->color->height;
    c->ms_s = c->msaa;
    return 1;
}

/* average the samples of rect r into the color target */
static void fm3d_ms_resolve(fm3d_ctx* c, const int r[4])
{
    for (int y = r[1]; y < r[3]; y++)
        fm_k->resolve(c->ms_color + ((size_t)y * (size_t)c->ms_w + (size_t)r[0]) * (size_t)c->ms_s, c->ms_s,
                      r[2] - r[0], fm_surface_row32(c->color, y) + r[0]);
}

void fm3d_set_msaa(fm3d_ctx* c, int samples)
{
    fm3d_flush(c);
    c->msaa = samples >= 8 ? 8 : (samples >= 4 ? 4 : 1);
}
int fm3d_get_msaa(fm3d_ctx* c) { return c->msaa > 1 ? c->msaa : 1; }

void fm3d_set_target(fm3d_ctx* c, fm_surface* color, fm_surface* depth)
{
    fm3d_flush(c);
    c->color = (color && color->format == FM_FORMAT_ARGB32) ? color : NULL;
    c->depth = (depth && fm_format_is_depth(depth->format) && color && depth->width >= color->width &&
                depth->height >= color->height)
                   ? depth
                   : NULL;
    c->vp_set     = 0;
    c->scissor_on = 0;
    if (c->stencil && (!c->color || c->stencil->width < c->color->width || c->stencil->height < c->color->height))
        c->stencil = NULL;
}

void fm3d_set_stencil_buffer(fm3d_ctx* c, fm_surface* s)
{
    fm3d_flush(c);
    c->stencil = (s && s->format == FM_FORMAT_A8 && c->color && s->width >= c->color->width &&
                  s->height >= c->color->height)
                     ? s
                     : NULL;
}
void fm3d_set_stencil_test(fm3d_ctx* c, int enable) { c->st.stencil_on = enable != 0; }
void fm3d_set_stencil_func(fm3d_ctx* c, fm3d_face face, fm3d_compare func, uint8_t ref, uint8_t read_mask)
{
    for (int f = 0; f < 2; f++)
        if (face & (1 << f)) {
            c->st.stencil[f].func      = func;
            c->st.stencil[f].ref       = ref;
            c->st.stencil[f].read_mask = read_mask;
        }
}
void fm3d_set_stencil_op(fm3d_ctx* c, fm3d_face face, fm3d_stencil_op sfail, fm3d_stencil_op dpfail,
                         fm3d_stencil_op dppass)
{
    for (int f = 0; f < 2; f++)
        if (face & (1 << f)) {
            c->st.stencil[f].sfail  = sfail;
            c->st.stencil[f].dpfail = dpfail;
            c->st.stencil[f].dppass = dppass;
        }
}
void fm3d_set_stencil_write_mask(fm3d_ctx* c, fm3d_face face, uint8_t mask)
{
    for (int f = 0; f < 2; f++)
        if (face & (1 << f)) c->st.stencil[f].write_mask = mask;
}
void fm3d_set_color_write(fm3d_ctx* c, int enable) { c->st.color_write = enable != 0; }
void fm3d_set_depth_bias(fm3d_ctx* c, float factor, float units)
{
    c->st.depth_bias_factor = isfinite(factor) ? factor : 0.0f;
    c->st.depth_bias_units  = isfinite(units) ? units : 0.0f;
}
void fm3d_set_depth_clamp(fm3d_ctx* c, int on) { c->st.depth_clamp = on != 0; }
void fm3d_set_depth_range(fm3d_ctx* c, float n, float f)
{
    c->st.depth_near = FM_CLAMP(n, 0.0f, 1.0f);
    c->st.depth_far  = FM_CLAMP(f, 0.0f, 1.0f);
}

void fm3d_set_model(fm3d_ctx* c, const fm_mat4* m) { c->st.model = m ? *m : fm_mat4_identity(); }
void fm3d_set_view(fm3d_ctx* c, const fm_mat4* m) { c->st.view = m ? *m : fm_mat4_identity(); }
void fm3d_set_projection(fm3d_ctx* c, const fm_mat4* m) { c->st.proj = m ? *m : fm_mat4_identity(); }
void fm3d_set_clip_depth(fm3d_ctx* c, fm3d_clip_depth m) { c->st.clip_depth = m; }
void fm3d_set_origin(fm3d_ctx* c, fm3d_origin o) { c->st.origin = o; }
void fm3d_set_primitive(fm3d_ctx* c, fm3d_primitive p)
{
    if (p >= FM3D_PRIM_TRIANGLES && p <= FM3D_PRIM_POINTS) c->st.prim = p;
}
void fm3d_set_line_width(fm3d_ctx* c, float w) { c->st.line_width = w > 0 ? w : 1.0f; }
void fm3d_set_point_size(fm3d_ctx* c, float s) { c->st.point_size = s > 0 ? s : 1.0f; }
void fm3d_set_viewport(fm3d_ctx* c, int x, int y, int w, int h)
{
    c->st.vp[0] = x;
    c->st.vp[1] = y;
    c->st.vp[2] = FM_MAX(w, 1);
    c->st.vp[3] = FM_MAX(h, 1);
    c->vp_set   = 1;
}
void fm3d_set_scissor(fm3d_ctx* c, int enable, int x, int y, int w, int h)
{
    c->scissor_on = enable;
    c->scissor[0] = x;
    c->scissor[1] = y;
    c->scissor[2] = x + w;
    c->scissor[3] = y + h;
}
void fm3d_set_cull(fm3d_ctx* c, fm3d_cull cull, fm3d_winding front)
{
    c->st.cull  = cull;
    c->st.front = front;
}
void fm3d_set_perspective_correct(fm3d_ctx* c, int on) { c->st.perspective = on != 0; }
void fm3d_set_depth_test(fm3d_ctx* c, fm3d_compare f, int write)
{
    c->st.depth_func  = f;
    c->st.depth_write = write != 0;
}
void fm3d_set_texture(fm3d_ctx* c, fm3d_texture* tex, const fm3d_sampler* s)
{
    fm3d_texture_retain(tex);
    fm3d_texture_release(c->st.tex);
    c->st.tex = tex;
    if (s) c->st.sampler = *s;
}

#if FM_FEATURE_SHADERS
void fm3d_set_texture_unit(fm3d_ctx* c, int unit, fm3d_texture* tex, const fm3d_sampler* s)
{
    if (unit < 0 || unit >= FM3D_MAX_TEXTURE_UNITS) return;
    if (unit == 0) {
        fm3d_set_texture(c, tex, s);
        return;
    }
    fm3d_texture_retain(tex);
    fm3d_texture_release(c->st.units[unit]);
    c->st.units[unit] = tex;
    if (s) c->st.usamp[unit] = *s;
}
#endif
void fm3d_set_texenv(fm3d_ctx* c, fm3d_texenv env) { c->st.texenv = env; }
void fm3d_set_texture_stage1(fm3d_ctx* c, fm3d_texture* tex, const fm3d_sampler* s, fm3d_texenv env)
{
    fm3d_texture_retain(tex);
    fm3d_texture_release(c->st.tex1);
    c->st.tex1 = tex;
    if (s) c->st.sampler1 = *s;
    c->st.texenv1 = env;
}
void fm3d_set_fog(fm3d_ctx* c, fm3d_fog mode, fm_color color, float start, float end, float density)
{
    c->st.fog         = mode;
    c->st.fog_color   = color; /* premultiplied at draw time if the target is */
    c->st.fog_start   = start;
    c->st.fog_end     = end;
    c->st.fog_density = density;
}

#if FM_FEATURE_SHADERS
static void fm3d_draw_impl(fm3d_ctx* c, const void* v, int stride, int nv,
                           const uint32_t* idx, int count, fm3d_buffer* buf);

void fm3d_set_program(fm3d_ctx* c, const fm3d_program* p)
{
    fm3d_dstate* s = &c->st;
    s->user_vs     = p ? p->vs : NULL;
    s->user_fs     = p ? p->fs : NULL;
    s->fs_discards = p && p->fs && p->discards;
    s->user        = p ? p->user : NULL;
    s->vs          = s->user_vs ? fm3d_vs_program : fm3d_vs_fixed;
    s->fs          = s->user_fs ? fm3d_fs_program : fm3d_fs_fixed;
    int nv         = s->user_vs ? p->nvaryings : FM3D_FIXED_NVAR;
    if (!s->user_fs && nv < FM3D_FIXED_NVAR) nv = FM3D_FIXED_NVAR; /* the fixed fragment stage reads 6 */
    s->nvar = nv < 1 ? 1 : (nv > FM3D_MAX_VARYINGS ? FM3D_MAX_VARYINGS : nv);
    int ps = p && s->user_vs ? p->point_size_var - 1 : -1, pc = p && s->user_fs ? p->point_coord_var - 1 : -1;
    s->psize_var  = ps >= 0 && ps < s->nvar ? ps : -1;
    s->pcoord_var = pc >= 0 && pc + 1 < s->nvar ? pc : -1;
}

void fm3d_set_uniform_block(fm3d_ctx* c, int binding, const void* data, size_t bytes)
{
    if (!c || binding < 0 || binding >= FM3D_MAX_UNIFORM_BLOCKS) return;
    if (!data || !bytes || bytes > 65536) {
        c->uni_size[binding]       = 0;
        c->st.blocks[binding]      = NULL;
        c->st.block_sizes[binding] = 0;
    } else {
        if (bytes > c->uni_cap[binding]) {
            void* n = realloc(c->uni[binding], bytes);
            if (!n) return;
            c->uni[binding]     = n;
            c->uni_cap[binding] = bytes;
        }
        memcpy(c->uni[binding], data, bytes);
        c->uni_size[binding]       = bytes;
        c->st.blocks[binding]      = c->uni[binding];
        c->st.block_sizes[binding] = bytes;
    }
    c->st.uniforms     = c->st.blocks[0];
    c->st.uniform_size = c->st.block_sizes[0];
}

void fm3d_set_uniforms(fm3d_ctx* c, const void* data, size_t bytes) { fm3d_set_uniform_block(c, 0, data, bytes); }

void fm3d_set_draw_ids(fm3d_ctx* c, int base_vertex, int instance)
{
    if (!c) return;
    c->st.base_vertex = base_vertex;
    c->st.instance    = instance;
}

void fm3d_draw_vertices(fm3d_ctx* c, const void* v, int stride, int vertex_count, const uint32_t* indices, int index_count)
{
    if (stride <= 0) return;
    fm3d_draw_impl(c, v, stride, vertex_count, indices, indices ? index_count : vertex_count, NULL);
}

void fm3d_sample(const fm3d_texture* t, const fm3d_sampler* s, const float* u, const float* v, int n, float* r, float* g,
                 float* b, float* a)
{
    if (!t || !s || n <= 0) return;
    fm_sampler fs;
    fs.filter = (s->filter == FM3D_FILTER_NEAREST || s->filter == FM3D_FILTER_NEAREST_MIPMAP) ? FM_FILTER_NEAREST
                                                                                               : FM_FILTER_BILINEAR;
    fs.wrap_u = s->wrap_u;
    fs.wrap_v = s->wrap_v;
    const fm_surface* L = t->level[0];
    uint32_t          px[256];
    float             us[256], vs[256];
    for (int i0 = 0; i0 < n; i0 += 256) {
        int m = FM_MIN(256, n - i0);
        for (int i = 0; i < m; i++) { /* texel space */
            us[i] = u[i0 + i] * (float)L->width;
            vs[i] = v[i0 + i] * (float)L->height;
        }
        fm_sample_points(L, &fs, us, vs, m, px);
        for (int i = 0; i < m; i++) { /* premultiplied ARGB -> straight floats */
            uint32_t p  = px[i];
            float    al = (float)(p >> 24) * (1.0f / 255.0f), ia = al > 0 ? 1.0f / (al * 255.0f) : 0.0f;
            r[i0 + i]   = (float)((p >> 16) & 255) * ia;
            g[i0 + i]   = (float)((p >> 8) & 255) * ia;
            b[i0 + i]   = (float)(p & 255) * ia;
            a[i0 + i]   = al;
        }
    }
}

/* per point level of detail (textureLod): mip filters pick / blend levels,
 * others use the base level */
void fm3d_sample_lod(const fm3d_texture* t, const fm3d_sampler* s, const float* U, const float* V, const float* lod, int n,
                     float* r, float* g, float* b, float* a)
{
    if (!t || !s || n <= 0) return;
    fm_sampler fs;
    fs.filter = (s->filter == FM3D_FILTER_NEAREST || s->filter == FM3D_FILTER_NEAREST_MIPMAP) ? FM_FILTER_NEAREST
                                                                                               : FM_FILTER_BILINEAR;
    fs.wrap_u = s->wrap_u;
    fs.wrap_v = s->wrap_v;
    int mip = s->filter >= FM3D_FILTER_NEAREST_MIPMAP && t->levels > 1, tri = s->filter == FM3D_FILTER_TRILINEAR;
    for (int i0 = 0; i0 < n; i0 += 64) {
        int      m = FM_MIN(64, n - i0);
        int      la[64], lb[64];
        float    fw[64];
        uint32_t pa[64], pb[64];
        for (int l = 0; l < m; l++) {
            float lv = lod ? lod[i0 + l] : 0.0f, maxl = (float)(t->levels - 1);
            la[l] = lb[l] = 0;
            fw[l]         = 0;
            if (!mip) continue;
            if (tri) {
                float c = fminf(fmaxf(lv, 0.0f), maxl);
                la[l]   = (int)floorf(c);
                lb[l]   = la[l] + 1 < t->levels ? la[l] + 1 : la[l];
                fw[l]   = c - (float)la[l];
            } else {
                la[l] = lb[l] = (int)floorf(fminf(fmaxf(lv + 0.5f, 0.0f), maxl));
            }
        }
        for (int pass = 0; pass < (tri ? 2 : 1); pass++) { /* the points of one level at a time */
            uint64_t todo = m == 64 ? ~0ull : ((1ull << m) - 1);
            while (todo) {
                int lvl = -1;
                for (int l = 0; l < m && lvl < 0; l++)
                    if ((todo >> l) & 1) lvl = pass ? lb[l] : la[l];
                const fm_surface* L = t->level[lvl];
                float             us[64], vs[64];
                int               ix[64], k = 0;
                for (int l = 0; l < m; l++)
                    if (((todo >> l) & 1) && (pass ? lb[l] : la[l]) == lvl) {
                        us[k] = U[i0 + l] * (float)L->width, vs[k] = V[i0 + l] * (float)L->height, ix[k++] = l;
                        todo &= ~(1ull << l);
                    }
                uint32_t px[64];
                fm_sample_points(L, &fs, us, vs, k, px);
                for (int j = 0; j < k; j++) (pass ? pb : pa)[ix[j]] = px[j];
            }
        }
        for (int l = 0; l < m; l++) { /* premultiplied ARGB -> straight floats, blended across levels */
            float c4[2][4];
            for (int j = 0; j < (tri && fw[l] > 0 ? 2 : 1); j++) {
                uint32_t p  = j ? pb[l] : pa[l];
                float    al = (float)(p >> 24) * (1.0f / 255.0f), ia = al > 0 ? 1.0f / (al * 255.0f) : 0.0f;
                c4[j][0]    = (float)((p >> 16) & 255) * ia;
                c4[j][1]    = (float)((p >> 8) & 255) * ia;
                c4[j][2]    = (float)(p & 255) * ia;
                c4[j][3]    = al;
            }
            if (tri && fw[l] > 0)
                for (int k = 0; k < 4; k++) c4[0][k] += (c4[1][k] - c4[0][k]) * fw[l];
            r[i0 + l] = c4[0][0], g[i0 + l] = c4[0][1], b[i0 + l] = c4[0][2], a[i0 + l] = c4[0][3];
        }
    }
}
#endif

#if FM_FEATURE_TNL
fm3d_light fm3d_light_default(fm3d_light_type type)
{
    fm3d_light l;
    memset(&l, 0, sizeof(l));
    l.type          = type;
    l.direction     = fm_v3(0, 0, -1);
    l.diffuse       = fm_v3(1, 1, 1);
    l.specular      = fm_v3(1, 1, 1);
    l.constant      = 1.0f;
    l.spot_cutoff   = 0.5f;
    return l;
}

fm3d_material fm3d_material_default(void)
{
    fm3d_material m;
    memset(&m, 0, sizeof(m));
    m.ambient = fm_v3(0.2f, 0.2f, 0.2f);
    m.diffuse = fm_v3(0.8f, 0.8f, 0.8f);
    m.alpha   = 1.0f;
    return m;
}

void fm3d_set_lighting(fm3d_ctx* c, int on) { c->lighting = on != 0; }
void fm3d_set_light(fm3d_ctx* c, int i, const fm3d_light* l)
{
    if (i < 0 || i >= FM3D_MAX_LIGHTS) return;
    c->light_on[i] = l != NULL;
    if (l) c->lights[i] = *l;
}
void fm3d_set_material(fm3d_ctx* c, const fm3d_material* m)
{
    if (m) c->material = *m;
}
void fm3d_set_ambient_light(fm3d_ctx* c, fm_vec3 col) { c->ambient_light = col; }
void fm3d_set_color_material(fm3d_ctx* c, int on) { c->color_material = on != 0; }

static fm_vec3 fm3d_xform_dir(const fm_mat4* m, fm_vec3 d) /* upper 3x3 */
{
    return fm_v3(m->c[0].x * d.x + m->c[1].x * d.y + m->c[2].x * d.z, m->c[0].y * d.x + m->c[1].y * d.y + m->c[2].y * d.z,
                 m->c[0].z * d.x + m->c[1].z * d.y + m->c[2].z * d.z);
}

/* lights to eye space, material, normal matrix */
static void fm3d_resolve_lighting(fm3d_ctx* c, fm3d_dstate* s)
{
    s->lp = NULL;
    if (!c->lighting) return;
    fm_light_params* p = &c->lp;
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < FM3D_MAX_LIGHTS; i++) {
        if (!c->light_on[i]) continue;
        const fm3d_light*  L = &c->lights[i];
        struct fm_light_k* k = &p->l[p->nlights++];
        k->type              = (int)L->type;
        fm_vec4 pe           = fm_mat4_mul_vec4(s->view, fm_v4(L->position.x, L->position.y, L->position.z, 1));
        k->pos[0] = pe.x, k->pos[1] = pe.y, k->pos[2] = pe.z;
        fm_vec3 d = fm_v3_normalize(fm3d_xform_dir(&s->view, L->direction));
        if (L->type == FM3D_LIGHT_DIRECTIONAL) d = fm_v3_negate(d); /* kernel: towards the light */
        k->dir[0] = d.x, k->dir[1] = d.y, k->dir[2] = d.z;
        float a[3] = { L->ambient.x, L->ambient.y, L->ambient.z }, df[3] = { L->diffuse.x, L->diffuse.y, L->diffuse.z };
        float sp[3] = { L->specular.x, L->specular.y, L->specular.z };
        for (int j = 0; j < 3; j++) k->amb[j] = a[j], k->dif[j] = df[j], k->spe[j] = sp[j];
        k->katt[0]  = L->constant, k->katt[1] = L->linear, k->katt[2] = L->quadratic;
        k->spot_cos = cosf(L->spot_cutoff);
        k->spot_exp = L->spot_exponent;
    }
    const fm3d_material* m = &c->material;
    float                ma[3] = { m->ambient.x, m->ambient.y, m->ambient.z }, md[3] = { m->diffuse.x, m->diffuse.y, m->diffuse.z };
    float                ms[3] = { m->specular.x, m->specular.y, m->specular.z }, me[3] = { m->emission.x, m->emission.y, m->emission.z };
    float                ga[3] = { c->ambient_light.x, c->ambient_light.y, c->ambient_light.z };
    for (int j = 0; j < 3; j++) p->mat_amb[j] = ma[j], p->mat_dif[j] = md[j], p->mat_spe[j] = ms[j], p->mat_emi[j] = me[j], p->gamb[j] = ga[j];
    p->mat_dif[3]     = m->alpha;
    p->shininess      = m->shininess;
    p->color_material = c->color_material;
    s->mv             = fm_mat4_mul(s->view, s->model);
    fm_mat4 it        = fm_mat4_transpose(fm_mat4_inverse(s->mv));
    for (int col = 0; col < 3; col++) {
        s->nrm[3 * col + 0] = it.c[col].x;
        s->nrm[3 * col + 1] = it.c[col].y;
        s->nrm[3 * col + 2] = it.c[col].z;
    }
    s->lp = p;
}
#endif
void fm3d_set_alpha_test(fm3d_ctx* c, fm3d_compare f, float ref)
{
    c->st.alpha_func = f;
    c->st.alpha_ref8 = (uint32_t)(FM_CLAMP(ref, 0.0f, 1.0f) * 255.0f + 0.5f);
}
void fm3d_set_blend(fm3d_ctx* c, fm_blend_op op)
{
    if (op >= 0 && op < FM_OP_COUNT) c->st.op = op;
}
void fm3d_set_blend_state(fm3d_ctx* c, const fm3d_blend_state* s)
{
    c->st.straight = s != NULL;
    if (!s) return;
    fm_glblend* g = &c->st.gb;
    g->src_rgb    = (uint8_t)s->src_rgb, g->dst_rgb = (uint8_t)s->dst_rgb;
    g->src_a      = (uint8_t)s->src_alpha, g->dst_a = (uint8_t)s->dst_alpha;
    g->eq_rgb     = (uint8_t)s->eq_rgb, g->eq_a = (uint8_t)s->eq_alpha;
    g->constant   = s->constant;
}
void fm3d_set_opacity(fm3d_ctx* c, float a) { c->st.opacity8 = (uint32_t)(FM_CLAMP(a, 0.0f, 1.0f) * 255.0f + 0.5f); }
void fm3d_set_deferred(fm3d_ctx* c, int on)
{
    if (!on) fm3d_flush(c);
    c->deferred = on != 0;
}
void fm3d_set_executor(fm3d_ctx* c, fm_executor* ex) { c->exec = ex; }
void fm3d_set_tile_size(fm3d_ctx* c, int px)
{
    fm3d_flush(c);
    px      = FM_CLAMP(px, 8, 1024);
    c->tile = (px + 1) & ~1;
}

fm3d_stats fm3d_get_stats(fm3d_ctx* c)
{
    fm3d_stats s = c->stats; /* immediate mode fragment counters live in the batch */
    s.fragments_in += c->batch.frag_in;
    s.fragments_shaded += c->batch.frag_shaded;
    return s;
}
void fm3d_reset_stats(fm3d_ctx* c)
{
    memset(&c->stats, 0, sizeof(c->stats));
    c->batch.frag_in = c->batch.frag_shaded = 0;
}

static void fm3d_stats_add(fm3d_stats* d, const fm3d_stats* s)
{
    d->triangles_in += s->triangles_in;
    d->triangles_clipped += s->triangles_clipped;
    d->triangles_culled += s->triangles_culled;
    d->triangles_drawn += s->triangles_drawn;
    d->hiz_rejected += s->hiz_rejected;
    d->fragments_in += s->fragments_in;
    d->fragments_shaded += s->fragments_shaded;
}

/* resolve derived state for a draw / clear */
static int fm3d_resolve(fm3d_ctx* c, fm3d_dstate* s)
{
    if (!c->color) return 0;
    *s             = c->st;
    s->color       = c->color;
    s->depth       = c->depth;
    s->stencil_buf = c->stencil ? c->stencil : ((c->depth && c->depth->format == FM_FORMAT_D24S8) ? c->depth : NULL);
    s->msaa        = 1;
    if (c->msaa > 1 && fm3d_ms_ensure(c)) {
        s->msaa       = c->msaa;
        s->ms_color   = c->ms_color;
        s->ms_depth   = c->ms_depth;
        s->ms_stencil = c->ms_stencil;
        s->ms_w       = c->ms_w;
    }
    if (!c->vp_set) {
        s->vp[0] = 0;
        s->vp[1] = 0;
        s->vp[2] = c->color->width;
        s->vp[3] = c->color->height;
    }
    s->mvp     = fm_mat4_mul(s->proj, fm_mat4_mul(s->view, s->model));
#if FM_FEATURE_TNL
    fm3d_resolve_lighting(c, s);
#endif
    int* r     = s->rect;
    r[0]       = FM_MAX(0, s->vp[0]);
    r[1]       = FM_MAX(0, s->vp[1]);
    r[2]       = FM_MIN(c->color->width, s->vp[0] + s->vp[2]);
    r[3]       = FM_MIN(c->color->height, s->vp[1] + s->vp[3]);
    if (c->scissor_on) {
        r[0] = FM_MAX(r[0], c->scissor[0]);
        r[1] = FM_MAX(r[1], c->scissor[1]);
        r[2] = FM_MIN(r[2], c->scissor[2]);
        r[3] = FM_MIN(r[3], c->scissor[3]);
    }
    return r[2] > r[0] && r[3] > r[1];
}

/* ---- clears ----------------------------------------------------------------------------- */

static void fm3d_clear_rect_ms(fm3d_ctx* c, int type, const int r[4], uint32_t col, float d)
{
    int S = c->ms_s;
    for (int y = r[1]; y < r[3]; y++) {
        size_t i0 = ((size_t)y * (size_t)c->ms_w + (size_t)r[0]) * (size_t)S, n = (size_t)(r[2] - r[0]) * (size_t)S;
        if (type == FM3D_CMD_CLEAR_COLOR)
            fm_fill_span(c->ms_color + i0, col, (int)n);
        else if (type == FM3D_CMD_CLEAR_STENCIL)
            memset(c->ms_stencil + i0, (int)(col & 255), n);
        else {
            float dc = FM_CLAMP(d, 0.0f, 1.0f);
            for (size_t k = 0; k < n; k++) c->ms_depth[i0 + k] = dc;
        }
    }
}

static void fm3d_clear_rect(fm_surface* color, fm_surface* depth, fm_surface* stencil, int type, const int r[4],
                            uint32_t col, float d)
{
    for (int y = r[1]; y < r[3]; y++) {
        if (type == FM3D_CMD_CLEAR_COLOR)
            fm_fill_span(fm_surface_row32(color, y) + r[0], col, r[2] - r[0]);
        else if (type == FM3D_CMD_CLEAR_STENCIL) {
            if (stencil) {
                memset(fm_surface_row8(stencil, y) + r[0], (int)(col & 255), (size_t)(r[2] - r[0]));
            } else if (depth && depth->format == FM_FORMAT_D24S8) {
                uint32_t* w = (uint32_t*)fm_surface_row8(depth, y);
                for (int x = r[0]; x < r[2]; x++) w[x] = (w[x] & 0xffffffu) | ((col & 255u) << 24);
            }
        } else if (depth) {
            uint8_t* row = fm_surface_row8(depth, y);
            float    dc  = FM_CLAMP(d, 0.0f, 1.0f);
            uint32_t k   = fm3d_zkey(depth->format, dc);
            switch (depth->format) {
            case FM_FORMAT_D16:
                for (int x = r[0]; x < r[2]; x++) ((uint16_t*)row)[x] = (uint16_t)k;
                break;
            case FM_FORMAT_D24S8:
                for (int x = r[0]; x < r[2]; x++) ((uint32_t*)row)[x] = (((uint32_t*)row)[x] & 0xff000000u) | k;
                break;
            default:
                for (int x = r[0]; x < r[2]; x++) ((float*)row)[x] = dc;
                break;
            }
        }
    }
}

static void fm3d_push_cmd(fm3d_ctx* c, const fm3d_cmd* cmd)
{
    if (c->ncmd == c->ccmd) {
        int       nc = c->ccmd ? c->ccmd * 2 : 64;
        fm3d_cmd* n  = (fm3d_cmd*)realloc(c->cmds, (size_t)nc * sizeof(fm3d_cmd));
        if (!n) return;
        c->cmds = n;
        c->ccmd = nc;
    }
    c->cmds[c->ncmd++] = *cmd;
}

static void fm3d_clear_impl(fm3d_ctx* c, int type, uint32_t col, float d)
{
    fm3d_dstate s;
    if (!fm3d_resolve(c, &s)) return;
    int r[4] = { 0, 0, c->color->width, c->color->height };
    if (c->scissor_on) {
        r[0] = FM_MAX(r[0], c->scissor[0]);
        r[1] = FM_MAX(r[1], c->scissor[1]);
        r[2] = FM_MIN(r[2], c->scissor[2]);
        r[3] = FM_MIN(r[3], c->scissor[3]);
    }
    if (r[2] <= r[0] || r[3] <= r[1]) return;
    if (c->deferred) {
        fm3d_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type  = type;
        cmd.color = col;
        cmd.depth = d;
        memcpy(cmd.rect, r, sizeof(r));
        fm3d_push_cmd(c, &cmd);
        return;
    }
    if (c->msaa > 1 && fm3d_ms_ensure(c))
        fm3d_clear_rect_ms(c, type, r, col, d);
    else
        fm3d_clear_rect(c->color, c->depth, c->stencil, type, r, col, d);
}

void fm3d_clear_color(fm3d_ctx* c, fm_color col)
{
    fm3d_clear_impl(c, FM3D_CMD_CLEAR_COLOR, c->st.straight ? col : fm_premultiply(col), 0);
}
void fm3d_clear_depth(fm3d_ctx* c, float d) { fm3d_clear_impl(c, FM3D_CMD_CLEAR_DEPTH, 0, d); }
void fm3d_clear_stencil(fm3d_ctx* c, uint8_t v) { fm3d_clear_impl(c, FM3D_CMD_CLEAR_STENCIL, v, 0); }

/* ---- immediate draws -------------------------------------------------------------------- */

static void fm3d_emit_now(fm3d_sink* s, fm3d_tri* t)
{
    fm3d_ctx* c = (fm3d_ctx*)s->user;
    fm3d_raster_tri(t, t->st->rect, &c->batch);
}

/* buf != NULL: v / idx point into that (immutable, validated) buffer.
 * v holds nv vertices of `stride` bytes (fm3d_vertex unless a program
 * vertex shader is bound). */
static void fm3d_draw_impl(fm3d_ctx* c, const void* v, int stride, int nv, const uint32_t* idx,
                           int count, fm3d_buffer* buf)
{
    int per = c->st.prim == FM3D_PRIM_LINES ? 2 : (c->st.prim == FM3D_PRIM_POINTS ? 1 : 3);
    if (!v || count < per || nv <= 0) return;
    int         ntri = count / per; /* primitives */
    fm3d_dstate s;
    if (!fm3d_resolve(c, &s)) return;
    s.vstride = stride;
#if FM_FEATURE_SHADERS
    if (!s.user_vs && stride != (int)sizeof(fm3d_vertex) && stride != (int)sizeof(fm3d_vertex_mt))
        return; /* the fixed stage reads fm3d_vertex (_mt) */
#else
    if (stride != (int)sizeof(fm3d_vertex) && stride != (int)sizeof(fm3d_vertex_mt)) return;
#endif
    if (s.vs == fm3d_vs_fixed && s.nvar < FM3D_FIXED_NVAR_MT) s.nvar = FM3D_FIXED_NVAR_MT; /* the second coordinates */
    if (s.straight || s.fog == FM3D_FOG_OFF) {
    } else {
        s.fog_color = fm_premultiply(s.fog_color);
    }
    if (idx && !buf)
        for (int i = 0; i < ntri * per; i++)
            if (idx[i] >= (uint32_t)nv) return; /* reject out of range indices */

    if (c->deferred) {
        FM_PROF_BEGIN(z, "3d.record");
        fm3d_dstate* st = (fm3d_dstate*)fm_arena_alloc(&c->rec, sizeof(fm3d_dstate));
        void*        vc = buf ? (void*)v : fm_arena_alloc(&c->rec, (size_t)nv * (size_t)stride);
        uint32_t*    ic = (idx && !buf) ? (uint32_t*)fm_arena_alloc(&c->rec, (size_t)ntri * (size_t)per * sizeof(uint32_t))
                                        : (uint32_t*)idx;
        if (!st || !vc || (idx && !ic)) {
            FM_PROF_END(z);
            return;
        }
#if FM_FEATURE_VBO
        if (buf) { /* keep the buffer alive until the flush instead of copying */
            if (c->nhbuf == c->chbuf) {
                int           nc = c->chbuf ? c->chbuf * 2 : 16;
                fm3d_buffer** n  = (fm3d_buffer**)realloc(c->hbuf, (size_t)nc * sizeof(fm3d_buffer*));
                if (!n) {
                    FM_PROF_END(z);
                    return;
                }
                c->hbuf  = n;
                c->chbuf = nc;
            }
            c->hbuf[c->nhbuf++] = fm3d_buffer_retain(buf);
        }
#endif
        *st = s;
#if FM_FEATURE_TNL
        if (s.lp) { /* snapshot, shared with the previous draw when equal (lights rarely change) */
            if (!c->lp_rec || memcmp(c->lp_rec, s.lp, sizeof(fm_light_params)) != 0) {
                c->lp_rec = (fm_light_params*)fm_arena_alloc(&c->rec, sizeof(fm_light_params));
                if (!c->lp_rec) {
                    FM_PROF_END(z);
                    return;
                }
                memcpy(c->lp_rec, s.lp, sizeof(fm_light_params));
            }
            st->lp = c->lp_rec;
        }
#endif
        if (!buf) memcpy(vc, v, (size_t)nv * (size_t)stride);
#if FM_FEATURE_SHADERS
        for (int ub = 0; ub < FM3D_MAX_UNIFORM_BLOCKS; ub++) { /* snapshots, shared with the previous draw when unchanged */
            if (!s.blocks[ub]) continue;
            if (!c->uni_rec[ub] || c->uni_rec_size[ub] != c->uni_size[ub] || memcmp(c->uni_rec[ub], c->uni[ub], c->uni_size[ub]) != 0) {
                c->uni_rec[ub] = fm_arena_alloc(&c->rec, c->uni_size[ub]);
                if (!c->uni_rec[ub]) {
                    FM_PROF_END(z);
                    return;
                }
                memcpy(c->uni_rec[ub], c->uni[ub], c->uni_size[ub]);
                c->uni_rec_size[ub] = c->uni_size[ub];
            }
            st->blocks[ub] = c->uni_rec[ub];
        }
        st->uniforms = st->blocks[0];
#endif
        if (idx && !buf) memcpy(ic, idx, (size_t)ntri * (size_t)per * sizeof(uint32_t));
        for (int u = 0; u <= FM3D_NUNITS; u++) { /* u == FM3D_NUNITS: the fixed second stage */
#if FM_FEATURE_SHADERS
            fm3d_texture* ut = u == FM3D_NUNITS ? s.tex1 : (u ? s.units[u] : s.tex);
#else
            fm3d_texture* ut = u == FM3D_NUNITS ? s.tex1 : (u ? NULL : s.tex);
#endif
            if (!ut) continue;
            if (c->nheld == c->cheld) {
                int             nc = c->cheld ? c->cheld * 2 : 16;
                fm3d_texture** n  = (fm3d_texture**)realloc(c->held, (size_t)nc * sizeof(fm3d_texture*));
                if (!n) {
                    FM_PROF_END(z);
                    return;
                }
                c->held  = n;
                c->cheld = nc;
            }
            c->held[c->nheld++] = fm3d_texture_retain(ut);
        }
        if (c->ndraw == c->cdraw) {
            int           nc = c->cdraw ? c->cdraw * 2 : 64;
            fm3d_drawrec* n  = (fm3d_drawrec*)realloc(c->draws, (size_t)nc * sizeof(fm3d_drawrec));
            if (!n) {
                FM_PROF_END(z);
                return;
            }
            c->draws = n;
            c->cdraw = nc;
        }
        fm3d_drawrec* d = &c->draws[c->ndraw];
        memset(d, 0, sizeof(*d));
        d->st   = st;
        d->v    = vc;
        d->nv   = nv;
        d->idx  = ic;
        d->ntri = ntri;
        fm3d_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = FM3D_CMD_DRAW;
        cmd.draw = c->ndraw++;
        fm3d_push_cmd(c, &cmd);
        FM_PROF_END(z);
        return;
    }

    FM_PROF_BEGIN(zv, "3d.vertex");
    if (nv > c->vbuf_cap) {
        fm3d_vout* n = (fm3d_vout*)realloc(c->vbuf, (size_t)nv * sizeof(fm3d_vout));
        if (!n) {
            FM_PROF_END(zv);
            return;
        }
        c->vbuf     = n;
        c->vbuf_cap = nv;
    }
    s.vs(&s, v, nv, 0, c->vbuf);
    FM_PROF_END(zv);

    FM_PROF_BEGIN(zr, "3d.draw");
    fm3d_sink sink;
    memset(&sink, 0, sizeof(sink));
    c->batch.hiz = NULL;
    sink.emit = fm3d_emit_now;
    sink.user = c;
    for (int i = 0; i < ntri; i++) fm3d_process_prim(&s, c->vbuf, idx, i, &sink);
    fm3d_stats_add(&c->stats, &sink.stats);
    FM_PROF_ITEMS(zr, ntri);
    FM_PROF_END(zr);
}

void fm3d_draw(fm3d_ctx* c, const fm3d_vertex* v, int count)
{
    fm3d_draw_impl(c, v, (int)sizeof(fm3d_vertex), count, NULL, count, NULL);
}

#if FM_FEATURE_VBO
void fm3d_draw_buffer(fm3d_ctx* c, fm3d_buffer* b, int first, int count)
{
    if (!b || first < 0 || count < 3) return;
    if (b->idx) {
        if (first > b->ni - count) return;
        fm3d_draw_impl(c, b->v, (int)sizeof(fm3d_vertex), b->nv, b->idx + first, count, b);
    } else {
        if (first > b->nv - count) return;
        fm3d_draw_impl(c, b->v + first, (int)sizeof(fm3d_vertex), count, NULL, count, b);
    }
}
#endif

void fm3d_draw_mt(fm3d_ctx* c, const fm3d_vertex_mt* v, int vertex_count, const uint32_t* indices, int index_count)
{
    fm3d_draw_impl(c, v, (int)sizeof(fm3d_vertex_mt), vertex_count, indices, indices ? index_count : vertex_count, NULL);
}

void fm3d_draw_indexed(fm3d_ctx* c, const fm3d_vertex* v, int vertex_count, const uint32_t* indices, int index_count)
{
    if (indices) fm3d_draw_impl(c, v, (int)sizeof(fm3d_vertex), vertex_count, indices, index_count, NULL);
}

/* ---- deferred flush ------------------------------------------------------------------------ */

static void fm3d_run(fm_executor* ex, fm_task_fn fn, void* arg, int count)
{
    if (!ex || ex->workers <= 1 || count <= 1) {
        for (int i = 0; i < count; i++) fn(arg, i, 0);
        return;
    }
    ex->parallel_for(ex, fn, arg, count);
}

static void fm3d_phase_vertex(void* arg, int index, int worker)
{
    (void)worker;
    fm3d_ctx*         c = (fm3d_ctx*)arg;
    const fm3d_vtask* t = &c->vtasks[index];
    fm3d_drawrec*     d = &c->draws[t->draw];
    d->st->vs(d->st, (const char*)d->v + (size_t)t->v0 * (size_t)d->st->vstride, t->n, t->v0, d->vout + t->v0);
}

typedef struct fm3d_binsink {
    fm3d_sink   base;
    fm3d_chunk* chunk;
    int         cap;
} fm3d_binsink;

static void fm3d_emit_bin(fm3d_sink* s, fm3d_tri* t)
{
    fm3d_binsink* b  = (fm3d_binsink*)s;
    fm3d_chunk*   ch = b->chunk;
    if (ch->ntris == b->cap) {
        int        nc = b->cap * 2;
        fm3d_tri** n  = (fm3d_tri**)fm_arena_alloc(s->arena, (size_t)nc * sizeof(fm3d_tri*));
        if (!n) return;
        memcpy(n, ch->tris, (size_t)ch->ntris * sizeof(fm3d_tri*));
        ch->tris = n;
        b->cap   = nc;
    }
    ch->tris[ch->ntris++] = t;
}

static void fm3d_phase_setup(void* arg, int index, int worker)
{
    fm3d_ctx*     c  = (fm3d_ctx*)arg;
    fm3d_worker*  w  = &c->workers[worker];
    fm3d_chunk*   ch = &c->chunks[index];
    fm3d_drawrec* d  = &c->draws[ch->draw];
    fm3d_binsink  bs;
    memset(&bs.base.stats, 0, sizeof(bs.base.stats));
    bs.base.arena = &w->arena;
    bs.base.emit  = fm3d_emit_bin;
    bs.base.user  = c;
    bs.chunk      = ch;
    bs.cap        = ch->ntri + 8;
    ch->ntris     = 0;
    ch->tris      = (fm3d_tri**)fm_arena_alloc(&w->arena, (size_t)bs.cap * sizeof(fm3d_tri*));
    if (!ch->tris) return;
    for (int i = ch->tri0; i < ch->tri0 + ch->ntri; i++) fm3d_process_prim(d->st, d->vout, d->idx, i, &bs.base);
    fm3d_stats_add(&w->stats, &bs.base.stats);

    /* bin: count per tile, prefix sum, fill (keeps submission order). Bins
     * only cover the chunk's tile bounding box, so small draws stay cheap
     * and tiles outside it skip the chunk with one compare. */
    ch->bin_start = NULL;
    if (ch->ntris == 0) return;
    int T = c->tile, bx0 = 1 << 30, by0 = 1 << 30, bx1 = -1, by1 = -1;
    for (int k = 0; k < ch->ntris; k++) {
        const fm3d_tri* t = ch->tris[k];
        bx0 = FM_MIN(bx0, t->minx / T);
        by0 = FM_MIN(by0, t->miny / T);
        bx1 = FM_MAX(bx1, t->maxx / T);
        by1 = FM_MAX(by1, t->maxy / T);
    }
    int       bw = bx1 - bx0 + 1, bh = by1 - by0 + 1, nb = bw * bh;
    uint32_t* start = (uint32_t*)fm_arena_alloc(&w->arena, (size_t)(nb + 1) * sizeof(uint32_t));
    uint32_t* fill  = (uint32_t*)fm_arena_alloc(&w->arena, (size_t)nb * sizeof(uint32_t));
    if (!start || !fill) {
        ch->ntris = 0;
        return;
    }
    memset(start, 0, (size_t)(nb + 1) * sizeof(uint32_t));
    for (int k = 0; k < ch->ntris; k++) {
        const fm3d_tri* t = ch->tris[k];
        for (int ty = t->miny / T; ty <= t->maxy / T; ty++)
            for (int tx = t->minx / T; tx <= t->maxx / T; tx++) start[(ty - by0) * bw + (tx - bx0) + 1]++;
    }
    for (int i = 0; i < nb; i++) start[i + 1] += start[i];
    uint32_t* bin = (uint32_t*)fm_arena_alloc(&w->arena, (size_t)FM_MAX(start[nb], 1u) * sizeof(uint32_t));
    if (!bin) {
        ch->ntris = 0;
        return;
    }
    memcpy(fill, start, (size_t)nb * sizeof(uint32_t));
    for (int k = 0; k < ch->ntris; k++) {
        const fm3d_tri* t = ch->tris[k];
        for (int ty = t->miny / T; ty <= t->maxy / T; ty++)
            for (int tx = t->minx / T; tx <= t->maxx / T; tx++) bin[fill[(ty - by0) * bw + (tx - bx0)]++] = (uint32_t)k;
    }
    ch->btx       = bx0;
    ch->bty       = by0;
    ch->btw       = bw;
    ch->bth       = bh;
    ch->bin       = bin;
    ch->bin_start = start;
}

/* exact bounds of the depth keys stored in rect r */
static void fm3d_hiz_scan(const fm_surface* D, const int r[4], fm3d_hiz* h)
{
    uint32_t kmin = 0xffffffffu, kmax = 0;
    int      n    = r[2] - r[0];
    if (D->format == FM_FORMAT_D32F) {
        /* stored depths are clamped to [0, 1]: float order = key order */
        float fmin = 2.0f, fmax = -1.0f;
        for (int y = r[1]; y < r[3]; y++) {
            float a, b;
            fm_k->minmax_f32(fm_surface_rowf(D, y) + r[0], n, &a, &b);
            fmin = a < fmin ? a : fmin;
            fmax = b > fmax ? b : fmax;
        }
        kmin = fm3d_fkey(fmin);
        kmax = fm3d_fkey(fmax);
    } else if (D->format == FM_FORMAT_D16) {
        for (int y = r[1]; y < r[3]; y++) {
            const uint16_t* row = (const uint16_t*)fm_surface_row8(D, y) + r[0];
            for (int x = 0; x < n; x++) {
                kmin = FM_MIN(kmin, (uint32_t)row[x]);
                kmax = FM_MAX(kmax, (uint32_t)row[x]);
            }
        }
    } else {
        for (int y = r[1]; y < r[3]; y++) {
            const uint32_t* row = fm_surface_row32(D, y) + r[0];
            for (int x = 0; x < n; x++) {
                uint32_t k = row[x] & 0xffffffu;
                kmin       = FM_MIN(kmin, k);
                kmax       = FM_MAX(kmax, k);
            }
        }
    }
    h->kmin    = kmin;
    h->kmax    = kmax;
    h->written = 0;
    h->valid   = 1;
}

/* can hierarchical z decide this draw? (no side effects when skipping) */
static int fm3d_hiz_eligible(const fm3d_dstate* st)
{
    if (!st->depth || (st->stencil_on && st->stencil_buf)) return 0;
    switch (st->depth_func) {
    case FM3D_NEVER:
    case FM3D_LESS:
    case FM3D_LEQUAL:
    case FM3D_GREATER:
    case FM3D_GEQUAL:
    case FM3D_EQUAL: return 1;
    default: return 0;
    }
}

/* 1 if every fragment of t inside r fails the depth test against bounds h */
static int fm3d_hiz_reject(const fm3d_tri* t, const int r[4], const fm3d_hiz* h)
{
    const fm3d_dstate* st = t->st;
    int                x0 = FM_MAX(r[0], t->minx), x1 = FM_MIN(r[2] - 1, t->maxx);
    int                y0 = FM_MAX(r[1], t->miny), y1 = FM_MIN(r[3] - 1, t->maxy);
    if (x1 < x0 || y1 < y0) return 1; /* nothing to draw here anyway */
    if (st->depth_func == FM3D_NEVER) return 1;
    /* the plane is linear: extremes at the rect corners (pixel centers) */
    float zmin = 1e30f, zmax = -1e30f;
    for (int k = 0; k < 4; k++) {
        float px = (float)((k & 1) ? x1 : x0) + 0.5f - t->x0f;
        float py = (float)((k & 2) ? y1 : y0) + 0.5f - t->y0f;
        float z  = (t->z[0] + t->z[2] * py) + t->z[1] * px;
        zmin     = FM_MIN(zmin, z);
        zmax     = FM_MAX(zmax, z);
    }
    zmin -= 1e-6f; /* margin for per pixel rounding */
    zmax += 1e-6f;
    fm_format f  = st->depth->format;
    uint32_t  lo = fm3d_zkey(f, FM_CLAMP(zmin, 0.0f, 1.0f)), hi = fm3d_zkey(f, FM_CLAMP(zmax, 0.0f, 1.0f));
    switch (st->depth_func) {
    case FM3D_LESS: return lo >= h->kmax;
    case FM3D_LEQUAL: return lo > h->kmax;
    case FM3D_GREATER: return hi <= h->kmin;
    case FM3D_GEQUAL: return hi < h->kmin;
    case FM3D_EQUAL: return hi < h->kmin || lo > h->kmax;
    default: return 0;
    }
}

#ifndef FM3D_TILE_PREFETCH
#  define FM3D_TILE_PREFETCH 1
#endif

#define FM3D_TL_CLEAR 0x80000000u

/* tile range [t0, t1] x [u0, u1] covered by the pixel rect r (exclusive
 * max); 0 if empty */
static int fm3d_rect_tiles(const fm3d_ctx* c, const int r[4], int* t0, int* u0, int* t1, int* u1)
{
    int x0 = FM_MAX(r[0], 0), y0 = FM_MAX(r[1], 0);
    int x1 = FM_MIN(r[2], c->color->width), y1 = FM_MIN(r[3], c->color->height);
    if (x1 <= x0 || y1 <= y0) return 0;
    *t0 = x0 / c->tile, *u0 = y0 / c->tile, *t1 = (x1 - 1) / c->tile, *u1 = (y1 - 1) / c->tile;
    return 1;
}

/* the tile range of work item i of the list pass (clear or chunk) */
static int fm3d_item_tiles(const fm3d_ctx* c, int ci, int k, int* t0, int* u0, int* t1, int* u1)
{
    const fm3d_cmd* cmd = &c->cmds[ci];
    if (cmd->type != FM3D_CMD_DRAW) return fm3d_rect_tiles(c, cmd->rect, t0, u0, t1, u1);
    const fm3d_chunk* ch = &c->chunks[k];
    if (!ch->bin_start) return 0;
    *t0 = FM_MAX(ch->btx, 0), *u0 = FM_MAX(ch->bty, 0);
    *t1 = FM_MIN(ch->btx + ch->btw - 1, c->tiles_x - 1), *u1 = FM_MIN(ch->bty + ch->bth - 1, c->tiles_y - 1);
    return *t0 <= *t1 && *u0 <= *u1;
}

/* Build the per tile work lists (count, prefix sum, fill: keeps order).
 * Without them every tile walked every command and chunk of the frame,
 * which dominated phase C for scenes with many draws. */
static int fm3d_build_tile_lists(fm3d_ctx* c)
{
    int       nt    = c->tiles_x * c->tiles_y;
    uint32_t* start = (uint32_t*)fm_arena_alloc(&c->frame, (size_t)(nt + 1) * sizeof(uint32_t));
    uint32_t* fill  = (uint32_t*)fm_arena_alloc(&c->frame, (size_t)nt * sizeof(uint32_t));
    if (!start || !fill) return 0;
    memset(start, 0, (size_t)(nt + 1) * sizeof(uint32_t));
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < c->ncmd; i++) {
            const fm3d_cmd* cmd = &c->cmds[i];
            int             k0 = 0, k1 = 1;
            if (cmd->type == FM3D_CMD_DRAW) {
                const fm3d_drawrec* d = &c->draws[cmd->draw];
                k0 = d->chunk0, k1 = d->chunk0 + d->nchunks;
            }
            for (int k = k0; k < k1; k++) {
                int t0, u0, t1, u1;
                if (!fm3d_item_tiles(c, i, k, &t0, &u0, &t1, &u1)) continue;
                uint32_t item = cmd->type == FM3D_CMD_DRAW ? (uint32_t)k : (FM3D_TL_CLEAR | (uint32_t)i);
                for (int u = u0; u <= u1; u++)
                    for (int t = t0; t <= t1; t++) {
                        int ti = u * c->tiles_x + t;
                        if (pass == 0)
                            start[ti + 1]++;
                        else
                            c->tl[fill[ti]++] = item;
                    }
            }
        }
        if (pass == 0) {
            for (int i = 0; i < nt; i++) start[i + 1] += start[i];
            c->tl = (uint32_t*)fm_arena_alloc(&c->frame, (size_t)FM_MAX(start[nt], 1u) * sizeof(uint32_t));
            if (!c->tl) return 0;
            memcpy(fill, start, (size_t)nt * sizeof(uint32_t));
        }
    }
    c->tl_start = start;
    return 1;
}

static void fm3d_phase_tile(void* arg, int tile, int worker)
{
    fm3d_ctx*    c  = (fm3d_ctx*)arg;
    fm3d_worker* w  = &c->workers[worker];
    int          T  = c->tile;
    int          tx = tile % c->tiles_x, ty = tile / c->tiles_x;
    int          tr[4] = { tx * T, ty * T, FM_MIN((tx + 1) * T, c->color->width), FM_MIN((ty + 1) * T, c->color->height) };
    int          area  = (tr[2] - tr[0]) * (tr[3] - tr[1]);
    fm3d_hiz*    hz    = (c->hiz && c->depth) ? &c->hiz[tile] : NULL;
    w->batch.hiz       = hz;
#if FM3D_TILE_PREFETCH /* measured: up to 8 % single threaded on fill heavy scenes, never slower */
    /* every tile row is on its own page (pitch > 4 KB), where the hardware
     * prefetchers stop: request all lines of the tile up front */
    for (int y = tr[1]; y < tr[3]; y++) {
        const uint8_t* cr = (const uint8_t*)fm_surface_row32(c->color, y);
        for (int x = tr[0] * 4; x < tr[2] * 4; x += 64) FM_PREFETCH_W(cr + x);
        if (c->depth) {
            const uint8_t* dr = fm_surface_row8(c->depth, y);
            int            bpp = fm_format_bpp(c->depth->format);
            for (int x = tr[0] * bpp; x < tr[2] * bpp; x += 64) FM_PREFETCH_W(dr + x);
        }
    }
#endif
    for (uint32_t li = c->tl_start[tile]; li < c->tl_start[tile + 1]; li++) {
        uint32_t item = c->tl[li];
        if (item & FM3D_TL_CLEAR) {
            const fm3d_cmd* cmd = &c->cmds[item & ~FM3D_TL_CLEAR];
            int r[4] = { FM_MAX(tr[0], cmd->rect[0]), FM_MAX(tr[1], cmd->rect[1]), FM_MIN(tr[2], cmd->rect[2]),
                         FM_MIN(tr[3], cmd->rect[3]) };
            if (r[2] > r[0] && r[3] > r[1]) {
                if (c->msaa > 1 && c->ms_color)
                    fm3d_clear_rect_ms(c, cmd->type, r, cmd->color, cmd->depth);
                else
                    fm3d_clear_rect(c->color, c->depth, c->stencil, cmd->type, r, cmd->color, cmd->depth);
                if (hz && cmd->type == FM3D_CMD_CLEAR_DEPTH) {
                    uint32_t k = fm3d_zkey(c->depth->format, FM_CLAMP(cmd->depth, 0.0f, 1.0f));
                    if ((r[2] - r[0]) * (r[3] - r[1]) == area) {
                        hz->kmin = hz->kmax = k;
                        hz->written         = 0;
                        hz->valid           = 1;
                    } else if (hz->valid) {
                        hz->kmin = FM_MIN(hz->kmin, k);
                        hz->kmax = FM_MAX(hz->kmax, k);
                    }
                }
            }
            continue;
        }
        const fm3d_chunk*   ch = &c->chunks[item];
        const fm3d_drawrec* d  = &c->draws[ch->draw];
        int r[4] = { FM_MAX(tr[0], d->st->rect[0]), FM_MAX(tr[1], d->st->rect[1]), FM_MIN(tr[2], d->st->rect[2]),
                     FM_MIN(tr[3], d->st->rect[3]) };
        if (r[2] <= r[0] || r[3] <= r[1]) continue;
        int hiz = hz && d->st->depth == c->depth && fm3d_hiz_eligible(d->st);
        {
            int lx = tx - ch->btx, ly = ty - ch->bty;
            int lt = ly * ch->btw + lx;
            for (uint32_t j = ch->bin_start[lt]; j < ch->bin_start[lt + 1]; j++) {
                const fm3d_tri* t = ch->tris[ch->bin[j]];
                if (hiz) {
                    /* refresh the bounds once enough depth was written */
                    if (!hz->valid || hz->written * 2 >= area) fm3d_hiz_scan(c->depth, tr, hz);
                    if (fm3d_hiz_reject(t, r, hz)) {
                        w->stats.hiz_rejected++;
                        continue;
                    }
                }
                fm3d_raster_tri(t, r, &w->batch);
            }
        }
    }
    if (c->msaa > 1 && c->ms_color) fm3d_ms_resolve(c, tr); /* resolve while the tile is in cache */
}

void fm3d_flush(fm3d_ctx* c)
{
    if (!c) return;
    if (!c->deferred && c->msaa > 1 && c->ms_color && c->color && c->ms_w == c->color->width &&
        c->ms_h == c->color->height) {
        int r[4] = { 0, 0, c->color->width, c->color->height };
        fm3d_ms_resolve(c, r);
    }
    if (c->ncmd == 0) return;
    if (!c->color) {
        c->ncmd = c->ndraw = 0;
        fm3d_release_held(c);
        fm_arena_reset(&c->rec);
        return;
    }
    FM_PROF_BEGIN(zf, "3d.flush");
    fm_executor* ex = c->exec;
    int          nw = (ex && ex->workers > 1) ? ex->workers : 1;
    if (nw > c->nworkers) {
        fm3d_worker* n = (fm3d_worker*)realloc(c->workers, (size_t)nw * sizeof(fm3d_worker));
        if (n) {
            memset(n + c->nworkers, 0, (size_t)(nw - c->nworkers) * sizeof(fm3d_worker));
            c->workers  = n;
            c->nworkers = nw;
        } else {
            ex = NULL;
        }
    }
    for (int i = 0; i < c->nworkers; i++) {
        fm_arena_reset(&c->workers[i].arena);
        memset(&c->workers[i].stats, 0, sizeof(fm3d_stats));
    }
    fm_arena_reset(&c->frame);

    /* plan tasks */
    int nvt = 0, nch = 0;
    for (int i = 0; i < c->ndraw; i++) {
        fm3d_drawrec* d = &c->draws[i];
        nvt += (d->nv + FM3D_VBLOCK - 1) / FM3D_VBLOCK;
        nch += (d->ntri + FM3D_TCHUNK - 1) / FM3D_TCHUNK;
    }
    c->vtasks  = (fm3d_vtask*)fm_arena_alloc(&c->frame, (size_t)FM_MAX(nvt, 1) * sizeof(fm3d_vtask));
    c->chunks  = (fm3d_chunk*)fm_arena_alloc(&c->frame, (size_t)FM_MAX(nch, 1) * sizeof(fm3d_chunk));
    c->nvtasks = 0;
    c->nchunks = 0;
    int ok     = c->vtasks && c->chunks;
    for (int i = 0; ok && i < c->ndraw; i++) {
        fm3d_drawrec* d = &c->draws[i];
        d->vout         = (fm3d_vout*)fm_arena_alloc(&c->frame, (size_t)d->nv * sizeof(fm3d_vout));
        if (!d->vout) {
            ok = 0;
            break;
        }
        for (int v0 = 0; v0 < d->nv; v0 += FM3D_VBLOCK) {
            fm3d_vtask* t = &c->vtasks[c->nvtasks++];
            t->draw       = i;
            t->v0         = v0;
            t->n          = FM_MIN(FM3D_VBLOCK, d->nv - v0);
        }
        d->chunk0  = c->nchunks;
        d->nchunks = 0;
        for (int t0 = 0; t0 < d->ntri; t0 += FM3D_TCHUNK) {
            fm3d_chunk* ch = &c->chunks[c->nchunks++];
            memset(ch, 0, sizeof(*ch));
            ch->draw = i;
            ch->tri0 = t0;
            ch->ntri = FM_MIN(FM3D_TCHUNK, d->ntri - t0);
            d->nchunks++;
        }
    }
    if (ok) {
        c->tiles_x = (c->color->width + c->tile - 1) / c->tile;
        c->tiles_y = (c->color->height + c->tile - 1) / c->tile;
        c->hiz     = c->msaa > 1 ? NULL
                                   : (fm3d_hiz*)fm_arena_alloc(&c->frame, (size_t)(c->tiles_x * c->tiles_y) * sizeof(fm3d_hiz));
        if (c->hiz) memset(c->hiz, 0, (size_t)(c->tiles_x * c->tiles_y) * sizeof(fm3d_hiz));
        if (c->msaa > 1) fm3d_ms_ensure(c);

        FM_PROF_BEGIN(za, "3d.vertex");
        fm3d_run(ex, fm3d_phase_vertex, c, c->nvtasks);
        FM_PROF_END(za);
        FM_PROF_BEGIN(zb, "3d.setup_bin");
        fm3d_run(ex, fm3d_phase_setup, c, c->nchunks);
        ok = fm3d_build_tile_lists(c);
        FM_PROF_END(zb);
    }
    if (ok) {
        FM_PROF_BEGIN(zc, "3d.tiles");
        /* tile affinity: the same thread renders the same tiles every frame
         * (their color / depth stay in its caches; dynamic hand out moved
         * tiles across the 9950X3D's CCDs), stealing keeps the balance */
        fm__parallel_for_affine(ex, fm3d_phase_tile, c, c->tiles_x * c->tiles_y);
        FM_PROF_ITEMS(zc, c->tiles_x * c->tiles_y);
        FM_PROF_END(zc);
        for (int i = 0; i < c->nworkers; i++) {
            fm3d_worker* w = &c->workers[i];
            w->stats.fragments_in += w->batch.frag_in;
            w->stats.fragments_shaded += w->batch.frag_shaded;
            w->batch.frag_in = w->batch.frag_shaded = 0;
            fm3d_stats_add(&c->stats, &w->stats);
        }
    }

    c->ncmd  = 0;
    c->ndraw = 0;
    fm3d_release_held(c);
    fm_arena_reset(&c->rec);
#if FM_FEATURE_TNL
    c->lp_rec = NULL; /* the recorded copies are gone with the arena */
#endif
#if FM_FEATURE_SHADERS
    for (int b = 0; b < FM3D_MAX_UNIFORM_BLOCKS; b++) c->uni_rec[b] = NULL;
#endif
    FM_PROF_END(zf);
}
