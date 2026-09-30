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
    const fm3d_vertex* v;
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
    fm3d_dstate   st; /* current state (rect / mvp resolved per draw) */
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
    fm3d_worker*  workers;
    int           nworkers;
    /* flush data */
    fm3d_chunk*   chunks;
    int           nchunks;
    fm3d_vtask*   vtasks;
    int           nvtasks;
    int           tiles_x, tiles_y;
    fm3d_hiz*     hiz; /* per tile depth bounds (flush scratch) */
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
    s->fs                                 = fm3d_fs_fixed;
    c->tile                               = FM3D_TILE_DEF;
    return c;
}

static void fm3d_release_held(fm3d_ctx* c)
{
    for (int i = 0; i < c->nheld; i++) fm3d_texture_release(c->held[i]);
    c->nheld = 0;
}

void fm3d_destroy(fm3d_ctx* c)
{
    if (!c) return;
    fm3d_flush(c);
    fm3d_texture_release(c->st.tex);
    fm3d_release_held(c);
    for (int i = 0; i < c->nworkers; i++) fm_arena_free(&c->workers[i].arena);
    free(c->workers);
    fm_arena_free(&c->rec);
    fm_arena_free(&c->frame);
    free(c->cmds);
    free(c->draws);
    free(c->held);
    free(c->vbuf);
    free(c);
}

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
void fm3d_set_texenv(fm3d_ctx* c, fm3d_texenv env) { c->st.texenv = env; }
void fm3d_set_alpha_test(fm3d_ctx* c, fm3d_compare f, float ref)
{
    c->st.alpha_func = f;
    c->st.alpha_ref8 = (uint32_t)(FM_CLAMP(ref, 0.0f, 1.0f) * 255.0f + 0.5f);
}
void fm3d_set_blend(fm3d_ctx* c, fm_blend_op op)
{
    if (op >= 0 && op < FM_OP_COUNT) c->st.op = op;
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

fm3d_stats fm3d_get_stats(fm3d_ctx* c) { return c->stats; }
void       fm3d_reset_stats(fm3d_ctx* c) { memset(&c->stats, 0, sizeof(c->stats)); }

static void fm3d_stats_add(fm3d_stats* d, const fm3d_stats* s)
{
    d->triangles_in += s->triangles_in;
    d->triangles_clipped += s->triangles_clipped;
    d->triangles_culled += s->triangles_culled;
    d->triangles_drawn += s->triangles_drawn;
    d->hiz_rejected += s->hiz_rejected;
}

/* resolve derived state for a draw / clear */
static int fm3d_resolve(fm3d_ctx* c, fm3d_dstate* s)
{
    if (!c->color) return 0;
    *s             = c->st;
    s->color       = c->color;
    s->depth       = c->depth;
    s->stencil_buf = c->stencil ? c->stencil : ((c->depth && c->depth->format == FM_FORMAT_D24S8) ? c->depth : NULL);
    if (!c->vp_set) {
        s->vp[0] = 0;
        s->vp[1] = 0;
        s->vp[2] = c->color->width;
        s->vp[3] = c->color->height;
    }
    s->mvp     = fm_mat4_mul(s->proj, fm_mat4_mul(s->view, s->model));
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
    fm3d_clear_rect(c->color, c->depth, c->stencil, type, r, col, d);
}

void fm3d_clear_color(fm3d_ctx* c, fm_color col) { fm3d_clear_impl(c, FM3D_CMD_CLEAR_COLOR, fm_premultiply(col), 0); }
void fm3d_clear_depth(fm3d_ctx* c, float d) { fm3d_clear_impl(c, FM3D_CMD_CLEAR_DEPTH, 0, d); }
void fm3d_clear_stencil(fm3d_ctx* c, uint8_t v) { fm3d_clear_impl(c, FM3D_CMD_CLEAR_STENCIL, v, 0); }

/* ---- immediate draws -------------------------------------------------------------------- */

static void fm3d_emit_now(fm3d_sink* s, fm3d_tri* t)
{
    fm3d_ctx* c = (fm3d_ctx*)s->user;
    fm3d_raster_tri(t, t->st->rect, &c->batch);
}

static void fm3d_draw_impl(fm3d_ctx* c, const fm3d_vertex* v, int nv, const uint32_t* idx, int count)
{
    if (!v || count < 3 || nv <= 0) return;
    int         ntri = count / 3;
    fm3d_dstate s;
    if (!fm3d_resolve(c, &s)) return;
    if (idx)
        for (int i = 0; i < ntri * 3; i++)
            if (idx[i] >= (uint32_t)nv) return; /* reject out of range indices */

    if (c->deferred) {
        FM_PROF_BEGIN(z, "3d.record");
        fm3d_dstate*  st = (fm3d_dstate*)fm_arena_alloc(&c->rec, sizeof(fm3d_dstate));
        fm3d_vertex*  vc = (fm3d_vertex*)fm_arena_alloc(&c->rec, (size_t)nv * sizeof(fm3d_vertex));
        uint32_t*     ic = idx ? (uint32_t*)fm_arena_alloc(&c->rec, (size_t)ntri * 3 * sizeof(uint32_t)) : NULL;
        if (!st || !vc || (idx && !ic)) {
            FM_PROF_END(z);
            return;
        }
        *st = s;
        memcpy(vc, v, (size_t)nv * sizeof(fm3d_vertex));
        if (idx) memcpy(ic, idx, (size_t)ntri * 3 * sizeof(uint32_t));
        if (s.tex) {
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
            c->held[c->nheld++] = fm3d_texture_retain(s.tex);
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
    s.vs(&s, v, nv, c->vbuf);
    FM_PROF_END(zv);

    FM_PROF_BEGIN(zr, "3d.draw");
    fm3d_sink sink;
    memset(&sink, 0, sizeof(sink));
    c->batch.hiz = NULL;
    sink.emit = fm3d_emit_now;
    sink.user = c;
    for (int i = 0; i < ntri; i++) {
        const fm3d_vout* a = &c->vbuf[idx ? idx[3 * i] : (uint32_t)(3 * i)];
        const fm3d_vout* b = &c->vbuf[idx ? idx[3 * i + 1] : (uint32_t)(3 * i + 1)];
        const fm3d_vout* e = &c->vbuf[idx ? idx[3 * i + 2] : (uint32_t)(3 * i + 2)];
        fm3d_process_tri(&s, a, b, e, &sink);
    }
    fm3d_stats_add(&c->stats, &sink.stats);
    FM_PROF_ITEMS(zr, ntri);
    FM_PROF_END(zr);
}

void fm3d_draw(fm3d_ctx* c, const fm3d_vertex* v, int count) { fm3d_draw_impl(c, v, count, NULL, count); }

void fm3d_draw_indexed(fm3d_ctx* c, const fm3d_vertex* v, int vertex_count, const uint32_t* indices, int index_count)
{
    if (indices) fm3d_draw_impl(c, v, vertex_count, indices, index_count);
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
    d->st->vs(d->st, d->v + t->v0, t->n, d->vout + t->v0);
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
    for (int i = ch->tri0; i < ch->tri0 + ch->ntri; i++) {
        const fm3d_vout* a = &d->vout[d->idx ? d->idx[3 * i] : (uint32_t)(3 * i)];
        const fm3d_vout* b = &d->vout[d->idx ? d->idx[3 * i + 1] : (uint32_t)(3 * i + 1)];
        const fm3d_vout* e = &d->vout[d->idx ? d->idx[3 * i + 2] : (uint32_t)(3 * i + 2)];
        fm3d_process_tri(d->st, a, b, e, &bs.base);
    }
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
    for (int y = r[1]; y < r[3]; y++) {
        const uint8_t* row = fm_surface_row8(D, y);
        for (int x = r[0]; x < r[2]; x++) {
            uint32_t k;
            switch (D->format) {
            case FM_FORMAT_D16: k = ((const uint16_t*)row)[x]; break;
            case FM_FORMAT_D24S8: k = ((const uint32_t*)row)[x] & 0xffffffu; break;
            default: k = fm3d_fkey(((const float*)row)[x]); break;
            }
            kmin = FM_MIN(kmin, k);
            kmax = FM_MAX(kmax, k);
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
    for (int i = 0; i < c->ncmd; i++) {
        const fm3d_cmd* cmd = &c->cmds[i];
        if (cmd->type != FM3D_CMD_DRAW) {
            int r[4] = { FM_MAX(tr[0], cmd->rect[0]), FM_MAX(tr[1], cmd->rect[1]), FM_MIN(tr[2], cmd->rect[2]),
                         FM_MIN(tr[3], cmd->rect[3]) };
            if (r[2] > r[0] && r[3] > r[1]) {
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
        const fm3d_drawrec* d = &c->draws[cmd->draw];
        int r[4] = { FM_MAX(tr[0], d->st->rect[0]), FM_MAX(tr[1], d->st->rect[1]), FM_MIN(tr[2], d->st->rect[2]),
                     FM_MIN(tr[3], d->st->rect[3]) };
        if (r[2] <= r[0] || r[3] <= r[1]) continue;
        int hiz = hz && d->st->depth == c->depth && fm3d_hiz_eligible(d->st);
        for (int k = d->chunk0; k < d->chunk0 + d->nchunks; k++) {
            const fm3d_chunk* ch = &c->chunks[k];
            if (!ch->bin_start) continue;
            int lx = tx - ch->btx, ly = ty - ch->bty;
            if ((unsigned)lx >= (unsigned)ch->btw || (unsigned)ly >= (unsigned)ch->bth) continue;
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
}

void fm3d_flush(fm3d_ctx* c)
{
    if (!c || c->ncmd == 0) return;
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
        c->hiz     = (fm3d_hiz*)fm_arena_alloc(&c->frame, (size_t)(c->tiles_x * c->tiles_y) * sizeof(fm3d_hiz));
        if (c->hiz) memset(c->hiz, 0, (size_t)(c->tiles_x * c->tiles_y) * sizeof(fm3d_hiz));

        FM_PROF_BEGIN(za, "3d.vertex");
        fm3d_run(ex, fm3d_phase_vertex, c, c->nvtasks);
        FM_PROF_END(za);
        FM_PROF_BEGIN(zb, "3d.setup_bin");
        fm3d_run(ex, fm3d_phase_setup, c, c->nchunks);
        FM_PROF_END(zb);
        FM_PROF_BEGIN(zc, "3d.tiles");
        fm3d_run(ex, fm3d_phase_tile, c, c->tiles_x * c->tiles_y);
        FM_PROF_ITEMS(zc, c->tiles_x * c->tiles_y);
        FM_PROF_END(zc);
        for (int i = 0; i < c->nworkers; i++) fm3d_stats_add(&c->stats, &c->workers[i].stats);
    }

    c->ncmd  = 0;
    c->ndraw = 0;
    fm3d_release_held(c);
    fm_arena_reset(&c->rec);
    FM_PROF_END(zf);
}
