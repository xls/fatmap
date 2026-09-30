/*
 * fatmap - command lists: record now, execute later in parallel.
 *
 * Phase 1 (parallel over command chunks): run each command's geometry
 * callback into the worker's rasterizer, sort, copy the edges into the
 * worker's edge arena.
 * Phase 2 (parallel over strips): every strip walks the command list in
 * order and renders the commands that overlap it, clipped to its rows.
 * Each worker owns a rasterizer (accumulation buffers) and a pipeline.
 */
#include <fatmap/fm_exec.h>
#include "fm_internal.h"
#include "fm_raster_internal.h"

#define FM_CMD_CHUNK       8
#define FM_DEFAULT_STRIP   32
#define FM_ARENA_BLOCK     (64 * 1024)

/* ---- arena ------------------------------------------------------------------- */

typedef struct fm_arena_block {
    struct fm_arena_block* next;
    size_t                 used, cap;
} fm_arena_block;

typedef struct fm_arena {
    fm_arena_block* head; /* current block */
    fm_arena_block* spare; /* recycled blocks */
} fm_arena;

#define FM_ARENA_HDR ((sizeof(fm_arena_block) + 15) & ~(size_t)15)

static void* fm_arena_alloc(fm_arena* a, size_t size)
{
    size = (size + 15) & ~(size_t)15;
    if (!a->head || a->head->used + size > a->head->cap) {
        fm_arena_block* b = NULL;
        if (a->spare && a->spare->cap >= size) {
            b        = a->spare;
            a->spare = b->next;
        } else {
            size_t cap = size > FM_ARENA_BLOCK ? size : FM_ARENA_BLOCK;
            b          = (fm_arena_block*)malloc(FM_ARENA_HDR + cap);
            if (!b) return NULL;
            b->cap = cap;
        }
        b->used = 0;
        b->next = a->head;
        a->head = b;
    }
    void* p = (uint8_t*)a->head + FM_ARENA_HDR + a->head->used;
    a->head->used += size;
    return p;
}

/* move all blocks to the spare list (keeps memory for the next frame) */
static void fm_arena_reset(fm_arena* a)
{
    while (a->head) {
        fm_arena_block* b = a->head;
        a->head           = b->next;
        b->next           = a->spare;
        a->spare          = b;
    }
}

static void fm_arena_free(fm_arena* a)
{
    fm_arena_reset(a);
    while (a->spare) {
        fm_arena_block* b = a->spare;
        a->spare          = b->next;
        free(b);
    }
}

/* ---- list ---------------------------------------------------------------------- */

enum { FM_CMD_FILL, FM_CMD_RECT, FM_CMD_MASK };

typedef struct fm_cmd {
    int               type;
    int               evenodd;
    int               aa;
    int               clip[4];
    fm_geometry_fn    build;
    void*             data;
    fm_draw_state     st;
    fm_surface*       mask_out;
    const fm_surface* mask_in;
    /* phase 1 output (or the rect for FM_CMD_RECT) */
    const fm_redge* edges;
    int             nedges;
    int             bb[4];
    int             empty;
} fm_cmd;

typedef struct fm_release {
    fm_release_fn fn;
    void*         p;
} fm_release;

typedef struct fm_worker {
    fm_rasterizer* rast;
    fm_pipeline*   pipe;
    fm_arena       edges;
    fm_scratch     scratch;
} fm_worker;

struct fm_cmdlist {
    fm_surface* target;
    fm_cmd*     cmds;
    int         n, cap;
    fm_arena    arena;
    fm_release* rel;
    int         nrel, crel;
    fm_worker*  workers;
    int         nworkers;
    int         strip_h;
};

fm_cmdlist* fm_cmdlist_create(void)
{
    fm__init();
    fm_cmdlist* l = (fm_cmdlist*)calloc(1, sizeof(fm_cmdlist));
    if (l) l->strip_h = FM_DEFAULT_STRIP;
    return l;
}

static void fm_run_releases(fm_cmdlist* l)
{
    /* release in reverse order of registration */
    for (int i = l->nrel - 1; i >= 0; i--) l->rel[i].fn(l->rel[i].p);
    l->nrel = 0;
}

void fm_cmdlist_reset(fm_cmdlist* l, fm_surface* target)
{
    fm_run_releases(l);
    l->n = 0;
    fm_arena_reset(&l->arena);
    l->target = target;
}

void fm_cmdlist_destroy(fm_cmdlist* l)
{
    if (!l) return;
    fm_run_releases(l);
    for (int i = 0; i < l->nworkers; i++) {
        fm_worker* w = &l->workers[i];
        fm_rasterizer_destroy(w->rast);
        fm_pipeline_destroy(w->pipe);
        fm_arena_free(&w->edges);
        if (w->scratch.ptr && w->scratch.free_fn) w->scratch.free_fn(w->scratch.ptr);
    }
    free(l->workers);
    fm_arena_free(&l->arena);
    free(l->cmds);
    free(l->rel);
    free(l);
}

fm_surface* fm_cmdlist_target(const fm_cmdlist* l) { return l->target; }
int         fm_cmdlist_count(const fm_cmdlist* l) { return l->n; }
void*       fm_cmdlist_alloc(fm_cmdlist* l, size_t size) { return fm_arena_alloc(&l->arena, size); }

void fm_cmdlist_set_strip_height(fm_cmdlist* l, int rows)
{
    if (rows < FM_RASTER_BAND) rows = FM_RASTER_BAND;
    l->strip_h = (rows + FM_RASTER_BAND - 1) & ~(FM_RASTER_BAND - 1);
}

void fm_cmdlist_defer_release(fm_cmdlist* l, fm_release_fn fn, void* p)
{
    if (!fn) return;
    if (l->nrel == l->crel) {
        int         nc = l->crel ? l->crel * 2 : 64;
        fm_release* nr = (fm_release*)realloc(l->rel, (size_t)nc * sizeof(fm_release));
        if (!nr) {
            fn(p); /* cannot defer: release now (safe only if unused) */
            return;
        }
        l->rel  = nr;
        l->crel = nc;
    }
    l->rel[l->nrel].fn = fn;
    l->rel[l->nrel].p  = p;
    l->nrel++;
}

static fm_cmd* fm_cmd_push(fm_cmdlist* l)
{
    if (l->n == l->cap) {
        int     nc = l->cap ? l->cap * 2 : 256;
        fm_cmd* ns = (fm_cmd*)realloc(l->cmds, (size_t)nc * sizeof(fm_cmd));
        if (!ns) return NULL;
        l->cmds = ns;
        l->cap  = nc;
    }
    fm_cmd* c = &l->cmds[l->n++];
    memset(c, 0, sizeof(*c));
    return c;
}

void fm_cmdlist_fill(fm_cmdlist* l, fm_geometry_fn build, void* data, fm_fill_rule rule, fm_aa_mode aa,
                     const int clip[4], const fm_draw_state* st)
{
    if (!build || !st) return;
    fm_cmd* c = fm_cmd_push(l);
    if (!c) return;
    c->type    = FM_CMD_FILL;
    c->build   = build;
    c->data    = data;
    c->evenodd = rule == FM_FILL_EVENODD;
    c->aa      = aa;
    memcpy(c->clip, clip, sizeof(c->clip));
    c->st = *st;
}

void fm_cmdlist_fill_rect(fm_cmdlist* l, int x0, int y0, int x1, int y1, const fm_draw_state* st)
{
    if (!st) return;
    fm_cmd* c = fm_cmd_push(l);
    if (!c) return;
    c->type    = FM_CMD_RECT;
    c->clip[0] = x0;
    c->clip[1] = y0;
    c->clip[2] = x1;
    c->clip[3] = y1;
    c->st      = *st;
}

void fm_cmdlist_mask(fm_cmdlist* l, fm_geometry_fn build, void* data, fm_fill_rule rule, fm_aa_mode aa,
                     const int clip[4], fm_surface* mask_out, const fm_surface* mask_in)
{
    if (!build || !mask_out || mask_out->format != FM_FORMAT_A8) return;
    fm_cmd* c = fm_cmd_push(l);
    if (!c) return;
    c->type    = FM_CMD_MASK;
    c->build   = build;
    c->data    = data;
    c->evenodd = rule == FM_FILL_EVENODD;
    c->aa      = aa;
    memcpy(c->clip, clip, sizeof(c->clip));
    c->mask_out = mask_out;
    c->mask_in  = mask_in;
}

/* ---- execution --------------------------------------------------------------------- */

static int fm_workers_ensure(fm_cmdlist* l, int n)
{
    if (n <= l->nworkers) return 1;
    fm_worker* nw = (fm_worker*)realloc(l->workers, (size_t)n * sizeof(fm_worker));
    if (!nw) return 0;
    l->workers = nw;
    for (int i = l->nworkers; i < n; i++) {
        fm_worker* w = &l->workers[i];
        memset(w, 0, sizeof(*w));
        w->rast = fm_rasterizer_create();
        w->pipe = fm_pipeline_create();
        if (!w->rast || !w->pipe) {
            fm_rasterizer_destroy(w->rast);
            fm_pipeline_destroy(w->pipe);
            l->nworkers = i;
            return 0;
        }
    }
    l->nworkers = n;
    return 1;
}

static void fm_phase_geometry(void* arg, int index, int worker)
{
    fm_cmdlist* l  = (fm_cmdlist*)arg;
    fm_worker*  w  = &l->workers[worker];
    int         c0 = index * FM_CMD_CHUNK, c1 = FM_MIN(l->n, c0 + FM_CMD_CHUNK);
    int         tw = l->target->width, th = l->target->height;
    for (int ci = c0; ci < c1; ci++) {
        fm_cmd* c = &l->cmds[ci];
        c->empty  = 1;
        if (c->type == FM_CMD_RECT) {
            c->bb[0] = FM_MAX(c->clip[0], 0);
            c->bb[1] = FM_MAX(c->clip[1], 0);
            c->bb[2] = FM_MIN(c->clip[2], tw);
            c->bb[3] = FM_MIN(c->clip[3], th);
            c->empty = c->bb[2] <= c->bb[0] || c->bb[3] <= c->bb[1];
            continue;
        }
        int cx0 = FM_MAX(c->clip[0], 0), cy0 = FM_MAX(c->clip[1], 0);
        int cx1 = FM_MIN(c->clip[2], tw), cy1 = FM_MIN(c->clip[3], th);
        if (cx1 <= cx0 || cy1 <= cy0) continue;
        fm_rasterizer_reset(w->rast, cx0, cy0, cx1, cy1);
        c->build(c->data, w->rast, &w->scratch);
        if (!fm__raster_prepare(w->rast, c->bb)) continue;
        int             ne;
        const fm_redge* e    = fm__raster_edges(w->rast, &ne);
        fm_redge*       copy = (fm_redge*)fm_arena_alloc(&w->edges, (size_t)ne * sizeof(fm_redge));
        if (!copy) continue;
        memcpy(copy, e, (size_t)ne * sizeof(fm_redge));
        c->edges  = copy;
        c->nedges = ne;
        c->empty  = 0;
    }
}

static void fm_phase_raster(void* arg, int strip, int worker)
{
    fm_cmdlist* l  = (fm_cmdlist*)arg;
    fm_worker*  w  = &l->workers[worker];
    int         y0 = strip * l->strip_h;
    int         y1 = FM_MIN(l->target->height, y0 + l->strip_h);
    fm_pipeline_set_target(w->pipe, l->target);
    for (int ci = 0; ci < l->n; ci++) {
        const fm_cmd* c = &l->cmds[ci];
        if (c->empty || c->bb[3] <= y0 || c->bb[1] >= y1) continue;
        switch (c->type) {
        case FM_CMD_RECT:
            fm_pipeline_set_state(w->pipe, &c->st);
            fm_pipeline_fill_rect(w->pipe, c->bb[0], FM_MAX(c->bb[1], y0), c->bb[2], FM_MIN(c->bb[3], y1));
            break;
        case FM_CMD_FILL:
            fm_pipeline_set_state(w->pipe, &c->st);
            fm__raster_render(w->rast, c->edges, c->nedges, c->bb, y0, y1, c->evenodd, c->aa, fm_pipeline_span,
                              w->pipe);
            break;
        case FM_CMD_MASK: {
            fm_mask_sink sink;
            sink.out = c->mask_out;
            sink.in  = c->mask_in;
            fm__raster_render(w->rast, c->edges, c->nedges, c->bb, y0, y1, c->evenodd, c->aa, fm_mask_span, &sink);
            break;
        }
        }
    }
}

static void fm_run(fm_executor* ex, fm_task_fn fn, void* arg, int count)
{
    if (!ex || ex->workers <= 1 || count <= 1) {
        for (int i = 0; i < count; i++) fn(arg, i, 0);
        return;
    }
    ex->parallel_for(ex, fn, arg, count);
}

void fm_cmdlist_execute(fm_cmdlist* l, fm_executor* ex)
{
    if (!l || l->n == 0 || !l->target || l->target->format != FM_FORMAT_ARGB32) return;
    int nw = (ex && ex->workers > 1) ? ex->workers : 1;
    if (!fm_workers_ensure(l, nw)) {
        ex = NULL;
        if (!fm_workers_ensure(l, 1)) return;
    }
    for (int i = 0; i < l->nworkers; i++) fm_arena_reset(&l->workers[i].edges);

    FM_PROF_BEGIN(zg, "cmd.geometry");
    fm_run(ex, fm_phase_geometry, l, (l->n + FM_CMD_CHUNK - 1) / FM_CMD_CHUNK);
    FM_PROF_ITEMS(zg, l->n);
    FM_PROF_END(zg);

    FM_PROF_BEGIN(zr, "cmd.raster");
    int strips = (l->target->height + l->strip_h - 1) / l->strip_h;
    fm_run(ex, fm_phase_raster, l, strips);
    FM_PROF_ITEMS(zr, strips);
    FM_PROF_END(zr);
}
