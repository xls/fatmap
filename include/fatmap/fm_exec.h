/*
 * fatmap - executors and command lists (multithreaded rendering).
 *
 * A command list records draws (geometry + a snapshot of the draw state)
 * and executes them in two parallel phases:
 *
 *   1. geometry  - every command builds its edge list (flatten, stroke,
 *                  clip, sort) independently: parallel over commands
 *   2. raster    - the target is split into horizontal strips; each strip
 *                  replays all commands that touch it, in order: parallel
 *                  over strips, no locks, exact painter's order per pixel
 *
 * Output is bit-identical for any thread count and to immediate rendering.
 *
 * Threads are optional. Build with -Dthreads=disabled (FM_NO_THREADS) for
 * platforms without a thread library: fm_executor_create() then returns a
 * serial executor and everything runs on the calling thread. Platforms with
 * their own task system can supply an fm_executor with a custom
 * parallel_for instead of using the built-in pool.
 */
#ifndef FATMAP_FM_EXEC_H
#define FATMAP_FM_EXEC_H

#include "fm_core.h"
#include "fm_raster.h"
#include "fm_pipe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- executors ------------------------------------------------------------ */

typedef void (*fm_task_fn)(void* arg, int index, int worker);

typedef struct fm_executor fm_executor;
struct fm_executor {
    /* Call fn(arg, i, worker) for every i in [0, count) and return when all
     * calls finished. worker must be in [0, workers) and no two concurrent
     * calls may share a worker id. */
    void (*parallel_for)(fm_executor* ex, fm_task_fn fn, void* arg, int count);
    int   workers; /* max concurrency (>= 1) */
    void* user;
};

FM_API int fm_threads_supported(void); /* 0 when built without thread support */
FM_API int fm_cpu_count(void);
/* Built-in thread pool with `threads` workers (<= 0: one per CPU). The
 * calling thread participates, so threads - 1 OS threads are started.
 * Without thread support this returns a serial executor. */
FM_API fm_executor* fm_executor_create(int threads);
FM_API void         fm_executor_destroy(fm_executor* ex); /* only for fm_executor_create() results */

/* ---- command lists ---------------------------------------------------------- */

/* Per worker scratch owned by the command list (e.g. flattening buffers).
 * Set ptr and free_fn on first use; freed with the list. */
typedef struct fm_scratch {
    void* ptr;
    void (*free_fn)(void* ptr);
} fm_scratch;

/* Phase 1 callback: add the command's lines to `out` (its clip rect is set). */
typedef void (*fm_geometry_fn)(void* data, fm_rasterizer* out, fm_scratch* scratch);
typedef void (*fm_release_fn)(void* p);

typedef struct fm_cmdlist fm_cmdlist;

FM_API fm_cmdlist* fm_cmdlist_create(void);
FM_API void        fm_cmdlist_destroy(fm_cmdlist* l);
/* Drops all commands, runs deferred releases, recycles memory. */
FM_API void        fm_cmdlist_reset(fm_cmdlist* l, fm_surface* target);
FM_API fm_surface* fm_cmdlist_target(const fm_cmdlist* l);
FM_API int         fm_cmdlist_count(const fm_cmdlist* l);
/* Record-time memory (16 byte aligned), valid until the next reset. */
FM_API void* fm_cmdlist_alloc(fm_cmdlist* l, size_t size);
/* fn(p) runs at the next reset / destroy (release retained objects). */
FM_API void  fm_cmdlist_defer_release(fm_cmdlist* l, fm_release_fn fn, void* p);
/* Rows per strip, rounded up to a multiple of 16 (default 32). */
FM_API void  fm_cmdlist_set_strip_height(fm_cmdlist* l, int rows);

/* clip: integer device clip rect {x0, y0, x1, y1}. st is copied; pointers
 * inside must stay valid until the list is reset. */
FM_API void fm_cmdlist_fill(fm_cmdlist* l, fm_geometry_fn build, void* data, fm_fill_rule rule, fm_aa_mode aa,
                            const int clip[4], const fm_draw_state* st);
FM_API void fm_cmdlist_fill_rect(fm_cmdlist* l, int x0, int y0, int x1, int y1, const fm_draw_state* st);
/* Coverage into an A8 mask: mask_out = coverage * mask_in (mask_in may be NULL). */
FM_API void fm_cmdlist_mask(fm_cmdlist* l, fm_geometry_fn build, void* data, fm_fill_rule rule, fm_aa_mode aa,
                            const int clip[4], fm_surface* mask_out, const fm_surface* mask_in);

/* Execute all commands (ex NULL = serial on the calling thread). The list
 * can be executed again; call fm_cmdlist_reset() to start a new one. */
FM_API void fm_cmdlist_execute(fm_cmdlist* l, fm_executor* ex);

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_EXEC_H */
