/* fatmap - 3D internals: stage interfaces shared by immediate and tiled paths */
#ifndef FATMAP_FM3D_INTERNAL_H
#define FATMAP_FM3D_INTERNAL_H

#include <fatmap/fm3d.h>
#include "../core/fm_internal.h"
#include "../core/fm_arena.h"
#include <math.h>

#define FM3D_QCOLS    32               /* fragment batch width (16 quads) */
#define FM3D_QN       (2 * FM3D_QCOLS) /* pixels per batch: 2 rows */
#define FM3D_GUARD    32.0f            /* guard band in NDC units */
#define FM3D_MAX_CLIP 12               /* vertices of a clipped triangle */

/* fixed function varying layout */
enum { FM3D_VAR_U = 0, FM3D_VAR_V, FM3D_VAR_R, FM3D_VAR_G, FM3D_VAR_B, FM3D_VAR_A, FM3D_FIXED_NVAR };

struct fm3d_texture {
    int         refs;
    int         levels;
    fm_surface* level[16];
};

/* vertex stage output: clip space position + generic varyings */
typedef struct fm3d_vout {
    float pos[4];
    float var[FM3D_MAX_VARYINGS];
} fm3d_vout;

typedef struct fm3d_dstate fm3d_dstate;
typedef struct fm3d_batch  fm3d_batch;

/* stage function types (fixed function now, programmable later) */
typedef void (*fm3d_vs_fn)(const fm3d_dstate* st, const fm3d_vertex* in, int n, fm3d_vout* out);
typedef void (*fm3d_fs_fn)(const fm3d_dstate* st, fm3d_batch* b);

/* snapshot of everything a draw needs */
struct fm3d_dstate {
    fm_mat4         model, view, proj, mvp;
    fm3d_clip_depth clip_depth;
    int             vp[4];   /* viewport x, y, w, h */
    int             rect[4]; /* raster rect x0, y0, x1, y1 = viewport & scissor & target */
    fm3d_cull       cull;
    fm3d_winding    front;
    int             perspective;
    fm3d_compare    depth_func;
    int             depth_write;
    fm3d_texture*   tex;
    fm3d_sampler    sampler;
    fm3d_texenv     texenv;
    fm3d_compare    alpha_func;
    uint32_t        alpha_ref8;
    fm_blend_op     op;
    uint32_t        opacity8;
    int             nvar;
    fm3d_vs_fn      vs;
    fm3d_fs_fn      fs;
    fm_surface*     color;
    fm_surface*     depth;
};

enum { FM3D_TRI_FLAT = 1 };

/* set up triangle: edges in 28.4 fixed point, planes for z, 1/w, varyings */
typedef struct fm3d_tri {
    const fm3d_dstate* st;
    int64_t            A[3], B[3], K0[3]; /* E(px, py) = 16*A*px + K0 + B*(16*py + 8) >= 0 */
    int                minx, miny, maxx, maxy; /* inclusive pixel bounds, inside st->rect */
    float              x0f, y0f;               /* plane reference point (vertex 0) */
    float              z[3];                   /* c0, d/dx, d/dy */
    float              w[3];                   /* 1/w plane (perspective) */
    uint32_t           flat;                   /* premultiplied color if FM3D_TRI_FLAT */
    int                flags;
    int                nvar;
    float              var[]; /* 3 * nvar: c0, d/dx, d/dy (of var/w when perspective) */
} fm3d_tri;

/* SoA fragment batch: 2 rows x up to 32 columns, quads = column pairs.
 * index = row * FM3D_QCOLS + col. */
struct fm3d_batch {
    const fm3d_tri* tri;
    int             x, y, cols;
    uint32_t        need;    /* bit k: varying k is evaluated */
    int             uniform; /* set by the fragment stage: every pixel = color[0] */
    uint8_t         mask[FM3D_QN];
    float           z[FM3D_QN];
    float           w[FM3D_QN];
    float           var[FM3D_MAX_VARYINGS][FM3D_QN];
    uint32_t        color[FM3D_QN];
};

/* receives set up triangles from primitive processing */
typedef struct fm3d_sink {
    fm_arena*  arena; /* tri storage (NULL: use tmp) */
    void       (*emit)(struct fm3d_sink* s, fm3d_tri* t);
    void*      user;
    fm3d_stats stats;
    /* immediate mode storage for one triangle (8 byte aligned) */
    double tmp[(sizeof(fm3d_tri) + sizeof(float) * 3 * FM3D_MAX_VARYINGS) / sizeof(double) + 1];
} fm3d_sink;

/* fixed function stages */
void fm3d_vs_fixed(const fm3d_dstate* st, const fm3d_vertex* in, int n, fm3d_vout* out);
void fm3d_fs_fixed(const fm3d_dstate* st, fm3d_batch* b);

/* clip + cull + project + setup one triangle, emitting 0..n triangles */
void fm3d_process_tri(const fm3d_dstate* st, const fm3d_vout* a, const fm3d_vout* b, const fm3d_vout* c,
                      fm3d_sink* sink);
/* rasterize and shade t inside rect r (x0, y0, x1, y1); batch is scratch */
void fm3d_raster_tri(const fm3d_tri* t, const int r[4], fm3d_batch* batch);

#endif
