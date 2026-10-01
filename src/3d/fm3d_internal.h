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
#define FM3D_MAX_SAMPLES 8

/* sample offsets from the pixel center in 1/16 pixel (D3D standard patterns) */
extern const int8_t fm3d_samples4[4][2];
extern const int8_t fm3d_samples8[8][2];

/* fixed function varying layout */
enum { FM3D_VAR_U = 0, FM3D_VAR_V, FM3D_VAR_R, FM3D_VAR_G, FM3D_VAR_B, FM3D_VAR_A, FM3D_FIXED_NVAR };
enum { FM3D_VAR_U2 = FM3D_FIXED_NVAR, FM3D_VAR_V2, FM3D_FIXED_NVAR_MT }; /* fm3d_vertex_mt */

struct fm3d_texture {
    int               refs;
    int               levels;
    fm_surface*       level[16]; /* layer 0 */
    fm3d_texture_kind kind;
    int               nlayers;
    int               straight; /* texels hold straight alpha */
    fm_surface**      lv;       /* layers 1 .. nlayers - 1: lv[(layer - 1) * 16 + level] */
    uint32_t*         pack;     /* layer 0's levels in one block (level[] wraps it): gathers reach any level */
    int32_t           poff[16]; /* texel offset of each level in pack (rows of width texels) */
    int32_t           pw[16], ph[16]; /* level sizes (the SIMD sampler's tables) */
};

/* log2 for mip level selection (exponent + quadratic on the mantissa) */
static inline float fm3d_log2_fast(float x)
{
    union {
        float    f;
        uint32_t i;
    } u;
    u.f     = x;
    float e = (float)(int)((u.i >> 23) & 255) - 127.0f;
    u.i     = (u.i & 0x007fffffu) | 0x3f800000u; /* mantissa in [1, 2) */
    float m = u.f - 1.0f;
    return e + m * (1.3465f - 0.3465f * m);
}

/* the AVX2 sampler of one 16 lane quad group (fm3d_sample_avx2.c); 0 when it does not cover the texture */
int fm3d_sample16_avx2(const fm3d_texture* t, const fm3d_sampler* s, const float* U, const float* V, float* r, float* g, float* b,
                       float* a);
int fm3d_sample16_avx512(const fm3d_texture* t, const fm3d_sampler* s, const float* U, const float* V, float* r, float* g, float* b,
                         float* a); /* the same at AVX-512 (fm3d_sample_avx512.c) */
static inline const fm_surface* fm3d_tex_level(const fm3d_texture* t, int layer, int level)
{
    return layer <= 0 ? t->level[level] : t->lv[(layer - 1) * 16 + level];
}

/* vertex stage output: clip space position + generic varyings */
typedef struct fm3d_vout {
    float pos[4];
    float var[FM3D_MAX_VARYINGS];
} fm3d_vout;

typedef struct fm3d_dstate fm3d_dstate;
typedef struct fm3d_batch  fm3d_batch;

/* stage function types (fixed function now, programmable later) */
/* in: n vertices of st->vstride bytes (fm3d_vertex for the fixed stage) */
typedef void (*fm3d_vs_fn)(const fm3d_dstate* st, const void* in, int n, int first, fm3d_vout* out);
typedef void (*fm3d_fs_fn)(const fm3d_dstate* st, fm3d_batch* b);

/* snapshot of everything a draw needs */
struct fm3d_dstate {
    fm_mat4         model, view, proj, mvp;
    fm3d_clip_depth clip_depth;
    fm3d_origin     origin;
    fm3d_primitive  prim;
    float           line_width, point_size;
    int             psize_var, pcoord_var; /* program varyings (-1: none) */
    int             vp[4];   /* viewport x, y, w, h */
    int             rect[4]; /* raster rect x0, y0, x1, y1 = viewport & scissor & target */
    fm3d_cull       cull;
    fm3d_winding    front;
    int             perspective;
    fm3d_compare    depth_func;
    int             depth_write;
    float           depth_bias_factor, depth_bias_units;
    float           depth_near, depth_far; /* depth range */
    int             depth_clamp;
    int             stencil_on;
    struct fm3d_stencil_face {
        fm3d_compare    func;
        uint8_t         ref, read_mask, write_mask;
        fm3d_stencil_op sfail, dpfail, dppass;
    } stencil[2]; /* 0 front, 1 back */
    int             color_write;
    uint32_t        color_mask; /* ARGB32 bits written (0xFFFFFFFF: all) */
    fm3d_texture*   tex;
    fm3d_sampler    sampler;
    fm3d_texenv     texenv;
    fm3d_texture*   tex1; /* fixed second texture stage (NULL: off) */
    fm3d_sampler    sampler1;
    fm3d_texenv     texenv1;
    fm3d_fog        fog;
    uint32_t        fog_color; /* straight or premultiplied as the target */
    float           fog_start, fog_end, fog_density;
    fm3d_compare    alpha_func;
    uint32_t        alpha_ref8;
    fm_blend_op     op;
    int             straight; /* fm3d_set_blend_state: straight colors, GL blending with gb */
    fm_glblend      gb;
    uint32_t        opacity8;
    int             nvar;
#if FM_FEATURE_TNL
    /* lighting (NULL = off): eye space parameters, model * view and its
     * normal matrix (inverse transpose, column major 3x3) */
    const fm_light_params* lp;
    fm_mat4                mv;
    float                  nrm[9];
#endif
#if FM_FEATURE_SHADERS
    fm3d_vertex_shader   user_vs;
    fm3d_fragment_shader user_fs;
    int                  fs_discards;
    int                  fs_depth;  /* the fragment stage writes depth */
    int                  fs_interp; /* the fragment stage interpolates its varyings (fm3d_program.interpolates) */
    int                  fs_packs;  /* the fragment stage packs straight colors (fm3d_program.packs_color) */
    const void*          uniforms;     /* = blocks[0] */
    size_t               uniform_size;
    const void*          blocks[FM3D_MAX_UNIFORM_BLOCKS];
    size_t               block_sizes[FM3D_MAX_UNIFORM_BLOCKS];
    void*                user;
    int                  base_vertex, instance; /* fm3d_set_draw_ids */
    fm3d_texture*        units[FM3D_MAX_TEXTURE_UNITS];  /* [0] mirrors tex */
    fm3d_sampler         usamp[FM3D_MAX_TEXTURE_UNITS];
#endif
    int             vstride; /* bytes per input vertex */
    fm3d_vs_fn      vs;
    fm3d_fs_fn      fs;
    fm_surface*     color;
    fm_surface*     depth;
    fm_surface*     stencil_buf; /* A8, or the depth surface itself when it is D24S8 */
    /* MSAA: samples per pixel (1 = off) and the per sample buffers */
    int             msaa;
    uint32_t*       ms_color;   /* (y * ms_w + x) * msaa + s */
    float*          ms_depth;
    uint8_t*        ms_stencil;
    int             ms_w;
};

/* texture units a draw retains (1 without programmable stages) */
#if FM_FEATURE_SHADERS
#  define FM3D_NUNITS FM3D_MAX_TEXTURE_UNITS
#else
#  define FM3D_NUNITS 1
#endif

enum { FM3D_TRI_FLAT = 1, FM3D_TRI_BACK = 2, FM3D_TRI_ZCLAMP = 4 };

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

/* hierarchical z: conservative bounds of the depth keys stored in a tile */
typedef struct fm3d_hiz {
    uint32_t kmin, kmax;
    int      written; /* depth writes since the bounds were last recomputed */
    int      valid;
} fm3d_hiz;

/* depth keys: one ordered uint32 domain for every depth format */
static inline uint32_t fm3d_fkey(float f)
{
    union {
        float    f;
        uint32_t u;
    } v;
    v.f = f;
    return v.u == 0x80000000u ? 0u : v.u; /* -0 -> 0; non negative floats order like their bits */
}
static inline uint32_t fm3d_zkey(fm_format fmt, float z) /* z in [0, 1] */
{
    switch (fmt) {
    case FM_FORMAT_D16: return (uint32_t)(z * 65535.0f + 0.5f);
    case FM_FORMAT_D24S8: return (uint32_t)((double)z * 16777215.0 + 0.5);
    default: return fm3d_fkey(z);
    }
}

/* SoA fragment batch: 2 rows x up to 32 columns, quads = column pairs.
 * index = row * FM3D_QCOLS + col. */
struct fm3d_batch {
    const fm3d_tri* tri;
    fm3d_hiz*       hiz; /* tile depth bounds to keep up to date (tiled mode) */
    int             x, y, cols;
    uint64_t        need;    /* bit k: varying k is evaluated */
    int             uniform; /* set by the fragment stage: every pixel = color[0] */
    int             full;    /* every mask byte of both rows is 255 (cols wide): no mask work */
    uint64_t        frag_in, frag_shaded; /* fragment counters (folded into fm3d_stats) */
    uint8_t         mask[FM3D_QN];
    uint8_t         smask[FM3D_QN]; /* MSAA: covered samples per pixel */
    float           z[FM3D_QN];
    float           w[FM3D_QN];
    float           var[FM3D_MAX_VARYINGS][FM3D_QN];
    uint32_t        color[FM3D_QN];
    float           dxv[FM3D_QCOLS], dyr[2]; /* pixel center offsets from the triangle's origin (last: the arrays above keep their alignment) */
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
void fm3d_vs_fixed(const fm3d_dstate* st, const void* in, int n, int first, fm3d_vout* out);
#if FM_FEATURE_SHADERS
void fm3d_vs_program(const fm3d_dstate* st, const void* in, int n, int first, fm3d_vout* out);
void fm3d_fs_program(const fm3d_dstate* st, fm3d_batch* b);
#endif
void fm3d_fs_fixed(const fm3d_dstate* st, fm3d_batch* b);

/* clip + cull + project + setup one triangle, emitting 0..n triangles */
void fm3d_process_tri(const fm3d_dstate* st, const fm3d_vout* a, const fm3d_vout* b, const fm3d_vout* c,
                      fm3d_sink* sink);
/* primitive i of an index list (NULL: sequential) as st->prim says */
void fm3d_process_prim(const fm3d_dstate* st, const fm3d_vout* vb, const uint32_t* idx, int i, fm3d_sink* sink);
/* rasterize and shade t inside rect r (x0, y0, x1, y1); batch is scratch */
void fm3d_raster_tri(const fm3d_tri* t, const int r[4], fm3d_batch* batch);

#endif
