/*
 * fatmap - 3D renderer (fixed function today, designed for T&L and
 * programmable shaders later).
 *
 * Pipeline (each stage is replaceable internally; the fixed function
 * versions are the defaults):
 *
 *   vertex fetch -> vertex stage (transform; T&L later; SPIR-V later)
 *     -> clip (homogeneous, near/far + guard band) -> cull -> viewport
 *     -> triangle setup (fatmap style constant gradients: plane equations
 *        for z, 1/w and every varying/w, 28.4 fixed point edges, top-left)
 *     -> raster in 2x2 quads (row pairs), SoA fragment batches
 *     -> early depth -> fragment stage (texenv: texture sampling with mip
 *        LOD from quad derivatives + combine op) -> alpha test / late depth
 *     -> output merger (depth write, any fm_blend_op) -> back buffer
 *
 * Stages exchange generic float varyings (up to FM3D_MAX_VARYINGS), so a
 * programmable vertex/fragment stage can later replace the fixed ones.
 *
 * Deferred mode records draws; fm3d_flush() then runs vertex processing and
 * setup in parallel, bins triangles into screen tiles and rasterizes tiles
 * in parallel on an fm_executor. Output is bit-identical to immediate mode.
 *
 * Conventions follow OpenGL / glm: right handed, column major matrices,
 * counter clockwise front faces, clip depth -1..1 (FM3D_DEPTH_ZERO_ONE for
 * Direct3D / Vulkan style projections), window depth 0..1, y down in the
 * render target, pixel centers at +0.5.
 */
#ifndef FATMAP_FM3D_H
#define FATMAP_FM3D_H

#include "fm_core.h"
#include "fm_pipe.h"
#include "fm_exec.h"
#include "fm_math.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FM3D_MAX_VARYINGS 16

/* Fixed function vertex. The normal is carried for T&L (lighting comes
 * later); texcoords are normalized (0..1 spans the texture). */
typedef struct fm3d_vertex {
    float    x, y, z;
    float    nx, ny, nz;
    float    u, v;
    fm_color color; /* straight alpha ARGB, modulates / replaces per texenv */
} fm3d_vertex;

typedef enum fm3d_cull { FM3D_CULL_NONE = 0, FM3D_CULL_BACK, FM3D_CULL_FRONT } fm3d_cull;
typedef enum fm3d_winding { FM3D_FRONT_CCW = 0, FM3D_FRONT_CW } fm3d_winding;
typedef enum fm3d_clip_depth { FM3D_DEPTH_NEG_ONE_ONE = 0, FM3D_DEPTH_ZERO_ONE } fm3d_clip_depth;

typedef enum fm3d_compare {
    FM3D_NEVER = 0,
    FM3D_LESS,
    FM3D_EQUAL,
    FM3D_LEQUAL,
    FM3D_GREATER,
    FM3D_NOTEQUAL,
    FM3D_GEQUAL,
    FM3D_ALWAYS
} fm3d_compare;

typedef enum fm3d_filter {
    FM3D_FILTER_NEAREST = 0,
    FM3D_FILTER_BILINEAR,
    FM3D_FILTER_NEAREST_MIPMAP, /* nearest texel, nearest mip level */
    FM3D_FILTER_BILINEAR_MIPMAP, /* bilinear, nearest mip level */
    FM3D_FILTER_TRILINEAR        /* bilinear, blend of two mip levels */
} fm3d_filter;

typedef struct fm3d_sampler {
    fm3d_filter filter;
    fm_wrap     wrap_u, wrap_v;
    float       lod_bias;
} fm3d_sampler;

/* Texture combine (GL texenv style), on premultiplied colors:
 *   REPLACE   texel
 *   MODULATE  texel * vertex color
 *   DECAL     texel over vertex color
 *   ADD       texel + vertex color (saturated)
 * Without a bound texture the vertex color is used. */
typedef enum fm3d_texenv { FM3D_TEXENV_MODULATE = 0, FM3D_TEXENV_REPLACE, FM3D_TEXENV_DECAL, FM3D_TEXENV_ADD } fm3d_texenv;

/* ---- textures (immutable after creation, safe to share between threads) --- */

typedef struct fm3d_texture fm3d_texture;

/* Copies the ARGB32 image; mipmaps != 0 builds the full box filtered chain. */
FM_API fm3d_texture*     fm3d_texture_create(const fm_surface* image, int mipmaps);
FM_API fm3d_texture*     fm3d_texture_retain(fm3d_texture* t);
FM_API void              fm3d_texture_release(fm3d_texture* t);
FM_API int               fm3d_texture_levels(const fm3d_texture* t);
FM_API const fm_surface* fm3d_texture_level(const fm3d_texture* t, int level);

/* ---- context ---------------------------------------------------------------- */

typedef struct fm3d_ctx fm3d_ctx;

typedef struct fm3d_stats {
    uint64_t triangles_in;
    uint64_t triangles_clipped; /* needed clipping */
    uint64_t triangles_culled;  /* back/front face or zero area */
    uint64_t triangles_drawn;   /* sent to the rasterizer (after clipping) */
} fm3d_stats;

FM_API fm3d_ctx* fm3d_create(void);
FM_API void      fm3d_destroy(fm3d_ctx* ctx);
/* Render targets: ARGB32 back buffer (e.g. fm_swapchain_back) and an
 * optional D32F depth buffer of the same size. Resets viewport + scissor. */
FM_API void      fm3d_set_target(fm3d_ctx* ctx, fm_surface* color, fm_surface* depth);

/* transforms (kept separate for fixed function T&L) */
FM_API void fm3d_set_model(fm3d_ctx* ctx, const fm_mat4* m);
FM_API void fm3d_set_view(fm3d_ctx* ctx, const fm_mat4* m);
FM_API void fm3d_set_projection(fm3d_ctx* ctx, const fm_mat4* m);
FM_API void fm3d_set_clip_depth(fm3d_ctx* ctx, fm3d_clip_depth mode);

FM_API void fm3d_set_viewport(fm3d_ctx* ctx, int x, int y, int w, int h);
FM_API void fm3d_set_scissor(fm3d_ctx* ctx, int enable, int x, int y, int w, int h);

/* rasterizer state */
FM_API void fm3d_set_cull(fm3d_ctx* ctx, fm3d_cull cull, fm3d_winding front);
FM_API void fm3d_set_perspective_correct(fm3d_ctx* ctx, int on); /* 0 = affine (retro look) */

/* depth */
FM_API void fm3d_set_depth_test(fm3d_ctx* ctx, fm3d_compare func, int write);

/* fixed function fragment stage */
FM_API void fm3d_set_texture(fm3d_ctx* ctx, fm3d_texture* tex, const fm3d_sampler* s); /* NULL = none */
FM_API void fm3d_set_texenv(fm3d_ctx* ctx, fm3d_texenv env);
FM_API void fm3d_set_alpha_test(fm3d_ctx* ctx, fm3d_compare func, float ref); /* FM3D_ALWAYS = off */

/* output merger */
FM_API void fm3d_set_blend(fm3d_ctx* ctx, fm_blend_op op);
FM_API void fm3d_set_opacity(fm3d_ctx* ctx, float alpha); /* constant coverage multiplier */

/* clears honour the scissor rect */
FM_API void fm3d_clear_color(fm3d_ctx* ctx, fm_color c);
FM_API void fm3d_clear_depth(fm3d_ctx* ctx, float depth);

/* triangle lists */
FM_API void fm3d_draw(fm3d_ctx* ctx, const fm3d_vertex* v, int count);
FM_API void fm3d_draw_indexed(fm3d_ctx* ctx, const fm3d_vertex* v, int vertex_count, const uint32_t* indices,
                              int index_count);

/* deferred / threaded rendering (see header comment) */
FM_API void fm3d_set_deferred(fm3d_ctx* ctx, int on); /* off flushes */
FM_API void fm3d_set_executor(fm3d_ctx* ctx, fm_executor* ex);
FM_API void fm3d_set_tile_size(fm3d_ctx* ctx, int pixels); /* even, default 64 */
FM_API void fm3d_flush(fm3d_ctx* ctx);

FM_API fm3d_stats fm3d_get_stats(fm3d_ctx* ctx);
FM_API void       fm3d_reset_stats(fm3d_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM3D_H */
