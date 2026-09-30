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
 *        LOD from quad derivatives + combine op) -> alpha test / stencil + depth
 *     -> output merger (depth / stencil writes, any fm_blend_op) -> back buffer
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

/* vertex blending (skinning): up to 4 bones per vertex, weights sum to 1 */
typedef struct fm3d_skin_vertex {
    uint16_t joint[4];
    float    weight[4];
} fm3d_skin_vertex;

typedef enum fm3d_cull { FM3D_CULL_NONE = 0, FM3D_CULL_BACK, FM3D_CULL_FRONT, FM3D_CULL_FRONT_AND_BACK } fm3d_cull;
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

typedef enum fm3d_stencil_op {
    FM3D_STENCIL_KEEP = 0,
    FM3D_STENCIL_ZERO,
    FM3D_STENCIL_REPLACE,
    FM3D_STENCIL_INCR, /* clamp at 255 */
    FM3D_STENCIL_DECR, /* clamp at 0 */
    FM3D_STENCIL_INVERT,
    FM3D_STENCIL_INCR_WRAP,
    FM3D_STENCIL_DECR_WRAP
} fm3d_stencil_op;

/* which faces a stencil setting applies to (two sided stencil) */
typedef enum fm3d_face { FM3D_FACE_FRONT = 1, FM3D_FACE_BACK = 2, FM3D_FACE_FRONT_AND_BACK = 3 } fm3d_face;

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
    uint64_t hiz_rejected;      /* triangle x tile pairs skipped by hierarchical z */
    uint64_t fragments_in;      /* covered pixels entering depth / stencil (no MSAA) */
    uint64_t fragments_shaded;  /* pixels that reached the fragment stage */
} fm3d_stats;

FM_API fm3d_ctx* fm3d_create(void);
FM_API void      fm3d_destroy(fm3d_ctx* ctx);
/* Render targets: ARGB32 back buffer (e.g. fm_swapchain_back) and an
 * optional depth buffer at least as large: FM_FORMAT_D32F, D16 or D24S8
 * (D24S8 also provides the stencil buffer). Depth compares use the stored
 * precision. Resets viewport + scissor. */
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
/* polygon offset: depth += factor * max slope + units * 2^-24 (glPolygonOffset) */
FM_API void fm3d_set_depth_bias(fm3d_ctx* ctx, float factor, float units);
/* window depth = n + (f - n) * depth01 (glDepthRange) */
FM_API void fm3d_set_depth_range(fm3d_ctx* ctx, float n, float f);
/* GL_DEPTH_CLAMP: no near/far clipping, depth clamped to [0, 1] per pixel */
FM_API void fm3d_set_depth_clamp(fm3d_ctx* ctx, int enable);

/* stencil: 8 bit buffer (an FM_FORMAT_A8 surface at least as large as the
 * color target, or the stencil bits of a D24S8 depth target). Test: (ref & read_mask) FUNC (stencil & read_mask), then
 * sfail / dpfail (depth fail) / dppass ops, written through write_mask.
 * Front and back faces can be configured separately (shadow volumes). */
FM_API void fm3d_set_stencil_buffer(fm3d_ctx* ctx, fm_surface* stencil);
FM_API void fm3d_set_stencil_test(fm3d_ctx* ctx, int enable);
FM_API void fm3d_set_stencil_func(fm3d_ctx* ctx, fm3d_face face, fm3d_compare func, uint8_t ref, uint8_t read_mask);
FM_API void fm3d_set_stencil_op(fm3d_ctx* ctx, fm3d_face face, fm3d_stencil_op sfail, fm3d_stencil_op dpfail,
                                fm3d_stencil_op dppass);
FM_API void fm3d_set_stencil_write_mask(fm3d_ctx* ctx, fm3d_face face, uint8_t mask);

/* 0 disables color writes (depth / stencil only passes) */
FM_API void fm3d_set_color_write(fm3d_ctx* ctx, int enable);

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
FM_API void fm3d_clear_stencil(fm3d_ctx* ctx, uint8_t value);

/* skinning: bone matrices (joint global transform * inverse bind matrix)
 * used by fm3d_draw_skinned; copied at call time (max 256) */
FM_API void fm3d_set_bones(fm3d_ctx* ctx, const fm_mat4* bones, int count);
/* skinned triangle list (indices may be NULL): positions are blended by the
 * bones before the model / view / projection transforms */
FM_API void fm3d_draw_skinned(fm3d_ctx* ctx, const fm3d_vertex* v, const fm3d_skin_vertex* skin, int vertex_count,
                              const uint32_t* indices, int index_count);

/* triangle lists (the data is copied at call time in deferred mode) */
FM_API void fm3d_draw(fm3d_ctx* ctx, const fm3d_vertex* v, int count);
FM_API void fm3d_draw_indexed(fm3d_ctx* ctx, const fm3d_vertex* v, int vertex_count, const uint32_t* indices,
                              int index_count);

#if FM_FEATURE_VBO
/* Vertex buffers (GL VBO / IBO style): immutable vertices + optional indices,
 * copied once at creation and then drawn by reference, with no per draw
 * copy or index validation (deferred mode keeps a reference until the
 * flush). Reference counted like textures; NULL on invalid input or an out
 * of range index. */
typedef struct fm3d_buffer fm3d_buffer;
FM_API fm3d_buffer* fm3d_buffer_create(const fm3d_vertex* v, int vertex_count, const uint32_t* indices,
                                       int index_count);
FM_API fm3d_buffer* fm3d_buffer_retain(fm3d_buffer* b);
FM_API void         fm3d_buffer_release(fm3d_buffer* b);
FM_API int          fm3d_buffer_vertex_count(const fm3d_buffer* b);
FM_API int          fm3d_buffer_index_count(const fm3d_buffer* b);
/* Triangle list of `count` indices starting at index `first` when the
 * buffer has indices (every vertex of the buffer is transformed), else of
 * `count` vertices starting at vertex `first`. */
FM_API void fm3d_draw_buffer(fm3d_ctx* ctx, fm3d_buffer* b, int first, int count);
#endif

#if FM_FEATURE_TNL
/* ---- fixed function lighting (T&L, GL 1.x style) -----------------------
 * Evaluated per vertex in eye space when enabled; the result replaces the
 * vertex color (then texenv combines it with the texture as usual).
 *   color = emission + ambient_light * Ma
 *         + sum over lights of att * (La * Ma + max(N.L, 0) * Ld * Md
 *                                     + max(N.H, 0)^shininess * Ls * Ms)
 * Local viewer, Blinn half vector, att = spot / (c + l * d + q * d^2).
 * Ma, Md come from the material, or from the vertex color with
 * fm3d_set_color_material(ctx, 1). Normals are transformed with the inverse
 * transpose of model * view (skinned normals by the bones too); they need
 * not be unit length. Lights are given in world space. */
#define FM3D_MAX_LIGHTS 8
typedef enum fm3d_light_type { FM3D_LIGHT_DIRECTIONAL = 0, FM3D_LIGHT_POINT, FM3D_LIGHT_SPOT } fm3d_light_type;
typedef struct fm3d_light {
    fm3d_light_type type;
    fm_vec3         position;  /* point, spot */
    fm_vec3         direction; /* directional: the direction the light travels; spot: the cone axis */
    fm_vec3         ambient, diffuse, specular;
    float           constant, linear, quadratic; /* attenuation (point, spot) */
    float           spot_cutoff;                 /* cone half angle in radians */
    float           spot_exponent;
} fm3d_light;
typedef struct fm3d_material {
    fm_vec3 ambient, diffuse, specular, emission;
    float   alpha;     /* vertex alpha unless color material is on */
    float   shininess; /* specular exponent, 0..128 typical */
} fm3d_material;
/* white light / grey material defaults (GL values) */
FM_API fm3d_light    fm3d_light_default(fm3d_light_type type);
FM_API fm3d_material fm3d_material_default(void);
FM_API void          fm3d_set_lighting(fm3d_ctx* ctx, int on);
FM_API void          fm3d_set_light(fm3d_ctx* ctx, int index, const fm3d_light* light); /* NULL = off */
FM_API void          fm3d_set_material(fm3d_ctx* ctx, const fm3d_material* m);
FM_API void          fm3d_set_ambient_light(fm3d_ctx* ctx, fm_vec3 color); /* default 0.2 */
FM_API void          fm3d_set_color_material(fm3d_ctx* ctx, int on);
#endif

#if FM_FEATURE_SHADERS
/* ---- programmable stages ------------------------------------------------
 * C callbacks over blocks / batches (SoA): the interface a SPIR-V backend
 * will generate code for. Either stage may be NULL = the fixed function one:
 *  - fixed vertex stage: outputs 6 varyings (u, v, r, g, b, a; lit colors
 *    with T&L) from fm3d_vertex input;
 *  - fixed fragment stage: reads varyings 0..5 as u, v, r, g, b, a
 *    (texture + texenv + alpha test).
 * Varyings are interpolated perspective correct (unless disabled). */
#define FM3D_MAX_SHADER_VARYINGS 16
#define FM3D_BATCH_COLS          32 /* fragment batch: 2 rows of 32 pixels */
#define FM3D_BATCH_PIXELS        64

/* vertex shader: `count` vertices of `stride` bytes each. Write the clip
 * space position of vertex i to pos[i * out_stride + 0..3] and its
 * varyings to varyings[i * out_stride + 0..nvaryings-1]. */
#define FM3D_MAX_TEXTURE_UNITS 8

typedef struct fm3d_vs_io {
    const void* vertices;
    int         stride;
    int         count;
    const void* uniforms;
    float*      pos;
    float*      varyings;
    int         out_stride; /* floats from one vertex's output to the next */
    void*       user;       /* fm3d_program.user */
    const fm3d_texture* const* textures; /* FM3D_MAX_TEXTURE_UNITS units (entries may be NULL) */
    const fm3d_sampler*        samplers;
} fm3d_vs_io;
typedef void (*fm3d_vertex_shader)(const fm3d_vs_io* io);

/* fragment shader: one batch; pixel index i = row * 32 + col is at
 * (x + col, y + row). Covered pixels have mask[i] = 255; write straight
 * alpha RGBA (0..1) to out[0..3][i], set mask[i] = 0 to discard (declare
 * `discards` in the program so depth is tested after the shader). */
typedef struct fm3d_fs_io {
    int                  x, y, cols;
    const float* const*  varyings; /* nvaryings arrays of 64 */
    const float*         z;        /* window depth (0..1) */
    uint8_t*             mask;
    float*               out[4];
    const void*          uniforms;
    const fm3d_texture*  texture;  /* the bound texture (fm3d_set_texture = unit 0), may be NULL */
    const fm3d_sampler*  sampler;
    void*                user;     /* fm3d_program.user */
    const fm3d_texture* const* textures; /* FM3D_MAX_TEXTURE_UNITS units (entries may be NULL) */
    const fm3d_sampler*        samplers;
} fm3d_fs_io;
typedef void (*fm3d_fragment_shader)(const fm3d_fs_io* io);

/* Screen space derivatives of a varying inside a fragment batch. Pixels
 * come in 2x2 quads and varyings are evaluated for every pixel of a quad
 * (covered or not), so the differences are valid for any covered pixel
 * i < 64 with (i % 32) < cols (the GPU dFdx / dFdy of the quad). */
static inline float fm3d_ddx(const float* v, int i) { return v[i | 1] - v[i & ~1]; }
static inline float fm3d_ddy(const float* v, int i) { return v[(i & 31) + 32] - v[i & 31]; }

typedef struct fm3d_program {
    fm3d_vertex_shader   vs;        /* NULL: fixed function (fm3d_vertex input) */
    fm3d_fragment_shader fs;        /* NULL: fixed function */
    int                  nvaryings; /* written by vs (1..16; ignored with the fixed vs: 6) */
    int                  discards;  /* fs may clear mask bytes */
    void*                user;      /* handed to both stages (io->user) */
} fm3d_program;

/* NULL = fixed function pipeline. The program is copied. */
FM_API void fm3d_set_program(fm3d_ctx* ctx, const fm3d_program* program);
/* texture units for programmable stages: unit 0 is fm3d_set_texture's
 * texture, 1 .. FM3D_MAX_TEXTURE_UNITS-1 are extra (retained like it) */
FM_API void fm3d_set_texture_unit(fm3d_ctx* ctx, int unit, fm3d_texture* tex, const fm3d_sampler* sampler);
/* uniform block handed to both stages; copied (up to 64 KB), so the caller
 * may change its struct between draws */
FM_API void fm3d_set_uniforms(fm3d_ctx* ctx, const void* data, size_t bytes);
/* draw with an arbitrary vertex layout (needs a program vertex shader;
 * without one the layout must be fm3d_vertex). indices may be NULL. */
FM_API void fm3d_draw_vertices(fm3d_ctx* ctx, const void* vertices, int stride, int vertex_count,
                               const uint32_t* indices, int index_count);
/* texture sampling for fragment shaders: n points at normalized (u, v),
 * base level, the sampler's filter (mip filters use level 0) and wrap
 * modes; straight alpha RGBA SoA out */
FM_API void fm3d_sample(const fm3d_texture* t, const fm3d_sampler* s, const float* u, const float* v, int n,
                        float* r, float* g, float* b, float* a);
/* texture sampling for a whole fragment batch with mipmapping: the level
 * of detail comes from the quad derivatives of (u, v), as in the fixed
 * pipeline (nearest / bilinear use the base level, *_MIPMAP the nearest
 * level, trilinear blends two). u, v, r, g, b, a: 64 entries in batch
 * layout; pixels past io->cols are left untouched. */
FM_API void fm3d_sample_batch(const fm3d_fs_io* io, const fm3d_texture* t, const fm3d_sampler* s, const float* u,
                              const float* v, float* r, float* g, float* b, float* a);
#endif

/* Multisample anti-aliasing: 1 (off), 4 or 8 samples per pixel (standard
 * D3D sample positions). Coverage, depth and stencil are per sample in
 * internal buffers (the depth / stencil targets then only select whether
 * depth / stencil are used); fragments are shaded once per pixel and
 * blended per sample. fm3d_flush() resolves the samples into the color
 * target (per tile in deferred mode). */
FM_API void fm3d_set_msaa(fm3d_ctx* ctx, int samples);
FM_API int  fm3d_get_msaa(fm3d_ctx* ctx);

/* deferred / threaded rendering (see header comment). Deferred mode also
 * keeps per tile depth bounds (hierarchical z): triangles that fail the
 * depth test on a whole tile are skipped without changing the output. */
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
