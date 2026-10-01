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

#define FM3D_MAX_VARYINGS 64

/* Fixed function vertex. The normal is carried for T&L (lighting comes
 * later); texcoords are normalized (0..1 spans the texture). */
typedef struct fm3d_vertex {
    float    x, y, z;
    float    nx, ny, nz;
    float    u, v;
    fm_color color; /* straight alpha ARGB, modulates / replaces per texenv */
} fm3d_vertex;

/* a fixed vertex stage vertex with a second set of texture coordinates
 * (multitexture: fm3d_set_texture_stage1) */
typedef struct fm3d_vertex_mt {
    fm3d_vertex v;
    float       u2, v2;
} fm3d_vertex_mt;

typedef enum fm3d_cull { FM3D_CULL_NONE = 0, FM3D_CULL_BACK, FM3D_CULL_FRONT, FM3D_CULL_FRONT_AND_BACK } fm3d_cull;
typedef enum fm3d_winding { FM3D_FRONT_CCW = 0, FM3D_FRONT_CW } fm3d_winding;
typedef enum fm3d_clip_depth { FM3D_DEPTH_NEG_ONE_ONE = 0, FM3D_DEPTH_ZERO_ONE } fm3d_clip_depth;
/* window origin: UPPER_LEFT maps NDC y = +1 to row 0 (D3D / Vulkan style,
 * the default); LOWER_LEFT maps NDC y = -1 to row 0 (OpenGL: viewport,
 * scissor and fragment coordinates count rows bottom up, ddy points up) */
typedef enum fm3d_origin { FM3D_ORIGIN_UPPER_LEFT = 0, FM3D_ORIGIN_LOWER_LEFT } fm3d_origin;

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
    fm_wrap     wrap_w; /* 3D textures: the third coordinate */
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
/* textures with layers, all of the same size: CUBE 6 faces in +X -X +Y -Y
 * +Z -Z order, 2D_ARRAY layers, 3D slices (mipmaps are built per layer: 3D
 * textures keep their depth). FM3D_TEXTURE_STRAIGHT: texels hold straight
 * alpha (API layers that store colors as given) and are sampled as stored,
 * where the default (premultiplied) is un-premultiplied for shaders. */
typedef enum fm3d_texture_kind { FM3D_TEX_2D = 0, FM3D_TEX_CUBE, FM3D_TEX_2D_ARRAY, FM3D_TEX_3D } fm3d_texture_kind;
enum { FM3D_TEXTURE_MIPMAPS = 1, FM3D_TEXTURE_STRAIGHT = 2 };
FM_API fm3d_texture*     fm3d_texture_create_layers(fm3d_texture_kind kind, const fm_surface* const* layers, int count, unsigned flags);
FM_API fm3d_texture_kind fm3d_texture_get_kind(const fm3d_texture* t);
FM_API int               fm3d_texture_layers(const fm3d_texture* t);
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
    /* work and time (for performance overlays; wall clock nanoseconds) */
    uint64_t draws;      /* draw calls */
    uint64_t flushes;    /* deferred flushes that ran work */
    uint64_t tiles;      /* tiles with work (deferred) */
    uint64_t tile_items; /* tile x triangle chunk / clear pairs binned */
    uint64_t ns_vertex;  /* deferred: the vertex phase of the flushes */
    uint64_t ns_setup;   /* deferred: triangle setup + binning */
    uint64_t ns_raster;  /* deferred: the tile phase; immediate mode: the draw calls */
    uint64_t ns_busy;    /* summed time the workers spent in tasks (utilization:
                          * ns_busy / ((ns_vertex + ns_setup + ns_raster) * workers)) */
    uint64_t workers;    /* threads of the last flush (1 without an executor) */
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
FM_API void fm3d_set_origin(fm3d_ctx* ctx, fm3d_origin origin);

/* Primitive type of the following draws: their vertex / index lists hold
 * triangles (3 per primitive), line segments (2) or points (1). Lines and
 * points are expanded after the vertex stage into screen aligned quads
 * (OpenGL's non antialiased rules: a line covers `width` pixels across its
 * minor axis, a point a size x size square around its center, culled when
 * the center is outside the view volume) and then clipped, depth / stencil
 * tested, shaded and blended like triangles. They are never face culled
 * and are front facing. */
typedef enum fm3d_primitive { FM3D_PRIM_TRIANGLES = 0, FM3D_PRIM_LINES, FM3D_PRIM_POINTS } fm3d_primitive;
FM_API void fm3d_set_primitive(fm3d_ctx* ctx, fm3d_primitive prim);
FM_API void fm3d_set_line_width(fm3d_ctx* ctx, float width); /* pixels, default 1 */
FM_API void fm3d_set_point_size(fm3d_ctx* ctx, float size);  /* pixels, default 1 (a program's point_size_var wins) */

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
/* per channel color writes (glColorMask): the channels left out keep the
 * target's values (all off = fm3d_set_color_write(ctx, 0)) */
FM_API void fm3d_set_color_mask(fm3d_ctx* ctx, int r, int g, int b, int a);

/* fixed function fragment stage */
FM_API void fm3d_set_texture(fm3d_ctx* ctx, fm3d_texture* tex, const fm3d_sampler* s); /* NULL = none */
FM_API void fm3d_set_texenv(fm3d_ctx* ctx, fm3d_texenv env);
/* second fixed function texture stage (multitexture, as GL 1.3 / D3D7
 * texture stages): tex sampled at the vertex's second texture coordinates
 * (fm3d_vertex_mt; the first ones with plain fm3d_vertex) and combined
 * with the result of stage 0 by env. tex NULL: off. */
FM_API void fm3d_set_texture_stage1(fm3d_ctx* ctx, fm3d_texture* tex, const fm3d_sampler* s, fm3d_texenv env);
/* fixed function fog: rgb = mix(fog color, rgb, f) with f from the eye
 * distance d (the clip w): linear (end - d) / (end - start), exp
 * e^(-density d), exp2 e^(-(density d)^2), clamped to [0, 1] */
typedef enum fm3d_fog { FM3D_FOG_OFF = 0, FM3D_FOG_LINEAR, FM3D_FOG_EXP, FM3D_FOG_EXP2 } fm3d_fog;
FM_API void fm3d_set_fog(fm3d_ctx* ctx, fm3d_fog mode, fm_color color, float start, float end, float density);
FM_API void fm3d_set_alpha_test(fm3d_ctx* ctx, fm3d_compare func, float ref); /* FM3D_ALWAYS = off */

/* output merger */
FM_API void fm3d_set_blend(fm3d_ctx* ctx, fm_blend_op op);
/* OpenGL / Direct3D style output merger on straight (not premultiplied)
 * colors: with a state, fragment colors and clears are written straight
 * and combined with the target as d = eq(s * src, d * dst) per channel
 * (src ONE, dst ZERO, ADD: blending off); fm3d_set_blend is ignored. Used
 * by API layers whose render targets and textures hold straight colors.
 * NULL returns to premultiplied fm_blend_op compositing. */
typedef enum fm3d_blend_factor {
    FM3D_BF_ZERO = 0,
    FM3D_BF_ONE,
    FM3D_BF_SRC_COLOR,
    FM3D_BF_ONE_MINUS_SRC_COLOR,
    FM3D_BF_DST_COLOR,
    FM3D_BF_ONE_MINUS_DST_COLOR,
    FM3D_BF_SRC_ALPHA,
    FM3D_BF_ONE_MINUS_SRC_ALPHA,
    FM3D_BF_DST_ALPHA,
    FM3D_BF_ONE_MINUS_DST_ALPHA,
    FM3D_BF_CONSTANT_COLOR,
    FM3D_BF_ONE_MINUS_CONSTANT_COLOR,
    FM3D_BF_CONSTANT_ALPHA,
    FM3D_BF_ONE_MINUS_CONSTANT_ALPHA,
    FM3D_BF_SRC_ALPHA_SATURATE
} fm3d_blend_factor;
typedef enum fm3d_blend_eq {
    FM3D_BLEND_ADD = 0,
    FM3D_BLEND_SUBTRACT,         /* s * src - d * dst */
    FM3D_BLEND_REVERSE_SUBTRACT, /* d * dst - s * src */
    FM3D_BLEND_MIN,              /* min(s, d), factors ignored */
    FM3D_BLEND_MAX
} fm3d_blend_eq;
typedef struct fm3d_blend_state {
    fm3d_blend_factor src_rgb, dst_rgb, src_alpha, dst_alpha;
    fm3d_blend_eq     eq_rgb, eq_alpha;
    fm_color          constant; /* straight ARGB (glBlendColor) */
} fm3d_blend_state;
FM_API void fm3d_set_blend_state(fm3d_ctx* ctx, const fm3d_blend_state* state);
FM_API void fm3d_set_opacity(fm3d_ctx* ctx, float alpha); /* constant coverage multiplier */

/* clears honour the scissor rect */
FM_API void fm3d_clear_color(fm3d_ctx* ctx, fm_color c);
FM_API void fm3d_clear_depth(fm3d_ctx* ctx, float depth);
FM_API void fm3d_clear_stencil(fm3d_ctx* ctx, uint8_t value);


/* triangle lists (the data is copied at call time in deferred mode) */
FM_API void fm3d_draw(fm3d_ctx* ctx, const fm3d_vertex* v, int count);
/* multitexture vertices for the fixed vertex stage (indices NULL: a list) */
FM_API void fm3d_draw_mt(fm3d_ctx* ctx, const fm3d_vertex_mt* v, int vertex_count, const uint32_t* indices, int index_count);
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
 * transpose of model * view; they need
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
#define FM3D_MAX_SHADER_VARYINGS 64
#define FM3D_BATCH_COLS          32 /* fragment batch: 2 rows of 32 pixels */
#define FM3D_BATCH_PIXELS        64

/* vertex shader: `count` vertices of `stride` bytes each. Write the clip
 * space position of vertex i to pos[i * out_stride + 0..3] and its
 * varyings to varyings[i * out_stride + 0..nvaryings-1]. */
#define FM3D_MAX_TEXTURE_UNITS 8
#define FM3D_MAX_UNIFORM_BLOCKS 16 /* uniform buffer bindings */

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
    size_t                     uniform_size; /* bytes behind uniforms */
    const void* const*         blocks;      /* FM3D_MAX_UNIFORM_BLOCKS uniform blocks by binding ([0] = uniforms) */
    const size_t*              block_sizes; /* their sizes (0: not set) */
    int                        first_vertex; /* vertex index (gl_VertexIndex) of vertices[0] */
    int                        instance;     /* instance index (gl_InstanceIndex) */
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
    size_t                     uniform_size; /* bytes behind uniforms */
    const void* const*         blocks;      /* FM3D_MAX_UNIFORM_BLOCKS uniform blocks by binding ([0] = uniforms) */
    const size_t*              block_sizes; /* their sizes (0: not set) */
    int                        back_facing; /* the batch's triangle is a back face */
    float*                     depth_out;   /* programs with writes_depth: window depth per pixel (in: z) */
    /* programs with interpolates: the stage evaluates its varyings itself (varyings is
     * NULL). Varying k at pixel i = row * 32 + col is
     * (planes[3k] + planes[3k + 2] * dy[row] + planes[3k + 1] * dx[col]) * w[i],
     * in that operation order (the bits fatmap's interpolation gives). */
    const float*               planes; /* 3 per varying */
    const float*               dx;     /* 32 column offsets */
    const float*               dy;     /* 2 row offsets */
    const float*               w;      /* 64 perspective weights (1 without perspective) */
    /* programs with packs_color, straight color targets: write straight 8 bit ARGB
     * (0xAARRGGBB, each channel (int)(clamp01(c) * 255 + 0.5)) here instead of out[] */
    uint32_t*                  color;
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
    int                  nvaryings; /* written by vs (1..64; ignored with the fixed vs: 6) */
    int                  discards;  /* fs may clear mask bytes */
    void*                user;      /* handed to both stages (io->user) */
    /* points: varying index + 1 (0: none) holding the size the vertex
     * stage wrote (gl_PointSize), and of two varyings that receive the
     * point coordinate (gl_PointCoord: 0..1, t = 0 at the top) */
    int point_size_var, point_coord_var;
    /* the fragment stage writes depth (gl_FragDepth) into io->depth_out:
     * depth / stencil run after it, hierarchical z does not cull */
    int writes_depth;
    /* the fragment stage interpolates its varyings from io->planes (see fm3d_fs_io) */
    int interpolates;
    /* the fragment stage writes packed colors to io->color when it is not NULL */
    int packs_color;
} fm3d_program;

/* NULL = fixed function pipeline. The program is copied. */
FM_API void fm3d_set_program(fm3d_ctx* ctx, const fm3d_program* program);
/* texture units for programmable stages: unit 0 is fm3d_set_texture's
 * texture, 1 .. FM3D_MAX_TEXTURE_UNITS-1 are extra (retained like it) */
FM_API void fm3d_set_texture_unit(fm3d_ctx* ctx, int unit, fm3d_texture* tex, const fm3d_sampler* sampler);
/* uniform block handed to both stages; copied (up to 64 KB), so the caller
 * may change its struct between draws */
FM_API void fm3d_set_uniforms(fm3d_ctx* ctx, const void* data, size_t bytes);
/* the uniform block at a binding (0 .. FM3D_MAX_UNIFORM_BLOCKS - 1; 0 is
 * fm3d_set_uniforms'), copied the same way; NULL / 0 clears it */
FM_API void fm3d_set_uniform_block(fm3d_ctx* ctx, int binding, const void* data, size_t bytes);
/* vertex index of the next draws' first vertex and their instance index
 * (the vertex shader's gl_VertexIndex = base_vertex + vertex number, gl_InstanceIndex) */
FM_API void fm3d_set_draw_ids(fm3d_ctx* ctx, int base_vertex, int instance);
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
/* the same for quads in the 16 lane group order of the shader backends:
 * group k (16 lanes) is columns 8k..8k+7 of both rows, lane = row * 8 +
 * column % 8 inside it; quad q covers lanes (q / 4) * 16 + (q % 4) * 2 +
 * { 0, 1, 8, 9 }. Samples quads 0 .. nquads - 1 (at most 16). */
FM_API void fm3d_sample_quads(const fm3d_texture* t, const fm3d_sampler* s, const float* u, const float* v, int nquads,
                              float* r, float* g, float* b, float* a);
/* texture sampling at an explicit level of detail per point (textureLod;
 * lod NULL = 0): mip filters pick (or blend, trilinear) levels, the others
 * use the base level. Straight alpha RGBA SoA out. */
/* any texture kind, for programmable stages: coordinates c0, c1, c2 (n
 * each; c2 unused for 2D): 2D (s, t), CUBE (x, y, z direction), 2D_ARRAY (s,
 * t, layer), 3D (s, t, r). nquads > 0: implicit level of detail per 2x2
 * quad in fm3d_sample_quads' order (n = 64), else lod per point (NULL: 0). */
FM_API void fm3d_sample_tex(const fm3d_texture* t, const fm3d_sampler* s, const float* c0, const float* c1, const float* c2, int n,
                            int nquads, const float* lod, float* r, float* g, float* b, float* a);
FM_API void fm3d_sample_lod(const fm3d_texture* t, const fm3d_sampler* s, const float* u, const float* v, const float* lod,
                            int n, float* r, float* g, float* b, float* a);
#endif

#if FM_FEATURE_SPIRV
/* ---- SPIR-V shaders --------------------------------------------------------
 * Vertex / fragment SPIR-V modules (e.g. glslc -O shader.vert) run by
 * fatmap's batch interpreter and bound like any fm3d_program. Supported:
 * GLSL.std.450 shaders with functions inlined (glslc -O), scalars /
 * vectors / matrices / arrays / structs, one uniform block (std140, from
 * fm3d_set_uniforms) or push constants, sampler2D at binding = texture
 * unit, inputs / outputs by location, gl_Position, gl_FragCoord, discard,
 * dFdx / dFdy / fwidth, structured if / loops and the common GLSL
 * functions. Either stage may be NULL (the fixed function stage: its
 * varyings are location 0 = vec2 uv, location 1 = vec4 color).
 * Vertex inputs come from the draw's vertices (fm3d_draw_vertices) through
 * the attribute table. */
typedef struct fm3d_vertex_attrib {
    int location;   /* layout(location = ...) of the vertex shader input */
    int components; /* floats read (1..4); missing components are 0, w is 1 */
    int offset;     /* bytes from the start of the vertex */
} fm3d_vertex_attrib;
typedef struct fm3d_spirv fm3d_spirv;
/* NULL on error (message in error); words = SPIR-V 32 bit words */
FM_API fm3d_spirv*  fm3d_spirv_create(const uint32_t* vs, size_t vs_words, const uint32_t* fs, size_t fs_words,
                                      const fm3d_vertex_attrib* attribs, int nattribs, char* error, size_t error_size);
FM_API void         fm3d_spirv_destroy(fm3d_spirv* p);
/* the program to bind with fm3d_set_program; p must stay alive until the
 * draws using it are flushed */
FM_API fm3d_program fm3d_spirv_program(const fm3d_spirv* p);
/* Shader math precision (default: off): sin / cos / tan / exp / log / pow
 * as fm_fast_* (float, GPU like precision inside Vulkan's limits, about
 * twice as fast) instead of fm_* (within 1 ulp). Both are deterministic.
 * Applies to the interpreter and to fm3d_spirv_to_c output. */
FM_API void fm3d_spirv_set_fast_math(fm3d_spirv* p, int on);
/* Execution (default 1, or the environment variable FM_JIT): 0 the
 * interpreter; 1 machine code from the JIT (x86 with AVX2 / AVX-512; the
 * interpreter where the JIT cannot run a program); 2 the JIT's portable
 * reference executor (testing). Every mode renders the same bits. */
FM_API void        fm3d_spirv_set_jit(fm3d_spirv* p, int mode);
/* why the JIT left the program to the interpreter (NULL: it did not) */
FM_API const char* fm3d_spirv_jit_error(const fm3d_spirv* p);
/* Ahead of time compilation: C source of the program (both stages as
 * fm3d_program callbacks; no SPIR-V or interpreter needed at run time, only
 * fatmap with -Dshaders). It defines `fm3d_program <name>_program(void)`;
 * compiled without floating point contraction it renders exactly what the
 * interpreter renders. Returns a string to release with fm3d_spirv_free_c,
 * or NULL (message in error). */
FM_API char* fm3d_spirv_to_c(const fm3d_spirv* p, const char* name, char* error, size_t error_size);
FM_API void  fm3d_spirv_free_c(char* source);
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
