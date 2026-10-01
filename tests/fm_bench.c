/*
 * fatmap benchmark: runs each workload at every supported SIMD level on a
 * 1280x720 target and reports ms per iteration + Mpixel/s.
 *
 *   fm_bench                 all workloads, all levels
 *   fm_bench circles         workloads whose name contains "circles"
 *   fm_bench --csv out.csv   also append results to a CSV file
 *   fm_bench --prof          print the profiler zone report per workload
 *   fm_bench --time 0.5      seconds per measurement (default 0.3)
 *   fm_bench --threads 8     worker count for the "mt" column (default: all CPUs)
 *   fm_bench --strip 16      rows per strip for command lists (default 32)
 *   fm_bench --tile 32       3D tile size in pixels for deferred modes (default 64)
 *   fm_bench --column avx2   run one column only (for sampling profilers)
 *
 * Columns: one per SIMD level (immediate mode), then "cmdlist" (deferred,
 * serial) and "mtN" (deferred, N threads), both at the best SIMD level.
 */
#include <fatmap/fatmap.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 1280
#define H 720

static uint32_t g_rng;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float rndf(void) { return (float)(rnd() & 0xffffff) / 16777216.0f; }
static fm_color rnd_color(int alpha) { return FM_RGBA(rnd() & 255, rnd() & 255, rnd() & 255, alpha < 0 ? rnd() & 255 : (uint32_t)alpha); }

typedef struct bench_env {
    fm_surface*   fb;
    fm_surface*   zb;
    fm2d_ctx*     c;
    fm3d_ctx*     c3;
    fm_surface*   tex;
    fm_surface*   sprite;
    fm3d_texture* tex3;
    fm3d_texture* tex3_big; /* 4096 x 4096, no mips (texture cache behaviour) */
    fm3d_vertex*  cube;  /* 36 vertices */
#if FM_FEATURE_VBO
    fm3d_buffer*  cube_buf; /* the same cube as a vertex buffer */
#endif
    fm3d_vertex*  grid;  /* floor grid vertices */
    uint32_t*     gidx;
    int           ngrid, ngidx;
} bench_env;

typedef struct workload {
    const char* name;
    double      pixels; /* nominal pixels touched per iteration (for Mpix/s) */
    void (*run)(bench_env* e);
} workload;

static void w_clear(bench_env* e) { fm2d_clear(e->c, FM_RGB(10, 20, 30)); }

static void w_rect_opaque(bench_env* e)
{
    g_rng = 1;
    for (int i = 0; i < 200; i++) {
        fm2d_set_fill_color(e->c, rnd_color(255));
        fm2d_fill_rect(e->c, (float)(rnd() % (W - 100)), (float)(rnd() % (H - 100)), 100, 100);
    }
}

static void w_rect_alpha(bench_env* e)
{
    g_rng = 2;
    for (int i = 0; i < 200; i++) {
        fm2d_set_fill_color(e->c, rnd_color(128));
        fm2d_fill_rect(e->c, (float)(rnd() % (W - 100)), (float)(rnd() % (H - 100)), 100, 100);
    }
}

static void w_rect_aa(bench_env* e)
{
    g_rng = 3;
    for (int i = 0; i < 200; i++) {
        fm2d_set_fill_color(e->c, rnd_color(200));
        fm2d_fill_rect(e->c, rndf() * (W - 100), rndf() * (H - 100), 100.3f, 99.6f);
    }
}

static void w_circles_small(bench_env* e)
{
    g_rng = 4;
    for (int i = 0; i < 1000; i++) {
        fm2d_set_fill_color(e->c, rnd_color(-1));
        fm2d_begin_path(e->c);
        fm2d_arc(e->c, rndf() * W, rndf() * H, 4 + rndf() * 16, 0, 6.2831853f, 0);
        fm2d_fill(e->c, FM_FILL_NONZERO);
    }
}

static void w_circles_large(bench_env* e)
{
    g_rng = 5;
    for (int i = 0; i < 20; i++) {
        fm2d_set_fill_color(e->c, rnd_color(-1));
        fm2d_begin_path(e->c);
        fm2d_arc(e->c, rndf() * W, rndf() * H, 150 + rndf() * 150, 0, 6.2831853f, 0);
        fm2d_fill(e->c, FM_FILL_NONZERO);
    }
}

static void w_circles_aliased(bench_env* e)
{
    fm2d_set_antialias(e->c, FM_AA_NONE);
    w_circles_small(e);
    fm2d_set_antialias(e->c, FM_AA_ANALYTIC);
}

static void w_lines(bench_env* e)
{
    g_rng = 6;
    fm2d_set_line_cap(e->c, FM2D_CAP_ROUND);
    for (int i = 0; i < 2000; i++) {
        fm2d_set_stroke_color(e->c, rnd_color(-1));
        fm2d_set_line_width(e->c, 1 + rndf() * 3);
        fm2d_begin_path(e->c);
        float x = rndf() * W, y = rndf() * H;
        fm2d_move_to(e->c, x, y);
        fm2d_line_to(e->c, x + (rndf() - 0.5f) * 200, y + (rndf() - 0.5f) * 200);
        fm2d_stroke(e->c);
    }
    fm2d_set_line_cap(e->c, FM2D_CAP_BUTT);
}

static void w_polyline(bench_env* e)
{
    fm2d_set_line_join(e->c, FM2D_JOIN_ROUND);
    fm2d_set_line_width(e->c, 6);
    fm2d_set_stroke_color(e->c, FM_RGBA(80, 220, 255, 200));
    for (int k = 0; k < 8; k++) {
        fm2d_begin_path(e->c);
        for (int i = 0; i <= 400; i++) {
            float x = (float)i * (W / 400.0f), y = H * 0.5f + sinf((float)i * 0.08f + (float)k) * 250.0f;
            if (i == 0)
                fm2d_move_to(e->c, x, y);
            else
                fm2d_line_to(e->c, x, y);
        }
        fm2d_stroke(e->c);
    }
    fm2d_set_line_join(e->c, FM2D_JOIN_MITER);
}

static void w_linear_gradient(bench_env* e)
{
    fm2d_paint* g = fm2d_paint_linear(0, 0, W, H);
    fm2d_paint_add_stop(g, 0, FM_RGB(255, 0, 0));
    fm2d_paint_add_stop(g, 0.5f, FM_RGBA(0, 255, 0, 128));
    fm2d_paint_add_stop(g, 1, FM_RGB(0, 0, 255));
    fm2d_set_fill_paint(e->c, g);
    fm2d_fill_rect(e->c, 0, 0, W, H);
    fm2d_paint_release(g);
}

static void w_radial_gradient(bench_env* e)
{
    fm2d_paint* g = fm2d_paint_radial(W / 2, H / 2, 10, W / 2 + 50, H / 2, 600);
    fm2d_paint_add_stop(g, 0, FM_RGB(255, 255, 255));
    fm2d_paint_add_stop(g, 1, FM_RGB(0, 0, 80));
    fm2d_set_fill_paint(e->c, g);
    fm2d_fill_rect(e->c, 0, 0, W, H);
    fm2d_paint_release(g);
}

static void w_pattern_bilinear(bench_env* e)
{
    fm2d_paint* p = fm2d_paint_pattern(e->tex, FM2D_REPEAT);
    fm2d_save(e->c);
    fm2d_translate(e->c, W / 2, H / 2);
    fm2d_rotate(e->c, 0.3f);
    fm2d_scale(e->c, 1.3f, 1.3f);
    fm2d_set_fill_paint(e->c, p);
    fm2d_fill_rect(e->c, -W, -H, 2 * W, 2 * H);
    fm2d_restore(e->c);
    fm2d_paint_release(p);
}

static void w_pattern_nearest(bench_env* e)
{
    fm2d_set_image_smoothing(e->c, 0);
    w_pattern_bilinear(e);
    fm2d_set_image_smoothing(e->c, 1);
}

static void w_sprites(bench_env* e)
{
    g_rng = 7;
    for (int i = 0; i < 1000; i++)
        fm2d_draw_image(e->c, e->sprite, (float)(rnd() % (W - 64)), (float)(rnd() % (H - 64)));
}

static void w_sprites_scaled(bench_env* e)
{
    g_rng = 8;
    for (int i = 0; i < 300; i++) {
        float s = 0.5f + rndf() * 1.5f;
        fm2d_draw_image_scaled(e->c, e->sprite, rndf() * (W - 128), rndf() * (H - 128), 64 * s, 64 * s);
    }
}

static void w_blend_multiply(bench_env* e)
{
    fm2d_set_composite_op(e->c, FM_OP_MULTIPLY);
    fm2d_set_fill_color(e->c, FM_RGBA(200, 150, 100, 200));
    fm2d_fill_rect(e->c, 0, 0, W, H);
    fm2d_set_composite_op(e->c, FM_OP_SRC_OVER);
}

static void w_blend_overlay(bench_env* e)
{
    fm2d_set_composite_op(e->c, FM_OP_OVERLAY);
    fm2d_set_fill_color(e->c, FM_RGBA(200, 150, 100, 200));
    fm2d_fill_rect(e->c, 0, 0, W, H);
    fm2d_set_composite_op(e->c, FM_OP_SRC_OVER);
}

static void w_clip_circle(bench_env* e)
{
    fm2d_save(e->c);
    fm2d_begin_path(e->c);
    fm2d_arc(e->c, W / 2, H / 2, 300, 0, 6.2831853f, 0);
    fm2d_clip(e->c, FM_FILL_NONZERO);
    fm2d_set_fill_color(e->c, FM_RGBA(255, 0, 0, 128));
    fm2d_fill_rect(e->c, 0, 0, W, H);
    fm2d_restore(e->c);
}

/* ---- 3D ---- */

static fm3d_vertex bv(float x, float y, float z, float u, float v, fm_color c)
{
    fm3d_vertex r;
    memset(&r, 0, sizeof(r));
    r.x = x, r.y = y, r.z = z, r.u = u, r.v = v, r.color = c;
    return r;
}

static void cam3d(bench_env* e)
{
    fm3d_ctx* c    = e->c3;
    fm_mat4   proj = fm_perspective(fm_radians(60), (float)W / H, 0.1f, 500.0f);
    fm_mat4   view = fm_lookat(fm_v3(0, 6, 22), fm_v3(0, 0, 0), fm_v3(0, 1, 0));
    fm_mat4   id   = fm_mat4_identity();
    fm3d_set_target(c, e->fb, e->zb);
    fm3d_set_projection(c, &proj);
    fm3d_set_view(c, &view);
    fm3d_set_model(c, &id);
    fm3d_set_depth_test(c, FM3D_LESS, 1);
    fm3d_set_cull(c, FM3D_CULL_BACK, FM3D_FRONT_CCW);
    fm3d_set_blend(c, FM_OP_SRC_OVER);
    fm3d_clear_color(c, FM_RGB(10, 20, 30));
    fm3d_clear_depth(c, 1.0f);
}

static void w3_cubes_body(bench_env* e, int vbo)
{
    (void)vbo;
    fm3d_sampler s = { FM3D_FILTER_BILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture(e->c3, e->tex3, &s);
    g_rng = 11;
    for (int i = 0; i < 2000; i++) {
        fm_mat4 m = fm_translate(fm_mat4_identity(), fm_v3((rndf() - 0.5f) * 40, (rndf() - 0.5f) * 14, -rndf() * 40));
        m         = fm_rotate(m, rndf() * 6.28f, fm_v3(rndf(), 1, rndf()));
        fm3d_set_model(e->c3, &m);
#if FM_FEATURE_VBO
        if (vbo)
            fm3d_draw_buffer(e->c3, e->cube_buf, 0, 36);
        else
#endif
            fm3d_draw(e->c3, e->cube, 36);
    }
}

static void w3_cubes(bench_env* e)
{
    cam3d(e);
    w3_cubes_body(e, 0);
}
#if FM_FEATURE_VBO
static void w3_cubes_vbo(bench_env* e)
{
    cam3d(e);
    w3_cubes_body(e, 1);
}
#endif

static void w3_floor(bench_env* e, fm3d_filter f)
{
    cam3d(e);
    fm3d_sampler s = { f, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture(e->c3, e->tex3, &s);
    fm3d_draw_indexed(e->c3, e->grid, e->ngrid, e->gidx, e->ngidx);
}
static void w3_floor_bilinear(bench_env* e) { w3_floor(e, FM3D_FILTER_BILINEAR); }
static void w3_floor_trilinear(bench_env* e) { w3_floor(e, FM3D_FILTER_TRILINEAR); }

/* full screen quad at 1 texel per pixel, bilinear, no mips: the texture
 * footprint per frame is 1280 x 720 texels. rot90 walks the texture along
 * v (one texture row per screen pixel), the worst case for row major
 * texture storage. The small texture repeats and stays in cache. */
static void w3_tex1to1(bench_env* e, fm3d_texture* t, int size, int rot90)
{
    cam3d(e);
    fm_mat4 id = fm_mat4_identity(), ortho = fm_ortho(0, W, H, 0, -1, 1);
    fm3d_set_projection(e->c3, &ortho);
    fm3d_set_view(e->c3, &id);
    fm3d_set_depth_test(e->c3, FM3D_ALWAYS, 0);
    fm3d_set_cull(e->c3, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_sampler s = { FM3D_FILTER_BILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture(e->c3, t, &s);
    float    fx = (float)W / (float)size, fy = (float)H / (float)size;
    fm_color c  = FM_RGB(255, 255, 255);
    /* corner (sx, sy) -> (u, v); rot90 swaps the axes */
    float    U[4], V[4], X[4] = { 0, W, W, 0 }, Y[4] = { 0, 0, H, H };
    for (int k = 0; k < 4; k++) {
        float a = X[k] / W * fx, b = Y[k] / H * fy;
        U[k]    = rot90 ? b : a;
        V[k]    = rot90 ? a : b;
    }
    fm3d_vertex q[6];
    const int   o[6] = { 0, 1, 2, 0, 2, 3 };
    for (int i = 0; i < 6; i++) q[i] = bv(X[o[i]], Y[o[i]], 0, U[o[i]], V[o[i]], c);
    fm3d_draw(e->c3, q, 6);
}
static void w3_tex_small_rot0(bench_env* e) { w3_tex1to1(e, e->tex3, 256, 0); }
static void w3_tex_small_rot90(bench_env* e) { w3_tex1to1(e, e->tex3, 256, 1); }
static void w3_tex_large_rot0(bench_env* e) { w3_tex1to1(e, e->tex3_big, 4096, 0); }
static void w3_tex_large_rot90(bench_env* e) { w3_tex1to1(e, e->tex3_big, 4096, 1); }

#if FM_FEATURE_SPIRV
/* full screen quad through tests/spirv/t_basic.vert + t_control.frag (loop,
 * branches, pow, texture, fwidth) as SPIR-V, and the same math as C shaders:
 * the interpreter's overhead over compiled C */
#  include "spirv_shaders.h"
typedef struct bsv_vert {
    float pos[3], color[4];
} bsv_vert;
typedef struct bsv_u {
    fm_mat4 mvp;
    float   tint[4], time, pad[3];
} bsv_u;
static void bsv_vs(const fm3d_vs_io* io)
{
    const bsv_u* U = (const bsv_u*)io->uniforms;
    const float* m = &U->mvp.c[0].x;
    for (int i = 0; i < io->count; i++) {
        const bsv_vert* v = (const bsv_vert*)((const char*)io->vertices + (size_t)i * (size_t)io->stride);
        float*          o = io->pos + (size_t)i * (size_t)io->out_stride;
        float*          q = io->varyings + (size_t)i * (size_t)io->out_stride;
        float           p4[4] = { v->pos[0], v->pos[1], v->pos[2], 1.0f };
        for (int r = 0; r < 4; r++) {
            float d = m[r] * p4[0];
            for (int c = 1; c < 4; c++) d += m[c * 4 + r] * p4[c];
            o[r] = d;
        }
        for (int k = 0; k < 4; k++) q[k] = v->color[k];
        q[4] = v->pos[0] * 0.01f, q[5] = v->pos[1] * 0.01f;
    }
}
static void bsv_fs(const fm3d_fs_io* io)
{
    const bsv_u* U = (const bsv_u*)io->uniforms;
    float        tr[64], tg[64], tb[64], ta[64];
    fm3d_sample_batch(io, io->textures[1], &io->samplers[1], io->varyings[4], io->varyings[5], tr, tg, tb, ta);
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++) {
        if ((i & 31) >= io->cols) continue;
        float c[3], acc = 0.0f;
        for (int k = 0; k < 3; k++) c[k] = io->varyings[k][i] * U->tint[k];
        for (int k = 0; k < 4; k++) acc += fm_sinf((float)k * io->varyings[4][i] + U->time);
        if (io->varyings[0][i] > 0.5f)
            for (int k = 0; k < 3; k++) c[k] = c[k] * 0.5f + acc * 0.25f * 0.5f;
        else
            for (int k = 0; k < 3; k++) c[k] = fm_powf(c[k], 2.2f);
        if (io->varyings[5][i] > 1.9f) {
            io->mask[i] = 0;
            continue;
        }
        float w = fabsf(fm3d_ddx(io->varyings[4], i)) + fabsf(fm3d_ddy(io->varyings[4], i));
        float t[3] = { tr[i], tg[i], tb[i] };
        for (int k = 0; k < 3; k++) io->out[k][i] = fminf(fmaxf(c[k] + t[k] * 0.1f + w, 0.0f), 1.0f);
        io->out[3][i] = io->varyings[3][i];
    }
}
static fm3d_spirv* g_bsv;
#  if FM_TEST_AOT
fm3d_program aot_control_program(void); /* spirv_aot.c (fm3d_spirv_to_c at build time) */
fm3d_program aot_seascape_program(void);
fm3d_program aot_seascape_fast_program(void);
#  endif
static void w3_shader(bench_env* e, int spirv)
{
    cam3d(e);
    bsv_u U;
    memset(&U, 0, sizeof(U));
    U.mvp = fm_ortho(0, W, H, 0, -1, 1);
    U.tint[0] = 1, U.tint[1] = 0.8f, U.tint[2] = 0.6f, U.tint[3] = 1, U.time = 0.7f;
    bsv_vert q[6] = { { { 0, 0, 0 }, { 1, 0.2f, 0.2f, 1 } }, { { W, 0, 0 }, { 0.2f, 1, 0.2f, 1 } }, { { W, H, 0 }, { 0.2f, 0.2f, 1, 1 } },
                      { { 0, 0, 0 }, { 1, 0.2f, 0.2f, 1 } }, { { W, H, 0 }, { 0.2f, 0.2f, 1, 1 } }, { { 0, H, 0 }, { 0.9f, 0.9f, 0.2f, 1 } } };
    if (!g_bsv) {
        fm3d_vertex_attrib a[2] = { { 0, 3, 0 }, { 1, 4, 12 } };
        char               err[256];
        g_bsv = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_control_frag,
                                  sizeof(spv_t_control_frag) / 4, a, 2, err, sizeof(err));
        if (!g_bsv) printf("spirv: %s%c", err, 10);
    }
    fm3d_program cp = { bsv_vs, bsv_fs, 6, 1, NULL, 0, 0, 0, 0 }, sp = fm3d_spirv_program(g_bsv);
#  if FM_TEST_AOT
    if (spirv == 2) sp = aot_control_program();
#  endif
    fm3d_set_program(e->c3, spirv ? &sp : &cp);
    fm3d_set_uniforms(e->c3, &U, sizeof(U));
    fm3d_sampler s = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture_unit(e->c3, 1, e->tex3, &s);
    fm3d_set_depth_test(e->c3, FM3D_ALWAYS, 0);
    fm3d_set_cull(e->c3, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_draw_vertices(e->c3, q, (int)sizeof(bsv_vert), 6, NULL, 6);
    fm3d_set_program(e->c3, NULL);
    fm3d_set_texture_unit(e->c3, 1, NULL, NULL);
}
static void w3_shader_c(bench_env* e) { w3_shader(e, 0); }
static void w3_shader_spirv(bench_env* e) { w3_shader(e, 1); }
#  if FM_TEST_AOT
static void w3_shader_aot(bench_env* e) { w3_shader(e, 2); }
#  endif

/* "Seascape" (TDM, Shadertoy; tests/spirv/seascape.frag): a ray marched
 * sea, heavy on loops, divergent breaks and transcendentals, on a SEA_W x
 * SEA_H quad through the fixed vertex stage */
#  define SEA_W 480
#  define SEA_H 270
typedef struct bsea_u { /* std140: vec3 iResolution @0, float iTime @12, vec4 iMouse @16 */
    float res[3], time, mouse[4];
} bsea_u;
static fm3d_spirv* g_bsea;
static void w3_seascape_draw(bench_env* e, const fm3d_program* p)
{
    fm3d_ctx* c  = e->c3;
    fm_mat4   pr = fm_ortho(0, W, H, 0, -1, 1), id = fm_mat4_identity();
    cam3d(e);
    fm3d_set_projection(c, &pr);
    fm3d_set_view(c, &id);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    bsea_u U;
    memset(&U, 0, sizeof(U));
    U.res[0] = SEA_W, U.res[1] = SEA_H, U.res[2] = 1, U.time = 3.0f;
    fm3d_set_program(c, p);
    fm3d_set_uniforms(c, &U, sizeof(U));
    fm3d_vertex q[6];
    const float X[4] = { 0, SEA_W, SEA_W, 0 }, Y[4] = { 0, 0, SEA_H, SEA_H };
    const int   o[6] = { 0, 1, 2, 0, 2, 3 };
    for (int i = 0; i < 6; i++) q[i] = bv(X[o[i]], Y[o[i]], 0, 0, 0, FM_RGB(255, 255, 255));
    fm3d_draw(c, q, 6);
    fm3d_set_program(c, NULL);
}
static void w3_seascape_spirv(bench_env* e)
{
    if (!g_bsea) {
        char err[256];
        g_bsea = fm3d_spirv_create(NULL, 0, spv_seascape_frag, sizeof(spv_seascape_frag) / 4, NULL, 0, err, sizeof(err));
        if (!g_bsea) printf("seascape: %s%c", err, 10);
    }
    fm3d_program p = fm3d_spirv_program(g_bsea);
    w3_seascape_draw(e, &p);
}
static fm3d_spirv* g_bsea_fast;
static void        w3_seascape_fast(bench_env* e) /* fm3d_spirv_set_fast_math */
{
    if (!g_bsea_fast) {
        char err[256];
        g_bsea_fast = fm3d_spirv_create(NULL, 0, spv_seascape_frag, sizeof(spv_seascape_frag) / 4, NULL, 0, err, sizeof(err));
        if (!g_bsea_fast) printf("seascape: %s%c", err, 10);
        fm3d_spirv_set_fast_math(g_bsea_fast, 1);
    }
    fm3d_program p = fm3d_spirv_program(g_bsea_fast);
    w3_seascape_draw(e, &p);
}
/* Doom 3 BFG style light interactions (tests/spirv/bfg_interaction.*,
 * tests/bfg_scene.h): 1280 x 720, a 64 x 36 grid, 4 additive light passes
 * with 5 trilinear textures; fatgl and Mesa llvmpipe render the same scene
 * (fatgl's tools/bench). _O0: glslang's unoptimized SPIR-V, what fatgl
 * passes on today */
#  include "bfg_scene.h"
typedef struct bfg_vert { /* the stream the program reads: locations 0 .. 4, vec4 each */
    float pos[4], st[4], nrm[4], tan[4], col[4];
} bfg_vert;
static fm3d_spirv*   g_bfg[2];
static fm3d_texture* g_bfg_tex[5];
static bfg_vert*     g_bfg_v;
static uint32_t*     g_bfg_i;
static const fm3d_vertex_attrib g_bfg_attr[5] = { { 0, 4, 0 }, { 1, 4, 16 }, { 2, 4, 32 }, { 3, 4, 48 }, { 4, 4, 64 } };
static void bfg_setup(void)
{
    if (g_bfg_v) return;
    for (int i = 0; i < 5; i++) {
        int         w = scene_tex_w(i), h = scene_tex_h(i);
        uint8_t*    rgba = (uint8_t*)malloc((size_t)w * (size_t)h * 4);
        fm_surface* s    = fm_surface_create(w, h, FM_FORMAT_ARGB32);
        scene_texture(i, rgba);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                const uint8_t* p           = rgba + ((size_t)y * (size_t)w + (size_t)x) * 4;
                fm_surface_row32(s, y)[x] = FM_RGBA(p[0], p[1], p[2], p[3]); /* straight, like GL */
            }
        const fm_surface* img = s;
        g_bfg_tex[i]          = fm3d_texture_create_layers(FM3D_TEX_2D, &img, 1, FM3D_TEXTURE_STRAIGHT | FM3D_TEXTURE_MIPMAPS);
        fm_surface_destroy(s);
        free(rgba);
    }
    static scene_vert sv[SCENE_NV];
    g_bfg_v = (bfg_vert*)malloc(sizeof(bfg_vert) * SCENE_NV);
    g_bfg_i = (uint32_t*)malloc(sizeof(uint32_t) * SCENE_NI);
    scene_mesh(sv, g_bfg_i);
    for (int i = 0; i < SCENE_NV; i++) {
        bfg_vert* o = &g_bfg_v[i];
        memcpy(o->pos, sv[i].xyzw, 16);
        o->st[0] = sv[i].st[0], o->st[1] = sv[i].st[1], o->st[2] = 0, o->st[3] = 1;
        for (int k = 0; k < 4; k++)
            o->nrm[k] = sv[i].normal[k] / 255.0f, o->tan[k] = sv[i].tangent[k] / 255.0f, o->col[k] = sv[i].color[k] / 255.0f;
    }
}
static void bfg_draw(bench_env* e, const fm3d_program* p)
{
    fm3d_ctx* c = e->c3;
    bfg_setup();
    cam3d(e);
    fm3d_set_origin(c, FM3D_ORIGIN_LOWER_LEFT);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_blend_state bs = { FM3D_BF_ONE, FM3D_BF_ONE, FM3D_BF_ONE, FM3D_BF_ONE, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
    fm3d_set_blend_state(c, &bs);
    fm3d_clear_color(c, FM_RGBA(0, 0, 0, 255));
    for (int i = 0; i < 5; i++) {
        fm_wrap      wr = scene_tex_clamp(i) ? FM_WRAP_BORDER : FM_WRAP_REPEAT;
        fm3d_sampler s  = { FM3D_FILTER_TRILINEAR, wr, wr, 0, wr };
        fm3d_set_texture_unit(c, i, g_bfg_tex[i], &s);
    }
    fm3d_set_program(c, p);
    for (int l = 0; l < SCENE_LIGHTS; l++) {
        float va[18][4], fa[2][4];
        scene_uniforms(l, va, fa);
        fm3d_set_uniform_block(c, 0, va, sizeof(va));
        fm3d_set_uniform_block(c, 1, fa, sizeof(fa));
        fm3d_draw_vertices(c, g_bfg_v, (int)sizeof(bfg_vert), SCENE_NV, g_bfg_i, SCENE_NI);
    }
    fm3d_set_program(c, NULL);
    for (int i = 0; i < 5; i++) fm3d_set_texture_unit(c, i, NULL, NULL);
    fm3d_set_origin(c, FM3D_ORIGIN_UPPER_LEFT);
    fm3d_set_blend(c, FM_OP_SRC_OVER);
}
static void bfg_interp(bench_env* e, int o0)
{
    if (!g_bfg[o0]) {
        char err[256];
        g_bfg[o0] = o0 ? fm3d_spirv_create(spv_bfg_interaction_vert_O0, sizeof(spv_bfg_interaction_vert_O0) / 4, spv_bfg_interaction_frag_O0,
                                           sizeof(spv_bfg_interaction_frag_O0) / 4, g_bfg_attr, 5, err, sizeof(err))
                       : fm3d_spirv_create(spv_bfg_interaction_vert, sizeof(spv_bfg_interaction_vert) / 4, spv_bfg_interaction_frag,
                                           sizeof(spv_bfg_interaction_frag) / 4, g_bfg_attr, 5, err, sizeof(err));
        if (!g_bfg[o0]) printf("bfg: %s%c", err, 10);
        else fm3d_spirv_set_fast_math(g_bfg[o0], 1);
    }
    fm3d_program p = fm3d_spirv_program(g_bfg[o0]);
    bfg_draw(e, &p);
}
static void w3_bfg_spirv(bench_env* e) { bfg_interp(e, 0); }
static void w3_bfg_spirv_O0(bench_env* e) { bfg_interp(e, 1); }
#  if FM_TEST_AOT
fm3d_program aot_bfg_fast_program(void);
static void  w3_bfg_aot_fast(bench_env* e)
{
    fm3d_program p = aot_bfg_fast_program();
    bfg_draw(e, &p);
}
#  endif
#  if FM_TEST_AOT
static void w3_seascape_aot(bench_env* e)
{
    fm3d_program p = aot_seascape_program();
    w3_seascape_draw(e, &p);
}
static void w3_seascape_aot_fast(bench_env* e)
{
    fm3d_program p = aot_seascape_fast_program();
    w3_seascape_draw(e, &p);
}
#  endif
#endif

static void w3_alpha_quads(bench_env* e)
{
    cam3d(e);
    fm3d_set_texture(e->c3, NULL, NULL);
    fm3d_set_depth_test(e->c3, FM3D_ALWAYS, 0);
    fm3d_set_cull(e->c3, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm_mat4 id = fm_mat4_identity(), ortho = fm_ortho(0, W, H, 0, -1, 1);
    fm3d_set_projection(e->c3, &ortho);
    fm3d_set_view(e->c3, &id);
    g_rng = 12;
    for (int i = 0; i < 30; i++) {
        fm_color    col = rnd_color(100);
        float       x = rndf() * 200, y = rndf() * 100;
        fm3d_vertex q[6] = { bv(x, y, 0, 0, 0, col),        bv(x + 1000, y, 0, 1, 0, col),
                             bv(x + 1000, y + 600, 0, 1, 1, col), bv(x, y, 0, 0, 0, col),
                             bv(x + 1000, y + 600, 0, 1, 1, col), bv(x, y + 600, 0, 0, 1, col) };
        fm3d_draw(e->c3, q, 6);
    }
}

static void w3_small_tris(bench_env* e)
{
    cam3d(e);
    fm3d_set_texture(e->c3, NULL, NULL);
    fm3d_set_cull(e->c3, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    static fm3d_vertex v[30000];
    g_rng = 13;
    for (int i = 0; i < 10000; i++) {
        float x = (rndf() - 0.5f) * 40, y = (rndf() - 0.5f) * 14, z = -rndf() * 30;
        for (int k = 0; k < 3; k++)
            v[3 * i + k] = bv(x + (rndf() - 0.5f) * 0.6f, y + (rndf() - 0.5f) * 0.6f, z, 0, 0, rnd_color(255));
    }
    fm3d_draw(e->c3, v, 30000);
}

static void w3_occluded(bench_env* e)
{
    cam3d(e);
    /* a wall right in front of the camera hides most of the cubes behind it */
    fm3d_set_texture(e->c3, NULL, NULL);
    fm3d_set_cull(e->c3, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm_color    wc   = FM_RGB(60, 60, 70);
    fm3d_vertex wall[6] = { bv(-30, -12, 8, 0, 0, wc), bv(30, -12, 8, 0, 0, wc), bv(30, 20, 8, 0, 0, wc),
                            bv(-30, -12, 8, 0, 0, wc), bv(30, 20, 8, 0, 0, wc), bv(-30, 20, 8, 0, 0, wc) };
    fm3d_draw(e->c3, wall, 6);
    fm3d_set_cull(e->c3, FM3D_CULL_BACK, FM3D_FRONT_CCW);
    w3_cubes_body(e, 0);
}

static void w3_cubes_msaa4(bench_env* e)
{
    fm3d_set_msaa(e->c3, 4);
    w3_cubes(e);
    fm3d_flush(e->c3);
    fm3d_set_msaa(e->c3, 1);
}

static const workload g_workloads[] = {
    { "clear", W * H, w_clear },
    { "rect_opaque", 200 * 100 * 100, w_rect_opaque },
    { "rect_alpha", 200 * 100 * 100, w_rect_alpha },
    { "rect_aa", 200 * 100 * 100, w_rect_aa },
    { "circles_small_aa", 1000 * 3.14159 * 144, w_circles_small },
    { "circles_small_aliased", 1000 * 3.14159 * 144, w_circles_aliased },
    { "circles_large_aa", 20 * 3.14159 * 225 * 225, w_circles_large },
    { "lines_2000", 2000 * 100 * 2.5, w_lines },
    { "polyline_round", 8 * 1280 * 6 * 1.5, w_polyline },
    { "linear_gradient", W * H, w_linear_gradient },
    { "radial_gradient", W * H, w_radial_gradient },
    { "pattern_bilinear", W * H, w_pattern_bilinear },
    { "pattern_nearest", W * H, w_pattern_nearest },
    { "sprites_1to1", 1000 * 64 * 64, w_sprites },
    { "sprites_scaled", 300 * 64 * 64 * 1.56, w_sprites_scaled },
    { "blend_multiply", W * H, w_blend_multiply },
    { "blend_overlay", W * H, w_blend_overlay },
    { "clip_circle", W * H, w_clip_circle },
    { "3d_cubes_2000", W * H, w3_cubes },
#if FM_FEATURE_VBO
    { "3d_cubes_2000_vbo", W * H, w3_cubes_vbo },
#endif
    { "3d_cubes_occluded", W * H, w3_occluded },
    { "3d_cubes_msaa4", W * H, w3_cubes_msaa4 },
    { "3d_floor_bilinear", W * H * 0.6, w3_floor_bilinear },
    { "3d_floor_trilinear", W * H * 0.6, w3_floor_trilinear },
    { "3d_alpha_quads_30", 30.0 * 1000 * 600, w3_alpha_quads },
    { "3d_small_tris_10k", 10000 * 30, w3_small_tris },
    { "3d_tex_small_rot0", W * H, w3_tex_small_rot0 },
    { "3d_tex_small_rot90", W * H, w3_tex_small_rot90 },
    { "3d_tex_large_rot0", W * H, w3_tex_large_rot0 },
    { "3d_tex_large_rot90", W * H, w3_tex_large_rot90 },
#if FM_FEATURE_SPIRV
    { "3d_shader_c", W * H, w3_shader_c },
    { "3d_shader_spirv", W * H, w3_shader_spirv },
#  if FM_TEST_AOT
    { "3d_shader_aot", W * H, w3_shader_aot },
#  endif
    { "3d_seascape_spirv", SEA_W * SEA_H, w3_seascape_spirv },
    { "3d_seascape_fast", SEA_W * SEA_H, w3_seascape_fast },
#  if FM_TEST_AOT
    { "3d_seascape_aot", SEA_W * SEA_H, w3_seascape_aot },
    { "3d_seascape_aot_fast", SEA_W * SEA_H, w3_seascape_aot_fast },
#  endif
    { "3d_bfg_spirv", W * H * SCENE_LIGHTS, w3_bfg_spirv },
    { "3d_bfg_spirv_O0", W * H * SCENE_LIGHTS, w3_bfg_spirv_O0 },
#  if FM_TEST_AOT
    { "3d_bfg_aot_fast", W * H * SCENE_LIGHTS, w3_bfg_aot_fast },
#  endif
#endif
};

static fm_surface* make_tex(int w, int h)
{
    fm_surface* s = fm_surface_create(w, h, FM_FORMAT_ARGB32);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int c                      = ((x >> 4) ^ (y >> 4)) & 1;
            fm_surface_row32(s, y)[x]  = fm_premultiply(c ? FM_RGB(x & 255, y & 255, 128) : FM_RGBA(20, 40, 60, 180));
        }
    return s;
}

int main(int argc, char** argv)
{
    const char* filter = NULL;
    const char* csv    = NULL;
    int         prof   = 0;
    double      secs   = 0.3;
    int         nthreads = 0;
    int         strip    = 32;
    int         tile     = 0;
    const char* only     = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc)
            csv = argv[++i];
        else if (!strcmp(argv[i], "--prof"))
            prof = 1;
        else if (!strcmp(argv[i], "--time") && i + 1 < argc)
            secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--strip") && i + 1 < argc)
            strip = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc)
            tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--column") && i + 1 < argc)
            only = argv[++i];
        else
            filter = argv[i];
    }

    bench_env e;
    e.fb     = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    e.c      = fm2d_create(e.fb);
    e.tex    = make_tex(256, 256);
    e.sprite = make_tex(64, 64);
    e.zb     = fm_surface_create(W, H, FM_FORMAT_D32F);
    e.c3     = fm3d_create();
    e.tex3   = fm3d_texture_create(e.tex, 1);
    {
        fm_surface* big = make_tex(4096, 4096);
        e.tex3_big      = fm3d_texture_create(big, 0);
        fm_surface_destroy(big);
    }
    {
        static fm3d_vertex cube[36];
        static const float P[8][3] = { { -1, -1, -1 }, { 1, -1, -1 }, { 1, 1, -1 }, { -1, 1, -1 },
                                       { -1, -1, 1 },  { 1, -1, 1 },  { 1, 1, 1 },  { -1, 1, 1 } };
        static const int   F[6][4] = { { 4, 5, 6, 7 }, { 1, 0, 3, 2 }, { 0, 4, 7, 3 },
                                       { 5, 1, 2, 6 }, { 7, 6, 2, 3 }, { 0, 1, 5, 4 } };
        const float        uv[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
        const int          tri[6]   = { 0, 1, 2, 0, 2, 3 };
        int                k        = 0;
        for (int f = 0; f < 6; f++)
            for (int i = 0; i < 6; i++) {
                const float* p = P[F[f][tri[i]]];
                cube[k++]      = bv(p[0] * 0.5f, p[1] * 0.5f, p[2] * 0.5f, uv[tri[i]][0], uv[tri[i]][1], 0xffffffffu);
            }
        e.cube     = cube;
#if FM_FEATURE_VBO
        e.cube_buf = fm3d_buffer_create(cube, 36, NULL, 0);
#endif
        enum { GN = 64 };
        static fm3d_vertex gv[(GN + 1) * (GN + 1)];
        static uint32_t    gi[GN * GN * 6];
        for (int z = 0; z <= GN; z++)
            for (int x = 0; x <= GN; x++)
                gv[z * (GN + 1) + x] = bv((float)x - GN / 2, -3, (float)z - GN / 2 - 10, (float)x * 0.25f,
                                          (float)z * 0.25f, 0xffffffffu);
        int n = 0;
        for (int z = 0; z < GN; z++)
            for (int x = 0; x < GN; x++) {
                uint32_t a = (uint32_t)(z * (GN + 1) + x), b = a + 1, c2 = a + GN + 1, d = c2 + 1;
                gi[n++] = a, gi[n++] = c2, gi[n++] = b, gi[n++] = b, gi[n++] = c2, gi[n++] = d;
            }
        e.grid = gv, e.gidx = gi, e.ngrid = (GN + 1) * (GN + 1), e.ngidx = n;
    }
    fm2d_set_strip_height(e.c, strip);
    if (tile > 0) fm3d_set_tile_size(e.c3, tile);

    fm_simd_level levels[FM_SIMD_LEVELS];
    int           nl = 0;
    for (int l = FM_SIMD_SCALAR; l < FM_SIMD_LEVELS; l++)
        if (fm_simd_supported((fm_simd_level)l)) levels[nl++] = (fm_simd_level)l;

    printf("fatmap %s bench, %dx%d, best=%s\n\n", fm_version_string(), W, H, fm_simd_name(fm_simd_best()));
    printf("%-24s", "workload (ms/iter)");
    for (int l = 0; l < nl; l++) printf(" %10s", fm_simd_name(levels[l]));

    fm_executor* ex = fm_executor_create(nthreads);
    char         mtname[32];
    snprintf(mtname, sizeof(mtname), "mt%d", ex->workers);
    printf(" %10s %10s %12s %8s\n", "cmdlist", mtname, "Mpix/s best", "speedup");

    FILE* cf = csv ? fopen(csv, "a") : NULL;
    for (size_t wi = 0; wi < sizeof(g_workloads) / sizeof(g_workloads[0]); wi++) {
        const workload* w = &g_workloads[wi];
        if (filter && !strstr(w->name, filter)) continue;
        printf("%-24s", w->name);
        double     best = 1e30, scalar = 0;
        fm3d_stats st3;
        memset(&st3, 0, sizeof(st3));
        /* SIMD levels (immediate), then deferred serial, then deferred threaded */
        for (int l = 0; l < nl + 2; l++) {
            int         mode = l < nl ? 0 : (l == nl ? 1 : 2);
            const char* name = mode == 0 ? fm_simd_name(levels[l]) : (mode == 1 ? "cmdlist" : mtname);
            if (only && strcmp(name, only) != 0 && !(mode == 2 && !strcmp(only, "mt"))) {
                printf(" %10s", "-");
                continue;
            }
            fm_simd_set(mode == 0 ? levels[l] : fm_simd_best());
            fm2d_reset(e.c);
            fm2d_set_deferred(e.c, mode != 0);
            fm2d_set_executor(e.c, mode == 2 ? ex : NULL);
            fm3d_set_deferred(e.c3, mode != 0);
            fm3d_set_executor(e.c3, mode == 2 ? ex : NULL);
            fm_surface_clear(e.fb, FM_RGB(10, 20, 30));
            w->run(&e); /* warm up */
            fm2d_flush(e.c);
            fm3d_flush(e.c3);
            fm_prof_reset();
            int      iters = 0;
            uint64_t t0 = fm_time_ns(), t1;
            do {
                fm3d_reset_stats(e.c3);
                w->run(&e);
                fm2d_flush(e.c);
                fm3d_flush(e.c3);
                fm_prof_frame();
                iters++;
                t1 = fm_time_ns();
            } while ((double)(t1 - t0) < secs * 1e9 || iters < 3);
            double ms = (double)(t1 - t0) / 1e6 / iters;
            printf(" %10.3f", ms);
            fflush(stdout);
            if (ms < best) best = ms;
            if (mode == 0 && levels[l] == FM_SIMD_SCALAR) scalar = ms;
            if (cf) fprintf(cf, "%s,%s,%.4f\n", w->name, name, ms);
            if (mode == 1) st3 = fm3d_get_stats(e.c3);
            if (prof && (only ? 1 : l == nl + 1)) {
                char buf[4096];
                fm_prof_report(buf, sizeof(buf));
                printf("\n%s", buf);
            }
        }
        fm2d_set_deferred(e.c, 0);
        fm2d_set_executor(e.c, NULL);
        fm3d_set_deferred(e.c3, 0);
        fm3d_set_executor(e.c3, NULL);
        printf(" %12.1f %7.2fx", w->pixels / (best * 1e3), scalar > 0 ? scalar / best : 1.0);
        if (st3.triangles_in) {
            printf("  tris %llu drawn %llu hiz-skipped %llu", (unsigned long long)st3.triangles_in,
                   (unsigned long long)st3.triangles_drawn, (unsigned long long)st3.hiz_rejected);
            if (st3.fragments_in)
                printf(" frag/px in %.2f shaded %.2f", (double)st3.fragments_in / (W * H),
                       (double)st3.fragments_shaded / (W * H));
        }
        printf("\n");
    }
    if (cf) fclose(cf);
    fm_simd_set(fm_simd_best());
    fm_surface_write_png(e.fb, "bench_last.png");
    fm2d_destroy(e.c);
    fm_surface_destroy(e.fb);
    fm_surface_destroy(e.tex);
    fm_surface_destroy(e.sprite);
    fm3d_texture_release(e.tex3);
    fm3d_texture_release(e.tex3_big);
#if FM_FEATURE_VBO
    fm3d_buffer_release(e.cube_buf);
#endif
    fm3d_destroy(e.c3);
    fm_surface_destroy(e.zb);
    fm_executor_destroy(ex);
    return 0;
}
