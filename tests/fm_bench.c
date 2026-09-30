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
    fm3d_buffer*  cube_buf; /* the same cube as a vertex buffer */
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
    fm3d_sampler s = { FM3D_FILTER_BILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0 };
    fm3d_set_texture(e->c3, e->tex3, &s);
    g_rng = 11;
    for (int i = 0; i < 2000; i++) {
        fm_mat4 m = fm_translate(fm_mat4_identity(), fm_v3((rndf() - 0.5f) * 40, (rndf() - 0.5f) * 14, -rndf() * 40));
        m         = fm_rotate(m, rndf() * 6.28f, fm_v3(rndf(), 1, rndf()));
        fm3d_set_model(e->c3, &m);
        if (vbo)
            fm3d_draw_buffer(e->c3, e->cube_buf, 0, 36);
        else
            fm3d_draw(e->c3, e->cube, 36);
    }
}

static void w3_cubes(bench_env* e)
{
    cam3d(e);
    w3_cubes_body(e, 0);
}
static void w3_cubes_vbo(bench_env* e)
{
    cam3d(e);
    w3_cubes_body(e, 1);
}

static void w3_floor(bench_env* e, fm3d_filter f)
{
    cam3d(e);
    fm3d_sampler s = { f, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0 };
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
    fm3d_sampler s = { FM3D_FILTER_BILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0 };
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
    { "3d_cubes_2000_vbo", W * H, w3_cubes_vbo },
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
        e.cube_buf = fm3d_buffer_create(cube, 36, NULL, 0);
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

    fm_simd_level levels[4];
    int           nl = 0;
    for (int l = FM_SIMD_SCALAR; l <= FM_SIMD_NEON; l++)
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
    fm3d_buffer_release(e.cube_buf);
    fm3d_destroy(e.c3);
    fm_surface_destroy(e.zb);
    fm_executor_destroy(ex);
    return 0;
}
