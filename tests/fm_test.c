/*
 * fatmap tests:
 *   1. every SIMD backend is bit-identical to the scalar reference
 *      (all blend ops, masks, tails, accumulation, bilinear, gradients)
 *   2. rendering correctness (coverage, fill rules, AA modes, strokes,
 *      blend math, gradients, clipping, images, hit testing, colors)
 *   3. full scenes rendered per SIMD level must match exactly; PNGs are
 *      written to the directory given as argv[1] for visual inspection
 */
#include <fatmap/fatmap.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_pass;
static const char* g_outdir = ".";

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (cond) {                                       \
            g_pass++;                                     \
        } else {                                          \
            g_fail++;                                     \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
        }                                                 \
    } while (0)

static uint32_t g_rng = 12345;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}


static uint32_t rnd_premul(void)
{
    uint32_t a = rnd() & 255;
    switch (rnd() & 7) {
    case 0: a = 0; break;
    case 1: a = 255; break;
    default: break;
    }
    uint32_t r = a ? rnd() % (a + 1) : 0, g = a ? rnd() % (a + 1) : 0, b = a ? rnd() % (a + 1) : 0;
    return (a << 24) | (r << 16) | (g << 8) | b;
}

static uint8_t rnd_cov(void)
{
    switch (rnd() & 3) {
    case 0: return 0;
    case 1: return 255;
    default: return (uint8_t)rnd();
    }
}

/* ---------------------------------------------------------------------------------------- */

#define NMAX 203

static void test_kernels_equivalence(void)
{
    static uint32_t src[NMAX], dst0[NMAX], ref[NMAX], out[NMAX];
    static uint8_t  cov[NMAX];
    fm_simd_level   levels[] = { FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_NEON };
    for (int li = 0; li < 3; li++) {
        fm_simd_level lv = levels[li];
        if (!fm_simd_supported(lv)) continue;
        int mism = 0;
        for (int op = 0; op < FM_OP_COUNT; op++) {
            for (int iter = 0; iter < 40; iter++) {
                int n = 1 + (int)(rnd() % NMAX);
                /* runs of equal coverage exercise the 0 / 255 fast paths */
                uint8_t c = rnd_cov();
                for (int i = 0; i < n; i++) {
                    src[i]  = rnd_premul();
                    dst0[i] = rnd_premul();
                    if ((rnd() & 7) == 0) c = rnd_cov();
                    cov[i] = c;
                }
                int use_cov = iter & 1;
                int solid   = (iter & 2) != 0;
                uint32_t s0 = src[0];
                memcpy(ref, dst0, (size_t)n * 4);
                memcpy(out, dst0, (size_t)n * 4);
                fm_simd_set(FM_SIMD_SCALAR);
                if (solid)
                    fm_blend_solid(ref, s0, use_cov ? cov : NULL, n, (fm_blend_op)op);
                else
                    fm_blend_span(ref, src, use_cov ? cov : NULL, n, (fm_blend_op)op);
                fm_simd_set(lv);
                if (solid)
                    fm_blend_solid(out, s0, use_cov ? cov : NULL, n, (fm_blend_op)op);
                else
                    fm_blend_span(out, src, use_cov ? cov : NULL, n, (fm_blend_op)op);
                if (memcmp(ref, out, (size_t)n * 4) != 0) {
                    if (mism < 5) {
                        for (int i = 0; i < n; i++)
                            if (ref[i] != out[i]) {
                                printf("  %s op=%s n=%d i=%d s=%08x d=%08x m=%d ref=%08x got=%08x\n",
                                       fm_simd_name(lv), fm_blend_op_name((fm_blend_op)op), n, i,
                                       solid ? s0 : src[i], dst0[i], use_cov ? cov[i] : 255, ref[i], out[i]);
                                break;
                            }
                    }
                    mism++;
                }
            }
        }
        CHECK(mism == 0, "%s blend kernels differ from scalar in %d cases", fm_simd_name(lv), mism);
        fm_simd_set(FM_SIMD_SCALAR);
    }
    fm_simd_set(fm_simd_best());
}

/* ---------------------------------------------------------------------------------------- */

static int count_diff(const fm_surface* a, const fm_surface* b, int* maxd)
{
    int n = 0;
    *maxd = 0;
    for (int y = 0; y < a->height; y++) {
        const uint32_t* ra = fm_surface_row32(a, y);
        const uint32_t* rb = fm_surface_row32(b, y);
        for (int x = 0; x < a->width; x++) {
            if (ra[x] != rb[x]) {
                n++;
                for (int s = 0; s < 32; s += 8) {
                    int d = abs((int)((ra[x] >> s) & 255) - (int)((rb[x] >> s) & 255));
                    if (d > *maxd) *maxd = d;
                }
            }
        }
    }
    return n;
}

static fm_surface* make_checker(int w, int h, int cell)
{
    fm_surface* s = fm_surface_create(w, h, FM_FORMAT_ARGB32);
    for (int y = 0; y < h; y++) {
        uint32_t* r = fm_surface_row32(s, y);
        for (int x = 0; x < w; x++) {
            int c = ((x / cell) + (y / cell)) & 1;
            r[x]  = c ? fm_premultiply(FM_RGBA(240, 180, 40, 255))
                      : fm_premultiply(FM_RGBA((uint8_t)(x * 255 / w), 40, (uint8_t)(y * 255 / h), 200));
        }
    }
    return s;
}

/* A scene touching most features */
static void draw_scene(fm2d_ctx* c, const fm_surface* tex, float t)
{
    fm2d_reset(c);
    fm2d_clear(c, FM_RGB(24, 26, 32));

    /* gradients */
    fm2d_paint* lg = fm2d_paint_linear(20, 20, 300, 180);
    fm2d_paint_add_stop(lg, 0.0f, FM_RGB(255, 60, 60));
    fm2d_paint_add_stop(lg, 0.5f, FM_RGB(255, 220, 60));
    fm2d_paint_add_stop(lg, 1.0f, FM_RGBA(60, 120, 255, 128));
    fm2d_set_fill_paint(c, lg);
    fm2d_fill_rect(c, 20, 20, 280, 160);
    fm2d_paint_release(lg);

    fm2d_paint* rg = fm2d_paint_radial(420, 100, 5, 440, 110, 90);
    fm2d_paint_add_stop(rg, 0.0f, FM_RGB(255, 255, 255));
    fm2d_paint_add_stop(rg, 1.0f, FM_RGBA(20, 160, 90, 0));
    fm2d_set_fill_paint(c, rg);
    fm2d_begin_path(c);
    fm2d_arc(c, 430, 100, 90, 0, 6.2831853f, 0);
    fm2d_fill(c, FM_FILL_NONZERO);
    fm2d_paint_release(rg);

    fm2d_paint* cg = fm2d_paint_conic(t, 620, 100);
    fm2d_paint_add_stop(cg, 0.0f, FM_RGB(255, 0, 0));
    fm2d_paint_add_stop(cg, 0.33f, FM_RGB(0, 255, 0));
    fm2d_paint_add_stop(cg, 0.66f, FM_RGB(0, 0, 255));
    fm2d_paint_add_stop(cg, 1.0f, FM_RGB(255, 0, 0));
    fm2d_set_fill_paint(c, cg);
    fm2d_begin_path(c);
    fm2d_arc(c, 620, 100, 80, 0, 6.2831853f, 0);
    fm2d_fill(c, FM_FILL_NONZERO);
    fm2d_paint_release(cg);

    /* strokes: joins, caps, dashes */
    const int joins[3] = { FM2D_JOIN_MITER, FM2D_JOIN_ROUND, FM2D_JOIN_BEVEL };
    const int caps[3]  = { FM2D_CAP_BUTT, FM2D_CAP_ROUND, FM2D_CAP_SQUARE };
    for (int i = 0; i < 3; i++) {
        fm2d_set_line_width(c, 14);
        fm2d_set_line_join(c, (fm2d_line_join)joins[i]);
        fm2d_set_line_cap(c, (fm2d_line_cap)caps[i]);
        fm2d_set_stroke_color(c, FM_RGBA(120, 200, 255, 220));
        fm2d_begin_path(c);
        float ox = 40.0f + (float)i * 150.0f;
        fm2d_move_to(c, ox, 330);
        fm2d_line_to(c, ox + 50, 230);
        fm2d_line_to(c, ox + 100, 330);
        fm2d_stroke(c);
    }
    float dash[2] = { 18, 8 };
    fm2d_set_line_dash(c, dash, 2);
    fm2d_set_line_dash_offset(c, t * 20.0f);
    fm2d_set_line_width(c, 4);
    fm2d_set_line_cap(c, FM2D_CAP_ROUND);
    fm2d_set_stroke_color(c, FM_RGB(255, 200, 80));
    fm2d_begin_path(c);
    fm2d_round_rect(c, 500, 220, 220, 120, (float[]){ 30 }, 1);
    fm2d_stroke(c);
    fm2d_set_line_dash(c, NULL, 0);

    /* beziers */
    fm2d_set_line_width(c, 3);
    fm2d_set_stroke_color(c, FM_RGB(255, 255, 255));
    fm2d_begin_path(c);
    fm2d_move_to(c, 760, 40);
    fm2d_bezier_to(c, 900, 0, 800, 300, 1000, 200);
    fm2d_quad_to(c, 1100, 60, 1240, 200);
    fm2d_stroke(c);

    /* transformed pattern fill (bilinear + repeat) */
    fm2d_paint* pat = fm2d_paint_pattern(tex, FM2D_REPEAT);
    fm2d_save(c);
    fm2d_translate(c, 900, 480);
    fm2d_rotate(c, t * 0.3f + 0.4f);
    fm2d_scale(c, 1.7f, 1.2f);
    fm2d_set_fill_paint(c, pat);
    fm2d_begin_path(c);
    fm2d_rect(c, -100, -80, 200, 160);
    fm2d_fill(c, FM_FILL_NONZERO);
    fm2d_restore(c);
    fm2d_paint_release(pat);

    /* even-odd star */
    fm2d_set_fill_color(c, FM_RGBA(255, 80, 200, 200));
    fm2d_begin_path(c);
    for (int i = 0; i < 5; i++) {
        float a = -1.5707963f + (float)i * 2.5132741f;
        float x = 150 + cosf(a) * 110, y = 530 + sinf(a) * 110;
        if (i == 0)
            fm2d_move_to(c, x, y);
        else
            fm2d_line_to(c, x, y);
    }
    fm2d_close_path(c);
    fm2d_fill(c, FM_FILL_EVENODD);

    /* clip + blend modes + images */
    fm2d_save(c);
    fm2d_begin_path(c);
    fm2d_ellipse(c, 450, 540, 170, 110, t * 0.2f, 0, 6.2831853f, 0);
    fm2d_clip(c, FM_FILL_NONZERO);
    fm2d_draw_image_scaled(c, tex, 280, 430, 340, 220);
    fm2d_set_composite_op(c, FM_OP_MULTIPLY);
    fm2d_set_fill_color(c, FM_RGBA(80, 200, 255, 255));
    fm2d_fill_rect(c, 300, 450, 150, 180);
    fm2d_set_composite_op(c, FM_OP_SCREEN);
    fm2d_set_fill_color(c, FM_RGBA(255, 60, 60, 200));
    fm2d_fill_rect(c, 450, 450, 150, 180);
    fm2d_restore(c);

    fm2d_set_global_alpha(c, 0.6f);
    fm2d_draw_image(c, tex, 1000, 30);
    fm2d_set_global_alpha(c, 1.0f);
    fm2d_set_image_smoothing(c, 0);
    fm2d_draw_image_sub(c, tex, 8, 8, 32, 32, 1100, 250, 128, 128);
    fm2d_set_image_smoothing(c, 1);

    /* hairlines */
    fm2d_set_line_width(c, 1);
    for (int i = 0; i < 40; i++) {
        float a = (float)i * 0.157f + t;
        fm2d_set_stroke_color(c, FM_RGBA(255, 255, 255, 160));
        fm2d_begin_path(c);
        fm2d_move_to(c, 1180, 560);
        fm2d_line_to(c, 1180 + cosf(a) * 90, 560 + sinf(a) * 90);
        fm2d_stroke(c);
    }
}

static void test_scene_equivalence(void)
{
    fm_surface* tex = make_checker(96, 96, 12);
    fm_surface* ref = fm_surface_create(1280, 720, FM_FORMAT_ARGB32);
    fm_surface* out = fm_surface_create(1280, 720, FM_FORMAT_ARGB32);
    fm2d_ctx*   c   = fm2d_create(ref);
    char        path[512];

    fm_simd_set(FM_SIMD_SCALAR);
    draw_scene(c, tex, 0.7f);
    snprintf(path, sizeof(path), "%s/scene_scalar.png", g_outdir);
    CHECK(fm_surface_write_png(ref, path), "write %s", path);

    fm_simd_level levels[] = { FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_NEON };
    for (int i = 0; i < 3; i++) {
        if (!fm_simd_supported(levels[i])) continue;
        fm_simd_set(levels[i]);
        fm2d_set_target(c, out);
        draw_scene(c, tex, 0.7f);
        int maxd, n = count_diff(ref, out, &maxd);
        CHECK(n == 0, "scene %s vs scalar: %d pixels differ (max channel diff %d)", fm_simd_name(levels[i]), n,
              maxd);
    }
    /* deferred command lists: serial and threaded, several strip heights */
    fm_simd_set(fm_simd_best());
    struct {
        int threads, strip;
    } modes[] = { { 0, 32 }, { 2, 16 }, { 4, 32 }, { 8, 64 }, { -1, 32 } };
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        fm_executor* ex = modes[i].threads ? fm_executor_create(modes[i].threads) : NULL;
        fm2d_set_target(c, out);
        fm2d_set_deferred(c, 1);
        fm2d_set_executor(c, ex);
        fm2d_set_strip_height(c, modes[i].strip);
        for (int frame = 0; frame < 3; frame++) { /* repeat to exercise list reuse */
            draw_scene(c, tex, 0.7f);
            fm2d_flush(c);
        }
        int maxd, n = count_diff(ref, out, &maxd);
        CHECK(n == 0, "deferred (threads=%d workers=%d strip=%d): %d pixels differ (max %d)", modes[i].threads,
              ex ? ex->workers : 1, modes[i].strip, n, maxd);
        fm2d_set_deferred(c, 0);
        fm2d_set_executor(c, NULL);
        fm_executor_destroy(ex);
    }
    fm2d_set_strip_height(c, 32);

    /* aliased (GL rule) rendering of the same scene */
    fm_simd_set(fm_simd_best());
    fm2d_set_target(c, out);
    fm2d_set_antialias(c, FM_AA_NONE);
    draw_scene(c, tex, 0.7f);
    snprintf(path, sizeof(path), "%s/scene_aliased.png", g_outdir);
    fm_surface_write_png(out, path);
    fm2d_set_antialias(c, FM_AA_ANALYTIC);

    fm2d_destroy(c);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(tex);
}

/* ---------------------------------------------------------------------------------------- */

static double alpha_sum(const fm_surface* s)
{
    double sum = 0;
    for (int y = 0; y < s->height; y++) {
        const uint32_t* r = fm_surface_row32(s, y);
        for (int x = 0; x < s->width; x++) sum += (double)(r[x] >> 24) / 255.0;
    }
    return sum;
}

static void test_coverage(void)
{
    fm_surface* s = fm_surface_create(200, 200, FM_FORMAT_ARGB32);
    fm2d_ctx*   c = fm2d_create(s);

    /* aligned rect: exact */
    fm2d_set_fill_color(c, FM_RGB(255, 0, 0));
    fm2d_fill_rect(c, 10, 10, 10, 10);
    CHECK(fm_surface_get_pixel(s, 10, 10) == FM_RGB(255, 0, 0), "rect pixel");
    CHECK(fm_surface_get_pixel(s, 19, 19) == FM_RGB(255, 0, 0), "rect pixel br");
    CHECK(fm_surface_get_pixel(s, 20, 20) == 0, "outside rect");
    CHECK(fabs(alpha_sum(s) - 100.0) < 1e-9, "rect area %f", alpha_sum(s));

    /* half pixel offset: 4 pixels at 25% */
    fm_surface_clear(s, 0);
    fm2d_set_fill_color(c, FM_RGB(255, 255, 255));
    fm2d_fill_rect(c, 50.5f, 50.5f, 1, 1);
    uint32_t a = fm_surface_get_pixel(s, 50, 50) >> 24;
    CHECK(a >= 63 && a <= 65, "quarter coverage alpha %u", a);
    CHECK(fabs(alpha_sum(s) - 1.0) < 0.02, "sub-pixel rect area %f", alpha_sum(s));

    /* circle area */
    fm_surface_clear(s, 0);
    fm2d_begin_path(c);
    fm2d_arc(c, 100.3f, 99.7f, 60, 0, 6.2831853f, 0);
    fm2d_fill(c, FM_FILL_NONZERO);
    double area = alpha_sum(s), expect = 3.14159265 * 3600.0;
    CHECK(fabs(area - expect) / expect < 0.002, "circle area %f expected %f", area, expect);

    /* fill rules: two same-direction nested squares */
    for (int rule = 0; rule < 2; rule++) {
        fm_surface_clear(s, 0);
        fm2d_begin_path(c);
        fm2d_rect(c, 20, 20, 100, 100);
        fm2d_rect(c, 50, 50, 40, 40);
        fm2d_fill(c, (fm_fill_rule)rule);
        uint32_t center = fm_surface_get_pixel(s, 70, 70) >> 24, ring = fm_surface_get_pixel(s, 30, 30) >> 24;
        CHECK(ring == 255, "ring filled (rule %d)", rule);
        CHECK(center == (rule == FM_FILL_NONZERO ? 255u : 0u), "center rule %d alpha %u", rule, center);
    }

    /* aliased: pixel centers + top-left rule */
    fm2d_set_antialias(c, FM_AA_NONE);
    fm_surface_clear(s, 0);
    fm2d_begin_path(c);
    fm2d_rect(c, 10.5f, 10.5f, 5, 5); /* centers 10.5 .. 14.5 inside, 15.5 excluded */
    fm2d_fill(c, FM_FILL_NONZERO);
    CHECK(fm_surface_get_pixel(s, 10, 10) >> 24 == 255, "aliased tl in");
    CHECK(fm_surface_get_pixel(s, 14, 14) >> 24 == 255, "aliased br in");
    CHECK(fm_surface_get_pixel(s, 15, 15) >> 24 == 0, "aliased br out");
    CHECK(fm_surface_get_pixel(s, 9, 9) >> 24 == 0, "aliased tl out");
    CHECK(fabs(alpha_sum(s) - 25.0) < 1e-9, "aliased area %f", alpha_sum(s));
    fm_surface_clear(s, 0);
    fm2d_begin_path(c);
    fm2d_arc(c, 100, 100, 60, 0, 6.2831853f, 0);
    fm2d_fill(c, FM_FILL_NONZERO);
    area = alpha_sum(s);
    CHECK(fabs(area - 3.14159265 * 3600.0) / (3.14159265 * 3600.0) < 0.01, "aliased circle area %f", area);
    fm2d_set_antialias(c, FM_AA_ANALYTIC);

    /* stroke: 2px horizontal line centered on y=10 covers rows 9,10 */
    fm_surface_clear(s, 0);
    fm2d_set_line_width(c, 2);
    fm2d_set_stroke_color(c, FM_RGB(0, 255, 0));
    fm2d_begin_path(c);
    fm2d_move_to(c, 10, 10);
    fm2d_line_to(c, 110, 10);
    fm2d_stroke(c);
    CHECK(fm_surface_get_pixel(s, 50, 9) == FM_RGB(0, 255, 0), "stroke row 9");
    CHECK(fm_surface_get_pixel(s, 50, 10) == FM_RGB(0, 255, 0), "stroke row 10");
    CHECK(fm_surface_get_pixel(s, 50, 11) == 0, "stroke row 11 empty");
    CHECK(fabs(alpha_sum(s) - 200.0) < 0.05, "stroke area %f", alpha_sum(s));

    /* square cap extends by half width, round cap adds half circles */
    fm_surface_clear(s, 0);
    fm2d_set_line_width(c, 10);
    fm2d_set_line_cap(c, FM2D_CAP_SQUARE);
    fm2d_begin_path(c);
    fm2d_move_to(c, 50, 50);
    fm2d_line_to(c, 150, 50);
    fm2d_stroke(c);
    CHECK(fabs(alpha_sum(s) - 1100.0) < 0.5, "square cap area %f", alpha_sum(s));
    fm_surface_clear(s, 0);
    fm2d_set_line_cap(c, FM2D_CAP_ROUND);
    fm2d_stroke(c);
    expect = 1000.0 + 3.14159265 * 25.0;
    CHECK(fabs(alpha_sum(s) - expect) / expect < 0.003, "round cap area %f expected %f", alpha_sum(s), expect);
    fm2d_set_line_cap(c, FM2D_CAP_BUTT);

    /* closed stroke of a square: ring area = outer - inner */
    fm_surface_clear(s, 0);
    fm2d_begin_path(c);
    fm2d_rect(c, 40, 40, 100, 100);
    fm2d_stroke(c);
    CHECK(fabs(alpha_sum(s) - (110.0 * 110.0 - 90.0 * 90.0)) < 1.0, "square ring area %f", alpha_sum(s));

    /* dashes: 10 on / 10 off over 100px -> half the area */
    fm_surface_clear(s, 0);
    fm2d_set_line_width(c, 4);
    float d[2] = { 10, 10 };
    fm2d_set_line_dash(c, d, 2);
    fm2d_begin_path(c);
    fm2d_move_to(c, 20, 100);
    fm2d_line_to(c, 120, 100);
    fm2d_stroke(c);
    CHECK(fabs(alpha_sum(s) - 200.0) < 0.5, "dash area %f", alpha_sum(s));
    fm2d_set_line_dash(c, NULL, 0);

    fm2d_destroy(c);
    fm_surface_destroy(s);
}

static void test_blend_math(void)
{
    fm_surface* s = fm_surface_create(8, 8, FM_FORMAT_ARGB32);
    fm2d_ctx*   c = fm2d_create(s);
    struct {
        fm_blend_op op;
        fm_color    dst, src, expect;
    } cases[] = {
        { FM_OP_SRC_OVER, FM_RGB(255, 255, 255), FM_RGBA(255, 0, 0, 128), FM_RGB(255, 127, 127) },
        { FM_OP_MULTIPLY, FM_RGB(255, 255, 255), FM_RGB(10, 100, 200), FM_RGB(10, 100, 200) },
        { FM_OP_MULTIPLY, FM_RGB(128, 128, 128), FM_RGB(128, 255, 0), FM_RGB(64, 128, 0) },
        { FM_OP_SCREEN, FM_RGB(0, 0, 0), FM_RGB(10, 100, 200), FM_RGB(10, 100, 200) },
        { FM_OP_DARKEN, FM_RGB(100, 50, 200), FM_RGB(50, 100, 150), FM_RGB(50, 50, 150) },
        { FM_OP_LIGHTEN, FM_RGB(100, 50, 200), FM_RGB(50, 100, 150), FM_RGB(100, 100, 200) },
        { FM_OP_DIFFERENCE, FM_RGB(100, 50, 200), FM_RGB(50, 100, 150), FM_RGB(50, 50, 50) },
        { FM_OP_EXCLUSION, FM_RGB(255, 0, 255), FM_RGB(255, 255, 0), FM_RGB(0, 255, 255) },
        { FM_OP_COPY, FM_RGB(1, 2, 3), FM_RGBA(255, 0, 0, 0), 0 },
        { FM_OP_DST_OUT, FM_RGB(9, 9, 9), FM_RGB(0, 0, 0), 0 },
        { FM_OP_LIGHTER, FM_RGB(200, 100, 0), FM_RGB(100, 100, 100), FM_RGB(255, 200, 100) },
        { FM_OP_OVERLAY, FM_RGB(0, 255, 128), FM_RGB(200, 10, 255), FM_RGB(0, 255, 255) },
        { FM_OP_LUMINOSITY, FM_RGB(255, 0, 0), FM_RGB(255, 255, 255), FM_RGB(255, 255, 255) },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        fm_surface_clear(s, cases[i].dst);
        fm2d_set_composite_op(c, cases[i].op);
        fm2d_set_fill_color(c, cases[i].src);
        fm2d_fill_rect(c, 0, 0, 8, 8);
        fm_color got = fm_surface_get_pixel(s, 3, 3);
        int      ok  = 1;
        for (int sh = 0; sh < 32; sh += 8)
            if (abs((int)((got >> sh) & 255) - (int)((cases[i].expect >> sh) & 255)) > 1) ok = 0;
        CHECK(ok, "blend %s: dst %08x src %08x -> %08x expected %08x", fm_blend_op_name(cases[i].op), cases[i].dst,
              cases[i].src, got, cases[i].expect);
    }
    fm2d_destroy(c);
    fm_surface_destroy(s);
}

static void test_paints_clip_images(void)
{
    fm_surface* s = fm_surface_create(256, 64, FM_FORMAT_ARGB32);
    fm2d_ctx*   c = fm2d_create(s);

    /* linear gradient endpoints and midpoint */
    fm2d_paint* g = fm2d_paint_linear(0, 0, 256, 0);
    fm2d_paint_add_stop(g, 0, FM_RGB(0, 0, 0));
    fm2d_paint_add_stop(g, 1, FM_RGB(255, 255, 255));
    fm2d_set_fill_paint(c, g);
    fm2d_fill_rect(c, 0, 0, 256, 64);
    uint32_t l = fm_surface_get_pixel(s, 0, 5) & 255, m = fm_surface_get_pixel(s, 128, 5) & 255,
             r = fm_surface_get_pixel(s, 255, 5) & 255;
    CHECK(l <= 1 && r >= 254 && abs((int)m - 128) <= 2, "linear gradient %u %u %u", l, m, r);
    fm2d_paint_release(g);

    /* hard stop */
    g = fm2d_paint_linear(0, 0, 256, 0);
    fm2d_paint_add_stop(g, 0.5f, FM_RGB(255, 0, 0));
    fm2d_paint_add_stop(g, 0.5f, FM_RGB(0, 0, 255));
    fm2d_set_fill_paint(c, g);
    fm2d_fill_rect(c, 0, 0, 256, 64);
    CHECK(fm_surface_get_pixel(s, 100, 5) == FM_RGB(255, 0, 0), "hard stop left");
    CHECK(fm_surface_get_pixel(s, 156, 5) == FM_RGB(0, 0, 255), "hard stop right");
    fm2d_paint_release(g);

    /* clip */
    fm_surface_clear(s, 0);
    fm2d_save(c);
    fm2d_begin_path(c);
    fm2d_rect(c, 10, 10, 20, 20);
    fm2d_clip(c, FM_FILL_NONZERO);
    fm2d_set_fill_color(c, FM_RGB(255, 255, 0));
    fm2d_fill_rect(c, 0, 0, 256, 64);
    fm2d_restore(c);
    CHECK(fabs(alpha_sum(s) - 400.0) < 1e-6, "rect clip area %f", alpha_sum(s));
    fm_surface_clear(s, 0);
    fm2d_save(c);
    fm2d_begin_path(c);
    fm2d_arc(c, 128, 32, 20, 0, 6.2831853f, 0);
    fm2d_clip(c, FM_FILL_NONZERO);
    fm2d_fill_rect(c, 0, 0, 256, 64);
    fm2d_restore(c);
    double expect = 3.14159265 * 400.0;
    CHECK(fabs(alpha_sum(s) - expect) / expect < 0.008, "circle clip area %f", alpha_sum(s));
    fm_surface_clear(s, 0);
    fm2d_fill_rect(c, 0, 0, 256, 64);
    CHECK(fabs(alpha_sum(s) - 256.0 * 64.0) < 1e-6, "clip restored");

    /* drawImage 1:1 is an exact copy */
    fm_surface* img = make_checker(40, 30, 5);
    fm_surface_clear(s, 0);
    fm2d_draw_image(c, img, 7, 9);
    int same = 1;
    for (int y = 0; y < 30; y++)
        for (int x = 0; x < 40; x++)
            if (fm_surface_row32(s, y + 9)[x + 7] != fm_surface_row32(img, y)[x]) same = 0;
    CHECK(same, "drawImage 1:1 copy");
    /* 2x nearest upscale */
    fm2d_set_image_smoothing(c, 0);
    fm_surface_clear(s, 0);
    fm2d_draw_image_scaled(c, img, 0, 0, 80, 60);
    CHECK(fm_surface_row32(s, 11)[13] == fm_surface_row32(img, 5)[6], "nearest 2x upscale");
    fm2d_set_image_smoothing(c, 1);
    fm_surface_destroy(img);

    /* hit testing */
    fm2d_begin_path(c);
    fm2d_arc(c, 50, 30, 20, 0, 6.2831853f, 0);
    CHECK(fm2d_is_point_in_path(c, 50, 30, FM_FILL_NONZERO), "point in path");
    CHECK(!fm2d_is_point_in_path(c, 80, 30, FM_FILL_NONZERO), "point outside path");
    fm2d_set_line_width(c, 6);
    CHECK(fm2d_is_point_in_stroke(c, 71, 30), "point in stroke");
    CHECK(!fm2d_is_point_in_stroke(c, 50, 30), "center not in stroke");

    fm2d_destroy(c);
    fm_surface_destroy(s);
}

static void test_colors(void)
{
    fm_color c = 0;
    CHECK(fm_color_parse("#f80", &c) && c == FM_RGB(255, 136, 0), "#f80 -> %08x", c);
    CHECK(fm_color_parse("#11223344", &c) && c == FM_RGBA(0x11, 0x22, 0x33, 0x44), "#rrggbbaa -> %08x", c);
    CHECK(fm_color_parse("rgba(10, 20, 30, 0.5)", &c) && c == FM_RGBA(10, 20, 30, 128), "rgba() -> %08x", c);
    CHECK(fm_color_parse("rgb(100% 0% 0% / 25%)", &c) && c == FM_RGBA(255, 0, 0, 64), "rgb space syntax -> %08x", c);
    CHECK(fm_color_parse("hsl(120, 100%, 50%)", &c) && c == FM_RGB(0, 255, 0), "hsl -> %08x", c);
    CHECK(fm_color_parse("RebeccaPurple", &c) && c == FM_RGB(0x66, 0x33, 0x99), "named -> %08x", c);
    CHECK(!fm_color_parse("notacolor", &c), "reject unknown");
    CHECK(fm_blend_op_from_name("soft-light") == FM_OP_SOFT_LIGHT, "op name lookup");
    CHECK(fm_unpremultiply(fm_premultiply(FM_RGBA(200, 100, 50, 255))) == FM_RGBA(200, 100, 50, 255),
          "premul roundtrip opaque");
}

static void test_math(void)
{
    fm_mat4 p   = fm_perspective(fm_radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    fm_mat4 v   = fm_lookat(fm_v3(0, 2, 5), fm_v3(0, 0, 0), fm_v3(0, 1, 0));
    fm_mat4 m   = fm_rotate(fm_translate(fm_mat4_identity(), fm_v3(1, 2, 3)), 0.5f, fm_v3(0, 1, 0));
    fm_mat4 mvp = fm_mat4_mul(p, fm_mat4_mul(v, m));
    fm_mat4 inv = fm_mat4_inverse(mvp);
    fm_mat4 id  = fm_mat4_mul(mvp, inv);
    float   err = 0;
    const float* f = fm_mat4_ptr(&id);
    for (int i = 0; i < 16; i++) err = fmaxf(err, fabsf(f[i] - ((i % 5) == 0 ? 1.0f : 0.0f)));
    CHECK(err < 1e-4f, "mat4 inverse error %g", err);
    fm_vec4 o = fm_mat4_mul_vec4(fm_translate(fm_mat4_identity(), fm_v3(1, 2, 3)), fm_v4(0, 0, 0, 1));
    CHECK(o.x == 1 && o.y == 2 && o.z == 3 && o.w == 1, "translate");
}

int main(int argc, char** argv)
{
    if (argc > 1) g_outdir = argv[1];
    printf("fatmap %s, cpu features 0x%x, best SIMD: %s\n", fm_version_string(), fm_cpu_features(),
           fm_simd_name(fm_simd_best()));
    test_kernels_equivalence();
    test_coverage();
    test_blend_math();
    test_paints_clip_images();
    test_colors();
    test_math();
    test_scene_equivalence();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
