/*
 * fatmap 3D tests: fill convention, coverage, depth, culling, clipping,
 * perspective correct texturing (checked against ray casting), mipmapping,
 * and bit-exact equality of immediate vs tiled multithreaded rendering
 * across tile sizes, thread counts and SIMD levels.
 */
#include <fatmap/fatmap.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int         g_fail, g_pass;
static const char* g_outdir = ".";

#define CHECK(cond, ...)                                \
    do {                                                \
        if (cond) {                                     \
            g_pass++;                                   \
        } else {                                        \
            g_fail++;                                   \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
        }                                               \
    } while (0)

#define W 320
#define H 240

static fm3d_vertex vtx(float x, float y, float z, float u, float v, fm_color c)
{
    fm3d_vertex r;
    memset(&r, 0, sizeof(r));
    r.x     = x;
    r.y     = y;
    r.z     = z;
    r.u     = u;
    r.v     = v;
    r.color = c;
    r.ny    = 1.0f;
    return r;
}

/* ortho projection with vertex coordinates = pixel coordinates (y down) */
static void pixel_space(fm3d_ctx* c)
{
    fm_mat4 p = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
    fm_mat4 i = fm_mat4_identity();
    fm3d_set_projection(c, &p);
    fm3d_set_view(c, &i);
    fm3d_set_model(c, &i);
}

static int count_nonzero(const fm_surface* s)
{
    int n = 0;
    for (int y = 0; y < s->height; y++)
        for (int x = 0; x < s->width; x++) n += fm_surface_row32(s, y)[x] != 0;
    return n;
}

/* point strictly inside convex polygon (counter clockwise or clockwise), with margin */
static int inside_convex(const float* px, const float* py, int n, float x, float y, float margin)
{
    int sign = 0;
    for (int i = 0; i < n; i++) {
        int   j  = (i + 1) % n;
        float ex = px[j] - px[i], ey = py[j] - py[i];
        float cr = ex * (y - py[i]) - ey * (x - px[i]);
        float d  = cr / sqrtf(ex * ex + ey * ey);
        if (fabsf(d) < margin) return -1; /* on the boundary: undecided */
        int s = d > 0 ? 1 : -1;
        if (sign == 0) sign = s;
        if (s != sign) return 0;
    }
    return 1;
}

static void test_fill_convention(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    pixel_space(c);
    fm3d_set_blend(c, FM_OP_LIGHTER);
    /* fan of 13 triangles around a fractional center: every shared edge must
     * be owned by exactly one triangle */
    enum { N = 13 };
    float       px[N], py[N], cx = 160.37f, cy = 119.61f;
    fm3d_vertex tris[N * 3];
    for (int i = 0; i < N; i++) {
        float a = (float)i / N * 6.2831853f + 0.1f;
        px[i]   = cx + cosf(a) * 97.3f;
        py[i]   = cy + sinf(a) * 88.1f;
    }
    for (int i = 0; i < N; i++) {
        int j           = (i + 1) % N;
        tris[3 * i]     = vtx(cx, cy, 0, 0, 0, FM_RGBA(255, 255, 255, 10));
        tris[3 * i + 1] = vtx(px[i], py[i], 0, 0, 0, FM_RGBA(255, 255, 255, 10));
        tris[3 * i + 2] = vtx(px[j], py[j], 0, 0, 0, FM_RGBA(255, 255, 255, 10));
    }
    fm3d_draw(c, tris, N * 3);
    int bad_in = 0, bad_out = 0, inside = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint32_t p  = fm_surface_row32(fb, y)[x];
            int      in = inside_convex(px, py, N, (float)x + 0.5f, (float)y + 0.5f, 0.07f);
            if (in == 1) {
                inside++;
                if (p != 0x0a0a0a0au) bad_in++;
            } else if (in == 0 && p != 0) {
                bad_out++;
            }
        }
    CHECK(inside > 20000, "fan covers pixels (%d)", inside);
    CHECK(bad_in == 0, "fan interior: %d pixels not covered exactly once", bad_in);
    CHECK(bad_out == 0, "fan exterior: %d pixels drawn outside", bad_out);

    /* pixel aligned quad covers exactly its pixel centers */
    fm_surface_clear(fb, 0);
    fm3d_set_blend(c, FM_OP_SRC_OVER);
    fm_color    wc = FM_RGB(255, 255, 255);
    fm3d_vertex q[6] = { vtx(10, 10, 0, 0, 0, wc), vtx(20, 10, 0, 0, 0, wc), vtx(20, 20, 0, 0, 0, wc),
                         vtx(10, 10, 0, 0, 0, wc), vtx(20, 20, 0, 0, 0, wc), vtx(10, 20, 0, 0, 0, wc) };
    fm3d_draw(c, q, 6);
    CHECK(count_nonzero(fb) == 100, "10x10 quad covers %d pixels", count_nonzero(fb));
    CHECK(fm_surface_get_pixel(fb, 10, 10) == wc && fm_surface_get_pixel(fb, 19, 19) == wc, "quad corners");
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

static void test_depth_cull_clip(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, zb);
    pixel_space(c);
    fm3d_clear_color(c, 0);
    fm3d_clear_depth(c, 1.0f);
    fm_color    g = FM_RGB(0, 255, 0), r = FM_RGB(255, 0, 0);
    fm3d_vertex near_q[3] = { vtx(50, 50, 0.5f, 0, 0, g), vtx(150, 50, 0.5f, 0, 0, g), vtx(50, 150, 0.5f, 0, 0, g) };
    fm3d_vertex far_q[3]  = { vtx(40, 40, -0.5f, 0, 0, r), vtx(160, 40, -0.5f, 0, 0, r), vtx(40, 160, -0.5f, 0, 0, r) };
    fm3d_draw(c, near_q, 3);
    fm3d_draw(c, far_q, 3);
    CHECK(fm_surface_get_pixel(fb, 60, 60) == g, "depth: near stays in front (%08x)", fm_surface_get_pixel(fb, 60, 60));
    CHECK(fm_surface_get_pixel(fb, 45, 45) == r, "depth: far visible outside near");
    CHECK(fabsf(fm_surface_rowf(zb, 60)[60] - 0.25f) < 1e-5f, "depth value %f", fm_surface_rowf(zb, 60)[60]);

    /* culling: winding is what the viewer sees; cw_screen is visually clockwise
     * (a back face with CCW fronts), ccw_screen is visually counter clockwise */
    fm_surface_clear(fb, 0);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_set_cull(c, FM3D_CULL_BACK, FM3D_FRONT_CCW);
    fm3d_vertex cw_screen[3]  = { vtx(10, 10, 0, 0, 0, g), vtx(60, 10, 0, 0, 0, g), vtx(10, 60, 0, 0, 0, g) };
    fm3d_vertex ccw_screen[3] = { vtx(100, 10, 0, 0, 0, g), vtx(100, 60, 0, 0, 0, g), vtx(150, 10, 0, 0, 0, g) };
    fm3d_draw(c, cw_screen, 3);
    fm3d_draw(c, ccw_screen, 3);
    CHECK(fm_surface_get_pixel(fb, 15, 15) == 0, "back face (clockwise) culled");
    CHECK(fm_surface_get_pixel(fb, 105, 15) == g, "front face (counter clockwise) drawn");
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);

    /* near plane clipping with a perspective camera */
    fm_mat4 proj = fm_perspective(fm_radians(70), (float)W / H, 0.5f, 100.0f);
    fm_mat4 view = fm_lookat(fm_v3(0, 1, 0), fm_v3(0, 1, -1), fm_v3(0, 1, 0));
    fm_mat4 id   = fm_mat4_identity();
    fm3d_set_projection(c, &proj);
    fm3d_set_view(c, &view);
    fm3d_set_model(c, &id);
    fm_surface_clear(fb, 0);
    fm3d_reset_stats(c);
    fm3d_vertex cross[3] = { vtx(-5, 0, 5, 0, 0, g), vtx(5, 0, 5, 0, 0, g), vtx(0, 0, -20, 0, 0, g) };
    fm3d_draw(c, cross, 3);
    fm3d_stats st = fm3d_get_stats(c);
    CHECK(st.triangles_clipped == 1 && count_nonzero(fb) > 1000, "near clipped triangle drawn (%d px)", count_nonzero(fb));
    fm_surface_clear(fb, 0);
    fm3d_vertex behind[3] = { vtx(-5, 0, 5, 0, 0, g), vtx(5, 0, 5, 0, 0, g), vtx(0, 0, 8, 0, 0, g) };
    fm3d_draw(c, behind, 3);
    CHECK(count_nonzero(fb) == 0, "triangle behind the camera rejected");

    fm3d_destroy(c);
    fm_surface_destroy(fb);
    fm_surface_destroy(zb);
}

/* FM3D_ORIGIN_LOWER_LEFT: GL window coordinates, row 0 at the bottom */
static void test_origin(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    fm3d_set_origin(c, FM3D_ORIGIN_LOWER_LEFT);
    fm_mat4 p = fm_ortho(0, (float)W, 0, (float)H, -1, 1), i = fm_mat4_identity(); /* y up: vertex y = row */
    fm3d_set_projection(c, &p);
    fm3d_set_view(c, &i);
    fm3d_set_model(c, &i);
    fm3d_clear_color(c, 0);
    fm3d_set_cull(c, FM3D_CULL_BACK, FM3D_FRONT_CCW);
    fm_color    g = FM_RGB(0, 255, 0);
    fm3d_vertex ccw[3] = { vtx(10, 10, 0, 0, 0, g), vtx(60, 10, 0, 0, 0, g), vtx(10, 60, 0, 0, 0, g) };
    fm3d_vertex cw[3]  = { vtx(100, 10, 0, 0, 0, g), vtx(100, 60, 0, 0, 0, g), vtx(150, 10, 0, 0, 0, g) };
    fm3d_draw(c, ccw, 3);
    fm3d_draw(c, cw, 3);
    fm3d_flush(c);
    CHECK(fm_surface_get_pixel(fb, 15, 15) == g, "lower left: counter clockwise (y up) front drawn at row 15");
    CHECK(fm_surface_get_pixel(fb, 15, H - 16) == 0, "lower left: nothing at the top");
    CHECK(fm_surface_get_pixel(fb, 105, 15) == 0, "lower left: clockwise back face culled");
    /* the viewport counts rows bottom up as well */
    fm3d_clear_color(c, 0);
    fm3d_set_viewport(c, 0, 0, W / 2, H / 2);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_vertex full[6] = { vtx(0, 0, 0, 0, 0, g), vtx((float)W, 0, 0, 0, 0, g), vtx(0, (float)H, 0, 0, 0, g),
                            vtx((float)W, 0, 0, 0, 0, g), vtx((float)W, (float)H, 0, 0, 0, g), vtx(0, (float)H, 0, 0, 0, g) };
    fm3d_draw(c, full, 6);
    fm3d_flush(c);
    CHECK(count_nonzero(fb) == W / 2 * (H / 2) && fm_surface_get_pixel(fb, 1, 1) == g, "lower left viewport covers rows 0 .. h/2 (%d px)",
          count_nonzero(fb));
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

/* fm3d_set_blend_state: straight colors, GL blend factors */
static void test_blend_state(void)
{
    /* raw words: fm_surface_get_pixel would un-premultiply */
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    pixel_space(c);
    fm3d_blend_state bs = { FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
    fm3d_set_blend_state(c, &bs);
    fm3d_clear_color(c, FM_RGBA(0, 0, 255, 128)); /* stored as given, not premultiplied */
    fm3d_flush(c);
    CHECK(fm_surface_row32(fb, 5)[5] == FM_RGBA(0, 0, 255, 128), "straight clear (%08x)", fm_surface_row32(fb, 5)[5]);
    bs.src_rgb = FM3D_BF_SRC_ALPHA, bs.dst_rgb = FM3D_BF_ONE_MINUS_SRC_ALPHA; /* glBlendFuncSeparate(SA, 1-SA, 1, 1-SA) */
    bs.src_alpha = FM3D_BF_ONE, bs.dst_alpha = FM3D_BF_ONE_MINUS_SRC_ALPHA;
    fm3d_set_blend_state(c, &bs);
    fm_color    h = FM_RGBA(255, 0, 0, 128);
    fm3d_vertex tri[3] = { vtx(0, 0, 0, 0, 0, h), vtx(100, 0, 0, 0, 0, h), vtx(0, 100, 0, 0, 0, h) };
    fm3d_draw(c, tri, 3);
    fm3d_flush(c);
    uint32_t p = fm_surface_row32(fb, 10)[10];
    int      r = (int)(p >> 16 & 255), b = (int)(p & 255), a = (int)(p >> 24);
    CHECK(abs(r - 128) <= 1 && abs(b - 127) <= 1 && abs(a - 192) <= 1, "GL blend of straight colors (%08x)", p);
    bs.eq_rgb = FM3D_BLEND_MIN;
    fm3d_set_blend_state(c, &bs);
    fm3d_draw(c, tri, 3);
    fm3d_flush(c);
    p = fm_surface_row32(fb, 10)[10];
    CHECK((p & 0xFFFFFF) == ((uint32_t)r << 16), "GL blend MIN (%08x)", p);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

/* count pixels equal to c inside rows [y0, y1] */
static int count_rows(const fm_surface* s, fm_color c, int y0, int y1)
{
    int n = 0;
    for (int y = y0; y <= y1; y++)
        for (int x = 0; x < s->width; x++) n += fm_surface_row32(s, y)[x] == c;
    return n;
}

/* lines and points: quads after the vertex stage */
static void test_prims(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, zb);
    pixel_space(c);
    fm_color g = FM_RGB(0, 255, 0), r = FM_RGB(255, 0, 0);
    fm3d_set_cull(c, FM3D_CULL_BACK, FM3D_FRONT_CCW); /* lines and points ignore culling */
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);

    /* a horizontal line, 1 and 3 pixels wide */
    fm3d_clear_color(c, 0);
    fm3d_set_primitive(c, FM3D_PRIM_LINES);
    fm3d_vertex hl[2] = { vtx(10, 20.5f, 0, 0, 0, g), vtx(110, 20.5f, 0, 0, 0, g) };
    fm3d_draw(c, hl, 2);
    fm3d_flush(c);
    CHECK(count_nonzero(fb) == 100 && count_rows(fb, g, 20, 20) == 100, "1 px line: 100 pixels in row 20 (%d)", count_nonzero(fb));
    fm3d_clear_color(c, 0);
    fm3d_set_line_width(c, 3);
    fm3d_draw(c, hl, 2);
    fm3d_flush(c);
    CHECK(count_nonzero(fb) == 300 && count_rows(fb, g, 19, 21) == 300, "3 px line: rows 19 .. 21 (%d)", count_nonzero(fb));
    /* a steep line widens along x; indexed */
    fm3d_clear_color(c, 0);
    fm3d_set_line_width(c, 1);
    fm3d_vertex vl[2] = { vtx(50.5f, 100, 0, 0, 0, g), vtx(52.5f, 160, 0, 0, 0, g) };
    uint32_t    li[2] = { 1, 0 };
    fm3d_draw_indexed(c, vl, 2, li, 2);
    fm3d_flush(c);
    CHECK(count_nonzero(fb) == 60, "steep line: one pixel per row (%d)", count_nonzero(fb));

    /* depth applies: a near red line over a far green one */
    fm3d_clear_color(c, 0);
    fm3d_clear_depth(c, 1.0f);
    fm3d_set_depth_test(c, FM3D_LESS, 1);
    fm3d_vertex far_l[2]  = { vtx(10, 40.5f, -0.5f, 0, 0, g), vtx(110, 40.5f, -0.5f, 0, 0, g) };
    fm3d_vertex near_l[2] = { vtx(10, 40.5f, 0.5f, 0, 0, r), vtx(110, 40.5f, 0.5f, 0, 0, r) };
    fm3d_draw(c, near_l, 2);
    fm3d_draw(c, far_l, 2);
    fm3d_flush(c);
    CHECK(count_rows(fb, r, 40, 40) == 100 && count_rows(fb, g, 40, 40) == 0, "lines are depth tested");
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);

    /* points: size x size squares; centers outside the view are culled */
    fm3d_clear_color(c, 0);
    fm3d_set_primitive(c, FM3D_PRIM_POINTS);
    fm3d_set_point_size(c, 4);
    fm3d_vertex pt[3] = { vtx(50, 50, 0, 0, 0, g), vtx(200.5f, 100.5f, 0, 0, 0, g), vtx(-1, 50, 0, 0, 0, g) };
    fm3d_draw(c, pt, 3);
    fm3d_flush(c);
    CHECK(count_nonzero(fb) == 32 && fm_surface_row32(fb, 48)[48] == g && fm_surface_row32(fb, 51)[51] == g,
          "4 px points: 16 pixels each, the offscreen one culled (%d)", count_nonzero(fb));
    fm3d_set_point_size(c, 1);
    fm3d_set_primitive(c, FM3D_PRIM_TRIANGLES);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);

    fm3d_destroy(c);
    fm_surface_destroy(fb);
    fm_surface_destroy(zb);
}

/* fixed function multitexture (second stage, fm3d_vertex_mt) and fog */
static void test_multitexture_fog(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    pixel_space(c);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_blend_state bs = { FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
    fm3d_set_blend_state(c, &bs); /* straight colors, as an API layer uses it */
    /* base: a white texture; lightmap: 2 x 1, left gray (128), right red */
    fm_surface* base = fm_surface_create(1, 1, FM_FORMAT_ARGB32);
    fm_surface* lm   = fm_surface_create(2, 1, FM_FORMAT_ARGB32);
    fm_surface_row32(base, 0)[0] = FM_RGB(255, 255, 255);
    fm_surface_row32(lm, 0)[0]   = FM_RGB(128, 128, 128);
    fm_surface_row32(lm, 0)[1]   = FM_RGB(255, 0, 0);
    fm3d_texture* tb = fm3d_texture_create(base, 0);
    fm3d_texture* tl = fm3d_texture_create(lm, 0);
    fm3d_sampler  ns = { FM3D_FILTER_NEAREST, FM_WRAP_CLAMP, FM_WRAP_CLAMP, 0, FM_WRAP_CLAMP };
    fm3d_set_texture(c, tb, &ns);
    fm3d_set_texenv(c, FM3D_TEXENV_MODULATE);
    fm3d_set_texture_stage1(c, tl, &ns, FM3D_TEXENV_MODULATE);
    fm3d_clear_color(c, 0);
    fm_color       wc = FM_RGB(255, 255, 255);
    fm3d_vertex_mt q[6];
    float          P[6][2] = { { 0, 0 }, { 100, 0 }, { 100, 50 }, { 0, 0 }, { 100, 50 }, { 0, 50 } };
    for (int i = 0; i < 6; i++) {
        q[i].v  = vtx(P[i][0], P[i][1], 0, 0.5f, 0.5f, wc); /* stage 0 at the center */
        q[i].u2 = P[i][0] / 100.0f, q[i].v2 = 0.5f;          /* stage 1 across */
    }
    fm3d_draw_mt(c, q, 6, NULL, 6);
    fm3d_flush(c);
    uint32_t l = fm_surface_row32(fb, 25)[20], r = fm_surface_row32(fb, 25)[80];
    CHECK(l == FM_RGB(128, 128, 128) && r == FM_RGB(255, 0, 0), "second texture stage modulates (%08x %08x)", l, r);
    /* fm3d_vertex: the second stage reads the first coordinates */
    fm3d_clear_color(c, 0);
    fm3d_vertex q1[6];
    for (int i = 0; i < 6; i++) q1[i] = vtx(P[i][0], P[i][1], 0, 0.9f, 0.5f, wc);
    fm3d_draw(c, q1, 6);
    fm3d_flush(c);
    CHECK(fm_surface_row32(fb, 25)[20] == FM_RGB(255, 0, 0), "second stage with plain vertices (%08x)", fm_surface_row32(fb, 25)[20]);
    fm3d_set_texture_stage1(c, NULL, NULL, FM3D_TEXENV_MODULATE);
    fm3d_set_texture(c, NULL, NULL);

    /* fog: ortho, so the eye distance (clip w) is 1 everywhere */
    fm3d_clear_color(c, 0);
    fm3d_set_fog(c, FM3D_FOG_LINEAR, FM_RGB(0, 0, 255), 0.0f, 2.0f, 0.0f); /* f = (2 - 1) / 2 = 0.5 */
    fm3d_draw(c, q1, 6);
    fm3d_flush(c);
    uint32_t f = fm_surface_row32(fb, 25)[20];
    CHECK(abs((int)((f >> 16) & 255) - 128) <= 1 && (int)((f >> 8) & 255) >= 126 && (int)((f >> 8) & 255) <= 129 && (f & 255) >= 254,
          "linear fog half way to the fog color (%08x)", f);
    fm3d_set_fog(c, FM3D_FOG_EXP, FM_RGB(0, 0, 0), 0, 0, 0.6931472f); /* e^-ln2 = 0.5 */
    fm3d_clear_color(c, 0);
    fm3d_draw(c, q1, 6);
    fm3d_flush(c);
    f = fm_surface_row32(fb, 25)[20];
    CHECK(abs((int)((f >> 16) & 255) - 128) <= 1 && (f >> 24) == 255, "exp fog (%08x)", f);
    fm3d_set_fog(c, FM3D_FOG_OFF, 0, 0, 0, 0);
    fm3d_texture_release(tb);
    fm3d_texture_release(tl);
    fm_surface_destroy(base);
    fm_surface_destroy(lm);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

/* the work / time counters of fm3d_stats */
static void test_stats_work(void)
{
    fm_surface*  fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*    c  = fm3d_create();
    fm_executor* ex = fm_executor_create(4);
    fm3d_set_target(c, fb, NULL);
    pixel_space(c);
    if (ex) fm3d_set_executor(c, ex), fm3d_set_deferred(c, 1);
    fm_color    g = FM_RGB(0, 255, 0);
    fm3d_vertex tri[3] = { vtx(0, 0, 0, 0, 0, g), vtx(200, 0, 0, 0, 0, g), vtx(0, 200, 0, 0, 0, g) };
    fm3d_draw(c, tri, 3);
    fm3d_draw(c, tri, 3);
    fm3d_flush(c);
    fm3d_stats st = fm3d_get_stats(c);
    CHECK(st.draws == 2 && st.ns_raster > 0 && st.workers >= 1, "stats: draws %d, raster ns %d, workers %d", (int)st.draws,
          (int)st.ns_raster, (int)st.workers);
    if (ex) CHECK(st.flushes == 1 && st.tiles > 0 && st.tile_items >= st.tiles && st.ns_busy > 0, "stats: flushes %d tiles %d items %d",
                  (int)st.flushes, (int)st.tiles, (int)st.tile_items);
    fm3d_destroy(c);
    fm_executor_destroy(ex);
    fm_surface_destroy(fb);
}

static int diff_count(const fm_surface* a, const fm_surface* b);

static int count_color(const fm_surface* s, fm_color c)
{
    int n = 0;
    for (int y = 0; y < s->height; y++)
        for (int x = 0; x < s->width; x++) n += fm_surface_get_pixel(s, x, y) == c;
    return n;
}

static void test_depth_stencil(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm_surface* sb = fm_surface_create(W, H, FM_FORMAT_A8);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, zb);
    fm3d_set_stencil_buffer(c, sb);
    pixel_space(c);
    fm_color g = FM_RGB(0, 255, 0), r = FM_RGB(255, 0, 0), bl = FM_RGB(0, 0, 255);
    fm3d_vertex tri[3]  = { vtx(20, 20, 0, 0, 0, g), vtx(200, 40, 0, 0, 0, g), vtx(60, 200, 0, 0, 0, g) };
    fm3d_vertex tri_r[3] = { vtx(20, 20, 0, 0, 0, r), vtx(200, 40, 0, 0, 0, r), vtx(60, 200, 0, 0, 0, r) };

    /* cull front and back: nothing is drawn */
    fm3d_clear_color(c, 0);
    fm3d_set_cull(c, FM3D_CULL_FRONT_AND_BACK, FM3D_FRONT_CCW);
    fm3d_reset_stats(c);
    fm3d_draw(c, tri, 3);
    CHECK(count_nonzero(fb) == 0 && fm3d_get_stats(c).triangles_culled == 1, "cull front and back");
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);

    /* reference pixel count of the triangle */
    fm3d_clear_depth(c, 1.0f);
    fm3d_set_depth_test(c, FM3D_LESS, 1);
    fm3d_draw(c, tri, 3);
    int area = count_color(fb, g);
    CHECK(area > 10000, "triangle area %d", area);

    /* coplanar decal: LESS fails without bias, passes with negative bias */
    fm3d_draw(c, tri_r, 3);
    CHECK(count_color(fb, r) == 0, "coplanar without bias z-fights to the first (%d)", count_color(fb, r));
    fm3d_set_depth_bias(c, -1.0f, -4.0f);
    fm3d_draw(c, tri_r, 3);
    CHECK(count_color(fb, r) == area, "negative depth bias pulls the decal in front (%d of %d)", count_color(fb, r), area);
    fm3d_set_depth_bias(c, 0, 0);

    /* depth range */
    fm3d_clear_depth(c, 1.0f);
    fm3d_set_depth_range(c, 0.5f, 1.0f);
    fm3d_draw(c, tri, 3); /* z = 0 -> depth01 0.5 -> window 0.75 */
    CHECK(fabsf(fm_surface_rowf(zb, 60)[80] - 0.75f) < 1e-6f, "depth range (%f)", fm_surface_rowf(zb, 60)[80]);
    fm3d_set_depth_range(c, 0.0f, 1.0f);

    /* stencil mask: stencil-only pass, then draw only where stencil == 1 */
    fm3d_clear_color(c, 0);
    fm3d_clear_stencil(c, 0);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_set_stencil_test(c, 1);
    fm3d_set_color_write(c, 0);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_ALWAYS, 1, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_REPLACE);
    fm3d_draw(c, tri, 3);
    CHECK(count_nonzero(fb) == 0, "color writes disabled");
    fm3d_set_color_write(c, 1);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_EQUAL, 1, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP);
    fm3d_vertex full[6] = { vtx(0, 0, 0, 0, 0, bl), vtx(W, 0, 0, 0, 0, bl), vtx(W, H, 0, 0, 0, bl),
                            vtx(0, 0, 0, 0, 0, bl), vtx(W, H, 0, 0, 0, bl), vtx(0, H, 0, 0, 0, bl) };
    fm3d_draw(c, full, 6);
    CHECK(count_color(fb, bl) == area && count_nonzero(fb) == area, "stencil masked fill covers exactly the mask (%d vs %d)",
          count_color(fb, bl), area);

    /* two sided stencil: front faces increment, back faces decrement */
    fm3d_clear_stencil(c, 0);
    fm3d_set_color_write(c, 0);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_ALWAYS, 0, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_INCR_WRAP);
    fm3d_set_stencil_op(c, FM3D_FACE_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_DECR_WRAP);
    /* visually counter clockwise = front, visually clockwise = back */
    fm3d_vertex front_q[3] = { vtx(10, 10, 0, 0, 0, g), vtx(10, 110, 0, 0, 0, g), vtx(110, 10, 0, 0, 0, g) };
    fm3d_vertex back_q[3]  = { vtx(40, 40, 0, 0, 0, g), vtx(140, 40, 0, 0, 0, g), vtx(40, 140, 0, 0, 0, g) };
    fm3d_draw(c, front_q, 3);
    fm3d_draw(c, back_q, 3);
    CHECK(fm_surface_row8(sb, 15)[15] == 1, "front face incremented (%u)", fm_surface_row8(sb, 15)[15]);
    CHECK(fm_surface_row8(sb, 130)[45] == 255, "back face decremented with wrap (%u)", fm_surface_row8(sb, 130)[45]);
    CHECK(fm_surface_row8(sb, 45)[45] == 0, "overlap cancels (%u)", fm_surface_row8(sb, 45)[45]);

    /* write mask */
    fm3d_clear_stencil(c, 0xf0);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_ALWAYS, 0x3c, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_REPLACE);
    fm3d_set_stencil_write_mask(c, FM3D_FACE_FRONT_AND_BACK, 0x0f);
    fm3d_draw(c, tri, 3);
    CHECK(fm_surface_row8(sb, 60)[80] == 0xfc, "stencil write mask (%02x)", fm_surface_row8(sb, 60)[80]);
    /* read mask: (ref & 0x0f) EQUAL (stencil & 0x0f) */
    fm3d_set_color_write(c, 1);
    fm3d_set_stencil_write_mask(c, FM3D_FACE_FRONT_AND_BACK, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_EQUAL, 0xac, 0x0f);
    fm3d_clear_color(c, 0);
    fm3d_draw(c, full, 6);
    CHECK(count_color(fb, bl) == area, "stencil read mask (%d vs %d)", count_color(fb, bl), area);

    /* stencil fail / depth fail ops */
    fm3d_clear_stencil(c, 5);
    fm3d_clear_depth(c, 0.0f); /* everything fails LESS */
    fm3d_set_depth_test(c, FM3D_LESS, 0);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_ALWAYS, 9, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_INVERT, FM3D_STENCIL_KEEP);
    fm3d_draw(c, tri, 3);
    CHECK(fm_surface_row8(sb, 60)[80] == (uint8_t)~5u, "depth fail op (%02x)", fm_surface_row8(sb, 60)[80]);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_NEVER, 9, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_ZERO, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP);
    fm3d_draw(c, tri, 3);
    CHECK(fm_surface_row8(sb, 60)[80] == 0 && fm_surface_row8(sb, 5)[5] == 5, "stencil fail op");

    fm3d_destroy(c);
    fm_surface_destroy(fb);
    fm_surface_destroy(zb);
    fm_surface_destroy(sb);
}

static void test_depth_formats(void)
{
    fm_surface* fb  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_color    g = FM_RGB(0, 255, 0), r = FM_RGB(255, 0, 0), bl = FM_RGB(0, 0, 255);
    fm_format   fmts[3] = { FM_FORMAT_D32F, FM_FORMAT_D16, FM_FORMAT_D24S8 };
    const char* names[3] = { "D32F", "D16", "D24S8" };
    for (int fi = 0; fi < 3; fi++) {
        fm_surface* zb = fm_surface_create(W, H, fmts[fi]);
        fm3d_ctx*   c  = fm3d_create();
        fm3d_set_target(c, fb, zb);
        pixel_space(c);
        fm3d_clear_color(c, 0);
        fm3d_clear_depth(c, 1.0f);
        /* ortho: z = 0.5 -> depth 0.25 */
        fm3d_vertex nq[3] = { vtx(50, 50, 0.5f, 0, 0, g), vtx(150, 50, 0.5f, 0, 0, g), vtx(50, 150, 0.5f, 0, 0, g) };
        fm3d_vertex fq[3] = { vtx(40, 40, -0.5f, 0, 0, r), vtx(160, 40, -0.5f, 0, 0, r), vtx(40, 160, -0.5f, 0, 0, r) };
        fm3d_draw(c, nq, 3);
        fm3d_draw(c, fq, 3);
        float step = fmts[fi] == FM_FORMAT_D16 ? 1.0f / 65535 : 1e-6f;
        CHECK(fm_surface_get_pixel(fb, 60, 60) == g && fm_surface_get_pixel(fb, 45, 45) == r, "%s depth order",
              names[fi]);
        CHECK(fabsf(fm_surface_get_depth(zb, 60, 60) - 0.25f) <= step, "%s stored depth %f", names[fi],
              fm_surface_get_depth(zb, 60, 60));
        /* depth difference below 16 bit precision: equal keys in D16, LESS fails */
        float       dz   = 1.0f / 300000.0f; /* ortho: depth = 0.5 - z / 2 */
        fm3d_vertex nq2[3] = { vtx(50, 50, 0.5f + 2 * dz, 0, 0, bl), vtx(150, 50, 0.5f + 2 * dz, 0, 0, bl),
                               vtx(50, 150, 0.5f + 2 * dz, 0, 0, bl) };
        fm3d_draw(c, nq2, 3);
        int nearer_won = fm_surface_get_pixel(fb, 60, 60) == bl;
        CHECK(fmts[fi] == FM_FORMAT_D16 ? !nearer_won : nearer_won, "%s compares in stored precision", names[fi]);

        if (fmts[fi] == FM_FORMAT_D24S8) {
            /* the packed stencil bits act as the stencil buffer; depth survives stencil clears */
            fm3d_clear_stencil(c, 0);
            fm3d_set_stencil_test(c, 1);
            fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
            fm3d_set_color_write(c, 0);
            fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_ALWAYS, 7, 0xff);
            fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_REPLACE);
            fm3d_draw(c, nq, 3);
            uint32_t word = ((uint32_t*)fm_surface_row8(zb, 60))[60];
            CHECK((word >> 24) == 7 && fabsf(fm_surface_get_depth(zb, 60, 60) - 0.25f) < 1e-5f,
                  "D24S8 packed stencil %08x", word);
            fm3d_set_color_write(c, 1);
            fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_EQUAL, 7, 0xff);
            fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP);
            fm3d_clear_color(c, 0);
            fm3d_vertex full[6] = { vtx(0, 0, 0, 0, 0, bl), vtx(W, 0, 0, 0, 0, bl), vtx(W, H, 0, 0, 0, bl),
                                    vtx(0, 0, 0, 0, 0, bl), vtx(W, H, 0, 0, 0, bl), vtx(0, H, 0, 0, 0, bl) };
            fm3d_draw(c, full, 6);
            CHECK(fm_surface_get_pixel(fb, 60, 60) == bl && fm_surface_get_pixel(fb, 45, 45) == 0,
                  "D24S8 stencil masking");
            fm3d_clear_depth(c, 1.0f);
            CHECK((((uint32_t*)fm_surface_row8(zb, 60))[60] >> 24) == 7, "depth clear keeps D24S8 stencil");
        }
        fm3d_destroy(c);
        fm_surface_destroy(zb);
    }

    /* depth clamp: a triangle crossing the near plane is not clipped */
    fm_surface* zb = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, zb);
    fm_mat4 proj = fm_perspective(fm_radians(70), (float)W / H, 1.0f, 100.0f);
    fm_mat4 view = fm_lookat(fm_v3(0, 1, 0), fm_v3(0, 1, -1), fm_v3(0, 1, 0));
    fm_mat4 id   = fm_mat4_identity();
    fm3d_set_projection(c, &proj);
    fm3d_set_view(c, &view);
    fm3d_set_model(c, &id);
    fm3d_vertex cross[3] = { vtx(-1, 0.2f, -0.5f, 0, 0, g), vtx(1, 0.2f, -0.5f, 0, 0, g), vtx(0, 1.5f, -6, 0, 0, g) };
    int         px[2];
    for (int k = 0; k < 2; k++) {
        fm3d_set_depth_clamp(c, k);
        fm3d_clear_color(c, 0);
        fm3d_clear_depth(c, 1.0f);
        fm3d_reset_stats(c);
        fm3d_draw(c, cross, 3);
        px[k] = count_nonzero(fb);
        if (k)
            CHECK(fm3d_get_stats(c).triangles_clipped == 0, "depth clamp disables near clipping");
        else
            CHECK(fm3d_get_stats(c).triangles_clipped == 1, "near plane clips without depth clamp");
    }
    CHECK(px[1] > px[0], "depth clamp keeps the part in front of the near plane (%d vs %d px)", px[1], px[0]);
    float dmin = 1.0f;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) dmin = fminf(dmin, fm_surface_get_depth(zb, x, y));
    CHECK(dmin == 0.0f, "clamped depth reaches 0 (%f)", dmin);
    fm3d_destroy(c);
    fm_surface_destroy(zb);
    fm_surface_destroy(fb);
}

/* hierarchical z: big occluder first, many hidden triangles after it */
static void test_hiz(void)
{
    fm_surface* ref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* out = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb  = fm_surface_create(W, H, FM_FORMAT_D24S8);
    fm_executor* ex = fm_executor_create(4);
    for (int reversed = 0; reversed < 2; reversed++) {
        fm3d_ctx* c = fm3d_create();
        for (int pass = 0; pass < 2; pass++) {
            fm3d_set_target(c, pass ? out : ref, zb);
            fm3d_set_deferred(c, pass);
            fm3d_set_executor(c, pass ? ex : NULL);
            pixel_space(c);
            fm3d_set_depth_test(c, reversed ? FM3D_GREATER : FM3D_LESS, 1);
            fm3d_clear_color(c, 0);
            fm3d_clear_depth(c, reversed ? 0.0f : 1.0f);
            fm3d_reset_stats(c);
            float       zo = reversed ? -0.5f : 0.5f; /* occluder near (depth 0.25), or far side for reversed */
            fm_color    oc = FM_RGB(90, 90, 90);
            fm3d_vertex occ[6] = { vtx(0, 0, zo, 0, 0, oc), vtx(W, 0, zo, 0, 0, oc), vtx(W, 200, zo, 0, 0, oc),
                                   vtx(0, 0, zo, 0, 0, oc), vtx(W, 200, zo, 0, 0, oc), vtx(0, 200, zo, 0, 0, oc) };
            fm3d_draw(c, occ, 6);
            static fm3d_vertex hidden[3 * 400];
            uint32_t           seed = 99;
            for (int i = 0; i < 400; i++) {
                seed      = seed * 1664525u + 1013904223u;
                float x   = (float)(seed % (W - 40)), y = (float)((seed >> 12) % (H - 40));
                float z   = reversed ? 0.4f : -0.4f; /* behind the occluder */
                fm_color col = FM_RGB(seed & 255, (seed >> 8) & 255, 255);
                hidden[3 * i]     = vtx(x, y, z, 0, 0, col);
                hidden[3 * i + 1] = vtx(x + 40, y, z, 0, 0, col);
                hidden[3 * i + 2] = vtx(x, y + 40, z, 0, 0, col);
            }
            fm3d_draw(c, hidden, 3 * 400);
            fm3d_flush(c);
            if (pass)
                CHECK(fm3d_get_stats(c).hiz_rejected > 200, "hi-z rejects hidden triangles (%s, %llu)",
                      reversed ? "reversed z" : "less", (unsigned long long)fm3d_get_stats(c).hiz_rejected);
        }
        CHECK(diff_count(ref, out) == 0, "hi-z output identical to immediate (%s)", reversed ? "reversed z" : "less");
        fm3d_destroy(c);
    }
    fm_executor_destroy(ex);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
}

/* texel (x, y) encodes x in red, y in green */
static fm_surface* coord_texture(int n)
{
    fm_surface* s = fm_surface_create(n, n, FM_FORMAT_ARGB32);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) fm_surface_row32(s, y)[x] = FM_RGB(x, y, 0);
    return s;
}

static void test_perspective(void)
{
    fm_surface* fb  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* img = coord_texture(256);
    fm3d_texture* tex = fm3d_texture_create(img, 0);
    fm3d_ctx*   c   = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    fm_mat4 proj = fm_perspective(fm_radians(60), (float)W / H, 0.1f, 100.0f);
    fm_mat4 view = fm_lookat(fm_v3(0.3f, 1.5f, 2.0f), fm_v3(0, 0, -4), fm_v3(0, 1, 0));
    fm_mat4 id   = fm_mat4_identity();
    fm3d_set_projection(c, &proj);
    fm3d_set_view(c, &view);
    fm3d_set_model(c, &id);
    fm3d_sampler smp = { FM3D_FILTER_NEAREST, FM_WRAP_CLAMP, FM_WRAP_CLAMP, 0, FM_WRAP_CLAMP };
    fm3d_set_texture(c, tex, &smp);
    fm3d_set_texenv(c, FM3D_TEXENV_REPLACE);
    /* floor quad y = 0, x in [-2, 2], z in [-8, 0], u along x, v along z */
    fm_color    wc = FM_RGB(255, 255, 255);
    fm3d_vertex q[6] = { vtx(-2, 0, 0, 0, 0, wc),  vtx(2, 0, 0, 1, 0, wc),  vtx(2, 0, -8, 1, 1, wc),
                         vtx(-2, 0, 0, 0, 0, wc),  vtx(2, 0, -8, 1, 1, wc), vtx(-2, 0, -8, 0, 1, wc) };
    fm_mat4 inv = fm_mat4_inverse(fm_mat4_mul(proj, view));
    for (int mode = 1; mode >= 0; mode--) {
        fm_surface_clear(fb, 0);
        fm3d_set_perspective_correct(c, mode);
        fm3d_draw(c, q, 6);
        int tested = 0, bad = 0;
        for (int y = 0; y < H; y += 3)
            for (int x = 0; x < W; x += 3) {
                uint32_t p = fm_surface_row32(fb, y)[x];
                if (!p) continue;
                float   nx = ((float)x + 0.5f) / W * 2 - 1, ny = 1 - ((float)y + 0.5f) / H * 2;
                fm_vec4 a  = fm_mat4_mul_vec4(inv, fm_v4(nx, ny, -1, 1));
                fm_vec4 b  = fm_mat4_mul_vec4(inv, fm_v4(nx, ny, 1, 1));
                fm_vec3 p0 = fm_v3(a.x / a.w, a.y / a.w, a.z / a.w), p1 = fm_v3(b.x / b.w, b.y / b.w, b.z / b.w);
                float   t  = p0.y / (p0.y - p1.y);
                float   wx = p0.x + (p1.x - p0.x) * t, wz = p0.z + (p1.z - p0.z) * t;
                float   eu = (wx + 2) / 4 * 256, ev = -wz / 8 * 256;
                if (eu < 1 || eu > 255 || ev < 1 || ev > 255) continue; /* skip quad borders */
                float gu = (float)((p >> 16) & 255) + 0.5f, gv = (float)((p >> 8) & 255) + 0.5f;
                tested++;
                if (fabsf(gu - eu) > 1.0f || fabsf(gv - ev) > 1.0f) bad++;
            }
        if (mode)
            CHECK(tested > 1000 && bad * 100 < tested, "perspective correct texturing: %d of %d samples off", bad,
                  tested);
        else
            CHECK(bad * 5 > tested, "affine mode visibly differs (%d of %d off)", bad, tested);
    }
    fm3d_texture_release(tex);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
    fm_surface_destroy(img);
}

static fm_surface* checker(int n, int cell)
{
    fm_surface* s = fm_surface_create(n, n, FM_FORMAT_ARGB32);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            fm_surface_row32(s, y)[x] = (((x / cell) + (y / cell)) & 1) ? 0xffffffffu : 0xff000000u;
    return s;
}

static void test_mipmaps(void)
{
    fm_surface*   fb  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   img = checker(256, 1);
    fm3d_texture* tex = fm3d_texture_create(img, 1);
    CHECK(fm3d_texture_levels(tex) == 9, "mip levels %d", fm3d_texture_levels(tex));
    const fm_surface* l1 = fm3d_texture_level(tex, 1);
    CHECK(l1 && ((fm_surface_row32(l1, 3)[3] >> 8) & 255) == 128, "mip 1 of 1px checker is gray");
    fm3d_ctx* c = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    fm_mat4 proj = fm_perspective(fm_radians(60), (float)W / H, 0.1f, 500.0f);
    fm_mat4 view = fm_lookat(fm_v3(0, 2, 0), fm_v3(0, 0, -50), fm_v3(0, 1, 0));
    fm_mat4 id   = fm_mat4_identity();
    fm3d_set_projection(c, &proj);
    fm3d_set_view(c, &view);
    fm3d_set_model(c, &id);
    fm3d_set_texenv(c, FM3D_TEXENV_REPLACE);
    fm_color    wc = FM_RGB(255, 255, 255);
    fm3d_vertex q[6] = { vtx(-50, 0, 0, 0, 0, wc),  vtx(50, 0, 0, 40, 0, wc),    vtx(50, 0, -200, 40, 80, wc),
                         vtx(-50, 0, 0, 0, 0, wc),  vtx(50, 0, -200, 40, 80, wc), vtx(-50, 0, -200, 0, 80, wc) };
    double var[2];
    for (int k = 0; k < 2; k++) {
        fm3d_sampler smp = { k ? FM3D_FILTER_TRILINEAR : FM3D_FILTER_NEAREST, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
        fm3d_set_texture(c, tex, &smp);
        fm_surface_clear(fb, 0);
        fm3d_draw(c, q, 6);
        /* variance of the far half of the floor (rows just below the horizon) */
        double s = 0, s2 = 0;
        int    n = 0;
        for (int y = H / 2 + 4; y < H / 2 + 30; y++)
            for (int x = 40; x < W - 40; x++) {
                uint32_t p = fm_surface_row32(fb, y)[x];
                if (!p) continue;
                double g = (double)((p >> 8) & 255);
                s += g;
                s2 += g * g;
                n++;
            }
        double mean = n ? s / n : 0;
        var[k]      = n ? s2 / n - mean * mean : 0;
        if (k) CHECK(fabs(mean - 128) < 20, "trilinear far floor mean %f (expected mid gray)", mean);
    }
    CHECK(var[1] * 10 < var[0], "trilinear removes aliasing: variance %f vs nearest %f", var[1], var[0]);
    char path[512];
    snprintf(path, sizeof(path), "%s/3d_trilinear.png", g_outdir);
    fm_surface_write_png(fb, path);
    fm3d_texture_release(tex);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
    fm_surface_destroy(img);
}

/* ---- scene used for equivalence tests ------------------------------------------------ */

static void cube(fm3d_vertex* out, float s, fm_color base)
{
    static const float P[8][3] = { { -1, -1, -1 }, { 1, -1, -1 }, { 1, 1, -1 }, { -1, 1, -1 },
                                   { -1, -1, 1 },  { 1, -1, 1 },  { 1, 1, 1 },  { -1, 1, 1 } };
    static const int   F[6][4] = { { 4, 5, 6, 7 }, { 1, 0, 3, 2 }, { 0, 4, 7, 3 }, { 5, 1, 2, 6 }, { 7, 6, 2, 3 }, { 0, 1, 5, 4 } };
    static const float shade[6] = { 1.0f, 0.55f, 0.7f, 0.85f, 0.95f, 0.45f };
    int                k        = 0;
    for (int f = 0; f < 6; f++) {
        uint32_t r = (uint32_t)(((base >> 16) & 255) * shade[f]), g = (uint32_t)(((base >> 8) & 255) * shade[f]),
                 b = (uint32_t)((base & 255) * shade[f]);
        fm_color col = FM_ARGB(base >> 24, r, g, b);
        const float uv[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
        const int   tri[6]   = { 0, 1, 2, 0, 2, 3 };
        for (int i = 0; i < 6; i++) {
            const float* p = P[F[f][tri[i]]];
            out[k++]       = vtx(p[0] * s, p[1] * s, p[2] * s, uv[tri[i]][0], uv[tri[i]][1], col);
        }
    }
}

static void draw_scene3d(fm3d_ctx* c, fm3d_texture* tex, fm3d_texture* tex2, float t)
{
    fm3d_clear_color(c, FM_RGB(20, 24, 40));
    fm3d_clear_depth(c, 1.0f);
    fm_mat4 proj = fm_perspective(fm_radians(60), (float)W / H, 0.1f, 200.0f);
    fm_mat4 view = fm_lookat(fm_v3(sinf(t) * 3, 3, 7), fm_v3(0, 0, 0), fm_v3(0, 1, 0));
    fm_mat4 id   = fm_mat4_identity();
    fm3d_set_projection(c, &proj);
    fm3d_set_view(c, &view);
    fm3d_set_model(c, &id);
    fm3d_set_depth_test(c, FM3D_LESS, 1);
    fm3d_set_cull(c, FM3D_CULL_BACK, FM3D_FRONT_CCW);
    fm3d_set_blend(c, FM_OP_SRC_OVER);
    fm3d_set_texenv(c, FM3D_TEXENV_MODULATE);

    /* floor: indexed grid, trilinear, repeat */
    enum { G = 16 };
    static fm3d_vertex fv[(G + 1) * (G + 1)];
    static uint32_t    fi[G * G * 6];
    for (int z = 0; z <= G; z++)
        for (int x = 0; x <= G; x++)
            fv[z * (G + 1) + x] = vtx((float)x - G / 2, -1, (float)z - G / 2, (float)x * 0.5f, (float)z * 0.5f,
                                      FM_RGB(255, 255, 255));
    int k = 0;
    for (int z = 0; z < G; z++)
        for (int x = 0; x < G; x++) {
            uint32_t a = (uint32_t)(z * (G + 1) + x), b = a + 1, cc = a + G + 1, d = cc + 1;
            fi[k++] = a;
            fi[k++] = cc;
            fi[k++] = b;
            fi[k++] = b;
            fi[k++] = cc;
            fi[k++] = d;
        }
    fm3d_sampler tri = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture(c, tex, &tri);
    fm3d_draw_indexed(c, fv, (G + 1) * (G + 1), fi, G * G * 6);

    /* cubes: bilinear, vertex color modulate */
    fm3d_vertex  cb[36];
    fm3d_sampler bil = { FM3D_FILTER_BILINEAR, FM_WRAP_CLAMP, FM_WRAP_CLAMP, 0, FM_WRAP_CLAMP };
    fm3d_set_texture(c, tex2, &bil);
    for (int i = 0; i < 5; i++) {
        fm_mat4 m = fm_rotate(fm_translate(fm_mat4_identity(), fm_v3((float)i * 1.6f - 3.2f, 0, 0)),
                              t + (float)i, fm_v3(0.3f, 1, 0.2f));
        fm3d_set_model(c, &m);
        cube(cb, 0.6f, FM_RGB(255, 255 - i * 40, 128 + i * 25));
        fm3d_draw(c, cb, 36);
    }
    /* gouraud untextured triangle */
    fm3d_set_texture(c, NULL, NULL);
    fm3d_set_model(c, &id);
    fm3d_vertex gt[3] = { vtx(-3, 0.5f, -2, 0, 0, FM_RGB(255, 0, 0)), vtx(3, 0.5f, -2, 0, 0, FM_RGB(0, 255, 0)),
                          vtx(0, 3, -2, 0, 0, FM_RGB(0, 0, 255)) };
    fm3d_draw(c, gt, 3);
    /* transparent + multiply quads, no depth write, no culling */
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_set_depth_test(c, FM3D_LEQUAL, 0);
    fm3d_vertex tq[6] = { vtx(-2, -0.5f, 1, 0, 0, FM_RGBA(80, 200, 255, 120)), vtx(2, -0.5f, 1, 1, 0, FM_RGBA(80, 200, 255, 120)),
                          vtx(2, 1.5f, 1, 1, 1, FM_RGBA(255, 80, 200, 160)),  vtx(-2, -0.5f, 1, 0, 0, FM_RGBA(80, 200, 255, 120)),
                          vtx(2, 1.5f, 1, 1, 1, FM_RGBA(255, 80, 200, 160)),  vtx(-2, 1.5f, 1, 0, 1, FM_RGBA(255, 80, 200, 160)) };
    fm3d_draw(c, tq, 6);
    fm3d_set_blend(c, FM_OP_MULTIPLY);
    fm_mat4 m2 = fm_translate(fm_mat4_identity(), fm_v3(1.5f, 0.5f, 2.0f));
    fm3d_set_model(c, &m2);
    fm3d_draw(c, tq, 6);
    /* alpha tested cutout using the texture alpha */
    fm3d_set_blend(c, FM_OP_SRC_OVER);
    fm3d_set_depth_test(c, FM3D_LESS, 1);
    fm3d_set_alpha_test(c, FM3D_GREATER, 0.5f);
    fm3d_set_texture(c, tex2, &bil);
    fm_mat4 m3 = fm_translate(fm_mat4_identity(), fm_v3(-2.5f, 0.8f, 2.5f));
    fm3d_set_model(c, &m3);
    fm3d_draw(c, tq, 6);
    fm3d_set_alpha_test(c, FM3D_ALWAYS, 0);
    /* stencil: mark the gouraud triangle, then a masked blue pass */
    fm3d_set_texture(c, NULL, NULL);
    fm3d_set_model(c, &id);
    fm3d_clear_stencil(c, 0);
    fm3d_set_stencil_test(c, 1);
    fm3d_set_color_write(c, 0);
    fm3d_set_depth_test(c, FM3D_LEQUAL, 0);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_ALWAYS, 1, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_INCR);
    fm3d_draw(c, gt, 3);
    fm3d_set_color_write(c, 1);
    fm3d_set_stencil_func(c, FM3D_FACE_FRONT_AND_BACK, FM3D_EQUAL, 1, 0xff);
    fm3d_set_stencil_op(c, FM3D_FACE_FRONT_AND_BACK, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP, FM3D_STENCIL_KEEP);
    fm3d_set_blend(c, FM_OP_SCREEN);
    fm_mat4 m4 = fm_translate(fm_mat4_identity(), fm_v3(0.5f, -0.3f, -1.0f));
    fm3d_set_model(c, &m4);
    fm3d_draw(c, tq, 6);
    fm3d_set_blend(c, FM_OP_SRC_OVER);
    fm3d_set_stencil_test(c, 0);
    /* coplanar decal on the gouraud triangle via depth bias */
    fm3d_set_model(c, &id);
    fm3d_set_depth_test(c, FM3D_LESS, 1);
    fm3d_set_depth_bias(c, -1.0f, -2.0f);
    fm3d_vertex decal[3] = { vtx(-1, 1, -2, 0, 0, FM_RGBA(255, 255, 255, 180)), vtx(1, 1, -2, 0, 0, FM_RGBA(255, 255, 255, 180)),
                             vtx(0, 2.2f, -2, 0, 0, FM_RGBA(255, 255, 255, 180)) };
    fm3d_draw(c, decal, 3);
    fm3d_set_depth_bias(c, 0, 0);

    /* an indexed strip */
    {
        fm3d_vertex sv[8];
        uint32_t    si[18];
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 2; k++)
                sv[i * 2 + k] = vtx(-3.0f + (float)k * 0.6f + 0.3f * (float)i * sinf(0.5f + t * 0.3f), -0.8f + (float)i * 0.8f, 1.5f,
                                    (float)k, (float)i / 3.0f, FM_RGB(255, 200 - i * 40, 80 + i * 50));
        for (int i = 0; i < 3; i++) {
            uint32_t a0 = (uint32_t)(i * 2), a1 = a0 + 1, b0 = a0 + 2, b1 = a0 + 3;
            si[i * 6] = a0, si[i * 6 + 1] = a1, si[i * 6 + 2] = b1, si[i * 6 + 3] = a0, si[i * 6 + 4] = b1, si[i * 6 + 5] = b0;
        }
        fm3d_set_model(c, &id);
        fm3d_set_texture(c, tex2, NULL);
        fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
        fm3d_draw_indexed(c, sv, 8, si, 18);
        fm3d_set_texture(c, NULL, NULL);
    }

    /* scissored draw */
    fm3d_set_scissor(c, 1, 30, 20, 90, 70);
    fm3d_set_model(c, &id);
    fm3d_set_texture(c, NULL, NULL);
    fm3d_draw(c, gt, 3);
    fm3d_set_scissor(c, 0, 0, 0, 0, 0);
    fm3d_set_texture(c, NULL, NULL);
}

static fm_surface* tex_image(int n)
{
    fm_surface* s = fm_surface_create(n, n, FM_FORMAT_ARGB32);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int      cc = ((x / 8) + (y / 8)) & 1;
            float    d  = sqrtf((float)((x - n / 2) * (x - n / 2) + (y - n / 2) * (y - n / 2))) / (n / 2);
            uint32_t a  = d < 0.8f ? 255 : (d < 1.0f ? (uint32_t)((1.0f - d) * 5 * 255) : 0);
            fm_surface_row32(s, y)[x] = fm_premultiply(cc ? FM_RGBA(240, 200, 60, a) : FM_RGBA(60, 90, 200, a));
        }
    return s;
}

static int diff_count(const fm_surface* a, const fm_surface* b)
{
    int n = 0;
    for (int y = 0; y < a->height; y++)
        n += memcmp(fm_surface_row32(a, y), fm_surface_row32(b, y), (size_t)a->width * 4) != 0;
    return n;
}

static void test_equivalence_fmt(fm_format zfmt)
{
    fm_surface*   img  = checker(128, 4);
    fm_surface*   img2 = tex_image(64);
    fm3d_texture* tex  = fm3d_texture_create(img, 1);
    fm3d_texture* tex2 = fm3d_texture_create(img2, 1);
    fm_surface*   ref  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   out  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   zb   = fm_surface_create(W, H, zfmt);
    /* D24S8 uses its packed stencil bits, the others a separate A8 buffer */
    fm_surface*   sb   = zfmt == FM_FORMAT_D24S8 ? NULL : fm_surface_create(W, H, FM_FORMAT_A8);
    fm3d_ctx*     c    = fm3d_create();

    fm_simd_set(FM_SIMD_SCALAR);
    fm3d_set_target(c, ref, zb);
    if (sb) fm3d_set_stencil_buffer(c, sb);
    draw_scene3d(c, tex, tex2, 0.6f);
    char path[512];
    snprintf(path, sizeof(path), "%s/3d_scene_%d.png", g_outdir, (int)zfmt);
    CHECK(fm_surface_write_png(ref, path), "write %s", path);
    fm3d_stats st = fm3d_get_stats(c);
    CHECK(st.triangles_drawn > 200, "scene draws triangles (%llu)", (unsigned long long)st.triangles_drawn);

    fm_simd_level lv[4] = { FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_NEON, FM_SIMD_AVX512 };
    for (int i = 0; i < 4; i++) {
        if (!fm_simd_supported(lv[i])) continue;
        fm_simd_set(lv[i]);
        fm3d_set_target(c, out, zb);
        if (sb) fm3d_set_stencil_buffer(c, sb);
        draw_scene3d(c, tex, tex2, 0.6f);
        int d = diff_count(ref, out);
        CHECK(d == 0, "3d immediate %s vs scalar (depth fmt %d): %d rows differ", fm_simd_name(lv[i]), (int)zfmt, d);
    }
    fm_simd_set(fm_simd_best());
    struct {
        int threads, tile;
    } modes[] = { { 0, 64 }, { 3, 32 }, { 4, 64 }, { 8, 128 }, { -1, 64 }, { -1, 16 } };
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        fm_executor* ex = modes[i].threads ? fm_executor_create(modes[i].threads) : NULL;
        fm3d_set_target(c, out, zb);
        if (sb) fm3d_set_stencil_buffer(c, sb);
        fm3d_set_tile_size(c, modes[i].tile);
        fm3d_set_deferred(c, 1);
        fm3d_set_executor(c, ex);
        for (int f = 0; f < 2; f++) {
            draw_scene3d(c, tex, tex2, 0.6f);
            fm3d_flush(c);
        }
        int d = diff_count(ref, out);
        CHECK(d == 0, "3d tiled (depth fmt %d threads=%d workers=%d tile=%d): %d rows differ", (int)zfmt,
              modes[i].threads, ex ? ex->workers : 1, modes[i].tile, d);
        fm3d_set_deferred(c, 0);
        fm3d_set_executor(c, NULL);
        fm_executor_destroy(ex);
    }
    fm3d_destroy(c);
    fm3d_texture_release(tex);
    fm3d_texture_release(tex2);
    fm_surface_destroy(img);
    fm_surface_destroy(img2);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
    fm_surface_destroy(sb);
}

static void test_equivalence(void)
{
    test_equivalence_fmt(FM_FORMAT_D32F);
    test_equivalence_fmt(FM_FORMAT_D16);
    test_equivalence_fmt(FM_FORMAT_D24S8);
}

static void test_msaa(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm_color    wc = FM_RGB(255, 255, 255);
    for (int S = 4; S <= 8; S += 4) {
        fm3d_ctx* c = fm3d_create();
        fm3d_set_target(c, fb, zb);
        fm3d_set_msaa(c, S);
        CHECK(fm3d_get_msaa(c) == S, "msaa %d set", S);
        pixel_space(c);
        fm3d_clear_color(c, FM_RGB(0, 0, 0));
        fm3d_clear_depth(c, 1.0f);
        /* a slanted edge: resolved edge pixels take exact coverage levels */
        fm3d_vertex tri[3] = { vtx(10, 10, 0, 0, 0, wc), vtx(300, 70, 0, 0, 0, wc), vtx(10, 220, 0, 0, 0, wc) };
        fm3d_draw(c, tri, 3);
        fm3d_flush(c); /* resolve */
        int partial = 0, bad_level = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                uint32_t g = (fm_surface_get_pixel(fb, x, y) >> 8) & 255;
                if (g != 0 && g != 255) {
                    partial++;
                    int ok = 0;
                    for (int k = 1; k < S; k++) ok |= g == (uint32_t)((k * 255 + S / 2) / S);
                    bad_level += !ok;
                }
            }
        CHECK(partial > 300, "msaa %dx: anti-aliased edge pixels (%d)", S, partial);
        CHECK(bad_level == 0, "msaa %dx: edge values are coverage levels (%d off)", S, bad_level);
        CHECK(fm_surface_get_pixel(fb, 40, 100) == wc && fm_surface_get_pixel(fb, 310, 230) == FM_RGB(0, 0, 0),
              "msaa %dx interior / exterior", S);
        /* the depth target gets the resolved depth (sample 0): copies of it see the scene */
        CHECK(((const float*)fm_surface_row8(zb, 100))[40] < 1.0f && ((const float*)fm_surface_row8(zb, 230))[310] == 1.0f,
              "msaa %dx: depth resolved into the depth target", S);

        /* no seam: two triangles sharing a diagonal cover every sample once */
        fm3d_clear_color(c, FM_RGB(0, 0, 0));
        fm3d_clear_depth(c, 1.0f);
        fm3d_vertex q[6] = { vtx(20.3f, 20.6f, 0, 0, 0, wc), vtx(290.2f, 30.1f, 0, 0, 0, wc), vtx(280.7f, 210.4f, 0, 0, 0, wc),
                             vtx(20.3f, 20.6f, 0, 0, 0, wc), vtx(280.7f, 210.4f, 0, 0, 0, wc), vtx(30.9f, 200.2f, 0, 0, 0, wc) };
        fm3d_draw(c, q, 6);
        fm3d_flush(c);
        int seam = 0;
        for (int i = 0; i < 200; i++) { /* pixels along the shared diagonal */
            float t  = (float)i / 199.0f;
            int   x  = (int)(20.3f + (280.7f - 20.3f) * t), y = (int)(20.6f + (210.4f - 20.6f) * t);
            float px[4] = { 20.3f, 290.2f, 280.7f, 30.9f }, py[4] = { 20.6f, 30.1f, 210.4f, 200.2f };
            if (inside_convex(px, py, 4, (float)x + 0.5f, (float)y + 0.5f, 1.5f) != 1) continue;
            seam += fm_surface_get_pixel(fb, x, y) != wc;
        }
        CHECK(seam == 0, "msaa %dx: no seam along a shared edge (%d)", S, seam);
        fm3d_destroy(c);
    }
    fm_surface_destroy(fb);
    fm_surface_destroy(zb);
}

static void test_msaa_equivalence(void)
{
    fm_surface*   img  = checker(128, 4);
    fm_surface*   img2 = tex_image(64);
    fm3d_texture* tex  = fm3d_texture_create(img, 1);
    fm3d_texture* tex2 = fm3d_texture_create(img2, 1);
    fm_surface*   ref  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   out  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   zb   = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm_surface*   sb   = fm_surface_create(W, H, FM_FORMAT_A8);
    for (int S = 4; S <= 8; S += 4) {
        fm3d_ctx* c = fm3d_create();
        fm3d_set_msaa(c, S);
        fm_simd_set(FM_SIMD_SCALAR);
        fm3d_set_target(c, ref, zb);
        fm3d_set_stencil_buffer(c, sb);
        draw_scene3d(c, tex, tex2, 0.6f);
        fm3d_flush(c);
        if (S == 4) {
            char path[512];
            snprintf(path, sizeof(path), "%s/3d_scene_msaa4.png", g_outdir);
            fm_surface_write_png(ref, path);
        }
        fm_simd_level lv[4] = { FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_NEON, FM_SIMD_AVX512 };
        for (int i = 0; i < 4; i++) {
            if (!fm_simd_supported(lv[i])) continue;
            fm_simd_set(lv[i]);
            fm3d_set_target(c, out, zb);
            fm3d_set_stencil_buffer(c, sb);
            draw_scene3d(c, tex, tex2, 0.6f);
            fm3d_flush(c);
            CHECK(diff_count(ref, out) == 0, "msaa %dx %s vs scalar", S, fm_simd_name(lv[i]));
        }
        fm_simd_set(fm_simd_best());
        struct {
            int threads, tile;
        } modes[] = { { 0, 64 }, { 4, 32 }, { -1, 64 }, { -1, 16 } };
        for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
            fm_executor* ex = modes[i].threads ? fm_executor_create(modes[i].threads) : NULL;
            fm3d_set_target(c, out, zb);
            fm3d_set_stencil_buffer(c, sb);
            fm3d_set_tile_size(c, modes[i].tile);
            fm3d_set_deferred(c, 1);
            fm3d_set_executor(c, ex);
            draw_scene3d(c, tex, tex2, 0.6f);
            fm3d_flush(c);
            CHECK(diff_count(ref, out) == 0, "msaa %dx tiled (threads %d, tile %d) vs immediate", S, modes[i].threads,
                  modes[i].tile);
            fm3d_set_deferred(c, 0);
            fm3d_set_executor(c, NULL);
            fm_executor_destroy(ex);
        }
        fm3d_destroy(c);
    }
    fm3d_texture_release(tex);
    fm3d_texture_release(tex2);
    fm_surface_destroy(img);
    fm_surface_destroy(img2);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
    fm_surface_destroy(sb);
}

static void test_swapchain(void)
{
    fm_swapchain* sc = fm_swapchain_create(64, 32, 2, 1);
    fm_surface*   b0 = fm_swapchain_back(sc);
    fm_surface_clear(b0, FM_RGB(1, 2, 3));
    fm_surface* f0 = fm_swapchain_present(sc);
    fm_surface* b1 = fm_swapchain_back(sc);
    CHECK(f0 == b0 && b1 != b0 && fm_swapchain_front(sc) == b0, "swapchain present swaps buffers");
    CHECK(fm_swapchain_depth(sc) && fm_swapchain_depth(sc)->format == FM_FORMAT_D32F, "swapchain depth");
    fm_swapchain_present(sc);
    CHECK(fm_swapchain_back(sc) == b0, "double buffer cycles");
    fm_swapchain_destroy(sc);
}

#if FM_FEATURE_VBO
/* vertex buffers: same pixels as copying draws (indexed / not, immediate /
 * deferred on a pool), alive until the flush after release, validation */
static void draw_grid(fm3d_ctx* c, fm3d_buffer* vb, fm3d_buffer* ib, const fm3d_vertex* v, const uint32_t* idx,
                      int nv, int ni)
{
    for (int k = 0; k < 24; k++) {
        fm_mat4 m = fm_translate(fm_mat4_identity(), fm_v3((float)(k % 6) * 50.0f + 5.0f, (float)(k / 6) * 55.0f + 8.0f, 0));
        m         = fm_rotate(m, 0.1f * (float)k, fm_v3(0, 0, 1));
        fm3d_set_model(c, &m);
        if (k & 1) {
            if (ib)
                fm3d_draw_buffer(c, ib, 0, ni);
            else
                fm3d_draw_indexed(c, v, nv, idx, ni);
        } else {
            if (vb)
                fm3d_draw_buffer(c, vb, 3, 6); /* sub range: the second quad's two triangles */
            else
                fm3d_draw(c, v + 3, 6);
        }
    }
}

static void test_buffers(void)
{
    fm_color    col[4] = { FM_RGB(255, 60, 60), FM_RGB(60, 255, 60), FM_RGB(60, 60, 255), FM_RGB(250, 220, 40) };
    fm3d_vertex v[12];
    for (int i = 0; i < 12; i++) {
        float x = (float)((i * 7) % 5) * 9.0f, y = (float)((i * 3) % 4) * 11.0f;
        v[i]    = vtx(x, y, 0.1f * (float)(i % 3), x / 40.0f, y / 40.0f, col[i % 4]);
    }
    uint32_t     idx[18] = { 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8 };
    fm3d_buffer* vb      = fm3d_buffer_create(v, 12, NULL, 0);
    fm3d_buffer* ib      = fm3d_buffer_create(v, 12, idx, 18);
    CHECK(vb && ib, "buffer creation");
    CHECK(fm3d_buffer_vertex_count(ib) == 12 && fm3d_buffer_index_count(ib) == 18 && fm3d_buffer_index_count(vb) == 0,
          "buffer counts");
    uint32_t bad[3] = { 0, 1, 12 };
    CHECK(fm3d_buffer_create(v, 12, bad, 3) == NULL, "buffer rejects an out of range index");

    fm_surface*  ref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*  out = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*  zb  = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm_executor* ex  = fm_executor_create(4);
    fm3d_ctx*    c   = fm3d_create();
    for (int mode = 0; mode < 2; mode++) {
        fm3d_set_deferred(c, mode);
        fm3d_set_executor(c, mode ? ex : NULL);
        fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
        fm3d_set_depth_test(c, FM3D_LEQUAL, 1);
        fm3d_set_blend(c, FM_OP_SRC_OVER);
        fm3d_set_opacity(c, 0.8f);
        for (int pass = 0; pass < 2; pass++) {
            fm3d_set_target(c, pass ? out : ref, zb);
            pixel_space(c);
            fm3d_clear_color(c, FM_RGB(10, 10, 10));
            fm3d_clear_depth(c, 1.0f);
            if (pass)
                draw_grid(c, vb, ib, v, idx, 12, 18);
            else
                draw_grid(c, NULL, NULL, v, idx, 12, 18);
            fm3d_flush(c);
        }
        CHECK(diff_count(ref, out) == 0, "buffer draws match copying draws (%s)", mode ? "deferred mt" : "immediate");
        CHECK(count_nonzero(ref) > 0, "buffer scene drew something");
    }
    /* deferred: release before the flush, the recorded draw keeps it alive */
    fm3d_buffer* tmp = fm3d_buffer_create(v, 12, idx, 18);
    fm3d_set_target(c, out, zb);
    pixel_space(c);
    fm3d_clear_color(c, FM_RGB(10, 10, 10));
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw_buffer(c, tmp, 0, 18);
    fm3d_buffer_release(tmp);
    for (int i = 0; i < 12; i++) v[i].color = 0; /* the source array is no longer referenced either */
    fm3d_flush(c);
    CHECK(fm_surface_get_pixel(out, 9, 11) != FM_RGB(10, 10, 10), "buffer released before flush still renders");
    fm3d_draw_buffer(c, vb, 10, 6); /* out of range: ignored */
    fm3d_flush(c);
    fm3d_destroy(c);
    fm_executor_destroy(ex);
    fm3d_buffer_release(vb);
    fm3d_buffer_release(ib);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
}
#endif

#if FM_FEATURE_TNL
/* ---- fixed function lighting ---- */

/* full screen quad at z = 0 facing the viewer (+z), normal n, white */
static void lit_quad(fm3d_ctx* c, fm_vec3 n)
{
    fm3d_vertex q[6];
    float       P[6][2] = { { 0, 0 }, { W, 0 }, { W, H }, { 0, 0 }, { W, H }, { 0, H } };
    for (int i = 0; i < 6; i++) {
        q[i]    = vtx(P[i][0], P[i][1], 0, 0, 0, FM_RGB(255, 255, 255));
        q[i].nx = n.x, q[i].ny = n.y, q[i].nz = n.z;
    }
    fm3d_draw(c, q, 6);
}

/* equilateral triangle of circumradius r around (cx, cy) at z = 0: with the
 * light on the axis through the center every vertex gets the same value */
static void lit_tri(fm3d_ctx* c, float cx, float cy, float r)
{
    fm3d_vertex q[3];
    for (int i = 0; i < 3; i++) {
        float a = 2.0943951f * (float)i;
        q[i]    = vtx(cx + r * cosf(a), cy + r * sinf(a), 0, 0, 0, FM_RGB(255, 255, 255));
        q[i].nx = 0, q[i].ny = 0, q[i].nz = 1;
    }
    fm3d_draw(c, q, 3);
}

static int near8(uint32_t v, int r, int g, int b)
{
    int dr = (int)((v >> 16) & 255) - r, dg = (int)((v >> 8) & 255) - g, db = (int)(v & 255) - b;
    return abs(dr) <= 1 && abs(dg) <= 1 && abs(db) <= 1;
}

static void lighting_setup(fm3d_ctx* c, fm_surface* fb)
{
    fm3d_set_target(c, fb, NULL);
    /* eye at the origin looking down -z: pixel space quad at z = -5 */
    fm_mat4 p = fm_ortho(0, (float)W, (float)H, 0, 1, 100), v = fm_translate(fm_mat4_identity(), fm_v3(0, 0, -5));
    fm_mat4 i = fm_mat4_identity();
    fm3d_set_projection(c, &p);
    fm3d_set_view(c, &v);
    fm3d_set_model(c, &i);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_clear_color(c, 0);
    fm3d_set_lighting(c, 1);
    fm3d_set_ambient_light(c, fm_v3(0, 0, 0));
    fm3d_material m = fm3d_material_default();
    m.ambient       = fm_v3(0, 0, 0);
    m.diffuse       = fm_v3(1.0f, 0.5f, 0.25f);
    m.specular      = fm_v3(0, 0, 0);
    fm3d_set_material(c, &m);
    for (int k = 0; k < FM3D_MAX_LIGHTS; k++) fm3d_set_light(c, k, NULL);
}

static void test_lighting(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();

    /* directional, head on: N.L = 1 -> the diffuse material color */
    lighting_setup(c, fb);
    fm3d_light L = fm3d_light_default(FM3D_LIGHT_DIRECTIONAL);
    L.direction  = fm_v3(0, 0, -1); /* travelling into the screen */
    fm3d_set_light(c, 0, &L);
    lit_quad(c, fm_v3(0, 0, 1));
    CHECK(near8(fm_surface_get_pixel(fb, 160, 120), 255, 128, 64), "directional head on = diffuse (got %08x)",
          fm_surface_get_pixel(fb, 160, 120));
    /* 60 degrees off: N.L = 0.5 */
    L.direction = fm_v3(0, -0.8660254f, -0.5f);
    fm3d_set_light(c, 0, &L);
    lit_quad(c, fm_v3(0, 0, 1));
    CHECK(near8(fm_surface_get_pixel(fb, 160, 120), 128, 64, 32), "directional at 60 degrees = half (got %08x)",
          fm_surface_get_pixel(fb, 160, 120));
    /* from behind: nothing but ambient (0) */
    L.direction = fm_v3(0, 0, 1);
    fm3d_set_light(c, 0, &L);
    lit_quad(c, fm_v3(0, 0, 1));
    CHECK(near8(fm_surface_get_pixel(fb, 160, 120), 0, 0, 0), "light from behind the surface");

    /* point light 4 units in front of pixel (160, 120), 1 / d^2 attenuation */
    lighting_setup(c, fb);
    fm3d_light P = fm3d_light_default(FM3D_LIGHT_POINT);
    P.position   = fm_v3(160, 120, 4);
    P.constant   = 0, P.quadratic = 1.0f / 16.0f; /* att = 1 at d = 4 */
    fm3d_set_light(c, 0, &P);
    lit_tri(c, 160, 120, 30); /* vertices: d^2 = 16 + 900, N.L = 4 / d, att = 16 / d^2 */
    uint32_t pt = fm_surface_get_pixel(fb, 160, 120);
    double   d2 = 16.0 + 900.0, e = (4.0 / sqrt(d2)) * (16.0 / d2);
    CHECK(near8(pt, (int)(e * 255 + 0.5), (int)(e * 127.5 + 0.5), (int)(e * 63.75 + 0.5)),
          "point light attenuation + angle (got %08x, want %.1f)", pt, e * 255);

    /* spot light: inside the cone lit, outside ambient only */
    lighting_setup(c, fb);
    fm3d_light S = fm3d_light_default(FM3D_LIGHT_SPOT);
    S.position   = fm_v3(160.5f, 120.5f, 10);
    S.direction  = fm_v3(0, 0, -1);
    S.spot_cutoff = 0.2f; /* 10 units above the quad */
    fm3d_set_light(c, 0, &S);
    lit_tri(c, 160.5f, 120.5f, 1.5f); /* vertices 0.15 rad off the axis: inside */
    lit_tri(c, 200.5f, 120.5f, 1.5f); /* ~1.33 rad off: outside */
    double sn = 10.0 / sqrt(100.0 + 1.5 * 1.5); /* N.L at the vertices */
    CHECK(near8(fm_surface_get_pixel(fb, 160, 120), (int)(sn * 255 + 0.5), (int)(sn * 127.5 + 0.5), (int)(sn * 63.75 + 0.5)) &&
              near8(fm_surface_get_pixel(fb, 200, 120), 0, 0, 0),
          "spot light cone (%08x inside, %08x outside)", fm_surface_get_pixel(fb, 160, 120), fm_surface_get_pixel(fb, 200, 120));

    /* specular: material specular white, highlight straight on */
    lighting_setup(c, fb);
    fm3d_material m = fm3d_material_default();
    m.ambient = fm_v3(0, 0, 0), m.diffuse = fm_v3(0, 0, 0), m.specular = fm_v3(1, 1, 1), m.shininess = 20;
    fm3d_set_material(c, &m);
    fm3d_light D = fm3d_light_default(FM3D_LIGHT_DIRECTIONAL);
    D.direction  = fm_v3(0, 0, -1);
    fm3d_set_light(c, 0, &D);
    lit_tri(c, 0.5f, 0.5f, 0.4f);     /* next to the eye axis: N.H ~ 1 */
    lit_tri(c, 300.5f, 220.5f, 1.5f); /* far off the axis: dim */
    uint32_t spot = fm_surface_get_pixel(fb, 0, 0), side = fm_surface_get_pixel(fb, 300, 220);
    CHECK((spot & 255) > 200 && (side & 255) < (spot & 255), "specular highlight where N.H = 1 (%08x vs %08x)", spot, side);

    /* color material: the vertex color replaces ambient + diffuse */
    lighting_setup(c, fb);
    fm3d_set_light(c, 0, &D);
    fm3d_set_color_material(c, 1);
    {
        fm3d_vertex q[3] = { vtx(0, 0, 0, 0, 0, FM_RGB(0, 200, 100)), vtx(W, 0, 0, 0, 0, FM_RGB(0, 200, 100)),
                             vtx(0, H, 0, 0, 0, FM_RGB(0, 200, 100)) };
        for (int i = 0; i < 3; i++) q[i].nx = 0, q[i].ny = 0, q[i].nz = 1;
        fm3d_draw(c, q, 3);
    }
    CHECK(near8(fm_surface_get_pixel(fb, 20, 20), 0, 200, 100), "color material");
    fm3d_set_color_material(c, 0);

    /* normals under non uniform scale: inverse transpose. Normal (1, 0, 1)
     * with the model scaled 2x in x must light like (0.5, 0, 1). */
    lighting_setup(c, fb);
    fm3d_set_light(c, 0, &D);
    fm_mat4 sc = fm_scale(fm_mat4_identity(), fm_v3(2, 1, 1));
    fm3d_set_model(c, &sc);
    lit_quad(c, fm_v3(1, 0, 1));
    double want = 1.0 / sqrt(1.25); /* (0.5, 0, 1) normalized . (0, 0, 1) */
    CHECK(near8(fm_surface_get_pixel(fb, 20, 20), (int)(want * 255 + 0.5), (int)(want * 127.5 + 0.5), (int)(want * 63.75 + 0.5)),
          "normal matrix under non uniform scale (got %08x)", fm_surface_get_pixel(fb, 20, 20));

    fm_surface* fb2 = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    /* immediate == deferred on a pool, several lights of every type */
    fm_executor* ex = fm_executor_create(4);
    for (int mode = 0; mode < 2; mode++) {
        fm_surface* t = mode ? fb2 : fb;
        lighting_setup(c, t);
        fm3d_set_deferred(c, mode);
        fm3d_set_executor(c, mode ? ex : NULL);
        fm3d_set_ambient_light(c, fm_v3(0.1f, 0.1f, 0.15f));
        fm3d_set_material(c, &m);
        fm3d_set_light(c, 0, &L);
        fm3d_set_light(c, 1, &P);
        fm3d_set_light(c, 2, &S);
        for (int k = 0; k < 12; k++) {
            fm_mat4 mm = fm_rotate(fm_translate(fm_mat4_identity(), fm_v3(20.0f + 25.0f * (float)k, 100, 0)), 0.3f * (float)k,
                                   fm_v3(0.2f, 1, 0.1f));
            mm = fm_scale(mm, fm_v3(0.2f, 0.4f, 0.3f));
            fm3d_set_model(c, &mm);
            lit_quad(c, fm_v3(0.1f * (float)k, 1, 0.5f));
        }
        fm3d_flush(c);
    }
    fm3d_set_deferred(c, 0);
    CHECK(diff_count(fb, fb2) == 0, "lighting: deferred on a pool = immediate");
    CHECK(count_nonzero(fb) > 0, "lighting scene drew something");
    fm_executor_destroy(ex);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
    fm_surface_destroy(fb2);
}
#endif

#if FM_FEATURE_SHADERS
/* ---- programmable stages ---- */

typedef struct sh_uniforms {
    fm_mat4  mvp;
    float    tint[4];
} sh_uniforms;

/* the fixed vertex stage's math, as a shader (fm3d_vertex input) */
static void sh_vs_fixed_like(const fm3d_vs_io* io)
{
    const sh_uniforms* U = (const sh_uniforms*)io->uniforms;
    const fm_mat4*     m = &U->mvp;
    for (int i = 0; i < io->count; i++) {
        const fm3d_vertex* v = (const fm3d_vertex*)((const char*)io->vertices + (size_t)i * (size_t)io->stride);
        float*             o = io->pos + (size_t)i * (size_t)io->out_stride;
        float*             q = io->varyings + (size_t)i * (size_t)io->out_stride;
        float              x = v->x, y = v->y, z = v->z;
        o[0] = m->c[0].x * x + m->c[1].x * y + m->c[2].x * z + m->c[3].x;
        o[1] = m->c[0].y * x + m->c[1].y * y + m->c[2].y * z + m->c[3].y;
        o[2] = m->c[0].z * x + m->c[1].z * y + m->c[2].z * z + m->c[3].z;
        o[3] = m->c[0].w * x + m->c[1].w * y + m->c[2].w * z + m->c[3].w;
        q[0] = v->u;
        q[1] = v->v;
        q[2] = (float)((v->color >> 16) & 255) * (1.0f / 255.0f);
        q[3] = (float)((v->color >> 8) & 255) * (1.0f / 255.0f);
        q[4] = (float)(v->color & 255) * (1.0f / 255.0f);
        q[5] = (float)(v->color >> 24) * (1.0f / 255.0f);
    }
}

/* interpolated vertex color */
static void sh_fs_color(const fm3d_fs_io* io)
{
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++)
        for (int k = 0; k < 4; k++) io->out[k][i] = io->varyings[2 + k][i];
}

/* color times the uniform tint */
static void sh_fs_tint(const fm3d_fs_io* io)
{
    const sh_uniforms* U = (const sh_uniforms*)io->uniforms;
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++)
        for (int k = 0; k < 4; k++) io->out[k][i] = io->varyings[2 + k][i] * U->tint[k];
}

/* discard the left half of the screen */
static void sh_fs_discard_left(const fm3d_fs_io* io)
{
    sh_fs_color(io);
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++)
        if (io->x + (i % FM3D_BATCH_COLS) < W / 2) io->mask[i] = 0;
}

/* texture through fm3d_sample */
static void sh_fs_sample(const fm3d_fs_io* io)
{
    fm3d_sample(io->texture, io->sampler, io->varyings[0], io->varyings[1], FM3D_BATCH_PIXELS, io->out[0], io->out[1],
                io->out[2], io->out[3]);
}

/* a custom vertex layout: 2D position + packed color, pixel space via uniforms */
typedef struct sh_vert2 {
    float    x, y;
    uint32_t rgba;
} sh_vert2;
static void sh_vs_2d(const fm3d_vs_io* io)
{
    const sh_uniforms* U = (const sh_uniforms*)io->uniforms;
    for (int i = 0; i < io->count; i++) {
        const sh_vert2* v = (const sh_vert2*)((const char*)io->vertices + (size_t)i * (size_t)io->stride);
        fm_vec4         p = fm_mat4_mul_vec4(U->mvp, fm_v4(v->x, v->y, 0, 1));
        float*          o = io->pos + (size_t)i * (size_t)io->out_stride;
        float*          q = io->varyings + (size_t)i * (size_t)io->out_stride;
        o[0] = p.x, o[1] = p.y, o[2] = p.z, o[3] = p.w;
        q[0] = q[1] = 0;
        q[2] = (float)(v->rgba & 255) / 255.0f; /* r g b a byte order */
        q[3] = (float)((v->rgba >> 8) & 255) / 255.0f;
        q[4] = (float)((v->rgba >> 16) & 255) / 255.0f;
        q[5] = (float)(v->rgba >> 24) / 255.0f;
    }
}

/* derivatives: u = x / W and v = y / H over a pixel space quad, so
 * ddx(u) = 1 / W and ddy(v) = 1 / H everywhere (written scaled to 0.4) */
static void sh_fs_deriv(const fm3d_fs_io* io)
{
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++) {
        io->out[0][i] = fm3d_ddx(io->varyings[0], i) * (float)W * 0.4f;
        io->out[1][i] = fm3d_ddy(io->varyings[1], i) * (float)H * 0.4f;
        io->out[2][i] = fm3d_ddy(io->varyings[0], i) * 1000.0f; /* 0 */
        io->out[3][i] = 1.0f;
    }
}

/* mipmapped sampling through fm3d_sample_batch */
static void sh_fs_sample_batch(const fm3d_fs_io* io)
{
    fm3d_sample_batch(io, io->texture, io->sampler, io->varyings[0], io->varyings[1], io->out[0], io->out[1], io->out[2],
                      io->out[3]);
}

static void sh_scene(fm3d_ctx* c, const fm3d_vertex* tri, int n)
{
    fm3d_clear_color(c, FM_RGB(5, 5, 5));
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw(c, tri, n);
}

static void test_shaders(void)
{
    fm_surface* ref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* out = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb  = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c   = fm3d_create();
    fm3d_vertex tri[9] = { vtx(10, 10, 0.2f, 0, 0, FM_RGB(255, 0, 0)),   vtx(300, 30, 0.5f, 1, 0, FM_RGB(0, 255, 0)),
                           vtx(40, 230, 0.8f, 0, 1, FM_RGBA(0, 0, 255, 200)), vtx(200, 5, 0.1f, 0, 0, FM_RGB(255, 255, 0)),
                           vtx(310, 220, 0.9f, 1, 1, FM_RGB(0, 255, 255)), vtx(120, 200, 0.3f, 1, 0, FM_RGB(255, 0, 255)),
                           vtx(0, 120, 0.4f, 0, 0, FM_RGB(90, 30, 200)),  vtx(160, 0, 0.6f, 1, 0, FM_RGB(20, 200, 90)),
                           vtx(320, 240, 0.7f, 1, 1, FM_RGBA(200, 90, 20, 128)) };
    fm3d_set_target(c, ref, zb);
    pixel_space(c);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_set_depth_test(c, FM3D_LEQUAL, 1);
    sh_uniforms U;
    memset(&U, 0, sizeof(U));
    fm_mat4 proj = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
    U.mvp        = proj; /* view = model = identity */
    for (int k = 0; k < 4; k++) U.tint[k] = 1.0f;

    /* 1. vs + fs reproducing the fixed pipeline: identical pixels */
    sh_scene(c, tri, 9);
    fm3d_program pr = { sh_vs_fixed_like, sh_fs_color, 6, 0, NULL, 0, 0, 0, 0, 0 };
    fm3d_set_program(c, &pr);
    fm3d_set_uniforms(c, &U, sizeof(U));
    fm3d_set_target(c, out, zb);
    sh_scene(c, tri, 9);
    CHECK(diff_count(ref, out) == 0, "shader program reproducing the fixed pipeline: identical pixels");

    /* 2. custom vs + fixed fragment stage (textured) = fixed pipeline */
    fm_surface*   img = fm_surface_create(16, 16, FM_FORMAT_ARGB32);
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) fm_surface_row32(img, y)[x] = ((x ^ y) & 4) ? FM_RGB(240, 200, 30) : FM_RGB(30, 60, 200);
    fm3d_texture* tex = fm3d_texture_create(img, 0);
    fm3d_set_texture(c, tex, NULL);
    fm3d_set_program(c, NULL);
    fm3d_set_target(c, ref, zb);
    sh_scene(c, tri, 9);
    fm3d_program vs_only = { sh_vs_fixed_like, NULL, 6, 0, NULL, 0, 0, 0, 0, 0 };
    fm3d_set_program(c, &vs_only);
    fm3d_set_target(c, out, zb);
    sh_scene(c, tri, 9);
    CHECK(diff_count(ref, out) == 0, "custom vertex shader + fixed textured fragment stage = fixed pipeline");

    /* 3. fixed vs + fragment shader sampling a solid texture */
    fm_surface_clear(img, FM_RGB(12, 150, 222));
    fm3d_texture* solid = fm3d_texture_create(img, 0);
    fm3d_set_texture(c, solid, NULL);
    fm3d_program fs_only = { NULL, sh_fs_sample, 0, 0, NULL, 0, 0, 0, 0, 0 };
    fm3d_set_program(c, &fs_only);
    sh_scene(c, tri, 3);
    CHECK(fm_surface_get_pixel(out, 60, 40) == FM_RGB(12, 150, 222), "fragment shader + fm3d_sample (got %08x)",
          fm_surface_get_pixel(out, 60, 40));
    fm3d_set_texture(c, NULL, NULL);

    /* 4. custom vertex layout */
    sh_vert2 q[3] = { { 20, 20, 0xff4080c0u }, { 300, 20, 0xff4080c0u }, { 20, 220, 0xff4080c0u } };
    fm3d_program p2 = { sh_vs_2d, sh_fs_color, 6, 0, NULL, 0, 0, 0, 0, 0 };
    fm3d_set_program(c, &p2);
    fm3d_clear_color(c, 0);
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw_vertices(c, q, (int)sizeof(sh_vert2), 3, NULL, 3);
    CHECK(fm_surface_get_pixel(out, 40, 40) == FM_RGB(0xc0, 0x80, 0x40), "custom vertex layout (got %08x)",
          fm_surface_get_pixel(out, 40, 40));

    /* 5. discard: no color, no depth for discarded pixels (this ortho maps larger z nearer) */
    fm3d_program pd = { sh_vs_fixed_like, sh_fs_discard_left, 6, 1, NULL, 0, 0, 0, 0, 0 };
    fm3d_set_program(c, &pd);
    fm3d_clear_color(c, 0);
    fm3d_clear_depth(c, 1.0f);
    fm3d_vertex nearq[6] = { vtx(0, 0, 0.5f, 0, 0, FM_RGB(255, 0, 0)), vtx(W, 0, 0.5f, 0, 0, FM_RGB(255, 0, 0)),
                             vtx(W, H, 0.5f, 0, 0, FM_RGB(255, 0, 0)), vtx(0, 0, 0.5f, 0, 0, FM_RGB(255, 0, 0)),
                             vtx(W, H, 0.5f, 0, 0, FM_RGB(255, 0, 0)), vtx(0, H, 0.5f, 0, 0, FM_RGB(255, 0, 0)) };
    fm3d_draw(c, nearq, 6);
    fm3d_set_program(c, &pr);
    fm3d_vertex farq[6];
    for (int i = 0; i < 6; i++) farq[i] = nearq[i], farq[i].z = 0.1f, farq[i].color = FM_RGB(0, 0, 255);
    fm3d_draw(c, farq, 6);
    CHECK(fm_surface_get_pixel(out, 40, 100) == FM_RGB(0, 0, 255) && fm_surface_get_pixel(out, 280, 100) == FM_RGB(255, 0, 0),
          "discard writes neither color nor depth (%08x left, %08x right)", fm_surface_get_pixel(out, 40, 100),
          fm_surface_get_pixel(out, 280, 100));

    /* 5b. screen space derivatives inside a batch */
    {
        fm3d_vertex dq[6];
        float       P[6][2] = { { 0, 0 }, { W, 0 }, { W, H }, { 0, 0 }, { W, H }, { 0, H } };
        for (int i = 0; i < 6; i++) dq[i] = vtx(P[i][0], P[i][1], 0, P[i][0] / W, P[i][1] / H, FM_RGB(255, 255, 255));
        fm3d_program pdv = { NULL, sh_fs_deriv, 0, 0, NULL, 0, 0, 0, 0, 0 };
        fm3d_set_program(c, &pdv);
        fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
        fm3d_clear_color(c, 0);
        fm3d_draw(c, dq, 6);
        int bad = 0;
        for (int y = 0; y < H; y += 7)
            for (int x = 0; x < W; x += 5) bad += fm_surface_get_pixel(out, x, y) != FM_RGB(102, 102, 0);
        CHECK(bad == 0, "fm3d_ddx / fm3d_ddy give the screen derivatives (%d wrong, e.g. %08x)", bad,
              fm_surface_get_pixel(out, 13, 17));

        /* 5c. fm3d_sample_batch: trilinear like the fixed pipeline on a minified checker */
        fm_surface* ck = fm_surface_create(256, 256, FM_FORMAT_ARGB32);
        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++) fm_surface_row32(ck, y)[x] = ((x ^ y) & 2) ? FM_RGB(250, 250, 250) : FM_RGB(10, 10, 10);
        fm3d_texture* ct = fm3d_texture_create(ck, 1);
        fm3d_sampler  ts = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
        fm3d_vertex   mq[6];
        for (int i = 0; i < 6; i++) mq[i] = vtx(P[i][0], P[i][1], 0, P[i][0] / W * 3.0f, P[i][1] / H * 2.0f, FM_RGB(255, 255, 255));
        fm3d_set_texture(c, ct, &ts);
        fm3d_set_program(c, NULL);
        fm3d_set_target(c, ref, zb);
        fm3d_clear_color(c, 0);
        fm3d_draw(c, mq, 6);
        fm3d_program psb = { NULL, sh_fs_sample_batch, 0, 0, NULL, 0, 0, 0, 0, 0 };
        fm3d_set_program(c, &psb);
        fm3d_set_target(c, out, zb);
        fm3d_clear_color(c, 0);
        fm3d_draw(c, mq, 6);
        double sum = 0;
        int    maxd = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int d = abs((int)(fm_surface_get_pixel(ref, x, y) & 255) - (int)(fm_surface_get_pixel(out, x, y) & 255));
                sum += d;
                maxd = d > maxd ? d : maxd;
            }
        double mean = sum / (W * H);
        /* a 256^2 checker of 2 texel squares at 3x2 repeats on 320x240 is
         * minified ~2.4x: without mip selection the result would alias to
         * black / white pixels (measured: mean difference 43 with level 0 only) */
        CHECK(mean < 3.0 && maxd < 40, "fm3d_sample_batch trilinear ~ fixed pipeline (mean %.2f, max %d)", mean, maxd);
        fm3d_set_texture(c, NULL, NULL);
        fm3d_texture_release(ct);
        fm_surface_destroy(ck);
        fm3d_set_depth_test(c, FM3D_LEQUAL, 1);
    }

    /* 6. deferred on a pool = immediate, uniforms changing between draws */
    fm_executor* ex = fm_executor_create(4);
    fm3d_program pt = { sh_vs_fixed_like, sh_fs_tint, 6, 0, NULL, 0, 0, 0, 0, 0 };
    for (int mode = 0; mode < 2; mode++) {
        fm3d_set_deferred(c, mode);
        fm3d_set_executor(c, mode ? ex : NULL);
        fm3d_set_program(c, &pt);
        fm3d_set_target(c, mode ? out : ref, zb);
        fm3d_clear_color(c, 0);
        fm3d_clear_depth(c, 1.0f);
        for (int k = 0; k < 3; k++) {
            U.tint[0] = 1.0f - 0.3f * (float)k, U.tint[1] = 0.4f + 0.2f * (float)k;
            fm3d_set_uniforms(c, &U, sizeof(U)); /* copied: the struct is reused */
            fm3d_draw(c, tri + 3 * k, 3);
        }
        fm3d_flush(c);
    }
    fm3d_set_deferred(c, 0);
    CHECK(diff_count(ref, out) == 0, "shaders: deferred on a pool with per draw uniforms = immediate");
    CHECK(count_nonzero(ref) > 0, "shader scene drew something");
    fm3d_set_program(c, NULL);

    fm_executor_destroy(ex);
    fm3d_texture_release(tex);
    fm3d_texture_release(solid);
    fm_surface_destroy(img);
    fm3d_destroy(c);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
}
#endif

#if FM_FEATURE_SPIRV
/* ---- SPIR-V: the interpreter against C shaders doing the same math ---- */
#include "spirv_shaders.h"
#include <math.h>

typedef struct sv_tvert {
    float pos[3];
    float color[4];
} sv_tvert;

typedef struct sv_tu { /* std140: mat4 @0, vec4 @64, float @80 */
    fm_mat4 mvp;
    float   tint[4];
    float   time;
    float   pad[3];
} sv_tu;

/* t_basic.vert: varyings loc 0 vec4 color -> slots 0..3, loc 1 vec2 uv -> 4..5 */
static void svc_vs(const fm3d_vs_io* io)
{
    const sv_tu* U = (const sv_tu*)io->uniforms;
    const float* m = &U->mvp.c[0].x;
    for (int i = 0; i < io->count; i++) {
        const sv_tvert* v = (const sv_tvert*)((const char*)io->vertices + (size_t)i * (size_t)io->stride);
        float*          o = io->pos + (size_t)i * (size_t)io->out_stride;
        float*          q = io->varyings + (size_t)i * (size_t)io->out_stride;
        float           in4[4] = { v->pos[0], v->pos[1], v->pos[2], 1.0f };
        for (int r = 0; r < 4; r++) {
            float d = m[r] * in4[0];
            for (int c = 1; c < 4; c++) d += m[c * 4 + r] * in4[c];
            o[r] = d;
        }
        for (int k = 0; k < 4; k++) q[k] = v->color[k];
        q[4] = v->pos[0] * 0.01f, q[5] = v->pos[1] * 0.01f;
    }
}

/* t_fixedfs.vert: varyings u, v, r, g, b, a for the fixed fragment stage */
static void svc_vs_fixedfs(const fm3d_vs_io* io)
{
    svc_vs(io);
    for (int i = 0; i < io->count; i++) {
        float* q = io->varyings + (size_t)i * (size_t)io->out_stride;
        float  c[4] = { q[0], q[1], q[2], q[3] }, u = q[4], v = q[5];
        q[0] = u, q[1] = v;
        for (int k = 0; k < 4; k++) q[2 + k] = c[k];
    }
}

static void svc_fs_color(const fm3d_fs_io* io)
{
    const sv_tu* U = (const sv_tu*)io->uniforms;
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++)
        for (int k = 0; k < 4; k++) io->out[k][i] = io->varyings[k][i] * U->tint[k];
}

static void svc_fs_fixedvs(const fm3d_fs_io* io)
{
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++) {
        for (int k = 0; k < 3; k++) io->out[k][i] = io->varyings[2 + k][i] * 0.5f;
        io->out[3][i] = io->varyings[5][i];
    }
}

/* t_control.frag, operation for operation */
static void svc_fs_control(const fm3d_fs_io* io)
{
    const sv_tu* U = (const sv_tu*)io->uniforms;
    float        tr[64], tg[64], tb[64], ta[64];
    fm3d_sample_batch(io, io->textures[1], &io->samplers[1], io->varyings[4], io->varyings[5], tr, tg, tb, ta);
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++) {
        if ((i & 31) >= io->cols) continue;
        float col[4] = { io->varyings[0][i], io->varyings[1][i], io->varyings[2][i], io->varyings[3][i] };
        float uvx = io->varyings[4][i], uvy = io->varyings[5][i];
        float c[3];
        for (int k = 0; k < 3; k++) c[k] = col[k] * U->tint[k];
        float acc = 0.0f;
        for (int k = 0; k < 4; k++) acc += fm_sinf((float)k * uvx + U->time);
        if (col[0] > 0.5f) {
            float y = acc * 0.25f;
            for (int k = 0; k < 3; k++) c[k] = c[k] * (1.0f - 0.5f) + y * 0.5f;
        } else {
            for (int k = 0; k < 3; k++) c[k] = fm_powf(c[k], 2.2f);
        }
        if (uvy > 1.9f) {
            io->mask[i] = 0;
            continue;
        }
        float w = fabsf(fm3d_ddx(io->varyings[4], i)) + fabsf(fm3d_ddy(io->varyings[4], i));
        float t[3] = { tr[i], tg[i], tb[i] };
        for (int k = 0; k < 3; k++) io->out[k][i] = fminf(fmaxf((c[k] + t[k] * 0.1f) + w, 0.0f), 1.0f);
        io->out[3][i] = col[3];
    }
}

/* t_switch.frag */
static void svc_fs_switch(const fm3d_fs_io* io)
{
    for (int i = 0; i < FM3D_BATCH_PIXELS; i++) {
        float ux = io->varyings[4][i], uy = io->varyings[5][i];
        int   k  = (int)(ux * 4.0f) & 3;
        float px = ux * 3.0f - floorf(ux * 3.0f), py = uy * 3.0f - floorf(uy * 3.0f);
        float c[3];
        if (px < 0.5f) c[0] = 1, c[1] = 0, c[2] = 0;
        else if (py < 0.5f) c[0] = 0, c[1] = 1, c[2] = 0;
        else c[0] = 0, c[1] = 0, c[2] = 1;
        switch (k) {
        case 0: for (int j = 0; j < 3; j++) c[j] *= 0.5f; break;
        case 1: for (int j = 0; j < 3; j++) c[j] += 0.25f; /* fall through */
        case 2: { float t = c[0]; c[0] = c[2]; c[2] = t; } break;
        default: for (int j = 0; j < 3; j++) c[j] = 1.0f - c[j]; break;
        }
        for (int j = 0; j < 3; j++) io->out[j][i] = c[j];
        io->out[3][i] = 1.0f;
    }
}

static void sv_scene_draw(fm3d_ctx* c, const sv_tvert* v, int n)
{
    fm3d_clear_color(c, FM_RGB(3, 4, 5));
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw_vertices(c, v, (int)sizeof(sv_tvert), n, NULL, n);
    fm3d_flush(c);
}

static void sv_scene_fixed(fm3d_ctx* c, const sv_tvert* v, int n)
{
    fm3d_vertex fv[12];
    for (int i = 0; i < n; i++) {
        fv[i] = vtx(v[i].pos[0], v[i].pos[1], v[i].pos[2], v[i].pos[0] * 0.01f, v[i].pos[1] * 0.01f,
                    FM_RGBA((uint8_t)(v[i].color[0] * 255.0f + 0.5f), (uint8_t)(v[i].color[1] * 255.0f + 0.5f),
                            (uint8_t)(v[i].color[2] * 255.0f + 0.5f), (uint8_t)(v[i].color[3] * 255.0f + 0.5f)));
    }
    fm3d_clear_color(c, FM_RGB(3, 4, 5));
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw(c, fv, n);
    fm3d_flush(c);
}

static void test_spirv(void)
{
    fm_surface* ref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* out = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb  = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c   = fm3d_create();
    /* colors are multiples of 1/255 so the fixed stage (8 bit vertex colors) sees the same values */
    sv_tvert v[9] = { { { 10, 10, 0.2f }, { 1, 0.2f, 0.2f, 1 } },          { { 300, 30, 0.5f }, { 0.2f, 1, 0.2f, 1 } },
                      { { 40, 230, 0.8f }, { 0.2f, 0.2f, 1, 0.8f } },      { { 200, 5, 0.1f }, { 1, 1, 0.2f, 1 } },
                      { { 310, 220, 0.9f }, { 0.2f, 1, 1, 1 } },           { { 120, 200, 0.3f }, { 1, 0.2f, 1, 1 } },
                      { { 0, 120, 0.4f }, { 0.6f, 0.2f, 0.8f, 1 } },       { { 160, 0, 0.6f }, { 0.4f, 0.8f, 0.6f, 1 } },
                      { { 320, 240, 0.7f }, { 0.8f, 0.6f, 0.4f, 0.6f } } };
    fm3d_vertex_attrib attr[2] = { { 0, 3, 0 }, { 1, 4, 12 } };
    sv_tu       U;
    memset(&U, 0, sizeof(U));
    U.mvp = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
    U.tint[0] = 1.0f, U.tint[1] = 0.8f, U.tint[2] = 0.6f, U.tint[3] = 1.0f;
    U.time = 0.7f;
    fm_surface* ck = fm_surface_create(64, 64, FM_FORMAT_ARGB32);
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) fm_surface_row32(ck, y)[x] = ((x ^ y) & 4) ? FM_RGB(240, 200, 30) : FM_RGB(30, 60, 200);
    fm3d_texture* tex = fm3d_texture_create(ck, 1);
    fm3d_sampler  ts  = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture_unit(c, 1, tex, &ts);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_set_depth_test(c, FM3D_LEQUAL, 1);
    fm3d_set_uniforms(c, &U, sizeof(U));
    char err[256];

    /* 1. vertex + fragment SPIR-V = the same C program */
    fm3d_spirv* p1 = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_color_frag,
                                       sizeof(spv_t_color_frag) / 4, attr, 2, err, sizeof(err));
    CHECK(p1 != NULL, "spirv basic program: %s", err);
    if (p1) {
        fm3d_program cp = { svc_vs, svc_fs_color, 6, 0, NULL, 0, 0, 0, 0, 0 }, sp = fm3d_spirv_program(p1);
        CHECK(sp.nvaryings == 6, "spirv varyings linked (%d)", sp.nvaryings);
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &cp);
        sv_scene_draw(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &sp);
        sv_scene_draw(c, v, 9);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "spirv vs + fs = C shaders (%d rows differ)", diff_count(ref, out));
    }

    /* 2. control flow: loop + phis, if / else, discard, fwidth, texture() */
    fm3d_spirv* p2 = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_control_frag,
                                       sizeof(spv_t_control_frag) / 4, attr, 2, err, sizeof(err));
    CHECK(p2 != NULL, "spirv control program: %s", err);
    if (p2) {
        fm3d_program cp = { svc_vs, svc_fs_control, 6, 1, NULL, 0, 0, 0, 0, 0 }, sp = fm3d_spirv_program(p2);
        CHECK(sp.discards == 1, "spirv program with discard is flagged");
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &cp);
        sv_scene_draw(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &sp);
        sv_scene_draw(c, v, 9);
        int    nd = diff_count(ref, out), maxd = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int sh = 0; sh < 32; sh += 8) {
                    int d = abs((int)((fm_surface_get_pixel(ref, x, y) >> sh) & 255) - (int)((fm_surface_get_pixel(out, x, y) >> sh) & 255));
                    maxd  = d > maxd ? d : maxd;
                }
        CHECK(nd == 0, "spirv control flow / texture / discard = C shader (%d rows differ, max %d)", nd, maxd);
        CHECK(fm_surface_get_pixel(out, 30, 215) == FM_RGB(3, 4, 5), "spirv discard (uv.y > 1.9)");

        /* deferred on a pool = immediate */
        fm_executor* ex = fm_executor_create(4);
        fm3d_set_deferred(c, 1);
        fm3d_set_executor(c, ex);
        sv_scene_draw(c, v, 9);
        fm3d_set_deferred(c, 0);
        fm3d_set_executor(c, NULL);
        fm_executor_destroy(ex);
        fm3d_set_target(c, ref, zb);
        sv_scene_draw(c, v, 9);
        CHECK(diff_count(ref, out) == 0, "spirv: deferred on a pool = immediate");
    }

    /* 2b. switch constructs: inlined early returns + a switch with fall through */
    fm3d_spirv* p5 = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_switch_frag,
                                       sizeof(spv_t_switch_frag) / 4, attr, 2, err, sizeof(err));
    CHECK(p5 != NULL, "spirv switch program: %s", err);
    if (p5) {
        fm3d_program cp = { svc_vs, svc_fs_switch, 6, 0, NULL, 0, 0, 0, 0, 0 }, sp = fm3d_spirv_program(p5);
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &cp);
        sv_scene_draw(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &sp);
        sv_scene_draw(c, v, 9);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "spirv switch / early returns = C (%d rows differ)", diff_count(ref, out));
        fm3d_set_program(c, NULL);
        fm3d_spirv_destroy(p5);
    }

    /* 3. fixed vertex stage + SPIR-V fragment stage */
    fm3d_spirv* p3 = fm3d_spirv_create(NULL, 0, spv_t_fixedvs_frag, sizeof(spv_t_fixedvs_frag) / 4, NULL, 0, err, sizeof(err));
    CHECK(p3 != NULL, "spirv fs only: %s", err);
    if (p3) {
        fm3d_program cp = { NULL, svc_fs_fixedvs, 0, 0, NULL, 0, 0, 0, 0, 0 }, sp = fm3d_spirv_program(p3);
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &cp);
        sv_scene_fixed(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &sp);
        sv_scene_fixed(c, v, 9);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "fixed vs + spirv fs = C fs (%d rows differ)", diff_count(ref, out));
    }

    /* 4. SPIR-V vertex stage + fixed fragment stage (textured, unit 0) */
    fm3d_spirv* p4 = fm3d_spirv_create(spv_t_fixedfs_vert, sizeof(spv_t_fixedfs_vert) / 4, NULL, 0, attr, 2, err, sizeof(err));
    CHECK(p4 != NULL, "spirv vs only: %s", err);
    if (p4) {
        fm3d_set_texture(c, tex, &ts);
        fm3d_program cp = { svc_vs_fixedfs, NULL, 6, 0, NULL, 0, 0, 0, 0, 0 }, sp = fm3d_spirv_program(p4);
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &cp);
        sv_scene_draw(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &sp);
        sv_scene_draw(c, v, 9);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "spirv vs + fixed textured fs = C vs (%d rows differ)", diff_count(ref, out));
        fm3d_set_texture(c, NULL, NULL);
    }

    /* 5. rejected modules, with a message */
    uint32_t junk[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    err[0]           = 0;
    CHECK(fm3d_spirv_create(NULL, 0, junk, 8, NULL, 0, err, sizeof(err)) == NULL && err[0], "garbage module rejected (%s)", err);
    err[0] = 0;
    /* function calls (glslc -O0) are inlined: the same image as glslc -O's own inlining */
    fm3d_spirv* pf = fm3d_spirv_create(NULL, 0, spv_t_func_frag_O0, sizeof(spv_t_func_frag_O0) / 4, NULL, 0, err, sizeof(err));
    CHECK(pf != NULL, "function calls (-O0) are inlined: %s", err);
    fm3d_spirv* pi = fm3d_spirv_create(NULL, 0, spv_t_func_frag, sizeof(spv_t_func_frag) / 4, NULL, 0, err, sizeof(err));
    CHECK(pi != NULL, "the same shader compiled with -O is accepted: %s", err);
    if (pf && pi) {
        fm3d_program a = fm3d_spirv_program(pf), b = fm3d_spirv_program(pi);
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &b);
        sv_scene_fixed(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &a);
        sv_scene_fixed(c, v, 9);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "inlined calls = glslc -O (%d rows differ)", diff_count(ref, out));
        fm3d_set_program(c, NULL);
    }
    fm3d_spirv_destroy(pf);
    /* early returns from loops / branches, out / inout parameters, nested calls, calls in conditions */
    fm3d_spirv* ca = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_calls_frag_O0,
                                       sizeof(spv_t_calls_frag_O0) / 4, attr, 2, err, sizeof(err));
    CHECK(ca != NULL, "calls (-O0): %s", err);
    fm3d_spirv* cb = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_calls_frag,
                                       sizeof(spv_t_calls_frag) / 4, attr, 2, err, sizeof(err));
    CHECK(cb != NULL, "calls (-O): %s", err);
    if (ca && cb) {
        fm3d_program a = fm3d_spirv_program(ca), b = fm3d_spirv_program(cb);
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &b);
        sv_scene_draw(c, v, 9);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &a);
        sv_scene_draw(c, v, 9);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "inlined early returns / out params / nested calls = glslc -O (%d rows differ)",
              diff_count(ref, out));
        fm3d_set_program(c, NULL);
    }
    fm3d_spirv_destroy(ca);
    fm3d_spirv_destroy(cb);
    err[0] = 0;
    CHECK(fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, NULL, 0, attr, 1, err, sizeof(err)) == NULL &&
              strstr(err, "attribute"),
          "missing vertex attribute rejected (%s)", err);

    fm3d_set_program(c, NULL);
    fm3d_spirv_destroy(p1);
    fm3d_spirv_destroy(p2);
    fm3d_spirv_destroy(p3);
    fm3d_spirv_destroy(p4);
    fm3d_spirv_destroy(pi);
    fm3d_texture_release(tex);
    fm_surface_destroy(ck);
    fm3d_destroy(c);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
}

/* ---- the SPIR-V JIT (fm3d_spirv_set_jit): its reference executor (2) and
 * machine code (1) render what the interpreter (0) renders, bit for bit ---- */
#include "bfg_scene.h"

typedef struct jt_sea_u { /* std140: vec3 iResolution @0, float iTime @12, vec4 iMouse @16 */
    float res[3], time, mouse[4];
} jt_sea_u;

static void jt_quad(fm3d_ctx* c, int sw, int sh)
{
    fm_mat4 pr = fm_ortho(0, (float)W, (float)H, 0, -1, 1), id = fm_mat4_identity();
    fm3d_set_projection(c, &pr);
    fm3d_set_view(c, &id);
    fm3d_set_model(c, &id);
    fm3d_vertex q[6];
    const float X[4] = { 0, (float)sw, (float)sw, 0 }, Y[4] = { 0, 0, (float)sh, (float)sh };
    const int   o[6] = { 0, 1, 2, 0, 2, 3 };
    for (int i = 0; i < 6; i++) q[i] = vtx(X[o[i]], Y[o[i]], 0, X[o[i]] / (float)sw, Y[o[i]] / (float)sh, FM_RGB(255, 255, 255));
    fm3d_clear_color(c, FM_RGB(3, 4, 5));
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw(c, q, 6);
    fm3d_flush(c);
}

typedef struct jt_bfg_vert {
    float pos[4], st[4], nrm[4], tan[4], col[4];
} jt_bfg_vert;

/* Doom 3 BFG's light interactions (tests/bfg_scene.h), 4 additive passes */
static void jt_bfg(fm3d_ctx* c, fm3d_texture* const* tex)
{
    static jt_bfg_vert v[SCENE_NV];
    static uint32_t    idx[SCENE_NI];
    static scene_vert  sv[SCENE_NV];
    scene_mesh(sv, idx);
    for (int i = 0; i < SCENE_NV; i++) {
        memcpy(v[i].pos, sv[i].xyzw, 16);
        v[i].st[0] = sv[i].st[0], v[i].st[1] = sv[i].st[1], v[i].st[2] = 0, v[i].st[3] = 1;
        for (int k = 0; k < 4; k++)
            v[i].nrm[k] = sv[i].normal[k] / 255.0f, v[i].tan[k] = sv[i].tangent[k] / 255.0f, v[i].col[k] = sv[i].color[k] / 255.0f;
    }
    fm3d_set_origin(c, FM3D_ORIGIN_LOWER_LEFT);
    fm3d_blend_state bs = { FM3D_BF_ONE, FM3D_BF_ONE, FM3D_BF_ONE, FM3D_BF_ONE, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
    fm3d_set_blend_state(c, &bs);
    fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
    fm3d_clear_color(c, FM_RGBA(0, 0, 0, 255));
    for (int i = 0; i < 5; i++) {
        fm_wrap      wr = scene_tex_clamp(i) ? FM_WRAP_BORDER : FM_WRAP_REPEAT;
        fm3d_sampler s  = { FM3D_FILTER_TRILINEAR, wr, wr, 0, wr };
        fm3d_set_texture_unit(c, i, tex[i], &s);
    }
    for (int l = 0; l < SCENE_LIGHTS; l++) {
        float va[18][4], fa[2][4];
        scene_uniforms(l, va, fa);
        fm3d_set_uniform_block(c, 0, va, sizeof(va));
        fm3d_set_uniform_block(c, 1, fa, sizeof(fa));
        fm3d_draw_vertices(c, v, (int)sizeof(jt_bfg_vert), SCENE_NV, idx, SCENE_NI);
    }
    fm3d_flush(c);
    for (int i = 0; i < 5; i++) fm3d_set_texture_unit(c, i, NULL, NULL);
    fm3d_set_uniform_block(c, 0, NULL, 0);
    fm3d_set_uniform_block(c, 1, NULL, 0);
    fm3d_set_origin(c, FM3D_ORIGIN_UPPER_LEFT);
    fm3d_set_blend(c, FM_OP_SRC_OVER);
}

/* the SIMD quad samplers (fm3d_sample_quads at AVX2 / AVX-512) return the
 * scalar path's bits for every filter, wrap mode and texture shape */
static void test_sampler_simd(void)
{
    fm_simd_level keep = fm_simd_current();
    if (!fm_simd_set(FM_SIMD_AVX2)) {
        printf("sampler: no AVX2, skipped\n");
        return;
    }
    fm_simd_set(keep);
    static const int sizes[3][2] = { { 64, 64 }, { 37, 21 }, { 128, 16 } };
    uint32_t         seed = 12345u;
    int              bad = 0, total = 0;
    for (int sz = 0; sz < 3; sz++)
        for (int st = 0; st < 2; st++) {
            int         w = sizes[sz][0], h = sizes[sz][1];
            fm_surface* img = fm_surface_create(w, h, FM_FORMAT_ARGB32);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    seed                        = seed * 1664525u + 1013904223u;
                    uint32_t aa                 = (seed >> 24) | 0x10u;
                    fm_surface_row32(img, y)[x] = (aa << 24) | ((seed >> 8) & 0xffffffu);
                }
            const fm_surface* L = img;
            fm3d_texture* tex = fm3d_texture_create_layers(FM3D_TEX_2D, &L, 1, FM3D_TEXTURE_MIPMAPS | (st ? FM3D_TEXTURE_STRAIGHT : 0));
            for (int f = 0; f < 5; f++)
                for (int wr = 0; wr < 4; wr++)
                    for (int rep = 0; rep < 40; rep++) {
                        fm3d_sampler sm = { (fm3d_filter)f, (fm_wrap)wr, (fm_wrap)((wr + rep) & 3), (float)(rep % 5) * 0.25f - 0.5f, FM_WRAP_CLAMP };
                        float        U[64], V[64], o[3][4][64];
                        /* quads with a random footprint each (scale 0.01 .. 8 texels per pixel), around [-1, 2] */
                        for (int q = 0; q < 16; q++) {
                            seed     = seed * 1664525u + 1013904223u;
                            float u0 = (float)(seed >> 8) / 16777216.0f * 3.0f - 1.0f;
                            seed     = seed * 1664525u + 1013904223u;
                            float v0 = (float)(seed >> 8) / 16777216.0f * 3.0f - 1.0f;
                            seed     = seed * 1664525u + 1013904223u;
                            float sc = (0.01f + (float)(seed >> 24) / 32.0f) / (float)w;
                            int   b  = (q >> 2) * 16 + (q & 3) * 2;
                            U[b] = u0, U[b + 1] = u0 + sc, U[b + 8] = u0 + 0.3f * sc, U[b + 9] = u0 + 1.3f * sc;
                            V[b] = v0, V[b + 1] = v0 + 0.2f * sc, V[b + 8] = v0 + sc, V[b + 9] = v0 + 1.2f * sc;
                        }
                        int nq = rep % 7 == 6 ? 9 : 16;
                        for (int k = 0; k < 3; k++) {
                            memset(o[k], 0, sizeof(o[k]));
                            if (k == 2 && !fm_simd_set(FM_SIMD_AVX512)) {
                                memcpy(o[2], o[1], sizeof(o[2])); /* no AVX-512 here */
                                continue;
                            }
                            if (k < 2) fm_simd_set(k ? FM_SIMD_AVX2 : FM_SIMD_SSE2);
                            fm3d_sample_quads(tex, &sm, U, V, nq, o[k][0], o[k][1], o[k][2], o[k][3]);
                        }
                        for (int q = 0; q < nq; q++) {
                            int b = (q >> 2) * 16 + (q & 3) * 2, li[4] = { b, b + 1, b + 8, b + 9 };
                            for (int j = 0; j < 4; j++)
                                for (int c = 0; c < 4; c++) {
                                    total++;
                                    if (memcmp(&o[0][c][li[j]], &o[1][c][li[j]], 4) || memcmp(&o[0][c][li[j]], &o[2][c][li[j]], 4)) {
                                        if (bad < 4)
                                            printf("  sampler %dx%d %s filter %d wrap %d/%d lane %d ch %d: %.9g vs %.9g\n", w, h, st ? "straight" : "premul", f,
                                                   wr, sm.wrap_v, li[j], c, (double)o[0][c][li[j]], (double)o[memcmp(&o[0][c][li[j]], &o[1][c][li[j]], 4) ? 1 : 2][c][li[j]]);
                                        bad++;
                                    }
                                }
                        }
                    }
            fm3d_texture_release(tex);
            fm_surface_destroy(img);
        }
    fm_simd_set(keep);
    CHECK(bad == 0, "SIMD quad sampler = scalar sampler (%d of %d values differ)", bad, total);
}

static void test_jit(void)
{
    fm_surface* ref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* out = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb  = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c   = fm3d_create();
    sv_tvert v[9] = { { { 10, 10, 0.2f }, { 1, 0.2f, 0.2f, 1 } },          { { 300, 30, 0.5f }, { 0.2f, 1, 0.2f, 1 } },
                      { { 40, 230, 0.8f }, { 0.2f, 0.2f, 1, 0.8f } },      { { 200, 5, 0.1f }, { 1, 1, 0.2f, 1 } },
                      { { 310, 220, 0.9f }, { 0.2f, 1, 1, 1 } },           { { 120, 200, 0.3f }, { 1, 0.2f, 1, 1 } },
                      { { 0, 120, 0.4f }, { 0.6f, 0.2f, 0.8f, 1 } },       { { 160, 0, 0.6f }, { 0.4f, 0.8f, 0.6f, 1 } },
                      { { 320, 240, 0.7f }, { 0.8f, 0.6f, 0.4f, 0.6f } } };
    fm3d_vertex_attrib attr[2] = { { 0, 3, 0 }, { 1, 4, 12 } };
    static const fm3d_vertex_attrib bfg_attr[5] = { { 0, 4, 0 }, { 1, 4, 16 }, { 2, 4, 32 }, { 3, 4, 48 }, { 4, 4, 64 } };
    sv_tu U;
    memset(&U, 0, sizeof(U));
    U.mvp = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
    U.tint[0] = 1.0f, U.tint[1] = 0.8f, U.tint[2] = 0.6f, U.tint[3] = 1.0f;
    U.time = 0.7f;
    fm_surface* ck = fm_surface_create(64, 64, FM_FORMAT_ARGB32);
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) fm_surface_row32(ck, y)[x] = ((x ^ y) & 4) ? FM_RGB(240, 200, 30) : FM_RGB(30, 60, 200);
    fm3d_texture* tex = fm3d_texture_create(ck, 1);
    fm3d_sampler  ts  = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_texture* btex[5];
    for (int i = 0; i < 5; i++) {
        int         w = scene_tex_w(i), h = scene_tex_h(i);
        uint8_t*    rgba = (uint8_t*)malloc((size_t)w * (size_t)h * 4);
        fm_surface* sf   = fm_surface_create(w, h, FM_FORMAT_ARGB32);
        scene_texture(i, rgba);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                const uint8_t* q            = rgba + ((size_t)y * (size_t)w + (size_t)x) * 4;
                fm_surface_row32(sf, y)[x] = FM_RGBA(q[0], q[1], q[2], q[3]);
            }
        const fm_surface* img = sf;
        btex[i]               = fm3d_texture_create_layers(FM3D_TEX_2D, &img, 1, FM3D_TEXTURE_STRAIGHT | FM3D_TEXTURE_MIPMAPS);
        fm_surface_destroy(sf);
        free(rgba);
    }
    char err[256];
    enum { SC_TRIS, SC_FIXED, SC_SEA, SC_BFG };
    struct {
        const char*     what;
        const uint32_t* vs;
        size_t          nvs;
        const uint32_t* fs;
        size_t          nfs;
        int             scene, fast;
    } cases[] = {
        { "vs + fs", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_color_frag, sizeof(spv_t_color_frag) / 4, SC_TRIS, 0 },
        { "loop / if / discard / texture / fwidth", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_control_frag,
          sizeof(spv_t_control_frag) / 4, SC_TRIS, 0 },
        { "loop / if / discard / texture / fwidth, fast math", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_control_frag,
          sizeof(spv_t_control_frag) / 4, SC_TRIS, 1 },
        { "switch / early returns", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_switch_frag, sizeof(spv_t_switch_frag) / 4, SC_TRIS, 0 },
        { "fixed vs + fs", NULL, 0, spv_t_fixedvs_frag, sizeof(spv_t_fixedvs_frag) / 4, SC_FIXED, 0 },
        { "vs + fixed textured fs", spv_t_fixedfs_vert, sizeof(spv_t_fixedfs_vert) / 4, NULL, 0, SC_TRIS, 0 },
        { "inlined functions", NULL, 0, spv_t_func_frag, sizeof(spv_t_func_frag) / 4, SC_FIXED, 0 },
        { "calls (O0)", NULL, 0, spv_t_calls_frag_O0, sizeof(spv_t_calls_frag_O0) / 4, SC_FIXED, 0 },
        { "seascape", NULL, 0, spv_seascape_frag, sizeof(spv_seascape_frag) / 4, SC_SEA, 0 },
        { "seascape, fast math", NULL, 0, spv_seascape_frag, sizeof(spv_seascape_frag) / 4, SC_SEA, 1 },
        { "seascape (O0), fast math", NULL, 0, spv_seascape_frag_O0, sizeof(spv_seascape_frag_O0) / 4, SC_SEA, 1 },
        { "BFG interaction", spv_bfg_interaction_vert, sizeof(spv_bfg_interaction_vert) / 4, spv_bfg_interaction_frag,
          sizeof(spv_bfg_interaction_frag) / 4, SC_BFG, 1 },
        { "BFG interaction (O0)", spv_bfg_interaction_vert_O0, sizeof(spv_bfg_interaction_vert_O0) / 4, spv_bfg_interaction_frag_O0,
          sizeof(spv_bfg_interaction_frag_O0) / 4, SC_BFG, 1 },
    };
    int native = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int                       bfg = cases[i].scene == SC_BFG;
        const fm3d_vertex_attrib* at  = bfg ? bfg_attr : attr;
        fm3d_spirv* sp = fm3d_spirv_create(cases[i].vs, cases[i].nvs, cases[i].fs, cases[i].nfs, cases[i].vs ? at : NULL,
                                           cases[i].vs ? (bfg ? 5 : 2) : 0, err, sizeof(err));
        CHECK(sp != NULL, "jit %s: %s", cases[i].what, err);
        if (!sp) continue;
        fm3d_spirv_set_fast_math(sp, cases[i].fast);
        fm_simd_level keep = fm_simd_current();
        for (int mode = 0; mode < 4; mode++) { /* interpreter, reference, machine code at AVX2 and at AVX-512 */
            int m = mode == 0 ? 0 : (mode == 1 ? 2 : 1);
            if (mode >= 2 && !fm_simd_set(mode == 2 ? FM_SIMD_AVX2 : FM_SIMD_AVX512)) continue;
            fm3d_spirv_set_jit(sp, 0); /* rebuilt for this level */
            fm3d_spirv_set_jit(sp, m);
            fm3d_program p = fm3d_spirv_program(sp);
            if (m) {
                const char* je = fm3d_spirv_jit_error(sp);
                if (m == 1 && je && strstr(je, "no machine code")) continue; /* no backend here (yet / this CPU) */
                CHECK(je == NULL, "jit %s (%s): %s", cases[i].what, m == 2 ? "reference" : "machine code", je ? je : "");
                if (je) continue;
                if (m == 1) native++;
            }
            fm3d_set_target(c, m ? out : ref, zb);
            fm3d_set_program(c, &p);
            { /* every other case on a straight target: the JIT packs the colors itself there */
                fm3d_blend_state off = { FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
                fm3d_set_blend_state(c, (i & 1) ? &off : NULL);
            }
            fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
            fm3d_set_depth_test(c, FM3D_LEQUAL, 1);
            fm3d_set_texture(c, tex, &ts);
            fm3d_set_texture_unit(c, 1, tex, &ts);
            if (cases[i].scene == SC_SEA) {
                jt_sea_u su;
                memset(&su, 0, sizeof(su));
                su.res[0] = 160, su.res[1] = 90, su.res[2] = 1, su.time = 7.0f;
                fm3d_set_uniforms(c, &su, sizeof(su));
                fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
                jt_quad(c, 160, 90);
            } else if (bfg) {
                jt_bfg(c, btex);
            } else {
                fm3d_set_uniforms(c, &U, sizeof(U));
                if (cases[i].scene == SC_FIXED) sv_scene_fixed(c, v, 9);
                else sv_scene_draw(c, v, 9);
            }
            fm3d_set_texture(c, NULL, NULL);
            fm3d_set_program(c, NULL);
            if (m)
                CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "jit %s (%s) = interpreter (%d rows differ)", cases[i].what,
                      m == 2 ? "reference" : (mode == 2 ? "machine code avx2" : "machine code avx512"), diff_count(ref, out));
        }
        fm_simd_set(keep);
        fm3d_spirv_destroy(sp);
    }
    printf("jit: %d programs as machine code\n", native);
    for (int i = 0; i < 5; i++) fm3d_texture_release(btex[i]);
    fm3d_texture_release(tex);
    fm_surface_destroy(ck);
    fm3d_destroy(c);
    fm_surface_destroy(ref), fm_surface_destroy(out), fm_surface_destroy(zb);
}

#if FM_TEST_AOT
/* ---- SPIR-V compiled ahead of time (spirv_aot.c, generated at build time
 * by tests/spirv_aot_gen.c): the same images as the interpreter, bit for bit */
fm3d_program aot_color_program(void);
fm3d_program aot_control_program(void);
fm3d_program aot_switch_program(void);
fm3d_program aot_fixedvs_program(void);
fm3d_program aot_fixedfs_program(void);
fm3d_program aot_func_program(void);
fm3d_program aot_seascape_program(void);
fm3d_program aot_seascape_fast_program(void);
fm3d_program aot_ubos_program(void);

typedef struct aot_sea_u { /* std140: vec3 iResolution @0, float iTime @12, vec4 iMouse @16 */
    float res[3], time, mouse[4];
} aot_sea_u;

static void aot_sea_draw(fm3d_ctx* c, int sw, int sh, float t)
{
    aot_sea_u U;
    memset(&U, 0, sizeof(U));
    U.res[0] = (float)sw, U.res[1] = (float)sh, U.res[2] = 1.0f, U.time = t;
    fm3d_set_uniforms(c, &U, sizeof(U));
    fm_mat4 pr = fm_ortho(0, (float)W, (float)H, 0, -1, 1), id = fm_mat4_identity();
    fm3d_set_projection(c, &pr);
    fm3d_set_view(c, &id);
    fm3d_set_model(c, &id);
    fm3d_vertex q[6];
    const float X[4] = { 0, (float)sw, (float)sw, 0 }, Y[4] = { 0, 0, (float)sh, (float)sh };
    const int   o[6] = { 0, 1, 2, 0, 2, 3 };
    for (int i = 0; i < 6; i++) q[i] = vtx(X[o[i]], Y[o[i]], 0, 0, 0, FM_RGB(255, 255, 255));
    fm3d_clear_color(c, FM_RGB(3, 4, 5));
    fm3d_clear_depth(c, 1.0f);
    fm3d_draw(c, q, 6);
    fm3d_flush(c);
}

/* lines and points through SPIR-V: gl_PointSize, gl_PointCoord, gl_FrontFacing */
static void test_prims_spirv(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    fm_color g = FM_RGB(0, 255, 0), r = FM_RGB(255, 0, 0);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    /* gl_PointSize from the vertex shader, gl_PointCoord in the fragment shader */
    char               err[256];
    fm3d_vertex_attrib attr[2] = { { 0, 3, 0 }, { 1, 4, 12 } };
    fm3d_spirv*        sp = fm3d_spirv_create(spv_t_points_vert, sizeof(spv_t_points_vert) / 4, spv_t_points_frag,
                                              sizeof(spv_t_points_frag) / 4, attr, 2, err, sizeof(err));
    CHECK(sp != NULL, "point sprite program: %s", err);
    if (sp) {
        sv_tu U;
        memset(&U, 0, sizeof(U));
        U.mvp = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
        fm3d_set_uniforms(c, &U, sizeof(U));
        fm3d_program pr = fm3d_spirv_program(sp);
        CHECK(pr.point_size_var > 0 && pr.point_coord_var > 0, "point size / coord varyings (%d, %d)", pr.point_size_var,
              pr.point_coord_var);
        fm3d_set_program(c, &pr);
        fm3d_set_primitive(c, FM3D_PRIM_POINTS);
        fm3d_clear_color(c, 0);
        sv_tvert pv[1] = { { { 100, 100, 0 }, { 1, 1, 1, 0.5f } } }; /* size 0.5 * 16 = 8 */
        fm3d_draw_vertices(c, pv, (int)sizeof(sv_tvert), 1, NULL, 1);
        fm3d_flush(c);
        /* pixel space has y down: the top row of the sprite is row 96, t = 1/16 there; s grows left to right */
        uint32_t tl = fm_surface_row32(fb, 96)[96], br = fm_surface_row32(fb, 103)[103];
        CHECK(count_nonzero(fb) == 64, "shader point size 8: 64 pixels (%d)", count_nonzero(fb));
        CHECK(((tl >> 16) & 255) < 24 && ((tl >> 8) & 255) < 24 && ((br >> 16) & 255) > 230 && ((br >> 8) & 255) > 230 && (tl & 255) == 255,
              "gl_PointCoord (0,0) top left .. (1,1) bottom right, front facing (%08x %08x)", tl, br);
        fm3d_set_primitive(c, FM3D_PRIM_TRIANGLES);
        fm3d_set_program(c, NULL);
        fm3d_spirv_destroy(sp);
    }
    /* gl_FrontFacing */
    fm3d_spirv* fp = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_facing_frag,
                                       sizeof(spv_t_facing_frag) / 4, attr, 2, err, sizeof(err));
    CHECK(fp != NULL, "facing program: %s", err);
    if (fp) {
        sv_tu U;
        memset(&U, 0, sizeof(U));
        U.mvp = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
        U.tint[0] = U.tint[1] = U.tint[2] = U.tint[3] = 1;
        fm3d_set_uniforms(c, &U, sizeof(U));
        fm3d_program pr = fm3d_spirv_program(fp);
        fm3d_set_program(c, &pr);
        fm3d_clear_color(c, 0);
        /* y down pixel space: visually clockwise = back with CCW fronts */
        sv_tvert t2[6] = { { { 10, 10, 0 }, { 1, 1, 1, 1 } },   { { 60, 10, 0 }, { 1, 1, 1, 1 } },  { { 10, 60, 0 }, { 1, 1, 1, 1 } },
                           { { 100, 10, 0 }, { 1, 1, 1, 1 } },  { { 100, 60, 0 }, { 1, 1, 1, 1 } }, { { 150, 10, 0 }, { 1, 1, 1, 1 } } };
        fm3d_draw_vertices(c, t2, (int)sizeof(sv_tvert), 6, NULL, 6);
        fm3d_flush(c);
        CHECK(fm_surface_row32(fb, 15)[15] == r && fm_surface_row32(fb, 15)[105] == g, "gl_FrontFacing: back red, front green (%08x %08x)",
              fm_surface_row32(fb, 15)[15], fm_surface_row32(fb, 15)[105]);
        fm3d_set_program(c, NULL);
        fm3d_spirv_destroy(fp);
    }
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

/* cube maps and texture arrays through SPIR-V (fm3d_texture_create_layers, fm3d_sample_tex) */
/* fm3d_set_color_mask: masked clears keep the other channels */
static void test_color_mask(void)
{
    fm_surface* fb = fm_surface_create(16, 16, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    fm3d_blend_state bs = { FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
    fm3d_set_blend_state(c, &bs);
    fm3d_clear_color(c, 0x80336699u);
    fm3d_set_color_mask(c, 1, 0, 0, 1);
    fm3d_clear_color(c, 0xFFFFFFFFu);
    fm3d_set_color_mask(c, 0, 0, 0, 0);
    fm3d_clear_color(c, 0u);
    fm3d_flush(c);
    CHECK(fm_surface_row32(fb, 5)[5] == 0xFFFF6699u, "masked clears (%08x)", fm_surface_row32(fb, 5)[5]);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

static void test_cube_array(void)
{
    fm_surface* fb = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm3d_ctx*   c  = fm3d_create();
    fm3d_set_target(c, fb, NULL);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_blend_state bs = { FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BF_ONE, FM3D_BF_ZERO, FM3D_BLEND_ADD, FM3D_BLEND_ADD, 0 };
    fm3d_set_blend_state(c, &bs);
    /* six faces, one color each (+X red, -X green, +Y blue, -Y yellow, +Z cyan, -Z magenta); 4 x 4 */
    static const fm_color fc[6] = { 0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu, 0xFFFFFF00u, 0xFF00FFFFu, 0xFFFF00FFu };
    fm_surface* face[6];
    for (int f = 0; f < 6; f++) {
        face[f] = fm_surface_create(4, 4, FM_FORMAT_ARGB32);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) fm_surface_row32(face[f], y)[x] = fc[f];
    }
    fm3d_texture* cube = fm3d_texture_create_layers(FM3D_TEX_CUBE, (const fm_surface* const*)face, 6, FM3D_TEXTURE_STRAIGHT);
    fm3d_texture* arr  = fm3d_texture_create_layers(FM3D_TEX_2D_ARRAY, (const fm_surface* const*)face, 4, FM3D_TEXTURE_STRAIGHT);
    CHECK(cube && arr && fm3d_texture_layers(cube) == 6 && fm3d_texture_get_kind(arr) == FM3D_TEX_2D_ARRAY, "layered textures");
    fm3d_sampler ns = { FM3D_FILTER_NEAREST, FM_WRAP_CLAMP, FM_WRAP_CLAMP, 0, FM_WRAP_CLAMP };
    fm3d_set_texture_unit(c, 1, cube, &ns);
    fm3d_set_texture_unit(c, 2, arr, &ns);
    char               err[256];
    fm3d_vertex_attrib attr[2] = { { 0, 3, 0 }, { 1, 4, 12 } };
    fm3d_spirv*        p = fm3d_spirv_create(spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_cube_frag, sizeof(spv_t_cube_frag) / 4,
                                             attr, 2, err, sizeof(err));
    CHECK(p != NULL, "cube / array program: %s", err);
    if (p && cube && arr) {
        fm3d_program pr = fm3d_spirv_program(p);
        fm3d_set_program(c, &pr);
        sv_tu U;
        memset(&U, 0, sizeof(U));
        U.mvp = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
        /* directions (color * 2 - 1) at 6 quads: +X -X +Y -Y +Z -Z */
        static const float dir[6][3] = { { 1, 0.5f, 0.5f }, { 0, 0.5f, 0.5f }, { 0.5f, 1, 0.5f }, { 0.5f, 0, 0.5f }, { 0.5f, 0.5f, 1 },
                                         { 0.5f, 0.5f, 0 } };
        for (int pass = 0; pass < 2; pass++) {
            U.tint[0] = pass == 0 ? 1.0f : 0.0f;
            fm3d_set_uniforms(c, &U, sizeof(U));
            fm3d_clear_color(c, 0);
            for (int q = 0; q < (pass ? 4 : 6); q++) {
                float    x0 = 10.0f + 50.0f * (float)q, x1 = x0 + 40.0f;
                float    d0 = pass ? 0.0f : dir[q][0], d1 = pass ? 0.0f : dir[q][1], d2 = pass ? ((float)q + 0.5f) / 3.0f : dir[q][2];
                sv_tvert v[6] = { { { x0, 10, 0 }, { d0, d1, d2, 1 } }, { { x1, 10, 0 }, { d0, d1, d2, 1 } }, { { x1, 50, 0 }, { d0, d1, d2, 1 } },
                                  { { x0, 10, 0 }, { d0, d1, d2, 1 } }, { { x1, 50, 0 }, { d0, d1, d2, 1 } }, { { x0, 50, 0 }, { d0, d1, d2, 1 } } };
                fm3d_draw_vertices(c, v, (int)sizeof(sv_tvert), 6, NULL, 6);
            }
            fm3d_flush(c);
            int bad = 0;
            for (int q = 0; q < (pass ? 4 : 6); q++) {
                /* arrays: layer = round(z * 3) of (q + 0.5) / 3 * 3 = q + 0.5 -> q + 1 (clamped to 3) */
                int want = pass ? (q + 1 < 3 ? q + 1 : 3) : q;
                if (fm_surface_row32(fb, 30)[30 + 50 * q] != fc[want]) bad++;
            }
            CHECK(bad == 0, "%s: %d wrong (%08x %08x %08x)", pass ? "texture array layers" : "cube map faces", bad, fm_surface_row32(fb, 30)[30],
                  fm_surface_row32(fb, 30)[80], fm_surface_row32(fb, 30)[130]);
        }
        fm3d_set_program(c, NULL);
        fm3d_spirv_destroy(p);
    }
    fm3d_texture_release(cube);
    fm3d_texture_release(arr);
    for (int f = 0; f < 6; f++) fm_surface_destroy(face[f]);
    fm3d_destroy(c);
    fm_surface_destroy(fb);
}

static void test_spirv_aot(void)
{
    fm_surface* ref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* out = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface* zb  = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*   c   = fm3d_create();
    sv_tvert v[9] = { { { 10, 10, 0.2f }, { 1, 0.2f, 0.2f, 1 } },          { { 300, 30, 0.5f }, { 0.2f, 1, 0.2f, 1 } },
                      { { 40, 230, 0.8f }, { 0.2f, 0.2f, 1, 0.8f } },      { { 200, 5, 0.1f }, { 1, 1, 0.2f, 1 } },
                      { { 310, 220, 0.9f }, { 0.2f, 1, 1, 1 } },           { { 120, 200, 0.3f }, { 1, 0.2f, 1, 1 } },
                      { { 0, 120, 0.4f }, { 0.6f, 0.2f, 0.8f, 1 } },       { { 160, 0, 0.6f }, { 0.4f, 0.8f, 0.6f, 1 } },
                      { { 320, 240, 0.7f }, { 0.8f, 0.6f, 0.4f, 0.6f } } };
    fm3d_vertex_attrib attr[2] = { { 0, 3, 0 }, { 1, 4, 12 } };
    sv_tu       U;
    memset(&U, 0, sizeof(U));
    U.mvp = fm_ortho(0, (float)W, (float)H, 0, -1, 1);
    U.tint[0] = 1.0f, U.tint[1] = 0.8f, U.tint[2] = 0.6f, U.tint[3] = 1.0f;
    U.time = 0.7f;
    fm_surface* ck = fm_surface_create(64, 64, FM_FORMAT_ARGB32);
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) fm_surface_row32(ck, y)[x] = ((x ^ y) & 4) ? FM_RGB(240, 200, 30) : FM_RGB(30, 60, 200);
    fm3d_texture* tex = fm3d_texture_create(ck, 1);
    fm3d_sampler  ts  = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0, FM_WRAP_REPEAT };
    fm3d_set_texture_unit(c, 1, tex, &ts);
    fm3d_set_cull(c, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_set_depth_test(c, FM3D_LEQUAL, 1);
    char err[256];

    struct {
        const char*   what;
        const uint32_t* vs;
        size_t        nvs;
        const uint32_t* fs;
        size_t        nfs;
        fm3d_program  (*aot)(void);
        int           fixed_scene; /* fixed vertex stage: fm3d_vertex input */
    } cases[] = {
        { "vs + fs", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_color_frag, sizeof(spv_t_color_frag) / 4, aot_color_program, 0 },
        { "loop / if / discard / texture / fwidth", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_control_frag,
          sizeof(spv_t_control_frag) / 4, aot_control_program, 0 },
        { "switch / early returns", spv_t_basic_vert, sizeof(spv_t_basic_vert) / 4, spv_t_switch_frag, sizeof(spv_t_switch_frag) / 4,
          aot_switch_program, 0 },
        { "fixed vs + fs", NULL, 0, spv_t_fixedvs_frag, sizeof(spv_t_fixedvs_frag) / 4, aot_fixedvs_program, 1 },
        { "vs + fixed textured fs", spv_t_fixedfs_vert, sizeof(spv_t_fixedfs_vert) / 4, NULL, 0, aot_fixedfs_program, 0 },
        { "inlined functions", NULL, 0, spv_t_func_frag, sizeof(spv_t_func_frag) / 4, aot_func_program, 1 },
    };
    fm3d_set_texture(c, tex, &ts);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        fm3d_spirv* sp = fm3d_spirv_create(cases[i].vs, cases[i].nvs, cases[i].fs, cases[i].nfs, cases[i].vs ? attr : NULL,
                                           cases[i].vs ? 2 : 0, err, sizeof(err));
        CHECK(sp != NULL, "aot %s: %s", cases[i].what, err);
        if (!sp) continue;
        fm3d_program ip = fm3d_spirv_program(sp), ap = cases[i].aot();
        CHECK(ap.nvaryings == ip.nvaryings && ap.discards == ip.discards && !ap.vs == !ip.vs && !ap.fs == !ip.fs,
              "aot %s: program shape", cases[i].what);
        for (int pass = 0; pass < 2; pass++) {
            fm3d_set_uniforms(c, &U, sizeof(U));
            fm3d_set_target(c, pass ? out : ref, zb);
            fm3d_set_program(c, pass ? &ap : &ip);
            if (cases[i].fixed_scene) sv_scene_fixed(c, v, 9);
            else sv_scene_draw(c, v, 9);
        }
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "aot %s = interpreter (%d rows differ)", cases[i].what,
              diff_count(ref, out));
        fm3d_set_program(c, NULL);
        fm3d_spirv_destroy(sp);
    }
    fm3d_set_texture(c, NULL, NULL);

    /* uniform blocks at bindings 0 and 3, both backends: 0.5 * (0.8, 0.4, 1.0) * 1.5 = (0.6, 0.3, 0.75) */
    {
        fm3d_spirv* ub = fm3d_spirv_create(NULL, 0, spv_t_ubos_frag, sizeof(spv_t_ubos_frag) / 4, NULL, 0, err, sizeof(err));
        CHECK(ub != NULL, "two uniform blocks: %s", err);
        float A[4] = { 0.5f, 0.5f, 0.5f, 1.0f }, Bk[8] = { 0.8f, 0.4f, 1.0f, 1.0f, 1.5f, 0, 0, 0 };
        fm_mat4 pr = fm_ortho(0, (float)W, (float)H, 0, -1, 1), id = fm_mat4_identity(); /* the fixed vertex stage */
        fm3d_set_projection(c, &pr);
        fm3d_set_view(c, &id);
        fm3d_set_model(c, &id);
        for (int k = 0; ub && k < 2; k++) {
            fm3d_program p = k ? aot_ubos_program() : fm3d_spirv_program(ub);
            fm3d_set_uniforms(c, A, sizeof(A));
            fm3d_set_uniform_block(c, 3, Bk, sizeof(Bk));
            fm3d_set_target(c, out, zb);
            fm3d_set_program(c, &p);
            sv_scene_fixed(c, v, 9);
            int hits = 0, other = 0; /* every covered pixel has the expected color */
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    fm_color px = fm_surface_get_pixel(out, x, y);
                    if (px == FM_RGB(153, 77, 191)) hits++;
                    else if (px != FM_RGB(3, 4, 5)) other++;
                }
            CHECK(hits > 0 && other == 0, "%s reads blocks 0 and 3 (%d pixels right, %d wrong)", k ? "compiled" : "interpreter", hits, other);
        }
        fm3d_set_uniform_block(c, 3, NULL, 0);
        fm3d_set_program(c, NULL);
        fm3d_spirv_destroy(ub);
    }

    /* Seascape (ray marching: loops with divergent breaks, inlined early
     * returns, out parameters, many transcendentals), on a pool */
    fm3d_spirv* sea = fm3d_spirv_create(NULL, 0, spv_seascape_frag, sizeof(spv_seascape_frag) / 4, NULL, 0, err, sizeof(err));
    CHECK(sea != NULL, "aot seascape: %s", err);
    if (sea) {
        fm_executor* ex = fm_executor_create(0);
        fm3d_set_deferred(c, 1);
        fm3d_set_executor(c, ex);
        fm3d_set_depth_test(c, FM3D_ALWAYS, 0);
        fm3d_program ip = fm3d_spirv_program(sea), ap = aot_seascape_program();
        fm3d_set_target(c, ref, zb);
        fm3d_set_program(c, &ip);
        aot_sea_draw(c, 160, 90, 7.0f);
        fm3d_set_target(c, out, zb);
        fm3d_set_program(c, &ap);
        aot_sea_draw(c, 160, 90, 7.0f);
        CHECK(diff_count(ref, out) == 0 && count_nonzero(ref) > 0, "aot seascape = interpreter (%d rows differ)", diff_count(ref, out));
        uint32_t hsh = 2166136261u; /* the same on every platform (fm_vmath.h): compare across builds */
        for (int y = 0; y < 90; y++)
            for (int x = 0; x < 160; x++) hsh = (hsh ^ fm_surface_row32(ref, y)[x]) * 16777619u;
        printf("seascape 160x90 image hash: %08x\n", hsh);
        /* every SIMD level, both backends (the executor / generated stages per ISA): the same image */
        fm_simd_level lv[5] = { FM_SIMD_SCALAR, FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_AVX512, FM_SIMD_NEON };
        fm_simd_level keep  = fm_simd_current();
        for (int i = 0; i < 5; i++) {
            if (!fm_simd_set(lv[i])) continue;
            for (int b = 0; b < 2; b++) {
                fm3d_set_target(c, out, zb);
                fm3d_set_program(c, b ? &ap : &ip);
                aot_sea_draw(c, 160, 90, 7.0f);
                CHECK(diff_count(ref, out) == 0, "seascape %s at %s = the reference (%d rows differ)", b ? "compiled" : "interpreted",
                      fm_simd_name(lv[i]), diff_count(ref, out));
            }
        }
        fm_simd_set(keep);

        /* fast math: the backends agree with each other at every level, and stay close to the precise image */
        fm3d_spirv_set_fast_math(sea, 1);
        fm3d_program fip = fm3d_spirv_program(sea), fap = aot_seascape_fast_program();
        fm_surface*  fref = fm_surface_create(W, H, FM_FORMAT_ARGB32);
        fm3d_set_target(c, fref, zb);
        fm3d_set_program(c, &fip);
        aot_sea_draw(c, 160, 90, 7.0f);
        double dsum = 0; /* the waves' fine detail is chaotic; on average the images agree */
        for (int y = 0; y < 90; y++)
            for (int x = 0; x < 160; x++) {
                uint32_t p = fm_surface_row32(ref, y)[x], q = fm_surface_row32(fref, y)[x];
                for (int sh = 0; sh < 24; sh += 8) dsum += abs((int)((p >> sh) & 255) - (int)((q >> sh) & 255));
            }
        dsum /= 160.0 * 90.0 * 3.0;
        CHECK(dsum < 2.0, "fast math seascape close to the precise one (mean channel difference %.2f)", dsum);
        for (int i = 0; i < 5; i++) {
            if (!fm_simd_set(lv[i])) continue;
            for (int b = 0; b < 2; b++) {
                fm3d_set_target(c, out, zb);
                fm3d_set_program(c, b ? &fap : &fip);
                aot_sea_draw(c, 160, 90, 7.0f);
                CHECK(diff_count(fref, out) == 0, "fast seascape %s at %s = the fast reference (%d rows differ)",
                      b ? "compiled" : "interpreted", fm_simd_name(lv[i]), diff_count(fref, out));
            }
        }
        fm_simd_set(keep);
        fm_surface_destroy(fref);
        fm3d_spirv_set_fast_math(sea, 0);

        /* the shader's dozen functions (out parameters, early returns, loops with breaks) inlined by fatmap */
        fm3d_spirv* sea0 = fm3d_spirv_create(NULL, 0, spv_seascape_frag_O0, sizeof(spv_seascape_frag_O0) / 4, NULL, 0, err, sizeof(err));
        CHECK(sea0 != NULL, "seascape -O0: %s", err);
        if (sea0) {
            fm3d_program p0 = fm3d_spirv_program(sea0);
            fm3d_set_target(c, out, zb);
            fm3d_set_program(c, &p0);
            aot_sea_draw(c, 160, 90, 7.0f);
            CHECK(diff_count(ref, out) == 0, "seascape -O0 (fatmap inlining) = -O (%d rows differ)", diff_count(ref, out));
            fm3d_spirv_destroy(sea0);
        }
        fm3d_set_program(c, NULL);
        fm3d_set_deferred(c, 0);
        fm3d_set_executor(c, NULL);
        fm_executor_destroy(ex);
        fm3d_spirv_destroy(sea);
    }
    fm3d_texture_release(tex);
    fm_surface_destroy(ck);
    fm3d_destroy(c);
    fm_surface_destroy(ref);
    fm_surface_destroy(out);
    fm_surface_destroy(zb);
}
#endif
#endif

int main(int argc, char** argv)
{
    if (argc > 1) g_outdir = argv[1];
    printf("fatmap 3d tests, SIMD %s\n", fm_simd_name(fm_simd_best()));
    test_fill_convention();
    test_depth_cull_clip();
    test_origin();
    test_blend_state();
    test_depth_stencil();
    test_depth_formats();
    test_hiz();
    test_perspective();
    test_mipmaps();
    test_equivalence();
    test_msaa();
    test_msaa_equivalence();
    test_swapchain();
#if FM_FEATURE_VBO
    test_buffers();
#endif
#if FM_FEATURE_TNL
    test_lighting();
#endif
#if FM_FEATURE_SHADERS
    test_shaders();
#endif
#if FM_FEATURE_SPIRV
    test_spirv();
    test_prims_spirv();
    test_cube_array();
    test_color_mask();
#endif
    test_prims();
    test_multitexture_fog();
    test_stats_work();
#if FM_FEATURE_SPIRV
    test_sampler_simd();
    test_jit();
#endif
#if FM_TEST_AOT
    test_spirv_aot();
#endif
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
