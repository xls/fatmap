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
    fm3d_sampler smp = { FM3D_FILTER_NEAREST, FM_WRAP_CLAMP, FM_WRAP_CLAMP, 0 };
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
        fm3d_sampler smp = { k ? FM3D_FILTER_TRILINEAR : FM3D_FILTER_NEAREST, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0 };
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
    fm3d_sampler tri = { FM3D_FILTER_TRILINEAR, FM_WRAP_REPEAT, FM_WRAP_REPEAT, 0 };
    fm3d_set_texture(c, tex, &tri);
    fm3d_draw_indexed(c, fv, (G + 1) * (G + 1), fi, G * G * 6);

    /* cubes: bilinear, vertex color modulate */
    fm3d_vertex  cb[36];
    fm3d_sampler bil = { FM3D_FILTER_BILINEAR, FM_WRAP_CLAMP, FM_WRAP_CLAMP, 0 };
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

    fm_simd_level lv[3] = { FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_NEON };
    for (int i = 0; i < 3; i++) {
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

int main(int argc, char** argv)
{
    if (argc > 1) g_outdir = argv[1];
    printf("fatmap 3d tests, SIMD %s\n", fm_simd_name(fm_simd_best()));
    test_fill_convention();
    test_depth_cull_clip();
    test_depth_stencil();
    test_depth_formats();
    test_hiz();
    test_perspective();
    test_mipmaps();
    test_equivalence();
    test_swapchain();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
