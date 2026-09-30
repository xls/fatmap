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

static void test_equivalence(void)
{
    fm_surface*   img  = checker(128, 4);
    fm_surface*   img2 = tex_image(64);
    fm3d_texture* tex  = fm3d_texture_create(img, 1);
    fm3d_texture* tex2 = fm3d_texture_create(img2, 1);
    fm_surface*   ref  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   out  = fm_surface_create(W, H, FM_FORMAT_ARGB32);
    fm_surface*   zb   = fm_surface_create(W, H, FM_FORMAT_D32F);
    fm3d_ctx*     c    = fm3d_create();

    fm_simd_set(FM_SIMD_SCALAR);
    fm3d_set_target(c, ref, zb);
    draw_scene3d(c, tex, tex2, 0.6f);
    char path[512];
    snprintf(path, sizeof(path), "%s/3d_scene.png", g_outdir);
    CHECK(fm_surface_write_png(ref, path), "write %s", path);
    fm3d_stats st = fm3d_get_stats(c);
    CHECK(st.triangles_drawn > 200, "scene draws triangles (%llu)", (unsigned long long)st.triangles_drawn);

    fm_simd_level lv[3] = { FM_SIMD_SSE2, FM_SIMD_AVX2, FM_SIMD_NEON };
    for (int i = 0; i < 3; i++) {
        if (!fm_simd_supported(lv[i])) continue;
        fm_simd_set(lv[i]);
        fm3d_set_target(c, out, zb);
        draw_scene3d(c, tex, tex2, 0.6f);
        int d = diff_count(ref, out);
        CHECK(d == 0, "3d immediate %s vs scalar: %d rows differ", fm_simd_name(lv[i]), d);
    }
    fm_simd_set(fm_simd_best());
    struct {
        int threads, tile;
    } modes[] = { { 0, 64 }, { 3, 32 }, { 4, 64 }, { 8, 128 }, { -1, 64 }, { -1, 16 } };
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        fm_executor* ex = modes[i].threads ? fm_executor_create(modes[i].threads) : NULL;
        fm3d_set_target(c, out, zb);
        fm3d_set_tile_size(c, modes[i].tile);
        fm3d_set_deferred(c, 1);
        fm3d_set_executor(c, ex);
        for (int f = 0; f < 2; f++) {
            draw_scene3d(c, tex, tex2, 0.6f);
            fm3d_flush(c);
        }
        int d = diff_count(ref, out);
        CHECK(d == 0, "3d tiled (threads=%d workers=%d tile=%d): %d rows differ", modes[i].threads,
              ex ? ex->workers : 1, modes[i].tile, d);
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
    test_perspective();
    test_mipmaps();
    test_equivalence();
    test_swapchain();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
