// fatmap++ smoke test + fm_math vs glm compatibility check.
#if FM_TEST_GLM
#  include <glm/glm.hpp>
#  include <glm/gtc/matrix_transform.hpp>
#endif
#include <fatmap/fatmap.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define EXPECT(c, msg)                                   \
    do {                                                 \
        if (!(c)) {                                      \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_fail++;                                    \
        }                                                \
    } while (0)

#if FM_TEST_GLM
static float max_diff(const fm_mat4& a, const glm::mat4& b)
{
    const float* pa = fm_mat4_ptr(&a);
    const float* pb = &b[0][0];
    float        d  = 0;
    for (int i = 0; i < 16; i++) d = std::fmax(d, std::fabs(pa[i] - pb[i]));
    return d;
}

static void test_glm()
{
    glm::mat4 gp = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    fm_mat4   fp = fm_perspective(fm_radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    EXPECT(max_diff(fp, gp) < 1e-6f, "perspective matches glm");

    glm::mat4 gv = glm::lookAt(glm::vec3(1, 2, 5), glm::vec3(0, 0.5f, 0), glm::vec3(0, 1, 0));
    fm_mat4   fv = fm_lookat(fm_v3(1, 2, 5), fm_v3(0, 0.5f, 0), fm_v3(0, 1, 0));
    EXPECT(max_diff(fv, gv) < 1e-6f, "lookAt matches glm");

    glm::mat4 gm = glm::scale(glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(1, 2, 3)), 0.7f,
                                          glm::vec3(0.3f, 1.0f, 0.2f)),
                              glm::vec3(2, 3, 4));
    fm_mat4   fm_ = fm_scale(fm_rotate(fm_translate(fm_mat4_identity(), fm_v3(1, 2, 3)), 0.7f, fm_v3(0.3f, 1.0f, 0.2f)),
                             fm_v3(2, 3, 4));
    EXPECT(max_diff(fm_, gm) < 1e-5f, "translate/rotate/scale matches glm");

    EXPECT(max_diff(fm_mat4_mul(fp, fv), gp * gv) < 1e-5f, "mat4 multiply matches glm");
    EXPECT(max_diff(fm_mat4_inverse(fm_), glm::inverse(gm)) < 1e-4f, "inverse matches glm");
    EXPECT(max_diff(fm_ortho(-1, 2, -3, 4, 0.5f, 50), glm::ortho(-1.0f, 2.0f, -3.0f, 4.0f, 0.5f, 50.0f)) < 1e-6f,
           "ortho matches glm");

    // zero-copy interop
    glm::mat4 back = fm::toGlm(fm::toFm(gm));
    EXPECT(back == gm, "glm roundtrip");
    std::printf("glm %d.%d.%d compatibility: ok\n", GLM_VERSION_MAJOR, GLM_VERSION_MINOR, GLM_VERSION_PATCH);
}
#endif

int main()
{
    fm::Surface  fb(320, 200);
    fm::Canvas2D ctx(fb);
    fb.clear(fm::color("white"));

    auto grad = fm::Canvas2D::createLinearGradient(0, 0, 320, 0);
    grad.addColorStop(0, "red").addColorStop(1, "#00f");
    ctx.fillStyle(grad);
    ctx.fillRect(0, 0, 320, 100);

    fm::Path2D p;
    p.arc(160, 150, 40, 0, 2 * fm::pi).closePath();
    ctx.fillStyle("rgba(0, 128, 0, 0.5)");
    ctx.fill(p);
    ctx.lineWidth(4);
    ctx.strokeStyle(fm::rgb(0, 0, 0));
    ctx.stroke(p);
    ctx.globalCompositeOperation("multiply");
    ctx.fillStyle("gray");
    ctx.fillRect(150, 0, 20, 200);

    EXPECT(fb.pixel(1, 50) == fm::rgb(255, 0, 0) || (fb.pixel(1, 50) & 0xff) < 2, "gradient left is red");
    EXPECT(ctx.isPointInPath(0, 0) == false, "empty current path");
    EXPECT(fm::simdName(fm::simdBest()) != nullptr, "simd name");

    // 2D multithreaded: deferred canvas on a pool capped at 2 workers must
    // produce exactly the immediate mode pixels
    {
        fm::Surface  imm(256, 256), mt(256, 256);
        fm::Executor ex(2);
        EXPECT(ex.workers() == (fm_threads_supported() ? 2 : 1), "executor pool capped at 2 workers");
        auto scene = [](fm::Canvas2D& c) {
            c.clear(fm::rgb(255, 255, 255));
            for (int i = 0; i < 40; i++) {
                fm::Path2D q;
                q.arc(20.0f + (float)(i * 37 % 216), 20.0f + (float)(i * 53 % 216), 8.0f + (float)(i % 5) * 6.0f, 0,
                      2 * fm::pi);
                c.fillStyle(fm::rgba((uint8_t)(i * 29), (uint8_t)(i * 71), (uint8_t)(255 - i * 5), 160));
                c.fill(q);
            }
        };
        fm::Canvas2D a(imm);
        scene(a);
        fm::Canvas2D b(mt);
        b.executor(ex);
        b.deferred(true);
        scene(b);
        b.flush();
        EXPECT(b.deferred(), "canvas stays deferred after flush");
        int rows_differ = 0;
        for (int y = 0; y < 256; y++)
            rows_differ += std::memcmp(fm_surface_row32(imm.get(), y), fm_surface_row32(mt.get(), y), 256 * 4) != 0;
        EXPECT(rows_differ == 0, "2d deferred on executor matches immediate");
    }

    // 3D: swapchain back buffer, glm (if available) matrices, threaded tiles
    {
        fm::Swapchain sc(160, 120);
        fm::Canvas3D  c3;
        fm::Executor  ex(4);
        c3.deferred(true);
        c3.executor(ex);
        c3.setTarget(sc);
        c3.clearColor(fm::rgb(0, 0, 0));
        c3.clearDepth();
#if FM_TEST_GLM
        c3.setProjection(glm::perspective(glm::radians(60.0f), 160.0f / 120.0f, 0.1f, 10.0f));
        c3.setView(glm::lookAt(glm::vec3(0, 0, 3), glm::vec3(0), glm::vec3(0, 1, 0)));
#else
        c3.setProjection(fm_perspective(fm_radians(60), 160.0f / 120.0f, 0.1f, 10.0f));
        c3.setView(fm_lookat(fm_v3(0, 0, 3), fm_v3(0, 0, 0), fm_v3(0, 1, 0)));
#endif
        std::vector<fm::Vertex3D> tri(3);
        tri[0] = { -1, -1, 0, 0, 0, 1, 0, 0, fm::rgb(255, 0, 0) };
        tri[1] = { 1, -1, 0, 0, 0, 1, 1, 0, fm::rgb(0, 255, 0) };
        tri[2] = { 0, 1, 0, 0, 0, 1, 0.5f, 1, fm::rgb(0, 0, 255) };
        c3.draw(tri);
#if FM_FEATURE_VBO
        fm::Buffer vb(tri);
        EXPECT(vb.vertexCount() == 3 && vb.indexCount() == 0, "vertex buffer counts");
        c3.draw(vb); /* same triangle again from the buffer: LESS depth test keeps the first */
#endif
        c3.flush();
        fm_surface* front = sc.present();
        EXPECT((fm_surface_get_pixel(front, 80, 70) >> 24) == 255, "3d triangle rendered into back buffer");
        EXPECT(fm_surface_get_pixel(front, 2, 2) == fm::rgb(0, 0, 0), "3d clear");
    }

#if FM_TEST_GLM
    test_glm();
#endif
    std::printf("cpp_test: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
