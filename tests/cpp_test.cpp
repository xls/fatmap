// fatmap++ smoke test + fm_math vs glm compatibility check.
#if FM_TEST_GLM
#  include <glm/glm.hpp>
#  include <glm/gtc/matrix_transform.hpp>
#endif
#include <fatmap/fatmap.hpp>

#include <cmath>
#include <cstdio>

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

#if FM_TEST_GLM
    test_glm();
#endif
    std::printf("cpp_test: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
