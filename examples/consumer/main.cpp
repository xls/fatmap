// fatmap consumer example (C++17 wrapper): a threaded 2D canvas.
#include <fatmap/fatmap.hpp>

#include <cstdio>

int main()
{
    fm::Surface  fb(320, 240);
    fm::Executor pool(2); // at most 2 workers
    fm::Canvas2D ctx(fb);
    ctx.executor(pool);
    ctx.deferred(true);
    ctx.clear(fm::rgb(255, 255, 255));

    auto grad = fm::Canvas2D::createLinearGradient(0, 0, 320, 0);
    grad.addColorStop(0, "red").addColorStop(1, "#00f");
    ctx.fillStyle(grad);
    fm::Path2D p;
    p.arc(160, 120, 90, 0, 2 * fm::pi).closePath();
    ctx.fill(p);
    ctx.lineWidth(6);
    ctx.strokeStyle("black");
    ctx.stroke(p);
    ctx.flush();

    fm::Color center = fb.pixel(160, 120);
    std::printf("fatmap++ %s, %d workers, center %08x\n", fm_version_string(), pool.workers(), (unsigned)center);
    fb.savePNG("consumer_cpp.png");
    return fb.pixel(2, 2) == fm::rgb(255, 255, 255) && center != fm::rgb(255, 255, 255) ? 0 : 1;
}
