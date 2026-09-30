/* fatmap consumer example (C11): a threaded 2D canvas and a 3D triangle.
 * Built against an installed / downloaded fatmap package, see README.md. */
#include <fatmap/fatmap.h>

#include <stdio.h>

int main(void)
{
    fm_surface*  fb = fm_surface_create(320, 240, FM_FORMAT_ARGB32);
    fm_executor* ex = fm_executor_create(0); /* one worker per CPU; pass N to cap the pool */

    /* 2D: recorded into a command list, rasterized in parallel strips on flush */
    fm2d_ctx* c = fm2d_create(fb);
    fm2d_set_executor(c, ex);
    fm2d_set_deferred(c, 1);
    fm2d_clear(c, FM_RGB(255, 255, 255));
    fm2d_set_fill_color(c, FM_RGBA(0, 128, 255, 200));
    fm2d_begin_path(c);
    fm2d_arc(c, 160, 120, 90, 0, 6.2831853f, 0);
    fm2d_fill(c, FM_FILL_NONZERO);
    fm2d_flush(c);

    /* 3D: pixel space projection, one Gouraud triangle on top */
    fm3d_ctx* d = fm3d_create();
    fm3d_set_target(d, fb, NULL);
    fm_mat4 proj = fm_ortho(0, 320, 240, 0, -1, 1);
    fm3d_set_projection(d, &proj);
    fm3d_vertex tri[3] = {
        { 40, 200, 0, 0, 0, 1, 0, 0, FM_RGB(255, 0, 0) },
        { 280, 200, 0, 0, 0, 1, 0, 0, FM_RGB(0, 255, 0) },
        { 160, 30, 0, 0, 0, 1, 0, 0, FM_RGB(0, 0, 255) },
    };
    fm3d_set_cull(d, FM3D_CULL_NONE, FM3D_FRONT_CCW);
    fm3d_draw(d, tri, 3);

    fm_color center = fm_surface_get_pixel(fb, 160, 150);
    fm_color corner = fm_surface_get_pixel(fb, 2, 2);
    printf("fatmap %s, simd %s, %d workers, center %08x corner %08x\n", fm_version_string(),
           fm_simd_name(fm_simd_current()), ex->workers, (unsigned)center, (unsigned)corner);
    int ok = corner == FM_RGB(255, 255, 255) && (center >> 24) == 0xff && center != FM_RGB(255, 255, 255);
    fm_surface_write_png(fb, "consumer_c.png");

    fm3d_destroy(d);
    fm2d_destroy(c);
    fm_executor_destroy(ex);
    fm_surface_destroy(fb);
    return ok ? 0 : 1;
}
