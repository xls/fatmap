/* fatmap - 3D textures: immutable mip chains (box filtered, premultiplied) */
#include "fm3d_internal.h"

fm3d_texture* fm3d_texture_create(const fm_surface* image, int mipmaps)
{
    fm__init();
    if (!image || image->format != FM_FORMAT_ARGB32) return NULL;
    fm3d_texture* t = (fm3d_texture*)calloc(1, sizeof(fm3d_texture));
    if (!t) return NULL;
    t->refs     = 1;
    t->level[0] = fm_surface_clone(image);
    if (!t->level[0]) {
        free(t);
        return NULL;
    }
    t->levels = 1;
    while (mipmaps && t->levels < 16) {
        const fm_surface* src = t->level[t->levels - 1];
        if (src->width == 1 && src->height == 1) break;
        int         w = FM_MAX(1, src->width / 2), h = FM_MAX(1, src->height / 2);
        fm_surface* dst = fm_surface_create(w, h, FM_FORMAT_ARGB32);
        if (!dst) break;
        for (int y = 0; y < h; y++) {
            const uint32_t* r0 = fm_surface_row32(src, FM_MIN(2 * y, src->height - 1));
            const uint32_t* r1 = fm_surface_row32(src, FM_MIN(2 * y + 1, src->height - 1));
            uint32_t*       d  = fm_surface_row32(dst, y);
            for (int x = 0; x < w; x++) {
                int      x0 = FM_MIN(2 * x, src->width - 1), x1 = FM_MIN(2 * x + 1, src->width - 1);
                uint32_t a = r0[x0], b = r0[x1], c = r1[x0], e = r1[x1], o = 0;
                for (int sh = 0; sh < 32; sh += 8) {
                    uint32_t sum = ((a >> sh) & 255) + ((b >> sh) & 255) + ((c >> sh) & 255) + ((e >> sh) & 255);
                    o |= ((sum + 2) >> 2) << sh;
                }
                d[x] = o;
            }
        }
        t->level[t->levels++] = dst;
    }
    return t;
}

fm3d_texture* fm3d_texture_retain(fm3d_texture* t)
{
    if (t) t->refs++;
    return t;
}

void fm3d_texture_release(fm3d_texture* t)
{
    if (!t || --t->refs > 0) return;
    for (int i = 0; i < t->levels; i++) fm_surface_destroy(t->level[i]);
    free(t);
}

int               fm3d_texture_levels(const fm3d_texture* t) { return t ? t->levels : 0; }
const fm_surface* fm3d_texture_level(const fm3d_texture* t, int level)
{
    return (t && level >= 0 && level < t->levels) ? t->level[level] : NULL;
}
