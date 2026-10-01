/* fatmap - 3D textures: immutable mip chains (box filtered, premultiplied) */
#include "fm3d_internal.h"

fm3d_texture* fm3d_texture_create(const fm_surface* image, int mipmaps)
{
    return fm3d_texture_create_layers(FM3D_TEX_2D, &image, 1, mipmaps ? FM3D_TEXTURE_MIPMAPS : 0);
}

/* the box filtered chain of level[0] into level[1 ..]; returns the count */
static int fm3d_build_mips(fm_surface** level)
{
    int levels = 1;
    while (levels < 16) {
        const fm_surface* src = level[levels - 1];
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
        level[levels++] = dst;
    }
    return levels;
}

fm3d_texture* fm3d_texture_create_layers(fm3d_texture_kind kind, const fm_surface* const* layers, int count, unsigned flags)
{
    fm__init();
    if (!layers || count < 1 || (kind == FM3D_TEX_CUBE && count != 6) || (kind == FM3D_TEX_2D && count != 1)) return NULL;
    for (int i = 0; i < count; i++)
        if (!layers[i] || layers[i]->format != FM_FORMAT_ARGB32 || layers[i]->width != layers[0]->width ||
            layers[i]->height != layers[0]->height)
            return NULL;
    fm3d_texture* t = (fm3d_texture*)calloc(1, sizeof(fm3d_texture));
    if (!t) return NULL;
    t->refs = 1, t->kind = kind, t->nlayers = count, t->straight = (flags & FM3D_TEXTURE_STRAIGHT) != 0;
    if (count > 1) {
        t->lv = (fm_surface**)calloc((size_t)(count - 1) * 16, sizeof(fm_surface*));
        if (!t->lv) {
            free(t);
            return NULL;
        }
    }
    for (int i = 0; i < count; i++) {
        fm_surface** lv = i ? t->lv + (size_t)(i - 1) * 16 : t->level;
        lv[0]           = fm_surface_clone(layers[i]);
        if (!lv[0]) {
            fm3d_texture_release(t);
            return NULL;
        }
        int n     = (flags & FM3D_TEXTURE_MIPMAPS) ? fm3d_build_mips(lv) : 1;
        t->levels = i == 0 ? n : FM_MIN(t->levels, n);
    }
    return t;
}

fm3d_texture_kind fm3d_texture_get_kind(const fm3d_texture* t) { return t ? t->kind : FM3D_TEX_2D; }
int               fm3d_texture_layers(const fm3d_texture* t) { return t ? (t->nlayers > 0 ? t->nlayers : 1) : 0; }

fm3d_texture* fm3d_texture_retain(fm3d_texture* t)
{
    if (t) t->refs++;
    return t;
}

void fm3d_texture_release(fm3d_texture* t)
{
    if (!t || --t->refs > 0) return;
    for (int i = 0; i < 16; i++) fm_surface_destroy(t->level[i]);
    if (t->lv)
        for (int i = 0; i < (t->nlayers - 1) * 16; i++) fm_surface_destroy(t->lv[i]);
    free(t->lv);
    free(t);
}

int               fm3d_texture_levels(const fm3d_texture* t) { return t ? t->levels : 0; }
const fm_surface* fm3d_texture_level(const fm3d_texture* t, int level)
{
    return (t && level >= 0 && level < t->levels) ? t->level[level] : NULL;
}
