/* sandbox helper: JPEG / PNG decoding (stb_image) into fatmap surfaces */
#include "image_load.h"

#include <stb_image.h>
#include <stdio.h>
#include <stdlib.h>

fm_surface* img_load_mem(const void* data, size_t len)
{
    int            w, h, n;
    unsigned char* px = stbi_load_from_memory((const unsigned char*)data, (int)len, &w, &h, &n, 4);
    if (!px) return NULL;
    fm_surface* s = fm_surface_from_rgba8(px, w, h, w * 4);
    stbi_image_free(px);
    return s;
}

fm_surface* img_load_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void*       d = sz > 0 ? malloc((size_t)sz) : NULL;
    fm_surface* s = NULL;
    if (d && fread(d, 1, (size_t)sz, f) == (size_t)sz) s = img_load_mem(d, (size_t)sz);
    fclose(f);
    free(d);
    return s;
}
