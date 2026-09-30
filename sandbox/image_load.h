/* sandbox helper: JPEG / PNG decoding (stb_image) into fatmap surfaces */
#ifndef FATMAP_SANDBOX_IMAGE_LOAD_H
#define FATMAP_SANDBOX_IMAGE_LOAD_H

#include <fatmap/fatmap.h>
#include <stddef.h>

/* straight alpha input, premultiplied ARGB32 surface out; NULL on failure */
fm_surface* img_load_mem(const void* data, size_t len);
fm_surface* img_load_file(const char* path);

#endif
