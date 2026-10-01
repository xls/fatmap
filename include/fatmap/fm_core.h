/*
 * fatmap - core: version, colors, pixel formats, surfaces, blend ops,
 * low-level span kernels and CPU/SIMD dispatch.
 *
 * Pixel convention: FM_FORMAT_ARGB32 surfaces store premultiplied alpha,
 * one uint32_t per pixel laid out as 0xAARRGGBB in native endianness
 * (bytes B,G,R,A on little endian). This matches SDL_PIXELFORMAT_ARGB8888,
 * Windows DIBs and most GPU "BGRA8" swapchains.
 *
 * API colors (fm_color) are straight (non-premultiplied) 0xAARRGGBB.
 */
#ifndef FATMAP_FM_CORE_H
#define FATMAP_FM_CORE_H

#include <fatmap/fm_config.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Static linking is the default. Define FM_SHARED when building or using
 * fatmap as a Windows DLL. */
#if defined(_WIN32) && defined(FM_SHARED)
#  ifdef FM_BUILD
#    define FM_API __declspec(dllexport)
#  else
#    define FM_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(FM_BUILD)
#  define FM_API __attribute__((visibility("default")))
#else
#  define FM_API
#endif

#define FM_VERSION_MAJOR 0
#define FM_VERSION_MINOR 1
#define FM_VERSION_PATCH 0

FM_API const char* fm_version_string(void);
/* Detects the CPU and selects SIMD kernels. Called implicitly by every
 * create function; thread safe. */
FM_API void        fm_init(void);

/* ------------------------------------------------------------------------
 * Colors
 * ---------------------------------------------------------------------- */

typedef uint32_t fm_color; /* straight alpha, 0xAARRGGBB */

#define FM_ARGB(a, r, g, b) \
    ((fm_color)(((uint32_t)(a) << 24) | ((uint32_t)(r) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(b)))
#define FM_RGB(r, g, b)     FM_ARGB(255, r, g, b)
#define FM_RGBA(r, g, b, a) FM_ARGB(a, r, g, b)

FM_API fm_color fm_color_from_floats(float r, float g, float b, float a);
/* CSS color strings: #rgb #rgba #rrggbb #rrggbbaa rgb() rgba() and the
 * CSS named colors. Returns 1 on success. */
FM_API int      fm_color_parse(const char* css, fm_color* out);
FM_API uint32_t fm_premultiply(fm_color c);
FM_API fm_color fm_unpremultiply(uint32_t premul);

/* ------------------------------------------------------------------------
 * Surfaces
 * ---------------------------------------------------------------------- */

typedef enum fm_format {
    FM_FORMAT_ARGB32 = 0, /* premultiplied 0xAARRGGBB, 4 bytes/pixel */
    FM_FORMAT_A8     = 1, /* coverage / alpha masks, 1 byte/pixel */
    FM_FORMAT_D32F   = 2, /* depth buffer, float 0..1, 4 bytes/pixel */
    FM_FORMAT_D16    = 3, /* depth buffer, 16 bit unorm, 2 bytes/pixel */
    FM_FORMAT_D24S8  = 4  /* packed: bits 0-23 depth (unorm), bits 24-31 stencil */
} fm_format;

typedef struct fm_surface {
    void*     data;
    int       width;
    int       height;
    int       stride; /* bytes per row */
    fm_format format;
    int       owns_data;
} fm_surface;

FM_API int fm_format_bpp(fm_format f);      /* bytes per pixel */
FM_API int fm_format_is_depth(fm_format f); /* D32F, D16, D24S8 */

/* Rows are 64-byte aligned for owned surfaces. */
FM_API fm_surface* fm_surface_create(int width, int height, fm_format format);
/* Wrap external memory (e.g. a locked SDL texture or a GPU staging buffer). */
FM_API fm_surface* fm_surface_wrap(void* data, int width, int height, int stride_bytes, fm_format format);
/* A view into a sub-rectangle of another surface (shares memory). */
FM_API fm_surface* fm_surface_sub(const fm_surface* parent, int x, int y, int w, int h);
FM_API fm_surface* fm_surface_clone(const fm_surface* s);
/* Import straight-alpha RGBA8 bytes (R,G,B,A byte order) into a new ARGB32 surface. */
FM_API fm_surface* fm_surface_from_rgba8(const void* rgba, int width, int height, int stride_bytes);
FM_API void        fm_surface_destroy(fm_surface* s);
FM_API void        fm_surface_clear(fm_surface* s, fm_color c);
/* depth surfaces: fill with a depth value (D24S8 keeps the stencil bits) */
FM_API void        fm_surface_clear_depth(fm_surface* s, float depth);
/* depth value at (x, y) as 0..1 (depth formats), 0 otherwise */
FM_API float       fm_surface_get_depth(const fm_surface* s, int x, int y);
FM_API fm_color    fm_surface_get_pixel(const fm_surface* s, int x, int y);
/* TGA (types 2 and 10: truecolor, uncompressed or RLE, 24/32 bit) into a
 * premultiplied ARGB32 surface. NULL on failure. */
FM_API fm_surface* fm_surface_load_tga(const char* path);
/* uncompressed 32 bit TGA (straight alpha). Returns 1 on success. */
FM_API int         fm_surface_write_tga(const fm_surface* s, const char* path);
/* Writes an uncompressed PNG (no dependencies). Returns 1 on success. */
FM_API int         fm_surface_write_png(const fm_surface* s, const char* path);

static inline uint32_t* fm_surface_row32(const fm_surface* s, int y)
{
    return (uint32_t*)((uint8_t*)s->data + (size_t)y * (size_t)s->stride);
}
static inline uint8_t* fm_surface_row8(const fm_surface* s, int y)
{
    return (uint8_t*)s->data + (size_t)y * (size_t)s->stride;
}
static inline float* fm_surface_rowf(const fm_surface* s, int y)
{
    return (float*)((uint8_t*)s->data + (size_t)y * (size_t)s->stride);
}

/* ------------------------------------------------------------------------
 * Swapchain: always render into the back buffer, present to swap.
 * The front buffer stays untouched until the next present, so it can be
 * displayed / uploaded while the next frame renders.
 * ---------------------------------------------------------------------- */

typedef struct fm_swapchain fm_swapchain;

/* count: 2 (double) or 3 (triple) color buffers; with_depth adds one shared
 * D32F depth buffer. */
FM_API fm_swapchain* fm_swapchain_create(int width, int height, int count, int with_depth);
/* same with an explicit depth format (FM_FORMAT_D32F / D16 / D24S8) */
FM_API fm_swapchain* fm_swapchain_create_depth(int width, int height, int count, fm_format depth_format);
FM_API void          fm_swapchain_destroy(fm_swapchain* sc);
FM_API fm_surface*   fm_swapchain_back(fm_swapchain* sc);  /* render target for this frame */
FM_API fm_surface*   fm_swapchain_depth(fm_swapchain* sc); /* shared depth buffer or NULL */
FM_API fm_surface*   fm_swapchain_front(fm_swapchain* sc); /* last presented frame */
/* Back becomes front (returned), the next buffer becomes the back buffer. */
FM_API fm_surface*   fm_swapchain_present(fm_swapchain* sc);

/* ------------------------------------------------------------------------
 * Blend ops (HTML canvas globalCompositeOperation + clear)
 * Coverage is applied as: dst = lerp(dst, op(src, dst), coverage).
 * ---------------------------------------------------------------------- */

typedef enum fm_blend_op {
    FM_OP_SRC_OVER = 0,
    FM_OP_SRC_IN,
    FM_OP_SRC_OUT,
    FM_OP_SRC_ATOP,
    FM_OP_DST_OVER,
    FM_OP_DST_IN,
    FM_OP_DST_OUT,
    FM_OP_DST_ATOP,
    FM_OP_LIGHTER,
    FM_OP_COPY,
    FM_OP_XOR,
    FM_OP_CLEAR,
    FM_OP_MULTIPLY,
    FM_OP_SCREEN,
    FM_OP_OVERLAY,
    FM_OP_DARKEN,
    FM_OP_LIGHTEN,
    FM_OP_COLOR_DODGE,
    FM_OP_COLOR_BURN,
    FM_OP_HARD_LIGHT,
    FM_OP_SOFT_LIGHT,
    FM_OP_DIFFERENCE,
    FM_OP_EXCLUSION,
    FM_OP_HUE,
    FM_OP_SATURATION,
    FM_OP_COLOR,
    FM_OP_LUMINOSITY,
    FM_OP_COUNT
} fm_blend_op;

FM_API const char* fm_blend_op_name(fm_blend_op op);       /* canvas name, e.g. "source-over" */
FM_API int         fm_blend_op_from_name(const char* name); /* -1 if unknown */

/* ------------------------------------------------------------------------
 * Low-level span kernels (SIMD dispatched). All pixels premultiplied ARGB32.
 * cov may be NULL (full coverage). These are the building blocks the
 * rasterizer/pipeline use; exposed for custom backends.
 * ---------------------------------------------------------------------- */

FM_API void fm_fill_span(uint32_t* dst, uint32_t value, int n);
FM_API void fm_blend_span(uint32_t* dst, const uint32_t* src, const uint8_t* cov, int n, fm_blend_op op);
FM_API void fm_blend_solid(uint32_t* dst, uint32_t src_premul, const uint8_t* cov, int n, fm_blend_op op);

/* ------------------------------------------------------------------------
 * CPU features / SIMD dispatch
 * ---------------------------------------------------------------------- */

typedef enum fm_simd_level {
    FM_SIMD_SCALAR = 0,
    FM_SIMD_SSE2   = 1,
    FM_SIMD_AVX2   = 2,
    FM_SIMD_NEON   = 3,
    FM_SIMD_AVX512 = 4 /* AVX-512 F / DQ / BW / VL; the 2D / 3D kernels are the AVX2 ones for now,
                          the shader backends use the full width */
} fm_simd_level;
#define FM_SIMD_LEVELS 5

#define FM_CPU_SSE2  (1u << 0)
#define FM_CPU_SSE41 (1u << 1)
#define FM_CPU_AVX2  (1u << 2)
#define FM_CPU_FMA   (1u << 3)
#define FM_CPU_NEON  (1u << 4)
#define FM_CPU_AVX512 (1u << 5) /* F + DQ + BW + VL, enabled by the OS */

FM_API unsigned      fm_cpu_features(void);
FM_API fm_simd_level fm_simd_best(void);    /* best level compiled in and supported by this CPU */
FM_API fm_simd_level fm_simd_current(void);
FM_API int           fm_simd_supported(fm_simd_level level);
/* Process wide; do not change while other threads are rendering. */
FM_API int           fm_simd_set(fm_simd_level level); /* returns 1 if applied */
FM_API const char*   fm_simd_name(fm_simd_level level);

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_CORE_H */
