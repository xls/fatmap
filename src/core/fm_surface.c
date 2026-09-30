/* fatmap - surfaces + dependency free PNG writer */
#include "fm_internal.h"
#include <stdio.h>

int fm_format_bpp(fm_format f)
{
    switch (f) {
    case FM_FORMAT_A8: return 1;
    case FM_FORMAT_D16: return 2;
    default: return 4;
    }
}
int fm_format_is_depth(fm_format f) { return f == FM_FORMAT_D32F || f == FM_FORMAT_D16 || f == FM_FORMAT_D24S8; }
static int fm_bpp(fm_format f) { return fm_format_bpp(f); }

void fm_surface_clear_depth(fm_surface* s, float depth)
{
    if (!s || !fm_format_is_depth(s->format)) return;
    depth = FM_CLAMP(depth, 0.0f, 1.0f);
    for (int y = 0; y < s->height; y++) {
        uint8_t* row = fm_surface_row8(s, y);
        if (s->format == FM_FORMAT_D32F) {
            float* r = (float*)row;
            for (int x = 0; x < s->width; x++) r[x] = depth;
        } else if (s->format == FM_FORMAT_D16) {
            uint16_t* r = (uint16_t*)row;
            uint16_t  q = (uint16_t)(depth * 65535.0f + 0.5f);
            for (int x = 0; x < s->width; x++) r[x] = q;
        } else {
            uint32_t* r = (uint32_t*)row;
            uint32_t  q = (uint32_t)((double)depth * 16777215.0 + 0.5);
            for (int x = 0; x < s->width; x++) r[x] = (r[x] & 0xff000000u) | q;
        }
    }
}

float fm_surface_get_depth(const fm_surface* s, int x, int y)
{
    if (!s || x < 0 || y < 0 || x >= s->width || y >= s->height) return 0.0f;
    const uint8_t* row = fm_surface_row8(s, y);
    switch (s->format) {
    case FM_FORMAT_D32F: return ((const float*)row)[x];
    case FM_FORMAT_D16: return (float)((const uint16_t*)row)[x] / 65535.0f;
    case FM_FORMAT_D24S8: return (float)((double)(((const uint32_t*)row)[x] & 0xffffffu) / 16777215.0);
    default: return 0.0f;
    }
}

/* ---- swapchain ---------------------------------------------------------------- */

struct fm_swapchain {
    fm_surface* buf[3];
    fm_surface* depth;
    int         count;
    int         back;
};

fm_swapchain* fm_swapchain_create(int width, int height, int count, int with_depth)
{
    return fm_swapchain_create_depth(width, height, count, with_depth ? FM_FORMAT_D32F : FM_FORMAT_ARGB32);
}

fm_swapchain* fm_swapchain_create_depth(int width, int height, int count, fm_format depth_format)
{
    int with_depth = fm_format_is_depth(depth_format);
    if (count < 2) count = 2;
    if (count > 3) count = 3;
    fm_swapchain* sc = (fm_swapchain*)calloc(1, sizeof(fm_swapchain));
    if (!sc) return NULL;
    sc->count = count;
    for (int i = 0; i < count; i++) {
        sc->buf[i] = fm_surface_create(width, height, FM_FORMAT_ARGB32);
        if (!sc->buf[i]) {
            fm_swapchain_destroy(sc);
            return NULL;
        }
    }
    if (with_depth) {
        sc->depth = fm_surface_create(width, height, depth_format);
        if (!sc->depth) {
            fm_swapchain_destroy(sc);
            return NULL;
        }
        fm_surface_clear_depth(sc->depth, 1.0f);
    }
    return sc;
}

void fm_swapchain_destroy(fm_swapchain* sc)
{
    if (!sc) return;
    for (int i = 0; i < 3; i++) fm_surface_destroy(sc->buf[i]);
    fm_surface_destroy(sc->depth);
    free(sc);
}

fm_surface* fm_swapchain_back(fm_swapchain* sc) { return sc->buf[sc->back]; }
fm_surface* fm_swapchain_depth(fm_swapchain* sc) { return sc->depth; }
fm_surface* fm_swapchain_front(fm_swapchain* sc) { return sc->buf[(sc->back + sc->count - 1) % sc->count]; }

fm_surface* fm_swapchain_present(fm_swapchain* sc)
{
    fm_surface* front = sc->buf[sc->back];
    sc->back          = (sc->back + 1) % sc->count;
    return front;
}

fm_surface* fm_surface_create(int width, int height, fm_format format)
{
    fm__init();
    if (width <= 0 || height <= 0) return NULL;
    fm_surface* s = (fm_surface*)calloc(1, sizeof(fm_surface));
    if (!s) return NULL;
    int stride = (width * fm_bpp(format) + 63) & ~63;
    s->data    = fm_aligned_alloc((size_t)stride * (size_t)height, 64);
    if (!s->data) {
        free(s);
        return NULL;
    }
    memset(s->data, 0, (size_t)stride * (size_t)height);
    s->width     = width;
    s->height    = height;
    s->stride    = stride;
    s->format    = format;
    s->owns_data = 1;
    return s;
}

fm_surface* fm_surface_wrap(void* data, int width, int height, int stride_bytes, fm_format format)
{
    fm__init();
    if (!data || width <= 0 || height <= 0) return NULL;
    fm_surface* s = (fm_surface*)calloc(1, sizeof(fm_surface));
    if (!s) return NULL;
    s->data   = data;
    s->width  = width;
    s->height = height;
    s->stride = stride_bytes;
    s->format = format;
    return s;
}

fm_surface* fm_surface_sub(const fm_surface* p, int x, int y, int w, int h)
{
    if (!p) return NULL;
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > p->width) w = p->width - x;
    if (y + h > p->height) h = p->height - y;
    if (w <= 0 || h <= 0) return NULL;
    uint8_t* base = (uint8_t*)p->data + (size_t)y * (size_t)p->stride + (size_t)x * (size_t)fm_bpp(p->format);
    return fm_surface_wrap(base, w, h, p->stride, p->format);
}

fm_surface* fm_surface_clone(const fm_surface* src)
{
    if (!src) return NULL;
    fm_surface* s = fm_surface_create(src->width, src->height, src->format);
    if (!s) return NULL;
    size_t row = (size_t)src->width * (size_t)fm_bpp(src->format);
    for (int y = 0; y < src->height; y++)
        memcpy((uint8_t*)s->data + (size_t)y * s->stride, (const uint8_t*)src->data + (size_t)y * src->stride, row);
    return s;
}

fm_surface* fm_surface_from_rgba8(const void* rgba, int width, int height, int stride_bytes)
{
    fm_surface* s = fm_surface_create(width, height, FM_FORMAT_ARGB32);
    if (!s || !rgba) return s;
    for (int y = 0; y < height; y++) {
        const uint8_t* src = (const uint8_t*)rgba + (size_t)y * (size_t)stride_bytes;
        uint32_t*      dst = fm_surface_row32(s, y);
        for (int x = 0; x < width; x++, src += 4) dst[x] = fm_premul_inline(FM_ARGB(src[3], src[0], src[1], src[2]));
    }
    return s;
}

void fm_surface_destroy(fm_surface* s)
{
    if (!s) return;
    if (s->owns_data) fm_aligned_free(s->data);
    free(s);
}

void fm_surface_clear(fm_surface* s, fm_color c)
{
    if (!s) return;
    if (s->format == FM_FORMAT_A8) {
        for (int y = 0; y < s->height; y++) memset(fm_surface_row8(s, y), (int)(c >> 24), (size_t)s->width);
        return;
    }
    if (fm_format_is_depth(s->format)) return; /* use fm_surface_clear_depth */
    uint32_t p = fm_premul_inline(c);
    for (int y = 0; y < s->height; y++) fm_k->fill(fm_surface_row32(s, y), p, s->width);
}

fm_color fm_surface_get_pixel(const fm_surface* s, int x, int y)
{
    if (!s || x < 0 || y < 0 || x >= s->width || y >= s->height) return 0;
    if (s->format == FM_FORMAT_A8) return (fm_color)fm_surface_row8(s, y)[x] << 24;
    if (fm_format_is_depth(s->format)) return 0;
    return fm_unpremultiply(fm_surface_row32(s, y)[x]);
}

/* ---- PNG (stored deflate blocks, no compression) --------------------------- */

static uint32_t g_crc_table[256];

static void fm_crc_init(void)
{
    if (g_crc_table[1]) return;
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
        g_crc_table[n] = c;
    }
}

static uint32_t fm_crc(uint32_t crc, const uint8_t* p, size_t n)
{
    crc = ~crc;
    while (n--) crc = g_crc_table[(crc ^ *p++) & 255] ^ (crc >> 8);
    return ~crc;
}

typedef struct {
    FILE*    f;
    uint32_t crc;
    int      ok;
} fm_png_w;

static void fm_png_bytes(fm_png_w* w, const void* p, size_t n)
{
    if (fwrite(p, 1, n, w->f) != n) w->ok = 0;
    w->crc = fm_crc(w->crc, (const uint8_t*)p, n);
}

static void fm_be32(uint8_t* b, uint32_t v)
{
    b[0] = (uint8_t)(v >> 24);
    b[1] = (uint8_t)(v >> 16);
    b[2] = (uint8_t)(v >> 8);
    b[3] = (uint8_t)v;
}

static void fm_png_chunk_begin(fm_png_w* w, const char* type, uint32_t len)
{
    uint8_t b[4];
    fm_be32(b, len);
    if (fwrite(b, 1, 4, w->f) != 4) w->ok = 0;
    w->crc = 0;
    fm_png_bytes(w, type, 4);
}

static void fm_png_chunk_end(fm_png_w* w)
{
    uint8_t b[4];
    fm_be32(b, w->crc);
    if (fwrite(b, 1, 4, w->f) != 4) w->ok = 0;
}

int fm_surface_write_png(const fm_surface* s, const char* path)
{
    if (!s || !path) return 0;
    fm_crc_init();
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    fm_png_w w = { f, 0, 1 };
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    if (fwrite(sig, 1, 8, f) != 8) w.ok = 0;

    uint8_t ihdr[13];
    fm_be32(ihdr, (uint32_t)s->width);
    fm_be32(ihdr + 4, (uint32_t)s->height);
    ihdr[8]  = 8; /* bit depth */
    ihdr[9]  = 6; /* RGBA */
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    fm_png_chunk_begin(&w, "IHDR", 13);
    fm_png_bytes(&w, ihdr, 13);
    fm_png_chunk_end(&w);

    size_t   row_len = (size_t)s->width * 4 + 1;
    size_t   raw_len = row_len * (size_t)s->height;
    size_t   nblocks = (raw_len + 65534) / 65535;
    uint32_t idat_len = (uint32_t)(2 + raw_len + nblocks * 5 + 4);
    uint8_t* row = (uint8_t*)malloc(row_len);
    if (!row) {
        fclose(f);
        return 0;
    }
    fm_png_chunk_begin(&w, "IDAT", idat_len);
    static const uint8_t zhdr[2] = { 0x78, 0x01 };
    fm_png_bytes(&w, zhdr, 2);

    uint32_t a1 = 1, a2 = 0; /* adler32 */
    size_t   left_in_block = 0, remaining = raw_len;
    for (int y = 0; y < s->height; y++) {
        row[0] = 0;
        for (int x = 0; x < s->width; x++) {
            fm_color c;
            if (s->format == FM_FORMAT_A8) {
                uint8_t a = fm_surface_row8(s, y)[x];
                c         = FM_ARGB(255, a, a, a);
            } else if (fm_format_is_depth(s->format)) {
                float   d = fm_surface_get_depth(s, x, y);
                uint8_t g = (uint8_t)(FM_CLAMP(d, 0.0f, 1.0f) * 255.0f + 0.5f);
                c         = FM_ARGB(255, g, g, g);
            } else {
                c = fm_unpremultiply(fm_surface_row32(s, y)[x]);
            }
            uint8_t* o = row + 1 + x * 4;
            o[0]       = (uint8_t)(c >> 16);
            o[1]       = (uint8_t)(c >> 8);
            o[2]       = (uint8_t)c;
            o[3]       = (uint8_t)(c >> 24);
        }
        size_t off = 0;
        while (off < row_len) {
            if (left_in_block == 0) {
                size_t  bl = remaining < 65535 ? remaining : 65535;
                uint8_t bh[5];
                bh[0] = (uint8_t)(remaining <= 65535 ? 1 : 0);
                bh[1] = (uint8_t)(bl & 255);
                bh[2] = (uint8_t)(bl >> 8);
                bh[3] = (uint8_t)(~bl & 255);
                bh[4] = (uint8_t)((~bl >> 8) & 255);
                fm_png_bytes(&w, bh, 5);
                left_in_block = bl;
            }
            size_t take = row_len - off;
            if (take > left_in_block) take = left_in_block;
            fm_png_bytes(&w, row + off, take);
            for (size_t i = 0; i < take; i++) {
                a1 = (a1 + row[off + i]) % 65521;
                a2 = (a2 + a1) % 65521;
            }
            off += take;
            left_in_block -= take;
            remaining -= take;
        }
    }
    uint8_t ad[4];
    fm_be32(ad, (a2 << 16) | a1);
    fm_png_bytes(&w, ad, 4);
    fm_png_chunk_end(&w);
    free(row);

    fm_png_chunk_begin(&w, "IEND", 0);
    fm_png_chunk_end(&w);
    int ok = w.ok;
    if (fclose(f) != 0) ok = 0;
    return ok;
}

/* ---- TGA ------------------------------------------------------------------ */

fm_surface* fm_surface_load_tga(const char* path)
{
    FILE* f = path ? fopen(path, "rb") : NULL;
    if (!f) return NULL;
    uint8_t h[18];
    if (fread(h, 1, 18, f) != 18) {
        fclose(f);
        return NULL;
    }
    int type = h[2], w = h[12] | (h[13] << 8), hh = h[14] | (h[15] << 8), bpp = h[16], desc = h[17];
    if ((type != 2 && type != 10) || (bpp != 24 && bpp != 32) || w <= 0 || hh <= 0 || h[1] != 0) {
        fclose(f);
        return NULL;
    }
    fseek(f, h[0], SEEK_CUR); /* image id */
    fm_surface* s  = fm_surface_create(w, hh, FM_FORMAT_ARGB32);
    int         bp = bpp / 8, top = (desc & 0x20) != 0, ok = s != NULL;
    size_t      total = (size_t)w * (size_t)hh;
    uint8_t*    px    = ok ? (uint8_t*)malloc(total * 4) : NULL; /* BGRA */
    ok               = ok && px;
    size_t i         = 0;
    while (ok && i < total) {
        uint8_t c[4] = { 0, 0, 0, 255 };
        if (type == 2) {
            if (fread(c, 1, (size_t)bp, f) != (size_t)bp) ok = 0;
            memcpy(px + i * 4, c, 4);
            if (bp == 3) px[i * 4 + 3] = 255;
            i++;
            continue;
        }
        int hdr = fgetc(f);
        if (hdr < 0) {
            ok = 0;
            break;
        }
        size_t n = (size_t)(hdr & 127) + 1;
        if (i + n > total) n = total - i;
        if (hdr & 128) { /* run */
            if (fread(c, 1, (size_t)bp, f) != (size_t)bp) ok = 0;
            if (bp == 3) c[3] = 255;
            for (size_t k = 0; k < n; k++) memcpy(px + (i + k) * 4, c, 4);
        } else {
            for (size_t k = 0; k < n && ok; k++) {
                if (fread(c, 1, (size_t)bp, f) != (size_t)bp) ok = 0;
                if (bp == 3) c[3] = 255;
                memcpy(px + (i + k) * 4, c, 4);
            }
        }
        i += n;
    }
    fclose(f);
    if (!ok) {
        free(px);
        fm_surface_destroy(s);
        return NULL;
    }
    for (int y = 0; y < hh; y++) {
        const uint8_t* src = px + (size_t)(top ? y : hh - 1 - y) * (size_t)w * 4;
        uint32_t*      dst = fm_surface_row32(s, y);
        for (int x = 0; x < w; x++, src += 4) dst[x] = fm_premul_inline(FM_ARGB(src[3], src[2], src[1], src[0]));
    }
    free(px);
    return s;
}

int fm_surface_write_tga(const fm_surface* s, const char* path)
{
    if (!s || !path || s->format != FM_FORMAT_ARGB32 || s->width > 65535 || s->height > 65535) return 0;
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    uint8_t h[18] = { 0 };
    h[2]          = 2;
    h[12]         = (uint8_t)(s->width & 255);
    h[13]         = (uint8_t)(s->width >> 8);
    h[14]         = (uint8_t)(s->height & 255);
    h[15]         = (uint8_t)(s->height >> 8);
    h[16]         = 32;
    h[17]         = 0x28; /* top-left origin, 8 alpha bits */
    int ok        = fwrite(h, 1, 18, f) == 18;
    uint8_t* row  = (uint8_t*)malloc((size_t)s->width * 4);
    ok            = ok && row;
    for (int y = 0; ok && y < s->height; y++) {
        const uint32_t* src = fm_surface_row32(s, y);
        for (int x = 0; x < s->width; x++) {
            fm_color c     = fm_unpremultiply(src[x]);
            row[x * 4]     = (uint8_t)c;
            row[x * 4 + 1] = (uint8_t)(c >> 8);
            row[x * 4 + 2] = (uint8_t)(c >> 16);
            row[x * 4 + 3] = (uint8_t)(c >> 24);
        }
        ok = fwrite(row, 1, (size_t)s->width * 4, f) == (size_t)s->width * 4;
    }
    free(row);
    if (fclose(f) != 0) ok = 0;
    return ok;
}
