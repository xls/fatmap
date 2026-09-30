/* fatmap - colors, CSS color parsing, blend op names */
#include "fm_internal.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

fm_color fm_color_from_floats(float r, float g, float b, float a)
{
#define FM_CH(v) ((uint32_t)(FM_CLAMP((v), 0.0f, 1.0f) * 255.0f + 0.5f))
    return FM_ARGB(FM_CH(a), FM_CH(r), FM_CH(g), FM_CH(b));
#undef FM_CH
}

uint32_t fm_premultiply(fm_color c) { return fm_premul_inline(c); }

fm_color fm_unpremultiply(uint32_t p)
{
    uint32_t a = p >> 24;
    if (a == 255) return p;
    if (a == 0) return 0;
    uint32_t r = FM_MIN(255u, (((p >> 16) & 255) * 255 + a / 2) / a);
    uint32_t g = FM_MIN(255u, (((p >> 8) & 255) * 255 + a / 2) / a);
    uint32_t b = FM_MIN(255u, ((p & 255) * 255 + a / 2) / a);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

/* ---- blend op names -------------------------------------------------------- */

static const char* const g_op_names[FM_OP_COUNT] = {
    "source-over", "source-in",   "source-out",  "source-atop", "destination-over", "destination-in",
    "destination-out", "destination-atop", "lighter", "copy", "xor", "clear", "multiply", "screen",
    "overlay", "darken", "lighten", "color-dodge", "color-burn", "hard-light", "soft-light", "difference",
    "exclusion", "hue", "saturation", "color", "luminosity",
};

const char* fm_blend_op_name(fm_blend_op op)
{
    return (op >= 0 && op < FM_OP_COUNT) ? g_op_names[op] : "unknown";
}

int fm_blend_op_from_name(const char* name)
{
    if (!name) return -1;
    for (int i = 0; i < FM_OP_COUNT; i++)
        if (strcmp(name, g_op_names[i]) == 0) return i;
    return -1;
}

/* ---- CSS colors ------------------------------------------------------------ */

typedef struct { const char* name; uint32_t rgb; } fm_named_color;

static const fm_named_color g_named[] = {
    { "aliceblue", 0xf0f8ff }, { "antiquewhite", 0xfaebd7 }, { "aqua", 0x00ffff }, { "aquamarine", 0x7fffd4 },
    { "azure", 0xf0ffff }, { "beige", 0xf5f5dc }, { "bisque", 0xffe4c4 }, { "black", 0x000000 },
    { "blanchedalmond", 0xffebcd }, { "blue", 0x0000ff }, { "blueviolet", 0x8a2be2 }, { "brown", 0xa52a2a },
    { "burlywood", 0xdeb887 }, { "cadetblue", 0x5f9ea0 }, { "chartreuse", 0x7fff00 }, { "chocolate", 0xd2691e },
    { "coral", 0xff7f50 }, { "cornflowerblue", 0x6495ed }, { "cornsilk", 0xfff8dc }, { "crimson", 0xdc143c },
    { "cyan", 0x00ffff }, { "darkblue", 0x00008b }, { "darkcyan", 0x008b8b }, { "darkgoldenrod", 0xb8860b },
    { "darkgray", 0xa9a9a9 }, { "darkgreen", 0x006400 }, { "darkgrey", 0xa9a9a9 }, { "darkkhaki", 0xbdb76b },
    { "darkmagenta", 0x8b008b }, { "darkolivegreen", 0x556b2f }, { "darkorange", 0xff8c00 },
    { "darkorchid", 0x9932cc }, { "darkred", 0x8b0000 }, { "darksalmon", 0xe9967a }, { "darkseagreen", 0x8fbc8f },
    { "darkslateblue", 0x483d8b }, { "darkslategray", 0x2f4f4f }, { "darkslategrey", 0x2f4f4f },
    { "darkturquoise", 0x00ced1 }, { "darkviolet", 0x9400d3 }, { "deeppink", 0xff1493 },
    { "deepskyblue", 0x00bfff }, { "dimgray", 0x696969 }, { "dimgrey", 0x696969 }, { "dodgerblue", 0x1e90ff },
    { "firebrick", 0xb22222 }, { "floralwhite", 0xfffaf0 }, { "forestgreen", 0x228b22 }, { "fuchsia", 0xff00ff },
    { "gainsboro", 0xdcdcdc }, { "ghostwhite", 0xf8f8ff }, { "gold", 0xffd700 }, { "goldenrod", 0xdaa520 },
    { "gray", 0x808080 }, { "green", 0x008000 }, { "greenyellow", 0xadff2f }, { "grey", 0x808080 },
    { "honeydew", 0xf0fff0 }, { "hotpink", 0xff69b4 }, { "indianred", 0xcd5c5c }, { "indigo", 0x4b0082 },
    { "ivory", 0xfffff0 }, { "khaki", 0xf0e68c }, { "lavender", 0xe6e6fa }, { "lavenderblush", 0xfff0f5 },
    { "lawngreen", 0x7cfc00 }, { "lemonchiffon", 0xfffacd }, { "lightblue", 0xadd8e6 }, { "lightcoral", 0xf08080 },
    { "lightcyan", 0xe0ffff }, { "lightgoldenrodyellow", 0xfafad2 }, { "lightgray", 0xd3d3d3 },
    { "lightgreen", 0x90ee90 }, { "lightgrey", 0xd3d3d3 }, { "lightpink", 0xffb6c1 }, { "lightsalmon", 0xffa07a },
    { "lightseagreen", 0x20b2aa }, { "lightskyblue", 0x87cefa }, { "lightslategray", 0x778899 },
    { "lightslategrey", 0x778899 }, { "lightsteelblue", 0xb0c4de }, { "lightyellow", 0xffffe0 },
    { "lime", 0x00ff00 }, { "limegreen", 0x32cd32 }, { "linen", 0xfaf0e6 }, { "magenta", 0xff00ff },
    { "maroon", 0x800000 }, { "mediumaquamarine", 0x66cdaa }, { "mediumblue", 0x0000cd },
    { "mediumorchid", 0xba55d3 }, { "mediumpurple", 0x9370db }, { "mediumseagreen", 0x3cb371 },
    { "mediumslateblue", 0x7b68ee }, { "mediumspringgreen", 0x00fa9a }, { "mediumturquoise", 0x48d1cc },
    { "mediumvioletred", 0xc71585 }, { "midnightblue", 0x191970 }, { "mintcream", 0xf5fffa },
    { "mistyrose", 0xffe4e1 }, { "moccasin", 0xffe4b5 }, { "navajowhite", 0xffdead }, { "navy", 0x000080 },
    { "oldlace", 0xfdf5e6 }, { "olive", 0x808000 }, { "olivedrab", 0x6b8e23 }, { "orange", 0xffa500 },
    { "orangered", 0xff4500 }, { "orchid", 0xda70d6 }, { "palegoldenrod", 0xeee8aa }, { "palegreen", 0x98fb98 },
    { "paleturquoise", 0xafeeee }, { "palevioletred", 0xdb7093 }, { "papayawhip", 0xffefd5 },
    { "peachpuff", 0xffdab9 }, { "peru", 0xcd853f }, { "pink", 0xffc0cb }, { "plum", 0xdda0dd },
    { "powderblue", 0xb0e0e6 }, { "purple", 0x800080 }, { "rebeccapurple", 0x663399 }, { "red", 0xff0000 },
    { "rosybrown", 0xbc8f8f }, { "royalblue", 0x4169e1 }, { "saddlebrown", 0x8b4513 }, { "salmon", 0xfa8072 },
    { "sandybrown", 0xf4a460 }, { "seagreen", 0x2e8b57 }, { "seashell", 0xfff5ee }, { "sienna", 0xa0522d },
    { "silver", 0xc0c0c0 }, { "skyblue", 0x87ceeb }, { "slateblue", 0x6a5acd }, { "slategray", 0x708090 },
    { "slategrey", 0x708090 }, { "snow", 0xfffafa }, { "springgreen", 0x00ff7f }, { "steelblue", 0x4682b4 },
    { "tan", 0xd2b48c }, { "teal", 0x008080 }, { "thistle", 0xd8bfd8 }, { "tomato", 0xff6347 },
    { "turquoise", 0x40e0d0 }, { "violet", 0xee82ee }, { "wheat", 0xf5deb3 }, { "white", 0xffffff },
    { "whitesmoke", 0xf5f5f5 }, { "yellow", 0xffff00 }, { "yellowgreen", 0x9acd32 },
};

static int fm_hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    c = tolower(c);
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static const char* fm_skip_ws(const char* p)
{
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

/* Parses a number, optional '%'. Returns pointer after it or NULL. */
static const char* fm_parse_num(const char* p, float* v, int* pct)
{
    char* end;
    p       = fm_skip_ws(p);
    float f = strtof(p, &end);
    if (end == p) return NULL;
    *v = f;
    p    = end;
    *pct = 0;
    if (*p == '%') {
        *pct = 1;
        p++;
    }
    p = fm_skip_ws(p);
    if (*p == ',' || *p == '/') p++;
    return p;
}

static float fm_hue2rgb(float p, float q, float t)
{
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1.0f / 6) return p + (q - p) * 6 * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3) return p + (q - p) * (2.0f / 3 - t) * 6;
    return p;
}

int fm_color_parse(const char* css, fm_color* out)
{
    if (!css || !out) return 0;
    char        buf[64];
    const char* p = fm_skip_ws(css);
    size_t      n = 0;
    while (p[n] && n < sizeof(buf) - 1) {
        buf[n] = (char)tolower((unsigned char)p[n]);
        n++;
    }
    while (n > 0 && isspace((unsigned char)buf[n - 1])) n--;
    buf[n] = 0;
    p      = buf;

    if (p[0] == '#') {
        int d[8], len = (int)n - 1;
        if (len != 3 && len != 4 && len != 6 && len != 8) return 0;
        for (int i = 0; i < len; i++)
            if ((d[i] = fm_hexval(p[1 + i])) < 0) return 0;
        uint32_t r, g, b, a = 255;
        if (len <= 4) {
            r = (uint32_t)d[0] * 17;
            g = (uint32_t)d[1] * 17;
            b = (uint32_t)d[2] * 17;
            if (len == 4) a = (uint32_t)d[3] * 17;
        } else {
            r = (uint32_t)(d[0] * 16 + d[1]);
            g = (uint32_t)(d[2] * 16 + d[3]);
            b = (uint32_t)(d[4] * 16 + d[5]);
            if (len == 8) a = (uint32_t)(d[6] * 16 + d[7]);
        }
        *out = FM_ARGB(a, r, g, b);
        return 1;
    }
    if (strcmp(p, "transparent") == 0) {
        *out = 0;
        return 1;
    }
    int is_rgb = strncmp(p, "rgb", 3) == 0, is_hsl = strncmp(p, "hsl", 3) == 0;
    if (is_rgb || is_hsl) {
        const char* q = strchr(p, '(');
        if (!q) return 0;
        q++;
        float v[4] = { 0, 0, 0, 1 };
        int   pc[4] = { 0, 0, 0, 0 }, cnt = 0;
        while (cnt < 4) {
            const char* r = fm_parse_num(q, &v[cnt], &pc[cnt]);
            if (!r) break;
            q = r;
            cnt++;
        }
        if (cnt < 3 || *fm_skip_ws(q) != ')') return 0;
        float a = pc[3] ? v[3] / 100.0f : v[3];
        float r, g, b;
        if (is_rgb) {
            r = pc[0] ? v[0] / 100.0f : v[0] / 255.0f;
            g = pc[1] ? v[1] / 100.0f : v[1] / 255.0f;
            b = pc[2] ? v[2] / 100.0f : v[2] / 255.0f;
        } else {
            float h = fmodf(v[0], 360.0f) / 360.0f, s = v[1] / 100.0f, l = v[2] / 100.0f;
            if (h < 0) h += 1.0f;
            s       = FM_CLAMP(s, 0.0f, 1.0f);
            l       = FM_CLAMP(l, 0.0f, 1.0f);
            float q2 = l < 0.5f ? l * (1 + s) : l + s - l * s;
            float p2 = 2 * l - q2;
            r        = fm_hue2rgb(p2, q2, h + 1.0f / 3);
            g        = fm_hue2rgb(p2, q2, h);
            b        = fm_hue2rgb(p2, q2, h - 1.0f / 3);
        }
        *out = fm_color_from_floats(r, g, b, a);
        return 1;
    }
    for (size_t i = 0; i < sizeof(g_named) / sizeof(g_named[0]); i++) {
        if (strcmp(p, g_named[i].name) == 0) {
            *out = 0xff000000u | g_named[i].rgb;
            return 1;
        }
    }
    return 0;
}
