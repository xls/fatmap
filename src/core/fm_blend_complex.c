/* fatmap - non-trivial blend modes (W3C Compositing and Blending Level 1).
 * Float math per pixel; shared by all SIMD backends so output is identical. */
#include "fm_internal.h"
#include <math.h>

int fm_op_is_complex(int op)
{
    switch (op) {
    case FM_OP_OVERLAY:
    case FM_OP_COLOR_DODGE:
    case FM_OP_COLOR_BURN:
    case FM_OP_HARD_LIGHT:
    case FM_OP_SOFT_LIGHT:
    case FM_OP_HUE:
    case FM_OP_SATURATION:
    case FM_OP_COLOR:
    case FM_OP_LUMINOSITY: return 1;
    default: return 0;
    }
}

static float fm_screenf(float a, float b) { return a + b - a * b; }

static float fm_hard_light(float cb, float cs)
{
    return cs <= 0.5f ? cb * 2.0f * cs : fm_screenf(cb, 2.0f * cs - 1.0f);
}

static float fm_sep(int op, float cs, float cb)
{
    switch (op) {
    case FM_OP_OVERLAY: return fm_hard_light(cs, cb);
    case FM_OP_HARD_LIGHT: return fm_hard_light(cb, cs);
    case FM_OP_COLOR_DODGE:
        if (cb <= 0.0f) return 0.0f;
        if (cs >= 1.0f) return 1.0f;
        return fminf(1.0f, cb / (1.0f - cs));
    case FM_OP_COLOR_BURN:
        if (cb >= 1.0f) return 1.0f;
        if (cs <= 0.0f) return 0.0f;
        return 1.0f - fminf(1.0f, (1.0f - cb) / cs);
    case FM_OP_SOFT_LIGHT:
        if (cs <= 0.5f) return cb - (1.0f - 2.0f * cs) * cb * (1.0f - cb);
        {
            float d = cb <= 0.25f ? ((16.0f * cb - 12.0f) * cb + 4.0f) * cb : sqrtf(cb);
            return cb + (2.0f * cs - 1.0f) * (d - cb);
        }
    default: return cs;
    }
}

static float fm_lum(const float* c) { return 0.3f * c[0] + 0.59f * c[1] + 0.11f * c[2]; }

static void fm_clip_color(float* c)
{
    float l = fm_lum(c);
    float n = fminf(c[0], fminf(c[1], c[2]));
    float x = fmaxf(c[0], fmaxf(c[1], c[2]));
    for (int k = 0; k < 3; k++) {
        if (n < 0.0f && l - n > 1e-6f) c[k] = l + (c[k] - l) * l / (l - n);
        if (x > 1.0f && x - l > 1e-6f) c[k] = l + (c[k] - l) * (1.0f - l) / (x - l);
    }
}

static void fm_set_lum(float* c, float l)
{
    float d = l - fm_lum(c);
    c[0] += d;
    c[1] += d;
    c[2] += d;
    fm_clip_color(c);
}

static float fm_sat(const float* c) { return fmaxf(c[0], fmaxf(c[1], c[2])) - fminf(c[0], fminf(c[1], c[2])); }

static void fm_set_sat(float* c, float s)
{
    int imax = 0, imin = 0;
    for (int k = 1; k < 3; k++) {
        if (c[k] > c[imax]) imax = k;
        if (c[k] < c[imin]) imin = k;
    }
    if (imax == imin) {
        c[0] = c[1] = c[2] = 0.0f;
        return;
    }
    int imid = 3 - imax - imin;
    float range = c[imax] - c[imin];
    c[imid] = (c[imid] - c[imin]) * s / range;
    c[imax] = s;
    c[imin] = 0.0f;
}

static uint32_t fm_blend_complex_px(uint32_t s, uint32_t d, int op)
{
    float sa = (float)(s >> 24) / 255.0f, da = (float)(d >> 24) / 255.0f;
    float sc[3], dc[3], cs[3], cd[3], b[3];
    for (int k = 0; k < 3; k++) {
        int sh = 16 - 8 * k; /* R, G, B */
        sc[k]  = (float)((s >> sh) & 255) / 255.0f;
        dc[k]  = (float)((d >> sh) & 255) / 255.0f;
        cs[k]  = sa > 0.0f ? fminf(1.0f, sc[k] / sa) : 0.0f;
        cd[k]  = da > 0.0f ? fminf(1.0f, dc[k] / da) : 0.0f;
    }
    switch (op) {
    case FM_OP_HUE:
        memcpy(b, cs, sizeof(b));
        fm_set_sat(b, fm_sat(cd));
        fm_set_lum(b, fm_lum(cd));
        break;
    case FM_OP_SATURATION:
        memcpy(b, cd, sizeof(b));
        fm_set_sat(b, fm_sat(cs));
        fm_set_lum(b, fm_lum(cd));
        break;
    case FM_OP_COLOR:
        memcpy(b, cs, sizeof(b));
        fm_set_lum(b, fm_lum(cd));
        break;
    case FM_OP_LUMINOSITY:
        memcpy(b, cd, sizeof(b));
        fm_set_lum(b, fm_lum(cs));
        break;
    default:
        for (int k = 0; k < 3; k++) b[k] = fm_sep(op, cs[k], cd[k]);
        break;
    }
    float    ao = sa + da - sa * da;
    uint32_t a8 = (uint32_t)(FM_CLAMP(ao, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint32_t out = a8 << 24;
    for (int k = 0; k < 3; k++) {
        float co = sc[k] * (1.0f - da) + dc[k] * (1.0f - sa) + sa * da * b[k];
        co       = FM_CLAMP(co, 0.0f, ao);
        uint32_t c8 = (uint32_t)(co * 255.0f + 0.5f);
        if (c8 > a8) c8 = a8;
        out |= c8 << (16 - 8 * k);
    }
    return out;
}

void fm_span_op_complex(uint32_t* d, const uint32_t* s, const uint8_t* m, int n, int op)
{
    for (int i = 0; i < n; i++) {
        uint32_t c = m ? m[i] : 255u;
        if (c == 0) continue;
        uint32_t r = fm_blend_complex_px(s[i], d[i], op);
        if (c != 255) {
            uint32_t dd = d[i], o = 0, ic = 255 - c;
            for (int sh = 0; sh < 32; sh += 8)
                o |= fm_div255(((r >> sh) & 255) * c + ((dd >> sh) & 255) * ic) << sh;
            r = o;
        }
        d[i] = r;
    }
}
