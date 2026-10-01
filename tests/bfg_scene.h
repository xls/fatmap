/* The Doom 3 BFG style interaction benchmark scene (fm_bench 3d_bfg_*): a
 * copy of fatgl's tools/bench/bench_scene.h, which fatgl and Mesa llvmpipe
 * (bench_interaction.c / .py) render; keep them in step. */
#ifndef BENCH_SCENE_H
#define BENCH_SCENE_H
#include <stdint.h>

#define SCENE_W      1280
#define SCENE_H      720
#define SCENE_GX     64 /* grid cells */
#define SCENE_GY     36
#define SCENE_NV     ((SCENE_GX + 1) * (SCENE_GY + 1))
#define SCENE_NI     (SCENE_GX * SCENE_GY * 6)
#define SCENE_LIGHTS 4

static uint32_t scene_hash(uint32_t x, uint32_t y, uint32_t k)
{
    uint32_t h = (x * 73856093u) ^ (y * 19349663u) ^ (k * 83492791u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

/* 0 bump (normal in a / g), 1 falloff, 2 light projection, 3 YCoCg diffuse, 4 specular */
static int scene_tex_w(int i) { return i == 1 ? 64 : (i == 2 ? 128 : 256); }
static int scene_tex_h(int i) { return i == 1 ? 16 : (i == 2 ? 128 : 256); }
static int scene_tex_clamp(int i) { return i == 1 || i == 2; }

static void scene_texture(int i, uint8_t* out)
{
    int w = scene_tex_w(i), h = scene_tex_h(i);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t* p = out + ((size_t)y * (size_t)w + (size_t)x) * 4;
            uint32_t r = scene_hash((uint32_t)x, (uint32_t)y, (uint32_t)i);
            int      v;
            switch (i) {
            case 0: p[0] = 128, p[1] = (uint8_t)(96 + (r & 63)), p[2] = 255, p[3] = (uint8_t)(96 + ((r >> 8) & 63)); break;
            case 1:
                v    = 255 - 4 * x;
                v    = v < 0 ? 0 : v;
                p[0] = p[1] = p[2] = (uint8_t)v, p[3] = 255;
                break;
            case 2: {
                int d = (x - 64) * (x - 64) + (y - 64) * (y - 64);
                v     = 255 - d * 255 / 4096;
                v     = v < 0 ? 0 : v;
                p[0] = (uint8_t)v, p[1] = (uint8_t)(v * 3 / 4), p[2] = (uint8_t)(v / 2), p[3] = 255;
                break;
            }
            case 3: p[0] = (uint8_t)(120 + (r & 15)), p[1] = (uint8_t)(120 + ((r >> 4) & 15)), p[2] = 8, p[3] = (uint8_t)(64 + ((x ^ y) & 127)); break;
            default: p[0] = p[1] = p[2] = (uint8_t)(64 + (r & 63)), p[3] = 255; break;
            }
        }
}

typedef struct scene_vert {
    float   xyzw[4];
    float   st[2];
    uint8_t normal[4], tangent[4], color[4];
} scene_vert; /* 36 bytes */

static void scene_mesh(scene_vert* v, uint32_t* idx)
{
    for (int j = 0; j <= SCENE_GY; j++)
        for (int i = 0; i <= SCENE_GX; i++) {
            scene_vert* o = &v[j * (SCENE_GX + 1) + i];
            o->xyzw[0] = -1.0f + 2.0f * (float)i / (float)SCENE_GX;
            o->xyzw[1] = -1.0f + 2.0f * (float)j / (float)SCENE_GY;
            o->xyzw[2] = 0.02f * (float)((i * 7 + j * 3) % 11) / 11.0f;
            o->xyzw[3] = 1.0f;
            o->st[0] = (float)i / (float)SCENE_GX, o->st[1] = (float)j / (float)SCENE_GY;
            o->normal[0] = (uint8_t)(120 + (i * 5 + j) % 17), o->normal[1] = (uint8_t)(120 + (j * 3 + i) % 17), o->normal[2] = 250, o->normal[3] = 0;
            o->tangent[0] = 255, o->tangent[1] = 128, o->tangent[2] = 128, o->tangent[3] = 255;
            o->color[0] = o->color[1] = o->color[2] = o->color[3] = 255;
        }
    int k = 0;
    for (int j = 0; j < SCENE_GY; j++)
        for (int i = 0; i < SCENE_GX; i++) {
            uint32_t a = (uint32_t)(j * (SCENE_GX + 1) + i), b = a + 1, c = a + SCENE_GX + 1, d = c + 1;
            idx[k++] = a, idx[k++] = b, idx[k++] = d, idx[k++] = a, idx[k++] = d, idx[k++] = c;
        }
}

static void scene_uniforms(int l, float va[18][4], float fa[2][4])
{
    static const float lp[4][2] = { { -0.5f, -0.4f }, { 0.5f, -0.3f }, { -0.3f, 0.5f }, { 0.4f, 0.4f } };
    static const float lc[4][3] = { { 1.0f, 0.8f, 0.6f }, { 0.6f, 0.8f, 1.0f }, { 0.8f, 1.0f, 0.7f }, { 1.0f, 1.0f, 1.0f } };
    float              lx = lp[l][0], ly = lp[l][1];
    const float        v[18][4] = { { lx, ly, 0.6f, 1.0f },
                                    { 0.0f, 0.0f, 3.0f, 1.0f },
                                    { 0.6f, 0.0f, 0.0f, 0.5f - 0.6f * lx }, /* light projection S T Q */
                                    { 0.0f, 0.6f, 0.0f, 0.5f - 0.6f * ly },
                                    { 0.0f, 0.0f, 0.0f, 1.0f },
                                    { 0.0f, 0.0f, 10.0f, 0.3f }, /* falloff S */
                                    { 4.0f, 0.0f, 0.0f, 0.0f },  /* bump S T, diffuse S T, specular S T */
                                    { 0.0f, 4.0f, 0.0f, 0.0f },
                                    { 4.0f, 0.0f, 0.0f, 0.0f },
                                    { 0.0f, 4.0f, 0.0f, 0.0f },
                                    { 4.0f, 0.0f, 0.0f, 0.0f },
                                    { 0.0f, 4.0f, 0.0f, 0.0f },
                                    { 0.0f, 0.0f, 0.0f, 0.0f }, /* vertex color modulate, add */
                                    { 1.0f, 1.0f, 1.0f, 1.0f },
                                    { 1.0f, 0.0f, 0.0f, 0.0f }, /* MVP rows */
                                    { 0.0f, 1.0f, 0.0f, 0.0f },
                                    { 0.0f, 0.0f, 0.5f, 0.0f },
                                    { 0.0f, 0.0f, 0.0f, 1.0f } };
    for (int r = 0; r < 18; r++)
        for (int c = 0; c < 4; c++) va[r][c] = v[r][c];
    for (int c = 0; c < 3; c++) fa[0][c] = lc[l][c], fa[1][c] = 0.8f * lc[l][c];
    fa[0][3] = fa[1][3] = 1.0f;
}

#endif
