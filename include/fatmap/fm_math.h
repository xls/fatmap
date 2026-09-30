/*
 * fatmap - glm style math for C (header only).
 *
 * Conventions follow glm defaults: column-major matrices, right handed,
 * clip space depth -1..1 (fm_perspective / fm_ortho). The memory layout of
 * fm_vec*, fm_mat3, fm_mat4 and fm_quat (x,y,z,w) is identical to the glm
 * types, so C++ code can pass glm::mat4 straight through (see fatmap.hpp).
 *
 * glm                      fatmap
 * glm::translate(m, v)     fm_translate(m, v)
 * glm::rotate(m, a, axis)  fm_rotate(m, a, axis)
 * glm::scale(m, v)         fm_scale(m, v)
 * glm::perspective(...)    fm_perspective(...)
 * glm::lookAt(e, c, u)     fm_lookat(e, c, u)
 * m * n                    fm_mat4_mul(m, n)
 * m * v                    fm_mat4_mul_vec4(m, v)
 */
#ifndef FATMAP_FM_MATH_H
#define FATMAP_FM_MATH_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FM_PI 3.14159265358979323846f

typedef struct fm_vec2 { float x, y; } fm_vec2;
typedef struct fm_vec3 { float x, y, z; } fm_vec3;
typedef struct fm_vec4 { float x, y, z, w; } fm_vec4;
typedef struct fm_quat { float x, y, z, w; } fm_quat;
typedef struct fm_mat3 { fm_vec3 c[3]; } fm_mat3; /* c[column] */
typedef struct fm_mat4 { fm_vec4 c[4]; } fm_mat4; /* c[column] */

#define FM_MATH_FN static inline

FM_MATH_FN float fm_radians(float deg) { return deg * (FM_PI / 180.0f); }
FM_MATH_FN float fm_degrees(float rad) { return rad * (180.0f / FM_PI); }
FM_MATH_FN float fm_clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
FM_MATH_FN float fm_mixf(float a, float b, float t) { return a + (b - a) * t; }

/* ---- vec2 ---- */
FM_MATH_FN fm_vec2 fm_v2(float x, float y) { fm_vec2 r = { x, y }; return r; }
FM_MATH_FN fm_vec2 fm_v2_add(fm_vec2 a, fm_vec2 b) { return fm_v2(a.x + b.x, a.y + b.y); }
FM_MATH_FN fm_vec2 fm_v2_sub(fm_vec2 a, fm_vec2 b) { return fm_v2(a.x - b.x, a.y - b.y); }
FM_MATH_FN fm_vec2 fm_v2_scale(fm_vec2 a, float s) { return fm_v2(a.x * s, a.y * s); }
FM_MATH_FN float   fm_v2_dot(fm_vec2 a, fm_vec2 b) { return a.x * b.x + a.y * b.y; }
FM_MATH_FN float   fm_v2_length(fm_vec2 a) { return sqrtf(fm_v2_dot(a, a)); }

/* ---- vec3 ---- */
FM_MATH_FN fm_vec3 fm_v3(float x, float y, float z) { fm_vec3 r = { x, y, z }; return r; }
FM_MATH_FN fm_vec3 fm_v3_add(fm_vec3 a, fm_vec3 b) { return fm_v3(a.x + b.x, a.y + b.y, a.z + b.z); }
FM_MATH_FN fm_vec3 fm_v3_sub(fm_vec3 a, fm_vec3 b) { return fm_v3(a.x - b.x, a.y - b.y, a.z - b.z); }
FM_MATH_FN fm_vec3 fm_v3_mul(fm_vec3 a, fm_vec3 b) { return fm_v3(a.x * b.x, a.y * b.y, a.z * b.z); }
FM_MATH_FN fm_vec3 fm_v3_scale(fm_vec3 a, float s) { return fm_v3(a.x * s, a.y * s, a.z * s); }
FM_MATH_FN fm_vec3 fm_v3_negate(fm_vec3 a) { return fm_v3(-a.x, -a.y, -a.z); }
FM_MATH_FN float   fm_v3_dot(fm_vec3 a, fm_vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
FM_MATH_FN fm_vec3 fm_v3_cross(fm_vec3 a, fm_vec3 b)
{
    return fm_v3(a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y);
}
FM_MATH_FN float   fm_v3_length(fm_vec3 a) { return sqrtf(fm_v3_dot(a, a)); }
FM_MATH_FN fm_vec3 fm_v3_normalize(fm_vec3 a)
{
    float l = fm_v3_length(a);
    return l > 0.0f ? fm_v3_scale(a, 1.0f / l) : a;
}
FM_MATH_FN fm_vec3 fm_v3_mix(fm_vec3 a, fm_vec3 b, float t)
{
    return fm_v3(fm_mixf(a.x, b.x, t), fm_mixf(a.y, b.y, t), fm_mixf(a.z, b.z, t));
}

/* ---- vec4 ---- */
FM_MATH_FN fm_vec4 fm_v4(float x, float y, float z, float w) { fm_vec4 r = { x, y, z, w }; return r; }
FM_MATH_FN fm_vec4 fm_v4_from_v3(fm_vec3 v, float w) { return fm_v4(v.x, v.y, v.z, w); }
FM_MATH_FN fm_vec4 fm_v4_add(fm_vec4 a, fm_vec4 b) { return fm_v4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
FM_MATH_FN fm_vec4 fm_v4_sub(fm_vec4 a, fm_vec4 b) { return fm_v4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
FM_MATH_FN fm_vec4 fm_v4_scale(fm_vec4 a, float s) { return fm_v4(a.x * s, a.y * s, a.z * s, a.w * s); }
FM_MATH_FN float   fm_v4_dot(fm_vec4 a, fm_vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

/* ---- mat4 ---- */
FM_MATH_FN fm_mat4 fm_mat4_diag(float s)
{
    fm_mat4 m = { { { s, 0, 0, 0 }, { 0, s, 0, 0 }, { 0, 0, s, 0 }, { 0, 0, 0, s } } };
    return m;
}
FM_MATH_FN fm_mat4 fm_mat4_identity(void) { return fm_mat4_diag(1.0f); }
FM_MATH_FN const float* fm_mat4_ptr(const fm_mat4* m) { return &m->c[0].x; } /* glm::value_ptr */

FM_MATH_FN fm_vec4 fm_mat4_mul_vec4(fm_mat4 m, fm_vec4 v)
{
    fm_vec4 r;
    r.x = m.c[0].x * v.x + m.c[1].x * v.y + m.c[2].x * v.z + m.c[3].x * v.w;
    r.y = m.c[0].y * v.x + m.c[1].y * v.y + m.c[2].y * v.z + m.c[3].y * v.w;
    r.z = m.c[0].z * v.x + m.c[1].z * v.y + m.c[2].z * v.z + m.c[3].z * v.w;
    r.w = m.c[0].w * v.x + m.c[1].w * v.y + m.c[2].w * v.z + m.c[3].w * v.w;
    return r;
}

FM_MATH_FN fm_mat4 fm_mat4_mul(fm_mat4 a, fm_mat4 b)
{
    fm_mat4 r;
    for (int j = 0; j < 4; j++) r.c[j] = fm_mat4_mul_vec4(a, b.c[j]);
    return r;
}

FM_MATH_FN fm_mat4 fm_mat4_transpose(fm_mat4 m)
{
    fm_mat4 r;
    const float* s = fm_mat4_ptr(&m);
    float*       d = &r.c[0].x;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) d[i * 4 + j] = s[j * 4 + i];
    return r;
}

FM_MATH_FN fm_mat4 fm_mat4_inverse(fm_mat4 mat)
{
    const float* m = fm_mat4_ptr(&mat);
    float        inv[16];
    inv[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    fm_mat4 r;
    float*  d  = &r.c[0].x;
    float   id = det != 0.0f ? 1.0f / det : 0.0f;
    for (int i = 0; i < 16; i++) d[i] = inv[i] * id;
    return r;
}

FM_MATH_FN fm_mat4 fm_translate(fm_mat4 m, fm_vec3 v)
{
    fm_mat4 r = m;
    r.c[3]    = fm_v4_add(fm_v4_add(fm_v4_scale(m.c[0], v.x), fm_v4_scale(m.c[1], v.y)),
                          fm_v4_add(fm_v4_scale(m.c[2], v.z), m.c[3]));
    return r;
}

FM_MATH_FN fm_mat4 fm_scale(fm_mat4 m, fm_vec3 v)
{
    fm_mat4 r;
    r.c[0] = fm_v4_scale(m.c[0], v.x);
    r.c[1] = fm_v4_scale(m.c[1], v.y);
    r.c[2] = fm_v4_scale(m.c[2], v.z);
    r.c[3] = m.c[3];
    return r;
}

FM_MATH_FN fm_mat4 fm_rotate(fm_mat4 m, float angle, fm_vec3 v)
{
    float   c = cosf(angle), s = sinf(angle);
    fm_vec3 axis = fm_v3_normalize(v);
    fm_vec3 t    = fm_v3_scale(axis, 1.0f - c);
    float   r00 = c + t.x * axis.x, r01 = t.x * axis.y + s * axis.z, r02 = t.x * axis.z - s * axis.y;
    float   r10 = t.y * axis.x - s * axis.z, r11 = c + t.y * axis.y, r12 = t.y * axis.z + s * axis.x;
    float   r20 = t.z * axis.x + s * axis.y, r21 = t.z * axis.y - s * axis.x, r22 = c + t.z * axis.z;
    fm_mat4 r;
    r.c[0] = fm_v4_add(fm_v4_add(fm_v4_scale(m.c[0], r00), fm_v4_scale(m.c[1], r01)), fm_v4_scale(m.c[2], r02));
    r.c[1] = fm_v4_add(fm_v4_add(fm_v4_scale(m.c[0], r10), fm_v4_scale(m.c[1], r11)), fm_v4_scale(m.c[2], r12));
    r.c[2] = fm_v4_add(fm_v4_add(fm_v4_scale(m.c[0], r20), fm_v4_scale(m.c[1], r21)), fm_v4_scale(m.c[2], r22));
    r.c[3] = m.c[3];
    return r;
}

/* Right handed, depth -1..1 (glm::perspective default). */
FM_MATH_FN fm_mat4 fm_perspective(float fovy, float aspect, float znear, float zfar)
{
    float   th = tanf(fovy * 0.5f);
    fm_mat4 r  = fm_mat4_diag(0.0f);
    r.c[0].x   = 1.0f / (aspect * th);
    r.c[1].y   = 1.0f / th;
    r.c[2].z   = -(zfar + znear) / (zfar - znear);
    r.c[2].w   = -1.0f;
    r.c[3].z   = -(2.0f * zfar * znear) / (zfar - znear);
    return r;
}

/* Right handed, depth 0..1 (glm::perspectiveRH_ZO, Direct3D/Vulkan style). */
FM_MATH_FN fm_mat4 fm_perspective_zo(float fovy, float aspect, float znear, float zfar)
{
    float   th = tanf(fovy * 0.5f);
    fm_mat4 r  = fm_mat4_diag(0.0f);
    r.c[0].x   = 1.0f / (aspect * th);
    r.c[1].y   = 1.0f / th;
    r.c[2].z   = zfar / (znear - zfar);
    r.c[2].w   = -1.0f;
    r.c[3].z   = -(zfar * znear) / (zfar - znear);
    return r;
}

FM_MATH_FN fm_mat4 fm_ortho(float l, float r_, float b, float t, float n, float f)
{
    fm_mat4 r = fm_mat4_identity();
    r.c[0].x  = 2.0f / (r_ - l);
    r.c[1].y  = 2.0f / (t - b);
    r.c[2].z  = -2.0f / (f - n);
    r.c[3].x  = -(r_ + l) / (r_ - l);
    r.c[3].y  = -(t + b) / (t - b);
    r.c[3].z  = -(f + n) / (f - n);
    return r;
}

FM_MATH_FN fm_mat4 fm_lookat(fm_vec3 eye, fm_vec3 center, fm_vec3 up)
{
    fm_vec3 f = fm_v3_normalize(fm_v3_sub(center, eye));
    fm_vec3 s = fm_v3_normalize(fm_v3_cross(f, up));
    fm_vec3 u = fm_v3_cross(s, f);
    fm_mat4 r = fm_mat4_identity();
    r.c[0].x  = s.x;
    r.c[1].x  = s.y;
    r.c[2].x  = s.z;
    r.c[0].y  = u.x;
    r.c[1].y  = u.y;
    r.c[2].y  = u.z;
    r.c[0].z  = -f.x;
    r.c[1].z  = -f.y;
    r.c[2].z  = -f.z;
    r.c[3].x  = -fm_v3_dot(s, eye);
    r.c[3].y  = -fm_v3_dot(u, eye);
    r.c[3].z  = fm_v3_dot(f, eye);
    return r;
}

/* ---- mat3 ---- */
FM_MATH_FN fm_mat3 fm_mat3_from_mat4(fm_mat4 m)
{
    fm_mat3 r;
    for (int i = 0; i < 3; i++) r.c[i] = fm_v3(m.c[i].x, m.c[i].y, m.c[i].z);
    return r;
}

FM_MATH_FN fm_vec3 fm_mat3_mul_vec3(fm_mat3 m, fm_vec3 v)
{
    return fm_v3(m.c[0].x * v.x + m.c[1].x * v.y + m.c[2].x * v.z, m.c[0].y * v.x + m.c[1].y * v.y + m.c[2].y * v.z,
                 m.c[0].z * v.x + m.c[1].z * v.y + m.c[2].z * v.z);
}

/* Normal matrix: transpose(inverse(mat3(m))). */
FM_MATH_FN fm_mat3 fm_mat3_normal(fm_mat4 m4)
{
    fm_mat3 m = fm_mat3_from_mat4(m4);
    fm_vec3 a = m.c[0], b = m.c[1], c = m.c[2];
    fm_vec3 r0 = fm_v3_cross(b, c), r1 = fm_v3_cross(c, a), r2 = fm_v3_cross(a, b);
    float   det = fm_v3_dot(a, r0);
    float   id  = det != 0.0f ? 1.0f / det : 0.0f;
    fm_mat3 r;
    r.c[0] = fm_v3_scale(r0, id);
    r.c[1] = fm_v3_scale(r1, id);
    r.c[2] = fm_v3_scale(r2, id);
    return r;
}

/* ---- quaternions ---- */
FM_MATH_FN fm_quat fm_quat_identity(void) { fm_quat q = { 0, 0, 0, 1 }; return q; }
FM_MATH_FN fm_quat fm_quat_angle_axis(float angle, fm_vec3 axis)
{
    fm_vec3 a = fm_v3_normalize(axis);
    float   s = sinf(angle * 0.5f);
    fm_quat q = { a.x * s, a.y * s, a.z * s, cosf(angle * 0.5f) };
    return q;
}
FM_MATH_FN fm_quat fm_quat_mul(fm_quat p, fm_quat q)
{
    fm_quat r;
    r.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
    r.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
    r.y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z;
    r.z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x;
    return r;
}
FM_MATH_FN fm_mat4 fm_mat4_from_quat(fm_quat q)
{
    float   xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float   xz = q.x * q.z, xy = q.x * q.y, yz = q.y * q.z;
    float   wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    fm_mat4 r = fm_mat4_identity();
    r.c[0].x  = 1.0f - 2.0f * (yy + zz);
    r.c[0].y  = 2.0f * (xy + wz);
    r.c[0].z  = 2.0f * (xz - wy);
    r.c[1].x  = 2.0f * (xy - wz);
    r.c[1].y  = 1.0f - 2.0f * (xx + zz);
    r.c[1].z  = 2.0f * (yz + wx);
    r.c[2].x  = 2.0f * (xz + wy);
    r.c[2].y  = 2.0f * (yz - wx);
    r.c[2].z  = 1.0f - 2.0f * (xx + yy);
    return r;
}
FM_MATH_FN fm_quat fm_quat_slerp(fm_quat a, fm_quat b, float t)
{
    float cs = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (cs < 0.0f) {
        b.x = -b.x; b.y = -b.y; b.z = -b.z; b.w = -b.w;
        cs = -cs;
    }
    float ka, kb;
    if (cs > 1.0f - 1e-6f) {
        ka = 1.0f - t;
        kb = t;
    } else {
        float ang = acosf(cs), sn = sinf(ang);
        ka = sinf((1.0f - t) * ang) / sn;
        kb = sinf(t * ang) / sn;
    }
    fm_quat r = { a.x * ka + b.x * kb, a.y * ka + b.y * kb, a.z * ka + b.z * kb, a.w * ka + b.w * kb };
    return r;
}

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_MATH_H */
