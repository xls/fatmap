#version 450
// bands + checker hoist, 4x rotated grid supersampled along the derivatives
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_n;
layout(location = 0) out vec4 frag;
layout(binding = 0) uniform U { mat4 mvp; mat4 mv; vec4 light; float t; } u;
vec3 pattern(vec2 p)
{
    if (p.x < 0.3 && p.y > 0.45) {
        int k = (int(p.x * 20.0) + int(p.y * 12.0)) & 1;
        return k == 1 ? vec3(0.95) : vec3(0.1, 0.1, 0.12);
    }
    if (p.y > 0.66) return vec3(0.95, 0.55, 0.1);
    if (p.y > 0.33) return vec3(0.92, 0.92, 0.88);
    return vec3(0.15, 0.4, 0.85);
}
void main()
{
    const vec2 o[4] = vec2[4](vec2(-0.125, -0.375), vec2(0.375, -0.125), vec2(0.125, 0.375), vec2(-0.375, 0.125));
    vec2 dx = dFdx(v_uv), dy = dFdy(v_uv);
    vec3 c  = vec3(0.0);
    for (int k = 0; k < 4; k++) c += 0.25 * pattern(v_uv + dx * o[k].x + dy * o[k].y);
    float d  = dot(normalize(v_n), u.light.xyz);
    float li = 0.25 + 0.85 * abs(d);
    frag     = vec4(clamp(c * li, 0.0, 1.0), 1.0);
}
