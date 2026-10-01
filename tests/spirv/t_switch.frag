#version 450
// early returns (inlined by glslc -O into a switch construct) + a real switch with fall through
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 frag;
vec3 pat(vec2 p)
{
    if (p.x < 0.5) return vec3(1.0, 0.0, 0.0);
    if (p.y < 0.5) return vec3(0.0, 1.0, 0.0);
    return vec3(0.0, 0.0, 1.0);
}
void main()
{
    int  k = int(v_uv.x * 4.0) & 3;
    vec3 c = pat(fract(v_uv * 3.0));
    switch (k) {
    case 0: c *= 0.5; break;
    case 1: c += 0.25; // falls through
    case 2: c = c.bgr; break;
    default: c = vec3(1.0) - c; break;
    }
    frag = vec4(c, 1.0);
}
