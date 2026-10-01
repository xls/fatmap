#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 frag;
layout(binding = 1) uniform samplerCube cube;
layout(binding = 2) uniform sampler2DArray arr;
layout(binding = 0) uniform U { mat4 mvp; vec4 tint; float time; } u;
void main()
{
    /* v_color.xyz: a direction (cube) or (s, t, layer) */
    if (u.tint.x > 0.5) frag = texture(cube, v_color.xyz * 2.0 - 1.0);
    else frag = texture(arr, vec3(0.5, 0.5, v_color.z * 3.0));
}
