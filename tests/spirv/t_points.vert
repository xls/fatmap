#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;
layout(binding = 0) uniform U { mat4 mvp; vec4 tint; float time; } u;
void main()
{
    gl_Position = u.mvp * vec4(pos, 1.0);
    gl_PointSize = color.a * 16.0; /* the size comes from the vertex */
}
