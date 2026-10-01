#version 450
// fm3d_vertex: position @0, normal @12, uv @24 (attributes 0, 1, 2)
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 nrm;
layout(location = 2) in vec2 uv;
layout(binding = 0) uniform U { mat4 mvp; mat4 mv; vec4 light; float pulse; } u;
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_eye;
layout(location = 2) out vec3 v_n;
void main()
{
    gl_Position = u.mvp * vec4(pos, 1.0);
    v_uv  = uv;
    v_eye = (u.mv * vec4(pos, 1.0)).xyz;
    v_n   = (u.mv * vec4(nrm, 0.0)).xyz;
}
