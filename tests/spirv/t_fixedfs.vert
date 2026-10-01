#version 450
// for the fixed function fragment stage: location 0 = uv, location 1 = color
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(binding = 0) uniform U { mat4 mvp; vec4 tint; float time; } u;
void main() { gl_Position = u.mvp * vec4(pos, 1.0); v_uv = pos.xy * 0.01; v_color = color; }
