#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 frag;
layout(binding = 0) uniform U { mat4 mvp; vec4 tint; float time; } u;
void main() { frag = v_color * u.tint; }
