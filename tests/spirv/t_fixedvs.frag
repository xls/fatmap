#version 450
// fixed function vertex stage: location 0 = uv, location 1 = color
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 frag;
void main() { frag = vec4(v_color.rgb * 0.5, v_color.a); }
