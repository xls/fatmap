#version 450
// uniform blocks at two bindings (fm3d_set_uniform_block)
layout(location = 0) out vec4 frag;
layout(std140, binding = 0) uniform A { vec4 tint; } a;
layout(std140, binding = 3) uniform B { vec4 color; float scale; } b;
void main() { frag = vec4(a.tint.rgb * b.color.rgb * b.scale, 1.0); }
