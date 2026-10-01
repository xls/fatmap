#version 450
layout(location = 0) out vec4 frag;
vec4 helper(float x) { return vec4(x); }
void main() { frag = helper(0.5); }
