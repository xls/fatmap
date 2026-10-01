#version 450
layout(location = 0) out vec4 frag;
void main() { frag = vec4(gl_PointCoord, gl_FrontFacing ? 1.0 : 0.0, 1.0); }
