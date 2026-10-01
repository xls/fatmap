#version 450
// a {x, y} grid in 0..1, pinned at x = 0, travelling wave, analytic normal
layout(location = 0) in vec2 xy;
layout(binding = 0) uniform U { mat4 mvp; mat4 mv; vec4 light; float t; } u;
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_n;
void main()
{
    float ph   = 7.0 * xy.x - 3.2 * u.t + 1.3 * xy.y;
    float a    = 0.16 * xy.x;
    vec3  p    = vec3(xy.x * 2.0, xy.y * 1.2, a * sin(ph));
    float dzdx = (0.16 * sin(ph) + a * 7.0 * cos(ph)) / 2.0;
    float dzdy = (a * 1.3 * cos(ph)) / 1.2;
    gl_Position = u.mvp * vec4(p, 1.0);
    v_uv = xy;
    v_n  = (u.mv * vec4(-dzdx, -dzdy, 1.0, 0.0)).xyz;
}
