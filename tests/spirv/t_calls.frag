#version 450
// function calls: compiled with -O0 (calls kept) the inliner must render what
// glslc -O (its own inlining) renders
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 frag;

// early returns from inside a loop and a branch
float first_above(float x, float limit)
{
    float acc = 0.0;
    for (int i = 0; i < 8; i++) {
        acc += x * float(i + 1);
        if (acc > limit) return float(i) / 8.0;
        if (acc < -1.0) return -1.0;
    }
    return 1.0;
}

// out / inout parameters
void split(vec3 c, out float lo, out float hi, inout int count)
{
    lo = min(c.r, min(c.g, c.b));
    hi = max(c.r, max(c.g, c.b));
    count += 1;
}

// nested calls, a call in a condition
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
bool bright(vec3 c) { return lum(c) > 0.5; }
vec3 tone(vec3 c)
{
    if (bright(c)) return c * 0.8;
    return pow(c, vec3(0.9));
}

void main()
{
    int   n = 0;
    float lo, hi;
    split(v_color.rgb, lo, hi, n);
    float s = first_above(v_uv.x * 0.37 + 0.05, 2.0 + v_uv.y);
    vec3  c = tone(v_color.rgb);
    for (int k = 0; k < 3; k++) // a call in a loop body
        if (bright(c * float(k + 1) * 0.5)) n += 2;
    frag = vec4(c * (0.5 + 0.5 * s) + vec3(lo, hi, 0.0) * 0.1, float(n) / 8.0 * v_color.a);
}
