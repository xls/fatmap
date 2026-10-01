#version 450
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 frag;
layout(binding = 0) uniform U { mat4 mvp; vec4 tint; float time; } u;
layout(binding = 1) uniform sampler2D tex;
void main()
{
    vec3 c = v_color.rgb * u.tint.rgb;
    float acc = 0.0;
    for (int i = 0; i < 4; i++)
        acc += sin(float(i) * v_uv.x + u.time);
    if (v_color.r > 0.5)
        c = mix(c, vec3(acc * 0.25), 0.5);
    else
        c = pow(c, vec3(2.2));
    if (v_uv.y > 1.9) discard;
    float w = fwidth(v_uv.x);
    vec4 t = texture(tex, v_uv);
    frag = vec4(clamp(c + t.rgb * 0.1 + w, 0.0, 1.0), v_color.a);
}
