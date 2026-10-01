#version 450
// per pixel Blinn-Phong + rim light + pulsing emissive (as the C shader)
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_eye;
layout(location = 2) in vec3 v_n;
layout(location = 0) out vec4 frag;
layout(binding = 0) uniform U { mat4 mvp; mat4 mv; vec4 light; float pulse; } u;
layout(binding = 1) uniform sampler2D base_ao;
layout(binding = 2) uniform sampler2D emissive;
void main()
{
    vec3  base = texture(base_ao, v_uv).rgb;
    vec3  em   = texture(emissive, v_uv).rgb;
    vec3  n    = normalize(v_n);
    vec3  v    = normalize(-v_eye);
    vec3  l    = u.light.xyz;
    float ndl  = max(dot(n, l), 0.0);
    float ndh  = dot(n, normalize(l + v));
    float sp   = (ndl > 0.0 && ndh > 0.0) ? pow(ndh, 48.0) * 0.7 : 0.0;
    float rim  = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.55;
    float lit  = 0.16 + 0.9 * ndl;
    vec3  c    = base * lit + vec3(sp) + rim * vec3(0.30, 0.55, 1.0) + em * u.pulse;
    frag       = vec4(clamp(c, 0.0, 1.0), 1.0);
}
