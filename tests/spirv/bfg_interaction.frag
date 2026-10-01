#version 450
// Doom 3 BFG Edition's light interaction fragment program (GPL, id Software;
// fatgl's tools/bench/interaction.frag) with Vulkan bindings: bump, light
// falloff and projection, YCoCg diffuse, specular (fm_bench 3d_bfg_*)
layout(std140, binding = 1) uniform FA { vec4 _fa_[2]; };
layout(binding = 0) uniform sampler2D samp0;
layout(binding = 1) uniform sampler2D samp1;
layout(binding = 2) uniform sampler2D samp2;
layout(binding = 3) uniform sampler2D samp3;
layout(binding = 4) uniform sampler2D samp4;

layout(location = 0) in vec4 vofi_TexCoord0;
layout(location = 1) in vec4 vofi_TexCoord1;
layout(location = 2) in vec4 vofi_TexCoord2;
layout(location = 3) in vec4 vofi_TexCoord3;
layout(location = 4) in vec4 vofi_TexCoord4;
layout(location = 5) in vec4 vofi_TexCoord5;
layout(location = 6) in vec4 vofi_TexCoord6;
layout(location = 7) in vec4 vofi_Color;

layout(location = 0) out vec4 fragColor;

float dot3(vec3 a, vec3 b) { return dot(a, b); }
float dot4(vec4 a, vec4 b) { return dot(a, b); }
const vec4 matrixCoCg1YtoRGB1X = vec4(1.0, -1.0, 0.0, 1.0);
const vec4 matrixCoCg1YtoRGB1Y = vec4(0.0, 1.0, -0.50196078, 1.0);
const vec4 matrixCoCg1YtoRGB1Z = vec4(-1.0, -1.0, 1.00392156, 1.0);
vec3 ConvertYCoCgToRGB(vec4 YCoCg) {
    vec3 rgbColor;
    YCoCg.z = (YCoCg.z * 31.875) + 1.0;
    YCoCg.z = 1.0 / YCoCg.z;
    YCoCg.xy *= YCoCg.z;
    rgbColor.x = dot4(YCoCg, matrixCoCg1YtoRGB1X);
    rgbColor.y = dot4(YCoCg, matrixCoCg1YtoRGB1Y);
    rgbColor.z = dot4(YCoCg, matrixCoCg1YtoRGB1Z);
    return rgbColor;
}

void main() {
    vec4 bumpMap = texture(samp0, vofi_TexCoord1.xy);
    vec4 lightFalloff = textureProj(samp1, vofi_TexCoord2.xyw);
    vec4 lightProj = textureProj(samp2, vofi_TexCoord3.xyw);
    vec4 YCoCG = texture(samp3, vofi_TexCoord4.xy);
    vec4 specMap = texture(samp4, vofi_TexCoord5.xy);
    vec3 lightVector = normalize(vofi_TexCoord0.xyz);
    vec3 diffuseMap = ConvertYCoCgToRGB(YCoCG);
    vec3 localNormal;
    localNormal.xy = bumpMap.wy - 0.5;
    localNormal.z = sqrt(abs(dot(localNormal.xy, localNormal.xy) - 0.25));
    localNormal = normalize(localNormal);
    float specularPower = 10.0;
    float hDotN = dot3(normalize(vofi_TexCoord6.xyz), localNormal);
    vec3 specularContribution = vec3(pow(max(hDotN, 0.0), specularPower));
    vec3 diffuseColor = diffuseMap * _fa_[0].xyz;
    vec3 specularColor = specMap.xyz * specularContribution * _fa_[1].xyz;
    vec3 lightColor = dot3(lightVector, localNormal) * lightProj.xyz * lightFalloff.xyz;
    fragColor.xyz = (diffuseColor + specularColor) * lightColor * vofi_Color.xyz;
    fragColor.w = 1.0;
}
