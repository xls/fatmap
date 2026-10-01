#version 450
// Doom 3 BFG Edition's light interaction vertex program (GPL, id Software;
// fatgl's tools/bench/interaction.vert) with Vulkan bindings: the
// game-like shader benchmark (fm_bench 3d_bfg_*)
layout(std140, binding = 0) uniform VA { vec4 _va_[18]; };

float dot3(vec3 a, vec4 b) { return dot(a, b.xyz); }
float dot4(vec4 a, vec4 b) { return dot(a, b); }
float dot4(vec2 a, vec4 b) { return dot(vec4(a, 0, 1), b); }

layout(location = 0) in vec4 in_Position;
layout(location = 1) in vec4 in_TexCoord;
layout(location = 2) in vec4 in_Normal;
layout(location = 3) in vec4 in_Tangent;
layout(location = 4) in vec4 in_Color;

layout(location = 0) out vec4 vofi_TexCoord0;
layout(location = 1) out vec4 vofi_TexCoord1;
layout(location = 2) out vec4 vofi_TexCoord2;
layout(location = 3) out vec4 vofi_TexCoord3;
layout(location = 4) out vec4 vofi_TexCoord4;
layout(location = 5) out vec4 vofi_TexCoord5;
layout(location = 6) out vec4 vofi_TexCoord6;
layout(location = 7) out vec4 vofi_Color;

void main() {
    vec3 vNormal = in_Normal.xyz * 2.0 - 1.0;
    vec4 vTangent = in_Tangent * 2.0 - 1.0;
    vec3 vBinormal = cross(vNormal.xyz, vTangent.xyz) * vTangent.w;
    gl_Position.x = dot4(in_Position, _va_[14]);
    gl_Position.y = dot4(in_Position, _va_[15]);
    gl_Position.z = dot4(in_Position, _va_[16]);
    gl_Position.w = dot4(in_Position, _va_[17]);
    vec4 defaultTexCoord = vec4(0.0, 0.5, 0.0, 1.0);
    vec4 toLight = _va_[0] - in_Position;
    vofi_TexCoord0.x = dot3(vTangent.xyz, toLight);
    vofi_TexCoord0.y = dot3(vBinormal, toLight);
    vofi_TexCoord0.z = dot3(vNormal, toLight);
    vofi_TexCoord0.w = 1.0;
    vofi_TexCoord1 = defaultTexCoord;
    vofi_TexCoord1.x = dot4(in_TexCoord.xy, _va_[6]);
    vofi_TexCoord1.y = dot4(in_TexCoord.xy, _va_[7]);
    vofi_TexCoord2 = defaultTexCoord;
    vofi_TexCoord2.x = dot4(in_Position, _va_[5]);
    vofi_TexCoord3.x = dot4(in_Position, _va_[2]);
    vofi_TexCoord3.y = dot4(in_Position, _va_[3]);
    vofi_TexCoord3.z = 0.0;
    vofi_TexCoord3.w = dot4(in_Position, _va_[4]);
    vofi_TexCoord4 = defaultTexCoord;
    vofi_TexCoord4.x = dot4(in_TexCoord.xy, _va_[8]);
    vofi_TexCoord4.y = dot4(in_TexCoord.xy, _va_[9]);
    vofi_TexCoord5 = defaultTexCoord;
    vofi_TexCoord5.x = dot4(in_TexCoord.xy, _va_[10]);
    vofi_TexCoord5.y = dot4(in_TexCoord.xy, _va_[11]);
    toLight = normalize(toLight);
    vec4 toView = normalize(_va_[1] - in_Position);
    vec4 halfAngleVector = toLight + toView;
    vofi_TexCoord6.x = dot3(vTangent.xyz, halfAngleVector);
    vofi_TexCoord6.y = dot3(vBinormal, halfAngleVector);
    vofi_TexCoord6.z = dot3(vNormal, halfAngleVector);
    vofi_TexCoord6.w = 1.0;
    vofi_Color = (in_Color * _va_[12]) + _va_[13];
}
