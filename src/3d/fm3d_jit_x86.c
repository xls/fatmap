/*
 * fatmap - SPIR-V JIT: x86 machine code (AVX2 / AVX-512, x86-64 and x86-32).
 */
#include "fm3d_jit.h"

#if FM_FEATURE_SPIRV && FM_FEATURE_JIT

#include <stdio.h>

int fmj_compile_x86(fmj_prog* p, char* err, size_t errn)
{
    (void)p;
    if (err && errn) snprintf(err, errn, "no machine code yet");
    return 0;
}

#endif
