/* Build step: the test / benchmark SPIR-V programs (tests/spirv_shaders.h)
 * translated to C with fm3d_spirv_to_c, written to argv[1]. fm3d_test and
 * fm_bench compare them with the interpreter. */
#include <fatmap/fatmap.h>
#include <stdio.h>
#include <string.h>
#include "spirv_shaders.h"

typedef struct aot_prog {
    const char*     name;
    const uint32_t* vs;
    size_t          vs_words;
    const uint32_t* fs;
    size_t          fs_words;
    int             fast; /* fm3d_spirv_set_fast_math */
} aot_prog;

#define SPV(a) a, sizeof(a) / 4

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: spirv_aot_gen out.c\n");
        return 2;
    }
    static const fm3d_vertex_attrib attr[2] = { { 0, 3, 0 }, { 1, 4, 12 } }; /* sv_tvert / bsv_vert */
    const aot_prog progs[] = {
        { "aot_color", SPV(spv_t_basic_vert), SPV(spv_t_color_frag), 0 },
        { "aot_control", SPV(spv_t_basic_vert), SPV(spv_t_control_frag), 0 },
        { "aot_switch", SPV(spv_t_basic_vert), SPV(spv_t_switch_frag), 0 },
        { "aot_fixedvs", NULL, 0, SPV(spv_t_fixedvs_frag), 0 },
        { "aot_fixedfs", SPV(spv_t_fixedfs_vert), NULL, 0, 0 },
        { "aot_func", NULL, 0, SPV(spv_t_func_frag), 0 },
        { "aot_seascape", NULL, 0, SPV(spv_seascape_frag), 0 },
        { "aot_seascape_fast", NULL, 0, SPV(spv_seascape_frag), 1 },
    };
    FILE* f = fopen(argv[1], "wb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    for (size_t i = 0; i < sizeof(progs) / sizeof(progs[0]); i++) {
        const aot_prog* p = &progs[i];
        char            err[256];
        fm3d_spirv*     sp = fm3d_spirv_create(p->vs, p->vs_words, p->fs, p->fs_words, p->vs ? attr : NULL, p->vs ? 2 : 0,
                                               err, sizeof(err));
        if (sp) fm3d_spirv_set_fast_math(sp, p->fast);
        char*           src = sp ? fm3d_spirv_to_c(sp, p->name, err, sizeof(err)) : NULL;
        if (!src) {
            fprintf(stderr, "%s: %s\n", p->name, err);
            fclose(f);
            remove(argv[1]);
            return 1;
        }
        fwrite(src, 1, strlen(src), f);
        fprintf(f, "\n");
        fm3d_spirv_free_c(src);
        fm3d_spirv_destroy(sp);
    }
    fclose(f);
    return 0;
}
