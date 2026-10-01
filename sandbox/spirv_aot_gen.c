/* Build step: the sandbox's SPIR-V programs (sandbox/spirv_shaders.h)
 * translated to C with fm3d_spirv_to_c, written to argv[1]. The shaders
 * scene (G) switches between them and the interpreter. */
#include <fatmap/fatmap.h>
#include <stdio.h>
#include <string.h>
#include "spirv_shaders.h"

#define SPV(a) a, sizeof(a) / 4

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: sandbox_aot_gen out.c\n");
        return 2;
    }
    static const fm3d_vertex_attrib ha[3] = { { 0, 3, 0 }, { 1, 3, 12 }, { 2, 2, 24 } }; /* fm3d_vertex */
    static const fm3d_vertex_attrib fa[1] = { { 0, 2, 0 } };                              /* flag_vtx */
    struct {
        const char*               name;
        const uint32_t*           vs;
        size_t                    nvs;
        const uint32_t*           fs;
        size_t                    nfs;
        const fm3d_vertex_attrib* attr;
        int                       nattr;
    } progs[] = {
        { "aot_helmet", SPV(spv_helmet_vert), SPV(spv_helmet_frag), ha, 3 },
        { "aot_flag", SPV(spv_flag_vert), SPV(spv_flag_frag), fa, 1 },
        { "aot_seascape", NULL, 0, SPV(spv_seascape_frag), NULL, 0 },
    };
    FILE* f = fopen(argv[1], "wb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    for (size_t i = 0; i < sizeof(progs) / sizeof(progs[0]); i++) {
        char        err[256];
        fm3d_spirv* sp  = fm3d_spirv_create(progs[i].vs, progs[i].nvs, progs[i].fs, progs[i].nfs, progs[i].attr, progs[i].nattr,
                                            err, sizeof(err));
        char*       src = sp ? fm3d_spirv_to_c(sp, progs[i].name, err, sizeof(err)) : NULL;
        if (!src) {
            fprintf(stderr, "%s: %s\n", progs[i].name, err);
            fclose(f);
            remove(argv[1]);
            return 1;
        }
        fwrite(src, 1, strlen(src), f);
        fputs("\n", f);
        fm3d_spirv_free_c(src);
        fm3d_spirv_destroy(sp);
    }
    fclose(f);
    return 0;
}
