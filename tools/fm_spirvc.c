/*
 * fm-spirvc - compile SPIR-V shaders to C ahead of time (fm3d_spirv_to_c).
 *
 *   fm-spirvc [-o out.c] [-n name] [--vs a.spv] [--fs b.spv] [-a loc:floats:offset ...]
 *
 * Writes C source defining `fm3d_program <name>_program(void)` (default
 * name: shader). Either stage may be left out (the fixed function stage).
 * -a describes a vertex attribute: shader input location, float count and
 * byte offset in the vertex (as fm3d_vertex_attrib). Compile the output
 * with -O3 -ffp-contract=off (MSVC /O2 /fp:precise) and link fatmap.
 */
#include <fatmap/fatmap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t* read_spv(const char* path, size_t* words)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* w = n > 0 && n % 4 == 0 ? (uint32_t*)malloc((size_t)n) : NULL;
    if (!w || fread(w, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "%s: not a SPIR-V binary\n", path);
        free(w);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *words = (size_t)n / 4;
    return w;
}

static int usage(void)
{
    fprintf(stderr, "usage: fm-spirvc [-o out.c] [-n name] [--vs a.spv] [--fs b.spv] [-a loc:floats:offset ...]\n");
    return 2;
}

int main(int argc, char** argv)
{
    const char*        out = NULL, *name = "shader", *vs_path = NULL, *fs_path = NULL;
    fm3d_vertex_attrib attr[16];
    int                nattr = 0;
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (i + 1 >= argc) return usage();
        if (!strcmp(a, "-o")) out = argv[++i];
        else if (!strcmp(a, "-n")) name = argv[++i];
        else if (!strcmp(a, "--vs")) vs_path = argv[++i];
        else if (!strcmp(a, "--fs")) fs_path = argv[++i];
        else if (!strcmp(a, "-a") && nattr < 16) {
            fm3d_vertex_attrib* v = &attr[nattr++];
            if (sscanf(argv[++i], "%d:%d:%d", &v->location, &v->components, &v->offset) != 3) return usage();
        } else
            return usage();
    }
    if (!vs_path && !fs_path) return usage();
    size_t    nvs = 0, nfs = 0;
    uint32_t* vs = vs_path ? read_spv(vs_path, &nvs) : NULL;
    uint32_t* fs = fs_path ? read_spv(fs_path, &nfs) : NULL;
    if ((vs_path && !vs) || (fs_path && !fs)) return 1;
    char        err[512];
    fm3d_spirv* p   = fm3d_spirv_create(vs, nvs, fs, nfs, attr, nattr, err, sizeof(err));
    char*       src = p ? fm3d_spirv_to_c(p, name, err, sizeof(err)) : NULL;
    free(vs);
    free(fs);
    if (!src) {
        fprintf(stderr, "fm-spirvc: %s\n", err);
        fm3d_spirv_destroy(p);
        return 1;
    }
    FILE* f = out ? fopen(out, "wb") : stdout;
    if (!f) {
        perror(out);
        return 1;
    }
    fputs(src, f);
    if (out) fclose(f);
    fm3d_spirv_free_c(src);
    fm3d_spirv_destroy(p);
    return 0;
}
