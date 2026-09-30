/* sandbox helper: Wavefront OBJ + MTL loader producing fm3d_vertex meshes */
#ifndef FATMAP_SANDBOX_OBJ_LOADER_H
#define FATMAP_SANDBOX_OBJ_LOADER_H

#include <fatmap/fatmap.h>

typedef struct obj_part {
    char         group[64];
    char         material[64];
    char         map_kd[256]; /* diffuse texture file name (no path) or "" */
    float        kd[3];
    fm3d_vertex* v;
    int          nv;
    uint32_t*    idx;
    int          ni;
} obj_part;

typedef struct obj_model {
    obj_part* parts;
    int       nparts;
    float     bmin[3], bmax[3];
} obj_model;

/* Loads path (and its mtllib). Faces are triangulated, vertices deduplicated
 * per part (one part per group + material), v flipped to top-down. */
int  obj_load(const char* path, obj_model* out);
void obj_free(obj_model* m);

#endif
