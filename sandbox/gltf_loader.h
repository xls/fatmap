/* sandbox helper: minimal glTF 2.0 binary (.glb) loader for skinned, animated meshes */
#ifndef FATMAP_SANDBOX_GLTF_LOADER_H
#define FATMAP_SANDBOX_GLTF_LOADER_H

#include <fatmap/fatmap.h>

typedef struct gltf_node {
    float t[3], r[4], s[3]; /* rest pose TRS (r = quaternion x, y, z, w) */
    float matrix[16];
    int   has_matrix;
    int   parent;
} gltf_node;

typedef struct gltf_channel {
    int          node;
    int          path; /* 0 translation, 1 rotation, 2 scale */
    int          step; /* STEP interpolation (else LINEAR) */
    int          nkeys;
    const float* times;
    const float* values; /* nkeys * (path == 1 ? 4 : 3) */
} gltf_channel;

typedef struct gltf_anim {
    char          name[64];
    gltf_channel* ch;
    int           nch;
    float         duration;
} gltf_anim;

typedef struct gltf_model {
    /* the first mesh's triangle primitives, merged */
    fm3d_vertex*      v;
    fm3d_skin_vertex* skin; /* NULL if not skinned */
    int               nv;
    uint32_t*         idx;
    int               ni;
    int               image; /* index of the base color image, -1 if none */
    gltf_node*        nodes;
    int               nnodes;
    int*              joints; /* skin joint -> node */
    float*            inv_bind; /* 16 * njoints */
    int               njoints;
    gltf_anim*        anims;
    int               nanims;
    float*            anim_data; /* owned key data */
} gltf_model;

int  gltf_load(const char* path, gltf_model* m);
void gltf_free(gltf_model* m);
/* skin matrices for animation `anim` at time t (seconds, wraps); bones must
 * hold m->njoints entries. anim < 0 = rest pose. */
void gltf_pose(const gltf_model* m, int anim, float t, fm_mat4* bones);

#endif
