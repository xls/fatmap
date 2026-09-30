/* fatmap - bump allocator with block recycling (internal) */
#ifndef FATMAP_FM_ARENA_H
#define FATMAP_FM_ARENA_H

#include <stddef.h>

typedef struct fm_arena_block {
    struct fm_arena_block* next;
    size_t                 used, cap;
} fm_arena_block;

typedef struct fm_arena {
    fm_arena_block* head;  /* current block */
    fm_arena_block* spare; /* recycled blocks */
} fm_arena;

void* fm_arena_alloc(fm_arena* a, size_t size); /* 16 byte aligned, NULL on OOM */
void  fm_arena_reset(fm_arena* a);              /* keep memory for reuse */
void  fm_arena_free(fm_arena* a);

#endif
