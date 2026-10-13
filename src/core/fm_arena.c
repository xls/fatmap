/* fatmap - bump allocator with block recycling */
#include "fm_arena.h"
#include "fm_internal.h"

#define FM_ARENA_BLOCK (64 * 1024)

/* ---- arena ------------------------------------------------------------------- */

#define FM_ARENA_HDR ((sizeof(fm_arena_block) + 15) & ~(size_t)15)

void* fm_arena_alloc(fm_arena* a, size_t size)
{
    size = (size + 15) & ~(size_t)15;
    if (!a->head || a->head->used + size > a->head->cap) {
        fm_arena_block* b = NULL;
        if (size <= FM_ARENA_BLOCK && a->spare) { /* standard blocks all fit */
            b        = a->spare;
            a->spare = b->next;
        } else if (size > FM_ARENA_BLOCK) { /* the first recycled big block that fits (a short list) */
            fm_arena_block** pp = &a->big;
            while (*pp && (*pp)->cap < size) pp = &(*pp)->next;
            if ((b = *pp) != NULL) *pp = b->next;
        }
        if (!b) {
            size_t cap = size > FM_ARENA_BLOCK ? size : FM_ARENA_BLOCK;
            b          = (fm_arena_block*)malloc(FM_ARENA_HDR + cap);
            if (!b) return NULL;
            b->cap = cap;
        }
        b->used = 0;
        b->next = a->head;
        a->head = b;
    }
    void* p = (uint8_t*)a->head + FM_ARENA_HDR + a->head->used;
    a->head->used += size;
    return p;
}

/* move all blocks to the spare list (keeps memory for the next frame) */
void fm_arena_reset(fm_arena* a)
{
    while (a->head) {
        fm_arena_block*  b    = a->head;
        fm_arena_block** list = b->cap > FM_ARENA_BLOCK ? &a->big : &a->spare;
        a->head               = b->next;
        b->next               = *list;
        *list                 = b;
    }
}

void fm_arena_free(fm_arena* a)
{
    fm_arena_reset(a);
    while (a->spare) {
        fm_arena_block* b = a->spare;
        a->spare          = b->next;
        free(b);
    }
    while (a->big) {
        fm_arena_block* b = a->big;
        a->big            = b->next;
        free(b);
    }
}

