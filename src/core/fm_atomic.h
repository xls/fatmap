/* fatmap - minimal portable atomics (internal). With FM_NO_THREADS these
 * compile to plain operations. */
#ifndef FATMAP_FM_ATOMIC_H
#define FATMAP_FM_ATOMIC_H

#include <stdint.h>

#if defined(FM_NO_THREADS)

static inline long     fm_atomic_inc(volatile long* p) { return ++*p; }
static inline long     fm_atomic_dec(volatile long* p) { return --*p; }
static inline long     fm_atomic_load(volatile long* p) { return *p; }
static inline void     fm_atomic_store(volatile long* p, long v) { *p = v; }
static inline int      fm_atomic_cas(volatile long* p, long expect, long desired)
{
    if (*p != expect) return 0;
    *p = desired;
    return 1;
}
static inline void     fm_atomic_add64(volatile uint64_t* p, uint64_t v) { *p += v; }
static inline uint64_t fm_atomic_load64(volatile uint64_t* p) { return *p; }
static inline int      fm_atomic_cas64(volatile uint64_t* p, uint64_t expect, uint64_t desired)
{
    if (*p != expect) return 0;
    *p = desired;
    return 1;
}
static inline void fm_cpu_relax(void) {}

#elif defined(_MSC_VER)

#include <intrin.h>
static inline long fm_atomic_inc(volatile long* p) { return _InterlockedIncrement(p); }
static inline long fm_atomic_dec(volatile long* p) { return _InterlockedDecrement(p); }
static inline long fm_atomic_load(volatile long* p) { return _InterlockedCompareExchange(p, 0, 0); }
static inline void fm_atomic_store(volatile long* p, long v) { _InterlockedExchange(p, v); }
static inline int  fm_atomic_cas(volatile long* p, long expect, long desired)
{
    return _InterlockedCompareExchange(p, desired, expect) == expect;
}
static inline void fm_atomic_add64(volatile uint64_t* p, uint64_t v)
{
    _InterlockedExchangeAdd64((volatile __int64*)p, (__int64)v);
}
static inline uint64_t fm_atomic_load64(volatile uint64_t* p)
{
    return (uint64_t)_InterlockedCompareExchange64((volatile __int64*)p, 0, 0);
}
static inline int fm_atomic_cas64(volatile uint64_t* p, uint64_t expect, uint64_t desired)
{
    return (uint64_t)_InterlockedCompareExchange64((volatile __int64*)p, (__int64)desired, (__int64)expect) == expect;
}
#if defined(_M_X64) || defined(_M_IX86)
static inline void fm_cpu_relax(void) { _mm_pause(); }
#else
static inline void fm_cpu_relax(void) { __yield(); }
#endif

#else /* GCC / Clang */

static inline long fm_atomic_inc(volatile long* p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
static inline long fm_atomic_dec(volatile long* p) { return __atomic_sub_fetch(p, 1, __ATOMIC_SEQ_CST); }
static inline long fm_atomic_load(volatile long* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void fm_atomic_store(volatile long* p, long v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static inline int  fm_atomic_cas(volatile long* p, long expect, long desired)
{
    return __atomic_compare_exchange_n(p, &expect, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
static inline void     fm_atomic_add64(volatile uint64_t* p, uint64_t v) { __atomic_add_fetch(p, v, __ATOMIC_RELAXED); }
static inline uint64_t fm_atomic_load64(volatile uint64_t* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
static inline int      fm_atomic_cas64(volatile uint64_t* p, uint64_t expect, uint64_t desired)
{
    return __atomic_compare_exchange_n(p, &expect, desired, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
#if defined(__x86_64__) || defined(__i386__)
static inline void fm_cpu_relax(void) { __builtin_ia32_pause(); }
#elif defined(__aarch64__)
static inline void fm_cpu_relax(void) { __asm__ volatile("yield"); }
#else
static inline void fm_cpu_relax(void) {}
#endif

#endif

/* tiny spinlock for rare paths (registration, init) */
static inline void fm_spin_lock(volatile long* l)
{
    while (!fm_atomic_cas(l, 0, 1)) fm_cpu_relax();
}
static inline void fm_spin_unlock(volatile long* l) { fm_atomic_store(l, 0); }

#endif
