/* fatmap - built-in profiler */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L /* clock_gettime, pthreads, sysconf under -std=c11 */
#endif
#include "fm_internal.h"
#include "fm_atomic.h"
#include <stdio.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <time.h>
#endif

static fm_prof_zone* volatile g_zones;
static volatile long          g_zone_lock;
static int                    g_enabled = 1;
static uint64_t               g_frames;

uint64_t fm_time_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER        c;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    /* split to avoid overflow of c * 1e9 */
    uint64_t q = (uint64_t)c.QuadPart / (uint64_t)freq.QuadPart;
    uint64_t r = (uint64_t)c.QuadPart % (uint64_t)freq.QuadPart;
    return q * 1000000000ull + r * 1000000000ull / (uint64_t)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

void fm_prof_set_enabled(int on) { g_enabled = on; }
int  fm_prof_enabled(void) { return g_enabled; }
void fm_prof_frame(void) { g_frames++; }
uint64_t fm_prof_frames(void) { return g_frames; }

void fm_prof_reset(void)
{
    for (fm_prof_zone* z = g_zones; z; z = z->next) {
        z->calls = z->total_ns = z->max_ns = z->items = 0;
    }
    g_frames = 0;
}

uint64_t fm_prof__begin(fm_prof_zone* z)
{
    if (!g_enabled) return 0;
    if (!z->registered) {
        fm_spin_lock(&g_zone_lock);
        if (!z->registered) {
            z->next       = g_zones;
            g_zones       = z;
            z->registered = 1;
        }
        fm_spin_unlock(&g_zone_lock);
    }
    return fm_time_ns();
}

void fm_prof__end(fm_prof_zone* z, uint64_t t0)
{
    if (!t0) return;
    uint64_t dt = fm_time_ns() - t0;
    fm_atomic_add64(&z->calls, 1);
    fm_atomic_add64(&z->total_ns, dt);
    uint64_t m = fm_atomic_load64(&z->max_ns);
    while (dt > m && !fm_atomic_cas64(&z->max_ns, m, dt)) m = fm_atomic_load64(&z->max_ns);
}

void fm_prof__items(fm_prof_zone* z, uint64_t n) { fm_atomic_add64(&z->items, n); }

int fm_prof_zones(const fm_prof_zone** out, int max)
{
    int n = 0;
    for (fm_prof_zone* z = g_zones; z; z = z->next) {
        if (n < max) out[n] = z;
        n++;
    }
    return n;
}

int fm_prof_report(char* buf, int size)
{
    const fm_prof_zone* zs[128];
    int                 n = fm_prof_zones(zs, 128);
    if (n > 128) n = 128;
    /* sort by total time, descending */
    for (int i = 1; i < n; i++) {
        const fm_prof_zone* z = zs[i];
        int                 j = i - 1;
        while (j >= 0 && zs[j]->total_ns < z->total_ns) {
            zs[j + 1] = zs[j];
            j--;
        }
        zs[j + 1] = z;
    }
    int      w      = 0;
    uint64_t frames = g_frames ? g_frames : 1;
#define FM_APPEND(...)                                                 \
    do {                                                               \
        if (w < size) {                                                \
            int k_ = snprintf(buf + w, (size_t)(size - w), __VA_ARGS__); \
            if (k_ > 0) w += k_;                                       \
        }                                                              \
    } while (0)
    FM_APPEND("%-24s %10s %10s %10s %10s %12s\n", "zone", "calls", "total ms", "ms/frame", "max us", "items/frame");
    for (int i = 0; i < n; i++) {
        const fm_prof_zone* z = zs[i];
        if (!z->calls) continue;
        FM_APPEND("%-24s %10llu %10.3f %10.3f %10.1f %12.0f\n", z->name, (unsigned long long)z->calls,
                  (double)z->total_ns / 1e6, (double)z->total_ns / 1e6 / (double)frames, (double)z->max_ns / 1e3,
                  (double)z->items / (double)frames);
    }
#undef FM_APPEND
    if (w >= size) w = size - 1;
    return w;
}
