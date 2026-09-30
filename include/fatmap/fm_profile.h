/*
 * fatmap - built-in profiler.
 *
 * Zones are static objects that register themselves on first use. Each zone
 * accumulates call count, total/max time and an "items" counter (usually
 * pixels or spans). Instrument per draw call, never per pixel.
 *
 *   FM_PROF_BEGIN(z, "fill");
 *   ...
 *   FM_PROF_ITEMS(z, pixel_count);
 *   FM_PROF_END(z);
 *
 * Compile with FM_PROFILE=0 to remove all instrumentation. When built with
 * FM_TRACY=1 zones are also forwarded to the Tracy profiler (external/tracy).
 * Zone counters are updated atomically, zones may be used from any thread.
 */
#ifndef FATMAP_FM_PROFILE_H
#define FATMAP_FM_PROFILE_H

#include "fm_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FM_PROFILE
#define FM_PROFILE 1
#endif

typedef struct fm_prof_zone {
    const char*          name;
    volatile uint64_t    calls;
    volatile uint64_t    total_ns;
    volatile uint64_t    max_ns;
    volatile uint64_t    items;
    struct fm_prof_zone* next;
    int                  registered;
} fm_prof_zone;

FM_API uint64_t fm_time_ns(void); /* monotonic, high resolution */

FM_API void fm_prof_set_enabled(int on);
FM_API int  fm_prof_enabled(void);
FM_API void fm_prof_reset(void);
FM_API void fm_prof_frame(void);  /* marks a frame boundary */
FM_API uint64_t fm_prof_frames(void);
/* Fills out[] with registered zones, returns total zone count. */
FM_API int  fm_prof_zones(const fm_prof_zone** out, int max);
/* Human readable table sorted by total time. Returns bytes written. */
FM_API int  fm_prof_report(char* buf, int size);

FM_API uint64_t fm_prof__begin(fm_prof_zone* z);
FM_API void     fm_prof__end(fm_prof_zone* z, uint64_t t0);
FM_API void     fm_prof__items(fm_prof_zone* z, uint64_t n);

#if FM_PROFILE
#  if defined(FM_TRACY) && FM_TRACY
#    include <tracy/TracyC.h>
#    define FM_PROF__TRACY_BEGIN(id, name) TracyCZoneN(id##_tc, name, 1)
#    define FM_PROF__TRACY_END(id)         TracyCZoneEnd(id##_tc)
#  else
#    define FM_PROF__TRACY_BEGIN(id, name)
#    define FM_PROF__TRACY_END(id)
#  endif
#  define FM_PROF_BEGIN(id, name)                             \
      static fm_prof_zone id = { name, 0, 0, 0, 0, NULL, 0 }; \
      FM_PROF__TRACY_BEGIN(id, name);                         \
      uint64_t id##_t0 = fm_prof__begin(&id)
#  define FM_PROF_END(id)        \
      do {                       \
          fm_prof__end(&id, id##_t0); \
          FM_PROF__TRACY_END(id);     \
      } while (0)
#  define FM_PROF_ITEMS(id, n) fm_prof__items(&(id), (uint64_t)(n))
#else
#  define FM_PROF_BEGIN(id, name) ((void)0)
#  define FM_PROF_END(id)         ((void)0)
#  define FM_PROF_ITEMS(id, n)    ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* FATMAP_FM_PROFILE_H */
