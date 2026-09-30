/*
 * fatmap - built-in thread pool executor (Win32 / pthreads / none).
 *
 * parallel_for publishes a job, the caller works on it as worker 0 and the
 * pool threads pull indices from an atomic counter. A new job is only
 * published after every pool thread has left the previous one, so a late
 * thread can never run an old callback with a new index.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L /* clock_gettime, pthreads, sysconf under -std=c11 */
#endif
#include "fm_internal.h"
#include "fm_atomic.h"
#include <fatmap/fm_exec.h>

#if !defined(FM_NO_THREADS)
#  if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    define FM_THREADS_WIN32 1
#  else
#    include <pthread.h>
#    include <unistd.h>
#    define FM_THREADS_PTHREAD 1
#  endif
#endif

int fm_threads_supported(void)
{
#if defined(FM_NO_THREADS)
    return 0;
#else
    return 1;
#endif
}

int fm_cpu_count(void)
{
#if defined(FM_THREADS_WIN32)
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n > 0 ? (int)n : 1;
#elif defined(FM_THREADS_PTHREAD)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#else
    return 1;
#endif
}

#if defined(FM_NO_THREADS)

static void fm_serial_for(fm_executor* ex, fm_task_fn fn, void* arg, int count)
{
    (void)ex;
    for (int i = 0; i < count; i++) fn(arg, i, 0);
}

fm_executor* fm_executor_create(int threads)
{
    (void)threads;
    fm_executor* ex = (fm_executor*)calloc(1, sizeof(fm_executor));
    if (!ex) return NULL;
    ex->parallel_for = fm_serial_for;
    ex->workers      = 1;
    return ex;
}

void fm_executor_destroy(fm_executor* ex) { free(ex); }

#else

#if defined(FM_THREADS_WIN32)
typedef SRWLOCK            fm_mutex;
typedef CONDITION_VARIABLE fm_cond;
typedef HANDLE             fm_thread;
#  define fm_mutex_init(m)     InitializeSRWLock(m)
#  define fm_mutex_destroy(m)  ((void)0)
#  define fm_mutex_lock(m)     AcquireSRWLockExclusive(m)
#  define fm_mutex_unlock(m)   ReleaseSRWLockExclusive(m)
#  define fm_cond_init(c)      InitializeConditionVariable(c)
#  define fm_cond_destroy(c)   ((void)0)
#  define fm_cond_wait(c, m)   SleepConditionVariableSRW(c, m, INFINITE, 0)
#  define fm_cond_broadcast(c) WakeAllConditionVariable(c)
#else
typedef pthread_mutex_t fm_mutex;
typedef pthread_cond_t  fm_cond;
typedef pthread_t       fm_thread;
#  define fm_mutex_init(m)     pthread_mutex_init(m, NULL)
#  define fm_mutex_destroy(m)  pthread_mutex_destroy(m)
#  define fm_mutex_lock(m)     pthread_mutex_lock(m)
#  define fm_mutex_unlock(m)   pthread_mutex_unlock(m)
#  define fm_cond_init(c)      pthread_cond_init(c, NULL)
#  define fm_cond_destroy(c)   pthread_cond_destroy(c)
#  define fm_cond_wait(c, m)   pthread_cond_wait(c, m)
#  define fm_cond_broadcast(c) pthread_cond_broadcast(c)
#endif

typedef struct fm_pool fm_pool;

/* pause iterations to spin before blocking (~tens of microseconds) */
#define FM_POOL_SPIN 20000

typedef struct fm_pool_thread {
    fm_pool*  pool;
    int       id;
    fm_thread handle;
} fm_pool_thread;

struct fm_pool {
    fm_executor     pub; /* must be first */
    int             nthreads;
    fm_pool_thread* threads;
    fm_mutex        lock;
    fm_cond         work_cv;
    fm_cond         done_cv;
    /* current job */
    fm_task_fn    fn;
    void*         arg;
    int           count;
    volatile long next;
    volatile long remaining;
    unsigned      gen;
    int           active; /* pool threads inside the current job (under lock) */
    int           quit;
    volatile long busy; /* serializes parallel_for callers */
};

static void fm_pool_work(fm_pool* p, fm_task_fn fn, void* arg, int count, int id)
{
    for (;;) {
        long i = fm_atomic_inc(&p->next) - 1;
        if (i >= count) break;
        fn(arg, (int)i, id);
        if (fm_atomic_dec(&p->remaining) == 0) {
            fm_mutex_lock(&p->lock);
            fm_cond_broadcast(&p->done_cv);
            fm_mutex_unlock(&p->lock);
        }
    }
}

#if defined(FM_THREADS_WIN32)
static DWORD WINAPI fm_pool_main(LPVOID param)
#else
static void* fm_pool_main(void* param)
#endif
{
    fm_pool_thread* t    = (fm_pool_thread*)param;
    fm_pool*        p    = t->pool;
    unsigned        seen = 0;
    for (;;) {
        /* spin a little before sleeping: frames issue jobs back to back */
        for (int s = 0; s < FM_POOL_SPIN && *(volatile unsigned*)&p->gen == seen && !*(volatile int*)&p->quit; s++)
            fm_cpu_relax();
        fm_mutex_lock(&p->lock);
        while (p->gen == seen && !p->quit) fm_cond_wait(&p->work_cv, &p->lock);
        if (p->quit) {
            fm_mutex_unlock(&p->lock);
            break;
        }
        seen          = p->gen;
        fm_task_fn fn = p->fn;
        void*      arg = p->arg;
        int        cnt = p->count;
        p->active++;
        fm_mutex_unlock(&p->lock);

        fm_pool_work(p, fn, arg, cnt, t->id);

        fm_mutex_lock(&p->lock);
        if (--p->active == 0) fm_cond_broadcast(&p->done_cv);
        fm_mutex_unlock(&p->lock);
    }
#if defined(FM_THREADS_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static void fm_pool_for(fm_executor* ex, fm_task_fn fn, void* arg, int count)
{
    fm_pool* p = (fm_pool*)ex;
    if (count <= 0) return;
    if (count == 1 || p->nthreads == 0) {
        for (int i = 0; i < count; i++) fn(arg, i, 0);
        return;
    }
    fm_spin_lock(&p->busy);
    fm_mutex_lock(&p->lock);
    while (p->active > 0) fm_cond_wait(&p->done_cv, &p->lock);
    p->fn    = fn;
    p->arg   = arg;
    p->count = count;
    fm_atomic_store(&p->next, 0);
    fm_atomic_store(&p->remaining, count);
    p->gen++;
    fm_cond_broadcast(&p->work_cv);
    fm_mutex_unlock(&p->lock);

    fm_pool_work(p, fn, arg, count, 0);

    for (int s = 0; s < FM_POOL_SPIN && fm_atomic_load(&p->remaining) > 0; s++) fm_cpu_relax();
    fm_mutex_lock(&p->lock);
    while (fm_atomic_load(&p->remaining) > 0) fm_cond_wait(&p->done_cv, &p->lock);
    fm_mutex_unlock(&p->lock);
    fm_spin_unlock(&p->busy);
}

fm_executor* fm_executor_create(int threads)
{
    fm__init();
    if (threads <= 0) threads = fm_cpu_count();
    if (threads > 256) threads = 256;
    fm_pool* p = (fm_pool*)calloc(1, sizeof(fm_pool));
    if (!p) return NULL;
    p->pub.parallel_for = fm_pool_for;
    p->pub.workers      = threads;
    p->pub.user         = p;
    fm_mutex_init(&p->lock);
    fm_cond_init(&p->work_cv);
    fm_cond_init(&p->done_cv);
    p->nthreads = threads - 1;
    if (p->nthreads > 0) {
        p->threads = (fm_pool_thread*)calloc((size_t)p->nthreads, sizeof(fm_pool_thread));
        if (!p->threads) p->nthreads = 0;
    }
    int started = 0;
    for (int i = 0; i < p->nthreads; i++) {
        fm_pool_thread* t = &p->threads[i];
        t->pool           = p;
        t->id             = i + 1;
#if defined(FM_THREADS_WIN32)
        t->handle = CreateThread(NULL, 0, fm_pool_main, t, 0, NULL);
        if (!t->handle) break;
#else
        if (pthread_create(&t->handle, NULL, fm_pool_main, t) != 0) break;
#endif
        started++;
    }
    p->nthreads    = started;
    p->pub.workers = started + 1;
    return &p->pub;
}

void fm_executor_destroy(fm_executor* ex)
{
    if (!ex) return;
    fm_pool* p = (fm_pool*)ex;
    fm_mutex_lock(&p->lock);
    p->quit = 1;
    fm_cond_broadcast(&p->work_cv);
    fm_mutex_unlock(&p->lock);
    for (int i = 0; i < p->nthreads; i++) {
#if defined(FM_THREADS_WIN32)
        WaitForSingleObject(p->threads[i].handle, INFINITE);
        CloseHandle(p->threads[i].handle);
#else
        pthread_join(p->threads[i].handle, NULL);
#endif
    }
    fm_cond_destroy(&p->work_cv);
    fm_cond_destroy(&p->done_cv);
    fm_mutex_destroy(&p->lock);
    free(p->threads);
    free(p);
}

#endif /* FM_NO_THREADS */
