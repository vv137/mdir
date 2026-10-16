/* A sampling profiler for a process that runs JIT code, loaded with
   LD_PRELOAD. Two kinds of samples:
     - wall: the main thread, every SAMPLER_WALL_US microseconds of real time
       (a timer of CLOCK_MONOTONIC aimed at that thread);
     - cpu: any thread, every SAMPLER_CPU_US microseconds of the CPU time of
       the process (ITIMER_PROF).
   Each sample keeps the thread, the interrupted PC, and a short stack from
   backtrace(). The code that the JIT made is kept by copying the objects of
   the GDB JIT interface (__jit_debug_descriptor), whose sections carry the
   addresses where they were loaded. Output in SAMPLER_OUT (a directory):
   samples.bin, maps.txt, jit-<n>.o. */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define DEPTH 10
typedef struct {
  uint32_t kind; /* 1 wall, 2 cpu */
  uint32_t tid;
  uint64_t time_ns;
  uint32_t depth;
  uint32_t pad;
  uint64_t pc[DEPTH];
} Sample;

static Sample *samples;
static size_t capacity;
static atomic_size_t count;
static atomic_int active;
static pid_t mainTid;
static char outDir[4096];
static timer_t wallTimer;
static pthread_t poller;
static atomic_int stopPolling;

struct jit_code_entry {
  struct jit_code_entry *next_entry;
  struct jit_code_entry *prev_entry;
  const char *symfile_addr;
  uint64_t symfile_size;
};
struct jit_descriptor {
  uint32_t version;
  uint32_t action_flag;
  struct jit_code_entry *relevant_entry;
  struct jit_code_entry *first_entry;
};

static uint64_t now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + t.tv_nsec;
}

static void record(int kind, void *context) {
  if (!atomic_load(&active))
    return;
  size_t i = atomic_fetch_add(&count, 1);
  if (i >= capacity)
    return;
  Sample *s = &samples[i];
  s->kind = kind;
  s->tid = (uint32_t)syscall(SYS_gettid);
  s->time_ns = now();
  ucontext_t *uc = (ucontext_t *)context;
  s->pc[0] = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
  void *frames[DEPTH + 3];
  int n = getenv("SAMPLER_NOSTACK") ? 0 : backtrace(frames, DEPTH + 3);
  /* frames[0] is this function, [1] the handler, [2] the signal trampoline,
     [3] the interrupted function (its PC again). */
  int d = 1;
  for (int k = 4; k < n && d < DEPTH; ++k)
    s->pc[d++] = (uint64_t)frames[k];
  s->depth = d;
}

static void onWall(int sig, siginfo_t *info, void *context) {
  (void)sig; (void)info;
  int saved = errno;
  record(1, context);
  errno = saved;
}
static void onCpu(int sig, siginfo_t *info, void *context) {
  (void)sig; (void)info;
  int saved = errno;
  record(2, context);
  errno = saved;
}

static void copyFile(const char *from, const char *to) {
  FILE *in = fopen(from, "r"), *out = fopen(to, "w");
  if (in && out) {
    char buffer[65536];
    size_t n;
    while ((n = fread(buffer, 1, sizeof buffer, in)) > 0)
      fwrite(buffer, 1, n, out);
  }
  if (in) fclose(in);
  if (out) fclose(out);
}

#define MAX_SEEN 4096
static const void *seen[MAX_SEEN];
static int numSeen;

static void pollJit(void) {
  struct jit_descriptor *d = dlsym(RTLD_DEFAULT, "__jit_debug_descriptor");
  if (!d)
    return;
  int added = 0;
  for (struct jit_code_entry *e = d->first_entry; e; e = e->next_entry) {
    int known = 0;
    for (int i = 0; i < numSeen; ++i)
      if (seen[i] == e->symfile_addr)
        known = 1;
    if (known || numSeen == MAX_SEEN)
      continue;
    char path[4200];
    snprintf(path, sizeof path, "%s/jit-%d.o", outDir, numSeen);
    FILE *f = fopen(path, "w");
    if (f) {
      fwrite(e->symfile_addr, 1, e->symfile_size, f);
      fclose(f);
    }
    seen[numSeen++] = e->symfile_addr;
    added = 1;
  }
  if (added) {
    char path[4200];
    snprintf(path, sizeof path, "%s/maps.txt", outDir);
    copyFile("/proc/self/maps", path);
  }
}

static void *pollLoop(void *unused) {
  (void)unused;
  sigset_t all;
  sigfillset(&all);
  pthread_sigmask(SIG_BLOCK, &all, NULL);
  while (!atomic_load(&stopPolling)) {
    pollJit();
    usleep(200000);
  }
  return NULL;
}

static void finish(void) {
  atomic_store(&active, 0);
  struct itimerval off = {{0, 0}, {0, 0}};
  setitimer(ITIMER_PROF, &off, NULL);
  timer_delete(wallTimer);
  atomic_store(&stopPolling, 1);
  pollJit();
  char path[4200];
  snprintf(path, sizeof path, "%s/maps-exit.txt", outDir);
  copyFile("/proc/self/maps", path);
  snprintf(path, sizeof path, "%s/samples.bin", outDir);
  FILE *f = fopen(path, "w");
  size_t n = atomic_load(&count);
  if (n > capacity)
    n = capacity;
  if (f) {
    fwrite(samples, sizeof(Sample), n, f);
    fclose(f);
  }
  fprintf(stderr, "sampler: %zu samples in %s\n", n, outDir);
}

__attribute__((constructor)) static void start(void) {
  const char *dir = getenv("SAMPLER_OUT");
  if (!dir)
    return;
  /* Only the first process (not the children it may start). */
  unsetenv("LD_PRELOAD");
  snprintf(outDir, sizeof outDir, "%s", dir);
  capacity = 4u << 20;
  samples = calloc(capacity, sizeof(Sample));
  mainTid = (pid_t)syscall(SYS_gettid);
  void *warm[4];
  backtrace(warm, 4); /* loads the unwinder before any signal */

  struct sigaction a;
  memset(&a, 0, sizeof a);
  a.sa_flags = SA_SIGINFO | SA_RESTART;
  /* Each handler blocks the other signal: backtrace() takes the lock of
     dl_iterate_phdr, and a wall sample that interrupted a CPU sample's
     backtrace on the same thread waited for that lock forever. */
  sigemptyset(&a.sa_mask);
  sigaddset(&a.sa_mask, SIGPROF);
  sigaddset(&a.sa_mask, SIGRTMIN + 4);
  a.sa_sigaction = onWall;
  sigaction(SIGRTMIN + 4, &a, NULL);
  a.sa_sigaction = onCpu;
  sigaction(SIGPROF, &a, NULL);

  long wallUs = getenv("SAMPLER_WALL_US") ? atol(getenv("SAMPLER_WALL_US")) : 500;
  long cpuUs = getenv("SAMPLER_CPU_US") ? atol(getenv("SAMPLER_CPU_US")) : 4000;
  struct sigevent ev;
  memset(&ev, 0, sizeof ev);
  ev.sigev_notify = SIGEV_THREAD_ID;
  ev.sigev_signo = SIGRTMIN + 4;
  ev._sigev_un._tid = mainTid;
  if (timer_create(CLOCK_MONOTONIC, &ev, &wallTimer) == 0) {
    struct itimerspec spec;
    spec.it_interval.tv_sec = wallUs / 1000000;
    spec.it_interval.tv_nsec = (wallUs % 1000000) * 1000;
    spec.it_value = spec.it_interval;
    timer_settime(wallTimer, 0, &spec, NULL);
  }
  struct itimerval prof;
  prof.it_interval.tv_sec = cpuUs / 1000000;
  prof.it_interval.tv_usec = cpuUs % 1000000;
  prof.it_value = prof.it_interval;
  setitimer(ITIMER_PROF, &prof, NULL);
  atomic_store(&active, 1);
  pthread_create(&poller, NULL, pollLoop, NULL);
  atexit(finish);
}
