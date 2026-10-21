/* The MDIR runtime for NVIDIA GPUs.

   These are the functions that the lowering of the upstream gpu dialect
   calls. Their names and signatures are fixed by that lowering. They are
   implemented on the CUDA driver API, so the library needs the driver and
   the header of the toolkit, and nothing else of the toolkit. */

#include <cuda.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The kernel that was launched last, and its name, which a failure names:
   a kernel that fails is reported by the driver at the next call that
   waits for it. */
static CUfunction lastFunction = NULL;
static const char *getKernelName(CUfunction function);

/* A failure of the driver is reported and ends the run: compiled code has no
   way to continue without its device. */
static void check(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = NULL;
  cuGetErrorName(result, &name);
  fprintf(stderr, "mdrt: %s failed with %s\n", what,
          name ? name : "an unknown error");
  if (lastFunction)
    fprintf(stderr,
            "mdrt: the last kernel launched was %s; the driver reports the "
            "failure of a kernel at a later call, and with MDRT_WAIT=1 each "
            "launch is waited for, so that the report names the kernel that "
            "failed. compute-sanitizer names the access; see "
            "docs/debugging.md\n",
            getKernelName(lastFunction));
  fflush(stderr);
  abort();
}

/*===----------------------------------------------------------------------===
  Profile
  ===----------------------------------------------------------------------===

  With MDRT_PROFILE set, the library counts its calls and the time they
  take, and reports when the program ends. With MDRT_WAIT set as well, it
  reports the time of each kernel: the time from the launch to the end of
  the kernel. With MDRT_TRACE set, it prints each module that it loads, the
  kernels of each module, and each launch with its grid, so that a kernel
  that a tool such as compute-sanitizer reports can be found in the module
  that the pass `gpu-kernel-outlining` prints (MDIR_PRINT_AFTER). */

enum { LAUNCH, WAIT, COPY, ALLOCATE, LOOKUP, NUM_COUNTERS };
static const char *const counterNames[NUM_COUNTERS] = {
    "launches", "waits", "copies", "allocations", "lookups of kernels"};
static int64_t counts[NUM_COUNTERS];
static double seconds[NUM_COUNTERS];
static int profile = -1;

static double now(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (double)time.tv_sec + 1.0e-9 * (double)time.tv_nsec;
}

/* The kernels that were launched. */
enum { MAX_KERNELS = 1024 };
struct Kernel {
  CUfunction function;
  char name[128];
  int64_t count;
  double seconds;
  /* The threads of its last launch, which tell what it runs over. */
  int64_t threads;
  /* Whether it was launched (the accounts of MDIR_DEVICE_MEMORY=2). */
  int launched;
};
static struct Kernel kernels[MAX_KERNELS];
static int numKernels = 0;

/* The kernel that was launched last, and when; -1 if its time is
   accounted for. */
static int lastKernel = -1;
static double lastLaunch = 0.0;

static void report(void) {
  fprintf(stderr, "mdrt: %-40s %10s %12s %12s %10s\n", "", "calls",
          "seconds", "microseconds", "threads");
  for (int i = 0; i != NUM_COUNTERS; ++i)
    fprintf(stderr, "mdrt: %-40s %10lld %12.4f %12.2f\n", counterNames[i],
            (long long)counts[i], seconds[i],
            counts[i] ? 1.0e6 * seconds[i] / (double)counts[i] : 0.0);
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].count)
      fprintf(stderr, "mdrt: %-40s %10lld %12.4f %12.2f %10lld\n",
              kernels[i].name, (long long)kernels[i].count,
              kernels[i].seconds,
              1.0e6 * kernels[i].seconds / (double)kernels[i].count,
              (long long)kernels[i].threads);
}

/* Notes that `function` has the name `name`. */
static void noteKernel(CUfunction function, const char *name) {
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].function == function)
      return;
  if (numKernels == MAX_KERNELS)
    return;
  kernels[numKernels].function = function;
  snprintf(kernels[numKernels].name, sizeof(kernels[numKernels].name), "%s",
           name);
  ++numKernels;
}

static int findKernel(CUfunction function) {
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].function == function)
      return i;
  return -1;
}

static const char *getKernelName(CUfunction function) {
  int i = findKernel(function);
  return i < 0 ? "a kernel without a name" : kernels[i].name;
}

/* The time at which a call begins, or a negative number if the profile is
   off. */
static double begin(void) {
  if (profile < 0) {
    profile = getenv("MDRT_PROFILE") != NULL;
    if (profile)
      atexit(report);
  }
  return profile ? now() : -1.0;
}

static void end(int counter, double start) {
  if (start < 0.0)
    return;
  counts[counter] += 1;
  seconds[counter] += now() - start;
}

static int32_t defaultDevice = 0;
static CUcontext context = NULL;

/* One stream serves all launches and copies, so the device runs them in the
   order in which they were issued. The lowering waits after every launch.
   The library does not: the host goes on issuing work while the device
   runs, and waits only where it reads what the device has computed. With
   MDRT_WAIT set, the library waits after every launch, which tells an
   error of the order apart from other errors and names the kernel that
   fails. */
static CUstream sharedStream = NULL;
static int waitsAlways = -1;

/* A second stream, on which work that is independent of what follows it on
   the first can run beside it (mdrtSideBegin). Work issued between
   mdrtSideBegin and mdrtSideEnd goes there, after what the first stream
   had been given; mdrtSideJoin makes the first stream wait for it. */
static CUstream sideStream = NULL;
static int onSide = 0;
static CUevent forkEvent = NULL, joinEvent = NULL;
static int joinPending = 0;

/* Whether the device may still be running work that was issued. */
static int isPending = 0;

/* With MDIR_DEVICE_MEMORY=2 (D311), the runtime also
   keeps accounts of the device memory that its pool of blocks does not
   hold: at each event that may take memory of the device outside the pool,
   how much the driver's use of the device grew (cuMemGetInfo, the total
   less the free, which counts every process on the device), and how many
   bytes the runtime asked for there. The events are the making of the
   context, the loads of modules, the first launch of each kernel (the
   driver reserves the local memory of the threads there), the plans of
   cuFFT and their workspaces, the handle of cuBLAS, and the allocations
   outside the pool. The blocks of the pool are an account too: the growth
   at the driver's allocations of blocks against their bytes. */
enum {
  ACCOUNT_CONTEXT,
  ACCOUNT_MODULES,
  ACCOUNT_LAUNCHES,
  ACCOUNT_BLOCKS,
  ACCOUNT_FFT,
  ACCOUNT_BLAS,
  ACCOUNT_TUPLES,
  ACCOUNT_CONSTANTS,
  ACCOUNT_MANAGED,
  ACCOUNT_KEPT,
  NUM_ACCOUNTS
};
static const char *const accountNames[NUM_ACCOUNTS] = {
    "the context",
    "the modules",
    "the first launch of each kernel",
    "the blocks of the pool",
    "the plans of cuFFT",
    "the handle of cuBLAS",
    "the structures of the tuples",
    "the constants of the bodies",
    "the memory shared with the host",
    "the memory of an embedding program"};
static int64_t accountGrowth[NUM_ACCOUNTS], accountAsked[NUM_ACCOUNTS];
static int64_t accountEvents[NUM_ACCOUNTS];
static int memoryDetail = -1;
static int isMemoryDetailed(void) {
  if (memoryDetail < 0) {
    const char *value = getenv("MDIR_DEVICE_MEMORY");
    memoryDetail = value && strcmp(value, "2") == 0;
  }
  return memoryDetail;
}
/* The driver's use of the device, in bytes, and the most it was at an
   event; -1 if it cannot be read. */
static int64_t usedMost = 0;
static int64_t getDeviceUsed(void) {
  size_t free = 0, total = 0;
  if (cuMemGetInfo(&free, &total) != CUDA_SUCCESS)
    return -1;
  int64_t used = (int64_t)(total - free);
  if (used > usedMost)
    usedMost = used;
  return used;
}
/* The driver's use before an event of the accounts, or -1 if they are not
   kept. */
static int64_t beginAccount(void) {
  return isMemoryDetailed() ? getDeviceUsed() : -1;
}
static void endAccount(int account, int64_t before, int64_t asked) {
  if (before < 0)
    return;
  int64_t after = getDeviceUsed();
  if (after >= 0)
    accountGrowth[account] += after - before;
  accountAsked[account] += asked;
  accountEvents[account] += 1;
}

/* Makes the context of the device current for the calling thread. The
   device is the one that MDRT_DEVICE names, among those that
   CUDA_VISIBLE_DEVICES leaves visible, or the first. */
static void enter(void) {
  if (!context) {
    check(cuInit(0), "cuInit");
    const char *chosen = getenv("MDRT_DEVICE");
    if (chosen)
      defaultDevice = (int32_t)strtol(chosen, NULL, 10);
    CUdevice device;
    check(cuDeviceGet(&device, defaultDevice), "cuDeviceGet");
    check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    /* The driver's use once the context is made: what it was before is
       the idle use of the device, which nvidia-smi reports. */
    if (isMemoryDetailed()) {
      accountGrowth[ACCOUNT_CONTEXT] = getDeviceUsed();
      accountEvents[ACCOUNT_CONTEXT] = 1;
    }
  }
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

/* What the products of the bodies of neural stages hold on the device
   (D302, at the end of this file): the copies of the constants
   that `owner` made (an activation, or null outside one; every owner with
   `all`), and the handle of cuBLAS. `gone` says that the driver has shut
   down, and the memory of the device with it. */
static void releaseDeviceConstants(void *owner, int all, int gone);
static void releaseBlas(int gone);

void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  if (context) {
    /* The handle and the copies belong to the context that is released. */
    int gone = cuCtxSetCurrent(context) != CUDA_SUCCESS;
    releaseDeviceConstants(NULL, 1, gone);
    releaseBlas(gone);
    CUdevice old;
    check(cuCtxGetDevice(&old), "cuCtxGetDevice");
    check(cuDevicePrimaryCtxRelease(old), "cuDevicePrimaryCtxRelease");
    context = NULL;
    sharedStream = NULL;
  }
}

/*===----------------------------------------------------------------------===
  Modules and kernels
  ===----------------------------------------------------------------------===*/

/* The kernels are PTX that the driver compiles at load, so the driver must
   know the PTX ISA version that LLVM wrote; the toolkit supplies only
   libdevice. A driver older than that version fails the load with
   CUDA_ERROR_UNSUPPORTED_PTX_VERSION, or a log that names `.version`. That
   failure ends the run with what to do, not a generic failure. */
static void reportUnsupportedPTX(CUresult result, const void *data,
                                 size_t size, const char *log) {
  if (result != CUDA_ERROR_UNSUPPORTED_PTX_VERSION &&
      !(log && strstr(log, ".version")))
    return;
  /* The `.version` line of the module, within its first kilobyte. */
  char isa[16] = "unknown";
  const char *text = (const char *)data;
  size_t length = size ? size : strnlen(text, 1 << 20);
  for (size_t i = 0; i + 9 < length && i < 1024; ++i)
    if (strncmp(text + i, ".version ", 9) == 0) {
      size_t n = 0;
      for (size_t j = i + 9; j < length && n + 1 < sizeof(isa) &&
                             (text[j] == '.' || (text[j] >= '0' && text[j] <= '9'));
           ++j)
        isa[n++] = text[j];
      isa[n] = '\0';
      break;
    }
  int driver = 0;
  cuDriverGetVersion(&driver);
  fprintf(stderr,
          "mdrt: the NVIDIA driver supports CUDA %d.%d, which cannot compile "
          "kernels of PTX ISA %s, the version that this build of MDIR emits. "
          "Update the NVIDIA driver to one that supports PTX ISA %s, or run "
          "on the CPU (target = \"CPU\")\n",
          driver / 1000, driver % 1000 / 10, isa, isa);
  fflush(stderr);
  exit(1);
}

CUmodule mgpuModuleLoad(void *data, size_t size) {
  enter();
  CUmodule module = NULL;
  int64_t before = beginAccount();
  CUresult result = cuModuleLoadData(&module, data);
  endAccount(ACCOUNT_MODULES, before, 0);
  if (result != CUDA_SUCCESS)
    reportUnsupportedPTX(result, data, size, NULL);
  /* A cubin is compiled for the architecture of the device that the
     compiler asked the driver for (D214). */
  if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
    fprintf(stderr,
            "mdrt: the kernels were compiled for the architecture of another "
            "device than the one that runs them; compile the program with "
            "the device that runs it selected (MDRT_DEVICE or the device of "
            "the execution)\n");
    fflush(stderr);
  }
  check(result, "cuModuleLoadData");
  static int loads = 0;
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE module %p #%d\n", (void *)module, loads++);
  return module;
}

/* Loads a module from PTX text, which the driver compiles. */
CUmodule mgpuModuleLoadJIT(void *data, int optLevel, size_t size) {
  enter();
  char log[4096] = {0};
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *values[] = {(void *)log, (void *)(uintptr_t)sizeof(log),
                    (void *)(uintptr_t)optLevel};
  CUmodule module = NULL;
  int64_t before = beginAccount();
  CUresult result = cuModuleLoadDataEx(&module, data, 3, options, values);
  endAccount(ACCOUNT_MODULES, before, 0);
  if (result != CUDA_SUCCESS) {
    reportUnsupportedPTX(result, data, size, log);
    fprintf(stderr, "mdrt: the driver could not compile a kernel:\n%s\n",
            log);
  }
  check(result, "cuModuleLoadDataEx");
  static int jitLoads = 0;
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE module %p #%d\n", (void *)module, jitLoads++);
  return module;
}

void mgpuModuleUnload(CUmodule module) {
  /* This runs when the program ends, possibly after the driver has shut
     down. */
  if (isPending && sharedStream)
    cuStreamSynchronize(sharedStream);
  isPending = 0;
  /* The program ends: the copies of its constants made outside an
     activation go with it, since another program may later put its own
     constants at the same addresses of the host. A program that is still
     loaded copies its own again when it next needs them. */
  {
    CUresult current = context ? cuCtxSetCurrent(context) : CUDA_SUCCESS;
    releaseDeviceConstants(NULL, 0,
                           !context || current != CUDA_SUCCESS);
  }
  CUresult result = cuModuleUnload(module);
  if (result != CUDA_ERROR_DEINITIALIZED &&
      result != CUDA_ERROR_INVALID_CONTEXT &&
      result != CUDA_ERROR_CONTEXT_IS_DESTROYED)
    check(result, "cuModuleUnload");
}

CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  double start = begin();
  CUfunction function = NULL;
  check(cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE function %p module %p %s\n", (void *)function,
            (void *)module, name);
  noteKernel(function, name);
  end(LOOKUP, start);
  return function;
}

static void finish(void);
static void endKernel(void);

void mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                      intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                      intptr_t blockZ, int32_t sharedMemory, CUstream stream,
                      void **parameters, void **extra, size_t count) {
  (void)count;
  enter();
  /* A launch over no particles is no launch. The driver rejects it. */
  if (gridX <= 0 || gridY <= 0 || gridZ <= 0)
    return;
  double start = begin();
  static long traceCount = 0;
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE launch %ld fn %p grid %ld block %ld\n",
            traceCount++, (void *)function, (long)gridX, (long)blockX);
  int64_t before = -1;
  int launching = -1;
  if (isMemoryDetailed()) {
    launching = findKernel(function);
    if (launching >= 0 && !kernels[launching].launched)
      before = getDeviceUsed();
  }
  check(cuLaunchKernel(function, (unsigned)gridX, (unsigned)gridY,
                       (unsigned)gridZ, (unsigned)blockX, (unsigned)blockY,
                       (unsigned)blockZ, (unsigned)sharedMemory, stream,
                       parameters, extra),
        "cuLaunchKernel");
  if (before >= 0) {
    kernels[launching].launched = 1;
    endAccount(ACCOUNT_LAUNCHES, before, 0);
  }
  lastFunction = function;
  isPending = 1;
  if (start >= 0.0) {
    lastKernel = findKernel(function);
    lastLaunch = start;
    if (lastKernel >= 0)
      kernels[lastKernel].threads =
          (int64_t)(gridX * gridY * gridZ * blockX * blockY * blockZ);
  }
  end(LAUNCH, start);
  /* With MDRT_WAIT set, every launch is waited for, so that a failure is
     reported with the kernel that failed. */
  if (waitsAlways < 0)
    waitsAlways = getenv("MDRT_WAIT") != NULL;
  if (waitsAlways) {
    finish();
    endKernel();
  }
}

/*===----------------------------------------------------------------------===
  Streams and events
  ===----------------------------------------------------------------------===*/

CUstream mgpuStreamCreate(void) {
  enter();
  if (!sharedStream)
    check(cuStreamCreate(&sharedStream, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate");
  return onSide ? sideStream : sharedStream;
}

void mdrtSideBegin(void) {
  mgpuStreamCreate();
  if (!sideStream) {
    check(cuStreamCreate(&sideStream, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate");
    check(cuEventCreate(&forkEvent, CU_EVENT_DISABLE_TIMING),
          "cuEventCreate");
    check(cuEventCreate(&joinEvent, CU_EVENT_DISABLE_TIMING),
          "cuEventCreate");
  }
  check(cuEventRecord(forkEvent, sharedStream), "cuEventRecord");
  check(cuStreamWaitEvent(sideStream, forkEvent, 0), "cuStreamWaitEvent");
  onSide = 1;
}

void mdrtSideEnd(void) {
  check(cuEventRecord(joinEvent, sideStream), "cuEventRecord");
  onSide = 0;
  joinPending = 1;
}

void mdrtSideJoin(void) {
  if (!joinPending)
    return;
  check(cuStreamWaitEvent(sharedStream, joinEvent, 0), "cuStreamWaitEvent");
  joinPending = 0;
}

void mgpuStreamDestroy(CUstream stream) { (void)stream; }

/* The descriptor of a memref of one dimension, as the C interface of MLIR
   passes it. */
struct Buffer1 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

/* A memref of rank 2 as the C interface of MLIR passes it. */
struct Buffer2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

/* Reads of flags of the device that do not wait for the work issued after
   them (D113): mdrtFlagStart copies the flag into memory of the host that
   the device writes while the host goes on, and marks the copy with an
   event; mdrtFlagFinish waits for that event only. The host can then issue
   the work that does not depend on the flag, such as the reciprocal sum,
   before it waits, and the device does not stand idle while the host
   decides. A run has few flags in flight; the slots are reused in turn. */
enum { FLAG_SLOTS = 32 };
static int32_t *flagSlots = NULL;
static CUevent flagEvents[FLAG_SLOTS];
static volatile int32_t *flagSources[FLAG_SLOTS];
static int64_t nextFlagSlot = 0;

/* The flags themselves are in memory of the host that is mapped for the
   device (D118): a kernel stores 1 there across the bus, so a read needs no
   copy on the stream, only an event after the kernels that set it. A copy
   between two kernels held the device for some microseconds each, and the
   two flags of a dual list cost 13 us a step on ubiquitin. The host clears
   a flag when it reads it: no kernel that sets it is issued before the host
   has decided on it. Each place of the module that takes a flag has a
   number of its own and the same memory at every call of its function, so
   that a function called many times, as a step of a minimization is, takes
   no other function's flag. */
enum { MAPPED_FLAGS = 1024 };
static volatile int32_t *mappedFlags = NULL;
static CUdeviceptr mappedDevice = 0;

void _mlir_ciface_mdrtFlagMemory(struct Buffer1 *result, int64_t place) {
  mgpuStreamCreate();
  if (!mappedFlags) {
    void *host = NULL;
    check(cuMemHostAlloc(&host, MAPPED_FLAGS * sizeof(int32_t),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "cuMemHostAlloc");
    memset(host, 0, MAPPED_FLAGS * sizeof(int32_t));
    check(cuMemHostGetDevicePointer(&mappedDevice, host, 0),
          "cuMemHostGetDevicePointer");
    mappedFlags = (volatile int32_t *)host;
  }
  if (place < 0 || place >= MAPPED_FLAGS) {
    fprintf(stderr, "mdrt: the module takes more than %d flags\n",
            MAPPED_FLAGS);
    exit(1);
  }
  int64_t slot = place;
  mappedFlags[slot] = 0;
  void *device = (void *)(uintptr_t)(mappedDevice + slot * sizeof(int32_t));
  result->allocated = device;
  result->aligned = device;
  result->offset = 0;
  result->size = 1;
  result->stride = 1;
}

int64_t mdrtFlagStart(void *device) {
  mgpuStreamCreate();
  if (!flagSlots) {
    check(cuMemAllocHost((void **)&flagSlots,
                         FLAG_SLOTS * sizeof(*flagSlots)),
          "cuMemAllocHost");
    for (int i = 0; i != FLAG_SLOTS; ++i)
      check(cuEventCreate(&flagEvents[i], CU_EVENT_DISABLE_TIMING),
            "cuEventCreate");
  }
  /* The kernels that set flags run on the first stream. */
  int64_t slot = nextFlagSlot++ % FLAG_SLOTS;
  CUdeviceptr address = (CUdeviceptr)device;
  if (mappedFlags && address >= mappedDevice &&
      address < mappedDevice + MAPPED_FLAGS * sizeof(int32_t)) {
    flagSources[slot] = mappedFlags + (address - mappedDevice) / sizeof(int32_t);
  } else {
    flagSources[slot] = NULL;
    check(cuMemcpyDtoHAsync(&flagSlots[slot], address, sizeof(*flagSlots),
                            sharedStream),
          "cuMemcpyDtoHAsync");
  }
  check(cuEventRecord(flagEvents[slot], sharedStream), "cuEventRecord");
  isPending = 1;
  return slot;
}

int32_t mdrtFlagFinish(int64_t slot) {
  double start = begin();
  check(cuEventSynchronize(flagEvents[slot]), "cuEventSynchronize");
  end(WAIT, start);
  volatile int32_t *source = flagSources[slot];
  if (!source)
    return flagSlots[slot];
  int32_t value = *source;
  *source = 0;
  return value;
}

/* Reads and clears a flag that the kernel of an other flag set, after
   mdrtFlagFinish of that flag has waited for the event after the kernel
   (D278): the loops of a run, one kernel, have a flag
   each and one event. A flag outside the mapped words is copied as
   mdrtFlagStart does and waited for. */
int32_t mdrtFlagTake(void *device) {
  CUdeviceptr address = (CUdeviceptr)device;
  if (mappedFlags && address >= mappedDevice &&
      address < mappedDevice + MAPPED_FLAGS * sizeof(int32_t)) {
    volatile int32_t *source =
        mappedFlags + (address - mappedDevice) / sizeof(int32_t);
    int32_t value = *source;
    *source = 0;
    return value;
  }
  return mdrtFlagFinish(mdrtFlagStart(device));
}

/* Waits until the device has run everything that was issued. */
static void finish(void) {
  if (!isPending || !sharedStream)
    return;
  double start = begin();
  if (sideStream)
    check(cuStreamSynchronize(sideStream), "cuStreamSynchronize");
  check(cuStreamSynchronize(sharedStream), "cuStreamSynchronize");
  end(WAIT, start);
  isPending = 0;
}

/* Adds the time since the last launch to the time of its kernel. */
static void endKernel(void) {
  if (lastKernel < 0)
    return;
  kernels[lastKernel].count += 1;
  kernels[lastKernel].seconds += now() - lastLaunch;
  lastKernel = -1;
}

void mgpuStreamSynchronize(CUstream stream) {
  (void)stream;
  if (waitsAlways < 0)
    waitsAlways = getenv("MDRT_WAIT") != NULL;
  if (waitsAlways) {
    finish();
    endKernel();
  }
}

void mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  check(cuStreamWaitEvent(stream, event, 0), "cuStreamWaitEvent");
}

CUevent mgpuEventCreate(void) {
  enter();
  CUevent event = NULL;
  check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
  return event;
}

void mgpuEventDestroy(CUevent event) {
  check(cuEventDestroy(event), "cuEventDestroy");
}

void mgpuEventSynchronize(CUevent event) {
  check(cuEventSynchronize(event), "cuEventSynchronize");
}

void mgpuEventRecord(CUevent event, CUstream stream) {
  check(cuEventRecord(event, stream), "cuEventRecord");
}

/*===----------------------------------------------------------------------===
  Memory
  ===----------------------------------------------------------------------===*/

/* Memory of the device is kept for reuse when it is freed. All work goes
   to one stream, in order, so a block that is freed may be handed to work
   issued after the free without waiting: that work runs after the work that
   used the block. Freeing memory with the driver would wait for the device,
   and a build of a neighbor structure frees a dozen blocks. The live blocks
   are recorded with their sizes, and the free blocks are reused for
   requests of the same size. A block of a size that no later request asks
   for would stay in the pool for the rest of the run, so the sizes that
   compiled code asks for at every step do not follow the builds: a buffer
   indexed by the places of the groups takes the room of the places, not
   the number that a build occupied, and a buffer of the runtime that a
   growth replaces goes back to the driver (D312). */
#define MAX_BLOCKS 4096
/* An activation of an entry that an embedding program keeps across its
   calls (a Python simulation, D215): what the activation
   allocates is its own, and returns to the pool when the program closes
   it, so the memory of the device stays that of one activation however
   many parts it runs (#110). Several activations may be open at once, one
   for each live simulation, and one of them is current while its code
   runs. `mdir run` opens none. */
struct Activation {
  /* Memory shared with the host that the activation allocated. */
  CUdeviceptr *managed;
  size_t numManaged, roomManaged;
  /* The structures of neighbors (groups and matrices) that it made: their
     host records are freed when it closes, and their memory of the device
     returns to the pool with its other blocks. */
  void **handles;
  size_t numHandles, roomHandles;
};
static struct Activation *currentActivation = NULL;

struct Block {
  CUdeviceptr pointer;
  uint64_t size;
  /* The activation that allocated the block, or null. */
  struct Activation *owner;
  /* What the block was last allocated for: a buffer of the runtime's
     structures, or null for compiled code (MDIR_DEVICE_MEMORY=2). */
  const char *tag;
};
static struct Block liveBlocks[MAX_BLOCKS];
static int numLive = 0;
static struct Block freeBlocks[MAX_BLOCKS];
static int numFree = 0;
/* Whether memory of the device was allocated that `liveBlocks` does not
   record, for want of room. */
static int untracked = 0;
/* What the runtime allocates a block for, while it does (null for
   compiled code). */
static const char *allocationTag = NULL;

/* The bytes of the blocks in use and of those in the pool, and their peaks
   since the program started (D294): with MDIR_DEVICE_MEMORY
   set, they are printed when the process ends. */
static uint64_t liveBytes = 0, poolBytes = 0, peakLive = 0, peakReserved = 0;
/* With MDIR_DEVICE_MEMORY=2: the blocks at the peak in use and at the peak
   with the pool (those in the pool after those in use), the driver's use
   at the latter, and the counts of the allocations. */
static struct Block peakLiveBlocks[MAX_BLOCKS];
static int numPeakLive = 0;
static struct Block peakReservedBlocks[2 * MAX_BLOCKS];
static int numPeakReservedLive = 0, numPeakReservedFree = 0;
static int64_t usedAtPeakReserved = -1;
static int64_t numAllocations = 0, numFromPool = 0, numToDriver = 0;
static void noteBytes(void) {
  int detailed = isMemoryDetailed();
  if (liveBytes > peakLive) {
    peakLive = liveBytes;
    if (detailed) {
      memcpy(peakLiveBlocks, liveBlocks, numLive * sizeof(struct Block));
      numPeakLive = numLive;
    }
  }
  if (liveBytes + poolBytes > peakReserved) {
    peakReserved = liveBytes + poolBytes;
    if (detailed) {
      memcpy(peakReservedBlocks, liveBlocks, numLive * sizeof(struct Block));
      memcpy(peakReservedBlocks + numLive, freeBlocks,
             numFree * sizeof(struct Block));
      numPeakReservedLive = numLive;
      numPeakReservedFree = numFree;
      usedAtPeakReserved = getDeviceUsed();
    }
  }
}

/* The blocks of `blocks`, by what they were allocated for and their size,
   the largest totals first: one line for each, and one for each tag for
   the blocks smaller than a MiB. */
struct BlockKind {
  const char *tag;
  uint64_t size;
  int64_t count;
};
static int compareKinds(const void *a, const void *b) {
  const struct BlockKind *x = a, *y = b;
  /* A count of 0 is the blocks under a MiB, whose bytes are the size. */
  uint64_t bx = x->size * (uint64_t)(x->count ? x->count : 1);
  uint64_t by = y->size * (uint64_t)(y->count ? y->count : 1);
  return bx < by ? 1 : bx > by ? -1 : 0;
}
static void printBlocks(const char *state, const struct Block *blocks,
                        int count) {
  static struct BlockKind kinds[2 * MAX_BLOCKS];
  int numKinds = 0;
  for (int i = 0; i != count; ++i) {
    const char *tag = blocks[i].tag ? blocks[i].tag : "compiled code";
    uint64_t size = blocks[i].size < 1048576 ? 0 : blocks[i].size;
    int k = 0;
    while (k != numKinds &&
           !(kinds[k].size == size && strcmp(kinds[k].tag, tag) == 0))
      ++k;
    if (k == numKinds)
      kinds[numKinds++] = (struct BlockKind){tag, size, 0};
    /* The blocks under a MiB are counted together, by their bytes. */
    kinds[k].count += size ? 1 : (int64_t)blocks[i].size;
  }
  for (int k = 0; k != numKinds; ++k)
    if (!kinds[k].size) {
      kinds[k].size = (uint64_t)kinds[k].count;
      kinds[k].count = 0;
    }
  qsort(kinds, numKinds, sizeof(*kinds), compareKinds);
  for (int k = 0; k != numKinds; ++k) {
    int small = kinds[k].count == 0;
    int64_t count = small ? 1 : kinds[k].count;
    fprintf(stderr, "mdrt:   %-7s %10.1f MiB  %5lld x %12llu bytes  %s%s\n",
            state, kinds[k].size * (double)count / 1048576.0,
            (long long)count, (unsigned long long)kinds[k].size,
            kinds[k].tag, small ? " (the blocks under 1 MiB)" : "");
  }
}

static void printDeviceMemory(void) {
  fprintf(stderr,
          "mdrt: device memory (MDIR_DEVICE_MEMORY): peak %.1f MiB allocated, "
          "%.1f MiB with the pool\n",
          peakLive / 1048576.0, peakReserved / 1048576.0);
  if (!isMemoryDetailed())
    return;
  const double mib = 1048576.0;
  int64_t used = context && cuCtxSetCurrent(context) == CUDA_SUCCESS
                     ? getDeviceUsed()
                     : -1;
  fprintf(stderr,
          "mdrt: device memory: %lld allocations of blocks, %lld from the "
          "pool, %lld returned to the driver%s\n",
          (long long)numAllocations, (long long)numFromPool,
          (long long)numToDriver,
          untracked ? "; some blocks were not recorded" : "");
  fprintf(stderr,
          "mdrt: device memory: the driver's use of the device: %.1f MiB "
          "once the context was made, %.1f MiB at the peak with the pool, "
          "%.1f MiB at most at the events, %.1f MiB at the end\n",
          accountGrowth[ACCOUNT_CONTEXT] / mib, usedAtPeakReserved / mib,
          usedMost / mib, used / mib);
  fprintf(stderr, "mdrt: device memory: by account, the growth of the "
                  "driver's use, the bytes asked for, and the events:\n");
  for (int i = 1; i != NUM_ACCOUNTS; ++i)
    if (accountEvents[i])
      fprintf(stderr, "mdrt:   %-36s %10.1f MiB %10.1f MiB %6lld\n",
              accountNames[i], accountGrowth[i] / mib, accountAsked[i] / mib,
              (long long)accountEvents[i]);
  fprintf(stderr,
          "mdrt: device memory at the peak in use (%.1f MiB), by tag and "
          "size:\n",
          peakLive / mib);
  printBlocks("in use", peakLiveBlocks, numPeakLive);
  fprintf(stderr,
          "mdrt: device memory at the peak with the pool (%.1f MiB), by "
          "tag and size:\n",
          peakReserved / mib);
  printBlocks("in use", peakReservedBlocks, numPeakReservedLive);
  printBlocks("pool", peakReservedBlocks + numPeakReservedLive,
              numPeakReservedFree);
}
static void watchDeviceMemory(void) {
  static int registered = 0;
  if (registered)
    return;
  registered = 1;
  const char *value = getenv("MDIR_DEVICE_MEMORY");
  if (value && *value && strcmp(value, "0") != 0)
    atexit(printDeviceMemory);
}

/* Makes the room of a list that grows twice as large. */
static void *growList(void *list, size_t *room, size_t size) {
  *room = *room ? 2 * *room : 8;
  void *grown = realloc(list, *room * size);
  if (!grown) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  return grown;
}

void *mgpuMemAlloc(uint64_t size, CUstream stream, bool isHostShared) {
  (void)stream;
  enter();
  CUdeviceptr pointer = 0;
  if (size == 0)
    return NULL;
  double start = begin();
  if (isHostShared) {
    int64_t before = beginAccount();
    check(cuMemAllocManaged(&pointer, size, CU_MEM_ATTACH_GLOBAL),
          "cuMemAllocManaged");
    endAccount(ACCOUNT_MANAGED, before, (int64_t)size);
    struct Activation *a = currentActivation;
    if (a) {
      if (a->numManaged == a->roomManaged)
        a->managed = growList(a->managed, &a->roomManaged, sizeof(*a->managed));
      a->managed[a->numManaged++] = pointer;
    }
    end(ALLOCATE, start);
    return (void *)pointer;
  }
  watchDeviceMemory();
  for (int i = numFree; i-- != 0;) {
    if (freeBlocks[i].size != size)
      continue;
    pointer = freeBlocks[i].pointer;
    freeBlocks[i] = freeBlocks[--numFree];
    poolBytes -= size;
    ++numFromPool;
    break;
  }
  ++numAllocations;
  if (!pointer) {
    int64_t before = beginAccount();
    check(cuMemAlloc(&pointer, size), "cuMemAlloc");
    endAccount(ACCOUNT_BLOCKS, before, (int64_t)size);
  }
  liveBytes += size;
  if (numLive < MAX_BLOCKS)
    liveBlocks[numLive++] =
        (struct Block){pointer, size, currentActivation, allocationTag};
  else
    untracked = 1;
  noteBytes();
  end(ALLOCATE, start);
  return (void *)pointer;
}

void mgpuMemFree(void *pointer, CUstream stream) {
  (void)stream;
  if (!pointer)
    return;
  double start = begin();
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = numLive; i-- != 0;) {
    if (liveBlocks[i].pointer != address)
      continue;
    struct Block block = liveBlocks[i];
    liveBlocks[i] = liveBlocks[--numLive];
    liveBytes -= block.size;
    if (numFree < MAX_BLOCKS) {
      freeBlocks[numFree++] = block;
      poolBytes += block.size;
      end(ALLOCATE, start);
      return;
    }
    break;
  }
  /* Memory that is not recorded, or that the pool has no room for, goes
     back to the driver once the work that may use it is done. */
  finish();
  check(cuMemFree(address), "cuMemFree");
  ++numToDriver;
  end(ALLOCATE, start);
}

/* Returns a buffer of the runtime's structures that a growth replaces to
   the driver, not to the pool (D312): the buffer that
   replaces it is larger, and a later request of the old size is unlikely,
   so in the pool it would stay for the rest of the run. The caller has
   waited for the device, as a growth does, so the free does not wait. */
static void releaseReplaced(void *pointer) {
  if (!pointer)
    return;
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = numLive; i-- != 0;) {
    if (liveBlocks[i].pointer != address)
      continue;
    liveBytes -= liveBlocks[i].size;
    liveBlocks[i] = liveBlocks[--numLive];
    break;
  }
  check(cuMemFree(address), "cuMemFree");
  ++numToDriver;
}

/* Records a structure of neighbors that the current activation made. */
static void recordCallHandle(void *handle) {
  struct Activation *a = currentActivation;
  if (!a)
    return;
  if (a->numHandles == a->roomHandles)
    a->handles = growList(a->handles, &a->roomHandles, sizeof(*a->handles));
  a->handles[a->numHandles++] = handle;
}

/* Returns true if `pointer` is memory of the device: in a block that the
   library has allocated. Only where it has not recorded every block does it
   ask the driver, which reports an error for memory of the host (and tools
   such as compute-sanitizer count it). */
static bool isOnDevice(void *pointer) {
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = 0; i != numLive; ++i)
    if (address >= liveBlocks[i].pointer &&
        address < liveBlocks[i].pointer + liveBlocks[i].size)
      return true;
  if (!untracked)
    return false;
  unsigned type = 0;
  CUresult result = cuPointerGetAttribute(
      &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)pointer);
  return result == CUDA_SUCCESS && type == CU_MEMORYTYPE_DEVICE;
}

/* Copies between the host and the device, in either direction: the driver
   tells them apart by the address. */
void mgpuMemcpy(void *destination, void *source, size_t size,
                CUstream stream) {
  if (size == 0)
    return;
  double start = begin();
  check(cuMemcpyAsync((CUdeviceptr)destination, (CUdeviceptr)source, size,
                      stream),
        "cuMemcpyAsync");
  isPending = 1;
  end(COPY, start);

  /* The host reads what it has copied from the device, and it may change
     what it has copied to the device. Either way the copy must be done
     when the function returns. */
  if (!isOnDevice(destination) || !isOnDevice(source))
    finish();
}

void mgpuMemset32(void *destination, unsigned int value, size_t count,
                  CUstream stream) {
  check(cuMemsetD32Async((CUdeviceptr)destination, value, count, stream),
        "cuMemsetD32Async");
  isPending = 1;
}

void mgpuMemset16(void *destination, unsigned short value, size_t count,
                  CUstream stream) {
  check(cuMemsetD16Async((CUdeviceptr)destination, value, count, stream),
        "cuMemsetD16Async");
  isPending = 1;
}

/*===----------------------------------------------------------------------===
  FFT of particle mesh Ewald
  ===----------------------------------------------------------------------===

  The transforms of the template of particle mesh Ewald on a device, by
  cuFFT, on the stream that the kernels run on, so that they run in the
  order of the program without a wait. A plan is kept for each grid. */

#include <cufft.h>

/* The descriptor of a buffer of one dimension of f64 on the device. */
typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} DeviceBuffer1D;

/* The same, of f32. */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} DeviceBuffer1DF32;

/* A failure that ends the run but need not end the process: the program
   that embeds the run (a simulation of the Python interface, D196) takes
   it through its handler, which records it, and the run goes on to the end
   of its part, whose state the program discards; without a handler the
   failure is printed and the process exits with status 1. The calls that
   fail return without their work. */
static void (*stopHandler)(const char *) = NULL;
void mdrtCudaSetStopHandler(void (*handler)(const char *)) {
  stopHandler = handler;
}

static void stopWith(const char *message) {
  if (stopHandler) {
    stopHandler(message);
    return;
  }
  fprintf(stderr, "mdir: %s\n", message);
  fflush(stderr);
  exit(1);
}

/* The plans of cuFFT of the process, one for each grid, direction, and
   precision that a program has transformed, kept until the process ends
   and shared by the programs that transform the same grid. A program in
   mixed precision with tunable charges takes four a grid (both directions,
   f32 and f64), so the table grows as programs of other grids come (#323);
   it held 8 before, and the ninth plan ended the process. */
struct FFTPlan {
  int64_t k1, k2, k3;
  int forward, single;
  cufftHandle plan;
};
static struct FFTPlan *fftPlans = NULL;
static int64_t numFFTPlans = 0, fftPlansCapacity = 0;

/* Whether `result` is a success; otherwise the run stops, naming `what`. */
static int checkFFT(cufftResult result, const char *what, int64_t k1,
                    int64_t k2, int64_t k3) {
  if (result == CUFFT_SUCCESS)
    return 1;
  char message[256];
  snprintf(message, sizeof message,
           "%s failed with the cuFFT error %d for the grid of %lld x %lld x "
           "%lld points; the run has failed",
           what, (int)result, (long long)k1, (long long)k2, (long long)k3);
  stopWith(message);
  return 0;
}

/* The plan of a grid, a direction, and a precision (f32 if `single`), on
   the stream of the kernels, in `plan`; 0 if it cannot be made, and the
   run stops. */
static int getFFTPlan(int64_t k1, int64_t k2, int64_t k3, int forward,
                      int single, cufftHandle *plan) {
  for (int64_t i = 0; i != numFFTPlans; ++i)
    if (fftPlans[i].k1 == k1 && fftPlans[i].k2 == k2 &&
        fftPlans[i].k3 == k3 && fftPlans[i].forward == forward &&
        fftPlans[i].single == single) {
      *plan = fftPlans[i].plan;
      return checkFFT(cufftSetStream(*plan, (cudaStream_t)mgpuStreamCreate()),
                      "cufftSetStream", k1, k2, k3);
    }
  if (numFFTPlans == fftPlansCapacity) {
    int64_t capacity = fftPlansCapacity ? 2 * fftPlansCapacity : 8;
    struct FFTPlan *grown =
        realloc(fftPlans, (size_t)capacity * sizeof(struct FFTPlan));
    if (!grown) {
      char message[160];
      snprintf(message, sizeof message,
               "no memory of the host for a plan of FFT of the grid of %lld x "
               "%lld x %lld points; the run has failed",
               (long long)k1, (long long)k2, (long long)k3);
      stopWith(message);
      return 0;
    }
    fftPlans = grown;
    fftPlansCapacity = capacity;
  }
  cufftType type = single ? (forward ? CUFFT_R2C : CUFFT_C2R)
                          : (forward ? CUFFT_D2Z : CUFFT_Z2D);
  cufftHandle made;
  int64_t before = beginAccount();
  if (!checkFFT(cufftPlan3d(&made, (int)k1, (int)k2, (int)k3, type),
                "cufftPlan3d", k1, k2, k3))
    return 0;
  if (before >= 0) {
    /* The bytes asked for are the plan's workspace. */
    size_t workspace = 0;
    cufftGetSize(made, &workspace);
    endAccount(ACCOUNT_FFT, before, (int64_t)workspace);
  }
  if (!checkFFT(cufftSetStream(made, (cudaStream_t)mgpuStreamCreate()),
                "cufftSetStream", k1, k2, k3)) {
    cufftDestroy(made);
    return 0;
  }
  struct FFTPlan *entry = &fftPlans[numFFTPlans++];
  entry->k1 = k1;
  entry->k2 = k2;
  entry->k3 = k3;
  entry->forward = forward;
  entry->single = single;
  entry->plan = made;
  *plan = made;
  return 1;
}

/* The forward transform of the real grid of k1 x k2 x k3 points into the
   half-complex grid of k1 x k2 x (k3 / 2 + 1) numbers, interleaved, as
   mdrtFFTForward3D of the host computes it. */
void _mlir_ciface_mdrtCudaFFTForward3D(DeviceBuffer1D *real,
                                       DeviceBuffer1D *complex, int64_t k1,
                                       int64_t k2, int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  if (getFFTPlan(k1, k2, k3, 1, 0, &plan))
    checkFFT(cufftExecD2Z(plan,
                          (cufftDoubleReal *)(real->aligned + real->offset),
                          (cufftDoubleComplex *)(complex->aligned +
                                                 complex->offset)),
             "cufftExecD2Z", k1, k2, k3);
  isPending = 1;
  end(LAUNCH, start);
}

/* The backward transform, not normalized, which overwrites the
   half-complex grid. */
void _mlir_ciface_mdrtCudaFFTBackward3D(DeviceBuffer1D *complex,
                                        DeviceBuffer1D *real, int64_t k1,
                                        int64_t k2, int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  if (getFFTPlan(k1, k2, k3, 0, 0, &plan))
    checkFFT(cufftExecZ2D(plan,
                          (cufftDoubleComplex *)(complex->aligned +
                                                 complex->offset),
                          (cufftDoubleReal *)(real->aligned + real->offset)),
             "cufftExecZ2D", k1, k2, k3);
  isPending = 1;
  end(LAUNCH, start);
}

/* The transforms of a grid in f32, for the mixed and single modes. */
void _mlir_ciface_mdrtCudaFFTForward3DF32(DeviceBuffer1DF32 *real,
                                          DeviceBuffer1DF32 *complex,
                                          int64_t k1, int64_t k2,
                                          int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  if (getFFTPlan(k1, k2, k3, 1, 1, &plan))
    checkFFT(cufftExecR2C(plan, (cufftReal *)(real->aligned + real->offset),
                          (cufftComplex *)(complex->aligned +
                                           complex->offset)),
             "cufftExecR2C", k1, k2, k3);
  isPending = 1;
  end(LAUNCH, start);
}

void _mlir_ciface_mdrtCudaFFTBackward3DF32(DeviceBuffer1DF32 *complex,
                                           DeviceBuffer1DF32 *real,
                                           int64_t k1, int64_t k2,
                                           int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  if (getFFTPlan(k1, k2, k3, 0, 1, &plan))
    checkFFT(cufftExecC2R(plan,
                          (cufftComplex *)(complex->aligned + complex->offset),
                          (cufftReal *)(real->aligned + real->offset)),
             "cufftExecC2R", k1, k2, k3);
  isPending = 1;
  end(LAUNCH, start);
}

/*===----------------------------------------------------------------------===
 * The buffers of a structure of groups of neighbors (D89)
 *===----------------------------------------------------------------------===*/

/* The buffers that a build of a structure of groups fills, which grow when
   a build finds them too small: the particle at each place, the number of
   entries of each group (a group is 16 places), the entries and the masks
   in blocks of 64, the group and the number within its list of each
   block, and the shift of each place to the frame of its group (D95).
   A dual list (D114) adds the entries and the masks of the inner list,
   each block of the outer list pruned into the same block, and the number
   of entries that each block keeps; they are allocated where compiled code
   first asks for them. The pruning of a dual list adds, also where first
   asked for (D268): the block of each group's list by its
   number within the list (`blockOf`, a group's row as wide as the longest
   list takes blocks; its length is set by `mdrtGroupsReserveBlockOf` at
   each build), the units of work that hold entries of the inner list and
   their number (the active units, in the order of the blocks), their
   counts in runs of 32 and of 1024 blocks (on the way to the active
   units), and the positions of the places in the frames of their
   groups in f32 (`mdrtGroupsPlaced`, three numbers a place). Compiled code
   takes a buffer where it uses it (`mdrtGroupsBuffer`), so a buffer that
   grew is the one it takes. */
enum {
  GROUPS_ORDER,
  GROUPS_COUNTS,
  GROUPS_ENTRIES,
  GROUPS_MASKS,
  GROUPS_UNIT_GROUPS,
  GROUPS_UNIT_ORDINALS,
  GROUPS_SHIFTS,
  GROUPS_INNER_ENTRIES,
  GROUPS_INNER_MASKS,
  GROUPS_INNER_COUNTS,
  GROUPS_BLOCK_OF,
  GROUPS_ACTIVE_UNITS,
  GROUPS_ACTIVE_COUNT,
  GROUPS_ACTIVE_RUNS,
  GROUPS_ACTIVE_WIDE_RUNS,
  GROUPS_PLACED,
  GROUPS_BUFFERS
};

struct Groups {
  void *data[GROUPS_BUFFERS];
  int64_t places;
  int64_t blocks;
  /* The length of `blockOf`. */
  int64_t blockOf;
};

/* A memref of rank 1 as the C interface of MLIR passes it. */
static int64_t getGroupsLength(const struct Groups *groups, int which) {
  switch (which) {
  case GROUPS_ORDER:
  case GROUPS_SHIFTS:
    return groups->places;
  case GROUPS_COUNTS:
    return groups->places / 16;
  case GROUPS_ENTRIES:
  case GROUPS_MASKS:
  case GROUPS_INNER_ENTRIES:
  case GROUPS_INNER_MASKS:
    return groups->blocks * 64;
  case GROUPS_BLOCK_OF:
    return groups->blockOf > 0 ? groups->blockOf : 1;
  case GROUPS_ACTIVE_COUNT:
    return 1;
  case GROUPS_ACTIVE_RUNS:
    return groups->blocks / 32 + 1;
  case GROUPS_ACTIVE_WIDE_RUNS:
    return groups->blocks / 1024 + 1;
  case GROUPS_PLACED:
    return groups->places * 3;
  default:
    return groups->blocks;
  }
}

/* The tags of the buffers (MDIR_DEVICE_MEMORY=2). */
static const char *const groupsTags[GROUPS_BUFFERS] = {
    "groups: order",          "groups: counts",
    "groups: entries",        "groups: masks",
    "groups: unit groups",    "groups: unit ordinals",
    "groups: shifts",         "groups: inner entries",
    "groups: inner masks",    "groups: inner counts",
    "groups: block of",       "groups: active units",
    "groups: active count",   "groups: active runs",
    "groups: active wide runs", "groups: placed"};

static void allocateGroup(struct Groups *groups, int which) {
  allocationTag = groupsTags[which];
  groups->data[which] = mgpuMemAlloc(
      (uint64_t)getGroupsLength(groups, which) * sizeof(int32_t), NULL,
      false);
  allocationTag = NULL;
}

/* The buffers of the outer list; those of the inner one wait for their
   first use. */
static void allocateGroups(struct Groups *groups) {
  for (int which = 0; which != GROUPS_INNER_ENTRIES; ++which)
    allocateGroup(groups, which);
}

/* After a wait for the device: the buffers go back to the driver. */
static void releaseGroups(struct Groups *groups) {
  for (int which = 0; which != GROUPS_BUFFERS; ++which) {
    releaseReplaced(groups->data[which]);
    groups->data[which] = NULL;
  }
}

/* A structure with room for `places` places (a multiple of 16) and
   `blocks` blocks. */
int64_t mdrtGroupsCreate(int64_t places, int64_t blocks) {
  struct Groups *groups = calloc(1, sizeof(struct Groups));
  if (!groups) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  groups->places = (places + 15) / 16 * 16;
  groups->blocks = blocks > 0 ? blocks : 1;
  allocateGroups(groups);
  recordCallHandle(groups);
  return (int64_t)(intptr_t)groups;
}

void _mlir_ciface_mdrtGroupsBuffer(struct Buffer1 *result, int64_t handle,
                                   int64_t which) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (!groups->data[which])
    allocateGroup(groups, (int)which);
  result->allocated = groups->data[which];
  result->aligned = groups->data[which];
  result->offset = 0;
  result->size = getGroupsLength(groups, (int)which);
  result->stride = 1;
}

/* Makes room for `places` places and `blocks` blocks, a quarter more than
   asked, if the structure has less. What the buffers held is lost: the
   caller builds again. */
void mdrtGroupsGrow(int64_t handle, int64_t places, int64_t blocks) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (places <= groups->places && blocks <= groups->blocks)
    return;
  /* No kernel may still use the buffers that are freed. */
  enter();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  releaseGroups(groups);
  if (places > groups->places)
    groups->places = (places + places / 4 + 15) / 16 * 16;
  if (blocks > groups->blocks)
    groups->blocks = blocks + blocks / 4;
  allocateGroups(groups);
}

/* Makes room for `length` entries of `blockOf`, a quarter more than asked,
   if it has less. What it held is lost: the caller fills it again. */
void mdrtGroupsReserveBlockOf(int64_t handle, int64_t length) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (length <= groups->blockOf && groups->data[GROUPS_BLOCK_OF])
    return;
  if (length > groups->blockOf)
    groups->blockOf = length + length / 4;
  if (groups->data[GROUPS_BLOCK_OF]) {
    /* No kernel may still use the buffer that is freed. */
    enter();
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    releaseReplaced(groups->data[GROUPS_BLOCK_OF]);
    groups->data[GROUPS_BLOCK_OF] = NULL;
  }
  allocateGroup(groups, GROUPS_BLOCK_OF);
}

/* The positions of the places in the frames of their groups, f32 and three
   a place, as the C interface of MLIR passes a memref of rank 2. */
void _mlir_ciface_mdrtGroupsPlaced(struct Buffer2 *result, int64_t handle) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (!groups->data[GROUPS_PLACED])
    allocateGroup(groups, GROUPS_PLACED);
  result->allocated = groups->data[GROUPS_PLACED];
  result->aligned = groups->data[GROUPS_PLACED];
  result->offset = 0;
  result->sizes[0] = groups->places;
  result->sizes[1] = 3;
  result->strides[0] = 3;
  result->strides[1] = 1;
}

/*===----------------------------------------------------------------------===
 * The rows of a neighbor matrix
 *===----------------------------------------------------------------------===*/

/* The entries of a neighbor matrix, a row for each particle, which grow
   wider when a build finds a particle with more neighbors than a row
   holds. Compiled code takes the buffer where it uses it
   (`mdrtMatrixEntries`). */
struct Matrix {
  void *data;
  int64_t rows;
  int64_t width;
};

static void allocateMatrix(struct Matrix *matrix) {
  allocationTag = "matrix: rows";
  matrix->data = mgpuMemAlloc(
      (uint64_t)(matrix->rows * matrix->width) * sizeof(int32_t), NULL,
      false);
  allocationTag = NULL;
}

int64_t mdrtMatrixCreate(int64_t rows, int64_t width) {
  struct Matrix *matrix = calloc(1, sizeof(struct Matrix));
  if (!matrix) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  matrix->rows = rows > 0 ? rows : 1;
  matrix->width = width > 0 ? width : 1;
  allocateMatrix(matrix);
  recordCallHandle(matrix);
  return (int64_t)(intptr_t)matrix;
}

void _mlir_ciface_mdrtMatrixEntries(struct Buffer2 *result, int64_t handle) {
  struct Matrix *matrix = (struct Matrix *)(intptr_t)handle;
  result->allocated = matrix->data;
  result->aligned = matrix->data;
  result->offset = 0;
  result->sizes[0] = matrix->rows;
  result->sizes[1] = matrix->width;
  result->strides[0] = matrix->width;
  result->strides[1] = 1;
}

/* Makes the rows hold `width` entries, a quarter more than asked, if they
   hold fewer. What they held is lost: the caller builds again. */
void mdrtMatrixGrow(int64_t handle, int64_t width) {
  struct Matrix *matrix = (struct Matrix *)(intptr_t)handle;
  if (width <= matrix->width)
    return;
  enter();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  releaseReplaced(matrix->data);
  matrix->width = width + width / 4;
  allocateMatrix(matrix);
}

/*===----------------------------------------------------------------------===
 * The slots of the pairs of images of a learned potential
 *===----------------------------------------------------------------------===*/

/* The candidates of the pairs of images that each build of a neighbor
   matrix gives its particles, a row (j, s_a, s_b, s_c) for each, and the
   slots that each step writes them into, a row (i, j, s_a, s_b, s_c) for
   each (D277, docs/metatomic-ml1.md, Section 12).
   A build sets their number, `length`; the buffers grow, a quarter more
   than asked, when a build needs more. What they held is lost: the build
   writes the candidates anew. Compiled code takes a buffer where it uses it
   (`mdrtEdgesBuffer`). */
struct Edges {
  void *candidates;
  void *slots;
  int64_t capacity;
  int64_t length;
};

static void allocateEdges(struct Edges *edges) {
  allocationTag = "edges: candidates";
  edges->candidates = mgpuMemAlloc(
      (uint64_t)edges->capacity * 4 * sizeof(int32_t), NULL, false);
  allocationTag = "edges: slots";
  edges->slots = mgpuMemAlloc(
      (uint64_t)edges->capacity * 5 * sizeof(int32_t), NULL, false);
  allocationTag = NULL;
}

int64_t mdrtEdgesCreate(void) {
  struct Edges *edges = calloc(1, sizeof(struct Edges));
  if (!edges) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  edges->capacity = 1;
  allocateEdges(edges);
  recordCallHandle(edges);
  return (int64_t)(intptr_t)edges;
}

/* The builds of slots in the process, which numbers each. */
static int64_t edgeBuilds = 0;

/* Makes the buffers hold `length` rows, and sets their number. Returns the
   number of this build in the process, which no other build has, so that a
   reader can tell that the slots changed. */
int64_t mdrtEdgesReserve(int64_t handle, int64_t length) {
  struct Edges *edges = (struct Edges *)(intptr_t)handle;
  if (length < 0) {
    fprintf(stderr, "mdrt: a build counted %lld pairs of images\n",
            (long long)length);
    abort();
  }
  if (length > edges->capacity) {
    enter();
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    releaseReplaced(edges->candidates);
    releaseReplaced(edges->slots);
    edges->capacity = length + length / 4;
    allocateEdges(edges);
  }
  edges->length = length;
  return ++edgeBuilds;
}

/* The candidates (`which` 0, four numbers a row) or the slots (1, five),
   as many rows as the last build set. */
void _mlir_ciface_mdrtEdgesBuffer(struct Buffer2 *result, int64_t handle,
                                  int64_t which) {
  struct Edges *edges = (struct Edges *)(intptr_t)handle;
  int64_t columns = which == 0 ? 4 : 5;
  void *data = which == 0 ? edges->candidates : edges->slots;
  result->allocated = data;
  result->aligned = data;
  result->offset = 0;
  result->sizes[0] = edges->length;
  result->sizes[1] = columns;
  result->strides[0] = columns;
  result->strides[1] = 1;
}

/*===----------------------------------------------------------------------===
 * Calls of an entry by an embedding program
 *===----------------------------------------------------------------------===*/

/* An embedding program opens an activation before it first calls an entry
   (D215), enters it whenever the code of the activation
   runs, leaves it when the code waits between parts, and closes it when
   the activation ends: its blocks return to the pool, where later
   activations take the blocks of the same sizes. */
void *mdrtDeviceActivationOpen(void) {
  struct Activation *a = calloc(1, sizeof(struct Activation));
  if (!a) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  return a;
}

void mdrtDeviceActivationEnter(void *activation) {
  currentActivation = activation;
}

void mdrtDeviceActivationLeave(void) { currentActivation = NULL; }

/* Whether the driver has shut down, as it may have when a simulation ends
   after the program's own end (mgpuModuleUnload): the memory of the device
   is then gone with it. */
static int isDriverGone(void) {
  if (!context)
    return 1;
  CUresult result = cuCtxSetCurrent(context);
  return result == CUDA_ERROR_DEINITIALIZED ||
         result == CUDA_ERROR_INVALID_CONTEXT ||
         result == CUDA_ERROR_CONTEXT_IS_DESTROYED;
}

void mdrtDeviceActivationClose(void *activation) {
  struct Activation *a = activation;
  if (!a)
    return;
  if (currentActivation == a)
    currentActivation = NULL;
  int gone = isDriverGone();
  /* No work of the activation may still use a block that is reused or
     freed. */
  if (!gone)
    finish();
  for (int i = numLive; i-- != 0;) {
    if (liveBlocks[i].owner != a)
      continue;
    struct Block block = liveBlocks[i];
    liveBlocks[i] = liveBlocks[--numLive];
    block.owner = NULL;
    liveBytes -= block.size;
    if (numFree < MAX_BLOCKS) {
      freeBlocks[numFree++] = block;
      poolBytes += block.size;
    } else if (!gone)
      check(cuMemFree(block.pointer), "cuMemFree");
  }
  for (size_t i = 0; i != a->numManaged && !gone; ++i)
    check(cuMemFree(a->managed[i]), "cuMemFree");
  /* The copies of the constants of the bodies that the activation made. */
  releaseDeviceConstants(a, 0, gone);
  for (size_t i = 0; i != a->numHandles; ++i)
    free(a->handles[i]);
  free(a->managed);
  free(a->handles);
  free(a);
}

/* Memory of the device that an embedding program holds outside any
   activation, and copies to and from it on the stream of the kernels: the
   state of a simulation at the end of its last part, which a part that
   fails returns to (D196, D215). */
void *mdrtDeviceAllocateKept(uint64_t size) {
  enter();
  CUdeviceptr pointer = 0;
  if (size) {
    int64_t before = beginAccount();
    check(cuMemAlloc(&pointer, size), "cuMemAlloc");
    endAccount(ACCOUNT_KEPT, before, (int64_t)size);
  }
  return (void *)pointer;
}

void mdrtDeviceFreeKept(void *pointer) {
  if (!pointer || isDriverGone())
    return;
  finish();
  check(cuMemFree((CUdeviceptr)pointer), "cuMemFree");
}

/* Copies `size` bytes from the device to the device, after the work issued
   before it and before the work issued after it; the host does not wait. */
void mdrtDeviceCopyWithin(void *destination, const void *source,
                          uint64_t size) {
  if (!size)
    return;
  mgpuStreamCreate();
  check(cuMemcpyAsync((CUdeviceptr)destination, (CUdeviceptr)source, size,
                      sharedStream),
        "cuMemcpyAsync");
  isPending = 1;
}

/* Copies `size` bytes from the device to the host once the work issued
   before it is done, and returns when they are there. */
void mdrtDeviceCopyToHost(void *destination, const void *source,
                          uint64_t size) {
  if (!size)
    return;
  mgpuStreamCreate();
  check(cuMemcpyDtoHAsync(destination, (CUdeviceptr)source, size,
                          sharedStream),
        "cuMemcpyDtoHAsync");
  isPending = 1;
  finish();
}

/*===----------------------------------------------------------------------===
 * Work of a plugin of learned potentials (D270)
 *===----------------------------------------------------------------------===*/

/* The stream of the kernels, for a plugin that issues work there itself (a
   learned potential, docs/metatomic-ml1.md): that work runs in order with
   the kernels, and the library counts it as issued, so that the host waits
   for it before it reads what the device computed. */
void *mdrtDeviceExternalStream(void) {
  mgpuStreamCreate();
  isPending = 1;
  return (void *)sharedStream;
}

/* Memory of the device that the library records as its own, outside any
   activation: a copy from or to it does not wait for the device, and an
   activation that closes does not take it back. A plugin's outputs, which
   live as long as the plugin's model. */
void *mdrtDeviceAllocateRecorded(uint64_t size) {
  enter();
  CUdeviceptr pointer = 0;
  if (!size)
    return NULL;
  int64_t before = beginAccount();
  check(cuMemAlloc(&pointer, size), "cuMemAlloc");
  endAccount(ACCOUNT_KEPT, before, (int64_t)size);
  if (numLive < MAX_BLOCKS)
    liveBlocks[numLive++] =
        (struct Block){pointer, size, NULL, "a plugin's outputs"};
  else
    untracked = 1;
  return (void *)pointer;
}

void mdrtDeviceFreeRecorded(void *pointer) {
  if (!pointer || isDriverGone())
    return;
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = numLive; i-- != 0;)
    if (liveBlocks[i].pointer == address) {
      liveBlocks[i] = liveBlocks[--numLive];
      break;
    }
  finish();
  check(cuMemFree(address), "cuMemFree");
}

/*===----------------------------------------------------------------------===
 * Views of the buffers of an embedding program (D220)
 *===----------------------------------------------------------------------===*/

/* The ordinal of the device of the context, among the devices that
   CUDA_VISIBLE_DEVICES leaves visible: the index under which a consumer of
   a view (the DLPack specification's device_id) names it. */
int32_t mdrtDeviceOrdinal(void) {
  enter();
  CUdevice device;
  check(cuCtxGetDevice(&device), "cuCtxGetDevice");
  return (int32_t)device;
}

/* Makes the stream of a consumer wait for the work issued on the stream of
   the kernels so far, without a wait of the host, as the DLPack Python
   specification asks of a producer (`__dlpack__(stream=...)`): 1 is the
   legacy default stream, 2 the per-thread default stream, any other value a
   CUstream of the context. */
void mdrtDeviceHandOff(uint64_t consumer) {
  static CUevent handoffEvent = NULL;
  mgpuStreamCreate();
  if (!handoffEvent)
    check(cuEventCreate(&handoffEvent, CU_EVENT_DISABLE_TIMING),
          "cuEventCreate");
  CUstream stream = consumer == 1   ? CU_STREAM_LEGACY
                    : consumer == 2 ? CU_STREAM_PER_THREAD
                                    : (CUstream)(uintptr_t)consumer;
  check(cuEventRecord(handoffEvent, sharedStream), "cuEventRecord");
  check(cuStreamWaitEvent(stream, handoffEvent, 0), "cuStreamWaitEvent");
}

/* Waits for all the work issued in the context, on any stream: that of a
   consumer that read the buffers of a view, before they are written or
   freed. */
void mdrtDeviceWaitAll(void) {
  if (isDriverGone())
    return;
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  isPending = 0;
}

/*===----------------------------------------------------------------------===
  The structures of the tuples, kept from one activation to the next
  (D253, mdrt_tuples.h)
  ===----------------------------------------------------------------------===

  As in the runtime of the host (mdrt.c), with the incidence structure and
  the copy of the members on the device. What the store keeps on the device
  is memory of its own, outside the activations and the pool of blocks: an
  activation that ends leaves it, and it is freed when the store closes. The
  renumbered members are on the host; the identities are read from the
  device where the members are renumbered. */

#include "mdrt_tuples.h"

static struct MdrtTupleStore *currentTuples = NULL;

void *mdrtDeviceTuplesOpen(void) { return mdrtTupleStoreOpen(); }

void mdrtDeviceTuplesBegin(void *store) { mdrtTupleStoreBegin(store); }

void mdrtDeviceTuplesKeep(void *store, int32_t keeps) {
  ((struct MdrtTupleStore *)store)->keeps = keeps != 0;
}

void mdrtDeviceTuplesEnter(void *store) { currentTuples = store; }

void mdrtDeviceTuplesLeave(void) { currentTuples = NULL; }

void mdrtDeviceTuplesClose(void *opaque) {
  struct MdrtTupleStore *store = opaque;
  if (!store)
    return;
  if (currentTuples == store)
    currentTuples = NULL;
  if (begin() >= 0.0 && (store->deviceBytes[0] || store->deviceBytes[3]))
    fprintf(stderr,
            "mdrt: the store of the tuples kept on the device %lld bytes of "
            "incidence structures, %lld of renumbered members, %lld of the "
            "members and indexes of the tuple sets, and %lld of the order\n",
            (long long)store->deviceBytes[0], (long long)store->deviceBytes[1],
            (long long)store->deviceBytes[2], (long long)store->deviceBytes[3]);
  int gone = isDriverGone();
  /* No work may still read what is freed. */
  if (!gone)
    finish();
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    free(e->file);
    free(e->ids);
    free(e->members);
    if (!gone && e->deviceIncidence)
      check(cuMemFree((CUdeviceptr)e->deviceIncidence), "cuMemFree");
    if (!gone && e->deviceMembers)
      check(cuMemFree((CUdeviceptr)e->deviceMembers), "cuMemFree");
    if (!gone && e->deviceRenumbered)
      check(cuMemFree((CUdeviceptr)e->deviceRenumbered), "cuMemFree");
    if (!gone && e->ownsSet) {
      void *shared[3] = {e->deviceFile, e->deviceFirst, e->deviceEntries};
      for (int k = 0; k != 3; ++k)
        if (shared[k])
          check(cuMemFree((CUdeviceptr)shared[k]), "cuMemFree");
    }
    free(e);
  }
  if (!gone) {
    void *own[3] = {store->deviceIds, store->deviceFlag, store->devicePlace};
    for (int k = 0; k != 3; ++k)
      if (own[k])
        check(cuMemFree((CUdeviceptr)own[k]), "cuMemFree");
  }
  free(store->staging);
  free(store->entries);
  free(store);
}

void mdrtDeviceTuplesCounts(void *opaque, int64_t *counts) {
  struct MdrtTupleStore *store = opaque;
  counts[0] = store ? store->kept : 0;
  counts[1] = store ? store->built : 0;
}

/* Copies between the host and the device, done when they return. */
static void tuplesToDevice(void *device, const void *host, size_t bytes) {
  if (!bytes)
    return;
  mgpuStreamCreate();
  double start = begin();
  check(cuMemcpyHtoDAsync((CUdeviceptr)device, host, bytes, sharedStream),
        "cuMemcpyHtoDAsync");
  isPending = 1;
  end(COPY, start);
  finish();
}

static void tuplesToHost(void *host, const void *device, size_t bytes) {
  if (!bytes)
    return;
  mgpuStreamCreate();
  double start = begin();
  check(cuMemcpyDtoHAsync(host, (CUdeviceptr)device, bytes, sharedStream),
        "cuMemcpyDtoHAsync");
  isPending = 1;
  end(COPY, start);
  finish();
}

/* Memory of the device that the store keeps. */
static void *tuplesDeviceKept(size_t bytes) {
  CUdeviceptr pointer = 0;
  if (bytes) {
    int64_t before = beginAccount();
    check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
    endAccount(ACCOUNT_TUPLES, before, (int64_t)bytes);
  }
  return (void *)pointer;
}

/* Builds the incidence structure of `members` at `device`: the rows are
   written on the host a range of particles at a time, in memory of a few
   megabytes (16 MiB at most), and each range is
   uploaded before the next is written. The host never holds the whole
   structure (355 MB for the dihedrals of Cellulose), whose memory would be
   mapped and faulted in at every build (#294). */
static void tuplesBuildOnDevice(void *device, const int32_t *members,
                                int64_t tuples, int64_t arity,
                                int64_t particles, int64_t width) {
  /* The memory of the ranges is the store's, taken once and cleared once:
     taking and clearing it for every structure cost more than the build of
     the structures of JAC (0.5 ms with 4 MiB, 1.2 ms with 16). */
  enum { STAGING_BYTES = 16 << 20 };
  struct MdrtTupleStore *store = currentTuples;
  struct MdrtTupleIndex index;
  mdrtTupleIndexBuild(&index, members, tuples, arity, particles);
  size_t rowBytes = (size_t)width * sizeof(int32_t);
  int64_t rows = (int64_t)(STAGING_BYTES / rowBytes);
  if (rows < 1)
    rows = 1;
  size_t needed = (size_t)rows * rowBytes;
  int32_t *staging = store ? store->staging : NULL;
  if (!staging || (store && store->stagingBytes < needed)) {
    if (store)
      free(store->staging);
    staging = calloc(1, needed);
    if (!staging) {
      fprintf(stderr, "mdrt: out of memory of the host\n");
      abort();
    }
    if (store) {
      store->staging = staging;
      store->stagingBytes = needed;
    }
  }
  for (int64_t begin = 0; begin < particles; begin += rows) {
    int64_t end = begin + rows < particles ? begin + rows : particles;
    mdrtTuplesRows(&index, members, arity, width, begin, end, staging);
    tuplesToDevice((char *)device + (size_t)begin * rowBytes, staging,
                   (size_t)(end - begin) * rowBytes);
  }
  if (!store)
    free(staging);
  mdrtTupleIndexFree(&index);
}

double mdrtDeviceTuplesSeconds(void *opaque) {
  struct MdrtTupleStore *store = opaque;
  return store ? store->seconds : 0.0;
}

static void tuplesRenumber(struct MdrtMemref2 *result,
                           struct MdrtMemref2 *members,
                           struct MdrtMemref1 *ids) {
  enter();
  const int32_t *file = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  int64_t particles = ids->size;
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  int32_t *id = mdrtTuplesAllocate((size_t)particles * sizeof(int32_t));
  tuplesToHost(id, (const int32_t *)ids->aligned + ids->offset,
               (size_t)particles * sizeof(int32_t));
  if (!currentTuples) {
    int32_t *fresh = mdrtTuplesAllocate(bytes);
    mdrtTuplesRenumber(file, tuples, arity, id, particles, fresh);
    mdrtTuplesSetMemref2(result, fresh, tuples, arity);
    free(id);
    return;
  }
  int found = 0;
  struct MdrtTuples *e = mdrtTupleStoreTake(currentTuples, file, tuples, arity,
                                            id, particles, &found);
  if (!found) {
    mdrtTuplesRenumber(file, tuples, arity, id, particles, e->members);
  } else if (mdrtTuplesChecks()) {
    int32_t *anew = mdrtTuplesAllocate(bytes);
    mdrtTuplesRenumber(file, tuples, arity, id, particles, anew);
    if (memcmp(anew, e->members, bytes) != 0)
      mdrtTuplesDiffer("the renumbered members of a tuple set");
    free(anew);
  }
  free(id);
  ++e->held;
  mdrtTuplesSetMemref2(result, e->members, tuples, arity);
}

void _mlir_ciface_mdrtDeviceTuplesRenumber(struct MdrtMemref2 *result,
                                           struct MdrtMemref2 *members,
                                           struct MdrtMemref1 *ids) {
  double start = mdrtTuplesNow();
  tuplesRenumber(result, members, ids);
  mdrtTuplesSpent(currentTuples, start);
}

static void tuplesIncidence(struct MdrtMemref2 *result,
                            struct MdrtMemref2 *members, int64_t particles) {
  enter();
  const int32_t *given = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  struct MdrtTuples *e = mdrtTupleStoreFind(currentTuples, given);
  if (!e && currentTuples) {
    /* The members of the files, in their order. */
    int found = 0;
    e = mdrtTupleStoreTake(currentTuples, given, tuples, arity, NULL,
                           particles, &found);
    e->given = given;
  }
  if (e && e->particles != particles)
    e = NULL;
  /* The width does not depend on the order: an entry that has had a
     structure has the width of the next. */
  int64_t width = e && e->width
                      ? e->width
                      : mdrtTuplesWidth(given, tuples, arity, particles);
  size_t bytes = (size_t)(particles * width) * sizeof(int32_t);
  if (!e) {
    allocationTag = "tuples: incidence";
    void *device = mgpuMemAlloc(bytes, NULL, false);
    allocationTag = NULL;
    tuplesBuildOnDevice(device, given, tuples, arity, particles, width);
    mdrtTuplesSetMemref2(result, device, particles, width);
    return;
  }
  if (!e->hasIncidence) {
    if (!e->deviceIncidence || e->width != width) {
      if (e->deviceIncidence) {
        finish();
        check(cuMemFree((CUdeviceptr)e->deviceIncidence), "cuMemFree");
      }
      e->deviceIncidence = tuplesDeviceKept(bytes);
    }
    e->width = width;
    tuplesBuildOnDevice(e->deviceIncidence, given, tuples, arity, particles,
                        width);
    e->hasIncidence = 1;
  } else if (mdrtTuplesChecks()) {
    int64_t anewWidth = mdrtTuplesWidth(given, tuples, arity, particles);
    int32_t *anew =
        mdrtTuplesAllocate((size_t)(particles * anewWidth) * sizeof(int32_t));
    mdrtTuplesFill(given, tuples, arity, particles, anewWidth, anew);
    int32_t *kept = mdrtTuplesAllocate(bytes);
    tuplesToHost(kept, e->deviceIncidence, bytes);
    if (anewWidth != width ||
        !mdrtTuplesSameRows(anew, kept, arity, particles, width))
      mdrtTuplesDiffer("the incidence structure of a tuple set");
    free(kept);
    free(anew);
  }
  ++e->held;
  mdrtTuplesSetMemref2(result, e->deviceIncidence, particles, width);
}

void _mlir_ciface_mdrtDeviceTuplesIncidence(struct MdrtMemref2 *result,
                                            struct MdrtMemref2 *members,
                                            int64_t particles) {
  double start = mdrtTuplesNow();
  tuplesIncidence(result, members, particles);
  mdrtTuplesSpent(currentTuples, start);
}

/* The members on the device, for the loops that evaluate each tuple once:
   those that the incidence structure was built from. */
static void tuplesMembers(struct MdrtMemref2 *result,
                          struct MdrtMemref2 *members) {
  enter();
  const int32_t *given = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  struct MdrtTuples *e = mdrtTupleStoreFind(currentTuples, given);
  if (!e) {
    allocationTag = "tuples: members";
    void *device = mgpuMemAlloc(bytes, NULL, false);
    allocationTag = NULL;
    tuplesToDevice(device, given, bytes);
    mdrtTuplesSetMemref2(result, device, tuples, arity);
    return;
  }
  if (!e->hasDeviceMembers) {
    if (!e->deviceMembers)
      e->deviceMembers = tuplesDeviceKept(bytes);
    tuplesToDevice(e->deviceMembers, given, bytes);
    e->hasDeviceMembers = 1;
  } else if (mdrtTuplesChecks()) {
    int32_t *kept = mdrtTuplesAllocate(bytes);
    tuplesToHost(kept, e->deviceMembers, bytes);
    if (memcmp(kept, given, bytes) != 0)
      mdrtTuplesDiffer("the copy on the device of the members of a tuple "
                       "set");
    free(kept);
  }
  ++e->held;
  mdrtTuplesSetMemref2(result, e->deviceMembers, tuples, arity);
}

void _mlir_ciface_mdrtDeviceTuplesMembers(struct MdrtMemref2 *result,
                                          struct MdrtMemref2 *members) {
  double start = mdrtTuplesNow();
  tuplesMembers(result, members);
  mdrtTuplesSpent(currentTuples, start);
}

/* Where the code would free what it took: the renumbered members, on the
   host, or a structure on the device. */
void _mlir_ciface_mdrtDeviceTuplesRelease(struct MdrtMemref2 *buffer) {
  void *pointer = buffer->allocated;
  if (!pointer || mdrtTupleStoreGiveBack(currentTuples, pointer))
    return;
  if (isOnDevice(pointer))
    mgpuMemFree(pointer, NULL);
  else
    free(pointer);
}

/*===----------------------------------------------------------------------===
  The structures of the tuples built on the device (D257, #295)
  ===----------------------------------------------------------------------===

  The code compares the identities that it holds with the store's copy of
  the last order (a kernel that writes a flag, which mdrtDeviceTuplesOrderEnd
  reads), takes an entry for each tuple set (mdrtDeviceTuplesTake), and,
  where the entry is to be built, launches three kernels: the inverse of
  the order, the renumbered members, and the rows of the incidence
  structure, each element written by one thread. The rows of place p are
  those of the particle at p in the order of the files, from the index of
  the tuples of each particle in that order, which the host builds once and
  uploads once for each tuple set; their members are relabeled. The entry
  becomes valid when the code says that it is built
  (mdrtDeviceTuplesBuilt), never before: an activation that ends before
  leaves it to be built again. A failed launch ends the process (check). */

static struct MdrtTupleStore *deviceTupleStore(void) {
  if (!currentTuples) {
    fprintf(stderr, "mdrt: the structures of the tuples are built on the "
                    "device without a store; this is a defect of mdir\n");
    abort();
  }
  return currentTuples;
}

static void setMemref1(struct MdrtMemref1 *result, void *memory,
                       int64_t size) {
  result->allocated = memory;
  result->aligned = memory;
  result->offset = 0;
  result->size = size;
  result->stride = 1;
}

/* The store's copy of the identities of the last order, with the flag of
   the comparison cleared; a copy that is new holds no identity (-1), so
   that the comparison finds a new order. */
void _mlir_ciface_mdrtDeviceTuplesOrderKept(struct MdrtMemref1 *result,
                                            struct MdrtMemref1 *ids) {
  enter();
  double start = mdrtTuplesNow();
  struct MdrtTupleStore *store = deviceTupleStore();
  int64_t n = ids->size;
  mgpuStreamCreate();
  if (store->deviceIdsCount != n) {
    if (store->deviceIds) {
      finish();
      check(cuMemFree((CUdeviceptr)store->deviceIds), "cuMemFree");
    }
    store->deviceIds = tuplesDeviceKept((size_t)n * sizeof(int32_t));
    store->deviceIdsCount = n;
    store->deviceBytes[3] += n * (int64_t)sizeof(int32_t) + 4;
    if (n)
      check(cuMemsetD32Async((CUdeviceptr)store->deviceIds, 0xFFFFFFFFu,
                             (size_t)n, sharedStream),
            "cuMemsetD32Async");
    ++store->order;
  }
  if (!store->deviceFlag)
    store->deviceFlag = tuplesDeviceKept(sizeof(int32_t));
  check(cuMemsetD32Async((CUdeviceptr)store->deviceFlag, 0, 1, sharedStream),
        "cuMemsetD32Async");
  isPending = 1;
  setMemref1(result, store->deviceIds, n);
  mdrtTuplesSpent(store, start);
}

void _mlir_ciface_mdrtDeviceTuplesOrderFlag(struct MdrtMemref1 *result) {
  setMemref1(result, deviceTupleStore()->deviceFlag, 1);
}

/* The flag is read (4 bytes); where an identity differed, the identities
   are copied on the device and the number of the order advances. */
void _mlir_ciface_mdrtDeviceTuplesOrderEnd(struct MdrtMemref1 *ids) {
  enter();
  double start = mdrtTuplesNow();
  struct MdrtTupleStore *store = deviceTupleStore();
  int32_t differs = 0;
  tuplesToHost(&differs, store->deviceFlag, sizeof(int32_t));
  if (differs) {
    if (ids->size)
      check(cuMemcpyDtoDAsync(
                (CUdeviceptr)store->deviceIds,
                (CUdeviceptr)((const int32_t *)ids->aligned + ids->offset),
                (size_t)ids->size * sizeof(int32_t), sharedStream),
            "cuMemcpyDtoDAsync");
    isPending = 1;
    ++store->order;
  }
  mdrtTuplesSpent(store, start);
}

/* Compares a structure built or kept on the device with the one that the
   host builds from the same inputs (MDRT_CHECK_KEPT_TUPLES). */
static void tuplesCheckOnDevice(struct MdrtTupleStore *store,
                                struct MdrtTuples *e) {
  int64_t tuples = e->tuples, arity = e->arity, particles = e->particles;
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  int32_t *renumbered = mdrtTuplesAllocate(bytes);
  if (e->ordered) {
    int32_t *ids = mdrtTuplesAllocate((size_t)particles * sizeof(int32_t));
    tuplesToHost(ids, store->deviceIds, (size_t)particles * sizeof(int32_t));
    mdrtTuplesRenumber(e->file, tuples, arity, ids, particles, renumbered);
    free(ids);
    int32_t *kept = mdrtTuplesAllocate(bytes);
    tuplesToHost(kept, e->deviceRenumbered, bytes);
    if (memcmp(kept, renumbered, bytes) != 0)
      mdrtTuplesDiffer("the renumbered members of a tuple set on the device");
    free(kept);
  } else {
    memcpy(renumbered, e->file, bytes);
  }
  int64_t width = mdrtTuplesWidth(renumbered, tuples, arity, particles);
  size_t structure = (size_t)(particles * width) * sizeof(int32_t);
  int32_t *anew = mdrtTuplesAllocate(structure);
  mdrtTuplesFill(renumbered, tuples, arity, particles, width, anew);
  int32_t *kept = mdrtTuplesAllocate(structure);
  if (width == e->width)
    tuplesToHost(kept, e->deviceIncidence, structure);
  if (width != e->width ||
      !mdrtTuplesSameRows(anew, kept, arity, particles, width))
    mdrtTuplesDiffer("the incidence structure of a tuple set on the device");
  free(kept);
  free(anew);
  free(renumbered);
}

static int sameSet(const struct MdrtTuples *e, const int32_t *file,
                   int64_t tuples, int64_t arity, int64_t particles) {
  return e->device && e->tuples == tuples && e->arity == arity &&
         e->particles == particles &&
         memcmp(e->file, file, (size_t)(tuples * arity) * sizeof(int32_t)) ==
             0;
}

/* The members in the order of the files and the index of the tuples of
   each particle, on the device, for a tuple set that has none yet. */
static void uploadSet(struct MdrtTuples *e) {
  int64_t tuples = e->tuples, arity = e->arity, particles = e->particles;
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  struct MdrtTupleIndex index;
  mdrtTupleIndexBuild(&index, e->file, tuples, arity, particles);
  int32_t *first = mdrtTuplesAllocate((size_t)(particles + 1) * sizeof(int32_t));
  int64_t widest = 0;
  for (int64_t p = 0; p <= particles; ++p) {
    first[p] = (int32_t)index.first[p];
    if (p < particles && index.first[p + 1] - index.first[p] > widest)
      widest = index.first[p + 1] - index.first[p];
  }
  e->widest = widest;
  e->deviceFile = tuplesDeviceKept(bytes);
  tuplesToDevice(e->deviceFile, e->file, bytes);
  e->deviceFirst = tuplesDeviceKept((size_t)(particles + 1) * sizeof(int32_t));
  tuplesToDevice(e->deviceFirst, first, (size_t)(particles + 1) * sizeof(int32_t));
  e->deviceEntries = tuplesDeviceKept(2 * bytes);
  tuplesToDevice(e->deviceEntries, index.entries, 2 * bytes);
  e->ownsSet = 1;
  free(first);
  mdrtTupleIndexFree(&index);
}

/* An entry for the members `members` of `particles` particles, in the
   current order (`ordered`) or in the order of the files: one taken as it
   is, or one to build, which mdrtDeviceTuplesBuild then says. */
int64_t _mlir_ciface_mdrtDeviceTuplesTake(struct MdrtMemref2 *members,
                                          int64_t particles, int32_t ordered) {
  enter();
  double start = mdrtTuplesNow();
  struct MdrtTupleStore *store = deviceTupleStore();
  const int32_t *file = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    if (!sameSet(e, file, tuples, arity, particles) ||
        e->ordered != (ordered != 0) || !e->deviceValid ||
        (ordered && e->order != store->order) ||
        (!store->keeps && e->held == 0))
      continue;
    ++store->kept;
    e->building = 0;
    ++e->held;
    if (mdrtTuplesChecks())
      tuplesCheckOnDevice(store, e);
    mdrtTuplesSpent(store, start);
    return (int64_t)(intptr_t)e;
  }
  ++store->built;
  struct MdrtTuples *e = NULL;
  for (size_t i = 0; i != store->count && !e; ++i) {
    struct MdrtTuples *other = store->entries[i];
    if (sameSet(other, file, tuples, arity, particles) &&
        other->ordered == (ordered != 0) && other->held == 0)
      e = other;
  }
  if (!e) {
    e = calloc(1, sizeof(struct MdrtTuples));
    if (!e) {
      fprintf(stderr, "mdrt: out of memory of the host\n");
      abort();
    }
    e->device = 1;
    e->ordered = ordered != 0;
    e->tuples = tuples;
    e->arity = arity;
    e->particles = particles;
    e->file = mdrtTuplesAllocate(bytes);
    memcpy(e->file, file, bytes);
    struct MdrtTuples *owner = NULL;
    for (size_t i = 0; i != store->count && !owner; ++i)
      if (sameSet(store->entries[i], file, tuples, arity, particles) &&
          store->entries[i]->ownsSet)
        owner = store->entries[i];
    if (owner) {
      e->deviceFile = owner->deviceFile;
      e->deviceFirst = owner->deviceFirst;
      e->deviceEntries = owner->deviceEntries;
      e->widest = owner->widest;
    } else {
      uploadSet(e);
      store->deviceBytes[2] +=
          (int64_t)(3 * bytes + (size_t)(particles + 1) * sizeof(int32_t));
    }
    e->width = e->widest * (arity + 2) + 1;
    e->deviceIncidence =
        tuplesDeviceKept((size_t)(particles * e->width) * sizeof(int32_t));
    store->deviceBytes[0] += particles * e->width * (int64_t)sizeof(int32_t);
    if (isMemoryDetailed())
      fprintf(stderr,
              "mdrt: device memory: the incidence structure of %lld tuples "
              "of %lld: %lld particles, rows of %lld entries (at most %lld "
              "tuples a particle), %lld bytes\n",
              (long long)tuples, (long long)arity, (long long)particles,
              (long long)e->width, (long long)e->widest,
              (long long)(particles * e->width * (int64_t)sizeof(int32_t)));
    if (e->ordered) {
      e->deviceRenumbered = tuplesDeviceKept(bytes);
      store->deviceBytes[1] += (int64_t)bytes;
    }
    if (store->count == store->room) {
      store->room = store->room ? 2 * store->room : 8;
      store->entries =
          realloc(store->entries, store->room * sizeof(*store->entries));
      if (!store->entries) {
        fprintf(stderr, "mdrt: out of memory of the host\n");
        abort();
      }
    }
    store->entries[store->count++] = e;
  }
  if (ordered && store->devicePlaceCount < particles) {
    if (store->devicePlace) {
      finish();
      check(cuMemFree((CUdeviceptr)store->devicePlace), "cuMemFree");
    }
    store->devicePlace = tuplesDeviceKept((size_t)particles * sizeof(int32_t));
    store->devicePlaceCount = particles;
    store->deviceBytes[3] += particles * (int64_t)sizeof(int32_t);
  }
  e->deviceValid = 0;
  e->building = 1;
  e->order = store->order;
  ++e->held;
  mdrtTuplesSpent(store, start);
  return (int64_t)(intptr_t)e;
}

int32_t mdrtDeviceTuplesBuild(int64_t handle) {
  return ((struct MdrtTuples *)(intptr_t)handle)->building;
}

void _mlir_ciface_mdrtDeviceTuplesFile(struct MdrtMemref2 *result,
                                       int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  mdrtTuplesSetMemref2(result, e->deviceFile, e->tuples, e->arity);
}

void _mlir_ciface_mdrtDeviceTuplesRenumbered(struct MdrtMemref2 *result,
                                             int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  mdrtTuplesSetMemref2(result,
                       e->ordered ? e->deviceRenumbered : e->deviceFile,
                       e->tuples, e->arity);
}

void _mlir_ciface_mdrtDeviceTuplesFirst(struct MdrtMemref1 *result,
                                        int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  setMemref1(result, e->deviceFirst, e->particles + 1);
}

void _mlir_ciface_mdrtDeviceTuplesEntries(struct MdrtMemref1 *result,
                                          int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  setMemref1(result, e->deviceEntries, 2 * e->tuples * e->arity);
}

void _mlir_ciface_mdrtDeviceTuplesPlace(struct MdrtMemref1 *result,
                                        int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  struct MdrtTupleStore *store = deviceTupleStore();
  setMemref1(result, e->ordered ? store->devicePlace : e->deviceFirst,
             e->ordered ? e->particles : 0);
}

void _mlir_ciface_mdrtDeviceTuplesStructure(struct MdrtMemref2 *result,
                                            int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  mdrtTuplesSetMemref2(result, e->deviceIncidence, e->particles, e->width);
}

/* The code has launched the kernels of the build, or taken the entry: it
   is valid from here on. */
void mdrtDeviceTuplesBuilt(int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  int built = e->building;
  e->building = 0;
  e->deviceValid = 1;
  if (built && mdrtTuplesChecks()) {
    enter();
    tuplesCheckOnDevice(deviceTupleStore(), e);
  }
}

/* Where the code would free what it took. */
void mdrtDeviceTuplesDone(int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  if (e->held > 0)
    --e->held;
}

/*===----------------------------------------------------------------------===
  Matrix products of the bodies of neural stages (D302)
  ===----------------------------------------------------------------------===*/

#include <cublas_v2.h>

/* One handle, on the stream of the kernels, in the order of their launches.
   The default math mode: f32 products are computed in f32, without TF32. */
static cublasHandle_t blasHandle = NULL;

static int checkBlas(cublasStatus_t status, const char *what) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  char message[160];
  snprintf(message, sizeof message,
           "%s failed with the cuBLAS status %d; the run has failed", what,
           (int)status);
  stopWith(message);
  return 0;
}

/* C = A B + beta C in f32, for row-major matrices: C is m x n with rows ldc
   apart, A is m x k (k x m with transA) with rows lda apart, and B is k x n
   (n x k with transB) with rows ldb apart; the pointers are the device's.
   With batch > 1, `batch` such products, the matrices of each the previous
   ones' moved by strideA, strideB, and strideC elements: cuBLAS's
   strided-batched GEMM. cuBLAS takes column-major matrices, as which a
   row-major one is its transpose: C^T = B^T A^T. */
void mdrtGemmF32(int64_t m, int64_t n, int64_t k, int64_t a, int64_t lda,
                 int64_t strideA, int32_t transA, int64_t b, int64_t ldb,
                 int64_t strideB, int32_t transB, int64_t c, int64_t ldc,
                 int64_t strideC, float beta, int64_t batch) {
  enter();
  double start = begin();
  if (!blasHandle) {
    int64_t before = beginAccount();
    if (!checkBlas(cublasCreate(&blasHandle), "cublasCreate"))
      return;
    endAccount(ACCOUNT_BLAS, before, 0);
    if (!checkBlas(cublasSetMathMode(blasHandle, CUBLAS_DEFAULT_MATH),
                   "cublasSetMathMode"))
      return;
  }
  if (!checkBlas(cublasSetStream(blasHandle, (cudaStream_t)mgpuStreamCreate()),
                 "cublasSetStream"))
    return;
  float one = 1.0f;
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (batch == 1)
    checkBlas(cublasSgemm(blasHandle, opB, opA, (int)n, (int)m, (int)k, &one,
                          (const float *)(intptr_t)b, (int)ldb,
                          (const float *)(intptr_t)a, (int)lda, &beta,
                          (float *)(intptr_t)c, (int)ldc),
              "cublasSgemm");
  else
    checkBlas(cublasSgemmStridedBatched(
                  blasHandle, opB, opA, (int)n, (int)m, (int)k, &one,
                  (const float *)(intptr_t)b, (int)ldb, (long long)strideB,
                  (const float *)(intptr_t)a, (int)lda, (long long)strideA,
                  &beta, (float *)(intptr_t)c, (int)ldc, (long long)strideC,
                  (int)batch),
              "cublasSgemmStridedBatched");
  isPending = 1;
  end(LAUNCH, start);
}

/* The constants that the products read (the weights of the bodies), copied
   to the device when first read, and found again by the address and size
   of their host copy and by their owner: the activation of the embedding
   program that made the copy (D215), or none outside an activation. An
   address of the host identifies a constant only while its program is
   loaded: after the program is released, another one may put its own
   constants there. So the copies go with their owner: an activation's when
   it closes, those made outside one when a program unloads its modules
   (mgpuModuleUnload), and all of them with the context
   (mgpuSetDefaultDevice). */
struct DeviceConstant {
  void *owner;
  int64_t host, bytes;
  CUdeviceptr device;
};
static struct DeviceConstant *deviceConstants = NULL;
static int64_t numDeviceConstants = 0, deviceConstantsCapacity = 0;

static void releaseDeviceConstants(void *owner, int all, int gone) {
  for (int64_t i = numDeviceConstants; i-- != 0;) {
    if (!all && deviceConstants[i].owner != owner)
      continue;
    if (!gone)
      check(cuMemFree(deviceConstants[i].device), "cuMemFree");
    deviceConstants[i] = deviceConstants[--numDeviceConstants];
  }
}

static void releaseBlas(int gone) {
  if (blasHandle && !gone)
    cublasDestroy(blasHandle);
  blasHandle = NULL;
}

int64_t mdrtDeviceConstant(int64_t host, int64_t bytes) {
  enter();
  void *owner = currentActivation;
  for (int64_t i = 0; i != numDeviceConstants; ++i)
    if (deviceConstants[i].owner == owner && deviceConstants[i].host == host &&
        deviceConstants[i].bytes == bytes)
      return (int64_t)deviceConstants[i].device;
  if (numDeviceConstants == deviceConstantsCapacity) {
    int64_t capacity =
        deviceConstantsCapacity ? 2 * deviceConstantsCapacity : 64;
    struct DeviceConstant *grown = realloc(
        deviceConstants, (size_t)capacity * sizeof(struct DeviceConstant));
    if (!grown) {
      stopWith("no memory of the host for the constants of a body's "
               "products; the run has failed");
      return 0;
    }
    deviceConstants = grown;
    deviceConstantsCapacity = capacity;
  }
  CUdeviceptr device = 0;
  int64_t before = beginAccount();
  check(cuMemAlloc(&device, (size_t)bytes), "cuMemAlloc");
  endAccount(ACCOUNT_CONSTANTS, before, bytes);
  check(cuMemcpyHtoD(device, (const void *)(intptr_t)host, (size_t)bytes),
        "cuMemcpyHtoD");
  /* From pageable memory the copy may still be under way when it returns,
     and the stream of the kernels does not wait for the null stream: the
     products that read the constant would race with its copy. */
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  struct DeviceConstant *entry = &deviceConstants[numDeviceConstants++];
  entry->owner = owner;
  entry->host = host;
  entry->bytes = bytes;
  entry->device = device;
  return (int64_t)device;
}
