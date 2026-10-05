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
  }
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  if (context) {
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
  CUresult result = cuModuleLoadData(&module, data);
  if (result != CUDA_SUCCESS)
    reportUnsupportedPTX(result, data, size, NULL);
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
  CUresult result = cuModuleLoadDataEx(&module, data, 3, options, values);
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
  check(cuLaunchKernel(function, (unsigned)gridX, (unsigned)gridY,
                       (unsigned)gridZ, (unsigned)blockX, (unsigned)blockY,
                       (unsigned)blockZ, (unsigned)sharedMemory, stream,
                       parameters, extra),
        "cuLaunchKernel");
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
   requests of the same size. */
#define MAX_BLOCKS 4096
struct Block {
  CUdeviceptr pointer;
  uint64_t size;
  /* Whether a call of an entry that an embedding program opened
     (mdrtDeviceBeginCall) allocated the block; it returns to the pool when
     the call closes (#110). */
  int ofCall;
};
static int inCall = 0;
/* Memory shared with the host that a call allocated, freed when it closes. */
#define MAX_CALL_MANAGED 256
static CUdeviceptr callManaged[MAX_CALL_MANAGED];
static int numCallManaged = 0;
static struct Block liveBlocks[MAX_BLOCKS];
static int numLive = 0;
static struct Block freeBlocks[MAX_BLOCKS];
static int numFree = 0;
/* Whether memory of the device was allocated that `liveBlocks` does not
   record, for want of room. */
static int untracked = 0;

void *mgpuMemAlloc(uint64_t size, CUstream stream, bool isHostShared) {
  (void)stream;
  enter();
  CUdeviceptr pointer = 0;
  if (size == 0)
    return NULL;
  double start = begin();
  if (isHostShared) {
    check(cuMemAllocManaged(&pointer, size, CU_MEM_ATTACH_GLOBAL),
          "cuMemAllocManaged");
    if (inCall && numCallManaged < MAX_CALL_MANAGED)
      callManaged[numCallManaged++] = pointer;
    end(ALLOCATE, start);
    return (void *)pointer;
  }
  for (int i = numFree; i-- != 0;) {
    if (freeBlocks[i].size != size)
      continue;
    pointer = freeBlocks[i].pointer;
    freeBlocks[i] = freeBlocks[--numFree];
    break;
  }
  if (!pointer)
    check(cuMemAlloc(&pointer, size), "cuMemAlloc");
  if (numLive < MAX_BLOCKS)
    liveBlocks[numLive++] = (struct Block){pointer, size, inCall};
  else
    untracked = 1;
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
    if (numFree < MAX_BLOCKS) {
      freeBlocks[numFree++] = block;
      end(ALLOCATE, start);
      return;
    }
    break;
  }
  /* Memory that is not recorded, or that the pool has no room for, goes
     back to the driver once the work that may use it is done. */
  finish();
  check(cuMemFree(address), "cuMemFree");
  end(ALLOCATE, start);
}

/* The structures of neighbors (groups and matrices) that a call made: their
   host records are freed when it closes, and their memory of the device
   returns to the pool with the call's other blocks. */
static void **callHandles = NULL;
static size_t numCallHandles = 0, roomCallHandles = 0;

static void recordCallHandle(void *handle) {
  if (!inCall)
    return;
  if (numCallHandles == roomCallHandles) {
    roomCallHandles = roomCallHandles ? 2 * roomCallHandles : 8;
    callHandles = realloc(callHandles, roomCallHandles * sizeof(void *));
    if (!callHandles) {
      fprintf(stderr, "mdrt: out of memory of the host\n");
      abort();
    }
  }
  callHandles[numCallHandles++] = handle;
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

enum { NUM_FFT_PLANS = 8 };
static struct {
  int64_t k1, k2, k3;
  int forward, single;
  cufftHandle plan;
} fftPlans[NUM_FFT_PLANS];
static int numFFTPlans = 0;

static void checkFFT(cufftResult result, const char *what) {
  if (result == CUFFT_SUCCESS)
    return;
  fprintf(stderr, "mdrt: %s failed with the cuFFT error %d\n", what,
          (int)result);
  fflush(stderr);
  abort();
}

/* The plan of a grid, a direction, and a precision (f32 if `single`). */
static cufftHandle getFFTPlan(int64_t k1, int64_t k2, int64_t k3,
                              int forward, int single) {
  for (int i = 0; i != numFFTPlans; ++i)
    if (fftPlans[i].k1 == k1 && fftPlans[i].k2 == k2 &&
        fftPlans[i].k3 == k3 && fftPlans[i].forward == forward &&
        fftPlans[i].single == single)
      return fftPlans[i].plan;
  if (numFFTPlans == NUM_FFT_PLANS) {
    fprintf(stderr, "mdrt: too many grids of FFT\n");
    abort();
  }
  cufftHandle plan;
  cufftType type = single ? (forward ? CUFFT_R2C : CUFFT_C2R)
                          : (forward ? CUFFT_D2Z : CUFFT_Z2D);
  checkFFT(cufftPlan3d(&plan, (int)k1, (int)k2, (int)k3, type),
           "cufftPlan3d");
  checkFFT(cufftSetStream(plan, (cudaStream_t)mgpuStreamCreate()),
           "cufftSetStream");
  fftPlans[numFFTPlans].k1 = k1;
  fftPlans[numFFTPlans].k2 = k2;
  fftPlans[numFFTPlans].k3 = k3;
  fftPlans[numFFTPlans].forward = forward;
  fftPlans[numFFTPlans].single = single;
  fftPlans[numFFTPlans].plan = plan;
  ++numFFTPlans;
  return plan;
}

/* The forward transform of the real grid of k1 x k2 x k3 points into the
   half-complex grid of k1 x k2 x (k3 / 2 + 1) numbers, interleaved, as
   mdrtFFTForward3D of the host computes it. */
void _mlir_ciface_mdrtCudaFFTForward3D(DeviceBuffer1D *real,
                                       DeviceBuffer1D *complex, int64_t k1,
                                       int64_t k2, int64_t k3) {
  enter();
  double start = begin();
  checkFFT(cufftSetStream(getFFTPlan(k1, k2, k3, 1, 0),
                          (cudaStream_t)mgpuStreamCreate()),
           "cufftSetStream");
  checkFFT(cufftExecD2Z(getFFTPlan(k1, k2, k3, 1, 0),
                        (cufftDoubleReal *)(real->aligned + real->offset),
                        (cufftDoubleComplex *)(complex->aligned +
                                               complex->offset)),
           "cufftExecD2Z");
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
  checkFFT(cufftSetStream(getFFTPlan(k1, k2, k3, 0, 0),
                          (cudaStream_t)mgpuStreamCreate()),
           "cufftSetStream");
  checkFFT(cufftExecZ2D(getFFTPlan(k1, k2, k3, 0, 0),
                        (cufftDoubleComplex *)(complex->aligned +
                                               complex->offset),
                        (cufftDoubleReal *)(real->aligned + real->offset)),
           "cufftExecZ2D");
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
  checkFFT(cufftSetStream(getFFTPlan(k1, k2, k3, 1, 1),
                          (cudaStream_t)mgpuStreamCreate()),
           "cufftSetStream");
  checkFFT(cufftExecR2C(getFFTPlan(k1, k2, k3, 1, 1),
                        (cufftReal *)(real->aligned + real->offset),
                        (cufftComplex *)(complex->aligned + complex->offset)),
           "cufftExecR2C");
  isPending = 1;
  end(LAUNCH, start);
}

void _mlir_ciface_mdrtCudaFFTBackward3DF32(DeviceBuffer1DF32 *complex,
                                           DeviceBuffer1DF32 *real,
                                           int64_t k1, int64_t k2,
                                           int64_t k3) {
  enter();
  double start = begin();
  checkFFT(cufftSetStream(getFFTPlan(k1, k2, k3, 0, 1),
                          (cudaStream_t)mgpuStreamCreate()),
           "cufftSetStream");
  checkFFT(cufftExecC2R(getFFTPlan(k1, k2, k3, 0, 1),
                        (cufftComplex *)(complex->aligned + complex->offset),
                        (cufftReal *)(real->aligned + real->offset)),
           "cufftExecC2R");
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
   first asks for them. Compiled code takes a buffer where it uses it
   (`mdrtGroupsBuffer`), so a buffer that grew is the one it takes. */
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
  GROUPS_BUFFERS
};

struct Groups {
  void *data[GROUPS_BUFFERS];
  int64_t places;
  int64_t blocks;
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
  default:
    return groups->blocks;
  }
}

static void allocateGroup(struct Groups *groups, int which) {
  groups->data[which] = mgpuMemAlloc(
      (uint64_t)getGroupsLength(groups, which) * sizeof(int32_t), NULL,
      false);
}

/* The buffers of the outer list; those of the inner one wait for their
   first use. */
static void allocateGroups(struct Groups *groups) {
  for (int which = 0; which != GROUPS_INNER_ENTRIES; ++which)
    allocateGroup(groups, which);
}

static void freeGroups(struct Groups *groups) {
  for (int which = 0; which != GROUPS_BUFFERS; ++which) {
    if (groups->data[which])
      mgpuMemFree(groups->data[which], NULL);
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
  freeGroups(groups);
  if (places > groups->places)
    groups->places = (places + places / 4 + 15) / 16 * 16;
  if (blocks > groups->blocks)
    groups->blocks = blocks + blocks / 4;
  allocateGroups(groups);
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

/* A memref of rank 2 as the C interface of MLIR passes it. */
struct Buffer2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

static void allocateMatrix(struct Matrix *matrix) {
  matrix->data = mgpuMemAlloc(
      (uint64_t)(matrix->rows * matrix->width) * sizeof(int32_t), NULL,
      false);
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
  mgpuMemFree(matrix->data, NULL);
  matrix->width = width + width / 4;
  allocateMatrix(matrix);
}

/*===----------------------------------------------------------------------===
 * Calls of an entry by an embedding program
 *===----------------------------------------------------------------------===*/

/* A program that calls an entry more than once (a Python simulation, D196)
   opens a call before each and closes it after. What the call allocated is
   its own: its blocks return to the pool, where the next call takes the
   blocks of the same sizes, so the memory of the device stays that of one
   call (#110). `mdir run` calls its entry once and opens none. */
void mdrtDeviceBeginCall(void) { inCall = 1; }

void mdrtDeviceEndCall(void) {
  if (!inCall)
    return;
  /* No work of the call may still use a block that is reused or freed. */
  finish();
  for (int i = numLive; i-- != 0;) {
    if (!liveBlocks[i].ofCall)
      continue;
    struct Block block = liveBlocks[i];
    liveBlocks[i] = liveBlocks[--numLive];
    block.ofCall = 0;
    if (numFree < MAX_BLOCKS)
      freeBlocks[numFree++] = block;
    else
      check(cuMemFree(block.pointer), "cuMemFree");
  }
  for (int i = 0; i != numCallManaged; ++i)
    check(cuMemFree(callManaged[i]), "cuMemFree");
  numCallManaged = 0;
  for (size_t i = 0; i != numCallHandles; ++i)
    free(callHandles[i]);
  numCallHandles = 0;
  inCall = 0;
}
