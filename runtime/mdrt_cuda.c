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
#include <time.h>

/* A failure of the driver is reported and ends the run: compiled code has no
   way to continue without its device. */
static void check(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = NULL;
  cuGetErrorName(result, &name);
  fprintf(stderr, "mdrt: %s failed with %s\n", what,
          name ? name : "an unknown error");
  fflush(stderr);
  abort();
}

/*===----------------------------------------------------------------------===
  Profile
  ===----------------------------------------------------------------------===

  With MDRT_PROFILE set, the library counts its calls and the time they
  take, and reports when the program ends. With MDRT_WAIT set as well, it
  reports the time of each kernel: the time from the launch to the end of
  the kernel. */

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
enum { MAX_KERNELS = 256 };
struct Kernel {
  CUfunction function;
  char name[64];
  int64_t count;
  double seconds;
};
static struct Kernel kernels[MAX_KERNELS];
static int numKernels = 0;

/* The kernel that was launched last, and when; -1 if its time is
   accounted for. */
static int lastKernel = -1;
static double lastLaunch = 0.0;

static void report(void) {
  fprintf(stderr, "mdrt: %-40s %10s %12s %12s\n", "", "calls", "seconds",
          "microseconds");
  for (int i = 0; i != NUM_COUNTERS; ++i)
    fprintf(stderr, "mdrt: %-40s %10lld %12.4f %12.2f\n", counterNames[i],
            (long long)counts[i], seconds[i],
            counts[i] ? 1.0e6 * seconds[i] / (double)counts[i] : 0.0);
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].count)
      fprintf(stderr, "mdrt: %-40s %10lld %12.4f %12.2f\n", kernels[i].name,
              (long long)kernels[i].count, kernels[i].seconds,
              1.0e6 * kernels[i].seconds / (double)kernels[i].count);
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
   MDRT_WAIT set, the library waits wherever the lowering does, which
   tells an error of the order apart from other errors. */
static CUstream sharedStream = NULL;
static int waitsAlways = -1;

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

CUmodule mgpuModuleLoad(void *data, size_t size) {
  (void)size;
  enter();
  CUmodule module = NULL;
  check(cuModuleLoadData(&module, data), "cuModuleLoadData");
  return module;
}

/* Loads a module from PTX text, which the driver compiles. */
CUmodule mgpuModuleLoadJIT(void *data, int optLevel, size_t size) {
  (void)size;
  enter();
  char log[4096] = {0};
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *values[] = {(void *)log, (void *)(uintptr_t)sizeof(log),
                    (void *)(uintptr_t)optLevel};
  CUmodule module = NULL;
  CUresult result = cuModuleLoadDataEx(&module, data, 3, options, values);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "mdrt: the driver could not compile a kernel:\n%s\n",
            log);
  check(result, "cuModuleLoadDataEx");
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
  if (start >= 0.0)
    noteKernel(function, name);
  end(LOOKUP, start);
  return function;
}

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
  check(cuLaunchKernel(function, (unsigned)gridX, (unsigned)gridY,
                       (unsigned)gridZ, (unsigned)blockX, (unsigned)blockY,
                       (unsigned)blockZ, (unsigned)sharedMemory, stream,
                       parameters, extra),
        "cuLaunchKernel");
  isPending = 1;
  if (start >= 0.0) {
    lastKernel = findKernel(function);
    lastLaunch = start;
  }
  end(LAUNCH, start);
}

/*===----------------------------------------------------------------------===
  Streams and events
  ===----------------------------------------------------------------------===*/

CUstream mgpuStreamCreate(void) {
  enter();
  if (!sharedStream)
    check(cuStreamCreate(&sharedStream, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate");
  return sharedStream;
}

void mgpuStreamDestroy(CUstream stream) { (void)stream; }

/* Waits until the device has run everything that was issued. */
static void finish(void) {
  if (!isPending || !sharedStream)
    return;
  double start = begin();
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

void *mgpuMemAlloc(uint64_t size, CUstream stream, bool isHostShared) {
  (void)stream;
  enter();
  CUdeviceptr pointer = 0;
  if (size == 0)
    return NULL;
  double start = begin();
  if (isHostShared)
    check(cuMemAllocManaged(&pointer, size, CU_MEM_ATTACH_GLOBAL),
          "cuMemAllocManaged");
  else
    check(cuMemAlloc(&pointer, size), "cuMemAlloc");
  end(ALLOCATE, start);
  return (void *)pointer;
}

void mgpuMemFree(void *pointer, CUstream stream) {
  (void)stream;
  if (!pointer)
    return;
  /* Work that was issued may use the memory. */
  finish();
  double start = begin();
  check(cuMemFree((CUdeviceptr)pointer), "cuMemFree");
  end(ALLOCATE, start);
}

/* Returns true if `pointer` is memory of the device. The driver knows the
   memory that it has allocated, and nothing about memory of the host. */
static bool isOnDevice(void *pointer) {
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
