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

static int32_t defaultDevice = 0;
static CUcontext context = NULL;

/* One stream serves all launches. The lowering asks for a stream per launch
   and waits for it, so the launches never overlap. */
static CUstream sharedStream = NULL;

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
  CUresult result = cuModuleUnload(module);
  if (result != CUDA_ERROR_DEINITIALIZED &&
      result != CUDA_ERROR_INVALID_CONTEXT &&
      result != CUDA_ERROR_CONTEXT_IS_DESTROYED)
    check(result, "cuModuleUnload");
}

CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = NULL;
  check(cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
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
  check(cuLaunchKernel(function, (unsigned)gridX, (unsigned)gridY,
                       (unsigned)gridZ, (unsigned)blockX, (unsigned)blockY,
                       (unsigned)blockZ, (unsigned)sharedMemory, stream,
                       parameters, extra),
        "cuLaunchKernel");
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

void mgpuStreamSynchronize(CUstream stream) {
  check(cuStreamSynchronize(stream), "cuStreamSynchronize");
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
  if (isHostShared)
    check(cuMemAllocManaged(&pointer, size, CU_MEM_ATTACH_GLOBAL),
          "cuMemAllocManaged");
  else
    check(cuMemAlloc(&pointer, size), "cuMemAlloc");
  return (void *)pointer;
}

void mgpuMemFree(void *pointer, CUstream stream) {
  (void)stream;
  if (pointer)
    check(cuMemFree((CUdeviceptr)pointer), "cuMemFree");
}

/* Copies between the host and the device, in either direction: the driver
   tells them apart by the address. */
void mgpuMemcpy(void *destination, void *source, size_t size,
                CUstream stream) {
  if (size == 0)
    return;
  check(cuMemcpyAsync((CUdeviceptr)destination, (CUdeviceptr)source, size,
                      stream),
        "cuMemcpyAsync");
}

void mgpuMemset32(void *destination, unsigned int value, size_t count,
                  CUstream stream) {
  check(cuMemsetD32Async((CUdeviceptr)destination, value, count, stream),
        "cuMemsetD32Async");
}

void mgpuMemset16(void *destination, unsigned short value, size_t count,
                  CUstream stream) {
  check(cuMemsetD16Async((CUdeviceptr)destination, value, count, stream),
        "cuMemsetD16Async");
}
