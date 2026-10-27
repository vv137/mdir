/* The scoped copy must complete without completing unrelated side work.
   A device-side stream wait holds that work until the host releases it,
   so the proof does not depend on the duration of a kernel. */
#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern CUstream mgpuStreamCreate(void);
extern void *mgpuMemAlloc(uint64_t, CUstream, bool);
extern void mgpuMemFree(void *, CUstream);
extern void mgpuMemcpy(void *, void *, size_t, CUstream);
extern void mdrtCudaMemcpyScoped(uintptr_t, uintptr_t, uint64_t);
extern void mdrtSideBegin(void);
extern void mdrtSideEnd(void);
extern void mdrtSideJoin(void);

static void check(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS)
    return;
  const char *name = NULL;
  cuGetErrorName(status, &name);
  fprintf(stderr, "%s: %s\n", operation, name ? name : "unknown CUDA error");
  abort();
}

int main(int argc, char **argv) {
  (void)argv;
  CUstream mainStream = mgpuStreamCreate();
  int32_t source[16], copied[16];
  for (unsigned i = 0; i < 16; ++i)
    source[i] = 17 * (int32_t)i - 31;
  void *device = mgpuMemAlloc(sizeof(source), mainStream, false);
  CUdeviceptr release;
  check(cuMemAlloc(&release, sizeof(uint32_t)), "allocate release flag");
  check(cuMemsetD32Async(release, 0, 1, mainStream), "clear release flag");
  check(cuStreamSynchronize(mainStream), "complete test setup");

  mdrtSideBegin();
  CUstream sideStream = mgpuStreamCreate();
  check(cuStreamWaitValue32(sideStream, release, 1, CU_STREAM_WAIT_VALUE_EQ),
        "hold unrelated side work");
  CUevent sideCompleted;
  check(cuEventCreate(&sideCompleted, CU_EVENT_DISABLE_TIMING), "create event");
  check(cuEventRecord(sideCompleted, sideStream), "record side completion");
  mdrtSideEnd();

  /* The optional argument is a manual negative control: the legacy copy
     waits for the held side stream and must be stopped by the caller's
     timeout. The normal regression never selects it. */
  if (argc > 1)
    mgpuMemcpy(device, source, sizeof(source), mainStream);
  else
    mdrtCudaMemcpyScoped((uintptr_t)device, (uintptr_t)source, sizeof(source));
  for (unsigned i = 0; i < 16; ++i)
    source[i] = 0;
  mdrtCudaMemcpyScoped((uintptr_t)copied, (uintptr_t)device, sizeof(copied));
  for (unsigned i = 0; i < 16; ++i)
    if (copied[i] != 17 * (int32_t)i - 31) {
      fprintf(stderr, "copy mismatch at %u\n", i);
      abort();
    }
  if (cuEventQuery(sideCompleted) != CUDA_ERROR_NOT_READY) {
    fprintf(stderr, "unrelated side work completed before its release\n");
    abort();
  }
  mdrtCudaMemcpyScoped(0, 0, 0);
  check(cuStreamWriteValue32(mainStream, release, 1, 0), "release side work");
  check(cuEventSynchronize(sideCompleted), "complete side work");
  mdrtSideJoin();
  check(cuStreamSynchronize(mainStream), "complete test cleanup");
  check(cuEventDestroy(sideCompleted), "destroy event");
  check(cuMemFree(release), "free release flag");
  mgpuMemFree(device, mainStream);
  puts("Scoped copies: 16 exact integers; difference 0, tolerance 0");
  puts("Unrelated side work stayed pending until explicit release");
  return 0;
}
