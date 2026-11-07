/* Positive controls for measured device-frame host transfer counts. */
#include "mdir/Runtime/CopyAudit.h"
#include <cuda.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
extern CUstream mgpuStreamCreate(void);
extern void mgpuMemcpy(void *, void *, size_t, CUstream);
extern void mdrtCudaMemcpyScoped(uintptr_t, uintptr_t, uint64_t);
extern void mdrtDeviceCopyToHost(void *, const void *, uint64_t);
extern void *mdrtDeviceAuditCopies(void *);
extern void mdrtDeviceAuditMetadataCopy(uintptr_t);
int main(void) {
  CUstream stream = mgpuStreamCreate();
  CUdeviceptr device;
  if (cuMemAlloc(&device, 64) != CUDA_SUCCESS) abort();
  struct MdirDeviceCopyAudit audit = {0}, nested = {0};
  audit.ranges = 1;
  audit.first[0] = device;
  audit.last[0] = device + 64;
  int source[16], result[16];
  for (int i = 0; i != 16; ++i) source[i] = 3 * i - 7;
  void *previous = mdrtDeviceAuditCopies(&audit);
  mgpuMemcpy((void *)device, source, 64, stream);
  mdrtCudaMemcpyScoped((uintptr_t)result, device, 64);
  mdrtDeviceCopyToHost(result, (void *)device, 64);
  mdrtDeviceAuditMetadataCopy((uintptr_t)result);
  mgpuMemcpy(result, (void *)device, 64, stream);
  void *outer = mdrtDeviceAuditCopies(&nested);
  mdrtDeviceCopyToHost(result, (void *)device, 64);
  if (mdrtDeviceAuditCopies(outer) != &nested) abort();
  mdrtDeviceCopyToHost(result, (void *)device, 64);
  if (mdrtDeviceAuditCopies(previous) != &audit) abort();
  if (audit.uploads != 64 || audit.downloads != 64 ||
      audit.metadataUploads != 0 || audit.metadataDownloads != 192 ||
      nested.uploads != 0 || nested.downloads != 0 || nested.metadataDownloads != 64)
    abort();
  for (int i = 0; i != 16; ++i) if (source[i] != result[i]) abort();
  if (cuMemFree(device) != CUDA_SUCCESS) abort();
  puts("Copy audit: 64 upload bytes, 64 download bytes; nested metadata scopes exact");
  return 0;
}
