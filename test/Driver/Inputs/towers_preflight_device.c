// Device limits for CPU preflight. A program without a force-only region
// needs no CUDA context, compiled trial kernel, or occupancy query.
#include <cuda.h>

CUresult cuInit(unsigned flags) { return CUDA_SUCCESS; }

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  *device = ordinal;
  return CUDA_SUCCESS;
}

#undef cuDeviceGetUuid
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice device) {
  *uuid = (CUuuid){0};
  return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *version) {
  *version = CUDA_VERSION;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attribute,
                              CUdevice device) {
  switch (attribute) {
  case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *value = 8; break;
  case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *value = 9; break;
  case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: *value = 101376; break;
  case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: *value = 102400; break;
  case CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: *value = 1024; break;
  case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *value = 1024; break;
  case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: *value = 1536; break;
  case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: *value = 24; break;
  default: return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}
