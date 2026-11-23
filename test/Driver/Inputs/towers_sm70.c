// Model a visible Volta device for the mdir check preflight.
#include <cuda.h>

CUresult cuInit(unsigned flags) { return CUDA_SUCCESS; }

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attribute,
                              CUdevice device) {
  if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    *value = 7;
  else if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    *value = 0;
  else
    return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}
