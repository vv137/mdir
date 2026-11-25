#ifndef MDIR_RUNTIME_HOST_MATRIX_H
#define MDIR_RUNTIME_HOST_MATRIX_H
#include <stdint.h>

struct HostBuffer2 {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

/* Matrices owned by a halo are released with that halo, independently of
   the lifetime of the activation that executes its pair kernels. */
int64_t mdrtHostMatrixAllocate(int64_t rows, int64_t width);
void mdrtHostMatrixRelease(int64_t handle);
void mdrtHostMatrixGrow(int64_t handle, int64_t width);
void _mlir_ciface_mdrtHostMatrixEntries(struct HostBuffer2 *result,
                                        int64_t handle);
#endif
