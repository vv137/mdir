// Device layout shared by the MATRIX producer, pair kernels, and runtime.
#ifndef MDIR_RUNTIME_MATRIXSPILL_H
#define MDIR_RUNTIME_MATRIXSPILL_H

#define MDRT_MATRIX_SPILL_ABI_VERSION 1

// Records and sorted neighbors start at index one. Offsets and lengths
// are indexed by matrix row starting at zero. Row zero's offset doubles
// as the device predicate: one for a sorted tail, zero otherwise.
enum {
  MATRIX_SPILL_ROW,
  MATRIX_SPILL_NEIGHBOR,
  MATRIX_SPILL_RANK,
  MATRIX_SPILL_SORTED,
  MATRIX_SPILL_OFFSET,
  MATRIX_SPILL_LENGTH,
  MATRIX_SPILL_COLUMNS
};

// The first four words of record zero are transferred together. The first
// two words form the aligned 64-bit reservation counter.
enum { MATRIX_REQUIRED_WIDTH = 2, MATRIX_INVALID_POSITIONS = 3 };

#endif
