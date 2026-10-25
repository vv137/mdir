### Fixed

- On a GPU, a run with more than 4096 device blocks alive at once no longer
  leaves memory allocated after it ends. The memory that the runtime keeps
  for reuse is now bounded in bytes (`MDIR_DEVICE_POOL_BYTES`,
  `MDIR_FFT_PLAN_BYTES`). It is given back to the driver when an
  allocation would otherwise run out of memory, or by
  `mdir.trim_device_memory()` (D[device-memory-reclaim], #622).
