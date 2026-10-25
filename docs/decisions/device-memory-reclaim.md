# D[device-memory-reclaim]

Recorded at its merge.

**The registry of the device's blocks grows with them, the idle blocks of the pool and the cached plans of cuFFT are bounded in bytes, a request that the driver refuses for want of memory is made again after they are given back, and an embedding program may give them back itself** (#622).

An external review checked the defects against the original functions with a mocked driver.

**Before.**
- The registry of the blocks in use (`liveBlocks`) held 4096 entries. Past that an allocation went on and set `untracked`. Closing an activation takes back the blocks that the registry records as its own, so a block past the limit stayed allocated for the rest of the process. The mocked run allocated 4097 blocks and closed the activation: 4097 driver allocations remained, and one was recorded nowhere.
- The pool reused a block only for a request of exactly its size. A miss went to `cuMemAlloc` with no retry, and the pool was bounded by a count of blocks (4096), not bytes. A long Python process that alternates systems of other sizes could run out of memory with most of the device idle in the pool.
- The plans of cuFFT were cached per grid, direction, and precision, and never destroyed.

**Now.** All of it is in `runtime/mdrt_cuda.c`.
- **The registry grows.** The lists of the blocks in use and of the idle blocks double as needed (`reserveBlock`). If the host has no memory to grow them, the new block is given back to the driver and the run stops with a message, so no block is ever handed out untracked. The bound of 4096 stays only for the copies of the blocks at the peaks that `MDIR_DEVICE_MEMORY=2` prints. `isOnDevice` no longer asks the driver about memory the registry might have missed, since it misses none.
- **The pool has a byte budget.** `MDIR_DEVICE_POOL_BYTES` defaults to half of the memory of the device. A freed block that would take the idle blocks past it goes back to the driver, after the device has run the work that may use it.
- **The plan cache has a byte budget.** `MDIR_FFT_PLAN_BYTES` defaults to a quarter of the device and bounds the workspaces of the cached plans, by `cufftGetSize`. Past it, the plans taken least recently are destroyed, after the device has run its work. A plan is made anew when its grid comes back.
- **A refused request is retried once.** When the driver reports `CUDA_ERROR_OUT_OF_MEMORY`, or cuFFT reports `CUFFT_ALLOC_FAILED` for a plan, `trimIdle` waits for the device, gives every idle block and every plan back, and the request is made once more. This covers every allocation of the runtime: the pool's blocks, the kept state, a plugin's outputs, the structures of the tuples, and the constants of the bodies.
- **An explicit trim.**
  - `mdrtDeviceTrim()` gives back the idle blocks and the plans and returns the bytes. Blocks in use and open activations stay.
  - `compiler::trimDeviceMemory()` calls it under the lock of the runs.
  - Python's `mdir.trim_device_memory()` calls that.
- **For tests.**
  - `mdrtDeviceMemoryCounts` reports the counts and bytes of the blocks in use and idle, the blocks of the pool's allocator that the driver holds, the plans and their bytes, and the driver's free memory.
  - `MDRT_FAIL_ALLOCATION=n` makes the n-th request report out-of-memory without asking the driver, so the retry is tested without filling a device that the other tests of a suite share.

**Defaults and the no-loss rule (#444).** The budgets are large enough that no run of the Amber suite reaches them. With no trim:
- the blocks, the reuse from the pool, and the returns to the driver are those of before;
- the rates are those of before.

The PR gives both measurements. A smaller budget is for a process that holds the device with other work.

**Not done.**
- A pool that serves a request with a larger idle block, or splits one, would change which memory a run takes and its timings. That is a separate decision.
- Trimming at the end of each simulation would cost a later simulation of the same sizes its reuse.

**Tests.**
- `test/Runtime/device-blocks-gpu.test`:
  - 5000 blocks in one activation, all idle after it closes, a second activation served from the pool with no new driver allocation, and a trim to zero;
  - a budget of 1 MiB that keeps 256 idle blocks of 4 KiB;
  - a request refused once (`MDRT_FAIL_ALLOCATION`) and made again after the trim.
- `test/Sanitizer/device-blocks-gpu.test` runs the same under compute-sanitizer memcheck with leak checking.
- `test/Driver/python-device-memory-gpu.test` alternates the dipeptide and the OPC water at PME grids from 24 to 40, in double and mixed precision, in one Python process under a budget of 64 MiB. No block stays in use, the idle bytes stay within the budget, and `mdir.trim_device_memory()` leaves no idle block, no plan, and no block of the pool held by the driver.
