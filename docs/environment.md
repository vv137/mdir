# Environment variables

The environment variables that MDIR reads, in one place. Each row names the
document that describes the variable in full. None is needed for an ordinary
run. Control files, the Python API, and lit parameters remain the way to
choose what a run does; these variables are for the machine, for debugging,
and for tests.

## Devices and toolkits

| Variable | Read by | Effect |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | the CUDA driver; the test suite; the compiler of a GPU program | The GPUs that a process may use. An empty value hides every GPU, and the suite then skips the tests that need one ([README](../README.md#testing)). The compiler asks NVML for the architecture of the device it selects (D[gpu-module-compile]). |
| `MDRT_DEVICE=<n>` | the device runtime (`runtime/mdrt_cuda.c`); the compiler of a GPU program | The device, among the visible ones, that a run uses; the first by default ([mdrt-m0.md](mdrt-m0.md)). The kernels are compiled for its architecture (D[gpu-module-compile]). |
| `CUDA_ROOT`, `CUDA_HOME`, `CUDA_PATH` | the compiler of a GPU program | The CUDA toolkit whose libdevice the kernels take their math functions from, and whose `ptxas` compiles them to cubins, the first of the three that is set; otherwise the toolkit of the build ([mdrt-m0.md](mdrt-m0.md), [compile-cache.md](compile-cache.md#cubins-for-the-device)). |
| `MDIR_RUNTIME_DIR=<dir>` | a Python simulation | The directory of the runtime libraries (`libmdrt`, `libmdrt_cuda`); otherwise `lib` next to the Python module, as a build tree and an installation place it ([python-segments.md](python-segments.md)). |

## Compilation

| Variable | Read by | Effect |
|---|---|---|
| `MDIR_COMPILE_THREADS=<n>` | every lowering (`mdir run`, `mdir.compile`, a Python simulation) | Bounds the threads of the process-wide pool that lowers programs; as many as the host has cores by default (D211). The test suite sets it to the cores over `gpu_workers` ([python-segments.md](python-segments.md)). |
| `MDIR_COMPILE_CACHE_DIR=<dir>` | the JIT of a Python simulation (host objects); every lowering of a GPU program (`mdir run`, `mdir.compile`, a Python simulation) for the PTX and cubins of its GPU modules | Enables the compile cache and puts its entries in `<dir>/host/` and `<dir>/gpu/` (D212, D[gpu-module-compile], [compile-cache.md](compile-cache.md)). |
| `MDIR_COMPILE_CACHE=off` | the same | Disables the cache even when a directory is set. |
| `MDIR_COMPILE_CACHE_MAX_MB=<n>` | the same | Bounds the cache directory, host and GPU entries together, to $n$ MiB, 2048 by default, removing the entries used least recently. |
| `MDIR_GPU_BINARY=auto\|cubin\|ptx` | every lowering of a GPU program (`mdir run`, `mdir.compile`, a Python simulation) | The binaries of the kernels (D[gpu-module-compile]). `auto`, the default, makes cubins for the device's architecture with the toolkit's `ptxas`, and PTX where it cannot. `cubin` makes a missing architecture or `ptxas` an error. `ptx` keeps PTX, which the driver compiles at load. See [compile-cache.md](compile-cache.md#cubins-for-the-device). |
| `MDIR_GPU_ARCH=sm_XY` | the same | The architecture to compile the kernels for, instead of the one that NVML reports for the device (D[gpu-module-compile], [compile-cache.md](compile-cache.md#cubins-for-the-device)). |

## Debugging

These are described in [debugging.md](debugging.md). They act on `mdir run`.

| Variable | Effect |
|---|---|
| `MDIR_PIPELINE=<passes>` | Replaces the pass pipeline, to try another order of passes or to stop part of the way. |
| `MDIR_PRINT_AFTER=<pass>` | Prints the module after each run of that pass, for example `gpu-kernel-outlining` for the kernels. |
| `MDIR_REPRODUCER=<file>` | Where a failing pass writes the module and pipeline that `mdir-opt --run-reproducer` repeats; `mdir-reproducer.mlir` by default. `mdir bug-report` sets it. |
| `MDRT_TRACE` (set) | The device runtime prints each module that it loads, its kernels, and each launch with its grid. |
| `MDRT_PROFILE` (set) | The device runtime counts its calls and their time, and reports them at the end. |
| `MDRT_WAIT=1` | The device runtime waits for each launch. A failing kernel is then reported at its own launch, and with `MDRT_PROFILE`, each kernel is timed. |

## Tests and validation

| Variable | Read by | Effect |
|---|---|---|
| `MDIR_BENCH_DIR=<dir>` | `test/Scale`, `scripts/benchmarks/amber` | The prepared Amber benchmark suite. Without it, those tests are unsupported ([debugging.md](debugging.md)). |
| `CHARMM_TOPPAR=<dir>` | `scripts/validation/charmm/run.py` | The CHARMM topology and parameter files of the validation against CHARMM ([charmm-m1.md](charmm-m1.md)). |

The suite also takes lit parameters, given as `-D<name>=<value>` in
`LIT_OPTS` (`test/mdir_lit.py`):

- `gpu_workers` sets how many tests that need the GPU run at once (16 by default, D208);
- `compile_threads` sets the `MDIR_COMPILE_THREADS` of each test;
- `compile_cache=off` runs the suite without its compile cache.

A change that adds or reads an environment variable adds its row here.
