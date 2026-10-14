# Environment variables

The environment variables that MDIR reads, in one place. Each row names the
document that describes the variable in full. None is needed for an ordinary
run. Control files, the Python API, and lit parameters remain the way to
choose what a run does; these variables are for the machine, for debugging,
and for tests.

## Devices and toolkits

| Variable | Read by | Effect |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | the CUDA driver; the test suite; the compiler of a GPU program | The GPUs that a process may use. An empty value hides every GPU, and the suite then skips the tests that need one ([README](../README.md#testing)). The compiler asks NVML for the architecture of the device it selects (D214). |
| `CUDA_CACHE_PATH=<dir>` | the CUDA driver; the test suite forwards it | Where the driver keeps the kernels that it compiles from PTX (its default is in the home directory). Set it for a suite whose home directory is on a network file system; the suite hands it to its tests (#223). Temporary files of the tests are in a directory that lit makes for the run under the `TMPDIR` that lit was given. |
| `MDRT_DEVICE=<n>` | the device runtime (`runtime/mdrt_cuda.c`); the compiler of a GPU program | The device, among the visible ones, that a run uses; the first by default ([mdrt-m0.md](mdrt-m0.md)). The kernels are compiled for its architecture (D214). |
| `CUDA_ROOT`, `CUDA_HOME`, `CUDA_PATH` | the compiler of a GPU program | The CUDA toolkit whose libdevice the kernels take their math functions from, and whose `ptxas` compiles them to cubins, the first of the three that is set; otherwise the toolkit of the build ([mdrt-m0.md](mdrt-m0.md), [compile-cache.md](compile-cache.md#cubins-for-the-device)). The three are also part of the key of the code that a `Program` keeps for its simulations on a GPU (D236). |
| `MDIR_RUNTIME_DIR=<dir>` | a Python simulation | The directory of the runtime libraries (`libmdrt`, `libmdrt_cuda`); otherwise `lib` next to the Python module, as a build tree and an installation place it ([python-segments.md](python-segments.md)). |

## Compilation

| Variable | Read by | Effect |
|---|---|---|
| `MDIR_COMPILE_THREADS=<n>` | every lowering (`mdir run`, `mdir.compile`, a Python simulation) | Bounds the threads of the process-wide pool that lowers programs; as many as the host has cores by default (D211). The test suite sets it to the cores over `gpu_workers` ([python-segments.md](python-segments.md)). |
| `MDIR_COMPILE_CACHE_DIR=<dir>` | the JIT of a Python simulation (host objects); every lowering of a GPU program (`mdir run`, `mdir.compile`, a Python simulation) for the PTX and cubins of its GPU modules | Enables the compile cache and puts its entries in `<dir>/host/` and `<dir>/gpu/` (D212, D214, [compile-cache.md](compile-cache.md)). `mdir.clear_compile_cache()` clears this directory when it is given none, even under `MDIR_COMPILE_CACHE=off`; `cache=False` on `mdir.compile` or `Simulation` bypasses it for one compile (D217). |
| `MDIR_COMPILE_CACHE=off` | the same | Disables the cache even when a directory is set. |
| `MDIR_COMPILE_CACHE_MAX_MB=<n>` | the same | Bounds the cache directory, host and GPU entries together, to $n$ MiB, 2048 by default, removing the entries used least recently when a store takes the total, kept in `<dir>/size`, over the bound (D234). |
| `MDIR_GPU_BINARY=auto\|cubin\|ptx` | every lowering of a GPU program (`mdir run`, `mdir.compile`, a Python simulation) | The binaries of the kernels (D214). `auto`, the default, makes cubins for the device's architecture with the toolkit's `ptxas`, and PTX where it cannot. `cubin` makes a missing architecture or `ptxas` an error. `ptx` keeps PTX, which the driver compiles at load. See [compile-cache.md](compile-cache.md#cubins-for-the-device). |
| `MDIR_GPU_ARCH=sm_XY` | the same | The architecture to compile the kernels for, instead of the one that NVML reports for the device (D214, [compile-cache.md](compile-cache.md#cubins-for-the-device)). |

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
| `MDRT_CHECK_KEPT_TUPLES` (set) | The runtimes (`runtime/mdrt.c`, `runtime/mdrt_cuda.c`) build anew every structure of the tuples that a simulation takes from those it keeps across its activations (D253), or a segment of `mdir run` from those of the segment before (D[run-tuples-memory]; the log of such a run then says at its end how many structures were built and how many taken), and compare the two, on a device after reading the kept one back; a difference ends the program with a message. A start then costs what it cost without the kept structures, and more ([jit-invariants.md](jit-invariants.md#what-a-simulation-keeps-across-activations-d253)). |

## Tests and validation

| Variable | Read by | Effect |
|---|---|---|
| `MDIR_BENCH_DIR=<dir>` | `test/Scale`, `scripts/benchmarks/amber` | The prepared Amber benchmark suite. Without it, those tests are unsupported ([debugging.md](debugging.md)). |
| `CHARMM_TOPPAR=<dir>` | `scripts/validation/charmm/run.py` | The CHARMM topology and parameter files of the validation against CHARMM ([charmm-m1.md](charmm-m1.md)). |

## Scripts

The scripts of the repository name no path of one machine (#289;
`test/Lit/script-paths.test` keeps it so). What a machine provides comes
from the environment, by one convention (`scripts/environment.sh`):

- A **program** is a variable that holds its name or its path. Its default
  is the bare name, looked up on the path.
- A **directory or a choice of the machine** (an install prefix, where to
  write, a device) is a variable without a default.
- A script checks all that it needs at its start, before it fetches,
  removes, or writes anything. It prints every variable that is missing, or
  that names nothing, with one line on what it is, and exits with status 2
  (`test/Lit/script-environment.test`).
- A machine may keep its values in one file outside the repository. If
  `MDIR_ENV_FILE` is set, the shell scripts below (but the argon example,
  which is installed without the helper) read the lines `NAME=value` of
  that file: the value is the rest of the line as it is (no quotes, no
  expansion, nothing is executed), and lines that are empty or start with
  `#` are skipped. A variable that the environment already has keeps its
  value, so one call can override the file. A file that is named and cannot
  be read, or a line of another form, is an error.

| Variable | Read by | What it is | Without it |
|---|---|---|---|
| `MDIR_ENV_FILE=<file>` | the shell scripts of this table but `examples/argon/run.sh` | The file of the machine, as above. | Only the environment is read. |
| `MDIR_RELEASE_DIR=<dir>` | `scripts/release/publish.sh` | Where a release writes: the worktree `src-vVERSION`, the build tree `build-vVERSION`, the assets `vVERSION`, and the logs `logs/release-vVERSION-*.log`; a dry run adds `-dry-run` to each name ([workflow.md](workflow.md#releases)). | The script stops. |
| `MDIR_LLVM=<prefix>` | `publish.sh` | The install prefix of LLVM and MLIR. | The script stops. |
| `HDF5_PREFIX=<prefix>` | `publish.sh` | The install prefix of HDF5, as `scripts/build-hdf5.sh` names its own. | The script stops. |
| `CUDA_ROOT=<dir>` | `publish.sh` | The CUDA toolkit of the build of a release: the same name, and the same toolkit, that `mdir` reads at run time (above), so that a machine states it once. The manylinux tree is run without it. | The script stops. |
| `MDIR_PYTHON=<interpreter>` | `publish.sh` | The Python interpreter of the build of a release, which is built with the Python interface so that its suite holds the Python tests (#297). It runs those tests, so it has NumPy, and OpenMM for the tests that compare with it. It has no default: the interpreter and its packages are a choice of the machine. | The script stops. |
| `PYBIND11_DIR=<dir>` | `publish.sh` | The directory of pybind11's CMake package (CMake's `pybind11_DIR`). | The script stops. |
| `MDIR_TORCH_PYTHON=<interpreter>`, `MDIR_H5MD_PYTHON=<interpreter>` | `publish.sh` | Interpreters with PyTorch (and torch-pme) and with h5py, which the suite of a release gives to lit as `-Dtorch_python` and `-Dh5md_python` (below). | lit tries the Python of the build; the tests that need a package it lacks are unsupported. |
| `MDIR_RELEASE_GPU=<n>` | `publish.sh` | The index of the GPU that runs the suite of a release and the manylinux tree. | The script stops. |
| `MDIR_RELEASE_SUITE_LOCK=<file>`, `MDIR_RELEASE_GPU_LOCK=<file>` | `publish.sh` | Lock files of a shared machine ([workflow.md](workflow.md#shared-machines)), held with `flock`: the first around the suite, the second while the GPU is used, taken in that order. | No lock is taken. |
| `CMAKE`, `NINJA`, `LIT`, `DOCKER` | `publish.sh` | Programs: `cmake`, `ninja`, `lit` (the one beside `ninja` if there is one), and `docker` by default. `DOCKER` is needed for the manylinux tree and for `--container`. | The script stops if the program is not found. |
| `PANDOC`, `TECTONIC` | `publish.sh`, `scripts/paper/build-pdf.sh` | Programs: `pandoc` (3.x) and `tectonic` by default. | The script stops if the program is not found. |
| `ENGINES=<prefix>` | `scripts/validation/gromacs/run.sh` | A prefix whose `bin` is searched first for `tleap` and `gmx`. | The path is searched. |
| `TLEAP`, `GMX` | `scripts/validation/gromacs/run.sh` | Programs: `tleap` of AmberTools and `gmx` of GROMACS by default. | The script stops if the program is not found. |
| `MDIR` | `scripts/validation/gromacs/run.sh`, `compare.py` | The `mdir` program to compare; `mdir` by default. | The script stops if the program is not found. |
| `MDIR_BENCH_DIR=<dir>` | `scripts/experiments/neighbor-structures/run.sh` | The prepared Amber suite (above); this script has no default for it. | The script stops. |
| `NVCC`, `CUDA_ARCH` | `scripts/experiments/neighbor-structures/run.sh` | The program `nvcc` (by default) and the architecture it compiles for (`sm_86` by default). | The script stops if `nvcc` is not found. |
| `LLVM_PREFIX=<prefix>` | `examples/argon/run.sh` | The install prefix of LLVM, as `scripts/build-llvm.sh` names its own. | The script stops. |
| `MDIR_BUILD=<dir>` | `examples/argon/run.sh` | The build directory of MDIR; `build` by default. | The script stops if it holds no `bin/mdir-opt`. |

`scripts/build-llvm.sh` and `scripts/build-hdf5.sh` install, and list their
own variables in their headers; their default prefix is under the home
directory, and nothing has to exist there. `CC`, `CXX`, `PATH`,
`LD_LIBRARY_PATH`, `LIT_OPTS`, `CUDA_CACHE_PATH`, and `MDIR_BENCH_DIR` reach
the build and the suite of a release as the environment has them.

## Lit parameters

The suite also takes lit parameters, given as `-D<name>=<value>` in
`LIT_OPTS` (`test/mdir_lit.py`):

- `gpu_workers` sets how many tests that need the GPU run at once (16 by default, D208);
- `compile_threads` sets the `MDIR_COMPILE_THREADS` of each test;
- `compile_cache=off` runs the suite without its compile cache.
- `torch_python=<interpreter>` runs the tests that need PyTorch with it (the feature `torch`), and those that need torch-pme if it imports `torchpme` as well (the feature `torchpme`, D231).
- `h5md_python=<interpreter>` runs the tests that read a trajectory with h5py with it (the feature `h5py`).
- `release=1` enables the tests of a release ([workflow.md](workflow.md#releases)).

A change that adds or reads an environment variable adds its row here.
