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
| `MDIR_MLIP_PLUGIN=<path>` | `mdir run` and a Python simulation with a term of `[[energy.model]]` | The plugin of learned potentials to load (D270, [metatomic-ml1.md](metatomic-ml1.md)); by default `libmdir_metatomic.so` in the directory of the runtime libraries, where `-DMDIR_ENABLE_METATOMIC=ON` builds it. |
| `MDIR_MLIP_TIMING=1` | the plugin of learned potentials | Measures the plugin's part of the pair list (its own search, or the taking of MDIR's) and its model apart: the plugin waits for the device before and after the pair list and after the model at every evaluation, and when the model is freed writes the mean of each per evaluation, after the first, and how it took the list, to the standard error (D273, D277, [metatomic-ml1.md](metatomic-ml1.md#11-the-aotinductor-backend-d273)). For measurements only: the waits cost time. |
| `MDIR_MLIP_PAIRS=exact\|padded\|search` | `mdir run` and a Python simulation with a term of `[[energy.model]]`, when the program is built and when the model is loaded | How the model takes its pair list (D277, [metatomic-ml1.md](metatomic-ml1.md#12-the-pair-list-from-mdirs-neighbor-structures-d277)): `exact` (the default), MDIR's list of the pairs within the cutoff, with one copy of their number per call of the model; `padded`, every slot of MDIR's list, the pairs within the cutoff and the skin, without a wait; `search`, the plugin's own search of every image (D270), for comparison, and the program builds no list. Another value is an error. |
| `MDIR_MLIP_CHECK_PAIRS=1` | the plugin of learned potentials | At every evaluation that takes MDIR's pair list, the plugin also searches its own and compares the pairs within the cutoff of the two as sets; a difference stops the run with both counts, and the first agreement is written to the standard error (D277). For tests and validation: the search costs what it cost before. |
| `TMPDIR=<dir>` | the plugin's backend `AOTI` | Where `AOTIModelPackageLoader` extracts an AOTInductor package (about 45 MB for Allegro-OAM-L) before it loads it (D273). |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch's allocator, in the plugin of learned potentials | MDIR sets it to `expandable_segments:True` before it loads the plugin, unless it is set already (D270, [metatomic-ml1.md](metatomic-ml1.md#65-sharing-the-device)). |
| `MDIR_ALLEGRO_PACKAGE=<path>` | the test suite | The model package `Allegro-OAM-L-0.1.nequip.zip` that the tests of the model term build their artifact from (feature `allegro-package`); without it they are unsupported. |
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
| `MDIR_FUSED_PRUNE=off\|always` | every lowering of a GPU program (`mdir run`, `mdir.compile`, a Python simulation) | Where a refresh of a dual list of groups prunes, the first loop over pairs that takes its result prunes in its own kernel, a warp for each group, by default from 4096 groups of the last build, 20,480 on a device of `sm_12x` ([groups-m1.md](groups-m1.md), Section 6.2; D[fused-prune]). `off` leaves every pruning to the refresh and compiles no such kernel; `always` takes it at any number of groups, as the tests of small systems do. Any other value is the default. The forces are those of the refresh's pruning to the order of their sums. The generated code differs, so the compile cache keys it apart. |

## Debugging

These are described in [debugging.md](debugging.md). They act on `mdir run`.

| Variable | Effect |
|---|---|
| `MDIR_PIPELINE=<passes>` | Replaces the pass pipeline, to try another order of passes or to stop part of the way. |
| `MDIR_PRINT_AFTER=<pass>` | Prints the module after each run of that pass, for example `gpu-kernel-outlining` for the kernels. |
| `MDIR_REPRODUCER=<file>` | Where a failing pass writes the module and pipeline that `mdir-opt --run-reproducer` repeats; `mdir-reproducer.mlir` by default. `mdir bug-report` sets it. |
| `MDRT_TRACE` (set) | The device runtime prints each module that it loads, its kernels, and each launch with its grid. |
| `MDRT_DIST_TRACE` (nonempty) | The in-process transport of `md_dist` (`runtime/mdrt_dist.c`) prints the owned and ghost rows of each domain when a halo is built, and the rows between domains that each `halo_copy` and `halo_sum` moves, on stderr (D275). |
| `MDRT_PROFILE` (set) | The device runtime counts its calls and their time, and reports them at the end; a store of the structures of the tuples reports, when it closes, the bytes it kept on the device (D257). |
| `MDRT_WAIT=1` | The device runtime waits for each launch. A failing kernel is then reported at its own launch, and with `MDRT_PROFILE`, each kernel is timed. |
| `MDRT_CHECK_KEPT_TUPLES` (set) | The runtimes (`runtime/mdrt.c`, `runtime/mdrt_cuda.c`) build anew every structure of the tuples that a simulation takes from those it keeps across its activations (D253), or a segment of `mdir run` from those of the segment before (D255; the log of such a run then says at its end how many structures were built and how many taken), and, on a GPU, every structure that the kernels of the program build there (D257), and compare the two, on a device after reading the kept one back; a difference ends the program with a message. A start then costs what it cost without the kept structures, and more ([jit-invariants.md](jit-invariants.md#what-a-simulation-keeps-across-activations-d253)). With it, every update of the tunables of a simulation also builds its program from a copy of the model of the compile, as before D254, and compares it, member for member and to the bit, with the one built from the model that the simulation keeps for its updates; a difference fails the update with a message that names what differs (`compiler::Simulation`; [jit-invariants.md](jit-invariants.md#what-an-update-of-the-tunables-keeps-d254)). An update then costs what it cost before, and more. With it too, every conversion of an array of the program that a start of an activation takes as the simulation kept it is made anew and compared, and every buffer that the entry takes without a copy is copied at the start and compared when the activation ends; a difference ends the program with a message that names the buffer (D258; `compiler::Simulation`; [jit-invariants.md](jit-invariants.md#what-a-start-gives-the-entry-d258)). |

## Tests and validation

| Variable | Read by | Effect |
|---|---|---|
| `MDIR_BENCH_DIR=<dir>` | `test/Scale`, `scripts/benchmarks/amber` | The prepared Amber benchmark suite. Without it, those tests are unsupported ([debugging.md](debugging.md)). |
| `MDIR_CHECK_PRUNING` (set) | every lowering of a GPU program (`mdir run`, `mdir.compile`, a Python simulation) | Each refresh of a dual list of groups counts, at the positions of its step, the pairs within the cutoff in the outer list and in the inner list through its active units, and the runtime stops the run if the two differ; at its end the run reports the steps it checked and the pairs it counted ([groups-m1.md](groups-m1.md), Section 6.2; D268). A step then costs more, and waits for the counts. The generated code differs, so the compile cache keys it apart. |
| `MDIR_CONSTRAINT_TEST` (`separate`) | the builder of every program of dynamics with constraints (`mdir run`, `mdir.compile`) | Each set of constraints is tested after the step by a loop of its own over its tuples instead of in the kernel of the constraints, which the integration kernel of D110 joins with the loops over particles around them (D278). For comparison: the results, the stop, and its message are the same, to the bit in the deterministic mode; a step takes one kernel more. The program differs, so the compile cache keys it apart. |
| `MDIR_PROLATE_CELLS=1` | the builder of a program with `[pme] kernel = "PROLATE"` (`mdir run`, `mdir.compile`, a Python simulation) | The direct sum of the prolate kernel in the loop over pairs evaluates the cells of polynomials in $x = r/r_c$ of `pme_far`, as before D280, instead of the Chebyshev series that a table of $r^2$ takes under `fast_math` in f32 kernels ([pme-m1.md](pme-m1.md), Section 7). For comparing the two; the generated code differs, so the compile cache keys it apart. `0` or empty is the default. |
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
| `CPU_BENCH_DIR=<dir>` | `scripts/benchmarks/cpu-breakdown/bench.py`, `summarize.py`, `pairs.py` | The directory of the systems of the CPU breakdown, one directory each, where `bench.py` writes its runs and `summarize.py` its summary (D267). | The script stops. |
| `MDIR`, `GMX`, `MLIR_TRANSLATE` | `scripts/benchmarks/cpu-breakdown/bench.py`, `prof.py` | Programs: `mdir`, `gmx`, and `mlir-translate` by default. | A run that cannot start records no time. |
| `SAMPLER=<library>` | `scripts/benchmarks/cpu-breakdown/bench.py` | The sampler built from `sampler.c`, which the sampled variants preload. | The sampled variants record nothing. |
| `SAMPLER_OUT=<dir>`, `SAMPLER_WALL_US`, `SAMPLER_CPU_US` | `scripts/benchmarks/cpu-breakdown/sampler.c` (preloaded) | Where the sampler writes its samples and the objects of the JIT, and its periods: of the main thread's wall clock, 500 µs, and of the CPU time of the process, 4000 µs, by default. `SAMPLER_NOSTACK` (set) keeps the interrupted address alone. | Without `SAMPLER_OUT` the library does nothing. |
| `NVCC`, `CUDA_ARCH` | `scripts/experiments/neighbor-structures/run.sh` | The program `nvcc` (by default) and the architecture it compiles for (`sm_86` by default). | The script stops if `nvcc` is not found. |
| `CUDA_ROOT=<dir>` | `scripts/benchmarks/prolate-ewald/resources.py`, `p0_job.py` | The CUDA toolkit whose `cuobjdump`, `cu++filt`, and `ncu` the measurement of #337 runs. | `resources.py` stops; `p0_job.py` skips Nsight Compute. |
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
- `mlip_python=<interpreter>` runs the tests of the model term with the Python of the environment that the plugin was built against (the feature `mlip-python`; `MDIR_MLIP_PYTHON` of the build by default), and `allegro_package=<path>` gives them the model package as `MDIR_ALLEGRO_PACKAGE` does.
- `release=1` enables the tests of a release ([workflow.md](workflow.md#releases)).

A change that adds or reads an environment variable adds its row here.
