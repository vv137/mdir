# Debugging and reports of defects

Status: 2026-09-30.

## Checking the installation

Run `mdir doctor` to report the build, probe the CUDA driver and visible
devices, and compile and run a small system on each built target (D155).
`mdir doctor --target=cpu` checks the CPU alone without needing a GPU;
`--target=gpu` checks just the GPU. Failures return a nonzero status and
keep their temporary inputs and logs at the printed path. The doctor
checks basic execution; a failure specific to your system needs the
report below.

The kernels are cubins for the architecture of the device. They are
compiled by the `ptxas` of the toolkit, which also supplies libdevice
(D214), and the driver loads them without compiling them.
A cubin compiled for one architecture does not load on a device of
another; such a run ends with `mdrt: the kernels were compiled for the
architecture of another device`.

The kernels are PTX instead in these cases:

- NVML cannot tell the device's architecture, and `MDIR_GPU_ARCH` does
  not give it;
- LLVM does not know the device's architecture;
- no `ptxas` is found. Without a toolkit, the compiler finds no `ptxas`.

The driver then compiles the PTX when a run loads it. A driver older than
the PTX ISA version that MDIR emits cannot do that, and the run, or the
doctor's GPU check, ends with `mdrt: the NVIDIA driver supports CUDA
<x.y>, which cannot compile kernels of PTX ISA <a.b>`; update the driver.

`MDIR_GPU_BINARY=ptx` keeps the kernels PTX, to rule out `ptxas`, and
`MDIR_GPU_ARCH=sm_XY` sets the architecture
([compile-cache.md](compile-cache.md#cubins-for-the-device)).

`mdir version` prints the toolkit's version and the driver's (`driver
API`). `mdir emit --stage=pipeline` shows the architecture that was
chosen, for example
`mdir-gpu-lower-to-nvvm{cubin-chip=sm_86 cubin-format=bin binary=auto}`,
or `cubin-format=isa` for PTX.

## Reporting a defect

A report that can be reproduced needs the build, the machine, the inputs,
and what the driver made of them. `mdir bug-report` collects all of it:

```sh
mdir bug-report run.toml -o report          # the inputs and each stage
mdir bug-report run.toml -o report --run    # and a run, each kernel waited for
tar czf report.tar.gz report
```

The directory holds:

| File | What it holds |
|---|---|
| `version.txt` | The version, the commit and whether the tree had changes, LLVM, the CUDA toolkit, HDF5 |
| `gpus.out`, `host.out` | The devices and their driver (`nvidia-smi`), the host (`uname -a`) |
| `environment.txt` | The variables that MDIR and the runtime read: `MDIR_*`, `MDRT_*`, `CUDA_*`, `OMP_*`, `LD_LIBRARY_PATH` |
| `inputs.txt`, `inputs/` | The control file and each input it names, with size and SHA-256; inputs up to 16 MiB are copied |
| `check.out` | What `mdir check` reads from the inputs |
| `module.out` | The program of the run, as `mdir emit` prints it |
| `pipeline.out` | The passes that lower it (`mdir emit --stage=pipeline`) |
| `lowered.out` | The program after the passes (`mdir emit --stage=lowered`) |
| `reproducer.mlir` | Written only if a pass failed or crashed: the input of the passes and the pipeline |
| `run.out`, `run.err` | With `--run`: the log of the run under `MDRT_WAIT=1` |
| `*.err` | The errors of each step, if it wrote any |
| `README.txt` | The exit status of each step |

Each step runs in a process of its own, so a crash of one leaves the rest.
Inputs larger than 16 MiB are named by their hash; attach them separately
if they are not public (the systems of the Amber benchmark suite are, and
`scripts/benchmarks/amber` names them by their hashes too).

Open an issue with the template "Defect" and attach the archive.

## Failures of the passes

When a pass fails or crashes, `mdir run` writes the input of the pass
pipeline to `mdir-reproducer.mlir` (or where `MDIR_REPRODUCER` says). The
file carries the pipeline, so

```sh
mdir-opt --run-reproducer mdir-reproducer.mlir
```

repeats the failure without the inputs of the run. `MDIR_PRINT_AFTER=<pass>`
prints the module after each run of that pass, and `MDIR_PIPELINE=<passes>`
replaces the pipeline (`mdir emit --stage=pipeline` prints the one that
`mdir run` uses). `mdir` and the Python package register only the dialects
and passes of MLIR that their lowerings use (`lib/Compiler/Registry.cpp`,
#655), so `MDIR_PIPELINE` may name those and MDIR's own; `mdir-opt` knows
every pass and dialect of MLIR, for trying others on a reproducer.

## Failures of a run

A run whose state is no longer that of the system stops with a message
on the standard error and the status 1; a part of a Python simulation
raises `SimulationError` instead and keeps the state from before the part
(D225).

| Message | What failed |
|---|---|
| `a position is not a number or is far outside the cell at a build of the neighbor structure` | The positions, at a build (D107, D176, D225), for example after a blow-up to about 1e190 nm, or to positions that are not numbers with PME, which ended the process on a GPU with an illegal address before D225 |
| `the constraints of the set 'S' failed at step N` | After the constraints of the positions of step N, a length of a group of the set `S` (`settles`, or `shake1` to `shake3` for the groups of SHAKE of one to three hydrogens) is off its target by more than $10^{-4}$ of its square below double precision, $10^{-8}$ in double, or is not a number (D278). In a run with constraints this message takes precedence: a build of the step that finds positions that are not numbers reports them only if every set passed its test |

| `cufftPlan3d failed with the cuFFT error E for the grid of K1 x K2 x K3 points` | On a device, the plan of the FFT of a grid of PME could not be made, usually for want of device memory for its work area; a transform that fails (`cufftExecR2C` and the others) is reported the same way (#323). The plans of a process are kept until it ends, one for each grid, direction, and precision, in a table that grows; before #323 a process that used more than 8 plans ended with `mdrt: too many grids of FFT` and `abort()` (a program in mixed precision with tunable charges takes four a grid), and `python-frame-gradients-torchpme-gpu.test` runs its two systems in one process again |

The first two usually mean a time step too long for the system, a bad
contact in the starting structure, or a start that was not minimized; a
smaller `time_step` or a minimization first tells which.

## Failures on the device

The runtime reports a failure of the CUDA driver with the last kernel that
was launched. Kernels run asynchronously, so the failure of a kernel shows
at a later call. To find the kernel that failed:

1. `MDRT_WAIT=1`: every launch is waited for and checked, so the report
   names the kernel that failed.
2. `compute-sanitizer --tool memcheck mdir run run.toml` names the access,
   the thread, and the kernel. `initcheck` finds reads of memory that was
   never written; `racecheck`, races through shared memory.
3. Kernels are named `<function>_<op>[_<kind>]_<n>`, such as
   `mdir_run_particle_for_integration3_188`: the function, the op of the
   loop that the kernel comes from, what a fused kernel holds
   (`integration3`, `run2`), and the ordinal of the kernel in the lowering
   of the module (D[kernel-names]). The ordinal follows the ops of the
   program, not the lines of a text, so a name stays when the builder's
   text changes around its loop. `MDRT_TRACE=1` prints each module, its
   kernels, and each launch with its grid;
   `MDIR_PRINT_AFTER=gpu-kernel-outlining` prints the kernels themselves,
   in the order of their ordinals.
4. `MDRT_PROFILE=1` (with `MDRT_WAIT=1`, the time of each kernel) reports
   where the time goes.
5. `MDIR_DEVICE_MEMORY=2` reports where the device memory goes, for a run
   that is out of memory on a smaller system or that holds more than
   expected: what the context, the modules, cuFFT, and the structures of
   the tuples took outside the runtime's pool, and the pool's blocks at its
   peaks by size and use ([environment.md](environment.md)).

## The time of the outputs

`MDIR_CHECKPOINT_TIMES=1` makes `mdir run` print, on the standard error,
where the time of its outputs goes (#548). The lines below are from STMV
on one RTX 4090, with a row of the log every 100 steps:

```
MDIR: times of the frame at step 1000: copy 5.6 ms, write 22.6 ms
MDIR: times of the checkpoint at step 1000: copy 5.9 ms, wait 0.0 ms, state 10.9 ms, snapshot 2.7 ms; 14.1 ms in the call
MDIR: times of the row at step 1100: 447.1 ms since the row at step 1000, 36.7 ms of it in frames and checkpoints
...
MDIR: times of the write of the checkpoint at step 1000: order 21.7 ms, hash 237.1 ms, file 92.9 ms, file sync 0.1 ms, names 0.1 ms, directory sync 0.1 ms; 352.1 ms on its thread, of which the steps waited 0.0 ms
```

The copy is the time from the return of the call of the host before
(the row of the log, or the frame) to the start of this one: on a device,
the copy of the buffers to the host, and any wait for the device. In the
call of a checkpoint, the wait is for the write of the checkpoint before,
the state the copy of the buffers as the program holds them (in a cell
that is not periodic, their order into that of the input), and the
snapshot the copy of the rest of what the file holds. The thread that
writes the checkpoint (D336) reports when it is waited for,
at the next checkpoint or at the end of the run: the order of the input,
the hash of the state, the H5MD file up to its close, and the syncs and the
names that put it on stable storage under its name (D132, D173). The
interval between two rows holds, besides the steps, the frames and the
checkpoints between them and the start of a segment after a checkpoint.

## Tiers of tests

| Tier | How | What |
|---|---|---|
| Default | `lit test` | Units, dialects, lowerings, short runs on CPU and GPU |
| Sanitizer | `lit -Dsanitize=1 test` | The GPU tests under compute-sanitizer |
| Release | `lit -Drelease=1 test` | The whole matrix of the tests that run a part of their cases by default (`*-release.test`, `*-release-gpu.test`); the suite of `scripts/release/publish.sh` |
| PyTorch | `lit -Dtorch_python=<python> test` | The DLPack views with PyTorch as the consumer (`python-dlpack-torch*.test`), with an interpreter that imports torch and the module of the build; without it, and without torch in the interpreter of the tests, they are unsupported |
| Scale | `MDIR_BENCH_DIR=<dir> lit test/Scale` | Short runs of the Amber benchmark suite |
| Host sanitizers | `scripts/build-sanitized.sh <dir>` | All tests, with the driver and runtimes under AddressSanitizer and UndefinedBehaviorSanitizer |
