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
(D[gpu-module-compile]), and the driver loads them without compiling them.
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
`mdir run` uses).

## Failures on the device

The runtime reports a failure of the CUDA driver with the last kernel that
was launched. Kernels run asynchronously, so the failure of a kernel shows
at a later call. To find the kernel that failed:

1. `MDRT_WAIT=1`: every launch is waited for and checked, so the report
   names the kernel that failed.
2. `compute-sanitizer --tool memcheck mdir run run.toml` names the access,
   the thread, and the kernel. `initcheck` finds reads of memory that was
   never written; `racecheck`, races through shared memory.
3. Kernels are named `<function>_<op>_l<line>_<n>`, where `<line>` is the
   line of the op in the output of `mdir emit`. `MDRT_TRACE=1` prints each
   module, its kernels, and each launch with its grid;
   `MDIR_PRINT_AFTER=gpu-kernel-outlining` prints the kernels themselves.
4. `MDRT_PROFILE=1` (with `MDRT_WAIT=1`, the time of each kernel) reports
   where the time goes.

## Tiers of tests

| Tier | How | What |
|---|---|---|
| Default | `lit test` | Units, dialects, lowerings, short runs on CPU and GPU |
| Sanitizer | `lit -Dsanitize=1 test` | The GPU tests under compute-sanitizer |
| Scale | `MDIR_BENCH_DIR=<dir> lit test/Scale` | Short runs of the Amber benchmark suite |
| Host sanitizers | `scripts/build-sanitized.sh <dir>` | All tests, with the driver and runtimes under AddressSanitizer and UndefinedBehaviorSanitizer |
