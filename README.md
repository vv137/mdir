# MDIR

MDIR is an MLIR-based compiler stack for general-purpose molecular dynamics.
It is at an early stage. A Lennard-Jones system compiles and runs on the
CPU, sequentially or with OpenMP, and on NVIDIA GPUs, in single, mixed, or
double precision, and reproduces reference values. A driver reads a control
file and writes a log, a trajectory, and checkpoints, from which a run
continues exactly.

## Documents

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The design |
| [docs/ops-m0.md](docs/ops-m0.md) | Types and ops for the first milestone |
| [docs/mdrt-m0.md](docs/mdrt-m0.md) | Proposal for the runtime and execution of the first milestone |
| [docs/driver-m0.md](docs/driver-m0.md) | The driver and its control file |
| [docs/decisions.md](docs/decisions.md) | Decisions and their status |
| [docs/prior-art.md](docs/prior-art.md) | Earlier work and what is taken from it |

## Requirements

- A C++17 compiler
- CMake 3.20 or later
- Ninja
- LLVM and MLIR 23.1.2, built with `scripts/build-llvm.sh`
- `lit`, for the tests (`pip install lit`)
- For NVIDIA GPUs: the driver, and the CUDA toolkit for its header and its
  device math library. Version 11.2 is known to work.
- For checkpoints: HDF5, built with `scripts/build-hdf5.sh`

## Building LLVM

```sh
LLVM_ROOT=$HOME/opt/llvm scripts/build-llvm.sh
```

The script clones the release, builds it with MLIR and the X86, NVPTX, and
AMDGPU targets, and installs it to `$LLVM_ROOT/23.1.2`. It then builds the
OpenMP runtime from the same source and installs it to the same place.

## Building HDF5

```sh
HDF5_ROOT=$HOME/opt/hdf5 scripts/build-hdf5.sh
```

The script installs HDF5 1.14.6 to `$HDF5_ROOT/1.14.6`. Without HDF5, MDIR
is built all the same and writes no checkpoints.

## Building MDIR

```sh
LLVM_PREFIX=$HOME/opt/llvm/23.1.2

cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(command -v lit) \
    -DHDF5_ROOT=$HOME/opt/hdf5/1.14.6

cmake --build build
```

If the CUDA toolkit is found, the runtime for NVIDIA GPUs is built as
`build/lib/libmdrt_cuda.so`. `-DCUDAToolkit_ROOT=<path>` names a toolkit
that is not on the search path, and `-DMDIR_ENABLE_CUDA=OFF` leaves the
runtime out.

## Testing

```sh
cmake --build build --target check-mdir
```

The tests need `mlir-opt`, `mlir-runner`, and `FileCheck` from the LLVM
installation. Some tests run generated code and compare its results with
reference values.

## Tools

`mdir-opt` parses, verifies, transforms, and prints MDIR modules.

```sh
build/bin/mdir-opt test/Dialect/MD/ops.mlir
```

| Pass | Effect |
|---|---|
| `--md-check-exchange` | Proves the exchange contracts of pair kernels. |
| `--md-expand-truncation` | Expands truncation attributes into kernels. |
| `--md-differentiate` | Replaces `md.evaluate` with calls to generated derivative functions. |
| `--md-inline` | Inlines potentials, functions, and programs into the code that calls them. |
| `--convert-md-to-md-exec` | Converts `md` and `dyn` ops to loops over particles and pairs. |
| `--md-exec-reuse-neighbors` | Makes a neighbor structure that is built in a loop a value that the loop carries and refreshes. |
| `--md-exec-fuse-loops` | Fuses loops over the pairs of one neighbor structure. Run `--cse` after it. |
| `--md-exec-simplify-distance` | Rewrites pair kernels in powers of the squared distance. Changes rounding. Run `--canonicalize --cse` after it. |
| `--md-exec-assign-precision` | Assigns `f32` or `f64` to fields and kernels. Options: `mode=single`, `mixed`, or `double`, and a type per role. Run it last before the lowering. |
| `--md-exec-assign-storage` | Gives every field a buffer and converts the loops to the storage form, in which they update buffers where they are. With `memory=device` the buffers are on a GPU. |
| `--convert-md-exec-to-loops` | Converts the loops in the storage form to `scf` loops over `memref`s. |
| `--convert-md-exec-to-gpu` | Converts the loops in the storage form, with buffers on a device, to kernels of the upstream `gpu` dialect. |

After the last pass the module holds only upstream dialects, so `mlir-opt`
lowers it to LLVM and `mlir-runner` runs it:

```sh
build/bin/mdir-opt input.mlir \
    --md-check-exchange --md-differentiate --md-expand-truncation \
    --md-inline --convert-md-to-md-exec="skin=0.3 width=96" \
    --md-exec-reuse-neighbors --md-exec-fuse-loops --cse \
    --md-exec-assign-precision="mode=mixed" \
    --md-exec-assign-storage --convert-md-exec-to-loops \
  | mlir-opt --convert-scf-to-openmp --canonicalize \
      --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
      --convert-vector-to-llvm --expand-strided-metadata \
      --finalize-memref-to-llvm --convert-arith-to-llvm \
      --convert-func-to-llvm --convert-cf-to-llvm \
      --convert-openmp-to-llvm --reconcile-unrealized-casts \
  | mlir-runner -e main --entry-point-result=void \
      --shared-libs=build/lib/libmdrt.so,$LLVM_PREFIX/lib/libomp.so,$LLVM_PREFIX/lib/libmlir_c_runner_utils.so
```

Without `--convert-scf-to-openmp --canonicalize` and
`--convert-openmp-to-llvm`, the loops run sequentially and `libomp.so` is
not needed. `test/Integration` holds complete programs.

For a GPU, the last two passes of `mdir-opt` and the passes of `mlir-opt`
are others, and the runtime for GPUs is loaded:

```sh
export CUDA_ROOT=/usr/local/cuda

build/bin/mdir-opt input.mlir \
    ... \
    --md-exec-assign-precision="mode=mixed" \
    --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
  | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
      --reconcile-unrealized-casts \
  | mlir-runner -e main --entry-point-result=void \
      --shared-libs=build/lib/libmdrt.so,build/lib/libmdrt_cuda.so,$LLVM_PREFIX/lib/libmlir_c_runner_utils.so
```

The kernels are embedded as PTX text, which the driver compiles when the
program starts.

## Running a simulation

`mdir-run` reads a control file, compiles the run, and executes it.

```sh
build/bin/mdir-run examples/argon.toml
```

```text
MDIR: 864 particles, 2000 steps of 0.005 ps
MDIR: compiled in 0.33 s
INFO:      STEP           TIME      TOTAL_ENE  POTENTIAL_ENE    KINETIC_ENE    TEMPERATURE
INFO:         0         0.0000      -834.1316     -1304.8874       470.7558       183.0000
INFO:       100         0.5000      -834.1466     -1087.9806       253.8339        98.6745
...
INFO:      2000        10.0000      -834.1467     -1083.2389       249.0922        96.8312
MDIR: ran in 0.39 s, 0.20 ms per step, 2198.1 ns per day
MDIR: the total energy changed by 1.813e-05 of its value
```

The control file is a TOML file, in Å, kcal/mol, and ps. `[execution]`
selects the target, `cpu` or `gpu`, the number of threads, and the
precision. `mdir-run --template md` prints a control file with every
keyword. See [docs/driver-m0.md](docs/driver-m0.md).

## Examples

`examples/argon.toml` is liquid argon at constant energy: 864 atoms, 2000
steps of 5 fs with velocity Verlet. `examples/argon.mlir` is the same
system as a module that is written by hand.

```sh
export MDIR_BUILD=build LLVM_PREFIX=$HOME/opt/llvm/23.1.2
examples/run.sh examples/argon.mlir             # double precision
examples/run.sh examples/argon.mlir mixed 16    # mixed, 16 threads
CUDA_ROOT=/usr/local/cuda \
    examples/run.sh examples/argon.mlir mixed gpu
```

It prints the time, the potential, kinetic, and total energy, and the
temperature every 100 steps, and at the end the relative change of the
total energy, which is about 2e-5.

`examples/run.sh` compiles and runs a module with the tools, pass by pass.

## License

MIT. See [LICENSE](LICENSE).
