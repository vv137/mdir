# Building MDIR from source

This document says how to build LLVM, HDF5, and MDIR from source, how to
install the build, how to run the tests, and how to run the passes by hand
with `mdir-opt`. A release, its container, and the Python wheel need none
of it ([README.md](../README.md#installing)).
[CONTRIBUTING.md](../CONTRIBUTING.md) and
[Appendix B of the white paper](paper/B-contributing.md) say what a change
must carry and the tiers of tests.

## 1. Requirements

- A C++17 compiler whose standard library has the floating-point
  `std::from_chars` (`std::chars_format`): GCC 11 or later, or a Clang whose
  libc++ or libstdc++ has it. The configure step checks it and refuses an
  older library, such as that of GCC 9 (#654); CMake keeps the compiler of
  an existing build tree, so a new compiler takes a new tree
- CMake 3.20 or later
- Ninja
- LLVM and MLIR 23.1.2, built with `scripts/build-llvm.sh`
- `lit`, for the tests (`pip install lit`)
- For NVIDIA GPUs: the driver, and the CUDA toolkit for its header and its
  device math library. Versions 11.2 and 13.x are known to work.
- For checkpoints: HDF5, built with `scripts/build-hdf5.sh`
- For the Python interface (optional): Python 3.10 to 3.13, pybind11
  3.0.1, and NumPy 1.23 or later

## 2. Building LLVM

```sh
LLVM_ROOT=$HOME/opt/llvm scripts/build-llvm.sh
```

The script clones the release, builds it with MLIR and the X86, NVPTX, and
AMDGPU targets, and installs it to `$LLVM_ROOT/23.1.2`. It then builds the
OpenMP runtime from the same source and installs it to the same place.

## 3. Building HDF5

```sh
HDF5_ROOT=$HOME/opt/hdf5 scripts/build-hdf5.sh
```

The script installs HDF5 1.14.6 to `$HDF5_ROOT/1.14.6`. Without HDF5, MDIR
is built all the same and writes no checkpoints.

## 4. Building MDIR

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

| Option | Default | What it builds |
|---|---|---|
| `MDIR_ENABLE_CUDA` | `AUTO`: on where the CUDA toolkit is found | The runtime for NVIDIA GPUs; `ON` fails without the toolkit |
| `MDIR_ENABLE_PYTHON` | `OFF` | The Python interface (Section 4.1) |
| `MDIR_ENABLE_METATOMIC` | `OFF` | The plugin of learned potentials (Section 4.2) |
| `MDIR_BUILD_TESTS` | `ON` | The tests (`test/`), their tools, and `mdir-opt`, which links their test passes. The wheel builds with it `OFF`, since the Python sdist leaves `test/` out (#624); a build without it has no `check-mdir` |

### 4.1 The Python interface

In a build of the tree the interface is optional and off by
default. When enabled, NumPy is required and checked at configuration:

```sh
pip install pybind11==3.0.1 'numpy>=1.23'

cmake -G Ninja -S . -B build <the options above> \
    -DMDIR_ENABLE_PYTHON=ON \
    -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build

PYTHONPATH=build/python python3 -c "import mdir"
```

The package `mdir` is built in `build/python/mdir`; `PYTHONPATH=build/python`
imports it ([python-package.md](python-package.md), Section 2).

### 4.2 The plugin of learned potentials

`-DMDIR_ENABLE_METATOMIC=ON` (off by default) builds `libmdir_metatomic`,
the optional plugin that evaluates metatomic and AOTInductor models
through libtorch (D270, D273); MDIR's core loads it at run time and never
links it. What it needs and how it is found are in
[metatomic-ml1.md](metatomic-ml1.md), Section 6.1.

## 5. Installing a build

From a build tree, or from the source archive of a release built as above:

```sh
cmake --install build --prefix $HOME/opt/mdir
```

The installed tree is `bin/mdir` with its runtime and the OpenMP runtime in
`lib/`; it does not need the build trees of MDIR or LLVM. With the Python
interface, `cmake --install` puts the package in `MDIR_PYTHON_INSTALL_DIR`,
which names the package's directory (by default
`lib/pythonX.Y/site-packages/mdir`; D228). At run time it
needs HDF5 (found through the RPATH of the build), and on a GPU the NVIDIA
driver and the CUDA toolkit's `nvvm/libdevice`, cuFFT, and cuBLAS, under `CUDA_ROOT`
or the toolkit it was built with.

## 6. Testing

```sh
cmake --build build --target check-mdir
```

The tests need `mlir-opt`, `mlir-runner`, and `FileCheck` from the LLVM
installation. Some tests run generated code and compare its results with
reference values. A build with `MDIR_ENABLE_PYTHON=ON` adds the tests of
the Python interface.

The suite runs its tests in parallel, but at most sixteen of the tests that
need a GPU (`REQUIRES: cuda`) run at once, on the device that
`CUDA_VISIBLE_DEVICES` selects; an empty value hides the GPU and skips
them. `-Dgpu_workers=N` in `LIT_OPTS` changes that number; `-Dgpu_workers=1`
runs them one at a time. Each test lowers its programs with the cores over
that number of threads (`MDIR_COMPILE_THREADS`; `-Dcompile_threads=N`
changes it). The tests share the host objects they compile through the
compile cache in `build/test/compile-cache`; `-Dcompile_cache=off` runs
the suite without it. The same holds for the sanitized suite (`scripts/build-sanitized.sh`)
and the suite of a release (`scripts/release/publish.sh`). On a machine
that others share, hold a lock of the device around the whole suite, as
[workflow.md](workflow.md) says, so that no other suite or timing
run uses the device at the same time.

With sixteen GPU tests at once in place of four, and the three Python
tests that bounded its wall time split into one file per scenario, the
suite took about 260 s instead of about 1090 s on one RTX 3090 (D208,
#141). The suite hands `CUDA_CACHE_PATH` to its tests, as it hands `HOME`,
so that the driver's kernel cache of a suite goes where it says (#223).

Some tests check the tree itself:

- `test/Docs/markdown-links.test` checks that every relative link of
  `README.md`, `CHANGELOG.md`, and `docs/` names a file and a heading that
  exist (#390).
- `test/Docs/dialect-docs.test` compares the reference of the dialects in
  `docs/dialects/` with what `mlir-tblgen` generates from their TableGen
  definitions; `ninja mdir-dialect-docs` writes it again after a change of
  a `.td` file (#610).
- `test/Runtime/generated-templates.test` compares what
  `scripts/generate-pme-gpu-template.py` and
  `scripts/generate-groups-gpu-template.py` generate with the checked-in
  templates, byte for byte (#211); the generators emit the guards of #168
  that the templates hold.
- `test/Lit/test-outputs.test` lists any test that runs `mdir run` or a
  Python interpreter on a path under `%t` that no `rm` removes before its
  first use (#423). A test starts from an empty `%t`
  (`rm -rf %t && mkdir -p %t`, or `rm -rf %t` before `split-file` or a
  script) and writes inside it, not next to it (`%t.dcd`, `%t.toml`):
  outputs written at fixed paths left one numbered backup more
  (`#name.dcd.N#`) at every run of the suite in one build tree, and after
  99 runs `mdir` refused to write.
- Under compute-sanitizer (`-Dsanitize=1`,
  [debugging.md](debugging.md)), `test/Sanitizer/npt-velocity-verlet-gpu.test`
  runs 20 steps of NPT on examples/ala3 in mixed precision under initcheck,
  memcheck, and racecheck, whose initcheck reported 4.7 million reads of
  unwritten slots of the incidence structure before the build wrote zeros
  past each particle's count (D110, #347);
  `test/Sanitizer/groups-empty-gpu.test` builds three empty groups (#413);
  and `test/Sanitizer/pme-tiles-gpu.test` runs the spreading of order 8,
  with and without tiles, synccheck included (#427).

## 7. Tools

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
| `--md-exec-expose-validity` | Makes the test of validity of a neighbor structure a loop over particles, which `--md-exec-fuse-loops` fuses with the loop that writes the positions. |
| `--md-exec-fuse-loops` | Fuses loops over the pairs of one neighbor structure, and loops over the particles of one set. Run `--cse` after it. |
| `--md-exec-simplify-distance` | Rewrites pair kernels in powers of the squared distance. Changes rounding. Run `--canonicalize --cse` after it. |
| `--md-exec-assign-precision` | Assigns `f32` or `f64` to fields and kernels. Options: `mode=single`, `mixed`, or `double`, and a type per role. Run it last before the lowering. |
| `--md-exec-assign-storage` | Gives every field a buffer and converts the loops to the storage form, in which they update buffers where they are. With `memory=device` the buffers are on a GPU. |
| `--convert-md-exec-to-loops` | Converts the loops in the storage form to `scf` loops over `memref`s. |
| `--convert-md-exec-to-gpu` | Converts the loops in the storage form, with buffers on a device, to kernels of the upstream `gpu` dialect. |

The table names the passes of the semantic levels; `mdir emit FILE
--stage=pipeline` prints the whole pipeline that a run takes.

After the last pass the module holds only upstream dialects, so `mlir-opt`
lowers it to LLVM and `mlir-runner` runs it:

```sh
build/bin/mdir-opt input.mlir \
    --md-check-exchange --md-differentiate --md-expand-truncation \
    --md-inline --convert-md-to-md-exec="skin=0.3 width=96" \
    --md-exec-reuse-neighbors --md-exec-expose-validity \
    --md-exec-fuse-loops --cse \
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

### 7.1 The argon example by hand

`examples/argon/argon.toml` is liquid argon at constant energy: 864 atoms,
2000 steps of 5 fs with velocity Verlet. `examples/argon/argon.mlir` is the
same system as a module that is written by hand, which
`examples/argon/run.sh` compiles and runs with the tools, pass by pass:

```sh
export MDIR_BUILD=build LLVM_PREFIX=$HOME/opt/llvm/23.1.2   # both are needed
examples/argon/run.sh examples/argon/argon.mlir             # double precision
examples/argon/run.sh examples/argon/argon.mlir mixed 16    # mixed, 16 threads
CUDA_ROOT=/usr/local/cuda \
    examples/argon/run.sh examples/argon/argon.mlir mixed gpu
```

It prints the time, the potential, kinetic, and total energy, and the
temperature every 100 steps, and at the end the relative change of the
total energy, which is about 2e-5.
