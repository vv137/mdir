# Installing MDIR

This manual is for users who install MDIR and for administrators who
install it for others, on a workstation or a cluster. It covers the
release's binary tarball and Python wheel, the container, the build from
source, the installed tree, the check of an installation with
`mdir doctor`, clusters, and the common failures. The details of the
source build, its CMake options, the tests, and `mdir-opt` are in
[building.md](building.md).

## 1. Choosing a route

| Route | Gives | Needs on the machine | GPUs |
|---|---|---|---|
| Binary tarball (Section 3) | `bin/mdir` with its runtime, HDF5, cuFFT, cuBLAS, the OpenMP runtime, and libdevice | Linux x86-64, glibc 2.28 or later; for a GPU, an NVIDIA driver that supports CUDA 13.0 | Turing (compute capability 7.5) and later |
| Python wheel (Section 4) | The package `mdir`, without the `mdir` command | CPython 3.10 to 3.13, glibc 2.28 or later; for a GPU, the driver and the extra `cuda` | Turing and later |
| Container (Section 5) | `mdir` in a Docker or Apptainer image | Docker or Apptainer; for a GPU, the driver and the NVIDIA container toolkit (Docker) or `--nv` (Apptainer) | Turing and later |
| Source build (Section 6) | `mdir`, optionally the Python package, built against the toolkit of the machine | GCC 11 or later, CMake, Ninja, LLVM 23.1.2, optionally HDF5 and a CUDA toolkit | Every architecture that the chosen toolkit supports, Volta (7.0) included with a CUDA 12.x toolkit |

The prebuilt routes are built with CUDA 13.0. CUDA 13 removed offline
compilation and cuFFT for Maxwell, Pascal, and Volta (Section 2.2), so a
Tesla V100 needs a source build with a CUDA 12.x toolkit.

## 2. Requirements

### 2.1 The matrix

| Component | Prebuilt tarball and wheel | Source build | Notes |
|---|---|---|---|
| OS | Linux x86-64 with glibc 2.28 or later: RHEL, Rocky, and Alma 8 and later, Ubuntu 20.04 and later, Debian 11 and later | Linux x86-64 with a compiler of the next row. Built and tested on Ubuntu 24.04 (the container and CI), on AlmaLinux 8 (manylinux_2_28, the release), and on RHEL 9 derivatives | `ldd --version` prints the glibc |
| C and C++ compiler | None: libstdc++ and libgcc are linked statically (GCC 14 of the manylinux_2_28 image) | GCC 11 or later, or a Clang whose standard library has the floating-point `std::from_chars`; the configure step refuses an older library, such as that of GCC 9 (#654) | A GCC newer than the system's needs its own libstdc++ at run time (Section 2.4) |
| CMake, Ninja | None | CMake 3.20 or later, Ninja | |
| LLVM and MLIR | None (linked into the tarball and the wheel) | 23.1.2, built with `scripts/build-llvm.sh` ([building.md](building.md#2-building-llvm)) | Another release of LLVM is not supported: MDIR follows the API of one release |
| CUDA toolkit | None: the tarball carries cuFFT, cuBLAS, and libdevice; the wheel's extra `cuda` brings cuFFT, cuBLAS, and ptxas from NVIDIA's wheels of CUDA 13 | Optional. CUDA 13.x for Turing and later, CUDA 12.x for Volta and earlier (Section 2.2). Versions 11.2, 12.9, and 13.x are known to work | The toolkit supplies `ptxas`, which compiles the kernels at run time, libdevice, cuFFT, and cuBLAS |
| NVIDIA driver | Supports CUDA 13.0: 580 or later | Supports the major version of the toolkit of the build (Section 2.3) | `nvidia-smi` prints the highest CUDA version that the driver supports |
| HDF5 | Carried in the tarball and the wheel | Optional, for checkpoints: 1.14.6, built with `scripts/build-hdf5.sh` ([building.md](building.md#3-building-hdf5)) | Without HDF5, MDIR runs and writes no checkpoints |
| Python | CPython 3.10 to 3.13; NumPy 1.23 or later | Optional (`-DMDIR_ENABLE_PYTHON=ON`): Python 3.10 to 3.13, pybind11 3.0.1, NumPy 1.23 or later | |
| `lit` | None | For the tests: `pip install lit` | |

### 2.2 GPU architectures and CUDA toolkits

MDIR compiles its kernels at run time for the architecture of the device
that it runs on (D214), with the `ptxas` of the toolkit. The toolkit
therefore has to support the device's architecture both when MDIR is built
and when it runs.

| Architecture | Compute capability | Examples | CUDA toolkit | MDIR |
|---|---|---|---|---|
| Maxwell, Pascal | 5.x, 6.x | GTX 900 and 10 series, P100 | 12.x or earlier | Not tested |
| Volta | 7.0 | V100 | 12.x or earlier; not 13 | Source build only. Tested with CUDA 12.9 on V100; two tests of the suite are open (#992) |
| Turing | 7.5 | T4, RTX 20 series | 12.x or 13.x | Expected to work |
| Ampere | 8.0, 8.6 | A100, RTX 30 series | 12.x or 13.x | Tested on RTX 3090 |
| Ada Lovelace | 8.9 | L40, RTX 40 series | 12.x or 13.x | Tested on RTX 4090 |
| Hopper | 9.0 | H100, H200 | 12.x or 13.x | Expected to work |
| Blackwell | 10.0, 12.0 | B200, RTX 50 series, RTX PRO 6000 Blackwell | 12.8 or later | Tested on RTX 5090 and RTX PRO 6000 Blackwell |

The sources of the toolkit column:

- CUDA 13.0 removed Maxwell, Pascal, and Volta: "Offline compilation and
  library support for these architectures have been removed in CUDA
  Toolkit 13.0 major version release", and the 12.x toolkits remain
  supported for them (CUDA 13.0 release notes, Section 2.6.2, Deprecated
  Architectures). cuFFT 13.0 "Removed support for Maxwell, Pascal, and
  Volta GPUs" (Section 3.2.1).
  <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html>
- CUDA 12.8 added compiler support for the Blackwell architectures
  SM_100, SM_101, and SM_120 (CUDA 12.8 release notes, Section 1.2).
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html>

With a CUDA 13.x toolkit on a Volta device, `ptxas` stops with `Value
'sm_70' is not defined for option 'gpu-name'` and cuFFT fails to make its
plans; use a 12.x toolkit for the build and at run time (`CUDA_ROOT`,
Section 7).

A build with a toolkit older than 13.0 leaves out the cuFFT callback of
the fused PME convolution (`pme_callback.cu`, built only with CUDA 13 and
`nvcc`), and the runs take the separate pass instead.

### 2.3 Driver minimums

The driver must support the major version of CUDA of the toolkit that
MDIR was built with (D177): a run on an older driver stops with
`mdrt: the NVIDIA driver supports CUDA <x.y>; this build of MDIR, with
CUDA <a.b>, needs a driver that supports CUDA <a>.0 or later`
([debugging.md](debugging.md#checking-the-installation)).

| Toolkit of the build | Linux driver | Source |
|---|---|---|
| 13.x (the prebuilt routes) | 580 or later (580.65.06 for the 13.0 release) | CUDA 13.0 release notes, Section 2.2, Tables 2 and 3 |
| 12.x | 525 or later (525.60.13 for the 12.0 release); later 12.x releases name their own, for example 575.51.03 for 12.9 | the same tables |

Where the kernels are kept as PTX (no `ptxas`, or `MDIR_GPU_BINARY=ptx`),
the driver compiles them, and it must be at least as new as the PTX ISA
that MDIR emits ([debugging.md](debugging.md#checking-the-installation)).

### 2.4 The libstdc++ of a newer GCC

A program built with a GCC newer than the system's needs the libstdc++ of
that GCC at run time. On the system's older library it stops at load:

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

This holds for `mdir`, the runtime libraries, the Python extension, and
any other program built with that compiler, such as an MPI or NCCL built
for MDIR. The GCC of an environment module, or one built from source, has
this property; the `gcc-toolset` and `devtoolset` packages of RHEL link
the newer parts of the library statically and do not. The release tarball
and wheel link libstdc++ statically.

To check a build, compare the newest version that the binary needs with
those that the library on the loader's path provides:

```sh
objdump -T <prefix>/bin/mdir | grep -o 'GLIBCXX_[0-9.]*' | sort -uV | tail -1
strings "$(g++ -print-file-name=libstdc++.so.6)" | grep '^GLIBCXX_' | sort -uV | tail -1   # the compiler's
strings /lib64/libstdc++.so.6 | grep '^GLIBCXX_' | sort -uV | tail -1                       # the system's
ldd <prefix>/bin/mdir | grep libstdc++
```

The remedy is to load the same compiler module in every job that runs
MDIR, or in the module file of MDIR (Section 9.1), so that the compiler's
`lib64` is on `LD_LIBRARY_PATH`. Linking that directory into the
binaries instead (`-Wl,-rpath,<dir>` in `CMAKE_EXE_LINKER_FLAGS` and
`CMAKE_SHARED_LINKER_FLAGS`) has not been tested; check the installed
result with `readelf -d <prefix>/bin/mdir | grep -i path` and `ldd`.

## 3. The binary tarball

`mdir-<version>-manylinux_2_28_x86_64.tar.gz`, from
[GitHub Releases](https://github.com/vv137/mdir/releases), is relocatable:

```sh
tar -xzf mdir-0.2.1-manylinux_2_28_x86_64.tar.gz -C <prefix-parent>
<prefix>/bin/mdir doctor
```

It carries HDF5, cuFFT, cuBLAS, the OpenMP runtime, and libdevice, and
takes `libcuda` from the driver. A GPU run needs no CUDA toolkit, but
`ptxas` from a toolkit or `PATH` compiles the kernels to cubins; without
it the kernels stay PTX, which the driver compiles
([compile-cache.md](compile-cache.md#cubins-for-the-device)). The layout of
the tree and what each part is found by are in
[packaging/README.md](../packaging/README.md).

## 4. The Python wheel

The wheel is for CPython 3.10 to 3.13 on Linux x86-64 (manylinux_2_28),
among the assets of the release:

```sh
python -m venv mdir-env
mdir-env/bin/pip install "mdir-0.2.1-cp312-cp312-manylinux_2_28_x86_64.whl[cuda]"
mdir-env/bin/python -c "import mdir; print(mdir.__version__)"
```

The extra `cuda` adds NVIDIA's wheels of cuFFT, cuBLAS, and `ptxas` of
CUDA 13; without it the package runs on the CPU. The `mdir` command is not
in the wheel; it comes with the tarball. What the package finds where, and
the variables that override it, are in
[python-package.md](python-package.md).

On a cluster, make the environment on a file system that the compute
nodes see, and put pip's cache on a scratch or local file system
(`PIP_CACHE_DIR`, or `UV_CACHE_DIR` for `uv`), not in a home directory
with a quota.

## 5. The container

[packaging/](../packaging/README.md) builds a Docker image and an
Apptainer image from it, with LLVM, HDF5, CUDA 13.0, and MDIR:

```sh
docker build -f packaging/Dockerfile -t mdir:0.2.1 .
docker run --rm --gpus all -v "$PWD":/work mdir:0.2.1 doctor
apptainer build mdir-0.2.1.sif docker-daemon://mdir:0.2.1
apptainer run --nv mdir-0.2.1.sif doctor
```

The base images of CUDA must not be newer than the driver of the host
supports. Apptainer, which needs no daemon or root at run time, is the
usual choice on clusters; `--nv` binds the host's driver into the image.

## 6. The source build

The order is LLVM, then HDF5, then MDIR. Each step is in
[building.md](building.md):

```sh
LLVM_ROOT=<dir>/llvm scripts/build-llvm.sh          # Section 2; installs <dir>/llvm/23.1.2
HDF5_ROOT=<dir>/hdf5 scripts/build-hdf5.sh          # Section 3; installs <dir>/hdf5/1.14.6

cmake -G Ninja -S . -B <build> \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=<dir>/llvm/23.1.2/lib/cmake/mlir \
    -DLLVM_DIR=<dir>/llvm/23.1.2/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(command -v lit) \
    -DHDF5_ROOT=<dir>/hdf5/1.14.6 \
    -DCUDAToolkit_ROOT=<cuda toolkit>               # Section 4
cmake --build <build>
cmake --build <build> --target check-mdir           # Section 6, optional
cmake --install <build> --prefix <prefix>           # Section 5
```

Building LLVM takes most of the time, about an hour with 32 jobs, and its
links take much memory: `LINK_JOBS` (8 by default) bounds them. Keep the
build trees (`LLVM_BUILD`, `<build>`) on a local disk; the install trees
may be on a shared file system.

The CMake options of MDIR (`MDIR_ENABLE_CUDA`, `MDIR_ENABLE_PYTHON`,
`MDIR_ENABLE_METATOMIC`, `MDIR_BUILD_TESTS`) are in
[building.md](building.md#4-building-mdir). CMake keeps the compiler and
the CUDA toolkit of an existing build tree: after a change of either, for
example another module, configure a new tree.

## 7. The installed tree

`cmake --install` writes `bin/mdir` with its runtime (`lib/libmdrt.so`,
`lib/libmdrt_cuda.so`) and the OpenMP runtime in `lib/`; it needs neither
the build tree of MDIR nor that of LLVM
([building.md](building.md#5-installing-a-build)). At run time it takes:

| Part | From |
|---|---|
| HDF5 | the RPATH of the build: keep the HDF5 install where it was, or on `LD_LIBRARY_PATH` |
| `libcuda.so.1` | the NVIDIA driver |
| libdevice, `ptxas` | the toolkit that `CUDA_ROOT`, `CUDA_HOME`, or `CUDA_PATH` names, the first that is set, else the toolkit of the build |
| cuFFT, cuBLAS | under `CUDA_ROOT` or the toolkit of the build ([building.md](building.md#5-installing-a-build)) |
| libstdc++ | the system, unless the compiler was newer (Section 2.4) |
| The Python package | `MDIR_PYTHON_INSTALL_DIR` (by default `lib/pythonX.Y/site-packages/mdir`, D228); put its parent on `PYTHONPATH` |

`-DMDIR_BUNDLE_LIBRARIES=ON` (off by default) installs HDF5, cuFFT, and
libdevice into the tree as well, as the release tarball does, so that the
tree no longer depends on where they were at the build.

On a cluster, install into a prefix that every compute node mounts, and
give users a module file that sets `PATH`, `PYTHONPATH` where the Python
package is installed, and `CUDA_ROOT` when the toolkit of the nodes is at
another path than at the build. The variables are in
[environment.md](environment.md).

## 8. Checking an installation

`mdir doctor` prints the build information, probes the driver and the
visible devices, and compiles and runs a two-atom system on each target
that was built ([driver.md](driver.md#15-installation-doctor), D155):

```sh
mdir doctor                   # every built target
mdir doctor --target=cpu      # without a GPU or a driver
mdir doctor --target=gpu      # the GPU alone; fails on a build without CUDA
mdir doctor --timeout=600     # a busy machine; the default limit is 120 s
mdir version                  # the toolkit and the driver API version
```

The exit status is 0 when every requested check passes. A failure keeps
its inputs and logs in a temporary directory and prints its path and a
command that runs the check again. A CUDA build on a node without a
device fails the GPU check instead of skipping it: on a cluster, run the
GPU check inside a job on a GPU node, and `--target=cpu` on a login node.

The doctor checks loading, compilation, and a short run. It does not test
checkpoints, PME, or constraints; the test suite
([building.md](building.md#6-testing)) and `mdir check <control file>` do.
The wheel has no `mdir` command: `python -c "import mdir"` checks its
loading, and a short simulation from
[python-package.md](python-package.md) its GPU path.

## 9. Clusters

### 9.1 Environment modules

Load the same compiler and CUDA toolkit for the build and for every job
that runs it:

```sh
module load <gcc module> <cuda module> <cmake module>
cmake ... -DCUDAToolkit_ROOT="$CUDA_HOME"
```

Name the toolkit explicitly (`CUDAToolkit_ROOT`); CMake may otherwise find
another one on the search path. For a Volta partition, build a tree of its
own with a 12.x module (Section 2.2). A module file for MDIR should load
the compiler module when the compiler was newer than the system's
(Section 2.4).

### 9.2 A login node or a job

The build needs no GPU: the kernels are compiled when a run starts, for
the device it finds. Build where the site allows long, parallel builds,
usually a build or CPU node, or a login node within its limits; a build
inside a job on a GPU partition holds GPUs that it does not use. LLVM is
built once per site and release; MDIR builds in minutes. The tests that
need a GPU (`REQUIRES: cuda`) run inside a GPU job; elsewhere they are
reported as unsupported.

### 9.3 Caches on network file systems

| Cache | Default | On a cluster |
|---|---|---|
| The driver's kernel cache, `CUDA_CACHE_PATH`, for kernels kept as PTX | `~/.nv/ComputeCache` | Point it at a local or scratch directory; the test suite passes it on to its tests (#223) |
| MDIR's compile cache, `MDIR_COMPILE_CACHE_DIR` (D397) | `${XDG_CACHE_HOME:-$HOME/.cache}/mdir/compile` if that file system is local, else `/var/tmp/mdir-<uid>/compile` if local, else none | A directory local to each node, or on scratch. Warm a shared directory with one compile before many ranks read it ([compile-cache.md](compile-cache.md#on-clusters)) |
| The cache of the chosen skins, `MDIR_SKIN_CACHE_DIR` | `$XDG_CACHE_HOME/mdir/skins`, else `~/.cache/mdir/skins` | Small JSON files; removing it is always safe |
| pip or uv | the home directory | `PIP_CACHE_DIR`, `UV_CACHE_DIR` on scratch |

Removing any of them only costs a compile.

### 9.4 Slurm

A run on one GPU, `md.sbatch`, which continues itself over jobs
([driver.md](driver.md#27-runs-longer-than-a-job)):

```sh
#!/bin/bash
#SBATCH --job-name=mdir-md
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=<n>
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@300
module load <modules of the build>
export MDIR_COMPILE_CACHE_DIR=<node-local or scratch dir>/mdir-cache
export CUDA_CACHE_PATH=<node-local or scratch dir>/nv-cache
srun mdir run --continue --max-walltime 23:50 md.toml
status=$?
# 75: stopped at a checkpoint; submit the script again (if the site allows
# sbatch from a job), or let a job array or a dependency chain go on
[ "$status" -eq 75 ] && sbatch "$SLURM_SUBMIT_DIR/md.sbatch"
exit "$status"
```

The check of a node, and the test suite of a build tree on a GPU node:

```sh
srun --partition=<gpu partition> --gres=gpu:1 --time=10 mdir doctor --target=gpu
srun --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=<n> --time=60 \
    env CUDA_CACHE_PATH=<scratch>/nv-cache LIT_OPTS="-j <n>" \
    cmake --build <build> --target check-mdir
```

The scheduler sets `CUDA_VISIBLE_DEVICES` to the GPUs of the job; MDIR
runs on the first visible device, or on the one that `MDRT_DEVICE`
selects. `[execution] threads` sets the threads of a CPU run; match it to
`--cpus-per-task`. On a partition of mixed devices, request the kind with
the site's constraint or GRES type (`--gres=gpu:<type>:1`), since the
kernels are compiled for the device that the run finds.

## 10. Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| `GLIBCXX_3.4.<n>' not found` or `CXXABI_1.3.<n>' not found` at load | Built with a GCC newer than the system's libstdc++ | Load the compiler module at run time (Section 2.4) |
| `mdrt: the NVIDIA driver supports CUDA <x.y>; this build of MDIR, with CUDA <a.b>, needs ...` | The driver is older than the toolkit's major version | Update the driver, or build with a toolkit it supports (Section 2.3) |
| `mdrt: the NVIDIA driver supports CUDA <x.y>, which cannot compile kernels of PTX ISA <a.b>` | Kernels kept as PTX on an old driver | Update the driver, or let `ptxas` compile cubins ([debugging.md](debugging.md#checking-the-installation)) |
| `ptxas fatal : Value 'sm_70' is not defined for option 'gpu-name'`; cuFFT fails to make a plan on a V100 | A CUDA 13 toolkit on Volta | Build and run with a 12.x toolkit (Section 2.2) |
| `mdrt: the kernels were compiled for the architecture of another device` | `MDIR_GPU_ARCH` names another architecture than the device's, for example in a module file shared by partitions of different devices | Unset it, or set it to the device's ([compile-cache.md](compile-cache.md#cubins-for-the-device)) |
| The GPU check of `mdir doctor` fails on a login node | No device there | Run it in a GPU job, or `--target=cpu` (Section 8) |
| Configure stops at the probe of `std::from_chars` | GCC older than 11 | Load a newer compiler and configure a new build tree (Section 2.1) |
| The suite reports most GPU tests unsupported | `CUDA_VISIBLE_DEVICES` empty, or no GPU in the job | Run it in a GPU job (Section 9.2) |
| Slow first steps and a full home directory | Caches on a network file system or under a quota | Move them (Section 9.3) |
| No checkpoints are written | MDIR was built without HDF5 | Build HDF5 and reconfigure (Section 6) |

`mdir bug-report` collects the build, the machine, the inputs, and each
stage of a failing run ([debugging.md](debugging.md#reporting-a-defect)).

In build scripts that run under `set -o pipefail`, a pipeline that ends in
`head` (`ompi_info | grep ... | head -5`) can fail: `head` exits after its
lines, the writer gets SIGPIPE, and the pipeline's status is not zero.
Write the output to a file first, or use `grep -m <n>`.
