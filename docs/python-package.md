# The Python package (D228)

The last item of M2a ([python-m2.md](python-m2.md), Sections 1, 4 and 6;
[#133](https://github.com/vv137/mdir/issues/133)). The Python interface of
D192 is the package `mdir`, built as a pip wheel for CPython 3.10–3.13 on
manylinux_2_28 (x86-64) on the packaging baseline of the release tarball
(D177). A wheel installs into a clean virtual environment and runs on the
CPU, and on an NVIDIA GPU with only the driver, without the source tree,
the build tree, or a CUDA toolkit. The maintainer decided the open choices
in [PR #190](https://github.com/vv137/mdir/pull/190) as recommended.

## 1. Installing

```sh
python -m venv mdir-env
mdir-env/bin/pip install "mdir-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl[cuda]"
mdir-env/bin/python -c "import mdir; print(mdir.__version__)"
```

The wheel depends on NumPy 1.23 or later. The extra `cuda` adds two of
NVIDIA's wheels of CUDA 13, `nvidia-cufft` (cuFFT, which GPU runs need)
and `nvidia-cuda-nvcc` (for its ptxas, which compiles the kernels);
CPU-only installs leave it out. The `mdir` command is not in the wheel: it
comes with the release tarball and the container (D177, D174), which carry
their own copy of the same version.

## 2. Layout

```
site-packages/
  mdir/__init__.py                     the names of the extension, as mdir.X
  mdir/_core.cpython-3XY-x86_64-linux-gnu.so
  mdir/lib/libmdrt.so, libmdrt_cuda.so, libomp.so
  mdir/cuda/nvvm/libdevice/libdevice.10.bc, mdir/cuda/EULA.txt, version.txt
  mdir/licenses/LICENSE, HDF5-COPYING, pybind11-LICENSE, pocketfft-LICENSE,
    tomlplusplus-LICENSE, LLVM-LICENSE.TXT, OpenMP-LICENSE.TXT
  mdir.libs/libhdf5-<hash>.so.310.5.1  grafted and renamed by auditwheel
  nvidia/cu13/lib/libcufft.so.12       NVIDIA's wheels, with the extra cuda
  nvidia/cu13/bin/ptxas
```

The extension module is `mdir._core`; `mdir/__init__.py` imports its names
and sets the `__module__` of its classes and exceptions to `mdir`, so that
`mdir.InputError` is the name in tracebacks. A build of the tree has the
same layout in `<build>/python/mdir`, so `PYTHONPATH=<build>/python` imports
the build as before.

## 3. Runtime libraries

| Library | Source | Found by |
|---|---|---|
| `libmdrt.so`, `libmdrt_cuda.so`, `libomp.so` | the wheel, `mdir/lib` | the extension: `MDIR_RUNTIME_DIR`, else `lib` beside the extension (dladdr), else `lib` beside a directory above it (a build tree, an installed prefix), else the build's |
| HDF5 | the wheel, `mdir.libs`, renamed by auditwheel so that another HDF5 in the process (h5py's) is not taken for it | the extension's RPATH |
| libdevice | the wheel, `mdir/cuda`, 464 KB | without `CUDA_ROOT`, `CUDA_HOME`, `CUDA_PATH`: `cuda` beside the extension, else `../share/mdir/cuda` of an installed tree, else the toolkit of the build |
| cuFFT (`libcufft.so.12`) | NVIDIA's `nvidia-cufft` wheel (extra `cuda`) | `libmdrt_cuda.so`'s RPATH, `$ORIGIN/../../nvidia/cu13/lib`, else the system's search path (a CUDA toolkit on `LD_LIBRARY_PATH` or in the loader's cache) |
| `libcuda.so.1` | the NVIDIA driver | the loader, when a GPU program is loaded |
| ptxas | NVIDIA's `nvidia-cuda-nvcc` wheel (extra `cuda`) | the `bin` of a toolkit that `CUDA_ROOT` names, else `nvidia/cu13/bin/ptxas` beside the package, else `PATH` (D214) |

libdevice is redistributable under Attachment A of the CUDA EULA, which
the wheel carries beside it, as the tarball does (D177). The wheel carries
no other part of CUDA: cuFFT comes from NVIDIA's own wheel, which keeps
each MDIR wheel at 78 MB (cuFFT alone is 287 MB, its wheel 214 MB, beyond
PyPI's default limit of 100 MB per file) and is shared with other CUDA
packages of the environment. Without it and without a toolkit's cuFFT, a
GPU program fails to load with a `CompileError` that names the extra.

The extra also brings ptxas, in `nvidia-cuda-nvcc` (`>=13.0,<14`: any
ptxas of CUDA 13 assembles the PTX that MDIR emits; the wheel was run with
13.0.88 and 13.4.92), so the kernels are compiled to cubins as in a build
of the tree. MDIR uses only `nvidia/cu13/bin/ptxas` of that wheel and of
the three it depends on (`nvidia-nvvm`, `nvidia-cuda-runtime`,
`nvidia-cuda-crt`); libdevice stays the wheel's own. An environment of
CPython 3.13 with `mdir[cuda]` holds 964 MB in site-packages: 644 MB of
NVIDIA's files (325 MB of them cuFFT with nvJitLink, the rest for ptxas),
264 MB of MDIR with its HDF5, and 57 MB of NumPy.

libdevice is that of the toolkit the wheel was built with, CUDA 13.0,
where a build of the tree on this machine links that of CUDA 13.4. In
mixed precision on the GPU the two give energies that differ in the
seventh digit (9.3e-7 of the potential of ala3, Section 8); with
`CUDA_ROOT` at the same toolkit the wheel and the build agree to the bit.

Without ptxas (the package installed without the extra, beside a system
cuFFT), the kernels are loaded as PTX, which the driver compiles (D214):
the first load of a program that the driver's cache (`~/.nv/ComputeCache`,
or `CUDA_CACHE_PATH`) has not seen takes 3–13 s more per stage of the ala3
tutorial (Section 8), and later loads come from that cache. On a home
directory on a network file system, point `CUDA_CACHE_PATH` at a local
disk.

## 4. A CPU-only machine

`import mdir` loads no CUDA library: the extension does not link the
driver, `libmdrt_cuda.so` is loaded only with a GPU program, and NVML is
asked for the architecture only when a GPU program is compiled.
`python-package.test` checks that the import maps none of `libcuda`,
`libcufft`, `libnvidia-ml`, or `libmdrt_cuda`; the wheel's validation
checked it again after a CPU run, and in a container without GPUs a GPU
program fails with a `CompileError` that names `libcuda.so.1`.

## 5. Version

`project(mdir VERSION ...)` in `CMakeLists.txt` is the one source (D174).
scikit-build-core's regex provider reads the wheel's version from it, the
extension's `__version__` is compiled from it, as `mdir version` is, and
`scripts/release/prepare.sh` sets the ala3 example's `mdir[cuda]==X.Y.Z`
with it. `python-package.test` checks that the five agree, and
`check-wheel.sh` that a wheel's name and metadata carry it.

## 6. Building

`pyproject.toml` at the root of the repository: scikit-build-core 0.11
builds the target `mdir_wheel` (the extension and the runtime) and
installs the CMake component `python` into the wheel (`SKBUILD` selects the
package's destinations in `python/CMakeLists.txt` and
`runtime/CMakeLists.txt`). Build requirements: `scikit-build-core>=0.11,<0.12`,
`pybind11==3.0.1` (the version that CMake requires exactly), and NumPy,
for the configuration check of D193 (its C API is not used, so one wheel
takes NumPy 1.23 to 2.x). The build needs LLVM/MLIR 23.1.2, HDF5, and a
CUDA toolkit, named through CMake's variables:

```sh
pip wheel . --no-deps -w dist \
  -C cmake.define.LLVM_DIR=<llvm>/lib/cmake/llvm \
  -C cmake.define.MLIR_DIR=<llvm>/lib/cmake/mlir \
  -C cmake.define.HDF5_ROOT=<hdf5> -C cmake.define.CUDAToolkit_ROOT=<cuda>
```

Release wheels come from the target `wheels` of
`packaging/Dockerfile.manylinux`, on the toolchain of the tarball (PyPA's
manylinux_2_28 image, GCC 14, LLVM/MLIR 23.1.2, HDF5 1.14.6, CUDA 13.0):
one wheel per CPython 3.10–3.13 of the image, with libstdc++ and libgcc
linked statically, repaired by auditwheel to `manylinux_2_28_x86_64`
(`--exclude libcuda.so.1 --exclude libcufft.so.12`):

```sh
DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile.manylinux \
  --build-arg MDIR_GIT_COMMIT=$(git rev-parse HEAD) \
  --target wheels --output type=local,dest=dist .
scripts/release/check-wheel.sh dist/*.whl
```

With the toolchain in Docker's cache the four wheels take about 35
minutes, most of it auditwheel's repair of the 270 MB extension.
`check-wheel.sh` checks the tag and the version, that no ELF file needs
GLIBC newer than 2.28, that the extension and the runtime need no system
libstdc++, that every needed library is in the wheel or allowed (the
manylinux_2_28 list, the driver's libcuda, cuFFT), that libdevice is
there, and that the wheel carries the notice of each third party whose
code it holds in binary form: in `mdir/licenses`, those of pybind11,
toml++, and LLVM (in the extension), pocketfft (in `libmdrt.so`), the
OpenMP runtime, and HDF5, beside MDIR's license, and the CUDA EULA beside
libdevice. A library that auditwheel grafts other than libhdf5 fails the
check, since no notice is known for it.
[packaging/licenses/README.md](../packaging/licenses/README.md) gives the
clause that asks for each. The metadata names MDIR's own license
(`License-Expression: MIT`, `License-File: LICENSE`).

A CLI-only build is unchanged: with `MDIR_ENABLE_PYTHON` off it needs
neither Python nor NumPy.

## 7. The tutorial

`examples/ala3/run.py` declares `mdir[cuda]==<version>` and NumPy in its
PEP 723 header, for Python 3.10–3.13. Until the wheels are published,
uv takes them from a directory or a release page:

```sh
uv run --find-links <wheels> examples/ala3/run.py --out ala3-python
```

The script reads its inputs beside itself, so a copy of `examples/ala3`
runs anywhere; `python run.py` in an environment where the wheel is
installed does the same, and so does `PYTHONPATH=<build>/python python
run.py` with a build of the tree.

## 8. Validation

The four wheels of the Docker target; host Ubuntu 22.04
(glibc 2.35), RTX 3090; CPython 3.10.19, 3.11.14, 3.12.12, 3.13.12 in clean
virtual environments (made by uv for 3.10, 3.11, and 3.13, by `python -m
venv` and pip 24.3.1 for 3.12), the wheel installed with `[cuda]` (NumPy
2.2.6 to 2.5.3, `nvidia-cufft` 12.4.0.43, `nvidia-cuda-nvcc` 13.4.92), run
with a `PATH` of `/usr/bin:/bin` and an empty cache of the driver from a
copy of
`examples/ala3` outside the trees, with `PYTHONPATH`, `CUDA_ROOT`,
`CUDA_HOME`, `CUDA_PATH`, and `LD_LIBRARY_PATH` unset.

| Check | Result |
|---|---|
| `check-wheel.sh` | ok for the four: GLIBC at most 2.28 (HDF5), 2.27 (extension); no GLIBCXX or CXXABI need; HDF5 in `mdir.libs` |
| Import, each version | `mdir.__version__` = `importlib.metadata.version("mdir")` = 0.1.0; `mdir.InputError.__module__` = `mdir`; no CUDA library mapped after the import and after a CPU run of 10 steps |
| Tutorial on the CPU, each version (8 threads, `--steps-scale 0.002`) | the energy files and the trajectory equal to the bit those of the build tree (`PYTHONPATH`, the same commit) |
| Single point of ala3, the 3.12 wheel against the build tree | CPU double, CPU mixed, GPU double: energies and forces equal to the bit. GPU mixed: potential −52473.7247 against −52473.6758 kJ/mol (9.3e-7 of it), largest force difference 0.0011 kJ/mol/nm (0.17 between mixed and double); with `CUDA_ROOT` at the build's toolkit (CUDA 13.4) the wheel equals the build to the bit, so the difference is libdevice 13.0 against 13.4 |
| Tutorial on the GPU, each version (`--steps-scale 0.01`), mixed; 3.12 double and deterministic | four stages complete; deterministic mixed repeated, equal to the bit |
| Tutorial on the GPU at full length, 3.12 | 1 ns of production in 95.5 s; mean density 1.0011 g/cm³ over 100 reports |
| NumPy 1.23.5 (the floor), 3.10 | single points on the CPU and the GPU in double equal to those with NumPy 2.5.3 |
| AlmaLinux 8.10, glibc 2.28 (PyPA's image), no GPU, no toolkit, 3.12 | install, the import checks, the CPU tutorial; a GPU program fails with `CompileError: ... libcuda.so.1: cannot open shared object file` |
| The same image with GPU 1 and the driver only | without `[cuda]`: `CompileError` for `libcufft.so.12`; with it, the GPU single point equals the host wheel's to the bit, and the tutorial runs |
| CLI-only tree, `MDIR_ENABLE_PYTHON` off, Python interpreters set to a path that does not exist | configures and builds `mdir`; `mdir doctor --target=cpu` passes; no NumPy or pybind11 entry in the cache |

### Performance

`mdir run` of a build of the tree against the 3.12 wheel on GPU 0 (RTX
3090, 300 W cap, idle, under its lock), mixed precision, the same settings
in a control file and in Python: the suite's β and grid, 8 Å cutoff, lists
to 10 Å, SHAKE and SETTLE, velocity Verlet; NPT with stochastic velocity
and cell rescaling every 25 steps; `MDIR_COMPILE_CACHE=off`. ms/step is
that of the second half of the run, as `mdir run` reports it. One run per
cell.

| System | Steps | `mdir run` ms/step | Wheel ms/step | Build tree's Python ms/step |
|---|---|---|---|---|
| JAC NVE | 20,000 | 0.259 | 0.2594 | — |
| JAC NPT | 20,000 | 0.276 | 0.2756 | 0.2754 |
| Factor IX NVE | 10,000 | 0.870 | 0.8676 | 0.8677 |
| Cellulose NVE | 4,000 | 4.056 | 4.0506 | 4.0581 |
| STMV NPT 4 fs | 2,000 | 13.581 | 13.6144 | 13.5867 |

The wheel runs at the rate of `mdir run` on every system measured.

Compile time, from the model to a simulation ready to run (`mdir.compile`
and `Simulation`), with ptxas or a driver cache that holds the kernels, in
s, taken before D224, which has since removed the lowering that
`mdir.compile` did only to fill `Program.lowered_ir` (16.8 s of the 38.9 s
on Cellulose):

| System | Wheel | Build tree's Python | `mdir run` "compiled in" |
|---|---|---|---|
| JAC NPT | 25.4 | 29.8 | 10.6 |
| Factor IX NVE | 24.1 | 24.1 | 5.3 |
| Cellulose NVE | 37.9 | 36.2 | 4.5 |
| STMV NPT 4 fs | 186.6 | 173.0 | 12.4 |

The wheel compiles as the build tree's Python does. Python's time is
longer than `mdir run`'s "compiled in", which leaves out the reading and
the preparation of the inputs: over whole processes of 200 steps, `mdir
run` took 24.6 s on Cellulose and 97.8 s on STMV, the wheel 46.9 s and
182.7 s. The package does not change that.

With the extra, MDIR finds ptxas beside the package and prints no
warning; without ptxas the driver compiles the PTX (Section 3). The four
stages of the ala3 tutorial (`--steps-scale 0.01`, after D224; the 3.12
wheel, GPU 1, the driver's cache on a local disk) compiled in:

| ptxas | Driver's cache | Compile time of the stages (s) | Sum (s) |
|---|---|---|---|
| none (installed without the extra's nvcc) | empty | 7.8, 18.1, 31.4, 29.7 | 87 |
| none | filled by that run | 4.9, 11.4, 18.7, 17.9 | 53 |
| `nvidia-cuda-nvcc` 13.0.88, installed by hand beside the package | empty | 5.0, 11.2, 18.9, 17.8 | 53 |
| `nvidia-cuda-nvcc` 13.4.92, from the extra | empty | 5.0, 11.2, 20.0, 19.3 | 56 |
