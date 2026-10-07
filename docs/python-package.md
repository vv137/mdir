# The Python package (D[python-package])

Status: design under review in the pull request that closes
[#133](https://github.com/vv137/mdir/issues/133), the last item of M2a
([python-m2.md](python-m2.md), Sections 1, 4 and 6).

The Python interface of D192 is an extension module built with
`-DMDIR_ENABLE_PYTHON=ON` and found through `PYTHONPATH=<build>/python`.
This item makes it a pip wheel for CPython 3.10–3.13 on manylinux_2_28
(x86-64), built on the packaging baseline of D177, that installs into a
clean virtual environment and runs on the CPU and on an NVIDIA GPU without
the source tree, the build tree, or a CUDA toolkit.

## 1. Layout

```
site-packages/
  mdir/
    __init__.py                        # re-exports the extension
    _core.cpython-3XY-x86_64-linux-gnu.so
    lib/libmdrt.so, libmdrt_cuda.so, libomp.so
    cuda/nvvm/libdevice/libdevice.10.bc, cuda/EULA.txt, cuda/version.txt
    licenses/                          # MDIR, HDF5
  mdir.libs/                           # HDF5 and zlib, renamed by auditwheel
  mdir-X.Y.Z.dist-info/
```

The extension's module becomes `mdir._core`; `mdir/__init__.py` imports
its names into `mdir` and sets their `__module__` to `mdir`, so `import
mdir`, `mdir.InputError`, and the rest are unchanged. The build tree has
the same layout (`<build>/python/mdir/`), so `PYTHONPATH=<build>/python`
still works.

## 2. Runtime libraries

| Library | Where it comes from | Found by |
|---|---|---|
| `libmdrt.so`, `libmdrt_cuda.so`, `libomp.so` | the wheel, `mdir/lib` | the extension, beside itself (dladdr), before the build tree |
| HDF5, zlib | the wheel, `mdir.libs` (auditwheel), names made unique so that h5py's HDF5 in the same process does not collide | RPATH of the extension |
| libdevice | the wheel, `mdir/cuda` (464 KB), redistributable under Attachment A of the CUDA EULA, carried beside it as in D177 | without `CUDA_ROOT`, `CUDA_HOME`, `CUDA_PATH`, the extension's `cuda` directory |
| cuFFT (`libcufft.so.12`) | to decide (Section 6) | |
| `libcuda.so.1` | the NVIDIA driver of the system | the loader, only when a GPU program loads |

## 3. Version

`project(mdir VERSION ...)` in `CMakeLists.txt` stays the single source
(D174). The wheel's version is read from it by scikit-build-core's regex
provider, and the extension's `__version__` is compiled from it, as `mdir
version` is.

## 4. Build

`pyproject.toml` at the root of the repository, scikit-build-core as the
build backend, `pybind11==3.0.1` (the version that CMake requires
exactly) and NumPy (for the configuration check of D193) as build
requirements, `numpy>=1.23` as the dependency. The wheel builds only the
extension and the runtime, installs only the `python` component, and
needs LLVM/MLIR 23.1.2, HDF5, and CUDA 13.0 as the tarball does.

`packaging/Dockerfile.manylinux` gets a `wheels` target on the cached
toolchain of the tarball (PyPA's manylinux_2_28 image, which has CPython
3.10–3.13 and auditwheel) that builds one wheel per interpreter and
repairs it with auditwheel to the manylinux_2_28 tag.

## 5. CPU-only import

`import mdir` loads no CUDA library: the extension does not link the
driver, `libmdrt_cuda.so` is loaded only when a GPU program is loaded, and
the architecture of a device is asked of NVML only when a GPU program is
compiled.

## 6. Open choices

To be decided in the pull request; recommendations in the PR description.
