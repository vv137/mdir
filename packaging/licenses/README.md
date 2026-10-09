# Notices of third parties in binary distributions

Three artifacts distribute code of third parties in binary form: the
release tarball (D177, `MDIR_BUNDLE_LIBRARIES`), the Python wheels (D228),
and the container image (`packaging/Dockerfile`). Each holds libraries it
bundles and code compiled or linked into MDIR's binaries, and carries the
notices that their licenses ask for, which a script checks:

| Artifact | Where its notices are | Check |
|---|---|---|
| The tarball | `share/mdir/LICENSE`, `share/mdir/licenses`, `share/mdir/cuda/EULA.txt` | `scripts/release/check-binary.sh` |
| A wheel | `mdir/licenses`, `mdir/cuda/EULA.txt` (in `site-packages`) | `scripts/release/check-wheel.sh` |
| The image | `/opt/mdir/share/mdir/LICENSE`, `/opt/mdir/share/mdir/licenses`, `/usr/local/cuda/EULA.txt` | `scripts/release/check-image.sh` |

`runtime/CMakeLists.txt` installs the notices of pocketfft, toml++, LLVM,
and the OpenMP runtime with every install of MDIR
(`mdir_install_notices`), and HDF5's where libhdf5 is bundled. The files
have the same names in the three artifacts. The first table below gives
the clause that asks for each notice, for the tarball; the sections on the
wheels and the image say what differs there.

## The release tarball

| Component | Where it is in the tarball | License | Notice in the tarball |
|---|---|---|---|
| HDF5 1.14.6 | `lib/libhdf5.so.*`, bundled | BSD 3-clause with the notices of its contributors | `licenses/HDF5-COPYING`, from HDF5's install tree: "Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or materials provided with the distribution." |
| pocketfft | compiled into `lib/libmdrt.so` | BSD 3-clause | `licenses/pocketfft-LICENSE`, from `third_party/pocketfft/LICENSE.md`: "Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution." |
| toml++ | compiled into `bin/mdir` | MIT | `licenses/tomlplusplus-LICENSE`, from `third_party/tomlplusplus/LICENSE`: "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software." |
| LLVM and MLIR 23.1.2 | linked statically into `bin/mdir` | Apache 2.0 with LLVM exceptions; parts under the legacy license of the University of Illinois | `licenses/LLVM-LICENSE.TXT`, the `llvm/LICENSE.TXT` of the release; see below |
| The OpenMP runtime of LLVM 23.1.2 | `lib/libomp.so`, bundled whole | Apache 2.0 with LLVM exceptions; parts under the University of Illinois and MIT licenses | `licenses/OpenMP-LICENSE.TXT`, the `openmp/LICENSE.TXT` of the release; see below |
| cuFFT, libdevice (CUDA 13.0) | `lib/libcufft.so.*`, `share/mdir/cuda/nvvm/libdevice` | The CUDA EULA, which lists both as redistributable (Attachment A) | `cuda/EULA.txt`, from the toolkit |
| libstdc++, libgcc (GCC 14) | linked statically into `bin/mdir`, `lib/libmdrt.so`, `lib/libmdrt_cuda.so` | GPL 3 with the GCC Runtime Library Exception 3.1 | None; see below |
| glibc, libz, the driver's `libcuda` | not in the tarball; taken from the system | | None: they are not distributed |

**LLVM and its OpenMP runtime.** The LLVM exception reads: "As an
exception, if, as a result of your compiling your source code, portions of
this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without
complying with the conditions of Sections 4(a), 4(b) and 4(d) of the
License." Section 4(a) is "You must give any other recipients of the Work
or Derivative Works a copy of this License".

- `lib/libomp.so` is LLVM's runtime distributed whole, as a file of its
  own. It is not a portion embedded into the object form of MDIR's source
  by compiling it, so the exception does not reach it, and Section 4(a)
  applies. Its license file also holds the University of Illinois license
  of its older parts ("Redistributions in binary form must reproduce the
  above copyright notice, this list of conditions and the following
  disclaimers in the documentation and/or other materials provided with
  the distribution").
- The LLVM and MLIR libraries linked into `bin/mdir` are the compiler
  itself, used as a library, and not what a compiler puts into its output.
  Whether the exception reaches a static link of them is at best unclear;
  and the parts of LLVM that remain under the legacy license, which
  `llvm/LICENSE.TXT` reproduces, ask for their notice in binary form
  whatever the exception says. The tarball therefore carries the file.

Both files are copies from the source of the LLVM release that
`scripts/build-llvm.sh` builds, since an install tree of LLVM does not
hold them; replace them when the version of LLVM changes.

**libstdc++ and libgcc.** The GCC Runtime Library Exception 3.1, Section
1, grants: "You have permission to propagate a work of Target Code formed
by combining the Runtime Library with Independent Modules, even if such
propagation would otherwise violate the terms of GPLv3, provided that all
Target Code was generated by Eligible Compilation Processes. You may then
convey such a combination under terms of your choice, consistent with the
licensing of the Independent Modules." MDIR's binaries are compiled by GCC
without a plugin, an eligible process, so no notice is asked for.

## The Python wheels

Established from a wheel built by the target `wheels` of
`packaging/Dockerfile.manylinux` (its file list, the libraries each ELF
file needs, and the symbols and strings of the extension):

| Component | Where it is in a wheel | Notice in the wheel |
|---|---|---|
| MDIR | `mdir/_core.*.so`, `mdir/lib/libmdrt.so`, `mdir/lib/libmdrt_cuda.so` | `mdir/licenses/LICENSE`; also `mdir-<version>.dist-info/licenses/LICENSE`, which the metadata names (`License-File`) |
| pybind11 3.0.1 | compiled into the extension `mdir/_core.*.so` | `mdir/licenses/pybind11-LICENSE`, the copy `python/pybind11-LICENSE` (BSD 3-clause): "Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution." |
| toml++ | compiled into the extension, which parses control files | `mdir/licenses/tomlplusplus-LICENSE` |
| LLVM and MLIR 23.1.2 | linked statically into the extension | `mdir/licenses/LLVM-LICENSE.TXT` |
| pocketfft | compiled into `mdir/lib/libmdrt.so` | `mdir/licenses/pocketfft-LICENSE` |
| The OpenMP runtime of LLVM 23.1.2 | `mdir/lib/libomp.so`, bundled whole | `mdir/licenses/OpenMP-LICENSE.TXT` |
| HDF5 1.14.6 | `mdir.libs/libhdf5-<hash>.so.310.5.1`, grafted by auditwheel | `mdir/licenses/HDF5-COPYING` |
| libdevice (CUDA 13.0) | `mdir/cuda/nvvm/libdevice/libdevice.10.bc` | `mdir/cuda/EULA.txt`; Attachment A of the EULA lists `libdevice.10.bc` among the files that "may be distributed with applications developed by you" |
| libstdc++, libgcc (GCC 14) | linked statically into the extension and the runtime | None, as for the tarball |
| cuFFT, ptxas | not in the wheel: NVIDIA's wheels `nvidia-cufft` and `nvidia-cuda-nvcc` (the extra `cuda`) distribute them, with their own licenses | None |
| glibc, libz, the driver's `libcuda` | not in the wheel; taken from the system | None |

The clauses for toml++, LLVM, pocketfft, the OpenMP runtime, and HDF5 are
those of the tarball's table, and the reasoning on the LLVM exception is
the same: the extension links the compiler as a library, and `libomp.so`
is the runtime distributed whole.

auditwheel grafts only libhdf5, which is built without compression and
needs only glibc. HDF5's notice is asked for whenever a libhdf5 is in
`mdir.libs`, and `check-wheel.sh` fails on any other library there: a
library that a later build grafts has no notice until a row is added here
and its notice to the wheel.

**Where the notices are, and the metadata.** The notices are files of the
package, in `mdir/licenses`, so that they are installed with it and found
beside the binaries they belong to, under the same names as in the
tarball. The core metadata keeps `License-Expression: MIT` and
`License-File: LICENSE` (MDIR's own license, in `.dist-info/licenses`),
for two reasons. The license files of `[project]` in `pyproject.toml` are
paths in the source tree, and HDF5's `COPYING` and the CUDA EULA come from
the install trees of the build, so `.dist-info/licenses` could hold only a
part of the notices. And the specification of the license metadata (PEP
639) describes one license for the source distribution and the wheels of a
project; it states that it does not "cover cases where the source
distribution and binary distribution packages don't have the same
licenses". The source distribution holds none of the bundled binaries.

## The container image

The runtime stage of `packaging/Dockerfile` starts from NVIDIA's image
`nvidia/cuda:13.0.1-runtime-ubuntu24.04` and copies into it the installed
tree of MDIR, HDF5's libraries, and libdevice:

| Component | Where it is in the image | Notice in the image |
|---|---|---|
| MDIR | `/opt/mdir/bin/mdir`, `/opt/mdir/lib/libmdrt.so`, `libmdrt_cuda.so` | `/opt/mdir/share/mdir/LICENSE` (MIT: "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.") |
| toml++; LLVM and MLIR 23.1.2 | in `/opt/mdir/bin/mdir` | `/opt/mdir/share/mdir/licenses/tomlplusplus-LICENSE`, `LLVM-LICENSE.TXT` |
| pocketfft | in `/opt/mdir/lib/libmdrt.so` | `/opt/mdir/share/mdir/licenses/pocketfft-LICENSE` |
| The OpenMP runtime of LLVM 23.1.2 | `/opt/mdir/lib/libomp.so` | `/opt/mdir/share/mdir/licenses/OpenMP-LICENSE.TXT` |
| HDF5 1.14.6 | `/opt/hdf5/1.14.6/lib/libhdf5.so.*`, copied from the build stage | `/opt/mdir/share/mdir/licenses/HDF5-COPYING`, from `share/COPYING` of HDF5's install tree |
| libdevice (CUDA 13.0) | `/usr/local/cuda/nvvm/libdevice/libdevice.10.bc`, copied from NVIDIA's devel image | `/usr/local/cuda/EULA.txt`, the copyright file of the package that holds libdevice in the devel image (`libnvvm-13-0`), which is the CUDA EULA |
| The base image: Ubuntu 24.04 with the CUDA runtime libraries (cuFFT among them) | the layers of NVIDIA's image, unchanged | `/NGC-DL-CONTAINER-LICENSE`, and the copyright file of each package in `/usr/share/doc` (for cuFFT, `/usr/share/doc/libcufft-13-0/copyright`, the same EULA), as NVIDIA's image carries them |
| libstdc++, libgcc, libz, glibc | Ubuntu's packages of the base image, which `mdir` links dynamically | Those of the base image, in `/usr/share/doc`; MDIR's binaries hold no copy |

The clauses are those of the tarball's table. The first four notices come
from the install of MDIR; the recipe copies MDIR's license, HDF5's
`COPYING`, and the EULA.

**The base image.** The NVIDIA Deep Learning Container License of the base
image grants, in Section 1(c), the right to "Develop and extend the
CONTAINER to create a Compatible (as defined below) derived CONTAINER that
includes the entire CONTAINER plus other software with primary
functionality, to develop and compile applications, and distribute such
derived CONTAINER to run applications, subject to the distribution
requirements indicated in this license." The image is such a derived
container: it keeps every layer of the base, with the license at its root,
and adds MDIR. Section 2(c) asks that it be distributed "subject to the
terms at least as protective as the terms of this license"; a release
that publishes the image must say so where it is published. The image is
built by `scripts/release/publish.sh --container` and is not uploaded
today.

libdevice is not in the runtime base image, so the EULA's own grant
applies to the copy: Section 2.2, "The portions of the SDK that are
distributable under the Agreement are listed in Attachment A", which
lists `libdevice.10.bc` for Linux, and Section 1.1.2(5), "The terms under
which you distribute your application must be consistent with the terms of
this Agreement". The image therefore carries the EULA beside it, as the
tarball and the wheels do.
