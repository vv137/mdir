# Containers

The recipes here build MDIR with CUDA in a container. A container carries
LLVM/MLIR 23.1.2, HDF5 1.14.6, and MDIR, so a workstation or a cluster node
needs only Docker or Apptainer and the NVIDIA driver.

| File | Builds |
|---|---|
| [Dockerfile](Dockerfile) | A Docker image, in stages: system packages, HDF5, LLVM with MLIR and the OpenMP runtime, MDIR; the final stage holds only the installed tree on CUDA's runtime image |
| [mdir.def](mdir.def) | An Apptainer (Singularity) image made from the Docker image |
| [Dockerfile.manylinux](Dockerfile.manylinux) | The release tarball (D177) and the Python wheels (D[python-package]) for manylinux_2_28 (glibc 2.28 and later), built on the PyPA image of that baseline |

**Layers.** The stages go from what changes least to what changes most:
the base image (pinned by digest) and its packages, HDF5 1.14.6, LLVM
23.1.2, and MDIR's sources last. Each copies only the script it builds
from, so a change to MDIR's sources leaves the HDF5 and LLVM layers cached,
and MDIR is rebuilt incrementally in a BuildKit cache mount. Keep this
order when the recipe changes: a file copied early invalidates every layer
after it. `.dockerignore` keeps the repository metadata, build trees, and
archives out of the context.

## Docker

Build from the root of the source tree. Building LLVM takes most of the
time: about an hour with 32 jobs.

```sh
docker build -f packaging/Dockerfile -t mdir:0.1.0 \
    --build-arg MDIR_GIT_COMMIT=$(git rev-parse HEAD) .
```

Build arguments:

| Argument | Default | Meaning |
|---|---|---|
| `DEVEL`, `RUNTIME` | CUDA 13.0.1 on Ubuntu 24.04, by digest | The base images. Change both together; their CUDA must not be newer than the host driver supports (`nvidia-smi` prints the highest version the driver supports). |
| `JOBS`, `LINK_JOBS` | `16`, `4` | Parallel compile and link jobs of LLVM. Linking takes much memory. |
| `MDIR_GIT_COMMIT` | empty | The commit that `mdir version` and logs report, since the context holds no `.git` |

The image runs `mdir`. With no arguments it runs `mdir doctor`. The working
directory is `/work`; mount the directory of the run there. On a GPU, the
NVIDIA container toolkit passes the devices with `--gpus`:

```sh
docker run --rm --gpus all mdir:0.1.0                 # mdir doctor
docker run --rm --gpus '"device=1"' -v "$PWD":/work mdir:0.1.0 run md.toml
docker run --rm -v "$PWD":/work mdir:0.1.0 run md.toml # target = "CPU"
```

Files written in `/work` belong to root unless the run is given the
user: `--user $(id -u):$(id -g)`. The examples of the repository are in
the image under `/opt/mdir/share/mdir/examples`.

## Apptainer

Apptainer keeps no cache of layers, so its image is made from the Docker
image, and LLVM is built once:

```sh
apptainer build mdir-0.1.0.sif packaging/mdir.def    # labels and a test
apptainer build mdir-0.1.0.sif docker-daemon://mdir:0.1.0
```

On a cluster without Docker, push the Docker image to a registry and
build from it (`Bootstrap: docker`, `From: <registry>/mdir:0.1.0`).

`--nv` binds the host's NVIDIA driver into the container. Apptainer runs as
the user and mounts the home and working directories:

```sh
apptainer run --nv mdir-0.1.0.sif doctor
apptainer run --nv mdir-0.1.0.sif run md.toml
CUDA_VISIBLE_DEVICES=1 apptainer run --nv mdir-0.1.0.sif run md.toml
```

## What the runtime image holds

- `/opt/mdir/bin/mdir`, and in `/opt/mdir/lib` the runtime, the GPU
  runtime, and the OpenMP runtime;
- HDF5's libraries in `/opt/hdf5/1.14.6/lib`, found through the RPATH;
- from CUDA: the runtime libraries of the base image (cuFFT among them),
  and `nvvm/libdevice` copied from the devel image, with
  `CUDA_ROOT=/usr/local/cuda`. The kernels are compiled at the start of a
  run and take their math functions from libdevice.

## The release tarball (manylinux_2_28)

`Dockerfile.manylinux` builds the binary tarball of a release on
`quay.io/pypa/manylinux_2_28_x86_64` (AlmaLinux 8, glibc 2.28, GCC 14), so
that it runs on RHEL, Rocky, and Alma 8 and later, Ubuntu 20.04 and later,
and Debian 11 and later:

```sh
DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile.manylinux \
  --build-arg MDIR_GIT_COMMIT=$(git rev-parse HEAD) --build-arg VERSION=0.1.0 \
  --target tarball --output type=local,dest=out .
scripts/release/check-binary.sh <extracted tree>
```

The layers follow the same rule as the container: the base image and
ninja, the parts of CUDA 13.0 for RHEL 8 that the build reads, HDF5, LLVM,
then MDIR's sources. The tarball holds:

| Path | What |
|---|---|
| `bin/mdir` | The driver; libstdc++ and libgcc are linked statically |
| `lib/libmdrt.so`, `lib/libmdrt_cuda.so` | The runtime |
| `lib/libomp.so` | The OpenMP runtime of LLVM, for threaded runs on the CPU |
| `lib/libhdf5.so.*` | HDF5, for checkpoints |
| `lib/libcufft.so.*` | cuFFT, for particle mesh Ewald on a GPU |
| `share/mdir/cuda` | libdevice, the math functions the kernels link, with the CUDA EULA |
| `share/mdir/examples`, `share/mdir/LICENSE` | The examples and MDIR's license |

cuFFT and libdevice are redistributable under Attachment A of the CUDA
EULA, which the tarball carries. With them a GPU run needs only the NVIDIA
driver: when `CUDA_ROOT`, `CUDA_HOME`, and `CUDA_PATH` are unset, the driver
links the kernels against `share/mdir/cuda`. Setting `CUDA_ROOT` still
selects another toolkit. `scripts/release/check-binary.sh` checks that no
binary needs GLIBC newer than 2.28 or the system's libstdc++ (MDIR's own
binaries), and that every library is bundled or one that manylinux_2_28
allows from the system.

## The Python wheels (manylinux_2_28)

The target `wheels` of `Dockerfile.manylinux` builds the wheels of the
Python package `mdir`, one for each of CPython 3.10–3.13 of the image, on
the same toolchain as the tarball, and repairs them with auditwheel to
`manylinux_2_28_x86_64` (D[python-package],
[python-package.md](../docs/python-package.md)):

```sh
DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile.manylinux \
  --build-arg MDIR_GIT_COMMIT=$(git rev-parse HEAD) \
  --target wheels --output type=local,dest=dist .
scripts/release/check-wheel.sh dist/*.whl
```

A wheel holds the extension, the runtime, libdevice with the CUDA EULA,
and HDF5 (in `mdir.libs`); cuFFT comes from NVIDIA's `nvidia-cufft` wheel
through the extra `mdir[cuda]`, and the driver's `libcuda` from the
system. The `mdir` command is not in the wheels.
