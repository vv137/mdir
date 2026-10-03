# Containers

The recipes here build MDIR with CUDA in a container. A container carries
LLVM/MLIR 23.1.2, HDF5 1.14.6, and MDIR, so a workstation or a cluster node
needs only Docker or Apptainer and the NVIDIA driver.

| File | Builds |
|---|---|
| [Dockerfile](Dockerfile) | A Docker image, in stages: system packages, HDF5, LLVM with MLIR and the OpenMP runtime, MDIR; the final stage holds only the installed tree on CUDA's runtime image |
| [mdir.def](mdir.def) | An Apptainer (Singularity) image made from the Docker image |

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
