# Containers

The recipes here build MDIR with CUDA in a container. A container carries
LLVM/MLIR 23.1.2, HDF5 1.14.6, and MDIR, so a workstation or a cluster node
needs only Docker or Apptainer and the NVIDIA driver.

| File | Builds |
|---|---|
| [Dockerfile](Dockerfile) | A Docker image. The `toolchain` stage builds LLVM, MLIR, the OpenMP runtime, and HDF5; `build` builds and installs MDIR; the final stage holds only the installed tree on CUDA's runtime image |
| [mdir.def](mdir.def) | The same build as an Apptainer (Singularity) image, in two stages |

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
| `CUDA_VERSION` | `13.0.1` | The CUDA of the base images. It must not be newer than the host driver supports: `nvidia-smi` prints the highest version the driver supports. |
| `UBUNTU` | `ubuntu24.04` | The distribution of the base images |
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

On a cluster, convert the Docker image, or build from a clean tree with the
definition file:

```sh
apptainer build mdir-0.1.0.sif docker-daemon://mdir:0.1.0
apptainer build mdir-0.1.0.sif packaging/mdir.def    # from a fresh clone
```

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
  and `nvvm/libdevice` with `version.json` copied from the devel image, with
  `CUDA_ROOT=/usr/local/cuda`. The kernels are compiled at the start of a
  run and take their math functions from libdevice.
