# syntax=docker/dockerfile:1
#
# The container of MDIR: HDF5, LLVM/MLIR 23.1.2, and MDIR built with CUDA,
# and a small runtime image with the installed tree. Build from the root of
# the repository:
#
#   docker build -f packaging/Dockerfile -t mdir:0.1.0 \
#     --build-arg MDIR_GIT_COMMIT=$(git rev-parse HEAD) .
#
# and run on a GPU with the NVIDIA container toolkit:
#
#   docker run --rm --gpus all -v "$PWD":/work mdir:0.1.0 run md.toml
#
# Layers go from what changes least to what changes most, so that a rebuild
# reuses the cached ones:
#   1. the base image (pinned by digest) and the system packages;
#   2. HDF5, from scripts/build-hdf5.sh alone;
#   3. LLVM, MLIR, and the OpenMP runtime, from scripts/build-llvm.sh alone,
#      the slowest layer (about an hour with 32 jobs);
#   4. the sources of MDIR, last.
# Each stage copies only the files it needs: a change to MDIR's sources
# leaves the HDF5 and LLVM layers cached, and a rebuild then compiles MDIR
# alone, incrementally, in a build tree kept in a BuildKit cache mount.
# MDIR has no package manifest of its own: its dependencies are exactly the
# two scripts above and the CUDA of the base image.
#
# Build arguments: JOBS and LINK_JOBS bound the parallel compile and link
# jobs of LLVM; MDIR_GIT_COMMIT names the commit, since the context holds no
# .git. To change CUDA, change both digests below together (the CUDA of the
# image must not be newer than the host driver supports).

# nvidia/cuda:13.0.1-devel-ubuntu24.04 and -runtime-ubuntu24.04.
ARG DEVEL=nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
ARG RUNTIME=nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7

# --- 1. System packages ------------------------------------------------------
FROM ${DEVEL} AS base
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git python3 ca-certificates curl \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# --- 2. HDF5 1.14.6 ----------------------------------------------------------
FROM base AS hdf5
ARG JOBS=16
COPY scripts/build-hdf5.sh /src/scripts/build-hdf5.sh
RUN HDF5_ROOT=/opt/hdf5 HDF5_VERSION=1.14.6 JOBS=${JOBS} \
      /src/scripts/build-hdf5.sh \
 && rm -rf /opt/hdf5/src /opt/hdf5/build

# --- 3. LLVM, MLIR, and the OpenMP runtime 23.1.2 ----------------------------
FROM hdf5 AS toolchain
ARG JOBS=16
ARG LINK_JOBS=4
COPY scripts/build-llvm.sh /src/scripts/build-llvm.sh
# Only the targets MDIR compiles for: the host and NVIDIA GPUs.
RUN LLVM_ROOT=/opt/llvm LLVM_VERSION=23.1.2 LLVM_TARGETS="X86;NVPTX" \
      JOBS=${JOBS} LINK_JOBS=${LINK_JOBS} /src/scripts/build-llvm.sh \
 && rm -rf /opt/llvm/src /opt/llvm/build

# --- 4. MDIR -----------------------------------------------------------------
FROM toolchain AS build
ARG MDIR_GIT_COMMIT=
ENV MDIR_GIT_COMMIT=${MDIR_GIT_COMMIT}
COPY . /src/mdir
# The build tree stays in a cache mount between builds, so that a change of
# a few sources recompiles only those; the installed tree is in the layer.
RUN --mount=type=cache,target=/build/mdir,id=mdir-build \
    cmake -G Ninja -S /src/mdir -B /build/mdir \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=/opt/llvm/23.1.2/lib/cmake/llvm \
      -DMLIR_DIR=/opt/llvm/23.1.2/lib/cmake/mlir \
      -DHDF5_ROOT=/opt/hdf5/1.14.6 \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DMDIR_ENABLE_CUDA=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/mdir \
 && cmake --build /build/mdir --target mdir mdrt mdrt_cuda \
 && rm -rf /opt/mdir \
 && cmake --install /build/mdir

# --- Runtime -----------------------------------------------------------------
# The runtime image of CUDA carries cuFFT; the kernels also take the math
# functions of libdevice, from CUDA_ROOT, which only the devel image has.
# The libraries and libdevice come before the installed tree, which changes
# with every build.
FROM ${RUNTIME}
COPY --from=hdf5 /opt/hdf5/1.14.6/lib /opt/hdf5/1.14.6/lib
COPY --from=base /usr/local/cuda/nvvm/libdevice /usr/local/cuda/nvvm/libdevice
ENV PATH=/opt/mdir/bin:${PATH} \
    CUDA_ROOT=/usr/local/cuda
COPY examples /opt/mdir/share/mdir/examples
COPY --from=build /opt/mdir /opt/mdir
WORKDIR /work
ENTRYPOINT ["mdir"]
CMD ["doctor"]
