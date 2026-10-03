# syntax=docker/dockerfile:1
#
# The container of MDIR: LLVM/MLIR 23.1.2, HDF5, and MDIR built with CUDA,
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
# Build arguments: CUDA_VERSION and UBUNTU choose the base images (the CUDA
# of the image must not be newer than the host driver supports); JOBS and
# LINK_JOBS bound the parallel compile and link jobs of LLVM.

ARG CUDA_VERSION=13.0.1
ARG UBUNTU=ubuntu24.04

# --- LLVM, MLIR, the OpenMP runtime, and HDF5 --------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS toolchain
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git python3 ca-certificates curl \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
ARG JOBS=16
ARG LINK_JOBS=4
COPY scripts/build-llvm.sh scripts/build-hdf5.sh /src/scripts/
# Only the targets MDIR compiles for: the host and NVIDIA GPUs.
RUN LLVM_ROOT=/opt/llvm LLVM_TARGETS="X86;NVPTX" JOBS=${JOBS} \
      LINK_JOBS=${LINK_JOBS} /src/scripts/build-llvm.sh \
 && rm -rf /opt/llvm/src /opt/llvm/build
RUN HDF5_ROOT=/opt/hdf5 JOBS=${JOBS} /src/scripts/build-hdf5.sh \
 && rm -rf /opt/hdf5/src /opt/hdf5/build

# --- MDIR ------------------------------------------------------------------
FROM toolchain AS build
# The context has no .git; the commit comes from a build argument (above).
ARG MDIR_GIT_COMMIT=
ENV MDIR_GIT_COMMIT=${MDIR_GIT_COMMIT}
COPY . /src/mdir
RUN cmake -G Ninja -S /src/mdir -B /build/mdir \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=/opt/llvm/23.1.2/lib/cmake/llvm \
      -DMLIR_DIR=/opt/llvm/23.1.2/lib/cmake/mlir \
      -DHDF5_ROOT=/opt/hdf5/1.14.6 \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DMDIR_ENABLE_CUDA=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/mdir \
 && cmake --build /build/mdir --target mdir mdrt mdrt_cuda \
 && cmake --install /build/mdir

# --- Runtime -----------------------------------------------------------------
# The runtime image of CUDA carries cuFFT; the kernels also take the math
# functions of libdevice, from CUDA_ROOT, which only the devel image has.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
COPY --from=build /opt/mdir /opt/mdir
COPY --from=build /opt/hdf5/1.14.6/lib /opt/hdf5/1.14.6/lib
COPY --from=build /usr/local/cuda/nvvm/libdevice /usr/local/cuda/nvvm/libdevice
COPY --from=build /usr/local/cuda/version.json /usr/local/cuda/version.json
COPY examples /opt/mdir/share/mdir/examples
ENV PATH=/opt/mdir/bin:${PATH} \
    CUDA_ROOT=/usr/local/cuda
WORKDIR /work
ENTRYPOINT ["mdir"]
CMD ["doctor"]
