#!/usr/bin/env bash
# Build and install the HDF5 release that MDIR writes checkpoints with.
#
# Usage:
#   scripts/build-hdf5.sh
#
# The default is the last release of the 1.14 series. The releases from 2.0
# on need CMake 3.26 or later.
#
# Environment variables (all optional):
#   HDF5_VERSION  Release to build.
#   HDF5_ROOT     Directory that holds the source tree and the install tree.
#   HDF5_SRC      Source directory. Downloaded if it does not exist.
#   HDF5_BUILD    Build directory. Prefer a local disk over a network mount.
#   HDF5_PREFIX   Install prefix.
#   JOBS          Parallel compile jobs.

set -euo pipefail

HDF5_VERSION="${HDF5_VERSION:-1.14.6}"
HDF5_ROOT="${HDF5_ROOT:-$HOME/opt/hdf5}"
HDF5_SRC="${HDF5_SRC:-$HDF5_ROOT/src/hdf5-$HDF5_VERSION}"
HDF5_BUILD="${HDF5_BUILD:-$HDF5_ROOT/build/$HDF5_VERSION}"
HDF5_PREFIX="${HDF5_PREFIX:-$HDF5_ROOT/$HDF5_VERSION}"
JOBS="${JOBS:-$(nproc)}"

log() { printf '[build-hdf5 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
else
    GENERATOR="Unix Makefiles"
fi

log "version: $HDF5_VERSION"
log "source:  $HDF5_SRC"
log "build:   $HDF5_BUILD"
log "prefix:  $HDF5_PREFIX"

if [ -f "$HDF5_PREFIX/include/hdf5.h" ]; then
    log "HDF5 is already installed"
    exit 0
fi

if [ ! -f "$HDF5_SRC/CMakeLists.txt" ]; then
    log "downloading"
    mkdir -p "$(dirname "$HDF5_SRC")"
    ARCHIVE="$(dirname "$HDF5_SRC")/hdf5-$HDF5_VERSION.tar.gz"
    # The releases before 2.0 have tags of the form hdf5_1.14.6.
    case "$HDF5_VERSION" in
        1.*) TAG="hdf5_$HDF5_VERSION" ;;
        *)   TAG="$HDF5_VERSION" ;;
    esac
    curl -fsSL -o "$ARCHIVE" \
        "https://github.com/HDFGroup/hdf5/releases/download/$TAG/hdf5-$HDF5_VERSION.tar.gz"

    # The archive holds one directory. Unpack it next to the source
    # directory and give it its name.
    UNPACKED="$(mktemp -d "$(dirname "$HDF5_SRC")/unpack.XXXXXX")"
    tar -xzf "$ARCHIVE" -C "$UNPACKED"
    rm -rf "$HDF5_SRC"
    mv "$UNPACKED"/hdf5-* "$HDF5_SRC"
    rmdir "$UNPACKED"
fi

# The C library alone, without tools, tests, or compression.
log "configuring"
cmake -G "$GENERATOR" -S "$HDF5_SRC" -B "$HDF5_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$HDF5_PREFIX" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DHDF5_BUILD_TOOLS=OFF \
    -DHDF5_BUILD_EXAMPLES=OFF \
    -DHDF5_BUILD_HL_LIB=OFF \
    -DHDF5_BUILD_CPP_LIB=OFF \
    -DHDF5_BUILD_FORTRAN=OFF \
    -DHDF5_BUILD_JAVA=OFF \
    -DHDF5_ENABLE_PARALLEL=OFF \
    -DHDF5_ENABLE_ZLIB_SUPPORT=OFF \
    -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF \
    -DHDF5_ENABLE_SZIP_SUPPORT=OFF

log "building"
cmake --build "$HDF5_BUILD" --parallel "$JOBS"

log "installing"
cmake --install "$HDF5_BUILD"

log "done: $HDF5_PREFIX"
