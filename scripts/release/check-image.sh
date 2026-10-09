#!/usr/bin/env bash
# Checks that the container image of packaging/Dockerfile carries the
# notices that the licenses of what it distributes in binary form ask for
# (packaging/licenses/README.md), as check-binary.sh does for the tarball
# and check-wheel.sh for the wheels:
#
#   scripts/release/check-image.sh IMAGE     an image of the local Docker
#   scripts/release/check-image.sh DIR       a root file system, extracted
#
# With an image, the parts that the check reads (/opt/mdir, /opt/hdf5,
# libdevice and the EULA of /usr/local/cuda, and the license of the base
# image) are copied out of a container made from it, which is not started.
# The notices, each a file that is not empty and holds a line of its text:
#  - /opt/mdir/share/mdir/LICENSE, MDIR's license;
#  - in /opt/mdir/share/mdir/licenses: pocketfft's (in libmdrt.so), toml++'s
#    and LLVM's (in mdir), the OpenMP runtime's beside /opt/mdir/lib/libomp.so,
#    and HDF5's beside a libhdf5 in /opt/hdf5 or /opt/mdir/lib;
#  - /usr/local/cuda/EULA.txt beside libdevice, which the recipe copies from
#    the devel image;
#  - /NGC-DL-CONTAINER-LICENSE, the license of NVIDIA's base image, which
#    covers its CUDA libraries (cuFFT among them).
# Exit status 1 on any failure, with each one listed.
set -euo pipefail
target=${1:?usage: $0 IMAGE|DIR}
fail=0
complain() { echo "check-image: $*" >&2; fail=1; }

if [[ -d "$target" ]]; then
  root=$target
else
  root=$(mktemp -d)
  container=$(docker create "$target")
  trap 'docker rm "$container" > /dev/null; rm -rf "$root"' EXIT
  mkdir -p "$root/opt" "$root/usr/local/cuda"
  docker cp -q "$container:/opt/mdir" "$root/opt/mdir" \
    || complain "$target has no /opt/mdir"
  docker cp -q "$container:/opt/hdf5" "$root/opt/hdf5" 2> /dev/null || true
  # /usr/local/cuda is a symbolic link in NVIDIA's images.
  docker cp -q -L "$container:/usr/local/cuda/nvvm" "$root/usr/local/cuda/nvvm" 2> /dev/null || true
  docker cp -q -L "$container:/usr/local/cuda/EULA.txt" "$root/usr/local/cuda/EULA.txt" 2> /dev/null || true
  docker cp -q "$container:/NGC-DL-CONTAINER-LICENSE" "$root/NGC-DL-CONTAINER-LICENSE" 2> /dev/null || true
fi

notice() {
  if [[ ! -s "$root/$1" ]]; then
    complain "/$1 is missing: $3"
  elif ! grep -q -- "$2" "$root/$1"; then
    complain "/$1 does not hold '$2'"
  fi
}
exists() { compgen -G "$1" > /dev/null; }
licenses=opt/mdir/share/mdir/licenses
for f in opt/mdir/bin/mdir opt/mdir/lib/libmdrt.so; do
  [[ -f "$root/$f" ]] || complain "/$f is missing"
done
notice opt/mdir/share/mdir/LICENSE 'MIT License' "MDIR's license"
notice $licenses/pocketfft-LICENSE 'Max-Planck-Society' "pocketfft is compiled into libmdrt.so"
notice $licenses/tomlplusplus-LICENSE 'Mark Gillard' "toml++ is compiled into mdir"
notice $licenses/LLVM-LICENSE.TXT 'LLVM Exceptions to the Apache 2.0 License' "LLVM is linked into mdir"
if exists "$root/opt/mdir/lib/libomp.so*"; then
  notice $licenses/OpenMP-LICENSE.TXT 'Intel Corporation' "/opt/mdir/lib/libomp.so is in the image"
fi
if exists "$root/opt/hdf5/*/lib/libhdf5*.so*" || exists "$root/opt/mdir/lib/libhdf5*.so*"; then
  notice $licenses/HDF5-COPYING 'The HDF Group' "libhdf5 is in the image"
fi
if exists "$root/usr/local/cuda/nvvm/libdevice/libdevice.*.bc"; then
  notice usr/local/cuda/EULA.txt 'NVIDIA' "libdevice is copied into the image"
  notice NGC-DL-CONTAINER-LICENSE 'NVIDIA' "the image is derived from NVIDIA's CUDA image"
fi
if (( fail )); then echo "check-image: FAILED" >&2; exit 1; fi
echo "check-image: ok"
