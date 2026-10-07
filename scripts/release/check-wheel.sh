#!/usr/bin/env bash
# Checks that wheels of the Python package run on manylinux_2_28 systems
# (D228), as check-binary.sh does for the tarball (D177):
#
#   scripts/release/check-wheel.sh WHEEL...
#
# For each wheel:
#  - its tag is cpXY-cpXY-manylinux_2_28_x86_64 for CPython 3.10-3.13;
#  - no ELF file needs a symbol of GLIBC newer than 2.28;
#  - the extension and the runtime of MDIR need no GLIBCXX or CXXABI symbol
#    (libstdc++ is linked statically); a grafted library may need the
#    system libstdc++ only up to GLIBCXX_3.4.25 and CXXABI_1.3.11;
#  - every needed library is in mdir/lib or mdir.libs, is one that
#    manylinux_2_28 lets a binary take from the system, is the driver's
#    libcuda, or is cuFFT (libcufft.so.12, from NVIDIA's wheel, the extra
#    `cuda`);
#  - the extra `cuda` of its metadata requires nvidia-cufft and
#    nvidia-cuda-nvcc (ptxas), and nothing else;
#  - mdir/cuda holds libdevice and the CUDA EULA, and mdir/licenses the
#    notices of MDIR, HDF5, and pybind11;
#  - the version of its metadata is that of project() in CMakeLists.txt.
# Exit status 1 on any failure, with each one listed.
set -euo pipefail
(( $# )) || { echo "usage: $0 WHEEL..." >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
version=$(sed -n 's/^project(mdir VERSION \([0-9.]*\) .*/\1/p' "$repo/CMakeLists.txt")
fail=0
complain() { echo "check-wheel: $*" >&2; fail=1; }

allowed='^(libc|libm|libdl|libpthread|librt|libgcc_s|libstdc\+\+|libz|libcuda|libcufft)\.so\.[0-9]+$|^ld-linux-x86-64\.so\.2$|^linux-vdso\.so\.1$'
max_version() { { grep -o "$1_[0-9.]*" || true; } | sed "s/$1_//" | sort -V | tail -1; }
newer() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]]; }

for wheel in "$@"; do
  name=$(basename "$wheel")
  echo "== $name"
  [[ "$name" =~ ^mdir-([0-9.]+)-(cp31[0-3])-(cp31[0-3])-manylinux_2_28_x86_64\.whl$ ]] \
    || complain "$name: not a manylinux_2_28 wheel for CPython 3.10-3.13"
  [[ "${BASH_REMATCH[1]:-}" == "$version" ]] \
    || complain "$name: version ${BASH_REMATCH[1]:-?}, CMakeLists.txt says $version"
  tree=$(mktemp -d)
  trap 'rm -rf "$tree"' EXIT
  python3 -m zipfile -e "$wheel" "$tree"
  grep -qx "Version: $version" "$tree"/mdir-*.dist-info/METADATA \
    || complain "$name: METADATA does not say Version: $version"
  extra=$(sed -n 's/^Requires-Dist: \(.*\); extra == "cuda"$/\1/p' "$tree"/mdir-*.dist-info/METADATA | sort | tr '\n' ' ')
  [[ "$extra" == "nvidia-cuda-nvcc<14,>=13.0 nvidia-cufft<13,>=12.0 " ]] \
    || complain "$name: the extra cuda requires '$extra', not nvidia-cuda-nvcc<14,>=13.0 and nvidia-cufft<13,>=12.0"
  while IFS= read -r -d '' f; do
    file "$f" | grep -q ELF || continue
    base=$(basename "$f")
    symbols=$(objdump -T "$f" 2>/dev/null || true)
    glibc=$(max_version GLIBC <<<"$symbols")
    [[ -n "$glibc" ]] && newer "$glibc" 2.28 && complain "$base needs GLIBC_$glibc (> 2.28)"
    glibcxx=$(max_version GLIBCXX <<<"$symbols")
    cxxabi=$(max_version CXXABI <<<"$symbols")
    case $base in
      _core.*|libmdrt.so|libmdrt_cuda.so)
        [[ -n "$glibcxx$cxxabi" ]] && complain "$base needs the system libstdc++ (GLIBCXX_$glibcxx CXXABI_$cxxabi)" ;;
      *)
        [[ -n "$glibcxx" ]] && newer "$glibcxx" 3.4.25 && complain "$base needs GLIBCXX_$glibcxx (> 3.4.25)"
        [[ -n "$cxxabi" ]] && newer "$cxxabi" 1.3.11 && complain "$base needs CXXABI_$cxxabi (> 1.3.11)" ;;
    esac
    for needed in $(objdump -p "$f" | awk '/NEEDED/ {print $2}'); do
      [[ "$needed" =~ $allowed ]] && continue
      [[ -e "$tree/mdir/lib/$needed" || -e "$tree/mdir.libs/$needed" ]] && continue
      complain "$base needs $needed, which is neither in the wheel nor allowed"
    done
    printf '%-44s GLIBC %-6s GLIBCXX %-8s CXXABI %s\n' "${f#$tree/}" "${glibc:--}" "${glibcxx:--}" "${cxxabi:--}"
  done < <(find "$tree" -type f -name '*.so*' -print0)
  for f in mdir/__init__.py mdir/lib/libmdrt.so mdir/lib/libmdrt_cuda.so mdir/lib/libomp.so \
           mdir/cuda/nvvm/libdevice/libdevice.10.bc mdir/cuda/EULA.txt \
           mdir/licenses/LICENSE mdir/licenses/HDF5-COPYING \
           mdir/licenses/pybind11-LICENSE; do
    [[ -f "$tree/$f" ]] || complain "$name has no $f"
  done
  rm -rf "$tree"
  trap - EXIT
done
if (( fail )); then echo "check-wheel: FAILED" >&2; exit 1; fi
echo "check-wheel: ok"
