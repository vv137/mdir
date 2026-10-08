#!/usr/bin/env bash
# Checks that an extracted release tree runs on manylinux_2_28 systems
# (D177):
#
#   scripts/release/check-binary.sh DIR
#
# For bin/mdir and every lib/*.so*:
#  - no symbol needs GLIBC newer than 2.28;
#  - mdir and the runtime of MDIR need no GLIBCXX or CXXABI symbol (libstdc++
#    is linked statically); a bundled third-party library may need the
#    system libstdc++ only up to GLIBCXX_3.4.25 and CXXABI_1.3.11, the
#    manylinux_2_28 limits;
#  - every needed library is bundled in lib/ or is one that manylinux_2_28
#    lets a binary take from the system (libc, libm, libdl, libpthread,
#    librt, ld-linux, libgcc_s, libstdc++, libz) or the driver's libcuda.
# It also checks that share/mdir/cuda holds libdevice, and that the tree
# carries the notices that the licenses of what it distributes ask for
# (packaging/licenses/README.md): MDIR's license; in share/mdir/licenses,
# HDF5's beside a bundled libhdf5, pocketfft's (in libmdrt), toml++'s (in
# mdir), and LLVM's and its OpenMP runtime's (in mdir, and libomp.so); and
# the CUDA EULA beside a bundled cuFFT or libdevice. Exit status 1 on any
# failure, with each one listed.
set -euo pipefail
dir=${1:?usage: $0 DIR}
fail=0
complain() { echo "check-binary: $*" >&2; fail=1; }

allowed='^(libc|libm|libdl|libpthread|librt|libgcc_s|libstdc\+\+|libz|libcuda)\.so\.[0-9]+$|^ld-linux-x86-64\.so\.2$|^linux-vdso\.so\.1$'
max_version() { { grep -o "$1_[0-9.]*" || true; } | sed "s/$1_//" | sort -V | tail -1; }
newer() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]]; }

files=("$dir/bin/mdir")
for f in "$dir"/lib/*.so*; do [[ -f "$f" && ! -L "$f" ]] && files+=("$f"); done
for f in "${files[@]}"; do
  name=$(basename "$f")
  symbols=$(objdump -T "$f" 2>/dev/null || true)
  glibc=$(max_version GLIBC <<<"$symbols")
  [[ -n "$glibc" ]] && newer "$glibc" 2.28 && complain "$name needs GLIBC_$glibc (> 2.28)"
  glibcxx=$(max_version GLIBCXX <<<"$symbols")
  cxxabi=$(max_version CXXABI <<<"$symbols")
  case $name in
    mdir|libmdrt.so|libmdrt_cuda.so)
      [[ -n "$glibcxx$cxxabi" ]] && complain "$name needs the system libstdc++ (GLIBCXX_$glibcxx CXXABI_$cxxabi); link it statically" ;;
    *)
      [[ -n "$glibcxx" ]] && newer "$glibcxx" 3.4.25 && complain "$name needs GLIBCXX_$glibcxx (> 3.4.25)"
      [[ -n "$cxxabi" ]] && newer "$cxxabi" 1.3.11 && complain "$name needs CXXABI_$cxxabi (> 1.3.11)" ;;
  esac
  for needed in $(objdump -p "$f" | awk '/NEEDED/ {print $2}'); do
    [[ "$needed" =~ $allowed ]] && continue
    [[ -e "$dir/lib/$needed" ]] && continue
    complain "$name needs $needed, which is neither bundled nor allowed"
  done
  printf '%-28s GLIBC %-6s GLIBCXX %-8s CXXABI %s\n' "$name" "${glibc:--}" "${glibcxx:--}" "${cxxabi:--}"
done
[[ -f "$dir/share/mdir/cuda/nvvm/libdevice/libdevice.10.bc" ]] || complain "share/mdir/cuda has no libdevice"
# The notices: each a file that is not empty and holds a line of its text.
notice() {
  local file=$dir/share/mdir/$1
  if [[ ! -s "$file" ]]; then
    complain "share/mdir/$1 is missing: $3"
  elif ! grep -q -- "$2" "$file"; then
    complain "share/mdir/$1 does not hold '$2'"
  fi
}
exists() { compgen -G "$1" > /dev/null; }
notice LICENSE 'MIT License' "MDIR's license"
notice licenses/pocketfft-LICENSE 'Max-Planck-Society' "pocketfft is compiled into libmdrt.so"
notice licenses/tomlplusplus-LICENSE 'Mark Gillard' "toml++ is compiled into mdir"
notice licenses/LLVM-LICENSE.TXT 'LLVM Exceptions to the Apache 2.0 License' "LLVM is linked into mdir"
if exists "$dir/lib/libomp.so*"; then
  notice licenses/OpenMP-LICENSE.TXT 'Intel Corporation' "lib/libomp.so is bundled"
fi
if exists "$dir/lib/libhdf5*.so*"; then
  notice licenses/HDF5-COPYING 'The HDF Group' "lib/libhdf5 is bundled"
fi
if exists "$dir/lib/libcufft.so*" || [[ -d "$dir/share/mdir/cuda" ]]; then
  notice cuda/EULA.txt 'NVIDIA' "cuFFT or libdevice is bundled"
fi
if (( fail )); then echo "check-binary: FAILED" >&2; exit 1; fi
echo "check-binary: ok"
