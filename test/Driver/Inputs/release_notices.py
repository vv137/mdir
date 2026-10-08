"""What scripts/release/check-binary.sh says about the notices of a release
tree (issue #194).

    release_notices.py <source root> <work directory>

Builds trees of the tarball's layout, with any ELF file for the binaries,
and prints the complaints about notices for each: none for a complete
tree, one for each notice taken away or emptied, and none for HDF5's when
no libhdf5 is bundled. The other checks of the script (GLIBC, the needed
libraries), which the stand-in binaries may fail, are not looked at.
"""
import pathlib
import shutil
import subprocess
import sys

root, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
check = root / "scripts/release/check-binary.sh"
NOTICES = {
    "LICENSE": root / "LICENSE",
    "licenses/pocketfft-LICENSE": root / "third_party/pocketfft/LICENSE.md",
    "licenses/tomlplusplus-LICENSE": root / "third_party/tomlplusplus/LICENSE",
    "licenses/LLVM-LICENSE.TXT": root / "packaging/licenses/LLVM-LICENSE.TXT",
    "licenses/OpenMP-LICENSE.TXT": root / "packaging/licenses/OpenMP-LICENSE.TXT",
    # HDF5's and NVIDIA's files are not in the repository.
    "licenses/HDF5-COPYING": "Copyright 2006 by The HDF Group.\n",
    "cuda/EULA.txt": "End User License Agreement, NVIDIA Corporation\n",
}


def tree(name, without=(), empty=(), libraries=("libhdf5.so.310", "libomp.so",
                                                "libcufft.so.12")):
    top = work / name
    shutil.rmtree(top, ignore_errors=True)
    (top / "bin").mkdir(parents=True)
    (top / "lib").mkdir()
    shutil.copy(sys.executable, top / "bin" / "mdir")
    for library in libraries:
        shutil.copy(sys.executable, top / "lib" / library)
    for path, source in NOTICES.items():
        if path in without:
            continue
        target = top / "share/mdir" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path in empty:
            target.write_text("")
        elif isinstance(source, str):
            target.write_text(source)
        else:
            shutil.copy(source, target)
    return top


def complaints(top):
    result = subprocess.run(["bash", str(check), str(top)], text=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    lines = [line for line in result.stderr.splitlines()
             if "share/mdir/LICENSE" in line or "share/mdir/licenses" in line
             or "EULA" in line]
    return result.returncode, lines


status, lines = complaints(tree("complete"))
print(f"complete: {len(lines)} complaints about notices")
for path in NOTICES:
    status, lines = complaints(tree("without", without=(path,)))
    print(f"without {path}: exit {status}, {len(lines)}: "
          + "; ".join(line.split("check-binary: ", 1)[-1] for line in lines))
status, lines = complaints(tree("empty", empty=("licenses/HDF5-COPYING",)))
print(f"empty HDF5-COPYING: exit {status}, {len(lines)}: "
      + "; ".join(line.split("check-binary: ", 1)[-1] for line in lines))
status, lines = complaints(tree("other", empty=("licenses/LLVM-LICENSE.TXT",)))
(work / "other/share/mdir/licenses/LLVM-LICENSE.TXT").write_text("another text\n")
status, lines = complaints(work / "other")
print(f"another text as LLVM-LICENSE.TXT: exit {status}, {len(lines)}")
# Without the libraries, their notices are not asked for.
status, lines = complaints(tree(
    "unbundled", libraries=("libcufft.so.12",),
    without=("licenses/HDF5-COPYING", "licenses/OpenMP-LICENSE.TXT")))
print(f"no libhdf5 and no libomp, without their notices: {len(lines)} "
      "complaints about notices")
