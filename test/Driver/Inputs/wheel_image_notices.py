"""What scripts/release/check-wheel.sh and check-image.sh say about the
notices of a wheel and of the container image (issue #221).

    wheel_image_notices.py wheel|image <source root> <work directory>

Builds stand-ins, with any ELF file for the binaries: wheels (zip files of
the layout of D228) or root file systems of the image's layout. It prints
the complaints about notices for each: none for a complete one, one for
each notice taken away or emptied, and none for HDF5's and the OpenMP
runtime's when their libraries are absent. The other checks of the scripts
(GLIBC, the needed libraries), which the stand-in binaries may fail, are not
looked at.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

kind, root, work = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
version = re.search(r"project\(mdir VERSION ([0-9.]+) ",
                    (root / "CMakeLists.txt").read_text()).group(1)
# HDF5's and NVIDIA's files are not in the repository.
HDF5 = "Copyright 2006 by The HDF Group.\n"
NVIDIA = "End User License Agreement, NVIDIA Corporation\n"
SHARED = {
    "pocketfft-LICENSE": root / "third_party/pocketfft/LICENSE.md",
    "tomlplusplus-LICENSE": root / "third_party/tomlplusplus/LICENSE",
    "LLVM-LICENSE.TXT": root / "packaging/licenses/LLVM-LICENSE.TXT",
    "OpenMP-LICENSE.TXT": root / "packaging/licenses/OpenMP-LICENSE.TXT",
    "HDF5-COPYING": HDF5,
}
if kind == "wheel":
    check = root / "scripts/release/check-wheel.sh"
    NOTICES = {"mdir/licenses/LICENSE": root / "LICENSE",
               "mdir/licenses/pybind11-LICENSE": root / "python/pybind11-LICENSE"}
    NOTICES.update({f"mdir/licenses/{k}": v for k, v in SHARED.items()})
    NOTICES["mdir/cuda/EULA.txt"] = NVIDIA
    LIBRARIES = ("mdir/_core.cpython-312-x86_64-linux-gnu.so",
                 "mdir/lib/libmdrt.so", "mdir/lib/libmdrt_cuda.so",
                 "mdir/lib/libomp.so", "mdir.libs/libhdf5-0a1b2c3d.so.310.5.1")
    OPTIONAL = ("mdir/lib/libomp.so", "mdir.libs/libhdf5-0a1b2c3d.so.310.5.1")
    FILES = {
        "mdir/__init__.py": "",
        "mdir/_frames.py": "",
        "mdir/torch.py": "",
        "mdir/cuda/nvvm/libdevice/libdevice.10.bc": "BC",
        f"mdir-{version}.dist-info/licenses/LICENSE": root / "LICENSE",
        f"mdir-{version}.dist-info/METADATA":
            f"Metadata-Version: 2.4\nName: mdir\nVersion: {version}\n"
            "License-Expression: MIT\nLicense-File: LICENSE\n"
            "Provides-Extra: cuda\n"
            'Requires-Dist: nvidia-cufft<13,>=12.0; extra == "cuda"\n'
            'Requires-Dist: nvidia-cuda-nvcc<14,>=13.0; extra == "cuda"\n',
    }
    MARKS = ("licenses", "EULA", "grafted")
else:
    check = root / "scripts/release/check-image.sh"
    NOTICES = {"opt/mdir/share/mdir/LICENSE": root / "LICENSE"}
    NOTICES.update({f"opt/mdir/share/mdir/licenses/{k}": v
                    for k, v in SHARED.items()})
    NOTICES["usr/local/cuda/EULA.txt"] = NVIDIA
    NOTICES["NGC-DL-CONTAINER-LICENSE"] = "NVIDIA DEEP LEARNING CONTAINER LICENSE\n"
    LIBRARIES = ("opt/mdir/bin/mdir", "opt/mdir/lib/libmdrt.so",
                 "opt/mdir/lib/libmdrt_cuda.so", "opt/mdir/lib/libomp.so",
                 "opt/hdf5/1.14.6/lib/libhdf5.so.310")
    OPTIONAL = ("opt/mdir/lib/libomp.so", "opt/hdf5/1.14.6/lib/libhdf5.so.310")
    FILES = {"usr/local/cuda/nvvm/libdevice/libdevice.10.bc": "BC"}
    MARKS = ("LICENSE", "licenses", "EULA")


def put(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, str):
        target.write_text(source)
    else:
        shutil.copy(source, target)


def stand_in(name, without=(), other=None, extra=()):
    """A tree, and for a wheel the zip file of it, which is returned."""
    top = work / name
    shutil.rmtree(top, ignore_errors=True)
    for path in LIBRARIES + tuple(extra):
        if path not in without:
            put(top / path, pathlib.Path(sys.executable))
    for path, source in {**FILES, **NOTICES}.items():
        if path not in without:
            put(top / path, source)
    for path, text in (other or {}).items():
        (top / path).write_text(text)
    if kind == "image":
        return top
    wheel = work / name / f"mdir-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in sorted(top.rglob("*")):
            if path.is_file() and path != wheel:
                archive.write(path, path.relative_to(top))
    return wheel


def report(label, target):
    result = subprocess.run(["bash", str(check), str(target)], text=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    lines = [line.split(": ", 1)[-1].replace(target.name + ": ", "")
             for line in result.stderr.splitlines()
             if any(mark in line for mark in MARKS) and "FAILED" not in line]
    print(f"{label}: exit {result.returncode}, {len(lines)}"
          + (": " + "; ".join(lines) if lines else ""))


result = subprocess.run(["bash", str(check), str(stand_in("complete"))],
                        text=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
count = sum(any(mark in line for mark in MARKS) and "FAILED" not in line
            for line in result.stderr.splitlines())
print(f"complete: {count} complaints about notices")
for path in NOTICES:
    report(f"without {path}", stand_in("without", without=(path,)))
hdf5 = next(path for path in NOTICES if path.endswith("HDF5-COPYING"))
llvm = next(path for path in NOTICES if path.endswith("LLVM-LICENSE.TXT"))
openmp = next(path for path in NOTICES if path.endswith("OpenMP-LICENSE.TXT"))
report("empty HDF5-COPYING", stand_in("empty", other={hdf5: ""}))
report("another text as LLVM-LICENSE.TXT",
       stand_in("other", other={llvm: "another text\n"}))
# Without the libraries, their notices are not asked for.
report("no libhdf5 and no libomp, without their notices",
       stand_in("unbundled", without=OPTIONAL + (hdf5, openmp)))
if kind == "wheel":
    # A library that auditwheel grafted and no notice is known for.
    report("libaec grafted",
           stand_in("grafted", extra=("mdir.libs/libaec-1f2e3d4c.so.0.1.3",)))
    # A License-File of the metadata that the wheel does not hold.
    report("License-File not in .dist-info",
           stand_in("metadata", without=(
               f"mdir-{version}.dist-info/licenses/LICENSE",)))
