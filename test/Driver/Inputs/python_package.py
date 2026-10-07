"""The package `mdir` of a build tree (D[python-package]).

python_package.py SOURCE_ROOT VERSION_FILE (the output of `mdir version`)

Checks that `mdir` is the package of the build tree with the extension
`mdir._core`; that its classes and exceptions are named `mdir.X`; that its
version is that of `mdir version`, of project() in CMakeLists.txt, of the
regex by which pyproject.toml reads it, and of the package that the ala3
example declares; and that importing it maps no CUDA library.
"""
import pathlib
import re
import sys

import mdir

root = pathlib.Path(sys.argv[1])
cli = pathlib.Path(sys.argv[2]).read_text().splitlines()[0].split()[-1]
here = pathlib.Path(mdir.__file__).resolve().parent
assert here.name == "mdir" and (here / "__init__.py").is_file(), here
assert mdir._core.__name__ == "mdir._core", mdir._core.__name__
assert pathlib.Path(mdir._core.__file__).resolve().parent == here
for name in ("InputError", "UnsupportedError", "CompileError",
             "StaleProgramError", "SimulationError", "System", "Simulation",
             "Target", "Precision"):
    assert getattr(mdir, name).__module__ == "mdir", (name, getattr(mdir, name).__module__)

cmake = (root / "CMakeLists.txt").read_text()
project = re.search(r"^project\(mdir VERSION ([0-9.]+) ", cmake, re.M).group(1)
pyproject = (root / "pyproject.toml").read_text()
regex = re.search(r"^regex = '(.*)'$", pyproject, re.M).group(1)
read = re.search(regex, cmake).group("value")
header = re.search(r'^# dependencies = \["mdir\[cuda\]==([0-9.]+)"',
                   (root / "examples/ala3/run.py").read_text(), re.M).group(1)
versions = {"mdir.__version__": mdir.__version__, "mdir version": cli,
            "project()": project, "pyproject.toml": read, "ala3 header": header}
assert len(set(versions.values())) == 1, versions
print("one version:", mdir.__version__)

maps = pathlib.Path("/proc/self/maps").read_text()
loaded = [n for n in ("libcuda.so", "libcufft", "libnvidia-ml", "libmdrt_cuda") if n in maps]
assert not loaded, loaded
print("import mdir maps no CUDA library")
