"""Configures a small project with cmake/CheckFloatFromChars.cmake (#654):
with the C++ compiler of the machine, then with a wrapper of it that fails
on a source that uses std::chars_format, as the libstdc++ of GCC 9 does.

Usage: float_from_chars.py MODULE WORK
The compiler is CXX, else c++; cmake is CMAKE, else cmake on the path.
"""
import os
import pathlib
import shutil
import stat
import subprocess
import sys

module = pathlib.Path(sys.argv[1]).resolve()
work = pathlib.Path(sys.argv[2]).resolve()
work.mkdir(parents=True)
cmake = os.environ.get("CMAKE") or shutil.which("cmake")
cxx = shutil.which(os.environ.get("CXX") or "c++")
project = work / "project"
project.mkdir()
(project / "CMakeLists.txt").write_text(
    "cmake_minimum_required(VERSION 3.20)\nproject(probe LANGUAGES CXX)\n"
    'set(CMAKE_CXX_STANDARD 17)\n'
    f'include("{module}")\nmdir_check_float_from_chars()\n')

wrapper = work / "old-c++"
wrapper.write_text(
    "#!/bin/sh\n"
    "# A compiler whose library has no floating std::from_chars.\n"
    'for a in "$@"; do\n'
    '  case $a in *.cxx|*.cpp|*.cc)\n'
    '    if grep -q chars_format "$a" 2>/dev/null; then\n'
    '      echo "$a: error: \'std::chars_format\' has not been declared" >&2; exit 1\n'
    '    fi;;\n'
    '  esac\n'
    'done\n'
    f'exec "{cxx}" "$@"\n')
wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)


def configure(name, compiler):
    return subprocess.run([cmake, "-S", str(project), "-B", str(work / name),
                           f"-DCMAKE_CXX_COMPILER={compiler}"],
                          capture_output=True, text=True)


good = configure("good", cxx)
assert good.returncode == 0, good.stdout + good.stderr
print("the compiler of the machine: configured")
old = configure("old", wrapper)
assert old.returncode != 0, old.stdout
print("a library without it: refused")
text = " ".join((old.stdout + old.stderr).split())  # CMake wraps its messages
for phrase in ("MDIR needs a C++ standard library with the floating-point std::from_chars of C++17 (std::chars_format)",
               "use GCC 11 or later, or a Clang whose libc++ or libstdc++ has it"):
    assert phrase in text, text
    print(phrase)
