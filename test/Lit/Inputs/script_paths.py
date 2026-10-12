"""script_paths.py ROOT: fails if a tracked file of the scripts, the examples,
the packaging, the tools, the tests, or the build files of the repository
at ROOT names a path of one machine (#289, test/Lit/script-paths.test).

What a machine provides (the compilers and libraries, the reference engines,
where to write, the device) comes from the environment
(docs/environment.md, "Scripts"), so that a script runs on any machine that
provides it. A path of one machine is, here:

  - a user's directory on a home, a store, or a local disk, written out;
  - anything under the home directory (~, $HOME, expanduser);
  - a CUDA toolkit with its version in its name.

EXCEPTIONS lists the lines that may name one, each with its reason: an
example for a reader, the default prefix of a script that installs, or a
path inside a container image. An exception that no line uses any longer is
an error as well, so that the list stays what the repository holds.

The files are those that git tracks under ROOT; in a tree without git (a
source archive), every file under the same directories.
"""
import fnmatch
import os
import re
import subprocess
import sys

# README.md and docs/ are prose for a reader and are not searched: their
# install prefixes are examples, and docs/decisions.md and the paper record
# where a measurement was made.
SEARCHED = ["scripts", "examples", "cmake", "packaging", "tools", "test",
            "CMakeLists.txt", "pyproject.toml"]

PATTERNS = [
    ("a user's directory",
     re.compile(r"(?<![\w.~}])/(home|store|runs|mypool|var/tmp|Users)/\w")),
    ("a path under the home directory",
     re.compile(r"(~|\$HOME|\$\{HOME\})/[\w.]|expanduser\(")),
    ("a CUDA toolkit with its version",
     re.compile(r"/usr/local/cuda-\d")),
]

# (files, what the line holds, why it stays).
EXCEPTIONS = [
    ("scripts/build-hdf5.sh", r'HDF5_ROOT="\$\{HDF5_ROOT:-\$HOME/opt/hdf5\}"',
     "the default prefix of a script that installs: nothing has to exist "
     "there, and the README builds with the same prefix"),
    ("scripts/build-llvm.sh", r'LLVM_ROOT="\$\{LLVM_ROOT:-\$HOME/opt/llvm\}"',
     "the default prefix of a script that installs, as for HDF5"),
    ("scripts/benchmarks/amber/bench.py", r'expanduser\("~/mdir-benchmarks"\)',
     "the default of MDIR_BENCH_DIR, a work directory that the script "
     "makes: of no machine in particular (left out of question by #289)"),
    ("scripts/benchmarks/amber/README.md", r"or `~/mdir-benchmarks`",
     "the same default, documented"),
    ("scripts/paper/run-suite-repeats.sh", r"\$HOME/mdir-benchmarks",
     "the same default, where the script reads what bench.py wrote"),
    ("packaging/Dockerfile.manylinux", r"ln -sfn /usr/local/cuda-13\.0 /usr/local/cuda",
     "a path inside the container image: the link that the recipe makes "
     "to the toolkit it installed"),
]


def files(root):
    if os.path.exists(os.path.join(root, ".git")):
        out = subprocess.run(["git", "-C", root, "ls-files", "-z", "--"] + SEARCHED,
                             capture_output=True, check=True).stdout
        return sorted(p for p in out.decode().split("\0") if p)
    found = []
    for entry in SEARCHED:
        path = os.path.join(root, entry)
        if os.path.isfile(path):
            found.append(entry)
        for directory, _, names in os.walk(path):
            found += [os.path.relpath(os.path.join(directory, n), root) for n in names]
    return sorted(found)


root = sys.argv[1]
used = [0] * len(EXCEPTIONS)
complaints = []
count = 0
for path in files(root):
    try:
        with open(os.path.join(root, path), encoding="utf-8") as file:
            lines = file.read().splitlines()
    except (UnicodeDecodeError, OSError):
        continue  # A binary input of a test, or a link that leads nowhere.
    count += 1
    for number, line in enumerate(lines, 1):
        for what, pattern in PATTERNS:
            if not pattern.search(line):
                continue
            allowed = False
            for i, (glob, text, _) in enumerate(EXCEPTIONS):
                if fnmatch.fnmatch(path, glob) and re.search(text, line):
                    used[i] += 1
                    allowed = True
            if not allowed:
                complaints.append(f"{path}:{number}: {what}: {line.strip()[:120]}")
            break
for i, (glob, text, _) in enumerate(EXCEPTIONS):
    if not used[i]:
        complaints.append(f"the exception for {glob} ({text}) matches no line: remove it")
for complaint in complaints:
    print(complaint)
if complaints:
    print(f"{len(complaints)} complaints: what a machine provides comes from "
          "the environment (docs/environment.md, Scripts)")
    sys.exit(1)
print(f"no path of one machine in {count} files, {sum(used)} lines excepted")
