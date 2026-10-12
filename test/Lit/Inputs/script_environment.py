"""script_environment.py ROOT WORK: runs the scripts of the repository at
ROOT that need something of the machine without it, in an environment that
holds nothing, and prints how each one stops
(test/Lit/script-environment.test, #289, scripts/environment.sh).

The path of the runs is a directory of WORK that holds bash and dirname
alone, so that no program of the machine that runs the test is found.
"""
import os
import shutil
import stat
import subprocess
import sys

root, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
shutil.rmtree(work, ignore_errors=True)
bin_dir = os.path.join(work, "bin")
os.makedirs(bin_dir)
bash = shutil.which("bash")
for program in ("bash", "dirname"):
    os.symlink(shutil.which(program), os.path.join(bin_dir, program))


def run(title, script, *arguments, **environment):
    result = subprocess.run(
        [bash, os.path.join(root, script), *arguments], cwd=work,
        env={"PATH": bin_dir, **environment}, capture_output=True, text=True)
    print(f"{title}: exit {result.returncode}")
    for line in (result.stdout + result.stderr).splitlines():
        # The scripts name themselves by their path, and bash adds a line
        # number to the message of a parameter that is not set.
        line = line.replace(root + "/", "").replace(work + "/", "WORK/")
        print("  " + line.strip())


def program(name):
    path = os.path.join(work, "tools", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write("#!/bin/sh\nexit 0\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


publish = "scripts/release/publish.sh"
run("the release, nothing set", publish, "0.1.0")
run("a dry run without the binary, nothing set", publish, "0.1.0", "--dry-run", "--no-binary")
run("the release without its binary", publish, "0.1.0", "--no-binary")
run("a dry run that publishes", publish, "0.1.0", "--dry-run", "--publish")
run("the release, an unknown option", publish, "0.1.0", "--tag")

# The file of the machine: everything the release needs, with stand-ins for
# the programs; the environment wins over the file. The script then stops
# at the device, before it reads the repository or writes anything.
for directory in ("llvm", "hdf5", "cuda", "pybind11"):
    os.makedirs(os.path.join(work, directory))
machine = os.path.join(work, "machine.env")
with open(machine, "w") as file:
    file.write("# The values of this machine.\n\n")
    file.write(f"MDIR_RELEASE_DIR={work}/release\n")
    file.write(f"MDIR_LLVM={work}/llvm\nHDF5_PREFIX={work}/hdf5\n")
    file.write(f"CUDA_ROOT={work}/cuda\nMDIR_RELEASE_GPU=0\n")
    file.write(f"PYBIND11_DIR={work}/pybind11\nMDIR_PYTHON={program('python')}\n")
    for name in ("cmake", "ninja", "lit", "pandoc", "tectonic", "docker"):
        file.write(f"{name.upper()}={program(name)}\n")
run("the release, the file of the machine and a device from the environment",
    publish, "0.1.0", MDIR_ENV_FILE=machine, MDIR_RELEASE_GPU="first")
print("  the release directory was made:", os.path.exists(os.path.join(work, "release")))
run("the release, a prefix that is no directory", publish, "0.1.0",
    MDIR_ENV_FILE=machine, HDF5_PREFIX=os.path.join(work, "none"),
    PANDOC=os.path.join(work, "tools", "no-pandoc"),
    MDIR_TORCH_PYTHON=os.path.join(work, "tools", "no-torch"))
run("a file of the machine that is missing", publish, "0.1.0",
    MDIR_ENV_FILE=os.path.join(work, "no.env"))
with open(os.path.join(work, "bad.env"), "w") as file:
    file.write("MDIR_LLVM=/x\nexport HDF5_PREFIX=/y\n")
run("a file of the machine with a line that is not NAME=value", publish, "0.1.0",
    MDIR_ENV_FILE=os.path.join(work, "bad.env"))

run("the paper, nothing set", "scripts/paper/build-pdf.sh", "out")
print("  the output directory was made:", os.path.exists(os.path.join(work, "out")))
run("the comparison with GROMACS, nothing set", "scripts/validation/gromacs/run.sh", "compare")
run("the comparison with GROMACS, a prefix that is no directory",
    "scripts/validation/gromacs/run.sh", "compare", ENGINES=os.path.join(work, "none"),
    MDIR=program("mdir"))
print("  the work directory was made:", os.path.exists(os.path.join(work, "compare")))
run("the argon example, nothing set", "examples/argon/run.sh", "argon.mlir")
run("the argon example, a prefix without LLVM", "examples/argon/run.sh", "argon.mlir",
    LLVM_PREFIX=os.path.join(work, "llvm"))
run("the experiment on neighbor structures, nothing set",
    "scripts/experiments/neighbor-structures/run.sh")
