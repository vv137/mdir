"""scripts/packaging/build.sh --dry-run and --print-tag (#639): the command of
each target, the tag of the CI image as the hash of packaging/Dockerfile and
the two build scripts, and the refusals. No container program is needed:
the source is a copy of the files the script reads, outside any git tree, so
the commit is given."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
script = root / "scripts" / "packaging" / "build.sh"
tree = work / "tree"
for path in ("packaging/Dockerfile", "scripts/build-llvm.sh", "scripts/build-hdf5.sh", "CMakeLists.txt"):
    (tree / path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / path, tree / path)
cmake = (tree / "CMakeLists.txt").read_text().replace(
    (tree / "CMakeLists.txt").read_text().split("project(mdir VERSION ")[1].split(" ")[0], "9.9.9", 1)
(tree / "CMakeLists.txt").write_text(cmake)
data = b"".join((tree / p).read_bytes() for p in
                ("packaging/Dockerfile", "scripts/build-llvm.sh", "scripts/build-hdf5.sh"))
expected = "ghcr.io/vv137/mdir-ci:llvm-23.1.2-" + hashlib.sha256(data).hexdigest()[:16]
env = {"PATH": "/usr/bin:/bin", "USER": "test", "WORK": str(work / "w")}


def run(*args):
    return subprocess.run(["bash", str(script), *args], text=True, capture_output=True, env=env)


def show(result):
    return result.stdout.replace(str(tree), "TREE").replace(str(work), "WORK")


for target, extra in (("runtime", []), ("tarball", []), ("wheels", ["--out", str(work / "wheels")]),
                      ("ci", ["--push"])):
    result = run(target, "--dry-run", "--source", str(tree), "--commit", "abc123", *extra)
    print(f"== {target}: status {result.returncode}")
    print(show(result), end="")
tag = run("ci", "--print-tag", "--source", str(tree)).stdout.strip()
print("the tag of ci.yml and ci-image.yml:", "yes" if tag == expected else f"NO ({tag})")
for args in (["bogus"], ["ci", "--bogus"], ["tarball", "--push"], ["runtime", "--print-tag"],
             ["ci", "--tag", "x"], ["runtime", "--out", "x"], ["ci", "--login"],
             ["ci", "--source", ".", "--clone", "main"], ["ci", "--keep"]):
    result = run(*args)
    message = result.stderr.strip().splitlines()[0].replace(str(script), "build.sh")
    print(f"{' '.join(args)}: status {result.returncode}, {message}")
