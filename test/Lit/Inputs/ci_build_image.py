"""scripts/ci/build-image.sh --dry-run: the tag it computes is the hash of
packaging/Dockerfile and the two build scripts, as ci-image.yml and ci.yml
compute it, and its options are checked (#611). No Docker is needed."""

import hashlib
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
script = root / "scripts" / "ci" / "build-image.sh"
data = b"".join((root / p).read_bytes() for p in
                ("packaging/Dockerfile", "scripts/build-llvm.sh", "scripts/build-hdf5.sh"))
expected = "ghcr.io/vv137/mdir-ci:llvm-23.1.2-" + hashlib.sha256(data).hexdigest()[:16]


def run(*args):
    return subprocess.run(["bash", str(script), *args], text=True,
                          capture_output=True, env={"PATH": "/usr/bin:/bin", "USER": "test"})


dry = run("--dry-run", "--push", "--source", str(root))
print(dry.stdout, end="")
lines = dict(line.split(": ", 1) for line in dry.stdout.splitlines())
print("the tag of ci-image.yml:", "yes" if lines.get("image") == expected else "NO")
for args in (["--bogus"], ["--source"], ["--dry-run"]):
    result = run(*args)
    print(f"{' '.join(args)}: status {result.returncode}, {result.stderr.strip()}")
