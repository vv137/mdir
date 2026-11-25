"""Compare a continued checkpoint and its frames with uninterrupted CPU runs."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

root = Path(sys.argv[1]).resolve()
example = Path(sys.argv[2]).resolve()
mpi = sys.argv[3:]
root.mkdir(parents=True, exist_ok=True)
base = (example / "argon.toml").read_text().replace("argon.pdb", str(example / "argon.pdb"))
base = re.sub(r"(?m)^steps\s*=.*$", "steps = 10", base)
base = base.replace("= 100", "= 5").replace("= 16", "= 1").replace("= 9.5", "= 8.65")
base = base.replace('trajectory          = "argon.dcd"', '''trajectory = "traj.h5"
trajectory_format = "H5MD"
checkpoint = "state.h5"
checkpoint_interval = 5
manifest = "manifest.json"''')
base += "\ndeterministic = true\n[distribute]\n"


def run(name, domains, ranks, *, stop=False, resume=False, trace=False):
    path = root / name
    path.mkdir(exist_ok=True)
    transport = 'mpi' if ranks and mpi else 'local'
    control = base + f'domains = {domains}\ntransport = "{transport}"\n'
    (path / "run.toml").write_text(control)
    cmd = [*mpi, "-n", str(ranks)] if transport == "mpi" else []
    cmd.extend(["mdir", "run"])
    if resume or stop:
        cmd.append("--continue")
    if stop:
        cmd.extend(["--max-walltime", "0.0000001"])
    env = os.environ.copy()
    if trace:
        env["MDIR_DIST_STEP_TRACE"] = str(path / "trace")
    with (path / ("resume.log" if resume else "run.log")).open("w") as log:
        result = subprocess.run(cmd + ["run.toml"], cwd=path, env=env, stdout=log)
    assert result.returncode == (75 if stop else 0), (name, result.returncode)
    if trace:
        for rank in range(max(1, ranks)):
            detail = (path / f"trace.rank{rank}.log").read_text()
            steps = re.findall(rf"^DIST: trace step (\d+) rank {rank} wall ", detail, re.M)
            assert list(map(int, steps)) == list(range(1, 11)), steps
    events = [json.loads(line) for line in (path / "manifest.json").read_text().splitlines()]
    setup = [event["domains"] for event in events if event["event"] == "domain_setup"][-1]
    assert len(setup) == domains and sum(d["owned"] for d in setup) == 864
    assert all(d["position_import_bytes"] == 24*d["ghosts"] for d in setup)
    return path


def compare(a, b, *, exact):
    for filename in ["state.h5", "traj.h5"]:
        with h5py.File(a / filename) as ref, h5py.File(b / filename) as current:
            datasets = []
            ref.visititems(lambda name, obj: datasets.append(name)
                           if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'iuf' else None)
            assert datasets
            for name in datasets:
                x, y = ref[name][()], current[name][()]
                same = x.tobytes() == y.tobytes()
                if exact or x.dtype.kind != "f":
                    assert same, (filename, name)
                else:
                    err = float(np.max(np.abs(x-y), initial=0))
                    print(b.name, filename, name, "unequal", np.count_nonzero(x != y),
                          "max_abs", err)
                    assert err < 1e-9, (filename, name, err)


ranks = 2 if mpi else 0
reference = run("whole", 2, ranks)
traced = run("traced", 2, ranks, trace=True)
compare(reference, traced, exact=True)
seed = run("stop", 2, ranks, stop=True)
for domains, count in [(2, 1 if mpi else 0), (1, 1 if mpi else 0), (8, 2 if mpi else 0)]:
    name = f"continued-{domains}"
    shutil.copytree(seed, root / name)
    resumed = run(name, domains, count, resume=True)
    compare(reference, resumed, exact=(domains == 2))
    if domains != 2:
        target = run(f"whole-{domains}", domains, count)
        compare(target, resumed, exact=False)
print("trace and fixed-grid restart exact; changed-grid errors below 1e-9")
