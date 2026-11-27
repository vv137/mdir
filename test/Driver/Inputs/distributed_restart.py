"""Compare a continued checkpoint and its frames with uninterrupted CPU runs."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py

root = Path(sys.argv[1]).resolve()
example = Path(sys.argv[2]).resolve()
program = shutil.which(sys.argv[3]) or sys.argv[3]
mpi = sys.argv[4:]
root.mkdir(parents=True, exist_ok=True)
if example.is_dir():
    control_file = example / "argon.toml"
    coordinates = example / "argon.pdb"
    skin_edge = 8.65
else:
    control_file = example
    coordinates = example.parent / "Inputs" / "mixture.pdb"
    skin_edge = 9.01

base = control_file.read_text()
base = re.sub(r"(?ms)^\[execution\].*", "", base)
base = re.sub(r"(?ms)^\[\[energy.pair_override\]\].*?(?=^\[\[energy.type\]\])", "", base)
base = re.sub(r"(?m)^dispersion_correction.*\n", "", base)
base = re.sub(r"(?m)^coordinates\s*=.*$", f'coordinates = "{coordinates}"', base)
base = re.sub(r"(?m)^steps\s*=.*$", "steps = 10", base)
base = re.sub(r"(?m)^pairlist_distance\s*=.*$", f"pairlist_distance = {skin_edge}", base)
if not re.search(r"(?m)^seed\s*=", base):
    base = base.replace("[dynamics]", "[dynamics]\nseed = 314159")
particles = sum(line.startswith(("HETATM", "ATOM  ")) for line in coordinates.read_text().splitlines())
output = """[output]
trajectory = "traj.h5"
trajectory_format = "H5MD"
trajectory_interval = 5
energy = "energy.dat"
energy_interval = 5
checkpoint = "state.h5"
checkpoint_interval = 5
manifest = "manifest.json"
"""
base = re.sub(r"(?ms)^\[output\].*?(?=^\[energy\])", output, base)
base += """
[execution]
target = "CPU"
threads = 1
precision = "DOUBLE"
deterministic = true
[distribute]
"""


def run(name, domains, ranks, *, stop=False, resume=False, trace=False):
    path = root / name
    path.mkdir(exist_ok=True)
    transport = 'mpi' if ranks and mpi else 'local'
    control = base + f'domains = {domains}\ntransport = "{transport}"\n'
    (path / "run.toml").write_text(control)
    cmd = [*mpi, "-n", str(ranks)] if transport == "mpi" else []
    cmd.extend([program, "run"])
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
    assert len(setup) == domains and sum(d["owned"] for d in setup) == particles
    assert all(d["import_volume_angstrom3"] > 0 for d in setup)
    assert all(d["position_import_bytes"] == 24*d["ghosts"] for d in setup)
    return path


def grid(path):
    """The grid that the fingerprint of the checkpoint names."""
    with h5py.File(path / "state.h5") as state:
        text = state["parameters/mdir/fingerprint/execution"][()].decode()
    return [e for e in text.splitlines() if e.startswith("[distribute] domains")]


def compare(a, b):
    assert (a / "traj.h5").read_bytes() == (b / "traj.h5").read_bytes()
    # Continuations increment the checkpoint part even for an identical
    # physical state, and the fingerprint names the grid of the run.
    same = (a / "resume.log").exists() == (b / "resume.log").exists()
    if same and grid(a) == grid(b):
        assert (a / "state.h5").read_bytes() == (b / "state.h5").read_bytes()
    assert (a / "energy.dat").read_bytes() == (b / "energy.dat").read_bytes()
    for filename in ["state.h5", "traj.h5"]:
        with h5py.File(a / filename) as ref, h5py.File(b / filename) as current:
            datasets = []
            ref.visititems(lambda name, obj: datasets.append(name)
                           if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'iuf' else None)
            assert datasets
            if filename == "state.h5":
                meta = ref["parameters/mdir"]
                assert meta.attrs["seed"] == current["parameters/mdir"].attrs["seed"] == 314159
                names = []
                current.visit(names.append)
                words = {"domain", "domains", "grid", "owner", "ownership", "ghosts", "epoch"}
                assert not any(words.intersection(path.split("/")) for path in names)
                # The fingerprints differ in the grid alone.
                for group in ["physics", "coupling", "execution"]:
                    key = f"parameters/mdir/fingerprint/{group}"
                    x, y = ([e for e in h[key][()].decode().splitlines()
                             if not e.startswith("[distribute] domains")]
                            for h in (ref, current))
                    assert x == y, (a.name, b.name, group)
            other_datasets = []
            current.visititems(lambda name, obj: other_datasets.append(name)
                               if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'iuf' else None)
            assert other_datasets == datasets
            for name in datasets:
                x, y = ref[name][()], current[name][()]
                assert x.dtype == y.dtype and x.shape == y.shape
                assert x.tobytes() == y.tobytes(), (a.name, b.name, filename, name)


ranks = 2 if mpi else 0
reference = run("whole", 2, ranks)
traced = run("traced", 2, ranks, trace=True)
compare(reference, traced)
compare(reference, run("whole-16", 16, ranks))
seed = run("stop", 2, ranks, stop=True)
for domains, count in [(2, 1 if mpi else 0), (1, 1 if mpi else 0), (8, 2 if mpi else 0)]:
    name = f"continued-{domains}"
    shutil.copytree(seed, root / name)
    resumed = run(name, domains, count, resume=True)
    compare(reference, resumed)
    compare(root / "continued-2", resumed)
    if domains != 2:
        target = run(f"whole-{domains}", domains, count)
        compare(target, resumed)
seed8 = run("stop-8", 8, ranks, stop=True)
shutil.copytree(seed8, root / "from-8")
from_eight = run("from-8", 2, ranks, resume=True)
compare(reference, from_eight)
compare(root / "continued-2", from_eight)
print("all checkpoint and trajectory datasets agree bitwise across restarts")
