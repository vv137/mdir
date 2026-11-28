"""Compare runs whose coupling step runs on the device
(`[execution] coupling_step = "DEVICE"`, the default) with runs whose
coupling step runs on the host (`"HOST"`, the path before
D[device-coupling]), and count the copies to the host."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

inputs, work, mdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
cases = sys.argv[4].split(",") if len(sys.argv) > 4 else ["all"]

DECK = """[input]
format      = "AMBER"
topology    = "{inputs}/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide.inpcrd"
{restart}
[output]
manifest = "choices.jsonl"
energy_interval = {energy}
checkpoint = "state.h5"
checkpoint_interval = {checkpoint}

[energy]
cutoff            = 8.0
pairlist_distance = 9.0
electrostatics    = "PME"

[pme]
grid  = [32, 32, 32]
order = 4

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}
seed       = 7

[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{pressure}
[thermostat]
method        = "V-RESCALE"
time_constant = 0.5
interval      = {interval}
{barostat}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "{precision}"
neighbor_structure = "{structure}"
deterministic = {deterministic}
coupling_step = "{coupling}"
{mode}
"""
BAROSTAT = """
[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = {interval}
{extra}"""


def run(name, coupling, precision="MIXED", structure="MATRIX", deterministic=True,
        ensemble="NVT", steps=300, energy=50, interval=10, mode=None,
        checkpoint=None, barostat_extra="", checkpoint_interval=None):
    directory = work / name
    directory.mkdir(parents=True)
    npt = ensemble == "NPT"
    (directory / "control.toml").write_text(DECK.format(
        inputs=inputs, steps=steps, ensemble=ensemble, interval=interval,
        pressure="pressure    = 1.0\n" if npt else "",
        barostat=BAROSTAT.format(interval=interval, extra=barostat_extra) if npt else "",
        precision=precision, structure=structure, coupling=coupling,
        deterministic=str(deterministic).lower(), energy=energy,
        checkpoint=checkpoint_interval or steps,
        mode=f'step_graph = "{mode}"' if mode else "",
        restart=f'checkpoint = "{checkpoint}"\n' if checkpoint else ""))
    environment = dict(os.environ)
    environment.pop("MDRT_SPANS", None)
    environment["MDRT_HOST_READS"] = "1"
    log = subprocess.run([mdir, "run", "control.toml"], cwd=directory, text=True,
                         env=environment, capture_output=True)
    assert log.returncode == 0, log.stdout + log.stderr
    start = json.loads((directory / "choices.jsonl").read_text().splitlines()[0])
    reads = re.search(r"host reads: (\d+) copies to the host \((\d+) bytes\), (\d+) waits",
                      log.stderr)
    with h5py.File(directory / "state.h5") as f:
        state = {k: f["particles/all/" + k][()] for k in
                 ("position/value", "velocity/value", "force/value", "box/edges/value")
                 if "particles/all/" + k in f}
        bath = f["parameters/mdir"].attrs.get("bath")
    rows = re.findall(r"^INFO: +\d+ .+$", log.stdout, re.M)
    sizes = dict((int(a), int(b)) for a, b in
                 re.findall(r"host reads of (\d+) bytes: (\d+)", log.stderr))
    return dict(choice=start["resolved_choices"]["coupling_step"], state=state, sizes=sizes,
                bath=bath, rows=rows, copies=int(reads[1]), waits=int(reads[3]),
                directory=directory)


def bitwise(a, b, what):
    for key in a["state"]:
        assert a["state"][key].tobytes() == b["state"][key].tobytes(), (what, key)
    assert a["rows"] == b["rows"], (what, a["rows"], b["rows"])
    assert len(a["rows"]) > 0, what


def wanted(case):
    return "all" in cases or case in cases


# The record: DEVICE by default on the GPU, HOST on request.
if wanted("record"):
    default = run("record-device", "DEVICE", steps=20, energy=10)
    assert default["choice"]["value"] == "DEVICE", default["choice"]
    assert default["choice"]["category"] == "rule", default["choice"]
    host = run("record-host", "HOST", steps=20, energy=10)
    assert host["choice"]["value"] == "HOST", host["choice"]
    assert host["choice"]["reason"] == "as requested", host["choice"]
    print("record: DEVICE, HOST")

# Deterministic mode, MATRIX: the checkpoint and every row bitwise those of
# the coupling step on the host, in mixed and double precision, with spans
# (the default) and with launches one by one.
if wanted("nvt"):
    for precision in ("MIXED", "DOUBLE"):
        for mode in (None, "LAUNCH"):
            tag = f"{precision.lower()}-{(mode or 'spans').lower()}-nvt"
            host = run(f"host-{tag}", "HOST", precision, mode=mode)
            device = run(f"device-{tag}", "DEVICE", precision, mode=mode)
            bitwise(host, device, tag)
            # The bath of the conserved energy, as the checkpoint keeps it.
            assert device["bath"] is not None and device["bath"] != 0.0, tag
            assert device["bath"].tobytes() == host["bath"].tobytes(), tag
            print(f"{tag}: bitwise")

# The copies to the host do not grow with the coupling steps: the same run
# of 200 and 400 steps, a row and a checkpoint at the end, couples 20 and
# 40 times. The two paths run the same trajectory bit for bit, so every
# copy that the steps make apart from the coupling (the counts that each
# build of the neighbor matrix reads) grows alike in both.
def growth(coupling, ensemble, tag):
    counts = [run(f"reads-{tag}-{coupling.lower()}-{steps}", coupling, steps=steps,
                  energy=steps, ensemble=ensemble) for steps in (200, 400)]
    sizes = set(counts[0]["sizes"]) | set(counts[1]["sizes"])
    return {size: counts[1]["sizes"].get(size, 0) - counts[0]["sizes"].get(size, 0)
            for size in sizes}, counts[1]["waits"] - counts[0]["waits"]


def check_reads(ensemble, tag, kept=()):
    host, host_waits = growth("HOST", ensemble, tag)
    device, device_waits = growth("DEVICE", ensemble, tag)
    removed = {size: host.get(size, 0) - device.get(size, 0) for size in host}
    # The copy of the results of the sums, one for each coupling, is gone.
    assert host.get(8192, 0) >= 20 and device.get(8192, 0) == 0, (host, device)
    # Nothing that the device path copies grows unless the host path's
    # copies of that size grow as much.
    for size, more in device.items():
        if size not in kept:
            assert more == 0 or more == host.get(size, 0), (size, host, device)
    print(f"reads-{tag}: 20 more couplings, host {sum(host.values())} more copies "
          f"and {host_waits} more waits, device {sum(device.values())} and "
          f"{device_waits}; removed {dict(sorted(removed.items()))}")
    return host, device


if wanted("reads-nvt"):
    check_reads("NVT", "nvt")

# A continuation from a checkpoint at step 150, with rows every 40 steps:
# the bath goes to the device from the checkpoint.
if wanted("continuation"):
    pair = []
    for coupling in ("HOST", "DEVICE"):
        first = run(f"first-{coupling.lower()}", coupling, steps=150, energy=50)
        pair.append(run(f"continued-{coupling.lower()}", coupling, steps=160, energy=40,
                        checkpoint=first["directory"] / "state.h5"))
    bitwise(pair[0], pair[1], "continuation")
    print("continuation: bitwise")

# GROUPS, which is not deterministic: as close to the host's coupling as two
# runs of the host are to each other.
for ensemble in ("NVT", "NPT"):
    if not wanted("groups"):
        break
    runs = [run(f"groups-{ensemble}-{c.lower()}-{k}", c, structure="GROUPS",
                deterministic=False, steps=40, energy=20, ensemble=ensemble)
            for c, k in (("HOST", 0), ("HOST", 1), ("DEVICE", 0))]

    def difference(a, b, key):
        x, y = a["state"][key], b["state"][key]
        return np.linalg.norm(x - y) / np.linalg.norm(x)

    for key in ("position/value", "velocity/value"):
        spread = difference(runs[0], runs[1], key)
        change = difference(runs[0], runs[2], key)
        assert change <= max(10 * spread, 1e-5), (ensemble, key, change, spread)
    print(f"groups-{ensemble.lower()}: close")

# NPT, stochastic cell rescaling in each coupling: bitwise the host's.
if wanted("npt"):
    extras = {"isotropic": "", "semi": 'coupling = "SEMI_ISOTROPIC"\n',
              "anisotropic": 'coupling = "ANISOTROPIC"\n'}
    for precision in ("MIXED", "DOUBLE"):
        for name, extra in extras.items():
            if precision == "DOUBLE" and name != "isotropic":
                continue
            tag = f"{precision.lower()}-{name}-npt"
            host = run(f"host-{tag}", "HOST", precision, ensemble="NPT", barostat_extra=extra)
            device = run(f"device-{tag}", "DEVICE", precision, ensemble="NPT",
                         barostat_extra=extra)
            bitwise(host, device, tag)
            print(f"{tag}: bitwise")

if wanted("reads-npt"):
    host, device = check_reads("NPT", "npt", kept=(64,))
    # What the host still reads at a scaling: the pressures, in one small
    # copy, which give the new cell that the next step's launches take.
    assert device.get(64, 0) == 20, device

# TOWERS, which has no deterministic mode: within ten times the spread of
# two runs of the host's coupling step with launches one by one, with spans
# and with graph replay (D439, D450), in NVT and NPT. The copies of the
# virial that the dispatch observes for a scaling are gone too.
TOWERS = Path(__file__).with_name("step-graph.toml.in").read_text()


def towers(name, coupling, mode, ensemble, steps=200, energy=100, streams=None):
    directory = work / name
    directory.mkdir(parents=True)
    control = TOWERS.format(inputs=inputs, mode=mode, steps=steps, period=10)
    control = control.replace("energy_interval = 10", f"energy_interval = {energy}")
    control = control.replace("checkpoint_interval = 10", f"checkpoint_interval = {energy}")
    control = control.replace('step_graph = "', f'coupling_step = "{coupling}"\nstep_graph = "')
    if streams:
        control += f'term_streams = "{streams}"\n'

    if ensemble == "NPT":
        control = control.replace('ensemble    = "NVT"', 'ensemble    = "NPT"\npressure    = 500.0')
        control += '\n[barostat]\nmethod = "C-RESCALE"\ntime_constant = 0.1\n'
    (directory / "control.toml").write_text(control)
    environment = dict(os.environ)
    environment.pop("MDRT_SPANS", None)
    environment["MDRT_HOST_READS"] = "1"
    log = subprocess.run([mdir, "run", "control.toml"], cwd=directory, text=True,
                         env=environment, capture_output=True)
    assert log.returncode == 0, (name, log.stdout[-3000:], log.stderr[-3000:])
    with h5py.File(directory / "state.h5") as f:
        group = f["particles/all"]
        state = {k: group[k + "/value"][()] for k in ("position", "velocity")}
    rows = [row.split() for row in re.findall(r"^INFO: +(\d+ .+)$", log.stdout, re.M)]
    sizes = dict((int(a), int(b)) for a, b in
                 re.findall(r"host reads of (\d+) bytes: (\d+)", log.stderr))
    return dict(state=state, rows=rows, sizes=sizes)


def same_rows(reference, device, what):
    """The rows fall on the same steps, and their values agree within the
    tolerances of check_term_streams_graph.py."""
    assert [r[0] for r in device["rows"]] == [r[0] for r in reference["rows"]], what
    assert len(device["rows"]) > 0, what
    a = np.array([r[1:] for r in reference["rows"]], dtype=float)
    b = np.array([r[1:] for r in device["rows"]], dtype=float)
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=0.05, err_msg=str(what))


if wanted("towers"):
    for ensemble in ("NVT", "NPT"):
        reference = towers(f"towers-{ensemble}-host-0", "HOST", "LAUNCH", ensemble)
        again = towers(f"towers-{ensemble}-host-1", "HOST", "LAUNCH", ensemble)
        for mode in ("LAUNCH", "SPANS", "GRAPH"):
            device = towers(f"towers-{ensemble}-device-{mode}", "DEVICE", mode, ensemble)
            same_rows(reference, device, (ensemble, mode))
            report = []
            for key in ("position", "velocity"):
                x = reference["state"][key]
                spread = np.linalg.norm(again["state"][key] - x) / np.linalg.norm(x)
                change = np.linalg.norm(device["state"][key] - x) / np.linalg.norm(x)
                assert change <= max(10 * spread, 1e-5), (ensemble, mode, key, change, spread)
                report.append(f"{key} {change:.1e} (spread {spread:.1e})")
            print(f"towers-{ensemble.lower()}-{mode.lower()}: close; " + ", ".join(report))
        # Per coupling, from runs of 200 and 400 steps with one row: the
        # sums (8 kB), the observed virial (24 bytes) and the reciprocal
        # virial of PME (56 bytes) stay on the device; a scaling reads its
        # pressures (64 bytes).
        grown = {}
        for coupling in ("HOST", "DEVICE"):
            pair = [towers(f"towers-{ensemble}-{coupling}-{steps}", coupling, "SPANS",
                           ensemble, steps=steps, energy=steps) for steps in (200, 400)]
            sizes = set(pair[0]["sizes"]) | set(pair[1]["sizes"])
            grown[coupling] = {k: pair[1]["sizes"].get(k, 0) - pair[0]["sizes"].get(k, 0)
                               for k in sizes if pair[1]["sizes"].get(k, 0) != pair[0]["sizes"].get(k, 0)}
        print(f"towers-{ensemble.lower()} reads for 20 more couplings: host "
              f"{dict(sorted(grown['HOST'].items()))}, device {dict(sorted(grown['DEVICE'].items()))}")
        assert grown["DEVICE"] == ({64: 20} if ensemble == "NPT" else {}), grown
        assert grown["HOST"].get(8192, 0) >= 20, grown

# The reciprocal sum on the second stream (term_streams, D449), whose
# virial a scaling reads on the device after a join: NPT within the spread
# of the host's coupling step, the rows' values too, and the copies.
if wanted("towers-streams"):
    for streams in ("RECIPROCAL", "TERMS"):
        reference = towers(f"streams-{streams}-host-0", "HOST", "LAUNCH", "NPT", streams=streams)
        again = towers(f"streams-{streams}-host-1", "HOST", "LAUNCH", "NPT", streams=streams)
        for mode in ("LAUNCH", "SPANS"):
            device = towers(f"streams-{streams}-device-{mode}", "DEVICE", mode, "NPT",
                            streams=streams)
            same_rows(reference, device, (streams, mode))
            for key in ("position", "velocity"):
                x = reference["state"][key]
                spread = np.linalg.norm(again["state"][key] - x) / np.linalg.norm(x)
                change = np.linalg.norm(device["state"][key] - x) / np.linalg.norm(x)
                assert change <= max(10 * spread, 1e-5), (streams, mode, key, change, spread)
            print(f"towers-npt-{streams.lower()}-{mode.lower()}: close")
        grown = {}
        for coupling in ("HOST", "DEVICE"):
            pair = [towers(f"streams-{streams}-{coupling}-{steps}", coupling, "SPANS", "NPT",
                           steps=steps, energy=steps, streams=streams) for steps in (200, 400)]
            sizes = set(pair[0]["sizes"]) | set(pair[1]["sizes"])
            grown[coupling] = {k: pair[1]["sizes"].get(k, 0) - pair[0]["sizes"].get(k, 0)
                               for k in sizes if pair[1]["sizes"].get(k, 0) != pair[0]["sizes"].get(k, 0)}
        print(f"towers-npt-{streams.lower()} reads for 20 more couplings: host "
              f"{dict(sorted(grown['HOST'].items()))}, device {dict(sorted(grown['DEVICE'].items()))}")
        assert grown["DEVICE"] == {64: 20}, grown
