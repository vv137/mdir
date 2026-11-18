"""segments_trajectory.py MODEL INTEGRATOR ENSEMBLE INPUTS WORK [--record=N]:
the short trajectories of #795, the units against the general path
(D[md-alg-segments-writer]; D417, Section 8, S4a).

Three waters near each other in a cubic cell of 4 nm (OPC: Inputs/opc/three,
from three.py; TIP3P: the three waters of Inputs/triclinic nearest the first,
as #795's water_template_tip3p.py takes them), Coulomb and Lennard-Jones cut
off at 0.9 nm, 1 fs, rigid water. One step of the general path in double
precision constrains the input; from that state the list of units on the
test route (MDIR_RIGID_UNITS_TEST=1, mixed) and the general path in double
precision, deterministic, run 100 steps. Every 10 steps, #795's declared
bounds:

  - the positions (each to the image nearest the double route's), the
    velocities at the step projected onto the constraints (leapfrog's
    v_{n+1/2} + (h/2) f/m with each route's carried forces, then projected,
    #795's fullVelocity), and the volume of the units within 3 times the
    general mixed route's differences from the double route, or 1e-5 of the
    edge of the cell, of the largest speed, of the volume, where those are
    smaller.

The baseline of the general mixed route is the largest of N runs at each
sample, recorded once by --record=N in Inputs/segments_trajectory_bounds.json
(5 runs, 2026-11-18): one run's baseline changes from run to run, and the
gate flickered (the leader's ruling on #906). Each sample also prints, for
the diagnosis, the largest difference of the unprojected velocities at the
step, its component along the constraints, and that of h dF/m, the
difference of the carried forces (the placements of a site's force by the
weights and by the Jacobian, D421, Section 5).

MODEL is OPC or TIP3P, INTEGRATOR VELOCITY_VERLET or LEAPFROG, ENSEMBLE NVE
or NPT (V-rescale and C-rescale, both every 10 steps, as #795's).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ["MDIR_COMPILE_CACHE"] = "off"
import mdir  # noqa: E402

model, integrator, ensemble = sys.argv[1:4]
inputs, work = Path(sys.argv[4]), Path(sys.argv[5])
assert model in ("OPC", "TIP3P") and integrator in ("VELOCITY_VERLET", "LEAPFROG")
assert ensemble in ("NVE", "NPT")
work.mkdir(parents=True, exist_ok=True)


def tip3p_files():
    source = (inputs / "triclinic" / "dodecahedron.gro").read_text().splitlines()
    count = int(source[1])
    atoms = source[2:2 + count]
    xyz = [[float(r[20 + 8 * k:28 + 8 * k]) for k in range(3)] for r in atoms]
    oxygens = xyz[::3]
    closest = sorted(range(len(oxygens)),
                     key=lambda i: sum((a - b) ** 2 for a, b in zip(oxygens[i], oxygens[0])))[:3]
    chosen = [3 * i + j for i in closest for j in range(3)]
    center = [sum(xyz[i][k] for i in chosen) / 9 for k in range(3)]
    out = ["three nearby TIP3P waters", f"{9:5d}"]
    for index, original in enumerate(chosen):
        p = [xyz[original][k] - center[k] + 2.0 for k in range(3)]
        out.append(f"{index // 3 + 1:5d}{'SOL':<5}{atoms[original][10:15]:>5}{index + 1:5d}"
                   + "".join(f"{v:8.3f}" for v in p))
    out.append("   4.00000   4.00000   4.00000")
    (work / "three.gro").write_text("\n".join(out) + "\n")
    top = (inputs / "triclinic" / "water.top").read_text()
    assert top.count("SOL 403") == 1
    (work / "three.top").write_text(top.replace("SOL 403", "SOL 3"))


def build(route):
    if model == "OPC":
        loaded = mdir.load_amber(str(inputs / "opc" / "three.prmtop"), str(inputs / "opc" / "three.inpcrd"))
    else:
        tip3p_files()
        loaded = mdir.load_gromacs(str(work / "three.top"), str(work / "three.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = 0.9
    system.pairlist_distance = 1.0
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.periodic = True
    system.rigid_water = True
    method, kind = (mdir.IntegratorMethod.VelocityVerlet, mdir.EnsembleKind.NVE) if route == "initial" else (
        mdir.IntegratorMethod.VelocityVerlet if integrator == "VELOCITY_VERLET" else mdir.IntegratorMethod.Leapfrog,
        mdir.EnsembleKind.NVE if ensemble == "NVE" else mdir.EnsembleKind.NPT)
    i, e, x = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    i.method = method
    i.timestep = 0.001
    e.kind = kind
    if kind == mdir.EnsembleKind.NPT:
        e.temperature = 300.0
        e.tau_t = 1.0
        e.thermostat = mdir.ThermostatMethod.VRescale
        e.pressure = 1.01325
        e.tau_p = 5.0
        e.coupling_period = 10
    x.target = mdir.Target.GPU
    double = route in ("initial", "double")
    # The deterministic mode takes no groups (D415).
    x.neighbor_structure = mdir.NeighborStructure.Matrix if double else mdir.NeighborStructure.Groups
    x.precision = mdir.Precision.Double if double else mdir.Precision.Mixed
    x.deterministic = double
    return system, state, i, e, x


def simulation(route, start=None):
    if route == "units":
        os.environ["MDIR_RIGID_UNITS_TEST"] = "1"
    else:
        os.environ.pop("MDIR_RIGID_UNITS_TEST", None)
    system, state, i, e, x = build(route)
    if start is not None:
        state.positions = start.positions
        state.velocities = start.velocities
        state.cell = start.cell
    program = mdir.compile(system, state, i, e, x, mdir.Schedule())
    # The route of the units builds its pair term over the segments of the
    # rigid kinds (D405).
    assert (route == "units") == ("md.sum_segment_window" in program.ir), route
    return mdir.Simulation(program), program


def project(v, x, waters, masses, box):
    """v with the components along the gradients of each rigid water's three
    constraints (O-H1, O-H2, H1-H2) removed in the metric of the masses:
    P v = v - M^-1 G^T (G M^-1 G^T)^-1 G v."""
    v = np.array(v, dtype=np.float64, copy=True)
    for o, h1, h2 in waters:
        atoms = [o, h1, h2]
        inv = np.array([1.0 / masses[a] for a in atoms])
        g = np.zeros((3, 9))
        for row, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            d = image(x[atoms[b]] - x[atoms[a]], box)
            g[row, 3 * a:3 * a + 3] = -d
            g[row, 3 * b:3 * b + 3] = d
        minv = np.repeat(inv, 3)
        lhs = (g * minv) @ g.T
        flat = v[atoms].ravel()
        lam = np.linalg.solve(lhs, g @ flat)
        v[atoms] = (flat - minv * (g.T @ lam)).reshape(3, 3)
    return v


def full_step(state, masses, dt):
    """The velocity at the step: the stored one of velocity Verlet, or, for
    leapfrog's v_{n+1/2}, v + (dt/2) F/m with the route's carried forces
    (#795's fullVelocity)."""
    v = np.asarray(state.velocities, dtype=np.float64)
    if state.velocity_offset < 0:
        f = np.asarray(state.forces, dtype=np.float64)
        inv = np.where(masses > 0, 1.0 / np.where(masses > 0, masses, 1.0), 0.0)
        v = v + 0.5 * dt * f * inv[:, None]
    return v


def image(d, box):
    return d - box * np.round(d / box)


def error(a, b):
    d = np.abs(a - b)
    return float(d.max()), float(np.sqrt(np.mean(d * d)))



initial, _ = simulation("initial")
initial.run(1, energy=True)
start = initial.state()
DT = 0.001
NAME = f"{model} {integrator} {ensemble}"
BOUNDS = inputs / "segments_trajectory_bounds.json"


def trajectory(route):
    """The states every 10 steps of 100 of a route from the start."""
    sim, program = simulation(route, start)
    topology = program.topology
    states = []
    for _ in range(10):
        sim.run(10, energy=True)
        states.append(sim.state())
    return states, np.asarray(topology.masses), np.asarray(topology.rigid_waters)


def compare(a, b, masses, waters):
    """The differences of route a from route b at a sample: positions (to the
    nearest image), velocities at the step projected onto the constraints,
    volume; and, for the diagnosis, the unprojected velocities at the step,
    their component along the constraints, and that of h (F_a - F_b) / m."""
    box = np.asarray(b.cell.diagonal)
    xb, xa = np.asarray(b.positions), np.asarray(a.positions)
    xa = xb + image(xa - xb, box)
    fa, fb = full_step(a, masses, DT), full_step(b, masses, DT)
    pa, pb = project(fa, xa, waters, masses, box), project(fb, xb, waters, masses, box)
    raw = fa - fb
    along = raw - (pa - pb)
    inv = np.where(masses > 0, 1.0 / np.where(masses > 0, masses, 1.0), 0.0)[:, None]
    if a.forces is not None and b.forces is not None:
        kick = 0.5 * DT * (np.asarray(a.forces) - np.asarray(b.forces)) * inv
        kick_along = float(np.abs(kick - project(kick, xb, waters, masses, box)).max())
    else:
        kick_along = float("nan")
    return dict(x=error(xa, xb), v=error(pa, pb),
                volume=abs(a.energies["volume"] - b.energies["volume"]),
                raw=float(np.abs(raw).max()), along=float(np.abs(along).max()),
                kick_along=kick_along,
                scale=(float(box.max()), max(1.0, float(np.abs(pb).max())), b.energies["volume"]))


if len(sys.argv) > 6 and sys.argv[6].startswith("--record="):
    # The baseline of the bounds: the largest difference of N runs of the
    # general mixed route from the double one, at each sample, recorded once.
    runs = int(sys.argv[6].split("=")[1])
    double, masses, waters = trajectory("double")
    worst = [dict(x=[0.0, 0.0], v=[0.0, 0.0], volume=0.0) for _ in range(10)]
    for _ in range(runs):
        mixed, _, _ = trajectory("mixed")
        for k in range(10):
            d = compare(mixed[k], double[k], masses, waters)
            w = worst[k]
            w["x"] = [max(w["x"][0], d["x"][0]), max(w["x"][1], d["x"][1])]
            w["v"] = [max(w["v"][0], d["v"][0]), max(w["v"][1], d["v"][1])]
            w["volume"] = max(w["volume"], d["volume"])
    table = json.loads(BOUNDS.read_text()) if BOUNDS.exists() else {}
    table[NAME] = dict(runs=runs, samples=worst)
    BOUNDS.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"{NAME}: the baseline of {runs} general mixed runs recorded")
    sys.exit(0)

baseline = json.loads(BOUNDS.read_text())[NAME]["samples"]
units, masses, waters = trajectory("units")
double, _, _ = trajectory("double")
failed = False
for k in range(10):
    d = compare(units[k], double[k], masses, waters)
    w = baseline[k]
    xs, vs, volume = d["scale"]
    xb = (max(3 * w["x"][0], 1e-5 * xs), max(3 * w["x"][1], 1e-5 * xs))
    vb = (max(3 * w["v"][0], 1e-5 * vs), max(3 * w["v"][1], 1e-5 * vs))
    volb = max(3 * w["volume"], 1e-5 * volume)
    ok = (d["x"][0] <= xb[0] and d["x"][1] <= xb[1] and d["v"][0] <= vb[0] and d["v"][1] <= vb[1]
          and d["volume"] <= volb)
    failed |= not ok
    print(f"step {units[k].step}: x max {d['x'][0]:.3g} / {xb[0]:.3g} RMS {d['x'][1]:.3g} / {xb[1]:.3g}; "
          f"projected v max {d['v'][0]:.3g} / {vb[0]:.3g} RMS {d['v'][1]:.3g} / {vb[1]:.3g}; "
          f"volume {d['volume']:.3g} / {volb:.3g}; unprojected v {d['raw']:.3g}, along the constraints "
          f"{d['along']:.3g}, h dF/m along them {d['kick_along']:.3g}{'' if ok else '  BEYOND'}")
if failed:
    print(f"{NAME}: the units beyond the declared bounds")
    sys.exit(1)
print(f"{NAME}: 10 samples of 100 steps within the declared bounds")
