"""segments_trajectory.py MODEL INTEGRATOR ENSEMBLE INPUTS WORK: the short
trajectories of #795, the units against the general path
(D[md-alg-segments-writer]; D417, Section 8, S4a).

Three waters near each other in a cubic cell of 4 nm (OPC: Inputs/opc/three,
from three.py; TIP3P: the three waters of Inputs/triclinic nearest the first,
as #795's water_template_tip3p.py takes them), Coulomb and Lennard-Jones cut
off at 0.9 nm, 1 fs, rigid water. One step of the general path in double
precision constrains the input; from that state three routes run 100 steps:
the list of units on the test route (MDIR_RIGID_UNITS_TEST=1, mixed), the
general path in mixed precision, and the general path in double precision
and deterministic. Every 10 steps, #795's declared bounds:

  - positions (each to the image nearest the double route's) and
    velocities of the units within 3 times the general mixed route's
    largest and RMS differences from the double route, or 1e-5 of the edge
    of the cell (of the largest speed) where those are smaller;
  - the volume within 3 times the mixed route's difference, or 1e-5 of it.

The potential energy's difference from the double route is printed with 3
times the mixed route's, or 1e-5 of its magnitude, for information: #795
held energies to bounds at the same frame, which the contract of D421
holds the schedule to.

MODEL is OPC or TIP3P, INTEGRATOR VELOCITY_VERLET or LEAPFROG, ENSEMBLE NVE
or NPT (V-rescale and C-rescale, both every 10 steps, as #795's). Prints
each sample and a last line that names the case.
"""
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
    return mdir.Simulation(program)


initial = simulation("initial")
initial.run(1, energy=True)
start = initial.state()
routes = ("units", "mixed", "double")
sims = {r: simulation(r, start) for r in routes}


def image(d, box):
    return d - box * np.round(d / box)


def error(a, b):
    d = np.abs(a - b)
    return float(d.max()), float(np.sqrt(np.mean(d * d)))


failed = False
for sample in range(10):
    s = {}
    for r in routes:
        sims[r].run(10, energy=True)
        s[r] = sims[r].state()
    ref = s["double"]
    box = np.asarray(ref.cell.diagonal)
    x0 = np.asarray(ref.positions)
    v0 = np.asarray(ref.velocities)
    xe, ve = {}, {}
    for r in ("units", "mixed"):
        x = x0 + image(np.asarray(s[r].positions) - x0, box)
        xe[r] = error(x, x0)
        ve[r] = error(np.asarray(s[r].velocities), v0)
    xs, vs = float(box.max()), max(1.0, float(np.abs(v0).max()))
    xb = (max(3 * xe["mixed"][0], 1e-5 * xs), max(3 * xe["mixed"][1], 1e-5 * xs))
    vb = (max(3 * ve["mixed"][0], 1e-5 * vs), max(3 * ve["mixed"][1], 1e-5 * vs))
    volume = {r: s[r].energies["volume"] for r in routes}
    vd = abs(volume["units"] - volume["double"])
    vbound = max(3 * abs(volume["mixed"] - volume["double"]), 1e-5 * volume["double"])
    u = {r: s[r].energies["potential"] for r in routes}
    ud = abs(u["units"] - u["double"])
    ub = max(3 * abs(u["mixed"] - u["double"]), 1e-5 * abs(u["double"]))  # printed only
    ok = (xe["units"][0] <= xb[0] and xe["units"][1] <= xb[1] and ve["units"][0] <= vb[0]
          and ve["units"][1] <= vb[1] and vd <= vbound)
    failed |= not ok
    print(f"step {ref.step}: x max {xe['units'][0]:.3g} / {xb[0]:.3g} RMS {xe['units'][1]:.3g} / {xb[1]:.3g}; "
          f"v max {ve['units'][0]:.3g} / {vb[0]:.3g} RMS {ve['units'][1]:.3g} / {vb[1]:.3g}; "
          f"volume {vd:.3g} / {vbound:.3g}; potential (for information) {ud:.3g} / {ub:.3g}; "
          f"mixed x {xe['mixed'][0]:.3g} v {ve['mixed'][0]:.3g}{'' if ok else '  BEYOND'}")
name = f"{model} {integrator} {ensemble}"
if failed:
    print(f"{name}: the units beyond the declared bounds")
    sys.exit(1)
print(f"{name}: 10 samples of 100 steps within the declared bounds")
