"""Same-frame forces and energies of the route of the rigid units
(D[rigid-units-route]; #795's validation table, moved to the key).

  rigid_units_frames.py NAME TOPOLOGY COORDINATES FRAMES STEPS [--pme] [--npt]
                        [--oracle] [--after-input] [--cutoff=nm] [--pairlist=nm] [--pruned=nm]

A trajectory of `rigid_units = ON` in mixed precision gives FRAMES frames,
the first the input itself (before any constraint, whose covering radii the
resolution widens where the input lies beyond them) and then one every STEPS
steps. At each frame three programs evaluate the forces and the potential at
the same positions and cell: the units (ON, mixed), and the general route in
mixed and in double precision. The units must agree with the general double
route within 1.5 times the general mixed route's own difference, or 1e-5 of
the largest force (the forces) and 1e-6 of the potential (the energy), where
those are larger (#795's criterion).

--after-input leaves the input out of the comparison, for a site that is
not linear in its frame (Amber's extra point): the input lies off the
constraints, where the weights of D403 and the topology's placement put the
site in different places, so the two routes evaluate different positions
there (D421, Section 5); from the first step on the frame is constrained.

The forces are compared projected onto the constraints of the rigid waters
(P a = a - M^-1 G^T (G M^-1 G^T)^-1 G a for a = F/m, then m P a): the units
move the force of a site to its frame by the constant weights of D403, the
general route by the Jacobian of its placement, and the two differ along the
constraints alone where the site is not linear in its frame (D421,
Section 5). The potential does not depend on it.

--oracle, for a system of GROMACS's files with plain cutoffs: an independent
f64 all-pairs sum as well (Lennard-Jones by the rule of Lorentz and
Berthelot and Coulomb with CODATA's constants, every pair of particles of two
molecules within the cutoff in the minimum image, a site placed by its
weights and its force moved by them), against which the general double
route must agree within 1e-9 of the largest force and of the potential, and
the units and the general mixed route are compared in the same way as
against the general double one, the floor of the energy then the error
model of one evaluation in f32 (2 gamma_n M + 8 u L D: M and D the sums of
the magnitudes of the terms and of their derivatives in r, n their number,
L the largest coordinate, u = 2^-24) in place of 1e-6 of the potential,
which a small system's cancelling sum makes too small; the forces then need
no projection, since
the sites of [ virtual_sites3 ] of type 1 are linear in their frames.

It prints, for each frame, the largest difference of each route and the
bound, and fails if one exceeds it.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ["MDIR_COMPILE_CACHE"] = "off"
import mdir  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))

name, topology, coordinates = sys.argv[1:4]
frames, steps = int(sys.argv[4]), int(sys.argv[5])
flags = {a for a in sys.argv[6:] if "=" not in a}
options = dict(a[2:].split("=", 1) for a in sys.argv[6:] if "=" in a)
cutoff = float(options.get("cutoff", 0.9))
pairlist = float(options.get("pairlist", 1.0))
pruned = float(options.get("pruned", 0.0))


def build(units, precision, state=None):
    if topology.endswith(".prmtop"):
        loaded = mdir.load_amber(topology, coordinates)
    else:
        loaded = mdir.load_gromacs(topology, coordinates)
    system = loaded.make_system()
    initial = loaded.make_state()
    if state is not None:
        initial.positions = state.positions
        initial.velocities = state.velocities
        initial.cell = state.cell
    system.cutoff = cutoff
    system.pairlist_distance = pairlist
    if pruned:
        system.pruned_distance = pruned
    system.electrostatics = mdir.Electrostatics.PME if "--pme" in flags else mdir.Electrostatics.Cutoff
    if "--oracle" in flags:
        system.dispersion = mdir.DispersionCorrection.None_
    system.periodic = True
    system.rigid_water = True
    system.rigid_hydrogen_bonds = True
    i, e, x = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    i.timestep = 0.002
    if "--npt" in flags:
        e.kind = mdir.EnsembleKind.NPT
        e.temperature = 300.0
        e.thermostat = mdir.ThermostatMethod.VRescale
        e.tau_t = 1.0
        e.pressure = 1.01325
        e.tau_p = 5.0
    else:
        e.kind = mdir.EnsembleKind.NVE
    x.target = mdir.Target.GPU
    x.precision = precision
    x.rigid_units = mdir.RigidUnits.On if units else mdir.RigidUnits.Off
    program = mdir.compile(system, initial, i, e, x, mdir.Schedule())
    assert (program.plan["rigid_units"] == "units") == units, program.plan["rigid_units_reason"]
    return mdir.Simulation(program), program


def evaluate(units, precision, state):
    simulation, program = build(units, precision, state)
    simulation.run(0, energy=True)
    at = simulation.state()
    return (np.asarray(at.forces, dtype=np.float64), float(at.energies["potential"]),
            program)


# The frames: the input, then every STEPS steps of the units.
units, program = build(True, mdir.Precision.Mixed)
print(f"{name}: kinds taken {program.plan['rigid_units_kinds']}")
topo = program.topology
masses = np.asarray(topo.masses, dtype=np.float64)
waters = np.asarray(topo.rigid_waters).reshape(-1, 3)
states = [units.state()]
for _ in range(frames - 1):
    units.run(steps, energy=True)
    states.append(units.state())


def cell_matrix(state):
    (ax, by, cz), (bx, cx, cy) = state.cell.diagonal, state.cell.tilt
    return np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])


def image(d, H):
    f = d @ np.linalg.inv(H)
    return (f - np.round(f)) @ H


def project(forces, x, H):
    """m P (F / m) with P the projection off the constraints of each rigid
    water, in the metric of the masses."""
    a = forces.copy()
    for o, h1, h2 in waters:
        atoms = [o, h1, h2]
        minv = np.repeat([1.0 / masses[k] for k in atoms], 3)
        g = np.zeros((3, 9))
        for row, (p, q) in enumerate(((0, 1), (0, 2), (1, 2))):
            d = image(x[atoms[q]] - x[atoms[p]], H)
            g[row, 3 * p:3 * p + 3] = -d
            g[row, 3 * q:3 * q + 3] = d
        flat = (forces[atoms].ravel()) * minv
        lam = np.linalg.solve((g * minv) @ g.T, g @ flat)
        a[atoms] = ((flat - minv * (g.T @ lam)) / minv).reshape(3, 3)
    return a


def oracle(state):
    """The f64 all-pairs sum of the GROMACS system, its energy and forces."""
    from kind_pairs_oracle import COULOMB, read_top
    types, moltypes, order = read_top(topology)
    sigma_of, epsilon_of, charge, molecule, sites = [], [], [], [], []
    particle = 0
    for mol, count in order:
        for copy in range(count):
            atoms = moltypes[mol]["atoms"]
            for type_name, q, _ in atoms:
                _, s, e = types[type_name]
                sigma_of.append(s)
                epsilon_of.append(e)
                charge.append(q)
                molecule.append((mol, copy))
            site = moltypes[mol]["site"]
            if site:
                place, a, b = site
                sites.append((particle + place, particle, particle + 1, particle + 2,
                              1 - a - b, a, b))
            particle += len(atoms)
    x = np.asarray(state.positions, dtype=np.float64).copy()
    for s, i, j, k, wi, wj, wk in sites:
        x[s] = wi * x[i] + wj * x[j] + wk * x[k]
    H = cell_matrix(state)
    n = len(x)
    forces = np.zeros((n, 3))
    energy = 0.0
    # The error model of an evaluation in f32 (D405's oracle): the sum of
    # the magnitudes of the terms M, of their derivatives in r D, and the
    # number of terms.
    magnitude, slope, terms = 0.0, 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if molecule[i] == molecule[j]:
                continue
            d = image(x[i] - x[j], H)
            r = math.sqrt(float(d @ d))
            if r >= cutoff:
                continue
            du = 0.0
            eps = math.sqrt(epsilon_of[i] * epsilon_of[j])
            sig = 0.5 * (sigma_of[i] + sigma_of[j])
            if eps != 0.0:
                sr6 = (sig / r) ** 6
                energy += 4 * eps * (sr6 * sr6 - sr6)
                du += 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / r
                magnitude += 4 * eps * (sr6 * sr6 + sr6)
                slope += 4 * eps * (12 * sr6 * sr6 + 6 * sr6) / r
                terms += 1
            if charge[i] != 0.0 and charge[j] != 0.0:
                energy += COULOMB * charge[i] * charge[j] / r
                du += -COULOMB * charge[i] * charge[j] / (r * r)
                magnitude += abs(COULOMB * charge[i] * charge[j] / r)
                slope += abs(COULOMB * charge[i] * charge[j]) / (r * r)
                terms += 1
            f = -du * d / r
            forces[i] += f
            forces[j] -= f
    for s, i, j, k, wi, wj, wk in sites:
        f = forces[s].copy()
        forces[i] += wi * f
        forces[j] += wj * f
        forces[k] += wk * f
        forces[s] = 0.0
    u = 2.0 ** -24
    gamma = terms * u / (1 - terms * u)
    scale = float(np.abs(x).max())
    floor = 2 * gamma * magnitude + 8 * u * scale * slope
    return forces, energy, floor


failed = False
worst = 0.0
for index, state in enumerate(states):
    if index == 0 and "--after-input" in flags:
        continue
    x = np.asarray(state.positions, dtype=np.float64)
    H = cell_matrix(state)
    fu, eu, _ = evaluate(True, mdir.Precision.Mixed, state)
    fm, em, _ = evaluate(False, mdir.Precision.Mixed, state)
    fd, ed, _ = evaluate(False, mdir.Precision.Double, state)
    references = [("double", fd, ed)]
    if "--oracle" in flags:
        fo, eo, floor = oracle(state)
        scale = float(np.abs(fo).max())
        df, de = float(np.abs(fd - fo).max()), abs(ed - eo)
        ok = df <= 1e-9 * scale and de <= 1e-9 * max(1.0, abs(eo))
        print(f"frame {index}: general double against the oracle: forces {df:.3e} "
              f"(bound {1e-9 * scale:.3e}), energy {de:.3e} of {eo:.6f}")
        failed |= not ok
        references.append(("oracle", fo, eo))
        print(f"frame {index}: the floor of the energy from the error model {floor:.3e}")
    for label, fr, er in references:
        if label == "double":
            pu, pm, pr = project(fu, x, H), project(fm, x, H), project(fr, x, H)
        else:
            pu, pm, pr = fu, fm, fr
        scale = float(np.abs(pr).max())
        du, dm = float(np.abs(pu - pr).max()), float(np.abs(pm - pr).max())
        bound = max(1.5 * dm, 1e-5 * scale)
        eu_d, em_d = abs(eu - er), abs(em - er)
        # With the oracle, the floor of the energy is the error model of one
        # evaluation in f32: 2 gamma_n M for the order of the sum and
        # 8 u L D for the rounding of the displacements, L the largest
        # coordinate (D405's oracle); 1e-6 of the potential otherwise.
        ebound = max(1.5 * em_d, floor if "--oracle" in flags else 1e-6 * abs(er))
        ok = du <= bound and eu_d <= ebound
        worst = max(worst, du / bound, eu_d / ebound)
        print(f"frame {index} against {label}: forces units {du:.3e}, mixed {dm:.3e}, "
              f"bound {bound:.3e}; energy units {eu_d:.3e}, mixed {em_d:.3e}, "
              f"bound {ebound:.3e} {'ok' if ok else 'FAILED'}")
        failed |= not ok
print(f"{name}: largest ratio to its bound {worst:.3f}")
if failed:
    sys.exit(f"{name}: a frame is outside its bound")
print(f"{name}: rigid units frames passed")
