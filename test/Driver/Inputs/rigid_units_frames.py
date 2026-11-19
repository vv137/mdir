"""Same-frame forces and energies of the route of the rigid units
(D424; #795's validation table, moved to the key).

  rigid_units_frames.py NAME TOPOLOGY COORDINATES FRAMES STEPS [--pme] [--npt]
                        [--oracle] [--floor] [--after-input] [--cutoff=nm] [--pairlist=nm]
                        [--pruned=nm]

A trajectory of `rigid_units = ON` in mixed precision gives FRAMES frames,
the first the input itself (before any constraint, whose covering radii the
resolution widens where the input lies beyond them) and then one every STEPS
steps. At each frame three programs evaluate the forces and the potential at
the same positions and cell: the units (ON, mixed), and the general route of
the groups (D415) in mixed and in double precision. The units must agree with the general double
route within 1.5 times the general mixed route's own difference, the
largest of 5 runs at each frame and over the frames of the system, or 1e-5 of the largest force (the
forces) and 1e-6 of the potential (the energy), where those are larger
(#795's criterion, with D422's baseline over several samples). The sums of
both routes in mixed precision are made in f32 and widened once (D415's
rule), so their energies differ from the double route's by the rounding of
those sums.

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
against the general double one; the forces then need no projection, since
the sites of [ virtual_sites3 ] of type 1 are linear in their frames.

--floor, for a system of GROMACS's files: the bounds take as a floor as
well the error model of the sums in f32 (below), which follows the
magnitudes of the pair terms of the frame.

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
    # The general route that the units replace: the groups of D415, whose
    # sums in mixed precision are made as the units' are (D415's rule).
    x.neighbor_structure = mdir.NeighborStructure.Groups
    program = mdir.compile(system, initial, i, e, x, mdir.Schedule())
    assert (program.plan["rigid_units"] == "units") == units, program.plan["rigid_units_reason"]
    return mdir.Simulation(program), program


def evaluate(units, precision, state, runs=1):
    """The forces and the potential of `runs` simulations of one program from
    the same state: the sums of the groups in mixed precision add with
    atomics in no fixed order, so their rounding changes from run to run."""
    simulation, program = build(units, precision, state)
    results = []
    for k in range(runs):
        if k:
            simulation = mdir.Simulation(program)
        simulation.run(0, energy=True)
        at = simulation.state()
        results.append((np.asarray(at.forces, dtype=np.float64), float(at.energies["potential"])))
    return results


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
            if charge[i] != 0.0 and charge[j] != 0.0:
                energy += COULOMB * charge[i] * charge[j] / r
                du += -COULOMB * charge[i] * charge[j] / (r * r)
            f = -du * d / r
            forces[i] += f
            forces[j] -= f
    for s, i, j, k, wi, wj, wk in sites:
        f = forces[s].copy()
        forces[i] += wi * f
        forces[j] += wj * f
        forces[k] += wk * f
        forces[s] = 0.0
    return forces, energy


# The error model of the sums in f32 (the maintainer's ruling on #915),
# for a system of GROMACS's files (--floor): a floor of the bounds that
# follows the magnitudes of the pair terms, not the baseline of the other
# route, which near close contacts does not bound this route's rounding.
#
# Each route evaluates the pair terms in f32 and adds the m terms of a unit
# (the units) or of a particle (the groups) in f32 before it widens the sum
# (D415's rule). In the model of Higham and Mary (2019) -- each rounding
# x(1 + delta), the deltas independent, of mean zero, |delta| <= u -- a
# quantity that depends on the roundings to first order through
# sum_k a_k delta_k lies within lambda u sqrt(sum_k a_k^2), except with
# probability at most 2 exp(-lambda^2 / 2) (the inequality of Hoeffding that
# their analysis rests on); lambda = 6.5 makes that 1.3e-9. With u = 2^-24:
#   - the evaluation of a term, k roundings of its magnitude: k = 28 for the
#     Lennard-Jones (r^2 in three products and two sums, its inverse, the
#     cube and the square: 6 (3 + 1) + 4) and k = 12 for Coulomb (erfc's
#     approximation within 3.3e-7, D90, about 6 u, and six operations), so
#     a = k |U_ij|;
#   - the sum of the m terms of a partial sum, each addition rounding a
#     partial sum of at most the sum M of the magnitudes of the terms, so
#     a = sqrt(m) M for the sum.
# The difference of the two routes is at most the sum of their bounds, and
# the units' partial sums are taken as a molecule's, the larger:
#   energy: 2 lambda u (sqrt(sum_ij (k_ij |U_ij|)^2) + sqrt(sum_g m_g M_g^2)),
#   forces: 2 lambda u (sqrt(sum_j (k_ij |F_ij|)^2) + sqrt(m_i) sum_j |F_ij|),
#           the largest over the particles i,
# g the molecules, m_g and M_g the count and the sum of the magnitudes of
# their terms, m_i the terms of particle i, F_ij = U'_ij. The roundings of
# the positions and of the displacements are not in the floor: each route
# narrows the positions to f32 once, the anchors alike, and the general
# mixed route's own difference, the baseline, measures their effect. The
# reciprocal sum of PME is the same program in both routes; its rounding
# from run to run is in the baseline too.
LAMBDA, U32 = 6.5, 2.0 ** -24


def error_floor(state):
    from scipy.special import erfc
    from scipy.optimize import brentq
    from kind_pairs_oracle import COULOMB, read_top
    types, moltypes, order = read_top(topology)
    sig, eps, q, mol = [], [], [], []
    for number, (m, count) in enumerate(order):
        for copy in range(count):
            for type_name, charge, _ in moltypes[m]["atoms"]:
                _, s_, e_ = types[type_name]
                sig.append(s_)
                eps.append(e_)
                q.append(charge)
                mol.append(number * 100000 + copy)
    sig, eps, q, mol = map(np.array, (sig, eps, q, mol))
    x = np.asarray(state.positions, dtype=np.float64)
    H = cell_matrix(state)
    Hi = np.linalg.inv(H)
    pme = "--pme" in flags
    beta = brentq(lambda b: erfc(b * cutoff) - 1e-5, 0.1, 50.0) if pme else 0.0
    images = [np.array([a, b, c]) @ H for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    n = len(x)
    e_terms = 0.0
    group_m, group_M = {}, {}
    f_terms, f_sum, f_count = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n - 1):
        d = x[i + 1:] - x[i]
        d = ((d @ Hi) - np.round(d @ Hi)) @ H
        r2 = None
        for im in images:
            dd = d + im
            q2 = (dd * dd).sum(1)
            r2 = q2 if r2 is None else np.minimum(r2, q2)
        j = np.arange(i + 1, n)
        keep = (r2 < cutoff * cutoff) & (mol[j] != mol[i])
        r = np.sqrt(r2[keep])
        j = j[keep]
        e = np.sqrt(eps[i] * eps[j])
        s6 = (0.5 * (sig[i] + sig[j])) ** 6
        a12, a6 = 4 * e * s6 * s6 / r ** 12, 4 * e * s6 / r ** 6
        u_lj, f_lj = a12 + a6, (12 * a12 + 6 * a6) / r
        kq = np.abs(COULOMB * q[i] * q[j])
        if pme:
            ex = 2 * beta / math.sqrt(math.pi) * np.exp(-(beta * r) ** 2)
            u_c, f_c = kq * erfc(beta * r) / r, kq * (erfc(beta * r) / r ** 2 + ex / r)
        else:
            u_c, f_c = kq / r, kq / r ** 2
        e_terms += float(((28 * u_lj) ** 2 + (12 * u_c) ** 2).sum())
        err_f = (28 * f_lj) ** 2 + (12 * f_c) ** 2
        f_terms[i] += err_f.sum()
        np.add.at(f_terms, j, err_f)
        f_t = f_lj + f_c
        f_sum[i] += f_t.sum()
        np.add.at(f_sum, j, f_t)
        f_count[i] += len(j)
        np.add.at(f_count, j, 1)
        u_t = u_lj + u_c
        group_m[mol[i]] = group_m.get(mol[i], 0) + len(j)
        group_M[mol[i]] = group_M.get(mol[i], 0.0) + float(u_t.sum())
        for owner, value in zip(mol[j], u_t):
            group_m[owner] = group_m.get(owner, 0) + 1
            group_M[owner] = group_M.get(owner, 0.0) + float(value)
    sums = math.sqrt(sum(group_m[k] * group_M[k] ** 2 for k in group_m))
    energy = 2 * LAMBDA * U32 * (math.sqrt(e_terms) + sums)
    forces = 2 * LAMBDA * U32 * float((np.sqrt(f_terms) + np.sqrt(f_count) * f_sum).max())
    return energy, forces


RUNS = 5
failed = False
worst = 0.0
# Each frame's differences from each reference; the baseline of the general
# mixed route is its largest difference over RUNS runs at each frame and over
# the frames of the system, so that one sample where mixed happens to round
# close does not make the bound (a baseline of one sample flickers, D422).
rows = []
for index, state in enumerate(states):
    if index == 0 and "--after-input" in flags:
        continue
    x = np.asarray(state.positions, dtype=np.float64)
    H = cell_matrix(state)
    (fu, eu), = evaluate(True, mdir.Precision.Mixed, state)
    mixed = evaluate(False, mdir.Precision.Mixed, state, RUNS)
    (fd, ed), = evaluate(False, mdir.Precision.Double, state)
    references = [("double", fd, ed)]
    floor_e, floor_f = error_floor(state) if "--floor" in flags else (0.0, 0.0)
    if "--floor" in flags:
        print(f"frame {index}: the floors of the error model: energy {floor_e:.3e}, "
              f"forces {floor_f:.3e}")
    if "--oracle" in flags:
        fo, eo = oracle(state)
        scale = float(np.abs(fo).max())
        df, de = float(np.abs(fd - fo).max()), abs(ed - eo)
        ok = df <= 1e-9 * scale and de <= 1e-9 * max(1.0, abs(eo))
        print(f"frame {index}: general double against the oracle: forces {df:.3e} "
              f"(bound {1e-9 * scale:.3e}), energy {de:.3e} of {eo:.6f}")
        failed |= not ok
        references.append(("oracle", fo, eo))
    for label, fr, er in references:
        if label == "double":
            pu, pr = project(fu, x, H), project(fr, x, H)
            pms = [project(fm, x, H) for fm, _ in mixed]
        else:
            pu, pr = fu, fr
            pms = [fm for fm, _ in mixed]
        rows.append((index, label, float(np.abs(pu - pr).max()),
                     max(float(np.abs(pm - pr).max()) for pm in pms),
                     float(np.abs(pr).max()), abs(eu - er),
                     max(abs(em - er) for _, em in mixed), abs(er), floor_e, floor_f))
for label in sorted({r[1] for r in rows}):
    mine = [r for r in rows if r[1] == label]
    dm_all = max(r[3] for r in mine)
    em_all = max(r[6] for r in mine)
    for index, _, du, dm, scale, eu_d, em_d, er, floor_e, floor_f in mine:
        bound = max(1.5 * dm_all, 1e-5 * scale, floor_f)
        ebound = max(1.5 * em_all, 1e-6 * er, floor_e)
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
