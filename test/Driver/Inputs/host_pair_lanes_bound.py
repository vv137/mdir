"""The lanes of the host's pair rows against the scalar row at a bound of
the order of a sum (D402).

    host_pair_lanes_bound.py DIPEPTIDE_DIR MDIR_OPT PRECISION WIDTH...

The dipeptide in 382 waters with PME, a reach of 1 nm: ten types, the
exclusions of the Amber topology, and rows of the list (the candidates
within the reach that are not excluded) of 118 to 371, 1,017 of 1,168 not a
multiple of 8, so that the tail's padding repeats a candidate. For each
width, the program's loops over pairs must take host(lanes(W)) (the
pipeline up to md-exec-choose-host-pairs with its remarks), and the forces
and the potential at step 0 with MDIR_HOST_PAIRS=lanes:W are compared with
MDIR_HOST_PAIRS=scalar.

The lanes add the same terms as the scalar row in another order. The terms
are the same numbers in both, since the lanes evaluate each op of the
kernel as the scalar row does (lanes(1) is the scalar row to the bit). A
sum of n terms t_k in two orders differs by at most 2 gamma_{n-1} sum |t_k|,
gamma_m = m u / (1 - m u) [Higham2002, Section 4.2], u the unit roundoff of
the type of the row's sum (2^-24 in mixed, 2^-53 in double). The bound of
each atom and component takes n as the candidates of its row within the
cutoff and sum |t_k| over them, from a NumPy reference of the direct terms
of the kernel (Lennard-Jones and erfc(beta r)/r) without the excluded
pairs, as the list leaves them out, times 1.01 for the reference's own
rounding against the kernel's. The potential's bound is the sum of the
rows' bounds of the energy (each pair's energy weighted 1/2), plus the
rounding of the sum over the rows in f64.

Prints, for each width, the largest ratio of a force difference to its
bound, the ratio of the RMS difference to the RMS bound, the potential's
ratio, and whether every ratio is at most 1.
"""
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import parmed

import mdir

sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import COULOMB_NATIVE, KJ_PER_KCAL  # noqa: E402

root, mdir_opt, precision = sys.argv[1], sys.argv[2], sys.argv[3]
widths = [int(w) for w in sys.argv[4:]]
prmtop, inpcrd = os.path.join(root, "dipeptide.prmtop"), os.path.join(root, "dipeptide.inpcrd")
CUTOFF, REACH, TOLERANCE = 0.9, 1.0, 1e-5  # nm, nm; erfc(beta r_c)
KE = COULOMB_NATIVE  # kJ mol^-1 nm e^-2


erfc = np.vectorize(math.erfc)


def erfcinv(y):
    """x with erfc(x) = y, by bisection."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erfc(mid) > y else (lo, mid)
    return 0.5 * (lo + hi)


def lanes_of_loops(program, width):
    """The loops over pairs of the program take host(lanes(width)): the
    pipeline up to md-exec-choose-host-pairs, with its remarks, on the
    program's module. Every loop is remarked; none keeps the scalar row."""
    pipeline = program.pipeline
    end = pipeline.index("md-exec-choose-host-pairs{")
    end = pipeline.index("}", end)
    prefix = pipeline[:end] + " explain=true}"
    out = subprocess.run([shutil.which(mdir_opt) or mdir_opt, "-", f"--pass-pipeline=builtin.module({prefix})",
                          "-o", os.devnull], input=program.ir, capture_output=True, text=True, check=True)
    chosen = re.findall(r"remark: host pairs: lanes\((\d+)\)", out.stderr)
    refused = re.findall(r"remark: host pairs: scalar: (.*)", out.stderr)
    assert chosen and not refused and set(chosen) == {str(width)}, (chosen, refused)
    print(f"lanes({width}): {len(chosen)} loops over pairs take host(lanes({width})), none keeps the scalar row")


def step0(mode):
    os.environ["MDIR_HOST_PAIRS"] = mode
    loaded = mdir.load_amber(prmtop, inpcrd)
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = CUTOFF
    system.pairlist_distance = REACH
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_tolerance = TOLERANCE
    system.rigid_water, system.rigid_hydrogen_bonds = True, True
    execution, schedule = mdir.Execution(), mdir.Schedule()
    execution.target = mdir.Target.CPU
    execution.precision = getattr(mdir.Precision, precision)
    execution.threads = 2
    schedule.steps, schedule.energy_period = 1, 1
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution, schedule)
    assert "mode=" + mode.replace(":", " width=") in program.pipeline, program.pipeline
    if mode.startswith("lanes:"):
        lanes_of_loops(program, int(mode.split(":")[1]))
    sim = mdir.Simulation(program)
    sim.run(0, energy=True)
    s = sim.state()
    return np.asarray(s.forces), float(s.energies["potential"])


def reference():
    """The bounds' sums: for each atom, the candidates within the cutoff, the
    sums of |f_ij| by component, and the sum of |u_ij| / 2, over the pairs
    of its row (the excluded pairs of the topology, which the build of the
    list leaves out, are not); and the length of its row, the candidates
    within the reach."""
    p = parmed.load_file(prmtop, inpcrd)
    x = np.asarray(p.coordinates) / 10.0
    box = np.asarray(p.box[:3]) / 10.0
    q = np.array([a.charge for a in p.atoms])
    ntypes = p.parm_data["POINTERS"][1]
    types = np.array(p.parm_data["ATOM_TYPE_INDEX"]) - 1
    index = np.array(p.parm_data["NONBONDED_PARM_INDEX"]).reshape(ntypes, ntypes) - 1
    acoef = np.array(p.parm_data["LENNARD_JONES_ACOEF"]) * KJ_PER_KCAL * 1e-12  # kJ nm^12
    bcoef = np.array(p.parm_data["LENNARD_JONES_BCOEF"]) * KJ_PER_KCAL * 1e-6  # kJ nm^6
    beta = erfcinv(TOLERANCE) / CUTOFF
    n = len(x)
    excluded = [set() for _ in range(n)]
    counts, listed = p.parm_data["NUMBER_EXCLUDED_ATOMS"], p.parm_data["EXCLUDED_ATOMS_LIST"]
    at = 0
    for i in range(n):
        for j in listed[at:at + counts[i]]:
            if j > 0:
                excluded[i].add(j - 1)
                excluded[j - 1].add(i)
        at += counts[i]
    count = np.zeros(n)
    rows = np.zeros(n, dtype=int)
    force = np.zeros((n, 3))
    energy = np.zeros(n)
    for i in range(n):
        d = x[i] - x
        d -= box * np.round(d / box)
        r2 = np.sum(d * d, axis=1)
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        keep[list(excluded[i])] = False
        rows[i] = int(np.sum(keep & (r2 < REACH * REACH)))
        near = keep & (r2 < CUTOFF * CUTOFF)
        d, r = d[near], np.sqrt(r2[near])
        k = index[types[i], types[near]]
        a, b = acoef[k], bcoef[k]
        qq = KE * q[i] * q[near]
        u = a / r**12 - b / r**6 + qq * erfc(beta * r) / r
        # -dU/dr / r, times d, the force of the pair on i.
        dudr = (-12 * a / r**13 + 6 * b / r**7
                - qq * (erfc(beta * r) / r**2 + 2 * beta / np.sqrt(np.pi) * np.exp(-(beta * r) ** 2) / r))
        f = (-dudr / r)[:, None] * d
        count[i] = near.sum()
        force[i] = np.abs(f).sum(axis=0)
        energy[i] = 0.5 * np.abs(u).sum()
    return count, rows, force, energy


u = 2.0**-24 if precision == "Mixed" else 2.0**-53
count, rows, force, energy = reference()
gamma = lambda m: m * u / (1 - m * u)
row = 2 * gamma(np.maximum(count - 1, 1)) * 1.01
fbound = row[:, None] * force
ebound = float(np.sum(row * energy)) + len(count) * 2.0**-53 * float(np.sum(energy))
print(f"rows of the list (within the reach): {int(rows.min())} to {int(rows.max())} candidates; "
      f"{int(np.sum(rows % 8 != 0))} of {len(rows)} not a multiple of 8, "
      f"{int(np.sum(rows % 4 != 0))} not a multiple of 4; "
      f"{int(count.min())} to {int(count.max())} within the cutoff")
f0, u0 = step0("scalar")
ok = True
for w in widths:
    f, e = step0(f"lanes:{w}")
    diff = np.abs(f - f0)
    ratio = float(np.max(diff / fbound))
    rms = float(np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(fbound**2)))
    eratio = abs(e - u0) / ebound
    good = ratio <= 1 and eratio <= 1
    ok = ok and good
    print(f"{precision} lanes({w}): max |dF|/bound {ratio:.3g}, rms |dF|/rms bound {rms:.3g}, "
          f"|dU|/bound {eratio:.3g}, max |dF| {float(diff.max()):.3g} kJ/mol/nm: "
          + ("within the bound" if good else "BEYOND THE BOUND"))
print("all within the bound" if ok else "FAILED")
