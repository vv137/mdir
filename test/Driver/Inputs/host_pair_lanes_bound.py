"""The lanes of the host's pair rows against the scalar row at a bound of
the order of a sum (D[host-pair-vector]).

    host_pair_lanes_bound.py DIPEPTIDE_DIR PRECISION WIDTH...

The dipeptide in 382 waters with PME: ten types, the exclusions of the
Amber topology, and rows of 83 to 275 candidates within the cutoff, whose
lengths are mostly not multiples of the width, so that the tail's padding repeats a
candidate that contributes. For each width, the forces and the potential at
step 0 with MDIR_HOST_PAIRS=lanes:W against MDIR_HOST_PAIRS=scalar.

The lanes add the same terms as the scalar row in another order. The terms
are the same numbers in both, since the lanes evaluate each op of the
kernel as the scalar row does (lanes(1) is the scalar row to the bit). A
sum of n terms t_k in two orders differs by at most 2 gamma_{n-1} sum |t_k|,
gamma_m = m u / (1 - m u) [Higham2002, Section 4.2], u the unit roundoff of
the type of the row's sum (2^-24 in mixed, 2^-53 in double). The bound of
each atom and component takes n as the candidates within the cutoff and
sum |t_k| over every pair within the cutoff, excluded or not, from a NumPy
reference of the direct terms of the kernel (Lennard-Jones and erfc(beta
r)/r), times 1.01 for the reference's own rounding against the kernel's;
counting excluded pairs only raises the bound. The potential's bound is the
sum of the rows' bounds of the energy (each pair's energy weighted 1/2),
plus the rounding of the sum over the rows in f64.

Prints, for each width, the largest ratio of a force difference to its
bound, the ratio of the RMS difference to the RMS bound, the potential's
ratio, and whether every ratio is at most 1.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import parmed

import mdir

sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import COULOMB_NATIVE, KJ_PER_KCAL  # noqa: E402

root, precision, widths = sys.argv[1], sys.argv[2], [int(w) for w in sys.argv[3:]]
prmtop, inpcrd = os.path.join(root, "dipeptide.prmtop"), os.path.join(root, "dipeptide.inpcrd")
CUTOFF, TOLERANCE = 0.9, 1e-5  # nm; erfc(beta r_c)
KE = COULOMB_NATIVE  # kJ mol^-1 nm e^-2


erfc = np.vectorize(math.erfc)


def erfcinv(y):
    """x with erfc(x) = y, by bisection."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erfc(mid) > y else (lo, mid)
    return 0.5 * (lo + hi)


def step0(mode):
    os.environ["MDIR_HOST_PAIRS"] = mode
    loaded = mdir.load_amber(prmtop, inpcrd)
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = CUTOFF
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_tolerance = TOLERANCE
    system.rigid_water, system.rigid_hydrogen_bonds = True, True
    execution, schedule = mdir.Execution(), mdir.Schedule()
    execution.target = mdir.Target.CPU
    execution.precision = getattr(mdir.Precision, precision)
    execution.threads = 2
    schedule.steps, schedule.energy_period = 1, 1
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution, schedule)
    assert f"mode={mode.split(':')[0]}" in program.pipeline, program.pipeline
    sim = mdir.Simulation(program)
    sim.run(0, energy=True)
    s = sim.state()
    return np.asarray(s.forces), float(s.energies["potential"])


def reference():
    """The bounds' sums: for each atom, the candidates within the cutoff, the
    sums of |f_ij| by component, and the sum of |u_ij| / 2."""
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
    count = np.zeros(n)
    force = np.zeros((n, 3))
    energy = np.zeros(n)
    for i in range(n):
        d = x[i] - x
        d -= box * np.round(d / box)
        r2 = np.sum(d * d, axis=1)
        near = (r2 < CUTOFF * CUTOFF) & (np.arange(n) != i)
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
    return count, force, energy


u = 2.0**-24 if precision == "Mixed" else 2.0**-53
count, force, energy = reference()
gamma = lambda m: m * u / (1 - m * u)
row = 2 * gamma(np.maximum(count - 1, 1)) * 1.01
fbound = row[:, None] * force
ebound = float(np.sum(row * energy)) + len(count) * 2.0**-53 * float(np.sum(energy))
print(f"rows: {int(count.min())} to {int(count.max())} candidates within the cutoff; "
      f"{int(np.sum(count % 8 != 0))} of {len(count)} not a multiple of 8")
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
