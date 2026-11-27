"""Compare one distributed step with the ordinary CPU step (D[distribute-driver]).

Usage: distribute_oracle.py ROOT CONTROL MDIR [--without-override]

CONTROL is examples/argon/argon.toml or test/Driver/charged.toml, taken as
the file gives it but for its outputs, its `[execution]`, and a skin of
0.15 Angstrom; `--without-override` also drops `[[energy.pair_override]]`,
whose tables of types the domains do not carry (distribute-refusals.test
checks that refusal), and the pair then takes the combination rule. Both programs start from the checkpoint of an ordinary run of
100 steps, so that the forces are those of a liquid, not those of the
lattice of the input, which cancel. Each pair of runs then takes one step
on a grid of 2 x 2 x 2 domains, in `MIXED` and `DOUBLE`, deterministic or
not, over a directed and a unique traversal (16 distributed runs against
8 ordinary ones). The test compares, at that step, the potential energy and
the virial of the energy file, the potential energy of the H5MD file
(binary64), and every component of the forces of the checkpoint (binary64).

Error model. A pair term q_t (an energy, a virial, or a component of a
force) is computed by the ordinary program in the type of its kernels, of
unit roundoff e = 2^-24 in `MIXED` and 2^-53 in `DOUBLE`, and by the
reference executor of the domains in binary64 (e = 2^-53 bounds both).
Three errors make up the difference of the two programs:

1. The evaluation of a term from its exact displacement: a relative error
   of at most c e, with c = 32 (the Lennard-Jones force is a polynomial of
   degree 7 in 1/r^2, about 20 rounded operations with the switch and the
   vector; the margin covers the rest), and its sum over n terms in any
   order and any blocking (Higham 2002, eq. 4.4): (n - 1) e sum_t |q_t|.
2. The displacement itself: each program rounds the coordinates it
   subtracts (the mixed kernels compute in binary32, from positions stored
   in binary64), so a component of d = x_i - x_j is off by at most
   2 e X, with X the largest coordinate either program holds, and
   |delta d| <= 2 sqrt(3) e X. A term then changes by at most
   |dq_t/dd| |delta d|: |U'| for an energy, |U'| + r |U''| for a virial,
   and max(|U''|, |U'|/r) for a force, the norm of the Jacobian of
   -U'(r) d/r, with U'' of the switched potential bounded term by term.
3. The deterministic mode rounds each pair term to a fixed-point word
   before its first addition: a force term to Q40 (2^-41 kJ/mol/nm each,
   n_i on particle i) and an energy or virial term to Q32 at each of its
   two directed visits or one unique visit (2^-33 kJ/mol each).

A pair within |delta d| of the cutoff may be inside it in one program and
outside in the other, which adds |q_t(r_c)| of each such pair (none at the
steps compared). The energy file prints 6 decimals, a quantum of
5e-7 kcal/mol in each of the two values. So, for each quantity,

    tol = 2 [(n + c) e S + e X D] + quantization + band + print,

where S = sum_t |q_t| and D = 2 sqrt(3) sum_t |dq_t/dd|, both from the
NumPy pair sum below (per particle and axis for the forces), and the factor
2 takes the bound of each program. The bound is relative (to S and D) with
an absolute floor (quantization and print), and is a bound, not an
estimate: the largest ratio of a difference to its bound is printed.
A doubled directed pair, a sign error in an image shift, or a wrong
fixed-point scale changes a quantity by a fraction of S of order one, far
outside it (the decision D[distribute-driver] records these trials).
"""
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import COULOMB_CONTROL, KJ_PER_KCAL  # noqa: E402

root = Path(sys.argv[1]).resolve()
control_file = Path(sys.argv[2]).resolve()
program = shutil.which(sys.argv[3]) or sys.argv[3]
without_override = "--without-override" in sys.argv[4:]
root.mkdir(parents=True, exist_ok=True)

KCAL = KJ_PER_KCAL
COULOMB = COULOMB_CONTROL  # kcal/mol Angstrom / e^2
C = 32
spec = tomllib.loads(control_file.read_text())
cutoff = spec["energy"]["cutoff"]
switch = spec["energy"].get("switch_distance", cutoff)
coordinates = control_file.parent / spec["input"]["coordinates"]
text = control_file.read_text()
text = re.sub(r"(?m)^coordinates\s*=.*$", f'coordinates = "{coordinates}"', text)
text = re.sub(r"(?ms)^\[execution\].*?(?=^\[|\Z)", "", text)
text = re.sub(r"(?ms)^\[output\].*?(?=^\[)", "", text)
text = re.sub(r"(?m)^steps\s*=.*$", "steps = 100", text)
if without_override:
    text = re.sub(r"(?ms)^\[\[energy.pair_override\]\].*?(?=^\[)", "", text)
    spec["energy"].pop("pair_override", None)
text = re.sub(r"(?m)^pairlist_distance\s*=.*$",
              f"pairlist_distance = {cutoff + 0.15}", text)
if not re.search(r"(?m)^seed\s*=", text):
    text = text.replace("[dynamics]", "[dynamics]\nseed = 314159")
execution = '[execution]\ntarget = "CPU"\nthreads = 1\n'


def run(name, body):
    path = root / name
    path.mkdir(exist_ok=True)
    (path / "run.toml").write_text(body)
    with (path / "run.log").open("w") as log:
        result = subprocess.run([program, "run", "run.toml"], cwd=path,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        sys.exit(f"{name}: mdir exited with {result.returncode}:\n"
                 + (path / "run.log").read_text()[-2000:])
    return path


run("start", text + '[output]\ncheckpoint = "start.h5"\n'
    'checkpoint_interval = 100\n' + execution + 'precision = "DOUBLE"\n')
start = root / "start" / "start.h5"
step = re.sub(r"(?m)^steps\s*=.*$", "steps = 1", text)
step = step.replace("[input]", f'[input]\ncheckpoint = "{start}"', 1)
step += """[output]
energy = "energy.dat"
energy_interval = 1
trajectory = "traj.h5"
trajectory_format = "H5MD"
trajectory_interval = 1
checkpoint = "state.h5"
checkpoint_interval = 1
"""
with h5py.File(start) as state:
    order = np.argsort(state["particles/all/id"][()])
    species = state["particles/all/species"][()][order]


def results(path):
    row = np.atleast_2d(np.loadtxt(path / "energy.dat"))[-1]
    with h5py.File(path / "state.h5") as state, h5py.File(path / "traj.h5") as traj:
        order = np.argsort(state["particles/all/id"][()])
        forces = state["particles/all/force/value"][0][order]
        assert forces.dtype == np.float64, forces.dtype
        positions = state["particles/all/position/value"][0][order]
        potential = traj["observables/potential_energy/value"][()][-1]
    return {"potential": row[3], "virial": row[6], "h5md": potential,
            "forces": forces, "positions": positions}


def magnitudes(positions, edges, delta):
    """S, D, and the band of the error model, in kcal/mol and Angstrom."""
    types = [t["name"] for t in spec["energy"]["type"]]
    eps = np.array([t["epsilon"] for t in spec["energy"]["type"]])
    sig = np.array([t["sigma"] for t in spec["energy"]["type"]])
    q = np.array([t.get("q", 0.0) for t in spec["energy"]["type"]])
    e2 = np.sqrt(np.outer(eps, eps))
    s2 = 0.5 * (sig[:, None] + sig[None, :])
    for o in spec["energy"].get("pair_override", []):
        a, b = (types.index(n) for n in o["types"])
        e2[a, b] = e2[b, a] = o["epsilon"]
        s2[a, b] = s2[b, a] = o["sigma"]
    x = positions * 10.0
    L = np.asarray(edges)
    i, j = np.triu_indices(len(x), 1)
    d = x[i] - x[j]
    d -= L * np.round(d / L)
    r = np.sqrt((d * d).sum(1))
    ti, tj = species[i], species[j]
    eps_ij, sig_ij = e2[ti, tj], s2[ti, tj]
    qq = q[ti] * q[tj] * COULOMB
    s6 = (sig_ij / r) ** 6
    u = 4 * eps_ij * (s6 * s6 - s6)
    du = -24 * eps_ij * (2 * s6 * s6 - s6) / r
    ddu = 24 * eps_ij * (26 * s6 * s6 - 7 * s6) / r**2
    if switch < cutoff:
        w = cutoff - switch
        t = np.clip((r - switch) / w, 0, 1)
        S = 1 + t**3 * (-10 + t * (15 - 6 * t))
        dS = (-30 * t**2 + 60 * t**3 - 30 * t**4) / w
        ddS = (-60 * t + 180 * t**2 - 120 * t**3) / w**2
    else:
        S, dS, ddS = 1.0, 0.0, 0.0
    U = np.abs(u * S) + np.abs(qq / r)
    dU = np.abs(du * S) + np.abs(u * dS) + np.abs(qq / r**2)
    ddU = (np.abs(ddu * S) + 2 * np.abs(du * dS) + np.abs(u * ddS)
           + np.abs(2 * qq / r**3))
    inside = r < cutoff
    band = np.abs(r - cutoff) <= delta
    jump = np.abs(u) + np.abs(qq / r)
    force_d = np.maximum(ddU, dU / r)
    count = np.zeros(len(x))
    np.add.at(count, i, inside)
    np.add.at(count, j, inside)

    def per_particle(values):
        total = np.zeros_like(x)
        np.add.at(total, i, values)
        np.add.at(total, j, values)
        return total

    return {
        "pairs": int(inside.sum()), "band": int(band.sum()), "count": count,
        "potential": (U[inside].sum(), dU[inside].sum(), jump[band].sum()),
        "virial": ((r * dU)[inside].sum(), (dU + r * ddU)[inside].sum(),
                   (r * dU)[band].sum()),
        "forces": (per_particle((np.abs(d) / r[:, None] * dU[:, None]) * inside[:, None]),
                   per_particle((force_d * inside)[:, None] * np.ones((1, 3))),
                   per_particle((dU * band)[:, None] * np.ones((1, 3)))),
    }


edges = np.asarray(spec["boundary"]["box"], dtype=float)
failures = 0
worst = {}
lines = []
for precision in ["DOUBLE", "MIXED"]:
    e = 2.0**-53 if precision == "DOUBLE" else 2.0**-24
    for deterministic in [False, True]:
        head = execution + f'precision = "{precision}"\n' \
            f'deterministic = {"true" if deterministic else "false"}\n'
        ordinary = results(run(f"ordinary-{precision}-{int(deterministic)}",
                               step + head))
        for traversal in ["directed", "unique"]:
            name = f"{precision.lower()} deterministic={str(deterministic).lower()} {traversal}"
            dist = results(run(name.replace(" ", "-"), step + head +
                               '[distribute]\ndomains = [2, 2, 2]\n'
                               f'traversal = "{traversal}"\n'))
            X = 10 * max(np.abs(ordinary["positions"]).max(),
                         np.abs(dist["positions"]).max(), edges.max() / 10)
            delta = 2 * np.sqrt(3) * e * X
            m = magnitudes(ordinary["positions"], edges, delta)
            n = m["pairs"]
            visits = 2 if traversal == "directed" else 1
            q32 = visits * n * 2.0**-33 / KCAL if deterministic else 0.0
            checks = []
            for key in ["potential", "virial"]:
                S, D, band = m[key]
                tol = 2 * ((n + C) * e * S + 2 * np.sqrt(3) * e * X * D) \
                    + q32 + band + 2 * 5e-7
                checks.append((key, dist[key], ordinary[key], tol, "kcal/mol"))
            S, D, band = m["potential"]
            tol = KCAL * (2 * ((n + C) * e * S + 2 * np.sqrt(3) * e * X * D)
                          + q32 + band)
            checks.append(("H5MD potential", dist["h5md"], ordinary["h5md"], tol,
                           "kJ/mol"))
            S, D, band = m["forces"]
            nf = m["count"][:, None]
            q40 = nf * 2.0**-41 if deterministic else 0.0
            # kcal/mol/Angstrom to kJ/mol/nm.
            tol = KCAL * 10 * (2 * ((nf + C) * e * S + 2 * np.sqrt(3) * e * X * D)
                               + band) + q40
            diff = np.abs(dist["forces"] - ordinary["forces"])
            k = np.unravel_index(np.argmax(diff / tol), diff.shape)
            checks.append((f"force {k[0]}.{'xyz'[k[1]]}", dist["forces"][k],
                           ordinary["forces"][k], tol[k], "kJ/mol/nm"))
            outside = int((diff > tol).sum())
            if outside:
                failures += 1
                print(f"FAIL {name}: {outside} of {diff.size} force components "
                      "outside their bounds")
            for what, a, b, tol, unit in checks:
                ratio = abs(a - b) / tol
                ok = ratio <= 1
                failures += not ok
                lines.append(f"{'ok  ' if ok else 'FAIL'} {name:34} {what:16} "
                             f"distributed {a:+.12e} ordinary {b:+.12e} "
                             f"difference {abs(a - b):.3e} bound {tol:.3e} {unit} "
                             f"ratio {ratio:.3f}")
            print(f"{name}: {n} pairs, {m['band']} within {delta:.1e} A of the "
                  f"cutoff, largest coordinate {X:.3f} A")
print("\n".join(lines))
if failures:
    sys.exit(f"{failures} comparisons outside their bounds")
print("distributed and ordinary steps agree within the error model")
