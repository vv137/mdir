"""A NumPy reference of one leapfrog step of Langevin dynamics by the middle
scheme with constraints, against the plain step and the step of energy of
a Python simulation (#285).

    leapfrog_langevin_reference.py INPUTS LIBMDRT TARGET PRECISION

The documented step (design-m1.md, Sections 11.6 and 23; [Zhang2019],
BAOAB of [Leimkuhler2013]) is that of velocity Verlet, of which leapfrog
stores the velocities of the middle. From the positions x, the stored
velocities v of the half step before, and the forces F, with h = dt/2,
a = F/m, c = exp(-gamma dt), and P_x the projection that leaves no
velocity along the constraints at x (the second half of RATTLE
[Andersen1983]):

    u  = P_x(v + h a) + h a
    x' = x + h u;  u' = c u + sqrt((1 - c^2) k_B T / m) R;  x' = x' + h u'
    x1 = SHAKE(x'), along the bonds at x [Ryckaert1977]
    v1 = u' + (x1 - x') / dt

The reference has a SHAKE and a RATTLE of its own, iterated bond by bond
to 1e-14, over every bond with a hydrogen of the topology, which holds the
three distances of each water. It takes the forces of the state (the step
is what is checked, not the forces) and the normal numbers R of Philox of
the runtime under the key of A13 (the seed, the step, the particle,
stream 2), as check_langevin_noise.py does.

The dipeptide in water (1,168 particles, PME, 2 fs, 1/ps): step 2 from the
state of step 1, as a plain step (no row of energies) and as a step of
energy (a row at every step). Both must be the documented step within
`TOLERANCE`. The reference also takes the step whose kick is that of a
whole step without the projection, u = v + dt a, which the plain step was:
its stored velocities differ from the documented ones by (1 - c)/2 of the
part of the kick along the constraints, about 1e-3 nm/ps here, which the
test must tell apart.
"""
import ctypes
import math
import sys

import numpy as np
import mdir

inputs, runtime, target, precision = sys.argv[1:5]
SEED, TEMPERATURE, FRICTION, DT = 2024, 300.0, 1.0, 0.002
KT = 0.0083144626181532 * TEMPERATURE
# nm and nm/ps. In mixed precision the velocities and the changes of the
# positions are of f32: measured 2.1e-8 nm and 1.0e-5 nm/ps.
TOLERANCE = {"Double": (1e-12, 1e-9), "Mixed": (1e-7, 1e-4)}[precision]
library = ctypes.CDLL(runtime)


def section(lines, flag):
    """The numbers of a section of an Amber topology."""
    start = next(i for i, line in enumerate(lines) if line.startswith("%FLAG " + flag))
    width = int(lines[start + 1].split(")")[0].split("(")[1].lower()
                .replace("e", "i").split("i")[1].split(".")[0])
    values = []
    for line in lines[start + 2:]:
        if line.startswith("%"):
            break
        values += [line[k:k + width] for k in range(0, len(line.rstrip("\n")), width)]
    return values


def constraints():
    lines = open(inputs + "/dipeptide/dipeptide.prmtop").readlines()
    bonds = [int(v) for v in section(lines, "BONDS_INC_HYDROGEN")]
    lengths = [float(v) for v in section(lines, "BOND_EQUIL_VALUE")]
    pairs = [(bonds[k] // 3, bonds[k + 1] // 3) for k in range(0, len(bonds), 3)]
    masses = np.array([float(v) for v in section(lines, "MASS")])
    return pairs, [0.1 * lengths[bonds[k + 2] - 1] for k in range(0, len(bonds), 3)], masses


PAIRS, LENGTHS, MASSES = constraints()


def normals(step, count):
    Words4, Words2 = ctypes.c_uint32 * 4, ctypes.c_uint32 * 2
    out = np.zeros((count, 3))
    for index in range(count):
        words = Words4()
        library.mdrtPhilox4x32(Words4(step & 0xFFFFFFFF, step >> 32, index, 2 << 24),
                               Words2(SEED & 0xFFFFFFFF, SEED >> 32), words)
        u = [(w + 0.5) * 2.0**-32 for w in words]
        r0, r1 = math.sqrt(-2 * math.log(u[0])), math.sqrt(-2 * math.log(u[2]))
        out[index] = (r0 * math.cos(2 * math.pi * u[1]), r0 * math.sin(2 * math.pi * u[1]),
                      r1 * math.cos(2 * math.pi * u[3]))
    return out


def shake(old, x, w):
    """`x` with the bonds at their lengths, moved along the bonds of `old`."""
    x = x.copy()
    for _ in range(1000):
        worst = 0.0
        for (i, j), length in zip(PAIRS, LENGTHS):
            e, d = old[j] - old[i], x[j] - x[i]
            error = d @ d - length * length
            worst = max(worst, abs(error) / (length * length))
            g = error / (2.0 * (w[i] + w[j]) * (d @ e))
            x[i] += g * w[i] * e
            x[j] -= g * w[j] * e
        if worst < 1e-14:
            return x
    raise AssertionError("SHAKE of the reference did not converge")


def rattle(x, v, w):
    """`v` with no part along the bonds of `x`."""
    v = v.copy()
    for _ in range(1000):
        worst = 0.0
        for i, j in PAIRS:
            d = x[j] - x[i]
            rate = d @ (v[j] - v[i])
            worst = max(worst, abs(rate))
            g = rate / ((w[i] + w[j]) * (d @ d))
            v[i] += g * w[i] * d
            v[j] -= g * w[j] * d
        if worst < 1e-14:
            return v
    raise AssertionError("RATTLE of the reference did not converge")


def step(x, v, f, m, R, projected):
    w = 1.0 / m
    a, h = f * w[:, None], 0.5 * DT
    u = rattle(x, v + h * a, w) + h * a if projected else v + DT * a
    c = math.exp(-FRICTION * DT)
    middle = c * u + np.sqrt((1 - c * c) * KT * w)[:, None] * R
    drifted = x + h * u + h * middle
    x1 = shake(x, drifted, w)
    return x1, middle + (x1 - drifted) / DT


def states(period):
    """The states of steps 1 and 2 of a simulation with a row of energies
    every `period` steps."""
    loaded = mdir.load_amber(inputs + "/dipeptide/dipeptide.prmtop",
                             inputs + "/dipeptide/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [32] * 3
    system.periodic = True
    system.rigid_water = system.rigid_hydrogen_bonds = True
    integrator, bath, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method, integrator.timestep = mdir.IntegratorMethod.Leapfrog, DT
    bath.seed, bath.kind, bath.temperature = SEED, mdir.EnsembleKind.NVT, TEMPERATURE
    bath.thermostat, bath.friction = mdir.ThermostatMethod.Langevin, FRICTION
    bath.coupling_period = 0
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    schedule = mdir.Schedule()
    schedule.steps, schedule.energy_period = 4, period
    simulation = mdir.Simulation(
        mdir.compile(system, state, integrator, bath, execution, schedule))
    out = []
    for _ in range(2):
        simulation.run(1, energy=period == 1)
        taken = simulation.state()
        out.append([np.array(getattr(taken, field), dtype=float)
                    for field in ("positions", "velocities", "forces")])
    return MASSES, out


failed = False
for period, name in ((4, "plain step"), (1, "step of energy")):
    masses, ((x, v, f), (x2, v2, _)) = states(period)
    R = normals(2, len(masses))
    documented = step(x, v, f, masses, R, projected=True)
    whole = step(x, v, f, masses, R, projected=False)
    dx, dv = np.abs(x2 - documented[0]).max(), np.abs(v2 - documented[1]).max()
    apart = np.abs(whole[1] - documented[1]).max()
    ok = dx < TOLERANCE[0] and dv < TOLERANCE[1] and apart > 10 * TOLERANCE[1]
    failed |= not ok
    print(f"{target} {precision} {name}: positions within {dx:.1e} nm and velocities within "
          f"{dv:.1e} nm/ps of the documented step; the kick of a whole step is "
          f"{apart:.1e} nm/ps away, {np.abs(v2 - whole[1]).max():.1e} from this step: "
          f"{'ok' if ok else 'FAILED'}", flush=True)
sys.exit(1 if failed else 0)
