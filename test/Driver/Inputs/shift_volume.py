"""The estimate of the shift (D210) under a barostat (#224). It is

    E_sh(V) = X(V) (1 - V / (N (4 pi / 3) r_c^3)),  X(V) proportional to 1 / V,

so its part -X V / (N K) does not depend on the volume, and only X follows
the cell as the tail does. Every place it enters, after the barostat has
changed the volume, against closed forms at the present volume:

    shift_volume.py mixture-directory target mdir work

The mixture of pair_tail_system.py (60 A + 60 B Lennard-Jones particles of
one sigma and epsilon, no pair excluded, r_c = 1.2 nm), with an NBFIX
written as a pair term between A and B, compressed by a barostat at 500
bar. What the correction for the dispersion adds to a quantity is the
quantity less that of the same state evaluated with the correction off:

  Python     the columns of `observe` of the pair term (energy and both
             derivatives); the energy of the state under the shift;
             `gradient().energy` and the derivatives in a constant of the
             pair term and in the sigma of the types (D230)
  mdir run   the potential of the energy file under POTENTIAL_SHIFT, the
             observables file, and dH/dlambda of the free-energy file, at
             the last row, against a run from the state of its checkpoint
             with the correction off
"""
import math
import pathlib
import subprocess
import sys

import numpy as np
import mdir

mixture = pathlib.Path(sys.argv[1])
target_name, cli = sys.argv[2], sys.argv[3]
work = pathlib.Path(sys.argv[4])
KJ = 4.184
RC, N, NA, NB = 1.2, 120, 60, 60
SIGMA, EPSILON = 0.34, 0.24 * KJ
NBFIX = "4*eps*((sig/r)^12 - (sig/r)^6) - 4*eps0*((sig0/r)^12 - (sig0/r)^6)"
C = {"sig": 0.37, "eps": 0.5 * KJ, "sig0": 0.34, "eps0": 0.24 * KJ}
ALCH = "lambda_s*4*e*((s/r)^12 - (s/r)^6)"
A = {"e": 0.1 * KJ, "s": 0.36}
STEPS = 1000
# With `report`, every difference is printed and none is an error.
REPORT = len(sys.argv) > 5 and sys.argv[5] == "report"


def check(ok, label, found, value):
    if REPORT:
        print(f"  {label}: found {found:+.9f} closed form {value:+.9f} difference "
              f"{found - value:+.3e}")
    else:
        assert ok, (label, found, value)


def lennard_jones(sig, eps):
    """Of 4 eps ((sig/r)^12 - (sig/r)^6): the integral of r^2 u beyond r_c,
    u(r_c), and their derivatives in sig."""
    i = 4 * eps * (sig ** 12 / (9 * RC ** 9) - sig ** 6 / (3 * RC ** 3))
    u = 4 * eps * ((sig / RC) ** 12 - (sig / RC) ** 6)
    di = 4 * eps * (12 * sig ** 11 / (9 * RC ** 9) - 6 * sig ** 5 / (3 * RC ** 3))
    du = 4 * eps * (12 * sig ** 11 / RC ** 12 - 6 * sig ** 5 / RC ** 6)
    return i, u, di, du


def closed(volume):
    """What the correction adds at the volume, in kJ/mol and nm: the tail
    and the estimate of the shift of the pair terms, nu (4 pi / V) N_A N_B
    (I + f r_c^3 u(r_c) / 3) with nu = N / (N - 1) and f = 1 - V / (N (4 pi
    / 3) r_c^3), and of the Lennard-Jones of the topology, whose tail takes
    the r^-6 part alone, E = -(2 pi / 3 V) N^2 C6 / r_c^3, and whose
    estimate is E f."""
    f = 1 - volume / (N * 4 * math.pi / 3 * RC ** 3)
    factor = 4 * math.pi / volume * N / (N - 1) * NA * NB

    def pair(sig, eps):
        i, u, di, du = lennard_jones(sig, eps)
        return factor * (i + f * RC ** 3 * u / 3), factor * (di + f * RC ** 3 * du / 3)
    energy, d_sig = pair(C["sig"], C["eps"])
    energy0, _ = pair(C["sig0"], C["eps0"])
    alch, _ = pair(A["s"], A["e"])
    c6 = 4 * EPSILON * SIGMA ** 6
    topology = -2 * math.pi / (3 * volume) * N * N * c6 / RC ** 3 * (1 + f)
    return {"nbfix.energy": energy - energy0, "nbfix.d_sig": d_sig,
            "nbfix.d_eps": energy / C["eps"], "alch": alch, "topology": topology,
            "topology.d_sigma": 6 * topology / SIGMA}


def program(precision, shift, kind="NPT", state_from=None, correction=True):
    loaded = mdir.load_gromacs(str(mixture / "plain.top"), str(mixture / "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = RC, 1.3
    system.truncation = mdir.Truncation.Shift if shift else mdir.Truncation.None_
    if not correction:
        system.dispersion = mdir.DispersionCorrection.None_
    term = mdir.PairTerm()
    term.name, term.groups, term.expression = "nbfix", [":A", ":B"], NBFIX
    term.constants, term.observe = list(C.items()), ["sig", "eps"]
    system.pair_terms = [term]
    system.tunables = [mdir.Tunable("nb_sig", "sig", term="nbfix"),
                       mdir.Tunable("lj_sigma", "sigma", map=np.array([0, 0]))]
    system.tunable_gradient = True
    state = (state.draw_velocities(system, 100.0, 7) if state_from is None
             else mdir.InitialState.from_state(state_from))
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.002, 100.0, 7
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    if kind == "NPT":
        ensemble.com_period = ensemble.coupling_period = 10
        ensemble.pressure, ensemble.tau_p = 500.0, 0.5
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                        mdir.Schedule()))


def volume_of(state):
    return float(np.prod(np.diag(state.cell.vectors)))


def run_python(precision):
    # The energies of the pairs within the cutoff cancel in the difference
    # of two evaluations of one state; in mixed precision to the rounding of
    # f32 sums.
    tolerance = 1e-9 if precision == "Double" else 1e-6
    worst, ratio = 0.0, 1.0
    for shift in (False, True):
        sim = program(precision, shift)
        first = volume_of(sim.state())
        sim.run(STEPS, energy=True)
        row = sim.state()
        gradient = sim.tunables.gradient()
        volume = volume_of(row)
        ratio = volume / first
        assert abs(ratio - 1) > 0.02, ratio
        off = program(precision, shift, "NVE", row, correction=False)
        off.run(0, energy=True)
        bare, bare_gradient = off.state(), off.tunables.gradient()
        expected = closed(volume)
        whole = expected["topology"] + expected["nbfix.energy"]
        found = {key: row.observables[key] - bare.observables[key] for key in row.observables}
        found["gradient.energy"] = gradient.energy - bare_gradient.energy
        found["gradient.nb_sig"] = gradient["nb_sig"][0] - bare_gradient["nb_sig"][0]
        found["gradient.lj_sigma"] = gradient["lj_sigma"][0] - bare_gradient["lj_sigma"][0]
        wanted = {key: expected[key] for key in row.observables}
        wanted["gradient.energy"] = whole
        wanted["gradient.nb_sig"] = expected["nbfix.d_sig"]
        wanted["gradient.lj_sigma"] = expected["topology.d_sigma"]
        if shift:
            # The energy that the run reports holds the estimate under the
            # shift only.
            found["potential"] = row.energies["potential"] - bare.energies["potential"]
            wanted["potential"] = whole
        for key, value in wanted.items():
            gap = abs(found[key] - value)
            scale = max(abs(value), 1.0)
            check(gap <= tolerance * scale, f"{precision} {'shift' if shift else 'cutoff'} {key}",
                  found[key], value)
            worst = max(worst, gap / scale)
    print(f"{target_name} {precision} Python: at V/V0 = {ratio:.3f} the correction adds its "
          f"closed forms at the present volume to the columns of observe, to the energy under "
          f"the shift, and to gradient() and its energy, within {worst:.1e}")


def control(name, precision, kind, steps, coordinates=None, correction=True):
    coupling = ('[thermostat]\nmethod = "V-RESCALE"\ninterval = 10\n'
                '[barostat]\nmethod = "C-RESCALE"\ntime_constant = 0.5\n'
                if kind == "NPT" else "")
    com = "center_of_mass_interval = 10" if kind == "NPT" else ""
    pressure = f"pressure = {500.0 / 1.01325!r}" if kind == "NPT" else ""
    constants = "\n".join(f"{key} = {value!r}" for key, value in C.items())
    (work / f"{name}.toml").write_text(f"""[input]
topology = "{mixture}/plain.top"
coordinates = "{coordinates or str(mixture / 'system.gro')}"
[output]
energy_interval = {steps}
energy = "{name}.dat"
observables = "{name}.obs"
free_energy = "{name}.dhdl"
checkpoint = "{name}.h5"
checkpoint_interval = {steps}
[energy]
cutoff = 12.0
pairlist_distance = 13.0
electrostatics = "CUTOFF"
lennard_jones_modifier = "POTENTIAL_SHIFT"
dispersion_correction = "{'ENERGY_PRESSURE' if correction else 'NONE'}"
[[energy.pair]]
name = "nbfix"
groups = [":A", ":B"]
expression = "({NBFIX.replace('/r)', '/(r*0.1))')})/4.184"
observe = ["sig", "eps"]
{constants}
[[energy.pair]]
name = "alch"
groups = [":A", ":B"]
expression = "({ALCH.replace('/r)', '/(r*0.1))')})/4.184"
e = {A['e']!r}
s = {A['s']!r}
[free_energy]
[free_energy.lambdas]
s = [1.0]
[dynamics]
time_step = 0.002
steps = {steps}
seed = 7
{com}
[ensemble]
ensemble = "{kind}"
temperature = 100.0
{pressure}
{coupling}[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    subprocess.run([cli, "run", f"{name}.toml"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = {}
    for suffix in ("dat", "obs", "dhdl"):
        lines = (work / f"{name}.{suffix}").read_text().splitlines()
        names = lines[0].lstrip("#").split()
        for line in lines[2:]:
            values = line.split()
            rows.setdefault(int(values[0]), {}).update(zip(names[2:], map(float, values[2:])))
    return rows


def write_gro(checkpoint, path):
    """The positions and the cell of a checkpoint as GROMACS coordinates
    with nine decimals of nm."""
    c = mdir.read_checkpoint(str(checkpoint))
    lines = ["the state of a checkpoint", str(len(c.positions))]
    for index, (x, y, z) in enumerate(c.positions):
        name = "A" if index < NA else "B"
        lines.append(f"{(index + 1) % 100000:5d}{name:<5s}{name:>5s}{(index + 1) % 100000:5d}"
                     f"{x:15.9f}{y:15.9f}{z:15.9f}")
    cell = c.cell.vectors
    lines.append(f"{cell[0][0]:15.9f}{cell[1][1]:15.9f}{cell[2][2]:15.9f}")
    path.write_text("\n".join(lines) + "\n")
    return float(np.prod(np.diag(cell)))


def run_cli(precision):
    # Six decimals of kcal/mol in the files, and nine of nm in the
    # coordinates of the evaluation.
    tolerance = 5e-6
    name = f"npt-{precision}".lower()
    first = float((mixture / "system.gro").read_text().splitlines()[-1].split()[0]) ** 3
    row = control(name, precision, "NPT", STEPS)[STEPS]
    volume = write_gro(work / f"{name}.h5", work / f"{name}.gro")
    assert abs(volume / first - 1) > 0.02, volume / first
    bare = control(name + "-off", precision, "NVE", 1, str(work / f"{name}.gro"), False)[0]
    expected = closed(volume)
    wanted = {"potential": expected["topology"] + expected["nbfix.energy"] + expected["alch"],
              "nbfix.energy": expected["nbfix.energy"], "nbfix.d_sig": expected["nbfix.d_sig"],
              "nbfix.d_eps": expected["nbfix.d_eps"], "dHdl.s": expected["alch"]}
    worst = 0.0
    for key, value in wanted.items():
        found = row[key] - bare[key]
        gap = abs(found - value / KJ)
        scale = max(abs(value / KJ), 1.0)
        check(gap <= tolerance * scale, f"{precision} mdir run {key} (kcal/mol)", found,
              value / KJ)
        worst = max(worst, gap / scale)
    print(f"{target_name} {precision} mdir run: at V/V0 = {volume / first:.3f} the correction "
          f"adds its closed forms at the present volume to the potential under the shift, to "
          f"the observables file, and to dH/dlambda, within {worst:.1e}")


for precision in ("Double", "Mixed"):
    run_python(precision)
    run_cli(precision)
print("shift volume passed")
