"""The correction for the dispersion of the Python model
(D[python-dispersion]): a pair term's own `dispersion`, and an explicit
correction against the default, against `mdir run` on the same input and
against the tails of check_pair_tail.py.

On the mixture of pair_tail_system.py (60 A, 60 B, 32 Å) with the pair term
-c8/r^8 - a exp(-r/l)/r^4 over A-A and A-B of pair-dispersion.test, under a
plain cutoff and under the shift, with the correction by default, given on
the system, asked for by the term, left out by the term, and off: the
potential, the trace of the virial, and the pressure at step 0 equal the
rows of `mdir run` to its printed digits, and the difference of the default
and the term's opt-out is the term's tail (and, under the shift, the
estimate of the shift), by Simpson's rule in ln r. Then a tail that
diverges: refused when asked for, left out with a warning by default, silent
when the term leaves itself out; and a tunable exponent of a term left out.

    python_dispersion.py directory target mdir-binary
"""
import math
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import mdir

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import check_pair_tail as oracle  # noqa: E402

work = pathlib.Path(sys.argv[1])
target_name = sys.argv[2]
cli = sys.argv[3]
KCAL = 4.184
BAR = 1.01325  # bar per atm
C8, A, L = oracle.C8, oracle.A, oracle.L  # kcal/mol Å^8, kcal/mol Å^4, Å
RC = 12.0
EP, NONE = mdir.DispersionCorrection.EnergyPressure, mdir.DispersionCorrection.None_
UNSET = object()
# name: (the system's correction, the term's), as the control file gives
# them; UNSET leaves the key out.
CASES = {
    "default": (UNSET, UNSET),
    "explicit": (EP, UNSET),
    "term": (UNSET, EP),
    "optout": (UNSET, NONE),
    "off": (NONE, UNSET),
}
WORDS = {EP: "ENERGY_PRESSURE", NONE: "NONE"}


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def write_toml(name, precision, shift, system, term, expression):
    path = work / f"{name}.toml"
    lines = [f"""[input]
topology    = "plain.top"
coordinates = "system.gro"
[output]
energy_interval = 1
energy = "{name}.dat"
[energy]
cutoff            = {RC}
pairlist_distance = 13.0
electrostatics    = "CUTOFF"
lennard_jones_modifier = "{'POTENTIAL_SHIFT' if shift else 'NONE'}"
"""]
    if system is not UNSET:
        lines.append(f'dispersion_correction = "{WORDS[system]}"\n')
    lines.append(f"""[[energy.pair]]
name       = "r8"
groups     = [":A", ":A,B"]
expression = "{expression}"
c8 = {C8}
a  = {A}
l  = {L}
""")
    if term is not UNSET:
        lines.append(f'dispersion_correction = "{WORDS[term]}"\n')
    lines.append(f"""[dynamics]
time_step  = 0.001
steps      = 1
[ensemble]
ensemble    = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[execution]
target    = "{target_name}"
precision = "{precision.upper()}"
""")
    path.write_text("".join(lines))
    return path


def cli_row(name, precision, shift, system, term, expression):
    path = write_toml(name, precision, shift, system, term, expression)
    subprocess.run([cli, "run", path.name], cwd=work, check=True, stdout=subprocess.DEVNULL)
    text = (work / f"{name}.dat").read_text().splitlines()
    names = text[0].lstrip("#").split()
    rows = [r.split() for r in text if not r.startswith("#")]
    return dict(zip(names, rows[0]))


LOADED = mdir.load_gromacs(str(work / "plain.top"), str(work / "system.gro"))


def model(shift, system_correction, term_correction, expression):
    system, state = LOADED.make_system(), LOADED.make_state()
    system.cutoff, system.pairlist_distance = RC / 10, 1.3
    system.truncation = mdir.Truncation.Shift if shift else mdir.Truncation.None_
    if system_correction is not UNSET:
        system.dispersion = system_correction
    term = mdir.PairTerm()
    term.name, term.groups = "r8", [":A", ":A,B"]
    term.expression = expression
    # kJ/mol and nm.
    term.constants = [("c8", C8 * KCAL * 1e-8), ("a", A * KCAL * 1e-4), ("l", L / 10)]
    if term_correction is not UNSET:
        term.dispersion = term_correction
    system.pair_terms = [term]
    return system, state


def python_run(precision, shift, system_correction, term_correction,
               expression="-c8/r^8 - a*exp(-r/l)/r^4"):
    system, state = model(shift, system_correction, term_correction, expression)
    ensemble, execution = mdir.Ensemble(), mdir.Execution()
    ensemble.temperature = 0.0
    execution.target, execution.precision = getattr(mdir.Target, target_name), getattr(
        mdir.Precision, precision)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        program = mdir.compile(system, state, mdir.Integrator(), ensemble, execution,
                               mdir.Schedule())
    simulation = mdir.Simulation(program)
    simulation.run(0, energy=True)
    return simulation.state().energies, program.plan["dispersion"], [str(w.message) for w in caught]


# The defaults: the system's correction is EnergyPressure and not given; a
# term follows it (Python's None).
assert mdir.System().dispersion == EP and not mdir.System().dispersion_given
assert mdir.PairTerm().dispersion is None
probe = mdir.System()
probe.dispersion = EP
assert probe.dispersion_given
probe.dispersion = None
assert probe.dispersion == EP and not probe.dispersion_given

# The tails of check_pair_tail.py, kcal/mol and Å.
gro = (work / "system.gro").read_text().splitlines()
count = int(gro[1])
kinds = [line[5:10].strip() for line in gro[2:2 + count]]
volume = (float(gro[2 + count].split()[0]) * 10) ** 3
na, nb = kinds.count("A"), kinds.count("B")
n = na + nb
f = 1 - volume / (n * 4 * math.pi / 3 * RC ** 3)
factor = 4 * math.pi / volume * n * n / (n * (n - 1)) * (na * nb + na * (na - 1) / 2)
u = lambda r: -C8 / r ** 8 - A * math.exp(-r / L) / r ** 4  # noqa: E731
du = lambda r: 8 * C8 / r ** 9 + A * math.exp(-r / L) * (1 / (L * r ** 4) + 4 / r ** 5)  # noqa: E731
i, j = oracle.tails(u, du, RC)
tail_energy, tail_virial = factor * i, -factor * j
shift_estimate = f * factor * RC ** 3 * u(RC) / 3
COLUMNS = {"potential": KCAL, "virial": KCAL, "pressure": BAR}

for precision in ("Double", "Mixed"):
    for shift in (False, True):
        label = f"{target_name} {precision} {'shift' if shift else 'cutoff'}"
        energies = {}
        for case, (system_correction, term_correction) in CASES.items():
            name = f"{case}-{precision}-{'shift' if shift else 'cutoff'}".lower()
            row = cli_row(name, precision, shift, system_correction, term_correction,
                          "-c8/r^8 - a*exp(-r/l)/r^4")
            mine, plan, caught = python_run(precision, shift, system_correction, term_correction)
            assert not caught, caught
            energies[case] = mine
            # The rows of `mdir run` print six decimals of kcal/mol and atm.
            for column, scale in COLUMNS.items():
                value = mine[column] / scale
                difference = abs(value - float(row[column]))
                assert difference <= 1.5e-6 * max(1.0, abs(value)), (case, column, value, row)
            expected_in = case in ("default", "explicit", "term")
            assert plan["pair_terms"] == {"r8": expected_in}, (case, plan)
            assert plan["given"] == (system_correction is not UNSET), (case, plan)
        print(f"{label}: potential, virial and pressure of default, explicit, term, optout, off "
              f"equal to mdir run's rows")
        # Explicit and default give the same numbers; the term's opt-out
        # takes off its tail and estimate of the shift, with no virial of
        # the shift (D210).
        for case in ("explicit", "term"):
            assert energies[case] == energies["default"], (case, energies[case], energies["default"])
        energy = (energies["default"]["potential"] - energies["optout"]["potential"]) / KCAL
        virial = (energies["default"]["virial"] - energies["optout"]["virial"]) / KCAL
        pressure = energies["default"]["pressure"] - energies["optout"]["pressure"]
        reference = tail_energy + (shift_estimate if shift else 0.0)
        # P = (2K + W) / 3V: the tail's virial over 3V, in bar (kJ/mol/nm^3
        # is 10^25 / N_A bar).
        reference_pressure = tail_virial * KCAL / (3 * volume * 1e-3) * 1e25 / 6.02214076e23
        tolerance = 1e-9 if precision == "Double" else 1e-5
        for what, value, ref in (("energy", energy, reference), ("virial", virial, tail_virial),
                                 ("pressure", pressure, reference_pressure)):
            relative = abs(value - ref) / abs(ref)
            assert relative <= tolerance, (what, value, ref)
            print(f"{label}: the term's {what} beyond the cutoff {value:.9f} against "
                  f"{ref:.9f} (Simpson), relative {relative:.1e}")

# A tail that diverges: an error when asked for, by the system or by the
# term; left out with a warning by default; silent when the term leaves
# itself out.
# -c8/(l^5 r^3), whose constants keep the units of the term above.
diverges = "-c8/(l^5*r^3)"
expect(mdir.InputError, lambda: python_run("Double", False, EP, UNSET, diverges), "'r8'")
expect(mdir.InputError, lambda: python_run("Double", False, UNSET, EP, diverges), "'r8'")
for case in ("explicit", "term"):
    path = write_toml(f"diverges-{case}", "Double", False, *CASES[case], diverges)
    result = subprocess.run([cli, "check", path.name], cwd=work, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    assert result.returncode != 0 and "'r8'" in result.stdout, result.stdout
default, plan, caught = python_run("Double", False, UNSET, UNSET, diverges)
assert plan["pair_terms"] == {"r8": False} and len(caught) == 1, (plan, caught)
assert "leaves out the pair term 'r8'" in caught[0], caught
assert "set PairTerm.dispersion to DispersionCorrection.None_" in caught[0], caught
optout, plan, caught = python_run("Double", False, UNSET, NONE, diverges)
assert plan["pair_terms"] == {"r8": False} and not caught, (plan, caught)
assert optout == default, (optout, default)
row = cli_row("diverges-default", "Double", False, UNSET, UNSET, diverges)
assert abs(default["potential"] / KCAL - float(row["potential"])) <= 1.5e-6 * abs(float(row["potential"]))
print("diverges: refused when asked for, as by mdir check; by default left out with a "
      "warning, equal to the opt-out and to mdir run")
expect(mdir.InputError, lambda: python_run("Double", False, NONE, EP), "off")

# Without a periodic cell the default is off, as in the control file; an
# explicit correction is refused.
system, state = model(False, UNSET, UNSET, "-c8/r^8")
system.periodic = False
cell = mdir.Cell()
state.cell = cell
mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), mdir.Execution(), mdir.Schedule())
system.dispersion = EP
expect(mdir.InputError, lambda: mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                             mdir.Execution(), mdir.Schedule()), "periodic")
# A switch with an explicit correction is refused, as in the control file.
system, state = model(False, EP, UNSET, "-c8/r^8")
system.truncation, system.switch_distance = mdir.Truncation.Switch, 1.0
expect(mdir.InputError, lambda: mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                             mdir.Execution(), mdir.Schedule()), "plain cutoff")
# With a switch the default correction is off, with the warning
# dispersion_switched; the energies are those of the correction set off.
switched = {}
for correction in (UNSET, NONE):
    system, state = model(False, correction, UNSET, "-c8/r^8")
    system.truncation, system.switch_distance = mdir.Truncation.Switch, 1.0
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution,
                               mdir.Schedule())
    messages = [str(w.message) for w in caught]
    assert program.plan["dispersion"]["correction"] == NONE, program.plan
    assert program.plan["dispersion"]["pair_terms"] == {"r8": False}, program.plan
    assert len(messages) == (1 if correction is UNSET else 0), messages
    assert all("switch" in m for m in messages), messages
    simulation = mdir.Simulation(program)
    simulation.run(0, energy=True)
    switched[correction is UNSET] = simulation.state().energies
assert switched[True] == switched[False], switched
print("no cell, switch: as the control file; by default a switch turns the correction off "
      "with a warning")
# The words of the Python model, and a type error for a value of another
# kind.
expect(mdir.InputError, lambda: python_run("Double", False, EP, UNSET, diverges),
       "set PairTerm.dispersion to DispersionCorrection.None_")
expect(TypeError, lambda: setattr(mdir.System(), "dispersion", "NONE"))
expect(TypeError, lambda: setattr(mdir.PairTerm(), "dispersion", 0))

# A tunable exponent: at p = 3 the tail diverges. A term left out of the
# correction takes it; one in it refuses an update that would leave it out.
for term_correction in (NONE, UNSET):
    system, state = model(False, UNSET, term_correction, "-c8/r^p")
    # pair_terms gives copies: change one and set it back.
    term = system.pair_terms[0]
    term.constants = [("c8", C8 * KCAL * 1e-8), ("p", 8.0)]
    system.pair_terms = [term]
    system.tunables = [mdir.Tunable("p", "p", term="r8")]
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target_name)
    simulation = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                              execution, mdir.Schedule()))
    update = lambda: simulation.tunables.update({"p": np.array([3.0])})  # noqa: E731
    if term_correction is NONE:
        update()
        simulation.run(0, energy=True)
        assert math.isfinite(simulation.state().energies["potential"])
    else:
        expect(mdir.InputError, update, "leave it out")
print("tunable exponent: taken by a term left out, refused by a term in the correction")
print("python dispersion passed")
