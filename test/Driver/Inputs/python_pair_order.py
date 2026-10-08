"""The order of the pair terms of the Python model when the correction for
the dispersion leaves one out (#238): a term whose tail diverges, left out
with a warning under the default, before and after a term whose tail is
integrated.

On the mixture of pair_tail_system.py (60 A, 60 B, 32 Å) with -c3/r^3 over
all pairs, whose tail diverges, and -c8/r^8 over A-A and A-B: in both orders
the program compiles with one warning that names the first, the plan of the
correction has the second and not the first, and the potential, the trace of
the virial, the pressure, and the forces are equal. The tail of the second
after the first, the potential less that with the term's own opt-out, is its
closed form at a uniform density, the sum over the pairs of
-nu (4 pi / V) c8 / (5 r_c^5) (D209). A tunable c8 of the second term gives,
after an update, the energies of the program compiled with the new value.

    python_pair_order.py directory target
"""
import math
import pathlib
import sys
import warnings

import numpy as np
import mdir

work = pathlib.Path(sys.argv[1])
target_name = sys.argv[2]
KCAL = 4.184
RC = 12.0
C3, C8 = 30.0, 2.0e4  # kcal/mol Å^3, kcal/mol Å^8
NONE = mdir.DispersionCorrection.None_
LOADED = mdir.load_gromacs(str(work / "plain.top"), str(work / "system.gro"))


def terms(c8=C8, r8_correction=None):
    far = mdir.PairTerm()
    far.name, far.expression = "far", "-c3/r^3"
    far.constants = [("c3", C3 * KCAL * 1e-3)]  # kJ/mol nm^3
    r8 = mdir.PairTerm()
    r8.name, r8.groups, r8.expression = "r8", [":A", ":A,B"], "-c8/r^8"
    r8.constants = [("c8", c8 * KCAL * 1e-8)]  # kJ/mol nm^8
    if r8_correction is not None:
        r8.dispersion = r8_correction
    return {"far": far, "r8": r8}


def run(order, precision, tunable=False, **keywords):
    system, state = LOADED.make_system(), LOADED.make_state()
    system.cutoff, system.pairlist_distance = RC / 10, 1.3
    system.truncation = mdir.Truncation.None_
    made = terms(**keywords)
    system.pair_terms = [made[name] for name in order]
    if tunable:
        system.tunables = [mdir.Tunable("c8", "c8", term="r8")]
    ensemble, execution = mdir.Ensemble(), mdir.Execution()
    ensemble.temperature = 0.0
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        program = mdir.compile(system, state, mdir.Integrator(), ensemble, execution,
                               mdir.Schedule())
    simulation = mdir.Simulation(program)
    simulation.run(0, energy=True)
    return simulation, program.plan["dispersion"], [str(w.message) for w in caught]


def energies(simulation):
    state = simulation.state()
    return state.energies, np.array(state.forces)


def close(first, second, tolerance):
    worst = 0.0
    for column in ("potential", "virial", "pressure"):
        scale = max(1.0, abs(first[0][column]))
        worst = max(worst, abs(first[0][column] - second[0][column]) / scale)
    scale = np.abs(first[1]).max()
    worst = max(worst, np.abs(first[1] - second[1]).max() / scale)
    assert worst <= tolerance, worst
    return worst


gro = (work / "system.gro").read_text().splitlines()
count = int(gro[1])
kinds = [line[5:10].strip() for line in gro[2:2 + count]]
volume = (float(gro[2 + count].split()[0]) * 10) ** 3
na, nb = kinds.count("A"), kinds.count("B")
n = na + nb
# No pair is excluded: nu = N^2 / (N (N - 1)).
tail = -4 * math.pi / volume * n * n / (n * (n - 1)) * (na * nb + na * (na - 1) / 2) \
    * C8 / (5 * RC ** 5)

for precision, tolerance in (("Double", 1e-12), ("Mixed", 2e-5)):
    label = f"{target_name} {precision}"
    results = {}
    for order in (("far", "r8"), ("r8", "far")):
        simulation, plan, caught = run(order, precision)
        assert len(caught) == 1 and "leaves out the pair term 'far'" in caught[0], caught
        assert plan["pair_terms"] == {"far": False, "r8": True}, plan
        results[order] = energies(simulation)
    worst = close(results[("far", "r8")], results[("r8", "far")], tolerance)
    print(f"{label}: far, r8 and r8, far give the same potential, virial, pressure and "
          f"forces, relative {worst:.1e}")
    simulation, plan, caught = run(("far", "r8"), precision, r8_correction=NONE)
    assert plan["pair_terms"] == {"far": False, "r8": False}, plan
    without = energies(simulation)
    mine = (results[("far", "r8")][0]["potential"] - without[0]["potential"]) / KCAL
    relative = abs(mine - tail) / abs(tail)
    assert relative <= (1e-9 if precision == "Double" else 1e-3), (mine, tail)
    print(f"{label}: the tail of r8 after far {mine:.9f} against its closed form "
          f"{tail:.9f}, relative {relative:.1e}")
    # A tunable of the term after the one left out.
    simulation, plan, caught = run(("far", "r8"), precision, tunable=True)
    close(energies(simulation), results[("far", "r8")], tolerance)
    simulation.tunables.update({"c8": np.array([1.5 * C8 * KCAL * 1e-8])})
    simulation.run(0, energy=True)
    fresh, plan, caught = run(("far", "r8"), precision, c8=1.5 * C8)
    close(energies(simulation), energies(fresh), tolerance)
    print(f"{label}: a tunable c8 of r8 after far gives the program compiled with its value")
print("python pair order passed")
