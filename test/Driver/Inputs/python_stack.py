"""The stack of the entry of a Python simulation does not grow with its steps
of energy (#305).

    python_stack.py dipeptide-directory target precision

The entry of a program of segments never returns between parts (D215): what
it takes of its stack in a loop over steps it keeps for the life of the
activation. The row of the observed columns was allocated where it is
written, in the loops, and taken again at every step of energy: 16 bytes for
every two columns on the CPU, of the 64 MiB of an activation.

For each program the depth of the stack at the boundary where the entry
waits (`Simulation._stack_in_use`, private) is read after a first step of
energy and after CALLS and 2 CALLS more, each way a step of energy is taken:
as the last step of a part (`run(1, energy=True)`) and at the rows of a
reporter inside a part (`run(STEPS)`). It must be the same to the byte.
A program without `observe`, one with tunables and their derivative, and a
run with a barostat and its periods of coupling are there as well; only the
observed columns grew. The lowered module of each program (`lowered_ir`, the
program of `mdir run`, which has the same rows in its loops) is read too: on
the CPU no allocation of the stack is outside the first block of its
function; on a GPU the lowering releases the stack where an iteration ends
(`releaseStack`), so an allocation outside the first block stands with a
`stacksave`.
"""
import re
import sys

import mdir

root = sys.argv[1]
target = getattr(mdir.Target, sys.argv[2])
precision = getattr(mdir.Precision, sys.argv[3])
mode = f"{sys.argv[2]} {sys.argv[3]}"
SEED = 271828
CALLS = 8  # steps of energy between two readings, as last steps of parts
STEPS = 8  # steps of energy in a part, at the rows of a reporter


def observed(lists):
    """A pair term and two walls that observe the constants of `lists`."""
    def change(system):
        soft = mdir.PairTerm()
        soft.name, soft.expression = "soft", "a*exp(-r/l)"
        soft.constants = [("a", 2.0), ("l", 0.05)]
        lower, upper = mdir.ExternalTerm(), mdir.ExternalTerm()
        lower.name, lower.expression, lower.selection = "lower", "0.5*k*max(0, z0 - z)^2", ":WAT"
        lower.constants = [("k", 4184.0), ("z0", 0.6)]
        upper.name, upper.expression, upper.selection = "upper", "0.5*k*max(0, z - z0)^2", ":WAT"
        upper.constants = [("k", 4184.0), ("z0", 2.0)]
        soft.observe, lower.observe, upper.observe = lists
        lower.scaling = upper.scaling = mdir.ExternalScaling.None_
        system.pair_terms = [soft]
        system.external_terms = [lower, upper]
    return change


def tuned(system):
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    soft.observe = ["l"]
    system.pair_terms = [soft]
    system.tunables = [mdir.Tunable("soft_l", "l", term="soft")]
    system.tunable_gradient = True


def program(change=None, kind="NVE", coupling=None):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.Cutoff
    if change:
        change(system)
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    if coupling:
        ensemble.coupling_period = coupling
    execution.target, execution.precision = target, precision
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def allocations(text):
    """The allocations of the stack of a lowered module outside the first
    block of their function, and the ops that save the stack."""
    outside, saves, block = 0, 0, 0
    for line in text.splitlines():
        if re.match(r"\s*llvm\.func ", line):
            block = 0
        elif re.match(r"\s*\^bb\d+", line):
            block += 1
        elif "llvm.intr.stacksave" in line:
            saves += 1
        elif re.search(r"\.alloca(?!\w)", line) and block > 0:
            outside += 1
    return outside, saves


# name: (the program, the columns of its row, the steps of a call that ends
# with a step of energy, the period of the reporter).
PROGRAMS = {
    "no observed column": (lambda: program(), 0, 1, 1),
    "4 observed columns": (lambda: program(observed((["l"], ["z0"], None))), 4, 1, 1),
    "5 observed columns": (lambda: program(observed((["l", "a"], ["z0"], None))), 5, 1, 1),
    "2 observed columns, a tunable, and the derivative in the tunables": (
        lambda: program(tuned), 2, 1, 1),
    "4 observed columns under a barostat, coupled every 5 steps": (
        lambda: program(observed((["l"], ["z0"], None)), kind="NPT", coupling=5), 4, 5, 5),
}

failed = False
for name, (build, columns, last, period) in PROGRAMS.items():
    compiled = build()
    assert len(compiled.plan["observables"]) == columns, compiled.plan["observables"]
    sim = mdir.Simulation(compiled)
    assert sim._stack_in_use == 0
    sim.reporters.append(mdir.CallbackReporter(lambda s, state: None, period))
    sim.run(last, energy=True)
    depths = [sim._stack_in_use]
    for _ in range(2):
        for _ in range(CALLS):
            sim.run(last, energy=True)
        depths.append(sim._stack_in_use)
    for _ in range(2):
        sim.run(STEPS * period)
        depths.append(sim._stack_in_use)
    level = depths[0] > 0 and len(set(depths)) == 1
    failed |= not level
    print(f"{mode}: {name}: the stack of the entry at the boundary after 1 step of energy: "
          f"{depths[0]} bytes; after {CALLS} and {2 * CALLS} more as last steps of parts "
          f"{depths[1] - depths[0]:+d} and {depths[2] - depths[0]:+d} bytes, and after {STEPS} "
          f"and {2 * STEPS} more at the rows of a reporter {depths[3] - depths[2]:+d} and "
          f"{depths[4] - depths[2]:+d} bytes (bound 0): {level}")
    outside, saves = allocations(compiled.lowered_ir)
    if sys.argv[2] == "CPU":
        clean = outside == 0
        print(f"{mode}: {name}: the lowered module has {outside} allocations of the stack "
              f"outside the first block of a function: {clean}")
    else:
        clean = outside == 0 or saves > 0
        print(f"{mode}: {name}: the lowered module has no allocation of the stack outside "
              f"the first block of a function, or releases the stack of an iteration: {clean}")
    failed |= not clean
if failed:
    sys.exit("python stack FAILED")
print("python stack passed")
