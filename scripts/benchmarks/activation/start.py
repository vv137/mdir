"""The time of a start of an activation of a Python simulation (#294,
D253).

A simulation of a system of the Amber benchmark suite, with PME, a cutoff
of 0.8 nm, flexible, at 0.5 fs, and the length of a soft pair term
a exp(-r/l) as its tunable, takes steps, and then begins activations in the
ways a fit does (docs/python-segments.md, D215):

  gradient   `tunables.gradient()` again and again: the particles have not
             moved since the last start, so their order is the same;
  update     `tunables.update()` to another value and `gradient()`: two
             starts;
  moved      `run(STEPS)` and `gradient()`: the particles have moved, and
             the start orders them anew. The time of the steps, taken from
             runs of the same length that begin no activation, is
             subtracted.

Each is repeated and the least and the median are printed, in ms, with the
time of a step. A start includes its evaluation (the kernels of the forces
and the build of the neighbor structure). Where the build has them
(`Simulation._kept_tuples`, private), the line also says how many
structures of the tuples the starts with the least time took from those
that the simulation keeps, how many they built, and the time their code
spent on both.

Usage:
  start.py PRMTOP INPCRD [PRMTOP INPCRD ...] [--target GPU]
           [--precision Mixed] [--repeats 5] [--steps 20] [--input-order]
           [--deterministic] [--no-keep] [--only gradient|update|moved]
           [--long N] [--tunable pair|pair-tail|lj|charge|tuple|all]
           [--marks] [--alone]

`--tunable` chooses what is tunable, since what an update costs follows
what its values enter (#324, D[update-cost]): `pair`, the length of the
pair term (the default); `pair-tail`, the same with the tail of the term
in the correction for the dispersion, under a shift at the cutoff; `lj`,
sigma and epsilon of the types with Lennard-Jones, with the correction;
`charge`, the charges; `tuple`, the constant of a weak spring on every
bond; `all`, the four together. An update alternates between the values of
the model and values 0.1% from them. With `--alone` the updates are also
timed without the `gradient()` after them. With `--marks` each line is
followed by where the host spent the time of the call
(`Simulation._host_times`, private): the build of the program anew
(`rebuild`) with its parts, the copy of the state from the activation
(`download`), the end of an activation (`end`), and the arguments
(`start.arguments`), the stack and the records (`start.open`), and the
program (`start.program`) of each start.

With `--long N` one run of N steps follows, with its ms per step and ns
per day: the rate of a plain run, which a change of the starts must leave.
Several systems are measured one after another, each in a simulation of
its own. With `--no-keep` the simulation builds its structures anew at
every start (`Simulation._keep_tuples = False`, private), in the memory
that it keeps. With `MDRT_PROFILE` in the environment the device runtime
prints its counters when the script ends; with `--only gradient
--repeats N` the difference of two values of N gives what one start takes
of each.
"""
import argparse
import re
import time

import numpy as np
import mdir

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", metavar="PRMTOP INPCRD")
parser.add_argument("--target", default="GPU")
parser.add_argument("--precision", default="Mixed")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--warm", type=int, default=200)
parser.add_argument("--input-order", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--no-keep", action="store_true")
parser.add_argument("--long", type=int, default=0)
parser.add_argument("--only", default=None, choices=("gradient", "update", "moved"))
parser.add_argument("--tunable", default="pair",
                    choices=("pair", "pair-tail", "lj", "charge", "tuple", "all"))
parser.add_argument("--marks", action="store_true")
parser.add_argument("--alone", action="store_true")
args = parser.parse_args()
if len(args.files) % 2:
    parser.error("each system is a topology and its coordinates")


def declare(system):
    """Declares the tunables of `--tunable` on `system`, with the terms that
    they need."""
    kinds = ("pair", "lj", "charge", "tuple") if args.tunable == "all" else (args.tunable,)
    tunables = []
    if "pair" in kinds or "pair-tail" in kinds:
        soft = mdir.PairTerm()
        soft.name, soft.expression = "soft", "a*exp(-r/l)"
        soft.constants = [("a", 2.0), ("l", 0.05)]
        system.pair_terms = [soft]
        tunables.append(mdir.Tunable("l", "l", term="soft"))
    # The correction for the dispersion is a derived quantity of the table of
    # the Lennard-Jones, and holds the tail of the pair term where it is on
    # (D209): `lj`, `pair-tail`, and `all` keep it, with the shift at the
    # cutoff in place of the switch, which takes no correction.
    if args.tunable in ("lj", "pair-tail", "all"):
        system.truncation = mdir.Truncation.Shift
    else:
        system.dispersion = mdir.DispersionCorrection.None_
    if "lj" in kinds:
        tunables += [mdir.Tunable("sigma", "sigma"), mdir.Tunable("epsilon", "epsilon")]
    if "charge" in kinds:
        tunables.append(mdir.Tunable("q", "charge"))
    if "tuple" in kinds:
        # A weak spring on every bond of the topology, a constant for each.
        bonds = np.asarray(system.topology.bonds, dtype=np.int64)
        spring = mdir.TupleTerm()
        spring.name, spring.expression, spring.arity = "spring", "0.5*k*(r - r0)^2", 2
        spring.particles = bonds
        spring.parameters = [("k", np.full(len(bonds), 100.0)),
                             ("r0", np.full(len(bonds), 0.1))]
        system.tuple_terms = [spring]
        tunables.append(mdir.Tunable("k", "k", term="spring"))
    system.tunables = tunables


def measure(prmtop, inpcrd):
    loaded = mdir.load_amber(prmtop, inpcrd)
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    declare(system)
    system.tunable_gradient = True
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, args.target)
    execution.precision = getattr(mdir.Precision, args.precision)
    execution.deterministic = args.deterministic
    if args.input_order:
        execution.reorder = False
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    # A type without Lennard-Jones (the hydrogen of a water) keeps its
    # values: the combining rule has no derivative in an epsilon of 0, which
    # the compile says, naming the type.
    names = list(system.topology.type_names)
    left = set()
    while True:
        try:
            program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution,
                                   mdir.Schedule())
            break
        except mdir.InputError as error:
            found = re.search(r"is 0 for the type (\S+), where the combining rule", str(error))
            if not found or found.group(1) in left:
                raise
            left.add(found.group(1))
            keep = np.array([name not in left for name in names])
            entries = np.where(keep, np.cumsum(keep) - 1, -1)
            system.tunables = [mdir.Tunable(t.name, t.parameter, term=t.term, map=entries)
                               if t.parameter in ("sigma", "epsilon") and t.term is None else t
                               for t in system.tunables]

    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    counted = hasattr(sim, "_kept_tuples")
    if args.no_keep:
        if not counted:
            raise SystemExit("this build keeps no structures of the tuples")
        sim._keep_tuples = False
    t0 = time.perf_counter()
    sim.run(args.warm)
    sim.state()
    first = time.perf_counter() - t0

    marked = hasattr(sim, "_host_times")

    def timed(call):
        """The time of `call` in ms; what it took, built, and spent on the
        structures of the tuples; and the marks of the host in it, each
        (ms, page faults, calls)."""
        before = sim._kept_tuples if counted else (0, 0, 0.0)
        marks = sim._host_times if marked else {}
        t0 = time.perf_counter()
        call()
        elapsed = 1e3 * (time.perf_counter() - t0)
        after = sim._kept_tuples if counted else (0, 0, 0.0)
        parts = {}
        for name, (seconds, faults, calls) in (sim._host_times if marked else {}).items():
            was = marks.get(name, (0.0, 0, 0))
            if calls != was[2]:
                parts[name] = (1e3 * (seconds - was[0]), faults - was[1], calls - was[2])
        return (elapsed, after[0] - before[0], after[1] - before[1],
                1e3 * (after[2] - before[2]), parts)

    def run_steps():
        sim.run(args.steps)
        sim.state()

    steps_ms = min(timed(run_steps)[0] for _ in range(3))
    atoms = len(sim.state().positions)
    print(f"{args.target} {args.precision}{' deterministic' if args.deterministic else ''}"
          f"{' input order' if args.input_order else ''}"
          f"{' building every structure anew' if args.no_keep else ''}: {atoms} atoms, "
          f"compilation and the first {args.warm} steps {first:.2f} s, "
          f"a step {steps_ms / args.steps:.4f} ms", flush=True)

    def report(name, rows, less=0.0):
        times = [row[0] - less for row in rows]
        least = rows[int(np.argmin(times))]
        line = (f"{name}: least {min(times):.3f} ms, median {float(np.median(times)):.3f} ms, "
                f"{1e3 * min(times) / atoms:.3f} us per atom, "
                f"{min(times) / (steps_ms / args.steps):.0f} steps")
        if counted:
            line += (f"; structures of the tuples: {least[1]} taken, {least[2]} built, "
                     f"{least[3]:.3f} ms")
        print(line, flush=True)
        # Where the host spent the time of that call: a name with a dot is a
        # part of the name before the dot.
        if args.marks:
            for mark, (ms, faults, calls) in sorted(least[4].items()):
                print(f"    {mark:34s} {ms:9.3f} ms {faults:8d} faults {calls:3d} calls",
                      flush=True)

    sim.tunables.gradient()
    if args.only in (None, "gradient"):
        report("gradient", [timed(sim.tunables.gradient) for _ in range(args.repeats)])

    if args.only in (None, "update"):
        # The values of the model, and others 0.1% from them, in turn.
        base = {name: np.array(sim.tunables[name]) for name in sim.tunables.keys()}
        values = [{name: value * 1.001 for name, value in base.items()}, base]

        def update(k):
            sim.tunables.update(values[k % 2])
            sim.tunables.gradient()

        report(f"update ({args.tunable})",
               [timed(lambda: update(k)) for k in range(args.repeats)])
        if args.alone:
            report(f"update alone ({args.tunable})",
                   [timed(lambda: sim.tunables.update(values[k % 2]))
                    for k in range(args.repeats)])
        if hasattr(sim, "_tunable_builds"):
            print(f"the updates: programs built from the model that the simulation keeps, "
                  f"and from a copy of the model of the compile: "
                  f"{tuple(sim._tunable_builds)}", flush=True)

    if args.only in (None, "moved"):
        def moved():
            sim.run(args.steps)
            sim.tunables.gradient()

        report("moved", [timed(moved) for _ in range(args.repeats)], less=steps_ms)

    if args.long:
        # One long run in one part, which begins no activation: the rate of
        # a step, to compare between builds.
        sim.run(args.steps)
        t0 = time.perf_counter()
        sim.run(args.long)
        sim.state()
        step = 1e3 * (time.perf_counter() - t0) / args.long
        print(f"a run of {args.long} steps: {step:.4f} ms per step, "
              f"{86400.0 * integrator.timestep / step:.1f} ns per day", flush=True)


for prmtop, inpcrd in zip(args.files[0::2], args.files[1::2]):
    measure(prmtop, inpcrd)
