"""The time of a start of an activation of a Python simulation (#294,
D[activation-start]).

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

Several systems are measured one after another, each in a simulation of
its own. With `--no-keep` the simulation builds its structures anew at
every start (`Simulation._keep_tuples = False`, private), in the memory
that it keeps. With `MDRT_PROFILE` in the environment the device runtime
prints its counters when the script ends; with `--only gradient
--repeats N` the difference of two values of N gives what one start takes
of each.
"""
import argparse
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
parser.add_argument("--only", default=None, choices=("gradient", "update", "moved"))
args = parser.parse_args()
if len(args.files) % 2:
    parser.error("each system is a topology and its coordinates")


def measure(prmtop, inpcrd):
    loaded = mdir.load_amber(prmtop, inpcrd)
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    system.pair_terms = [soft]
    system.tunables = [mdir.Tunable("l", "l", term="soft")]
    system.tunable_gradient = True
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, args.target)
    execution.precision = getattr(mdir.Precision, args.precision)
    execution.deterministic = args.deterministic
    if args.input_order:
        execution.reorder = False
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution,
                           mdir.Schedule())

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

    def timed(call):
        """The time of `call` in ms, and what it took, built, and spent on
        the structures of the tuples."""
        before = sim._kept_tuples if counted else (0, 0, 0.0)
        t0 = time.perf_counter()
        call()
        elapsed = 1e3 * (time.perf_counter() - t0)
        after = sim._kept_tuples if counted else (0, 0, 0.0)
        return (elapsed, after[0] - before[0], after[1] - before[1],
                1e3 * (after[2] - before[2]))

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

    sim.tunables.gradient()
    if args.only in (None, "gradient"):
        report("gradient", [timed(sim.tunables.gradient) for _ in range(args.repeats)])

    if args.only in (None, "update"):
        values = [0.0501, 0.05]

        def update(k):
            sim.tunables.update({"l": np.array([values[k % 2]])})
            sim.tunables.gradient()

        report("update", [timed(lambda: update(k)) for k in range(args.repeats)])

    if args.only in (None, "moved"):
        def moved():
            sim.run(args.steps)
            sim.tunables.gradient()

        report("moved", [timed(moved) for _ in range(args.repeats)], less=steps_ms)


for prmtop, inpcrd in zip(args.files[0::2], args.files[1::2]):
    measure(prmtop, inpcrd)
