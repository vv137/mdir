"""The time of a start of an activation of a Python simulation (#294).

A simulation of a system of the Amber benchmark suite, with PME, a cutoff
of 0.8 nm, flexible, at 0.5 fs, and the length of a soft pair term
a exp(-r/l) as its tunable, takes steps, and then begins activations in the
ways a fit does (docs/python-segments.md, D215):

  gradient   `tunables.gradient()` again and again: the particles have not
             moved since the last start, so their order is the same;
  update     `tunables.update()` to another value and `gradient()`;
  moved      `run(STEPS)` and `gradient()`: the particles have moved, and
             the start orders them anew. The time of the steps, taken from
             a run of the same length that begins no activation, is
             subtracted.

Each is repeated and the least and the median are printed, in ms, with the
time of a step. A start includes its evaluation (the kernels of the forces
and the build of the neighbor structure).

Usage:
  start.py PRMTOP INPCRD [--target GPU] [--precision Mixed]
           [--repeats 5] [--steps 20] [--input-order] [--deterministic]

With `MDRT_PROFILE` in the environment the device runtime prints its
counters when the script ends; with `--only gradient --repeats N` the
difference of two values of N gives what one start takes of each.
"""
import argparse
import time

import numpy as np
import mdir

parser = argparse.ArgumentParser()
parser.add_argument("prmtop")
parser.add_argument("inpcrd")
parser.add_argument("--target", default="GPU")
parser.add_argument("--precision", default="Mixed")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--warm", type=int, default=200)
parser.add_argument("--input-order", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--only", default=None, choices=("gradient", "update", "moved"))
args = parser.parse_args()

loaded = mdir.load_amber(args.prmtop, args.inpcrd)
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
program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())

sim = mdir.Simulation(program)
sim.part_seconds = 1e9
t0 = time.perf_counter()
sim.run(args.warm)
sim.state()
first = time.perf_counter() - t0


def timed(call):
    t0 = time.perf_counter()
    call()
    return 1e3 * (time.perf_counter() - t0)


def run_steps():
    sim.run(args.steps)
    sim.state()


steps_ms = min(timed(run_steps) for _ in range(3))
atoms = len(sim.state().positions)
print(f"{args.target} {args.precision}{' deterministic' if args.deterministic else ''}"
      f"{' input order' if args.input_order else ''}: {atoms} atoms, "
      f"compilation and the first {args.warm} steps {first:.2f} s, "
      f"a step {steps_ms / args.steps:.4f} ms", flush=True)


def report(name, times):
    least, median = min(times), float(np.median(times))
    print(f"{name}: least {least:.3f} ms, median {median:.3f} ms, "
          f"{1e3 * least / atoms:.3f} us per atom, "
          f"{least / (steps_ms / args.steps):.0f} steps", flush=True)


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

    report("moved", [timed(moved) - steps_ms for _ in range(args.repeats)])
