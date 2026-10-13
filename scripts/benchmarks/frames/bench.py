"""The time of a frame of `mdir.FrameEvaluator` (docs/python-frames.md).

Frames are kept from a run of the program in float64, and `evaluate` is
timed over the first K of them for each K given: the time of a call over
K, in ms per frame, the least of the repeats, after three passes over all
the frames that are not timed. The evaluator and its first frame (the
compilation and the start of the first activation) are timed apart and are
not in the rate.

Systems: the alanine dipeptide in water of the tests (1,168 atoms), and
JAC of the Amber benchmark suite (23,558 atoms), both with PME, a cutoff
of 0.8 nm, flexible, at 0.5 fs. Tunables: `soft`, the length of a soft pair
term a exp(-r/l); `charges`, the charges of the water; `lj`, sigma and
epsilon of the pair of types of the oxygens of the water.

Usage:
  bench.py PRMTOP INPCRD [--tunable soft|charges|lj] [--terms all|dependent]
           [--target GPU] [--precision Mixed] [--deterministic]
           [--frames 1,4,16,64,256] [--stride 40] [--repeats 5] [--profile N]
           [--check-gpu N] [--store FILE] [--input-order] [--groups]
           [--gradients positions,strain]

With `--gradients` every timed call asks for the gradients of the energy
named, in the positions, in the strain, or both (D[frame-gradients]).

With `--profile N` the script evaluates N frames once after the first and
stops, for the counters of `MDRT_PROFILE`: the difference of two runs with
two values of N gives what a frame takes of each counter. With
`--check-gpu N` it first prints what `nvidia-smi` says of the device N of
the machine and stops if a process holds memory there: a timing is taken on
a device that nothing else uses. With `--store FILE` the frames are kept
in FILE (`.npz`) and read from it if it is there, so that runs that are
compared evaluate the same frames and a run for the counters takes no step.
With `--input-order` the program is compiled without the spatial order of
the particles (`Execution.reorder = False`); with `--groups`, with the
groups of 16 as its neighbor structure in place of the matrix.
"""
import argparse
import os
import subprocess
import time

import numpy as np
import mdir

parser = argparse.ArgumentParser()
parser.add_argument("prmtop")
parser.add_argument("inpcrd")
parser.add_argument("--tunable", default="soft", choices=("soft", "charges", "lj"))
parser.add_argument("--terms", default="all", choices=("all", "dependent"))
parser.add_argument("--target", default="GPU")
parser.add_argument("--precision", default="Mixed")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--frames", default="1,4,16,64,256")
parser.add_argument("--stride", type=int, default=40)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--profile", type=int, default=0)
parser.add_argument("--check-gpu", type=int, default=None)
parser.add_argument("--store", default=None)
parser.add_argument("--input-order", action="store_true")
parser.add_argument("--groups", action="store_true")
parser.add_argument("--gradients", default="")
args = parser.parse_args()
# The gradients of the energy that every timed call asks for besides the
# derivative in the tunables (D[frame-gradients]); without the option the
# call names none and takes the default, which is that derivative alone.
asked = ({"gradients": ("tunables",) + tuple(n for n in args.gradients.split(",") if n)}
         if args.gradients else {})

if args.check_gpu is not None:
    said = subprocess.run(
        ["nvidia-smi", "-i", str(args.check_gpu),
         "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    print(f"device {args.check_gpu}: {said}", flush=True)
    if int(said.split(",")[-1]) > 64:
        raise SystemExit("the device is in use")


def program_():
    loaded = mdir.load_amber(args.prmtop, args.inpcrd)
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    residues = [top.residue_names[r] for r in top.residue_indices]
    water = np.array([r == "WAT" for r in residues])
    if args.tunable == "soft":
        soft = mdir.PairTerm()
        soft.name, soft.expression = "soft", "a*exp(-r/l)"
        soft.constants = [("a", 2.0), ("l", 0.05)]
        system.pair_terms = [soft]
        system.tunables = [mdir.Tunable("l", "l", term="soft")]
    elif args.tunable == "charges":
        entries = {}
        charge_map = np.array([entries.setdefault(atom[0], len(entries)) if w else -1
                               for atom, w in zip(top.atom_names, water)], dtype=np.int64)
        system.tunables = [mdir.Tunable("q", "charge", map=charge_map)]
    else:
        pairs = np.array(top.type_pairs)
        oxygen = sorted({int(top.particle_types[i]) for i in np.flatnonzero(water)
                         if top.atom_names[i][0] == "O"})[0]
        pair_map = np.where((pairs[:, 0] == oxygen) & (pairs[:, 1] == oxygen), 0, -1)
        system.tunables = [mdir.Tunable("sigma", "sigma_pair", map=pair_map),
                           mdir.Tunable("epsilon", "epsilon_pair", map=pair_map)]
    system.tunable_gradient = True
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, args.target)
    execution.precision = getattr(mdir.Precision, args.precision)
    execution.deterministic = args.deterministic
    if args.input_order:
        execution.reorder = False
    if args.groups:
        execution.neighbor_structure = mdir.NeighborStructure.Groups
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    return mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())


counts = sorted({int(k) for k in args.frames.split(",")})
most = args.profile or max(counts)
program = program_()
if args.store and os.path.exists(args.store):
    kept = np.load(args.store)
    frames, cells, step_ms = kept["frames"], kept["cells"], float(kept["step_ms"])
    if len(frames) < most:
        raise SystemExit(f"{args.store} holds {len(frames)} frames; {most} are asked for")
else:
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    sim.run(200)
    t0 = time.perf_counter()
    sim.run(1000)
    sim.state()
    step_ms = 1e3 * (time.perf_counter() - t0) / 1000
    frames = np.empty((most, len(sim.state().positions), 3))
    cell = None
    for n in range(most):
        sim.run(args.stride)
        state = sim.state()
        frames[n], cell = state.positions, state.cell
    cells = np.broadcast_to(np.asarray(cell.diagonal, dtype=np.float64), (most, 3)).copy()
    del sim
    if args.store:
        np.savez(args.store, frames=frames, cells=cells, step_ms=step_ms)

t0 = time.perf_counter()
evaluator = mdir.FrameEvaluator(program, terms=args.terms)
first = evaluator.evaluate(frames[:1], cells[:1])
made = time.perf_counter() - t0
energy = first.energy if args.terms == "all" else first.dependent_energy
print(f"{args.tunable} terms={args.terms} {args.target} {args.precision}"
      f"{' deterministic' if args.deterministic else ''}: {frames.shape[1]} atoms, "
      f"step {step_ms:.4f} ms, evaluator and first frame {made:.2f} s, U_0 = {energy[0]:.6f}",
      flush=True)
if args.profile:
    t0 = time.perf_counter()
    evaluator.evaluate(frames[:args.profile], cells[:args.profile], **asked)
    print(f"profile: {args.profile} frames, "
          f"{1e3 * (time.perf_counter() - t0) / args.profile:.4f} ms per frame")
    raise SystemExit(0)

# The device has taken its clocks up before anything is timed.
for _ in range(3):
    evaluator.evaluate(frames, cells, **asked)
print(f"gradients asked: {args.gradients or 'none'}")
print("K, ms per frame (least of the repeats), ms per call, frames/s, in steps")
for count in counts:
    # At least 64 frames in a repeat, so that a call of few frames is not
    # timed once.
    calls = max(1, 64 // count)
    least = np.inf
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(calls):
            evaluator.evaluate(frames[:count], cells[:count], **asked)
        least = min(least, (time.perf_counter() - t0) / calls)
    per_frame = 1e3 * least / count
    print(f"{count}, {per_frame:.4f}, {1e3 * least:.3f}, {1e3 / per_frame:.0f}, "
          f"{per_frame / step_ms:.1f}", flush=True)
