"""What a start of an activation gives the entry (D258,
#326, docs/jit-invariants.md): the arrays of the program, which the entry
never writes, without a copy, and the conversions of them into the type of
the entry and the buffers of the state, kept from one start to the next.

On the dipeptide in water with PME, the shift at the cutoff with the
correction for the dispersion, a pair term a exp(-r/l), and a spring over
three pairs of atoms; tunable are the charges, sigma and epsilon of the
types with Lennard-Jones, both constants of the pair term, and the
constants of the springs, with the derivative in all of them.
Deterministic, in double and mixed precision.

Scenarios:
  sequence ROOT TARGET WORK [--write FILE | --compare FILE]
        A sequence of starts on one simulation: steps, `gradient()` again
        and again, an evaluation, an update of each kind of tunable and
        back, updates between steps, a commit of a borrow of the charges
        and (on the CPU) of the positions and the velocities, a checkpoint
        and a run continued from it, a frame evaluator beside its sampler,
        and a second simulation of the same program that runs between the
        starts of the first. Every state, force, energy, and derivative
        equals, to the bit, that of a simulation whose starts copy and
        convert every array anew (`Simulation._keep_start_arguments =
        False`, private), and the counts of what the starts handed without
        a copy, took as kept, and made (`Simulation._start_arguments`,
        private) are as they must be. With `--write FILE` the records are
        written, and with `--compare FILE` compared with those that another
        build wrote (a build before D258 has neither the
        switch nor the counts, and writes the records alone).
  spoil ROOT TARGET
        A byte of a buffer that the entry took without a copy, written over
        by the test as a write of the entry would: with MDRT_CHECK_KEPT_TUPLES
        the end of the activation finds it, names it, and ends the process;
        without it nothing is said.
"""
import os
import re
import subprocess
import sys

import numpy as np
import mdir

NAMES = ("l", "a", "sigma", "epsilon", "q", "k")
HERE = hasattr(mdir.Simulation, "_start_arguments")


def make(root, target, precision):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, start = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Shift
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [32, 32, 32]
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    system.pair_terms = [soft]
    spring = mdir.TupleTerm()
    spring.name, spring.expression, spring.arity = "spring", "0.5*k*(r - r0)^2", 2
    spring.particles = np.array([[1, 4], [4, 6], [6, 8]], dtype=np.int64)
    spring.parameters = [("k", np.array([500.0, 600.0, 600.0])),
                         ("r0", np.array([0.25, 0.26, 0.27]))]
    system.tuple_terms = [spring]
    state = start.draw_velocities(system, 300.0, 7)
    system.tunable_gradient = True
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.001
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = True
    # A type without Lennard-Jones keeps its values: the combining rule has
    # no derivative in an epsilon of 0, which the compile says, naming it.
    names = list(system.topology.type_names)
    left = set()
    while True:
        keep = np.array([name not in left for name in names])
        entries = np.where(keep, np.cumsum(keep) - 1, -1)
        system.tunables = [mdir.Tunable("l", "l", term="soft"),
                           mdir.Tunable("a", "a", term="soft"),
                           mdir.Tunable("sigma", "sigma", map=entries),
                           mdir.Tunable("epsilon", "epsilon", map=entries),
                           mdir.Tunable("q", "charge"),
                           mdir.Tunable("k", "k", term="spring")]
        try:
            return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
        except mdir.InputError as error:
            found = re.search(r"is 0 for the type (\S+), where the combining rule", str(error))
            if not found or found.group(1) in left:
                raise
            left.add(found.group(1))


class Sequence:
    """The sequence of starts on one simulation, with what each leaves."""

    def __init__(self, root, target, precision, work, keeps):
        self.root, self.target, self.precision = root, target, precision
        self.keeps = keeps
        self.records = {}
        # For each call that began activations: (handed, kept, made).
        self.counts = {}
        self.path = os.path.join(work, f"start-{target}-{precision}-{int(keeps)}.h5")
        self.program = make(root, target, precision)
        self.sim = self.simulation()

    def simulation(self, **arguments):
        sim = mdir.Simulation(self.program, **arguments)
        sim.part_seconds = 1e9
        if HERE:
            sim._keep_start_arguments = self.keeps
        return sim

    def counted(self, name, call, sim=None):
        """`call`, with what its starts gave the entry."""
        sim = sim or self.sim
        before = sim._start_arguments if HERE else (0, 0, 0)
        result = call()
        after = sim._start_arguments if HERE else (0, 0, 0)
        self.counts[name] = tuple(b - a for a, b in zip(before, after))
        return result

    def state(self, name, sim=None):
        state = (sim or self.sim).state()
        self.records[name + ".x"] = np.array(state.positions)
        self.records[name + ".v"] = np.array(state.velocities)
        self.records[name + ".f"] = np.array(state.forces)
        energies = state.energies or {}
        self.records[name + ".u"] = np.array([energies[key] for key in sorted(energies)])

    def gradient(self, name, sim=None):
        g = self.counted(name, (sim or self.sim).tunables.gradient, sim)
        self.records[name + ".energy"] = np.array([g.energy])
        for key in g.keys():
            self.records[name + ".d" + key] = np.array(g[key])
        return g

    def update(self, name, values, sim=None):
        sim = sim or self.sim
        self.counted(name + " (update)", lambda: sim.tunables.update(values), sim)

    def run(self):
        sim = self.sim
        base = {name: np.array(sim.tunables[name]) for name in NAMES}
        other = {name: value * factor for (name, value), factor in
                 zip(base.items(), (1.1, 0.9, 1.02, 1.05, 0.97, 1.2))}
        other["q"][::3] *= 1.01
        self.counted("run 8", lambda: sim.run(8))
        self.state("run 8")
        first = self.gradient("gradient")
        self.gradient("gradient again")
        self.gradient("gradient a third time")
        # An evaluation of the state, which begins an activation.
        self.counted("evaluation", lambda: sim.run(0, energy=True))
        self.state("evaluation")
        # Each tunable alone, at the state of step 8, and back.
        for name in NAMES:
            self.update(f"{name} alone", {name: other[name]})
            self.state(f"{name} alone")
            self.gradient(f"{name} alone")
            self.update(f"{name} back", {name: base[name]})
        back = self.gradient("back")
        assert back.energy == first.energy, (back.energy, first.energy)
        # All together, and steps at those values.
        self.update("all together", other)
        self.state("all together")
        self.gradient("all together")
        sim.run(6)
        self.state("6 steps on")
        # Updates between steps, one kind at a time.
        for k, name in enumerate(NAMES):
            self.update(f"{name} between steps", {name: base[name]})
            sim.run(3 + k)
            self.state(f"{name} between steps")
        self.gradient("after the steps")
        # A commit of a borrow: the charges, in a buffer of the host.
        with sim.borrow() as borrow:
            q = np.from_dlpack(borrow.tunables["q"])
            q *= 1.02
            del q
            assert self.counted("the commit of the charges", borrow.commit) == ("tunables",)
        self.state("commit of the charges")
        self.gradient("after the commit of the charges")
        if self.target == "CPU":
            # A commit of the state, written where the activation keeps it:
            # in the buffers that the simulation keeps for its starts.
            with sim.borrow() as borrow:
                x = np.from_dlpack(borrow.positions)
                v = np.from_dlpack(borrow.velocities)
                x += 0.001
                v *= 0.98
                del x, v
                assert self.counted("the commit of the state",
                                    borrow.commit) == ("positions", "velocities")
            self.state("commit of the state")
            self.gradient("after the commit of the state")
            sim.run(4)
            self.state("4 steps after the commit of the state")
        # A second simulation of the same program, whose starts come between
        # those of the first: it shares the arrays of the program.
        second = self.simulation()
        second.run(5)
        self.state("second, 5 steps", second)
        self.gradient("second", second)
        self.update("second", {"q": other["q"]}, second)
        self.gradient("first, beside the second")
        self.gradient("second, updated", second)
        sim.run(3)
        second.run(3)
        self.state("first, 3 steps beside the second")
        self.state("second, 3 steps more", second)
        del second
        # A checkpoint, and a simulation that continues it.
        sim.save_checkpoint(self.path)
        self.gradient("after the checkpoint")
        sim.run(7)
        self.state("7 steps after the checkpoint")
        continued = self.simulation(checkpoint=self.path)
        continued.run(7)
        self.state("continued, 7 steps", continued)
        self.gradient("continued", continued)
        self.update("continued", {"a": other["a"]}, continued)
        self.gradient("continued, updated", continued)
        del continued
        # A frame evaluator beside its sampler, with a simulation of its
        # own: the frame is the state of the sampler.
        evaluator = mdir.FrameEvaluator(self.program)
        if HERE:
            evaluator._simulation._keep_start_arguments = self.keeps
        state = sim.state()
        frames = np.array(state.positions)[None]
        cells = np.array(state.cell.diagonal)[None]
        at_base = evaluator.evaluate(frames, cells).energy.copy()
        self.records["frame.energy"] = at_base
        for name in NAMES:
            evaluator.tunables.update({name: other[name]})
            self.records[f"frame, {name}.energy"] = evaluator.evaluate(frames, cells).energy.copy()
        evaluator.tunables.update(base)
        assert np.array_equal(evaluator.evaluate(frames, cells).energy, at_base)
        self.evaluator_counts = evaluator._simulation._start_arguments if HERE else None
        self.update("after the evaluator", {"k": other["k"]})
        sim.run(4)
        self.state("4 steps after the evaluator")
        return self


def check_counts(subject, anew, precision):
    """What the starts handed, took, and made: a start hands the same
    arrays every time; one at which no array of another type has changed
    takes every conversion as kept and makes none; a simulation that keeps
    nothing hands nothing."""
    starts = {name: c for name, c in subject.counts.items() if sum(c) > 0}
    handed = {c[0] for c in starts.values()}
    # Each start of a simulation hands the same arrays; a call that begins
    # two starts hands twice as many.
    unit = min(handed)
    assert unit > 0 and all(h % unit == 0 for h in handed), starts
    converted = {(c[1] + c[2]) * unit // c[0] for c in starts.values()}
    assert len(converted) == 1, starts
    slots = converted.pop()
    if precision == "Mixed":
        # The fields of the particles in f32 at least.
        assert slots > 0, starts
    # `gradient()` again: the particles and the values are those of the last
    # start, and every conversion is taken as it was kept.
    for name in ("gradient again", "gradient a third time", "after the checkpoint"):
        assert starts[name] == (unit, slots, 0), (name, starts[name])
    # An update of the charges makes the conversion of the charges anew in
    # the precision whose charges are converted, and the start after it takes
    # it as kept.
    if precision == "Mixed":
        assert starts["q alone (update)"][2] == 1, starts["q alone (update)"]
    assert starts["q alone"] == (unit, slots, 0), starts["q alone"]
    # A simulation whose starts copy everything hands nothing.
    for name, c in anew.counts.items():
        assert c[0] == 0 and c[1] == 0, (name, c)
    return unit, slots


def run_sequence():
    root, target, work = sys.argv[2], sys.argv[3], sys.argv[4]
    mode = sys.argv[5] if len(sys.argv) > 5 else None
    checked = "MDRT_CHECK_KEPT_TUPLES" in os.environ
    written = {}
    for precision in ("Double", "Mixed"):
        subject = Sequence(root, target, precision, work, keeps=True).run()
        for name, values in subject.records.items():
            written[f"{precision}: {name}"] = values
        if not HERE:
            continue
        anew = Sequence(root, target, precision, work, keeps=False).run()
        assert subject.records.keys() == anew.records.keys()
        for name, values in subject.records.items():
            assert np.array_equal(values, anew.records[name]), (precision, name)
        unit, slots = check_counts(subject, anew, precision)
        assert subject.evaluator_counts[0] > 0, subject.evaluator_counts
        print(f"{target} {precision}: {len(subject.records)} states, energies, and derivatives "
              f"over {len(subject.counts)} calls that begin activations (steps, gradient() "
              f"again, an evaluation, updates of each tunable, a commit of the charges"
              f"{' and of the state' if target == 'CPU' else ''}, a second simulation of the "
              f"program, a checkpoint with a run continued from it, and a frame evaluator) "
              f"equal those of a simulation whose starts copy every array anew, to the bit; "
              f"{unit} arrays handed without a copy at each start, {slots} conversions kept"
              f"{', each made anew and compared, and every buffer handed compared at the end of its activation' if checked else ''}")
    if mode == "--write":
        np.savez(sys.argv[6], **written)
        print(f"written: {len(written)} records")
    elif mode == "--compare":
        there = np.load(sys.argv[6])
        assert set(there.files) == set(written), set(there.files) ^ set(written)
        worst = 0.0
        for name, values in written.items():
            if not np.array_equal(values, there[name]):
                worst = max(worst, float(np.abs(values - there[name]).max()))
                print(f"differs: {name}")
        print(f"{target}: {len(written)} records compared with those of the other build; "
              f"largest difference {worst}")
        assert worst == 0.0
    print(f"start arguments sequence {target} passed")


def run_spoil_child():
    root, target = sys.argv[2], sys.argv[3]
    sim = mdir.Simulation(make(root, target, "Double"))
    sim.part_seconds = 1e9
    sim.run(4)
    assert sim._spoil_start_argument()
    # The update ends the activation, where the buffers are compared.
    sim.tunables.gradient()
    print("the activation ended without a word", flush=True)


def run_spoil():
    root, target = sys.argv[2], sys.argv[3]
    child = [sys.executable, __file__, "spoil-child", root, target]
    checked = dict(os.environ, MDRT_CHECK_KEPT_TUPLES="1")
    found = subprocess.run(child, env=checked, capture_output=True, text=True)
    assert found.returncode != 0, (found.returncode, found.stdout, found.stderr)
    assert ("the masses, which the entry of the program takes without a copy and never "
            "writes, differs at the end of the activation from what it was given; this is a "
            "defect of mdir") in found.stderr, found.stderr
    assert "without a word" not in found.stdout, found.stdout
    unchecked = {k: v for k, v in os.environ.items() if k != "MDRT_CHECK_KEPT_TUPLES"}
    quiet = subprocess.run(child, env=unchecked, capture_output=True, text=True)
    assert quiet.returncode == 0, (quiet.returncode, quiet.stdout, quiet.stderr)
    assert "differs at the end of the activation" not in quiet.stderr, quiet.stderr
    print(f"{target}: a byte of the masses that the entry took without a copy, written over, "
          f"ends the process at the end of the activation under the variable, naming it, and "
          f"is not looked for without it")
    print(f"start arguments spoil {target} passed")


scenario = sys.argv[1]
if scenario == "sequence":
    run_sequence()
elif scenario == "spoil":
    run_spoil()
elif scenario == "spoil-child":
    run_spoil_child()
else:
    raise SystemExit(f"unknown scenario {scenario}")
