"""What an update of the tunables keeps (D[update-cost],
docs/jit-invariants.md): the model that the values are put into, in place.

On the dipeptide in water with PME, the shift at the cutoff with the
correction for the dispersion, a pair term a exp(-r/l) whose tail is in the
correction, and a spring over three pairs of atoms; tunable are the charges,
sigma and epsilon of the types with Lennard-Jones, both constants of the
pair term, and the constants of the springs, with the derivative in all of
them. Deterministic, in double and mixed precision.

Scenarios:
  updates ROOT TARGET WORK [--write FILE | --compare FILE]
        A sequence on one simulation: an update of each tunable alone, of
        all together, and back to the values of before; updates between
        steps; a commit of a borrow; `gradient()`; a checkpoint and a run
        continued from it; an update that is refused, and the one after it;
        a frame evaluator beside its sampler. Every state, force, energy,
        and derivative equals, to the bit, that of a simulation whose
        updates build their programs from a copy of the model of the
        compile (`Simulation._keep_model = False`, private), and the counts
        of the programs built from the kept model and from a copy
        (`Simulation._tunable_builds`, private) are as they must be. With
        `--write FILE` the records are written, and with `--compare FILE`
        compared with those that another build wrote.
  loud ROOT TARGET
        A number of the kept model that no tunable owns, written over by
        the test: with MDRT_CHECK_KEPT_TUPLES the next update fails and
        names what differs, and the simulation builds every program from a
        copy of the model of the compile from then on, with the right
        results.
"""
import os
import re
import sys

import numpy as np
import mdir

NAMES = ("l", "a", "sigma", "epsilon", "q", "k")


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
    """The sequence of updates on one simulation, with what each leaves."""

    def __init__(self, root, target, precision, work, keeps):
        self.root, self.target, self.precision = root, target, precision
        self.keeps = keeps
        self.records = {}
        self.builds = {}
        self.path = os.path.join(work, f"update-{target}-{precision}-{int(keeps)}.h5")
        self.program = make(root, target, precision)
        self.sim = self.simulation()

    def simulation(self, **arguments):
        sim = mdir.Simulation(self.program, **arguments)
        sim.part_seconds = 1e9
        if hasattr(sim, "_keep_model"):
            sim._keep_model = self.keeps
        return sim

    def state(self, name, sim=None):
        state = (sim or self.sim).state()
        self.records[name + ".x"] = np.array(state.positions)
        self.records[name + ".v"] = np.array(state.velocities)
        self.records[name + ".f"] = np.array(state.forces)
        energies = state.energies or {}
        self.records[name + ".u"] = np.array([energies[key] for key in sorted(energies)])

    def gradient(self, name, sim=None):
        g = (sim or self.sim).tunables.gradient()
        self.records[name + ".energy"] = np.array([g.energy])
        for key in g.keys():
            self.records[name + ".d" + key] = np.array(g[key])
        return g

    def update(self, name, values, sim=None):
        """An update, with the programs that it built from the kept model
        and from a copy of the model of the compile."""
        sim = sim or self.sim
        before = sim._tunable_builds if hasattr(sim, "_tunable_builds") else (0, 0)
        sim.tunables.update(values)
        after = sim._tunable_builds if hasattr(sim, "_tunable_builds") else (0, 0)
        self.builds[name] = (after[0] - before[0], after[1] - before[1])

    def run(self):
        sim = self.sim
        base = {name: np.array(sim.tunables[name]) for name in NAMES}
        other = {name: value * factor for (name, value), factor in
                 zip(base.items(), (1.1, 0.9, 1.02, 1.05, 0.97, 1.2))}
        other["q"][::3] *= 1.01
        sim.run(8)
        self.state("run 8")
        first = self.gradient("gradient")
        # Each tunable alone, at the state of step 8, and back.
        for name in NAMES:
            self.update(f"{name} alone", {name: other[name]})
            self.state(f"{name} alone")
            self.gradient(f"{name} alone")
            self.update(f"{name} back", {name: base[name]})
        back = self.gradient("back")
        assert back.energy == first.energy, (back.energy, first.energy)
        for key in first.keys():
            assert np.array_equal(back[key], first[key]), key
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
            assert borrow.commit() == ("tunables",)
        self.state("commit of the charges")
        self.gradient("after the commit")
        # An update that is refused, a tail that the build cannot integrate,
        # after its values were put into the model that the updates keep;
        # and the one after.
        version = sim.tunables.version
        try:
            sim.tunables.update({"sigma": other["sigma"], "l": np.array([-0.05])})
            raise AssertionError("the update was not refused")
        except mdir.InputError as error:
            assert "its tail cannot be integrated" in str(error), str(error)
        assert sim.tunables.version == version
        assert np.array_equal(sim.tunables["sigma"], base["sigma"])
        self.gradient("after the refusal")
        self.update("after the refusal", {"epsilon": other["epsilon"]})
        self.state("after the refusal")
        self.gradient("the update after the refusal")
        sim.run(5)
        self.state("5 steps on")
        # A checkpoint, and a simulation that continues it: its program is
        # the compiled one, and takes the values of the file.
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
        if hasattr(evaluator._simulation, "_keep_model"):
            evaluator._simulation._keep_model = self.keeps
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
        self.evaluator_builds = (evaluator._simulation._tunable_builds
                                 if hasattr(evaluator._simulation, "_tunable_builds") else None)
        # The sampler has not seen the updates of the evaluator.
        self.update("after the evaluator", {"k": other["k"]})
        sim.run(4)
        self.state("4 steps after the evaluator")
        self.final = sim._tunable_builds if hasattr(sim, "_tunable_builds") else None
        return self


def run_updates():
    root, target, work = sys.argv[2], sys.argv[3], sys.argv[4]
    mode = sys.argv[5] if len(sys.argv) > 5 else None
    here = hasattr(mdir.Simulation, "_keep_model")
    checked = "MDRT_CHECK_KEPT_TUPLES" in os.environ
    written = {}
    for precision in ("Double", "Mixed"):
        subject = Sequence(root, target, precision, work, keeps=True).run()
        for name, values in subject.records.items():
            written[f"{precision}: {name}"] = values
        if not here:
            continue
        anew = Sequence(root, target, precision, work, keeps=False).run()
        assert subject.records.keys() == anew.records.keys()
        for name, values in subject.records.items():
            assert np.array_equal(values, anew.records[name]), (precision, name)
        # A simulation that keeps no model counts nothing.
        assert anew.final == (0, 0), anew.final
        # Every update but a refused one built one program from the kept
        # model (the one from a copy beside it, under the variable, is the
        # comparison and is not counted), and none from a copy alone.
        kept, copied = subject.final
        assert kept > 0 and copied == 0, subject.final
        for name, counts in subject.builds.items():
            assert counts == (1, 0), (name, counts)
        assert subject.evaluator_builds == (len(NAMES) + 1, 0), subject.evaluator_builds
        print(f"{target} {precision}: {len(subject.records)} states, energies, and derivatives "
              f"over {len(subject.builds)} updates of each tunable alone and of all, between "
              f"steps, a commit, a refusal, a checkpoint with a run continued from it, and a "
              f"frame evaluator equal those of a simulation whose updates copy the model of "
              f"the compile, to the bit; {kept} programs built from the kept model"
              f"{', each compared with one built from a copy' if checked else ''}")
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
        print(f"{target}: {len(written)} records compared with those of the other build "
              f"({'this build keeps its model' if here else 'this build keeps none'}); "
              f"largest difference {worst}")
        assert worst == 0.0
    print(f"update cost updates {target} passed")


def run_loud():
    root, target = sys.argv[2], sys.argv[3]
    assert "MDRT_CHECK_KEPT_TUPLES" in os.environ
    program = make(root, target, "Double")
    sim, right = mdir.Simulation(program), mdir.Simulation(program)
    for one in (sim, right):
        one.part_seconds = 1e9
        one.run(4)
    length = np.array(sim.tunables["l"])
    right.tunables.update({"l": length * 1.1})
    assert sim._spoil_kept_model()
    # The comparison of every update: the update fails, names what differs
    # (the length of the first bond), and changes nothing.
    try:
        sim.tunables.update({"l": length * 1.1})
        raise AssertionError("the spoiled model was not found")
    except mdir.SimulationError as error:
        message = str(error)
        assert "differs from the one built from a copy of the model of the compile" in message, \
            message
        assert "the field 'r0' of 'bonds'" in message, message
    assert sim.tunables.version == 0
    assert np.array_equal(sim.tunables["l"], length)
    # The next one builds from a copy of the model of the compile.
    sim.tunables.update({"l": length * 1.1})
    assert sim._tunable_builds == (1, 1), sim._tunable_builds
    for one in (sim, right):
        one.run(4)
    a, b = sim.state(), right.state()
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.forces, b.forces)
    assert sim.tunables.gradient().energy == right.tunables.gradient().energy
    print(f"{target}: a number of the kept model that was written over is found by the "
          f"comparison with a build from a copy of the model of the compile, which fails the "
          f"update and names it; the simulation builds from a copy from then on, with the "
          f"results of one that was not spoiled")
    print(f"update cost loud {target} passed")


scenario = sys.argv[1]
if scenario == "updates":
    run_updates()
elif scenario == "loud":
    run_loud()
else:
    raise SystemExit(f"unknown scenario {scenario}")
