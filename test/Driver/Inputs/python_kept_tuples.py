"""The structures of the tuples that a simulation keeps from one activation
of its entry to the next (D[activation-start], #294,
docs/jit-invariants.md).

Usage:
  python_kept_tuples.py store
  python_kept_tuples.py loud [child]
  python_kept_tuples.py starts ROOT TARGET WORK [--write FILE | --compare FILE]

store   The store of the runtime of the host, called as the code of a
        program calls it: the renumbered members and the incidence
        structure against NumPy, written here; a structure is taken as it
        is only where the identities and the members are those of before,
        and built anew where either differs in one element.
loud    With MDRT_CHECK_KEPT_TUPLES, a kept structure that is no longer
        what would be built ends the program with a message (in a child
        process, which writes into the kept memory).
starts  A sequence of starts of an activation of the dipeptide in water, in
        double and mixed precision, in the deterministic mode: evaluations,
        an update of the tunables, steps that order the particles anew, a
        commit of a borrow, a checkpoint, a run continued from it, and (on
        the CPU, where NumPy writes the buffer) a commit that moves a
        particle. Every state, energy, and derivative of a simulation that
        keeps its structures equals, to the bit, that of one that builds
        them anew at every start; the counts of the structures taken and
        built say that both happened where they must. With `--write FILE`
        the records are written instead, and with `--compare FILE` compared
        with those written by another build (`main`).
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
import mdir


# --- The store, through the C interface of the code ----------------------------

class Memref1(ctypes.Structure):
    _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                ("offset", ctypes.c_int64), ("size", ctypes.c_int64),
                ("stride", ctypes.c_int64)]


class Memref2(ctypes.Structure):
    _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                ("offset", ctypes.c_int64), ("sizes", ctypes.c_int64 * 2),
                ("strides", ctypes.c_int64 * 2)]


def runtime():
    here = os.path.dirname(os.path.abspath(mdir.__file__))
    candidates = [os.environ.get("MDIR_RUNTIME_DIR", ""), os.path.join(here, "lib"),
                  os.path.join(here, "..", "..", "lib"), os.path.join(here, "..", "..", "..", "lib")]
    for directory in candidates:
        path = os.path.join(directory, "libmdrt.so")
        if directory and os.path.exists(path):
            lib = ctypes.CDLL(os.path.realpath(path))
            lib.mdrtTuplesOpen.restype = ctypes.c_void_p
            for name in ("mdrtTuplesBegin", "mdrtTuplesEnter", "mdrtTuplesClose"):
                getattr(lib, name).argtypes = [ctypes.c_void_p]
            lib.mdrtTuplesCounts.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
            lib._mlir_ciface_mdrtTuplesIncidence.argtypes = [
                ctypes.POINTER(Memref2), ctypes.POINTER(Memref2), ctypes.c_int64]
            return lib
    raise SystemExit("cannot find libmdrt.so")


def memref1(array):
    return Memref1(array.ctypes.data, array.ctypes.data, 0, array.shape[0], 1)


def memref2(array):
    return Memref2(array.ctypes.data, array.ctypes.data, 0,
                   (ctypes.c_int64 * 2)(*array.shape), (ctypes.c_int64 * 2)(array.shape[1], 1))


def view(buffer):
    shape = (buffer.sizes[0], buffer.sizes[1])
    return np.ctypeslib.as_array(ctypes.cast(buffer.aligned, ctypes.POINTER(ctypes.c_int32)),
                                 shape=shape)


def renumber(lib, members, ids):
    result = Memref2()
    lib._mlir_ciface_mdrtTuplesRenumber(ctypes.byref(result), ctypes.byref(memref2(members)),
                                        ctypes.byref(memref1(ids)))
    return result


def incidence(lib, members, particles):
    result = Memref2()
    given = members if isinstance(members, Memref2) else memref2(members)
    lib._mlir_ciface_mdrtTuplesIncidence(ctypes.byref(result), ctypes.byref(given), particles)
    return result


def counts(lib, store):
    numbers = (ctypes.c_int64 * 2)()
    lib.mdrtTuplesCounts(store, numbers)
    return numbers[0], numbers[1]


def reference(members, ids):
    """The members at their places, and for each particle the rows of its
    tuples (the number of the tuple, the place of the particle in it, its
    members), in the order of the tuples: in NumPy and Python, from the
    definitions of md_exec.renumber and md_exec.build_incidence."""
    place = np.empty(len(ids), dtype=np.int64)
    place[ids] = np.arange(len(ids))
    placed = place[members].astype(np.int32)
    rows = [[] for _ in ids]
    for t, tuple_ in enumerate(placed):
        for s, particle in enumerate(tuple_):
            rows[particle].append([t, s, *tuple_])
    return placed, rows


def check_structure(structure, rows, arity):
    widest = max(len(row) for row in rows)
    assert structure.shape == (len(rows), 1 + widest * (arity + 2)), structure.shape
    for particle, row in enumerate(rows):
        assert structure[particle, 0] == len(row), particle
        got = structure[particle, 1:1 + len(row) * (arity + 2)].reshape(len(row), arity + 2)
        assert np.array_equal(got, np.array(row, dtype=np.int32).reshape(len(row), arity + 2)), particle


def run_store():
    lib = runtime()
    rng = np.random.default_rng(294)
    particles = 257
    sets = {arity: rng.integers(0, particles, size=(count, arity), dtype=np.int32)
            for arity, count in ((2, 300), (3, 181), (4, 97))}
    order = rng.permutation(particles).astype(np.int32)
    other = order.copy()
    other[[3, 200]] = other[[200, 3]]

    # Without a store every call builds, and the caller frees.
    for arity, members in sets.items():
        placed, rows = reference(members, order)
        taken = renumber(lib, members, order)
        assert np.array_equal(view(taken), placed), arity
        structure = incidence(lib, taken, particles)
        check_structure(view(structure), rows, arity)
        again = renumber(lib, members, order)
        assert again.allocated != taken.allocated
        for buffer in (structure, again, taken):
            lib._mlir_ciface_mdrtTuplesRelease(ctypes.byref(buffer))
    print("without a store: the renumbered members and the incidence structures of tuples of "
          "2, 3, and 4 equal the reference, each built for its caller")

    store = lib.mdrtTuplesOpen()
    lib.mdrtTuplesEnter(store)

    def start(ids, members_of=sets, expect=None):
        """One activation: every set renumbered and its structure taken."""
        before = counts(lib, store)
        lib.mdrtTuplesBegin(store)
        pointers = {}
        for arity, members in members_of.items():
            placed, rows = reference(members, ids)
            taken = renumber(lib, members, ids)
            assert np.array_equal(view(taken), placed), arity
            structure = incidence(lib, taken, particles)
            check_structure(view(structure), rows, arity)
            pointers[arity] = (taken.allocated, structure.allocated)
        after = counts(lib, store)
        change = (after[0] - before[0], after[1] - before[1])
        assert expect is None or change == expect, (change, expect)
        return pointers

    first = start(order, expect=(0, 3))
    assert start(order, expect=(3, 0)) == first
    # Another order: built anew, in the memory of the last, which is free.
    assert start(other, expect=(0, 3)) == first
    assert start(other, expect=(3, 0)) == first
    assert start(order, expect=(0, 3)) == first
    # One member of one set differs: that set is built, the others taken.
    changed = {arity: members.copy() for arity, members in sets.items()}
    changed[3][17, 1] = (changed[3][17, 1] + 1) % particles
    start(order, members_of=changed, expect=(2, 1))
    start(order, members_of=changed, expect=(3, 0))
    # The set of before is still there, under its members.
    start(order, expect=(3, 0))
    start(other, expect=(0, 3))
    # The order of the files, without identities: the structure of the
    # members given, kept by what they hold.
    lib.mdrtTuplesBegin(store)
    before = counts(lib, store)
    plain = incidence(lib, sets[4], particles)
    check_structure(view(plain), reference(sets[4], np.arange(particles, dtype=np.int32))[1], 4)
    lib.mdrtTuplesBegin(store)
    copy = sets[4].copy()
    assert incidence(lib, copy, particles).allocated == plain.allocated
    assert counts(lib, store) == (before[0] + 1, before[1] + 1)
    # What the code holds is not built over: a second order in the same
    # activation has memory of its own, and gives it back.
    lib.mdrtTuplesBegin(store)
    held = renumber(lib, sets[2], order)
    second = renumber(lib, sets[2], other)
    assert second.allocated != held.allocated
    assert np.array_equal(view(held), reference(sets[2], order)[0])
    lib._mlir_ciface_mdrtTuplesRelease(ctypes.byref(second))
    third = renumber(lib, sets[2], other)
    assert third.allocated == second.allocated
    lib.mdrtTuplesClose(store)
    print("with a store: a structure is taken where the identities and the members are those of "
          "before and built where one element of either differs; the order of the files is kept "
          "by its members; what the code holds is not built over")
    print("kept tuples store passed")


def run_loud():
    if len(sys.argv) > 2:
        lib = runtime()
        store = lib.mdrtTuplesOpen()
        lib.mdrtTuplesEnter(store)
        members = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        ids = np.array([3, 1, 0, 2], dtype=np.int32)
        lib.mdrtTuplesBegin(store)
        taken = renumber(lib, members, ids)
        structure = incidence(lib, taken, 4)
        if sys.argv[2] == "members":
            view(taken)[1, 0] ^= 1
        else:
            view(structure)[2, 2] ^= 1
        lib.mdrtTuplesBegin(store)
        taken = renumber(lib, members, ids)
        incidence(lib, taken, 4)
        print("not reached")
        return
    for what, text in (("members", "the renumbered members of a tuple set"),
                       ("incidence", "the incidence structure of a tuple set")):
        env = dict(os.environ, MDRT_CHECK_KEPT_TUPLES="1")
        child = subprocess.run([sys.executable, __file__, "loud", what], env=env,
                               capture_output=True, text=True)
        message = f"mdrt: {text} that was kept from an earlier activation differs"
        assert child.returncode != 0 and message in child.stderr, (child.returncode, child.stderr)
        assert "not reached" not in child.stdout
        # Without the variable nothing is built anew to compare.
        env.pop("MDRT_CHECK_KEPT_TUPLES")
        child = subprocess.run([sys.executable, __file__, "loud", what], env=env,
                               capture_output=True, text=True)
        assert child.returncode == 0 and "not reached" in child.stdout, child.stderr
    print("a kept structure written over ends the program under MDRT_CHECK_KEPT_TUPLES, with "
          "the structure named")
    print("kept tuples loud passed")


# --- Starts of an activation ----------------------------------------------------

def make(root, target, precision, deterministic=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, start = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [32, 32, 32]
    term = mdir.PairTerm()
    term.name, term.expression = "soft", "a*exp(-r/l)"
    term.constants = [("a", 2.0), ("l", 0.05)]
    system.pair_terms = [term]
    system.tunables = [mdir.Tunable("q", "charge"), mdir.Tunable("l", "l", term="soft")]
    system.tunable_gradient = True
    state = start.draw_velocities(system, 300.0, 7)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.001
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


class Sequence:
    """The sequence of starts on one simulation, with what each leaves."""

    def __init__(self, root, target, precision, work, keeps):
        self.root, self.target, self.precision = root, target, precision
        self.keeps = keeps
        self.records = {}
        self.taken = {}
        self.path = os.path.join(work, f"kept-{target}-{precision}-{int(keeps)}.h5")
        self.sim = self.simulation()

    def simulation(self, **arguments):
        sim = mdir.Simulation(make(self.root, self.target, self.precision), **arguments)
        sim.part_seconds = 1e9
        if hasattr(sim, "_keep_tuples"):
            sim._keep_tuples = self.keeps
        return sim

    def counts(self, sim=None):
        sim = sim or self.sim
        return tuple(sim._kept_tuples[:2]) if hasattr(sim, "_kept_tuples") else (0, 0)

    def state(self, name, sim=None):
        state = (sim or self.sim).state()
        self.records[name + ".x"] = np.array(state.positions)
        self.records[name + ".v"] = np.array(state.velocities)
        self.records[name + ".f"] = np.array(state.forces)
        self.records[name + ".cell"] = np.array(state.cell.diagonal)
        # The energies of the last evaluation, where the state is at one.
        energies = state.energies or {}
        self.records[name + ".u"] = np.array([energies[key] for key in sorted(energies)])

    def counted(self, name, call, sim=None):
        """What `call` took of the kept structures, and what it built."""
        before = self.counts(sim)
        result = call()
        after = self.counts(sim)
        self.taken[name] = (after[0] - before[0], after[1] - before[1])
        return result

    def gradient(self, name, sim=None):
        g = self.counted(name, (sim or self.sim).tunables.gradient, sim)
        self.records[name + ".energy"] = np.array([g.energy])
        for key in g.keys():
            self.records[name + ".d" + key] = np.array(g[key])

    def run(self):
        sim = self.sim
        sim.run(8)
        self.state("run 8")
        self.gradient("gradient")
        self.gradient("gradient again")
        sim.tunables.update({"l": sim.tunables["l"] * 1.1})
        self.gradient("after an update")
        sim.run(6)
        self.state("6 steps on")
        # Long enough for particles to leave their cells of the order.
        sim.run(400)
        self.state("400 steps on")
        self.gradient("after 400 steps")
        self.gradient("after 400 steps, again")
        # A commit of a borrow: the charges, in a buffer of the host.
        with sim.borrow() as borrow:
            q = np.from_dlpack(borrow.tunables["q"])
            q *= 1.02
            del q
            assert self.counted("the commit of the charges", borrow.commit) == ("tunables",)
        self.state("commit of the charges")
        self.gradient("after the commit")
        sim.run(5)
        self.state("5 steps after the commit")
        self.gradient("before the checkpoint")
        sim.save_checkpoint(self.path)
        self.gradient("after the checkpoint")
        sim.run(7)
        self.state("7 steps after the checkpoint")
        # A simulation that continues the file, with a store of its own.
        other = self.simulation(checkpoint=self.path)
        other.run(7)
        self.state("continued, 7 steps", other)
        self.gradient("continued", other)
        del other
        if self.target == "CPU":
            # A particle moved by more than a cell of the order.
            with sim.borrow() as borrow:
                x = np.from_dlpack(borrow.positions)
                ids = np.from_dlpack(borrow.ids)
                x[int(np.nonzero(ids == 40)[0][0])] += np.array([0.9, -0.7, 0.5])
                del x, ids
                assert self.counted("the commit of a moved particle",
                                    borrow.commit) == ("positions",)
            self.state("commit of a moved particle")
            self.gradient("after the moved particle")
            self.gradient("after the moved particle, again")
            sim.run(4)
            self.state("4 steps after the moved particle")
        self.final = self.counts()
        return self


def run_apart(root, target, precision):
    """Two simulations and a frame evaluator of one program: each keeps its
    own structures and takes none of another's, though the three build the
    same ones."""
    program = make(root, target, precision)
    first, second = mdir.Simulation(program), mdir.Simulation(program)
    for sim in (first, second):
        sim.part_seconds = 1e9
    first.run(8)
    one = first.tunables.gradient()
    sets = sum(first._kept_tuples[:2]) // 2
    assert sets > 0 and first._kept_tuples[1] >= sets
    # The second is at the state of the first, in its order, with its
    # members: a store that they shared would hand it the structures.
    assert second._kept_tuples[:2] == (0, 0)
    second.run(8)
    assert second._kept_tuples[:2] == (0, sets), second._kept_tuples
    other = second.tunables.gradient()
    assert other.energy == one.energy
    assert all(np.array_equal(other[key], one[key]) for key in one.keys())
    # The evaluator has a simulation of its own, in the order of the input:
    # it builds once and takes its structures at every start after.
    evaluator = mdir.FrameEvaluator(program)
    state = first.state()
    frames = np.array(state.positions)[None]
    cells = np.array(state.cell.diagonal)[None]
    before = evaluator.evaluate(frames, cells).energy.copy()
    private = evaluator._simulation
    assert private._kept_tuples[:2] == (0, sets), private._kept_tuples
    length = np.array(evaluator.tunables["l"])
    evaluator.tunables.update({"l": length * 1.1})
    evaluator.evaluate(frames, cells)
    evaluator.tunables.update({"l": length})
    after = evaluator.evaluate(frames, cells).energy
    assert np.array_equal(before, after), (before, after)
    taken, built = private._kept_tuples[:2]
    assert built == sets and taken >= 2 * sets, private._kept_tuples
    assert second._kept_tuples[0] + second._kept_tuples[1] == 2 * sets
    print(f"{target} {precision}: two simulations and a frame evaluator of one program keep "
          f"their structures apart; the evaluator builds its {sets} once and takes them at "
          f"every start after")


def run_starts():
    root, target, work = sys.argv[2], sys.argv[3], sys.argv[4]
    mode = sys.argv[5] if len(sys.argv) > 5 else None
    kept_here = hasattr(mdir.Simulation, "_keep_tuples")
    written = {}
    for precision in ("Double", "Mixed"):
        subject = Sequence(root, target, precision, work, keeps=True).run()
        for name, values in subject.records.items():
            written[f"{precision}: {name}"] = values
        if not kept_here:
            continue
        anew = Sequence(root, target, precision, work, keeps=False).run()
        assert subject.records.keys() == anew.records.keys()
        for name, values in subject.records.items():
            assert np.array_equal(values, anew.records[name]), (precision, name)
        # A simulation that keeps nothing takes nothing.
        assert all(taken == 0 for taken, _ in anew.taken.values()), anew.taken
        assert anew.final[0] == 0, anew.final
        # Where nothing has moved since the last start, every structure is
        # taken and none is built; after the steps, and after the particle
        # was moved, they are built.
        sets = subject.taken["gradient again"][0]
        assert sets > 0, subject.taken
        still = ["gradient again", "after an update", "after 400 steps, again",
                 "the commit of the charges", "after the commit", "after the checkpoint"]
        # (A simulation that continues a file has kept nothing yet.)
        moved = ["after 400 steps", "continued"]
        if target == "CPU":
            still += ["after the moved particle", "after the moved particle, again"]
            moved.append("the commit of a moved particle")
        for name in still:
            assert subject.taken[name] == (sets, 0), (name, subject.taken)
        for name in moved:
            assert subject.taken[name] == (0, sets), (name, subject.taken)
        print(f"{target} {precision}: {len(subject.records)} states, energies, and derivatives "
              f"over {len(subject.taken)} evaluations, an update, a commit, a checkpoint, and a "
              f"run continued from it equal those of a simulation that builds its structures "
              f"anew, to the bit; {sets} structures taken where nothing moved, {sets} built "
              f"after the particles were ordered anew")
    if kept_here:
        run_apart(root, target, "Mixed")
    if target == "GPU" and kept_here:
        # Outside the deterministic mode a GPU evaluates each tuple once,
        # from a copy of the members on the device, which is kept with
        # their structure (and checked with it, under the variable). Its
        # sums are not in a fixed order, so the energies agree to rounding.
        sim = mdir.Simulation(make(root, target, "Mixed", deterministic=False))
        sim.part_seconds = 1e9
        sim.run(8)
        first = sim.tunables.gradient()
        before = sim._kept_tuples[:2]
        again = sim.tunables.gradient()
        sim.tunables.update({"l": sim.tunables["l"] * 1.1})
        sim.tunables.update({"l": sim.tunables["l"] / 1.1})
        third = sim.tunables.gradient()
        after = sim._kept_tuples[:2]
        assert after[1] == before[1] and after[0] > before[0], (before, after)
        for other in (again, third):
            assert abs(other.energy - first.energy) < 1e-6 * abs(first.energy), \
                (other.energy, first.energy)
        print("GPU Mixed, not deterministic: the copies of the members on the device are "
              "taken with their structures, and the energy of the state is that of the "
              "first evaluation to rounding")
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
              f"({'this build keeps its structures' if kept_here else 'this build keeps none'}); "
              f"largest difference {worst}")
        assert worst == 0.0
    print(f"kept tuples starts {target} passed")


scenario = sys.argv[1]
if scenario == "store":
    run_store()
elif scenario == "loud":
    run_loud()
else:
    run_starts()
