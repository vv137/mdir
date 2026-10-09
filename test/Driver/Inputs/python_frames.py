"""The frame evaluator (D240, docs/python-frames.md).

Usage: python_frames.py ROOT SCENARIO TARGET PRECISION [MIXTURE | FILE | WORK]

ROOT is Inputs/dipeptide; MIXTURE a directory that pair_tail_system.py
filled with 60 A + 60 B; FILE the H5MD trajectory of a run of `mdir run`
at constant pressure in a rhombic dodecahedron; WORK a directory to write
in.

Scenarios:
  frames   the dipeptide in water with PME, an observed pair term and an
           observed wall; tied charges and sigma and epsilon of a pair of
           types tunable (a tunable constant of a pair term is in `mixture`:
           with tunable charges it does not lower, #256). The energy, the derivative, the virial, and the observed
           columns of each frame against the sampler's own gradient() and
           state at that step; frames as states, arrays, float32, in
           another order, from a generator; the product with the Jacobian
           kept and by a second pass, against central differences in theta
  npt      frames of leapfrog under a barostat, each with its cell, with a
           plain cutoff, the correction for the dispersion, PME, the table
           by pairs of types and the charges tunable: against the sampler's
           gradient() at each frame, against a simulation compiled from the
           frame, and, at values of the charges with a net charge, the
           product against central differences of the energies
  refusals what is refused, with which error, and that the evaluator stays
           usable (CPU)
  mixture  60 A + 60 B with the A-B pair term tunable: the derivative of a
           reweighted average, from the energies and the Jacobian of the
           evaluator, against central differences in sigma' of the
           reweighted average on the same frames; uniform weights at the
           values that sampled
  triclinic  the frames of a run at constant pressure in a triclinic cell,
           whose barostat scales the tilts, read with mdir.read_h5md: the
           energy, the derivative in the charges and in sigma of a pair of
           types, the virial, and the volume of each against a simulation
           compiled from that frame; the same frames as arrays with their
           tilts and as pairs with the matrix of the cell vectors; the
           product against central differences; cells that are refused
  h5md     frames that an H5MDReporter wrote from a run of the dipeptide,
           read with mdir.read_h5md and evaluated one at a time: in f64
           against the states of the same steps, to the bit, and in f32
  torch    the same average differentiated by PyTorch through
           mdir.torch.evaluate; gradcheck; a fit of sigma' to a synthetic
           target; refusals of the adapter; torch.compile
"""
import sys

import numpy as np
import mdir

root, scenario, target, precision = sys.argv[1:5]
mixture = extra = sys.argv[5] if len(sys.argv) > 5 else None
double = precision == "Double"
mode = f"{target} {precision}"


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text})")


def execution_(deterministic=True):
    execution = mdir.Execution()
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return execution


def simulation(program):
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    return sim


def tied_charges(system):
    """One entry per atom name of the waters and one per atom of the
    peptide."""
    top = system.topology
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(top.atom_names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    return np.array(charge_map, dtype=np.int64)


def one_pair(system, a, b):
    top = system.topology
    names, pairs = list(top.type_names), np.array(top.type_pairs)
    a, b = sorted((names.index(a), names.index(b)))
    return np.where((pairs[:, 0] == a) & (pairs[:, 1] == b), 0, -1)


def relative(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def product_differences(evaluator, frames, cotangent, product, entries, step):
    """The largest difference, relative to the largest entry of each
    tunable, between the product `product` and central differences in the
    entries `entries[name]` of sum_n g_n U_n, extrapolated from the steps
    `step` and `step` / 2 times each value (Richardson)."""
    worst = 0.0
    for name, chosen in entries.items():
        v0 = evaluator.tunables[name].copy()
        scale = max(np.abs(product[name]).max(), 1e-12)
        for m in chosen:
            def central(h):
                sums = []
                for sign in (1.0, -1.0):
                    v = v0.copy()
                    v[m] += sign * h
                    evaluator.tunables[name] = v
                    sums.append(float(cotangent @ evaluator.evaluate(frames, gradient=False).energy))
                return (sums[0] - sums[1]) / (2.0 * h)
            h = step * max(abs(v0[m]), 1e-3)
            fd = (4.0 * central(0.5 * h) - central(h)) / 3.0
            worst = max(worst, abs(product[name][m] - fd) / scale)
        evaluator.tunables[name] = v0
    return worst


def largest(values, count):
    order = np.argsort(-np.abs(values))
    return [int(m) for m in order[:count]]


# The dipeptide in water with PME.
def dipeptide_model(gradient=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    soft.observe = ["l"]
    wall = mdir.ExternalTerm()
    wall.name, wall.expression, wall.selection = "wall", "0.5*k*max(0, z - z0)^2", ":WAT"
    wall.constants = [("k", 100.0), ("z0", 1.5)]
    wall.observe = ["z0"]
    system.pair_terms, system.external_terms = [soft], [wall]
    system.tunables = [mdir.Tunable("q", "charge", map=tied_charges(system)),
                       mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW")),
                       mdir.Tunable("epsilon", "epsilon_pair", map=one_pair(system, "OW", "OW"))]
    system.tunable_gradient = gradient
    return system, state


def dipeptide_program(deterministic=True, gradient=True):
    system, state = dipeptide_model(gradient)
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    return mdir.compile(system, state, integrator, mdir.Ensemble(), execution_(deterministic),
                        mdir.Schedule())


def sample(sim, count, stride):
    """`count` states `stride` steps apart with the derivative of the
    sampler itself at each."""
    states, gradients = [], []
    for _ in range(count):
        sim.run(stride)
        gradients.append(sim.tunables.gradient())
        states.append(sim.state())
    return states, gradients


def run_frames():
    program = dipeptide_program()
    sim = simulation(program)
    states, gradients = sample(sim, 5, 4)
    count = len(states)
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(states)
    names = list(gradients[0].keys())
    assert out.count == count and out.version == 0 and out.zero == frozenset()
    assert out.energy.shape == (count,) and out.energy.dtype == np.float64
    assert not out.energy.flags.writeable
    assert out.jacobian["q"].shape == (count, len(sim.tunables["q"]))
    # Each frame against the sampler at that state: the energy that the
    # derivative is of, the derivative, the virial, the volume, the columns.
    exact = True
    worst_energy = worst_row = worst_virial = worst_column = 0.0
    for n, (state, g) in enumerate(zip(states, gradients)):
        exact = exact and out.energy[n] == g.energy and out.virial[n] == state.energies["virial"]
        exact = exact and out.volume[n] == state.energies["volume"]
        worst_energy = max(worst_energy, abs(out.energy[n] - g.energy))
        worst_virial = max(worst_virial, abs(out.virial[n] - state.energies["virial"]))
        for name in names:
            exact = exact and np.array_equal(out.jacobian[name][n], g[name])
            worst_row = max(worst_row, relative(out.jacobian[name][n], g[name]))
        assert list(out.observables) == list(state.observables), list(out.observables)
        for column, value in state.observables.items():
            exact = exact and out.observables[column][n] == value
            worst_column = max(worst_column, abs(out.observables[column][n] - value))
    print(f"{mode}: {count} frames against the sampler's gradient() and state: the energy "
          f"{worst_energy:.1e} kJ/mol, the derivative {worst_row:.1e} relative, the virial "
          f"{worst_virial:.1e} kJ/mol, the observed columns {worst_column:.1e}; equal to the "
          f"bit: {exact}")
    assert exact

    # What an output may read.
    reads = frozenset({"q", "sigma", "epsilon"})
    assert out.depends["energy"] == reads and out.depends["virial"] == reads
    assert out.depends["volume"] == frozenset()
    assert out.depends["soft.d_l"] == frozenset({"q"}), out.depends
    assert out.depends["wall.d_z0"] == frozenset({"q"}), out.depends

    # The same frames as an array, in another order, as pairs from a
    # generator, and one frame alone: nothing is carried between frames.
    positions = np.stack([s.positions for s in states])
    by_array = evaluator.evaluate(positions)
    order = np.array([3, 0, 4, 1, 2])
    shuffled = evaluator.evaluate(positions[order])
    pairs = evaluator.evaluate((s.positions, s.cell) for s in states)
    one = evaluator.evaluate(states[2])
    same = (np.array_equal(by_array.energy, out.energy) and
            np.array_equal(shuffled.energy, out.energy[order]) and
            np.array_equal(pairs.energy, out.energy) and one.energy[0] == out.energy[2] and
            all(np.array_equal(shuffled.jacobian[name], out.jacobian[name][order]) and
                np.array_equal(pairs.jacobian[name], out.jacobian[name]) for name in names))
    print(f"{mode}: as an array, in another order, from a generator, and one frame alone, "
          f"equal to the bit: {same}")
    assert same
    single = evaluator.evaluate(positions.astype(np.float32))
    change = float(np.abs(single.energy - out.energy).max())
    print(f"{mode}: positions in float32 change the energy by at most {change:.1e} kJ/mol")
    assert 0.0 < change < 2e-2, change

    # The product: with the Jacobian kept, and by a second pass.
    cotangent = np.random.default_rng(7).normal(size=count)
    product = out.vjp(cotangent)
    assert product.zero == out.zero and product.version == 0 and list(product) == names
    second = mdir.FrameEvaluator(program, jacobian_bytes=0)
    without = second.evaluate(positions)
    assert without.jacobian is None and np.array_equal(without.energy, out.energy)
    passed = without.vjp(cotangent)
    worst = max(relative(product[name], cotangent @ out.jacobian[name]) for name in names)
    again = max(relative(passed[name], product[name]) for name in names)
    print(f"{mode}: the product against the rows {worst:.1e}, by a second pass {again:.1e}")
    assert worst < 1e-14 and again < 1e-13, (worst, again)
    entries = {"q": largest(product["q"], 3), "sigma": [0], "epsilon": [0]}
    fd = product_differences(evaluator, positions, cotangent, product, entries,
                             1e-3 if double else 5e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{mode}: the product against central differences in theta of the sum of the "
          f"energies, {fd:.1e} (tolerance {tolerance:.0e})")
    assert fd < tolerance, fd
    assert evaluator.tunables.version > 0 and np.array_equal(evaluator.evaluate(positions).energy,
                                                             out.energy)

    # The default mode of the precision, which is not deterministic.
    if not double:
        program = dipeptide_program(deterministic=False)
        sim = simulation(program)
        states, gradients = sample(sim, 3, 4)
        out = mdir.FrameEvaluator(program).evaluate(states)
        energy = max(abs(out.energy[n] - g.energy) for n, g in enumerate(gradients))
        rows = max(relative(out.jacobian[name][n], g[name]) for n, g in enumerate(gradients)
                   for name in names)
        print(f"{mode}: without the deterministic mode, the energy {energy:.1e} kJ/mol "
              f"(tolerance 5e-3), the derivative {rows:.1e} relative (tolerance 1e-5)")
        assert energy < 5e-3 and rows < 1e-5, (energy, rows)
    print("frames passed")


# Frames of a run under a barostat.
def npt_model(state=None, grid=None):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, first = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.EnergyPressure
    if grid is not None:
        system.pme_grid = grid
    system.tunables = [mdir.Tunable("sigma", "sigma_pair"), mdir.Tunable("epsilon", "epsilon_pair"),
                       mdir.Tunable("q", "charge", map=tied_charges(system))]
    system.tunable_gradient = True
    return system, first if state is None else state


def npt_program(state=None, grid=None):
    system, state = npt_model(state, grid)
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep = 0.0005
    integrator.method = mdir.IntegratorMethod.Leapfrog
    ensemble.kind, ensemble.temperature, ensemble.coupling_period = mdir.EnsembleKind.NPT, 300.0, 10
    return mdir.compile(system, state, integrator, ensemble, execution_(), mdir.Schedule())


def run_npt():
    program = npt_program()
    sim = simulation(program)
    states, gradients = sample(sim, 4, 50)
    count = len(states)
    names = list(gradients[0].keys())
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(states)
    volumes = np.array([float(np.linalg.det(s.cell.vectors)) for s in states])
    assert len(set(volumes)) == count and relative(out.volume, volumes) < 1e-14
    exact = all(out.energy[n] == g.energy and
                all(np.array_equal(out.jacobian[name][n], g[name]) for name in names)
                for n, g in enumerate(gradients))
    worst = max(max(relative(out.jacobian[name][n], g[name]) for name in names)
                for n, g in enumerate(gradients))
    print(f"{mode}: {count} frames of a run under a barostat, volumes {volumes.min():.4f} to "
          f"{volumes.max():.4f} nm^3, against the sampler's gradient() at each: the derivative "
          f"{worst:.1e} relative; equal to the bit: {exact}")
    assert exact
    # The edges as an array (K, 3), the form of a trajectory.
    positions = np.stack([s.positions for s in states])
    cells = np.stack([s.cell.diagonal for s in states])
    by_array = evaluator.evaluate(positions, cells)
    assert np.array_equal(by_array.energy, out.energy)
    # Without cells, the cell of the program, not that of the last frame.
    start = simulation(program)
    first = start.tunables.gradient()
    initial = start.state()
    assert evaluator.evaluate(initial.positions).energy[0] == first.energy

    # A simulation compiled from a frame, with its cell as the cell of the
    # program: what follows the volume is then computed at it, not scaled.
    grid = list(program.plan["pme_grid"])
    worst_energy = worst_row = 0.0
    for n in (0, count - 1):
        start = mdir.InitialState.from_state(states[n], velocities=False)
        fresh = simulation(npt_program(start, grid)).tunables.gradient()
        worst_energy = max(worst_energy, abs(fresh.energy - out.energy[n]))
        worst_row = max(worst_row, max(relative(out.jacobian[name][n], fresh[name])
                                       for name in names))
    tolerances = (1e-9, 1e-12) if double else (5e-3, 1e-5)
    print(f"{mode}: against a simulation compiled from the frame: the energy {worst_energy:.1e} "
          f"kJ/mol (tolerance {tolerances[0]:.0e}), the derivative {worst_row:.1e} relative "
          f"(tolerance {tolerances[1]:.0e})")
    assert worst_energy < tolerances[0] and worst_row < tolerances[1], (worst_energy, worst_row)

    # At other values than those that sampled, with a net charge of 0.5 e
    # (the background of PME follows the volume of each frame).
    q = evaluator.tunables["q"].copy()
    charge_map = np.array(program.plan["tunables"][2]["map"])
    q[np.unique(charge_map[:22])] += 0.5 / 22
    evaluator.tunables.update({"q": q, "sigma": evaluator.tunables["sigma"] * 1.01})
    moved = evaluator.evaluate(states)
    assert moved.version == 1 and not np.array_equal(moved.energy, out.energy)
    cotangent = np.random.default_rng(11).normal(size=count)
    product = moved.vjp(cotangent)
    entries = {name: largest(product[name], 3) for name in names}
    fd = product_differences(evaluator, states, cotangent, product, entries,
                             1e-3 if double else 5e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{mode}: with a net charge of 0.5 e and sigma scaled by 1.01, the product against "
          f"central differences of the energies at the volume of each frame, {fd:.1e} "
          f"(tolerance {tolerance:.0e})")
    assert fd < tolerance, fd
    print("npt passed")


def run_refusals():
    program = dipeptide_program()
    sim = simulation(program)
    states, _ = sample(sim, 3, 2)
    positions = np.stack([s.positions for s in states])
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(positions)
    count = positions.shape[1]

    expect(mdir.InputError, lambda: evaluator.evaluate(positions[:, :-1]), f"(K, {count}, 3)")
    expect(mdir.InputError, lambda: evaluator.evaluate(positions.astype(np.int64)), "float64")
    expect(mdir.InputError, lambda: evaluator.evaluate([]), "no frames")
    bad = positions.copy()
    bad[1, 17, 2] = np.nan
    text = expect(mdir.InputError, lambda: evaluator.evaluate(bad), "frame 1: ")
    assert "particle 17" in text and "not finite" in text, text
    edges = np.tile(states[0].cell.diagonal, (3, 1))
    small = edges.copy()
    small[2, 0] = 1.5
    text = expect(mdir.InputError, lambda: evaluator.evaluate(positions, small), "frame 2: ")
    assert "twice the cutoff" in text, text
    expect(mdir.InputError, lambda: evaluator.evaluate(positions, edges[:2]), "cells")
    expect(mdir.InputError, lambda: evaluator.evaluate(states, edges), "an iterable gives")
    expect(mdir.InputError, lambda: evaluator.evaluate(positions, None, edges), "tilts")
    # Tilts in a program compiled for an orthorhombic cell (D238).
    text = expect(mdir.InputError,
                  lambda: evaluator.evaluate(positions, edges, np.full((3, 3), 0.1)), "frame 0: ")
    assert "compiled for an orthorhombic cell" in text and "committed" not in text, text
    vectors = np.diag(states[0].cell.diagonal)
    assert evaluator.evaluate([(positions[1], vectors)]).energy[0] == out.energy[1]
    vectors[0, 1] = 0.1
    expect(mdir.InputError, lambda: evaluator.evaluate([(positions[1], vectors)]),
           "lower-triangular")
    assert np.array_equal(evaluator.evaluate(positions, edges, np.zeros((3, 3))).energy, out.energy)
    # A frame whose evaluation fails: every particle at one place.
    collapsed = positions.copy()
    collapsed[2] = 1.0
    text = expect(mdir.SimulationError, lambda: evaluator.evaluate(collapsed), "frame 2: ")
    # The evaluator is usable after each refusal, and gives what it gave.
    assert np.array_equal(evaluator.evaluate(positions).energy, out.energy)
    print("refused: shapes, types, no frames, a position that is not finite, a cell below twice "
          "the cutoff, tilts in an orthorhombic program, a frame whose evaluation fails; the "
          "evaluator stays usable")

    # The Jacobian above its bound: a second pass needs the frames again.
    bounded = mdir.FrameEvaluator(program, jacobian_bytes=8 * 2 * evaluator._entries)
    assert bounded.evaluate(positions[:2]).jacobian is not None
    listed = bounded.evaluate(states)
    assert listed.jacobian is None
    expect(mdir.InputError, lambda: bounded.evaluate(iter(states)), "frame 2: the Jacobian passes")
    assert bounded.evaluate(iter(states), gradient=False).jacobian is None
    expect(mdir.InputError, lambda: bounded.evaluate(iter(states), gradient=False).vjp(np.ones(3)),
           "gradient=False")
    expect(mdir.InputError, lambda: out.vjp(np.ones(2)), "shape (3,)")
    cotangent = np.array([1.0, -2.0, 0.5])
    before = listed.vjp(cotangent)
    assert relative(before["q"], out.vjp(cotangent)["q"]) < 1e-13
    bounded.tunables["sigma"] = bounded.tunables["sigma"] * 1.01
    expect(mdir.SimulationError, lambda: listed.vjp(cotangent), "changed since")
    # With the Jacobian kept, the product is that of the values evaluated.
    evaluator.tunables["sigma"] = evaluator.tunables["sigma"] * 1.01
    assert np.array_equal(out.vjp(cotangent)["q"], cotangent @ out.jacobian["q"])
    print("refused: a generator when the Jacobian passes its bound, a product without the "
          "derivative, a second pass after the values changed")

    # A program without the derivative, and one without tunables.
    expect(mdir.InputError, lambda: mdir.FrameEvaluator(dipeptide_program(gradient=False)),
           "System.tunable_gradient = True")
    system, state = dipeptide_model(gradient=False)
    system.tunables = []
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    plain = mdir.compile(system, state, integrator, mdir.Ensemble(), execution_(), mdir.Schedule())
    expect(mdir.InputError, lambda: mdir.FrameEvaluator(plain), "declares no tunable")
    assert abs(mdir.KB - 0.0083144626181532) == 0.0
    print("refusals passed")


# Frames of a run at constant pressure in a rhombic dodecahedron of water.
def triclinic_program(state=None, capacity=0):
    triclinic = root + "/../triclinic"
    loaded = mdir.load_gromacs(triclinic + "/water.top", triclinic + "/dodecahedron.gro",
                               defines=["FLEXIBLE"])
    system, start = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28] * 3
    names = system.topology.atom_names
    water_map = np.array([i if i < 9 else 9 + (name != "OW") for i, name in enumerate(names)])
    system.tunables = [mdir.Tunable("q", "charge", map=water_map),
                       mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW"))]
    system.tunable_gradient = True
    integrator = mdir.Integrator()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.0005
    execution = execution_()
    execution.neighbor_capacity = capacity
    return mdir.compile(system, start if state is None else state, integrator, mdir.Ensemble(),
                        execution, mdir.Schedule())


def run_triclinic():
    trajectory = mdir.read_h5md(extra)
    count = len(trajectory)
    assert count == 20 and trajectory.dtype == np.float64, (count, trajectory.dtype)
    cells = [frame.cell for frame in trajectory]
    tilts = np.array([c.tilt for c in cells])
    diagonals = np.array([c.diagonal for c in cells])
    assert np.ptp(tilts[:, 1]) > 1e-4, "the barostat did not change the tilts"
    program = triclinic_program()
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(trajectory)
    names = list(out.jacobian)
    volumes = diagonals.prod(axis=1)
    assert relative(out.volume, volumes) < 1e-14 and len(set(out.volume)) == count
    # A simulation compiled from the frame, whose cell is then the cell of
    # the program.
    exact, worst_energy, worst_row, worst_virial = True, 0.0, 0.0, 0.0
    chosen = (0, 6, 13, 19)
    for n in chosen:
        frame = trajectory[n]
        start = mdir.InitialState()
        start.positions, start.cell = frame.positions, frame.cell
        start.velocities = np.zeros_like(frame.positions)
        fresh_program = triclinic_program(start, program.plan["neighbor_capacity"])
        assert fresh_program.ir == program.ir
        fresh = simulation(fresh_program)
        g = fresh.tunables.gradient()
        energies = fresh.state().energies
        exact = exact and out.energy[n] == g.energy and out.virial[n] == energies["virial"]
        exact = exact and out.volume[n] == energies["volume"]
        exact = exact and all(np.array_equal(out.jacobian[name][n], g[name]) for name in names)
        worst_energy = max(worst_energy, abs(out.energy[n] - g.energy))
        worst_virial = max(worst_virial, abs(out.virial[n] - energies["virial"]))
        worst_row = max(worst_row, max(relative(out.jacobian[name][n], g[name]) for name in names))
    print(f"{mode}: {count} frames of a run at constant pressure in a triclinic cell, read from "
          f"H5MD (c_x from {tilts[:, 1].min():.4f} to {tilts[:, 1].max():.4f} nm, volumes "
          f"{volumes.min():.4f} to {volumes.max():.4f} nm^3); {len(chosen)} of them against a "
          f"simulation compiled from the frame: the energy {worst_energy:.1e} kJ/mol, the "
          f"derivative {worst_row:.1e} relative, the virial {worst_virial:.1e} kJ/mol; equal to "
          f"the bit: {exact}")
    assert exact

    # The same frames as arrays with their tilts, as pairs with an
    # mdir.Cell, and as pairs with the matrix of the cell vectors, which a
    # frame of the reader unpacks to.
    positions = np.stack([frame.positions for frame in trajectory])
    by_array = evaluator.evaluate(positions, diagonals, tilts)
    by_cell = evaluator.evaluate(list(zip(positions, cells)))
    unpacked = [tuple(frame) for frame in trajectory]
    assert unpacked[0][1].shape == (3, 3)
    by_vectors = evaluator.evaluate(unpacked)
    same = all(np.array_equal(other.energy, out.energy) and
               all(np.array_equal(other.jacobian[name], out.jacobian[name]) for name in names)
               for other in (by_array, by_cell, by_vectors))
    print(f"{mode}: as arrays with tilts, as pairs with a cell, and as pairs with the cell "
          f"vectors, equal to the bit: {same}")
    assert same
    # Without a cell, the cell of the program with its tilts.
    start = simulation(program)
    first = start.tunables.gradient()
    assert evaluator.evaluate(start.state().positions).energy[0] == first.energy

    cotangent = np.random.default_rng(5).normal(size=count)
    product = out.vjp(cotangent)
    entries = {"q": largest(product["q"], 3), "sigma": [0]}
    fd = product_differences(evaluator, trajectory, cotangent, product, entries,
                             1e-3 if double else 5e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{mode}: the product against central differences in theta of the sum of the "
          f"energies, {fd:.1e} (tolerance {tolerance:.0e})")
    assert fd < tolerance, fd

    # What a commit asks of a cell (D238), as errors of a frame.
    bad = tilts.copy()
    bad[1, 0] = 0.6 * diagonals[1, 0]
    text = expect(mdir.InputError, lambda: evaluator.evaluate(positions, diagonals, bad),
                  "frame 1: ")
    assert "not reduced" in text and "committed" not in text and "written" not in text, text
    text = expect(mdir.InputError, lambda: evaluator.evaluate(positions, diagonals), "frame 0: ")
    assert "all zero" in text, text
    small = diagonals.copy()
    small[2, 2] = 1.5
    text = expect(mdir.InputError, lambda: evaluator.evaluate(positions, small, tilts), "frame 2: ")
    assert "twice the cutoff" in text, text
    assert np.array_equal(evaluator.evaluate(trajectory).energy, out.energy)
    print(f"{mode}: refused: a cell that is not reduced, a triclinic program without tilts, a "
          "diagonal below twice the cutoff; the evaluator stays usable")
    print("triclinic passed")


def run_h5md():
    import pathlib
    work = pathlib.Path(extra)
    program = dipeptide_program()
    results = {}
    for kind in ("f64", "f32"):
        path = work / f"frames-{target}-{precision}-{kind}.h5md"
        sim = simulation(program)
        states = []
        sim.reporters.append(mdir.H5MDReporter(str(path), 4, positions=kind))
        sim.reporters.append(mdir.CallbackReporter(lambda s, state: states.append(state), 4))
        sim.run(20)
        sim.close_reporters()
        trajectory = mdir.read_h5md(str(path))
        steps = {int(s): k for k, s in enumerate(trajectory.steps)}
        frames = [trajectory[steps[state.step]] for state in states]
        assert len(frames) == 5 and trajectory.dtype == (np.float64 if kind == "f64" else np.float32)
        evaluator = mdir.FrameEvaluator(program)
        from_states = evaluator.evaluate(states)
        from_file = evaluator.evaluate(frames)
        whole = evaluator.evaluate(trajectory)
        assert whole.count == len(trajectory)
        assert all(whole.energy[steps[s.step]] == from_file.energy[k] for k, s in enumerate(states))
        names = list(from_states.jacobian)
        results[kind] = (
            float(np.abs(from_file.energy - from_states.energy).max()),
            max(relative(from_file.jacobian[name], from_states.jacobian[name]) for name in names),
            np.array_equal(from_file.energy, from_states.energy) and
            np.array_equal(from_file.virial, from_states.virial) and
            all(np.array_equal(from_file.jacobian[name], from_states.jacobian[name])
                for name in names))
        trajectory.close()
    energy, rows, exact = results["f64"]
    print(f"{mode}: 5 frames of an H5MDReporter in f64, read one at a time, against the states "
          f"of the same steps: the energy {energy:.1e} kJ/mol, the derivative {rows:.1e} "
          f"relative; equal to the bit: {exact}")
    assert exact
    energy, rows, exact = results["f32"]
    print(f"{mode}: the same from a file in f32: the energy within {energy:.1e} kJ/mol, the "
          f"derivative within {rows:.1e} relative")
    assert not exact and energy < 2e-2 and rows < 1e-4, (energy, rows)
    print("h5md passed")


# 60 A + 60 B, the A-B pair as a pair term with sigma' tunable.
NBFIX = "4*eps*((sig/r)^12 - (sig/r)^6) - 4*eps0*((sig0/r)^12 - (sig0/r)^6)"
SIGMA, TEMPERATURE = 0.37, 100.0


def mixture_program(sigma=SIGMA):
    loaded = mdir.load_gromacs(mixture + "/plain.top", mixture + "/system.gro")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 1.2, 1.3
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    term = mdir.PairTerm()
    term.name, term.groups, term.expression = "nbfix", [":A", ":B"], NBFIX
    term.constants = [("sig", sigma), ("eps", 0.5 * 4.184), ("sig0", 0.34), ("eps0", 0.24 * 4.184),
                      ("unused", 1.0)]
    term.observe = ["sig"]
    slab = mdir.ExternalTerm()
    slab.name, slab.expression, slab.selection = "slab", "c*step(1.6 - z)", ":A"
    slab.constants = [("c", 0.0)]
    slab.observe = ["c"]
    system.pair_terms, system.external_terms = [term], [slab]
    # `unused` is a constant that the expression does not read: its
    # derivative is zero by proof (D230).
    system.tunables = [mdir.Tunable("sig", "sig", term="nbfix"),
                       mdir.Tunable("unused", "unused", term="nbfix")]
    system.tunable_gradient = True
    state = state.draw_velocities(system, TEMPERATURE, 271828)
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.002, TEMPERATURE, 271828
    ensemble.kind = mdir.EnsembleKind.NVT
    ensemble.com_period = ensemble.coupling_period = 10
    return mdir.compile(system, state, integrator, ensemble, execution_(), mdir.Schedule())


def mixture_frames(program, count, stride=20, warm=500):
    sim = simulation(program)
    sim.run(warm)
    frames = []
    for _ in range(count):
        sim.run(stride)
        state = sim.state()
        frames.append((state.positions, state.cell))
    return frames


def reweighted(evaluator, frames, reference, sigma, kT):
    """The weights of the frames at sigma', and the average of the
    observable that the program observes: the number of A below z = 1.6 nm
    (the derivative in the strength 0 of a step), a function of the
    positions alone."""
    evaluator.tunables["sig"] = np.array([sigma])
    out = evaluator.evaluate(frames)
    log = -(out.energy - reference) / kT
    w = np.exp(log - log.max())
    w /= w.sum()
    observable = out.observables["slab.d_c"]
    return out, w, float(w @ observable), observable


def run_mixture():
    kT = mdir.KB * TEMPERATURE
    program = mixture_program()
    frames = mixture_frames(program, 150)
    evaluator = mdir.FrameEvaluator(program)
    reference = evaluator.evaluate(frames).energy
    out, w, average, observable = reweighted(evaluator, frames, reference, SIGMA, kT)
    count = len(frames)
    n_eff = float(np.exp(-(w * np.log(w)).sum()))
    uniform = np.array_equal(out.energy, reference) and np.all(w == 1.0 / count)
    print(f"{mode}: at the values that sampled, the energies equal the reference to the bit "
          f"and the weights are uniform: {uniform}; N_eff = {n_eff:.6f} of {count}")
    assert uniform and abs(n_eff - count) < 1e-9 and observable.std() > 0.5
    assert out.depends["slab.d_c"] == frozenset() and out.depends["nbfix.d_sig"] == {"sig"}
    assert out.zero == {"unused"} and np.all(out.jacobian["unused"] == 0.0)
    assert out.depends["energy"] == {"sig"} and out.vjp(np.ones(count)).zero == {"unused"}
    # The observed derivative of the term is that of the energy (D232).
    column = relative(out.observables["nbfix.d_sig"], out.jacobian["sig"][:, 0])
    assert column < 1e-12, column

    worst = 0.0
    for sigma in (SIGMA, SIGMA + 0.002):
        out, w, average, observable = reweighted(evaluator, frames, reference, sigma, kT)
        g = out.jacobian["sig"][:, 0]
        derivative = -(w @ (observable * g) - (w @ observable) * (w @ g)) / kT

        def central(h):
            return (reweighted(evaluator, frames, reference, sigma + h, kT)[2] -
                    reweighted(evaluator, frames, reference, sigma - h, kT)[2]) / (2.0 * h)
        h = 2e-4 if double else 2e-3
        fd = (4.0 * central(0.5 * h) - central(h)) / 3.0
        n_eff = float(np.exp(-(w * np.log(w)).sum()))
        difference = abs(derivative - fd) / abs(fd)
        worst = max(worst, difference)
        print(f"{mode}: sigma' = {sigma:.3f} nm, N_eff = {n_eff:.1f}: the derivative of the "
              f"reweighted average {derivative:+.4f} /nm, central differences {fd:+.4f}, "
              f"{difference:.1e} relative")
    tolerance = 1e-5 if double else 3e-2
    assert worst < tolerance, worst
    print(f"mixture passed (tolerance {tolerance:.0e})")


def run_torch():
    import torch
    import mdir.torch

    device = "cuda" if target == "GPU" else "cpu"
    kT = mdir.KB * TEMPERATURE
    program = mixture_program()
    frames = mixture_frames(program, 150)
    evaluator = mdir.FrameEvaluator(program)
    with torch.no_grad():
        hat = {"sig": torch.tensor([SIGMA], dtype=torch.float64, device=device)}
        first = mdir.torch.evaluate(evaluator, hat, frames)
        reference = first.energy
    assert reference.device.type == device and reference.dtype == torch.float64
    assert reference.grad_fn is None and first.count == len(frames)

    def average(sig, frames=frames, reference=reference, evaluator=evaluator):
        out = mdir.torch.evaluate(evaluator, {"sig": sig}, frames)
        w = torch.softmax(-(out.energy - reference) / kT, dim=0)
        return (w * out.observables["slab.d_c"]).sum(), out, w

    # The gradient of the reweighted average from autograd against the
    # covariance formed from the evaluator's own arrays.
    worst = 0.0
    for sigma in (SIGMA, SIGMA + 0.002):
        sig = torch.tensor([sigma], dtype=torch.float64, device=device, requires_grad=True)
        value, out, w = average(sig)
        value.backward()
        plain = mdir.FrameEvaluator(program)
        plain.tunables["sig"] = np.array([sigma])
        arrays = plain.evaluate(frames)
        wn = w.detach().cpu().numpy()
        o, g = arrays.observables["slab.d_c"], arrays.jacobian["sig"][:, 0]
        expected = -(wn @ (o * g) - (wn @ o) * (wn @ g)) / kT
        worst = max(worst, abs(float(sig.grad[0]) - expected) / abs(expected))
        assert out.energy.grad_fn is not None and out.volume.grad_fn is None
        assert out.observables["slab.d_c"].grad_fn is None
        assert out.observables["nbfix.d_sig"].grad_fn is not None
        assert out.virial.grad_fn is not None and out.energy.device.type == device
    print(f"{mode}: the gradient of the reweighted average from autograd against the covariance "
          f"of the arrays, {worst:.1e} relative")
    assert worst < 1e-11, worst

    # The Jacobian not kept: the backward is a second pass.
    second = mdir.FrameEvaluator(program, jacobian_bytes=0)
    sig = torch.tensor([SIGMA + 0.002], dtype=torch.float64, device=device, requires_grad=True)
    kept, _, _ = average(sig)
    kept.backward()
    again = torch.tensor([SIGMA + 0.002], dtype=torch.float64, device=device, requires_grad=True)
    value, _, _ = average(again, evaluator=second)
    value.backward()
    passed = abs(float(again.grad[0]) - float(sig.grad[0])) / abs(float(sig.grad[0]))
    print(f"{mode}: with the Jacobian not kept, the backward by a second pass, {passed:.1e} relative")
    assert passed < 1e-12, passed
    stale = torch.tensor([SIGMA], dtype=torch.float64, device=device, requires_grad=True)
    value, _, _ = average(stale, evaluator=second)
    mdir.torch.evaluate(second, {"sig": stale.detach() + 0.001}, frames)
    expect(mdir.SimulationError, value.backward, "other frames since")

    if double:
        few = frames[:12]
        with torch.no_grad():
            few_reference = mdir.torch.evaluate(evaluator, hat, few).energy

        def function(sig):
            return average(sig, few, few_reference)[0]
        sig = torch.tensor([SIGMA + 0.001], dtype=torch.float64, device=device, requires_grad=True)
        checked = torch.autograd.gradcheck(function, (sig,), eps=1e-5, atol=1e-6, rtol=1e-4)
        print(f"{mode}: torch.autograd.gradcheck of the reweighted average: {checked}")
        assert checked

    # A fit of sigma' to a synthetic target: the reweighted average of the
    # same frames at sigma* = sigma-hat + 0.0005 nm (the average of these
    # 150 frames is monotonic in sigma' up to about 0.371 nm), by Newton
    # steps on the difference with the derivative from autograd.
    shift = 0.0005
    with torch.no_grad():
        star = torch.tensor([SIGMA + shift], dtype=torch.float64, device=device)
        goal = float(average(star)[0])
    sig = torch.tensor([SIGMA], dtype=torch.float64, device=device, requires_grad=True)
    start = None
    for step in range(12):
        value, _, w = average(sig)
        loss = (value - goal) ** 2
        start = float(loss.detach()) if start is None else start
        gradient, = torch.autograd.grad(value, sig)
        with torch.no_grad():
            sig -= (value - goal) / gradient
    found = float(sig.detach()[0])
    n_eff = float(torch.exp(-(w * torch.log(w)).sum()))
    print(f"{mode}: a fit to the average at sigma* = {SIGMA + shift:.4f} nm from {SIGMA:.4f}: "
          f"sigma' = {found:.8f} nm, {abs(found - SIGMA - shift):.1e} nm from sigma*; the loss "
          f"from {start:.1e} to {float(loss.detach()):.1e}; N_eff = {n_eff:.1f}")
    assert abs(found - SIGMA - shift) < (1e-9 if double else 1e-5), found

    # What the adapter refuses.
    sig = torch.tensor([SIGMA], dtype=torch.float64, device=device, requires_grad=True)
    out = mdir.torch.evaluate(evaluator, {"sig": sig}, frames)
    text = expect(mdir.UnsupportedError, lambda: out.virial.sum().backward(), "'virial'")
    assert "not implemented" in text and "sig" in text, text
    out = mdir.torch.evaluate(evaluator, {"sig": sig}, frames)
    expect(mdir.UnsupportedError, lambda: out.observables["nbfix.d_sig"].sum().backward(),
           "'nbfix.d_sig'")
    out = mdir.torch.evaluate(evaluator, {"sig": sig}, frames)
    (out.energy.sum() + out.virial.detach().sum() + out.observables["slab.d_c"].sum()).backward()
    assert sig.grad is not None
    out = mdir.torch.evaluate(evaluator, {"sig": sig}, frames)
    once, = torch.autograd.grad(out.energy.sum(), sig, create_graph=True)
    expect(RuntimeError, lambda: once.sum().backward())
    moving = torch.tensor(np.stack([f[0] for f in frames[:3]]), requires_grad=True)
    expect(mdir.UnsupportedError, lambda: mdir.torch.evaluate(evaluator, {"sig": sig}, moving),
           "positions require a gradient")
    expect(mdir.InputError, lambda: mdir.torch.evaluate(evaluator, {"sig": sig.float()}, frames),
           "float64")
    expect(mdir.InputError, lambda: mdir.torch.evaluate(evaluator, {"eps": sig}, frames),
           "no tunable named 'eps'")
    assert all(r.out is None and r.frames is None for r in mdir.torch._RECORDS.values())
    print(f"{mode}: refused: a backward through the virial and through a column that the tunable "
          "enters, a second backward, positions that require a gradient")

    # Inside torch.compile.
    def compiled_loss(sig):
        return average(sig)[0]
    sig = torch.tensor([SIGMA + 0.001], dtype=torch.float64, device=device, requires_grad=True)
    eager = compiled_loss(sig)
    eager_gradient, = torch.autograd.grad(eager, sig)
    compiled = torch.compile(compiled_loss, backend="aot_eager")
    traced = compiled(sig)
    traced_gradient, = torch.autograd.grad(traced, sig)
    same = traced.item() == eager.item() and traced_gradient[0].item() == eager_gradient[0].item()
    print(f"{mode}: under torch.compile (aot_eager), the same average and gradient: {same}")
    assert same
    print("torch passed")


{"frames": run_frames, "npt": run_npt, "refusals": run_refusals, "mixture": run_mixture,
 "torch": run_torch, "triclinic": run_triclinic, "h5md": run_h5md}[scenario]()
