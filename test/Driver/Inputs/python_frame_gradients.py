"""The gradients of the energy of a frame in the positions and in the
strain (D[frame-gradients], docs/python-frames.md).

Usage: python_frame_gradients.py ROOT SCENARIO TARGET PRECISION

ROOT is Inputs/dipeptide. G is `gradients["positions"]`, dU/dx; S is
`gradients["strain"]`, dU/d(epsilon) under x -> (1 + epsilon) x of the
positions and the cell vectors.

Scenarios:
  finite     G and S against central differences of the evaluator's own
             energy (in double precision; in mixed precision against the
             same differences of a program in double precision): the
             dipeptide in water with PME, a pair term, and a wall, in an
             orthorhombic cell (G of the solute and of some waters, the
             diagonal of S); flexible water in a sheared rhombic
             dodecahedron (the nine components of S, those below the
             diagonal with the deformed cell rotated back to the
             lower-triangular form); the trace of S against the virial; G
             against the forces of the sampler's state, to the bit
  special    with the correction for the dispersion under a plain cutoff
             and a net charge (what the host adds at the volume of the
             frame); with a wall, which no rotation leaves unchanged (the
             components of S on and above the diagonal); with rigid water
             (the plain gradient, and what the virial has besides); with
             virtual sites (OPC: the rows of the sites are zero)
  interface  what is asked is what is given, in the type of the forces;
             calls that ask and calls that do not share an activation and
             give the same numbers to the bit; the refusals
"""
import sys

import numpy as np
import mdir

root, scenario, target, precision = sys.argv[1:5]
double = precision == "Double"
mode = f"{target} {precision}"
# The steps of the central differences. The energy is only continuous where
# a pair crosses the cutoff (the direct sum of PME is shifted there, and its
# force jumps by about 1e-3 kJ/mol/nm a pair), so a difference over a long
# step is rough at the particles that have such a pair: measured on the
# dipeptide in double precision, the largest error of 102 components is
# 7e-3 kJ/mol/nm at 1e-3 nm, 3e-3 at 2e-5, 1e-4 at 4e-6, and 3e-4 at 1e-6,
# where the rounding of the energy takes over.
POSITION_STEP = 4e-6
STRAIN_STEP = 1e-6


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text})")


def execution_(precision_=None, deterministic=True):
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision_ or precision)
    execution.deterministic = deterministic
    execution.reorder = False
    return execution


def compile_(system, state, precision_=None, deterministic=True):
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    return mdir.compile(system, state, integrator, mdir.Ensemble(),
                        execution_(precision_, deterministic), mdir.Schedule())


def one_pair(system, a, b):
    top = system.topology
    names, pairs = list(top.type_names), np.array(top.type_pairs)
    a, b = sorted((names.index(a), names.index(b)))
    return np.where((pairs[:, 0] == a) & (pairs[:, 1] == b), 0, -1)


def dipeptide(kind="plain"):
    """The dipeptide in 382 waters with PME. `plain`: a switched
    Lennard-Jones and a soft pair term; `wall`: a wall on the waters besides;
    `tail`: a plain cutoff with the correction for the dispersion, and a
    net charge; `rigid`: rigid water."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    if kind == "tail":
        system.truncation = mdir.Truncation.None_
        system.dispersion = mdir.DispersionCorrection.EnergyPressure
    else:
        system.truncation, system.switch_distance = mdir.Truncation.Switch, 0.7
        system.dispersion = mdir.DispersionCorrection.None_
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    system.pair_terms = [soft]
    if kind == "wall":
        wall = mdir.ExternalTerm()
        wall.name, wall.expression, wall.selection = "wall", "0.5*k*max(0, z - z0)^2", ":WAT"
        wall.constants = [("k", 100.0), ("z0", 1.5)]
        system.external_terms = [wall]
    if kind == "rigid":
        system.rigid_water = True
    system.tunables = [mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW"))]
    if kind == "tail":
        system.tunables += [mdir.Tunable("q", "charge", map=net_charge_map(system),
                                         values=np.array([0.0]))]
    system.tunable_gradient = True
    return system, state


def net_charge_map(system):
    """One entry for the 22 atoms of the solute, whose charges a test moves
    to give the system a net charge."""
    count = len(system.topology.atom_names)
    return np.where(np.arange(count) < 22, 0, -1)


def water(kind="plain"):
    """403 flexible waters with PME in a rhombic dodecahedron."""
    triclinic = root + "/../triclinic"
    loaded = mdir.load_gromacs(triclinic + "/water.top", triclinic + "/dodecahedron.gro",
                               defines=["FLEXIBLE"])
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28] * 3
    if kind == "wall":
        wall = mdir.ExternalTerm()
        wall.name, wall.expression = "wall", "0.5*k*max(0, z - z0)^2 + 0.3*k*x*z"
        wall.particles = np.arange(len(state.positions))
        wall.constants = [("k", 100.0), ("z0", 1.0)]
        system.external_terms = [wall]
    system.tunables = [mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW"))]
    system.tunable_gradient = True
    return system, state


def vectors(diagonal, tilt):
    """The cell vectors in rows, lower triangular."""
    return np.array([[diagonal[0], 0.0, 0.0], [tilt[0], diagonal[1], 0.0],
                     [tilt[1], tilt[2], diagonal[2]]])


def deformed(x, h, epsilon):
    """The positions and the cell vectors under x -> (1 + epsilon) x, with
    the cell rotated back to the lower-triangular form if epsilon has a part
    below the diagonal: h' = h (1 + epsilon)^T Q with Q orthogonal, by a QR
    factorization of the cell vectors in columns. A potential of pairs,
    tuples, and the reciprocal sum does not change under Q."""
    f = np.eye(3) + epsilon
    x1, h1 = x @ f.T, h @ f.T
    if np.any(np.abs(np.triu(h1, 1)) > 0.0):
        q, r = np.linalg.qr(h1.T)
        q = q * np.sign(np.diag(r))
        x1, h1 = x1 @ q, np.tril(h1 @ q)
    return x1, h1


def strain_differences(evaluator, x, h, components, step, triclinic):
    """Central differences of the energy in the components of the strain,
    extrapolated from the steps `step` and `step` / 2 (Richardson). The
    energy is only continuous where a pair crosses the cutoff, so the steps
    are short."""
    frames = []
    for a, b in components:
        for s in (step, -step, 0.5 * step, -0.5 * step):
            epsilon = np.zeros((3, 3))
            epsilon[a, b] = s
            frames.append(deformed(x, h, epsilon))
    positions = np.stack([f[0] for f in frames])
    cells = np.stack([np.diag(f[1]) for f in frames])
    tilts = np.stack([[f[1][1, 0], f[1][2, 0], f[1][2, 1]] for f in frames]) if triclinic else None
    u = evaluator.evaluate(positions, cells, tilts, gradient=False).energy.reshape(-1, 4)
    return (4.0 * (u[:, 2] - u[:, 3]) / step - (u[:, 0] - u[:, 1]) / (2.0 * step)) / 3.0


def position_differences(evaluator, x, cell, tilt, entries, step):
    """Central differences of the energy in the coordinates `entries`, pairs
    of a particle and an axis, extrapolated as above."""
    frames = np.repeat(x[None], 4 * len(entries), axis=0)
    for k, (i, c) in enumerate(entries):
        for j, s in enumerate((step, -step, 0.5 * step, -0.5 * step)):
            frames[4 * k + j, i, c] += s
    u = evaluator.evaluate(frames, cell, tilt, gradient=False).energy.reshape(-1, 4)
    return (4.0 * (u[:, 2] - u[:, 3]) / step - (u[:, 0] - u[:, 1]) / (2.0 * step)) / 3.0


def sheared_water(state):
    """The state of the dodecahedron sheared by 1%, positions and cell
    together, so that its cell is inside the reduced form and stays so under
    the strains of a difference (the dodecahedron itself has c_x = a_x / 2
    and c_y = b_y / 2, the bound)."""
    x = np.array(state.positions)
    h = vectors(state.cell.diagonal, state.cell.tilt)
    epsilon = np.zeros((3, 3))
    epsilon[0, 2] = epsilon[1, 2] = epsilon[0, 1] = -0.01
    return deformed(x, h, epsilon)


def checked(label, found, reference, scale, tolerance):
    error = float(np.abs(np.asarray(found) - np.asarray(reference)).max())
    print(f"{mode}: {label}: {error:.1e} kJ/mol of {scale:.5g}, {error / scale:.1e} relative "
          f"(tolerance {tolerance:.0e})")
    assert error / scale < tolerance, (label, error, scale)
    return error / scale


def reference_evaluator(program, system, state):
    """The evaluator whose energy is differenced: the program itself in
    double precision, and in mixed precision the same model in double
    precision on the same target."""
    return mdir.FrameEvaluator(program if double else compile_(system, state, "Double"))


def run_finite():
    # The dipeptide, orthorhombic.
    system, state = dipeptide()
    program = compile_(system, state)
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    sim.run(40)
    sampled = sim.state()
    x, cell = np.array(sampled.positions), np.array(sampled.cell.diagonal)
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(x[None], cell[None], gradients=("positions", "strain"))
    g, s = out.gradients["positions"][0], out.gradients["strain"][0]
    assert out.gradients["positions"].shape == (1, len(x), 3) and s.shape == (3, 3)
    assert out.gradients["strain"].dtype == np.float64
    forces = np.asarray(sampled.forces)
    apart = float(np.abs(g + forces).max() / np.abs(forces).max())
    # The sampler takes its sums in a step, the evaluator at a start: the
    # same forces to the rounding of sums in another order.
    tolerance = 1e-13 if double else 1e-5
    print(f"{mode}: the gradient in the positions of a frame of the dipeptide, {g.dtype}, against "
          f"the forces of the sampler's state at that step: {apart:.1e} relative (tolerance "
          f"{tolerance:.0e})")
    assert apart < tolerance, apart

    reference = reference_evaluator(program, system, state)
    rng = np.random.default_rng(5)
    entries = [(i, c) for i in range(22) for c in range(3)]
    entries += [(int(i), c) for i in rng.choice(np.arange(22, len(x)), 12, replace=False)
                for c in range(3)]
    found = np.array([g[i, c] for i, c in entries], dtype=np.float64)
    fd = position_differences(reference, x, cell, None, entries, POSITION_STEP)
    scale = float(np.abs(g).max())
    checked(f"G, {len(entries)} components of the solute and of 12 atoms of the water, against "
            "central differences of the energy", found, fd, scale, 5e-7 if double else 1e-4)
    h = np.diag(cell)
    diagonal = [(0, 0), (1, 1), (2, 2)]
    fd = strain_differences(reference, x, h, diagonal, STRAIN_STEP, False)
    scale = float(np.abs(s).max())
    checked("S, its diagonal in an orthorhombic cell, against central differences of the energy",
            np.diag(s), fd, scale, 5e-7 if double else 1e-4)
    trace = abs(float(np.trace(s)) + float(out.virial[0]))
    print(f"{mode}: the virial plus the trace of S: {trace:.1e} kJ/mol of {out.virial[0]:.3f}")
    assert trace < 1e-9 * abs(out.virial[0])
    asymmetry = float(np.abs(s - s.T).max())
    print(f"{mode}: S less its transpose: {asymmetry:.1e} kJ/mol of {scale:.5g}")
    assert asymmetry < (1e-9 if double else 1e-4) * scale

    # Flexible water in a sheared rhombic dodecahedron: the nine components.
    system, state = water()
    program = compile_(system, state)
    x, h = sheared_water(state)
    cell, tilt = np.diag(h), np.array([h[1, 0], h[2, 0], h[2, 1]])
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(x[None], cell[None], tilt[None], gradients=("positions", "strain"))
    g, s = out.gradients["positions"][0], out.gradients["strain"][0]
    reference = reference_evaluator(program, system, state)
    components = [(a, b) for a in range(3) for b in range(3)]
    fd = strain_differences(reference, x, h, components, STRAIN_STEP, True).reshape(3, 3)
    scale = float(np.abs(s).max())
    tolerance = 5e-7 if double else 1e-4
    checked("S of water in a triclinic cell, on and above the diagonal, against central "
            "differences of the energy", s[np.triu_indices(3)], fd[np.triu_indices(3)], scale,
            tolerance)
    checked("S of the same, below the diagonal, the deformed cell rotated back",
            s[np.tril_indices(3, -1)], fd[np.tril_indices(3, -1)], scale, tolerance)
    entries = [(int(i), c) for i in rng.choice(len(x), 12, replace=False) for c in range(3)]
    fd = position_differences(reference, x, cell, tilt, entries, POSITION_STEP)
    found = np.array([g[i, c] for i, c in entries], dtype=np.float64)
    checked("G of the same, 36 components, against central differences of the energy", found,
            fd, float(np.abs(g).max()), 5e-7 if double else 1e-4)
    trace = abs(float(np.trace(s)) + float(out.virial[0]))
    print(f"{mode}: the virial plus the trace of S, triclinic: {trace:.1e} kJ/mol of "
          f"{out.virial[0]:.3f}")
    assert trace < 1e-9 * abs(out.virial[0])
    print("finite passed")


def run_special():
    rng = np.random.default_rng(7)
    g_tolerance, s_tolerance = (5e-7, 5e-7) if double else (1e-4, 1e-4)

    # What the host adds at the volume of the frame: the tail of the
    # dispersion with the estimate of the shift under a plain cutoff, and
    # the background of a net charge of 0.44 e.
    system, state = dipeptide("tail")
    program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    reference = reference_evaluator(program, system, state)
    for each in {id(evaluator): evaluator, id(reference): reference}.values():
        each.tunables.update({"q": np.array([0.02])})
    x, cell = np.array(state.positions), np.array(state.cell.diagonal)
    out = evaluator.evaluate(x[None], cell[None], gradients=("strain",))
    s = out.gradients["strain"][0]
    fd = strain_differences(reference, x, np.diag(cell), [(0, 0), (1, 1), (2, 2)], STRAIN_STEP,
                            False)
    checked("with the correction for the dispersion under a plain cutoff and a net charge of "
            "0.44 e, the diagonal of S against central differences of the energy", np.diag(s), fd,
            float(np.abs(s).max()), s_tolerance)
    trace = abs(float(np.trace(s)) + float(out.virial[0]))
    print(f"{mode}: there, the virial plus the trace of S: {trace:.1e} kJ/mol of "
          f"{out.virial[0]:.3f}")
    assert trace < 1e-9 * abs(out.virial[0])

    # A term of the absolute positions: no rotation leaves it unchanged, S
    # is not symmetric, and the differences are those of the strains that
    # keep the cell lower triangular, on and above the diagonal.
    system, state = water("wall")
    program = compile_(system, state)
    x, h = sheared_water(state)
    cell, tilt = np.diag(h), np.array([h[1, 0], h[2, 0], h[2, 1]])
    evaluator = mdir.FrameEvaluator(program)
    reference = reference_evaluator(program, system, state)
    out = evaluator.evaluate(x[None], cell[None], tilt[None], gradients=("positions", "strain"))
    g, s = out.gradients["positions"][0], out.gradients["strain"][0]
    upper = [(a, b) for a in range(3) for b in range(a, 3)]
    fd = strain_differences(reference, x, h, upper, STRAIN_STEP, True)
    scale = float(np.abs(s).max())
    checked("with a term of the absolute positions, S on and above the diagonal against central "
            "differences of the energy", [s[a, b] for a, b in upper], fd, scale, s_tolerance)
    # Below the diagonal no cell of the program takes the strain; there S
    # is the sum over the particles of the term, G_i (x) x_i, added to a
    # symmetric rest.
    asymmetry = float(np.abs(s - s.T).max())
    print(f"{mode}: there, S less its transpose: {asymmetry:.4g} kJ/mol (not symmetric)")
    assert asymmetry > 1e-3 * scale
    entries = [(int(i), c) for i in rng.choice(len(x), 8, replace=False) for c in range(3)]
    fd = position_differences(reference, x, cell, tilt, entries, POSITION_STEP)
    checked("there, G, 24 components", [g[i, c] for i, c in entries], fd,
            float(np.abs(g).max()), g_tolerance)

    # Rigid water: the positions are taken as given, G is the plain
    # gradient in every position, and S has no term of the constraint
    # forces, which the virial of the pressure has.
    system, state = dipeptide("rigid")
    program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    reference = reference_evaluator(program, system, state)
    x, cell = np.array(state.positions), np.array(state.cell.diagonal)
    out = evaluator.evaluate(x[None], cell[None], gradients=("positions", "strain"))
    g, s = out.gradients["positions"][0], out.gradients["strain"][0]
    entries = [(int(i), c) for i in rng.choice(len(x), 12, replace=False) for c in range(3)]
    fd = position_differences(reference, x, cell, None, entries, POSITION_STEP)
    checked("with rigid water, G, 36 components, against central differences of the energy in "
            "every position", [g[i, c] for i, c in entries], fd, float(np.abs(g).max()),
            g_tolerance)
    fd = strain_differences(reference, x, np.diag(cell), [(0, 0), (1, 1), (2, 2)], STRAIN_STEP,
                            False)
    checked("with rigid water, the diagonal of S", np.diag(s), fd, float(np.abs(s).max()),
            s_tolerance)
    constraint = float(out.virial[0]) + float(np.trace(s))
    print(f"{mode}: with rigid water, the virial plus the trace of S, the term of the constraint "
          f"forces: {constraint:.3f} kJ/mol of {out.virial[0]:.3f}")
    assert abs(constraint) > 1.0

    # Virtual sites (OPC): a site is placed from its atoms whatever the
    # frame says, so the energy does not depend on its row.
    opc = root + "/../opc"
    loaded = mdir.load_amber(opc + "/opc.prmtop", opc + "/opc.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    reference = reference_evaluator(program, system, state)
    x, cell = np.array(state.positions), np.array(state.cell.diagonal)
    masses = np.array(system.topology.masses)
    sites = np.flatnonzero(masses == 0.0)
    atoms = np.flatnonzero(masses > 0.0)
    assert len(sites) > 0
    moved = x.copy()
    moved[sites] += 0.05
    out = evaluator.evaluate(np.stack([x, moved]), cell, gradients=("positions", "strain"))
    g, s = out.gradients["positions"], out.gradients["strain"]
    # To the rounding of sums: the neighbor structures bin the rows given.
    assert abs(out.energy[0] - out.energy[1]) < 1e-9 * abs(out.energy[0])
    assert np.abs(g[0] - g[1]).max() < (1e-12 if double else 1e-5) * np.abs(g[0]).max()
    assert not np.any(g[0][sites]) and not np.any(g[1][sites]) and np.any(g[0][atoms])
    entries = [(int(i), c) for i in rng.choice(atoms, 12, replace=False) for c in range(3)]
    fd = position_differences(reference, x, cell, None, entries, POSITION_STEP)
    checked(f"with {len(sites)} virtual sites, whose rows of G are zero whatever their rows of the "
            "frame hold, G of 36 components of their atoms against central differences",
            [g[0][i, c] for i, c in entries], fd, float(np.abs(g[0]).max()), g_tolerance)
    fd = strain_differences(reference, x, np.diag(cell), [(0, 0), (1, 1), (2, 2)], STRAIN_STEP,
                            False)
    checked("with virtual sites, the diagonal of S", np.diag(s[0]), fd, float(np.abs(s[0]).max()),
            s_tolerance)
    print("special passed")


def run_interface():
    system, state = dipeptide()
    program = compile_(system, state)
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    frames = []
    for _ in range(4):
        sim.run(20)
        frames.append(sim.state())
    x = np.stack([f.positions for f in frames])
    count = x.shape[1]
    evaluator = mdir.FrameEvaluator(program)
    plain = evaluator.evaluate(frames)
    assert len(plain.gradients) == 0 and dict(plain.gradients) == {}
    both = evaluator.evaluate(frames, gradients=("positions", "strain"))
    g, s = both.gradients["positions"], both.gradients["strain"]
    assert list(both.gradients) == ["positions", "strain"]
    assert g.shape == (4, count, 3) and s.shape == (4, 3, 3) and s.dtype == np.float64
    assert not g.flags.writeable and not s.flags.writeable
    assert not hasattr(both.gradients, "__setitem__")
    # In the deterministic mode the forces of a program in mixed precision
    # are f64 (#102); without it they are f32, and so is the gradient.
    assert g.dtype == np.float64, g.dtype
    loose = mdir.FrameEvaluator(compile_(system, state, deterministic=False))
    narrow = loose.evaluate(frames, gradients=("positions",)).gradients["positions"]
    expected = np.float64 if double else np.float32
    assert narrow.dtype == expected, narrow.dtype
    apart = float(np.abs(narrow - g).max() / np.abs(g).max())
    print(f"{mode}: the gradient in the positions is {g.dtype} in the deterministic mode and "
          f"{narrow.dtype} without it, in the type of the forces of the program; the two differ "
          f"by {apart:.1e} relative")
    assert apart < (1e-12 if double else 1e-5)

    # What is asked travels with each frame: calls that ask for nothing,
    # for one, and for both share an activation and give the same numbers.
    def same(out, of=both):
        return (np.array_equal(out.energy, of.energy) and np.array_equal(out.virial, of.virial)
                and all(np.array_equal(out.jacobian[n], of.jacobian[n]) for n in of.jacobian)
                and all(np.array_equal(a, of.gradients[n]) for n, a in out.gradients.items()))

    depth = evaluator._simulation._stack_in_use
    only_g = evaluator.evaluate(x, gradients=("positions",))
    only_s = evaluator.evaluate(frames, gradients="strain")
    assert list(only_g.gradients) == ["positions"] and list(only_s.gradients) == ["strain"]
    again = evaluator.evaluate(x, gradients=("strain", "positions", "strain"))
    assert list(again.gradients) == ["strain", "positions"]
    one = [evaluator.evaluate(x[n], gradients=("positions", "strain")) for n in (3, 0)]
    exact = (same(plain) and same(only_g) and same(only_s) and same(again) and
             same(evaluator.evaluate(frames)) and
             all(np.array_equal(o.gradients[name][0], both.gradients[name][n])
                 for o, n in zip(one, (3, 0)) for name in ("positions", "strain")))
    assert evaluator._simulation._stack_in_use == depth
    print(f"{mode}: calls that ask for nothing, for one gradient, and for both, as states, as an "
          f"array, and one frame alone, in one activation, equal to the bit: {exact}")
    assert exact

    # Refused.
    expect(mdir.InputError, lambda: evaluator.evaluate(x, gradients=("forces",)),
           "gradients names 'positions' and 'strain'")
    expect(mdir.InputError, lambda: evaluator.evaluate(x, gradient=("positions",)),
           "gradient is True or False")
    dependent = mdir.FrameEvaluator(program, terms="dependent")
    assert "gradients" in dependent.unavailable and "gradients" not in evaluator.unavailable
    expect(mdir.UnsupportedError, lambda: dependent.evaluate(x, gradients=("strain",)),
           "terms='dependent'")
    assert len(dependent.evaluate(x).gradients) == 0
    assert same(evaluator.evaluate(frames, gradients=("positions", "strain")))
    print(f"{mode}: refused: a name that is not a gradient, a gradient that is not a bool, the "
          "gradients of an evaluator of the dependent terms; the evaluator stays usable")
    print("interface passed")


if scenario == "finite":
    run_finite()
elif scenario == "special":
    run_special()
elif scenario == "interface":
    run_interface()
else:
    raise SystemExit(f"unknown scenario {scenario}")
