"""The gradients of the energy of a frame in the positions and in the
strain (D252, docs/python-frames.md).

Usage: python_frame_gradients.py ROOT SCENARIO TARGET PRECISION

ROOT is Inputs/dipeptide. G is `gradients["positions"]`, dU/dx; S is
`gradients["strain"]`, dU/d(epsilon) under x -> (1 + epsilon) x of the
positions and the cell vectors.

Scenarios:
  finite     G and S against central differences of the evaluator's own
             energy (in double precision; in mixed precision against the
             same differences of a program in double precision): the
             dipeptide in water with PME and a pair term, in an
             orthorhombic cell (G of the solute and of some waters, the
             diagonal of S); flexible water in a sheared rhombic
             dodecahedron (the nine components of S, those below the
             diagonal with the deformed cell rotated back to the
             lower-triangular form); the trace of S against the virial; G
             against the forces of the sampler's state
  prolate    `finite` with the prolate kernel of PME (D266):
             its reciprocal virial, whose off-diagonal components the
             other tests of the kernel do not reach, against central
             differences of the energy under each component of the strain
  special    with the correction for the dispersion under a plain cutoff
             and a net charge (what the host adds at the volume of the
             frame); with a wall, which no rotation leaves unchanged (the
             components of S on and above the diagonal); with rigid water
             (the plain gradient, and what the virial has besides); with
             virtual sites (OPC: the rows of the sites are zero)
  interface  what is asked is what is given, in the type of the forces;
             calls that ask and calls that do not share an activation and
             give the same numbers to the bit; the refusals
  torchpme   the Coulomb part of G and of the nine components of S, the
             difference of the gradients at the charges of the model and at
             charges of zero, against torch-pme (Loche et al., J. Chem.
             Phys. 162, 142501 (2025)) at the same positions, cell, charges,
             splitting parameter, and grid, by the autograd of PyTorch in
             the positions and in a strain: its sum of the mesh with the
             pair sums written in PyTorch, as python_tunable_gradient.py
             has them for the derivative in the charges; the dipeptide in
             an orthorhombic cell, and with `torchpme-water` water in a
             rhombic dodecahedron (a process of its own), at
             two grids. Needs an interpreter with torch and torchpme
  openmm     G and the nine components of S of the whole potential against
             OpenMM: its forces, and central differences of its energy
             under the strain (it reports no virial), with the same
             splitting parameter and grid, at two grids. Needs openmm

`torchpme` and `openmm` take after PRECISION, optionally, PRMTOP INPCRD
GRID of another system in an orthorhombic cell (JAC of the Amber suite, by
hand: it is not in the repository), in place of the dipeptide and its grid
of 64.
"""
import math
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
    if scenario == "prolate":
        system.pme_kernel = mdir.PMEKernel.Prolate
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
    if kind == "restrained":
        system.restraints = [mdir.Restraint("!:WAT & !@H*", 500.0)]
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
    if scenario == "prolate":
        system.pme_kernel = mdir.PMEKernel.Prolate
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
    u = evaluator.evaluate(positions, cells, tilts, gradients=()).energy.reshape(-1, 4)
    return (4.0 * (u[:, 2] - u[:, 3]) / step - (u[:, 0] - u[:, 1]) / (2.0 * step)) / 3.0


def position_differences(evaluator, x, cell, tilt, entries, step):
    """Central differences of the energy in the coordinates `entries`, pairs
    of a particle and an axis, extrapolated as above."""
    frames = np.repeat(x[None], 4 * len(entries), axis=0)
    for k, (i, c) in enumerate(entries):
        for j, s in enumerate((step, -step, 0.5 * step, -0.5 * step)):
            frames[4 * k + j, i, c] += s
    u = evaluator.evaluate(frames, cell, tilt, gradients=()).energy.reshape(-1, 4)
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
    print(f"{scenario} passed")


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

    # Restraints to reference positions, a term of the absolute positions
    # that the program adds to its forces after the evaluation.
    system, state = dipeptide("restrained")
    program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    reference = reference_evaluator(program, system, state)
    x, cell = np.array(state.positions), np.array(state.cell.diagonal)
    x = x + 0.01 * rng.standard_normal(x.shape)
    out = evaluator.evaluate(x[None], cell[None], gradients=("positions",))
    g = out.gradients["positions"][0]
    entries = [(i, c) for i in range(10) for c in range(3)]
    fd = position_differences(reference, x, cell, None, entries, POSITION_STEP)
    checked("with restraints to reference positions, G of the first 10 atoms of the solute",
            [g[i, c] for i, c in entries], fd, float(np.abs(g).max()), g_tolerance)
    # Their virial is taken about the references, and is not the derivative
    # of the energy of a frame in the strain: that gradient is refused.
    text = expect(mdir.UnsupportedError,
                  lambda: evaluator.evaluate(x[None], cell[None], gradients=("strain",)),
                  "restraints to reference positions")
    assert text.startswith("frame 0: "), text
    assert np.array_equal(evaluator.evaluate(x[None], cell[None]).energy, out.energy)
    print(f"{mode}: with restraints, the gradient in the strain is refused, and the evaluator "
          "stays usable")

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
    assert abs(out.energy[0] - out.energy[1]) < (1e-9 if double else 1e-6) * abs(out.energy[0])
    assert np.abs(g[0] - g[1]).max() < (1e-12 if double else 1e-5) * np.abs(g[0]).max()
    # The rows of the sites: zero in double precision; in mixed precision
    # what the spread leaves of an f32 force, an ulp of it (measured: 1.2e-4
    # kJ/mol/nm where the largest entry is 3,876).
    left = float(np.abs(g[:, sites]).max() / np.abs(g[0]).max())
    assert left == 0.0 if double else left < 1e-6, left
    assert np.any(g[0][atoms])
    entries = [(int(i), c) for i in rng.choice(atoms, 12, replace=False) for c in range(3)]
    fd = position_differences(reference, x, cell, None, entries, POSITION_STEP)
    checked(f"with {len(sites)} virtual sites, whose rows of G are {left:.1e} of the largest whatever their rows of the "
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
    # The default is the derivative in the tunables alone: `jacobian` and
    # `vjp`, and no gradient in the positions or the strain.
    plain = evaluator.evaluate(frames)
    assert plain.jacobian is not None and dict(plain.gradients) == {}
    assert np.array_equal(evaluator.evaluate(frames, gradients=("tunables",)).jacobian["sigma"],
                          plain.jacobian["sigma"])
    everything = ("tunables", "positions", "strain")
    both = evaluator.evaluate(frames, gradients=everything)
    g, s = both.gradients["positions"], both.gradients["strain"]
    assert list(both.gradients) == ["positions", "strain"] and both.jacobian is not None
    assert g.shape == (4, count, 3) and s.shape == (4, 3, 3) and s.dtype == np.float64
    assert not g.flags.writeable and not s.flags.writeable
    assert not hasattr(both.gradients, "__setitem__")
    # In the deterministic mode the forces of a program in mixed precision
    # are f64 (#102); without it they are f32, and so is the gradient.
    assert g.dtype == np.float64, g.dtype
    loose_program = compile_(system, state, deterministic=False)
    loose = mdir.FrameEvaluator(loose_program)
    narrow = loose.evaluate(frames, gradients=("positions",)).gradients["positions"]
    expected = np.float64 if double else np.float32
    assert narrow.dtype == expected, narrow.dtype
    apart = float(np.abs(narrow - g).max() / np.abs(g).max())
    print(f"{mode}: the gradient in the positions is {g.dtype} in the deterministic mode and "
          f"{narrow.dtype} without it, in the type of the forces of the program; the two differ "
          f"by {apart:.1e} relative")
    # Two programs, whose kernels differ: on a device the deterministic mode
    # contracts no product into a fused multiply-add and takes its sums in
    # one order (D201). Measured in double precision: 0 on the CPU and
    # 6.1e-12 on an RTX 4090, the rounding of sums of forces that cancel.
    assert apart < (1e-10 if double else 1e-5), apart
    # Without the deterministic mode, what a script compares: the gradient
    # of a frame against the forces of the sampler's state at that step.
    sampler = mdir.Simulation(loose_program)
    sampler.part_seconds = 1e9
    sampler.run(30)
    at = sampler.state()
    found = loose.evaluate(at, gradients=("positions",)).gradients["positions"][0]
    forces = np.asarray(at.forces)
    apart = float(np.abs(found + forces).max() / np.abs(forces).max())
    # A sampler and an evaluator without the deterministic mode: on a
    # device neither takes its sums in one order.
    tolerance = 1e-10 if double else 1e-5
    print(f"{mode}: without the deterministic mode, the gradient in the positions of a frame, "
          f"{found.dtype}, against the forces of the sampler's state at that step, "
          f"{forces.dtype}: {apart:.1e} of the largest ({np.abs(forces).max():.5g} kJ/mol/nm; "
          f"tolerance {tolerance:.0e})")
    assert apart < tolerance, apart

    # What is asked travels with each frame: calls that ask for nothing,
    # for one gradient, and for all three share an activation and give the
    # same numbers.
    def same(out, of=both):
        return (np.array_equal(out.energy, of.energy) and np.array_equal(out.virial, of.virial)
                and (out.jacobian is None or
                     all(np.array_equal(out.jacobian[n], of.jacobian[n]) for n in of.jacobian))
                and all(np.array_equal(a, of.gradients[n]) for n, a in out.gradients.items()))

    depth = evaluator._simulation._stack_in_use
    nothing = evaluator.evaluate(frames, gradients=())
    assert nothing.jacobian is None and len(nothing.gradients) == 0
    expect(mdir.InputError, lambda: nothing.vjp(np.ones(4)), "without 'tunables' in gradients")
    only_g = evaluator.evaluate(x, gradients=("positions",))
    only_s = evaluator.evaluate(frames, gradients=("strain",))
    assert list(only_g.gradients) == ["positions"] and list(only_s.gradients) == ["strain"]
    assert only_g.jacobian is None and only_s.jacobian is None
    again = evaluator.evaluate(x, gradients=("strain", "tunables", "positions", "strain"))
    assert list(again.gradients) == ["strain", "positions"] and again.jacobian is not None
    one = [evaluator.evaluate(x[n], gradients=everything) for n in (3, 0)]
    exact = (same(plain) and same(nothing) and same(only_g) and same(only_s) and same(again) and
             same(evaluator.evaluate(frames)) and
             all(np.array_equal(o.gradients[name][0], both.gradients[name][n])
                 for o, n in zip(one, (3, 0)) for name in ("positions", "strain")))
    assert evaluator._simulation._stack_in_use == depth
    print(f"{mode}: calls that ask for nothing, for one gradient, and for all three, as states, "
          f"as an array, and one frame alone, in one activation, equal to the bit: {exact}")
    assert exact

    # Refused.
    expect(mdir.InputError, lambda: evaluator.evaluate(x, gradients=("forces",)),
           "gradients names 'tunables', 'positions', and 'strain'")
    expect(mdir.InputError, lambda: evaluator.evaluate(x, gradients="strain"),
           "gradients is a tuple of names")
    expect(mdir.InputError, lambda: evaluator.evaluate(x, gradients=True),
           "gradients is a tuple of names")
    expect(TypeError, lambda: evaluator.evaluate(x, gradient=True), "gradient")
    dependent = mdir.FrameEvaluator(program, terms="dependent")
    assert "gradients" in dependent.unavailable and "gradients" not in evaluator.unavailable
    expect(mdir.UnsupportedError, lambda: dependent.evaluate(x, gradients=("strain",)),
           "terms='dependent'")
    partial = dependent.evaluate(x, gradients=("tunables",))
    assert len(partial.gradients) == 0 and partial.jacobian is not None
    assert same(evaluator.evaluate(frames, gradients=everything))
    print(f"{mode}: refused: a name that is not a gradient, names that are not a tuple, the "
          "argument `gradient` of before, the gradients in the positions and the strain of an "
          "evaluator of the dependent terms; the evaluator stays usable")
    print("interface passed")


BETA = 3.5  # the splitting parameter of the comparisons with other programs, 1/nm
COULOMB = 138.935457644  # kJ nm / (mol e^2), the constant of the programs


def oracle_model(points, kind="amber", charges=False):
    """The model of a comparison with another program: PME with the
    splitting parameter BETA and `points` points along each edge, a cutoff
    of 0.8 nm, the Lennard-Jones shifted to 0 there (the forces of a plain
    cutoff), no correction for the dispersion. `amber`: the dipeptide, or
    the system given after PRECISION; `water`: the rhombic dodecahedron."""
    if kind == "water":
        triclinic = root + "/../triclinic"
        loaded = mdir.load_gromacs(triclinic + "/water.top", triclinic + "/dodecahedron.gro",
                                   defines=["FLEXIBLE"])
    elif len(sys.argv) > 6:
        loaded = mdir.load_amber(sys.argv[5], sys.argv[6])
    else:
        loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Shift
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_alpha, system.pme_grid = BETA, [points] * 3
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = ([mdir.Tunable("q", "charge")] if charges else
                       [mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW"))])
    system.tunable_gradient = True
    return system, state


def first_grid():
    return int(sys.argv[7]) if len(sys.argv) > 7 else 64


def compared(label, found, reference, own, theirs, floor):
    """`found` against `reference` relative to the largest entry of the
    reference, within three times what the two programs change by between
    the two grids (`own`, `theirs`, relative as well), and in mixed
    precision the rounding of the kernels of the pairs besides."""
    scale = float(np.abs(reference).max())
    difference = float(np.abs(found - reference).max()) / scale
    tolerance = 3.0 * (own + theirs) + (0.0 if double else floor)
    print(f"{mode}: {label}: {difference:.1e} of the largest ({scale:.5g}); with twice the "
          f"points MDIR changes by {own:.1e} and the reference by {theirs:.1e}; tolerance "
          f"{tolerance:.1e}")
    assert difference < tolerance, (label, difference, tolerance)


def run_torchpme():
    import torch
    import torchpme
    torch.set_default_dtype(torch.float64)
    rc = 0.8

    def coulomb(points, kind):
        """The Coulomb part of G and S: the gradients at the charges of the
        model less those at charges of zero, at the same frame."""
        system, state = oracle_model(points, kind, charges=True)
        evaluator = mdir.FrameEvaluator(compile_(system, state))
        x = np.array(state.positions)
        cell = (np.array(state.cell.diagonal), np.array(state.cell.tilt))
        tilts = cell[1][None] if kind == "water" else None
        q0 = evaluator.tunables["q"].copy()
        charged = evaluator.evaluate(x[None], cell[0][None], tilts,
                                     gradients=("positions", "strain"))
        evaluator.tunables.update({"q": np.zeros_like(q0)})
        bare = evaluator.evaluate(x[None], cell[0][None], tilts,
                                  gradients=("positions", "strain"))
        g = charged.gradients["positions"][0].astype(np.float64) - bare.gradients["positions"][0]
        s = charged.gradients["strain"][0] - bare.gradients["strain"][0]
        return system, x, vectors(*cell), q0, g, s

    def reference(system, x, h, q0, spacing):
        """dU/dx and dU/d(epsilon) of the Coulomb energy of MDIR's model
        from torch-pme: its sum over the pairs within the cutoff that are
        not excluded and its sum of the mesh, less f q_i q_j erf(beta r) / r
        of the excluded pairs, which the mesh holds, plus the 1-4 pairs
        (f q_i q_j / (1.2 r), ff14SB). The shift of the direct sum (D210)
        is a constant of a frame. The distances are functions of the
        positions and the cell through the image of each pair, and both are
        deformed by a strain of zero that autograd differentiates in."""
        top = system.topology
        n = len(x)
        graph = {i: set() for i in range(n)}
        for i, j in top.bonds:
            graph[int(i)].add(int(j))
            graph[int(j)].add(int(i))
        amber = hasattr(top, "residue_names") and "WAT" in set(top.residue_names)
        excluded, scaled = set(), set()
        for i in range(n):
            seen, frontier = {i}, {i}
            for depth in range(3):
                frontier = {k for m in frontier for k in graph[m]} - seen
                seen |= frontier
                for j in frontier:
                    if j > i:
                        (scaled if depth == 2 else excluded).add((i, j))
        if not amber:
            scaled = set()      # the water of GROMACS: no pair three bonds apart

        def image(d):
            """The lattice vector, in units of the cell vectors, that brings
            each difference to its nearest image, for a reduced cell."""
            k = np.zeros_like(d)
            for axis in (2, 1, 0):
                k[:, axis] = -np.round((d + k @ h)[:, axis] / h[axis, axis])
            return k

        near, shifts = [], []
        for i in range(n - 1):
            d = x[i + 1:] - x[i]
            k = image(d)
            r = np.sqrt(((d + k @ h) ** 2).sum(axis=1))
            for j in np.nonzero(r < rc)[0]:
                pair = (i, i + 1 + int(j))
                if pair not in excluded and pair not in scaled:
                    near.append(pair)
                    shifts.append(k[j])
        held =np.array(sorted(excluded | scaled))
        fourteen = np.array(sorted(scaled)).reshape(-1, 2)
        held_shift = image(x[held[:, 1]] - x[held[:, 0]])
        fourteen_shift = image(x[fourteen[:, 1]] - x[fourteen[:, 0]]) if len(fourteen) else None
        positions = torch.tensor(x, requires_grad=True)
        strain = torch.zeros(3, 3, requires_grad=True)
        deform = torch.eye(3) + strain
        moved, cell = positions @ deform.T, torch.tensor(h) @ deform.T
        charges = torch.tensor(q0)

        def distances(pairs, shift):
            pairs = torch.tensor(np.asarray(pairs))
            d = moved[pairs[:, 1]] - moved[pairs[:, 0]] + torch.tensor(np.asarray(shift)) @ cell
            return pairs, torch.sqrt((d * d).sum(dim=1))

        product = lambda pairs: charges[pairs[:, 0]] * charges[pairs[:, 1]]
        near_pairs, near_r = distances(near, shifts)
        potential = torchpme.CoulombPotential(smearing=1.0 / (np.sqrt(2.0) * BETA),
                                              prefactor=COULOMB)
        calculator = torchpme.PMECalculator(potential, mesh_spacing=spacing,
                                            interpolation_nodes=5)
        column = charges.unsqueeze(1)
        energy = (column * calculator(column, cell, moved, near_pairs, near_r)).sum()
        held_pairs, held_r = distances(held, held_shift)
        energy = energy - (COULOMB * product(held_pairs) * torch.erf(BETA * held_r) / held_r).sum()
        if len(fourteen):
            scaled_pairs, scaled_r = distances(fourteen, fourteen_shift)
            energy = energy + (COULOMB * product(scaled_pairs) / (1.2 * scaled_r)).sum()
        g, s = torch.autograd.grad(energy, (positions, strain))
        return g.numpy(), s.numpy(), len(near)

    def one(kind, points, name):
        system, x, h, q0, g, s = coulomb(points, kind)
        _, _, _, _, g2, s2 = coulomb(2 * points, kind)
        # torch-pme takes the power of 2 at or above 2 L / spacing + 1.
        length = float(np.sqrt((h * h).sum(axis=1)).max())
        spacings = [2.0 * length / (m - 1.5) for m in (points, 2 * points)]
        for m, spacing in zip((points, 2 * points), spacings):
            mesh = [int(k) for k in torchpme.lib.get_ns_mesh(torch.tensor(h), spacing)]
            assert mesh == [m] * 3, (mesh, m)
        rg, rs, pairs = reference(system, x, h, q0, spacings[0])
        rg2, rs2, _ = reference(system, x, h, q0, spacings[1])
        scale_g, scale_s = np.abs(rg).max(), np.abs(rs).max()
        compared(f"{name}, {len(x)} particles, {pairs} pairs: the Coulomb part of G against "
                 f"torch-pme, grid {points}", g, rg, np.abs(g - g2).max() / scale_g,
                 np.abs(rg - rg2).max() / scale_g, 2e-5)
        compared(f"{name}: the Coulomb part of the nine components of S, grid {points}", s, rs,
                 np.abs(s - s2).max() / scale_s, np.abs(rs - rs2).max() / scale_s, 2e-5)
        compared(f"{name}: the same at {2 * points} points, G", g2, rg2,
                 np.abs(g - g2).max() / scale_g, np.abs(rg - rg2).max() / scale_g, 2e-5)
        compared(f"{name}: the same at {2 * points} points, S", s2, rs2,
                 np.abs(s - s2).max() / scale_s, np.abs(rs - rs2).max() / scale_s, 2e-5)

    # One system a process: on a device the runtime keeps a plan of FFT for
    # each grid, direction, and precision of a process, 8 at most, and a
    # program in mixed precision with tunable charges takes four a grid;
    # the two grids of one system fill that.
    if scenario == "torchpme-water":
        one("water", 32, "water in a rhombic dodecahedron")
    else:
        one("amber", first_grid(),
            "the dipeptide in water" if len(sys.argv) <= 6 else "the system")
    print("torchpme passed")


def run_openmm():
    import openmm
    from openmm import app, unit

    def own(points):
        system, state = oracle_model(points)
        evaluator = mdir.FrameEvaluator(compile_(system, state))
        x, cell = np.array(state.positions), np.array(state.cell.diagonal)
        out = evaluator.evaluate(x[None], cell[None], gradients=("positions", "strain"))
        return x, cell, (out.gradients["positions"][0].astype(np.float64),
                         out.gradients["strain"][0])

    prmtop = app.AmberPrmtopFile(sys.argv[5] if len(sys.argv) > 6 else root + "/dipeptide.prmtop")
    rc = 0.8

    def theirs(points, x, cell):
        system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=rc * unit.nanometer,
                                     constraints=None, rigidWater=False, removeCMMotion=False)
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setPMEParameters(BETA, points, points, points)
                force.setUseDispersionCorrection(False)
                nonbonded = force
        # OpenMM cuts the Lennard-Jones and the direct sum at the cutoff
        # without a shift, so its energy jumps where a pair crosses it, and
        # a difference of it has the jumps. The energy differenced is the
        # one whose forces OpenMM gives: its energy less u_ij(r_c) of every
        # pair within the cutoff that is not an exception, the Lennard-Jones
        # with the combining rule of Lorentz and Berthelot and
        # f q_i q_j erfc(beta r_c) / r_c, summed here from its parameters.
        count = system.getNumParticles()
        parameters = [nonbonded.getParticleParameters(i) for i in range(count)]
        charge = np.array([p[0].value_in_unit(unit.elementary_charge) for p in parameters])
        lj_sigma = np.array([p[1].value_in_unit(unit.nanometer) for p in parameters])
        lj_epsilon = np.array([p[2].value_in_unit(unit.kilojoule_per_mole) for p in parameters])
        partners = [[] for _ in range(count)]
        for k in range(nonbonded.getNumExceptions()):
            i, j = nonbonded.getExceptionParameters(k)[:2]
            partners[min(i, j)].append(max(i, j))
        direct = COULOMB * math.erfc(BETA * rc) / rc

        def shifts(positions, h):
            total = 0.0
            for i in range(count - 1):
                d = positions[i + 1:] - positions[i]
                k = np.zeros_like(d)
                for axis in (2, 1, 0):
                    k[:, axis] = -np.round((d + k @ h)[:, axis] / h[axis, axis])
                d = d + k @ h
                within = (d * d).sum(axis=1) < rc * rc
                if partners[i]:
                    within[np.array(partners[i]) - i - 1] = False
                s6 = (0.5 * (lj_sigma[i] + lj_sigma[i + 1:][within]) / rc) ** 6
                total += float((4.0 * np.sqrt(lj_epsilon[i] * lj_epsilon[i + 1:][within]) *
                                (s6 * s6 - s6)).sum())
                total += direct * charge[i] * float(charge[i + 1:][within].sum())
            return total
        # The Reference platform, in double precision, whatever the target
        # of MDIR.
        context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName("Reference"))

        def at(positions, h, forces=False):
            context.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in h])
            context.setPositions(positions)
            state = context.getState(getEnergy=True, getForces=forces)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if not forces:
                return energy - shifts(positions, h)
            return energy, state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer)

        h = np.diag(cell)
        _, forces = at(x, h, forces=True)
        # No virial is reported: central differences of that energy under
        # the strain, the cell rotated back to the reduced form, which
        # OpenMM takes as well.
        step, strain = 2e-5, np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                u = []
                for s in (step, -step, 0.5 * step, -0.5 * step):
                    epsilon = np.zeros((3, 3))
                    epsilon[a, b] = s
                    u.append(at(*deformed(x, h, epsilon)))
                strain[a, b] = (4.0 * (u[2] - u[3]) / step - (u[0] - u[1]) / (2.0 * step)) / 3.0
        return -np.asarray(forces), strain

    points = first_grid()
    x, cell, (g, s) = own(points)
    _, _, (g2, s2) = own(2 * points)
    rg, rs = theirs(points, x, cell)
    rg2, rs2 = theirs(2 * points, x, cell)
    scale_g, scale_s = np.abs(rg).max(), np.abs(rs).max()
    name = "the dipeptide in water" if len(sys.argv) <= 6 else "the system"
    compared(f"{name}, {len(x)} particles: G of the whole potential against the forces of OpenMM, "
             f"grid {points}", g, rg, np.abs(g - g2).max() / scale_g,
             np.abs(rg - rg2).max() / scale_g, 5e-5)
    compared(f"{name}: the nine components of S against central differences of the energy of "
             f"OpenMM, grid {points}", s, rs, np.abs(s - s2).max() / scale_s,
             np.abs(rs - rs2).max() / scale_s, 5e-5)
    compared(f"{name}: the same at {2 * points} points, G", g2, rg2,
             np.abs(g - g2).max() / scale_g, np.abs(rg - rg2).max() / scale_g, 5e-5)
    compared(f"{name}: the same at {2 * points} points, S", s2, rs2,
             np.abs(s - s2).max() / scale_s, np.abs(rs - rs2).max() / scale_s, 5e-5)
    print("openmm passed")


def run_torch():
    import torch
    import mdir.torch

    device = "cuda" if target == "GPU" else "cpu"
    rng = np.random.default_rng(3)

    # The dipeptide: three frames of a run, each with a cell of its own.
    system, state = dipeptide()
    program = compile_(system, state)
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    frames = []
    for _ in range(3):
        sim.run(20)
        frames.append(np.array(sim.state().positions))
    x = np.stack(frames)
    cells = np.array(state.cell.diagonal) * (1.0 + 1e-3 * rng.standard_normal((3, 3)))
    evaluator = mdir.FrameEvaluator(program)
    arrays = evaluator.evaluate(x, cells, gradients=("tunables", "positions", "strain"))
    g, s = arrays.gradients["positions"], arrays.gradients["strain"]
    sigma = torch.tensor(evaluator.tunables["sigma"], device=device, requires_grad=True)
    weights = torch.tensor(rng.normal(size=3), device=device)

    def tensors(positions=x, cell=cells, where="cpu"):
        return (torch.tensor(positions, device=where, requires_grad=True),
                torch.tensor(cell, device=where, requires_grad=True))

    # The backward in the positions is the product with G, and the one in
    # the tunables is what it was.
    for where in dict.fromkeys(("cpu", device)):
        tx, tc = tensors(where=where)
        out = mdir.torch.evaluate(evaluator, {"sigma": sigma}, tx, tc)
        by_x, by_c, by_sigma = torch.autograd.grad((weights * out.energy).sum(), (tx, tc, sigma))
        assert by_x.shape == tx.shape and by_x.dtype == tx.dtype and by_x.device == tx.device
        assert by_c.shape == tc.shape and by_c.device == tc.device
        expected = weights.cpu().numpy()[:, None, None] * g.astype(np.float64)
        exact = np.array_equal(by_x.cpu().numpy(), expected)
        product = arrays.vjp(weights.cpu().numpy())["sigma"]
        assert np.allclose(by_sigma.cpu().numpy(), product, rtol=1e-13, atol=0.0)
    print(f"{mode}: the gradient of autograd in positions (K, N, 3) that require one against the "
          f"cotangent times G of the evaluator, to the bit: {exact}")
    assert exact
    # In float32 positions, and for one frame without its axis.
    tx = torch.tensor(x[1].astype(np.float32), requires_grad=True)
    out = mdir.torch.evaluate(evaluator, {"sigma": sigma.detach()}, tx, cells[1])
    by_x, = torch.autograd.grad(out.energy.sum(), tx)
    narrow = mdir.FrameEvaluator(program).evaluate(x[1].astype(np.float32), cells[1],
                                                   gradients=("positions",))
    assert by_x.shape == (x.shape[1], 3) and by_x.dtype == torch.float32
    assert np.array_equal(by_x.numpy(), narrow.gradients["positions"][0].astype(np.float32))
    assert out.energy.grad_fn is not None

    # The gradient in the cell at fixed positions against the differences
    # of PyTorch, in double precision: the three edges of each frame, and
    # one cell for every frame.
    def cell_gradient(evaluator_, positions, cell, tilt=None):
        tc = torch.tensor(cell, requires_grad=True)
        tt = None if tilt is None else torch.tensor(tilt, requires_grad=True)
        out = mdir.torch.evaluate(evaluator_, {"sigma": sigma.detach()}, positions, tc, tt)
        return [t.numpy() for t in torch.autograd.grad(
            (weights[:len(out.energy)].to(out.energy.device) * out.energy).sum(),
            (tc,) if tt is None else (tc, tt))]

    def differences(evaluator_, positions, cell, tilt, which, at, step=2e-5):
        """A central difference of the weighted sum of the energies in the
        entry `at` of the cells (`which` 0) or the tilts (1), the positions
        fixed."""
        values = []
        for sign in (1.0, -1.0, 0.5, -0.5):
            moved = [np.array(cell, dtype=np.float64), None if tilt is None else np.array(tilt)]
            moved[which][at] += sign * step
            u = evaluator_.evaluate(positions, moved[0], moved[1], gradients=()).energy
            values.append(float(weights.cpu().numpy()[:len(u)] @ u))
        return (4.0 * (values[2] - values[3]) / step - (values[0] - values[1]) / (2.0 * step)) / 3.0

    reference = reference_evaluator(program, system, state)
    by_c, = cell_gradient(evaluator, x, cells)
    entries = [(n, a) for n in range(3) for a in range(3)]
    fd = np.array([differences(reference, x, cells, None, 0, at) for at in entries])
    # The differences bound the comparison: an edge moved at fixed
    # positions carries every image across the cutoff with it, and the
    # gradient is a small difference of S and the moment of G (636 kJ/mol/nm
    # where S is 14,000 kJ/mol). Measured in double precision, the largest
    # error of the nine entries is 2e-2 kJ/mol/nm at a step of 2e-4 nm,
    # 1.3e-3 at 2e-5, and 1.2e-2 at 2e-6.
    tolerance = 1e-5 if double else 1e-3
    checked("the gradient of autograd in the edges of the cells (K, 3) at fixed positions against "
            "central differences of the energy", [by_c[at] for at in entries], fd,
            float(np.abs(by_c).max()), tolerance)
    shared, = cell_gradient(evaluator, x, cells[0])
    assert shared.shape == (3,)
    each, = cell_gradient(evaluator, x, np.broadcast_to(cells[0], (3, 3)).copy())
    assert np.allclose(shared, each.sum(axis=0), rtol=1e-13, atol=0.0)

    # A triclinic cell: the diagonal and the tilts.
    wsystem, wstate = water()
    wprogram = compile_(wsystem, wstate)
    wx, wh = sheared_water(wstate)
    wcell, wtilt = np.diag(wh).copy(), np.array([wh[1, 0], wh[2, 0], wh[2, 1]])
    wevaluator = mdir.FrameEvaluator(wprogram)
    wsigma = torch.tensor(wevaluator.tunables["sigma"])
    wreference = reference_evaluator(wprogram, wsystem, wstate)
    tc, tt = torch.tensor(wcell, requires_grad=True), torch.tensor(wtilt, requires_grad=True)
    out = mdir.torch.evaluate(wevaluator, {"sigma": wsigma}, wx, tc, tt)
    by_c, by_t = [t.numpy() for t in torch.autograd.grad(out.energy.sum(), (tc, tt))]
    one = torch.ones(1)
    weights, kept = one, weights
    fd_c = np.array([differences(wreference, wx, wcell, wtilt, 0, a) for a in range(3)])
    fd_t = np.array([differences(wreference, wx, wcell, wtilt, 1, a) for a in range(3)])
    weights = kept
    scale = float(max(np.abs(by_c).max(), np.abs(by_t).max()))
    checked("in a triclinic cell, the gradient of autograd in the diagonal against central "
            "differences of the energy", by_c, fd_c, scale, tolerance)
    checked("in a triclinic cell, the gradient in the tilts", by_t, fd_t, scale, tolerance)
    if double:
        def function(c, t):
            return mdir.torch.evaluate(wevaluator, {"sigma": wsigma}, wx, c, t).energy
        passed = torch.autograd.gradcheck(function, (tc, tt), eps=2e-5, atol=2e-2, rtol=1e-5)
        print(f"{mode}: torch.autograd.gradcheck of the energy in the diagonal and the tilts of a "
              f"triclinic cell: {passed}")
        assert passed

    # The gradients as outputs: the arrays of the evaluator, on the device
    # of theta; a backward that reaches one is refused, and so is every
    # second derivative.
    tx, tc = tensors(where=device)
    out = mdir.torch.evaluate(evaluator, {"sigma": sigma}, tx, tc, gradients=("strain", "positions"))
    assert list(out.gradients) == ["strain", "positions"]
    given_g, given_s = out.gradients["positions"], out.gradients["strain"]
    assert given_g.device.type == device and given_s.dtype == torch.float64
    assert np.array_equal(given_g.detach().cpu().numpy(), g)
    assert np.array_equal(given_s.detach().cpu().numpy(), s)
    text = expect(mdir.UnsupportedError, lambda: (given_g ** 2).sum().backward(),
                  "the gradient of the energy in the positions")
    assert "not implemented" in text and "not zero" in text, text
    out = mdir.torch.evaluate(evaluator, {"sigma": sigma}, x, cells, gradients=("strain",))
    expect(mdir.UnsupportedError, lambda: out.gradients["strain"].sum().backward(),
           "the gradient of the energy in the strain")
    out = mdir.torch.evaluate(evaluator, {"sigma": sigma}, x, cells, gradients=("strain",))
    (out.energy.sum() + out.gradients["strain"].detach().sum()).backward()
    with torch.no_grad():
        constant = mdir.torch.evaluate(evaluator, {"sigma": sigma}, x, cells,
                                       gradients=("positions",))
    assert constant.gradients["positions"].grad_fn is None
    # With positions that take a gradient the virial is a function of them.
    tx, tc = tensors()
    out = mdir.torch.evaluate(evaluator, {"sigma": sigma.detach()}, tx, tc)
    expect(mdir.UnsupportedError, lambda: out.virial.sum().backward(), "'virial'")
    for inputs in ("positions", "cells", "both"):
        tx, tc = tensors()
        out = mdir.torch.evaluate(evaluator, {"sigma": sigma}, tx, tc)
        of = {"positions": (tx,), "cells": (tc,), "both": (tx, tc, sigma)}[inputs]
        first = torch.autograd.grad(out.energy.sum(), of, create_graph=True)
        expect(mdir.UnsupportedError, lambda: sum((f ** 2).sum() for f in first).backward(),
               "second derivative")
        # With another path to the inputs as well: not a zero.
        tx, tc = tensors()
        out = mdir.torch.evaluate(evaluator, {"sigma": sigma}, tx, tc)
        of = {"positions": (tx,), "cells": (tc,), "both": (tx, tc, sigma)}[inputs]
        first = torch.autograd.grad(out.energy.sum(), of, create_graph=True)
        loss = sum((f ** 2).sum() for f in first) + (tx ** 2).sum() + (tc ** 2).sum()
        expect(mdir.UnsupportedError, loss.backward, "second derivative")
    dependent = mdir.FrameEvaluator(program, terms="dependent")
    tx, tc = tensors()
    for call in (lambda: mdir.torch.evaluate(dependent, {"sigma": sigma}, tx),
                 lambda: mdir.torch.evaluate(dependent, {"sigma": sigma}, x, tc),
                 lambda: mdir.torch.evaluate(dependent, {"sigma": sigma}, x, gradients=("strain",))):
        expect(mdir.UnsupportedError, call, "terms='dependent'")
    expect(mdir.InputError,
           lambda: mdir.torch.evaluate(evaluator, {"sigma": sigma}, x, gradients=("forces",)),
           "gradients names")
    expect(mdir.UnsupportedError,
           lambda: mdir.torch.evaluate(evaluator, {"sigma": sigma}, [tx[0], tx[1]]),
           "positions require a gradient")
    assert all(r.out is None and r.frames is None for r in mdir.torch._RECORDS.values())
    print(f"{mode}: refused: a backward through a gradient that is an output, through the virial "
          "of positions that require a gradient, every second derivative, alone and beside "
          "another path, the gradients of the dependent terms, frames of an iterable that "
          "require a gradient")

    # Inside torch.compile.
    def loss(positions, cell, sig):
        out = mdir.torch.evaluate(evaluator, {"sigma": sig}, positions, cell)
        return (weights * out.energy).sum()
    tx, tc = tensors(where=device)
    eager = torch.autograd.grad(loss(tx, tc, sigma), (tx, tc, sigma))
    traced = torch.autograd.grad(torch.compile(loss, backend="aot_eager")(tx, tc, sigma),
                                 (tx, tc, sigma))
    same = all(torch.equal(a, b) for a, b in zip(eager, traced))
    print(f"{mode}: under torch.compile (aot_eager), the same gradients in the positions, the "
          f"cells, and the tunable: {same}")
    assert same
    print("torch passed")


if scenario in ("finite", "prolate"):
    run_finite()
elif scenario == "torch":
    run_torch()
elif scenario in ("torchpme", "torchpme-water"):
    run_torchpme()
elif scenario == "openmm":
    run_openmm()
elif scenario == "special":
    run_special()
elif scenario == "interface":
    run_interface()
else:
    raise SystemExit(f"unknown scenario {scenario}")
