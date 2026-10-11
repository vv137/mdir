"""The derivative of the energy in the tunable parameters
(D230, docs/python-gradient.md).

Usage: python_tunable_gradient.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  refusals  what is refused and with which error, the plan, and a tunable
            that the energy does not read (CPU)
  terms     constants of a pair term and parameters of a term over tuples on
            the dipeptide in water with PME: the derivative against central
            differences of the energy that it is the derivative of, and
            against NumPy for the term over tuples; a run with the
            derivative asked for against one without it, bit for bit
  tails     a constant of a pair term under a plain cutoff with the
            correction for the dispersion, on propane and water: the
            derivative of the shifted potential with the tail and the
            estimate of the shift (D209, D210) against NumPy and closed
            forms, and against central differences
  charges   the charges under a Coulomb cutoff on the dipeptide in water,
            tied by a map (one entry per atom name of the waters): the
            derivative of the shifted Coulomb of the pairs within the cutoff
            and of the 1-4 pairs against NumPy, and against central
            differences
  dynamics  leapfrog under a barostat (NPT): after 200 steps on the dipeptide
            with the Lennard-Jones cut without a shift, the correction for
            the dispersion, and the table tunable by pairs of types, the
            derivative against central differences of the energy at the
            volume that the run has reached
  checkpoint  a simulation continued from a checkpoint, and a new stage from
            it, give the derivative of the simulation that wrote it, to the
            bit (needs HDF5)
  pme       the charges with particle mesh Ewald on the dipeptide in water,
            tied by a map, with a net charge: the derivative (the direct
            sum, the potential of the grid at the particles, the self term,
            the background, the excluded and the 1-4 pairs) against central
            differences of the energy
  torchpme  the charges with particle mesh Ewald, every particle an entry of
            its own, against torch-pme, an independent differentiable PME
            (Loche et al., J. Chem. Phys. 162, 142501 (2025)), at the same
            positions, cell, charges, splitting parameter, cutoff, and
            numbers of grid points: the derivative in the charges from
            torch autograd, and the change of the energy for other charges.
            The tolerance is what each of the two changes by when its grid
            is refined. Needs an interpreter with torch and torchpme.
  lj        per-type sigma and epsilon and the table by pairs of types: on
            the dipeptide against central differences; on propane and
            water under a plain cutoff with the correction for the
            dispersion, per-type (geometric) and pair tunables together
            with a pair set apart from the rule, against a NumPy sum of the
            shifted Lennard-Jones with the correction and its shift
            estimate
"""
import sys

import numpy as np
import mdir

root = sys.argv[1]
scenario = sys.argv[2]


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text})")


def soft_term():
    # A repulsion between all pairs, in nm and kJ/mol; `z` is a constant
    # that the expression does not read.
    term = mdir.PairTerm()
    term.name, term.expression = "soft", "a*exp(-r/l)"
    term.constants = [("a", 2.0), ("l", 0.05), ("z", 1.0)]
    return term


SPRINGS = np.array([[1, 4], [4, 6], [6, 8], [1, 6], [1, 8], [4, 8]], dtype=np.int64)


def spring_term():
    # Springs whose tuples share particles at both places, so that the
    # derivative in the parameter of each tuple needs several fields.
    term = mdir.TupleTerm()
    term.name, term.expression, term.arity = "spring", "0.5*k*(r - r0)^2", 2
    term.particles = SPRINGS
    term.parameters = [("k", np.array([500.0, 600.0, 600.0, 300.0, 200.0, 100.0])),
                       ("r0", np.array([0.25, 0.26, 0.27, 0.3, 0.35, 0.4]))]
    return term


K_MAP = np.array([0, 1, 1, 2, -1, 0])


def dipeptide(tunables, gradient=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.pair_terms = [soft_term()]
    system.tuple_terms = [spring_term()]
    system.tunables = tunables
    system.tunable_gradient = gradient
    return system, state


def term_tunables():
    return [mdir.Tunable("soft_a", "a", term="soft"),
            mdir.Tunable("soft_l", "l", term="soft"),
            mdir.Tunable("soft_z", "z", term="soft"),
            mdir.Tunable("k", "k", term="spring", map=K_MAP,
                         values=np.array([450.0, 600.0, 310.0])),
            mdir.Tunable("r0", "r0", term="spring")]


def compile_(system, state, target="CPU", precision="Double"):
    integrator, execution = mdir.Integrator(), mdir.Execution()
    integrator.timestep = 0.0005
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())


def simulation(program):
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    return sim


def differences(sim, g, step, entries=None):
    """The largest difference, relative to the largest entry of each
    tunable, between the derivative `g` and central differences of the
    energy that it is the derivative of, extrapolated from the steps `step`
    and `step`/2 times each value (Richardson); of the entries
    `entries[name]` of each tunable, or of all."""
    worst = 0.0
    for name in g.keys():
        if name in g.zero:
            continue
        v0 = sim.tunables[name].copy()
        scale = max(np.abs(g[name]).max(), 1e-12)
        for m in (entries[name] if entries else range(len(v0))):
            def central(h):
                e = []
                for sign in (1.0, -1.0):
                    v = v0.copy()
                    v[m] += sign * h
                    sim.tunables[name] = v
                    e.append(sim.tunables.gradient().energy)
                return (e[0] - e[1]) / (2.0 * h)
            h = step * max(abs(v0[m]), 1e-3)
            fd = (4.0 * central(0.5 * h) - central(h)) / 3.0
            worst = max(worst, abs(g[name][m] - fd) / scale)
        sim.tunables[name] = v0
    return worst


def run_refusals():
    system, state = dipeptide(term_tunables())
    program = compile_(system, state)
    plan = {d["name"]: d["gradient"] for d in program.plan["tunables"]}
    assert plan == {"soft_a": "rule", "soft_l": "rule", "soft_z": "zero", "k": "rule",
                    "r0": "rule"}, plan
    sim = simulation(program)
    # Before the first run the evaluation is the start of the run.
    g = sim.tunables.gradient()
    assert sorted(g.keys()) == sorted(plan) and len(g) == 5 and "k" in g
    assert g.zero == frozenset({"soft_z"}) and np.all(g["soft_z"] == 0.0)
    assert g["k"].shape == (3,) and g["r0"].shape == (6,) and g["soft_a"].shape == (1,)
    assert g["k"].dtype == np.float64 and not g["k"].flags.writeable
    assert g.step == 0 and g.version == 0
    assert abs(g.energy - sim.state().energies["potential"]) < 0.5
    sim.tunables["soft_a"] = np.array([2.5])
    assert sim.tunables.gradient().version == 1
    # A live writable borrow refuses it, as it refuses an evaluation (D229).
    sim.run(2)
    borrow = sim.borrow()
    expect(mdir.SimulationError, sim.tunables.gradient, "writable borrow")
    borrow.abandon()
    del borrow
    assert sim.tunables.gradient().step == 2
    with sim.view():
        expect(mdir.SimulationError, sim.tunables.gradient, "an evaluation")
    expect(KeyError, lambda: g["q"])
    print("refusals: the plan, the shapes, and a tunable that the energy does not read")

    # Without the compile input the program has no derivative, and says so.
    system, state = dipeptide(term_tunables(), gradient=False)
    program = compile_(system, state)
    assert all("gradient" not in d for d in program.plan["tunables"])
    assert "tunable" not in program.ir.replace("tunable_constants", "")
    sim = simulation(program)
    expect(mdir.InputError, sim.tunables.gradient, "System.tunable_gradient = True")
    # Without tunables.
    system, state = dipeptide([], gradient=False)
    sim = simulation(compile_(system, state))
    expect(mdir.InputError, sim.tunables.gradient, "declares no tunable")
    system, state = dipeptide([])
    expect(mdir.InputError, lambda: compile_(system, state), "the system declares none")
    print("refusals: 5 refusals")


def run_terms(target, precision):
    double = precision == "Double"
    system, state = dipeptide(term_tunables())
    program = compile_(system, state, target, precision)
    sim = simulation(program)
    sim.run(6)
    g = sim.tunables.gradient()
    at = sim.state()

    # NumPy: the springs at the positions of the state, in the minimum
    # image: dU/dk = (r - r0)^2 / 2 and dU/dr0 = -k (r - r0) for each tuple,
    # added over the tuples of each entry.
    x, box = at.positions, np.diag(at.cell.vectors)
    d = x[SPRINGS[:, 0]] - x[SPRINGS[:, 1]]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    k_sites = np.array([500.0, 600.0, 600.0, 300.0, 200.0, 100.0])
    theta_k = sim.tunables["k"]
    k_sites = np.where(K_MAP >= 0, theta_k[np.maximum(K_MAP, 0)], k_sites)
    r0 = sim.tunables["r0"]
    dk = np.zeros(3)
    np.add.at(dk, K_MAP[K_MAP >= 0], (0.5 * (r - r0) ** 2)[K_MAP >= 0])
    dr0 = -k_sites * (r - r0)
    err_k = np.abs(g["k"] - dk).max() / np.abs(dk).max()
    err_r0 = np.abs(g["r0"] - dr0).max() / np.abs(dr0).max()
    tolerance = 1e-11 if double else 2e-6
    print(f"{target} {precision}: the term over tuples against NumPy, k {err_k:.1e}, "
          f"r0 {err_r0:.1e} relative (tolerance {tolerance:.0e})")
    assert err_k < tolerance and err_r0 < tolerance, (g["k"], dk, g["r0"], dr0)

    # Central differences of the energy that it is the derivative of.
    worst = differences(sim, g, 1e-3 if double else 4e-2)
    tolerance = 1e-8 if double else 2e-4
    print(f"{target} {precision}: against central differences of the energy, "
          f"{worst:.1e} of the largest entry of each tunable (tolerance {tolerance:.0e})")
    assert worst < tolerance, worst

    # The energy is that which the run reports, but for the shift of the
    # direct sum of PME to 0 at the cutoff (D210): with erfc(beta r_c) =
    # 1e-5 here, 1.8e-3 kJ/mol for each product of charges of 1 e^2 within
    # the cutoff.
    shift = g.energy - at.energies["potential"]
    print(f"{target} {precision}: the energy less the reported one {shift:.3f} kJ/mol")
    assert 0.0 < shift < 0.5, shift

    # A run through an evaluation with the derivative equals one through an
    # evaluation without it, to the bit: the steps take nothing of it.
    def final(asks):
        system, state = dipeptide(term_tunables(), gradient=asks)
        sim = simulation(compile_(system, state, target, precision))
        sim.run(6)
        if asks:
            sim.tunables.gradient()
        else:
            sim.run(0, energy=True)
        sim.run(6)
        return sim.state()

    a, b = final(True), final(False)
    same = all(np.array_equal(getattr(a, f), getattr(b, f))
               for f in ("positions", "velocities", "forces"))
    print(f"{target} {precision}: 12 steps through the derivative equal those without it "
          f"to the bit: {same}")
    assert same


def run_charges(target, precision):
    double = precision == "Double"
    rc, f = 0.8, 138.935457644
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = rc, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    names = top.atom_names
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    charge_map = np.array(charge_map, dtype=np.int64)
    system.tunables = [mdir.Tunable("q", "charge", map=charge_map)]
    system.tunable_gradient = True
    sim = simulation(compile_(system, state, target, precision))
    sim.run(6)
    g = sim.tunables.gradient()
    at = sim.state()
    x, box = at.positions, np.diag(at.cell.vectors)
    n = len(x)
    q = sim.tunables["q"][charge_map]

    # NumPy. The pairs up to two bonds apart are excluded; those three
    # bonds apart are the 1-4 pairs, f q_i q_j / (1.2 r) in the force
    # field of the file (ff14SB), and are excluded from the others, which
    # take f q_i q_j (1 / r - 1 / rc) within the cutoff: the Coulomb that
    # the forces sample (D210).
    graph = {i: set() for i in range(n)}
    for i, j in top.bonds:
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    excluded, scaled = set(), set()
    for i in range(n):
        seen, frontier = {i}, {i}
        for depth in range(3):
            frontier = {k for m in frontier for k in graph[m]} - seen
            seen |= frontier
            for j in frontier:
                if j > i:
                    (scaled if depth == 2 else excluded).add((i, j))
    potential = np.zeros(n)
    for i in range(n - 1):
        d = x[i + 1:] - x[i]
        d -= box * np.round(d / box)
        r = np.sqrt((d * d).sum(axis=1))
        for j in np.nonzero(r < rc)[0]:
            pair = (i, i + 1 + int(j))
            if pair in excluded or pair in scaled:
                continue
            k = f * (1.0 / r[j] - 1.0 / rc)
            potential[i] += k * q[pair[1]]
            potential[pair[1]] += k * q[i]
    for i, j in scaled:
        d = x[j] - x[i]
        d -= box * np.round(d / box)
        k = f / (1.2 * np.sqrt((d * d).sum()))
        potential[i] += k * q[j]
        potential[j] += k * q[i]
    reference = np.zeros(len(g["q"]))
    np.add.at(reference, charge_map, potential)
    error = np.abs(g["q"] - reference).max() / np.abs(reference).max()
    tolerance = 1e-9 if double else 5e-6
    print(f"{target} {precision}: the charges under a Coulomb cutoff, {len(reference)} "
          f"entries for {n} particles, against NumPy, {error:.1e} of the largest "
          f"(tolerance {tolerance:.0e})")
    assert error < tolerance, error
    worst = differences(sim, g, 1e-3 if double else 5e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{target} {precision}: against central differences of the energy, {worst:.1e} "
          f"(tolerance {tolerance:.0e})")
    assert worst < tolerance, worst


def pair_model():
    """The dipeptide in water with PME, the Lennard-Jones cut without a
    shift with the correction for the dispersion, and the table tunable by
    pairs of types."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.EnergyPressure
    system.tunables = [mdir.Tunable("sigma", "sigma_pair"),
                       mdir.Tunable("epsilon", "epsilon_pair")]
    system.tunable_gradient = True
    return system, state


def run_dynamics(target, precision):
    double = precision == "Double"
    system, state = pair_model()
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    ensemble.com_period = 0  # the default of before D[python-defaults]
    integrator.timestep = 0.0005
    integrator.method = mdir.IntegratorMethod.Leapfrog
    ensemble.kind, ensemble.temperature, ensemble.coupling_period = mdir.EnsembleKind.NPT, 300.0, 10
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = True
    sim = simulation(mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule()))
    first = float(np.linalg.det(sim.state().cell.vectors))
    sim.run(200)
    volume = float(np.linalg.det(sim.state().cell.vectors))
    assert volume != first
    g = sim.tunables.gradient()
    assert g.step == 200
    # The four largest entries of each and the smallest that is not 0.
    entries = {}
    for name in g.keys():
        order = np.argsort(-np.abs(g[name]))
        nonzero = [int(m) for m in order if g[name][m] != 0.0]
        entries[name] = nonzero[:4] + nonzero[-1:]
    worst = differences(sim, g, 1e-3 if double else 2e-2, entries)
    tolerance = 1e-7 if double else 5e-4
    print(f"{target} {precision}: leapfrog under a barostat, 200 steps, the volume "
          f"{volume / first:.4f} of the first, sigma and epsilon by pairs against central "
          f"differences of the energy, {worst:.1e} (tolerance {tolerance:.0e})")
    assert worst < tolerance, worst
    again = sim.tunables.gradient()
    assert all(np.array_equal(again[name], g[name]) for name in g.keys())
    sim.run(10)


def run_checkpoint(target, precision, work):
    import pathlib
    path = str(pathlib.Path(work) / "gradient.h5")
    system, state = dipeptide(term_tunables())
    sim = simulation(compile_(system, state, target, precision))
    sim.run(6)
    sim.tunables["soft_a"] = np.array([2.5])
    sim.run(4)
    g = sim.tunables.gradient()
    sim.save_checkpoint(path)
    for stage in (False, True):
        system, state = dipeptide(term_tunables())
        other = mdir.Simulation(compile_(system, state, target, precision), checkpoint=path,
                                stage=stage)
        other.part_seconds = 1e9
        if stage:
            # A stage takes the values of its own compile.
            other.tunables["soft_a"] = np.array([2.5])
        h = other.tunables.gradient()
        same = h.step == g.step and h.energy == g.energy and all(
            np.array_equal(h[name], g[name]) for name in g.keys())
        print(f"{target} {precision}: {'a new stage from' if stage else 'continued from'} a "
              f"checkpoint, the derivative and its energy equal those of the simulation that "
              f"wrote it to the bit: {same}")
        assert same, (dict(h.items()), dict(g.items()), h.energy, g.energy)
        other.run(4)
        assert other.tunables.gradient().step == g.step + 4


def charge_model(pme, net=0.0):
    """The dipeptide in water with the charges tunable, one entry per atom
    name of the waters and one per atom of the peptide, each entry of the
    peptide moved by `net` / 22 so that the system has the net charge
    `net`."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME if pme else mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(top.atom_names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    charge_map = np.array(charge_map, dtype=np.int64)
    values = np.zeros(len(entries))
    values[charge_map] = top.charges
    peptide = sorted({m for m, r in zip(charge_map, residues) if r != "WAT"})
    values[peptide] += net / len(peptide)
    system.tunables = [mdir.Tunable("q", "charge", map=charge_map, values=values)]
    system.tunable_gradient = True
    return system, state, charge_map


def run_pme(target, precision):
    double = precision == "Double"
    system, state, charge_map = charge_model(pme=True, net=0.5)
    sim = simulation(compile_(system, state, target, precision))
    sim.run(6)
    g = sim.tunables.gradient()
    net = float(sim.tunables["q"][charge_map].sum())
    assert abs(net - 0.5) < 1e-9, net
    worst = differences(sim, g, 1e-3 if double else 5e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{target} {precision}: the charges with particle mesh Ewald, {len(g['q'])} entries "
          f"for {len(charge_map)} particles, a net charge of {net:.1f} e, against central "
          f"differences of the energy, {worst:.1e} (tolerance {tolerance:.0e})")
    assert worst < tolerance, worst
    again = sim.tunables.gradient()
    assert np.array_equal(again["q"], g["q"]) or not double

    # A triclinic cell (a rhombic dodecahedron of 403 waters), which takes
    # the kernels of its own: one entry for the oxygens, one for the
    # hydrogens, and the 9 particles of three waters each their own.
    triclinic = root + "/../triclinic"
    loaded = mdir.load_gromacs(triclinic + "/water.top", triclinic + "/dodecahedron.gro",
                               defines=["FLEXIBLE"])
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28] * 3
    names = system.topology.atom_names
    water_map = np.array([i if i < 9 else 9 + (name != "OW") for i, name in enumerate(names)])
    system.tunables = [mdir.Tunable("q", "charge", map=water_map)]
    system.tunable_gradient = True
    sim = simulation(compile_(system, state, target, precision))
    assert np.any(sim.state().cell.tilt != 0.0)
    sim.run(6)
    g = sim.tunables.gradient()
    worst = differences(sim, g, 1e-3 if double else 5e-2)
    print(f"{target} {precision}: in a triclinic cell, {len(g['q'])} entries for {len(names)} "
          f"particles, against central differences of the energy, {worst:.1e} "
          f"(tolerance {tolerance:.0e})")
    assert worst < tolerance, worst


def run_torchpme(target, precision):
    import torch
    import torchpme
    torch.set_default_dtype(torch.float64)
    double = precision == "Double"
    rc, f, beta = 0.8, 138.935457644, 3.5

    def simulate(points):
        loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
        system, state = loaded.make_system(), loaded.make_state()
        # The defaults of before D[python-defaults], with which this was written.
        system.truncation = mdir.Truncation.Switch
        system.cutoff, system.pairlist_distance, system.switch_distance = rc, 0.9, 0.7
        system.electrostatics = mdir.Electrostatics.PME
        system.pme_alpha, system.pme_grid = beta, [points, points, points]
        system.dispersion = mdir.DispersionCorrection.None_
        system.tunables = [mdir.Tunable("q", "charge")]
        system.tunable_gradient = True
        sim = simulation(compile_(system, state, target, precision))
        return system, sim

    system, sim = simulate(64)
    top = system.topology
    g = sim.tunables.gradient()
    at = sim.state()
    x, box = at.positions, np.diag(at.cell.vectors)
    n = len(x)
    q0 = sim.tunables["q"].copy()
    assert g["q"].shape == (n,)

    # The pairs: up to two bonds apart excluded, three bonds apart the 1-4
    # pairs (f q_i q_j / (1.2 r), ff14SB), excluded from the others as well.
    graph = {i: set() for i in range(n)}
    for i, j in top.bonds:
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    excluded, scaled = set(), set()
    for i in range(n):
        seen, frontier = {i}, {i}
        for depth in range(3):
            frontier = {k for m in frontier for k in graph[m]} - seen
            seen |= frontier
            for j in frontier:
                if j > i:
                    (scaled if depth == 2 else excluded).add((i, j))
    near, distances = [], []
    for i in range(n - 1):
        d = x[i + 1:] - x[i]
        d -= box * np.round(d / box)
        r = np.sqrt((d * d).sum(axis=1))
        for j in np.nonzero(r < rc)[0]:
            pair = (i, i + 1 + int(j))
            if pair not in excluded and pair not in scaled:
                near.append(pair)
                distances.append(r[j])

    def separation(pairs):
        pairs = np.array(sorted(pairs))
        d = x[pairs[:, 1]] - x[pairs[:, 0]]
        d -= box * np.round(d / box)
        return torch.tensor(pairs), torch.tensor(np.sqrt((d * d).sum(axis=1)))

    near_pairs, near_r = torch.tensor(np.array(near)), torch.tensor(np.array(distances))
    held_pairs, held_r = separation(excluded | scaled)
    scaled_pairs, scaled_r = separation(scaled)
    positions, cell = torch.tensor(x), torch.tensor(np.diag(box))

    def reference(charges, spacing):
        """The Coulomb energy of MDIR's model from torch-pme: its Ewald sum
        with the pairs within the cutoff that are not excluded (the direct
        sum, the sum of the mesh, the self term, the background), less the
        shift of the direct sum to 0 at the cutoff (D210), less
        f q_i q_j erf(beta r) / r of the excluded pairs, which the mesh
        holds, plus the 1-4 pairs."""
        potential = torchpme.CoulombPotential(smearing=1.0 / (np.sqrt(2.0) * beta), prefactor=f)
        calculator = torchpme.PMECalculator(potential, mesh_spacing=spacing, interpolation_nodes=5)
        column = charges.unsqueeze(1)
        energy = (column * calculator(column, cell, positions, near_pairs, near_r)).sum()
        product = lambda pairs: charges[pairs[:, 0]] * charges[pairs[:, 1]]
        shift = f * torch.erfc(torch.tensor(beta * rc)) / rc * product(near_pairs).sum()
        held = (f * product(held_pairs) * torch.erf(beta * held_r) / held_r).sum()
        fourteen = (f * product(scaled_pairs) / (1.2 * scaled_r)).sum()
        return energy - shift - held + fourteen

    def torch_gradient(spacing):
        charges = torch.tensor(q0, requires_grad=True)
        energy = reference(charges, spacing)
        energy.backward()
        return charges.grad.numpy().copy()

    # 64 points along each edge in both (torch-pme takes the power of 2 at
    # or above 2 L / spacing + 1), then 128 in both.
    length = float(box.max())
    spacings = {64: 2.0 * length / 62.5, 128: 2.0 * length / 126.5}
    for points, spacing in spacings.items():
        mesh = torchpme.lib.get_ns_mesh(cell, spacing)
        assert [int(k) for k in mesh] == [points] * 3, mesh
    coarse, fine = torch_gradient(spacings[64]), torch_gradient(spacings[128])
    _, finer = simulate(128)
    refined = finer.tunables.gradient()["q"]
    scale = np.abs(coarse).max()
    own = np.abs(g["q"] - refined).max() / scale
    theirs = np.abs(coarse - fine).max() / scale
    difference = np.abs(g["q"] - coarse).max() / scale
    # In mixed precision the kernels of the pairs add their own.
    tolerance = 3.0 * (own + theirs) + (0.0 if double else 2e-6)
    print(f"{target} {precision}: dU/dq of {n} particles against torch-pme, grid 64: "
          f"{difference:.1e} of the largest ({scale:.1f} kJ/mol/e); with 128 points MDIR "
          f"changes by {own:.1e} and torch-pme by {theirs:.1e}; tolerance {tolerance:.1e}")
    assert difference < tolerance, (difference, tolerance)
    difference = np.abs(refined - fine).max() / scale
    print(f"{target} {precision}: the same at 128 points: {difference:.1e}")
    assert difference < tolerance, (difference, tolerance)

    # The change of the energy for other charges, at the same positions.
    rng = np.random.default_rng(11)
    q1 = q0 * (1.0 + 0.05 * rng.standard_normal(n))
    sim.tunables["q"] = q1
    change = sim.tunables.gradient().energy - g.energy
    with torch.no_grad():
        expected = float(reference(torch.tensor(q1), spacings[64]) -
                         reference(torch.tensor(q0), spacings[64]))
    print(f"{target} {precision}: the change of the energy for other charges, {change:.4f} "
          f"against torch-pme {expected:.4f} kJ/mol, difference {abs(change - expected):.1e}")
    assert abs(change - expected) < (2e-3 if double else 2e-2) + 3.0 * (own + theirs) * abs(expected)


def run_lj_dipeptide(target, precision):
    double = precision == "Double"
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    types, pairs = len(top.type_names), top.type_pairs
    # The hydrogen of the water has no Lennard-Jones: sqrt(x y) has no
    # derivative at x = 0, which the compile says.
    system.tunables = [mdir.Tunable("epsilon", "epsilon")]
    system.tunable_gradient = True
    text = expect(mdir.InputError, lambda: compile_(system, state), "has no derivative")
    assert "give the type -1 in its map" in text, text
    system.tunable_gradient = False
    system.tunables = [mdir.Tunable("epsilon", "epsilon")]
    epsilon = simulation(compile_(system, state)).tunables["epsilon"]
    keep = np.where(epsilon > 0.0, np.cumsum(epsilon > 0.0) - 1, -1)
    # Three pairs of the table, one of a type with itself.
    taken = np.full(len(pairs), -1)
    taken[0], taken[3], taken[types + 2] = 1, 0, 2
    system.tunables = [mdir.Tunable("sigma", "sigma", map=keep),
                       mdir.Tunable("epsilon", "epsilon", map=keep),
                       mdir.Tunable("sigma_pair", "sigma_pair", map=taken),
                       mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken)]
    system.tunable_gradient = True
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simulation(compile_(system, state, target, precision))
    sim.run(6)
    g = sim.tunables.gradient()
    assert g["sigma"].shape == (int(keep.max()) + 1,) and g["sigma_pair"].shape == (3,)
    worst = differences(sim, g, 1e-3 if double else 2e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{target} {precision}: the dipeptide, per-type and pair sigma and epsilon against "
          f"central differences of the energy, {worst:.1e} (tolerance {tolerance:.0e})")
    assert worst < tolerance, worst


def run_lj_propane(target, precision):
    double = precision == "Double"
    rc = 0.8
    gromacs = root + "/../gromacs"

    def model(tunables, gradient=True):
        loaded = mdir.load_gromacs(gromacs + "/system.top", gromacs + "/system.gro",
                                   defines=["FLEXIBLE"])
        system, state = loaded.make_system(), loaded.make_state()
        system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
        system.truncation = mdir.Truncation.None_
        system.dispersion = mdir.DispersionCorrection.EnergyPressure
        system.tunables = tunables(system)
        system.tunable_gradient = gradient
        return system, state

    # The table of the model, by pairs of types.
    system, state = model(lambda s: [mdir.Tunable("s", "sigma_pair"),
                                     mdir.Tunable("e", "epsilon_pair")], gradient=False)
    top = system.topology
    names, pairs, types = list(top.type_names), top.type_pairs, top.particle_types
    count = len(names)
    sim = simulation(compile_(system, state))
    sigma0, epsilon0 = sim.tunables["s"].copy(), sim.tunables["e"].copy()
    site = {(a, b): k for k, (a, b) in enumerate(pairs)}
    diagonal = np.array([site[a, a] for a in range(count)])
    ct, ow, hc = names.index("CT"), names.index("OW"), names.index("HC")
    nbfix = site[min(ct, ow), max(ct, ow)]
    keep = np.where(epsilon0[diagonal] > 0.0, np.cumsum(epsilon0[diagonal] > 0.0) - 1, -1)
    taken_sigma = np.full(len(pairs), -1)
    taken_sigma[nbfix], taken_sigma[site[ow, ow]] = 0, 1
    taken_epsilon = np.full(len(pairs), -1)
    taken_epsilon[nbfix], taken_epsilon[site[min(ct, hc), max(ct, hc)]] = 0, 1

    def tunables(system):
        return [mdir.Tunable("sigma", "sigma", map=keep, mixing="geometric"),
                mdir.Tunable("epsilon", "epsilon", map=keep),
                mdir.Tunable("sigma_pair", "sigma_pair", map=taken_sigma),
                mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken_epsilon)]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, state = model(tunables)
        sim = simulation(compile_(system, state, target, precision))
    sim.run(4)
    g = sim.tunables.gradient()
    at = sim.state()
    x, box = at.positions, np.diag(at.cell.vectors)
    n, volume = len(x), float(np.prod(np.diag(at.cell.vectors)))
    excluded = propane_excluded(n)

    # NumPy. The sums over the pairs of each pair of types that the
    # Lennard-Jones of the table takes: r^-12 - rc^-12 and r^-6 - rc^-6 over
    # the pairs not excluded within the cutoff (the 1-4 pairs, excluded,
    # keep parameters of their own), and the number of ordered pairs not
    # excluded for the correction.
    s12, s6 = np.zeros(len(pairs)), np.zeros(len(pairs))
    ordered = np.zeros(len(pairs))
    numbers = np.bincount(types, minlength=count).astype(float)
    for (a, b), k in site.items():
        ordered[k] = numbers[a] * (numbers[b] - 1.0) if a == b else 2.0 * numbers[a] * numbers[b]
    for i, j in excluded:
        a, b = sorted((types[i], types[j]))
        ordered[site[a, b]] -= 2.0
    for i in range(n - 1):
        d = x[i + 1:] - x[i]
        d -= box * np.round(d / box)
        r = np.sqrt((d * d).sum(axis=1))
        for j in np.nonzero(r < rc)[0]:
            if (i, i + 1 + j) in excluded:
                continue
            a, b = sorted((types[i], types[i + 1 + j]))
            s12[site[a, b]] += r[j] ** -12 - rc ** -12
            s6[site[a, b]] += r[j] ** -6 - rc ** -6
    total = n * (n - 1.0) - 2.0 * len(excluded)

    def energy(theta):
        # The table: the rule from the values of the types, the pair set
        # apart from it (CT-OW) keeping its values, then the pairs that the
        # pair tunables take.
        sigma_type, epsilon_type = sigma0[diagonal].copy(), epsilon0[diagonal].copy()
        sigma_type[keep >= 0] = theta["sigma"][keep[keep >= 0]]
        epsilon_type[keep >= 0] = theta["epsilon"][keep[keep >= 0]]
        sigma, epsilon = sigma0.copy(), epsilon0.copy()
        for (a, b), k in site.items():
            if k != nbfix:
                sigma[k] = np.sqrt(sigma_type[a] * sigma_type[b])
                epsilon[k] = np.sqrt(epsilon_type[a] * epsilon_type[b])
        sigma[taken_sigma >= 0] = theta["sigma_pair"][taken_sigma[taken_sigma >= 0]]
        epsilon[taken_epsilon >= 0] = theta["epsilon_pair"][taken_epsilon[taken_epsilon >= 0]]
        within = (4.0 * epsilon * (sigma ** 12 * s12 - sigma ** 6 * s6)).sum()
        # -(2 pi / 3 V) N^2 <C6> / rc^3, and its shift estimate, the
        # correction times 1 - V / (N (4 pi / 3) rc^3) (D210).
        mean = (ordered * 4.0 * epsilon * sigma ** 6).sum() / total
        correction = -2.0 * np.pi / (3.0 * volume) * n * n * mean / rc ** 3
        return within + correction * (2.0 - volume / (n * 4.0 * np.pi / 3.0 * rc ** 3))

    theta0 = {name: sim.tunables[name].copy() for name in g.keys()}
    worst = 0.0
    for name in g.keys():
        scale = np.abs(g[name]).max()
        for m in range(len(theta0[name])):
            def central(h):
                e = []
                for sign in (1.0, -1.0):
                    theta = {k: v.copy() for k, v in theta0.items()}
                    theta[name][m] += sign * h
                    e.append(energy(theta))
                return (e[0] - e[1]) / (2.0 * h)
            h = 1e-3 * theta0[name][m]
            reference = (4.0 * central(0.5 * h) - central(h)) / 3.0
            worst = max(worst, abs(g[name][m] - reference) / scale)
    tolerance = 1e-8 if double else 2e-5
    print(f"{target} {precision}: propane and water, per-type and pair sigma and epsilon "
          f"with a pair set apart, the shifted Lennard-Jones with the correction and its "
          f"shift estimate against NumPy, {worst:.1e} of the largest entry of each tunable "
          f"(tolerance {tolerance:.0e})")
    assert worst < tolerance, worst
    worst = differences(sim, g, 1e-3 if double else 2e-2)
    tolerance = 1e-7 if double else 5e-4
    print(f"{target} {precision}: propane and water against central differences of the "
          f"energy, {worst:.1e} (tolerance {tolerance:.0e})")
    assert worst < tolerance, worst


def propane(gradient=True, dispersion=True):
    """Propane and water (Inputs/gromacs, 224 atoms), the Lennard-Jones cut
    without a shift, a Coulomb cutoff, and a pair term -c/r^8 whose
    constant is tunable."""
    gromacs = root + "/../gromacs"
    loaded = mdir.load_gromacs(gromacs + "/system.top", gromacs + "/system.gro", defines=["FLEXIBLE"])
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.dispersion = (mdir.DispersionCorrection.EnergyPressure if dispersion
                         else mdir.DispersionCorrection.None_)
    term = mdir.PairTerm()
    term.name, term.expression = "eighth", "-c/r^8"
    term.constants = [("c", 2.0e-6)]
    system.pair_terms = [term]
    system.tunables = [mdir.Tunable("c", "c", term="eighth")]
    system.tunable_gradient = gradient
    return system, state


def propane_excluded(n):
    """The pairs of particles of the GROMACS model that its topology
    excludes: up to three bonds apart in propane (1-4 pairs included, which
    have parameters of their own), and within a water."""
    top = open(root + "/../gromacs/system.top").read()
    bonds, counts = [], {}
    section = None
    for line in top.splitlines():
        line = line.split(";")[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section == "bonds" and line:
            bonds.append(tuple(int(x) - 1 for x in line.split()[:2]))
        if section == "molecules" and line:
            counts[line.split()[0]] = int(line.split()[1])
    n_pro = 11
    graph = {i: set() for i in range(n_pro)}
    for i, j in bonds:
        graph[i].add(j); graph[j].add(i)
    excluded = set()
    for m in range(counts["PRO"]):
        o = n_pro * m
        for i in range(n_pro):
            seen, frontier = {i}, {i}
            for _ in range(3):
                frontier = {k for f in frontier for k in graph[f]} - seen
                seen |= frontier
            excluded |= {(o + min(i, j), o + max(i, j)) for j in seen if j != i}
    for w in range(counts["SOL"]):
        o = n_pro * counts["PRO"] + 3 * w
        excluded |= {(o, o + 1), (o, o + 2), (o + 1, o + 2)}
    assert n_pro * counts["PRO"] + 3 * counts["SOL"] == n
    return excluded


def run_tails(target, precision):
    double = precision == "Double"
    rc = 0.8
    system, state = propane()
    sim = simulation(compile_(system, state, target, precision))
    sim.run(4)
    g = sim.tunables.gradient()
    at = sim.state()
    x = at.positions
    n = len(x)
    box = np.diag(at.cell.vectors)
    volume = float(np.prod(box))
    excluded = propane_excluded(n)

    # NumPy: dU/dc of the shifted pair term, -(r^-8 - rc^-8) over the pairs
    # not excluded within the cutoff; of its tail, nu (4 pi / V) P I' with
    # I' = -1 / (5 rc^5) for each of the P unordered pairs not excluded and
    # nu = N^2 / (N (N - 1) - 2 N_excluded) (D209); and of the estimate of
    # the shift, nu (4 pi rc^3 / (3 V) - 1 / N) P (-rc^-8) (D210).
    inside = 0.0
    for i in range(n - 1):
        d = x[i + 1:] - x[i]
        d -= box * np.round(d / box)
        r = np.sqrt((d * d).sum(axis=1))
        keep = (r < rc) & np.array([(i, j) not in excluded for j in range(i + 1, n)])
        inside += -(r[keep] ** -8 - rc ** -8).sum()
    pairs = 0.5 * (n * (n - 1.0) - 2.0 * len(excluded))
    nu = n * n / (n * (n - 1.0) - 2.0 * len(excluded))
    tail = nu * 4.0 * np.pi / volume * pairs * (-1.0 / (5.0 * rc ** 5))
    shift = nu * (4.0 * np.pi * rc ** 3 / (3.0 * volume) - 1.0 / n) * pairs * (-rc ** -8)
    reference = inside + tail + shift
    error = abs(g["c"][0] - reference) / abs(reference)
    # In mixed precision the kernel takes r^-8, steep at the closest pairs,
    # from its table of the squared distance in f32 (D94).
    tolerance = 1e-10 if double else 1e-4
    print(f"{target} {precision}: dU/dc {g['c'][0]:.6e} against NumPy {reference:.6e} "
          f"(within the cutoff {inside:.6e}, tail {tail:.6e}, shift estimate {shift:.6e}), "
          f"relative difference {error:.1e} (tolerance {tolerance:.0e})")
    assert error < tolerance, (g["c"][0], reference)

    # The energy is linear in c: the derivative times c is the energy of the
    # term, and a central difference of any step is exact up to rounding.
    worst = differences(sim, g, 1e-2 if double else 0.5)
    tolerance = 1e-9 if double else 1e-4
    print(f"{target} {precision}: against central differences of the energy, {worst:.1e} "
          f"(tolerance {tolerance:.0e})")
    assert worst < tolerance, worst

    # Without the correction for the dispersion only the pairs within the
    # cutoff remain.
    system, state = propane(dispersion=False)
    sim = simulation(compile_(system, state, target, precision))
    sim.run(4)
    g = sim.tunables.gradient()
    error = abs(g["c"][0] - inside) / abs(inside)
    print(f"{target} {precision}: without the correction, relative difference {error:.1e}")
    assert error < tolerance, (g["c"][0], inside)


if scenario == "refusals":
    run_refusals()
elif scenario == "terms":
    run_terms(sys.argv[3], sys.argv[4])
elif scenario == "tails":
    run_tails(sys.argv[3], sys.argv[4])
elif scenario == "dynamics":
    run_dynamics(sys.argv[3], sys.argv[4])
elif scenario == "checkpoint":
    run_checkpoint(sys.argv[3], sys.argv[4], sys.argv[5])
elif scenario == "pme":
    run_pme(sys.argv[3], sys.argv[4])
elif scenario == "torchpme":
    run_torchpme(sys.argv[3], sys.argv[4])
elif scenario == "charges":
    run_charges(sys.argv[3], sys.argv[4])
elif scenario == "lj":
    run_lj_dipeptide(sys.argv[3], sys.argv[4])
    run_lj_propane(sys.argv[3], sys.argv[4])
else:
    raise SystemExit(f"unknown scenario {scenario}")
print(f"{scenario} passed")
