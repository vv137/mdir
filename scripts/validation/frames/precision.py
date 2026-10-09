"""Which stored precision serves which reweighting (#264, docs/python-frames.md).

Frames of one run are kept in f64, rounded to f32, and rounded to 1e-3 nm
(XTC at its default precision), and the same reweighting is made from each
with the frame evaluator: the energies U(theta) and U(theta-hat) are both
evaluated at the stored frame, so that what the rounding adds to the fixed
terms cancels in their difference. For each tunable and each rounding the
script prints

  the rise of the energy of a frame, U(rounded) - U(f64);
  the error of the difference dU = U(theta) - U(theta-hat) against f64;
  the number of effective frames, and the largest change of a weight;
  the error of the reweighted mean of dU/dtheta (the derivative of the free
  energy in theta), which needs no observable;
  the error of the derivative of a reweighted average, -Cov_w(O, dU/dtheta)/kT,
  beside its statistical error from blocks of the frames;
  and the same reweighting with the reference energy taken from what the
  sampler recorded at the frame it visited, in place of an evaluation at the
  stored frame.

Systems: the alanine dipeptide in 382 flexible waters with PME, with (a) a
constant of a soft pair term, (b) the charges of the water, (c) the force
constant and the length of a harmonic term over the O-H bonds of the water
(a stiff term) tunable; and 60 A + 60 B Lennard-Jones particles with the
sigma of the A-B pair term tunable. theta is theta-hat moved by kT/2 over
the standard deviation of dU/dtheta, so that the weights spread.

Usage: precision.py DIPEPTIDE-DIRECTORY MIXTURE-DIRECTORY [TARGET [FRAMES]]

DIPEPTIDE-DIRECTORY is test/Driver/Inputs/dipeptide; MIXTURE-DIRECTORY is
filled by test/Driver/Inputs/pair_tail_system.py DIR 60 60 32 3.7 0.5.
"""
import sys

import numpy as np
import mdir

root, mixture = sys.argv[1:3]
target = sys.argv[3] if len(sys.argv) > 3 else "CPU"
FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 200

ROUNDINGS = (("f64", lambda x: x),
             ("f32", lambda x: x.astype(np.float32).astype(np.float64)),
             ("1e-3 nm", lambda x: np.round(x * 1000.0) / 1000.0))


def execution_():
    execution = mdir.Execution()
    execution.target, execution.precision = getattr(mdir.Target, target), mdir.Precision.Double
    execution.deterministic = True
    return execution


def water_bonds(top):
    names, residues = top.atom_names, [top.residue_names[r] for r in top.residue_indices]
    pairs = [(int(i), int(j)) for i, j in np.array(top.bonds)[:, :2]
             if residues[i] == "WAT" and "O" in (names[i][0], names[j][0])]
    return np.array(pairs, dtype=np.int64)


def dipeptide(tunable):
    """One potential, three declarations: the soft pair term and the added
    term over the O-H bonds are in every program, so the frames of one run
    serve the three."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    bonds = water_bonds(top)
    stiff = mdir.TupleTerm()
    stiff.name, stiff.expression, stiff.arity = "oh", "0.5*k*(r - r0)^2", 2
    stiff.particles = bonds
    stiff.parameters = [("k", np.full(len(bonds), 5.0e4)), ("r0", np.full(len(bonds), 0.09572))]
    system.pair_terms, system.tuple_terms = [soft], [stiff]
    every = np.zeros(len(bonds), dtype=np.int64)
    if tunable == "soft":
        system.tunables = [mdir.Tunable("l", "l", term="soft")]
    elif tunable == "charges":
        residues = [top.residue_names[r] for r in top.residue_indices]
        entries = {}
        charge_map = np.array([entries.setdefault(atom[0], len(entries)) if res == "WAT" else -1
                               for atom, res in zip(top.atom_names, residues)])
        system.tunables = [mdir.Tunable("q", "charge", map=charge_map)]
    elif tunable == "bond k":
        system.tunables = [mdir.Tunable("k", "k", term="oh", map=every)]
    else:
        system.tunables = [mdir.Tunable("r0", "r0", term="oh", map=every)]
    system.tunable_gradient = True
    state = state.draw_velocities(system, 300.0, 20261009)
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, 20261009
    ensemble.kind, ensemble.coupling_period = mdir.EnsembleKind.NVT, 10
    program = mdir.compile(system, state, integrator, ensemble, execution_(), mdir.Schedule())
    return program, len(bonds)


def lj_mixture():
    loaded = mdir.load_gromacs(mixture + "/plain.top", mixture + "/system.gro")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 1.2, 1.3
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    term = mdir.PairTerm()
    term.name, term.groups = "nbfix", [":A", ":B"]
    term.expression = "4*eps*((sig/r)^12 - (sig/r)^6) - 4*eps0*((sig0/r)^12 - (sig0/r)^6)"
    term.constants = [("sig", 0.37), ("eps", 0.5 * 4.184), ("sig0", 0.34), ("eps0", 0.24 * 4.184)]
    system.pair_terms = [term]
    system.tunables = [mdir.Tunable("sig", "sig", term="nbfix")]
    system.tunable_gradient = True
    state = state.draw_velocities(system, 100.0, 271828)
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.002, 100.0, 271828
    ensemble.kind = mdir.EnsembleKind.NVT
    ensemble.com_period = ensemble.coupling_period = 10
    return mdir.compile(system, state, integrator, ensemble, execution_(), mdir.Schedule())


def sample(program, count, stride, warm):
    """Frames in f64 with the potential energy that the run reports at
    each, the number that a sampler records."""
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    sim.run(warm)
    positions, recorded, cell = [], [], None
    for _ in range(count):
        sim.run(stride, energy=True)
        state = sim.state()
        positions.append(np.array(state.positions))
        recorded.append(state.energies["potential"])
        cell = state.cell
    return np.stack(positions), np.array(recorded), cell


def weights(delta, kT):
    log = -delta / kT
    w = np.exp(log - log.max())
    return w / w.sum()


def effective(w):
    return float(np.exp(-(w[w > 0] * np.log(w[w > 0])).sum()))


def derivative(w, observable, g, kT):
    return -(w @ (observable * g) - (w @ observable) * (w @ g)) / kT


def block_error(observable, g, delta, kT, blocks=10):
    """The standard error of the derivative from blocks of the frames."""
    values = []
    for part in np.array_split(np.arange(len(g)), blocks):
        values.append(derivative(weights(delta[part], kT), observable[part], g[part], kT))
    return float(np.std(values, ddof=1) / np.sqrt(blocks))


def study(label, program, name, positions, recorded, observable_of, kT, edges):
    evaluator = mdir.FrameEvaluator(program)
    hat = evaluator.tunables[name].copy()
    rows, exact = [], None
    for rounding, rounded in ROUNDINGS:
        frames = rounded(positions)
        evaluator.tunables[name] = hat
        reference = evaluator.evaluate(frames)
        if exact is None:
            # theta: theta-hat moved along its first entry by kT/2 over the
            # spread of the derivative, the same for every rounding.
            spread = reference.jacobian[name][:, 0].std()
            theta = hat.copy()
            theta[0] += 0.5 * kT / spread
        evaluator.tunables[name] = theta
        moved = evaluator.evaluate(frames)
        delta = moved.energy - reference.energy
        g = moved.jacobian[name][:, 0]
        observable = observable_of(frames, edges)
        w = weights(delta, kT)
        row = {"rounding": rounding, "energy": reference.energy, "delta": delta, "w": w,
               "n_eff": effective(w), "derivative": derivative(w, observable, g, kT),
               "mean": float(w @ g),
               "error": block_error(observable, g, delta, kT)}
        # The reference taken from the run: the energy that the sampler
        # reports at the frame that it visited, less its mean difference
        # from the evaluator's energy of the f64 frames, a constant that
        # the weights do not see.
        if exact is None:
            exact = row
            offset = float((recorded - reference.energy).mean())
            row["recorded_spread"] = float((recorded - reference.energy).std())
        from_run = moved.energy - (recorded - offset)
        row["w_run"] = weights(from_run, kT)
        row["n_eff_run"] = effective(row["w_run"])
        row["derivative_run"] = derivative(row["w_run"], observable, g, kT)
        rows.append(row)
    count = len(recorded)
    step = float(theta[0] - hat[0])
    print(f"\n{label}: {count} frames, theta-hat = {hat[0]:.6g}, theta - theta-hat = {step:.3g}, "
          f"kT = {kT:.4f} kJ/mol; in f64 the reweighted mean of dU/dtheta is "
          f"{exact['mean']:.5g} and the derivative of the reweighted average "
          f"{exact['derivative']:.4g} +- {exact['error']:.2g} (blocks); the recorded energy less "
          f"the evaluator's has a spread of {exact['recorded_spread']:.2e} kJ/mol over the frames")
    print("| Frames | Rise of U, kJ/mol | Error of dU, kJ/mol (max; rms) | N_eff | Largest change "
          "of a weight, in 1/K | Error of <dU/dtheta> | Error of the derivative | The same in "
          "its statistical error | N_eff, reference from the run | Error of the derivative, "
          "reference from the run, in its statistical error |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        rise = row["energy"] - exact["energy"]
        d = row["delta"] - exact["delta"]
        change = np.abs(row["w"] - exact["w"]).max() * count
        error = abs(row["derivative"] - exact["derivative"]) / abs(exact["derivative"])
        sigma = abs(row["derivative"] - exact["derivative"]) / exact["error"]
        run = abs(row["derivative_run"] - exact["derivative"]) / exact["error"]
        mean = abs(row["mean"] - exact["mean"]) / abs(exact["mean"])
        print(f"| {row['rounding']} | {rise.mean():+.3g} +- {rise.std():.2g} | "
              f"{np.abs(d).max():.1e}; {np.sqrt((d * d).mean()):.1e} | {row['n_eff']:.2f} | "
              f"{change:.1e} | {mean:.1e} | {error:.1e} | {sigma:.1e} | {row['n_eff_run']:.2f} | "
              f"{run:.1e} |")


def end_to_end(frames, edges):
    """A function of the positions alone: the distance between the two
    methyl carbons at the ends of the dipeptide, in the minimum image."""
    d = frames[:, 1, :] - frames[:, 18, :]
    d -= edges * np.round(d / edges)
    return np.sqrt((d * d).sum(axis=1))


def below(frames, edges):
    """The number of A (the first 60 particles) below z = 1.6 nm."""
    z = frames[:, :60, 2] % edges[2]
    return (z < 1.6).sum(axis=1).astype(np.float64)


kT = mdir.KB * 300.0
program, bonds = dipeptide("soft")
positions, recorded, cell = sample(program, FRAMES, 40, 4000)
edges = np.array(cell.diagonal)
print(f"The alanine dipeptide in water, {positions.shape[1]} atoms, {bonds} O-H bonds with the "
      f"added harmonic term, {target} in double precision; the observable is the distance "
      "between the carbons of the two methyl groups")
study("(a) the length l of the soft pair term a exp(-r/l)", program, "l", positions, recorded,
      end_to_end, kT, edges)
for label, tunable, name in (("(b) the charge of the oxygens of the water, PME", "charges", "q"),
                             ("(c) the force constant k of the O-H term", "bond k", "k"),
                             ("(c) the length r0 of the O-H term", "bond r0", "r0")):
    other, _ = dipeptide(tunable)
    study(label, other, name, positions, recorded, end_to_end, kT, edges)

kT = mdir.KB * 100.0
program = lj_mixture()
positions, recorded, cell = sample(program, FRAMES, 20, 20000)
print(f"\n60 A + 60 B Lennard-Jones particles, no stiff term; the observable is the number of A "
      "below z = 1.6 nm")
study("sigma' of the A-B pair term", program, "sig", positions, recorded, below, kT,
      np.array(cell.diagonal))
