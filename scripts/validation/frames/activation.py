"""The frame evaluator of one build against that of another on the same
stored frames (D251, docs/python-frames.md): the oracle of
the frames inside one activation is the evaluator that starts an activation
per frame.

  activation.py sample ROOT FRAMES.npz
      runs the samplers on the CPU in double precision, in the
      deterministic mode, and keeps their frames, in float64, with the cell
      of each.
  activation.py evaluate ROOT FRAMES.npz OUT.npz TARGET PRECISION
                [--default-mode] [--input-order]
      evaluates the frames with the build that is imported and keeps what
      the evaluator gives.
  activation.py compare A.npz B.npz
      prints, for each quantity, the largest difference of the two builds,
      absolute and relative to the largest value, and whether every bit is
      equal.

ROOT is test/Driver/Inputs. The programs:

  dipeptide   the alanine dipeptide in water (1,168 atoms, PME), 25 tied
              charges and sigma and epsilon of the pair OW-OW tunable, an
              observed pair term and an observed wall; 8 frames.
  dependent   the same with terms="dependent".
  npt         the same system under a barostat with leapfrog, a plain
              cutoff, the correction for the dispersion; the 45 pairs of
              types and the charges tunable; 6 frames, each with its cell.
  triclinic   403 flexible waters in a rhombic dodecahedron, 11 tied
              charges and sigma of OW-OW tunable; 6 frames of a run at
              constant pressure, each with its diagonal and its tilts.

Each set is evaluated in one call; one frame per call (K = 1); in the
reverse order; in an order that repeats frames and jumps between distant
ones (0, 0, 0, last, 0, last, last, 1), where a neighbor structure that
were kept would be kept and then be wrong; and again after a call in which
a frame fails (every particle at one place), which must raise and leave
the next call what it was. The evaluations are in the deterministic mode
unless `--default-mode`. With `--input-order` the programs are compiled
without the spatial order of the particles (`Execution.reorder = False`).
"""
import sys

import numpy as np
import mdir

SETS = ("dipeptide", "dependent", "npt", "triclinic")


def execution_(target, precision, deterministic, input_order):
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    if input_order:
        execution.reorder = False
    return execution


def tied_charges(system):
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


def program_(root, name, execution, sampler):
    """The program of the set `name`; with `sampler`, the one that the
    frames are sampled with."""
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep = 0.0005
    if name == "triclinic":
        directory = root + "/triclinic"
        loaded = mdir.load_gromacs(directory + "/water.top", directory + "/dodecahedron.gro",
                                   defines=["FLEXIBLE"])
        system, state = loaded.make_system(), loaded.make_state()
        system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
        system.truncation = mdir.Truncation.None_
        system.dispersion = mdir.DispersionCorrection.None_
        system.electrostatics = mdir.Electrostatics.PME
        system.pme_grid = [28] * 3
        names = system.topology.atom_names
        water_map = np.array([i if i < 9 else 9 + (n != "OW") for i, n in enumerate(names)])
        system.tunables = [mdir.Tunable("q", "charge", map=water_map),
                           mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW"))]
        if sampler:
            state = state.draw_velocities(system, 300.0, 2024)
            ensemble.kind, ensemble.temperature, ensemble.seed = mdir.EnsembleKind.NPT, 300.0, 2024
            ensemble.com_period = ensemble.coupling_period = 10
    else:
        directory = root + "/dipeptide"
        loaded = mdir.load_amber(directory + "/dipeptide.prmtop", directory + "/dipeptide.inpcrd")
        system, state = loaded.make_system(), loaded.make_state()
        system.electrostatics = mdir.Electrostatics.PME
        if name == "npt":
            system.cutoff, system.pairlist_distance = 0.8, 0.9
            system.truncation = mdir.Truncation.None_
            system.dispersion = mdir.DispersionCorrection.EnergyPressure
            system.tunables = [mdir.Tunable("sigma", "sigma_pair"),
                               mdir.Tunable("epsilon", "epsilon_pair"),
                               mdir.Tunable("q", "charge", map=tied_charges(system))]
            integrator.method = mdir.IntegratorMethod.Leapfrog
            ensemble.kind, ensemble.temperature = mdir.EnsembleKind.NPT, 300.0
            ensemble.coupling_period = 10
        else:
            system.truncation = mdir.Truncation.Switch
            system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
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
                               mdir.Tunable("sigma", "sigma_pair",
                                            map=one_pair(system, "OW", "OW")),
                               mdir.Tunable("epsilon", "epsilon_pair",
                                            map=one_pair(system, "OW", "OW")),
                               mdir.Tunable("l", "l", term="soft")]
    system.tunable_gradient = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def sample(root, path):
    kept = {}
    for name, count, stride in (("dipeptide", 8, 20), ("npt", 6, 50), ("triclinic", 6, 50)):
        sim = mdir.Simulation(program_(root, name, execution_("CPU", "Double", True, False), True))
        positions, diagonals, tilts = [], [], []
        for _ in range(count):
            sim.run(stride)
            state = sim.state()
            positions.append(np.array(state.positions))
            diagonals.append(np.array(state.cell.diagonal))
            tilts.append(np.array(state.cell.tilt))
        kept[name + ".positions"] = np.stack(positions)
        kept[name + ".diagonals"] = np.stack(diagonals)
        kept[name + ".tilts"] = np.stack(tilts)
        print(f"{name}: {count} frames, volumes {np.prod(diagonals, axis=1).min():.4f} to "
              f"{np.prod(diagonals, axis=1).max():.4f} nm^3, tilts moved by "
              f"{np.ptp(np.stack(tilts), axis=0).max():.1e} nm")
    np.savez(path, **kept)


def evaluate(root, frames, path, target, precision, deterministic, input_order):
    frames, kept = np.load(frames), {}
    execution = execution_(target, precision, deterministic, input_order)
    for name in SETS:
        source = "dipeptide" if name == "dependent" else name
        x, d, t = (frames[f"{source}.{k}"] for k in ("positions", "diagonals", "tilts"))
        tilts = t if name == "triclinic" else None
        evaluator = mdir.FrameEvaluator(program_(root, source, execution, False),
                                        terms="dependent" if name == "dependent" else "all")
        energy = "dependent_energy" if name == "dependent" else "energy"

        def take(out, key, order=None):
            columns = {"energy": getattr(out, energy), "volume": out.volume}
            if name != "dependent":
                columns["virial"] = out.virial
            for column, values in out.observables.items():
                columns["observable " + column] = values
            for tunable, rows in out.jacobian.items():
                columns["row " + tunable] = rows
            for column, values in columns.items():
                if order is not None:
                    restored = np.empty_like(values)
                    restored[order] = values
                    values = restored
                kept[f"{name}|{key}|{column}"] = values

        def call(order):
            return evaluator.evaluate(x[order], d[order], None if tilts is None else tilts[order])

        count = len(x)
        everything = np.arange(count)
        out = call(everything)
        take(out, "one call")
        cotangent = np.random.default_rng(7).normal(size=count)
        for tunable, product in out.vjp(cotangent).items():
            kept[f"{name}|one call|vjp {tunable}"] = product
        singles = [call(np.array([n])) for n in range(count)]
        for column in [c for c in kept if c.startswith(f"{name}|one call|")
                       and "|vjp " not in c]:
            what = column.split("|")[2]
            first = {"energy": lambda o: getattr(o, energy), "volume": lambda o: o.volume,
                     "virial": lambda o: o.virial}
            if what in first:
                values = np.concatenate([first[what](o) for o in singles])
            elif what.startswith("observable "):
                values = np.concatenate([o.observables[what[11:]] for o in singles])
            else:
                values = np.concatenate([o.jacobian[what[4:]] for o in singles])
            kept[f"{name}|K = 1|{what}"] = values
        take(call(everything[::-1]), "reversed", everything[::-1])
        last = count - 1
        jumps = np.array([0, 0, 0, last, 0, last, last, 1])
        hops = call(jumps)
        for at, key in ((2, "frame 0 after itself twice"), (4, "frame 0 after the last"),
                        (6, "the last after itself"), (7, "frame 1 after the last")):
            kept[f"{name}|{key}|energy"] = getattr(hops, energy)[at:at + 1]
            for tunable, rows in hops.jacobian.items():
                kept[f"{name}|{key}|row {tunable}"] = rows[at]
        # A frame that fails in the middle of a call, and the next call.
        collapsed = x.copy()
        collapsed[count // 2] = 1.0
        try:
            evaluator.evaluate(collapsed, d, tilts)
        except mdir.SimulationError as error:
            assert f"frame {count // 2}: " in str(error), str(error)
        else:
            raise AssertionError("a frame with every particle at one place was evaluated")
        take(call(everything), "after a failed frame")
        print(f"{name}: {count} frames evaluated in five ways")
    np.savez(path, **kept)


def compare(first, second):
    a, b = np.load(first), np.load(second)
    assert sorted(a.files) == sorted(b.files), "the two files hold other quantities"
    worst = {}
    for key in a.files:
        name, how, what = key.split("|")
        group = "row" if what.startswith("row ") else "vjp" if what.startswith("vjp ") else \
            "observable" if what.startswith("observable ") else what
        x, y = a[key], b[key]
        absolute = float(np.abs(x - y).max())
        relative = absolute / max(float(np.abs(y).max()), 1e-300)
        entry = worst.setdefault((name, how, group), [0.0, 0.0, True, 0.0])
        entry[0], entry[1] = max(entry[0], absolute), max(entry[1], relative)
        entry[2] = entry[2] and np.array_equal(x, y)
        entry[3] = max(entry[3], float(np.abs(y).max()))
    print("set | evaluated | quantity | largest value | difference | relative | every bit equal")
    for (name, how, group), (absolute, relative, equal, size) in worst.items():
        print(f"{name} | {how} | {group} | {size:.6g} | {absolute:.1e} | {relative:.1e} | {equal}")
    summary = {}
    for (name, how, group), (absolute, relative, equal, size) in worst.items():
        entry = summary.setdefault(group, [0.0, 0.0, True])
        entry[0], entry[1] = max(entry[0], absolute), max(entry[1], relative)
        entry[2] = entry[2] and equal
    for group, (absolute, relative, equal) in summary.items():
        print(f"all sets, {group}: {absolute:.1e} absolute, {relative:.1e} relative, "
              f"every bit equal: {equal}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sample":
        sample(*sys.argv[2:4])
    elif mode == "evaluate":
        options = sys.argv[7:]
        evaluate(*sys.argv[2:7], "--default-mode" not in options, "--input-order" in options)
    elif mode == "compare":
        compare(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)
