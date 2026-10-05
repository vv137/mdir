"""Check actual Python lowering, lifetime and boundary semantics against the CLI."""
import difflib
import gc
import math
import pathlib
import subprocess
import sys
import mdir
import numpy as np

root = pathlib.Path(sys.argv[1]).resolve()
target = getattr(mdir.Target, sys.argv[2])
cli = sys.argv[3]


def expect(error, call):
    try:
        call()
    except error as exc:
        assert str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def inputs(name, precision):
    if name.startswith("charmm"):
        cell = mdir.Cell()
        if name == "charmm-triclinic":
            # The 80 degree alpha cell in model_controls.py.
            cell.diagonal = np.array([3.2, 3.2, 3.2 * math.sin(math.radians(80))])
            cell.tilt = np.array([0, 0, 3.2 * math.cos(math.radians(80))])
        else:
            cell.diagonal = np.full(3, 3.2)
        loaded = mdir.load_charmm(str(root / "charmm/toy.psf"),
                                  str(root / "charmm/toy.crd"),
                                  [str(root / "charmm/toy.rtf"), str(root / "charmm/toy.prm")], cell)
    elif name in ("gromacs", "triclinic"):
        top, coords = (("triclinic/water.top", "triclinic/dodecahedron.gro")
                       if name == "triclinic" else ("gromacs/system.top", "gromacs/system.gro"))
        loaded = mdir.load_gromacs(str(root / top), str(root / coords), defines=["FLEXIBLE"])
    else:
        loaded = mdir.load_amber(str(root / "dipeptide/dipeptide.prmtop"),
                                str(root / "dipeptide/dipeptide.inpcrd"))
    system, state = loaded.make_system(), loaded.make_state()
    assert state.positions.shape == (system.particle_count, 3)
    assert loaded.sources
    # Loaded data and the two consumers own separate values.
    pos = state.positions.copy()
    pos[0, 0] += 0.125
    assert loaded.make_state().positions[0, 0] != pos[0, 0]
    assert state.positions[0, 0] != pos[0, 0]
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    if name == "constraints":
        system.rigid_hydrogen_bonds, system.rigid_water = True, True
    if name == "triclinic":
        system.electrostatics = mdir.Electrostatics.PME
        system.pme_grid = [28] * 3
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep, ensemble.temperature = 0.0005, 300
    if name == "nvt":
        ensemble.kind = mdir.EnsembleKind.NVT
    elif name == "npt":
        ensemble.kind = mdir.EnsembleKind.NPT
    if name in ("nvt", "npt"):
        ensemble.com_period = 10
    execution, schedule = mdir.Execution(), mdir.Schedule()
    execution.target, execution.precision = target, precision
    schedule.steps, schedule.energy_period = 20, 10
    return system, state, integrator, ensemble, execution, schedule


for name in ("amber", "gromacs", "constraints", "triclinic", "nvt", "npt", "charmm", "charmm-triclinic"):
    for precision in (mdir.Precision.Double, mdir.Precision.Mixed):
        args = inputs(name, precision)
        control = root / f"{name}-{precision.name.lower()}.toml"
        native = subprocess.check_output([str(pathlib.Path(cli).with_name("mdir-model-test")), str(control), "--arrays"])
        expected = np.frombuffer(native, dtype=np.float64)
        actual = np.concatenate((args[1].positions.ravel(), args[1].velocities.ravel(), args[1].cell.vectors.ravel()))
        assert actual.tobytes() == expected.tobytes(), f"{name}: native array bytes differ"
        saved_positions = args[1].positions.copy()
        args[1].positions = saved_positions
        assert args[1].positions.tobytes() == saved_positions.tobytes()
        before = {p.name for p in root.iterdir()}
        program = mdir.compile(*args)
        assert {p.name for p in root.iterdir()} == before
        cli_ir = subprocess.check_output([cli, "emit", str(control)], text=True)
        assert program.ir == cli_ir, f"{name}/{precision}: " + "".join(difflib.unified_diff(cli_ir.splitlines(True), program.ir.splitlines(True)))
        assert program.pipeline == subprocess.check_output([cli, "emit", "--stage=pipeline", str(control)], text=True).strip()
        assert "llvm.func" in program.lowered_ir
        assert program.plan["target"] == target
        assert program.plan["precision"] == precision
        assert program.plan["state_dtype"] == "float64"
        assert program.plan["force_dtype"] == ("float32" if precision == mdir.Precision.Mixed else "float64")
        assert not program.stale
        program.check_current()
        # Failed assignment and writes to detached snapshots preserve versions.
        original = args[1].positions.tobytes()
        expect(mdir.InputError, lambda: setattr(args[1], "positions", np.zeros((1, 3))))
        assert args[1].positions.tobytes() == original and not program.stale
        snapshot = args[1].positions
        expect(ValueError, lambda: snapshot.__setitem__((0, 0), 99))
        snapshot.setflags(write=True)
        snapshot[0, 0] += 1
        assert args[1].positions.tobytes() == original and not program.stale
        expect(AttributeError, lambda: setattr(program, "ir", "changed"))
        plan = program.plan
        plan["device"] = -1
        assert program.plan["device"] == 0
        if name == "amber" and precision == mdir.Precision.Double:
            # Each independently mutable dependency invalidates the result.
            for i, attr in enumerate(("cutoff", "positions", "timestep", "temperature", "device", "steps")):
                setattr(args[i], attr, getattr(args[i], attr))
                assert program.stale
                expect(mdir.StaleProgramError, program.check_current)
                program = mdir.compile(*args)
                assert not program.stale
        del args
        gc.collect()
        program.check_current()  # retained input owners outlive the caller
        del program

args = inputs("amber", mdir.Precision.Double)
system, state, integrator, ensemble, execution, schedule = args
term = mdir.TupleTerm()
term.name, term.expression = "spring", "k*(r-r0)^2"
term.particles, term.parameters = np.array([[0, 1]], dtype=np.int64), [("k", np.array([100.0])), ("r0", np.array([0.2]))]
system.tuple_terms = [term]
program = mdir.compile(*args)
term.expression = "0"
assert system.tuple_terms[0].expression != "0"
copy = system.tuple_terms
copy[0].parameters = [("k", np.array([200.0])), ("r0", np.array([0.2]))]
np.testing.assert_array_equal(system.tuple_terms[0].parameters[0][1], [100.0])
assert not program.stale
system.tuple_terms = copy
assert program.stale
state_copy = state.cell
state_copy.diagonal = np.full(3, 9.0)
assert not np.array_equal(state.cell.diagonal, np.full(3, 9.0))
state.cell = state_copy
assert program.stale
expect(mdir.InputError, lambda: setattr(state, "positions", np.zeros((1, 3))))
expect(mdir.InputError, lambda: setattr(state, "positions", np.full((system.particle_count, 3), np.nan)))
expect(mdir.InputError, lambda: mdir.load_amber("missing.prmtop", "missing.inpcrd"))
args = inputs("amber", mdir.Precision.Double)
args[4].precision = mdir.Precision.Single
expect(mdir.UnsupportedError, lambda: mdir.compile(*args))
args[4].precision = mdir.Precision.Double
expect(mdir.InputError, lambda: mdir.compile(None, *args[1:]))
# Errors leave the interpreter and compiler usable.
assert not mdir.compile(*args).stale
print("compilation parity, copied inputs, stale checks, typed errors and lifecycle passed")
